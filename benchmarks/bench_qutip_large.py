"""QuTiP ``sesolve`` vs kryanneal の work-precision diagram ベンチ (issue #65 Phase 6 C4).

比較指標を **「精度 (1 - fidelity) vs 計算時間 (wall sec)」** の 2 軸 Pareto
にした (issue #65 user review). 旧版は dt sweep で QuTiP/kryanneal を同じ dt
で比較していたが, QuTiP の adaptive solver では dt が必ずしも step を律しない
ため指標として不公平だった. 本版では:

* 各 solver は固有の **「精度つまみ」** を持ち, 独自 sweep する.
  - kryanneal m2 / trotter / cfm4 (固定 dt): ``n_steps``
  - kryanneal cfm4_adaptive_richardson: ``atol``
  - QuTiP: ``tol`` (内部で ``atol = rtol = tol`` として渡す.
    TFIM の量子状態は |ψ_k| ≤ 1/√dim 級なので atol/rtol の影響は同オーダ,
    1 軸 sweep で十分捕らえられる)

* **reference state** は QuTiP ``sesolve`` at ``tol=1e-13`` で 1 回だけ計算
  (n ごとに 1 回; ``--ref-tol`` で上書き可). adams ODE solver を極限まで
  絞った独立 ground truth として全 cell の fidelity 基準に使う. kryanneal
  自身を reference にしないことで「kryanneal vs QuTiP」の公平性を保つ.

* 各 sweep cell は **1 回だけ実行** し state と wall_sec を同時記録. fidelity
  は事後計算 (issue #65 user 指示: 不要な計算反復を避ける).

* QuTiP Hamiltonian は **常に sparse 構築** (``qutip.tensor`` of ``sigmax``).
  dense backing は TFIM 構造 (non-zero ~ n·2^n) に対して dim^2 メモリ +
  matvec の無駄, かつ比較対象を遅くする偏ったベンチになるため廃止.

レポート: ``benchmarks/results/<YYYYMMDD-HHMMSS>/`` に以下を吐く.

* ``bench_qutip_large.csv``: per-cell raw data (n, solver, knob_name,
  knob_value, n_steps_effective, wall_sec, infidelity, log10_infidelity,
  is_pareto).
* ``bench_qutip_large.md``: machine info + per-n work-precision 表 (infidelity
  昇順, Pareto-optimal cell に ``✓`` マーク). 「目標精度 X を最速で出す
  solver はどれか」が 1 表で読める形.

machine baseline 規約 (``CLAUDE.md`` ベンチマーク節) に従い, BLAS thread / NumPy /
プラットフォームを machine info に記録する.

依存: kryanneal (default features) + qutip (dev dep). QuTiP 未 install で
``ImportError``.

CLI 例::

    uv run python benchmarks/bench_qutip_large.py
    uv run python benchmarks/bench_qutip_large.py --n-values 10,12 --blas-threads 1
    # 各 solver のつまみ sweep を個別に絞る
    uv run python benchmarks/bench_qutip_large.py \\
        --cfm4-n-steps 5,10,20,50 \\
        --qutip-tols 1e-3,1e-6,1e-9
    # reference をさらに厳しく取る
    uv run python benchmarks/bench_qutip_large.py --ref-tol 1e-14
"""

from __future__ import annotations

import argparse
import csv
import importlib
import os
import platform
import sys
import time
import warnings
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from kryanneal import IsingProblem, QuantumAnnealer, Schedule, set_blas_threads
from kryanneal.initial_states import uniform_superposition

# QuTiP は dev dep のみ. 未 install で明示的に失敗させる (sesolve 無しでは
# 本ベンチの目的が成立しないため pytest 系 importorskip は使わない).
try:
    import qutip
except ImportError as exc:  # pragma: no cover - dev dep 必須
    raise ImportError(
        "benchmarks/bench_qutip_large.py requires `qutip` (dev dependency). "
        "Install with `uv sync --group dev`."
    ) from exc


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_ROOT = REPO_ROOT / "benchmarks" / "results"

# 比較対象 solver. 各 solver は固有の "精度つまみ" を持つ
# (``_SOLVER_KNOBS`` でマッピング). adaptive Richardson も sweep 対象に
# 含めることで「atol で精度を調整する経路」も Pareto 比較できる.
_VALID_SOLVERS: tuple[str, ...] = (
    "qutip",
    "m2",
    "trotter",
    "cfm4",
    "cfm4_adaptive_richardson",
)

# 各 solver の「精度つまみ名」. CSV / MD の knob_name 列および sweep CLI
# 引数の自動生成に使う.
_SOLVER_KNOBS: dict[str, str] = {
    "qutip": "tol",
    "m2": "n_steps",
    "trotter": "n_steps",
    "cfm4": "n_steps",
    "cfm4_adaptive_richardson": "atol",
}

# 各 solver の sweep 既定値. user は CLI で `--<solver>-<knob>s` を渡して
# 上書きできる (e.g. `--cfm4-n-steps 5,10,20`). kryanneal 固定 dt 経路の
# n_steps は global error order に応じて散布: m2/trotter (p=2) は大きい
# n_steps が必要, cfm4 (p=4) は少なくて済む. adaptive / qutip の tol も
# 1-fid ~ atol^2 の Taylor 展開で 1e-3 ~ 1e-12 の 5 桁 sweep にすれば
# 精度 1e-6 ~ 1e-24 オーダの Pareto 範囲を覆える.
_DEFAULT_M2_N_STEPS: list[int] = [10, 30, 100, 300, 1000, 3000]
_DEFAULT_TROTTER_N_STEPS: list[int] = [10, 30, 100, 300, 1000, 3000]
_DEFAULT_CFM4_N_STEPS: list[int] = [5, 10, 20, 50, 100, 200]
_DEFAULT_ADAPTIVE_TOLS: list[float] = [1e-3, 1e-5, 1e-7, 1e-9, 1e-11]
_DEFAULT_QUTIP_TOLS: list[float] = [1e-3, 1e-5, 1e-7, 1e-9, 1e-12]

# Reference 計算の QuTiP tol. これより下では floating-point eps が支配的に
# なるので default 1e-13 が現実的な極限. user は ``--ref-tol`` で上書き可.
_DEFAULT_REF_TOL: float = 1e-13


# ---------------------------------------------------------------------------
# Hamiltonian builder (常に sparse; フェアな比較のための判断)
# ---------------------------------------------------------------------------


def _build_qutip_hamiltonian(h_x: np.ndarray, h_p_diag: np.ndarray, T: float) -> list:
    """QuTiP ``sesolve`` 用 ``[[H_drv, A(t)], [H_p, B(t)]]`` を sparse で組む.

    ``H_drv = -Σ_i h_x[i] X_i`` を ``qutip.tensor`` (kron) ベースで CSR sparse
    構築. kryanneal の LSB bit 規約と QuTiP の MSB-first tensor 規約の差を
    吸収するため X は tensor list の位置 ``n-1-i`` に挿入する
    (``tests/test_reference_qutip.py`` と同じ規約変換).

    全 n で sparse 経路に固定する (dense backing は TFIM 構造に対して dim^2
    メモリ + matvec の無駄, かつ比較対象を遅くするので不公平).
    """
    n = h_x.shape[0]
    sx = qutip.sigmax()
    si = qutip.qeye(2)
    h_drv: object | None = None
    for i in range(n):
        ops = [si] * n
        ops[n - 1 - i] = sx
        term = -float(h_x[i]) * qutip.tensor(ops)
        h_drv = term if h_drv is None else h_drv + term
    h_p = qutip.qdiags(h_p_diag, 0, dims=[[2] * n, [2] * n])
    return [
        [h_drv, f"(1 - t/{T})"],
        [h_p, f"(t/{T})"],
    ]


# ---------------------------------------------------------------------------
# Cell runners (1 cell = 1 run; state と wall_sec を同時取得)
# ---------------------------------------------------------------------------


def _run_qutip(
    h_t: list, psi0: np.ndarray, T: float, n: int, tol: float
) -> tuple[float, np.ndarray]:
    """QuTiP ``sesolve`` を ``atol = rtol = tol`` で走らせて ``(wall_sec, psi)``.

    精度つまみは ``tol`` 1 軸. max_step 制約は **付けない** (内部 step は
    atol/rtol で完全に決まる; これが本 bench の「QuTiP の精度つまみは atol
    である」というスタンス). dt sweep でなく tol sweep が QuTiP の adaptive
    solver には自然.

    ``n`` (スピン数) を必須引数で受けるのは psi0 を tensor product dims
    (``[[2]*n, [1]*n]``) で構築するため. Hamiltonian 側 dims (``[[2]*n, [2]*n]``)
    と整合させないと QuTiP solver が ``TypeError: incompatible dimensions``.
    """
    psi0_q = qutip.Qobj(psi0.reshape(-1, 1), dims=[[2] * n, [1] * n])
    options = {
        "atol": float(tol),
        "rtol": float(tol),
        "nsteps": 1_000_000,
    }
    t_start = time.perf_counter()
    sol = qutip.sesolve(h_t, psi0_q, np.array([0.0, T]), options=options)
    elapsed = time.perf_counter() - t_start
    psi_final = sol.states[-1].full().ravel().astype(np.complex128)
    return elapsed, psi_final


def _run_kryanneal_fixed_dt(
    prob: IsingProblem,
    sched: Schedule,
    psi0: np.ndarray,
    T: float,
    method: str,
    n_steps: int,
) -> tuple[float, np.ndarray]:
    """kryanneal 固定 dt 経路 (m2 / trotter / cfm4) を ``n_steps`` 回走らせる."""
    ann = QuantumAnnealer(prob, sched)
    t_start = time.perf_counter()
    res = ann.run(
        psi0,
        0.0,
        T,
        method=method,  # type: ignore[arg-type]
        n_steps=n_steps,
    )
    elapsed = time.perf_counter() - t_start
    return elapsed, np.ascontiguousarray(res.psi_final)


def _run_kryanneal_adaptive(
    prob: IsingProblem,
    sched: Schedule,
    psi0: np.ndarray,
    T: float,
    atol: float,
) -> tuple[float, np.ndarray, int]:
    """kryanneal ``cfm4_adaptive_richardson`` 経路を ``atol`` 指定で走らせる.

    PI controller が実際に踏んだ step 数を ``n_steps_actual`` として返す
    (kryanneal の ``QuantumResult.n_steps_actual``). 出力テーブルの
    ``n_steps_effective`` 列に表示する.
    """
    ann = QuantumAnnealer(prob, sched)
    t_start = time.perf_counter()
    res = ann.run(
        psi0,
        0.0,
        T,
        method="cfm4_adaptive_richardson",
        atol=atol,
    )
    elapsed = time.perf_counter() - t_start
    return elapsed, np.ascontiguousarray(res.psi_final), int(res.n_steps_actual)


# ---------------------------------------------------------------------------
# Sample problem
# ---------------------------------------------------------------------------


def _make_random_problem(
    n: int, T: float, seed: int
) -> tuple[IsingProblem, Schedule, np.ndarray, np.ndarray, np.ndarray]:
    """seed 固定の random Ising problem を作る.

    ``h_x ~ Uniform(0.5, 1.5)``, ``H_p_diag ~ Uniform(-1, 1)``, linear schedule,
    ``|+⟩^N`` 始状態. ``(problem, schedule, psi0, h_x, h_p_diag)`` を返す
    (QuTiP 側で h_x / h_p_diag を再利用するため明示的に返す).
    """
    rng = np.random.default_rng(seed)
    h_x = rng.uniform(0.5, 1.5, size=n).astype(np.float64)
    h_p_diag = rng.uniform(-1.0, 1.0, size=1 << n).astype(np.float64)
    prob = IsingProblem(n=n, H_p_diag=h_p_diag, h_x=h_x)
    sched = Schedule.linear(T=T)
    psi0 = uniform_superposition(n)
    return prob, sched, psi0, h_x, h_p_diag


# ---------------------------------------------------------------------------
# Per-cell record + sweep orchestration
# ---------------------------------------------------------------------------


class _CellRecord:
    """1 (n, solver, knob_value) cell の生データ.

    fidelity は後段 (reference 確定後) に計算するため, この段階では
    ``psi_final`` と ``wall_sec`` と ``n_steps_effective`` を保持する.
    """

    __slots__ = (
        "n",
        "solver",
        "knob_name",
        "knob_value",
        "n_steps_effective",
        "wall_sec",
        "psi_final",
    )

    def __init__(
        self,
        n: int,
        solver: str,
        knob_name: str,
        knob_value: float,
        n_steps_effective: int | None,
        wall_sec: float,
        psi_final: np.ndarray,
    ) -> None:
        self.n = n
        self.solver = solver
        self.knob_name = knob_name
        self.knob_value = knob_value
        self.n_steps_effective = n_steps_effective
        self.wall_sec = wall_sec
        self.psi_final = psi_final


def _fidelity(psi_a: np.ndarray, psi_b: np.ndarray) -> float:
    """``|⟨ψ_a|ψ_b⟩|^2`` (normalize 済み state 前提)."""
    return float(np.abs(np.vdot(psi_a, psi_b)) ** 2)


def _format_knob_value(knob_name: str, value: float) -> str:
    """sweep つまみ値の human-readable 文字列化.

    ``n_steps`` は整数のまま, それ以外 (``tol`` / ``atol``) は科学表記.
    """
    if knob_name == "n_steps":
        return str(int(value))
    return f"{value:.1e}"


def _sweep_one_n(
    n: int,
    T: float,
    seed: int,
    solvers: list[str],
    m2_n_steps: list[int],
    trotter_n_steps: list[int],
    cfm4_n_steps: list[int],
    adaptive_atols: list[float],
    qutip_tols: list[float],
    ref_tol: float,
) -> tuple[list[_CellRecord], _CellRecord]:
    """1 つの n について全 sweep cell と reference cell を計算する.

    Returns
    -------
    records
        sweep cell の ``_CellRecord`` 列 (reference を含まない).
    ref_record
        reference 用に走らせた QuTiP ``tol=ref_tol`` の cell. テーブルの
        基準点として MD 出力に明示する.
    """
    prob, sched, psi0, h_x, h_p_diag = _make_random_problem(n, T, seed)
    h_t = _build_qutip_hamiltonian(h_x, h_p_diag, T)

    # Reference を最初に計算 (sweep cell の fidelity 計算で参照する).
    print(
        f"[bench_qutip_large] n={n} reference: QuTiP tol={ref_tol:.1e} ...",
        flush=True,
    )
    ref_wall, ref_psi = _run_qutip(h_t, psi0, T, n, ref_tol)
    print(f"  reference wall = {ref_wall:.3f}s", flush=True)
    ref_record = _CellRecord(
        n=n,
        solver="qutip",
        knob_name="tol",
        knob_value=ref_tol,
        n_steps_effective=None,
        wall_sec=ref_wall,
        psi_final=ref_psi,
    )

    records: list[_CellRecord] = []

    if "qutip" in solvers:
        print(f"  qutip sweep tols={qutip_tols} ...", flush=True)
        for tol in qutip_tols:
            wall, psi = _run_qutip(h_t, psi0, T, n, tol)
            records.append(
                _CellRecord(
                    n=n,
                    solver="qutip",
                    knob_name="tol",
                    knob_value=tol,
                    n_steps_effective=None,
                    wall_sec=wall,
                    psi_final=psi,
                )
            )

    for method, sweep in (
        ("m2", m2_n_steps),
        ("trotter", trotter_n_steps),
        ("cfm4", cfm4_n_steps),
    ):
        if method not in solvers:
            continue
        print(f"  {method} sweep n_steps={sweep} ...", flush=True)
        for n_steps in sweep:
            wall, psi = _run_kryanneal_fixed_dt(prob, sched, psi0, T, method, n_steps)
            records.append(
                _CellRecord(
                    n=n,
                    solver=method,
                    knob_name="n_steps",
                    knob_value=float(n_steps),
                    n_steps_effective=int(n_steps),
                    wall_sec=wall,
                    psi_final=psi,
                )
            )

    if "cfm4_adaptive_richardson" in solvers:
        print(
            f"  cfm4_adaptive_richardson sweep atols={adaptive_atols} ...", flush=True
        )
        for atol in adaptive_atols:
            wall, psi, n_steps_actual = _run_kryanneal_adaptive(
                prob, sched, psi0, T, atol
            )
            records.append(
                _CellRecord(
                    n=n,
                    solver="cfm4_adaptive_richardson",
                    knob_name="atol",
                    knob_value=atol,
                    n_steps_effective=n_steps_actual,
                    wall_sec=wall,
                    psi_final=psi,
                )
            )

    return records, ref_record


# ---------------------------------------------------------------------------
# Pareto 検出
# ---------------------------------------------------------------------------


def _pareto_mask(infids: list[float], walls: list[float]) -> list[bool]:
    """点列 ``(infid_i, wall_i)`` から **Pareto 最適 mask** を返す.

    Pareto 最適: 「infidelity も wall_sec も自分以下で, かつ少なくとも一方が
    厳密に小さい」点が他に無いこと. 両方とも "lower is better".

    O(N^2) で計算するが, cell 数は 30-50 程度なので問題ない.
    """
    n = len(infids)
    mask = [True] * n
    for i in range(n):
        if not mask[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if (
                infids[j] <= infids[i]
                and walls[j] <= walls[i]
                and (infids[j] < infids[i] or walls[j] < walls[i])
            ):
                mask[i] = False
                break
    return mask


# ---------------------------------------------------------------------------
# Output (CSV + Markdown)
# ---------------------------------------------------------------------------


def _build_machine_info(args: argparse.Namespace) -> dict[str, str]:
    """マシン同定情報を辞書化 (CSV/MD への記録用)."""
    info: dict[str, str] = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "qutip_version": getattr(qutip, "__version__", "?"),
        "rayon_threads": os.environ.get("RAYON_NUM_THREADS", "<unset>"),
        "blas_threads_requested": (
            str(args.blas_threads) if args.blas_threads is not None else "<unset>"
        ),
        "ref_tol": f"{args.ref_tol:.1e}",
    }
    try:
        rust_mod = importlib.import_module("kryanneal._rust")
        info["has_blas"] = str(bool(getattr(rust_mod, "__has_blas__", False)))
        info["has_rayon"] = str(bool(getattr(rust_mod, "__has_rayon__", False)))
        info["has_simd"] = str(bool(getattr(rust_mod, "__has_simd__", False)))
    except ImportError:
        info["has_blas"] = "?"
        info["has_rayon"] = "?"
        info["has_simd"] = "?"
    return info


def _compute_infidelities(
    records: list[_CellRecord], ref_psi: np.ndarray
) -> list[float]:
    """各 record の infidelity = 1 - |⟨ref|ψ⟩|^2 を計算."""
    out: list[float] = []
    for r in records:
        fid = _fidelity(ref_psi, r.psi_final)
        out.append(max(0.0, 1.0 - fid))
    return out


def _write_csv(
    records_per_n: dict[int, list[_CellRecord]],
    refs_per_n: dict[int, _CellRecord],
    infids_per_n: dict[int, list[float]],
    pareto_per_n: dict[int, list[bool]],
    out_path: Path,
) -> None:
    """``bench_qutip_large.csv`` を書く.

    per-cell 1 行 + reference を別 row として記録. ``is_pareto`` 列で Pareto
    最適 mask, ``is_reference`` 列で reference cell を識別.
    """
    fieldnames = [
        "n",
        "solver",
        "knob_name",
        "knob_value",
        "n_steps_effective",
        "wall_sec",
        "infidelity",
        "log10_infidelity",
        "is_pareto",
        "is_reference",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for n in sorted(records_per_n):
            ref = refs_per_n[n]
            # reference 行 (sweep には含まれないが文脈用に出力)
            writer.writerow(
                {
                    "n": n,
                    "solver": ref.solver,
                    "knob_name": ref.knob_name,
                    "knob_value": f"{ref.knob_value:.6e}",
                    "n_steps_effective": (
                        "" if ref.n_steps_effective is None else ref.n_steps_effective
                    ),
                    "wall_sec": f"{ref.wall_sec:.6f}",
                    "infidelity": "0.0",
                    "log10_infidelity": "nan",
                    "is_pareto": "",
                    "is_reference": "1",
                }
            )
            for r, infid, pareto in zip(
                records_per_n[n], infids_per_n[n], pareto_per_n[n], strict=True
            ):
                log10_infid = f"{np.log10(infid):.6f}" if infid > 0.0 else "nan"
                writer.writerow(
                    {
                        "n": n,
                        "solver": r.solver,
                        "knob_name": r.knob_name,
                        "knob_value": f"{r.knob_value:.6e}",
                        "n_steps_effective": (
                            "" if r.n_steps_effective is None else r.n_steps_effective
                        ),
                        "wall_sec": f"{r.wall_sec:.6f}",
                        "infidelity": f"{infid:.6e}",
                        "log10_infidelity": log10_infid,
                        "is_pareto": "1" if pareto else "0",
                        "is_reference": "0",
                    }
                )


def _write_md(
    records_per_n: dict[int, list[_CellRecord]],
    refs_per_n: dict[int, _CellRecord],
    infids_per_n: dict[int, list[float]],
    pareto_per_n: dict[int, list[bool]],
    machine_info: dict[str, str],
    args: argparse.Namespace,
    out_path: Path,
) -> None:
    """``bench_qutip_large.md`` を書く: machine info + per-n work-precision 表.

    各 n の表は infidelity 昇順で並べ, Pareto 最適 cell に ``✓`` マークを付ける.
    "目標精度 X を最速で出すのはどの solver か" が表 1 つで読める.
    """
    lines: list[str] = []
    lines.append("# bench_qutip_large.py")
    lines.append("")
    lines.append(
        "Work-precision diagram ベンチ: QuTiP ``sesolve`` vs kryanneal 各 method "
        "(issue #65 Phase 6 C4)."
    )
    lines.append("")
    lines.append(
        "各 solver は固有の精度つまみを sweep し, 共通 reference "
        "(QuTiP ``tol=ref_tol``) に対する infidelity と wall time を計測. "
        "下の表は **infidelity 昇順** に並べ, "
        "**Pareto 最適 cell (他に infidelity も wall time も同等以下な点が無い)** に "
        "✓ マークを付ける."
    )
    lines.append("")

    lines.append("## Machine info")
    lines.append("")
    for k, v in machine_info.items():
        lines.append(f"- **{k}**: `{v}`")
    lines.append("")
    lines.append(f"- **T**: `{args.T}`")
    lines.append(f"- **n_values**: `{args.n_values}`")
    lines.append(f"- **solvers**: `{args.solvers}`")
    lines.append(f"- **m2 n_steps sweep**: `{args.m2_n_steps}`")
    lines.append(f"- **trotter n_steps sweep**: `{args.trotter_n_steps}`")
    lines.append(f"- **cfm4 n_steps sweep**: `{args.cfm4_n_steps}`")
    lines.append(f"- **cfm4_adaptive_richardson atol sweep**: `{args.adaptive_tols}`")
    lines.append(f"- **qutip tol sweep**: `{args.qutip_tols}`")
    lines.append(f"- **reference**: QuTiP `tol={args.ref_tol:.1e}`")
    lines.append("")

    for n in sorted(records_per_n):
        ref = refs_per_n[n]
        records = records_per_n[n]
        infids = infids_per_n[n]
        pareto = pareto_per_n[n]

        lines.append(
            f"## n = {n} (reference: QuTiP tol={ref.knob_value:.1e}, "
            f"wall={ref.wall_sec:.3f}s)"
        )
        lines.append("")
        lines.append("| Pareto | solver | knob | n_steps_eff | infidelity | wall (s) |")
        lines.append("|---|---|---|---|---|---|")

        # infidelity 昇順にソート (ties は wall 昇順で次に解決).
        order = sorted(
            range(len(records)),
            key=lambda i: (infids[i], records[i].wall_sec),
        )
        for i in order:
            r = records[i]
            infid = infids[i]
            pareto_mark = "✓" if pareto[i] else ""
            knob_str = f"{r.knob_name}={_format_knob_value(r.knob_name, r.knob_value)}"
            n_steps_str = (
                str(r.n_steps_effective) if r.n_steps_effective is not None else "-"
            )
            infid_str = f"{infid:.3e}" if infid > 0.0 else "<1e-16"
            lines.append(
                f"| {pareto_mark} | {r.solver} | {knob_str} | {n_steps_str} | "
                f"{infid_str} | {r.wall_sec:.4f} |"
            )
        lines.append("")

    out_path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_int_list(text: str) -> list[int]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return [int(p) for p in parts]


def _parse_float_list(text: str) -> list[float]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("expected at least one float")
    return [float(p) for p in parts]


def _parse_solver_list(text: str) -> list[str]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("expected at least one solver")
    for p in parts:
        if p not in _VALID_SOLVERS:
            raise argparse.ArgumentTypeError(
                f"solver must be one of {_VALID_SOLVERS!r}, got {p!r}"
            )
    return parts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI 引数を parse する."""
    parser = argparse.ArgumentParser(
        description=(
            "Work-precision diagram ベンチ: QuTiP sesolve vs kryanneal "
            "(各 solver の精度つまみを sweep して infidelity vs wall_sec を Pareto 比較). "
            "issue #65 Phase 6 C4."
        )
    )
    parser.add_argument(
        "--n-values",
        type=_parse_int_list,
        default=[10, 12, 14],
        help="comma-separated sweep over spin counts (default: 10,12,14)",
    )
    parser.add_argument(
        "--solvers",
        type=_parse_solver_list,
        default=list(_VALID_SOLVERS),
        help=(
            f"comma-separated solver list (choices: {','.join(_VALID_SOLVERS)}; "
            f"default: all)."
        ),
    )
    parser.add_argument(
        "--T",
        type=float,
        default=1.0,
        help="total annealing time (default: 1.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260517,
        help="random seed for problem generation (default: 20260517). "
        "n ごとに seed + n を使うため n 軸に対しても再現可能.",
    )
    parser.add_argument(
        "--blas-threads",
        type=int,
        default=None,
        help=(
            "kryanneal.set_blas_threads(N) で BLAS pool を統一. None で no-op. "
            "machine-independent baseline には `--blas-threads 1` 推奨."
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help=(f"results 出力先 (default: {DEFAULT_RESULTS_ROOT}/<YYYYMMDD-HHMMSS>/)"),
    )
    # 各 solver の精度つまみ sweep.
    parser.add_argument(
        "--m2-n-steps",
        type=_parse_int_list,
        default=list(_DEFAULT_M2_N_STEPS),
        help=f"m2 の n_steps sweep (default: {_DEFAULT_M2_N_STEPS})",
    )
    parser.add_argument(
        "--trotter-n-steps",
        type=_parse_int_list,
        default=list(_DEFAULT_TROTTER_N_STEPS),
        help=f"trotter の n_steps sweep (default: {_DEFAULT_TROTTER_N_STEPS})",
    )
    parser.add_argument(
        "--cfm4-n-steps",
        type=_parse_int_list,
        default=list(_DEFAULT_CFM4_N_STEPS),
        help=f"cfm4 の n_steps sweep (default: {_DEFAULT_CFM4_N_STEPS})",
    )
    parser.add_argument(
        "--adaptive-tols",
        type=_parse_float_list,
        default=list(_DEFAULT_ADAPTIVE_TOLS),
        help=(
            f"cfm4_adaptive_richardson の atol sweep "
            f"(default: {_DEFAULT_ADAPTIVE_TOLS})"
        ),
    )
    parser.add_argument(
        "--qutip-tols",
        type=_parse_float_list,
        default=list(_DEFAULT_QUTIP_TOLS),
        help=(
            f"QuTiP sesolve の tol sweep (内部で atol = rtol = tol として使う; "
            f"default: {_DEFAULT_QUTIP_TOLS})"
        ),
    )
    parser.add_argument(
        "--ref-tol",
        type=float,
        default=_DEFAULT_REF_TOL,
        help=(
            f"reference state 計算用 QuTiP tol (default: {_DEFAULT_REF_TOL:.1e}). "
            "これより下では floating-point eps が支配的になるため 1e-13 が現実的な極限."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    warnings.filterwarnings("ignore", category=UserWarning, module="qutip")

    if args.blas_threads is not None:
        set_blas_threads(args.blas_threads)

    if args.results_dir is None:
        ts = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        results_dir = DEFAULT_RESULTS_ROOT / ts
    else:
        results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    machine_info = _build_machine_info(args)

    records_per_n: dict[int, list[_CellRecord]] = {}
    refs_per_n: dict[int, _CellRecord] = {}
    infids_per_n: dict[int, list[float]] = {}
    pareto_per_n: dict[int, list[bool]] = {}

    for n in args.n_values:
        records, ref = _sweep_one_n(
            n=n,
            T=args.T,
            seed=args.seed + n,
            solvers=args.solvers,
            m2_n_steps=args.m2_n_steps,
            trotter_n_steps=args.trotter_n_steps,
            cfm4_n_steps=args.cfm4_n_steps,
            adaptive_atols=args.adaptive_tols,
            qutip_tols=args.qutip_tols,
            ref_tol=args.ref_tol,
        )
        records_per_n[n] = records
        refs_per_n[n] = ref
        infids = _compute_infidelities(records, ref.psi_final)
        infids_per_n[n] = infids
        pareto_per_n[n] = _pareto_mask(infids, [r.wall_sec for r in records])
        print(
            f"  done ({len(records)} sweep cells, "
            f"{sum(pareto_per_n[n])} Pareto-optimal)",
            flush=True,
        )

    csv_path = results_dir / "bench_qutip_large.csv"
    md_path = results_dir / "bench_qutip_large.md"
    _write_csv(records_per_n, refs_per_n, infids_per_n, pareto_per_n, csv_path)
    _write_md(
        records_per_n,
        refs_per_n,
        infids_per_n,
        pareto_per_n,
        machine_info,
        args,
        md_path,
    )
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
