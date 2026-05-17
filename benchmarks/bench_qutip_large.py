"""QuTiP ``sesolve`` vs kryanneal の dt sweep ベンチ (issue #65 Phase 6 C4).

比較方針:

* **精度 (fidelity)**: ``dt → 0`` の極限を「収束した参照状態」と見做す.
  QuTiP は内部 ODE solver の最大 step 幅 ``max_step`` を ``dt`` にキャップ
  すれば実効的に dt sweep になる. 本ベンチでは sweep する dt のうち
  **最小 dt の QuTiP cell** を reference state とし, 全 cell (QuTiP / kryanneal
  m2 / trotter / cfm4) の fidelity を ``|⟨ψ_ref | ψ_cell⟩|^2`` で測る.

* **計算時間 (wall time)**: 各 (solver, dt) cell を 1 回だけ実行し
  ``time.perf_counter`` で測定する. fidelity 計算は state を保持しておけば
  事後計算で済むため, **状態取得と wall time 測定を同じ run で同時に行う**
  (issue #65 user 指示: 不要な計算反復を避ける).

* **QuTiP Hamiltonian は常に sparse 構築**: ``qutip.tensor`` of ``sigmax`` の
  和で CSR backing にする (``_build_qutip_hamiltonian``). 旧版は n=13 で
  dense / sparse 切替していたが, dense backing は dim^2 メモリ + dim^2 matvec
  という TFIM 構造に対して明らかに無駄 (non-zero は ``n · 2^n`` しかない).
  比較対象 QuTiP を不必要に遅くしないよう全 n で sparse 固定 (issue #65 review
  コメント). 旧 dense backing で n=12 が 18.6s だったセルは sparse で <1s
  レベル.

レポート: ``benchmarks/results/<YYYYMMDD-HHMMSS>/`` に以下を吐く.

* ``bench_qutip_large.csv``: per-cell raw data (n, solver, dt, n_steps,
  wall_sec, infidelity, log10_infidelity). reference cell は infidelity=0.
* ``bench_qutip_large.md``: machine info + per-n summary 表 (各 solver の
  fidelity-vs-dt と wall_time-vs-dt) + 経験的収束 order (連続 dt の
  infidelity 比から ``log2`` で見積もる).

machine baseline 規約 (``CLAUDE.md`` ベンチマーク節) に従い, BLAS thread / NumPy /
プラットフォームを machine info に記録する. 「○○× 速くなった」型の主張で
本ベンチを使う際は同一マシンで before/after を取ること.

依存: kryanneal (default features) + qutip (dev dep). QuTiP 未 install で
``ImportError``.

CLI 例::

    uv run python benchmarks/bench_qutip_large.py
    uv run python benchmarks/bench_qutip_large.py --n-values 10,12 --dt-values 0.1,0.05,0.025,0.01
    uv run python benchmarks/bench_qutip_large.py --solvers qutip,m2,cfm4 --blas-threads 1
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

# 比較対象 solver. ``qutip`` は QuTiP sesolve, それ以外は kryanneal の固定 dt
# 経路. adaptive Richardson は dt 指定が無いため本 dt-sweep 構造に乗らない
# (PI 制御の dt は動的決定). 固定 dt 経路の m2 / trotter / cfm4 で十分.
_VALID_SOLVERS: tuple[str, ...] = ("qutip", "m2", "trotter", "cfm4")


# ---------------------------------------------------------------------------
# Hamiltonian builder (常に sparse; フェアな比較のための判断)
# ---------------------------------------------------------------------------


def _build_qutip_hamiltonian(h_x: np.ndarray, h_p_diag: np.ndarray, T: float) -> list:
    """QuTiP ``sesolve`` 用 ``[[H_drv, A(t)], [H_p, B(t)]]`` を sparse で組む.

    ``H_drv = -Σ_i h_x[i] X_i`` を ``qutip.tensor`` (kron) ベースで CSR sparse
    構築. kryanneal の LSB bit 規約と QuTiP の MSB-first tensor 規約の差を
    吸収するため X は tensor list の位置 ``n-1-i`` に挿入する
    (``tests/test_reference_qutip.py`` と同じ規約変換).

    全 n で sparse 経路に固定する: dense backing にする理由が無い
    (TFIM の H_drv は non-zero が ``n · 2^n`` しかないので dim^2 dense は無駄,
    かつ比較対象 QuTiP の matvec を不必要に遅くすると bench がフェアでなくなる).
    旧版の ``_DENSE_THRESHOLD_N`` ベース dense/sparse 切替は issue #65 レビューで
    廃止 (dense backing の n=12 で 18.6s だったセルが sparse で <1s レベルに
    落ちることを確認済み).
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
# Cell runners (state + wall time を同時に取得; 1 cell = 1 run)
# ---------------------------------------------------------------------------


def _run_qutip_cell(
    h_t: list, psi0: np.ndarray, T: float, dt: float, n: int
) -> tuple[float, np.ndarray]:
    """QuTiP sesolve を ``max_step=dt`` で走らせて ``(wall_sec, psi_final)``.

    ``atol = 1e-12``, ``rtol = 1e-10`` で内部 ODE solver の局所誤差を絞り,
    ``max_step = dt`` で「最大 1 step 幅 = dt」を強制する. 実効的に dt が
    QuTiP の step 上限となり, dt → 0 で参照解に収束する想定.

    ``n`` (スピン数) を必須引数で受けるのは, psi0 を tensor product dims
    (``[[2]*n, [1]*n]``) で構築するため. dense / sparse 経路ともに
    Hamiltonian 側の dims は ``[[2]*n, [2]*n]`` に統一されており, psi0 もこの
    tensor product dims と整合させないと QuTiP solver が
    ``TypeError: incompatible dimensions`` を投げる.
    """
    psi0_q = qutip.Qobj(psi0.reshape(-1, 1), dims=[[2] * n, [1] * n])
    options = {
        "atol": 1e-12,
        "rtol": 1e-10,
        "max_step": float(dt),
        "nsteps": 1_000_000,  # max_step 拘束下でも全 [0,T] を 1 segment で抜けられる量
    }
    t_start = time.perf_counter()
    sol = qutip.sesolve(h_t, psi0_q, np.array([0.0, T]), options=options)
    elapsed = time.perf_counter() - t_start
    psi_final = sol.states[-1].full().ravel().astype(np.complex128)
    return elapsed, psi_final


def _run_kryanneal_cell(
    prob: IsingProblem,
    sched: Schedule,
    psi0: np.ndarray,
    T: float,
    method: str,
    n_steps: int,
) -> tuple[float, np.ndarray]:
    """kryanneal 固定 dt 経路を ``n_steps`` step で走らせて ``(wall_sec, psi_final)``.

    ``dt = T / n_steps``. method ∈ {m2, trotter, cfm4}.
    """
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
    psi_final = np.ascontiguousarray(res.psi_final)
    return elapsed, psi_final


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
    """1 (n, solver, dt) cell の生データ.

    fidelity は後段 (全 cell が出揃ってから reference を確定して) 計算する
    ので, この段階では ``psi_final`` と ``wall_sec`` だけ保持する.
    """

    __slots__ = ("n", "solver", "dt", "n_steps", "wall_sec", "psi_final")

    def __init__(
        self,
        n: int,
        solver: str,
        dt: float,
        n_steps: int,
        wall_sec: float,
        psi_final: np.ndarray,
    ) -> None:
        self.n = n
        self.solver = solver
        self.dt = dt
        self.n_steps = n_steps
        self.wall_sec = wall_sec
        self.psi_final = psi_final


def _fidelity(psi_a: np.ndarray, psi_b: np.ndarray) -> float:
    """``|⟨ψ_a|ψ_b⟩|^2`` (normalize 済み state 前提)."""
    return float(np.abs(np.vdot(psi_a, psi_b)) ** 2)


def _sweep_one_n(
    n: int,
    T: float,
    dt_list: list[float],
    solvers: list[str],
    seed: int,
) -> list[_CellRecord]:
    """1 つの n について全 (solver, dt) cell を実行し ``_CellRecord`` 列を返す.

    QuTiP Hamiltonian は dt sweep をまたいで再利用するため (sparse 構築は
    n=16 で数秒オーダ), n ループの先頭で 1 回だけ構築する. kryanneal 側も
    同じ ``IsingProblem`` / ``Schedule`` を全 dt cell で使い回す.
    """
    prob, sched, psi0, h_x, h_p_diag = _make_random_problem(n, T, seed)

    # Hamiltonian 構築. qutip cell を 1 つでも回すなら必要.
    h_t: list | None = None
    if "qutip" in solvers:
        h_t = _build_qutip_hamiltonian(h_x, h_p_diag, T)

    records: list[_CellRecord] = []
    for dt in dt_list:
        n_steps = max(1, int(round(T / dt)))
        # QuTiP cell.
        if "qutip" in solvers and h_t is not None:
            wall, psi = _run_qutip_cell(h_t, psi0, T, dt, n)
            records.append(_CellRecord(n, "qutip", dt, n_steps, wall, psi))
        # kryanneal cells.
        for method in ("m2", "trotter", "cfm4"):
            if method not in solvers:
                continue
            wall, psi = _run_kryanneal_cell(prob, sched, psi0, T, method, n_steps)
            records.append(_CellRecord(n, method, dt, n_steps, wall, psi))
    return records


def _pick_reference(records: list[_CellRecord]) -> _CellRecord:
    """全 cell から reference を選ぶ.

    優先順位: ``solver=="qutip"`` のうち最小 dt cell. QuTiP が含まれていない
    場合は **m2 の最小 dt cell** を fallback として返す (kryanneal 同士の
    self-consistency check).
    """
    qutip_cells = [r for r in records if r.solver == "qutip"]
    if qutip_cells:
        return min(qutip_cells, key=lambda r: r.dt)
    m2_cells = [r for r in records if r.solver == "m2"]
    if m2_cells:
        return min(m2_cells, key=lambda r: r.dt)
    cfm4_cells = [r for r in records if r.solver == "cfm4"]
    if cfm4_cells:
        return min(cfm4_cells, key=lambda r: r.dt)
    # どの solver も無いケースは caller 側でガードされている想定だが
    # 念のため明示的にエラーにする.
    raise RuntimeError("no records to derive reference from")


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
    }
    # _rust build flag (BLAS/rayon/simd) を露出.
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


def _write_csv(
    records: list[_CellRecord],
    ref: dict[int, _CellRecord],
    out_path: Path,
) -> None:
    """``bench_qutip_large.csv`` を書く.

    per-cell 1 行: (n, solver, dt, n_steps, wall_sec, fidelity, infidelity, log10_infidelity).
    ``infidelity = 1 - fidelity``, ``log10_infidelity = log10(infidelity)``
    (infidelity=0 で ``-inf`` 代わりに ``NaN`` 表記).
    """
    fieldnames = [
        "n",
        "solver",
        "dt",
        "n_steps",
        "wall_sec",
        "fidelity",
        "infidelity",
        "log10_infidelity",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            ref_cell = ref[r.n]
            if r is ref_cell:
                fid = 1.0
                infid = 0.0
            else:
                fid = _fidelity(ref_cell.psi_final, r.psi_final)
                infid = max(0.0, 1.0 - fid)
            log10_infid = f"{np.log10(infid):.6f}" if infid > 0.0 else "nan"
            writer.writerow(
                {
                    "n": r.n,
                    "solver": r.solver,
                    "dt": f"{r.dt:.6e}",
                    "n_steps": r.n_steps,
                    "wall_sec": f"{r.wall_sec:.6f}",
                    "fidelity": f"{fid:.12e}",
                    "infidelity": f"{infid:.6e}",
                    "log10_infidelity": log10_infid,
                }
            )


def _write_md(
    records: list[_CellRecord],
    ref: dict[int, _CellRecord],
    machine_info: dict[str, str],
    out_path: Path,
    args: argparse.Namespace,
) -> None:
    """``bench_qutip_large.md`` を書く (machine info + per-n table)."""
    lines: list[str] = []
    lines.append("# bench_qutip_large.py")
    lines.append("")
    lines.append(
        "QuTiP sesolve vs kryanneal の dt sweep ベンチ (issue #65 Phase 6 C4)."
    )
    lines.append("")
    lines.append(
        "fidelity の基準は **dt sweep 中の最小 dt の QuTiP cell** "
        "(QuTiP が無効なら m2 / cfm4 最小 dt cell に fallback). 各 (solver, dt) "
        "cell は 1 回だけ実行し state と wall time を同時に取る."
    )
    lines.append("")

    # Machine info.
    lines.append("## Machine info")
    lines.append("")
    for k, v in machine_info.items():
        lines.append(f"- **{k}**: `{v}`")
    lines.append("")
    lines.append(f"- **T**: `{args.T}`")
    lines.append(f"- **n_values**: `{args.n_values}`")
    lines.append(f"- **dt_values**: `{args.dt_values}`")
    lines.append(f"- **solvers**: `{args.solvers}`")
    lines.append("")

    # n ごとの table.
    n_values = sorted({r.n for r in records})
    for n in n_values:
        ref_cell = ref[n]
        lines.append(f"## n = {n} (reference: {ref_cell.solver}, dt={ref_cell.dt:.4g})")
        lines.append("")
        # 列: dt, 各 solver の wall_sec, infidelity (1-fid).
        solvers_in_n = sorted(
            {r.solver for r in records if r.n == n}, key=_solver_sort_key
        )
        header = ["dt", "n_steps"]
        for s in solvers_in_n:
            header.append(f"{s} wall (s)")
            header.append(f"{s} 1-fid")
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")

        # dt 列を昇順 (small first = 高精度側).
        dt_values_in_n = sorted({r.dt for r in records if r.n == n})
        for dt in dt_values_in_n:
            row_records = {r.solver: r for r in records if r.n == n and r.dt == dt}
            any_record = next(iter(row_records.values()))
            row = [f"{dt:.4g}", str(any_record.n_steps)]
            for s in solvers_in_n:
                if s not in row_records:
                    row.extend(["-", "-"])
                    continue
                cell = row_records[s]
                wall_str = f"{cell.wall_sec:.3f}"
                if cell is ref_cell:
                    infid_str = "0 (ref)"
                else:
                    fid = _fidelity(ref_cell.psi_final, cell.psi_final)
                    infid = max(0.0, 1.0 - fid)
                    infid_str = f"{infid:.3e}" if infid > 0.0 else "<1e-16"
                row.append(wall_str)
                row.append(infid_str)
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    out_path.write_text("\n".join(lines))


def _solver_sort_key(solver: str) -> int:
    """MD 表内の列順を ``qutip → m2 → trotter → cfm4`` で固定する."""
    order = {"qutip": 0, "m2": 1, "trotter": 2, "cfm4": 3}
    return order.get(solver, 100)


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
            "QuTiP sesolve vs kryanneal の dt sweep ベンチ "
            "(精度 + wall time を 1 pass で測定). issue #65 Phase 6 C4."
        )
    )
    parser.add_argument(
        "--n-values",
        type=_parse_int_list,
        default=[10, 12, 14],
        help="comma-separated sweep over spin counts (default: 10,12,14)",
    )
    parser.add_argument(
        "--dt-values",
        type=_parse_float_list,
        default=[0.1, 0.05, 0.025, 0.01, 0.005, 0.0025, 0.001],
        help=(
            "comma-separated sweep over time step sizes dt (default: "
            "0.1,0.05,0.025,0.01,0.005,0.0025,0.001). 最小 dt の QuTiP cell が "
            "fidelity の reference として使われる."
        ),
    )
    parser.add_argument(
        "--solvers",
        type=_parse_solver_list,
        default=list(_VALID_SOLVERS),
        help=(
            f"comma-separated solver list (choices: {','.join(_VALID_SOLVERS)}; "
            f"default: all). 'qutip' を外すと kryanneal 同士の self-consistency "
            "ベンチになる (reference = m2 最小 dt)."
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # warmup 警告抑制: matplotlib 未 install 警告 (qutip が出す) は手元 dev でも
    # 出るので noise になる. CI / プロダクションでも matplotlib は要らない.
    warnings.filterwarnings("ignore", category=UserWarning, module="qutip")

    if args.blas_threads is not None:
        set_blas_threads(args.blas_threads)

    # 出力先の確定.
    if args.results_dir is None:
        ts = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        results_dir = DEFAULT_RESULTS_ROOT / ts
    else:
        results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    machine_info = _build_machine_info(args)

    # n ごとに sweep を回し, record を集める.
    all_records: list[_CellRecord] = []
    ref_per_n: dict[int, _CellRecord] = {}
    for n in args.n_values:
        print(
            f"[bench_qutip_large] n={n}, dt sweep over "
            f"{args.dt_values} with solvers={args.solvers}",
            flush=True,
        )
        records = _sweep_one_n(
            n=n,
            T=args.T,
            dt_list=args.dt_values,
            solvers=args.solvers,
            seed=args.seed + n,
        )
        all_records.extend(records)
        ref_per_n[n] = _pick_reference(records)
        print(
            f"  done ({len(records)} cells); "
            f"reference = (solver={ref_per_n[n].solver}, dt={ref_per_n[n].dt:.4g})",
            flush=True,
        )

    csv_path = results_dir / "bench_qutip_large.csv"
    md_path = results_dir / "bench_qutip_large.md"
    _write_csv(all_records, ref_per_n, csv_path)
    _write_md(all_records, ref_per_n, machine_info, md_path, args)
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
