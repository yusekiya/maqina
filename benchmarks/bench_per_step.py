"""per-step wall time benchmark (M2 / Trotter / Suzuki S_4 / CFM4:2 / adaptive Richardson).

``QuantumAnnealer.run`` の 1 step あたりの実時間を, ``method`` ×
``n`` (スピン数 / Hilbert 空間次元 ``2^n``) で sweep して計測する.
Phase 1 の DoD (issue #1) と Phase 2 の DoD (issue #18), Phase 3 の DoD
(issue #30) で「``bench_per_step.py`` で M2 / Trotter / Suzuki S_4 / CFM4:2
の per-step 数値が記録され ``benchmarks/results/`` 配下に置かれている
(クロスオーバ実測)」を満たすためのベンチエントリポイント.
Phase 4 C3 (issue #39) で ``method="cfm4_adaptive_richardson_krylov"`` を追加.

設計規約は ``docs/design/10-benchmarks.md`` §10, 実行手順は ``benchmarks/README.md``,
全体規約は ``CLAUDE.md`` 「ベンチマーク」節を参照する.

サポート ``method`` (Phase 4 末時点):

* ``"m2"``: M2 中点則 + Lanczos (Phase 1).
* ``"trotter"``: Strang 2 次 Trotter (Phase 2 C3).
* ``"trotter_suzuki4"``: Suzuki S_4 4 次 Trotter (Phase 2 C4).
* ``"cfm4"``: CFM4:2 commutator-free Magnus + Lanczos (Phase 3 C2).
* ``"cfm4_adaptive_richardson_krylov"``: CFM4:2 + step-doubling Richardson +
  PI controller (Phase 4 C3). 固定 dt 経路と違い ``--n-steps`` は
  「初期 dt 提案 (``dt_init = T / n_steps``)」と「per-step 列の見せ方
  (実 step は driver が決める)」の補助パラメータ扱い.

``method`` ごとに per-step コストの内訳が異なる:

* M2: per-step ``m·dim`` flops (Lanczos m matvec).
* Trotter: per-step ``(N+1)·dim`` flops (phase 1 + bit-flip N).
* Suzuki S_4: per-step ``5·(N+1)·dim`` flops (Strang 5 回).
* CFM4:2: per-step ``2m·dim`` flops (Lanczos 2 回, M2 の 2 倍重い).
* CFM4 adaptive Richardson: per-step ``6m·dim`` flops (full ``2m`` +
  half×2 ``4m`` = ``6m``, CFM4:2 fixed の 3×). 実 step 数は driver が
  PI 制御で決めるので, raw wall time に加えて ``n_steps_actual`` と
  ``final_err_vs_ref`` (高精度 fixed-cfm4 を参照とした state 差) を
  CSV / md に併記する.

LTE order も異なる (M2 / Strang は ``O(dt^3)``, Suzuki S_4 / CFM4:2 /
Richardson は ``O(dt^5)``, Richardson w/ extrapolate は実効 6 次) ので,
同じ ``n_steps`` での生 wall time 比較に加えて, 同じ精度を要求したときの
required ``n_steps`` のずれを別途見積もる必要がある.

出力:

* ``benchmarks/results/<YYYYMMDD-HHMMSS>/bench_per_step.csv``: per-trial
  生データ (n, dim, method, trial, n_steps, dt, m, total_wall_sec,
  per_step_sec, states_per_sec, n_steps_actual, final_err_vs_ref,
  m_eff_median, m_eff_max). 末尾 m_eff_* 列は adaptive 経路でのみ
  実値が入る (issue #52 B).
* ``benchmarks/results/<YYYYMMDD-HHMMSS>/bench_per_step.md``: 集計表
  (per-method summary + cross-method 比較表 + adaptive driver detail) +
  machine info. adaptive 経路 (``cfm4_adaptive_richardson_krylov`` 等) を含む
  実行では ``## Adaptive driver detail`` 節が追加され, PI controller が
  accept した実 step 数 ``n_steps_actual`` と高精度参照解との差
  ``final_err_vs_ref`` (adaptive driver の性能・精度評価に最も重要な
  2 値) が n × method 単位で記録される. reference 計算 (高精度 fixed
  CFM4:2) の wall time も machine info の ``reference_wall_sec_total``
  と adaptive section の per-n ``reference_wall_sec`` 列に記録され,
  大 n で reference が 1-2 時間かかる現実問題に対する透明性を担保する.

CLI 例::

    uv run python benchmarks/bench_per_step.py --n-values 4,8,12 --n-steps 50
    uv run python benchmarks/bench_per_step.py --methods m2,trotter
    # adaptive Richardson の smoke
    uv run python benchmarks/bench_per_step.py --methods cfm4_adaptive_richardson_krylov --n-values 4
    # BLAS thread を 1 に固定して machine-independent baseline を取る
    uv run python benchmarks/bench_per_step.py --blas-threads 1

ベンチは原則 ``--release`` build (``maturin develop --uv --release``) で
取る. debug build (``maturin develop --uv`` のみ) の値はベースラインに
ふさわしくないため,本スクリプトは実行時に ``_rust.__has_blas__`` を含む
build フラグを記録するに留め, build profile 自体は呼び出し側の責任とする.

BLAS thread 数の固定 (``--blas-threads N``) は Linux + numpy bundled
OpenBLAS 環境で特に重要. default では numpy bundled OpenBLAS が物理
コア数までスレッドを張り, dim が小さい (n=4..12) cell で thread-launch
overhead が支配して per-step 値がノイジーになる. machine-independent
な scalar single-thread baseline を取りたい場合は ``--blas-threads 1``
を必ず付ける. macOS Apple Accelerate は default で挙動が異なる
(自動 tuning) ので, 機種間比較を主張するときも明示的に thread 数を
合わせる.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import os
import platform
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from maqina import IsingProblem, QuantumAnnealer, Schedule, set_blas_threads
from maqina.initial_states import uniform_superposition

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_ROOT = REPO_ROOT / "benchmarks" / "results"

# サポートする method 一覧. ``QuantumAnnealer.run`` の Literal 型ヒントと
# 揃える. Phase 4 C3 で ``cfm4_adaptive_richardson_krylov`` 追加.
_VALID_METHODS: tuple[str, ...] = (
    "m2",
    "trotter",
    "trotter_suzuki4",
    "cfm4",
    "cfm4_adaptive_richardson_krylov",
)

# adaptive 経路の集合. n_steps を driver に渡さず, ``atol`` / ``dt_init``
# を渡す経路を判別するための小ヘルパ.
_ADAPTIVE_METHODS: frozenset[str] = frozenset({"cfm4_adaptive_richardson_krylov"})

# adaptive 経路で final state を比較する参照解の生成パラメータ.
# 同じ ``T`` / ``schedule`` で fixed CFM4:2 を多 step 走らせた終端 ψ を
# 「真値の代用」とする (Phase 4 C3 の DoD では QuTiP との end-to-end
# fidelity は ``tests/test_adaptive.py`` 側で別途検証されているため,
# ベンチでは「ベンチ自身が再現する高精度経路」と比較すれば十分).
_REFERENCE_N_STEPS: int = 2000
_REFERENCE_METHOD: str = "cfm4"


def _parse_int_list(text: str) -> list[int]:
    """``"4,8,12"`` のような CSV 文字列を ``[4, 8, 12]`` にする."""
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return [int(p) for p in parts]


def _parse_method_list(text: str) -> list[str]:
    """``"m2,trotter"`` のような CSV 文字列を `_VALID_METHODS` で検証してリスト化."""
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("expected at least one method")
    for p in parts:
        if p not in _VALID_METHODS:
            raise argparse.ArgumentTypeError(
                f"method must be one of {_VALID_METHODS!r}, got {p!r}"
            )
    return parts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI 引数を parse する.

    sweep 対象の ``n`` 列は ``--n-values`` で受け取り, 計測安定性のための
    ``--repeat`` / ``--warmup`` は分けて指定可能にする (warmup は cache
    warm のために常に必要だが結果には残さない設計).
    """
    parser = argparse.ArgumentParser(
        description=(
            "maqina per-step wall time benchmark "
            "(M2 / Trotter / Suzuki S_4 / CFM4:2; "
            "see docs/design/10-benchmarks.md §10)"
        )
    )
    parser.add_argument(
        "--n-values",
        type=_parse_int_list,
        default=[4, 8, 12, 16],
        help="comma-separated sweep over spin counts (default: 4,8,12,16)",
    )
    parser.add_argument(
        "--methods",
        type=_parse_method_list,
        default=list(_VALID_METHODS),
        help=(
            f"comma-separated propagator methods to benchmark "
            f"(choices: {','.join(_VALID_METHODS)}; "
            f"default: {','.join(_VALID_METHODS)})"
        ),
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=50,
        help="number of driver steps per measurement (default: 50)",
    )
    parser.add_argument(
        "--m-values",
        type=_parse_int_list,
        default=[24],
        help=(
            "comma-separated sweep over Lanczos subspace dimension "
            "(default: 24). Trotter methods ignore m; m2 / cfm4 / "
            "cfm4_adaptive_richardson_krylov use it. issue #52 B: 列形式により "
            "m=16,24,32 等の cell 比較が 1 run で取れる."
        ),
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="number of timed trials per (n) cell (default: 3)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="number of warmup trials per (n) cell (default: 1)",
    )
    parser.add_argument(
        "--T",
        type=float,
        default=1.0,
        help="total anneal time T (default: 1.0)",
    )
    parser.add_argument(
        "--blas-threads",
        type=int,
        default=None,
        help=(
            "if specified, call maqina.set_blas_threads(N) at startup to "
            "lock all loaded BLAS pools (numpy bundled + system OpenBLAS that "
            "the Rust extension links against) to N threads. Use "
            "`--blas-threads 1` to take a machine-independent single-thread "
            "baseline; on Linux + numpy-bundled OpenBLAS the default is the "
            "physical core count, which makes per-step measurements at small "
            "dim noisy due to thread-launch overhead. Default: None "
            "(leave BLAS thread counts untouched)."
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help=(
            "output directory; if omitted, "
            "benchmarks/results/<YYYYMMDD-HHMMSS>/ is auto-created"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260512,
        help="numpy RNG seed for random H_p_diag / h_x (default: 20260512)",
    )
    return parser.parse_args(argv)


def make_random_problem(n: int, seed: int) -> IsingProblem:
    """ランダム ``H_p_diag`` (実数 ``[-1, 1]``) と一様 ``h_x = 1`` で
    ``IsingProblem`` を作る.

    アルゴリズム測定としては係数の具体値より dim と Lanczos 部分空間
    次元が支配するので, ベンチは再現可能な乱数 seed で済ませる.
    """
    rng = np.random.default_rng(seed)
    dim = 1 << n
    h_p = rng.uniform(-1.0, 1.0, size=dim).astype(np.float64)
    h_x = np.ones(n, dtype=np.float64)
    return IsingProblem(n=n, H_p_diag=h_p, h_x=h_x)


def time_method_run(
    problem: IsingProblem,
    schedule: Schedule,
    psi0: np.ndarray,
    t0: float,
    t1: float,
    n_steps: int,
    method: str,
    m: int,
) -> tuple[float, int, np.ndarray, dict[str, int | float] | None]:
    """``QuantumAnnealer.run`` の wall time (秒) を ``time.perf_counter`` で計る.

    壁時計のジッタを最小化するため, 1 試行で run 1 回ぶん計測する.
    ``method`` ごとに ``QuantumAnnealer.run`` への呼び方が変わる:

    * 固定 dt 経路 (``"m2"`` / ``"trotter"`` / ``"trotter_suzuki4"`` /
      ``"cfm4"``): ``n_steps`` をそのまま渡す.
    * adaptive 経路 (``"cfm4_adaptive_richardson_krylov"``): ``atol = 1e-8``
      (driver 既定値と同) + ``dt_init = (t1 - t0) / n_steps`` を渡し,
      driver が実 step 数を決める.

    Trotter 系は Lanczos を呼ばないため ``m`` は無視されるが, デフォルト値で
    渡しても害は無いので統一的に渡す. ``"m2"`` / ``"cfm4"`` /
    ``"cfm4_adaptive_richardson_krylov"`` 経路は ``m`` がそのまま Lanczos 部分空間
    次元として効く.

    Returns
    -------
    wall : float
        測定 wall time (秒).
    n_steps_actual : int
        固定 dt 経路では ``n_steps`` をそのまま, adaptive 経路では driver
        が決めた実 step 数を返す.
    psi_final : np.ndarray
        終端波動関数 (final_err_vs_ref の算出に使う). complex128.
    m_eff_stats : dict[str, int | float] | None
        adaptive Richardson 経路では ``QuantumResult.m_eff_stats`` を
        そのまま返す (issue #52 A). 固定 dt 経路では ``None``.
    """
    if method not in _VALID_METHODS:
        raise ValueError(f"unsupported method {method!r}; valid: {_VALID_METHODS!r}")
    ann = QuantumAnnealer(problem, schedule, m=m)
    if method in _ADAPTIVE_METHODS:
        dt_init = (float(t1) - float(t0)) / int(n_steps)
        t_start = time.perf_counter()
        res = ann.run(
            psi0,
            t0,
            t1,
            method=method,  # type: ignore[arg-type]
            atol=1e-8,
            dt_init=dt_init,
        )
        t_end = time.perf_counter()
        n_steps_eff = int(res.n_steps_actual) if res.n_steps_actual is not None else 0
    else:
        t_start = time.perf_counter()
        res = ann.run(psi0, t0, t1, method=method, n_steps=n_steps)  # type: ignore[arg-type]
        t_end = time.perf_counter()
        n_steps_eff = int(n_steps)
    # res を黒箱に積んでおいて dead code elimination されないようにする.
    _ = res.n_matvec
    return t_end - t_start, n_steps_eff, np.asarray(res.psi_final), res.m_eff_stats


def _compute_reference_psi(
    problem: IsingProblem,
    schedule: Schedule,
    psi0: np.ndarray,
    t0: float,
    t1: float,
    m: int,
) -> tuple[np.ndarray, float]:
    """adaptive 経路の ``final_err_vs_ref`` 算出のための参照解 ψ を取る.

    fixed CFM4:2 を ``_REFERENCE_N_STEPS`` 多 step で走らせた終端 ψ を
    「真値の代用」とする. QuTiP との fidelity 検証は
    ``tests/test_adaptive.py`` で別途行われており, ベンチではコスト次元
    比較が主目的なので追加依存を避ける.

    Returns
    -------
    psi_ref : np.ndarray
        高精度参照解の終端波動関数 (complex128).
    wall_sec : float
        参照解の計算自体に要した wall time (秒). 大 n で 1-2 時間かかる
        場合があるため bench 出力に透明性として記録する.
    """
    ann_ref = QuantumAnnealer(problem, schedule, m=m)
    t_start = time.perf_counter()
    res = ann_ref.run(
        psi0,
        t0,
        t1,
        method=_REFERENCE_METHOD,  # type: ignore[arg-type]
        n_steps=_REFERENCE_N_STEPS,
    )
    t_end = time.perf_counter()
    return np.asarray(res.psi_final), t_end - t_start


def collect_machine_info() -> dict[str, str]:
    """マシン特性と numpy / BLAS pool 情報を文字列辞書で返す.

    `bench_per_step.md` の machine info 節にそのまま書き出す.
    `threadpool_info()` は numpy の BLAS pool を露出するため,
    Apple Accelerate / OpenBLAS / MKL のどれが動いているかを記録する.
    """
    info: dict[str, str] = {
        "timestamp_utc": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "python": sys.version.split()[0],
        "numpy": np.__version__,
    }

    # Rust 拡張の build フラグ.
    try:
        rust_mod = importlib.import_module("maqina._rust")
        info["rust_extension"] = "loaded"
        info["__has_blas__"] = str(bool(getattr(rust_mod, "__has_blas__", False)))
    except ImportError:
        info["rust_extension"] = "missing (Python fallback path)"
        info["__has_blas__"] = "n/a"

    # BLAS pool (numpy / scipy / system).
    try:
        from threadpoolctl import threadpool_info

        pools = threadpool_info()
        blas = [p for p in pools if p.get("user_api") == "blas"]
        if blas:
            info["blas_pools"] = "; ".join(
                f"{p.get('internal_api', '?')}/{p.get('prefix', '?')}"
                f" threads={p.get('num_threads', '?')}"
                for p in blas
            )
        else:
            info["blas_pools"] = "no BLAS pool detected"
    except ImportError:
        info["blas_pools"] = "threadpoolctl unavailable"

    info["cpu_count"] = str(os.cpu_count() or 0)
    return info


def write_outputs(
    out_dir: Path,
    rows: list[dict[str, float | int | str]],
    summary: list[dict[str, float | int | str]],
    adaptive_summary: list[dict[str, float | int | str]],
    machine_info: dict[str, str],
    args: argparse.Namespace,
    reference_wall_by_n: dict[int, float],
) -> None:
    """CSV (生データ) と markdown (集計 + machine info) を ``out_dir`` に書く.

    markdown は **per-method summary** と **method ごとの median 比較表**
    の 2 種類を出す. 比較表は M2 を基準にした ratio を併記し, 同じ
    ``n_steps`` での raw per-step コスト比較を一目で見られるようにする
    (LTE order の違いから「精度を揃えた場合の wall time 比較」は別途
    必要; 詳細は冒頭 docstring の Notes 参照).

    さらに adaptive 経路 (``_ADAPTIVE_METHODS``) を含む実行では
    ``## Adaptive driver detail`` 節を追加し, PI controller が accept した
    実 step 数 ``n_steps_actual`` (median, min/max) と高精度参照解との
    state 差 ``final_err_vs_ref`` (median) を per-n × method で出す.
    これらは adaptive driver の性能・精度評価で最重要 2 値だが,
    Summary 表の per-step / states/sec だけでは見えないため別節で並べる
    (cv_ising の bench pattern とは異なる, maqina 固有の adaptive 評価軸).
    ``reference_wall_sec`` 列は ``_compute_reference_psi`` 自体の wall time
    を per-n で記録する (大 n で reference が 1-2 時間かかる現実問題への
    透明性).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "bench_per_step.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_path = out_dir / "bench_per_step.md"
    lines: list[str] = []
    lines.append("# bench_per_step results (M2 / Trotter / Suzuki S_4 / CFM4:2)")
    lines.append("")
    lines.append("## Machine info")
    lines.append("")
    for key, val in machine_info.items():
        lines.append(f"- **{key}**: {val}")
    lines.append("")
    lines.append("## CLI arguments")
    lines.append("")
    for key, val in vars(args).items():
        lines.append(f"- **{key}**: {val}")
    lines.append("")
    lines.append("## Summary (per-n × method × m)")
    lines.append("")
    lines.append(
        "| n | dim | method | m | per-step (sec) min | per-step (sec) median | "
        "states/sec (median) | trials |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for row in summary:
        lines.append(
            f"| {row['n']} | {row['dim']} | {row['method']} | {row['m']} | "
            f"{row['per_step_sec_min']:.6e} | "
            f"{row['per_step_sec_median']:.6e} | "
            f"{row['states_per_sec_median']:.3e} | "
            f"{row['trials']} |"
        )
    lines.append("")

    # Cross-method 比較: 各 n で method ごとの median per-step を横並びに
    # 配置し, M2 を基準とした ratio (m2/method) を併記する.
    # issue #52 B: m sweep がある場合は cell が重複するため cross-method 表は
    # 出さない (代わりに Summary 表で m 列付き).
    methods_in_summary = sorted({str(s["method"]) for s in summary})
    distinct_m_values = sorted({int(s["m"]) for s in summary})
    if len(methods_in_summary) > 1 and len(distinct_m_values) == 1:
        lines.append("## Cross-method per-step median (sec)")
        lines.append("")
        header_cells = ["n", "dim", *methods_in_summary]
        # M2 を基準とした ratio (m2 / x) 列を, m2 以外の method について並べる.
        non_m2_methods = [m for m in methods_in_summary if m != "m2"]
        has_m2 = "m2" in methods_in_summary
        if has_m2:
            header_cells.extend(f"m2 / {m}" for m in non_m2_methods)
        lines.append("| " + " | ".join(header_cells) + " |")
        lines.append("|" + "|".join(["---"] * len(header_cells)) + "|")

        by_n_method: dict[int, dict[str, float]] = {}
        for s in summary:
            by_n_method.setdefault(int(s["n"]), {})[str(s["method"])] = float(
                s["per_step_sec_median"]
            )
        for n in sorted(by_n_method.keys()):
            dim = 1 << n
            row_cells: list[str] = [str(n), str(dim)]
            for method in methods_in_summary:
                val = by_n_method[n].get(method)
                row_cells.append(f"{val:.6e}" if val is not None else "n/a")
            if has_m2:
                m2_val = by_n_method[n].get("m2")
                for method in non_m2_methods:
                    other = by_n_method[n].get(method)
                    if m2_val is not None and other is not None and other > 0:
                        row_cells.append(f"{m2_val / other:.3f}")
                    else:
                        row_cells.append("n/a")
            lines.append("| " + " | ".join(row_cells) + " |")
        lines.append("")

    # adaptive 経路がある場合のみ Adaptive driver detail 節を出す.
    # Summary / Cross-method 表は per-step 値しか見ないため,
    # n_steps_actual (PI controller が accept した実 step 数) と
    # final_err_vs_ref (高精度 reference との state 差) が落ちる.
    # adaptive driver の評価軸として最重要なので別節で並べる.
    if adaptive_summary:
        lines.append("## Adaptive driver detail")
        lines.append("")
        # issue #52 B: m_eff_median / m_eff_max 列を追加. m sweep の cell
        # ごとに per-step Lanczos 部分空間の実使用量を見られるようにする.
        lines.append(
            "| n | dim | method | m | n_steps_actual (median) | "
            "n_steps_actual (min/max) | final_err_vs_ref (median) | "
            "m_eff (median) | m_eff (max) | "
            "per_step (sec, median) | total_wall (sec, median) | "
            "reference_wall (sec) |"
        )
        lines.append("|" + "|".join(["---"] * 12) + "|")
        for row in adaptive_summary:
            n_val = int(row["n"])
            ref_wall = reference_wall_by_n.get(n_val)
            ref_wall_str = f"{ref_wall:.3e}" if ref_wall is not None else "n/a"
            lines.append(
                f"| {row['n']} | {row['dim']} | {row['method']} | {row['m']} | "
                f"{float(row['n_steps_actual_median']):.1f} | "
                f"{int(row['n_steps_actual_min'])}/{int(row['n_steps_actual_max'])} | "
                f"{float(row['final_err_vs_ref_median']):.6e} | "
                f"{float(row['m_eff_median']):.2f} | "
                f"{int(row['m_eff_max'])} | "
                f"{float(row['per_step_sec_median']):.6e} | "
                f"{float(row['total_wall_sec_median']):.6e} | "
                f"{ref_wall_str} |"
            )
        lines.append("")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # ``--results-dir`` が repo 外 (例: pytest の tmp_path) を指している
    # 場合に備え, ``relative_to`` の ``walk_up=True`` を使う (Python 3.12+).
    # 失敗時は absolute path を素直に出す.
    for p in (csv_path, md_path):
        try:
            shown = p.relative_to(REPO_ROOT, walk_up=True)
        except ValueError:
            shown = p
        print(f"wrote {shown}")


def main(argv: list[str] | None = None) -> int:
    """エントリポイント. 各 ``n`` について warmup → 計測を回し, CSV + md を書く."""
    args = parse_args(argv)

    # ``--blas-threads`` 指定時は BLAS pool スレッド数を統一する.
    # numpy bundled / scipy bundled / system OpenBLAS の最大 3 pool が同居
    # しうるが, threadpoolctl 経由で全 BLAS pool を一括設定するため,
    # `set_blas_threads` が ``threadpool_info`` 後の load も含めて拾う.
    # `OPENBLAS_NUM_THREADS` 等の env var は pool 初期化時にしか効かないが,
    # こちらは load 済 pool にも反映できるため信頼性が高い.
    # default ``None`` 時は何もしない (Phase 1 baseline と同じ挙動).
    if args.blas_threads is not None:
        set_blas_threads(args.blas_threads)

    out_dir = args.results_dir
    if out_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_dir = DEFAULT_RESULTS_ROOT / timestamp

    machine_info = collect_machine_info()

    rows: list[dict[str, float | int | str]] = []
    summary: list[dict[str, float | int | str]] = []
    adaptive_summary: list[dict[str, float | int | str]] = []
    reference_wall_by_n: dict[int, float] = {}
    schedule = Schedule.linear(T=args.T)

    for n in args.n_values:
        problem = make_random_problem(n, seed=args.seed)
        psi0 = uniform_superposition(n)
        dim = 1 << n
        print(
            f"[n={n} dim={dim}] methods={args.methods}, m_values={args.m_values}, "
            f"warmup={args.warmup}, repeat={args.repeat}"
        )

        # adaptive 経路がある場合のみ高精度参照解を 1 度だけ計算する.
        # reference 計算自体は m に依存しない (β_k tol で早期打切が起きる
        # 設定でも `_REFERENCE_METHOD="cfm4"` で多 step なため m=24 fixed
        # で十分な精度が出る前提). issue #52 B では m sweep を入れたが
        # reference は per-n 1 回固定で OK.
        needs_reference = any(m in _ADAPTIVE_METHODS for m in args.methods)
        psi_ref: np.ndarray | None = None
        if needs_reference:
            print(
                f"  computing reference ψ ({_REFERENCE_METHOD},"
                f" n_steps={_REFERENCE_N_STEPS}, m={args.m_values[0]})..."
            )
            psi_ref, ref_wall = _compute_reference_psi(
                problem, schedule, psi0, 0.0, args.T, args.m_values[0]
            )
            reference_wall_by_n[n] = ref_wall
            print(f"  reference ψ wall: {ref_wall:.3f}s")

        # issue #52 B: m を sweep する外側ループ. 同じ method × m の cell
        # ごとに summary / adaptive_summary を 1 行ずつ追加する.
        for m_val in args.m_values:
            for method in args.methods:
                is_adaptive = method in _ADAPTIVE_METHODS
                print(f"  method={method}, m={m_val}")
                for _ in range(args.warmup):
                    time_method_run(
                        problem,
                        schedule,
                        psi0,
                        0.0,
                        args.T,
                        args.n_steps,
                        method,
                        m_val,
                    )

                trial_times: list[float] = []
                trial_n_steps_actual: list[int] = []
                trial_final_err: list[float] = []
                trial_m_eff_stats: list[dict[str, int | float]] = []
                for trial in range(args.repeat):
                    wall, n_steps_actual, psi_final, m_eff_stats = time_method_run(
                        problem,
                        schedule,
                        psi0,
                        0.0,
                        args.T,
                        args.n_steps,
                        method,
                        m_val,
                    )
                    trial_times.append(wall)
                    trial_n_steps_actual.append(n_steps_actual)
                    if m_eff_stats is not None:
                        trial_m_eff_stats.append(m_eff_stats)
                    steps_for_per_step = max(n_steps_actual, 1)
                    per_step = wall / steps_for_per_step
                    states_per_sec = dim / per_step if per_step > 0 else float("inf")
                    if is_adaptive and psi_ref is not None:
                        final_err = float(np.linalg.norm(psi_final - psi_ref))
                        trial_final_err.append(final_err)
                        final_err_field = f"{final_err:.6e}"
                    else:
                        final_err_field = "n/a"
                    m_eff_median_field = (
                        f"{m_eff_stats['median']:.2f}"
                        if m_eff_stats is not None
                        else "n/a"
                    )
                    m_eff_max_field = (
                        f"{int(m_eff_stats['max'])}"
                        if m_eff_stats is not None
                        else "n/a"
                    )
                    rows.append(
                        {
                            "n": n,
                            "dim": dim,
                            "method": method,
                            "trial": trial,
                            "n_steps": args.n_steps,
                            "dt": args.T / args.n_steps,
                            "m": m_val,
                            "total_wall_sec": f"{wall:.9e}",
                            "per_step_sec": f"{per_step:.9e}",
                            "states_per_sec": f"{states_per_sec:.9e}",
                            "n_steps_actual": n_steps_actual,
                            "final_err_vs_ref": final_err_field,
                            "m_eff_median": m_eff_median_field,
                            "m_eff_max": m_eff_max_field,
                        }
                    )
                    extra = (
                        f", n_steps_actual={n_steps_actual}, err={final_err_field}"
                        f", m_eff_median={m_eff_median_field}"
                        if is_adaptive
                        else ""
                    )
                    print(
                        f"    trial {trial}: wall={wall:.4f}s, "
                        f"per_step={per_step:.4e}s ({states_per_sec:.3e} states/sec)"
                        f"{extra}"
                    )

                per_step_times = [
                    t / max(s, 1)
                    for t, s in zip(trial_times, trial_n_steps_actual, strict=True)
                ]
                summary.append(
                    {
                        "n": n,
                        "dim": dim,
                        "method": method,
                        "m": m_val,
                        "trials": len(per_step_times),
                        "per_step_sec_min": min(per_step_times),
                        "per_step_sec_median": statistics.median(per_step_times),
                        "states_per_sec_median": dim
                        / statistics.median(per_step_times),
                    }
                )
                if is_adaptive and trial_final_err:
                    # m_eff_stats per-trial 統計を per-cell に集約.
                    m_eff_medians = [s["median"] for s in trial_m_eff_stats]
                    m_eff_maxes = [s["max"] for s in trial_m_eff_stats]
                    adaptive_summary.append(
                        {
                            "n": n,
                            "dim": dim,
                            "method": method,
                            "m": m_val,
                            "n_steps_actual_median": statistics.median(
                                trial_n_steps_actual
                            ),
                            "n_steps_actual_min": min(trial_n_steps_actual),
                            "n_steps_actual_max": max(trial_n_steps_actual),
                            "final_err_vs_ref_median": statistics.median(
                                trial_final_err
                            ),
                            "per_step_sec_median": statistics.median(per_step_times),
                            "total_wall_sec_median": statistics.median(trial_times),
                            "m_eff_median": (
                                statistics.median(m_eff_medians)
                                if m_eff_medians
                                else float("nan")
                            ),
                            "m_eff_max": (max(m_eff_maxes) if m_eff_maxes else 0),
                        }
                    )

    # reference 計算自体の総 wall time を machine info に併記する.
    # 大 n で reference 1 本に 1-2 時間かかる現実問題に対する透明性として,
    # adaptive section の per-n 値とは別に「合算値」が機械情報節に出る.
    if reference_wall_by_n:
        machine_info["reference_wall_sec_total"] = (
            f"{sum(reference_wall_by_n.values()):.3f}"
        )

    write_outputs(
        out_dir,
        rows,
        summary,
        adaptive_summary,
        machine_info,
        args,
        reference_wall_by_n=reference_wall_by_n,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
