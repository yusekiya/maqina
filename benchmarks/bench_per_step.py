"""per-step wall time benchmark (M2 / Trotter / Suzuki S_4 / CFM4:2).

``QuantumAnnealer.run`` の 1 step あたりの実時間を, ``method`` ×
``n`` (スピン数 / Hilbert 空間次元 ``2^n``) で sweep して計測する.
Phase 1 の DoD (issue #1) と Phase 2 の DoD (issue #18), Phase 3 の DoD
(issue #30) で「``bench_per_step.py`` で M2 / Trotter / Suzuki S_4 / CFM4:2
の per-step 数値が記録され ``benchmarks/results/`` 配下に置かれている
(クロスオーバ実測)」を満たすためのベンチエントリポイント.

設計規約は ``docs/design.md`` §10, 実行手順は ``benchmarks/README.md``,
全体規約は ``CLAUDE.md`` 「ベンチマーク」節を参照する.

サポート ``method`` (Phase 3 末時点):

* ``"m2"``: M2 中点則 + Lanczos (Phase 1).
* ``"trotter"``: Strang 2 次 Trotter (Phase 2 C3).
* ``"trotter_suzuki4"``: Suzuki S_4 4 次 Trotter (Phase 2 C4).
* ``"cfm4"``: CFM4:2 commutator-free Magnus + Lanczos (Phase 3 C2).

``method`` ごとに per-step コストの内訳が異なる:

* M2: per-step ``m·dim`` flops (Lanczos m matvec).
* Trotter: per-step ``(N+1)·dim`` flops (phase 1 + bit-flip N).
* Suzuki S_4: per-step ``5·(N+1)·dim`` flops (Strang 5 回).
* CFM4:2: per-step ``2m·dim`` flops (Lanczos 2 回, M2 の 2 倍重い).

LTE order も異なる (M2 / Strang は ``O(dt^3)``, Suzuki S_4 / CFM4:2 は
``O(dt^5)``) ので, 同じ ``n_steps`` での生 wall time 比較に加えて, 同じ
精度を要求したときの required ``n_steps`` のずれを別途見積もる必要がある.

出力:

* ``benchmarks/results/<YYYYMMDD-HHMMSS>/bench_per_step.csv``: per-trial
  生データ (n, dim, method, trial, n_steps, dt, m, total_wall_sec,
  per_step_sec, states_per_sec).
* ``benchmarks/results/<YYYYMMDD-HHMMSS>/bench_per_step.md``: 集計表
  (per-method summary + cross-method 比較表) + machine info.

CLI 例::

    uv run python benchmarks/bench_per_step.py --n-values 4,8,12 --n-steps 50
    uv run python benchmarks/bench_per_step.py --methods m2,trotter
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

Phase 3 で ``method="cfm4"`` の sweep を追加済 (issue #32 / #30 の DoD).
Richardson 経路 (``method="cfm4_adaptive_richardson"``) の追加は Phase 4 で
本ファイルに sweep を増やす予定.
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

from kryanneal import IsingProblem, QuantumAnnealer, Schedule, set_blas_threads
from kryanneal.initial_states import uniform_superposition

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_ROOT = REPO_ROOT / "benchmarks" / "results"

# サポートする method 一覧. ``QuantumAnnealer.run`` の Literal 型ヒントと
# 揃える. Richardson 系は Phase 4 で追加予定.
_VALID_METHODS: tuple[str, ...] = ("m2", "trotter", "trotter_suzuki4", "cfm4")


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
            "kryanneal per-step wall time benchmark "
            "(M2 / Trotter / Suzuki S_4 / CFM4:2; "
            "see docs/design.md §10)"
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
        "--m",
        type=int,
        default=24,
        help=(
            "Lanczos subspace dimension (default: 24). Only used by "
            "method='m2'; Trotter methods do not invoke Lanczos."
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
            "if specified, call kryanneal.set_blas_threads(N) at startup to "
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
) -> float:
    """``QuantumAnnealer.run`` の wall time (秒) を ``time.perf_counter`` で計る.

    壁時計のジッタを最小化するため, 1 試行で run 1 回ぶん計測する.
    ``n_steps`` 内のループは Python 側に閉じているため step 単位の
    overhead もそこそこ乗る. ``method`` ごとに :class:`QuantumAnnealer` の
    コンストラクタ引数 (``m`` を渡すかどうか) が変わる: Trotter 系
    (``"trotter"`` / ``"trotter_suzuki4"``) は Lanczos を呼ばないため ``m``
    は無視されるが, デフォルト値で渡しても害は無いので統一的に渡す.
    ``"m2"`` / ``"cfm4"`` 経路は ``m`` がそのまま Lanczos 部分空間次元として
    効く.
    """
    if method not in _VALID_METHODS:
        raise ValueError(f"unsupported method {method!r}; valid: {_VALID_METHODS!r}")
    ann = QuantumAnnealer(problem, schedule, m=m)
    t_start = time.perf_counter()
    res = ann.run(psi0, t0, t1, method=method, n_steps=n_steps)  # type: ignore[arg-type]
    t_end = time.perf_counter()
    # res を黒箱に積んでおいて dead code elimination されないようにする.
    # `n_matvec` を最後に touch するだけで JIT/AOT 関係無しに副作用化する.
    _ = res.n_matvec
    return t_end - t_start


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
        rust_mod = importlib.import_module("kryanneal._rust")
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
    machine_info: dict[str, str],
    args: argparse.Namespace,
) -> None:
    """CSV (生データ) と markdown (集計 + machine info) を ``out_dir`` に書く.

    markdown は **per-method summary** と **method ごとの median 比較表**
    の 2 種類を出す. 比較表は M2 を基準にした ratio を併記し, 同じ
    ``n_steps`` での raw per-step コスト比較を一目で見られるようにする
    (LTE order の違いから「精度を揃えた場合の wall time 比較」は別途
    必要; 詳細は冒頭 docstring の Notes 参照).
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
    lines.append("## Summary (per-n × method)")
    lines.append("")
    lines.append(
        "| n | dim | method | per-step (sec) min | per-step (sec) median | "
        "states/sec (median) | trials |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for row in summary:
        lines.append(
            f"| {row['n']} | {row['dim']} | {row['method']} | "
            f"{row['per_step_sec_min']:.6e} | "
            f"{row['per_step_sec_median']:.6e} | "
            f"{row['states_per_sec_median']:.3e} | "
            f"{row['trials']} |"
        )
    lines.append("")

    # Cross-method 比較: 各 n で method ごとの median per-step を横並びに
    # 配置し, M2 を基準とした ratio (m2/method) を併記する.
    methods_in_summary = sorted({str(s["method"]) for s in summary})
    if len(methods_in_summary) > 1:
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

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {csv_path.relative_to(REPO_ROOT)}")
    print(f"wrote {md_path.relative_to(REPO_ROOT)}")


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
    schedule = Schedule.linear(T=args.T)

    for n in args.n_values:
        problem = make_random_problem(n, seed=args.seed)
        psi0 = uniform_superposition(n)
        dim = 1 << n
        print(
            f"[n={n} dim={dim}] methods={args.methods}, "
            f"warmup={args.warmup}, repeat={args.repeat}"
        )

        for method in args.methods:
            print(f"  method={method}")
            for _ in range(args.warmup):
                time_method_run(
                    problem, schedule, psi0, 0.0, args.T, args.n_steps, method, args.m
                )

            trial_times: list[float] = []
            for trial in range(args.repeat):
                wall = time_method_run(
                    problem,
                    schedule,
                    psi0,
                    0.0,
                    args.T,
                    args.n_steps,
                    method,
                    args.m,
                )
                trial_times.append(wall)
                per_step = wall / args.n_steps
                states_per_sec = dim / per_step if per_step > 0 else float("inf")
                rows.append(
                    {
                        "n": n,
                        "dim": dim,
                        "method": method,
                        "trial": trial,
                        "n_steps": args.n_steps,
                        "dt": args.T / args.n_steps,
                        "m": args.m,
                        "total_wall_sec": f"{wall:.9e}",
                        "per_step_sec": f"{per_step:.9e}",
                        "states_per_sec": f"{states_per_sec:.9e}",
                    }
                )
                print(
                    f"    trial {trial}: wall={wall:.4f}s, "
                    f"per_step={per_step:.4e}s ({states_per_sec:.3e} states/sec)"
                )

            per_step_times = [t / args.n_steps for t in trial_times]
            summary.append(
                {
                    "n": n,
                    "dim": dim,
                    "method": method,
                    "trials": len(per_step_times),
                    "per_step_sec_min": min(per_step_times),
                    "per_step_sec_median": statistics.median(per_step_times),
                    "states_per_sec_median": dim / statistics.median(per_step_times),
                }
            )

    write_outputs(out_dir, rows, summary, machine_info, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
