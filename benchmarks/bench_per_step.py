"""M2 (Phase 1) per-step wall time benchmark.

``QuantumAnnealer.run(method="m2")`` の M2 中点則 1 step あたりの実時間を
``n`` (スピン数 / Hilbert 空間次元 ``2^n``) を sweep して計測する. Phase 1
の DoD (issue #1) で「``bench_per_step.py`` で M2 ベースライン数値が 1 回
記録され ``benchmarks/results/`` 配下に置かれている」を満たすための
ベンチエントリポイント.

設計規約は ``docs/design.md`` §10, 実行手順は ``benchmarks/README.md``,
全体規約は ``CLAUDE.md`` 「ベンチマーク」節を参照する.

出力:

* ``benchmarks/results/<YYYYMMDD-HHMMSS>/bench_per_step.csv``: per-trial
  生データ (n, trial, dt, total_step_count, total_wall_sec, per_step_sec,
  states_per_sec).
* ``benchmarks/results/<YYYYMMDD-HHMMSS>/bench_per_step.md``: 集計表 +
  machine info.

CLI 例::

    uv run python benchmarks/bench_per_step.py --n-values 4,8,12 --n-steps 50

ベンチは原則 ``--release`` build (``maturin develop --uv --release``) で
取る. debug build (``maturin develop --uv`` のみ) の値はベースラインに
ふさわしくないため,本スクリプトは実行時に ``_rust.__has_blas__`` を含む
build フラグを記録するに留め, build profile 自体は呼び出し側の責任とする.

CFM4 / Richardson 経路の追加は Phase 3 / Phase 4 で本ファイルに sweep を
増やす予定 (``method="cfm4" / "cfm4_adaptive_richardson"`` を追加).
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

from kryanneal import IsingProblem, QuantumAnnealer, Schedule
from kryanneal.initial_states import uniform_superposition

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_ROOT = REPO_ROOT / "benchmarks" / "results"


def _parse_int_list(text: str) -> list[int]:
    """``"4,8,12"`` のような CSV 文字列を ``[4, 8, 12]`` にする."""
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return [int(p) for p in parts]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI 引数を parse する.

    sweep 対象の ``n`` 列は ``--n-values`` で受け取り, 計測安定性のための
    ``--repeat`` / ``--warmup`` は分けて指定可能にする (warmup は cache
    warm のために常に必要だが結果には残さない設計).
    """
    parser = argparse.ArgumentParser(
        description=(
            "kryanneal M2 per-step wall time benchmark "
            "(Phase 1 baseline, see docs/design.md §10)"
        )
    )
    parser.add_argument(
        "--n-values",
        type=_parse_int_list,
        default=[4, 8, 12, 16],
        help="comma-separated sweep over spin counts (default: 4,8,12,16)",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=50,
        help="number of M2 driver steps per measurement (default: 50)",
    )
    parser.add_argument(
        "--m",
        type=int,
        default=24,
        help="Lanczos subspace dimension (default: 24)",
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


def time_m2_run(
    problem: IsingProblem,
    schedule: Schedule,
    psi0: np.ndarray,
    t0: float,
    t1: float,
    n_steps: int,
    m: int,
) -> float:
    """``QuantumAnnealer.run`` の wall time (秒) を ``time.perf_counter`` で計る.

    壁時計のジッタを最小化するため, 1 試行で run 1 回ぶん計測する.
    ``n_steps`` 内のループは Python 側に閉じているため step 単位の
    overhead もそこそこ乗るが, Phase 1 はそのスナップショットを取るのが
    目的.
    """
    ann = QuantumAnnealer(problem, schedule, m=m)
    t_start = time.perf_counter()
    res = ann.run(psi0, t0, t1, method="m2", n_steps=n_steps)
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
    """CSV (生データ) と markdown (集計 + machine info) を ``out_dir`` に書く."""
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "bench_per_step.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_path = out_dir / "bench_per_step.md"
    lines: list[str] = []
    lines.append("# bench_per_step results (M2, Phase 1 baseline)")
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
    lines.append("## Summary (per-n)")
    lines.append("")
    lines.append(
        "| n | dim | per-step (sec) min | per-step (sec) median | "
        "states/sec (median) | trials |"
    )
    lines.append("|---|---|---|---|---|---|")
    for row in summary:
        lines.append(
            f"| {row['n']} | {row['dim']} | "
            f"{row['per_step_sec_min']:.6e} | "
            f"{row['per_step_sec_median']:.6e} | "
            f"{row['states_per_sec_median']:.3e} | "
            f"{row['trials']} |"
        )
    lines.append("")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {csv_path.relative_to(REPO_ROOT)}")
    print(f"wrote {md_path.relative_to(REPO_ROOT)}")


def main(argv: list[str] | None = None) -> int:
    """エントリポイント. 各 ``n`` について warmup → 計測を回し, CSV + md を書く."""
    args = parse_args(argv)

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
        print(f"[n={n} dim={dim}] warmup={args.warmup}, repeat={args.repeat}")

        for _ in range(args.warmup):
            time_m2_run(problem, schedule, psi0, 0.0, args.T, args.n_steps, args.m)

        trial_times: list[float] = []
        for trial in range(args.repeat):
            wall = time_m2_run(
                problem, schedule, psi0, 0.0, args.T, args.n_steps, args.m
            )
            trial_times.append(wall)
            per_step = wall / args.n_steps
            states_per_sec = dim / per_step if per_step > 0 else float("inf")
            rows.append(
                {
                    "n": n,
                    "dim": dim,
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
                f"  trial {trial}: wall={wall:.4f}s, "
                f"per_step={per_step:.4e}s ({states_per_sec:.3e} states/sec)"
            )

        per_step_times = [t / args.n_steps for t in trial_times]
        summary.append(
            {
                "n": n,
                "dim": dim,
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
