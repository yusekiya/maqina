"""per-step wall time overhead benchmark with observables / save_tlist (Phase 5).

Phase 5 (issue #47) で追加した ``observables`` / ``save_tlist`` / ``store_states``
経路の per-step overhead を, **observables なし / observables あり /
observables + save_tlist** の 3 ケースで比較する. ``bench_per_step.py`` の
測定構造を流用し, 出力経路を ``benchmarks/results/<YYYYMMDD-HHMMSS>/``
配下に置く規約も同様.

CLI 例::

    # smoke (リリースビルド推奨): n=4 / cfm4 で 1 cell だけ走らせる.
    uv run python benchmarks/bench_observables.py --methods cfm4 --n-values 4 --repeat 1 --warmup 1

    # m2 / cfm4 / adaptive Richardson を n=4,8 で sweep.
    uv run python benchmarks/bench_observables.py \\
        --methods m2,cfm4,cfm4_adaptive_richardson_krylov --n-values 4,8 --repeat 3

issue #47 の Definition of Done では本スクリプトの smoke 動作確認のみが
要件で, **本番 sweep は merge 後に Linux サーバー上で実行して結果を別途
コメント添付する運用** とする (``CLAUDE.md`` ベンチマーク節「同一マシン
上の before / after」原則 + macOS で取った結果は Phase 5 リリース比較
対象に使わない).

サポート ``method``:

* ``"m2"`` (固定 dt)
* ``"cfm4"`` (固定 dt)
* ``"cfm4_adaptive_richardson_krylov"`` (adaptive PI controller; ``save_tlist``
  指定時は dt を target に clamp)

Trotter 系を含めない理由: Phase 5 で観測量経路は ``Z`` 基底対角 Observable
のみ対応 (``Observable.magnetization`` / ``Observable.ising_energy``).
Trotter / Trotter-Suzuki S_4 も同じ recorder を通せるが, Phase 5 の最も
代表的な経路として M2 / CFM4:2 / adaptive Richardson に絞って overhead
を測れば足りる. 残りは必要になった時点で追加する.

出力:

* ``benchmarks/results/<YYYYMMDD-HHMMSS>/bench_observables.csv``: per-trial
  生データ (n, dim, method, mode, trial, n_steps, total_wall_sec,
  per_step_sec, states_per_sec).
* ``benchmarks/results/<YYYYMMDD-HHMMSS>/bench_observables.md``: 集計表
  (per-method × mode summary + machine info).
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

from kryanneal import (
    IsingProblem,
    Observable,
    QuantumAnnealer,
    Schedule,
    set_blas_threads,
)
from kryanneal.initial_states import uniform_superposition

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_ROOT = REPO_ROOT / "benchmarks" / "results"

_VALID_METHODS: tuple[str, ...] = ("m2", "cfm4", "cfm4_adaptive_richardson_krylov")

# observables / save_tlist の組合せを実験する mode 一覧.
# * "baseline": save_tlist=None, observables=None, store_states=False (最節約モード)
# * "obs_only": save_tlist=<linspace>, observables=<M_z+H_p>, store_states=False
# * "obs_states": save_tlist=<linspace>, observables=<M_z+H_p>, store_states=True
_MODES: tuple[str, ...] = ("baseline", "obs_only", "obs_states")

# adaptive 経路では n_steps を渡さない. fixed dt 経路と同じ呼出 API に
# 揃えるため bench 側で `--n-steps` を dt_init 提案に使う.
_ADAPTIVE_METHODS: frozenset[str] = frozenset({"cfm4_adaptive_richardson_krylov"})


def _parse_int_list(text: str) -> list[int]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return [int(p) for p in parts]


def _parse_method_list(text: str) -> list[str]:
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
    parser = argparse.ArgumentParser(
        description=(
            "kryanneal observables / save_tlist overhead benchmark (Phase 5, issue #47)"
        )
    )
    parser.add_argument(
        "--n-values",
        type=_parse_int_list,
        default=[4, 8],
        help="comma-separated sweep over spin counts (default: 4,8)",
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
        "--save-tlist-len",
        type=int,
        default=11,
        help=(
            "number of save_tlist samples for obs_only / obs_states modes "
            "(default: 11; np.linspace(t0, t1, K) で生成)"
        ),
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="number of timed trials per (n, method, mode) cell (default: 3)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="number of warmup trials per (n, method, mode) cell (default: 1)",
    )
    parser.add_argument(
        "--T",
        type=float,
        default=1.0,
        help="total anneal time T (default: 1.0)",
    )
    parser.add_argument(
        "--m",
        type=int,
        default=24,
        help="Lanczos subspace dimension for m2 / cfm4 / adaptive (default: 24)",
    )
    parser.add_argument(
        "--blas-threads",
        type=int,
        default=None,
        help=(
            "if specified, call kryanneal.set_blas_threads(N) at startup. "
            "Use --blas-threads 1 for a machine-independent single-thread "
            "baseline. Default: None (leave BLAS thread counts untouched)."
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
        default=20260516,
        help="numpy RNG seed (default: 20260516)",
    )
    return parser.parse_args(argv)


def make_random_problem(n: int, seed: int) -> IsingProblem:
    rng = np.random.default_rng(seed)
    dim = 1 << n
    h_p = rng.uniform(-1.0, 1.0, size=dim).astype(np.float64)
    h_x = np.ones(n, dtype=np.float64)
    return IsingProblem(n=n, H_p_diag=h_p, h_x=h_x)


def _build_run_kwargs(
    *,
    method: str,
    mode: str,
    n_steps: int,
    m: int,
    T: float,
    save_tlist: np.ndarray,
    observables: dict[str, Observable],
) -> dict:
    """``QuantumAnnealer.run`` の kw 引数を method × mode 組合せで組み立てる."""
    kwargs: dict = {"method": method}
    if method in _ADAPTIVE_METHODS:
        # adaptive 経路: n_steps を dt_init 提案に流用. atol は default 1e-8.
        kwargs["dt_init"] = T / n_steps
    else:
        kwargs["n_steps"] = n_steps
    if mode == "baseline":
        pass  # 何も渡さない (最節約モード).
    elif mode == "obs_only":
        kwargs["observables"] = observables
        kwargs["save_tlist"] = save_tlist
    elif mode == "obs_states":
        kwargs["observables"] = observables
        kwargs["save_tlist"] = save_tlist
        kwargs["store_states"] = True
    else:
        raise ValueError(f"unknown mode: {mode!r}")
    return kwargs


def time_one_run(
    annealer: QuantumAnnealer,
    psi0: np.ndarray,
    t0: float,
    t1: float,
    run_kwargs: dict,
) -> tuple[float, int]:
    """``annealer.run`` の wall time (秒) を ``time.perf_counter`` で計る."""
    tic = time.perf_counter()
    res = annealer.run(psi0, t0, t1, **run_kwargs)
    wall = time.perf_counter() - tic
    n_actual = res.n_steps_actual if res.n_steps_actual is not None else res.n_steps
    return wall, int(n_actual)


def _gather_machine_info(args: argparse.Namespace) -> dict[str, str]:
    """machine info 行 (markdown 出力用)."""
    try:
        rust_mod = importlib.import_module("kryanneal._rust")
        has_blas = bool(getattr(rust_mod, "__has_blas__", False))
    except ImportError:
        has_blas = False
    return {
        "timestamp_utc": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": str(os.process_cpu_count() or 1),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "kryanneal_blas_feature": str(has_blas),
        "blas_threads_arg": str(args.blas_threads),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.blas_threads is not None:
        set_blas_threads(int(args.blas_threads))

    results_dir = args.results_dir
    if results_dir is None:
        ts = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        results_dir = DEFAULT_RESULTS_ROOT / ts
    results_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / "bench_observables.csv"
    md_path = results_dir / "bench_observables.md"

    machine_info = _gather_machine_info(args)
    print(f"# bench_observables: writing to {results_dir}")
    for k, v in machine_info.items():
        print(f"  {k}: {v}")

    rows: list[dict] = []
    for n in args.n_values:
        dim = 1 << n
        prob = make_random_problem(n, args.seed)
        sched = Schedule.linear(T=args.T)
        psi0 = uniform_superposition(n)
        ann = QuantumAnnealer(prob, sched, m=args.m)
        save_tlist = np.linspace(0.0, args.T, args.save_tlist_len, dtype=np.float64)
        observables = {
            "M_z": Observable.magnetization(n),
            "H_p": Observable.ising_energy(prob),
        }
        for method in args.methods:
            for mode in _MODES:
                run_kwargs = _build_run_kwargs(
                    method=method,
                    mode=mode,
                    n_steps=args.n_steps,
                    m=args.m,
                    T=args.T,
                    save_tlist=save_tlist,
                    observables=observables,
                )
                # warmup (結果不採用).
                for _ in range(int(args.warmup)):
                    time_one_run(ann, psi0, 0.0, args.T, run_kwargs)
                for trial in range(int(args.repeat)):
                    wall, n_actual = time_one_run(ann, psi0, 0.0, args.T, run_kwargs)
                    rows.append(
                        {
                            "n": n,
                            "dim": dim,
                            "method": method,
                            "mode": mode,
                            "trial": trial,
                            "n_steps": int(args.n_steps),
                            "n_steps_actual": int(n_actual),
                            "wall_sec": float(wall),
                            "per_step_sec": float(wall / max(n_actual, 1)),
                        }
                    )
                    print(
                        f"  n={n:2d} method={method:30s} mode={mode:11s} "
                        f"trial={trial} wall={wall:.4e}s "
                        f"per_step={wall / max(n_actual, 1):.4e}s "
                        f"n_actual={n_actual}"
                    )

    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "n",
                "dim",
                "method",
                "mode",
                "trial",
                "n_steps",
                "n_steps_actual",
                "wall_sec",
                "per_step_sec",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    with md_path.open("w") as fh:
        fh.write("# bench_observables results\n\n")
        fh.write("## Machine info\n\n")
        for k, v in machine_info.items():
            fh.write(f"- **{k}**: {v}\n")
        fh.write("\n## Summary (per_step_sec median across trials)\n\n")
        fh.write("| n | method | mode | per_step_sec (median) |\n")
        fh.write("|---|---|---|---|\n")
        # rows を (n, method, mode) で集約して median を取る.
        agg: dict[tuple[int, str, str], list[float]] = {}
        for row in rows:
            key = (int(row["n"]), str(row["method"]), str(row["mode"]))
            agg.setdefault(key, []).append(float(row["per_step_sec"]))
        for (n, method, mode), vals in sorted(agg.items()):
            fh.write(f"| {n} | {method} | {mode} | {statistics.median(vals):.4e} |\n")

    print(f"\nWrote {csv_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
