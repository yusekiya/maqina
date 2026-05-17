"""block-fusion (Phase 6 C3, issue #64) bench: trotter_step / apply_h_kryanneal の
per-step time を計測する.

## 計測対象と目的

| kernel | 計測理由 |
|---|---|
| ``trotter_step`` | Phase 6 C3 の主スコープ. ``apply_single_mode_axis_i`` を 1 軸ずつ ``n`` 回呼ぶ旧実装から, 連続 ``FUSE_K = 4`` qubit を 1 つの rayon chunk closure 内で per-axis 逐次に適用する ``apply_multi_qubit_gate_fused`` 経路に書き換えた効果を測る. 期待: per-step rayon barrier 数 ``2n+2 → n/FUSE_K + 2``, N=20 で 40 → 7. compute は per-axis × k と同じ (`2k·dim` ops, 増えない) で chunk-resident cache 効果と barrier 削減のみで稼ぐ (dense 2^k×2^k matmul 経路は本 PR 初版で 0.81× regression したため放棄). |
| ``apply_h_kryanneal`` | Phase 6 D (issue #79) の主スコープ. ``apply_h_kryanneal_rayon`` を **group-fused 3-phase 形** (per-chunk diag + low-i / group-fused 高 i / per-chunk 残り高 i) に書き換えた効果. 連続 ``fused_k`` 個の高 i axes (mask ≥ chunk_size) の partner 参照を ``2^fused_k`` 個の連続 chunk から成る group (L2 resident) で完結させて DRAM v traffic を ``dim · (1 + h_baseline) → dim · (1 + h_naive)`` (``h_naive = h_baseline - fused_k``) に削減することを期待. C3 期の Phase 6 C3 副次スコープ (``RAYON_CHUNK_MAX`` 縮小) は本 issue で superseded. 詳細は ``docs/design.md`` §5.1.4. |

両 kernel とも本 script は **計測本体のみ** 提供する. baseline (Phase 6 C2
完了時点 = main branch tip) と after (C3 適用 = 本 PR branch tip) の per-step
比較は ``CLAUDE.md`` 「ベンチマーク」節の作法に従い操作員側で 2 回計測する:

::

    # 1. baseline (main tip)
    git switch main
    RUSTFLAGS="-C target-cpu=native" uv run maturin develop --uv --release
    RAYON_NUM_THREADS=64 uv run python benchmarks/bench_block_fusion.py \
        --label c2-baseline \
        --output-json benchmarks/results/<run-id>/c2-baseline.json

    # 2. after (C3 適用)
    git switch <pr-branch>
    RUSTFLAGS="-C target-cpu=native" uv run maturin develop --uv --release
    RAYON_NUM_THREADS=64 uv run python benchmarks/bench_block_fusion.py \
        --label c3-after \
        --output-json benchmarks/results/<run-id>/c3-after.json

    # 3. 手動 diff (md 表) は per-cell speedup = baseline_median / after_median を
    #    表計算で組む. 自動化が必要になったら mode_compare をフォローアップで足す.

## Acceptance (issue #64 / #79)

- (issue #64, 達成済み) N=20, cpu_count=64 Linux サーバー, ``RAYON_NUM_THREADS=64``,
  ``BLAS_THREADS=1`` で ``trotter_step`` の per-step time が baseline の
  **>= 1.3×** 改善 (実測 4.01×).
- (issue #79, 本 PR で検証) 同条件で ``apply_h_kryanneal`` の per-step time が
  Phase 6 C2.5 完了時点 (main tip) の baseline に対し **>= 1.3×** 改善.
  副次目標: N ∈ {18, 22} で regression なし.
- 数値一致: 別途 ``cargo test`` + ``uv run pytest`` で ``rel < 1e-13`` 確認済み
  (本 bench では検証しない).

## Thread pool 運用

``CLAUDE.md`` 「Thread pool 運用 (rayon × BLAS)」節に従い:

- rayon thread 数: ``RAYON_NUM_THREADS`` 環境変数で **プロセス起動時に** 固定
  (rayon global pool は最初の rayon op で構築され, それ以降は変更不可).
- BLAS thread 数: 本 script 冒頭で ``kryanneal.set_blas_threads(args.blas_threads)``
  を呼び 1 thread に固定 (default 1). rayon と BLAS の thread pool が同時に
  cpu_count 個ずつ thread を張ると context-switch で性能劣化するため.

## CLI 例

::

    # 本番 sweep
    RAYON_NUM_THREADS=64 uv run python benchmarks/bench_block_fusion.py

    # smoke (macOS, 小 N, 少 repeat)
    uv run python benchmarks/bench_block_fusion.py \
        --n-values 16,18 --repeat 3 --warmup 1

    # output 先指定
    uv run python benchmarks/bench_block_fusion.py \
        --output-dir benchmarks/results/20260517-block-fusion/

## 出力

``benchmarks/results/<YYYYMMDD-HHMMSS>/`` 配下に:

- ``bench_block_fusion.csv``: 全 trial raw データ
  (label, n, dim, kernel, trial, wall_sec, calls_per_sec).
- ``bench_block_fusion.md``: machine info + per-cell summary (median wall_sec,
  median calls_per_sec) を ``(n, kernel)`` 行列で書く.
- ``bench_block_fusion.json``: ``--output-json`` 指定時に raw + machine_info
  を JSON で書き出す (baseline vs after の手動 diff 用).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

UTC = timezone.utc

DEFAULT_N_VALUES = (18, 20, 22)
DEFAULT_REPEAT = 7
DEFAULT_WARMUP = 2
DEFAULT_DT = 0.01


def _measure_trotter_step(n: int, repeat: int, warmup: int, dt: float) -> list[float]:
    """``_rust.trotter_step_py`` の wall time を repeat 回計測して返す.

    Strang 2 次 Trotter 1 step. C3 (multi-qubit gate fusion) の主スコープ.
    """
    from kryanneal import _rust  # pyright: ignore[reportMissingImports]

    dim = 1 << n
    rng = np.random.default_rng(0xC3_B10C_F0_7807_7E40 ^ n)
    psi = (
        rng.uniform(-1.0, 1.0, size=dim) + 1j * rng.uniform(-1.0, 1.0, size=dim)
    ).astype(np.complex128)
    # L2 正規化 (unitarity 維持).
    psi /= np.linalg.norm(psi)
    h_x = rng.uniform(0.5, 1.5, size=n).astype(np.float64)
    h_p_diag = rng.uniform(-1.0, 1.0, size=dim).astype(np.float64)
    a_t = 0.5
    b_t = 0.5

    for _ in range(warmup):
        _ = _rust.trotter_step_py(psi, h_x, h_p_diag, a_t, b_t, dt, n)

    timings: list[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        _ = _rust.trotter_step_py(psi, h_x, h_p_diag, a_t, b_t, dt, n)
        t1 = time.perf_counter()
        timings.append(t1 - t0)
    return timings


def _measure_apply_h_kryanneal(n: int, repeat: int, warmup: int) -> list[float]:
    """``_rust.apply_h_kryanneal_py`` の wall time を repeat 回計測して返す.

    matvec primitive (Lanczos / CFM4:2 内部). C3 の副次スコープ (chunk_size 縮小).
    h_x は all-ones で全 i bit-flip pass を踏ませる.
    """
    from kryanneal import _rust  # pyright: ignore[reportMissingImports]

    dim = 1 << n
    rng = np.random.default_rng(0xC3_B10C_F0_AA77EC ^ n)
    v = (
        rng.uniform(-1.0, 1.0, size=dim) + 1j * rng.uniform(-1.0, 1.0, size=dim)
    ).astype(np.complex128)
    h_x = np.ones(n, dtype=np.float64)
    h_p_diag = rng.uniform(-1.0, 1.0, size=dim).astype(np.float64)
    a_t = 0.5
    b_t = 0.5

    for _ in range(warmup):
        _ = _rust.apply_h_kryanneal_py(v, h_x, h_p_diag, a_t, b_t)

    timings: list[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        _ = _rust.apply_h_kryanneal_py(v, h_x, h_p_diag, a_t, b_t)
        t1 = time.perf_counter()
        timings.append(t1 - t0)
    return timings


def _machine_info(label: str) -> dict[str, Any]:
    info: dict[str, Any] = {
        "label": label,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "cpu_count_logical": os.cpu_count(),
        "timestamp_utc": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "rayon_num_threads_env": os.environ.get("RAYON_NUM_THREADS"),
    }
    try:
        import kryanneal
        from kryanneal import _rust  # pyright: ignore[reportMissingImports]

        info["kryanneal_version"] = getattr(kryanneal, "__version__", "unknown")
        info["__has_blas__"] = bool(getattr(_rust, "__has_blas__", False))
        info["__has_rayon__"] = bool(getattr(_rust, "__has_rayon__", False))
        info["__has_simd__"] = bool(getattr(_rust, "__has_simd__", False))
    except ImportError:
        info["kryanneal_version"] = "import-failed"
        info["__has_blas__"] = None
        info["__has_rayon__"] = None
        info["__has_simd__"] = None
    return info


def _write_csv(path: Path, trials: list[dict[str, Any]]) -> None:
    if not trials:
        return
    fieldnames = list(trials[0].keys())
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trials)


def _write_markdown(
    path: Path,
    machine: dict[str, Any],
    trials: list[dict[str, Any]],
) -> None:
    # per-cell median 集約 (n, kernel) → median(wall_sec).
    cells: dict[tuple[int, str], list[float]] = {}
    for t in trials:
        key = (int(t["n"]), str(t["kernel"]))
        cells.setdefault(key, []).append(float(t["wall_sec"]))

    lines: list[str] = []
    lines.append("# bench_block_fusion results")
    lines.append("")
    lines.append(f"label: `{machine['label']}`")
    lines.append("")
    lines.append("## machine info")
    lines.append("")
    for k, v in machine.items():
        lines.append(f"- **{k}**: `{v}`")
    lines.append("")
    lines.append(
        "## per-cell median wall time (repeat={})".format(
            len(next(iter(cells.values()))) if cells else 0,
        )
    )
    lines.append("")
    lines.append("| n | kernel | median wall_sec | median calls/sec |")
    lines.append("|---|---|---|---|")
    for (n, kernel), wall_list in sorted(cells.items()):
        wall_list_sorted = sorted(wall_list)
        med = wall_list_sorted[len(wall_list_sorted) // 2]
        calls_per_sec = 1.0 / med if med > 0 else float("inf")
        lines.append(f"| {n} | {kernel} | {med:.6e} | {calls_per_sec:.3e} |")
    lines.append("")
    lines.append("## 使い方 (baseline vs after の手動 diff)")
    lines.append("")
    lines.append(
        "1. main tip と PR tip で本 script を **同条件** (RAYON_NUM_THREADS, "
        "blas_threads, n_values, repeat) で 2 回回す."
    )
    lines.append(
        "2. 2 つの md / CSV を見比べ, per-cell speedup = "
        "`baseline_median / after_median` を計算."
    )
    lines.append("3. acceptance: `n=20`, `kernel=trotter_step` で `speedup >= 1.3`.")
    path.write_text("\n".join(lines) + "\n")


def _parse_int_list(s: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in s.split(",") if x.strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase 6 C3 block-fusion bench (trotter_step / apply_h_kryanneal)",
    )
    parser.add_argument(
        "--n-values",
        type=_parse_int_list,
        default=DEFAULT_N_VALUES,
        help=f"sweep する n (default {','.join(str(x) for x in DEFAULT_N_VALUES)})",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=DEFAULT_REPEAT,
        help=f"各 cell の trial 回数 (default {DEFAULT_REPEAT})",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP,
        help=f"warmup 試行数 (default {DEFAULT_WARMUP})",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=DEFAULT_DT,
        help=f"trotter_step に渡す dt (default {DEFAULT_DT})",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="bench-block-fusion",
        help="machine_info に書き込む label (baseline / after 識別用)",
    )
    parser.add_argument(
        "--blas-threads",
        type=int,
        default=1,
        help=(
            "kryanneal.set_blas_threads(N) で BLAS thread 数を固定 "
            "(default 1; rayon × BLAS 干渉を分離)"
        ),
    )
    parser.add_argument(
        "--kernels",
        type=str,
        default="trotter_step,apply_h_kryanneal",
        help="計測対象 kernel (カンマ区切り; trotter_step / apply_h_kryanneal)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=("結果出力ディレクトリ (default benchmarks/results/<YYYYMMDD-HHMMSS>/)"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="raw trials + machine_info を単一 JSON ファイルに書き出す (任意)",
    )
    args = parser.parse_args(argv)

    import kryanneal

    if args.blas_threads is not None:
        kryanneal.set_blas_threads(args.blas_threads)

    machine = _machine_info(args.label)

    kernels = [k.strip() for k in args.kernels.split(",") if k.strip()]
    valid_kernels = {"trotter_step", "apply_h_kryanneal"}
    unknown = set(kernels) - valid_kernels
    if unknown:
        print(
            f"[error] unknown kernels: {sorted(unknown)} (allowed: "
            f"{sorted(valid_kernels)})",
            file=sys.stderr,
        )
        return 2

    trials: list[dict[str, Any]] = []
    for n in args.n_values:
        for kernel in kernels:
            dim = 1 << n
            print(
                f"[measure] n={n} dim={dim} kernel={kernel} "
                f"repeat={args.repeat} warmup={args.warmup}",
                flush=True,
            )
            if kernel == "trotter_step":
                timings = _measure_trotter_step(
                    n,
                    repeat=args.repeat,
                    warmup=args.warmup,
                    dt=args.dt,
                )
            elif kernel == "apply_h_kryanneal":
                timings = _measure_apply_h_kryanneal(
                    n,
                    repeat=args.repeat,
                    warmup=args.warmup,
                )
            else:
                raise AssertionError(f"unreachable: kernel={kernel!r}")
            for trial_idx, t in enumerate(timings):
                trials.append(
                    {
                        "label": args.label,
                        "n": n,
                        "dim": dim,
                        "kernel": kernel,
                        "trial": trial_idx,
                        "wall_sec": t,
                        "calls_per_sec": 1.0 / t if t > 0 else float("inf"),
                    }
                )

    # 出力先準備.
    if args.output_dir is None:
        ts = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        out_dir = Path("benchmarks/results") / ts
    else:
        out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(out_dir / "bench_block_fusion.csv", trials)
    _write_markdown(out_dir / "bench_block_fusion.md", machine, trials)
    print(f"[done] wrote {out_dir / 'bench_block_fusion.csv'}", flush=True)
    print(f"[done] wrote {out_dir / 'bench_block_fusion.md'}", flush=True)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(
                {"machine_info": machine, "trials": trials},
                indent=2,
            )
        )
        print(f"[done] wrote {args.output_json}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
