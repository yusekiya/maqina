"""SIMD scaling bench: `apply_h_kryanneal` per-pass time の SIMD ON/OFF 比較.

issue #63 (Phase 6 C2) の acceptance「N=18, i=0,1,2 集中時の per-pass time
**>=1.5× 改善** (cpu_count=64 Linux サーバーで, BLAS thread=1)」を計測する
専用 bench. `bench_parallel_scaling.py` (C1) と同じ subprocess + child mode
構造を踏襲するが, SIMD feature の切替は **build 時** に行う必要があるため
parent/child 自動切替はせず, **操作員が異なる build で 2 回 measure を回し
3 回目に `--mode compare` で統合する** 運用にする.

## 計測対象

`_rust.apply_h_kryanneal_py` の per-call wall time を以下 2 設定で取る:

- ``all-i`` (h_x = all-ones): 全 i bit-flip pass が寄与. 本番ホットパス
  (Lanczos / CFM4:2) を最も忠実に再現する.
- ``i012-focus`` (h_x = [1, 1, 1, 0, ..., 0]): i = 0, 1, 2 のみ寄与.
  SIMD カーネルが効く範囲を分離し, issue acceptance の「i=0,1,2 集中時」
  シナリオを最大限引き出す.

各セルで warmup → repeat 回計測し, JSON に machine info / build flags
(``__has_simd__``, ``__has_blas__``, ``__has_rayon__``) と共に出す.

## 実行手順 (本番 Linux サーバー sweep)

::

    # 1. SIMD ON (default build) で measure
    RUSTFLAGS="-C target-cpu=native" uv run maturin develop --uv --release
    uv run python benchmarks/bench_simd_scaling.py \\
        --mode measure --label simd-on \\
        --output benchmarks/results/bench_simd/simd-on.json

    # 2. SIMD OFF (rayon + blas のみ, simd feature off) で measure
    #    `maturin develop` は PEP 517 backend を介さず cargo を直接呼ぶので,
    #    `MATURIN_PEP517_ARGS` は **無視される** 点に注意. feature 制御は
    #    `maturin develop --no-default-features --features ...` で直接渡す.
    #    `extension-module` は pyproject.toml の `[tool.maturin] features` 経由で
    #    通常自動付与されるが, `--features` を明示すると上書きされるため一緒に
    #    渡し直す (これがないと libpython リンクエラーになる).
    RUSTFLAGS="-C target-cpu=native" \\
        uv run maturin develop --uv --release \\
        --no-default-features --features extension-module,blas,rayon
    # build flag 確認 (期待: __has_simd__ = False)
    uv run python -c "from kryanneal import _rust; print('simd:', _rust.__has_simd__)"
    uv run python benchmarks/bench_simd_scaling.py \\
        --mode measure --label simd-off \\
        --output benchmarks/results/bench_simd/simd-off.json

    # 3. SIMD ON build に戻して compare → markdown + CSV 出力
    RUSTFLAGS="-C target-cpu=native" uv run maturin develop --uv --release
    uv run python benchmarks/bench_simd_scaling.py \\
        --mode compare \\
        --simd-on benchmarks/results/bench_simd/simd-on.json \\
        --simd-off benchmarks/results/bench_simd/simd-off.json \\
        --output-dir benchmarks/results/<YYYYMMDD-HHMMSS>/

macOS では smoke (NEON 経路の動作確認 + 数値同一性) のみ. 速度比較は
**Phase 4-5 と同じ Linux サーバー (x86_64 / glibc 2.35 / cpu_count=64 /
OpenBLAS)** 上で取る (``CLAUDE.md`` ベンチマーク節「同一マシン上の
before / after」原則). **bench は PR 本体に含めず, merge 後に issue #63
コメントとして添付する運用** (issue #47 で確定).

## 注意点

- 実 SIMD 性能向上は build 時の ``target-cpu`` 設定に依存する.
  default ``x86_64`` target では ``wide`` クレートが scalar fallback を選び
  正確性のみ提供する. 本番 measure は ``RUSTFLAGS="-C target-cpu=native"``
  を必ず設定する (AVX2 / AVX-512 / NEON を `wide` の `target_feature`
  cfg が拾えるようになる).
- BLAS thread は ``--blas-threads 1`` で 1 thread 固定にする
  (``bench_parallel_scaling.py`` と同じ acceptance 条件).
- rayon thread は ``RAYON_NUM_THREADS`` 環境変数で固定する. issue #63
  acceptance「Phase 6 C1 baseline (rayon あり, SIMD なし) と比較」と
  apples-to-apples にするには **``RAYON_NUM_THREADS=cpu_count``** (本番
  Linux server で 64) を使う. ``RAYON_NUM_THREADS=1`` は per-thread SIMD
  孤立観測用で N≥16 では memory-bandwidth bound のため SIMD 効果が見えない
  ことが #63 bench で判明 (参考: issue #63 コメント).

## 測定ノイズと推奨 repeat / runs

issue #63 final bench (Linux x86_64, cpu_count=64, OpenBLAS) で **N≥18
multi-thread cell の inter-run 変動が ~3-5× に達する** ケースが観測された
(代表例: N=20 ``i012-focus`` で 3 runs の speedup が 0.58×, 0.95×, 2.71×).
原因は per-call wall time が ms オーダで, 64 threads × 短時間 work が
背景 process / NUMA / scheduler / thermal の影響を強く受けるため.

このため SIMD ON/OFF 比較は次の運用を推奨:

1. ``--repeat 50 --warmup 10`` で 1 measure 内の中央値を安定化させる.
2. **同じ build profile で 3 回 back-to-back に measure を回し**, inter-run
   variance を確認する. 1 run だけで判断しない. 単発で 1.5× 出ても
   別 run で 0.5× になることがある (issue #63 で確認済).
3. N=16 (serial path, dim < ``MIN_RAYON_DIM = 1<<17``) は inter-run 変動が
   小さく (±0.05 程度), 安定した signal が得られる cell. N=18 / N=20
   multi-thread cell は noise が乗る前提で 3 runs の median を取る.
4. 必要なら ``taskset -c 0-31`` で physical core に pin して HT contention
   を回避する (HT enabled 64 logical = 32 physical の典型 Xeon Gold 構成
   の場合).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_N_VALUES = (16, 18, 20)
DEFAULT_REPEAT = 7
DEFAULT_WARMUP = 2
# SIMD カーネルが特化する i の集合.
SIMD_TARGET_I_VALUES = (0, 1, 2)


def _measure_apply_h(n: int, h_x: np.ndarray, repeat: int, warmup: int) -> list[float]:
    """`_rust.apply_h_kryanneal_py` の wall time を repeat 回計測して返す.

    warmup 回数だけ捨てた後, repeat 回の wall time を秒単位で返す.
    """
    from kryanneal import _rust  # pyright: ignore[reportMissingImports]

    dim = 1 << n
    rng = np.random.default_rng(0xBEEF_FACE ^ n)
    h_p_diag = rng.uniform(-1.0, 1.0, size=dim).astype(np.float64)
    v = (
        rng.uniform(-1.0, 1.0, size=dim) + 1j * rng.uniform(-1.0, 1.0, size=dim)
    ).astype(np.complex128)
    a_t = float(rng.uniform(-1.0, 1.0))
    b_t = float(rng.uniform(-1.0, 1.0))

    for _ in range(warmup):
        _ = _rust.apply_h_kryanneal_py(v, h_x, h_p_diag, a_t, b_t)

    timings: list[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        _ = _rust.apply_h_kryanneal_py(v, h_x, h_p_diag, a_t, b_t)
        t1 = time.perf_counter()
        timings.append(t1 - t0)
    return timings


def _measure_single_mode(n: int, i: int, repeat: int, warmup: int) -> list[float]:
    """`_rust.apply_single_mode_axis_i_py` の wall time を repeat 回計測.

    issue #71 (Phase 6 C2.5) acceptance 用: axis i に 2×2 ユニタリ U を
    apply する per-call wall time を測る. U は U(2) を phase + rotation の
    形で乱数生成 (Trotter R_i 形だと自由度が 1 つだけで FMA 量が少なく
    なるので一般 U(2) を取る).
    """
    from kryanneal import _rust  # pyright: ignore[reportMissingImports]

    dim = 1 << n
    rng = np.random.default_rng(0xC0FE_FACE ^ n ^ (i << 20))
    psi = (
        rng.uniform(-1.0, 1.0, size=dim) + 1j * rng.uniform(-1.0, 1.0, size=dim)
    ).astype(np.complex128)
    theta = float(rng.uniform(-np.pi, np.pi))
    alpha = float(rng.uniform(-np.pi, np.pi))
    beta = float(rng.uniform(-np.pi, np.pi))
    c = np.cos(theta)
    s = np.sin(theta)
    u = np.array(
        [
            np.exp(1j * alpha) * c,
            np.exp(1j * beta) * s,
            -np.exp(-1j * beta) * s,
            np.exp(-1j * alpha) * c,
        ],
        dtype=np.complex128,
    )

    for _ in range(warmup):
        _ = _rust.apply_single_mode_axis_i_py(psi, u, i, n)

    timings: list[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        _ = _rust.apply_single_mode_axis_i_py(psi, u, i, n)
        t1 = time.perf_counter()
        timings.append(t1 - t0)
    return timings


def _build_h_x(n: int, mode: str) -> np.ndarray:
    """h_x ベクトルを mode に応じて構築.

    - ``all-i``: 全要素 1.
    - ``i012-focus``: 先頭 3 要素のみ 1, 残りは 0 (i = 0, 1, 2 のみ寄与).
    """
    h_x = np.zeros(n, dtype=np.float64)
    if mode == "all-i":
        h_x[:] = 1.0
    elif mode == "i012-focus":
        # n < 3 だと i012-focus の意味が無いので呼び出し側で除外する.
        h_x[: min(3, n)] = 1.0
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return h_x


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


def mode_measure(args: argparse.Namespace) -> int:
    """1 build profile 分の measure を走らせて JSON に書き出す."""
    import kryanneal

    if args.blas_threads is not None:
        kryanneal.set_blas_threads(args.blas_threads)

    machine = _machine_info(args.label)

    trials: list[dict[str, Any]] = []
    for n in args.n_values:
        if n < 3:
            print(f"[measure] skip n={n} (n < 3, SIMD i=2 が踏めない)", flush=True)
            continue
        # apply_h_kryanneal_py sweep (issue #63 / Phase 6 C2).
        for mode in ("all-i", "i012-focus"):
            h_x = _build_h_x(n, mode)
            print(
                f"[measure] n={n} dim={1 << n} kernel=apply_h_kryanneal mode={mode}",
                flush=True,
            )
            timings = _measure_apply_h(n, h_x, repeat=args.repeat, warmup=args.warmup)
            for trial_idx, t in enumerate(timings):
                trials.append(
                    {
                        "label": args.label,
                        "n": n,
                        "dim": 1 << n,
                        "kernel": "apply_h_kryanneal",
                        "mode": mode,
                        "trial": trial_idx,
                        "wall_sec": t,
                    }
                )
            print(
                f"  median={statistics.median(timings):.6e}s, "
                f"min={min(timings):.6e}s, max={max(timings):.6e}s",
                flush=True,
            )

        # apply_single_mode_axis_i_py sweep (issue #71 / Phase 6 C2.5).
        # SIMD カーネルが特化する i ∈ {0,1,2} の per-axis time を個別に
        # 測る. issue acceptance「N=18, i=0,1,2 で per-pass time >=1.5×
        # 改善 (PR #70 baseline 比)」の判定に使う.
        for i in SIMD_TARGET_I_VALUES:
            if i >= n:
                continue
            print(
                f"[measure] n={n} dim={1 << n} kernel=apply_single_mode_axis_i i={i}",
                flush=True,
            )
            timings = _measure_single_mode(n, i, repeat=args.repeat, warmup=args.warmup)
            for trial_idx, t in enumerate(timings):
                trials.append(
                    {
                        "label": args.label,
                        "n": n,
                        "dim": 1 << n,
                        "kernel": "apply_single_mode_axis_i",
                        "mode": f"i{i}",
                        "trial": trial_idx,
                        "wall_sec": t,
                    }
                )
            print(
                f"  median={statistics.median(timings):.6e}s, "
                f"min={min(timings):.6e}s, max={max(timings):.6e}s",
                flush=True,
            )

    out: dict[str, Any] = {
        "machine_info": machine,
        "args": {
            "n_values": list(args.n_values),
            "repeat": args.repeat,
            "warmup": args.warmup,
            "blas_threads": args.blas_threads,
        },
        "trials": trials,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"[measure] wrote {out_path}")
    return 0


def _summarize_median(
    trials: list[dict[str, Any]],
) -> dict[tuple[int, str, str], float]:
    """(n, kernel, mode) ごとの median wall_sec を返す.

    issue #71 で `kernel` 次元を追加 (apply_h_kryanneal /
    apply_single_mode_axis_i). 旧 JSON (kernel 欠落) は
    `kernel = "apply_h_kryanneal"` として扱い後方互換.
    """
    buckets: dict[tuple[int, str, str], list[float]] = {}
    for r in trials:
        kernel = str(r.get("kernel", "apply_h_kryanneal"))
        buckets.setdefault((int(r["n"]), kernel, str(r["mode"])), []).append(
            float(r["wall_sec"])
        )
    return {key: statistics.median(vs) for key, vs in buckets.items()}


def mode_compare(args: argparse.Namespace) -> int:
    """SIMD ON / OFF の JSON を読んで speedup table を MD + CSV で出す."""
    on_data = json.loads(Path(args.simd_on).read_text(encoding="utf-8"))
    off_data = json.loads(Path(args.simd_off).read_text(encoding="utf-8"))

    # build flags の sanity check.
    on_info = on_data["machine_info"]
    off_info = off_data["machine_info"]
    if on_info.get("__has_simd__") is not True:
        print(
            f"[compare] WARNING: --simd-on file has __has_simd__={on_info.get('__has_simd__')!r} "
            "(expected True)",
            file=sys.stderr,
        )
    if off_info.get("__has_simd__") is not False:
        print(
            f"[compare] WARNING: --simd-off file has __has_simd__={off_info.get('__has_simd__')!r} "
            "(expected False)",
            file=sys.stderr,
        )

    on_medians = _summarize_median(on_data["trials"])
    off_medians = _summarize_median(off_data["trials"])
    keys = sorted(set(on_medians.keys()) | set(off_medians.keys()))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = out_dir / "bench_simd_scaling.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("n,dim,kernel,mode,simd_off_median_sec,simd_on_median_sec,speedup\n")
        for n, kernel, mode in keys:
            on = on_medians.get((n, kernel, mode))
            off = off_medians.get((n, kernel, mode))
            speedup = (off / on) if (on and off and on > 0) else None
            f.write(
                f"{n},{1 << n},{kernel},{mode},"
                f"{off if off is not None else ''},"
                f"{on if on is not None else ''},"
                f"{speedup if speedup is not None else ''}\n"
            )

    # MD
    md_path = out_dir / "bench_simd_scaling.md"
    lines: list[str] = []
    lines.append("# bench_simd_scaling (Phase 6 C2 / C2.5, issue #63 / #71)")
    lines.append("")
    lines.append(
        "`apply_h_kryanneal` (C2) と `apply_single_mode_axis_i` (C2.5) の "
        "per-pass time を SIMD ON/OFF で比較する."
    )
    lines.append("")
    lines.append("## Machine info (simd-on side)")
    lines.append("")
    for k, v in on_info.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")
    lines.append("## Machine info (simd-off side)")
    lines.append("")
    for k, v in off_info.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")
    lines.append("## Per-call median wall time and speedup")
    lines.append("")
    lines.append(
        "| n | dim | kernel | mode | simd-off median (ms) | "
        "simd-on median (ms) | speedup (off / on) |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for n, kernel, mode in keys:
        on = on_medians.get((n, kernel, mode))
        off = off_medians.get((n, kernel, mode))
        speedup = (off / on) if (on and off and on > 0) else float("nan")
        on_ms = (on * 1e3) if on is not None else float("nan")
        off_ms = (off * 1e3) if off is not None else float("nan")
        lines.append(
            f"| {n} | {1 << n} | {kernel} | {mode} | "
            f"{off_ms:.4f} | {on_ms:.4f} | {speedup:.2f}× |"
        )
    lines.append("")
    lines.append(
        "issue #63 acceptance: N=18 `apply_h_kryanneal` `i012-focus` で "
        "speedup ≥ 1.5× を満たすこと."
    )
    lines.append(
        "issue #71 acceptance: N=18 `apply_single_mode_axis_i` の i ∈ "
        "{0,1,2} 各々で speedup ≥ 1.5× を満たすこと."
    )
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[compare] wrote {csv_path}")
    print(f"[compare] wrote {md_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="mode", required=True)

    p_measure = sub.add_parser(
        "measure", help="1 build profile 分の measure を JSON 出力"
    )
    p_measure.add_argument(
        "--n-values",
        type=lambda s: [int(x) for x in s.split(",")],
        default=list(DEFAULT_N_VALUES),
        help=f"sweep する N (default {','.join(str(n) for n in DEFAULT_N_VALUES)})",
    )
    p_measure.add_argument("--repeat", type=int, default=DEFAULT_REPEAT)
    p_measure.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    p_measure.add_argument(
        "--blas-threads", type=int, default=1, help="BLAS thread 固定数 (default 1)"
    )
    p_measure.add_argument(
        "--label",
        type=str,
        required=True,
        choices=("simd-on", "simd-off"),
        help="build profile label (compare 側で識別に使う)",
    )
    p_measure.add_argument("--output", type=str, required=True, help="出力 JSON path")

    p_compare = sub.add_parser(
        "compare", help="SIMD ON / OFF JSON を統合して MD + CSV 出力"
    )
    p_compare.add_argument("--simd-on", type=str, required=True)
    p_compare.add_argument("--simd-off", type=str, required=True)
    p_compare.add_argument("--output-dir", type=str, required=True)

    # legacy --mode <name> も許容 (argparse subcommand を flag 互換に).
    args = p.parse_args(argv)

    if args.mode == "measure":
        return mode_measure(args)
    elif args.mode == "compare":
        return mode_compare(args)
    else:
        p.error(f"unknown mode {args.mode!r}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
