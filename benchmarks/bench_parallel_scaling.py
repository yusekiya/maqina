"""rayon parallel scaling sweep for matvec / Trotter primitives.

issue #62 (Phase 6 C1) の DoD「物理コア数 vs スループット sweep をベンチに
含める, メモリ帯域律速点を出力に明示」用. `apply_h_kryanneal` と
`apply_single_mode_axis_i` の per-call wall time を, スピン数
``N ∈ {16, 18, 20}`` × rayon thread 数 ``{1, 2, 4, 8, 16, 32, 64}`` で
sweep する.

## rayon thread 制御方法

rayon の global thread pool は **プロセス内で最初に rayon op が走った時点で
構築され, それ以降は変更できない**. したがって thread 数 sweep は
``RAYON_NUM_THREADS`` 環境変数を変えながら **subprocess を spawn し直す**
形でしか取れない (新規 Python API は追加しない方針, issue #62 参照).

本 script は 2 モードを持つ:

- **親モード (default)**: 各 ``(N, threads)`` cell につき本 script を
  ``--child --threads T --n N`` 引数 + ``RAYON_NUM_THREADS=T`` 環境変数で
  subprocess 起動し JSON 結果を集約する.
- **子モード (``--child``)**: rayon thread pool が起動環境変数通りに
  構築されることを期待し, 計測本体を実行して結果を stdout に JSON で
  print する.

## BLAS thread 干渉の固定

acceptance criteria: 「cpu_count=64 Linux サーバーで, BLAS thread=1 にして
測定」. 子モード冒頭で ``kryanneal.set_blas_threads(1)`` を呼び, BLAS pool
を 1 thread に固定して rayon の thread 数効果を分離する (`docs/design.md`
§12 Phase 6, ``CLAUDE.md`` 「Thread pool 運用」節).

## 実行ハードウェア

ベンチ本番 sweep は **Phase 4/5 と同じ Linux サーバー (x86_64 / glibc 2.35 /
cpu_count=64 / OpenBLAS)** 上で実行する (``CLAUDE.md`` ベンチマーク節
「同一マシン上の before / after」原則). macOS では smoke (``--threads-list
1,2``) のみ. issue #47 で確定した「bench は PR 本体に含めず merge 後コメント
添付運用」に従い, 本 script の追加自体は PR に含めるが本番 sweep の結果
artefact は merge 後 issue コメントとして付与する.

## CLI 例

::

    # 親モード default (本番 sweep)
    uv run python benchmarks/bench_parallel_scaling.py

    # 親モード smoke (macOS)
    uv run python benchmarks/bench_parallel_scaling.py \\
        --n-values 16 --threads-list 1,2 --repeat 3

    # 親モード, 出力先指定
    uv run python benchmarks/bench_parallel_scaling.py \\
        --results-dir benchmarks/results/20260517-rayon-scaling/

## 出力

``benchmarks/results/<YYYYMMDD-HHMMSS>/`` 配下に:

- ``bench_parallel_scaling.csv``: 全 trial 生データ (n, dim, threads, kernel,
  trial, wall_sec, calls_per_sec). ``kernel`` は ``apply_h_kryanneal`` と
  ``apply_single_mode_axis_i_sum`` (= n サイト全 i を 1 回ずつ apply する
  Trotter step 相当の合計).
- ``bench_parallel_scaling.md``: 集計表 (n × threads 行列の median wall_sec /
  speedup vs threads=1 / efficiency) と **メモリ帯域律速点** (thread 数を
  増やしても rate が伸びなくなる knee = 連続 2 点で speedup 改善 < 5% の
  最初の thread 数) を機械情報と合わせて記録.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

DEFAULT_N_VALUES = (16, 18, 20)
DEFAULT_THREADS_LIST = (1, 2, 4, 8, 16, 32, 64)
DEFAULT_REPEAT = 5
DEFAULT_WARMUP = 1
KNEE_IMPROVEMENT_THRESHOLD = 0.05  # 5% 未満の伸びを knee と判定


@dataclasses.dataclass
class TrialResult:
    n: int
    dim: int
    threads: int
    kernel: str
    trial: int
    wall_sec: float

    @property
    def calls_per_sec(self) -> float:
        return 1.0 / self.wall_sec if self.wall_sec > 0 else float("inf")


def child_run(
    *,
    n: int,
    threads: int,
    repeat: int,
    warmup: int,
) -> list[TrialResult]:
    """子モード本体. 単一 ``(n, threads)`` セルを計測して結果を返す.

    ``RAYON_NUM_THREADS`` は親プロセス側で set 済み. ここでは確認のみ.
    """
    # 必ず Rust 拡張をロード (BLAS 設定込み).
    import kryanneal
    from kryanneal import _rust  # pyright: ignore[reportMissingImports]

    # BLAS pool を 1 thread に固定 (acceptance criteria).
    kryanneal.set_blas_threads(1)

    dim = 1 << n
    rng = np.random.default_rng(0xBEEF_FACE ^ n)
    h_x = rng.uniform(-1.0, 1.0, size=n).astype(np.float64)
    h_p_diag = rng.uniform(-1.0, 1.0, size=dim).astype(np.float64)
    v = (
        rng.uniform(-1.0, 1.0, size=dim) + 1j * rng.uniform(-1.0, 1.0, size=dim)
    ).astype(np.complex128)
    psi = v.copy()
    a_t, b_t = float(rng.uniform(-1.0, 1.0)), float(rng.uniform(-1.0, 1.0))

    # 2×2 unitary (ランダム位相付き cos/sin).
    theta = float(rng.uniform(-np.pi, np.pi))
    c, s = np.cos(theta), np.sin(theta)
    u = np.array([c, 1j * s, 1j * s, c], dtype=np.complex128)

    results: list[TrialResult] = []

    # ---- apply_h_kryanneal ----
    # warm up
    for _ in range(warmup):
        _ = _rust.apply_h_kryanneal_py(v, h_x, h_p_diag, a_t, b_t)
    for trial in range(repeat):
        t0 = time.perf_counter()
        _ = _rust.apply_h_kryanneal_py(v, h_x, h_p_diag, a_t, b_t)
        t1 = time.perf_counter()
        results.append(
            TrialResult(
                n=n,
                dim=dim,
                threads=threads,
                kernel="apply_h_kryanneal",
                trial=trial,
                wall_sec=t1 - t0,
            )
        )

    # ---- apply_single_mode_axis_i, 全 i サイト合計 (Trotter step 相当) ----
    # warm up
    for _ in range(warmup):
        psi_warm = psi.copy()
        for i in range(n):
            psi_warm = _rust.apply_single_mode_axis_i_py(psi_warm, u, i, n)
    for trial in range(repeat):
        psi_run = psi.copy()
        t0 = time.perf_counter()
        for i in range(n):
            psi_run = _rust.apply_single_mode_axis_i_py(psi_run, u, i, n)
        t1 = time.perf_counter()
        results.append(
            TrialResult(
                n=n,
                dim=dim,
                threads=threads,
                kernel="apply_single_mode_axis_i_sum",
                trial=trial,
                wall_sec=t1 - t0,
            )
        )

    return results


def _machine_info() -> dict:
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "cpu_count_logical": os.cpu_count(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        import kryanneal
        from kryanneal import _rust  # pyright: ignore[reportMissingImports]

        info["kryanneal_version"] = getattr(kryanneal, "__version__", "unknown")
        info["has_blas"] = bool(getattr(_rust, "__has_blas__", False))
    except ImportError:
        info["kryanneal_version"] = "import-failed"
        info["has_blas"] = None
    return info


def parent_run(
    *,
    n_values: Sequence[int],
    threads_list: Sequence[int],
    repeat: int,
    warmup: int,
    results_dir: Path,
) -> None:
    """親モード: 各 ``(n, threads)`` cell につき本 script を subprocess
    として起動し JSON 集約 → CSV + md を書き出す."""
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[TrialResult] = []
    for n in n_values:
        for threads in threads_list:
            env = os.environ.copy()
            env["RAYON_NUM_THREADS"] = str(threads)
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--child",
                "--n",
                str(n),
                "--threads",
                str(threads),
                "--repeat",
                str(repeat),
                "--warmup",
                str(warmup),
            ]
            print(
                f"[parent] subprocess: n={n} threads={threads} "
                f"(RAYON_NUM_THREADS={threads})",
                flush=True,
            )
            proc = subprocess.run(
                cmd,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(proc.stdout)
            for r in payload["trials"]:
                all_results.append(
                    TrialResult(
                        n=r["n"],
                        dim=r["dim"],
                        threads=r["threads"],
                        kernel=r["kernel"],
                        trial=r["trial"],
                        wall_sec=r["wall_sec"],
                    )
                )

    csv_path = results_dir / "bench_parallel_scaling.csv"
    md_path = results_dir / "bench_parallel_scaling.md"
    _write_csv(csv_path, all_results)
    _write_md(md_path, all_results, _machine_info())
    print(f"[parent] CSV : {csv_path}")
    print(f"[parent] MD  : {md_path}")


def _write_csv(path: Path, results: list[TrialResult]) -> None:
    with path.open("w") as f:
        f.write("n,dim,threads,kernel,trial,wall_sec,calls_per_sec\n")
        for r in results:
            f.write(
                f"{r.n},{r.dim},{r.threads},{r.kernel},{r.trial},"
                f"{r.wall_sec:.9g},{r.calls_per_sec:.9g}\n"
            )


def _write_md(path: Path, results: list[TrialResult], machine: dict) -> None:
    # (n, threads, kernel) → list of wall_sec
    buckets: dict[tuple[int, int, str], list[float]] = {}
    for r in results:
        buckets.setdefault((r.n, r.threads, r.kernel), []).append(r.wall_sec)
    n_values = sorted({k[0] for k in buckets})
    threads_values = sorted({k[1] for k in buckets})
    kernels = sorted({k[2] for k in buckets})

    lines: list[str] = []
    lines.append("# bench_parallel_scaling")
    lines.append("")
    lines.append("issue #62 (Phase 6 C1): rayon parallel scaling sweep.")
    lines.append("")
    lines.append("## Machine info")
    lines.append("")
    for k, v in machine.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")

    for kernel in kernels:
        lines.append(f"## {kernel}")
        lines.append("")
        # median wall_sec 表 (n 行 × threads 列).
        header = "| n \\ threads |" + "".join(f" {t} |" for t in threads_values)
        sep = "|---|" + "---|" * len(threads_values)
        lines.append(header)
        lines.append(sep)
        for n in n_values:
            row = [f"| {n}"]
            for t in threads_values:
                vals = buckets.get((n, t, kernel))
                if not vals:
                    row.append("—")
                else:
                    med = float(np.median(vals))
                    row.append(f"{med * 1e3:.3f} ms")
            lines.append(" | ".join(row) + " |")
        lines.append("")

        # Speedup 表 (vs threads=1).
        lines.append(f"### {kernel} — speedup vs threads=1 (median)")
        lines.append("")
        lines.append(header)
        lines.append(sep)
        for n in n_values:
            row = [f"| {n}"]
            base = buckets.get((n, threads_values[0], kernel))
            base_med = float(np.median(base)) if base else float("nan")
            for t in threads_values:
                vals = buckets.get((n, t, kernel))
                if not vals or not np.isfinite(base_med):
                    row.append("—")
                else:
                    med = float(np.median(vals))
                    speedup = base_med / med if med > 0 else float("inf")
                    row.append(f"{speedup:.2f}×")
            lines.append(" | ".join(row) + " |")
        lines.append("")

        # メモリ帯域律速点 (knee): 連続 2 thread 間で speedup の伸びが
        # KNEE_IMPROVEMENT_THRESHOLD 未満になる最初の thread 数.
        lines.append(
            f"### {kernel} — knee (memory-bandwidth saturation point, "
            f"speedup improvement < {KNEE_IMPROVEMENT_THRESHOLD * 100:.0f}%)"
        )
        lines.append("")
        lines.append("| n | knee threads | speedup at knee |")
        lines.append("|---|---|---|")
        for n in n_values:
            base = buckets.get((n, threads_values[0], kernel))
            base_med = float(np.median(base)) if base else float("nan")
            prev_speedup = 1.0
            knee_threads: int | None = None
            knee_speedup: float = float("nan")
            for t in threads_values[1:]:
                vals = buckets.get((n, t, kernel))
                if not vals or not np.isfinite(base_med):
                    continue
                med = float(np.median(vals))
                speedup = base_med / med if med > 0 else float("inf")
                rel_improve = (speedup - prev_speedup) / max(prev_speedup, 1e-12)
                if rel_improve < KNEE_IMPROVEMENT_THRESHOLD:
                    knee_threads = t
                    knee_speedup = speedup
                    break
                prev_speedup = speedup
            if knee_threads is None:
                lines.append(f"| {n} | > {threads_values[-1]} (未到達) | — |")
            else:
                lines.append(f"| {n} | {knee_threads} | {knee_speedup:.2f}× |")
        lines.append("")

    path.write_text("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--child", action="store_true", help="(internal) child mode")
    p.add_argument("--n", type=int, help="(child mode) スピン数 N")
    p.add_argument("--threads", type=int, help="(child mode) rayon thread 数")
    p.add_argument(
        "--n-values",
        type=str,
        default=",".join(str(n) for n in DEFAULT_N_VALUES),
        help=f"sweep する N (default '{','.join(str(n) for n in DEFAULT_N_VALUES)}')",
    )
    p.add_argument(
        "--threads-list",
        type=str,
        default=",".join(str(t) for t in DEFAULT_THREADS_LIST),
        help=(
            "sweep する thread 数 (default "
            f"'{','.join(str(t) for t in DEFAULT_THREADS_LIST)}')"
        ),
    )
    p.add_argument("--repeat", type=int, default=DEFAULT_REPEAT)
    p.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    p.add_argument("--results-dir", type=str, default=None)
    args = p.parse_args()

    if args.child:
        if args.n is None or args.threads is None:
            p.error("--child requires --n and --threads")
        trials = child_run(
            n=args.n,
            threads=args.threads,
            repeat=args.repeat,
            warmup=args.warmup,
        )
        payload = {
            "trials": [dataclasses.asdict(t) for t in trials],
            "rayon_num_threads_env": os.environ.get("RAYON_NUM_THREADS"),
        }
        print(json.dumps(payload))
        return

    if args.results_dir is None:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        results_dir = Path(__file__).resolve().parent / "results" / ts
    else:
        results_dir = Path(args.results_dir)

    n_values = [int(x) for x in args.n_values.split(",") if x.strip()]
    threads_list = [int(x) for x in args.threads_list.split(",") if x.strip()]
    parent_run(
        n_values=n_values,
        threads_list=threads_list,
        repeat=args.repeat,
        warmup=args.warmup,
        results_dir=results_dir,
    )


if __name__ == "__main__":
    main()
