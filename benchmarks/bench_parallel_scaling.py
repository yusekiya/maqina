"""rayon parallel scaling sweep for matvec / Trotter primitives.

issue #62 (Phase 6 C1) の DoD「物理コア数 vs スループット sweep をベンチに
含める, メモリ帯域律速点を出力に明示」用. `apply_h` と
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
測定」. 子モード冒頭で ``maqina.set_blas_threads(1)`` を呼び, BLAS pool
を 1 thread に固定して rayon の thread 数効果を分離する (`docs/design/12-release-plan.md`
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
  trial, wall_sec, calls_per_sec). ``kernel`` は次の 3 種:
    - ``apply_h``: matvec primitive 単発 (Lanczos / CFM4:2 の
      ホットパスを 1 call 単位で計測).
    - ``trotter_step``: ``_rust.trotter_step_py`` 1 回 = full Strang Trotter
      step (phase pass + 全 i bit-flip pass + phase pass). 本番ホットパス
      (`trotter_step` を loop で叩く path) の per-step cost を最も忠実に表す
      (issue #68).
    - ``apply_single_mode_axis_i_py_sum_diagnostic``: Python 側で
      ``for i in range(n): _rust.apply_single_mode_axis_i_py(...)`` を回した
      合計時間. **本番ホットパスではなく diagnostic 用** (Python wrap の
      to_vec allocation overhead 検出, issue #68). per-i wrap allocation が
      支配するため rayon scaling は見えない. 実際のホットパスは
      ``trotter_step`` 列を見ること.
- ``bench_parallel_scaling.md``: 集計表 (n × threads 行列の median wall_sec /
  speedup vs threads=1 / max speedup / efficiency) と **メモリ帯域律速点**
  (knee = thread 数を増やしても speedup が max の 95% を維持する最小 thread
  数, smoothing + plateau detection で regression に強いヒューリスティック)
  を機械情報と合わせて記録 (issue #68 で前点比版から置換).
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
# knee = max_speedup の 95% を維持する最小 thread 数 (smoothing + plateau
# detection). 旧 "前点比 < 5%" 版は regression (例: 1→2 threads で 0.57×)
# を knee と誤判定する問題があったため issue #68 で置換.
KNEE_PLATEAU_TOLERANCE = 0.05  # max の 5% 下まで plateau とみなす


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
    import maqina
    from maqina import _rust  # pyright: ignore[reportMissingImports]

    # BLAS pool を 1 thread に固定 (acceptance criteria).
    maqina.set_blas_threads(1)

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

    # ---- apply_h ----
    # in-place 版を使い ``y_out`` を warmup 前に 1 回 alloc して再利用する
    # (issue #85). 旧 ``apply_h_py`` だと毎 call で alloc/copy が
    # 計測域に混入し rayon scaling 評価を歪める.
    y_out = np.empty(dim, dtype=np.complex128)
    # warm up
    for _ in range(warmup):
        _rust.apply_h_into_py(v, y_out, h_x, h_p_diag, a_t, b_t)
    for trial in range(repeat):
        t0 = time.perf_counter()
        _rust.apply_h_into_py(v, y_out, h_x, h_p_diag, a_t, b_t)
        t1 = time.perf_counter()
        results.append(
            TrialResult(
                n=n,
                dim=dim,
                threads=threads,
                kernel="apply_h",
                trial=trial,
                wall_sec=t1 - t0,
            )
        )

    # ---- trotter_step (full Strang step, 本番ホットパス) ----
    # `_rust.trotter_step_inplace_py` 1 回 = phase pass + 全 i bit-flip pass +
    # phase pass を Rust 内部で完走し ``psi`` を in-place 更新する. Python 越境
    # は 1 回だけなので per-i wrap allocation overhead が混入せず, rayon scaling
    # が公平に見える (issue #68 / in-place 化 #86).
    dt = 0.01  # 任意の小値. 計測は wall time だけで数値正確性は問わない.
    psi_trotter = psi.copy()  # 計測内 mutation 用の owned buffer.
    # warm up
    for _ in range(warmup):
        _rust.trotter_step_inplace_py(psi_trotter, h_x, h_p_diag, a_t, b_t, dt, n)
    for trial in range(repeat):
        t0 = time.perf_counter()
        _rust.trotter_step_inplace_py(psi_trotter, h_x, h_p_diag, a_t, b_t, dt, n)
        t1 = time.perf_counter()
        results.append(
            TrialResult(
                n=n,
                dim=dim,
                threads=threads,
                kernel="trotter_step",
                trial=trial,
                wall_sec=t1 - t0,
            )
        )

    # ---- apply_single_mode_axis_i_py_sum_diagnostic (本番ホットパスではない) ----
    # Python 側で per-i 呼び出しを n 回ループする. in-place 版 (#86) を採用後は
    # **Python boundary 越え × n + Rust 内部 single-axis kernel** の合計を見る
    # 形になる (旧運用は wrap allocation overhead 計測が主目的だったが, それを
    # 排した状態で本番 ``trotter_step`` 1 call との差分が「Python loop driver
    # 自体のオーバヘッド (+ Rust 内部 multi-qubit gate fusion が消える分)」と
    # 解釈できる). kernel 名は bench history continuity のため維持.
    psi_per_i = psi.copy()  # 計測内 mutation 用の owned buffer.
    # warm up
    for _ in range(warmup):
        for i in range(n):
            _rust.apply_single_mode_axis_i_inplace_py(psi_per_i, u, i, n)
    for trial in range(repeat):
        t0 = time.perf_counter()
        for i in range(n):
            _rust.apply_single_mode_axis_i_inplace_py(psi_per_i, u, i, n)
        t1 = time.perf_counter()
        results.append(
            TrialResult(
                n=n,
                dim=dim,
                threads=threads,
                kernel="apply_single_mode_axis_i_py_sum_diagnostic",
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
        import maqina
        from maqina import _rust  # pyright: ignore[reportMissingImports]

        info["maqina_version"] = getattr(maqina, "__version__", "unknown")
        info["has_blas"] = bool(getattr(_rust, "__has_blas__", False))
    except ImportError:
        info["maqina_version"] = "import-failed"
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

        # メモリ帯域律速点 (knee): max_speedup の 95% を維持する最小 thread.
        # smoothing + plateau detection で regression に強いヒューリスティック
        # (issue #68 で前点比 < 5% 版から置換).
        lines.append(
            f"### {kernel} — knee (memory-bandwidth saturation point, "
            f"smallest threads achieving ≥ {(1 - KNEE_PLATEAU_TOLERANCE) * 100:.0f}% of max speedup)"
        )
        lines.append("")
        lines.append(
            "| n | max speedup | max @ threads | knee threads | speedup at knee |"
        )
        lines.append("|---|---|---|---|---|")
        for n in n_values:
            # 全 thread 数の (t, speedup) を集める. base = threads=1 の median.
            base = buckets.get((n, threads_values[0], kernel))
            base_med = float(np.median(base)) if base else float("nan")
            speedups: list[tuple[int, float]] = []
            for t in threads_values:
                vals = buckets.get((n, t, kernel))
                if not vals or not np.isfinite(base_med):
                    continue
                med = float(np.median(vals))
                if med <= 0:
                    continue
                speedups.append((t, base_med / med))
            if not speedups:
                lines.append(f"| {n} | — | — | — | — |")
                continue
            max_speedup = max(sp for _, sp in speedups)
            # max を最初に達成した thread 数 (同じ max を複数の t で取った場合).
            max_threads = min(t for t, sp in speedups if sp >= max_speedup - 1e-12)
            # plateau = max の (1 - tolerance) 以上を維持する最小 thread.
            plateau_floor = max_speedup * (1 - KNEE_PLATEAU_TOLERANCE)
            knee_threads = min(t for t, sp in speedups if sp >= plateau_floor)
            # knee における speedup (knee_threads の速度).
            knee_speedup = next(sp for t, sp in speedups if t == knee_threads)
            lines.append(
                f"| {n} | {max_speedup:.2f}× | {max_threads} | "
                f"{knee_threads} | {knee_speedup:.2f}× |"
            )
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
