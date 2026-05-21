"""README 用 fidelity-vs-runtime 散布図の **計測** スクリプト.

`build_readme_problem.py` と `compute_readme_reference.py` で **事前生成済**
の npz (問題ファイル + 参照解ファイル) を読み, kryanneal
``cfm4_adaptive_richardson`` と QuTiP ``sesolve`` (Adams, sparse) の精度
パラメータ sweep を回して各 cell の wall time + infidelity を CSV に dump
する. 描画は `plot_readme_figure.py` が担当 (本 script は描画しない).

途中測定なし (``save_tlist=None``, kryanneal 最節約モード) で最終状態
``ψ(T)`` のみ取得し参照解との infidelity ``1 - |<ψ_ref|ψ>|^2`` を測る.

## 出力 CSV

``<output_dir>/bench_<scenario>.csv``:

```
scenario,n,T,seed,solver,knob_name,knob_value,wall_sec,infidelity,n_steps_eff
non-stiff,18,10000.0,20260518,kryanneal,atol,1e-05,1234.5,4.218e-09,8766
non-stiff,18,10000.0,20260518,qutip,tol,1e-05,9876.3,1.234e-06,
...
```

`plot_readme_figure.py` がこれを読んで scatter plot を作る.

## 中断耐性 + 既存 cell スキップ

各 cell (1 つの atol または tol) 完了ごとに **CSV を atomic に書き直す**
(tmp + os.replace) ため, 途中で kill / 電源断 / SSH 切断が起きても **既に
完了した cell の結果は保持** される. 起動時に既存の CSV を読んでロード済の
``(solver, knob_value)`` を skip するため, 同じ引数で再起動するだけで残りの
cell から再開できる.

## `--solver` フラグによる kryanneal / QuTiP 分離実行 (戦略 B)

kryanneal cell は memory bandwidth bound, QuTiP cell は sparse 経路で
single-thread CPU bound と支配的リソースが異なる. 2 scenario を同時並列
実行すると kryanneal cells は DRAM 帯域を取り合って互いに遅くなるため,
**scenario 単位は順次 + 最後に QuTiP cells だけ 2 scenario 並列実行**
する戦略 (= 戦略 B) が時間と正確さの両立に向く. これを実現するため
``--solver {both, kryanneal, qutip}`` で実行対象 cells を選択可能.

典型運用 (本番 N=18, T=10^4):

```bash
# Step 1: 全 scenario の kryanneal cells を順次 (DRAM 帯域競合避ける)
uv run python -m benchmarks.bench_readme_figure --solver kryanneal \\
    --problem-file   benchmarks/data/problem_non-stiff_n18_seed20260518.npz \\
    --reference-file benchmarks/data/reference_non-stiff_n18_T10000_seed20260518.npz \\
    --output-dir     benchmarks/data/0.8.0/
uv run python -m benchmarks.bench_readme_figure --solver kryanneal \\
    --problem-file   benchmarks/data/problem_stiff_n18_seed20260518.npz \\
    --reference-file benchmarks/data/reference_stiff_n18_T10000_seed20260518.npz \\
    --output-dir     benchmarks/data/0.8.0/

# Step 2: 両 scenario の QuTiP cells を並列実行 (sparse は single-thread なので競合せず)
uv run python -m benchmarks.bench_readme_figure --solver qutip \\
    --problem-file   benchmarks/data/problem_non-stiff_n18_seed20260518.npz \\
    --reference-file benchmarks/data/reference_non-stiff_n18_T10000_seed20260518.npz \\
    --output-dir     benchmarks/data/0.8.0/ &
uv run python -m benchmarks.bench_readme_figure --solver qutip \\
    --problem-file   benchmarks/data/problem_stiff_n18_seed20260518.npz \\
    --reference-file benchmarks/data/reference_stiff_n18_T10000_seed20260518.npz \\
    --output-dir     benchmarks/data/0.8.0/ &
wait
```

各 process は同じ ``bench_<scenario>.csv`` に append するが,
**Step 2 の 2 process は scenario が異なる = CSV ファイルも異なる** ため
file lock 不要. Step 1 は順次なので競合なし.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np

from benchmarks._readme_figure_helpers import (
    build_qutip_hamiltonian,
    infidelity,
    run_qutip,
)
from kryanneal import IsingProblem, QuantumAnnealer, Schedule
from kryanneal.initial_states import uniform_superposition

# CSV のカラム順. plot_readme_figure.py と同期.
CSV_FIELDNAMES = [
    "scenario",
    "n",
    "T",
    "seed",
    "solver",
    "knob_name",
    "knob_value",
    "wall_sec",
    "infidelity",
    "n_steps_eff",
]

# float の done-set キーを正規化するためのフォーマット. CLI から渡された
# 値と CSV から読んだ文字列を相互変換しても一致するよう scientific 6 桁.
_KNOB_FMT = "{:.6e}"


def _normalize_knob(value: float) -> str:
    """sweep 値を ``done_cells`` の比較に使う正規化文字列に変換する."""
    return _KNOB_FMT.format(float(value))


def _run_kryanneal_adaptive(
    prob: IsingProblem,
    sched: Schedule,
    psi0: np.ndarray,
    T: float,
    atol: float,
) -> tuple[float, np.ndarray, int]:
    """``cfm4_adaptive_richardson`` を ``atol`` で 1 回走らせ wall_sec / ψ_final / n_steps_actual."""
    ann = QuantumAnnealer(prob, sched)
    t_start = time.perf_counter()
    res = ann.run(
        psi0,
        0.0,
        T,
        method="cfm4_adaptive_richardson",
        atol=float(atol),
    )
    elapsed = time.perf_counter() - t_start
    n_steps_actual = res.n_steps_actual if res.n_steps_actual is not None else -1
    return elapsed, np.ascontiguousarray(res.psi_final), n_steps_actual


def _parse_floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def _load_existing(
    csv_path: Path,
) -> tuple[list[dict[str, object]], set[tuple[str, str]]]:
    """既存 CSV から ``(rows, done_cells)`` を作る. なければ空.

    ``done_cells`` は ``(solver, normalized_knob_str)`` の set.
    """
    if not csv_path.exists():
        return [], set()
    rows: list[dict[str, object]] = []
    done: set[tuple[str, str]] = set()
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(dict(r))
            try:
                knob_norm = _normalize_knob(float(r["knob_value"]))
            except (KeyError, ValueError):
                continue
            done.add((str(r["solver"]), knob_norm))
    return rows, done


def _save_csv_atomic(csv_path: Path, rows: list[dict[str, object]]) -> None:
    """``csv_path`` を tmp + os.replace で atomic に書き直す.

    1 cell ごとに呼び出す前提. 途中で kill されても tmp は残るが,
    csv_path 自体は前回 atomic save の状態を保つ.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_path, csv_path)


def run_bench(
    *,
    problem_file: Path,
    reference_file: Path,
    kryanneal_atols: Sequence[float],
    qutip_tols: Sequence[float],
    output_dir: Path,
    solver: Literal["both", "kryanneal", "qutip"] = "both",
) -> Path:
    # 問題ファイル
    pdata = np.load(problem_file)
    h_p_diag = np.asarray(pdata["H_p_diag"], dtype=np.float64)
    h_x = np.asarray(pdata["h_x"], dtype=np.float64)
    n = int(pdata["n"])
    seed = int(pdata["seed"])
    scenario = str(pdata["scenario"])

    # 参照解ファイル
    rdata = np.load(reference_file)
    psi_ref = np.asarray(rdata["psi_ref"], dtype=np.complex128)
    T = float(rdata["T"])
    ref_scenario = str(rdata["scenario"])
    ref_n = int(rdata["n"])
    ref_seed = int(rdata["seed"])
    converged = bool(rdata["converged"])
    solver_independent = bool(rdata["solver_independent"])

    # consistency check: problem と reference が同じ問題を指していること
    if (scenario, n, seed) != (ref_scenario, ref_n, ref_seed):
        raise ValueError(
            "problem_file と reference_file の (scenario, n, seed) が不一致:\n"
            f"  problem:   ({scenario}, {n}, {seed})\n"
            f"  reference: ({ref_scenario}, {ref_n}, {ref_seed})"
        )
    if psi_ref.shape != (1 << n,):
        raise ValueError(
            f"psi_ref shape {psi_ref.shape} does not match n={n} (expected {1 << n})"
        )

    if not converged:
        print(
            "WARNING: 参照解の Adams 収束 flag が False です. infidelity の解釈には注意.",
            flush=True,
        )
    if not solver_independent:
        print(
            "WARNING: 参照解の Adams vs BDF 一致 flag が False です. infidelity の解釈には注意.",
            flush=True,
        )

    # 問題セットアップ
    prob = IsingProblem(n=n, H_p_diag=h_p_diag, h_x=h_x)
    sched = Schedule.linear(T=T)
    psi0 = uniform_superposition(n)
    h_t = build_qutip_hamiltonian(h_x, h_p_diag, T)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"bench_{scenario}.csv"

    # 既存 CSV をロードして skip-existing に使う
    rows, done_cells = _load_existing(csv_path)

    print(
        f"\n=== bench: scenario={scenario}, n={n}, T={T:.0f}, seed={seed} "
        f"(solver={solver}) ===",
        flush=True,
    )
    print(f"problem:   {problem_file}", flush=True)
    print(f"reference: {reference_file}", flush=True)
    print(f"csv:       {csv_path}", flush=True)
    if rows:
        print(
            f"[resume] {csv_path} に既存 {len(rows)} 行. "
            f"同 (solver, knob_value) cell は skip して残りから再開する.",
            flush=True,
        )

    # ---- kryanneal cells ----
    if solver in ("both", "kryanneal"):
        for atol in kryanneal_atols:
            knob_key = ("kryanneal", _normalize_knob(atol))
            if knob_key in done_cells:
                print(
                    f"[skip] kryanneal atol={atol:.0e} (既存 cell, CSV に保存済み)",
                    flush=True,
                )
                continue
            print(f"\n[kryanneal] atol={atol:.0e} running ...", flush=True)
            wall, psi, n_steps = _run_kryanneal_adaptive(prob, sched, psi0, T, atol)
            inf = infidelity(psi, psi_ref)
            print(
                f"[kryanneal] atol={atol:.0e}: wall={wall:.2f}s, "
                f"infidelity={inf:.3e}, n_steps={n_steps}",
                flush=True,
            )
            rows.append(
                {
                    "scenario": scenario,
                    "n": n,
                    "T": T,
                    "seed": seed,
                    "solver": "kryanneal",
                    "knob_name": "atol",
                    "knob_value": atol,
                    "wall_sec": wall,
                    "infidelity": inf,
                    "n_steps_eff": n_steps,
                }
            )
            done_cells.add(knob_key)
            _save_csv_atomic(csv_path, rows)
            print(
                f"[saved] {csv_path} ({len(rows)} cells total)",
                flush=True,
            )

    # ---- QuTiP cells ----
    if solver in ("both", "qutip"):
        for tol in qutip_tols:
            knob_key = ("qutip", _normalize_knob(tol))
            if knob_key in done_cells:
                print(
                    f"[skip] qutip tol={tol:.0e} (既存 cell, CSV に保存済み)",
                    flush=True,
                )
                continue
            print(f"\n[qutip] tol={tol:.0e} running (Adams) ...", flush=True)
            wall, psi = run_qutip(h_t, psi0, T, n, tol, method="adams")
            inf = infidelity(psi, psi_ref)
            print(
                f"[qutip] tol={tol:.0e}: wall={wall:.2f}s, infidelity={inf:.3e}",
                flush=True,
            )
            rows.append(
                {
                    "scenario": scenario,
                    "n": n,
                    "T": T,
                    "seed": seed,
                    "solver": "qutip",
                    "knob_name": "tol",
                    "knob_value": tol,
                    "wall_sec": wall,
                    "infidelity": inf,
                    "n_steps_eff": "",
                }
            )
            done_cells.add(knob_key)
            _save_csv_atomic(csv_path, rows)
            print(
                f"[saved] {csv_path} ({len(rows)} cells total)",
                flush=True,
            )

    print(f"\n[done] {csv_path} (solver={solver}, {len(rows)} cells)", flush=True)
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--problem-file",
        type=Path,
        required=True,
        help="build_readme_problem.py が生成した問題 npz",
    )
    parser.add_argument(
        "--reference-file",
        type=Path,
        required=True,
        help="compute_readme_reference.py が生成した参照解 npz (T はここから取得)",
    )
    parser.add_argument(
        "--kryanneal-atols",
        type=_parse_floats,
        default=[1e-3, 1e-5, 1e-7, 1e-9],
        help="cfm4_adaptive_richardson atol sweep (CSV)",
    )
    parser.add_argument(
        "--qutip-tols",
        type=_parse_floats,
        default=[1e-3, 1e-5, 1e-7, 1e-9],
        help="QuTiP sesolve (Adams) tol sweep (CSV)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/data"),
        help="CSV 出力先 (default: benchmarks/data/). 本番は "
        "benchmarks/data/<X.Y.Z>/ を明示指定する想定",
    )
    parser.add_argument(
        "--solver",
        choices=["both", "kryanneal", "qutip"],
        default="both",
        help="実行する solver の cells を選択する. "
        "both=両方 (default), kryanneal=kryanneal cells のみ, qutip=QuTiP cells のみ. "
        "戦略 B (scenario 順次 + QuTiP cells 2 scenario 並列実行) で使う.",
    )
    parser.add_argument(
        "--blas-threads",
        type=int,
        default=1,
        help="全 BLAS pool (numpy bundled + system OpenBLAS) の thread 数. "
        "default 1 = OpenBLAS の spin wait による CPU 浪費を回避. "
        "QuTiP cells を 2 scenario 並列実行 (戦略 B Step 2) するとき, "
        "両 process が default thread 数 (=物理コア数) で spawn すると spin wait "
        "で互いに競合して計算時間が伸びるのを防ぐ. memory "
        "`project_bench_machine` の確立済運用と整合 (kryanneal cell 中の rayon と "
        'OpenBLAS pool の競合回避も兼ねる, CLAUDE.md "Thread pool 運用" 節参照).',
    )
    args = parser.parse_args()

    # set_blas_threads は kryanneal._init_ で export 済の API. threadpoolctl 経由で
    # numpy bundled + system OpenBLAS の thread 数を一括制御. rayon pool には
    # 影響しないので kryanneal cell の matvec 並列化は維持される.
    import kryanneal as _kryanneal  # noqa: PLC0415  (CLI 引数解決後の呼び出し)

    _kryanneal.set_blas_threads(args.blas_threads)
    print(
        f"[config] BLAS threads = {args.blas_threads} "
        f"(spin wait / rayon×BLAS 競合回避のため)",
        flush=True,
    )

    run_bench(
        problem_file=args.problem_file,
        reference_file=args.reference_file,
        kryanneal_atols=args.kryanneal_atols,
        qutip_tols=args.qutip_tols,
        output_dir=args.output_dir,
        solver=args.solver,
    )


if __name__ == "__main__":
    main()
