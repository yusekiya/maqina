"""README 用 fidelity-vs-runtime 散布図の **計測** スクリプト.

`build_readme_problem.py` と `compute_readme_reference.py` で **事前生成済**
の npz (問題ファイル + 参照解ファイル) を読み, kryanneal
``cfm4_adaptive_richardson`` と QuTiP ``sesolve`` (Adams, sparse) の精度
パラメータ sweep を回して各 cell の wall time + infidelity を CSV に dump
する. 描画は `plot_readme_figure.py` が担当 (本 script は描画しない).

途中測定なし (``save_tlist=None``, kryanneal 最節約モード) で最終状態
``ψ(T)`` のみ取得し参照解との infidelity ``1 - |<ψ_ref|ψ>|^2`` を測る.

## 出力 CSV

``<output_dir>/bench_readme_<scenario>.csv``:

```
scenario,n,T,seed,solver,knob_name,knob_value,wall_sec,infidelity,n_steps_eff
non-stiff,18,10000.0,20260518,kryanneal,atol,1e-05,1234.5,4.218e-09,8766
non-stiff,18,10000.0,20260518,qutip,tol,1e-05,9876.3,1.234e-06,
...
```

`plot_readme_figure.py` がこれを読んで scatter plot を作る.
"""

from __future__ import annotations

import argparse
import csv
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from benchmarks._readme_figure_helpers import (
    build_qutip_hamiltonian,
    infidelity,
    run_qutip,
)
from kryanneal import IsingProblem, QuantumAnnealer, Schedule
from kryanneal.initial_states import uniform_superposition


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


def run_bench(
    *,
    problem_file: Path,
    reference_file: Path,
    kryanneal_atols: Sequence[float],
    qutip_tols: Sequence[float],
    output_dir: Path,
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

    print(
        f"\n=== bench: scenario={scenario}, n={n}, T={T:.0f}, seed={seed} ===",
        flush=True,
    )
    print(f"problem:   {problem_file}", flush=True)
    print(f"reference: {reference_file}", flush=True)

    rows: list[dict[str, object]] = []

    for atol in kryanneal_atols:
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

    for tol in qutip_tols:
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

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"bench_readme_{scenario}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
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
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[done] wrote {csv_path}", flush=True)
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
        default=Path("benchmarks/results/readme-figure"),
        help="CSV 出力先",
    )
    args = parser.parse_args()

    run_bench(
        problem_file=args.problem_file,
        reference_file=args.reference_file,
        kryanneal_atols=args.kryanneal_atols,
        qutip_tols=args.qutip_tols,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
