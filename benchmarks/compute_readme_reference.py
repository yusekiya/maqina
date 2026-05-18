"""README figure 用の参照解を QuTiP で 1 度だけ計算して保存する.

``build_readme_problem.py`` の出力 npz を読み, 与えた ``T`` で
``qutip.sesolve`` を **Adams (default ODE 法) で許容誤差 sweep + BDF で
同精度 1 点** 走らせて以下を検証:

1. **Adams 収束**: tol を粗→細に sweep し, 隣接 tol ペアの infidelity が
   ``convergence_threshold`` (default 1e-13) 未満であることを確認. これで
   「許容誤差を下げても解が動かない」状態を担保.
2. **解法独立性**: 同 ``bdf_tol`` (Adams 最高精度と同値) で BDF を 1 回計算し,
   Adams 最高精度との infidelity が threshold 未満であることを確認. これで
   「ODE 法依存性なし」を担保 (Schrödinger 方程式の unitary 性は本来 stiff
   ではないが, 問題ハミルトニアンの dynamic range が大きい場合 Adams と
   BDF で実効的に挙動が違いうるため両方を試す).

両方 pass で **Adams 最高精度の ψ_final を参照解として採用**.

参照解 npz の内容:

- ``psi_ref`` (complex128, shape (2^n,)): 採用された参照解 ψ_ref
- ``T`` (float): 時間発展の最終時刻
- ``problem_file`` (str): 参照元 problem npz の path (識別子)
- ``adams_tols`` (1D float, 細→粗の sort 済): 試した tol 列
- ``adams_walls`` (1D float): 各 Adams 計算の wall_sec
- ``adams_pairwise_inf`` (1D float, len = len(adams_tols) - 1): 隣接 tol ペア
  の infidelity (粗→細の順)
- ``bdf_tol`` (float): BDF 計算に使った tol (Adams 最細値と同値)
- ``bdf_wall_sec`` (float)
- ``bdf_vs_adams_inf`` (float): infidelity(BDF, Adams 最高精度)
- ``converged`` (bool): Adams 列の収束判定
- ``solver_independent`` (bool): BDF 一致判定
- ``convergence_threshold`` (float)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from benchmarks._readme_figure_helpers import (
    build_qutip_hamiltonian,
    ensure_qutip,
    infidelity,
    run_qutip,
)
from kryanneal.initial_states import uniform_superposition


def _parse_floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def compute_reference(
    *,
    problem_file: Path,
    T: float,
    ref_tols: list[float],
    bdf_tol: float | None,
    convergence_threshold: float,
    output: Path,
) -> Path:
    ensure_qutip()

    data = np.load(problem_file)
    h_p_diag = np.asarray(data["H_p_diag"], dtype=np.float64)
    h_x = np.asarray(data["h_x"], dtype=np.float64)
    n = int(data["n"])
    seed = int(data["seed"])
    scenario = str(data["scenario"])

    if h_p_diag.shape != (1 << n,):
        raise ValueError(
            f"H_p_diag shape {h_p_diag.shape} does not match n={n} (expected {1 << n})"
        )

    psi0 = uniform_superposition(n)
    h_t = build_qutip_hamiltonian(h_x, h_p_diag, T)

    # tol 列を粗 (大) → 細 (小) で sort
    tols_sorted = sorted(ref_tols, reverse=True)
    if bdf_tol is None:
        bdf_tol = tols_sorted[-1]  # Adams 最細と同値

    print(
        f"=== compute_reference: scenario={scenario}, n={n}, T={T}, seed={seed} ===",
        flush=True,
    )
    print(f"Adams tol sweep (coarse → fine): {tols_sorted}", flush=True)
    print(f"BDF tol: {bdf_tol:.0e}", flush=True)
    print(f"convergence threshold: {convergence_threshold:.0e}\n", flush=True)

    # Adams sweep
    adams_walls: list[float] = []
    adams_psis: list[np.ndarray] = []
    for tol in tols_sorted:
        print(f"[Adams] tol={tol:.0e} computing ...", flush=True)
        wall, psi = run_qutip(h_t, psi0, T, n, tol, method="adams")
        print(f"[Adams] tol={tol:.0e}: wall={wall:.1f}s", flush=True)
        adams_walls.append(wall)
        adams_psis.append(psi)

    # 隣接 pair の infidelity (収束指標)
    adams_pairwise_inf: list[float] = []
    for i in range(len(adams_psis) - 1):
        inf = infidelity(adams_psis[i + 1], adams_psis[i])
        adams_pairwise_inf.append(inf)
        print(
            f"[Adams] infidelity(tol={tols_sorted[i]:.0e}, "
            f"tol={tols_sorted[i + 1]:.0e}) = {inf:.3e}",
            flush=True,
        )

    converged = (
        len(adams_pairwise_inf) > 0
        and adams_pairwise_inf[-1] < convergence_threshold
    )
    psi_adams_finest = adams_psis[-1]

    # BDF 1 点で解法独立性検証
    print(f"\n[BDF] tol={bdf_tol:.0e} computing ...", flush=True)
    bdf_wall, psi_bdf = run_qutip(h_t, psi0, T, n, bdf_tol, method="bdf")
    bdf_vs_adams_inf = infidelity(psi_bdf, psi_adams_finest)
    print(
        f"[BDF] tol={bdf_tol:.0e}: wall={bdf_wall:.1f}s, "
        f"infidelity(BDF, Adams finest) = {bdf_vs_adams_inf:.3e}",
        flush=True,
    )
    solver_independent = bdf_vs_adams_inf < convergence_threshold

    status_adams = "[ok]" if converged else "[WARN]"
    status_bdf = "[ok]" if solver_independent else "[WARN]"
    print(
        f"\n{status_adams} Adams pairwise convergence "
        f"(threshold {convergence_threshold:.0e})",
        flush=True,
    )
    print(
        f"{status_bdf} BDF vs Adams independence "
        f"(threshold {convergence_threshold:.0e})",
        flush=True,
    )
    if not converged:
        print(
            "WARNING: Adams 列が収束していません. ref-tols を細かくする / "
            "convergence-threshold を緩めるかを検討してください.",
            flush=True,
        )
    if not solver_independent:
        print(
            "WARNING: Adams と BDF の解が threshold 内で一致していません. "
            "問題が病的か, bdf-tol が緩すぎる可能性があります.",
            flush=True,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        psi_ref=psi_adams_finest,
        T=float(T),
        problem_file=str(problem_file),
        scenario=scenario,
        n=int(n),
        seed=int(seed),
        adams_tols=np.array(tols_sorted, dtype=np.float64),
        adams_walls=np.array(adams_walls, dtype=np.float64),
        adams_pairwise_inf=np.array(adams_pairwise_inf, dtype=np.float64),
        bdf_tol=float(bdf_tol),
        bdf_wall_sec=float(bdf_wall),
        bdf_vs_adams_inf=float(bdf_vs_adams_inf),
        converged=bool(converged),
        solver_independent=bool(solver_independent),
        convergence_threshold=float(convergence_threshold),
    )
    print(f"\n[done] wrote {output}", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--problem-file",
        type=Path,
        required=True,
        help="build_readme_problem.py が出力した npz",
    )
    parser.add_argument("--T", type=float, required=True, help="最終時刻")
    parser.add_argument(
        "--ref-tols",
        type=_parse_floats,
        default=[1e-9, 1e-10, 1e-11],
        help="Adams 収束 sweep の tol 列 (CSV, 粗→細でも細→粗でもよい)",
    )
    parser.add_argument(
        "--bdf-tol",
        type=float,
        default=None,
        help="BDF 計算の tol (default: ref-tols の最細値と同値)",
    )
    parser.add_argument(
        "--convergence-threshold",
        type=float,
        default=1e-13,
        help="Adams 列の隣接 pair / BDF vs Adams が一致を判定する閾値",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="npz 出力先 (default: benchmarks/data/readme_reference_<...>.npz)",
    )
    args = parser.parse_args()

    if args.output is None:
        data = np.load(args.problem_file)
        scenario = str(data["scenario"])
        n = int(data["n"])
        seed = int(data["seed"])
        args.output = (
            Path("benchmarks/data")
            / f"readme_reference_{scenario}_n{n}_T{args.T:.0f}_seed{seed}.npz"
        )

    compute_reference(
        problem_file=args.problem_file,
        T=args.T,
        ref_tols=args.ref_tols,
        bdf_tol=args.bdf_tol,
        convergence_threshold=args.convergence_threshold,
        output=args.output,
    )


if __name__ == "__main__":
    main()
