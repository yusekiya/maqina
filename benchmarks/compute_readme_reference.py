"""README figure 用の参照解を QuTiP で 1 度だけ計算して保存する.

``build_readme_problem.py`` の出力 npz を読み, 与えた ``T`` で
``qutip.sesolve`` を **Adams (default ODE 法) で許容誤差 sweep + BDF で
別解法または sweep** 走らせて以下を検証:

1. **primary 解法の自己収束**: sweep 内隣接 tol 間の infidelity が
   ``--convergence-threshold`` を下回るか. 下回らない場合 ``converged = False``.
2. **解法独立性**: Adams 最細と BDF 最細の infidelity が同 threshold を
   下回るか. ``solver_independent`` flag に記録.

primary 解法は ``--bdf-tols`` の指定で切替わる:

- ``--bdf-tols`` 未指定 (legacy): ``primary = "adams"``. ``psi_ref`` は Adams
  最細, BDF は ``--bdf-tol`` (default = Adams 最細 tol) で 1 点だけ cross-check.
- ``--bdf-tols A,B,C,...`` (BDF sweep mode): ``primary = "bdf"``. ``psi_ref``
  は BDF 最細, BDF 隣接 pair で自己収束判定. Adams sweep は cross-check 用に
  そのまま残す.

stiff scenario では Adams の adaptive step 制御が 1e-9 で頭打ちになり Adams
自己収束を threshold 1e-13 でクリアできない既知の現象がある (commit
2712a86 / issue: bench 0.12.0 SUMMARY §4.2). この場合 BDF sweep mode に
切替えて BDF 自身を primary にすると, より tight な参照解 (≲ 1e-12) が得られる
可能性がある.

出力 npz の主要フィールド:

- ``psi_ref``: 参照解 (primary 解法の最細 tol の最終状態), shape ``(2**n,)``
- ``T``, ``scenario``, ``n``, ``seed``: 問題メタデータ
- ``adams_tols``, ``adams_walls``, ``adams_pairwise_inf``: Adams sweep の
  tol 列 (coarse → fine), wall 列, 隣接 pair infidelity 列
- ``bdf_tols``, ``bdf_walls``, ``bdf_pairwise_inf``: BDF sweep の同上.
  legacy mode では ``bdf_tols`` は長さ 1, ``bdf_pairwise_inf`` は空配列.
- ``bdf_vs_adams_inf``: ``infidelity(adams_finest, bdf_finest)``
- ``converged``: primary 解法の自己収束 (legacy = Adams pairwise / BDF mode =
  BDF pairwise)
- ``solver_independent``: Adams 最細 ≡ BDF 最細 の独立性 flag
- ``convergence_threshold``: 判定に使った閾値
- ``primary_solver``: ``"adams"`` または ``"bdf"`` (新フィールド; legacy npz
  には存在しないが ``bench_readme_figure.py`` 側は読まないので互換)
- ``bdf_tol``, ``bdf_wall_sec``: legacy フィールド (= ``bdf_tols[-1]`` /
  ``bdf_walls[-1]``; 旧 consumer 用に保持)
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
from maqina.initial_states import uniform_superposition


def _parse_floats(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def compute_reference(
    problem_file: Path,
    T: float,
    ref_tols: list[float],
    bdf_tols: list[float],
    convergence_threshold: float,
    output: Path,
) -> None:
    """``problem_file`` を読み Adams + BDF で参照解を計算し ``output`` に保存する.

    primary 解法は ``len(bdf_tols) >= 2`` で BDF, それ以外で Adams.
    """
    ensure_qutip()

    pdata = np.load(problem_file)
    h_p_diag = np.asarray(pdata["H_p_diag"], dtype=np.float64)
    h_x = np.asarray(pdata["h_x"], dtype=np.float64)
    n = int(pdata["n"])
    seed = int(pdata["seed"])
    scenario = str(pdata["scenario"])

    expected_dim = 1 << n
    if h_p_diag.shape != (expected_dim,):
        raise ValueError(
            f"H_p_diag shape {h_p_diag.shape} does not match n={n} "
            f"(expected ({expected_dim},))"
        )

    primary_solver = "bdf" if len(bdf_tols) >= 2 else "adams"

    # tol 列を coarse (= 大) → fine (= 小) 順に並べる. pairwise infidelity は
    # 隣接 pair (k, k+1) で計算するので fine 側に近いほど小さくなるはず.
    adams_tols_sorted = sorted(ref_tols, reverse=True)
    bdf_tols_sorted = sorted(bdf_tols, reverse=True)

    psi0 = uniform_superposition(n)
    h_t = build_qutip_hamiltonian(h_x, h_p_diag, T)

    print(
        f"=== compute_reference: scenario={scenario}, n={n}, T={T}, seed={seed} ===",
        flush=True,
    )
    print(f"primary solver: {primary_solver}", flush=True)
    print(
        f"Adams tol sweep (coarse → fine): {[f'{t:.0e}' for t in adams_tols_sorted]}",
        flush=True,
    )
    print(
        f"BDF tol sweep (coarse → fine):   {[f'{t:.0e}' for t in bdf_tols_sorted]}",
        flush=True,
    )
    print(f"convergence threshold: {convergence_threshold:.0e}\n", flush=True)

    # --- Adams sweep ---------------------------------------------------
    adams_walls: list[float] = []
    adams_states: list[np.ndarray] = []
    for tol in adams_tols_sorted:
        print(f"[Adams] tol={tol:.0e} computing ...", flush=True)
        wall, psi = run_qutip(h_t, psi0, T, n, tol, method="adams")
        adams_walls.append(wall)
        adams_states.append(psi)
        print(f"[Adams] tol={tol:.0e}: wall={wall:.1f}s", flush=True)

    adams_pairwise_inf: list[float] = []
    for k in range(len(adams_states) - 1):
        inf = infidelity(adams_states[k + 1], adams_states[k])
        adams_pairwise_inf.append(inf)
        print(
            f"[Adams] infidelity(tol={adams_tols_sorted[k]:.0e}, "
            f"tol={adams_tols_sorted[k + 1]:.0e}) = {inf:.3e}",
            flush=True,
        )

    # --- BDF sweep -----------------------------------------------------
    print("", flush=True)
    bdf_walls: list[float] = []
    bdf_states: list[np.ndarray] = []
    for tol in bdf_tols_sorted:
        print(f"[BDF] tol={tol:.0e} computing ...", flush=True)
        wall, psi = run_qutip(h_t, psi0, T, n, tol, method="bdf")
        bdf_walls.append(wall)
        bdf_states.append(psi)
        print(f"[BDF] tol={tol:.0e}: wall={wall:.1f}s", flush=True)

    bdf_pairwise_inf: list[float] = []
    for k in range(len(bdf_states) - 1):
        inf = infidelity(bdf_states[k + 1], bdf_states[k])
        bdf_pairwise_inf.append(inf)
        print(
            f"[BDF] infidelity(tol={bdf_tols_sorted[k]:.0e}, "
            f"tol={bdf_tols_sorted[k + 1]:.0e}) = {inf:.3e}",
            flush=True,
        )

    # --- 解法独立性 + primary 解法選択 ----------------------------------
    adams_finest = adams_states[-1]
    bdf_finest = bdf_states[-1]
    bdf_vs_adams = infidelity(bdf_finest, adams_finest)
    print(
        f"\n[cross-check] infidelity(BDF finest, Adams finest) = {bdf_vs_adams:.3e}",
        flush=True,
    )

    if primary_solver == "bdf":
        psi_ref = bdf_finest
        primary_pairwise = bdf_pairwise_inf
        primary_label = "BDF"
    else:
        psi_ref = adams_finest
        primary_pairwise = adams_pairwise_inf
        primary_label = "Adams"

    if primary_pairwise:
        converged = max(primary_pairwise) < convergence_threshold
    else:
        # sweep が 1 点のみ → 自己収束判定不能. 保守的に False.
        converged = False
    solver_independent = bdf_vs_adams < convergence_threshold

    flag_ok = "[ok]"
    flag_warn = "[WARN]"
    print(
        f"{flag_ok if converged else flag_warn} "
        f"{primary_label} pairwise convergence (threshold "
        f"{convergence_threshold:.0e})",
        flush=True,
    )
    print(
        f"{flag_ok if solver_independent else flag_warn} "
        f"BDF vs Adams independence (threshold {convergence_threshold:.0e})",
        flush=True,
    )
    if not converged:
        print(
            f"WARNING: {primary_label} 列が収束していません. tol 列を細かくする "
            f"/ convergence-threshold を緩めるかを検討してください.",
            flush=True,
        )
    if not solver_independent:
        print(
            "WARNING: Adams と BDF の解が threshold 内で一致していません. "
            "問題が病的か, tol が緩すぎる可能性があります.",
            flush=True,
        )

    # --- 保存 ----------------------------------------------------------
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        psi_ref=psi_ref,
        T=float(T),
        problem_file=str(problem_file),
        scenario=scenario,
        n=int(n),
        seed=int(seed),
        adams_tols=np.array(adams_tols_sorted, dtype=np.float64),
        adams_walls=np.array(adams_walls, dtype=np.float64),
        adams_pairwise_inf=np.array(adams_pairwise_inf, dtype=np.float64),
        bdf_tols=np.array(bdf_tols_sorted, dtype=np.float64),
        bdf_walls=np.array(bdf_walls, dtype=np.float64),
        bdf_pairwise_inf=np.array(bdf_pairwise_inf, dtype=np.float64),
        # legacy fields (旧 consumer / 旧 npz schema 互換).
        bdf_tol=float(bdf_tols_sorted[-1]),
        bdf_wall_sec=float(bdf_walls[-1]),
        bdf_vs_adams_inf=float(bdf_vs_adams),
        converged=bool(converged),
        solver_independent=bool(solver_independent),
        convergence_threshold=float(convergence_threshold),
        primary_solver=primary_solver,
    )
    print(f"\n[done] wrote {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--problem-file",
        type=Path,
        required=True,
        help="build_readme_problem.py が出力した npz",
    )
    parser.add_argument(
        "--T",
        type=float,
        required=True,
        help="最終時刻",
    )
    parser.add_argument(
        "--ref-tols",
        type=_parse_floats,
        default=[1e-9, 1e-10, 1e-11],
        help="Adams 収束 sweep の tol 列 (CSV, 粗→細でも細→粗でもよい)",
    )
    parser.add_argument(
        "--bdf-tols",
        type=_parse_floats,
        default=None,
        help=(
            "BDF sweep の tol 列 (CSV). 2 点以上指定すると primary = BDF に "
            "切替わり psi_ref = BDF 最細, 自己収束判定も BDF pairwise になる. "
            "未指定または 1 点のみは legacy 動作 (primary = Adams, BDF は "
            "cross-check 1 点)."
        ),
    )
    parser.add_argument(
        "--bdf-tol",
        type=float,
        default=None,
        help=(
            "[legacy] 単点 BDF tol. --bdf-tols が指定されていれば無視される. "
            "default は --ref-tols の最細値と同値."
        ),
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
        help="npz 出力先 (default: benchmarks/data/reference_<...>.npz)",
    )
    args = parser.parse_args()

    # BDF tol 列を確定. 優先順:
    #   1. --bdf-tols が指定されていればそれ
    #   2. --bdf-tol が指定されていれば [bdf_tol]
    #   3. どちらも未指定なら [ref_tols の最細値]
    if args.bdf_tols is not None and len(args.bdf_tols) > 0:
        bdf_tols = args.bdf_tols
    elif args.bdf_tol is not None:
        bdf_tols = [args.bdf_tol]
    else:
        bdf_tols = [min(args.ref_tols)]

    # default output 名: 問題 npz のメタデータから組み立てる.
    if args.output is None:
        pdata = np.load(args.problem_file)
        scenario = str(pdata["scenario"])
        n = int(pdata["n"])
        seed = int(pdata["seed"])
        args.output = Path("benchmarks/data") / (
            f"reference_{scenario}_n{n}_T{args.T:.0f}_seed{seed}.npz"
        )

    compute_reference(
        problem_file=args.problem_file,
        T=args.T,
        ref_tols=args.ref_tols,
        bdf_tols=bdf_tols,
        convergence_threshold=args.convergence_threshold,
        output=args.output,
    )


if __name__ == "__main__":
    main()
