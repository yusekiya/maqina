#!/usr/bin/env python3
"""β_m a posteriori error estimator の数値妥当性検証 (Phase 7 #93 着手前).

issue #93 "Lanczos β_m exposure + Richardson 誤差源分離" の Approach C
(β_m 直接観測) が実装に値する精度を持つかを Phase 7 本格着手前に確認する.

検証対象の理論式 (Saad 1992 / Hochbruck-Lubich 1997):

    err_lanczos(m) ≈ β_m × |c_m| × ‖ψ‖

ここで T_m は構築した m × m 三重対角行列, c = exp(-i dt T_m) e_0,
β_m は Lanczos build の最終 off-diagonal (= 次の Krylov 方向への漏れ強度).

実験デザイン:

    H = a · H_driver(h_x) + b · H_problem(h_p_diag)  (TFIM)

    - 3 種類の状態 ψ:
        "initial": |+⟩^N
        "mid":     s=0.5 まで高精度に進めた state
        "late":    s=0.9 まで進めた state (準固有状態に近い)
    - dt ∈ {0.01, 0.1, 1.0}
    - m_test ∈ {4, 6, 8, 12, 16, 20}
    - m_ref = 48 (高精度 reference)
    - n ∈ {6, 8}

各 cell で:
    actual_err  = ‖ψ_test - ψ_ref‖
    saad_est    = β_m × |c_m| × ‖ψ‖
    ratio       = saad_est / actual_err

合格条件: ratio が全 cell で [0.01, 100] (2 桁以内) に収まること.
理想: ratio ∈ [0.5, 5] (1 桁未満).
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np

# 内部 API (Phase 7 着手前なので production 経路は触らず, raw matvec を直接使う)
from kryanneal._rust import apply_h_kryanneal_py
from kryanneal.initial_states import uniform_superposition


def make_tfim(n: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """ランダム TFIM の h_x / h_p_diag を返す.

    h_x_i ∈ U(0.8, 1.2), J_ij ∈ U(-1, 1) (full connectivity), h_z_i ∈ U(-0.5, 0.5).
    """
    rng = np.random.default_rng(seed)
    h_x = rng.uniform(0.8, 1.2, size=n).astype(np.float64)

    # H_problem = Σ_i h_z_i Z_i + Σ_{i<j} J_ij Z_i Z_j (LSB-first bit 規約)
    dim = 1 << n
    h_p_diag = np.zeros(dim, dtype=np.float64)
    h_z = rng.uniform(-0.5, 0.5, size=n)
    J = rng.uniform(-1.0, 1.0, size=(n, n))
    for x in range(dim):
        sigma = np.array([1 - 2 * ((x >> i) & 1) for i in range(n)], dtype=np.float64)
        diag = float(h_z @ sigma)
        for i in range(n):
            for j in range(i + 1, n):
                diag += J[i, j] * sigma[i] * sigma[j]
        h_p_diag[x] = diag
    return h_x, h_p_diag


def matvec_factory(h_x: np.ndarray, h_p_diag: np.ndarray, a_t: float, b_t: float):
    """exp(-i dt H) を Lanczos で近似するための matvec closure."""

    def matvec(v: np.ndarray) -> np.ndarray:
        return apply_h_kryanneal_py(v, h_x, h_p_diag, a_t, b_t)

    return matvec


def lanczos_with_diagnostics(
    matvec, psi: np.ndarray, dt: float, m: int, tol: float = 0.0
) -> tuple[np.ndarray, int, np.ndarray, np.ndarray]:
    """`_python_lanczos_propagate` 相当 + alpha/beta 全配列を返す.

    Returns
    -------
    psi_new : (dim,) complex128
    m_eff : int
    alpha : (m_eff,) float64
    beta  : (m_eff,) float64    # beta[m_eff - 1] = β_m (= 次の off-diagonal,
                                # build を 1 step 進めれば見える漏れ強度)
    """
    if m < 1:
        raise ValueError(f"m must be >= 1, got {m!r}")
    dim = psi.shape[0]
    psi_norm = float(np.linalg.norm(psi))
    if psi_norm == 0.0 or dim == 0:
        return np.zeros_like(psi), 0, np.zeros(0), np.zeros(0)

    v_mat = np.zeros((dim, m), dtype=np.complex128)
    alpha = np.zeros(m, dtype=np.float64)
    beta = np.zeros(m, dtype=np.float64)
    v_mat[:, 0] = psi / psi_norm
    m_eff = m

    for k in range(m):
        vk = np.ascontiguousarray(v_mat[:, k])
        w = matvec(vk).astype(np.complex128, copy=False)
        alpha_k = float(np.real(np.vdot(v_mat[:, k], w)))
        alpha[k] = alpha_k
        w = w - alpha_k * v_mat[:, k]
        if k >= 1:
            w = w - beta[k - 1] * v_mat[:, k - 1]
        # 2-pass full re-orthogonalization
        for _ in range(2):
            for j in range(k + 1):
                proj = np.vdot(v_mat[:, j], w)
                w = w - proj * v_mat[:, j]
        beta_k = float(np.linalg.norm(w))
        beta[k] = beta_k
        if beta_k < tol:
            m_eff = k + 1
            break
        if k + 1 < m:
            v_mat[:, k + 1] = w / beta_k

    if m_eff == 1:
        lam = alpha[:1].copy()
        q = np.array([[1.0]], dtype=np.float64)
    else:
        t_dense = np.zeros((m_eff, m_eff), dtype=np.float64)
        for i in range(m_eff):
            t_dense[i, i] = alpha[i]
        for i in range(m_eff - 1):
            t_dense[i, i + 1] = beta[i]
            t_dense[i + 1, i] = beta[i]
        lam, q = np.linalg.eigh(t_dense)

    phases = np.exp(-1j * dt * lam)
    coeff = psi_norm * (q @ (phases * q[0, :]))
    psi_new = v_mat[:, :m_eff] @ coeff
    return psi_new, m_eff, alpha[:m_eff], beta[:m_eff]


def saad_estimate(
    alpha: np.ndarray, beta: np.ndarray, dt: float, psi_norm: float
) -> tuple[float, float]:
    """Saad/Hochbruck-Lubich の β_m × |c_m| × ‖ψ‖ a posteriori 推定.

    Returns
    -------
    (saad_est, c_m_abs)
    """
    m_eff = alpha.shape[0]
    if m_eff == 0:
        return 0.0, 0.0
    if m_eff == 1:
        lam = alpha[:1].copy()
        q = np.array([[1.0]], dtype=np.float64)
    else:
        t_dense = np.zeros((m_eff, m_eff), dtype=np.float64)
        for i in range(m_eff):
            t_dense[i, i] = alpha[i]
        for i in range(m_eff - 1):
            t_dense[i, i + 1] = beta[i]
            t_dense[i + 1, i] = beta[i]
        lam, q = np.linalg.eigh(t_dense)
    phases = np.exp(-1j * dt * lam)
    # c = exp(-i dt T_m) e_0; その m-th component (last row)
    c_m = q[m_eff - 1, :] @ (phases * q[0, :])
    c_m_abs = float(abs(c_m))
    beta_m = float(beta[m_eff - 1])
    return beta_m * c_m_abs * psi_norm, c_m_abs


def prepare_state(
    h_x: np.ndarray, h_p_diag: np.ndarray, n: int, s: float, T: float
) -> np.ndarray:
    """linear schedule で s=s_target まで高精度 (m=32, dt=T/200 等) に進めた state.

    s=0 のとき |+⟩^N を return.
    """
    psi = uniform_superposition(n)
    if s <= 0.0:
        return psi
    t_target = s * T
    n_steps = 200
    dt = t_target / n_steps
    for k in range(n_steps):
        t_mid = (k + 0.5) * dt
        a_t = 1.0 - t_mid / T
        b_t = t_mid / T
        mv = matvec_factory(h_x, h_p_diag, a_t, b_t)
        psi, _, _, _ = lanczos_with_diagnostics(mv, psi, dt, m=32, tol=0.0)
    return psi


def run_cell(
    h_x: np.ndarray,
    h_p_diag: np.ndarray,
    psi: np.ndarray,
    a_t: float,
    b_t: float,
    dt: float,
    m_test: int,
    m_ref: int = 48,
) -> dict:
    mv = matvec_factory(h_x, h_p_diag, a_t, b_t)
    # reference (very high m)
    psi_ref, _, _, _ = lanczos_with_diagnostics(mv, psi, dt, m=m_ref, tol=0.0)
    # test
    psi_test, m_eff, alpha, beta = lanczos_with_diagnostics(
        mv, psi, dt, m=m_test, tol=0.0
    )
    psi_norm = float(np.linalg.norm(psi))
    actual_err = float(np.linalg.norm(psi_test - psi_ref))
    saad_est, c_m_abs = saad_estimate(alpha, beta, dt, psi_norm)
    beta_m = float(beta[m_eff - 1])
    ratio = saad_est / actual_err if actual_err > 0 else float("inf")
    return dict(
        m_eff=m_eff,
        beta_m=beta_m,
        c_m_abs=c_m_abs,
        saad_est=saad_est,
        actual_err=actual_err,
        ratio=ratio,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-values", type=str, default="6,8")
    parser.add_argument("--dt-values", type=str, default="0.01,0.1,1.0")
    parser.add_argument("--m-values", type=str, default="4,6,8,12,16,20")
    parser.add_argument("--m-ref", type=int, default=48)
    parser.add_argument(
        "--T-prep", type=float, default=10.0, help="prep evolution の総時間"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="default: tools/results/verify_beta_m_<timestamp>/",
    )
    args = parser.parse_args()

    n_values = [int(s) for s in args.n_values.split(",")]
    dt_values = [float(s) for s in args.dt_values.split(",")]
    m_values = [int(s) for s in args.m_values.split(",")]

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    if args.output_dir is None:
        out_dir = Path(__file__).parent / "results" / f"verify_beta_m_{timestamp}"
    else:
        out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "verify_beta_m.csv"
    md_path = out_dir / "verify_beta_m.md"

    # schedule midpoint (s=0.5 で a=b=0.5 にあたる代表点 + s=0.1/0.9 サンプル)
    state_specs = [
        ("initial", 0.0, 0.5),  # |+⟩^N で a=b=0.5 で評価
        ("mid", 0.5, 0.5),  # s=0.5 まで進めた state で a=b=0.5
        ("late", 0.9, 0.1),  # s=0.9 まで進めた state で a=0.1, b=0.9
    ]

    rows = []
    print("# β_m estimator validation (#93 Approach C pre-check)")
    print(f"output: {out_dir}")
    for n in n_values:
        print(f"[n={n}] building TFIM...")
        h_x, h_p_diag = make_tfim(n, seed=args.seed)
        for state_label, s_prep, _a_eval_unused in state_specs:
            psi = prepare_state(h_x, h_p_diag, n, s=s_prep, T=args.T_prep)
            # 評価点は state_label に応じて
            if state_label == "initial":
                a_eval, b_eval = 0.5, 0.5
            elif state_label == "mid":
                a_eval, b_eval = 0.5, 0.5
            else:  # late
                a_eval, b_eval = 0.1, 0.9
            print(
                f"  [state={state_label}] prepared (s={s_prep}, eval at a={a_eval}, b={b_eval})"
            )
            for dt in dt_values:
                for m_test in m_values:
                    res = run_cell(
                        h_x, h_p_diag, psi, a_eval, b_eval, dt, m_test, m_ref=args.m_ref
                    )
                    row = dict(
                        n=n,
                        state=state_label,
                        a_eval=a_eval,
                        b_eval=b_eval,
                        dt=dt,
                        m_test=m_test,
                        **res,
                    )
                    rows.append(row)

    # CSV
    with csv_path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # MD summary
    lines: list[str] = []
    lines.append("# β_m a posteriori estimator validation (#93)\n")
    lines.append(f"- seed: {args.seed}")
    lines.append(f"- m_ref: {args.m_ref}")
    lines.append(f"- T_prep: {args.T_prep}")
    lines.append(f"- n_values: {n_values}")
    lines.append(f"- dt_values: {dt_values}")
    lines.append(f"- m_values: {m_values}")
    lines.append("")
    lines.append("Saad/Hochbruck-Lubich 推定: `err_est = β_m · |c_m| · ‖ψ‖`,")
    lines.append("  ここで `c = exp(-i dt T_m) e_0`, `|c_m|` は最終成分の絶対値.")
    lines.append("")
    lines.append("合格条件: `ratio = err_est / err_actual` が [0.01, 100] (2 桁以内).")
    lines.append("")

    for n in n_values:
        lines.append(f"## n = {n}\n")
        for state_label, _, _ in state_specs:
            lines.append(f"### state = {state_label}\n")
            lines.append(
                "| dt | m_test | m_eff | β_m | \\|c_m\\| | saad_est | actual_err | ratio |"
            )
            lines.append("|---|---|---|---|---|---|---|---|")
            for r in rows:
                if r["n"] != n or r["state"] != state_label:
                    continue
                lines.append(
                    f"| {r['dt']:.3g} | {r['m_test']} | {r['m_eff']} | "
                    f"{r['beta_m']:.3e} | {r['c_m_abs']:.3e} | "
                    f"{r['saad_est']:.3e} | {r['actual_err']:.3e} | "
                    f"{r['ratio']:.3f} |"
                )
            lines.append("")

    # 合格判定 summary
    ratios = [
        r["ratio"]
        for r in rows
        if math.isfinite(r["ratio"]) and r["actual_err"] > 1e-15
    ]
    if ratios:
        log_ratios = [math.log10(r) for r in ratios if r > 0]
        if log_ratios:
            lines.append("## 合格判定\n")
            lines.append(f"- 有効 cell 数: {len(ratios)} / {len(rows)}")
            lines.append(f"- log10(ratio) min: {min(log_ratios):.2f}")
            lines.append(f"- log10(ratio) max: {max(log_ratios):.2f}")
            lines.append(
                f"- log10(ratio) median: {sorted(log_ratios)[len(log_ratios) // 2]:.2f}"
            )
            within_2 = sum(1 for x in log_ratios if abs(x) <= 2.0)
            within_1 = sum(1 for x in log_ratios if abs(x) <= 1.0)
            lines.append(
                f"- |log10(ratio)| ≤ 2 (2 桁以内): {within_2}/{len(log_ratios)}"
            )
            lines.append(
                f"- |log10(ratio)| ≤ 1 (1 桁以内): {within_1}/{len(log_ratios)}"
            )
            lines.append("")

    md_path.write_text("\n".join(lines))
    print(f"\nwrote {csv_path}")
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
