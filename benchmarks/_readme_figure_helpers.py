"""README figure pipeline 共有ヘルパ.

`build_readme_problem.py` (問題定義 → npz), `compute_readme_reference.py`
(参照解 → npz, Adams 収束 + BDF 解法独立性検証), `bench_readme_figure.py`
(両 npz を読んで sweep) の 3 script が共有する小物関数群.

公開する 4 つ:

- ``build_qutip_hamiltonian``: kinema の ``(h_x, h_p_diag)`` を QuTiP の
  ``[[H_drv, A(t)], [H_p, B(t)]]`` sparse 表現に変換.
- ``run_qutip``: ``qutip.sesolve`` を 1 回走らせ ``(wall_sec, psi_final)``.
  ``method`` で ``"adams"`` (default, non-stiff) / ``"bdf"`` (stiff) を切替.
- ``infidelity``: ``1 - |<ψ_ref | ψ>|^2`` (clip して非負).
- ``ensure_qutip``: ``qutip`` 未インストール時に分かりやすいエラーを上げる.

`bench_qutip_large.py` の同名ヘルパと意図的に重複した実装を持つ. 既存
script を壊さず, README figure pipeline の依存だけ局所化する.
"""

from __future__ import annotations

import time

import numpy as np

try:
    import qutip
except ImportError:  # pragma: no cover - dev 依存
    qutip = None  # type: ignore[assignment]


def ensure_qutip() -> None:
    """``qutip`` が import 済みであることを確認. 未インストールなら ImportError."""
    if qutip is None:
        raise ImportError(
            "qutip is required for README figure pipeline; "
            "install via `uv add --dev qutip`"
        )


def build_qutip_hamiltonian(h_x: np.ndarray, h_p_diag: np.ndarray, T: float) -> list:
    """``[[H_drv, A(t)], [H_p, B(t)]]`` の QuTiP sparse 表現を構築する.

    kinema の LSB-first ``σ_i = 1 - 2·b_i`` 規約と QuTiP の MSB-first
    tensor 規約を吸収するため X を tensor list の位置 ``n-1-i`` に挿入.
    n=18 以上の規模では dense backing がメモリに乗らないので sparse 固定.
    """
    ensure_qutip()
    n = h_x.shape[0]
    sx = qutip.sigmax()
    si = qutip.qeye(2)
    h_drv: object | None = None
    for i in range(n):
        ops = [si] * n
        ops[n - 1 - i] = sx
        term = -float(h_x[i]) * qutip.tensor(ops)
        h_drv = term if h_drv is None else h_drv + term
    h_p = qutip.qdiags(h_p_diag, 0, dims=[[2] * n, [2] * n])
    return [
        [h_drv, f"(1 - t/{T})"],
        [h_p, f"(t/{T})"],
    ]


def run_qutip(
    h_t: list,
    psi0: np.ndarray,
    T: float,
    n: int,
    tol: float,
    method: str = "adams",
) -> tuple[float, np.ndarray]:
    """``sesolve`` を ``atol = rtol = tol`` で 1 回走らせ ``(wall_sec, psi_final)``.

    ``method`` は QuTiP の ODE 法選択 (default ``"adams"`` = non-stiff 向き,
    ``"bdf"`` = stiff 向き). 参照解計算で両方を試すことで解法独立性 (Adams
    ≡ BDF) を担保する.
    """
    ensure_qutip()
    psi0_q = qutip.Qobj(psi0.reshape(-1, 1), dims=[[2] * n, [1] * n])
    options = {
        "atol": float(tol),
        "rtol": float(tol),
        "nsteps": 100_000_000,
        "method": method,
    }
    t_start = time.perf_counter()
    sol = qutip.sesolve(h_t, psi0_q, np.array([0.0, T]), options=options)
    elapsed = time.perf_counter() - t_start
    psi_final = sol.states[-1].full().ravel().astype(np.complex128)
    return elapsed, psi_final


def infidelity(psi: np.ndarray, psi_ref: np.ndarray) -> float:
    """``1 - |<ψ_ref | ψ>|^2`` (clip して非負).

    ``ψ`` / ``ψ_ref`` ともに L2-normalize されている前提.
    """
    overlap = np.vdot(psi_ref, psi)
    fidelity = abs(overlap) ** 2
    return max(0.0, 1.0 - float(fidelity))
