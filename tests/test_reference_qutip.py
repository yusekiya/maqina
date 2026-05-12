"""QuTiP ``sesolve`` を高精度 ODE 参照とした end-to-end fidelity テスト.

issue #8 / #21 acceptance: 小規模 (n=4) で random ``H_p_diag`` / ``h_x``
/ linear schedule に対し, ``QuantumAnnealer.run`` 各 method と QuTiP
``sesolve`` の終端状態 fidelity を以下のしきい値で要求する:

* ``method="m2"`` (n_steps=500): fidelity ``> 1 - 1e-6``.
* ``method="trotter"`` (n_steps=500): fidelity ``> 1 - 1e-4``.
  Strang 2 次は M2 と同じ ``O(dt^3)`` LTE オーダだが, 係数 / 中点採取の
  対称性誤差で ``1e-6`` までは届かないので ``1e-4`` 設定 (issue #21).

QuTiP は dev 依存のみで本番 wheel には入れない契約 (``docs/design.md``
§8). 拡張未ビルド or QuTiP 未 install の環境では ``pytest.importorskip``
で skip する.
"""

from __future__ import annotations

import numpy as np
import pytest

from kryanneal import IsingProblem, QuantumAnnealer, Schedule
from kryanneal.initial_states import uniform_superposition


qutip = pytest.importorskip("qutip")


def _build_qutip_hamiltonian(h_x: np.ndarray, h_p_diag: np.ndarray, T: float) -> list:
    """QuTiP ``sesolve`` 用 ``H(t) = [[H_drv, A(t)], [H_p, B(t)]]`` を組む.

    linear schedule (``A(s) = 1 - s``, ``B(s) = s``, ``s = t/T``) を前提.
    ``H_drv``, ``H_p`` を ``Qobj`` として構築し, 時間係数は文字列で渡す.
    """
    n = h_x.shape[0]
    dim = 1 << n

    h_drv = np.zeros((dim, dim), dtype=np.complex128)
    for i in range(n):
        mask = 1 << i
        for k in range(dim):
            h_drv[k, k ^ mask] += -h_x[i]
    h_p = np.diag(h_p_diag).astype(np.complex128)

    h_drv_q = qutip.Qobj(h_drv)
    h_p_q = qutip.Qobj(h_p)
    return [
        [h_drv_q, f"(1 - t/{T})"],
        [h_p_q, f"(t/{T})"],
    ]


def _fidelity(psi_a: np.ndarray, psi_b: np.ndarray) -> float:
    """``|⟨ψ_a|ψ_b⟩|^2`` (normalize 済み state 前提)."""
    return float(np.abs(np.vdot(psi_a, psi_b)) ** 2)


def test_quantum_annealer_matches_qutip_sesolve() -> None:
    """n=4 random H で QuTiP との fidelity > 1 - 1e-6 を要求する.

    seed 固定 (再現可能). ``n_steps=500`` で M2 中点則は十分小さい dt と
    なり, smooth schedule では Magnus M2 の LTE ~ O(dt^3) が支配する.
    """
    n = 4
    dim = 1 << n
    T = 5.0
    rng = np.random.default_rng(20251112)

    h_x = rng.uniform(0.5, 1.5, size=n).astype(np.float64)
    h_p_diag = rng.uniform(-1.0, 1.0, size=dim).astype(np.float64)

    prob = IsingProblem(n=n, H_p_diag=h_p_diag, h_x=h_x)
    sched = Schedule.linear(T=T)
    psi0 = uniform_superposition(n)

    ann = QuantumAnnealer(prob, sched)
    res = ann.run(psi0, 0.0, T, method="m2", n_steps=500)

    # QuTiP リファレンス.
    h_t = _build_qutip_hamiltonian(h_x, h_p_diag, T)
    psi0_q = qutip.Qobj(psi0.reshape(-1, 1))
    sol = qutip.sesolve(
        h_t,
        psi0_q,
        np.array([0.0, T]),
        options={"atol": 1e-12, "rtol": 1e-10, "nsteps": 100000},
    )
    psi_qutip = sol.states[-1].full().ravel()

    fid = _fidelity(res.psi_final, psi_qutip)
    assert fid > 1 - 1e-6, f"fidelity too low: {fid} (1 - fid = {1 - fid})"


def test_quantum_annealer_trotter_matches_qutip_sesolve() -> None:
    """``method="trotter"`` で QuTiP との fidelity ``> 1 - 1e-4`` (Phase 2 C3).

    M2 と同じ ``n=4`` random H / linear schedule / ``n_steps=500`` の
    設定で Strang 2 次 Trotter を走らせ, QuTiP との fidelity しきい値を
    issue #21 の規約に従って ``1 - 1e-4`` に設定する.
    """
    n = 4
    dim = 1 << n
    T = 5.0
    rng = np.random.default_rng(20251112)

    h_x = rng.uniform(0.5, 1.5, size=n).astype(np.float64)
    h_p_diag = rng.uniform(-1.0, 1.0, size=dim).astype(np.float64)

    prob = IsingProblem(n=n, H_p_diag=h_p_diag, h_x=h_x)
    sched = Schedule.linear(T=T)
    psi0 = uniform_superposition(n)

    ann = QuantumAnnealer(prob, sched)
    res = ann.run(psi0, 0.0, T, method="trotter", n_steps=500)

    h_t = _build_qutip_hamiltonian(h_x, h_p_diag, T)
    psi0_q = qutip.Qobj(psi0.reshape(-1, 1))
    sol = qutip.sesolve(
        h_t,
        psi0_q,
        np.array([0.0, T]),
        options={"atol": 1e-12, "rtol": 1e-10, "nsteps": 100000},
    )
    psi_qutip = sol.states[-1].full().ravel()

    fid = _fidelity(res.psi_final, psi_qutip)
    assert fid > 1 - 1e-4, f"fidelity too low (trotter): {fid} (1 - fid = {1 - fid})"
