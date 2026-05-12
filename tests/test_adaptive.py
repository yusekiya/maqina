"""adaptive driver (Phase 4 C3) の挙動テスト.

issue #39 の acceptance:

* ``evolve_schedule_adaptive_m2`` smoke: time-independent / 緩やかな
  time-dependent ``H`` で QuTiP との fidelity が ``> 1 - 1e-6``.
* ``evolve_schedule_adaptive_richardson`` smoke: 同上 (より高次なので
  同じ ``tol_step`` で M2 同等以上).
* PI controller の dt 増減: 緩やか schedule で ``dt`` が ``dt0`` から
  ``dt_max`` 方向に増加する.
* reject: 不自然に大きい ``dt0`` を渡すと最初の数 step が reject される.
* ``max_rejects`` 連続超過で ``RuntimeError``.
* ``save_tlist is not None`` で ``NotImplementedError``.

driver 単体テストなので ``QuantumAnnealer`` ファサードは経由せず
``evolve_schedule_adaptive_*`` を直接呼ぶ. QuTiP 比較は ``test_annealer.py``
の facade smoke と棲み分け (``test_adaptive.py`` は driver 単体, 同 facade
経路の end-to-end smoke は ``test_annealer.py`` 側で別途検証する).
"""

from __future__ import annotations

import numpy as np
import pytest

from kryanneal import IsingProblem, Schedule
from kryanneal.initial_states import uniform_superposition
from kryanneal.krylov import (
    evolve_schedule_adaptive_m2,
    evolve_schedule_adaptive_richardson,
)


qutip = pytest.importorskip("qutip")


def _make_random_problem(n: int, seed: int) -> IsingProblem:
    """ランダム ``H_p_diag`` (実数 ``[-1, 1]``) と一様 ``h_x = 1`` で
    ``IsingProblem`` を作る.
    """
    rng = np.random.default_rng(seed)
    dim = 1 << n
    h_p = rng.uniform(-1.0, 1.0, size=dim).astype(np.float64)
    h_x = np.ones(n, dtype=np.float64)
    return IsingProblem(n=n, H_p_diag=h_p, h_x=h_x)


def _build_qutip_hamiltonian(h_x: np.ndarray, h_p_diag: np.ndarray, T: float) -> list:
    """linear schedule での QuTiP ``H(t)`` を組む (``test_reference_qutip.py``
    と同形). ``A(s) = 1 - s``, ``B(s) = s``, ``s = t/T``.
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


def _qutip_reference(h_x: np.ndarray, h_p_diag: np.ndarray, T: float) -> np.ndarray:
    """QuTiP ``sesolve`` で linear schedule の終端 ψ を取り出す."""
    n = h_x.shape[0]
    psi0 = uniform_superposition(n)
    h_t = _build_qutip_hamiltonian(h_x, h_p_diag, T)
    psi0_q = qutip.Qobj(psi0.reshape(-1, 1))
    sol = qutip.sesolve(
        h_t,
        psi0_q,
        np.array([0.0, T]),
        options={"atol": 1e-12, "rtol": 1e-10, "nsteps": 100000},
    )
    return sol.states[-1].full().ravel()


def _fidelity(psi_a: np.ndarray, psi_b: np.ndarray) -> float:
    return float(np.abs(np.vdot(psi_a, psi_b)) ** 2)


def test_adaptive_m2_matches_qutip() -> None:
    """``evolve_schedule_adaptive_m2`` smoke: linear schedule の n=4 問題で
    QuTiP との fidelity が ``> 1 - 1e-6``.

    ``tol_step = 1e-8`` (PI 既定値) で local error を抑え, smooth schedule
    の T=5 で 10 step 程度以上は走る前提.
    """
    n = 4
    T = 5.0
    prob = _make_random_problem(n, seed=20260513)
    sched = Schedule.linear(T=T)
    psi0 = uniform_superposition(n)

    psi_final, t_history, dt_history, n_rejects = evolve_schedule_adaptive_m2(
        h_x=prob.h_x,
        h_p_diag=prob.H_p_diag,
        schedule=sched,
        psi0=psi0,
        t0=0.0,
        t1=T,
        tol_step=1e-8,
        dt0=0.1,
    )
    psi_ref = _qutip_reference(prob.h_x, prob.H_p_diag, T)
    fid = _fidelity(psi_final, psi_ref)
    assert fid > 1 - 1e-6, f"adaptive_m2 fidelity too low: {fid} (1-fid={1 - fid})"
    # t_history は t0=0 で始まり, 最後の値が T 近傍.
    assert t_history[0] == 0.0
    assert abs(t_history[-1] - T) < 1e-12
    # dt_history は t_history より 1 要素少ない (accept された step 数).
    assert dt_history.shape[0] == t_history.shape[0] - 1
    assert n_rejects >= 0
    # 終端 ψ の L2 は 1 (unitary).
    assert abs(np.linalg.norm(psi_final) - 1.0) < 1e-10


def test_adaptive_richardson_matches_qutip() -> None:
    """``evolve_schedule_adaptive_richardson`` smoke: 同条件で M2 同等以上
    の fidelity を要求 (Richardson は 4 次推定子で local error が
    より厳密に抑制される).
    """
    n = 4
    T = 5.0
    prob = _make_random_problem(n, seed=20260513)
    sched = Schedule.linear(T=T)
    psi0 = uniform_superposition(n)

    psi_final, t_history, dt_history, n_rejects = evolve_schedule_adaptive_richardson(
        h_x=prob.h_x,
        h_p_diag=prob.H_p_diag,
        schedule=sched,
        psi0=psi0,
        t0=0.0,
        t1=T,
        tol_step=1e-8,
        dt0=0.1,
    )
    psi_ref = _qutip_reference(prob.h_x, prob.H_p_diag, T)
    fid = _fidelity(psi_final, psi_ref)
    assert fid > 1 - 1e-6, (
        f"adaptive_richardson fidelity too low: {fid} (1-fid={1 - fid})"
    )
    assert t_history[0] == 0.0
    assert abs(t_history[-1] - T) < 1e-12
    assert dt_history.shape[0] == t_history.shape[0] - 1
    assert n_rejects >= 0
    assert abs(np.linalg.norm(psi_final) - 1.0) < 1e-10


def test_adaptive_pi_dt_grows_with_loose_tolerance() -> None:
    """緩やかな schedule + 緩い ``tol_step`` で ``dt`` が ``dt0`` から
    ``dt_max`` 方向に増加することを確認.

    constant schedule (``A``, ``B`` 一定) なら CFM4:2 は 1 step で厳密
    (時間依存項なし) なので err は機械精度の rounding に落ち, PI 式は
    growth_max ベースで dt を伸ばす. dt の末尾値が ``dt0`` よりも有意に
    大きいことを確認する.
    """
    n = 3
    T = 5.0
    prob = _make_random_problem(n, seed=42)
    # constant schedule: A = 0.5, B = 0.5 で固定 (s(t) は default).
    sched = Schedule(T=T, A=lambda s: 0.5, B=lambda s: 0.5)
    psi0 = uniform_superposition(n)

    _, _, dt_history, _ = evolve_schedule_adaptive_richardson(
        h_x=prob.h_x,
        h_p_diag=prob.H_p_diag,
        schedule=sched,
        psi0=psi0,
        t0=0.0,
        t1=T,
        tol_step=1e-6,
        dt0=0.01,
        dt_max=1.0,
    )
    # 最初の dt は dt0 = 0.01, 後半は dt_max=1.0 に向けて増加していくはず.
    assert dt_history[0] == pytest.approx(0.01, abs=1e-15)
    assert dt_history[-1] > dt_history[0] * 5.0, (
        f"dt did not grow: history={dt_history}"
    )


def test_adaptive_rejects_oversized_dt0() -> None:
    """不自然に大きい ``dt0`` を渡すと最初の数 step で reject が起きる.

    smooth schedule で ``dt0 = 4.0`` (T と同オーダ) を渡し,
    ``n_rejects > 0`` を要求.
    """
    n = 3
    T = 5.0
    prob = _make_random_problem(n, seed=137)
    sched = Schedule.linear(T=T)
    psi0 = uniform_superposition(n)

    _, _, _, n_rejects = evolve_schedule_adaptive_richardson(
        h_x=prob.h_x,
        h_p_diag=prob.H_p_diag,
        schedule=sched,
        psi0=psi0,
        t0=0.0,
        t1=T,
        tol_step=1e-10,  # tight tol で reject が起きやすい状況にする
        dt0=4.0,
        dt_max=4.0,
        dt_min=1e-4,
    )
    assert n_rejects > 0, "expected at least one reject for oversized dt0"


def test_adaptive_max_rejects_raises_runtime_error() -> None:
    """``max_rejects`` を超える連続 reject で ``RuntimeError``.

    ``tol_step = 1e-30`` (機械精度より厳しい), ``dt_min = 1e-20``
    (実用上, halving で到達できない小ささ) にすると, PI は dt を半減し
    続けるが dt_min に届かないため accept しない. ``max_rejects = 3``
    で 3 連続 reject 後に RuntimeError が出る.
    """
    n = 3
    prob = _make_random_problem(n, seed=11)
    sched = Schedule.linear(T=5.0)
    psi0 = uniform_superposition(n)

    with pytest.raises(RuntimeError, match="max_rejects"):
        evolve_schedule_adaptive_richardson(
            h_x=prob.h_x,
            h_p_diag=prob.H_p_diag,
            schedule=sched,
            psi0=psi0,
            t0=0.0,
            t1=5.0,
            tol_step=1e-30,
            dt0=0.5,
            dt_min=1e-20,
            max_rejects=3,
        )


def test_adaptive_save_tlist_not_implemented() -> None:
    """``save_tlist is not None`` で ``NotImplementedError`` (Phase 5)."""
    n = 3
    prob = _make_random_problem(n, seed=11)
    sched = Schedule.linear(T=1.0)
    psi0 = uniform_superposition(n)

    with pytest.raises(NotImplementedError, match="save_tlist"):
        evolve_schedule_adaptive_m2(
            h_x=prob.h_x,
            h_p_diag=prob.H_p_diag,
            schedule=sched,
            psi0=psi0,
            t0=0.0,
            t1=1.0,
            save_tlist=np.array([0.5]),
        )
    with pytest.raises(NotImplementedError, match="save_tlist"):
        evolve_schedule_adaptive_richardson(
            h_x=prob.h_x,
            h_p_diag=prob.H_p_diag,
            schedule=sched,
            psi0=psi0,
            t0=0.0,
            t1=1.0,
            save_tlist=np.array([0.5]),
        )
