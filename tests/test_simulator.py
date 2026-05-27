"""``AnnealingSimulator`` (Phase 5 C3, issue #48) のテスト.

Acceptance:

* ``step(dt)`` を 1 回呼んで ``t`` / ``psi`` が ``run`` の単 step 結果と
  bit-identical に更新される (``m2`` / ``cfm4`` / Trotter 各経路).
* ``advance_to(t1, n_steps=N)`` の最終 ψ が ``QuantumAnnealer.run`` の
  ``psi_final`` と ``rel < 1e-13`` (実質的に完全一致) で合致する (固定
  dt 経路).
* ``measure(observable)`` が ``observable.expectation(simulator.psi)``
  と一致.
* ``psi`` プロパティが defensive copy を返し, 戻り値の mutation が
  内部状態に影響しない.
* adaptive method (``cfm4_adaptive_richardson_krylov``) で ``step(dt)`` を
  呼ぶと dt が proposal 扱いになり PI controller の accept/reject 動作が
  観測される (大き過ぎる dt を渡すと内部で sub-step 化される).
* ``measure`` で ``Observable`` 以外を渡すと ``TypeError``.
* ``n_matvec`` の累積が ``QuantumAnnealer.run`` と一致 (固定 dt 経路).
"""

from __future__ import annotations

import numpy as np
import pytest

from maqina import (
    AnnealingSimulator,
    IsingProblem,
    Observable,
    QuantumAnnealer,
    Schedule,
)
from maqina.initial_states import uniform_superposition


def _ferromagnetic_chain_h_p_diag(n: int) -> np.ndarray:
    """強磁性 1D chain ``-Σ σ_i σ_{i+1}`` の対角ベクトル.

    基底状態は ``|0...0⟩`` と ``|1...1⟩`` の 2 重縮退 (test_annealer.py と
    同じ生成式).
    """
    dim = 1 << n
    h_p = np.zeros(dim, dtype=np.float64)
    for x in range(dim):
        bits = [(x >> i) & 1 for i in range(n)]
        spins = [1 - 2 * b for b in bits]
        h_p[x] = float(-sum(spins[i] * spins[i + 1] for i in range(n - 1)))
    return h_p


def _build_problem(n: int = 4) -> tuple[IsingProblem, np.ndarray]:
    h_x = np.ones(n, dtype=np.float64)
    prob = IsingProblem(
        n=n,
        H_p_diag=_ferromagnetic_chain_h_p_diag(n),
    )
    return prob, h_x


# ---------------------------------------------------------------------------
# 初期状態 / プロパティ
# ---------------------------------------------------------------------------


def test_init_stores_t0_psi0_and_zero_n_matvec() -> None:
    """``__init__`` 直後の状態 (``t == t0``, ψ ≈ psi0, n_matvec == 0)."""
    n = 3
    prob, h_x = _build_problem(n)
    sched = Schedule.linear(T=1.0, h_x=np.ones(n, dtype=np.float64))
    psi0 = uniform_superposition(n)

    sim = AnnealingSimulator(prob, sched, psi0, 0.5, method="m2")
    assert sim.t == 0.5
    assert sim.method == "m2"
    assert sim.n_matvec == 0
    assert np.allclose(sim.psi, psi0)
    assert sim.psi.shape == (1 << n,)
    assert sim.psi.dtype == np.complex128


def test_psi_property_returns_defensive_copy() -> None:
    """``sim.psi`` 戻り値への mutation が内部状態に影響しない."""
    n = 3
    prob, h_x = _build_problem(n)
    sched = Schedule.linear(T=1.0, h_x=h_x)
    psi0 = uniform_superposition(n)

    sim = AnnealingSimulator(prob, sched, psi0, 0.0, method="m2")
    psi_view = sim.psi
    psi_view[0] = 999.0 + 0.0j
    # 内部状態は不変, 別 view を取り直すと書き換えが反映されない.
    psi_again = sim.psi
    assert psi_again[0] != 999.0 + 0.0j
    assert np.allclose(psi_again, psi0)


def test_init_copies_psi0_so_external_mutation_does_not_leak() -> None:
    """構築後に外部 psi0 を書き換えても Simulator 内部状態は変わらない."""
    n = 3
    prob, h_x = _build_problem(n)
    sched = Schedule.linear(T=1.0, h_x=h_x)
    psi0 = uniform_superposition(n)
    psi0_snapshot = psi0.copy()

    sim = AnnealingSimulator(prob, sched, psi0, 0.0, method="m2")
    psi0[0] = 999.0 + 0.0j
    assert np.allclose(sim.psi, psi0_snapshot)


# ---------------------------------------------------------------------------
# step(dt) — 1 step の正しさ
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method",
    ["m2", "trotter", "trotter_suzuki4", "cfm4"],
)
def test_step_advances_one_fixed_dt_step(method: str) -> None:
    """``step(dt)`` 1 回が ``run(..., n_steps=1, [t0, t0+dt])`` と
    bit-identical な ψ を生む (固定 dt 経路).
    """
    n = 4
    prob, h_x = _build_problem(n)
    sched = Schedule.linear(T=5.0, h_x=h_x)
    psi0 = uniform_superposition(n)
    dt = 0.25

    ann = QuantumAnnealer(prob, sched)
    res = ann.run(psi0, 0.0, dt, method=method, n_steps=1)

    sim = AnnealingSimulator(prob, sched, psi0, 0.0, method=method)
    sim.step(dt)
    assert sim.t == pytest.approx(dt, rel=0, abs=1e-15)
    assert np.allclose(sim.psi, res.psi_final, rtol=0.0, atol=1e-15)


def test_step_dt_must_be_positive() -> None:
    """``step(dt <= 0)`` は ``ValueError``."""
    n = 3
    prob, h_x = _build_problem(n)
    sched = Schedule.linear(T=1.0, h_x=h_x)
    psi0 = uniform_superposition(n)
    sim = AnnealingSimulator(prob, sched, psi0, 0.0, method="m2")
    with pytest.raises(ValueError, match="dt must be > 0"):
        sim.step(0.0)
    with pytest.raises(ValueError, match="dt must be > 0"):
        sim.step(-0.1)


# ---------------------------------------------------------------------------
# advance_to — run との完全一致
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "n_steps"),
    [
        ("m2", 50),
        ("trotter", 50),
        ("trotter_suzuki4", 20),
        ("cfm4", 30),
    ],
)
def test_advance_to_matches_run_fixed_dt(method: str, n_steps: int) -> None:
    """``advance_to(t1, n_steps=N)`` の最終 ψ が ``run(..., n_steps=N)``
    の ``psi_final`` と完全一致 (``rel < 1e-13``).
    """
    n = 4
    prob, h_x = _build_problem(n)
    sched = Schedule.linear(T=3.0, h_x=h_x)
    psi0 = uniform_superposition(n)
    t1 = sched.T

    ann = QuantumAnnealer(prob, sched)
    res = ann.run(psi0, 0.0, t1, method=method, n_steps=n_steps)

    sim = ann.create_simulator(psi0, 0.0, method=method)
    sim.advance_to(t1, n_steps=n_steps)
    assert sim.t == pytest.approx(t1)
    # bit-identical を期待 (同じ driver を同じ schedule 評価点で呼んでいる).
    rel = float(np.linalg.norm(sim.psi - res.psi_final) / np.linalg.norm(res.psi_final))
    assert rel < 1e-13, f"method={method} rel={rel:.3e}"
    # n_matvec も完全一致.
    assert sim.n_matvec == res.n_matvec


def test_advance_to_t_target_must_be_greater_than_current() -> None:
    n = 3
    prob, h_x = _build_problem(n)
    sched = Schedule.linear(T=1.0, h_x=h_x)
    psi0 = uniform_superposition(n)
    sim = AnnealingSimulator(prob, sched, psi0, 0.5, method="m2")
    with pytest.raises(ValueError, match="t_target must be"):
        sim.advance_to(0.5, n_steps=1)
    with pytest.raises(ValueError, match="t_target must be"):
        sim.advance_to(0.4, n_steps=1)


def test_advance_to_fixed_dt_requires_n_steps() -> None:
    n = 3
    prob, h_x = _build_problem(n)
    sched = Schedule.linear(T=1.0, h_x=h_x)
    psi0 = uniform_superposition(n)
    sim = AnnealingSimulator(prob, sched, psi0, 0.0, method="cfm4")
    with pytest.raises(ValueError, match="n_steps is required"):
        sim.advance_to(1.0)
    with pytest.raises(ValueError, match="n_steps must be a positive integer"):
        sim.advance_to(1.0, n_steps=0)


def test_advance_to_adaptive_rejects_n_steps() -> None:
    n = 3
    prob, h_x = _build_problem(n)
    sched = Schedule.linear(T=1.0, h_x=h_x)
    psi0 = uniform_superposition(n)
    sim = AnnealingSimulator(
        prob, sched, psi0, 0.0, method="cfm4_adaptive_richardson_krylov"
    )
    with pytest.raises(ValueError, match="n_steps must be None"):
        sim.advance_to(1.0, n_steps=10)


def test_step_loop_matches_advance_to_fixed_dt() -> None:
    """``step(dt)`` を等間隔で n 回呼んだ累積結果が ``advance_to(t1, n_steps=n)``
    と一致 (両者とも同じ driver call を経由するため machine precision で一致).
    """
    n = 4
    prob, h_x = _build_problem(n)
    sched = Schedule.linear(T=2.0, h_x=h_x)
    psi0 = uniform_superposition(n)
    n_steps = 20
    dt = 2.0 / n_steps

    sim_step = AnnealingSimulator(prob, sched, psi0, 0.0, method="cfm4")
    for _ in range(n_steps):
        sim_step.step(dt)

    sim_advance = AnnealingSimulator(prob, sched, psi0, 0.0, method="cfm4")
    sim_advance.advance_to(2.0, n_steps=n_steps)

    # step ループは each step が独立 driver call なため schedule の浮動小数点
    # 端数累積が異なる可能性があり, advance_to (1 driver call で内部 n_steps
    # ループ) と完全 bit-identical にはならないが, 数値的に同等 (rel < 1e-12).
    diff = float(np.linalg.norm(sim_step.psi - sim_advance.psi))
    assert diff < 1e-12, f"step-loop vs advance_to diff={diff:.3e}"
    assert sim_step.n_matvec == sim_advance.n_matvec


# ---------------------------------------------------------------------------
# measure
# ---------------------------------------------------------------------------


def test_measure_matches_observable_expectation() -> None:
    n = 4
    prob, h_x = _build_problem(n)
    sched = Schedule.linear(T=1.0, h_x=h_x)
    psi0 = uniform_superposition(n)
    sim = AnnealingSimulator(prob, sched, psi0, 0.0, method="cfm4")
    sim.advance_to(1.0, n_steps=10)

    obs_mag = Observable.magnetization(n)
    obs_ene = Observable.ising_energy(prob)
    assert sim.measure(obs_mag) == pytest.approx(obs_mag.expectation(sim.psi))
    assert sim.measure(obs_ene) == pytest.approx(obs_ene.expectation(sim.psi))


def test_measure_rejects_non_observable() -> None:
    n = 3
    prob, h_x = _build_problem(n)
    sched = Schedule.linear(T=1.0, h_x=h_x)
    psi0 = uniform_superposition(n)
    sim = AnnealingSimulator(prob, sched, psi0, 0.0, method="m2")
    with pytest.raises(TypeError, match="observable must be an Observable"):
        sim.measure(np.zeros(1 << n))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        sim.measure(prob.H_p_diag)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# adaptive
# ---------------------------------------------------------------------------


def test_adaptive_step_advances_by_exactly_dt() -> None:
    """adaptive ``step(dt)`` 後の ``_t`` は exactly ``+dt``."""
    n = 4
    prob, h_x = _build_problem(n)
    sched = Schedule.linear(T=5.0, h_x=h_x)
    psi0 = uniform_superposition(n)
    sim = AnnealingSimulator(
        prob, sched, psi0, 0.0, method="cfm4_adaptive_richardson_krylov"
    )
    sim.step(0.3)
    assert sim.t == pytest.approx(0.3)
    assert sim.n_matvec > 0


def test_adaptive_step_large_dt_triggers_sub_stepping() -> None:
    """adaptive ``step(dt)`` で大きすぎる dt を渡すと PI controller が dt を
    縮めて sub-step 化することで extra m_eff cost が発生する.

    accept/reject 動作を観測するため, atol を厳しく (``1e-12``), 大きな dt
    (``T=1`` で全区間) を proposal にする. 単 step では満たせないので
    driver が複数 sub-step に分割し, n_matvec が単純な ``6m`` を上回る.
    """
    n = 4
    prob, h_x = _build_problem(n)
    sched = Schedule.linear(T=1.0, h_x=h_x)
    psi0 = uniform_superposition(n)
    sim = AnnealingSimulator(
        prob,
        sched,
        psi0,
        0.0,
        method="cfm4_adaptive_richardson_krylov",
        atol=1e-12,
        m=24,
    )
    sim.step(1.0)
    assert sim.t == pytest.approx(1.0)
    # 単 step なら m_eff_sum ≤ 6m = 144. PI sub-stepping が発生していれば
    # n_matvec はこれを大きく上回る. 厳しい atol で複数 sub-step を強制.
    assert sim.n_matvec > 144, (
        f"expected adaptive PI sub-stepping (n_matvec > 144), "
        f"got n_matvec={sim.n_matvec}"
    )


def test_adaptive_advance_to_matches_run_psi() -> None:
    """adaptive ``advance_to`` 後の ψ が ``QuantumAnnealer.run`` の adaptive
    経路と一致する (driver call が同じなので bit-identical).
    """
    n = 4
    prob, h_x = _build_problem(n)
    sched = Schedule.linear(T=2.0, h_x=h_x)
    psi0 = uniform_superposition(n)

    ann = QuantumAnnealer(prob, sched)
    res = ann.run(psi0, 0.0, 2.0, method="cfm4_adaptive_richardson_krylov", atol=1e-8)

    sim = ann.create_simulator(
        psi0, 0.0, method="cfm4_adaptive_richardson_krylov", atol=1e-8
    )
    sim.advance_to(2.0)
    rel = float(np.linalg.norm(sim.psi - res.psi_final) / np.linalg.norm(res.psi_final))
    # 同じ driver を同じ引数 (auto-resolved dt_init/dt_max 含め) で呼ぶので
    # bit-identical を期待.
    assert rel < 1e-13, f"adaptive sim vs run rel={rel:.3e}"
    assert sim.n_matvec == res.n_matvec


# ---------------------------------------------------------------------------
# 入力検証
# ---------------------------------------------------------------------------


def test_init_rejects_unsupported_method() -> None:
    n = 3
    prob, h_x = _build_problem(n)
    sched = Schedule.linear(T=1.0, h_x=h_x)
    psi0 = uniform_superposition(n)
    with pytest.raises(NotImplementedError, match="not supported"):
        AnnealingSimulator(prob, sched, psi0, 0.0, method="bogus")  # type: ignore[arg-type]


def test_init_validates_psi0() -> None:
    """``_validate_psi0`` 経路の伝播確認 (shape / dtype / 非正規化)."""
    n = 3
    prob, h_x = _build_problem(n)
    sched = Schedule.linear(T=1.0, h_x=h_x)

    bad_shape = np.zeros(1 << (n - 1), dtype=np.complex128)
    with pytest.raises(ValueError, match="psi0 shape mismatch"):
        AnnealingSimulator(prob, sched, bad_shape, 0.0, method="m2")

    bad_dtype = np.ones(1 << n, dtype=np.float64) / np.sqrt(1 << n)
    with pytest.raises(ValueError, match="psi0 dtype"):
        AnnealingSimulator(prob, sched, bad_dtype, 0.0, method="m2")

    bad_norm = np.zeros(1 << n, dtype=np.complex128)
    bad_norm[0] = 2.0 + 0.0j
    with pytest.raises(ValueError, match="L2-normalized"):
        AnnealingSimulator(prob, sched, bad_norm, 0.0, method="m2")


def test_init_validates_m_and_propagator_tol() -> None:
    n = 3
    prob, h_x = _build_problem(n)
    sched = Schedule.linear(T=1.0, h_x=h_x)
    psi0 = uniform_superposition(n)
    with pytest.raises(ValueError, match="m must be a positive integer"):
        AnnealingSimulator(prob, sched, psi0, 0.0, method="m2", m=0)
    with pytest.raises(ValueError, match="propagator_tol must be"):
        AnnealingSimulator(prob, sched, psi0, 0.0, method="m2", propagator_tol=-1.0)


@pytest.mark.parametrize("param_name", ["atol", "dt_init", "dt_max", "m_max"])
def test_init_rejects_adaptive_params_on_fixed_dt_method(param_name: str) -> None:
    """固定 dt method で adaptive 専用パラメータを指定すると ``ValueError``."""
    n = 3
    prob, h_x = _build_problem(n)
    sched = Schedule.linear(T=1.0, h_x=h_x)
    psi0 = uniform_superposition(n)
    kwargs: dict[str, float | int] = {param_name: 1e-5 if "m" not in param_name else 16}
    with pytest.raises(ValueError, match=f"{param_name} is only valid"):
        AnnealingSimulator(prob, sched, psi0, 0.0, method="cfm4", **kwargs)  # type: ignore[arg-type]


def test_init_validates_adaptive_param_positivity() -> None:
    n = 3
    prob, h_x = _build_problem(n)
    sched = Schedule.linear(T=1.0, h_x=h_x)
    psi0 = uniform_superposition(n)
    base_kwargs: dict[str, object] = {"method": "cfm4_adaptive_richardson_krylov"}
    with pytest.raises(ValueError, match="atol must be > 0"):
        AnnealingSimulator(prob, sched, psi0, 0.0, atol=-1.0, **base_kwargs)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="dt_init must be > 0"):
        AnnealingSimulator(prob, sched, psi0, 0.0, dt_init=0.0, **base_kwargs)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="dt_max must be > 0"):
        AnnealingSimulator(prob, sched, psi0, 0.0, dt_max=-0.1, **base_kwargs)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="m_max must be a positive integer"):
        AnnealingSimulator(prob, sched, psi0, 0.0, m_max=0, **base_kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 観測量との統合 (中間時刻 measure 経由でアニーリングの GS 収束を確認)
# ---------------------------------------------------------------------------


def test_measure_during_anneal_tracks_ising_energy_decreasing() -> None:
    """強磁性 chain (T=10 linear) で中間 measure を挟むと, ising_energy
    期待値が時間とともに下降する (基底状態に近付くため).
    """
    n = 4
    prob, h_x = _build_problem(n)
    sched = Schedule.linear(T=10.0, h_x=h_x)
    psi0 = uniform_superposition(n)
    obs = Observable.ising_energy(prob)

    sim = AnnealingSimulator(prob, sched, psi0, 0.0, method="cfm4")
    energies = [sim.measure(obs)]
    for i in range(1, 11):
        sim.advance_to(float(i), n_steps=20)
        energies.append(sim.measure(obs))

    # 初期 (+) 状態は H_p に対して期待値 ≈ 0, 終端 (≈GS) は ≈ -3.0 を目指す.
    assert energies[0] == pytest.approx(0.0, abs=1e-10)
    assert energies[-1] < -2.5, f"final energy not low enough: {energies[-1]}"
    # 全体としては monotonic に近い下降 (厳密 monotonic は要求しない).
    assert energies[-1] < energies[0]
