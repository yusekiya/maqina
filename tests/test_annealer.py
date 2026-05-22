"""``QuantumAnnealer.run`` の end-to-end smoke test (Phase 1 / Phase 2 / Phase 3 / Phase 4).

issue #8 / #21 / #22 / #32 / #39 の acceptance:

* ``QuantumAnnealer(prob, sched).run(psi0, 0, T, method="m2", n_steps=200)``
  の戻り値が ``QuantumResult`` で, 線形 schedule で十分長い ``T`` を取ると
  基底状態到達確率がしきい値以上.
* ``method="trotter"`` 経路 (Phase 2 C3) でも同じ smoke を満たし,
  ``n_matvec = n_steps * (N + 1)`` の見積もりが返る.
* ``method="trotter_suzuki4"`` 経路 (Phase 2 C4) でも同じ smoke を満たし,
  ``n_matvec = n_steps * 5 * (N + 1)`` の見積もりが返る.
* ``method="cfm4"`` 経路 (Phase 3 C2) でも同じ smoke を満たし,
  ``n_matvec = n_steps * 2 * m`` の見積もりが返る.
* ``method="cfm4_adaptive_richardson_krylov"`` 経路 (Phase 4 C3) でも同じ smoke
  を満たし, ``n_matvec = n_steps_actual * 6 * m`` の見積もりと
  ``result.success`` / ``result.method`` / ``result.n_steps_actual``
  の新フィールドが正しく返る.
* 公開 API (psi0 検証, method/save_tlist の NotImplementedError) が
  仕様どおりに raise する.
"""

from __future__ import annotations

import numpy as np
import pytest

from kryanneal import IsingProblem, QuantumAnnealer, QuantumResult, Schedule
from kryanneal.initial_states import uniform_superposition


def _ferromagnetic_chain_h_p_diag(n: int) -> np.ndarray:
    """1D 反強磁性 (J_ij = +1, 隣接) Ising chain の対角ベクトル.

    ``H_p = -Σ_{<i,j>} J_ij σ_i σ_j``, ``J_{i,i+1} = +1`` (反強磁性) と
    すると, n=4 の chain で基底状態は全反転 / 1-deviation の縮退. ここでは
    強磁性 (``J_{i,i+1} = -1``) でテストし, 基底状態が ``|0...0⟩`` と
    ``|1...1⟩`` の 2 重縮退になる単純設定を使う.
    """
    dim = 1 << n
    h_p = np.zeros(dim, dtype=np.float64)
    for x in range(dim):
        bits = [(x >> i) & 1 for i in range(n)]
        spins = [1 - 2 * b for b in bits]
        # H_p = -Σ_{<i,i+1>} σ_i σ_{i+1} (open chain, 強磁性)
        energy = -sum(spins[i] * spins[i + 1] for i in range(n - 1))
        h_p[x] = float(energy)
    return h_p


def _ground_state_probability(psi: np.ndarray, h_p_diag: np.ndarray) -> float:
    """``|psi[k]|^2`` を ``h_p_diag`` 最小値を取る k 全部について和算する.

    縮退があるときは全縮退状態の合計確率を返す.
    """
    e_min = float(np.min(h_p_diag))
    mask = h_p_diag == e_min
    return float(np.sum(np.abs(psi[mask]) ** 2))


def test_run_returns_quantum_result_with_phase1_fields() -> None:
    """戻り値が ``QuantumResult`` で Phase 1 subset + Phase 4 追加フィールドを持つ."""
    n = 3
    prob = IsingProblem(
        n=n,
        H_p_diag=_ferromagnetic_chain_h_p_diag(n),
        h_x=np.ones(n, dtype=np.float64),
    )
    sched = Schedule.linear(T=1.0)
    psi0 = uniform_superposition(n)
    ann = QuantumAnnealer(prob, sched)
    res = ann.run(psi0, 0.0, 1.0, method="m2", n_steps=20)

    assert isinstance(res, QuantumResult)
    assert res.psi_final.shape == (1 << n,)
    assert res.psi_final.dtype == np.complex128
    assert res.t_history is None
    assert res.observables_history == {}
    assert res.n_steps == 20
    # n_matvec は Phase 1 では n_steps × m の見積もり (m=24 デフォルト).
    assert res.n_matvec == 20 * 24
    # Phase 4 で追加した新フィールド (固定 dt 経路では n_steps と一致).
    assert res.success is True
    assert res.method == "m2"
    assert res.n_steps_actual == 20
    # propagator は unitary なので ‖psi_final‖ ≈ 1.
    assert abs(np.linalg.norm(res.psi_final) - 1.0) < 1e-10


def test_run_reaches_ground_state_for_long_anneal() -> None:
    """十分長い ``T`` で線形 schedule の基底状態到達確率が高くなる.

    n=4 の強磁性 chain (``J = -1``, 縮退基底 ``|0000⟩`` / ``|1111⟩``) で
    ``T=10`` の linear schedule を実行し, 基底状態合計確率が 0.95 を
    超えることを確認する. 完全断熱でないので 1.0 には達しないが
    smoke check として十分.
    """
    n = 4
    prob = IsingProblem(
        n=n,
        H_p_diag=_ferromagnetic_chain_h_p_diag(n),
        h_x=np.ones(n, dtype=np.float64),
    )
    sched = Schedule.linear(T=10.0)
    psi0 = uniform_superposition(n)
    ann = QuantumAnnealer(prob, sched)
    res = ann.run(psi0, 0.0, sched.T, method="m2", n_steps=200)

    p_gs = _ground_state_probability(res.psi_final, prob.H_p_diag)
    assert p_gs > 0.95, f"ground state probability too low: {p_gs}"


def test_run_trotter_smoke_and_ground_state() -> None:
    """``method="trotter"`` の smoke + GS 到達確率テスト (Phase 2 C3).

    n=4, T=10 linear schedule の強磁性 chain (縮退基底 ``|0000⟩`` /
    ``|1111⟩``) で ``n_steps=200`` を取り, 終端波動関数の基底状態
    合計確率が 0.95 を超えることを確認する. Trotter は M2 と同じ
    O(dt^3) LTE のため, M2 と同等の精度を実現できる.

    あわせて ``n_matvec = n_steps × (N + 1)`` の Phase 2 規約 (Trotter
    は dim-walk 見積もり; ``docs/design/04-python-api.md`` §4.4) を検証する.
    """
    n = 4
    prob = IsingProblem(
        n=n,
        H_p_diag=_ferromagnetic_chain_h_p_diag(n),
        h_x=np.ones(n, dtype=np.float64),
    )
    sched = Schedule.linear(T=10.0)
    psi0 = uniform_superposition(n)
    ann = QuantumAnnealer(prob, sched)
    res = ann.run(psi0, 0.0, sched.T, method="trotter", n_steps=200)

    assert isinstance(res, QuantumResult)
    assert res.psi_final.shape == (1 << n,)
    assert res.psi_final.dtype == np.complex128
    assert res.n_steps == 200
    # Trotter 規約: n_matvec = n_steps × (N + 1).
    assert res.n_matvec == 200 * (n + 1)
    # propagator は unitary.
    assert abs(np.linalg.norm(res.psi_final) - 1.0) < 1e-10

    p_gs = _ground_state_probability(res.psi_final, prob.H_p_diag)
    assert p_gs > 0.95, f"ground state probability too low (trotter): {p_gs}"


def test_run_trotter_suzuki4_smoke_and_ground_state() -> None:
    """``method="trotter_suzuki4"`` の smoke + GS 到達確率テスト (Phase 2 C4).

    n=4, T=10 linear schedule の強磁性 chain (縮退基底 ``|0000⟩`` /
    ``|1111⟩``) で ``n_steps=200`` を取り, 終端波動関数の基底状態
    合計確率が 0.95 を超えることを確認する. Suzuki S_4 は O(dt^5) LTE
    なので Strang よりさらに精度が高い (M2/Strang と同じく十分到達).

    あわせて ``n_matvec = n_steps × 5 × (N + 1)`` の Phase 2 C4 規約
    (5 sub-step × Strang per-step コスト) を検証する.
    """
    n = 4
    prob = IsingProblem(
        n=n,
        H_p_diag=_ferromagnetic_chain_h_p_diag(n),
        h_x=np.ones(n, dtype=np.float64),
    )
    sched = Schedule.linear(T=10.0)
    psi0 = uniform_superposition(n)
    ann = QuantumAnnealer(prob, sched)
    res = ann.run(psi0, 0.0, sched.T, method="trotter_suzuki4", n_steps=200)

    assert isinstance(res, QuantumResult)
    assert res.psi_final.shape == (1 << n,)
    assert res.psi_final.dtype == np.complex128
    assert res.n_steps == 200
    # Suzuki S_4 規約: n_matvec = n_steps × 5 × (N + 1).
    assert res.n_matvec == 200 * 5 * (n + 1)
    assert abs(np.linalg.norm(res.psi_final) - 1.0) < 1e-10

    p_gs = _ground_state_probability(res.psi_final, prob.H_p_diag)
    assert p_gs > 0.95, f"ground state probability too low (trotter_suzuki4): {p_gs}"


def test_run_cfm4_smoke_and_ground_state() -> None:
    """``method="cfm4"`` の smoke + GS 到達確率テスト (Phase 3 C2).

    n=4, T=10 linear schedule の強磁性 chain (縮退基底 ``|0000⟩`` /
    ``|1111⟩``) で ``n_steps=200`` を取り, 終端波動関数の基底状態
    合計確率が 0.95 を超えることを確認する. CFM4:2 は ``O(dt^5)`` LTE で
    Suzuki S_4 と同じ局所オーダなので, ``n_steps=200`` で十分到達する.

    あわせて ``n_matvec = n_steps × 2 × m`` の Phase 3 C2 規約 (CFM4:2 は
    1 step あたり Lanczos 2 回) を検証する.
    """
    n = 4
    prob = IsingProblem(
        n=n,
        H_p_diag=_ferromagnetic_chain_h_p_diag(n),
        h_x=np.ones(n, dtype=np.float64),
    )
    sched = Schedule.linear(T=10.0)
    psi0 = uniform_superposition(n)
    ann = QuantumAnnealer(prob, sched)
    res = ann.run(psi0, 0.0, sched.T, method="cfm4", n_steps=200)

    assert isinstance(res, QuantumResult)
    assert res.psi_final.shape == (1 << n,)
    assert res.psi_final.dtype == np.complex128
    assert res.n_steps == 200
    # CFM4:2 規約: n_matvec = n_steps × 2 × m (m=24 デフォルト).
    assert res.n_matvec == 200 * 2 * 24
    # propagator は unitary.
    assert abs(np.linalg.norm(res.psi_final) - 1.0) < 1e-10

    p_gs = _ground_state_probability(res.psi_final, prob.H_p_diag)
    assert p_gs > 0.95, f"ground state probability too low (cfm4): {p_gs}"


def test_run_cfm4_adaptive_richardson_smoke_and_ground_state() -> None:
    """``method="cfm4_adaptive_richardson_krylov"`` smoke + GS 到達確率 (Phase 4 C3).

    n=4, T=10 linear schedule の強磁性 chain (縮退基底 ``|0000⟩`` /
    ``|1111⟩``) で adaptive 経路を実行し, 終端波動関数の基底状態
    合計確率が 0.95 を超えることを確認する.

    あわせて Phase 4 規約を検証:

    * ``n_matvec = n_steps_actual * 6 * m`` (full 2m + half×2 4m).
    * ``result.success == True``, ``result.method == "cfm4_adaptive_richardson_krylov"``,
      ``result.n_steps_actual is not None``.
    """
    n = 4
    prob = IsingProblem(
        n=n,
        H_p_diag=_ferromagnetic_chain_h_p_diag(n),
        h_x=np.ones(n, dtype=np.float64),
    )
    sched = Schedule.linear(T=10.0)
    psi0 = uniform_superposition(n)
    ann = QuantumAnnealer(prob, sched)
    res = ann.run(
        psi0,
        0.0,
        sched.T,
        method="cfm4_adaptive_richardson_krylov",
        atol=1e-8,
        dt_init=0.1,
    )

    assert isinstance(res, QuantumResult)
    assert res.psi_final.shape == (1 << n,)
    assert res.psi_final.dtype == np.complex128
    assert res.success is True
    assert res.method == "cfm4_adaptive_richardson_krylov"
    assert res.n_steps_actual is not None
    assert res.n_steps_actual >= 1
    # adaptive 経路は要求 step 数を持たないので n_steps = n_steps_actual.
    assert res.n_steps == res.n_steps_actual
    # Phase 4 follow-up (issue #52 A) 規約: n_matvec は m_eff_history の総和
    # (実 Lanczos コスト). 早期打切なしの upper bound は n_steps_actual × 6 × m
    # (m=24 デフォルト) だが, smooth schedule では m_eff < m が一般的なので
    # 上限のみ assert する.
    assert 1 <= res.n_matvec <= res.n_steps_actual * 6 * 24
    # propagator は unitary.
    assert abs(np.linalg.norm(res.psi_final) - 1.0) < 1e-10

    p_gs = _ground_state_probability(res.psi_final, prob.H_p_diag)
    assert p_gs > 0.95, (
        f"ground state probability too low (cfm4_adaptive_richardson_krylov): {p_gs}"
    )


def test_run_rejects_unsupported_method() -> None:
    """サポート対象外の ``method`` で ``NotImplementedError``.

    Phase 4 で ``cfm4_adaptive_richardson_krylov`` が追加されたため, 旧仕様の
    sentinel ``"adaptive_m2"`` (本リリースでは API 表面に出さない) を
    試して raise を確認する.
    """
    n = 3
    prob = IsingProblem(
        n=n,
        H_p_diag=_ferromagnetic_chain_h_p_diag(n),
        h_x=np.ones(n, dtype=np.float64),
    )
    sched = Schedule.linear(T=1.0)
    psi0 = uniform_superposition(n)
    ann = QuantumAnnealer(prob, sched)
    with pytest.raises(NotImplementedError):
        ann.run(psi0, 0.0, 1.0, method="adaptive_m2", n_steps=10)  # type: ignore[arg-type]


def test_run_save_tlist_snapshot_smoke() -> None:
    """Phase 5 (issue #47): ``save_tlist`` を渡すと ``QuantumResult.times``
    と ``observables_history`` / ``states`` が記録される.

    Phase 4 までの ``NotImplementedError`` 経路は除去された (issue #47 で
    有効化). 固定 dt 経路の m2 で smoke 確認.
    """
    from kryanneal import Observable

    n = 3
    prob = IsingProblem(
        n=n,
        H_p_diag=_ferromagnetic_chain_h_p_diag(n),
        h_x=np.ones(n, dtype=np.float64),
    )
    sched = Schedule.linear(T=1.0)
    psi0 = uniform_superposition(n)
    ann = QuantumAnnealer(prob, sched)
    save_tlist = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float64)
    res = ann.run(
        psi0,
        0.0,
        1.0,
        method="m2",
        n_steps=10,
        observables={"M_z": Observable.magnetization(n)},
        save_tlist=save_tlist,
        store_states=True,
    )
    assert res.times is not None
    np.testing.assert_array_equal(res.times, save_tlist)
    np.testing.assert_array_equal(res.t_history, save_tlist)
    assert "M_z" in res.observables_history
    assert res.observables_history["M_z"].shape == (5,)
    assert res.states is not None
    assert res.states.shape == (5, 1 << n)
    # 先頭は psi0 そのもの.
    np.testing.assert_array_equal(res.states[0], psi0)
    # 末尾は psi_final と一致 (target=t1 を踏むので).
    np.testing.assert_array_equal(res.states[-1], res.psi_final)
    # probabilities は常に計算される.
    assert res.probabilities is not None
    np.testing.assert_array_almost_equal(res.probabilities, np.abs(res.psi_final) ** 2)


def test_run_observables_without_save_tlist_raises() -> None:
    """Phase 5 (issue #47): ``save_tlist=None`` で ``observables`` 指定は
    最節約モードに矛盾するため ``ValueError``.
    """
    from kryanneal import Observable

    n = 3
    prob = IsingProblem(
        n=n,
        H_p_diag=_ferromagnetic_chain_h_p_diag(n),
        h_x=np.ones(n, dtype=np.float64),
    )
    sched = Schedule.linear(T=1.0)
    psi0 = uniform_superposition(n)
    ann = QuantumAnnealer(prob, sched)
    with pytest.raises(ValueError, match="save_tlist"):
        ann.run(
            psi0,
            0.0,
            1.0,
            method="m2",
            n_steps=10,
            observables={"M_z": Observable.magnetization(n)},
        )


def test_run_store_states_without_save_tlist_raises() -> None:
    """Phase 5 (issue #47): ``save_tlist=None`` で ``store_states=True`` は
    最節約モードに矛盾するため ``ValueError``.
    """
    n = 3
    prob = IsingProblem(
        n=n,
        H_p_diag=_ferromagnetic_chain_h_p_diag(n),
        h_x=np.ones(n, dtype=np.float64),
    )
    sched = Schedule.linear(T=1.0)
    psi0 = uniform_superposition(n)
    ann = QuantumAnnealer(prob, sched)
    with pytest.raises(ValueError, match="save_tlist"):
        ann.run(
            psi0,
            0.0,
            1.0,
            method="m2",
            n_steps=10,
            store_states=True,
        )


def test_run_validates_psi0_normalization() -> None:
    """L2-normalize 違反で ``ValueError``."""
    n = 3
    prob = IsingProblem(
        n=n,
        H_p_diag=_ferromagnetic_chain_h_p_diag(n),
        h_x=np.ones(n, dtype=np.float64),
    )
    sched = Schedule.linear(T=1.0)
    ann = QuantumAnnealer(prob, sched)
    psi0 = uniform_superposition(n) * 2.0  # ‖psi0‖ = 2
    with pytest.raises(ValueError, match="L2-normalized"):
        ann.run(psi0, 0.0, 1.0, method="m2", n_steps=10)


def test_run_validates_psi0_shape() -> None:
    """``psi0`` の shape が ``(2**n,)`` 以外で ``ValueError``."""
    n = 3
    prob = IsingProblem(
        n=n,
        H_p_diag=_ferromagnetic_chain_h_p_diag(n),
        h_x=np.ones(n, dtype=np.float64),
    )
    sched = Schedule.linear(T=1.0)
    ann = QuantumAnnealer(prob, sched)
    psi0_wrong = np.ones(7, dtype=np.complex128) / np.sqrt(7)
    with pytest.raises(ValueError, match="shape"):
        ann.run(psi0_wrong, 0.0, 1.0, method="m2", n_steps=10)


def test_constructor_validates_m_and_tol() -> None:
    """``m < 1`` / ``krylov_tol < 0`` で ``ValueError``."""
    n = 3
    prob = IsingProblem(
        n=n,
        H_p_diag=_ferromagnetic_chain_h_p_diag(n),
        h_x=np.ones(n, dtype=np.float64),
    )
    sched = Schedule.linear(T=1.0)
    with pytest.raises(ValueError, match="m must"):
        QuantumAnnealer(prob, sched, m=0)
    with pytest.raises(ValueError, match="krylov_tol"):
        QuantumAnnealer(prob, sched, krylov_tol=-1e-12)


# ---------------------------------------------------------------------------
# create_simulator (issue #48)
# ---------------------------------------------------------------------------


def test_create_simulator_returns_simulator_with_initial_state() -> None:
    """``create_simulator`` 戻り値が ``AnnealingSimulator`` で,
    ``t == t0`` / ``psi ≈ psi0`` / ``n_matvec == 0`` で初期化されている.
    """
    from kryanneal import AnnealingSimulator

    n = 3
    prob = IsingProblem(
        n=n,
        H_p_diag=_ferromagnetic_chain_h_p_diag(n),
        h_x=np.ones(n, dtype=np.float64),
    )
    sched = Schedule.linear(T=1.0)
    psi0 = uniform_superposition(n)
    ann = QuantumAnnealer(prob, sched)

    sim = ann.create_simulator(psi0, 0.0, method="m2")
    assert isinstance(sim, AnnealingSimulator)
    assert sim.t == 0.0
    assert sim.method == "m2"
    assert sim.n_matvec == 0
    assert np.allclose(sim.psi, psi0)


def test_create_simulator_supports_same_methods_as_run() -> None:
    """``create_simulator`` がサポートする method 集合が ``run`` と一致."""
    n = 3
    prob = IsingProblem(
        n=n,
        H_p_diag=_ferromagnetic_chain_h_p_diag(n),
        h_x=np.ones(n, dtype=np.float64),
    )
    sched = Schedule.linear(T=1.0)
    psi0 = uniform_superposition(n)
    ann = QuantumAnnealer(prob, sched)

    # `run` の valid method 集合 (annealer.py のソースに合わせて手で列挙).
    methods_supported = [
        "m2",
        "trotter",
        "trotter_suzuki4",
        "cfm4",
        "cfm4_adaptive_richardson_krylov",
    ]
    for method in methods_supported:
        sim = ann.create_simulator(psi0, 0.0, method=method)  # type: ignore[arg-type]
        assert sim.method == method

    # サポート外の method は ``NotImplementedError`` で弾かれる (``run`` と同じ).
    with pytest.raises(NotImplementedError):
        ann.create_simulator(psi0, 0.0, method="bogus")  # type: ignore[arg-type]


def test_create_simulator_inherits_m_and_krylov_tol_from_annealer() -> None:
    """``create_simulator`` は QuantumAnnealer の ``m`` / ``krylov_tol``
    を Simulator に引き継ぐ."""
    n = 3
    prob = IsingProblem(
        n=n,
        H_p_diag=_ferromagnetic_chain_h_p_diag(n),
        h_x=np.ones(n, dtype=np.float64),
    )
    sched = Schedule.linear(T=1.0)
    psi0 = uniform_superposition(n)
    ann = QuantumAnnealer(prob, sched, m=16, krylov_tol=1e-10)

    sim = ann.create_simulator(psi0, 0.0, method="cfm4")
    # 内部値の検証は public API 経由では難しいが, step を 1 回呼んで
    # m=16 経路の n_matvec (1 step × 2m = 32) が返ることで確認.
    sim.step(0.1)
    assert sim.n_matvec == 32
