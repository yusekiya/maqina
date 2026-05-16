"""``QuantumAnnealer.run`` × ``Observable`` / ``save_tlist`` / ``store_states``
の統合テスト (Phase 5, issue #47).

各 method × 観測経路の組合せで, ``save_tlist`` で指定した時刻軸に観測量
時系列と (要求すれば) 状態スナップショットが記録されること, エネルギー
保存則が時間に依存しない H で成り立つこと, 検証エラー (monotonicity /
範囲外 / dtype) を確認する.

スコープ:

* ``method="m2"`` smoke (固定 dt 中点則).
* ``method="cfm4_adaptive_richardson"`` smoke (adaptive PI controller).
* エネルギー保存則 (constant schedule で ``<ψ|H|ψ>`` が時間で不変).
* ``save_tlist`` 検証エラー (monotonicity / 範囲外 / dtype).
* ``store_states=True`` で ``states.shape == (K, 2**n)`` の契約.
"""

from __future__ import annotations

import numpy as np
import pytest

from kryanneal import IsingProblem, Observable, QuantumAnnealer, Schedule
from kryanneal.initial_states import uniform_superposition


def _make_problem(n: int = 3, seed: int = 7) -> IsingProblem:
    rng = np.random.default_rng(seed)
    dim = 1 << n
    h_p_diag = rng.normal(size=dim).astype(np.float64)
    h_x = np.ones(n, dtype=np.float64)
    return IsingProblem(n=n, H_p_diag=h_p_diag, h_x=h_x)


def test_observables_m2_returns_history_with_save_tlist_length() -> None:
    """``method='m2'`` で ``observables`` を渡すと ``observables_history``
    の各 array が ``len(save_tlist)`` の長さを持つ.
    """
    n = 3
    T = 1.0
    prob = _make_problem(n)
    sched = Schedule.linear(T=T)
    psi0 = uniform_superposition(n)
    ann = QuantumAnnealer(prob, sched)

    save_tlist = np.linspace(0.0, T, 6, dtype=np.float64)
    res = ann.run(
        psi0,
        0.0,
        T,
        method="m2",
        n_steps=20,
        observables={
            "M_z": Observable.magnetization(n),
            "H_p": Observable.ising_energy(prob),
        },
        save_tlist=save_tlist,
    )
    assert res.times is not None
    np.testing.assert_array_equal(res.times, save_tlist)
    assert set(res.observables_history.keys()) == {"M_z", "H_p"}
    for arr in res.observables_history.values():
        assert arr.shape == (save_tlist.shape[0],)
        assert arr.dtype == np.float64


def test_observables_adaptive_richardson_returns_history() -> None:
    """``method='cfm4_adaptive_richardson'`` でも同様に観測量時系列が
    記録される. PI が ``save_tlist`` を target に dt クランプする経路の smoke.
    """
    n = 3
    T = 2.0
    prob = _make_problem(n)
    sched = Schedule.linear(T=T)
    psi0 = uniform_superposition(n)
    ann = QuantumAnnealer(prob, sched)

    save_tlist = np.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype=np.float64)
    res = ann.run(
        psi0,
        0.0,
        T,
        method="cfm4_adaptive_richardson",
        atol=1e-8,
        observables={"M_z": Observable.magnetization(n)},
        save_tlist=save_tlist,
        store_states=True,
    )
    assert res.times is not None
    np.testing.assert_array_equal(res.times, save_tlist)
    assert "M_z" in res.observables_history
    assert res.observables_history["M_z"].shape == (5,)
    assert res.states is not None
    assert res.states.shape == (5, 1 << n)


def test_energy_conservation_constant_schedule_m2() -> None:
    """時間非依存 H で ``<ψ|H|ψ>`` がエネルギー保存則を満たす (m2).

    **A=0, B=1 の constant schedule** にすることで ``H(t) = H_problem``
    (時間非依存) となり, Schrödinger 方程式の解は ``exp(-i·H_p·t)·ψ0`` で
    ``<ψ(t)|H_p|ψ(t)> = <ψ0|H_p|ψ0>`` が厳密に保たれる. 別 driver 項を
    含む ``A=B=0.5`` のような constant schedule では H_driver (X 演算子)
    と H_p (Z 対角) が交換しないため H_p の期待値は時間で変化するので,
    Z 基底対角 Observable で保存則を確認するには A=0 にする必要がある.
    """
    n = 3
    T = 1.0
    prob = _make_problem(n)
    sched = Schedule(T=T, A=lambda s: 0.0, B=lambda s: 1.0)
    psi0 = uniform_superposition(n)
    ann = QuantumAnnealer(prob, sched)

    save_tlist = np.linspace(0.0, T, 11, dtype=np.float64)
    res = ann.run(
        psi0,
        0.0,
        T,
        method="m2",
        n_steps=200,
        observables={"H_p": Observable.ising_energy(prob)},
        save_tlist=save_tlist,
    )
    hp_hist = res.observables_history["H_p"]
    expected = float(hp_hist[0])
    rel = np.abs(hp_hist - expected) / max(abs(expected), 1.0)
    assert float(np.max(rel)) < 1e-10, (
        f"H_p not conserved in constant schedule: hist={hp_hist}, "
        f"max_rel_err={float(np.max(rel))}"
    )


def test_energy_conservation_constant_schedule_richardson() -> None:
    """adaptive Richardson 経路でも A=0, B=1 でエネルギー保存則.

    ``cfm4_adaptive_richardson`` は constant schedule + 対角 H で
    rounding 誤差レベルの err≈0 となり PI が dt を ``dt_max`` まで伸ばす.
    観測量を ``save_tlist`` 時刻で記録し H_p 期待値が保存することを確認.
    """
    n = 3
    T = 1.0
    prob = _make_problem(n)
    sched = Schedule(T=T, A=lambda s: 0.0, B=lambda s: 1.0)
    psi0 = uniform_superposition(n)
    ann = QuantumAnnealer(prob, sched)

    save_tlist = np.linspace(0.0, T, 6, dtype=np.float64)
    res = ann.run(
        psi0,
        0.0,
        T,
        method="cfm4_adaptive_richardson",
        atol=1e-10,
        observables={"H_p": Observable.ising_energy(prob)},
        save_tlist=save_tlist,
    )
    hp_hist = res.observables_history["H_p"]
    expected = float(hp_hist[0])
    rel = np.abs(hp_hist - expected) / max(abs(expected), 1.0)
    assert float(np.max(rel)) < 1e-10, (
        f"H_p not conserved (adaptive Richardson): hist={hp_hist}, "
        f"max_rel_err={float(np.max(rel))}"
    )


def test_save_tlist_out_of_range_raises() -> None:
    """``save_tlist[k] < t0`` または ``> t1`` で ``ValueError``."""
    n = 3
    prob = _make_problem(n)
    sched = Schedule.linear(T=1.0)
    psi0 = uniform_superposition(n)
    ann = QuantumAnnealer(prob, sched)

    with pytest.raises(ValueError, match=r"save_tlist"):
        ann.run(
            psi0,
            0.0,
            1.0,
            method="m2",
            n_steps=10,
            save_tlist=np.array([-0.1, 0.5], dtype=np.float64),
        )
    with pytest.raises(ValueError, match=r"save_tlist"):
        ann.run(
            psi0,
            0.0,
            1.0,
            method="m2",
            n_steps=10,
            save_tlist=np.array([0.5, 1.1], dtype=np.float64),
        )


def test_save_tlist_not_monotonic_raises() -> None:
    """``save_tlist`` が monotonic increasing でないと ``ValueError``."""
    n = 3
    prob = _make_problem(n)
    sched = Schedule.linear(T=1.0)
    psi0 = uniform_superposition(n)
    ann = QuantumAnnealer(prob, sched)

    with pytest.raises(ValueError, match=r"monoton"):
        ann.run(
            psi0,
            0.0,
            1.0,
            method="m2",
            n_steps=10,
            save_tlist=np.array([0.5, 0.3, 0.8], dtype=np.float64),
        )


def test_save_tlist_wrong_dtype_raises() -> None:
    """``save_tlist`` の dtype が float64 でないと ``ValueError``."""
    n = 3
    prob = _make_problem(n)
    sched = Schedule.linear(T=1.0)
    psi0 = uniform_superposition(n)
    ann = QuantumAnnealer(prob, sched)

    with pytest.raises(ValueError, match=r"dtype"):
        ann.run(
            psi0,
            0.0,
            1.0,
            method="m2",
            n_steps=10,
            save_tlist=np.array([0.5], dtype=np.float32),
        )


def test_save_tlist_empty_raises() -> None:
    """``save_tlist`` が空配列だと ``ValueError`` (``None`` を渡すよう要求)."""
    n = 3
    prob = _make_problem(n)
    sched = Schedule.linear(T=1.0)
    psi0 = uniform_superposition(n)
    ann = QuantumAnnealer(prob, sched)

    with pytest.raises(ValueError, match=r"non-empty"):
        ann.run(
            psi0,
            0.0,
            1.0,
            method="m2",
            n_steps=10,
            save_tlist=np.array([], dtype=np.float64),
        )


def test_store_states_shape_fixed_dt() -> None:
    """固定 dt 経路で ``store_states=True`` のとき
    ``states.shape == (len(save_tlist), 2**n)`` の契約.
    """
    n = 3
    T = 1.0
    prob = _make_problem(n)
    sched = Schedule.linear(T=T)
    psi0 = uniform_superposition(n)
    ann = QuantumAnnealer(prob, sched)

    save_tlist = np.linspace(0.0, T, 4, dtype=np.float64)
    res = ann.run(
        psi0,
        0.0,
        T,
        method="m2",
        n_steps=20,
        save_tlist=save_tlist,
        store_states=True,
    )
    assert res.states is not None
    assert res.states.shape == (4, 1 << n)
    assert res.states.dtype == np.complex128
    # 先頭は psi0 そのもの.
    np.testing.assert_array_equal(res.states[0], psi0)


def test_store_states_false_keeps_states_none() -> None:
    """``store_states=False`` のとき ``states is None`` (observables のみ記録)."""
    n = 3
    T = 1.0
    prob = _make_problem(n)
    sched = Schedule.linear(T=T)
    psi0 = uniform_superposition(n)
    ann = QuantumAnnealer(prob, sched)

    save_tlist = np.linspace(0.0, T, 4, dtype=np.float64)
    res = ann.run(
        psi0,
        0.0,
        T,
        method="m2",
        n_steps=20,
        observables={"M_z": Observable.magnetization(n)},
        save_tlist=save_tlist,
        store_states=False,
    )
    assert res.states is None
    assert res.times is not None
    assert "M_z" in res.observables_history


def test_observables_diag_length_mismatch_raises() -> None:
    """``Observable.diag`` の長さが ``2**n`` と整合しないと ``ValueError``."""
    n = 3
    prob = _make_problem(n)
    sched = Schedule.linear(T=1.0)
    psi0 = uniform_superposition(n)
    ann = QuantumAnnealer(prob, sched)

    # n=3 用問題に対し n=4 の magnetization を渡す.
    bad_obs = Observable.magnetization(4)
    with pytest.raises(ValueError, match=r"length mismatch|2\*\*n"):
        ann.run(
            psi0,
            0.0,
            1.0,
            method="m2",
            n_steps=10,
            observables={"M_z_wrong": bad_obs},
            save_tlist=np.array([0.5], dtype=np.float64),
        )


def test_probabilities_returned_even_without_save_tlist() -> None:
    """``save_tlist=None`` (最節約モード) でも ``probabilities`` は常に返る.

    ``QuantumResult.probabilities = |psi_final|^2`` の eager 計算は
    save_tlist の有無に依らない.
    """
    n = 3
    prob = _make_problem(n)
    sched = Schedule.linear(T=1.0)
    psi0 = uniform_superposition(n)
    ann = QuantumAnnealer(prob, sched)
    res = ann.run(psi0, 0.0, 1.0, method="m2", n_steps=20)
    assert res.times is None
    assert res.states is None
    assert res.observables_history == {}
    assert res.probabilities is not None
    np.testing.assert_array_almost_equal(
        res.probabilities, np.abs(res.psi_final) ** 2
    )
    np.testing.assert_array_almost_equal(float(res.probabilities.sum()), 1.0)
