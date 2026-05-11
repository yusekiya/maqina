"""``QuantumResult`` / ``Trajectory`` dataclass の挙動テスト.

Phase 1 subset の最小フィールド構成を検証する. frozen 化された
``np.ndarray`` 格納 dataclass が再代入で ``FrozenInstanceError`` を
返すこと, および値が初期化時のまま参照できることを確認する.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from kryanneal import QuantumResult, Trajectory


def test_quantum_result_construct() -> None:
    """``QuantumResult`` の全フィールドが指定通り保持される."""
    n = 3
    dim = 1 << n
    psi = np.full(dim, 1.0 / np.sqrt(dim), dtype=np.complex128)
    t_history = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    obs = {"energy": np.array([1.0, 0.8, 0.6], dtype=np.float64)}
    res = QuantumResult(
        psi_final=psi,
        t_history=t_history,
        observables_history=obs,
        n_steps=2,
        n_matvec=42,
    )
    assert res.psi_final is psi
    assert res.t_history is t_history
    assert res.observables_history is obs
    assert res.n_steps == 2
    assert res.n_matvec == 42


def test_quantum_result_frozen() -> None:
    """``QuantumResult`` への代入は ``FrozenInstanceError``."""
    psi = np.zeros(8, dtype=np.complex128)
    res = QuantumResult(
        psi_final=psi,
        t_history=None,
        observables_history={},
        n_steps=0,
        n_matvec=0,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.n_steps = 1  # type: ignore[misc]


def test_quantum_result_allows_none_t_history() -> None:
    """``t_history=None`` (観測量を記録しないケース) を許容."""
    psi = np.zeros(4, dtype=np.complex128)
    res = QuantumResult(
        psi_final=psi,
        t_history=None,
        observables_history={},
        n_steps=10,
        n_matvec=100,
    )
    assert res.t_history is None
    assert res.observables_history == {}


def test_trajectory_construct() -> None:
    """``Trajectory`` 単独で構築でき, 観測量を保持できる."""
    t = np.linspace(0.0, 1.0, 5, dtype=np.float64)
    obs = {"M_z": np.array([0.5, 0.4, 0.3, 0.2, 0.1], dtype=np.float64)}
    traj = Trajectory(t_history=t, observables_history=obs)
    assert traj.t_history is t
    assert traj.observables_history is obs


def test_trajectory_default_observables_empty() -> None:
    """``observables_history`` を省略すると空 dict (``default_factory``)."""
    t = np.array([0.0], dtype=np.float64)
    traj = Trajectory(t_history=t)
    assert traj.observables_history == {}


def test_trajectory_frozen() -> None:
    """``Trajectory`` も frozen."""
    t = np.array([0.0], dtype=np.float64)
    traj = Trajectory(t_history=t)
    with pytest.raises(dataclasses.FrozenInstanceError):
        traj.t_history = np.array([1.0])  # type: ignore[misc]
