"""``uniform_superposition`` の正規化と各成分値テスト.

driver Hamiltonian の最低エネルギー固有状態 ``|+⟩^N`` が L2 正規化
済みで, 各成分が ``1 / √(2^N)`` であることを検証する.
"""

from __future__ import annotations

import numpy as np
import pytest

from kinema.initial_states import uniform_superposition


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8])
def test_normalization(n: int) -> None:
    """``‖|+⟩^N‖ = 1`` を厳密に満たす."""
    psi = uniform_superposition(n)
    assert psi.dtype == np.complex128
    assert psi.shape == (1 << n,)
    # L2 ノルム
    norm = np.linalg.norm(psi)
    assert norm == pytest.approx(1.0, abs=1e-15)


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8])
def test_each_component_value(n: int) -> None:
    """各成分が ``1 / √(2^N)`` (実部のみ, 虚部 0)."""
    psi = uniform_superposition(n)
    dim = 1 << n
    expected = 1.0 / np.sqrt(dim)
    np.testing.assert_allclose(psi.real, expected, atol=1e-15)
    np.testing.assert_array_equal(psi.imag, 0.0)


def test_invalid_n_rejected() -> None:
    with pytest.raises(ValueError, match="n must be a positive integer"):
        uniform_superposition(0)
    with pytest.raises(ValueError, match="n must be a positive integer"):
        uniform_superposition(-1)


def test_probability_sums_to_one() -> None:
    """``Σ_x |ψ_x|^2 = 1`` を満たす."""
    psi = uniform_superposition(4)
    probs = np.abs(psi) ** 2
    assert probs.sum() == pytest.approx(1.0, abs=1e-15)
