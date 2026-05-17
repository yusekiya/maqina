"""``Observable`` (Z 基底対角 Hermitian 観測量) のテスト.

``__init__`` の shape / dtype / contiguity / finiteness 検証, ``expectation``
の数値正しさ, ``magnetization`` / ``ising_energy`` factory の挙動を確認
する. 仕様は ``docs/design/04-python-api.md`` §4.6 を参照.
"""

from __future__ import annotations

import numpy as np
import pytest

from kryanneal import IsingProblem, Observable


def test_construct_ok() -> None:
    """有効な diag で生成でき, ``dim`` が正しい."""
    diag = np.array([1.0, -1.0, 0.5, -0.5], dtype=np.float64)
    obs = Observable(diag)
    assert obs.dim == 4
    # diag を内部で共有する (copy しない) 契約.
    assert obs.diag is diag


def test_expectation_basis_state() -> None:
    """diag = [1, -1, 2, -2] と basis state ``|k⟩`` で期待値 = diag[k]."""
    diag = np.array([1.0, -1.0, 2.0, -2.0], dtype=np.float64)
    obs = Observable(diag)
    for k in range(4):
        psi = np.zeros(4, dtype=np.complex128)
        psi[k] = 1.0
        assert obs.expectation(psi) == pytest.approx(diag[k])


def test_expectation_returns_float() -> None:
    """戻り値は Python ``float`` (numpy scalar ではない)."""
    diag = np.array([1.0, -1.0], dtype=np.float64)
    obs = Observable(diag)
    psi = np.array([1.0, 0.0], dtype=np.complex128)
    val = obs.expectation(psi)
    assert isinstance(val, float)


def test_expectation_superposition() -> None:
    """重ね合わせ ``(|0⟩ + |1⟩)/√2`` に対して期待値 = (diag[0] + diag[1]) / 2."""
    diag = np.array([3.0, -1.0], dtype=np.float64)
    obs = Observable(diag)
    psi = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
    # |ψ_k|^2 = 1/2 each ⇒ <O> = (3 + (-1)) / 2 = 1
    assert obs.expectation(psi) == pytest.approx(1.0)


def test_expectation_complex_phase_ignored() -> None:
    """``|ψ_k|^2`` で計算されるので global / 各成分の位相は期待値に影響しない."""
    diag = np.array([2.0, 5.0], dtype=np.float64)
    obs = Observable(diag)
    psi = np.array([1.0, 1j], dtype=np.complex128) / np.sqrt(2.0)
    # |ψ_0|^2 = |ψ_1|^2 = 1/2 ⇒ <O> = (2 + 5)/2 = 3.5
    assert obs.expectation(psi) == pytest.approx(3.5)


def test_expectation_shape_mismatch() -> None:
    diag = np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float64)
    obs = Observable(diag)
    psi_wrong = np.zeros(2, dtype=np.complex128)
    with pytest.raises(ValueError, match="psi shape mismatch"):
        obs.expectation(psi_wrong)


def test_magnetization_shape_and_extremes() -> None:
    """``M_z`` diag は shape ``(2**n,)`` で, x=0 で +n, x=2^n-1 で -n."""
    n = 4
    obs = Observable.magnetization(n)
    assert obs.diag.shape == (1 << n,)
    assert obs.diag.dtype == np.float64
    # 全 spin up (bit 0): σ^z_total = +n
    assert obs.diag[0] == pytest.approx(float(n))
    # 全 spin down (全 bit 1): σ^z_total = -n
    assert obs.diag[(1 << n) - 1] == pytest.approx(-float(n))


def test_magnetization_per_bit_pattern() -> None:
    """個々の bit pattern で ``σ_i^z = 1 - 2·b_i`` の総和に一致."""
    n = 5
    obs = Observable.magnetization(n)
    for k in range(1 << n):
        expected = sum(1 - 2 * ((k >> i) & 1) for i in range(n))
        assert obs.diag[k] == pytest.approx(float(expected))


def test_magnetization_axis_x_not_implemented() -> None:
    """``axis="x"`` は v0.1 では NotImplementedError."""
    with pytest.raises(NotImplementedError, match="axis='x'"):
        Observable.magnetization(3, axis="x")  # type: ignore[arg-type]


def test_magnetization_invalid_n() -> None:
    with pytest.raises(ValueError, match="n must be a positive integer"):
        Observable.magnetization(0)


def test_ising_energy_matches_problem_diag() -> None:
    """``ising_energy(problem).diag`` は ``problem.H_p_diag`` と数値的に一致."""
    n = 3
    dim = 1 << n
    h_p_diag = np.linspace(-2.0, 2.0, dim, dtype=np.float64)
    h_x = np.ones(n, dtype=np.float64)
    prob = IsingProblem(n=n, H_p_diag=h_p_diag, h_x=h_x)

    obs = Observable.ising_energy(prob)
    assert obs.diag.shape == prob.H_p_diag.shape
    np.testing.assert_array_equal(obs.diag, prob.H_p_diag)


def test_ising_energy_is_deep_copy() -> None:
    """``ising_energy`` の diag は ``problem.H_p_diag`` と独立 (deep copy)."""
    n = 3
    dim = 1 << n
    h_p_diag = np.linspace(-2.0, 2.0, dim, dtype=np.float64)
    h_x = np.ones(n, dtype=np.float64)
    prob = IsingProblem(n=n, H_p_diag=h_p_diag, h_x=h_x)

    obs = Observable.ising_energy(prob)
    # 別実体である.
    assert obs.diag is not prob.H_p_diag
    # 一方を書き換えても他方に影響しない.
    obs.diag[0] = 999.0
    assert prob.H_p_diag[0] != 999.0


def test_constructor_rejects_non_ndarray() -> None:
    with pytest.raises(ValueError, match="diag must be a numpy.ndarray"):
        Observable([1.0, -1.0, 1.0, -1.0])  # type: ignore[arg-type]


def test_constructor_rejects_wrong_ndim() -> None:
    diag = np.zeros((2, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="diag must be 1-dimensional"):
        Observable(diag)


def test_constructor_rejects_non_power_of_two() -> None:
    diag = np.zeros(3, dtype=np.float64)
    with pytest.raises(ValueError, match="power of 2"):
        Observable(diag)


def test_constructor_rejects_wrong_dtype() -> None:
    diag = np.zeros(4, dtype=np.float32)
    with pytest.raises(ValueError, match="diag dtype must be float64"):
        Observable(diag)


def test_constructor_rejects_non_contiguous() -> None:
    big = np.zeros(8, dtype=np.float64)
    view = big[::2]  # shape (4,), non-contiguous
    assert not view.flags.c_contiguous
    with pytest.raises(ValueError, match="C-contiguous"):
        Observable(view)


def test_constructor_rejects_nan() -> None:
    diag = np.array([1.0, np.nan, 0.0, 0.0], dtype=np.float64)
    with pytest.raises(ValueError, match="NaN or inf"):
        Observable(diag)


def test_constructor_rejects_inf() -> None:
    diag = np.array([1.0, np.inf, 0.0, 0.0], dtype=np.float64)
    with pytest.raises(ValueError, match="NaN or inf"):
        Observable(diag)
