"""``builders`` モジュール (k-local → ``H_p_diag``) のテスト.

``diag_from_pauli_terms`` / ``diag_from_J_h`` が手計算と一致すること, 重複項の
加算・定数項・各種バリデーション (``ValueError``) の契約を検証する. ビット規約
(bit 0 = LSB, ``σ_i = 1 - 2·b_i``) は ``CLAUDE.md`` 「物理的取り決め」節準拠.
"""

from __future__ import annotations

import numpy as np
import pytest

from maqina.builders import diag_from_J_h, diag_from_pauli_terms


def _sigma(n: int) -> np.ndarray:
    """参照用 ``σ_i`` を素朴に構築する (shape ``(n, 2^n)``)."""
    dim = 1 << n
    sig = np.empty((n, dim), dtype=np.float64)
    for i in range(n):
        for x in range(dim):
            b_i = (x >> i) & 1
            sig[i, x] = 1.0 - 2.0 * b_i
    return sig


# ----------------------------------------------------------------------------
# diag_from_pauli_terms
# ----------------------------------------------------------------------------
def test_single_z_term_matches_sigma() -> None:
    """``1.5 · Z_0`` は ``1.5 · σ_0`` に一致する."""
    n = 3
    diag = diag_from_pauli_terms(n, [(1.5, (0,))])
    sig = _sigma(n)
    np.testing.assert_allclose(diag, 1.5 * sig[0], atol=1e-15)
    assert diag.dtype == np.float64
    assert diag.shape == (1 << n,)


def test_zz_term_matches_product() -> None:
    """``coeff · Z_0 Z_2`` は ``coeff · σ_0 σ_2`` に一致する."""
    n = 3
    diag = diag_from_pauli_terms(n, [(-0.7, (0, 2))])
    sig = _sigma(n)
    np.testing.assert_allclose(diag, -0.7 * sig[0] * sig[2], atol=1e-15)


@pytest.mark.parametrize("n", [2, 3, 4])
def test_sum_of_terms_against_bruteforce(n: int) -> None:
    """複数項の和が素朴な手計算と全成分一致する."""
    terms = [(1.0, (0,)), (-0.5, (0, 1)), (0.25, tuple(range(n)))]
    diag = diag_from_pauli_terms(n, terms)

    sig = _sigma(n)
    expected = np.zeros(1 << n, dtype=np.float64)
    for coeff, sites in terms:
        prod = np.ones(1 << n, dtype=np.float64)
        for s in sites:
            prod = prod * sig[s]
        expected += coeff * prod
    np.testing.assert_allclose(diag, expected, atol=1e-14)


def test_duplicate_terms_accumulate() -> None:
    """同じサイト集合の term が複数あれば係数が加算される."""
    n = 2
    combined = diag_from_pauli_terms(n, [(1.0, (0, 1)), (0.5, (0, 1))])
    single = diag_from_pauli_terms(n, [(1.5, (0, 1))])
    np.testing.assert_allclose(combined, single, atol=1e-15)


def test_permuted_sites_are_same_term() -> None:
    """``(0, 1)`` と ``(1, 0)`` は可換なので同一項として加算される."""
    n = 2
    permuted = diag_from_pauli_terms(n, [(1.0, (0, 1)), (0.5, (1, 0))])
    single = diag_from_pauli_terms(n, [(1.5, (0, 1))])
    np.testing.assert_allclose(permuted, single, atol=1e-15)


def test_identity_term_offsets_all_components() -> None:
    """空タプル ``()`` は単位演算子 ``I`` として全成分に係数を加算する."""
    n = 3
    base = diag_from_pauli_terms(n, [(1.0, (0,))])
    with_offset = diag_from_pauli_terms(n, [(1.0, (0,)), (2.5, ())])
    np.testing.assert_allclose(with_offset, base + 2.5, atol=1e-15)


def test_empty_terms_is_zero_vector() -> None:
    """項が空なら零ベクトルを返す."""
    diag = diag_from_pauli_terms(3, [])
    np.testing.assert_array_equal(diag, np.zeros(8))


def test_pauli_invalid_n_rejected() -> None:
    with pytest.raises(ValueError, match="n must be a positive integer"):
        diag_from_pauli_terms(0, [(1.0, (0,))])
    with pytest.raises(ValueError, match="n must be a positive integer"):
        diag_from_pauli_terms(-2, [])


def test_pauli_site_out_of_range_rejected() -> None:
    with pytest.raises(ValueError, match="out of range"):
        diag_from_pauli_terms(3, [(1.0, (3,))])
    with pytest.raises(ValueError, match="out of range"):
        diag_from_pauli_terms(3, [(1.0, (-1,))])


def test_pauli_duplicate_sites_within_term_rejected() -> None:
    """1 項内のサイト重複は ``ValueError`` (``Z_i^2 = I`` 簡約をしない)."""
    with pytest.raises(ValueError, match="duplicate site"):
        diag_from_pauli_terms(3, [(1.0, (0, 0))])
    with pytest.raises(ValueError, match="duplicate site"):
        diag_from_pauli_terms(3, [(1.0, (1, 2, 1))])


# ----------------------------------------------------------------------------
# diag_from_J_h
# ----------------------------------------------------------------------------
def test_J_h_matches_bruteforce() -> None:
    """``-Σ_{i<j} J_ij σ_i σ_j - Σ_i h_i σ_i`` の手計算と全成分一致."""
    n = 4
    rng = np.random.default_rng(0)
    J = rng.normal(size=(n, n))
    J = (J + J.T) / 2.0
    np.fill_diagonal(J, 0.0)
    h = rng.normal(size=n)

    diag = diag_from_J_h(J, h)

    sig = _sigma(n)
    expected = np.zeros(1 << n, dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            expected -= J[i, j] * sig[i] * sig[j]
    for i in range(n):
        expected -= h[i] * sig[i]
    np.testing.assert_allclose(diag, expected, atol=1e-13)
    assert diag.dtype == np.float64
    assert diag.shape == (1 << n,)


def test_J_h_single_coupling() -> None:
    """``J_01 = -1`` (強磁性), ``h = 0`` で ``-J_01 σ_0 σ_1 = σ_0 σ_1``."""
    n = 2
    J = np.zeros((n, n))
    J[0, 1] = J[1, 0] = -1.0
    diag = diag_from_J_h(J, np.zeros(n))
    sig = _sigma(n)
    np.testing.assert_allclose(diag, sig[0] * sig[1], atol=1e-15)


def test_J_h_equivalent_to_pauli_terms() -> None:
    """``diag_from_J_h`` は対応する Pauli-Z term 列と一致する."""
    n = 3
    rng = np.random.default_rng(1)
    J = rng.normal(size=(n, n))
    J = (J + J.T) / 2.0
    np.fill_diagonal(J, 0.0)
    h = rng.normal(size=n)

    via_jh = diag_from_J_h(J, h)

    terms: list[tuple[float, tuple[int, ...]]] = []
    for i in range(n):
        for j in range(i + 1, n):
            terms.append((-J[i, j], (i, j)))
        terms.append((-h[i], (i,)))
    via_terms = diag_from_pauli_terms(n, terms)
    np.testing.assert_allclose(via_jh, via_terms, atol=1e-13)


def test_J_not_square_rejected() -> None:
    with pytest.raises(ValueError, match="square 2D array"):
        diag_from_J_h(np.zeros((2, 3)), np.zeros(2))


def test_J_not_symmetric_rejected() -> None:
    J = np.array([[0.0, 1.0], [2.0, 0.0]])
    with pytest.raises(ValueError, match="symmetric"):
        diag_from_J_h(J, np.zeros(2))


def test_J_nonzero_diagonal_rejected() -> None:
    J = np.array([[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="zero diagonal"):
        diag_from_J_h(J, np.zeros(2))


def test_J_complex_rejected() -> None:
    J = np.zeros((2, 2), dtype=np.complex128)
    with pytest.raises(ValueError, match="real"):
        diag_from_J_h(J, np.zeros(2))


def test_h_shape_mismatch_rejected() -> None:
    J = np.zeros((3, 3))
    with pytest.raises(ValueError, match="h shape mismatch"):
        diag_from_J_h(J, np.zeros(2))


def test_h_complex_rejected() -> None:
    J = np.zeros((2, 2))
    with pytest.raises(ValueError, match="real"):
        diag_from_J_h(J, np.zeros(2, dtype=np.complex128))
