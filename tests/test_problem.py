"""``IsingProblem`` のコンストラクタ検証テスト.

``__post_init__`` で shape / dtype / contiguity / NaN / ``n`` 整合性が
ValueError に変換されることを確認する.

issue #142 Phase C で ``h_x`` は ``Schedule`` に移管されたため
``IsingProblem`` は ``H_p_diag`` のみ検証する (h_x の検証は ``test_schedule.py``).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from maqina import IsingProblem


def _ok_h_p_diag(n: int = 3) -> np.ndarray:
    dim = 1 << n
    return np.linspace(-1.0, 1.0, dim, dtype=np.float64)


def test_construct_ok() -> None:
    """正しい shape/dtype の入力ではエラー無く生成でき, ``dim`` が正しい."""
    n = 4
    h_p_diag = _ok_h_p_diag(n)
    prob = IsingProblem(n=n, H_p_diag=h_p_diag)
    assert prob.n == n
    assert prob.dim == 1 << n
    assert prob.H_p_diag is h_p_diag


def test_frozen_assignment_raises() -> None:
    """``frozen=True`` によりフィールド再代入は ``FrozenInstanceError``."""
    prob = IsingProblem(n=3, H_p_diag=_ok_h_p_diag(3))
    with pytest.raises(dataclasses.FrozenInstanceError):
        prob.n = 4  # type: ignore[misc]


def test_n_must_be_positive_int() -> None:
    h_p_diag = _ok_h_p_diag(3)
    with pytest.raises(ValueError, match="n must be a positive integer"):
        IsingProblem(n=0, H_p_diag=h_p_diag)
    with pytest.raises(ValueError, match="n must be a positive integer"):
        IsingProblem(n=-1, H_p_diag=h_p_diag)


def test_h_p_diag_shape_mismatch() -> None:
    """``H_p_diag.shape != (2**n,)`` → ValueError."""
    wrong = np.zeros(4, dtype=np.float64)  # 2**3 = 8 expected
    with pytest.raises(ValueError, match="H_p_diag shape mismatch"):
        IsingProblem(n=3, H_p_diag=wrong)


def test_h_p_diag_dtype_mismatch() -> None:
    """``H_p_diag`` が float64 以外 → ValueError."""
    wrong = np.zeros(8, dtype=np.float32)
    with pytest.raises(ValueError, match="H_p_diag dtype must be float64"):
        IsingProblem(n=3, H_p_diag=wrong)


def test_non_contiguous_rejected() -> None:
    """非 C-contiguous な ``H_p_diag`` → ValueError."""
    n = 3
    dim = 1 << n
    big = np.zeros(dim * 2, dtype=np.float64)
    view = big[::2]
    assert not view.flags.c_contiguous
    with pytest.raises(ValueError, match="H_p_diag must be C-contiguous"):
        IsingProblem(n=n, H_p_diag=view)


def test_nan_rejected() -> None:
    """NaN を含む配列 → ValueError."""
    n = 3
    h_p_diag = _ok_h_p_diag(n).copy()
    h_p_diag[0] = np.nan
    with pytest.raises(ValueError, match="H_p_diag contains NaN"):
        IsingProblem(n=n, H_p_diag=h_p_diag)


def test_h_p_diag_min_max_cached() -> None:
    """``_h_p_diag_min/_max`` が ``__post_init__`` で precompute される."""
    n = 3
    h_p_diag = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    prob = IsingProblem(n=n, H_p_diag=h_p_diag)
    assert prob.h_p_diag_min == pytest.approx(-2.0)
    assert prob.h_p_diag_max == pytest.approx(5.0)
