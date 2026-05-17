"""``IsingProblem`` のコンストラクタ検証テスト.

``__post_init__`` で shape / dtype / contiguity / NaN / ``n`` 整合性が
ValueError に変換されることを確認する. 物理的取り決めは ``CLAUDE.md``
「物理的取り決め」節と ``docs/design/02-physics.md`` §2.2 / §4.2 を参照.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from kryanneal import IsingProblem


def _ok_inputs(n: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """有効な ``H_p_diag``, ``h_x`` のペアを返す."""
    dim = 1 << n
    h_p_diag = np.linspace(-1.0, 1.0, dim, dtype=np.float64)
    h_x = np.ones(n, dtype=np.float64)
    return h_p_diag, h_x


def test_construct_ok() -> None:
    """正しい shape/dtype の入力ではエラー無く生成でき, ``dim`` が正しい."""
    n = 4
    h_p_diag, h_x = _ok_inputs(n)
    prob = IsingProblem(n=n, H_p_diag=h_p_diag, h_x=h_x)
    assert prob.n == n
    assert prob.dim == 1 << n
    assert prob.H_p_diag is h_p_diag
    assert prob.h_x is h_x


def test_frozen_assignment_raises() -> None:
    """``frozen=True`` によりフィールド再代入は ``FrozenInstanceError``."""
    h_p_diag, h_x = _ok_inputs(3)
    prob = IsingProblem(n=3, H_p_diag=h_p_diag, h_x=h_x)
    with pytest.raises(dataclasses.FrozenInstanceError):
        prob.n = 4  # type: ignore[misc]


def test_n_must_be_positive_int() -> None:
    h_p_diag, h_x = _ok_inputs(3)
    with pytest.raises(ValueError, match="n must be a positive integer"):
        IsingProblem(n=0, H_p_diag=h_p_diag, h_x=h_x)
    with pytest.raises(ValueError, match="n must be a positive integer"):
        IsingProblem(n=-1, H_p_diag=h_p_diag, h_x=h_x)


def test_h_p_diag_shape_mismatch() -> None:
    """``H_p_diag.shape != (2**n,)`` → ValueError."""
    _, h_x = _ok_inputs(3)
    wrong = np.zeros(4, dtype=np.float64)  # 2**3 = 8 expected
    with pytest.raises(ValueError, match="H_p_diag shape mismatch"):
        IsingProblem(n=3, H_p_diag=wrong, h_x=h_x)


def test_h_x_shape_mismatch() -> None:
    """``h_x.shape != (n,)`` → ValueError."""
    h_p_diag, _ = _ok_inputs(3)
    wrong_hx = np.ones(2, dtype=np.float64)
    with pytest.raises(ValueError, match="h_x shape mismatch"):
        IsingProblem(n=3, H_p_diag=h_p_diag, h_x=wrong_hx)


def test_h_p_diag_dtype_mismatch() -> None:
    """``H_p_diag`` が float64 以外 → ValueError."""
    _, h_x = _ok_inputs(3)
    wrong = np.zeros(8, dtype=np.float32)
    with pytest.raises(ValueError, match="H_p_diag dtype must be float64"):
        IsingProblem(n=3, H_p_diag=wrong, h_x=h_x)


def test_h_x_dtype_mismatch() -> None:
    h_p_diag, _ = _ok_inputs(3)
    wrong_hx = np.ones(3, dtype=np.int64)
    with pytest.raises(ValueError, match="h_x dtype must be float64"):
        IsingProblem(n=3, H_p_diag=h_p_diag, h_x=wrong_hx)


def test_non_contiguous_rejected() -> None:
    """非 C-contiguous な ``H_p_diag`` → ValueError."""
    n = 3
    dim = 1 << n
    # 倍長を取り stride-2 view を作って non-contiguous にする.
    big = np.zeros(dim * 2, dtype=np.float64)
    view = big[::2]
    assert not view.flags.c_contiguous
    _, h_x = _ok_inputs(n)
    with pytest.raises(ValueError, match="H_p_diag must be C-contiguous"):
        IsingProblem(n=n, H_p_diag=view, h_x=h_x)


def test_nan_rejected() -> None:
    """NaN を含む配列 → ValueError."""
    n = 3
    h_p_diag, h_x = _ok_inputs(n)
    h_p_diag = h_p_diag.copy()
    h_p_diag[0] = np.nan
    with pytest.raises(ValueError, match="H_p_diag contains NaN"):
        IsingProblem(n=n, H_p_diag=h_p_diag, h_x=h_x)


def test_inf_rejected() -> None:
    """inf を含む配列 → ValueError (``isfinite`` の対象)."""
    n = 3
    _, h_x = _ok_inputs(n)
    h_x = h_x.copy()
    h_x[0] = np.inf
    h_p_diag, _ = _ok_inputs(n)
    with pytest.raises(ValueError, match="h_x contains NaN"):
        IsingProblem(n=n, H_p_diag=h_p_diag, h_x=h_x)
