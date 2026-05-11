"""初期状態構築ヘルパ.

TFIM の標準的な始状態は driver Hamiltonian の最低エネルギー固有状態 ``|+⟩^N``
(各サイトが X 基底の ``|+⟩``). dimension ``2^N`` の complex128 ベクトル
として返す.

* ``uniform_superposition(n)``: ``|+⟩^N = (1/√(2^N)) Σ_x |x⟩``.

ユーザが ``psi0`` を ``QuantumAnnealer`` に渡すときは L2-normalize を
事前に検証される. default は提供しない (``CLAUDE.md`` の取り決め).
"""

from __future__ import annotations

import numpy as np

__all__ = ["uniform_superposition"]


def uniform_superposition(n: int) -> np.ndarray:
    """``|+⟩^N`` (全 ``2^N`` 計算基底の等振幅重ね合わせ) を返す.

    各成分が ``1 / √(2^N)`` の実数値で, 全成分の和を取ると ``√(2^N)``,
    L2 ノルムは厳密に 1. driver Hamiltonian ``H_driver = -Σ_i h_x_i X_i``
    の最低エネルギー固有状態 (``h_x_i > 0`` のとき) に一致する.

    Parameters
    ----------
    n
        スピン数. ``n >= 1``.

    Returns
    -------
    np.ndarray
        shape ``(2**n,)``, dtype ``complex128``.

    Raises
    ------
    ValueError
        ``n < 1`` の場合.
    """
    if not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError(f"n must be a positive integer, got {n!r}")
    dim = 1 << int(n)
    amp = 1.0 / np.sqrt(dim)
    return np.full(dim, amp, dtype=np.complex128)
