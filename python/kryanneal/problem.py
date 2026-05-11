"""TFIM 問題定義 (``IsingProblem``).

``IsingProblem`` は Hamiltonian の構造的・数値的入力を 1 箇所に集約する
データクラス. ユーザは以下を渡す:

* ``n``: スピン数
* ``H_p_diag``: shape ``(2^N,)`` float64. Z 基底における problem
  Hamiltonian の対角ベクトル (k-local 多項式は ``builders`` モジュールで
  この形に変換しておく).
* ``h_x``: shape ``(N,)`` float64. サイト依存横磁場の振幅 (driver
  Hamiltonian ``H_driver = -Σ_i h_x_i X_i`` の係数).

shape / dtype / NaN-free / `n` と各次元の整合性は本コンストラクタで検証
する. 物理的取り決め (bit 規約等) は ``docs/design.md`` および
``CLAUDE.md`` 「物理的取り決め」節を参照.

Note
----
``@dataclass(frozen=True, eq=False)`` を使う. ``frozen=True`` でフィールドを
不変化するが, ``eq=False`` にする理由は ``H_p_diag`` / ``h_x`` が
``numpy.ndarray`` であり, ``==`` がブロードキャストで array を返すため
dataclass 既定の ``__eq__`` (タプル比較) が ``ValueError: The truth value of
an array ...`` で破綻するため.

``from_pauli_terms`` / ``from_J_h`` classmethod は ``docs/design.md`` §4.2 に
記載されているが, ``builders`` モジュールの実装が別 issue のため本リリース
では未提供 (Phase 1 内, builders 実装後に追加予定).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["IsingProblem"]


@dataclass(frozen=True, eq=False)
class IsingProblem:
    """TFIM 問題定義 (immutable データコンテナ).

    Parameters
    ----------
    n
        スピン数. ``2**n`` が Hilbert 空間次元.
    H_p_diag
        shape ``(2**n,)`` float64 の C-contiguous ndarray.
        Z 基底における problem Hamiltonian の対角成分.
    h_x
        shape ``(n,)`` float64 の C-contiguous ndarray.
        サイト依存横磁場の振幅 (``H_driver = -Σ_i h_x_i X_i``).

    Raises
    ------
    ValueError
        以下のいずれかに該当する場合.

        * ``n`` が 1 以上の整数でない
        * ``H_p_diag.shape != (2**n,)``
        * ``H_p_diag.dtype != float64`` または C-contiguous でない
        * ``h_x.shape != (n,)``
        * ``h_x.dtype != float64`` または C-contiguous でない
        * 配列に NaN / inf が含まれる
    """

    n: int
    H_p_diag: np.ndarray
    h_x: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.n, (int, np.integer)) or self.n < 1:
            raise ValueError(f"n must be a positive integer, got {self.n!r}")

        expected_dim = 1 << int(self.n)
        self._validate_real_array("H_p_diag", self.H_p_diag, (expected_dim,))
        self._validate_real_array("h_x", self.h_x, (int(self.n),))

    @staticmethod
    def _validate_real_array(
        name: str, arr: np.ndarray, expected_shape: tuple[int, ...]
    ) -> None:
        if not isinstance(arr, np.ndarray):
            raise ValueError(
                f"{name} must be a numpy.ndarray, got {type(arr).__name__}"
            )
        if arr.shape != expected_shape:
            raise ValueError(
                f"{name} shape mismatch: expected {expected_shape}, got {arr.shape}"
            )
        if arr.dtype != np.float64:
            raise ValueError(f"{name} dtype must be float64, got {arr.dtype}")
        if not arr.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains NaN or inf")

    @property
    def dim(self) -> int:
        """Hilbert 空間次元 ``2**n``."""
        return 1 << int(self.n)
