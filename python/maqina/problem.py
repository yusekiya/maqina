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
する. 物理的取り決め (bit 規約等) は ``docs/design/INDEX.md`` および
``CLAUDE.md`` 「物理的取り決め」節を参照.

Note
----
``@dataclass(frozen=True, eq=False)`` を使う. ``frozen=True`` でフィールドを
不変化するが, ``eq=False`` にする理由は ``H_p_diag`` / ``h_x`` が
``numpy.ndarray`` であり, ``==`` がブロードキャストで array を返すため
dataclass 既定の ``__eq__`` (タプル比較) が ``ValueError: The truth value of
an array ...`` で破綻するため.

``from_pauli_terms`` / ``from_J_h`` classmethod は ``docs/design/04-python-api.md`` §4.2 に
記載されているが, ``builders`` モジュールの実装が別 issue のため本リリース
では未提供 (Phase 1 内, builders 実装後に追加予定).
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
    # 以下は __post_init__ で計算する Gershgorin 上下界の precompute 値.
    # init=False で __init__ 引数から外し, frozen=True でも
    # object.__setattr__ 経由で書き込む.
    _h_x_abs_sum: float = field(init=False, repr=False, compare=False, default=0.0)
    _h_p_diag_min: float = field(init=False, repr=False, compare=False, default=0.0)
    _h_p_diag_max: float = field(init=False, repr=False, compare=False, default=0.0)

    def __post_init__(self) -> None:
        if not isinstance(self.n, (int, np.integer)) or self.n < 1:
            raise ValueError(f"n must be a positive integer, got {self.n!r}")

        expected_dim = 1 << int(self.n)
        self._validate_real_array("H_p_diag", self.H_p_diag, (expected_dim,))
        self._validate_real_array("h_x", self.h_x, (int(self.n),))

        # Gershgorin 上下界の precompute (Chebyshev propagator が per-step
        # `gershgorin_bounds_cached` で O(1) 計算するための入力値). h_x /
        # H_p_diag は frozen なので 1 度だけ計算して持つ. frozen=True を
        # 維持しつつ属性を設定するため object.__setattr__ を使う.
        object.__setattr__(self, "_h_x_abs_sum", float(np.abs(self.h_x).sum()))
        object.__setattr__(self, "_h_p_diag_min", float(self.H_p_diag.min()))
        object.__setattr__(self, "_h_p_diag_max", float(self.H_p_diag.max()))

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

    @property
    def h_x_abs_sum(self) -> float:
        """``Σ_i |h_x_i|`` の precompute 値 (Gershgorin 行和上界の非対角寄与).

        Chebyshev propagator (``cfm4_step_chebyshev_*``) が per-step
        Gershgorin 上下界を O(1) で計算するための precompute. ``__post_init__``
        で 1 度だけ計算され, インスタンスが ``frozen=True`` のため以降不変.
        """
        return self._h_x_abs_sum

    @property
    def h_p_diag_min(self) -> float:
        """``min(H_p_diag)`` の precompute 値 (Gershgorin 行和下界の対角最小)."""
        return self._h_p_diag_min

    @property
    def h_p_diag_max(self) -> float:
        """``max(H_p_diag)`` の precompute 値 (Gershgorin 行和上界の対角最大)."""
        return self._h_p_diag_max
