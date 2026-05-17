"""観測量 ``Observable`` (Z 基底対角 Hermitian 演算子).

設計詳細は ``docs/design/04-python-api.md`` §4.6 を参照. Z 基底で対角な Hermitian
演算子のみを対象とするため, 期待値 ``<ψ|O|ψ> = Σ_k diag[k] · |ψ[k]|^2``
は実数で, dot product 1 回で計算できる. X / Y 期待値は ψ への bit-flip
を経由するため遅く, アニーリングのユースケースでは稀, という理由で
v0.1 では非対応 (future work).

ユーザ入力契約:

* ``diag``: shape ``(2**n,)`` float64, C-contiguous, finite. 内部では
  受け取った配列を **そのまま保持** する (copy しない). 呼び出し側が
  後で書き換える可能性があれば呼び出し側で copy してから渡すこと.
  ``ising_energy`` factory は ``problem.H_p_diag`` の不変性を担保する
  ために内部で ``.copy()`` を取る.

ビット規約は ``CLAUDE.md`` の「物理的取り決め」節に従う
(bit 0 = LSB, ``x = Σ_i b_i · 2^i``, spin ``σ_i = 1 - 2·b_i``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from kryanneal.problem import IsingProblem

__all__ = ["Observable"]


class Observable:
    """Z 基底対角の Hermitian 演算子.

    期待値は ``<ψ|O|ψ> = Σ_k diag[k] · |ψ[k]|^2`` で実数. diag を 1 本
    保持するだけのデータコンテナで, インスタンス化時に shape / dtype /
    finite チェックを行う.

    Parameters
    ----------
    diag
        shape ``(2**n,)`` float64 の C-contiguous ndarray. Z 基底で対角
        な Hermitian 演算子の対角成分.

    Raises
    ------
    ValueError
        以下のいずれかに該当する場合.

        * ``diag`` が ``numpy.ndarray`` でない
        * ``diag`` が 1 次元でない
        * ``len(diag)`` が 2 の冪でない (Hilbert 空間が ``2^n`` 次元という
          物理的取り決めに反する)
        * ``diag.dtype != float64`` または C-contiguous でない
        * ``diag`` に NaN / inf が含まれる
    """

    def __init__(self, diag: np.ndarray) -> None:
        if not isinstance(diag, np.ndarray):
            raise ValueError(f"diag must be a numpy.ndarray, got {type(diag).__name__}")
        if diag.ndim != 1:
            raise ValueError(f"diag must be 1-dimensional, got shape {diag.shape}")
        dim = diag.shape[0]
        if dim < 1 or (dim & (dim - 1)) != 0:
            raise ValueError(
                f"len(diag) must be a positive power of 2 (2**n), got {dim}"
            )
        if diag.dtype != np.float64:
            raise ValueError(f"diag dtype must be float64, got {diag.dtype}")
        if not diag.flags.c_contiguous:
            raise ValueError("diag must be C-contiguous")
        if not np.all(np.isfinite(diag)):
            raise ValueError("diag contains NaN or inf")

        self.diag: np.ndarray = diag

    @property
    def dim(self) -> int:
        """Hilbert 空間次元 ``2**n``."""
        return int(self.diag.shape[0])

    def expectation(self, psi: np.ndarray) -> float:
        """状態 ``psi`` に対する期待値 ``<ψ|O|ψ>`` を返す.

        Z 基底対角 Hermitian なので
        ``<ψ|O|ψ> = Σ_k diag[k] · |ψ[k]|^2`` は実数. dtype に依らず
        ``np.abs(psi)**2`` で振幅二乗を取り, ``diag`` との内積を float に
        キャストして返す (虚部は理論上 0).

        Parameters
        ----------
        psi
            shape ``(2**n,)`` の状態ベクトル. dtype は complex128 を想定
            するが ``np.abs`` 経由なので real / complex 両対応.

        Returns
        -------
        float
            期待値.

        Raises
        ------
        ValueError
            ``psi.shape != self.diag.shape`` の場合.
        """
        if not isinstance(psi, np.ndarray):
            raise ValueError(f"psi must be a numpy.ndarray, got {type(psi).__name__}")
        if psi.shape != self.diag.shape:
            raise ValueError(
                f"psi shape mismatch: expected {self.diag.shape}, got {psi.shape}"
            )
        return float(np.dot(self.diag, np.abs(psi) ** 2))

    @classmethod
    def magnetization(cls, n: int, axis: Literal["z"] = "z") -> "Observable":
        """全磁化 ``M_z = Σ_i σ_i^z`` の Observable を構築する.

        ``σ_i^z = 1 - 2·b_i`` (bit ``b_i`` が立つと spin down) という
        本パッケージの取り決めに従い, 状態 ``|x⟩`` に対する固有値は
        ``Σ_i (1 - 2·b_i) = n - 2·popcount(x)``.

        Parameters
        ----------
        n
            スピン数. ``n >= 1``.
        axis
            ``"z"`` のみ. ``"x"`` / ``"y"`` は v0.1 では非対応 (X / Y は
            ψ への bit-flip を経由するため Z 基底対角枠に収まらない).

        Returns
        -------
        Observable
            shape ``(2**n,)`` の ``M_z`` diag を持つ Observable.

        Raises
        ------
        ValueError
            ``n < 1`` の場合.
        NotImplementedError
            ``axis != "z"`` の場合.
        """
        if axis != "z":
            raise NotImplementedError(
                f"magnetization(axis={axis!r}) not supported in v0.1; "
                "only axis='z' is implemented (X / Y require bit-flip)."
            )
        if not isinstance(n, (int, np.integer)) or n < 1:
            raise ValueError(f"n must be a positive integer, got {n!r}")
        dim = 1 << int(n)
        k = np.arange(dim, dtype=np.int64)
        # popcount(k): k の立っているビット数. numpy 2.0+ の bitwise_count を
        # 使うが, 念のため Python の bit_count フォールバックを残しても良い
        # — 本リポジトリは numpy>=2.0 を pyproject で要求しているため直接使う.
        popcount = np.bitwise_count(k).astype(np.int64)
        diag = (int(n) - 2 * popcount).astype(np.float64)
        return cls(np.ascontiguousarray(diag))

    @classmethod
    def ising_energy(cls, problem: "IsingProblem") -> "Observable":
        """Problem Hamiltonian ``H_problem`` を観測量化する.

        ``problem.H_p_diag`` をそのまま diag として使う. 呼び出し側で
        ``problem`` を別途参照し続けても Observable 側の diag が独立な
        実体になるよう, 内部で ``.copy()`` を取る.

        Parameters
        ----------
        problem
            ``IsingProblem`` インスタンス. ``H_p_diag`` (shape ``(2**n,)``
            float64) を Observable の diag として採用する.

        Returns
        -------
        Observable
            ``problem.H_p_diag`` と数値的に一致 (deep copy で独立) する
            Observable.
        """
        return cls(np.ascontiguousarray(problem.H_p_diag.copy()))
