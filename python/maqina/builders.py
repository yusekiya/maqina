"""``H_problem`` 対角ベクトル構築ヘルパ.

ユーザが k-local 表現 (Z 演算子のみからなる Pauli term の和 / Sherrington–
Kirkpatrick 型の ``J, h``) で問題を書きたい場合に, Z 基底での
``H_p_diag: (2^N,)`` 形に変換する純粋関数群を提供する. ``IsingProblem`` は
対角ベクトルを **受け取る** だけのデータコンテナで, k-local 表現の展開は
本モジュールが担う (利用は任意; ユーザが自前で対角ベクトルを構築して
``IsingProblem`` に渡してもよい).

* :func:`diag_from_pauli_terms`: ``(coeff, sites)`` の Pauli-Z term リストを
  対角ベクトルへ.
* :func:`diag_from_J_h`: ``H = -Σ_{i<j} J_ij Z_i Z_j - Σ_i h_i Z_i`` の
  対角ベクトル化.

ビット規約 (bit 0 = LSB, ``x = Σ_i b_i · 2^i``, spin ``σ_i = 1 - 2·b_i``) は
``CLAUDE.md`` 「物理的取り決め」節と一致させる. いずれの関数も ``2^N`` 長の
配列を allocate するため ``N <= 28`` 程度が現実的な上限.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

__all__ = ["diag_from_J_h", "diag_from_pauli_terms"]


def diag_from_pauli_terms(
    n: int,
    terms: Iterable[tuple[float, tuple[int, ...]]],
) -> np.ndarray:
    """Pauli-Z term の和から ``H_p_diag`` を構築する (Z のみ, 対角).

    各 term ``(coeff, sites)`` は演算子 ``coeff · Z_{i_1} Z_{i_2} ⋯`` を表す
    (``sites = (i_1, i_2, ⋯)``). 計算基底 ``|x⟩`` における固有値は
    ``coeff · Π_j σ_{i_j}`` (``σ_i = 1 - 2·b_i ∈ {+1, -1}``) であり, これを
    長さ ``2^n`` の配列に **加算で蓄積** する. Hamiltonian は項の和なので,
    同じ (または順序違いの) サイト集合を持つ term が複数渡された場合はその
    係数が足し合わされる (``Z_i Z_j = Z_j Z_i`` で対角・可換なため順序は不問).

    演算子種は Z 固定 (本モジュールの対象は「Z 基底で対角な ``H_problem``」)
    なので, 演算子文字列は指定しない. 何体項かは ``len(sites)`` で決まる.

    Parameters
    ----------
    n
        スピン数. ``n >= 1``. ``dim = 2**n`` を allocate するので ``n <= 28``
        程度が現実的.
    terms
        ``(coeff, sites)`` の iterable. ``coeff`` は実数係数, ``sites`` は
        作用サイトの ``tuple[int, ...]``. 空タプル ``()`` は単位演算子 ``I``
        を表し, ``coeff`` を全成分に加算する (エネルギーオフセット).

    Returns
    -------
    np.ndarray
        shape ``(2**n,)``, dtype ``float64`` の C-contiguous な対角ベクトル.

    Raises
    ------
    ValueError
        以下のいずれかに該当する場合.

        * ``n`` が 1 以上の整数でない
        * いずれかの term の ``sites`` に範囲外 (``< 0`` または ``>= n``) の
          インデックスが含まれる
        * いずれかの term の ``sites`` 内でサイトが重複する (1 項内のサイトは
          相異なるべき; ``Z_i² = I`` の簡約は行わない)
    """
    if not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError(f"n must be a positive integer, got {n!r}")
    n = int(n)
    dim = 1 << n

    diag = np.zeros(dim, dtype=np.float64)
    # x[k] = 計算基底 |k⟩ のビット表現. σ_i は (x >> i) & 1 から導く.
    x = np.arange(dim, dtype=np.int64)

    for term_index, (coeff, sites) in enumerate(terms):
        site_tuple = tuple(int(s) for s in sites)
        for s in site_tuple:
            if s < 0 or s >= n:
                raise ValueError(
                    f"terms[{term_index}] has site index {s} out of range [0, {n})"
                )
        if len(set(site_tuple)) != len(site_tuple):
            raise ValueError(
                f"terms[{term_index}] has duplicate site indices {site_tuple}; "
                f"sites within a single term must be distinct (Z_i^2 = I is "
                f"not auto-simplified)"
            )
        # Π_j σ_{i_j}. 空タプルなら全成分 1 (= 単位演算子) で coeff を素通し.
        product = np.ones(dim, dtype=np.float64)
        for s in site_tuple:
            product *= 1.0 - 2.0 * ((x >> s) & 1)
        diag += float(coeff) * product

    return diag


def diag_from_J_h(J: np.ndarray, h: np.ndarray | None = None) -> np.ndarray:
    """``H_p = -Σ_{i<j} J_ij σ_i σ_j - Σ_i h_i σ_i`` の対角を構築する.

    Sherrington–Kirkpatrick 型の結合行列 ``J`` と局所場 ``h`` から, Z 基底に
    おける ``H_problem`` の対角ベクトル ``(2^n,)`` を作る. ``σ_i = 1 - 2·b_i``
    (基底 ``|x⟩`` のビット ``b_i`` 由来) を 1 度だけ precompute し, 上三角
    ``i < j`` のペア項と局所場項を蓄積する (符号は反強磁性慣習に合わせ負).

    Parameters
    ----------
    J
        shape ``(n, n)`` の実対称行列. 対角成分は ``J_ii = 0`` (自己結合なし).
        ``i < j`` の上三角のみ使用する (対称性より ``J_ij = J_ji``).
    h
        shape ``(n,)`` の実ベクトル (局所縦磁場). ``None`` (既定) は局所場なし
        (``h = 0``) を意味し, 内部で零ベクトルとして扱う (``h = 0`` のケースが
        多いため省略可能).

    Returns
    -------
    np.ndarray
        shape ``(2**n,)``, dtype ``float64`` の C-contiguous な対角ベクトル.

    Raises
    ------
    ValueError
        以下のいずれかに該当する場合.

        * ``J`` が 2 次元正方行列でない, または複素数を含む
        * ``J`` が対称でない, または対角成分が 0 でない
        * ``h`` が ``None`` でなく, shape が ``(n,)`` でない, または複素数を含む
    """
    J_arr = np.asarray(J)

    if J_arr.ndim != 2 or J_arr.shape[0] != J_arr.shape[1]:
        raise ValueError(f"J must be a square 2D array, got shape {J_arr.shape}")
    if np.iscomplexobj(J_arr):
        raise ValueError("J must be real, got a complex array")
    n = J_arr.shape[0]
    if n < 1:
        raise ValueError(f"J must be at least (1, 1), got shape {J_arr.shape}")

    # h=None は局所場なし (h=0) を意味する.
    if h is None:
        h_arr = np.zeros(n, dtype=np.float64)
    else:
        h_arr = np.asarray(h)
        if h_arr.shape != (n,):
            raise ValueError(f"h shape mismatch: expected ({n},), got {h_arr.shape}")
        if np.iscomplexobj(h_arr):
            raise ValueError("h must be real, got a complex array")

    J_f = J_arr.astype(np.float64, copy=False)
    h_f = h_arr.astype(np.float64, copy=False)
    if not np.allclose(J_f, J_f.T, atol=1e-12):
        raise ValueError("J must be symmetric (J_ij == J_ji)")
    if not np.allclose(np.diag(J_f), 0.0, atol=1e-12):
        raise ValueError("J must have zero diagonal (no self-coupling, J_ii = 0)")

    dim = 1 << n
    x = np.arange(dim, dtype=np.int64)
    # sigma[i] = σ_i over basis (shape (n, dim)). 1 度だけ precompute.
    sigma = np.empty((n, dim), dtype=np.float64)
    for i in range(n):
        sigma[i] = 1.0 - 2.0 * ((x >> i) & 1)

    diag = np.zeros(dim, dtype=np.float64)
    # ペア項 -Σ_{i<j} J_ij σ_i σ_j (上三角のみ; 対称性で 2 重計上を避ける).
    for i in range(n):
        row = J_f[i]
        for j in range(i + 1, n):
            jij = row[j]
            if jij != 0.0:
                diag -= jij * sigma[i] * sigma[j]
    # 局所場項 -Σ_i h_i σ_i.
    for i in range(n):
        if h_f[i] != 0.0:
            diag -= h_f[i] * sigma[i]

    return diag
