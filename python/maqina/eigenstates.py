"""瞬時固有状態への投影ユーティリティ.

``H(t)`` の下位 ``k`` 個の固有値・固有ベクトルを Lanczos / dense ``eigh``
ベースで取り出す API を提供する. 時間発展中の波動関数 ``ψ(t)`` を
``eigvecs`` に射影すれば瞬時固有空間内の amplitude が得られる:

    amps = eigvecs.conj().T @ psi_t            # shape (k,)
    probs = np.abs(amps) ** 2

実装方針:

* ``method="lanczos"`` (default): Python ループから
  ``_rust.apply_h_into_py`` (in-place 版) を呼んで Krylov 部分空間
  (次元 ``m``, default 64) を構築し, ``_rust.tridiag_eigh_py`` で三重対角の
  完全固有分解を取って下位 ``k`` 個の Ritz vector を再構築する.
  ``w`` buffer を loop 外で 1 回確保し再利用することで境界の alloc/copy
  overhead を回避する (issue #85). 新規 Rust 関数は追加せず, 既存 primitive を
  Python ループで組み合わせる方針 (固有値計算は時間発展に比べて頻度が
  低く Python 越境のオーバヘッドは無視できる).
* ``method="exact"``: ``n <= 12`` 制限で dense ``H(t)`` を組み立て
  ``numpy.linalg.eigh`` を呼ぶ参照経路.
"""

from __future__ import annotations

import importlib
from typing import Literal

import numpy as np

from maqina.problem import IsingProblem
from maqina.schedule import Schedule

__all__ = ["instantaneous_eigenstates"]


def instantaneous_eigenstates(
    problem: IsingProblem,
    schedule: Schedule,
    t: float,
    k: int = 8,
    method: Literal["lanczos", "exact"] = "lanczos",
    *,
    m: int = 64,
    seed: int | None = None,
    krylov_tol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """瞬時 ``H(t)`` の下位 ``k`` 固有値・固有ベクトルを返す.

    Hamiltonian は ``H(t) = A(s(t)) · H_driver + B(s(t)) · H_problem`` で,
    ``A``, ``B`` は ``schedule`` から評価され, ``H_driver``, ``H_problem``
    は ``problem`` から取り出される (``CLAUDE.md`` 「物理的取り決め」節).

    Parameters
    ----------
    problem
        TFIM 問題定義 (``n``, ``H_p_diag``, ``h_x``).
    schedule
        アニーリングスケジュール (``A(s(t))`` と ``B(s(t))`` を評価).
    t
        評価時刻 (real). ``schedule`` の domain ``[0, T]`` 内が想定だが
        外挿は ``schedule`` 側の挙動に従う.
    k
        取得する下位固有状態数. ``k >= 1`` かつ ``method="lanczos"`` では
        ``k <= m``, ``method="exact"`` では ``k <= 2**n``.
    method
        ``"lanczos"`` (default) は Krylov 部分空間で下位固有値を抽出.
        ``"exact"`` は ``n <= 12`` 制約で dense ``H(t)`` を ``numpy.linalg.eigh``
        にかける参照経路.
    m
        Lanczos 部分空間次元 (default 64). ``method="exact"`` では無視.
        eigenstates 用は時間発展の m (≈24) より大きめに取る (Ritz 値の
        収束を担保するため). ``m >= 1``.
    seed
        Lanczos の始ベクトル生成に使う ``np.random.default_rng`` の seed.
        ``None`` は entropy 由来の seed (再現性なし). ``method="exact"``
        では無視.
    krylov_tol
        ``β_k < krylov_tol`` で Krylov 部分空間構築を早期打切. default
        ``1e-12`` (``docs/design/05-2-lanczos.md`` §5.2 の Lanczos と同じ慣習).

    Returns
    -------
    eigvals : np.ndarray
        shape ``(k,)``, dtype ``float64``. 昇順 (最小固有値が ``eigvals[0]``).
    eigvecs : np.ndarray
        shape ``(2**n, k)``, dtype ``complex128``. 列 ``eigvecs[:, j]`` が
        ``eigvals[j]`` に対応する単位固有ベクトル.

    Raises
    ------
    ValueError
        ``k < 1`` / ``m < 1`` / ``k > m`` (lanczos) /
        ``n > 12`` (exact) / ``k > 2**n`` (exact) /
        未知の ``method`` の場合.
    ImportError
        ``_rust`` 拡張が import できない場合 (``method="lanczos"`` でのみ
        発生; ``method="exact"`` は NumPy のみで完結する).

    Examples
    --------
    >>> import numpy as np
    >>> from maqina import IsingProblem, Schedule
    >>> from maqina.eigenstates import instantaneous_eigenstates
    >>> n = 4
    >>> prob = IsingProblem(
    ...     n=n,
    ...     H_p_diag=np.arange(1 << n, dtype=np.float64),
    ...     h_x=np.ones(n),
    ... )
    >>> sched = Schedule.linear(T=10.0)
    >>> eigvals, eigvecs = instantaneous_eigenstates(prob, sched, t=5.0, k=2)
    >>> eigvals.shape, eigvecs.shape
    ((2,), (16, 2))
    """
    if not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError(f"k must be a positive integer, got {k!r}")
    k = int(k)

    if method == "lanczos":
        return _eigenstates_lanczos(
            problem, schedule, float(t), k, int(m), seed, float(krylov_tol)
        )
    if method == "exact":
        return _eigenstates_exact(problem, schedule, float(t), k)
    raise ValueError(f"method must be 'lanczos' or 'exact', got {method!r}")


def _eigenstates_lanczos(
    problem: IsingProblem,
    schedule: Schedule,
    t: float,
    k: int,
    m: int,
    seed: int | None,
    krylov_tol: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Lanczos 経路の実装本体.

    Park-Light 流の 3 項漸化式 + 2-pass full re-orthogonalization で
    Krylov 部分空間 ``span{v_0, H v_0, H² v_0, ...}`` の正規直交基底
    ``Q ∈ C^{dim × m_eff}`` と三重対角 ``T ∈ R^{m_eff × m_eff}`` を構築する.
    ``T`` の固有分解 ``T = S Λ Sᵀ`` (``Λ`` 昇順) から下位 ``k`` 個の Ritz
    値 ``θ_j = Λ[j]`` および Ritz ベクトル ``Q @ S[:, j]`` を取り出す.

    始ベクトルは ``rng.standard_normal`` で生成した real + imag 各成分の
    複素正規乱数を L2-normalize したもの (``H_driver`` / ``H_problem`` の
    固有空間に偏らせないため real ベクトルではなく complex で取る).

    早期打切 (``β_k < krylov_tol``) は Lanczos が ``H`` の不変部分空間に
    乗ったケース. このとき ``m_eff = k + 1 <= m`` で確定し, それ以降の
    Krylov vector は構築しない. 開始時 ``k > m_eff`` になった場合は
    ``RuntimeError`` を出す (Ritz vector を ``k`` 本要求されているが部分
    空間が足りないため; 初期 vector を変えるか ``m`` を下げる必要がある).
    """
    if m < 1:
        raise ValueError(f"m must be >= 1, got {m!r}")
    if k > m:
        raise ValueError(f"k={k} must be <= m={m}")

    rust = _import_rust()
    n = int(problem.n)
    dim = 1 << n
    if schedule.is_xyz_api:
        raise NotImplementedError(
            "instantaneous_eigenstates does not yet support Schedule.from_xyz "
            "(issue #142 Phase C; eigenstates needs per-axis matvec generalization). "
            "Use a legacy Schedule (X-only TFIM) for now."
        )
    h_x = schedule.h_x
    h_p_diag = problem.H_p_diag
    a_t, b_t = schedule.coeffs_at(t)

    # 始ベクトル: 複素正規乱数を L2-normalize.
    rng = np.random.default_rng(seed)
    psi0 = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
    psi0 = psi0.astype(np.complex128, copy=False)
    norm0 = np.linalg.norm(psi0)
    if norm0 == 0.0:
        # 確率 0 だが念のため.
        psi0[0] = 1.0
        norm0 = 1.0
    psi0 /= norm0

    # V: (m, dim) row-major. 各行が Lanczos vector (行ストレージ).
    # ``apply_h_into_py`` は C-contiguous な (dim,) を要求するので,
    # 行ストレージにしておけば V[j] がそのまま渡せる.
    V = np.zeros((m, dim), dtype=np.complex128)
    alpha = np.zeros(m, dtype=np.float64)
    beta = np.zeros(m, dtype=np.float64)
    V[0] = psi0

    # ``w_buf`` を Krylov loop 外で 1 回確保し毎周再利用 (issue #85).
    # 旧来 ``apply_h_py`` 経路では 1 周ごとに ``dim · 16 B`` の
    # 新規 alloc/copy が発生し m=64 で ~1 GB の不要な heap traffic になる.
    # ``apply_h`` は ``y`` を **上書き** するので ``np.empty`` で
    # 構わない (``np.zeros`` 不要).
    w_buf = np.empty(dim, dtype=np.complex128)

    m_eff = m
    for j in range(m):
        v_j = V[j]
        rust.apply_h_into_py(v_j, w_buf, h_x, h_p_diag, a_t, b_t)
        # ``w`` への以後の `-=` / 正規化が w_buf を破壊しても OK (次周冒頭で
        # apply_h_into_py が再度上書きするため).
        w = w_buf
        # α_j = Re ⟨v_j | w⟩. Hermitian H なら Im 部は浮動小数ノイズ.
        alpha_j = float(np.vdot(v_j, w).real)
        alpha[j] = alpha_j
        # 3-term 漸化式: w -= α_j v_j + β_{j-1} v_{j-1}.
        w -= alpha_j * v_j
        if j >= 1:
            w -= beta[j - 1] * V[j - 1]
        # 2-pass full re-orthogonalization (Daniel-Gragg-Kaufman-Stewart 1976).
        # v_0..v_j の全行に対して ⟨v_i | w⟩ を引き戻す.
        for _pass in range(2):
            for i in range(j + 1):
                proj = np.vdot(V[i], w)
                w -= proj * V[i]
        beta_j = float(np.linalg.norm(w))
        beta[j] = beta_j
        if beta_j < krylov_tol:
            m_eff = j + 1
            break
        if j + 1 < m:
            V[j + 1] = w / beta_j

    if k > m_eff:
        raise RuntimeError(
            f"Lanczos subspace deflated to m_eff={m_eff} before k={k} Ritz "
            f"vectors could be extracted (β fell below krylov_tol={krylov_tol}). "
            f"Try a different seed or reduce k."
        )

    # 三重対角の固有分解.
    alpha_eff = np.ascontiguousarray(alpha[:m_eff])
    beta_eff = np.ascontiguousarray(beta[: m_eff - 1])
    theta, S = rust.tridiag_eigh_py(alpha_eff, beta_eff)
    # Ritz vectors: V[:m_eff].T @ S[:, :k]. V は行ストレージ (行 j = v_j) なので
    # ``V[:m_eff].T`` で (dim, m_eff) の column-vector 表現に戻る.
    eigvals = np.ascontiguousarray(theta[:k])
    eigvecs = V[:m_eff].T @ S[:, :k]
    eigvecs = np.ascontiguousarray(eigvecs.astype(np.complex128, copy=False))
    return eigvals, eigvecs


def _eigenstates_exact(
    problem: IsingProblem,
    schedule: Schedule,
    t: float,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """``method="exact"`` 経路: dense ``H(t)`` を組み立て ``numpy.linalg.eigh``.

    ``H(t)`` の列を ``apply_h_into_py`` (in-place 版) を ``e_j =
    δ_{·, j}`` に当てて 1 列ずつ抽出する. ``out_col`` は列ループ外で
    1 回確保して再利用する (issue #85, 旧 alloc-and-return 経路の Python
    境界 alloc/copy overhead を回避). Kronecker product より重複コードが無く, ビット
    規約 (LSB 規約) の取り違えも自動的に避けられる. ``n <= 12`` 制約は
    ``dim^2`` メモリ (``complex128`` で最大 ``4096^2 × 16 B ≈ 256 MB``) と
    ``eigh`` コスト (``O(dim^3)``) の上限を担保するためのソフトキャップ.
    """
    n = int(problem.n)
    if n > 12:
        raise ValueError(f"method='exact' requires n <= 12, got n={n}")
    dim = 1 << n
    if k > dim:
        raise ValueError(f"k={k} must be <= 2**n = {dim}")

    rust = _import_rust()
    if schedule.is_xyz_api:
        raise NotImplementedError(
            "instantaneous_eigenstates does not yet support Schedule.from_xyz "
            "(issue #142 Phase C; eigenstates needs per-axis matvec generalization). "
            "Use a legacy Schedule (X-only TFIM) for now."
        )
    h_x = schedule.h_x
    h_p_diag = problem.H_p_diag
    a_t, b_t = schedule.coeffs_at(t)

    H = np.empty((dim, dim), dtype=np.complex128)
    e_j = np.zeros(dim, dtype=np.complex128)
    # 列ループ外で 1 本確保し ``apply_h_into_py`` で毎周上書き
    # (issue #85). 旧来 ``apply_h_py`` 経路だと ``dim`` 回ぶんの
    # ``dim · 16 B`` 新規 alloc/copy が走る.
    out_col = np.empty(dim, dtype=np.complex128)
    for j in range(dim):
        e_j[j] = 1.0
        rust.apply_h_into_py(e_j, out_col, h_x, h_p_diag, a_t, b_t)
        H[:, j] = out_col
        e_j[j] = 0.0

    # H は実数係数 (h_x, H_p_diag は float64) なので Hermitian かつ実対称.
    # numpy.linalg.eigh は complex Hermitian を受けるので そのまま渡す.
    eigvals_all, eigvecs_all = np.linalg.eigh(H)
    eigvals = np.ascontiguousarray(eigvals_all[:k])
    eigvecs = np.ascontiguousarray(eigvecs_all[:, :k].astype(np.complex128, copy=False))
    return eigvals, eigvecs


def _import_rust():
    """``_rust`` を import (失敗時は ``ImportError``).

    eigenstates は per-call で多数の matvec を呼ぶため Python リファレンス
    fallback は提供しない (実用にならないため). Rust 拡張が無い環境では
    明示的に ``ImportError`` を投げる.
    """
    try:
        return importlib.import_module("maqina._rust")
    except ImportError as err:
        raise ImportError(
            "maqina._rust is not available; `uv run maturin develop --uv` "
            "to build the Rust extension before calling instantaneous_eigenstates."
        ) from err
