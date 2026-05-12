"""Krylov + Magnus 時間発展ドライバ.

ここに **公開 driver 関数** (``evolve_schedule_*``) と **Python リファレンス
実装** を置く. Rust 拡張 (``kryanneal._rust``) が利用可能なら fast path に
ディスパッチし, 利用不可なら Python リファレンスで silent fallback する
契約 (詳細は ``docs/design.md`` §3, §5).

Phase 1 実装範囲
----------------
* ``evolve_schedule_m2``: 固定 dt の M2 中点則ドライバ.
* ``_python_lanczos_propagate``: 純 NumPy の Lanczos 短時間プロパゲータ
  リファレンス.
* ``_python_m2_step``: 純 NumPy の M2 中点則 1 step リファレンス.

Phase 2 で Trotter (Strang 2 次) 経路を追加:

* ``evolve_schedule_trotter``: 固定 dt の Strang Trotter ドライバ.
* ``_python_trotter_step``: 純 NumPy の Strang 1 step リファレンス
  (Rust 拡張 ``_rust.trotter_step_py`` と ``rel < 1e-13`` で一致する契約).

Phase 3 で CFM4:2, Phase 4 で adaptive driver
(``evolve_schedule_adaptive_m2`` / ``evolve_schedule_adaptive_richardson``)
を追加する.

Rust 拡張へのアクセスは ``kryanneal._rust`` を **遅延 import** で行う:
``_rust`` のロード失敗 (拡張未ビルド環境) を ``ImportError`` で捕捉して
``_rust`` モジュール参照を ``None`` にし, fast path を選ぶ関数側で
``None`` を見て Python リファレンスにフォールバックする.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Callable

import numpy as np

from kryanneal.schedule import Schedule


def _try_import_rust() -> ModuleType | None:
    """``kryanneal._rust`` を動的 import で取り込む.

    ``importlib`` 経由にすることで, maturin develop 前の状態
    (``_rust.so`` 未生成) を ``ImportError`` で捕捉して silent fallback
    できる. ty / mypy 等の静的解析からは Rust 拡張モジュールが見えないため,
    ``from kryanneal import _rust`` 形式だと未解決 import エラーになる.
    動的 import なら静的解析の対象外になり, 実行時のみ可用性を判定する.
    """
    try:
        return importlib.import_module("kryanneal._rust")
    except ImportError:  # pragma: no cover - 拡張未ビルド環境向けフォールバック
        return None


_rust_mod: ModuleType | None = _try_import_rust()

__all__ = [
    "evolve_schedule_m2",
    "evolve_schedule_trotter",
    "evolve_schedule_trotter_suzuki4",
]


# Trotter-Suzuki S_4 のサブステップ係数係数 `p = 1 / (4 - 4^{1/3})`.
# 5 サブステップの係数は `[p, p, 1 - 4p, p, p]` で和は `1`. 中央 sub-step
# は `1 - 4p ≈ -0.658` で **逆向き**. 詳細は `docs/design.md` §5.3
# (Trotter-Suzuki S_4 サブセクション) と `src/trotter.rs` 冒頭 docstring.
_SUZUKI4_P: float = 1.0 / (4.0 - 4.0 ** (1.0 / 3.0))
_SUZUKI4_COEFFS: tuple[float, float, float, float, float] = (
    _SUZUKI4_P,
    _SUZUKI4_P,
    1.0 - 4.0 * _SUZUKI4_P,
    _SUZUKI4_P,
    _SUZUKI4_P,
)
# 各 sub-step の中点 offset (sub-step `k` を `[t + start_k·dt, t + end_k·dt]`
# としたときの `(start_k + end_k) / 2`). `t + dt/2` を中心に対称.
_SUZUKI4_MID_OFFSETS: tuple[float, float, float, float, float] = (
    0.5 * _SUZUKI4_P,
    1.5 * _SUZUKI4_P,
    0.5,
    1.0 - 1.5 * _SUZUKI4_P,
    1.0 - 0.5 * _SUZUKI4_P,
)


def _python_lanczos_propagate(
    matvec: Callable[[np.ndarray], np.ndarray],
    psi: np.ndarray,
    dt: float,
    m: int,
    tol: float,
) -> np.ndarray:
    """``exp(-i dt H) ψ`` の Lanczos + 三重対角固有分解による近似 (Python リファレンス).

    Rust 側 ``lanczos_propagate`` と同一アルゴリズム (Park-Light 1986,
    full re-orthogonalization 付き) を pure NumPy で実装する.
    ``rel < 1e-13`` で一致するのが契約 (``tests/test_krylov.py``).

    Parameters
    ----------
    matvec
        ``v -> H · v`` の callable. 入力 / 出力は ``(dim,) complex128``.
    psi
        shape ``(dim,)`` complex128 の入力状態.
    dt
        時刻刻み幅 (real).
    m
        Krylov 部分空間次元 (典型値 24, ``m >= 1`` を要求).
    tol
        Lanczos の β 打切り閾値. ``β_k < tol`` で ``m_eff = k+1`` として
        早期終了.

    Returns
    -------
    np.ndarray
        shape ``(dim,)`` complex128 の新状態 ``ψ_new``.

    Raises
    ------
    ValueError
        ``m < 1`` のとき.
    """
    if m < 1:
        raise ValueError(f"m must be >= 1, got {m!r}")
    dim = psi.shape[0]
    if dim == 0:
        return psi.copy()

    psi_norm = float(np.linalg.norm(psi))
    if psi_norm == 0.0:
        return np.zeros_like(psi)

    # V: shape (dim, m), 各列が Lanczos vector v_0..v_{m-1}.
    v_mat = np.zeros((dim, m), dtype=np.complex128)
    alpha = np.zeros(m, dtype=np.float64)
    beta = np.zeros(m, dtype=np.float64)

    v_mat[:, 0] = psi / psi_norm
    m_eff = m

    for k in range(m):
        # w = H · v_k
        w = matvec(v_mat[:, k]).astype(np.complex128, copy=False)

        # α_k = Re ⟨v_k | w⟩ (Hermitian H なら虚部は 0)
        alpha_k = float(np.real(np.vdot(v_mat[:, k], w)))
        alpha[k] = alpha_k

        # w -= α_k · v_k + β_{k-1} · v_{k-1}
        w = w - alpha_k * v_mat[:, k]
        if k >= 1:
            w = w - beta[k - 1] * v_mat[:, k - 1]

        # Full re-orthogonalization (2-pass Gram-Schmidt).
        for _pass in range(2):
            for j in range(k + 1):
                proj = np.vdot(v_mat[:, j], w)
                w = w - proj * v_mat[:, j]

        beta_k = float(np.linalg.norm(w))
        beta[k] = beta_k

        if beta_k < tol:
            m_eff = k + 1
            break

        if k + 1 < m:
            v_mat[:, k + 1] = w / beta_k

    # 三重対角 T (m_eff × m_eff) の固有分解
    # scipy.linalg.eigh_tridiagonal が無くても numpy.linalg.eigh で十分
    # (m_eff ≤ 24 程度なので dense でも数 μs).
    if m_eff == 1:
        lam = alpha[:1].copy()
        q = np.array([[1.0]], dtype=np.float64)
    else:
        t_dense = np.zeros((m_eff, m_eff), dtype=np.float64)
        for i in range(m_eff):
            t_dense[i, i] = alpha[i]
        for i in range(m_eff - 1):
            t_dense[i, i + 1] = beta[i]
            t_dense[i + 1, i] = beta[i]
        lam, q = np.linalg.eigh(t_dense)

    # c = ‖ψ‖ · Q · diag(exp(-i dt λ)) · Qᵀ · e_0
    # numpy.linalg.eigh は q の列が固有ベクトル → Q[i, j] = j 番目固有ベクトル
    # の i 成分. 三重対角規約 (q の行 j が λ_j に対応) と転置の関係:
    # Q^T (numpy) ↔ Q (Rust). e_0 を Q (= q.T) に掛けると q[0, :] = q.T の
    # 0 列目に相当する. ここでは numpy 慣習で q[i, j] = j 番目固有ベクトル
    # の i 番目成分とし, e_0 にあたる成分は q[0, j].
    phases = np.exp(-1j * dt * lam)
    coeff = psi_norm * (q @ (phases * q[0, :]))
    psi_new = v_mat[:, :m_eff] @ coeff
    return psi_new


def _python_m2_step(
    psi: np.ndarray,
    h_x: np.ndarray,
    h_p_diag: np.ndarray,
    a_mid: float,
    b_mid: float,
    dt: float,
    m: int,
    krylov_tol: float,
) -> np.ndarray:
    """M2 中点則 1 step の Python リファレンス実装.

    ``psi_new = exp(-i dt · H(a_mid, b_mid)) · psi`` を
    ``_python_lanczos_propagate`` 経由で計算する. ``a_mid`` / ``b_mid`` は
    呼出側で ``schedule.coeffs_at(t + dt/2)`` を評価して渡す前提.

    Parameters
    ----------
    psi
        shape ``(2**n,)`` complex128 の入力状態.
    h_x
        shape ``(n,)`` float64 のサイト依存横磁場振幅.
    h_p_diag
        shape ``(2**n,)`` float64 の Z 基底 problem 対角.
    a_mid, b_mid
        中点でフリーズ済の ``A(s(t+dt/2))``, ``B(s(t+dt/2))``.
    dt
        時刻刻み幅.
    m
        Krylov 部分空間次元.
    krylov_tol
        Lanczos の β 打切り閾値.

    Returns
    -------
    np.ndarray
        shape ``(2**n,)`` complex128 の新状態.
    """
    matvec = _make_python_matvec(h_x, h_p_diag, a_mid, b_mid)
    return _python_lanczos_propagate(matvec, psi, dt, m, krylov_tol)


def _make_python_matvec(
    h_x: np.ndarray,
    h_p_diag: np.ndarray,
    a_t: float,
    b_t: float,
) -> Callable[[np.ndarray], np.ndarray]:
    """``v -> H(a_t, b_t) · v`` の純 NumPy matvec closure を作る.

    ``H(a_t, b_t) = a_t · H_driver + b_t · diag(h_p_diag)`` (``H_driver =
    -Σ_i h_x_i X_i``). bit-flip 部分は ``i`` 軸ごとに ``np.bitwise_xor`` で
    インデックス並び替えを行うのが最も簡素. dim が大きい場合は cache
    不利だが Phase 1 の参照実装としては許容範囲.
    """
    n = int(h_x.shape[0])
    dim = 1 << n
    if h_p_diag.shape != (dim,):
        raise ValueError(
            f"h_p_diag shape mismatch: expected ({dim},), got {h_p_diag.shape}"
        )

    diag = (b_t * h_p_diag).astype(np.float64, copy=False)
    coeffs = (-a_t * h_x).astype(np.float64, copy=False)
    masks = np.array([1 << i for i in range(n)], dtype=np.int64)
    idx = np.arange(dim, dtype=np.int64)

    def matvec(v: np.ndarray) -> np.ndarray:
        y = diag * v
        for i in range(n):
            if coeffs[i] == 0.0:
                continue
            y = y + coeffs[i] * v[idx ^ int(masks[i])]
        return y

    return matvec


def evolve_schedule_m2(
    h_x: np.ndarray,
    h_p_diag: np.ndarray,
    schedule: Schedule,
    psi0: np.ndarray,
    t0: float,
    t1: float,
    n_steps: int,
    *,
    m: int = 24,
    krylov_tol: float = 1e-12,
) -> tuple[np.ndarray, int]:
    """固定 dt = (t1 - t0) / n_steps の M2 中点則ドライバ.

    各 step で ``schedule.coeffs_at(t + dt/2)`` を評価して
    ``m2_midpoint_step`` を呼ぶ. Rust 拡張が import 済なら
    ``_rust.m2_midpoint_step_py`` を, そうでなければ Python リファレンス
    ``_python_m2_step`` を使う (silent fallback).

    Parameters
    ----------
    h_x
        shape ``(n,)`` float64. サイト依存横磁場振幅.
    h_p_diag
        shape ``(2**n,)`` float64. Z 基底 problem 対角.
    schedule
        ``Schedule`` インスタンス. ``coeffs_at(t)`` から
        ``(A(s(t)), B(s(t)))`` を取り出す.
    psi0
        shape ``(2**n,)`` complex128. 初期状態 (L2-normalize 済みであること).
    t0, t1
        積分区間 ``[t0, t1]``. ``t1 > t0`` を要求.
    n_steps
        固定 step 数 (``n_steps >= 1``).
    m
        Krylov 部分空間次元.
    krylov_tol
        Lanczos の β 打切り閾値.

    Returns
    -------
    psi_final : np.ndarray
        shape ``(2**n,)`` complex128 の終端状態.
    n_matvec : int
        累積 matvec 呼出回数 (Lanczos の ``m`` 回 × ``n_steps`` の見積もり).

    Raises
    ------
    ValueError
        ``n_steps < 1`` または ``t1 <= t0`` のとき.
    """
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps!r}")
    if not (t1 > t0):
        raise ValueError(f"t1 must be > t0, got t0={t0!r}, t1={t1!r}")

    dt = (float(t1) - float(t0)) / int(n_steps)
    psi = np.ascontiguousarray(psi0, dtype=np.complex128)
    h_x_arr = np.ascontiguousarray(h_x, dtype=np.float64)
    h_p_diag_arr = np.ascontiguousarray(h_p_diag, dtype=np.float64)

    rust_mod = _rust_mod
    for k in range(n_steps):
        t_mid = t0 + (k + 0.5) * dt
        a_mid, b_mid = schedule.coeffs_at(t_mid)
        if rust_mod is not None:
            psi = rust_mod.m2_midpoint_step_py(
                psi,
                h_x_arr,
                h_p_diag_arr,
                a_mid,
                b_mid,
                dt,
                m,
                krylov_tol,
            )
        else:
            psi = _python_m2_step(
                psi,
                h_x_arr,
                h_p_diag_arr,
                a_mid,
                b_mid,
                dt,
                m,
                krylov_tol,
            )

    n_matvec = int(n_steps) * int(m)
    return psi, n_matvec


def _python_trotter_step(
    psi: np.ndarray,
    h_x: np.ndarray,
    h_p_diag: np.ndarray,
    a_mid: float,
    b_mid: float,
    dt: float,
) -> np.ndarray:
    """Strang 2 次 Trotter 1 step の Python リファレンス実装.

    Rust 側 ``trotter_step`` と同一アルゴリズム (``src/trotter.rs``):

    .. code-block:: text

        U(dt) ≈ phase_p(dt/2) · (Π_i R_i(dt)) · phase_p(dt/2)

    各因子は

    * ``phase_p(dt/2)``: 各 ``k`` に ``exp(-i · b_mid · h_p_diag[k] · dt/2)`` を乗算.
    * ``R_i(dt)``: ``θ_i = +a_mid · h_x_i · dt`` の 2×2 ユニタリ
      ``[cos θ, i·sin θ; i·sin θ, cos θ]`` を bit ``i`` 軸に in-place 適用.
      符号 convention は ``src/trotter.rs`` 冒頭 docstring 参照 (``H_drv =
      -Σ h_x_i X_i`` の負号は ``θ`` 側に巻き取らず ``apply_h_kryanneal``
      の ``coeff = -a_t · h_x_i`` と統一).

    全因子が unitary なので ``‖psi_new‖ = ‖psi‖`` が machine precision で
    保たれる. ``(a_mid, b_mid)`` は呼出側で ``schedule.coeffs_at(t + dt/2)``
    を評価して渡す前提 (中点採取則; Strang 2 次の対称性を保つために必須).

    Parameters
    ----------
    psi
        shape ``(2**n,)`` complex128 の入力状態.
    h_x
        shape ``(n,)`` float64 のサイト依存横磁場振幅.
    h_p_diag
        shape ``(2**n,)`` float64 の Z 基底 problem 対角.
    a_mid, b_mid
        中点でフリーズ済の ``A(s(t+dt/2))``, ``B(s(t+dt/2))``.
    dt
        時刻刻み幅. 符号は任意 (``-dt`` で逆向きの propagator).

    Returns
    -------
    np.ndarray
        shape ``(2**n,)`` complex128 の新状態.

    Raises
    ------
    ValueError
        ``h_p_diag`` の長さが ``2**len(h_x)`` と一致しないとき.
    """
    n = int(h_x.shape[0])
    dim = 1 << n
    if h_p_diag.shape != (dim,):
        raise ValueError(
            f"h_p_diag shape mismatch: expected ({dim},), got {h_p_diag.shape}"
        )
    if psi.shape != (dim,):
        raise ValueError(f"psi shape mismatch: expected ({dim},), got {psi.shape}")

    half = 0.5 * float(dt)
    # phase_p(dt/2): exp(-i · b_mid · h_p_diag · dt/2) を要素ごとに掛ける.
    phase_half = np.exp(-1j * b_mid * h_p_diag.astype(np.float64, copy=False) * half)
    out = phase_half * psi.astype(np.complex128, copy=True)

    # Π_i R_i(dt): bit i 軸での 2×2 ユニタリ in-place 適用.
    # k と k ^ (1 << i) のペアに対し
    #   u = [[c, i·s], [i·s, c]],   c = cos θ_i, s = sin θ_i, θ_i = a_mid·h_x_i·dt
    # を作用させる. 「bit i = 0 のインデックス集合」と
    # 「bit i = 1 の対応インデックス集合」をペアにして一括計算.
    idx = np.arange(dim, dtype=np.int64)
    for i in range(n):
        theta = float(a_mid) * float(h_x[i]) * float(dt)
        c = np.cos(theta)
        s = np.sin(theta)
        mask = 1 << i
        bit_zero = (idx & mask) == 0
        idx0 = idx[bit_zero]
        idx1 = idx0 ^ mask
        a0 = out[idx0]
        a1 = out[idx1]
        out[idx0] = c * a0 + 1j * s * a1
        out[idx1] = 1j * s * a0 + c * a1

    # phase_p(dt/2) を再度掛ける.
    out *= phase_half
    return out


def evolve_schedule_trotter(
    h_x: np.ndarray,
    h_p_diag: np.ndarray,
    schedule: Schedule,
    psi0: np.ndarray,
    t0: float,
    t1: float,
    n_steps: int,
) -> tuple[np.ndarray, int]:
    """固定 dt = (t1 - t0) / n_steps の Strang Trotter ドライバ.

    各 step で ``schedule.coeffs_at(t + dt/2)`` を評価して
    ``trotter_step`` を呼ぶ. Rust 拡張が import 済なら
    ``_rust.trotter_step_py`` を, そうでなければ Python リファレンス
    ``_python_trotter_step`` を使う (silent fallback).

    Lanczos を介さず ``exp(-i·dt·H_drv) = Π_i R_i(dt)`` を閉形式で書く
    operator splitting 経路. LTE は ``O(dt^3)`` で M2 と同じ局所オーダだが,
    per-step コストは ``(N+1)·dim`` 要素アクセス (m=24 の Lanczos より軽い).
    詳細は ``docs/design.md`` §5.3 の Trotter サブセクションを一次資料とする.

    Parameters
    ----------
    h_x
        shape ``(n,)`` float64. サイト依存横磁場振幅.
    h_p_diag
        shape ``(2**n,)`` float64. Z 基底 problem 対角.
    schedule
        ``Schedule`` インスタンス. ``coeffs_at(t)`` から
        ``(A(s(t)), B(s(t)))`` を取り出す.
    psi0
        shape ``(2**n,)`` complex128. 初期状態 (L2-normalize 済みであること).
    t0, t1
        積分区間 ``[t0, t1]``. ``t1 > t0`` を要求.
    n_steps
        固定 step 数 (``n_steps >= 1``).

    Returns
    -------
    psi_final : np.ndarray
        shape ``(2**n,)`` complex128 の終端状態.
    n_matvec : int
        Trotter 経路は Lanczos を呼ばないため真の matvec カウント概念は
        無いが, ``M2`` ドライバの ``n_steps × m`` と同様の「dim-walk
        見積もり」として ``n_steps × (N + 1)`` (phase pass 1 + bit-flip
        pass N の合計) を返す. ``QuantumResult.n_matvec`` の解釈は
        ``docs/design.md`` §4.4 (Trotter 注記) を参照.

    Raises
    ------
    ValueError
        ``n_steps < 1`` または ``t1 <= t0`` のとき.
    """
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps!r}")
    if not (t1 > t0):
        raise ValueError(f"t1 must be > t0, got t0={t0!r}, t1={t1!r}")

    dt = (float(t1) - float(t0)) / int(n_steps)
    psi = np.ascontiguousarray(psi0, dtype=np.complex128)
    h_x_arr = np.ascontiguousarray(h_x, dtype=np.float64)
    h_p_diag_arr = np.ascontiguousarray(h_p_diag, dtype=np.float64)
    n = int(h_x_arr.shape[0])

    rust_mod = _rust_mod
    for k in range(n_steps):
        t_mid = t0 + (k + 0.5) * dt
        a_mid, b_mid = schedule.coeffs_at(t_mid)
        if rust_mod is not None:
            psi = rust_mod.trotter_step_py(
                psi,
                h_x_arr,
                h_p_diag_arr,
                a_mid,
                b_mid,
                dt,
                n,
            )
        else:
            psi = _python_trotter_step(
                psi,
                h_x_arr,
                h_p_diag_arr,
                a_mid,
                b_mid,
                dt,
            )

    n_matvec = int(n_steps) * (n + 1)
    return psi, n_matvec


def _python_trotter_suzuki4_step(
    psi: np.ndarray,
    h_x: np.ndarray,
    h_p_diag: np.ndarray,
    a_t_list: np.ndarray,
    b_t_list: np.ndarray,
    dt: float,
) -> np.ndarray:
    """4 次 Suzuki Trotter 1 step の Python リファレンス実装.

    Trotter-Suzuki S_4 公式

    .. code-block:: text

        S_4(dt) = S_2(p·dt) · S_2(p·dt) · S_2((1 - 4p)·dt) · S_2(p·dt) · S_2(p·dt)
        p = 1 / (4 - 4^{1/3}) ≈ 0.41449

    で Strang S_2 (``_python_trotter_step``) を 5 回適用する. 各 sub-step の
    ``(a_t, b_t)`` は呼出側で **中点 offset** ``[p/2, 3p/2, 1/2, 1 - 3p/2,
    1 - p/2]`` (``t + dt/2`` を中心に対称) で評価して長さ 5 の配列として
    渡す前提. Rust 側 ``trotter_suzuki4_step`` と ``rel < 1e-13`` で一致する
    のが契約 (``tests/test_trotter.py``).

    全因子が unitary なので ``‖psi_new‖ = ‖psi‖`` が machine precision で
    保たれる. LTE は ``O(dt^5)``.

    Parameters
    ----------
    psi
        shape ``(2**n,)`` complex128 の入力状態.
    h_x
        shape ``(n,)`` float64 のサイト依存横磁場振幅.
    h_p_diag
        shape ``(2**n,)`` float64 の Z 基底 problem 対角.
    a_t_list, b_t_list
        shape ``(5,)`` float64. 各 sub-step の中点で評価された
        ``A(s(·))`` / ``B(s(·))``.
    dt
        外側 1 step の時間刻み. 符号は任意 (``-dt`` で逆向き propagator).

    Returns
    -------
    np.ndarray
        shape ``(2**n,)`` complex128 の新状態.

    Raises
    ------
    ValueError
        ``a_t_list`` / ``b_t_list`` の長さが 5 でないとき, または
        ``h_p_diag`` / ``psi`` の長さが ``2**len(h_x)`` と不整合のとき.
    """
    if a_t_list.shape != (5,):
        raise ValueError(
            f"a_t_list shape mismatch: expected (5,) for Suzuki S_4 sub-steps, got {a_t_list.shape}"
        )
    if b_t_list.shape != (5,):
        raise ValueError(
            f"b_t_list shape mismatch: expected (5,) for Suzuki S_4 sub-steps, got {b_t_list.shape}"
        )
    out = psi
    for k in range(5):
        out = _python_trotter_step(
            out,
            h_x,
            h_p_diag,
            float(a_t_list[k]),
            float(b_t_list[k]),
            _SUZUKI4_COEFFS[k] * float(dt),
        )
    return out


def evolve_schedule_trotter_suzuki4(
    h_x: np.ndarray,
    h_p_diag: np.ndarray,
    schedule: Schedule,
    psi0: np.ndarray,
    t0: float,
    t1: float,
    n_steps: int,
) -> tuple[np.ndarray, int]:
    """固定 dt = (t1 - t0) / n_steps の Suzuki S_4 Trotter ドライバ.

    各 step で 5 sub-step の中点 ``t + offset_k · dt`` (``offset_k ∈ [p/2,
    3p/2, 1/2, 1 - 3p/2, 1 - p/2]``) で ``schedule.coeffs_at`` を評価し,
    ``trotter_suzuki4_step`` (Rust) または ``_python_trotter_suzuki4_step``
    (silent fallback) を呼ぶ.

    Lanczos を介さない operator splitting 経路の 4 次版. LTE は ``O(dt^5)``
    で CFM4:2 と同じ局所オーダだが, per-step は ``5·(N + 1)·dim`` 要素アクセス
    (Strang S_2 の 5 倍, M2 の Lanczos m=24 と比べて N の係数次第).
    詳細は ``docs/design.md`` §5.3 (Trotter-Suzuki S_4 サブセクション).

    Parameters
    ----------
    h_x
        shape ``(n,)`` float64. サイト依存横磁場振幅.
    h_p_diag
        shape ``(2**n,)`` float64. Z 基底 problem 対角.
    schedule
        ``Schedule`` インスタンス.
    psi0
        shape ``(2**n,)`` complex128. 初期状態 (L2-normalize 済みであること).
    t0, t1
        積分区間 ``[t0, t1]``. ``t1 > t0`` を要求.
    n_steps
        固定 step 数 (``n_steps >= 1``).

    Returns
    -------
    psi_final : np.ndarray
        shape ``(2**n,)`` complex128 の終端状態.
    n_matvec : int
        Trotter 経路は Lanczos を呼ばないため真の matvec カウント概念は
        無いが, Strang ドライバとの整合のため
        ``n_steps × 5 × (N + 1)`` (5 sub-step × ``phase pass 1 + bit-flip
        pass N``) を返す. ``QuantumResult.n_matvec`` の解釈は
        ``docs/design.md`` §4.4 (Trotter 注記) を参照.

    Raises
    ------
    ValueError
        ``n_steps < 1`` または ``t1 <= t0`` のとき.
    """
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps!r}")
    if not (t1 > t0):
        raise ValueError(f"t1 must be > t0, got t0={t0!r}, t1={t1!r}")

    dt = (float(t1) - float(t0)) / int(n_steps)
    psi = np.ascontiguousarray(psi0, dtype=np.complex128)
    h_x_arr = np.ascontiguousarray(h_x, dtype=np.float64)
    h_p_diag_arr = np.ascontiguousarray(h_p_diag, dtype=np.float64)
    n = int(h_x_arr.shape[0])
    offsets = np.asarray(_SUZUKI4_MID_OFFSETS, dtype=np.float64)

    rust_mod = _rust_mod
    for k in range(n_steps):
        t_step_start = t0 + k * dt
        a_list = np.empty(5, dtype=np.float64)
        b_list = np.empty(5, dtype=np.float64)
        for j in range(5):
            t_mid = t_step_start + float(offsets[j]) * dt
            a_mid, b_mid = schedule.coeffs_at(t_mid)
            a_list[j] = a_mid
            b_list[j] = b_mid
        if rust_mod is not None:
            psi = rust_mod.trotter_suzuki4_step_py(
                psi,
                h_x_arr,
                h_p_diag_arr,
                a_list,
                b_list,
                dt,
                n,
            )
        else:
            psi = _python_trotter_suzuki4_step(
                psi,
                h_x_arr,
                h_p_diag_arr,
                a_list,
                b_list,
                dt,
            )

    n_matvec = int(n_steps) * 5 * (n + 1)
    return psi, n_matvec
