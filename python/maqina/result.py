"""時間発展の出力データ構造.

* ``QuantumResult``: ``QuantumAnnealer.run`` の戻り値. 最終波動関数と
  観測量時系列・step 統計を保持する immutable データクラス.
* ``Trajectory``: 観測量の時系列のみを切り出した補助コンテナ
  (post-processing 用途).

保持フィールド
--------------
``docs/design/04-python-api.md`` §4.4 の ``QuantumResult`` に対応するフィールドを保持する
(各フィールドの導入 Phase は ``QuantumResult`` の Parameters を参照):

* ``psi_final`` — 終端 ``ψ(T)`` (常に非 None)
* ``t_history`` — 互換目的の deprecated 別名 (Phase 1 以来の名前を残す;
  Phase 5 からは ``save_tlist`` 経路で ``times`` と同値を返す)
* ``observables_history`` — 観測量時系列 (``save_tlist=None`` のとき空 dict)
* ``n_steps``
* ``n_matvec``
* ``success`` (Phase 4 で追加, adaptive driver 失敗時の指標)
* ``method`` (Phase 4 で追加, 実行された propagator 名)
* ``n_steps_actual`` (Phase 4 で追加, adaptive 経路の実 step 数;
  固定 dt 経路では ``n_steps`` と一致)
* ``m_eff_stats`` (Phase 4 follow-up, issue #52 A)
* ``times`` (Phase 5 で追加, ``save_tlist`` で指定した観測時刻軸;
  ``save_tlist=None`` 経路では ``None``)
* ``states`` (Phase 5 で追加, ``store_states=True`` 経路で指定時刻の
  ψ スナップショット; それ以外は ``None``)
* ``probabilities`` (Phase 5 で追加, ``|psi_final|^2`` の eager 計算;
  どの経路でも常に非 None で返る — 最終状態の付随情報)

新フィールドは default 付きで dataclass 末尾に置き backward compatible
な追加とした (既存呼出側は変更不要).

``@dataclass(frozen=True, eq=False)`` を使う理由は ``problem.py`` と同様で,
``np.ndarray`` を持つため dataclass 既定の ``__eq__`` が
``ValueError: The truth value of an array ...`` で破綻するのを避けるため.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["QuantumResult", "Trajectory"]


@dataclass(frozen=True, eq=False)
class Trajectory:
    """観測量時系列の補助コンテナ.

    Parameters
    ----------
    t_history
        shape ``(K,)`` の float64. サンプル時刻列.
    observables_history
        ``{name: ndarray of shape (K,)}``. 各観測量の時系列値.
        ``QuantumAnnealer.run`` の ``observables`` 引数に渡された
        観測量ごとに 1 エントリ.
    """

    t_history: np.ndarray
    observables_history: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass(frozen=True, eq=False)
class QuantumResult:
    """時間発展実行結果 (frozen dataclass).

    Parameters
    ----------
    psi_final
        shape ``(2**n,)`` complex128. 終端 ``ψ(T)``.
    t_history
        shape ``(K,)`` float64 または ``None``. 観測量を記録した時刻列.
        Phase 1 driver では各 step 終了時刻が記録される想定.
    observables_history
        ``{name: ndarray of shape (K,)}``. 各観測量の時系列値.
        観測量を渡さない場合は空 dict.
    n_steps
        要求された driver step 数. adaptive 経路では呼出側 (``QuantumAnnealer``)
        が要求値を別途持たないため, 実 step 数と一致する.
    n_matvec
        累積 matvec 呼出回数 (Lanczos 内部の ``apply_h`` 含む).
    success
        駆動が ``RuntimeError`` を出さずに完走したか (Phase 4). 固定 dt 経路
        では常に ``True``. adaptive 経路で ``max_rejects`` 連続超過時は
        ``RuntimeError`` が呼出側に伝播するため, ここまで到達したら ``True``
        を返す契約 (将来 ``catch`` 経路を入れる場合 ``False`` を返す余地を
        残すための signal).
    method
        実行された propagator 名 (``"m2"`` / ``"trotter"`` /
        ``"trotter_suzuki4"`` / ``"cfm4"`` / ``"cfm4_adaptive_richardson_krylov"``
        / ``"cfm4_adaptive_richardson_chebyshev"``; Phase 4, Chebyshev variant
        は Phase B #122). ``m_eff_stats`` のキー意味は Lanczos / Chebyshev で
        「per-step propagator 評価コスト統計」として共用.
    n_steps_actual
        adaptive 経路で実際に accept された step 数 (Phase 4). 固定 dt 経路
        では ``n_steps`` と一致する整数を返す (default ``None`` も許容).
    m_eff_stats
        adaptive Richardson 経路で per-step の Lanczos 部分空間次元合計
        ``m_eff_sum`` (= 6 Lanczos 呼出の m_eff 合計, 早期打切なしで ``6m``)
        の累積統計 (Phase 4 follow-up, issue #52 A). キーは ``"total"``
        (全 step 合算 = 実 matvec 数の見積もり), ``"mean"`` / ``"min"`` /
        ``"max"`` / ``"median"`` (per-step 統計). 値の型は ``"total"`` のみ
        ``int``, それ以外は ``float``. 固定 dt 経路と adaptive M2 経路
        (未対応) では ``None``.
    times
        ``QuantumAnnealer.run`` に ``save_tlist`` を渡したときに記録される
        観測時刻軸 (Phase 5, issue #47). shape ``(K,)`` float64 で
        ``save_tlist`` と同一の値, ``observables_history`` / ``states`` の
        time index と整合する. ``save_tlist=None`` (最節約モード) では
        ``None``.
    states
        ``store_states=True`` かつ ``save_tlist`` 非 None の経路でのみ非 None
        (Phase 5, issue #47). shape ``(K, 2**n)`` complex128, ``states[i]``
        が ``times[i]`` 時刻での ψ スナップショット. それ以外では ``None``.
    probabilities
        ``psi_final`` の振幅二乗 ``|psi_final|^2`` を eager 計算した shape
        ``(2**n,)`` float64 (Phase 5, issue #47). 最終状態の付随情報なので
        ``save_tlist`` の有無に依らずどの経路でも常に非 None で返る.
    beta_m_stats
        adaptive Richardson 経路で per-step の Lanczos a posteriori 誤差代表値
        ``β_m_eff = err_lanczos_total / dt`` の累積統計 (Phase 7, issue #93).
        キーは ``"mean"`` / ``"median"`` / ``"min"`` / ``"max"`` / ``"p10"``
        / ``"p90"`` で全て ``float``. 固定 dt 経路では ``None``. 値が小さい
        (≪ ``tol_step``) なら Krylov 部分空間で十分閉じており, 大きい
        (~ ``tol_step`` 以上) なら ``m`` 増大を検討すべき診断指標.
    n_krylov_insufficient
        adaptive Richardson 経路で ``err_lanczos_total > tol_step`` を検出した
        累積 step 数 (Phase 7, issue #93). Krylov 充分性の集計診断指標で,
        0 なら全 step が Krylov 充分, 非ゼロなら ``m`` 増大を検討する.
        固定 dt 経路では ``None``.
    """

    psi_final: np.ndarray
    t_history: np.ndarray | None
    observables_history: dict[str, np.ndarray]
    n_steps: int
    n_matvec: int
    success: bool = True
    method: str = "m2"
    n_steps_actual: int | None = None
    m_eff_stats: dict[str, int | float] | None = None
    times: np.ndarray | None = None
    states: np.ndarray | None = None
    probabilities: np.ndarray | None = None
    beta_m_stats: dict[str, float] | None = None
    n_krylov_insufficient: int | None = None
