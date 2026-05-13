"""時間発展の出力データ構造.

* ``QuantumResult``: ``QuantumAnnealer.run`` の戻り値. 最終波動関数と
  観測量時系列・step 統計を保持する immutable データクラス.
* ``Trajectory``: 観測量の時系列のみを切り出した補助コンテナ
  (post-processing 用途).

Phase 1–4 subset
----------------
``docs/design.md`` §4.4 の ``QuantumResult`` には ``times`` / ``states``
(``store_states`` / ``store_times`` 用) や ``success`` / ``method`` /
``n_steps_actual`` (adaptive driver 用), ``probabilities`` フィールドが
含まれる. 本リリースでは以下を実装する:

* ``psi_final``
* ``t_history``
* ``observables_history``
* ``n_steps``
* ``n_matvec``
* ``success`` (Phase 4 で追加, adaptive driver 失敗時の指標)
* ``method`` (Phase 4 で追加, 実行された propagator 名)
* ``n_steps_actual`` (Phase 4 で追加, adaptive 経路の実 step 数;
  固定 dt 経路では ``n_steps`` と一致)

新フィールドは default 付きで dataclass 末尾に置き backward compatible
な追加とした (既存呼出側は変更不要). ``store_states`` / Observable 連動
の ``times`` / ``states`` / ``observables`` 追加は Phase 5 で導入予定
(parent issue #1 の Out of scope 表).

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
    """時間発展実行結果 (Phase 1–4 subset).

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
        累積 matvec 呼出回数 (Lanczos 内部の ``apply_h_kryanneal`` 含む).
    success
        Phase 4 追加. 駆動が ``RuntimeError`` を出さずに完走したか.
        固定 dt 経路では常に ``True``. adaptive 経路で ``max_rejects``
        連続超過時は ``RuntimeError`` が呼出側に伝播するため, ここまで
        到達したら ``True`` を返す契約 (将来 ``catch`` 経路を入れる場合
        ``False`` を返す余地を残すための signal).
    method
        Phase 4 追加. 実行された propagator 名 (``"m2"`` / ``"trotter"``
        / ``"trotter_suzuki4"`` / ``"cfm4"`` / ``"cfm4_adaptive_richardson"``).
    n_steps_actual
        Phase 4 追加. adaptive 経路で実際に accept された step 数.
        固定 dt 経路では ``n_steps`` と一致する整数値を返し,
        Phase 1–3 互換のために default ``None`` も許容する.
    m_eff_stats
        Phase 4 follow-up (issue #52 A) 追加. adaptive Richardson 経路で
        per-step の Lanczos 部分空間次元合計 ``m_eff_sum`` (= 6 Lanczos
        呼出の m_eff 合計, 早期打切なしで ``6m``) の累積統計. キーは
        ``"total"`` (全 step 合算 = 実 matvec 数の見積もり), ``"mean"``
        (per-step 平均), ``"min"`` / ``"max"`` (per-step 最小 / 最大),
        ``"median"`` (per-step 中央値). 固定 dt 経路 (m2 / trotter /
        cfm4 / trotter_suzuki4) では ``None`` を返す. adaptive M2 経路は
        本 Phase では未対応のため ``None`` (将来 driver 拡張で支援する案
        あり). 値の型は ``"total"`` のみ ``int``, それ以外は ``float``
        (median / mean が非整数値になりうるため).

    Notes
    -----
    Phase 5 で以下のフィールドが追加される予定:

    * ``times`` / ``states`` (``store_states`` 用の中間波動関数列)
    * ``probabilities`` (``|psi_final|^2`` の caching)
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
