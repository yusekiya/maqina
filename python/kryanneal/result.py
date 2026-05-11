"""時間発展の出力データ構造.

* ``QuantumResult``: ``QuantumAnnealer.run`` の戻り値. 最終波動関数と
  観測量時系列・step 統計を保持する immutable データクラス.
* ``Trajectory``: 観測量の時系列のみを切り出した補助コンテナ
  (post-processing 用途).

Phase 1 subset
--------------
``docs/design.md`` §4.4 の ``QuantumResult`` には ``times`` / ``states``
(``store_states`` / ``store_times`` 用) や ``success`` / ``method`` /
``n_steps_actual`` (adaptive driver 用), ``probabilities`` フィールドが
含まれるが, Phase 1 では fixed-step M2 driver のみが提供されるため
本リリースでは以下の最小フィールドのみを実装する:

* ``psi_final``
* ``t_history``
* ``observables_history``
* ``n_steps``
* ``n_matvec``

Adaptive / store_states / Observable 連動の追加フィールドは
それぞれ Phase 4 / Phase 5 で導入される (parent issue #1 の Out of scope
表を参照).

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
    """時間発展実行結果 (Phase 1 subset).

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
        実行された driver step 数.
    n_matvec
        累積 matvec 呼出回数 (Lanczos 内部の ``apply_h_kryanneal`` 含む).

    Notes
    -----
    Phase 4 / Phase 5 で以下のフィールドが追加される予定 (本リリース
    では未提供):

    * ``times`` / ``states`` (``store_states`` 用の中間波動関数列)
    * ``success`` / ``method`` / ``n_steps_actual`` (adaptive driver 用)
    * ``probabilities`` (``|psi_final|^2`` の caching)
    """

    psi_final: np.ndarray
    t_history: np.ndarray | None
    observables_history: dict[str, np.ndarray]
    n_steps: int
    n_matvec: int
