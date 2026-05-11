"""瞬時固有状態への投影ユーティリティ.

``H(t_k)`` の低次固有状態を Lanczos で抽出し, 時間発展中の波動関数を
それらに射影してプロット用の確率系列を返す. 小規模 ``n`` 向けには dense
``eigh`` ベースの参照実装も同梱して数値一致のテストに使う.

Phase 1 で実装予定 (現状は API スケルトン).
"""

from __future__ import annotations

__all__: list[str] = []
