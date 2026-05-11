"""アニーリングスケジュール (``Schedule``).

``s(t)`` および ``A(s)`` / ``B(s)`` の時間依存パラメータを 1 つの
``Schedule`` オブジェクトに集約する. 公開コンストラクタ:

* ``Schedule.linear(T)``: 標準的な線形スケジュール ``s(t) = t / T``,
  ``A(s) = 1 - s``, ``B(s) = s``.
* ``Schedule.from_callable(s, A, B, T)``: 任意の callable から構築する
  低レベル API.

Phase 1 で実装予定 (現状は API スケルトン).
"""

from __future__ import annotations

__all__: list[str] = []
