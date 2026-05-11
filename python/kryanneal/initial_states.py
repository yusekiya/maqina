"""初期状態構築ヘルパ.

TFIM の標準的な始状態は driver Hamiltonian の最低エネルギー固有状態 ``|+⟩^N``
(各サイトが X 基底の ``|+⟩``). dimension ``2^N`` の complex128 ベクトル
として返す.

* ``uniform_superposition(n)``: ``|+⟩^N = (1/√(2^N)) Σ_x |x⟩``.

ユーザが ``psi0`` を ``QuantumAnnealer`` に渡すときは L2-normalize を
事前に検証される. default は提供しない (``CLAUDE.md`` の取り決め).

Phase 1 で実装予定 (現状は API スケルトン).
"""

from __future__ import annotations

__all__: list[str] = []
