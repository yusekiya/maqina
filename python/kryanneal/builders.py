"""``H_problem`` 対角ベクトル構築ヘルパ.

ユーザが k-local 表現 (Pauli term の和 / Sherrington–Kirkpatrick 型の
``J, h``) で問題を書きたい場合に, Z 基底での ``H_p_diag: (2^N,)`` 形に
変換する純粋関数群を提供する.

* ``diag_from_pauli_terms(terms, n)``: Z 演算子のみからなる Pauli term の
  リストを対角ベクトルへ.
* ``diag_from_J_h(J, h)``: ``H = -Σ_{i<j} J_ij Z_i Z_j - Σ_i h_i Z_i`` の
  対角ベクトル化.

ビット規約 (bit 0 = LSB, ``σ_i = 1 - 2·b_i``) は ``CLAUDE.md`` 「物理的
取り決め」節と一致させる.

Phase 1 で実装予定 (現状は API スケルトン).
"""

from __future__ import annotations

__all__: list[str] = []
