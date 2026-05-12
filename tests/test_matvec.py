"""``_rust.apply_h_kryanneal_py`` の Python smoke test.

Rust 側 ``#[cfg(test)] mod tests`` で nalgebra dense 構築との rel < 1e-13
一致を網羅的に検証している (``src/matvec.rs``). 本テストは

1. PyO3 wrap が numpy array を正しく授受できること,
2. ``y = A·H_drv·v + B·diag(H_p_diag)·v`` の物理 contract が Python
   越境後も保たれていること

の **smoke 1 個** に絞る. Phase 1 段階リリース計画 (``docs/design.md`` §12).
"""

from __future__ import annotations

import numpy as np
import pytest


def test_apply_h_kryanneal_py_matches_dense_reference() -> None:
    """``_rust.apply_h_kryanneal_py`` の出力が dense 構築 H · v と一致する.

    n=3 (dim=8) で固定 seed の擬似乱数を使い, 結果が **relative error
    1e-13 未満** で dense reference と一致することを確認する.
    Rust 拡張未ビルドの環境では skip (fallback 経路は別テストで網羅).
    """
    _rust = pytest.importorskip("kryanneal._rust")

    n = 3
    dim = 1 << n
    rng = np.random.default_rng(20251112)
    h_x = rng.uniform(-1.0, 1.0, size=n).astype(np.float64)
    h_p_diag = rng.uniform(-1.0, 1.0, size=dim).astype(np.float64)
    v = (
        rng.uniform(-1.0, 1.0, size=dim) + 1j * rng.uniform(-1.0, 1.0, size=dim)
    ).astype(np.complex128)
    a_t = float(rng.uniform(-1.0, 1.0))
    b_t = float(rng.uniform(-1.0, 1.0))

    # 被テスト.
    y = _rust.apply_h_kryanneal_py(v, h_x, h_p_diag, a_t, b_t)

    # 参照: dense H · v.
    # H_drv = -Σ_i h_x[i] · X_i, X_i は bit i を flip する作用素.
    h_drv = np.zeros((dim, dim), dtype=np.complex128)
    for i in range(n):
        mask = 1 << i
        for k in range(dim):
            h_drv[k, k ^ mask] += -h_x[i]
    h_dense = a_t * h_drv + b_t * np.diag(h_p_diag).astype(np.complex128)
    y_expected = h_dense @ v

    rel = np.linalg.norm(y - y_expected) / max(np.linalg.norm(y_expected), 1.0)
    assert rel < 1e-13, f"relative error {rel} >= 1e-13"

    # PyO3 wrap が返すのは numpy.ndarray, 形状とdtypeも contract.
    assert isinstance(y, np.ndarray)
    assert y.shape == (dim,)
    assert y.dtype == np.complex128
