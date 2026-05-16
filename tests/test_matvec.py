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


def test_apply_h_kryanneal_py_simd_path_smoke_for_i0_i1_i2() -> None:
    """SIMD bit-flip pass (i=0,1,2) を踏みやすい設定で apply_h_kryanneal_py が
    dense reference と ``rel < 1e-13`` で一致することを確認する (issue #63).

    SIMD 経路の有無は build 時の ``simd`` feature で決まる. default build
    (``__has_simd__ = True``) では Rust 側 ``apply_h_kryanneal_serial`` /
    ``_rayon`` のいずれも i ∈ {0,1,2} について ``simd_kernels::bitflip_iN``
    に dispatch される. SIMD ON のビルドが正しい数値を返すことを Python 越境
    後にも smoke 確認する役割 (Rust 単体テスト
    ``simd_bitflip_kernels_match_scalar_fuzz_100iter`` で fuzz 網羅済み).

    n=5 (dim=32) は SIMD i=2 (block=8) を 4 block 踏み, i=0,1 はその上をなぞる
    最小サイズ. n を小さく保ち rel が ulp 累積で破綻しないことも同時確認.
    """
    _rust = pytest.importorskip("kryanneal._rust")

    n = 5
    dim = 1 << n
    rng = np.random.default_rng(20260517)
    h_x = rng.uniform(-1.0, 1.0, size=n).astype(np.float64)
    h_p_diag = rng.uniform(-1.0, 1.0, size=dim).astype(np.float64)
    v = (
        rng.uniform(-1.0, 1.0, size=dim) + 1j * rng.uniform(-1.0, 1.0, size=dim)
    ).astype(np.complex128)
    a_t = float(rng.uniform(-1.0, 1.0))
    b_t = float(rng.uniform(-1.0, 1.0))

    y = _rust.apply_h_kryanneal_py(v, h_x, h_p_diag, a_t, b_t)

    # 参照: dense H · v.
    h_drv = np.zeros((dim, dim), dtype=np.complex128)
    for i in range(n):
        mask = 1 << i
        for k in range(dim):
            h_drv[k, k ^ mask] += -h_x[i]
    h_dense = a_t * h_drv + b_t * np.diag(h_p_diag).astype(np.complex128)
    y_expected = h_dense @ v

    rel = float(np.linalg.norm(y - y_expected) / max(np.linalg.norm(y_expected), 1.0))
    assert rel < 1e-13, f"SIMD-path apply_h_kryanneal_py rel={rel} >= 1e-13"
