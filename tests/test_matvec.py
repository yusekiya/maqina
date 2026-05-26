"""``_rust.apply_h_kinema_py`` の Python smoke test.

Rust 側 ``#[cfg(test)] mod tests`` で nalgebra dense 構築との rel < 1e-13
一致を網羅的に検証している (``src/matvec.rs``). 本テストは

1. PyO3 wrap が numpy array を正しく授受できること,
2. ``y = A·H_drv·v + B·diag(H_p_diag)·v`` の物理 contract が Python
   越境後も保たれていること

の **smoke 1 個** に絞る. Phase 1 段階リリース計画 (``docs/design/12-release-plan.md`` §12).
"""

from __future__ import annotations

import numpy as np
import pytest


def test_apply_h_kinema_py_matches_dense_reference() -> None:
    """``_rust.apply_h_kinema_py`` の出力が dense 構築 H · v と一致する.

    n=3 (dim=8) で固定 seed の擬似乱数を使い, 結果が **relative error
    1e-13 未満** で dense reference と一致することを確認する.
    Rust 拡張未ビルドの環境では skip (fallback 経路は別テストで網羅).
    """
    _rust = pytest.importorskip("maqina._rust")

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
    y = _rust.apply_h_kinema_py(v, h_x, h_p_diag, a_t, b_t)

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


def test_apply_h_kinema_py_simd_path_smoke_for_i0_i1_i2() -> None:
    """SIMD bit-flip pass (i=0,1,2) を踏みやすい設定で apply_h_kinema_py が
    dense reference と ``rel < 1e-13`` で一致することを確認する (issue #63).

    SIMD 経路の有無は build 時の ``simd`` feature で決まる. default build
    (``__has_simd__ = True``) では Rust 側 ``apply_h_kinema_serial`` /
    ``_rayon`` のいずれも i ∈ {0,1,2} について ``simd_kernels::bitflip_iN``
    に dispatch される. SIMD ON のビルドが正しい数値を返すことを Python 越境
    後にも smoke 確認する役割 (Rust 単体テスト
    ``simd_bitflip_kernels_match_scalar_fuzz_100iter`` で fuzz 網羅済み).

    n=5 (dim=32) は SIMD i=2 (block=8) を 4 block 踏み, i=0,1 はその上をなぞる
    最小サイズ. n を小さく保ち rel が ulp 累積で破綻しないことも同時確認.
    """
    _rust = pytest.importorskip("maqina._rust")

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

    y = _rust.apply_h_kinema_py(v, h_x, h_p_diag, a_t, b_t)

    # 参照: dense H · v.
    h_drv = np.zeros((dim, dim), dtype=np.complex128)
    for i in range(n):
        mask = 1 << i
        for k in range(dim):
            h_drv[k, k ^ mask] += -h_x[i]
    h_dense = a_t * h_drv + b_t * np.diag(h_p_diag).astype(np.complex128)
    y_expected = h_dense @ v

    rel = float(np.linalg.norm(y - y_expected) / max(np.linalg.norm(y_expected), 1.0))
    assert rel < 1e-13, f"SIMD-path apply_h_kinema_py rel={rel} >= 1e-13"


def test_apply_h_kinema_into_py_matches_alloc_variant_bitwise() -> None:
    """``apply_h_kinema_into_py`` の結果が ``apply_h_kinema_py`` と
    **bit-for-bit** 一致する (issue #85 acceptance).

    両者は内部で同じ ``apply_h_kinema`` を呼ぶので, ``y`` を新規 alloc
    して返すか caller 提供の buffer に上書きするかが唯一の違い. 演算順序
    は完全に同一なので bit-identical を期待する.

    SIMD path を踏みやすいよう n=5 (i ∈ {0,1,2} の SIMD block ≤ dim) で
    実施する.
    """
    _rust = pytest.importorskip("maqina._rust")

    n = 5
    dim = 1 << n
    rng = np.random.default_rng(20260518)
    h_x = rng.uniform(-1.0, 1.0, size=n).astype(np.float64)
    h_p_diag = rng.uniform(-1.0, 1.0, size=dim).astype(np.float64)
    v = (
        rng.uniform(-1.0, 1.0, size=dim) + 1j * rng.uniform(-1.0, 1.0, size=dim)
    ).astype(np.complex128)
    a_t = float(rng.uniform(-1.0, 1.0))
    b_t = float(rng.uniform(-1.0, 1.0))

    y_alloc = _rust.apply_h_kinema_py(v, h_x, h_p_diag, a_t, b_t)

    y_inplace = np.empty(dim, dtype=np.complex128)
    ret = _rust.apply_h_kinema_into_py(v, y_inplace, h_x, h_p_diag, a_t, b_t)
    assert ret is None  # PyResult<()> は Python では None.

    # 演算順序が同じなので bit-identical を期待 (`np.array_equal` は
    # NaN 同士を等価とみなさないが, ここでは NaN は出ない前提).
    assert np.array_equal(y_inplace, y_alloc), (
        f"in-place / alloc 変種が bitwise 一致しない: max abs diff = "
        f"{np.max(np.abs(y_inplace - y_alloc))}"
    )

    # contract: caller の buffer がそのまま書き換わる.
    assert y_inplace.shape == (dim,)
    assert y_inplace.dtype == np.complex128


def test_apply_h_kinema_into_py_rejects_wrong_shape() -> None:
    """``apply_h_kinema_into_py`` の境界チェック.

    ``y_out`` の長さが ``dim = 2^len(h_x)`` と不一致なら ``ValueError``.
    """
    _rust = pytest.importorskip("maqina._rust")

    n = 3
    dim = 1 << n
    rng = np.random.default_rng(20260518)
    h_x = rng.uniform(-1.0, 1.0, size=n).astype(np.float64)
    h_p_diag = rng.uniform(-1.0, 1.0, size=dim).astype(np.float64)
    v = (
        rng.uniform(-1.0, 1.0, size=dim) + 1j * rng.uniform(-1.0, 1.0, size=dim)
    ).astype(np.complex128)
    a_t, b_t = 0.5, 0.5

    # 長さ違いの y_out で ValueError を期待.
    y_wrong = np.empty(dim + 1, dtype=np.complex128)
    with pytest.raises(ValueError, match="y_out length"):
        _rust.apply_h_kinema_into_py(v, y_wrong, h_x, h_p_diag, a_t, b_t)


@pytest.mark.parametrize("i", [0, 1, 2])
def test_apply_single_mode_axis_i_py_simd_path_smoke(i: int) -> None:
    """``_rust.apply_single_mode_axis_i_py`` の SIMD 経路 smoke (issue #71).

    Phase 6 C2.5 で `apply_single_mode_axis_i` の i ∈ {0,1,2} を
    `wide::f64x4` 特化版 (`simd_kernels::single_mode_iN`) に dispatch する
    変更を入れた. Python 越境後にも dense Kronecker 参照 (`I ⊗ ... ⊗ U_i ⊗
    ... ⊗ I`) と ``rel < 1e-13`` で一致することを smoke 確認する役割.
    Rust 単体テスト ``simd_single_mode_kernels_match_scalar_fuzz_100iter`` で
    fuzz 網羅済み.

    n=5 (dim=32) は SIMD i=2 (block=8) を 4 block 踏み, i=0,1 はその上を
    なぞる最小サイズ. n を小さく保ち rel が ulp 累積で破綻しないことも
    同時確認.
    """
    _rust = pytest.importorskip("maqina._rust")

    n = 5
    dim = 1 << n
    rng = np.random.default_rng(20260517 ^ (i << 8))
    psi = (
        rng.uniform(-1.0, 1.0, size=dim) + 1j * rng.uniform(-1.0, 1.0, size=dim)
    ).astype(np.complex128)
    # ランダム U(2): Trotter R_i(θ) = cos θ · I + i sin θ · X 形を採用すると
    # `[c, i·s, i·s, c]` で 1 自由度しか踏めないので, より一般な U(2) を
    # phase + rotation で組む.
    theta = float(rng.uniform(-np.pi, np.pi))
    alpha = float(rng.uniform(-np.pi, np.pi))
    beta = float(rng.uniform(-np.pi, np.pi))
    c = np.cos(theta)
    s = np.sin(theta)
    u = np.array(
        [
            np.exp(1j * alpha) * c,
            np.exp(1j * beta) * s,
            -np.exp(-1j * beta) * s,
            np.exp(-1j * alpha) * c,
        ],
        dtype=np.complex128,
    )

    # 被テスト: SIMD 経路 (default build) を踏む.
    psi_actual = _rust.apply_single_mode_axis_i_py(psi, u, i, n)

    # 参照: dense `I ⊗ ... ⊗ U_i ⊗ ... ⊗ I` を直接構築する (maqina の
    # bit 規約: bit 0 = LSB, `psi[k]` の bit_i(k)=0 / =1 が pair (lo, hi) を
    # 形成). U_full[k, k]      = u[0] if bit_i(k)=0 else u[3]
    #         U_full[k, k^mask] = u[1] if bit_i(k)=0 else u[2]
    mask = 1 << i
    u_full = np.zeros((dim, dim), dtype=np.complex128)
    for k in range(dim):
        if (k & mask) == 0:
            u_full[k, k] = u[0]
            u_full[k, k ^ mask] = u[1]
        else:
            u_full[k, k] = u[3]
            u_full[k, k ^ mask] = u[2]
    psi_expected = u_full @ psi

    rel = float(
        np.linalg.norm(psi_actual - psi_expected)
        / max(np.linalg.norm(psi_expected), 1.0)
    )
    assert rel < 1e-13, (
        f"SIMD-path apply_single_mode_axis_i_py i={i} rel={rel} >= 1e-13"
    )

    # PyO3 wrap contract.
    assert isinstance(psi_actual, np.ndarray)
    assert psi_actual.shape == (dim,)
    assert psi_actual.dtype == np.complex128


def test_apply_single_mode_axis_i_inplace_py_matches_alloc_variant_bitwise() -> None:
    """``apply_single_mode_axis_i_inplace_py`` の結果が
    ``apply_single_mode_axis_i_py`` と **bit-for-bit** 一致する (issue #86).

    両者は内部で同じ ``apply_single_mode_axis_i`` を呼ぶので, ``psi`` を
    新規 alloc して返すか caller 提供の buffer を in-place 上書きするかが
    唯一の違い. 演算順序は同一なので bit-identical を期待する.

    SIMD path を踏みやすいよう n=5 (i ∈ {0,1,2} の SIMD block ≤ dim) で
    実施し, axis i は 0, 1, 2 を sweep する.
    """
    _rust = pytest.importorskip("maqina._rust")

    n = 5
    dim = 1 << n
    for i in (0, 1, 2, 3, 4):
        rng = np.random.default_rng(20260601 ^ (i << 8))
        psi0 = (
            rng.uniform(-1.0, 1.0, size=dim) + 1j * rng.uniform(-1.0, 1.0, size=dim)
        ).astype(np.complex128)
        u = (
            rng.uniform(-1.0, 1.0, size=4) + 1j * rng.uniform(-1.0, 1.0, size=4)
        ).astype(np.complex128)

        psi_alloc = _rust.apply_single_mode_axis_i_py(psi0, u, i, n)

        psi_inplace = psi0.copy()
        ret = _rust.apply_single_mode_axis_i_inplace_py(psi_inplace, u, i, n)
        assert ret is None  # PyResult<()> は Python では None.

        assert np.array_equal(psi_inplace, psi_alloc), (
            f"i={i} in-place / alloc 変種が bitwise 一致しない: "
            f"max abs diff = {np.max(np.abs(psi_inplace - psi_alloc))}"
        )
        # psi0 が破壊されていないことの確認 (alloc 変種の contract と整合).
        assert not np.array_equal(psi0, psi_inplace), (
            f"psi0 が in-place 経路で書き換わってしまった (i={i})"
        )


def test_apply_single_mode_axis_i_inplace_py_rejects_wrong_shape() -> None:
    """``apply_single_mode_axis_i_inplace_py`` の境界チェック.

    ``psi`` の長さが ``2^n`` と不一致 / ``u`` 長さが 4 でない / ``i >= n`` で
    ``ValueError`` (alloc 変種と同じ shape 検査ヘルパを共有する).
    """
    _rust = pytest.importorskip("maqina._rust")

    n = 3
    dim = 1 << n
    rng = np.random.default_rng(20260601)
    psi = (
        rng.uniform(-1.0, 1.0, size=dim) + 1j * rng.uniform(-1.0, 1.0, size=dim)
    ).astype(np.complex128)
    u = (rng.uniform(-1.0, 1.0, size=4) + 1j * rng.uniform(-1.0, 1.0, size=4)).astype(
        np.complex128
    )

    # psi 長さ違い.
    psi_wrong = np.empty(dim + 1, dtype=np.complex128)
    with pytest.raises(ValueError, match="psi length"):
        _rust.apply_single_mode_axis_i_inplace_py(psi_wrong, u, 0, n)

    # u 長さ違い.
    u_wrong = np.empty(3, dtype=np.complex128)
    with pytest.raises(ValueError, match="length-4"):
        _rust.apply_single_mode_axis_i_inplace_py(psi, u_wrong, 0, n)

    # i >= n.
    with pytest.raises(ValueError, match="must be < n"):
        _rust.apply_single_mode_axis_i_inplace_py(psi, u, n, n)
