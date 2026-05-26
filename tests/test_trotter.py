"""Strang 2 次 / Suzuki 4 次 Trotter 経路 (Phase 2 C3 / C4) の等価性 / 性質テスト.

issue #21 (Strang) の acceptance:

* ``_rust.trotter_step_py`` (Rust 経路) と ``_python_trotter_step``
  (純 NumPy 経路) が ``rel < 1e-13`` で一致する (ランダム ``h_x`` /
  ``h_p_diag`` / ``a_mid`` / ``b_mid`` / ``dt``).
* time-independent ``H`` に対して 1 step の local truncation error が
  ``O(dt^3)`` (dt 半減で err 比 ~ 8).
* 全因子が unitary なので ``‖psi_new‖ = ‖psi‖`` が ``rel < 1e-13`` で保たれる.

issue #22 (Suzuki S_4) の acceptance:

* ``_rust.trotter_suzuki4_step_py`` (Rust 経路) と
  ``_python_trotter_suzuki4_step`` (純 NumPy 経路) が ``rel < 1e-13`` で一致.
* time-independent ``H`` に対して 1 step LTE が ``O(dt^5)`` (dt 半減で
  err 比 ~ 32).
* 全因子が unitary (5 サブステップの合成も unitary) で ``rel < 1e-13``.
* 同じ ``n_steps`` で Strang より精度が高い.

Rust 拡張が未ビルドの環境では Python リファレンス単独の性質テスト
(unitarity, dt=0 で恒等, LTE) のみを検証する.
"""

from __future__ import annotations

import numpy as np
import pytest

from maqina.krylov import (
    _SUZUKI4_COEFFS,
    _SUZUKI4_MID_OFFSETS,
    _python_trotter_step,
    _python_trotter_suzuki4_step,
)

try:
    from maqina import _rust as _rust_mod
except ImportError:  # pragma: no cover
    _rust_mod = None  # type: ignore[assignment]


_HAS_RUST = _rust_mod is not None


def _random_setup(
    n: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """``(h_x, h_p_diag, psi, a_mid, b_mid)`` を再現可能に生成する.

    ``psi`` は L2-normalize 済 complex128.
    """
    rng = np.random.default_rng(seed)
    dim = 1 << n
    h_x = rng.uniform(-1.0, 1.0, size=n).astype(np.float64)
    h_p_diag = rng.uniform(-1.0, 1.0, size=dim).astype(np.float64)
    psi = (
        rng.uniform(-1.0, 1.0, size=dim) + 1j * rng.uniform(-1.0, 1.0, size=dim)
    ).astype(np.complex128)
    psi = psi / np.linalg.norm(psi)
    a_mid = float(rng.uniform(-1.0, 1.0))
    b_mid = float(rng.uniform(-1.0, 1.0))
    return h_x, h_p_diag, psi, a_mid, b_mid


def _dense_exp_minus_i_dt_h(
    h_x: np.ndarray, h_p_diag: np.ndarray, a_t: float, b_t: float, dt: float
) -> np.ndarray:
    """``exp(-i dt · (a_t · H_drv + b_t · diag(h_p_diag)))`` の dense 構築.

    ``H_drv = -Σ_i h_x_i X_i`` (``apply_h_kinema`` と同 convention,
    ``coeff = -a_t · h_x_i``).
    """
    n = h_x.shape[0]
    dim = 1 << n
    h = np.diag(b_t * h_p_diag).astype(np.complex128)
    for i in range(n):
        mask = 1 << i
        coeff = -a_t * h_x[i]
        for k in range(dim):
            h[k, k ^ mask] += coeff
    lam, u = np.linalg.eigh(h)
    return u @ np.diag(np.exp(-1j * dt * lam)) @ u.conj().T


@pytest.mark.skipif(not _HAS_RUST, reason="maqina._rust extension not built")
@pytest.mark.parametrize("n", [2, 3, 4, 5])
@pytest.mark.parametrize("seed", [11, 137, 8675309])
def test_python_trotter_matches_rust(n: int, seed: int) -> None:
    """``_python_trotter_step`` ↔ ``_rust.trotter_step_py`` が ``rel < 1e-13``.

    同一 ``(h_x, h_p_diag, a_mid, b_mid, dt)`` を渡し, 浮動小数の演算順序
    差を超えた数値破綻が無いことを確認する.
    """
    assert _rust_mod is not None
    h_x, h_p_diag, psi, a_mid, b_mid = _random_setup(n, seed)
    dt = 0.17

    psi_py = _python_trotter_step(psi, h_x, h_p_diag, a_mid, b_mid, dt)
    psi_rust = _rust_mod.trotter_step_py(psi, h_x, h_p_diag, a_mid, b_mid, dt, n)
    rel = np.linalg.norm(psi_py - psi_rust) / max(np.linalg.norm(psi_rust), 1.0)
    assert rel < 1e-13, f"n={n}, seed={seed}: rel = {rel}"


def test_python_trotter_preserves_norm() -> None:
    """``_python_trotter_step`` の全因子は unitary なので ``‖ψ_new‖ = ‖ψ‖``."""
    for n in [2, 3, 4]:
        h_x, h_p_diag, psi, a_mid, b_mid = _random_setup(n, seed=2025 + n)
        for dt in [0.05, 0.3, 1.2]:
            psi_new = _python_trotter_step(psi, h_x, h_p_diag, a_mid, b_mid, dt)
            rel = abs(np.linalg.norm(psi_new) - np.linalg.norm(psi)) / max(
                np.linalg.norm(psi), 1.0
            )
            assert rel < 1e-13, f"n={n}, dt={dt}: unitarity violated, rel = {rel}"


def test_python_trotter_dt_zero_is_identity() -> None:
    """``dt = 0`` で ``exp(0·H) · ψ = ψ``. phase / R_i すべて単位行列."""
    n = 4
    h_x, h_p_diag, psi, a_mid, b_mid = _random_setup(n, seed=2026)
    psi_new = _python_trotter_step(psi, h_x, h_p_diag, a_mid, b_mid, 0.0)
    rel = np.linalg.norm(psi_new - psi) / max(np.linalg.norm(psi), 1.0)
    assert rel < 1e-13, f"dt=0 identity violated: rel = {rel}"


def test_python_trotter_zero_h_x_reduces_to_diag_propagator() -> None:
    """``h_x = 0`` のとき trotter step は ``exp(-i·b·diag·dt)`` と厳密一致.

    内部の ``R_i(θ=0) = I`` なので 2 つの ``phase_p(dt/2)`` が重なって
    ``phase_p(dt) = exp(-i·b·h_p_diag·dt)`` の対角プロパゲータと数値的に
    一致する (``rel < 1e-13``).
    """
    n = 4
    dim = 1 << n
    rng = np.random.default_rng(2027)
    h_x = np.zeros(n, dtype=np.float64)
    h_p_diag = rng.uniform(-1.0, 1.0, size=dim).astype(np.float64)
    psi = (
        rng.uniform(-1.0, 1.0, size=dim) + 1j * rng.uniform(-1.0, 1.0, size=dim)
    ).astype(np.complex128)
    psi = psi / np.linalg.norm(psi)
    a_mid = 0.7
    b_mid = -0.3
    dt = 0.4

    psi_new = _python_trotter_step(psi, h_x, h_p_diag, a_mid, b_mid, dt)
    expected = np.exp(-1j * b_mid * h_p_diag * dt) * psi
    rel = np.linalg.norm(psi_new - expected) / max(np.linalg.norm(expected), 1.0)
    assert rel < 1e-13, f"zero h_x rel = {rel}"


def test_python_trotter_inverse_dt_negate() -> None:
    """``trotter(dt) ∘ trotter(-dt) ≈ I``.

    phase / R_i の各因子は ``dt → -dt`` で exact inverse になる (sin/cos
    の引数符号反転). 数値誤差は accumulated FP rounding のみ
    (``rel < 1e-12``).
    """
    n = 4
    h_x, h_p_diag, psi, a_mid, b_mid = _random_setup(n, seed=2028)
    for dt in [0.05, 0.3, 1.5]:
        psi_fwd = _python_trotter_step(psi, h_x, h_p_diag, a_mid, b_mid, dt)
        psi_back = _python_trotter_step(psi_fwd, h_x, h_p_diag, a_mid, b_mid, -dt)
        rel = np.linalg.norm(psi_back - psi) / max(np.linalg.norm(psi), 1.0)
        assert rel < 1e-12, f"dt={dt}: inverse rel = {rel}"


def test_python_trotter_time_independent_lte_order_3() -> None:
    """time-independent ``H`` に対する 1 step LTE が ``O(dt^3)``.

    dt を半減するごとに ``err`` が ~ ``1/8`` に減衰する.
    dts = [0.2, 0.1, 0.05, 0.025] で計測し, 中間 2 点の比率を
    ``[5, 11]`` の窓で許容する (Strang の係数次第で ``8`` から少しずれる).
    """
    n = 3
    dim = 1 << n
    rng = np.random.default_rng(2029)
    h_x = rng.uniform(-1.0, 1.0, size=n).astype(np.float64)
    h_p_diag = rng.uniform(-1.0, 1.0, size=dim).astype(np.float64)
    psi = (
        rng.uniform(-1.0, 1.0, size=dim) + 1j * rng.uniform(-1.0, 1.0, size=dim)
    ).astype(np.complex128)
    psi = psi / np.linalg.norm(psi)
    a_mid = 0.7
    b_mid = 0.9

    dts = [0.2, 0.1, 0.05, 0.025]
    errs: list[float] = []
    for dt in dts:
        expected = _dense_exp_minus_i_dt_h(h_x, h_p_diag, a_mid, b_mid, dt) @ psi
        actual = _python_trotter_step(psi, h_x, h_p_diag, a_mid, b_mid, dt)
        errs.append(
            float(
                np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1.0)
            )
        )

    # 単調減少 (符号 / 式の bug 粗検出).
    for i in range(1, len(errs)):
        assert errs[i] < errs[i - 1], f"errs not monotonically decreasing: {errs}"

    # dt 半減で err 比 ≈ 8 (O(dt^3) LTE). 細かい dt 域は FP rounding が
    # 効くので大きい側 3 点だけ確認.
    for i in range(1, len(dts) - 1):
        ratio = errs[i - 1] / errs[i]
        assert 5.0 <= ratio <= 11.0, (
            f"dt {dts[i - 1]} -> {dts[i]}: ratio = {ratio} (expected ~8), errs = {errs}"
        )


# ----------------------------------------------------------------------
# Suzuki S_4 ケース (issue #22)
# ----------------------------------------------------------------------


def _constant_ab_lists(a: float, b: float) -> tuple[np.ndarray, np.ndarray]:
    """time-independent H 用の長さ 5 サブステップ ``(a_list, b_list)``."""
    return np.full(5, a, dtype=np.float64), np.full(5, b, dtype=np.float64)


def test_suzuki4_coeffs_consistency() -> None:
    """係数 ``[p, p, 1-4p, p, p]`` の和が 1, 中点 offset が ``t+dt/2`` を
    中心に対称, p の解析値 (``1 / (4 - 4^{1/3})``) と一致.
    """
    p_expected = 1.0 / (4.0 - 4.0 ** (1.0 / 3.0))
    assert abs(_SUZUKI4_COEFFS[0] - p_expected) < 1e-15
    assert abs(sum(_SUZUKI4_COEFFS) - 1.0) < 1e-15

    offsets = _SUZUKI4_MID_OFFSETS
    # offsets[0] + offsets[4] = 1, offsets[1] + offsets[3] = 1, offsets[2] = 0.5
    assert abs(offsets[0] + offsets[4] - 1.0) < 1e-15
    assert abs(offsets[1] + offsets[3] - 1.0) < 1e-15
    assert abs(offsets[2] - 0.5) < 1e-15


@pytest.mark.skipif(not _HAS_RUST, reason="maqina._rust extension not built")
@pytest.mark.parametrize("n", [2, 3, 4, 5])
@pytest.mark.parametrize("seed", [11, 137, 8675309])
def test_python_trotter_suzuki4_matches_rust(n: int, seed: int) -> None:
    """``_python_trotter_suzuki4_step`` ↔ ``_rust.trotter_suzuki4_step_py``
    が ``rel < 1e-13``. 5 sub-step の ``(a, b)`` をランダムに振り (実 driver
    の中点フリーズが各 sub-step で異なる値になりうる状況を模擬), 同一入力で
    両経路を比較する.
    """
    assert _rust_mod is not None
    h_x, h_p_diag, psi, _, _ = _random_setup(n, seed)
    rng = np.random.default_rng(seed + 1)
    a_list = rng.uniform(-1.0, 1.0, size=5).astype(np.float64)
    b_list = rng.uniform(-1.0, 1.0, size=5).astype(np.float64)
    dt = 0.17

    psi_py = _python_trotter_suzuki4_step(psi, h_x, h_p_diag, a_list, b_list, dt)
    psi_rust = _rust_mod.trotter_suzuki4_step_py(
        psi, h_x, h_p_diag, a_list, b_list, dt, n
    )
    rel = np.linalg.norm(psi_py - psi_rust) / max(np.linalg.norm(psi_rust), 1.0)
    assert rel < 1e-13, f"suzuki4: n={n}, seed={seed}: rel = {rel}"


def test_python_trotter_suzuki4_preserves_norm() -> None:
    """5 sub-step の合成も unitary なので ``‖psi_new‖ = ‖psi‖`` が
    ``rel < 1e-13`` で保たれる.
    """
    for n in [2, 3, 4]:
        h_x, h_p_diag, psi, a_mid, b_mid = _random_setup(n, seed=3025 + n)
        a_list, b_list = _constant_ab_lists(a_mid, b_mid)
        for dt in [0.05, 0.3, 1.2]:
            psi_new = _python_trotter_suzuki4_step(
                psi, h_x, h_p_diag, a_list, b_list, dt
            )
            rel = abs(np.linalg.norm(psi_new) - np.linalg.norm(psi)) / max(
                np.linalg.norm(psi), 1.0
            )
            assert rel < 1e-13, (
                f"suzuki4: n={n}, dt={dt}: unitarity violated, rel = {rel}"
            )


def test_python_trotter_suzuki4_dt_zero_is_identity() -> None:
    """``dt = 0`` で 5 sub-step すべてが identity, 合成も identity."""
    n = 4
    h_x, h_p_diag, psi, a_mid, b_mid = _random_setup(n, seed=3026)
    a_list, b_list = _constant_ab_lists(a_mid, b_mid)
    psi_new = _python_trotter_suzuki4_step(psi, h_x, h_p_diag, a_list, b_list, 0.0)
    rel = np.linalg.norm(psi_new - psi) / max(np.linalg.norm(psi), 1.0)
    assert rel < 1e-13, f"suzuki4: dt=0 identity violated: rel = {rel}"


def test_python_trotter_suzuki4_zero_h_x_reduces_to_diag_propagator() -> None:
    """``h_x = 0`` のとき S_4 は phase_p の合成のみ. 5 sub-step の phase が
    連結して合計 ``dt`` の対角プロパゲータ ``exp(-i·b·diag·dt)`` と数値的に
    一致する (``rel < 1e-13``).
    """
    n = 4
    dim = 1 << n
    rng = np.random.default_rng(3027)
    h_x = np.zeros(n, dtype=np.float64)
    h_p_diag = rng.uniform(-1.0, 1.0, size=dim).astype(np.float64)
    psi = (
        rng.uniform(-1.0, 1.0, size=dim) + 1j * rng.uniform(-1.0, 1.0, size=dim)
    ).astype(np.complex128)
    psi = psi / np.linalg.norm(psi)
    a_mid = 0.7
    b_mid = -0.3
    dt = 0.4
    a_list, b_list = _constant_ab_lists(a_mid, b_mid)

    psi_new = _python_trotter_suzuki4_step(psi, h_x, h_p_diag, a_list, b_list, dt)
    expected = np.exp(-1j * b_mid * h_p_diag * dt) * psi
    rel = np.linalg.norm(psi_new - expected) / max(np.linalg.norm(expected), 1.0)
    assert rel < 1e-13, f"suzuki4: zero h_x rel = {rel}"


def test_python_trotter_suzuki4_inverse_dt_negate() -> None:
    """``S_4(dt) ∘ S_4(-dt) ≈ I``. Suzuki S_4 は time-symmetric なので
    同じ sub-step リストを使って ``dt`` の符号だけ反転すれば inverse になる.
    数値誤差は accumulated FP rounding のみ (``rel < 1e-12``).
    """
    n = 4
    h_x, h_p_diag, psi, a_mid, b_mid = _random_setup(n, seed=3028)
    a_list, b_list = _constant_ab_lists(a_mid, b_mid)
    for dt in [0.05, 0.3, 1.5]:
        psi_fwd = _python_trotter_suzuki4_step(psi, h_x, h_p_diag, a_list, b_list, dt)
        psi_back = _python_trotter_suzuki4_step(
            psi_fwd, h_x, h_p_diag, a_list, b_list, -dt
        )
        rel = np.linalg.norm(psi_back - psi) / max(np.linalg.norm(psi), 1.0)
        assert rel < 1e-12, f"suzuki4: dt={dt}: inverse rel = {rel}"


def test_python_trotter_suzuki4_time_independent_lte_order_5() -> None:
    """time-independent ``H`` に対する 1 step LTE が ``O(dt^5)``.

    ``dt`` を半減するごとに ``err`` が ~ ``1/32`` に減衰する.
    dts = [0.4, 0.2, 0.1] で計測し, 各窓の比率を ``[16, 64]`` で許容する.
    """
    n = 3
    dim = 1 << n
    rng = np.random.default_rng(3029)
    h_x = rng.uniform(-1.0, 1.0, size=n).astype(np.float64)
    h_p_diag = rng.uniform(-1.0, 1.0, size=dim).astype(np.float64)
    psi = (
        rng.uniform(-1.0, 1.0, size=dim) + 1j * rng.uniform(-1.0, 1.0, size=dim)
    ).astype(np.complex128)
    psi = psi / np.linalg.norm(psi)
    a_mid = 0.7
    b_mid = 0.9
    a_list, b_list = _constant_ab_lists(a_mid, b_mid)

    dts = [0.4, 0.2, 0.1]
    errs: list[float] = []
    for dt in dts:
        expected = _dense_exp_minus_i_dt_h(h_x, h_p_diag, a_mid, b_mid, dt) @ psi
        actual = _python_trotter_suzuki4_step(psi, h_x, h_p_diag, a_list, b_list, dt)
        errs.append(
            float(
                np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1.0)
            )
        )

    for i in range(1, len(errs)):
        assert errs[i] < errs[i - 1], (
            f"suzuki4 errs not monotonically decreasing: {errs}"
        )

    for i in range(1, len(dts)):
        ratio = errs[i - 1] / errs[i]
        assert 16.0 <= ratio <= 64.0, (
            f"suzuki4 dt {dts[i - 1]} -> {dts[i]}: ratio = {ratio} (expected ~32), errs = {errs}"
        )


def test_python_trotter_suzuki4_more_accurate_than_strang() -> None:
    """同じ ``dt`` (= 1 step) で Suzuki S_4 が Strang より厳密に精度が高い.
    LTE オーダ違い (dt^5 vs dt^3) なので dt = 0.2 程度で必ず差が出る.
    ratio 下限は保守的に 5× で設定し, Suzuki 係数の数値オーダ依存性に
    余裕を持たせる.
    """
    n = 3
    dim = 1 << n
    rng = np.random.default_rng(3030)
    h_x = rng.uniform(-1.0, 1.0, size=n).astype(np.float64)
    h_p_diag = rng.uniform(-1.0, 1.0, size=dim).astype(np.float64)
    psi = (
        rng.uniform(-1.0, 1.0, size=dim) + 1j * rng.uniform(-1.0, 1.0, size=dim)
    ).astype(np.complex128)
    psi = psi / np.linalg.norm(psi)
    a_mid = 0.7
    b_mid = 0.9
    a_list, b_list = _constant_ab_lists(a_mid, b_mid)
    dt = 0.2

    expected = _dense_exp_minus_i_dt_h(h_x, h_p_diag, a_mid, b_mid, dt) @ psi

    psi_strang = _python_trotter_step(psi, h_x, h_p_diag, a_mid, b_mid, dt)
    err_strang = float(
        np.linalg.norm(psi_strang - expected) / max(np.linalg.norm(expected), 1.0)
    )

    psi_s4 = _python_trotter_suzuki4_step(psi, h_x, h_p_diag, a_list, b_list, dt)
    err_s4 = float(
        np.linalg.norm(psi_s4 - expected) / max(np.linalg.norm(expected), 1.0)
    )

    ratio = err_strang / err_s4
    assert ratio > 5.0, (
        f"expected S_4 to be more accurate than Strang, but err_strang / err_s4 = {ratio} "
        f"(err_strang = {err_strang}, err_s4 = {err_s4})"
    )


@pytest.mark.skipif(not _HAS_RUST, reason="maqina._rust extension not built")
@pytest.mark.parametrize("n", [2, 3, 4, 5])
@pytest.mark.parametrize("seed", [11, 137, 8675309])
def test_trotter_step_inplace_py_matches_alloc_variant_bitwise(
    n: int, seed: int
) -> None:
    """``trotter_step_inplace_py`` の結果が ``trotter_step_py`` と
    **bit-for-bit** 一致する (issue #86).

    両者は内部で同じ ``trotter_step`` を呼ぶので, ``psi`` を新規 alloc して
    返すか caller 提供の buffer を in-place 上書きするかが唯一の違い.
    演算順序は完全に同一なので bit-identical を期待する.
    """
    assert _rust_mod is not None
    h_x, h_p_diag, psi, a_mid, b_mid = _random_setup(n, seed)
    dt = 0.17

    psi_alloc = _rust_mod.trotter_step_py(psi, h_x, h_p_diag, a_mid, b_mid, dt, n)

    psi_inplace = psi.copy()
    ret = _rust_mod.trotter_step_inplace_py(
        psi_inplace, h_x, h_p_diag, a_mid, b_mid, dt, n
    )
    assert ret is None
    assert np.array_equal(psi_inplace, psi_alloc), (
        f"n={n}, seed={seed}: in-place / alloc が bitwise 一致しない: "
        f"max abs diff = {np.max(np.abs(psi_inplace - psi_alloc))}"
    )


@pytest.mark.skipif(not _HAS_RUST, reason="maqina._rust extension not built")
@pytest.mark.parametrize("n", [2, 3, 4, 5])
@pytest.mark.parametrize("seed", [11, 137, 8675309])
def test_trotter_suzuki4_step_inplace_py_matches_alloc_variant_bitwise(
    n: int, seed: int
) -> None:
    """``trotter_suzuki4_step_inplace_py`` の結果が ``trotter_suzuki4_step_py``
    と **bit-for-bit** 一致する (issue #86).
    """
    assert _rust_mod is not None
    h_x, h_p_diag, psi, _, _ = _random_setup(n, seed)
    rng = np.random.default_rng(seed + 1)
    a_list = rng.uniform(-1.0, 1.0, size=5).astype(np.float64)
    b_list = rng.uniform(-1.0, 1.0, size=5).astype(np.float64)
    dt = 0.17

    psi_alloc = _rust_mod.trotter_suzuki4_step_py(
        psi, h_x, h_p_diag, a_list, b_list, dt, n
    )

    psi_inplace = psi.copy()
    ret = _rust_mod.trotter_suzuki4_step_inplace_py(
        psi_inplace, h_x, h_p_diag, a_list, b_list, dt, n
    )
    assert ret is None
    assert np.array_equal(psi_inplace, psi_alloc), (
        f"suzuki4: n={n}, seed={seed}: in-place / alloc が bitwise 一致しない: "
        f"max abs diff = {np.max(np.abs(psi_inplace - psi_alloc))}"
    )
