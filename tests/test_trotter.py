"""Strang 2 次 Trotter 経路 (Phase 2 C3) の等価性 / 性質テスト.

issue #21 の acceptance:

* ``_rust.trotter_step_py`` (Rust 経路) と ``_python_trotter_step``
  (純 NumPy 経路) が ``rel < 1e-13`` で一致する (ランダム ``h_x`` /
  ``h_p_diag`` / ``a_mid`` / ``b_mid`` / ``dt``).
* time-independent ``H`` に対して 1 step の local truncation error が
  ``O(dt^3)`` (dt 半減で err 比 ~ 8).
* 全因子が unitary なので ``‖psi_new‖ = ‖psi‖`` が ``rel < 1e-13`` で保たれる.

Rust 拡張が未ビルドの環境では Python リファレンス単独の性質テスト
(unitarity, dt=0 で恒等, LTE) のみを検証する.
"""

from __future__ import annotations

import numpy as np
import pytest

from kryanneal.krylov import _python_trotter_step

try:
    from kryanneal import _rust as _rust_mod
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

    ``H_drv = -Σ_i h_x_i X_i`` (``apply_h_kryanneal`` と同 convention,
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


@pytest.mark.skipif(not _HAS_RUST, reason="kryanneal._rust extension not built")
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
    psi = (rng.uniform(-1.0, 1.0, size=dim) + 1j * rng.uniform(-1.0, 1.0, size=dim)).astype(
        np.complex128
    )
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
    psi = (rng.uniform(-1.0, 1.0, size=dim) + 1j * rng.uniform(-1.0, 1.0, size=dim)).astype(
        np.complex128
    )
    psi = psi / np.linalg.norm(psi)
    a_mid = 0.7
    b_mid = 0.9

    dts = [0.2, 0.1, 0.05, 0.025]
    errs: list[float] = []
    for dt in dts:
        expected = _dense_exp_minus_i_dt_h(h_x, h_p_diag, a_mid, b_mid, dt) @ psi
        actual = _python_trotter_step(psi, h_x, h_p_diag, a_mid, b_mid, dt)
        errs.append(float(np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1.0)))

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
