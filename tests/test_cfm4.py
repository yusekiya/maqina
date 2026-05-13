"""CFM4:2 経路 (Phase 3 C2) の等価性 / 性質テスト.

issue #32 (C2) の acceptance:

* ``_python_cfm4_step`` (純 NumPy 経路) と ``_rust.cfm4_step_py`` (Rust
  経路) が ``rel < 1e-13`` で一致する (ランダム ``h_x`` / ``h_p_diag`` /
  ``(a_s1, b_s1, a_s2, b_s2)`` / ``dt``).
* time-independent ``H`` (``a_s1 == a_s2``, ``b_s1 == b_s2``) に対して
  1 step の local truncation error が ``O(dt^5)`` (dt 半減で err 比 ~ 32).
* 全因子が unitary なので ``‖psi_new‖ = ‖psi‖`` が ``rel < 1e-13`` で保たれる.
* ``dt = 0`` で恒等変換.

Rust 単体テスト (``src/cfm4.rs`` の ``#[cfg(test)] mod tests``) と二段で
運用する (``CLAUDE.md`` 参照): Rust 側は係数整合性 / 可換時間依存での
global order / M2 比精度などを別途検証する.
"""

from __future__ import annotations

import numpy as np
import pytest

from kryanneal.krylov import (
    _CFM4_A_HIGH,
    _CFM4_A_LOW,
    _CFM4_C1,
    _CFM4_C2,
    _python_cfm4_step,
)

try:
    from kryanneal import _rust as _rust_mod
except ImportError:  # pragma: no cover
    _rust_mod = None  # type: ignore[assignment]


_HAS_RUST = _rust_mod is not None


def _random_setup(
    n: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float, float]:
    """``(h_x, h_p_diag, psi, a_s1, b_s1, a_s2, b_s2)`` を再現可能に生成する.

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
    a_s1 = float(rng.uniform(-1.0, 1.0))
    b_s1 = float(rng.uniform(-1.0, 1.0))
    a_s2 = float(rng.uniform(-1.0, 1.0))
    b_s2 = float(rng.uniform(-1.0, 1.0))
    return h_x, h_p_diag, psi, a_s1, b_s1, a_s2, b_s2


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


def test_cfm4_constants_match_formula() -> None:
    """係数 ``c_1, c_2, a_high, a_low`` が ``c_1 + c_2 = 1``,
    ``a_high + a_low = 1/2``, ``c_1 < 0.5 < c_2``, ``a_low < 0.25 < a_high``
    の不変量を満たす. Rust 側 ``cfm4_coefficients_match_formula`` と同等のガード.
    """
    assert abs(_CFM4_C1 + _CFM4_C2 - 1.0) < 1e-15
    assert abs(_CFM4_A_HIGH + _CFM4_A_LOW - 0.5) < 1e-15
    assert _CFM4_C1 < 0.5 < _CFM4_C2
    assert _CFM4_A_LOW < 0.25 < _CFM4_A_HIGH


@pytest.mark.skipif(not _HAS_RUST, reason="kryanneal._rust extension not built")
@pytest.mark.parametrize("n", [2, 3, 4, 5])
@pytest.mark.parametrize("seed", [11, 137, 8675309])
def test_python_cfm4_matches_rust(n: int, seed: int) -> None:
    """``_python_cfm4_step`` ↔ ``_rust.cfm4_step_py`` が ``rel < 1e-13``.

    同一 ``(h_x, h_p_diag, a_s1, b_s1, a_s2, b_s2, dt, m, krylov_tol)`` を
    渡し, 浮動小数の演算順序差を超えた数値破綻が無いことを確認する.
    Lanczos 内部の Gram-Schmidt 順序が Python / Rust で揃っているため
    ``rel < 1e-13`` が contract.
    """
    assert _rust_mod is not None
    h_x, h_p_diag, psi, a_s1, b_s1, a_s2, b_s2 = _random_setup(n, seed)
    dt = 0.17
    m = 24
    krylov_tol = 1e-12

    # issue #52 A: cfm4_step (Rust / Python) は (psi, m_eff_sum) を返す.
    # m_eff_sum (= 2 stage の Lanczos 部分空間次元の合計) も完全一致する
    # のが新しい契約 (β_k 早期打切が決定論的).
    psi_py, m_eff_py = _python_cfm4_step(
        psi, h_x, h_p_diag, a_s1, b_s1, a_s2, b_s2, dt, m, krylov_tol
    )
    psi_rust, m_eff_rust = _rust_mod.cfm4_step_py(
        psi, h_x, h_p_diag, a_s1, b_s1, a_s2, b_s2, dt, m, krylov_tol
    )
    assert m_eff_py == m_eff_rust, (
        f"m_eff_sum mismatch: py={m_eff_py}, rust={m_eff_rust}"
    )
    rel = np.linalg.norm(psi_py - psi_rust) / max(np.linalg.norm(psi_rust), 1.0)
    assert rel < 1e-13, f"n={n}, seed={seed}: rel = {rel}"


def test_python_cfm4_preserves_norm() -> None:
    """``_python_cfm4_step`` の 2 stage は両者 unitary, 合成も unitary なので
    ``‖ψ_new‖ = ‖ψ‖`` が ``rel < 1e-13`` で保たれる.
    """
    for n in [2, 3, 4]:
        h_x, h_p_diag, psi, a_s1, b_s1, a_s2, b_s2 = _random_setup(n, seed=4025 + n)
        for dt in [0.05, 0.3, 1.2]:
            psi_new, _m_eff = _python_cfm4_step(
                psi, h_x, h_p_diag, a_s1, b_s1, a_s2, b_s2, dt, 24, 1e-12
            )
            rel = abs(np.linalg.norm(psi_new) - np.linalg.norm(psi)) / max(
                np.linalg.norm(psi), 1.0
            )
            assert rel < 1e-13, f"n={n}, dt={dt}: unitarity violated, rel = {rel}"


def test_python_cfm4_dt_zero_is_identity() -> None:
    """``dt = 0`` で ``exp(-i · 0 · B_2) · exp(-i · 0 · B_1) · ψ = ψ``.
    Lanczos は dt=0 で位相 1 を返すので部分空間内の数値誤差のみ残る
    (``rel < 1e-13``).
    """
    n = 4
    h_x, h_p_diag, psi, a_s1, b_s1, a_s2, b_s2 = _random_setup(n, seed=4026)
    psi_new, _m_eff = _python_cfm4_step(
        psi, h_x, h_p_diag, a_s1, b_s1, a_s2, b_s2, 0.0, 24, 1e-12
    )
    rel = np.linalg.norm(psi_new - psi) / max(np.linalg.norm(psi), 1.0)
    assert rel < 1e-13, f"dt=0 identity violated: rel = {rel}"


def test_python_cfm4_time_independent_lte_order_5() -> None:
    """time-independent ``H`` (``a_s1 == a_s2``, ``b_s1 == b_s2``) では
    1 step の CFM4:2 が厳密に ``exp(-i dt H) · ψ`` を返すべきだが,
    Lanczos の打切り誤差が支配して LTE オーダ実測には不向き.

    そこで「可換だが時間依存」 ``H(t) = f(t) · H_0`` 構造を使う
    (Rust 側 ``cfm4_global_error_order_4_on_commuting_time_dependent_h`` と
    同流). このとき 1 step CFM4:2 は

    .. code-block:: text

        B_1 = (a_high·f(t_1) + a_low ·f(t_2)) · H_0
        B_2 = (a_low ·f(t_1) + a_high·f(t_2)) · H_0

    の両者が H_0 の倍数で可換, propagator は

    .. code-block:: text

        U_step = exp(-i dt · (1/2)(f(t_1) + f(t_2)) · H_0)

    Gauss-Legendre 2 点求積で ``∫_0^{dt} f(τ)dτ`` を近似することと等価.
    f が滑らかなら per-step LTE は ``O(dt^5)``. dt 半減で err 比 ~32 を要求
    (``[16, 64]`` 窓).
    """
    n = 3
    dim = 1 << n
    rng = np.random.default_rng(2030)
    h_x = rng.uniform(-1.0, 1.0, size=n).astype(np.float64)
    h_p_diag = rng.uniform(-1.0, 1.0, size=dim).astype(np.float64)
    psi = (
        rng.uniform(-1.0, 1.0, size=dim) + 1j * rng.uniform(-1.0, 1.0, size=dim)
    ).astype(np.complex128)
    psi = psi / np.linalg.norm(psi)

    a_base = 0.7
    b_base = 0.9

    def f(t: float) -> float:
        return float(np.sin(t))

    dts = [0.2, 0.1, 0.05, 0.025]
    errs: list[float] = []
    for dt in dts:
        # 1 step の参照解: U = exp(-i · F · H_0), F = ∫_0^{dt} f(τ)dτ
        # = cos(0) - cos(dt) = 1 - cos(dt).
        f_int = 1.0 - float(np.cos(dt))
        # H_0 = a_base · H_drv + b_base · diag(h_p_diag) の作用. dense は
        # `_dense_exp_minus_i_dt_h(... dt=1)` の log を取って ·F する代わりに
        # 「a_base · F の H_drv 係数」「b_base · F の diag 係数」で直接構築する.
        expected = _dense_exp_minus_i_dt_h(h_x, h_p_diag, a_base, b_base, f_int) @ psi
        a_s1 = a_base * f(_CFM4_C1 * dt)
        b_s1 = b_base * f(_CFM4_C1 * dt)
        a_s2 = a_base * f(_CFM4_C2 * dt)
        b_s2 = b_base * f(_CFM4_C2 * dt)
        actual, _m_eff = _python_cfm4_step(
            psi, h_x, h_p_diag, a_s1, b_s1, a_s2, b_s2, dt, 24, 1e-14
        )
        errs.append(
            float(
                np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1.0)
            )
        )

    # 単調減少 (符号 / 式の bug 粗検出).
    for i in range(1, len(errs)):
        assert errs[i] < errs[i - 1], f"errs not monotonically decreasing: {errs}"

    # dt 半減で err 比 ≈ 32 (O(dt^5) LTE). FP rounding 床は ~1e-13 程度な
    # ので, 細かい dt 域で頭打ちになる前の 2 区間で比率を確認する.
    # 大きい側 dt=0.2→0.1 の遷移を主に評価し, [16, 64] 窓を許容.
    ratio_coarse = errs[0] / errs[1]
    assert 16.0 <= ratio_coarse <= 64.0, (
        f"dt 0.2 -> 0.1: ratio = {ratio_coarse} (expected ~32), errs = {errs}"
    )
