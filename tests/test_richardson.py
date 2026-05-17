"""CFM4:2 + step-doubling Richardson 推定子 (Phase 4 C2) の等価性 / 性質テスト.

issue #39 (C3) の acceptance:

* ``_python_cfm4_step_with_richardson_estimate`` ↔
  ``_rust.cfm4_step_with_richardson_estimate_py`` が ``rel < 1e-13``
  (psi / err 双方).
* err のスケーリングが ``O(dt^5)`` (CFM4:2 LTE; dt 半減で err 比 ~32).
* ``extrapolate=True`` で実効 6 次精度 (smooth time-dependent ``H`` で
  外挿 OFF 比 ``dt`` 1 つぶん高速に減少 -- 実装は ``ψ_h2`` よりさらに
  1 オーダ精度).
* ``dt = 0`` で ``err = 0`` (機械精度内) かつ psi 更新は identity.

Rust 単体テスト (``src/cfm4.rs`` の ``#[cfg(test)] mod tests``) と二段で
運用する.
"""

from __future__ import annotations

import numpy as np
import pytest

from kryanneal.krylov import (
    _CFM4_C1,
    _CFM4_C2,
    _python_cfm4_step,
    _python_cfm4_step_with_richardson_estimate,
)

try:
    from kryanneal import _rust as _rust_mod
except ImportError:  # pragma: no cover
    _rust_mod = None  # type: ignore[assignment]


_HAS_RUST = _rust_mod is not None


def _random_setup(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    dim = 1 << n
    h_x = rng.uniform(-1.0, 1.0, size=n).astype(np.float64)
    h_p_diag = rng.uniform(-1.0, 1.0, size=dim).astype(np.float64)
    psi = (
        rng.uniform(-1.0, 1.0, size=dim) + 1j * rng.uniform(-1.0, 1.0, size=dim)
    ).astype(np.complex128)
    psi = psi / np.linalg.norm(psi)
    return h_x, h_p_diag, psi


def _smooth_schedule_coeffs(
    t: float, a_drv: float = 0.7, b_diag: float = 0.9
) -> tuple[float, float]:
    """``A(t) = a_drv · cos(t)``, ``B(t) = b_diag · sin(t)``.

    Richardson 推定子のテスト用 smooth time-dependent schedule.
    """
    return a_drv * float(np.cos(t)), b_diag * float(np.sin(t))


def _full_h1_h2_nodes(
    t: float, dt: float
) -> tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]:
    """ある時刻 ``t`` と刻み ``dt`` に対して, full / h1 / h2 の 6 ノードでの
    ``(a, b)`` 係数を ``_smooth_schedule_coeffs`` で評価し, 3 タプルに整理する.

    Returns
    -------
    full
        ``(a_s1_full, b_s1_full, a_s2_full, b_s2_full)``.
    h1
        ``(a_s1_h1, b_s1_h1, a_s2_h1, b_s2_h1)`` (前半 half-step).
    h2
        ``(a_s1_h2, b_s1_h2, a_s2_h2, b_s2_h2)`` (後半 half-step).
    """
    half = 0.5 * dt
    a_s1_full, b_s1_full = _smooth_schedule_coeffs(t + _CFM4_C1 * dt)
    a_s2_full, b_s2_full = _smooth_schedule_coeffs(t + _CFM4_C2 * dt)
    a_s1_h1, b_s1_h1 = _smooth_schedule_coeffs(t + _CFM4_C1 * half)
    a_s2_h1, b_s2_h1 = _smooth_schedule_coeffs(t + _CFM4_C2 * half)
    a_s1_h2, b_s1_h2 = _smooth_schedule_coeffs(t + half + _CFM4_C1 * half)
    a_s2_h2, b_s2_h2 = _smooth_schedule_coeffs(t + half + _CFM4_C2 * half)
    return (
        (a_s1_full, b_s1_full, a_s2_full, b_s2_full),
        (a_s1_h1, b_s1_h1, a_s2_h1, b_s2_h1),
        (a_s1_h2, b_s1_h2, a_s2_h2, b_s2_h2),
    )


@pytest.mark.skipif(not _HAS_RUST, reason="kryanneal._rust extension not built")
@pytest.mark.parametrize("n", [2, 3, 4])
@pytest.mark.parametrize("seed", [11, 137, 4093])
@pytest.mark.parametrize("extrapolate", [False, True])
def test_python_richardson_matches_rust(n: int, seed: int, extrapolate: bool) -> None:
    """``_python_cfm4_step_with_richardson_estimate`` ↔ Rust が ``rel < 1e-13``.

    psi (extrapolate フラグに応じて ``ψ_acc`` or ``ψ_h2``) と err スカラの
    両方を突き合わせる.
    """
    assert _rust_mod is not None
    h_x, h_p_diag, psi = _random_setup(n, seed)
    dt = 0.17
    m = 24
    krylov_tol = 1e-12

    (
        (a_s1_full, b_s1_full, a_s2_full, b_s2_full),
        (
            a_s1_h1,
            b_s1_h1,
            a_s2_h1,
            b_s2_h1,
        ),
        (a_s1_h2, b_s1_h2, a_s2_h2, b_s2_h2),
    ) = _full_h1_h2_nodes(0.5, dt)

    # issue #93 (Phase 7): Rust / Python ref とも
    # (psi, err, m_eff_sum, err_lanczos_total) を返す. 末尾 err_lanczos_total は
    # 6 Lanczos 呼出の a posteriori 誤差上界 triangle inequality 和.
    # 全 4 要素が完全一致が新契約.
    psi_py, err_py, m_eff_py, err_lanczos_py = (
        _python_cfm4_step_with_richardson_estimate(
            psi,
            h_x,
            h_p_diag,
            a_s1_full,
            b_s1_full,
            a_s2_full,
            b_s2_full,
            a_s1_h1,
            b_s1_h1,
            a_s2_h1,
            b_s2_h1,
            a_s1_h2,
            b_s1_h2,
            a_s2_h2,
            b_s2_h2,
            dt,
            m,
            krylov_tol,
            extrapolate,
        )
    )
    psi_rust, err_rust, m_eff_rust, err_lanczos_rust = (
        _rust_mod.cfm4_step_with_richardson_estimate_py(
            psi,
            h_x,
            h_p_diag,
            a_s1_full,
            b_s1_full,
            a_s2_full,
            b_s2_full,
            a_s1_h1,
            b_s1_h1,
            a_s2_h1,
            b_s2_h1,
            a_s1_h2,
            b_s1_h2,
            a_s2_h2,
            b_s2_h2,
            dt,
            m,
            krylov_tol,
            extrapolate,
        )
    )
    assert m_eff_py == m_eff_rust, (
        f"m_eff_sum mismatch: py={m_eff_py}, rust={m_eff_rust}"
    )
    rel_psi = np.linalg.norm(psi_py - psi_rust) / max(np.linalg.norm(psi_rust), 1.0)
    rel_err = abs(err_py - err_rust) / max(abs(err_rust), 1.0)
    rel_err_lanczos = abs(err_lanczos_py - err_lanczos_rust) / max(
        abs(err_lanczos_rust), 1.0
    )
    assert rel_psi < 1e-13, (
        f"n={n}, seed={seed}, extrapolate={extrapolate}: psi rel = {rel_psi}"
    )
    assert rel_err < 1e-13, (
        f"n={n}, seed={seed}, extrapolate={extrapolate}: err rel = {rel_err}"
    )
    assert rel_err_lanczos < 1e-13, (
        f"n={n}, seed={seed}, extrapolate={extrapolate}: "
        f"err_lanczos rel = {rel_err_lanczos}"
    )


def test_python_richardson_err_scaling_dt_fifth() -> None:
    """smooth time-dependent ``H`` で 1 step の err が ``O(dt^5)`` をスケール.

    CFM4:2 LTE は ``O(dt^5)``, half-step×2 の LTE は ``(1/16)`` 倍. 差は
    ``(15/16) · C_4 · dt^5`` で先頭次数の係数まで取れる. dt 半減で err
    比 ~32 を要求 (``[16, 64]`` 窓; FP rounding 床に注意).
    """
    n = 3
    h_x, h_p_diag, psi = _random_setup(n, seed=2024)

    dts = [0.4, 0.2, 0.1, 0.05]
    errs: list[float] = []
    for dt in dts:
        (
            (a_s1_full, b_s1_full, a_s2_full, b_s2_full),
            (a_s1_h1, b_s1_h1, a_s2_h1, b_s2_h1),
            (a_s1_h2, b_s1_h2, a_s2_h2, b_s2_h2),
        ) = _full_h1_h2_nodes(0.3, dt)
        _, err, _m_eff, _err_lanczos = _python_cfm4_step_with_richardson_estimate(
            psi,
            h_x,
            h_p_diag,
            a_s1_full,
            b_s1_full,
            a_s2_full,
            b_s2_full,
            a_s1_h1,
            b_s1_h1,
            a_s2_h1,
            b_s2_h1,
            a_s1_h2,
            b_s1_h2,
            a_s2_h2,
            b_s2_h2,
            dt,
            24,
            1e-14,
            False,
        )
        errs.append(err)

    for i in range(1, len(errs)):
        assert errs[i] < errs[i - 1], f"errs not monotonically decreasing: {errs}"
    # 粗い dt 域で評価 (細かい dt は FP rounding 床に頭打ち).
    ratio = errs[0] / errs[1]
    assert 16.0 <= ratio <= 64.0, (
        f"dt 0.4 -> 0.2: ratio = {ratio} (expected ~32, O(dt^5)), errs = {errs}"
    )


def test_python_richardson_extrapolate_improves_accuracy() -> None:
    """smooth time-dependent ``H`` で ``extrapolate=True`` が ``False``
    比で 1 step error をより速く減らす (実効 6 次精度).

    厳密な参照解は ``可換だが時間依存`` 設定 (``H(t) = f(t) · H_0``) で
    閉形式 ``exp(-i · F · H_0) · ψ`` (``F = ∫_0^{dt} f(τ)dτ``) を取り,
    extrapolate=True と False で 1 step 後の誤差を比較する. dt 半減で
    False は ~32 倍, True は ~64 倍以上に減ることを要求 (smooth schedule
    なら 5 次→6 次の差で True 側がより早く減る).
    """
    n = 3
    dim = 1 << n
    rng = np.random.default_rng(20260513)
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

    def _dense_exp_minus_i_dt_h(
        h_x_arr: np.ndarray,
        h_p_diag_arr: np.ndarray,
        a_t: float,
        b_t: float,
        dt: float,
    ) -> np.ndarray:
        """``test_cfm4.py`` と同型の dense 構築."""
        n_local = h_x_arr.shape[0]
        dim_local = 1 << n_local
        h = np.diag(b_t * h_p_diag_arr).astype(np.complex128)
        for i in range(n_local):
            mask = 1 << i
            coeff = -a_t * h_x_arr[i]
            for k in range(dim_local):
                h[k, k ^ mask] += coeff
        lam, u = np.linalg.eigh(h)
        return u @ np.diag(np.exp(-1j * dt * lam)) @ u.conj().T

    def _ratio_set(extrapolate: bool, dts: list[float]) -> list[float]:
        errs_set: list[float] = []
        for dt in dts:
            f_int = 1.0 - float(np.cos(dt))  # ∫_0^{dt} sin(τ)dτ
            expected = (
                _dense_exp_minus_i_dt_h(h_x, h_p_diag, a_base, b_base, f_int) @ psi
            )
            # full / h1 / h2 ノードを `H(t) = f(t) · H_0` 構造で評価.
            half = 0.5 * dt
            a_s1_full = a_base * f(_CFM4_C1 * dt)
            b_s1_full = b_base * f(_CFM4_C1 * dt)
            a_s2_full = a_base * f(_CFM4_C2 * dt)
            b_s2_full = b_base * f(_CFM4_C2 * dt)
            a_s1_h1 = a_base * f(_CFM4_C1 * half)
            b_s1_h1 = b_base * f(_CFM4_C1 * half)
            a_s2_h1 = a_base * f(_CFM4_C2 * half)
            b_s2_h1 = b_base * f(_CFM4_C2 * half)
            a_s1_h2 = a_base * f(half + _CFM4_C1 * half)
            b_s1_h2 = b_base * f(half + _CFM4_C1 * half)
            a_s2_h2 = a_base * f(half + _CFM4_C2 * half)
            b_s2_h2 = b_base * f(half + _CFM4_C2 * half)
            psi_new, _err, _m_eff, _err_lanczos = (
                _python_cfm4_step_with_richardson_estimate(
                    psi,
                    h_x,
                    h_p_diag,
                    a_s1_full,
                    b_s1_full,
                    a_s2_full,
                    b_s2_full,
                    a_s1_h1,
                    b_s1_h1,
                    a_s2_h1,
                    b_s2_h1,
                    a_s1_h2,
                    b_s1_h2,
                    a_s2_h2,
                    b_s2_h2,
                    dt,
                    24,
                    1e-14,
                    extrapolate,
                )
            )
            errs_set.append(
                float(np.linalg.norm(psi_new - expected) / np.linalg.norm(expected))
            )
        return errs_set

    dts = [0.2, 0.1]
    errs_false = _ratio_set(False, dts)
    errs_true = _ratio_set(True, dts)
    # 同一 dt で True は False より誤差が小さい (smooth schedule で外挿が効く).
    for dt_idx, dt in enumerate(dts):
        assert errs_true[dt_idx] < errs_false[dt_idx], (
            f"extrapolate=True did not improve over False at dt={dt}: "
            f"true={errs_true[dt_idx]}, false={errs_false[dt_idx]}"
        )
    # dt 半減で True 側の誤差比が False 側より大きい (= 高次).
    ratio_false = errs_false[0] / errs_false[1]
    ratio_true = errs_true[0] / errs_true[1]
    assert ratio_true > ratio_false, (
        f"extrapolate=True did not converge faster: "
        f"ratio_true={ratio_true}, ratio_false={ratio_false}"
    )


def test_python_richardson_dt_zero_err_zero() -> None:
    """``dt = 0`` のとき full / half×2 すべて identity → err = 0,
    psi 更新は (extrapolate に依らず) 入口 ψ と一致.
    """
    n = 3
    h_x, h_p_diag, psi = _random_setup(n, seed=4096)
    # ノード係数は任意 (dt=0 で identity).
    for extrapolate in (False, True):
        psi_new, err, _m_eff, _err_lanczos = _python_cfm4_step_with_richardson_estimate(
            psi,
            h_x,
            h_p_diag,
            0.3,
            0.1,
            0.7,
            0.2,
            0.4,
            0.15,
            0.6,
            0.25,
            0.5,
            0.18,
            0.65,
            0.22,
            0.0,
            24,
            1e-12,
            extrapolate,
        )
        assert err < 1e-13, (
            f"extrapolate={extrapolate}: dt=0 err should be ~0, got {err}"
        )
        rel = np.linalg.norm(psi_new - psi) / max(np.linalg.norm(psi), 1.0)
        assert rel < 1e-13, (
            f"extrapolate={extrapolate}: dt=0 identity violated, rel = {rel}"
        )


def test_python_richardson_psi_h2_matches_two_step_cfm4() -> None:
    """``extrapolate=False`` の psi 出力は ``_python_cfm4_step`` を h1/h2 で
    2 段呼んだ結果と bit-exact 一致する (Rust 側と同じ「ψ_h2 を返す」契約).
    """
    n = 4
    h_x, h_p_diag, psi = _random_setup(n, seed=8001)
    dt = 0.27
    (
        (a_s1_full, b_s1_full, a_s2_full, b_s2_full),
        (a_s1_h1, b_s1_h1, a_s2_h1, b_s2_h1),
        (a_s1_h2, b_s1_h2, a_s2_h2, b_s2_h2),
    ) = _full_h1_h2_nodes(0.7, dt)

    psi_new, _err, _m_eff, _err_lanczos = _python_cfm4_step_with_richardson_estimate(
        psi,
        h_x,
        h_p_diag,
        a_s1_full,
        b_s1_full,
        a_s2_full,
        b_s2_full,
        a_s1_h1,
        b_s1_h1,
        a_s2_h1,
        b_s2_h1,
        a_s1_h2,
        b_s1_h2,
        a_s2_h2,
        b_s2_h2,
        dt,
        24,
        1e-12,
        False,
    )
    psi_mid, _m_eff_mid, _err_lanczos_mid = _python_cfm4_step(
        psi, h_x, h_p_diag, a_s1_h1, b_s1_h1, a_s2_h1, b_s2_h1, 0.5 * dt, 24, 1e-12
    )
    psi_h2_ref, _m_eff_h2, _err_lanczos_h2 = _python_cfm4_step(
        psi_mid,
        h_x,
        h_p_diag,
        a_s1_h2,
        b_s1_h2,
        a_s2_h2,
        b_s2_h2,
        0.5 * dt,
        24,
        1e-12,
    )
    diff = np.linalg.norm(psi_new - psi_h2_ref)
    assert diff == 0.0, f"extrapolate=False psi mismatch: ‖diff‖ = {diff}"
