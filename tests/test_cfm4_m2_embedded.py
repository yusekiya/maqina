"""CFM4:2 + M2 embedded error 推定子 (Phase 4 C1) の等価性 / 性質テスト.

issue #39 (C3) の acceptance:

* ``_python_cfm4_step_with_m2_estimate`` (純 NumPy 経路) と
  ``_rust.cfm4_step_with_m2_estimate_py`` (Rust 経路) が ``rel < 1e-13``
  で一致する (ランダム ``h_x`` / ``h_p_diag`` / ノード係数 / ``dt``).
* err のスケーリングが O(dt^3) (dt 半減で err 比 ~ 8) -- M2 LTE が
  ``O(dt^3)`` なので embedded error は M2 と exact の差で支配される.
* ``psi`` 更新が ``cfm4_step`` 単体呼出と bit-exact 一致 (Rust 側契約と
  揃える; Phase 4 C1 の `tests/test_cfm4.py` で既に Rust ↔ Rust の
  bit-exact 一致が確認済).

Rust 単体テスト (``src/cfm4.rs`` の ``#[cfg(test)] mod tests``) と二段で
運用する: Rust 側は LTE スケーリング / cfm4_step との bit-exact 一致 /
M2 経路の他経路への副作用無しなどを別途検証する.
"""

from __future__ import annotations

import numpy as np
import pytest

from maqina.krylov import (
    _CFM4_C1,
    _CFM4_C2,
    _python_cfm4_step,
    _python_cfm4_step_with_m2_estimate,
)

try:
    from maqina import _rust as _rust_mod
except ImportError:  # pragma: no cover
    _rust_mod = None  # type: ignore[assignment]


_HAS_RUST = _rust_mod is not None


def _random_setup(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(h_x, h_p_diag, psi)`` を再現可能に生成する.

    ``psi`` は L2-normalize 済 complex128, ``h_x`` / ``h_p_diag`` は
    ``[-1, 1]`` 一様の f64.
    """
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
    """smooth な time-dependent schedule の closed-form 評価.

    ``A(t) = a_drv · cos(t)``, ``B(t) = b_diag · sin(t)``. CFM4:2 ノードと
    M2 中点ノードでこの関数を呼んで係数を作る.
    """
    return a_drv * float(np.cos(t)), b_diag * float(np.sin(t))


@pytest.mark.skipif(not _HAS_RUST, reason="maqina._rust extension not built")
@pytest.mark.parametrize("n", [2, 3, 4])
@pytest.mark.parametrize("seed", [11, 137, 4093])
def test_python_m2_estimate_matches_rust(n: int, seed: int) -> None:
    """``_python_cfm4_step_with_m2_estimate`` ↔ Rust が ``rel < 1e-13``.

    psi 更新 (CFM4:2 後の状態) と err スカラの両方を Rust 経路と突き合わせる.
    Lanczos 内部の Gram-Schmidt 順序が Python / Rust で揃っているので
    機械精度近傍まで一致するのが contract.
    """
    assert _rust_mod is not None
    h_x, h_p_diag, psi = _random_setup(n, seed)

    rng = np.random.default_rng(seed + 7)
    a_s1 = float(rng.uniform(-1.0, 1.0))
    b_s1 = float(rng.uniform(-1.0, 1.0))
    a_s2 = float(rng.uniform(-1.0, 1.0))
    b_s2 = float(rng.uniform(-1.0, 1.0))
    a_mid = float(rng.uniform(-1.0, 1.0))
    b_mid = float(rng.uniform(-1.0, 1.0))
    dt = 0.13
    m = 24
    krylov_tol = 1e-12

    psi_py, err_py = _python_cfm4_step_with_m2_estimate(
        psi, h_x, h_p_diag, a_s1, b_s1, a_s2, b_s2, a_mid, b_mid, dt, m, krylov_tol
    )
    psi_rust, err_rust = _rust_mod.cfm4_step_with_m2_estimate_py(
        psi, h_x, h_p_diag, a_s1, b_s1, a_s2, b_s2, a_mid, b_mid, dt, m, krylov_tol
    )

    rel_psi = np.linalg.norm(psi_py - psi_rust) / max(np.linalg.norm(psi_rust), 1.0)
    rel_err = abs(err_py - err_rust) / max(abs(err_rust), 1.0)
    assert rel_psi < 1e-13, f"n={n}, seed={seed}: psi rel = {rel_psi}"
    assert rel_err < 1e-13, f"n={n}, seed={seed}: err rel = {rel_err}"


def test_python_m2_estimate_psi_matches_cfm4_step() -> None:
    """``_python_cfm4_step_with_m2_estimate`` の psi 出力は同じ入口 ψ /
    同じノード係数 / 同じ dt で ``_python_cfm4_step`` 単体を呼んだ結果と
    bit-exact 一致する (Rust 側 ``cfm4_step_with_m2_estimate`` と
    ``cfm4_step`` 単体の bit-exact 一致契約を Python 側でも反映).
    """
    n = 4
    h_x, h_p_diag, psi = _random_setup(n, seed=2024)
    a_s1 = 0.31
    b_s1 = -0.42
    a_s2 = 0.55
    b_s2 = 0.18
    a_mid = 0.4
    b_mid = 0.1
    dt = 0.21
    m = 24
    krylov_tol = 1e-12

    psi_estimate, _err = _python_cfm4_step_with_m2_estimate(
        psi, h_x, h_p_diag, a_s1, b_s1, a_s2, b_s2, a_mid, b_mid, dt, m, krylov_tol
    )
    psi_step, _m_eff, _err_lanczos = _python_cfm4_step(
        psi, h_x, h_p_diag, a_s1, b_s1, a_s2, b_s2, dt, m, krylov_tol
    )
    diff = np.linalg.norm(psi_estimate - psi_step)
    assert diff == 0.0, f"psi mismatch: ‖diff‖ = {diff} (expected bit-exact)"


def test_python_m2_estimate_err_scaling_dt_cubed() -> None:
    """smooth time-dependent ``H`` で 1 step の err が ``O(dt^3)`` をスケール.

    M2 LTE が ``O(dt^3)``, CFM4:2 LTE が ``O(dt^5)`` なので embedded
    ``‖ψ_cfm4 - ψ_m2‖`` は M2 経路の誤差で支配され ``O(dt^3)``. dt 半減で
    err 比 ~8 を要求 (``[4, 16]`` 窓; FP rounding 床は ~1e-13 程度なので
    粗い dt で評価).
    """
    n = 3
    h_x, h_p_diag, psi = _random_setup(n, seed=137)

    dts = [0.4, 0.2, 0.1]
    errs: list[float] = []
    for dt in dts:
        a_s1, b_s1 = _smooth_schedule_coeffs(_CFM4_C1 * dt)
        a_s2, b_s2 = _smooth_schedule_coeffs(_CFM4_C2 * dt)
        a_mid, b_mid = _smooth_schedule_coeffs(0.5 * dt)
        _, err = _python_cfm4_step_with_m2_estimate(
            psi, h_x, h_p_diag, a_s1, b_s1, a_s2, b_s2, a_mid, b_mid, dt, 24, 1e-12
        )
        errs.append(err)

    for i in range(1, len(errs)):
        assert errs[i] < errs[i - 1], f"errs not monotonically decreasing: {errs}"
    # dt 半減で err 比 ≈ 8 (O(dt^3) LTE). 粗い dt 域で評価.
    ratio = errs[0] / errs[1]
    assert 4.0 <= ratio <= 16.0, (
        f"dt 0.4 -> 0.2: ratio = {ratio} (expected ~8, O(dt^3)), errs = {errs}"
    )


def test_python_m2_estimate_dt_zero_err_zero() -> None:
    """``dt = 0`` のとき CFM4:2 / M2 ともに identity propagator → err = 0
    (機械精度内).
    """
    n = 3
    h_x, h_p_diag, psi = _random_setup(n, seed=4096)
    psi_new, err = _python_cfm4_step_with_m2_estimate(
        psi, h_x, h_p_diag, 0.3, 0.1, 0.7, 0.2, 0.5, 0.15, 0.0, 24, 1e-12
    )
    # err は機械精度の rounding 程度に小さい (CFM4:2 / M2 双方が identity).
    assert err < 1e-13, f"dt=0 err should be ~0, got {err}"
    rel = np.linalg.norm(psi_new - psi) / max(np.linalg.norm(psi), 1.0)
    assert rel < 1e-13, f"dt=0 identity violated: rel = {rel}"
