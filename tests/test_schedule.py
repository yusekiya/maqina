"""``Schedule`` の preset / coeffs_at / from_xyz / _eval_stage テスト.

旧 API (一様横磁場 X-only TFIM) と新 API (per-axis 時間依存場, issue #142) の
両方を検証する.

旧 API:
* ``Schedule(T, A, B, h_x, s=None)`` で h_x が必須引数になった (Phase C).
* ``linear`` / ``reverse`` / ``pause`` / ``from_callable`` も h_x 必須化.

新 API:
* ``Schedule.from_xyz(T, g_x, b, g_y=None, g_z=None)`` callable list ベース.
* 内部 evaluator ``_eval_stage(t)`` が ``(g_x_arr, g_y_arr_opt, g_z_arr_opt, b)`` を返す.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from maqina import Schedule


# ---------------------------------------------------------------------------
# 旧 API: Schedule(T, A, B, h_x)
# ---------------------------------------------------------------------------


def test_T_must_be_positive() -> None:
    h_x = np.ones(3, dtype=np.float64)
    with pytest.raises(ValueError, match="T must be positive"):
        Schedule(T=0.0, A=lambda s: 1.0 - s, B=lambda s: s, h_x=h_x)
    with pytest.raises(ValueError, match="T must be positive"):
        Schedule(T=-1.0, A=lambda s: 1.0 - s, B=lambda s: s, h_x=h_x)


def test_h_x_validation() -> None:
    """h_x が ndarray / 1D / float64 / NaN-free であることを検証."""
    # Non-ndarray
    with pytest.raises(ValueError, match="h_x must be a numpy.ndarray"):
        Schedule(T=1.0, A=lambda s: 1.0 - s, B=lambda s: s, h_x=[1.0, 2.0])  # type: ignore[arg-type]
    # Wrong dtype
    with pytest.raises(ValueError, match="h_x dtype must be float64"):
        Schedule(
            T=1.0, A=lambda s: 1.0 - s, B=lambda s: s, h_x=np.ones(3, dtype=np.int64)
        )
    # NaN
    h_x_nan = np.ones(3, dtype=np.float64)
    h_x_nan[0] = np.nan
    with pytest.raises(ValueError, match="h_x contains NaN or inf"):
        Schedule(T=1.0, A=lambda s: 1.0 - s, B=lambda s: s, h_x=h_x_nan)


def test_linear_boundary_values() -> None:
    """``linear(T, h_x)``: s(0)=0, s(T)=1, A(0)=1, B(0)=0, A(T)=0, B(T)=1."""
    T = 5.0
    h_x = np.ones(3, dtype=np.float64)
    sched = Schedule.linear(T, h_x=h_x)
    assert sched.s_at(0.0) == pytest.approx(0.0)
    assert sched.s_at(T) == pytest.approx(1.0)
    a0, b0 = sched.coeffs_at(0.0)
    aT, bT = sched.coeffs_at(T)
    assert a0 == pytest.approx(1.0)
    assert b0 == pytest.approx(0.0)
    assert aT == pytest.approx(0.0)
    assert bT == pytest.approx(1.0)
    # h_x も Schedule に保存されていること.
    np.testing.assert_array_equal(sched.h_x, h_x)
    assert sched.h_x_abs_sum == pytest.approx(float(np.abs(h_x).sum()))


def test_linear_intermediate_monotone() -> None:
    """linear は単調増加の s(t)."""
    T = 4.0
    h_x = np.ones(2, dtype=np.float64)
    sched = Schedule.linear(T, h_x=h_x)
    ts = [i * T / 8 for i in range(9)]
    s_vals = [sched.s_at(t) for t in ts]
    assert all(s2 >= s1 for s1, s2 in zip(s_vals, s_vals[1:], strict=False))
    for t, s in zip(ts, s_vals, strict=True):
        assert s == pytest.approx(t / T)


def test_linear_coeffs_sum_to_one() -> None:
    T = 3.0
    sched = Schedule.linear(T, h_x=np.ones(2, dtype=np.float64))
    for t in [0.0, 0.5, 1.0, 1.5, 2.5, T]:
        a, b = sched.coeffs_at(t)
        assert a + b == pytest.approx(1.0)


def test_reverse_v_shape() -> None:
    T = 10.0
    s_init, s_target = 1.0, 0.5
    h_x = np.ones(2, dtype=np.float64)
    sched = Schedule.reverse(T, h_x=h_x, s_init=s_init, s_target=s_target)
    assert sched.s_at(0.0) == pytest.approx(s_init)
    assert sched.s_at(T / 2) == pytest.approx(s_target)
    assert sched.s_at(T) == pytest.approx(s_init)


def test_reverse_half_monotonic() -> None:
    T = 8.0
    h_x = np.ones(2, dtype=np.float64)
    sched = Schedule.reverse(T, h_x=h_x, s_init=1.0, s_target=0.3)
    ts1 = [i * (T / 2) / 4 for i in range(5)]
    s1 = [sched.s_at(t) for t in ts1]
    assert all(b <= a for a, b in zip(s1, s1[1:], strict=False))
    ts2 = [T / 2 + i * (T / 2) / 4 for i in range(5)]
    s2 = [sched.s_at(t) for t in ts2]
    assert all(b >= a for a, b in zip(s2, s2[1:], strict=False))


def test_pause_constant_region() -> None:
    T = 10.0
    t_pause = 3.0
    duration = 4.0
    h_x = np.ones(2, dtype=np.float64)
    sched = Schedule.pause(T, h_x=h_x, t_pause=t_pause, duration=duration)
    s_hold = sched.s_at(t_pause)
    for t in [t_pause, t_pause + 1.0, t_pause + duration - 1e-9, t_pause + duration]:
        assert sched.s_at(t) == pytest.approx(s_hold)


def test_pause_boundary_s_values() -> None:
    T = 6.0
    t_pause = 1.0
    duration = 2.0
    ramp_total = T - duration
    h_x = np.ones(2, dtype=np.float64)
    sched = Schedule.pause(T, h_x=h_x, t_pause=t_pause, duration=duration)
    assert sched.s_at(0.0) == pytest.approx(0.0)
    assert sched.s_at(T) == pytest.approx(1.0)
    assert sched.s_at(t_pause) == pytest.approx(t_pause / ramp_total)
    s_after_1 = sched.s_at(t_pause + duration + 1.0)
    expected = t_pause / ramp_total + 1.0 / ramp_total
    assert s_after_1 == pytest.approx(expected)


def test_pause_invalid_duration() -> None:
    h_x = np.ones(2, dtype=np.float64)
    with pytest.raises(ValueError, match="duration must satisfy"):
        Schedule.pause(T=5.0, h_x=h_x, t_pause=0.0, duration=-1.0)
    with pytest.raises(ValueError, match="duration must satisfy"):
        Schedule.pause(T=5.0, h_x=h_x, t_pause=0.0, duration=5.0)


def test_pause_invalid_t_pause() -> None:
    h_x = np.ones(2, dtype=np.float64)
    with pytest.raises(ValueError, match="t_pause must satisfy"):
        Schedule.pause(T=5.0, h_x=h_x, t_pause=-1.0, duration=1.0)
    with pytest.raises(ValueError, match="t_pause must satisfy"):
        Schedule.pause(T=5.0, h_x=h_x, t_pause=4.0, duration=2.0)


def test_from_callable_custom_schedule() -> None:
    T = math.pi
    h_x = np.ones(2, dtype=np.float64)
    sched = Schedule.from_callable(
        T=T,
        A=lambda s: s * s,
        B=lambda s: 1.0 - s * s,
        h_x=h_x,
        s=lambda t: math.sin(t / 2),
    )
    a, b = sched.coeffs_at(0.0)
    assert a == pytest.approx(0.0)
    assert b == pytest.approx(1.0)
    a, b = sched.coeffs_at(T)
    assert a == pytest.approx(1.0)
    assert b == pytest.approx(0.0)


def test_default_s_is_linear() -> None:
    T = 2.0
    h_x = np.ones(2, dtype=np.float64)
    sched = Schedule(T=T, A=lambda s: 1.0 - s, B=lambda s: s, h_x=h_x)
    for t in [0.0, 0.5, 1.0, 2.0]:
        assert sched.s_at(t) == pytest.approx(t / T)


# ---------------------------------------------------------------------------
# 新 API: Schedule.from_xyz
# ---------------------------------------------------------------------------


def test_from_xyz_basic() -> None:
    """``from_xyz`` 構築. is_xyz_api / n / T / _eval_stage の基本動作."""
    T = 2.0
    n = 3
    g_x = [lambda t, k=k: float(k + 1) * t for k in range(n)]
    sched = Schedule.from_xyz(T=T, g_x=g_x, b=lambda t: 0.5 * t)
    assert sched.is_xyz_api
    assert sched.n == n
    assert sched.T == pytest.approx(T)
    # _eval_stage で X-only path.
    gx, gy, gz, b = sched._eval_stage(1.0)
    np.testing.assert_array_almost_equal(gx, np.array([1.0, 2.0, 3.0]))
    assert gy is None
    assert gz is None
    assert b == pytest.approx(0.5)


def test_from_xyz_with_y_z() -> None:
    """``from_xyz`` で g_y / g_z を渡したケース."""
    T = 1.0
    g_x = [lambda t: 1.0, lambda t: 2.0]
    g_y = [lambda t: 0.5, lambda t: -0.5]
    g_z = [lambda t: t, lambda t: -t]
    sched = Schedule.from_xyz(T=T, g_x=g_x, b=lambda t: t, g_y=g_y, g_z=g_z)
    gx, gy, gz, b = sched._eval_stage(0.5)
    np.testing.assert_array_almost_equal(gx, np.array([1.0, 2.0]))
    assert gy is not None
    np.testing.assert_array_almost_equal(gy, np.array([0.5, -0.5]))
    assert gz is not None
    np.testing.assert_array_almost_equal(gz, np.array([0.5, -0.5]))
    assert b == pytest.approx(0.5)


def test_from_xyz_validation() -> None:
    with pytest.raises(ValueError, match="T must be positive"):
        Schedule.from_xyz(T=0.0, g_x=[lambda t: 1.0], b=lambda t: 0.0)
    with pytest.raises(ValueError, match="non-empty"):
        Schedule.from_xyz(T=1.0, g_x=[], b=lambda t: 0.0)
    with pytest.raises(ValueError, match="g_y length mismatch"):
        Schedule.from_xyz(
            T=1.0,
            g_x=[lambda t: 1.0, lambda t: 2.0],
            b=lambda t: 0.0,
            g_y=[lambda t: 1.0],  # length 1 vs n=2
        )


def test_from_xyz_disallows_legacy_methods() -> None:
    """``from_xyz`` 構築 schedule では ``s_at`` / ``coeffs_at`` / ``h_x`` が
    RuntimeError を投げる (旧 API 概念が無いため)."""
    sched = Schedule.from_xyz(T=1.0, g_x=[lambda t: 1.0], b=lambda t: 0.0)
    with pytest.raises(RuntimeError, match="only available for legacy"):
        sched.s_at(0.5)
    with pytest.raises(RuntimeError, match="only available for legacy"):
        sched.coeffs_at(0.5)
    with pytest.raises(RuntimeError, match="only available for legacy"):
        _ = sched.h_x
    with pytest.raises(RuntimeError, match="only available for legacy"):
        _ = sched.h_x_abs_sum


def test_legacy_eval_stage_matches_xyz_equivalent() -> None:
    """旧 API と新 API で同じ Hamiltonian を組んだとき ``_eval_stage`` 一致."""
    T = 2.0
    h_x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    sched_old = Schedule.linear(T, h_x=h_x)
    # 新 API: g_x_i(t) = -(1 - t/T) · h_x_i, g_y = g_z = None, b(t) = t/T.
    g_x = [lambda t, hi=float(h): -(1.0 - t / T) * hi for h in h_x]
    sched_new = Schedule.from_xyz(T=T, g_x=g_x, b=lambda t: t / T)
    for t in [0.0, 0.25, 0.5, 1.0, 1.5, T]:
        gx_old, gy_old, gz_old, b_old = sched_old._eval_stage(t)
        gx_new, gy_new, gz_new, b_new = sched_new._eval_stage(t)
        np.testing.assert_allclose(gx_old, gx_new, rtol=1e-13, atol=1e-13)
        assert gy_old is None and gy_new is None
        assert gz_old is None and gz_new is None
        assert b_old == pytest.approx(b_new)
