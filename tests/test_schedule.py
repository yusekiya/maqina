"""``Schedule`` の preset / coeffs_at テスト.

``linear`` / ``reverse`` / ``pause`` / ``from_callable`` が ``s(t)`` の
境界値と単調性を満たし, ``coeffs_at(t)`` が ``(A(s(t)), B(s(t)))`` を
正しく返すことを検証する.
"""

from __future__ import annotations

import math

import pytest

from kinema import Schedule


def test_T_must_be_positive() -> None:
    with pytest.raises(ValueError, match="T must be positive"):
        Schedule(T=0.0, A=lambda s: 1.0 - s, B=lambda s: s)
    with pytest.raises(ValueError, match="T must be positive"):
        Schedule(T=-1.0, A=lambda s: 1.0 - s, B=lambda s: s)


def test_linear_boundary_values() -> None:
    """``linear(T)``: s(0)=0, s(T)=1, A(0)=1, B(0)=0, A(T)=0, B(T)=1."""
    T = 5.0
    sched = Schedule.linear(T)
    assert sched.s_at(0.0) == pytest.approx(0.0)
    assert sched.s_at(T) == pytest.approx(1.0)
    a0, b0 = sched.coeffs_at(0.0)
    aT, bT = sched.coeffs_at(T)
    assert a0 == pytest.approx(1.0)
    assert b0 == pytest.approx(0.0)
    assert aT == pytest.approx(0.0)
    assert bT == pytest.approx(1.0)


def test_linear_intermediate_monotone() -> None:
    """linear は単調増加の s(t)."""
    T = 4.0
    sched = Schedule.linear(T)
    ts = [i * T / 8 for i in range(9)]
    s_vals = [sched.s_at(t) for t in ts]
    assert all(s2 >= s1 for s1, s2 in zip(s_vals, s_vals[1:], strict=False))
    # 各点で s(t) = t/T
    for t, s in zip(ts, s_vals, strict=True):
        assert s == pytest.approx(t / T)


def test_linear_coeffs_sum_to_one() -> None:
    """A(s) + B(s) = (1-s) + s = 1 — schedule の保存量."""
    T = 3.0
    sched = Schedule.linear(T)
    for t in [0.0, 0.5, 1.0, 1.5, 2.5, T]:
        a, b = sched.coeffs_at(t)
        assert a + b == pytest.approx(1.0)


def test_reverse_v_shape() -> None:
    """``reverse``: s(0)=s_init, s(T/2)=s_target, s(T)=s_init."""
    T = 10.0
    s_init, s_target = 1.0, 0.5
    sched = Schedule.reverse(T, s_init=s_init, s_target=s_target)
    assert sched.s_at(0.0) == pytest.approx(s_init)
    assert sched.s_at(T / 2) == pytest.approx(s_target)
    assert sched.s_at(T) == pytest.approx(s_init)


def test_reverse_half_monotonic() -> None:
    """``reverse``: 前半は s 単調減少, 後半は単調増加."""
    T = 8.0
    sched = Schedule.reverse(T, s_init=1.0, s_target=0.3)
    # 前半
    ts1 = [i * (T / 2) / 4 for i in range(5)]
    s1 = [sched.s_at(t) for t in ts1]
    assert all(b <= a for a, b in zip(s1, s1[1:], strict=False))
    # 後半
    ts2 = [T / 2 + i * (T / 2) / 4 for i in range(5)]
    s2 = [sched.s_at(t) for t in ts2]
    assert all(b >= a for a, b in zip(s2, s2[1:], strict=False))


def test_pause_constant_region() -> None:
    """``pause``: 指定区間で s が一定."""
    T = 10.0
    t_pause = 3.0
    duration = 4.0
    sched = Schedule.pause(T, t_pause=t_pause, duration=duration)
    s_hold = sched.s_at(t_pause)
    # 区間内で s が一定
    for t in [t_pause, t_pause + 1.0, t_pause + duration - 1e-9, t_pause + duration]:
        assert sched.s_at(t) == pytest.approx(s_hold)


def test_pause_boundary_s_values() -> None:
    """``pause``: s(0)=0, s(T)=1 を維持. ramp 中の傾きが
    ``1 / (T - duration)``."""
    T = 6.0
    t_pause = 1.0
    duration = 2.0
    ramp_total = T - duration  # 4.0
    sched = Schedule.pause(T, t_pause=t_pause, duration=duration)
    assert sched.s_at(0.0) == pytest.approx(0.0)
    assert sched.s_at(T) == pytest.approx(1.0)
    # ramp 中の傾き確認 (例: t=0 と t=t_pause で差が t_pause / ramp_total)
    assert sched.s_at(t_pause) == pytest.approx(t_pause / ramp_total)
    # pause 後ランプの再開後 1 単位時刻分
    s_after_1 = sched.s_at(t_pause + duration + 1.0)
    expected = t_pause / ramp_total + 1.0 / ramp_total
    assert s_after_1 == pytest.approx(expected)


def test_pause_invalid_duration() -> None:
    with pytest.raises(ValueError, match="duration must satisfy"):
        Schedule.pause(T=5.0, t_pause=0.0, duration=-1.0)
    with pytest.raises(ValueError, match="duration must satisfy"):
        Schedule.pause(T=5.0, t_pause=0.0, duration=5.0)


def test_pause_invalid_t_pause() -> None:
    with pytest.raises(ValueError, match="t_pause must satisfy"):
        Schedule.pause(T=5.0, t_pause=-1.0, duration=1.0)
    with pytest.raises(ValueError, match="t_pause must satisfy"):
        Schedule.pause(T=5.0, t_pause=4.0, duration=2.0)  # 4 + 2 > 5


def test_from_callable_custom_schedule() -> None:
    """任意の callable で sin 波スケジュールを構築できる."""
    T = math.pi
    # s(t) = sin(t/2), A(s) = s^2, B(s) = 1-s^2
    sched = Schedule.from_callable(
        T=T,
        A=lambda s: s * s,
        B=lambda s: 1.0 - s * s,
        s=lambda t: math.sin(t / 2),
    )
    # t=0: s=0, A=0, B=1
    a, b = sched.coeffs_at(0.0)
    assert a == pytest.approx(0.0)
    assert b == pytest.approx(1.0)
    # t=T=π: s=sin(π/2)=1, A=1, B=0
    a, b = sched.coeffs_at(T)
    assert a == pytest.approx(1.0)
    assert b == pytest.approx(0.0)


def test_default_s_is_linear() -> None:
    """``s=None`` で構築すると ``s(t) = t/T`` (線形)."""
    T = 2.0
    sched = Schedule(T=T, A=lambda s: 1.0 - s, B=lambda s: s)
    for t in [0.0, 0.5, 1.0, 2.0]:
        assert sched.s_at(t) == pytest.approx(t / T)
