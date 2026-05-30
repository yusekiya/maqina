"""``tests/_controller_metrics.py`` の振動メトリクスヘルパの単体テスト (issue #152).

既知の人工 dt 列に対して lag-1 自己相関 / 反転回数 / std / 受理率が期待値に
なることを固定する。合成ハーネス (層 A) や bench (層 B) がこれらのメトリクスに
依存する前に、メトリクス自体の正しさをここで保証する。
"""

from __future__ import annotations

import math

import numpy as np

from _controller_metrics import (
    ControllerTrace,
    StepAttempt,
    acceptance_rate,
    log_dt_lag1_autocorr,
    n_reversals,
    n_rejects_in_window,
    std_log_dt,
    summarize,
)


def _trace_from_dt(dt_history: list[float], *, t0: float = 0.0) -> ControllerTrace:
    """accept のみ (reject なし) の人工 trace を dt 列から作る.

    start time は ``dt`` の cumsum で割り当てる (窓選択テスト用)。attempts は
    全 accept として 1 対 1 で生成する。
    """
    dt_arr = np.asarray(dt_history, dtype=np.float64)
    t_hist = np.concatenate([[t0], t0 + np.cumsum(dt_arr)])
    attempts = [
        StepAttempt(t=float(t_hist[i]), dt=float(dt_arr[i]), err=0.0, accepted=True)
        for i in range(dt_arr.shape[0])
    ]
    return ControllerTrace(
        t_history=t_hist,
        dt_history=dt_arr,
        n_rejects=0,
        attempts=attempts,
    )


# ---------------------------------------------------------------------------
# lag-1 autocorrelation
# ---------------------------------------------------------------------------


def test_lag1_autocorr_sawtooth_is_near_minus_one():
    """1 step おきに拡大↔縮小する dt 列は lag-1 自己相関 ≈ −1."""
    dt = [1.0, 0.4] * 10  # 20 点の完全交番
    trace = _trace_from_dt(dt)
    r1 = log_dt_lag1_autocorr(trace)
    # 完全交番なら −(n-1)/n ≈ −0.95。−0.8 より負を要求。
    assert r1 < -0.8


def test_lag1_autocorr_monotone_is_positive():
    """単調幾何減少する dt 列 (log 線形) は lag-1 自己相関 > 0."""
    dt = [0.9**i for i in range(15)]
    trace = _trace_from_dt(dt)
    r1 = log_dt_lag1_autocorr(trace)
    assert r1 > 0.5


def test_lag1_autocorr_constant_is_nan():
    """定数 dt 列は分母 0 で nan."""
    trace = _trace_from_dt([0.3] * 10)
    assert math.isnan(log_dt_lag1_autocorr(trace))


def test_lag1_autocorr_too_short_is_nan():
    """要素数 < 3 では nan."""
    trace = _trace_from_dt([0.5, 0.25])
    assert math.isnan(log_dt_lag1_autocorr(trace))


# ---------------------------------------------------------------------------
# n_reversals
# ---------------------------------------------------------------------------


def test_n_reversals_sawtooth():
    """交番列は (n-2) 回反転する."""
    dt = [1.0, 0.4] * 5  # n = 10
    trace = _trace_from_dt(dt)
    assert n_reversals(trace) == 8


def test_n_reversals_monotone_is_zero():
    """単調列は反転 0."""
    dt = [0.9**i for i in range(10)]
    trace = _trace_from_dt(dt)
    assert n_reversals(trace) == 0


def test_n_reversals_constant_is_zero():
    """定数列は (符号 0 を除外して) 反転 0."""
    trace = _trace_from_dt([0.3] * 10)
    assert n_reversals(trace) == 0


# ---------------------------------------------------------------------------
# std_log_dt
# ---------------------------------------------------------------------------


def test_std_log_dt_constant_is_zero():
    trace = _trace_from_dt([0.3] * 8)
    assert std_log_dt(trace) == 0.0


def test_std_log_dt_positive_for_varying():
    trace = _trace_from_dt([1.0, 0.4] * 5)
    assert std_log_dt(trace) > 0.0


# ---------------------------------------------------------------------------
# windowing
# ---------------------------------------------------------------------------


def test_window_selects_subrange():
    """窓指定で start time が窓内の accepted step のみに絞られる.

    dt = 1.0 を 10 step (start time 0,1,...,9)。窓 [3, 6) は start time
    3,4,5 の 3 step を拾うので std=0 (全部同じ dt) / autocorr=nan (定数)。
    一方 dt を交番にすれば窓内でも振動が見える。
    """
    dt = [1.0, 0.4] * 10
    trace = _trace_from_dt(dt)
    full = std_log_dt(trace)
    # 全区間の終端側 1/4 窓: start time は cumsum(dt) ベース。広めの窓で
    # 要素が減ることだけ確認する。
    t_end = float(trace.t_history[-1])
    windowed_n = summarize(trace, window=(0.5 * t_end, t_end))["n_accepts"]
    assert windowed_n < trace.n_accepts
    assert full > 0.0


def test_window_empty_gives_nan_and_zero():
    trace = _trace_from_dt([1.0, 0.4] * 5)
    # start time がどの step とも一致しない窓 (負域)
    window = (-100.0, -50.0)
    assert math.isnan(log_dt_lag1_autocorr(trace, window))
    assert math.isnan(std_log_dt(trace, window))
    assert n_reversals(trace, window) == 0
    assert math.isnan(acceptance_rate(trace, window))
    assert n_rejects_in_window(trace, window) == 0


# ---------------------------------------------------------------------------
# acceptance_rate / n_rejects_in_window
# ---------------------------------------------------------------------------


def test_acceptance_rate_with_rejects():
    """accept 2 / reject 2 を混在させると受理率 0.5."""
    attempts = [
        StepAttempt(t=0.0, dt=1.0, err=0.0, accepted=False),
        StepAttempt(t=0.0, dt=0.5, err=0.0, accepted=True),
        StepAttempt(t=0.5, dt=0.5, err=0.0, accepted=False),
        StepAttempt(t=0.5, dt=0.25, err=0.0, accepted=True),
    ]
    trace = ControllerTrace(
        t_history=np.array([0.0, 0.5, 0.75]),
        dt_history=np.array([0.5, 0.25]),
        n_rejects=2,
        attempts=attempts,
    )
    assert acceptance_rate(trace) == 0.5
    assert n_rejects_in_window(trace) == 2
    # 窓 [0.4, 1.0) は t=0.5 の 2 attempt のみ → 受理率 0.5, reject 1
    assert acceptance_rate(trace, window=(0.4, 1.0)) == 0.5
    assert n_rejects_in_window(trace, window=(0.4, 1.0)) == 1


def test_summarize_keys():
    trace = _trace_from_dt([1.0, 0.4] * 5)
    s = summarize(trace)
    assert set(s) == {
        "acceptance_rate",
        "n_rejects",
        "n_accepts",
        "log_dt_lag1_autocorr",
        "n_reversals",
        "std_log_dt",
    }
