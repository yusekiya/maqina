"""adaptive step-size controller の「ノコギリ波らしさ」を数値化する共有 util.

issue #152 (umbrella #148) のテスト/計測基盤。``evolve_schedule_adaptive_*``
の PI controller が臨界領域でノコギリ波 (受理率 ≈ 50% / dt の振動) に陥るかを
**決定論的かつ閾値非依存** に測るためのメトリクス群を提供する。後続 sub-issue
(#149 / #150 / #151) は本 util を再利用して「baseline → 各修正で ``log(dt)`` の
lag-1 自己相関が −1 から改善 / 受理率回復」を assert する。

設計方針 (絶対閾値を避ける理由):
    reject 回数は accept/reject 境界 (``err ≈ tol_step``) で float 最下位差に
    敏感で、絶対値を CI で固定すると別マシンで落ちる。そこで本 util の主役は
    **``log(dt)`` の lag-1 自己相関** (ノコギリ波 ≈ −1、平滑 ≈ 0〜正) のように
    **形状を捉える無次元量** にする。各 sub-issue では同一実行内 old vs new の
    差分 + マージン付き不等式で assert する。

メトリクス一覧:
    - :func:`acceptance_rate` / :func:`n_rejects_in_window` — 受理率 / reject 数
    - :func:`log_dt_lag1_autocorr` — accepted ``log(dt)`` の lag-1 自己相関
    - :func:`n_reversals` — ``log(dt)`` 増減の反転回数
    - :func:`std_log_dt` — 臨界窓内 ``std(log dt)``
    - :func:`summarize` — 上記を 1 dict に集約 (bench からの利用を想定)

いずれも ``window=(t_lo, t_hi)`` を渡すと「臨界窓」内に start time が入る
accepted step / attempt のみで算出する。``t_lo <= t < t_hi`` の半開区間。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StepAttempt:
    """controller の 1 回の step 試行 (accept / reject 両方を記録).

    Attributes
    ----------
    t : float
        step 開始時刻 (reject による再試行では不変).
    dt : float
        試行した dt (reject 時は次でさらに縮小される).
    err : float
        step が返した局所誤差推定 (合成ハーネスでは ``C₄(t)·dt⁵``).
    accepted : bool
        この試行が accept されたか.
    """

    t: float
    dt: float
    err: float
    accepted: bool


@dataclass
class ControllerTrace:
    """adaptive driver 1 回の走行から再構成した controller の軌跡.

    ``evolve_schedule_adaptive_*`` の返り値 (``t_history`` / ``dt_history`` /
    ``n_rejects``) と、合成ハーネスが記録した全 attempt を保持する。メトリクス
    関数はこの trace を受け取って算出する。

    Attributes
    ----------
    t_history : np.ndarray
        shape ``(K,)`` float64。``t0`` を含み各 accept 後の時刻列。
    dt_history : np.ndarray
        shape ``(K-1,)`` float64。各 accept された step の dt。
    n_rejects : int
        累積 reject 回数 (driver 返り値そのまま)。
    attempts : list[StepAttempt]
        全 step 試行 (accept / reject) を呼び出し順に並べたもの。
    """

    t_history: np.ndarray
    dt_history: np.ndarray
    n_rejects: int
    attempts: list[StepAttempt]

    @property
    def n_accepts(self) -> int:
        """accept された step 数 (= ``len(dt_history)``)."""
        return int(self.dt_history.shape[0])

    @property
    def accept_start_times(self) -> np.ndarray:
        """各 accept された step の開始時刻 (``t_history[:-1]``)."""
        return self.t_history[:-1]


def critical_window(
    trace: ControllerTrace, *, pad: float = 0.0
) -> tuple[float, float] | None:
    """reject が起きた attempt の時刻範囲 ``[min_t − pad, max_t + pad]`` を返す.

    ノコギリ波 (過剰縮小ループ) が発火する「臨界窓」を **自己同定** する。
    reject は臨界領域に集中するため、その時刻範囲がそのまま臨界窓になる。窓内
    metrics を絶対時刻のハードコードなしに算出できるので、別マシン / scenario
    変更に対して頑健。reject が 1 つも無ければ ``None``。
    """
    reject_times = [a.t for a in trace.attempts if not a.accepted]
    if not reject_times:
        return None
    return (min(reject_times) - pad, max(reject_times) + pad)


def _window_mask(times: np.ndarray, window: tuple[float, float] | None) -> np.ndarray:
    """``window=(t_lo, t_hi)`` の半開区間 ``[t_lo, t_hi)`` に入る bool マスク."""
    if window is None:
        return np.ones(times.shape[0], dtype=bool)
    t_lo, t_hi = window
    return (times >= t_lo) & (times < t_hi)


def _accepted_log_dt(
    trace: ControllerTrace, window: tuple[float, float] | None
) -> np.ndarray:
    """accept された step の ``log(dt)`` 列 (窓指定時は窓内のみ)."""
    dt = np.asarray(trace.dt_history, dtype=np.float64)
    if dt.shape[0] == 0:
        return np.empty(0, dtype=np.float64)
    mask = _window_mask(np.asarray(trace.accept_start_times, dtype=np.float64), window)
    return np.log(dt[mask])


def log_dt_lag1_autocorr(
    trace: ControllerTrace, window: tuple[float, float] | None = None
) -> float:
    r"""accept された ``log(dt)`` 列の lag-1 自己相関を返す.

    .. code-block:: text

        r1 = Σ_{i} (x_i - μ)(x_{i+1} - μ) / Σ_i (x_i - μ)²

    ここで ``x_i = log(dt_i)``、``μ`` は窓内平均。完全なノコギリ波 (1 step おきに
    拡大↔縮小) では連続偏差が逆符号になり ``r1 → −1``、単調 / 平滑な dt 列では
    ``r1 ≳ 0``。要素数 < 3 または定数列 (分母 0) では ``nan`` を返す。
    """
    x = _accepted_log_dt(trace, window)
    n = x.shape[0]
    if n < 3:
        return float("nan")
    xc = x - x.mean()
    denom = float(np.dot(xc, xc))
    if denom <= 1.0e-300:
        return float("nan")
    num = float(np.dot(xc[:-1], xc[1:]))
    return num / denom


def n_reversals(
    trace: ControllerTrace, window: tuple[float, float] | None = None
) -> int:
    """accept された ``log(dt)`` 列の増減反転回数を返す.

    ``diff(log dt)`` の符号変化数。flat (符号 0) は無視してから数える。
    ノコギリ波では ``n_accepts - 2`` 近くまで増える。要素数 < 3 で ``0``。
    """
    x = _accepted_log_dt(trace, window)
    if x.shape[0] < 3:
        return 0
    s = np.sign(np.diff(x))
    s = s[s != 0.0]
    if s.shape[0] < 2:
        return 0
    return int(np.count_nonzero(s[1:] != s[:-1]))


def std_log_dt(
    trace: ControllerTrace, window: tuple[float, float] | None = None
) -> float:
    """accept された ``log(dt)`` 列の標準偏差を返す (窓指定時は窓内のみ).

    要素が空のとき ``nan``。
    """
    x = _accepted_log_dt(trace, window)
    if x.shape[0] == 0:
        return float("nan")
    return float(np.std(x))


def _attempts_in_window(
    trace: ControllerTrace, window: tuple[float, float] | None
) -> list[StepAttempt]:
    """``window`` 内に start time が入る attempt のみを返す."""
    if window is None:
        return list(trace.attempts)
    t_lo, t_hi = window
    return [a for a in trace.attempts if t_lo <= a.t < t_hi]


def acceptance_rate(
    trace: ControllerTrace, window: tuple[float, float] | None = None
) -> float:
    """受理率 = accept された attempt 数 / 全 attempt 数 (窓指定時は窓内).

    ノコギリ波では理論上 ≈ 0.5 に張り付く。窓内に attempt が無いとき ``nan``。
    """
    atts = _attempts_in_window(trace, window)
    if not atts:
        return float("nan")
    n_acc = sum(1 for a in atts if a.accepted)
    return n_acc / len(atts)


def n_rejects_in_window(
    trace: ControllerTrace, window: tuple[float, float] | None = None
) -> int:
    """``window`` 内で reject された attempt 数 (窓 ``None`` で全 reject 数)."""
    atts = _attempts_in_window(trace, window)
    return sum(1 for a in atts if not a.accepted)


def summarize(
    trace: ControllerTrace, window: tuple[float, float] | None = None
) -> dict[str, float]:
    """全メトリクスを 1 dict に集約する (bench からの利用を想定).

    Returns
    -------
    dict[str, float]
        ``acceptance_rate`` / ``n_rejects`` / ``n_accepts`` /
        ``log_dt_lag1_autocorr`` / ``n_reversals`` / ``std_log_dt`` を含む。
    """
    return {
        "acceptance_rate": acceptance_rate(trace, window),
        "n_rejects": float(n_rejects_in_window(trace, window)),
        "n_accepts": float(len(_accepted_log_dt(trace, window))),
        "log_dt_lag1_autocorr": log_dt_lag1_autocorr(trace, window),
        "n_reversals": float(n_reversals(trace, window)),
        "std_log_dt": std_log_dt(trace, window),
    }
