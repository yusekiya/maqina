"""issue #151 — 真の PI 比例項 (I 制御 → PI 制御化).

``_pi_dt_next`` の accept 時 dt 予測式に Gustafsson / Hairer-Wanner II §IV.2 の
predictive PI 比例項 ``(err_prev / err)^{pi_beta/(p+1)}`` を加える。比例項は誤差の
増加傾向 (Magnus 4 次係数 C₄ の上昇) を ``err_prev / err`` で先読みして dt 拡大を
抑制し、臨界領域でのノコギリ波を「再オーバーシュート前に」平坦化する。

カバーする軸:

1. **回帰アンカー (関数レベル / ビット一致)**: ``_pi_dt_next`` を
   ``pi_alpha=1.0, pi_beta=0.0`` で呼ぶと、``err_prev`` に何を渡しても従来の純 I
   制御式 ``dt · safety · (tol_step / err)^{1/(p+1)}`` (+ クランプ) に **ビット一致**
   する (比例項が恒等的に 1.0 に縮退し、``err_prev`` が無視される)。
2. **層 A (合成誤差ハーネス / 比例項の単独効果)**: I 制御がノコギリ波に陥る
   regime (over-shrink + freeze 無効) で ``freeze`` を固定したまま ``pi_beta`` のみ
   ``0 → 0.4`` に上げ、同一 ``C₄(t)`` で受理率↑ / ``n_rejects`` 非劣化 /
   ``log(dt)`` の lag-1 自己相関が −1 から改善 / reversals 非劣化 を assert。
3. **3 手段合算**: main baseline (#149 前 = 固定 0.5 縮小 + freeze 無効 + 純 I) vs
   全適用 (#149 + #150 + #151 既定) を同一 ``C₄(t)`` で比較し、ノコギリ波消滅
   (受理率回復 + autocorr が −1 から大きく改善) を 1 件で固定する。
4. **平滑 schedule の非回帰**: 非臨界 (定数 C₄) schedule で ``n_rejects`` が
   増えず step 数が病的に増えない (Gustafsson の小さい指数による穏やかな保守化を
   許容範囲内に収める)。
5. **入力検証 + ControllerConfig 既定 + facade 配線**: ``pi_alpha > 0`` /
   ``pi_beta >= 0`` の検証、既定 ``(0.7, 0.4)``、``controller=ControllerConfig(...)``
   が facade から driver まで ``pi_alpha`` / ``pi_beta`` を伝播すること。

注 (比例項の単独効果を freeze=False で測る理由):
    比例項は「reject → 過剰縮小 → 楽々 accept → I 制御が大きく再拡大」の **再拡大
    側を、reject を経ずに先読みで** 抑える機構。#150 の成長凍結を有効にすると
    over-shrink ノコギリ波が freeze 側で先に潰れてしまい比例項の寄与が見えにくい
    ので、比例項 **単独** の効果を測る層 A シナリオでは ``freeze=False`` に固定し、
    ``pi_beta`` のみを振る。実運用 (既定) では freeze と比例項が二重に働く。
"""

from __future__ import annotations

import numpy as np
import pytest

from _controller_harness import exp_c4, run_synthetic
from _controller_metrics import (
    acceptance_rate,
    critical_window,
    log_dt_lag1_autocorr,
    n_reversals,
)

import maqina.annealer as _annealer_mod
import maqina.simulator as _simulator_mod
from maqina import (
    AnnealingSimulator,
    ControllerConfig,
    IsingProblem,
    QuantumAnnealer,
    Schedule,
)
from maqina.initial_states import uniform_superposition
from maqina.krylov import _pi_dt_next

# ---------------------------------------------------------------------------
# シナリオ: I 制御がノコギリ波に陥る over-shrink regime (freeze 無効で固定)
# ---------------------------------------------------------------------------

_T_STAR = 5.0
_SCENARIO: dict[str, dict] = {
    "richardson": dict(k=40.0, t1=5.5, dt_min=1e-3, max_rejects=500),
    "chebyshev": dict(k=40.0, t1=5.5, dt_min=1e-3, max_rejects=500),
    "m2": dict(k=250.0, t1=5.05, dt_min=5e-4, max_rejects=2000),
}
_ALL_METHODS = ["m2", "richardson", "chebyshev"]
# 過剰縮小 regime (I 制御のノコギリ波を発火させる; モジュール docstring 注参照).
_OVERSHRINK = dict(reject_shrink_min=0.5, reject_shrink_max=0.5)
# 比例項以外を I 制御 baseline に固定する共通 kwarg (freeze 無効 + 純 I).
_BASE = dict(freeze_growth_after_reject=False)
# tuned PI 既定 (Gustafsson predictive PI controller 標準).
_PI = dict(pi_alpha=0.7, pi_beta=0.4)
# 純 I 制御 (回帰アンカー).
_I = dict(pi_alpha=1.0, pi_beta=0.0)


def _common_cfg(method: str) -> dict:
    cfg = dict(t0=0.0, tol_step=1e-8, dt0=0.1, dt_max=0.5)
    s = dict(_SCENARIO[method])
    s.pop("k")
    cfg.update(s)
    return cfg


def _pi_off_on(method: str):
    """同一 ``C₄`` (over-shrink, freeze 無効) で比例項 off / on の trace を返す."""
    cfg = _common_cfg(method)
    c4 = exp_c4(k=_SCENARIO[method]["k"], t_star=_T_STAR)
    off = run_synthetic(method, c4, **_BASE, **_OVERSHRINK, **_I, **cfg)
    on = run_synthetic(method, c4, **_BASE, **_OVERSHRINK, **_PI, **cfg)
    return off, on


# ---------------------------------------------------------------------------
# 1. 回帰アンカー (関数レベル / ビット一致)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("p", [2, 4])
@pytest.mark.parametrize("err", [1e-6, 1e-8, 1e-10, 1e-12])
@pytest.mark.parametrize("err_prev", [None, 1e-9, 1e-6, 1e-3, 0.0, 1e-40])
def test_pi_dt_next_reduces_to_i_control_bitexact(p, err, err_prev):
    """``pi_alpha=1.0, pi_beta=0.0`` で純 I 制御式にビット一致 (err_prev 無視).

    比例項が恒等的に ``1.0`` に縮退し、``err_prev`` の値に依らず従来の I 制御式
    ``dt · safety · (tol_step / err)^{1/(p+1)}`` + クランプを完全再現することを、
    ``==`` (浮動小数ビット一致) で確認する。
    """
    dt_try = 0.1
    tol_step = 1e-8
    safety = 0.9
    growth_max = 4.0
    dt_max = 1.0
    dt_min = 1e-4

    # 従来 (pre-#151) の純 I 制御式を明示的に再計算した参照値.
    if err <= 1.0e-30:
        ref = dt_try * growth_max
    else:
        ref = dt_try * safety * (tol_step / err) ** (1.0 / (p + 1))
    ref = min(ref, dt_try * growth_max, dt_max)
    ref = max(ref, dt_min)

    got = _pi_dt_next(
        dt_try,
        err,
        tol_step=tol_step,
        safety=safety,
        growth_max=growth_max,
        dt_max=dt_max,
        dt_min=dt_min,
        p=p,
        err_prev=err_prev,
        pi_alpha=1.0,
        pi_beta=0.0,
    )
    assert got == ref, f"not bit-exact: got={got!r} ref={ref!r}"


def test_pi_dt_next_proportional_term_shrinks_growth_on_rising_error():
    """``err > err_prev`` (誤差増加) のとき比例項が dt 拡大を I 制御より抑える.

    ``err_prev / err < 1`` なので ``p_term < 1`` となり、PI の ``dt_next`` が純 I の
    ``dt_next`` を下回る (先読み抑制)。0 近傍ガードを避けた通常 regime で確認。
    """
    kw = dict(
        tol_step=1e-8,
        safety=0.9,
        growth_max=4.0,
        dt_max=1.0,
        dt_min=1e-12,
        p=4,
    )
    # err は tol より小さい (= accept で拡大したい) が err_prev はさらに小さい
    # → 誤差が増加局面 → 比例項が拡大を抑える.
    dt_i = _pi_dt_next(0.1, 1e-9, err_prev=1e-11, pi_alpha=1.0, pi_beta=0.0, **kw)
    dt_pi = _pi_dt_next(0.1, 1e-9, err_prev=1e-11, pi_alpha=0.7, pi_beta=0.4, **kw)
    assert dt_pi < dt_i, (
        f"proportional term did not suppress growth: pi={dt_pi} i={dt_i}"
    )


# ---------------------------------------------------------------------------
# 2. 層 A 主役: 比例項単独で over-shrink ノコギリ波が緩和される
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", _ALL_METHODS)
def test_proportional_term_reduces_sawtooth(method):
    """``pi_beta>0`` で I 制御の over-shrink ノコギリ波が緩和される (freeze 固定).

    freeze を無効に固定し比例項のみを ``0 → 0.4`` に上げる。比例項が誤差増加を
    先読みして reject 直前の過大な dt 拡大を抑えるため、reject が減り受理率が
    回復し、``log(dt)`` の lag-1 自己相関が −1 から改善する。
    """
    off, on = _pi_off_on(method)

    # baseline (純 I) は病的: reject 多発 + 強い負の自己相関.
    assert off.n_rejects >= 3, f"{method}: baseline should be pathological"
    win = critical_window(off)
    assert win is not None
    r1_off = log_dt_lag1_autocorr(off, win)
    assert r1_off < -0.4, f"{method}: baseline sawtooth not captured (r1={r1_off})"

    # 非劣化: reject 数は増えない.
    assert on.n_rejects <= off.n_rejects, (
        f"{method}: rejects regressed, off={off.n_rejects} on={on.n_rejects}"
    )
    # 受理率は非劣化 (全区間).
    assert acceptance_rate(on, None) >= acceptance_rate(off, None) - 1e-12, (
        f"{method}: acceptance regressed"
    )
    # 自己相関が改善 (−1 から離れる). 窓内 accept が < 3 で nan になるのは
    # ノコギリ波消滅の最良ケースなので許容.
    r1_on = log_dt_lag1_autocorr(on, win)
    assert np.isnan(r1_on) or r1_on > r1_off, (
        f"{method}: autocorr did not improve, off={r1_off} on={r1_on}"
    )
    # reversals (dt 増減反転回数) が非劣化.
    assert n_reversals(on, win) <= n_reversals(off, win), (
        f"{method}: reversals regressed"
    )


# ---------------------------------------------------------------------------
# 3. 3 手段合算: main baseline vs 全適用でノコギリ波消滅
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", _ALL_METHODS)
def test_all_fixes_combined_kill_sawtooth(method):
    """#149 前 main baseline vs #149+#150+#151 全適用でノコギリ波が消える.

    main baseline = 固定 0.5 縮小 + 成長凍結なし + 純 I 制御。全適用 = 既定
    (予測式クランプ ``[0.2, 0.9]`` + 成長凍結 + PI 比例項 ``0.7/0.4``)。
    同一 ``C₄(t)`` で受理率回復 + autocorr が −1 から大きく改善することを 1 件で固定。
    """
    cfg = _common_cfg(method)
    c4 = exp_c4(k=_SCENARIO[method]["k"], t_star=_T_STAR)
    base = run_synthetic(
        method,
        c4,
        reject_shrink_min=0.5,
        reject_shrink_max=0.5,
        freeze_growth_after_reject=False,
        pi_alpha=1.0,
        pi_beta=0.0,
        **cfg,
    )
    allf = run_synthetic(method, c4, **_PI, **cfg)  # 残りは production 既定

    win = critical_window(base)
    assert win is not None
    r1_base = log_dt_lag1_autocorr(base, win)
    assert r1_base < -0.4, f"{method}: baseline sawtooth not captured (r1={r1_base})"

    # reject 激減 + 受理率回復.
    assert allf.n_rejects <= base.n_rejects, (
        f"{method}: rejects regressed, base={base.n_rejects} all={allf.n_rejects}"
    )
    assert acceptance_rate(allf, None) >= acceptance_rate(base, None), (
        f"{method}: acceptance regressed"
    )
    # 自己相関が大幅改善 (窓内 accept < 3 で nan = ノコギリ波消滅の最良ケース).
    r1_all = log_dt_lag1_autocorr(allf, win)
    assert np.isnan(r1_all) or r1_all > r1_base + 0.3, (
        f"{method}: autocorr not substantially improved, base={r1_base} all={r1_all}"
    )


# ---------------------------------------------------------------------------
# 4. 平滑 schedule の非回帰
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", _ALL_METHODS)
def test_smooth_schedule_no_regression(method):
    """非臨界 (定数 C₄) schedule で PI が病的な step 増 / reject 増を起こさない.

    比例項は誤差が定常の領域では ``err_prev / err ≈ 1`` でほぼ no-op。Gustafsson
    の小さい指数 (``pi_alpha=0.7``) による穏やかな保守化で step 数は微増しうるが、
    reject は増えず step 増は許容範囲 (< 1.15×) に収まる。
    """
    c4 = lambda t: 1.0  # noqa: E731 (定数 C₄ = 平滑)
    cfg = dict(t0=0.0, t1=20.0, tol_step=1e-8, dt0=0.1, dt_min=1e-4, dt_max=1.0)
    i_ctrl = run_synthetic(method, c4, **_I, **cfg)
    pi_ctrl = run_synthetic(method, c4, **_PI, **cfg)

    assert pi_ctrl.n_rejects <= i_ctrl.n_rejects + 1, (
        f"{method}: rejects regressed, I={i_ctrl.n_rejects} PI={pi_ctrl.n_rejects}"
    )
    assert pi_ctrl.n_accepts <= int(i_ctrl.n_accepts * 1.15), (
        f"{method}: step count blew up, I={i_ctrl.n_accepts} PI={pi_ctrl.n_accepts}"
    )


# ---------------------------------------------------------------------------
# 5a. ControllerConfig 既定 + 入力検証
# ---------------------------------------------------------------------------


def test_controller_config_pi_defaults():
    """default 構築は Gustafsson 標準 (0.7 / 0.4) と一致する."""
    c = ControllerConfig()
    assert c.pi_alpha == 0.7
    assert c.pi_beta == 0.4


def test_controller_config_i_control_anchor_is_valid():
    """``pi_alpha=1.0, pi_beta=0.0`` (純 I 制御) は構築できる."""
    c = ControllerConfig(pi_alpha=1.0, pi_beta=0.0)
    assert c.pi_alpha == 1.0
    assert c.pi_beta == 0.0


@pytest.mark.parametrize("pi_alpha", [0.0, -0.1, -1.0])
def test_controller_config_pi_alpha_must_be_positive(pi_alpha):
    """``pi_alpha <= 0`` は ``ValueError``."""
    with pytest.raises(ValueError):
        ControllerConfig(pi_alpha=pi_alpha)


@pytest.mark.parametrize("pi_beta", [-1e-9, -0.1, -1.0])
def test_controller_config_pi_beta_must_be_nonnegative(pi_beta):
    """``pi_beta < 0`` は ``ValueError``."""
    with pytest.raises(ValueError):
        ControllerConfig(pi_beta=pi_beta)


@pytest.mark.parametrize("method", _ALL_METHODS)
def test_driver_rejects_invalid_pi_params(method):
    """driver 直叩きでも ``pi_alpha <= 0`` / ``pi_beta < 0`` を ``ValueError``."""
    c4 = exp_c4(k=1.0, t_star=10.0)
    cfg = dict(t0=0.0, t1=1.0, tol_step=1e-8, dt0=0.1, dt_min=1e-4, dt_max=1.0)
    with pytest.raises(ValueError):
        run_synthetic(method, c4, pi_alpha=0.0, **cfg)
    with pytest.raises(ValueError):
        run_synthetic(method, c4, pi_beta=-0.1, **cfg)


# ---------------------------------------------------------------------------
# 5b. facade 配線: pi_alpha / pi_beta が driver まで伝播する
# ---------------------------------------------------------------------------

_CUSTOM = ControllerConfig(pi_alpha=0.85, pi_beta=0.25)


def _make_capture(store: dict):
    """driver 差し替え用 fake: kwargs を記録し有効な 10-tuple を返す."""

    def fake(*, psi0, t0, t1, **kwargs):
        store.update(kwargs)
        return (
            np.asarray(psi0, dtype=np.complex128),
            np.array([t0, t1], dtype=np.float64),
            np.array([t1 - t0], dtype=np.float64),
            0,
            np.array([10], dtype=np.int64),
            np.array([0.0], dtype=np.float64),
            np.array([0.0], dtype=np.float64),
            np.array([0.0], dtype=np.float64),
            0,
            None,
        )

    return fake


@pytest.mark.parametrize(
    "method,driver_attr",
    [
        ("cfm4_adaptive_richardson_krylov", "evolve_schedule_adaptive_richardson"),
        (
            "cfm4_adaptive_richardson_chebyshev",
            "evolve_schedule_adaptive_richardson_chebyshev",
        ),
    ],
)
def test_pi_fields_propagate_through_run(method, driver_attr, monkeypatch):
    """``QuantumAnnealer.run(controller=...)`` が driver に pi_alpha/pi_beta を渡す."""
    n = 2
    prob = IsingProblem(n=n, H_p_diag=np.linspace(-1.0, 1.0, 1 << n))
    sched = Schedule.linear(T=1.0, h_x=np.ones(n))
    psi0 = uniform_superposition(n)

    captured: dict = {}
    monkeypatch.setattr(_annealer_mod, driver_attr, _make_capture(captured))

    ann = QuantumAnnealer(prob, sched)
    ann.run(psi0, 0.0, 1.0, method=method, controller=_CUSTOM)
    assert captured["pi_alpha"] == _CUSTOM.pi_alpha
    assert captured["pi_beta"] == _CUSTOM.pi_beta


@pytest.mark.parametrize(
    "method,driver_attr",
    [
        ("cfm4_adaptive_richardson_krylov", "evolve_schedule_adaptive_richardson"),
        (
            "cfm4_adaptive_richardson_chebyshev",
            "evolve_schedule_adaptive_richardson_chebyshev",
        ),
    ],
)
def test_pi_fields_propagate_through_simulator(method, driver_attr, monkeypatch):
    """``AnnealingSimulator(controller=...)`` が driver に pi_alpha/pi_beta を渡す."""
    n = 2
    prob = IsingProblem(n=n, H_p_diag=np.linspace(-1.0, 1.0, 1 << n))
    sched = Schedule.linear(T=1.0, h_x=np.ones(n))
    psi0 = uniform_superposition(n)

    captured: dict = {}
    monkeypatch.setattr(_simulator_mod, driver_attr, _make_capture(captured))

    sim = AnnealingSimulator(prob, sched, psi0, 0.0, method=method, controller=_CUSTOM)
    sim.step(0.5)
    assert captured["pi_alpha"] == _CUSTOM.pi_alpha
    assert captured["pi_beta"] == _CUSTOM.pi_beta


def test_pi_fields_default_when_none(monkeypatch):
    """``controller=None`` (既定) で pi 既定 (0.7 / 0.4) が driver に渡る."""
    n = 2
    prob = IsingProblem(n=n, H_p_diag=np.linspace(-1.0, 1.0, 1 << n))
    sched = Schedule.linear(T=1.0, h_x=np.ones(n))
    psi0 = uniform_superposition(n)

    captured: dict = {}
    monkeypatch.setattr(
        _annealer_mod, "evolve_schedule_adaptive_richardson", _make_capture(captured)
    )
    ann = QuantumAnnealer(prob, sched)
    ann.run(psi0, 0.0, 1.0, method="cfm4_adaptive_richardson_krylov")
    assert captured["pi_alpha"] == 0.7
    assert captured["pi_beta"] == 0.4
