"""issue #149 — reject 時の dt 縮小を固定 0.5 から予測式 + クランプに変更.

3 軸でカバーする:

1. **層 A (主役 / 合成誤差ハーネス)**: #152 の monkeypatch ハーネスで ``C₄(t)``
   をノコギリ波発火域に設定し、旧挙動 (``reject_shrink_min=max=0.5``) vs 新
   クランプ既定 (``[0.2, 0.9]``) を同一 ``C₄(t)`` で比較。新既定で受理率↑ /
   ``n_rejects``↓ / ``log(dt)`` の lag-1 自己相関が −1 から改善することを assert
   (決定論的・プラットフォーム非依存)。
2. **回帰アンカー**: ``_pi_dt_reject`` を ``reject_shrink_min=max=0.5`` で呼ぶと
   旧式 ``max(dt_try · 0.5, dt_min)`` とビット一致する (= 現行 main の挙動)。
3. **入力検証 + facade 配線**: ``ControllerConfig.__post_init__`` の範囲検証と、
   ``controller=ControllerConfig(...)`` が facade (``QuantumAnnealer.run`` /
   ``AnnealingSimulator``) から driver まで全 6 field 伝播すること。

層 B (end-to-end) は ``benchmarks/bench_stepsize_controller.py`` + Linux サーバー
での pre-merge bench に委ねる (本ファイルは決定論的な単体/合成のみ)。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from _controller_harness import exp_c4, run_synthetic
from _controller_metrics import (
    acceptance_rate,
    critical_window,
    log_dt_lag1_autocorr,
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
from maqina.krylov import _pi_dt_reject

# ---------------------------------------------------------------------------
# 層 A: 同一 C₄ での旧 (0.5/0.5) vs 新 (default [0.2, 0.9]) 比較
# ---------------------------------------------------------------------------

# test_controller_sawtooth.py と同じノコギリ波発火シナリオ (ただし reject
# クランプは比較対象なので外に出す)。``k`` は推定子 order 別の閾値。
_T_STAR = 5.0
_SCENARIO: dict[str, dict] = {
    "richardson": dict(
        k=40.0,
        t0=0.0,
        t1=5.5,
        tol_step=1e-8,
        dt0=0.1,
        dt_min=1e-3,
        dt_max=0.5,
        max_rejects=500,
    ),
    "chebyshev": dict(
        k=40.0,
        t0=0.0,
        t1=5.5,
        tol_step=1e-8,
        dt0=0.1,
        dt_min=1e-3,
        dt_max=0.5,
        max_rejects=500,
    ),
    "m2": dict(
        k=250.0,
        t0=0.0,
        t1=5.05,
        tol_step=1e-8,
        dt0=0.1,
        dt_min=5e-4,
        dt_max=0.5,
        max_rejects=2000,
    ),
}
_ALL_METHODS = ["m2", "richardson", "chebyshev"]


def _old_new(method: str):
    """同一 ``C₄`` で旧 (固定 0.5) と新 (default クランプ) の trace を返す."""
    cfg = dict(_SCENARIO[method])
    k = cfg.pop("k")
    c4 = exp_c4(k=k, t_star=_T_STAR)
    old = run_synthetic(method, c4, reject_shrink_min=0.5, reject_shrink_max=0.5, **cfg)
    new = run_synthetic(method, c4, **cfg)  # default [0.2, 0.9]
    return old, new


@pytest.mark.parametrize("method", _ALL_METHODS)
def test_reject_clamp_reduces_sawtooth(method):
    """新クランプ既定でノコギリ波が緩和される (受理率↑ / reject↓ / 自己相関改善).

    旧挙動 (固定 0.5 半減) は臨界窓で強い負の lag-1 自己相関 (dt 振動) を出す。
    新既定 (予測式 + ``[0.2, 0.9]`` クランプ) では ``err`` が tol をわずかに
    超えただけのとき ``factor ≈ 0.9`` で済むため過剰縮小が消え、受理率が回復し
    reject が減り、自己相関が −1 から改善する。
    """
    old, new = _old_new(method)

    # baseline (旧挙動) は病的: reject 多発.
    assert old.n_rejects >= 3, f"{method}: baseline should be pathological"

    # 改善 1: reject 数が厳密に減る.
    assert new.n_rejects < old.n_rejects, (
        f"{method}: expected fewer rejects, old={old.n_rejects} new={new.n_rejects}"
    )

    # 改善 2: 全区間の受理率が上がる.
    acc_old = acceptance_rate(old, None)
    acc_new = acceptance_rate(new, None)
    assert acc_new > acc_old, (
        f"{method}: expected higher acceptance, old={acc_old} new={acc_new}"
    )

    # 改善 3: 旧の臨界窓で lag-1 自己相関が −1 から改善.
    win = critical_window(old)
    assert win is not None
    r1_old = log_dt_lag1_autocorr(old, win)
    assert r1_old < -0.4, f"{method}: baseline sawtooth not captured (r1={r1_old})"
    r1_new = log_dt_lag1_autocorr(new, win)
    # 新挙動は同じ窓で振動が消える: 自己相関が改善 (>) するか、振動が無く
    # サンプル不足/定数列で nan (= ノコギリ波消滅) のいずれか.
    assert math.isnan(r1_new) or r1_new > r1_old, (
        f"{method}: autocorr did not improve, old={r1_old} new={r1_new}"
    )


# ---------------------------------------------------------------------------
# 回帰アンカー: reject_shrink_min=max=0.5 が旧式 max(dt·0.5, dt_min) とビット一致
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("p", [2, 4])
def test_legacy_reject_shrink_reproduces_fixed_half(p):
    """``reject_shrink_min=max=0.5`` で ``_pi_dt_reject`` が旧式とビット一致.

    旧 driver は reject 時 ``dt = max(dt_try · 0.5, dt_min)`` 固定半減だった。
    クランプ範囲を ``[0.5, 0.5]`` に潰すと予測式の値に関わらず ``factor = 0.5``
    に固定されるため、err / dt_try によらず旧式と完全一致する (= 現行 main の
    挙動の回帰アンカー)。
    """
    tol_step = 1e-8
    dt_min = 1e-4
    rng = np.random.default_rng(149)
    for _ in range(200):
        dt_try = float(10.0 ** rng.uniform(-4.0, 0.0))
        err = float(10.0 ** rng.uniform(-12.0, 2.0))
        got = _pi_dt_reject(
            dt_try,
            err,
            tol_step=tol_step,
            safety=0.9,
            reject_shrink_min=0.5,
            reject_shrink_max=0.5,
            dt_min=dt_min,
            p=p,
        )
        want = max(dt_try * 0.5, dt_min)
        assert got == want, f"p={p} dt_try={dt_try} err={err}: {got} != {want}"


def test_reject_clamp_barely_over_tol_stays_near_max():
    """``err`` が tol をわずかに超えただけなら factor ≈ reject_shrink_max.

    過剰縮小 (固定 0.5) の解消を直接確認する: ``err = 2·tol`` (order-5 で
    factor = 0.9·(1/2)^{1/5} ≈ 0.783) は ``[0.2, 0.9]`` クランプ内なので
    dt は半減せず 0.78 倍程度に留まる (旧固定 0.5 より大きい dt を維持).
    """
    tol_step = 1e-8
    dt_try = 0.1
    dt = _pi_dt_reject(
        dt_try,
        2.0 * tol_step,
        tol_step=tol_step,
        safety=0.9,
        reject_shrink_min=0.2,
        reject_shrink_max=0.9,
        dt_min=1e-6,
        p=4,
    )
    factor = dt / dt_try
    assert 0.5 < factor <= 0.9, f"expected mild shrink near max, got factor={factor}"


def test_reject_clamp_far_over_tol_hits_min():
    """``err`` が tol を大きく超えれば factor は reject_shrink_min まで落ちる."""
    tol_step = 1e-8
    dt_try = 0.1
    dt = _pi_dt_reject(
        dt_try,
        1e6 * tol_step,
        tol_step=tol_step,
        safety=0.9,
        reject_shrink_min=0.2,
        reject_shrink_max=0.9,
        dt_min=1e-6,
        p=4,
    )
    assert dt == pytest.approx(dt_try * 0.2), f"expected floor at min, got {dt}"


# ---------------------------------------------------------------------------
# ControllerConfig 入力検証
# ---------------------------------------------------------------------------


def test_controller_config_defaults():
    """default 構築は driver 既定値と一致する."""
    c = ControllerConfig()
    assert c.safety == 0.9
    assert c.growth_max == 4.0
    assert c.max_rejects == 50
    assert c.dt_min == 1e-4
    assert c.reject_shrink_min == 0.2
    assert c.reject_shrink_max == 0.9


def test_controller_config_is_frozen():
    """frozen dataclass: 属性代入は禁止."""
    c = ControllerConfig()
    with pytest.raises(Exception):
        c.safety = 0.5  # type: ignore[misc]


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(safety=0.0),
        dict(safety=-0.1),
        dict(growth_max=1.0),
        dict(growth_max=0.5),
        dict(max_rejects=0),
        dict(max_rejects=-1),
        dict(dt_min=0.0),
        dict(dt_min=-1e-4),
        dict(reject_shrink_min=0.0),
        dict(reject_shrink_max=1.0),
        dict(reject_shrink_max=1.5),
        dict(reject_shrink_min=0.9, reject_shrink_max=0.2),  # min > max
    ],
)
def test_controller_config_validation_raises(kwargs):
    """範囲外フィールドは ``ValueError``."""
    with pytest.raises(ValueError):
        ControllerConfig(**kwargs)


def test_controller_config_max_rejects_must_be_int():
    """``max_rejects`` に float を渡すと ``ValueError`` (int 厳格)."""
    with pytest.raises(ValueError):
        ControllerConfig(max_rejects=50.0)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# facade 配線: ControllerConfig が driver まで伝播する
# ---------------------------------------------------------------------------

_CUSTOM = ControllerConfig(
    safety=0.7,
    growth_max=2.0,
    max_rejects=7,
    dt_min=2e-3,
    reject_shrink_min=0.3,
    reject_shrink_max=0.8,
)


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


def _assert_controller_kwargs(captured: dict, ctrl: ControllerConfig):
    assert captured["safety"] == ctrl.safety
    assert captured["growth_max"] == ctrl.growth_max
    assert captured["max_rejects"] == ctrl.max_rejects
    assert captured["dt_min"] == ctrl.dt_min
    assert captured["reject_shrink_min"] == ctrl.reject_shrink_min
    assert captured["reject_shrink_max"] == ctrl.reject_shrink_max


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
def test_controller_propagates_through_run(method, driver_attr, monkeypatch):
    """``QuantumAnnealer.run(controller=...)`` が driver に全 6 field を渡す."""
    n = 2
    prob = IsingProblem(n=n, H_p_diag=np.linspace(-1.0, 1.0, 1 << n))
    sched = Schedule.linear(T=1.0, h_x=np.ones(n))
    psi0 = uniform_superposition(n)

    captured: dict = {}
    monkeypatch.setattr(_annealer_mod, driver_attr, _make_capture(captured))

    ann = QuantumAnnealer(prob, sched)
    ann.run(psi0, 0.0, 1.0, method=method, controller=_CUSTOM)
    _assert_controller_kwargs(captured, _CUSTOM)


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
def test_controller_propagates_through_simulator(method, driver_attr, monkeypatch):
    """``AnnealingSimulator(controller=...)`` が driver に全 6 field を渡す."""
    n = 2
    prob = IsingProblem(n=n, H_p_diag=np.linspace(-1.0, 1.0, 1 << n))
    sched = Schedule.linear(T=1.0, h_x=np.ones(n))
    psi0 = uniform_superposition(n)

    captured: dict = {}
    monkeypatch.setattr(_simulator_mod, driver_attr, _make_capture(captured))

    sim = AnnealingSimulator(prob, sched, psi0, 0.0, method=method, controller=_CUSTOM)
    sim.step(0.5)
    _assert_controller_kwargs(captured, _CUSTOM)


def test_controller_default_when_none(monkeypatch):
    """``controller=None`` (既定) で全 default が driver に渡る."""
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
    _assert_controller_kwargs(captured, ControllerConfig())


# ---------------------------------------------------------------------------
# fixed-dt method への controller: simulator は strict / run は寛容
# ---------------------------------------------------------------------------


def test_controller_rejected_for_fixed_dt_simulator():
    """``AnnealingSimulator(method='m2', controller=...)`` は ``ValueError``."""
    n = 2
    prob = IsingProblem(n=n, H_p_diag=np.linspace(-1.0, 1.0, 1 << n))
    sched = Schedule.linear(T=1.0, h_x=np.ones(n))
    psi0 = uniform_superposition(n)
    with pytest.raises(ValueError, match="controller"):
        AnnealingSimulator(
            prob, sched, psi0, 0.0, method="m2", controller=ControllerConfig()
        )


def test_controller_rejected_for_fixed_dt_create_simulator():
    """``create_simulator(method='m2', controller=...)`` 経由でも ``ValueError``."""
    n = 2
    prob = IsingProblem(n=n, H_p_diag=np.linspace(-1.0, 1.0, 1 << n))
    sched = Schedule.linear(T=1.0, h_x=np.ones(n))
    psi0 = uniform_superposition(n)
    ann = QuantumAnnealer(prob, sched)
    with pytest.raises(ValueError, match="controller"):
        ann.create_simulator(psi0, 0.0, method="m2", controller=ControllerConfig())


def test_controller_ignored_for_fixed_dt_run():
    """``QuantumAnnealer.run(method='m2', controller=...)`` は寛容 (無視, 例外なし).

    ``run`` は ``atol`` 等と同じく adaptive 専用パラメータを固定 dt 経路で silent
    無視する (strict に弾くのは Simulator のみ; 設計判断 issue #149)。
    """
    n = 2
    prob = IsingProblem(n=n, H_p_diag=np.linspace(-1.0, 1.0, 1 << n))
    sched = Schedule.linear(T=1.0, h_x=np.ones(n))
    psi0 = uniform_superposition(n)
    ann = QuantumAnnealer(prob, sched)
    # 例外を投げず正常終了すること (controller は無視される).
    res = ann.run(psi0, 0.0, 1.0, method="m2", n_steps=4, controller=ControllerConfig())
    assert res.method == "m2"
