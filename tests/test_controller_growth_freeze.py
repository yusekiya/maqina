"""issue #150 — reject 後の dt 成長凍結 (Gustafsson ヒステリシス).

4 軸でカバーする:

1. **層 A (主役 / 合成誤差ハーネス)**: reject 縮小が **過剰縮小** する regime
   (legacy ``reject_shrink_min=max=0.5``) でノコギリ波を発火させ、同一 ``C₄(t)``
   で ``freeze=False`` (= #149 完了時点) vs ``freeze=True`` (新既定) を比較。
   ``True`` で reject 直後の dt 再上昇が断たれ、受理率↑ / ``n_rejects`` 非劣化〜減
   / ``log(dt)`` の lag-1 自己相関が −1 から改善 / reversals 減 を assert。
   なぜ over-shrink regime かは下のモジュール docstring 注を参照。
2. **非劣化 (#149 既定クランプ ``[0.2, 0.9]``)**: 予測式 reject が reject 直後の
   accept を「PI 成長率 ≈ 1.0」に着地させるため、既定クランプ下では成長凍結の
   増分はほぼ無い。ここでは ``freeze=True`` が ``False`` を **悪化させない**
   (``n_rejects`` 非劣化 / autocorr 非劣化) ことだけを担保する。
3. **解除基準 (``growth_freeze_steps``)**: 凍結を解除するまでの連続 accept 回数を
   増やすと、reject 直後の dt 再上昇がより長く抑止され抑制が強まる
   (reversals 単調非増 + 大きい ``growth_freeze_steps`` でノコギリ波消滅)。
4. **回帰アンカー + 入力検証 + facade 配線**: ``freeze=False`` では
   ``growth_freeze_steps`` に依らず ``dt_history`` がビット一致する (flag が機能を
   完全 gate)。``ControllerConfig`` の範囲検証と、``controller=ControllerConfig(...)``
   が facade から driver まで全 8 field 伝播することを確認。

注 (over-shrink regime を主役シナリオに選ぶ理由):
    成長凍結は「reject → 過剰縮小 → 楽々 accept (err ≪ tol) → I 制御が大きく
    再拡大 → 再オーバーシュート」という limit cycle の **再拡大側** を断つ機構。
    #149 の予測式 reject (既定 ``[0.2, 0.9]``) は reject 縮小量を「次 accept が
    err ≈ tol に着地し PI 成長率 ≈ 1.0 になる」よう自己整合的に選ぶので、既定
    クランプ下では over-shrink 起因の再拡大がそもそも起きず、成長凍結は発火
    しない (合成ハーネスで ``freeze`` on/off がビット一致する)。よって成長凍結
    **固有の効果** を測るには、reject 縮小を過剰にする legacy ``0.5/0.5`` regime
    (= #149 の ``test_controller_reject_clamp`` が病的と実証した設定) を使うのが
    適切。実運用では既定クランプとの二重の安全網として働き、user が
    ``reject_shrink`` を攻めた値にしても再拡大ループに陥りにくくする。
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

# ---------------------------------------------------------------------------
# シナリオ: #149 のノコギリ波発火設定 + legacy over-shrink (0.5/0.5)
# ---------------------------------------------------------------------------

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
# 過剰縮小 regime (成長凍結の効果を発火させる; モジュール docstring 注参照).
_OVERSHRINK = dict(reject_shrink_min=0.5, reject_shrink_max=0.5)


def _freeze_off_on(method: str, *, growth_freeze_steps: int = 1):
    """同一 ``C₄`` (over-shrink regime) で freeze off / on の trace を返す."""
    cfg = dict(_SCENARIO[method])
    k = cfg.pop("k")
    c4 = exp_c4(k=k, t_star=_T_STAR)
    off = run_synthetic(
        method, c4, freeze_growth_after_reject=False, **_OVERSHRINK, **cfg
    )
    on = run_synthetic(
        method,
        c4,
        freeze_growth_after_reject=True,
        growth_freeze_steps=growth_freeze_steps,
        **_OVERSHRINK,
        **cfg,
    )
    return off, on


# ---------------------------------------------------------------------------
# 層 A 主役: 成長凍結で over-shrink ノコギリ波が緩和される
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", _ALL_METHODS)
def test_growth_freeze_reduces_overshrink_sawtooth(method):
    """``freeze=True`` で over-shrink 起因のノコギリ波が緩和される.

    legacy 固定 0.5 縮小は reject 直後に err を 32× 削減し、I 制御が大きく
    再拡大 → 再オーバーシュートする (lag-1 自己相関が −1 寄り)。成長凍結は
    reject 直後の accept で拡大を禁止するため再拡大ループが断たれ、受理率が
    回復し reject が減り、自己相関が改善する。
    """
    off, on = _freeze_off_on(method, growth_freeze_steps=2)

    # baseline (freeze off) は病的: reject 多発 + 強い負の自己相関.
    assert off.n_rejects >= 3, f"{method}: baseline should be pathological"
    win = critical_window(off)
    assert win is not None
    r1_off = log_dt_lag1_autocorr(off, win)
    assert r1_off < -0.4, f"{method}: baseline sawtooth not captured (r1={r1_off})"

    # 非劣化〜減: reject 数は増えない.
    assert on.n_rejects <= off.n_rejects, (
        f"{method}: rejects regressed, off={off.n_rejects} on={on.n_rejects}"
    )
    # 受理率は非劣化 (全区間).
    assert acceptance_rate(on, None) >= acceptance_rate(off, None) - 1e-12, (
        f"{method}: acceptance regressed"
    )
    # 自己相関が改善 (−1 から離れる) する.
    r1_on = log_dt_lag1_autocorr(on, win)
    assert np.isnan(r1_on) or r1_on > r1_off, (
        f"{method}: autocorr did not improve, off={r1_off} on={r1_on}"
    )
    # reversals (dt 増減反転回数) が減る.
    assert n_reversals(on, win) <= n_reversals(off, win), (
        f"{method}: reversals regressed"
    )


# ---------------------------------------------------------------------------
# 非劣化: #149 既定クランプ [0.2, 0.9] では悪化させない
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", _ALL_METHODS)
def test_growth_freeze_no_regression_on_default_clamp(method):
    """既定クランプ ``[0.2, 0.9]`` では ``freeze=True`` が ``False`` を悪化させない.

    予測式 reject が reject 直後の accept を PI 成長率 ≈ 1.0 に着地させるため、
    既定クランプ下では成長凍結はほぼ no-op (発火しない)。ここでは「reject が
    増えない」「臨界窓 autocorr が悪化しない」非劣化だけを担保する。
    """
    cfg = dict(_SCENARIO[method])
    k = cfg.pop("k")
    c4 = exp_c4(k=k, t_star=_T_STAR)  # default [0.2, 0.9] クランプ
    off = run_synthetic(method, c4, freeze_growth_after_reject=False, **cfg)
    on = run_synthetic(method, c4, freeze_growth_after_reject=True, **cfg)

    assert on.n_rejects <= off.n_rejects, (
        f"{method}: rejects regressed on default clamp, "
        f"off={off.n_rejects} on={on.n_rejects}"
    )
    win = critical_window(off)
    if win is not None:
        r1_off = log_dt_lag1_autocorr(off, win)
        r1_on = log_dt_lag1_autocorr(on, win)
        if not (np.isnan(r1_off) or np.isnan(r1_on)):
            assert r1_on >= r1_off - 1e-9, (
                f"{method}: autocorr regressed on default clamp, "
                f"off={r1_off} on={r1_on}"
            )


# ---------------------------------------------------------------------------
# 解除基準: growth_freeze_steps を増やすと凍結が長く持続し抑制が強まる
# ---------------------------------------------------------------------------


def test_growth_freeze_steps_monotone_suppression():
    """``growth_freeze_steps`` を増やすと reject 直後の再拡大抑止が強まる.

    over-shrink regime (richardson) で ``growth_freeze_steps ∈ {1, 2, 3}`` を
    振り、臨界窓内の dt 増減反転回数 ``n_reversals`` が単調非増になることを確認
    (大きい凍結ステップほどノコギリ波が平坦化)。``freeze=False`` を起点に少なくとも
    1 段は厳密に減ることも assert。
    """
    cfg = dict(_SCENARIO["richardson"])
    k = cfg.pop("k")
    c4 = exp_c4(k=k, t_star=_T_STAR)

    def rev(freeze: bool, gfs: int) -> int:
        tr = run_synthetic(
            "richardson",
            c4,
            freeze_growth_after_reject=freeze,
            growth_freeze_steps=gfs,
            **_OVERSHRINK,
            **cfg,
        )
        win = critical_window(tr)
        return n_reversals(tr, win) if win is not None else 0

    win_off = critical_window(
        run_synthetic(
            "richardson",
            c4,
            freeze_growth_after_reject=False,
            **_OVERSHRINK,
            **cfg,
        )
    )
    # 反転回数は freeze off の臨界窓基準で測ると scenario 依存になるので、各 run
    # 自身の臨界窓で測った反転回数列が単調非増であることを見る (凍結強化の単調性).
    rev_off = rev(False, 1)
    rev_1 = rev(True, 1)
    rev_2 = rev(True, 2)
    rev_3 = rev(True, 3)

    assert rev_1 <= rev_off, f"gfs=1 should not worsen vs off: {rev_1} > {rev_off}"
    assert rev_2 <= rev_1, f"gfs=2 should not worsen vs gfs=1: {rev_2} > {rev_1}"
    assert rev_3 <= rev_2, f"gfs=3 should not worsen vs gfs=2: {rev_3} > {rev_2}"
    # 凍結を十分強める (gfs>=2) と off より厳密に平坦化する.
    assert rev_2 < rev_off, (
        f"sufficient freeze (gfs=2) should strictly flatten: rev_2={rev_2} "
        f"rev_off={rev_off} (win_off={win_off})"
    )


# ---------------------------------------------------------------------------
# 回帰アンカー: freeze=False は growth_freeze_steps に依らずビット一致
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", _ALL_METHODS)
def test_freeze_off_is_independent_of_growth_freeze_steps(method):
    """``freeze=False`` では ``growth_freeze_steps`` を変えても dt_history がビット一致.

    flag が機能を完全に gate していること (= #149 完了時点の挙動を厳密再現する
    回帰アンカー) を確認する。``freeze=False`` の accept 経路は新コードを通らない
    ので ``growth_freeze_steps`` は無視されねばならない。
    """
    cfg = dict(_SCENARIO[method])
    k = cfg.pop("k")
    c4 = exp_c4(k=k, t_star=_T_STAR)
    base = run_synthetic(
        method,
        c4,
        freeze_growth_after_reject=False,
        growth_freeze_steps=1,
        **_OVERSHRINK,
        **cfg,
    )
    other = run_synthetic(
        method,
        c4,
        freeze_growth_after_reject=False,
        growth_freeze_steps=99,
        **_OVERSHRINK,
        **cfg,
    )
    np.testing.assert_array_equal(base.dt_history, other.dt_history)
    np.testing.assert_array_equal(base.t_history, other.t_history)
    assert base.n_rejects == other.n_rejects


# ---------------------------------------------------------------------------
# ControllerConfig 入力検証 (新 field)
# ---------------------------------------------------------------------------


def test_controller_config_freeze_defaults():
    """default 構築は新挙動 (freeze 有効 / 凍結 1 step) と一致する."""
    c = ControllerConfig()
    assert c.freeze_growth_after_reject is True
    assert c.growth_freeze_steps == 1


@pytest.mark.parametrize("gfs", [0, -1, -5])
def test_controller_config_growth_freeze_steps_must_be_ge_1(gfs):
    """``growth_freeze_steps < 1`` は ``ValueError``."""
    with pytest.raises(ValueError):
        ControllerConfig(growth_freeze_steps=gfs)


def test_controller_config_growth_freeze_steps_must_be_int():
    """``growth_freeze_steps`` に float を渡すと ``ValueError`` (int 厳格)."""
    with pytest.raises(ValueError):
        ControllerConfig(growth_freeze_steps=1.0)  # type: ignore[arg-type]


def test_controller_config_freeze_disabled_is_valid():
    """``freeze_growth_after_reject=False`` は (gfs 既定のまま) 構築できる."""
    c = ControllerConfig(freeze_growth_after_reject=False)
    assert c.freeze_growth_after_reject is False
    assert c.growth_freeze_steps == 1


# ---------------------------------------------------------------------------
# facade 配線: ControllerConfig の全 8 field が driver まで伝播する
# ---------------------------------------------------------------------------

_CUSTOM = ControllerConfig(
    safety=0.7,
    growth_max=2.0,
    max_rejects=7,
    dt_min=2e-3,
    reject_shrink_min=0.3,
    reject_shrink_max=0.8,
    freeze_growth_after_reject=False,
    growth_freeze_steps=4,
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


def _assert_all_fields(captured: dict, ctrl: ControllerConfig):
    assert captured["safety"] == ctrl.safety
    assert captured["growth_max"] == ctrl.growth_max
    assert captured["max_rejects"] == ctrl.max_rejects
    assert captured["dt_min"] == ctrl.dt_min
    assert captured["reject_shrink_min"] == ctrl.reject_shrink_min
    assert captured["reject_shrink_max"] == ctrl.reject_shrink_max
    assert captured["freeze_growth_after_reject"] == ctrl.freeze_growth_after_reject
    assert captured["growth_freeze_steps"] == ctrl.growth_freeze_steps


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
def test_freeze_fields_propagate_through_run(method, driver_attr, monkeypatch):
    """``QuantumAnnealer.run(controller=...)`` が driver に全 8 field を渡す."""
    n = 2
    prob = IsingProblem(n=n, H_p_diag=np.linspace(-1.0, 1.0, 1 << n))
    sched = Schedule.linear(T=1.0, h_x=np.ones(n))
    psi0 = uniform_superposition(n)

    captured: dict = {}
    monkeypatch.setattr(_annealer_mod, driver_attr, _make_capture(captured))

    ann = QuantumAnnealer(prob, sched)
    ann.run(psi0, 0.0, 1.0, method=method, controller=_CUSTOM)
    _assert_all_fields(captured, _CUSTOM)


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
def test_freeze_fields_propagate_through_simulator(method, driver_attr, monkeypatch):
    """``AnnealingSimulator(controller=...)`` が driver に全 8 field を渡す."""
    n = 2
    prob = IsingProblem(n=n, H_p_diag=np.linspace(-1.0, 1.0, 1 << n))
    sched = Schedule.linear(T=1.0, h_x=np.ones(n))
    psi0 = uniform_superposition(n)

    captured: dict = {}
    monkeypatch.setattr(_simulator_mod, driver_attr, _make_capture(captured))

    sim = AnnealingSimulator(prob, sched, psi0, 0.0, method=method, controller=_CUSTOM)
    sim.step(0.5)
    _assert_all_fields(captured, _CUSTOM)


def test_freeze_fields_default_when_none(monkeypatch):
    """``controller=None`` (既定) で freeze 既定 (True / 1) が driver に渡る."""
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
    assert captured["freeze_growth_after_reject"] is True
    assert captured["growth_freeze_steps"] == 1
