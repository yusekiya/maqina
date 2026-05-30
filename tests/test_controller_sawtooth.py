"""層 A — 合成誤差ハーネスによるノコギリ波 characterization (issue #152).

``evolve_schedule_adaptive_*`` の現行 PI controller (旧 I 制御 + 固定 0.5 reject)
が、Magnus 4 次係数 ``C₄`` を急上昇させる合成シナリオで臨界領域に **ノコギリ波**
(過剰縮小 → 再上昇 → 再オーバーシュートの dt 振動) を起こすことを **決定論的に
固定** する characterization テスト。

これは umbrella #148 の各 sub-issue (#149 reject 予測式 + クランプ / #150 成長
凍結 / #151 真の PI 化) が「baseline → 各修正で ``log(dt)`` の lag-1 自己相関が
−1 から改善 / 受理率回復」を示すための **baseline** になる。本テストが green で
あること = 現行 main の病的挙動を正しく捉えていること。

3 ドライバ共通でノコギリ波が出るが、発火に必要な C₄ 立ち上がり率 ``k`` は推定子
order ``p`` で異なる:

- **Richardson / Chebyshev (p=4)**: operating dt ≈ ``tol^{1/5}`` ≈ 0.012 と大きく、
  per-step 成長率 ``exp(k·dt) > 1/safety⁵ ≈ 1.7`` の閾値を ``k ≈ 40`` で超える。
- **M2 (p=2)**: operating dt ≈ ``tol^{1/3}`` ≈ 0.002 と小さいため、同じ振動を起こす
  には急峻な ``k ≈ 250`` が要る (閾値 ``exp(k·dt) > 1/safety³``)。order が低いほど
  half-reject の over-shrink 量 (誤差 ``2^{p+1}`` 倍削減) も小さく、可視化に
  steep schedule を要する。

**重要 (order 整合)**: 合成誤差は ``err = C₄(t)·dt^{p+1}`` とし、order ``p`` を
driver 内 ``_pi_dt_next(p=...)`` に揃える (ハーネス側で自動)。揃えないと健全
schedule でも誤差予測がずれて *別種の* 持続的 reject が出てしまい、ノコギリ波
characterization を汚染する。

シナリオは純 float64 (実 Rust / BLAS を一切呼ばない)。``math.exp`` の 1-ulp 差で
境界 step の accept/reject が稀に揺れても、本テストは **絶対 ``n_rejects`` では
なく形状メトリクス (自己相関 / 受理率) を広いマージンで** assert するため
プラットフォーム非依存。
"""

from __future__ import annotations

import numpy as np
import pytest

from _controller_harness import exp_c4, run_synthetic
from _controller_metrics import (
    acceptance_rate,
    critical_window,
    log_dt_lag1_autocorr,
    summarize,
)

# 臨界点 t*=5 近傍で C₄ を急上昇させる per-driver シナリオ。``k`` は各 order の
# 閾値 (umbrella #148 の解析) から選定。dt_min / t1 は臨界帯を捉えつつ dt_min
# プラトーを短く保って高速化するための値 (数百 step で完了)。
#
# **issue #149 / #150 以後の重要事項**: 本ファイルは *umbrella #148 着手前の
# 旧挙動* (固定 0.5 半減 + 成長凍結なし) を baseline として固定し続ける
# characterization。issue #149 で reject 縮小の既定が予測式 + クランプ
# ``[0.2, 0.9]`` に、issue #150 で成長凍結の既定が ``True`` に変わったため、各
# シナリオに明示的に ``reject_shrink_min=reject_shrink_max=0.5`` と
# ``freeze_growth_after_reject=False`` を渡して旧挙動を再現する。新既定での改善
# (ノコギリ波解消) は ``test_controller_reject_clamp.py`` (#149) /
# ``test_controller_growth_freeze.py`` (#150) が検証する。
_T_STAR = 5.0
_LEGACY_REJECT = dict(
    reject_shrink_min=0.5,
    reject_shrink_max=0.5,
    freeze_growth_after_reject=False,
)
_CRITICAL: dict[str, dict] = {
    "richardson": dict(
        k=40.0,
        t0=0.0,
        t1=5.5,
        tol_step=1e-8,
        dt0=0.1,
        dt_min=1e-3,
        dt_max=0.5,
        max_rejects=500,
        **_LEGACY_REJECT,
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
        **_LEGACY_REJECT,
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
        **_LEGACY_REJECT,
    ),
}

_ALL_METHODS = ["m2", "richardson", "chebyshev"]


def _run_critical(method: str):
    """method 固有の臨界 C₄ シナリオを走らせて trace を返す."""
    cfg = dict(_CRITICAL[method])
    k = cfg.pop("k")
    c4 = exp_c4(k=k, t_star=_T_STAR)
    return run_synthetic(method, c4, **cfg)


# ---------------------------------------------------------------------------
# determinism (acceptance: 同一入力で n_rejects / dt_history 再現)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", _ALL_METHODS)
def test_synthetic_harness_is_deterministic(method):
    """合成ハーネスは決定論的: 2 回走らせて dt_history / n_rejects がビット一致."""
    a = _run_critical(method)
    b = _run_critical(method)
    assert np.array_equal(a.dt_history, b.dt_history)
    assert np.array_equal(a.t_history, b.t_history)
    assert a.n_rejects == b.n_rejects
    assert [(x.dt, x.accepted) for x in a.attempts] == [
        (y.dt, y.accepted) for y in b.attempts
    ]


# ---------------------------------------------------------------------------
# 層 A characterization: 3 ドライバ共通のノコギリ波 baseline
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", _ALL_METHODS)
def test_sawtooth_characterization(method):
    """現行 main が臨界窓でノコギリ波に入る (baseline; 3 ドライバ共通).

    臨界窓 (reject 集中域を自己同定) 内で:
        - accept された ``log(dt)`` の lag-1 自己相関が強く負 (dt 振動).
        - 受理率が 1.0 を大きく下回る (reject 多発).

    sub-issue #149-151 はこの assertion を「自己相関が −1 から改善 / 受理率回復」
    へ反転させて改善を示す。実測 baseline: richardson/chebyshev r1≈−0.80 acc≈0.42,
    m2 r1≈−0.76 acc≈0.42。マージン込みで r1<−0.4 / acc<0.6 を要求。
    """
    trace = _run_critical(method)
    assert trace.n_rejects >= 3
    window = critical_window(trace)
    assert window is not None

    r1 = log_dt_lag1_autocorr(trace, window)
    acc = acceptance_rate(trace, window)
    assert r1 < -0.4, f"{method}: expected sawtooth (r1<-0.4), got r1={r1}"
    assert acc < 0.6, f"{method}: expected low acceptance (<0.6), got acc={acc}"


# ---------------------------------------------------------------------------
# healthy contrast: 緩やかな C₄ では病態が出ない (メトリクスの sanity anchor)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", _ALL_METHODS)
def test_gentle_schedule_is_not_pathological(method):
    """緩やかな C₄ (立ち上がり率 1/20) では reject がほぼ無く受理率がほぼ 1.

    「メトリクスが何を走らせても病態を返す」わけではないことを保証する対照群。
    病的シナリオの ``k`` を 1/20 にすると I 制御でも追従でき、ノコギリ波が消える。
    """
    cfg = dict(_CRITICAL[method])
    k = cfg.pop("k")
    c4 = exp_c4(k=k / 20.0, t_star=_T_STAR)
    trace = run_synthetic(method, c4, **cfg)
    s = summarize(trace, None)
    assert s["acceptance_rate"] > 0.9, (
        f"{method}: gentle schedule should be healthy, got acc={s['acceptance_rate']}"
    )
