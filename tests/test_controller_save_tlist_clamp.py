"""``save_tlist`` クランプ step が PI controller state を汚染しないことの検証
(issue #167).

adaptive driver は ``save_tlist`` の観測時刻を厳密に踏むため ``dt`` を
``next_save_target - t`` にクランプする。このクランプ step の局所誤差を
controller へ流すと ``_pi_dt_next`` の成長制限 ``dt_next <= dt_try·growth_max``
がクランプ後の微小 ``dt_try`` を基準にするため、次 step の ``dt`` が ``dt_min``
床まで潰れ、自然な ``dt`` へ戻るまで ``log_{growth_max}(dt_nat / dt_min)`` step を
空費する。観測点数 ``K`` に比例して総 step 数が増える欠陥だった。

修正 (issue #167): **クランプ step が accept されたときは controller state
(``dt`` / ``err_prev`` / 成長凍結カウンタ) を据え置く** (DOPRI / CVODE の tstop
処理と同じ扱い)。

検証は #152 の合成誤差ハーネス (``_controller_harness``) を使う。``err =
C₄(t)·dt⁵`` を返す fake dispatch で controller ダイナミクスだけを切り出すため、
Rust 拡張の有無に依らず決定論的に走る。

``C₄`` 一定の合成則では純 I 制御 (``pi_alpha=1, pi_beta=0``) の更新式

.. code-block:: text

    dt_next = dt_try · safety · (tol_step / (C₄·dt_try⁵))^{1/5}
            = safety · (tol_step / C₄)^{1/5}     (dt_try が相殺)

が ``dt_try`` に依存しないため、非クランプ step の ``dt`` は **常に同一値**
``dt*`` になる。これを基準値として「クランプ step の次の ``dt`` が ``dt*`` に
戻っているか」を機械精度で検査できる (欠陥版では ``dt_min`` 床に落ちる)。

観測時刻は **ペア** ``(T_k, T_k + eps)`` で与える。``T_k`` を踏んだ直後に
``remaining = eps`` の微小クランプが必ず発生するので、step boundary との
偶然の一致に依存せず欠陥を再現できる (issue #167 の「あるテストで再現しな
かったは無害の証拠にならない」への対策)。
"""

from __future__ import annotations

import numpy as np
import pytest

from maqina import Schedule
from maqina.initial_states import uniform_superposition
from maqina.krylov import evolve_schedule_adaptive_richardson

from _controller_harness import exp_c4, run_synthetic

# 合成ハーネス共通条件. ``dt_min=1e-6`` は欠陥版の回復コスト
# ``log_4(dt* / dt_min) ≈ 7 step`` を明瞭に出すための値 (production 既定は 1e-4).
_T1 = 20.0
_TOL_STEP = 1e-8
_DT0 = 0.02
_DT_MIN = 1e-6
# 観測ペアの間隔. ``_MERGE_TOL = 1e-12`` より十分大きく (2 点が merge されない),
# ``dt_min`` より十分小さい (クランプが「微小 step」になる) 範囲で選ぶ.
_PAIR_EPS = 1e-8
_METHODS = ("richardson", "chebyshev")


def _paired_targets(n_pairs: int = 20) -> np.ndarray:
    """``(T_k, T_k + eps)`` のペアからなる観測時刻列を返す.

    ``T_k`` を厳密に踏んだ直後に ``remaining = eps`` の微小クランプが必ず起きる
    ため、step boundary と観測時刻の偶然の位相関係に依存せず欠陥を再現できる。
    """
    anchors = np.linspace(1.0, _T1 - 1.0, n_pairs)
    return np.sort(np.concatenate([anchors, anchors + _PAIR_EPS]))


@pytest.mark.parametrize("method", _METHODS)
def test_clamped_step_does_not_collapse_dt(method: str) -> None:
    """クランプ step の次の ``dt`` が自然 ``dt*`` に据え置かれている.

    ``C₄`` 一定なので非クランプ step の ``dt`` は一意な ``dt*``。微小クランプ
    (``dt = eps``) の直後の step が ``dt*`` と機械精度で一致することを検査する。
    欠陥版ではここが ``dt_min`` (= 1e-6) に潰れ、4 倍ずつの指数回復が始まる。
    """
    c4 = exp_c4(0.0, 0.0)  # C₄(t) = 1 (定数)
    base = run_synthetic(
        method, c4, t1=_T1, tol_step=_TOL_STEP, dt0=_DT0, dt_min=_DT_MIN
    )
    # 基準 dt*: 終端クランプ (最終 step) と warm-up (先頭 step) を除く定常値.
    dt_star = float(np.median(base.dt_history[1:-1]))
    assert np.allclose(base.dt_history[1:-1], dt_star, rtol=1e-12, atol=0.0), (
        "C₄ 一定なら非クランプ step の dt は一意なはず"
    )

    trace = run_synthetic(
        method,
        c4,
        t1=_T1,
        tol_step=_TOL_STEP,
        dt0=_DT0,
        dt_min=_DT_MIN,
        save_tlist=_paired_targets(),
    )
    dts = trace.dt_history
    tiny = np.flatnonzero(dts <= _PAIR_EPS * 10.0)
    assert tiny.size == 20, f"微小クランプ step が想定数だけ発生していない: {tiny.size}"
    for i in tiny:
        assert i + 1 < dts.size, "微小 step が最終 step になる構成は想定外"
        assert dts[i + 1] == pytest.approx(dt_star, rel=1e-12), (
            f"クランプ step (idx={i}) の次 dt が据え置かれていない: "
            f"dt={dts[i + 1]:.6e}, expected={dt_star:.6e}"
        )
    # 欠陥版の指標: dt_min 床への張り付きが 1 度も起きない.
    assert not np.any(np.isclose(dts, _DT_MIN, rtol=1e-12, atol=0.0)), (
        "dt が dt_min 床まで潰れている (controller state 汚染の徴候)"
    )


@pytest.mark.parametrize("method", _METHODS)
@pytest.mark.parametrize("c4_label", ("const", "exp"))
def test_save_tlist_step_overhead_is_bounded(method: str, c4_label: str) -> None:
    """``save_tlist`` 有無の accept step 数比が 1.10 未満に収まる.

    クランプ step そのものは避けられない (target ごとに 1 step 増える) が、
    controller 汚染による回復コストが乗ると比が跳ね上がる。20 ペア (= 40 target)
    の構成で欠陥版は 1.18〜1.21、修正版は 1.03〜1.04。
    """
    c4 = exp_c4(0.0, 0.0) if c4_label == "const" else exp_c4(0.3, 12.0)
    base = run_synthetic(
        method, c4, t1=_T1, tol_step=_TOL_STEP, dt0=_DT0, dt_min=_DT_MIN
    )
    trace = run_synthetic(
        method,
        c4,
        t1=_T1,
        tol_step=_TOL_STEP,
        dt0=_DT0,
        dt_min=_DT_MIN,
        save_tlist=_paired_targets(),
    )
    ratio = trace.dt_history.size / base.dt_history.size
    assert ratio < 1.10, (
        f"save_tlist の step 数 overhead が大きすぎる: {ratio:.4f}x "
        f"(base={base.dt_history.size}, save={trace.dt_history.size})"
    )
    # 据え置きで reject が増えていないこと (精度側の契約は不変).
    assert trace.n_rejects == base.n_rejects


@pytest.mark.parametrize("method", _METHODS)
def test_save_tlist_targets_are_hit_exactly(method: str) -> None:
    """据え置き修正後も全 target を厳密に踏む (merge 契約の回帰防止)."""
    targets = _paired_targets()
    trace = run_synthetic(
        method,
        exp_c4(0.3, 12.0),
        t1=_T1,
        tol_step=_TOL_STEP,
        dt0=_DT0,
        dt_min=_DT_MIN,
        save_tlist=targets,
    )
    for target in targets:
        assert np.min(np.abs(trace.t_history - target)) <= 1e-12, (
            f"target t={target!r} を踏んでいない"
        )


def test_save_tlist_final_state_matches_no_save() -> None:
    """実 driver (Lanczos) で ``save_tlist`` 有無の終端状態が ``tol_step`` 内で一致.

    据え置き修正は step 列を変えるのでビット一致は取れない。局所誤差制御が
    効いている限り終端状態は ``tol_step`` オーダで一致するべき、という契約を
    確認する (合成ハーネスは ψ を更新しないので実 driver で見る必要がある)。
    """
    n = 3
    rng = np.random.default_rng(167)
    h_p_diag = rng.normal(size=1 << n).astype(np.float64)
    h_x = np.ones(n, dtype=np.float64)
    t1 = 2.0
    sched = Schedule.linear(T=t1, h_x=h_x)
    psi0 = uniform_superposition(n)
    # 各 anchor 直後に微小クランプを起こすペア構成 (合成テストと同じ狙い).
    anchors = np.linspace(0.2, t1 - 0.2, 8)
    save_tlist = np.sort(np.concatenate([[0.0], anchors, anchors + 1e-8, [t1]]))

    common = dict(
        h_p_diag=h_p_diag,
        schedule=sched,
        psi0=psi0,
        t0=0.0,
        t1=t1,
        tol_step=1e-10,
        dt0=0.05,
    )
    psi_no_save = evolve_schedule_adaptive_richardson(**common)[0]
    psi_save = evolve_schedule_adaptive_richardson(**common, save_tlist=save_tlist)[0]

    rel = float(np.linalg.norm(psi_save - psi_no_save) / np.linalg.norm(psi_no_save))
    assert rel < 1e-8, f"save_tlist 経路の終端状態がずれすぎ: rel={rel:.3e}"
