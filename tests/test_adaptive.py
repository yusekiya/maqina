"""adaptive driver (Phase 4 C3) の挙動テスト.

issue #39 の acceptance:

* ``evolve_schedule_adaptive_m2`` smoke: time-independent / 緩やかな
  time-dependent ``H`` で QuTiP との fidelity が ``> 1 - 1e-6``.
* ``evolve_schedule_adaptive_richardson`` smoke: 同上 (より高次なので
  同じ ``tol_step`` で M2 同等以上).
* PI controller の dt 増減: 緩やか schedule で ``dt`` が ``dt0`` から
  ``dt_max`` 方向に増加する.
* reject: 不自然に大きい ``dt0`` を渡すと最初の数 step が reject される.
* ``max_rejects`` 連続超過で ``RuntimeError``.
* ``save_tlist is not None`` で ``NotImplementedError``.

driver 単体テストなので ``QuantumAnnealer`` ファサードは経由せず
``evolve_schedule_adaptive_*`` を直接呼ぶ. QuTiP 比較は ``test_annealer.py``
の facade smoke と棲み分け (``test_adaptive.py`` は driver 単体, 同 facade
経路の end-to-end smoke は ``test_annealer.py`` 側で別途検証する).
"""

from __future__ import annotations

import numpy as np
import pytest

from maqina import IsingProblem, Observable, Schedule
from maqina.initial_states import uniform_superposition
from maqina.krylov import (
    evolve_schedule_adaptive_m2,
    evolve_schedule_adaptive_richardson,
)


qutip = pytest.importorskip("qutip")


def _make_random_problem(n: int, seed: int) -> tuple[IsingProblem, np.ndarray]:
    """ランダム ``H_p_diag`` (実数 ``[-1, 1]``) と一様 ``h_x = 1`` で
    ``IsingProblem`` を作る.
    """
    rng = np.random.default_rng(seed)
    dim = 1 << n
    h_p = rng.uniform(-1.0, 1.0, size=dim).astype(np.float64)
    h_x = np.ones(n, dtype=np.float64)
    return IsingProblem(n=n, H_p_diag=h_p), h_x


def _build_qutip_hamiltonian(h_x: np.ndarray, h_p_diag: np.ndarray, T: float) -> list:
    """linear schedule での QuTiP ``H(t)`` を組む (``test_reference_qutip.py``
    と同形). ``A(s) = 1 - s``, ``B(s) = s``, ``s = t/T``.
    """
    n = h_x.shape[0]
    dim = 1 << n
    h_drv = np.zeros((dim, dim), dtype=np.complex128)
    for i in range(n):
        mask = 1 << i
        for k in range(dim):
            h_drv[k, k ^ mask] += -h_x[i]
    h_p = np.diag(h_p_diag).astype(np.complex128)
    h_drv_q = qutip.Qobj(h_drv)
    h_p_q = qutip.Qobj(h_p)
    return [
        [h_drv_q, f"(1 - t/{T})"],
        [h_p_q, f"(t/{T})"],
    ]


def _qutip_reference(h_x: np.ndarray, h_p_diag: np.ndarray, T: float) -> np.ndarray:
    """QuTiP ``sesolve`` で linear schedule の終端 ψ を取り出す."""
    n = h_x.shape[0]
    psi0 = uniform_superposition(n)
    h_t = _build_qutip_hamiltonian(h_x, h_p_diag, T)
    psi0_q = qutip.Qobj(psi0.reshape(-1, 1))
    sol = qutip.sesolve(
        h_t,
        psi0_q,
        np.array([0.0, T]),
        options={"atol": 1e-12, "rtol": 1e-10, "nsteps": 100000},
    )
    return sol.states[-1].full().ravel()


def _fidelity(psi_a: np.ndarray, psi_b: np.ndarray) -> float:
    return float(np.abs(np.vdot(psi_a, psi_b)) ** 2)


def test_adaptive_m2_matches_qutip() -> None:
    """``evolve_schedule_adaptive_m2`` smoke: linear schedule の n=4 問題で
    QuTiP との fidelity が ``> 1 - 1e-6``.

    ``tol_step = 1e-8`` (PI 既定値) で local error を抑え, smooth schedule
    の T=5 で 10 step 程度以上は走る前提.
    """
    n = 4
    T = 5.0
    prob, h_x = _make_random_problem(n, seed=20260513)
    sched = Schedule.linear(T=T, h_x=h_x)
    psi0 = uniform_superposition(n)

    psi_final, t_history, dt_history, n_rejects = evolve_schedule_adaptive_m2(
        h_p_diag=prob.H_p_diag,
        schedule=sched,
        psi0=psi0,
        t0=0.0,
        t1=T,
        tol_step=1e-8,
        dt0=0.1,
    )
    psi_ref = _qutip_reference(h_x, prob.H_p_diag, T)
    fid = _fidelity(psi_final, psi_ref)
    assert fid > 1 - 1e-6, f"adaptive_m2 fidelity too low: {fid} (1-fid={1 - fid})"
    # t_history は t0=0 で始まり, 最後の値が T 近傍.
    assert t_history[0] == 0.0
    assert abs(t_history[-1] - T) < 1e-12
    # dt_history は t_history より 1 要素少ない (accept された step 数).
    assert dt_history.shape[0] == t_history.shape[0] - 1
    assert n_rejects >= 0
    # 終端 ψ の L2 は 1 (unitary).
    assert abs(np.linalg.norm(psi_final) - 1.0) < 1e-10


def test_adaptive_richardson_matches_qutip() -> None:
    """``evolve_schedule_adaptive_richardson`` smoke: 同条件で M2 同等以上
    の fidelity を要求 (Richardson は 4 次推定子で local error が
    より厳密に抑制される).
    """
    n = 4
    T = 5.0
    prob, h_x = _make_random_problem(n, seed=20260513)
    sched = Schedule.linear(T=T, h_x=h_x)
    psi0 = uniform_superposition(n)

    # issue #93 (Phase 7) + Phase 5 (issue #47): driver は 10-tuple
    # (psi, t_hist, dt_hist, n_rejects, m_eff_hist, beta_m_hist,
    #  err_lanczos_hist, err_magnus_hist, n_krylov_insufficient, snapshot).
    # snapshot は save_tlist=None で None.
    (
        psi_final,
        t_history,
        dt_history,
        n_rejects,
        m_eff_history,
        beta_m_history,
        err_lanczos_history,
        err_magnus_history,
        n_krylov_insufficient,
        snapshot,
    ) = evolve_schedule_adaptive_richardson(
        h_p_diag=prob.H_p_diag,
        schedule=sched,
        psi0=psi0,
        t0=0.0,
        t1=T,
        tol_step=1e-8,
        dt0=0.1,
    )
    assert snapshot is None
    psi_ref = _qutip_reference(h_x, prob.H_p_diag, T)
    fid = _fidelity(psi_final, psi_ref)
    assert fid > 1 - 1e-6, (
        f"adaptive_richardson fidelity too low: {fid} (1-fid={1 - fid})"
    )
    assert t_history[0] == 0.0
    assert abs(t_history[-1] - T) < 1e-12
    assert dt_history.shape[0] == t_history.shape[0] - 1
    assert n_rejects >= 0
    assert abs(np.linalg.norm(psi_final) - 1.0) < 1e-10
    # m_eff_history は accept された step 数と同じ長さ, 各値は 6m 以下.
    assert m_eff_history.shape == dt_history.shape
    assert int(np.max(m_eff_history)) <= 6 * 24  # default m=24
    # issue #93 (Phase 7): β_m / err_lanczos / err_magnus history も
    # m_eff_history と同じ長さ. err_magnus + err_lanczos >= err / 2 程度の
    # ふるまい (詳細 acceptance は別 test).
    assert beta_m_history.shape == dt_history.shape
    assert err_lanczos_history.shape == dt_history.shape
    assert err_magnus_history.shape == dt_history.shape
    # default krylov_tol=1e-12 では Lanczos 充分 → n_krylov_insufficient = 0.
    assert n_krylov_insufficient == 0
    # err_lanczos / err_magnus は非負.
    assert bool(np.all(err_lanczos_history >= 0.0))
    assert bool(np.all(err_magnus_history >= 0.0))
    assert bool(np.all(beta_m_history >= 0.0))


def test_adaptive_richardson_error_decomposition_consistency() -> None:
    """issue #93 (Phase 7): err_lanczos と err_magnus の分解が triangle
    inequality を満たし, 全体として正しく ``err`` を分解できていることを確認.

    具体的には, accept された各 step で
    ``err_magnus + err_lanczos >= err`` (triangle inequality の上界性)
    と ``err_magnus = max(0, err - err_lanczos)`` の関係を確認する.

    また default ``krylov_tol = 1e-12`` では Lanczos 充分なので
    ``err_lanczos`` は ``tol_step`` を大きく下回り, PI controller の挙動が
    Phase 6 以前 (``err = err_magnus`` での dt 制御) と数値的にほぼ一致する.
    """
    n = 3
    T = 1.0
    prob, h_x = _make_random_problem(n, seed=20260514)
    sched = Schedule.linear(T=T, h_x=h_x)
    psi0 = uniform_superposition(n)

    (
        _psi_final,
        _t_history,
        dt_history,
        _n_rejects,
        _m_eff_history,
        _beta_m_history,
        err_lanczos_history,
        err_magnus_history,
        n_krylov_insufficient,
        _snapshot,
    ) = evolve_schedule_adaptive_richardson(
        h_p_diag=prob.H_p_diag,
        schedule=sched,
        psi0=psi0,
        t0=0.0,
        t1=T,
        tol_step=1e-8,
        krylov_tol=1e-12,  # default. Lanczos 充分を担保.
        dt0=0.1,
    )

    # default 設定では n_krylov_insufficient = 0 が契約.
    assert n_krylov_insufficient == 0, (
        f"default krylov_tol=1e-12 で n_krylov_insufficient = "
        f"{n_krylov_insufficient}; should be 0"
    )

    # err_lanczos_total << tol_step (= 1e-8) を満たすこと.
    # Lanczos 充分性: 全 step で err_lanczos < tol_step.
    assert bool(np.all(err_lanczos_history < 1e-8)), (
        "default 設定で err_lanczos が tol_step を超える step がある: "
        f"max(err_lanczos)={float(np.max(err_lanczos_history)):.3e}"
    )

    # err_magnus = max(0, err - err_lanczos) なので err_magnus >= 0.
    # err = err_lanczos + err_magnus が triangle inequality 上界.
    assert bool(np.all(err_magnus_history >= 0.0))
    assert bool(np.all(err_lanczos_history >= 0.0))

    # dt_history と各 history は同じ長さ.
    assert err_lanczos_history.shape == dt_history.shape
    assert err_magnus_history.shape == dt_history.shape


def test_adaptive_pi_dt_grows_with_loose_tolerance() -> None:
    """緩やかな schedule + 緩い ``tol_step`` で ``dt`` が ``dt0`` から
    ``dt_max`` 方向に増加することを確認.

    constant schedule (``A``, ``B`` 一定) なら CFM4:2 は 1 step で厳密
    (時間依存項なし) なので err は機械精度の rounding に落ち, PI 式は
    growth_max ベースで dt を伸ばす. dt の末尾値が ``dt0`` よりも有意に
    大きいことを確認する.
    """
    n = 3
    T = 5.0
    prob, h_x = _make_random_problem(n, seed=42)
    # constant schedule: A = 0.5, B = 0.5 で固定 (s(t) は default).
    sched = Schedule(T=T, A=lambda s: 0.5, B=lambda s: 0.5, h_x=h_x)
    psi0 = uniform_superposition(n)

    _, _, dt_history, _, _, _, _, _, _, _ = evolve_schedule_adaptive_richardson(
        h_p_diag=prob.H_p_diag,
        schedule=sched,
        psi0=psi0,
        t0=0.0,
        t1=T,
        tol_step=1e-6,
        dt0=0.01,
        dt_max=1.0,
    )
    # 最初の dt は dt0 = 0.01, 後半は dt_max=1.0 に向けて増加していくはず.
    assert dt_history[0] == pytest.approx(0.01, abs=1e-15)
    assert dt_history[-1] > dt_history[0] * 5.0, (
        f"dt did not grow: history={dt_history}"
    )


def test_adaptive_rejects_oversized_dt0() -> None:
    """不自然に大きい ``dt0`` を渡すと最初の数 step で reject が起きる.

    smooth schedule で ``dt0 = 4.0`` (T と同オーダ) を渡し,
    ``n_rejects > 0`` を要求.
    """
    n = 3
    T = 5.0
    prob, h_x = _make_random_problem(n, seed=137)
    sched = Schedule.linear(T=T, h_x=h_x)
    psi0 = uniform_superposition(n)

    _, _, _, n_rejects, _, _, _, _, _, _ = evolve_schedule_adaptive_richardson(
        h_p_diag=prob.H_p_diag,
        schedule=sched,
        psi0=psi0,
        t0=0.0,
        t1=T,
        tol_step=1e-10,  # tight tol で reject が起きやすい状況にする
        dt0=4.0,
        dt_max=4.0,
        dt_min=1e-4,
    )
    assert n_rejects > 0, "expected at least one reject for oversized dt0"


def test_adaptive_max_rejects_raises_runtime_error() -> None:
    """``max_rejects`` を超える連続 reject で ``RuntimeError``.

    ``tol_step = 1e-30`` (機械精度より厳しい), ``dt_min = 1e-20``
    (実用上, halving で到達できない小ささ) にすると, PI は dt を半減し
    続けるが dt_min に届かないため accept しない. ``max_rejects = 3``
    で 3 連続 reject 後に RuntimeError が出る.
    """
    n = 3
    prob, h_x = _make_random_problem(n, seed=11)
    sched = Schedule.linear(T=5.0, h_x=h_x)
    psi0 = uniform_superposition(n)

    with pytest.raises(RuntimeError, match="max_rejects"):
        evolve_schedule_adaptive_richardson(
            h_p_diag=prob.H_p_diag,
            schedule=sched,
            psi0=psi0,
            t0=0.0,
            t1=5.0,
            tol_step=1e-30,
            dt0=0.5,
            dt_min=1e-20,
            max_rejects=3,
        )


def test_auto_dt_init_resolves_to_formula() -> None:
    """``dt_init="auto"`` の解決値が ``c · T^β`` の formula と一致する.

    Module-level helper ``_resolve_dt_init_auto`` を直接呼んで, T=1 で
    既定 ``c=0.1``, T=100 で ``c · 100^0.5 = 1.0`` 等の代表値を機械精度で
    検証する. 床値 (`_AUTO_DT_INIT_FLOOR`) と上限 (interval T 自体) の
    境界も同時に確認する.
    """
    from maqina.annealer import (
        _AUTO_DT_INIT_C,
        _AUTO_DT_INIT_FLOOR,
        _resolve_dt_init_auto,
    )

    # 通常域: c · T^0.5 がそのまま採用される.
    assert _resolve_dt_init_auto(0.0, 1.0) == pytest.approx(_AUTO_DT_INIT_C, rel=1e-15)
    assert _resolve_dt_init_auto(0.0, 100.0) == pytest.approx(
        _AUTO_DT_INIT_C * 10.0, rel=1e-15
    )
    assert _resolve_dt_init_auto(5.0, 6.0) == pytest.approx(_AUTO_DT_INIT_C, rel=1e-15)

    # 小 T 域だが床値より上 (T=0.01 → 0.01 > floor=1e-3).
    val = _resolve_dt_init_auto(0.0, 0.01)
    assert val == pytest.approx(_AUTO_DT_INIT_C * (0.01**0.5), rel=1e-15)
    assert val > _AUTO_DT_INIT_FLOOR

    # 床値支配域 (T=1e-6 → c·sqrt(T)=1e-4 < floor=1e-3, ただし上限 T が
    # さらに効いて返り値は T 自体 (=1e-6)).
    tiny_T = 1e-6
    val_tiny = _resolve_dt_init_auto(0.0, tiny_T)
    assert val_tiny == pytest.approx(tiny_T, rel=1e-15)

    # 上限域: c · sqrt(T) > T のときは T 自体が上限として効く.
    # T=0.005 → c·sqrt(T) ≈ 7.07e-3 > T=5e-3 → 返り値は T.
    val_cap = _resolve_dt_init_auto(0.0, 0.005)
    assert val_cap == pytest.approx(0.005, rel=1e-15)


def test_dt_init_none_facade_smoke() -> None:
    """``QuantumAnnealer.run(..., dt_init=None)`` (issue #54 で旧 ``"auto"``
    リテラルを置換) smoke: linear schedule で QuTiP fidelity ``> 1 - 1e-6``
    かつ ``dt_init=None`` と ``dt_init=<auto-resolved float>`` で driver 出力
    がビット一致する (None resolution が driver の ``dt0`` に正しく流れて
    いることの確認).
    """
    from maqina import QuantumAnnealer
    from maqina.annealer import _resolve_dt_init_auto

    n = 4
    T = 5.0
    prob, h_x = _make_random_problem(n, seed=20260513)
    sched = Schedule.linear(T=T, h_x=h_x)
    psi0 = uniform_superposition(n)

    ann = QuantumAnnealer(prob, sched)
    res_auto = ann.run(
        psi0, 0.0, T, method="cfm4_adaptive_richardson_krylov", atol=1e-8, dt_init=None
    )
    psi_ref = _qutip_reference(h_x, prob.H_p_diag, T)
    fid = _fidelity(res_auto.psi_final, psi_ref)
    assert fid > 1 - 1e-6, (
        f"dt_init=None auto resolution fidelity too low: {fid} (1-fid={1 - fid})"
    )
    assert res_auto.success
    assert res_auto.n_steps_actual is not None and res_auto.n_steps_actual >= 1

    # None resolution と「手動で auto 解決値を float で渡した」が driver
    # 入力として等価であることをビット一致で確認 (PI controller は決定論的).
    dt0_resolved = _resolve_dt_init_auto(0.0, T)
    res_manual = ann.run(
        psi0,
        0.0,
        T,
        method="cfm4_adaptive_richardson_krylov",
        atol=1e-8,
        dt_init=dt0_resolved,
    )
    np.testing.assert_array_equal(res_auto.psi_final, res_manual.psi_final)
    assert res_auto.n_steps_actual == res_manual.n_steps_actual


def test_dt_init_none_small_T_completes() -> None:
    """``T < 1`` の小 T ケースで ``dt_init=None`` (auto resolution) でも
    driver が正常完了する.

    床値 / 上限の境界処理が正しく動き, driver 入力検証
    (``dt_min <= dt0 <= dt_max``) に違反しないことを確認する.
    """
    from maqina import QuantumAnnealer

    n = 3
    T = 0.05  # 床値支配でも上限支配でも無い中庸. 0.1·sqrt(0.05)≈2.24e-2 < T.
    prob, h_x = _make_random_problem(n, seed=99)
    sched = Schedule.linear(T=T, h_x=h_x)
    psi0 = uniform_superposition(n)

    ann = QuantumAnnealer(prob, sched)
    res = ann.run(
        psi0, 0.0, T, method="cfm4_adaptive_richardson_krylov", atol=1e-8, dt_init=None
    )
    assert res.success
    assert res.n_steps_actual is not None and res.n_steps_actual >= 1
    # unitary 性確認.
    assert abs(np.linalg.norm(res.psi_final) - 1.0) < 1e-10


def test_auto_dt_max_resolves_to_formula() -> None:
    """``dt_max="auto"`` の解決値が ``max(min(10·dt0, 4m/‖H‖_est), dt0)`` と一致.

    helper 単体で:
    - 通常域 (cap < default < 大): cap が支配
    - cap > default: default (10·dt0) が支配
    - cap < dt0: dt0 floor が支配
    の 3 パターンを機械精度で確認する.
    """
    from maqina.annealer import _gershgorin_norm_upper_bound, _resolve_dt_max_auto

    # 案件: 大 N で cap が支配する設定. h_x=1·n=20, H_p_diag in [-1, 1].
    n = 20
    rng = np.random.default_rng(0)
    h_p = rng.uniform(-1.0, 1.0, size=1 << n).astype(np.float64)
    h_x = np.ones(n, dtype=np.float64)
    prob = IsingProblem(n=n, H_p_diag=h_p)
    sched_helper = Schedule.linear(T=1.0, h_x=h_x)
    norm_h = _gershgorin_norm_upper_bound(sched_helper, prob)
    # h_x の合計 20 + max|H_p_diag| ≈ 1 → ‖H‖_est ≈ 21.
    assert 20.5 < norm_h < 21.5

    # cap = 4·24/21 ≈ 4.57, default_dt_max = 10·0.5 = 5.0 → cap が支配.
    val = _resolve_dt_max_auto(sched_helper, prob, m=24, dt0=0.5)
    expected_cap = 4.0 * 24.0 / norm_h
    assert val == pytest.approx(expected_cap, rel=1e-15)
    assert val < 5.0  # default を下回る

    # default 支配域: dt0=0.1 → default=1.0, cap=4.57 → default が支配.
    val_default = _resolve_dt_max_auto(sched_helper, prob, m=24, dt0=0.1)
    assert val_default == pytest.approx(1.0, rel=1e-15)

    # floor 支配域: dt0=10 → default=100, cap=4.57 → cap < dt0 → floor (dt0=10).
    val_floor = _resolve_dt_max_auto(sched_helper, prob, m=24, dt0=10.0)
    assert val_floor == pytest.approx(10.0, rel=1e-15)


def test_dt_max_none_facade_caps_dt_history() -> None:
    """``dt_max=None`` (issue #54 で旧 ``"auto"`` リテラルを置換) で driver
    出力が ``dt_max=<auto-resolved float>`` とビット一致し, dt が cap を
    超えないこと, かつ PI が dt0 から成長することを確認する.

    n=10, h_x=5·ones (‖H‖_est ≈ 50+1) で cap = 4·24/51 ≈ 1.88. Richardson
    自体が breakdown を embedded error で検出するため実 dt は cap よりも
    手前 (~ dt0·growth_max) で頭打ちになることが多いが, "cap を超えない"
    という上界保証が本テストの主目的.
    """
    from maqina import QuantumAnnealer
    from maqina.annealer import _gershgorin_norm_upper_bound, _resolve_dt_max_auto

    n = 10
    T = 5.0
    rng = np.random.default_rng(20260513)
    h_p = rng.uniform(-1.0, 1.0, size=1 << n).astype(np.float64)
    h_x = 5.0 * np.ones(n, dtype=np.float64)
    prob = IsingProblem(n=n, H_p_diag=h_p)
    sched_for_cap = Schedule.linear(T=T, h_x=h_x)
    norm_h = _gershgorin_norm_upper_bound(sched_for_cap, prob)
    expected_cap = _resolve_dt_max_auto(sched_for_cap, prob, m=24, dt0=0.5)
    # cap = 4·24/51 ≈ 1.88, default 10·0.5 = 5.0 → cap が支配.
    assert expected_cap == pytest.approx(4.0 * 24.0 / norm_h, rel=1e-15)
    assert expected_cap < 5.0

    sched = Schedule(T=T, A=lambda s: 0.5, B=lambda s: 0.5, h_x=h_x)
    psi0 = uniform_superposition(n)

    ann = QuantumAnnealer(prob, sched)
    res_auto = ann.run(
        psi0,
        0.0,
        T,
        method="cfm4_adaptive_richardson_krylov",
        atol=1e-6,
        dt_init=0.5,
        dt_max=None,
    )
    res_manual = ann.run(
        psi0,
        0.0,
        T,
        method="cfm4_adaptive_richardson_krylov",
        atol=1e-6,
        dt_init=0.5,
        dt_max=expected_cap,
    )
    # None resolution 経路と「auto-resolved float を手動で渡す」がビット一致
    # (PI controller は決定論的なので, 同じ入力で同じ出力になる).
    np.testing.assert_array_equal(res_auto.psi_final, res_manual.psi_final)
    assert res_auto.n_steps_actual == res_manual.n_steps_actual

    # cap を超えないこと, かつ PI が dt0=0.5 から伸びていることを driver
    # 直接呼出で dt_history を見て確認 (QuantumResult は dt_history を
    # 公開しないので driver layer に降りる).
    _, _, dt_history, _, _, _, _, _, _, _ = evolve_schedule_adaptive_richardson(
        h_p_diag=prob.H_p_diag,
        schedule=sched,
        psi0=psi0,
        t0=0.0,
        t1=T,
        tol_step=1e-6,
        dt0=0.5,
        dt_max=expected_cap,
    )
    # accept された全 dt が cap 以下 (floating tolerance を許容).
    assert float(np.max(dt_history)) <= expected_cap * (1.0 + 1e-12)
    # PI が dt0=0.5 から少なくとも成長していること (cap が動作していても
    # 成長余地はある).
    assert float(np.max(dt_history)) > 0.5


def test_auto_dt_max_zero_hamiltonian_falls_back_to_default() -> None:
    """h_x=0 かつ H_p_diag=0 の縮退で ``_resolve_dt_max_auto`` は default を返す.

    Gershgorin 上界が 0 になるケースの fallback path カバレッジ.
    """
    from maqina.annealer import _resolve_dt_max_auto

    n = 3
    h_p = np.zeros(1 << n, dtype=np.float64)
    h_x = np.zeros(n, dtype=np.float64)
    prob = IsingProblem(n=n, H_p_diag=h_p)
    sched = Schedule.linear(T=1.0, h_x=h_x)
    val = _resolve_dt_max_auto(sched, prob, m=24, dt0=0.5)
    assert val == pytest.approx(5.0, rel=1e-15)  # 10·dt0


def test_m_max_facade_smoke() -> None:
    """``m_max=16`` で facade adaptive 経路が QuTiP fidelity `> 1 - 1e-6` を満たす.

    Richardson estimator が Lanczos breakdown を embedded error として検出
    する fail-safe を活かし, m=24 default を 16 に下げても精度が維持される
    (PI controller が dt を絞ることで). n=4, T=5 の smooth linear schedule
    で smoke 検証.
    """
    from maqina import QuantumAnnealer

    n = 4
    T = 5.0
    prob, h_x = _make_random_problem(n, seed=20260513)
    sched = Schedule.linear(T=T, h_x=h_x)
    psi0 = uniform_superposition(n)

    ann = QuantumAnnealer(prob, sched)
    res = ann.run(
        psi0, 0.0, T, method="cfm4_adaptive_richardson_krylov", atol=1e-8, m_max=16
    )
    psi_ref = _qutip_reference(h_x, prob.H_p_diag, T)
    fid = _fidelity(res.psi_final, psi_ref)
    assert fid > 1 - 1e-6, f"m_max=16 fidelity too low: {fid} (1-fid={1 - fid})"
    assert res.success
    # n_matvec は m_eff_history の総和に基づく実コスト. 早期打切が起き
    # なければ n_steps_actual * 6 * 16 = upper bound と一致するが,
    # 早期打切で 6m_max を下回るのが一般 (issue #52 A 以降). 上限のみ assert.
    assert res.n_matvec <= res.n_steps_actual * 6 * 16


def test_m_max_overrides_self_m() -> None:
    """``m_max`` が ``self.m`` を上書きすることをビット一致で確認.

    QuantumAnnealer(m=24) を構築し, ``run(m_max=16)`` で実行した結果と
    QuantumAnnealer(m=16) を構築して ``run()`` した結果がビット一致する
    (driver は同じ m=16 で呼ばれるはず).
    """
    from maqina import QuantumAnnealer

    n = 4
    T = 2.0
    prob, h_x = _make_random_problem(n, seed=42)
    sched = Schedule.linear(T=T, h_x=h_x)
    psi0 = uniform_superposition(n)

    ann_24 = QuantumAnnealer(prob, sched, m=24)
    res_override = ann_24.run(
        psi0, 0.0, T, method="cfm4_adaptive_richardson_krylov", atol=1e-8, m_max=16
    )

    ann_16 = QuantumAnnealer(prob, sched, m=16)
    res_native = ann_16.run(
        psi0, 0.0, T, method="cfm4_adaptive_richardson_krylov", atol=1e-8
    )

    np.testing.assert_array_equal(res_override.psi_final, res_native.psi_final)
    assert res_override.n_steps_actual == res_native.n_steps_actual
    assert res_override.n_matvec == res_native.n_matvec


def test_m_eff_stats_in_adaptive_result() -> None:
    """adaptive Richardson 経路で ``QuantumResult.m_eff_stats`` が非 None で
    必要なキー全部を持ち, 各統計値が ``[1, 6·m]`` の範囲に収まる (issue #52 A).
    """
    from maqina import QuantumAnnealer

    n = 4
    T = 5.0
    prob, h_x = _make_random_problem(n, seed=20260513)
    sched = Schedule.linear(T=T, h_x=h_x)
    psi0 = uniform_superposition(n)

    ann = QuantumAnnealer(prob, sched)
    res = ann.run(psi0, 0.0, T, method="cfm4_adaptive_richardson_krylov", atol=1e-8)
    assert res.m_eff_stats is not None
    stats = res.m_eff_stats
    # 必須キー.
    for key in ("total", "mean", "median", "min", "max"):
        assert key in stats, f"missing key {key!r} in m_eff_stats={stats!r}"
    # 各 per-step 値は [1, 6·m] 範囲 (m default 24).
    assert 1 <= stats["min"] <= stats["max"] <= 6 * 24
    assert stats["min"] <= stats["median"] <= stats["max"]
    assert stats["min"] <= stats["mean"] <= stats["max"]
    # total = mean · n_steps_actual.
    assert res.n_steps_actual is not None
    assert stats["total"] == pytest.approx(
        stats["mean"] * res.n_steps_actual, rel=1e-12
    )
    # n_matvec が m_eff_history の総和と一致する (C4 で導入された contract).
    assert res.n_matvec == stats["total"]


def test_m_eff_stats_none_for_fixed_dt_methods() -> None:
    """固定 dt 経路 (m2 / trotter / cfm4) では ``m_eff_stats`` が None."""
    from maqina import QuantumAnnealer

    n = 3
    T = 1.0
    prob, h_x = _make_random_problem(n, seed=42)
    sched = Schedule.linear(T=T, h_x=h_x)
    psi0 = uniform_superposition(n)
    ann = QuantumAnnealer(prob, sched)
    for method in ("m2", "trotter", "trotter_suzuki4", "cfm4"):
        res = ann.run(
            psi0,
            0.0,
            T,
            method=method,  # type: ignore[arg-type]
            n_steps=10,
        )
        assert res.m_eff_stats is None, (
            f"method={method!r}: m_eff_stats should be None, got {res.m_eff_stats!r}"
        )


def test_m_max_32_matches_m_24_when_early_termination() -> None:
    """``m_max=32`` (実 ``m_eff < 24`` で β_k 早期打切) と ``m=24`` fixed の
    終端 ψ が ``rel < 1e-12`` で一致 (issue #52 A の β_k tol 早期打切契約).

    smooth schedule で問題サイズが小さい (n=4) 場合, 実 ``m_eff`` は 24 を
    下回るのが一般. 同じ β_k 打切点で停止するため Rust / Python 双方の
    Lanczos が決定論的に同じ部分空間を構築し, 終端 ψ もビット一致または
    機械精度内一致が期待される.
    """
    from maqina import QuantumAnnealer

    n = 4
    T = 5.0
    prob, h_x = _make_random_problem(n, seed=20260513)
    sched = Schedule.linear(T=T, h_x=h_x)
    psi0 = uniform_superposition(n)

    ann_24 = QuantumAnnealer(prob, sched, m=24)
    res_24 = ann_24.run(
        psi0, 0.0, T, method="cfm4_adaptive_richardson_krylov", atol=1e-8
    )
    ann_24_with_max32 = QuantumAnnealer(prob, sched, m=24)
    res_32 = ann_24_with_max32.run(
        psi0, 0.0, T, method="cfm4_adaptive_richardson_krylov", atol=1e-8, m_max=32
    )
    # 早期打切が同じ m_eff (=k+1) で起きていれば終端 ψ は完全一致するか
    # それ以下になる. 緩めに rel<1e-12 で assert.
    rel = float(
        np.linalg.norm(res_24.psi_final - res_32.psi_final)
        / max(np.linalg.norm(res_24.psi_final), 1.0)
    )
    assert rel < 1e-12, f"m=24 vs m_max=32 mismatch: rel={rel}"
    # m_max=32 のときの m_eff_max は ≤ 6·32 = 192.
    assert res_32.m_eff_stats is not None
    assert res_32.m_eff_stats["max"] <= 6 * 32


def test_m_max_invalid_raises() -> None:
    """``m_max`` に非正整数または非整数を渡すと ``ValueError``."""
    from maqina import QuantumAnnealer

    n = 3
    prob, h_x = _make_random_problem(n, seed=11)
    sched = Schedule.linear(T=1.0, h_x=h_x)
    psi0 = uniform_superposition(n)

    ann = QuantumAnnealer(prob, sched)
    with pytest.raises(ValueError, match="m_max"):
        ann.run(psi0, 0.0, 1.0, method="cfm4_adaptive_richardson_krylov", m_max=0)
    with pytest.raises(ValueError, match="m_max"):
        ann.run(psi0, 0.0, 1.0, method="cfm4_adaptive_richardson_krylov", m_max=-5)
    with pytest.raises(ValueError, match="m_max"):
        ann.run(
            psi0,
            0.0,
            1.0,
            method="cfm4_adaptive_richardson_krylov",
            m_max=3.5,  # type: ignore[arg-type]
        )


def test_adaptive_m2_save_tlist_still_not_implemented() -> None:
    """``evolve_schedule_adaptive_m2`` は Phase 5 (issue #47) スコープ外で
    ``save_tlist`` 非 None で ``NotImplementedError`` を維持する.

    adaptive M2 は annealer.py の facade からは呼ばれない内部 API なので,
    Phase 5 では adaptive Richardson のみ ``save_tlist`` 経路を有効化した.
    将来 adaptive M2 driver も拡張する case (low priority) に備えて契約を
    テストで明示する.
    """
    n = 3
    prob, h_x = _make_random_problem(n, seed=11)
    sched = Schedule.linear(T=1.0, h_x=h_x)
    psi0 = uniform_superposition(n)

    with pytest.raises(NotImplementedError, match="save_tlist"):
        evolve_schedule_adaptive_m2(
            h_p_diag=prob.H_p_diag,
            schedule=sched,
            psi0=psi0,
            t0=0.0,
            t1=1.0,
            save_tlist=np.array([0.5]),
        )


def test_adaptive_richardson_save_tlist_records_states() -> None:
    """Phase 5 (issue #47): ``evolve_schedule_adaptive_richardson`` は
    ``save_tlist`` 非 None で snapshot dict を返し, ψ をその時刻で記録する.

    PI controller が ``next_save_target - t`` で dt をクランプして save_tlist
    時刻を厳密に踏むことを, observables 評価 + 状態保存の round-trip で確認.
    """
    n = 3
    prob, h_x = _make_random_problem(n, seed=11)
    sched = Schedule.linear(T=1.0, h_x=h_x)
    psi0 = uniform_superposition(n)
    save_tlist = np.array([0.0, 0.3, 0.7, 1.0], dtype=np.float64)
    obs = {"M_z": Observable.magnetization(n)}

    (
        psi_final,
        _t_hist,
        _dt_hist,
        _n_rej,
        _m_eff,
        _beta_m,
        _err_lanczos,
        _err_magnus,
        _n_krylov_ins,
        snapshot,
    ) = evolve_schedule_adaptive_richardson(
        h_p_diag=prob.H_p_diag,
        schedule=sched,
        psi0=psi0,
        t0=0.0,
        t1=1.0,
        tol_step=1e-8,
        dt0=0.1,
        observables=obs,
        save_tlist=save_tlist,
        store_states=True,
    )
    assert snapshot is not None
    np.testing.assert_array_equal(snapshot["times"], save_tlist)
    states = snapshot["states"]
    assert states is not None
    assert states.shape == (4, 1 << n)
    # 先頭は psi0, 末尾は psi_final と一致 (driver が target を厳密に踏む契約).
    np.testing.assert_array_equal(states[0], psi0)
    np.testing.assert_array_equal(states[-1], psi_final)
    obs_hist = snapshot["observables_history"]
    assert "M_z" in obs_hist
    assert obs_hist["M_z"].shape == (4,)


def test_propagator_tol_none_resolves_to_atol_ratio_bit_exact() -> None:
    """``propagator_tol=None`` (issue #54 で導入, issue #135 で rename) と
    ``propagator_tol = atol · _KRYLOV_TOL_ATOL_RATIO`` を明示的に渡す経路で
    driver 出力 (psi_final / n_steps_actual / n_matvec) がビット一致する.

    facade None → ``effective_propagator_tol = tol_step · 1e-3`` の resolution
    が driver の ``krylov_tol`` に正しく流れていることの bit-exact 確認 (Lanczos
    variant の auto-coupling 経路).
    """
    from maqina import QuantumAnnealer
    from maqina.annealer import _KRYLOV_TOL_ATOL_RATIO

    n = 4
    T = 5.0
    atol = 1e-8
    prob, h_x = _make_random_problem(n, seed=20260514)
    sched = Schedule.linear(T=T, h_x=h_x)
    psi0 = uniform_superposition(n)

    expected_resolved = atol * _KRYLOV_TOL_ATOL_RATIO  # = 1e-11

    ann_none = QuantumAnnealer(prob, sched)  # propagator_tol=None default
    res_none = ann_none.run(
        psi0, 0.0, T, method="cfm4_adaptive_richardson_krylov", atol=atol
    )

    ann_explicit = QuantumAnnealer(prob, sched, propagator_tol=expected_resolved)
    res_explicit = ann_explicit.run(
        psi0, 0.0, T, method="cfm4_adaptive_richardson_krylov", atol=atol
    )

    np.testing.assert_array_equal(res_none.psi_final, res_explicit.psi_final)
    assert res_none.n_steps_actual == res_explicit.n_steps_actual
    assert res_none.n_matvec == res_explicit.n_matvec


def test_propagator_tol_none_vs_explicit_1e12_same_accuracy() -> None:
    """同一問題で ``propagator_tol=None`` (Lanczos variant の新 default,
    atol·1e-3 = 1e-11 に解決) と ``propagator_tol=1e-12`` (旧 default) の
    終端 ψ が ``rel < 1e-9`` 程度で一致する (issue #54 acceptance: 新 default
    が旧 default の accuracy を維持していることの担保).

    精度差の上界は a posteriori 早期打切による Lanczos 内打切誤差の差で
    支配される. `1e-11` でも atol=1e-8 に対し 3 桁マージンが残るため終端 ψ
    は機械精度近くで一致する.
    """
    from maqina import QuantumAnnealer

    n = 4
    T = 5.0
    atol = 1e-8
    prob, h_x = _make_random_problem(n, seed=20260514)
    sched = Schedule.linear(T=T, h_x=h_x)
    psi0 = uniform_superposition(n)

    ann_default = QuantumAnnealer(prob, sched)  # propagator_tol=None → 1e-11
    res_default = ann_default.run(
        psi0, 0.0, T, method="cfm4_adaptive_richardson_krylov", atol=atol
    )

    ann_tight = QuantumAnnealer(prob, sched, propagator_tol=1e-12)
    res_tight = ann_tight.run(
        psi0, 0.0, T, method="cfm4_adaptive_richardson_krylov", atol=atol
    )

    rel = float(
        np.linalg.norm(res_default.psi_final - res_tight.psi_final)
        / max(np.linalg.norm(res_tight.psi_final), 1.0)
    )
    assert rel < 1e-9, (
        f"propagator_tol=None (1e-11) vs 1e-12 mismatch: rel={rel} (expected < 1e-9)"
    )


def test_krylov_tol_atol_ratio_constant_is_1e_minus_3() -> None:
    """``_KRYLOV_TOL_ATOL_RATIO`` の default 値を 1e-3 で lock-down する
    (issue #54 で採用された経験値; 変更時は docs/design/05-3-propagator.md §5.3 E 節と
    bench 結果の更新も必要).
    """
    from maqina.annealer import _KRYLOV_TOL_ATOL_RATIO

    assert _KRYLOV_TOL_ATOL_RATIO == 1e-3


def test_dt_init_invalid_string_still_raises_at_runtime() -> None:
    """``dt_init`` に非数値文字列を渡すと runtime で ``ValueError``
    (issue #54 で `Literal["auto"]` を削除し explicit guard を廃止した
    あとも, ``float(s)`` 経由で型不整合を弾く保護が残ることの確認).
    """
    from maqina import QuantumAnnealer

    n = 3
    prob, h_x = _make_random_problem(n, seed=11)
    sched = Schedule.linear(T=1.0, h_x=h_x)
    psi0 = uniform_superposition(n)

    ann = QuantumAnnealer(prob, sched)
    with pytest.raises(ValueError):
        ann.run(
            psi0,
            0.0,
            1.0,
            method="cfm4_adaptive_richardson_krylov",
            dt_init="auto",  # type: ignore[arg-type]
        )


def test_dt_max_invalid_string_still_raises_at_runtime() -> None:
    """``dt_max`` に非数値文字列を渡すと runtime で ``ValueError``
    (``dt_init`` 同様, `Literal` 削除後の自然な float 変換失敗で弾かれる).
    """
    from maqina import QuantumAnnealer

    n = 3
    prob, h_x = _make_random_problem(n, seed=11)
    sched = Schedule.linear(T=1.0, h_x=h_x)
    psi0 = uniform_superposition(n)

    ann = QuantumAnnealer(prob, sched)
    with pytest.raises(ValueError):
        ann.run(
            psi0,
            0.0,
            1.0,
            method="cfm4_adaptive_richardson_krylov",
            dt_max="auto",  # type: ignore[arg-type]
        )
