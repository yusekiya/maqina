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

from kryanneal import IsingProblem, Schedule
from kryanneal.initial_states import uniform_superposition
from kryanneal.krylov import (
    evolve_schedule_adaptive_m2,
    evolve_schedule_adaptive_richardson,
)


qutip = pytest.importorskip("qutip")


def _make_random_problem(n: int, seed: int) -> IsingProblem:
    """ランダム ``H_p_diag`` (実数 ``[-1, 1]``) と一様 ``h_x = 1`` で
    ``IsingProblem`` を作る.
    """
    rng = np.random.default_rng(seed)
    dim = 1 << n
    h_p = rng.uniform(-1.0, 1.0, size=dim).astype(np.float64)
    h_x = np.ones(n, dtype=np.float64)
    return IsingProblem(n=n, H_p_diag=h_p, h_x=h_x)


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
    prob = _make_random_problem(n, seed=20260513)
    sched = Schedule.linear(T=T)
    psi0 = uniform_superposition(n)

    psi_final, t_history, dt_history, n_rejects = evolve_schedule_adaptive_m2(
        h_x=prob.h_x,
        h_p_diag=prob.H_p_diag,
        schedule=sched,
        psi0=psi0,
        t0=0.0,
        t1=T,
        tol_step=1e-8,
        dt0=0.1,
    )
    psi_ref = _qutip_reference(prob.h_x, prob.H_p_diag, T)
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
    prob = _make_random_problem(n, seed=20260513)
    sched = Schedule.linear(T=T)
    psi0 = uniform_superposition(n)

    # issue #52 A: driver は 5-tuple (psi, t_hist, dt_hist, n_rejects, m_eff_hist).
    psi_final, t_history, dt_history, n_rejects, m_eff_history = (
        evolve_schedule_adaptive_richardson(
            h_x=prob.h_x,
            h_p_diag=prob.H_p_diag,
            schedule=sched,
            psi0=psi0,
            t0=0.0,
            t1=T,
            tol_step=1e-8,
            dt0=0.1,
        )
    )
    psi_ref = _qutip_reference(prob.h_x, prob.H_p_diag, T)
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
    prob = _make_random_problem(n, seed=42)
    # constant schedule: A = 0.5, B = 0.5 で固定 (s(t) は default).
    sched = Schedule(T=T, A=lambda s: 0.5, B=lambda s: 0.5)
    psi0 = uniform_superposition(n)

    _, _, dt_history, _, _ = evolve_schedule_adaptive_richardson(
        h_x=prob.h_x,
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
    prob = _make_random_problem(n, seed=137)
    sched = Schedule.linear(T=T)
    psi0 = uniform_superposition(n)

    _, _, _, n_rejects, _ = evolve_schedule_adaptive_richardson(
        h_x=prob.h_x,
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
    prob = _make_random_problem(n, seed=11)
    sched = Schedule.linear(T=5.0)
    psi0 = uniform_superposition(n)

    with pytest.raises(RuntimeError, match="max_rejects"):
        evolve_schedule_adaptive_richardson(
            h_x=prob.h_x,
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
    from kryanneal.annealer import (
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


def test_auto_dt_init_facade_smoke() -> None:
    """``QuantumAnnealer.run(..., dt_init="auto")`` smoke: linear schedule で
    QuTiP との fidelity が ``> 1 - 1e-6``, かつ ``dt_init="auto"`` と
    ``dt_init=<auto-resolved float>`` で driver 出力がビット一致する
    (auto path が driver の ``dt0`` に正しく流れていることの確認).
    """
    from kryanneal import QuantumAnnealer
    from kryanneal.annealer import _resolve_dt_init_auto

    n = 4
    T = 5.0
    prob = _make_random_problem(n, seed=20260513)
    sched = Schedule.linear(T=T)
    psi0 = uniform_superposition(n)

    ann = QuantumAnnealer(prob, sched)
    res_auto = ann.run(
        psi0,
        0.0,
        T,
        method="cfm4_adaptive_richardson",
        atol=1e-8,
        dt_init="auto",
    )
    psi_ref = _qutip_reference(prob.h_x, prob.H_p_diag, T)
    fid = _fidelity(res_auto.psi_final, psi_ref)
    assert fid > 1 - 1e-6, f"auto dt_init fidelity too low: {fid} (1-fid={1 - fid})"
    assert res_auto.success
    assert res_auto.n_steps_actual is not None and res_auto.n_steps_actual >= 1

    # auto path と「手動で auto 解決値を float で渡した」が driver 入力として
    # 等価であることをビット一致で確認 (PI controller は決定論的なので
    # 同じ dt0 / tol_step / problem / psi0 で psi_final / n_steps_actual が
    # 厳密一致するはず).
    dt0_resolved = _resolve_dt_init_auto(0.0, T)
    res_manual = ann.run(
        psi0,
        0.0,
        T,
        method="cfm4_adaptive_richardson",
        atol=1e-8,
        dt_init=dt0_resolved,
    )
    np.testing.assert_array_equal(res_auto.psi_final, res_manual.psi_final)
    assert res_auto.n_steps_actual == res_manual.n_steps_actual


def test_auto_dt_init_small_T_completes() -> None:
    """``T < 1`` の小 T ケースで ``dt_init="auto"`` でも driver が正常完了する.

    床値 / 上限の境界処理が正しく動き, driver 入力検証
    (``dt_min <= dt0 <= dt_max``) に違反しないことを確認する.
    """
    from kryanneal import QuantumAnnealer

    n = 3
    T = 0.05  # 床値支配でも上限支配でも無い中庸. 0.1·sqrt(0.05)≈2.24e-2 < T.
    prob = _make_random_problem(n, seed=99)
    sched = Schedule.linear(T=T)
    psi0 = uniform_superposition(n)

    ann = QuantumAnnealer(prob, sched)
    res = ann.run(
        psi0,
        0.0,
        T,
        method="cfm4_adaptive_richardson",
        atol=1e-8,
        dt_init="auto",
    )
    assert res.success
    assert res.n_steps_actual is not None and res.n_steps_actual >= 1
    # unitary 性確認.
    assert abs(np.linalg.norm(res.psi_final) - 1.0) < 1e-10


def test_auto_dt_init_invalid_string_raises() -> None:
    """``dt_init`` に ``"auto"`` 以外の文字列を渡すと ``ValueError``.

    ``Literal["auto"]`` は型レベルでは弾けるが, runtime では ``isinstance(str)``
    + 明示比較で検証する (cv_ising 流 fail-fast)。
    """
    from kryanneal import QuantumAnnealer

    n = 3
    prob = _make_random_problem(n, seed=11)
    sched = Schedule.linear(T=1.0)
    psi0 = uniform_superposition(n)

    ann = QuantumAnnealer(prob, sched)
    with pytest.raises(ValueError, match="dt_init"):
        ann.run(
            psi0,
            0.0,
            1.0,
            method="cfm4_adaptive_richardson",
            dt_init="bogus",  # type: ignore[arg-type]
        )


def test_auto_dt_max_resolves_to_formula() -> None:
    """``dt_max="auto"`` の解決値が ``max(min(10·dt0, 4m/‖H‖_est), dt0)`` と一致.

    helper 単体で:
    - 通常域 (cap < default < 大): cap が支配
    - cap > default: default (10·dt0) が支配
    - cap < dt0: dt0 floor が支配
    の 3 パターンを機械精度で確認する.
    """
    from kryanneal.annealer import _gershgorin_norm_upper_bound, _resolve_dt_max_auto

    # 案件: 大 N で cap が支配する設定. h_x=1·n=20, H_p_diag in [-1, 1].
    n = 20
    rng = np.random.default_rng(0)
    h_p = rng.uniform(-1.0, 1.0, size=1 << n).astype(np.float64)
    h_x = np.ones(n, dtype=np.float64)
    prob = IsingProblem(n=n, H_p_diag=h_p, h_x=h_x)
    norm_h = _gershgorin_norm_upper_bound(prob)
    # h_x の合計 20 + max|H_p_diag| ≈ 1 → ‖H‖_est ≈ 21.
    assert 20.5 < norm_h < 21.5

    # cap = 4·24/21 ≈ 4.57, default_dt_max = 10·0.5 = 5.0 → cap が支配.
    val = _resolve_dt_max_auto(prob, m=24, dt0=0.5)
    expected_cap = 4.0 * 24.0 / norm_h
    assert val == pytest.approx(expected_cap, rel=1e-15)
    assert val < 5.0  # default を下回る

    # default 支配域: dt0=0.1 → default=1.0, cap=4.57 → default が支配.
    val_default = _resolve_dt_max_auto(prob, m=24, dt0=0.1)
    assert val_default == pytest.approx(1.0, rel=1e-15)

    # floor 支配域: dt0=10 → default=100, cap=4.57 → cap < dt0 → floor (dt0=10).
    val_floor = _resolve_dt_max_auto(prob, m=24, dt0=10.0)
    assert val_floor == pytest.approx(10.0, rel=1e-15)


def test_auto_dt_max_facade_caps_dt_history() -> None:
    """``dt_max="auto"`` で driver 出力が ``dt_max=<auto-resolved float>`` と
    ビット一致し, dt が cap を超えないこと, かつ PI が dt0 から成長する
    ことを確認する.

    n=10, h_x=5·ones (‖H‖_est ≈ 50+1) で cap = 4·24/51 ≈ 1.88. Richardson
    自体が breakdown を embedded error で検出するため実 dt は cap よりも
    手前 (~ dt0·growth_max) で頭打ちになることが多いが, "cap を超えない"
    という上界保証が本テストの主目的.
    """
    from kryanneal import QuantumAnnealer
    from kryanneal.annealer import _gershgorin_norm_upper_bound, _resolve_dt_max_auto

    n = 10
    T = 5.0
    rng = np.random.default_rng(20260513)
    h_p = rng.uniform(-1.0, 1.0, size=1 << n).astype(np.float64)
    h_x = 5.0 * np.ones(n, dtype=np.float64)
    prob = IsingProblem(n=n, H_p_diag=h_p, h_x=h_x)
    norm_h = _gershgorin_norm_upper_bound(prob)
    expected_cap = _resolve_dt_max_auto(prob, m=24, dt0=0.5)
    # cap = 4·24/51 ≈ 1.88, default 10·0.5 = 5.0 → cap が支配.
    assert expected_cap == pytest.approx(4.0 * 24.0 / norm_h, rel=1e-15)
    assert expected_cap < 5.0

    sched = Schedule(T=T, A=lambda s: 0.5, B=lambda s: 0.5)
    psi0 = uniform_superposition(n)

    ann = QuantumAnnealer(prob, sched)
    res_auto = ann.run(
        psi0,
        0.0,
        T,
        method="cfm4_adaptive_richardson",
        atol=1e-6,
        dt_init=0.5,
        dt_max="auto",
    )
    res_manual = ann.run(
        psi0,
        0.0,
        T,
        method="cfm4_adaptive_richardson",
        atol=1e-6,
        dt_init=0.5,
        dt_max=expected_cap,
    )
    # auto 経路と「auto-resolved float を手動で渡す」がビット一致
    # (PI controller は決定論的なので, 同じ入力で同じ出力になる).
    np.testing.assert_array_equal(res_auto.psi_final, res_manual.psi_final)
    assert res_auto.n_steps_actual == res_manual.n_steps_actual

    # cap を超えないこと, かつ PI が dt0=0.5 から伸びていることを driver
    # 直接呼出で dt_history を見て確認 (QuantumResult は dt_history を
    # 公開しないので driver layer に降りる).
    _, _, dt_history, _, _ = evolve_schedule_adaptive_richardson(
        h_x=prob.h_x,
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


def test_auto_dt_max_invalid_string_raises() -> None:
    """``dt_max`` に ``"auto"`` 以外の文字列を渡すと ``ValueError``."""
    from kryanneal import QuantumAnnealer

    n = 3
    prob = _make_random_problem(n, seed=11)
    sched = Schedule.linear(T=1.0)
    psi0 = uniform_superposition(n)

    ann = QuantumAnnealer(prob, sched)
    with pytest.raises(ValueError, match="dt_max"):
        ann.run(
            psi0,
            0.0,
            1.0,
            method="cfm4_adaptive_richardson",
            dt_max="bogus",  # type: ignore[arg-type]
        )


def test_auto_dt_max_zero_hamiltonian_falls_back_to_default() -> None:
    """h_x=0 かつ H_p_diag=0 の縮退で ``_resolve_dt_max_auto`` は default を返す.

    Gershgorin 上界が 0 になるケースの fallback path カバレッジ.
    """
    from kryanneal.annealer import _resolve_dt_max_auto

    n = 3
    h_p = np.zeros(1 << n, dtype=np.float64)
    h_x = np.zeros(n, dtype=np.float64)
    prob = IsingProblem(n=n, H_p_diag=h_p, h_x=h_x)
    val = _resolve_dt_max_auto(prob, m=24, dt0=0.5)
    assert val == pytest.approx(5.0, rel=1e-15)  # 10·dt0


def test_m_max_facade_smoke() -> None:
    """``m_max=16`` で facade adaptive 経路が QuTiP fidelity `> 1 - 1e-6` を満たす.

    Richardson estimator が Lanczos breakdown を embedded error として検出
    する fail-safe を活かし, m=24 default を 16 に下げても精度が維持される
    (PI controller が dt を絞ることで). n=4, T=5 の smooth linear schedule
    で smoke 検証.
    """
    from kryanneal import QuantumAnnealer

    n = 4
    T = 5.0
    prob = _make_random_problem(n, seed=20260513)
    sched = Schedule.linear(T=T)
    psi0 = uniform_superposition(n)

    ann = QuantumAnnealer(prob, sched)
    res = ann.run(
        psi0,
        0.0,
        T,
        method="cfm4_adaptive_richardson",
        atol=1e-8,
        m_max=16,
    )
    psi_ref = _qutip_reference(prob.h_x, prob.H_p_diag, T)
    fid = _fidelity(res.psi_final, psi_ref)
    assert fid > 1 - 1e-6, f"m_max=16 fidelity too low: {fid} (1-fid={1 - fid})"
    assert res.success
    # n_matvec が m_eff_param=16 ベースで計算されていること (per-step 6m).
    assert res.n_matvec == res.n_steps_actual * 6 * 16


def test_m_max_overrides_self_m() -> None:
    """``m_max`` が ``self.m`` を上書きすることをビット一致で確認.

    QuantumAnnealer(m=24) を構築し, ``run(m_max=16)`` で実行した結果と
    QuantumAnnealer(m=16) を構築して ``run()`` した結果がビット一致する
    (driver は同じ m=16 で呼ばれるはず).
    """
    from kryanneal import QuantumAnnealer

    n = 4
    T = 2.0
    prob = _make_random_problem(n, seed=42)
    sched = Schedule.linear(T=T)
    psi0 = uniform_superposition(n)

    ann_24 = QuantumAnnealer(prob, sched, m=24)
    res_override = ann_24.run(
        psi0,
        0.0,
        T,
        method="cfm4_adaptive_richardson",
        atol=1e-8,
        m_max=16,
    )

    ann_16 = QuantumAnnealer(prob, sched, m=16)
    res_native = ann_16.run(
        psi0,
        0.0,
        T,
        method="cfm4_adaptive_richardson",
        atol=1e-8,
    )

    np.testing.assert_array_equal(res_override.psi_final, res_native.psi_final)
    assert res_override.n_steps_actual == res_native.n_steps_actual
    assert res_override.n_matvec == res_native.n_matvec


def test_m_max_invalid_raises() -> None:
    """``m_max`` に非正整数または非整数を渡すと ``ValueError``."""
    from kryanneal import QuantumAnnealer

    n = 3
    prob = _make_random_problem(n, seed=11)
    sched = Schedule.linear(T=1.0)
    psi0 = uniform_superposition(n)

    ann = QuantumAnnealer(prob, sched)
    with pytest.raises(ValueError, match="m_max"):
        ann.run(
            psi0,
            0.0,
            1.0,
            method="cfm4_adaptive_richardson",
            m_max=0,
        )
    with pytest.raises(ValueError, match="m_max"):
        ann.run(
            psi0,
            0.0,
            1.0,
            method="cfm4_adaptive_richardson",
            m_max=-5,
        )
    with pytest.raises(ValueError, match="m_max"):
        ann.run(
            psi0,
            0.0,
            1.0,
            method="cfm4_adaptive_richardson",
            m_max=3.5,  # type: ignore[arg-type]
        )


def test_adaptive_save_tlist_not_implemented() -> None:
    """``save_tlist is not None`` で ``NotImplementedError`` (Phase 5)."""
    n = 3
    prob = _make_random_problem(n, seed=11)
    sched = Schedule.linear(T=1.0)
    psi0 = uniform_superposition(n)

    with pytest.raises(NotImplementedError, match="save_tlist"):
        evolve_schedule_adaptive_m2(
            h_x=prob.h_x,
            h_p_diag=prob.H_p_diag,
            schedule=sched,
            psi0=psi0,
            t0=0.0,
            t1=1.0,
            save_tlist=np.array([0.5]),
        )
    with pytest.raises(NotImplementedError, match="save_tlist"):
        evolve_schedule_adaptive_richardson(
            h_x=prob.h_x,
            h_p_diag=prob.H_p_diag,
            schedule=sched,
            psi0=psi0,
            t0=0.0,
            t1=1.0,
            save_tlist=np.array([0.5]),
        )
