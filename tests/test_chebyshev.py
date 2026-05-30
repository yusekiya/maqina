"""Chebyshev propagator + CFM4:2 + adaptive Richardson 統合 (issue #122 Phase B).

主な acceptance:

* ``evolve_schedule_adaptive_richardson_chebyshev`` smoke: linear schedule で
  QuTiP との fidelity が ``> 1 - 1e-6``.
* Lanczos 経路 (``evolve_schedule_adaptive_richardson``) との結果一致が
  ``tol_step`` 程度に収まる (``rel < 1e-7``; PI controller の atol margin).
* ``QuantumAnnealer.run(method="cfm4_adaptive_richardson_chebyshev")`` の
  end-to-end smoke (QuTiP fidelity + Lanczos との一致).
* ``AnnealingSimulator(method=...)`` 経路の smoke.
* ``m_max`` を Chebyshev method で渡すと ``ValueError`` (semantic 不一致を
  silent 無視しない契約).

Rust 拡張 (``maqina._rust.cfm4_step_chebyshev_with_richardson_estimate_py``)
が必須. fallback path は提供しない設計のため, 拡張が無い環境では本テスト
ファイルは ``importorskip`` で skip.
"""

from __future__ import annotations

import numpy as np
import pytest

from maqina import IsingProblem, Schedule
from maqina.annealer import QuantumAnnealer
from maqina.initial_states import uniform_superposition
from maqina.krylov import (
    evolve_schedule_adaptive_richardson,
    evolve_schedule_adaptive_richardson_chebyshev,
)
from maqina.simulator import AnnealingSimulator


qutip = pytest.importorskip("qutip")

# Rust 拡張が無いと Chebyshev driver は NotImplementedError を上げる. 本ファイル
# 全体を skip.
try:
    from maqina import _rust as _rust_mod  # noqa: F401
except ImportError:  # pragma: no cover - 拡張なし環境
    pytest.skip(
        "maqina._rust extension required for Chebyshev tests", allow_module_level=True
    )


def _make_random_problem(n: int, seed: int) -> tuple[IsingProblem, np.ndarray]:
    rng = np.random.default_rng(seed)
    dim = 1 << n
    h_p = rng.uniform(-1.0, 1.0, size=dim).astype(np.float64)
    h_x = np.ones(n, dtype=np.float64)
    return IsingProblem(n=n, H_p_diag=h_p), h_x


def _build_qutip_hamiltonian(h_x: np.ndarray, h_p_diag: np.ndarray, T: float) -> list:
    """linear schedule の QuTiP ``H(t)`` を組む (test_adaptive.py と同型)."""
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


def test_adaptive_chebyshev_matches_qutip() -> None:
    """``evolve_schedule_adaptive_richardson_chebyshev`` smoke: linear schedule
    の n=4 問題で QuTiP との fidelity が ``> 1 - 1e-6``.
    """
    n = 4
    T = 5.0
    prob, h_x = _make_random_problem(n, seed=20260522)
    sched = Schedule.linear(T=T, h_x=h_x)
    psi0 = uniform_superposition(n)

    (
        psi_final,
        _t_hist,
        dt_hist,
        n_rejects,
        k_used_hist,
        _beta_m_hist,
        err_cheb_hist,
        _err_magnus_hist,
        n_cheb_insufficient,
        _snapshot,
    ) = evolve_schedule_adaptive_richardson_chebyshev(
        h_p_diag=prob.H_p_diag,
        schedule=sched,
        psi0=psi0,
        t0=0.0,
        t1=T,
        h_p_min=prob.h_p_diag_min,
        h_p_max=prob.h_p_diag_max,
        tol_step=1e-8,
        dt0=0.5,
    )

    expected = _qutip_reference(h_x, prob.H_p_diag, T)
    fid = _fidelity(psi_final, expected)
    assert fid > 1.0 - 1e-6, (
        f"adaptive_chebyshev vs QuTiP fidelity = {fid} (< 1 - 1e-6); "
        f"n_steps={dt_hist.size}, n_rejects={n_rejects}, "
        f"k_used_mean={float(np.mean(k_used_hist)) if k_used_hist.size > 0 else 0}, "
        f"n_cheb_insufficient={n_cheb_insufficient}"
    )
    assert dt_hist.size > 0, "driver should produce >=1 accept step"
    assert k_used_hist.size == dt_hist.size, "k_used history must match accepts"
    assert err_cheb_hist.size == dt_hist.size, (
        "err_chebyshev history must match accepts"
    )
    # K_used は z = R·dt に応じて動的に決まる; tol=1e-11 規模なら K ~ 10-40 程度.
    assert int(np.max(k_used_hist)) >= 1, "K_used should be >= 1 for each step"


def test_adaptive_chebyshev_matches_lanczos_rel_small() -> None:
    """同じ tol_step / dt0 で Lanczos 経路と Chebyshev 経路が ``rel < 1e-7``
    で一致することを確認.

    両者は同じ CFM4:2 Magnus + step-doubling Richardson 構造を持ち, 短時間
    プロパゲータの実装だけが異なる (Lanczos vs Chebyshev 3 項漸化). 同じ
    tol_step なら PI controller が同じ精度水準で dt を選ぶため, 終端 ψ も
    tol_step 程度の精度で一致する.
    """
    n = 4
    T = 3.0
    prob, h_x = _make_random_problem(n, seed=20260523)
    sched = Schedule.linear(T=T, h_x=h_x)
    psi0 = uniform_superposition(n)

    psi_lan = evolve_schedule_adaptive_richardson(
        h_p_diag=prob.H_p_diag,
        schedule=sched,
        psi0=psi0,
        t0=0.0,
        t1=T,
        tol_step=1e-8,
        dt0=0.5,
    )[0]
    psi_che = evolve_schedule_adaptive_richardson_chebyshev(
        h_p_diag=prob.H_p_diag,
        schedule=sched,
        psi0=psi0,
        t0=0.0,
        t1=T,
        h_p_min=prob.h_p_diag_min,
        h_p_max=prob.h_p_diag_max,
        tol_step=1e-8,
        dt0=0.5,
    )[0]

    rel = float(np.linalg.norm(psi_che - psi_lan))
    assert rel < 1e-7, f"Chebyshev vs Lanczos rel = {rel} (should be ≤ tol_step margin)"


def test_chebyshev_h_p_bounds_match_h_p_diag_min_max() -> None:
    """``Schedule.from_xyz`` で時間依存 Z 磁場なしの設定 (g_z=None) を組み,
    Chebyshev driver に **正しい** ``h_p_min/h_p_max`` を渡したときと, 故意に
    **緩めた** 境界 (例えば `-10·|H_p|, +10·|H_p|`) を渡したときで, 短時間
    プロパゲータの数値結果が一致することを確認する.

    Chebyshev の Gershgorin 上下界は **スペクトル中心 E_c と半径 R** を決める
    入力でしかなく, R を上方にスケールしても K_used が増えるだけで切り捨て
    残差は ``chebyshev_tol`` 以下に保たれる. したがって正しい bound と緩い
    bound で終端 ψ は ``chebyshev_tol`` 程度で一致しなければならない.

    issue #142 PR #146 follow-up: ``compute_stage_gershgorin`` を precompute
    h_p_min/max 経由で算出する変更が, Gershgorin 自体の意味的役割を壊して
    いないことの契約.
    """
    n = 4
    T = 2.0
    prob, h_x = _make_random_problem(n, seed=20260530)
    sched = Schedule.linear(T=T, h_x=h_x)
    psi0 = uniform_superposition(n)

    # 正しい bound (= IsingProblem の precompute).
    psi_tight = evolve_schedule_adaptive_richardson_chebyshev(
        h_p_diag=prob.H_p_diag,
        schedule=sched,
        psi0=psi0,
        t0=0.0,
        t1=T,
        h_p_min=prob.h_p_diag_min,
        h_p_max=prob.h_p_diag_max,
        tol_step=1e-8,
        dt0=0.5,
    )[0]

    # 故意に緩めた bound (10× spread). chebyshev_tol で許容される範囲で
    # K_used が増えるだけで, 終端 ψ は一致するはず.
    spread = 10.0 * max(abs(prob.h_p_diag_min), abs(prob.h_p_diag_max), 1.0)
    psi_loose = evolve_schedule_adaptive_richardson_chebyshev(
        h_p_diag=prob.H_p_diag,
        schedule=sched,
        psi0=psi0,
        t0=0.0,
        t1=T,
        h_p_min=-spread,
        h_p_max=spread,
        tol_step=1e-8,
        dt0=0.5,
    )[0]

    rel = float(np.linalg.norm(psi_tight - psi_loose))
    assert rel < 1e-7, (
        f"tight vs loose Gershgorin bound rel = {rel} "
        "(should be within chebyshev_tol margin; Chebyshev bound 緩めても "
        "K_used が増えるだけで終端 ψ は一致するはず)"
    )


def test_annealer_chebyshev_smoke() -> None:
    """``QuantumAnnealer.run(method="cfm4_adaptive_richardson_chebyshev")``
    の end-to-end smoke.

    QuTiP fidelity > 1 - 1e-6 と Lanczos 経路との一致を ``rel < 1e-7`` で
    両方確認する.
    """
    n = 4
    T = 4.0
    prob, h_x = _make_random_problem(n, seed=20260524)
    sched = Schedule.linear(T=T, h_x=h_x)
    psi0 = uniform_superposition(n)

    annealer = QuantumAnnealer(prob, sched)
    result_che = annealer.run(
        psi0, t0=0.0, t1=T, method="cfm4_adaptive_richardson_chebyshev"
    )
    result_lan = annealer.run(
        psi0, t0=0.0, t1=T, method="cfm4_adaptive_richardson_krylov"
    )

    expected = _qutip_reference(h_x, prob.H_p_diag, T)
    fid_che = _fidelity(result_che.psi_final, expected)
    fid_lan = _fidelity(result_lan.psi_final, expected)
    assert fid_che > 1.0 - 1e-6, (
        f"annealer Chebyshev vs QuTiP fidelity = {fid_che}; "
        f"reference Lanczos fidelity = {fid_lan}"
    )

    rel = float(np.linalg.norm(result_che.psi_final - result_lan.psi_final))
    assert rel < 1e-7, f"annealer Chebyshev vs Lanczos rel = {rel}"

    # Chebyshev では k_used_history を m_eff_stats スロットに格納 (用途同じ).
    assert result_che.method == "cfm4_adaptive_richardson_chebyshev"
    assert result_che.m_eff_stats is not None
    assert "total" in result_che.m_eff_stats
    assert result_che.beta_m_stats is None  # Chebyshev に β_m は無い.
    # n_matvec は K_used 合計から推定.
    assert result_che.n_matvec > 0


def test_simulator_chebyshev_smoke() -> None:
    """``AnnealingSimulator(method="cfm4_adaptive_richardson_chebyshev")``
    の smoke. ``advance_to`` で QuTiP fidelity が出る.
    """
    n = 4
    T = 3.0
    prob, h_x = _make_random_problem(n, seed=20260525)
    sched = Schedule.linear(T=T, h_x=h_x)
    psi0 = uniform_superposition(n)

    sim = AnnealingSimulator(
        prob, sched, psi0, 0.0, method="cfm4_adaptive_richardson_chebyshev"
    )
    sim.advance_to(T)
    expected = _qutip_reference(h_x, prob.H_p_diag, T)
    fid = _fidelity(sim.psi, expected)
    assert fid > 1.0 - 1e-6, f"simulator Chebyshev vs QuTiP fidelity = {fid}"
    assert sim.method == "cfm4_adaptive_richardson_chebyshev"
    assert sim.n_matvec > 0


def test_chebyshev_rejects_m_max_param() -> None:
    """``m_max`` を Chebyshev method で渡すと ``ValueError``.

    Chebyshev は K_used を動的決定するため Krylov 部分空間次元の概念が無く,
    ``m_max`` は意味的に不適合. silent 無視は debug 罠なので明示的に弾く契約.
    """
    n = 4
    T = 1.0
    prob, h_x = _make_random_problem(n, seed=20260526)
    sched = Schedule.linear(T=T, h_x=h_x)
    psi0 = uniform_superposition(n)

    annealer = QuantumAnnealer(prob, sched)
    with pytest.raises(ValueError, match="m_max is not supported"):
        annealer.run(
            psi0, t0=0.0, t1=T, method="cfm4_adaptive_richardson_chebyshev", m_max=16
        )

    # AnnealingSimulator も同様.
    with pytest.raises(ValueError, match="m_max is not supported"):
        AnnealingSimulator(
            prob,
            sched,
            psi0,
            0.0,
            method="cfm4_adaptive_richardson_chebyshev",
            m_max=16,
        )


def test_chebyshev_default_propagator_tol_is_fixed_1e_minus_12() -> None:
    """issue #135: Chebyshev variant の ``propagator_tol`` default は
    ``_KRYLOV_TOL_FIXED_DEFAULT`` (= 1e-12) 固定.

    atol を 2 桁振っても (1e-6 vs 1e-8) 同じ問題に対する平均 K_used が
    僅かしか変わらないことで, propagator_tol が atol に連動していない
    (auto-coupling されていない) ことを確認する. Lanczos variant とは
    挙動が異なる軸 (Lanczos は atol scaling 連動).
    """
    from maqina.annealer import _KRYLOV_TOL_FIXED_DEFAULT

    n = 5
    T = 5.0
    prob, h_x = _make_random_problem(n, seed=20260526)
    sched = Schedule.linear(T=T, h_x=h_x)
    psi0 = uniform_superposition(n)

    # propagator_tol=None default で atol を 2 桁振る.
    k_used_means: list[float] = []
    for atol in (1e-6, 1e-8):
        ann = QuantumAnnealer(prob, sched)  # propagator_tol=None default
        result = ann.run(
            psi0, t0=0.0, t1=T, method="cfm4_adaptive_richardson_chebyshev", atol=atol
        )
        assert result.m_eff_stats is not None
        k_used_means.append(float(result.m_eff_stats["mean"]))

    # propagator_tol が atol 非連動 (固定 1e-12) なので, atol を変えても
    # K_used 平均は per-step matvec 数 (= K_used) で ~10% 以内の変動に留まる
    # (実測の上限見積もり; auto-coupling だと数倍動く).
    k_mean_a, k_mean_b = k_used_means
    rel_diff = abs(k_mean_a - k_mean_b) / max(k_mean_a, k_mean_b)
    assert rel_diff < 0.20, (
        f"propagator_tol が固定 1e-12 のはずだが atol 変動で K_used 平均が "
        f"{rel_diff:.2%} 変動: atol=1e-6 mean={k_mean_a:.2f}, "
        f"atol=1e-8 mean={k_mean_b:.2f}"
    )

    # 加えて, 明示的に propagator_tol=1e-12 を渡しても default と同等の結果.
    ann_default = QuantumAnnealer(prob, sched)
    ann_explicit = QuantumAnnealer(
        prob, sched, propagator_tol=_KRYLOV_TOL_FIXED_DEFAULT
    )
    res_default = ann_default.run(
        psi0, t0=0.0, t1=T, method="cfm4_adaptive_richardson_chebyshev", atol=1e-8
    )
    res_explicit = ann_explicit.run(
        psi0, t0=0.0, t1=T, method="cfm4_adaptive_richardson_chebyshev", atol=1e-8
    )
    np.testing.assert_array_equal(res_default.psi_final, res_explicit.psi_final)


def test_old_krylov_tol_kwarg_raises_typeerror() -> None:
    """issue #135 (API 破壊変更): 旧 ``krylov_tol`` kwarg は受け付けない
    (deprecation alias は残さない方針). 旧コードを誤って動かさないことを
    保証するため ``TypeError`` を contract とする.
    """
    n = 3
    prob, h_x = _make_random_problem(n, seed=20260526)
    sched = Schedule.linear(T=1.0, h_x=h_x)
    psi0 = uniform_superposition(n)

    with pytest.raises(TypeError):
        QuantumAnnealer(prob, sched, krylov_tol=1e-12)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        AnnealingSimulator(prob, sched, psi0, 0.0, krylov_tol=1e-12)  # type: ignore[call-arg]
