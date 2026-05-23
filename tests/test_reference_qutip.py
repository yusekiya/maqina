"""QuTiP ``sesolve`` を高精度 ODE 参照とした end-to-end fidelity テスト.

issue #8 / #21 / #65 acceptance:

* 小規模 (n=4) で random ``H_p_diag`` / ``h_x`` / linear schedule に対し,
  ``QuantumAnnealer.run`` 各 method と QuTiP ``sesolve`` の終端状態
  fidelity を以下のしきい値で要求する:

  * ``method="m2"`` (n_steps=500): fidelity ``> 1 - 1e-6``.
  * ``method="trotter"`` (n_steps=500): fidelity ``> 1 - 1e-4``.
    Strang 2 次は M2 と同じ ``O(dt^3)`` LTE オーダだが, 係数 / 中点採取の
    対称性誤差で ``1e-6`` までは届かないので ``1e-4`` 設定 (issue #21).

* 大規模 (n=12-16, issue #65 Phase 6 C4) で QuTiP ``sesolve`` を ground truth
  として 4 method (m2 / trotter / cfm4 / cfm4_adaptive_richardson_krylov) と比較する.

  * 固定 dt (m2 / trotter / cfm4): ``n=12-14``, fidelity ``> 1 - 1e-6``.
  * adaptive (cfm4_adaptive_richardson_krylov): ``n=12-16``, fidelity ``> 1 - 1e-6``
    (atol=1e-8 で局所誤差を絞った設定).

  ``n >= 14`` は ``@pytest.mark.slow`` で除外可能 (CI fast loop 用).

QuTiP Hamiltonian は **常に sparse 構築** (``qutip.tensor`` of ``sigmax``) を使う.
QuTiP の Qobj は dense / CSR の data backing を持ち得るが, TFIM の構造では
``H_drv = -Σ_i h_x_i X_i`` の non-zero が ``n · 2^n`` しかなく, dense 表現
(``dim^2 = 2^{2n}`` 要素) を取る理由が全く無い. 全 n で sparse 構築すれば
``sesolve`` 内部の matvec が常に sparse 経路 (≈ ``(n+1)·dim`` flops/step) で
走り, 数値精度は dense 経路と完全一致のまま wall time を最小化できる
(dense backing の n=12 で 18.6s だったセルが sparse で <1s レベルになる).
QuTiP との比較で kinema だけが有利になる, という偏ったベンチを避ける
ための判断 (issue #65 review コメント).

QuTiP は dev 依存のみで本番 wheel には入れない契約
(``docs/design/08-qutip-comparison.md`` §8). 拡張未ビルド or QuTiP 未 install の
環境では ``pytest.importorskip`` で skip する.

ビット規約変換ノート (sparse 経路の根幹):

* kinema: bit 0 = LSB, ``x = Σ_i b_i · 2^i``. X_i は bit i を flip.
* QuTiP: ``qutip.tensor([A_0, A_1, ..., A_{n-1}])`` は ``np.kron`` 順に積み,
  最初の引数 ``A_0`` が **MSB 側** (最も "遅い" qubit) に対応する.
* したがって kinema の bit i に X を作用させるには
  ``qutip.tensor([I, ..., X, ..., I])`` の **位置 ``n-1-i``** に sx を入れる.
  これにより kinema の bit-flip pattern (``k ^ (1 << i)``) と数値一致する.
"""

from __future__ import annotations

import numpy as np
import pytest

from kinema import IsingProblem, QuantumAnnealer, Schedule
from kinema.initial_states import uniform_superposition


qutip = pytest.importorskip("qutip")


# ---------------------------------------------------------------------------
# Hamiltonian builder (常に sparse; モジュール docstring の判断根拠参照)
# ---------------------------------------------------------------------------


def _build_qutip_hamiltonian(h_x: np.ndarray, h_p_diag: np.ndarray, T: float) -> list:
    """QuTiP ``sesolve`` 用 ``H(t) = [[H_drv, A(t)], [H_p, B(t)]]`` を sparse で組む.

    linear schedule (``A(s) = 1 - s``, ``B(s) = s``, ``s = t/T``) を前提.
    ``H_drv = -Σ_i h_x[i] X_i`` を ``qutip.tensor`` (``np.kron``) ベースで
    sparse (CSR backing) 構築する. kinema の LSB bit 規約と QuTiP の
    MSB-first tensor 規約の差を吸収するため, X は tensor list の位置
    ``n-1-i`` に挿入する (モジュール docstring の規約変換ノート参照).

    ``H_p`` は対角なので ``qutip.qdiags`` で sparse 構築する.

    n=4 から n=16 まで同じ経路で構築する: dense backing にする利得が無く
    (TFIM の H_drv は non-zero が ``n · 2^n`` しかない), 比較対象として
    QuTiP を不利な dense matvec に走らせる理由がないため (issue #65 review
    コメントで dense backing を廃止).
    """
    n = h_x.shape[0]
    sx = qutip.sigmax()
    si = qutip.qeye(2)

    h_drv: object | None = None
    for i in range(n):
        ops = [si] * n
        ops[n - 1 - i] = sx  # LSB i in kinema → position n-1-i in QuTiP MSB-first
        term = -float(h_x[i]) * qutip.tensor(ops)
        h_drv = term if h_drv is None else h_drv + term

    # qutip.qdiags(diag, offset): offset=0 で対角 sparse Qobj. dims を
    # n 次 tensor product と整合させる ([[2]*n, [2]*n]).
    h_p = qutip.qdiags(h_p_diag, 0, dims=[[2] * n, [2] * n])
    return [
        [h_drv, f"(1 - t/{T})"],
        [h_p, f"(t/{T})"],
    ]


def _fidelity(psi_a: np.ndarray, psi_b: np.ndarray) -> float:
    """``|⟨ψ_a|ψ_b⟩|^2`` (normalize 済み state 前提)."""
    return float(np.abs(np.vdot(psi_a, psi_b)) ** 2)


def _qutip_sesolve_final(
    h_t: list,
    psi0: np.ndarray,
    T: float,
    n: int,
    *,
    atol: float = 1e-12,
    rtol: float = 1e-10,
) -> np.ndarray:
    """QuTiP ``sesolve`` を高精度設定で走らせて終端状態を取り出す.

    ``[0, T]`` 区間で 2 点 (端点のみ) を要求し, ``states[-1]`` を ndarray で
    返す. atol / rtol は default で十分高精度 (sparse 経路でも数 GB 級の
    intermediate state はメモリに乗らないため high-quality CVODE/LSODA 設定で
    まず ψ の連続性を担保する).

    ``n`` (スピン数) を必須引数で受けるのは, psi0 を tensor product dims
    (``[[2]*n, [1]*n]``) で構築するため. dense / sparse 経路ともに
    Hamiltonian 側の dims は ``[[2]*n, [2]*n]`` に統一されており, psi0 もこの
    tensor product dims と整合させないと QuTiP solver が
    ``TypeError: incompatible dimensions`` を投げる.
    """
    psi0_q = qutip.Qobj(psi0.reshape(-1, 1), dims=[[2] * n, [1] * n])
    sol = qutip.sesolve(
        h_t,
        psi0_q,
        np.array([0.0, T]),
        options={"atol": atol, "rtol": rtol, "nsteps": 100000},
    )
    return sol.states[-1].full().ravel()


# ---------------------------------------------------------------------------
# 小規模 (n=4) end-to-end tests (issue #8 / #21)
# ---------------------------------------------------------------------------


def test_quantum_annealer_matches_qutip_sesolve() -> None:
    """n=4 random H で QuTiP との fidelity > 1 - 1e-6 を要求する.

    seed 固定 (再現可能). ``n_steps=500`` で M2 中点則は十分小さい dt と
    なり, smooth schedule では Magnus M2 の LTE ~ O(dt^3) が支配する.
    """
    n = 4
    dim = 1 << n
    T = 5.0
    rng = np.random.default_rng(20251112)

    h_x = rng.uniform(0.5, 1.5, size=n).astype(np.float64)
    h_p_diag = rng.uniform(-1.0, 1.0, size=dim).astype(np.float64)

    prob = IsingProblem(n=n, H_p_diag=h_p_diag, h_x=h_x)
    sched = Schedule.linear(T=T)
    psi0 = uniform_superposition(n)

    ann = QuantumAnnealer(prob, sched)
    res = ann.run(psi0, 0.0, T, method="m2", n_steps=500)

    h_t = _build_qutip_hamiltonian(h_x, h_p_diag, T)
    psi_qutip = _qutip_sesolve_final(h_t, psi0, T, n)

    fid = _fidelity(res.psi_final, psi_qutip)
    assert fid > 1 - 1e-6, f"fidelity too low: {fid} (1 - fid = {1 - fid})"


def test_quantum_annealer_trotter_matches_qutip_sesolve() -> None:
    """``method="trotter"`` で QuTiP との fidelity ``> 1 - 1e-4`` (Phase 2 C3).

    M2 と同じ ``n=4`` random H / linear schedule / ``n_steps=500`` の
    設定で Strang 2 次 Trotter を走らせ, QuTiP との fidelity しきい値を
    issue #21 の規約に従って ``1 - 1e-4`` に設定する.
    """
    n = 4
    dim = 1 << n
    T = 5.0
    rng = np.random.default_rng(20251112)

    h_x = rng.uniform(0.5, 1.5, size=n).astype(np.float64)
    h_p_diag = rng.uniform(-1.0, 1.0, size=dim).astype(np.float64)

    prob = IsingProblem(n=n, H_p_diag=h_p_diag, h_x=h_x)
    sched = Schedule.linear(T=T)
    psi0 = uniform_superposition(n)

    ann = QuantumAnnealer(prob, sched)
    res = ann.run(psi0, 0.0, T, method="trotter", n_steps=500)

    h_t = _build_qutip_hamiltonian(h_x, h_p_diag, T)
    psi_qutip = _qutip_sesolve_final(h_t, psi0, T, n)

    fid = _fidelity(res.psi_final, psi_qutip)
    assert fid > 1 - 1e-4, f"fidelity too low (trotter): {fid} (1 - fid = {1 - fid})"


# ---------------------------------------------------------------------------
# 大規模 (n=12-16) end-to-end tests (issue #65 Phase 6 C4)
# ---------------------------------------------------------------------------


def _make_random_problem(
    n: int, seed: int
) -> tuple[IsingProblem, Schedule, np.ndarray, float]:
    """seed 固定の random ``IsingProblem`` を作る (n=12-16 共通).

    T=1.0 / linear schedule / ``|+⟩^N`` 始状態. ``h_x`` は
    ``Uniform(0.5, 1.5)``, ``H_p_diag`` は ``Uniform(-1, 1)``. 結果は
    ``(problem, schedule, psi0, T)``.
    """
    T = 1.0
    rng = np.random.default_rng(seed)
    h_x = rng.uniform(0.5, 1.5, size=n).astype(np.float64)
    h_p_diag = rng.uniform(-1.0, 1.0, size=1 << n).astype(np.float64)
    prob = IsingProblem(n=n, H_p_diag=h_p_diag, h_x=h_x)
    sched = Schedule.linear(T=T)
    psi0 = uniform_superposition(n)
    return prob, sched, psi0, T


# n=12, 13 は fast tier (slow marker 無し), n=14 以上は slow marker.
_LARGE_FIXED_DT_N: tuple[tuple[int, bool], ...] = (
    (12, False),
    (13, False),
    (14, True),  # slow
)

_LARGE_ADAPTIVE_N: tuple[tuple[int, bool], ...] = (
    (12, False),
    (13, False),
    (14, True),  # slow
    (15, True),  # slow
    (16, True),  # slow
)

# 固定 dt 経路の n_steps. T=1.0 で n_steps=200 → dt=5e-3. M2 / CFM4 の
# LTE ~ O(dt^3) ~ 1e-7 程度のオーダで 1 - 1e-6 fidelity をクリアできる
# (smooth linear schedule で経路 error が支配項にならない設定).
_LARGE_FIXED_N_STEPS: int = 200
_LARGE_FIDELITY_FIXED: float = 1.0 - 1e-6
_LARGE_FIDELITY_TROTTER: float = 1.0 - 1e-4
_LARGE_FIDELITY_ADAPTIVE: float = 1.0 - 1e-6


def _maybe_skip_slow_for_large_n(n: int, is_slow: bool) -> None:
    """``is_slow`` 指定がある n では ``pytest.mark.slow`` 経由で除外可能だが,
    動的 skip ではなく test 定義側で ``pytest.mark.slow`` を適用する形に統一
    したいので, ここでは何もしない (関数自体は parametrize id 用の placeholder)."""
    _ = n
    _ = is_slow


def _build_id(n: int, method: str, is_slow: bool) -> str:
    """parametrize id 用. ``n12-m2`` / ``n14-cfm4-slow`` のような形."""
    tag = f"n{n}-{method}"
    if is_slow:
        tag += "-slow"
    return tag


_FIXED_DT_METHODS: tuple[str, ...] = ("m2", "cfm4")


@pytest.mark.parametrize(
    ("n", "method"),
    [
        pytest.param(
            n,
            m,
            marks=[pytest.mark.slow] if is_slow else [],
            id=_build_id(n, m, is_slow),
        )
        for n, is_slow in _LARGE_FIXED_DT_N
        for m in _FIXED_DT_METHODS
    ],
)
def test_quantum_annealer_large_n_matches_qutip_fixed_dt(n: int, method: str) -> None:
    """n=12-14 の (m2, cfm4) を QuTiP sesolve と比較 (issue #65 Phase 6 C4).

    fidelity 閾値: ``1 - 1e-6`` (smooth linear schedule, n_steps=200).
    sample 入力は ``_make_random_problem`` で seed 固定.

    QuTiP Hamiltonian は ``_build_qutip_hamiltonian`` で常に sparse 構築
    (モジュール docstring の判断根拠参照).
    """
    prob, sched, psi0, T = _make_random_problem(n, seed=20260517 + n)

    ann = QuantumAnnealer(prob, sched)
    res = ann.run(
        psi0,
        0.0,
        T,
        method=method,  # type: ignore[arg-type]
        n_steps=_LARGE_FIXED_N_STEPS,
    )

    h_t = _build_qutip_hamiltonian(prob.h_x, prob.H_p_diag, T)
    psi_qutip = _qutip_sesolve_final(h_t, psi0, T, n)

    fid = _fidelity(res.psi_final, psi_qutip)
    assert fid > _LARGE_FIDELITY_FIXED, (
        f"fidelity too low (n={n}, method={method!r}): {fid} (1 - fid = {1 - fid})"
    )


@pytest.mark.parametrize(
    "n",
    [
        pytest.param(
            n,
            marks=[pytest.mark.slow] if is_slow else [],
            id=f"n{n}{'-slow' if is_slow else ''}",
        )
        for n, is_slow in _LARGE_FIXED_DT_N
    ],
)
def test_quantum_annealer_large_n_matches_qutip_trotter(n: int) -> None:
    """n=12-14 で ``method="trotter"`` を QuTiP sesolve と比較 (issue #65).

    Strang 2 次 Trotter は M2 と同じ ``O(dt^3)`` LTE オーダだが係数 / 中点
    採取の対称性で M2 より 2 桁 loose な閾値 (``1 - 1e-4``) を使う
    (issue #21 で確立した規約と整合).
    """
    prob, sched, psi0, T = _make_random_problem(n, seed=20260517 + n)

    ann = QuantumAnnealer(prob, sched)
    res = ann.run(psi0, 0.0, T, method="trotter", n_steps=_LARGE_FIXED_N_STEPS)

    h_t = _build_qutip_hamiltonian(prob.h_x, prob.H_p_diag, T)
    psi_qutip = _qutip_sesolve_final(h_t, psi0, T, n)

    fid = _fidelity(res.psi_final, psi_qutip)
    assert fid > _LARGE_FIDELITY_TROTTER, (
        f"fidelity too low (n={n}, method='trotter'): {fid} (1 - fid = {1 - fid})"
    )


@pytest.mark.parametrize(
    "n",
    [
        pytest.param(
            n,
            marks=[pytest.mark.slow] if is_slow else [],
            id=f"n{n}{'-slow' if is_slow else ''}",
        )
        for n, is_slow in _LARGE_ADAPTIVE_N
    ],
)
def test_quantum_annealer_large_n_matches_qutip_adaptive(n: int) -> None:
    """n=12-16 で ``method="cfm4_adaptive_richardson_krylov"`` を QuTiP sesolve と比較.

    PI controller atol=1e-8 で局所誤差を絞り, fidelity ``> 1 - 1e-6`` を
    要求する (issue #65 acceptance).

    adaptive 経路は dt 履歴が問題ごとに変わるが, atol=1e-8 は十分小さく
    PI が許容する最大 dt も Lanczos capacity (``4m / ‖H‖``) を超えないため
    安定して高精度. QuTiP Hamiltonian は ``_build_qutip_hamiltonian``
    で sparse 構築する (n=15-16 を含め全 n で同じ経路).
    """
    prob, sched, psi0, T = _make_random_problem(n, seed=20260517 + n)

    ann = QuantumAnnealer(prob, sched)
    res = ann.run(
        psi0,
        0.0,
        T,
        method="cfm4_adaptive_richardson_krylov",
        atol=1e-8,
    )

    h_t = _build_qutip_hamiltonian(prob.h_x, prob.H_p_diag, T)
    psi_qutip = _qutip_sesolve_final(h_t, psi0, T, n)

    fid = _fidelity(res.psi_final, psi_qutip)
    assert fid > _LARGE_FIDELITY_ADAPTIVE, (
        f"fidelity too low (n={n}, method='cfm4_adaptive_richardson_krylov'): "
        f"{fid} (1 - fid = {1 - fid})"
    )
