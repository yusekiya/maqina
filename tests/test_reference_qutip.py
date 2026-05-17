"""QuTiP ``sesolve`` を高精度 ODE 参照とした end-to-end fidelity テスト.

issue #8 / #21 / #65 acceptance:

* 小規模 (n=4) で random ``H_p_diag`` / ``h_x`` / linear schedule に対し,
  ``QuantumAnnealer.run`` 各 method と QuTiP ``sesolve`` の終端状態
  fidelity を以下のしきい値で要求する:

  * ``method="m2"`` (n_steps=500): fidelity ``> 1 - 1e-6``.
  * ``method="trotter"`` (n_steps=500): fidelity ``> 1 - 1e-4``.
    Strang 2 次は M2 と同じ ``O(dt^3)`` LTE オーダだが, 係数 / 中点採取の
    対称性誤差で ``1e-6`` までは届かないので ``1e-4`` 設定 (issue #21).

* 大規模 (n=12-16, issue #65 Phase 6 C4) で QuTiP ``sesolve`` を sparse
  Hamiltonian 経路 (dense 2^n × 2^n がメモリに乗らない n=15-16 領域) で
  ground truth として 4 method (m2 / trotter / cfm4 / cfm4_adaptive_richardson)
  と比較する.

  * 固定 dt (m2 / trotter / cfm4): ``n=12-14``, fidelity ``> 1 - 1e-6``.
  * adaptive (cfm4_adaptive_richardson): ``n=12-16``, fidelity ``> 1 - 1e-6``
    (atol=1e-8 で局所誤差を絞った設定).

  ``n >= 14`` は ``@pytest.mark.slow`` で除外可能 (CI fast loop 用).
  ``n=15-16`` は dense Hamiltonian がメモリに乗らないため必ず sparse 経路
  (``sigmax`` の tensor 和) で構築する.

QuTiP は dev 依存のみで本番 wheel には入れない契約
(``docs/design/08-qutip-comparison.md`` §8). 拡張未ビルド or QuTiP 未 install の
環境では ``pytest.importorskip`` で skip する.

ビット規約変換ノート (n=12+ の sparse 経路で重要):

* kryanneal: bit 0 = LSB, ``x = Σ_i b_i · 2^i``. X_i は bit i を flip.
* QuTiP: ``qutip.tensor([A_0, A_1, ..., A_{n-1}])`` は ``np.kron`` 順に積み,
  最初の引数 ``A_0`` が **MSB 側** (最も "遅い" qubit) に対応する.
* したがって kryanneal の bit i に X を作用させるには
  ``qutip.tensor([I, ..., X, ..., I])`` の **位置 ``n-1-i``** に sx を入れる.
  これにより既存 dense 経路 (``h_drv[k, k ^ (1 << i)] += -h_x[i]``) と数値
  一致する.
"""

from __future__ import annotations

import numpy as np
import pytest

from kryanneal import IsingProblem, QuantumAnnealer, Schedule
from kryanneal.initial_states import uniform_superposition


qutip = pytest.importorskip("qutip")


# ---------------------------------------------------------------------------
# Hamiltonian builders (dense / sparse)
# ---------------------------------------------------------------------------


def _build_qutip_hamiltonian(h_x: np.ndarray, h_p_diag: np.ndarray, T: float) -> list:
    """QuTiP ``sesolve`` 用 ``H(t) = [[H_drv, A(t)], [H_p, B(t)]]`` を **dense** で組む.

    linear schedule (``A(s) = 1 - s``, ``B(s) = s``, ``s = t/T``) を前提.
    ``H_drv``, ``H_p`` を dense ``Qobj`` として構築する. n が小さい
    (n <= ~10) ときに使う. n >= 12 では ``_build_qutip_hamiltonian_sparse``
    に切り替える (n=15 で dense は 16 GB / n=16 で 64 GB を要する).
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


def _build_qutip_hamiltonian_sparse(
    h_x: np.ndarray, h_p_diag: np.ndarray, T: float
) -> list:
    """QuTiP ``sesolve`` 用 ``H(t) = [[H_drv, A(t)], [H_p, B(t)]]`` を **sparse** で組む.

    ``H_drv = -Σ_i h_x[i] X_i`` を ``qutip.tensor`` (``np.kron``) ベースで
    sparse 構築. kryanneal の LSB bit 規約と QuTiP の MSB-first tensor 規約
    の差を吸収するため, X は tensor list の位置 ``n-1-i`` に挿入する
    (モジュール docstring の規約変換ノート参照).

    ``H_p`` は対角なので ``qutip.qdiags`` で sparse 構築する.

    n >= 12 の large 比較 (issue #65) で使う. n=16 (dim=65536) でも
    sparse non-zero は ~1M 程度なので memory に余裕がある.
    """
    n = h_x.shape[0]
    sx = qutip.sigmax()
    si = qutip.qeye(2)

    h_drv: object | None = None
    for i in range(n):
        ops = [si] * n
        ops[n - 1 - i] = sx  # LSB i in kryanneal → position n-1-i in QuTiP MSB-first
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
    h_t: list, psi0: np.ndarray, T: float, *, atol: float = 1e-12, rtol: float = 1e-10
) -> np.ndarray:
    """QuTiP ``sesolve`` を高精度設定で走らせて終端状態を取り出す.

    ``[0, T]`` 区間で 2 点 (端点のみ) を要求し, ``states[-1]`` を ndarray で
    返す. atol / rtol は default で十分高精度 (sparse 経路でも数 GB 級の
    intermediate state はメモリに乗らないため high-quality CVODE/LSODA 設定で
    まず ψ の連続性を担保する).
    """
    psi0_q = qutip.Qobj(psi0.reshape(-1, 1))
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
    psi_qutip = _qutip_sesolve_final(h_t, psi0, T)

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
    psi_qutip = _qutip_sesolve_final(h_t, psi0, T)

    fid = _fidelity(res.psi_final, psi_qutip)
    assert fid > 1 - 1e-4, f"fidelity too low (trotter): {fid} (1 - fid = {1 - fid})"


# ---------------------------------------------------------------------------
# 大規模 (n=12-16) end-to-end tests (issue #65 Phase 6 C4)
# ---------------------------------------------------------------------------

# n=12 / 13 は dense でも数十 MB 級 (n=13 で 128 MB) なので dense 経路で OK.
# n=14 (256 MB) も dense に乗るが時間がかかる. n=15 (16 GB) / 16 (64 GB) は
# dense 不可なので必ず sparse 経路.
_DENSE_THRESHOLD_N: int = 13


def _qutip_hamiltonian_auto(h_x: np.ndarray, h_p_diag: np.ndarray, T: float) -> list:
    """n に応じて dense / sparse 構築を自動切り替えする helper.

    ``n <= _DENSE_THRESHOLD_N`` で dense, それ以上で sparse. dense / sparse
    どちらも数値的に同じ Hamiltonian を表すので fidelity 比較結果は
    一致する (sparse の inner solver も dense と同じ CVODE/LSODA を使う).
    """
    n = h_x.shape[0]
    if n <= _DENSE_THRESHOLD_N:
        return _build_qutip_hamiltonian(h_x, h_p_diag, T)
    return _build_qutip_hamiltonian_sparse(h_x, h_p_diag, T)


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
    (15, True),  # slow + sparse-only
    (16, True),  # slow + sparse-only
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

    n=12 / 13 は dense Hamiltonian で QuTiP に渡し, n=14 は dense 256 MB
    級なので sparse 経路に切り替える (auto helper ``_qutip_hamiltonian_auto``).
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

    h_t = _qutip_hamiltonian_auto(prob.h_x, prob.H_p_diag, T)
    psi_qutip = _qutip_sesolve_final(h_t, psi0, T)

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

    h_t = _qutip_hamiltonian_auto(prob.h_x, prob.H_p_diag, T)
    psi_qutip = _qutip_sesolve_final(h_t, psi0, T)

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
    """n=12-16 で ``method="cfm4_adaptive_richardson"`` を QuTiP sesolve と比較.

    PI controller atol=1e-8 で局所誤差を絞り, fidelity ``> 1 - 1e-6`` を
    要求する (issue #65 acceptance).

    n=15-16 では dense 2^n × 2^n が GB 級になるため sparse 経路を使う
    (auto helper). adaptive 経路は dt 履歴が問題ごとに変わるが, atol=1e-8
    は十分小さく PI が許容する最大 dt も Lanczos capacity (``4m / ‖H‖``)
    を超えないため安定して高精度. 実行時間は QuTiP sesolve 側が支配 (大 n
    で minutes オーダ).
    """
    prob, sched, psi0, T = _make_random_problem(n, seed=20260517 + n)

    ann = QuantumAnnealer(prob, sched)
    res = ann.run(
        psi0,
        0.0,
        T,
        method="cfm4_adaptive_richardson",
        atol=1e-8,
    )

    h_t = _qutip_hamiltonian_auto(prob.h_x, prob.H_p_diag, T)
    psi_qutip = _qutip_sesolve_final(h_t, psi0, T)

    fid = _fidelity(res.psi_final, psi_qutip)
    assert fid > _LARGE_FIDELITY_ADAPTIVE, (
        f"fidelity too low (n={n}, method='cfm4_adaptive_richardson'): "
        f"{fid} (1 - fid = {1 - fid})"
    )
