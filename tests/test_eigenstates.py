"""``instantaneous_eigenstates`` の Lanczos / exact 経路一致テスト.

Phase 5 C4 (issue #49) の acceptance:

* ``method="lanczos"`` と ``method="exact"`` が小規模 ``n`` (3 / 4) で
  ``rel < 1e-10`` 一致する (eigvals + |eigvec overlap| ≈ 1).
* スケジュール端点で物理的に期待される値を取る.
* 入力検証 (``k > m``, ``n > 12`` exact, ``method`` 不正).
* ``seed`` 固定で再現性.
"""

from __future__ import annotations

import numpy as np
import pytest

from maqina import IsingProblem, Schedule, instantaneous_eigenstates
from maqina import _rust as _rust_mod


def _make_problem(n: int, seed: int = 0) -> tuple[IsingProblem, np.ndarray]:
    """``IsingProblem`` を作るヘルパ. ``H_p_diag`` は乱数で縮退を避ける.

    ``h_x`` は incommensurate (``± h_x_i`` の和に 0 が現れない) になるよう
    無理数倍数の組を使う. これで driver 単独 (``t=0``) の固有値も非縮退.
    """
    rng = np.random.default_rng(seed)
    h_p = rng.normal(size=1 << n).astype(np.float64)
    # incommensurate な h_x: 1, √2, √3, √5 (n <= 4 まで).
    base = np.array([1.0, np.sqrt(2.0), np.sqrt(3.0), np.sqrt(5.0)], dtype=np.float64)
    h_x = base[:n].copy()
    return IsingProblem(n=n, H_p_diag=h_p), h_x


def _apply_ht(
    prob: IsingProblem, sched: Schedule, t: float, v: np.ndarray
) -> np.ndarray:
    """``H(t) @ v`` を Rust matvec 経由で計算する (テスト残差用)."""
    a_t, b_t = sched.coeffs_at(t)
    v_c = np.ascontiguousarray(v)
    return _rust_mod.apply_h_py(v_c, sched.h_x, prob.H_p_diag, a_t, b_t)


def _eigvec_residuals(
    prob: IsingProblem,
    sched: Schedule,
    t: float,
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
) -> np.ndarray:
    """各列 ``j`` について ``||H(t) v_j - λ_j v_j|| / max(|λ_j|, 1)`` を返す.

    縮退があっても固有ベクトル成分は正しい (固有値方程式を満たす) はずなので,
    残差で判定すれば eigvec の位相 / 縮退基底回転に依らない比較ができる.
    """
    k = eigvecs.shape[1]
    res = np.empty(k, dtype=np.float64)
    for j in range(k):
        v = eigvecs[:, j]
        Hv = _apply_ht(prob, sched, t, v)
        scale = max(abs(float(eigvals[j])), 1.0)
        res[j] = float(np.linalg.norm(Hv - eigvals[j] * v) / scale)
    return res


@pytest.mark.parametrize("n", [3, 4])
@pytest.mark.parametrize("t_frac", [0.0, 0.3, 1.0])
def test_lanczos_matches_exact(n: int, t_frac: float) -> None:
    """Lanczos と exact 経路が ``rel < 1e-10`` で一致 (eigvals + overlap).

    端点 (``t=0``, ``t=T``) では一方の Hamiltonian 成分が消えるので
    Krylov 部分空間が縮退しやすいが, 始ベクトルを random complex で
    引いている限り下位 ``k`` Ritz 値が下位 ``k`` 厳密固有値に一致する
    (m=32 は dim=16 (n=4) 以上なので flat にならない).
    """
    prob, h_x = _make_problem(n, seed=11)
    sched = Schedule.linear(T=2.0, h_x=h_x)
    t = t_frac * sched.T
    k = 4
    m = 32

    ev_l, vec_l = instantaneous_eigenstates(
        prob, sched, t=t, k=k, method="lanczos", m=m, seed=0
    )
    ev_e, vec_e = instantaneous_eigenstates(prob, sched, t=t, k=k, method="exact")

    assert ev_l.shape == (k,)
    assert ev_e.shape == (k,)
    assert vec_l.shape == (1 << n, k)
    assert vec_e.shape == (1 << n, k)
    assert ev_l.dtype == np.float64
    assert ev_e.dtype == np.float64
    assert vec_l.dtype == np.complex128
    assert vec_e.dtype == np.complex128

    np.testing.assert_allclose(ev_l, ev_e, rtol=1e-10, atol=1e-12)

    # 固有ベクトル比較は残差 ``||H v - λ v||`` で行う (縮退があると個別 vec の
    # overlap で 1 にならないが, 残差は基底回転に依らず 0 になる).
    res_l = _eigvec_residuals(prob, sched, t, ev_l, vec_l)
    res_e = _eigvec_residuals(prob, sched, t, ev_e, vec_e)
    assert np.max(res_l) < 1e-10, f"Lanczos residual {np.max(res_l):.3e} >= 1e-10"
    assert np.max(res_e) < 1e-10, f"Exact residual {np.max(res_e):.3e} >= 1e-10"


def test_lanczos_eigvecs_are_orthonormal() -> None:
    """Ritz vectors の直交性 (Gram 行列が恒等行列に近い)."""
    prob, h_x = _make_problem(n=4, seed=3)
    sched = Schedule.linear(T=1.0, h_x=h_x)
    k = 5
    _, vec = instantaneous_eigenstates(
        prob, sched, t=0.5, k=k, method="lanczos", m=32, seed=7
    )
    gram = vec.conj().T @ vec
    np.testing.assert_allclose(gram, np.eye(k), atol=1e-10)


def test_exact_rejects_large_n() -> None:
    """``method='exact'`` は ``n > 12`` で ``ValueError``."""
    n = 13
    prob = IsingProblem(n=n, H_p_diag=np.zeros(1 << n, dtype=np.float64))
    sched = Schedule.linear(T=1.0, h_x=np.ones(n))
    with pytest.raises(ValueError, match=r"n <= 12"):
        instantaneous_eigenstates(prob, sched, t=0.5, k=1, method="exact")


def test_k_equals_one() -> None:
    """``k=1`` で shape ``(1)`` / ``(2**n, 1)`` を返す."""
    prob, h_x = _make_problem(n=3, seed=5)
    sched = Schedule.linear(T=1.0, h_x=h_x)
    ev_l, vec_l = instantaneous_eigenstates(
        prob, sched, t=0.5, k=1, method="lanczos", m=16, seed=0
    )
    ev_e, vec_e = instantaneous_eigenstates(prob, sched, t=0.5, k=1, method="exact")
    assert ev_l.shape == (1,)
    assert vec_l.shape == (1 << 3, 1)
    assert ev_e.shape == (1,)
    assert vec_e.shape == (1 << 3, 1)
    np.testing.assert_allclose(ev_l, ev_e, rtol=1e-10)


def test_t_zero_gives_driver_groundstate_energy() -> None:
    """``t = 0`` (``A=1, B=0``) で下位固有値が ``-Σ_i |h_x_i|``.

    ``H(0) = -Σ_i h_x_i X_i``. 計算基底ではなく ``X`` 基底の最低エネルギーが
    最小固有値で, 各サイトが ``+h_x_i`` の固有ベクトル ``|+⟩`` を持つ場合に
    エネルギー ``-Σ h_x_i`` を取る (``h_x_i > 0`` の仮定下).
    """
    n = 4
    h_x = np.array([0.7, 1.4, 2.1, 2.8], dtype=np.float64)
    prob = IsingProblem(
        n=n,
        H_p_diag=np.arange(1 << n, dtype=np.float64),  # 任意, t=0 では効かない
    )
    sched = Schedule.linear(T=10.0, h_x=h_x)
    ev, _ = instantaneous_eigenstates(prob, sched, t=0.0, k=1, method="exact")
    expected = -float(np.sum(h_x))
    np.testing.assert_allclose(ev[0], expected, atol=1e-12)


def test_t_equals_T_gives_min_h_p_diag() -> None:
    """``t = T`` (``A=0, B=1``) で下位固有値が ``min(H_p_diag)``.

    ``H(T) = diag(H_p_diag)`` なので固有値は ``H_p_diag`` 自体, 最小値は
    計算基底の "ground state" のエネルギー.
    """
    n = 4
    rng = np.random.default_rng(99)
    h_p = rng.normal(size=1 << n).astype(np.float64)
    prob = IsingProblem(n=n, H_p_diag=h_p)
    sched = Schedule.linear(T=1.0, h_x=np.ones(n))
    ev, vec = instantaneous_eigenstates(prob, sched, t=sched.T, k=1, method="exact")
    np.testing.assert_allclose(ev[0], float(np.min(h_p)), atol=1e-12)
    # ground state は計算基底の min ビット位置 (純粋な |x⟩) のはず.
    j_min = int(np.argmin(h_p))
    np.testing.assert_allclose(np.abs(vec[j_min, 0]), 1.0, atol=1e-12)


def test_k_greater_than_m_raises() -> None:
    """``method='lanczos'`` で ``k > m`` は ``ValueError``."""
    prob, h_x = _make_problem(n=3, seed=0)
    sched = Schedule.linear(T=1.0, h_x=h_x)
    with pytest.raises(ValueError, match=r"k=\d+ must be <= m=\d+"):
        instantaneous_eigenstates(
            prob, sched, t=0.5, k=10, method="lanczos", m=4, seed=0
        )


def test_k_invalid_raises() -> None:
    """``k < 1`` は ``ValueError``."""
    prob, h_x = _make_problem(n=3, seed=0)
    sched = Schedule.linear(T=1.0, h_x=h_x)
    with pytest.raises(ValueError, match=r"k must be a positive integer"):
        instantaneous_eigenstates(prob, sched, t=0.5, k=0)


def test_unknown_method_raises() -> None:
    """未知の ``method`` は ``ValueError``."""
    prob, h_x = _make_problem(n=3, seed=0)
    sched = Schedule.linear(T=1.0, h_x=h_x)
    with pytest.raises(ValueError, match=r"method must be"):
        instantaneous_eigenstates(prob, sched, t=0.5, k=1, method="bogus")  # type: ignore[arg-type]


def test_exact_k_too_large_raises() -> None:
    """``method='exact'`` で ``k > 2**n`` は ``ValueError``."""
    prob, h_x = _make_problem(n=2, seed=0)  # dim = 4
    sched = Schedule.linear(T=1.0, h_x=h_x)
    with pytest.raises(ValueError, match=r"k=\d+ must be <= 2\*\*n"):
        instantaneous_eigenstates(prob, sched, t=0.5, k=5, method="exact")


def test_seed_reproducibility() -> None:
    """同じ ``seed`` で 2 回呼ぶと bit-exact に同じ結果を返す."""
    prob, h_x = _make_problem(n=3, seed=0)
    sched = Schedule.linear(T=1.0, h_x=h_x)
    ev1, vec1 = instantaneous_eigenstates(
        prob, sched, t=0.5, k=3, method="lanczos", m=16, seed=42
    )
    ev2, vec2 = instantaneous_eigenstates(
        prob, sched, t=0.5, k=3, method="lanczos", m=16, seed=42
    )
    np.testing.assert_array_equal(ev1, ev2)
    np.testing.assert_array_equal(vec1, vec2)


def test_seed_different_gives_same_eigvals() -> None:
    """異なる seed でも eigvals は同じ (始ベクトルが部分空間を spans する限り)."""
    prob, h_x = _make_problem(n=3, seed=0)
    sched = Schedule.linear(T=1.0, h_x=h_x)
    ev1, _ = instantaneous_eigenstates(
        prob, sched, t=0.5, k=3, method="lanczos", m=16, seed=1
    )
    ev2, _ = instantaneous_eigenstates(
        prob, sched, t=0.5, k=3, method="lanczos", m=16, seed=999
    )
    np.testing.assert_allclose(ev1, ev2, rtol=1e-10)
