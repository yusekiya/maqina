"""Rust ``lanczos_propagate`` と Python リファレンスの等価性テスト.

Phase 1 C7 (issue #8) の acceptance:

* ``_rust.lanczos_propagate_py`` (Rust 経路) と ``_python_lanczos_propagate``
  (純 NumPy 経路) が ``rel < 1e-13`` で一致する (ランダム Hermitian, n=3..5).
* ``_python_m2_step`` が同一 ``(h_x, h_p_diag, a, b, dt)`` の下で
  ``_rust.m2_midpoint_step_py`` と一致する.

Rust 拡張が未ビルドの環境では Python リファレンス同士の自己一貫性
(propagator 性質: unitarity, dt=0 で恒等) のみを検証する. ビルド済の
場合は Rust 経路との rel < 1e-13 一致を追加検証する.
"""

from __future__ import annotations

import numpy as np
import pytest

from kinema.krylov import (
    _make_python_matvec,
    _python_lanczos_propagate,
    _python_m2_step,
)

try:
    from kinema import _rust as _rust_mod
except ImportError:  # pragma: no cover
    _rust_mod = None  # type: ignore[assignment]


_HAS_RUST = _rust_mod is not None


def _random_hermitian_setup(
    n: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """``(h_x, h_p_diag, psi, a_t, b_t)`` を再現可能に生成する.

    ``psi`` は正規化済 complex128. すべての浮動小数は ``[-1, 1]`` 範囲.
    """
    rng = np.random.default_rng(seed)
    dim = 1 << n
    h_x = rng.uniform(-1.0, 1.0, size=n).astype(np.float64)
    h_p_diag = rng.uniform(-1.0, 1.0, size=dim).astype(np.float64)
    psi = (
        rng.uniform(-1.0, 1.0, size=dim) + 1j * rng.uniform(-1.0, 1.0, size=dim)
    ).astype(np.complex128)
    psi = psi / np.linalg.norm(psi)
    a_t = float(rng.uniform(-1.0, 1.0))
    b_t = float(rng.uniform(-1.0, 1.0))
    return h_x, h_p_diag, psi, a_t, b_t


def _dense_exp_minus_i_dt_h(
    h_x: np.ndarray, h_p_diag: np.ndarray, a_t: float, b_t: float, dt: float
) -> np.ndarray:
    """``exp(-i dt · (a_t · H_drv + b_t · diag(h_p_diag)))`` の dense 構築."""
    n = h_x.shape[0]
    dim = 1 << n
    h = np.diag(b_t * h_p_diag).astype(np.complex128)
    for i in range(n):
        mask = 1 << i
        coeff = -a_t * h_x[i]
        for k in range(dim):
            h[k, k ^ mask] += coeff
    lam, u = np.linalg.eigh(h)
    return u @ np.diag(np.exp(-1j * dt * lam)) @ u.conj().T


@pytest.mark.parametrize("n", [3, 4, 5])
@pytest.mark.parametrize("seed", [11, 137, 8675309])
def test_python_lanczos_matches_dense_propagator(n: int, seed: int) -> None:
    """``_python_lanczos_propagate`` が dense ``exp(-i dt H)·ψ`` と一致する.

    Python リファレンスを単独で先に検証する. Lanczos 部分空間が dim と
    同じ次元 (m = dim) まで届けば理論上は厳密値と一致するので
    ``rel < 1e-10`` を要求する.
    """
    h_x, h_p_diag, psi, a_t, b_t = _random_hermitian_setup(n, seed)
    dt = 0.27
    matvec = _make_python_matvec(h_x, h_p_diag, a_t, b_t)

    dim = 1 << n
    # issue #93 (Phase 7): _python_lanczos_propagate は (psi, m_eff, β_m, |c_m|) を返す.
    # issue #98 (Phase 8): krylov_tol は **Krylov 近似の許容誤差** (a posteriori
    # 判定式 β · |c_last| · |dt| / (k+1) < krylov_tol で早期打切).
    psi_lanczos, m_eff, beta_m, c_m_abs = _python_lanczos_propagate(
        matvec, psi, dt, m=dim, krylov_tol=1e-14
    )
    assert 1 <= m_eff <= dim
    # a posteriori diagnostic は非負実数.
    assert beta_m >= 0.0, f"β_m = {beta_m} must be >= 0"
    assert c_m_abs >= 0.0, f"|c_m| = {c_m_abs} must be >= 0"

    u_full = _dense_exp_minus_i_dt_h(h_x, h_p_diag, a_t, b_t, dt)
    psi_expected = u_full @ psi

    rel = np.linalg.norm(psi_lanczos - psi_expected) / max(
        np.linalg.norm(psi_expected), 1.0
    )
    assert rel < 1e-10, f"n={n}, seed={seed}: rel = {rel}"


@pytest.mark.skipif(not _HAS_RUST, reason="kinema._rust extension not built")
@pytest.mark.parametrize("n", [3, 4, 5])
@pytest.mark.parametrize("seed", [11, 137, 8675309])
def test_rust_lanczos_matches_python_reference(n: int, seed: int) -> None:
    """Rust ``lanczos_propagate_py`` ↔ Python ``_python_lanczos_propagate`` が
    ``rel < 1e-13`` で一致する.

    同一 ``(h_x, h_p_diag, a, b, dt, m, tol)`` を渡し, 浮動小数の演算順序
    差を超えた数値破綻が無いことを確認する.
    """
    assert _rust_mod is not None
    h_x, h_p_diag, psi, a_t, b_t = _random_hermitian_setup(n, seed)
    dt = 0.31
    m = 24
    krylov_tol = 1e-12

    matvec = _make_python_matvec(h_x, h_p_diag, a_t, b_t)
    # issue #93 (Phase 7): Rust も Python ref も (psi, m_eff, β_m, |c_m|) を返す.
    # issue #98 (Phase 8): krylov_tol は **Krylov 近似の許容誤差** (a posteriori
    # 判定式). m_eff / β_m / |c_m| すべて完全一致するのが新しい契約
    # (BLAS feature on/off も同様, 早期打切条件が決定論的なため).
    psi_py, m_eff_py, beta_m_py, c_m_abs_py = _python_lanczos_propagate(
        matvec, psi, dt, m, krylov_tol
    )
    psi_rust, m_eff_rust, beta_m_rust, c_m_abs_rust = _rust_mod.lanczos_propagate_py(
        psi, h_x, h_p_diag, a_t, b_t, dt, m, krylov_tol
    )
    assert m_eff_py == m_eff_rust, f"m_eff mismatch: py={m_eff_py}, rust={m_eff_rust}"
    # β_m / |c_m| も rel < 1e-13 で一致 (浮動小数の演算順序差を超えない).
    assert abs(beta_m_py - beta_m_rust) <= 1e-13 * max(abs(beta_m_rust), 1.0), (
        f"β_m mismatch: py={beta_m_py}, rust={beta_m_rust}"
    )
    assert abs(c_m_abs_py - c_m_abs_rust) <= 1e-13 * max(abs(c_m_abs_rust), 1.0), (
        f"|c_m| mismatch: py={c_m_abs_py}, rust={c_m_abs_rust}"
    )

    rel = np.linalg.norm(psi_py - psi_rust) / max(np.linalg.norm(psi_rust), 1.0)
    assert rel < 1e-13, f"n={n}, seed={seed}: rel = {rel}"


@pytest.mark.skipif(not _HAS_RUST, reason="kinema._rust extension not built")
@pytest.mark.parametrize("n", [3, 4])
def test_python_m2_step_matches_rust(n: int) -> None:
    """``_python_m2_step`` が ``_rust.m2_midpoint_step_py`` と
    ``rel < 1e-13`` で一致する.

    ``a_mid`` / ``b_mid`` を呼出側で計算する Phase 1 driver の契約 (中点
    schedule 評価は krylov 層の外) を踏襲し, 両経路に同一の ``a_mid`` /
    ``b_mid`` を渡す.
    """
    assert _rust_mod is not None
    h_x, h_p_diag, psi, a_mid, b_mid = _random_hermitian_setup(n, seed=4242)
    dt = 0.17
    m = 24
    tol = 1e-12

    psi_py = _python_m2_step(psi, h_x, h_p_diag, a_mid, b_mid, dt, m, tol)
    psi_rust = _rust_mod.m2_midpoint_step_py(
        psi, h_x, h_p_diag, a_mid, b_mid, dt, m, tol
    )
    rel = np.linalg.norm(psi_py - psi_rust) / max(np.linalg.norm(psi_rust), 1.0)
    assert rel < 1e-13, f"n={n}: rel = {rel}"


def test_python_lanczos_preserves_norm() -> None:
    """``_python_lanczos_propagate`` は unitary を返すので ``‖ψ_new‖ = ‖ψ‖``."""
    n = 4
    h_x, h_p_diag, psi, a_t, b_t = _random_hermitian_setup(n, seed=2025)
    dt = 0.23
    matvec = _make_python_matvec(h_x, h_p_diag, a_t, b_t)
    psi_new, _m_eff, _beta_m, _c_m_abs = _python_lanczos_propagate(
        matvec, psi, dt, m=24, krylov_tol=1e-12
    )
    rel = abs(np.linalg.norm(psi_new) - np.linalg.norm(psi)) / max(
        np.linalg.norm(psi), 1.0
    )
    assert rel < 1e-13, f"unitarity violated: rel = {rel}"


def test_python_lanczos_dt_zero_is_identity() -> None:
    """``dt = 0`` のとき ``exp(0) · ψ = ψ`` の数値一致を確認する.

    issue #98 (Phase 8) で a posteriori 判定式は ``β · |c_last| · |dt| / (k+1)``
    なので ``dt = 0`` では iter 0 の判定で必ず 0 < krylov_tol が成立し,
    ``m_eff = 1`` で即打切される (Krylov 部分空間が 1 次元になる). このとき
    T_1 = [α_0], c = exp(-i · 0 · α_0) = 1 となるため ``|c_m| = 1`` (Phase 7
    時代の "dt=0 で c_m = 0" assertion は仕様変更で意味を失う).
    """
    n = 4
    h_x, h_p_diag, psi, a_t, b_t = _random_hermitian_setup(n, seed=2026)
    matvec = _make_python_matvec(h_x, h_p_diag, a_t, b_t)
    psi_new, m_eff, beta_m, c_m_abs = _python_lanczos_propagate(
        matvec, psi, 0.0, m=24, krylov_tol=1e-12
    )
    rel = np.linalg.norm(psi_new - psi) / max(np.linalg.norm(psi), 1.0)
    assert rel < 1e-13, f"dt=0 identity violated: rel = {rel}"
    # issue #98 (Phase 8): dt=0 では a posteriori 判定が iter 0 で 0 < krylov_tol
    # を満たし即打切. m_eff = 1, c_m_abs = 1, β_m は β_0 (matvec の漏れ強度).
    assert m_eff == 1, f"m_eff at dt=0 should be 1 (a posteriori 即打切), got {m_eff}"
    assert abs(c_m_abs - 1.0) < 1e-13, f"|c_m| at dt=0 should be ~1 but got {c_m_abs}"
    # β_m は dt 非依存なので非負.
    assert beta_m >= 0.0, f"β_m must be non-negative: {beta_m}"


def test_python_lanczos_beta_m_zero_for_invariant_subspace() -> None:
    """H が ψ_0 の不変部分空間に閉じている場合は早期打切, β_m が機械精度.

    具体的には, ψ = e_0 (計算基底の 0 番目) かつ H = h_x_0 · X_0 と
    すると K_2(H, e_0) = span{e_0, X_0 e_0} = span{e_0, e_1} で閉じ, β_2 ≈ 0.

    issue #93 (Phase 7) で β_m exposure の acceptance, issue #98 (Phase 8) で
    a posteriori 判定の前段に置かれた numerical breakdown (β < 1e-14) 即打切で
    同じ挙動が得られることを確認する.
    """
    n = 3
    dim = 1 << n
    # H = -X_0 のみ (h_x[0]=1, 他=0, h_p_diag=0). K_2 が閉じる.
    h_x = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    h_p_diag = np.zeros(dim, dtype=np.float64)
    psi = np.zeros(dim, dtype=np.complex128)
    psi[0] = 1.0  # e_0
    matvec = _make_python_matvec(h_x, h_p_diag, a_t=1.0, b_t=0.0)
    krylov_tol = 1e-10
    psi_new, m_eff, beta_m, c_m_abs = _python_lanczos_propagate(
        matvec, psi, dt=0.3, m=8, krylov_tol=krylov_tol
    )
    # K_2 で閉じるので m_eff <= 3 (k=0,1 のどちらかで早期打切).
    assert m_eff <= 3, f"m_eff should be small for invariant subspace, got {m_eff}"
    # numerical breakdown (1e-14) または a posteriori 判定で打切.
    # β_m は前者では機械精度近辺, 後者では `est < krylov_tol` 達成時の値.
    assert beta_m < krylov_tol, (
        f"β_m={beta_m} must be < krylov_tol={krylov_tol} after early termination"
    )
    assert c_m_abs >= 0.0, f"|c_m|={c_m_abs} must be non-negative"


# ---------------------------------------------------------------------------
# issue #98 (Phase 8): a posteriori 早期打切 acceptance テスト
# ---------------------------------------------------------------------------


def test_python_lanczos_aposteriori_termination_fires() -> None:
    """``krylov_tol`` を緩めると a posteriori 判定で m_eff < m_max が観測される.

    Phase 7 では中間 β_j が O(‖H‖) なので β 単体閾値では発火しなかったが,
    Phase 8 では Hochbruck-Lubich 1997 上界
    ``est = β_k · |c_last| · |dt| / (k+1)`` を見るので
    |c_last| の幾何級数減衰 (~ 1e-4 / iter) で発火する.
    """
    # 典型的な TFIM 設定 (n=4 程度, dt=0.1) を再現. tools/diag_beta_j_dump.py
    # 由来の経験値で krylov_tol=1e-7 → m_eff ≈ 5-6 を期待.
    n = 4
    h_x, h_p_diag, psi, a_t, b_t = _random_hermitian_setup(n, seed=98_01)
    dt = 0.1
    m_max = 24
    matvec = _make_python_matvec(h_x, h_p_diag, a_t, b_t)

    # 緩い tol: 早期打切が確実に発火する範囲.
    _psi_loose, m_eff_loose, _bm_loose, _cm_loose = _python_lanczos_propagate(
        matvec, psi, dt, m=m_max, krylov_tol=1e-7
    )
    # 厳しい tol: m_max まで build される (or numerical breakdown 寸前まで).
    _psi_tight, m_eff_tight, _bm_tight, _cm_tight = _python_lanczos_propagate(
        matvec, psi, dt, m=m_max, krylov_tol=1e-14
    )

    # 緩いほど m_eff が小さくなる (a posteriori 判定が機能している証拠).
    assert m_eff_loose < m_eff_tight, (
        f"a posteriori 判定が発火していない: "
        f"loose={m_eff_loose}, tight={m_eff_tight} (loose >= tight)"
    )
    # 緩い tol では m_max よりかなり小さい次元で打切される.
    assert m_eff_loose < m_max, (
        f"a posteriori 早期打切で m_eff < m_max が期待される: "
        f"m_eff_loose={m_eff_loose}, m_max={m_max}"
    )


def test_python_lanczos_aposteriori_accuracy_preserved() -> None:
    """早期打切しても ``rel < krylov_tol · 10`` 程度の精度は保たれる.

    a posteriori 判定式は誤差上界なので, 打切した時点の ``est < krylov_tol``
    は実際の誤差 ``rel`` の上界を意味する. 安全マージン 1 桁を見て
    ``rel < krylov_tol · 10`` を契約とする.
    """
    n = 4
    h_x, h_p_diag, psi, a_t, b_t = _random_hermitian_setup(n, seed=98_02)
    dt = 0.1
    matvec = _make_python_matvec(h_x, h_p_diag, a_t, b_t)

    # 複数の krylov_tol レベルで rel と m_eff の対応を確認.
    u_full = _dense_exp_minus_i_dt_h(h_x, h_p_diag, a_t, b_t, dt)
    psi_expected = u_full @ psi
    ref_norm = max(np.linalg.norm(psi_expected), 1.0)

    for krylov_tol in (1e-4, 1e-6, 1e-8, 1e-10):
        psi_new, m_eff, _bm, _cm = _python_lanczos_propagate(
            matvec, psi, dt, m=24, krylov_tol=krylov_tol
        )
        rel = np.linalg.norm(psi_new - psi_expected) / ref_norm
        # 1 桁の安全マージンで上界を保証.
        assert rel < krylov_tol * 10, (
            f"krylov_tol={krylov_tol}: rel={rel:.3e} exceeds krylov_tol*10="
            f"{krylov_tol * 10:.3e} (m_eff={m_eff})"
        )


def test_python_lanczos_aposteriori_monotone_compression() -> None:
    """``krylov_tol`` を緩めるほど m_eff が単調に縮む.

    a posteriori 判定式 ``β · |c_last| · |dt| / (k+1) < krylov_tol`` は
    `krylov_tol` について単調なので, ある同じ (matvec, ψ, dt, m_max) で
    krylov_tol を大きくすれば m_eff は decrease (or 同じ) を保つ.
    """
    n = 4
    h_x, h_p_diag, psi, a_t, b_t = _random_hermitian_setup(n, seed=98_03)
    dt = 0.1
    matvec = _make_python_matvec(h_x, h_p_diag, a_t, b_t)

    tols = [1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2]
    m_effs: list[int] = []
    for krylov_tol in tols:
        _psi, m_eff, _bm, _cm = _python_lanczos_propagate(
            matvec, psi, dt, m=24, krylov_tol=krylov_tol
        )
        m_effs.append(m_eff)

    # 単調非増加.
    for i in range(1, len(tols)):
        assert m_effs[i] <= m_effs[i - 1], (
            f"monotonic compression violated: tol={tols[i - 1]} m_eff={m_effs[i - 1]}, "
            f"tol={tols[i]} m_eff={m_effs[i]}"
        )


@pytest.mark.skipif(not _HAS_RUST, reason="kinema._rust extension not built")
@pytest.mark.parametrize("n", [3, 4])
def test_m2_midpoint_step_inplace_py_matches_alloc_variant_bitwise(n: int) -> None:
    """``m2_midpoint_step_inplace_py`` の結果が ``m2_midpoint_step_py`` と
    **bit-for-bit** 一致する (issue #86).

    両者は内部で同じ ``m2_midpoint_step`` (= ``lanczos_propagate`` 1 回) を
    呼ぶので, ``psi_new`` を ``into_pyarray`` で新規 alloc して返すか
    caller 提供の ``psi`` に ``copy_from_slice`` で書き戻すかが唯一の違い.
    演算順序は同一なので bit-identical を期待する.
    """
    assert _rust_mod is not None
    h_x, h_p_diag, psi, a_mid, b_mid = _random_hermitian_setup(n, seed=4242)
    dt = 0.17
    m = 24
    tol = 1e-12

    psi_alloc = _rust_mod.m2_midpoint_step_py(
        psi, h_x, h_p_diag, a_mid, b_mid, dt, m, tol
    )

    psi_inplace = psi.copy()
    ret = _rust_mod.m2_midpoint_step_inplace_py(
        psi_inplace, h_x, h_p_diag, a_mid, b_mid, dt, m, tol
    )
    assert ret is None
    assert np.array_equal(psi_inplace, psi_alloc), (
        f"n={n}: in-place / alloc が bitwise 一致しない: "
        f"max abs diff = {np.max(np.abs(psi_inplace - psi_alloc))}"
    )
