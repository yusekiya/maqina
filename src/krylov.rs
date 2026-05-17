//! `lanczos_propagate`: matrix-free 短時間プロパゲータ.
//!
//! Park-Light (1986) の Lanczos 短時間プロパゲータを `m` 次元 Krylov 部分
//! 空間で構築する. 1 step あたりの主コストは `m` 回の matvec と
//! Level-1 / Level-2 BLAS で, 部分空間内の `m × m` 実対称三重対角の完全
//! 固有分解は `tridiag_eigh` (LAPACK 非依存, `src/tridiag.rs`) で行う.
//! 詳細は `docs/design/05-2-lanczos.md` §5.2, §7.1.
//!
//! 設計ポイント:
//!
//! * **Closure matvec**: `F: FnMut(&[Complex64], &mut [Complex64])` を取り,
//!   `y = H · v` を 1 回適用する. CFM4:2 の各 stage では
//!   `(c_drv, c_diag)` を畳み込んだ `apply_h_kryanneal` をクロージャに
//!   ラップして渡せる (§5.2 末尾).
//! * **Full re-orthogonalization (Gram-Schmidt 2-pass)**: m が小さく (m ≈ 24)
//!   かつ Lanczos の数値直交性が直ちに崩れる経験則 (Paige 1971) に従い,
//!   3 項漸化式に加えて 2-pass full reortho を毎ステップ実行. m ≪ dim なので
//!   `m·(m+1)·dim` の追加コストは無視できる.
//! * **`β_k < tol` 打切り**: 部分空間が flat になった場合は `m_eff = k+1` で
//!   早期終了し, それ以降の Lanczos vector / tridiag 行は構築しない.
//! * **終端再構成**: column-major の Lanczos vector 行列
//!   `V[:, :m_eff]` に対し `psi_new = V · c` を `gemv_col_major` 1 回で
//!   組む (BLAS ON では `zgemv`, OFF では axpy 相当の自前ループ).
//!
//! Phase 1 では本関数を Python に公開しない (`pub(crate)`). M2 / CFM4:2 が
//! 上位で wrap した形で公開する (`docs/design/05-3-propagator.md` §5.3). 着地直後は内部
//! caller がいないため `dead_code` lint を許容する.

#![allow(dead_code)]

use num_complex::Complex64;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::blas::{axpy, dot_conj, gemv_col_major, nrm2, scal_real};
use crate::matvec::apply_h_kryanneal;
use crate::tridiag::tridiag_eigh;

/// `psi_new = exp(-i dt H) ψ` を m 次元 Lanczos + 三重対角固有分解で近似.
///
/// # 引数
/// * `matvec`: `H · v` を計算する closure. 戻り値は `y[..] = H · v` の上書き
///   (additive ではない). CFM4:2 の `(c_drv, c_diag)` 畳み込み closure を
///   渡せる.
/// * `psi`: 入力状態. `dim = psi.len()`.
/// * `dt`: 時刻刻み幅 (real).
/// * `m`: Krylov 部分空間次元 (典型値 24). `m ≥ 1` を要求.
/// * `tol`: Lanczos の β 打切り閾値. `β_k < tol` で `m_eff = k+1` として
///   早期終了する.
///
/// # 戻り値
/// * `Ok(psi_new)`: 長さ `dim` の新状態.
/// * `Err`: 三重対角固有分解が収束しなかった場合 (`PyRuntimeError`).
///
/// `‖ψ‖ = 0` のときは零ベクトルを返す (matvec も呼ばない). 三重対角の
/// `α_k = Re ⟨v_k | H v_k⟩` で `Im` 部は Hermitian 性から 0 であるべきだが,
/// 浮動小数点の swamp ノイズ分は明示的に切り捨てる.
///
/// # Panics
/// `m == 0`.
pub(crate) fn lanczos_propagate<F>(
    mut matvec: F,
    psi: &[Complex64],
    dt: f64,
    m: usize,
    tol: f64,
) -> PyResult<(Vec<Complex64>, usize, f64, f64)>
where
    F: FnMut(&[Complex64], &mut [Complex64]),
{
    // issue #93 (Phase 7): return tuple は (psi_new, m_eff, beta_m, c_m_abs).
    // 末尾 2 要素は a posteriori 誤差推定 (Saad 1992 / Hochbruck-Lubich 1997):
    //   err_lanczos ≈ β_m · |c_m| · ‖ψ‖ · dt / m_eff
    // adaptive Richardson driver が Lanczos 誤差を Magnus 誤差から分離するため
    // に使う. β_m は build 完了時の最終 off-diagonal, c_m は exp(-i dt T_m) e_0
    // の m 番目成分. m_eff = 0 (空状態 / ‖ψ‖=0) のときは 0.0 を返す.
    assert!(m >= 1, "m must be >= 1");
    let dim = psi.len();
    if dim == 0 {
        // 空状態は m_eff=0 として扱う (Lanczos vector が 1 本も構築されない).
        return Ok((Vec::new(), 0, 0.0, 0.0));
    }

    // ‖ψ‖ が 0 のときは 0 ベクトルを返す. m_eff は 0 を返す (実 Krylov 部分空間
    // 次元 0; tridiag 固有分解にも入らない). issue #52 A.
    let psi_norm = nrm2(psi);
    if psi_norm == 0.0 {
        return Ok((vec![Complex64::new(0.0, 0.0); dim], 0, 0.0, 0.0));
    }

    // V: column-major (dim × m). 各列が Lanczos vector v_0, v_1, ....
    // 早期打切時は v_0..v_{m_eff-1} のみ充填され, 残りは未使用.
    let mut v_mat = vec![Complex64::new(0.0, 0.0); dim * m];
    // 三重対角: α_k = ⟨v_k | H v_k⟩ (real), β_k = ‖w_k‖ (off-diagonal).
    // β は β_0..β_{m_eff-2} が有効. β_{m_eff-1} は構築しない (sentinel は
    // tridiag_eigh 側で 0 にされる).
    let mut alpha = vec![0.0_f64; m];
    let mut beta = vec![0.0_f64; m];

    // 作業バッファ. matvec の出力先 / Gram-Schmidt の残差.
    let mut w = vec![Complex64::new(0.0, 0.0); dim];

    // v_0 = ψ / ‖ψ‖.
    let inv_norm = 1.0 / psi_norm;
    {
        let v0 = &mut v_mat[0..dim];
        for k in 0..dim {
            v0[k] = psi[k] * inv_norm;
        }
    }

    let mut m_eff = m;

    for k in 0..m {
        // w = H · v_k.
        // 借用衝突回避のため v_k を一度スライス化し, matvec の引数として
        // 渡す. matvec は w にのみ書き込むので v_mat の他列とは干渉しない.
        {
            let (v_segment, _rest) = v_mat.split_at(dim * m);
            let v_k = &v_segment[k * dim..(k + 1) * dim];
            matvec(v_k, &mut w);
        }

        // α_k = Re ⟨v_k | w⟩. Hermitian H なら Im 部は 0 だが浮動小数の
        // swamp 分を明示的に落とす.
        let alpha_k = {
            let v_k = &v_mat[k * dim..(k + 1) * dim];
            dot_conj(v_k, &w).re
        };
        alpha[k] = alpha_k;

        // 3-term 漸化式: w -= α_k · v_k + β_{k-1} · v_{k-1}.
        {
            let v_k = &v_mat[k * dim..(k + 1) * dim];
            axpy(Complex64::new(-alpha_k, 0.0), v_k, &mut w);
        }
        if k >= 1 {
            let v_km1 = &v_mat[(k - 1) * dim..k * dim];
            axpy(Complex64::new(-beta[k - 1], 0.0), v_km1, &mut w);
        }

        // Full re-orthogonalization (2-pass Gram-Schmidt).
        // v_0..v_k の全列に対し ⟨v_j | w⟩ を引き戻す. 2 pass で
        // ‖orth-error‖ ~ ε に収まることが知られている (Daniel et al. 1976).
        for _pass in 0..2 {
            for j in 0..=k {
                let proj = {
                    let v_j = &v_mat[j * dim..(j + 1) * dim];
                    dot_conj(v_j, &w)
                };
                let v_j = &v_mat[j * dim..(j + 1) * dim];
                axpy(-proj, v_j, &mut w);
            }
        }

        // β_k = ‖w‖.
        let beta_k = nrm2(&w);
        beta[k] = beta_k;

        if beta_k < tol {
            // 部分空間が flat. v_{k+1} 以降は構築せず Krylov を打ち切る.
            m_eff = k + 1;
            break;
        }

        // v_{k+1} = w / β_k (k+1 < m のときのみ; 最終 step では不要).
        if k + 1 < m {
            let dest_range = (k + 1) * dim..(k + 2) * dim;
            v_mat[dest_range.clone()].copy_from_slice(&w);
            scal_real(1.0 / beta_k, &mut v_mat[dest_range]);
        }
    }

    // 三重対角 T (m_eff × m_eff) の固有分解.
    // d_buf に α_0..α_{m_eff-1}, e_buf に β_0..β_{m_eff-2} (+ sentinel).
    let mut d_buf = alpha[..m_eff].to_vec();
    let mut e_buf = vec![0.0_f64; m_eff];
    let off_diag = m_eff.saturating_sub(1);
    e_buf[..off_diag].copy_from_slice(&beta[..off_diag]);
    // e_buf[m_eff - 1] は sentinel として tridiag_eigh 内で 0 化される.
    let mut q_buf = vec![0.0_f64; m_eff * m_eff];
    tridiag_eigh(&mut d_buf, &mut e_buf, &mut q_buf)
        .map_err(|e| PyRuntimeError::new_err(format!("tridiag_eigh failed: {e}")))?;

    // c_k = ‖ψ‖ · Σ_j Q[j, k] · exp(-i dt λ_j) · Q[j, 0].
    // tridiag_eigh の規約は T = Qᵀ diag(λ) Q, Q row-major で row j が
    // λ_j に対応する単位固有ベクトル. exp(-i dt T) · (β_0 e_1) は
    //   ψ_new = V · c, c = ‖ψ‖ · Qᵀ · diag(exp(-i dt λ)) · Q · e_1
    // となり, e_1 によって Q · e_1 は Q の 0 列目 (= 各行の 0 番目要素).
    let mut c = vec![Complex64::new(0.0, 0.0); m_eff];
    for k in 0..m_eff {
        let mut acc = Complex64::new(0.0, 0.0);
        for j in 0..m_eff {
            let lambda_j = d_buf[j];
            // exp(-i dt λ_j) = cos(dt λ_j) - i · sin(dt λ_j).
            let phase = Complex64::new(0.0, -dt * lambda_j).exp();
            let q_j_k = q_buf[j * m_eff + k];
            let q_j_0 = q_buf[j * m_eff];
            acc += Complex64::new(q_j_k, 0.0) * phase * Complex64::new(q_j_0, 0.0);
        }
        c[k] = acc * Complex64::new(psi_norm, 0.0);
    }

    // ψ_new = V[:, :m_eff] · c. column-major で V の先頭 dim·m_eff 要素が
    // ちょうど V[:, :m_eff] になる.
    let mut psi_new = vec![Complex64::new(0.0, 0.0); dim];
    gemv_col_major(&v_mat[..dim * m_eff], dim, m_eff, &c, &mut psi_new);

    // a posteriori 誤差推定子の出力 (issue #93 Phase 7):
    // - β_m: build 完了時の最終 off-diagonal (= 次の Krylov 方向への漏れ強度).
    //   早期打切時は β_{m_eff-1} < tol を満たした最終 β.
    //   非打切時は loop 末尾 (k = m-1) で計算された β_{m-1}.
    // - |c_m|: c = exp(-i dt T_m) e_0 の m 番目 (= 末尾) 成分の絶対値.
    //   **‖ψ‖ は含めない** (literature 標準, Hochbruck-Lubich 1997 eq. 5.4-5.5).
    //   現状 c[k] には psi_norm を乗じた値が入っているので, ‖ψ‖ で割って
    //   pure な行列要素 |⟨e_m, exp(-i dt T_m) e_0⟩| を返す.
    //   a posteriori 推定式は呼出側で
    //   `err_lanczos ≈ β_m · |c_m| · ‖ψ‖ · dt / m_eff` の形で組み立てる.
    let beta_m = beta[m_eff - 1];
    let c_m_abs = c[m_eff - 1].norm() / psi_norm;

    Ok((psi_new, m_eff, beta_m, c_m_abs))
}

/// `lanczos_propagate` の **テスト用 Python wrap**.
///
/// 内部 `lanczos_propagate` (closure 受け取り) は Python から直接呼べないため,
/// `apply_h_kryanneal` を closure として固定した形で
/// `psi_new = exp(-i dt · H(a_t, b_t)) · ψ` を計算する関数を露出する.
/// `H(a_t, b_t) = a_t · H_driver + b_t · H_problem` で時間に独立な
/// (フリーズ済の) Hamiltonian を仮定する点が `m2_midpoint_step_py` と異なる
/// (本関数は中点採取をしない: ユーザ側で a, b を midpoint で評価して渡す
/// 経路は `m2_midpoint_step_py` 側で提供).
///
/// 主用途は Python リファレンス実装 `_python_lanczos_propagate` との
/// `rel < 1e-13` 等価性テスト (`tests/test_krylov.py`). `m2_midpoint_step_py`
/// と本関数は **本体は同一** だが, 「lanczos = 中点採取しない時間独立
/// プロパゲータ」「m2_midpoint = 中点で a, b を凍結する時間依存
/// プロパゲータ」の **意味論を別 entry point で示す** ことで, Python
/// テストが何を比較しているかを呼出名から読めるようにしている.
///
/// # Python 側シグネチャ
/// ```python
/// psi_new, m_eff, beta_m, c_m_abs = _rust.lanczos_propagate_py(
///     psi, h_x, h_p_diag, a_t, b_t, dt, m, krylov_tol,
/// )
/// ```
///
/// issue #93 (Phase 7) で 4-tuple 化. 末尾 2 要素は
/// `err_lanczos ≈ β_m · |c_m| · ‖ψ‖ · dt / m_eff` の a posteriori 推定に使う.
#[pyfunction]
#[pyo3(signature = (psi, h_x, h_p_diag, a_t, b_t, dt, m, krylov_tol))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn lanczos_propagate_py<'py>(
    py: Python<'py>,
    psi: PyReadonlyArray1<'py, Complex64>,
    h_x: PyReadonlyArray1<'py, f64>,
    h_p_diag: PyReadonlyArray1<'py, f64>,
    a_t: f64,
    b_t: f64,
    dt: f64,
    m: usize,
    krylov_tol: f64,
) -> PyResult<(Bound<'py, PyArray1<Complex64>>, usize, f64, f64)> {
    let psi_slice = psi.as_slice()?;
    let h_x_slice = h_x.as_slice()?;
    let h_p_diag_slice = h_p_diag.as_slice()?;

    let n = h_x_slice.len();
    let dim = 1usize << n;
    if h_p_diag_slice.len() != dim {
        return Err(PyValueError::new_err(format!(
            "h_p_diag length {} does not match 2^len(h_x) = 2^{} = {}",
            h_p_diag_slice.len(),
            n,
            dim,
        )));
    }
    if psi_slice.len() != dim {
        return Err(PyValueError::new_err(format!(
            "psi length {} does not match 2^len(h_x) = {}",
            psi_slice.len(),
            dim,
        )));
    }
    if m == 0 {
        return Err(PyValueError::new_err("m must be >= 1"));
    }

    let matvec = |v: &[Complex64], y: &mut [Complex64]| {
        apply_h_kryanneal(v, y, h_x_slice, h_p_diag_slice, a_t, b_t, n);
    };
    // issue #93 (Phase 7): lanczos_propagate は (psi, m_eff, β_m, |c_m|) を返す.
    // Python テスト (`tests/test_krylov.py`) も対応する 4-tuple destructure に追従.
    let (psi_new, m_eff, beta_m, c_m_abs) =
        lanczos_propagate(matvec, psi_slice, dt, m, krylov_tol)?;
    Ok((psi_new.into_pyarray(py), m_eff, beta_m, c_m_abs))
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, SymmetricEigen};

    /// 軽量決定論的 PRNG (xorshift64). matvec / tridiag のテストと同実装.
    struct Xor64(u64);

    impl Xor64 {
        fn new(seed: u64) -> Self {
            let s = if seed == 0 {
                0xdead_beef_cafe_babe
            } else {
                seed
            };
            Self(s)
        }

        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }

        fn signed(&mut self) -> f64 {
            const SCALE: f64 = 1.0 / (1u64 << 53) as f64;
            let u = (self.next_u64() >> 11) as f64 * SCALE;
            2.0 * u - 1.0
        }

        fn complex_signed(&mut self) -> Complex64 {
            Complex64::new(self.signed(), self.signed())
        }
    }

    fn random_complex_vec(n: usize, seed: u64) -> Vec<Complex64> {
        let mut rng = Xor64::new(seed);
        (0..n).map(|_| rng.complex_signed()).collect()
    }

    /// `‖a - b‖ / max(‖b‖, 1)`.
    fn relative_error(actual: &[Complex64], expected: &[Complex64]) -> f64 {
        assert_eq!(actual.len(), expected.len());
        let mut diff_sq = 0.0_f64;
        let mut ref_sq = 0.0_f64;
        for k in 0..actual.len() {
            let d = actual[k] - expected[k];
            diff_sq += d.norm_sqr();
            ref_sq += expected[k].norm_sqr();
        }
        diff_sq.sqrt() / ref_sq.sqrt().max(1.0)
    }

    /// `(a_t, b_t, h_x, h_p_diag)` から `dim × dim` の実 dense Hamiltonian を
    /// 構築. H_drv (X 演算子の和) も diag も実係数のため H 全体は実対称.
    fn build_dense_h_real(
        n: usize,
        h_x: &[f64],
        h_p_diag: &[f64],
        a_t: f64,
        b_t: f64,
    ) -> DMatrix<f64> {
        let dim = 1usize << n;
        let mut h = DMatrix::<f64>::zeros(dim, dim);
        for k in 0..dim {
            h[(k, k)] = b_t * h_p_diag[k];
        }
        for (i, &h_x_i) in h_x.iter().enumerate() {
            let coeff = -a_t * h_x_i;
            let mask = 1usize << i;
            for k in 0..dim {
                h[(k, k ^ mask)] += coeff;
            }
        }
        h
    }

    /// 実対称 H の eigendecomp を経由した `exp(-i dt H) · ψ` の参照計算.
    /// `H = U Λ Uᵀ` (U の列が固有ベクトル) より
    /// `ψ_new = U · diag(exp(-i dt λ)) · Uᵀ · ψ`.
    fn reference_propagate_real_h(h: &DMatrix<f64>, psi: &[Complex64], dt: f64) -> Vec<Complex64> {
        let dim = h.nrows();
        assert_eq!(h.ncols(), dim);
        let eig = SymmetricEigen::new(h.clone());
        let lam = &eig.eigenvalues;
        let u = &eig.eigenvectors;

        // c_j = Σ_l U[l, j] · ψ_l, then c_j *= exp(-i dt λ_j).
        let mut c = vec![Complex64::new(0.0, 0.0); dim];
        for j in 0..dim {
            let mut s = Complex64::new(0.0, 0.0);
            for l in 0..dim {
                s += Complex64::new(u[(l, j)], 0.0) * psi[l];
            }
            let phase = Complex64::new(0.0, -dt * lam[j]).exp();
            c[j] = s * phase;
        }
        // ψ_new = U · c.
        let mut psi_new = vec![Complex64::new(0.0, 0.0); dim];
        for k in 0..dim {
            let mut s = Complex64::new(0.0, 0.0);
            for j in 0..dim {
                s += Complex64::new(u[(k, j)], 0.0) * c[j];
            }
            psi_new[k] = s;
        }
        psi_new
    }

    #[test]
    fn zero_psi_returns_zero() {
        // ‖ψ‖ = 0 → 0 ベクトル. matvec は呼ばれないはず.
        let dim = 8_usize;
        let psi = vec![Complex64::new(0.0, 0.0); dim];
        let mut call_count = 0_usize;
        let matvec = |_v: &[Complex64], _y: &mut [Complex64]| {
            call_count += 1;
        };
        let (result, m_eff, beta_m, c_m_abs) =
            lanczos_propagate(matvec, &psi, 0.1, 8, 1e-12).expect("ok");
        assert_eq!(result.len(), dim);
        for &c in &result {
            assert_eq!(c, Complex64::new(0.0, 0.0));
        }
        assert_eq!(call_count, 0, "matvec must not be called on zero psi");
        // ‖ψ‖=0 fast-path では m_eff=0 (Krylov 構築なし). β_m / |c_m| も 0.0.
        assert_eq!(m_eff, 0);
        assert_eq!(beta_m, 0.0);
        assert_eq!(c_m_abs, 0.0);
    }

    #[test]
    fn zero_hamiltonian_preserves_state() {
        // H = 0 → exp(-i dt · 0) · ψ = ψ. β_0 = 0 で即打切り (m_eff = 1).
        let n = 4_usize;
        let dim = 1usize << n;
        let psi = random_complex_vec(dim, 11);
        let matvec = |_v: &[Complex64], y: &mut [Complex64]| {
            for slot in y.iter_mut() {
                *slot = Complex64::new(0.0, 0.0);
            }
        };
        let (result, m_eff, beta_m, c_m_abs) =
            lanczos_propagate(matvec, &psi, 0.37, 24, 1e-12).expect("ok");
        let rel = relative_error(&result, &psi);
        assert!(rel < 1e-13, "H=0 case rel = {}", rel);
        // H=0 → β_0 = 0 で即打切. m_eff = 1, β_m = 0, |c_m| = 1 (e_0 のまま).
        assert_eq!(m_eff, 1);
        assert!(beta_m < 1e-13, "β_m = {} should be ~0", beta_m);
        assert!(
            (c_m_abs - 1.0).abs() < 1e-13,
            "|c_m| = {} should be ~1",
            c_m_abs
        );
    }

    #[test]
    fn diagonal_h_applies_phase_per_component() {
        // H = diag(λ) → ψ_new[k] = ψ[k] · exp(-i dt λ_k).
        // (h_x = 0 で apply_h_kryanneal は対角項のみ計算する.)
        let n = 4_usize;
        let dim = 1usize << n;
        let h_x = vec![0.0_f64; n];
        let mut rng = Xor64::new(42);
        let h_p_diag: Vec<f64> = (0..dim).map(|_| rng.signed()).collect();
        let psi = random_complex_vec(dim, 13);
        let dt = 0.5_f64;
        let a_t = 0.0_f64; // どうせ h_x = 0 だが明示的に.
        let b_t = 1.0_f64;

        let matvec = |v: &[Complex64], y: &mut [Complex64]| {
            apply_h_kryanneal(v, y, &h_x, &h_p_diag, a_t, b_t, n);
        };
        let (result, _m_eff, beta_m, c_m_abs) =
            lanczos_propagate(matvec, &psi, dt, 16, 1e-12).expect("ok");

        let mut expected = vec![Complex64::new(0.0, 0.0); dim];
        for k in 0..dim {
            let lam_k = b_t * h_p_diag[k];
            let phase = Complex64::new(0.0, -dt * lam_k).exp();
            expected[k] = phase * psi[k];
        }
        let rel = relative_error(&result, &expected);
        assert!(rel < 1e-13, "diag H rel = {}", rel);
        // β_m / |c_m| は非負実数.
        assert!(beta_m >= 0.0, "β_m = {} must be >= 0", beta_m);
        assert!(c_m_abs >= 0.0, "|c_m| = {} must be >= 0", c_m_abs);
    }

    #[test]
    fn hermitian_h_preserves_norm() {
        // exp(-i dt H) は unitary なので ‖ψ_new‖ = ‖ψ‖.
        let n = 4_usize;
        let dim = 1usize << n;
        let mut rng = Xor64::new(123);
        let h_x: Vec<f64> = (0..n).map(|_| rng.signed()).collect();
        let h_p_diag: Vec<f64> = (0..dim).map(|_| rng.signed()).collect();
        let psi = random_complex_vec(dim, 99);
        let psi_norm = nrm2(&psi);
        let dt = 0.3_f64;
        let a_t = 0.4_f64;
        let b_t = 1.1_f64;

        let matvec = |v: &[Complex64], y: &mut [Complex64]| {
            apply_h_kryanneal(v, y, &h_x, &h_p_diag, a_t, b_t, n);
        };
        let (result, _m_eff, _beta_m, _c_m_abs) =
            lanczos_propagate(matvec, &psi, dt, 24, 1e-12).expect("ok");
        let new_norm = nrm2(&result);
        let rel = (new_norm - psi_norm).abs() / psi_norm.max(1.0);
        assert!(
            rel < 1e-13,
            "Hermitian norm rel = {} (before={}, after={})",
            rel,
            psi_norm,
            new_norm,
        );
    }

    /// n = 3, 4 の dense H に対して `exp(-i dt H) · ψ` の参照計算と比較する.
    fn dense_propagator_match(n: usize, seed: u64) {
        let dim = 1usize << n;
        let mut rng = Xor64::new(seed);
        let h_x: Vec<f64> = (0..n).map(|_| rng.signed()).collect();
        let h_p_diag: Vec<f64> = (0..dim).map(|_| rng.signed()).collect();
        let psi = random_complex_vec(dim, seed.wrapping_mul(17));
        let dt = 0.25_f64;
        let a_t = rng.signed();
        let b_t = rng.signed();

        let matvec = |v: &[Complex64], y: &mut [Complex64]| {
            apply_h_kryanneal(v, y, &h_x, &h_p_diag, a_t, b_t, n);
        };
        // dim 以上の m は意味が無いので min(24, dim).
        let m = std::cmp::min(24, dim);
        let (result, _m_eff, _beta_m, _c_m_abs) =
            lanczos_propagate(matvec, &psi, dt, m, 1e-14).expect("ok");

        let h_real = build_dense_h_real(n, &h_x, &h_p_diag, a_t, b_t);
        let expected = reference_propagate_real_h(&h_real, &psi, dt);

        let rel = relative_error(&result, &expected);
        assert!(rel < 1e-12, "n={}, seed={}: rel = {}", n, seed, rel);
    }

    #[test]
    fn dense_propagator_match_n3() {
        for seed in [1_u64, 7, 0x1234] {
            dense_propagator_match(3, seed);
        }
    }

    #[test]
    fn dense_propagator_match_n4() {
        for seed in [2_u64, 11, 0xbeef] {
            dense_propagator_match(4, seed);
        }
    }
}
