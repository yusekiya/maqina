//! `cfm4.rs`: 中点則 M2 / CFM4:2 / Richardson 推定子.
//!
//! Phase 1 では **M2 中点則 1 step** のみを実装する.
//!
//! ```text
//! U(t+dt, t) ≈ exp(-i dt · H(t + dt/2))
//! ```
//!
//! 中点で H をフリーズし `lanczos_propagate` を 1 回呼ぶだけの薄いラッパで,
//! LTE ~ O(dt^3). 詳細は `docs/design.md` §5.3 M2 サブセクション.
//!
//! Phase 3 で `cfm4_step` (Alvermann-Fehske 2011 の 4 次 commutator-free
//! Magnus), Phase 4 で `cfm4_step_with_m2_estimate` / `cfm4_step_with_
//! richardson_estimate` を本ファイルに追加する.
//!
//! 本関数は `lanczos_propagate` を介して Python に状態を返す **Phase 1 の
//! 公開プロパゲータ** であり, PyO3 wrap `m2_midpoint_step_py` 経由で
//! `_rust.m2_midpoint_step_py` として exposure される. `lanczos_propagate`
//! 自身は `pub(crate)` のままで, M2 / CFM4:2 が上位 wrap として公開する
//! 設計 (`docs/design.md` §5.2 末尾).
//!
//! PyO3 の `wrap_pyfunction!` 経由で `_rust` module に登録される関数は
//! Rust の dead_code 解析からは「呼ばれていない」と見えるため, matvec.rs /
//! krylov.rs と同様に module 全体で lint を抑制する (Phase 1 で内部
//! caller がいない `m2_midpoint_step` 本体にも同じ抑制が必要).

#![allow(dead_code)]

use num_complex::Complex64;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::krylov::lanczos_propagate;
use crate::matvec::apply_h_kryanneal;

/// `psi_new = exp(-i dt · H(t + dt/2)) · psi` を中点則で計算する.
///
/// 時間依存 `H(t) = A(s(t)) · H_driver + B(s(t)) · H_problem` の中点で
/// スケジュール係数を凍結し, `apply_h_kryanneal(·, ·, h_x, h_p_diag, a_mid,
/// b_mid, n)` を closure として `lanczos_propagate` に渡す.
///
/// # 引数
/// * `psi` (length `2^n`): 入力状態.
/// * `h_x` (length `n`): サイト依存横磁場振幅.
/// * `h_p_diag` (length `2^n`): Z 基底での `H_problem` 対角ベクトル.
/// * `a_mid`: 中点でのドライバ係数 `A(s(t + dt/2))` (Python 側で計算済).
/// * `b_mid`: 中点での problem 係数 `B(s(t + dt/2))`.
/// * `dt`: 時刻刻み幅 (real).
/// * `m`: Krylov 部分空間次元 (典型値 24).
/// * `krylov_tol`: Lanczos の β 打切り閾値.
/// * `n`: サイト数. `dim = 2^n` を呼出側と一意に決める.
///
/// # 戻り値
/// * `Ok(psi_new)`: 長さ `2^n` の新状態.
/// * `Err`: `lanczos_propagate` 内で tridiag 固有分解が収束しなかった場合.
///
/// # Panics
/// `lanczos_propagate` / `apply_h_kryanneal` の precondition と同じ
/// (長さ不整合, `m == 0`).
//
// 数値カーネル primitive は cv_ising 流に引数フラットで持つ. 構造体化は
// 将来の `cfm4_step` 系で引数が更に増えた段階で再検討する.
#[allow(clippy::too_many_arguments)]
pub(crate) fn m2_midpoint_step(
    psi: &[Complex64],
    h_x: &[f64],
    h_p_diag: &[f64],
    a_mid: f64,
    b_mid: f64,
    dt: f64,
    m: usize,
    krylov_tol: f64,
    n: usize,
) -> PyResult<Vec<Complex64>> {
    let matvec = |v: &[Complex64], y: &mut [Complex64]| {
        apply_h_kryanneal(v, y, h_x, h_p_diag, a_mid, b_mid, n);
    };
    lanczos_propagate(matvec, psi, dt, m, krylov_tol)
}

/// `m2_midpoint_step` の Python wrap.
///
/// Python 側 (`_rust.m2_midpoint_step_py`) からは
///
/// ```python
/// psi_new = _rust.m2_midpoint_step_py(
///     psi, h_x, h_p_diag, a_mid, b_mid, dt, m, krylov_tol,
/// )
/// ```
///
/// として呼ぶ. サイト数 `n = len(h_x)` / 状態次元 `dim = 2^n` は
/// `len(h_p_diag)` から取り出し, 整合性を検証する.
#[pyfunction]
#[pyo3(signature = (psi, h_x, h_p_diag, a_mid, b_mid, dt, m, krylov_tol))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn m2_midpoint_step_py<'py>(
    py: Python<'py>,
    psi: PyReadonlyArray1<'py, Complex64>,
    h_x: PyReadonlyArray1<'py, f64>,
    h_p_diag: PyReadonlyArray1<'py, f64>,
    a_mid: f64,
    b_mid: f64,
    dt: f64,
    m: usize,
    krylov_tol: f64,
) -> PyResult<Bound<'py, PyArray1<Complex64>>> {
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

    let psi_new = m2_midpoint_step(
        psi_slice,
        h_x_slice,
        h_p_diag_slice,
        a_mid,
        b_mid,
        dt,
        m,
        krylov_tol,
        n,
    )?;
    Ok(psi_new.into_pyarray(py))
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, SymmetricEigen};

    use crate::blas::nrm2;

    /// 軽量決定論的 PRNG (xorshift64). matvec / krylov のテストと同実装を
    /// 再掲 (Rust 単体テストは同じファイル内に置く方針, CLAUDE.md 参照).
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
    /// 構築する. H_driver (X 演算子の和) も diag も実係数のため H 全体は
    /// 実対称.
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

        let mut c = vec![Complex64::new(0.0, 0.0); dim];
        for j in 0..dim {
            let mut s = Complex64::new(0.0, 0.0);
            for l in 0..dim {
                s += Complex64::new(u[(l, j)], 0.0) * psi[l];
            }
            let phase = Complex64::new(0.0, -dt * lam[j]).exp();
            c[j] = s * phase;
        }
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

    /// dt = 0 で恒等変換になる: `exp(-i · 0 · H) · ψ = ψ`.
    /// Lanczos は dt = 0 のとき位相 `exp(0) = 1` を返すので, 部分空間内の
    /// 数値誤差のみが残る. rel < 1e-13 で一致するはず.
    #[test]
    fn dt_zero_is_identity() {
        let n = 4_usize;
        let dim = 1usize << n;
        let mut rng = Xor64::new(31);
        let h_x: Vec<f64> = (0..n).map(|_| rng.signed()).collect();
        let h_p_diag: Vec<f64> = (0..dim).map(|_| rng.signed()).collect();
        let psi = random_complex_vec(dim, 71);
        let a_mid = rng.signed();
        let b_mid = rng.signed();

        let result =
            m2_midpoint_step(&psi, &h_x, &h_p_diag, a_mid, b_mid, 0.0, 24, 1e-12, n).expect("ok");
        let rel = relative_error(&result, &psi);
        assert!(rel < 1e-13, "dt=0 rel = {}", rel);
    }

    /// `exp(-i dt H)` は unitary なので ‖ψ_new‖ = ‖ψ‖.
    #[test]
    fn hermitian_h_preserves_norm() {
        let n = 4_usize;
        let dim = 1usize << n;
        let mut rng = Xor64::new(127);
        let h_x: Vec<f64> = (0..n).map(|_| rng.signed()).collect();
        let h_p_diag: Vec<f64> = (0..dim).map(|_| rng.signed()).collect();
        let psi = random_complex_vec(dim, 211);
        let psi_norm = nrm2(&psi);
        let a_mid = 0.4_f64;
        let b_mid = 1.1_f64;
        let dt = 0.3_f64;

        let result =
            m2_midpoint_step(&psi, &h_x, &h_p_diag, a_mid, b_mid, dt, 24, 1e-12, n).expect("ok");
        let new_norm = nrm2(&result);
        let rel = (new_norm - psi_norm).abs() / psi_norm.max(1.0);
        assert!(
            rel < 1e-13,
            "norm rel = {} (before={}, after={})",
            rel,
            psi_norm,
            new_norm,
        );
    }

    /// time-independent H に対しチェイン適用すると
    /// `m2(dt) ∘ m2(dt) ∘ ... ∘ m2(dt) = exp(-i n_steps·dt · H)` に
    /// **数値的に** 一致するはず (中点則は H が定数なら厳密に
    /// `exp(-i dt H)` を返し, Lanczos 誤差のみ残る).
    /// n_steps=100, T=1, n=4 で rel < 1e-10 を要求 (issue 仕様).
    #[test]
    fn time_independent_h_matches_exact_chain() {
        let n = 4_usize;
        let dim = 1usize << n;
        let mut rng = Xor64::new(0x1234);
        let h_x: Vec<f64> = (0..n).map(|_| rng.signed()).collect();
        let h_p_diag: Vec<f64> = (0..dim).map(|_| rng.signed()).collect();
        let psi0 = random_complex_vec(dim, 0xbeef);

        let a_t = rng.signed();
        let b_t = rng.signed();

        let n_steps = 100_usize;
        let total_t = 1.0_f64;
        let dt = total_t / n_steps as f64;

        let mut psi = psi0.clone();
        for _ in 0..n_steps {
            psi = m2_midpoint_step(&psi, &h_x, &h_p_diag, a_t, b_t, dt, 24, 1e-14, n).expect("ok");
        }

        let h_real = build_dense_h_real(n, &h_x, &h_p_diag, a_t, b_t);
        let expected = reference_propagate_real_h(&h_real, &psi0, total_t);

        let rel = relative_error(&psi, &expected);
        assert!(rel < 1e-10, "chain n_steps={} rel = {}", n_steps, rel);
    }
}
