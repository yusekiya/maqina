//! 複素ベクトル Level-1 / Level-2 BLAS の薄いラッパ.
//!
//! Lanczos (`src/krylov.rs`) / CFM4:2 (`src/cfm4.rs`, Phase 3 以降) の dim 依存
//! ops を共通インタフェースで書くため, `cblas` クレート (`feature = "blas"`)
//! と純 Rust scalar 経路の双方を `#[cfg]` で切替える. 両経路は同一の数値結果
//! を `rel < 1e-13` で返す契約 (`docs/design.md` §7.2 / §7.4).
//!
//! BLAS feature ON の場合は target ごとに自動選択された backend
//! (macOS = Apple Accelerate, Linux = system OpenBLAS) を経由する. リンカに
//! シンボルを引かせるための `extern crate blas_src;` 相当は `src/lib.rs` 側で
//! `use blas_src as _;` として保持する.
//!
//! Phase 1 の matvec primitive 単独着地時は本モジュールが未参照になるため
//! `dead_code` lint を許容する (`src/matvec.rs`, `src/tridiag.rs` と同じ運用).

#![allow(dead_code)]

use num_complex::Complex64;

#[cfg(feature = "blas")]
use cblas::{dznrm2, zaxpy, zdotc_sub, zdscal, zgemv, Layout, Transpose};

/// `y += alpha · x` (`cblas::zaxpy` 相当).
///
/// # Panics
/// `x.len() != y.len()`
pub(crate) fn axpy(alpha: Complex64, x: &[Complex64], y: &mut [Complex64]) {
    assert_eq!(x.len(), y.len(), "axpy: x and y must have the same length");
    let n = x.len();

    #[cfg(feature = "blas")]
    {
        if n == 0 {
            return;
        }
        // cblas::zaxpy(n, alpha, x, incx, y, incy).
        // unsafe は cblas が C ABI で extern している (BLAS の契約) ため.
        unsafe {
            zaxpy(n as i32, alpha, x, 1, y, 1);
        }
    }

    #[cfg(not(feature = "blas"))]
    {
        for k in 0..n {
            y[k] += alpha * x[k];
        }
    }
}

/// `<x | y> = Σ conj(x_k) · y_k` (`cblas::zdotc_sub` 相当).
///
/// `zdotc` の戻り値で AB 不整合を踏むビルドが過去にあったため, `_sub` 版
/// (戻り値を out-parameter で受け取る) を使う.
///
/// # Panics
/// `x.len() != y.len()`
pub(crate) fn dot_conj(x: &[Complex64], y: &[Complex64]) -> Complex64 {
    assert_eq!(
        x.len(),
        y.len(),
        "dot_conj: x and y must have the same length"
    );
    let n = x.len();

    #[cfg(feature = "blas")]
    {
        if n == 0 {
            return Complex64::new(0.0, 0.0);
        }
        // cblas crate は dotc を `&mut [c64]` (長さ 1) で受け取る形に
        // ラップしているので, 1 要素のスタック上スライスを渡す.
        let mut result = [Complex64::new(0.0, 0.0)];
        unsafe {
            zdotc_sub(n as i32, x, 1, y, 1, &mut result);
        }
        result[0]
    }

    #[cfg(not(feature = "blas"))]
    {
        let mut s = Complex64::new(0.0, 0.0);
        for k in 0..n {
            s += x[k].conj() * y[k];
        }
        s
    }
}

/// `||x||_2` (`cblas::dznrm2` 相当).
pub(crate) fn nrm2(x: &[Complex64]) -> f64 {
    #[cfg(feature = "blas")]
    {
        let n = x.len();
        if n == 0 {
            return 0.0;
        }
        unsafe { dznrm2(n as i32, x, 1) }
    }

    #[cfg(not(feature = "blas"))]
    {
        // 単純な Σ|x_k|² ループで sqrt. n ≤ 数千万 の dim でも overflow
        // しないため hypot scan は不要 (BLAS 経路と異なり過剰精度は狙わない).
        let mut s = 0.0_f64;
        for &v in x.iter() {
            s += v.norm_sqr();
        }
        s.sqrt()
    }
}

/// `x *= alpha` (`alpha` は実スカラ, `cblas::zdscal` 相当).
pub(crate) fn scal_real(alpha: f64, x: &mut [Complex64]) {
    #[cfg(feature = "blas")]
    {
        let n = x.len();
        if n == 0 {
            return;
        }
        unsafe {
            zdscal(n as i32, alpha, x, 1);
        }
    }

    #[cfg(not(feature = "blas"))]
    {
        for v in x.iter_mut() {
            *v = Complex64::new(v.re * alpha, v.im * alpha);
        }
    }
}

/// `y = A · x` (`A` は column-major `(rows × cols)`, `cblas::zgemv(NoTrans)`
/// の `alpha=1, beta=0` 相当).
///
/// Lanczos の終端再構成 `psi_new = V[:, :m_eff] @ c` を `m_eff` 回の axpy
/// ではなく 1 度の Level-2 BLAS 呼出で行うために使う (`docs/design.md` §5.2).
///
/// # Panics
/// - `a.len() != rows * cols`
/// - `x.len() != cols`
/// - `y.len() != rows`
pub(crate) fn gemv_col_major(
    a: &[Complex64],
    rows: usize,
    cols: usize,
    x: &[Complex64],
    y: &mut [Complex64],
) {
    assert_eq!(a.len(), rows * cols, "gemv: a must have rows*cols entries");
    assert_eq!(x.len(), cols, "gemv: x.len() must equal cols");
    assert_eq!(y.len(), rows, "gemv: y.len() must equal rows");

    if rows == 0 {
        return;
    }

    #[cfg(feature = "blas")]
    {
        if cols == 0 {
            // y = 0 · y + 1 · (空和) → ゼロ初期化.
            for slot in y.iter_mut() {
                *slot = Complex64::new(0.0, 0.0);
            }
            return;
        }
        let alpha = Complex64::new(1.0, 0.0);
        let beta = Complex64::new(0.0, 0.0);
        // lda = rows (column-major で各列は連続 rows 要素).
        unsafe {
            zgemv(
                Layout::ColumnMajor,
                Transpose::None,
                rows as i32,
                cols as i32,
                alpha,
                a,
                rows as i32,
                x,
                1,
                beta,
                y,
                1,
            );
        }
    }

    #[cfg(not(feature = "blas"))]
    {
        for slot in y.iter_mut() {
            *slot = Complex64::new(0.0, 0.0);
        }
        for j in 0..cols {
            let x_j = x[j];
            let col = &a[j * rows..(j + 1) * rows];
            for k in 0..rows {
                y[k] += x_j * col[k];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 軽量決定論的 PRNG (xorshift64). `src/matvec.rs` / `src/tridiag.rs` の
    /// テストと同じ実装. 共通化は将来課題.
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

    /// scalar 経路の参照実装. cargo test の cfg からも常に call できるよう
    /// 公開ラッパとは独立に閉じた実装を残す.
    fn axpy_scalar(alpha: Complex64, x: &[Complex64], y: &mut [Complex64]) {
        for k in 0..x.len() {
            y[k] += alpha * x[k];
        }
    }

    fn dot_conj_scalar(x: &[Complex64], y: &[Complex64]) -> Complex64 {
        let mut s = Complex64::new(0.0, 0.0);
        for k in 0..x.len() {
            s += x[k].conj() * y[k];
        }
        s
    }

    fn nrm2_scalar(x: &[Complex64]) -> f64 {
        let mut s = 0.0_f64;
        for &v in x.iter() {
            s += v.norm_sqr();
        }
        s.sqrt()
    }

    fn scal_real_scalar(alpha: f64, x: &mut [Complex64]) {
        for v in x.iter_mut() {
            *v = Complex64::new(v.re * alpha, v.im * alpha);
        }
    }

    fn gemv_scalar(a: &[Complex64], rows: usize, cols: usize, x: &[Complex64]) -> Vec<Complex64> {
        let mut y = vec![Complex64::new(0.0, 0.0); rows];
        for j in 0..cols {
            let x_j = x[j];
            let col = &a[j * rows..(j + 1) * rows];
            for k in 0..rows {
                y[k] += x_j * col[k];
            }
        }
        y
    }

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

    #[test]
    fn axpy_matches_scalar_reference() {
        for &n in &[0_usize, 1, 7, 64, 1024] {
            for seed in [1_u64, 17, 0xdead_beef] {
                let x = random_complex_vec(n, seed);
                let y0 = random_complex_vec(n, seed.wrapping_add(1));
                let alpha = Complex64::new(0.7, -0.3);

                let mut y_blas = y0.clone();
                axpy(alpha, &x, &mut y_blas);

                let mut y_ref = y0.clone();
                axpy_scalar(alpha, &x, &mut y_ref);

                let rel = relative_error(&y_blas, &y_ref);
                assert!(rel < 1e-13, "axpy n={}, seed={}: rel = {}", n, seed, rel);
            }
        }
    }

    #[test]
    fn dot_conj_matches_scalar_reference() {
        for &n in &[0_usize, 1, 7, 64, 1024] {
            for seed in [2_u64, 19, 0xcafe_babe] {
                let x = random_complex_vec(n, seed);
                let y = random_complex_vec(n, seed.wrapping_add(11));

                let d_blas = dot_conj(&x, &y);
                let d_ref = dot_conj_scalar(&x, &y);

                let denom = d_ref.norm().max(1.0);
                let rel = (d_blas - d_ref).norm() / denom;
                assert!(
                    rel < 1e-13,
                    "dot_conj n={}, seed={}: rel = {} (blas={:?}, ref={:?})",
                    n,
                    seed,
                    rel,
                    d_blas,
                    d_ref,
                );
            }
        }
    }

    #[test]
    fn nrm2_matches_scalar_reference() {
        for &n in &[0_usize, 1, 7, 64, 1024] {
            for seed in [3_u64, 23, 0x1234_5678] {
                let x = random_complex_vec(n, seed);

                let r_blas = nrm2(&x);
                let r_ref = nrm2_scalar(&x);

                let denom = r_ref.abs().max(1.0);
                let rel = (r_blas - r_ref).abs() / denom;
                assert!(
                    rel < 1e-13,
                    "nrm2 n={}, seed={}: rel = {} (blas={}, ref={})",
                    n,
                    seed,
                    rel,
                    r_blas,
                    r_ref,
                );
            }
        }
    }

    #[test]
    fn scal_real_matches_scalar_reference() {
        for &n in &[0_usize, 1, 7, 64, 1024] {
            for seed in [4_u64, 29, 0xaabb_ccdd] {
                let x0 = random_complex_vec(n, seed);
                let alpha = 0.375_f64;

                let mut x_blas = x0.clone();
                scal_real(alpha, &mut x_blas);

                let mut x_ref = x0.clone();
                scal_real_scalar(alpha, &mut x_ref);

                let rel = relative_error(&x_blas, &x_ref);
                assert!(
                    rel < 1e-13,
                    "scal_real n={}, seed={}: rel = {}",
                    n,
                    seed,
                    rel
                );
            }
        }
    }

    #[test]
    fn gemv_col_major_matches_scalar_reference() {
        // rows × cols のスイープ. rows=0 / cols=0 のコーナーも含める.
        for &(rows, cols) in &[(0, 0), (1, 1), (3, 5), (8, 4), (64, 24), (1024, 8)] {
            for seed in [5_u64, 31, 0xfeed_face] {
                let a = random_complex_vec(rows * cols, seed);
                let x = random_complex_vec(cols, seed.wrapping_add(7));

                let mut y_blas = vec![Complex64::new(0.0, 0.0); rows];
                gemv_col_major(&a, rows, cols, &x, &mut y_blas);

                let y_ref = gemv_scalar(&a, rows, cols, &x);

                let rel = relative_error(&y_blas, &y_ref);
                assert!(
                    rel < 1e-13,
                    "gemv rows={}, cols={}, seed={}: rel = {}",
                    rows,
                    cols,
                    seed,
                    rel,
                );
            }
        }
    }
}
