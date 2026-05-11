//! `apply_h_kryanneal`: matvec primitive (bit-flip driver + 対角 problem).
//!
//! 横磁場イジングモデル
//!
//! ```text
//! H(t) = A(s(t)) · H_driver + B(s(t)) · H_problem
//! H_driver  = -Σ_i h_x_i X_i              (サイト依存横磁場, bit-flip)
//! H_problem = Z 基底で対角 (diag(H_p_diag))
//! ```
//!
//! の Hamiltonian を計算ベクトル `v` に対し `y = H(t) v` の形で 1 回 apply
//! する低レベル primitive. Lanczos (m 回) や CFM4:2 (各 stage) から繰り返し
//! 呼ばれる. 詳細は `docs/design.md` §5.1.1.
//!
//! Phase 1 はスカラ単スレッド実装. SIMD ヒント / unroll は入れない (Phase 6
//! で rayon + SIMD + cache block-fusion を載せる).
//!
//! matvec primitive が単体で landed する Phase では Lanczos / CFM4 から呼ばれ
//! ないため `pub(crate)` 項目が未参照になる. Lanczos 着地までの間は
//! `dead_code` lint を許容する.

#![allow(dead_code)]

use num_complex::Complex64;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// `y = a_t · H_driver · v + b_t · diag(H_p_diag) · v` を計算する.
///
/// `H_driver = -Σ_i h_x_i X_i` (サイト依存横磁場の inhomogeneous 拡張).
///
/// # 入出力
/// - `v` (length `2^n`): 入力状態ベクトル.
/// - `y` (length `2^n`): 結果を **上書き** する出力バッファ. `v` と alias し
///   てはならない.
/// - `h_x` (length `n`): サイトごとの横磁場振幅.
/// - `h_p_diag` (length `2^n`): Z 基底での `H_problem` 対角ベクトル.
/// - `a_t`, `b_t`: 時刻 `t` でのスケジュール係数 `A(s(t))`, `B(s(t))`.
/// - `n`: サイト数. `dim = 2^n` を呼び出し側と一意に決める.
///
/// # アルゴリズム
/// 1. 対角部分: `y[k] = b_t · H_p_diag[k] · v[k]` を全 `k` に上書き.
/// 2. bit-flip 部分: 各サイト `i` について `coeff = -a_t · h_x[i]` と
///    `mask = 1 << i` を用い, `y[k] += coeff · v[k ^ mask]` を全 `k` で
///    accumulate.
///
/// Phase 1 はシンプルな `for k in 0..dim` を維持する. inner-loop 最適化
/// (cache-blocking / SIMD / rayon) は Phase 6 で導入する.
///
/// # Panics
/// - `v.len() != 1 << n`
/// - `y.len() != 1 << n`
/// - `h_x.len() != n`
/// - `h_p_diag.len() != 1 << n`
pub(crate) fn apply_h_kryanneal(
    v: &[Complex64],
    y: &mut [Complex64],
    h_x: &[f64],
    h_p_diag: &[f64],
    a_t: f64,
    b_t: f64,
    n: usize,
) {
    let dim = 1usize << n;
    assert_eq!(v.len(), dim, "v must have length 2^n");
    assert_eq!(y.len(), dim, "y must have length 2^n");
    assert_eq!(h_x.len(), n, "h_x must have length n");
    assert_eq!(h_p_diag.len(), dim, "h_p_diag must have length 2^n");

    // 対角部分: y[k] = b_t · H_p_diag[k] · v[k] (上書き).
    for k in 0..dim {
        y[k] = Complex64::new(b_t * h_p_diag[k], 0.0) * v[k];
    }
    // bit-flip 部分: y[k] += -a_t · h_x[i] · v[k ^ mask] を i について accumulate.
    for (i, &h_x_i) in h_x.iter().enumerate() {
        let coeff = -a_t * h_x_i;
        let mask = 1usize << i;
        for k in 0..dim {
            y[k] += Complex64::new(coeff, 0.0) * v[k ^ mask];
        }
    }
}

/// `apply_h_kryanneal` の Python wrap. `y` を allocate して返す.
///
/// Python 側 (`_rust.apply_h_kryanneal_py`) からは
///
/// ```python
/// y = _rust.apply_h_kryanneal_py(v, h_x, h_p_diag, a_t, b_t)
/// ```
///
/// として呼ぶ. サイト数 `n` は `len(h_x)`, 状態次元 `dim = 2^n` は
/// `len(h_p_diag)` から取り出す. Lanczos / CFM4 内部呼出は Rust 側で
/// 完結するため, 本関数は **参照実装比較とテスト用** の公開 API である
/// (`docs/design.md` §7.3).
#[pyfunction]
#[pyo3(signature = (v, h_x, h_p_diag, a_t, b_t))]
pub(crate) fn apply_h_kryanneal_py<'py>(
    py: Python<'py>,
    v: PyReadonlyArray1<'py, Complex64>,
    h_x: PyReadonlyArray1<'py, f64>,
    h_p_diag: PyReadonlyArray1<'py, f64>,
    a_t: f64,
    b_t: f64,
) -> PyResult<Bound<'py, PyArray1<Complex64>>> {
    let v_slice = v.as_slice()?;
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
    if v_slice.len() != dim {
        return Err(PyValueError::new_err(format!(
            "v length {} does not match 2^len(h_x) = {}",
            v_slice.len(),
            dim,
        )));
    }

    let mut y = vec![Complex64::new(0.0, 0.0); dim];
    apply_h_kryanneal(v_slice, &mut y, h_x_slice, h_p_diag_slice, a_t, b_t, n);
    Ok(y.into_pyarray(py))
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector};

    /// 軽量決定論的 PRNG (xorshift64). テスト用途のみ. `src/tridiag.rs` と
    /// 同じ実装を再掲する (両者を共有モジュールに括る判断は将来の課題).
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

        /// 一様分布 `[-1, 1)`.
        fn signed(&mut self) -> f64 {
            const SCALE: f64 = 1.0 / (1u64 << 53) as f64;
            let u = (self.next_u64() >> 11) as f64 * SCALE;
            2.0 * u - 1.0
        }

        fn complex_signed(&mut self) -> Complex64 {
            Complex64::new(self.signed(), self.signed())
        }
    }

    /// `(a_t, b_t, h_x, h_p_diag)` から `dim × dim` dense Hamiltonian を構築.
    /// 比較用参照実装 (nalgebra).
    fn build_dense_h(
        n: usize,
        h_x: &[f64],
        h_p_diag: &[f64],
        a_t: f64,
        b_t: f64,
    ) -> DMatrix<Complex64> {
        let dim = 1usize << n;
        let mut h = DMatrix::<Complex64>::zeros(dim, dim);
        // 対角 problem 項: B · H_p_diag[k]
        for k in 0..dim {
            h[(k, k)] = Complex64::new(b_t * h_p_diag[k], 0.0);
        }
        // 駆動項: -A · Σ_i h_x[i] · X_i
        for (i, &h_x_i) in h_x.iter().enumerate() {
            let coeff = -a_t * h_x_i;
            let mask = 1usize << i;
            for k in 0..dim {
                let kf = k ^ mask;
                h[(k, kf)] += Complex64::new(coeff, 0.0);
            }
        }
        h
    }

    /// `||y_actual - y_expected|| / max(||y_expected||, 1)` (相対誤差).
    fn relative_error(actual: &[Complex64], expected: &DVector<Complex64>) -> f64 {
        assert_eq!(actual.len(), expected.len());
        let mut diff_sq = 0.0_f64;
        let mut ref_sq = 0.0_f64;
        for k in 0..actual.len() {
            let d = actual[k] - expected[k];
            diff_sq += d.norm_sqr();
            ref_sq += expected[k].norm_sqr();
        }
        (diff_sq.sqrt()) / (ref_sq.sqrt().max(1.0))
    }

    /// Random スカラ `(a_t, b_t)` / ランダム `h_x, h_p_diag` / ランダム複素 `v`
    /// を作り, dense H · v との一致を rel < 1e-13 で検証する.
    fn dense_equivalence_check(n: usize, seed: u64) {
        let dim = 1usize << n;
        let mut rng = Xor64::new(seed);

        let h_x: Vec<f64> = (0..n).map(|_| rng.signed()).collect();
        let h_p_diag: Vec<f64> = (0..dim).map(|_| rng.signed()).collect();
        let v: Vec<Complex64> = (0..dim).map(|_| rng.complex_signed()).collect();
        let a_t = rng.signed();
        let b_t = rng.signed();

        // 参照: dense H · v.
        let h_dense = build_dense_h(n, &h_x, &h_p_diag, a_t, b_t);
        let v_vec = DVector::<Complex64>::from_vec(v.clone());
        let y_expected = &h_dense * &v_vec;

        // 被テスト: apply_h_kryanneal.
        let mut y = vec![Complex64::new(0.0, 0.0); dim];
        apply_h_kryanneal(&v, &mut y, &h_x, &h_p_diag, a_t, b_t, n);

        let rel = relative_error(&y, &y_expected);
        assert!(
            rel < 1e-13,
            "n={}, seed={}: relative error {} >= 1e-13",
            n,
            seed,
            rel,
        );
    }

    #[test]
    fn dense_equivalence_small_n() {
        // n=3..=6 を複数 seed で検証. dim = 8, 16, 32, 64.
        for n in 3..=6 {
            for seed in [1, 2, 17, 0xdead_beef] {
                dense_equivalence_check(n, seed);
            }
        }
    }

    #[test]
    fn zero_h_x_reduces_to_diag() {
        // h_x = 0 で y = b·diag(H_p_diag)·v に厳密一致.
        let n = 5;
        let dim = 1usize << n;
        let mut rng = Xor64::new(42);
        let h_x = vec![0.0_f64; n];
        let h_p_diag: Vec<f64> = (0..dim).map(|_| rng.signed()).collect();
        let v: Vec<Complex64> = (0..dim).map(|_| rng.complex_signed()).collect();
        let a_t = rng.signed();
        let b_t = rng.signed();

        let mut y = vec![Complex64::new(0.0, 0.0); dim];
        apply_h_kryanneal(&v, &mut y, &h_x, &h_p_diag, a_t, b_t, n);

        // 期待値: y[k] = b·H_p[k]·v[k].
        for k in 0..dim {
            let expected = Complex64::new(b_t * h_p_diag[k], 0.0) * v[k];
            let diff = (y[k] - expected).norm();
            assert!(
                diff < 1e-15 * (expected.norm() + 1.0),
                "k={}: actual {} vs expected {}",
                k,
                y[k],
                expected,
            );
        }
    }

    #[test]
    fn zero_h_p_diag_reduces_to_driver() {
        // H_p_diag = 0 で y = -a·Σ_i h_x[i]·v[k^mask] に一致.
        let n = 4;
        let dim = 1usize << n;
        let mut rng = Xor64::new(7);
        let h_x: Vec<f64> = (0..n).map(|_| rng.signed()).collect();
        let h_p_diag = vec![0.0_f64; dim];
        let v: Vec<Complex64> = (0..dim).map(|_| rng.complex_signed()).collect();
        let a_t = rng.signed();
        let b_t = rng.signed(); // 出力には影響しないが渡しておく.

        let mut y = vec![Complex64::new(0.0, 0.0); dim];
        apply_h_kryanneal(&v, &mut y, &h_x, &h_p_diag, a_t, b_t, n);

        // 参照: dense H_driver = -A · Σ_i h_x[i] · X_i.
        let h_dense = build_dense_h(n, &h_x, &h_p_diag, a_t, 0.0);
        let y_expected = &h_dense * DVector::<Complex64>::from_vec(v.clone());
        let rel = relative_error(&y, &y_expected);
        assert!(rel < 1e-13, "rel = {}", rel);
    }

    #[test]
    fn matches_dense_for_n1() {
        // 退化ケース n=1 (dim=2). 単純な 2x2 行列で手検算可能.
        // H = [[B·h_p[0], -A·h_x[0]],
        //      [-A·h_x[0], B·h_p[1]]]
        let n = 1;
        let h_x = [0.3_f64];
        let h_p_diag = [1.5_f64, -2.5_f64];
        let v = [Complex64::new(0.7, -0.2), Complex64::new(-0.4, 0.9)];
        let a_t = 0.6;
        let b_t = 1.1;

        let mut y = [Complex64::new(0.0, 0.0); 2];
        apply_h_kryanneal(&v, &mut y, &h_x, &h_p_diag, a_t, b_t, n);

        let off = -a_t * h_x[0];
        let expected = [
            Complex64::new(b_t * h_p_diag[0], 0.0) * v[0] + Complex64::new(off, 0.0) * v[1],
            Complex64::new(b_t * h_p_diag[1], 0.0) * v[1] + Complex64::new(off, 0.0) * v[0],
        ];
        for k in 0..2 {
            let diff = (y[k] - expected[k]).norm();
            assert!(
                diff < 1e-15,
                "k={}: y={}, expected={}",
                k,
                y[k],
                expected[k]
            );
        }
    }
}
