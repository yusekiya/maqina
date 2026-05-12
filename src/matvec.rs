//! matvec / single-mode-axis primitives.
//!
//! 横磁場イジングモデル
//!
//! ```text
//! H(t) = A(s(t)) · H_driver + B(s(t)) · H_problem
//! H_driver  = -Σ_i h_x_i X_i              (サイト依存横磁場, bit-flip)
//! H_problem = Z 基底で対角 (diag(H_p_diag))
//! ```
//!
//! 本モジュールは以下の bit-flip pass 系 primitive を提供する:
//!
//! - [`apply_h_kryanneal`]: 計算ベクトル `v` に対し `y = H(t) v` を 1 回 apply
//!   する additive matvec. Lanczos (m 回) や CFM4:2 (各 stage) から繰り返し
//!   呼ばれる. 詳細は `docs/design.md` §5.1.1.
//! - [`apply_single_mode_axis_i`]: Trotter 経路で `R_i(θ) = cos(θ)·I + i·sin(θ)·X_i`
//!   のような 2×2 ユニタリ `U` を axis `i` のペア `(psi[k], psi[k^(1<<i)])`
//!   に **in-place** 適用する Phase 2 primitive. 詳細は `docs/design.md` §5.1.2.
//!
//! 両者を同じファイルに置くのは, 同型の bit-flip pass パターン (i 外側 / k 内側,
//! `mask = 1<<i` で stride を持つ走査) を共有するため. Phase 6 の cache
//! block-fusion 最適化が両者に均等に効く前提.
//!
//! Phase 1 / 2 はスカラ単スレッド実装. SIMD ヒント / unroll は入れない (Phase 6
//! で rayon + SIMD + cache block-fusion を載せる).
//!
//! Phase 2 で `apply_single_mode_axis_i_py` を `#[pyfunction]` として宣言するが,
//! `#[pymodule]` への登録は trotter 経路を整える C3 issue でまとめて行う.
//! このため一時的に `pub(crate)` 項目が外部から未参照になる. `dead_code`
//! lint をモジュール全体で許容する.

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

/// `psi` を axis `i` で 2 元化したペア `(psi[k], psi[k | mask])` に 2×2
/// ユニタリ `u` を **in-place** 適用する Phase 2 primitive.
///
/// `mask = 1 << i` とし, `bit_i(k) = 0` を満たす `k` (`= k_lo`) について
/// `k_hi = k_lo | mask` のペアを取り出し,
///
/// ```text
/// psi'[k_lo] = u[0]·psi[k_lo] + u[1]·psi[k_hi]
/// psi'[k_hi] = u[2]·psi[k_lo] + u[3]·psi[k_hi]
/// ```
///
/// で更新する. `u` は row-major 2×2 行列 `[[u00, u01], [u10, u11]]`. Trotter
/// 経路で `R_i(θ) = cos(θ)·I + i·sin(θ)·X_i` を渡すときは
/// `u = [c, i·s, i·s, c]` (`c = cos θ`, `s = sin θ`).
///
/// # 実装メモ
/// 設計書 §5.1.2 末尾の通り **2 重ループ形** で書く. 外側で長さ
/// `block = 1 << (i + 1)` のブロックを `base = 0, block, 2·block, ...`
/// と進め, 内側で `offset in 0..mask` を走り `(lo = base + offset,
/// hi = lo + mask)` のペアを直接処理する. これにより
/// `if k & mask != 0 { continue; }` の分岐スキップを完全に避けられ,
/// 内側ループは予測可能な連続アクセス + `mask` stride アクセスに揃う.
///
/// # 入出力
/// - `psi` (length `2^n`): in-place で更新される状態ベクトル.
/// - `u`: row-major 2×2 ユニタリ (本関数自体はユニタリ性を要求しないが,
///   呼び出し側は `‖psi‖` を保つために unitary を渡すのが通常).
/// - `i`: 適用するサイト index. `0 <= i < n`.
/// - `n`: サイト数. `dim = 2^n`.
///
/// # Panics
/// - `psi.len() != 1 << n`
/// - `i >= n`
pub(crate) fn apply_single_mode_axis_i(
    psi: &mut [Complex64],
    u: &[Complex64; 4],
    i: usize,
    n: usize,
) {
    let dim = 1usize << n;
    assert_eq!(psi.len(), dim, "psi must have length 2^n");
    assert!(i < n, "i={} must be < n={}", i, n);

    let mask = 1usize << i;
    let block = mask << 1; // 2 * mask
    let mut base = 0usize;
    while base < dim {
        for offset in 0..mask {
            let lo = base + offset;
            let hi = lo + mask;
            let a = psi[lo];
            let b = psi[hi];
            psi[lo] = u[0] * a + u[1] * b;
            psi[hi] = u[2] * a + u[3] * b;
        }
        base += block;
    }
}

/// `apply_single_mode_axis_i` の Python wrap. 結果を新規 array で返す
/// (in-place ではなく allocate-and-return パターン. `apply_h_kryanneal_py`
/// と統一).
///
/// Python 側 (C3 で `_rust.apply_single_mode_axis_i_py` として登録予定) は
///
/// ```python
/// psi_new = _rust.apply_single_mode_axis_i_py(psi, u, i, n)
/// ```
///
/// として呼ぶ. `u` は length 4 の `complex128` 配列 (row-major 2×2).
/// Trotter 経路の Rust 内部呼出は `apply_single_mode_axis_i` を直接使うため,
/// 本 wrap は **参照実装比較とテスト用** の公開 API である (`docs/design.md`
/// §7.3).
#[pyfunction]
#[pyo3(signature = (psi, u, i, n))]
pub(crate) fn apply_single_mode_axis_i_py<'py>(
    py: Python<'py>,
    psi: PyReadonlyArray1<'py, Complex64>,
    u: PyReadonlyArray1<'py, Complex64>,
    i: usize,
    n: usize,
) -> PyResult<Bound<'py, PyArray1<Complex64>>> {
    let psi_slice = psi.as_slice()?;
    let u_slice = u.as_slice()?;

    let dim = 1usize << n;
    if psi_slice.len() != dim {
        return Err(PyValueError::new_err(format!(
            "psi length {} does not match 2^n = 2^{} = {}",
            psi_slice.len(),
            n,
            dim,
        )));
    }
    if u_slice.len() != 4 {
        return Err(PyValueError::new_err(format!(
            "u must be a length-4 row-major 2x2 matrix, got length {}",
            u_slice.len(),
        )));
    }
    if i >= n {
        return Err(PyValueError::new_err(format!("i={} must be < n={}", i, n,)));
    }

    let u_arr: [Complex64; 4] = [u_slice[0], u_slice[1], u_slice[2], u_slice[3]];
    let mut out: Vec<Complex64> = psi_slice.to_vec();
    apply_single_mode_axis_i(&mut out, &u_arr, i, n);
    Ok(out.into_pyarray(py))
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

    // ===== apply_single_mode_axis_i のテスト =====

    /// `[-π, π)` の一様乱数.
    fn random_angle(rng: &mut Xor64) -> f64 {
        std::f64::consts::PI * rng.signed()
    }

    /// ランダムな 2×2 ユニタリ (U(2)) を `[u00, u01, u10, u11]` row-major で返す.
    ///
    /// `U = e^{iφ} · [[e^{iα} cos θ, e^{iβ} sin θ],
    ///                [-e^{-iβ} sin θ, e^{-iα} cos θ]]`
    /// と分解し, `θ ∈ [-π/4, π/4]` (混合角の代表値) と `α, β, φ` をランダム
    /// 化する. ユニタリ性は構成上 machine precision で保証される.
    fn random_unitary_2x2(rng: &mut Xor64) -> [Complex64; 4] {
        let theta = 0.25 * std::f64::consts::PI * rng.signed();
        let alpha = random_angle(rng);
        let beta = random_angle(rng);
        let phi = random_angle(rng);
        let (s, c) = theta.sin_cos();

        let e_phi = Complex64::from_polar(1.0, phi);
        let e_alpha = Complex64::from_polar(1.0, alpha);
        let e_beta = Complex64::from_polar(1.0, beta);
        let e_neg_alpha = Complex64::from_polar(1.0, -alpha);
        let e_neg_beta = Complex64::from_polar(1.0, -beta);

        [
            e_phi * e_alpha * Complex64::new(c, 0.0),
            e_phi * e_beta * Complex64::new(s, 0.0),
            e_phi * (-e_neg_beta) * Complex64::new(s, 0.0),
            e_phi * e_neg_alpha * Complex64::new(c, 0.0),
        ]
    }

    /// 設計書 §5.1.2 擬似コードの素朴版 (`k & mask != 0` の skip 形). 本実装
    /// (2 重ループ形) との数値一致確認用 reference.
    fn apply_single_mode_axis_i_skip(
        psi: &mut [Complex64],
        u: &[Complex64; 4],
        i: usize,
        n: usize,
    ) {
        let dim = 1usize << n;
        let mask = 1usize << i;
        let mut k = 0usize;
        while k < dim {
            if k & mask != 0 {
                k += 1;
                continue;
            }
            let a = psi[k];
            let b = psi[k | mask];
            psi[k] = u[0] * a + u[1] * b;
            psi[k | mask] = u[2] * a + u[3] * b;
            k += 1;
        }
    }

    /// dense reference: `dim × dim` の `I ⊗ ... ⊗ U_i ⊗ ... ⊗ I` 相当を直接
    /// 構築する. bit `i` を行/列で抜き出し, 以下のみ非零:
    ///
    /// - `U_full[k, k]      = u[0]` if `bit_i(k) = 0` else `u[3]`
    /// - `U_full[k, k^mask] = u[1]` if `bit_i(k) = 0` else `u[2]`
    ///
    /// Kronecker 順序の符号合わせをせずに直接表現するほうが
    /// `apply_single_mode_axis_i` の規約と 1:1 対応するため誤り混入しにくい.
    fn build_dense_single_mode(n: usize, u: &[Complex64; 4], i: usize) -> DMatrix<Complex64> {
        let dim = 1usize << n;
        let mask = 1usize << i;
        let mut m = DMatrix::<Complex64>::zeros(dim, dim);
        for k in 0..dim {
            if k & mask == 0 {
                m[(k, k)] = u[0];
                m[(k, k ^ mask)] = u[1];
            } else {
                m[(k, k)] = u[3];
                m[(k, k ^ mask)] = u[2];
            }
        }
        m
    }

    #[test]
    fn single_mode_identity_preserves_psi() {
        // u = I で psi が要素ごとに不変.
        let id: [Complex64; 4] = [
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ];
        for n in 1..=4 {
            let dim = 1usize << n;
            let mut rng = Xor64::new(0x1234_5678_9abc_def0 ^ n as u64);
            let psi0: Vec<Complex64> = (0..dim).map(|_| rng.complex_signed()).collect();
            for i in 0..n {
                let mut psi = psi0.clone();
                apply_single_mode_axis_i(&mut psi, &id, i, n);
                for k in 0..dim {
                    let diff = (psi[k] - psi0[k]).norm();
                    assert!(
                        diff < 1e-15,
                        "n={}, i={}, k={}: identity changed psi: {} -> {}",
                        n,
                        i,
                        k,
                        psi0[k],
                        psi[k],
                    );
                }
            }
        }
    }

    #[test]
    fn single_mode_preserves_norm_for_unitary() {
        // ランダム unitary U で ‖psi‖ が rel < 1e-13 で保たれる.
        for n in 1..=6 {
            let dim = 1usize << n;
            for seed in [3, 11, 29, 0xface_feed_u64] {
                let mut rng = Xor64::new(seed.wrapping_add(n as u64));
                let psi0: Vec<Complex64> = (0..dim).map(|_| rng.complex_signed()).collect();
                let norm0_sq: f64 = psi0.iter().map(|z| z.norm_sqr()).sum();

                for i in 0..n {
                    let u = random_unitary_2x2(&mut rng);
                    let mut psi = psi0.clone();
                    apply_single_mode_axis_i(&mut psi, &u, i, n);

                    let norm_sq: f64 = psi.iter().map(|z| z.norm_sqr()).sum();
                    let rel = (norm_sq - norm0_sq).abs() / norm0_sq.max(1.0);
                    assert!(
                        rel < 1e-13,
                        "n={}, i={}, seed={}: norm not preserved (‖psi0‖^2={}, ‖psi‖^2={}, rel={})",
                        n,
                        i,
                        seed,
                        norm0_sq,
                        norm_sq,
                        rel,
                    );
                }
            }
        }
    }

    #[test]
    fn single_mode_matches_dense_kronecker() {
        // n ∈ {2, 3, 4}, i ∈ {0, ..., n-1} で dense `I ⊗ ... ⊗ U_i ⊗ ... ⊗ I`
        // との rel < 1e-13 一致.
        for n in 2..=4 {
            let dim = 1usize << n;
            for seed in [5, 23, 71] {
                let mut rng = Xor64::new(seed ^ (n as u64));
                let psi0: Vec<Complex64> = (0..dim).map(|_| rng.complex_signed()).collect();
                for i in 0..n {
                    let u = random_unitary_2x2(&mut rng);
                    let m = build_dense_single_mode(n, &u, i);
                    let psi_vec = DVector::<Complex64>::from_vec(psi0.clone());
                    let expected = &m * &psi_vec;

                    let mut psi = psi0.clone();
                    apply_single_mode_axis_i(&mut psi, &u, i, n);
                    let rel = relative_error(&psi, &expected);
                    assert!(
                        rel < 1e-13,
                        "n={}, i={}, seed={}: dense Kronecker mismatch rel={}",
                        n,
                        i,
                        seed,
                        rel,
                    );
                }
            }
        }
    }

    #[test]
    fn single_mode_matches_skip_variant() {
        // 2 重ループ形と `k & mask != 0` skip 版が要素ごとに一致.
        for n in 1..=6 {
            let dim = 1usize << n;
            for seed in [13, 31, 0xdead_beef_u64] {
                let mut rng = Xor64::new(seed.wrapping_add((n as u64) * 7));
                let psi0: Vec<Complex64> = (0..dim).map(|_| rng.complex_signed()).collect();
                for i in 0..n {
                    let u = random_unitary_2x2(&mut rng);

                    let mut psi_block = psi0.clone();
                    apply_single_mode_axis_i(&mut psi_block, &u, i, n);

                    let mut psi_skip = psi0.clone();
                    apply_single_mode_axis_i_skip(&mut psi_skip, &u, i, n);

                    for k in 0..dim {
                        let diff = (psi_block[k] - psi_skip[k]).norm();
                        // 実装が異なるが演算順序は同じ (a, b に依存しない)
                        // ため bit-for-bit 一致を期待.
                        assert!(
                            diff < 1e-15,
                            "n={}, i={}, k={}, seed={}: block vs skip mismatch \
                             ({} vs {})",
                            n,
                            i,
                            k,
                            seed,
                            psi_block[k],
                            psi_skip[k],
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn single_mode_trotter_x_rotation_n1() {
        // n=1 の Trotter R_0(θ) = cos θ · I + i sin θ · X を直接適用し,
        // 手書きの 2x2 行列適用結果と一致.
        let theta = 0.37_f64;
        let (s, c) = theta.sin_cos();
        let u = [
            Complex64::new(c, 0.0),
            Complex64::new(0.0, s),
            Complex64::new(0.0, s),
            Complex64::new(c, 0.0),
        ];
        let psi0 = [Complex64::new(0.6, -0.2), Complex64::new(-0.3, 0.8)];
        let expected = [
            u[0] * psi0[0] + u[1] * psi0[1],
            u[2] * psi0[0] + u[3] * psi0[1],
        ];
        let mut psi = psi0;
        apply_single_mode_axis_i(&mut psi, &u, 0, 1);
        for k in 0..2 {
            let diff = (psi[k] - expected[k]).norm();
            assert!(
                diff < 1e-15,
                "k={}: psi={}, expected={}",
                k,
                psi[k],
                expected[k]
            );
        }
    }
}
