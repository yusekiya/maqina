//! 実対称三重対角行列の hand-rolled 完全固有分解.
//!
//! [`tridiag_eigh`] は Lanczos 内部 (m ≈ 24) で得られる実対称三重対角行列の
//! 完全固有分解を行う低レベルユーティリティ. 詳細は `docs/design/07-rust-extension.md` §7.1.
//!
//! - アルゴリズム: implicit shift QL (Wilkinson shift), EISPACK `tql2` 互換
//!   (Golub-Van Loan, "Matrix Computations" 4th ed., §8.3).
//! - LAPACK 非依存: BLAS feature ON / OFF どちらでも同じ scalar 経路.
//! - 出力: 固有値 λ_k (昇順, `d` を上書き) と固有ベクトル行列 `Q` を
//!   row-major で返す. **`q` の row k が λ_k に対応する単位固有
//!   ベクトル** という規約.
//!
//! Phase 1 では Lanczos (`src/krylov.rs`) からのみ呼ばれる. tridiag だけが
//! landed した時点では `pub(crate)` 項目が未参照になるため, Lanczos 着地
//! までの間 `dead_code` lint を許容する.

#![allow(dead_code)]

use std::fmt;

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

/// 三重対角固有分解で発生し得るエラー.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum TridiagError {
    /// QL 反復が `30 · m` 回を超えても deflate しなかった.
    ///
    /// `l` は deflate に失敗した sub-block の先頭インデックス.
    NoConvergence { l: usize },
}

impl fmt::Display for TridiagError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TridiagError::NoConvergence { l } => write!(
                f,
                "implicit QL did not converge within 30·m iterations (sub-block l={l})"
            ),
        }
    }
}

impl std::error::Error for TridiagError {}

/// 実対称三重対角行列の完全固有分解.
///
/// 実対称 m × m 三重対角行列 T を `T = Qᵀ diag(λ) Q` の形に分解する.
/// 行列 T は対角成分 `d` (長さ m) と副対角成分 `e[0..m-1]` (長さ m,
/// 末尾は sentinel) で表現する.
///
/// # 入出力
/// - `d` (length m): 入力で対角成分 `d_k = T[k, k]`. 戻り時に **昇順の
///   固有値 λ_k** で上書きされる.
/// - `e` (length m): 入力で `e[i] = T[i, i+1] = T[i+1, i]` (`i = 0..m-2`).
///   `e[m-1]` は sentinel として使われるので任意の値で渡してよい
///   (関数内で 0 にリセットする). 戻り時の内容は不定.
/// - `q` (length m·m): 戻り時に row-major で固有ベクトル行列が入る.
///   row k が λ_k = d[k] に対応する単位固有ベクトル.
///   入力時の内容は無視される (関数内で identity に初期化する).
///
/// # アルゴリズム
/// implicit shift QL with Wilkinson shift (Golub-Van Loan §8.3,
/// EISPACK `tql2` 互換). 詳細は `docs/design/07-rust-extension.md` §7.1.
///
/// - Deflation 閾値: `|e[k]| ≤ ε · (|d[k]| + |d[k+1]|)` (ε = `f64::EPSILON`).
/// - 最大反復数: `30 · m` 回 (LAPACK `dsteqr` と同じ). 超過時は
///   [`TridiagError::NoConvergence`].
/// - Givens rotation は `f64::hypot` で overflow / underflow を回避.
///
/// # Panics
/// `e.len() != m` または `q.len() != m * m` のとき.
pub(crate) fn tridiag_eigh(
    d: &mut [f64],
    e: &mut [f64],
    q: &mut [f64],
) -> Result<(), TridiagError> {
    let m = d.len();
    assert_eq!(e.len(), m, "e must have length m (last entry is sentinel)");
    assert_eq!(q.len(), m * m, "q must have length m * m");

    // q を identity に初期化 (row k = 標準基底 e_k).
    for entry in q.iter_mut() {
        *entry = 0.0;
    }
    for i in 0..m {
        q[i * m + i] = 1.0;
    }

    if m <= 1 {
        return Ok(());
    }

    // Sentinel: e[m-1] は副対角ではないので明示的に 0 にしておく.
    e[m - 1] = 0.0;

    const EPS: f64 = f64::EPSILON;
    let max_iter_total = 30 * m;
    let mut iter_total: usize = 0;

    for l in 0..m {
        loop {
            // 最小の n_idx ∈ [l, m) で e[n_idx] が deflation 閾値以下に
            // 落ちている位置を探す. e[m-1] = 0 が sentinel として終端を
            // 保証する.
            let mut n_idx = l;
            while n_idx < m - 1 {
                if e[n_idx].abs() <= EPS * (d[n_idx].abs() + d[n_idx + 1].abs()) {
                    break;
                }
                n_idx += 1;
            }

            if n_idx == l {
                // d[l] は単独で deflate 済み. 次の l へ.
                break;
            }

            if iter_total >= max_iter_total {
                return Err(TridiagError::NoConvergence { l });
            }
            iter_total += 1;

            // Wilkinson shift:
            //     μ - d[l] = e[l] / (g0 + sign(g0) · sqrt(g0² + 1))
            // ただし g0 = (d[l+1] - d[l]) / (2 e[l]).
            // 以降の bulge chase 用に g を
            //     g = d[n_idx] - d[l] + (μ - d[l])
            // で初期化する (NR `tqli` の慣習).
            let mut g = (d[l + 1] - d[l]) / (2.0 * e[l]);
            let r0 = g.hypot(1.0);
            // SIGN(r0, g): r0 ≥ 0 なので符号は g に従う. g == 0 のときは
            // どちらでも良いが, 正方向に倒しておく.
            let denom = if g >= 0.0 { g + r0 } else { g - r0 };
            g = d[n_idx] - d[l] + e[l] / denom;

            let mut s: f64 = 1.0;
            let mut c: f64 = 1.0;
            let mut p: f64 = 0.0;

            // QL 1 sweep: i = n_idx - 1, n_idx - 2, ..., l.
            let mut zero_subdiag = false;
            for i in (l..n_idx).rev() {
                let f = s * e[i];
                let b = c * e[i];
                let r = f.hypot(g);
                e[i + 1] = r;

                if r == 0.0 {
                    // f = g = 0 (underflow). 早期復帰してこの l を再評価.
                    d[i + 1] -= p;
                    e[n_idx] = 0.0;
                    zero_subdiag = true;
                    break;
                }

                s = f / r;
                c = g / r;
                let g_prev = d[i + 1] - p;
                let r2 = (d[i] - g_prev) * s + 2.0 * c * b;
                p = s * r2;
                d[i + 1] = g_prev + p;
                g = c * r2 - b;

                // 固有ベクトル累積: q の row i と row i+1 に Givens
                // rotation を作用させる (実装上は Q = Z^T を row-major で
                // 持つことで, 標準 tql2 の column rotation と同型).
                let (lo, hi) = q.split_at_mut((i + 1) * m);
                let row_i = &mut lo[i * m..(i + 1) * m];
                let row_ip1 = &mut hi[..m];
                for k in 0..m {
                    let g_k = row_i[k];
                    let f_k = row_ip1[k];
                    row_ip1[k] = s * g_k + c * f_k;
                    row_i[k] = c * g_k - s * f_k;
                }
            }

            if zero_subdiag {
                continue;
            }

            d[l] -= p;
            e[l] = g;
            e[n_idx] = 0.0;
        }
    }

    // 固有値を昇順 sort し, 対応する q の row も並べ替える.
    sort_eigh_ascending(d, q);
    Ok(())
}

/// `tridiag_eigh` の Python wrap. 実対称三重対角の完全固有分解を Python
/// 側に露出する.
///
/// 主用途は `python/maqina/eigenstates.py` の `instantaneous_eigenstates`
/// (Lanczos 経路で得た `(α, β)` の固有分解を Python 側から呼ぶ). Lanczos /
/// CFM4:2 内部の固有分解は Rust 内で完結するため, 本関数は **eigenstates /
/// 参照実装比較用** の公開 API である (`apply_h_kinema_py` 等と同じ位置付).
///
/// # Python 側シグネチャ
/// ```python
/// eigvals, eigvecs = _rust.tridiag_eigh_py(alpha, beta)
/// ```
/// - `alpha`: shape `(m,)` float64. 対角成分 `T[k, k]`.
/// - `beta`: shape `(m-1,)` float64. 副対角成分 `T[i, i+1]`.
/// - `eigvals`: shape `(m,)` float64. 昇順固有値.
/// - `eigvecs`: shape `(m, m)` float64. **column `k` が `eigvals[k]` に対応
///   する単位固有ベクトル** (scipy.linalg.eigh と同じ規約). Rust 内部の
///   row-major 行列を本関数内で転置して返す.
///
/// `m = 1` のとき `beta` は長さ 0 の空配列で渡す.
// PyO3 の戻り型 `(Bound<PyArray1<_>>, Bound<PyArray2<_>>)` は clippy の
// `type_complexity` lint に引っかかるが, 1 箇所のみの thin-wrap で型 alias を
// 切るほどの再利用は無いので allow する.
#[pyfunction]
#[pyo3(signature = (alpha, beta))]
#[allow(clippy::type_complexity)]
pub(crate) fn tridiag_eigh_py<'py>(
    py: Python<'py>,
    alpha: PyReadonlyArray1<'py, f64>,
    beta: PyReadonlyArray1<'py, f64>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>)> {
    let a = alpha.as_slice()?;
    let b = beta.as_slice()?;
    let m = a.len();
    if m == 0 {
        return Err(PyValueError::new_err("alpha must have length >= 1"));
    }
    if b.len() + 1 != m {
        return Err(PyValueError::new_err(format!(
            "beta length {} must equal alpha length {} - 1",
            b.len(),
            m,
        )));
    }
    let mut d = a.to_vec();
    // e of length m: e[0..m-1] = beta, e[m-1] = sentinel (set by tridiag_eigh).
    let mut e = vec![0.0_f64; m];
    if m > 1 {
        e[..m - 1].copy_from_slice(b);
    }
    let mut q = vec![0.0_f64; m * m];
    tridiag_eigh(&mut d, &mut e, &mut q)
        .map_err(|err| PyRuntimeError::new_err(format!("tridiag_eigh failed: {err}")))?;

    // Rust 規約: row k = eigvals[k] に対応する単位固有ベクトル (row-major).
    // Python 規約 (scipy): eigvecs[:, k] = eigvals[k] の固有ベクトル.
    // 転置して返す.
    let mut q_t = vec![0.0_f64; m * m];
    for k in 0..m {
        for i in 0..m {
            q_t[i * m + k] = q[k * m + i];
        }
    }
    let arr = Array2::from_shape_vec((m, m), q_t).expect("shape (m, m) matches m*m");
    Ok((d.into_pyarray(py), PyArray2::from_owned_array(py, arr)))
}

/// 固有値 `d` を昇順 sort し, `q` の row も同じ permutation で並べ替える.
/// m ≤ 数十の小さい m を想定しているので O(m²) selection sort で十分.
fn sort_eigh_ascending(d: &mut [f64], q: &mut [f64]) {
    let m = d.len();
    for i in 0..m {
        let mut min_idx = i;
        for j in (i + 1)..m {
            if d[j] < d[min_idx] {
                min_idx = j;
            }
        }
        if min_idx != i {
            d.swap(i, min_idx);
            for k in 0..m {
                q.swap(i * m + k, min_idx * m + k);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, SymmetricEigen};

    /// 軽量決定論的 PRNG (xorshift64). テスト用途のみ.
    struct Xor64(u64);

    impl Xor64 {
        fn new(seed: u64) -> Self {
            // 0 を入れると全ビット 0 になり xorshift が縮退するため,
            // 適当な non-zero 定数で置き換える.
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
            let u = (self.next_u64() >> 11) as f64 * SCALE; // [0, 1)
            2.0 * u - 1.0
        }
    }

    /// (d, e) から full m × m 対称行列を構築 (nalgebra 比較用).
    fn build_full_matrix(d: &[f64], e: &[f64]) -> DMatrix<f64> {
        let m = d.len();
        DMatrix::from_fn(m, m, |i, j| {
            if i == j {
                d[i]
            } else if i + 1 == j {
                e[i]
            } else if j + 1 == i {
                e[j]
            } else {
                0.0
            }
        })
    }

    /// 入力 (d, e) を破壊せずに `tridiag_eigh` を呼ぶ簡易ラッパ.
    fn eigh(d: &[f64], e: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let m = d.len();
        let mut d_buf = d.to_vec();
        let mut e_buf = e.to_vec();
        if m > 0 {
            // sentinel を明示的に 0 padding.
            let last = e_buf.len();
            if last == m {
                e_buf[m - 1] = 0.0;
            } else {
                e_buf.resize(m, 0.0);
            }
        }
        let mut q_buf = vec![0.0_f64; m * m];
        tridiag_eigh(&mut d_buf, &mut e_buf, &mut q_buf).expect("QL converged");
        (d_buf, q_buf)
    }

    /// nalgebra の `SymmetricEigen` 結果を昇順 sort して返す.
    fn nalgebra_eigh_sorted(d: &[f64], e: &[f64]) -> (Vec<f64>, DMatrix<f64>) {
        let m = d.len();
        let full = build_full_matrix(d, e);
        let eig = SymmetricEigen::new(full);
        let mut order: Vec<usize> = (0..m).collect();
        order.sort_by(|&a, &b| eig.eigenvalues[a].partial_cmp(&eig.eigenvalues[b]).unwrap());
        let lam: Vec<f64> = order.iter().map(|&i| eig.eigenvalues[i]).collect();
        let mut vec_sorted = DMatrix::zeros(m, m);
        for (new_col, &old_col) in order.iter().enumerate() {
            for row in 0..m {
                vec_sorted[(row, new_col)] = eig.eigenvectors[(row, old_col)];
            }
        }
        (lam, vec_sorted)
    }

    /// `q` (row-major, row k = 固有ベクトル k) の直交性誤差 `‖Q Qᵀ - I‖_F`.
    fn orth_error(q: &[f64], m: usize) -> f64 {
        let mut err = 0.0_f64;
        for i in 0..m {
            for j in 0..m {
                let mut s = 0.0_f64;
                for k in 0..m {
                    s += q[i * m + k] * q[j * m + k];
                }
                let target = if i == j { 1.0 } else { 0.0 };
                err += (s - target).powi(2);
            }
        }
        err.sqrt()
    }

    /// `‖Qᵀ diag(λ) Q - T‖_F`. T は (d_in, e_in) から構築する.
    /// row k = 固有ベクトル k 規約に基づき (Qᵀ Λ Q)[i, j] = Σ_k q[k, i] λ_k q[k, j].
    fn reconstruction_error(d_in: &[f64], e_in: &[f64], lam: &[f64], q: &[f64]) -> f64 {
        let m = d_in.len();
        let mut err = 0.0_f64;
        for i in 0..m {
            for j in 0..m {
                let mut s = 0.0_f64;
                for k in 0..m {
                    s += q[k * m + i] * lam[k] * q[k * m + j];
                }
                let t = if i == j {
                    d_in[i]
                } else if i + 1 == j {
                    e_in[i]
                } else if j + 1 == i {
                    e_in[j]
                } else {
                    0.0
                };
                err += (s - t).powi(2);
            }
        }
        err.sqrt()
    }

    /// T のフロベニウスノルム (対角 + 副対角 × 2).
    fn t_norm(d: &[f64], e_offdiag: &[f64]) -> f64 {
        let mut s = 0.0_f64;
        for &v in d {
            s += v * v;
        }
        for &v in e_offdiag {
            s += 2.0 * v * v;
        }
        s.sqrt()
    }

    /// 固有値ベクトルの相対誤差 (max over k of |λ_my - λ_ref| / max(|λ_ref|, 1)).
    fn eigenvalue_rel_error(my: &[f64], reference: &[f64]) -> f64 {
        my.iter()
            .zip(reference.iter())
            .map(|(a, b)| {
                let scale = a.abs().max(b.abs()).max(1.0);
                ((a - b) / scale).abs()
            })
            .fold(0.0_f64, f64::max)
    }

    #[test]
    fn m1_trivial() {
        let d = vec![3.0];
        let e = vec![0.0];
        let (lam, q) = eigh(&d, &e);
        assert_eq!(lam, vec![3.0]);
        assert_eq!(q, vec![1.0]);
    }

    #[test]
    fn m2_analytic() {
        // T = [[1, 3], [3, 2]]. λ = 1.5 ± sqrt(0.25 + 9).
        let d_in = vec![1.0, 2.0];
        let e_in = vec![3.0, 0.0];
        let (lam, q) = eigh(&d_in, &e_in);
        let disc = (0.25_f64 + 9.0).sqrt();
        let lam_ref = [1.5 - disc, 1.5 + disc];
        for (a, b) in lam.iter().zip(lam_ref.iter()) {
            assert!((a - b).abs() < 1e-13, "λ mismatch: {a} vs {b}");
        }
        assert!(orth_error(&q, 2) < 1e-13);
        assert!(reconstruction_error(&d_in, &e_in, &lam, &q) < 1e-13);
    }

    #[test]
    fn diagonal_identity_and_distinct() {
        // (a) すべて 0 off-diag, 同一対角 → 全固有値同じ.
        let m = 8;
        let d_in = vec![1.0; m];
        let e_in = vec![0.0; m];
        let (lam, q) = eigh(&d_in, &e_in);
        for &l in lam.iter() {
            assert!((l - 1.0).abs() < 1e-14);
        }
        assert!(orth_error(&q, m) < 1e-13);

        // (b) 0 off-diag, 区別可能な対角 → 固有値は対角そのもの (昇順).
        let d_in: Vec<f64> = (1..=m).map(|i| i as f64).collect();
        let e_in = vec![0.0; m];
        let (lam, q) = eigh(&d_in, &e_in);
        for (i, &l) in lam.iter().enumerate() {
            assert!((l - (i as f64 + 1.0)).abs() < 1e-14);
        }
        assert!(orth_error(&q, m) < 1e-13);
    }

    /// 単一ケースを検証するヘルパ. ランダム m × 50 seeds テストで共有.
    fn run_random_case(m: usize, seed: u64) {
        let mut rng = Xor64::new(seed);
        let d_in: Vec<f64> = (0..m).map(|_| rng.signed()).collect();
        let mut e_in: Vec<f64> = (0..m).map(|_| rng.signed()).collect();
        e_in[m - 1] = 0.0;

        // 固有値・固有ベクトル両方を計算して比較.
        let (lam_ref, q_ref) = nalgebra_eigh_sorted(&d_in, &e_in);

        // ※ d_in, e_in は eigh で破壊しないので順序問わずよい.
        let (lam, q) = eigh(&d_in, &e_in);
        // sanity: 固有値が昇順か.
        for k in 1..m {
            assert!(lam[k - 1] <= lam[k] + 1e-15);
        }

        // 固有値: rel < 1e-13.
        let err = eigenvalue_rel_error(&lam, &lam_ref);
        assert!(
            err < 1e-13,
            "eigval rel err {err} too large (m={m} seed={seed})"
        );

        // 固有ベクトル: |⟨ref, mine⟩| > 1 - 1e-12. 縮退ペアは skip.
        for k in 0..m {
            let lo = if k > 0 {
                (lam_ref[k] - lam_ref[k - 1]).abs() < 1e-10
            } else {
                false
            };
            let hi = if k + 1 < m {
                (lam_ref[k + 1] - lam_ref[k]).abs() < 1e-10
            } else {
                false
            };
            if lo || hi {
                continue;
            }
            let mut dot = 0.0_f64;
            for i in 0..m {
                dot += q[k * m + i] * q_ref[(i, k)];
            }
            assert!(
                dot.abs() > 1.0 - 1e-12,
                "eigvec overlap {} (m={m} seed={seed} k={k})",
                dot.abs()
            );
        }

        // 直交性 + 再構成 (これは縮退があっても成立する基本検証).
        assert!(
            orth_error(&q, m) < 1e-12,
            "Q not orthogonal (m={m} seed={seed})"
        );
        let rec = reconstruction_error(&d_in, &e_in, &lam, &q);
        let tn = t_norm(&d_in, &e_in[..m - 1]).max(1.0);
        assert!(
            rec / tn < 1e-12,
            "reconstruction rel err {} (m={m} seed={seed})",
            rec / tn
        );
    }

    #[test]
    fn random_m8() {
        for seed in 1..=50 {
            run_random_case(8, seed);
        }
    }

    #[test]
    fn random_m16() {
        for seed in 1..=50 {
            run_random_case(16, seed);
        }
    }

    #[test]
    fn random_m24() {
        for seed in 1..=50 {
            run_random_case(24, seed);
        }
    }

    #[test]
    fn degenerate_block_diagonal() {
        // T = block_diag(T_half, T_half) を作る. 同一の 6 × 6 ブロックを
        // 2 個並べることで全固有値が 2 重縮退する.
        let mut rng = Xor64::new(20260512);
        let half = 6;
        let d_half: Vec<f64> = (0..half).map(|_| rng.signed()).collect();
        let mut e_half: Vec<f64> = (0..half).map(|_| rng.signed()).collect();
        e_half[half - 1] = 0.0;

        let m = 2 * half;
        let mut d_in = Vec::with_capacity(m);
        d_in.extend_from_slice(&d_half);
        d_in.extend_from_slice(&d_half);

        let mut e_in = Vec::with_capacity(m);
        // 前半ブロック内: e_half[0..half-1].
        e_in.extend_from_slice(&e_half[..half - 1]);
        // ブロック境界で結合を切る.
        e_in.push(0.0);
        // 後半ブロック内: e_half[0..half-1].
        e_in.extend_from_slice(&e_half[..half - 1]);
        // sentinel.
        e_in.push(0.0);

        let (lam, q) = eigh(&d_in, &e_in);
        let (lam_ref, _) = nalgebra_eigh_sorted(&d_in, &e_in);

        // 固有値: rel < 1e-13.
        let err = eigenvalue_rel_error(&lam, &lam_ref);
        assert!(err < 1e-13, "eigval rel err {err} for degenerate case");

        // 期待される 2 重縮退の確認 (ペアごとに ~equal).
        for i in (0..m).step_by(2) {
            assert!(
                (lam_ref[i] - lam_ref[i + 1]).abs() < 1e-12,
                "expected degeneracy at i={i}: {} vs {}",
                lam_ref[i],
                lam_ref[i + 1]
            );
        }

        // 直交性 + 再構成 (縮退でも成立).
        assert!(orth_error(&q, m) < 1e-12);
        let rec = reconstruction_error(&d_in, &e_in, &lam, &q);
        let tn = t_norm(&d_in, &e_in[..m - 1]).max(1.0);
        assert!(rec / tn < 1e-12, "reconstruction rel err {}", rec / tn);
    }

    #[test]
    fn convergence_error_on_degenerate_pathology_is_unreachable() {
        // sanity: ランダム m=24, 50 シードで NoConvergence が出ない (実害が
        // ないことを確認するだけのスモークテスト).
        for seed in 1..=50 {
            let mut rng = Xor64::new(0xabcd_0000 + seed);
            let m = 24;
            let mut d = vec![0.0; m];
            let mut e = vec![0.0; m];
            for v in d.iter_mut() {
                *v = rng.signed();
            }
            for v in e.iter_mut().take(m - 1) {
                *v = rng.signed();
            }
            let mut q = vec![0.0; m * m];
            tridiag_eigh(&mut d, &mut e, &mut q).expect("QL converged");
        }
    }
}
