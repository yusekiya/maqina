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
//! Phase 6 C1 (issue #62) で rayon `par_chunks_mut` 経由の L2 並列化を導入済み
//! (`feature = "rayon"`, default ON). 同じ chunk closure 内で diag pass + 全 i
//! bit-flip pass を完走する **cache-blocked 形** を採用し, `y_chunk` を L1 cache
//! resident に保つことで後段 SIMD (Phase 6 C2) / cache block-fusion (Phase 6 C3)
//! の足場とする (`docs/design.md` §5.1.1 / §12 Phase 6). `--no-default-features`
//! ビルドでは scalar 単スレッド経路 (`*_serial`) が呼ばれ, 既存挙動を維持する.
//!
//! ## Thread pool 競合の注意 (rayon × BLAS)
//!
//! rayon thread 数は環境変数 `RAYON_NUM_THREADS` で **プロセス起動時に** 設定
//! する (rayon の global pool は最初の rayon op で構築される). 一方 BLAS
//! thread 数は Python 側 `kryanneal.set_blas_threads(n)` で動的に変えられる.
//! 両者が `cpu_count` × `cpu_count` の総スレッド数を取ると context-switch で
//! 性能が落ちるため, **rayon 経路で並列化する場合は `set_blas_threads(1)` に
//! 落として BLAS pool を 1 thread に固定する** 運用が推奨 (詳細は `CLAUDE.md`
//! 「Thread pool 運用」節).
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

#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// rayon chunk あたりの **最大** 要素数. y_chunk + v の参照範囲が L2 cache
/// (~256 KB ≈ 16K Complex64) に収まる上限を狙う.
#[cfg(feature = "rayon")]
const RAYON_CHUNK_MAX: usize = 1 << 14;

/// rayon chunk あたりの **最小** 要素数. closure / scheduling overhead を
/// 償却するため 64 要素 (cache-line 4 要素 × 16) 以上を保証する.
#[cfg(feature = "rayon")]
const RAYON_CHUNK_MIN: usize = 1 << 6;

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
/// # 実装
/// `feature = "rayon"` (default ON) 時は [`apply_h_kryanneal_rayon`] が呼ばれ
/// `y` を `par_chunks_mut` で分割し chunk closure 内で diag + 全 i bit-flip
/// pass を完走する. 各 `y[k]` への書き込みは単一スレッドからしか発生せず,
/// `v` は read-only のため race-free. 演算順序は chunk 内で serial と同じ
/// (diag → i=0 → i=1 → ...) なので **rayon あり/なし両ビルドで bit-identical**
/// に y[k] を生成する (詳細は `apply_h_kryanneal_rayon_matches_serial_*`
/// テスト). `--no-default-features` 時は [`apply_h_kryanneal_serial`] にフォール
/// バック.
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

    #[cfg(feature = "rayon")]
    {
        apply_h_kryanneal_rayon(v, y, h_x, h_p_diag, a_t, b_t, n);
    }
    #[cfg(not(feature = "rayon"))]
    {
        apply_h_kryanneal_serial(v, y, h_x, h_p_diag, a_t, b_t, n);
    }
}

/// `apply_h_kryanneal` の scalar 単スレッド実装. `feature = "rayon"` OFF
/// ビルドおよび `#[cfg(test)]` 経路から rayon 経路との数値同一性比較に使う.
fn apply_h_kryanneal_serial(
    v: &[Complex64],
    y: &mut [Complex64],
    h_x: &[f64],
    h_p_diag: &[f64],
    a_t: f64,
    b_t: f64,
    n: usize,
) {
    let dim = 1usize << n;
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

/// `apply_h_kryanneal` の rayon 並列実装. `y` を `par_chunks_mut(chunk_size)`
/// で分割し, 各 chunk closure 内で diag pass → 全 i bit-flip pass を完走する.
///
/// # 並列化スキーム (cache-blocked 形)
///
/// chunk 単位の closure に diag + 全 i を **fuse** することで:
///
/// 1. `y_chunk` (~`chunk_size`要素) が closure 実行中 L1 cache resident に
///    保たれ, `(n+1)` 回の touch を re-load なしで処理できる.
/// 2. rayon barrier が **per-call 1 個** で済む (i ごとに par section を
///    張る形だと `(n+1)` 個入る). dim が小さい matvec を Lanczos の各
///    step で繰り返す呼出パターンに有利.
/// 3. 後段の SIMD (Phase 6 C2) は内側 `li` loop に, cache block-fusion
///    (Phase 6 C3) は同 closure 内 i loop に直接重ねられる.
///
/// `v[k ^ mask]` の読み込みは chunk 境界を跨ぐことがある (mask > chunk_size)
/// が `v` は read-only で race-free. 各 `y[k]` への書き込みは disjoint な
/// chunk 内に閉じる.
///
/// # チャンクサイズ
///
/// `current_num_threads() * 4` 個のチャンクを目標に `dim / (nth*4)` を取り,
/// [`RAYON_CHUNK_MIN`] (closure overhead 償却) と [`RAYON_CHUNK_MAX`]
/// (L2 cache 上限) で clamp する.
#[cfg(feature = "rayon")]
fn apply_h_kryanneal_rayon(
    v: &[Complex64],
    y: &mut [Complex64],
    h_x: &[f64],
    h_p_diag: &[f64],
    a_t: f64,
    b_t: f64,
    n: usize,
) {
    let dim = 1usize << n;
    let nth = rayon::current_num_threads().max(1);
    let chunk_size = (dim / (nth * 4)).clamp(RAYON_CHUNK_MIN, RAYON_CHUNK_MAX);

    y.par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(idx, y_chunk)| {
            let base = idx * chunk_size;
            // 対角 pass: y_chunk[li] = b_t · H_p[k] · v[k].
            for (li, y_k) in y_chunk.iter_mut().enumerate() {
                let k = base + li;
                *y_k = Complex64::new(b_t * h_p_diag[k], 0.0) * v[k];
            }
            // bit-flip pass: 全 i を同一 chunk 内で完走 (y_chunk が L1 resident).
            for (i, &h_x_i) in h_x.iter().enumerate() {
                let coeff = -a_t * h_x_i;
                let mask = 1usize << i;
                for (li, y_k) in y_chunk.iter_mut().enumerate() {
                    let k = base + li;
                    *y_k += Complex64::new(coeff, 0.0) * v[k ^ mask];
                }
            }
        });
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
/// `feature = "rayon"` (default ON) では [`apply_single_mode_axis_i_rayon`]
/// が呼ばれ block 単位 (`2·mask`) で `par_chunks_mut` 並列化される. 退化
/// ケース (i = n-1, block = dim, chunk が 1 個になる) は psi を上下半分に
/// `split_at_mut` した上で `par_iter_mut().zip(par_iter_mut())` のペア並列に
/// 切り替える. 各ペア `(lo, hi)` は単一スレッドが処理し write は disjoint
/// なので race-free + rayon あり/なし両ビルドで bit-identical.
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

    #[cfg(feature = "rayon")]
    {
        apply_single_mode_axis_i_rayon(psi, u, i, n);
    }
    #[cfg(not(feature = "rayon"))]
    {
        apply_single_mode_axis_i_serial(psi, u, i, n);
    }
}

/// `apply_single_mode_axis_i` の scalar 単スレッド実装. テスト経路から
/// rayon 結果との bit-identical 比較に使う.
fn apply_single_mode_axis_i_serial(psi: &mut [Complex64], u: &[Complex64; 4], i: usize, n: usize) {
    let dim = 1usize << n;
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

/// `apply_single_mode_axis_i` の rayon 並列実装.
///
/// # 並列化スキーム
///
/// `block = 2·mask` 単位で操作を分割する.
///
/// - **複数 block** (`block < dim`, すなわち `i < n-1`): `par_chunks_mut(chunk_size)`
///   で `chunk_size` を `block` の整数倍 (≥ [`RAYON_CHUNK_MIN`]) に揃え, 各
///   chunk 内で block ごとに `split_at_mut(mask)` してペア処理する. chunk
///   境界が `block` 境界に揃うので block を跨ぐ書き込み競合は起きない.
/// - **単一 block** (`block == dim`, `i = n-1` 退化ケース): `psi.split_at_mut(mask)`
///   で上下半分に分け `par_iter_mut().zip(par_iter_mut())` で個別ペアを並列に
///   処理する.
///
/// どちらの経路もペア `(lo, hi)` は単一スレッドが排他処理し, 操作内容も
/// scalar 経路と同じ `a = psi[lo]; b = psi[hi]; psi[lo] = u[0]·a + u[1]·b;
/// psi[hi] = u[2]·a + u[3]·b` のためスケジュールに依らず bit-identical.
#[cfg(feature = "rayon")]
fn apply_single_mode_axis_i_rayon(psi: &mut [Complex64], u: &[Complex64; 4], i: usize, n: usize) {
    let dim = 1usize << n;
    let mask = 1usize << i;
    let block = mask << 1; // 2 * mask

    if block == dim {
        // 単一 block (i == n-1): 上下半分に分けて pair 並列.
        let (lo_half, hi_half) = psi.split_at_mut(mask);
        lo_half
            .par_iter_mut()
            .zip(hi_half.par_iter_mut())
            .with_min_len(RAYON_CHUNK_MIN)
            .for_each(|(a_ref, b_ref)| {
                let a = *a_ref;
                let b = *b_ref;
                *a_ref = u[0] * a + u[1] * b;
                *b_ref = u[2] * a + u[3] * b;
            });
    } else {
        // 複数 block: chunk_size を block の整数倍で RAYON_CHUNK_MIN 以上に揃える.
        // block も chunk_size も power of 2 で dim も power of 2 なので chunk
        // は全て等サイズ (last chunk が短くなることはない).
        let chunk_size = if block >= RAYON_CHUNK_MIN {
            block
        } else {
            // block=2 (i=0) なら chunk_size=64, block=4 (i=1) なら 64, ...
            // block * ceil(RAYON_CHUNK_MIN / block) = block の整数倍で最初の
            // ≥ RAYON_CHUNK_MIN を取る.
            RAYON_CHUNK_MIN.next_multiple_of(block)
        };
        let chunk_size = chunk_size.min(dim);
        psi.par_chunks_mut(chunk_size).for_each(|chunk| {
            let mut local_base = 0usize;
            while local_base < chunk.len() {
                let sub = &mut chunk[local_base..local_base + block];
                let (lo_half, hi_half) = sub.split_at_mut(mask);
                for (a_ref, b_ref) in lo_half.iter_mut().zip(hi_half.iter_mut()) {
                    let a = *a_ref;
                    let b = *b_ref;
                    *a_ref = u[0] * a + u[1] * b;
                    *b_ref = u[2] * a + u[3] * b;
                }
                local_base += block;
            }
        });
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

    // ===== rayon 並列化 (Phase 6 C1, issue #62) のテスト =====

    /// rayon あり/なしで [`apply_h_kryanneal`] が要素ごとに bit-identical な
    /// 結果を返すこと. 各 `y[k]` は単一スレッドで diag pass → i=0,1,...,n-1
    /// の順に同じ演算順序を踏むため bit-for-bit 一致を期待できる.
    ///
    /// `n=10` (dim=1024) は rayon path の chunk 数 ≥ 2 を確実に超える
    /// (chunk_size ≤ `RAYON_CHUNK_MAX`, dim/chunk_size ≥ 1024/16384 < 1 でも
    /// `max(dim/(nth·4), RAYON_CHUNK_MIN)` で複数 chunk になる典型サイズ).
    #[cfg(feature = "rayon")]
    #[test]
    fn apply_h_kryanneal_rayon_matches_serial() {
        for n in [3, 6, 10, 12] {
            let dim = 1usize << n;
            for seed in [1u64, 17, 0xdead_beef] {
                let mut rng = Xor64::new(seed.wrapping_add(n as u64));
                let h_x: Vec<f64> = (0..n).map(|_| rng.signed()).collect();
                let h_p_diag: Vec<f64> = (0..dim).map(|_| rng.signed()).collect();
                let v: Vec<Complex64> = (0..dim).map(|_| rng.complex_signed()).collect();
                let a_t = rng.signed();
                let b_t = rng.signed();

                let mut y_serial = vec![Complex64::new(0.0, 0.0); dim];
                apply_h_kryanneal_serial(&v, &mut y_serial, &h_x, &h_p_diag, a_t, b_t, n);

                let mut y_par = vec![Complex64::new(0.0, 0.0); dim];
                apply_h_kryanneal_rayon(&v, &mut y_par, &h_x, &h_p_diag, a_t, b_t, n);

                for k in 0..dim {
                    assert_eq!(
                        y_par[k].re.to_bits(),
                        y_serial[k].re.to_bits(),
                        "n={}, seed={}, k={}: rayon re-bits differ from serial \
                         (rayon={:?}, serial={:?})",
                        n,
                        seed,
                        k,
                        y_par[k],
                        y_serial[k],
                    );
                    assert_eq!(
                        y_par[k].im.to_bits(),
                        y_serial[k].im.to_bits(),
                        "n={}, seed={}, k={}: rayon im-bits differ from serial \
                         (rayon={:?}, serial={:?})",
                        n,
                        seed,
                        k,
                        y_par[k],
                        y_serial[k],
                    );
                }
            }
        }
    }

    /// rayon あり/なしで [`apply_single_mode_axis_i`] が bit-identical.
    /// `i = n-1` (block == dim, split_at_mut 経路) と `i < n-1`
    /// (par_chunks_mut 経路) の両方を踏むよう n と i を組み合わせる.
    #[cfg(feature = "rayon")]
    #[test]
    fn apply_single_mode_axis_i_rayon_matches_serial() {
        for n in [3usize, 6, 10, 12] {
            let dim = 1usize << n;
            for seed in [3u64, 31, 0xcafe_babe] {
                let mut rng = Xor64::new(seed.wrapping_add(n as u64));
                let psi0: Vec<Complex64> = (0..dim).map(|_| rng.complex_signed()).collect();
                for i in 0..n {
                    let u = random_unitary_2x2(&mut rng);

                    let mut psi_serial = psi0.clone();
                    apply_single_mode_axis_i_serial(&mut psi_serial, &u, i, n);

                    let mut psi_par = psi0.clone();
                    apply_single_mode_axis_i_rayon(&mut psi_par, &u, i, n);

                    for k in 0..dim {
                        assert_eq!(
                            psi_par[k].re.to_bits(),
                            psi_serial[k].re.to_bits(),
                            "n={}, i={}, seed={}, k={}: rayon re-bits differ from serial",
                            n,
                            i,
                            seed,
                            k,
                        );
                        assert_eq!(
                            psi_par[k].im.to_bits(),
                            psi_serial[k].im.to_bits(),
                            "n={}, i={}, seed={}, k={}: rayon im-bits differ from serial",
                            n,
                            i,
                            seed,
                            k,
                        );
                    }
                }
            }
        }
    }

    /// 8 thread の rayon pool で `apply_h_kryanneal` を 100 回反復実行し,
    /// 結果が **毎回 bit-identical** であることを確認する (race condition
    /// 検出). 各 `y[k]` への書き込みは disjoint な chunk に閉じるため
    /// thread スケジュールに依らない決定性を保つ. issue #62 acceptance.
    #[cfg(feature = "rayon")]
    #[test]
    fn apply_h_kryanneal_rayon_determinism_8thread_100iter() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .expect("failed to build rayon pool with 8 threads");

        let n = 12;
        let dim = 1usize << n;
        let mut rng = Xor64::new(0xfeed_face_dead_beef);
        let h_x: Vec<f64> = (0..n).map(|_| rng.signed()).collect();
        let h_p_diag: Vec<f64> = (0..dim).map(|_| rng.signed()).collect();
        let v: Vec<Complex64> = (0..dim).map(|_| rng.complex_signed()).collect();
        let a_t = rng.signed();
        let b_t = rng.signed();

        // 1 回目を reference として保存し, 残り 99 回が全て bit-identical かを検証.
        let reference: Vec<Complex64> = pool.install(|| {
            let mut y = vec![Complex64::new(0.0, 0.0); dim];
            apply_h_kryanneal(&v, &mut y, &h_x, &h_p_diag, a_t, b_t, n);
            y
        });
        for iter in 1..100 {
            let actual: Vec<Complex64> = pool.install(|| {
                let mut y = vec![Complex64::new(0.0, 0.0); dim];
                apply_h_kryanneal(&v, &mut y, &h_x, &h_p_diag, a_t, b_t, n);
                y
            });
            for k in 0..dim {
                assert_eq!(
                    actual[k].re.to_bits(),
                    reference[k].re.to_bits(),
                    "iter={}, k={}: non-deterministic re bits",
                    iter,
                    k,
                );
                assert_eq!(
                    actual[k].im.to_bits(),
                    reference[k].im.to_bits(),
                    "iter={}, k={}: non-deterministic im bits",
                    iter,
                    k,
                );
            }
        }
    }

    /// 同 fuzz: `apply_single_mode_axis_i` を 8 thread pool × 100 反復で
    /// bit-identical 検証. `i = n-1` (split_at_mut path) を含む複数の i を踏む.
    #[cfg(feature = "rayon")]
    #[test]
    fn apply_single_mode_axis_i_rayon_determinism_8thread_100iter() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .expect("failed to build rayon pool with 8 threads");

        let n = 12;
        let dim = 1usize << n;
        let mut rng = Xor64::new(0xcafe_babe_face_feed);
        let psi0: Vec<Complex64> = (0..dim).map(|_| rng.complex_signed()).collect();

        for i in [0usize, 1, 2, 5, 10, 11] {
            // i=11 = n-1 で split_at_mut path, それ以外は par_chunks_mut path.
            let u = random_unitary_2x2(&mut rng);

            let reference: Vec<Complex64> = pool.install(|| {
                let mut psi = psi0.clone();
                apply_single_mode_axis_i(&mut psi, &u, i, n);
                psi
            });
            for iter in 1..100 {
                let actual: Vec<Complex64> = pool.install(|| {
                    let mut psi = psi0.clone();
                    apply_single_mode_axis_i(&mut psi, &u, i, n);
                    psi
                });
                for k in 0..dim {
                    assert_eq!(
                        actual[k].re.to_bits(),
                        reference[k].re.to_bits(),
                        "i={}, iter={}, k={}: non-deterministic re bits",
                        i,
                        iter,
                        k,
                    );
                    assert_eq!(
                        actual[k].im.to_bits(),
                        reference[k].im.to_bits(),
                        "i={}, iter={}, k={}: non-deterministic im bits",
                        i,
                        iter,
                        k,
                    );
                }
            }
        }
    }
}
