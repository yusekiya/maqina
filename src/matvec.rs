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

/// rayon chunk あたりの **最大** 要素数. y_chunk + v の partner block (高 i pass
/// で chunk 外を読む先) の **両方** が L2 cache に同時に乗ることを狙う.
///
/// # 値の見直し (Phase 6 C3, issue #64, 第 2 版)
///
/// PR #78 初版で `1 << 14 = 16K` (= 256 KB) を `1 << 13 = 8K` (= 128 KB) に
/// 縮めたが, Linux 本番 bench で `apply_h_kryanneal` の **N=18 が 0.69×,
/// N=20 が 0.91× regression**, N=22 では 1.09× 改善という mixed な結果に
/// なった (小 dim で並列度が落ちる方が L2 fit 改善より影響大). 旧値に戻す.
///
/// L2 pressure 仮説は **chunk_size を縮める代わりに `apply_h_kryanneal_rayon`
/// 既存の動的 chunk_size 計算 `(dim / (nth * 4)).clamp(MIN, MAX)` の MAX
/// 上限値で抑える** ことで間接的に効くことを期待する (thread 数が小さい
/// 環境では大きい chunk, 64 thread 環境では target が MAX 未満になるので
/// 自動的に小さい chunk を選ぶ仕組み).
#[cfg(feature = "rayon")]
const RAYON_CHUNK_MAX: usize = 1 << 14;

/// rayon chunk あたりの **最小** 要素数. closure / scheduling overhead を
/// 償却するため 64 要素 (cache-line 4 要素 × 16) 以上を保証する.
#[cfg(feature = "rayon")]
const RAYON_CHUNK_MIN: usize = 1 << 6;

/// SIMD bit-flip kernel が処理する最大 block サイズ (Complex64 要素数).
/// i=2 (mask=4) の block = `2 * mask = 8` Complex64 = 16 f64 = 4 × f64x4 を
/// 1 SIMD ループ内で処理する. rayon path で chunk_size をこの倍数に揃える
/// ことで, 各 chunk start (`idx * chunk_size`) が SIMD block 境界に揃い,
/// chunk-internal な v[k^mask] アクセス (i=0,1,2 で mask ≤ 4) が SIMD カーネル
/// の block-aligned 前提を満たす (issue #63 Phase 6 C2).
#[cfg(feature = "simd")]
const SIMD_BLOCK_MAX: usize = 8;

/// rayon dispatch を起動する **最小 dim 閾値**. これ未満では scalar 単スレッド
/// 経路 (`*_serial`) にフォールバックする (issue #68).
///
/// # 根拠 (issue #62 本番 bench, cpu_count=64 Linux サーバー, BLAS thread=1)
///
/// `apply_h_kryanneal` の speedup vs threads=1:
///
/// | N | dim | 2 threads | 4 threads | 評価 |
/// |---|---|---|---|---|
/// | 16 | 2^16 = 64K  | **0.57×** (regression) | 1.58× | rayon overhead が単スレッド計算時間を超える |
/// | 18 | 2^18 = 256K | 1.70× | 2.09× | 並列化が利得を出し始める |
/// | 20 | 2^20 = 1M   | 1.84× | 3.18× | 大きく positive |
///
/// 転換点が N=16-18 にあるので, 安全側の保守的閾値として `1 << 17` (128K
/// 要素 = 2 MB Complex64) を採用. dim < 128K (= N ≤ 16) は rayon を起動
/// しない. N=17 (dim=131072) で rayon dispatch に入る (`>=` 判定).
///
/// 値の再評価は issue #68 follow-up bench で行う想定. const なので
/// release rebuild が必要な点に注意.
#[cfg(feature = "rayon")]
const MIN_RAYON_DIM: usize = 1 << 17;

/// `apply_h_kryanneal` の bit-flip pass と `apply_single_mode_axis_i` の
/// 2×2 ユニタリ pair update の i ∈ {0, 1, 2} (stride 1/2/4 連続アクセス領域)
/// を `wide::f64x4` で SIMD 特化するカーネル群.
///
/// - `bitflip_i{0,1,2}`: `apply_h_kryanneal` の bit-flip pass
///   (`y[k] += coeff · v[k ^ mask]`) 用 (issue #63 Phase 6 C2).
/// - `single_mode_i{0,1,2}`: `apply_single_mode_axis_i` /
///   `apply_fused_axes_to_chunk` (Phase 6 C3) の 2×2 ユニタリ pair update
///   (`new_lo = u[0]·a + u[1]·b`, `new_hi = u[2]·a + u[3]·b`) 用
///   (issue #71 Phase 6 C2.5).
///
/// # bit-flip pass の設計
///
/// bit-flip pass の操作 `y[k] += coeff * v[k ^ mask]` は, `block = 2 * mask`
/// 単位で見ると **block 内の前半と後半を入れ替えた v を後半 / 前半に足し
/// 込む** パターンになる:
///
/// ```text
/// y_lo := y[base..base+mask]
/// y_hi := y[base+mask..base+block]
/// v_lo := v[base..base+mask]
/// v_hi := v[base+mask..base+block]
/// y_lo += coeff * v_hi
/// y_hi += coeff * v_lo
/// ```
///
/// 各 Complex64 は f64 2 要素なので, block を f64 view で見ると:
///
/// | i | mask | block (complex) | block (f64) | 各 half (f64) | f64x4 lanes / half |
/// |---|---|---|---|---|---|
/// | 0 | 1 | 2 | 4 | 2 | 1/2 (要 in-register lane swap) |
/// | 1 | 2 | 4 | 8 | 4 | 1 (in-register swap 不要, 半分が 1 SIMD reg) |
/// | 2 | 4 | 8 | 16 | 8 | 2 (半分が 2 SIMD reg) |
///
/// i=1, 2 は half がちょうど f64x4 (=256-bit) の倍数で並ぶため shuffle 不要で
/// 載る. i=0 のみ half が 2 f64 しかないので, 1 block を 1 × f64x4 で処理し
/// in-register lane swap (LLVM が `f64x4::new([..])` を vperm2f128 等に compile
/// 出力する想定) で対処する.
///
/// # 高 i との直交性
///
/// i ≥ 3 (stride ≥ 8) は load/store が cache line を跨ぐため SIMD vectorize の
/// 利得が小さい (`docs/design.md` §12 Phase 6 C2). このカーネル群は i ≤ 2 のみ
/// 提供し, i ≥ 3 は呼び出し側 (`apply_h_kryanneal_serial` /
/// `apply_h_kryanneal_rayon`) で scalar inner loop に fallback する.
///
/// # `apply_h_kryanneal_serial` / `apply_h_kryanneal_rayon` 共用
///
/// 同じ SIMD inner kernel を両 path から呼ぶ (issue #63 Implementation note,
/// issue #68 follow-up で導入された `MIN_RAYON_DIM` dispatch との直交性).
/// rayon path では各 chunk が `SIMD_BLOCK_MAX = 8` の倍数長になるよう
/// chunk_size を align し, chunk slice (= `&v[base..base+chunk_size]`,
/// `y_chunk`) をそのまま渡せばよい (i ≤ 2 で mask ≤ 4 < SIMD_BLOCK_MAX なので
/// chunk-internal な v[k ^ mask] 参照は chunk 内に閉じる).
///
/// # 数値同一性
///
/// SIMD 経路と scalar 経路の演算順序は完全には一致しない (SIMD 内で 4 lane
/// の積を並列にまとめて加算するため). しかし `wide::f64x4` の `*` `+` は
/// 各 lane を独立に演算するので, 各 `y[k]` への更新は scalar 経路と同じ
/// `y[k] + coeff * v[k ^ mask]` の単一 fadd になる. このため SIMD ON/OFF
/// 両ビルドで **bit-identical** な結果になることを期待する (テスト
/// `apply_h_kryanneal_simd_matches_serial` で検証).
///
/// `single_mode_iN` の方は 2×2 complex matmul を **complex broadcast +
/// in-register swizzle** で f64x4 化する: `u_k = u_k_re + i·u_k_im` の作用
/// `u_k · x` (`x ∈ Complex64`) を 2 lane (2 Complex64) 並列で
///
/// ```text
/// (u_k · x).re = u_k_re · x.re − u_k_im · x.im
/// (u_k · x).im = u_k_re · x.im + u_k_im · x.re
/// ```
///
/// と書き下し, `u_k_re_v = splat(u_k.re)`, `u_k_im_signed_v = [−u_k.im,
/// u_k.im, −u_k.im, u_k.im]`, `x_swap = re/im swap of x_pair` を用意して
///
/// ```text
/// u_k · x_pair = u_k_re_v · x_pair + u_k_im_signed_v · x_swap
/// ```
///
/// を 1 つの `mul_add` + 1 つの `*` で計算する. 全 4 個の `u[0..4]` 適用結果を
/// FMA で足し合わせると `new_lo`, `new_hi` (各 f64x4) が得られ, in-place で
/// 書き戻す. SIMD/scalar 経路の演算は各 `psi[k]` への
/// `u[0]·a + u[1]·b` (lo 側) または `u[2]·a + u[3]·b` (hi 側) の単一値生成で
/// 算術的に等価だが, FMA 折りたたみ ON/OFF や lane の演算順序差で ulp 差が
/// 出うるため数値一致は `rel < 1e-13` で評価する (テスト
/// `simd_single_mode_kernels_match_scalar_fuzz_100iter`).
#[cfg(feature = "simd")]
mod simd_kernels {
    use num_complex::Complex64;
    use wide::f64x4;

    /// 4 連続 f64 を 256-bit unaligned load で `f64x4` に取り込む.
    ///
    /// `wide::f64x4` は `#[repr(C, align(32))]` で内部 SIMD register (AVX `__m256d`
    /// 等) と layout が一致するため, `ptr::read_unaligned` で memcpy された 32-byte
    /// 領域を直接 `f64x4` 値として解釈できる. AVX target では LLVM が **1 個の
    /// vmovupd** に折り畳む (旧版の `f64x4::new([ptr[0], ptr[1], ptr[2], ptr[3]])`
    /// 要素 load は 4 個の scalar load + insert に展開され実 SIMD にならない
    /// ことがあった).
    ///
    /// # Safety
    /// `ptr` は **少なくとも 32 bytes (= 4 個の `f64`) が連続して読み出せる**
    /// 領域を指していること. alignment は不要 (`read_unaligned` は 1-byte align
    /// でも sound).
    #[inline(always)]
    unsafe fn load_f64x4_unaligned(ptr: *const f64) -> f64x4 {
        // SAFETY: 呼び出し側が 4 f64 readable を保証. wide::f64x4 の layout は
        // align(32), size(32), 内容は f64×4 と同じビットパターンなので
        // read_unaligned で正しい f64x4 値が再構成される.
        unsafe { std::ptr::read_unaligned(ptr as *const f64x4) }
    }

    /// `f64x4` 値を 4 連続 f64 へ 256-bit unaligned store する.
    ///
    /// AVX target では LLVM が **1 個の vmovupd** に折り畳む. 旧版の
    /// `to_array()` + `copy_from_slice` パターンは 4 個の scalar store
    /// になり得る.
    ///
    /// # Safety
    /// `ptr` は **少なくとも 32 bytes が書き込み可能** であること.
    /// alignment は不要.
    #[inline(always)]
    unsafe fn store_f64x4_unaligned(ptr: *mut f64, val: f64x4) {
        // SAFETY: 呼び出し側が 4 f64 writable を保証.
        unsafe { std::ptr::write_unaligned(ptr as *mut f64x4, val) }
    }

    /// `&[Complex64]` を `&[f64]` (長さ 2 倍) として view する.
    ///
    /// # Safety
    /// `Complex64` (= `num_complex::Complex<f64>`) は `#[repr(C)]` で `re: f64`,
    /// `im: f64` の 2 フィールド構造体なので, メモリレイアウトは `[f64; 2]` と
    /// 一致する. align も 8 で共通. このため `&[Complex64]` を長さ `2 * len` の
    /// `&[f64]` として alias して読むことは sound.
    #[inline]
    fn as_f64_slice(v: &[Complex64]) -> &[f64] {
        // SAFETY: 上記コメント参照.
        unsafe { std::slice::from_raw_parts(v.as_ptr() as *const f64, v.len() * 2) }
    }

    /// `&mut [Complex64]` を `&mut [f64]` として view する.
    ///
    /// # Safety
    /// `as_f64_slice` と同じレイアウト前提. mut 排他性も `&mut [Complex64]`
    /// から派生するため一意に保たれる.
    #[inline]
    fn as_f64_slice_mut(y: &mut [Complex64]) -> &mut [f64] {
        // SAFETY: 上記コメント参照.
        unsafe { std::slice::from_raw_parts_mut(y.as_mut_ptr() as *mut f64, y.len() * 2) }
    }

    /// `y[k] += coeff * v[k ^ 1]` を `k ∈ [0, len)` の全範囲に適用する
    /// SIMD 特化版 (i=0, mask=1, block=2 Complex64 = 4 f64).
    ///
    /// `len = v.len() = y.len()` は `>= 2` かつ `2` の倍数であること
    /// (block-aligned 前提). `dim = 2^n` (n ≥ 1) の serial path および rayon
    /// path の `SIMD_BLOCK_MAX = 8` 倍数 chunk のいずれでも自動的に満たされる.
    ///
    /// # 命令計画 (AVX2 + FMA target)
    /// 1 block あたり: vmovupd (v load) → vperm2f128 (128-bit half swap) →
    /// vmovupd (y load) → vfmadd231pd (`y + coeff * v_swap`) → vmovupd (store).
    /// i=0 のみ in-register half-swap が必要で, LLVM は `to_array()` + `new()`
    /// 経由の reorder を vperm2f128 1 命令に折り畳む.
    #[inline]
    pub(super) fn bitflip_i0(v: &[Complex64], y: &mut [Complex64], coeff: f64) {
        debug_assert_eq!(v.len(), y.len(), "v and y must have equal length");
        debug_assert!(v.len() >= 2, "len must be >= 2 (i=0 block)");
        debug_assert_eq!(v.len() % 2, 0, "len must be a multiple of 2 (i=0 block)");

        let coeff_v = f64x4::splat(coeff);
        let v_f64 = as_f64_slice(v);
        let y_f64 = as_f64_slice_mut(y);
        // 1 block = 2 Complex64 = 4 f64 = 1 × f64x4.
        for (v_chunk, y_chunk) in v_f64.chunks_exact(4).zip(y_f64.chunks_exact_mut(4)) {
            // SAFETY: chunks_exact(4) は各 chunk の長さを 4 (= 32 bytes f64) と
            // 保証する. load_f64x4_unaligned / store_f64x4_unaligned の
            // precondition を満たす.
            unsafe {
                // v_raw = [v[0].re, v[0].im, v[1].re, v[1].im] を 1 vmovupd で load.
                let v_raw = load_f64x4_unaligned(v_chunk.as_ptr());
                // 128-bit lane swap で半分を入れ替え (vperm2f128 想定):
                //   [a, b, c, d] -> [c, d, a, b]
                let arr = v_raw.to_array();
                let v_swap = f64x4::new([arr[2], arr[3], arr[0], arr[1]]);
                // y_load を 1 vmovupd で load.
                let y_load = load_f64x4_unaligned(y_chunk.as_ptr());
                // vfmadd231pd: new_y = coeff_v * v_swap + y_load (single FMA op
                // on FMA-enabled CPU; fallback = mul + add).
                let new_y = coeff_v.mul_add(v_swap, y_load);
                // 1 vmovupd で store.
                store_f64x4_unaligned(y_chunk.as_mut_ptr(), new_y);
            }
        }
    }

    /// `y[k] += coeff * v[k ^ 2]` を `k ∈ [0, len)` の全範囲に適用する
    /// SIMD 特化版 (i=1, mask=2, block=4 Complex64 = 8 f64).
    ///
    /// `len = v.len() = y.len()` は `>= 4` かつ `4` の倍数であること.
    ///
    /// # 命令計画
    /// 1 block (= 2 × f64x4) あたり: 2 × vmovupd (v_lo, v_hi load) →
    /// 2 × vmovupd (y_lo, y_hi load) → 2 × vfmadd231pd → 2 × vmovupd (store).
    /// shuffle 不要 (half がちょうど 1 × f64x4 に収まる).
    #[inline]
    pub(super) fn bitflip_i1(v: &[Complex64], y: &mut [Complex64], coeff: f64) {
        debug_assert_eq!(v.len(), y.len(), "v and y must have equal length");
        debug_assert!(v.len() >= 4, "len must be >= 4 (i=1 block)");
        debug_assert_eq!(v.len() % 4, 0, "len must be a multiple of 4 (i=1 block)");

        let coeff_v = f64x4::splat(coeff);
        let v_f64 = as_f64_slice(v);
        let y_f64 = as_f64_slice_mut(y);
        for (v_chunk, y_chunk) in v_f64.chunks_exact(8).zip(y_f64.chunks_exact_mut(8)) {
            // SAFETY: chunks_exact(8) で各 chunk 長 8 f64 = 2 × 32 bytes が保証
            // される. 各 load/store の 4-f64 範囲は chunk 内に収まる.
            unsafe {
                let v_lo = load_f64x4_unaligned(v_chunk.as_ptr());
                let v_hi = load_f64x4_unaligned(v_chunk.as_ptr().add(4));
                let y_lo = load_f64x4_unaligned(y_chunk.as_ptr());
                let y_hi = load_f64x4_unaligned(y_chunk.as_ptr().add(4));
                // y_lo += coeff * v_hi, y_hi += coeff * v_lo.
                let new_lo = coeff_v.mul_add(v_hi, y_lo);
                let new_hi = coeff_v.mul_add(v_lo, y_hi);
                store_f64x4_unaligned(y_chunk.as_mut_ptr(), new_lo);
                store_f64x4_unaligned(y_chunk.as_mut_ptr().add(4), new_hi);
            }
        }
    }

    /// `y[k] += coeff * v[k ^ 4]` を `k ∈ [0, len)` の全範囲に適用する
    /// SIMD 特化版 (i=2, mask=4, block=8 Complex64 = 16 f64).
    ///
    /// `len = v.len() = y.len()` は `>= 8` かつ `8` の倍数であること.
    /// この block サイズが `SIMD_BLOCK_MAX = 8` (Complex64 単位) に対応し,
    /// rayon path の chunk_size align 基準になる.
    ///
    /// # 命令計画
    /// 1 block (= 4 × f64x4) あたり: 4 × vmovupd (v load) →
    /// 4 × vmovupd (y load) → 4 × vfmadd231pd → 4 × vmovupd (store).
    /// shuffle 不要.
    #[inline]
    pub(super) fn bitflip_i2(v: &[Complex64], y: &mut [Complex64], coeff: f64) {
        debug_assert_eq!(v.len(), y.len(), "v and y must have equal length");
        debug_assert!(v.len() >= 8, "len must be >= 8 (i=2 block)");
        debug_assert_eq!(v.len() % 8, 0, "len must be a multiple of 8 (i=2 block)");

        let coeff_v = f64x4::splat(coeff);
        let v_f64 = as_f64_slice(v);
        let y_f64 = as_f64_slice_mut(y);
        for (v_chunk, y_chunk) in v_f64.chunks_exact(16).zip(y_f64.chunks_exact_mut(16)) {
            // SAFETY: chunks_exact(16) で各 chunk 長 16 f64 = 4 × 32 bytes が保証
            // される. 各 load/store の 4-f64 範囲は chunk 内に収まる.
            unsafe {
                // 前半 (低 4 complex = 8 f64): 2 × f64x4
                let v_lo_a = load_f64x4_unaligned(v_chunk.as_ptr());
                let v_lo_b = load_f64x4_unaligned(v_chunk.as_ptr().add(4));
                // 後半 (高 4 complex = 8 f64): 2 × f64x4
                let v_hi_a = load_f64x4_unaligned(v_chunk.as_ptr().add(8));
                let v_hi_b = load_f64x4_unaligned(v_chunk.as_ptr().add(12));
                let y_lo_a = load_f64x4_unaligned(y_chunk.as_ptr());
                let y_lo_b = load_f64x4_unaligned(y_chunk.as_ptr().add(4));
                let y_hi_a = load_f64x4_unaligned(y_chunk.as_ptr().add(8));
                let y_hi_b = load_f64x4_unaligned(y_chunk.as_ptr().add(12));
                // y_lo += coeff * v_hi, y_hi += coeff * v_lo (各 half 2 × f64x4).
                let new_lo_a = coeff_v.mul_add(v_hi_a, y_lo_a);
                let new_lo_b = coeff_v.mul_add(v_hi_b, y_lo_b);
                let new_hi_a = coeff_v.mul_add(v_lo_a, y_hi_a);
                let new_hi_b = coeff_v.mul_add(v_lo_b, y_hi_b);
                store_f64x4_unaligned(y_chunk.as_mut_ptr(), new_lo_a);
                store_f64x4_unaligned(y_chunk.as_mut_ptr().add(4), new_lo_b);
                store_f64x4_unaligned(y_chunk.as_mut_ptr().add(8), new_hi_a);
                store_f64x4_unaligned(y_chunk.as_mut_ptr().add(12), new_hi_b);
            }
        }
    }

    // ====================================================================
    // single_mode_iN: 2×2 complex matmul を broadcast + swizzle で SIMD 化.
    //
    // `apply_single_mode_axis_i` の規約: `(psi[lo], psi[hi]) = (a, b)` に
    // 対し `(new_lo, new_hi) = (u[0]·a + u[1]·b, u[2]·a + u[3]·b)` を
    // in-place 書き込み. `mask = 1 << i`, `block = 2·mask` で base を
    // block 単位に進めて全ペアを走査する.
    //
    // 各 Complex64 = 2 f64 で, 1 つの `f64x4` には 2 個の Complex64 を載せて
    // 2 ペア並列で update する. broadcast 形の `u_k_re_v = splat(u[k].re)` と
    // `u_k_im_signed_v = [−u[k].im, u[k].im, −u[k].im, u[k].im]`, そして
    // x の re/im を入れ替えた `x_swap` を使って:
    //
    //   u_k · x_pair = u_k_re_v · x_pair + u_k_im_signed_v · x_swap   (1 FMA)
    //
    // を計算し, `new_lo = u[0]·A + u[1]·B`, `new_hi = u[2]·A + u[3]·B`
    // (各 f64x4) を 1 block の出力として store する.
    //
    // bit-flip 系 (`bitflip_iN`) と違って `apply_single_mode_axis_i` は
    // **in-place** なので, 1 block 内で load → compute → store の順を守る
    // 必要がある (compute 後に書き戻した psi を同じ iteration で再 load しない).
    // chunks_exact_mut(...) の各 chunk 内では load を全て先に行い, store は
    // 最後にまとめるので問題ない.
    // ====================================================================

    /// 4 つの `u[k]` から `(u_re_v[k], u_im_signed_v[k])` ペアを一括生成する
    /// ヘルパ. 4 個 × 2 = 8 個の f64x4 定数を kernel 入口で 1 回だけ準備する.
    #[inline]
    fn unpack_u_broadcast(u: &[Complex64; 4]) -> [(f64x4, f64x4); 4] {
        // 各 u[k] = u[k].re + i·u[k].im について:
        //   u_re_v       = splat(u[k].re)
        //   u_im_signed_v = [-u[k].im, u[k].im, -u[k].im, u[k].im]
        // を用意する. SIMD 内では:
        //   u[k] · x_pair = u_re_v · x_pair + u_im_signed_v · x_swap
        // で 2 lane (2 Complex64) 並列に複素積を計算する.
        let mut out: [(f64x4, f64x4); 4] = [(f64x4::splat(0.0), f64x4::splat(0.0)); 4];
        for (k, uk) in u.iter().enumerate() {
            out[k] = (
                f64x4::splat(uk.re),
                f64x4::new([-uk.im, uk.im, -uk.im, uk.im]),
            );
        }
        out
    }

    /// 1 個の `f64x4` (= 2 Complex64) の re/im を要素内で入れ替える.
    ///
    /// `[x0.re, x0.im, x1.re, x1.im] -> [x0.im, x0.re, x1.im, x1.re]`.
    /// LLVM は AVX target で `vpermilpd` (immediate=0x05) 1 命令に折り畳む.
    #[inline(always)]
    fn swap_reim(x: f64x4) -> f64x4 {
        let a = x.to_array();
        f64x4::new([a[1], a[0], a[3], a[2]])
    }

    /// `psi` の axis i=0 (mask=1, block=2 Complex64) ペアに 2×2 ユニタリ u を
    /// in-place 適用する SIMD 特化版 (Phase 6 C2.5, issue #71).
    ///
    /// `psi.len()` は `>= 4` かつ `4` の倍数であること. 1 SIMD iteration で
    /// **2 連続 block (= 4 Complex64 = 8 f64 = 2 × f64x4)** を処理する. 各
    /// block には 1 ペアしか入らないので, 2 ペアを 1 つの `f64x4` に集めるため
    /// **deinterleave (lo half load + hi half load → A=[a0,a1], B=[b0,b1])**
    /// と書き戻し時の **interleave (new_lo, new_hi → block0, block1)** が必要.
    /// LLVM は AVX target で各々 `vperm2f128` 1 命令に折り畳む.
    ///
    /// `apply_single_mode_axis_i_serial` の `n=1` (dim=2) 退化ケースは
    /// `len < 4` で SIMD 経路をスキップして scalar fallback に流れる.
    #[inline]
    pub(super) fn single_mode_i0(psi: &mut [Complex64], u: &[Complex64; 4]) {
        debug_assert!(psi.len() >= 4, "len must be >= 4 (2 i=0 blocks)");
        debug_assert_eq!(
            psi.len() % 4,
            0,
            "len must be a multiple of 4 (2 i=0 blocks)"
        );

        let ub = unpack_u_broadcast(u);
        let psi_f64 = as_f64_slice_mut(psi);
        // 1 SIMD iter = 2 blocks = 4 Complex64 = 8 f64.
        for psi_chunk in psi_f64.chunks_exact_mut(8) {
            // SAFETY: chunks_exact_mut(8) で 8 f64 = 64 bytes が保証される.
            unsafe {
                // block 0 = [a0.re, a0.im, b0.re, b0.im], block 1 = [a1, b1].
                let blk0 = load_f64x4_unaligned(psi_chunk.as_ptr());
                let blk1 = load_f64x4_unaligned(psi_chunk.as_ptr().add(4));
                let blk0_arr = blk0.to_array();
                let blk1_arr = blk1.to_array();
                // Deinterleave: A = [a0, a1] (lo halves), B = [b0, b1] (hi halves).
                let a_v = f64x4::new([blk0_arr[0], blk0_arr[1], blk1_arr[0], blk1_arr[1]]);
                let b_v = f64x4::new([blk0_arr[2], blk0_arr[3], blk1_arr[2], blk1_arr[3]]);
                let a_swap = swap_reim(a_v);
                let b_swap = swap_reim(b_v);

                // u[k] · A / B を 2 ペア並列で.
                let u0_a = ub[0].0.mul_add(a_v, ub[0].1 * a_swap);
                let u1_b = ub[1].0.mul_add(b_v, ub[1].1 * b_swap);
                let u2_a = ub[2].0.mul_add(a_v, ub[2].1 * a_swap);
                let u3_b = ub[3].0.mul_add(b_v, ub[3].1 * b_swap);

                let new_lo = u0_a + u1_b;
                let new_hi = u2_a + u3_b;

                // Interleave back: block 0 = [new_a0, new_b0], block 1 = [new_a1, new_b1].
                let lo_arr = new_lo.to_array();
                let hi_arr = new_hi.to_array();
                let out0 = f64x4::new([lo_arr[0], lo_arr[1], hi_arr[0], hi_arr[1]]);
                let out1 = f64x4::new([lo_arr[2], lo_arr[3], hi_arr[2], hi_arr[3]]);
                store_f64x4_unaligned(psi_chunk.as_mut_ptr(), out0);
                store_f64x4_unaligned(psi_chunk.as_mut_ptr().add(4), out1);
            }
        }
    }

    /// `psi` の axis i=1 (mask=2, block=4 Complex64 = 8 f64) ペアに 2×2 ユニタリ
    /// u を in-place 適用する SIMD 特化版.
    ///
    /// `psi.len()` は `>= 4` かつ `4` の倍数であること. 1 SIMD iteration で
    /// **1 block (= 4 Complex64 = 8 f64 = 2 × f64x4)** を処理する. lo_half と
    /// hi_half がちょうど 1 個ずつの f64x4 に乗るので **deinterleave 不要**
    /// (i=0 と違って shuffle が要らない).
    #[inline]
    pub(super) fn single_mode_i1(psi: &mut [Complex64], u: &[Complex64; 4]) {
        debug_assert!(psi.len() >= 4, "len must be >= 4 (i=1 block)");
        debug_assert_eq!(psi.len() % 4, 0, "len must be a multiple of 4 (i=1 block)");

        let ub = unpack_u_broadcast(u);
        let psi_f64 = as_f64_slice_mut(psi);
        // 1 SIMD iter = 1 block = 4 Complex64 = 8 f64. lo_half / hi_half がそれぞれ
        // 1 つの f64x4 に乗る.
        for psi_chunk in psi_f64.chunks_exact_mut(8) {
            // SAFETY: chunks_exact_mut(8).
            unsafe {
                let a_v = load_f64x4_unaligned(psi_chunk.as_ptr());
                let b_v = load_f64x4_unaligned(psi_chunk.as_ptr().add(4));
                let a_swap = swap_reim(a_v);
                let b_swap = swap_reim(b_v);

                let u0_a = ub[0].0.mul_add(a_v, ub[0].1 * a_swap);
                let u1_b = ub[1].0.mul_add(b_v, ub[1].1 * b_swap);
                let u2_a = ub[2].0.mul_add(a_v, ub[2].1 * a_swap);
                let u3_b = ub[3].0.mul_add(b_v, ub[3].1 * b_swap);

                let new_lo = u0_a + u1_b;
                let new_hi = u2_a + u3_b;

                store_f64x4_unaligned(psi_chunk.as_mut_ptr(), new_lo);
                store_f64x4_unaligned(psi_chunk.as_mut_ptr().add(4), new_hi);
            }
        }
    }

    /// `psi` の axis i=2 (mask=4, block=8 Complex64 = 16 f64) ペアに 2×2
    /// ユニタリ u を in-place 適用する SIMD 特化版.
    ///
    /// `psi.len()` は `>= 8` かつ `8` の倍数であること. 1 SIMD iteration で
    /// **1 block (= 8 Complex64 = 16 f64 = 4 × f64x4)** を処理する. lo_half
    /// (4 Complex64) と hi_half (4 Complex64) がそれぞれ 2 個の f64x4 に乗る.
    /// この block サイズが `SIMD_BLOCK_MAX = 8` (Complex64 単位) に対応する.
    #[inline]
    pub(super) fn single_mode_i2(psi: &mut [Complex64], u: &[Complex64; 4]) {
        debug_assert!(psi.len() >= 8, "len must be >= 8 (i=2 block)");
        debug_assert_eq!(psi.len() % 8, 0, "len must be a multiple of 8 (i=2 block)");

        let ub = unpack_u_broadcast(u);
        let psi_f64 = as_f64_slice_mut(psi);
        // 1 SIMD iter = 1 block = 8 Complex64 = 16 f64 = 4 × f64x4.
        // lo_half [0..8 f64] = 2 × f64x4 (= a 軸), hi_half [8..16 f64] = 2 × f64x4
        // (= b 軸).
        for psi_chunk in psi_f64.chunks_exact_mut(16) {
            // SAFETY: chunks_exact_mut(16).
            unsafe {
                let a_lo = load_f64x4_unaligned(psi_chunk.as_ptr());
                let a_hi = load_f64x4_unaligned(psi_chunk.as_ptr().add(4));
                let b_lo = load_f64x4_unaligned(psi_chunk.as_ptr().add(8));
                let b_hi = load_f64x4_unaligned(psi_chunk.as_ptr().add(12));
                let a_lo_swap = swap_reim(a_lo);
                let a_hi_swap = swap_reim(a_hi);
                let b_lo_swap = swap_reim(b_lo);
                let b_hi_swap = swap_reim(b_hi);

                // u[0]·A + u[1]·B → new_lo (各 2 × f64x4)
                let u0_a_lo = ub[0].0.mul_add(a_lo, ub[0].1 * a_lo_swap);
                let u0_a_hi = ub[0].0.mul_add(a_hi, ub[0].1 * a_hi_swap);
                let u1_b_lo = ub[1].0.mul_add(b_lo, ub[1].1 * b_lo_swap);
                let u1_b_hi = ub[1].0.mul_add(b_hi, ub[1].1 * b_hi_swap);
                let new_lo_a = u0_a_lo + u1_b_lo;
                let new_lo_b = u0_a_hi + u1_b_hi;

                // u[2]·A + u[3]·B → new_hi
                let u2_a_lo = ub[2].0.mul_add(a_lo, ub[2].1 * a_lo_swap);
                let u2_a_hi = ub[2].0.mul_add(a_hi, ub[2].1 * a_hi_swap);
                let u3_b_lo = ub[3].0.mul_add(b_lo, ub[3].1 * b_lo_swap);
                let u3_b_hi = ub[3].0.mul_add(b_hi, ub[3].1 * b_hi_swap);
                let new_hi_a = u2_a_lo + u3_b_lo;
                let new_hi_b = u2_a_hi + u3_b_hi;

                store_f64x4_unaligned(psi_chunk.as_mut_ptr(), new_lo_a);
                store_f64x4_unaligned(psi_chunk.as_mut_ptr().add(4), new_lo_b);
                store_f64x4_unaligned(psi_chunk.as_mut_ptr().add(8), new_hi_a);
                store_f64x4_unaligned(psi_chunk.as_mut_ptr().add(12), new_hi_b);
            }
        }
    }
}

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
///    accumulate. `coeff == 0` (`a_t == 0` または `h_x[i] == 0`) の場合は
///    その i pass を完全スキップする (数値的に no-op). sparse h_x で
///    支配的に効く最適化.
///
/// # 実装
/// `feature = "rayon"` (default ON) 時は **dim 閾値 dispatch**: `dim >=
/// MIN_RAYON_DIM` (= 1 << 17 = 128K 要素) のときだけ [`apply_h_kryanneal_rayon`]
/// が呼ばれ, それ未満では [`apply_h_kryanneal_serial`] にフォールバックする
/// (issue #68: 小 dim では rayon barrier overhead が単スレッド計算時間を超え
/// て regression する). `apply_h_kryanneal_rayon` 内では `y` を `par_chunks_mut`
/// で分割し chunk closure 内で diag + 全 i bit-flip pass を完走する. 各 `y[k]`
/// への書き込みは単一スレッドからしか発生せず, `v` は read-only のため
/// race-free. 演算順序は chunk 内で serial と同じ (diag → i=0 → i=1 → ...)
/// なので **rayon あり/なし両ビルドで bit-identical** に y[k] を生成する
/// (詳細は `apply_h_kryanneal_rayon_matches_serial_*` テスト).
/// `--no-default-features` 時は常に [`apply_h_kryanneal_serial`] にフォール
/// バック.
///
/// `feature = "simd"` (default ON, Phase 6 C2 / issue #63) では bit-flip pass
/// の i ∈ {0, 1, 2} (stride 1/2/4 連続アクセス領域) を [`simd_kernels`] の
/// `wide::f64x4` 特化版に dispatch する. SIMD inner kernel は serial / rayon
/// の両 path から共通で呼び出され, `MIN_RAYON_DIM` dispatch と直交している
/// (per-thread 最適化なので rayon 並列化と重複なし). i ≥ 3 は scalar inner
/// loop のまま (stride ≥ 8 で SIMD vectorize の利得が小さく cache line を
/// 跨ぐ). rayon path では SIMD kernel の block-aligned 前提を満たすため
/// chunk_size を `SIMD_BLOCK_MAX = 8` Complex64 の倍数に丸める. SIMD と
/// scalar 経路は各 `y[k]` への単一の `coeff * v[k^mask] + y[k]` を独立 lane で
/// 並列実行するだけで演算順序は変わらないため, SIMD あり/なし両ビルドで
/// **bit-identical** な結果を期待する (`apply_h_kryanneal_simd_matches_serial`
/// テスト). 実 SIMD 速度向上は build 時の `target-cpu` 設定 (AVX2 / AVX-512 /
/// NEON 有効化) に依存し, default `x86_64` target では `wide` が scalar
/// fallback を選び正確性のみ提供する (`benchmarks/bench_simd_scaling.py` は
/// 本番 sweep 時に `RUSTFLAGS=-C target-cpu=native` 等を前提とする).
///
/// # Panics
/// - `v.len() != 1 << n`
/// - `y.len() != 1 << n`
/// - `h_x.len() != n`
/// - `h_p_diag.len() != 1 << n`
///
/// # 可視性
///
/// `pub(crate)` から `pub` に上げているのは, `src/lib.rs` の `pub mod
/// bench_api` 経由で in-tree binary (`src/bin/perf_apply_h.rs`) から呼べる
/// ようにするため (Phase 6 D follow-up の perf 計測用). Python 側 API は
/// 引き続き `apply_h_kryanneal_py` 経由なので外部影響なし.
pub fn apply_h_kryanneal(
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
        if dim < MIN_RAYON_DIM {
            apply_h_kryanneal_serial(v, y, h_x, h_p_diag, a_t, b_t, n);
        } else {
            apply_h_kryanneal_rayon(v, y, h_x, h_p_diag, a_t, b_t, n);
        }
    }
    #[cfg(not(feature = "rayon"))]
    {
        apply_h_kryanneal_serial(v, y, h_x, h_p_diag, a_t, b_t, n);
    }
}

/// `apply_h_kryanneal` の scalar 単スレッド実装. `feature = "rayon"` OFF
/// ビルドおよび `#[cfg(test)]` 経路から rayon 経路との数値同一性比較に使う.
///
/// `feature = "simd"` (default ON) では bit-flip pass のうち i ∈ {0, 1, 2}
/// (stride 1/2/4 連続アクセス領域) を [`simd_kernels`] の f64x4 特化版に
/// dispatch する (Phase 6 C2, issue #63). 残りの i ≥ 3 は scalar inner loop の
/// まま. `feature = "simd"` OFF では従来の scalar pass を全 i で使う.
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
        // coeff == 0 (a_t == 0 もしくは h_x[i] == 0) のときは数値的に no-op
        // (y[k] += 0 * v[k ^ mask] = y[k]) なので i pass 全体をスキップする.
        // sparse h_x (`bench_simd_scaling.py` の i012-focus 等) で支配的に効く
        // 最適化. `coeff == 0.0` は IEEE 754 で +0 / -0 両方に対し true.
        if coeff == 0.0 {
            continue;
        }
        let mask = 1usize << i;

        // SIMD dispatch for i ∈ {0, 1, 2}.
        #[cfg(feature = "simd")]
        {
            match i {
                0 => {
                    simd_kernels::bitflip_i0(v, y, coeff);
                    continue;
                }
                1 => {
                    simd_kernels::bitflip_i1(v, y, coeff);
                    continue;
                }
                2 => {
                    simd_kernels::bitflip_i2(v, y, coeff);
                    continue;
                }
                _ => {} // i ≥ 3: scalar inner loop 経路にフォールスルー.
            }
        }

        // Scalar inner loop: i ≥ 3 (mask ≥ 8) もしくは `simd` feature OFF.
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

    // `feature = "simd"` ON では SIMD kernel が block-aligned 入力を前提とする
    // ため, chunk_size を SIMD_BLOCK_MAX (=8 Complex64) の倍数に丸める. これに
    // より各 chunk start `idx * chunk_size` が SIMD block 境界に揃い, i ∈ {0,1,2}
    // で chunk-internal な v[k ^ mask] アクセスが SIMD カーネルの前提を満たす.
    // RAYON_CHUNK_MIN = 64 も SIMD_BLOCK_MAX の倍数なので, 丸めた後でも
    // chunk_size ≥ MIN が保たれる. SIMD OFF では従来の C1 chunking のまま.
    #[cfg(feature = "simd")]
    let chunk_size = chunk_size - (chunk_size % SIMD_BLOCK_MAX);

    y.par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(idx, y_chunk)| {
            let base = idx * chunk_size;
            let chunk_len = y_chunk.len();
            // 対角 pass: y_chunk[li] = b_t · H_p[k] · v[k].
            for (li, y_k) in y_chunk.iter_mut().enumerate() {
                let k = base + li;
                *y_k = Complex64::new(b_t * h_p_diag[k], 0.0) * v[k];
            }
            // bit-flip pass: 全 i を同一 chunk 内で完走 (y_chunk が L1 resident).
            for (i, &h_x_i) in h_x.iter().enumerate() {
                let coeff = -a_t * h_x_i;
                // coeff == 0 のときは数値的に no-op なのでスキップ. sparse h_x で
                // 支配的に効く最適化 (serial path と同じ; 詳細はそちらコメント).
                if coeff == 0.0 {
                    continue;
                }
                let mask = 1usize << i;

                // SIMD dispatch for i ∈ {0, 1, 2}: mask ≤ 4 < SIMD_BLOCK_MAX なので
                // chunk-internal な v[k ^ mask] (k ∈ [base, base+chunk_len)) が
                // chunk subslice 内に閉じる. base は SIMD_BLOCK_MAX 倍数で揃って
                // いるため SIMD kernel の block-aligned 前提も満たす.
                #[cfg(feature = "simd")]
                match i {
                    0 => {
                        simd_kernels::bitflip_i0(&v[base..base + chunk_len], y_chunk, coeff);
                        continue;
                    }
                    1 => {
                        simd_kernels::bitflip_i1(&v[base..base + chunk_len], y_chunk, coeff);
                        continue;
                    }
                    2 => {
                        simd_kernels::bitflip_i2(&v[base..base + chunk_len], y_chunk, coeff);
                        continue;
                    }
                    _ => {} // i ≥ 3: scalar inner loop 経路にフォールスルー.
                }

                // Scalar inner loop: i ≥ 3 (mask ≥ 8) は v[k ^ mask] が chunk 外に
                // 出ることがあるため full v 参照. `feature = "simd"` OFF では
                // 全 i がここに来る.
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
/// `feature = "rayon"` (default ON) では **dim 閾値 dispatch**:
/// `dim >= MIN_RAYON_DIM` のときだけ [`apply_single_mode_axis_i_rayon`] が
/// 呼ばれ, それ未満では [`apply_single_mode_axis_i_serial`] にフォールバック
/// (issue #68). rayon path では block 単位 (`2·mask`) で `par_chunks_mut`
/// 並列化される. 退化ケース (i = n-1, block = dim, chunk が 1 個になる) は
/// psi を上下半分に `split_at_mut` した上で
/// `par_iter_mut().zip(par_iter_mut())` のペア並列に切り替える. 各ペア
/// `(lo, hi)` は単一スレッドが処理し write は disjoint なので race-free +
/// rayon あり/なし両ビルドで bit-identical.
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
        if dim < MIN_RAYON_DIM {
            apply_single_mode_axis_i_serial(psi, u, i, n);
        } else {
            apply_single_mode_axis_i_rayon(psi, u, i, n);
        }
    }
    #[cfg(not(feature = "rayon"))]
    {
        apply_single_mode_axis_i_serial(psi, u, i, n);
    }
}

/// `apply_single_mode_axis_i` の scalar 単スレッド実装. テスト経路から
/// rayon 結果との bit-identical 比較に使う.
///
/// `feature = "simd"` (default ON, Phase 6 C2.5 / issue #71) では i ∈ {0,1,2}
/// (stride 1/2/4 連続アクセス領域) を [`simd_kernels::single_mode_i0`] /
/// `_i1` / `_i2` に dispatch する. `dim < SIMD min` の退化ケース (例: i=0
/// で n=1 → dim=2) は scalar 経路にフォールバック.
fn apply_single_mode_axis_i_serial(psi: &mut [Complex64], u: &[Complex64; 4], i: usize, n: usize) {
    let dim = 1usize << n;
    let mask = 1usize << i;
    let block = mask << 1; // 2 * mask

    #[cfg(feature = "simd")]
    {
        // SIMD dispatch for i ∈ {0,1,2}. min dim 要件:
        //   i=0 -> psi.len() >= 4 (2-block SIMD iter)
        //   i=1 -> psi.len() >= 4 (1-block = 2 × f64x4)
        //   i=2 -> psi.len() >= 8 (1-block = 4 × f64x4 = SIMD_BLOCK_MAX 単位)
        // dim は power-of-two なので, psi.len() >= min を満たせば倍数条件も自動成立.
        match i {
            0 if psi.len() >= 4 => {
                simd_kernels::single_mode_i0(psi, u);
                return;
            }
            1 if psi.len() >= 4 => {
                simd_kernels::single_mode_i1(psi, u);
                return;
            }
            2 if psi.len() >= 8 => {
                simd_kernels::single_mode_i2(psi, u);
                return;
            }
            _ => {}
        }
    }

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
        //
        // i ∈ {0,1,2} の SIMD 経路を `apply_single_mode_axis_i_serial` と
        // 揃えるため, `block == dim` のときも SIMD kernel を直接呼ぶ
        // (Phase 6 C2.5, issue #71). この分岐は dim = block = 2·mask が
        // 成立する退化ケースで, SIMD-target な i ∈ {0,1,2} では dim ≤ 8 と
        // 極小. MIN_RAYON_DIM = 2^17 を超える本番 dim では i = n-1 が常に
        // 17 以上で SIMD range 外なので, ここで rayon 並列性を失うことは
        // 実用上ない (テスト用の小 dim 直接呼び出しでのみ通る経路).
        #[cfg(feature = "simd")]
        {
            match i {
                0 if psi.len() >= 4 => {
                    simd_kernels::single_mode_i0(psi, u);
                    return;
                }
                1 if psi.len() >= 4 => {
                    simd_kernels::single_mode_i1(psi, u);
                    return;
                }
                2 if psi.len() >= 8 => {
                    simd_kernels::single_mode_i2(psi, u);
                    return;
                }
                _ => {}
            }
        }

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
            // ≥ RAYON_CHUNK_MIN を取る. block ≤ 8 では chunk_size = 64 で
            // SIMD_BLOCK_MAX = 8 の倍数を自動的に満たす (SIMD i ∈ {0,1,2}
            // dispatch の前提).
            RAYON_CHUNK_MIN.next_multiple_of(block)
        };
        let chunk_size = chunk_size.min(dim);
        psi.par_chunks_mut(chunk_size).for_each(|chunk| {
            // SIMD dispatch for i ∈ {0,1,2}: chunk_size = 64 (block ≤ 8 のとき)
            // なので i ≤ 2 で必要な min dim (4 または 8) と chunk.len() % SIMD
            // chunk size 倍数の要件は両方満たす. block == dim 経路は本 else 枝に
            // 入らない (上で分岐済み) ので i = n-1 退化ケースとは衝突しない.
            #[cfg(feature = "simd")]
            {
                match i {
                    0 if chunk.len() >= 4 => {
                        simd_kernels::single_mode_i0(chunk, u);
                        return;
                    }
                    1 if chunk.len() >= 4 => {
                        simd_kernels::single_mode_i1(chunk, u);
                        return;
                    }
                    2 if chunk.len() >= 8 => {
                        simd_kernels::single_mode_i2(chunk, u);
                        return;
                    }
                    _ => {}
                }
            }

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

/// Multi-qubit gate fusion (Phase 6 C3, issue #64) でサポートする最大 qubit 数.
///
/// `k = 6` は qsim の経験 (`max_fused_size = 4-5` 推奨, `lib/fuser_mqubit.h`)
/// から十分大きい上限値. 内部 u_list 配列は `[[Complex64; 4]; 6]` = `192 B`
/// で stack 確保に余裕で収まる (`docs/design.md` §5.1.3).
pub(crate) const MAX_FUSED_K: usize = 6;

/// `psi` の **連続** k qubit (i_start, i_start+1, ..., i_start+k-1) に
/// k 個の 2×2 ユニタリ `u_list` を **in-place** で fused apply する
/// (Phase 6 C3, multi-qubit gate fusion, issue #64).
///
/// # 用途
///
/// `trotter_step` で `Π_{i=0..n} R_i(dt)` を 1 軸ずつ
/// `apply_single_mode_axis_i` で in-place 適用すると per-step rayon barrier が
/// **2n+2** 個入って scaling が頭打ちになる (issue #68 follow-up bench で
/// 1.55× @ 16 threads で飽和観測). 本関数は kryanneal の
/// `H_drv = -Σ h_x_i X_i` が per-site で commuting であることを利用して
/// 連続 k qubit の R_i を **1 つの chunk closure 内で逐次適用** する. これに
/// より `trotter_step` の barrier 数は `2n+2 → n/k + 2` に縮み (n=20, k=4 で
/// 40 → 7), 同時に chunk が L2 resident な間に全 k pass を完走して DRAM
/// round trip も削減する.
///
/// # 設計判断 (dense matmul ではなく per-axis 逐次)
///
/// **以前の試行** (PR #78 初版): 連続 k qubit の tensor product を
/// `2^k × 2^k` dense matrix に畳んで chunk あたり 1 回の dense matmul で
/// 適用する qsim `MultiQubitGateFuser` 同型実装. **Linux サーバー本番 bench で
/// `trotter_step` が 0.81× regression** したため放棄. 理由: per-axis × k の
/// compute は `2k·dim` ops だが dense matmul は `2^k·dim` ops で k=4 のとき
/// 2× 多く, kryanneal の TFIM 規模では memory-bandwidth gain よりも compute
/// 増のほうが勝ったため.
///
/// **現実装**: chunk closure 内で k 個の axis に対して **per-axis 2-pair
/// update を逐次** 実行. compute は per-axis × k と同じ (`2k·dim` ops, 増え
/// ない). barrier は 1 per fused call (chunk_size 単位の `par_chunks_mut` 1 つ)
/// で `n/k 倍` 削減. chunk が L2 fit していれば全 k pass の `v[k^mask]` 参照
/// が cache resident → memory bandwidth bound も緩和.
///
/// # 数値同一性
///
/// chunk closure 内の per-axis update は [`apply_single_mode_axis_i_rayon`]
/// の inner kernel と **同じ演算順序** (`split_at_mut(mask)` + zip で low /
/// high half pair processing). このため `u_list = [u_{i_start}, u_{i_start+1},
/// ..., u_{i_start+k-1}]` で呼んだ結果は `apply_single_mode_axis_i` を
/// i_start..i_start+k で k 回順次呼ぶのと **bit-identical** (chunk_size が
/// 最大 mask `2^(i_start+k-1)` の整数倍を保証している前提).
///
/// # rayon 経路 (chunk_size 戦略)
///
/// `group_block = 2^(i_start+k)` は連続 k qubit pass のうち最大 mask の 2 倍
/// で, chunk_size はこの整数倍に取る必要がある (axis `i_start+k-1` の pair
/// が chunk 境界を跨がないように). 既存 [`apply_single_mode_axis_i_rayon`]
/// と同じ pattern で `chunk_size = group_block · floor(RAYON_CHUNK_MAX /
/// group_block)` (≥ group_block) を採用. これで chunk 数 `dim / chunk_size`
/// が thread 数に対して十分多くなり, 並列度を確保しつつ chunk が L2 fit する.
///
/// small dim (`dim < MIN_RAYON_DIM`) では serial fallback.
///
/// # Panics
/// - `psi.len() != 1 << n`
/// - `u_list.len() < 1` or `u_list.len() > MAX_FUSED_K`
/// - `i_start + u_list.len() > n`
pub(crate) fn apply_multi_qubit_gate_fused(
    psi: &mut [Complex64],
    u_list: &[[Complex64; 4]],
    i_start: usize,
    n: usize,
) {
    let dim = 1usize << n;
    let k = u_list.len();
    assert_eq!(psi.len(), dim, "psi must have length 2^n");
    assert!(
        (1..=MAX_FUSED_K).contains(&k),
        "u_list length k = {} must be in [1, {}]",
        k,
        MAX_FUSED_K,
    );
    assert!(
        i_start + k <= n,
        "i_start ({}) + k ({}) must be <= n ({})",
        i_start,
        k,
        n,
    );

    #[cfg(feature = "rayon")]
    {
        if dim >= MIN_RAYON_DIM {
            apply_multi_qubit_gate_fused_rayon(psi, u_list, i_start);
            return;
        }
    }
    apply_multi_qubit_gate_fused_serial(psi, u_list, i_start);
}

/// `apply_multi_qubit_gate_fused` の scalar 単スレッド実装. psi 全体を
/// 1 つの "chunk" とみなして per-axis 逐次 update を k 回実行する.
fn apply_multi_qubit_gate_fused_serial(
    psi: &mut [Complex64],
    u_list: &[[Complex64; 4]],
    i_start: usize,
) {
    apply_fused_axes_to_chunk(psi, u_list, i_start);
}

/// `apply_multi_qubit_gate_fused` の rayon 並列実装.
///
/// chunk_size を `group_block = 2^(i_start+k)` の整数倍で `RAYON_CHUNK_MAX`
/// 程度に揃え, 各 chunk closure 内で全 k axis を per-axis 2-pair update で
/// 順次適用する. axis 最大 mask `2^(i_start+k-1)` < `group_block` ≤
/// `chunk_size` なので全 axis pair が chunk 内に閉じる (chunk 境界跨ぎ無し).
#[cfg(feature = "rayon")]
fn apply_multi_qubit_gate_fused_rayon(
    psi: &mut [Complex64],
    u_list: &[[Complex64; 4]],
    i_start: usize,
) {
    let k = u_list.len();
    let group_block = 1usize << (i_start + k);

    // 既存 `apply_h_kryanneal_rayon` と同じ pattern で thread 数に応じた
    // chunk_size を取り, group_block (= 最大 axis の block 2 倍) の整数倍に
    // 揃える. これで dim が小さいときでも 64 thread に十分な chunk 数を
    // 配給できる (PR #78 v2 で N=18 が 0.69× regression したのを修正).
    let nth = rayon::current_num_threads().max(1);
    let target = (psi.len() / (nth * 4)).clamp(RAYON_CHUNK_MIN, RAYON_CHUNK_MAX);
    let chunk_size = if group_block >= target {
        group_block
    } else {
        let n_groups = (target / group_block).max(1);
        n_groups * group_block
    };
    let chunk_size = chunk_size.min(psi.len());

    psi.par_chunks_mut(chunk_size).for_each(|chunk| {
        apply_fused_axes_to_chunk(chunk, u_list, i_start);
    });
}

/// 1 つの chunk (長さは `2^(i_start+k)` の整数倍, 呼び出し側が保証) 内で
/// k 個の axis (i_start, ..., i_start+k-1) に対して **per-axis 2-pair update
/// を逐次** 実行する.
///
/// 各 axis i (= i_start + j) について `block = 2·mask = 2^(i+1)` 単位で
/// chunk 内を走り, `split_at_mut(mask)` で low / high ペアに分けて
/// `(psi[lo], psi[hi]) := (u[0]·psi[lo] + u[1]·psi[hi], u[2]·psi[lo] + u[3]·psi[hi])`
/// を計算する. [`apply_single_mode_axis_i_rayon`] の chunk closure 内 inner loop
/// と **同じ演算順序** なので, fused 版を `u_list = [u_{i_start}, ...,
/// u_{i_start+k-1}]` で呼んだ結果は per-axis × k と bit-identical になる.
#[inline]
fn apply_fused_axes_to_chunk(chunk: &mut [Complex64], u_list: &[[Complex64; 4]], i_start: usize) {
    for (j, u) in u_list.iter().enumerate() {
        let i = i_start + j;
        let mask = 1usize << i;
        let block = mask << 1; // = 2 * mask

        // SIMD dispatch for i ∈ {0,1,2} (Phase 6 C2.5, issue #71). C3 (#64) の
        // fused 経路は同型の 2-pair update を per-axis で逐次実行するので,
        // C2.5 の SIMD inner kernel を同じく流用できる. これで trotter_step
        // (連続 k=4 qubit fusion) の最初の 3 axis が SIMD で走る.
        //
        // fused rayon の chunk_size は `n_groups · group_block` (group_block =
        // 2^(i_start+k)) で構築されるが, dim/(nth·4) が非 power-of-2 のときに
        // n_groups が奇数になり chunk.len() が SIMD min unit (4 or 8 Complex64)
        // の倍数とならない場合があり得る. その時は scalar fallback に流す.
        #[cfg(feature = "simd")]
        {
            match i {
                0 if chunk.len() >= 4 && chunk.len().is_multiple_of(4) => {
                    simd_kernels::single_mode_i0(chunk, u);
                    continue;
                }
                1 if chunk.len() >= 4 && chunk.len().is_multiple_of(4) => {
                    simd_kernels::single_mode_i1(chunk, u);
                    continue;
                }
                2 if chunk.len() >= 8 && chunk.len().is_multiple_of(8) => {
                    simd_kernels::single_mode_i2(chunk, u);
                    continue;
                }
                _ => {}
            }
        }

        let mut local_base = 0usize;
        while local_base + block <= chunk.len() {
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
    fn sparse_h_x_matches_dense_reference() {
        // h_x にゼロ要素を混ぜたケース. apply_h_kryanneal の bit-flip pass は
        // `coeff == 0` (= h_x[i] == 0) の i pass をスキップする最適化を持つが,
        // それでも数値結果は dense H · v と rel < 1e-13 で一致するはず
        // (スキップした i の dense H への寄与は元々 0 なので等価).
        //
        // n を 18 (issue #63 acceptance の N と同じ) より小さい n=7 にして
        // テスト時間を抑える. dim=128 で SIMD path (i=0,1,2) と scalar path
        // (i=3..6) の両方を踏み, さらに i=1,3,5 を 0 にすることで短絡 path も
        // 同時に踏む. n を **奇数** にして h_x のゼロ pattern と非ゼロ pattern
        // が混在するよう調整 (i=0,2,4,6 が non-zero, i=1,3,5 が zero).
        let n = 7;
        let dim = 1usize << n;
        let mut rng = Xor64::new(0xc1c2_c3c4_c5c6_c7c8);
        let mut h_x: Vec<f64> = (0..n).map(|_| rng.signed()).collect();
        for i in [1, 3, 5] {
            h_x[i] = 0.0;
        }
        let h_p_diag: Vec<f64> = (0..dim).map(|_| rng.signed()).collect();
        let v: Vec<Complex64> = (0..dim).map(|_| rng.complex_signed()).collect();
        let a_t = rng.signed();
        let b_t = rng.signed();

        // 参照: dense H · v (build_dense_h は h_x[i]=0 で coeff=0 のため i pass
        // を寄与なしで足し込む形, apply_h_kryanneal の短絡経路と等価).
        let h_dense = build_dense_h(n, &h_x, &h_p_diag, a_t, b_t);
        let v_vec = DVector::<Complex64>::from_vec(v.clone());
        let y_expected = &h_dense * &v_vec;

        let mut y = vec![Complex64::new(0.0, 0.0); dim];
        apply_h_kryanneal(&v, &mut y, &h_x, &h_p_diag, a_t, b_t, n);

        let rel = relative_error(&y, &y_expected);
        assert!(
            rel < 1e-13,
            "sparse h_x: rel={} >= 1e-13 (短絡が壊れている可能性)",
            rel,
        );
    }

    #[test]
    fn a_t_zero_skips_all_bitflip_passes() {
        // a_t = 0 で coeff = -a_t · h_x[i] = 0 (全 i). 全 bit-flip pass を
        // 短絡で skip し, 結果は diag pass のみ (y[k] = b_t · H_p[k] · v[k]).
        let n = 5;
        let dim = 1usize << n;
        let mut rng = Xor64::new(0xa7a7_a7a7_a7a7_a7a7);
        let h_x: Vec<f64> = (0..n).map(|_| rng.signed()).collect();
        let h_p_diag: Vec<f64> = (0..dim).map(|_| rng.signed()).collect();
        let v: Vec<Complex64> = (0..dim).map(|_| rng.complex_signed()).collect();
        let a_t = 0.0; // ← 短絡を発動させる
        let b_t = rng.signed();

        let mut y = vec![Complex64::new(0.0, 0.0); dim];
        apply_h_kryanneal(&v, &mut y, &h_x, &h_p_diag, a_t, b_t, n);

        // 期待値: y[k] = b_t · H_p_diag[k] · v[k] (diag のみ).
        for k in 0..dim {
            let expected = Complex64::new(b_t * h_p_diag[k], 0.0) * v[k];
            let diff = (y[k] - expected).norm();
            assert!(
                diff < 1e-15 * (expected.norm() + 1.0),
                "k={}: a_t=0 short-circuit broken: actual={} vs diag={}",
                k,
                y[k],
                expected,
            );
        }
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
        // 注: public `apply_h_kryanneal` は dim < MIN_RAYON_DIM = 1<<17 で scalar
        // path に dispatch されるため (issue #68), rayon path 自体の決定性を
        // テストするには `apply_h_kryanneal_rayon` を直接呼ぶ必要がある.
        let reference: Vec<Complex64> = pool.install(|| {
            let mut y = vec![Complex64::new(0.0, 0.0); dim];
            apply_h_kryanneal_rayon(&v, &mut y, &h_x, &h_p_diag, a_t, b_t, n);
            y
        });
        for iter in 1..100 {
            let actual: Vec<Complex64> = pool.install(|| {
                let mut y = vec![Complex64::new(0.0, 0.0); dim];
                apply_h_kryanneal_rayon(&v, &mut y, &h_x, &h_p_diag, a_t, b_t, n);
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
            // 注: public `apply_single_mode_axis_i` は dim < MIN_RAYON_DIM で
            // scalar path に dispatch されるため (issue #68), rayon path 自体の
            // 決定性をテストするには `apply_single_mode_axis_i_rayon` を直接呼ぶ.
            let u = random_unitary_2x2(&mut rng);

            let reference: Vec<Complex64> = pool.install(|| {
                let mut psi = psi0.clone();
                apply_single_mode_axis_i_rayon(&mut psi, &u, i, n);
                psi
            });
            for iter in 1..100 {
                let actual: Vec<Complex64> = pool.install(|| {
                    let mut psi = psi0.clone();
                    apply_single_mode_axis_i_rayon(&mut psi, &u, i, n);
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

    // ===== SIMD bit-flip kernel (Phase 6 C2, issue #63) のテスト =====

    /// SIMD 特化版 ([`simd_kernels::bitflip_i0`] / `_i1` / `_i2`) と scalar 経路
    /// (`y[k] += coeff · v[k ^ mask]`) の数値同一性 fuzz.
    ///
    /// issue #63 acceptance: ランダム n ∈ {2..8}, ランダム入力で **100 回反復**,
    /// `rel < 1e-13`. 各 iteration で n, i ∈ {0, 1, 2} (capped at n-1), coeff,
    /// v 初期値, y 初期値をランダム化する.
    ///
    /// SIMD と scalar の演算は各 `y[k]` について同一の `y[k] + coeff *
    /// v[k ^ mask]` (lane 独立 mul + add, FMA を強制しない `*` `+` 経路) で
    /// 構成されているため, default build (target-cpu = generic) では
    /// **bit-identical** が成り立つが, ここでは将来 target-cpu=native + FMA で
    /// fused-mul-add に compile された場合の ulp 差を許容するため
    /// `rel < 1e-13` で評価する.
    #[cfg(feature = "simd")]
    #[test]
    fn simd_bitflip_kernels_match_scalar_fuzz_100iter() {
        let mut rng = Xor64::new(0xa1b2_c3d4_e5f6_0708);
        for iter in 0..100 {
            // n ∈ {2..=8}, dim = 2^n ∈ {4..256}.
            let n = 2 + (rng.next_u64() % 7) as usize;
            let dim = 1usize << n;
            // i ∈ {0..min(n-1, 2)}; SIMD カーネルは i ∈ {0, 1, 2} に特化.
            let i_cap = (n - 1).min(2);
            let i = (rng.next_u64() % (i_cap as u64 + 1)) as usize;
            let mask = 1usize << i;
            let coeff = rng.signed();
            let v: Vec<Complex64> = (0..dim).map(|_| rng.complex_signed()).collect();
            let y0: Vec<Complex64> = (0..dim).map(|_| rng.complex_signed()).collect();

            // SIMD 経路.
            let mut y_simd = y0.clone();
            match i {
                0 => simd_kernels::bitflip_i0(&v, &mut y_simd, coeff),
                1 => simd_kernels::bitflip_i1(&v, &mut y_simd, coeff),
                2 => simd_kernels::bitflip_i2(&v, &mut y_simd, coeff),
                _ => unreachable!("i is capped to {{0,1,2}}"),
            }

            // Scalar reference: `y[k] += coeff · v[k ^ mask]`.
            let mut y_scalar = y0;
            for k in 0..dim {
                y_scalar[k] += Complex64::new(coeff, 0.0) * v[k ^ mask];
            }

            // 相対誤差.
            let mut diff_sq = 0.0_f64;
            let mut ref_sq = 0.0_f64;
            for k in 0..dim {
                let d = y_simd[k] - y_scalar[k];
                diff_sq += d.norm_sqr();
                ref_sq += y_scalar[k].norm_sqr();
            }
            let rel = diff_sq.sqrt() / ref_sq.sqrt().max(1.0);
            assert!(
                rel < 1e-13,
                "iter={}, n={}, i={}, coeff={}: SIMD-vs-scalar rel={} >= 1e-13",
                iter,
                n,
                i,
                coeff,
                rel,
            );
        }
    }

    /// SIMD カーネルが小 dim 境界 (= block size に等しい dim) でも正しく動く.
    ///
    /// - i=0, n=1 (dim=2): block=2 で 1 block ぴったり.
    /// - i=1, n=2 (dim=4): block=4 で 1 block ぴったり.
    /// - i=2, n=3 (dim=8): block=8 で 1 block ぴったり.
    ///
    /// これらは `apply_h_kryanneal_serial` 経路から直接 SIMD カーネルに渡る
    /// 最小 dim ケース. 各々 scalar reference と要素ごとに比較する.
    #[cfg(feature = "simd")]
    #[test]
    fn simd_bitflip_kernels_min_dim_boundaries() {
        let mut rng = Xor64::new(0xbaba_face_dead_fade);
        let coeff = rng.signed();

        // i=0, dim=2 (block ぴったり).
        {
            let v: Vec<Complex64> = (0..2).map(|_| rng.complex_signed()).collect();
            let y0: Vec<Complex64> = (0..2).map(|_| rng.complex_signed()).collect();
            let mut y_simd = y0.clone();
            simd_kernels::bitflip_i0(&v, &mut y_simd, coeff);
            let mut y_scalar = y0;
            y_scalar[0] += Complex64::new(coeff, 0.0) * v[1];
            y_scalar[1] += Complex64::new(coeff, 0.0) * v[0];
            for k in 0..2 {
                let diff = (y_simd[k] - y_scalar[k]).norm();
                assert!(
                    diff < 1e-15,
                    "i=0, k={}: SIMD={} vs scalar={}",
                    k,
                    y_simd[k],
                    y_scalar[k]
                );
            }
        }

        // i=1, dim=4.
        {
            let v: Vec<Complex64> = (0..4).map(|_| rng.complex_signed()).collect();
            let y0: Vec<Complex64> = (0..4).map(|_| rng.complex_signed()).collect();
            let mut y_simd = y0.clone();
            simd_kernels::bitflip_i1(&v, &mut y_simd, coeff);
            let mut y_scalar = y0;
            for k in 0..4 {
                y_scalar[k] += Complex64::new(coeff, 0.0) * v[k ^ 2];
            }
            for k in 0..4 {
                let diff = (y_simd[k] - y_scalar[k]).norm();
                assert!(
                    diff < 1e-15,
                    "i=1, k={}: SIMD={} vs scalar={}",
                    k,
                    y_simd[k],
                    y_scalar[k]
                );
            }
        }

        // i=2, dim=8.
        {
            let v: Vec<Complex64> = (0..8).map(|_| rng.complex_signed()).collect();
            let y0: Vec<Complex64> = (0..8).map(|_| rng.complex_signed()).collect();
            let mut y_simd = y0.clone();
            simd_kernels::bitflip_i2(&v, &mut y_simd, coeff);
            let mut y_scalar = y0;
            for k in 0..8 {
                y_scalar[k] += Complex64::new(coeff, 0.0) * v[k ^ 4];
            }
            for k in 0..8 {
                let diff = (y_simd[k] - y_scalar[k]).norm();
                assert!(
                    diff < 1e-15,
                    "i=2, k={}: SIMD={} vs scalar={}",
                    k,
                    y_simd[k],
                    y_scalar[k]
                );
            }
        }
    }

    // ===== SIMD single-mode kernel (Phase 6 C2.5, issue #71) のテスト =====

    /// 2×2 SIMD single-mode kernel (`simd_kernels::single_mode_iN`) と
    /// scalar in-place 経路の数値同一性 fuzz.
    ///
    /// issue #71 acceptance: ランダム n ∈ {2..=8}, ランダム unitary, 100 iter,
    /// `rel < 1e-13`. 各 iteration で n, i ∈ {0,1,2} (capped at n-1), unitary,
    /// psi 初期値 をランダム化する. SIMD と scalar は同じ
    /// `(new_lo, new_hi) = (u[0]·a + u[1]·b, u[2]·a + u[3]·b)` を計算するが,
    /// SIMD 側は broadcast + swizzle で FMA 経路に lower するため ulp 差が
    /// 出うる. `rel < 1e-13` で評価する.
    #[cfg(feature = "simd")]
    #[test]
    fn simd_single_mode_kernels_match_scalar_fuzz_100iter() {
        let mut rng = Xor64::new(0x715e_c071_5ec0);
        for iter in 0..100 {
            // n ∈ {2..=8}, dim = 2^n ∈ {4..256}.
            let n = 2 + (rng.next_u64() % 7) as usize;
            let dim = 1usize << n;
            // i ∈ {0..min(n-1, 2)}; SIMD カーネルは i ∈ {0,1,2} に特化.
            let i_cap = (n - 1).min(2);
            let i = (rng.next_u64() % (i_cap as u64 + 1)) as usize;

            let u = random_unitary_2x2(&mut rng);
            let psi0: Vec<Complex64> = (0..dim).map(|_| rng.complex_signed()).collect();

            // SIMD 経路: kernel を直接呼ぶ.
            let mut psi_simd = psi0.clone();
            match i {
                0 => simd_kernels::single_mode_i0(&mut psi_simd, &u),
                1 => simd_kernels::single_mode_i1(&mut psi_simd, &u),
                2 => simd_kernels::single_mode_i2(&mut psi_simd, &u),
                _ => unreachable!("i is capped to {{0,1,2}}"),
            }

            // Scalar reference: 設計書 §5.1.2 の 2 重ループ形を直接実行.
            let mut psi_scalar = psi0;
            let mask = 1usize << i;
            let block = mask << 1;
            let mut base = 0usize;
            while base < dim {
                for offset in 0..mask {
                    let lo = base + offset;
                    let hi = lo + mask;
                    let a = psi_scalar[lo];
                    let b = psi_scalar[hi];
                    psi_scalar[lo] = u[0] * a + u[1] * b;
                    psi_scalar[hi] = u[2] * a + u[3] * b;
                }
                base += block;
            }

            // 相対誤差.
            let mut diff_sq = 0.0_f64;
            let mut ref_sq = 0.0_f64;
            for k in 0..dim {
                let d = psi_simd[k] - psi_scalar[k];
                diff_sq += d.norm_sqr();
                ref_sq += psi_scalar[k].norm_sqr();
            }
            let rel = diff_sq.sqrt() / ref_sq.sqrt().max(1.0);
            assert!(
                rel < 1e-13,
                "iter={}, n={}, i={}: SIMD-vs-scalar single_mode rel={} >= 1e-13",
                iter,
                n,
                i,
                rel,
            );
        }
    }

    /// SIMD single-mode カーネルの境界ケース (block ぴったりの最小 dim).
    ///
    /// - i=0: psi.len() = 4 (2 block ぴったり, SIMD i=0 の 1 SIMD iter).
    ///   n=1 (dim=2) は SIMD min 未満で `apply_single_mode_axis_i_serial`
    ///   側で scalar fallback に流れるため, ここでは直接 dim=4 を踏む.
    /// - i=1: psi.len() = 4 (1 block ぴったり).
    /// - i=2: psi.len() = 8 (1 block ぴったり = SIMD_BLOCK_MAX).
    ///
    /// 各々 scalar reference (in-place 2 重ループ) と要素ごとに比較する.
    #[cfg(feature = "simd")]
    #[test]
    fn simd_single_mode_kernels_min_dim_boundaries() {
        let mut rng = Xor64::new(0x_b0ad_1234_5678_9abc);

        // 共通: scalar reference を inline で書く.
        let run_scalar = |psi: &mut [Complex64], u: &[Complex64; 4], i: usize| {
            let mask = 1usize << i;
            let block = mask << 1;
            let dim = psi.len();
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
        };

        // i=0, dim=4 (2 block = SIMD 1 iter).
        {
            let u = random_unitary_2x2(&mut rng);
            let psi0: Vec<Complex64> = (0..4).map(|_| rng.complex_signed()).collect();
            let mut psi_simd = psi0.clone();
            simd_kernels::single_mode_i0(&mut psi_simd, &u);
            let mut psi_scalar = psi0;
            run_scalar(&mut psi_scalar, &u, 0);
            for k in 0..4 {
                let diff = (psi_simd[k] - psi_scalar[k]).norm();
                assert!(
                    diff < 1e-14,
                    "i=0, k={}: SIMD={} vs scalar={} (diff={})",
                    k,
                    psi_simd[k],
                    psi_scalar[k],
                    diff,
                );
            }
        }

        // i=1, dim=4 (1 block).
        {
            let u = random_unitary_2x2(&mut rng);
            let psi0: Vec<Complex64> = (0..4).map(|_| rng.complex_signed()).collect();
            let mut psi_simd = psi0.clone();
            simd_kernels::single_mode_i1(&mut psi_simd, &u);
            let mut psi_scalar = psi0;
            run_scalar(&mut psi_scalar, &u, 1);
            for k in 0..4 {
                let diff = (psi_simd[k] - psi_scalar[k]).norm();
                assert!(
                    diff < 1e-14,
                    "i=1, k={}: SIMD={} vs scalar={} (diff={})",
                    k,
                    psi_simd[k],
                    psi_scalar[k],
                    diff,
                );
            }
        }

        // i=2, dim=8 (1 block = SIMD_BLOCK_MAX).
        {
            let u = random_unitary_2x2(&mut rng);
            let psi0: Vec<Complex64> = (0..8).map(|_| rng.complex_signed()).collect();
            let mut psi_simd = psi0.clone();
            simd_kernels::single_mode_i2(&mut psi_simd, &u);
            let mut psi_scalar = psi0;
            run_scalar(&mut psi_scalar, &u, 2);
            for k in 0..8 {
                let diff = (psi_simd[k] - psi_scalar[k]).norm();
                assert!(
                    diff < 1e-14,
                    "i=2, k={}: SIMD={} vs scalar={} (diff={})",
                    k,
                    psi_simd[k],
                    psi_scalar[k],
                    diff,
                );
            }
        }
    }

    // ===== Multi-qubit gate fusion (Phase 6 C3, issue #64) のテスト =====

    /// ランダム 2×2 ユニタリ (Trotter R_i 形 `[c, i·s, i·s, c]`).
    fn random_axis_unitary(rng: &mut Xor64) -> [Complex64; 4] {
        let theta = rng.signed() * std::f64::consts::PI;
        let (s, c) = theta.sin_cos();
        [
            Complex64::new(c, 0.0),
            Complex64::new(0.0, s),
            Complex64::new(0.0, s),
            Complex64::new(c, 0.0),
        ]
    }

    /// `apply_multi_qubit_gate_fused` (per-axis 逐次経路) を
    /// `apply_single_mode_axis_i` を k 回連続で呼ぶのと比較する fuzz テスト.
    ///
    /// 両者は同じ演算順序 (`split_at_mut(mask)` + zip pair update を i_start
    /// から i_start+k-1 の順で適用) なので **bit-identical** を期待する.
    /// `MIN_RAYON_DIM` を跨ぐ n でも rayon 経路の chunk_size 戦略が axis pair を
    /// chunk 内に閉じる前提なので, dim ∈ {small (serial), large (rayon)} の
    /// 両領域で fuzz する.
    #[test]
    fn apply_multi_qubit_gate_fused_matches_per_axis_fuzz_50iter() {
        let mut rng = Xor64::new(0x_64fa_5edf_a571_0a64);
        for iter in 0..50 {
            let n = 4 + (rng.next_u64() % 5) as usize; // n ∈ {4..=8}
            let dim = 1usize << n;
            let k = 1 + (rng.next_u64() % (MAX_FUSED_K.min(n) as u64)) as usize;
            let i_start_max = n - k;
            let i_start = (rng.next_u64() % (i_start_max as u64 + 1)) as usize;

            let u_list: Vec<[Complex64; 4]> =
                (0..k).map(|_| random_axis_unitary(&mut rng)).collect();
            let psi0: Vec<Complex64> = (0..dim).map(|_| rng.complex_signed()).collect();

            // Reference: per-axis を k 回連続 apply.
            let mut psi_ref = psi0.clone();
            for (j, u) in u_list.iter().enumerate() {
                apply_single_mode_axis_i(&mut psi_ref, u, i_start + j, n);
            }

            // Under test: per-axis 逐次経路の fused 版.
            let mut psi_actual = psi0;
            apply_multi_qubit_gate_fused(&mut psi_actual, &u_list, i_start, n);

            let mut diff_sq = 0.0_f64;
            let mut ref_sq = 0.0_f64;
            for kk in 0..dim {
                let d = psi_actual[kk] - psi_ref[kk];
                diff_sq += d.norm_sqr();
                ref_sq += psi_ref[kk].norm_sqr();
            }
            let rel = diff_sq.sqrt() / ref_sq.sqrt().max(1.0);
            assert!(
                rel < 1e-13,
                "iter={}, n={}, k={}, i_start={}: fused vs per-axis rel={} >= 1e-13",
                iter,
                n,
                k,
                i_start,
                rel,
            );
        }
    }

    /// `apply_multi_qubit_gate_fused_serial` と `_rayon` の数値同一性 fuzz.
    /// 両 path は同じ inner kernel `apply_fused_axes_to_chunk` を呼ぶが,
    /// rayon 側は chunk 単位で per-axis pair を分割する. axis 最大 mask が
    /// chunk_size より小さいことを保証する chunk_size 戦略が正しければ
    /// bit-identical になることを期待する. `MIN_RAYON_DIM` を跨ぐ n=17, 18 で確認.
    #[cfg(feature = "rayon")]
    #[test]
    fn apply_multi_qubit_gate_fused_rayon_matches_serial_fuzz() {
        let mut rng = Xor64::new(0x_dead_8eef_cafe_babe);
        for n in [17usize, 18] {
            let dim = 1usize << n;
            for k in [2usize, 3, 4] {
                let i_start = 1usize;
                let u_list: Vec<[Complex64; 4]> =
                    (0..k).map(|_| random_axis_unitary(&mut rng)).collect();
                let psi0: Vec<Complex64> = (0..dim).map(|_| rng.complex_signed()).collect();

                let mut psi_serial = psi0.clone();
                apply_multi_qubit_gate_fused_serial(&mut psi_serial, &u_list, i_start);

                let mut psi_rayon = psi0;
                apply_multi_qubit_gate_fused_rayon(&mut psi_rayon, &u_list, i_start);

                for kk in 0..dim {
                    assert_eq!(
                        psi_serial[kk].re.to_bits(),
                        psi_rayon[kk].re.to_bits(),
                        "n={}, k={}, kk={}: serial vs rayon re bits differ",
                        n,
                        k,
                        kk,
                    );
                    assert_eq!(
                        psi_serial[kk].im.to_bits(),
                        psi_rayon[kk].im.to_bits(),
                        "n={}, k={}, kk={}: serial vs rayon im bits differ",
                        n,
                        k,
                        kk,
                    );
                }
            }
        }
    }

    /// `apply_multi_qubit_gate_fused` の `k = 1` (=単一 qubit) 退化ケースが
    /// `apply_single_mode_axis_i` と一致することを確認.
    ///
    /// k=1 のとき fused 経路の per-axis 逐次 update は `apply_single_mode_axis_i`
    /// 1 回と同じ. ただし rayon 経路の chunk_size は `2^(i+1)` の整数倍に揃え
    /// られるので, `apply_single_mode_axis_i_rayon` の chunk_size 選び方
    /// (`block.next_multiple_of(RAYON_CHUNK_MIN)`) とは細部が異なる可能性がある.
    /// 数値結果は両者で同一だが演算順序が完全一致するとは限らないため
    /// `diff < 1e-14` (machine epsilon の数倍) で比較.
    #[test]
    fn apply_multi_qubit_gate_fused_k1_matches_axis_i() {
        let mut rng = Xor64::new(0x_face_b00c_1234_5678);
        let n = 6;
        let dim = 1usize << n;

        for i in 0..n {
            let u = random_axis_unitary(&mut rng);
            let u_list = [u]; // length 1

            let psi0: Vec<Complex64> = (0..dim).map(|_| rng.complex_signed()).collect();

            let mut psi_axis = psi0.clone();
            apply_single_mode_axis_i(&mut psi_axis, &u, i, n);

            let mut psi_fused = psi0;
            apply_multi_qubit_gate_fused(&mut psi_fused, &u_list, i, n);

            for k in 0..dim {
                let diff = (psi_axis[k] - psi_fused[k]).norm();
                assert!(
                    diff < 1e-14,
                    "i={}, k={}: axis vs fused diff = {}",
                    i,
                    k,
                    diff,
                );
            }
        }
    }
}
