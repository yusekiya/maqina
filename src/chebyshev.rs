//! Chebyshev polynomial expansion propagator (Phase 9+ POC, issue #120).
//!
//! 時間独立 H に対する `exp(-i H dt) ψ` を Chebyshev 多項式の 3 項漸化で計算する.
//! 既存 Lanczos (`src/krylov.rs`) が抱える V matrix (dim×m_max ≈ 96 MB @ N=18,
//! m_max=24) の cache stall (PR #118 で per-step bottleneck と確定) を **アルゴリズム
//! 軸で構造的に回避** することを狙う POC.
//!
//! # 数式
//!
//! Jacobi-Anger 展開:
//!
//! ```text
//! exp(-i z x) = J_0(z) + 2 Σ_{k=1}^∞ (-i)^k J_k(z) T_k(x), x ∈ [-1, 1]
//! ```
//!
//! を `x = \tilde H = (H - E_c·I) / R` (centering + scaling して固有値域を [-1, 1]
//! に閉じ込めた版) に適用すると:
//!
//! ```text
//! exp(-i H dt) ψ = exp(-i E_c dt) · Σ_{k=0}^{K} c_k(z) T_k(\tilde H) ψ
//! ```
//!
//! ここで `z = R · dt`, `c_k(z) = (2 - δ_{k0}) · (-i)^k · J_k(z)`. ベクトル
//! `φ_k = T_k(\tilde H) ψ` は Chebyshev 3 項漸化:
//!
//! ```text
//! φ_0 = ψ
//! φ_1 = \tilde H ψ                       = (H ψ - E_c ψ) / R
//! φ_{k+1} = 2 \tilde H φ_k - φ_{k-1}   = 2 (H φ_k - E_c φ_k) / R - φ_{k-1}
//! ```
//!
//! で計算する. 各 step の `H φ_k` 作用は既存 [`crate::matvec::apply_h_kryanneal`]
//! を再利用 (新 primitive 不要).
//!
//! # メモリと cache 戦略
//!
//! Lanczos と異なり Krylov basis 全列 `V[:, :m]` の保持を要求しない. **3 vector
//! のローテーション** (`φ_{k-1}, φ_k, scratch=H·φ_k → φ_{k+1}`) と accumulator
//! `ψ_acc` だけで動き, N=18 で計 16 MB 程度. これは CCX L3 (32 MB) に収まる
//! ので, Lanczos の V (96 MB > L3) に起因する cache stall が **アルゴリズム
//! レベルで消滅** する見込み. Gram-Schmidt 直交化も 3 項漸化が数学的に直交
//! 保証するため不要 (BLAS-1 ops の n² 二次項が消える).
//!
//! # スペクトル境界推定
//!
//! `E_c` と `R` は Gershgorin の行和上界から **closed form O(N) で算出**
//! ([`gershgorin_bounds`]). matvec 不要なので per-step オーバヘッドは無視可.
//! 上界が loose だと `z = R·dt` が大きくなり K も大きくなるため, Phase B で
//! 必要に応じて Power iteration で R を tighten する option を持たせる
//! (本 POC では Gershgorin one-shot に限定).
//!
//! # 切り捨て次数
//!
//! `|J_K(z)|` は `K > z` で superexponential に減衰する (Bessel asymptotic).
//! [`determine_truncation`] は `|J_K(z)| < tol/2` を満たす最小 K を返す
//! (`k_max_cap` で上限 clamp). Tal-Ezer & Kosloff (1984) の経験則
//! `K ≈ z + O(log(1/tol))` に従う.
//!
//! # Bessel 係数
//!
//! [`bessel_j_array`] は Miller's downward recurrence (forward は大 k で blow-up,
//! downward は self-stabilizing). `J_0(z) + 2 Σ_{j≥1} J_{2j}(z) = 1` の総和規約
//! で正規化. 計算量 O(K_start), K_start ≈ max(2z, K+20). overflow guard 付き.
//!
//! # 適用範囲 (本 POC は Phase A scope のみ)
//!
//! - **時間独立 H** のみ (issue #120 Phase A). 時間依存 (Magnus 統合) は Phase B
//!   (別 issue) で.
//! - Python 公開 API には露出しない (`_rust` には登録せず, in-tree perf binary
//!   `src/bin/perf_chebyshev.rs` から `bench_api` 経由でのみ呼ぶ).
//! - adaptive Richardson 経路への組込・bench_qutip_large.py への追加も Phase B.

#![allow(dead_code)]

use num_complex::Complex64;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

use crate::matvec::apply_h_kryanneal;

/// rayon dispatch を起動する **最小 dim 閾値** (issue #127 PoC).
///
/// `chebyshev_recurrence_fused` (k ≥ 2 hot loop) を `par_chunks_mut` で並列化
/// する経路を起動するかどうかの境界. これ未満では single-thread (SIMD or
/// scalar) fallback で動く. 小 dim で fork/join overhead が並列化 gain を
/// 超える領域を避けるための safety net.
///
/// 初期値は `src/matvec.rs::MIN_RAYON_DIM` (= 1 << 17) と揃える. Chebyshev
/// non-matvec hot loop は matvec より per-element cost が小さい (memory bound)
/// ため, 本来はより低い閾値でも改善が出る可能性があるが, PoC 段階では保守
/// 寄りで始め, Linux 本番 bench (N ∈ {14, 16, 18, 20} sweep) の結果次第で
/// tuning する方針 (issue #127 判定 gate: N=18 で per-step wall 10%+ 改善 +
/// N=12 で 5% 未満劣化 → full merge).
///
/// const なので tuning には release rebuild が必要.
#[cfg(feature = "rayon")]
const MIN_RAYON_DIM_CHEB: usize = 1 << 17;

/// chebyshev rayon chunk の **最小** 要素数. closure / scheduling overhead を
/// 償却するため 64 要素以上を保証する. matvec の `RAYON_CHUNK_MIN` と同値.
#[cfg(feature = "rayon")]
const RAYON_CHUNK_MIN_CHEB: usize = 1 << 6;

/// chebyshev rayon chunk の **最大** 要素数. L2 fit を狙う上限. matvec の
/// `RAYON_CHUNK_MAX` と同値.
#[cfg(feature = "rayon")]
const RAYON_CHUNK_MAX_CHEB: usize = 1 << 14;

// ============================================================================
// SIMD + fusion kernel (issue #126 PoC)
// ============================================================================
//
// k ≥ 2 の 3 項漸化 inner loop で `scratch` / `phi_curr` / `phi_prev` /
// `psi_acc` の 4 vector を **1 dim-walk** で処理するための fused kernel.
// 旧実装は 2 scalar loop (recurrence scaling + accumulate) で計 3 dim-walk
// (matvec walk 1 + scalar walk 2) を回していた. 本 kernel は scalar walk 2
// を fuse して 2 dim-walk に削減 + `wide::f64x4` SIMD で並列化する.
//
// dispatch: [`chebyshev_recurrence_fused`] が `feature = "simd"` ON では SIMD
// 経路, OFF では [`chebyshev_recurrence_fused_scalar`] にフォールバック.
// `cfm4_step_chebyshev_*` 経由でも自動で乗る (同じ `chebyshev_propagate` を
// 呼ぶため).

/// Chebyshev 3 項漸化 inner-loop の fused 更新を **scalar** で実行する.
///
/// 1 ループ内で:
/// * `scratch[j] = 2 · (scratch[j] - e_c · phi_curr[j]) · inv_r - phi_prev[j]`
/// * `psi_acc[j] += c_k · scratch[j]`
///
/// を 1 pass で計算する. `feature = "simd"` OFF (`--no-default-features`
/// ビルド) で呼ばれる基準実装. SIMD 経路との数値同一性検証にも使う.
#[inline]
fn chebyshev_recurrence_fused_scalar(
    scratch: &mut [Complex64],
    phi_curr: &[Complex64],
    phi_prev: &[Complex64],
    psi_acc: &mut [Complex64],
    e_c: f64,
    inv_r: f64,
    c_k: Complex64,
) {
    let n = scratch.len();
    debug_assert_eq!(phi_curr.len(), n);
    debug_assert_eq!(phi_prev.len(), n);
    debug_assert_eq!(psi_acc.len(), n);
    let e_c_c = Complex64::new(e_c, 0.0);
    let inv_r_c = Complex64::new(inv_r, 0.0);
    let two_c = Complex64::new(2.0, 0.0);
    for j in 0..n {
        let tilde = (scratch[j] - e_c_c * phi_curr[j]) * inv_r_c;
        let new_s = two_c * tilde - phi_prev[j];
        scratch[j] = new_s;
        psi_acc[j] += c_k * new_s;
    }
}

/// Chebyshev 3 項漸化 inner-loop の fused 更新を `wide::f64x4` で実行する
/// SIMD 特化版 (issue #126 PoC, Phase 9+).
///
/// 1 SIMD iter = 1 × f64x4 = 4 f64 = 2 Complex64. 入力 dim は `2^N` (N ≥ 1)
/// で必ず 2 の倍数になるため scalar tail は発生しない (debug_assert で確認).
///
/// # 命令計画 (AVX2 + FMA target)
///
/// 各 lane の operation:
/// 1. `tilde = (scratch - e_c · phi_curr) · inv_r`
///    (`e_c`, `inv_r` が real scalar なので f64x4 splat + 普通の `*` で OK.
///    各 Complex の re / im 両方が同係数でスケールされる).
/// 2. `new_s = 2 · tilde - phi_prev`
/// 3. `psi_acc += c_k · new_s`
///    (`c_k` は complex scalar なので broadcast + swap pattern:
///    `c_k · x_pair = c_re_v · x_pair + c_im_signed_v · swap_reim(x_pair)`,
///    `matvec.rs::simd_kernels::single_mode_iN` と同じ. 詳細は同モジュール
///    の docstring 参照).
///
/// 4 個の input slice (scratch RW / phi_curr R / phi_prev R / psi_acc RW) を
/// `chunks_exact_mut(4 f64)` / `chunks_exact(4 f64)` のロックステップ走査で
/// 取り出す. `scratch` と `psi_acc` は呼出側で disjoint な `Vec<Complex64>`
/// から来ているので, 2 本の `&mut [f64]` が同時に生きていても aliasing なし.
///
/// # 数値同一性
///
/// SIMD / scalar 経路の演算順序は理論的に各 `scratch[j]` / `psi_acc[j]` への
/// 単一値生成と等価. FMA 折りたたみ ON/OFF や lane の演算順序差で ulp 差が
/// 出うるため `rel < 1e-13` で比較する (テスト
/// `chebyshev_recurrence_fused_simd_matches_scalar`).
#[cfg(feature = "simd")]
mod simd_kernels {
    use num_complex::Complex64;
    use wide::f64x4;

    /// 4 連続 f64 を 256-bit unaligned load で `f64x4` に取り込む.
    /// `matvec.rs::simd_kernels::load_f64x4_unaligned` と同じパターン. localize
    /// duplication で chebyshev module 内に閉じる (matvec の private mod 経路を
    /// 跨いだ visibility 変更を避ける).
    ///
    /// # Safety
    /// `ptr` は少なくとも 32 bytes (4 f64) が連続して読み出せる領域を指していること.
    #[inline(always)]
    unsafe fn load_f64x4_unaligned(ptr: *const f64) -> f64x4 {
        // SAFETY: caller が 4 f64 readable を保証. wide::f64x4 は repr(C, align(32)),
        // size 32, 内容は f64×4 と同じビットパターン.
        unsafe { std::ptr::read_unaligned(ptr as *const f64x4) }
    }

    /// `f64x4` を 4 連続 f64 へ 256-bit unaligned store する.
    ///
    /// # Safety
    /// `ptr` は少なくとも 32 bytes が書き込み可能であること.
    #[inline(always)]
    unsafe fn store_f64x4_unaligned(ptr: *mut f64, val: f64x4) {
        // SAFETY: caller が 4 f64 writable を保証.
        unsafe { std::ptr::write_unaligned(ptr as *mut f64x4, val) }
    }

    /// `&[Complex64]` を `&[f64]` (長さ 2 倍) として view する.
    /// `Complex64 = num_complex::Complex<f64>` は `#[repr(C)]` で `(re, im) =
    /// (f64, f64)` レイアウト, align 8. `[Complex64]` ↔ `[f64; 2N]` の bit-
    /// equivalent な解釈は sound (matvec.rs と同じ前提).
    #[inline]
    fn as_f64_slice(v: &[Complex64]) -> &[f64] {
        // SAFETY: 上記コメント参照.
        unsafe { std::slice::from_raw_parts(v.as_ptr() as *const f64, v.len() * 2) }
    }

    /// `&mut [Complex64]` を `&mut [f64]` として view する.
    #[inline]
    fn as_f64_slice_mut(y: &mut [Complex64]) -> &mut [f64] {
        // SAFETY: 上記コメント参照.
        unsafe { std::slice::from_raw_parts_mut(y.as_mut_ptr() as *mut f64, y.len() * 2) }
    }

    /// `[x0.re, x0.im, x1.re, x1.im] -> [x0.im, x0.re, x1.im, x1.re]`.
    /// LLVM は AVX target で `vpermilpd` 1 命令に折り畳む.
    #[inline(always)]
    fn swap_reim(x: f64x4) -> f64x4 {
        let a = x.to_array();
        f64x4::new([a[1], a[0], a[3], a[2]])
    }

    /// fused Chebyshev recurrence + accumulation kernel. 詳細は module 外の
    /// [`super::chebyshev_recurrence_fused`] docstring 参照.
    #[inline]
    pub(super) fn chebyshev_recurrence_fused(
        scratch: &mut [Complex64],
        phi_curr: &[Complex64],
        phi_prev: &[Complex64],
        psi_acc: &mut [Complex64],
        e_c: f64,
        inv_r: f64,
        c_k: Complex64,
    ) {
        let n = scratch.len();
        debug_assert_eq!(phi_curr.len(), n);
        debug_assert_eq!(phi_prev.len(), n);
        debug_assert_eq!(psi_acc.len(), n);
        debug_assert!(n >= 2, "len must be >= 2 (SIMD fused kernel)");
        debug_assert!(
            n.is_multiple_of(2),
            "len must be a multiple of 2 (SIMD fused kernel)"
        );

        let e_c_v = f64x4::splat(e_c);
        let inv_r_v = f64x4::splat(inv_r);
        let two_v = f64x4::splat(2.0);
        // c_k · x_pair = c_re_v · x_pair + c_im_signed_v · swap_reim(x_pair).
        let c_re_v = f64x4::splat(c_k.re);
        let c_im_signed_v = f64x4::new([-c_k.im, c_k.im, -c_k.im, c_k.im]);

        let scratch_f64 = as_f64_slice_mut(scratch);
        let phi_curr_f64 = as_f64_slice(phi_curr);
        let phi_prev_f64 = as_f64_slice(phi_prev);
        let psi_acc_f64 = as_f64_slice_mut(psi_acc);

        // 1 chunk = 4 f64 = 2 Complex64 = 1 × f64x4.
        scratch_f64
            .chunks_exact_mut(4)
            .zip(psi_acc_f64.chunks_exact_mut(4))
            .zip(phi_curr_f64.chunks_exact(4))
            .zip(phi_prev_f64.chunks_exact(4))
            .for_each(|(((s_ch, pa_ch), pc_ch), pp_ch)| {
                // SAFETY: chunks_exact{,_mut}(4) は各 chunk 長 4 f64 = 32 bytes
                // を保証. load_f64x4_unaligned / store_f64x4_unaligned の
                // precondition (4 f64 readable / writable) を満たす. scratch と
                // psi_acc は disjoint な &mut [Complex64] 由来なので, それぞれ
                // 派生した &mut [f64] view も disjoint で aliasing なし.
                unsafe {
                    let s = load_f64x4_unaligned(s_ch.as_ptr());
                    let pc = load_f64x4_unaligned(pc_ch.as_ptr());
                    let pp = load_f64x4_unaligned(pp_ch.as_ptr());
                    let pa = load_f64x4_unaligned(pa_ch.as_ptr());

                    // tilde = (s - e_c · pc) · inv_r.
                    let tilde = (s - e_c_v * pc) * inv_r_v;
                    // new_s = 2 · tilde - pp. AVX2+FMA target では LLVM が
                    // mul + sub を vfmsub に折り畳む.
                    let new_s = two_v * tilde - pp;
                    // psi_acc += c_k · new_s.
                    let new_s_swap = swap_reim(new_s);
                    let new_pa = pa + c_re_v * new_s + c_im_signed_v * new_s_swap;

                    store_f64x4_unaligned(s_ch.as_mut_ptr(), new_s);
                    store_f64x4_unaligned(pa_ch.as_mut_ptr(), new_pa);
                }
            });
    }
}

/// `chebyshev_recurrence_fused` の **rayon 並列実装** (issue #127 PoC).
///
/// `scratch` / `psi_acc` を `par_chunks_mut(chunk_size)` で並列分割し, 各
/// chunk 内で既存 SIMD (or scalar) fused kernel を呼ぶ 2 段構造. matvec.rs の
/// `apply_h_kryanneal_rayon` と同じ chunking 戦略 (`(dim / (nth * 4)).clamp(
/// MIN, MAX)`) を踏襲する.
///
/// # chunk_size の align
///
/// SIMD kernel は `n >= 2 && n % 2 == 0` (2 Complex64 = 4 f64 = 1 f64x4 単位)
/// を要求する. chunk_size を 2 の倍数に揃えれば
///
///   * 通常 chunk: `chunk_size` (= 偶数) → OK
///   * 末尾 chunk: `dim - (n_chunks - 1) · chunk_size`. `dim = 2^n` は常に
///     偶数, `chunk_size` も偶数なので末尾 chunk_len も偶数.
///
/// となり SIMD 前提が満たされる. `RAYON_CHUNK_MIN_CHEB` (=64) と
/// `RAYON_CHUNK_MAX_CHEB` (=16384) は共に偶数なので, 偶数化丸めは min/max
/// invariant を破らない.
///
/// # 4 slice の borrow
///
/// `scratch` (RW) と `psi_acc` (RW) は呼出側で disjoint な `Vec<Complex64>`
/// 由来. `par_chunks_mut` を 2 本独立に取って `.zip()` し, `enumerate()` で
/// 取った `base` から `phi_curr` / `phi_prev` (R) を共有 sub-slice として
/// 切り出す. mut / immut borrow が rayon の closure 内で disjoint に閉じる
/// ため aliasing なし.
#[cfg(feature = "rayon")]
fn chebyshev_recurrence_fused_rayon(
    scratch: &mut [Complex64],
    phi_curr: &[Complex64],
    phi_prev: &[Complex64],
    psi_acc: &mut [Complex64],
    e_c: f64,
    inv_r: f64,
    c_k: Complex64,
) {
    let dim = scratch.len();
    debug_assert_eq!(phi_curr.len(), dim);
    debug_assert_eq!(phi_prev.len(), dim);
    debug_assert_eq!(psi_acc.len(), dim);

    let nth = rayon::current_num_threads().max(1);
    let mut chunk_size = (dim / (nth * 4)).clamp(RAYON_CHUNK_MIN_CHEB, RAYON_CHUNK_MAX_CHEB);
    // SIMD kernel の偶数長前提を満たすため 2 倍数に揃える.
    chunk_size -= chunk_size % 2;

    scratch
        .par_chunks_mut(chunk_size)
        .zip(psi_acc.par_chunks_mut(chunk_size))
        .enumerate()
        .for_each(|(idx, (s_chunk, pa_chunk))| {
            let base = idx * chunk_size;
            let chunk_len = s_chunk.len();
            let pc_chunk = &phi_curr[base..base + chunk_len];
            let pp_chunk = &phi_prev[base..base + chunk_len];

            #[cfg(feature = "simd")]
            {
                if chunk_len >= 2 && chunk_len.is_multiple_of(2) {
                    simd_kernels::chebyshev_recurrence_fused(
                        s_chunk, pc_chunk, pp_chunk, pa_chunk, e_c, inv_r, c_k,
                    );
                    return;
                }
            }
            // SIMD OFF, または退化ケース (実用上発生しないが防御的) の fallback.
            chebyshev_recurrence_fused_scalar(
                s_chunk, pc_chunk, pp_chunk, pa_chunk, e_c, inv_r, c_k,
            );
        });
}

/// Chebyshev 3 項漸化 inner-loop の fused 更新を dispatch する wrapper.
///
/// 3 段 dispatch (issue #127 PoC で rayon 段を追加):
/// 1. `feature = "rayon"` ON かつ `dim >= MIN_RAYON_DIM_CHEB` →
///    [`chebyshev_recurrence_fused_rayon`] (内側で SIMD or scalar fused kernel).
/// 2. `feature = "simd"` ON かつ `n >= 2 && n % 2 == 0` →
///    [`simd_kernels::chebyshev_recurrence_fused`] (single-thread SIMD).
/// 3. それ以外 (`--no-default-features`, 退化ケース) →
///    [`chebyshev_recurrence_fused_scalar`].
#[inline]
fn chebyshev_recurrence_fused(
    scratch: &mut [Complex64],
    phi_curr: &[Complex64],
    phi_prev: &[Complex64],
    psi_acc: &mut [Complex64],
    e_c: f64,
    inv_r: f64,
    c_k: Complex64,
) {
    #[cfg(feature = "rayon")]
    {
        if scratch.len() >= MIN_RAYON_DIM_CHEB {
            chebyshev_recurrence_fused_rayon(scratch, phi_curr, phi_prev, psi_acc, e_c, inv_r, c_k);
            return;
        }
    }
    #[cfg(feature = "simd")]
    {
        let n = scratch.len();
        if n >= 2 && n.is_multiple_of(2) {
            simd_kernels::chebyshev_recurrence_fused(
                scratch, phi_curr, phi_prev, psi_acc, e_c, inv_r, c_k,
            );
            return;
        }
    }
    chebyshev_recurrence_fused_scalar(scratch, phi_curr, phi_prev, psi_acc, e_c, inv_r, c_k);
}

/// Gershgorin の行和上界 / 下界による `H = a_t · H_drv + b_t · diag(h_p_diag)`
/// のスペクトル境界推定 `(E_min, E_max)` を返す (E_min ≤ λ_min(H), λ_max(H) ≤ E_max).
///
/// 横磁場イジング Hamiltonian の構造:
///   * 対角: `H_kk = b_t · h_p_diag[k]`
///   * 非対角行和: `Σ_{j≠k} |H_jk| = |a_t| · Σ_i |h_x_i|` (TFIM bit-flip 構造により k に独立)
///
/// したがって Gershgorin disk の中心 = 対角値, 半径 = 非対角行和 (一定). 各
/// 行の disk は `[H_kk - off, H_kk + off]`. その union で全固有値が含まれる:
///
/// ```text
/// E_max = max_k (b_t · h_p_diag[k]) + |a_t| · Σ |h_x_i|
/// E_min = min_k (b_t · h_p_diag[k]) - |a_t| · Σ |h_x_i|
/// ```
///
/// `b_t` が負のときに `b_t · max` / `b_t · min` の sign flip を扱うため,
/// `b_t · h_p_diag[k]` を per-element に計算してから min/max を取る (`b_t` 自体
/// は schedule で `[0, 1]` に収まることが多いが, ロバスト性のため).
///
/// `h_p_diag` が空 (dim = 0; n = 0 は呼ばれない前提だが防御的に) のときは
/// `(0.0, 0.0)` を返す.
pub fn gershgorin_bounds(h_x: &[f64], h_p_diag: &[f64], a_t: f64, b_t: f64) -> (f64, f64) {
    let off_sum: f64 = h_x.iter().map(|x| x.abs()).sum();
    let off = a_t.abs() * off_sum;

    let mut diag_min = f64::INFINITY;
    let mut diag_max = f64::NEG_INFINITY;
    for &d in h_p_diag.iter() {
        let v = b_t * d;
        if v < diag_min {
            diag_min = v;
        }
        if v > diag_max {
            diag_max = v;
        }
    }
    if !diag_min.is_finite() {
        // h_p_diag 空 (dim=0 など): 対角寄与なし.
        return (-off, off);
    }
    (diag_min - off, diag_max + off)
}

/// Bessel 関数 `J_0(z), J_1(z), ..., J_{k_max}(z)` を一括計算して長さ `k_max + 1`
/// の `Vec<f64>` で返す (Miller's downward recurrence).
///
/// # アルゴリズム
///
/// forward recurrence `J_{k+1} = (2k/z) J_k - J_{k-1}` は大 `k` で数値的に
/// 不安定 (`J_k(z)` が指数的に小さくなる一方で誤差は forward に増殖する).
/// 一方 downward recurrence は self-stabilizing で, 出発値を arbitrary に
/// 取っても収束する. 標準的な手順:
///
/// 1. `N_start >> k_max` から `b_{N_start+1} = 0, b_{N_start} = 1` で出発
/// 2. 漸化 `b_{k-1} = (2k/z) · b_k - b_{k+1}` で `k = N_start, ..., 1` を下降
/// 3. 総和規約 `J_0(z) + 2 Σ_{j≥1} J_{2j}(z) = 1` (`exp(0·cosθ) = 1` の
///    Jacobi-Anger 経由) で正規化倍率を計算
/// 4. 各 `b_k` を倍率で正規化して返す
///
/// 参考: Abramowitz & Stegun §9.12, NR §6.5.
///
/// # 数値安定性
///
/// 下降中に `b_{k-1}` が overflow しそうになったら全 `b` 配列と総和を 1e-100
/// 倍に rescale する (正規化倍率に吸収される). `z = 0` は special-case
/// (`J_0(0) = 1, J_k(0) = 0 for k ≥ 1`).
///
/// # K_start の選び方
///
/// `max(k_max + 30, 2|z| + 50)` を採用. Miller 下降漸化は b[N_start]=1 の
/// "Y_n contamination" が k 降下に伴い exponentially 抑制される性質
/// (ratio J_{N_start+1}/J_n) を持ち, 通常 N_start - k_max が 30+ あれば double
/// precision (1e-15) に到達する. `k_max + 30` の floor で k_max < |z| の小 z
/// case も保護し, `2|z| + 50` で k_max が z より小さい場合の z 主導 case も
/// 押さえる. 30 + 50 は安全率込み (実証: scipy.special.jv との rel < 1e-13
/// 一致を `bessel_jv_matches_scipy_reference` で確認済み).
pub fn bessel_j_array(z: f64, k_max: usize) -> Vec<f64> {
    if z == 0.0 {
        let mut out = vec![0.0_f64; k_max + 1];
        out[0] = 1.0;
        return out;
    }

    let abs_z = z.abs();
    let abs_z_u = abs_z.ceil() as usize;
    // Miller's start index. 充分大きい所から下降する. 上のコメント参照.
    let n_start = (k_max + 30).max(2 * abs_z_u + 50);

    // b[n_start+1] = 0, b[n_start] = 1 で出発. `+2` は senitnel `b[n_start+1]` の分.
    let mut b = vec![0.0_f64; n_start + 2];
    b[n_start] = 1.0;

    // 下降 recurrence: b[k-1] = (2k / z) · b[k] - b[k+1].
    // overflow guard: |b| > 1e100 になったら全要素を 1e-100 倍 (rescale).
    // 正規化倍率に吸収されるので最終 J_k 値には影響しない.
    for k in (1..=n_start).rev() {
        b[k - 1] = (2.0 * (k as f64) / z) * b[k] - b[k + 1];
        if b[k - 1].abs() > 1e100 {
            let scale = 1e-100;
            for v in b.iter_mut() {
                *v *= scale;
            }
        }
    }

    // 正規化: J_0(z) + 2 Σ_{j≥1} J_{2j}(z) = 1.
    let mut sum_norm = b[0];
    let mut j = 1_usize;
    loop {
        let idx = 2 * j;
        if idx > n_start {
            break;
        }
        sum_norm += 2.0 * b[idx];
        j += 1;
    }

    let inv = 1.0 / sum_norm;
    let mut out = vec![0.0_f64; k_max + 1];
    for k in 0..=k_max {
        out[k] = b[k] * inv;
    }
    out
}

/// `|J_K(z)| < tol / 2` を満たす最小 `K ≥ 1` を返す (`k_max_cap` で上限 clamp).
///
/// `tol / 2` は Chebyshev 展開の 1 項当たりの coefficient `c_k = 2 (-i)^k J_k(z)`
/// が `|c_k| = 2 |J_k|` を持つことから, "次の項 K の追加で生じる相対誤差を tol
/// 未満に抑える" 規約.
///
/// # 戻り値
///
/// 通常は K ≪ k_max_cap で smallest such K を返す. 充分大 z で K_cap に達した
/// 場合は K_cap をそのまま返す (上位呼出側で K_used を見れば判定可).
///
/// # `z` が小さい場合
///
/// `z ≪ 1` では `|J_1(z)| ≈ z/2` が即 tol/2 を切り, K = 1 が返る (1 stage で
/// 充分の意).
pub fn determine_truncation(z: f64, tol: f64, k_max_cap: usize) -> usize {
    // K_search: |J_k| の減衰観察に十分な上限. 2z + 50 程度で OK.
    let k_search = (((2.0 * z.abs()) as usize) + 50).min(k_max_cap);
    if k_search < 1 {
        return 1;
    }
    let jvals = bessel_j_array(z, k_search);
    let thresh = 0.5 * tol;
    for (k, &jv) in jvals.iter().enumerate().skip(1) {
        if jv.abs() < thresh {
            return k;
        }
    }
    k_search
}

/// 時間独立 `H = a_t · H_drv + b_t · diag(h_p_diag)` に対し `ψ_new = exp(-i H dt) · ψ`
/// を Chebyshev 多項式の 3 項漸化で計算する.
///
/// # 引数
///
/// * `h_x` (length `n`): サイト依存横磁場振幅 (`H_drv = -Σ_i h_x_i X_i` の係数).
/// * `h_p_diag` (length `2^n`): Z 基底での `H_problem` 対角ベクトル.
/// * `a_t`, `b_t`: 時刻 `t` での schedule 係数 (本 POC は時間独立なので 1 step
///   の間 frozen).
/// * `psi` (length `2^n`): 入力状態.
/// * `dt`: 時刻刻み幅 (real).
/// * `tol`: Chebyshev 切り捨て次数 K の決定閾値 (`|J_K(z)| < tol/2`).
/// * `n`: サイト数. `dim = 2^n` を呼出側と一意に決める.
///
/// # 戻り値
///
/// `(psi_new, K_used, err_estimate)`:
/// * `psi_new` (length `2^n`): `exp(-i H dt) · ψ` の近似.
/// * `K_used`: 実際に使った Chebyshev 多項式の最大次数. `K_used = 0` は
///   `R < 1e-15` の zero Hamiltonian fast-path.
/// * `err_estimate`: 末尾 truncation residual `2 · |J_{K_used+1}(z)| · ‖ψ‖`.
///   1-term 近似 (Tal-Ezer & Kosloff の経験則による上界推定).
///
/// # アルゴリズム概要
///
/// 1. Gershgorin で `(E_min, E_max)` を取り, `E_c = (E_max+E_min)/2`, `R = (E_max-E_min)/2`.
/// 2. `R < 1e-15` (zero Hamiltonian) は special-case: `exp(-i E_c dt) · ψ` 即返し.
/// 3. `z = R · dt` で `K_used = determine_truncation(z, tol, K_cap)` を決定.
/// 4. `bessel_j_array(z, K_used + 1)` で `J_0..J_{K_used+1}(z)` を一括計算
///    (`J_{K_used+1}` は err estimate 用).
/// 5. 3 項漸化:
///    * `φ_0 = ψ`, `c_0 = J_0(z)`, `ψ_acc = c_0 · φ_0`
///    * `φ_1 = (H ψ - E_c ψ) / R`, `c_1 = -2i J_1(z)`, `ψ_acc += c_1 φ_1`
///    * `k ≥ 2`: `φ_{k} = 2 (H φ_{k-1} - E_c φ_{k-1}) / R - φ_{k-2}`,
///      `c_k = 2 (-i)^k J_k(z)`, `ψ_acc += c_k φ_k`
/// 6. global phase: `ψ_acc *= exp(-i E_c dt)`.
///
/// # メモリ
///
/// 3 個の作業ベクトル `(phi_prev, phi_curr, scratch)` + accumulator `psi_acc`
/// で動く. `scratch` は matvec 出力先と次 `φ_{k+1}` を兼ねる. dim × 4 = 4·2^n
/// Complex64. N=18 で計 16 MB (L3 = 32 MB / CCX に収まる).
///
/// # Panics
///
/// * `psi.len() != 1 << n`
/// * `h_x.len() != n`
/// * `h_p_diag.len() != 1 << n`
#[allow(clippy::too_many_arguments)]
pub fn chebyshev_propagate(
    h_x: &[f64],
    h_p_diag: &[f64],
    a_t: f64,
    b_t: f64,
    psi: &[Complex64],
    dt: f64,
    tol: f64,
    n: usize,
) -> (Vec<Complex64>, usize, f64) {
    let dim = 1usize << n;
    assert_eq!(psi.len(), dim, "psi must have length 2^n");
    assert_eq!(h_x.len(), n, "h_x must have length n");
    assert_eq!(h_p_diag.len(), dim, "h_p_diag must have length 2^n");

    // 1. Gershgorin bounds.
    let (e_min, e_max) = gershgorin_bounds(h_x, h_p_diag, a_t, b_t);
    let e_c = 0.5 * (e_max + e_min);
    let r = 0.5 * (e_max - e_min);

    // 2. Zero Hamiltonian fast-path: R = 0 で T_k(\tilde H) が ill-defined.
    //    exp(-i H dt) ψ = exp(-i E_c dt) ψ をそのまま返す (E_c も 0 ならただの id).
    if r < 1e-15 {
        let phase = Complex64::new(0.0, -dt * e_c).exp();
        let psi_new: Vec<Complex64> = psi.iter().map(|&p| phase * p).collect();
        return (psi_new, 0, 0.0);
    }

    // 3. K_used decision.
    let z = r * dt;
    // K_cap = 5000 は実用上 over-kill (n=18, dt=1.0 で z ~ O(N), K ~ 30-50).
    // pathological dt や large ‖H‖ でも 4-5 桁の精度が出る上限として確保.
    let k_max_cap: usize = 5000;
    let k_used = determine_truncation(z, tol, k_max_cap);

    // 4. Bessel 係数を K+1 までまとめて計算 (J_{K+1} は err estimate 用).
    let jvals = bessel_j_array(z, k_used + 1);

    // 5. 3 項漸化. 3 vector rotation:
    //    phi_prev (= φ_{k-1}), phi_curr (= φ_k), scratch (= H · φ_k → φ_{k+1}).
    //    psi_acc accumulates Σ c_k φ_k.
    let mut phi_prev: Vec<Complex64> = psi.to_vec();
    let mut phi_curr: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); dim];
    let mut scratch: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); dim];

    // c_0 = J_0(z) (factor (2-δ_{k0}) = 1 for k=0, (-i)^0 = 1).
    let c0 = Complex64::new(jvals[0], 0.0);
    let mut psi_acc: Vec<Complex64> = phi_prev.iter().map(|&p| c0 * p).collect();

    let inv_r = 1.0 / r;

    if k_used >= 1 {
        // φ_1 = \tilde H ψ = (H ψ - E_c ψ) / R.
        apply_h_kryanneal(&phi_prev, &mut scratch, h_x, h_p_diag, a_t, b_t, n);
        for j in 0..dim {
            phi_curr[j] =
                (scratch[j] - Complex64::new(e_c, 0.0) * phi_prev[j]) * Complex64::new(inv_r, 0.0);
        }
        // c_1 = 2 · (-i)^1 · J_1(z) = -2i J_1(z).
        let c1 = Complex64::new(0.0, -2.0 * jvals[1]);
        for j in 0..dim {
            psi_acc[j] += c1 * phi_curr[j];
        }
    }

    // 6. k ≥ 2 の漸化. issue #126 PoC で walk 2 (recurrence scaling) と walk 3
    // (accumulate) を `chebyshev_recurrence_fused` で 1 walk + SIMD に fuse.
    // walk 1 (matvec) は `apply_h_kryanneal` のまま. 計 3 walk → 2 walk + SIMD.
    // k_ord は jvals[k_ord] と `k_ord % 4` の両方で必要 (前者は coefficient,
    // 後者は `(-i)^k` の 4 周期 dispatch) なので iterator chain への書き換えは
    // 可読性を下げる. needless_range_loop を allow.
    #[allow(clippy::needless_range_loop)]
    for k_ord in 2..=k_used {
        // walk 1: scratch := H · phi_curr.
        apply_h_kryanneal(&phi_curr, &mut scratch, h_x, h_p_diag, a_t, b_t, n);
        // c_{k_ord} = 2 · (-i)^{k_ord} · J_{k_ord}(z). (-i)^k は 4 周期で循環.
        let pow_minus_i = match k_ord % 4 {
            0 => Complex64::new(1.0, 0.0),
            1 => Complex64::new(0.0, -1.0),
            2 => Complex64::new(-1.0, 0.0),
            3 => Complex64::new(0.0, 1.0),
            _ => unreachable!(),
        };
        let ck = Complex64::new(2.0 * jvals[k_ord], 0.0) * pow_minus_i;
        // walk 2 (fused, SIMD): scratch <- 2·(scratch - e_c·phi_curr)·inv_r - phi_prev;
        //                       psi_acc += ck · scratch.
        chebyshev_recurrence_fused(
            &mut scratch,
            &phi_curr,
            &phi_prev,
            &mut psi_acc,
            e_c,
            inv_r,
            ck,
        );
        // rotation: phi_prev <- phi_curr, phi_curr <- scratch (= φ_{k_ord}).
        // 旧 phi_prev は scratch 用の使い回しバッファになる.
        std::mem::swap(&mut phi_prev, &mut phi_curr);
        std::mem::swap(&mut phi_curr, &mut scratch);
    }

    // 7. Global phase exp(-i E_c dt).
    let global_phase = Complex64::new(0.0, -dt * e_c).exp();
    for v in psi_acc.iter_mut() {
        *v *= global_phase;
    }

    // 8. Error estimate (1-term truncation residual).
    //    err ≈ |c_{K+1}| · ‖ψ‖ = 2 |J_{K+1}(z)| · ‖ψ‖.
    let psi_norm: f64 = psi.iter().map(|p| p.norm_sqr()).sum::<f64>().sqrt();
    let err_estimate = 2.0 * jvals[k_used + 1].abs() * psi_norm;

    (psi_acc, k_used, err_estimate)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, SymmetricEigen};

    /// 軽量決定論的 PRNG (xorshift64). matvec / krylov / tridiag テストと同実装.
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

    /// scipy.special.jv による hardcoded reference 値と rel < 1e-13 で一致.
    ///
    /// 参照値は `scipy.special.jv(k, z)` (scipy 1.13.x, double precision) で
    /// 計算済み. uv run python の出力をそのまま埋め込む.
    #[test]
    fn bessel_jv_matches_scipy_reference() {
        // (z, k, expected) — expected は scipy.special.jv(k, z) の f64 値.
        // 値は Python `repr()` の shortest-round-trip 形式 (clippy::excessive_precision
        // 回避).
        let cases: &[(f64, usize, f64)] = &[
            (1.0, 0, 0.7651976865579666),
            (1.0, 1, 0.44005058574493355),
            (1.0, 2, 0.1149034849319005),
            (10.0, 5, -0.2340615281867936),
            (10.0, 10, 0.2074861066333589),
            (10.0, 20, 1.1513369247813391e-5),
            (0.5, 0, 0.938469807240813),
            (5.0, 3, 0.364831230613667),
        ];

        for &(z, k, expected) in cases {
            let jvals = bessel_j_array(z, (k + 5).max(30));
            let actual = jvals[k];
            let rel = (actual - expected).abs() / expected.abs().max(1e-300);
            assert!(
                rel < 1e-13,
                "J_{k}({z}): actual = {actual}, expected = {expected}, rel = {rel:e}",
            );
        }
    }

    /// `z = 0` の特別 case: `J_0(0) = 1`, `J_k(0) = 0 for k ≥ 1`.
    #[test]
    fn bessel_jv_at_zero() {
        let jvals = bessel_j_array(0.0, 5);
        assert_eq!(jvals[0], 1.0);
        for (k, &jv) in jvals.iter().enumerate().skip(1) {
            assert_eq!(jv, 0.0, "J_{k}(0) should be exactly 0");
        }
    }

    /// 切り捨て次数 K(z, tol) のスケール感: K_search が k_max_cap で頭打ちしない
    /// 範囲で `K ≈ O(z)` + `log(1/tol)` 程度に収まる.
    #[test]
    fn truncation_scales_with_z_and_tol() {
        // z = 10, tol = 1e-10: K_used ≈ 25-40 程度を期待 (Tal-Ezer & Kosloff 経験則).
        let k_1 = determine_truncation(10.0, 1e-10, 1000);
        assert!(
            (20..=50).contains(&k_1),
            "K(z=10, tol=1e-10) = {k_1} should be ~20-50",
        );

        // tol を緩めると K は減少する.
        let k_2 = determine_truncation(10.0, 1e-4, 1000);
        assert!(
            k_2 < k_1,
            "K(z=10, tol=1e-4) = {k_2} should be < K(z=10, tol=1e-10) = {k_1}",
        );

        // z を大きくすると K は増加する.
        let k_3 = determine_truncation(50.0, 1e-10, 1000);
        assert!(
            k_3 > k_1,
            "K(z=50, tol=1e-10) = {k_3} should be > K(z=10, tol=1e-10) = {k_1}",
        );
    }

    /// 小 dim (n=3) で nalgebra 固有分解 `exp(-i dt H) · ψ` と一致.
    #[test]
    fn chebyshev_matches_dense_n3() {
        let n = 3_usize;
        let dim = 1usize << n;
        let mut rng = Xor64::new(2024);
        let h_x: Vec<f64> = (0..n).map(|_| rng.signed()).collect();
        let h_p_diag: Vec<f64> = (0..dim).map(|_| rng.signed()).collect();
        let psi = random_complex_vec(dim, 19);
        let dt = 0.5_f64;
        let a_t = 0.6_f64;
        let b_t = 0.4_f64;
        let tol = 1e-13;

        let h_dense = build_dense_h_real(n, &h_x, &h_p_diag, a_t, b_t);
        let expected = reference_propagate_real_h(&h_dense, &psi, dt);

        let (actual, k_used, err_estimate) =
            chebyshev_propagate(&h_x, &h_p_diag, a_t, b_t, &psi, dt, tol, n);
        assert!(k_used >= 1, "K_used = {k_used} should be >= 1");
        let rel = relative_error(&actual, &expected);
        assert!(
            rel < 1e-12,
            "rel = {rel:e} (n=3, dt={dt}, tol={tol}, K_used={k_used}, err_est={err_estimate:e})",
        );
    }

    /// 小 dim (n=4) で長め dt + 大きい schedule 係数.
    #[test]
    fn chebyshev_matches_dense_n4() {
        let n = 4_usize;
        let dim = 1usize << n;
        let mut rng = Xor64::new(31415);
        let h_x: Vec<f64> = (0..n).map(|_| rng.signed()).collect();
        let h_p_diag: Vec<f64> = (0..dim).map(|_| rng.signed()).collect();
        let psi = random_complex_vec(dim, 99);
        let dt = 1.5_f64;
        let a_t = 1.0_f64;
        let b_t = 1.0_f64;
        let tol = 1e-13;

        let h_dense = build_dense_h_real(n, &h_x, &h_p_diag, a_t, b_t);
        let expected = reference_propagate_real_h(&h_dense, &psi, dt);

        let (actual, k_used, _) = chebyshev_propagate(&h_x, &h_p_diag, a_t, b_t, &psi, dt, tol, n);
        let rel = relative_error(&actual, &expected);
        assert!(
            rel < 1e-12,
            "rel = {rel:e} (n=4, dt={dt}, tol={tol}, K_used={k_used})",
        );
    }

    /// unitary: ‖ψ_new‖ ≈ ‖ψ‖.
    #[test]
    fn chebyshev_preserves_norm() {
        let n = 5_usize;
        let dim = 1usize << n;
        let mut rng = Xor64::new(7);
        let h_x: Vec<f64> = (0..n).map(|_| rng.signed()).collect();
        let h_p_diag: Vec<f64> = (0..dim).map(|_| rng.signed()).collect();
        let psi = random_complex_vec(dim, 42);
        let psi_norm: f64 = psi.iter().map(|p| p.norm_sqr()).sum::<f64>().sqrt();
        let dt = 0.7_f64;
        let a_t = 0.5_f64;
        let b_t = 0.5_f64;

        let (result, _, _) = chebyshev_propagate(&h_x, &h_p_diag, a_t, b_t, &psi, dt, 1e-13, n);
        let new_norm: f64 = result.iter().map(|p| p.norm_sqr()).sum::<f64>().sqrt();
        let rel = (new_norm - psi_norm).abs() / psi_norm.max(1.0);
        assert!(
            rel < 1e-12,
            "norm rel = {rel:e} (before = {psi_norm}, after = {new_norm})",
        );
    }

    /// H = 0 fast-path: ψ_new = ψ (E_c=0 のとき) または ψ·exp(-i E_c dt).
    /// Gershgorin 上下界が両方 0 になる条件は h_x = 0 かつ h_p_diag = 0.
    #[test]
    fn chebyshev_zero_hamiltonian_returns_input() {
        let n = 4_usize;
        let dim = 1usize << n;
        let h_x = vec![0.0_f64; n];
        let h_p_diag = vec![0.0_f64; dim];
        let psi = random_complex_vec(dim, 314);
        let dt = 0.5_f64;

        let (result, k_used, err) =
            chebyshev_propagate(&h_x, &h_p_diag, 1.0, 1.0, &psi, dt, 1e-13, n);
        assert_eq!(k_used, 0, "zero Hamiltonian fast-path expected K_used = 0");
        assert_eq!(err, 0.0, "zero Hamiltonian fast-path expected err = 0");
        // E_c = 0 のはずなのでそのまま一致.
        let rel = relative_error(&result, &psi);
        assert!(rel < 1e-14, "rel = {rel:e}, expected ψ_new = ψ");
    }

    /// 対角 H: h_x = 0 で `ψ_new[k] = exp(-i dt · b_t · h_p_diag[k]) · ψ[k]`.
    #[test]
    fn chebyshev_diagonal_h_applies_phase() {
        let n = 4_usize;
        let dim = 1usize << n;
        let h_x = vec![0.0_f64; n];
        let mut rng = Xor64::new(42);
        let h_p_diag: Vec<f64> = (0..dim).map(|_| rng.signed()).collect();
        let psi = random_complex_vec(dim, 13);
        let dt = 0.5_f64;
        let b_t = 1.0_f64;

        let (result, _, _) = chebyshev_propagate(&h_x, &h_p_diag, 0.0, b_t, &psi, dt, 1e-13, n);

        let mut expected = vec![Complex64::new(0.0, 0.0); dim];
        for k in 0..dim {
            let lam_k = b_t * h_p_diag[k];
            let phase = Complex64::new(0.0, -dt * lam_k).exp();
            expected[k] = phase * psi[k];
        }
        let rel = relative_error(&result, &expected);
        assert!(rel < 1e-12, "diagonal H rel = {rel:e}");
    }

    /// `chebyshev_recurrence_fused_scalar` が scalar 2-loop 旧実装と
    /// machine-epsilon オーダで一致する (= fusion で再現される演算順序の
    /// sanity check).
    #[test]
    fn chebyshev_recurrence_fused_scalar_matches_legacy_two_loop() {
        let mut rng = Xor64::new(0xFEED_FACE_F00D_BABE);
        for _ in 0..20 {
            let n = 2 + (rng.next_u64() as usize % 5);
            let dim = 1usize << n;
            let scratch_in = random_complex_vec(dim, rng.next_u64());
            let phi_curr = random_complex_vec(dim, rng.next_u64());
            let phi_prev = random_complex_vec(dim, rng.next_u64());
            let psi_acc_in = random_complex_vec(dim, rng.next_u64());
            let e_c = rng.signed() * 3.0;
            let inv_r = 0.1 + rng.signed().abs();
            let c_k = rng.complex_signed();

            // Legacy two-loop reference.
            let mut scratch_l = scratch_in.clone();
            let mut psi_acc_l = psi_acc_in.clone();
            for j in 0..dim {
                let tilde = (scratch_l[j] - Complex64::new(e_c, 0.0) * phi_curr[j])
                    * Complex64::new(inv_r, 0.0);
                scratch_l[j] = Complex64::new(2.0, 0.0) * tilde - phi_prev[j];
            }
            for j in 0..dim {
                psi_acc_l[j] += c_k * scratch_l[j];
            }

            // Fused scalar.
            let mut scratch_f = scratch_in.clone();
            let mut psi_acc_f = psi_acc_in.clone();
            chebyshev_recurrence_fused_scalar(
                &mut scratch_f,
                &phi_curr,
                &phi_prev,
                &mut psi_acc_f,
                e_c,
                inv_r,
                c_k,
            );

            let rel_s = relative_error(&scratch_f, &scratch_l);
            let rel_p = relative_error(&psi_acc_f, &psi_acc_l);
            assert!(
                rel_s < 1e-14,
                "scratch rel = {rel_s:e} (n={n}, e_c={e_c}, inv_r={inv_r}, c_k={c_k})"
            );
            assert!(
                rel_p < 1e-14,
                "psi_acc rel = {rel_p:e} (n={n}, e_c={e_c}, inv_r={inv_r}, c_k={c_k})"
            );
        }
    }

    /// SIMD fused kernel と scalar fused kernel が rel < 1e-13 で一致する
    /// (issue #126). 浮動小数演算順序差で ulp 差は出るが ≤ 1e-13 を要求.
    /// random inputs 100 iter で fuzz する.
    #[cfg(feature = "simd")]
    #[test]
    fn chebyshev_recurrence_fused_simd_matches_scalar() {
        let mut rng = Xor64::new(0xCAFE_BABE_DEAD_BEEF);
        for iter in 0..100 {
            let n = 2 + (rng.next_u64() as usize % 5);
            let dim = 1usize << n;
            let scratch_in = random_complex_vec(dim, rng.next_u64());
            let phi_curr = random_complex_vec(dim, rng.next_u64());
            let phi_prev = random_complex_vec(dim, rng.next_u64());
            let psi_acc_in = random_complex_vec(dim, rng.next_u64());
            let e_c = rng.signed() * 3.0;
            let inv_r = 0.1 + rng.signed().abs();
            let c_k = rng.complex_signed();

            // scalar baseline.
            let mut scratch_s = scratch_in.clone();
            let mut psi_acc_s = psi_acc_in.clone();
            chebyshev_recurrence_fused_scalar(
                &mut scratch_s,
                &phi_curr,
                &phi_prev,
                &mut psi_acc_s,
                e_c,
                inv_r,
                c_k,
            );

            // SIMD kernel.
            let mut scratch_v = scratch_in.clone();
            let mut psi_acc_v = psi_acc_in.clone();
            simd_kernels::chebyshev_recurrence_fused(
                &mut scratch_v,
                &phi_curr,
                &phi_prev,
                &mut psi_acc_v,
                e_c,
                inv_r,
                c_k,
            );

            let rel_s = relative_error(&scratch_v, &scratch_s);
            let rel_p = relative_error(&psi_acc_v, &psi_acc_s);
            assert!(
                rel_s < 1e-13,
                "iter {iter}: scratch rel = {rel_s:e} (n={n}, e_c={e_c}, inv_r={inv_r}, c_k={c_k})"
            );
            assert!(
                rel_p < 1e-13,
                "iter {iter}: psi_acc rel = {rel_p:e} (n={n}, e_c={e_c}, inv_r={inv_r}, c_k={c_k})"
            );
        }
    }

    /// rayon path (`chebyshev_recurrence_fused_rayon`) と single-thread SIMD
    /// kernel が `rel < 1e-13` で一致する (issue #127 PoC).
    ///
    /// rayon path に確実に乗せるため `dim = MIN_RAYON_DIM_CHEB = 1 << 17` で
    /// 走らせる. 1 vector あたり 2 MB Complex64, 4 vector + clones で per-iter
    /// 16-32 MB なので iter 数は控えめ (10).
    #[cfg(feature = "rayon")]
    #[test]
    fn chebyshev_recurrence_fused_rayon_matches_serial() {
        let mut rng = Xor64::new(0xBEEF_CAFE_DEAD_FACE);
        let dim = MIN_RAYON_DIM_CHEB;
        for iter in 0..10 {
            let scratch_in = random_complex_vec(dim, rng.next_u64());
            let phi_curr = random_complex_vec(dim, rng.next_u64());
            let phi_prev = random_complex_vec(dim, rng.next_u64());
            let psi_acc_in = random_complex_vec(dim, rng.next_u64());
            let e_c = rng.signed() * 3.0;
            let inv_r = 0.1 + rng.signed().abs();
            let c_k = rng.complex_signed();

            // single-thread baseline: SIMD ON では SIMD kernel, OFF では scalar
            // fused. これが rayon path の "内側 kernel" と完全一致するため,
            // 演算順序の差は **rayon の chunking 起因のみ** に絞られる.
            let mut scratch_s = scratch_in.clone();
            let mut psi_acc_s = psi_acc_in.clone();
            #[cfg(feature = "simd")]
            simd_kernels::chebyshev_recurrence_fused(
                &mut scratch_s,
                &phi_curr,
                &phi_prev,
                &mut psi_acc_s,
                e_c,
                inv_r,
                c_k,
            );
            #[cfg(not(feature = "simd"))]
            chebyshev_recurrence_fused_scalar(
                &mut scratch_s,
                &phi_curr,
                &phi_prev,
                &mut psi_acc_s,
                e_c,
                inv_r,
                c_k,
            );

            // rayon path.
            let mut scratch_r = scratch_in.clone();
            let mut psi_acc_r = psi_acc_in.clone();
            chebyshev_recurrence_fused_rayon(
                &mut scratch_r,
                &phi_curr,
                &phi_prev,
                &mut psi_acc_r,
                e_c,
                inv_r,
                c_k,
            );

            // 各 chunk が独立に同じ kernel を呼ぶので, 同じ chunk 内では bit-
            // identical, chunk 境界の有無で順序差は出ない. よって理論上は
            // bit-identical だが, safety margin で 1e-13 を要求する.
            let rel_s = relative_error(&scratch_r, &scratch_s);
            let rel_p = relative_error(&psi_acc_r, &psi_acc_s);
            assert!(
                rel_s < 1e-13,
                "iter {iter}: scratch rel = {rel_s:e} (dim={dim}, e_c={e_c}, inv_r={inv_r}, c_k={c_k})"
            );
            assert!(
                rel_p < 1e-13,
                "iter {iter}: psi_acc rel = {rel_p:e} (dim={dim}, e_c={e_c}, inv_r={inv_r}, c_k={c_k})"
            );
        }
    }

    /// `chebyshev_propagate` end-to-end で rayon 経路と single-thread 経路が
    /// `rel < 1e-13` で一致することの間接確認 (issue #127). dim < MIN_RAYON_DIM_CHEB
    /// では single-thread 経路, dim ≥ MIN_RAYON_DIM_CHEB では rayon 経路に乗る.
    /// N=17 で 2^17 = 131072 dim を 1 step だけ走らせる (heavy だが回数 1 で 抑制).
    #[cfg(feature = "rayon")]
    #[test]
    fn chebyshev_propagate_rayon_path_smoke() {
        let n = 17_usize;
        let dim = 1usize << n;
        let mut rng = Xor64::new(0xDEAD_BEEF_BABE_CAFE);
        let h_x: Vec<f64> = (0..n).map(|_| rng.signed()).collect();
        let h_p_diag: Vec<f64> = (0..dim).map(|_| rng.signed()).collect();
        let psi = random_complex_vec(dim, 17);
        let dt = 0.1_f64;
        let a_t = 0.6_f64;
        let b_t = 0.4_f64;
        let tol = 1e-10;

        let (result, k_used, _) = chebyshev_propagate(&h_x, &h_p_diag, a_t, b_t, &psi, dt, tol, n);
        assert!(k_used >= 1, "K_used = {k_used} should be >= 1");
        // unitarity: ‖ψ_new‖ ≈ ‖ψ‖.
        let psi_norm: f64 = psi.iter().map(|p| p.norm_sqr()).sum::<f64>().sqrt();
        let new_norm: f64 = result.iter().map(|p| p.norm_sqr()).sum::<f64>().sqrt();
        let rel = (new_norm - psi_norm).abs() / psi_norm.max(1.0);
        assert!(
            rel < 1e-10,
            "norm rel = {rel:e} (before = {psi_norm}, after = {new_norm}, K_used = {k_used})"
        );
    }

    /// Gershgorin bounds の sanity: 真の eigenvalue 範囲を含む.
    #[test]
    fn gershgorin_includes_true_spectrum() {
        let n = 4_usize;
        let dim = 1usize << n;
        let mut rng = Xor64::new(2024);
        let h_x: Vec<f64> = (0..n).map(|_| rng.signed()).collect();
        let h_p_diag: Vec<f64> = (0..dim).map(|_| rng.signed()).collect();
        let a_t = 0.7_f64;
        let b_t = 0.3_f64;

        let (e_min, e_max) = gershgorin_bounds(&h_x, &h_p_diag, a_t, b_t);

        let h_dense = build_dense_h_real(n, &h_x, &h_p_diag, a_t, b_t);
        let eig = SymmetricEigen::new(h_dense);
        let true_min = eig
            .eigenvalues
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let true_max = eig
            .eigenvalues
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        assert!(
            e_min <= true_min + 1e-12,
            "e_min = {e_min} should be ≤ true_min = {true_min}",
        );
        assert!(
            e_max >= true_max - 1e-12,
            "e_max = {e_max} should be ≥ true_max = {true_max}",
        );
    }
}
