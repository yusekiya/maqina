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

use crate::matvec::apply_h_kryanneal;

/// Power iteration で `R_refined = max |λ(H - E_c·I)|` を推定するヘルパ
/// (issue #125 PoC, R-only refine).
///
/// # 動機
///
/// 既存 [`gershgorin_bounds`] は閉形式 O(N) で `(E_min, E_max)` を返すが,
/// オフ対角の正負キャンセル + 対角値の局在性により TFIM では典型的に
/// `R_true / R_gershgorin ~ 0.5-0.7` の loose 上界となる. `z = R · dt` で
/// `K_used` が決まる (`determine_truncation`) ので, R を真値に近づければ
/// per-step matvec を ~1.3-1.5× 削減できる (issue #125 期待値).
///
/// # アルゴリズム
///
/// shifted operator `M = H - E_c·I` に対する標準 power iteration. eigenvalue
/// 推定子は **operator norm `‖M v‖`** (= `sqrt(<v, M^2 v>)`) を使う:
///
/// 1. 初期ベクトル `v_0` を xorshift64 (seed) で複素一様サンプリング + L2 正規化
/// 2. 各 iter:
///    * `w = H v - E_c v`
///    * `λ_est ← ‖w‖` (v は正規化済み, ‖v‖ = 1 なので `‖M v‖` がそのまま値)
///    * `v ← w / ‖w‖`
/// 3. 最後の `λ_est` を返す
///
/// Rayleigh quotient `<v, M v>` を使わない理由: TFIM 構造で **spectrum が 0 周りに
/// ±λ 対称** な scenario (= 本 perf bench の `a_t = b_t = 0.5`, `h_p_diag` 0 中心)
/// では `<v, M v> = Σ |c_λ|² λ` の符号付き総和で **±λ projection が cancel** し
/// `λ_est ≈ 0` を誤って返す. 一方 `‖M v‖² = Σ |c_λ|² λ²` は各項非負なので
/// cancel しない (古典的 power iter の symmetric spectrum 弱点を回避する標準手法,
/// issue #125 PoC Linux bench で確認).
///
/// # 戻り値
///
/// `R_refined`: `max |λ(H - E_c·I)|` の下界推定値. 呼出側で安全マージン
/// (e.g. `R_used = R_refined * 1.05`) を掛けて Chebyshev `tilde H` の
/// `|eigenvalue| ≤ 1` 不変条件を保証する.
///
/// # 計算量
///
/// `n_iter + 1` 回の `apply_h_kryanneal` 呼出 (= matvec) + 2·n_iter 回の
/// O(dim) BLAS-1 ops. `n_iter = 5-10` 想定. per-step amortize で 1 step あたり
/// ~0.1 ms (#125 期待値).
///
/// # 案 B (E_c も refine する 2-pass 版) との関係 (本 PoC は採用しない)
///
/// `E_c` も Power iter で refine する案は本 helper を **2-pass** に拡張する形:
///
/// 1. pass 1: 生 `H` に power iter → `λ_ext = argmax|λ(H)|` (=λ_max または λ_min)
/// 2. pass 2: shifted `(H - λ_ext·I)` に power iter → 反対端 `λ_other`
/// 3. return `(min, max) = (min(λ_ext, λ_other), max(λ_ext, λ_other))`
///
/// 利点: `E_c_true = (λ_max+λ_min)/2`, `R_true = (λ_max-λ_min)/2` (最小 R) が
/// 取れる. 欠点: matvec が 2× / degenerate spectrum (λ_max ≈ -λ_min) で 2-pass
/// stability check が要 / TFIM では `h_p_diag` が 0 周りに対称なシナリオで
/// `E_c_g ≈ E_c_true` となり追加利得 α が小さい見込み.
///
/// 案 B 採用時は本関数より `(f64, f64)` を返す `power_iter_spectrum_bounds` に
/// rename し, `chebyshev_propagate` の `r_override: Option<f64>` も
/// `bounds_override: Option<(f64, f64)>` に置き換えるのが API 対称性として
/// 自然 (Lanczos の `(E_min, E_max)` インタフェースに揃う). 本 PoC では
/// scope 外として R-only に絞り, full 実装 (CFM4 統合 + adaptive driver) で
/// 必要なら起票する.
#[allow(clippy::too_many_arguments)]
pub fn power_iter_spectral_radius(
    h_x: &[f64],
    h_p_diag: &[f64],
    a_t: f64,
    b_t: f64,
    e_c: f64,
    n: usize,
    n_iter: usize,
    seed: u64,
) -> f64 {
    let dim = 1usize << n;
    assert_eq!(h_x.len(), n, "h_x must have length n");
    assert_eq!(h_p_diag.len(), dim, "h_p_diag must have length 2^n");

    if n_iter == 0 {
        return 0.0;
    }

    // 1. 初期ベクトル: xorshift64 で複素一様サンプリング + L2 正規化.
    //    決定論 (seed 経由) で run-to-run 再現性を担保する.
    let mut state = seed | 1;
    let mut next_u64 = || {
        let mut x = state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        state = x;
        x
    };
    let mut signed = || (next_u64() as f64) / (u64::MAX as f64) * 2.0 - 1.0;
    let mut v: Vec<Complex64> = (0..dim)
        .map(|_| Complex64::new(signed(), signed()))
        .collect();
    let mut nrm: f64 = v.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
    if nrm < 1e-300 {
        // pathological: all-zero seed produced零ベクトル. unit basis に逃がす.
        v[0] = Complex64::new(1.0, 0.0);
        nrm = 1.0;
    }
    let inv_n = 1.0 / nrm;
    for c in v.iter_mut() {
        *c *= Complex64::new(inv_n, 0.0);
    }

    let mut w = vec![Complex64::new(0.0, 0.0); dim];
    let mut lambda_abs = 0.0_f64;

    for _ in 0..n_iter {
        // w := H v (apply_h_kryanneal は overwrite 形なので out-buffer reuse OK).
        apply_h_kryanneal(&v, &mut w, h_x, h_p_diag, a_t, b_t, n);
        // w := H v - E_c v = M v.
        for k in 0..dim {
            w[k] -= Complex64::new(e_c, 0.0) * v[k];
        }
        // λ_est := ‖M v‖ = ‖w‖. v は正規化済み (‖v‖=1) なので, これがそのまま
        // |λ_max(M)| への下界推定値となる. Rayleigh quotient `<v, M v>` を使わない
        // 理由は docstring 参照 (symmetric spectrum での ±λ cancel 回避).
        let w_norm: f64 = w.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
        lambda_abs = w_norm;

        // v <- w / ‖w‖. ‖w‖ ~ 0 (M v が零空間にしか乗らない pathological) は
        // power iter が動かなくなる極端ケースだが, 1 iter で抜けるのが安全.
        if w_norm < 1e-300 {
            break;
        }
        let inv_w = 1.0 / w_norm;
        for k in 0..dim {
            v[k] = w[k] * Complex64::new(inv_w, 0.0);
        }
    }

    lambda_abs
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
/// * `r_override`: opt-in spectral radius 上書き (issue #125 PoC). `Some(r)`
///   のとき Gershgorin 由来の `R` をこの値で置き換える. `E_c` は引き続き
///   Gershgorin 由来 (本 PoC は R-only refine スコープ). 呼出側で
///   [`power_iter_spectral_radius`] 等で `R_true` を推定して安全マージン込みで
///   渡す. **必ず `R_override ≥ R_true` を満たすこと** (下回ると `tilde H` の
///   固有値が [-1, 1] を超え Chebyshev 展開が divergent になる). `None` で
///   従来 Gershgorin 経路 (safe default).
///
///   案 B (E_c も refine する 2-pass 版) に拡張するときは, 本引数を
///   `bounds_override: Option<(f64, f64)>` (E_c, R 両方) に置換するのが
///   `gershgorin_bounds` の `(E_min, E_max)` インタフェースとの API 対称性
///   として自然. 詳細は [`power_iter_spectral_radius`] docstring 末尾.
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
    r_override: Option<f64>,
) -> (Vec<Complex64>, usize, f64) {
    let dim = 1usize << n;
    assert_eq!(psi.len(), dim, "psi must have length 2^n");
    assert_eq!(h_x.len(), n, "h_x must have length n");
    assert_eq!(h_p_diag.len(), dim, "h_p_diag must have length 2^n");

    // 1. Gershgorin bounds (E_c は常にここから; R は r_override で上書き可).
    //    案 B (E_c も refine) を将来採用する場合は bounds_override: Option<(f64, f64)>
    //    に置換し E_c もここで上書きする.
    let (e_min, e_max) = gershgorin_bounds(h_x, h_p_diag, a_t, b_t);
    let e_c = 0.5 * (e_max + e_min);
    let r_gershgorin = 0.5 * (e_max - e_min);
    let r = r_override.unwrap_or(r_gershgorin);

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

    // 6. k ≥ 2 の漸化. k_ord は jvals[k_ord] と `k_ord % 4` 両方で必要
    // (前者は coefficient, 後者は `(-i)^k` の 4 周期 dispatch) なので
    // iterator chain への書き換えは可読性を下げる. needless_range_loop を allow.
    #[allow(clippy::needless_range_loop)]
    for k_ord in 2..=k_used {
        // scratch := H · phi_curr.
        apply_h_kryanneal(&phi_curr, &mut scratch, h_x, h_p_diag, a_t, b_t, n);
        // scratch := 2 · (scratch - E_c · phi_curr) / R - phi_prev = φ_{k_ord}.
        for j in 0..dim {
            let tilde_h_phi =
                (scratch[j] - Complex64::new(e_c, 0.0) * phi_curr[j]) * Complex64::new(inv_r, 0.0);
            scratch[j] = Complex64::new(2.0, 0.0) * tilde_h_phi - phi_prev[j];
        }
        // c_{k_ord} = 2 · (-i)^{k_ord} · J_{k_ord}(z). (-i)^k は 4 周期で循環.
        let pow_minus_i = match k_ord % 4 {
            0 => Complex64::new(1.0, 0.0),
            1 => Complex64::new(0.0, -1.0),
            2 => Complex64::new(-1.0, 0.0),
            3 => Complex64::new(0.0, 1.0),
            _ => unreachable!(),
        };
        let ck = Complex64::new(2.0 * jvals[k_ord], 0.0) * pow_minus_i;
        for j in 0..dim {
            psi_acc[j] += ck * scratch[j];
        }
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
            chebyshev_propagate(&h_x, &h_p_diag, a_t, b_t, &psi, dt, tol, n, None);
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

        let (actual, k_used, _) =
            chebyshev_propagate(&h_x, &h_p_diag, a_t, b_t, &psi, dt, tol, n, None);
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

        let (result, _, _) =
            chebyshev_propagate(&h_x, &h_p_diag, a_t, b_t, &psi, dt, 1e-13, n, None);
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
            chebyshev_propagate(&h_x, &h_p_diag, 1.0, 1.0, &psi, dt, 1e-13, n, None);
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

        let (result, _, _) =
            chebyshev_propagate(&h_x, &h_p_diag, 0.0, b_t, &psi, dt, 1e-13, n, None);

        let mut expected = vec![Complex64::new(0.0, 0.0); dim];
        for k in 0..dim {
            let lam_k = b_t * h_p_diag[k];
            let phase = Complex64::new(0.0, -dt * lam_k).exp();
            expected[k] = phase * psi[k];
        }
        let rel = relative_error(&result, &expected);
        assert!(rel < 1e-12, "diagonal H rel = {rel:e}");
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

    // ============================================================
    // issue #125 PoC: Power iteration による R refine + r_override 経路
    // ============================================================

    /// Power iter が `R_true = max|λ(H - E_c·I)|` の **下界** として動作し,
    /// かつ Gershgorin 上界より tighter であることを確認.
    ///
    /// 注: 絶対精度 (rel < 1e-6 等) は小 dim では gap ratio が大きく遅収束する.
    /// たとえば N=4 / 50 iter で rel ~ 7% にとどまる. PoC scope の判定 gate は
    /// **N=16-20 での K_used 縮小率** (perf binary) で行うため, ここでは
    /// アルゴリズム正当性のみを検証する (from-below convergence + Gershgorin
    /// より tight).
    #[test]
    fn power_iter_bounds_true_radius_from_below() {
        let n = 4_usize;
        let dim = 1usize << n;
        let mut rng = Xor64::new(2024);
        let h_x: Vec<f64> = (0..n).map(|_| rng.signed()).collect();
        let h_p_diag: Vec<f64> = (0..dim).map(|_| rng.signed()).collect();
        let a_t = 0.7_f64;
        let b_t = 0.3_f64;

        // 真値: H の固有値 λ_i に対し R_true = max|λ_i - E_c_gershgorin|.
        let (e_min_g, e_max_g) = gershgorin_bounds(&h_x, &h_p_diag, a_t, b_t);
        let e_c_g = 0.5 * (e_max_g + e_min_g);
        let r_gershgorin = 0.5 * (e_max_g - e_min_g);
        let h_dense = build_dense_h_real(n, &h_x, &h_p_diag, a_t, b_t);
        let eig = SymmetricEigen::new(h_dense);
        let r_true = eig
            .eigenvalues
            .iter()
            .map(|l| (l - e_c_g).abs())
            .fold(0.0_f64, f64::max);

        let r_pi = power_iter_spectral_radius(&h_x, &h_p_diag, a_t, b_t, e_c_g, n, 50, 12345);

        // (1) from-below convergence: R_pi ≤ R_true + tiny tolerance (収束途中で
        //     overshoot しない). f64 算術 noise を許容する微小マージン込み.
        assert!(
            r_pi <= r_true * (1.0 + 1e-10),
            "Power iter must be from below: r_pi = {r_pi} > r_true = {r_true}",
        );

        // (2) Gershgorin より tighter (= 小さい) であること. TFIM ではこれが
        //     期待動作 (PoC が成立する前提).
        assert!(
            r_pi < r_gershgorin,
            "Power iter R ({r_pi}) should be < R_gershgorin ({r_gershgorin})",
        );

        // (3) progress sanity: 50 iter で R_true の少なくとも 50% は埋めている.
        //     gap ratio が極端 (=1 近傍) でも 50 iter なら半分は埋まる経験則.
        assert!(
            r_pi > 0.5 * r_true,
            "Power iter R ({r_pi}) should make significant progress toward r_true ({r_true})",
        );
    }

    /// `r_override = Some(R_gershgorin)` が `None` 経路と数値的に一致.
    /// 内部で R 値が同じになる経路の sanity check (bit-for-bit までは保証しない;
    /// f64 算術の同一性で rel < 1e-15).
    #[test]
    fn chebyshev_with_r_override_matches_default() {
        let n = 4_usize;
        let dim = 1usize << n;
        let mut rng = Xor64::new(31415);
        let h_x: Vec<f64> = (0..n).map(|_| rng.signed()).collect();
        let h_p_diag: Vec<f64> = (0..dim).map(|_| rng.signed()).collect();
        let psi = random_complex_vec(dim, 99);
        let dt = 1.0_f64;
        let a_t = 0.5_f64;
        let b_t = 0.5_f64;
        let tol = 1e-13;

        let (e_min, e_max) = gershgorin_bounds(&h_x, &h_p_diag, a_t, b_t);
        let r_g = 0.5 * (e_max - e_min);

        let (psi_default, k_def, _) =
            chebyshev_propagate(&h_x, &h_p_diag, a_t, b_t, &psi, dt, tol, n, None);
        let (psi_over, k_over, _) =
            chebyshev_propagate(&h_x, &h_p_diag, a_t, b_t, &psi, dt, tol, n, Some(r_g));

        assert_eq!(
            k_def, k_over,
            "K_used should match when r_override = R_gershgorin",
        );
        let rel = relative_error(&psi_over, &psi_default);
        assert!(
            rel < 1e-15,
            "rel = {rel:e}: r_override=Some(R_g) must match None path",
        );
    }

    /// Power iter + 5% safety margin で得た R で Chebyshev を回し,
    /// 真値 (dense eigendecomp) と rel < 1e-12 一致. 安全マージンが効いて
    /// `\tilde H` の固有値が [-1, 1] 内に収まり Chebyshev 展開が divergent
    /// しないことを保証する.
    #[test]
    fn power_iter_safety_margin_keeps_chebyshev_stable() {
        let n = 4_usize;
        let dim = 1usize << n;
        let mut rng = Xor64::new(7);
        let h_x: Vec<f64> = (0..n).map(|_| rng.signed()).collect();
        let h_p_diag: Vec<f64> = (0..dim).map(|_| rng.signed()).collect();
        let psi = random_complex_vec(dim, 42);
        let dt = 0.8_f64;
        let a_t = 0.6_f64;
        let b_t = 0.4_f64;
        let tol = 1e-13;

        let (e_min_g, e_max_g) = gershgorin_bounds(&h_x, &h_p_diag, a_t, b_t);
        let e_c_g = 0.5 * (e_max_g + e_min_g);
        let r_pi = power_iter_spectral_radius(&h_x, &h_p_diag, a_t, b_t, e_c_g, n, 20, 4242);
        let r_used = r_pi * 1.05; // 5% safety margin (= POWER_ITER_SAFETY in perf binary).

        let (actual, k_used, _) =
            chebyshev_propagate(&h_x, &h_p_diag, a_t, b_t, &psi, dt, tol, n, Some(r_used));

        // 真値 (dense): exp(-i dt H) ψ.
        let h_dense = build_dense_h_real(n, &h_x, &h_p_diag, a_t, b_t);
        let expected = reference_propagate_real_h(&h_dense, &psi, dt);

        let rel = relative_error(&actual, &expected);
        assert!(
            rel < 1e-12,
            "rel = {rel:e} (K_used = {k_used}, R_used = {r_used}, R_g = {})",
            0.5 * (e_max_g - e_min_g),
        );

        // refined R は Gershgorin R より小さく, K_used も同条件比で小さく出る
        // ことが期待される (PoC の存在意義の sanity check).
        let (_, k_g, _) = chebyshev_propagate(&h_x, &h_p_diag, a_t, b_t, &psi, dt, tol, n, None);
        assert!(
            k_used <= k_g,
            "K_used (power_iter) = {k_used} should be ≤ K_used (gershgorin) = {k_g}",
        );
    }

    /// `b_t = 0` で H = -a_t·Σ h_x_i X_i (TFIM driver 単独, 純 bit-flip 構造)
    /// は **0 周りに完全 ±対称な spectrum** を持つ. Rayleigh quotient ベースの
    /// 旧実装は ±λ projection cancel で `λ_est ≈ 0` を誤って返す (issue #125
    /// Linux N=18 bench で確認). 現実装は ‖M v‖ ベースで cancel 不要なので
    /// この case でも R_true に収束することを検証する.
    #[test]
    fn power_iter_handles_symmetric_spectrum() {
        let n = 5_usize;
        let dim = 1usize << n;
        let mut rng = Xor64::new(2024);
        let h_x: Vec<f64> = (0..n).map(|_| rng.signed()).collect();
        let h_p_diag = vec![0.0_f64; dim]; // 純 driver: H = -a_t · Σ h_x_i X_i.
        let a_t = 0.7_f64;
        let b_t = 0.0_f64; // 対角寄与なし → spectrum は ±対称.

        // 真値: H の固有値は ±a_t Σ ε_i h_x_i for ε_i ∈ {±1}.
        // dense diag で確認.
        let (e_min_g, e_max_g) = gershgorin_bounds(&h_x, &h_p_diag, a_t, b_t);
        let e_c_g = 0.5 * (e_max_g + e_min_g);
        let h_dense = build_dense_h_real(n, &h_x, &h_p_diag, a_t, b_t);
        let eig = SymmetricEigen::new(h_dense);
        let r_true = eig
            .eigenvalues
            .iter()
            .map(|l| (l - e_c_g).abs())
            .fold(0.0_f64, f64::max);

        // E_c_g = 0 が期待されるが gershgorin の round noise で完全 0 ではない場合も
        // 許容. R_true は λ_max(H) = a_t · Σ |h_x_i| になるはず.
        let expected_r = a_t * h_x.iter().map(|h| h.abs()).sum::<f64>();
        assert!(
            (r_true - expected_r).abs() < 1e-12,
            "r_true sanity: dense = {r_true}, expected = {expected_r}",
        );

        let r_pi = power_iter_spectral_radius(&h_x, &h_p_diag, a_t, b_t, e_c_g, n, 20, 4242);

        // symmetric spectrum でも R_pi が R_true に有意に到達 (≥ 80%) する.
        // 旧 Rayleigh quotient 実装ではここで ~0 が返り fail する設計.
        assert!(
            r_pi > 0.8 * r_true,
            "symmetric spectrum: r_pi = {r_pi} should be ≥ 80% of r_true = {r_true}",
        );

        // from-below 性質も維持.
        assert!(
            r_pi <= r_true * (1.0 + 1e-10),
            "from-below violated: r_pi = {r_pi} > r_true = {r_true}",
        );
    }

    /// `n_iter = 0` は no-op で 0 を返す (edge case).
    #[test]
    fn power_iter_zero_iter_returns_zero() {
        let n = 3_usize;
        let dim = 1usize << n;
        let mut rng = Xor64::new(2024);
        let h_x: Vec<f64> = (0..n).map(|_| rng.signed()).collect();
        let h_p_diag: Vec<f64> = (0..dim).map(|_| rng.signed()).collect();
        let r = power_iter_spectral_radius(&h_x, &h_p_diag, 1.0, 1.0, 0.0, n, 0, 1);
        assert_eq!(r, 0.0, "n_iter=0 should return 0.0");
    }
}
