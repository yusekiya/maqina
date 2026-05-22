//! Phase 9+ POC (issue #120): Chebyshev polynomial propagator (時間独立 H) を
//! Lanczos の代替として per-call wall を実測する pure-Rust 計測 binary.
//!
//! # 用途
//!
//! `perf_cfm4_richardson` の `single_lanczos` mode で per-call ~129 ms / IPC=0.78
//! の V matrix cache stall が #65 long-T シナリオの per-step bottleneck と
//! 確定 (PR #118). 本 binary は **同じ time-frozen Hamiltonian** に対し
//! Chebyshev 3 項漸化での per-call wall を実測し,
//! `perf_cfm4_richardson 18 100 single_lanczos` (Lanczos baseline) との直接比較を
//! 可能にする.
//!
//! 判定 gate (issue #120 README, Linux AMD EPYC 7713P, NT=64, OPENBLAS=8 default):
//!
//! | Chebyshev per-call wall | 判定 |
//! |---:|---|
//! | ≤ 50 ms (Lanczos の 40% 以下) | ✅ Phase B 進行 (CFM4 統合 + Richardson 経路) |
//! | 50-100 ms (Lanczos の 40-80%) | ⚠️ 設計再検討 (K_used 過大 / Gershgorin loose) |
//! | > 100 ms (Lanczos の 80% 以上) | ❌ 中止 (理論予測の前提崩れ) |
//!
//! # issue #125 PoC: spectral radius R refine via Power iteration
//!
//! `chebyshev_propagate` の spectral radius `R` を Gershgorin (default; 閉形式
//! 上界) と Power iteration (refine 後の真値近傍 R) で切替え, K_used / per-call
//! wall を比較する mode を 4 番目引数で持つ. 詳細は issue #125 と
//! `src/chebyshev.rs::power_iter_spectral_radius` docstring.
//!
//! 判定 gate (#125 PoC 完了基準):
//!
//! | K_used 縮小率 (N=16-20) | 次 action |
//! |---|---|
//! | ≥ 30% | full 実装 (CFM4 統合 + adaptive driver) を別 issue で起票 |
//! | 10-30% | 改善余地は小. Trotter pre-conditioning と組合せた form を検討 |
//! | < 10%  | 中止 + archive. Gershgorin で十分タイトだった結論 |
//!
//! # ビルド
//!
//! ```bash
//! RUSTFLAGS="-C target-cpu=native" cargo build --release --bin perf_chebyshev
//! ```
//!
//! `extension-module` feature は OFF (default features の `blas` / `rayon` /
//! `simd` のみ) なので pyo3 が libpython を静的リンクする (他の perf binary と同じ).
//!
//! # 計測例 (Linux AMD EPYC 7713P, Zen 3)
//!
//! ```bash
//! # 基本: IPC + cache + L2 fill latency (perf_cfm4_richardson と同じ counter セット)
//! RAYON_NUM_THREADS=64 perf stat \
//!     -e cycles,instructions,branch-misses \
//!     -e stalled-cycles-backend,stalled-cycles-frontend \
//!     -e cache-references,cache-misses \
//!     -e L1-dcache-loads,L1-dcache-load-misses \
//!     -e LLC-loads,LLC-load-misses \
//!     -e l2_request_g1.all_no_prefetch \
//!     -e l2_cache_req_stat.ic_dc_miss_in_l2 \
//!     -e l2_latency.l2_cycles_waiting_on_fills \
//!     -- ./target/release/perf_chebyshev 18 100 1e-10 gershgorin
//!
//! # Power iter refine variants (#125 PoC):
//! for niter in 5 10 20; do
//!     RAYON_NUM_THREADS=64 perf stat -e cycles,instructions,branch-misses \
//!         -- ./target/release/perf_chebyshev 18 100 1e-10 power_iter:$niter
//! done
//! ```
//!
//! # 引数
//!
//! `./perf_chebyshev <N> <n_iters> [tol] [r_mode]`
//!
//! - `N`: TFIM サイト数. dim = 2^N. default = 18 (PR #118 perf_cfm4_richardson と
//!   同じく本番想定; perf_cfm4_richardson との直接 wall 比較を可能にするため).
//! - `n_iters`: 計測 iter 数 (chebyshev_propagate を何回呼ぶか). default = 100.
//! - `tol`: Chebyshev 切り捨て次数決定の閾値. default = 1e-10 (perf_cfm4_richardson の
//!   `krylov_tol = 1e-10` と精度水準を揃える).
//! - `r_mode` (issue #125 PoC): `gershgorin` (default) または `power_iter:<N_iter>`.
//!   - `gershgorin`: 従来通り Gershgorin の閉形式行和上界で R を決定.
//!   - `power_iter:<N>`: 計測 loop 外で Power iteration を `N` 回回し
//!     `R_refined = max|λ(H - E_c_gershgorin·I)|` を推定. 安全マージン
//!     `R_used = R_refined · 1.05` を `chebyshev_propagate` の `r_override` に
//!     渡す. `E_c` は引き続き Gershgorin 由来 (本 PoC は R-only refine スコープ).
//!     例: `power_iter:5`, `power_iter:10`, `power_iter:20`.
//!
//! 安全マージン 5% は Tal-Ezer & Kosloff (1984) の経験則 `δ ∈ [0.01, 0.05]` の
//! 上端. Power iter は Rayleigh quotient が下から単調収束するため, 未収束時に
//! `R_refined < R_true` となる可能性があり, その場合 `\tilde H` の固有値が
//! [-1, 1] を超えて Chebyshev 展開が divergent になる. 5% でほぼ常に
//! `R_used ≥ R_true` を保証.
//!
//! # Schedule 係数
//!
//! 時間独立 frozen schedule `a_t = b_t = 0.5` (perf_cfm4_richardson の matvec_only /
//! single_lanczos mode と完全一致). H の compute pattern を bench 間で完全に揃える
//! ことで, Lanczos vs Chebyshev の差分を **アルゴリズム軸のみに帰着** させる.
//!
//! # 出力
//!
//! stderr に wall time / per-iter time / K_used 統計 / spectral radius 情報 /
//! sink (DCE 防止) を出す. stdout は perf の出力で汚さないよう空に保つ.

use std::env;
use std::time::Instant;

use _rust::bench_api::{chebyshev_propagate, gershgorin_bounds, power_iter_spectral_radius};
use num_complex::Complex64;

/// Power iter の安全マージン (issue #125 PoC). `R_used = R_refined · (1 + SAFETY)`
/// で Chebyshev `\tilde H` の固有値が [-1, 1] を超えないよう保証.
const POWER_ITER_SAFETY: f64 = 0.05;

enum RMode {
    Gershgorin,
    PowerIter(usize),
}

fn parse_r_mode(s: &str) -> Result<RMode, String> {
    if s == "gershgorin" {
        Ok(RMode::Gershgorin)
    } else if let Some(rest) = s.strip_prefix("power_iter:") {
        rest.parse::<usize>()
            .map(RMode::PowerIter)
            .map_err(|e| format!("invalid power_iter N_iter '{rest}': {e}"))
    } else {
        Err(format!(
            "unknown r_mode '{s}'. expected 'gershgorin' or 'power_iter:<N>'."
        ))
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(18);
    let n_iters: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(100);
    let tol: f64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1e-10);
    let r_mode = match args.get(4) {
        Some(s) => match parse_r_mode(s) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("error: {e}");
                std::process::exit(2);
            }
        },
        None => RMode::Gershgorin,
    };

    let dim: usize = 1 << n;
    let dt: f64 = 1.0;
    let a_t = 0.5_f64;
    let b_t = 0.5_f64;
    let rayon_threads = std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "(auto)".to_string());

    eprintln!("== perf_chebyshev ==");
    eprintln!("n = {n}, dim = {dim}");
    eprintln!("n_iters = {n_iters}");
    eprintln!("dt = {dt}, tol = {tol:e}");
    eprintln!("a_t = {a_t}, b_t = {b_t}");
    eprintln!("RAYON_NUM_THREADS = {rayon_threads}");
    match &r_mode {
        RMode::Gershgorin => eprintln!("r_mode = gershgorin"),
        RMode::PowerIter(ni) => {
            eprintln!("r_mode = power_iter:{ni} (safety = {POWER_ITER_SAFETY})")
        }
    }

    // 決定論的 seed で入力初期化. perf_cfm4_richardson と異なる seed で
    // run-to-run 再現性は確保しつつ data 衝突を避ける.
    let mut rng = XorShift64::new(0xCEEB_0FED_FACE_BABE ^ (n as u64));
    let h_x: Vec<f64> = (0..n).map(|_| rng.signed_unit()).collect();
    let h_p_diag: Vec<f64> = (0..dim).map(|_| rng.signed_unit()).collect();
    // psi 初期状態 (‖psi‖ = 1 に正規化).
    let mut psi: Vec<Complex64> = (0..dim)
        .map(|_| Complex64::new(rng.signed_unit(), rng.signed_unit()))
        .collect();
    let nrm = psi.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
    if nrm > 0.0 {
        for z in psi.iter_mut() {
            *z /= nrm;
        }
    }

    // Gershgorin baseline (常に計算; r_mode 判定にも使用).
    let (e_min_g, e_max_g) = gershgorin_bounds(&h_x, &h_p_diag, a_t, b_t);
    let e_c_g = 0.5 * (e_max_g + e_min_g);
    let r_gershgorin = 0.5 * (e_max_g - e_min_g);

    // r_override / Power iter wall を loop 外で 1 回確定 (H は frozen).
    // amortize 評価のため power_iter wall は独立に報告.
    let (r_override, r_used, r_refined, power_iter_wall_ms) = match &r_mode {
        RMode::Gershgorin => (None, r_gershgorin, None, 0.0),
        RMode::PowerIter(ni) => {
            // seed は H と独立 (再現性のみ担保).
            let seed = 0xBABE_FEED_DEAD_FACE_u64 ^ ((*ni as u64) << 8);
            let t0 = Instant::now();
            let r_ref = power_iter_spectral_radius(&h_x, &h_p_diag, a_t, b_t, e_c_g, n, *ni, seed);
            let dt_ms = t0.elapsed().as_secs_f64() * 1000.0;
            let r_used = r_ref * (1.0 + POWER_ITER_SAFETY);
            (Some(r_used), r_used, Some(r_ref), dt_ms)
        }
    };

    eprintln!("R_gershgorin = {r_gershgorin:.6}");
    if let Some(r_ref) = r_refined {
        eprintln!(
            "R_power_iter = {r_ref:.6} (ratio vs gershgorin = {:.4})",
            r_ref / r_gershgorin
        );
        eprintln!(
            "R_used       = {r_used:.6} (= R_power_iter · {:.2})",
            1.0 + POWER_ITER_SAFETY
        );
        eprintln!("power_iter wall = {power_iter_wall_ms:.3} ms (1-time, amortized over n_iters)");
    } else {
        eprintln!("R_used       = {r_used:.6} (= R_gershgorin)");
    }

    // warmup (rayon pool 起動, page fault 解消, cache warm).
    // matvec_only と異なり 1 iter のコストが大きいので 3 回程度.
    for _ in 0..3 {
        let _ = chebyshev_propagate(&h_x, &h_p_diag, a_t, b_t, &psi, dt, tol, n, r_override);
    }

    let t0 = Instant::now();
    let mut k_total: usize = 0;
    let mut sink_acc = 0.0_f64;
    for _ in 0..n_iters {
        let (psi_new, k_used, _err_estimate) =
            chebyshev_propagate(&h_x, &h_p_diag, a_t, b_t, &psi, dt, tol, n, r_override);
        k_total += k_used;
        // 出口 ψ_new の先頭数要素を sink に畳む (DCE 防止).
        sink_acc += psi_new.iter().take(8).map(|c| c.re + c.im).sum::<f64>();
    }
    let elapsed_secs = t0.elapsed().as_secs_f64();
    let sink = sink_acc + (k_total as f64) * 1e-30;

    eprintln!("---");
    eprintln!("K_used (avg) ≈ {:.2}", (k_total as f64) / (n_iters as f64));
    eprintln!("total = {:.6} sec", elapsed_secs);
    eprintln!(
        "per-iter = {:.6} ms",
        elapsed_secs / (n_iters as f64) * 1000.0
    );
    if let Some(_r_ref) = r_refined {
        let amortized_per_iter =
            elapsed_secs / (n_iters as f64) * 1000.0 + power_iter_wall_ms / (n_iters as f64);
        eprintln!("per-iter (incl. power_iter amortized) = {amortized_per_iter:.6} ms");
    }
    eprintln!("sink (anti-DCE) = {sink}");
}

/// 軽量 xorshift64 PRNG. `src/bin/perf_apply_h.rs::XorShift64` と同実装
/// (依存追加を避けるため inline; binary 間で `mod` 共有しない方が cargo の
/// dependency graph がシンプル).
struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self { state: seed | 1 }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// `[-1.0, 1.0)` の f64 を返す.
    fn signed_unit(&mut self) -> f64 {
        let bits = self.next_u64();
        let normalized = (bits as f64) / (u64::MAX as f64);
        2.0 * normalized - 1.0
    }
}
