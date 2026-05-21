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
//!     -- ./target/release/perf_chebyshev 18 100
//! ```
//!
//! # 引数
//!
//! `./perf_chebyshev <N> <n_iters> [tol]`
//!
//! - `N`: TFIM サイト数. dim = 2^N. default = 18 (PR #118 perf_cfm4_richardson と
//!   同じく本番想定; perf_cfm4_richardson との直接 wall 比較を可能にするため).
//! - `n_iters`: 計測 iter 数 (chebyshev_propagate を何回呼ぶか). default = 100.
//! - `tol`: Chebyshev 切り捨て次数決定の閾値. default = 1e-10 (perf_cfm4_richardson の
//!   `krylov_tol = 1e-10` と精度水準を揃える).
//!
//! # Schedule 係数
//!
//! 時間独立 frozen schedule `a_t = b_t = 0.5` (perf_cfm4_richardson の matvec_only /
//! single_lanczos mode と完全一致). H の compute pattern を bench 間で完全に揃える
//! ことで, Lanczos vs Chebyshev の差分を **アルゴリズム軸のみに帰着** させる.
//!
//! # 出力
//!
//! stderr に wall time / per-iter time / K_used 統計 / sink (DCE 防止) を出す.
//! stdout は perf の出力で汚さないよう空に保つ.

use std::env;
use std::time::Instant;

use _rust::bench_api::chebyshev_propagate;
use num_complex::Complex64;

fn main() {
    let args: Vec<String> = env::args().collect();
    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(18);
    let n_iters: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(100);
    let tol: f64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1e-10);

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

    // warmup (rayon pool 起動, page fault 解消, cache warm).
    // matvec_only と異なり 1 iter のコストが大きいので 3 回程度.
    for _ in 0..3 {
        let _ = chebyshev_propagate(&h_x, &h_p_diag, a_t, b_t, &psi, dt, tol, n);
    }

    let t0 = Instant::now();
    let mut k_total: usize = 0;
    let mut sink_acc = 0.0_f64;
    for _ in 0..n_iters {
        let (psi_new, k_used, _err_estimate) =
            chebyshev_propagate(&h_x, &h_p_diag, a_t, b_t, &psi, dt, tol, n);
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
