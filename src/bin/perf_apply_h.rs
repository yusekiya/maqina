//! Phase 6 D (issue #79) follow-up: `apply_h_kryanneal` の真の bottleneck を
//! Linux `perf stat` の hardware counter で特定するための pure-Rust 計測 binary.
//!
//! # 用途
//!
//! Python の `bench_block_fusion.py` は per-call wall-time だけしか出さないため,
//! Phase D 実装で N=18 regression / N=20 acceptance 未達となった原因が
//! - DRAM bandwidth bound (Phase D の前提)
//! - L3 contention
//! - rayon barrier overhead
//! - hyperthread / scheduling 干渉
//!
//! のどれなのか切り分けられない. 本 binary は `apply_h_kryanneal` を N 回ループ
//! するだけのミニマルな実行体で, `perf stat` で hardware counter (cache miss,
//! DRAM controller throughput, stall reasons, IPC) を採取してボトルネックを
//! 切り分ける.
//!
//! # ビルド
//!
//! ```bash
//! RUSTFLAGS="-C target-cpu=native" cargo build --release --bin perf_apply_h
//! ```
//!
//! `extension-module` feature は OFF (default features の `blas` / `rayon` /
//! `simd` のみ) なので pyo3 が libpython を静的リンクする (cargo test 経路と同じ).
//!
//! # 計測例 (Intel Xeon)
//!
//! ```bash
//! # 基本的な cache/IPC 計測
//! RAYON_NUM_THREADS=64 perf stat \
//!     -e cycles,instructions,cache-references,cache-misses,LLC-loads,LLC-load-misses,dTLB-load-misses,branch-misses \
//!     -- ./target/release/perf_apply_h 20 1000
//!
//! # stall reason 切り分け (Intel 専用)
//! RAYON_NUM_THREADS=64 perf stat \
//!     -e cycle_activity.stalls_l1d_miss,cycle_activity.stalls_l2_miss,cycle_activity.stalls_l3_miss,cycle_activity.stalls_mem_any \
//!     -- ./target/release/perf_apply_h 20 1000
//!
//! # DRAM controller throughput (Intel uncore, root 推奨)
//! sudo RAYON_NUM_THREADS=64 perf stat -a \
//!     -e uncore_imc_0/cas_count_read/,uncore_imc_0/cas_count_write/ \
//!     -- ./target/release/perf_apply_h 20 1000
//! ```
//!
//! # 引数
//!
//! `./perf_apply_h <n> [iterations]`
//!
//! - `n`: TFIM サイト数. dim = 2^n. default = 20.
//! - `iterations`: matvec 呼出回数. default = 1000. 実行時間が 5-10 秒に
//!   なるよう調整 (perf counter の統計的安定性確保).
//!
//! # 出力
//!
//! stderr に wall time / per-iter time / sink (DCE 防止) を出す. stdout は
//! perf の出力で汚さないよう空に保つ.

use std::env;
use std::time::Instant;

use _rust::bench_api::apply_h_kryanneal;
use num_complex::Complex64;

fn main() {
    let args: Vec<String> = env::args().collect();
    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(20);
    let iterations: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1000);

    let dim: usize = 1 << n;
    let rayon_threads = std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "(auto)".to_string());

    eprintln!("== perf_apply_h ==");
    eprintln!("n = {n}, dim = {dim}");
    eprintln!("iterations = {iterations}");
    eprintln!("RAYON_NUM_THREADS = {rayon_threads}");

    // 決定論的 seed で入力初期化 (bench 間で同一データ).
    let mut rng = XorShift64::new(0xDEAD_BEEF_FEED_FACE ^ (n as u64));
    let h_x: Vec<f64> = (0..n).map(|_| rng.signed_unit()).collect();
    let h_p_diag: Vec<f64> = (0..dim).map(|_| rng.signed_unit()).collect();
    let v: Vec<Complex64> = (0..dim)
        .map(|_| Complex64::new(rng.signed_unit(), rng.signed_unit()))
        .collect();
    let mut y: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); dim];
    let a_t = 0.5_f64;
    let b_t = 0.5_f64;

    // warmup (rayon pool 起動, page fault 解消, cache warm).
    for _ in 0..10 {
        apply_h_kryanneal(&v, &mut y, &h_x, &h_p_diag, a_t, b_t, n);
    }

    let t0 = Instant::now();
    for _ in 0..iterations {
        apply_h_kryanneal(&v, &mut y, &h_x, &h_p_diag, a_t, b_t, n);
    }
    let elapsed = t0.elapsed();

    // DCE 防止: y の先頭数要素を sink. compiler が apply_h_kryanneal 呼出を
    // 削除しないことを保証する.
    let sink: f64 = y.iter().take(8).map(|c| c.re + c.im).sum();

    eprintln!("---");
    eprintln!("total = {:.6} sec", elapsed.as_secs_f64());
    eprintln!(
        "per-iter = {:.6} ms",
        elapsed.as_secs_f64() / (iterations as f64) * 1000.0
    );
    eprintln!("sink (anti-DCE) = {sink}");
}

/// 軽量 xorshift64 PRNG. テスト用途のみ. `src/matvec.rs::tests::Xor64` の
/// 等価実装を bin に持ち込む (依存追加を避けるため inline).
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
