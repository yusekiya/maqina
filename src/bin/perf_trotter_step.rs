//! Phase 6 audit (issue #82): `trotter_step` の真の compute speedup を
//! `perf stat` の hardware counter で測るための pure-Rust 計測 binary.
//!
//! # 用途
//!
//! Python の `bench_block_fusion.py` は `trotter_step_py` (allocate-and-return)
//! を経由するため, dim · 16 B の `complex128` array を毎回 alloc/copy する
//! overhead が wall time に乗る (`docs/design/05-1-matvec.md` §5.1.4 参照).
//! `apply_h_py` ほど顕著ではないが (trotter_step は per-step compute が
//! 大きい), C3 (issue #64) の 4.01× speedup 主張のように compute speedup と
//! noise の切り分けが必要なケースで Python bench は信頼性が落ちる.
//!
//! 本 binary は `trotter_step` を N 回 in-place で呼ぶだけのミニマルな実行体
//! で, `perf stat` で hardware counter (IPC, cache miss, L2 fill wait, branch
//! miss) を採取することで micro-optimization の真の効果を切り分ける.
//!
//! # ビルド
//!
//! ```bash
//! RUSTFLAGS="-C target-cpu=native" cargo build --release --bin perf_trotter_step
//! ```
//!
//! `extension-module` feature は OFF (default features の `blas` / `rayon` /
//! `simd` のみ) なので pyo3 が libpython を静的リンクする (cargo test 経路と同じ).
//!
//! # 計測例 (AMD Zen 3 / EPYC, issue #79 / #82 で実用したセット)
//!
//! ```bash
//! # 基本: IPC + cache
//! RAYON_NUM_THREADS=64 perf stat \
//!     -e cycles,instructions,branch-misses \
//!     -e cache-references,cache-misses \
//!     -e L1-dcache-loads,L1-dcache-load-misses \
//!     -e dTLB-loads,dTLB-load-misses \
//!     -- ./target/release/perf_trotter_step 20 500
//!
//! # AMD Zen 3 専用: L2 fill latency / stall
//! RAYON_NUM_THREADS=64 perf stat \
//!     -e cycles,instructions,branch-misses \
//!     -e stalled-cycles-backend,stalled-cycles-frontend \
//!     -e l2_request_g1.all_no_prefetch,l2_cache_req_stat.ic_dc_miss_in_l2 \
//!     -e l2_latency.l2_cycles_waiting_on_fills \
//!     -- ./target/release/perf_trotter_step 20 500
//! ```
//!
//! # 引数
//!
//! `./perf_trotter_step <n> [iterations]`
//!
//! - `n`: TFIM サイト数. dim = 2^n. default = 20.
//! - `iterations`: trotter_step 呼出回数. default = 500. 実行時間が 5-10 秒に
//!   なるよう調整 (perf counter の統計的安定性確保). `apply_h` より
//!   per-iter cost が大きいので default は 500 (perf_apply_h の 1000 の半分).
//!
//! # 出力
//!
//! stderr に wall time / per-iter time / sink (DCE 防止) を出す. stdout は
//! perf の出力で汚さないよう空に保つ.

use std::env;
use std::time::Instant;

use _rust::bench_api::trotter_step;
use num_complex::Complex64;

fn main() {
    let args: Vec<String> = env::args().collect();
    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(20);
    let iterations: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(500);

    let dim: usize = 1 << n;
    let rayon_threads = std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "(auto)".to_string());

    eprintln!("== perf_trotter_step ==");
    eprintln!("n = {n}, dim = {dim}");
    eprintln!("iterations = {iterations}");
    eprintln!("RAYON_NUM_THREADS = {rayon_threads}");

    // 決定論的 seed で入力初期化 (bench 間で同一データ). perf_apply_h と同じ
    // 種に揃える必要はない (異なる kernel なので) が, run-to-run 再現性のため
    // 固定 seed を使う.
    let mut rng = XorShift64::new(0xCAFE_F00D_BEEF_DEAD ^ (n as u64));
    let h_x: Vec<f64> = (0..n).map(|_| rng.signed_unit()).collect();
    let h_p_diag: Vec<f64> = (0..dim).map(|_| rng.signed_unit()).collect();
    // trotter_step は psi を in-place で更新するので, 各 iter で初期化し直す
    // 必要はない (unitary なので ‖psi‖ は保たれる). 数値桁落ちは数百 step では
    // 起きないので per-iter cost を変えずに測れる.
    let mut psi: Vec<Complex64> = (0..dim)
        .map(|_| Complex64::new(rng.signed_unit(), rng.signed_unit()))
        .collect();
    // |psi| を 1 に正規化 (本番運用と同じスケール感. 桁落ち対策は不要).
    let nrm = psi.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
    if nrm > 0.0 {
        for z in psi.iter_mut() {
            *z /= nrm;
        }
    }
    let a_t = 0.5_f64;
    let b_t = 0.5_f64;
    // schedule 全体 (例えば τ=10) を iterations step で割った刻みを採用.
    // 物理的意味はないが trotter_step の compute pattern は dt に依存しない
    // (sin/cos の引数だけ変わる).
    let dt = 10.0_f64 / (iterations as f64);

    // warmup (rayon pool 起動, page fault 解消, cache warm).
    for _ in 0..10 {
        trotter_step(&mut psi, &h_x, &h_p_diag, a_t, b_t, dt, n);
    }

    let t0 = Instant::now();
    for _ in 0..iterations {
        trotter_step(&mut psi, &h_x, &h_p_diag, a_t, b_t, dt, n);
    }
    let elapsed = t0.elapsed();

    // DCE 防止: psi の先頭数要素を sink. compiler が trotter_step 呼出を
    // 削除しないことを保証する.
    let sink: f64 = psi.iter().take(8).map(|c| c.re + c.im).sum();

    eprintln!("---");
    eprintln!("total = {:.6} sec", elapsed.as_secs_f64());
    eprintln!(
        "per-iter = {:.6} ms",
        elapsed.as_secs_f64() / (iterations as f64) * 1000.0
    );
    eprintln!("sink (anti-DCE) = {sink}");
}

/// 軽量 xorshift64 PRNG. テスト用途のみ. `src/bin/perf_apply_h.rs::XorShift64`
/// と同実装を再掲 (依存追加を避けるため inline; binary 間で `mod` 共有しない
/// 方が cargo の dependency graph がシンプルになる).
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
