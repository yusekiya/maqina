//! Phase 6 audit (issue #90): `apply_single_mode_axis_i` の真の compute
//! 性能を `perf stat` の hardware counter で測るための pure-Rust 計測 binary.
//!
//! # 用途
//!
//! issue #71 fixup commit `578d050` で採用しかけて Revert された
//! 動的 chunk_size の **真の per-iter time / IPC** を perf binary で再評価
//! するために導入. 当時の bench (`benchmarks/bench_simd_scaling.py`) は
//! `apply_single_mode_axis_i_py` (allocate-and-return wrap) を経由しており
//! Python alloc/copy noise を被っていた可能性がある (`docs/design/05-1-matvec.md`
//! §5.1.4 で確立した「Python bench double-edged noise」). 本 binary は
//! Rust から直接 `apply_single_mode_axis_i` を in-place で呼ぶことで,
//! 以下を切り分けられる:
//!
//! - rayon chunk_size 戦略 (静的 vs 動的) の真の compute 効果
//! - SIMD path (i ∈ {0,1,2}) vs scalar path (i ≥ 3) の per-iter cost
//!
//! # ビルド
//!
//! ```bash
//! RUSTFLAGS="-C target-cpu=native" cargo build --release \
//!     --bin perf_apply_single_mode_axis_i
//! ```
//!
//! `extension-module` feature は OFF (default features の `blas` / `rayon` /
//! `simd` のみ) なので pyo3 が libpython を静的リンクする (cargo test 経路と同じ).
//!
//! # 計測例 (AMD Zen 3 / EPYC, issue #79 / #82 で実用したセット)
//!
//! ```bash
//! # 基本: IPC + cache (i=0 は SIMD path)
//! RAYON_NUM_THREADS=64 perf stat \
//!     -e cycles,instructions,branch-misses \
//!     -e cache-references,cache-misses \
//!     -e L1-dcache-loads,L1-dcache-load-misses \
//!     -e dTLB-loads,dTLB-load-misses \
//!     -- ./target/release/perf_apply_single_mode_axis_i 20 500 0
//!
//! # AMD Zen 3 専用: L2 fill latency / stall (issue #90 の核心メトリクス)
//! RAYON_NUM_THREADS=64 perf stat \
//!     -e cycles,instructions,branch-misses \
//!     -e stalled-cycles-backend,stalled-cycles-frontend \
//!     -e l2_request_g1.all_no_prefetch,l2_cache_req_stat.ic_dc_miss_in_l2 \
//!     -e l2_latency.l2_cycles_waiting_on_fills \
//!     -- ./target/release/perf_apply_single_mode_axis_i 20 500 0
//!
//! # scalar path (i=8): SIMD なし経路の baseline
//! RAYON_NUM_THREADS=64 perf stat \
//!     -e cycles,instructions,branch-misses \
//!     -- ./target/release/perf_apply_single_mode_axis_i 20 500 8
//! ```
//!
//! # 引数
//!
//! `./perf_apply_single_mode_axis_i <n> [iterations] [i]`
//!
//! - `n`: TFIM サイト数. dim = 2^n. default = 20.
//! - `iterations`: `apply_single_mode_axis_i` 呼出回数. default = 500.
//!   per-iter cost は `apply_h` の半分程度 (1 pass のみ) だが,
//!   trotter_step と同じ規模感に合わせて 500 を default にして実行時間を
//!   5-10 秒オーダーに保つ.
//! - `i`: 適用する axis (qubit index, `0 <= i < n`). default = 0
//!   (SIMD path). `i ∈ {0, 1, 2}` は SIMD path, `i ≥ 3` は scalar path.
//!   issue #90 では `i ∈ {0, 2, 8}` の比較を想定.
//!
//! # 出力
//!
//! stderr に wall time / per-iter time / sink (DCE 防止) を出す. stdout は
//! perf の出力で汚さないよう空に保つ.

use std::env;
use std::time::Instant;

use _rust::bench_api::apply_single_mode_axis_i;
use num_complex::Complex64;

fn main() {
    let args: Vec<String> = env::args().collect();
    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(20);
    let iterations: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(500);
    let i: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0);

    assert!(i < n, "i = {i} must be < n = {n}");

    let dim: usize = 1 << n;
    let rayon_threads = std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "(auto)".to_string());

    eprintln!("== perf_apply_single_mode_axis_i ==");
    eprintln!("n = {n}, dim = {dim}");
    eprintln!(
        "i = {i} ({})",
        if i <= 2 { "SIMD path" } else { "scalar path" }
    );
    eprintln!("iterations = {iterations}");
    eprintln!("RAYON_NUM_THREADS = {rayon_threads}");

    // 決定論的 seed で入力初期化 (bench 間で同一データ). perf_apply_h /
    // perf_trotter_step とは異なる kernel なので種を揃える必要はない.
    let mut rng = XorShift64::new(0xDECA_F00D_C0DE_F11E ^ (n as u64) ^ ((i as u64) << 32));
    // 2×2 ユニタリ. random だと unitary でないが per-iter 内容を変えない方が
    // 計測安定するので fixed unitary (任意の単純な回転) を採用.
    // R_x(θ=π/3) 相当: cos(π/6) I - i·sin(π/6) X.
    let c = 0.8660254037844387_f64; // cos(π/6)
    let s = 0.5_f64; // sin(π/6)
    let u: [Complex64; 4] = [
        Complex64::new(c, 0.0),
        Complex64::new(0.0, -s),
        Complex64::new(0.0, -s),
        Complex64::new(c, 0.0),
    ];
    // psi を in-place で更新するので各 iter で初期化し直す必要はない (上の u
    // は unitary なので ‖psi‖ は保たれる). 数値桁落ちは数百 step では起きない.
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

    // warmup (rayon pool 起動, page fault 解消, cache warm).
    for _ in 0..10 {
        apply_single_mode_axis_i(&mut psi, &u, i, n);
    }

    let t0 = Instant::now();
    for _ in 0..iterations {
        apply_single_mode_axis_i(&mut psi, &u, i, n);
    }
    let elapsed = t0.elapsed();

    // DCE 防止: psi の先頭数要素を sink. compiler が apply_single_mode_axis_i
    // 呼出を削除しないことを保証する.
    let sink: f64 = psi.iter().take(8).map(|c| c.re + c.im).sum();

    eprintln!("---");
    eprintln!("total = {:.6} sec", elapsed.as_secs_f64());
    eprintln!(
        "per-iter = {:.6} ms",
        elapsed.as_secs_f64() / (iterations as f64) * 1000.0
    );
    eprintln!("sink (anti-DCE) = {sink}");
}

/// 軽量 xorshift64 PRNG. テスト用途のみ.
/// `src/bin/perf_apply_h.rs::XorShift64` / `src/bin/perf_trotter_step.rs::XorShift64`
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
