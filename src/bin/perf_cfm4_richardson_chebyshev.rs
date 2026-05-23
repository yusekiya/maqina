//! Phase B (issue #122): `cfm4_adaptive_richardson_chebyshev` の per-step
//! bottleneck を Linux `perf stat` の hardware counter で切り分けるための
//! pure-Rust 計測 binary (既存 `perf_cfm4_richardson` の Chebyshev variant).
//!
//! # 用途
//!
//! Phase A (#120 / PR #121) で時間独立 H 単体での `chebyshev_propagate` は
//! per-call 29 ms / 4.45× Lanczos 高速を実測 (`perf_chebyshev`). 本 binary は
//! その Chebyshev propagator を **CFM4:2 + step-doubling Richardson 構造**
//! に組み込んだ後の per-step wall を計測する. component 別 breakdown:
//!
//! | mode | 呼び出す計算 | 用途 |
//! |---|---|---|
//! | `full` (default) | `cfm4_step_chebyshev_with_richardson_estimate` 1 step | Richardson 1 step (6 chebyshev_propagate call) 全体 |
//! | `single_chebyshev` | `chebyshev_propagate` 1 call (frozen schedule) | Chebyshev layer 単独 (`perf_chebyshev` と同等; 同一 binary で counter set を揃えるため重複露出) |
//! | `matvec_only` | `apply_h_kinema` 1 call | matvec 単独 (`perf_apply_h` と同等; Chebyshev / Lanczos 共通の最下層) |
//!
//! 3 mode を **同じ counter set** で取って per-step → propagator → matvec の
//! 各層 wall % を実測 breakdown する. 既存 `perf_cfm4_richardson` の同名
//! mode (`full` / `single_lanczos` / `matvec_only`) と直接比較することで
//! "Chebyshev vs Lanczos の compute 効果差" を hardware counter まで掘れる
//! (IPC / L2 fill latency / Stalled cycles の差).
//!
//! Lanczos 版に存在した `gram_schmidt` mode は Chebyshev では原理的に存在
//! しない (3 項漸化が直交性を保証するため re-orthogonalization 不要; これが
//! Phase A の 4.45× speedup を生んだ要因の 1 つ).
//!
//! # ビルド
//!
//! ```bash
//! RUSTFLAGS="-C target-cpu=native" cargo build --release --bin perf_cfm4_richardson_chebyshev
//! ```
//!
//! `extension-module` feature は OFF (default features の `blas` / `rayon` /
//! `simd` のみ) なので pyo3 が libpython を静的リンクする (`perf_cfm4_richardson`
//! と同じ).
//!
//! # 計測例 (Linux AMD EPYC 7713P, Zen 3)
//!
//! ```bash
//! for mode in full single_chebyshev matvec_only; do
//!     RAYON_NUM_THREADS=64 perf stat \
//!         -e cycles,instructions,branch-misses \
//!         -e stalled-cycles-backend,stalled-cycles-frontend \
//!         -e cache-references,cache-misses \
//!         -e L1-dcache-loads,L1-dcache-load-misses \
//!         -e LLC-loads,LLC-load-misses \
//!         -e l2_request_g1.all_no_prefetch \
//!         -e l2_cache_req_stat.ic_dc_miss_in_l2 \
//!         -e l2_latency.l2_cycles_waiting_on_fills \
//!         -- ./target/release/perf_cfm4_richardson_chebyshev 18 100 $mode
//! done
//! ```
//!
//! # 引数
//!
//! `./perf_cfm4_richardson_chebyshev <N> <n_steps> [mode]`
//!
//! - `N`: TFIM サイト数. dim = 2^N. default = 18.
//! - `n_steps`: 計測 step 数. default = 100.
//! - `mode`: `full` (default) / `single_chebyshev` / `matvec_only`.
//!
//! # 出力
//!
//! stderr に wall time / per-iter time / K_used (avg) / sink (DCE 防止) を
//! 出力. stdout は perf の counter 出力を汚さないよう空に保つ.

use std::env;
use std::time::Instant;

use _rust::bench_api::{
    apply_h_kinema, cfm4_step_chebyshev_with_richardson_estimate, chebyshev_propagate,
};
use num_complex::Complex64;

#[derive(Clone, Copy, Debug)]
enum Mode {
    Full,
    SingleChebyshev,
    MatvecOnly,
}

impl Mode {
    fn parse(s: &str) -> Option<Self> {
        match s {
            "full" => Some(Mode::Full),
            "single_chebyshev" => Some(Mode::SingleChebyshev),
            "matvec_only" => Some(Mode::MatvecOnly),
            _ => None,
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            Mode::Full => "full",
            Mode::SingleChebyshev => "single_chebyshev",
            Mode::MatvecOnly => "matvec_only",
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(18);
    let n_steps: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(100);
    let mode = match args.get(3).map(|s| s.as_str()) {
        None => Mode::Full,
        Some(s) => Mode::parse(s).unwrap_or_else(|| {
            eprintln!("ERROR: unknown mode '{s}'. valid: full / single_chebyshev / matvec_only");
            std::process::exit(1);
        }),
    };

    let dim: usize = 1 << n;
    let dt: f64 = 1.0;
    // Chebyshev 切り捨て次数 K_used を決める tol. perf_chebyshev と揃える.
    let chebyshev_tol: f64 = 1e-10;
    let rayon_threads = std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "(auto)".to_string());

    eprintln!("== perf_cfm4_richardson_chebyshev ==");
    eprintln!("mode = {}", mode.as_str());
    eprintln!("n = {n}, dim = {dim}");
    eprintln!("n_steps = {n_steps}");
    eprintln!("dt = {dt}, chebyshev_tol = {chebyshev_tol:e}");
    eprintln!("RAYON_NUM_THREADS = {rayon_threads}");

    // 決定論的 seed. perf_cfm4_richardson と同じ XorShift で同じパターンの
    // 入力を作る (mode 間で input パターンが揃うので counter 比較が公平).
    let mut rng = XorShift64::new(0xB0BA_FEED_DEAD_FACE ^ (n as u64));
    let h_x: Vec<f64> = (0..n).map(|_| rng.signed_unit()).collect();
    let h_p_diag: Vec<f64> = (0..dim).map(|_| rng.signed_unit()).collect();
    let mut psi: Vec<Complex64> = (0..dim)
        .map(|_| Complex64::new(rng.signed_unit(), rng.signed_unit()))
        .collect();
    let nrm = psi.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
    if nrm > 0.0 {
        for z in psi.iter_mut() {
            *z /= nrm;
        }
    }

    // perf_cfm4_richardson と同じ mid-schedule 風の schedule 係数. 物理意味は
    // 問わず compute pattern を揃える目的.
    let a_s1_full = 0.45;
    let b_s1_full = 0.30;
    let a_s2_full = 0.40;
    let b_s2_full = 0.35;
    let a_s1_h1 = 0.48;
    let b_s1_h1 = 0.27;
    let a_s2_h1 = 0.46;
    let b_s2_h1 = 0.29;
    let a_s1_h2 = 0.43;
    let b_s1_h2 = 0.32;
    let a_s2_h2 = 0.41;
    let b_s2_h2 = 0.34;
    // single_chebyshev / matvec_only で使う time-frozen schedule.
    let a_t = 0.5_f64;
    let b_t = 0.5_f64;

    // Gershgorin 上下界の precompute. h_x / h_p_diag は loop 不変なので 1 度だけ.
    let h_x_abs_sum: f64 = h_x.iter().map(|x| x.abs()).sum();
    let h_p_min = h_p_diag.iter().cloned().fold(f64::INFINITY, f64::min);
    let h_p_max = h_p_diag.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let sink: f64;
    let elapsed_secs: f64;
    let extra_summary: Option<String>;

    match mode {
        Mode::Full => {
            // warmup.
            for _ in 0..3 {
                let _ = cfm4_step_chebyshev_with_richardson_estimate(
                    &mut psi,
                    &h_x,
                    &h_p_diag,
                    a_s1_full,
                    b_s1_full,
                    a_s2_full,
                    b_s2_full,
                    a_s1_h1,
                    b_s1_h1,
                    a_s2_h1,
                    b_s2_h1,
                    a_s1_h2,
                    b_s1_h2,
                    a_s2_h2,
                    b_s2_h2,
                    dt,
                    chebyshev_tol,
                    n,
                    true,
                    h_x_abs_sum,
                    h_p_min,
                    h_p_max,
                )
                .expect("Chebyshev Richardson step (warmup) failed");
            }

            let t0 = Instant::now();
            let mut k_used_total: usize = 0;
            for _ in 0..n_steps {
                let (_err, k_used, _err_cheb) = cfm4_step_chebyshev_with_richardson_estimate(
                    &mut psi,
                    &h_x,
                    &h_p_diag,
                    a_s1_full,
                    b_s1_full,
                    a_s2_full,
                    b_s2_full,
                    a_s1_h1,
                    b_s1_h1,
                    a_s2_h1,
                    b_s2_h1,
                    a_s1_h2,
                    b_s1_h2,
                    a_s2_h2,
                    b_s2_h2,
                    dt,
                    chebyshev_tol,
                    n,
                    true,
                    h_x_abs_sum,
                    h_p_min,
                    h_p_max,
                )
                .expect("Chebyshev Richardson step failed");
                k_used_total += k_used;
            }
            elapsed_secs = t0.elapsed().as_secs_f64();
            sink = psi.iter().take(8).map(|c| c.re + c.im).sum::<f64>()
                + (k_used_total as f64) * 1e-30;
            // Richardson step は 6 chebyshev_propagate call / step. k_used_total
            // は 6 個の chebyshev_propagate call の K_used の和を n_steps 回
            // 足したもの.
            extra_summary = Some(format!(
                "K_used (per chebyshev_propagate call, avg) ≈ {:.2}",
                (k_used_total as f64) / (n_steps as f64) / 6.0
            ));
        }
        Mode::SingleChebyshev => {
            // warmup.
            for _ in 0..3 {
                let _ = chebyshev_propagate(
                    &h_x,
                    &h_p_diag,
                    a_t,
                    b_t,
                    &psi,
                    dt,
                    chebyshev_tol,
                    n,
                    h_x_abs_sum,
                    h_p_min,
                    h_p_max,
                );
            }

            let t0 = Instant::now();
            let mut k_used_total: usize = 0;
            let mut sink_acc = 0.0_f64;
            for _ in 0..n_steps {
                let (psi_new, k_used, _err_estimate) = chebyshev_propagate(
                    &h_x,
                    &h_p_diag,
                    a_t,
                    b_t,
                    &psi,
                    dt,
                    chebyshev_tol,
                    n,
                    h_x_abs_sum,
                    h_p_min,
                    h_p_max,
                );
                k_used_total += k_used;
                sink_acc += psi_new.iter().take(8).map(|c| c.re + c.im).sum::<f64>();
            }
            elapsed_secs = t0.elapsed().as_secs_f64();
            sink = sink_acc + (k_used_total as f64) * 1e-30;
            extra_summary = Some(format!(
                "K_used (avg) ≈ {:.2}",
                (k_used_total as f64) / (n_steps as f64)
            ));
        }
        Mode::MatvecOnly => {
            let mut y = vec![Complex64::new(0.0, 0.0); dim];
            // warmup.
            for _ in 0..10 {
                apply_h_kinema(&psi, &mut y, &h_x, &h_p_diag, a_t, b_t, n);
            }

            let t0 = Instant::now();
            for _ in 0..n_steps {
                apply_h_kinema(&psi, &mut y, &h_x, &h_p_diag, a_t, b_t, n);
            }
            elapsed_secs = t0.elapsed().as_secs_f64();
            sink = y.iter().take(8).map(|c| c.re + c.im).sum();
            extra_summary = None;
        }
    }

    eprintln!("---");
    if let Some(s) = extra_summary {
        eprintln!("{s}");
    }
    eprintln!("total = {:.6} sec", elapsed_secs);
    eprintln!(
        "per-iter = {:.6} ms",
        elapsed_secs / (n_steps as f64) * 1000.0
    );
    eprintln!("sink (anti-DCE) = {sink}");
}

/// 軽量 xorshift64 PRNG (`src/bin/perf_cfm4_richardson.rs::XorShift64` の再掲).
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

    fn signed_unit(&mut self) -> f64 {
        let bits = self.next_u64();
        let normalized = (bits as f64) / (u64::MAX as f64);
        2.0 * normalized - 1.0
    }
}
