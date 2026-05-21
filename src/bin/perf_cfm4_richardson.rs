//! Phase 6 D follow-up (issue #113): `cfm4_adaptive_richardson` の per-step
//! bottleneck を Linux `perf stat` の hardware counter で切り分けるための
//! pure-Rust 計測 binary.
//!
//! # 用途
//!
//! 既存 perf binary は単一 kernel 粒度:
//!   - `perf_apply_h` (#79): `apply_h_kryanneal` 単独
//!   - `perf_trotter_step` (#82): `trotter_step` 単独
//!   - `perf_apply_single_mode_axis_i` (#90): `apply_single_mode_axis_i` 単独
//!
//! 本 binary は **Richardson 1 step (6 Lanczos call / step) 全体** の hardware
//! counter を取り, mode 切替で "step → Lanczos call → matvec / Gram-Schmidt" の
//! 各層を **同一 PMU セット** で計測することで, **component 別 wall % の実測
//! breakdown** を得る. Phase 9+ で取り組む最適化軸 (mixed precision / V tiling /
//! Python driver Rust 移植 等) の優先度を実測で確定するための tooling.
//!
//! PR #106 (README figure pipeline) で N=18 / T=10^4 の per-step 0.88s を観測
//! したが, "何が dominant か" は推測ベースに留まっていた. 本 binary でその
//! component 別 wall % を実測する.
//!
//! # ビルド
//!
//! ```bash
//! RUSTFLAGS="-C target-cpu=native" cargo build --release --bin perf_cfm4_richardson
//! ```
//!
//! `extension-module` feature は OFF (default features の `blas` / `rayon` /
//! `simd` のみ) なので pyo3 が libpython を静的リンクする (cargo test 経路と同じ).
//!
//! # 計測例 (Linux AMD EPYC 7713P, Zen 3)
//!
//! ```bash
//! # 基本: IPC + cache + L2 fill latency (4 mode 全部で同じ counter セット)
//! for mode in full single_lanczos matvec_only gram_schmidt; do
//!     RAYON_NUM_THREADS=64 perf stat \
//!         -e cycles,instructions,branch-misses \
//!         -e stalled-cycles-backend,stalled-cycles-frontend \
//!         -e cache-references,cache-misses \
//!         -e L1-dcache-loads,L1-dcache-load-misses \
//!         -e LLC-loads,LLC-load-misses \
//!         -e l2_request_g1.all_no_prefetch \
//!         -e l2_cache_req_stat.ic_dc_miss_in_l2 \
//!         -e l2_latency.l2_cycles_waiting_on_fills \
//!         -- ./target/release/perf_cfm4_richardson 18 100 $mode
//! done
//! ```
//!
//! 4 mode 間で `cycles` / IPC / cache-miss / L2 fill latency を比較することで,
//! "どの層が dominant か" を判定できる.
//!
//! # 引数
//!
//! `./perf_cfm4_richardson <N> <n_steps> [mode]`
//!
//! - `N`: TFIM サイト数. dim = 2^N. default = 18 (本番想定; PR #106 N=18/T=10^4).
//! - `n_steps`: 計測 step 数 (各 mode で何回ループするか). default = 100.
//! - `mode`: `full` (default) / `single_lanczos` / `matvec_only` / `gram_schmidt`.
//!
//! | mode | 呼び出す計算 | 1 iter のコスト目安 |
//! |---|---|---|
//! | `full` | `cfm4_step_with_richardson_estimate` 1 step (= 6 Lanczos call) | Richardson 1 step (本番 N=18 で ~0.88s) |
//! | `single_lanczos` | `lanczos_propagate` 1 call (m_max=24) | 1 Lanczos call (full の ~1/6) |
//! | `matvec_only` | `apply_h_kryanneal` 1 call | matvec 単独 (perf_apply_h と同じパターン) |
//! | `gram_schmidt` | Lanczos 1 call 相当の 2-pass GS ループ | re-orthogonalization 単独 (matvec を抜いた残り) |
//!
//! # Schedule 係数
//!
//! `full` mode の Richardson 推定子は 12 個の `(a_s*, b_s*)` 係数を取る. 物理
//! 意味は無視して "stage 間で僅かに違う値" を mid-schedule 風に与える (compute
//! pattern は係数値そのものに依存せず, m_eff 挙動だけが krylov_tol との関係で
//! 変わる). 本 binary では `krylov_tol = 1e-10` (adaptive auto-resolve の典型値)
//! を採用し m_eff 圧縮が起きる条件で計測する.
//!
//! # 出力
//!
//! stderr に wall time / per-iter time / sink (DCE 防止) を出す. stdout は perf
//! の出力で汚さないよう空に保つ.

use std::env;
use std::time::Instant;

use _rust::bench_api::{
    apply_h_kryanneal, axpy, cfm4_step_with_richardson_estimate, dot_conj, lanczos_propagate,
};
use num_complex::Complex64;

#[derive(Clone, Copy, Debug)]
enum Mode {
    Full,
    SingleLanczos,
    MatvecOnly,
    GramSchmidt,
}

impl Mode {
    fn parse(s: &str) -> Option<Self> {
        match s {
            "full" => Some(Mode::Full),
            "single_lanczos" => Some(Mode::SingleLanczos),
            "matvec_only" => Some(Mode::MatvecOnly),
            "gram_schmidt" => Some(Mode::GramSchmidt),
            _ => None,
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            Mode::Full => "full",
            Mode::SingleLanczos => "single_lanczos",
            Mode::MatvecOnly => "matvec_only",
            Mode::GramSchmidt => "gram_schmidt",
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
            eprintln!(
                "ERROR: unknown mode '{s}'. valid: full / single_lanczos / matvec_only / gram_schmidt"
            );
            std::process::exit(1);
        }),
    };

    let dim: usize = 1 << n;
    let m_max: usize = 24;
    let dt: f64 = 1.0;
    let krylov_tol: f64 = 1e-10;
    let rayon_threads = std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "(auto)".to_string());

    eprintln!("== perf_cfm4_richardson ==");
    eprintln!("mode = {}", mode.as_str());
    eprintln!("n = {n}, dim = {dim}");
    eprintln!("n_steps = {n_steps}");
    eprintln!("m_max = {m_max}, dt = {dt}, krylov_tol = {krylov_tol:e}");
    eprintln!("RAYON_NUM_THREADS = {rayon_threads}");

    // 決定論的 seed で入力初期化. perf_apply_h と異なる seed を用意して
    // run-to-run 再現性は確保しつつ既存 binary との data 衝突を避ける.
    let mut rng = XorShift64::new(0xB0BA_FEED_DEAD_FACE ^ (n as u64));
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

    // mid-schedule 風 (a + b ≈ 0.75 帯) の synthetic schedule 係数. stage 間で
    // 僅かに違う値を使うことで full / h1 / h2 の Lanczos が完全に同じ ψ を
    // 返さない (Richardson err ≠ 0) ように仕込む. 物理意味は問わず compute
    // pattern の現実性を確保するだけ.
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
    // single_lanczos / matvec_only で使う time-frozen schedule.
    let a_t = 0.5_f64;
    let b_t = 0.5_f64;

    // 各 mode で:
    //   1. mode 固有の setup (V 行列確保等) を t0 前に済ませる
    //   2. warmup を t0 前に済ませる (rayon pool 起動 / page fault 解消 / cache warm)
    //   3. t0 を取って n_steps 回の本ループを計測
    //   4. sink を計算して anti-DCE
    let sink: f64;
    let elapsed_secs: f64;
    let extra_summary: Option<String>;

    match mode {
        Mode::Full => {
            // warmup: 3 回程度 (per-iter cost が大きいので少なめ).
            for _ in 0..3 {
                let _ = cfm4_step_with_richardson_estimate(
                    &mut psi, &h_x, &h_p_diag, a_s1_full, b_s1_full, a_s2_full, b_s2_full, a_s1_h1,
                    b_s1_h1, a_s2_h1, b_s2_h1, a_s1_h2, b_s1_h2, a_s2_h2, b_s2_h2, dt, m_max,
                    krylov_tol, n, true,
                )
                .expect("Richardson step (warmup) failed");
            }

            let t0 = Instant::now();
            let mut m_eff_total: usize = 0;
            for _ in 0..n_steps {
                let (_err, m_eff, _err_lanczos) = cfm4_step_with_richardson_estimate(
                    &mut psi, &h_x, &h_p_diag, a_s1_full, b_s1_full, a_s2_full, b_s2_full, a_s1_h1,
                    b_s1_h1, a_s2_h1, b_s2_h1, a_s1_h2, b_s1_h2, a_s2_h2, b_s2_h2, dt, m_max,
                    krylov_tol, n, true,
                )
                .expect("Richardson step failed");
                m_eff_total += m_eff;
            }
            elapsed_secs = t0.elapsed().as_secs_f64();
            sink =
                psi.iter().take(8).map(|c| c.re + c.im).sum::<f64>() + (m_eff_total as f64) * 1e-30;
            // Richardson step は 6 Lanczos call / step. m_eff_total は 6 個の
            // Lanczos call の m_eff の和 (cfm4.rs の m_eff_sum 規約) を n_steps
            // 回足したものなので, "1 Lanczos call の平均 m_eff" は
            // m_eff_total / n_steps / 6.
            extra_summary = Some(format!(
                "m_eff (per Lanczos call, avg) ≈ {:.2}",
                (m_eff_total as f64) / (n_steps as f64) / 6.0
            ));
        }
        Mode::SingleLanczos => {
            // warmup. matvec closure を毎回構築する (per-iter overhead 無視可).
            for _ in 0..3 {
                let matvec = |v_in: &[Complex64], y_out: &mut [Complex64]| {
                    apply_h_kryanneal(v_in, y_out, &h_x, &h_p_diag, a_t, b_t, n);
                };
                let _ = lanczos_propagate(matvec, &psi, dt, m_max, krylov_tol)
                    .expect("Lanczos (warmup) failed");
            }

            let t0 = Instant::now();
            let mut m_eff_total: usize = 0;
            let mut sink_acc = 0.0_f64;
            for _ in 0..n_steps {
                let matvec = |v_in: &[Complex64], y_out: &mut [Complex64]| {
                    apply_h_kryanneal(v_in, y_out, &h_x, &h_p_diag, a_t, b_t, n);
                };
                let (psi_new, m_eff, _beta_m, _c_m_abs) =
                    lanczos_propagate(matvec, &psi, dt, m_max, krylov_tol).expect("Lanczos failed");
                m_eff_total += m_eff;
                // 出口 ψ_new の先頭数要素を sink に畳む (DCE 防止).
                sink_acc += psi_new.iter().take(8).map(|c| c.re + c.im).sum::<f64>();
            }
            elapsed_secs = t0.elapsed().as_secs_f64();
            sink = sink_acc + (m_eff_total as f64) * 1e-30;
            extra_summary = Some(format!(
                "m_eff (avg) ≈ {:.2}",
                (m_eff_total as f64) / (n_steps as f64)
            ));
        }
        Mode::MatvecOnly => {
            let mut y = vec![Complex64::new(0.0, 0.0); dim];
            // warmup.
            for _ in 0..10 {
                apply_h_kryanneal(&psi, &mut y, &h_x, &h_p_diag, a_t, b_t, n);
            }

            let t0 = Instant::now();
            for _ in 0..n_steps {
                apply_h_kryanneal(&psi, &mut y, &h_x, &h_p_diag, a_t, b_t, n);
            }
            elapsed_secs = t0.elapsed().as_secs_f64();
            sink = y.iter().take(8).map(|c| c.re + c.im).sum();
            extra_summary = None;
        }
        Mode::GramSchmidt => {
            // V (dim × m_max, column-major) を random で構築後 orthonormalize.
            // 実 Lanczos では V は構築時に orthonormal なので, perf binary でも
            // それを再現する必要がある (random V を直接使うと 2-pass GS が
            // 非直交基底に対して leak/発散し w が NaN になる).
            //
            // dim · m_max · 16 B = 2^N · 24 · 16 B のメモリを確保する点に注意.
            // N=18 で ~100 MB, N=20 で ~400 MB. perf 計測の "L2 fill latency"
            // 文脈で V は L2/L3 spill するサイズになる (実 Lanczos と同じ).
            let mut v_mat: Vec<Complex64> = (0..dim * m_max)
                .map(|_| Complex64::new(rng.signed_unit(), rng.signed_unit()))
                .collect();
            // Modified Gram-Schmidt で V を orthonormalize (一度きり, 計測外).
            for k in 0..m_max {
                // v_k に対して v_0..v_{k-1} を直交化.
                for j in 0..k {
                    let (left, right) = v_mat.split_at_mut((j + 1) * dim);
                    let v_j = &left[j * dim..(j + 1) * dim];
                    let v_k = &mut right[(k - j - 1) * dim..(k - j) * dim];
                    let proj = dot_conj(v_j, v_k);
                    axpy(-proj, v_j, v_k);
                }
                // v_k を正規化.
                let v_k = &mut v_mat[k * dim..(k + 1) * dim];
                let nrm = dot_conj(v_k, v_k).re.sqrt();
                let scale = if nrm > 0.0 { 1.0 / nrm } else { 0.0 };
                for c in v_k.iter_mut() {
                    *c *= scale;
                }
            }
            let w_init: Vec<Complex64> = (0..dim)
                .map(|_| Complex64::new(rng.signed_unit(), rng.signed_unit()))
                .collect();
            let mut w = vec![Complex64::new(0.0, 0.0); dim];

            // warmup (cache warm + page fault 解消).
            for _ in 0..3 {
                w.copy_from_slice(&w_init);
                for k in 0..m_max {
                    for _pass in 0..2 {
                        for j in 0..=k {
                            let v_j = &v_mat[j * dim..(j + 1) * dim];
                            let proj = dot_conj(v_j, &w);
                            axpy(-proj, v_j, &mut w);
                        }
                    }
                }
            }

            let t0 = Instant::now();
            let mut sink_acc = 0.0_f64;
            for _ in 0..n_steps {
                w.copy_from_slice(&w_init);
                // Lanczos 1 call 相当の 2-pass GS ループ. これは
                // `src/krylov.rs::lanczos_propagate` 内の "Full
                // re-orthogonalization (2-pass Gram-Schmidt)" ブロック
                // (k=0..m-1 で計 2·Σ(k+1) = m(m+1) BLAS-1 ops) と完全同型.
                // matvec / α_k 計算 / β_k 計算 / tridiag 固有分解 / 終端 gemv
                // は含まない.
                for k in 0..m_max {
                    for _pass in 0..2 {
                        for j in 0..=k {
                            let v_j = &v_mat[j * dim..(j + 1) * dim];
                            let proj = dot_conj(v_j, &w);
                            axpy(-proj, v_j, &mut w);
                        }
                    }
                }
                sink_acc += w.iter().take(8).map(|c| c.re + c.im).sum::<f64>();
            }
            elapsed_secs = t0.elapsed().as_secs_f64();
            sink = sink_acc;
            extra_summary = Some(format!(
                "V size = dim × m_max = {} × {} ({} MB)",
                dim,
                m_max,
                dim * m_max * 16 / (1 << 20),
            ));
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
