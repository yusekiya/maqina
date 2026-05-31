---
paths:
  - "src/bin/**/*.rs"
  - "src/**/*.rs"
---

# perf 計測用 binary (Phase 6 D follow-up, issue #79 / #82 / #90 / #113 / #120)

`apply_h_kinema` / `trotter_step` / `apply_single_mode_axis_i` /
`cfm4_adaptive_richardson_krylov` / `chebyshev_propagate` の真の bottleneck
(DRAM bound / L3 contention / barrier / chunk_size 戦略の差 / Lanczos vs GS の
wall 比率, Lanczos vs Chebyshev のアルゴリズム軸 等のどれか) を Linux
`perf stat` で hardware counter から特定するための pure-Rust 計測 binary を
`src/bin/` に配置:

| binary | 対象 kernel | 主な用途 |
|---|---|---|
| `src/bin/perf_apply_h.rs` | `apply_h_kinema` (matvec) | #79 Phase D 試行で確立した DRAM/L2 latency 計測 |
| `src/bin/perf_trotter_step.rs` | `trotter_step` (Strang 2 次 Trotter 1 step) | #82 で C3 multi-qubit gate fusion + phase_p rayon 化の真の compute speedup 検証 |
| `src/bin/perf_apply_single_mode_axis_i.rs` | `apply_single_mode_axis_i` (Trotter per-axis 2×2 ユニタリ) | #90 で #71 fixup `578d050` (動的 chunk_size) 棄却を perf binary で再評価し dynamic を採用 (詳細は `docs/design/05-1-matvec.md` §5.1.4 末尾) |
| `src/bin/perf_cfm4_richardson.rs` | `cfm4_step_with_richardson_estimate` (Richardson 1 step = 6 Lanczos call) | #113 で Phase 9+ scoping のため component 別 wall % を実測 breakdown. `full` / `single_lanczos` / `matvec_only` / `gram_schmidt` の 4 mode を持ち, "step → Lanczos call → matvec / GS" の各層を同一 PMU セットで比較する |
| `src/bin/perf_chebyshev.rs` | `chebyshev_propagate` (時間独立 H, Chebyshev 3 項漸化) | #120 Phase A POC で Lanczos の V matrix cache stall を **アルゴリズム軸で bypass** する Chebyshev 経路の per-call wall を Linux で実測. `perf_cfm4_richardson 18 100 single_lanczos` (Lanczos baseline ~129 ms / IPC=0.78) と直接比較し, 判定 gate (≤ 50 ms で Phase B 進行 / 50-100 ms で設計再検討 / > 100 ms で中止) を判断する. 時間独立 frozen schedule `a_t = b_t = 0.5` で Lanczos baseline と input pattern を完全一致させる |
| `src/bin/perf_cfm4_richardson_chebyshev.rs` | `cfm4_step_chebyshev_with_richardson_estimate` (Chebyshev variant Richardson 1 step) | #122 Phase B で Chebyshev を CFM4 Magnus + step-doubling Richardson に統合した後の per-step wall + K_used を Linux で実測 breakdown. 3 mode (`full` / `single_chebyshev` / `matvec_only`) を持ち, 既存 `perf_cfm4_richardson` の同名 mode (`full` / `single_lanczos` / `matvec_only`) と直接比較することで Chebyshev vs Lanczos の compute 効果差を IPC / L2 fill latency / Stalled cycles まで掘れる. `gram_schmidt` mode は Chebyshev では原理的に存在しない (3 項漸化が直交保証, re-orthogonalization 不要) |

いずれも Python の `bench_*.py` が `*_py` (allocate-and-return) 経路の
alloc/copy overhead で wall-time を歪めるのを回避し,
Rust 側 micro-optimization の compute 効果だけを切り出す目的.

ビルド:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release --bin perf_apply_h
RUSTFLAGS="-C target-cpu=native" cargo build --release --bin perf_trotter_step
RUSTFLAGS="-C target-cpu=native" cargo build --release --bin perf_apply_single_mode_axis_i
RUSTFLAGS="-C target-cpu=native" cargo build --release --bin perf_cfm4_richardson
RUSTFLAGS="-C target-cpu=native" cargo build --release --bin perf_chebyshev
RUSTFLAGS="-C target-cpu=native" cargo build --release --bin perf_cfm4_richardson_chebyshev
```

対象関数は `pub fn` に上げ, `crate::bench_api` (`src/lib.rs`) で再 export
している (`apply_h_kinema`, `trotter_step`, `apply_single_mode_axis_i`,
`lanczos_propagate`, `cfm4_step_with_richardson_estimate`,
`chebyshev_propagate`; あと `gram_schmidt` mode が直接呼ぶ BLAS-1 primitive
として `dot_conj` / `axpy`).
Python 側 API (`_rust.apply_h_kinema_py` / `_rust.trotter_step_py` /
`_rust.apply_single_mode_axis_i_inplace_py` /
`_rust.cfm4_step_with_richardson_estimate_py` 等) には影響なし.
`chebyshev_propagate` は POC Phase A 段階では Python binding を持たず
(`_rust` に登録しない), perf binary 経由のみで使う.

計測例 (Linux, AMD EPYC で実証済み):

```bash
# 基本: IPC + cache
RAYON_NUM_THREADS=64 perf stat \
    -e cycles,instructions,branch-misses \
    -e cache-references,cache-misses \
    -e L1-dcache-loads,L1-dcache-load-misses \
    -e dTLB-loads,dTLB-load-misses \
    -- ./target/release/perf_apply_h 20 500

# AMD Zen 3 専用: L2 fill latency / stall (issue #79 で実用したセット)
RAYON_NUM_THREADS=64 perf stat \
    -e cycles,instructions,branch-misses \
    -e stalled-cycles-backend,stalled-cycles-frontend \
    -e l2_request_g1.all_no_prefetch,l2_cache_req_stat.ic_dc_miss_in_l2 \
    -e l2_latency.l2_cycles_waiting_on_fills \
    -- ./target/release/perf_apply_h 20 500

# trotter_step (issue #82 C3 audit). per-iter cost が大きいので iter 数は
# default 500 (perf_apply_h の 1000 の半分).
RAYON_NUM_THREADS=64 perf stat \
    -e cycles,instructions,branch-misses \
    -e stalled-cycles-backend,stalled-cycles-frontend \
    -e l2_request_g1.all_no_prefetch,l2_cache_req_stat.ic_dc_miss_in_l2 \
    -e l2_latency.l2_cycles_waiting_on_fills \
    -- ./target/release/perf_trotter_step 20 500

# apply_single_mode_axis_i (issue #90 C2.5 chunk_size audit). 第 3 引数で
# axis i を指定 (default 0 = SIMD path). i ∈ {0,1,2} で SIMD path,
# i >= 3 で scalar path.
RAYON_NUM_THREADS=64 perf stat \
    -e cycles,instructions,branch-misses \
    -e stalled-cycles-backend,stalled-cycles-frontend \
    -e l2_request_g1.all_no_prefetch,l2_cache_req_stat.ic_dc_miss_in_l2 \
    -e l2_latency.l2_cycles_waiting_on_fills \
    -- ./target/release/perf_apply_single_mode_axis_i 20 500 0

# cfm4_step_with_richardson_estimate (issue #113). 第 3 引数で mode 切替:
# full (default) / single_lanczos / matvec_only / gram_schmidt. 4 mode 全部
# 同じ counter セットで取って "step → Lanczos call → matvec / GS" 各層の
# wall % を実測 breakdown する.
for mode in full single_lanczos matvec_only gram_schmidt; do
    RAYON_NUM_THREADS=64 perf stat \
        -e cycles,instructions,branch-misses \
        -e stalled-cycles-backend,stalled-cycles-frontend \
        -e l2_request_g1.all_no_prefetch,l2_cache_req_stat.ic_dc_miss_in_l2 \
        -e l2_latency.l2_cycles_waiting_on_fills \
        -- ./target/release/perf_cfm4_richardson 18 100 $mode
done

# chebyshev_propagate (issue #120 POC). 第 3 引数で tol 切替 (default 1e-10).
# perf_cfm4_richardson 18 100 single_lanczos と直接比較する想定なので n_iters=100
# default. K_used 平均 (stderr) で Chebyshev 切り捨て次数の実測値も同時に取れる.
RAYON_NUM_THREADS=64 perf stat \
    -e cycles,instructions,branch-misses \
    -e stalled-cycles-backend,stalled-cycles-frontend \
    -e l2_request_g1.all_no_prefetch,l2_cache_req_stat.ic_dc_miss_in_l2 \
    -e l2_latency.l2_cycles_waiting_on_fills \
    -- ./target/release/perf_chebyshev 18 100
```

binary は stderr に wall time / per-iter time / sink (DCE 防止) を出し,
stdout は空に保つ (perf の出力を汚さない).

比較対象の build を識別するときは `cargo build --target-dir target-<tag>`
で出力先を分ける (#79 で確立した方法論). 例: `RAYON_CHUNK_MAX` 値違いを
同時に持ちたい場合は `target-rayon14` / `target-rayon13` のように分離する.
