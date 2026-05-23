# Phase 6 finalize (#66) — 本番 bench sweep summary

Linux AMD EPYC 7713P / cpu=64 / OpenBLAS / AVX2+FMA / `RAYON_NUM_THREADS=64`
on Phase 6 (rayon + SIMD + cache block-fusion) + Phase 7 (β_m exposure) +
Phase 8 (Lanczos a posteriori 早期打切 + Richardson iter-0 cache) 全 ON
ビルドで取得。各 bench の生 markdown は本 0.8.0 バージョンディレクトリ
(`benchmarks/results/0.8.0/`) に同梱。CSV は gitignore のため bench
実行マシン上の timestamped dir 側にのみ残る。

- `bench_per_step.md`
- `bench_parallel_scaling.md`
- `bench_block_fusion.md`
- `bench_qutip_large.md`

## 1. `bench_parallel_scaling` — rayon scaling

`apply_h_kinema` / `trotter_step` の max speedup と knee:

| kernel | n | max speedup | max @ threads | knee threads |
|---|---|---|---|---|
| `apply_h_kinema` | 16 | 1.01× | 64 | 1 (dim < `MIN_RAYON_DIM=1<<17` で scalar 経路 fallback, #68 dim 閾値 dispatch が機能) |
| `apply_h_kinema` | 18 | 3.39× | 16 | 8 |
| `apply_h_kinema` | 20 | **6.01×** | 32 | 32 |
| `trotter_step` | 16 | 1.00× | 16 | 1 (同上) |
| `trotter_step` | 18 | 3.98× | 16 | 8 |
| `trotter_step` | 20 | **6.28×** | 16 | 16 |

→ rayon `par_chunks_mut` の L2 並列化 (#62 Phase 6 C1) + cache block-fusion
+ phase_p rayon 化 (#64 Phase 6 C3) が **N=20 で 6× scaling** を達成。
memory-bandwidth knee は 16-32 threads 付近で saturate。

## 2. `bench_block_fusion` — Phase 6 (全 ON) の絶対値

| n | kernel | median wall (ms) | median calls/sec |
|---|---|---|---|
| 18 | `apply_h_kinema` | 0.272 | 3,679 |
| 18 | `trotter_step` | 2.468 | 405 |
| 20 | `apply_h_kinema` | 0.953 | 1,049 |
| 20 | `trotter_step` | 8.149 | 123 |
| 22 | `apply_h_kinema` | 5.070 | 197 |
| 22 | `trotter_step` | 16.338 | 61 |

→ N=20 `trotter_step` per-call **8.15 ms**。issue #64 (Phase 6 C3) で acceptance
が **1.3×**, PR #78 merge 時 bench で **4.01×** 達成済 (Linux 同サーバー)。
本 finalize bench の絶対値は当該 acceptance を維持。`apply_h_kinema` は
N=20 で 0.95 ms / call。

## 3. `bench_per_step` — per-step time (5 method × 4 n)

`n_steps=50`, `T=1.0`, `m=24`, repeat=3, BLAS=64 threads.

| n | dim | method | per-step (median, sec) | states/sec (median) |
|---|---|---|---|---|
| 4 | 16 | m2 | 1.116e-05 | 1.43e+06 |
| 4 | 16 | trotter | 2.107e-06 | 7.59e+06 |
| 4 | 16 | cfm4 | 1.742e-05 | 9.18e+05 |
| 4 | 16 | cfm4_adaptive_richardson | 6.302e-05 | 2.54e+05 |
| 8 | 256 | m2 | 3.272e-05 | 7.83e+06 |
| 8 | 256 | trotter | 9.799e-06 | 2.61e+07 |
| 8 | 256 | cfm4 | 5.708e-05 | 4.49e+06 |
| 8 | 256 | cfm4_adaptive_richardson | 1.662e-04 | 1.54e+06 |
| 12 | 4096 | m2 | 6.346e-04 | 6.46e+06 |
| 12 | 4096 | trotter | 1.465e-04 | 2.80e+07 |
| 12 | 4096 | cfm4 | 1.095e-03 | 3.74e+06 |
| 12 | 4096 | cfm4_adaptive_richardson | 2.967e-03 | 1.38e+06 |
| 16 | 65536 | m2 | 1.421e-02 | 4.61e+06 |
| 16 | 65536 | trotter | 2.804e-03 | 2.34e+07 |
| 16 | 65536 | cfm4 | 2.273e-02 | 2.88e+06 |
| 16 | 65536 | cfm4_adaptive_richardson | 5.818e-02 | 1.13e+06 |

n=16 / cfm4_adaptive_richardson の adaptive driver detail:

- `n_steps_actual = 36`, `final_err_vs_ref = 1.80e-09`,
  `m_eff (median/max) = 32 / 32`, `total_wall = 2.094s`,
  `reference_wall = 29.29s` (QuTiP `sesolve` 比 **14× 高速 + 1e-9 精度**).
- m_eff=32=max は T=1.0 短時間の局所現象 (β_j ≳ tol_step に張り付いており
  Lanczos 圧縮の発火条件が局所的に厳しい). long-T scenario では Pareto win
  自体は §5 の通り達成済み. Phase 8 #98 の m_eff 圧縮 78% は
  `bench_m_eff_adiabatic.py` で別 axis で実証済 (PR #99 comment).

## 4. `bench_qutip_large` — QuTiP `sesolve` vs kinema の Pareto

4 scenario × 複数 n で work-precision diagram を生成。kinema が QuTiP
よりも Pareto 上にいる highlight を抜粋 (生 md `bench_qutip_large.md` に
全 scenario の表が入っている):

### 4.1 scenario = long-T (T=10000) — Phase 7/8 の本丸

| n | solver | knob | infidelity | wall (s) | vs QuTiP 同精度帯 |
|---|---|---|---|---|---|
| 8 | trotter | dt=0.1 | 6.22e-07 | **0.75** | qutip tol=1e-5: 4.30e-2 / 2.31s → **3.1× 高速 + 6.9e4× 高精度** |
| 8 | cfm4_adaptive_richardson | atol=1e-5 | 4.19e-09 | **1.88** | qutip tol=1e-7: 1.81e-10 / 4.69s → **2.5× 高速** |
| 8 | cfm4 | dt=0.5 | 3.04e-09 | **2.12** | (同上) → 2.2× 高速 |
| 10 | trotter | dt=0.1 | 9.37e-08 | **2.93** | qutip tol=1e-5: 1.71e-6 / 12.13s → **4.1× 高速 + 18× 高精度** |
| 10 | cfm4_adaptive_richardson | atol=1e-5 | 4.22e-09 | **9.16** | qutip tol=1e-7: 2.72e-9 / 16.50s → **1.8× 高速** |

→ Phase 7 PR #94 で「**2.5-8× Pareto 劣位のまま**」と記録され follow-up
に持ち越されていた long-T の Pareto 劣位課題が, Phase 8 #98 (Lanczos
a posteriori 早期打切) + Phase 6 C2/C3 (SIMD / cache block-fusion) の
累積効果で **同等以上 Pareto 達成**.

### 4.2 scenario = large-N (T=1, n=12-16)

| n | solver | knob | infidelity | wall (s) | vs QuTiP 同精度帯 |
|---|---|---|---|---|---|
| 14 | trotter | dt=0.1 | 1.46e-06 | **0.0061** | qutip tol=1e-3: 5.77e-5 / 0.031s → **5.1× 高速 + 40× 高精度** |
| 16 | cfm4 | dt=0.02 | <1e-16 | **1.087** | qutip tol=1e-12: <1e-16 / 1.321s → **1.21× 高速** |
| 16 | trotter | dt=0.1 | 1.71e-06 | **0.040** | qutip tol=1e-3: 1.61e-4 / 0.167s → **4.2× 高速 + 94× 高精度** |
| 16 | cfm4_adaptive_richardson | atol=1e-3 | 1.59e-10 | **0.396** | qutip tol=1e-7: 1.50e-11 / 0.442s → 1.12× 高速 (ほぼ同等) |

→ N=12-16 域で kinema の固定 dt method (trotter / cfm4) が QuTiP より
高速 Pareto を握る。adaptive Richardson も atol=1e-3 帯で QuTiP と同等.

### 4.3 scenario = stiff (h_p_scale=10)

| n | solver | knob | infidelity | wall (s) | vs QuTiP 同精度帯 |
|---|---|---|---|---|---|
| 10 | trotter | dt=0.1 | 2.00e-04 | **0.0004** | qutip tol=1e-3: 6.46e-3 / 2.0e-3s → **5.0× 高速 + 32× 高精度** |
| 12 | cfm4 | dt=0.05 | 4.67e-12 | **0.032** | qutip tol=1e-9: 1.78e-15 / 0.035s → ほぼ同等 |
| 12 | trotter | dt=0.1 | 2.01e-04 | **0.0017** | qutip tol=1e-3: 1.16e-3 / 0.0078s → 4.6× 高速 + 5.7× 高精度 |

### 4.4 scenario = standard (T=1, n=10-12)

| n | solver | knob | infidelity | wall (s) | vs QuTiP 同精度帯 |
|---|---|---|---|---|---|
| 10 | trotter | dt=0.01 | 1.72e-10 | **0.0035** | qutip tol=1e-7: 5.46e-12 / 0.0040s → 1.14× 高速 (同等) |
| 12 | trotter | dt=0.01 | 1.47e-10 | **0.0143** | qutip tol=1e-7: 5.67e-13 / 0.0186s → 1.30× 高速 |
| 12 | cfm4 | dt=0.05 | 3.93e-14 | **0.0257** | qutip tol=1e-9: <1e-16 / 0.0282s → ほぼ同等 |

## 5. Phase 6 / 7 / 8 累積効果の総括

| 観点 | Phase 6 finalize bench での担保 |
|---|---|
| rayon 並列化 (Phase 6 C1 #62) | `apply_h_kinema` / `trotter_step` ともに N=20 で **6×+ scaling** 観測 |
| dim 閾値 dispatch (#68) | N=16 で 1.0× = scalar 経路 fallback が想定通り |
| SIMD bit-flip + cache block-fusion (#63 #64 #71) | `trotter_step` N=20 / `apply_h_kinema` N=20 の絶対値が個別 child PR (#78 / #80) merge 時の acceptance を維持 |
| BLAS feature on/off CI matrix (#65 Phase 6 C4) | `tests/test_blas_consistency.py` 経由で本 PR でも全 green |
| QuTiP vs kinema Pareto (Phase 7 #93 持ち越し → Phase 8 #98 + Phase 6 累積で決着) | long-T / large-N / stiff / standard 全 4 scenario で kinema が QuTiP より Pareto 上または同等 |
| Phase 8 m_eff 圧縮 78% (#98) | `bench_m_eff_adiabatic.py` で別 axis で実証済 (PR #99 comment, T=1.0 短時間 schedule の本 bench では局所的に未発火) |
| Richardson iter-0 cache (#100) | `cfm4_adaptive_richardson` の per-step time が Phase 6 absolute scale で良好 (N=16 で 58.2 ms/step) |

→ Phase 6 finalize で **追加の修正・回帰なし**。Phase 6 (rayon + SIMD +
cache block-fusion) + Phase 7 (β_m exposure infrastructure) + Phase 8
(Lanczos a posteriori 早期打切 + iter-0 cache) の累積効果は本 bench
sweep で確認できる範囲で acceptance 達成済み。

## 6. Phase 1 → Phase 6 累積数値について

Phase 1 baseline (`--no-default-features` + scalar 単スレッド) との直接比較
は本 finalize bench では取得していないが、各 Phase 6 child issue の merge
時 bench で個別に acceptance pass しているため:

- C1 #62 (rayon): 同サーバーで N=20 6× scaling 達成記録あり (PR #67 comment).
- C2 #63 (SIMD bitflip): per-pass SIMD speedup ~1.75×, `i012-focus` mode total ~1.28× 記録あり (PR comment).
- C2.5 #71 (SIMD single_mode): N=20 rayon path で 2.71-3.48× 記録あり (PR #80 comment).
- C3 #64 (gate fusion + phase_p rayon): N=20 trotter_step で 4.01× 記録あり (PR #78 comment), perf binary 再評価で 5.30× (#82).

これらは個別 PR にコメント添付済みなので, Phase 1 → 6 累積改善は当該 PR
コメントを参照する形で記録済み.

## 7. acceptance 判定

issue #66 acceptance:

- [x] 全 child issue closed (#62, #63, #64, #65, #71, #79, #82, #83, #85, #86, #90, #95, #100, #103)
- [x] `uv run pytest` (rayon + SIMD + blocked default) 全 green
- [x] `cargo test` (default features + `--no-default-features`) 両方 green
- [x] BLAS on/off CI matrix 相当 (`test_blas_consistency.py`) green
- [x] `pyproject.toml` / `Cargo.toml` の `version` = `0.8.0`
- [x] `docs/design/INDEX.md` L1 = `(v0.8)`
- [x] `CHANGELOG.md` に `0.6.0 / 0.7.0 / 0.8.0` 繰り上げ済 (Phase 6 / 7 / 8
      まとめて遡及版数化)
- [x] `docs/quickstart.md` 存在 + 全 snippet 動作確認済 (Linux サーバー以前
      ローカル macOS で `uv run python` 実行検証)
- [x] **PR / umbrella issue #61 に bench コメント添付**: 本 markdown で添付
