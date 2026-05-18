# bench_per_step results (M2 / Trotter / Suzuki S_4 / CFM4:2)

## Machine info

- **timestamp_utc**: 2026-05-18 13:18:07 UTC
- **platform**: Linux-5.15.0-156-generic-x86_64-with-glibc2.35
- **machine**: x86_64
- **processor**: x86_64
- **python**: 3.13.3
- **numpy**: 2.4.4
- **rust_extension**: loaded
- **__has_blas__**: True
- **blas_pools**: openblas/libscipy_openblas threads=64; openblas/libopenblas threads=64
- **cpu_count**: 64
- **reference_wall_sec_total**: 30.773

## CLI arguments

- **n_values**: [4, 8, 12, 16]
- **methods**: ['m2', 'trotter', 'trotter_suzuki4', 'cfm4', 'cfm4_adaptive_richardson']
- **n_steps**: 50
- **m_values**: [24]
- **repeat**: 3
- **warmup**: 1
- **T**: 1.0
- **blas_threads**: None
- **results_dir**: benchmarks/results/20260518-221807-phase6-finalize
- **seed**: 20260512

## Summary (per-n × method × m)

| n | dim | method | m | per-step (sec) min | per-step (sec) median | states/sec (median) | trials |
|---|---|---|---|---|---|---|---|
| 4 | 16 | m2 | 24 | 1.104236e-05 | 1.116306e-05 | 1.433e+06 | 3 |
| 4 | 16 | trotter | 24 | 2.084300e-06 | 2.107024e-06 | 7.594e+06 | 3 |
| 4 | 16 | trotter_suzuki4 | 24 | 6.833971e-06 | 6.934032e-06 | 2.307e+06 | 3 |
| 4 | 16 | cfm4 | 24 | 1.738839e-05 | 1.742341e-05 | 9.183e+05 | 3 |
| 4 | 16 | cfm4_adaptive_richardson | 24 | 6.218607e-05 | 6.301623e-05 | 2.539e+05 | 3 |
| 8 | 256 | m2 | 24 | 3.248841e-05 | 3.271587e-05 | 7.825e+06 | 3 |
| 8 | 256 | trotter | 24 | 9.691119e-06 | 9.798557e-06 | 2.613e+07 | 3 |
| 8 | 256 | trotter_suzuki4 | 24 | 4.506789e-05 | 4.509747e-05 | 5.677e+06 | 3 |
| 8 | 256 | cfm4 | 24 | 5.692326e-05 | 5.708493e-05 | 4.485e+06 | 3 |
| 8 | 256 | cfm4_adaptive_richardson | 24 | 1.637940e-04 | 1.662244e-04 | 1.540e+06 | 3 |
| 12 | 4096 | m2 | 24 | 6.334843e-04 | 6.345568e-04 | 6.455e+06 | 3 |
| 12 | 4096 | trotter | 24 | 1.463269e-04 | 1.464917e-04 | 2.796e+07 | 3 |
| 12 | 4096 | trotter_suzuki4 | 24 | 7.117227e-04 | 7.252062e-04 | 5.648e+06 | 3 |
| 12 | 4096 | cfm4 | 24 | 1.090938e-03 | 1.095088e-03 | 3.740e+06 | 3 |
| 12 | 4096 | cfm4_adaptive_richardson | 24 | 2.950079e-03 | 2.967257e-03 | 1.380e+06 | 3 |
| 16 | 65536 | m2 | 24 | 1.397734e-02 | 1.421485e-02 | 4.610e+06 | 3 |
| 16 | 65536 | trotter | 24 | 2.475070e-03 | 2.803956e-03 | 2.337e+07 | 3 |
| 16 | 65536 | trotter_suzuki4 | 24 | 1.140434e-02 | 1.142749e-02 | 5.735e+06 | 3 |
| 16 | 65536 | cfm4 | 24 | 2.245819e-02 | 2.273216e-02 | 2.883e+06 | 3 |
| 16 | 65536 | cfm4_adaptive_richardson | 24 | 5.748580e-02 | 5.817639e-02 | 1.127e+06 | 3 |

## Cross-method per-step median (sec)

| n | dim | cfm4 | cfm4_adaptive_richardson | m2 | trotter | trotter_suzuki4 | m2 / cfm4 | m2 / cfm4_adaptive_richardson | m2 / trotter | m2 / trotter_suzuki4 |
|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 16 | 1.742341e-05 | 6.301623e-05 | 1.116306e-05 | 2.107024e-06 | 6.934032e-06 | 0.641 | 0.177 | 5.298 | 1.610 |
| 8 | 256 | 5.708493e-05 | 1.662244e-04 | 3.271587e-05 | 9.798557e-06 | 4.509747e-05 | 0.573 | 0.197 | 3.339 | 0.725 |
| 12 | 4096 | 1.095088e-03 | 2.967257e-03 | 6.345568e-04 | 1.464917e-04 | 7.252062e-04 | 0.579 | 0.214 | 4.332 | 0.875 |
| 16 | 65536 | 2.273216e-02 | 5.817639e-02 | 1.421485e-02 | 2.803956e-03 | 1.142749e-02 | 0.625 | 0.244 | 5.070 | 1.244 |

## Adaptive driver detail

| n | dim | method | m | n_steps_actual (median) | n_steps_actual (min/max) | final_err_vs_ref (median) | m_eff (median) | m_eff (max) | per_step (sec, median) | total_wall (sec, median) | reference_wall (sec) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 16 | cfm4_adaptive_richardson | 24 | 19.0 | 19/19 | 2.027689e-09 | 30.00 | 30 | 6.301623e-05 | 1.197308e-03 | 1.499e-02 |
| 8 | 256 | cfm4_adaptive_richardson | 24 | 27.0 | 27/27 | 2.173060e-09 | 32.00 | 32 | 1.662244e-04 | 4.488058e-03 | 6.531e-02 |
| 12 | 4096 | cfm4_adaptive_richardson | 24 | 31.0 | 31/31 | 2.209383e-09 | 32.00 | 32 | 2.967257e-03 | 9.198497e-02 | 1.401e+00 |
| 16 | 65536 | cfm4_adaptive_richardson | 24 | 36.0 | 36/36 | 1.797512e-09 | 32.00 | 32 | 5.817639e-02 | 2.094350e+00 | 2.929e+01 |
