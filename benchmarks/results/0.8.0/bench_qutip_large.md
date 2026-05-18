# bench_qutip_large.py

Work-precision diagram ベンチ: QuTiP ``sesolve`` vs kryanneal 各 method (issue #65 Phase 6 C4).

複数 **scenario** (T × dynamic range の組合せ) と複数 **n** に対し 各 solver の精度つまみを sweep, 共通 reference (QuTiP `tol=ref_tol`) に対する infidelity と wall time を 1 回ずつ測定. 各 (scenario, n) ごとに **infidelity 昇順 + Pareto 最適マーク (✓)** を付けた work-precision 表を出す.

## Machine info & bench params

- **timestamp_utc**: `2026-05-18T13:19:06.312616+00:00`
- **platform**: `Linux-5.15.0-156-generic-x86_64-with-glibc2.35`
- **machine**: `x86_64`
- **python_version**: `3.13.3`
- **numpy_version**: `2.4.4`
- **qutip_version**: `5.2.3`
- **rayon_threads**: `<unset>`
- **blas_threads_requested**: `<unset>`
- **ref_tol**: `1.0e-11`
- **has_blas**: `True`
- **has_rayon**: `True`
- **has_simd**: `True`
- **n_values**: per-scenario default (see Scenarios table below)
- **solvers**: `['qutip', 'm2', 'trotter', 'cfm4', 'cfm4_adaptive_richardson']`
- **m2 dt sweep**: `[0.001, 0.003, 0.01, 0.03, 0.1]` (dt_min=0.005)
- **trotter dt sweep**: `[0.001, 0.003, 0.01, 0.03, 0.1]` (dt_min=0.005)
- **cfm4 dt sweep**: `[0.005, 0.02, 0.05, 0.2, 0.5]` (dt_min=0.01)
- **cfm4_adaptive_richardson atol sweep**: `[0.001, 1e-05, 1e-07, 1e-09, 1e-11]`
- **cfm4_adaptive_richardson krylov_tol sweep**: `['auto']` (`auto` = `tol_step * 1e-3` 自動結合; Phase 7 / #93)
- **qutip tol sweep**: `[0.001, 1e-05, 1e-07, 1e-09, 1e-12]`

## Scenarios

| name | T | h_p_scale | h_x_scale | n_values |
|---|---|---|---|---|
| standard | 1 | 1 | 1 | 10,12 |
| long-T | 10000 | 1 | 1 | 8,10 |
| stiff | 1 | 10 | 1 | 10,12 |
| large-N | 1 | 1 | 1 | 12,14,16 |

## scenario = standard, n = 10 (reference: QuTiP tol=1.0e-11, wall=0.012s)

| Pareto | solver | knob | n_steps_eff | infidelity | wall (s) |
|---|---|---|---|---|---|
| ✓ | qutip | tol=1.0e-09 | - | <1e-16 | 0.0059 |
|  | qutip | tol=1.0e-12 | - | <1e-16 | 0.0104 |
|  | cfm4 | dt=0.02 | 50 | <1e-16 | 0.0131 |
|  | cfm4_adaptive_richardson | atol=1.0e-07 | 19 | <1e-16 | 0.0160 |
|  | cfm4_adaptive_richardson | atol=1.0e-09 | 45 | <1e-16 | 0.0356 |
|  | cfm4_adaptive_richardson | atol=1.0e-11 | 112 | <1e-16 | 0.0894 |
|  | cfm4 | dt=0.05 | 20 | 2.931e-14 | 0.0062 |
|  | cfm4_adaptive_richardson | atol=1.0e-05 | 8 | 9.837e-14 | 0.0062 |
| ✓ | qutip | tol=1.0e-07 | - | 5.462e-12 | 0.0040 |
| ✓ | trotter | dt=0.01 | 100 | 1.723e-10 | 0.0035 |
|  | m2 | dt=0.01 | 100 | 3.276e-10 | 0.0137 |
|  | cfm4_adaptive_richardson | atol=1.0e-03 | 4 | 3.461e-10 | 0.0037 |
| ✓ | qutip | tol=1.0e-05 | - | 8.831e-10 | 0.0033 |
| ✓ | cfm4 | dt=0.2 | 5 | 3.158e-09 | 0.0023 |
| ✓ | trotter | dt=0.03 | 33 | 1.450e-08 | 0.0012 |
|  | m2 | dt=0.03 | 33 | 2.763e-08 | 0.0053 |
| ✓ | trotter | dt=0.1 | 10 | 1.684e-06 | 0.0004 |
|  | m2 | dt=0.1 | 10 | 3.280e-06 | 0.0023 |
|  | cfm4 | dt=0.5 | 2 | 1.759e-05 | 0.0014 |
|  | qutip | tol=1.0e-03 | - | 5.097e-05 | 0.0019 |

## scenario = standard, n = 12 (reference: QuTiP tol=1.0e-11, wall=0.046s)

| Pareto | solver | knob | n_steps_eff | infidelity | wall (s) |
|---|---|---|---|---|---|
| ✓ | qutip | tol=1.0e-09 | - | <1e-16 | 0.0282 |
|  | qutip | tol=1.0e-12 | - | <1e-16 | 0.0513 |
|  | cfm4_adaptive_richardson | atol=1.0e-07 | 20 | <1e-16 | 0.0658 |
|  | cfm4_adaptive_richardson | atol=1.0e-09 | 47 | <1e-16 | 0.1592 |
|  | cfm4_adaptive_richardson | atol=1.0e-11 | 116 | <1e-16 | 0.3724 |
|  | cfm4 | dt=0.02 | 50 | 4.441e-16 | 0.0548 |
| ✓ | cfm4 | dt=0.05 | 20 | 3.930e-14 | 0.0257 |
| ✓ | cfm4_adaptive_richardson | atol=1.0e-05 | 8 | 1.099e-13 | 0.0254 |
| ✓ | qutip | tol=1.0e-07 | - | 5.667e-13 | 0.0186 |
| ✓ | trotter | dt=0.01 | 100 | 1.467e-10 | 0.0143 |
| ✓ | cfm4_adaptive_richardson | atol=1.0e-03 | 4 | 3.331e-10 | 0.0112 |
|  | m2 | dt=0.01 | 100 | 3.512e-10 | 0.0568 |
|  | qutip | tol=1.0e-05 | - | 6.277e-10 | 0.0125 |
| ✓ | cfm4 | dt=0.2 | 5 | 4.505e-09 | 0.0093 |
| ✓ | trotter | dt=0.03 | 33 | 1.234e-08 | 0.0048 |
|  | m2 | dt=0.03 | 33 | 2.962e-08 | 0.0234 |
| ✓ | trotter | dt=0.1 | 10 | 1.423e-06 | 0.0015 |
|  | m2 | dt=0.1 | 10 | 3.519e-06 | 0.0093 |
|  | qutip | tol=1.0e-03 | - | 1.509e-05 | 0.0066 |
|  | cfm4 | dt=0.5 | 2 | 2.579e-05 | 0.0053 |

## scenario = long-T, n = 8 (reference: QuTiP tol=1.0e-11, wall=9.771s)

| Pareto | solver | knob | n_steps_eff | infidelity | wall (s) |
|---|---|---|---|---|---|
| ✓ | m2 | dt=0.1 | 100000 | <1e-16 | 2.9784 |
|  | cfm4 | dt=0.2 | 50000 | <1e-16 | 3.0272 |
|  | m2 | dt=0.03 | 333333 | <1e-16 | 6.7064 |
|  | cfm4 | dt=0.05 | 200000 | <1e-16 | 7.9866 |
|  | cfm4_adaptive_richardson | atol=1.0e-09 | 64553 | <1e-16 | 8.5716 |
|  | cfm4 | dt=0.02 | 500000 | <1e-16 | 15.0853 |
|  | m2 | dt=0.01 | 1000000 | <1e-16 | 15.3438 |
|  | cfm4_adaptive_richardson | atol=1.0e-11 | 162578 | <1e-16 | 21.2764 |
|  | qutip | tol=1.0e-12 | - | 4.441e-16 | 14.6870 |
|  | qutip | tol=1.0e-09 | - | 1.243e-14 | 8.4412 |
|  | cfm4_adaptive_richardson | atol=1.0e-07 | 25284 | 6.255e-13 | 3.6182 |
|  | trotter | dt=0.01 | 1000000 | 6.221e-11 | 7.5373 |
|  | qutip | tol=1.0e-07 | - | 1.815e-10 | 4.6896 |
| ✓ | cfm4 | dt=0.5 | 20000 | 3.044e-09 | 2.1227 |
| ✓ | cfm4_adaptive_richardson | atol=1.0e-05 | 8679 | 4.194e-09 | 1.8800 |
|  | trotter | dt=0.03 | 333333 | 5.038e-09 | 2.5206 |
| ✓ | trotter | dt=0.1 | 100000 | 6.224e-07 | 0.7541 |
|  | cfm4_adaptive_richardson | atol=1.0e-03 | 1262 | 2.620e-05 | 1.9262 |
|  | qutip | tol=1.0e-03 | - | 5.432e-04 | 2.4168 |
|  | qutip | tol=1.0e-05 | - | 4.296e-02 | 2.3099 |

## scenario = long-T, n = 10 (reference: QuTiP tol=1.0e-11, wall=48.733s)

| Pareto | solver | knob | n_steps_eff | infidelity | wall (s) |
|---|---|---|---|---|---|
| ✓ | m2 | dt=0.1 | 100000 | <1e-16 | 15.5854 |
|  | cfm4 | dt=0.2 | 50000 | <1e-16 | 16.2396 |
|  | m2 | dt=0.03 | 333333 | <1e-16 | 40.1268 |
|  | cfm4 | dt=0.05 | 200000 | <1e-16 | 45.6081 |
|  | cfm4_adaptive_richardson | atol=1.0e-09 | 65498 | <1e-16 | 48.4420 |
|  | qutip | tol=1.0e-12 | - | <1e-16 | 55.4117 |
|  | cfm4 | dt=0.02 | 500000 | <1e-16 | 90.9712 |
|  | m2 | dt=0.01 | 1000000 | <1e-16 | 92.1502 |
|  | cfm4_adaptive_richardson | atol=1.0e-11 | 164972 | <1e-16 | 121.9542 |
|  | cfm4_adaptive_richardson | atol=1.0e-07 | 25636 | 8.453e-13 | 19.7232 |
|  | qutip | tol=1.0e-09 | - | 1.342e-12 | 29.9293 |
|  | trotter | dt=0.01 | 1000000 | 9.124e-12 | 29.2858 |
| ✓ | trotter | dt=0.03 | 333333 | 7.594e-10 | 9.7118 |
|  | qutip | tol=1.0e-07 | - | 2.723e-09 | 16.5003 |
| ✓ | cfm4 | dt=0.5 | 20000 | 3.319e-09 | 9.2372 |
| ✓ | cfm4_adaptive_richardson | atol=1.0e-05 | 8766 | 4.218e-09 | 9.1624 |
| ✓ | trotter | dt=0.1 | 100000 | 9.371e-08 | 2.9251 |
|  | qutip | tol=1.0e-05 | - | 1.709e-06 | 12.1323 |
|  | cfm4_adaptive_richardson | atol=1.0e-03 | 1258 | 2.613e-05 | 4.9486 |
|  | qutip | tol=1.0e-03 | - | 4.528e-04 | 9.5579 |

## scenario = stiff, n = 10 (reference: QuTiP tol=1.0e-11, wall=0.013s)

| Pareto | solver | knob | n_steps_eff | infidelity | wall (s) |
|---|---|---|---|---|---|
| ✓ | qutip | tol=1.0e-12 | - | <1e-16 | 0.0151 |
|  | cfm4 | dt=0.02 | 50 | <1e-16 | 0.0157 |
|  | cfm4_adaptive_richardson | atol=1.0e-09 | 83 | <1e-16 | 0.0824 |
|  | cfm4_adaptive_richardson | atol=1.0e-11 | 208 | <1e-16 | 0.1695 |
|  | cfm4_adaptive_richardson | atol=1.0e-07 | 33 | 4.441e-16 | 0.0280 |
| ✓ | qutip | tol=1.0e-09 | - | 1.332e-15 | 0.0087 |
|  | cfm4_adaptive_richardson | atol=1.0e-05 | 14 | 2.700e-13 | 0.0125 |
| ✓ | cfm4 | dt=0.05 | 20 | 3.689e-12 | 0.0082 |
| ✓ | qutip | tol=1.0e-07 | - | 1.060e-10 | 0.0057 |
|  | cfm4_adaptive_richardson | atol=1.0e-03 | 6 | 6.960e-10 | 0.0060 |
|  | m2 | dt=0.01 | 100 | 4.332e-09 | 0.0152 |
| ✓ | trotter | dt=0.01 | 100 | 2.047e-08 | 0.0036 |
|  | m2 | dt=0.03 | 33 | 3.661e-07 | 0.0067 |
| ✓ | cfm4 | dt=0.2 | 5 | 3.978e-07 | 0.0032 |
|  | qutip | tol=1.0e-05 | - | 4.409e-07 | 0.0037 |
| ✓ | trotter | dt=0.03 | 33 | 1.723e-06 | 0.0012 |
|  | m2 | dt=0.1 | 10 | 4.452e-05 | 0.0032 |
| ✓ | trotter | dt=0.1 | 10 | 2.002e-04 | 0.0004 |
|  | cfm4 | dt=0.5 | 2 | 6.340e-03 | 0.0020 |
|  | qutip | tol=1.0e-03 | - | 6.464e-03 | 0.0020 |

## scenario = stiff, n = 12 (reference: QuTiP tol=1.0e-11, wall=0.052s)

| Pareto | solver | knob | n_steps_eff | infidelity | wall (s) |
|---|---|---|---|---|---|
| ✓ | qutip | tol=1.0e-12 | - | <1e-16 | 0.0585 |
|  | cfm4_adaptive_richardson | atol=1.0e-07 | 34 | <1e-16 | 0.1166 |
|  | cfm4_adaptive_richardson | atol=1.0e-09 | 86 | <1e-16 | 0.2950 |
|  | cfm4_adaptive_richardson | atol=1.0e-11 | 214 | <1e-16 | 0.7156 |
| ✓ | qutip | tol=1.0e-09 | - | 1.776e-15 | 0.0349 |
|  | cfm4 | dt=0.02 | 50 | 5.773e-15 | 0.0646 |
|  | cfm4_adaptive_richardson | atol=1.0e-05 | 14 | 2.411e-13 | 0.0485 |
| ✓ | cfm4 | dt=0.05 | 20 | 4.668e-12 | 0.0320 |
| ✓ | qutip | tol=1.0e-07 | - | 2.631e-10 | 0.0220 |
| ✓ | cfm4_adaptive_richardson | atol=1.0e-03 | 6 | 4.755e-10 | 0.0190 |
|  | m2 | dt=0.01 | 100 | 4.212e-09 | 0.0662 |
| ✓ | trotter | dt=0.01 | 100 | 2.061e-08 | 0.0146 |
|  | m2 | dt=0.03 | 33 | 3.560e-07 | 0.0301 |
| ✓ | cfm4 | dt=0.2 | 5 | 5.249e-07 | 0.0122 |
|  | qutip | tol=1.0e-05 | - | 1.337e-06 | 0.0140 |
| ✓ | trotter | dt=0.03 | 33 | 1.734e-06 | 0.0049 |
|  | m2 | dt=0.1 | 10 | 4.343e-05 | 0.0130 |
| ✓ | trotter | dt=0.1 | 10 | 2.012e-04 | 0.0017 |
|  | qutip | tol=1.0e-03 | - | 1.155e-03 | 0.0078 |
|  | cfm4 | dt=0.5 | 2 | 8.226e-03 | 0.0075 |

## scenario = large-N, n = 12 (reference: QuTiP tol=1.0e-11, wall=0.046s)

| Pareto | solver | knob | n_steps_eff | infidelity | wall (s) |
|---|---|---|---|---|---|
| ✓ | qutip | tol=1.0e-09 | - | <1e-16 | 0.0296 |
|  | qutip | tol=1.0e-12 | - | <1e-16 | 0.0520 |
|  | cfm4_adaptive_richardson | atol=1.0e-07 | 20 | <1e-16 | 0.0665 |
|  | cfm4_adaptive_richardson | atol=1.0e-09 | 47 | <1e-16 | 0.1558 |
|  | cfm4_adaptive_richardson | atol=1.0e-11 | 116 | <1e-16 | 0.3704 |
|  | cfm4 | dt=0.02 | 50 | 4.441e-16 | 0.0593 |
| ✓ | cfm4 | dt=0.05 | 20 | 3.930e-14 | 0.0274 |
| ✓ | cfm4_adaptive_richardson | atol=1.0e-05 | 8 | 1.099e-13 | 0.0246 |
| ✓ | qutip | tol=1.0e-07 | - | 5.667e-13 | 0.0192 |
| ✓ | trotter | dt=0.01 | 100 | 1.467e-10 | 0.0145 |
| ✓ | cfm4_adaptive_richardson | atol=1.0e-03 | 4 | 3.331e-10 | 0.0113 |
|  | m2 | dt=0.01 | 100 | 3.512e-10 | 0.0595 |
|  | qutip | tol=1.0e-05 | - | 6.277e-10 | 0.0124 |
| ✓ | cfm4 | dt=0.2 | 5 | 4.505e-09 | 0.0098 |
| ✓ | trotter | dt=0.03 | 33 | 1.234e-08 | 0.0049 |
|  | m2 | dt=0.03 | 33 | 2.962e-08 | 0.0236 |
| ✓ | trotter | dt=0.1 | 10 | 1.423e-06 | 0.0015 |
|  | m2 | dt=0.1 | 10 | 3.519e-06 | 0.0095 |
|  | qutip | tol=1.0e-03 | - | 1.509e-05 | 0.0067 |
|  | cfm4 | dt=0.5 | 2 | 2.579e-05 | 0.0054 |

## scenario = large-N, n = 14 (reference: QuTiP tol=1.0e-11, wall=0.505s)

| Pareto | solver | knob | n_steps_eff | infidelity | wall (s) |
|---|---|---|---|---|---|
| ✓ | cfm4 | dt=0.02 | 50 | <1e-16 | 0.5131 |
|  | cfm4_adaptive_richardson | atol=1.0e-07 | 21 | <1e-16 | 0.5349 |
|  | cfm4_adaptive_richardson | atol=1.0e-09 | 51 | <1e-16 | 1.0480 |
|  | cfm4_adaptive_richardson | atol=1.0e-11 | 126 | <1e-16 | 2.3300 |
| ✓ | qutip | tol=1.0e-12 | - | 4.441e-16 | 0.2628 |
| ✓ | qutip | tol=1.0e-09 | - | 4.441e-15 | 0.1366 |
|  | cfm4 | dt=0.05 | 20 | 7.727e-14 | 0.3738 |
|  | cfm4_adaptive_richardson | atol=1.0e-05 | 9 | 1.539e-13 | 0.2989 |
| ✓ | qutip | tol=1.0e-07 | - | 4.133e-12 | 0.0958 |
|  | trotter | dt=0.01 | 100 | 1.541e-10 | 0.1638 |
|  | cfm4_adaptive_richardson | atol=1.0e-03 | 5 | 2.598e-10 | 0.2090 |
|  | m2 | dt=0.01 | 100 | 4.705e-10 | 0.5364 |
| ✓ | qutip | tol=1.0e-05 | - | 9.516e-10 | 0.0584 |
|  | cfm4 | dt=0.2 | 5 | 1.046e-08 | 0.2094 |
| ✓ | trotter | dt=0.03 | 33 | 1.294e-08 | 0.0558 |
|  | m2 | dt=0.03 | 33 | 3.969e-08 | 0.3057 |
| ✓ | trotter | dt=0.1 | 10 | 1.455e-06 | 0.0061 |
|  | m2 | dt=0.1 | 10 | 4.733e-06 | 0.2127 |
|  | qutip | tol=1.0e-03 | - | 5.765e-05 | 0.0312 |
|  | cfm4 | dt=0.5 | 2 | 6.267e-05 | 0.1863 |

## scenario = large-N, n = 16 (reference: QuTiP tol=1.0e-11, wall=1.164s)

| Pareto | solver | knob | n_steps_eff | infidelity | wall (s) |
|---|---|---|---|---|---|
| ✓ | cfm4 | dt=0.02 | 50 | <1e-16 | 1.0872 |
|  | qutip | tol=1.0e-12 | - | <1e-16 | 1.3208 |
|  | cfm4_adaptive_richardson | atol=1.0e-07 | 24 | <1e-16 | 1.4858 |
| ✓ | qutip | tol=1.0e-09 | - | 8.882e-16 | 0.6828 |
|  | cfm4_adaptive_richardson | atol=1.0e-09 | 59 | 2.665e-15 | 3.4363 |
|  | cfm4_adaptive_richardson | atol=1.0e-11 | 146 | 4.885e-15 | 8.3170 |
|  | cfm4_adaptive_richardson | atol=1.0e-05 | 10 | 1.239e-13 | 0.7189 |
| ✓ | cfm4 | dt=0.05 | 20 | 2.067e-13 | 0.6087 |
| ✓ | qutip | tol=1.0e-07 | - | 1.504e-11 | 0.4416 |
| ✓ | cfm4_adaptive_richardson | atol=1.0e-03 | 5 | 1.587e-10 | 0.3961 |
| ✓ | trotter | dt=0.01 | 100 | 1.885e-10 | 0.3376 |
|  | m2 | dt=0.01 | 100 | 6.433e-10 | 1.2525 |
| ✓ | trotter | dt=0.03 | 33 | 1.578e-08 | 0.0985 |
|  | cfm4 | dt=0.2 | 5 | 5.229e-08 | 0.3000 |
|  | m2 | dt=0.03 | 33 | 5.427e-08 | 0.5641 |
|  | qutip | tol=1.0e-05 | - | 5.475e-08 | 0.2908 |
| ✓ | trotter | dt=0.1 | 10 | 1.711e-06 | 0.0396 |
|  | m2 | dt=0.1 | 10 | 6.465e-06 | 0.3188 |
|  | qutip | tol=1.0e-03 | - | 1.607e-04 | 0.1671 |
|  | cfm4 | dt=0.5 | 2 | 1.907e-04 | 0.2364 |
