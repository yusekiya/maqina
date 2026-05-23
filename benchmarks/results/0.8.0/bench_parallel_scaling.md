# bench_parallel_scaling

issue #62 (Phase 6 C1): rayon parallel scaling sweep.

## Machine info

- **platform**: Linux-5.15.0-156-generic-x86_64-with-glibc2.35
- **machine**: x86_64
- **processor**: x86_64
- **python_version**: 3.13.3
- **numpy_version**: 2.4.4
- **cpu_count_logical**: 64
- **timestamp_utc**: 2026-05-18T13:19:04.794073+00:00
- **kinema_version**: unknown
- **has_blas**: True

## apply_h_kinema

| n \ threads | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| 16 | 0.896 ms | 0.888 ms | 0.899 ms | 0.904 ms | 0.890 ms | 0.887 ms | 0.887 ms |
| 18 | 3.846 ms | 1.940 ms | 1.501 ms | 1.165 ms | 1.135 ms | 1.719 ms | 3.456 ms |
| 20 | 17.313 ms | 8.740 ms | 4.861 ms | 3.451 ms | 4.246 ms | 2.878 ms | 4.337 ms |

### apply_h_kinema — speedup vs threads=1 (median)

| n \ threads | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| 16 | 1.00× | 1.01× | 1.00× | 0.99× | 1.01× | 1.01× | 1.01× |
| 18 | 1.00× | 1.98× | 2.56× | 3.30× | 3.39× | 2.24× | 1.11× |
| 20 | 1.00× | 1.98× | 3.56× | 5.02× | 4.08× | 6.01× | 3.99× |

### apply_h_kinema — knee (memory-bandwidth saturation point, smallest threads achieving ≥ 95% of max speedup)

| n | max speedup | max @ threads | knee threads | speedup at knee |
|---|---|---|---|---|
| 16 | 1.01× | 64 | 1 | 1.00× |
| 18 | 3.39× | 16 | 8 | 3.30× |
| 20 | 6.01× | 32 | 32 | 6.01× |

## apply_single_mode_axis_i_py_sum_diagnostic

| n \ threads | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| 16 | 0.879 ms | 0.876 ms | 0.920 ms | 0.882 ms | 0.880 ms | 0.877 ms | 0.879 ms |
| 18 | 3.608 ms | 2.189 ms | 1.592 ms | 1.872 ms | 2.794 ms | 5.331 ms | 4.042 ms |
| 20 | 12.874 ms | 8.778 ms | 5.894 ms | 4.654 ms | 4.688 ms | 4.864 ms | 5.718 ms |

### apply_single_mode_axis_i_py_sum_diagnostic — speedup vs threads=1 (median)

| n \ threads | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| 16 | 1.00× | 1.00× | 0.96× | 1.00× | 1.00× | 1.00× | 1.00× |
| 18 | 1.00× | 1.65× | 2.27× | 1.93× | 1.29× | 0.68× | 0.89× |
| 20 | 1.00× | 1.47× | 2.18× | 2.77× | 2.75× | 2.65× | 2.25× |

### apply_single_mode_axis_i_py_sum_diagnostic — knee (memory-bandwidth saturation point, smallest threads achieving ≥ 95% of max speedup)

| n | max speedup | max @ threads | knee threads | speedup at knee |
|---|---|---|---|---|
| 16 | 1.00× | 2 | 1 | 1.00× |
| 18 | 2.27× | 4 | 4 | 2.27× |
| 20 | 2.77× | 8 | 8 | 2.77× |

## trotter_step

| n \ threads | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| 16 | 2.467 ms | 2.465 ms | 5.644 ms | 2.473 ms | 2.465 ms | 2.472 ms | 2.467 ms |
| 18 | 10.187 ms | 5.255 ms | 2.953 ms | 2.602 ms | 2.559 ms | 6.811 ms | 8.647 ms |
| 20 | 42.821 ms | 22.999 ms | 13.692 ms | 8.634 ms | 6.823 ms | 8.398 ms | 8.276 ms |

### trotter_step — speedup vs threads=1 (median)

| n \ threads | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| 16 | 1.00× | 1.00× | 0.44× | 1.00× | 1.00× | 1.00× | 1.00× |
| 18 | 1.00× | 1.94× | 3.45× | 3.92× | 3.98× | 1.50× | 1.18× |
| 20 | 1.00× | 1.86× | 3.13× | 4.96× | 6.28× | 5.10× | 5.17× |

### trotter_step — knee (memory-bandwidth saturation point, smallest threads achieving ≥ 95% of max speedup)

| n | max speedup | max @ threads | knee threads | speedup at knee |
|---|---|---|---|---|
| 16 | 1.00× | 16 | 1 | 1.00× |
| 18 | 3.98× | 16 | 8 | 3.92× |
| 20 | 6.28× | 16 | 16 | 6.28× |
