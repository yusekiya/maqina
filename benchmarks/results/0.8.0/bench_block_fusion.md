# bench_block_fusion results

label: `bench-block-fusion`

## machine info

- **label**: `bench-block-fusion`
- **platform**: `Linux-5.15.0-156-generic-x86_64-with-glibc2.35`
- **machine**: `x86_64`
- **processor**: `x86_64`
- **python_version**: `3.13.3`
- **numpy_version**: `2.4.4`
- **cpu_count_logical**: `64`
- **timestamp_utc**: `2026-05-18 13:19:04 UTC`
- **rayon_num_threads_env**: `None`
- **kinema_version**: `unknown`
- **__has_blas__**: `True`
- **__has_rayon__**: `True`
- **__has_simd__**: `True`

## per-cell median wall time (repeat=7)

| n | kernel | median wall_sec | median calls/sec |
|---|---|---|---|
| 18 | apply_h_kinema | 2.718419e-04 | 3.679e+03 |
| 18 | trotter_step | 2.468068e-03 | 4.052e+02 |
| 20 | apply_h_kinema | 9.532124e-04 | 1.049e+03 |
| 20 | trotter_step | 8.149054e-03 | 1.227e+02 |
| 22 | apply_h_kinema | 5.069751e-03 | 1.972e+02 |
| 22 | trotter_step | 1.633804e-02 | 6.121e+01 |

## 使い方 (baseline vs after の手動 diff)

1. main tip と PR tip で本 script を **同条件** (RAYON_NUM_THREADS, blas_threads, n_values, repeat) で 2 回回す.
2. 2 つの md / CSV を見比べ, per-cell speedup = `baseline_median / after_median` を計算.
3. acceptance: `n=20`, `kernel=trotter_step` で `speedup >= 1.3`.
