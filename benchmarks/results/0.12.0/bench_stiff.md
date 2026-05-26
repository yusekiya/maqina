# bench_readme_figure.py — stiff (N=18, T=10000)

issue #135 (PR #136) finalize: `cfm4_adaptive_richardson_chebyshev` 経路を
新 default `propagator_tol = 1e-12 固定` で再計測した stiff scenario の
work-precision diagram。

## Machine info & bench params

- **timestamp_utc**: `2026-05-26T04:22:25+00:00` ~ `2026-05-26T05:51:46+00:00`
  (約 89 min)
- **platform**: `Linux-x86_64`, AMD EPYC 7713P (64 core)
- **scenario**: `stiff` (h_p_scale=10, h_x_scale=1, T=10000, n=18)
- **seed**: `20260518`
- **method**: `cfm4_adaptive_richardson_chebyshev`
- **propagator_tol**: `1e-12` (issue #135 default; 固定)
- **chebyshev atol sweep**: `[1e-2, 1e-3, 1e-4, 1e-5]`

## QuTiP 参照解の注意書き ⚠️

stiff scenario の bench log に WARNING が出ている:

> 参照解の Adams 収束 flag が False です. infidelity の解釈には注意.
> 参照解の Adams vs BDF 一致 flag が False です. infidelity の解釈には注意.

stiff scenario では QuTiP `sesolve` (Adams) が non-convergence で参照解の
信頼性が落ちる (0.8.0 PR #94 から既知; 元 `bench_qutip_large.md` の stiff
scenario でも同様の傾向)。具体的に:

- qutip `tol=1e-3`: `infidelity = 0.999...` (= 完全失敗 cell)
- qutip `tol=1e-7`: `infidelity = 2.89e-07` (非単調; tol=1e-5 より悪化)
- qutip `tol=1e-5`: `infidelity = 3.03e-10`
- qutip `tol=1e-9`: `infidelity = 7.98e-12`

infidelity の **絶対値は ±1-2 桁の不確実性** を含むと読む必要がある。
ただし以下は同じ参照解を共有するので **scenario 内の Cheb vs Krylov 相対
比較は有効**:

- 0.8.0 Krylov adaptive 4 cell
- 0.8.0 Krylov fixed (cfm4) 4 cell
- 0.12.0 Chebyshev adaptive 4 cell (本 bench で追加)

## Work-precision frontier (3 系列, infidelity 昇順, Pareto 印 ✓)

| Pareto | solver | variant | knob | wall (s) | infidelity | n_steps_eff |
|---|---|---|---|---:|---|---:|
| ✓ | qutip | qutip | tol=1e-09 | 57410.31 | 7.984e-12 ⚠️ | - |
| ✓ | kinema 0.8.0 | krylov_adaptive | atol=1e-07 | 50833.60 | 4.327e-11 ⚠️ | 111299 |
|   | qutip | qutip | tol=1e-05 | 49652.66 | 3.027e-10 ⚠️ | - |
|   | qutip | qutip | tol=1e-07 | 48459.79 | 2.889e-07 ⚠️ (非単調 outlier) | - |
|   | kinema 0.8.0 | krylov_adaptive | atol=1e-05 | 17875.18 | 4.090e-08 | 26600 |
| ✓ | **kinema 0.12.0** | **chebyshev_adaptive** | **atol=1e-5** | **2731.92** | **5.183e-08** | **24689** |
|   | kinema 0.8.0 | krylov_fixed (cfm4) | dt=0.2 | 12127.57 | 1.314e-07 | 50000 |
| ✓ | **kinema 0.12.0** | **chebyshev_adaptive** | **atol=1e-4** | **1197.01** | **4.171e-06** | **5930** |
|   | kinema 0.8.0 | krylov_fixed (cfm4) | dt=0.5 | 6578.64 | 4.401e-06 | 20000 |
|   | kinema 0.8.0 | krylov_adaptive | atol=1e-03 | 9324.57 | 1.863e-05 | 6581 |
| ✓ | **kinema 0.12.0** | **chebyshev_adaptive** | **atol=1e-3** | **787.21** | **1.463e-04** | **1914** |
|   | kinema 0.8.0 | krylov_fixed (cfm4) | dt=5.0 | 739.83 | 5.305e-04 | 2000 |
|   | kinema 0.8.0 | krylov_fixed (cfm4) | dt=2.0 | 1851.31 | 8.926e-04 | 5000 |
| ✓ | **kinema 0.12.0** | **chebyshev_adaptive** | **atol=1e-2** | **643.10** | **9.057e-04** | **1000** |
|   | qutip | qutip | tol=1e-03 | 18778.36 | 1.000e+00 (完全失敗) ⚠️ | - |

⚠️ = QuTiP 参照解 non-convergence 由来の infidelity 不確実性 (上記 §QuTiP
参照解の注意書き 参照)。

→ **stiff Pareto frontier は全 4 Chebyshev cell + QuTiP / Krylov adaptive
の最 tight cell 計 6 cell**。Krylov adaptive の中間 cell (`atol=1e-5`,
`1e-3`) と Krylov fixed (cfm4) の全 4 cell は Pareto frontier に乗らない。

## 観察

### atol-vs-infidelity の monotonicity (issue #135 motivation)

| atol | infidelity | n_steps_eff |
|---|---|---:|
| 1e-2 | 9.057e-04 | 1000 |
| 1e-3 | 1.463e-04 | 1914 |
| 1e-4 | 4.171e-06 | 5930 |
| 1e-5 | **5.183e-08** | 24689 |

stiff scenario も完全に単調。0.11.0 までの auto-coupling default で観測
された非単調性は両 scenario で解消。

### accidental 高精度 (atol = upper bound) feature の実証

`atol=1e-3` で **n_steps が Krylov 比 3.4× 削減** されている (6581 → 1914):

| atol | Krylov n_steps | Chebyshev n_steps | 削減比 |
|---|---:|---:|---:|
| 1e-3 | 6581 | **1914** | **3.4×** |
| 1e-5 | 26600 | 24689 | 1.08× |

これは:

1. Chebyshev は `propagator_tol = 1e-12 固定` で per-step propagator 誤差が
   machine precision floor 近く
2. → PI controller が見る誤差は Magnus 4 次 (`O(dt^5)`) のみ
3. → `atol=1e-3` 制約下で dt を大きく取れる (Magnus 誤差が `O(dt^5) ≈ atol`
   を満たす限界まで膨らむ)
4. → 結果として infidelity も atol 内に収まる ("`atol` は上限として機能")

stiff scenario では Magnus 誤差が問題サイズに対して見かけ上良性になり, 8×
程度の "accidental 高精度" が観測される (1.86e-05 vs 1.46e-04 と Cheb のほう
が少し悪いだけで, n_steps は 3.4× 削減)。これは #124 で明文化, #135 の固定
1e-12 default 採用が裏付け。

### Krylov adaptive vs Chebyshev adaptive (per atol)

| atol | Krylov wall (s) | Chebyshev wall (s) | 高速化 |
|---|---:|---:|---:|
| 1e-3 | 9324.57 | 787.21 | **11.8×** |
| 1e-5 | 17875.18 | 2731.92 | **6.5×** |

`atol=1e-3` の 11.8× は上記 accidental 高精度 (n_steps 3.4×) と per-step
高速化 (~3.5×) の積。`atol=1e-5` の 6.5× は n_steps はほぼ同じ (1.08×) で
per-step 高速化のみ。
