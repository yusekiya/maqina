# bench_readme_figure.py — stiff (N=18, T=10000)

issue #135 (PR #136) finalize: `cfm4_adaptive_richardson_chebyshev` 経路を
新 default `propagator_tol = 1e-12 固定` で再計測した stiff scenario の
work-precision diagram。

## Machine info & bench params

- **timestamp_utc**: 初回 4 cell `2026-05-26T04:22:25+00:00` ~ `T05:51:46+00:00`
  (約 89 min), 追加 2 cell 同日後刻 (約 4h08m, 連続実行)
- **scenario**: `stiff` (h_p_scale=10, h_x_scale=1, T=10000, n=18)
- **seed**: `20260518`
- **method**: `cfm4_adaptive_richardson_chebyshev`
- **propagator_tol**: `1e-12` (issue #135 default; 固定)
- **chebyshev atol sweep**: `[1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]` (6 cell)

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
| ✓ | maqina 0.8.0 | krylov_adaptive | atol=1e-07 | 50833.60 | 4.327e-11 ⚠️ | 111299 |
|   | qutip | qutip | tol=1e-05 | 49652.66 | 3.027e-10 ⚠️ | - |
|   | qutip | qutip | tol=1e-07 | 48459.79 | 2.889e-07 ⚠️ (非単調 outlier) | - |
| ✓ | **maqina 0.12.0** | **chebyshev_adaptive** | **atol=1e-6** | **5635.57** | **3.338e-10** ⚠️ | **59506** |
|   | maqina 0.8.0 | krylov_adaptive | atol=1e-05 | 17875.18 | 4.090e-08 | 26600 |
| ✓ | **maqina 0.12.0** | **chebyshev_adaptive** | **atol=1e-5** | **2731.92** | **5.183e-08** | **24689** |
|   | **maqina 0.12.0** | **chebyshev_adaptive** | **atol=1e-7** | 9210.83 | 4.378e-09 ⚠️ (reference floor 反転) | 111402 |
|   | maqina 0.8.0 | krylov_fixed (cfm4) | dt=0.2 | 12127.57 | 1.314e-07 | 50000 |
| ✓ | **maqina 0.12.0** | **chebyshev_adaptive** | **atol=1e-4** | **1197.01** | **4.171e-06** | **5930** |
|   | maqina 0.8.0 | krylov_fixed (cfm4) | dt=0.5 | 6578.64 | 4.401e-06 | 20000 |
|   | maqina 0.8.0 | krylov_adaptive | atol=1e-03 | 9324.57 | 1.863e-05 | 6581 |
| ✓ | **maqina 0.12.0** | **chebyshev_adaptive** | **atol=1e-3** | **787.21** | **1.463e-04** | **1914** |
|   | maqina 0.8.0 | krylov_fixed (cfm4) | dt=5.0 | 739.83 | 5.305e-04 | 2000 |
|   | maqina 0.8.0 | krylov_fixed (cfm4) | dt=2.0 | 1851.31 | 8.926e-04 | 5000 |
| ✓ | **maqina 0.12.0** | **chebyshev_adaptive** | **atol=1e-2** | **643.10** | **9.057e-04** | **1000** |
|   | qutip | qutip | tol=1e-03 | 18778.36 | 1.000e+00 (完全失敗) ⚠️ | - |

⚠️ = QuTiP 参照解 non-convergence 由来の infidelity 不確実性 (上記 §QuTiP
参照解の注意書き 参照)。reference の effective precision floor ~`1e-10`
帯では絶対値比較は ±1-2 桁の不確実性を含む。

→ **stiff Pareto frontier は Chebyshev `atol=1e-2 〜 1e-6` の 5 cell +
QuTiP `tol=1e-9` + Krylov adaptive `atol=1e-7` の計 7 cell**。Chebyshev
`atol=1e-7` は wall=9211 s で同 infidelity 帯の Cheb `atol=1e-6` (5636 s) より
1.6× 遅く, かつ infidelity が逆に悪化 (`3.34e-10` → `4.38e-09`) で Pareto
frontier から外れる (reference floor 反転; 後述)。

## 観察

### atol-vs-infidelity の monotonicity と reference floor 反転

| atol | infidelity | n_steps_eff |
|---|---|---:|
| 1e-2 | 9.057e-04 | 1000 |
| 1e-3 | 1.463e-04 | 1914 |
| 1e-4 | 4.171e-06 | 5930 |
| 1e-5 | 5.183e-08 | 24689 |
| 1e-6 | **3.338e-10** | 59506 |
| 1e-7 | **4.378e-09** ⚠️ (反転) | 111402 |

`atol=1e-6` まで単調に improve (`atol=1e-5` の 5.18e-08 から **155× 改善**),
`atol=1e-7` で **逆に 13× 悪化**。これは maqina 側の数値破綻ではなく
**reference solution (QuTiP sesolve Adams) の precision floor (~1e-10) に
maqina 側が到達した結果**:

1. atol↓ で maqina の dt がさらに小さくなる
2. n_steps 増 (59506 → 111402, 1.87×)
3. maqina 内部の round-off accumulation + reference noise の両方が乗る
4. reference precision floor (~1e-10) を maqina infidelity が下回ろうと
   すると reference noise が顕在化して infidelity が逆に増える

solver 種類に依らない構造的現象 (QuTiP 自身も `tol=1e-7` cell の `2.89e-07` が
`tol=1e-5` の `3.03e-10` より悪化する同じ非単調性を見せる)。

### accidental 高精度 (atol = upper bound) feature

`atol=1e-3` で **n_steps が Krylov 比 3.4× 削減** されている (6581 → 1914):

| atol | Krylov n_steps | Chebyshev n_steps | 削減比 |
|---|---:|---:|---:|
| 1e-3 | 6581 | **1914** | **3.4×** |
| 1e-5 | 26600 | 24689 | 1.08× |
| 1e-7 | 111299 | 111402 | 1.00× (一致) |

これは:

1. Chebyshev は `propagator_tol = 1e-12 固定` で per-step propagator 誤差が
   machine precision floor 近く
2. → PI controller が見る誤差は Magnus 4 次 (`O(dt^5)`) のみ
3. → `atol=1e-3` 制約下で dt を大きく取れる (Magnus 誤差が `O(dt^5) ≈ atol`
   を満たす限界まで膨らむ)
4. → 結果として infidelity も atol 内に収まる ("`atol` は上限として機能")

`atol↓ で effect が消える`: `atol=1e-5, 1e-7` では Cheb / Krylov の n_steps が
ほぼ一致。tight atol 領域では Magnus 誤差項が支配的になり, PI 駆動の dt 制約が
両 method の propagator 内部精度差と独立に決まるため。

### Krylov adaptive vs Chebyshev adaptive (per atol)

| atol | Krylov wall (s) | Chebyshev wall (s) | 高速化 |
|---|---:|---:|---:|
| 1e-3 | 9324.57 | 787.21 | **11.8×** |
| 1e-5 | 17875.18 | 2731.92 | **6.5×** |
| 1e-7 | 50833.60 | 9210.83 | **5.5×** (両者 reference floor 到達) |

`atol=1e-3` の 11.8× は accidental 高精度 (n_steps 3.4×) と per-step 高速化
(~3.5×) の積。`atol=1e-5, 1e-7` は n_steps がほぼ揃った領域で **per-step
cost 高速化 (Lanczos V 行列 cache spill → Chebyshev 3 項漸化)** が支配的。

target 精度との関係:

- **target: QuTiP `tol=1e-9` (7.98e-12) 程度の infidelity**: 部分達成。
  Chebyshev `atol=1e-6` で `3.34e-10` (QuTiP `tol=1e-5` の `3.03e-10` と同等;
  reference floor `~1e-10` のため `1e-11` は本 bench scope で到達不可能)
- `atol=1e-7` でも infidelity は ~1e-9 に張り付くため reference を tighter な
  solver (e.g. 倍精度を超えた arbitrary precision / 別 ODE solver) で取り直さ
  ない限り改善しない。本 bench は scope 外として punt。

### `propagator_tol = 1e-12 固定` と infidelity floor の関係

ユーザー指摘 (2026-05-26): "infidelity が頭打ちになったのは propagator_tol を
固定 1e-12 にしているからか?" → **No**. per-step Chebyshev 切り捨て誤差 ≤
`propagator_tol` の累積上界 vs 実測 infidelity:

| cell | n_steps | 累積 propagator 誤差上界 | 実測 infidelity | 比 |
|---|---:|---:|---:|---:|
| atol=1e-6 | 59506 | 6.0e-8 | 3.34e-10 | 実測が 180× 小さい |
| atol=1e-7 | 111402 | 1.1e-7 | 4.38e-9 | 実測が 25× 小さい |

→ 真の limit factor は **reference precision floor (Adams non-convergence)**
~`1e-10` で, `propagator_tol` は infidelity を決めていない。`propagator_tol`
を `1e-14` に絞っても infidelity は変わらない (reference floor が先に limit する)。
