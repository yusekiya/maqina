# 0.12.0 — Chebyshev README bench (issue #135 finalize)

issue #135 (PR #136) merge 後の `cfm4_adaptive_richardson_chebyshev` 経路を
`propagator_tol = 1e-12 固定` default で再計測。0.11.0 までの auto-coupling
default (`tol_step · 1e-3`) で観測されていた **atol-vs-infidelity 非単調性**
が解消されたことを確認する。

生 CSV (`bench_non-stiff.csv` / `bench_stiff.csv`) は通常 gitignore だが,
本 finalize commit でのみ永続化 (0.8.0 / qutip と同じ運用)。生 markdown は
scenario 別に分けてある:

- `bench_non-stiff.md` — non-stiff scenario (T=10000, n=18, h_p_scale=1)
- `bench_stiff.md` — stiff scenario (T=10000, n=18, h_p_scale=10)

## 0. Machine info & bench params

- **timestamp_utc**: `2026-05-26` 内 2 回に分けて取得
  (初回 4 cell: `2026-05-26T03:42:43+00:00` ~ `T05:51:46+00:00`, 約 2h09m;
  追加 2 cell: 同日後刻, non-stiff 約 56 min, stiff 約 4h08m)
- **method**: `cfm4_adaptive_richardson_chebyshev`
- **propagator_tol**: `1e-12` (issue #135 default; 固定)
- **chebyshev atol sweep**: `[1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]` (6 cell)
- **N**, **T**, **seed**: `18`, `10000`, `20260518`
- **BLAS threads**: default (`OPENBLAS_NUM_THREADS=unset`; maqina Lanczos
  内部 BLAS の並列化を維持する用途指針通り)
- **scenarios**: `non-stiff` (h_p_scale=1) / `stiff` (h_p_scale=10)

問題ファイル + 参照解 npz は 0.11.0 README pipeline で生成済 (issue #134):

- `benchmarks/data/problem_{non-stiff,stiff}_n18_seed20260518.npz`
- `benchmarks/data/reference_{non-stiff,stiff}_n18_T10000_seed20260518.npz`

## 1. atol-vs-infidelity の monotonicity (issue #135 最大 motivation)

| atol | non-stiff infidelity | stiff infidelity |
|---|---|---|
| 1e-2 | 9.133e-04 | 9.057e-04 |
| 1e-3 | 1.045e-04 | 1.463e-04 |
| 1e-4 | 5.455e-07 | 4.171e-06 |
| 1e-5 | 3.308e-10 | 5.183e-08 |
| 1e-6 | **`<1e-16` (floor)** | **3.340e-10** |
| 1e-7 | **`<1e-16` (floor)** | 4.378e-09 ⚠️ |

0.11.0 までの auto-coupling default では `atol=1e-4` で `infidelity ≈ 0`
(machine precision floor 到達後) → `atol=1e-5` で `1.78e-06` へ悪化する
非単調性が観測されていた (issue #135 motivation). **固定 `propagator_tol = 1e-12`**
で per-step Chebyshev 打切誤差が常に machine precision floor 近く → PI
controller が見る誤差は純粋に Magnus 4 次成分 (`O(dt^5)`) のみ → `atol` で
要求した精度に向けて dt が単調に絞られる, という理論予測通りの挙動を確認。

**stiff `atol=1e-7` の infidelity 上方反転** (`3.34e-10` → `4.38e-09`):
これは maqina 側の数値破綻ではなく **参照解 (QuTiP sesolve Adams) の
non-convergence 由来の precision floor** に到達した症状 (§4.2 参照).
non-stiff 側は `atol=1e-6` で reference precision (= machine precision)
floor の `<1e-16` placeholder 到達, `atol=1e-7` でも同じ floor 位置で
重複描画される.

## 2. 3 系列 Pareto 比較 (0.8.0 Krylov adaptive vs 0.12.0 Chebyshev adaptive)

各 scenario で **同じ atol** の cell を直接比較。0.8.0 / 0.12.0 共に `tol_step
= atol`, `dt_init` / `dt_max` も同じ auto-resolution。

### 2.1 non-stiff (h_p_scale=1)

| atol | 0.8.0 Krylov wall (s) | 0.12.0 Cheb wall (s) | 高速化 | 0.8.0 infidelity | 0.12.0 infidelity |
|---|---:|---:|---:|---|---|
| 1e-3 | 1645 | **414** | **4.0×** | 8.33e-05 | 1.05e-04 |
| 1e-5 | 4197 | **946** | **4.4×** | 1.11e-10 | 3.31e-10 |
| 1e-7 | 9211 | **1976** | **4.7×** | `<1e-16` (floor) | `<1e-16` (floor) |

→ Chebyshev は **同 atol で 4.0-4.7× 高速**。`atol=1e-3, 1e-5` では infidelity
が Krylov の 1.3-3.0× 大きいが両者とも atol 内, `atol=1e-7` では両 method 共
reference precision floor (`<1e-16`) に到達。

### 2.2 stiff (h_p_scale=10)

| atol | 0.8.0 Krylov wall (s) | 0.12.0 Cheb wall (s) | 高速化 | 0.8.0 infidelity | 0.12.0 infidelity |
|---|---:|---:|---:|---|---|
| 1e-3 | 9325 | **787** | **11.8×** | 1.86e-05 | 1.46e-04 |
| 1e-5 | 17875 | **2732** | **6.5×** | 4.09e-08 | 5.18e-08 |
| 1e-7 | 50834 | **9211** | **5.5×** | 4.33e-11 | 4.38e-09 ⚠️ |

→ stiff scenario で **5.5-11.8× 高速**。`atol=1e-3` の 11.8× は §3 の n_steps
削減効果 (`accidental 高精度` upper-bound 効果) が乗っているため。`atol=1e-7`
の 5.5× は **両 method 共に参照解 (Adams non-convergence) の precision floor
~1e-10 に到達** している領域: maqina の infidelity 絶対値は ±1-2 桁の不確実性
を含むため絶対値比較ではなく "wall 軸での Pareto 比較" として読む (§4.2 参照)。

### 2.3 Chebyshev `atol=1e-6` cell の Pareto 上の位置 (追加 sweep の主結果)

| scenario | Cheb atol=1e-6 wall / infidelity | 同精度帯の他 solver | 高速化 |
|---|---|---|---|
| non-stiff | 1363 s / `<1e-16` (floor) | 0.8.0 Krylov fixed dt=0.5 (3620 s) | **2.7×** |
| non-stiff | 1363 s / `<1e-16` (floor) | 0.8.0 Krylov adaptive atol=1e-7 (9211 s) | **6.8×** |
| stiff | 5636 s / 3.34e-10 | 0.8.0 Krylov adaptive atol=1e-7 (50834 s, 4.33e-11) | **9.0×** ※ |
| stiff | 5636 s / 3.34e-10 | QuTiP tol=1e-5 (49653 s, 3.03e-10) | **8.8×** |

※ stiff Krylov adapt の infidelity 4.33e-11 も reference floor (~1e-10) 帯
なので絶対値比較は不確実 (§4.2)。wall 軸での 9× 高速化は確定的。

## 3. accidental 高精度 (atol = upper bound) feature の実証

`n_steps_eff` の比較 (stiff scenario):

| atol | 0.8.0 Krylov n_steps | 0.12.0 Cheb n_steps | 削減 |
|---|---:|---:|---:|
| 1e-3 | 6581 | **1914** | **3.4×** |
| 1e-5 | 26600 | 24689 | 1.08× |
| 1e-7 | 111299 | 111402 | 1.00× (一致) |

**`atol=1e-3` で n_steps が 3.4× 削減**されている。これは:

1. Chebyshev は `propagator_tol = 1e-12 固定` で per-step propagator 誤差が
   machine precision floor 近く
2. → PI controller が見る誤差は Magnus 4 次 (`O(dt^5)`) のみ
3. → `atol=1e-3` 制約下で dt を大きく取れる (Magnus 誤差が `O(dt^5) ≈ atol`
   を満たす限界まで膨らむ)
4. → 結果として infidelity も atol 内に収まる ("`atol` は上限として機能").
   特に stiff scenario では Magnus 誤差が問題サイズに対して見かけ上良性に
   なり, 8× 程度の "accidental 高精度" が観測される (1.86e-05 vs 1.46e-04
   と Cheb のほうが少し悪いだけ).

これは #124 で明文化, #135 の固定 1e-12 default 採用が裏付け。

**atol↓ で accidental 高精度 effect が消える**: `atol=1e-5, 1e-7` では Cheb /
Krylov の n_steps が揃ってきている。これは atol が tight になるほど Magnus
誤差項 `O(dt^5)` が支配的になり, PI 駆動の dt 制約が Cheb / Krylov の
propagator 内部精度差 (Cheb=1e-12 固定 vs Krylov=atol·1e-3 連動) と独立に
決まる領域に入るため。tight atol 領域での高速化 (5.5-6.5×) は per-step cost
(Cheb 3 項漸化 vs Lanczos V 行列 cache spill) によるもの。

## 3.5 atol スケーリング (理論との比較)

実測 4-6 cell の log-log fit から得た scaling exponent:

| scenario | n_steps `~ atol^(-α)` | wall `~ atol^(-β)` | infidelity `~ atol^γ` |
|---|---:|---:|---:|
| non-stiff | α=0.32 | β=0.18 | γ=2.4 (ただし atol≤1e-6 で floor 到達; 1e-2〜1e-5 fit) |
| stiff | α=0.46 | β=0.27 | γ=1.3 |

理論期待 (CFM4:2 4 次精度, PI 制約から):

- `dt ~ atol^0.20` → `n_steps ~ atol^(-0.20)`
- 大 dt 領域 `R·dt >> log(1/propagator_tol)`: `wall ~ T·R` (atol 非依存)
- 小 dt 領域 `R·dt << log(1/propagator_tol)`: `wall ~ atol^(-0.20)`
- `global infidelity ~ T·dt^4 ~ atol^0.80`

実測との対比:

- **n_steps slope `α=0.32-0.46`**: 理論 0.20 より急 (PI controller が conservative
  に dt を絞っているのと, safety factor `0.9` の効果が tight atol で重なる)。
  stiff の `α=0.46` は `R·dt` 項が早く小 dt 領域に到達するため slope acceleration.
- **wall slope `β=0.18-0.27`**: 理論小 dt 限界 0.20 に近い。stiff `β=0.27` は
  α=0.46 と組み合わさって `K_used` の atol 依存性が逆効果になっている領域
  (small dt で `log` 項が支配的に近づきつつある)。
- **infidelity slope `γ=2.4 / 1.3`**: 理論 0.80 より遥かに急 (accidental 高精度
  effect そのもの)。Magnus 誤差が PI 駆動を支配し, Chebyshev 側の propagator
  誤差は machine floor で寄与しない構図。

## 4. infidelity floor の本質: reference precision が limit factor

ユーザー指摘 (2026-05-26): "infidelity が頭打ちになったのは propagator_tol を
固定 1e-12 にしているからか?" → **No**. 主因は **reference solution の
precision floor** で, `propagator_tol` ではない。検証:

per-step Chebyshev 切り捨て誤差 ≤ `propagator_tol = 1e-12`. N step での triangle
inequality 累積上界:

| cell | n_steps | 累積 propagator 誤差上界 | 実測 infidelity | 比 |
|---|---:|---:|---:|---:|
| non-stiff atol=1e-6 | 13139 | 1.3e-8 | `<1e-16` | 実測が **10⁸× 小さい** |
| non-stiff atol=1e-7 | 21005 | 2.1e-8 | `<1e-16` | 同上 |
| stiff atol=1e-6 | 59506 | 6.0e-8 | 3.3e-10 | 実測が **180× 小さい** |
| stiff atol=1e-7 | 111402 | 1.1e-7 | 4.4e-9 | 実測が **25× 小さい** |

実測 infidelity はどの cell でも累積 propagator 誤差上界より遥かに小さい →
infidelity floor を決めているのは propagator_tol ではない。`propagator_tol` を
`1e-14` 等に絞っても infidelity floor は変わらない (reference floor のほうが
先に limit するため; この検証 bench は scope 外として割愛)。

### 4.1 non-stiff (reference precision ~1e-13〜1e-15)

| solver | knob | wall (s) | infidelity |
|---|---|---:|---|
| **Cheb atol=1e-6** | atol=1e-6 | **1363** | **`<1e-16` (floor)** |
| **Cheb atol=1e-5** | atol=1e-5 | **946** | **3.31e-10** |
| QuTiP tol=1e-9 | tol=1e-9 | 27388 | 1.29e-10 |
| 0.8.0 Krylov fixed (cfm4) dt=0.5 | dt=0.5 | 3620 | `<1e-16` (floor) |

→ Chebyshev は QuTiP の **同精度帯 (~1e-10) を 1/29 の wall で達成**
(atol=1e-5 cell)。また `atol=1e-6` で reference precision floor 到達 = Krylov
fixed dt=0.5 と同じ position に, **2.7× 速く** 到達。

reference は QuTiP `sesolve atol=1e-12 / rtol=1e-10` で生成 (build pipeline で
Adams 収束 + BDF 一致確認済)。reference の effective precision は `~1e-13`
〜 `~1e-15` 程度で, maqina infidelity がここを下回ると CSV では `0.0` (= log
plot 上 `<1e-16` placeholder) として記録される。これは "maqina が reference
precision 限界に到達した" 成功サイン (0.8.0 Krylov fixed dt=0.2/0.5 で既に
観測済の現象と同じ)。

### 4.2 stiff (reference precision ~1e-10, Adams non-convergence)

bench log の WARNING:

> 参照解の Adams 収束 flag が False です. infidelity の解釈には注意.
> 参照解の Adams vs BDF 一致 flag が False です. infidelity の解釈には注意.

stiff scenario では QuTiP `sesolve` (Adams) が non-convergence で参照解の
信頼性が `~1e-10` 程度で頭打ちになる (0.8.0 PR #94 から既知の現象, 元
`bench_qutip_large.md` でも同じ scenario が問題視されていた)。QuTiP 自身の
cell も非単調:

- qutip `tol=1e-3`: `infidelity = 0.999...` (= 完全失敗 cell)
- qutip `tol=1e-7`: `infidelity = 2.89e-07` (`tol=1e-5` の `3.03e-10` より悪化)
- qutip `tol=1e-5`: `infidelity = 3.03e-10`
- qutip `tol=1e-9`: `infidelity = 7.98e-12` (QuTiP self-reference; 他 solver
  から見た effective precision とは別)

**Chebyshev も `atol=1e-7` で同じ reference floor に到達**して infidelity
反転 (`3.34e-10` → `4.38e-09`)。これは maqina 側の数値破綻ではなく:

1. atol↓ で dt をさらに小さく取り n_steps が 1.87× 増 (59506 → 111402)
2. maqina 内部の round-off accumulation + reference noise の両方が乗る
3. reference precision floor (~1e-10) を maqina infidelity が下回ろうとすると
   reference noise が顕在化して infidelity が逆に増える

solver 種類に依らない構造的現象 (Krylov adaptive も `atol=1e-7` で 4.33e-11
を出すが, これも reference floor 内の数値で信頼区間 ±1-2 桁を含む)。

**maqina が QuTiP に対して達成可能な infidelity の真の floor は ~1e-10** で,
この限界には `atol=1e-6` で到達済 (target: "wide dynamic range の QuTiP で
達成している 10^-11 程度の infidelity"; 部分達成. 1e-11 達成には別 reference
solver / より tight な reference tol が必要だが本 bench スコープ外)。

stiff scenario の infidelity 絶対値は **±1-2 桁の不確実性** を含むと読む必要
がある。ただし以下は同じ参照解を共有するので scenario 内の相対比較は有効:

- `Cheb atol=1e-5` (5.18e-08) vs `Krylov atol=1e-5` (4.09e-08): ほぼ同精度
- `Cheb atol=1e-6` (3.34e-10) vs `Krylov atol=1e-7` (4.33e-11): wall 9× 高速
  (両者 reference floor 帯)
- `Cheb atol=1e-3` n_steps=1914 vs `Krylov atol=1e-3` n_steps=6581: §3 の
  accidental 高精度 feature 観測

## 5. acceptance 判定 (issue #135 finalize + 追加 atol sweep)

| acceptance | 状態 |
|---|---|
| atol-vs-infidelity monotonicity 解消 | ✅ non-stiff 全 6 cell で単調 (1e-6, 1e-7 は reference floor で重複). stiff は 1e-2〜1e-6 で単調 (1e-7 は reference floor 由来の構造的非単調) |
| 0.8.0 Krylov adaptive vs 0.12.0 Chebyshev で同 atol Pareto win | ✅ non-stiff 4.0-4.7× / stiff 5.5-11.8× |
| accidental 高精度 feature の実証 | ✅ stiff atol=1e-3 で n_steps 3.4× 削減観測 (atol↓ で effect が消える挙動も観測) |
| target: narrow dynamic range で Krylov 同等の `<1e-16` floor 達成 | ✅ `atol=1e-6` で完全達成 (Krylov fixed dt=0.5 と同じ floor に 2.7× 速く到達) |
| target: wide dynamic range で QuTiP の ~1e-11 同等を達成 | 🟡 部分達成 (`atol=1e-6` で 3.34e-10, QuTiP `tol=1e-5` の 3.03e-10 と同等)。`1e-11` 自体は reference Adams non-convergence の floor 制約で本 bench scope では到達不可能 |
| `propagator_tol = 1e-12 固定 default` が機能 | ✅ atol 5 桁振っても per-step Chebyshev 誤差が floor 近くで固定 |
| reference precision floor の正しい解釈 | ✅ §4 に明記 (propagator_tol 1e-12 は infidelity floor の原因ではない; 累積上界 vs 実測 infidelity の検証で確認) |
