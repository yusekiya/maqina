# 0.14.0 — README fidelity-vs-runtime 図の取り直し (adaptive controller 改善 #148 + 新 BDF 参照)

> **ドラフト** — 数値は確定 (bench_*.csv より)。文面はレビュー前。

0.14.0 で adaptive step-size controller を「真の PI 化 (#151) / reject 過剰縮小解消
(#149) / reject 後成長凍結 (#150)」と変更した (umbrella #148)。この controller 変更は
adaptive 系 method (`cfm4_adaptive_richardson_krylov` / `_chebyshev`) の挙動に影響する
ため README 図を取り直す。あわせて issue #158 で wide dynamic range (stiff) の参照解を
Adams primary → **BDF tol-sweep primary** に差し替えた (bdf_pairwise_inf=[0,0] 自己収束)
ものを使用する。

図: `docs/figures/0.14.0_pareto_non_stiff.png` / `docs/figures/0.14.0_pareto_stiff.png`
(4 系列: Krylov adapt. / Krylov fixed / Chebyshev adapt. / QuTiP (Adams))。

## 0. Machine info & bench params

- **machine**: Intel Xeon Platinum 8470N, 2 socket × 52 core = **104 cores (HT off)**,
  2 NUMA, OpenBLAS 0.3.26, AVX-512, 503 GiB RAM。**README ベンチ専用機**
  (回帰/perf 用 AMD EPYC 7713P とは別機; 0.8.0/0.12.0 README 図と同一サーバー)。
- **threads**: BLAS / RAYON とも env 未設定 (default)。0.12.0 と同条件。
  CLAUDE.md の「本番 perf bench は `--blas-threads 8`」は EPYC sweep 由来の指針で
  README 図 (別機 + 0.12.0 が default 取得) には非適用。
- **maqina version**: 0.14.0。
- **N / T / seed**: 18 / 10000 / 20260518。
- **scenarios**: `non-stiff` (narrow dynamic range, h_p_scale=1) / `stiff`
  (wide dynamic range, h_p_scale=10)。
- **問題 / 参照解 npz**: `benchmarks/data/{problem,reference}_{non-stiff,stiff}_n18_*.npz`。
  非stiff 参照解 = Adams primary (commit 2712a86, 不変)。stiff 参照解 = **BDF primary**
  (commit b36b8f4, bdf_tol=1e-13, bdf_pairwise_inf=[0,0], 自己収束; solver_independent
  flag は False = Adams cross-check 不一致だが stiff の Adams 限界由来で参照解の不備
  ではない)。
- **sweep**: Krylov adapt `atol ∈ {1e-3,1e-5,1e-7}` / Chebyshev adapt
  `atol ∈ {1e-2..1e-7}` (propagator_tol=1e-12 固定) / cfm4 fixed `dt ∈ {5,2,0.5,0.2}`
  (stiff のみ) / QuTiP (Adams) `tol ∈ {1e-3,1e-5,1e-7,1e-9}`。

## 1. 取り直しスコープと流用

| 系列 | non-stiff | stiff | ψ 保存 |
|---|---|---|---|
| Krylov adaptive | 再計算 | 再計算 | ✅ `0.14.0/states/` |
| Chebyshev adaptive | 再計算 | 再計算 | ✅ `0.14.0/states/` |
| Krylov fixed (cfm4) | **0.8.0 流用** (controller 非依存 + 非stiff 参照不変) | 再計算 (新 BDF 参照) | stiff は本来 ✅ だが下記 ※ |
| QuTiP (Adams) | 再計算 (状態保存) | 再計算 (新 BDF 参照 + 状態保存) | ✅ `qutip/states/` |

- QuTiP は version 非依存なので共有 `benchmarks/results/qutip/` に CSV + `states/` を置く。
- 保存 ψ: maqina 18 (非stiff: krylov3+cheb6 / stiff: krylov3+cheb6) + QuTiP 8 = **26 cell**。
- ※ **stiff 固定 cfm4 の ψ は本 run では欠落**。本 run はスクリプト更新 (cfm4 ψ 保存化,
  commit 7f99e55) より前に起動したため (CSV 行は再生成済)。次回 run では保存される。
  cfm4 は version 依存で更新毎に再計算が必要なため保存価値は元々小さく、図/解析への影響なし。

## 2. infidelity の比較可能性 (重要)

- **非stiff は参照解不変** → infidelity は version 間で **直接比較可**。
- **stiff は参照解が Adams → BDF に差替** → infidelity は **version 間で比較不可**。
  stiff の version 間比較は **wall / n_steps のみ** (solver 側・参照非依存)。infidelity は
  **0.14.0 内 (同一 BDF 参照)** でのみ意味を持つ。
- これが stiff を全系列再計算した理由。

## 3. non-stiff 結果 (narrow dynamic range)

### 3.1 Krylov adaptive (vs 0.8.0, 同一 Adams 参照)

| atol | wall 0.8.0→0.14.0 | n_steps 0.8.0→0.14.0 | infidelity 0.8.0→0.14.0 |
|---|---|---|---|
| 1e-3 | 1645→1732 (+5.3%) | 1877→2054 (+9.4%) | 8.33e-5 → 7.98e-5 |
| 1e-5 | 4197→4371 (+4.1%) | 8098→8500 (+5.0%) | 1.11e-10 → 8.25e-11 |
| 1e-7 | 9211→9530 (+3.5%) | 20987→21967 (+4.7%) | 0.0 → 0.0 (floor) |

### 3.2 Chebyshev adaptive (vs 0.12.0, 同一 Adams 参照)

| atol | n_steps 0.12.0→0.14.0 | wall 0.12.0→0.14.0 | infidelity 0.12.0 → 0.14.0 |
|---|---|---|---|
| 1e-2 | 1000→1000 | 343→300 | 9.13e-4 → 9.13e-4 |
| 1e-3 | 1884→2050 (+8.8%) | 414→397 | 1.05e-4 → 9.86e-5 |
| 1e-4 | 4754→4805 (+1.1%) | 677→652 | 5.46e-7 → 1.78e-5 |
| 1e-5 | 8104→8506 (+5.0%) | 946→948 | 3.31e-10 → 0.0 (floor) |
| 1e-6 | 13139→13763 (+4.8%) | 1363→1352 | 0.0 → 0.0 |
| 1e-7 | 21005→21986 (+4.7%) | 1976→1932 | 0.0 → 0.0 |

→ **regression なし**。両系列とも **n_steps +5〜9%** (真の PI 化による dt 系列の保守化)、
wall は ±5% (ノイズ域; per-step は #100 等で微減)、infidelity は同等。atol=1e-4 で
Chebyshev が 5.46e-7→1.78e-5 とやや悪化して見えるが、これは過剰高精度の抑制 (atol への
忠実化) で、atol 内 (§7.3 参照)。

## 4. stiff 結果 (wide dynamic range) — version 間は wall/n_steps のみ

### 4.1 Krylov adaptive (vs 0.8.0)

| atol | wall 0.8.0→0.14.0 | n_steps 0.8.0→0.14.0 |
|---|---|---|
| 1e-3 | 9325→8089 (**−13.3%**) | 6581→6117 (**−7.1%**) |
| 1e-5 | 17875→17554 (−1.8%) | 26600→28243 (+6.2%) |
| 1e-7 | 50834→53124 (+4.5%) | 111299→116958 (+5.1%) |

→ controller 変更は **loose atol で step 減 (高速化)、tight atol で step 微増**という
トレードオフ。non-stiff の一律 +5% とは逆向きで、振動が起きやすい stiff・loose 領域で
効果が出ている。**wall は controller のみに帰属できない** (#100 iter-0 memoization 等の
カーネル変更が混在; §7.1)。**clean な指標は n_steps**。

### 4.2 Chebyshev adaptive (vs 0.12.0)

| atol | n_steps 0.12.0→0.14.0 | wall 0.12.0→0.14.0 |
|---|---|---|
| 1e-2 | 1000→1000 | 643→615 |
| 1e-3 | 1914→2078 (+8.6%) | 787→749 |
| 1e-4 | 5930→6822 (+15.0%) | 1197→1201 |
| 1e-5 | 24689→27223 (+10.3%) | 2732→2835 |
| 1e-6 | 59506→63955 (+7.5%) | 5636→5517 |
| 1e-7 | 111402→117080 (+5.1%) | 9211→8928 |

→ n_steps +5〜15% (controller 保守化)、wall ±5%。regression なし。

### 4.3 固定 cfm4 (vs 0.8.0) — 数値整合性チェック

| dt | wall 0.8.0→0.14.0 | infidelity 0.8.0 → 0.14.0 |
|---|---|---|
| 5 | 740→739 (−0.1%) | 5.305e-4 → **5.305e-4 (完全一致)** |
| 2 | 1851→1834 (−0.9%) | 8.926e-4 → **8.926e-4 (完全一致)** |
| 0.5 | 6579→6593 (+0.2%) | 4.401e-6 → **4.401e-6 (完全一致)** |
| 0.2 | 12128→12200 (+0.6%) | 1.314e-7 → **1.314e-7 (完全一致)** |

→ **infidelity 6桁完全一致 + wall ±1%**。固定 cfm4 は controller 非依存・決定的で、
Phase C #142 refactor が数値等価であること、matvec カーネルが wall 安定であることを裏付け。
また infid ≥ 1e-7 では **参照解変更 (8.5e-12) が不可視**であることの実証 (stiff だが旧 Adams
参照と新 BDF 参照で同値)。

## 5. 【本命】Chebyshev vs Krylov の Pareto 比較 (0.14.0 内, 同一 BDF 参照)

wide dynamic range で同 atol を直接比較:

| atol | Krylov wall / nsteps | Chebyshev wall / nsteps | **Cheb 高速化** |
|---|---|---|---|
| 1e-3 | 8089 s / 6117 | 749 s / 2078 | **10.8×** |
| 1e-5 | 17554 s / 28243 | 2835 s / 27223 | **6.2×** |
| 1e-7 | 53124 s / 116958 | 8928 s / 117080 | **6.0×** |

- atol=1e-7 は **n_steps がほぼ同一 (116958 vs 117080)** → 6.0× は **純粋に per-step
  コスト差** (Krylov 0.454 vs Cheb 0.0763 s/step ≈ 5.95×)。Lanczos の V 行列キャッシュ溢れ
  + 直交化 vs Chebyshev 3 項漸化の差 (#120/Phase B の動機が wide・実スケールで実証)。
- **reference 精度到達**: Chebyshev は atol=1e-6 で reference floor (infidelity 0.0,
  < BDF 精度 ≈1e-12) に **5517 s** で到達。Krylov の最良 (1e-7, 2.71e-11) は **53124 s**。
  → **Chebyshev は reference 精度に 9.6× 速く到達**。

## 6. QuTiP (Adams) の挙動

QuTiP `sesolve` (Adams, sparse) は version 非依存。infidelity (non-stiff = Adams 参照,
stiff = 新 BDF 参照):

| tol | non-stiff wall / infid | stiff wall / infid |
|---|---|---|
| 1e-3 | 3364 s / 4.39e-7 | 18501 s / **1.0 (失敗)** |
| 1e-5 | 16514 s / 1.15e-6 | 49572 s / 2.99e-10 |
| 1e-7 | 14302 s / **5.77e-2 (スパイク)** | 49797 s / 2.91e-7 |
| 1e-9 | 27399 s / 1.29e-10 | 58730 s / 0.0 (floor) |

→ **両 scenario で非単調** (non-stiff は tol=1e-7、stiff は tol=1e-7 で悪化)。これは
Adams (非剛性多段法) の **虚軸安定性 × 適応次数選択**由来の既知挙動 (§7.4)。図上では
QuTiP-Adams 系列が右側 (低速) かつ erratic で、汎用 ODE ソルバが long-T 量子ダイナミクスに
不向きなことを示し、maqina の構造保存伝播器の優位を裏付ける。

## 7. 解釈ノート (README には記載しない; 簡潔さ優先)

### 7.1 wall の帰属
0.8.0→0.14.0 で Krylov 経路に #100 (Richardson iter-0 matvec memoization, ≈3% per-step 減)
+ Phase C #142 refactor が入っており、**per-step wall は version 不変量ではない**。wall 差を
controller のみに帰属できない。**n_steps が controller の clean な指標**。dt 振動・受理率は
本 bench では観測していないので「ノコギリ波抑制が高速化要因」とは断定しない。

### 7.2 infidelity の cross-version 比較可否
非stiff = 参照不変 → 比較可。stiff = 参照 Adams→BDF 差替 → **比較不可** (wall/n_steps のみ)。

### 7.3 floor 近傍 cell の非単調性 (cancellation 残差)
Chebyshev stiff で atol=1e-6 (0.0) → 1e-7 (4.44e-9) と infidelity が増える。原因は
**round-off でも参照精度でもない** (定量的に round-off ≈1e-18 / 参照 floor ≈1e-12 で棄却済)。
realized infidelity は **局所 Magnus 誤差が時間方向に部分相殺した残差**で、相殺効率が
atol 依存の dt 系列に hypersensitive。1e-6 が偶然 deep cancellation、1e-7 は ほぼ random-walk。
**infid ≲ 1e-9 の cell は「相殺残差律速」**で精度差として読まない。Pareto は floor より上の
cell と wall で論じる。
- Magnus 由来なので Chebyshev/Krylov 共通だが、**同 atol で残差は一致しない**: realized
  infidelity = (Magnus 残差 + propagator 誤差) で、propagator 誤差は method 固有
  (Krylov: propagator_tol=atol×1e-3 / Chebyshev: 1e-12 固定)。決定実験で Krylov の
  propagator_tol を 1e-12 に揃えると Chebyshev に一致することを確認済。stiff 1e-7 で
  Krylov (2.71e-11) < Chebyshev (4.44e-9) なのは、propagator 誤差が controller の
  `err_magnus = max(0, err − err_prop)` 経由で dt 系列を僅かに分岐させ (N: 116958 vs 117080)、
  floor 近傍の hypersensitive な残差で順序が入れ替わったため (本質的に「くじ引き」、
  Krylov に内在的優位はない)。

### 7.4 QuTiP-Adams 非単調 (虚軸安定性)
量子発展は固有値が純虚数 (強振動)。Adams 多段法は虚軸安定領域が狭く次数依存。VODE の
適応次数/ステップ選択が特定 tol で不安定な組に当たりスパイク、tighter tol で回復。
小 n 検証で T=300 では出ず T=10000 で再現、BDF では単調 → Adams 固有と確認済。

## 8. Acceptance 判定

| 項目 | 状態 |
|---|---|
| 0.14.0 controller 変更の挙動を反映 | ✅ n_steps で loose=減/tight=微増 (stiff)、+5% (non-stiff) |
| stiff 新 BDF 参照を使用 | ✅ |
| regression なし | ✅ accuracy 維持 (atol 内)、cfm4 完全一致、wall ±数% |
| wide dynamic range で Chebyshev が Pareto 支配 | ✅ 同 atol 6〜11×、reference 精度到達 9.6× |
| QuTiP 比較 | ✅ Adams 非単調含め honest に提示 (図に QuTiP (Adams) 明記) |
| 状態ベクトル保存 (参照差替耐性) | ✅ 26 cell (stiff cfm4 ψ は次回 run で補完) |

## 9. データ来歴

- 生 CSV: `benchmarks/results/0.14.0/bench_{non-stiff,stiff}.csv` (maqina) /
  `benchmarks/results/qutip/bench_{non-stiff,stiff}.csv` (QuTiP, 新 BDF 参照で再生成)。
- 流用: `benchmarks/results/0.14.0/bench_reused_non-stiff.csv` (narrow 固定 cfm4 を
  0.8.0 から solver=kinema→maqina 変換)。
- 状態ベクトル: `benchmarks/results/0.14.0/states/state_*.npz` (maqina 18) /
  `benchmarks/results/qutip/states/state_*.npz` (QuTiP 8)。self-describing
  (problem/reference パス + sweep メタ + maqina version 同梱); 参照解差替時に保存 ψ から
  infidelity 再計算可。
- 図再生成: `benchmarks/plot_readme_figure.py --version 0.14.0` (入力 = 上記 3 CSV 群)。
