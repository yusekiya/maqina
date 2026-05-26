# bench_readme_figure.py — non-stiff (N=18, T=10000)

issue #135 (PR #136) finalize: `cfm4_adaptive_richardson_chebyshev` 経路を
新 default `propagator_tol = 1e-12 固定` で再計測した non-stiff scenario の
work-precision diagram。

## Machine info & bench params

- **timestamp_utc**: 初回 4 cell `2026-05-26T03:42:43+00:00` ~ `T04:22:25+00:00`
  (約 40 min), 追加 2 cell 同日後刻 (約 56 min, 連続実行)
- **scenario**: `non-stiff` (h_p_scale=1, h_x_scale=1, T=10000, n=18)
- **seed**: `20260518`
- **method**: `cfm4_adaptive_richardson_chebyshev`
- **propagator_tol**: `1e-12` (issue #135 default; 固定)
- **chebyshev atol sweep**: `[1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]` (6 cell)

## Work-precision frontier (3 系列, infidelity 昇順, Pareto 印 ✓)

reference: 0.11.0 で生成済 `reference_non-stiff_n18_T10000_seed20260518.npz`
(QuTiP sesolve `atol=1e-12 / rtol=1e-10` で Adams 収束 + BDF 一致確認済)。
reference の effective precision ~`1e-13` 〜 `1e-15`。maqina infidelity が
ここを下回ると `<1e-16` placeholder にマップされる。

| Pareto | solver | variant | knob | wall (s) | infidelity | n_steps_eff |
|---|---|---|---|---:|---|---:|
| ✓ | **maqina 0.12.0** | **chebyshev_adaptive** | **atol=1e-7** | **1975.58** | **`<1e-16` (floor)** | **21005** |
| ✓ | **maqina 0.12.0** | **chebyshev_adaptive** | **atol=1e-6** | **1362.51** | **`<1e-16` (floor)** | **13139** |
|   | maqina 0.8.0 | krylov_adaptive | atol=1e-09 | 22593.55 | `<1e-16` (floor) | 52964 |
|   | maqina 0.8.0 | krylov_adaptive | atol=1e-07 | 9211.35 | `<1e-16` (floor) | 20987 |
|   | maqina 0.8.0 | krylov_fixed (cfm4) | dt=0.2 | 7692.77 | `<1e-16` (floor) | 50000 |
|   | maqina 0.8.0 | krylov_fixed (cfm4) | dt=0.5 | 3619.56 | `<1e-16` (floor) | 20000 |
| ✓ | qutip | qutip | tol=1e-09 | 27387.64 | 1.286e-10 | - |
| ✓ | **maqina 0.12.0** | **chebyshev_adaptive** | **atol=1e-5** | **946.10** | **3.308e-10** | **8104** |
|   | maqina 0.8.0 | krylov_adaptive | atol=1e-05 | 4197.37 | 1.110e-10 | 8098 |
| ✓ | **maqina 0.12.0** | **chebyshev_adaptive** | **atol=1e-4** | **676.88** | **5.455e-07** | **4754** |
|   | qutip | qutip | tol=1e-05 | 16524.60 | 1.145e-06 | - |
|   | qutip | qutip | tol=1e-07 | 14221.90 | 5.767e-02 | - |
|   | qutip | qutip | tol=1e-03 | 3369.31 | 4.394e-07 | - |
|   | maqina 0.8.0 | krylov_fixed (cfm4) | dt=2.0 | 1547.21 | 1.937e-05 | 5000 |
|   | maqina 0.8.0 | krylov_adaptive | atol=1e-03 | 1645.17 | 8.331e-05 | 1877 |
| ✓ | **maqina 0.12.0** | **chebyshev_adaptive** | **atol=1e-3** | **413.78** | **1.045e-04** | **1884** |
|   | maqina 0.8.0 | krylov_fixed (cfm4) | dt=5.0 | 741.07 | 8.771e-04 | 2000 |
| ✓ | **maqina 0.12.0** | **chebyshev_adaptive** | **atol=1e-2** | **343.05** | **9.133e-04** | **1000** |

→ **non-stiff Pareto frontier は全 6 Chebyshev cell + QuTiP の最 tight cell
(`tol=1e-9`) のみ**。Krylov adaptive / fixed cfm4 はいずれも Pareto frontier
に乗らず, Chebyshev が work-precision 領域全体を支配する。

注: QuTiP `tol=1e-07` の `infidelity=5.77e-02` は **QuTiP 側の collocation
失敗** (Adams 法の中間精度帯で発生する既知の罠) で, Pareto curve には乗らない
outlier。

## 観察

### atol-vs-infidelity の monotonicity (issue #135 motivation)

| atol | infidelity | n_steps_eff |
|---|---|---:|
| 1e-2 | 9.133e-04 | 1000 |
| 1e-3 | 1.045e-04 | 1884 |
| 1e-4 | 5.455e-07 | 4754 |
| 1e-5 | 3.308e-10 | 8104 |
| 1e-6 | **`<1e-16` (floor)** | 13139 |
| 1e-7 | **`<1e-16` (floor)** | 21005 |

完全に単調 (1e-6, 1e-7 は reference precision floor の同位置に重複)。0.11.0
までの auto-coupling default (`propagator_tol = atol·1e-3`) で観測されていた
"1e-4 で floor → 1e-5 で 1.78e-6 悪化" の非単調性は完全に解消。

### Krylov adaptive vs Chebyshev adaptive (per atol)

| atol | Krylov wall (s) | Chebyshev wall (s) | 高速化 |
|---|---:|---:|---:|
| 1e-3 | 1645.17 | 413.78 | **4.0×** |
| 1e-5 | 4197.37 | 946.10 | **4.4×** |
| 1e-7 | 9211.35 | 1975.58 | **4.7×** (両者 reference floor 到達) |

同 atol で **4.0-4.7× 高速**。Lanczos の V 行列 (dim × m_max ≈ 96 MB for
N=18) が cache spill する一方, Chebyshev 3 項漸化は作業ベクトル 3 個 (~16
MB) のみで L3 内に収まる, という #122 設計の per-step cost 効果。

### QuTiP との比較

Chebyshev `atol=1e-5` (`infidelity=3.31e-10`, `wall=946 s`) は QuTiP の最
精度 cell (`tol=1e-9`: `1.29e-10`, `27388 s`) と **同精度帯を 1/29 の wall
で達成**。さらに Chebyshev `atol=1e-6` (1363 s) で reference precision floor
(`<1e-16`) に到達 → QuTiP `tol=1e-9` 比 **20× 高速 + 6 桁以上精度向上**。
QuTiP の中間 tol (`1e-5`, `1e-7`) は非単調 / 失敗で実用上 Pareto にいないので,
"QuTiP の精度帯にいる" 比較は最 tight cell のみ。

### `propagator_tol = 1e-12 固定` と infidelity floor の関係

`atol=1e-6, 1e-7` で infidelity が `<1e-16` placeholder (= 実質 0) に到達する
のは propagator_tol が直接の原因ではない。検証 (n_steps × propagator_tol で
triangle inequality 累積上界を取る):

| cell | n_steps | 累積 propagator 誤差上界 | 実測 infidelity |
|---|---:|---:|---:|
| atol=1e-6 | 13139 | 1.3e-8 | `<1e-16` (実測が 10⁸× 小さい) |
| atol=1e-7 | 21005 | 2.1e-8 | `<1e-16` (同上) |

→ 真の limit factor は **reference の effective precision**
(QuTiP `atol=1e-12 / rtol=1e-10` の collocation 限界 ~ `1e-13`〜`1e-15`)。
これは 0.8.0 の Krylov fixed dt=0.5 (propagator_tol 非依存) でも同じ floor を
打つことから solver 種類に依らない構造的現象。
