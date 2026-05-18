# §5.2 Lanczos: `lanczos_propagate`

`exp(-i dt H) ψ` を `m` 次元 Lanczos + 三重対角固有分解で計算する
matrix-free 短時間プロパゲータ (Park-Light 1986)。実装方針:

- 配列型は **ndarray** に統一。
- 三重対角の対称固有分解は **hand-rolled な implicit QL with Wilkinson shift**
  を `src/tridiag.rs` に持つ (LAPACK 依存を切る; 詳細 §7.1)。
- ベクトル長 dim 依存 ops は `cblas` クレート経由のラッパで書く
  (BLAS feature on/off を Cargo features で切替; §7.1, §7.4)。
- Full re-orthogonalization (Gram-Schmidt 2-pass) を採用。
- 部分空間打切り条件: **a posteriori 早期打切**
  `β_k · |c_last| · |dt| / (k+1) < krylov_tol` (issue #98 Phase 8;
  下記 "a posteriori 早期打切" 節) + numerical breakdown safety
  `β_k < 1e-14` で `m_eff = k+1`。
- 終端再構成: `psi_new = ‖ψ‖ · V[:, :m_eff] @ c` を `zgemv` 1 回で
  (`c` は psi_norm 抜きの pure な行列要素として保持し, 終端で
  coeff に `psi_norm` を畳み込む)。

API:

```rust
pub(crate) fn lanczos_propagate<F>(
    mut matvec: F,
    psi: &[Complex64],
    dt: f64,
    m: usize,
    krylov_tol: f64,
) -> PyResult<(Vec<Complex64>, usize, f64, f64)>
where
    F: FnMut(&[Complex64], &mut [Complex64]),  // y = H · v
```

return 値は `(psi_new, m_eff, β_m, |c_m|)`:

- `psi_new`: 近似された `exp(-i dt H) ψ`.
- `m_eff`: 実構築 Krylov 次元 (a posteriori 早期打切 / numerical breakdown の
  効果で `m_eff ≤ m`)。 `ψ=0` / `dim=0` 縮退時は `0`.
- `β_m`: build 完了時の最終 off-diagonal `β_{m_eff-1}` (= 次の Krylov 方向への
  漏れ強度). a posteriori 誤差推定の主因子.
- `|c_m|`: `c = exp(-i dt T_m) e_0` の m 番目 (= 末尾) 成分の絶対値 (`‖ψ‖` を
  含まない pure な行列要素).

`matvec` を closure に取り、`cfm4.rs` から `(c_drv, c_diag)` を畳み込んだ
線形結合版 closure を渡せるようにする (CFM4:2 の各 stage 用の Hamiltonian
を 1 つの線形結合 matvec として表現する経路)。これにより CFM4:2 の各
ステージで matvec 呼出は m 回のみ。

## a posteriori 誤差推定 (issue #93 Phase 7)

Saad (1992) / Hochbruck-Lubich (1997) より, Lanczos 近似の真の誤差は

```
‖exp(-i dt H) ψ - V_m exp(-i dt T_m) V_m^* ψ‖
  ≤ ‖ψ‖ · β_m · |⟨e_m, exp(-i dt T_m) e_0⟩|
```

の上界を持つ (`|⟨e_m, ...⟩|` は `e_m^T exp(-i dt T_m) e_0`, つまり c の末尾成分).
小 dt 領域では高次補正により

```
err_lanczos ≈ β_m · |c_m| · ‖ψ‖ · dt / m_eff
```

が 5% 精度で実誤差と一致することが `tools/verify_beta_m_estimator.py` で
108 cell sweep により確認済み (PR #94 / commit a posteriori validation).
本式は adaptive Richardson driver の **Lanczos 誤差と Magnus 誤差の分離**
(`evolve_schedule_adaptive_richardson` の PI controller) に使う:
- `err_lanczos > tol_step` → Krylov 不足 (m_max を増やす, 別 issue #93 Step 4)
- `err_lanczos < tol_step` → Lanczos は充分, dt 起因の Magnus 誤差が支配的
  (PI controller は `err_magnus = err - err_lanczos` で dt を制御)

詳細は §5.3 propagator の "Richardson 誤差源分離" 節を参照.

## a posteriori 早期打切 (issue #98 Phase 8)

Phase 7 (#93) で a posteriori 推定子そのものを Lanczos の戻り値として
expose した後, **Lanczos 内部の早期打切判定も同じ推定子で行う** よう
issue #98 (Phase 8) で書き換えた. これにより `krylov_tol` の意味を
**"Krylov 近似の許容誤差"** に再定義する (旧 β 単体閾値ではない).

### 旧仕様の問題 (Phase 7 までの動機付け)

旧仕様の判定は `β_k < krylov_tol` のみだったが, TFIM のような実用問題では
中間 β_j が O(‖H‖) ≈ 数 で機械精度に近い値を取らない. `tools/diag_beta_j_dump.py`
(#93 Step 0) で n=8, dt=0.1, s=0.5 のケースをダンプすると

```
j= 0: β=2.83e-01  |c_j|=1.00e+00
j= 1: β=1.81e+00  |c_j|=2.79e-02
j= 2: β=2.08e+00  |c_j|=2.53e-03
j= 3: β=2.50e+00  |c_j|=1.76e-04
j= 4: β=2.18e+00  |c_j|=1.10e-05
j= 5: β=2.20e+00  |c_j|=4.80e-07   ← β·|c|·dt/6 ≈ 1.8e-8 が atol=1e-7 を切る
j= 6: β=2.20e+00  |c_j|=1.76e-08
...
```

|c_j| は幾何級数的に減衰するのに対し β_j は O(‖H‖) で頭打ち.
**真の誤差は積 β · |c| · dt / m で決まる** ため, β 単体閾値では
`krylov_tol = 1e-12` でも発火せず m_eff = m_max 固定. これが #65 / #93
で観測された "Pareto 劣位" の真の原因だった.

### 新仕様 (Phase 8 / issue #98 C 案)

各 iter `k` (0-indexed) で T_{k+1} (size k+1) の三重対角固有分解
(`tridiag_c_last_abs` ヘルパ) を行い, Hochbruck-Lubich 1997 eq. 5.4-5.5 の
上界 `est = β_k · |c_last(T_{k+1})| · |dt| / (k+1)` を計算. これを
`krylov_tol` と比較し, 下回ったら `m_eff = k+1` で早期打切する.

```rust
// src/krylov.rs から抜粋
let c_last_abs = tridiag_c_last_abs(&alpha[..=k], &beta[..k], dt)?;
let est = beta_k * c_last_abs * dt.abs() / ((k + 1) as f64);
if est < krylov_tol {
    m_eff = k + 1;
    break;
}
```

`‖ψ‖ = 1` は Lanczos 内部の正規化空間の規約 (`v_0 = ψ / ‖ψ‖`). 物理的な
状態ノルムは最終 ψ_new = ‖ψ‖ · V · c で復元するので, 内部判定の式から
`‖ψ‖` ファクタは除外できる (= 相対誤差で判定).

### `krylov_tol` の意味再定義 (Phase 8 で破壊的セマンティクス変更)

| Phase | `krylov_tol` の意味 | 判定式 |
|---|---|---|
| 〜7 | β 単体閾値 (`β_k < krylov_tol`) | 実用では発火せず m_eff = m_max 固定 |
| 8+ | **Krylov 近似の許容誤差** (literature 標準) | `β · |c| · |dt| / m < krylov_tol` で発火 |

API シグネチャは不変 (Rust pyfunction も Python driver も同じ
`krylov_tol: float` 引数を受ける). しかし **同じ値を渡しても挙動が変わる**
ため minor bump (`0.x → 0.(x+1)`) を伴う. ユーザー影響:

- `QuantumAnnealer(krylov_tol=None)` (デフォルト): 経路ごとの auto-resolve
  ロジックは不変 (adaptive: `tol_step · 1e-3`, fixed-dt: `1e-12`).
  意味解釈が "β 閾値" → "Krylov 誤差閾値" に変わるだけ.
- `QuantumAnnealer(krylov_tol=1e-12)` (明示指定): 旧仕様では発火しなかった
  早期打切が新仕様では発火 → m_eff が縮み, 結果として step あたりの cost が
  下がる. 数値結果 (`ψ_new`) は誤差内で一致するが `m_eff_history` 系の
  統計値は変動.

### numerical breakdown safety

a posteriori 判定とは別に `β_k < 1e-14` (machine eps スケール) のときは
即打切する hard sanity check を持つ. これは `v_{k+1} = w / β_k` の
division by zero を回避する目的で, 物理的には「Krylov 部分空間が完全 flat
になった (誤差 = 0)」状況に対応. Phase 7 までの "`β_k < krylov_tol`" 打切と
表面的には似るが, 概念的には別物 (許容誤差ではなく数値破綻).

定数 `NUMERICAL_BREAKDOWN_TOL = 1e-14` は `src/krylov.rs` / Python ref
(`_LANCZOS_NUMERICAL_BREAKDOWN_TOL`) に hard-coded.

### overhead 計算量

per-iter で T_{k+1} の三重対角固有分解 + c_last 計算が必要 (O(k²) per iter,
累積 O(m³/3)). production 規模 (n=16, dim=65536, m=24) では:

| 量 | flop 量 |
|---|---|
| baseline Lanczos 累積 | ~3×10⁷ (matvec + reortho + axpy) |
| a posteriori 判定累積 (`tridiag_c_last_abs` × m) | ~10⁴ |
| **overhead 率** | **~0.03%** (無視可) |

小 dim でも overhead 率は 1% 未満. 実 m_eff の削減 (24 → 6-10 程度) で
3-4× speedup する savings が overhead を 3 桁圧勝する.

### Python リファレンス実装

`python/kryanneal/krylov.py::_python_lanczos_propagate` も完全に同一ロジックで
書かれており, Rust 経路と `rel < 1e-13` で一致するのが契約
(`tests/test_krylov.py::test_rust_lanczos_matches_python_reference`).
内部の `_tridiag_c_last_abs` ヘルパも対応する.

