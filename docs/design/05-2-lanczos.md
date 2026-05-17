# §5.2 Lanczos: `lanczos_propagate`

`exp(-i dt H) ψ` を `m` 次元 Lanczos + 三重対角固有分解で計算する
matrix-free 短時間プロパゲータ (Park-Light 1986)。実装方針:

- 配列型は **ndarray** に統一。
- 三重対角の対称固有分解は **hand-rolled な implicit QL with Wilkinson shift**
  を `src/tridiag.rs` に持つ (LAPACK 依存を切る; 詳細 §7.1)。
- ベクトル長 dim 依存 ops は `cblas` クレート経由のラッパで書く
  (BLAS feature on/off を Cargo features で切替; §7.1, §7.4)。
- Full re-orthogonalization (Gram-Schmidt 2-pass) を採用。
- 部分空間打切り条件 `β_k < tol` で `m_eff = k+1`。
- 終端再構成: `psi_new = V[:, :m_eff] @ c` を `zgemv` 1 回で。

API:

```rust
pub(crate) fn lanczos_propagate<F>(
    mut matvec: F,
    psi: &[Complex64],
    dt: f64,
    m: usize,
    tol: f64,
) -> PyResult<(Vec<Complex64>, usize, f64, f64)>
where
    F: FnMut(&[Complex64], &mut [Complex64]),  // y = H · v
```

return 値は `(psi_new, m_eff, β_m, |c_m|)`:

- `psi_new`: 近似された `exp(-i dt H) ψ`.
- `m_eff`: 実構築 Krylov 次元 (`β_k < tol` の早期打切後の k+1, または `m` 上限).
  `ψ=0` / `dim=0` 縮退時は `0`.
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

