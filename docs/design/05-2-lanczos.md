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
) -> PyResult<Vec<Complex64>>
where
    F: FnMut(&[Complex64], &mut [Complex64]),  // y = H · v
```

`matvec` を closure に取り、`cfm4.rs` から `(c_drv, c_diag)` を畳み込んだ
線形結合版 closure を渡せるようにする (CFM4:2 の各 stage 用の Hamiltonian
を 1 つの線形結合 matvec として表現する経路)。これにより CFM4:2 の各
ステージで matvec 呼出は m 回のみ。

