# §7. Rust 拡張

### 7.1 Crate 構成

- `pyo3 = "0.28"`
- `numpy = "0.28"`
- `ndarray = "0.16"`
- `num-complex = "0.4"`
- `cblas = "0.5"` (optional, BLAS feature)
- `blas-src = "0.12"`:
  - macOS: `accelerate` feature
  - Linux: `openblas` feature (system OpenBLAS)

**三重対角固有分解の実装方針** (確定):

Lanczos 1 step で必要な唯一の LAPACK 相当の処理は **m × m (m ~ 24) の
実対称三重対角固有分解**。これは hot path ではない (step 全体の <0.5%) ので、
LAPACK を引っ張ってくる ROI が低い。本パッケージは **`ndarray-linalg`
を依存に入れず**、`src/tridiag.rs` に **implicit QL with Wilkinson shift**
を hand-roll する。

選定理由:

- m=24 の三重対角固有分解は ~1.4 × 10⁴ FP ops で <10 μs。Lanczos 1 step
  全体は dim 依存 ops (`cblas` 経由) が支配的 (dim=2^20 で ~500 μs)
- LAPACK 依存を切ることで以下が解消される:
  - macOS で `brew install openblas` 等の追加 install 不要
  - Apple Accelerate を Level-1/2 BLAS でフル活用できる (AMX 経路)
  - `blas-src` と LAPACK backend の二重管理を避けられる
  - wheel の static 同梱が単純化する

実装規模:

- `src/tridiag.rs` は ~100〜150 行 (Wilkinson shift, Givens rotation,
  deflation 閾値, max-iter cap 含む)
- 出力 (固有値 λ_p の昇順、固有ベクトル行列 Q) は `dsteqr` 互換のシグネチャ
- Givens rotation は `f64::hypot` を使い overflow/underflow を回避
- Deflation 閾値は `|β_k| ≤ ε · (|α_{k-1}| + |α_k|)` (ε = `f64::EPSILON`)
- Max iter cap: 30·m (LAPACK `dsteqr` と同じ)
- 収束失敗時は `Err(PyRuntimeError::new_err("tridiag QL did not converge ..."))`
  を返し、Python 側で `RuntimeError` として伝播 (例外型方針は §4.8 を参照)

テスト戦略:

- `cargo test`: ランダム m×m tridiag (m ∈ {2, 8, 16, 24, 48}) を生成し、
  hand-rolled の出力と **`nalgebra::SymmetricTridiagonal`** (dev-dep のみ)
  の固有値/固有ベクトルを比較。`rel < 1e-13` で一致を要求
- `pytest`: 同じテストを Rust の公開 `_rust._tridiag_eigh_py` 経由で呼び、
  `scipy.linalg.eigh_tridiagonal` と `rel < 1e-13` で一致を確認
- Fuzzing: ランダムシード sweep でクラスタ・退化ケースを smoke test
- 収束失敗は明示的にハンドル (max_iter 超過 → `RuntimeError`)

### 7.2 BLAS 経由のホットパス

複素ベクトル Level-1 / Level-2 BLAS の inline ヘルパ群を `src/blas.rs`
に切り出す:

- `norm2` → `cblas::dznrm2`
- `dot_conj` → `cblas::zdotc_sub`
- `axpy` → `cblas::zaxpy`
- `scal_real` → `cblas::zdscal`
- `gemv_col_major_no_alpha` → `cblas::zgemv`

加えて maqina 固有のホットパス:

- bit-flip pass: site i について 2 元ストライドで stripe-by-stripe に
  進めるカスタムループ。SIMD ((AVX2 / NEON) や level-by-level スワップ
  パターンは v0.1 ではナイーブ実装、bench で頂上を確認してから最適化。
- 対角積: `zdscal` 系の Level-1 BLAS 風に書くのが自然だが、対角が **実数
  ベクトル** なので、`cblas::zdscal` を per-element ループにかける形 (`zaxpy`
  代用) より、Rust scalar loop の方が SIMD される可能性が高い。bench で
  決める。

### 7.3 `apply_h_kinema` の Python 公開

Rust 内では closure として完結させるが、Python リファレンス / テスト
比較のため **公開関数として 2 つ export** する (allocate-and-return と
in-place 版のペア; 後者は Phase 6 follow-up issue #85 で追加):

```rust
// 7.3.a allocate-and-return 版 (Phase 1 から存在).
#[pyfunction]
fn apply_h_kinema_py(
    py: Python<'_>,
    v: PyReadonlyArray1<Complex64>,
    h_x: PyReadonlyArray1<f64>,
    h_p_diag: PyReadonlyArray1<f64>,
    a_t: f64,
    b_t: f64,
) -> PyResult<Py<PyArray1<Complex64>>>;

// 7.3.b in-place 版 (issue #85, Phase 6 follow-up).
#[pyfunction]
fn apply_h_kinema_into_py(
    v: PyReadonlyArray1<Complex64>,
    y_out: PyReadwriteArray1<Complex64>,
    h_x: PyReadonlyArray1<f64>,
    h_p_diag: PyReadonlyArray1<f64>,
    a_t: f64,
    b_t: f64,
) -> PyResult<()>;
```

Python 側で `apply_h_kinema_py(...)` と `(A · h_x ⊗ X) + (B · diag)` を
qutip / 手書きで比べる単体テストを書く (`tests/test_matvec.py`)。両者の
bit-for-bit 一致は `test_apply_h_kinema_into_py_matches_alloc_variant_bitwise`
で固定する。

#### 7.3.1 in-place 版の運用方針 (issue #85)

`apply_h_kinema_py` は呼び出しごとに `dim · 16 B` の `complex128`
array を新規 allocate する。`dim = 2^20` で 16 MB / 1 call, m=64 の Krylov
loop なら 1 回の `lowest_eigenstates(method="lanczos")` で ~1 GB の不要な
heap traffic になる。Phase 6 D の bench 検証 (issue #79) で **Python bench
で観測した N=18 の 0.53× regression は実は alloc/GC noise**, Rust kernel
の micro 効果が完全に埋もれていた、という重要な知見が出た。

そのため **性能を出したい Python 側ループからは in-place 版を使う**:

```python
y_out = np.empty(dim, dtype=np.complex128)  # ループ外で 1 回確保
for ...:
    _rust.apply_h_kinema_into_py(v, y_out, h_x, h_p_diag, a_t, b_t)
    # y_out が H(t)·v で上書きされる
```

`apply_h_kinema` 本体は `y` を **上書き** (additive ではなく) するため、
caller は `np.zeros` ではなく `np.empty` で確保して構わない。

主な call site (issue #85 で移行済み):

| call site | 移行前 alloc / call 上位 |
|---|---|
| `python/maqina/eigenstates.py::_eigenstates_lanczos` | `dim·16 B` × m (=64) |
| `python/maqina/eigenstates.py::_eigenstates_exact` | `dim·16 B` × 2^n |
| `benchmarks/bench_block_fusion.py` | `dim·16 B` × repeat (warmup + 計測域) |
| `benchmarks/bench_simd_scaling.py` | 同上 |
| `benchmarks/bench_parallel_scaling.py` | 同上 |

allocate-and-return 版は **参照実装比較とテスト用** の補助 API として
維持する (既存 docstring 例・テストとの後方互換性のため)。

#### 7.3.2 step 系 `_py` wrap の in-place 入口 (issue #86)

#85 と同じ問題系列の step 系版。Rust 内本番経路 (Lanczos / CFM4 / Trotter
ドライバ) は Python 境界を 1 step ごとに踏まないため影響を受けないが、
**`_py` wrap 経由で per-step Python 越境する Python リファレンス /
fallback (`__has_blas__ = False` 等) / bench script** には alloc/copy が
残っていた。これらに in-place 入口を並立する形で対応 (`*_py` は thin
wrapper として残存):

```rust
// 7.3.2.a Strang 2 次 Trotter 1 step.
#[pyfunction]
fn trotter_step_inplace_py(
    psi: PyReadwriteArray1<Complex64>,
    h_x: PyReadonlyArray1<f64>,
    h_p_diag: PyReadonlyArray1<f64>,
    a_t: f64,
    b_t: f64,
    dt: f64,
    n: usize,
) -> PyResult<()>;

// 7.3.2.b Suzuki S_4 1 step.
#[pyfunction]
fn trotter_suzuki4_step_inplace_py(
    psi: PyReadwriteArray1<Complex64>,
    h_x: PyReadonlyArray1<f64>,
    h_p_diag: PyReadonlyArray1<f64>,
    a_t_list: PyReadonlyArray1<f64>,
    b_t_list: PyReadonlyArray1<f64>,
    dt: f64,
    n: usize,
) -> PyResult<()>;

// 7.3.2.c M2 中点則 1 step (内部で lanczos_propagate を 1 回呼ぶ).
#[pyfunction]
fn m2_midpoint_step_inplace_py(
    psi: PyReadwriteArray1<Complex64>,
    h_x: PyReadonlyArray1<f64>,
    h_p_diag: PyReadonlyArray1<f64>,
    a_mid: f64,
    b_mid: f64,
    dt: f64,
    m: usize,
    krylov_tol: f64,
) -> PyResult<()>;

// 7.3.2.d 2×2 ユニタリを axis i に in-place 適用.
#[pyfunction]
fn apply_single_mode_axis_i_inplace_py(
    psi: PyReadwriteArray1<Complex64>,
    u: PyReadonlyArray1<Complex64>,
    i: usize,
    n: usize,
) -> PyResult<()>;
```

引数順は対応する `*_py` (allocate-and-return) と完全一致 (`psi` を
mutable に変えただけ; 呼び出しコードの移行コストを最小化)。

**性能を出したい Python 側 step loop からは in-place 版を使う**:

```python
# defensive copy で psi0 の破壊を避けつつ owned & C-contiguous buffer に統一.
psi = np.array(psi0, dtype=np.complex128, order="C")
for k in range(n_steps):
    _rust.trotter_step_inplace_py(psi, h_x, h_p_diag, a_mid, b_mid, dt, n)
    # psi が in-place 更新される. 戻り値 None.
```

注意: `np.ascontiguousarray(psi0, dtype=np.complex128)` は psi0 が既に
C-contiguous complex128 なら同じ array を返すため、その後 in-place 更新
を回すと **caller 側の psi0 が破壊される**。defensive copy のために
`np.array(psi0, dtype=np.complex128, order="C")` を使う。

`m2_midpoint_step_inplace_py` 固有の注意: `lanczos_propagate`
(`src/krylov.rs`) は内部で `Vec<Complex64>` の作業バッファを確保するため、
**Rust 内部での `dim · 16 B` alloc は残る**。本 in-place 版が排するのは
Python 境界の `into_pyarray` (numpy buffer alloc + GIL 越え) と
caller 側 `psi = ...` 再代入による参照切り替えコストに限定される。
Trotter 系 (`trotter_step_inplace_py` / `trotter_suzuki4_step_inplace_py`
/ `apply_single_mode_axis_i_inplace_py`) は Rust 内部 alloc も無いため
per-step heap traffic がほぼゼロになる。

主な call site (issue #86 で移行済み):

| call site | 移行前 alloc / call 上位 |
|---|---|
| `python/maqina/krylov.py::evolve_schedule_m2` | `dim·16 B` × step 数 |
| `python/maqina/krylov.py::evolve_schedule_trotter` | `dim·16 B` × step 数 |
| `python/maqina/krylov.py::evolve_schedule_trotter_suzuki4` | `dim·16 B` × step 数 |
| `benchmarks/bench_block_fusion.py` (`_measure_trotter_step`) | `dim·16 B` × repeat |
| `benchmarks/bench_parallel_scaling.py` (`trotter_step` / `apply_single_mode_axis_i_py_sum_diagnostic`) | `dim·16 B` × (1 + n) × repeat |
| `benchmarks/bench_simd_scaling.py` (`_measure_single_mode`) | `dim·16 B` × repeat |

bit-for-bit 一致は
`test_trotter_step_inplace_py_matches_alloc_variant_bitwise` /
`test_trotter_suzuki4_step_inplace_py_matches_alloc_variant_bitwise` /
`test_m2_midpoint_step_inplace_py_matches_alloc_variant_bitwise` /
`test_apply_single_mode_axis_i_inplace_py_matches_alloc_variant_bitwise`
で固定する。

非ゴール (`*_py` の破壊的置換, Lanczos の `lanczos_propagate_py` /
adaptive ドライバの `_py` wrap の in-place 化) は本 issue の scope 外。
後者は内部で多数 step を回し戻り値が状態 1 本のみのため step 系とは
alloc プロファイルが大きく違う (需要が出たら別 issue)。

### 7.4 Cargo features

```toml
[features]
default = ["blas"]
blas = ["dep:cblas", "dep:blas-src"]
extension-module = ["pyo3/extension-module"]
```

`extension-module` を default に入れないのは、`cargo test` で test binary
が `libpython` シンボル未解決になるため。maturin 経由の wheel ビルドでは
`pyproject.toml` の `[tool.maturin] features` で明示的に有効化し、
普通の `cargo test` / `cargo build` では無効のままにする。

### 7.5 `__has_blas__` warning

`_rust.__has_blas__` を Python 側に export し、`maqina.krylov` の
import 時に False なら `RuntimeWarning` を発する。これにより scalar
fallback build (BLAS 無し) に気付かず長時間ベンチを回す事故を防ぐ。

### 7.6 maturin レイアウト上の注意点 (PyO3 stub の歴史的問題)

PyO3 + maturin 構成では過去に **型 stub と拡張モジュールの解決順序**で
詰まる事例が複数報告されていた。現在の maturin (≥ 1.0 系) では大部分が
解消されているが、設計時点で踏むべきは以下:

1. **`python-source` を `"python"` に設定する**

   [maturin#490](https://github.com/PyO3/maturin/issues/490) で報告されている
   `ModuleNotFoundError: No module named 'pkg.pkg'` 系の事故を避ける。
   プロジェクトルートに `maqina/` ディレクトリと拡張モジュール `.so`
   が同居していると、CWD = リポジトリルートで Python を起動した際に
   `maqina/` のソースディレクトリが先に解決され、隣の `_rust.so` が
   見つからない、という症状が出る。Python ソースを `python/maqina/` に
   分離することで CWD と無関係に拡張がロードされる。

2. **`.pyi` は `.py` と同じディレクトリに並べる**

   maturin docs の Project Layout 節は

   > "additional files in the Python source dir (but not in `.gitignore`)
   > will be automatically included in the build outputs"

   と明記しており、`python/maqina/*.pyi` は wheel に自動同梱される。
   `[tool.maturin]` 側に `include` 指定を足す必要はない。

3. **`py.typed` を置く (PEP 561)**

   `python/maqina/py.typed` を空ファイルで作成。これがないと
   `mypy` / `ty` / `pyright` は wheel 同梱の `.pyi` を発見しない。

4. **`.gitignore` には拡張モジュールのみ**

   `python/maqina/_rust*.so` (`maturin develop` 配置先) を `.gitignore`。
   `.pyi` は **コミットする** (自動生成だが diff レビューで API 変更を
   検知できるようにするため)。

5. **古い情報の取り扱い**

   - [maturin#771 (stub が wheel に入らない)](https://github.com/pyo3/maturin/issues/771)
     は古い挙動の報告。現行では Python source dir 配下の `.pyi` は
     自動同梱されるので、本設計では追加対処は不要。
   - [maturin#885 (Python source が wheel に入らない)](https://github.com/PyO3/maturin/issues/885)
     は `python-source` を設定しないと発火する症状。本設計では
     `python-source = "python"` を最初から宣言するため該当しない。

   実機ビルドで `unzip -l dist/maqina-*.whl | grep -E '\.(py|pyi|so)$'`
   を回し `.py` / `.pyi` / `.so` が揃って入っていることを CI で
   smoke test するのが堅い (将来 maturin の挙動が変わっても気付ける)。

これらは Phase 1 の最初のビルドが通った時点で `tests/test_packaging.py`
として固定する想定。

---

