# §11. 開発・ビルド基盤

→ `docs/conventions.md` §1 参照. (uv / maturin / ruff / ty / pre-commit /
gen_api_stubs ドリフト二段運用 / BLAS 多プロセス制御の export.)

---

## §11.1 リポジトリ同梱の最適化ビルド設定 (issue #103)

`uv add git+...` 経由でユーザーがソースビルドする運用前提 (現状 wheel 配布は
無い) のため, 環境変数の設定をユーザーに要求せずに最適化ビルドが default で
適用される設計を取る. 関連設定は 3 ファイルに分散している:

| ファイル | 設定 | 効果 |
|---|---|---|
| `Cargo.toml::[profile.production]` | `inherits = "release"`, `codegen-units = 1`, `lto = "fat"`, `panic = "abort"` | `release` を強化した最適化プロファイルを定義 |
| `pyproject.toml::[tool.maturin]` | `profile = "production"`, `strip = true` | maturin (PEP 517 backend) が cargo に `--profile production` で渡し, シンボル strip も有効化 |
| `.cargo/config.toml::[build] rustflags` | `["-C", "target-cpu=native"]` | rustc に `-C target-cpu=native` を自動付与し SIMD feature を build マシン CPU 限度まで有効化 |

### `uv add git+...` から rustc まで

1. uv → pip → PEP 517 backend (maturin)
2. maturin が `pyproject.toml::[tool.maturin] profile = "production"` を読む
3. maturin が `cargo build --profile production --features extension-module`
   を invoke (CWD = clone 先 repo root)
4. cargo が `<repo_root>/.cargo/config.toml::[build] rustflags = ["-C",
   "target-cpu=native"]` を merge し, 全 rustc 呼び出しに付与する

このフローにより, ユーザー側で `RUSTFLAGS` を設定しなくても
`production` profile + `target-cpu=native` の組み合わせが効いた状態の
`kinema._rust.*.so` が install される.

### 既知の注意点

- 生成される拡張は **build マシン専用バイナリ**. 別 CPU マシンへ wheel を
  再配布する用途には不向き (現状 wheel 配布なし方針なので問題なし). portable な
  build を要するときは `RUSTFLAGS=" "` 等で override する.
- `maturin build --release` / `uv run maturin develop --uv --release` を明示
  すると `[tool.maturin] profile = "production"` を override して `release`
  profile になり, production の `lto = "fat"` / `panic = "abort"` が効かない.
  PEP 517 経由の `uv add git+...` / `uv pip install .` は `--release` を渡さ
  ないので production が選ばれる. 開発時 `maturin develop --release` は速度
  目的なら充分 (lto=thin) なので実害は薄いが, 厳密な production と等価な
  build を取りたい場合は `--profile production` を明示する.
- user 側で `RUSTFLAGS` env var を set している場合は env が優先されて
  `.cargo/config.toml::[build] rustflags` は無視される (cargo の rustflags
  precedence). 想定挙動.

### ビルド構成の dump: `kinema.show_config()`

「target-cpu=native が本当に効いたか」「どの cargo feature でビルドしたか」を
ユーザー側で確認するための numpy 風ヘルパ. 既存の `_rust.__has_blas__`
パターン (compile-time const + `m.add` で expose) をそのまま延長し,
**build.rs (build script) は導入しない**.

`_rust` モジュールに expose する compile-time フラグ:

| 属性 | 由来 | 例 |
|---|---|---|
| `__has_blas__` | `cfg!(feature = "blas")` | True (default) |
| `__has_rayon__` | `cfg!(feature = "rayon")` | True (default) |
| `__has_simd__` | `cfg!(feature = "simd")` | True (default) |
| `__has_avx2__` | `cfg!(target_feature = "avx2")` | True (Zen 3 以降 + native), False (default x86_64) |
| `__has_fma__` | `cfg!(target_feature = "fma")` | True (Zen 3 以降 + native) |
| `__has_avx512f__` | `cfg!(target_feature = "avx512f")` | True (Zen 4 / SPR + native) |
| `__has_neon__` | `cfg!(target_feature = "neon")` | True (aarch64 default), False (x86_64) |
| `__target_arch__` | `std::env::consts::ARCH` | "x86_64" / "aarch64" |
| `__target_os__` | `std::env::consts::OS` | "linux" / "macos" |

Python 側 `kinema.show_config()` は上記を集約し pretty-print する:

```python
>>> import kinema
>>> kinema.show_config()
kinema build configuration
--------------------------------------------------
  version       : 0.8.0
  target arch   : x86_64
  target OS     : linux

  cargo features:
    BLAS  : True
    rayon : True
    SIMD  : True

  target_features (-C target-cpu=native の効きを反映):
    [ON ] avx2
    [ON ] fma
    [off] avx512f
    [off] neon
```

`target-cpu=native` が効いていない (例: `RUSTFLAGS=""` で override された)
build では `avx2` / `fma` も `off` と表示されるので「設定漏れ」が一目で判別
できる. bench を取る前の sanity check, あるいは issue report 添付情報として
使用する.

### build.rs を採用しない理由

target_feature の有無は `cfg!(target_feature = "...")` macro で compile-time
に評価できるため build script は不要. 既存の `__has_blas__` 等と同じパターン
で揃えられる. 将来「全 target_feature 文字列を生で見たい」要件が出てきたとき
に初めて build.rs を導入する余地は残しておく.
