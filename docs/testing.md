# testing.md

`/test` skill が読みに来る一次資料。`kryanneal` のテスト実行方法を集約する。

## クイックリファレンス

| やりたいこと | コマンド |
|---|---|
| 全 Python テスト | `uv run pytest` |
| slow を除外 | `uv run pytest -m "not slow"` |
| 個別ファイル | `uv run pytest tests/test_krylov.py` |
| 名前パターン | `uv run pytest -k "richardson"` |
| `print` 出力を表示 | `uv run pytest -s` |
| Rust 単体 (BLAS feature ON) | `cargo test` |
| Rust 単体 (scalar fallback) | `cargo test --no-default-features` |
| Lint | `uv run ruff check .` |
| 型チェック | `uv run ty check python/kryanneal` |
| Stub 再生成 | `uv run python tools/gen_api_stubs.py` |
| Wheel パッケージング smoke | `uv run pytest tests/test_packaging.py` |

> 注: `cargo test` は **プロジェクトルートで** 実行する (maturin 標準
> レイアウトでは `Cargo.toml` がリポジトリルートにあるため)。cv_ising は
> `cd rust && cargo test` だったが、kryanneal は `cd` 不要。

## ビルド前提

Rust 拡張 `kryanneal._rust` は `maturin develop` で `python/kryanneal/` 配下に
配置される。初回および Rust 変更後は以下:

```bash
uv run maturin develop                  # 既定 (debug 相当)
uv run maturin develop --release        # 性能計測時
uv run maturin develop --release --no-default-features   # BLAS なし fallback build
```

`uv run maturin develop` を **`uv run pytest` の前に必ず 1 回回す**。
忘れると古い `_rust.so` が読まれて Rust 変更がテストに反映されない。

## マーカ

- `@pytest.mark.slow`: 重い統合テスト (`QuantumAnnealer.run` の中規模 n、
  QuTiP 大規模比較等)。CI / 短時間開発では `-m "not slow"` で除外可能。

`pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
markers = [
    "slow: 重い統合テスト. pytest -m 'not slow' で除外可",
]
```

## テストの役割分担

### Rust 単体テスト (`cargo test`)

`src/*.rs` 内に inline `#[cfg(test)] mod tests` として書く。BLAS feature
on/off の両ブランチを別々に網羅:

```bash
cargo test                       # blas feature ON (default)
cargo test --no-default-features # scalar fallback
```

カバー対象:

- `blas.rs` の各ヘルパ (`norm2` / `dot_conj` / `axpy` / `scal_real` /
  `gemv_col_major_no_alpha`) の数値挙動
- `lanczos_propagate` の closure ベース版 (zero psi, H = 0, 対角 H の
  位相回転, Hermitian H でのノルム保存)
- `apply_h_kryanneal` の matvec が dense 構築版と一致

`Python<'py>` 引数に依存する PyO3 wrapper (`#[pyfunction]`) は `cargo test`
の対象外。Python 側 pytest で間接的にカバーする。

### Python 統合テスト (`uv run pytest`)

`tests/test_*.py`。pytest ベース。

| ファイル | カバー範囲 |
|---|---|
| `test_problem.py`           | `IsingProblem` の入力検証 (shape / dtype / NaN / 非正規化 psi0 拒否) |
| `test_schedule.py`          | `Schedule.linear` / `Schedule.from_callable` の境界値 |
| `test_builders.py`          | `diag_from_pauli_terms` / `diag_from_J_h` が手計算と一致 |
| `test_matvec.py`            | Rust `apply_h_kryanneal` が dense 構築版と一致 |
| `test_krylov.py`            | Python リファレンス vs Rust 実装 の `rel < 1e-13` |
| `test_cfm4.py`              | time-independent H で `exp(-i T H) · psi0` と一致 |
| `test_richardson.py`        | 既知 schedule に対する dt 自動調整の収束 |
| `test_annealer.py`          | 公開 API スモーク (linear schedule, GS 到達確率) |
| `test_eigenstates.py`       | 小規模 n=4 で dense `eigh` と Lanczos 結果が一致 |
| `test_reference_qutip.py`   | QuTiP sesolve との fidelity 比較 (`@pytest.mark.slow`) |
| `test_packaging.py`         | wheel に `.py` / `.pyi` / `_rust.*.so` が同梱されているか smoke |

### 等価性ペアの規約

3 種のペアテストを軸に、新しい propagator を増やすたびに同じパターンの
ペアテストを追加する:

1. Python リファレンス vs Rust (`rel < 1e-13`)
2. Rust vs QuTiP sesolve (fidelity `> 1 - 1e-8`)
3. Krylov vs exact `eigh` (小規模 n でのみ)

## CI

`.github/workflows/ci.yml` で以下を走らせる想定 (Phase 1 で整備):

1. `uv sync` → `uv run maturin develop`
2. `uv run ruff check .`
3. `uv run ty check python/kryanneal`
4. `cargo test` (BLAS feature ON)
5. `cargo test --no-default-features` (scalar fallback)
6. `uv run pytest -m "not slow"`
7. (release tag 時のみ) `uv run pytest` (slow を含む)

## 既知の落とし穴

- **`maturin develop` 忘れ**: Rust 変更後に `uv run pytest` だけ走らせると
  古い `_rust.so` が読まれる。Rust 変更を伴うコミットの前は必ず再ビルド。
- **`uv` 経由でない Python**: システム Python で `pytest` を直接叩くと
  ABI 不整合で `_rust.so` ロード失敗する。常に `uv run`。
- **BLAS feature 切替**: `cargo test` と `cargo test --no-default-features`
  で結果が `rel > 1e-13` ずれる場合は要調査 (本来両経路で 1e-13 以内に
  一致するはず、cv_ising と同じ契約)。
