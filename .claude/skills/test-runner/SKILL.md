---
name: test-runner
description: kryanneal のテスト・lint・build (`uv run pytest` / `cargo test` BLAS on/off / `uv run pre-commit run --all-files` / `uv run maturin develop --uv`) を走らせるときに参照する一次資料。コマンド・マーカ・テスト役割分担・CI フロー・既知の落とし穴を集約する。kryanneal 固有の `--uv` フラグや BLAS feature 切替などの手順をここに置く。`/test-runner` で発火可能。`test-runner` subagent (`.claude/agents/test-runner.md`) もここを `Read` してから実行する。
---

# test-runner skill

`kryanneal` のテスト実行手順を集約した一次資料 (skill body + 補助コマンド表).

## 役割と他資料との関係

- **本 skill (`/test-runner`)**: 人間 / メイン Claude / subagent の共通参照先. 単一情報源.
- **`test-runner` subagent (`.claude/agents/test-runner.md`)**: 並列実行とメイン context 削減のためテスト実行を引き受けるランナー. 起動時に本 SKILL を `Read` してから動く.
- **user-level `/test` skill** (汎用): どのプロジェクトでも `docs/testing.md` を読む実装. kryanneal では `docs/testing.md` が本 skill へのポインタなので、結局ここに辿り着く.

## クイックリファレンス

| やりたいこと | コマンド |
|---|---|
| 全 Python テスト | `uv run pytest` |
| slow を除外 | `uv run pytest -m "not slow"` |
| 個別ファイル | `uv run pytest tests/test_krylov.py` |
| 名前パターン | `uv run pytest -k "richardson"` |
| `print` 出力を表示 | `uv run pytest -s` |
| Rust 単体 (BLAS + rayon ON = default) | `cargo test` |
| Rust 単体 (scalar fallback, BLAS + rayon OFF) | `cargo test --no-default-features` |
| Rust 単体 (BLAS のみ, rayon OFF) | `cargo test --no-default-features --features blas` |
| Rust 単体 (rayon のみ, BLAS OFF) | `cargo test --no-default-features --features rayon` |
| Lint | `uv run ruff check .` |
| 型チェック | `uv run ty check python/kryanneal` |
| Stub 再生成 | `uv run python tools/gen_api_stubs.py` |
| 全 pre-commit フック | `uv run pre-commit run --all-files` |
| Wheel パッケージング smoke | `uv run pytest tests/test_packaging.py` |

> 注: `cargo test` は **プロジェクトルートで** 実行する (maturin 標準
> レイアウトでは `Cargo.toml` がリポジトリルートにあるため). `cd` 不要.

## ビルド前提

Rust 拡張 `kryanneal._rust` は `maturin develop` で `python/kryanneal/` 配下に
配置される. 初回および Rust 変更後は以下:

```bash
uv run maturin develop --uv                  # 既定 (debug 相当)
uv run maturin develop --uv --release        # 性能計測時
uv run maturin develop --uv --release --no-default-features   # BLAS なし fallback build
```

`uv run maturin develop --uv` を **`uv run pytest` の前に必ず 1 回回す**.
忘れると古い `_rust.so` が読まれて Rust 変更がテストに反映されない.

`--uv` フラグは, maturin が wheel を `pip install` する代わりに `uv pip
install` を使う指定. uv が作る venv には pip が同梱されないため,
`--uv` 無しだと `No module named pip` で失敗する.

## マーカ

- `@pytest.mark.slow`: 重い統合テスト (`QuantumAnnealer.run` の中規模 n,
  QuTiP 大規模比較等). CI / 短時間開発では `-m "not slow"` で除外可能.

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

`src/*.rs` 内に inline `#[cfg(test)] mod tests` として書く. BLAS / rayon
feature on/off の組合せを別々に網羅:

```bash
cargo test                                          # blas + rayon ON (default)
cargo test --no-default-features                    # 両方 OFF (scalar 単スレッド)
cargo test --no-default-features --features blas    # blas のみ
cargo test --no-default-features --features rayon   # rayon のみ
```

rayon 経路 (`apply_h_kryanneal_rayon` / `apply_single_mode_axis_i_rayon`,
issue #62) の決定性テスト (`apply_*_rayon_determinism_8thread_100iter`) は
`#[cfg(feature = "rayon")]` で gate されているため rayon ON ビルドでのみ
実行される.

カバー対象:

- `blas.rs` の各ヘルパ (`norm2` / `dot_conj` / `axpy` / `scal_real` /
  `gemv_col_major_no_alpha`) の数値挙動
- `lanczos_propagate` の closure ベース版 (zero psi, H = 0, 対角 H の
  位相回転, Hermitian H でのノルム保存)
- `apply_h_kryanneal` の matvec が dense 構築版と一致

`Python<'py>` 引数に依存する PyO3 wrapper (`#[pyfunction]`) は `cargo test`
の対象外. Python 側 pytest で間接的にカバーする.

### Python 統合テスト (`uv run pytest`)

`tests/test_*.py`. pytest ベース.

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

3 種のペアテストを軸に, 新しい propagator を増やすたびに同じパターンの
ペアテストを追加する:

1. Python リファレンス vs Rust (`rel < 1e-13`)
2. Rust vs QuTiP sesolve (fidelity `> 1 - 1e-8`)
3. Krylov vs exact `eigh` (小規模 n でのみ)

## pre-commit

`.pre-commit-config.yaml` で以下を走らせる. 全フックは `local` で uv-managed
venv のツールを呼ぶ (cv_ising と同方針):

| フック | 走るもの |
|---|---|
| `ruff-check` | `uv run ruff check --force-exclude` (Python ソース変更時) |
| `ruff-format` | `uv run ruff format --force-exclude` (Python ソース変更時) |
| `ty-check` | `uv run ty check python/kryanneal` (`.py`/`.pyi` 変更時, project mode) |
| `cargo-fmt` | `cargo fmt --check` (Rust ソース変更時) |
| `cargo-clippy` | `cargo clippy --all-targets -- -D warnings` (Rust ソース変更時) |
| `gen-api-stubs` | `uv run python tools/gen_api_stubs.py` (`python/kryanneal/**/*.py` または `tools/gen_api_stubs.py` 変更時) |

セットアップ:

```bash
uv run pre-commit install          # git commit 時に自動実行
uv run pre-commit run --all-files  # 全ファイルに対して手動実行
```

`gen-api-stubs` フックは `.pyi` を再生成して差分があればコミットを止める
セーフティネット. **通常は Claude / 人間が `.py` を編集したコミットに
再生成した `.pyi` も同梱する運用** (`CLAUDE.md` 「ドリフト防止は二段階」
節). pre-commit が落ちたら `git add python/kryanneal/*.pyi` で取り込み再 commit.

## CI

`.github/workflows/ci.yml` で以下を走らせる想定 (Phase 1 で整備):

1. `uv sync` → `uv run maturin develop --uv`
2. `uv run pre-commit run --all-files` (ruff / ty / cargo fmt / clippy / stub drift)
3. `cargo test` (BLAS feature ON)
4. `cargo test --no-default-features` (scalar fallback)
5. `uv run maturin develop --uv --no-default-features` → `uv run pytest tests/test_packaging.py` (BLAS off build の wheel smoke)
6. `uv run maturin develop --uv` → `uv run pytest -m "not slow"`
7. (release tag 時のみ) `uv run pytest` (slow を含む)

## 既知の落とし穴

- **`maturin develop` 忘れ**: Rust 変更後に `uv run pytest` だけ走らせると
  古い `_rust.so` が読まれる. Rust 変更を伴うコミットの前は必ず再ビルド.
- **`maturin develop` の `--uv` フラグ忘れ**: `uv run maturin develop` を
  `--uv` 無しで叩くと `No module named pip` で失敗する. `uv` 製 venv には
  pip が同梱されておらず, maturin の default 経路 (`pip install` で wheel を
  入れる) がそのままでは使えないため. **必ず `uv run maturin develop --uv`**
  と書く (§ビルド前提のコマンド例参照).
- **`uv` 経由でない Python**: システム Python で `pytest` を直接叩くと
  ABI 不整合で `_rust.so` ロード失敗する. 常に `uv run`.
- **BLAS feature 切替**: `cargo test` と `cargo test --no-default-features`
  で結果が `rel > 1e-13` ずれる場合は要調査 (本来両経路で 1e-13 以内に
  一致するはず, cv_ising と同じ契約).
- **clippy `needless_range_loop`**: `for i in 0..n { ... arr[i] ... }` の
  ような「インデックス変数で別スライスを引く」形式は clippy が reject
  する (`-D warnings` 運用のため即 build fail). `docs/design.md` の擬似
  コードはインデックス記法で書かれているが, 実装では
  `for (i, &x) in arr.iter().enumerate()` に書き換える. bit-flip 系のように
  `i` を `1 << i` などビット演算でも使う場合, `enumerate()` 経由の `i` を
  そのまま使えば一石二鳥.
- **pre-commit auto-fix と再 stage**: `ruff format` と `cargo fmt` の hook
  はファイルを **自動修正してから commit を止める** 挙動. `git commit` が
  hook で落ちたら, 修正されたファイルを `git add` し直してから再 commit
  する. `gen-api-stubs` も同じ運用 (`.pyi` 再生成節参照).

## 並列実行の方針 (subagent 経由で走らせるとき)

`test-runner` subagent はメインから複数 agent を同時起動する形で並列化する.
ペアごとの並列可否は:

- ✅ `cargo test` (BLAS on) + `uv run pytest`: **並列可**. cargo は `target/`,
  pytest は既存 `_rust.so` を読むだけで独立.
- ⚠️ `cargo test` (BLAS on) + `cargo test --no-default-features`: **実質シリアル**.
  同じ `target/` のロックを争うため. `CARGO_TARGET_DIR` を分けるか worktree
  isolation が必要. デフォルトでは順次実行する.
- ❌ `uv run maturin develop --uv` + `uv run pytest`: **serialize 必須**.
  `_rust.so` 上書き中に pytest がロードすると ABI 不整合.

## 関連資料

- `.claude/agents/test-runner.md`: 本 skill を読んで実行する subagent.
- `CLAUDE.md`: プロジェクト全体ガイド (テストは本 skill を参照).
- `.claude/solve-overrides.md`: `/solve` skill の kryanneal 固有 delta.
- `docs/design.md`: 数値カーネル設計 (テスト等価性ペアの定義もここ).
