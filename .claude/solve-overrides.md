# solve overrides for kryanneal

`/solve` skill の達成基準・権限境界に対するプロジェクト固有の追記。skill 側の基底規約を override するものではなく、delta を追加するためのファイル。

## ドキュメント整合の対象

- `docs/design.md`: 一次設計書。アーキテクチャ・公開 API・数値カーネル・段階リリース計画 (Phase 1–6) を集約する。設計に関わる変更 (公開 API 追加・カーネル差し替え・新フェーズの確定など) は同コミットに含める。
- `docs/testing.md`: テスト実行コマンド・マーカ規約・CI 想定。テスト規約や実行方法を変えたら同コミットに反映する。
- `docs/benchmarks.md` (存在する場合): 性能計測手順や結果の参照規約。
- `CLAUDE.md`: プロジェクトガイド (Claude 向け)。レイアウト・コマンド・物理的取り決めが変わったら同コミットで更新する。
- 既存コードの docstring 慣習: 数式や物理的意味を持つ変数は **日本語** docstring で意味を記述する (cv_ising 流)。`python/kryanneal/__init__.py` の既存 docstring を参考にする。
- 自動生成スタブ: `python/kryanneal/*.pyi` は `tools/gen_api_stubs.py` で再生成。`python/kryanneal/**/*.py` または `tools/gen_api_stubs.py` を変更したら **同コミットに再生成済み `.pyi` を含める**。`.pre-commit-config.yaml` の `gen-api-stubs` フックがセーフティネットとして drift を検出する。

## テスト実行規約

詳細手順は `docs/testing.md` を一次資料とする。下記は要点のみ。

### 実行経路 (default: test-runner subagent 経由)

長い `passed` 列や verbose 出力でメイン context を圧迫しないよう, **テスト・lint・maturin develop の実行は原則 `test-runner` subagent (`.claude/agents/test-runner.md`) に委譲する**。本 agent は `Bash` / `Read` のみ持つ read-only ランナーで, pass/fail サマリと失敗時の末尾 stdout 抜粋だけを返す.

並列実行の方針:

- `cargo test` (BLAS on) と `uv run pytest` は **独立** (前者は `target/`, 後者は既存 `_rust.so` を読むだけ) のため, メイン側から 2 つの test-runner agent を **同時起動** して並列化してよい.
- `cargo test` (BLAS on) と `cargo test --no-default-features` は **同じ `target/` のロックを争うため実質シリアル化** する. 並列起動するメリットは無いので順次実行する.
- `uv run maturin develop --uv` と `uv run pytest` は **serialize 必須**. `_rust.so` 上書き中に pytest がロードすると ABI 不整合になる. maturin が完了してから pytest を起動する.

直接 Bash で `cargo test` 等を叩くのは, **agent 起動オーバヘッドのほうが重い極小タスク** (単一テストの再実行など) や, **失敗の生 stdout を逐次見たいデバッグ局面** に限定する.

### Python (uv + pytest)

- 通常実行: `uv run pytest`
- 高速ループ用 (重いテスト除外): `uv run pytest -m "not slow"`
- 個別ファイル: `uv run pytest tests/test_<name>.py`
- `slow` マーカー: `@pytest.mark.slow` を付与した重い統合テスト (QuTiP 比較・大規模 n など) は CI / 短時間開発で除外可。

### Rust (cargo)

- BLAS feature ON (default): `cargo test`
- scalar fallback: `cargo test --no-default-features`
- 両ブランチで `rel < 1e-13` 一致が契約。差分が出たら回帰扱い。
- 注: `Cargo.toml` がリポジトリルートにある (maturin 標準レイアウト)。`cd` 不要で `cargo test` を打つ。

### 横断 (pre-commit)

- `pre-commit run --all-files`: ruff / ty / cargo fmt / cargo clippy / gen-api-stubs ドリフト検査。コミット前に必ず通す。

### Rust 変更時の必須手順

- `src/*.rs` 変更後は `uv run pytest` を回す前に **`uv run maturin develop --uv` を必ず 1 回回す**。忘れると古い `_rust.so` が読まれて Rust 変更がテストに反映されない (`docs/testing.md` の「既知の落とし穴」参照)。
- **`--uv` フラグは必須**。`uv` 製の venv には pip が同梱されないため、`--uv` なしの `uv run maturin develop` は `No module named pip` で失敗する。`--uv` を渡すと maturin が `pip install` ではなく `uv pip install` を使う (詳細は `docs/testing.md` §ビルド前提)。

## 自動化モード stop conditions

`autonomous` ラベル付き issue の処理中、以下に該当する変更が必要と判明した時点で skill の中止プロトコルへ移行する。

1. **公開 API の破壊的変更**: `python/kryanneal/__init__.py` の `__all__` に含まれる名前、または `IsingProblem` / `Schedule` / `QuantumAnnealer` / `QuantumResult` / `Trajectory` 等の公開シグネチャ・セマンティクスの非互換変更。バージョン bump を伴う変更も同様。
2. **数値計算の再現性・精度に影響する変更**: Lanczos / CFM4:2 / M2 / Trotter / Richardson の係数・演算順序・収束判定・PI controller 既定値、`apply_h_kryanneal` の bit-flip 規約、BLAS feature on/off 経路の数値一致 (`rel < 1e-13`) を壊しうる変更、デフォルト dt / max_m / tol などの既定値変更。

将来追加候補 (現時点では未定義):

- インフラ・CI ワークフローの破壊的変更 (Phase 1 では `.github/workflows/` 自体が未整備)
- リリース手順・wheel 公開先・依存バージョン制約の変更

## Post-apply actions

現時点で PR マージ後の手動オペは無し (`.github/workflows/` 未整備のため CI トリガもない)。Phase 1 で CI を導入し次第、必要に応じて本節を更新する。
