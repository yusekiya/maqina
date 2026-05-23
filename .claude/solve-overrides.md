# solve overrides for kinema

`/solve` skill の達成基準・権限境界に対するプロジェクト固有の追記。skill 側の基底規約を override するものではなく、delta を追加するためのファイル。

## ドキュメント整合の対象

- `docs/design/INDEX.md`: 一次設計書。アーキテクチャ・公開 API・数値カーネル・段階リリース計画 (Phase 1–6) を集約する。設計に関わる変更 (公開 API 追加・カーネル差し替え・新フェーズの確定など) は同コミットに含める。
- `docs/conventions.md`: 開発規約 (ビルド基盤・バージョニングポリシー・umbrella issue DoD 必須項目)。ツールチェイン更新・version bump タイミング・新 Phase 起票ルールを変えたら同コミットに反映する。
- `.claude/skills/test-runner/SKILL.md`: テスト・lint・build 実行手順の一次資料。テスト規約や実行方法を変えたら同コミットに反映する (`docs/testing.md` はポインタなので drift しない)。
- `docs/benchmarks.md` (存在する場合): 性能計測手順や結果の参照規約。
- `CLAUDE.md`: プロジェクトガイド (Claude 向け)。レイアウト・コマンド・物理的取り決めが変わったら同コミットで更新する。
- 既存コードの docstring 慣習: 数式や物理的意味を持つ変数は **日本語** docstring で意味を記述する (cv_ising 流)。`python/kinema/__init__.py` の既存 docstring を参考にする。
- 自動生成スタブ: `python/kinema/*.pyi` は `tools/gen_api_stubs.py` で再生成。`python/kinema/**/*.py` または `tools/gen_api_stubs.py` を変更したら **同コミットに再生成済み `.pyi` を含める**。`.pre-commit-config.yaml` の `gen-api-stubs` フックがセーフティネットとして drift を検出する。

## テスト実行規約

詳細手順は project skill `.claude/skills/test-runner/SKILL.md` を一次資料とする (`/test-runner` で skill 発火, `test-runner` subagent もここを読む)。実行経路 (default: test-runner subagent 経由) と並列化方針は **`CLAUDE.md` の「テスト → 実行経路」節** に集約済み (常時ロードされ `/solve` 経由か否かに関わらず適用される)。下記は要点のみ。

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

- `src/*.rs` 変更後は `uv run pytest` を回す前に **`uv run maturin develop --uv` を必ず 1 回回す**。忘れると古い `_rust.so` が読まれて Rust 変更がテストに反映されない (`.claude/skills/test-runner/SKILL.md` の「既知の落とし穴」参照)。
- **`--uv` フラグは必須**。`uv` 製の venv には pip が同梱されないため、`--uv` なしの `uv run maturin develop` は `No module named pip` で失敗する。`--uv` を渡すと maturin が `pip install` ではなく `uv pip install` を使う (詳細は `.claude/skills/test-runner/SKILL.md` §ビルド前提)。

## 自動化モード stop conditions

`autonomous` ラベル付き issue の処理中、以下に該当する変更が必要と判明した時点で skill の中止プロトコルへ移行する。

1. **公開 API の破壊的変更**: `python/kinema/__init__.py` の `__all__` に含まれる名前、または `IsingProblem` / `Schedule` / `QuantumAnnealer` / `QuantumResult` / `Trajectory` 等の公開シグネチャ・セマンティクスの非互換変更。バージョン bump を伴う変更も同様。
2. **数値計算の再現性・精度に影響する変更**: Lanczos / CFM4:2 / M2 / Trotter / Richardson の係数・演算順序・収束判定・PI controller 既定値、`apply_h_kinema` の bit-flip 規約、BLAS feature on/off 経路の数値一致 (`rel < 1e-13`) を壊しうる変更、デフォルト dt / max_m / tol などの既定値変更。

将来追加候補 (現時点では未定義):

- インフラ・CI ワークフローの破壊的変更 (Phase 1 では `.github/workflows/` 自体が未整備)
- リリース手順・wheel 公開先・依存バージョン制約の変更

## Bench を伴う issue の運用 (pre-merge bench cycle)

性能 acceptance を持つ issue (例: Phase 6 C2 #63, C3 #64, C4 #65, C5 #66, C2.5 #71 など) は **PR branch を Linux サーバーで checkout して bench を取り, acceptance pass を確認してから merge する pre-merge 運用** を取る (#63 で確定した更新版; 2026-05-16)。

### 役割分担 (重要)

| 操作 | 担当 |
|---|---|
| bench コマンドの実行 (`uv run python benchmarks/bench_*.py ...`) | **ユーザー** (Linux サーバー上で手動実行) |
| Linux サーバーで `gh pr checkout <PR#>` で PR branch を取得 | **ユーザー** |
| bench 結果 (markdown / CSV) の Claude への共有 | **ユーザー** (チャットに貼り付け) |
| 結果分析と修正方針の決定 | **Claude** |
| 実装・テスト・docs 整合の commit を同一 PR branch に追加 push | **Claude** |
| `gh pr comment <PR#> --body-file <result.md>` で bench 結果を PR に添付 | **ユーザー** |
| `gh pr merge <PR#>` で PR を merge | **ユーザー** |

**Claude は bench コマンドを直接実行しない**。 ローカルが macOS なら NEON fallback で数値同一性確認の smoke 程度は OK だが, acceptance 判定 (Linux x86_64 + AVX) は必ずユーザーに依頼する。 `gh pr merge` も Claude 側で先回りしない。

### フロー

1. Claude が PR を push したら **merge を急がず, ユーザーに「Linux サーバーで `gh pr checkout <PR#>` で PR branch を取得して bench をかけてください」と明示的に依頼する** (具体的な bench コマンドサンプルを併記)
2. ユーザーが bench を実行し, 結果をチャットに貼り付けて共有
3. acceptance 未達なら: Claude が結果を分析し, 修正 commit を同一 PR branch に push (新規 PR は立てない) → 1. に戻る
4. acceptance pass 後:
   - ユーザーが `gh pr comment <PR#> --body-file <result.md>` で bench 結果を PR コメントに添付
   - ユーザーが `gh pr merge <PR#>` で merge
5. merge 後の手動オペは無し (`.github/workflows/` 未整備のため CI トリガもない)

旧運用「bench は PR 本体に含めず merge 後にコメント添付」(issue #47 で当初確定) は #63 で **4 PR / 2 weeks の試行錯誤が分散発生** する問題を起こしたため差し替え。 1 issue = 1 PR で完結する pre-merge 運用に統一する。

非 perf な issue (公開 API 追加・bug fix・ドキュメント整備など bench acceptance を持たない issue) では本節は適用外で, 通常通り PR push → 即 merge で OK (merge 自体はユーザー操作)。

## Post-apply actions

現時点で PR マージ後の手動オペは無し (`.github/workflows/` 未整備のため CI トリガもない)。Phase 1 で CI を導入し次第、必要に応じて本節を更新する。
