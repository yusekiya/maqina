# testing.md

> **このファイルは移管されました.**
>
> maqina のテスト・lint・build 手順は **project skill
> `.claude/skills/test-runner/SKILL.md`** に集約されています.
> 詳細はそちらを参照してください.

## クイック誘導

| 用途 | 場所 |
|---|---|
| 人間がコマンドを調べたい | `.claude/skills/test-runner/SKILL.md` を直接開く |
| Claude Code セッションから呼ぶ | `/test-runner` skill を発火する |
| 自動テスト実行 (subagent 経由, 並列・context 分離) | `Agent(subagent_type="test-runner", ...)` |

## 最重要だけ抜粋

- Python テスト: `uv run pytest` (必ず `uv run` を付ける)
- Rust テスト: `cargo test` (BLAS on) / `cargo test --no-default-features` (両ブランチで `rel < 1e-13` 一致が契約)
- Rust 変更後の再ビルド: **`uv run maturin develop --uv` を必ず 1 回回してから** `uv run pytest` する (`--uv` フラグが無いと `No module named pip` で失敗)
- pre-commit: `uv run pre-commit run --all-files` (ruff / ty / cargo fmt / clippy / stub drift)

詳細・マーカ・テスト役割分担・CI フロー・既知の落とし穴・並列性ルールは
すべて `.claude/skills/test-runner/SKILL.md` 側にあります.
