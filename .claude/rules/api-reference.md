---
paths:
  - "python/maqina/**/*.py"
  - "python/maqina/**/*.pyi"
  - "tools/gen_api_stubs.py"
  - "tests/**/*.py"
  - "benchmarks/**/*.py"
---

# API リファレンス

`python/maqina/*.pyi` (per-module PEP 484 stub) に公開 API のシグネチャと
**full docstring** がダンプされている。**`maqina` を使うスクリプトを
書く際はまず該当モジュールの `.pyi` を読み、必要に応じてソース実装を
参照する** (cv_ising と同方式)。`.pyi` は手書きしない。再生成:

```bash
uv run python tools/gen_api_stubs.py
```

`.pyi` ドリフト防止は二段階:

1. **Claude 編集時 (一次)**: `.claude/rules/api-stubs-sync.md` (path-scoped rule)
   が `python/maqina/**/*.py` または `tools/gen_api_stubs.py` 編集時にロード
   され、再生成スクリプトを同じコミットに含めるよう Claude 側で運用する。
2. **コミット時 (セーフティネット)**: `.pre-commit-config.yaml` の `gen-api-stubs`
   フックが人間の手編集も含めて取りこぼしを拾う。
