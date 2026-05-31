---
paths:
  - "python/maqina/**/*.py"
  - "tools/gen_api_stubs.py"
---

# `.pyi` stub の同一コミット同期 (Claude 編集時の一次運用)

`python/maqina/**/*.py` または `tools/gen_api_stubs.py` を編集して公開 API の
シグネチャ・docstring を変えたときは, **同じコミット内で `.pyi` stub を再生成
して含める**。`.pyi` は手書きしない (自動生成のみ)。

```bash
uv run python tools/gen_api_stubs.py
```

ドリフト防止は二段階で, この rule はその一次 (Claude 編集時) を担う:

1. **Claude 編集時 (一次, この rule)**: 上記 path を触ったら再生成スクリプトを
   走らせ, 生成された `python/maqina/*.pyi` を同じコミットに含める。
2. **コミット時 (セーフティネット)**: `.pre-commit-config.yaml` の `gen-api-stubs`
   フックが人間の手編集も含めて取りこぼしを拾う。

詳細な API リファレンスの読み方は `.claude/rules/api-reference.md` を参照。
