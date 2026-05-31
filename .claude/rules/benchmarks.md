---
paths:
  - "benchmarks/**/*.py"
  - "scripts/**/*.sh"
---

# ベンチマーク

`benchmarks/` 配下に per-step 性能計測の CLI スクリプトを置く。

```bash
uv run python benchmarks/bench_per_step.py
uv run python benchmarks/bench_blas_compare.py   # BLAS feature on/off 同一マシン比較
uv run python benchmarks/bench_vs_qutip.py
uv run python benchmarks/bench_qutip_large.py    # work-precision diagram で QuTiP vs maqina を Pareto 比較 (issue #65)
```

性能改善の主張をするときの方法 (cv_ising 流):

- 「○○× 速くなった」という主張は **同一マシン上の before / after** で示す。
  CPU / BLAS バックエンド / NumPy バージョン / 熱状態が揃っている必要がある。
- BLAS feature on/off の比較は `bench_blas_compare.py` を使う (どの
  ハードウェアでも有効)。
- それ以外の性能変更 (アルゴリズム差し替え等) では `git stash` または
  `git switch` で実装を切り替えつつ `bench_per_step.py` を 2 回回し、
  自前で per-cell 比較表を作る。
- 結果は `benchmarks/results/<YYYYMMDD-HHMMSS>/` に CSV + markdown を残す
  (gitignored)。書き戻すときはハード (機種 / チップ / メモリ / OS / NumPy /
  BLAS backend) を節タイトルで明示する。
