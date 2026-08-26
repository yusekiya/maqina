---
paths:
  - "benchmarks/**/*.py"
  - "scripts/**/*.sh"
---

# ベンチマーク

`benchmarks/` 配下に per-step 性能計測の CLI スクリプトを置く。

```bash
uv run python benchmarks/bench_per_step.py
uv run python benchmarks/bench_qutip_large.py    # work-precision diagram で QuTiP vs maqina を Pareto 比較 (issue #65)
```

スクリプトの全一覧・各 CLI の引数・導入 Phase は `benchmarks/README.md` が
一次資料。**未実装**のものも「導入予定 Phase」付きで表に載っているので、
実行例を書き写す前に実在するか確認する (`bench_blas_compare.py` /
`bench_vs_qutip.py` は 2026-08 時点で未実装)。

性能改善の主張をするときの方法 (cv_ising 流):

- 「○○× 速くなった」という主張は **同一マシン上の before / after** で示す。
  CPU / BLAS バックエンド / NumPy バージョン / 熱状態が揃っている必要がある。
- BLAS feature on/off の比較は `bench_blas_compare.py` を使う想定 (どの
  ハードウェアでも有効) だが **未実装**。現時点では `bench_per_step.py` を
  feature 切替 build (`maturin develop --uv --release --no-default-features
  --features extension-module,rayon,simd` 等) で 2 回回して比較する。
- それ以外の性能変更 (アルゴリズム差し替え等) では `git stash` または
  `git switch` で実装を切り替えつつ `bench_per_step.py` を 2 回回し、
  自前で per-cell 比較表を作る。
- 結果は `benchmarks/results/<YYYYMMDD-HHMMSS>/` に CSV + markdown を残す
  (gitignored)。書き戻すときはハード (機種 / チップ / メモリ / OS / NumPy /
  BLAS backend) を節タイトルで明示する。
