# benchmarks/

`kryanneal` の per-step 性能計測 CLI スクリプト群を置く. 設計上の位置付けは
`docs/design.md` §10, ベンチ規約の詳細は `CLAUDE.md` 「ベンチマーク」節を
参照すること.

## スクリプト一覧

| スクリプト | 内容 | 導入 Phase |
|---|---|---|
| `bench_per_step.py` | M2 (Phase 1) / CFM4 (Phase 3) / Richardson (Phase 4) の per-step wall time を `n` で sweep | Phase 1 (M2 のみ) |
| `bench_blas_compare.py` | BLAS feature on/off の同一マシン比較 | Phase 6 予定 |
| `bench_vs_qutip.py` | QuTiP `sesolve` との fidelity vs wall time | Phase 3 以降予定 |

## 実行

```bash
uv run python benchmarks/bench_per_step.py
```

引数:

- `--n-values N1,N2,...`: sweep するスピン数の列 (default `4,8,12,16`).
- `--n-steps K`: 各 `n` で時間発展する step 数 (default 50).
- `--m M`: Lanczos 部分空間次元 (default 24).
- `--repeat R`: 各設定で wall time を測る試行回数. CSV には全試行を残し,
  markdown には min/median を要約する (default 3).
- `--warmup W`: 計測前に捨てる試行回数 (cache warm 用, default 1).
- `--T T`: 総アニーリング時間 (default 1.0).
- `--results-dir DIR`: 出力先 (default `benchmarks/results/<YYYYMMDD-HHMMSS>/`).

## 出力

`benchmarks/results/<YYYYMMDD-HHMMSS>/` を作り, 以下を書く:

- `bench_per_step.csv`: 全試行の raw タイムスタンプ (n, step, trial, dt, t_step_sec, ...).
- `bench_per_step.md`: 集計表 (n ごとに wall time min/median, throughput
  states/sec) と machine info (uname / Python / numpy version / BLAS pool).

ディレクトリは `.gitignore` で除外済み. 計測結果を共有する場合は
markdown を抜粋して PR / issue 本文に貼り付ける.

## 性能改善を主張するときの作法

`CLAUDE.md` 同節を一次資料とする. 要点だけ:

1. **同一マシン上の before/after** で示す. CPU / BLAS / NumPy バージョン /
   熱状態を揃える.
2. BLAS on/off の比較は `bench_blas_compare.py` を使う (どのマシンでも
   再現できる相対比較).
3. それ以外の改善 (アルゴリズム差し替え等) は `git stash` / `git switch`
   で実装を切替え, `bench_per_step.py` を 2 回回して per-cell 比較する.
