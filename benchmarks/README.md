# benchmarks/

`kryanneal` の per-step 性能計測 CLI スクリプト群を置く. 設計上の位置付けは
`docs/design.md` §10, ベンチ規約の詳細は `CLAUDE.md` 「ベンチマーク」節を
参照すること.

## スクリプト一覧

| スクリプト | 内容 | 導入 Phase |
|---|---|---|
| `bench_per_step.py` | M2 / Trotter / Suzuki S_4 / CFM4:2 / CFM4 adaptive Richardson の per-step wall time を `(method, n)` で sweep. adaptive 経路は `n_steps_actual` (PI controller が accept した実 step 数) と `final_err_vs_ref` (高精度参照解との state 差) も併記 | Phase 1 (M2) → Phase 2 (Trotter / Suzuki S_4) → Phase 3 (CFM4:2) → Phase 4 C3 (adaptive Richardson) |
| `bench_blas_compare.py` | BLAS feature on/off の同一マシン比較 | Phase 6 予定 |
| `bench_vs_qutip.py` | QuTiP `sesolve` との fidelity vs wall time | Phase 3 以降予定 |

## 実行

```bash
uv run python benchmarks/bench_per_step.py
```

引数:

- `--n-values N1,N2,...`: sweep するスピン数の列 (default `4,8,12,16`).
- `--methods M1,M2,...`: 計測する propagator method の列
  (`m2` / `trotter` / `trotter_suzuki4` / `cfm4` / `cfm4_adaptive_richardson`
  から選ぶ, default 全て). adaptive 経路を含む場合は同 `n` で高精度
  参照解 (fixed CFM4:2 ・ 多 step) を 1 回計算してから per-method 計測を
  回す.
- `--n-steps K`: 各 `(n, method)` で時間発展する step 数 (default 50).
- `--m M`: Lanczos 部分空間次元 (default 24). `method="m2"` のみで使用,
  Trotter / Suzuki S_4 経路では Lanczos を呼ばないため無視される.
- `--repeat R`: 各設定で wall time を測る試行回数. CSV には全試行を残し,
  markdown には min/median を要約する (default 3).
- `--warmup W`: 計測前に捨てる試行回数 (cache warm 用, default 1).
- `--T T`: 総アニーリング時間 (default 1.0).
- `--blas-threads N`: 指定時に `kryanneal.set_blas_threads(N)` を呼んで
  全 BLAS pool (numpy bundled + system OpenBLAS の双方を含む) のスレッド数を
  統一する. Linux + numpy bundled OpenBLAS では default 物理コア数まで張る
  ため、小 dim で thread-launch overhead が支配し per-step がノイジーになる.
  **machine-independent baseline には `--blas-threads 1` を推奨**.
  default は `None` (BLAS thread 数に手を加えない, Phase 1 baseline と同じ).
- `--results-dir DIR`: 出力先 (default `benchmarks/results/<YYYYMMDD-HHMMSS>/`).

## 出力

`benchmarks/results/<YYYYMMDD-HHMMSS>/` を作り, 以下を書く:

- `bench_per_step.csv`: 全試行の raw タイムスタンプ (n, dim, method,
  trial, n_steps, dt, m, total_wall_sec, per_step_sec, states_per_sec,
  n_steps_actual, final_err_vs_ref). 末尾 2 列は adaptive 経路でのみ
  実値が入り, 固定 dt 経路では `n_steps_actual=n_steps` / `final_err_vs_ref=n/a`.
- `bench_per_step.md`: 集計表 (per-method summary + cross-method 比較表)
  と machine info (uname / Python / numpy version / BLAS pool).
  cross-method 表は M2 を基準にした ratio (`m2 / trotter` 等) を併記する.
  adaptive 経路 (`cfm4_adaptive_richardson` 等) を含む実行ではさらに
  `## Adaptive driver detail` 節を追加し, PI controller が accept した
  実 step 数 `n_steps_actual` (median + min/max) と参照解との
  `final_err_vs_ref` (median) を per-n × method で並べる. これらは
  adaptive driver の性能・精度評価で最重要な 2 値だが Summary 表からは
  落ちるため別節で必ず出す. 参照解計算 (`_compute_reference_psi`) の
  wall time は machine info の `reference_wall_sec_total` (合算) と
  adaptive section の per-n `reference_wall (sec)` 列に記録する
  (大 n で reference 1 本に 1-2 時間かかる現実問題に対する透明性).

**重要**: 同じ `n_steps` での raw per-step 比較は LTE order の違い
(M2 / Strang は `O(dt^3)`, Suzuki S_4 は `O(dt^5)`) を **無視している**
ので, 「精度を揃えた wall time 比較」を主張するには別途 required
`n_steps` の見積もりが必要 (`docs/design.md` §5.3 のクロスオーバ議論).

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
