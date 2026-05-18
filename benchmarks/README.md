# benchmarks/

`kryanneal` の per-step 性能計測 CLI スクリプト群を置く. 設計上の位置付けは
`docs/design/10-benchmarks.md` §10, ベンチ規約の詳細は `CLAUDE.md` 「ベンチマーク」節を
参照すること.

## スクリプト一覧

| スクリプト | 内容 | 導入 Phase |
|---|---|---|
| `bench_per_step.py` | M2 / Trotter / Suzuki S_4 / CFM4:2 / CFM4 adaptive Richardson の per-step wall time を `(method, n)` で sweep. adaptive 経路は `n_steps_actual` (PI controller が accept した実 step 数) と `final_err_vs_ref` (高精度参照解との state 差) も併記 | Phase 1 (M2) → Phase 2 (Trotter / Suzuki S_4) → Phase 3 (CFM4:2) → Phase 4 C3 (adaptive Richardson) |
| `bench_parallel_scaling.py` | `apply_h_kryanneal` / `apply_single_mode_axis_i` の rayon thread × N sweep. subprocess 起動時に `RAYON_NUM_THREADS` を変えながら physical core 数 vs throughput の knee (memory-bandwidth saturation point) を CSV + md に出力. BLAS thread は 1 固定で rayon 効果を分離する | Phase 6 C1 (issue #62) |
| `bench_simd_scaling.py` | `apply_h_kryanneal` per-pass time の SIMD ON/OFF 比較. `--mode measure` を異なる build で 2 回回し (`simd-on` / `simd-off` label), `--mode compare` で speedup table を MD/CSV に統合. i=0,1,2 集中時 (h_x=[1,1,1,0,...,0]) と全 i (h_x=all-ones) の 2 モードを取る | Phase 6 C2 (issue #63) |
| `bench_block_fusion.py` | `trotter_step` (multi-qubit gate fusion) と `apply_h_kryanneal` (L2-aware chunk_size) の per-step time を `N ∈ {18, 20, 22}` で計測. C2 完了時点 (baseline) と C3 適用後 (after) の 2 回 measure を `--label` で識別して取り, 手動 diff で per-cell speedup を算出する運用. acceptance: N=20, `trotter_step` で >= 1.3× | Phase 6 C3 (issue #64) |
| `bench_blas_compare.py` | BLAS feature on/off の同一マシン比較 | Phase 6 予定 |
| `bench_vs_qutip.py` | QuTiP `sesolve` との fidelity vs wall time | Phase 3 以降予定 |
| `bench_qutip_large.py` | QuTiP `sesolve` vs kryanneal の **work-precision diagram** ベンチ. 複数 **scenario** (built-in: `standard` T=1 N=10,12 / `long-T` T=1e4 N=8,10 / `stiff` h_p_scale=10 N=10,12 / `large-N` T=1 N=12,14,16 / `stiff-long-T` opt-in N=6,8) を 1 invocation で sweep. 各 scenario は適切な N 範囲を内蔵 (long-T で N=12 にすると 1 cell 分単位なので絞る等). 各 solver は固有の精度つまみ (kryanneal m2/trotter/cfm4 = `dt`, adaptive = `atol`, QuTiP = `tol` で atol=rtol) を独自 sweep し共通 reference との infidelity + wall_sec を per-cell 1 回測定. MD/CSV は per-(scenario, n) で infidelity 昇順 + Pareto 最適 ✓ 付き. `--add-scenario "name:T=...,h_p=...,h_x=...,n=12;14"` でカスタム可. `--ref-validate` で QuTiP self-convergence + kryanneal cross-check による reference 妥当性検証 section を MD に追加 | Phase 6 C4 (issue #65) |
| `bench_m_eff_adiabatic.py` | kryanneal `cfm4_adaptive_richardson` の **Krylov 部分空間実効次元 `m_eff` の T 依存性** 計測. QuTiP 比較ではなく adaptive driver の内部挙動分析専用. `evolve_schedule_adaptive_richardson` driver を直接呼び `m_eff_history` を取得し, per-(n, T, atol, m_max) の m_eff 分布 (mean/median/min/max/P10/P90/compression_ratio) + per-time-bin histogram を MD/CSV に出す. `--krylov-tol-factor` で β_k 早期打切閾値を調整可 (default 1e-3 = QuantumAnnealer default と一致, 1.0 等で発火しやすくなる) | Phase 6 C4 (issue #65) |

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
- `--m-values M1,M2,...`: Lanczos 部分空間次元の sweep 列 (default `24`).
  `m2` / `cfm4` / `cfm4_adaptive_richardson` で使用. Trotter / Suzuki S_4
  経路では Lanczos を呼ばないため無視される. issue #52 B で列形式に拡張
  され, `--m-values 16,24,32` で `m=16` / `m=24` / `m=32` の cell 比較を
  1 run で取れる. m sweep を入れた場合 Cross-method 表は出力されない
  (Summary 表で `m` 列付きの形になる).
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
  n_steps_actual, final_err_vs_ref, m_eff_median, m_eff_max). 末尾 4 列は
  adaptive 経路でのみ実値が入り, 固定 dt 経路では `n_steps_actual=n_steps`
  / `final_err_vs_ref=n/a` / `m_eff_*=n/a`.
- `bench_per_step.md`: 集計表 (per-method × m summary + cross-method 比較
  表 + adaptive driver detail) と machine info (uname / Python / numpy
  version / BLAS pool). cross-method 表は M2 を基準にした ratio
  (`m2 / trotter` 等) を併記するが, `--m-values` で m sweep を入れた場合
  は cell が重複するため出力されない (代わりに Summary 表で `m` 列付き).
  adaptive 経路 (`cfm4_adaptive_richardson` 等) を含む実行ではさらに
  `## Adaptive driver detail` 節を追加し, PI controller が accept した
  実 step 数 `n_steps_actual` (median + min/max) と参照解との
  `final_err_vs_ref` (median), per-step Lanczos 部分空間次元
  `m_eff (median)` / `m_eff (max)` (issue #52 B で追加, β_k 早期打切の
  実効分布; 早期打切なしで `6m`, ありで小さく出る) を per-n × method × m
  で並べる. これらは adaptive driver の性能・精度評価で重要な指標
  なので Summary 表に加えて別節で必ず出す. 参照解計算
  (`_compute_reference_psi`) の wall time は machine info の
  `reference_wall_sec_total` (合算) と adaptive section の per-n
  `reference_wall (sec)` 列に記録する (大 n で reference 1 本に 1-2 時間
  かかる現実問題に対する透明性).

**重要**: 同じ `n_steps` での raw per-step 比較は LTE order の違い
(M2 / Strang は `O(dt^3)`, Suzuki S_4 は `O(dt^5)`) を **無視している**
ので, 「精度を揃えた wall time 比較」を主張するには別途 required
`n_steps` の見積もりが必要 (`docs/design/05-3-propagator.md` §5.3 のクロスオーバ議論).

ディレクトリは `.gitignore` で除外済み. 計測結果を共有する場合は
markdown を抜粋して PR / issue 本文に貼り付ける.

## README figure pipeline (Fidelity vs runtime 散布図)

README に埋め込む QuTiP vs kryanneal の Pareto 比較図
(`docs/figures/<X.Y.Z>_pareto_<scenario>.png`) を生成する 4 step pipeline.

**スクリプト**:

| script | 役割 |
|---|---|
| `_readme_figure_helpers.py` | 共有 helper (QuTiP sparse Hamiltonian / `run_qutip` / `infidelity`) |
| `build_readme_problem.py` | 問題定義 (`H_p_diag` / `h_x`) を `benchmarks/data/readme_problem_*.npz` に保存. non-stiff (SK random) / stiff (SK + 10% basis に penalty 100 加算) の 2 scenario |
| `compute_readme_reference.py` | 参照解 ψ_ref を QuTiP で 1 度だけ計算して保存. **Adams 許容誤差 sweep で収束確認 + 同精度 BDF で解法独立性確認**. 結果は `benchmarks/data/readme_reference_*.npz` |
| `bench_readme_figure.py` | 上記 2 npz を読み, kryanneal `cfm4_adaptive_richardson` の `atol` sweep + QuTiP `sesolve` (Adams) の `tol` sweep を回して各 cell の wall time + infidelity を CSV 出力 |
| `plot_readme_figure.py` | CSV を読んで matplotlib で散布図を描画 (両軸 log). PNG を `docs/figures/` に出力 |

**実行手順** (本番想定: N=18, T=10^4, scenario=non-stiff/stiff):

```bash
# 1) 問題ファイル生成 (各 scenario 1 回, fast)
uv run python -m benchmarks.build_readme_problem --scenario non-stiff --n 18
uv run python -m benchmarks.build_readme_problem --scenario stiff --n 18

# 2) 参照解計算 (各 scenario 1 回, QuTiP Adams + BDF, heavy)
uv run python -m benchmarks.compute_readme_reference \
    --problem-file benchmarks/data/readme_problem_non-stiff_n18_seed20260518.npz \
    --T 10000
uv run python -m benchmarks.compute_readme_reference \
    --problem-file benchmarks/data/readme_problem_stiff_n18_seed20260518.npz \
    --T 10000

# 3) bench sweep (各 scenario, kryanneal atol sweep + QuTiP tol sweep)
uv run python -m benchmarks.bench_readme_figure \
    --problem-file    benchmarks/data/readme_problem_non-stiff_n18_seed20260518.npz \
    --reference-file  benchmarks/data/readme_reference_non-stiff_n18_T10000_seed20260518.npz
uv run python -m benchmarks.bench_readme_figure \
    --problem-file    benchmarks/data/readme_problem_stiff_n18_seed20260518.npz \
    --reference-file  benchmarks/data/readme_reference_stiff_n18_T10000_seed20260518.npz

# 4) 描画 (両 scenario の CSV を渡して 2 PNG を一括生成)
uv run python -m benchmarks.plot_readme_figure \
    --input-csv  benchmarks/results/readme-figure/bench_readme_non-stiff.csv \
                 benchmarks/results/readme-figure/bench_readme_stiff.csv \
    --output-dir docs/figures --version 0.8.0
```

**永続化**:

- 問題 npz / 参照解 npz: `benchmarks/data/` (gitignore, 1 度だけ計算して
  reuse). 同 seed / scenario / n / T で再現可能.
- bench CSV: `benchmarks/results/readme-figure/` (gitignore).
- PNG: `docs/figures/<version>_pareto_<scenario>.png` (**git track**,
  README から `![](...)` で参照).

**参照解の妥当性検証** (compute_readme_reference.py 内蔵):

1. Adams (default ODE 法) を tol を粗 → 細に sweep し, 隣接 pair の
   infidelity が `--convergence-threshold` (default 1e-13) 未満であることを
   確認 (許容誤差を下げても解が動かない = 収束).
2. 同じ最細 tol で BDF (stiff 向け ODE 法) を 1 回計算し, Adams 最高精度
   との infidelity が同じ閾値未満であることを確認 (解法独立性).
3. 両方 pass で **Adams 最高精度を参照解として採用**. fail の場合 WARNING
   を出すが計算自体は続行 (ユーザーが tol を細かくするか threshold を緩める
   か判断する).

## リリース bench artifact (`benchmarks/results/<X.Y.Z>/`)

Phase 完了 bump 時の本番 bench sweep 結果は **`benchmarks/results/<X.Y.Z>/`**
(例 `benchmarks/results/0.8.0/`) に永続化する. 詳細運用は
[`docs/conventions.md`](../docs/conventions.md) §2.3 を一次資料とする. 要点:

- ディレクトリ名は semver `X.Y.Z` そのまま (prefix なし, `v0.8.0` の形は
  使わない).
- **必置**: `SUMMARY.md` (当該 version 全体の解釈 / 累積改善ハイライト /
  acceptance 判定).
- **コミット対象**: SUMMARY.md + 生 bench の `bench_*.md`.
- **除外**: 生 CSV (`.csv` は gitignore のまま, 必要なら bench 実行マシン上
  の timestamped dir で参照).
- `.gitignore` は `!/benchmarks/results/*/*.md` の except ルールで version
  dir 配下の markdown のみ track する.

## 性能改善を主張するときの作法

`CLAUDE.md` 同節を一次資料とする. 要点だけ:

1. **同一マシン上の before/after** で示す. CPU / BLAS / NumPy バージョン /
   熱状態を揃える.
2. BLAS on/off の比較は `bench_blas_compare.py` を使う (どのマシンでも
   再現できる相対比較).
3. それ以外の改善 (アルゴリズム差し替え等) は `git stash` / `git switch`
   で実装を切替え, `bench_per_step.py` を 2 回回して per-cell 比較する.
