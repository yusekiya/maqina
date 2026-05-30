# benchmarks/

`maqina` の per-step 性能計測 CLI スクリプト群を置く. 設計上の位置付けは
`docs/design/10-benchmarks.md` §10, ベンチ規約の詳細は `CLAUDE.md` 「ベンチマーク」節を
参照すること.

## スクリプト一覧

| スクリプト | 内容 | 導入 Phase |
|---|---|---|
| `bench_per_step.py` | M2 / Trotter / Suzuki S_4 / CFM4:2 / CFM4 adaptive Richardson の per-step wall time を `(method, n)` で sweep. adaptive 経路は `n_steps_actual` (PI controller が accept した実 step 数) と `final_err_vs_ref` (高精度参照解との state 差) も併記 | Phase 1 (M2) → Phase 2 (Trotter / Suzuki S_4) → Phase 3 (CFM4:2) → Phase 4 C3 (adaptive Richardson) |
| `bench_parallel_scaling.py` | `apply_h` / `apply_single_mode_axis_i` の rayon thread × N sweep. subprocess 起動時に `RAYON_NUM_THREADS` を変えながら physical core 数 vs throughput の knee (memory-bandwidth saturation point) を CSV + md に出力. BLAS thread は 1 固定で rayon 効果を分離する | Phase 6 C1 (issue #62) |
| `bench_simd_scaling.py` | `apply_h` per-pass time の SIMD ON/OFF 比較. `--mode measure` を異なる build で 2 回回し (`simd-on` / `simd-off` label), `--mode compare` で speedup table を MD/CSV に統合. i=0,1,2 集中時 (h_x=[1,1,1,0,...,0]) と全 i (h_x=all-ones) の 2 モードを取る | Phase 6 C2 (issue #63) |
| `bench_block_fusion.py` | `trotter_step` (multi-qubit gate fusion) と `apply_h` (L2-aware chunk_size) の per-step time を `N ∈ {18, 20, 22}` で計測. C2 完了時点 (baseline) と C3 適用後 (after) の 2 回 measure を `--label` で識別して取り, 手動 diff で per-cell speedup を算出する運用. acceptance: N=20, `trotter_step` で >= 1.3× | Phase 6 C3 (issue #64) |
| `bench_blas_compare.py` | BLAS feature on/off の同一マシン比較 | Phase 6 予定 |
| `bench_vs_qutip.py` | QuTiP `sesolve` との fidelity vs wall time | Phase 3 以降予定 |
| `bench_qutip_large.py` | QuTiP `sesolve` vs maqina の **work-precision diagram** ベンチ. 複数 **scenario** (built-in: `standard` T=1 N=10,12 / `long-T` T=1e4 N=8,10 / `stiff` h_p_scale=10 N=10,12 / `large-N` T=1 N=12,14,16 / `stiff-long-T` opt-in N=6,8) を 1 invocation で sweep. 各 scenario は適切な N 範囲を内蔵 (long-T で N=12 にすると 1 cell 分単位なので絞る等). 各 solver は固有の精度つまみ (maqina m2/trotter/cfm4 = `dt`, adaptive = `atol`, QuTiP = `tol` で atol=rtol) を独自 sweep し共通 reference との infidelity + wall_sec を per-cell 1 回測定. MD/CSV は per-(scenario, n) で infidelity 昇順 + Pareto 最適 ✓ 付き. `--add-scenario "name:T=...,h_p=...,h_x=...,n=12;14"` でカスタム可. `--ref-validate` で QuTiP self-convergence + maqina cross-check による reference 妥当性検証 section を MD に追加 | Phase 6 C4 (issue #65) |
| `bench_m_eff_adiabatic.py` | maqina `cfm4_adaptive_richardson_krylov` の **Krylov 部分空間実効次元 `m_eff` の T 依存性** 計測. QuTiP 比較ではなく adaptive driver の内部挙動分析専用. `evolve_schedule_adaptive_richardson` driver を直接呼び `m_eff_history` を取得し, per-(n, T, atol, m_max) の m_eff 分布 (mean/median/min/max/P10/P90/compression_ratio) + per-time-bin histogram を MD/CSV に出す. `--krylov-tol-factor` で β_k 早期打切閾値を調整可 (default 1e-3 = QuantumAnnealer default と一致, 1.0 等で発火しやすくなる) | Phase 6 C4 (issue #65) |

## 実行

```bash
uv run python benchmarks/bench_per_step.py
```

引数:

- `--n-values N1,N2,...`: sweep するスピン数の列 (default `4,8,12,16`).
- `--methods M1,M2,...`: 計測する propagator method の列
  (`m2` / `trotter` / `trotter_suzuki4` / `cfm4` / `cfm4_adaptive_richardson_krylov`
  から選ぶ, default 全て). adaptive 経路を含む場合は同 `n` で高精度
  参照解 (fixed CFM4:2 ・ 多 step) を 1 回計算してから per-method 計測を
  回す.
- `--n-steps K`: 各 `(n, method)` で時間発展する step 数 (default 50).
- `--m-values M1,M2,...`: Lanczos 部分空間次元の sweep 列 (default `24`).
  `m2` / `cfm4` / `cfm4_adaptive_richardson_krylov` で使用. Trotter / Suzuki S_4
  経路では Lanczos を呼ばないため無視される. issue #52 B で列形式に拡張
  され, `--m-values 16,24,32` で `m=16` / `m=24` / `m=32` の cell 比較を
  1 run で取れる. m sweep を入れた場合 Cross-method 表は出力されない
  (Summary 表で `m` 列付きの形になる).
- `--repeat R`: 各設定で wall time を測る試行回数. CSV には全試行を残し,
  markdown には min/median を要約する (default 3).
- `--warmup W`: 計測前に捨てる試行回数 (cache warm 用, default 1).
- `--T T`: 総アニーリング時間 (default 1.0).
- `--blas-threads N`: 指定時に `maqina.set_blas_threads(N)` を呼んで
  全 BLAS pool (numpy bundled + system OpenBLAS の双方を含む) のスレッド数を
  統一する. Linux + numpy bundled OpenBLAS では default 物理コア数まで張る
  ため、小 dim で thread-launch overhead が支配し per-step がノイジーになる.
  **machine-independent baseline には `--blas-threads 1` を推奨**.
  default は `None` (BLAS thread 数に手を加えない, Phase 1 baseline と同じ).
  - **本番 perf bench (Pareto / QuTiP 比較 等) は `--blas-threads 8` を default に**
    する運用 (issue #116 / PR #115 の Linux AMD EPYC 7713P perf sweep で NT=8 が
    sweet spot, NT=1 比 1.52× speedup; NT=64 default は NT=8 比 -9% 劣化).
    `--blas-threads` 不指定で起動すると OpenBLAS が物理コア数まで thread を
    張って sweet spot 比 ~1.10× 遅化する (2026-05-21 PR #106 bench で実測,
    PR #106 のコメントに数値あり). 本番 sweep を 0.8.0 で取った場合も
    `--blas-threads 8` を渡せば 0.9.0+ 相当の挙動になる
    (`set_blas_threads_auto()` は 0.9.0 で追加された自動算出 API だが,
    EPYC 7713P と分かっている本番 bench では `--blas-threads 8` 直指定で十分).
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
  adaptive 経路 (`cfm4_adaptive_richardson_krylov` 等) を含む実行ではさらに
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

## README figure pipeline (fidelity-vs-runtime 散布図)

README に埋め込む Pareto 図 (`docs/figures/<version>_pareto_<scenario>.png`)
を生成する 3 script の連携:

| script | 役割 |
|---|---|
| `compute_readme_reference.py` | 高精度参照解 `ψ_ref(T)` を QuTiP `sesolve` で生成し npz 化 (Adams 自己収束 + BDF 解法独立性を検証). `--bdf-tols` を指定すると BDF を primary にした sweep mode. |
| `bench_readme_figure.py` | 事前生成済 npz (問題 + 参照解) を読み, maqina 3 method (`cfm4_adaptive_richardson_krylov` / 固定 `cfm4` / `cfm4_adaptive_richardson_chebyshev`) と QuTiP の精度つまみ sweep を回し per-cell の `wall_sec` + `infidelity` を CSV に dump. |
| `plot_readme_figure.py` | 複数 CSV を結合して scenario 別散布図 PNG を生成. 系列 key は `(solver, variant)`. |

scenario 名と図タイトルの対応 (`plot_readme_figure.py::SCENARIO_TITLE`):

- `non-stiff` = **narrow dynamic range** (SK random, `h_p_scale=1`). 参照解は
  Adams primary。
- `stiff` = **wide dynamic range** (SK + penalty, `h_p_scale=10`). Schrödinger
  方程式は ODE 的には stiff にならない (H Hermitian) ので "stiff" は legacy ID,
  実体は `H_p` の dynamic range の広さ。issue #158 で参照解を **BDF sweep
  primary** に差し替え済 (Adams adaptive step が 1e-9 で頭打ちになる問題を回避)。

### 状態ベクトル保存 (`--save-states-dir`, issue #158 再発防止)

`bench_readme_figure.py --save-states-dir DIR` で各 cell の最終状態 `ψ(T)` を
self-describing 圧縮 npz (`state_<scenario>_<solver>_<variant>_<knob>_<value>.npz`)
として保存する。npz には `psi` 本体に加え problem/reference パス・sweep メタ
データ・`maqina` version を同梱するので、**参照解が後から差し替わっても保存済
`ψ` から infidelity を再計算できる** (`infidelity(psi, new_ref)`)。容量は
`2^n·16 byte` (n=18 で 4.0 MiB raw / ~3.84 MiB 圧縮)。`benchmarks/results/<X.Y.Z>/states/`
配下に置けば `.gitignore` 例外で `.npz` のみ track される。既存ファイルは上書き
しない (resume 冪等)。

> ⚠️ 過去 (≤0.12.0) の QuTiP / 旧 maqina cell は状態ベクトルを保存していなかった
> ため、参照解差し替え後に infidelity を再計算する手段が無く再実行が必要だった。
> 0.14.0 以降はこのフラグで全 cell の `ψ` を残す運用にする。

### 0.14.0 取り直し (CHANGELOG 0.14.0 / issue #148)

0.14.0 の adaptive controller 変更 (真の PI 化 #151 / reject 縮小 #149 /
成長凍結 #150) は adaptive 系 method の挙動に影響する。また stiff 参照解が
issue #158 で BDF に差し替わった。再計算スコープ (再計算時間削減のため流用最大化):

| 系列 | non-stiff (narrow) | stiff (wide) | ψ 保存 |
|---|---|---|---|
| Krylov adaptive | **再計算** (controller 変更) | **再計算** | ✅ |
| Chebyshev adaptive | **再計算** (controller 変更) | **再計算** | ✅ |
| Krylov fixed (cfm4) | 0.8.0 流用 (controller 非依存 + 参照解不変) | **再計算** (新 BDF 参照に統一) | ❌ |
| QuTiP | **再計算** (状態保存目的) | **再計算** (新 BDF 参照 + 状態保存) | ✅ |

> 固定 cfm4 の ψ を保存しない理由: cfm4 は version 依存 (伝播器実装が version で
> 変わりうる) で更新ごとに再計算が必要なため, 今保存しても再利用価値が小さい。
> stiff cfm4 は新 BDF 参照での infidelity を得るため CSV 行は再生成するが ψ は残さない。
> narrow cfm4 は 0.8.0 を流用 (固定 cfm4 は 0.14.0 の adaptive controller 変更の影響を
> 受けず, narrow 参照解も不変なので infidelity 有効; 同一マシン・同一アルゴリズムで
> wall_sec も比較可能)。

> QuTiP を **両 scenario とも再計算する理由**:
> - stiff: 参照解が Adams → BDF (#158) に差し替わった上, 過去 run が状態ベクトルを
>   未保存で新参照では再計算できず実行必須 (floor cell `tol=1e-9` は infidelity
>   7.98e-12 が参照差 8.5e-12 に支配される; `tol=1e-5` も ±35% 変わる)。
> - non-stiff: infidelity 値自体は参照解不変で変わらないが, 過去 run が状態ベクトルを
>   未保存だったため「将来の参照解変更時に再実行が必要」な負債だった。今のうちに
>   `--save-states-dir` 付きで再計算して ψ を残すことで, 問題 npz を変えない限り QuTiP
>   cell は version / 参照解変更を横断して再利用可能になる (#158 再発防止)。
>
> **保存先**: QuTiP は maqina version 非依存 (問題 + ODE ソルバのみで決まる) なので
> version dir ではなく共有 `benchmarks/results/qutip/` に置く (CSV + `states/`)。
> wall_sec は version 非依存, infidelity は参照解依存 (参照変更時は保存 ψ から再生成)。
> 既存 `results/qutip/bench_*.csv` は状態保存 + 新参照で取り直すため `.bak` に退避して
> fresh 再生成する (QuTiP は決定的なので psi 不変, non-stiff infidelity も同値,
> stiff infidelity のみ新 BDF 参照で更新)。

再計算は `scripts/run_bench_readme_0_14_0.sh` で一括起動 (README ベンチ専用
サーバー = 回帰/perf 用 EPYC 7713P とは別機; 0.8.0/0.12.0 README 図と同一サーバー・
同一設定 [BLAS/RAYON とも env 未設定 = default] に揃える)。出力:
- maqina (version 依存): `benchmarks/results/0.14.0/bench_<scenario>.csv`
  + `states/` (adaptive 18 cell; cfm4 は ψ 非保存)
- QuTiP (version 非依存): `benchmarks/results/qutip/bench_<scenario>.csv` + `states/` (8 cell)

保存 ψ 合計 26 cell ≒ 100 MiB。所要は QuTiP だけで約 65h (stiff ~48h + non-stiff ~17h,
既存実測ベース) 追加。

**プロット (流用 cell の結合)**: `plot_readme_figure.py` は系列を `(solver,
variant)` で判定する。流用する 0.8.0 narrow 固定 cfm4 は `solver=kinema` (旧
package 名) なので `solver=maqina` に直した reused CSV を作ってから渡す:

```bash
# 0.8.0 narrow 固定 cfm4 (krylov_fixed) を solver=maqina に直して再利用 CSV 化
# (ファイル名は bench_ 始まりで track 対象にする)
awk -F, 'NR==1 || $6=="krylov_fixed"' benchmarks/results/0.8.0/bench_non-stiff.csv \
  | sed 's/,kinema,krylov_fixed,/,maqina,krylov_fixed,/' \
  > benchmarks/results/0.14.0/bench_reused_non-stiff.csv

# plot に渡す CSV:
#  - benchmarks/results/0.14.0/bench_*.csv : 再計算した maqina 全系列
#                                            (+ bench_reused_non-stiff.csv = narrow 固定 cfm4)
#  - benchmarks/results/qutip/bench_*.csv  : 再計算した QuTiP (両 scenario, version 非依存共有)
uv run python -m benchmarks.plot_readme_figure \
  --input-csv benchmarks/results/0.14.0/bench_*.csv \
              benchmarks/results/qutip/bench_*.csv \
  --version 0.14.0
```
