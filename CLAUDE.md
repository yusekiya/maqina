# CLAUDE.md

Claude Code 向けのプロジェクトガイド。

## 要望

指示がない限りユーザーへの回答や質問は日本語で行うこと。

## プロジェクト概要

`kryanneal` (Krylov + Annealing): 横磁場イジングモデル (TFIM) の量子
ダイナミクスを matrix-free に計算するシミュレータ。Krylov 法 (Lanczos)
で短時間プロパゲータを近似し、Magnus 展開 (CFM4:2) で時間依存
Hamiltonian の時間発展演算子を近似する。adaptive dt ドライバ
(step-doubling Richardson + PI 制御) も提供。

設計の参照プロジェクト: [`cv-ising-solver`](https://github.com/Shu-Tanaka-Group/cv-ising-solver)
(同じ Krylov + CFM4:2 カーネルの連続変数版)。

- パッケージマネージャ: `uv` (Python `>=3.13`)
- ビルドバックエンド: `maturin` (Rust 拡張 `kryanneal._rust` を PyO3 経由でビルド)
- Lint: `ruff`
- 型チェック: `ty`
- 主要依存: `numpy`, `threadpoolctl`
- dev 依存: `pytest`, `qutip` (参照実装比較用), `pre-commit`, `ruff`, `ty`
- Rust: `pyo3 0.28`, `numpy 0.28`, `ndarray 0.16`, `num-complex 0.4`,
  `cblas 0.5` (optional)。LAPACK 非依存 (三重対角固有分解は
  `src/tridiag.rs` に hand-rolled 実装、§7.1 参照)

## 設計書

`docs/design/INDEX.md` が一次資料。実装に着手する前に必ず読む。主要セクション:

- §3 アーキテクチャ / ディレクトリレイアウト
- §4 公開 Python API (`IsingProblem`, `Schedule`, `QuantumAnnealer`, ...)
- §5 数値カーネル (Lanczos, M2, CFM4:2, Trotter, Richardson adaptive 含む)
- §7 Rust 拡張 (BLAS feature, maturin 標準レイアウト準拠の根拠)
- §8 QuTiP 比較
- §12 段階リリース計画 (Phase 1-6)

## 開発規約

`docs/conventions.md` が一次資料。開発プロセス / ビルド基盤 / バージョ
ニングはここを参照する:

- §1 開発・ビルド基盤 (uv / maturin / ruff / ty / pre-commit / API stub 二段運用)
- §2 バージョニングポリシー (Phase N 完了で `0.N.0` へ bump,
  umbrella issue Definition of Done 必須項目)

## レイアウト

[maturin 公式ドキュメント](https://www.maturin.rs/project_layout) 推奨の
mixed Rust/Python project 標準形 (`python-source = "python"`, Rust ルート
直下に `Cargo.toml` + `src/`)。

```
kryanneal/
├── pyproject.toml
├── Cargo.toml                  # Rust crate ルート (maturin 標準位置)
├── src/                        # Rust ソース
│   ├── lib.rs                  # PyO3 #[pymodule] fn _rust エントリポイント
│   ├── matvec.rs               # apply_h_kryanneal (bit-flip + diag)
│   ├── krylov.rs               # lanczos_propagate (ndarray ベース)
│   ├── cfm4.rs                 # CFM4:2 / M2 / Richardson 推定子
│   ├── tridiag.rs              # 実対称三重対角の implicit QL (hand-rolled)
│   └── blas.rs                 # 内積 / axpy / nrm2 / scal ラッパ
├── python/kryanneal/           # Python ソース (python-source = "python")
│   ├── __init__.py             # 公開 API
│   ├── __init__.pyi            # 自動生成 stub (wheel 同梱)
│   ├── py.typed                # PEP 561 マーカ
│   ├── problem.py              # IsingProblem
│   ├── schedule.py             # Schedule
│   ├── annealer.py             # QuantumAnnealer (one-shot run() ファサード)
│   ├── simulator.py            # AnnealingSimulator (step-wise stateful API)
│   ├── krylov.py               # adaptive ドライバ + Python リファレンス
│   ├── eigenstates.py          # 瞬時固有状態への投影
│   ├── builders.py             # PauliTerm → diag, J/h → diag
│   ├── initial_states.py       # |+⟩^N
│   ├── result.py               # QuantumResult, Trajectory
│   ├── reference_qutip.py      # QuTiP sesolve 比較
│   └── _rust.*.so              # maturin develop でここに配置
├── tools/
│   └── gen_api_stubs.py        # `.pyi` 自動生成
├── tests/                      # pytest (Python 統合テスト)
├── benchmarks/
└── docs/
    ├── design/                 # 一次設計書 (章別分割; INDEX.md がエントリポイント)
    │   ├── INDEX.md            # 目次 + §N.M → ファイル mapping + 横断トピック
    │   ├── 01-goals.md         # §1 ゴール
    │   ├── 02-physics.md       # §2 物理モデル (bit 規約)
    │   ├── 03-architecture.md  # §3 アーキテクチャ / レイアウト
    │   ├── 04-python-api.md    # §4 公開 Python API
    │   ├── 05-1-matvec.md      # §5.1 matvec / per-axis primitives
    │   ├── 05-2-lanczos.md     # §5.2 Lanczos
    │   ├── 05-3-propagator.md  # §5.3 M2/CFM4/Trotter/PI controller
    │   ├── 05-4-python-reference.md
    │   ├── 06-builders.md      # §6
    │   ├── 07-rust-extension.md  # §7
    │   ├── 08-qutip-comparison.md  # §8
    │   ├── 09-testing.md       # §9
    │   ├── 10-benchmarks.md    # §10
    │   ├── 11-build-infrastructure.md
    │   ├── 12-release-plan.md  # §12 Phase 1-6
    │   ├── 13-future-work.md
    │   └── 14-references.md
    ├── conventions.md          # 開発規約 (ビルド基盤 / バージョニング)
    ├── testing.md              # /test skill 用
    └── benchmarks.md
```

## テスト

詳細は project skill `.claude/skills/test-runner/SKILL.md` を一次資料とする
(slash command `/test-runner` で発火可能, `test-runner` subagent もここを
読んで実行する). 要点だけ:

```bash
uv run pytest                               # 全 Python テスト
uv run pytest -m "not slow"                 # slow を除外
uv run pytest tests/test_krylov.py          # 個別ファイル
cargo test                                  # Rust 単体 (BLAS feature ON)
cargo test --no-default-features            # scalar fallback
uv run maturin develop --uv                 # Rust 変更後に必須 (--uv は pip 非同梱回避)
```

`uv run` を必ず使う (PyO3 の `extension-module` feature とローカル Python の
ABI を揃える必要があるため、システム Python での実行は避ける)。

### BLAS feature on/off の数値一致検証 (issue #65 Phase 6 C4)

`cargo test --no-default-features` で Rust 内部単体の rel < 1e-13 一致は
担保されるが, **Python 公開 API レベルでの end-to-end 一致** は以下の
artifact フローで検証する:

```bash
# 1. BLAS on build で artifact 生成
uv run maturin develop --uv --release
KRYANNEAL_EXPECT_BLAS=1 uv run pytest tests/test_blas_consistency.py

# 2. BLAS off build に切り替えて再生成 (scalar fallback; rayon/simd は ON 維持)
uv run maturin develop --uv --release --no-default-features \
    --features extension-module,rayon,simd
KRYANNEAL_EXPECT_BLAS=0 uv run pytest tests/test_blas_consistency.py

# 3. 2 つの artifact を diff
uv run python tools/diff_blas_artifacts.py \
    tests/artifacts/blas_on.npz tests/artifacts/blas_off.npz
```

`KRYANNEAL_EXPECT_BLAS` を渡しておくと「誤った build に対する silent 上書き
保存」を防ぐ (build mode と env var が不一致なら test 自身が skip). diff
script は default で rel < 1e-13 / atol < 1e-13 を assert. ローカル切替の
都度 BLAS on/off build を行うため小規模 (n ∈ {4,6,8}) sample のみ.

## API リファレンス

`python/kryanneal/*.pyi` (per-module PEP 484 stub) に公開 API のシグネチャと
**full docstring** がダンプされている。**`kryanneal` を使うスクリプトを
書く際はまず該当モジュールの `.pyi` を読み、必要に応じてソース実装を
参照する** (cv_ising と同方式)。`.pyi` は手書きしない。再生成:

```bash
uv run python tools/gen_api_stubs.py
```

`.pyi` ドリフト防止は二段階:

1. **Claude 編集時 (一次)**: `.claude/rules/api-stubs-sync.md` (path-scoped rule)
   が `python/kryanneal/**/*.py` または `tools/gen_api_stubs.py` 編集時にロード
   され、再生成スクリプトを同じコミットに含めるよう Claude 側で運用する。
2. **コミット時 (セーフティネット)**: `.pre-commit-config.yaml` の `gen-api-stubs`
   フックが人間の手編集も含めて取りこぼしを拾う。

## ベンチマーク

`benchmarks/` 配下に per-step 性能計測の CLI スクリプトを置く。

```bash
uv run python benchmarks/bench_per_step.py
uv run python benchmarks/bench_blas_compare.py   # BLAS feature on/off 同一マシン比較
uv run python benchmarks/bench_vs_qutip.py
uv run python benchmarks/bench_qutip_large.py    # dt sweep で QuTiP vs kryanneal の fidelity & wall time を同時測定 (issue #65)
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

## 開発作業

issue 対応や問題解決には `/solve` skill を使う (達成基準・権限境界・自動化
プロトコルは skill 側で管理)。プロジェクト固有の delta は
`.claude/solve-overrides.md` に記載 (`/solve` 起動時のみロード)。

## コーディング規約

- 数式や物理的意味を持つ変数は日本語の docstring で意味を記述する
  (cv_ising の慣習)
- `ruff` / `ty` を尊重し、型ヒントは既存スタイルに合わせる
- 数値計算の等価性を壊す変更 (演算順序の変更など) は、テストで
  machine precision での一致を確認する
- Rust 側で新しい純 Rust ヘルパを `src/` に追加するときは、対応する
  `#[cfg(test)]` テストを同じファイル内に追加する (cv_ising の `rust/src/`
  と同様、Rust 単体テストと Python pytest を二段で運用する方針)

## 物理的取り決め (繰り返し参照される基本契約)

- **Hamiltonian 形**: `H(t) = A(s(t)) · H_driver + B(s(t)) · H_problem`
  - `H_driver = -Σ_i h_x_i X_i` (サイト依存横磁場)
  - `H_problem` は Z 演算子のみで書かれた k-local 多項式 → **Z 基底で対角**
- **ユーザー入力**: `H_p_diag: (2^N,) float64` および `h_x: (N,) float64`。
  対角ベクトル自体を渡してもらう (k-local 表現はパッケージ側で扱わない)
- **ビット規約**: bit 0 = LSB、`x = Σ_i b_i · 2^i`、spin `σ_i = 1 - 2·b_i`
- **初期状態**: ユーザーが必ず明示指定 (default なし)。L2-normalize 済みを
  コンストラクタで検証 (`‖psi0‖ - 1 < 1e-10`)
- **時間発展**: 純粋状態 Schrödinger 方程式のみ。Open system は非対応

## Thread pool 運用 (rayon × BLAS)

Phase 6 C1 (issue #62) で matvec / Trotter primitives を rayon
`par_chunks_mut` で L2 並列化済み (`feature = "rayon"`, default ON)。
thread pool が 2 系統並走するため運用ルール:

- **rayon thread 数**: 環境変数 `RAYON_NUM_THREADS` で **プロセス起動時に**
  set する。rayon の global pool は最初の rayon op で構築され, その後は
  変更不可。動的変更用の新規 API は持たない。
- **BLAS thread 数**: Python 側 `kryanneal.set_blas_threads(n)` で動的に
  変えられる (`threadpoolctl.threadpool_limits` 経由)。Lanczos / CFM4 内の
  Level-1/2 BLAS が対象。
- **競合回避**: 両 pool が同時に `cpu_count` 個ずつ thread を張ると総スレッド
  数が `cpu_count^2` 相当となり context-switch で性能劣化する。**rayon 経路
  で並列化するときは `kryanneal.set_blas_threads(1)` に落として BLAS pool
  を 1 thread に固定する** ことを推奨。ベンチマークでも同様
  (`benchmarks/bench_parallel_scaling.py` は子モード冒頭でこれを強制)。
- **`--no-default-features` ビルド**: scalar 単スレッド経路に戻り rayon 依存
  なし。`RAYON_NUM_THREADS` は無視される。

## SIMD 経路 (Phase 6 C2 / C2.5, issue #63 / #71)

Phase 6 C2 (issue #63) で `apply_h_kryanneal` の bit-flip pass の i ∈
{0, 1, 2} を `wide::f64x4` 特化, C2.5 (issue #71) で同じく
`apply_single_mode_axis_i` (Trotter 経路の 2×2 ユニタリ pair update) の
i ∈ {0, 1, 2} を SIMD 特化 (`feature = "simd"`, default ON)。

- SIMD inner kernel は `src/matvec.rs::simd_kernels` モジュールに集約:
  - `bitflip_i{0,1,2}` (C2): `y[k] += coeff · v[k ^ mask]` を broadcast +
    FMA で計算.
  - `single_mode_i{0,1,2}` (C2.5): 2×2 complex matmul を **complex
    broadcast + in-register swizzle** で f64x4 化
    (`u_k · x_pair = splat(u[k].re) · x_pair + [-u[k].im, u[k].im, ...]
    · x_swap` の 2 Complex64 並列, 詳細は `simd_kernels` モジュール docstring).
- `bitflip_iN` は `apply_h_kryanneal_{serial,rayon}` の両 path から,
  `single_mode_iN` は `apply_single_mode_axis_i_{serial,rayon}` および C3 の
  `apply_fused_axes_to_chunk` (trotter 経路の multi-qubit fusion inner kernel)
  から共通で呼ばれる。per-thread 最適化なので rayon 並列化と直交する。
- rayon path では SIMD カーネルの block-aligned 前提を満たすため
  chunk_size を `SIMD_BLOCK_MAX = 8` Complex64 の倍数に丸める。fused 経路は
  group_block (= 2^(i_start+k)) の倍数で構築されるが,
  `target = dim/(nth·4)` が非 power-of-2 のとき n_groups が奇数になる
  ケースがあり, defensive な alignment check (`chunk.len() % {4,8} == 0`) を
  `apply_fused_axes_to_chunk` の SIMD dispatch に入れている。
- **実 SIMD 性能向上は build 時の `target-cpu` 設定に依存する**: default の
  `x86_64` target では `wide` が scalar fallback ([f64; 4] 相当) を選び
  正確性のみ提供する。**本番 measure は `RUSTFLAGS="-C target-cpu=native"`
  を必ず設定する** (AVX2 / AVX-512 / NEON を `wide` の `target_feature` cfg
  が拾えるようになる)。
- **`__has_simd__` / `__has_rayon__` フラグ**: `_rust.__has_simd__: bool` /
  `_rust.__has_rayon__: bool` が build profile を露出する (`__has_blas__`
  と同様)。bench スクリプト (`bench_simd_scaling.py`) はこれで build を
  識別する。bench は C2 と C2.5 で kernel 軸を分け
  (`kernel = apply_h_kryanneal / apply_single_mode_axis_i`),
  C2.5 の per-axis (`i0/i1/i2`) は `mode` 軸で別カラムに展開する。
- **`--no-default-features` ビルド**: SIMD 依存も外れ scalar 経路に戻る。
  `wide` クレートはリンクされない。

## perf 計測用 binary (Phase 6 D follow-up, issue #79 / #82 / #90)

`apply_h_kryanneal` / `trotter_step` / `apply_single_mode_axis_i` の真の
bottleneck (DRAM bound / L3 contention / barrier / chunk_size 戦略の差 等の
どれか) を Linux `perf stat` で hardware counter から特定するための
pure-Rust 計測 binary を `src/bin/` に配置:

| binary | 対象 kernel | 主な用途 |
|---|---|---|
| `src/bin/perf_apply_h.rs` | `apply_h_kryanneal` (matvec) | #79 Phase D 試行で確立した DRAM/L2 latency 計測 |
| `src/bin/perf_trotter_step.rs` | `trotter_step` (Strang 2 次 Trotter 1 step) | #82 で C3 multi-qubit gate fusion + phase_p rayon 化の真の compute speedup 検証 |
| `src/bin/perf_apply_single_mode_axis_i.rs` | `apply_single_mode_axis_i` (Trotter per-axis 2×2 ユニタリ) | #90 で #71 fixup `578d050` (動的 chunk_size) 棄却を perf binary で再評価し dynamic を採用 (詳細は `docs/design/05-1-matvec.md` §5.1.4 末尾) |

いずれも Python の `bench_*.py` が `*_py` (allocate-and-return) 経路の
alloc/copy overhead で wall-time を歪めるのを回避し,
Rust 側 micro-optimization の compute 効果だけを切り出す目的.

ビルド:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release --bin perf_apply_h
RUSTFLAGS="-C target-cpu=native" cargo build --release --bin perf_trotter_step
RUSTFLAGS="-C target-cpu=native" cargo build --release --bin perf_apply_single_mode_axis_i
```

対象関数は `pub fn` に上げ, `crate::bench_api` (`src/lib.rs`) で再 export
している (`apply_h_kryanneal`, `trotter_step`, `apply_single_mode_axis_i`).
Python 側 API (`_rust.apply_h_kryanneal_py` / `_rust.trotter_step_py` /
`_rust.apply_single_mode_axis_i_inplace_py` 等) には影響なし.

計測例 (Linux, AMD EPYC で実証済み):

```bash
# 基本: IPC + cache
RAYON_NUM_THREADS=64 perf stat \
    -e cycles,instructions,branch-misses \
    -e cache-references,cache-misses \
    -e L1-dcache-loads,L1-dcache-load-misses \
    -e dTLB-loads,dTLB-load-misses \
    -- ./target/release/perf_apply_h 20 500

# AMD Zen 3 専用: L2 fill latency / stall (issue #79 で実用したセット)
RAYON_NUM_THREADS=64 perf stat \
    -e cycles,instructions,branch-misses \
    -e stalled-cycles-backend,stalled-cycles-frontend \
    -e l2_request_g1.all_no_prefetch,l2_cache_req_stat.ic_dc_miss_in_l2 \
    -e l2_latency.l2_cycles_waiting_on_fills \
    -- ./target/release/perf_apply_h 20 500

# trotter_step (issue #82 C3 audit). per-iter cost が大きいので iter 数は
# default 500 (perf_apply_h の 1000 の半分).
RAYON_NUM_THREADS=64 perf stat \
    -e cycles,instructions,branch-misses \
    -e stalled-cycles-backend,stalled-cycles-frontend \
    -e l2_request_g1.all_no_prefetch,l2_cache_req_stat.ic_dc_miss_in_l2 \
    -e l2_latency.l2_cycles_waiting_on_fills \
    -- ./target/release/perf_trotter_step 20 500

# apply_single_mode_axis_i (issue #90 C2.5 chunk_size audit). 第 3 引数で
# axis i を指定 (default 0 = SIMD path). i ∈ {0,1,2} で SIMD path,
# i >= 3 で scalar path.
RAYON_NUM_THREADS=64 perf stat \
    -e cycles,instructions,branch-misses \
    -e stalled-cycles-backend,stalled-cycles-frontend \
    -e l2_request_g1.all_no_prefetch,l2_cache_req_stat.ic_dc_miss_in_l2 \
    -e l2_latency.l2_cycles_waiting_on_fills \
    -- ./target/release/perf_apply_single_mode_axis_i 20 500 0
```

binary は stderr に wall time / per-iter time / sink (DCE 防止) を出し,
stdout は空に保つ (perf の出力を汚さない).

比較対象の build を識別するときは `cargo build --target-dir target-<tag>`
で出力先を分ける (#79 で確立した方法論). 例: `RAYON_CHUNK_MAX` 値違いを
同時に持ちたい場合は `target-rayon14` / `target-rayon13` のように分離する.

## Phase 6 D 実験と未採用の根拠 (issue #79, 2026-05-17)

Phase D で `apply_h_kryanneal_rayon` を **連続 k 個の高 i を group-fused
3-phase 形** に書き換える試み (DRAM v traffic を `dim · (1 + h_baseline) →
dim · (1 + h_naive)` に削減する設計) を行ったが, **本 Linux サーバー
(AMD EPYC 7713P, 64 物理コア, L2 = 512 KB/core, L3 = 32 MB/CCX × 8) で
perf 計測した結果 N=20 で 50% 真の compute regression を確認** し,
revert. 詳細な perf 値と判断は `docs/design/05-1-matvec.md` §5.1.4 にアーカイブ.

要点だけ抜粋:

- C1 baseline は IPC=2.98 (Zen 3 理論 max の 60-75%) で **既に compute-near-peak**.
  「DRAM bandwidth bound だから traffic 削減すれば改善」前提が成立しなかった.
- Phase D の chunk 跨ぎ XOR access pattern が HW prefetcher を破壊し,
  per-L2-miss avg latency が 195 → 251 cycles (+30%) に劣化.
- cache-miss rate は baseline/after とも 3-7% で **DRAM access はそもそも
  少なかった**. 真の bottleneck は L2 fill latency (L3 / cross-CCX).
- N=18 は実質変化なし (Python bench で見えた 0.53× は alloc/GC noise).

issue #79 で B (SIMD i≥3), C (prefetch), D (streaming store) として残されていた
代替カードも **IPC 3.0 baseline 前提では効果薄** が予想されるため別途
sub-issue 化していない. 再挑戦時は `src/bin/perf_apply_h.rs` + perf stat で
ハードウェア counter を最初に取り「何 bound か」を確認してから設計に入る運用.


## 設計判断の出典 (cv_ising 流用箇所)

- CFM4:2 係数: `cv_ising/rust/src/cfm4.rs` の `a_high = 1/4 + √3/6` 等
  (`docs/design/05-3-propagator.md` §5.3 に inline 済み)
- PI controller の式・既定値: `cv_ising/src/cv_ising/krylov.py` の
  `evolve_schedule_adaptive_m2` / `evolve_schedule_adaptive_richardson`
  (`docs/design/05-3-propagator.md` §5.3 に inline 済み)
- maturin レイアウトの「適切な」形と stub 配置: `docs/design/03-architecture.md` §3.3, §7.6
  (PyO3/maturin#490, #771, #885 を踏まえて選定)
- BLAS feature on/off の分岐パターン: cv_ising と同じ `cfg(feature = "blas")`
  + `blas-src` (macOS=Accelerate / Linux=OpenBLAS) で揃える
