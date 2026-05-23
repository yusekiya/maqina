# CLAUDE.md

Claude Code 向けのプロジェクトガイド。

## 要望

指示がない限りユーザーへの回答や質問は日本語で行うこと。

## プロジェクト概要

`kinema` ((Kine)tic quantum evolution by (Ma)gnus expansion): 横磁場
イジングモデル (TFIM) の量子ダイナミクスを matrix-free に計算する
シミュレータ。Krylov 法 (Lanczos) または Chebyshev 多項式展開で短時間
プロパゲータを近似し、Magnus 展開 (CFM4:2) で時間依存 Hamiltonian の
時間発展演算子を近似する。adaptive dt ドライバ (step-doubling
Richardson + PI 制御) も提供。

設計の参照プロジェクト: [`cv-ising-solver`](https://github.com/Shu-Tanaka-Group/cv-ising-solver)
(同じ Krylov + CFM4:2 カーネルの連続変数版)。

- パッケージマネージャ: `uv` (Python `>=3.13`)
- ビルドバックエンド: `maturin` (Rust 拡張 `kinema._rust` を PyO3 経由でビルド)
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
kinema/
├── pyproject.toml
├── Cargo.toml                  # Rust crate ルート (maturin 標準位置)
├── src/                        # Rust ソース
│   ├── lib.rs                  # PyO3 #[pymodule] fn _rust エントリポイント
│   ├── matvec.rs               # apply_h_kinema (bit-flip + diag)
│   ├── krylov.rs               # lanczos_propagate (ndarray ベース)
│   ├── cfm4.rs                 # CFM4:2 / M2 / Richardson 推定子
│   ├── tridiag.rs              # 実対称三重対角の implicit QL (hand-rolled)
│   └── blas.rs                 # 内積 / axpy / nrm2 / scal ラッパ
├── python/kinema/           # Python ソース (python-source = "python")
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
KINEMA_EXPECT_BLAS=1 uv run pytest tests/test_blas_consistency.py

# 2. BLAS off build に切り替えて再生成 (scalar fallback; rayon/simd は ON 維持)
uv run maturin develop --uv --release --no-default-features \
    --features extension-module,rayon,simd
KINEMA_EXPECT_BLAS=0 uv run pytest tests/test_blas_consistency.py

# 3. 2 つの artifact を diff
uv run python tools/diff_blas_artifacts.py \
    tests/artifacts/blas_on.npz tests/artifacts/blas_off.npz
```

`KINEMA_EXPECT_BLAS` を渡しておくと「誤った build に対する silent 上書き
保存」を防ぐ (build mode と env var が不一致なら test 自身が skip). diff
script は default で rel < 1e-13 / atol < 1e-13 を assert. ローカル切替の
都度 BLAS on/off build を行うため小規模 (n ∈ {4,6,8}) sample のみ.

## API リファレンス

`python/kinema/*.pyi` (per-module PEP 484 stub) に公開 API のシグネチャと
**full docstring** がダンプされている。**`kinema` を使うスクリプトを
書く際はまず該当モジュールの `.pyi` を読み、必要に応じてソース実装を
参照する** (cv_ising と同方式)。`.pyi` は手書きしない。再生成:

```bash
uv run python tools/gen_api_stubs.py
```

`.pyi` ドリフト防止は二段階:

1. **Claude 編集時 (一次)**: `.claude/rules/api-stubs-sync.md` (path-scoped rule)
   が `python/kinema/**/*.py` または `tools/gen_api_stubs.py` 編集時にロード
   され、再生成スクリプトを同じコミットに含めるよう Claude 側で運用する。
2. **コミット時 (セーフティネット)**: `.pre-commit-config.yaml` の `gen-api-stubs`
   フックが人間の手編集も含めて取りこぼしを拾う。

## ベンチマーク

`benchmarks/` 配下に per-step 性能計測の CLI スクリプトを置く。

```bash
uv run python benchmarks/bench_per_step.py
uv run python benchmarks/bench_blas_compare.py   # BLAS feature on/off 同一マシン比較
uv run python benchmarks/bench_vs_qutip.py
uv run python benchmarks/bench_qutip_large.py    # work-precision diagram で QuTiP vs kinema を Pareto 比較 (issue #65)
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
  変更不可。動的に縮小する手段は無く, 環境変数が一次的な制御手段。
  未設定時 default は `std::thread::available_parallelism()` (≒ 論理コア数,
  SMT/HT 込み; Linux では `sched_getaffinity` / cgroup を尊重)。
- **BLAS thread 数**: env で **pool size** を確定し, ランタイムで **active
  thread 数** を補助的に調整する 2 段:
  - **pool size** (= プロセスが確保する OS thread 数): 起動時 env で確定。
    Linux OpenBLAS は `OPENBLAS_NUM_THREADS`, MKL は `MKL_NUM_THREADS`,
    macOS Apple Accelerate は `VECLIB_MAXIMUM_THREADS`, fallback の
    OpenMP 系は `OMP_NUM_THREADS`。複数 pool 同居時は全 env を揃える。
  - **active thread 数** (= 並列 BLAS op で実際に使う thread): Python 側
    `kinema.set_blas_threads(n)` で動的に変えられる
    (`threadpoolctl.threadpool_limits` 経由)。**pool size 自体は縮まらない**
    (sleeping thread の stack / kernel resource は残る) ので, per-process
    thread budget の隔離が要件なら env 設定が必須。
- **競合回避と推奨 default (issue #116, 2026-05-21 改訂)**: rayon 経路でも
  Lanczos 内部の BLAS-1 (`gemv` / `axpy` / `nrm2` 等) は **適度な BLAS 並列化が
  prod 速い** ことが Linux AMD EPYC 7713P での perf sweep (#113 / PR #115) で
  実証された。NT=8 で **1.52× speedup** (NT=1 baseline 比), NT=16-32 でも
  +2% 程度の劣化で許容範囲, NT=64 で -9% に明確な劣化という curve。
  **新方針**:
  - 既定値は `kinema.set_blas_threads_auto()` を import 後に 1 度呼ぶ。
    内部で `os.process_cpu_count() // 8` を 1-16 でクランプし
    (EPYC SMT2 で 16, 32-core で 4, 8-core 以下で 1), さらに
    `OPENBLAS_NUM_THREADS` / `MKL_NUM_THREADS` / `VECLIB_MAXIMUM_THREADS` /
    `OMP_NUM_THREADS` (この優先順) が set されていればそれを strict な
    上限として `min(auto, env_cap)` を返す。冪等。
  - 完全隔離が要件なら `set_blas_threads(1)` を明示。あるいは起動前に
    env で `OPENBLAS_NUM_THREADS=1` 等を set。
  - 旧推奨「rayon 経路では `set_blas_threads(1)`」は perf 計測前の仮説で
    あり, 1.52× の改善余地を逃していた。**撤回済み**。
  ベンチ運用 (`benchmarks/bench_parallel_scaling.py` 子モード冒頭の強制) も
  同じく `set_blas_threads_auto()` 経由に揃える方向。当初懸念だった
  「spin-wait が rayon を圧迫」も実害無し (`OPENBLAS_THREAD_TIMEOUT=1` で spin
  抑制すると逆に +4-9% 遅化する。Lanczos の BLAS call 間隔が短く futex
  park → unpark の wake-up cost が遊休 core 占有より高くつくため)。
  - **本番 perf bench (Pareto / QuTiP 比較 等) の運用** (2026-05-21 確定):
    `bench_qutip_large.py` / `bench_per_step.py` の `--blas-threads N` フラグに
    **NT=8 を明示渡す** のを default にする。理由は上記 sweep の sweet spot で,
    `--blas-threads` 不指定 (= OpenBLAS の物理コア数 default で 64 threads) だと
    NT=8 比 ~1.10× 遅化する (PR #106 の 0.8.0 bench で実測, PR #106
    コメントに対比表)。`set_blas_threads_auto()` は EPYC 7713P で同じく NT=8 を
    自動算出するので, "machine 種別を意識せず使いたい" 場合は auto setter を
    bench script 冒頭で 1 度呼んでも等価。`--blas-threads 1` は machine-
    independent baseline (= 純シリアル比較) 用途で本番 perf bench とは別の
    意味づけ。
- **並列ジョブ実行 (multiprocessing / Slurm job array 等)**: 1 プロセス
  あたりの thread budget を絞るには **`kinema` / `numpy` を import する前**
  に上記 env (`RAYON_NUM_THREADS` / `OPENBLAS_NUM_THREADS` / `MKL_NUM_THREADS`
  / `VECLIB_MAXIMUM_THREADS` / `OMP_NUM_THREADS`) を一括 set する必要がある
  (BLAS / rayon の pool size は最初の op で確定し以降縮小不可)。具体的な
  multiprocessing パターン例は `docs/quickstart.md` 末尾節を参照。Slurm
  などジョブスケジューラの `cpuset` / cgroup で絞られていれば rayon
  `available_parallelism()` がそれを反映するので, env 未設定でも妥当に
  動くことが多いが, BLAS pool は cgroup を honor しない実装もあるため
  明示推奨。
- **`--no-default-features` ビルド**: scalar 単スレッド経路に戻り rayon 依存
  なし。`RAYON_NUM_THREADS` は無視される。

## SIMD 経路 (Phase 6 C2 / C2.5, issue #63 / #71)

Phase 6 C2 (issue #63) で `apply_h_kinema` の bit-flip pass の i ∈
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
- `bitflip_iN` は `apply_h_kinema_{serial,rayon}` の両 path から,
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
  正確性のみ提供する。issue #103 で **repo 同梱の `.cargo/config.toml` に
  `[build] rustflags = ["-C", "target-cpu=native"]` を入れて default 適用**
  しているため, `uv add git+...` / `cargo build` / `maturin develop` のどの
  経路でも build マシン CPU の AVX2 / AVX-512 / NEON が `wide` の
  `target_feature` cfg で自動的に拾われる。明示的に `RUSTFLAGS` を渡したい
  ときは env 経由が優先されるので override 可。
- **build profile 確認フラグ**: cargo feature 有効化を `_rust.__has_simd__`
  / `__has_rayon__` / `__has_blas__` (各 bool, `cfg!(feature = ...)` 由来),
  target_feature 有効化を `_rust.__has_avx2__` / `__has_fma__` /
  `__has_avx512f__` / `__has_neon__` (各 bool, `cfg!(target_feature = ...)`
  由来), ビルドターゲットを `_rust.__target_arch__` / `__target_os__`
  (各 str, `std::env::consts` 由来) が expose する (`m.add` 経由, build.rs
  不要)。ユーザー向けには `kinema.show_config()` (numpy.show_config 相当)
  でこれらを集約 dump できる (issue #103, 詳細は
  `docs/design/11-build-infrastructure.md` §11.1)。bench スクリプト
  (`bench_simd_scaling.py`) はこれで build を識別する。bench は C2 と C2.5 で
  kernel 軸を分け (`kernel = apply_h_kinema / apply_single_mode_axis_i`),
  C2.5 の per-axis (`i0/i1/i2`) は `mode` 軸で別カラムに展開する。
- **`--no-default-features` ビルド**: SIMD 依存も外れ scalar 経路に戻る。
  `wide` クレートはリンクされない。

## perf 計測用 binary (Phase 6 D follow-up, issue #79 / #82 / #90 / #113 / #120)

`apply_h_kinema` / `trotter_step` / `apply_single_mode_axis_i` /
`cfm4_adaptive_richardson_krylov` / `chebyshev_propagate` の真の bottleneck
(DRAM bound / L3 contention / barrier / chunk_size 戦略の差 / Lanczos vs GS の
wall 比率, Lanczos vs Chebyshev のアルゴリズム軸 等のどれか) を Linux
`perf stat` で hardware counter から特定するための pure-Rust 計測 binary を
`src/bin/` に配置:

| binary | 対象 kernel | 主な用途 |
|---|---|---|
| `src/bin/perf_apply_h.rs` | `apply_h_kinema` (matvec) | #79 Phase D 試行で確立した DRAM/L2 latency 計測 |
| `src/bin/perf_trotter_step.rs` | `trotter_step` (Strang 2 次 Trotter 1 step) | #82 で C3 multi-qubit gate fusion + phase_p rayon 化の真の compute speedup 検証 |
| `src/bin/perf_apply_single_mode_axis_i.rs` | `apply_single_mode_axis_i` (Trotter per-axis 2×2 ユニタリ) | #90 で #71 fixup `578d050` (動的 chunk_size) 棄却を perf binary で再評価し dynamic を採用 (詳細は `docs/design/05-1-matvec.md` §5.1.4 末尾) |
| `src/bin/perf_cfm4_richardson.rs` | `cfm4_step_with_richardson_estimate` (Richardson 1 step = 6 Lanczos call) | #113 で Phase 9+ scoping のため component 別 wall % を実測 breakdown. `full` / `single_lanczos` / `matvec_only` / `gram_schmidt` の 4 mode を持ち, "step → Lanczos call → matvec / GS" の各層を同一 PMU セットで比較する |
| `src/bin/perf_chebyshev.rs` | `chebyshev_propagate` (時間独立 H, Chebyshev 3 項漸化) | #120 Phase A POC で Lanczos の V matrix cache stall を **アルゴリズム軸で bypass** する Chebyshev 経路の per-call wall を Linux で実測. `perf_cfm4_richardson 18 100 single_lanczos` (Lanczos baseline ~129 ms / IPC=0.78) と直接比較し, 判定 gate (≤ 50 ms で Phase B 進行 / 50-100 ms で設計再検討 / > 100 ms で中止) を判断する. 時間独立 frozen schedule `a_t = b_t = 0.5` で Lanczos baseline と input pattern を完全一致させる |
| `src/bin/perf_cfm4_richardson_chebyshev.rs` | `cfm4_step_chebyshev_with_richardson_estimate` (Chebyshev variant Richardson 1 step) | #122 Phase B で Chebyshev を CFM4 Magnus + step-doubling Richardson に統合した後の per-step wall + K_used を Linux で実測 breakdown. 3 mode (`full` / `single_chebyshev` / `matvec_only`) を持ち, 既存 `perf_cfm4_richardson` の同名 mode (`full` / `single_lanczos` / `matvec_only`) と直接比較することで Chebyshev vs Lanczos の compute 効果差を IPC / L2 fill latency / Stalled cycles まで掘れる. `gram_schmidt` mode は Chebyshev では原理的に存在しない (3 項漸化が直交保証, re-orthogonalization 不要) |

いずれも Python の `bench_*.py` が `*_py` (allocate-and-return) 経路の
alloc/copy overhead で wall-time を歪めるのを回避し,
Rust 側 micro-optimization の compute 効果だけを切り出す目的.

ビルド:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release --bin perf_apply_h
RUSTFLAGS="-C target-cpu=native" cargo build --release --bin perf_trotter_step
RUSTFLAGS="-C target-cpu=native" cargo build --release --bin perf_apply_single_mode_axis_i
RUSTFLAGS="-C target-cpu=native" cargo build --release --bin perf_cfm4_richardson
RUSTFLAGS="-C target-cpu=native" cargo build --release --bin perf_chebyshev
RUSTFLAGS="-C target-cpu=native" cargo build --release --bin perf_cfm4_richardson_chebyshev
```

対象関数は `pub fn` に上げ, `crate::bench_api` (`src/lib.rs`) で再 export
している (`apply_h_kinema`, `trotter_step`, `apply_single_mode_axis_i`,
`lanczos_propagate`, `cfm4_step_with_richardson_estimate`,
`chebyshev_propagate`; あと `gram_schmidt` mode が直接呼ぶ BLAS-1 primitive
として `dot_conj` / `axpy`).
Python 側 API (`_rust.apply_h_kinema_py` / `_rust.trotter_step_py` /
`_rust.apply_single_mode_axis_i_inplace_py` /
`_rust.cfm4_step_with_richardson_estimate_py` 等) には影響なし.
`chebyshev_propagate` は POC Phase A 段階では Python binding を持たず
(`_rust` に登録しない), perf binary 経由のみで使う.

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

# cfm4_step_with_richardson_estimate (issue #113). 第 3 引数で mode 切替:
# full (default) / single_lanczos / matvec_only / gram_schmidt. 4 mode 全部
# 同じ counter セットで取って "step → Lanczos call → matvec / GS" 各層の
# wall % を実測 breakdown する.
for mode in full single_lanczos matvec_only gram_schmidt; do
    RAYON_NUM_THREADS=64 perf stat \
        -e cycles,instructions,branch-misses \
        -e stalled-cycles-backend,stalled-cycles-frontend \
        -e l2_request_g1.all_no_prefetch,l2_cache_req_stat.ic_dc_miss_in_l2 \
        -e l2_latency.l2_cycles_waiting_on_fills \
        -- ./target/release/perf_cfm4_richardson 18 100 $mode
done

# chebyshev_propagate (issue #120 POC). 第 3 引数で tol 切替 (default 1e-10).
# perf_cfm4_richardson 18 100 single_lanczos と直接比較する想定なので n_iters=100
# default. K_used 平均 (stderr) で Chebyshev 切り捨て次数の実測値も同時に取れる.
RAYON_NUM_THREADS=64 perf stat \
    -e cycles,instructions,branch-misses \
    -e stalled-cycles-backend,stalled-cycles-frontend \
    -e l2_request_g1.all_no_prefetch,l2_cache_req_stat.ic_dc_miss_in_l2 \
    -e l2_latency.l2_cycles_waiting_on_fills \
    -- ./target/release/perf_chebyshev 18 100
```

binary は stderr に wall time / per-iter time / sink (DCE 防止) を出し,
stdout は空に保つ (perf の出力を汚さない).

比較対象の build を識別するときは `cargo build --target-dir target-<tag>`
で出力先を分ける (#79 で確立した方法論). 例: `RAYON_CHUNK_MAX` 値違いを
同時に持ちたい場合は `target-rayon14` / `target-rayon13` のように分離する.

## Phase 6 D 実験と未採用の根拠 (issue #79, 2026-05-17)

Phase D で `apply_h_kinema_rayon` を **連続 k 個の高 i を group-fused
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


## Phase 7 (issue #93): Lanczos β_m exposure + Richardson 誤差源分離

Phase 6 完了後の follow-up. CFM4 adaptive Richardson driver が #65 long-T
シナリオで QuTiP に Pareto 劣位だった原因 (Richardson 推定子が Magnus 誤差と
Krylov 誤差を区別できない) を解消するための **infrastructure** を導入.

主要 API 変更:

- **`lanczos_propagate` (Rust + Python ref)**: return tuple が 4 要素
  `(psi, m_eff, β_m, |c_m|)` に拡張. 末尾 2 要素は Saad/Hochbruck-Lubich の
  a posteriori 誤差推定子 (`err_lanczos ≈ β_m · |c_m| · ‖ψ‖ · dt / m_eff`,
  5% 精度; `tools/verify_beta_m_estimator.py` で 108 cell sweep 実証).
- **`cfm4_step` / `cfm4_step_with_richardson_estimate`**: triangle
  inequality で `err_lanczos_sum` / `err_lanczos_total` を集約して上位伝播.
- **`evolve_schedule_adaptive_richardson`**: return tuple が 10 要素に拡張
  (`+ beta_m_history`, `err_lanczos_history`, `err_magnus_history`,
  `n_krylov_insufficient`). PI controller の駆動量を `err_magnus = max(0,
  err - err_lanczos_total)` に切替え.
- **`QuantumResult`**: `beta_m_stats` / `n_krylov_insufficient` フィールド追加.
- **`benchmarks/bench_qutip_large.py`**: `--krylov-tols` sweep で `atol ×
  krylov_tol` のクロス評価を可能に. `auto` キーワードで内部自動結合
  (= `tol_step * 1e-3`) を表現.

後方互換性: default `krylov_tol = 1e-12` では `err_lanczos << tol_step` で
`err_magnus ≈ err`. 既存 PI controller 挙動とほぼ等価
(`tests/test_adaptive.py::test_adaptive_richardson_error_decomposition_consistency`).

bench acceptance (Linux AMD EPYC 7713P, 2026-05-18):

- ✅ **Safety net 機能**: `bench_qutip_large.py --scenarios long-T
  --n-values 8,10 --krylov-tols auto,1e-8,1e-6` で `krylov_tol` を 4 桁緩めても
  `n_steps_eff` 差 0.01-0.02%, wall time 差 ±2%. PI controller が relaxed
  Krylov 設定下でも安定動作.
- ❌ **Pareto 劣位は未解消**: TFIM Lanczos の中間 β_j 値が O(‖H‖) で,
  `krylov_tol=1e-6` でも閾値を超えず m_eff=m_max=24 固定. Lanczos 圧縮そのもの
  が発火しないため Pareto は 2.5-8× 劣位のまま. 真の bottleneck は Richardson
  の構造的 6 Lanczos call / step. Phase 7 は **そこに到達するための前提
  infrastructure** として完了, Pareto win は follow-up に移管.

Follow-up issues:

- **#98 (Phase 8 で消化)**: Lanczos a posteriori 早期打切. Phase 7 で expose した
  推定子を Lanczos 内部の打切判定そのものに使う (下記 Phase 8 節).
- **#97**: Richardson 構造的 overhead 削減 (embedded estimator / time-reuse /
  adaptive frequency)

詳細は `docs/design/12-release-plan.md` Phase 7 / `docs/design/05-3-propagator.md`
"Richardson 誤差源分離" 節.

## Phase 8 (issue #98): Lanczos a posteriori 早期打切 (`krylov_tol` 意味再定義)

Phase 7 で expose した `β · |c|` a posteriori 推定子を **Lanczos 内部の早期打切
判定そのもの** に組み込み, Phase 7 で "infrastructure 完了 / Pareto 未解消"
だった #65 / #94 の本丸 (= Lanczos 圧縮を実際に発火させる) に踏み込む.

### 判定式と意味再定義

| 量 | Phase 7 まで | Phase 8 (現在) |
|---|---|---|
| `krylov_tol` の意味 | β 単体閾値 | **Krylov 近似の許容誤差** |
| Lanczos 早期打切判定 | `β_k < krylov_tol` (実用で発火しない) | `β_k · \|c_last\| · \|dt\| / (k+1) < krylov_tol` (Hochbruck-Lubich 1997) |
| β 単体の役割 | 打切判定 | numerical breakdown safety (`< 1e-14` で `v_{k+1} = w / β_k` の division by zero 回避のみ) |

`‖ψ‖ = 1` は Lanczos 内部の正規化空間規約 (`v_0 = ψ / ‖ψ‖`). 物理状態ノルム
は最終 `ψ_new = ‖ψ‖ · V · c` で復元するので, 判定式から `‖ψ‖` ファクタは
除外できる.

### API 互換性 / セマンティクス変更

公開 API シグネチャは **不変**:
- `QuantumAnnealer(krylov_tol=None)` / `AnnealingSimulator(krylov_tol=None)`
  の auto-resolve ロジック (adaptive: `tol_step · 1e-3`, fixed-dt: `1e-12`) も
  そのまま継承.
- 同じ default 値 (1e-11 / 1e-12) を渡しても **挙動が変わる** (旧: m_eff = m_max
  固定, 新: m_eff ≪ m_max になる scenario が増える). 数値結果 (`ψ_new`) は
  誤差内で一致するが `m_eff_history` 系統計値は変動.

このセマンティクス変更を伴うので **minor bump (`0.7 → 0.8`)**.

### Lanczos 内部 c の規約変更

内部 c 配列は `psi_norm` 抜きで保持し, 終端で `ψ_new = ‖ψ‖ · V · c` の gemv
coeff に畳み込む形にリファクタ. これにより判定式 `β · |c| · |dt| / m <
krylov_tol` が `‖ψ‖ = 1` 規約と意味的に整合し, `c_m_abs` (return 値の `|c_m|`)
も自然に "pure な行列要素" (`‖ψ‖` 抜き = literature 標準) で返せる.

### 詳細

- `docs/design/05-2-lanczos.md` "a posteriori 早期打切 (issue #98 Phase 8)" 節
  (旧仕様の問題 / 判定式 / overhead 試算).
- `docs/design/12-release-plan.md` Phase 8 (Definition of Done / Bench acceptance).
- `src/krylov.rs::tridiag_c_last_abs` / `python/kinema/krylov.py::_tridiag_c_last_abs`
  (per-iter ヘルパ; Rust ↔ Python ref `rel < 1e-13` 一致).

## Phase 8 follow-up (issue #100): Richardson iter-0 matvec memoization

Phase 8 で per-Lanczos call の m_eff 圧縮が達成された後の小規模直交最適化.
`cfm4_step_with_richardson_estimate` の **full_step stage 1** と **half_1
stage 1** は同じ入口 ψ から始まるため iter 0 で使う primitive matvec
(`H_drv · ψ` / `H_p_diag · ψ`) が共通. これを入口で 1 度だけ計算し両 Lanczos
call で再利用することで **2 個の primitive matvec / Richardson step** を削減
(削減量見積もり ~3% 純減; bench acceptance は「速くなれば accept」).

実装ポイント:

- `src/matvec.rs::apply_h_drv` / `apply_h_p_diag`: cache 計算専用 primitive.
  既存 `apply_h_kinema` の cache-blocked 形は **維持** (hot path 触らない).
  primitive は Richardson 入口で 1 step 1 回のみ呼ばれるので SIMD 非適用,
  rayon は MIN_RAYON_DIM 閾値で本体と同じ dispatch.
- `src/cfm4.rs::cfm4_step` のシグネチャに crate-internal `iter0_cache:
  Option<(&[Complex64], &[Complex64])>` 引数を追加. Lanczos に渡す matvec
  closure 内で `first_call` フラグを持たせ iter 0 のときだけ cache 線形結合
  `y = (c_drv_1 · cache_drv + c_diag_1 · cache_diag) / ‖ψ‖` に差し替える.
  Lanczos API (`lanczos_propagate`) 自体は不変.
- Public Python API のシグネチャは不変. crate-internal の `cfm4_step` 引数追加
  のみで, Python wrap (`cfm4_step_py`) は `iter0_cache = None` を渡して従来通り.

数値同等性: cache あり/なしで `rel < 2e-15` (machine epsilon の数倍).
詳細は `docs/design/05-1-matvec.md` §5.1.1.x / `docs/design/05-3-propagator.md`
"iter-0 primitive matvec memoization" / `docs/design/12-release-plan.md`
Phase 8 follow-up.

## Phase B (issue #122): Chebyshev propagator を CFM4 adaptive Richardson 経路に統合

Phase A (issue #120, PR #121) で時間独立 H 単体での `chebyshev_propagate` が
per-call 29 ms / 4.45× Lanczos 高速 (Linux AMD EPYC 7713P) を達成したことを
受け, 時間依存 H + CFM4 Magnus + step-doubling Richardson + PI controller 経路
に統合した variant を公開 Python API レベルで露出する.

### 公開 API 変更 (hard rename + 新 method 追加)

- 既存 `method="cfm4_adaptive_richardson"` を
  **`method="cfm4_adaptive_richardson_krylov"`** に hard rename
  (alias なし; pre-1.0 なので破壊的変更 OK). `_krylov` / `_chebyshev` で
  suffix 対称化.
- 新 `method="cfm4_adaptive_richardson_chebyshev"` を追加. Chebyshev 経路は
  Rust 拡張必須 (Python ref fallback 非提供).
- Rust 関数名は rename せず (`cfm4_step` / `cfm4_step_with_richardson_estimate`
  が Lanczos default, Chebyshev variant は `_chebyshev` suffix で対称).
- `m_max` を Chebyshev method で渡すと `ValueError` (Krylov 部分空間次元の
  概念なし; K_used は `chebyshev_tol` から動的決定).

### 主要実装ポイント

- `src/cfm4.rs::cfm4_step_chebyshev` / `cfm4_step_chebyshev_with_richardson_estimate`:
  既存 Lanczos 版と完全同じ 2 stage + step-doubling Richardson 構造を保ち,
  短時間プロパゲータだけが `chebyshev_propagate` に入れ替わる. per-stage で
  Gershgorin による `(E_c, R)` 再計算 (closed-form O(N), wall % 無視可).
- `evolve_schedule_adaptive_richardson_chebyshev` (Python driver): 既存
  Lanczos driver と同じ 10-tuple shape / PI controller 構造. `err_magnus =
  max(0, err - err_chebyshev_total)` で Magnus 起因駆動量を分離.
- `QuantumResult` の K_used 統計は既存 `m_eff_stats` スロットを流用
  (semantically 「per-step propagator 評価コスト統計」で同じ役割; method
  literal で Lanczos / Chebyshev を判別).
- iter-0 cache (Lanczos #100 の流用) は scope 外 (per-stage K_used ~20 個の
  matvec のうち 1 個と削減比小).

### 詳細

- `docs/design/05-3-propagator.md` "CFM4:2 + Chebyshev variant" 節
  (アルゴリズム / メモリ / cache 戦略).
- `docs/design/12-release-plan.md` Phase B (Definition of Done / 判定 gate).
- `tests/test_chebyshev.py` (QuTiP fidelity + Lanczos 一致 + annealer/simulator
  smoke + m_max ValueError).
- `tests/test_blas_consistency.py` 末尾 (Chebyshev direct call artifact dump;
  adaptive driver の dt 履歴分岐を避けるため Rust step 関数を fixed schedule
  係数で直接呼ぶ).

## Phase B follow-up (issue #126): Chebyshev 3 項漸化 inner loop の SIMD + fusion

Phase B 完了直後の直交最適化. `chebyshev_propagate` の k ≥ 2 hot loop は旧実装
で 3 つの dim-walk (walk 1: matvec, walk 2: recurrence scaling scalar, walk 3:
accumulate scalar) を発生させていたが, walk 2 / walk 3 を **1 dim-walk +
`wide::f64x4` SIMD** に fuse する.

- `src/chebyshev.rs::simd_kernels::chebyshev_recurrence_fused` (SIMD) /
  `chebyshev_recurrence_fused_scalar` (scalar fallback) + dispatch wrapper.
  `chebyshev_propagate` の k ≥ 2 hot loop だけ差し替え, k = 1 step は one-shot で
  scalar のまま (overhead 無視可).
- `cfm4_step_chebyshev_*` 経由でも自動で乗る (同じ `chebyshev_propagate` を
  呼ぶため).
- f64x4 helpers (`as_f64_slice` / `load/store_f64x4_unaligned` / `swap_reim`)
  は localize duplication で chebyshev module 内に持つ (`matvec.rs::simd_kernels`
  と同じパターンを再実装; visibility 経路を跨いだ変更を避ける).
- 数値同等性: `simd_kernels::chebyshev_recurrence_fused` ↔ `_scalar` の
  100-iter fuzz テスト (`chebyshev_recurrence_fused_simd_matches_scalar`,
  `rel < 1e-13`). FMA 折りたたみと lane 演算順序差で ulp 差は出るが ≤ 1e-13.
- bench acceptance (Linux AMD EPYC 7713P, NT=64): per-step wall 10%+ で full
  merge / 5-10% で marginal accept / < 5% で 中止. 計測は `perf_chebyshev 18 100`
  + `perf_cfm4_richardson_chebyshev 18 100 full` の 2 軸.
- 詳細: `docs/design/05-3-propagator.md` "Chebyshev recurrence の SIMD + fusion" /
  `docs/design/12-release-plan.md` "Phase B follow-up: Chebyshev 3 項漸化 inner
  loop の SIMD + fusion (#126)".

## Phase B follow-up (issue #127): Chebyshev non-matvec inner loop の rayon 並列化

#126 の SIMD + fusion 完了後の直交最適化. #124 perf archive で **Chebyshev の
parallel efficiency が 64 thread で 44%** (Lanczos 27% より良いが理想 100% には
程遠い) と判明. `apply_h_kinema` は #62 で rayon 並列化済だが,
`chebyshev_recurrence_fused` (k_ord ≥ 2 hot loop) が **scalar single-thread** で
走っており, ここがスケーリング bottleneck の一部.

- 実装: `src/chebyshev.rs::chebyshev_recurrence_fused_rayon` (rayon path).
  `scratch` / `psi_acc` の 2 RW slice を `par_chunks_mut` 2 本独立に取って
  `zip()`, `enumerate()` で base offset から `phi_curr` / `phi_prev` (R) を共有
  sub-slice 切り出し. chunk 内で `simd_kernels::chebyshev_recurrence_fused`
  (SIMD ON) または `chebyshev_recurrence_fused_scalar` (SIMD OFF) を呼ぶ 2 段構造.
- `chebyshev_recurrence_fused` dispatch wrapper を 3 段に拡張: rayon ON +
  `dim >= MIN_RAYON_DIM_CHEB` → rayon path / simd ON + 偶数長 → single-thread
  SIMD / それ以外 → scalar fused.
- chunk_size は matvec.rs の `apply_h_kinema_rayon` と同じ式
  `(dim / (nth * 4)).clamp(RAYON_CHUNK_MIN_CHEB, RAYON_CHUNK_MAX_CHEB)`. SIMD
  kernel の偶数長前提を満たすため 2 倍数に丸める (min/max 共 2 倍数なので
  invariant 不変).
- dispatch 閾値 `MIN_RAYON_DIM_CHEB` 初期値は `matvec.rs::MIN_RAYON_DIM = 1 << 17`
  と揃える. Chebyshev non-matvec hot loop は matvec より per-element cost が
  小さい (memory bound) ため本来はより低い閾値でも改善が出る可能性があるが,
  PoC 段階では保守寄りで始め, 本番 bench (N ∈ {14, 16, 18, 20} sweep) で tuning.
- `cfm4_step_chebyshev_*` 経由でも自動で乗る (同じ `chebyshev_propagate` を
  呼ぶため).
- 数値同等性: rayon path と single-thread SIMD/scalar fused の random fuzz
  10-iter テスト (`chebyshev_recurrence_fused_rayon_matches_serial`,
  `rel < 1e-13`). N=17 end-to-end の rayon path 経由 unitarity smoke
  (`chebyshev_propagate_rayon_path_smoke`).
- bench acceptance (Linux 本番サーバー, perf binary 計測; **本番計算環境とは
  別マシンで CPU 性能は本番より低い**): N=18 で per-step wall 10%+ 改善 + N=12
  (or N=14) で 5% 未満劣化 → full merge / N=18 改善 5-10% + dim 小劣化 5-15% →
  `MIN_RAYON_DIM_CHEB` を上げる方向で閾値 tuning 継続 / N=18 改善 5% 未満 →
  中止 + archive. 計測は `perf_chebyshev N 100` と
  `perf_cfm4_richardson_chebyshev N 100 full` を N ∈ {14, 16, 18, 20} ×
  RAYON_NUM_THREADS ∈ {1, 8, 16, 32, 64} で sweep.
- 詳細: `docs/design/05-3-propagator.md` "Chebyshev non-matvec inner loop の
  rayon 並列化" / `docs/design/12-release-plan.md` "Phase B follow-up:
  Chebyshev non-matvec inner loop の rayon 並列化 (#127)".

## Phase B follow-up (issue #124): Default method を Chebyshev variant に切替 + atol 仕様明文化

Phase B 本体 (#122) + #126 / #127 の perf 結果 (N=18 で Lanczos 比 5.49× wall
高速, branch-miss 158× 減, sys time 78× 減, parallel efficiency 27% → 44%) を
受けて, **judgement 系の follow-up** を確定. semantic 変更を伴うため
`0.10.0 → 0.11.0` で minor bump.

### Default method 切替

- `QuantumAnnealer.run(method=...)`: 旧 `"m2"` → `"cfm4_adaptive_richardson_chebyshev"`.
- `QuantumAnnealer.create_simulator(method=...)`: 旧 `"cfm4"` → 同上. ついでに
  `Literal` から欠落していた `_chebyshev` を追加 (Phase B #122 取りこぼし fixup).
- `AnnealingSimulator(method=...)`: 旧 `"cfm4"` → 同上.
- `docs/quickstart.md` の主例: `method=` 指定を削除 (default を使う形に統一).
- `bench_qutip_large.py --solvers` default は両 method を含む `_VALID_SOLVERS`
  全列挙のまま (Pareto 比較用なので両者走らせる方が有用). `_krylov` は literal
  として永続的に残す (旧 default 互換 + 比較ベンチ用途).

旧 default (`method="m2"` / `"cfm4"`) を使っていたユーザー向け migration: 新
default は **adaptive PI controller** を走らせるので `n_steps` の代わりに `atol`
で精度を制御する. 旧挙動を維持したい場合は `method="m2"` / `"cfm4"` を明示する.

### "Accidental 高精度" 仕様 (Chebyshev での atol の振舞い)

Chebyshev では `atol` (= PI controller の `tol_step`) は **upper bound** として
機能し, K_used 動的拡張により実際の精度がそれより良くなる場合がある (例:
`atol=1e-3` 設定で n=10 で `infidelity < 1e-16`). これは "feature" として
受け入れる方針 (issue #124 Scope 2 (a) + (d) 確定):

- `atol` で要求した精度を下回ることはない (予防的上限として機能).
- 速度を取りたいときは `atol` を大きくして PI step 数を減らすのが正しい使い方.
  `chebyshev_tol` を直接緩めても K_used が数個減るだけで wall-time 効果は限定的.
- default の auto-coupling 係数 `_KRYLOV_TOL_ATOL_RATIO = 1e-3` は変更しない.

明文化先:

- `QuantumAnnealer.run` / `AnnealingSimulator.__init__` の `atol` docstring に
  "Note (Chebyshev variant の atol 振舞い, issue #124)" 注を追加.
- `docs/design/05-3-propagator.md` "Chebyshev variant" 節に "`chebyshev_tol` と
  `atol` の関係 — accidental 高精度 (issue #124)" 小節を追加.
- `docs/quickstart.md` の主例下に Note を追加.

### 詳細

- `docs/design/12-release-plan.md` "Phase B follow-up: Default method を
  Chebyshev variant に切替 + atol 仕様明文化 (#124)" 節 (Definition of Done /
  migration note).

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
