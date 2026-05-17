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

`docs/design.md` が一次資料。実装に着手する前に必ず読む。主要セクション:

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
    ├── design.md               # 一次設計書
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

## 設計判断の出典 (cv_ising 流用箇所)

- CFM4:2 係数: `cv_ising/rust/src/cfm4.rs` の `a_high = 1/4 + √3/6` 等
  (`docs/design.md` §5.3 に inline 済み)
- PI controller の式・既定値: `cv_ising/src/cv_ising/krylov.py` の
  `evolve_schedule_adaptive_m2` / `evolve_schedule_adaptive_richardson`
  (`docs/design.md` §5.3 に inline 済み)
- maturin レイアウトの「適切な」形と stub 配置: `docs/design.md` §3.3, §7.6
  (PyO3/maturin#490, #771, #885 を踏まえて選定)
- BLAS feature on/off の分岐パターン: cv_ising と同じ `cfg(feature = "blas")`
  + `blas-src` (macOS=Accelerate / Linux=OpenBLAS) で揃える
