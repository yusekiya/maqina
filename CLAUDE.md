# CLAUDE.md

Claude Code 向けのプロジェクトガイド。

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
- Rust: `pyo3 0.28`, `numpy 0.28`, `ndarray 0.16`, `ndarray-linalg 0.17`,
  `num-complex 0.4`, `cblas 0.5` (optional)

## 設計書

`docs/design.md` が一次資料。実装に着手する前に必ず読む。主要セクション:

- §3 アーキテクチャ / ディレクトリレイアウト
- §4 公開 Python API (`IsingProblem`, `Schedule`, `QuantumAnnealer`, ...)
- §5 数値カーネル (Lanczos, M2, CFM4:2, Richardson adaptive 含む)
- §7 Rust 拡張 (BLAS feature, maturin 標準レイアウト準拠の根拠)
- §8 QuTiP 比較
- §12 段階リリース計画 (Phase 1-5)

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
│   ├── krylov.rs               # lanczos_propagate (ndarray + ndarray-linalg)
│   ├── cfm4.rs                 # CFM4:2 / M2 / Richardson 推定子
│   └── blas.rs                 # 内積 / axpy / nrm2 / scal ラッパ
├── python/kryanneal/           # Python ソース (python-source = "python")
│   ├── __init__.py             # 公開 API
│   ├── __init__.pyi            # 自動生成 stub (wheel 同梱)
│   ├── py.typed                # PEP 561 マーカ
│   ├── problem.py              # IsingProblem
│   ├── schedule.py             # Schedule
│   ├── annealer.py             # QuantumAnnealer / AnnealingSimulator
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
    ├── testing.md              # /test skill 用
    └── benchmarks.md
```

## テスト

`docs/testing.md` 参照。要点だけ:

```bash
uv run pytest                               # 全 Python テスト
uv run pytest -m "not slow"                 # slow を除外
uv run pytest tests/test_krylov.py          # 個別ファイル
cd src && cargo test                        # Rust 単体 (BLAS feature ON)
cd src && cargo test --no-default-features  # scalar fallback
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
