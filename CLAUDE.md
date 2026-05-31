# CLAUDE.md

Claude Code 向けのプロジェクトガイド。

## 要望

指示がない限りユーザーへの回答や質問は日本語で行うこと。

## プロジェクト概要

`maqina` (a (Ma)gnus-based (Q)uantum (I)sing (N)umerical (A)nnealer; 名前はラテン語 `machina` に由来): 横磁場
イジングモデル (TFIM) の量子ダイナミクスを matrix-free に計算する
シミュレータ。Krylov 法 (Lanczos) または Chebyshev 多項式展開で短時間
プロパゲータを近似し、Magnus 展開 (CFM4:2) で時間依存 Hamiltonian の
時間発展演算子を近似する。adaptive dt ドライバ (step-doubling
Richardson + PI 制御) も提供。

設計の参照プロジェクト: [`cv-ising-solver`](https://github.com/Shu-Tanaka-Group/cv-ising-solver)
(同じ Krylov + CFM4:2 カーネルの連続変数版)。

- パッケージマネージャ: `uv` (Python `>=3.13`)
- ビルドバックエンド: `maturin` (Rust 拡張 `maqina._rust` を PyO3 経由でビルド)
- Lint: `ruff`
- 型チェック: `ty`
- 主要依存: `numpy`, `threadpoolctl`
- dev 依存: `pytest`, `qutip` (参照実装比較用), `pre-commit`, `ruff`, `ty`
- Rust: `pyo3 0.28`, `numpy 0.28`, `ndarray 0.16`, `num-complex 0.4`,
  `cblas 0.5` (optional)。LAPACK 非依存 (三重対角固有分解は
  `src/tridiag.rs` に hand-rolled 実装、§7.1 参照)

## 詳細ルール (`.claude/rules/`)

分量が増えたため, トピックごとの詳細ルールは `.claude/rules/` 配下に分割
した。`paths` frontmatter 付きの path-scoped rule で, 該当ファイルを編集
するときに自動でロードされる (`/memory` でロード状況を確認できる)。CLAUDE.md
本体には常時必要な核 (物理契約 / コーディング規約 / レイアウト / 設計書
ポインタ) のみ残す。

| rule ファイル | 内容 | ロード対象 (`paths`) |
|---|---|---|
| `testing.md` | テスト実行 (test-runner subagent 経路 / BLAS on-off 一致検証) | `tests/**`, `src/**/*.rs`, `python/maqina/**` |
| `api-reference.md` | `.pyi` stub 運用 (公開 API リファレンス / ドリフト防止) | `python/maqina/**`, `tools/gen_api_stubs.py`, `tests/**`, `benchmarks/**` |
| `benchmarks.md` | ベンチマーク CLI / 性能主張の作法 (cv_ising 流) | `benchmarks/**`, `scripts/**/*.sh` |
| `thread-pool.md` | rayon × BLAS thread pool 運用 / 推奨 default | `src/**/*.rs`, `python/maqina/**`, `benchmarks/**` |
| `simd.md` | SIMD 経路 (Phase 6 C2 / C2.5) / build profile 確認フラグ | `src/**/*.rs` |
| `perf-binaries.md` | `src/bin/` perf 計測 binary / `perf stat` 計測例 | `src/bin/**/*.rs`, `src/**/*.rs` |
| `phase-history.md` | Phase 6 D 〜 Phase B follow-up の開発履歴 / 設計判断アーカイブ | `src/**/*.rs`, `python/maqina/**` |
| `api-stubs-sync.md` | `.pyi` 再生成を同一コミットに含める運用 (既存) | `python/maqina/**/*.py`, `tools/gen_api_stubs.py` |

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
maqina/
├── pyproject.toml
├── Cargo.toml                  # Rust crate ルート (maturin 標準位置)
├── src/                        # Rust ソース
│   ├── lib.rs                  # PyO3 #[pymodule] fn _rust エントリポイント
│   ├── matvec.rs               # apply_h_kinema (bit-flip + diag)
│   ├── krylov.rs               # lanczos_propagate (ndarray ベース)
│   ├── cfm4.rs                 # CFM4:2 / M2 / Richardson 推定子
│   ├── tridiag.rs              # 実対称三重対角の implicit QL (hand-rolled)
│   └── blas.rs                 # 内積 / axpy / nrm2 / scal ラッパ
├── python/maqina/           # Python ソース (python-source = "python")
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

- **Hamiltonian 形 (旧 API, X-only TFIM)**:
  `H(t) = A(s(t)) · H_driver + B(s(t)) · H_problem`
  - `H_driver = -Σ_i h_x_i X_i` (サイト依存横磁場, 静的振幅)
  - `H_problem` は Z 演算子のみで書かれた k-local 多項式 → **Z 基底で対角**
- **Hamiltonian 形 (新 API, per-site/per-axis 時間依存場, issue #142 Phase C)**:
  `H(t) = Σ_i [g_x_i(t)·X_i + g_y_i(t)·Y_i + g_z_i(t)·Z_i] + b(t)·H_p_diag`
  - per-site, per-axis に独立な時間依存係数 (g_y / g_z は `None` で skip 可)
  - `Schedule.from_xyz(T, g_x, b, *, g_y=None, g_z=None)` で構築
  - `method="trotter"` / `"trotter_suzuki4"` は実数係数前提 SIMD のため新 API
    では `ValueError` (out of scope, 必要時に別 issue)
- **責任分担 (issue #142 Phase C)**: `Schedule` が時間発展係数 (h_x 振幅含む)
  を一手に保持し, `IsingProblem` は問題側の静的構造 (`H_p_diag`) のみ保持する
  pure data container に再整理。旧 API でも `h_x` は `Schedule(T, A, B, h_x)`
  に渡す (IsingProblem に `h_x` 引数は無い)
- **ユーザー入力**: `IsingProblem(n, H_p_diag: (2^N,) float64)` および
  `Schedule(..., h_x: (N,) float64)` (旧 API) または `Schedule.from_xyz(...)`
  (新 API)。対角ベクトル自体を渡してもらう (k-local 表現はパッケージ側で扱わない)
- **ビット規約**: bit 0 = LSB、`x = Σ_i b_i · 2^i`、spin `σ_i = 1 - 2·b_i`
- **初期状態**: ユーザーが必ず明示指定 (default なし)。L2-normalize 済みを
  コンストラクタで検証 (`‖psi0‖ - 1 < 1e-10`)
- **時間発展**: 純粋状態 Schrödinger 方程式のみ。Open system は非対応

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
