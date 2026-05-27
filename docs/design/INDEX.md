# maqina: 設計書 (v0.13) — INDEX

横磁場イジングモデル (TFIM) の量子ダイナミクスを matrix-free に計算する
Python パッケージ。Krylov 法 (Lanczos) で matvec を介した短時間プロパゲータ
を近似し、Magnus 展開 (commutator-free Magnus, CFM4:2) で時間依存
Hamiltonian の時間発展演算子を近似する。adaptive step は CFM4:2 の
step-doubling Richardson 推定子を PI 制御に流す方式 (詳細 [§5.3](05-3-propagator.md))。

ユーザー向けインターフェースは Python、行列演算のホットループ
(Lanczos 反復、Magnus 段階指数積、対角 + bit-flip matvec) は Rust 拡張
(`pyo3` 経由) として実装する。状態 ψ ∈ ℂ^{2^N} は NumPy 配列として
保持し、2^N × 2^N の dense / sparse 行列を一度も組み立てない。

設計の参照プロジェクト (開発者向け): [`cv-ising-solver`](https://github.com/Shu-Tanaka-Group/cv-ising-solver)
は同じ Krylov + CFM4:2 カーネルの連続変数版で, 本パッケージのカーネル設計・
レイアウト・bench 慣習の流用元。

本ファイルは `docs/design/` 配下の分割された設計書ファイル群の **検索インデックス**。
かつての単一ファイル `docs/design.md` (v0.5 時点 2359 行) を issue #83 で章単位に
分割した (内容は変更なし)。`§N.M` 番号は文中・コード内 docstring の参照と
そのまま対応する。

---

## 全体目次

| § | ファイル | 概要 |
|---|---|---|
| §1 | [01-goals.md](01-goals.md) | ゴールと非ゴール。本パッケージのスコープ (TFIM 量子ダイナミクス, matrix-free) と扱わない範囲 (open system 等) |
| §2 | [02-physics.md](02-physics.md) | 物理モデル。全 Hamiltonian (`A·H_driver + B·H_problem`)、ユーザー入力 (`H_p_diag`, `h_x`)、bit 規約、初期状態の契約 |
| §3 | [03-architecture.md](03-architecture.md) | アーキテクチャ。全体構成、Python/Rust の役割分担、`maturin` 標準レイアウト |
| §4 | [04-python-api.md](04-python-api.md) | 公開 Python API。`IsingProblem` / `Schedule` / `QuantumAnnealer` / `AnnealingSimulator` / `Observable` / 例外型 |
| §5.1 | [05-1-matvec.md](05-1-matvec.md) | 数値カーネル — matvec / per-axis primitives。`apply_h`, `apply_single_mode_axis_i`, Phase 6 C3 (gate fusion), Phase 6 D アーカイブ |
| §5.2 | [05-2-lanczos.md](05-2-lanczos.md) | 数値カーネル — Lanczos `lanczos_propagate` (短時間プロパゲータ) |
| §5.3 | [05-3-propagator.md](05-3-propagator.md) | 数値カーネル — プロパゲータ。M2, CFM4:2, Trotter, Richardson, PI controller, adaptive driver DX |
| §5.4 | [05-4-python-reference.md](05-4-python-reference.md) | 数値カーネル — Python リファレンス実装 (Rust 経路との等価性検証用) |
| §6 | [06-builders.md](06-builders.md) | k-local ペア表現 → 対角ベクトル `H_p_diag` のビルダー |
| §7 | [07-rust-extension.md](07-rust-extension.md) | Rust 拡張。crate 構成、BLAS 経路、Cargo features、maturin レイアウトの歴史的注意点 |
| §8 | [08-qutip-comparison.md](08-qutip-comparison.md) | QuTiP `sesolve` 比較。位置付け、リファレンス実装、比較項目 |
| §9 | [09-testing.md](09-testing.md) | テスト戦略。Rust `cargo test`, Python `pytest`, BLAS feature on/off 等価性ペア |
| §10 | [10-benchmarks.md](10-benchmarks.md) | ベンチマーク戦略と期待される性能特性 |
| §11 | [11-build-infrastructure.md](11-build-infrastructure.md) | 開発・ビルド基盤の入口 (詳細は `docs/conventions.md`) |
| §12 | [12-release-plan.md](12-release-plan.md) | 段階リリース計画 Phase 1–8 + A/B/C (v0.1 → v0.13 完了までの DoD) |
| §13 | [13-future-work.md](13-future-work.md) | Future work (v0.11 までは対応しない範囲) |
| §14 | [14-references.md](14-references.md) | 参考文献 / 参照プロジェクト |

注: 旧 §5 (数値カーネル) は §5.1–§5.4 の 4 ファイルに分割した。コード内
docstring 等で「§5」と書かれているものは §5.1–§5.4 全体を指す。

---

## §N.M ↔ ファイル mapping (詳細)

サブセクション粒度の参照を引きたいとき用。

| § | 見出し | ファイル |
|---|---|---|
| §1 ゴール / 非ゴール | ゴール, 非ゴール | [01-goals.md](01-goals.md) |
| §2.1 | 全 Hamiltonian | [02-physics.md](02-physics.md) |
| §2.2 | ユーザー入力の表現 (bit 規約) | [02-physics.md](02-physics.md) |
| §2.3 | 初期状態 | [02-physics.md](02-physics.md) |
| §3.1 | 全体構成 | [03-architecture.md](03-architecture.md) |
| §3.2 | 役割分担 (matvec を Rust に置く理由) | [03-architecture.md](03-architecture.md) |
| §3.3 | ディレクトリレイアウト (maturin 標準形採用根拠) | [03-architecture.md](03-architecture.md) |
| §4.1 | 公開シンボル | [04-python-api.md](04-python-api.md) |
| §4.2 | `IsingProblem` | [04-python-api.md](04-python-api.md) |
| §4.3 | `Schedule` | [04-python-api.md](04-python-api.md) |
| §4.4 | `QuantumAnnealer` (`run()` ファサード) | [04-python-api.md](04-python-api.md) |
| §4.5 | `AnnealingSimulator` (step-wise) | [04-python-api.md](04-python-api.md) |
| §4.6 | `Observable` | [04-python-api.md](04-python-api.md) |
| §4.7 | 瞬時固有状態への投影 | [04-python-api.md](04-python-api.md) |
| §4.8 | 例外型ポリシー | [04-python-api.md](04-python-api.md) |
| §5.1.1 | `apply_h` (Phase 1) | [05-1-matvec.md](05-1-matvec.md) |
| §5.1.2 | `apply_single_mode_axis_i` (Phase 2) | [05-1-matvec.md](05-1-matvec.md) |
| §5.1.3 | Phase 6 C3 — gate fusion + phase_p 並列化 | [05-1-matvec.md](05-1-matvec.md) |
| §5.1.4 | Phase 6 D 実験アーカイブ (issue #79, 未採用) | [05-1-matvec.md](05-1-matvec.md) |
| §5.2 | Lanczos `lanczos_propagate` | [05-2-lanczos.md](05-2-lanczos.md) |
| §5.3 M2 | 中点則 1 step `m2_midpoint_step` | [05-3-propagator.md](05-3-propagator.md) |
| §5.3 CFM4:2 | `cfm4_step`, 係数出典 (`a_high = 1/4 + √3/6` 等) | [05-3-propagator.md](05-3-propagator.md) |
| §5.3 Richardson | `cfm4_step_with_richardson_estimate` | [05-3-propagator.md](05-3-propagator.md) |
| §5.3 Trotter | `trotter_step` (Strang 2 次 / Suzuki 4 次) | [05-3-propagator.md](05-3-propagator.md) |
| §5.3 PI controller | adaptive driver, PI 既定値, T 依存 auto resolution | [05-3-propagator.md](05-3-propagator.md) |
| §5.3 adaptive DX | issue #43 A/B/C/E follow-up (`dt_init=None` 等) | [05-3-propagator.md](05-3-propagator.md) |
| §5.4 | Python リファレンス実装 | [05-4-python-reference.md](05-4-python-reference.md) |
| §6 | ビルダー (k-local → 対角ベクトル) | [06-builders.md](06-builders.md) |
| §7.1 | Crate 構成 (tridiag.rs hand-rolled implicit QL 含む) | [07-rust-extension.md](07-rust-extension.md) |
| §7.2 | BLAS 経由のホットパス | [07-rust-extension.md](07-rust-extension.md) |
| §7.3 | `apply_h` の Python 公開 | [07-rust-extension.md](07-rust-extension.md) |
| §7.4 | Cargo features (blas / rayon / simd) | [07-rust-extension.md](07-rust-extension.md) |
| §7.5 | `__has_blas__` warning | [07-rust-extension.md](07-rust-extension.md) |
| §7.6 | maturin レイアウト上の注意点 (PyO3 stub 歴史的問題) | [07-rust-extension.md](07-rust-extension.md) |
| §8.1 | QuTiP 比較の位置付け | [08-qutip-comparison.md](08-qutip-comparison.md) |
| §8.2 | リファレンス実装 `reference_qutip.py` | [08-qutip-comparison.md](08-qutip-comparison.md) |
| §8.3 | 比較項目 | [08-qutip-comparison.md](08-qutip-comparison.md) |
| §9.1 | Rust 単体テスト (`cargo test`) | [09-testing.md](09-testing.md) |
| §9.2 | Python 統合テスト (`uv run pytest`) | [09-testing.md](09-testing.md) |
| §9.3 | 等価性ペアの規約 (BLAS feature on/off, Rust ↔ Python) | [09-testing.md](09-testing.md) |
| §10.1 | 期待される性能特性 (推定) | [10-benchmarks.md](10-benchmarks.md) |
| §12 Phase 1 | MVP / scalar baseline (~v0.1) | [12-release-plan.md](12-release-plan.md) |
| §12 Phase 2 | Trotter 経路 (~v0.2) | [12-release-plan.md](12-release-plan.md) |
| §12 Phase 3 | CFM4:2 (~v0.3) | [12-release-plan.md](12-release-plan.md) |
| §12 Phase 4 | Adaptive (~v0.4) | [12-release-plan.md](12-release-plan.md) |
| §12 Phase 5 | Simulator & Observables (~v0.5) | [12-release-plan.md](12-release-plan.md) |
| §12 Phase 6 | 並列化 + 仕上げ (~v0.6) | [12-release-plan.md](12-release-plan.md) |
| §12 Phase 7 | Lanczos β_m exposure + Richardson 誤差源分離 (~v0.7) | [12-release-plan.md](12-release-plan.md) |
| §12 Phase 8 | Lanczos a posteriori 早期打切 (~v0.8) | [12-release-plan.md](12-release-plan.md) |
| §12 Phase A | Chebyshev propagator POC (時間独立 H, issue #120) | [12-release-plan.md](12-release-plan.md) |
| §12 Phase B | Chebyshev propagator を CFM4 adaptive Richardson 経路に統合 (~v0.10, issue #122) | [12-release-plan.md](12-release-plan.md) |
| §12 Phase B follow-up | Default method を Chebyshev variant に切替 + atol 仕様明文化 (~v0.11, issue #124) | [12-release-plan.md](12-release-plan.md) |
| §12 Phase C | per-site/per-axis 時間依存場 (XYZ driver) 拡張 + Schedule に h_x 統合 (~v0.13, issue #142) | [12-release-plan.md](12-release-plan.md) |

---

## 横断トピック早見表

複数ファイルにまたがるトピックの逆引き。

| トピック | 主参照 | 関連 |
|---|---|---|
| **BLAS feature on/off** (`cfg(feature = "blas")`, Accelerate / OpenBLAS) | [§7.4](07-rust-extension.md), [§7.5](07-rust-extension.md) | [§9.3](09-testing.md) 等価性ペア, [§10](10-benchmarks.md) bench |
| **rayon 並列化** (thread pool 運用, `RAYON_NUM_THREADS`) | [§5.1.3](05-1-matvec.md) Phase 6 C1+C3 | [§7.4](07-rust-extension.md) `rayon` feature, `CLAUDE.md` Thread pool 運用節 |
| **SIMD カーネル** (`wide::f64x4`, AVX2/NEON, `target-cpu=native`) | [§5.1](05-1-matvec.md) (C2 bit-flip / C2.5 single-mode) | [§7.4](07-rust-extension.md) `simd` feature, `CLAUDE.md` SIMD 経路節, [§5.3](05-3-propagator.md) Chebyshev recurrence fused (#126) |
| **CFM4:2 係数の出典** (`a_high = 1/4 + √3/6`) | [§5.3](05-3-propagator.md) CFM4:2 節 | `cv-ising-solver/rust/src/cfm4.rs` |
| **PI controller 既定値・式** | [§5.3](05-3-propagator.md) PI controller 節 | `cv-ising/src/cv_ising/krylov.py` |
| **adaptive driver DX** (`dt_init=None`, `dt_max=None`, `m` 適応) | [§5.3](05-3-propagator.md) Phase 4 follow-up | issues #43, #54 |
| **bit 規約** (LSB=0, `σ_i = 1 - 2·b_i`) | [§2.2](02-physics.md) | `CLAUDE.md` 物理的取り決め節 |
| **maturin レイアウト** (`python-source = "python"`, stub 配置) | [§3.3](03-architecture.md), [§7.6](07-rust-extension.md) | PyO3 issues #490 / #771 / #885 |
| **`apply_h` の Phase 6 進化** | [§5.1.1](05-1-matvec.md) → [§5.1.3](05-1-matvec.md) → [§5.1.4](05-1-matvec.md) | issues #62, #63, #64, #79 |
| **Phase 6 D 失敗例アーカイブ** (DRAM 律速仮説の否定, perf 計測) | [§5.1.4](05-1-matvec.md) | `src/bin/perf_apply_h.rs`, issue #79, `CLAUDE.md` perf binary 節 |
| **Chebyshev propagator** (時間独立 / CFM4 統合 / アルゴリズム軸の cache stall 回避 / inner loop SIMD + fusion / default 切替 + atol upper bound) | [§5.3](05-3-propagator.md) "CFM4:2 + Chebyshev variant" 節 + "Chebyshev recurrence の SIMD + fusion" 節 + "`chebyshev_tol` と `atol` の関係" 節 | `src/chebyshev.rs`, `src/bin/perf_chebyshev.rs`, `src/bin/perf_cfm4_richardson_chebyshev.rs`, issues #120 #122 #124 #126 #127, `CLAUDE.md` Phase B 節 + Phase B follow-up |
| **XYZ driver / per-site/per-axis 時間依存場** (Phase C, #142) | [§2](02-physics.md) §2.1.2 / [§5.1](05-1-matvec.md) §5.1.1.y `apply_h_general` / [§5.3](05-3-propagator.md) per-stage Gershgorin + gz_eff_diag doubling 節 / [§12](12-release-plan.md) Phase C | `Schedule.from_xyz`, `IsingProblem(n, H_p_diag)`, PR #143 / #144 / #145, `CLAUDE.md` 物理的取り決め節 |
| **save_tlist セマンティクス** (最節約モード / cv-ising merge) | [§4.4](04-python-api.md), [§4.5](04-python-api.md) | memory `save_tlist_no_recording_mode`, `save_tlist_cv_ising_merge` |
| **Phase ロードマップ全体** | [§12](12-release-plan.md) | `docs/conventions.md` §2 バージョニングポリシー |
| **`tridiag.rs` hand-rolled implicit QL** (LAPACK 非依存の根拠) | [§7.1](07-rust-extension.md) | `Cargo.toml` の依存最小化方針 |

---

## ファイル分割の方針 (issue #83)

- 1 ファイル ~200 行を目安。ただし §4 Python API (~426 行), §5.1 matvec (~564 行),
  §5.3 propagator (~439 行), §12 release plan (~222 行) は **内容の凝集性が高い** ため
  単一ファイルに維持 (§5.1 内部の §5.1.1–§5.1.4 は読み筋が連続している, etc.)。
- ファイル名は `NN-slug.md` / `NN-M-slug.md` 形式でセクション番号を保存し、文中の
  `§N.M` 参照とファイルの対応を自明にする。
- 元 `docs/design.md` (2359 行) のヘッダーは
  H1=1 + H2=14 + H3=39 + H4=11 + H5=15 + H6=1 = 81 個。分割後ファイル群の
  ヘッダー総数は 81 個 (旧 H1 「設計書 (v0.5)」を本 INDEX.md 冒頭に, 旧 H2
  「## 5. 数値カーネル」を本 INDEX.md の目次に再配置), 17 個の旧 H2/H3
  (§1, §2, §3, §4, §5.1, §5.2, §5.3, §5.4, §6, §7, §8, §9, §10, §11, §12, §13, §14)
  が分割後の各ファイル冒頭 H1 に昇格。本文 H3/H4/H5/H6 は原文のまま (内容と
  ナンバリングを保存)。

## 内容の更新

- 本 issue (#83) は **物理的分割のみ**。誤記訂正・古い記述更新は別 issue で行う。
- 新規の節を追加する際は、適切な既存ファイルに足すか、新規ファイル
  `NN-M-<slug>.md` を作って本 INDEX.md の目次と mapping table に追記する。
