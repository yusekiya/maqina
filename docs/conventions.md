# maqina: 開発規約

設計書 (`docs/design/INDEX.md`) とは別に, **開発プロセス / ビルド基盤 /
バージョニング** の規約を集約する. `docs/design/` は「何を作るか (設計・
アルゴリズム・公開 API)」, 本ドキュメントは「どう運用するか (ツール
チェイン・リリース)」を担当する.

---

## 1. 開発・ビルド基盤

- パッケージマネージャ: `uv`、Python `>=3.13`
- ビルド: `maturin` (Rust 拡張 `maqina._rust` を PyO3 経由でビルド)
- Lint: `ruff` / 型: `ty`
- 主要依存: `numpy>=2.4.2`, `threadpoolctl>=3.0`
- dev 依存: `pytest>=8.3`, `qutip>=5.2.3`, `pre-commit>=4.0`, `ruff`, `ty`
- API stubs: `tools/gen_api_stubs.py` で `.py` から PEP 484 stub 自動生成。
  pre-commit hook と `.claude/rules/api-stubs-sync.md` で drift 防止する
  二段運用 (人間編集も hook が拾う)。
- BLAS 多プロセス制御: `set_blas_threads(n)` /
  `set_blas_threads_auto()` / `available_blas_threads()` を `__init__.py`
  に export (`threadpoolctl.threadpool_limits` を BLAS API 単位で呼び出す
  ラッパで、numpy/scipy bundled + system の OpenBLAS pool 同居問題に対処)。
  推奨 default は `set_blas_threads_auto()` (issue #116, EPYC 7713P perf
  実測で 1.52× speedup の sweet spot; `process_cpu_count / 8` を 1-16 で
  クランプ, env で上限指定可)。

---

## 2. バージョニングポリシー

`pyproject.toml` / `Cargo.toml` の `version` は **v0.x の範囲で SemVer の
通常規約に従う**。公開 API の追加または破壊的変更が積み上がった時点で
MINOR (`0.N.0` → `0.N+1.0`) を上げる (v0 段階のため, 互換性のない変更も
MINOR で吸収する)。MAJOR (`1.0.0`) は v1.0 ロードマップを別途引いてから.

bump の契機は 2 通り:

- **(a) Phase umbrella issue の完了**: Phase 計画は
  `docs/design/12-release-plan.md` §12 参照。Phase 1〜8 はこの契機で
  bump した (当時は Phase N → v0.N の一対一マッピングだった)。
- **(b) Phase に属さない issue バッチの区切り**: 単発 issue または関連
  issue 群がまとまった時点で bump する。`0.9.0` (issue #116) 以降は
  こちらが主流で, **Phase 番号と版数の対応は既に成立していない**。

> **個別バージョンと変更内容の対応は `CHANGELOG.md` (repo root) が唯一の
> 情報源**。本ファイルには「どう bump するか」の手順のみを置き, 時点と
> 版数の対応表は持たない (二重管理によるドリフトを避けるため)。

bump 操作は **そのバッチの最後の issue を解決する PR に同梱する**
(別 release commit を作る運用も可だが, ヒストリ簡潔化のため同梱を
基本とする). bump コミットに含めるファイル:

- `pyproject.toml` の `version`
- `Cargo.toml` (workspace なら `[package].version`) の `version`
- `docs/design/INDEX.md` L1 の "設計書 (v0.X)" 表記
- `CHANGELOG.md` (repo root) の "Unreleased" セクションを
  `## 0.N.0 - YYYY-MM-DD — <要約>` の released セクションに繰り上げる
  (詳細は §2.2)

### 2.1 umbrella issue の Definition of Done に必ず含める項目

新しい Phase の umbrella issue を起票するときは, Definition of Done に
以下の 2 項目を **必ず含める** (本ポリシーへの参照リンクを貼る):

- `pyproject.toml` / `Cargo.toml` の `version` を次の MINOR に bump 済み
- `docs/design/INDEX.md` L1 の版数表記を同じ MINOR に更新済み

これにより, バッチ完了タイミングで version bump を忘れて先に進む事故を
防ぐ. 上記の契機 (b) にあたる非 Phase バッチでも, 最後の issue を解決
する PR で同じ 2 項目を確認する.

### 2.2 破壊的変更ログ集約先: `CHANGELOG.md`

bump 前に蓄積した公開 API の破壊的変更 (次の MINOR bump でまとめて
版数化される) とリリース単位の差分は **すべて
`CHANGELOG.md` (repo root) に時系列で記録する**. 本ファイル
(`docs/conventions.md`) には個別の変更内容を書かず, ポリシーのみを
記載する.

運用ルール:

- bump 前に公開 API の追加・破壊的変更を取り込む PR は, **同 PR の中で**
  `CHANGELOG.md` の "Unreleased" セクションにエントリ (issue / PR 番号,
  シグネチャ変更, 移行手順, 根拠の参照先) を追記する. PR 単独で release notes 起こしに使える粒度で記述する.
- bump PR では `CHANGELOG.md` の "Unreleased" セクションを
  released バージョン (`## 0.N.0 - YYYY-MM-DD` 等) に繰り上げ, 必要に
  応じて Added / Changed / Fixed セクションも整理する.
- フォーマットは [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
  に概ね準拠 (v0.x 範囲では SemVer の破壊的変更を MINOR で吸収する慣習に
  合わせ, Breaking セクションも MINOR 配下に置く).
- 個別の変更の **詳細根拠** (アルゴリズム選定, 経験則の数値, follow-up
  scope 議論) は `docs/design/INDEX.md` 該当節に書き, `CHANGELOG.md` からは
  リンクで参照する (CHANGELOG が長文化するのを避ける).

### 2.3 リリース bench artifact: `benchmarks/results/<X.Y.Z>/`

bump 時に本番 bench sweep を取った場合, その結果は **`benchmarks/results/<X.Y.Z>/`**
ディレクトリにコミットして履歴を残す (例 `benchmarks/results/0.8.0/`).
従来は `benchmarks/results/<timestamped-dir>/` 配下に gitignore で残すだけ
だったが, リリース時の bench は累積改善の単一情報源として後から参照する
価値が高いため version 単位で正式に track する.

**ディレクトリ命名**:

- semver `X.Y.Z` (例 `0.8.0`) **そのまま**, prefix なし (`v0.8.0` の形は
  使わない).
- 開発中の ad-hoc bench は従来通り `benchmarks/results/<YYYYMMDD-HHMMSS>/`
  (gitignore) に出力する. version dir と timestamped dir は混在 OK.

**コミット対象**:

- **必置**: `SUMMARY.md` — 当該 version の bench 全体の解釈 / 累積改善の
  ハイライト / acceptance 判定をまとめた 1 ファイル. 数値だけだと将来読み
  返したときに文脈を失うため必ず添える.
- **生 bench の markdown**: 当該 finalize で取った `bench_*.md` を版数化と
  同 commit で配置. ファイル名はスクリプトの default 出力名そのまま
  (`bench_per_step.md` / `bench_parallel_scaling.md` /
  `bench_block_fusion.md` / `bench_qutip_large.md` 等).
- **生 CSV は除外**: `.csv` は引き続き gitignore (人が読まない生データで
  diff の意味が薄く size が大きいため). 必要なら bench 実行マシン上の
  timestamped dir に残す.

**`.gitignore` の現運用**: `/benchmarks/results/*` で root 直下を ignore
した上で `!/benchmarks/results/*/` + `!/benchmarks/results/*/*.md` で
version dir 配下の markdown のみ except する. CSV は default で
ignore のまま (`benchmarks/results/0.8.0/*.csv` も track されない).

**リリース finalize PR フロー** (#66 / 0.8.0 で確立):

1. finalize bench を Linux サーバーで実行
   (`benchmarks/results/<YYYYMMDD-HHMMSS>/` に出力, gitignore).
2. Claude が結果を分析し SUMMARY.md を起草. 4 bench `.md` と合わせて
   `benchmarks/results/<X.Y.Z>/` に配置 (CSV は ignore のため除外).
3. bump PR にコミット同梱して push.
4. PR コメントには `SUMMARY.md` を `gh pr comment --body-file` で添付
   (umbrella issue にも re-post).
5. merge 後は `benchmarks/results/<X.Y.Z>/` が永続記録として GitHub 上に
   残る (初版 → 当該 version の累積改善の参照可能アーカイブ).

**遡及はしない**: 過去リリースの bench は当該 child PR コメントで個別に
記録済みのため `benchmarks/results/0.1.0/` 〜 `0.7.0/` を遡って作らない.
本ポリシーは 0.8.0 以降の forward-looking 運用. **bench sweep 自体は毎回
必須ではない**: 性能に影響しないリリースでは省略する (実績では 0.8.0 /
0.12.0 / 0.14.0 のみ artifact を持つ).
