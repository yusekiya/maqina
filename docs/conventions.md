# kryanneal: 開発規約

設計書 (`docs/design.md`) とは別に, **開発プロセス / ビルド基盤 /
バージョニング** の規約を集約する. design.md は「何を作るか (設計・
アルゴリズム・公開 API)」, 本ドキュメントは「どう運用するか (ツール
チェイン・リリース)」を担当する.

---

## 1. 開発・ビルド基盤

- パッケージマネージャ: `uv`、Python `>=3.13`
- ビルド: `maturin` (Rust 拡張 `kryanneal._rust` を PyO3 経由でビルド)
- Lint: `ruff` / 型: `ty`
- 主要依存: `numpy>=2.4.2`, `threadpoolctl>=3.0`
- dev 依存: `pytest>=8.3`, `qutip>=5.2.3`, `pre-commit>=4.0`, `ruff`, `ty`
- API stubs: `tools/gen_api_stubs.py` で `.py` から PEP 484 stub 自動生成。
  pre-commit hook と `.claude/rules/api-stubs-sync.md` で drift 防止する
  二段運用 (人間編集も hook が拾う)。
- BLAS 多プロセス制御: `set_blas_threads(n)` /
  `available_blas_threads()` を `__init__.py` に export
  (`threadpoolctl.threadpool_limits` を BLAS API 単位で呼び出す
  ラッパで、numpy/scipy bundled + system の OpenBLAS pool 同居問題に対処)。

---

## 2. バージョニングポリシー

`pyproject.toml` / `Cargo.toml` の `version` は **Phase 完了時に
`0.N.0` へ bump** する (Phase N → v0.N の一対一マッピング, Phase 計画は
`docs/design.md` §12 参照)。

| 時点 | `version` |
|---|---|
| 初期状態 / Phase 1 進行中 | `0.0.0` |
| Phase 1 完了 (umbrella #1 close 時) | `0.1.0` |
| Phase 2 完了 | `0.2.0` |
| ... | ... |
| Phase 6 完了 | `0.6.0` |
| Phase 6 後 | `docs/design.md` §13 Future work を再評価して `0.7.0+` のロードマップを引く |

bump 操作は **Phase の最後の child issue を解決する PR に同梱する**
(別 release commit を作る運用も可だが, ヒストリ簡潔化のため同梱を
基本とする). bump コミットに含めるファイル:

- `pyproject.toml` の `version`
- `Cargo.toml` (workspace なら `[package].version`) の `version`
- `docs/design.md` L1 の "設計書 (v0.X draft)" / "(v0.X)" 表記
  (Phase 1 完了時は "draft" を外し "v0.1" に確定)
- (任意) `CHANGELOG.md` に Phase 単位の差分を追記

破壊変更がない限り MAJOR (`1.0.0`) は v1.0 ロードマップを別途引いてから.
v0.x の範囲では SemVer の通常規約に従い MINOR (`0.N.0` → `0.N+1.0`) で
互換性のない変更を吸収可能とする (v0 段階のため).

### 2.1 umbrella issue の Definition of Done に必ず含める項目

新しい Phase N の umbrella issue を起票するときは, Definition of Done に
以下の 2 項目を **必ず含める** (本ポリシーへの参照リンクを貼る):

- `pyproject.toml` / `Cargo.toml` の `version` を `0.N.0` に bump 済み
- `docs/design.md` L1 の "(v0.N draft)" → "(v0.N)" に更新済み

これにより, Phase 完了タイミングで version bump を忘れて先に進む事故を
防ぐ. 既存の Phase 1 umbrella (#1) も本ポリシー追加時に同様に更新済み.

### 2.2 破壊的変更ログ (Phase 完了 bump 時に Changelog として参照)

`0.N.x` 進行中に蓄積した公開 API の破壊的変更 (mid-Phase で取り込まれ,
次の Phase 完了 bump で版数化される) を時系列で記録する. Phase 完了
PR の commit message / release notes 起こしに使う一次資料.

#### Phase 4 follow-up (0.4.x → 0.5.0 で版数化予定)

- **issue #54 / PR #54-fix (本変更)**: `QuantumAnnealer` の adaptive driver
  default を `None` default + auto resolution の統一スタイルに揃え, 旧
  `Literal["auto"]` リテラルを削除した. 公開 API シグネチャの破壊的変更.
  - `QuantumAnnealer.__init__(krylov_tol: float = 1e-12)` →
    `krylov_tol: float | None = None`. None で adaptive Richardson 経路は
    `effective_krylov_tol = atol · _KRYLOV_TOL_ATOL_RATIO` (既定 `1e-3`,
    atol=1e-8 で `1e-11`) に解決. 固定 dt 経路は `atol` を取らないため
    None → `_KRYLOV_TOL_FIXED_DEFAULT = 1e-12` フォールバック (旧 default
    維持).
  - `QuantumAnnealer.run(dt_init: float | Literal["auto"] | None = None)` →
    `dt_init: float | None = None`. None で `_resolve_dt_init_auto(t0, t1)`
    (旧 `"auto"` 経路と同じ T-dep formula). `"auto"` リテラル受付は廃止.
  - `QuantumAnnealer.run(dt_max: float | Literal["auto"] | None = None)` →
    `dt_max: float | None = None`. None で `_resolve_dt_max_auto(...)`
    (旧 `"auto"` 経路と同じ Gershgorin cap). `"auto"` リテラル受付は廃止.
  - **移行手順**: `dt_init="auto"` / `dt_max="auto"` を明示していた呼出は
    `dt_init=None` / `dt_max=None` (または引数省略) に書き換える (ビット
    一致で挙動維持). `dt_init`/`dt_max`を引数省略していた呼出は driver
    旧 default (`0.5` / `10·dt0`) から問題依存 auto 値に挙動が変わる
    (issue #54 の motivation: 固定保守値より問題依存値の方が筋).
    `krylov_tol=1e-12` を再現したい呼出は明示的に渡す.
  - 詳細根拠は `docs/design.md` §5.3 follow-up 節 E "adaptive driver
    default の統一".
