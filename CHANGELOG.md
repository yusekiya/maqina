# Changelog

`kryanneal` の公開 API 破壊的変更と Phase 単位の差分を集約する.

- 運用ポリシー: `docs/conventions.md` §2 (バージョニング) / §2.2 を一次
  資料とする. **mid-Phase で取り込まれた破壊的変更も本ファイルに時系列
  で記録** し, 次の Phase 完了 bump 時に release notes / commit message
  起こしの一次資料として参照する.
- フォーマット: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
  と SemVer 0.x.y の慣習に概ね準拠. ただし v0 段階のため MINOR
  (`0.N.0` → `0.N+1.0`) で破壊的変更を吸収する (`docs/conventions.md`
  §2 参照).

## Unreleased — Phase 4 follow-up (0.4.x → 0.5.0 で版数化予定)

### Breaking

- **issue #54 / PR #55**: `QuantumAnnealer` の adaptive driver default を
  `None` default + auto resolution の統一スタイルに揃え, 旧
  `Literal["auto"]` リテラルを facade から完全削除. 公開 API シグネチャ
  の破壊的変更.
  - `QuantumAnnealer.__init__(krylov_tol: float = 1e-12)` →
    `krylov_tol: float | None = None`. None で adaptive Richardson 経路は
    `effective_krylov_tol = atol · _KRYLOV_TOL_ATOL_RATIO` (既定 `1e-3`,
    atol=1e-8 で `1e-11`) に解決. 固定 dt 経路 (`m2` / `cfm4`) は `atol`
    を取らないため None → `_KRYLOV_TOL_FIXED_DEFAULT = 1e-12` static
    fallback (旧 default 維持).
  - `QuantumAnnealer.run(dt_init: float | Literal["auto"] | None = None)` →
    `dt_init: float | None = None`. None で `_resolve_dt_init_auto(t0, t1)`
    (旧 `"auto"` 経路と同じ T-dep formula). `"auto"` リテラル受付は廃止.
  - `QuantumAnnealer.run(dt_max: float | Literal["auto"] | None = None)` →
    `dt_max: float | None = None`. None で `_resolve_dt_max_auto(...)`
    (旧 `"auto"` 経路と同じ Gershgorin cap). `"auto"` リテラル受付は廃止.
  - **移行手順**:
    - `dt_init="auto"` / `dt_max="auto"` を明示していた呼出は
      `dt_init=None` / `dt_max=None` (または引数省略) に書き換える
      (ビット一致で挙動維持).
    - `dt_init` / `dt_max` を引数省略していた呼出は driver 旧 default
      (`0.5` / `10·dt0`) から問題依存 auto 値に挙動が変わる (issue #54
      の motivation: 固定保守値より問題依存値の方が筋).
    - `krylov_tol=1e-12` を再現したい呼出は明示的に渡す.
  - 詳細根拠は `docs/design.md` §5.3 follow-up 節 E "adaptive driver
    default の統一".
