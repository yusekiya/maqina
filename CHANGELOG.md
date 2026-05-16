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

## Unreleased — Phase 6 in progress (0.5.x → 0.6.0 で版数化予定)

### Added

- **issue #68 / PR (TBD)**: Phase 6 C1 follow-up. 本番 bench (issue #62 sweep)
  で表面化した 2 つの bias / regression を修正:
  - `src/matvec.rs`: **dim 閾値 dispatch** (`MIN_RAYON_DIM = 1 << 17`) を
    public `apply_h_kryanneal` / `apply_single_mode_axis_i` に追加。dim <
    128K (= N ≤ 16) では rayon barrier overhead が単スレッド時間を超えて
    regression (N=16 / 2 threads = 0.57× 実測) するため scalar 経路に
    フォールバック。private `*_rayon` 関数は dispatch を含まず常に rayon
    実行のままで, rayon-path 自体のテストは `*_rayon` 直接呼出で継続。
  - `benchmarks/bench_parallel_scaling.py`:
    - **`trotter_step` cell を追加** (`_rust.trotter_step_py` 1 call =
      本番ホットパスの per-step cost). 既存 `apply_single_mode_axis_i_sum`
      は `apply_single_mode_axis_i_py_sum_diagnostic` にリネームし
      diagnostic 用途として残す (Python wrap の to_vec allocation overhead
      検出用).
    - **knee detection を max-speedup baseline + 95% plateau** に置換
      (旧 "前点比 < 5%" はノイズや regression を knee と誤判定するため,
      issue #62 の N=16 / 2 threads セルで顕在化). md 表に `max_speedup`
      / `max @ threads` 列を追加.

- **issue #62 / PR #67**: `src/matvec.rs` の bit-flip pass primitive を
  rayon `par_chunks_mut` 経由で L2 並列化 (Phase 6 C1).
  - `apply_h_kryanneal`: `y` を `(dim / (nth·4))` を目安に chunk 分割し,
    各 chunk closure 内で diag pass + 全 i bit-flip pass を **fuse**
    (cache-blocked 形). `y_chunk` を L1 cache resident に保ち, 後段 SIMD
    (C2) / cache block-fusion (C3) の足場とする.
  - `apply_single_mode_axis_i`: `block = 2·mask` 単位で `par_chunks_mut`
    並列化. 退化ケース `i = n-1` (block == dim) では `psi.split_at_mut(mask)`
    + `par_iter_mut().zip(par_iter_mut())` のペア並列にフォールバック.
  - Cargo: `rayon = "1"` optional dep + `[features] rayon` (**default ON**,
    BLAS と同じ on/off pattern). `--no-default-features` でビルドすると
    scalar 単スレッド経路に戻り従来挙動を維持する.
  - thread 数制御: 環境変数 `RAYON_NUM_THREADS` (rayon の global pool は
    プロセス起動時に決まる). 既存 `kryanneal.set_blas_threads(n)` と
    併用するときは BLAS pool を 1 thread に落とすことを推奨
    (`CLAUDE.md` 「Thread pool 運用」節).
  - `benchmarks/bench_parallel_scaling.py` を新規追加. subprocess で
    `RAYON_NUM_THREADS` を変えながら N × thread sweep し,
    `(median wall_sec, speedup vs threads=1, memory-bandwidth knee)` を
    `benchmarks/results/<ts>/bench_parallel_scaling.{csv,md}` に出力.
  - 数値: rayon あり/なし両ビルドで `y` / `psi` が **bit-identical**
    (`apply_*_rayon_matches_serial` テストで `to_bits()` 一致を検証).
    8 thread × 100 反復の race-detection fuzz test も追加.

### Unreleased internal note

CHANGELOG: Phase 5 finalize 時 (commit `49dd673`) に旧 `Unreleased — Phase 4
follow-up` セクションを `## 0.5.0` に繰り上げ忘れていたため, Phase 6 C1
の docs 更新と合わせて遡及的に促進した. 内容変更は無し (header の rename
のみ).

## 0.5.0 - 2026-05-16

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
