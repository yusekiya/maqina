# Changelog

`kinema` の公開 API 破壊的変更と Phase 単位の差分を集約する.

- 運用ポリシー: `docs/conventions.md` §2 (バージョニング) / §2.2 を一次
  資料とする. **mid-Phase で取り込まれた破壊的変更も本ファイルに時系列
  で記録** し, 次の Phase 完了 bump 時に release notes / commit message
  起こしの一次資料として参照する.
- フォーマット: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
  と SemVer 0.x.y の慣習に概ね準拠. ただし v0 段階のため MINOR
  (`0.N.0` → `0.N+1.0`) で破壊的変更を吸収する (`docs/conventions.md`
  §2 参照).

## 0.11.0 - 2026-05-22 — Default method を Chebyshev variant に切替 + atol 仕様明文化 (Phase B follow-up, issue #124)

Phase B 本体 (#122) と #126 / #127 の perf 結果 (Linux AMD EPYC 7713P, N=18
で Lanczos 比 5.49× wall 高速, branch-miss 158× 減, sys time 78× 減,
parallel efficiency 27% → 44%) を受けて, 判断系 follow-up を確定. default
method の semantic 変更を伴うため minor bump.

### Breaking

- **`QuantumAnnealer.run(method=...)` の default**: `"m2"` →
  `"cfm4_adaptive_richardson_chebyshev"`. 旧 default を使っていたユーザーは
  `method="m2"` を明示するか, 新 default 経路に切替えて `n_steps` の代わりに
  `atol` で精度を制御する.
- **`QuantumAnnealer.create_simulator(method=...)` の default**: `"cfm4"` →
  `"cfm4_adaptive_richardson_chebyshev"`. ついでに `Literal` から欠落していた
  `_chebyshev` を追加 (Phase B #122 取りこぼし fixup).
- **`AnnealingSimulator(method=...)` の default**: `"cfm4"` →
  `"cfm4_adaptive_richardson_chebyshev"`.

`_krylov` literal は永続的に残す (旧 default 互換 + 比較ベンチ用途).

### Changed

- **`QuantumAnnealer.run` / `AnnealingSimulator.__init__` の `atol` docstring**:
  "Note (Chebyshev variant の atol 振舞い, issue #124)" 注を追加. Chebyshev では
  `atol` を upper bound として扱い, K_used 動的拡張により実際の精度がそれより
  良くなる場合があることを明文化 ("feature" 仕様, Scope 2 (a) + (d) 確定).
- **`docs/design/05-3-propagator.md` "Chebyshev variant" 節**: "`chebyshev_tol`
  と `atol` の関係 — accidental 高精度 (issue #124)" 小節を追加.
- **`docs/quickstart.md` の主例**: `method=` 指定を削除して default を使う形に
  統一. Chebyshev variant の atol upper bound 注を追記.
- **`bench_qutip_large.py --adaptive-tols` / `--krylov-tols` ヘルプ**: 両 adaptive
  経路 (`_krylov` / `_chebyshev`) に対応する文言に更新. default solver list
  (`_VALID_SOLVERS` 全列挙) は変更なし (Pareto 比較目的なので両者走らせる).
- **`docs/design/12-release-plan.md` / `docs/design/INDEX.md` / `CLAUDE.md`**:
  Phase B follow-up (#124) 節を追加.

公開 API シグネチャ自体は不変 (default 値のみ変更). 既存 test は全て
`method=` を明示しているので default 切替で壊れない.

## 0.10.0 - 2026-05-22 — Phase B (Chebyshev propagator を CFM4 adaptive Richardson 経路に統合, issue #122)

Phase A (#120, PR #121) で時間独立 H 単体の `chebyshev_propagate` 3 項漸化が
**per-call 29 ms / 4.45× Lanczos 高速** を達成したのを受け, 時間依存 H + CFM4
Magnus + step-doubling Richardson + PI controller 経路に統合した variant を
公開 API レベルで露出. Phase B 完了で Pareto win を実証
(`bench_qutip_large` n=8/10/12 で 1.19-1.28×; perf binary 直接比較 N=18 で
5.49× — `#124` perf archive).

### Breaking

- **`method` literal の hard rename**: `"cfm4_adaptive_richardson"` →
  `"cfm4_adaptive_richardson_krylov"`. alias なし (pre-1.0 なので破壊的変更
  OK, `_krylov` / `_chebyshev` で suffix 対称化のため).

### Added

- **`method="cfm4_adaptive_richardson_chebyshev"`**: Phase A の
  `chebyshev_propagate` を CFM4:2 + step-doubling Richardson + PI controller
  経路に統合した新 method. `m_max` を渡すと `ValueError` (Chebyshev は
  K_used 動的決定で Krylov 部分空間次元の概念がない).
- **Rust 側**: `src/cfm4.rs::cfm4_step_chebyshev` /
  `cfm4_step_chebyshev_with_richardson_estimate`,
  `python/kinema/krylov.py::evolve_schedule_adaptive_richardson_chebyshev`,
  `src/bin/perf_cfm4_richardson_chebyshev.rs` (perf 計測 binary).
- **`tests/test_chebyshev.py`**: QuTiP fidelity + Lanczos 一致 + annealer/simulator
  smoke + m_max ValueError.

### Performance

- bench_qutip_large (long-T scenario, EPYC 7713P): n=8 で 1.19-1.25×, n=10 で
  1.19-1.28×, n=12 で 1.09-1.17× wall 高速 (Lanczos 比). infidelity は両者とも
  `<1e-16` で精度劣化なし.

### Phase B follow-up

- **#126**: Chebyshev 3 項漸化 inner loop の SIMD + fusion (`wide::f64x4`,
  walk 2/3 を 1 dim-walk に fuse).
- **#127**: Chebyshev non-matvec inner loop の rayon 並列化
  (`chebyshev_recurrence_fused_rayon`, parallel efficiency 改善).
- **#124**: Default method 切替 + atol 仕様明文化 (本 0.11.0 で実施).

## 0.9.0 - 2026-05-22 — BLAS thread default 方針改訂 (issue #116)

EPYC 7713P perf 実測 (#113 / PR #115) で「rayon 経路では BLAS=1」という
従来推奨が 1.52× の改善余地を逃していたことが判明し, 新ヘルパ
`set_blas_threads_auto()` を導入して default policy を改訂.

### Added

- **issue #116**: `kinema.set_blas_threads_auto()` 公開. 内部で
  `_recommended_blas_threads()` を呼んで `set_blas_threads(n)` を適用 (戻り値
  は適用した n). `_recommended_blas_threads()` は
  `os.process_cpu_count() // 8` を 1-16 でクランプし, さらに
  `OPENBLAS_NUM_THREADS` / `MKL_NUM_THREADS` / `VECLIB_MAXIMUM_THREADS` /
  `OMP_NUM_THREADS` (この優先順) が set されていれば strict な上限として
  `min(auto, env_cap)` を返す. `available_blas_threads()` (現在の BLAS
  pool 状態 query) とは意図的に分離し冪等性を担保.

### Changed

- **`CLAUDE.md` "Thread pool 運用 (rayon × BLAS)" 節**: 旧推奨
  `set_blas_threads(1)` を撤回し, 新推奨 `set_blas_threads_auto()` に
  全面書き換え. 撤回理由 (PR #115 perf 実測で NT=8 で 1.52× speedup,
  NT=16-32 でも +2% 以内, spin-wait の rayon 圧迫も実害無し) を併記.
- **`docs/quickstart.md`** "並列ジョブ実行時のスレッド数制御" 節:
  `set_blas_threads_auto()` を新 default の便利関数として追加紹介.
- **`python/kinema/__init__.py::set_blas_threads` docstring**: 旧
  「rayon 経路で `set_blas_threads(1)`」例示を新方針 (`set_blas_threads_auto()`
  を default 推奨, 完全隔離が要件なら明示 `set_blas_threads(1)` または env で
  `OPENBLAS_NUM_THREADS=1`) に差し替え.

公開 API シグネチャに新 helper 追加 + 推奨 default 変更 → **minor bump
(`0.8 → 0.9`)**.

## 0.8.0 - 2026-05-18 — Phase 8 (Lanczos a posteriori 早期打切)

Phase 8 (#98) で Lanczos 早期打切判定式を `β_k · |c_last| · |dt| / (k+1) <
krylov_tol` (Hochbruck-Lubich 1997) に置き換え, `krylov_tol` を **"Krylov
近似の許容誤差"** として意味再定義. β 単体閾値は numerical breakdown safety
(`< 1e-14` で division by zero 回避) に役割を絞った. 同じ default 値
(adaptive: `atol · 1e-3`, fixed-dt: `1e-12`) を渡しても挙動が変わる
(旧: m_eff = m_max 固定, 新: m_eff ≪ m_max になる scenario が増える) ため
**minor bump (`0.7 → 0.8`)** 扱い. 公開 API シグネチャは不変.

### Breaking

- **issue #98 / PR #99**: `krylov_tol` のセマンティクス変更. 旧 β 単体閾値
  → 新 a posteriori 許容誤差 (`β · |c| · dt / m`). `python/kinema/krylov.py`
  / `src/krylov.rs` 双方の `lanczos_propagate` 内ループ判定式を書き換え.
  - 内部 c 配列を `psi_norm` 抜きで保持し終端で `ψ_new = ‖ψ‖ · V · c` に
    coeff を畳み込む形にリファクタ. これにより `c_m_abs` (Phase 7 で expose
    した `|c_m|`) も `‖ψ‖` 抜きの "pure な行列要素" (literature 標準) で返る.
  - `tridiag_c_last_abs` ヘルパ (Rust + Python) を per-iter 用に新設.
    Rust ↔ Python ref `rel < 1e-13` 一致.
  - 既存テスト全 pass (default 設定で数値精度 regression なし).
  - 新規 acceptance テスト: `test_python_lanczos_aposteriori_*`
    (termination_fires / accuracy_preserved / monotone_compression).
  - 詳細: `docs/design/05-2-lanczos.md` "a posteriori 早期打切",
    `docs/design/12-release-plan.md` Phase 8 DoD.

### Performance

- **issue #100 / PR #101**: Richardson `cfm4_step_with_richardson_estimate`
  の **iter-0 primitive matvec memoization**. full_step stage 1 と
  half_1 stage 1 は同じ入口 ψ から始まるため `H_drv · ψ` / `H_p_diag · ψ`
  primitive を入口で 1 度だけ計算して両 Lanczos call で再利用. ~3% 純減
  (cache 計算 1 合成 matvec の overhead を引いた純削減). 数値同等性
  `rel < 2e-15`. Lanczos API 不変, 既存 `apply_h_kinema` の cache-blocked
  形は維持 (hot path 触らない). `apply_h_drv` / `apply_h_p_diag` primitive を
  `src/matvec.rs` に追加し, crate-internal `cfm4_step(iter0_cache: Option<...>)`
  引数で渡す.

## 0.7.0 - 2026-05-18 — Phase 7 (Lanczos β_m exposure + Richardson 誤差源分離)

Phase 6 C4 (#65) で観測された adaptive CFM4 Richardson driver の Pareto 劣位
(Krylov 誤差と Magnus 誤差が PI controller で区別されない問題) を解消する
ための **infrastructure** を導入. Phase 7 は a posteriori 推定子の expose と
PI controller の誤差源分離駆動までで, Lanczos 圧縮を実際に発火させる本丸
(早期打切判定式の更新) は Phase 8 (#98) に分離.

### Added

- **issue #93 / PR #94**: Lanczos β_m + |c_m| を return tuple に expose.
  - `lanczos_propagate` (Rust + Python ref) の return が 4 要素
    `(psi, m_eff, β_m, |c_m|)` に拡張. Saad/Hochbruck-Lubich の a posteriori
    誤差推定子 `err_lanczos ≈ β_m · |c_m| · ‖ψ‖ · dt / m_eff` (5% 精度;
    `tools/verify_beta_m_estimator.py` で 108 cell sweep 実証).
  - `cfm4_step` / `cfm4_step_with_richardson_estimate` が triangle inequality
    で `err_lanczos_sum` / `err_lanczos_total` を集約して上位伝播.
  - `evolve_schedule_adaptive_richardson` の return tuple が 10 要素に拡張
    (`+ beta_m_history`, `err_lanczos_history`, `err_magnus_history`,
    `n_krylov_insufficient`).
  - `QuantumResult` に `beta_m_stats` / `n_krylov_insufficient` フィールド追加.
    詳細: `docs/design/12-release-plan.md` Phase 7,
    `docs/design/05-3-propagator.md` "Richardson 誤差源分離".

### Changed

- **issue #93 / PR #94**: PI controller の駆動量を
  `err_magnus = max(0, err - err_lanczos_total)` に切替え. default
  `krylov_tol = 1e-12` では `err_lanczos << tol_step` で `err_magnus ≈ err`
  となり既存挙動とほぼ等価 (`test_adaptive_richardson_error_decomposition_consistency`
  で担保).
- **issue #93 / PR #94**: `benchmarks/bench_qutip_large.py` に
  `--krylov-tols` sweep オプションを追加 (`atol × krylov_tol` クロス評価).
  `auto` キーワードで内部自動結合 (= `tol_step · 1e-3`) を表現.

## 0.6.0 - 2026-05-18 — Phase 6 (並列化 + 仕上げ)

Phase 1-5 で算法面が出揃った後の実装面の並列化と仕上げ. rayon
`par_chunks_mut` による L2 並列化 (C1), `wide::f64x4` による SIMD 特化
(C2 / C2.5), multi-qubit gate fusion + phase_p 並列化 (C3), BLAS feature
on/off の数値一致 artifact test + 大規模 QuTiP 比較 (C4), Quick start
サンプル + docs / version 仕上げ (C5) を含む.

### Added

- **issue #62 / PR #67** (Phase 6 C1): `src/matvec.rs` の bit-flip pass
  primitive を rayon `par_chunks_mut` で L2 並列化.
  - `apply_h_kinema`: `y` を `(dim / (nth·4))` を目安に chunk 分割し,
    各 chunk closure 内で diag pass + 全 i bit-flip pass を fuse
    (cache-blocked 形). `y_chunk` を L1 cache resident に保ち, 後段 SIMD
    (C2) / cache block-fusion (C3) の足場とする.
  - `apply_single_mode_axis_i`: `block = 2·mask` 単位で `par_chunks_mut` 並列化.
    退化ケース `i = n-1` では `split_at_mut` + `par_iter_mut().zip` フォールバック.
  - Cargo: `rayon = "1"` optional dep + `[features] rayon` (default ON,
    BLAS と同じ on/off pattern). thread 数制御は `RAYON_NUM_THREADS` (rayon
    の global pool はプロセス起動時に決まる). 併用時は `set_blas_threads(1)`
    推奨 (CLAUDE.md "Thread pool 運用" 節).
  - `benchmarks/bench_parallel_scaling.py` を新規追加.
  - 数値: rayon あり/なしで `y` / `psi` **bit-identical** を
    `apply_*_rayon_matches_serial` (`to_bits()` 一致) + 8 thread × 100
    反復の race-detection fuzz test で担保.
- **issue #68** (Phase 6 C1 follow-up): `MIN_RAYON_DIM = 1 << 17` の dim 閾値
  dispatch を public `apply_h_kinema` / `apply_single_mode_axis_i` に追加.
  dim < 128K (= N ≤ 16) では rayon barrier overhead が単スレッド時間を超えて
  regression するため scalar 経路にフォールバック. `bench_parallel_scaling.py`
  に `trotter_step` cell 追加 + knee detection を max-speedup baseline +
  95% plateau に置換.
- **issue #63** (Phase 6 C2): `apply_h_kinema` の bit-flip pass の
  i ∈ {0, 1, 2} を `wide::f64x4` で SIMD 特化 (`simd_kernels::bitflip_iN`).
  PR #73 で `coeff == 0` 短絡, PR #74 で `read_unaligned` /
  `write_unaligned` + `mul_add` に書き直し AVX `vmovupd` + `vfmadd231pd`
  へ折り畳む. `feature = "simd"` (default ON, `--no-default-features` で
  scalar fallback). build マシン CPU の AVX2 / AVX-512 / NEON は
  `-C target-cpu=native` (#103 で repo 同梱の `.cargo/config.toml` 経由で
  default 適用) で自動的に拾う.
- **issue #71 / PR #80** (Phase 6 C2.5): `apply_single_mode_axis_i` の
  i ∈ {0, 1, 2} を SIMD 特化 (`simd_kernels::single_mode_iN`). 2×2 complex
  matmul を complex broadcast + in-register swizzle で `f64x4` 化
  (`u_k · x_pair = splat(u[k].re)·x_pair + [-u[k].im, u[k].im, ...]·x_swap`
  の 2 Complex64 並列). C3 の `apply_fused_axes_to_chunk` inner kernel から
  共通 dispatch.
- **issue #64 / PR #78** (Phase 6 C3): Trotter `trotter_step` の multi-qubit
  gate fusion + `phase_p` 並列化.
  - 連続 k qubit (default k=4) の R_i を 1 つの rayon chunk closure 内で
    per-axis 2-pair update として逐次実行 (TFIM の per-site commuting 性質を
    利用し exact). barrier 数を `2n+2` → `n/k + 2` に縮める.
    qsim 流 dense 2^k × 2^k matmul は per-axis 逐次へ書き直し (PR #78 fixup).
  - `phase_p(dt/2)` を rayon `par_iter_mut` で並列化.
  - chunk_size は動的計算 `(dim/(nth·4)).clamp(MIN, MAX)`. group_block の整数倍
    に丸めて SIMD path の block-aligned 前提を満たす.
- **issue #65** (Phase 6 C4): BLAS feature on/off の数値一致 artifact test +
  大規模 QuTiP 比較.
  - `tests/test_blas_consistency.py`: 固定 seed の sample 入力で psi_final /
    probabilities / observables 時系列を `.npz` に dump. `KINEMA_EXPECT_BLAS`
    env で build mode を pin できる.
  - `tools/diff_blas_artifacts.py`: BLAS on / off ビルドの `.npz` を読んで
    全 array が `rel < 1e-13` で一致することを assert する standalone script.
  - `tests/test_reference_qutip.py`: n=12-14 で 4 method を QuTiP `sesolve`
    (atol=1e-12) と fidelity 比較. n=15-16 は cfm4_adaptive_richardson のみ
    (sparse 経路). n>=14 は `@pytest.mark.slow`.
  - `benchmarks/bench_qutip_large.py`: dt sweep で QuTiP vs kinema 固定 dt
    method の fidelity と wall time を 1 pass 同時測定 (work-precision diagram).
  - 派生 bench `benchmarks/bench_m_eff_adiabatic.py`: Krylov subspace 次元の
    schedule 依存性を計測.
- **issue #66 / 本 PR** (Phase 6 C5 / finalize):
  - `docs/quickstart.md` を新規作成 (最小例 / Observable + save_tlist /
    AnnealingSimulator step-wise / instantaneous_eigenstates の 4 snippet).
  - `README.md` に Quick start リンクを追加.
  - `docs/design/*.md` の Phase 6 関連「予定」文言を実装済表記に整理.
    `INDEX.md` を `(v0.5)` → `(v0.8)`, `13-future-work.md` を `v0.6` →
    `v0.8` に追従.
  - `docs/conventions.md` バージョニング表に Phase 7 / Phase 8 / 遡及 bump
    の運用ノートを追加.
  - `pyproject.toml` / `Cargo.toml` の `version` を `0.5.0` → `0.8.0`
    (Phase 6 / 7 / 8 のマージ済み変更を遡及的にまとめて版数化).
- **issue #82**: `src/bin/perf_trotter_step.rs` 追加. Linux `perf stat` で
  hardware counter から `trotter_step` の真の compute speedup を再評価
  (Python bench の alloc/copy overhead を切り出し). C3 trotter_step の
  N=20 4.01× (Python bench) → 5.30× (perf binary) と再 verify.
- **issue #90**: `src/bin/perf_apply_single_mode_axis_i.rs` 追加. #71 fixup
  `578d050` (chunk_size 動的化) を perf binary で再評価し棄却を撤回, 動的
  chunk_size `(dim/(nth·4)).clamp(...)` を採用.
- **issue #85 / #86**: `apply_h_kinema_py` / step 系 `_py` wrap に in-place
  入口 (`*_into_py` / `*_inplace_py` 計 5 関数) を追加. Python bench の
  alloc-and-return overhead を排除する経路.
- **issue #95**: `bench_qutip_large.py` の ty 型診断 2 件を解消.
- **issue #103 / PR #104**: production profile + `target-cpu=native` を
  `uv add git+...` 経由のソースビルドに自動適用.
  - `Cargo.toml::[profile.production]`: `inherits = "release"`, `codegen-units = 1`,
    `lto = "fat"`, `panic = "abort"`.
  - `pyproject.toml::[tool.maturin]`: `profile = "production"`, `strip = true`.
  - `.cargo/config.toml::[build] rustflags`: `["-C", "target-cpu=native"]`.
  - `kinema.show_config()` (numpy.show_config 相当) を追加し,
    `_rust.__has_avx2__` / `__has_fma__` / `__has_avx512f__` / `__has_neon__`
    / `__target_arch__` / `__target_os__` を expose.

### Changed

- **issue #83**: 単一ファイル `docs/design.md` (v0.5 時点 2359 行) を
  `docs/design/INDEX.md` + 章別 17 ファイルに分割. 内容変更なし
  (§N.M 番号と章順を保存). 以降の docs 整合は分割ファイル単位で扱う.

### Performance

bench は Phase 6 finalize で 4 種類 (`bench_per_step` /
`bench_parallel_scaling` / `bench_block_fusion` / `bench_qutip_large`) を
固定 Linux サーバー (AMD EPYC 7713P, cpu_count=64, OpenBLAS, AVX2 + FMA)
で実行し PR コメントに添付する (memory `project_bench_machine`).
Phase 6 全体の **Phase 1 baseline → Phase 6 (rayon + SIMD + cache
block-fusion 全部 on) 累積改善** は PR / umbrella issue #61 コメントに
集約.

Phase 6 中の確定済み観測値 (Linux AMD EPYC 7713P):

- `trotter_step` N=20: **4.01×** (#64 C3, Python bench), perf binary 再評価で
  **5.30×** (#82).
- `apply_single_mode_axis_i` N=20 rayon path SIMD on/off (i=0/1/2):
  **2.71-3.48×** (#71 C2.5).
- `apply_single_mode_axis_i` N=16 serial path SIMD on/off (i=0/1/2):
  **1.88-2.43×** (#71 C2.5).
- `apply_h_kinema` per-pass SIMD on/off: **~1.75×**, `i012-focus` mode
  total ~1.28× (#63 C2; acceptance 1.5× は未達のまま, DRAM bandwidth は
  C3 では touch せず #79 D で試行・未採用).

### Archived (試行・未採用)

- **issue #79** (Phase 6 D): `apply_h_kinema_rayon` を **連続 k 個の高 i を
  group-fused 3-phase 形** に書き換える試み. DRAM v traffic を理論上
  `dim · (1 + h_baseline) → dim · (1 + h_naive)` に削減する設計だったが,
  Linux AMD EPYC 7713P で perf 計測 (`src/bin/perf_apply_h.rs` 新設) した
  結果 **N=20 で 50% 真の compute regression** を確認し revert. C1 baseline
  は IPC=2.98 で既に compute-near-peak で「DRAM bound」前提が誤り, 3-phase
  access pattern が HW prefetcher を破壊し per-L2-miss avg latency が
  195 → 251 cycles (+30%) に劣化. 詳細は `docs/design/05-1-matvec.md` §5.1.4.
  B (SIMD i≥3), C (prefetch), D (streaming store) も同前提では効果薄と判断,
  別 sub-issue 化していない. **残した資産**: `src/bin/perf_apply_h.rs`
  (今後の Phase 6+ 改善の検証基盤として価値あり).

### Internal note

CHANGELOG: Phase 5 finalize 時 (commit `49dd673`) に旧
`Unreleased — Phase 4 follow-up` セクションを `## 0.5.0` に繰り上げ忘れて
いたため, Phase 6 C1 PR で遡及的に促進した. Phase 6 / 7 / 8 はマージ済みの
段階で各 `0.6.0` / `0.7.0` / `0.8.0` への bump が行われず `v0.5.0` のまま
停止していたが, Phase 6 finalize (#66) で 3 リリースをまとめて遡及的に
版数化した (memory `project_version_bump_policy`). 内容は各 Phase 完了時に
版数化したケースと等価.

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
  - 詳細根拠は `docs/design/05-3-propagator.md` §5.3 follow-up 節 E "adaptive driver
    default の統一".
