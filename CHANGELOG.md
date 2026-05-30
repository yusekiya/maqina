# Changelog

`maqina` の公開 API 破壊的変更と Phase 単位の差分を集約する.

- 運用ポリシー: `docs/conventions.md` §2 (バージョニング) / §2.2 を一次
  資料とする. **mid-Phase で取り込まれた破壊的変更も本ファイルに時系列
  で記録** し, 次の Phase 完了 bump 時に release notes / commit message
  起こしの一次資料として参照する.
- フォーマット: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
  と SemVer 0.x.y の慣習に概ね準拠. ただし v0 段階のため MINOR
  (`0.N.0` → `0.N+1.0`) で破壊的変更を吸収する (`docs/conventions.md`
  §2 参照).

## 0.14.0 - 2026-05-30 — Phase C follow-up: adaptive step-size controller 追従性改善 (真の PI 化 / reject 過剰縮小解消 / `ControllerConfig` 公開, umbrella #148)

### Breaking

- **adaptive reject 時の dt 縮小を固定 0.5 から予測式 + クランプに変更**
  (issue #149, umbrella #148): 3 つの adaptive ドライバ
  (`evolve_schedule_adaptive_{m2,richardson,richardson_chebyshev}`) の reject
  時の dt 更新を **固定 0.5 倍** から **accept と同じ予測式
  `safety · (tol_step / err)^{1/(p+1)}` + reject 用クランプ
  `[reject_shrink_min, reject_shrink_max]` (既定 `[0.2, 0.9]`)** に変更。
  order-5 推定子で `err` が tol をわずかに超えただけでも誤差を 32× 削る過剰
  縮小がノコギリ波 (dt 振動 / 受理率 ≈ 50%) の主因だったのを断つ。既定挙動が
  変わるため破壊的変更。旧挙動 (固定半減) は
  `ControllerConfig(reject_shrink_min=0.5, reject_shrink_max=0.5)` で厳密再現
  可能 (回帰アンカー)。誤差源分離は accept 経路と整合させ Richardson /
  Chebyshev は `err_magnus = max(0, err - err_propagator_total)`、M2 は `err`
  を駆動量に使う。

- **reject 後の dt 成長凍結 (Gustafsson ヒステリシス)** (issue #150,
  umbrella #148): 3 つの adaptive ドライバの accept 経路で、reject 直後から
  `growth_freeze_steps` 回 (既定 1 = DOPRI/Hairer-Wanner 標準) 連続 accept する
  まで dt 拡大を `eff_growth_max = 1.0` に凍結する (拡大のみ禁止、縮小は許可)。
  「reject → 過剰縮小 → 楽々 accept → I 制御が大きく再拡大 → 再オーバーシュート」
  という limit cycle の **再拡大側** を断つ。`ControllerConfig` に
  `freeze_growth_after_reject` (既定 `True`) / `growth_freeze_steps` (既定 1) を
  追加。既定有効のため破壊的変更。#149 完了時点の挙動は
  `ControllerConfig(freeze_growth_after_reject=False)` でビット一致再現可能
  (回帰アンカー)。なお #149 の予測式 reject は既定クランプ `[0.2, 0.9]` 下では
  reject 直後の accept を PI 成長率 ≈ 1.0 に着地させるため成長凍結は発火せず、
  本機構は user が `reject_shrink` を攻めた値にした場合や縮小 factor が
  `reject_shrink_min` 床にクランプ → 過剰縮小する場面での二重の安全網として効く。

- **真の PI 比例項を導入し I 制御 → PI 制御化** (issue #151, umbrella #148):
  3 つの adaptive ドライバの accept 時 dt 予測式に Gustafsson / Hairer-Wanner
  II §IV.2 の predictive PI 比例項を追加し、
  `dt_next = dt · safety · (tol_step / err)^{pi_alpha/(p+1)} · (err_prev / err)^{pi_beta/(p+1)}`
  とした。比例項が誤差の増加傾向 (Magnus 4 次係数 C₄ の上昇) を `err_prev / err`
  で先読みして dt 拡大を抑制し、臨界領域のノコギリ波を「再オーバーシュート前に」
  平坦化する (#150 の reject 後成長凍結と相補的)。`err_prev` は直前に **accept**
  した step の駆動量 (M2 = `err`, Richardson / Chebyshev = `err_magnus`; Hairer の
  `errold` 規約で reject を挟んでも最後に accept した値を保持)。これまでの実装は
  比例項の無い純 I (積分) 制御で docstring / 設計書の "PI controller" 表記と実体が
  不整合だったのを解消する。`ControllerConfig` に `pi_alpha` (既定 `0.7`) /
  `pi_beta` (既定 `0.4`, Gustafsson 標準) を追加し、`__post_init__` で
  `pi_alpha > 0` / `pi_beta >= 0` を検証。既定挙動が I → PI に変わるため破壊的
  変更。純 I 制御 (旧挙動) は `ControllerConfig(pi_alpha=1.0, pi_beta=0.0)` で
  ビット一致再現可能 (回帰アンカー)。既定値は合成誤差ハーネス (#152) のノコギリ波
  シナリオで確認 (`pi_beta=0.4` が reject 削減と autocorr 改善の sweet spot、
  `pi_beta=0.6` は過補正で振動再発)。

### Added

- **`ControllerConfig` (frozen dataclass) 新設 + facade 配線** (issue #149,
  umbrella #148 方針 B): adaptive PI controller の純粋な数値挙動 knob
  (`safety` / `growth_max` / `max_rejects` / `dt_min` / `reject_shrink_min` /
  `reject_shrink_max`) を 1 つの dataclass に集約し `maqina` から export。
  `QuantumAnnealer.run(controller=...)` / `create_simulator(controller=...)` /
  `AnnealingSimulator(controller=...)` に配線し、**これまで facade から指定
  できなかった** `safety` / `growth_max` (= facmax) / `max_rejects` / `dt_min`
  をまとめて公開 (取りこぼし解消)。`None` 既定で全 default。`atol` / `dt_init`
  / `dt_max` / `m_max` は精度要求・auto-resolve ロジックを持つため
  `ControllerConfig` には入れず既存 kwarg 維持。固定 dt 経路への明示
  `controller` は `AnnealingSimulator` のみ `ValueError` (run は寛容)。
  後続 #151 (真の PI 化) はこの dataclass に field を追加していく。
  `tests/test_controller_reject_clamp.py` に層 A 合成比較 / 回帰アンカー /
  入力検証 / facade 配線テストを追加。
  issue #150 で `freeze_growth_after_reject` / `growth_freeze_steps` の 2 field を
  追加 (上記 Breaking 参照)。`tests/test_controller_growth_freeze.py` に層 A
  (over-shrink ノコギリ波緩和) / 非劣化 (既定クランプ) / 解除基準 (凍結ステップ数
  単調性) / 回帰アンカー / 入力検証 / facade 全 8 field 配線テストを追加。

- **`Schedule.reverse(..., pause_duration=0.0)`** (PR #147): `s_target` に到達
  後一定時間 `s = s_target` を保ってから `s_init` に戻る reverse + pause
  schedule (Marshall-Venturelli-Rieffel 2019 / Chen-Lidar 2020 流) を表現可能
  に。ramp 区間は前後対称で `(T - pause_duration) / 2` ずつ。`pause_duration < 0`
  または `>= T` で `ValueError`。`pause_duration = 0` で従来の純 V 字形に縮退
  するため既存ユーザー影響なし。Crosson-Harrow 2016 流の純 V 字専用 builder を
  別途追加するのではなく既存 `Schedule.reverse` のオプション引数として統合し,
  D-Wave 文献の実験プロトコル (warm-start + pause) を 1 builder で表現する設計
  に統一。`tests/test_schedule.py` に 3 テスト追加 (pause 区間定数性 /
  `pause_duration=0` の従来 V 字一致 / validation)。

## 0.13.0 - 2026-05-28 — Phase C: per-site/per-axis 時間依存場 (XYZ driver) 拡張 + `Schedule` に `h_x` 統合 (issue #142)

旧 API の **driver = X 単体 + global scalar envelope** という制約
(`h_x_i` 静的 / time dep は global `A(s(t))` のみ; Y / Z 軸の時間依存場や
per-site で異なる時間関数は表現不可) を解消し, per-site / per-axis に独立な
時間依存場に対応する Hamiltonian 形 `H(t) = Σ_i [g_x_i(t)·X_i + g_y_i(t)·Y_i
+ g_z_i(t)·Z_i] + b(t)·H_p_diag` に拡張する Phase C 完了。責任分担も同時に
再整理し **`Schedule` に `h_x` を統合**, **`IsingProblem` は問題側静的構造
(`H_p_diag`) のみ** の pure data container に slim 化。

### Breaking

- **`IsingProblem(n, H_p_diag, h_x)` → `IsingProblem(n, H_p_diag)`**: 横磁場
  振幅 `h_x` を `IsingProblem` から削除し `Schedule` に移管。`h_x` 引数を
  渡すと `TypeError`。`IsingProblem.h_x` / `IsingProblem._h_x_abs_sum` 属性も
  削除。問題側の static structure と時間発展係数の責任分担を明確化する変更。
- **`Schedule(T, A, B)` → `Schedule(T, A, B, h_x, s=None)`**: `h_x` を必須
  引数化。preset (`Schedule.linear` / `Schedule.from_callable` /
  `Schedule.reverse` / `Schedule.pause`) も `h_x` 必須化。`Schedule.h_x` /
  `Schedule.h_x_abs_sum` は旧 API only (新 API で呼ぶと `RuntimeError`)。
- **`Schedule.from_xyz(T, g_x, b, *, g_y=None, g_z=None)` 新 constructor**:
  per-site/per-axis 時間依存場の新 API。callable list ベース (`Sequence[
  Callable[[float], float]]`)。`g_y` / `g_z` は `None` で当該軸 skip (Rust
  側で real-only SIMD kernel に dispatch する fast path)。callable に振幅は
  既に組み込み済の前提 (例: 旧 API `g_x_i(t) := -A(s(t))·h_x_i` 相当を新
  API で書くなら `g_x = [lambda t, hi=hi: -(1 - t/T)*hi for hi in h_x]`)。
- **`method="trotter"` / `"trotter_suzuki4"` は新 API で `ValueError`**:
  Trotter 経路の `apply_single_mode_axis_i` は実数係数前提の SIMD で組まれて
  いるため XYZ 一般化は scope 外。必要時に別 issue 起票。Magnus 経路
  (`cfm4_adaptive_richardson_chebyshev` / `_krylov` / `cfm4` / `m2`) は新 API
  でも問題なく動作する。
- **driver function (`evolve_schedule_*`) から `h_x` 引数を撤廃**:
  `schedule._eval_stage(t)` 共通 evaluator 経由で per-axis arrays を取得して
  XYZ Rust wraps に dispatch する形に refactor。internal API なのでユーザー
  への影響は限定的だが, driver を直接呼んでいた場合は移行が必要。

### Added

- **Rust 側 XYZ 一般 matvec primitive `apply_h_general`** (`src/matvec.rs`,
  PR #143): per-site/per-axis (`g_x`, `g_y: Option`, `g_z` (per-stage 事前
  畳込み `gz_eff_diag: Option`)) を受け取る合成 matvec。X+Y は複素係数
  `c_i = g_x_i + i·g_y_i` に畳んで 1 bit-flip pass で完了, `g_y=None` /
  `gz_eff_diag=None` のとき既存 real-only SIMD/scalar kernel に dispatch
  する Option-based skip 経路。詳細は `docs/design/05-1-matvec.md` §5.1.1.y。
- **SIMD complex-coeff bit-flip kernels** (`src/matvec.rs::simd_kernels::
  bitflip_iN_complex`, i ∈ {0,1,2}, PR #143): `wide::f64x4` で complex coeff
  を broadcast + swizzle 経由で乗せる (Phase 6 C2.5 single_mode_iN と同方針)。
- **per-stage Gershgorin (Chebyshev variant)** (`src/cfm4.rs::compute_stage_
  gershgorin`, PR #144 / #145): `(R_off_stage, diag_min_stage, diag_max_stage)`
  を per-stage に計算 (gz_eff_diag doubling 構築と fuse して O(2^N) → 単一
  pass)。Lanczos variant は per-stage Gershgorin 不要 (`build_stage_arrays`
  と分離)。X-only 旧 API 向け O(1) 変換 helper `gershgorin_per_stage_x_only`
  も追加。
- **`compute_gz_eff_diag` doubling builder** (`src/matvec.rs`, PR #143):
  `Σ_i g_z[i]·σ_z_i(k)` (length 2^N) を degree-1 Walsh 多項式の doubling /
  butterfly で O(2^N) 構築 (償却 O(1) per 基底)。
- **`reference_qutip.build_qutip_hamiltonian_xyz(...)`**: per-axis 時間依存場の
  QuTiP `H(t)` を sparse 構築する helper。`tests/test_xyz_schedule.py` で XY
  rotating field / Z-only field 等の QuTiP fidelity 一致テストに使う。
- **新規テスト `tests/test_xyz_schedule.py`** (9 ケース): 旧 API smoke / 新
  API == 旧 API equivalence (m2 / cfm4 / adaptive Richardson) / XY rotating
  field QuTiP fidelity / Z-only field QuTiP fidelity / Trotter ValueError /
  IsingProblem h_x なし smoke。

### Changed (signature shuffle / 数値挙動は不変)

- **`cfm4_step` / `cfm4_step_with_richardson_estimate` (Lanczos variant)**
  (PR #145): 旧 scalar 4 個 `(a_s1, b_s1, a_s2, b_s2)` → per-axis array +
  scalar `(g_x_s1, g_y_s1, g_z_s1, b_s1, g_x_s2, g_y_s2, g_z_s2, b_s2)`。
  `build_stage_desc` を `build_stage_arrays` + `compute_stage_gershgorin`
  に split し Lanczos 経路で O(2^N) Gershgorin walk を回避。`iter0_cache`
  を 4-tuple `(cache_drv, cache_diag, a_s1_scalar, a_s2_scalar)` に拡張。
  perf_cfm4_richardson full mode で main 比 **-5.81% cycles** (per-matvec
  の g_x_buf alloc 消失の副次効果)。
- **`chebyshev_propagate` / `cfm4_step_chebyshev*` (Chebyshev variant)** (PR
  #144): 同じく per-axis 配列形に切替, `gershgorin_bounds_cached` を
  per-stage form (`r_off_stage`, `diag_min_stage`, `diag_max_stage`) に変更,
  `apply_h_general` 直呼出に切替。perf_cfm4_richardson_chebyshev full mode で
  main 比 +0.05% (実質不変, h_p_diag が L2 cache resident)。
- **`apply_h_kinema` → `apply_h`** (commit `845e4e3`): `apply_h_general` の
  X-only 特化 wrap として内部実装を統一。Rust 単体テスト互換 + perf 回帰
  測定 baseline のため shim を残置。
- **Lanczos / CFM4:2 系の Rust hot path は `apply_h_general` 経由に切替**
  (PR #144 / #145): per-matvec の引数 plumbing のみ変更, Lanczos / Chebyshev
  両 driver の数値挙動は不変。

### Documentation

- `docs/design/02-physics.md` §2.1.1 旧 API / §2.1.2 新 API (per-site/per-axis
  時間依存場) を分け, §2.2 ユーザー入力の `h_x` 所在 (旧 API: Schedule, 新
  API: callable 組込み) を反映。
- `docs/design/05-1-matvec.md` primitive table + §5.1.1.y `apply_h_general`
  節を追加 (per-axis bit-flip complex-coeff 拡張 + Option-based skip 経路 +
  SIMD per-axis 分岐の設計)。
- `docs/design/05-3-propagator.md` CFM4:2 節直後に per-stage 配列 /
  Gershgorin / `gz_eff_diag` doubling 構築の節を追加。
- `docs/design/12-release-plan.md` Phase C 節を新規追加 (動機 / Out of scope
  / Definition of Done / phasing 実績 / Out of scope follow-up)。
- `docs/design/INDEX.md` バージョン v0.11 → v0.13, Phase C エントリ + 横断
  トピック XYZ driver 行を追加。
- `CLAUDE.md` 物理的取り決め節を旧 / 新両 Hamiltonian 形 + 責任分担 (Schedule
  に h_x 統合) に更新。
- `docs/quickstart.md` 既存 5 例の `IsingProblem(... h_x=)` を
  `Schedule.linear(... h_x=)` に bind し直し, 新 §6 で `Schedule.from_xyz` の
  XY rotating field 例 + Trotter ValueError 注を追加 (旧 §6 thread-count は
  §7 に番号繰下げ)。

### Internal

- `tools/gen_api_stubs.py`: class メソッドの単一 underscore prefix
  (`_eval_stage` 等) も stub に含めるよう調整 (cross-module 型チェックの
  ため; 公開境界の制御は `__all__` / docstring に委ねる)。
- `_helpers.py::_gershgorin_norm_upper_bound(schedule, problem)` /
  `_resolve_dt_max_auto(schedule, problem, m, dt0)`: signature 変更
  (schedule から h_x_abs_sum (旧 API) または time-sampled
  `_norm_upper_bound_factor_at` (新 API) を取得)。`np.max(np.abs(H_p_diag))`
  の O(2^N) walk を `problem.h_p_diag_min/max` precompute 経由の O(1)
  closed-form に置換 (PR #146 follow-up と統合)。

### Fixed (PR #146 follow-up — Linux 本番 perf bench で発覚)

- **Chebyshev per-stage Gershgorin の O(2^N) walk regression**: PR #144 で
  `gershgorin_bounds_cached` (O(1)) を per-stage form (`compute_stage_gershgorin`)
  に再設計した際, `gz_eff_diag = None` (= 時間依存 Z 磁場なし; legacy API
  および新 API の XY-only 経路) でも `h_p_diag` の full walk が走っていた。
  per Richardson step で 6 stage × O(2^N) のオーバヘッドが発生し N=18 で
  per-step wall に数 ms / N=20 で十数 ms の regression として観測される。
  - 修正: `compute_stage_gershgorin(arrays, h_p_diag, h_p_min, h_p_max)`
    シグネチャに `IsingProblem` 構築時 precompute 値 (`h_p_diag_min` /
    `h_p_diag_max`) を渡し, `gz = None` 経路では closed-form
    `(c_b · h_p_min, c_b · h_p_max)` (c_b 負なら swap) で O(1) 計算する。
    `gz = Some(...)` 経路は構造的に O(2^N) walk 必要 (合成
    `c_b · h_p_diag[k] + gz[k]` の min/max).
  - 影響範囲 (内部 Rust API): `cfm4_step_chebyshev` /
    `cfm4_step_chebyshev_with_richardson_estimate` に `h_p_min: f64,
    h_p_max: f64` 引数追加。Python wrap (`cfm4_step_chebyshev_xyz_py` /
    `cfm4_step_chebyshev_with_richardson_estimate_xyz_py`) にも追加。
  - 影響範囲 (内部 Python API):
    `evolve_schedule_adaptive_richardson_chebyshev` driver に
    `h_p_min, h_p_max` キーワード引数追加 (必須)。`QuantumAnnealer` /
    `AnnealingSimulator` からは `problem.h_p_diag_min/max` を渡す形に更新。
  - 旧 X-only Python wrap (`cfm4_step_chebyshev_py` /
    `cfm4_step_chebyshev_with_richardson_estimate_py`) は既に
    `h_p_min, h_p_max` を末尾引数で受け取っており shim 内で ignore して
    いたが, 修正で内部 `cfm4_step_chebyshev*` に propagate する形に。
  - Lanczos 経路 (`cfm4_step` / `cfm4_step_with_richardson_estimate`) は
    per-stage Gershgorin を呼ばないため影響なし (`build_stage_arrays` のみ
    使用; PR #145 で分離した設計通り)。
  - **acceptance テスト追加**: `src/cfm4.rs` 単体に
    `compute_stage_gershgorin_no_z_matches_full_walk` (gz=None 経路で
    precompute 値 ↔ full walk の bit-identical 一致, 4 通りの c_b 符号で)
    と `compute_stage_gershgorin_with_z_walks_correctly` (gz=Some 経路で
    precompute 引数が誤って使われないことの保証)。Python 側に
    `test_chebyshev_h_p_bounds_match_h_p_diag_min_max` (tight bound と
    意図的に緩めた bound で終端 ψ が `chebyshev_tol` 程度に一致することの
    確認 — Gershgorin の意味的役割 (K_used を決める E_c/R 推定) が壊れて
    いないことの契約)。

### Verification (本セッション再開時の test-runner subagent 並列実行)

- ✅ `cargo test` (BLAS + rayon ON): 101 passed
- ✅ `cargo test --no-default-features`: 87 passed (BLAS / rayon 両 OFF, scalar
  単スレッド経路, `rel < 1e-13` で BLAS on と一致)
- ✅ `uv run maturin develop --uv` + `uv run pytest` (slow 含む): 362 passed
  (test_xyz_schedule.py 9 cases / test_schedule.py 18 / test_problem.py 8 /
  QuTiP 中規模 fidelity 含む)
- ⏸️ 旧 API 経路 `bench_per_step.py` regression ±1% 以内 (X-only path): Linux
  サーバー本番 bench で pre-merge cycle 中に確認 (`.claude/solve-overrides.md`
  「Bench を伴う issue の運用」節)。

## 0.12.0 - 2026-05-26 — `krylov_tol` → `propagator_tol` rename + Chebyshev default 仕様変更 (issue #135) + パッケージ rename `kinema → maqina`

`cfm4_adaptive_richardson_chebyshev` method の精度パラメータを 2 軸で整理:
parameter 名の semantic 統一 (Krylov 部分空間概念は Chebyshev には無いので
misleading だった) + Chebyshev variant の default 変更 (atol↓ で精度が
非単調に劣化する auto-coupling から, atol-vs-infidelity の monotonicity を
担保する固定値 1e-12 に変更).

### Breaking

- **`QuantumAnnealer(..., krylov_tol=...)` → `propagator_tol=...`**: 公開
  API シグネチャ rename. deprecation alias は残さない (旧 kwarg は
  ``TypeError``). 影響: `QuantumAnnealer` / `AnnealingSimulator` の
  constructor 引数, attribute (`self.krylov_tol` → `self.propagator_tol`),
  関連 docstring. `tests/test_chebyshev.py` の
  `test_old_krylov_tol_kwarg_raises_typeerror` で contract を保証.
- **Chebyshev variant の `propagator_tol = None` default 変更**:
  `cfm4_adaptive_richardson_chebyshev` 経路で `propagator_tol = None`
  (未指定) のとき, 旧挙動 `tol_step · _KRYLOV_TOL_ATOL_RATIO` (auto-coupling)
  から **固定値 `_KRYLOV_TOL_FIXED_DEFAULT` (= 1e-12)** に変更. 旧挙動を
  再現したい場合は `propagator_tol=tol_step * 1e-3` を明示渡し. Lanczos
  variant (`cfm4_adaptive_richardson_krylov`) は auto-coupling 維持
  (Lanczos a posteriori 早期打切は atol scaling 連動が望ましいため).
- **`benchmarks/bench_qutip_large.py --krylov-tols` → `--propagator-tols`**:
  CLI flag rename. parse 関数も `_parse_krylov_tol_list` →
  `_parse_propagator_tol_list`. 共通 sweep 軸として Lanczos / Chebyshev
  両 method で機能する.

### Added

- `benchmarks/bench_readme_figure.py` に `--propagator-tol` flag 追加
  (default `None` → Chebyshev は 1e-12 固定). `scripts/run_bench_readme_chebyshev.sh`
  も `CHEBYSHEV_PROPAGATOR_TOL` shell 変数で明示 pass.

### Tests

- `tests/test_chebyshev.py::test_chebyshev_default_propagator_tol_is_fixed_1e_minus_12`:
  atol を 2 桁振っても (1e-6 vs 1e-8) Chebyshev K_used 平均の変動が 20% 未満
  であることで auto-coupling されていないことを確認.
- `tests/test_chebyshev.py::test_old_krylov_tol_kwarg_raises_typeerror`:
  旧 `krylov_tol` kwarg で `TypeError` (alias 残さない契約).

### Motivation

PR #134 (README figure pipeline) で `atol = 1e-5` のとき
`atol = 1e-4` (machine precision 到達) より infidelity が悪化する非単調性
を実測 (PI controller が小 dt を選び round-off accumulation, per-step
Chebyshev 打切は既に machine precision floor なので atol tightening が
無効). 固定 1e-12 で K_used を atol 非依存にし, Pareto curve の解釈性を
上げる. K_used 増は non-stiff +16% / stiff +3.7% (R·dt 別の Bessel
減衰見積もり) と限定的.

詳細は `docs/design/05-3-propagator.md` "Chebyshev variant" 節.

### Breaking (rename)

- **パッケージ rename: `kinema` → `maqina`**: 公開名・import path・PyPI 名
  すべて変更. `maqina` は "A **Ma**gnus-based **Q**uantum **I**sing
  **N**umerical **A**nnealer" の頭文字 + ラテン語 `machina` (機械/装置;
  Romance 系の `máquina` / `macchina` / `machine` の語源) からの造語.
  semantic は本パッケージの positioning と一致する形に再整理 (旧 `kinema` の
  "Kinetic … Magnus" は本質を捉えていなかった).
  影響範囲は Python パッケージ名 (`python/kinema/` → `python/maqina/`,
  `git mv` で履歴保持), Rust crate / `[tool.maturin] module-name`,
  `.cargo/config.toml` コメント, env var (`KINEMA_EXPECT_BLAS` /
  `KINEMA_ARTIFACT_DIR` → `MAQINA_EXPECT_BLAS` / `MAQINA_ARTIFACT_DIR`),
  docs 全般. Rust 内部関数名 (`apply_h_kinema*` 等) は据置 (後続 issue で
  別途検討). 移行は `from kinema import X` → `from maqina import X` と
  env var prefix の単純置換のみで完了.

## 0.11.0 - 2026-05-23 — Chebyshev variant 統合完了 (Phase B finalize) + パッケージリブランド

Phase B 本体 (#122) で導入した `cfm4_adaptive_richardson_chebyshev` 経路を
follow-up #126 / #127 / #124 で仕上げ, さらに公開前リブランド (#1106f77
`kryanneal → kinema`) と Chebyshev hot path の Gershgorin precompute (#c3f20c7)
を同 minor に取り込む. Phase B + follow-up 全体の perf 効果は Linux AMD EPYC
7713P / N=18 で Lanczos 比 **5.49× wall 高速 / branch-miss 158× 減 / sys time
78× 減 / parallel efficiency 27% → 44%** (#124 perf archive).

### Breaking

- **パッケージリブランド (rename, commit `1106f77`)**: `kryanneal` →
  `kinema` ((Kine)tic quantum evolution by (Ma)gnus expansion) に rename.
  pure rename / 数値挙動変更なし. 影響範囲は Python パッケージ名
  (`python/kryanneal/` → `python/kinema/`), Rust crate / `[tool.maturin]
  module-name`, Rust 内部関数名 (`apply_h_kryanneal*` → `apply_h_kinema*`),
  env var (`KRYANNEAL_EXPECT_BLAS` → `KINEMA_EXPECT_BLAS`), docs 全般.
  v1.0 前の最後の機会としてプロジェクト名を確定. 移行はインポートと env を
  単純置換するだけで完了 (cargo test 92 passed / pytest 347 passed で確認).
- **`QuantumAnnealer.run(method=...)` の default** (#124): `"m2"` →
  `"cfm4_adaptive_richardson_chebyshev"`. 旧 default を使っていたユーザーは
  `method="m2"` を明示するか, 新 default 経路に切替えて `n_steps` の代わりに
  `atol` で精度を制御する.
- **`QuantumAnnealer.create_simulator(method=...)` の default** (#124):
  `"cfm4"` → `"cfm4_adaptive_richardson_chebyshev"`. ついでに `Literal` から
  欠落していた `_chebyshev` を追加 (Phase B #122 取りこぼし fixup).
- **`AnnealingSimulator(method=...)` の default** (#124): `"cfm4"` →
  `"cfm4_adaptive_richardson_chebyshev"`.
- **Chebyshev `gershgorin_bounds*` の precompute 化 (commit `c3f20c7`)**:
  Chebyshev propagator のスペクトル境界推定 (`gershgorin_bounds`) は
  per-call で `h_p_diag` (length 2^N) の full walk を行っており, 1 step
  あたり 6 回呼ばれるため per-step `O(2^N + N)` を消費 (N=18 で wall 1% 弱).
  `h_x` / `h_p_diag` が `IsingProblem` 構築後 immutable な性質を利用し
  `Σ|h_x|`, `min/max(h_p_diag)` を 1 度だけ precompute → `gershgorin_bounds_cached`
  で per-step を **O(1) (5 fp 演算)** に縮めた. pre-1.0 内部 API の破壊的変更:
  - `chebyshev_propagate` のシグネチャに `h_x_abs_sum, h_p_min, h_p_max: f64`
    の 3 引数を末尾追加 (PyO3 binding も同様).
  - `cfm4_step_chebyshev` / `cfm4_step_chebyshev_with_richardson_estimate`
    および両 `_py` wrap も同 3 引数を pass-through.
  - `IsingProblem` に `h_x_abs_sum` / `h_p_diag_min` / `h_p_diag_max` の
    read-only property を追加 (`frozen=True` 維持のため `__post_init__` で
    `object.__setattr__` 経由でセット).
  - Python driver `evolve_schedule_adaptive_richardson_chebyshev` 入口で
    precompute を 1 度実行し step ループ内 dispatcher に渡す形に変更.
  - legacy `gershgorin_bounds` は残存 (内部で cached 版を呼ぶ; docstring に
    hot path で使わない旨を注記).

`_krylov` literal は永続的に残す (旧 default 互換 + 比較ベンチ用途).

### Added

- **Chebyshev 3 項漸化 inner loop の SIMD + fusion (#126, Phase B follow-up)**:
  `src/chebyshev.rs::simd_kernels::chebyshev_recurrence_fused` (`wide::f64x4`) +
  `_scalar` fallback + dispatch wrapper. `chebyshev_propagate` の `k_ord ≥ 2`
  hot loop の 3 dim-walk (matvec / recurrence scaling / accumulate) を **1
  dim-walk + SIMD** に fuse. f64x4 helpers (`as_f64_slice`,
  `load/store_f64x4_unaligned`, `swap_reim`) は localize duplication として
  chebyshev module 内に再実装 (matvec の `simd_kernels` を visibility 跨ぎで
  触らないため). `cfm4_step_chebyshev_*` 経由でも自動で乗る.
- **Chebyshev non-matvec inner loop の rayon 並列化 (#127, Phase B follow-up)**:
  `src/chebyshev.rs::chebyshev_recurrence_fused_rayon`. `scratch` / `psi_acc`
  の 2 RW slice を `par_chunks_mut` 2 本独立に取り `zip().enumerate()` で base
  offset から `phi_curr` / `phi_prev` (R) を共有 sub-slice 切り出し → chunk 内で
  SIMD/scalar fused kernel を呼ぶ 2 段構造. dispatch 閾値
  `MIN_RAYON_DIM_CHEB = 1 << 17` (`matvec.rs::MIN_RAYON_DIM` と揃え). chunk_size
  は `(dim/(nth·4)).clamp(min,max)` を 2 倍数に丸めて SIMD kernel の偶数長前提を
  満たす. parallel efficiency (Phase B 完了時の 44%) のさらなる改善を狙う.
- **`docs/chebyshev-explained.md` (commit `d6d3e23`)**: 時間独立 H に対する
  `exp(-i H dt) ψ` を Chebyshev 多項式 3 項漸化で計算する手順 (Jacobi-Anger,
  スペクトル正規化, 切り捨て次数, Bessel 係数, 3 ベクトル rotation, SIMD/rayon,
  CFM4:2 Magnus 統合, Lanczos との対比) を 10 章構成で段階的に解説する読み物.
  一次資料の `src/chebyshev.rs` / `docs/design/05-3-propagator.md` を補完.
- **`IsingProblem.{h_x_abs_sum, h_p_diag_min, h_p_diag_max}` property** (#c3f20c7):
  Chebyshev Gershgorin precompute 用の read-only property を公開.

### Changed

- **`QuantumAnnealer.run` / `AnnealingSimulator.__init__` の `atol` docstring**
  (#124): "Note (Chebyshev variant の atol 振舞い)" 注を追加. Chebyshev では
  `atol` を upper bound として扱い, K_used 動的拡張により実際の精度がそれより
  良くなる場合があることを明文化 ("feature" 仕様, Scope 2 (a) + (d) 確定).
- **`docs/design/05-3-propagator.md`** (#124, #20d14d0):
  "`chebyshev_tol` と `atol` の関係 — accidental 高精度" 小節を追加.
  "Chebyshev variant の dt 選択戦略 — Lanczos との対比" subsection を追加
  (PI controller スケルトンは両経路で共通だが, err 分解先と per-step 内部
  コストの dt 依存性が異なる点を表形式で対比).
- **`docs/quickstart.md` の主例** (#124): `method=` 指定を削除して default を
  使う形に統一. Chebyshev variant の atol upper bound 注を追記.
- **`bench_qutip_large.py --adaptive-tols` / `--krylov-tols` ヘルプ** (#124):
  両 adaptive 経路 (`_krylov` / `_chebyshev`) に対応する文言に更新. default
  solver list (`_VALID_SOLVERS` 全列挙) は変更なし (Pareto 比較目的なので
  両者走らせる).
- **`README.md`**: Chebyshev 法の説明を追加 (#cbfda0e). プロジェクト description
  を修正 (#6401f38).
- **`CLAUDE.md` テスト実行経路節 (#f494f5b)**: 「テスト・lint・maturin develop
  の実行は原則 test-runner subagent に委譲する」運用ルールを
  `.claude/solve-overrides.md` から CLAUDE.md 本体に格上げ (default 経路化).
  並列実行の方針 (`cargo test` BLAS on/off は target/ lock 競合でシリアル,
  `cargo test` BLAS on と `pytest` は独立で並列可, `maturin develop` と
  `pytest` は serialize 必須) を明記.
- **`docs/design/12-release-plan.md` / `docs/design/INDEX.md` / `CLAUDE.md`**:
  Phase B 完了 + follow-up (#124 / #126 / #127) 節を追加 / 「予定」文言を
  「実装済」に整理.

### Performance

- **`chebyshev_propagate` Gershgorin precompute (#c3f20c7)**: per-step
  `O(2^N + N)` → `O(1) (5 fp 演算)`. N=18 で wall ~1% 削減 (Lanczos 比 5.49×
  speedup の baseline 上に乗る微小最適化).
- **Chebyshev recurrence の SIMD + fusion (#126)** と **rayon 並列化 (#127)**:
  per-step wall 改善 / parallel efficiency 向上の 2 軸. 詳細 perf 値は
  `docs/design/12-release-plan.md` Phase B follow-up 節と該当 PR コメントの
  perf binary 計測 (`perf_chebyshev` / `perf_cfm4_richardson_chebyshev`) を参照.

公開 API シグネチャ変更は (1) パッケージ rename (kryanneal → kinema), (2)
default `method` の semantic 変更, (3) Chebyshev internal API への precompute
引数 3 個追加の 3 軸. 既存 test は `method=` を明示 + パッケージ rename は
全箇所一括置換済のため default 切替で壊れるテストなし.

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
