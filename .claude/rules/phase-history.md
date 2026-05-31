---
paths:
  - "src/**/*.rs"
  - "python/maqina/**/*.py"
---

# Phase 開発履歴と設計判断のアーカイブ

Rust カーネル / Python ドライバを触る前に該当 Phase の節を確認すること。
各 Phase の Definition of Done / bench acceptance / 未採用根拠を集約する。

## Phase 6 D 実験と未採用の根拠 (issue #79, 2026-05-17)

Phase D で `apply_h_kinema_rayon` を **連続 k 個の高 i を group-fused
3-phase 形** に書き換える試み (DRAM v traffic を `dim · (1 + h_baseline) →
dim · (1 + h_naive)` に削減する設計) を行ったが, **本 Linux サーバー
(AMD EPYC 7713P, 64 物理コア, L2 = 512 KB/core, L3 = 32 MB/CCX × 8) で
perf 計測した結果 N=20 で 50% 真の compute regression を確認** し,
revert. 詳細な perf 値と判断は `docs/design/05-1-matvec.md` §5.1.4 にアーカイブ.

要点だけ抜粋:

- C1 baseline は IPC=2.98 (Zen 3 理論 max の 60-75%) で **既に compute-near-peak**.
  「DRAM bandwidth bound だから traffic 削減すれば改善」前提が成立しなかった.
- Phase D の chunk 跨ぎ XOR access pattern が HW prefetcher を破壊し,
  per-L2-miss avg latency が 195 → 251 cycles (+30%) に劣化.
- cache-miss rate は baseline/after とも 3-7% で **DRAM access はそもそも
  少なかった**. 真の bottleneck は L2 fill latency (L3 / cross-CCX).
- N=18 は実質変化なし (Python bench で見えた 0.53× は alloc/GC noise).

issue #79 で B (SIMD i≥3), C (prefetch), D (streaming store) として残されていた
代替カードも **IPC 3.0 baseline 前提では効果薄** が予想されるため別途
sub-issue 化していない. 再挑戦時は `src/bin/perf_apply_h.rs` + perf stat で
ハードウェア counter を最初に取り「何 bound か」を確認してから設計に入る運用.


## Phase 7 (issue #93): Lanczos β_m exposure + Richardson 誤差源分離

Phase 6 完了後の follow-up. CFM4 adaptive Richardson driver が #65 long-T
シナリオで QuTiP に Pareto 劣位だった原因 (Richardson 推定子が Magnus 誤差と
Krylov 誤差を区別できない) を解消するための **infrastructure** を導入.

主要 API 変更:

- **`lanczos_propagate` (Rust + Python ref)**: return tuple が 4 要素
  `(psi, m_eff, β_m, |c_m|)` に拡張. 末尾 2 要素は Saad/Hochbruck-Lubich の
  a posteriori 誤差推定子 (`err_lanczos ≈ β_m · |c_m| · ‖ψ‖ · dt / m_eff`,
  5% 精度; `tools/verify_beta_m_estimator.py` で 108 cell sweep 実証).
- **`cfm4_step` / `cfm4_step_with_richardson_estimate`**: triangle
  inequality で `err_lanczos_sum` / `err_lanczos_total` を集約して上位伝播.
- **`evolve_schedule_adaptive_richardson`**: return tuple が 10 要素に拡張
  (`+ beta_m_history`, `err_lanczos_history`, `err_magnus_history`,
  `n_krylov_insufficient`). PI controller の駆動量を `err_magnus = max(0,
  err - err_lanczos_total)` に切替え.
- **`QuantumResult`**: `beta_m_stats` / `n_krylov_insufficient` フィールド追加.
- **`benchmarks/bench_qutip_large.py`**: `--propagator-tols` sweep で
  `atol × propagator_tol` のクロス評価を可能に (issue #135 で
  `--krylov-tols` から rename). `auto` キーワードで内部 default 解決
  (Lanczos = `tol_step * 1e-3` 連動, Chebyshev = `1e-12` 固定) を表現.

後方互換性: default `krylov_tol = 1e-12` では `err_lanczos << tol_step` で
`err_magnus ≈ err`. 既存 PI controller 挙動とほぼ等価
(`tests/test_adaptive.py::test_adaptive_richardson_error_decomposition_consistency`).

bench acceptance (Linux AMD EPYC 7713P, 2026-05-18):

- ✅ **Safety net 機能**: `bench_qutip_large.py --scenarios long-T
  --n-values 8,10 --propagator-tols auto,1e-8,1e-6` (issue #135 で
  `--krylov-tols` から rename) で `propagator_tol` を 4 桁緩めても
  `n_steps_eff` 差 0.01-0.02%, wall time 差 ±2%. PI controller が relaxed
  Krylov 設定下でも安定動作.
- ❌ **Pareto 劣位は未解消**: TFIM Lanczos の中間 β_j 値が O(‖H‖) で,
  `krylov_tol=1e-6` でも閾値を超えず m_eff=m_max=24 固定. Lanczos 圧縮そのもの
  が発火しないため Pareto は 2.5-8× 劣位のまま. 真の bottleneck は Richardson
  の構造的 6 Lanczos call / step. Phase 7 は **そこに到達するための前提
  infrastructure** として完了, Pareto win は follow-up に移管.

Follow-up issues:

- **#98 (Phase 8 で消化)**: Lanczos a posteriori 早期打切. Phase 7 で expose した
  推定子を Lanczos 内部の打切判定そのものに使う (下記 Phase 8 節).
- **#97**: Richardson 構造的 overhead 削減 (embedded estimator / time-reuse /
  adaptive frequency)

詳細は `docs/design/12-release-plan.md` Phase 7 / `docs/design/05-3-propagator.md`
"Richardson 誤差源分離" 節.

## Phase 8 (issue #98): Lanczos a posteriori 早期打切 (`krylov_tol` 意味再定義)

Phase 7 で expose した `β · |c|` a posteriori 推定子を **Lanczos 内部の早期打切
判定そのもの** に組み込み, Phase 7 で "infrastructure 完了 / Pareto 未解消"
だった #65 / #94 の本丸 (= Lanczos 圧縮を実際に発火させる) に踏み込む.

### 判定式と意味再定義

| 量 | Phase 7 まで | Phase 8 (現在) |
|---|---|---|
| `krylov_tol` の意味 | β 単体閾値 | **Krylov 近似の許容誤差** |
| Lanczos 早期打切判定 | `β_k < krylov_tol` (実用で発火しない) | `β_k · \|c_last\| · \|dt\| / (k+1) < krylov_tol` (Hochbruck-Lubich 1997) |
| β 単体の役割 | 打切判定 | numerical breakdown safety (`< 1e-14` で `v_{k+1} = w / β_k` の division by zero 回避のみ) |

`‖ψ‖ = 1` は Lanczos 内部の正規化空間規約 (`v_0 = ψ / ‖ψ‖`). 物理状態ノルム
は最終 `ψ_new = ‖ψ‖ · V · c` で復元するので, 判定式から `‖ψ‖` ファクタは
除外できる.

### API 互換性 / セマンティクス変更

公開 API シグネチャは **不変**:
- `QuantumAnnealer(krylov_tol=None)` / `AnnealingSimulator(krylov_tol=None)`
  の auto-resolve ロジック (adaptive: `tol_step · 1e-3`, fixed-dt: `1e-12`) も
  そのまま継承.
- 同じ default 値 (1e-11 / 1e-12) を渡しても **挙動が変わる** (旧: m_eff = m_max
  固定, 新: m_eff ≪ m_max になる scenario が増える). 数値結果 (`ψ_new`) は
  誤差内で一致するが `m_eff_history` 系統計値は変動.

このセマンティクス変更を伴うので **minor bump (`0.7 → 0.8`)**.

### Lanczos 内部 c の規約変更

内部 c 配列は `psi_norm` 抜きで保持し, 終端で `ψ_new = ‖ψ‖ · V · c` の gemv
coeff に畳み込む形にリファクタ. これにより判定式 `β · |c| · |dt| / m <
krylov_tol` が `‖ψ‖ = 1` 規約と意味的に整合し, `c_m_abs` (return 値の `|c_m|`)
も自然に "pure な行列要素" (`‖ψ‖` 抜き = literature 標準) で返せる.

### 詳細

- `docs/design/05-2-lanczos.md` "a posteriori 早期打切 (issue #98 Phase 8)" 節
  (旧仕様の問題 / 判定式 / overhead 試算).
- `docs/design/12-release-plan.md` Phase 8 (Definition of Done / Bench acceptance).
- `src/krylov.rs::tridiag_c_last_abs` / `python/maqina/krylov.py::_tridiag_c_last_abs`
  (per-iter ヘルパ; Rust ↔ Python ref `rel < 1e-13` 一致).

## Phase 8 follow-up (issue #100): Richardson iter-0 matvec memoization

Phase 8 で per-Lanczos call の m_eff 圧縮が達成された後の小規模直交最適化.
`cfm4_step_with_richardson_estimate` の **full_step stage 1** と **half_1
stage 1** は同じ入口 ψ から始まるため iter 0 で使う primitive matvec
(`H_drv · ψ` / `H_p_diag · ψ`) が共通. これを入口で 1 度だけ計算し両 Lanczos
call で再利用することで **2 個の primitive matvec / Richardson step** を削減
(削減量見積もり ~3% 純減; bench acceptance は「速くなれば accept」).

実装ポイント:

- `src/matvec.rs::apply_h_drv` / `apply_h_p_diag`: cache 計算専用 primitive.
  既存 `apply_h_kinema` の cache-blocked 形は **維持** (hot path 触らない).
  primitive は Richardson 入口で 1 step 1 回のみ呼ばれるので SIMD 非適用,
  rayon は MIN_RAYON_DIM 閾値で本体と同じ dispatch.
- `src/cfm4.rs::cfm4_step` のシグネチャに crate-internal `iter0_cache:
  Option<(&[Complex64], &[Complex64])>` 引数を追加. Lanczos に渡す matvec
  closure 内で `first_call` フラグを持たせ iter 0 のときだけ cache 線形結合
  `y = (c_drv_1 · cache_drv + c_diag_1 · cache_diag) / ‖ψ‖` に差し替える.
  Lanczos API (`lanczos_propagate`) 自体は不変.
- Public Python API のシグネチャは不変. crate-internal の `cfm4_step` 引数追加
  のみで, Python wrap (`cfm4_step_py`) は `iter0_cache = None` を渡して従来通り.

数値同等性: cache あり/なしで `rel < 2e-15` (machine epsilon の数倍).
詳細は `docs/design/05-1-matvec.md` §5.1.1.x / `docs/design/05-3-propagator.md`
"iter-0 primitive matvec memoization" / `docs/design/12-release-plan.md`
Phase 8 follow-up.

## Phase B (issue #122): Chebyshev propagator を CFM4 adaptive Richardson 経路に統合

Phase A (issue #120, PR #121) で時間独立 H 単体での `chebyshev_propagate` が
per-call 29 ms / 4.45× Lanczos 高速 (Linux AMD EPYC 7713P) を達成したことを
受け, 時間依存 H + CFM4 Magnus + step-doubling Richardson + PI controller 経路
に統合した variant を公開 Python API レベルで露出する.

### 公開 API 変更 (hard rename + 新 method 追加)

- 既存 `method="cfm4_adaptive_richardson"` を
  **`method="cfm4_adaptive_richardson_krylov"`** に hard rename
  (alias なし; pre-1.0 なので破壊的変更 OK). `_krylov` / `_chebyshev` で
  suffix 対称化.
- 新 `method="cfm4_adaptive_richardson_chebyshev"` を追加. Chebyshev 経路は
  Rust 拡張必須 (Python ref fallback 非提供).
- Rust 関数名は rename せず (`cfm4_step` / `cfm4_step_with_richardson_estimate`
  が Lanczos default, Chebyshev variant は `_chebyshev` suffix で対称).
- `m_max` を Chebyshev method で渡すと `ValueError` (Krylov 部分空間次元の
  概念なし; K_used は `chebyshev_tol` から動的決定).

### 主要実装ポイント

- `src/cfm4.rs::cfm4_step_chebyshev` / `cfm4_step_chebyshev_with_richardson_estimate`:
  既存 Lanczos 版と完全同じ 2 stage + step-doubling Richardson 構造を保ち,
  短時間プロパゲータだけが `chebyshev_propagate` に入れ替わる. per-stage で
  Gershgorin による `(E_c, R)` 再計算 (closed-form O(N), wall % 無視可).
- `evolve_schedule_adaptive_richardson_chebyshev` (Python driver): 既存
  Lanczos driver と同じ 10-tuple shape / PI controller 構造. `err_magnus =
  max(0, err - err_chebyshev_total)` で Magnus 起因駆動量を分離.
- `QuantumResult` の K_used 統計は既存 `m_eff_stats` スロットを流用
  (semantically 「per-step propagator 評価コスト統計」で同じ役割; method
  literal で Lanczos / Chebyshev を判別).
- iter-0 cache (Lanczos #100 の流用) は scope 外 (per-stage K_used ~20 個の
  matvec のうち 1 個と削減比小).

### 詳細

- `docs/design/05-3-propagator.md` "CFM4:2 + Chebyshev variant" 節
  (アルゴリズム / メモリ / cache 戦略).
- `docs/design/12-release-plan.md` Phase B (Definition of Done / 判定 gate).
- `tests/test_chebyshev.py` (QuTiP fidelity + Lanczos 一致 + annealer/simulator
  smoke + m_max ValueError).
- `tests/test_blas_consistency.py` 末尾 (Chebyshev direct call artifact dump;
  adaptive driver の dt 履歴分岐を避けるため Rust step 関数を fixed schedule
  係数で直接呼ぶ).

## Phase B follow-up (issue #126): Chebyshev 3 項漸化 inner loop の SIMD + fusion

Phase B 完了直後の直交最適化. `chebyshev_propagate` の k ≥ 2 hot loop は旧実装
で 3 つの dim-walk (walk 1: matvec, walk 2: recurrence scaling scalar, walk 3:
accumulate scalar) を発生させていたが, walk 2 / walk 3 を **1 dim-walk +
`wide::f64x4` SIMD** に fuse する.

- `src/chebyshev.rs::simd_kernels::chebyshev_recurrence_fused` (SIMD) /
  `chebyshev_recurrence_fused_scalar` (scalar fallback) + dispatch wrapper.
  `chebyshev_propagate` の k ≥ 2 hot loop だけ差し替え, k = 1 step は one-shot で
  scalar のまま (overhead 無視可).
- `cfm4_step_chebyshev_*` 経由でも自動で乗る (同じ `chebyshev_propagate` を
  呼ぶため).
- f64x4 helpers (`as_f64_slice` / `load/store_f64x4_unaligned` / `swap_reim`)
  は localize duplication で chebyshev module 内に持つ (`matvec.rs::simd_kernels`
  と同じパターンを再実装; visibility 経路を跨いだ変更を避ける).
- 数値同等性: `simd_kernels::chebyshev_recurrence_fused` ↔ `_scalar` の
  100-iter fuzz テスト (`chebyshev_recurrence_fused_simd_matches_scalar`,
  `rel < 1e-13`). FMA 折りたたみと lane 演算順序差で ulp 差は出るが ≤ 1e-13.
- bench acceptance (Linux AMD EPYC 7713P, NT=64): per-step wall 10%+ で full
  merge / 5-10% で marginal accept / < 5% で 中止. 計測は `perf_chebyshev 18 100`
  + `perf_cfm4_richardson_chebyshev 18 100 full` の 2 軸.
- 詳細: `docs/design/05-3-propagator.md` "Chebyshev recurrence の SIMD + fusion" /
  `docs/design/12-release-plan.md` "Phase B follow-up: Chebyshev 3 項漸化 inner
  loop の SIMD + fusion (#126)".

## Phase B follow-up (issue #127): Chebyshev non-matvec inner loop の rayon 並列化

#126 の SIMD + fusion 完了後の直交最適化. #124 perf archive で **Chebyshev の
parallel efficiency が 64 thread で 44%** (Lanczos 27% より良いが理想 100% には
程遠い) と判明. `apply_h_kinema` は #62 で rayon 並列化済だが,
`chebyshev_recurrence_fused` (k_ord ≥ 2 hot loop) が **scalar single-thread** で
走っており, ここがスケーリング bottleneck の一部.

- 実装: `src/chebyshev.rs::chebyshev_recurrence_fused_rayon` (rayon path).
  `scratch` / `psi_acc` の 2 RW slice を `par_chunks_mut` 2 本独立に取って
  `zip()`, `enumerate()` で base offset から `phi_curr` / `phi_prev` (R) を共有
  sub-slice 切り出し. chunk 内で `simd_kernels::chebyshev_recurrence_fused`
  (SIMD ON) または `chebyshev_recurrence_fused_scalar` (SIMD OFF) を呼ぶ 2 段構造.
- `chebyshev_recurrence_fused` dispatch wrapper を 3 段に拡張: rayon ON +
  `dim >= MIN_RAYON_DIM_CHEB` → rayon path / simd ON + 偶数長 → single-thread
  SIMD / それ以外 → scalar fused.
- chunk_size は matvec.rs の `apply_h_kinema_rayon` と同じ式
  `(dim / (nth * 4)).clamp(RAYON_CHUNK_MIN_CHEB, RAYON_CHUNK_MAX_CHEB)`. SIMD
  kernel の偶数長前提を満たすため 2 倍数に丸める (min/max 共 2 倍数なので
  invariant 不変).
- dispatch 閾値 `MIN_RAYON_DIM_CHEB` 初期値は `matvec.rs::MIN_RAYON_DIM = 1 << 17`
  と揃える. Chebyshev non-matvec hot loop は matvec より per-element cost が
  小さい (memory bound) ため本来はより低い閾値でも改善が出る可能性があるが,
  PoC 段階では保守寄りで始め, 本番 bench (N ∈ {14, 16, 18, 20} sweep) で tuning.
- `cfm4_step_chebyshev_*` 経由でも自動で乗る (同じ `chebyshev_propagate` を
  呼ぶため).
- 数値同等性: rayon path と single-thread SIMD/scalar fused の random fuzz
  10-iter テスト (`chebyshev_recurrence_fused_rayon_matches_serial`,
  `rel < 1e-13`). N=17 end-to-end の rayon path 経由 unitarity smoke
  (`chebyshev_propagate_rayon_path_smoke`).
- bench acceptance (Linux 本番サーバー, perf binary 計測; **本番計算環境とは
  別マシンで CPU 性能は本番より低い**): N=18 で per-step wall 10%+ 改善 + N=12
  (or N=14) で 5% 未満劣化 → full merge / N=18 改善 5-10% + dim 小劣化 5-15% →
  `MIN_RAYON_DIM_CHEB` を上げる方向で閾値 tuning 継続 / N=18 改善 5% 未満 →
  中止 + archive. 計測は `perf_chebyshev N 100` と
  `perf_cfm4_richardson_chebyshev N 100 full` を N ∈ {14, 16, 18, 20} ×
  RAYON_NUM_THREADS ∈ {1, 8, 16, 32, 64} で sweep.
- 詳細: `docs/design/05-3-propagator.md` "Chebyshev non-matvec inner loop の
  rayon 並列化" / `docs/design/12-release-plan.md` "Phase B follow-up:
  Chebyshev non-matvec inner loop の rayon 並列化 (#127)".

## Phase B follow-up (issue #124): Default method を Chebyshev variant に切替 + atol 仕様明文化

Phase B 本体 (#122) + #126 / #127 の perf 結果 (N=18 で Lanczos 比 5.49× wall
高速, branch-miss 158× 減, sys time 78× 減, parallel efficiency 27% → 44%) を
受けて, **judgement 系の follow-up** を確定. semantic 変更を伴うため
`0.10.0 → 0.11.0` で minor bump.

### Default method 切替

- `QuantumAnnealer.run(method=...)`: 旧 `"m2"` → `"cfm4_adaptive_richardson_chebyshev"`.
- `QuantumAnnealer.create_simulator(method=...)`: 旧 `"cfm4"` → 同上. ついでに
  `Literal` から欠落していた `_chebyshev` を追加 (Phase B #122 取りこぼし fixup).
- `AnnealingSimulator(method=...)`: 旧 `"cfm4"` → 同上.
- `docs/quickstart.md` の主例: `method=` 指定を削除 (default を使う形に統一).
- `bench_qutip_large.py --solvers` default は両 method を含む `_VALID_SOLVERS`
  全列挙のまま (Pareto 比較用なので両者走らせる方が有用). `_krylov` は literal
  として永続的に残す (旧 default 互換 + 比較ベンチ用途).

旧 default (`method="m2"` / `"cfm4"`) を使っていたユーザー向け migration: 新
default は **adaptive PI controller** を走らせるので `n_steps` の代わりに `atol`
で精度を制御する. 旧挙動を維持したい場合は `method="m2"` / `"cfm4"` を明示する.

### "Accidental 高精度" 仕様 (Chebyshev での atol の振舞い)

Chebyshev では `atol` (= PI controller の `tol_step`) は **upper bound** として
機能し, K_used 動的拡張により実際の精度がそれより良くなる場合がある (例:
`atol=1e-3` 設定で n=10 で `infidelity < 1e-16`). これは "feature" として
受け入れる方針 (issue #124 Scope 2 (a) + (d) 確定):

- `atol` で要求した精度を下回ることはない (予防的上限として機能).
- 速度を取りたいときは `atol` を大きくして PI step 数を減らすのが正しい使い方.
  `propagator_tol` (旧 `chebyshev_tol`, issue #135 で rename) を直接緩めても
  K_used が数個減るだけで wall-time 効果は限定的.
- default の auto-coupling 係数 `_KRYLOV_TOL_ATOL_RATIO = 1e-3` は Lanczos
  variant でのみ有効 (Chebyshev variant の default は issue #135 で固定
  1e-12 に変更, 下記 issue #135 節参照).

明文化先:

- `QuantumAnnealer.run` / `AnnealingSimulator.__init__` の `atol` docstring に
  "Note (Chebyshev variant の atol 振舞い, issue #124 / #135)" 注を追加.
- `docs/design/05-3-propagator.md` "Chebyshev variant" 節に "`propagator_tol`
  と `atol` の関係 — accidental 高精度 (issue #124 / #135)" 小節を追加.
- `docs/quickstart.md` の主例下に Note を追加.

### 詳細

- `docs/design/12-release-plan.md` "Phase B follow-up: Default method を
  Chebyshev variant に切替 + atol 仕様明文化 (#124)" 節 (Definition of Done /
  migration note).

## Phase B follow-up (issue #135): `krylov_tol` → `propagator_tol` rename + Chebyshev default 仕様変更

PR #134 (README figure pipeline) で実測された Chebyshev variant の
**atol-vs-infidelity 非単調性** (`atol=1e-4` で machine precision floor 到達後,
`atol=1e-5` で逆に infidelity が悪化) を解消するための 2 軸 follow-up.
API 破壊変更を含むため `0.11.0 → 0.12.0` minor bump.

### parameter rename (semantic 統一)

- `QuantumAnnealer(..., krylov_tol=...)` / `AnnealingSimulator(..., krylov_tol=...)`
  を `propagator_tol=...` に hard rename (deprecation alias なし; 旧 kwarg は
  `TypeError`).
- attribute (`self.krylov_tol` → `self.propagator_tol`), internal helper
  (`_resolved_krylov_tol_*` → `_resolved_propagator_tol_*`),
  `_krylov_tol_user` → `_propagator_tol_user` も統一.
- 命名意図: 「短時間プロパゲータ U(dt) の per-step 許容誤差」を表す
  semantically 中立な名前. Chebyshev には Krylov 部分空間概念が無いので
  旧名 `krylov_tol` は misleading だった. design docs
  (`docs/design/05-3-propagator.md`) の章名と一致.
- **Out of scope**: Rust 側 (`src/krylov.rs::lanczos_propagate` の kwarg
  `krylov_tol` / `cfm4_step_chebyshev_*` の kwarg `chebyshev_tol`) は
  rename せず. 各 method 内部文脈で適切なため. driver function
  (`evolve_schedule_*`) の引数も同様で internal context は維持.
- 内部定数 `_KRYLOV_TOL_ATOL_RATIO` / `_KRYLOV_TOL_FIXED_DEFAULT` は
  private historical 名のまま据置.

### Chebyshev variant の default 値変更

- `cfm4_adaptive_richardson_chebyshev` 経路で `propagator_tol = None`
  (未指定) のとき, 旧 `tol_step · _KRYLOV_TOL_ATOL_RATIO` (auto-coupling)
  から **固定値 `_KRYLOV_TOL_FIXED_DEFAULT` (= 1e-12)** に変更.
- Lanczos variant (`cfm4_adaptive_richardson_krylov`) は auto-coupling 維持
  (Lanczos a posteriori 早期打切は atol scaling 連動が望ましいため).
- 理由: Chebyshev は `K_used ~ R·dt + log(1/propagator_tol)` の対数依存で
  auto-coupling の動機が弱く, atol↓ で K_used がほぼ変わらないまま
  PI step 数だけ増えて round-off accumulation が顕在化する. 固定 1e-12
  で atol-vs-infidelity の monotonicity を確保し Pareto curve の解釈性
  を上げる. K_used 増は non-stiff +16% / stiff +3.7% (R·dt 別の Bessel
  減衰見積もり) と限定的.

### bench / CLI rename

- `benchmarks/bench_qutip_large.py --krylov-tols` → `--propagator-tols`.
  parse 関数 `_parse_krylov_tol_list` → `_parse_propagator_tol_list`,
  internal `krylov_tols` → `propagator_tols`.
- `benchmarks/bench_readme_figure.py` に新 flag `--propagator-tol` 追加
  (default None → Chebyshev は 1e-12 固定).
  `scripts/run_bench_readme_chebyshev.sh` も `CHEBYSHEV_PROPAGATOR_TOL` shell
  変数で明示 pass.

### 詳細

- `docs/design/05-3-propagator.md` "Chebyshev variant" 節
  (`propagator_tol` semantic + default 1e-12 固定の理由).
- `tests/test_chebyshev.py::test_chebyshev_default_propagator_tol_is_fixed_1e_minus_12`
  (atol 2 桁振っても K_used 平均 20% 未満変動の acceptance).
- `tests/test_chebyshev.py::test_old_krylov_tol_kwarg_raises_typeerror`
  (旧 kwarg は TypeError の contract).
- `CHANGELOG.md` 0.12.0 entry.
