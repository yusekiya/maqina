# §5.3 プロパゲータ

以下 4 種のプロパゲータを提供する (M2 / CFM4:2 は Magnus 系、Trotter は
operator splitting 系)。Adaptive dt 経路は CFM4:2 系の embedded / Richardson
推定子から構成する (Trotter は固定 dt のみ; embedded estimator を持たない):

#### M2 (中点則 1 step) — `m2_midpoint_step`

```
U(t+dt, t) ≈ exp(-i dt · H(t + dt/2))
```

中点で H をフリーズして `lanczos_propagate` 1 回。LTE ~ O(dt^3)。

#### CFM4:2 (4 次 commutator-free Magnus) — `cfm4_step`

Alvermann-Fehske (2011) の 4 次 commutator-free Magnus:

```
U(t+dt, t) ≈ exp(-i dt · B_2) · exp(-i dt · B_1)
```

Gauss-Legendre 2 点求積ノード:

```
c_1 = 1/2 - √3/6 ≈ 0.21132486540518708
c_2 = 1/2 + √3/6 ≈ 0.78867513459481292
```

線形結合係数:

```
a_high = 1/4 + √3/6 ≈ 0.43301270189221935
a_low  = 1/4 - √3/6 ≈ 0.06698729810778066
a_high + a_low = 1/2
```

ステージごとに以下の Hamiltonian で Lanczos を 1 回ずつ呼ぶ:

```
H_1 = H(t + c_1·dt),    H_2 = H(t + c_2·dt)
B_1 = a_high · H_1 + a_low  · H_2
B_2 = a_low  · H_1 + a_high · H_2
```

maqina では `H(t) = A(s(t))·H_driver + B(s(t))·H_problem` の構造を持つ
ため、各 stage で必要なのは **driver / diag の前置係数 1 組ずつ** だけ:

```
stage 1 :  c_drv = a_high·A(s_1) + a_low·A(s_2)
           c_diag = a_high·B(s_1) + a_low·B(s_2)
stage 2 :  c_drv = a_low ·A(s_1) + a_high·A(s_2)
           c_diag = a_low ·B(s_1) + a_high·B(s_2)
```

ここで `s_i = s(t + c_i·dt)`。Rust 側 `apply_h_kinema` に
`(c_drv, c_diag)` のスカラー 2 つを渡せば 1 stage の matvec が組める
(線形結合係数を 1 つに畳み込む経路、§5.2 末尾参照)。Lanczos 2 回 / step、
LTE ~ O(dt^5)。

#### CFM4:2 + step-doubling Richardson — `cfm4_step_with_richardson_estimate`

CFM4:2 を full-step (dt) と half-step×2 (dt/2 + dt/2) で **同一入口 ψ**
から走らせ、

```
err = ‖ψ_full - ψ_h2‖ ≈ (1 - 1/16) · C_4 · dt^5
```

を CFM4:2 自身の LTE 推定値として返す。per-step matvec は **6m**
(full 2m + half×2 × 2m, Lanczos 呼出 6 回, 固定 dt CFM4:2 比 3×)。
M2 embedded 版より 2 オーダ高精度なので smooth schedule では許容 dt を
1〜2 桁伸ばせる。

オプション `extrapolate=True` で Richardson 外挿:
`ψ_acc = (16 · ψ_h2 - ψ_full) / 15` (実効 6 次精度)。

##### iter-0 primitive matvec memoization (Phase 8 follow-up / issue #100)

Richardson estimator の **full_step stage 1** と **half_1 stage 1** の 2 つの
Lanczos call は **同じ入口 ψ から始まる**。各 stage 1 Hamiltonian は

```
A_full1   = α_full1   · H_drv + β_full1   · H_p_diag      (中点 t + c1·dt)
A_half1_1 = α_half1_1 · H_drv + β_half1_1 · H_p_diag      (中点 t + c1·dt/2)
```

で **異なる** が, **iter 0 で使う primitive matvec `H_drv · ψ` と `H_p_diag · ψ`
は完全に同一**。これを `cfm4_step_with_richardson_estimate` の入口で 1 度だけ
計算 (`apply_h_drv` / `apply_h_p_diag`, §5.1.x primitive) して両 Lanczos call
に渡せば, 2 個の primitive matvec / Richardson step を削減できる。

実装:

- `src/matvec.rs::apply_h_drv` / `apply_h_p_diag`: cache 計算専用の primitive
  matvec 関数。既存 `apply_h_kinema` の cache-blocked 形 (diag + 全 i
  bit-flip pass を 1 chunk closure 内で完走) は維持し, 本 primitive は
  Richardson 入口で 1 step 1 回だけ呼ばれる。
- `src/cfm4.rs::cfm4_step` のシグネチャに `iter0_cache: Option<(&[Complex64],
  &[Complex64])>` 引数を追加 (crate-internal API)。Lanczos に渡す matvec
  closure 内で `first_call` フラグを持たせ, iter 0 のときだけ cache を線形結合:
  ```
  y = (c_drv_1 · cache_drv + c_diag_1 · cache_diag) / ‖ψ‖
  ```
  iter 1 以降は v_k が分岐するので従来通り `apply_h_kinema` 経路。
  Lanczos 内部 API は不変 (matvec closure を 1 個受けるだけ)。
- full_step stage 1 / half_1 stage 1 に `iter0_cache = Some(...)` を渡し,
  half_2 (入口は `psi_mid` で異なる) と stage 2 (入口は各 stage 1 出口で異なる)
  には `None`。

削減量見積もり: 2 primitive / 32 合成 matvec (`6 Lanczos × m_eff ≈ 5.33`,
Phase 8 後) = **~3-6%**。cache 計算自体が ~1 合成 matvec のコストを足すので
純削減は **~3% (1/32)** 程度。**bench acceptance は「速くなれば accept」**
(改善量問わず)。

数値同等性: cache 経路と非 cache 経路では演算順序が異なるため bit-identical
ではない。Lanczos m_eff ステージ全体で `rel < 2e-15` (machine epsilon の
数倍) で一致することを Rust 単体テスト
(`cfm4_step_iter0_cache_matches_no_cache_machine_eps` /
`cfm4_richardson_estimate_iter0_cache_matches_no_cache_chain`) で契約する。

#### CFM4:2 + Chebyshev variant — `cfm4_step_chebyshev` (Phase B / issue #122)

Phase A (#120, PR #121) で時間独立 H 単体で **per-call 29 ms / 4.45× Lanczos
高速** を実測した `chebyshev_propagate` 3 項漸化を CFM4:2 + step-doubling
Richardson + PI controller 経路に統合した variant。既存
`cfm4_step` (Lanczos) の **2 stage 構造を完全に保ったまま** 短時間
プロパゲータだけが入れ替わる:

```text
stage 1: B_1 = a_high · H_1 + a_low · H_2  (c_drv_1, c_diag_1 に畳み込み)
         → ψ_mid = chebyshev_propagate(h_x, h_p_diag, c_drv_1, c_diag_1,
                                       ψ, dt, chebyshev_tol)
stage 2: B_2 = a_low · H_1 + a_high · H_2
         → ψ_new = chebyshev_propagate(..., c_drv_2, c_diag_2, ψ_mid, dt, ...)
```

per-stage で **Gershgorin により `(E_c, R)` を再計算** する。`h_x_abs_sum
= Σ_i |h_x_i|` と `h_p_min / h_p_max = min/max(h_p_diag)` を `IsingProblem`
構築時に 1 度だけ precompute して上位 driver から渡し, per-step は
`gershgorin_bounds_cached` で **O(1) (数値演算 5 回)** で済ませる。素朴な
`gershgorin_bounds` (h_p_diag を毎回 full walk) だと per-step `O(2^N + N)`
で N=18 で wall time の 1% 弱を占めてしまうので, この precompute は明示的に
持たせる契約とする。Lanczos `m_eff` が `chebyshev_tol` から決まる K_used に
置き換わるだけで, 線形結合係数 / Richardson 構造 / PI controller の駆動量
(`err_magnus = max(0, err - err_chebyshev_total)`) は Lanczos 版と同型に保つ。

メモリ / cache:

- Lanczos の `V` 行列 (dim × m_max, N=18 で 96 MB → CCX L3 32 MB を超過 →
  cache spill) が **構造的に消滅**。Chebyshev 3 項漸化は 3 個の作業ベクトル
  (`φ_{k-1}, φ_k, scratch`) + accumulator のみ動作 (dim × 4 = 16 MB at N=18,
  L3 に収まる)。
- Gram-Schmidt 直交化も 3 項漸化が数学的に直交保証するため不要 (BLAS-1
  の `n²` 二次項が消える)。Phase A speedup の主因。

判定基準 (Phase B 完了 gate) は `12-release-plan.md` Phase B (#122) 参照。

**`propagator_tol` semantic (issue #135 で `krylov_tol` から rename)**:
Chebyshev 切り捨て次数 `K_used` を決める許容誤差。Public Python API では
`QuantumAnnealer.propagator_tol` をそのまま渡せる (両 method 共通軸; Lanczos
では Krylov 近似の許容誤差として機能, Chebyshev では K_used 切り捨て閾値として
機能, semantic は「短時間プロパゲータ U(dt) の per-step 許容誤差」で統一)。
Rust 側の Python wrap (`cfm4_step_chebyshev_*`) では internal kwarg 名
`chebyshev_tol` を維持 (method 内部文脈で適切なため)。

Lanczos 経路の iter-0 cache (#100) と同型の memoization も Chebyshev で原理的に
可能だが per-stage K_used ~ 20 個の matvec のうち 1 個と削減比小なので Phase B
スコープ外 (follow-up).

##### `propagator_tol` と `atol` の関係 — accidental 高精度 (issue #124 / #135)

PI controller の `atol` (= `tol_step`) と `propagator_tol` (= 短時間プロパゲータ
の内部精度) の関係は Lanczos 経路と **構造的に同型** だが, Chebyshev の K_used
動的拡張により「実際の精度が atol よりも遥かに良くなる」現象 (issue #124 Scope 2)
が起きやすい:

- default では `propagator_tol = _KRYLOV_TOL_FIXED_DEFAULT` (= **1e-12 固定**,
  issue #135 で旧 auto-coupling から変更)。Lanczos variant では引き続き
  `propagator_tol = tol_step · _KRYLOV_TOL_ATOL_RATIO` (auto-coupling) を維持。
- K_used は per-stage で 3 項漸化を進めながら部分和の収束を見て **動的に決まる**
  ため, `propagator_tol` を超えるまで自動的に拡張される。結果として
  per-stage の Chebyshev 切り捨て誤差は `tol_step` より遥かに小さくなる。
- PI controller が step doubling で計測する `err = ‖ψ_full - ψ_h2‖` は Magnus 4 次
  誤差 (`O(dt^5)`) と Chebyshev 切り捨て誤差の和だが, Chebyshev 側が無視可能に
  小さくなるため **Magnus 誤差のみが PI 駆動量となる**。
- 観測される現象 (bench_qutip_large の実測, n=10/12, `atol=1e-3` cell): `n_steps`
  は ~1250 step (`atol=1e-3` 相当の Magnus 誤差で許容される大きさ) でありながら,
  infidelity は machine precision (`< 1e-16`) に達する。

**Chebyshev default を固定 1e-12 にした理由 (issue #135)**: PR #134 (README
figure pipeline) で auto-coupling default 下に `atol=1e-4` (machine precision
floor 到達) より `atol=1e-5` で infidelity が悪化する非単調性を実測。
PI controller が小 dt を選んで n_steps が増え, per-step round-off accumulation
が顕在化する一方で per-step Chebyshev 打切は既に machine precision floor なので
`propagator_tol` の tightening は無意味, という構図。`propagator_tol` を atol
非依存の固定 1e-12 にすることで atol-vs-infidelity の monotonicity を担保し
Pareto curve の解釈性を上げる。K_used 増は non-stiff +16% / stiff +3.7% と
限定的 (Bessel 減衰の対数依存)。Lanczos variant は a posteriori 早期打切が
atol scaling と整合的に発火するため auto-coupling 維持。

**仕様としての解釈 (issue #124 Scope 2 (a) + (d) + #135, 確定)**: これは "bug"
ではなく "feature" として受け入れる:

- `atol` は **予防的な upper bound** として機能する (`atol` で要求した精度を
  下回ることはない)。
- 速度を取りたいときは `atol` を大きくして PI step 数を減らす運用が正しい。
  `propagator_tol` を直接緩めても per-step matvec 数 (K_used) が数個減るだけで
  Wall-time effect は限定的 (Bessel 減衰の対数依存)。
- internal 定数 `_KRYLOV_TOL_ATOL_RATIO` (= `1e-3`) は Lanczos variant 用の
  auto-coupling 係数で, Chebyshev では使われない (固定 1e-12)。historical 名
  だが rename はしない (private 内部定数で influence の sphere が狭い)。

Lanczos 経路でも同様の "accidental 高精度" は理論的に起こるが, Lanczos a posteriori
推定子 `β · |c_last| · |dt| / m_eff` が `chebyshev_tol`-相当の閾値を達成するまでに
m_eff を伸ばしきらないケースが多い (TFIM の β は ~‖H‖ で大きく, |c_last| が
急減しないと早期打切が発火しない) ため, default 設定下では Chebyshev ほど顕著では
ない。

##### dt 選択戦略 — Lanczos との対比

dt 選択ロジック自体 (CFM4:2 + step-doubling Richardson + PI controller, `p=4`)
は Lanczos / Chebyshev で共通 (本ファイル "PI controller / adaptive ドライバ"
節)。差分は **(1) `err` をどう分解するか** と **(2) per-step 内部コストが dt に
どう依存するか** の 2 点で, この 2 点が「dt をどこで頭打ちさせるか」と
「`atol` を緩めたときの高速化挙動」を決める。

**(1) 誤差源分離 (本ファイル "Richardson 誤差源分離" 節)**

PI controller の駆動量は両経路で `err_magnus = max(0, err - err_propagator_total)`。
ここで `err_propagator_total` は propagator 内部誤差を 6 stage ぶん triangle
inequality で集約した値:

| 経路 | `err_propagator` 推定子 | 由来 |
|---|---|---|
| Lanczos | `β_m · \|c_m\| · ‖ψ_in‖ · dt / m_eff` | Saad 1992 / Hochbruck-Lubich 1997 |
| Chebyshev | `2 · \|J_{K+1}(z)\| · ‖ψ_in‖` | Tal-Ezer & Kosloff 1984 (1-term residual) |

どちらも "dt 縮小で減らない誤差を PI 駆動量から外す" 目的は同じ。`err_chebyshev`
が `chebyshev_tol` まで動的に押し下げられる ("accidental 高精度", 前節) ため
**Chebyshev では実質 PI は Magnus 誤差のみを見る** ことが多い。Lanczos では
`err_lanczos` がそこまで小さくならず, `err_magnus + err_lanczos` の合成が PI
駆動量になる。

**(2) per-step 内部コストの dt 依存性**

per-step 内部コストは「short-time propagator 1 stage あたりの matvec 数」 ×
「stage 数 (CFM4:2 で 2 stage / step + Richardson で × 3 軌道)」で決まる:

| 経路 | per-stage matvec 数 | dt 依存性 |
|---|---|---|
| Lanczos | `m_eff` (`≤ m_max`, default 24) | **頭打ち** (大 dt でも m_eff ≤ m_max) |
| Chebyshev | `K_used` | **線形** (`K ≈ z + log(1/tol) = R · dt + log(1/tol)`) |

dt を 2 倍にしたときの per-step compute コスト変化:

| 経路 | dt → 2·dt の per-step matvec 比 | step 数 (T 固定で) | total matvec 比 |
|---|---|---|---|
| Lanczos | ~ 1 (m_max 固定) | 1/2 | ~ 1/2 (Magnus 制約内なら) |
| Chebyshev | ~ 2 (K ∝ z = R·dt) | 1/2 | **~ 1** (相殺) |

Chebyshev 側の "dt 倍にしても total matvec はほぼ変わらない" は, Bessel 漸近
$K \approx z + O(\log(1/\mathrm{tol}))$ から導かれる: 大 z 領域では K の dt 線形項が
支配, log(1/tol) の固定費は相対的に小さい。

**dt の上限を決めるもの**

- **Lanczos**: Magnus LTE `O(dt^5) ≤ tol_step` **+ Lanczos m_max での短時間
  プロパゲータ再現精度**。後者は dt が大きすぎると `err_lanczos` が膨張し
  Richardson 推定子経由で PI が dt を絞る (#93 "Krylov 不足" の n_krylov_insufficient
  カウンタ)
- **Chebyshev**: **Magnus LTE のみ**。K_used が無制限に伸びるため (実用 cap
  `k_max_cap = 5000`) propagator 内部側からは dt 上限制約がほぼ働かない

実装上の含意:

- **Chebyshev では dt は atol = Magnus LTE 許容値の 5 乗根に張り付く**。default
  `atol = 1e-8` で dt ≈ `(8 · tol_step)^{1/5} · 1/‖H‖^{1/5}` 相当の上限で
  PI が安定化
- Lanczos でも同じ Magnus LTE 上限は存在するが, m_max=24 の表現可能領域で
  追加制約がかかるため "Magnus LTE 上限よりさらに小さい dt" に頭打ちする
  ケースがある (大 N で `dt_max = 4m/‖H‖_est` で cap される; Phase 4 follow-up B)

**`atol` を緩めた効果**

ユーザーが速度寄りに振りたいとき (例: `atol = 1e-5`):

| 経路 | 帰結 |
|---|---|
| Lanczos | dt 増 → step 数減 → 加えて m_eff も自動で伸びる余地が出る (m_max 内で) → **二重高速化** |
| Chebyshev | dt 増 → step 数減 → **しかし** per-step K_used が増 (`K ∝ dt`) → **step 数減のみの高速化** |

つまり `atol` 緩和の効きは **Chebyshev のほうが穏やか**。同じ 1 桁緩和でも
Chebyshev は wall ~ 1/dt 比例だが, Lanczos は per-step matvec も削減方向に効く
可能性がある (TFIM 以外の問題で early termination が発火する場合)。

**実用判断**

- "atol を変えずに速くしたい" 用途 → `_KRYLOV_TOL_ATOL_RATIO = 1e-3` 連動の
  内部 tol を直接緩めても Chebyshev では K_used が数個減るだけで Wall 効果は
  限定的 (本ファイル "accidental 高精度" 節)。`atol` を緩めるのが正攻法
- "atol を緩めて速くしたい" 用途 → Chebyshev は step 数比例で素直に高速化。
  Lanczos は問題依存で early termination が効けば追加加速

##### Chebyshev recurrence の SIMD + fusion (issue #126, Phase B follow-up)

Phase B 完了直後の直交最適化。`chebyshev_propagate` の k ≥ 2 hot loop は
旧実装で 3 つの dim-walk を発生させていた:

```
walk 1: scratch := H · phi_curr                      (matvec, apply_h_kinema)
walk 2: scratch := 2·(scratch - E_c·phi_curr)/R - phi_prev  (scalar)
walk 3: psi_acc += c_k · scratch                     (scalar)
```

このうち walk 2 / walk 3 を **1 dim-walk + `wide::f64x4` SIMD** に fuse する:

```rust
chebyshev_recurrence_fused(
    &mut scratch, &phi_curr, &phi_prev, &mut psi_acc, e_c, inv_r, c_k,
)
// 1 ループ内で:
//   tilde = (scratch - e_c · phi_curr) · inv_r
//   scratch <- 2 · tilde - phi_prev
//   psi_acc += c_k · scratch
```

各 lane (f64x4 = 2 Complex64) で実 scalar (e_c, inv_r) は splat × 普通の積で
処理し, complex scalar `c_k` は `single_mode_iN` (issue #71) と同じ
**broadcast + swap** pattern (`c_k · x = c_re_v · x + c_im_signed_v · swap(x)`)
で計算する。

実装:

- `src/chebyshev.rs::simd_kernels::chebyshev_recurrence_fused`
  (`#[cfg(feature = "simd")]`, scalar tail なし: dim = 2^N で常に偶数長保証).
- `chebyshev_recurrence_fused_scalar` (`--no-default-features` ビルド / 退化
  ケース用フォールバック).
- `chebyshev_recurrence_fused` dispatch wrapper (上記 2 経路を SIMD feature と
  長さで切り分け).
- `chebyshev_propagate` の k ≥ 2 hot loop だけ差し替え. k = 1 step は one-shot
  なので scalar のまま (per-call 1 回, overhead 無視可).
- f64x4 helpers (`as_f64_slice` / `load/store_f64x4_unaligned` / `swap_reim`)
  は localize duplication で chebyshev module 内に持つ. `matvec.rs::simd_kernels`
  と同じ実装パターン (visibility 経路を跨いだ変更を避ける).

数値同等性: `simd_kernels::chebyshev_recurrence_fused` ↔ `_scalar` の random
fuzz 100-iter テスト (`chebyshev_recurrence_fused_simd_matches_scalar`,
`rel < 1e-13`). FMA 折りたたみと lane 演算順序差で ulp 差は出るが ≤ 1e-13.
`cfm4_step_chebyshev_*_py` 経由は既存 `test_blas_consistency.py` の Chebyshev
artifact dump (`rel < 1e-13`) で end-to-end カバー.

bench acceptance (Linux AMD EPYC 7713P, NT=64): per-step wall 10%+ で full
merge / 5-10% で marginal accept / < 5% で 中止. 詳細は `12-release-plan.md`
"Phase B follow-up: Chebyshev 3 項漸化 inner loop の SIMD + fusion (#126)".

##### Chebyshev non-matvec inner loop の rayon 並列化 (issue #127, Phase B follow-up)

#126 の SIMD + fusion 完了後の直交最適化。`chebyshev_recurrence_fused` は
single-thread で走っていたため, #124 perf archive で **Chebyshev の parallel
efficiency が 64 thread で 44% に留まる** (Lanczos の 27% より良いが理想 100%
には程遠い) ことが判明した。`apply_h_kinema` は #62 で rayon 並列化済だが,
Chebyshev 固有の non-matvec hot loop (recurrence scaling + accumulator) が
serial bottleneck になっている。

これを `rayon::par_chunks_mut` で並列化する 2 段構造に拡張する:

```text
外側: scratch.par_chunks_mut(chunk_size).zip(psi_acc.par_chunks_mut(chunk_size))
内側: 各 chunk 内で simd_kernels::chebyshev_recurrence_fused (SIMD ON) または
      chebyshev_recurrence_fused_scalar (SIMD OFF) を呼ぶ
```

chunk_size は `matvec.rs::apply_h_kinema_rayon` と同じ式
`(dim / (nth * 4)).clamp(RAYON_CHUNK_MIN_CHEB, RAYON_CHUNK_MAX_CHEB)` で動的
決定 (定数は matvec 側と同値 `1 << 6` / `1 << 14` を chebyshev module 内に
localize)。SIMD kernel の偶数長前提を満たすため chunk_size を 2 倍数に丸める
(min/max 共に 2 の倍数なので invariant 不変)。

実装:

- `src/chebyshev.rs::chebyshev_recurrence_fused_rayon`
  (`#[cfg(feature = "rayon")]`).
- `chebyshev_recurrence_fused` dispatch wrapper を 3 段に拡張:
  1. rayon ON + `dim >= MIN_RAYON_DIM_CHEB` → rayon path (内側 SIMD or scalar).
  2. simd ON + `n >= 2 && n % 2 == 0` → single-thread SIMD kernel.
  3. それ以外 → scalar fused (`--no-default-features` / 退化ケース).
- 4 slice の borrow: `scratch` (RW), `psi_acc` (RW) は呼出側 disjoint な
  `Vec<Complex64>` 由来。 `par_chunks_mut` を 2 本独立に取って `zip` し,
  `enumerate` で base offset から `phi_curr` / `phi_prev` (R) を共有 sub-slice
  で切り出す。

dispatch 閾値 `MIN_RAYON_DIM_CHEB` 初期値は `matvec.rs::MIN_RAYON_DIM = 1 << 17`
と揃える。Chebyshev non-matvec hot loop は matvec より per-element cost が
小さい (memory bound) ため本来はより低い閾値でも改善が出る可能性があるが,
PoC 段階では保守寄りで始め, Linux 本番 bench (N ∈ {14, 16, 18, 20} sweep)
の結果次第で tuning する方針。

数値同等性: rayon path と single-thread SIMD/scalar fused の random fuzz
10-iter テスト (`chebyshev_recurrence_fused_rayon_matches_serial`,
`rel < 1e-13`). 各 chunk が独立に同じ kernel を呼ぶため理論上は bit-identical,
chunk 境界が `scratch[k]` / `psi_acc[k]` の単一値生成位置に影響しない設計。

bench acceptance (Linux AMD EPYC 7713P, perf binary 計測): N=18 で per-step
wall 10%+ 改善 + N=12 で 5% 未満劣化 → full merge / N=18 改善 5-10% + N=12
劣化 5-15% → 閾値 tuning 継続 / N=18 改善 5% 未満 → 中止 + archive。dim 小
劣化が大きい場合は `MIN_RAYON_DIM_CHEB` を上げる方向で safety net を張る。
詳細は `12-release-plan.md` "Phase B follow-up: Chebyshev non-matvec inner
loop の rayon 並列化 (#127)".

#### Trotter (Strang 2 次 / Suzuki 4 次) — `trotter_step`

横磁場 driver の `[X_i, X_j] = 0` を活用し、`exp(-i dt H_drv)` を
`Π_i R_i(dt)` の閉形式 (Lanczos 不要) で書く operator splitting 経路。
Strang 2 次:

```
U(dt) ≈ exp(-i dt H_p / 2) · exp(-i dt H_drv) · exp(-i dt H_p / 2)
      = phase_p(dt/2) · (Π_i R_i(dt)) · phase_p(dt/2)
```

各 `R_i(dt) = cos(A·h_x_i·dt)·I + i·sin(A·h_x_i·dt)·X_i` は §5.1.2 の
`apply_single_mode_axis_i` で 1 軸 in-place 適用。`H_drv = -Σ h_x_i X_i`
の負符号は `exp(-i·dt·H_drv) = Π_i exp(+i·a·h_x_i·dt·X_i)` で打ち消されて
`R_i` の `θ = +a·h_x_i·dt` に乗る (`apply_h_kinema` の
`coeff = -a_t·h_x_i` と同 convention)。

per-step コスト: `(N + 1) · dim` 要素アクセス (matvec の 1 pass 相当が
N+1 回)。CFM4:2 の `2m·dim` (m=24 で ~48·dim) と比較すると N=20 で
~2.3× 軽量だが、LTE は O(dt^3) なので精度要求次第で総時間の優劣は変わる
(クロスオーバ実測は Phase 2 でベンチに含める、§12)。

API:

```rust
pub fn trotter_step(
    psi: &mut [Complex64],
    h_x: &[f64],
    h_p_diag: &[f64],
    a_t: f64,            // 中点 schedule 値 A(s(t + dt/2))
    b_t: f64,            // 同 B(s(t + dt/2))
    dt: f64,
    n: usize,
);
```

`(a_t, b_t)` は schedule の **中点で評価** することで Strang 2 次の対称性を
保つ (時間依存 H の Strang は中点採取で局所 O(dt^3) を維持する)。
固定 dt ドライバから直接呼ぶ。

**4 次 Suzuki (`trotter_suzuki4_step`)**: Trotter-Suzuki S_4 公式

```
S_4(dt) = S_2(p·dt) · S_2(p·dt) · S_2((1-4p)·dt) · S_2(p·dt) · S_2(p·dt)
p = 1 / (4 - 4^{1/3}) ≈ 0.4145
```

で Strang 5 回適用に分解。per-step は ~5·(N+1)·dim、LTE O(dt^5)。CFM4:2 と
同じ局所オーダだが、Lanczos の m 回 matvec を完全に排した経路としての
比較・検証用に Phase 2 末で追加 (`method="trotter_suzuki4"`)。

中央 sub-step は `1 - 4p ≈ -0.658` で **時間逆向き** に走る (Suzuki の
高次合成では正の係数しか持つ対称合成が存在しないことの帰結)。`trotter_step`
は `dt < 0` を許容するので呼出側で特別扱い不要。

API:

```rust
pub fn trotter_suzuki4_step(
    psi: &mut [Complex64],
    h_x: &[f64],
    h_p_diag: &[f64],
    a_t_list: &[f64],    // length 5: 各 sub-step の A(s(中点))
    b_t_list: &[f64],    // length 5: 各 sub-step の B(s(中点))
    dt: f64,             // 外側 1 step の時間刻み
    n: usize,
);
```

**サブステップ係数・中点 offset**: sub-step 幅 `[p, p, 1-4p, p, p]·dt` を
時間順に並べたとき, sub-step `k` の **中点 offset** (`(start_k + end_k)/2`)
は

```
offsets = [p/2, 3p/2, 1/2, 1 - 3p/2, 1 - p/2]
```

で `t + dt/2` を中心に対称。時間依存 H に対しては各 sub-step の中点で
`(A(s(·)), B(s(·)))` をフリーズ採取することで全体の LTE `O(dt^5)` を保つ。

**schedule 評価の責務**: ホスト言語 (Python driver
`evolve_schedule_trotter_suzuki4`) が中点 offset を内部で持ち, 各 step
ごとに 5 つの `(a_mid, b_mid)` を事前計算して長さ 5 の配列として Rust 側に
渡す。Rust 拡張は `schedule` callable を持ち込まず, Strang `trotter_step`
を `a_t_list[k] / b_t_list[k] / coeffs[k] * dt` で 5 回呼ぶだけのループ。
Strang 経路 API (`trotter_step(psi, ..., a_t, b_t, dt, n)`) と「呼出側で
schedule 評価 → Rust に純粋な係数を渡す」契約を統一する。

**embedded error estimator は持たない**: Strang↔Suzuki4 の差を embedded
推定子に使う案も理論上はあるが、Phase 4 の `cfm4_step_with_*_estimate`
体系と統合せず、Trotter 経路は **固定 dt 専用** とする。adaptive 経路は
CFM4:2 を使う。

#### PI controller / adaptive ドライバ

両 estimator を共通仕様の PI controller (Python 側ループ) が駆動する。
RK45 / Dormand-Prince 系の embedded estimator と同型の式・既定値を採用:

```
dt_next = dt · safety · (tol_step / err)^(1/(p+1))
```

| 推定子 | p | 指数 `1/(p+1)` | 該当 estimator |
|---|---|---|---|
| M2 embedded | 2 | 1/3 | `cfm4_step_with_m2_estimate` |
| Richardson  | 4 | 1/5 | `cfm4_step_with_richardson_estimate` |

既定パラメータ (`evolve_schedule_adaptive_*` の `*` パラメータ):

```
m            = 24            # Lanczos 部分空間次元
krylov_tol   = 1e-12         # Lanczos 早期打切閾値 (β_k < tol)
tol_step     = 1e-8          # accept 判定の局所誤差閾値
dt0          = 0.5           # 初期 dt
dt_min       = 1e-4          # 最小 dt (ここまで縮めると err 無視で accept)
dt_max       = 10 · dt0      # 既定値 (None 渡し時に解決)
safety       = 0.9           # PI 安全係数
growth_max   = 4.0           # 1 step での dt 拡大率上限
max_rejects  = 50            # 同一 step での連続 reject 上限 (超過で RuntimeError)
```

> **`tol_step = 1e-8` の選定根拠 (保守寄り)**: 1 step あたりの局所誤差
> ``1e-8`` を区間 ``[0, T]`` で蓄積すると, worst case (Lady Windermere's
> fan) では `N_step · 1e-8 ~ 1e-5` 程度, ランダムウォーク的合成では
> `√N_step · 1e-8 ~ 1e-7` 程度の終端誤差になる. 標準的な QuTiP 比較
> テストの fidelity 要件 `1 - 1e-6` を **どんな問題サイズでも安全
> マージン付きで満たす** よう保守寄りに選定した値.
> 実用上は `tol_step ∈ [1e-6, 1e-5]` も多くの応用で十分で, user が
> 速度を取りたい場合は facade の `atol` 引数を緩めることで opt-in
> できる (PI step 数が減り, `krylov_tol = None` ならば §5.3 follow-up
> E の `_KRYLOV_TOL_ATOL_RATIO` 連動で Lanczos 早期打切も自動的に
> 緩む). default の "未指定で robust" 性質を維持しつつ, user が
> 段階的に速度寄りに振れるよう設計してある.

ループ本体 (擬似コード):

```python
while t < t1:
    dt_try = min(dt, t1 - t, next_save - t)   # 終端 / 観測時刻にクランプ
    psi_new, err = step_with_estimate(psi, dt_try, ...)
    accept = (err <= tol_step) or (dt_try <= dt_min)
    if accept:
        psi = psi_new
        t += dt_try
        if err <= 1e-30:                       # 0 近傍ガード
            dt_next = dt_try * growth_max
        else:
            dt_next = dt_try * safety * (tol_step / err) ** (1/(p+1))
        dt_next = min(dt_next, dt_try * growth_max, dt_max)
        dt_next = max(dt_next, dt_min)
        dt = dt_next
        n_consecutive_rejects = 0
    else:
        n_consecutive_rejects += 1
        if n_consecutive_rejects > max_rejects:
            raise RuntimeError(...)
        dt = max(dt_try * 0.5, dt_min)         # reject 時は半減
```

Reject 時に schedule node `t + c_i · dt` は新しい dt で再評価する
(dt 依存ノードのため)。`save_tlist` が指定された場合は次の観測時刻
`t_obs` でも step 境界が揃うよう `dt_try` をクランプし、accept 後の `ψ`
を `states_at_save` に `ψ.copy()` で格納する。PI 状態 (`dt_next`) は観測境界
を跨いで連続持ち越しなので chain 呼出より再ウォームアップコストが少ない。

Python 側公開関数:

```python
# python/maqina/krylov.py

def evolve_schedule_adaptive_m2(
    problem, schedule, psi0, t0, t1, *,
    m=24, krylov_tol=1e-12, tol_step=1e-8,
    dt0=0.5, dt_min=1e-4, dt_max=None,
    safety=0.9, growth_max=4.0, max_rejects=50,
    save_tlist=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, list[np.ndarray]]:
    """(psi_final, t_history, dt_history, n_rejects, states_at_save)"""

def evolve_schedule_adaptive_richardson(
    problem, schedule, psi0, t0, t1, *,
    m=24, krylov_tol=1e-12, tol_step=1e-8,
    dt0=0.5, dt_min=1e-4, dt_max=None,
    safety=0.9, growth_max=4.0, max_rejects=50,
    richardson_extrapolate=False,
    save_tlist=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, list[np.ndarray]]: ...
```

`QuantumAnnealer.run(method="cfm4_adaptive_richardson_krylov", ...)` はこれを
内部で呼ぶ薄いラッパ。

#### Richardson 誤差源分離 (Phase 7 / issue #93)

Phase 6 C4 (issue #65) の `bench_qutip_large.py` long-T シナリオで CFM4
adaptive Richardson が QuTiP に Pareto 劣位だった原因として, Richardson
推定子の `err = ‖ψ_full - ψ_h2‖` が **dt 起因の Magnus 誤差** と
**Lanczos 部分空間有限性に起因の Krylov 誤差** を区別できず, PI controller
が両方を盲目的に dt 縮小で対処していた点が定量化された (#65 bench コメント,
#93 issue body, `tools/verify_beta_m_estimator.py`).

Phase 7 (#93) では Lanczos の a posteriori 誤差推定子

```
err_lanczos_per_call ≈ β_m · |c_m| · ‖ψ_in‖ · dt / m_eff
```

(Saad 1992 / Hochbruck-Lubich 1997 + 高次補正; `tools/verify_beta_m_estimator.py`
の 108 cell sweep で 5% 精度を実証) を `lanczos_propagate` から expose し,
`cfm4_step` / `cfm4_step_with_richardson_estimate` が triangle inequality で
6 Lanczos call ぶんを集約して `err_lanczos_total` を提供する.

adaptive Richardson driver は `err_lanczos_total` を `err` から差し引き,
Magnus 起因の駆動量を取り出す:

```python
err_magnus = max(0.0, err - err_lanczos_total)

# accept は err_magnus ベース.
accept = (err_magnus <= tol_step) or (dt_try <= dt_min)

if accept:
    # PI controller は err_magnus で dt 更新. dt 縮小で減らない Krylov 誤差を
    # 駆動量に含めないので, Krylov 飽和時の dt 過度縮小と step reject 爆発を
    # 回避する.
    err_for_pi = err_magnus if err_magnus > 0.0 else tol_step * 1e-3
    dt = pi_dt_next(dt_try, err_for_pi, p=4, ...)

# 診断: err_lanczos > tol_step なら Krylov 不足. m_max を増やす必要のサイン.
if err_lanczos_total > tol_step:
    n_krylov_insufficient += 1
```

`evolve_schedule_adaptive_richardson` の return tuple は 10-tuple に拡張:

```
(psi_final, t_history, dt_history, n_rejects,
 m_eff_history, beta_m_history, err_lanczos_history, err_magnus_history,
 n_krylov_insufficient, snapshot)
```

`QuantumResult` 側にも `beta_m_stats` (dict: mean/median/min/max/p10/p90) と
`n_krylov_insufficient` (int) を追加.

**後方互換性**: default `krylov_tol = 1e-12` では β_m が十分小さく
`err_lanczos_total << tol_step` となり `err_magnus ≈ err`. PI controller の
挙動は Phase 6 以前と数値的にほぼ同等 (regression check は
`tests/test_adaptive.py::test_adaptive_richardson_error_decomposition_consistency`).

**今後の拡張余地** (#93 Step 4 として別 issue 化検討): `err_lanczos > tol_step`
検出時に `m_max` を `2 × m_max` まで動的拡張 (expokit-style escalation) して
1 step 内で再 build する経路. 本 Phase では diagnostic counter
`n_krylov_insufficient` を expose するに留め, 自動 escalation は行わない.

#### adaptive driver の DX 改善 (Phase 4 follow-up, issue #43 / #54)

`m` / `dt_init` / `dt_max` / `krylov_tol` の手動チューニングを段階的に
自動化する follow-up 群。issue #43 A/B/C で `dt_init="auto"` /
`dt_max="auto"` / `m_max=` を v0.4.x patch リリースで導入し, issue #54 で
3 つの adaptive 経路パラメータ (`krylov_tol` / `dt_init` / `dt_max`) を
**`None` default + auto resolution + 明示 float で override** の統一
スタイルに揃え (`"auto"` リテラル廃止) 公開 API の破壊的変更を導入した。
版数は `docs/conventions.md` §2 ポリシー (Phase 完了時に `0.N.0` へ bump)
に従い, **Phase 5 完了の v0.5.0 minor bump で正式に版数化済** (本変更は
Phase 4 follow-up として 0.4.x 系で先行マージされ, Phase 5 umbrella PR
(issue #45) でまとめて bump された)。

##### A. `dt_init=None` で T 依存 auto resolution (issue #43 A で導入, issue #54 で None default 化)

`QuantumAnnealer.run(method="cfm4_adaptive_richardson_krylov", dt_init=None)`
(既定値) で, facade 側が線形 schedule の Magnus 級数 T スケーリング
(s-space scaling invariance) から導いた保守値を `dt0` に解決する:

```
dt0 = min(max(c · T^β, _AUTO_DT_INIT_FLOOR), T)
       (T = t1 - t0, 既定 c = 0.1, β = 0.5, floor = 1e-3)
```

理論最適は `β = 3/4` (Magnus 切断誤差 `~ K · dt^5`, `K ~ ‖[H_drv, H_p]‖²`
の T 依存と `T = (t1 - t0)` 区間長から導出, 本ファイル §5.3 内
PI controller 既定値の議論と整合) だが, schedule 非線形性や問題依存性に
対するロバスト性を取って `β = 0.5` を既定とした (issue 本文の motivation
参照)。`T < 1` の小 T ケースで `c · T^β` が driver の `dt_min` (default
`1e-4`) を下回らないよう床値 `1e-3` を, 逆に `dt0 > T` で driver 入力
検証 `dt_max >= dt0` (`dt_max = 10 · dt0` default) を満たさなくなる退化
ケースを避けるため上限 `T` を同時に張る。

resolution は facade 層 (`python/maqina/annealer.py` の
`_resolve_dt_init_auto`) で行い, driver (`evolve_schedule_adaptive_richardson`)
は受け取った `dt0` をそのまま使う。これにより driver 単体テスト
(`tests/test_adaptive.py` 既存) は変更不要で, facade 層のテスト (同
ファイル末尾) で None resolution 経路と PI controller との接続を smoke
検証する。issue #54 以前は `dt_init="auto"` で発動する opt-in だったが,
None default 化により既定経路に昇格 (旧 default `0.5` 固定は完全廃止)。

##### B. `dt_max=None` で Lanczos capacity 自動見積もり (issue #43 B で導入, issue #54 で None default 化)

`QuantumAnnealer.run(method="cfm4_adaptive_richardson_krylov", dt_max=None)`
(既定値) で, facade 側が Gershgorin 上界による Lanczos capacity 自動
見積もりを `dt_max` に解決する:

```
‖H‖_est = Σ_i |h_x_i| + max_k |H_p_diag[k]|
dt_max  = max(min(10·dt0, 4m / ‖H‖_est), dt0)
```

Lanczos m 部分空間で `exp(-i dt H) |ψ⟩` を `rel < tol` で再現できる
安全領域は経験的に `dt · ‖H‖ ≲ 4 m` (cv_ising 流, hand-rolled Lanczos の
collapsed safe radius)。`‖H‖` は Gershgorin 上界で closed form に
見積もる (Power method で 5–10 step 走らせる案 2 もあるが overhead が
あるので Phase 4 follow-up では closed form を採用)。最後の
`max(_, dt0)` は driver 入力検証 `dt_max >= dt0` を満たすための floor で,
`dt0` が Lanczos cap を超える縮退ケースでは Richardson 推定子が breakdown
を embedded error として検出し PI controller が dt を縮めるため
fail-safe で成立する (issue #43 B の motivation と整合)。

大 N で `‖H‖ ∝ N` が支配的になる領域では `4m/‖H‖` が default
`10·dt0` を下回り cap が効く。例えば m=24, dt0=0.5, n=10 (h_x=1·10),
H_p_diag in [-1, 1] では `‖H‖_est = 10 + 1 = 11` → cap = 4·24/11 ≈ 8.7,
default 5.0 が支配。n=50 では `‖H‖_est ≈ 51`, cap ≈ 1.88, default 5.0
より cap が支配。

resolution は facade 層 (`python/maqina/annealer.py` の
`_resolve_dt_max_auto`) で行い, driver は既存 `dt_max=` パラメータを
そのまま受ける (driver 内部の入力検証で `dt_max >= dt0` を担保)。
issue #54 以前は `dt_max="auto"` で発動する opt-in だったが, None
default 化により既定経路に昇格 (旧 driver default `10·dt0` 固定は
完全廃止; Lanczos capacity を考慮しないため大 N で危険だった)。

##### C. `m` の adaptive 化 (issue #43 C, v0.4.x で簡略 scope を導入)

`QuantumAnnealer.run(method="cfm4_adaptive_richardson_krylov", m_max=16)` を
渡すと, facade 側で adaptive Richardson 経路の Lanczos 部分空間次元
上限を `self.m` (コンストラクタ既定 24) から `m_max` で上書きする。
step-doubling Richardson 推定子が Lanczos breakdown も embedded error
として検出する fail-safe (Phase 4 C3) を活かし, `m_max=16` 等の保守値
で per-step matvec を 30% 程度削減する運用を許容する (Richardson が
破綻を検知すれば PI controller が dt を絞り精度を維持)。`β_k <
krylov_tol` の早期打切は既存 `lanczos_propagate` で実装済 (`src/krylov.rs`
§ `m_eff` 計算)で, 実効次元は `m_eff ≤ m_max`。

**簡略 scope の理由**: issue 本文の C task は `m_eff` の per-step
累積統計を `QuantumResult` に保存し, `bench_per_step.py` で
`m=adaptive vs m=16 fixed vs m=24 fixed` の wall time 比較を要求する。
これらは Rust 側 `lanczos_propagate` の戻り値拡張 (現状 `Vec<Complex64>`
→ `(Vec<Complex64>, usize)` で `m_eff` を返す) + PyO3 plumbing + Python
driver 集計が必要で, Phase 4 follow-up の DX 改善 PR としては
パッケージが大きすぎる。本リリースでは facade パラメータ `m_max` で
user-facing API を確定させ, m_eff 統計と bench 拡張は別 issue で起票
予定 (Phase 5 で `QuantumResult` の history 拡張と一緒に取り込む案を
含む)。

実機 benchmark 評価は `bench_per_step.py` で `--m 16` / `--m 24` の
2 経路を手動で sweep して per-cell wall time を比較する形でも検証可
(adaptive vs fixed の 1.5–1.7× 期待 speedup は issue 本文 motivation
参照)。

##### E. adaptive driver default の統一 (issue #54, Phase 5 完了の v0.5.0 で版数化済)

PR #53 (issue #52) で `QuantumResult.m_eff_stats` を露出させたところ,
adaptive Richardson の `krylov_tol = 1e-12` (旧 default) が `atol = 1e-8`
default に対して 4 桁過剰タイトで Lanczos β_k 早期打切が `m_eff =
6·m_max` のまま発火しないこと (N=16, T=100, m_max ∈ {16, 24, 32} で
`m_eff_median` が 96 / 144 / 192 と m_max 比例) が実機 bench で判明し,
3 つの adaptive 経路パラメータ (`krylov_tol` / `dt_init` / `dt_max`) を
**`None` default + auto resolution + 明示 float で override** の統一
スタイルに揃える契機となった。同時に PR #51 (issue #43 A/B) で導入した
`dt_init` / `dt_max` の **`None` (固定保守 default)** と **`"auto"`
(問題依存推定)** の 2 経路設計も「根拠の薄い固定保守 default より auto
解決値を default にする」方が筋という観点で再整理した。

| パラメータ | 旧 default | 旧 `"auto"` 挙動 | **新 default (`None` で auto resolution, Phase 5 完了の v0.5.0 で版数化済)** |
|---|---|---|---|
| `krylov_tol` | `1e-12` 固定 | (なし) | `atol · _KRYLOV_TOL_ATOL_RATIO` (既定 `1e-3`) |
| `dt_init` | `None → 0.5` 固定 / `"auto"` あり | `max(min(c·T^β, T), 1e-3)` | 同上 (旧 "auto" 式が default) |
| `dt_max` | `None → 10·dt0` / `"auto"` あり | `max(min(10·dt0, 4m/‖H‖_est), dt0)` | 同上 |

`Literal["auto"]` リテラルは facade から完全削除 (公開 API の破壊的変更;
v0.5.0 で版数化済)。None default = 旧 `"auto"` 経路と挙動上等価なので,
`dt_init="auto"` / `dt_max="auto"` を明示していた呼び出しを `dt_init=None`
/ `dt_max=None` (または引数省略) に置換すれば挙動はビット一致で維持される。

###### `krylov_tol` default の設計方針: accuracy 優先 / 早期打切は opt-in

issue #54 当初の motivation は「default で早期打切を発火させる」だったが,
PR #55 マージ後の実機検証 (N=10/12/16) で **`_KRYLOV_TOL_ATOL_RATIO = 1e-3`
(effective `1e-11`) でも早期打切は発火しない** ことが判明した. その上で
方針を整理した結果, 以下を設計契約とする:

- **default は accuracy 優先**. `atol=1e-8` default + `_KRYLOV_TOL_ATOL_RATIO
  = 1e-3` で effective `krylov_tol = 1e-11`. これは旧 `1e-12` 固定 default
  と挙動上ほぼ同等で, 早期打切は **default では発火しない**. 未指定で
  使った場合に robust に動く性質を維持する.
- **早期打切は user の opt-in** で発動する. user が "大きめの error を
  許容して打切速度を取りたい" と判断した場合に発動する経路:
  - `atol` を緩める (例: `atol=1e-5` → effective `krylov_tol = 1e-8`) →
    PI controller の step 数も減るので二重に高速化
  - `krylov_tol` を直接緩める (例: `1e-6` 等を明示渡し) → atol を変えず
    Lanczos 部分空間のみ早く切る

連動係数 `1e-3` の根拠:

- adaptive Richardson 推定子は `err = ‖ψ_full - ψ_h2‖` を `tol_step` 以下
  に保つよう PI 制御で dt を伸縮する。1 step あたりの Lanczos 内誤差
  (β_k 早期打切時の打切誤差) が PI controller の embedded error 推定を
  支配しないよう, `atol` より **少なくとも 3 桁タイト** に取る経験則.
- これにより default 動作 (`1e-11`) は accuracy 優先, user opt-in 経路
  (atol を 1 桁緩めるごとに effective krylov_tol も 1 桁緩む) は予測
  可能な段階的緩和となる. `atol=1e-5` で effective `1e-8`, `atol=1e-3`
  で effective `1e-6` 等.
- 係数自体の調整 (1e-4 / 5e-4 / 5e-3 等への変更) は `_KRYLOV_TOL_ATOL_RATIO`
  module 定数で 1 箇所集中管理しているので, 将来別問題サイズで挙動を
  tune したくなった際に局所的に変更可能.
- 固定 dt 経路 (`m2` / `cfm4`) は `atol` を取らないため None →
  `_KRYLOV_TOL_FIXED_DEFAULT = 1e-12` に static fallback (旧 default
  維持)。adaptive 経路の atol 連動とは独立。

resolution は全て facade 層 (`python/maqina/annealer.py`) で行い,
driver (`evolve_schedule_adaptive_richardson`) は受け取った `dt0` /
`dt_max` / `krylov_tol` をそのまま使う。これにより driver 単体テストは
変更不要で, facade 層のテスト (`tests/test_adaptive.py`) で None
resolution 経路と PI controller / Lanczos の接続を smoke + bit-exact で
検証する。

