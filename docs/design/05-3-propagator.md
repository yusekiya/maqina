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

kryanneal では `H(t) = A(s(t))·H_driver + B(s(t))·H_problem` の構造を持つ
ため、各 stage で必要なのは **driver / diag の前置係数 1 組ずつ** だけ:

```
stage 1 :  c_drv = a_high·A(s_1) + a_low·A(s_2)
           c_diag = a_high·B(s_1) + a_low·B(s_2)
stage 2 :  c_drv = a_low ·A(s_1) + a_high·A(s_2)
           c_diag = a_low ·B(s_1) + a_high·B(s_2)
```

ここで `s_i = s(t + c_i·dt)`。Rust 側 `apply_h_kryanneal` に
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
`R_i` の `θ = +a·h_x_i·dt` に乗る (`apply_h_kryanneal` の
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
# python/kryanneal/krylov.py

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

`QuantumAnnealer.run(method="cfm4_adaptive_richardson", ...)` はこれを
内部で呼ぶ薄いラッパ。

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

`QuantumAnnealer.run(method="cfm4_adaptive_richardson", dt_init=None)`
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

resolution は facade 層 (`python/kryanneal/annealer.py` の
`_resolve_dt_init_auto`) で行い, driver (`evolve_schedule_adaptive_richardson`)
は受け取った `dt0` をそのまま使う。これにより driver 単体テスト
(`tests/test_adaptive.py` 既存) は変更不要で, facade 層のテスト (同
ファイル末尾) で None resolution 経路と PI controller との接続を smoke
検証する。issue #54 以前は `dt_init="auto"` で発動する opt-in だったが,
None default 化により既定経路に昇格 (旧 default `0.5` 固定は完全廃止)。

##### B. `dt_max=None` で Lanczos capacity 自動見積もり (issue #43 B で導入, issue #54 で None default 化)

`QuantumAnnealer.run(method="cfm4_adaptive_richardson", dt_max=None)`
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

resolution は facade 層 (`python/kryanneal/annealer.py` の
`_resolve_dt_max_auto`) で行い, driver は既存 `dt_max=` パラメータを
そのまま受ける (driver 内部の入力検証で `dt_max >= dt0` を担保)。
issue #54 以前は `dt_max="auto"` で発動する opt-in だったが, None
default 化により既定経路に昇格 (旧 driver default `10·dt0` 固定は
完全廃止; Lanczos capacity を考慮しないため大 N で危険だった)。

##### C. `m` の adaptive 化 (issue #43 C, v0.4.x で簡略 scope を導入)

`QuantumAnnealer.run(method="cfm4_adaptive_richardson", m_max=16)` を
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

resolution は全て facade 層 (`python/kryanneal/annealer.py`) で行い,
driver (`evolve_schedule_adaptive_richardson`) は受け取った `dt0` /
`dt_max` / `krylov_tol` をそのまま使う。これにより driver 単体テストは
変更不要で, facade 層のテスト (`tests/test_adaptive.py`) で None
resolution 経路と PI controller / Lanczos の接続を smoke + bit-exact で
検証する。

