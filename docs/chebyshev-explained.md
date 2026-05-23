# Chebyshev 法によるプロパゲータ近似 — 段階的解説

`kinema` の `chebyshev_propagate` (`src/chebyshev.rs`) が短時間プロパゲータ
`exp(-i H dt) · ψ` をどう近似しているかを, 数式と直感の両面から段階的に
辿る. 既存 Lanczos 経路 (`src/krylov.rs`) との対比, および adaptive
Richardson driver への組込 (`cfm4_step_chebyshev`) も最後に触れる.

実装の一次資料は `src/chebyshev.rs` のモジュール docstring と
`docs/design/05-3-propagator.md` "CFM4:2 + Chebyshev variant" 節.
本ドキュメントは "なぜそうするか" を物理的・数値的直感から補う読み物.

---

## 0. ゴールと前提

時間独立 Hamiltonian
$$
H = a_t \cdot H_\mathrm{drv} + b_t \cdot \mathrm{diag}(h_p)
$$
が初期状態 $\psi$ に作用するプロパゲータ
$$
\psi_\mathrm{new} = \exp(-i H \, dt) \, \psi
$$
を, $H$ を **行列として陽に持たず** matrix-free に計算したい. ここで:

- $H_\mathrm{drv} = -\sum_i h_{x,i} X_i$ (サイト依存横磁場, off-diagonal)
- $\mathrm{diag}(h_p)$ は Z 基底で対角 (`h_p_diag: (2^N,) float64`)
- `apply_h_kinema(ψ, H_drv, h_p, a_t, b_t)` が $H \psi$ を 1 dim-walk で計算する
  既存の matvec primitive (`src/matvec.rs`)

ベンチマーク的事実 (issue #120 PoC):

> N=18 で per-call **29 ms / Lanczos 比 4.45× 高速** (Linux AMD EPYC 7713P).
> Lanczos の Krylov 基底行列 V (dim × m_max = 96 MB at N=18, m_max=24) が
> L3 cache (32 MB / CCX) を溢れるのが構造的 bottleneck で,
> Chebyshev はそこをアルゴリズム軸でバイパスする.

---

## 1. Step 1: 多項式近似で行列指数を書き直す

行列指数 $\exp(-i H \, dt)$ をテイラー展開で書けば
$$
\exp(-i H \, dt) = \sum_{k=0}^\infty \frac{(-i \, dt)^k}{k!} \, H^k
$$
となり, 行列ベクトル積 $H^k \psi$ さえ計算できれば近似できる. しかし
テイラー展開は

- **収束が遅い** ($k \sim \|H\|\, dt$ 程度の項数で初めて effective)
- **項ごとの大きさが暴れる** ($(-i \, dt)^k / k!$ は途中まで爆発し,
  後で減衰する)

ため数値的に扱いにくい. そこで **直交多項式族で展開する** という発想に切り替える.

### Chebyshev 多項式 $T_k(x)$ ($x \in [-1, 1]$)

$T_k(x) = \cos(k \arccos x)$ で定義され,
$$
T_0(x) = 1, \quad T_1(x) = x, \quad T_{k+1}(x) = 2 x \, T_k(x) - T_{k-1}(x)
$$
の **3 項漸化** で計算できる. 区間 $[-1, 1]$ 上で重み $1/\sqrt{1 - x^2}$
について直交し, **任意の連続関数の最良一様近似** を与える性質を持つ
(Chebyshev minimax theorem). 行列指数のような滑らかな関数を有限項で
打ち切ったときの誤差が, テイラー展開と比べて **指数的に小さい** ことが
ここで効いてくる.

---

## 2. Step 2: スペクトルを $[-1, 1]$ に押し込める

$T_k(x)$ は $x \in [-1, 1]$ でしか well-defined ではない. 一方
$H$ の固有値は実数だが範囲は任意 ($\mathbb{R}$ 全体に散らばる).
そこで **affine 変換** で $H$ のスペクトルを $[-1, 1]$ に押し込んだ
正規化 Hamiltonian
$$
\tilde H \;\equiv\; \frac{H - E_c \, I}{R}
$$
を導入する. ここで

- $E_c \;=\; \frac{E_\mathrm{max} + E_\mathrm{min}}{2}$ (スペクトル中心)
- $R \;\;=\; \frac{E_\mathrm{max} - E_\mathrm{min}}{2}$ (スペクトル半径)

とすれば $\tilde H$ の固有値は $[-1, 1]$ に収まる. 元の指数は
$$
\exp(-i H \, dt) \;=\; \exp(-i E_c \, dt) \, \exp(-i R \, dt \cdot \tilde H)
$$
と分解でき, 前半は **状態 $\psi$ に掛ける global phase** (スカラー乗算
1 回), 後半が $T_k(\tilde H)$ で展開する本体になる.

### $(E_c, R)$ の見積もり — Gershgorin の行和上界

$H$ の固有値範囲を厳密に求めるには対角化が必要で本末転倒. しかし
**Gershgorin の円板定理** で上下界を **行列ベクトル積なしの closed form** で
推定できる. TFIM の場合:

- 対角寄与: $H_{kk} = b_t \cdot h_p[k]$ ($k$ ごとに異なる)
- 非対角行和: $\sum_{j \neq k} |H_{jk}| = |a_t| \cdot \sum_i |h_{x,i}|$
  (TFIM の bit-flip 構造で $k$ に依存しない一定値)

各行の Gershgorin 円板は $[H_{kk} - \mathrm{off}, H_{kk} + \mathrm{off}]$
($\mathrm{off} = |a_t| \sum_i |h_{x,i}|$) なので, 全固有値はその union に
入る:
$$
E_\mathrm{max} \;\le\; \max_k (b_t \cdot h_p[k]) + |a_t| \sum_i |h_{x,i}|, \\
E_\mathrm{min} \;\ge\; \min_k (b_t \cdot h_p[k]) - |a_t| \sum_i |h_{x,i}|.
$$
実装は `src/chebyshev.rs::gershgorin_bounds_cached`. `IsingProblem` 構築時に
$h_x$ の絶対値和と $h_p$ の min / max を 1 度だけ計算しておき, per-step は
それを使って **O(1) (数値演算 5 回)** で済ませる。素朴に毎回 `h_p_diag` を
full walk すると $O(2^N)$ で N=18 で wall time の 1% 弱を占めるため,
hot path では明示的にこの precompute 経路を使う契約とする (`IsingProblem`
が `h_x_abs_sum` / `h_p_diag_min` / `h_p_diag_max` を property で公開し,
driver がこれを Rust 側に渡す)。上界の精度が緩いと後述の $z = R \, dt$ が
大きくなり項数 $K$ も増えるが, 安全側に倒している.

---

## 3. Step 3: Jacobi-Anger 展開

正規化 $\tilde H$ に対し $\exp(-i z \tilde H)$ を **Chebyshev 多項式の
無限和** に展開する公式が古典的に知られている (Jacobi-Anger expansion):
$$
\exp(-i z x) \;=\; J_0(z) + 2 \sum_{k=1}^\infty (-i)^k J_k(z) \, T_k(x), \quad x \in [-1, 1]
$$
ここで $J_k(z)$ は **第 1 種 Bessel 関数**. $x \to \tilde H$ (行列) に
formal に置き換えれば
$$
\boxed{
\exp(-i H \, dt) \, \psi
\;=\; \exp(-i E_c \, dt) \cdot \sum_{k=0}^\infty c_k(z) \, T_k(\tilde H) \, \psi
}
$$
ここで
- $z = R \, dt$ (無次元の "回転角" パラメータ)
- $c_k(z) = (2 - \delta_{k,0}) \cdot (-i)^k \cdot J_k(z)$
  ($k = 0$ のみ係数 1, $k \ge 1$ は係数 2)

この展開の **何が嬉しいか**:

1. **Bessel 関数 $J_k(z)$ の急減衰**: $k > z$ で $|J_k(z)|$ は
   **superexponential** (Bessel の漸近挙動) に小さくなる. 例えば
   $z = 10$ で $|J_{30}(10)| \sim 10^{-26}$. 有限項で打ち切る誤差が
   指数的に小さい.
2. **per-項コストが軽い**: $T_k(\tilde H) \psi$ は 3 項漸化で 1 matvec /
   項. つまり項数 $K$ に対し $O(K \cdot \text{matvec})$.
3. **メモリが定数**: Lanczos のように $K$ 個のベクトルを全部保持する必要が
   なく, 3 本 (現在/前/出力) で済む (Step 5 で詳述).

---

## 4. Step 4: 切り捨て次数 $K$ をどう決めるか

無限和を有限 $K$ で打ち切ったときの末尾誤差は
$$
\|\,\text{truncation residual}\,\|
\;\approx\; 2 \cdot |J_{K+1}(z)| \cdot \|\psi\|
$$
($c_k$ の絶対値が支配項; Tal-Ezer & Kosloff 1984). したがって
**「許容誤差 `tol` を切る最小 $K$」** を選びたい. 実装
(`determine_truncation`) は単純に
$$
K \;=\; \min \{\,k \ge 1 \;:\; |J_k(z)| < \mathrm{tol}/2 \,\}
$$
を返す. 経験則として $K \approx z + O(\log(1/\mathrm{tol}))$ で,
具体的な目安:

| $z$ | tol = $10^{-10}$ | tol = $10^{-4}$ |
|---|---|---|
| 1 | ~10 | ~5 |
| 10 | 25-40 | ~15 |
| 50 | 70-85 | ~55 |

これが Lanczos の $m_\mathrm{eff}$ に相当する **per-step matvec 回数** を
決める. 上限 `k_max_cap = 5000` で病的ケース (大 $\|H\| dt$) も clamp.

### Bessel 係数 $J_0, J_1, \ldots, J_{K+1}$ の計算

Miller's **downward recurrence** で `bessel_j_array(z, K+1)` が
一括計算する. forward $J_{k+1} = (2k/z) J_k - J_{k-1}$ は大 $k$ で
数値不安定 ($J_k$ が指数的に減衰するのに誤差が増殖) なため,
**任意の出発値**から下降させて self-stabilizing な性質を使う:

1. $N_\mathrm{start} \gg K$ から $b_{N_\mathrm{start}+1} = 0$,
   $b_{N_\mathrm{start}} = 1$ で出発
2. 下降漸化 $b_{k-1} = (2k/z) b_k - b_{k+1}$ で $k = N_\mathrm{start}, \ldots, 1$
3. 総和規約 $J_0(z) + 2 \sum_{j \ge 1} J_{2j}(z) = 1$ で正規化倍率を決定
4. 全 $b_k$ を倍率で正規化

実装は `src/chebyshev.rs::bessel_j_array`. 出発点 $N_\mathrm{start} =
\max(K + 30, 2|z| + 50)$ で `scipy.special.jv` と rel < $10^{-13}$ 一致を
test (`bessel_jv_matches_scipy_reference`) で実証.

---

## 5. Step 5: 3 項漸化で $T_k(\tilde H) \psi$ を回す

ここが Chebyshev 法の **計算カーネル**. $\phi_k \equiv T_k(\tilde H) \psi$
と置くと, $T_k$ の 3 項漸化から
$$
\phi_0 = \psi, \qquad \phi_1 = \tilde H \psi = \frac{H \psi - E_c \psi}{R}
$$
$$
\phi_{k+1} = 2 \tilde H \phi_k - \phi_{k-1} = \frac{2 (H \phi_k - E_c \phi_k)}{R} - \phi_{k-1}
$$
を回しながら, 同時にアキュムレータ
$$
\psi_\mathrm{acc} \;\leftarrow\; \psi_\mathrm{acc} + c_k \, \phi_k
$$
を更新していく. ループ終了後に global phase $\exp(-i E_c \, dt)$ を掛けて
$\psi_\mathrm{new}$ を得る.

### 3 ベクトル rotation でメモリ定数

実装は **3 本の作業ベクトル** で回る:

```
phi_prev (= φ_{k-1}), phi_curr (= φ_k), scratch (= H · φ_k → φ_{k+1})
```

各 $k$ ステップで:

1. **matvec**: `scratch := H · phi_curr` (`apply_h_kinema` 経由, $O(\dim)$ pass)
2. **scaling**: `scratch := 2 · (scratch - E_c · phi_curr) / R - phi_prev`
3. **accumulate**: `psi_acc += c_k · scratch`
4. **rotate**: `phi_prev ← phi_curr`, `phi_curr ← scratch`

メモリ消費は **dim × 4 = 16 MB at N=18** (φ_prev / φ_curr / scratch /
psi_acc) で **CCX L3 32 MB に収まる**. ここが Lanczos との決定的な差:

| 経路 | Krylov 基底メモリ | N=18 でのサイズ | L3 fit |
|---|---|---|---|
| Lanczos | dim × m_max (24) | 96 MB | **超過 → cache spill** |
| Chebyshev | dim × 4 (固定) | 16 MB | OK |

### Gram-Schmidt 直交化が不要

Lanczos は数値的に直交性が崩れるため re-orthogonalization が必要で,
これが BLAS-1 dot/axpy の $n^2$ 二次項を生む. Chebyshev の 3 項漸化は
**数学的に直交性を保証する** ため Gram-Schmidt が原理的に要らない.
Phase A speedup (4.45×) の主因.

---

## 6. Step 6: 数値安定性と性能の小さな仕掛け

### Fast-path: スペクトル半径 $R < 10^{-15}$

$h_x = 0$ かつ $h_p = 0$ (zero Hamiltonian) の場合 $R = 0$ で
$\tilde H$ が ill-defined. このとき
$\exp(-i H \, dt) \psi = \exp(-i E_c \, dt) \psi$ で global phase だけ.
fast-path として返す (`R < 1e-15` 判定).

### SIMD + fusion (issue #126)

$k \ge 2$ の inner loop で walk 2 (scaling) と walk 3 (accumulate) を
**1 dim-walk + `wide::f64x4` SIMD** に fuse する `chebyshev_recurrence_fused`
を持つ. 旧実装は 3 dim-walk (matvec + scalar scaling + scalar accumulate)
だったのを **2 dim-walk + SIMD** に圧縮.

### rayon 並列化 (issue #127)

`chebyshev_recurrence_fused_rayon` で $k \ge 2$ inner loop を
`par_chunks_mut` で chunk 分割し, 各 chunk 内で SIMD kernel を呼ぶ
2 段並列. `apply_h_kinema` (matvec) は #62 で既に rayon 並列化済なので,
これで Chebyshev 経路の non-matvec hot loop もスケールする
(64 thread で parallel efficiency 27% → 44%).

### 末尾誤差推定

ループ終了後に
$$
\mathrm{err\_estimate} \;\approx\; 2 \, |J_{K+1}(z)| \cdot \|\psi\|
$$
を 1-term 上界として返す. 上位の adaptive driver はこの値で per-step
誤差をモニタする.

---

## 7. アルゴリズム全体像 (擬似コード)

```text
Input: h_x[N], h_p[2^N], a_t, b_t, ψ[2^N], dt, tol
Output: ψ_new = exp(-i H dt) · ψ

# Step 2: スペクトル境界
(E_min, E_max) = gershgorin_bounds(h_x, h_p, a_t, b_t)
E_c = (E_max + E_min) / 2
R   = (E_max - E_min) / 2
if R < 1e-15:                         # zero-H fast-path
    return exp(-i E_c · dt) · ψ

# Step 4: 切り捨て次数
z = R · dt
K = determine_truncation(z, tol)      # min k s.t. |J_k(z)| < tol/2
J = bessel_j_array(z, K + 1)          # Miller downward recurrence

# Step 5: 3 項漸化
φ_prev = ψ                            # = T_0(H̃)ψ
ψ_acc  = J[0] · φ_prev                # c_0 = J_0
if K >= 1:
    φ_curr = (apply_H(ψ) - E_c · ψ) / R          # = T_1(H̃)ψ
    ψ_acc += (-2i · J[1]) · φ_curr               # c_1 = -2i J_1
for k = 2 .. K:
    scratch = apply_H(φ_curr)                    # walk 1 (matvec)
    # walk 2 (fused, SIMD):
    #   scratch := 2·(scratch - E_c · φ_curr) / R - φ_prev
    #   ψ_acc   += c_k · scratch                 # c_k = 2 · (-i)^k · J_k
    fused_recurrence(scratch, φ_curr, φ_prev, ψ_acc, E_c, 1/R, c_k)
    rotate: φ_prev ← φ_curr, φ_curr ← scratch

# Step 6: global phase
ψ_acc *= exp(-i E_c · dt)
return (ψ_acc, K, 2·|J[K+1]|·‖ψ‖)
```

---

## 8. 時間依存 $H(t)$ への拡張: CFM4:2 Magnus に挿す

ここまでは **時間独立 $H$** を前提にしていた. kinema の本来の目的は
時間依存 Schedule $H(t) = A(s(t)) H_\mathrm{drv} + B(s(t)) H_\mathrm{problem}$
の量子アニーリングダイナミクスなので, Chebyshev 単体では使えない.

これに Magnus 展開 (CFM4:2) を組み合わせる. CFM4:2 は短時間プロパゲータ
$U(t+dt, t)$ を
$$
U(t+dt, t) \;\approx\; \exp(-i \, dt \cdot B_2) \, \exp(-i \, dt \cdot B_1)
$$
の **2 段の "時間独立 H で frozen させた指数"** に分解する 4 次 commutator-free
Magnus (詳細は `docs/design/05-3-propagator.md` §5.3 CFM4:2 節). 各
stage は

```
stage 1: B_1 = a_high · H(t + c_1·dt) + a_low · H(t + c_2·dt)
              (時間独立の係数 (c_drv_1, c_diag_1) に畳み込み)
         → ψ_mid = chebyshev_propagate(h_x, h_p, c_drv_1, c_diag_1,
                                       ψ, dt, chebyshev_tol)
stage 2: B_2 = a_low · H(t + c_1·dt) + a_high · H(t + c_2·dt)
         → ψ_new = chebyshev_propagate(..., c_drv_2, c_diag_2, ψ_mid, dt, ...)
```

の形になる. つまり **時間独立 H のスナップショット 2 枚で 1 step を構成**
するので, Chebyshev はそのまま使える. per-stage で Gershgorin
$(E_c, R)$ を再計算する必要があるが closed form O(N) なので 1 step あたり
合計 0.006 ms 程度, wall time 0.003% で無視可.

これを Lanczos の代わりに `cfm4_step_chebyshev` として `src/cfm4.rs` に
実装してあり, さらに step-doubling Richardson (full-step と half-step×2 の
差で局所誤差を推定) と PI controller を上に乗せた adaptive driver が
`method="cfm4_adaptive_richardson_chebyshev"` (default) として公開
されている.

---

## 9. Lanczos との対比サマリ

| 観点 | Lanczos | Chebyshev |
|---|---|---|
| 展開基底 | Krylov 部分空間 $\{\psi, H\psi, H^2\psi, \ldots\}$ | Chebyshev 多項式 $T_k(\tilde H)\psi$ |
| 短時間プロパゲータ | 3 重対角化 + 部分空間内 exp | 3 項漸化 + Bessel 係数加算 |
| 直交化 | re-orthogonalization が要る | 漸化が直交保証 (不要) |
| メモリ | $\dim \times m_\mathrm{max}$ (N=18 で 96 MB) | $\dim \times 4$ (固定; 16 MB) |
| L3 fit | N ≳ 16 で **超過** | 余裕で収まる |
| per-step cost | $O(m_\mathrm{eff} \cdot \text{matvec}) + O(m^2)$ GS | $O(K \cdot \text{matvec})$ のみ |
| スペクトル情報 | 自己適応 (Lanczos 内で抽出) | 事前に Gershgorin で見積もり必要 |
| 終了条件 | a posteriori $\beta_m \cdot \|c_m\|$ 早期打切 | $\|J_K(z)\| < \mathrm{tol}/2$ で次数決定 |
| 適用範囲 | 任意の (Hermitian) $H$ | スペクトル境界 $(E_c, R)$ が要る |

実測 (N=18, Linux AMD EPYC 7713P): per-step wall **Lanczos 比 5.49×
高速**, branch-miss 158× 減, sys time 78× 減, parallel efficiency 27% → 44%.
これが **`kinema` 0.11.0 で default method を Chebyshev variant に切り替えた
根拠** (issue #124).

---

## 10. もっと深く知りたいときの読み順

1. `src/chebyshev.rs` モジュール docstring (本 doc の一次資料, 数式と実装の対応)
2. `docs/design/05-3-propagator.md` "CFM4:2 + Chebyshev variant" 節
   (Magnus 統合, $\mathrm{atol}$ と $\mathrm{chebyshev\_tol}$ の関係, accidental
   高精度仕様)
3. `docs/design/12-release-plan.md` Phase B (#122) / Phase B follow-up (#124, #126, #127)
   (Definition of Done, bench acceptance gate)
4. Tal-Ezer & Kosloff 1984 (J. Chem. Phys. 81, 3967) — Chebyshev propagator
   の古典論文
5. Abramowitz & Stegun §9.12 / Numerical Recipes §6.5 — Bessel 関数の
   Miller's recurrence
