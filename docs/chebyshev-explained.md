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

### Bessel 係数 $J_0, J_1, \ldots, J_{K+1}$ の計算 — Miller's downward recurrence

Step 5 の漸化に入る前に, **「$J_k(z)$ という数値スカラーをそもそもどう計算するか」**
という問題が残ります。kinema は **Miller の下降漸化** という古典的手法で
$J_0(z), J_1(z), \ldots, J_{K+1}(z)$ を **一括かつ機械精度** (相対誤差
$< 10^{-13}$) で求めます。Bessel 関数論を呼ばずに, **(a) 漸化式 (b) 急減衰
(c) 総和恒等式** の 3 道具だけで仕組みを説明できます。

#### 道具 1 — Bessel 自身の 3 項漸化

Bessel 関数も 3 項漸化を満たします:
$$
J_{n+1}(z) = \frac{2n}{z} J_n(z) - J_{n-1}(z).
$$
(これは Bessel の母関数 $\exp[\tfrac{z}{2}(t - 1/t)]$ を $t$ で微分して
係数比較するだけで導ける。`T_k(\tilde H)\psi` の漸化と式の形は同じだが,
こちらは **異なる関数列 $\{J_n\}_n$ 同士** の関係である点に注意。)

#### 道具 2 — $J_n(z)$ は大 $n$ で超指数的に減衰

§4 で「打ち切り次数 $K$ を $|J_{K+1}(z)| < \mathrm{tol}/2$ で決める」と
言いました。ここで使った $|J_n(z)| \to 0$ ($n \to \infty$) の性質, しかも
$n > z$ で
$$
|J_n(z)| \;\lesssim\; \frac{1}{\sqrt{2\pi n}} \!\left(\frac{e z}{2 n}\right)^{n}
$$
で **超指数的に減衰** することを再利用します。例: $z = 1, n = 30$ で
$|J_n| \sim 10^{-40}$。

#### 道具 3 — 漸化式の解空間は 2 次元

Step 1 の漸化式は $n$ について **2 階線形差分方程式**。連続 2 値 $(a_0, a_1)$
を指定すれば全 $n$ で一意に決まり, 解の集合は **2 次元ベクトル空間**。基底の
1 つは $\{J_n(z)\}$ 自身, もう 1 つは **何か別の独立解** (具体形は不要)。
任意の解は
$$
a_n = \alpha J_n(z) + \beta Z_n(z), \qquad (\alpha, \beta) \in \mathbb{C}^2.
$$
ここで $Z_n$ は「$J_n$ と独立な解」というだけで, 中身は知らなくてよい。

#### 観察 1: forward 漸化は不安定

$J_0(z), J_1(z)$ を Taylor 級数等で別計算し $n = 0 \to K$ に **forward** で
回す素朴な方法は **必ず破綻** します。理由:

道具 3 の "$Z_n$" は **道具 2 の "$J_n$ 急減衰" の裏返しで必ず急発散** します。
理由は線形代数: もし $Z_n$ も減衰したら, 2 連続値の Casoratian
$a_n b_{n+1} - a_{n+1} b_n$ が 0 に潰れ独立性が崩れる。よって **$Z_n$ は
$J_n$ と逆向きに超指数的に発散** していないと, $\{J_n, Z_n\}$ が基底足り得ない。

forward 漸化を回すと:
- $J_n$ は減衰
- 計算機の丸め誤差 $\sim 10^{-16}$ は必ず $Z_n$ 成分 (= $\epsilon Z_n$) を持ち込む
- $Z_n$ は **発散** → 誤差項が指数的に成長
- $n \sim 15$ あたりで丸め誤差が真値を圧倒し $J_{30} \sim 10^{-40}$ は出ない

#### 観察 2: backward 漸化は self-stabilizing

漸化式を **backward** ($n = N \to 0$) に回すと, **増殖方向が逆転** します。
線形性から, 任意の出発値 $(b_{N+1}, b_N)$ を解空間で展開:
$$
b_n = \alpha J_n(z) + \beta Z_n(z).
$$
$\alpha, \beta$ は出発値で決まり, **下降中ずっと固定**。値が変わるのは
$J_n, Z_n$ 自身:

- **下降中 $J_n$ は (大 $n$ で小, 小 $n$ で moderate に) 増える**
- **下降中 $Z_n$ は (大 $n$ で大, 小 $n$ で moderate に) 減る**

両者の比 $|Z_n / J_n|$ は道具 2 の漸近から **下降 1 ステップで
$\sim (z/(2n))^2$ 倍になります** (= $(2n/z)^2$ で割られる)。
$n \gg z$ の領域では $(z/(2n))^2 \ll 1$ なので **比は急減少 = 1 ステップで
数桁の精度向上**。導出は具体的に: $J_n$ 自身は下降 1 ステップで
$J_{n-1}/J_n \approx 2n/z$ 倍に **増え**, $Z_n$ は $Z_{n-1}/Z_n \approx z/(2n)$
倍に **減る**。比を取ると $J$ 側の逆数 ($z/(2n)$) と $Z$ 側 ($z/(2n)$) が
両方とも汚染比を縮める方向に効き, 積で $(z/(2n))^2$ となる。

#### 直感的な数値例 ($z = 1$)

出発 $N_\mathrm{start} = 30$, $(b_{31}, b_{30}) = (0, 1)$ で下降漸化を回す
場合, 出発値を $\alpha J_n + \beta Z_n$ 展開すると Casoratian 不変量から
$$
\alpha \sim 10^{38}, \qquad \beta \sim 10^{-42}.
$$
つまり **"適当な出発" 値は実は $J$ 方向に既に scale $10^{38}$ で振っており**,
$Z$ 成分は超指数的に小さい (= 道具 2 の $J_{30} \sim 10^{-40}$ の逆数オーダ)。

下降中 $\alpha, \beta$ は固定。target $n = 0$ で
$$
\left|\frac{\beta Z_0}{\alpha J_0}\right| \;\sim\; \frac{10^{-42}}{10^{38}} \cdot \mathcal{O}(1)
\;\sim\; 10^{-80}.
$$
**$Z$ 汚染は真値より 80 桁小さく**, 機械精度 ($10^{-16}$) を遥かに下回る。
よって $b_0 \approx \alpha J_0(z)$ が機械精度で成立。同じく $b_1 \approx \alpha
J_1$, $b_2 \approx \alpha J_2$, ... と全 $n$ で **scale $\alpha$ 倍された $J_n$**
が得られる。

#### 残った問題: scale 定数 $\alpha$ を決める

$b_n \approx \alpha J_n(z)$ までは分かったが, $\alpha$ は出発値次第で値が
変わるので未知。これを決めるのに **Jacobi-Anger の特殊値** を使います。

Step 3 で導いた展開
$$
e^{iz\cos\theta} = J_0(z) + 2 \sum_{m=1}^{\infty} i^m J_m(z) \cos(m\theta)
$$
に $\theta = \pi/2$ を代入。LHS は $e^{0} = 1$, RHS の $\cos(m\pi/2)$ は
$m$ 偶数で $\pm 1$, 奇数で $0$ なので, $m = 2j$ の和だけ残り
$i^{2j}\cos(j\pi) = (-1)^j \cdot (-1)^j = 1$ で整理:
$$
\boxed{\;J_0(z) + 2 \sum_{j=1}^{\infty} J_{2j}(z) = 1\;}
$$
**$z$ によらず常に成立する恒等式**。

#### 正規化手順

1. 下降漸化で $b_n \approx \alpha J_n$ ($n = 0, 1, \ldots, K+1$) を得る
2. 同じ scale で総和 $b_0 + 2(b_2 + b_4 + \cdots)$ を計算 → 値は $\alpha \cdot 1 = \alpha$
3. 全 $b_n$ を $\alpha$ で割れば真の $J_n(z)$

総和恒等式が **"$\alpha$ の温度計"** として働き scale を確定。

#### 何を仮定したか

この導入で使った前提は 3 つだけ:

1. Bessel 漸化式 (母関数 $\partial_t$ から導出可能)
2. $J_n(z)$ の急減衰 (§4 と同じ事実)
3. Jacobi-Anger 総和規約 (§3 の Jacobi-Anger に $\theta = \pi/2$ 代入)

**Bessel 微分方程式 / Frobenius 法 / 第 2 種 Bessel 関数 $Y_n$** などは
一切使っていません。"独立な解 $Z_n$" の具体形は知らなくてよく, **「2 次元
解空間の中で $J$ と独立な解は $J$ と逆向きに発散しないと整合しない」** という
線形代数の事実だけで足りています。

#### 実装 (`src/chebyshev.rs::bessel_j_array`)

```rust
// 出発点 (target k_max より深く取って汚染消去マージン確保)
let n_start = (k_max + 30).max(2 * abs_z_u + 50);

// 任意の出発値で下降漸化開始
b[n_start + 1] = 0.0;
b[n_start] = 1.0;

// 下降: b[k-1] = (2k/z) · b[k] - b[k+1]
for k in (1..=n_start).rev() {
    b[k - 1] = (2.0 * k as f64 / z) * b[k] - b[k + 1];
    // overflow guard: |b| > 1e100 で全配列を 1e-100 倍にスケール
    if b[k - 1].abs() > 1e100 {
        for v in b.iter_mut() { *v *= 1e-100; }
    }
}

// 総和規約で正規化
let mut sum_norm = b[0];
for j in 1.. {
    let idx = 2 * j;
    if idx > n_start { break; }
    sum_norm += 2.0 * b[idx];
}
let inv = 1.0 / sum_norm;
for k in 0..=k_max { out[k] = b[k] * inv; }
```

定数 `+30` は観察 2 の "汚染抑制 1 ステップあたり数桁" を 30 ステップ重ねて
**double 16 桁を遥かに超えるマージン** を確保するため。`+50` は $K < |z|$ の
小 $z$ case の保険。scipy.special.jv との rel < $10^{-13}$ 一致を Rust unit
test (`bessel_jv_matches_scipy_reference`) で 8 cells (low/medium/high の
$(z, k)$ 組) について検証済。

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
