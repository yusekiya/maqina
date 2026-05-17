# §5.1 matvec / per-axis primitives

Rust 側に持つ低レベル配列演算プリミティブは 2 種類:

| プリミティブ | 用途 | 導入 phase | 動作モード |
|---|---|---|---|
| `apply_h_kryanneal` | Lanczos / CFM4:2 (Magnus 系) の matvec | Phase 1 | additive: `y += c·H·v` 系 |
| `apply_single_mode_axis_i` | Trotter 系の 2×2 ユニタリ適用 | Phase 2 | in-place rotation |

両者とも bit-flip 構造 (`(psi[k], psi[k ^ (1<<i)])` のペア処理) を共有するが、
matvec は **複数 i の寄与を 1 つの y に足し込む** のに対し、Trotter は
**1 つの i ごとに in-place で psi を回転する** という違いがあり、メモリ
アクセスパターンと最適化の余地が異なるため別関数として書く。

#### 5.1.1 `apply_h_kryanneal` (Phase 1)

時間依存スカラー `(A_t, B_t)` を渡し、Rust 内で `(A_t · h_x, B_t · H_p_diag)`
を組み合わせた matvec を 1 回適用する。Python 越境は **Lanczos の 1 step
あたり呼ばない**。

擬似コード (Rust 側):

```rust
/// y = A·H_driver·v + B·H_p_diag·v
///
/// H_driver = -Σ_i h_x_i X_i (-Σ X_i の inhomogeneous 拡張).
fn apply_h(
    v: &[Complex64],
    y: &mut [Complex64],
    h_x: &[f64],          // length n
    h_p_diag: &[f64],     // length 2^n
    a_t: f64,
    b_t: f64,
    n: usize,
) {
    let dim = 1usize << n;
    // diagonal 部分: y[k] = B·H_p[k]·v[k]
    for k in 0..dim {
        y[k] = Complex64::new(b_t * h_p_diag[k], 0.0) * v[k];
    }
    // bit-flip 部分: y[k] += -A · h_x_i · v[k ^ (1<<i)]
    for i in 0..n {
        let coeff = -a_t * h_x[i];
        let mask = 1usize << i;
        // i ∈ {0,1,2} は Phase 6 C2 (issue #63) で wide::f64x4 特化版に dispatch.
        // i ≥ 3 は scalar inner loop のまま (stride ≥ 8 で SIMD 利得が小さい).
        for k in 0..dim {
            y[k] += Complex64::new(coeff, 0.0) * v[k ^ mask];
        }
    }
}
```

性能考慮:

- diagonal 部分は cblas `zdscal` + 要素積を unrolled inner loop で
- bit-flip 部分は `i` 外側 / `k` 内側で連続アクセスにできる: ビット i に
  ついて k と k^mask のペアを `mask=1<<i` で 2 元ストライドにより列挙。
  i=0 で stride 1、i=1 で stride 2、... と level-by-level に走査することで
  TLB / L2 ヒット率を上げる古典的テクニック (state-vector simulator の
  X-gate pass と同一)。
- **Phase 6 C1 (issue #62, v0.6 で版数化予定) で rayon `par_chunks_mut`
  経由の L2 並列化を導入済み**。`y` を chunk 分割し各 chunk closure 内で
  diag pass + 全 i bit-flip pass を **fuse** (cache-blocked 形)。`y_chunk`
  を L1 cache resident に保つことで後段 SIMD (C2) / cache block-fusion
  (C3) の足場とする。`feature = "rayon"` (default ON) で有効化, scalar
  単スレッドビルドは `--no-default-features` でフォールバック。
- **Phase 6 C2 (issue #63) で `wide::f64x4` 経由の SIMD 特化を導入済み**。
  `apply_h_kryanneal` の bit-flip pass の i ∈ {0,1,2} (stride 1/2/4 連続
  アクセス領域) を `simd_kernels::bitflip_iN` に dispatch する。SIMD inner
  kernel は `apply_h_kryanneal_serial` と `_rayon` の両 path から共通で
  呼ばれ, rayon path では chunk_size を `SIMD_BLOCK_MAX = 8` Complex64 の
  倍数に丸めて block-aligned 前提を満たす。i ≥ 3 は scalar inner loop の
  まま (stride ≥ 8 で SIMD 利得が小さく cache line を跨ぐ; §12 Phase 6 C2)。
  `feature = "simd"` (default ON) で有効化, `--no-default-features` で
  scalar fallback。**cache block-fusion (Phase 6 C3, #64)** は trotter_step
  側に集中して実装済み (§5.1.3) で `apply_h_kryanneal` 本体は touch せず.
  `apply_h_kryanneal` の DRAM bandwidth 改善は **issue #79 (Phase 6 D)** で
  group-fused 3-phase 形を試行したが Linux 本番 bench + perf 計測で
  **真の compute regression が確認され未採用** (詳細は §5.1.4).
- **dim 閾値による rayon dispatch (issue #68, follow-up)**: `apply_h_kryanneal`
  と `apply_single_mode_axis_i` の **public 関数側で `dim < MIN_RAYON_DIM`
  (= 1 << 17 = 128K 要素 = 2 MB Complex64) を判定し scalar 単スレッド経路
  (`*_serial`) にフォールバック** する。N ≤ 16 (dim ≤ 64K) では rayon
  barrier overhead が単スレッド計算時間を超えて regression (Phase 4-5 と
  同じ Linux サーバー本番 bench で N=16 / 2 threads = 0.57× 観測, issue #62
  本番 sweep)。private `*_rayon` 関数は dispatch を含まず常に rayon 実行
  するため, 既存 rayon-path テスト (`apply_*_rayon_matches_serial`,
  `apply_*_rayon_determinism_8thread_100iter`) は `*_rayon` を直接呼んで
  rayon path の正当性を検証し続ける。`MIN_RAYON_DIM` の値は const なので
  再評価には release rebuild が必要。

#### 5.1.2 `apply_single_mode_axis_i` (Phase 2)

Trotter 経路で `R_i(θ) = cos(θ)·I + i·sin(θ)·X_i` を psi に in-place で
適用する関数。X_i が bit-flip 演算子であることを活用し、`(psi[k],
psi[k ^ (1<<i)])` のペアごとに 2×2 ユニタリ U を直接乗じる:

```rust
/// psi を axis i で 2 元化したペアに 2×2 ユニタリ U を in-place 適用.
///
/// U は `[[u00, u01], [u10, u11]]` の row-major 2x2 行列。
/// Trotter で R_i(θ) を渡す場合は u00=u11=cos θ, u01=u10=i·sin θ.
fn apply_single_mode_axis_i(
    psi: &mut [Complex64],
    u: &[Complex64; 4],   // row-major 2x2
    i: usize,
    n: usize,
) {
    let dim = 1usize << n;
    let mask = 1usize << i;
    // bit i = 0 の k だけを enumerate (重複適用を避ける)
    let mut k = 0usize;
    while k < dim {
        if k & mask != 0 {
            k += 1;
            continue;
        }
        let a = psi[k];
        let b = psi[k | mask];
        psi[k]        = u[0] * a + u[1] * b;
        psi[k | mask] = u[2] * a + u[3] * b;
        k += 1;
    }
}
```

(実装は `i` ごとに stride を変える 2 重ループ形に整える: 外側で長さ
`1<<i` のブロックを走り、内側で連続 dim/2 ペアを処理する。`k & mask != 0`
のスキップを実行せずに済む形にする。詳細は実装時に決める。)

呼び出し側 (`trotter_step`):

```rust
pub fn trotter_step(
    psi: &mut [Complex64],
    h_x: &[f64],          // length n, サイトごとの横磁場振幅
    h_p_diag: &[f64],     // length 2^n
    a_t: f64,             // A(s(t + dt/2)) などの schedule 値 (中点)
    b_t: f64,
    dt: f64,
    n: usize,
) {
    let dim = 1usize << n;

    // Strang: phase_p(dt/2) -> Π_i R_i(dt) -> phase_p(dt/2)
    let half = 0.5 * dt;
    for k in 0..dim {
        let phi = -b_t * h_p_diag[k] * half;
        psi[k] *= Complex64::new(phi.cos(), phi.sin());
    }
    for i in 0..n {
        // H_drv = -Σ h_x_i X_i なので
        //   exp(-i·dt·a_t·H_drv) = Π_i exp(+i·θ_i·X_i), θ_i = a_t·h_x_i·dt
        // → u = cos(θ)·I + i·sin(θ)·X with θ = +a_t·h_x_i·dt.
        // (H_drv の負符号は θ では巻き取らない. apply_h_kryanneal の
        //  coeff = -a_t·h_x_i と同じ convention.)
        let theta = a_t * h_x[i] * dt;
        let c = theta.cos();
        let s = theta.sin();
        let u = [
            Complex64::new(c, 0.0),
            Complex64::new(0.0, s),
            Complex64::new(0.0, s),
            Complex64::new(c, 0.0),
        ];
        apply_single_mode_axis_i(psi, &u, i, n);
    }
    for k in 0..dim {
        let phi = -b_t * h_p_diag[k] * half;
        psi[k] *= Complex64::new(phi.cos(), phi.sin());
    }
}
```

**設計判断 (一般的な reshape + GEMM パターンを採用しない理由)**:

連続変数 / 一般 bosonic 系では `apply_single_mode(M, psi, axis)` を
「`psi` を `(N_fock,) * m` に reshape → 単一 BLAS GEMM」 パターンで書くのが
定石だが、本パッケージでは以下の理由で **N_fock=2 特化の自前 bit-flip pass**
を選ぶ:

1. **GEMM 呼び出しオーバヘッド**: N_fock=2 では右オペランドが (2, dim/2) の
   非常に細長い行列となり、N_fock が大きい (~20-40) ケースと比べて BLAS の
   per-call overhead が dim に対して相対的に重い (推定で dim < 2^12 程度で
   自前ループが優位)。
2. **中間軸 moveaxis のコピー**: 一般の reshape + GEMM 経路では中間軸を
   末尾に持ってくる `np.moveaxis` + workspace への物理コピーが必要。
   N_fock=2 だと「ペアの swap-with-mix」は連続 / 2-stride アクセスで
   コピー不要で済むため、moveaxis を挟むのは逆に損。
3. **`apply_h_kryanneal` と同じ層に揃える**: Phase 6 の cache
   block-fusion 最適化は両者で共通の bit-flip pass パターンに対して効くため、
   matvec と Trotter を同じ Rust モジュール (`src/matvec.rs`) 上で同型に
   書いておく方が後段の最適化が両者に均等に効く。

Trotter primitive は `src/trotter.rs` に置く想定だが、`apply_single_mode_axis_i`
自体は matvec primitive の隣 (`src/matvec.rs`) でも構わない (Phase 2 で
実装する際に判断する)。

**Phase 6 関連の現状** (`apply_single_mode_axis_i`):

- **C1 (#62, 実装済み)**: rayon `par_chunks_mut` 並列化. `2·mask` block 単位
  で分割し chunk 内で pair update.
- **C3 (#64, 実装済み)**: `trotter_step` 経路では本関数を直接呼ばず
  [`apply_multi_qubit_gate_fused`] (§5.1.3) 経由で連続 FUSE_K=4 qubit を
  1 chunk closure 内で per-axis 逐次 apply する形に書き換えた (端数のみ本関数で
  個別 apply). barrier 多重化解消で N=20 trotter_step が 4.01× 改善.
- **C2.5 (#71, 実装済み)**: SIMD 特化 (i ∈ {0,1,2}) を `wide::f64x4` で導入.
  C2 で `apply_h_kryanneal` 側に作った `simd_kernels::bitflip_iN` と同じ
  helper モジュール (`src/matvec.rs::simd_kernels`) に `single_mode_iN` を
  追加し, 2×2 complex matmul を **complex broadcast + in-register swizzle**
  で f64x4 化した:
  - `u_k_re_v = splat(u[k].re)`, `u_k_im_signed_v = [-u[k].im, u[k].im, -u[k].im, u[k].im]`,
    `x_swap = re/im swap` を用いて `u_k · x_pair = u_k_re_v · x_pair +
    u_k_im_signed_v · x_swap` を 2 lane (2 Complex64) 並列で計算.
  - i=0 は 1 block = 2 Complex64 しか入らないので 2 連続 block (= 4 Complex64
    = 8 f64) を 1 SIMD iter で処理 (vperm2f128 で `[a0,b0,a1,b1]` を
    `A=[a0,a1], B=[b0,b1]` に deinterleave → 書き戻し時に再 interleave).
  - i=1, 2 は lo_half / hi_half が f64x4 の倍数で並ぶので deinterleave 不要.
  - `apply_single_mode_axis_i_serial` / `_rayon` の両 path + C3 の
    `apply_fused_axes_to_chunk` inner kernel から共通で呼び出され, C3 で得た
    barrier 多重化解消の効果に SIMD compute を上乗せできる構造.
  - rayon の `split_at_mut` 退化ケース (block == dim, i = n-1) も SIMD-target
    な i ∈ {0,1,2} では SIMD 経路にフォールスルーする (実用 dim ≥
    MIN_RAYON_DIM では i = n-1 が常に 17 以上で SIMD range 外なのでこの分岐は
    テスト用の小 dim 直接呼び出しでのみ通る).
  - 数値同一性は SIMD ON 両 path で bit-identical, SIMD ON vs scalar 経路は
    `rel < 1e-13` (FMA 折り畳み差で ulp 差が出うる).
  - **chunk_size = 64 を静的に維持**: `apply_h_kryanneal_rayon` /
    `apply_multi_qubit_gate_fused_rayon` のような動的計算
    `(dim/(nth·4)).clamp(MIN, MAX)` には**しない**. 理由は本関数が 1 chunk
    あたり 1 pass の read-write しかなく (axis pair update のみで chunk 内
    data reuse なし), chunk が L1 (32 KB) を spill した瞬間に SIMD が
    memory bandwidth bound に転落するため. PR #80 fixup experiment で
    chunk_size=4096 (= 64 KB, L1 spill) にすると N=20 SIMD speedup が
    2.95× → 0.56× へ大幅 regression する観測あり.  L1 fit する 1 KB chunk
    を多数並べる方が SIMD throughput を最大化できる. これにより rayon
    scheduling overhead が大きくなって N=18 の SIMD speedup が ~1.0× で
    頭打ちになる副作用はあるが (§12 C2.5 観測, 構造的境界), N=16 serial と
    N=20 rayon の主要 size 帯では 1.9-3.5× を達成する.

#### 5.1.3 Phase 6 C3 — DRAM 律速の解消 (multi-qubit gate fusion + phase_p 並列化)

Phase 6 C1 (rayon) + C2 (SIMD) を入れた後の本番 bench (issue #68 follow-up)
で観測された制約:

| kernel @ N=20 | 1 thread | 64-thread peak | 飽和原因 |
|---|---|---|---|
| `apply_h_kryanneal` | 23.8 ms | **6.13× (64 threads)** | DRAM bandwidth 上限 (理論 64× の ~9.6%) |
| `trotter_step` | 54.8 ms | **1.55× (16 threads)** | per-step rayon barrier × 2n が compute 量を食い潰す |

両ボトルネックの本質は **「v / psi に対する DRAM round trip 回数」が compute
量に比して過大** であること: C1/C2 で per-pass 計算密度を上げても, memory
bound に当たれば飽和する。Phase 6 C3 はこの memory traffic 自体を削減する
最適化レイヤで, 以下の 2 つを **独立した** サブ最適化として扱う:

##### A. trotter_step / cfm4_step の barrier 多重化解消 (multi-qubit gate fusion)

`trotter_step` は `Π_{i=0..n} R_i(dt)` を 1 軸ずつ in-place で適用するため
per-step rayon barrier が **2n** 個入る (`apply_single_mode_axis_i` × n + diag
phase × 2)。N=20 で 1.55× で頭打ちになるのはこの barrier overhead が
compute 短縮分を食い潰すため (`apply_single_mode_axis_i` 1 回の compute
≈ 2.5 ms に対し rayon barrier ~ms 級, ratio 1:1 程度)。

**初期試行 (PR #78 初版, 放棄)**: 連続 k 個の `R_i` を **tensor product**
`R_{i+k-1} ⊗ ... ⊗ R_i` の `2^k × 2^k` dense unitary に畳んで chunk closure
内で 1 回の dense matmul (qsim `MultiQubitGateFuser` / Häner-Steiger 2017 §3.3
同型) で apply する方針を採った. しかし Linux 本番 bench で **N=20 で
`trotter_step` が 0.81× regression** した. 原因: per-axis × k の compute は
`2k·dim` ops だが dense matmul は `2^k·dim` ops で k=4 のとき 2× 多く,
TFIM 規模では memory-bandwidth gain よりも compute 増のほうが勝ったため.

**現実装 (per-axis 逐次)**: tensor product matmul を諦め, **chunk closure 内で
k 個の axis に対して per-axis 2-pair update を逐次** 実行する形に変更.
kryanneal の `H_drv = -Σ h_x_i X_i` は **per-site で commuting** なので
`exp(+i·a_t·dt·Σ_{j∈G} h_x_j X_j) = Π_{j∈G} exp(+i·θ_j X_j)` を逐次適用しても
exact (Trotter 誤差は導入しない). compute は per-axis × k と同じ
(`2k·dim` ops, 増えない), barrier は 1 per fused call (`n/k 倍` 削減),
chunk が L2 fit な間に全 k pass を完走することで DRAM round trip 削減
の効果を狙う.

擬似コード (`src/matvec.rs::apply_multi_qubit_gate_fused`):

```rust
/// `psi` の連続 k qubit (i_start, ..., i_start+k-1) に 2×2 ユニタリ列 u_list
/// を per-axis 逐次で in-place 適用する.
pub(crate) fn apply_multi_qubit_gate_fused(
    psi: &mut [Complex64],
    u_list: &[[Complex64; 4]],   // k 個の row-major 2×2 unitary
    i_start: usize,
    n: usize,
) {
    let k = u_list.len();
    let group_block = 1usize << (i_start + k);     // 最大 axis の block 2 倍
    // chunk_size を thread 数に応じた動的計算 (apply_h_kryanneal_rayon と同型)
    // + group_block (最大 axis の block 2 倍) の整数倍に揃える. PR #78 v2 で
    // 固定 RAYON_CHUNK_MAX を使ったところ N=18 で chunk 数 < 64 thread になり
    // 0.84× regression したため動的化 (PR #78 v3).
    let nth = rayon::current_num_threads().max(1);
    let target = (psi.len() / (nth * 4)).clamp(RAYON_CHUNK_MIN, RAYON_CHUNK_MAX);
    let chunk_size = if group_block >= target {
        group_block
    } else {
        (target / group_block).max(1) * group_block
    };
    psi.par_chunks_mut(chunk_size).for_each(|chunk| {
        // chunk 内で k 個の axis を順次 apply (per-axis 2-pair update).
        for (j, u) in u_list.iter().enumerate() {
            let i = i_start + j;
            let mask = 1usize << i;
            let block = mask << 1;
            let mut base = 0;
            while base + block <= chunk.len() {
                let (lo, hi) = chunk[base..base + block].split_at_mut(mask);
                for (a, b) in lo.iter_mut().zip(hi.iter_mut()) {
                    let (av, bv) = (*a, *b);
                    *a = u[0]*av + u[1]*bv;
                    *b = u[2]*av + u[3]*bv;
                }
                base += block;
            }
        }
    });
}
```

**k の選択方針** (qsim の経験値 `max_fused_size = 4-5` を踏襲):

- `k = 4`: barrier 削減 4× と chunk-resident cache 効果の bench でのバランス点
  として default. inner u_list 配列 `[[Complex64; 4]; 4]` = `128 B` で stack 確保.
- `k <= MAX_FUSED_K = 6`: hard 上限 (実用上の cost / benefit).

kryanneal は `k = 4` を default とし `bench_block_fusion.py` で sweep で確認.

**trotter_step の書き換え** (`src/trotter.rs`):

```rust
const FUSE_K: usize = 4;

pub(crate) fn trotter_step(psi, h_x, h_p_diag, a_t, b_t, dt, n) {
    // ... 前半 phase_p(dt/2) (serial loop, 別 issue でも並列化検討) ...

    // 連続 FUSE_K qubit ごとに R_i を fuse して 1 barrier で適用.
    let mut i = 0;
    let mut u_list = [[Complex64::new(0.0, 0.0); 4]; FUSE_K];
    while i + FUSE_K <= n {
        build_axis_unitaries(&h_x[i..i + FUSE_K], a_t, dt, &mut u_list);
        apply_multi_qubit_gate_fused(psi, &u_list, i, n);
        i += FUSE_K;
    }
    // 端数: n mod FUSE_K 個の qubit を per-axis で apply.
    while i < n {
        ...
        apply_single_mode_axis_i(psi, &u, i, n);
        i += 1;
    }

    // ... 後半 phase_p(dt/2) ...
}
```

**期待される barrier 削減**:

- 現状: `2n + 2` barrier @ trotter_step (phase_p × 2 + axis_i × n + 再 diag).
  N=20 で **40+**.
- 改修後: `n/k + 2 + (n mod k)` ≈ **`n/k + 2`** (端数なしケース).
  N=20, k=4 で **7** (phase_p × 2 + 5 fused gate sweep).

apply_h_kryanneal 経路 (cfm4_step / m2_midpoint_step 内) は既存実装で 1 回の
matvec = 1 barrier なので追加の fusion は不要 (上記スコープ A は trotter
経路のみ).

##### B. `trotter_step` の phase_p を rayon 並列化

`trotter_step` の前後 2 回の `phase_p(dt/2)` (`psi[k] *= exp(-i·b_t·h_p_diag[k]·dt/2)`)
は scalar serial loop だった. dim=2^20=1M で各 phase pass あたり数 ms 級
のコストがあり, A (multi-qubit gate fusion) で bit-flip pass の barrier が
削減された後はこちらが per-step time の支配項になる. 解決:

```rust
fn apply_phase_p(psi: &mut [Complex64], h_p_diag: &[f64], b_t: f64, dt_half: f64) {
    if psi.len() >= PHASE_RAYON_MIN_DIM {   // = 1 << 17, MIN_RAYON_DIM と同じ
        psi.par_iter_mut().zip(h_p_diag.par_iter()).for_each(|(psi_k, &h_p_k)| {
            let phi = -b_t * h_p_k * dt_half;
            let (s, c) = phi.sin_cos();
            *psi_k *= Complex64::new(c, s);
        });
        return;
    }
    // scalar fallback
    ...
}
```

各 k は独立な multiplicative update なので rayon 並列でも **bit-identical**
を保つ (任意 scheduling で同じ結果).

##### C. `apply_h_kryanneal` の chunk_size: 旧値維持

PR #78 初版で `RAYON_CHUNK_MAX` を `1 << 14` (= 256 KB) → `1 << 13` (= 128 KB)
に縮める変更を入れたが, Linux 本番 bench で `apply_h_kryanneal` の **N=18
で 0.69×, N=20 で 0.91× regression** という mixed な結果になった (小 dim
で並列度が落ちる方が L2 fit 改善より影響大). **旧値 `1 << 14` に戻す**.

L2 pressure は既存の動的 chunk_size 計算 `(dim / (nth * 4)).clamp(MIN, MAX)`
の中で 64 thread 環境では target が MAX 未満になるため間接的に効く設計に
依存する (大 thread 数で自動的に小 chunk になる).

A (multi-qubit gate fusion) の rayon 実装でも同じ動的 chunk_size 計算を
踏襲し group_block の整数倍に揃える pattern を採用 (PR #78 v2 で
chunk_size を固定 `RAYON_CHUNK_MAX` にしていたのを動的化, N=18 並列度
不足の regression を修正).

##### Acceptance と bench (実測結果)

- bench: `benchmarks/bench_block_fusion.py` で N ∈ {18, 20, 22} の per-step
  time を `trotter_step` / `apply_h_kryanneal` 両方計測.
- baseline: Phase 6 C2 完了時点 (`main` branch tip, PR #78 merge 前).
- acceptance: N=20 で `trotter_step` の per-step time が **>= 1.3×** 改善
  (issue #64 当初目標). 達成.
- **実測 (Linux x86_64, cpu_count=64, RAYON_NUM_THREADS=64, BLAS=1, PR #78 v3)**:

  | N | `trotter_step` speedup | `apply_h_kryanneal` speedup |
  |---|---|---|
  | 18 | **1.55×** | 0.94× (誤差範囲) |
  | 20 | **4.01×** ✅ | 0.94× (誤差範囲) |
  | 22 | **2.93×** | 1.01× |

  `trotter_step` は当初目標 1.3× を **3 倍以上クリア** (N=20 で 4.01×).
  `apply_h_kryanneal` は本 issue で touch せず ≈ 1.0× (regression なし).
  apply_h_kryanneal の DRAM bandwidth 改善は follow-up issue #79 (Phase 6 D)
  で試行したが本 Linux 本番環境では未採用 (perf 計測で IPC 2.98 baseline が
  既に compute-near-peak と判明, 詳細 §5.1.4).
- 数値一致: `cargo test` + `uv run pytest` で `rel < 1e-13` 確認済み.

##### 設計判断の論拠と参考実装

- **連続 k qubit 限定 (任意 qubit 集合の fusion は対象外)**: kryanneal の
  TFIM は per-site `H_drv` で qubit i は physical bit index i に固定. trotter
  経路は per-axis に逐次適用するため `R_i` の順序は **physical order
  i=0..n** で確定済み. qsim の `_pdep_u64` を使った任意 qubit 集合 fusion は
  汎用回路シミュレータの要件で, kryanneal では不要.
- **`apply_h_kryanneal` 側に gate fusion を入れない**: matvec は **sum of
  X_i + diag** で, 各 X_i は他 qubit に identity. これを tensor product
  に畳むと `Σ_i (I ⊗ ... ⊗ X_i ⊗ ... ⊗ I)` のままで 2^k × 2^k matrix と
  しての利点が出ない. 既存の per-axis bit-flip pass + chunk-resident な
  cache fusion (C1) が apply_h_kryanneal の最適解.
- **参考実装**: cv-ising-solver には対応する block-fusion 実装が **無い**
  (CV 版は連続 slab matvec で TFIM 固有の高 stride 問題が起きないため).
  qsim (`lib/simulator_avx.h::ApplyGateH<H>`, `lib/fuser_mqubit.h`) と
  Häner-Steiger 2017 (arXiv:1704.01127 §3.2-3.3) が一次根拠.

#### 5.1.4 Phase 6 D 実験アーカイブ — `apply_h_kryanneal` の DRAM bandwidth 改善試行と未採用 (issue #79, 2026-05-17)

issue #68 で `apply_h_kryanneal` が N=20 / 64 threads で **6.13× scaling
飽和** (理論 64× の 9.6%) と観測されたのを DRAM bandwidth 上限と解釈し,
Phase 6 D で連続 k 個の高 i (mask ≥ chunk_size) を **group-fused 3-phase
形** に書き換える試行を行った. しかし **本番 Linux サーバー (AMD EPYC 7713P
64 物理コア, L2 = 512 KB/core, L3 = 32 MB/CCX × 8 CCX = 256 MB) で
perf 計測した結果, 仮定が誤りで真の compute regression が確認**, revert.

本節は今後同種の最適化を再検討する際の判断材料として実験記録を残す.

##### 試行した設計 (revert 済み)

`apply_h_kryanneal_rayon` を 3 phase に分割し, 2^fused_k 個の連続 chunk を
1 つの super-chunk (= group) にまとめて 1 thread に渡す:

1. **Phase 1**: per-chunk diag + low-i (`i < chunk_log`, mask < chunk_size).
   partner は chunk-internal で L1 resident, 既存 C1 と同型.
2. **Phase 2**: group-fused 高 i (`i ∈ [chunk_log, i_split)`). partner は
   同じ group 内の別 chunk → L2 resident で完結 (DRAM 再 load 回避).
3. **Phase 3**: per-chunk 残り高 i (`i ≥ i_split`). partner は別 group の
   chunk → 従来通り DRAM 経由.

各 `y[k]` への accumulation 順序は `diag → i=0..n-1` で C1 / serial と
完全一致させ bit-identical を維持. 期待した DRAM v traffic 削減は
`dim · (1 + h_baseline) → dim · (1 + h_naive)`, `h_naive = h_baseline -
fused_k`. N=20 で fused_k=2 → 理論改善率 (1+8)/(1+6) = 9/7 ≈ 1.29×.

##### 観測結果 (Linux AMD EPYC 7713P, 64 threads, RUSTFLAGS="-C target-cpu=native")

`src/bin/perf_apply_h.rs` (純 Rust binary, `apply_h_kryanneal` を 500 回呼ぶ)
で baseline (main = C1) と after (本 PR = Phase D) を `perf stat` 計測:

| N | C1 baseline per-iter | Phase D after per-iter | Δ |
|---|---|---|---|
| 18 | 0.261 ms | 0.274 ms | +5% (誤差範囲) |
| 20 | **0.705 ms** | **1.060 ms** | **+50% (regression)** |
| 22 | 4.584 ms | 5.264 ms | +15% (regression) |

hardware counter (N=20 が代表):

| Metric | C1 baseline | Phase D after | 変化 | 解釈 |
|---|---|---|---|---|
| **IPC** | **2.98** | **1.80** | **-40%** | C1 はほぼ compute peak (Zen 3 max ~4-5 IPC) |
| cycles (G) | 53.6 | 83.1 | +55% | Phase D が 30G cycles 余計 |
| instructions (G) | 159.9 | 149.9 | -6% | Phase D は命令数自体は減 |
| **L2 fill wait cycles (G)** | **38.7** | **59.3** | **+53%** | L2 miss → L3/DRAM 待ち cycle が爆増 |
| L2 ic_dc miss (M) | 198 | 236 | +19% | L2 miss 数も増 |
| L2 wait / L2 miss = avg latency | 195 cycles | **251 cycles** | **+30%** | per-miss latency 自体が劣化 |
| L1d miss % | 13.6% | 17.1% | + | L1 効率も悪化 |
| branch-misses (M) | 37 | 83 | **+123%** | 3-phase 分岐で mispredict 倍増 |
| cache-miss (= L3 miss approx) | 2.65% | 3.00% | + | **DRAM access はそもそも少ない** |

##### 判定: 「DRAM bandwidth bound」仮説が誤りだった

1. **C1 baseline は既に compute-near-peak**. IPC 2.98 は Zen 3 理論 max の
   60-75% を実用化しており, i 外側 / k 内側の超予測可能な stride アクセスが
   HW prefetcher と L1/L2/L3 階層に完璧にフィットしていた.
2. **cache-miss rate 2.6-3.3%** で **DRAM access はそもそも少ない**.
   issue #68 の 6.13× scaling 飽和の真因は DRAM bandwidth ではなく,
   L2 fill latency (L3 / cross-CCX レイテンシ) の per-thread 並列度限界
   だった可能性が高い.
3. **Phase D の 3-phase chunk 跨ぎ XOR access pattern が prefetcher を破壊**.
   per-L2-miss avg latency が 195 → 251 cycles (+30%) に増加し,
   traffic 削減のはずが latency 悪化で打ち消され net regression.
4. **Python bench (`bench_block_fusion.py`) の N=18 で観測した 0.53× 大幅
   regression は alloc / GC noise** だった (perf binary では N=18 は ≈ neutral).
   `apply_h_kryanneal_py` が毎回 64 MB pyarray を alloc/copy する overhead が
   wall time の大半を占めるため, Rust 側 micro-optimization の検証には不適.
   **以降の micro 最適化検証は perf binary を使う**.

##### 残された代替カード (B/C/D) も再評価不要と判断

issue #79 本文に列挙されていた alternative:

- **B (SIMD i ≥ 3 拡張)**: IPC 2.98 baseline に対し更なる SIMD は意味なし
  (compute はすでに peak 付近).
- **C (prefetch)**: HW prefetcher が予測可能 stride で完璧に効いている
  baseline には寄与なし.
- **D (streaming store)**: cache-friendly な write pattern を cache bypass
  にすると逆効果 (現状 y_chunk は L1 resident で連続書き).

いずれも「Phase D で DRAM bound が確認されていれば」前提に立っており,
**本 hardware では効果が薄い** ため別 sub-issue 化していない. 将来 hardware
が変わった (例: より大規模 N で真に DRAM bound になる) ケースで再検討する.

##### 残した資産

- **`src/bin/perf_apply_h.rs`** + `_rust::bench_api::apply_h_kryanneal`
  re-export: 今回の hardware counter 計測に使った pure-Rust 計測 binary.
  Python bench の alloc noise を排した micro-optimization 検証経路として
  残す. 使い方は `CLAUDE.md`「perf 計測用 binary」節.
- **本節 (§5.1.4)**: Phase D の設計と perf 計測結果を archive. 同種の
  「DRAM bound 仮説に基づく fusion 最適化」を再提案する際は, まず perf
  binary で hardware counter を取って bottleneck を hardware に確認してから
  進める運用.

##### 設計判断の論拠

- **revert を選択した理由**: bench (Python) と perf (hardware counter) の
  両面で N=20 regression が確認されたため. compute は既に peak に近いので
  「access pattern を変えて traffic を減らす」最適化はほぼ無意味であり,
  さらに pattern を崩すと prefetcher 効率まで失う負の連鎖が観測された.
- **perf binary を残した理由**: 同じ過ち (Python bench の noise を真の
  regression と誤認) を避けるため. 本実験で「Python bench は Rust 側
  micro-optimization の評価に不向き」という重要な運用知見を得た.
  perf binary は将来の Phase 6 改善の検証基盤として価値がある.
- **issue #79 を re-open しない理由**: 本 hardware (AMD EPYC 7713P) では
  apply_h_kryanneal の「compute 効率」自体が改善余地ない. 別 hardware
  または大規模 N で真の DRAM bound が観測されたら新 issue を立てる方が
  scope が明確.

