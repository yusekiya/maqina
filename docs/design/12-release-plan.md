# §12. 段階リリース計画

バージョニングポリシー (Phase N → v0.N bump, umbrella issue DoD 必須項目)
は `docs/conventions.md` §2 を一次資料とする.

### Phase 1: MVP / scalar baseline (~v0.1)

- `IsingProblem`, `Schedule`, `QuantumAnnealer.run(method="m2")` のみ
- `Schedule` プリセット: `linear` / `from_callable` / `reverse` / `pause`
  (reverse annealing と pause schedule は研究用途で頻出するため Phase 1
  時点で同梱)
- Rust 拡張: `apply_h_kryanneal`, `lanczos_propagate`, `m2_midpoint_step`,
  `tridiag_eigh` (hand-rolled QL)
- Python リファレンス (`_python_*`) との等価性テスト
- 小規模 QuTiP 比較テスト
- **スカラ単スレッド・SIMD 明示利用なし** で実装。以降 (Phase 2 以降) の
  高速化施策の baseline として `bench_per_step.py` の数値を確定させる
- BLAS feature ON/OFF は両方ビルド可能だが、Level-1/2 ops が呼ばれるのは
  Lanczos 内部のみで、matvec / bit-flip pass は自前のスカラループ

### Phase 2: Trotter 経路 (~v0.2)

横磁場演算子 X_i の bit-flip 性と可換性 (`[X_i, X_j] = 0`) を活用し、
`exp(-i dt H_drv) = Π_i R_i(dt)` を **Lanczos を経由しない閉形式の
2×2 rotation で逐次適用** する経路。Strang 2 次 Trotter:

```
U(dt) ≈ phase_p(dt/2) · (Π_i R_i(dt)) · phase_p(dt/2)
```

- Rust 側に `apply_single_mode_axis_i` を新規実装 (詳細 §5.1.2):
  - `(psi[k], psi[k ^ (1<<i)])` ペアに 2×2 ユニタリを in-place 適用
  - N_fock=2 特化の自前 bit-flip pass で書く (一般的な reshape + GEMM
    パターンを採らない根拠は §5.1.2 末尾)
  - Phase 2 ではスカラ単スレッド (SIMD/threading は Phase 6 で乗せる)
- Rust 側に `trotter_step` (Strang 1 step エントリ) を新規実装
- 4 次 Suzuki (Trotter-Suzuki S_4) はオプションで追加可
- `method="trotter"`, `method="trotter_suzuki4"`
- Phase 1 の M2 と精度・速度を同一マシンで比較 (`bench_per_step.py` 拡張)。
  Trotter は per-step が ~(N+1)·dim flops と軽い反面 2 次精度なので
  「短時間 / 緩やかな schedule」で M2 / CFM4:2 比優位、「長時間 / 高精度」では
  CFM4:2 が勝つ、というクロスオーバを実測で示す

### Phase 3: CFM4:2 (~v0.3)

- `cfm4_step`, `method="cfm4"` 経路
- 線形結合 callback 形式 (§5.2 末尾) で per-step matvec を 4m → 2m に削減

### Phase 4: Adaptive (~v0.4)

- `cfm4_step_with_m2_estimate` (embedded M2 error)
- `cfm4_step_with_richardson_estimate` (step-doubling Richardson)
- Python 側 PI controller driver
- `method="cfm4_adaptive_richardson"`

### Phase 5: Simulator & Observables (~v0.5)

- `Observable` クラス, Z 基底対角 Hermitian 観測量 (issue #46, 済)
- `QuantumAnnealer.run` の `observables` / `save_tlist` / `store_states`
  引数を有効化 (issue #47, 済). `QuantumResult` に `times` / `states` /
  `probabilities` フィールドを追加. 詳細は §4.4. `save_tlist=None` は
  最節約モードで状態保存なし; 非 None で指定時刻を厳密に踏み (固定 dt:
  step boundary に merge, adaptive: PI dt クランプ), 観測量時系列と
  (オプションで) ψ スナップショットを記録する.
- `AnnealingSimulator` step-wise API (issue #48, 済). 中間時刻まで進めて
  `Observable` で測ってから続きを発展させる用途. `step(dt)` / `advance_to(t)` /
  `measure(observable)` で `QuantumAnnealer.run` と同じ propagator 集合を
  逐次操作する. 固定 dt 経路は `run` と bit-identical な数値 (`rel < 1e-13`).
  `QuantumAnnealer.create_simulator(psi0, t0, *, method=...)` が簡便 factory.
  詳細は §4.5.
- `instantaneous_eigenstates(problem, schedule, t, k, method, ...)` (issue #49, 済).
  瞬時 H(t) の下位 k 固有値・固有状態を返す. `method="lanczos"` (default,
  Krylov shift-invert + 自前 hand-rolled implicit QL) と `method="exact"`
  (`n <= 12` の dense `eigh` 経路) の 2 経路. 詳細は §4.7.

### Phase 6: 並列化 + 仕上げ (~v0.6)

Phase 1-5 でアルゴリズム面の機能が出揃った時点で実装面の並列化に着手する。
Phase 1 の baseline と比較できることが本 phase の前提。

- **L2 並列化 (C1, issue #62, 実装済み)**: matvec / Trotter primitives の
  bit-flip pass を rayon `par_chunks_mut` で並列化。`apply_h_kryanneal` と
  `apply_single_mode_axis_i` の両方が対象 (CFM4:2 / Trotter どちらの経路でも
  効く)。前者は cache-blocked 形 (chunk 内で diag + 全 i fuse), 後者は
  `2·mask` block 単位の par_chunks_mut + 退化ケース `i=n-1` で split_at_mut
  ペア並列。`feature = "rayon"` (default ON, `--no-default-features` で
  scalar 単スレッドフォールバック)。
- **SIMD (C2, issue #63, 実装済み)**: `wide` クレート (stable Rust + 自動
  target_feature 切替) で f64x4 (AVX2 / AVX-512 / NEON) を使い, `apply_h_kryanneal`
  の bit-flip pass の i ∈ {0, 1, 2} (stride 1/2/4 連続アクセス領域) を
  SIMD 特化。i ≥ 3 は scalar inner loop のまま (stride ≥ 8 で SIMD vectorize
  の利得が小さく cache line を跨ぐ)。`feature = "simd"` (default ON,
  `--no-default-features` で scalar fallback)。
  SIMD inner kernel (`simd_kernels::bitflip_i0` / `_i1` / `_i2`) は
  `apply_h_kryanneal_serial` と `apply_h_kryanneal_rayon` の両 path から
  共通で呼ばれ, rayon path では chunk_size を `SIMD_BLOCK_MAX = 8` Complex64
  の倍数に丸めて block-aligned 前提を満たす。SIMD load/store は
  `std::ptr::read_unaligned::<f64x4>` / `write_unaligned::<f64x4>` で AVX
  `vmovupd` 1 命令に折り畳み, compute は `f64x4::mul_add` で `vfmadd231pd`
  (FMA-enabled CPU) に折り畳む (PR #74)。SIMD 経路と scalar 経路は各
  `y[k]` への単一 `coeff * v[k^mask] + y[k]` を独立 lane で並列実行するため
  **両ビルドで bit-identical または rel < 1e-13** な数値結果を返す
  (`apply_h_kryanneal_simd_matches_scalar_fuzz_100iter` テスト, src/matvec.rs)。
  併せて `coeff == 0` (= `h_x[i] == 0` または `a_t == 0`) で i pass を完全
  スキップする短絡 (PR #73) も入っており, sparse h_x で SIMD 効果が最も
  ROI 高く見える `i012-focus` mode bench が成立する。
  `apply_single_mode_axis_i` の SIMD 特化 (2×2 complex matmul の broadcast +
  swizzle pattern) は **C2.5 (#71) で実装済み** (`simd_kernels::single_mode_iN`,
  下記参照).
  実 SIMD 速度向上は build 時の `target-cpu` 設定に依存し, default `x86_64`
  target では `wide` が scalar fallback を選び正確性のみ提供する
  (`benchmarks/bench_simd_scaling.py` の本番 sweep は
  `RUSTFLAGS="-C target-cpu=native"` を前提)。
  **観測 (Linux x86_64, cpu_count=64, OpenBLAS, AVX2 + FMA, DDR4 multi-channel,
  `RAYON_NUM_THREADS=64`, BLAS thread=1, 3 runs × repeat=50 median)**:
  per-pass SIMD speedup ≈ **1.75×** (理論 3-4× には届かないが scalar 比で
  確実に高速化), `i012-focus` mode (3 SIMD pass + 1 diag pass) の total
  speedup ≈ **1.28×** (N=18, 中央値 of 3 runs)。issue #63 の当初 acceptance
  「N=18 i012-focus ≥ 1.5×」は **未達**。 主因は (1) DRAM bandwidth 上限への
  per-pass の頭打ち, (2) SIMD 化していない diag pass による希釈 (4 pass 中
  1 つが scalar), (3) 大 dim multi-thread の per-call ms オーダ計測が背景負荷
  に強く依存し inter-run 変動 が ~5× に達することがある, の 3 つ。 acceptance
  ギャップ (apply_h_kryanneal の memory traffic 削減) は Phase 6 C3 (#64) では
  trotter_step 専用の最適化を取り `apply_h_kryanneal` 自体は touch しなかった
  ため未解消で残る (#64 で trotter_step は N=20 で 4.01× 改善達成)。
  apply_h_kryanneal 側の DRAM bandwidth 改善は **follow-up issue #79
  (Phase 6 D, open)** として切り出し (高 i sparse fused matvec / SIMD i≥3 拡張 /
  prefetch / streaming store の候補を sweep する形)。
  bench は PR 本体に含めず, 個別 issue (#63) コメントとして添付済み
  (issue #47 で確定した運用)。
- **cache block-fusion (C3, issue #64, 実装済み, PR #78 merged)**: DRAM 律速
  と barrier 多重化の解消を目的とする最適化レイヤ。issue #68 follow-up bench で
  `apply_h_kryanneal` が 6.13× (理論 64× の ~9.6%), `trotter_step` が 1.55×
  (rayon barrier × 2n の overhead) で頭打ちと判明していたうち, **本 issue では
  trotter_step 側を集中改善** (`apply_h_kryanneal` の DRAM bandwidth 改善は
  follow-up #79 として切り出し). C3 は以下 2 つの独立サブ最適化で構成
  (§5.1.3 を一次資料):
  - **A. multi-qubit gate fusion (per-axis 逐次経路)**: 連続 k qubit (default
    k=4) の R_i を **1 つの rayon chunk closure 内で per-axis 2-pair update を
    逐次** 実行. trotter_step の barrier 数を `2n+2` → `n/k + 2` に縮める.
    kryanneal の `H_drv = -Σ h_x_i X_i` が **per-site commuting** で逐次適用が
    exact (Trotter 誤差なし) という TFIM 固有の性質を活用. 初版 (PR #78 v1) は
    qsim 流 dense 2^k × 2^k matmul を採用したが Linux 本番 bench で
    `trotter_step` が **0.81× regression** したため per-axis 逐次経路に
    切替 (dense matmul の compute は per-axis × k の 2× 重く, TFIM 規模では
    memory-bandwidth gain よりも compute 増が勝った). 詳細は §5.1.3.
  - **B. phase_p の rayon 並列化**: `trotter_step` の前後 2 回の
    `phase_p(dt/2)` (`psi[k] *= exp(...)`) を rayon `par_iter_mut` で並列化.
    A 後の支配項 (dim=1M で数 ms 級) を 64 thread に分散させる.
    各 k 独立 multiplicative update なので bit-identical を維持.
  - **C. chunk_size 戦略**: PR #78 初版で `RAYON_CHUNK_MAX` を `1<<14` →
    `1<<13` に縮める変更を入れたが N=18 / N=20 で regression したため
    旧値に復活. A の fused 経路でも `apply_h_kryanneal_rayon` と同じ
    動的 chunk_size `(dim/(nth*4)).clamp(MIN, MAX)` を採用 (group_block の
    整数倍に揃える形で).

  **実測 (Linux x86_64, cpu_count=64, BLAS=1)**: `trotter_step` の per-step
  time が N=18 で 1.55×, **N=20 で 4.01× (acceptance 1.3× の 3 倍以上)**,
  N=22 で 2.93× の改善. `apply_h_kryanneal` は本 issue で touch せず
  ≈ 1.0× (regression なし).
- **C2.5 (issue #71, 実装済み)**: `apply_single_mode_axis_i` の SIMD 特化
  (i ∈ {0,1,2}). C2 で `apply_h_kryanneal` 側のみ SIMD 化したため,
  trotter_step 経路の axis_i も `wide::f64x4` で SIMD 化する follow-up.
  `simd_kernels::single_mode_iN` を 2×2 complex matmul の **complex
  broadcast + swizzle pattern** で実装し
  (`u_k · x_pair = splat(u[k].re) · x_pair + [-u[k].im, u[k].im, ...] · x_swap`
  を 2 Complex64 並列で実行する形, §5.1.2 参照), `apply_single_mode_axis_i_serial`
  / `_rayon` の両 path + C3 の `apply_fused_axes_to_chunk` inner kernel から
  共通で dispatch する. これにより C3 で得た N=20 trotter_step 4.01× の上に
  SIMD compute を上乗せできる構造になった.

  **観測 (Linux x86_64, cpu_count=64, OpenBLAS, AVX2 + FMA,
  `RAYON_NUM_THREADS=64`, BLAS thread=1, repeat=50, PR #80 final bench)**:

  | N | path | i=0 | i=1 | i=2 |
  |---|---|---|---|---|
  | 16 (serial) | scalar fallback | **2.33× / 2.43×** | **2.16× / 2.27×** | **1.97× / 1.88×** |
  | 18 (rayon)  | par_chunks_mut  | 0.97× / 0.95× | 0.96× / 1.00× | 0.94× / 1.01× |
  | 20 (rayon)  | par_chunks_mut  | **2.95×**     | **3.48×**     | **2.71×**     |

  N=18 だけ ~1.0× で頭打ちなのは C2 (issue #63) と同じ **cache-hierarchy /
  rayon-scheduling 境界** の現象: dim=2^18 (= 4 MB Complex64) は L3 fit する
  size で memory bandwidth bound に当たらず, かつ 64 thread × chunk_size=64
  (= 1 KB / chunk, L1 fit する選択) の rayon scheduling overhead が
  per-pass compute と同オーダになる. SIMD で compute を縮めても scheduling
  overhead が支配的になり speedup が出ない (絶対時間 simd-off 0.79 ms /
  simd-on 0.82 ms で +3.7%, ノイズの域).

  **chunk_size を `apply_h_kryanneal_rayon` と同じ動的計算
  `(dim/(nth·4)).clamp(...)` に変えると N=20 で chunk_size = 4096 (= 64 KB)
  となり L1 (= 32 KB) を spill, SIMD が memory bandwidth bound に転落して
  N=20 が 2.95× → 0.56× に大幅 regression する** ことが PR #80 の fixup
  experiment で判明 (`apply_single_mode_axis_i` は 1 chunk あたり 1 pass の
  read-write しかないので, chunk 内 data reuse がある `apply_h_kryanneal`
  (chunk あたり diag + n bit-flip = n+1 pass) と最適 chunk_size が異なる).
  C2.5 では **chunk_size = 64 を静的に維持** する判断とした.

  acceptance「N=18 i=0,1,2 で per-pass ≥ 1.5×」は **N=16 (serial) と N=20
  (rayon 主要 size) で達成** (1.88-2.43× / 2.71-3.48×), N=18 は構造的
  境界として limitation 扱い. C3 (#64) の trotter_step fusion 経路は
  default n_qubit ≥ 17 で fused 経路に乗るため, fused chunk_size が大きい
  方向 (C3 では k=4 pass 連続適用で data reuse あり) で C2.5 の SIMD
  inner kernel が効く.

  bench 結果は PR #80 コメントに添付済み.
- **D (issue #79, 試行・未採用)**: `apply_h_kryanneal` の DRAM bandwidth
  改善を狙って group-fused 3-phase 形を試行したが, 本番 Linux サーバー
  (AMD EPYC 7713P) の perf 計測で **C1 baseline は IPC=2.98 で既に
  compute-near-peak**, 「DRAM bound」前提が誤りと判明. Phase D の
  3-phase access pattern が HW prefetcher を破壊し N=20 で +50% compute
  regression を確認 (詳細 §5.1.4). B (SIMD i≥3), C (prefetch), D
  (streaming store) も同じく compute-bound baseline では効果薄と判断,
  別 sub-issue 化していない. 残した資産: `src/bin/perf_apply_h.rs`
  (hardware counter 計測用 binary).
- 物理コア数 vs スループットの sweep をベンチに含め、メモリ帯域律速点を
  明示する (`benchmarks/bench_parallel_scaling.py`, Phase 6 C1 で導入)
- BLAS feature ON/OFF の数値一致 CI (両ビルドで rel < 1e-13) — C4 (#65, open)
- 大規模 QuTiP 比較 (n=12-16 程度まで) — C4 (#65, open)
- ドキュメント整備、Quick start サンプル — C5 (#66, open)

---

