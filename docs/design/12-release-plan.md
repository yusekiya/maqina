# §12. 段階リリース計画

バージョニングポリシー (Phase N → v0.N bump, umbrella issue DoD 必須項目)
は `docs/conventions.md` §2 を一次資料とする.

### Phase 1: MVP / scalar baseline (~v0.1)

- `IsingProblem`, `Schedule`, `QuantumAnnealer.run(method="m2")` のみ
- `Schedule` プリセット: `linear` / `from_callable` / `reverse` / `pause`
  (reverse annealing と pause schedule は研究用途で頻出するため Phase 1
  時点で同梱)
- Rust 拡張: `apply_h_kinema`, `lanczos_propagate`, `m2_midpoint_step`,
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
- `method="cfm4_adaptive_richardson_krylov"`

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
  bit-flip pass を rayon `par_chunks_mut` で並列化。`apply_h_kinema` と
  `apply_single_mode_axis_i` の両方が対象 (CFM4:2 / Trotter どちらの経路でも
  効く)。前者は cache-blocked 形 (chunk 内で diag + 全 i fuse), 後者は
  `2·mask` block 単位の par_chunks_mut + 退化ケース `i=n-1` で split_at_mut
  ペア並列。`feature = "rayon"` (default ON, `--no-default-features` で
  scalar 単スレッドフォールバック)。
- **SIMD (C2, issue #63, 実装済み)**: `wide` クレート (stable Rust + 自動
  target_feature 切替) で f64x4 (AVX2 / AVX-512 / NEON) を使い, `apply_h_kinema`
  の bit-flip pass の i ∈ {0, 1, 2} (stride 1/2/4 連続アクセス領域) を
  SIMD 特化。i ≥ 3 は scalar inner loop のまま (stride ≥ 8 で SIMD vectorize
  の利得が小さく cache line を跨ぐ)。`feature = "simd"` (default ON,
  `--no-default-features` で scalar fallback)。
  SIMD inner kernel (`simd_kernels::bitflip_i0` / `_i1` / `_i2`) は
  `apply_h_kinema_serial` と `apply_h_kinema_rayon` の両 path から
  共通で呼ばれ, rayon path では chunk_size を `SIMD_BLOCK_MAX = 8` Complex64
  の倍数に丸めて block-aligned 前提を満たす。SIMD load/store は
  `std::ptr::read_unaligned::<f64x4>` / `write_unaligned::<f64x4>` で AVX
  `vmovupd` 1 命令に折り畳み, compute は `f64x4::mul_add` で `vfmadd231pd`
  (FMA-enabled CPU) に折り畳む (PR #74)。SIMD 経路と scalar 経路は各
  `y[k]` への単一 `coeff * v[k^mask] + y[k]` を独立 lane で並列実行するため
  **両ビルドで bit-identical または rel < 1e-13** な数値結果を返す
  (`apply_h_kinema_simd_matches_scalar_fuzz_100iter` テスト, src/matvec.rs)。
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
  ギャップ (apply_h_kinema の memory traffic 削減) は Phase 6 C3 (#64) では
  trotter_step 専用の最適化を取り `apply_h_kinema` 自体は touch しなかった
  ため未解消で残る (#64 で trotter_step は N=20 で 4.01× 改善達成)。
  apply_h_kinema 側の DRAM bandwidth 改善は **follow-up issue #79
  (Phase 6 D, open)** として切り出し (高 i sparse fused matvec / SIMD i≥3 拡張 /
  prefetch / streaming store の候補を sweep する形)。
  bench は PR 本体に含めず, 個別 issue (#63) コメントとして添付済み
  (issue #47 で確定した運用)。
- **cache block-fusion (C3, issue #64, 実装済み, PR #78 merged)**: DRAM 律速
  と barrier 多重化の解消を目的とする最適化レイヤ。issue #68 follow-up bench で
  `apply_h_kinema` が 6.13× (理論 64× の ~9.6%), `trotter_step` が 1.55×
  (rayon barrier × 2n の overhead) で頭打ちと判明していたうち, **本 issue では
  trotter_step 側を集中改善** (`apply_h_kinema` の DRAM bandwidth 改善は
  follow-up #79 として切り出し). C3 は以下 2 つの独立サブ最適化で構成
  (§5.1.3 を一次資料):
  - **A. multi-qubit gate fusion (per-axis 逐次経路)**: 連続 k qubit (default
    k=4) の R_i を **1 つの rayon chunk closure 内で per-axis 2-pair update を
    逐次** 実行. trotter_step の barrier 数を `2n+2` → `n/k + 2` に縮める.
    kinema の `H_drv = -Σ h_x_i X_i` が **per-site commuting** で逐次適用が
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
    旧値に復活. A の fused 経路でも `apply_h_kinema_rayon` と同じ
    動的 chunk_size `(dim/(nth*4)).clamp(MIN, MAX)` を採用 (group_block の
    整数倍に揃える形で).

  **実測 (Linux x86_64, cpu_count=64, BLAS=1)**: `trotter_step` の per-step
  time が N=18 で 1.55×, **N=20 で 4.01× (acceptance 1.3× の 3 倍以上)**,
  N=22 で 2.93× の改善. `apply_h_kinema` は本 issue で touch せず
  ≈ 1.0× (regression なし).
- **C2.5 (issue #71, 実装済み)**: `apply_single_mode_axis_i` の SIMD 特化
  (i ∈ {0,1,2}). C2 で `apply_h_kinema` 側のみ SIMD 化したため,
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

  **chunk_size を `apply_h_kinema_rayon` と同じ動的計算
  `(dim/(nth·4)).clamp(...)` に変えると N=20 で chunk_size = 4096 (= 64 KB)
  となり L1 (= 32 KB) を spill, SIMD が memory bandwidth bound に転落して
  N=20 が 2.95× → 0.56× に大幅 regression する** ことが PR #80 の fixup
  experiment で判明 (`apply_single_mode_axis_i` は 1 chunk あたり 1 pass の
  read-write しかないので, chunk 内 data reuse がある `apply_h_kinema`
  (chunk あたり diag + n bit-flip = n+1 pass) と最適 chunk_size が異なる).
  C2.5 では **chunk_size = 64 を静的に維持** する判断とした.

  acceptance「N=18 i=0,1,2 で per-pass ≥ 1.5×」は **N=16 (serial) と N=20
  (rayon 主要 size) で達成** (1.88-2.43× / 2.71-3.48×), N=18 は構造的
  境界として limitation 扱い. C3 (#64) の trotter_step fusion 経路は
  default n_qubit ≥ 17 で fused 経路に乗るため, fused chunk_size が大きい
  方向 (C3 では k=4 pass 連続適用で data reuse あり) で C2.5 の SIMD
  inner kernel が効く.

  bench 結果は PR #80 コメントに添付済み.
- **D (issue #79, 試行・未採用)**: `apply_h_kinema` の DRAM bandwidth
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
- **C4 (issue #65, 実装済み)**: BLAS feature ON/OFF の数値一致 artifact
  test + 大規模 QuTiP 比較 (n=12-16). 構成:
  - **`tests/test_blas_consistency.py`**: 固定 seed の sample 入力 (n ∈ {4,6,8},
    m2 / trotter / trotter_suzuki4 / cfm4 の 4 method) で psi_final /
    probabilities / observables 時系列を ``.npz`` に dump.
    ``KINEMA_EXPECT_BLAS`` env var で build mode を pin できる.
    artifact 出力先は ``KINEMA_ARTIFACT_DIR`` 上書き可
    (default ``tests/artifacts/``, gitignore 済み). adaptive Richardson は
    accept/reject 境界で dt 履歴が BLAS on/off 間で分岐しうるため除外.
  - **`tools/diff_blas_artifacts.py`**: BLAS on / off ビルドで生成した 2 つの
    ``.npz`` を読んで全 array が rel < 1e-13 で一致することを assert する
    standalone script (numpy のみ依存). build profile (``_meta_has_blas``) が
    一致してたら「同 build を比較してる」と検出して即 fail.
  - **`tests/test_reference_qutip.py`**: n=12-14 で 4 method (m2 / trotter /
    cfm4 / cfm4_adaptive_richardson_krylov) を QuTiP ``sesolve`` (atol=1e-12 = 収束
    参照) と fidelity 比較. n=15-16 は cfm4_adaptive_richardson_krylov のみ
    (dense 2^n × 2^n がメモリに乗らないため sparse 経路 = ``sigmax`` tensor 和
    で構築). n>=14 は ``@pytest.mark.slow``. bit 規約変換 (kinema LSB-first
    vs QuTiP MSB-first) は X を tensor list 位置 ``n-1-i`` に挿入する形で吸収.
  - **`benchmarks/bench_qutip_large.py`**: dt sweep で QuTiP sesolve vs
    kinema 固定 dt method の **fidelity と wall time を 1 pass 同時測定**.
    reference は最小 dt の QuTiP cell (dt → 0 収束の代用). QuTiP 側の dt 制御
    は ``options.max_step = dt`` を使う. CSV + markdown を
    ``benchmarks/results/<timestamp>/`` 配下に出力.
  - CI matrix は本 phase の scope 外 (``.github/workflows/`` 未整備). 必要時は
    follow-up issue で導入する (run 手順は本 artifact + diff スクリプトを
    そのまま CI から呼べる構成にしてある).
- **ドキュメント整備、Quick start サンプル — C5 (#66, 実装済み)**: 本 issue で
  Phase 6 finalize として `docs/quickstart.md` を追加 (IsingProblem + Schedule
  + QuantumAnnealer.run + Observable + AnnealingSimulator step-wise +
  instantaneous_eigenstates の 4 snippet), `README.md` から quickstart への
  リンクを追加, `docs/design/*.md` の Phase 6 関連「予定」文言を実装済に整理。
  併せて Phase 6 完了の v0.6.0 / Phase 7 完了の v0.7.0 / Phase 8 完了の
  v0.8.0 を **1 PR にまとめて版数化** し, `CHANGELOG.md` に 0.6.0 / 0.7.0 /
  0.8.0 の 3 リリースを Keep a Changelog 準拠で起こす (Phase 7, 8 はマージ
  済みだが v0.5.0 のまま停止していたものを finalize で遡及的に版数化)。
  本番 bench sweep の結果は `benchmarks/results/0.8.0/` に SUMMARY.md +
  bench_*.md (4 種) として永続化し, Linux AMD EPYC 7713P 上での Phase 6 +
  Phase 7 + Phase 8 累積効果 (rayon scaling 6×+, QuTiP との Pareto 同等
  以上達成等) を記録する。version dir 命名規則と CSV 除外運用は
  `docs/conventions.md` §2.3 で正規化 (本 PR で確立)。

## Phase 7 (v0.7) — Lanczos β_m exposure + Richardson 誤差源分離

主題: adaptive CFM4 Richardson driver の Pareto 劣位 (issue #65 で観測) を,
Lanczos 誤差 (Krylov 部分空間有限性に起因) と Magnus 誤差 (dt 切り捨てに
起因) の **分離制御** で解消する.

### 動機 (#65 → #93 への接続)

Phase 6 C4 (#65) の `bench_qutip_large.py` で **long-T シナリオ** の CFM4
adaptive Richardson が QuTiP に Pareto 劣位だった原因として, 以下が #65 PR
コメント + `bench_m_eff_adiabatic.py` + `tools/verify_beta_m_estimator.py`
の 3 段階で定量化された:

1. default `krylov_tol = 1e-12` では β_k がこれを下回らず m_eff = m_max
   固定 (Krylov 圧縮 0%).
2. `krylov_tol` を緩めると Lanczos 部分空間は半減するが PI controller の
   step reject 数が爆発 (wall time は逆に 7-8× 悪化).
3. 根本原因: Richardson 推定子の `err = ‖ψ_full - ψ_half²‖` は Magnus 誤差と
   Krylov 誤差を区別できず, PI controller が両方をまとめて dt 縮小で対処
   して破綻する.

`tools/verify_beta_m_estimator.py` (#93 prep, PR #94) で `err_lanczos ≈ β_m ·
|c_m| · ‖ψ‖ · dt / m_eff` (Saad 1992 / Hochbruck-Lubich 1997 + 高次補正)
が **5% 精度** で Lanczos 誤差を予測できることが 108 cell sweep で実証された.

### Definition of Done (#93 Step 1-3)

- [x] **Step 1a**: `src/krylov.rs::lanczos_propagate` の return tuple を
      4 要素 `(psi, m_eff, β_m, |c_m|)` に拡張. Rust / Python ref とも
      rel < 1e-13 で一致.
- [x] **Step 1b**: `src/cfm4.rs::cfm4_step` /
      `cfm4_step_with_richardson_estimate` が triangle inequality で
      `err_lanczos_total` を集約して上位に伝播.
- [x] **Step 1c + Step 3**: `evolve_schedule_adaptive_richardson` 内で
      `err_magnus = max(0, err - err_lanczos_total)` を PI controller の駆動量に
      切替え. `beta_m_history` / `err_lanczos_history` / `err_magnus_history`
      / `n_krylov_insufficient` を return tuple に追加. `QuantumResult` に
      `beta_m_stats` / `n_krylov_insufficient` フィールド追加.
- [x] **Step 2**: `benchmarks/bench_m_eff_adiabatic.py` を β_m / err_lanczos
      / err_magnus 軸で拡張. `bench_qutip_large.py` にも `--krylov-tols`
      sweep を追加 (atol × krylov_tol クロス sweep) — PR #94 で同梱.
- [x] **Bench acceptance — 安全性** (#93 perf 一部): Linux サーバー
      (AMD EPYC 7713P, 2026-05-18 計測) で
      `bench_qutip_large.py --scenarios long-T --n-values 8,10
      --krylov-tols auto,1e-8,1e-6` を取り, **default 設定で step reject
      数が増加しないこと** を確認. `auto` (=1e-10) / 1e-8 / 1e-6 で
      `n_steps_eff` 差 0.01-0.02%, wall time 差 ±2% に収まり, PI
      controller が relaxed krylov_tol 下でも安定動作 = Phase 7 の
      safety net が機能していることを実証.
- [ ] **Bench acceptance — Pareto 改善** (#93 perf もう片方): scope 外と
      再認定. 同 bench で CFM4 adaptive Richardson が QuTiP に対し
      **2.5-8× Pareto 劣位のまま** だった. TFIM Lanczos の中間 β_j 値
      が O(‖H‖) で krylov_tol=1e-6 でも閾値を超えず, m_eff=m_max=24 固定
      → Lanczos 圧縮そのものが発火しないことが bench で判明. Phase 7 は
      **そのとき安全な infrastructure** を提供する役割で完了, **真の
      Pareto win は構造的 overhead 削減を要する** ため follow-up へ.

### Out of scope (Phase 7+ follow-up issue へ移管)

- **#96 krylov_tol aggressive 検証**: bench で `krylov_tol=1e-2/1e-1`
  級まで上げて初めて Lanczos 早期打切が発火するか, その際 err_lanczos
  が Krylov 不足を診断するかを検証. Lanczos 圧縮側の axis.
- **#97 Richardson 構造的 overhead 削減**: Pareto 劣位の本質は
  Richardson 3 cfm4_step × 2 Lanczos = 6 Lanczos call / accepted step.
  embedded estimator / time-reuse / adaptive Richardson 頻度低減 等の
  構造改革候補を整理.
- **#93 Step 4 (m_max 動的拡張)**: `err_lanczos > tol_step` 検出時に m_max を
  動的拡張する expokit-style escalation. 本 Phase では diagnostic counter
  `n_krylov_insufficient` を expose するに留め, 自動 escalation は別 issue.
  Phase 7+ の #96 / #97 の知見と合わせて統合計画化予定.
- **β_k stagnation 検出 (#93 末尾候補 1)**: 絶対閾値ではなく `β_k / β_{k-1}`
  比率打切. physical floor (= adiabatic ε admixture) を当てにせず robust に
  発火させる. #96 と統合可能.
- **m_max schedule-aware 縮小 (#93 末尾候補 2)**: annealing 後段ほど ψ が
  固有状態に近づく → step 進行に応じて `m_max: 24 → 16 → 12` と段階的に
  下げる.
- **schedule-aware deflation (#93 末尾候補 3)**: ψ が follow する瞬時 ground
  state を Lanczos 前に deflate して残り部分空間で Krylov を構築.
- **任意精度演算検証 (#93 末尾候補 4)**: mpmath で同じ計算を行い double
  precision との差分を直接定量化.
- **`tol_lanczos` 別パラメータ化**: 本 Phase では `tol_step` と同値運用.
  実運用で必要性が判明したら別 issue.

---

## Phase 8 (v0.8) — Lanczos a posteriori 早期打切 (#98)

主題: Phase 7 で expose した `β_m · |c_m|` a posteriori 推定子を **Lanczos 内部の
早期打切判定そのもの** に使う. これにより Phase 7 で "infrastructure 完了 / Pareto
未解消" だった #65 / #94 の本丸 (= Lanczos 圧縮を実際に発火させる) に
踏み込む.

### 動機 (#94 bench acceptance からの接続)

Phase 7 bench (Linux AMD EPYC 7713P, 2026-05-18) で TFIM Lanczos の中間 β_j 値
が O(‖H‖) で `krylov_tol=1e-6` 級まで緩めても閾値を超えず m_eff = m_max 固定
だった原因が **判定式そのもの** にあると判明. β 単体閾値は誤差量を測れて
おらず, Hochbruck-Lubich 1997 の真の上界
`β_k · |c_last| · |dt| / (k+1) < tol` を見るべき.

### Definition of Done (#98, PR #99 merged)

- [x] **Step 1**: `src/krylov.rs::lanczos_propagate` の早期打切ロジックを
      a posteriori 判定式に書き換え. per-iter で T_{k+1} の三重対角固有分解
      + c_last 計算を行うヘルパ `tridiag_c_last_abs` を新設. β 単体閾値は
      `NUMERICAL_BREAKDOWN_TOL = 1e-14` の hard sanity check に役割を絞る
      (division by zero 回避のみ).
- [x] **Step 2**: 内部 c 配列を `psi_norm` 抜きで保持し, 終端で
      `ψ_new = ‖ψ‖ · V · c` と coeff に畳み込む形にリファクタ (判定式
      `β · |c| · |dt| / m < krylov_tol` が `‖ψ‖ = 1` の正規化空間で
      意味的に整合するため).
- [x] **Step 3**: `python/kinema/krylov.py::_python_lanczos_propagate` も
      完全に同一ロジックで書き直し. `_tridiag_c_last_abs` ヘルパも対応.
      `tests/test_krylov.py::test_rust_lanczos_matches_python_reference` で
      Rust ↔ Python ref が `rel < 1e-13` で一致するのを契約.
- [x] **Step 4**: `krylov_tol` の **意味再定義** を docstring / 設計書に明記
      (β 単体閾値 → "Krylov 近似の許容誤差"). `python/kinema/annealer.py`
      の `QuantumAnnealer.krylov_tol` docstring も更新. API シグネチャ自体は
      不変だがセマンティクス変更なので **minor bump (`0.7 → 0.8`)**.
- [x] **新規 acceptance テスト**: `test_krylov.py` 配下に
      `test_python_lanczos_aposteriori_*` テスト群を追加し
      termination_fires / accuracy_preserved / monotone_compression を担保.
- [x] **Bench acceptance — m_eff 圧縮**: `bench_m_eff_adiabatic.py` で
      `compression_ratio_via_beta` が意味のある値を取る scenario を確認
      (PR #99 コメント参照, 78% 圧縮達成).
- [x] **既存 test pass**: default 設定で既存 Python / Rust テスト全 pass
      = 数値精度 regression なし.
- Pareto win の追加検証 (`bench_qutip_large.py --scenarios long-T`) は
  Phase 6 finalize (#66) の本番 bench sweep で再評価する.

### Out of scope (Phase 8+ follow-up へ)

- **#97 Richardson 構造的 overhead 削減**: Phase 8 で per-Lanczos call の
  m_eff が縮んでも Richardson の 6 call / step 構造は不変. embedded estimator
  / time-reuse / adaptive frequency を別 axis で検討.
- **#93 Step 4 (m_max 動的拡張)**: Phase 8 の a posteriori 判定で
  `err_lanczos > tol_step` が起きにくくなる ため diagnostic は静観で済む
  状況になりやすい. expokit-style escalation の必要性を Phase 8 bench 結果で
  再評価.
- **#96 krylov_tol aggressive 検証**: Phase 8 で `krylov_tol` 意味再定義に
  伴い旧 issue の課題感は薄れる. 必要なら新規 axis として整理.

### Phase 8 follow-up: iter-0 primitive matvec memoization (#100)

#97 close 議論で「Richardson 6 Lanczos call / step は精度損失なし削減不可」と
確定したが, full_step stage 1 と half_1 stage 1 は **同じ入口 ψ から始まる**
ため iter 0 で使う primitive matvec (`H_drv · ψ` / `H_p_diag · ψ`) が共通.
これを `cfm4_step_with_richardson_estimate` 入口で 1 度だけ計算して両 Lanczos
call で再利用する直交最適化を Phase 8 follow-up として導入 (#100).

- 削減量見積もり: ~3% 純削減 (cache 計算 1 合成 matvec 相当の overhead を
  引いた純減). bench acceptance は「速くなれば accept」.
- 実装: `apply_h_drv` / `apply_h_p_diag` primitive を `src/matvec.rs` に追加,
  `cfm4_step` に crate-internal `iter0_cache: Option<...>` 引数を追加,
  closure 内 first-call 分岐で iter 0 だけ cache 線形結合. Lanczos API 不変.
  既存 `apply_h_kinema` の cache-blocked 形は維持 (hot path 触らない).
- 数値同等性: cache あり/なしで `rel < 2e-15` (machine epsilon 数倍).
- 詳細: `docs/design/05-3-propagator.md` "iter-0 primitive matvec memoization",
  `docs/design/05-1-matvec.md` §5.1.1.x.

---

## Phase B (#122) — Chebyshev propagator を CFM4 adaptive Richardson 経路に統合

主題: Phase A (#120, PR #121) で時間独立 H 単体 per-call 29 ms / 4.45×
Lanczos 高速を実測した Chebyshev 3 項漸化 propagator を **時間依存 H + CFM4
Magnus + step-doubling Richardson + PI controller** 経路に統合し,
**公開 Python API レベル** に新 method `cfm4_adaptive_richardson_chebyshev`
として露出する.

### 動機 (Phase A 判定 gate からの接続)

Phase A bench (Linux AMD EPYC 7713P, 2026-05-22 頃) で `chebyshev_propagate`
単体が **per-call 29 ms / K_used = 20** を達成. これは Lanczos baseline
(`perf_cfm4_richardson 18 100 single_lanczos`, ~129 ms / m_eff=20) に対し
**4.45×** で, 判定 gate `≤ 50 ms` を 22.5% で大幅クリア. matvec call 数は
同じ (20 ≒ 20) なので speedup は **計算量削減ではなく cache 戦略 (V matrix
spill 回避) + Gram-Schmidt 消滅** から来ている.

### 公開 API 変更

- **新 method `cfm4_adaptive_richardson_chebyshev`** を追加. 既存
  `cfm4_adaptive_richardson` (Lanczos) は **`cfm4_adaptive_richardson_krylov`
  に rename** して `_krylov` / `_chebyshev` で suffix 対称化 (後方互換 alias
  なしの hard rename; pre-1.0 なので許容).
- Rust 関数名 (`cfm4_step` / `cfm4_step_with_richardson_estimate`) は rename
  せず Lanczos が default. Chebyshev variant は `cfm4_step_chebyshev` /
  `cfm4_step_chebyshev_with_richardson_estimate` の suffix 命名.
- `m_max` を Chebyshev method で渡すと `ValueError` (Chebyshev は K_used
  動的決定で Krylov 部分空間次元概念なし).
- `QuantumResult` の K_used 統計は既存 `m_eff_stats` スロットに格納 (用途は
  per-step propagator 評価コスト統計で同一; method literal で Lanczos /
  Chebyshev を区別).

### 実装の要点

- `src/cfm4.rs::cfm4_step_chebyshev`: 既存 `cfm4_step` (Lanczos) の variant.
  CFM4:2 の 2 stage で per-stage Gershgorin による `(E_c, R)` 再計算 +
  `chebyshev_propagate` 呼出. 線形結合係数 `(c_drv, c_diag)` は Lanczos 版と
  完全同型.
- `src/cfm4.rs::cfm4_step_chebyshev_with_richardson_estimate`: 既存
  Lanczos 版と同じ step-doubling Richardson 構造 (full + half×2 = 6
  chebyshev_propagate call / step). `err_chebyshev_total` を triangle
  inequality で 3 軌道集約.
- `python/kinema/krylov.py::evolve_schedule_adaptive_richardson_chebyshev`:
  Lanczos 版 (`evolve_schedule_adaptive_richardson`) と同じ 10-tuple shape /
  PI controller 構造を保ち, 短時間プロパゲータだけが入れ替わる. **Rust 拡張
  必須** (Python ref fallback 非提供; Chebyshev は数値挙動 + 性能評価が
  Rust 前提の POC).

### Definition of Done (#122)

- [x] **Rust step 関数**: `cfm4_step_chebyshev` /
      `cfm4_step_chebyshev_with_richardson_estimate` + PyO3 wrap + `#[cfg(test)]`
      5 テスト (time-indep / Lanczos 一致 / Richardson dt^5 スケール / dt=0
      恒等 / half-chain bit-exact).
- [x] **Python driver**: `evolve_schedule_adaptive_richardson_chebyshev` +
      `QuantumAnnealer` / `AnnealingSimulator` への method 統合.
- [x] **公開 API 命名**: `cfm4_adaptive_richardson` →
      `cfm4_adaptive_richardson_krylov` hard rename (alias なし).
- [x] **テスト**: `tests/test_chebyshev.py` (QuTiP fidelity > 1-1e-6 +
      Lanczos 一致 rel < 1e-7 + annealer/simulator smoke + m_max ValueError).
- [x] **BLAS on/off 数値一致**: `tests/test_blas_consistency.py` に
      Chebyshev direct call (PI controller を介さない fixed schedule 係数で
      `cfm4_step_chebyshev_*_py` を呼ぶ) artifact dump を追加.
- [x] **Bench infra**: `benchmarks/bench_qutip_large.py` に
      `cfm4_adaptive_richardson_chebyshev` solver を追加.
- [x] **Perf binary**: `src/bin/perf_cfm4_richardson_chebyshev.rs` 新規.
      既存 `perf_cfm4_richardson` の同名 mode と直接比較できる構造で
      Linux perf stat の hardware counter で per-step wall / K_used / IPC を
      breakdown.
- [x] **本番 perf bench** (Linux AMD EPYC 7713P): `bench_qutip_large.py
      --scenarios long-T --solvers cfm4_adaptive_richardson_krylov,
      cfm4_adaptive_richardson_chebyshev` で Pareto plot を取得し PR に添付
      (n=8 1.19-1.25×, n=10 1.19-1.28×, n=12 1.09-1.17× wall 高速 / Lanczos 比).
- [x] **判定 gate 適用**: **Pareto win** で確定 (全 atol で Lanczos 上回り).
  follow-up #124 で default 切替を起票・実施.
- [x] **docs/design/INDEX.md / 05-3-propagator.md / CLAUDE.md** 更新.
- [x] **Version bump**: `0.9.0 → 0.10.0` 実施済 (`Cargo.toml` /
      `pyproject.toml` / `Cargo.lock` / `uv.lock`). 後続 follow-up #124 で
      `0.10.0 → 0.11.0` にさらに minor bump.

### Out of scope (Phase B+ follow-up へ)

- **f32 mixed precision**: Chebyshev は直交化不要なので f32 との相性が良い
  (Phase 9C 候補). f64 で Pareto を勝ち取ってから検討.
- **Power iteration で R refine**: Gershgorin loose で K_used 過大化した
  場合の最適化. Pareto draw に該当した時のみ着手.
- **Trotter 経路への Chebyshev**: Trotter 自体が短時間プロパゲータなので
  Chebyshev で置換する意味なし (out of scope のまま).
- **Default method 切替**: `cfm4_adaptive_richardson_krylov` →
  `cfm4_adaptive_richardson_chebyshev` への default 切替は bench 結果 archive
  後の **別 issue** で判断 (semantic 変更で minor bump が必要なため切り分け).
  → **#124 で確定** (下記 follow-up 節参照, `0.10.0 → 0.11.0` で minor bump).
- **iter-0 cache (Lanczos 経路の #100 流用)**: Chebyshev 経路で full_step /
  half_1 の入口が同じ ψ なので原理的に同型の memoization が可能だが,
  per-stage K_used ~ 20 個の matvec のうち 1 個と削減比小. Phase B では
  scope 外, 必要なら follow-up.

### Phase B follow-up: Chebyshev propagator hot path の Gershgorin precompute (commit `c3f20c7`)

Phase B 完了直後の hot path 最適化 (pre-1.0 内部 API 破壊的変更を含む).
`gershgorin_bounds(h_x, h_p_diag, a_t, b_t)` は per-call で `h_p_diag` (length
2^N) の full walk + `h_x` 絶対値和を計算しており, Chebyshev propagator が 1 step
あたり 6 回呼ぶため per-step `O(2^N + N)` を消費 (N=18 で wall 1% 弱). `h_x` /
`h_p_diag` は `IsingProblem` 構築後 immutable なので `Σ|h_x|` / `min/max(h_p_diag)`
を 1 度だけ precompute して使い回せば per-step は `gershgorin_bounds_cached` で
**O(1) (5 fp 演算)** に縮む.

- 影響範囲 (pre-1.0 内部 API): `chebyshev_propagate` / `cfm4_step_chebyshev*`
  / 両 `_py` wrap のシグネチャに `h_x_abs_sum, h_p_min, h_p_max: f64` 3 引数を
  末尾追加. PyO3 signature も更新.
- `IsingProblem` (Python): `_h_x_abs_sum` / `_h_p_diag_min` / `_h_p_diag_max`
  を `field(init=False, repr=False, compare=False)` で dataclass 宣言, frozen=True
  維持のため `__post_init__` で `object.__setattr__` 経由でセット. 対応する
  read-only property を公開.
- Python driver (`evolve_schedule_adaptive_richardson_chebyshev`): driver 入口で
  precompute を 1 度だけ実行し step ループ内 dispatcher に渡す.
- legacy `gershgorin_bounds` (untransformed) は残し, 内部で cached 版を呼ぶ形に
  整理 (hot path では使わない docstring 注記付き). `b_t` 符号で min/max を
  swap する分岐は cached 版にも保持 (現状 schedule は `b_t ∈ [0, 1]` だが
  robust 性のため).
- 検証: `cargo test --release` (BLAS on) 92 passed / `uv run pytest -m "not
  slow"` 341 passed. `tests/test_blas_consistency.py` の Chebyshev direct call
  artifact dump 経路も同 rel < 1e-13 を維持.
- 詳細: `docs/design/05-3-propagator.md` "スペクトル境界推定" 節 + `src/chebyshev.rs`
  module docstring (両方とも O(N) 表記を O(1) precompute / O(2^N) 素朴経路に
  訂正済).

### Phase B follow-up: Chebyshev 3 項漸化 inner loop の SIMD + fusion (#126)

Phase B で Chebyshev `chebyshev_propagate` の per-step wall は N=18 で 116 ms
(Lanczos の 5.49× 高速; #124 perf archive). per-step の内訳は **matvec ~70%**
+ **非 matvec inner loop ~22%** で, 非 matvec 部分は当初 scalar 2-loop
(recurrence scaling + accumulate) として書かれていた. これを **1 dim-walk +
`wide::f64x4` SIMD** に fuse する直交最適化 (#126).

- 削減量見積もり: 1 K iteration あたり 3 dim-walk → 2 dim-walk (DRAM traffic 33%
  削減) + SIMD で残り walk の throughput 1.7-2.5×. per-step wall 10-22% 改善が
  期待値 (matvec wall は不変).
- 実装: `src/chebyshev.rs::simd_kernels::chebyshev_recurrence_fused` (SIMD) /
  `chebyshev_recurrence_fused_scalar` (scalar fallback) + dispatch wrapper.
  `chebyshev_propagate` の k_ord ≥ 2 hot loop だけ差し替え. k = 1 step は
  one-shot で scalar のまま (overhead 無視可). `cfm4_step_chebyshev_*` 経由でも
  自動で乗る (同じ `chebyshev_propagate` を呼ぶため別途実装不要).
- 数値同等性: `simd_kernels::chebyshev_recurrence_fused` ↔ `_scalar` の
  100-iter fuzz テスト (`chebyshev_recurrence_fused_simd_matches_scalar`,
  `rel < 1e-13`). `cfm4_step_chebyshev_*_py` 経由の end-to-end は既存
  `test_blas_consistency.py::test_blas_consistency_chebyshev_artifact_dump`
  artifact (rel < 1e-13) でカバー.
- bench acceptance (Linux AMD EPYC 7713P, NT=64): per-step wall 10%+ で
  full merge / 5-10% で marginal accept / < 5% で 中止 archive. 計測は
  `perf_chebyshev 18 100` (Chebyshev 単体) +
  `perf_cfm4_richardson_chebyshev 18 100 full` (Richardson 統合経路) の 2 軸.
- 詳細: `docs/design/05-3-propagator.md` "Chebyshev recurrence の SIMD + fusion".

### Phase B follow-up: Chebyshev non-matvec inner loop の rayon 並列化 (#127)

#126 の SIMD + fusion 完了後の直交最適化。#124 perf archive で **Chebyshev の
parallel efficiency が 64 thread で 44%** (Lanczos 27% より良いが理想 100%
には程遠い) と判明。`apply_h_kinema` は #62 で rayon 並列化済だが,
`chebyshev_recurrence_fused` (k_ord ≥ 2 hot loop) が **scalar single-thread**
で走っており, ここがスケーリング bottleneck の一部。

- 期待値: parallel efficiency 44% → 55-65% (1.25-1.5×), per-step wall (N=18)
  116 ms → 90-105 ms (1.10-1.30×). dim 小では fork/join overhead で不変 or
  微悪化が出る可能性あり, dispatch 閾値で safety net を張る。
- 実装: `src/chebyshev.rs::chebyshev_recurrence_fused_rayon` (rayon path) +
  3 段 dispatch wrapper (rayon → SIMD → scalar). chunk_size は matvec と同じ
  式で動的決定 (`MIN_RAYON_DIM_CHEB = 1 << 17` 初期値). `scratch` / `psi_acc`
  の 2 RW slice を `par_chunks_mut` 2 本独立に取って `zip()`, `enumerate()` で
  base offset から `phi_curr` / `phi_prev` (R) を共有 sub-slice 切り出し。
  `cfm4_step_chebyshev_*` 経由でも自動で乗る。
- 数値同等性: rayon path と single-thread SIMD/scalar fused の random fuzz
  10-iter テスト (`chebyshev_recurrence_fused_rayon_matches_serial`,
  `rel < 1e-13`). dim ≥ MIN_RAYON_DIM_CHEB で rayon 経路に乗ることを
  `chebyshev_propagate_rayon_path_smoke` (N=17 end-to-end unitarity) で間接確認。
- bench acceptance (Linux 本番サーバー, perf binary 計測; **本番計算環境とは
  別マシンで CPU 性能は本番より低い**点に注意): N=18 で per-step wall 10%+
  改善 + N=12 (or N=14) で 5% 未満劣化 → full merge / N=18 改善 5-10% + dim 小
  劣化 5-15% → `MIN_RAYON_DIM_CHEB` を上げる方向で閾値 tuning 継続 / N=18 改善
  5% 未満 → 中止 + archive (matvec が memory-bound すぎて non-matvec 並列化
  gain が出ない結論)。計測は `perf_chebyshev N 100` と
  `perf_cfm4_richardson_chebyshev N 100 full` を N ∈ {14, 16, 18, 20} ×
  RAYON_NUM_THREADS ∈ {1, 8, 16, 32, 64} で sweep。
- 詳細: `docs/design/05-3-propagator.md` "Chebyshev non-matvec inner loop の
  rayon 並列化".

### Phase B follow-up: Default method を Chebyshev variant に切替 + atol 仕様明文化 (#124)

Phase B 完了後の **判断系 follow-up** (semantic 変更で minor bump 必要)。
Phase B では新 method `cfm4_adaptive_richardson_chebyshev` を opt-in として
追加するに留め, default 切替は「bench 結果 archive 後の別 issue で判断」と
していた (Out of scope 節)。issue #124 で perf binary 直接比較 (N=18, Linux
AMD EPYC 7713P) により **Lanczos 比 5.49× wall 高速** が確認され, 切替方針が
確定した。

#### 判断結果

1. **Default method 切替 (Scope 1, 候補 B)**: 以下の default を
   `cfm4_adaptive_richardson_chebyshev` に切替。
   - `QuantumAnnealer.run(method=...)`: 旧 `"m2"` → 新 default
   - `QuantumAnnealer.create_simulator(method=...)`: 旧 `"cfm4"` → 新 default
     (ついでに `Literal` から欠落していた `_chebyshev` を追加, Phase B
     #122 取りこぼし fixup)
   - `AnnealingSimulator(method=...)`: 旧 `"cfm4"` → 新 default
   - `docs/quickstart.md` の主例: `method` 指定を削除 (default を使う形)
   - `bench_qutip_large.py --solvers` default は **両 method を含む**
     `_VALID_SOLVERS` 全列挙のまま (Pareto 比較用なので default で両者走らせ
     る方が有用)。`_krylov` は literal として永続的に残す。
2. **"Accidental 高精度" 仕様 (Scope 2, 候補 (a) + (d))**: Chebyshev では
   `atol` を **upper bound** として運用し, K_used 動的拡張で実際の精度が
   それより良くなることを feature として受け入れる。`QuantumAnnealer.run` /
   `AnnealingSimulator` / `docs/design/05-3-propagator.md` /
   `docs/quickstart.md` の docstring/MD に明文化。`chebyshev_tol` の
   auto-coupling 係数 (`_KRYLOV_TOL_ATOL_RATIO = 1e-3`) は変更しない。

#### Definition of Done (#124)

- [x] **`annealer.py` / `simulator.py` の default 切替**.
  `QuantumAnnealer.run` / `create_simulator` / `AnnealingSimulator` の
  `method` Literal default を `cfm4_adaptive_richardson_chebyshev` に変更.
  `create_simulator` の `Literal` に欠落していた `_chebyshev` を追加.
- [x] **Scope 2(d) docstring 明文化**.
  `QuantumAnnealer.run` の `atol` docstring に "Chebyshev variant の
  atol 振舞い" 注を追加。`AnnealingSimulator.__init__` も同様。
  `result.py::QuantumResult.method` の docstring に新 method 名を追加。
- [x] **`bench_qutip_large.py` の `--adaptive-tols` / `--krylov-tols`
  ヘルプ文を両 adaptive 経路に対応**.
- [x] **`docs/quickstart.md` の例を default 経路に切替** (method 指定を削除し
  Chebyshev variant の atol upper bound 注も追記).
- [x] **`docs/design/05-3-propagator.md` "Chebyshev variant" 節に
  "`chebyshev_tol` と `atol` の関係 — accidental 高精度" 小節を追加**.
- [x] **`docs/design/12-release-plan.md` Phase B Out of scope の
  "Default method 切替" 行に "#124 で確定" マーカー追加 + 本 follow-up 節**.
- [x] **`CLAUDE.md` Phase B follow-up 節に #124 を追加** (運用ガイド更新).
- [x] **Version bump**: `0.10.0 → 0.11.0` (`Cargo.toml` / `pyproject.toml`,
  default の semantic 変更を伴うため minor bump).
- [x] **API stub 再生成** (`tools/gen_api_stubs.py`).
- [x] **既存テスト緑** (全 test は既に `method=` 明示なので default 切替で
  壊れるテストなし; 念のため `uv run pytest` で確認).

#### Bench acceptance

本 follow-up は **判断系のみで perf 変更を伴わない** (default 切替と
docstring 更新のみ)。Phase B 本体 (#122) と #126 / #127 の merge 済 bench
結果が perf 上の根拠として既に存在するため, 再計測不要。pre-merge bench
cycle (CLAUDE.md `bench を伴う issue の運用` 節) は適用外。

旧 default (`method="m2"` / `"cfm4"`) を使っていたユーザー向けの
migration note: 新 default 経路は **adaptive PI controller** を走らせるので
`n_steps` の代わりに `atol` で精度を制御する。旧挙動を維持したい場合は
`method="m2"` / `"cfm4"` を明示する。

### Phase B follow-up: パッケージリブランド `kryanneal → kinema` (commit `1106f77`)

v1.0 前の最後の機会としてプロジェクト名を `kryanneal` (Kry(lov) +
(an)neal(ing)) から `kinema` ((Kine)tic quantum evolution by (Ma)gnus
expansion) に rename。pure rename / 数値挙動 / 公開シグネチャ (API レイアウト)
変更なし。0.11.0 minor bump に同梱。

- Python パッケージ: `python/kryanneal/` → `python/kinema/` (git mv で履歴保持)
- Build config: `pyproject.toml` / `Cargo.toml` の `name`, `[tool.maturin]
  module-name = "kinema._rust"`, `.cargo/config.toml` コメント
- Rust 内部関数: `apply_h_kryanneal*` → `apply_h_kinema*` 全 path
- env var: `KRYANNEAL_EXPECT_BLAS` → `KINEMA_EXPECT_BLAS`
- 提案中の例外クラス名 (`docs/design/04-python-api.md` のみ):
  `KryannealError` / `KrySchemaError` / `KryConvergenceError` →
  `KinemaError` / `KineSchemaError` / `KineConvergenceError`
- 検証: pytest 347 passed / cargo test (default + `--no-default-features`)
  92 + 80 passed / pre-commit 全フック pass.

---

