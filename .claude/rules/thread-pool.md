---
paths:
  - "src/**/*.rs"
  - "python/maqina/**/*.py"
  - "benchmarks/**/*.py"
---

# Thread pool 運用 (rayon × BLAS)

Phase 6 C1 (issue #62) で matvec / Trotter primitives を rayon
`par_chunks_mut` で L2 並列化済み (`feature = "rayon"`, default ON)。
thread pool が 2 系統並走するため運用ルール:

- **rayon thread 数**: 環境変数 `RAYON_NUM_THREADS` で **プロセス起動時に**
  set する。rayon の global pool は最初の rayon op で構築され, その後は
  変更不可。動的に縮小する手段は無く, 環境変数が一次的な制御手段。
  未設定時 default は `std::thread::available_parallelism()` (≒ 論理コア数,
  SMT/HT 込み; Linux では `sched_getaffinity` / cgroup を尊重)。
- **BLAS thread 数**: env で **pool size** を確定し, ランタイムで **active
  thread 数** を補助的に調整する 2 段:
  - **pool size** (= プロセスが確保する OS thread 数): 起動時 env で確定。
    Linux OpenBLAS は `OPENBLAS_NUM_THREADS`, MKL は `MKL_NUM_THREADS`,
    macOS Apple Accelerate は `VECLIB_MAXIMUM_THREADS`, fallback の
    OpenMP 系は `OMP_NUM_THREADS`。複数 pool 同居時は全 env を揃える。
  - **active thread 数** (= 並列 BLAS op で実際に使う thread): Python 側
    `maqina.set_blas_threads(n)` で動的に変えられる
    (`threadpoolctl.threadpool_limits` 経由)。**pool size 自体は縮まらない**
    (sleeping thread の stack / kernel resource は残る) ので, per-process
    thread budget の隔離が要件なら env 設定が必須。
- **競合回避と推奨 default (issue #116, 2026-05-21 改訂)**: rayon 経路でも
  Lanczos 内部の BLAS-1 (`gemv` / `axpy` / `nrm2` 等) は **適度な BLAS 並列化が
  prod 速い** ことが Linux AMD EPYC 7713P での perf sweep (#113 / PR #115) で
  実証された。NT=8 で **1.52× speedup** (NT=1 baseline 比), NT=16-32 でも
  +2% 程度の劣化で許容範囲, NT=64 で -9% に明確な劣化という curve。
  **新方針**:
  - 既定値は `maqina.set_blas_threads_auto()` を import 後に 1 度呼ぶ。
    内部で `os.process_cpu_count() // 8` を 1-16 でクランプし
    (EPYC SMT2 で 16, 32-core で 4, 8-core 以下で 1), さらに
    `OPENBLAS_NUM_THREADS` / `MKL_NUM_THREADS` / `VECLIB_MAXIMUM_THREADS` /
    `OMP_NUM_THREADS` (この優先順) が set されていればそれを strict な
    上限として `min(auto, env_cap)` を返す。冪等。
  - 完全隔離が要件なら `set_blas_threads(1)` を明示。あるいは起動前に
    env で `OPENBLAS_NUM_THREADS=1` 等を set。
  - 旧推奨「rayon 経路では `set_blas_threads(1)`」は perf 計測前の仮説で
    あり, 1.52× の改善余地を逃していた。**撤回済み**。
  ベンチ運用 (`benchmarks/bench_parallel_scaling.py` 子モード冒頭の強制) も
  同じく `set_blas_threads_auto()` 経由に揃える方向。当初懸念だった
  「spin-wait が rayon を圧迫」も実害無し (`OPENBLAS_THREAD_TIMEOUT=1` で spin
  抑制すると逆に +4-9% 遅化する。Lanczos の BLAS call 間隔が短く futex
  park → unpark の wake-up cost が遊休 core 占有より高くつくため)。
  - **本番 perf bench (Pareto / QuTiP 比較 等) の運用** (2026-05-21 確定):
    `bench_qutip_large.py` / `bench_per_step.py` の `--blas-threads N` フラグに
    **NT=8 を明示渡す** のを default にする。理由は上記 sweep の sweet spot で,
    `--blas-threads` 不指定 (= OpenBLAS の物理コア数 default で 64 threads) だと
    NT=8 比 ~1.10× 遅化する (PR #106 の 0.8.0 bench で実測, PR #106
    コメントに対比表)。`set_blas_threads_auto()` は EPYC 7713P で同じく NT=8 を
    自動算出するので, "machine 種別を意識せず使いたい" 場合は auto setter を
    bench script 冒頭で 1 度呼んでも等価。`--blas-threads 1` は machine-
    independent baseline (= 純シリアル比較) 用途で本番 perf bench とは別の
    意味づけ。
- **並列ジョブ実行 (multiprocessing / Slurm job array 等)**: 1 プロセス
  あたりの thread budget を絞るには **`maqina` / `numpy` を import する前**
  に上記 env (`RAYON_NUM_THREADS` / `OPENBLAS_NUM_THREADS` / `MKL_NUM_THREADS`
  / `VECLIB_MAXIMUM_THREADS` / `OMP_NUM_THREADS`) を一括 set する必要がある
  (BLAS / rayon の pool size は最初の op で確定し以降縮小不可)。具体的な
  multiprocessing パターン例は `docs/quickstart.md` 末尾節を参照。Slurm
  などジョブスケジューラの `cpuset` / cgroup で絞られていれば rayon
  `available_parallelism()` がそれを反映するので, env 未設定でも妥当に
  動くことが多いが, BLAS pool は cgroup を honor しない実装もあるため
  明示推奨。
- **`--no-default-features` ビルド**: scalar 単スレッド経路に戻り rayon 依存
  なし。`RAYON_NUM_THREADS` は無視される。
