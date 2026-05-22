# kryanneal: Quick start

横磁場イジングモデル (TFIM) の量子アニーリングを最小の手順で動かすための
入門ガイド。設計の詳細は [`docs/design/INDEX.md`](design/INDEX.md), 公開
API の網羅的リファレンスは [`python/kryanneal/*.pyi`](../python/kryanneal/)
スタブを参照する。

## インストール

リモート (GitHub) からソースビルドして既存プロジェクトに追加する:

```bash
uv add 'git+https://github.com/yusekiya/kryanneal'
```

`.cargo/config.toml` 経由で `-C target-cpu=native` が自動適用されるため
build マシン CPU の AVX2 / AVX-512 / NEON が SIMD 経路 (`wide::f64x4`) で
有効化された状態でインストールされる (詳細は
[`docs/design/11-build-infrastructure.md`](design/11-build-infrastructure.md)
§11.1)。

ローカル開発時は repo を clone して:

```bash
uv sync
uv run maturin develop --uv             # debug build (--uv は uv venv 用必須フラグ)
uv run maturin develop --uv --release   # 性能計測時
```

ビルド構成 (どの cargo feature / target feature が有効か) は
`kryanneal.show_config()` で確認できる。

## 1. 最小例: 時間発展して基底状態への重なりを見る

`IsingProblem` (TFIM の定義) と `Schedule.linear` (線形アニーリング
schedule) を組み合わせて `QuantumAnnealer.run()` を回し,
``[t0, t1]`` 区間の時間発展を 1 回で取得する最小サンプル。
``method`` を指定しなければ default の
``"cfm4_adaptive_richardson_chebyshev"`` (issue #124, Phase B Pareto win に
基づく adaptive 経路) が使われる。
``H_problem`` は ``Z`` のみで書ける任意の k-local 多項式を Z 基底で
対角化した ``(2^N,)`` ベクトルで渡す (本パッケージは k-local 表現を
扱わない; 詳細は [`docs/design/02-physics.md`](design/02-physics.md))。

ここでは Sherrington–Kirkpatrick (SK) 型の ``H = -Σ_{i<j} J_ij Z_i Z_j``
を ``σ_i = 1 - 2·b_i`` (bit 0 = LSB) 規約で対角ベクトル化する:

```python
import numpy as np
from kryanneal import IsingProblem, Schedule, QuantumAnnealer
from kryanneal.initial_states import uniform_superposition

n = 6
rng = np.random.default_rng(0)
J = rng.normal(size=(n, n)) / np.sqrt(n)
J = (J + J.T) / 2
np.fill_diagonal(J, 0.0)

# H_problem = -Σ_{i<j} J_ij Z_i Z_j を Z 基底で対角化する.
# bit 0 = LSB, σ_i(x) = 1 - 2·b_i (CLAUDE.md 物理的取り決め節).
x = np.arange(1 << n, dtype=np.int64)
bits = ((x[:, None] >> np.arange(n)) & 1).astype(np.int64)
sigma = 1 - 2 * bits                                    # shape (2^n, n)
H_p_diag = -np.einsum("ij,xi,xj->x", J, sigma, sigma) / 2

prob = IsingProblem(n=n, H_p_diag=H_p_diag, h_x=np.ones(n))
sched = Schedule.linear(T=20.0)
psi0 = uniform_superposition(n)

ann = QuantumAnnealer(prob, sched)
result = ann.run(
    psi0,
    t0=0.0,
    t1=sched.T,
    atol=1e-8,
)  # method 未指定 → default "cfm4_adaptive_richardson_chebyshev"

# H_p の基底状態 (古典イジング解) との重なりを確認する.
gs_index = int(np.argmin(prob.H_p_diag))
gs_probability = float(np.abs(result.psi_final[gs_index]) ** 2)
print(f"|<gs|ψ(T)>|² = {gs_probability:.4f}")
print(f"n_steps     = {result.n_steps_actual}")
```

固定 dt 経路 (`method="m2"` / `"cfm4"` / `"trotter"` / `"trotter_suzuki4"`)
を使う場合は `atol` の代わりに `n_steps=N` を渡す。adaptive 経路の
`atol` 既定値・`dt_init` の auto resolution などの詳細は
[`docs/design/05-3-propagator.md`](design/05-3-propagator.md) §5.3 を参照。

**Note** (Chebyshev variant の atol 振舞い, issue #124): default の
`cfm4_adaptive_richardson_chebyshev` では `atol` は **upper bound** として
機能し, K_used 動的拡張により実際の精度がそれより良くなる場合がある
(例: `atol=1e-3` 設定でも n=10 で `infidelity < 1e-16`)。速度を取りたいときは
`atol` を大きくして PI step 数を減らす運用が正しい。Lanczos 経路を明示したい
場合は `method="cfm4_adaptive_richardson_krylov"` を渡す。

## 2. Observable を `save_tlist` で時系列計測

`Observable.magnetization(n)` で全磁化 ``M_z = Σ_i σ_i^z`` の観測量を組み立て,
`save_tlist` で指定時刻の期待値を時系列で記録する。`save_tlist` は固定 dt
経路では step boundary に merge され, adaptive 経路では PI 制御の dt を
target でクランプして **厳密にその時刻を踏む** (補間しない)。

```python
import numpy as np
from kryanneal import IsingProblem, Schedule, QuantumAnnealer, Observable
from kryanneal.initial_states import uniform_superposition

n = 6
prob = IsingProblem(
    n=n,
    H_p_diag=np.arange(1 << n, dtype=np.float64),
    h_x=np.ones(n),
)
sched = Schedule.linear(T=10.0)
psi0 = uniform_superposition(n)

ann = QuantumAnnealer(prob, sched)
m_z = Observable.magnetization(n)

save_tlist = np.linspace(0.0, sched.T, 11)
result = ann.run(
    psi0,
    t0=0.0,
    t1=sched.T,
    atol=1e-8,
    observables={"M_z": m_z},
    save_tlist=save_tlist,
)  # method 未指定 → default "cfm4_adaptive_richardson_chebyshev"

# 各 save_tlist 時刻での <M_z> 期待値を表示する.
for t, value in zip(result.times, result.observables_history["M_z"], strict=True):
    print(f"t = {t:5.2f}: <M_z> = {value: .6f}")
```

`save_tlist=None` (既定) は **最節約モード** で観測量・状態を一切保持
しない。`observables` を渡したい場合は `save_tlist` も必ず指定する
(無指定で `observables` を渡すと `ValueError`)。

## 3. `AnnealingSimulator` で step-wise に進める

`AnnealingSimulator` は `QuantumAnnealer.run` と同じプロパゲータ群を
**逐次操作** で扱える step-wise API。中間時刻まで進めて Observable で
測ってから, さらに先まで発展させる用途に使う。固定 dt 経路は `run` と
bit-identical な数値 (`rel < 1e-13`) を保証する。

```python
import numpy as np
from kryanneal import IsingProblem, Schedule, QuantumAnnealer, Observable
from kryanneal.initial_states import uniform_superposition

n = 6
prob = IsingProblem(
    n=n,
    H_p_diag=np.arange(1 << n, dtype=np.float64),
    h_x=np.ones(n),
)
sched = Schedule.linear(T=10.0)
psi0 = uniform_superposition(n)
m_z = Observable.magnetization(n)

ann = QuantumAnnealer(prob, sched)
sim = ann.create_simulator(psi0, t0=0.0, atol=1e-8)
# method 未指定 → default "cfm4_adaptive_richardson_chebyshev"

sim.advance_to(sched.T / 2)
mid = sim.measure(m_z)
print(f"<M_z> at t=T/2: {mid: .6f}")

sim.advance_to(sched.T)
final = sim.measure(m_z)
print(f"<M_z> at t=T  : {final: .6f}")
```

`step(dt)` で 1 step だけ進めることもできる。adaptive 経路でも
`advance_to(t)` は dt 系列を内部で adaptive に決め, target 時刻を厳密に踏む。

## 4. `instantaneous_eigenstates` で瞬時 gap を確認

アニーリング中の瞬時 ``H(t)`` の下位 ``k`` 固有状態 (Ritz vector) を
Lanczos 経路 / dense 経路の 2 方式で取得できる。最小 gap や ``ψ(t)`` の
瞬時固有状態への投影確認に使う。

```python
import numpy as np
from kryanneal import IsingProblem, Schedule
from kryanneal.eigenstates import instantaneous_eigenstates

n = 6
prob = IsingProblem(
    n=n,
    H_p_diag=np.arange(1 << n, dtype=np.float64),
    h_x=np.ones(n),
)
sched = Schedule.linear(T=10.0)

# t = T/2 における下位 2 固有値 / 固有ベクトル.
eigvals, eigvecs = instantaneous_eigenstates(
    prob, sched, t=sched.T / 2, k=2, method="lanczos", seed=0
)
gap = eigvals[1] - eigvals[0]
print(f"eigvals = {eigvals}")
print(f"gap     = {gap:.6f}")
```

`n <= 12` であれば `method="exact"` で dense ``H(t)`` の
`numpy.linalg.eigh` 参照経路も使える (Lanczos 結果の検証用)。

## 5. 瞬時基底状態との fidelity で断熱性を確認

量子アニーリングの解析では「瞬時 ``H(t)`` の基底状態 ``|gs(t)⟩`` と
動力学で得られた状態 ``|ψ(t)⟩``」の重なり
``F(t) = |⟨gs(t)|ψ(t)⟩|²`` が断熱性の主要な指標となる。初期状態を
``H(0)`` の基底状態に取れば断熱定理から ``T → ∞`` 極限で ``F(t) ≡ 1`` が
要請される。標準的な線形スケジュール (``A(0)=1``, ``B(0)=0``) では
``H(0) = -Σ h_x_i X_i`` の基底状態が一様重ね合わせ
``|+⟩^N = uniform_superposition(n)`` なので, この初期状態から始めれば
``F(0) = 1`` がスタート値になる。短い ``T`` では非断熱遷移で ``F(t) < 1``,
``T`` を大きくすると ``F(t) → 1`` に漸近することを以下で確認する。

実装の要点:

- ``save_tlist`` で観測時刻列を指定し ``store_states=True`` を渡すと
  各時刻の ``ψ(t)`` が ``QuantumResult.states`` に保存される。
- 同じ時刻列に対して ``instantaneous_eigenstates(prob, sched, t=t, k=1, ...)``
  で瞬時基底状態を取得し ``|⟨gs(t)|ψ(t)⟩|²`` を計算する。基底状態には
  大域位相 ``e^{iφ}`` の任意性があるが ``|⟨gs|ψ⟩|²`` は位相不変なので
  問題にならない。

```python
import numpy as np
from kryanneal import IsingProblem, Schedule, QuantumAnnealer
from kryanneal.initial_states import uniform_superposition
from kryanneal.eigenstates import instantaneous_eigenstates

n = 6
rng = np.random.default_rng(0)
J = rng.normal(size=(n, n)) / np.sqrt(n)
J = (J + J.T) / 2
np.fill_diagonal(J, 0.0)
x = np.arange(1 << n, dtype=np.int64)
bits = ((x[:, None] >> np.arange(n)) & 1).astype(np.int64)
sigma = 1 - 2 * bits
H_p_diag = -np.einsum("ij,xi,xj->x", J, sigma, sigma) / 2

prob = IsingProblem(n=n, H_p_diag=H_p_diag, h_x=np.ones(n))
psi0 = uniform_superposition(n)             # H(0) の基底状態

# 短 T (非断熱) と長 T (断熱極限に近い) を比較する.
for T in (1.0, 100.0):
    sched = Schedule.linear(T=T)
    save_tlist = np.linspace(0.0, sched.T, 11)

    ann = QuantumAnnealer(prob, sched)
    result = ann.run(
        psi0,
        t0=0.0,
        t1=sched.T,
        atol=1e-8,
        save_tlist=save_tlist,
        store_states=True,
    )  # method 未指定 → default "cfm4_adaptive_richardson_chebyshev"

    print(f"T = {T}")
    for t, psi_t in zip(result.times, result.states, strict=True):
        # k=1 で瞬時基底状態を取得 (n <= 12 なので exact が手軽).
        _, gs = instantaneous_eigenstates(
            prob, sched, t=float(t), k=1, method="exact"
        )
        fidelity = float(np.abs(np.vdot(gs[:, 0], psi_t)) ** 2)
        print(f"  t/T = {t / sched.T:4.2f}: F(t) = {fidelity:.6f}")
```

実行すると ``T = 1`` では終端 ``F(T)`` が ``1`` から有意に外れるのに対し,
``T = 100`` では全時刻で ``F(t) ≈ 1`` を保つことが観察できる。これが
「断熱極限で fidelity = 1」の数値的な確認になる。

``n > 12`` で dense ``eigh`` が現実的でない領域では
``method="lanczos"`` に切り替える (``k`` を増やせば
``|⟨gs_j(t)|ψ(t)⟩|²`` で low-lying state への漏れも追跡できる; 詳細は
[`docs/design/04-python-api.md`](design/04-python-api.md))。

## 6. 並列ジョブ実行時のスレッド数制御

複数の独立計算を並列に走らせる (例: ハイパーパラメータ sweep,
パラメータごとの独立 trajectory, Slurm の job array) シナリオでは,
1 プロセスあたりのスレッド数を絞らないと **rayon と BLAS の 2 系統 ×
ジョブ数** で総 OS thread が `cpu_count^2` 相当に膨れ context-switch で
性能が崩れる。

`kryanneal` の thread pool は **プロセス起動時の環境変数で確定し,
以降縮小できない** (rayon の global pool / BLAS の thread pool いずれも
最初の op で初期化される) ため, **`kryanneal` / `numpy` を import する
前に環境変数を set する** のが唯一正規の制御手段。

### 制御に使う環境変数

| 環境変数 | 対象 |
|---|---|
| `RAYON_NUM_THREADS` | rayon global pool (default: 論理コア数, SMT/HT 込み) |
| `OPENBLAS_NUM_THREADS` | Linux OpenBLAS pool |
| `MKL_NUM_THREADS` | MKL 利用時 |
| `VECLIB_MAXIMUM_THREADS` | macOS Apple Accelerate |
| `OMP_NUM_THREADS` | 上記未設定時の OpenMP fallback |

複数 BLAS pool (numpy bundled / scipy bundled / system) が同居しうるため,
迷ったら **全部同じ値で揃える** のが安全。`kryanneal.set_blas_threads(n)` /
`kryanneal.set_blas_threads_auto()` は import 後に動的に active BLAS
thread 数を絞る補助関数だが, **pool size 自体は縮まない**
(sleeping thread の stack / kernel resource は残る) ので per-process
隔離の主役にはならない。

`kryanneal.set_blas_threads_auto()` は issue #116 (2026-05-21) で導入された
推奨 default 設定の便利関数: `os.process_cpu_count() // 8` を 1-16 で
クランプし, 上記 env (`OPENBLAS_NUM_THREADS` 等) が set されていれば
それを strict な上限として `min(auto, env_cap)` を返す。EPYC 7713P 実測で
NT=8 で 1.52× speedup の sweet spot。

### パターン A: Python 内 multiprocessing (ProcessPoolExecutor + spawn)

`initializer` で子プロセス内 (= `kryanneal` import 前) に env を set する。
`fork` 文脈だと親で既に numpy / kryanneal を import 済みの場合に env が
効かないため, `mp_context="spawn"` で起動する。

```python
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

THREADS_PER_WORKER = 8

def _init_worker():
    # 子プロセスで kryanneal / numpy import より前に実行される.
    os.environ["RAYON_NUM_THREADS"] = str(THREADS_PER_WORKER)
    os.environ["OPENBLAS_NUM_THREADS"] = str(THREADS_PER_WORKER)
    os.environ["OMP_NUM_THREADS"] = str(THREADS_PER_WORKER)

def worker(job):
    import kryanneal  # この時点で初めて import (env が効く)
    # ... 計算 ...
    return result

if __name__ == "__main__":
    ctx = mp.get_context("spawn")  # fork でなく spawn
    with ProcessPoolExecutor(
        max_workers=8,
        mp_context=ctx,
        initializer=_init_worker,
    ) as pool:
        results = list(pool.map(worker, jobs))
```

数値計算系では SMT/HT は逆効果のことが多い (FPU 取り合い)。
`max_workers × THREADS_PER_WORKER` が **物理コア数** を超えないように
配分するのが基本ルール。

### パターン B: shell から N ジョブ並走 (Slurm srun / GNU parallel など)

env をジョブ起動時のコマンドライン側で set する。

```bash
# 8 ジョブを 8 thread/ジョブ で並走 (物理 64 コアを想定)
for i in $(seq 1 8); do
    RAYON_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 \
        uv run python my_job.py --idx "$i" &
done
wait
```

Slurm の `cpuset` / cgroup で 1 ジョブあたりの CPU 数が絞られていれば
rayon `available_parallelism()` がそれを反映するので env 未設定でも
妥当に動くことが多いが, BLAS pool は cgroup を honor しない実装もある
ため明示推奨。

### 関連

- 詳細な背景・rayon / BLAS の挙動差: CLAUDE.md "Thread pool 運用 (rayon × BLAS)" 節
- `kryanneal.set_blas_threads(n)` / `kryanneal.set_blas_threads_auto()` の
  用途と限界: docstring 参照 (import 後に動的に active BLAS thread 数を絞る
  補助; pool size は変えない)

## 次に読むもの

- 公開 API リファレンス: [`python/kryanneal/*.pyi`](../python/kryanneal/)
  (full docstring つき, スタブ生成は `tools/gen_api_stubs.py`)
- 設計詳細 (アルゴリズム・bit 規約・Lanczos / CFM4 / Richardson):
  [`docs/design/INDEX.md`](design/INDEX.md)
- バージョニングと開発規約: [`docs/conventions.md`](conventions.md)
- ベンチマーク方針と性能改善主張のフォーマット:
  [`docs/design/10-benchmarks.md`](design/10-benchmarks.md)
