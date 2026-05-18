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
schedule) を組み合わせて `QuantumAnnealer.run(method="cfm4_adaptive_richardson")`
を回し, ``[t0, t1]`` 区間の時間発展を 1 回で取得する最小サンプル。
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
    method="cfm4_adaptive_richardson",
    atol=1e-8,
)

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
    method="cfm4_adaptive_richardson",
    atol=1e-8,
    observables={"M_z": m_z},
    save_tlist=save_tlist,
)

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
sim = ann.create_simulator(psi0, t0=0.0, method="cfm4_adaptive_richardson", atol=1e-8)

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

## 次に読むもの

- 公開 API リファレンス: [`python/kryanneal/*.pyi`](../python/kryanneal/)
  (full docstring つき, スタブ生成は `tools/gen_api_stubs.py`)
- 設計詳細 (アルゴリズム・bit 規約・Lanczos / CFM4 / Richardson):
  [`docs/design/INDEX.md`](design/INDEX.md)
- バージョニングと開発規約: [`docs/conventions.md`](conventions.md)
- ベンチマーク方針と性能改善主張のフォーマット:
  [`docs/design/10-benchmarks.md`](design/10-benchmarks.md)
