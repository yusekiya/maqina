# maqina: Quick start

A minimal guide to running quantum annealing of the transverse-field Ising
model (TFIM). For design details see
[`docs/design/INDEX.md`](design/INDEX.md) (Japanese only). For the
comprehensive API reference, see the per-module stubs at
[`python/maqina/*.pyi`](../python/maqina/) (docstrings mostly in Japanese).

## Installation

Build from a remote (GitHub) source and add to an existing project:

```bash
uv add 'git+https://github.com/yusekiya/maqina'
```

`-C target-cpu=native` is applied automatically via `.cargo/config.toml`, so
the build machine's AVX2 / AVX-512 / NEON are enabled in the SIMD path
(`wide::f64x4`) at install time (details:
[`docs/design/11-build-infrastructure.md`](design/11-build-infrastructure.md)
§11.1, Japanese only).

For local development, clone the repo and:

```bash
uv sync
uv run maturin develop --uv             # debug build (--uv is required for uv venv)
uv run maturin develop --uv --release   # for performance measurement
```

The build configuration (which cargo / target features are active) can be
checked with `maqina.show_config()`.

## 1. Minimal example: time evolve and check overlap with the ground state

Combine `IsingProblem` (TFIM definition) and `Schedule.linear` (linear
annealing schedule), then run `QuantumAnnealer.run()` to perform the time
evolution over `[t0, t1]` in a single call. If `method` is not specified, the
default `"cfm4_adaptive_richardson_chebyshev"` (issue #124, Phase B Pareto
win-based adaptive path) is used.

`H_problem` is passed as a `(2^N,)` vector that diagonalizes any k-local
polynomial expressible in `Z` operators in the Z basis (this package does
not handle k-local symbolic representations; for the conventions see
[`docs/design/02-physics.md`](design/02-physics.md), Japanese only).

Here we diagonalize a Sherrington–Kirkpatrick (SK) Hamiltonian
``H = -Σ_{i<j} J_ij Z_i Z_j`` under the convention ``σ_i = 1 - 2·b_i``
(bit 0 = LSB):

```python
import numpy as np
from maqina import IsingProblem, Schedule, QuantumAnnealer
from maqina.initial_states import uniform_superposition

n = 6
rng = np.random.default_rng(0)
J = rng.normal(size=(n, n)) / np.sqrt(n)
J = (J + J.T) / 2
np.fill_diagonal(J, 0.0)

# Diagonalize H_problem = -Σ_{i<j} J_ij Z_i Z_j in the Z basis.
# bit 0 = LSB, σ_i(x) = 1 - 2·b_i (see CLAUDE.md "physical conventions").
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
)  # method unspecified → default "cfm4_adaptive_richardson_chebyshev"

# Check the overlap with the ground state of H_p (classical Ising solution).
gs_index = int(np.argmin(prob.H_p_diag))
gs_probability = float(np.abs(result.psi_final[gs_index]) ** 2)
print(f"|<gs|ψ(T)>|² = {gs_probability:.4f}")
print(f"n_steps     = {result.n_steps_actual}")
```

If you use a fixed-dt path (`method="m2"` / `"cfm4"` / `"trotter"` /
`"trotter_suzuki4"`), pass `n_steps=N` instead of `atol`. For details on
the adaptive path's `atol` defaults and `dt_init` auto resolution, see
[`docs/design/05-3-propagator.md`](design/05-3-propagator.md) §5.3
(Japanese only).

**Note** (Chebyshev variant `atol` semantics, issue #124): with the default
`cfm4_adaptive_richardson_chebyshev`, `atol` acts as an **upper bound**, and
the actual accuracy can be better than that due to dynamic K_used expansion
(e.g. even with `atol=1e-3` at n=10 you may see `infidelity < 1e-16`). To
trade accuracy for speed, the correct lever is to **increase `atol`** so
that the PI controller takes fewer steps. To explicitly request the Lanczos
path, pass `method="cfm4_adaptive_richardson_krylov"`.

## 2. Observable time-series via `save_tlist`

Use `Observable.magnetization(n)` to assemble the total magnetization
``M_z = Σ_i σ_i^z`` observable, and record expectation values at specified
times via `save_tlist`. On fixed-dt paths, `save_tlist` is merged into the
step boundaries; on adaptive paths, the PI-controlled `dt` is clamped at
the target so that **the exact target time is stepped on** (no
interpolation).

```python
import numpy as np
from maqina import IsingProblem, Schedule, QuantumAnnealer, Observable
from maqina.initial_states import uniform_superposition

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
)  # method unspecified → default "cfm4_adaptive_richardson_chebyshev"

# Print <M_z> expectation value at each save_tlist time.
for t, value in zip(result.times, result.observables_history["M_z"], strict=True):
    print(f"t = {t:5.2f}: <M_z> = {value: .6f}")
```

`save_tlist=None` (the default) is **maximum-economy mode** and stores no
observables or states. To use `observables`, you must also specify
`save_tlist` (omitting `save_tlist` while passing `observables` raises
`ValueError`).

## 3. Step-wise progression with `AnnealingSimulator`

`AnnealingSimulator` is a step-wise API that uses the same set of
propagators as `QuantumAnnealer.run` but exposes them as **incremental
operations**. Use it when you want to evolve to an intermediate time,
measure with an Observable, then continue evolving. Fixed-dt paths
guarantee bit-identical numerical results compared to `run` (`rel < 1e-13`).

```python
import numpy as np
from maqina import IsingProblem, Schedule, QuantumAnnealer, Observable
from maqina.initial_states import uniform_superposition

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
# method unspecified → default "cfm4_adaptive_richardson_chebyshev"

sim.advance_to(sched.T / 2)
mid = sim.measure(m_z)
print(f"<M_z> at t=T/2: {mid: .6f}")

sim.advance_to(sched.T)
final = sim.measure(m_z)
print(f"<M_z> at t=T  : {final: .6f}")
```

`step(dt)` advances by exactly one dt step. On adaptive paths,
`advance_to(t)` chooses the internal dt series adaptively and lands
exactly on the target time.

## 4. Check instantaneous gaps with `instantaneous_eigenstates`

The lowest ``k`` eigenstates (Ritz vectors) of the instantaneous ``H(t)``
along the schedule can be obtained via two paths: Lanczos or dense. Use it
to check the minimum gap, or project ``ψ(t)`` onto instantaneous
eigenstates.

```python
import numpy as np
from maqina import IsingProblem, Schedule
from maqina.eigenstates import instantaneous_eigenstates

n = 6
prob = IsingProblem(
    n=n,
    H_p_diag=np.arange(1 << n, dtype=np.float64),
    h_x=np.ones(n),
)
sched = Schedule.linear(T=10.0)

# The lowest 2 eigenvalues / eigenvectors at t = T/2.
eigvals, eigvecs = instantaneous_eigenstates(
    prob, sched, t=sched.T / 2, k=2, method="lanczos", seed=0
)
gap = eigvals[1] - eigvals[0]
print(f"eigvals = {eigvals}")
print(f"gap     = {gap:.6f}")
```

For `n <= 12`, you can also use `method="exact"` for a dense ``H(t)``
``numpy.linalg.eigh`` reference path (useful for validating Lanczos
results).

## 5. Adiabaticity check via fidelity with the instantaneous ground state

In quantum annealing analysis, the overlap between the instantaneous
ground state ``|gs(t)⟩`` of ``H(t)`` and the dynamically obtained state
``|ψ(t)⟩``, ``F(t) = |⟨gs(t)|ψ(t)⟩|²``, is the primary indicator of
adiabaticity. By the adiabatic theorem, if the initial state is the
ground state of ``H(0)``, ``F(t) ≡ 1`` is required in the ``T → ∞`` limit.
For the standard linear schedule (``A(0)=1``, ``B(0)=0``), the ground
state of ``H(0) = -Σ h_x_i X_i`` is the uniform superposition
``|+⟩^N = uniform_superposition(n)``, so starting from this initial state
gives ``F(0) = 1``. Short ``T`` induces non-adiabatic transitions with
``F(t) < 1``; large ``T`` brings ``F(t)`` asymptotically to ``1``, as
verified below.

Implementation points:

- Pass ``save_tlist`` to specify observation times, and set
  ``store_states=True`` so that ``ψ(t)`` at each time is stored in
  ``QuantumResult.states``.
- For the same time series, call
  ``instantaneous_eigenstates(prob, sched, t=t, k=1, ...)`` to obtain
  the instantaneous ground state, then compute
  ``|⟨gs(t)|ψ(t)⟩|²``. The ground state has an arbitrary global phase
  ``e^{iφ}`` but ``|⟨gs|ψ⟩|²`` is phase-invariant, so this is not an
  issue.

```python
import numpy as np
from maqina import IsingProblem, Schedule, QuantumAnnealer
from maqina.initial_states import uniform_superposition
from maqina.eigenstates import instantaneous_eigenstates

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
psi0 = uniform_superposition(n)             # ground state of H(0)

# Compare short T (non-adiabatic) vs long T (near-adiabatic limit).
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
    )  # method unspecified → default "cfm4_adaptive_richardson_chebyshev"

    print(f"T = {T}")
    for t, psi_t in zip(result.times, result.states, strict=True):
        # k=1 for instantaneous ground state (n <= 12, so exact is convenient).
        _, gs = instantaneous_eigenstates(
            prob, sched, t=float(t), k=1, method="exact"
        )
        fidelity = float(np.abs(np.vdot(gs[:, 0], psi_t)) ** 2)
        print(f"  t/T = {t / sched.T:4.2f}: F(t) = {fidelity:.6f}")
```

Running this, at ``T = 1`` the final ``F(T)`` deviates significantly from
``1``, while at ``T = 100`` ``F(t) ≈ 1`` holds at all times. This is the
numerical confirmation of "fidelity = 1 in the adiabatic limit".

For ``n > 12`` where dense ``eigh`` is impractical, switch to
``method="lanczos"`` (increasing ``k`` lets you also track leakage into
low-lying states via ``|⟨gs_j(t)|ψ(t)⟩|²``; see
[`docs/design/04-python-api.md`](design/04-python-api.md), Japanese only).

## 6. Thread-count control for parallel jobs

When running multiple independent calculations in parallel (e.g.
hyperparameter sweeps, independent trajectories per parameter, Slurm job
arrays), without limiting the threads per process the **two thread systems
(rayon + BLAS) × number of jobs** can balloon to roughly `cpu_count^2`
total OS threads, and performance collapses due to context switching.

`maqina`'s thread pools are **fixed at process startup via environment
variables, and cannot be shrunk later** (both the rayon global pool and the
BLAS thread pool are initialized at the first op). The only valid control
mechanism is to **set the environment variables before importing
`maqina` / `numpy`**.

### Environment variables used for control

| Variable | Target |
|---|---|
| `RAYON_NUM_THREADS` | rayon global pool (default: logical core count incl. SMT/HT) |
| `OPENBLAS_NUM_THREADS` | Linux OpenBLAS pool |
| `MKL_NUM_THREADS` | If MKL is used |
| `VECLIB_MAXIMUM_THREADS` | macOS Apple Accelerate |
| `OMP_NUM_THREADS` | OpenMP fallback when the above are unset |

Since multiple BLAS pools (numpy-bundled / scipy-bundled / system) can
coexist, when in doubt, **set them all to the same value**.
`maqina.set_blas_threads(n)` / `maqina.set_blas_threads_auto()` are helper
functions that dynamically shrink the **active** BLAS thread count
post-import, but the **pool size itself does not shrink** (the kernel
resources / stack of sleeping threads remain), so they are not the
primary lever for per-process isolation.

`maqina.set_blas_threads_auto()` was added in issue #116 (2026-05-21) as a
convenience function for the recommended default setting: it computes
`os.process_cpu_count() // 8` clamped to 1-16, and if the env vars above
(`OPENBLAS_NUM_THREADS` etc.) are set, it treats them as a strict upper
bound and returns `min(auto, env_cap)`. Measured on EPYC 7713P, NT=8 hits
a 1.52× speedup sweet spot.

### Pattern A: Python multiprocessing (ProcessPoolExecutor + spawn)

Use `initializer` to set env vars inside the child process (= before
`maqina` is imported). `fork` does not work if the parent has already
imported numpy / maqina (env vars are then ineffective), so launch with
`mp_context="spawn"`.

```python
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

THREADS_PER_WORKER = 8

def _init_worker():
    # Runs in the child process, before maqina / numpy import.
    os.environ["RAYON_NUM_THREADS"] = str(THREADS_PER_WORKER)
    os.environ["OPENBLAS_NUM_THREADS"] = str(THREADS_PER_WORKER)
    os.environ["OMP_NUM_THREADS"] = str(THREADS_PER_WORKER)

def worker(job):
    import maqina  # first import here (env vars take effect)
    # ... computation ...
    return result

if __name__ == "__main__":
    ctx = mp.get_context("spawn")  # spawn, not fork
    with ProcessPoolExecutor(
        max_workers=8,
        mp_context=ctx,
        initializer=_init_worker,
    ) as pool:
        results = list(pool.map(worker, jobs))
```

For numerical computation, SMT/HT is often counterproductive (FPU
contention). The basic rule is to keep `max_workers × THREADS_PER_WORKER`
**within the physical core count**.

### Pattern B: N concurrent jobs from the shell (Slurm srun, GNU parallel, etc.)

Set the env vars at job-launch time on the command line.

```bash
# 8 jobs at 8 threads/job (assuming 64 physical cores)
for i in $(seq 1 8); do
    RAYON_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 \
        uv run python my_job.py --idx "$i" &
done
wait
```

If Slurm's `cpuset` / cgroup limits the per-job CPU count, rayon's
`available_parallelism()` reflects that, so it usually works correctly
without explicit env vars. However, some BLAS pool implementations do not
honor cgroup limits, so explicit env vars are recommended.

### See also

- For background on rayon / BLAS behavioral differences: CLAUDE.md
  "Thread pool 運用 (rayon × BLAS)" section (Japanese only)
- For the use and limitations of `maqina.set_blas_threads(n)` /
  `maqina.set_blas_threads_auto()`: see the docstrings (helper to
  dynamically shrink the active BLAS thread count post-import; does not
  change the pool size)

## Further reading

- Public API reference: [`python/maqina/*.pyi`](../python/maqina/)
  (full docstrings included; stubs generated by `tools/gen_api_stubs.py`)
- Design details (algorithms, bit conventions, Lanczos / CFM4 /
  Richardson): [`docs/design/INDEX.md`](design/INDEX.md) (Japanese only)
- Versioning and development conventions:
  [`docs/conventions.md`](conventions.md) (Japanese only)
- Benchmarking policy and performance-claim format:
  [`docs/design/10-benchmarks.md`](design/10-benchmarks.md) (Japanese only)
