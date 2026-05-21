"""ローカル N=18 atol=1e-3 で adaptive Richardson の m_eff 圧縮効果を確認する probe.

memory `project_richardson_overhead_dead_end` で参照される Phase 8 #98 の
m_eff 圧縮 (24 → 5.33) が `bench_m_eff_adiabatic.py` の adiabatic 長時間
scenario で実証された一方, **N=18, T=10^4, atol=1e-3 という具体的 setting** で
実際に圧縮が発火しているかは未確認. このローカル 1-cell probe で確認する.

bench は他の setting (atol=1e-3,1e-5,1e-7,1e-9 / cfm4 / single thread / ...)
で並行 Linux 上を走っているので, ここでは macOS でローカル 1 cell だけ取る.
res.m_eff_stats (Phase 4 follow-up #52 で expose) を print して終了.

起動 (SSH 切断 + macOS sleep 耐性):
    caffeinate -i nohup uv run python -u probe_m_eff_n18.py \\
        > probe_m_eff_n18.log 2>&1 < /dev/null &
    echo $! > probe_m_eff_n18.pid
    disown

進捗確認: tail -f probe_m_eff_n18.log
停止: kill -TERM -- -"$(ps -o pgid= -p "$(cat probe_m_eff_n18.pid)" | tr -d ' ')"
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

import kryanneal
from kryanneal import IsingProblem, QuantumAnnealer, Schedule
from kryanneal.initial_states import uniform_superposition

# bench script と同じ運用: OpenBLAS spin wait 排除 + rayon×BLAS 競合回避
kryanneal.set_blas_threads(1)
print(f"[config] BLAS threads = 1 (spin wait / rayon×BLAS 競合回避)", flush=True)

# 既存の non-stiff 問題 npz を読み込む (Linux 本番 bench でも使う問題と同じ)
PROBLEM_FILE = Path("benchmarks/data/problem_non-stiff_n18_seed20260518.npz")
print(f"[probe] loading {PROBLEM_FILE}", flush=True)
data = np.load(PROBLEM_FILE)
n = int(data["n"])
seed = int(data["seed"])
scenario = str(data["scenario"])
h_p_diag = np.asarray(data["H_p_diag"], dtype=np.float64)
h_x = np.asarray(data["h_x"], dtype=np.float64)

prob = IsingProblem(n=n, H_p_diag=h_p_diag, h_x=h_x)
sched = Schedule.linear(T=10000.0)
psi0 = uniform_superposition(n)

print(
    f"[probe] scenario={scenario}, n={n}, T=10000, seed={seed}, "
    f"|H_p_diag| range = [{h_p_diag.min():.3g}, {h_p_diag.max():.3g}]",
    flush=True,
)

ann = QuantumAnnealer(prob, sched)
print(
    f"[probe] running cfm4_adaptive_richardson atol=1e-3 ... "
    f"(start: {time.strftime('%Y-%m-%d %H:%M:%S')})",
    flush=True,
)

t_start = time.perf_counter()
res = ann.run(
    psi0,
    0.0,
    10000.0,
    method="cfm4_adaptive_richardson",
    atol=1e-3,
)
wall = time.perf_counter() - t_start

print("", flush=True)
print(f"=== probe done (wall = {wall:.1f}s = {wall / 60:.2f} min) ===", flush=True)
print(f"n_steps_actual: {res.n_steps_actual}", flush=True)
print(f"m_eff_stats:    {res.m_eff_stats}", flush=True)

# m_eff_stats は dict[str, float | int]. 期待される keys:
#   total / mean / median / min / max
# Phase 8 #98 で long-T adiabatic で 5.33 まで圧縮の実証あり (memory
# `project_richardson_overhead_dead_end`). このローカル probe で N=18,
# atol=1e-3 設定での実態を確認する.
if res.m_eff_stats is not None:
    m_eff_mean = res.m_eff_stats.get("mean")
    m_eff_max = res.m_eff_stats.get("max")
    m_eff_min = res.m_eff_stats.get("min")
    print("", flush=True)
    print(
        f"[m_eff summary] mean={m_eff_mean}, min={m_eff_min}, max={m_eff_max}",
        flush=True,
    )
    print(
        f"  → Phase 8 #98 の long-T adiabatic で実証された 5.33 と比較. "
        f"mean が 5-7 程度なら N=18 でも圧縮が効いている. 24 fully なら未発火.",
        flush=True,
    )

# 念のため最終状態の基本情報
psi_final = res.psi_final
norm = float(np.linalg.norm(psi_final))
print(
    f"\n[psi_final] norm = {norm:.6f} (should be ~1.0), "
    f"max |amplitude| = {float(np.max(np.abs(psi_final))):.6f}",
    flush=True,
)
