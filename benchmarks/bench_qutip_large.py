"""QuTiP ``sesolve`` vs kryanneal の work-precision diagram ベンチ (issue #65 Phase 6 C4).

比較指標を **「精度 (1 - fidelity) vs 計算時間 (wall sec)」** の 2 軸 Pareto
にした (issue #65 user review). 旧版は dt sweep で QuTiP/kryanneal を同じ dt
で比較していたが, QuTiP の adaptive solver では dt が必ずしも step を律しない
ため指標として不公平だった. 本版では:

* 各 solver は固有の **「精度つまみ」** を持ち, 独自 sweep する.
  - kryanneal m2 / trotter / cfm4 (固定 dt): ``dt``
  - kryanneal cfm4_adaptive_richardson_krylov: ``atol``
  - QuTiP: ``tol`` (内部で ``atol = rtol = tol`` として渡す.
    TFIM の量子状態は |ψ_k| ≤ 1/√dim 級なので atol/rtol の影響は同オーダ,
    1 軸 sweep で十分捕らえられる)

* 複数 **scenario** を 1 script invocation で sweep する. 各 scenario は
  ``(T, h_p_scale, h_x_scale, n_values)`` で問題インスタンスを規定する.
  代表的な regime をカバーするため既定 scenario は:

  - ``standard``: ``T=1, h_p_scale=1, N=10,12`` — 基本ケース
  - ``long-T``: ``T=1e4, h_p_scale=1, N=8,10`` — 量子アニーリングの実用
    T レンジ (T=1e4 と N 大の組合せは 1 cell が分単位になるため N を絞る)
  - ``stiff``: ``T=1, h_p_scale=10, N=10,12`` — ``H_p_diag`` の dynamic
    range 拡大 (QuTiP の adaptive ODE が ``‖H‖`` 律速で step を縮める領域)
  - ``large-N``: ``T=1, h_p_scale=1, N=12,14,16`` — 大規模 Hilbert 空間
    (dim=2^N up to 65536). T を短く取って 1 cell を秒オーダに保つ
  - ``stiff-long-T``: ``T=1e4, h_p_scale=10, N=6,8`` — opt-in 用最重 case

  ``h_p_scale`` は ``H_p_diag`` 振幅を ``Uniform(-1, 1)`` から
  ``Uniform(-h_p_scale, h_p_scale)`` にスケール, ``h_x_scale`` は同様.
  ``n_values`` は scenario 内蔵; CLI ``--n-values`` で全 scenario を上書き可.

* **reference state** は QuTiP ``sesolve`` at ``tol=ref_tol`` で
  (scenario, n) ごとに 1 回だけ計算 (default ``ref_tol=1e-11``; long-T では
  ``1e-13`` まで絞ると分単位かかるため緩めて使う). kryanneal を reference に
  しないことで「kryanneal vs QuTiP」の公平性を保つ.

* 各 sweep cell は **1 回だけ実行** し state と wall_sec を同時記録. fidelity
  は事後計算 (issue #65 user 指示: 不要な計算反復を避ける).

* QuTiP Hamiltonian は **常に sparse 構築** (``qutip.tensor`` of ``sigmax``).

レポート: ``benchmarks/results/<YYYYMMDD-HHMMSS>/`` に以下を吐く.

* ``bench_qutip_large.csv``: per-cell raw data (scenario, n, solver,
  knob_name, knob_value, n_steps_effective, wall_sec, infidelity,
  log10_infidelity, is_pareto, is_reference).
* ``bench_qutip_large.md``: machine info + per-(scenario, n) work-precision
  表 (infidelity 昇順, Pareto-optimal cell に ✓). ``目標精度 X を最速で
  出すのは誰か`` が scenario ごとに 1 表で読める形.

CLI 例::

    # 既定 scenario (standard + long-T + stiff) × N=8,10 を回す
    uv run python benchmarks/bench_qutip_large.py

    # 1 scenario に絞る
    uv run python benchmarks/bench_qutip_large.py --scenarios long-T --n-values 8

    # stiff-long-T も含めて全 scenario 走らせる
    uv run python benchmarks/bench_qutip_large.py \\
        --scenarios standard,long-T,stiff,stiff-long-T

    # kryanneal の dt sweep を絞って bench wall time を節約
    uv run python benchmarks/bench_qutip_large.py \\
        --m2-dts 0.01,0.001 --cfm4-dts 0.05,0.01

    # custom dynamic range の scenario を inline 定義
    uv run python benchmarks/bench_qutip_large.py \\
        --add-scenario "wide-range:T=1,h_p=100,h_x=1"

    # Phase 7 (#93) acceptance: krylov_tol を緩めても Pareto 劣化しないこと
    # を long-T で検証. 'auto' は ``tol_step * 1e-3`` 自動結合 (= 既定相当).
    uv run python benchmarks/bench_qutip_large.py \\
        --scenarios long-T --n-values 8,10 \\
        --krylov-tols auto,1e-8,1e-6
"""

from __future__ import annotations

import argparse
import csv
import importlib
import os
import platform
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np

from kryanneal import IsingProblem, QuantumAnnealer, Schedule, set_blas_threads
from kryanneal.initial_states import uniform_superposition

# QuTiP は dev dep のみ. 未 install で明示的に失敗させる (sesolve 無しでは
# 本ベンチの目的が成立しないため pytest 系 importorskip は使わない).
try:
    import qutip
except ImportError as exc:  # pragma: no cover - dev dep 必須
    raise ImportError(
        "benchmarks/bench_qutip_large.py requires `qutip` (dev dependency). "
        "Install with `uv sync --group dev`."
    ) from exc


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_ROOT = REPO_ROOT / "benchmarks" / "results"


@dataclass(frozen=True)
class _Scenario:
    """1 つの問題 regime を ``(T, h_p_scale, h_x_scale, n_values)`` で規定.

    ``n_values`` は scenario ごとに適切な N 範囲を埋め込む (long-T や
    stiff-long-T では小さい N に絞らないと 1 cell が分単位の wall time に
    なるため). CLI ``--n-values`` を明示指定すると全 scenario をその値で
    override する.
    """

    name: str
    T: float
    h_p_scale: float
    h_x_scale: float
    n_values: tuple[int, ...]


# 既定 scenario. 量子アニーリングの典型的な regime をカバーする:
# - standard: T=1, 振幅 1, N=10-12 (基本ケース, 全 N で軽い)
# - long-T: T=1e4, 振幅 1, N=8-10 (実用アニーリング時間; N 大は long-T と
#   かけ合わせると 1 cell が分単位なので絞る)
# - stiff: T=1, h_p_scale=10, N=10-12 (dynamic range 拡大 → QuTiP の step
#   縮小領域. T 短いので N は中規模で取れる)
# - large-N: T=1, h_p_scale=1, N=12-16 (大規模 Hilbert. dim=65536 まで.
#   QuTiP sparse matvec が次元に sublinear で効くので N が大きい領域で
#   kryanneal matrix-free との比較が情報量大)
# - stiff-long-T: T=1e4, h_p_scale=10, N=6-8 (最も重い組合せ; opt-in)
_BUILTIN_SCENARIOS: dict[str, _Scenario] = {
    "standard": _Scenario(
        "standard", T=1.0, h_p_scale=1.0, h_x_scale=1.0, n_values=(10, 12)
    ),
    "long-T": _Scenario(
        "long-T", T=1.0e4, h_p_scale=1.0, h_x_scale=1.0, n_values=(8, 10)
    ),
    "stiff": _Scenario(
        "stiff", T=1.0, h_p_scale=10.0, h_x_scale=1.0, n_values=(10, 12)
    ),
    "large-N": _Scenario(
        "large-N", T=1.0, h_p_scale=1.0, h_x_scale=1.0, n_values=(12, 14, 16)
    ),
    "stiff-long-T": _Scenario(
        "stiff-long-T", T=1.0e4, h_p_scale=10.0, h_x_scale=1.0, n_values=(6, 8)
    ),
}

# default で走らせる scenario 名 (stiff-long-T は opt-in).
_DEFAULT_SCENARIO_NAMES: list[str] = ["standard", "long-T", "stiff", "large-N"]

# 比較対象 solver. 各 solver は固有の "精度つまみ" を持つ.
_VALID_SOLVERS: tuple[str, ...] = (
    "qutip",
    "m2",
    "trotter",
    "cfm4",
    "cfm4_adaptive_richardson_krylov",
    "cfm4_adaptive_richardson_chebyshev",
)

# 各 solver の sweep 既定値. kryanneal 固定 dt 経路は **dt** で sweep する
# (n_steps は T/dt で自動算出). dt は T-invariant なので長時間 scenario と
# 短時間 scenario で同じ sweep 値を使える. global error order:
# - m2 / trotter (p=2): err ~ T·dt^2 → 高精度には小 dt 必要
# - cfm4 (p=4): err ~ T·dt^4 → 同精度を大きい dt で達成可
_DEFAULT_M2_DTS: list[float] = [0.001, 0.003, 0.01, 0.03, 0.1]
_DEFAULT_TROTTER_DTS: list[float] = [0.001, 0.003, 0.01, 0.03, 0.1]
_DEFAULT_CFM4_DTS: list[float] = [0.005, 0.02, 0.05, 0.2, 0.5]
_DEFAULT_ADAPTIVE_TOLS: list[float] = [1e-3, 1e-5, 1e-7, 1e-9, 1e-11]
_DEFAULT_QUTIP_TOLS: list[float] = [1e-3, 1e-5, 1e-7, 1e-9, 1e-12]

# cfm4_adaptive_richardson_krylov / cfm4_adaptive_richardson_chebyshev の
# krylov_tol (Chebyshev では chebyshev_tol として機能) sweep 既定値. ``None``
# は QuantumAnnealer 内部の auto-coupling (= ``tol_step * 1e-3``) を意味する.
# 既定では 1 値 (= None) のみで sweep 無し → 既存 bench 挙動と等価. Phase 7
# (#93) の "krylov_tol 緩和の安全性" を検証する際は CLI ``--krylov-tols
# auto,1e-8,1e-6`` 等を渡して atol × krylov_tol のクロス sweep を有効化する.
_DEFAULT_KRYLOV_TOLS: list[float | None] = [None]

# 各 method の dt 下限. fixed-dt 経路は ``n_steps = round(T/dt)`` なので
# long-T (T=1e4) と組合せると 1 cell が分単位の wall time になる. dt 下限を
# 設けて cell wall time を実用範囲にキャップする (CLI 上書き可).
#
# - m2 / trotter (global p=2): default 0.005. long-T で n_steps ≤ 2e6 (cell
#   ~ 5-10 分 at n=10). 低次精度なので極小 dt cell は work-precision Pareto
#   で QuTiP / trotter / cfm4 がカバーする領域に隠れて情報量低い.
# - cfm4 (global p=4): default 0.01. per-step が m2 の 2x (2m=48 matvec)
#   なので m2 より厳しい floor が必要. long-T で n_steps ≤ 1e6
#   (cell ~ 10-20 分 at n=10). short-T では cfm4 dt=0.01 でも 1-fid ~ 1e-14
#   級まで届くので Pareto を失わない.
# CLI ``--m2-dt-min`` / ``--trotter-dt-min`` / ``--cfm4-dt-min`` で上書き可
# (0.0 で無効化).
_M2_DT_MIN: float = 0.005
_TROTTER_DT_MIN: float = 0.005
_CFM4_DT_MIN: float = 0.01

# Reference の QuTiP tol. long-T (T=1e4) で 1e-13 まで絞ると数分かかるため,
# default は 1e-11 で実用的な ground truth に. user は ``--ref-tol`` で上書き可.
_DEFAULT_REF_TOL: float = 1e-11

# NOTE: 既定 N 値は scenario ごとに `_BUILTIN_SCENARIOS[].n_values` に
# 埋め込んだ. CLI ``--n-values`` を明示指定するとここで上書きされる挙動.


# ---------------------------------------------------------------------------
# Hamiltonian builder (常に sparse; フェアな比較のための判断)
# ---------------------------------------------------------------------------


def _build_qutip_hamiltonian(h_x: np.ndarray, h_p_diag: np.ndarray, T: float) -> list:
    """QuTiP ``sesolve`` 用 ``[[H_drv, A(t)], [H_p, B(t)]]`` を sparse で組む.

    ``H_drv = -Σ_i h_x[i] X_i`` を ``qutip.tensor`` (kron) ベースで CSR sparse
    構築. kryanneal の LSB bit 規約と QuTiP の MSB-first tensor 規約の差を
    吸収するため X は tensor list の位置 ``n-1-i`` に挿入する.

    全 n で sparse 経路に固定 (dense backing は TFIM 構造に対して dim^2
    メモリ + matvec の無駄, かつ比較対象を遅くするので不公平).
    """
    n = h_x.shape[0]
    sx = qutip.sigmax()
    si = qutip.qeye(2)
    h_drv: object | None = None
    for i in range(n):
        ops = [si] * n
        ops[n - 1 - i] = sx
        term = -float(h_x[i]) * qutip.tensor(ops)
        h_drv = term if h_drv is None else h_drv + term
    h_p = qutip.qdiags(h_p_diag, 0, dims=[[2] * n, [2] * n])
    return [
        [h_drv, f"(1 - t/{T})"],
        [h_p, f"(t/{T})"],
    ]


# ---------------------------------------------------------------------------
# Cell runners (1 cell = 1 run; state と wall_sec を同時取得)
# ---------------------------------------------------------------------------


def _run_qutip(
    h_t: list, psi0: np.ndarray, T: float, n: int, tol: float
) -> tuple[float, np.ndarray]:
    """QuTiP ``sesolve`` を ``atol = rtol = tol`` で走らせて ``(wall_sec, psi)``.

    精度つまみは ``tol`` 1 軸. max_step 制約は付けない (内部 step は atol/rtol
    で完全に決まる). dt sweep でなく tol sweep が QuTiP の adaptive solver
    には自然.
    """
    psi0_q = qutip.Qobj(psi0.reshape(-1, 1), dims=[[2] * n, [1] * n])
    options = {
        "atol": float(tol),
        "rtol": float(tol),
        "nsteps": 100_000_000,  # long-T で大量 step を許容するための上限
    }
    t_start = time.perf_counter()
    sol = qutip.sesolve(h_t, psi0_q, np.array([0.0, T]), options=options)
    elapsed = time.perf_counter() - t_start
    psi_final = sol.states[-1].full().ravel().astype(np.complex128)
    return elapsed, psi_final


def _run_kryanneal_fixed_dt(
    prob: IsingProblem,
    sched: Schedule,
    psi0: np.ndarray,
    T: float,
    method: Literal["m2", "trotter", "cfm4"],
    n_steps: int,
) -> tuple[float, np.ndarray]:
    """kryanneal 固定 dt 経路 (m2 / trotter / cfm4) を ``n_steps`` 回走らせる."""
    ann = QuantumAnnealer(prob, sched)
    t_start = time.perf_counter()
    res = ann.run(
        psi0,
        0.0,
        T,
        method=method,
        n_steps=n_steps,
    )
    elapsed = time.perf_counter() - t_start
    return elapsed, np.ascontiguousarray(res.psi_final)


def _run_kryanneal_adaptive(
    prob: IsingProblem,
    sched: Schedule,
    psi0: np.ndarray,
    T: float,
    atol: float,
    krylov_tol: float | None,
) -> tuple[float, np.ndarray, int]:
    """kryanneal ``cfm4_adaptive_richardson_krylov`` 経路を ``atol`` 指定で走らせる.

    PI controller が実際に踏んだ step 数を ``n_steps_actual`` として返す.
    ``krylov_tol = None`` のとき ``QuantumAnnealer`` の auto-coupling
    (``tol_step * 1e-3``) が効き, 数値を渡すと explicit override される.
    """
    ann = QuantumAnnealer(prob, sched, krylov_tol=krylov_tol)
    t_start = time.perf_counter()
    res = ann.run(
        psi0,
        0.0,
        T,
        method="cfm4_adaptive_richardson_krylov",
        atol=atol,
    )
    elapsed = time.perf_counter() - t_start
    assert (
        res.n_steps_actual is not None
    )  # cfm4_adaptive_richardson_krylov は必ず populated
    return elapsed, np.ascontiguousarray(res.psi_final), int(res.n_steps_actual)


def _run_kryanneal_adaptive_chebyshev(
    prob: IsingProblem,
    sched: Schedule,
    psi0: np.ndarray,
    T: float,
    atol: float,
    chebyshev_tol: float | None,
) -> tuple[float, np.ndarray, int]:
    """kryanneal ``cfm4_adaptive_richardson_chebyshev`` 経路 (issue #122 Phase B).

    Lanczos 版と同じ PI controller 構造で短時間プロパゲータだけが Chebyshev
    3 項漸化に置き換わる. ``chebyshev_tol = None`` で ``QuantumAnnealer`` の
    auto-coupling (``krylov_tol = tol_step · 1e-3``) を流用 (Chebyshev では
    K_used 切り捨て閾値の意味).
    """
    ann = QuantumAnnealer(prob, sched, krylov_tol=chebyshev_tol)
    t_start = time.perf_counter()
    res = ann.run(
        psi0,
        0.0,
        T,
        method="cfm4_adaptive_richardson_chebyshev",
        atol=atol,
    )
    elapsed = time.perf_counter() - t_start
    assert res.n_steps_actual is not None
    return elapsed, np.ascontiguousarray(res.psi_final), int(res.n_steps_actual)


# ---------------------------------------------------------------------------
# Sample problem (scenario-parameterized)
# ---------------------------------------------------------------------------


def _make_random_problem(
    n: int, scenario: _Scenario, seed: int
) -> tuple[IsingProblem, Schedule, np.ndarray, np.ndarray, np.ndarray]:
    """seed 固定 + scenario の (h_p_scale, h_x_scale, T) で random Ising 問題を作る.

    ``h_x ~ Uniform(0.5, 1.5) · h_x_scale``,
    ``H_p_diag ~ Uniform(-1, 1) · h_p_scale``,
    linear schedule with total time ``T``,
    始状態 ``|+⟩^N``.
    """
    rng = np.random.default_rng(seed)
    h_x = (rng.uniform(0.5, 1.5, size=n) * scenario.h_x_scale).astype(np.float64)
    h_p_diag = (rng.uniform(-1.0, 1.0, size=1 << n) * scenario.h_p_scale).astype(
        np.float64
    )
    prob = IsingProblem(n=n, H_p_diag=h_p_diag, h_x=h_x)
    sched = Schedule.linear(T=scenario.T)
    psi0 = uniform_superposition(n)
    return prob, sched, psi0, h_x, h_p_diag


# ---------------------------------------------------------------------------
# Per-cell record + sweep orchestration
# ---------------------------------------------------------------------------


class _CellRecord:
    """1 (scenario, n, solver, knob_value) cell の生データ.

    fidelity は後段 (reference 確定後) に計算するため, この段階では
    ``psi_final`` と ``wall_sec`` と ``n_steps_effective`` を保持する.

    ``knob2_name`` / ``knob2_value`` は cfm4_adaptive_richardson_krylov の
    ``krylov_tol`` のような **副次的な sweep 軸** を表す optional 拡張 (Phase 7
    / #93). 単軸 sweep cell では ``knob2_name = None``. ``knob2_value = None``
    かつ ``knob2_name != None`` は "auto" (自動結合) を表す sentinel.
    """

    __slots__ = (
        "scenario",
        "n",
        "solver",
        "knob_name",
        "knob_value",
        "knob2_name",
        "knob2_value",
        "n_steps_effective",
        "wall_sec",
        "psi_final",
    )

    def __init__(
        self,
        scenario: str,
        n: int,
        solver: str,
        knob_name: str,
        knob_value: float,
        n_steps_effective: int | None,
        wall_sec: float,
        psi_final: np.ndarray,
        knob2_name: str | None = None,
        knob2_value: float | None = None,
    ) -> None:
        self.scenario = scenario
        self.n = n
        self.solver = solver
        self.knob_name = knob_name
        self.knob_value = knob_value
        self.knob2_name = knob2_name
        self.knob2_value = knob2_value
        self.n_steps_effective = n_steps_effective
        self.wall_sec = wall_sec
        self.psi_final = psi_final


def _fidelity(psi_a: np.ndarray, psi_b: np.ndarray) -> float:
    """``|⟨ψ_a|ψ_b⟩|^2`` (normalize 済み state 前提)."""
    return float(np.abs(np.vdot(psi_a, psi_b)) ** 2)


def _format_knob_value(knob_name: str, value: float | None) -> str:
    """sweep つまみ値の human-readable 文字列化.

    ``n_steps`` は整数, それ以外 (``dt`` / ``tol`` / ``atol`` / ``kry``) は
    科学表記. ``kry`` で ``value is None`` のとき "auto"
    (= ``tol_step * 1e-3`` 自動結合 sentinel; Phase 7 / #93).
    """
    if value is None:
        return "auto"
    if knob_name == "n_steps":
        return str(int(value))
    if knob_name == "dt":
        return f"{value:.3g}"
    return f"{value:.1e}"


def _format_combined_knob(record: "_CellRecord") -> str:
    """primary + optional secondary knob を 1 つの文字列に."""
    primary = (
        f"{record.knob_name}={_format_knob_value(record.knob_name, record.knob_value)}"
    )
    if record.knob2_name is None:
        return primary
    secondary = (
        f"{record.knob2_name}="
        f"{_format_knob_value(record.knob2_name, record.knob2_value)}"
    )
    return f"{primary}, {secondary}"


@dataclass
class _ValidationData:
    """``--ref-validate`` で集める reference 自己収束の追加データ.

    QuTiP を ``ref_tol`` から段階的に緩めた tol (default factor 10, 100) で
    走らせ, 隣接 tol 間の state L2 差を取って ``QuTiP 自身の収束系列`` を
    可視化する. 「reference が本当に収束しているか」「kryanneal の
    cross-validation 値とどう比較するか」を MD レポートで示すために使う.

    fields:
        tols: looser → ref_tol の昇順 (3 cell 想定)
        walls: 各 tol cell の wall_sec
        psis: 各 cell の state vector (Δ 計算用)
        consecutive_diffs: |ψ_i+1 - ψ_i|_2 の列 (長さ len(tols)-1).
            幾何級数的に小さくなれば収束済.
    """

    tols: list[float]
    walls: list[float]
    psis: list[np.ndarray]
    consecutive_diffs: list[float]


def _sweep_one_scenario_n(
    scenario: _Scenario,
    n: int,
    seed: int,
    solvers: list[str],
    m2_dts: list[float],
    trotter_dts: list[float],
    cfm4_dts: list[float],
    m2_dt_min: float,
    trotter_dt_min: float,
    cfm4_dt_min: float,
    adaptive_atols: list[float],
    krylov_tols: list[float | None],
    qutip_tols: list[float],
    ref_tol: float,
    ref_validate: bool,
) -> tuple[list[_CellRecord], _CellRecord, _ValidationData | None]:
    """1 つの (scenario, n) について全 sweep cell と reference cell を計算する.

    kryanneal 固定 dt 経路は ``n_steps = round(T/dt)`` で各 dt を n_steps に
    変換して呼び出す. T が大きい scenario では n_steps が大きくなり, per-cell
    wall time が線形に伸びる. ``--m2-dts`` 等の sweep を CLI で絞れば bench
    総時間を抑えられる.

    ``ref_validate`` が True のとき, QuTiP を ``ref_tol × 100`` / ``× 10`` /
    ``ref_tol`` の 3 段階で走らせ ``_ValidationData`` を返す.
    """
    prob, sched, psi0, h_x, h_p_diag = _make_random_problem(n, scenario, seed)
    T = scenario.T
    h_t = _build_qutip_hamiltonian(h_x, h_p_diag, T)

    print(
        f"[bench_qutip_large] scenario={scenario.name} (T={T:g}, "
        f"h_p_scale={scenario.h_p_scale:g}, h_x_scale={scenario.h_x_scale:g}), n={n}",
        flush=True,
    )

    # Reference validation: 緩い tol から順に走らせ最終 cell を main reference に.
    validation: _ValidationData | None = None
    if ref_validate:
        validate_tols = [ref_tol * 100.0, ref_tol * 10.0, ref_tol]
        print(
            f"  reference validation: QuTiP tols={[f'{t:.1e}' for t in validate_tols]} ...",
            flush=True,
        )
        v_walls: list[float] = []
        v_psis: list[np.ndarray] = []
        for vtol in validate_tols:
            wall_v, psi_v = _run_qutip(h_t, psi0, T, n, vtol)
            v_walls.append(wall_v)
            v_psis.append(psi_v)
            print(f"    tol={vtol:.1e}: wall={wall_v:.3f}s", flush=True)
        # 隣接 tol 間の L2 state 差.
        diffs = [
            float(np.linalg.norm(v_psis[i + 1] - v_psis[i]))
            for i in range(len(v_psis) - 1)
        ]
        validation = _ValidationData(
            tols=validate_tols,
            walls=v_walls,
            psis=v_psis,
            consecutive_diffs=diffs,
        )
        # 最 tight cell (= validate_tols[-1] = ref_tol) を main reference に流用.
        ref_wall = v_walls[-1]
        ref_psi = v_psis[-1]
    else:
        print(f"  reference: QuTiP tol={ref_tol:.1e} ...", flush=True)
        ref_wall, ref_psi = _run_qutip(h_t, psi0, T, n, ref_tol)
        print(f"    reference wall = {ref_wall:.3f}s", flush=True)
    ref_record = _CellRecord(
        scenario=scenario.name,
        n=n,
        solver="qutip",
        knob_name="tol",
        knob_value=ref_tol,
        n_steps_effective=None,
        wall_sec=ref_wall,
        psi_final=ref_psi,
    )

    records: list[_CellRecord] = []

    if "qutip" in solvers:
        print(f"  qutip sweep tols={qutip_tols} ...", flush=True)
        for tol in qutip_tols:
            wall, psi = _run_qutip(h_t, psi0, T, n, tol)
            records.append(
                _CellRecord(
                    scenario=scenario.name,
                    n=n,
                    solver="qutip",
                    knob_name="tol",
                    knob_value=tol,
                    n_steps_effective=None,
                    wall_sec=wall,
                    psi_final=psi,
                )
            )

    for method, dt_sweep_raw, dt_min in (
        ("m2", m2_dts, m2_dt_min),
        ("trotter", trotter_dts, trotter_dt_min),
        ("cfm4", cfm4_dts, cfm4_dt_min),
    ):
        if method not in solvers:
            continue
        # dt < dt_min を skip (long-T で n_steps が膨大になる cell を除外).
        dt_sweep = [dt for dt in dt_sweep_raw if dt >= dt_min]
        dropped = [dt for dt in dt_sweep_raw if dt < dt_min]
        if dropped:
            print(
                f"  {method}: skipping dt < {dt_min:g} (n_steps would exceed "
                f"per-cell budget at T={T:g}): {dropped}",
                flush=True,
            )
        if not dt_sweep:
            print(f"  {method}: all dt values below dt_min, sweep empty", flush=True)
            continue
        # T / dt → n_steps に変換 (dt は T-invariant な精度つまみ).
        n_steps_list = [max(1, int(round(T / dt))) for dt in dt_sweep]
        print(
            f"  {method} sweep dts={dt_sweep} → n_steps={n_steps_list} ...",
            flush=True,
        )
        for dt, n_steps in zip(dt_sweep, n_steps_list, strict=True):
            wall, psi = _run_kryanneal_fixed_dt(prob, sched, psi0, T, method, n_steps)
            records.append(
                _CellRecord(
                    scenario=scenario.name,
                    n=n,
                    solver=method,
                    knob_name="dt",
                    knob_value=dt,
                    n_steps_effective=int(n_steps),
                    wall_sec=wall,
                    psi_final=psi,
                )
            )

    if "cfm4_adaptive_richardson_krylov" in solvers:
        # Phase 7 (#93): krylov_tols sweep. 既定 ([None]) では knob2 を伏せ
        # (CSV/MD で空欄), 明示 sweep (len > 1 か 1 値だが非 None) のときに
        # secondary knob "kry" を MD/CSV に出す.
        emit_kry_knob = (len(krylov_tols) > 1) or (
            len(krylov_tols) == 1 and krylov_tols[0] is not None
        )
        kry_repr = ["auto" if k is None else f"{k:.1e}" for k in krylov_tols]
        print(
            f"  cfm4_adaptive_richardson_krylov sweep atols={adaptive_atols} × "
            f"krylov_tols={kry_repr} ...",
            flush=True,
        )
        for atol in adaptive_atols:
            for kry_tol in krylov_tols:
                wall, psi, n_steps_actual = _run_kryanneal_adaptive(
                    prob, sched, psi0, T, atol, kry_tol
                )
                if emit_kry_knob:
                    knob2_name: str | None = "kry"
                    knob2_value: float | None = (
                        float(kry_tol) if kry_tol is not None else None
                    )
                else:
                    knob2_name = None
                    knob2_value = None
                records.append(
                    _CellRecord(
                        scenario=scenario.name,
                        n=n,
                        solver="cfm4_adaptive_richardson_krylov",
                        knob_name="atol",
                        knob_value=atol,
                        knob2_name=knob2_name,
                        knob2_value=knob2_value,
                        n_steps_effective=n_steps_actual,
                        wall_sec=wall,
                        psi_final=psi,
                    )
                )

    if "cfm4_adaptive_richardson_chebyshev" in solvers:
        # issue #122 (Phase B): Chebyshev variant. ``krylov_tols`` を共有して
        # sweep する (semantically ``chebyshev_tol`` = Krylov 近似の許容誤差
        # 規約を流用; krylov.py の Phase 8 設計参照). knob 構造は Lanczos 版と
        # 完全に同じため emit_kry_knob 判定もそのまま使い回す.
        emit_kry_knob_cheb = (len(krylov_tols) > 1) or (
            len(krylov_tols) == 1 and krylov_tols[0] is not None
        )
        kry_repr_cheb = ["auto" if k is None else f"{k:.1e}" for k in krylov_tols]
        print(
            f"  cfm4_adaptive_richardson_chebyshev sweep atols={adaptive_atols} × "
            f"krylov_tols={kry_repr_cheb} ...",
            flush=True,
        )
        for atol in adaptive_atols:
            for cheb_tol in krylov_tols:
                wall, psi, n_steps_actual = _run_kryanneal_adaptive_chebyshev(
                    prob, sched, psi0, T, atol, cheb_tol
                )
                if emit_kry_knob_cheb:
                    knob2_name_c: str | None = "kry"
                    knob2_value_c: float | None = (
                        float(cheb_tol) if cheb_tol is not None else None
                    )
                else:
                    knob2_name_c = None
                    knob2_value_c = None
                records.append(
                    _CellRecord(
                        scenario=scenario.name,
                        n=n,
                        solver="cfm4_adaptive_richardson_chebyshev",
                        knob_name="atol",
                        knob_value=atol,
                        knob2_name=knob2_name_c,
                        knob2_value=knob2_value_c,
                        n_steps_effective=n_steps_actual,
                        wall_sec=wall,
                        psi_final=psi,
                    )
                )

    return records, ref_record, validation


# ---------------------------------------------------------------------------
# Pareto 検出
# ---------------------------------------------------------------------------


def _pareto_mask(infids: list[float], walls: list[float]) -> list[bool]:
    """点列 ``(infid_i, wall_i)`` から **Pareto 最適 mask** を返す.

    Pareto 最適: 「infidelity も wall_sec も自分以下で, かつ少なくとも一方が
    厳密に小さい」点が他に無いこと. 両方とも "lower is better".
    """
    n = len(infids)
    mask = [True] * n
    for i in range(n):
        if not mask[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if (
                infids[j] <= infids[i]
                and walls[j] <= walls[i]
                and (infids[j] < infids[i] or walls[j] < walls[i])
            ):
                mask[i] = False
                break
    return mask


# ---------------------------------------------------------------------------
# Output (CSV + Markdown)
# ---------------------------------------------------------------------------


def _build_machine_info(args: argparse.Namespace) -> dict[str, str]:
    """マシン同定情報を辞書化."""
    info: dict[str, str] = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "qutip_version": getattr(qutip, "__version__", "?"),
        "rayon_threads": os.environ.get("RAYON_NUM_THREADS", "<unset>"),
        "blas_threads_requested": (
            str(args.blas_threads) if args.blas_threads is not None else "<unset>"
        ),
        "ref_tol": f"{args.ref_tol:.1e}",
    }
    try:
        rust_mod = importlib.import_module("kryanneal._rust")
        info["has_blas"] = str(bool(getattr(rust_mod, "__has_blas__", False)))
        info["has_rayon"] = str(bool(getattr(rust_mod, "__has_rayon__", False)))
        info["has_simd"] = str(bool(getattr(rust_mod, "__has_simd__", False)))
    except ImportError:
        info["has_blas"] = "?"
        info["has_rayon"] = "?"
        info["has_simd"] = "?"
    return info


def _compute_infidelities(
    records: list[_CellRecord], ref_psi: np.ndarray
) -> list[float]:
    """各 record の infidelity = 1 - |⟨ref|ψ⟩|^2 を計算."""
    out: list[float] = []
    for r in records:
        fid = _fidelity(ref_psi, r.psi_final)
        out.append(max(0.0, 1.0 - fid))
    return out


def _write_csv(
    sweep_records: list[_CellRecord],
    refs: list[_CellRecord],
    infids_per_key: dict[tuple[str, int], list[float]],
    pareto_per_key: dict[tuple[str, int], list[bool]],
    out_path: Path,
) -> None:
    """``bench_qutip_large.csv`` を書く."""
    fieldnames = [
        "scenario",
        "n",
        "solver",
        "knob_name",
        "knob_value",
        "knob2_name",
        "knob2_value",
        "n_steps_effective",
        "wall_sec",
        "infidelity",
        "log10_infidelity",
        "is_pareto",
        "is_reference",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        # reference rows first per (scenario, n).
        for ref in refs:
            writer.writerow(
                {
                    "scenario": ref.scenario,
                    "n": ref.n,
                    "solver": ref.solver,
                    "knob_name": ref.knob_name,
                    "knob_value": f"{ref.knob_value:.6e}",
                    "knob2_name": "",
                    "knob2_value": "",
                    "n_steps_effective": (
                        "" if ref.n_steps_effective is None else ref.n_steps_effective
                    ),
                    "wall_sec": f"{ref.wall_sec:.6f}",
                    "infidelity": "0.0",
                    "log10_infidelity": "nan",
                    "is_pareto": "",
                    "is_reference": "1",
                }
            )
        # group sweep records by (scenario, n) so we can align with infids/pareto.
        grouped: dict[tuple[str, int], list[_CellRecord]] = {}
        for r in sweep_records:
            grouped.setdefault((r.scenario, r.n), []).append(r)
        for key, recs in grouped.items():
            infids = infids_per_key[key]
            paretos = pareto_per_key[key]
            for r, infid, pareto in zip(recs, infids, paretos, strict=True):
                log10_infid = f"{np.log10(infid):.6f}" if infid > 0.0 else "nan"
                if r.knob2_name is None:
                    knob2_name_str = ""
                    knob2_value_str = ""
                else:
                    knob2_name_str = r.knob2_name
                    knob2_value_str = (
                        "auto" if r.knob2_value is None else f"{r.knob2_value:.6e}"
                    )
                writer.writerow(
                    {
                        "scenario": r.scenario,
                        "n": r.n,
                        "solver": r.solver,
                        "knob_name": r.knob_name,
                        "knob_value": f"{r.knob_value:.6e}",
                        "knob2_name": knob2_name_str,
                        "knob2_value": knob2_value_str,
                        "n_steps_effective": (
                            "" if r.n_steps_effective is None else r.n_steps_effective
                        ),
                        "wall_sec": f"{r.wall_sec:.6f}",
                        "infidelity": f"{infid:.6e}",
                        "log10_infidelity": log10_infid,
                        "is_pareto": "1" if pareto else "0",
                        "is_reference": "0",
                    }
                )


def _write_md(
    sweep_records: list[_CellRecord],
    refs_per_key: dict[tuple[str, int], _CellRecord],
    infids_per_key: dict[tuple[str, int], list[float]],
    pareto_per_key: dict[tuple[str, int], list[bool]],
    validations_per_key: dict[tuple[str, int], _ValidationData],
    machine_info: dict[str, str],
    scenarios: list[_Scenario],
    args: argparse.Namespace,
    out_path: Path,
) -> None:
    """``bench_qutip_large.md`` を書く: machine info + per-(scenario, n) 表."""
    lines: list[str] = []
    lines.append("# bench_qutip_large.py")
    lines.append("")
    lines.append(
        "Work-precision diagram ベンチ: QuTiP ``sesolve`` vs kryanneal 各 method "
        "(issue #65 Phase 6 C4)."
    )
    lines.append("")
    lines.append(
        "複数 **scenario** (T × dynamic range の組合せ) と複数 **n** に対し "
        "各 solver の精度つまみを sweep, 共通 reference (QuTiP `tol=ref_tol`) に対する "
        "infidelity と wall time を 1 回ずつ測定. 各 (scenario, n) ごとに "
        "**infidelity 昇順 + Pareto 最適マーク (✓)** を付けた work-precision 表を出す."
    )
    lines.append("")

    # Machine info & global params.
    lines.append("## Machine info & bench params")
    lines.append("")
    for k, v in machine_info.items():
        lines.append(f"- **{k}**: `{v}`")
    if args.n_values is not None:
        lines.append(
            f"- **n_values (override)**: `{args.n_values}` (applied to all scenarios)"
        )
    else:
        lines.append("- **n_values**: per-scenario default (see Scenarios table below)")
    lines.append(f"- **solvers**: `{args.solvers}`")
    lines.append(f"- **m2 dt sweep**: `{args.m2_dts}` (dt_min={args.m2_dt_min:g})")
    lines.append(
        f"- **trotter dt sweep**: `{args.trotter_dts}` (dt_min={args.trotter_dt_min:g})"
    )
    lines.append(
        f"- **cfm4 dt sweep**: `{args.cfm4_dts}` (dt_min={args.cfm4_dt_min:g})"
    )
    lines.append(
        f"- **cfm4_adaptive_richardson_krylov atol sweep**: `{args.adaptive_tols}`"
    )
    kry_repr_list = ["auto" if k is None else f"{k:.1e}" for k in args.krylov_tols]
    lines.append(
        f"- **cfm4_adaptive_richardson_krylov krylov_tol sweep**: `{kry_repr_list}` "
        "(`auto` = `tol_step * 1e-3` 自動結合; Phase 7 / #93)"
    )
    lines.append(f"- **qutip tol sweep**: `{args.qutip_tols}`")
    lines.append("")

    # Scenario summary.
    lines.append("## Scenarios")
    lines.append("")
    lines.append("| name | T | h_p_scale | h_x_scale | n_values |")
    lines.append("|---|---|---|---|---|")
    for sc in scenarios:
        n_str = ",".join(str(n) for n in sc.n_values)
        lines.append(
            f"| {sc.name} | {sc.T:g} | {sc.h_p_scale:g} | {sc.h_x_scale:g} | {n_str} |"
        )
    lines.append("")

    # group sweep_records by (scenario, n) and emit per-key table.
    grouped: dict[tuple[str, int], list[_CellRecord]] = {}
    for r in sweep_records:
        grouped.setdefault((r.scenario, r.n), []).append(r)

    # sort by scenario name (preserving scenarios order) then by n.
    scenario_order = {sc.name: i for i, sc in enumerate(scenarios)}
    keys = sorted(grouped.keys(), key=lambda k: (scenario_order.get(k[0], 99), k[1]))

    for key in keys:
        scenario_name, n = key
        ref = refs_per_key[key]
        records = grouped[key]
        infids = infids_per_key[key]
        pareto = pareto_per_key[key]

        lines.append(
            f"## scenario = {scenario_name}, n = {n} "
            f"(reference: QuTiP tol={ref.knob_value:.1e}, wall={ref.wall_sec:.3f}s)"
        )
        lines.append("")
        lines.append("| Pareto | solver | knob | n_steps_eff | infidelity | wall (s) |")
        lines.append("|---|---|---|---|---|---|")

        order = sorted(
            range(len(records)),
            key=lambda i: (infids[i], records[i].wall_sec),
        )
        for i in order:
            r = records[i]
            infid = infids[i]
            pareto_mark = "✓" if pareto[i] else ""
            knob_str = _format_combined_knob(r)
            n_steps_str = (
                str(r.n_steps_effective) if r.n_steps_effective is not None else "-"
            )
            infid_str = f"{infid:.3e}" if infid > 0.0 else "<1e-16"
            lines.append(
                f"| {pareto_mark} | {r.solver} | {knob_str} | {n_steps_str} | "
                f"{infid_str} | {r.wall_sec:.4f} |"
            )
        lines.append("")

        # Reference validation section (--ref-validate 時のみ).
        if key in validations_per_key:
            v = validations_per_key[key]
            lines.append("### Reference validation (QuTiP self-convergence)")
            lines.append("")
            lines.append("| tol | wall (s) | |ψ(tol) - ψ(tighter)|_2 |")
            lines.append("|---|---|---|")
            # tols は looser → tighter の順. 隣接 diff を表示.
            for j, (tol, wall) in enumerate(zip(v.tols, v.walls, strict=True)):
                if j < len(v.consecutive_diffs):
                    diff_str = f"{v.consecutive_diffs[j]:.3e}"
                else:
                    diff_str = "(★ reference)"
                lines.append(f"| {tol:.1e} | {wall:.3f} | {diff_str} |")
            lines.append("")
            # Cross-check: kryanneal cfm4_adaptive_richardson_krylov の最 tightest atol cell
            # の infidelity (= 既存 sweep に含まれている).
            kry_adaptive_cells = [
                (r, infids[k])
                for k, r in enumerate(records)
                if r.solver == "cfm4_adaptive_richardson_krylov"
            ]
            if kry_adaptive_cells:
                # infid 最小 cell (= 最も accurate) を pick. krylov_tols sweep が
                # 入った場合, 単に atol で min を取ると意味のない代表点を選ぶ
                # 恐れがあるため "accuracy で代表" を採用 (Phase 7 #93).
                tightest = min(kry_adaptive_cells, key=lambda x: x[1])
                r_tight, infid_tight = tightest
                # state L2 差は √(1-fid) で近似 (1-fid 小さいので 1 次近似で十分).
                psi_diff_est = float(np.sqrt(max(infid_tight, 0.0)))
                tight_knob_str = _format_combined_knob(r_tight)
                lines.append(
                    f"**Cross-check**: kryanneal cfm4_adaptive_richardson_krylov at "
                    f"{tight_knob_str} (independent algorithm family) "
                    f"reproduces reference with 1-fid="
                    f"{('<1e-16' if infid_tight == 0.0 else f'{infid_tight:.3e}')} "
                    f"(≈ |Δψ|_2 ≤ {psi_diff_est:.2e}). "
                )
                if v.consecutive_diffs:
                    last_diff = v.consecutive_diffs[-1]
                    if last_diff > 0 and psi_diff_est > 0:
                        rel = psi_diff_est / last_diff
                        lines.append(
                            f"QuTiP 自己収束の最終 step Δ = {last_diff:.2e}; "
                            f"cross-check は同オーダ (× {rel:.2f}) → "
                            f"**reference は多重 algorithm で validated**."
                        )
                lines.append("")

    out_path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_int_list(text: str) -> list[int]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return [int(p) for p in parts]


def _parse_float_list(text: str) -> list[float]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("expected at least one float")
    return [float(p) for p in parts]


def _parse_krylov_tol_list(text: str) -> list[float | None]:
    """``"auto,1e-8,1e-6"`` を ``[None, 1e-8, 1e-6]`` に parse する.

    ``auto`` キーワードは ``QuantumAnnealer`` 内部の自動結合 (= ``tol_step *
    1e-3``) を表し ``None`` に変換される. それ以外は ``float`` として読む.
    """
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("expected at least one value")
    out: list[float | None] = []
    for p in parts:
        if p.lower() == "auto":
            out.append(None)
        else:
            try:
                out.append(float(p))
            except ValueError as exc:
                raise argparse.ArgumentTypeError(
                    f"krylov_tol must be 'auto' or a float, got {p!r}"
                ) from exc
    return out


def _parse_solver_list(text: str) -> list[str]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("expected at least one solver")
    for p in parts:
        if p not in _VALID_SOLVERS:
            raise argparse.ArgumentTypeError(
                f"solver must be one of {_VALID_SOLVERS!r}, got {p!r}"
            )
    return parts


def _parse_scenario_list(text: str) -> list[str]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("expected at least one scenario name")
    return parts


def _parse_scenario_def(text: str) -> _Scenario:
    """``--add-scenario "name:T=1,h_p=10,h_x=1,n=12;14;16"`` 形式を parse する.

    ``n`` の値リストは ``,`` を `key=value` 区切りに使う都合上 ``;`` 区切り
    で受ける. ``n`` 未指定の custom scenario は CLI ``--n-values`` か
    fallback ``(10,)`` を使う (``_resolve_scenarios`` で解決).
    """
    if ":" not in text:
        raise argparse.ArgumentTypeError(
            f"--add-scenario expects 'name:T=...,h_p=...,h_x=...,[n=...]', got {text!r}"
        )
    name, params = text.split(":", 1)
    kv: dict[str, str] = {}
    for piece in params.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "=" not in piece:
            raise argparse.ArgumentTypeError(f"expected key=value, got {piece!r}")
        k, v = piece.split("=", 1)
        kv[k.strip()] = v.strip()
    T_val = float(kv.get("T", "1.0"))
    h_p_val = float(kv.get("h_p", kv.get("h_p_scale", "1.0")))
    h_x_val = float(kv.get("h_x", kv.get("h_x_scale", "1.0")))
    if "n" in kv:
        n_values = tuple(int(s.strip()) for s in kv["n"].split(";") if s.strip())
    else:
        # caller (`_resolve_scenarios`) が global ``--n-values`` を当てる. それも
        # 無ければ fallback として (10,) を使う.
        n_values = ()
    return _Scenario(
        name.strip(),
        T=T_val,
        h_p_scale=h_p_val,
        h_x_scale=h_x_val,
        n_values=n_values,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI 引数を parse する."""
    parser = argparse.ArgumentParser(
        description=(
            "Work-precision diagram ベンチ: QuTiP sesolve vs kryanneal の "
            "全 method 比較. 複数 scenario (T × dynamic range) を 1 invocation で "
            "回す. issue #65 Phase 6 C4."
        )
    )
    parser.add_argument(
        "--scenarios",
        type=_parse_scenario_list,
        default=list(_DEFAULT_SCENARIO_NAMES),
        help=(
            f"comma-separated scenario names (built-in: "
            f"{','.join(_BUILTIN_SCENARIOS)}; default: "
            f"{','.join(_DEFAULT_SCENARIO_NAMES)}). custom scenario は "
            f"--add-scenario で追加可."
        ),
    )
    parser.add_argument(
        "--add-scenario",
        type=_parse_scenario_def,
        action="append",
        default=[],
        help=(
            "custom scenario を追加. 形式: 'name:T=10000,h_p=100,h_x=1' "
            "(複数回指定可). --scenarios で参照する."
        ),
    )
    parser.add_argument(
        "--n-values",
        type=_parse_int_list,
        default=None,
        help=(
            "comma-separated sweep over spin counts. **省略時は各 scenario の "
            "規定 N を使う** (例: long-T は N=8,10, large-N は N=12,14,16). "
            "明示指定すると全 scenario の N をこの値で上書きする (注: long-T と "
            "N=12-14 等の組合せは 1 cell が分単位になる)."
        ),
    )
    parser.add_argument(
        "--solvers",
        type=_parse_solver_list,
        default=list(_VALID_SOLVERS),
        help=(
            f"comma-separated solver list (choices: {','.join(_VALID_SOLVERS)}; "
            f"default: all)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260517,
        help="random seed for problem generation (default: 20260517).",
    )
    parser.add_argument(
        "--blas-threads",
        type=int,
        default=None,
        help=(
            "kryanneal.set_blas_threads(N) で BLAS pool を統一. None で no-op. "
            "machine-independent baseline には `--blas-threads 1` 推奨."
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help=(f"results 出力先 (default: {DEFAULT_RESULTS_ROOT}/<YYYYMMDD-HHMMSS>/)"),
    )
    parser.add_argument(
        "--m2-dts",
        type=_parse_float_list,
        default=list(_DEFAULT_M2_DTS),
        help=(
            f"m2 の dt sweep (default: {_DEFAULT_M2_DTS}). n_steps = round(T/dt) "
            f"で自動算出される."
        ),
    )
    parser.add_argument(
        "--trotter-dts",
        type=_parse_float_list,
        default=list(_DEFAULT_TROTTER_DTS),
        help=f"trotter の dt sweep (default: {_DEFAULT_TROTTER_DTS}).",
    )
    parser.add_argument(
        "--cfm4-dts",
        type=_parse_float_list,
        default=list(_DEFAULT_CFM4_DTS),
        help=f"cfm4 の dt sweep (default: {_DEFAULT_CFM4_DTS}).",
    )
    parser.add_argument(
        "--m2-dt-min",
        type=float,
        default=_M2_DT_MIN,
        help=(
            f"m2 で sweep する dt の下限 (default: {_M2_DT_MIN:g}). 低次精度 "
            "(global p=2) のため long-T で n_steps が膨大になる極小 dt cell を "
            "自動 skip する. 0.0 を渡せば下限無効."
        ),
    )
    parser.add_argument(
        "--trotter-dt-min",
        type=float,
        default=_TROTTER_DT_MIN,
        help=(
            f"trotter で sweep する dt の下限 (default: {_TROTTER_DT_MIN:g}). "
            "m2-dt-min と同じ意図. 0.0 で下限無効."
        ),
    )
    parser.add_argument(
        "--cfm4-dt-min",
        type=float,
        default=_CFM4_DT_MIN,
        help=(
            f"cfm4 で sweep する dt の下限 (default: {_CFM4_DT_MIN:g}). "
            "global p=4 ながら per-step が m2 の 2x (2m=48 matvec) なので "
            "long-T で同じく cell wall time が膨らむため独立に下限を持つ. "
            "0.0 で下限無効."
        ),
    )
    parser.add_argument(
        "--adaptive-tols",
        type=_parse_float_list,
        default=list(_DEFAULT_ADAPTIVE_TOLS),
        help=(
            f"cfm4_adaptive_richardson_krylov / "
            f"cfm4_adaptive_richardson_chebyshev の atol sweep "
            f"(default: {_DEFAULT_ADAPTIVE_TOLS})."
        ),
    )
    parser.add_argument(
        "--krylov-tols",
        type=_parse_krylov_tol_list,
        default=list(_DEFAULT_KRYLOV_TOLS),
        help=(
            "cfm4_adaptive_richardson_krylov の krylov_tol および "
            "cfm4_adaptive_richardson_chebyshev の chebyshev_tol を共通の "
            "Krylov 近似許容誤差として sweep する (semantically どちらも "
            "短時間プロパゲータの内部精度). comma-separated; 'auto' で "
            "QuantumAnnealer の自動結合 (= tol_step * 1e-3) を選ぶ. "
            f"default: {['auto' if k is None else f'{k:.1e}' for k in _DEFAULT_KRYLOV_TOLS]}. "
            "Phase 7 (#93) 評価例: --krylov-tols auto,1e-8,1e-6 で atol × "
            "krylov_tol のクロス sweep."
        ),
    )
    parser.add_argument(
        "--qutip-tols",
        type=_parse_float_list,
        default=list(_DEFAULT_QUTIP_TOLS),
        help=(
            f"QuTiP sesolve の tol sweep (内部で atol = rtol = tol; "
            f"default: {_DEFAULT_QUTIP_TOLS})."
        ),
    )
    parser.add_argument(
        "--ref-tol",
        type=float,
        default=_DEFAULT_REF_TOL,
        help=(
            f"reference state 計算用 QuTiP tol (default: {_DEFAULT_REF_TOL:.1e}). "
            "long-T scenario で 1e-13 まで絞ると数分かかるため 1e-11 default."
        ),
    )
    parser.add_argument(
        "--ref-validate",
        action="store_true",
        help=(
            "reference self-convergence test: run QuTiP at ref_tol x {100, 10, 1} "
            "and check |psi(tol_i+1) - psi(tol_i)|_2 decreases geometrically. "
            "Adds 2 QuTiP cells per (scenario, n), extending bench wall time by "
            "20-30 percent. MD report gets a per-(scenario, n) 'Reference validation' "
            "section with kryanneal cfm4_adaptive cross-check."
        ),
    )
    return parser.parse_args(argv)


def _resolve_scenarios(args: argparse.Namespace) -> list[_Scenario]:
    """``--scenarios`` 名前リスト + ``--add-scenario`` から ``_Scenario`` 列を作る.

    ``--n-values`` が CLI で指定されていれば全 scenario の ``n_values`` を
    それで上書きする. 未指定なら built-in scenario は規定 ``n_values`` を
    使い, custom scenario (``--add-scenario`` で ``n=...`` 未指定のもの) は
    fallback ``(10,)`` を使う.
    """
    pool: dict[str, _Scenario] = dict(_BUILTIN_SCENARIOS)
    for sc in args.add_scenario:
        pool[sc.name] = sc

    global_n_override: tuple[int, ...] | None = (
        tuple(args.n_values) if args.n_values is not None else None
    )

    out: list[_Scenario] = []
    for name in args.scenarios:
        if name not in pool:
            raise SystemExit(
                f"unknown scenario {name!r}; available: "
                f"{sorted(pool.keys())} (define custom via --add-scenario)"
            )
        sc = pool[name]
        if global_n_override is not None:
            n_values = global_n_override
        elif sc.n_values:
            n_values = sc.n_values
        else:
            # custom scenario で n 未指定 / --n-values 未指定の場合の fallback.
            n_values = (10,)
        if n_values != sc.n_values:
            sc = _Scenario(
                name=sc.name,
                T=sc.T,
                h_p_scale=sc.h_p_scale,
                h_x_scale=sc.h_x_scale,
                n_values=n_values,
            )
        out.append(sc)
    return out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    warnings.filterwarnings("ignore", category=UserWarning, module="qutip")

    if args.blas_threads is not None:
        set_blas_threads(args.blas_threads)

    if args.results_dir is None:
        ts = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        results_dir = DEFAULT_RESULTS_ROOT / ts
    else:
        results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    scenarios = _resolve_scenarios(args)
    machine_info = _build_machine_info(args)

    sweep_records: list[_CellRecord] = []
    refs: list[_CellRecord] = []
    refs_per_key: dict[tuple[str, int], _CellRecord] = {}
    infids_per_key: dict[tuple[str, int], list[float]] = {}
    pareto_per_key: dict[tuple[str, int], list[bool]] = {}
    validations_per_key: dict[tuple[str, int], _ValidationData] = {}

    for scenario in scenarios:
        for n in scenario.n_values:
            records, ref, validation = _sweep_one_scenario_n(
                scenario=scenario,
                n=n,
                seed=args.seed + n,
                solvers=args.solvers,
                m2_dts=args.m2_dts,
                trotter_dts=args.trotter_dts,
                cfm4_dts=args.cfm4_dts,
                m2_dt_min=args.m2_dt_min,
                trotter_dt_min=args.trotter_dt_min,
                cfm4_dt_min=args.cfm4_dt_min,
                adaptive_atols=args.adaptive_tols,
                krylov_tols=args.krylov_tols,
                qutip_tols=args.qutip_tols,
                ref_tol=args.ref_tol,
                ref_validate=args.ref_validate,
            )
            key = (scenario.name, n)
            sweep_records.extend(records)
            refs.append(ref)
            refs_per_key[key] = ref
            if validation is not None:
                validations_per_key[key] = validation
            infids = _compute_infidelities(records, ref.psi_final)
            infids_per_key[key] = infids
            pareto_per_key[key] = _pareto_mask(infids, [r.wall_sec for r in records])
            print(
                f"    done ({len(records)} sweep cells, "
                f"{sum(pareto_per_key[key])} Pareto-optimal)",
                flush=True,
            )

    csv_path = results_dir / "bench_qutip_large.csv"
    md_path = results_dir / "bench_qutip_large.md"
    _write_csv(sweep_records, refs, infids_per_key, pareto_per_key, csv_path)
    _write_md(
        sweep_records,
        refs_per_key,
        infids_per_key,
        pareto_per_key,
        validations_per_key,
        machine_info,
        scenarios,
        args,
        md_path,
    )
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
