"""kryanneal cfm4_adaptive_richardson_krylov の Krylov 部分空間縮退 (m_eff) 計測.

issue #65 review 中の議論: 量子断熱領域 (T 大, ψ が瞬時 H の固有状態に近い)
では Krylov 基底 {ψ, Hψ, H²ψ, ...} が縮退し, Lanczos の β_k 早期打切で
``m_eff`` が default ``m=24`` から大幅に縮むと予想される. 本 bench は実測値を
取って T 依存性を可視化する.

**QuTiP 比較ではなく kryanneal 内部 driver の挙動分析専用** のため
``bench_qutip_large.py`` とは別ファイル (出力先も別).

各 (n, T, atol, m_max) に対し:

* ``evolve_schedule_adaptive_richardson`` driver を直接呼び (QuantumAnnealer の
  入力検証層をバイパスして m_eff_history への直アクセスを確保)
* ``m_eff_history`` (shape ``(n_steps_actual,)``, 各値は full step + 2 half
  step = 6 Lanczos 呼出の m_eff_sum, max = 6m) を取得
* per Lanczos 呼出あたりの m_eff = ``m_eff_sum / 6`` で換算
* T 軸 + per-time-bin 分布を MD レポートに出す

出力:

* ``bench_m_eff.csv``: per (n, T, atol, m_max, step_idx) raw data
  (t_history, dt_history, m_eff_sum, m_eff_per_lanczos)
* ``bench_m_eff.md``: per (n, T, atol, m_max) summary table + per-time-bin
  histogram + cross-T 比較表 (m_eff vs T で adiabatic 領域の縮退を可視化)

ローカル macstudio で短時間に走る default を採用 (n=6,8 × T=1,100,1e4 ×
atol=1e-7, total 6 cell, ~2-3 分見込み).

CLI 例::

    uv run python benchmarks/bench_m_eff_adiabatic.py
    uv run python benchmarks/bench_m_eff_adiabatic.py --T-values 1,1e3,1e6
    uv run python benchmarks/bench_m_eff_adiabatic.py --m-max-values 8,16,24
"""

from __future__ import annotations

import argparse
import csv
import importlib
import os
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from kryanneal import IsingProblem, Schedule, set_blas_threads
from kryanneal.initial_states import uniform_superposition
from kryanneal.krylov import evolve_schedule_adaptive_richardson

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_ROOT = REPO_ROOT / "benchmarks" / "results"

# default 既定値. macstudio で 2-3 分で完走する size.
_DEFAULT_N_VALUES: list[int] = [6, 8]
_DEFAULT_T_VALUES: list[float] = [1.0, 100.0, 10000.0]
_DEFAULT_ATOLS: list[float] = [1e-7]
_DEFAULT_M_MAX_VALUES: list[int] = [24]
_DEFAULT_N_TIME_BINS: int = 10

# adaptive driver の per-step Lanczos 呼出回数. m_eff_history の値は
# 6 Lanczos の m_eff sum なので, 1 Lanczos 単位に直すには /6.
_LANCZOS_CALLS_PER_STEP: int = 6


def _make_random_problem(
    n: int, T: float, seed: int
) -> tuple[IsingProblem, Schedule, np.ndarray]:
    """seed 固定の random Ising 問題. bench_qutip_large.py と同じ前提
    (h_x ~ U(0.5, 1.5), H_p_diag ~ U(-1, 1), linear schedule, |+⟩^N 始状態).
    """
    rng = np.random.default_rng(seed)
    h_x = rng.uniform(0.5, 1.5, size=n).astype(np.float64)
    h_p_diag = rng.uniform(-1.0, 1.0, size=1 << n).astype(np.float64)
    prob = IsingProblem(n=n, H_p_diag=h_p_diag, h_x=h_x)
    sched = Schedule.linear(T=T)
    psi0 = uniform_superposition(n)
    return prob, sched, psi0


def _run_adaptive(
    prob: IsingProblem,
    sched: Schedule,
    psi0: np.ndarray,
    T: float,
    atol: float,
    m_max: int,
    krylov_tol_factor: float = 1e-3,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, int]:
    """``evolve_schedule_adaptive_richardson`` を直接呼んで内部履歴を取得.

    QuantumAnnealer 経由だと m_eff_history が ``QuantumResult.m_eff_stats``
    に summary 化されるため, driver を直接呼んで raw array を得る.

    ``krylov_tol_factor`` (default 1e-3) で kryanneal default ``atol·1e-3``
    を再現. 大きい値 (例 0.1, 1.0) を渡すと早期打切が発火しやすくなり
    Krylov 圧縮の挙動を観察できる.

    Returns
    -------
    wall_sec: 全体 wall time
    t_history: shape (K,) 各 accept 後の時刻
    dt_history: shape (K-1,) 各 step の dt
    m_eff_history: shape (K-1,) 各 step の m_eff sum (6 Lanczos)
    n_rejects: 累積 reject 回数
    """
    krylov_tol = atol * krylov_tol_factor
    # dt_init: T 依存 auto. _resolve_dt_init_auto と同じロジック.
    dt0 = min(max(0.1 * T**0.5, 1e-3), T)
    # dt_max: Gershgorin cap auto.
    norm_h = float(np.sum(np.abs(prob.h_x))) + float(np.max(np.abs(prob.H_p_diag)))
    dt_max = max(min(10.0 * dt0, 4.0 * m_max / max(norm_h, 1e-30)), dt0)

    t_start = time.perf_counter()
    # issue #93 (Phase 7): driver は 10-tuple. β_m / err_lanczos / err_magnus /
    # n_krylov_insufficient を Step 2 (#93) bench で出すために受け取る.
    (
        psi_final,
        t_history,
        dt_history,
        n_rejects,
        m_eff_history,
        beta_m_history,
        err_lanczos_history,
        err_magnus_history,
        n_krylov_insufficient,
        _snapshot,
    ) = evolve_schedule_adaptive_richardson(
        h_x=prob.h_x,
        h_p_diag=prob.H_p_diag,
        schedule=sched,
        psi0=psi0,
        t0=0.0,
        t1=T,
        m=m_max,
        krylov_tol=krylov_tol,
        tol_step=atol,
        dt0=dt0,
        dt_max=dt_max,
        observables=None,
        save_tlist=None,
        store_states=False,
    )
    wall = time.perf_counter() - t_start
    _ = psi_final
    return (
        wall,
        t_history,
        dt_history,
        m_eff_history,
        int(n_rejects),
        beta_m_history,
        err_lanczos_history,
        err_magnus_history,
        int(n_krylov_insufficient),
    )


def _per_time_bin_stats(
    t_history: np.ndarray,
    dt_history: np.ndarray,
    m_eff_history: np.ndarray,
    T: float,
    n_bins: int,
) -> list[dict[str, float]]:
    """annealing 区間 ``[0, T]`` を ``n_bins`` 等分し各 bin の median(m_eff_sum) と
    bin 内 step 数, mean dt を計算する.

    ``t_history`` は accept 後の時刻列 (K 点; t_history[0]=0, t_history[-1]=T).
    各 step i (∈ [0, K-1)) は時刻区間 ``[t_history[i], t_history[i+1]]`` をカバー
    し dt = t_history[i+1] - t_history[i]. 各 step を「step の終端時刻」が属する
    bin に振り分ける.
    """
    if t_history.size == 0:
        return []
    bin_edges = np.linspace(0.0, T, n_bins + 1)
    out: list[dict[str, float]] = []
    # step 数 = t_history.size - 1 = dt_history.size = m_eff_history.size の想定
    step_end_times = t_history[1:]
    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        mask = (step_end_times >= lo) & (
            step_end_times < hi if b < n_bins - 1 else step_end_times <= hi
        )
        n_in = int(mask.sum())
        if n_in == 0:
            out.append(
                {
                    "bin_lo": float(lo),
                    "bin_hi": float(hi),
                    "n_steps": 0,
                    "mean_dt": float("nan"),
                    "median_m_eff_sum": float("nan"),
                    "median_m_eff_per_lanczos": float("nan"),
                }
            )
            continue
        m_eff_in = m_eff_history[mask]
        dt_in = dt_history[mask]
        out.append(
            {
                "bin_lo": float(lo),
                "bin_hi": float(hi),
                "n_steps": n_in,
                "mean_dt": float(np.mean(dt_in)),
                "median_m_eff_sum": float(np.median(m_eff_in)),
                "median_m_eff_per_lanczos": float(
                    np.median(m_eff_in) / _LANCZOS_CALLS_PER_STEP
                ),
            }
        )
    return out


def _summary_stats(m_eff_history: np.ndarray, m_max: int) -> dict[str, float]:
    """m_eff_history の概要 stats (per-step sum と per-Lanczos 換算)."""
    if m_eff_history.size == 0:
        return {
            "n_steps": 0,
            "m_eff_sum_mean": float("nan"),
            "m_eff_sum_median": float("nan"),
            "m_eff_sum_min": float("nan"),
            "m_eff_sum_max": float("nan"),
            "m_eff_sum_p10": float("nan"),
            "m_eff_sum_p90": float("nan"),
            "m_eff_per_lanczos_median": float("nan"),
            "compression_ratio": float("nan"),
        }
    upper_bound = m_max * _LANCZOS_CALLS_PER_STEP  # 1 step あたりの理論上限
    mean = float(np.mean(m_eff_history))
    return {
        "n_steps": int(m_eff_history.size),
        "m_eff_sum_mean": mean,
        "m_eff_sum_median": float(np.median(m_eff_history)),
        "m_eff_sum_min": float(np.min(m_eff_history)),
        "m_eff_sum_max": float(np.max(m_eff_history)),
        "m_eff_sum_p10": float(np.percentile(m_eff_history, 10)),
        "m_eff_sum_p90": float(np.percentile(m_eff_history, 90)),
        "m_eff_per_lanczos_median": float(
            np.median(m_eff_history) / _LANCZOS_CALLS_PER_STEP
        ),
        "compression_ratio": mean / upper_bound,
    }


def _beta_m_stats(
    beta_m_history: np.ndarray,
    err_lanczos_history: np.ndarray,
    err_magnus_history: np.ndarray,
    tol_step: float,
) -> dict[str, float]:
    """β_m / err_lanczos / err_magnus history の概要 stats (issue #93 Step 2).

    ``beta_m_history`` は driver 内部で保存された代表 β_m_eff =
    err_lanczos_total / dt (raw β_m そのものではなく aggregated 値). raw β_m
    のような直接的な Krylov 漏れ強度ではなく, "Lanczos 誤差を dt で割った
    単位時間あたりの誤差率" として解釈する.
    """
    if beta_m_history.size == 0:
        return {
            "beta_m_mean": float("nan"),
            "beta_m_median": float("nan"),
            "beta_m_min": float("nan"),
            "beta_m_max": float("nan"),
            "beta_m_p10": float("nan"),
            "beta_m_p90": float("nan"),
            "err_lanczos_median": float("nan"),
            "err_lanczos_max": float("nan"),
            "err_magnus_median": float("nan"),
            "err_magnus_max": float("nan"),
            "frac_lanczos_dominated": float("nan"),
        }
    # err_lanczos > err_magnus の step 割合 = Lanczos 誤差が支配的な step.
    # default krylov_tol ではほぼ 0 (Lanczos 充分), 緩い krylov_tol で増加する.
    err_total = err_lanczos_history + err_magnus_history
    nonzero = err_total > 0.0
    if nonzero.any():
        frac_lanczos_dominated = float(
            np.sum(err_lanczos_history[nonzero] > err_magnus_history[nonzero])
        ) / float(nonzero.sum())
    else:
        frac_lanczos_dominated = 0.0
    # Lanczos 不足 (err_lanczos > tol_step) の比率も同時に記録 (driver の
    # n_krylov_insufficient と等価指標, MD per-cell で表示).
    _ = tol_step  # 互換用シグネチャに残すが本式では未使用 (driver 側でカウント).
    return {
        "beta_m_mean": float(np.mean(beta_m_history)),
        "beta_m_median": float(np.median(beta_m_history)),
        "beta_m_min": float(np.min(beta_m_history)),
        "beta_m_max": float(np.max(beta_m_history)),
        "beta_m_p10": float(np.percentile(beta_m_history, 10)),
        "beta_m_p90": float(np.percentile(beta_m_history, 90)),
        "err_lanczos_median": float(np.median(err_lanczos_history)),
        "err_lanczos_max": float(np.max(err_lanczos_history)),
        "err_magnus_median": float(np.median(err_magnus_history)),
        "err_magnus_max": float(np.max(err_magnus_history)),
        "frac_lanczos_dominated": frac_lanczos_dominated,
    }


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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "kryanneal cfm4_adaptive_richardson_krylov の m_eff (Krylov 部分空間実効次元) "
            "T 依存性 bench. 量子断熱領域での Krylov 縮退検証 (issue #65 review)."
        )
    )
    parser.add_argument(
        "--n-values",
        type=_parse_int_list,
        default=list(_DEFAULT_N_VALUES),
        help=f"sweep するスピン数 (default: {_DEFAULT_N_VALUES})",
    )
    parser.add_argument(
        "--T-values",
        type=_parse_float_list,
        default=list(_DEFAULT_T_VALUES),
        help=(
            f"sweep する総アニーリング時間 (default: {_DEFAULT_T_VALUES}). "
            "T 大で adiabatic 領域に入り m_eff 縮退が予想される."
        ),
    )
    parser.add_argument(
        "--atols",
        type=_parse_float_list,
        default=list(_DEFAULT_ATOLS),
        help=f"PI controller の atol sweep (default: {_DEFAULT_ATOLS})",
    )
    parser.add_argument(
        "--m-max-values",
        type=_parse_int_list,
        default=list(_DEFAULT_M_MAX_VALUES),
        help=(
            f"Lanczos 部分空間 m_max sweep (default: {_DEFAULT_M_MAX_VALUES}). "
            "小さい m_max を渡すと早期打切が起こる前に m_max に達するため "
            "実効的な Krylov 圧縮率の評価が変わる."
        ),
    )
    parser.add_argument(
        "--krylov-tol-factor",
        type=float,
        default=1e-3,
        help=(
            "krylov_tol = atol × factor で β_k 早期打切閾値を制御 "
            "(default: 1e-3, kryanneal の QuantumAnnealer default と一致). "
            "1.0 や 0.1 を渡すと早期打切が発火しやすくなり Krylov 圧縮の "
            "観察が容易になる (精度は犠牲になる)."
        ),
    )
    parser.add_argument(
        "--n-time-bins",
        type=int,
        default=_DEFAULT_N_TIME_BINS,
        help=f"annealing 区間の時間 bin 数 (default: {_DEFAULT_N_TIME_BINS})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260518,
        help="random seed for problem generation (default: 20260518)",
    )
    parser.add_argument(
        "--blas-threads",
        type=int,
        default=None,
        help="kryanneal.set_blas_threads(N) で BLAS pool 統一. None で no-op.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help=f"results 出力先 (default: {DEFAULT_RESULTS_ROOT}/<YYYYMMDD-HHMMSS>/)",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _build_machine_info(args: argparse.Namespace) -> dict[str, str]:
    info: dict[str, str] = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "rayon_threads": os.environ.get("RAYON_NUM_THREADS", "<unset>"),
        "blas_threads_requested": (
            str(args.blas_threads) if args.blas_threads is not None else "<unset>"
        ),
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


def _write_csv(records: list[dict], out_path: Path) -> None:
    # issue #93 (Phase 7): β_m_eff / err_lanczos_total / err_magnus を列追加.
    fieldnames = [
        "n",
        "T",
        "atol",
        "m_max",
        "step_idx",
        "t",
        "dt",
        "m_eff_sum",
        "m_eff_per_lanczos",
        "beta_m_eff",
        "err_lanczos_total",
        "err_magnus",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(r)


def _write_md(
    cell_results: list[dict],
    machine_info: dict[str, str],
    args: argparse.Namespace,
    out_path: Path,
) -> None:
    """cell_results: per-(n, T, atol, m_max) の dict 列.

    各 dict: {n, T, atol, m_max, wall_sec, n_rejects, summary, time_bins}.
    """
    lines: list[str] = []
    lines.append("# bench_m_eff_adiabatic.py")
    lines.append("")
    lines.append(
        "kryanneal ``cfm4_adaptive_richardson_krylov`` の Krylov 部分空間実効次元 "
        "``m_eff`` を T 依存で計測 (issue #65 review). 量子断熱領域 (T 大) で "
        "ψ が瞬時 H の固有状態に近づき Lanczos β_k 早期打切で m_eff が default "
        "``m_max=24`` から縮むことを実測で示す."
    )
    lines.append("")
    lines.append(
        "**読み方**: ``m_eff_sum`` は 1 accept step あたりの 6 Lanczos 呼出 "
        "(full + 2 half step) の m_eff 合計, 理論上限 = 6 × m_max. "
        "``m_eff_per_lanczos`` = m_eff_sum / 6 は per-Lanczos 平均. "
        "``compression_ratio`` = m_eff_sum_mean / (6·m_max) で実効的な Krylov "
        "圧縮率を示す (1.0 で打切なし, 小さいほど縮退顕著)."
    )
    lines.append("")

    lines.append("## Machine info & bench params")
    lines.append("")
    for k, v in machine_info.items():
        lines.append(f"- **{k}**: `{v}`")
    lines.append(f"- **n_values**: `{args.n_values}`")
    lines.append(f"- **T_values**: `{args.T_values}`")
    lines.append(f"- **atols**: `{args.atols}`")
    lines.append(f"- **m_max_values**: `{args.m_max_values}`")
    lines.append(
        f"- **krylov_tol_factor**: `{args.krylov_tol_factor:.1e}` "
        f"(krylov_tol = atol × factor)"
    )
    lines.append(f"- **n_time_bins**: `{args.n_time_bins}`")
    lines.append("")

    # 主要 summary 表 (cross-T 比較が見やすい形式).
    lines.append("## Summary: m_eff vs T (Krylov 圧縮率の T 依存性)")
    lines.append("")
    lines.append(
        "| n | T | atol | m_max | n_steps | wall (s) | "
        "m_eff_sum (median) | m_eff/Lanczos (median) | "
        "compression_ratio | n_rejects |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for c in cell_results:
        s = c["summary"]
        lines.append(
            f"| {c['n']} | {c['T']:g} | {c['atol']:.1e} | {c['m_max']} | "
            f"{s['n_steps']} | {c['wall_sec']:.3f} | "
            f"{s['m_eff_sum_median']:.1f} | "
            f"{s['m_eff_per_lanczos_median']:.2f} | "
            f"{s['compression_ratio']:.3f} | "
            f"{c['n_rejects']} |"
        )
    lines.append("")

    # issue #93 (Phase 7) Step 2: β_m / err_lanczos / err_magnus 分布の表.
    lines.append("## Summary: β_m / err_lanczos / err_magnus (issue #93 Step 2)")
    lines.append("")
    lines.append(
        "β_m_eff = err_lanczos_total / dt は driver 内部での代表 β_m 値. "
        "err_lanczos_total は 6 Lanczos call の Saad/Hochbruck-Lubich 推定子 "
        "(`β_m · |c_m| · ‖ψ‖ · dt / m_eff`) の triangle inequality 和. "
        "err_magnus = max(0, err - err_lanczos_total) が PI controller の "
        "Magnus 駆動量. n_krylov_insufficient は err_lanczos_total > atol を "
        "検出した step 数 (Krylov 不足の診断指標)."
    )
    lines.append("")
    lines.append(
        "| n | T | atol | m_max | β_m_median | β_m_p90 | "
        "err_lanczos (median) | err_lanczos (max) | "
        "err_magnus (median) | n_krylov_insufficient |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for c in cell_results:
        b = c["beta_stats"]
        lines.append(
            f"| {c['n']} | {c['T']:g} | {c['atol']:.1e} | {c['m_max']} | "
            f"{b['beta_m_median']:.3e} | {b['beta_m_p90']:.3e} | "
            f"{b['err_lanczos_median']:.3e} | {b['err_lanczos_max']:.3e} | "
            f"{b['err_magnus_median']:.3e} | {c['n_krylov_insufficient']} |"
        )
    lines.append("")

    # per-cell 詳細 (per-time-bin histogram).
    for c in cell_results:
        s = c["summary"]
        lines.append(
            f"## cell: n={c['n']}, T={c['T']:g}, atol={c['atol']:.1e}, "
            f"m_max={c['m_max']}"
        )
        lines.append("")
        lines.append(
            f"- **n_steps_actual**: {s['n_steps']}, **wall**: {c['wall_sec']:.3f}s, "
            f"**n_rejects**: {c['n_rejects']}"
        )
        lines.append(
            f"- **m_eff_sum stats**: mean={s['m_eff_sum_mean']:.1f}, "
            f"median={s['m_eff_sum_median']:.1f}, "
            f"min={s['m_eff_sum_min']:.0f}, max={s['m_eff_sum_max']:.0f}, "
            f"P10={s['m_eff_sum_p10']:.1f}, P90={s['m_eff_sum_p90']:.1f}"
        )
        lines.append(
            f"- **m_eff_per_lanczos (median)**: "
            f"{s['m_eff_per_lanczos_median']:.2f} (vs m_max={c['m_max']})"
        )
        lines.append(
            f"- **compression_ratio** (mean / 6·m_max): {s['compression_ratio']:.3f}"
        )
        # issue #93 (Phase 7) Step 2: β_m / err_lanczos / err_magnus 詳細.
        b = c["beta_stats"]
        lines.append(
            f"- **β_m_eff stats**: mean={b['beta_m_mean']:.3e}, "
            f"median={b['beta_m_median']:.3e}, min={b['beta_m_min']:.3e}, "
            f"max={b['beta_m_max']:.3e}, P10={b['beta_m_p10']:.3e}, "
            f"P90={b['beta_m_p90']:.3e}"
        )
        lines.append(
            f"- **err_lanczos_total**: median={b['err_lanczos_median']:.3e}, "
            f"max={b['err_lanczos_max']:.3e}; "
            f"vs atol={c['atol']:.1e} → "
            f"n_krylov_insufficient={c['n_krylov_insufficient']} / {s['n_steps']}"
        )
        lines.append(
            f"- **err_magnus**: median={b['err_magnus_median']:.3e}, "
            f"max={b['err_magnus_max']:.3e} (= PI controller の駆動量)"
        )
        lines.append(
            f"- **frac steps with err_lanczos > err_magnus**: "
            f"{b['frac_lanczos_dominated']:.3f} "
            "(Lanczos 誤差支配的な step 割合; default 設定で ~0 が期待値)"
        )
        lines.append("")

        # per-time-bin histogram
        if c["time_bins"]:
            lines.append("### Per-time-bin breakdown")
            lines.append("")
            lines.append(
                "| t_bin | n_steps | mean dt | median m_eff_sum | median m_eff/Lanczos |"
            )
            lines.append("|---|---|---|---|---|")
            for b in c["time_bins"]:
                if b["n_steps"] == 0:
                    lines.append(
                        f"| [{b['bin_lo']:.3g}, {b['bin_hi']:.3g}) | 0 | - | - | - |"
                    )
                    continue
                lines.append(
                    f"| [{b['bin_lo']:.3g}, {b['bin_hi']:.3g}) | "
                    f"{b['n_steps']} | "
                    f"{b['mean_dt']:.3g} | "
                    f"{b['median_m_eff_sum']:.1f} | "
                    f"{b['median_m_eff_per_lanczos']:.2f} |"
                )
            lines.append("")

    out_path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.blas_threads is not None:
        set_blas_threads(args.blas_threads)

    if args.results_dir is None:
        ts = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        results_dir = DEFAULT_RESULTS_ROOT / ts
    else:
        results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    machine_info = _build_machine_info(args)

    cell_results: list[dict] = []
    csv_records: list[dict] = []

    for n in args.n_values:
        for T in args.T_values:
            for atol in args.atols:
                for m_max in args.m_max_values:
                    print(
                        f"[bench_m_eff] n={n}, T={T:g}, atol={atol:.1e}, m_max={m_max} ...",
                        flush=True,
                    )
                    prob, sched, psi0 = _make_random_problem(n, T, args.seed + n)
                    # issue #93 (Phase 7): _run_adaptive は 9-tuple. β_m /
                    # err_lanczos / err_magnus / n_krylov_insufficient を
                    # 受け取り bench MD/CSV に出力する.
                    (
                        wall,
                        t_history,
                        dt_history,
                        m_eff_history,
                        n_rejects,
                        beta_m_history,
                        err_lanczos_history,
                        err_magnus_history,
                        n_krylov_insufficient,
                    ) = _run_adaptive(
                        prob,
                        sched,
                        psi0,
                        T,
                        atol,
                        m_max,
                        krylov_tol_factor=args.krylov_tol_factor,
                    )
                    summary = _summary_stats(m_eff_history, m_max)
                    beta_stats = _beta_m_stats(
                        beta_m_history,
                        err_lanczos_history,
                        err_magnus_history,
                        tol_step=atol,
                    )
                    time_bins = _per_time_bin_stats(
                        t_history, dt_history, m_eff_history, T, args.n_time_bins
                    )
                    print(
                        f"  n_steps={summary['n_steps']}, wall={wall:.3f}s, "
                        f"m_eff/Lanczos median={summary['m_eff_per_lanczos_median']:.2f} "
                        f"(vs m_max={m_max}, ratio={summary['compression_ratio']:.3f}), "
                        f"β_m median={beta_stats['beta_m_median']:.2e}, "
                        f"n_krylov_insufficient={n_krylov_insufficient}",
                        flush=True,
                    )
                    cell_results.append(
                        {
                            "n": n,
                            "T": T,
                            "atol": atol,
                            "m_max": m_max,
                            "wall_sec": wall,
                            "n_rejects": n_rejects,
                            "n_krylov_insufficient": int(n_krylov_insufficient),
                            "summary": summary,
                            "beta_stats": beta_stats,
                            "time_bins": time_bins,
                        }
                    )
                    # CSV per-step rows. issue #93 (Phase 7) で β_m /
                    # err_lanczos / err_magnus も列追加.
                    for step_idx, (
                        t_end,
                        dt_step,
                        m_eff_sum,
                        beta_m_step,
                        err_lanczos_step,
                        err_magnus_step,
                    ) in enumerate(
                        zip(
                            t_history[1:],
                            dt_history,
                            m_eff_history,
                            beta_m_history,
                            err_lanczos_history,
                            err_magnus_history,
                            strict=True,
                        )
                    ):
                        csv_records.append(
                            {
                                "n": n,
                                "T": T,
                                "atol": atol,
                                "m_max": m_max,
                                "step_idx": step_idx,
                                "t": f"{float(t_end):.6e}",
                                "dt": f"{float(dt_step):.6e}",
                                "m_eff_sum": int(m_eff_sum),
                                "m_eff_per_lanczos": (
                                    f"{float(m_eff_sum) / _LANCZOS_CALLS_PER_STEP:.3f}"
                                ),
                                "beta_m_eff": f"{float(beta_m_step):.6e}",
                                "err_lanczos_total": f"{float(err_lanczos_step):.6e}",
                                "err_magnus": f"{float(err_magnus_step):.6e}",
                            }
                        )

    csv_path = results_dir / "bench_m_eff.csv"
    md_path = results_dir / "bench_m_eff.md"
    _write_csv(csv_records, csv_path)
    _write_md(cell_results, machine_info, args, md_path)
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
