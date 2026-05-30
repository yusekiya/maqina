"""層 B — step-size controller の end-to-end ノコギリ波計測ベンチ (issue #152).

合成誤差ハーネス (層 A, ``tests/_controller_harness.py``) が controller の
ダイナミクスだけを切り出すのに対し、本ベンチは **実 driver (実 Lanczos /
Chebyshev + 実 matvec)** を急峻 schedule で走らせ、現実のノコギリ波を
end-to-end で計測する補完手段。

急峻 schedule:
    ``s(t) = ½(1 + tanh(β·(t − T/2)))`` の tanh バーストで ``A(t)=1−s`` /
    ``B(t)=s`` の時間微分を ``t ≈ T/2`` に局所的に立てる。``β`` を上げるほど臨界
    領域で Magnus 4 次係数 ``C₄`` の立ち上がり率が増し、I 制御 + 固定 0.5 reject
    が追従できずノコギリ波 (受理率低下 / dt 振動) を起こす。N 非依存の現象なので
    小 N (例 N∈{4,6}) で高速に再現できる。

出力指標 (per method × N):
    - 受理率 / ``n_rejects`` / accept step 数
    - 臨界窓内の dt ノコギリ振幅 (``std(log dt)`` / max-min swing 比 /
      lag-1 自己相関 / 反転回数)
    - total propagator call 数 (≈ ``n_step_attempts · 6``)
    - 終端 infidelity (tight 基準 ψ に対する ``1 − |⟨ref|ψ⟩|²``)

旧 config vs 新 config 比較モード (``--compare``):
    同一 schedule・同一 problem で 2 つの controller config を走らせて差分を出す。
    **assert は絶対閾値ではなく同一実行内の差分 (old vs new) + マージン** にする
    方針 (別マシン非依存)。後続 sub-issue (#149 reject 予測式 + クランプ /
    #150 成長凍結 / #151 真の PI 化) は本モードに *その issue が追加する旧挙動
    knob* (``reject_shrink_min=max=0.5`` / ``freeze=False`` / ``pi_beta=0``) を
    渡して「旧挙動 → 新挙動で受理率回復 / ノコギリ振幅減 / 終端 infidelity 非劣化」
    を示す。#152 時点ではそれらの knob は未実装なので、本モードは現行 driver が
    受ける既存 knob (例 ``growth_max``) の差で **機構が動くことの実証** に留める。

実行例::

    uv run python benchmarks/bench_stepsize_controller.py
    uv run python benchmarks/bench_stepsize_controller.py --n-values 4,6 --beta 40
    uv run python benchmarks/bench_stepsize_controller.py --compare

Chebyshev 経路は Rust 拡張必須 (未ビルドなら自動 skip)。Richardson (Lanczos) は
Python リファレンス fallback で Rust 無しでも走る (小 N なら十分高速)。
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
# 層 A と共有する振動メトリクス util は tests/ 配下にある (issue #152)。
# tests は site-packages に入らないので path 注入して flat import する。
_TESTS_DIR = REPO_ROOT / "tests"
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from maqina import IsingProblem, Schedule  # noqa: E402
from maqina.initial_states import uniform_superposition  # noqa: E402
from maqina.krylov import (  # noqa: E402
    evolve_schedule_adaptive_richardson,
    evolve_schedule_adaptive_richardson_chebyshev,
)

from _controller_metrics import (  # noqa: E402  # ty: ignore[unresolved-import]
    ControllerTrace,
    log_dt_lag1_autocorr,
    n_reversals,
    std_log_dt,
)

DEFAULT_RESULTS_ROOT = REPO_ROOT / "benchmarks" / "results"
_VALID_METHODS = ("richardson", "chebyshev")


def _rust_available() -> bool:
    """``maqina._rust`` が import 可能か (Chebyshev 経路の前提)."""
    try:
        import maqina._rust  # noqa: F401  # ty: ignore[unresolved-import]
    except ImportError:
        return False
    return True


# ---------------------------------------------------------------------------
# problem / schedule 構築
# ---------------------------------------------------------------------------


def make_problem(n: int, seed: int) -> tuple[IsingProblem, np.ndarray]:
    """ランダム ``H_p_diag`` (実 ``[-1, 1]``) と一様 ``h_x = 1`` の問題を作る."""
    rng = np.random.default_rng(seed)
    dim = 1 << n
    h_p = rng.uniform(-1.0, 1.0, size=dim).astype(np.float64)
    h_x = np.ones(n, dtype=np.float64)
    return IsingProblem(n=n, H_p_diag=h_p), h_x


def build_steep_schedule(T: float, beta: float, h_x: np.ndarray) -> Schedule:
    """``s(t) = ½(1 + tanh(β(t − T/2)))`` の tanh バースト legacy schedule.

    ``β`` が大きいほど ``t ≈ T/2`` で ``s`` が急変し、臨界領域で ``C₄`` 立ち上がり
    率が増す。``A(s) = 1 − s``, ``B(s) = s``。
    """
    t_mid = 0.5 * T

    def s_of_t(t: float, _beta: float = beta, _mid: float = t_mid) -> float:
        return 0.5 * (1.0 + math.tanh(_beta * (t - _mid)))

    return Schedule(
        T=T,
        A=lambda s: 1.0 - s,
        B=lambda s: s,
        h_x=h_x,
        s=s_of_t,
    )


# ---------------------------------------------------------------------------
# driver 駆動
# ---------------------------------------------------------------------------


def _drive(
    method: str,
    prob: IsingProblem,
    sched: Schedule,
    psi0: np.ndarray,
    T: float,
    *,
    tol_step: float,
    dt0: float,
    dt_min: float,
    dt_max: float | None,
    safety: float,
    growth_max: float,
    max_rejects: int,
    reject_shrink_min: float,
    reject_shrink_max: float,
    freeze_growth_after_reject: bool,
    growth_freeze_steps: int,
    pi_alpha: float,
    pi_beta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """実 driver を 1 回駆動して ``(psi, t_hist, dt_hist, n_rejects)`` を返す."""
    if method == "richardson":
        out = evolve_schedule_adaptive_richardson(
            h_p_diag=prob.H_p_diag,
            schedule=sched,
            psi0=psi0,
            t0=0.0,
            t1=T,
            tol_step=tol_step,
            dt0=dt0,
            dt_min=dt_min,
            dt_max=dt_max,
            safety=safety,
            growth_max=growth_max,
            max_rejects=max_rejects,
            reject_shrink_min=reject_shrink_min,
            reject_shrink_max=reject_shrink_max,
            freeze_growth_after_reject=freeze_growth_after_reject,
            growth_freeze_steps=growth_freeze_steps,
            pi_alpha=pi_alpha,
            pi_beta=pi_beta,
        )
        return out[0], out[1], out[2], int(out[3])
    if method == "chebyshev":
        out = evolve_schedule_adaptive_richardson_chebyshev(
            h_p_diag=prob.H_p_diag,
            schedule=sched,
            psi0=psi0,
            t0=0.0,
            t1=T,
            h_p_min=prob.h_p_diag_min,
            h_p_max=prob.h_p_diag_max,
            tol_step=tol_step,
            dt0=dt0,
            dt_min=dt_min,
            dt_max=dt_max,
            safety=safety,
            growth_max=growth_max,
            max_rejects=max_rejects,
            reject_shrink_min=reject_shrink_min,
            reject_shrink_max=reject_shrink_max,
            freeze_growth_after_reject=freeze_growth_after_reject,
            growth_freeze_steps=growth_freeze_steps,
            pi_alpha=pi_alpha,
            pi_beta=pi_beta,
        )
        return out[0], out[1], out[2], int(out[3])
    raise ValueError(f"method must be one of {_VALID_METHODS}, got {method!r}")


def _reference_psi(
    prob: IsingProblem, sched: Schedule, psi0: np.ndarray, T: float
) -> np.ndarray:
    """tight 基準 ψ (終端 infidelity の ground truth).

    Richardson driver を ``tol_step = 1e-13`` の極めて厳しい設定で走らせて
    実質的な exact 終端 ψ を得る (Lanczos 経路で Rust 非依存)。
    """
    out = evolve_schedule_adaptive_richardson(
        h_p_diag=prob.H_p_diag,
        schedule=sched,
        psi0=psi0,
        t0=0.0,
        t1=T,
        tol_step=1e-13,
        dt0=min(0.05, 0.05 * T),
        dt_min=1e-7,
    )
    return out[0]


# ---------------------------------------------------------------------------
# 計測
# ---------------------------------------------------------------------------


def _critical_time_window(T: float, frac: float) -> tuple[float, float]:
    """tanh バースト中心 ``T/2`` の周り ``±frac·T`` を臨界窓とする."""
    half = 0.5 * T
    return (half - frac * T, half + frac * T)


def run_scenario(
    method: str,
    n: int,
    *,
    T: float = 4.0,
    beta: float = 16.0,
    tol_step: float = 1e-8,
    dt0: float = 0.1,
    dt_min: float = 1e-4,
    dt_max: float | None = 0.5,
    safety: float = 0.9,
    growth_max: float = 4.0,
    max_rejects: int = 200,
    reject_shrink_min: float = 0.2,
    reject_shrink_max: float = 0.9,
    freeze_growth_after_reject: bool = True,
    growth_freeze_steps: int = 1,
    pi_alpha: float = 0.7,
    pi_beta: float = 0.4,
    window_frac: float = 0.15,
    seed: int = 20260530,
    ref_psi: np.ndarray | None = None,
) -> dict[str, Any]:
    """1 (method, N) シナリオを実 driver で走らせ計測 dict を返す.

    ``ref_psi`` を渡すと終端 infidelity をその基準で測る (compare モードで
    同一基準を共有するため)。``None`` なら内部で tight 基準を計算する。
    """
    prob, h_x = make_problem(n, seed)
    sched = build_steep_schedule(T, beta, h_x)
    psi0 = uniform_superposition(n)

    psi, t_hist, dt_hist, n_rejects = _drive(
        method,
        prob,
        sched,
        psi0,
        T,
        tol_step=tol_step,
        dt0=dt0,
        dt_min=dt_min,
        dt_max=dt_max,
        safety=safety,
        growth_max=growth_max,
        max_rejects=max_rejects,
        reject_shrink_min=reject_shrink_min,
        reject_shrink_max=reject_shrink_max,
        freeze_growth_after_reject=freeze_growth_after_reject,
        growth_freeze_steps=growth_freeze_steps,
        pi_alpha=pi_alpha,
        pi_beta=pi_beta,
    )

    if ref_psi is None:
        ref_psi = _reference_psi(prob, sched, psi0, T)
    infidelity = 1.0 - float(np.abs(np.vdot(ref_psi, psi)) ** 2)

    # 実 driver は per-attempt の accept/reject を返さないので、ControllerTrace
    # は attempts 空で組み、dt-shape メトリクスは時間窓 (バースト周辺) で算出する。
    # 受理率は集計値 (n_accepts / (n_accepts + n_rejects)) から直接出す。
    trace = ControllerTrace(
        t_history=np.asarray(t_hist, dtype=np.float64),
        dt_history=np.asarray(dt_hist, dtype=np.float64),
        n_rejects=int(n_rejects),
        attempts=[],
    )
    window = _critical_time_window(T, window_frac)
    n_accepts = int(dt_hist.shape[0])
    n_attempts = n_accepts + int(n_rejects)
    acc = n_accepts / n_attempts if n_attempts else float("nan")

    dt_win = dt_hist[(t_hist[:-1] >= window[0]) & (t_hist[:-1] < window[1])]
    if dt_win.shape[0] >= 1:
        swing = float(dt_win.max() / dt_win.min())
    else:
        swing = float("nan")

    return {
        "method": method,
        "n": n,
        "beta": beta,
        "tol_step": tol_step,
        "n_accepts": n_accepts,
        "n_rejects": int(n_rejects),
        "acceptance_rate": acc,
        # 1 step = full + half×2 の 6 propagator 評価。
        "propagator_calls": n_attempts * 6,
        "log_dt_lag1_autocorr": log_dt_lag1_autocorr(trace, window),
        "n_reversals": n_reversals(trace, window),
        "std_log_dt_window": std_log_dt(trace, window),
        "dt_swing_window": swing,
        "terminal_infidelity": infidelity,
    }


def compare_configs(
    method: str,
    n: int,
    cfg_old: dict[str, Any],
    cfg_new: dict[str, Any],
    *,
    T: float,
    beta: float,
    tol_step: float,
    seed: int,
) -> dict[str, Any]:
    """同一 schedule / problem で old vs new config を走らせ差分を返す.

    終端 infidelity は同一の tight 基準で測る (両 config 共通 ψ_ref)。
    後続 sub-issue が old/new に旧挙動 knob / 新挙動 knob を渡して使う想定。
    """
    prob, h_x = make_problem(n, seed)
    sched = build_steep_schedule(T, beta, h_x)
    psi0 = uniform_superposition(n)
    ref_psi = _reference_psi(prob, sched, psi0, T)

    row_old = run_scenario(
        method,
        n,
        T=T,
        beta=beta,
        tol_step=tol_step,
        seed=seed,
        ref_psi=ref_psi,
        **cfg_old,
    )
    row_new = run_scenario(
        method,
        n,
        T=T,
        beta=beta,
        tol_step=tol_step,
        seed=seed,
        ref_psi=ref_psi,
        **cfg_new,
    )
    return {
        "method": method,
        "n": n,
        "old": row_old,
        "new": row_new,
        "d_acceptance": row_new["acceptance_rate"] - row_old["acceptance_rate"],
        "d_n_rejects": row_new["n_rejects"] - row_old["n_rejects"],
        "d_terminal_infidelity": row_new["terminal_infidelity"]
        - row_old["terminal_infidelity"],
    }


# ---------------------------------------------------------------------------
# 出力整形
# ---------------------------------------------------------------------------

_COLUMNS = [
    "method",
    "n",
    "beta",
    "n_accepts",
    "n_rejects",
    "acceptance_rate",
    "propagator_calls",
    "log_dt_lag1_autocorr",
    "n_reversals",
    "std_log_dt_window",
    "dt_swing_window",
    "terminal_infidelity",
]


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        if abs(v) < 1e-3 or abs(v) >= 1e4:
            return f"{v:.3e}"
        return f"{v:.4f}"
    return str(v)


def format_markdown(rows: list[dict[str, Any]]) -> str:
    """計測 row を markdown テーブルに整形する."""
    header = "| " + " | ".join(_COLUMNS) + " |"
    sep = "| " + " | ".join("---" for _ in _COLUMNS) + " |"
    lines = [header, sep]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(c, "")) for c in _COLUMNS) + " |")
    return "\n".join(lines)


def _write_results(rows: list[dict[str, Any]], results_dir: Path) -> None:
    """CSV + markdown を ``results_dir`` に書く (gitignored)."""
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "stepsize_controller.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in _COLUMNS})
    md_path = results_dir / "stepsize_controller.md"
    md_path.write_text(format_markdown(rows) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_int_list(text: str) -> list[int]:
    try:
        vals = [int(x) for x in text.split(",") if x.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated ints: {text!r}"
        ) from exc
    if not vals:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return vals


def _parse_methods(text: str) -> list[str]:
    vals = [x.strip() for x in text.split(",") if x.strip()]
    for v in vals:
        if v not in _VALID_METHODS:
            raise argparse.ArgumentTypeError(
                f"method must be one of {_VALID_METHODS}, got {v!r}"
            )
    if not vals:
        raise argparse.ArgumentTypeError("expected at least one method")
    return vals


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else None
    )
    p.add_argument("--n-values", type=_parse_int_list, default=[4, 6])
    p.add_argument(
        "--methods",
        type=_parse_methods,
        default=list(_VALID_METHODS),
        help="richardson,chebyshev",
    )
    p.add_argument("--T", type=float, default=4.0)
    p.add_argument("--beta", type=float, default=16.0, help="tanh バースト急峻さ")
    p.add_argument("--tol-step", type=float, default=1e-8)
    p.add_argument("--window-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=20260530)
    p.add_argument(
        "--compare",
        action="store_true",
        help="既存 knob (growth_max) の old vs new 差分比較モードを実行する",
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="指定すると CSV + md を書き出す (既定は stdout のみ)",
    )
    return p


def run_all_scenarios(
    n_values: list[int],
    methods: list[str],
    *,
    T: float,
    beta: float,
    tol_step: float,
    window_frac: float,
    seed: int,
) -> list[dict[str, Any]]:
    """全 (method, N) シナリオを走らせて計測 row list を返す (smoke test 用 API)."""
    rust_ok = _rust_available()
    rows: list[dict[str, Any]] = []
    for method in methods:
        if method == "chebyshev" and not rust_ok:
            continue
        for n in n_values:
            rows.append(
                run_scenario(
                    method,
                    n,
                    T=T,
                    beta=beta,
                    tol_step=tol_step,
                    window_frac=window_frac,
                    seed=seed,
                )
            )
    return rows


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.compare:
        # #150: reject 後の dt 成長凍結の old (freeze=False = #149 完了時点) vs
        # new (freeze=True = driver 既定) を同一 schedule で比較し, 受理率↑ /
        # n_rejects↓ / 終端精度非劣化を end-to-end で実測する (層 B)。
        # 成長凍結は #149 既定クランプ [0.2, 0.9] 下では発火しない (予測式 reject が
        # 再拡大を自己抑制するため) ので, 効果を可視化するため reject 縮小を過剰に
        # する legacy 0.5/0.5 regime で比較する (= 成長凍結が断つ limit cycle が
        # 顕在化する設定)。#149 単体の比較は git 履歴 PR #154 を参照。
        # (#151 は pi_beta の old/new をここに渡していく。)
        cfg_old = {
            "reject_shrink_min": 0.5,
            "reject_shrink_max": 0.5,
            "freeze_growth_after_reject": False,
        }
        cfg_new = {
            "reject_shrink_min": 0.5,
            "reject_shrink_max": 0.5,
            "freeze_growth_after_reject": True,
        }
        for method in args.methods:
            if method == "chebyshev" and not _rust_available():
                print(f"[skip] {method}: Rust 拡張が未ビルド")
                continue
            for n in args.n_values:
                diff = compare_configs(
                    method,
                    n,
                    cfg_old,
                    cfg_new,
                    T=args.T,
                    beta=args.beta,
                    tol_step=args.tol_step,
                    seed=args.seed,
                )
                print(
                    f"[compare] method={method} n={n} "
                    f"Δacceptance={diff['d_acceptance']:+.4f} "
                    f"Δn_rejects={diff['d_n_rejects']:+d} "
                    f"Δinfidelity={diff['d_terminal_infidelity']:+.3e}"
                )
        return 0

    rows = run_all_scenarios(
        args.n_values,
        args.methods,
        T=args.T,
        beta=args.beta,
        tol_step=args.tol_step,
        window_frac=args.window_frac,
        seed=args.seed,
    )
    print(format_markdown(rows))

    if args.results_dir is not None:
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        out_dir = args.results_dir / stamp
        _write_results(rows, out_dir)
        print(f"\nwrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
