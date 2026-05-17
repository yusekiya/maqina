#!/usr/bin/env python3
"""verify_beta_m_estimator.py の出力 CSV を再解析.

機械精度床 (`actual_err < 1e-13`) で意味のある推定子検証ができない cell を
除外し, 残りで以下を評価:

1. `ratio_raw = saad_est / actual_err` の log10 統計
2. 補正項導入による改善:
   - `est_dt = saad_est × dt` (Hochbruck-Lubich の τ ファクター)
   - `est_dt_over_m = saad_est × dt / m_test` (高次補正)
3. 「Krylov 不足検出」用途 (b_m × |c_m| > tol_lanczos なら escalate) の
   実用性評価.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=str)
    parser.add_argument("--floor", type=float, default=1e-13,
                        help="actual_err の機械精度床. これ未満は除外 (default 1e-13).")
    args = parser.parse_args()

    rows = []
    with Path(args.csv_path).open() as f:
        r = csv.DictReader(f)
        for row in r:
            row["dt"] = float(row["dt"])
            row["m_test"] = int(row["m_test"])
            row["m_eff"] = int(row["m_eff"])
            row["beta_m"] = float(row["beta_m"])
            row["c_m_abs"] = float(row["c_m_abs"])
            row["saad_est"] = float(row["saad_est"])
            row["actual_err"] = float(row["actual_err"])
            row["ratio"] = float(row["ratio"])
            rows.append(row)

    print(f"# β_m estimator: refined analysis")
    print(f"")
    print(f"Total cells: {len(rows)}")

    # Filter: only cells where the estimator is in measurable range
    valid = [r for r in rows if r["actual_err"] >= args.floor and r["saad_est"] >= args.floor]
    print(f"Valid cells (actual_err >= {args.floor:.0e} and saad_est >= {args.floor:.0e}): {len(valid)}")
    print(f"")

    if not valid:
        print("No valid cells. Increase verification range.")
        return 1

    # Raw ratio
    log_ratios_raw = [math.log10(r["ratio"]) for r in valid]
    # With dt correction
    log_ratios_dt = [math.log10(r["saad_est"] * r["dt"] / r["actual_err"]) for r in valid]
    # With dt / m correction
    log_ratios_dt_m = [
        math.log10(r["saad_est"] * r["dt"] / r["m_test"] / r["actual_err"]) for r in valid
    ]

    def stats(label: str, log_r: list[float]) -> None:
        log_r_sorted = sorted(log_r)
        median = log_r_sorted[len(log_r_sorted) // 2]
        within_2 = sum(1 for x in log_r if abs(x) <= 2.0)
        within_1 = sum(1 for x in log_r if abs(x) <= 1.0)
        within_05 = sum(1 for x in log_r if abs(x) <= 0.5)
        mean = sum(log_r) / len(log_r)
        sd = (sum((x - mean) ** 2 for x in log_r) / len(log_r)) ** 0.5
        print(f"### {label}")
        print(f"- median log10(ratio): {median:+.3f}")
        print(f"- mean log10(ratio):   {mean:+.3f}")
        print(f"- std  log10(ratio):   {sd:.3f}")
        print(f"- range:               [{min(log_r):+.3f}, {max(log_r):+.3f}]")
        print(f"- |log10| <= 0.5 (3x): {within_05}/{len(log_r)} ({100*within_05/len(log_r):.0f}%)")
        print(f"- |log10| <= 1.0 (10x): {within_1}/{len(log_r)} ({100*within_1/len(log_r):.0f}%)")
        print(f"- |log10| <= 2.0 (100x): {within_2}/{len(log_r)} ({100*within_2/len(log_r):.0f}%)")
        print()

    print("## raw `saad_est / actual_err`")
    stats("raw", log_ratios_raw)
    print("## with τ correction: `saad_est × dt / actual_err`")
    stats("τ-corrected", log_ratios_dt)
    print("## with τ/m correction: `saad_est × dt / m_test / actual_err`")
    stats("τ/m-corrected", log_ratios_dt_m)

    # 「Krylov 不足検出」用途: saad_est > tol_lanczos なら不十分判定が正しいか?
    # tol_lanczos = atol = 1e-7 (典型値) と仮定して検査
    print("## 用途別: Krylov 不足検出 (escalation rule)\n")
    for tol in [1e-5, 1e-7, 1e-9]:
        # est_dt が "代表的" 補正後推定子
        true_insufficient = sum(1 for r in valid if r["actual_err"] > tol)
        flagged = sum(1 for r in valid if r["saad_est"] * r["dt"] > tol)
        true_positive = sum(
            1 for r in valid
            if r["actual_err"] > tol and r["saad_est"] * r["dt"] > tol
        )
        false_negative = sum(
            1 for r in valid
            if r["actual_err"] > tol and r["saad_est"] * r["dt"] <= tol
        )
        false_positive = sum(
            1 for r in valid
            if r["actual_err"] <= tol and r["saad_est"] * r["dt"] > tol
        )
        print(f"### tol_lanczos = {tol:.0e} (τ-corrected estimator)")
        print(f"- 真の Krylov 不足 cell 数:     {true_insufficient}")
        print(f"- 推定子が不足判定する cell 数: {flagged}")
        print(f"- True Positive:               {true_positive}")
        print(f"- False Negative (見逃し, 危険): {false_negative}")
        print(f"- False Positive (過剰判定):    {false_positive}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
