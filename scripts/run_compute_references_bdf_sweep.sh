#!/usr/bin/env bash
# README figure 用の参照解を BDF sweep mode で再計算するバッチスクリプト.
#
# 0.12.0 既存参照解 (commit 2712a86) は Adams 自己 sweep を primary にしていたため
# stiff scenario で Adams adaptive step 制御の頭打ち (1e-9) が precision floor の
# 上限を決めていた. このスクリプトは BDF を primary にして tol sweep
# [1e-11, 1e-12, 1e-13] で再計算し, より tight な参照解 (~1e-12 〜 1e-13) を得る.
# 詳細は benchmarks/results/0.12.0/SUMMARY.md §4.2 を参照.
#
# 環境変数で sweep を上書き可:
#
#   ADAMS_TOLS   : Adams cross-check 用の tol 列 (CSV, default "1e-11")
#                  primary = BDF なので Adams は cross-validation 1 点で十分.
#                  thoroughness が欲しければ "1e-9,1e-10,1e-11" を渡す
#                  (Adams 18-22h 追加).
#   BDF_TOLS     : BDF sweep の tol 列 (CSV, default "1e-11,1e-12,1e-13")
#   CONV_THRESH  : 自己収束 / 解法独立性の判定閾値 (default "1e-13")
#                  1e-13 で BDF pair (1e-12, 1e-13) が落ちるなら "1e-12" 等に
#                  緩めて再判定する.
#   T            : 最終時刻 (default "10000")
#   SCENARIOS    : 計算する scenario 列 (whitespace separated,
#                  default "non-stiff stiff")
#   SEED         : 問題 npz の seed 部分 (default "20260518")
#   N            : 問題 npz の n 部分 (default "18")
#   OUT_DIR      : 参照解 npz の出力先 (default "benchmarks/data")
#   LOG_DIR      : 実行 log 出力先 (default "benchmarks/data")
#
# 例 (Linux bench machine; 別 shell から再実行可能なように nohup + & で起動):
#
#   nohup bash scripts/run_compute_references_bdf_sweep.sh \
#       > run_compute_references_bdf_sweep.log 2>&1 &
#   echo $! > run_compute_references_bdf_sweep.pid

set -euo pipefail

ADAMS_TOLS="${ADAMS_TOLS:-1e-11}"
BDF_TOLS="${BDF_TOLS:-1e-11,1e-12,1e-13}"
CONV_THRESH="${CONV_THRESH:-1e-13}"
T="${T:-10000}"
SCENARIOS="${SCENARIOS:-non-stiff stiff}"
SEED="${SEED:-20260518}"
N="${N:-18}"
OUT_DIR="${OUT_DIR:-benchmarks/data}"
LOG_DIR="${LOG_DIR:-benchmarks/data}"

cd "$(git rev-parse --show-toplevel)"
mkdir -p "$OUT_DIR" "$LOG_DIR"

echo "=== run_compute_references_bdf_sweep.sh ==="
echo "ADAMS_TOLS  = $ADAMS_TOLS"
echo "BDF_TOLS    = $BDF_TOLS"
echo "CONV_THRESH = $CONV_THRESH"
echo "T           = $T"
echo "SCENARIOS   = $SCENARIOS"
echo "SEED        = $SEED"
echo "N           = $N"
echo "OUT_DIR     = $OUT_DIR"
echo "LOG_DIR     = $LOG_DIR"
echo "============================================="

for scenario in $SCENARIOS; do
    problem="${OUT_DIR}/problem_${scenario}_n${N}_seed${SEED}.npz"
    output="${OUT_DIR}/reference_${scenario}_n${N}_T$(printf '%.0f' "$T")_seed${SEED}.npz"
    logfile="${LOG_DIR}/compute_reference_${scenario}_bdf_sweep.log"

    if [[ ! -f "$problem" ]]; then
        echo "[skip] problem file not found: $problem"
        continue
    fi

    echo
    echo ">>> scenario = $scenario"
    echo ">>> problem  = $problem"
    echo ">>> output   = $output"
    echo ">>> log      = $logfile"
    echo ">>> started at $(date -Iseconds)"

    uv run python -m benchmarks.compute_readme_reference \
        --problem-file "$problem" \
        --T "$T" \
        --ref-tols "$ADAMS_TOLS" \
        --bdf-tols "$BDF_TOLS" \
        --convergence-threshold "$CONV_THRESH" \
        --output "$output" 2>&1 | tee "$logfile"

    echo ">>> finished $scenario at $(date -Iseconds)"
done

echo
echo "=== all done at $(date -Iseconds) ==="
