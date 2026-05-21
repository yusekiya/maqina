#!/bin/bash
# README figure 用の参照解計算 (long-running, 数十時間級).
# nohup + caffeinate で SSH 切断 + macOS sleep に耐える形で起動する.
#
# 起動 (repo root から起動する想定. log と pid も repo root に作られる):
#   nohup bash scripts/run_compute_references.sh > compute_references.log 2>&1 < /dev/null &
#   echo $! > compute_references.pid
#   disown
#
# 進捗確認: tail -f compute_references.log
# 停止:     kill $(cat compute_references.pid)

set -e
# script は scripts/ 配下にあるので 1 階層上 (= repo root) に移動.
cd "$(dirname "$0")/.."

echo "=========================================="
echo "run_compute_references.sh start: $(date)"
echo "=========================================="

echo ""
echo "=== [1/2] non-stiff start: $(date) ==="
caffeinate -i uv run python -m benchmarks.compute_readme_reference \
  --problem-file benchmarks/data/problem_non-stiff_n18_seed20260518.npz \
  --T 10000
echo "=== [1/2] non-stiff done: $(date) ==="

echo ""
echo "=== [2/2] stiff start: $(date) ==="
caffeinate -i uv run python -m benchmarks.compute_readme_reference \
  --problem-file benchmarks/data/problem_stiff_n18_seed20260518.npz \
  --T 10000
echo "=== [2/2] stiff done: $(date) ==="

echo ""
echo "=========================================="
echo "run_compute_references.sh done: $(date)"
echo "=========================================="
