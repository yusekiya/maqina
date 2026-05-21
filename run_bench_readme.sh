#!/bin/bash
# README 用 fidelity-vs-runtime 散布図のための bench を 1 コマンドで全実行する.
#
# 構成 (戦略 B + thread mode 切替): 計 5 系列 × 2 scenario (non-stiff / stiff).
#
#   Step 1-2: kryanneal cells (multi-thread) を順次実行
#             - adaptive_multi  : cfm4_adaptive_richardson, rayon 全 core
#             - cfm4_multi      : 固定 dt CFM4, rayon 全 core
#   Step 3-4: kryanneal cells (single-thread) を順次実行
#             - adaptive_single : cfm4_adaptive_richardson, RAYON_NUM_THREADS=1
#             - cfm4_single     : 固定 dt CFM4, RAYON_NUM_THREADS=1
#   Step 5  : QuTiP cells を 2 scenario 並列実行
#             - qutip           : sesolve Adams (sparse), single-thread が本質
#
# Step 1-4 は順次 (kryanneal multi の DRAM 帯域競合と single の rayon thread 数
# 切替を clean に分けるため). Step 5 のみ並列 (sparse は CPU 1 core 占有なので
# 2 scenario 同時起動でも互いに競合しない).
#
# BLAS thread は全 step で 1 固定 (bench_readme_figure.py の --blas-threads
# default. spin wait 回避 + kryanneal cell 中の rayon×BLAS 競合回避).
#
# 進捗: 各 step が完了するごとに [done] 行が log に出る. atol/tol 1 cell ごとに
# [saved] 行で CSV が atomic save されるので, 途中中断しても完了 cell は失われ
# ない (同じコマンドで再起動すれば skip-existing で続きから再開).
#
# 実行:
#   nohup bash run_bench_readme.sh > run_bench_readme.log 2>&1 < /dev/null &
#   echo $! > run_bench_readme.pid
#   disown
#
# 進捗確認:
#   tail -f run_bench_readme.log
#
# 停止 (子プロセスごと):
#   PID=$(cat run_bench_readme.pid)
#   kill -TERM -- -"$(ps -o pgid= -p "$PID" | tr -d ' ')"

set -euo pipefail
cd "$(dirname "$0")"

# 問題設定
N=18
SEED=20260518
T_INT=10000   # ファイル名で使う整数 T (T=10000.0 を seed 名規約で 10000 と表記)

PROBLEM_DIR="benchmarks/data"
OUTPUT_DIR="benchmarks/data/0.8.0"

scenarios=(non-stiff stiff)
kryanneal_methods=(adaptive cfm4)
thread_modes=(multi single)

# 各 scenario について必要な npz が揃っているか確認
for scenario in "${scenarios[@]}"; do
    problem_npz="$PROBLEM_DIR/problem_${scenario}_n${N}_seed${SEED}.npz"
    reference_npz="$PROBLEM_DIR/reference_${scenario}_n${N}_T${T_INT}_seed${SEED}.npz"
    if [ ! -f "$problem_npz" ]; then
        echo "ERROR: $problem_npz が見つかりません. build_readme_problem.py で生成してください." >&2
        exit 1
    fi
    if [ ! -f "$reference_npz" ]; then
        echo "ERROR: $reference_npz が見つかりません. compute_readme_reference.py で生成してください." >&2
        exit 1
    fi
done

mkdir -p "$OUTPUT_DIR"

echo "================================================================"
echo "run_bench_readme.sh start: $(date)"
echo "  N=$N, T=$T_INT, seed=$SEED"
echo "  scenarios: ${scenarios[*]}"
echo "  kryanneal methods: ${kryanneal_methods[*]}"
echo "  thread modes: ${thread_modes[*]}"
echo "================================================================"

# ============================================================================
# Step 1-4: kryanneal cells を順次 (thread mode × method × scenario)
# multi → single の順で先に Pareto 全体像を見えるようにする.
# ============================================================================
for thread_mode in "${thread_modes[@]}"; do
    for method in "${kryanneal_methods[@]}"; do
        variant="${method}_${thread_mode}"
        for scenario in "${scenarios[@]}"; do
            echo ""
            echo "================================================================"
            echo "=== kryanneal $variant $scenario start: $(date) ==="
            echo "================================================================"

            problem_npz="$PROBLEM_DIR/problem_${scenario}_n${N}_seed${SEED}.npz"
            reference_npz="$PROBLEM_DIR/reference_${scenario}_n${N}_T${T_INT}_seed${SEED}.npz"

            if [ "$thread_mode" = "single" ]; then
                RAYON_NUM_THREADS=1 uv run python -u -m benchmarks.bench_readme_figure \
                    --solver kryanneal \
                    --method "$method" \
                    --variant-tag "$variant" \
                    --problem-file   "$problem_npz" \
                    --reference-file "$reference_npz" \
                    --output-dir     "$OUTPUT_DIR"
            else
                uv run python -u -m benchmarks.bench_readme_figure \
                    --solver kryanneal \
                    --method "$method" \
                    --variant-tag "$variant" \
                    --problem-file   "$problem_npz" \
                    --reference-file "$reference_npz" \
                    --output-dir     "$OUTPUT_DIR"
            fi

            echo "=== kryanneal $variant $scenario done: $(date) ==="
        done
    done
done

# ============================================================================
# Step 5: QuTiP cells を 2 scenario 並列実行
# QuTiP sparse matvec は single-thread が本質で 1 core 占有のみ. 2 scenario 同時
# 起動しても互いに帯域/コア競合は最小 (BLAS=1 fix で spin wait も排除済み).
# ============================================================================
echo ""
echo "================================================================"
echo "=== QuTiP cells (2 scenario parallel) start: $(date) ==="
echo "================================================================"

for scenario in "${scenarios[@]}"; do
    problem_npz="$PROBLEM_DIR/problem_${scenario}_n${N}_seed${SEED}.npz"
    reference_npz="$PROBLEM_DIR/reference_${scenario}_n${N}_T${T_INT}_seed${SEED}.npz"
    uv run python -u -m benchmarks.bench_readme_figure \
        --solver qutip \
        --problem-file   "$problem_npz" \
        --reference-file "$reference_npz" \
        --output-dir     "$OUTPUT_DIR" &
done
wait

echo "=== QuTiP cells done: $(date) ==="

echo ""
echo "================================================================"
echo "run_bench_readme.sh done: $(date)"
echo "  output CSV: $OUTPUT_DIR/bench_<scenario>.csv (5 variants × 2 scenarios)"
echo "  next: plot_readme_figure.py で variant 別 PNG 生成"
echo "================================================================"
