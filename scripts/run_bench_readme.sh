#!/bin/bash
# README 用 fidelity-vs-runtime 散布図のための bench を 1 コマンドで全実行する.
#
# 構成 (戦略 B + thread mode 切替): 計 5 系列 × 2 scenario (non-stiff / stiff).
# **途中で止めても plot が描ける順** で実行する:
#
#   Step 1: adaptive_multi  (kryanneal cfm4_adaptive_richardson, rayon 全 core)
#   Step 2: cfm4_multi      (kryanneal 固定 dt CFM4, rayon 全 core)
#   Step 3: qutip           (QuTiP sesolve Adams, sparse) — 2 scenario 並列実行
#   ─────  ここで止めれば multi-thread × 2 method + QuTiP の Pareto 完成  ─────
#   Step 4: adaptive_single (kryanneal cfm4_adaptive_richardson, RAYON_NUM_THREADS=1)
#   Step 5: cfm4_single     (kryanneal 固定 dt CFM4, RAYON_NUM_THREADS=1)
#
# Step 1-2 は順次 (kryanneal multi が DRAM 帯域を使い切るため scenario 並列は競合).
# Step 3 のみ 2 scenario 並列 (sparse は CPU 1 core 占有のみで競合最小).
# Step 4-5 は順次 (single thread × scenario の組合せが多数並ぶ).
#
# **中間 plot は Step 3 完了後** (推定 ~4.5 日後) に取れる. その時点で README に
# 載せる core message (kryanneal multi vs QuTiP) は確定済. Step 4-5 は single
# thread vs multi thread の並列化効果を可視化する追加情報なので, 必要に応じて
# 途中で kill 可能.
#
# BLAS thread の運用:
#  - Step 1-2, 4-5 (kryanneal cells): BLAS=default (= 物理コア数) を使う.
#    kryanneal adaptive Richardson は Lanczos 内部で Gram-Schmidt + 終端 gemv
#    (BLAS Level-1/2) を多用するため BLAS=1 にすると wall time が ~1.5× 遅く
#    なる (実測 N=18, atol=1e-3 で 27.5 min → 40+ min).
#  - Step 3 (qutip cells): `--blas-threads 1` を明示. QuTiP sparse matvec は
#    BLAS を使わないので thread 数が wall time に影響しないが, default で
#    spawn される 64 thread が spin wait で CPU を空費する + 2 scenario 並列
#    実行時に互いに競合する. これを排除するため 1 thread に固定.
#
# 進捗: 各 step が完了するごとに [done] 行が log に出る. atol/tol 1 cell ごとに
# [saved] 行で CSV が atomic save されるので, 途中中断しても完了 cell は失われ
# ない (同じコマンドで再起動すれば skip-existing で続きから再開).
#
# 実行 (repo root から起動する想定. log と pid も repo root に作られる):
#   nohup bash scripts/run_bench_readme.sh > run_bench_readme.log 2>&1 < /dev/null &
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
# script は scripts/ 配下にあるので 1 階層上 (= repo root) に移動.
# これで PROBLEM_DIR / OUTPUT_DIR 等の相対 path が repo root 基準で解決される.
cd "$(dirname "$0")/.."

# 問題設定
N=18
SEED=20260518
T_INT=10000   # ファイル名で使う整数 T (T=10000.0 を seed 名規約で 10000 と表記)

PROBLEM_DIR="benchmarks/data"
OUTPUT_DIR="benchmarks/data/0.8.0"

scenarios=(non-stiff stiff)

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
echo "  order: adaptive_multi → cfm4_multi → qutip (並列) → adaptive_single → cfm4_single"
echo "================================================================"

# ----------------------------------------------------------------------------
# 内部ヘルパ: kryanneal の 1 (method, scenario) cell sweep を起動する.
# thread_mode = "multi" or "single" で rayon thread を制御.
# ----------------------------------------------------------------------------
run_kryanneal_cell() {
    local method=$1
    local thread_mode=$2
    local scenario=$3
    local variant="${method}_${thread_mode}"

    local problem_npz="$PROBLEM_DIR/problem_${scenario}_n${N}_seed${SEED}.npz"
    local reference_npz="$PROBLEM_DIR/reference_${scenario}_n${N}_T${T_INT}_seed${SEED}.npz"

    echo ""
    echo "================================================================"
    echo "=== kryanneal $variant $scenario start: $(date) ==="
    echo "================================================================"

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
}

# ============================================================================
# Step 1: adaptive_multi (2 scenarios 順次)
# ============================================================================
echo ""
echo "################################################################"
echo "### Step 1: adaptive_multi (kryanneal cfm4_adaptive_richardson) ###"
echo "################################################################"
for scenario in "${scenarios[@]}"; do
    run_kryanneal_cell adaptive multi "$scenario"
done

# ============================================================================
# Step 2: cfm4_multi (2 scenarios 順次)
# ============================================================================
echo ""
echo "################################################################"
echo "### Step 2: cfm4_multi (kryanneal 固定 dt CFM4) ###"
echo "################################################################"
for scenario in "${scenarios[@]}"; do
    run_kryanneal_cell cfm4 multi "$scenario"
done

# ============================================================================
# Step 3: QuTiP cells を 2 scenario 並列実行
# QuTiP sparse matvec は single-thread が本質で 1 core 占有のみ. 2 scenario 同時
# 起動しても互いに帯域/コア競合は最小 (BLAS=1 fix で spin wait も排除済み).
# ============================================================================
echo ""
echo "################################################################"
echo "### Step 3: qutip (2 scenario parallel) start: $(date) ###"
echo "################################################################"

for scenario in "${scenarios[@]}"; do
    problem_npz="$PROBLEM_DIR/problem_${scenario}_n${N}_seed${SEED}.npz"
    reference_npz="$PROBLEM_DIR/reference_${scenario}_n${N}_T${T_INT}_seed${SEED}.npz"
    # qutip cells のみ --blas-threads 1 を明示 (spin wait + 2 process 並列時の
    # 競合を排除). QuTiP sparse matvec は BLAS を使わないので影響なし.
    uv run python -u -m benchmarks.bench_readme_figure \
        --solver qutip \
        --blas-threads 1 \
        --problem-file   "$problem_npz" \
        --reference-file "$reference_npz" \
        --output-dir     "$OUTPUT_DIR" &
done
wait

echo "### Step 3: qutip done: $(date) ###"

# ----------------------------------------------------------------------------
# ここで止めても README 図の core message (multi-thread Pareto) は完成済.
# Step 4-5 は single thread vs multi thread の並列化効果可視化用追加情報.
# ----------------------------------------------------------------------------

# ============================================================================
# Step 4: adaptive_single (2 scenarios 順次)
# ============================================================================
echo ""
echo "################################################################"
echo "### Step 4: adaptive_single (RAYON_NUM_THREADS=1) ###"
echo "################################################################"
for scenario in "${scenarios[@]}"; do
    run_kryanneal_cell adaptive single "$scenario"
done

# ============================================================================
# Step 5: cfm4_single (2 scenarios 順次)
# ============================================================================
echo ""
echo "################################################################"
echo "### Step 5: cfm4_single (RAYON_NUM_THREADS=1) ###"
echo "################################################################"
for scenario in "${scenarios[@]}"; do
    run_kryanneal_cell cfm4 single "$scenario"
done

echo ""
echo "================================================================"
echo "run_bench_readme.sh done: $(date)"
echo "  output CSV: $OUTPUT_DIR/bench_<scenario>.csv (5 variants × 2 scenarios)"
echo "  next: plot_readme_figure.py で variant 別 PNG 生成"
echo "================================================================"
