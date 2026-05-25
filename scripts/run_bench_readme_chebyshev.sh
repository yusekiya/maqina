#!/bin/bash
# README 用 fidelity-vs-runtime 図の Phase 2: Chebyshev variant cell を 1
# コマンドで全実行する.
#
# 前提:
#   1. 入力 npz (benchmarks/data/problem_*.npz / reference_*.npz) が main に
#      cherry-pick 済 (Phase 1 で生成し bench branch から取り込んだもの).
#   2. kinema build (uv run maturin develop --uv --release) 済.
#
# 構成: kinema cfm4_adaptive_richardson_chebyshev (multi-thread) で 2 scenario
# 順次実行. 各 scenario は --chebyshev-atols [1e-1, 1e-2, 1e-3, 1e-4] sweep.
# scenario 順次 (Step 1-2 と同方針, kinema multi が DRAM 帯域を使い切るため).
#
# 出力 CSV (benchmarks/results/0.11.0/bench_<scenario>.csv) は Chebyshev cell
# のみ. Krylov 0.8.0 は benchmarks/results/0.8.0/, QuTiP は benchmarks/results/qutip/
# に独立して残り, plot 時に 3 dir を結合する.
#
# 実行 (repo root から起動する想定. log と pid も repo root に作られる):
#   nohup bash scripts/run_bench_readme_chebyshev.sh \
#       > run_bench_chebyshev.log 2>&1 < /dev/null &
#   echo $! > run_bench_chebyshev.pid
#   disown
#
# 進捗確認:
#   tail -f run_bench_chebyshev.log
#
# 停止 (子プロセスごと):
#   PID=$(cat run_bench_chebyshev.pid)
#   kill -TERM -- -"$(ps -o pgid= -p "$PID" | tr -d ' ')"

set -euo pipefail
# script は scripts/ 配下にあるので 1 階層上 (= repo root) に移動.
cd "$(dirname "$0")/.."

# 問題設定 (Phase 1 と完全に同一)
N=18
SEED=20260518
T_INT=10000   # ファイル名で使う整数 T (T=10000.0 を seed 名規約で 10000 と表記)

PROBLEM_DIR="benchmarks/data"
OUTPUT_DIR="benchmarks/results/0.11.0"

# Chebyshev sweep 用 atol (シェル変数で 1 ヶ所管理).
# このリストは echo (ログ表示) と --chebyshev-atols (Python 渡し) の両方に
# 使われるので, mismatch が起きないようにする.
CHEBYSHEV_ATOLS="1e-2,1e-3,1e-4,1e-5"

scenarios=(non-stiff stiff)

# 各 scenario について必要な npz が揃っているか確認
for scenario in "${scenarios[@]}"; do
    problem_npz="$PROBLEM_DIR/problem_${scenario}_n${N}_seed${SEED}.npz"
    reference_npz="$PROBLEM_DIR/reference_${scenario}_n${N}_T${T_INT}_seed${SEED}.npz"
    if [ ! -f "$problem_npz" ]; then
        echo "ERROR: $problem_npz が見つかりません. Phase 1 と同じ npz をコピーしてください." >&2
        exit 1
    fi
    if [ ! -f "$reference_npz" ]; then
        echo "ERROR: $reference_npz が見つかりません. Phase 1 と同じ npz をコピーしてください." >&2
        exit 1
    fi
done

# kinema build の確認 (Chebyshev は main 以降の機能なので kryanneal build では
# 動かない). import 試行 + method 名チェック.
if ! uv run python -c "import kinema; print(kinema.__name__)" >/dev/null 2>&1; then
    echo "ERROR: kinema が import できません. 'uv run maturin develop --uv --release' を実行してください." >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "================================================================"
echo "run_bench_readme_chebyshev.sh start: $(date)"
echo "  N=$N, T=$T_INT, seed=$SEED"
echo "  scenarios: ${scenarios[*]}"
echo "  method: cfm4_adaptive_richardson_chebyshev (multi-thread)"
echo "  atol sweep: [${CHEBYSHEV_ATOLS}]"
echo "================================================================"

# ----------------------------------------------------------------------------
# 内部ヘルパ: kinema Chebyshev cell sweep を 1 scenario 起動する.
# ----------------------------------------------------------------------------
run_chebyshev_cell() {
    local scenario=$1

    local problem_npz="$PROBLEM_DIR/problem_${scenario}_n${N}_seed${SEED}.npz"
    local reference_npz="$PROBLEM_DIR/reference_${scenario}_n${N}_T${T_INT}_seed${SEED}.npz"

    echo ""
    echo "================================================================"
    echo "=== kinema chebyshev_adaptive $scenario start: $(date) ==="
    echo "================================================================"

    uv run python -u -m benchmarks.bench_readme_figure \
        --solver kinema \
        --method chebyshev \
        --variant-tag chebyshev_adaptive \
        --chebyshev-atols "$CHEBYSHEV_ATOLS" \
        --problem-file   "$problem_npz" \
        --reference-file "$reference_npz" \
        --output-dir     "$OUTPUT_DIR"

    echo "=== kinema chebyshev_adaptive $scenario done: $(date) ==="
}

# ============================================================================
# Chebyshev cells を 2 scenario 順次実行
# ============================================================================
echo ""
echo "################################################################"
echo "### Chebyshev cells (kinema cfm4_adaptive_richardson_chebyshev) ###"
echo "################################################################"
for scenario in "${scenarios[@]}"; do
    run_chebyshev_cell "$scenario"
done

echo ""
echo "================================================================"
echo "run_bench_readme_chebyshev.sh done: $(date)"
echo "  output CSV: $OUTPUT_DIR/bench_<scenario>.csv (Chebyshev cells のみ)"
echo "  next: 4 系列散布図生成 ="
echo "    uv run python -m benchmarks.plot_readme_figure \\"
echo "      --input-csv benchmarks/results/0.8.0/bench_*.csv \\"
echo "                  benchmarks/results/qutip/bench_*.csv \\"
echo "                  benchmarks/results/0.11.0/bench_*.csv \\"
echo "      --version 0.11.0"
echo "================================================================"
