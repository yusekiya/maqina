#!/bin/bash
# README 用 fidelity-vs-runtime 図を maqina 0.14.0 で取り直すバッチスクリプト.
#
# 背景 (CLAUDE.md / CHANGELOG 0.14.0 参照):
#   0.14.0 で adaptive step-size controller を「真の PI 化 (#151) / reject 過剰
#   縮小解消 (#149) / reject 後成長凍結 (#150)」と挙動を変更した (umbrella #148).
#   この controller 変更は adaptive 系 method
#   (cfm4_adaptive_richardson_krylov / _chebyshev) の既定挙動に影響するため,
#   README 図の adaptive 系列を取り直す. 加えて issue #158 で wide dynamic range
#   (stiff) 参照解が Adams → BDF sweep に差し替わったので, stiff の固定 cfm4 も
#   新参照に対して測り直す.
#
# 再計算スコープ (ユーザー確定方針):
#   - non-stiff (narrow): krylov adaptive + chebyshev adaptive + QuTiP を再計算.
#       固定 cfm4 は controller 非依存 + 参照解不変 (commit 2712a86 のまま) なので
#       0.8.0 cell を流用. QuTiP は infidelity 値自体は不変 (参照解不変) だが,
#       過去 run が状態ベクトルを未保存で「将来の参照解変更時に再実行が必要」な
#       負債だったため, 今のうちに --save-states-dir 付きで再計算して ψ を残し,
#       問題 npz を変えない限り version 横断で再利用可能にする (きれいな状態化).
#   - stiff (wide):       krylov adaptive + chebyshev adaptive + 固定 cfm4
#       + QuTiP を再計算. 固定 cfm4 は controller 非依存だが新 BDF 参照に対して
#        測り直し全 maqina 系列を新参照で統一する. QuTiP は参照解が Adams → BDF に
#        差し替わった (#158) 上, 過去 run が ψ 未保存のため新参照で再計算できず
#        実行必須 (floor cell tol=1e-9 は参照差 8.5e-12 に支配される).
#   どちらの scenario も QuTiP の ψ を --save-states-dir で保存する.
#
# 状態ベクトル保存 (point 3 / issue #158 再発防止):
#   --save-states-dir で各 maqina cell の最終状態 ψ(T) を self-describing 圧縮 npz
#   として benchmarks/results/0.14.0/states/ に保存する. 参照解が将来また差し替わって
#   も保存済 ψ から infidelity を再計算できる. 容量見積もり:
#     1 状態 = 2^18 · 16 byte = 4.0 MiB raw / ~3.84 MiB 圧縮.
#     保存対象 (合計 26 cell ≒ 100 MiB, git に通常 commit; ユーザー許可済):
#       - maqina adaptive 18 cell → benchmarks/results/0.14.0/states/
#           non-stiff: krylov adapt 3 + cheb 6 = 9
#           stiff:     krylov adapt 3 + cheb 6 = 9
#         (固定 cfm4 は version 依存で保存価値小のため ψ 非保存; CSV 行のみ)
#       - QuTiP  8 cell → benchmarks/results/qutip/states/ (version 非依存共有)
#           non-stiff 4 + stiff 4
#
# 実行環境:
#   README ベンチ専用サーバー (回帰/perf 用の EPYC 7713P とは別機) で実行する.
#   0.8.0 / 0.12.0 の README 図を取得したのと同一サーバー・同一設定に揃え,
#   wall 軸の before/after 比較 + 流用 cell (narrow 固定 cfm4 / 0.8.0) との
#   x 軸整合を成立させる (CLAUDE.md ベンチ規約「同一マシン上の before/after」).
#   スレッド設定も 0.12.0 に合わせ BLAS / RAYON とも env 未設定 (default) のまま:
#   maqina cell では --blas-threads を渡さない (Lanczos 内部 BLAS の並列化を維持;
#   bench_readme_figure.py --blas-threads docstring 参照). CLAUDE.md の
#   「本番 perf bench は --blas-threads 8」は EPYC sweep 由来の指針で, README 図
#   (別サーバー + 0.12.0 が default で取得済) には適用しない.
#
# 所要時間の目安 (0.8.0 / 0.12.0 実測; 0.14.0 controller で増減しうる):
#   Phase 1 (maqina, 順次): stiff krylov adapt atol=1e-7 単体で ~14h, stiff cfm4
#     dt=0.2 で ~3.4h 等, 全体で数日規模.
#   Phase 2 (QuTiP, 2 scenario 並列): max(non-stiff ~17h, stiff ~48h) ≈ 48h
#     (直列なら ~65h を並列で短縮).
#   中断耐性あり (cell 完了ごとに CSV/state を atomic 保存, 同じ引数で再起動すれば
#   残り cell から再開. Phase 2 並列 QuTiP も skip-existing で再開可).
#
# 起動 (repo root から; 別 shell から監視・停止できるよう nohup + setsid):
#   nohup bash scripts/run_bench_readme_0_14_0.sh \
#       > run_bench_readme_0_14_0.log 2>&1 < /dev/null &
#   echo $! > run_bench_readme_0_14_0.pid
#   disown
#
# 進捗確認:
#   tail -f run_bench_readme_0_14_0.log          # Phase 1 (maqina) + Phase 2 起動状況
#   tail -f run_bench_qutip_non-stiff.log        # Phase 2 QuTiP non-stiff (並列)
#   tail -f run_bench_qutip_stiff.log            # Phase 2 QuTiP stiff (並列)
#
# 停止 (子プロセスごと):
#   PID=$(cat run_bench_readme_0_14_0.pid)
#   kill -TERM -- -"$(ps -o pgid= -p "$PID" | tr -d ' ')"

set -euo pipefail
# script は scripts/ 配下にあるので 1 階層上 (= repo root) に移動.
cd "$(dirname "$0")/.."

# ---- 問題設定 (Phase 1 / 0.12.0 と完全に同一) ----
N=18
SEED=20260518
T_INT=10000   # ファイル名で使う整数 T (T=10000.0 を seed 名規約で 10000 と表記)

PROBLEM_DIR="benchmarks/data"
OUTPUT_DIR="benchmarks/results/0.14.0"
STATES_DIR="$OUTPUT_DIR/states"
# QuTiP は maqina version 非依存なので version dir の外の共有 dir に置く
# (.gitignore の results/qutip/ 例外 + benchmarks/README.md の設計意図に合わせる).
# 状態ベクトルも同 dir 配下 states/ に保存し, 問題 npz を変えない限り version /
# 参照解変更を横断して再利用できるようにする.
QUTIP_OUTPUT_DIR="benchmarks/results/qutip"
QUTIP_STATES_DIR="$QUTIP_OUTPUT_DIR/states"

# ---- sweep 設定 (1 ヶ所管理; 0.8.0 / 0.12.0 と揃えて comparability を確保) ----
# krylov adaptive: 0.8.0 と同じ atol 3 点 (1e-7 で既に reference floor 到達).
KRYLOV_ATOLS="1e-3,1e-5,1e-7"
# chebyshev adaptive: 0.12.0 と同じ atol 6 点.
CHEBYSHEV_ATOLS="1e-2,1e-3,1e-4,1e-5,1e-6,1e-7"
# chebyshev propagator_tol: 0.12.0 / issue #135 default を明示渡し (固定 1e-12).
CHEBYSHEV_PROPAGATOR_TOL="1e-12"
# 固定 cfm4 (stiff のみ再計算): 0.8.0 と同じ dt 4 点.
CFM4_DTS="5.0,2.0,0.5,0.2"
# QuTiP (両 scenario 再計算; 新参照 + 状態保存): results/qutip と同じ tol 4 点.
QUTIP_TOLS="1e-3,1e-5,1e-7,1e-9"

scenarios=(non-stiff stiff)

# ---- 入力 npz の存在確認 ----
for scenario in "${scenarios[@]}"; do
    problem_npz="$PROBLEM_DIR/problem_${scenario}_n${N}_seed${SEED}.npz"
    reference_npz="$PROBLEM_DIR/reference_${scenario}_n${N}_T${T_INT}_seed${SEED}.npz"
    if [ ! -f "$problem_npz" ]; then
        echo "ERROR: $problem_npz が見つかりません." >&2
        exit 1
    fi
    if [ ! -f "$reference_npz" ]; then
        echo "ERROR: $reference_npz が見つかりません." >&2
        exit 1
    fi
done

# ---- maqina build / version 確認 (states metadata に version を埋めるため
#      0.14.0 であることを担保する) ----
if ! uv run python -c "import maqina" >/dev/null 2>&1; then
    echo "ERROR: maqina が import できません. 'uv run maturin develop --uv --release' を実行してください." >&2
    exit 1
fi
MAQINA_VER=$(uv run python -c "from importlib.metadata import version; print(version('maqina'))" 2>/dev/null)
echo "maqina version = $MAQINA_VER"
if [ "$MAQINA_VER" != "0.14.0" ]; then
    echo "WARNING: maqina version が 0.14.0 ではありません ($MAQINA_VER). 続行しますが states metadata の version はこの値になります." >&2
fi

mkdir -p "$OUTPUT_DIR" "$STATES_DIR" "$QUTIP_OUTPUT_DIR" "$QUTIP_STATES_DIR"

echo "================================================================"
echo "run_bench_readme_0_14_0.sh start: $(date)"
echo "  N=$N, T=$T_INT, seed=$SEED, maqina=$MAQINA_VER"
echo "  scenarios:        ${scenarios[*]}"
echo "  krylov atols:     [$KRYLOV_ATOLS]"
echo "  chebyshev atols:  [$CHEBYSHEV_ATOLS]"
echo "  chebyshev p_tol:  $CHEBYSHEV_PROPAGATOR_TOL"
echo "  cfm4 dts (stiff): [$CFM4_DTS]"
echo "  qutip tols (both): [$QUTIP_TOLS]  (両 scenario 再計算 + 状態保存)"
echo "  maqina output:    $OUTPUT_DIR  (states: $STATES_DIR)"
echo "  qutip  output:    $QUTIP_OUTPUT_DIR  (states: $QUTIP_STATES_DIR)"
echo "================================================================"

# ----------------------------------------------------------------------------
# 内部ヘルパ: 1 scenario / 1 method の cell sweep を起動する.
#   $1 = scenario, $2 = method (adaptive|chebyshev|cfm4)
# ----------------------------------------------------------------------------
run_method() {
    local scenario=$1
    local method=$2

    local problem_npz="$PROBLEM_DIR/problem_${scenario}_n${N}_seed${SEED}.npz"
    local reference_npz="$PROBLEM_DIR/reference_${scenario}_n${N}_T${T_INT}_seed${SEED}.npz"

    local -a extra=()
    local variant_tag method_arg
    # save_states: adaptive 系 (krylov / chebyshev) は高コスト + 再解析 / 参照解変更
    # 耐性のため ψ を保存する. 固定 cfm4 は version 依存 (伝播器実装が version で
    # 変わりうる) で更新ごとに再計算が必要なため保存価値が小さく, ψ は保存しない
    # (CSV 行のみ; stiff は新 BDF 参照 infidelity を得るために再実行する).
    local save_states=1
    case "$method" in
        adaptive)
            method_arg="adaptive"
            variant_tag="krylov_adaptive"
            extra=(--maqina-atols "$KRYLOV_ATOLS")
            ;;
        chebyshev)
            method_arg="chebyshev"
            variant_tag="chebyshev_adaptive"
            extra=(--chebyshev-atols "$CHEBYSHEV_ATOLS" --propagator-tol "$CHEBYSHEV_PROPAGATOR_TOL")
            ;;
        cfm4)
            method_arg="cfm4"
            variant_tag="krylov_fixed"
            extra=(--cfm4-dts "$CFM4_DTS")
            save_states=0
            ;;
        *)
            echo "ERROR: unknown method $method" >&2
            return 1
            ;;
    esac

    if [ "$save_states" -eq 1 ]; then
        extra+=(--save-states-dir "$STATES_DIR")
    fi

    echo ""
    echo "================================================================"
    echo "=== maqina $variant_tag $scenario start: $(date) ==="
    echo "================================================================"

    uv run python -u -m benchmarks.bench_readme_figure \
        --solver maqina \
        --method "$method_arg" \
        --variant-tag "$variant_tag" \
        "${extra[@]}" \
        --problem-file   "$problem_npz" \
        --reference-file "$reference_npz" \
        --output-dir     "$OUTPUT_DIR"

    echo "=== maqina $variant_tag $scenario done: $(date) ==="
}

# ----------------------------------------------------------------------------
# 内部ヘルパ: QuTiP cell sweep を 1 scenario 起動する (両 scenario 再計算 + 状態保存).
#   $1 = scenario
# 2 scenario を並列実行する前提なので --blas-threads 1 を渡し, numpy bundled
# OpenBLAS が 2 プロセス分 104-thread ずつ spin-wait して競合するのを防ぐ
# (QuTiP sesolve は sparse matvec で BLAS 非依存なので thread 数は wall に影響しない;
# bench_readme_figure.py --blas-threads docstring の strategy B 参照).
# ----------------------------------------------------------------------------
run_qutip() {
    local scenario=$1
    local problem_npz="$PROBLEM_DIR/problem_${scenario}_n${N}_seed${SEED}.npz"
    local reference_npz="$PROBLEM_DIR/reference_${scenario}_n${N}_T${T_INT}_seed${SEED}.npz"
    local qutip_csv="$QUTIP_OUTPUT_DIR/bench_${scenario}.csv"

    echo ""
    echo "================================================================"
    echo "=== qutip $scenario start: $(date) (再計算 + 状態保存) ==="
    echo "================================================================"

    # 既存 results/qutip/bench_<scenario>.csv には旧 cell (状態未保存 + stiff は
    # 旧 Adams 参照の infidelity) が入っており, そのままだと skip-existing で
    # 再実行されない. 状態保存 + 新参照 infidelity で取り直すため一旦 .bak に退避し
    # fresh に再生成する (QuTiP は決定的なので psi は不変, non-stiff の infidelity も
    # 参照不変で同値. wall_sec のみ再計測, stiff の infidelity は新 BDF 参照で更新).
    if [ -f "$qutip_csv" ]; then
        echo "[backup] 既存 $qutip_csv → $qutip_csv.bak (fresh 再生成のため退避)"
        mv "$qutip_csv" "$qutip_csv.bak"
    fi

    uv run python -u -m benchmarks.bench_readme_figure \
        --solver qutip \
        --qutip-tols "$QUTIP_TOLS" \
        --blas-threads 1 \
        --problem-file   "$problem_npz" \
        --reference-file "$reference_npz" \
        --output-dir     "$QUTIP_OUTPUT_DIR" \
        --save-states-dir "$QUTIP_STATES_DIR"

    echo "=== qutip $scenario done: $(date) ==="
}

# ============================================================================
# Phase 1: maqina cells を scenario 順次実行.
#   maqina の multi-thread (rayon matvec + 並列 BLAS) が DRAM 帯域を使い切るため
#   scenario 間 / method 間は並列化せず, 1 つずつ単独実行して wall_sec を正確に保つ.
#   non-stiff: adaptive, chebyshev
#   stiff:     adaptive, chebyshev, cfm4
# ============================================================================
for scenario in "${scenarios[@]}"; do
    run_method "$scenario" adaptive
    run_method "$scenario" chebyshev
    if [ "$scenario" = "stiff" ]; then
        run_method "$scenario" cfm4
    fi
done

# ============================================================================
# Phase 2: QuTiP cells を 2 scenario 並列実行 (strategy B).
#   QuTiP sesolve (Adams, sparse) は単一コアで走り DRAM 帯域もわずかなので 2 scenario
#   を並列実行して wall を短縮する (max(~17h, ~48h) ≈ 48h vs 直列 ~65h). maqina 計測の
#   後に回すことで maqina の wall_sec を単独実行のまま正確に保つ. 各 process は内部で
#   --blas-threads 1 を設定 (run_qutip 参照). 出力は scenario 別 log に分け interleave 回避.
#   両 scenario とも再計算 + --save-states-dir で ψ を保存し, 問題 npz を変えない限り
#   参照解差し替えを横断して QuTiP cell を再利用可能にする (#158 再発防止).
# ============================================================================
echo ""
echo "################################################################"
echo "### Phase 2: QuTiP cells (2 scenario 並列) ###"
echo "################################################################"
qutip_pids=()
qutip_scenarios_started=()
for scenario in "${scenarios[@]}"; do
    run_qutip "$scenario" > "run_bench_qutip_${scenario}.log" 2>&1 &
    qutip_pids+=($!)
    qutip_scenarios_started+=("$scenario")
    echo "[parallel] qutip $scenario started (pid $!) → run_bench_qutip_${scenario}.log"
done
echo "[parallel] waiting for ${#qutip_pids[@]} QuTiP processes ..."
# 各 PID を個別に wait して非ゼロ終了を取りこぼさない (wait 複数 PID は最後の
# exit status しか返さないため). set -e 下でも `if !` で受けるので abort しない.
qutip_fail=0
for i in "${!qutip_pids[@]}"; do
    pid=${qutip_pids[$i]}
    sc=${qutip_scenarios_started[$i]}
    if wait "$pid"; then
        echo "[parallel] qutip $sc (pid $pid) OK"
    else
        echo "[parallel] WARNING: qutip $sc (pid $pid) 非ゼロ終了. run_bench_qutip_${sc}.log を確認 (skip-existing で再開可)" >&2
        qutip_fail=1
    fi
done
if [ "$qutip_fail" -ne 0 ]; then
    echo "[parallel] 一部 QuTiP が失敗. 同じスクリプトを再実行すれば残り cell から再開する." >&2
fi
echo "[parallel] all QuTiP processes finished."

echo ""
echo "================================================================"
echo "run_bench_readme_0_14_0.sh done: $(date)"
echo "  maqina CSV:   $OUTPUT_DIR/bench_<scenario>.csv (krylov adapt / cheb / stiff cfm4)"
echo "  maqina states:$STATES_DIR/state_*.npz (18 cell; adaptive のみ, cfm4 は非保存)"
echo "  qutip CSV:    $QUTIP_OUTPUT_DIR/bench_<scenario>.csv (新参照で再生成; 旧は .bak)"
echo "  qutip states: $QUTIP_STATES_DIR/state_*.npz (8 cell) — version 非依存共有"
echo ""
echo "  流用するのは narrow 固定 cfm4 (0.8.0, solver=kinema) のみ."
echo ""
echo "  次のステップ (プロット): 流用 cell と結合して 4 系列散布図を生成する."
echo "  narrow 固定 cfm4 は solver タグを maqina に直した reused CSV を作ってから"
echo "  plot に渡す (詳細は benchmarks/README.md の README figure 0.14.0 節を参照)."
echo "================================================================"
