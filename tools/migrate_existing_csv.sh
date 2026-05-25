#!/bin/bash
# Phase 1 (bench branch / kryanneal API) で収集した CSV を, Phase 2 (main / kinema
# API) の命名規約に変換する 1 回限り migration script.
#
# 変換内容 (CSV の `solver` + `variant` 列 同時書換):
#
#   kryanneal,adaptive_multi → kinema,krylov_adaptive
#   kryanneal,cfm4_multi     → kinema,krylov_fixed
#   qutip,qutip              → (変更なし)
#
# CSV column 順は `scenario,n,T,seed,solver,variant,knob_name,knob_value,...` で
# 隣接 2 列を `,kryanneal,adaptive_multi,` の文字列として置換するので,
# `knob_name=adaptive_multi` のような偶発一致は構造上発生しない (`knob_name`
# は `atol|dt|tol` 限定).
#
# 安全策: 各 CSV を `.csv.bak` として残し, 変換後に diff を表示する.
#
# 使用例 (CSV を含むディレクトリを引数で指定):
#   bash tools/migrate_existing_csv.sh benchmarks/results/0.8.0/

set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <csv_dir>" >&2
    echo "  e.g.: $0 benchmarks/results/0.8.0/" >&2
    exit 1
fi

CSV_DIR="$1"
if [ ! -d "$CSV_DIR" ]; then
    echo "ERROR: $CSV_DIR is not a directory" >&2
    exit 1
fi

shopt -s nullglob
csv_files=("$CSV_DIR"/bench_*.csv)
if [ ${#csv_files[@]} -eq 0 ]; then
    echo "ERROR: no bench_*.csv found in $CSV_DIR" >&2
    exit 1
fi

echo "migrating ${#csv_files[@]} CSV file(s) in $CSV_DIR ..."
for f in "${csv_files[@]}"; do
    echo ""
    echo "--- $f ---"
    cp "$f" "$f.bak"
    sed -i \
        -e 's/,kryanneal,adaptive_multi,/,kinema,krylov_adaptive,/g' \
        -e 's/,kryanneal,cfm4_multi,/,kinema,krylov_fixed,/g' \
        "$f"
    if diff -u "$f.bak" "$f"; then
        echo "[no-op] $f は kryanneal/adaptive_multi または kryanneal/cfm4_multi 行を含まなかった."
        rm "$f.bak"
    else
        echo "[done] $f migrated. backup: $f.bak"
    fi
done

echo ""
echo "migration complete. backup files (.bak) は手動で削除してください."
