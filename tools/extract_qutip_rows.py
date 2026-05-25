#!/usr/bin/env python3
"""bench_readme_figure.py が吐いた CSV から QuTiP 行を別ディレクトリに切出.

QuTiP の per-cell wall_sec は kinema build version に依存しない
(QuTiP は外部 sparse sesolve, kinema を import せず動作するため). したがって
version 別ディレクトリ (`0.8.0/`, `0.11.0/` 等) に重複保存するのは無駄で,
**1 度測ったら version 横断で reuse** したい. 本 script はこれを実現するため:

1. 各 ``<input_dir>/bench_<scenario>.csv`` から ``solver == "qutip"`` 行を抽出
2. 同名で ``<output_dir>/bench_<scenario>.csv`` に書出 (CSV header 込み)
3. 元 CSV を **non-qutip 行のみ** で atomic に上書き保存

両 CSV の column 順は ``bench_readme_figure.py`` の ``CSV_FIELDNAMES`` と一致.

使用例:

    uv run python tools/extract_qutip_rows.py \\
        --input-dir benchmarks/data/0.8.0/ \\
        --output-dir benchmarks/data/qutip/

冪等: 2 回目以降の実行では元 CSV に qutip 行が無いので no-op (`[skip]` ログ).
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

# bench_readme_figure.py の CSV_FIELDNAMES と同期
CSV_FIELDNAMES = [
    "scenario",
    "n",
    "T",
    "seed",
    "solver",
    "variant",
    "knob_name",
    "knob_value",
    "wall_sec",
    "infidelity",
    "n_steps_eff",
]


def _save_csv_atomic(csv_path: Path, rows: list[dict[str, str]]) -> None:
    """tmp + os.replace で atomic に書き直す. bench_readme_figure と同方式."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_path, csv_path)


def _read_csv(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="bench_*.csv が並ぶ source dir (e.g. benchmarks/data/0.8.0/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="QuTiP 行の書出先 dir (e.g. benchmarks/data/qutip/). "
        "存在しなければ作成. 既存ファイルは上書き.",
    )
    args = parser.parse_args()

    csv_files = sorted(args.input_dir.glob("bench_*.csv"))
    if not csv_files:
        raise SystemExit(f"no bench_*.csv in {args.input_dir}")

    total_qutip = 0
    for src in csv_files:
        all_rows = _read_csv(src)
        qutip_rows = [r for r in all_rows if r.get("solver") == "qutip"]
        other_rows = [r for r in all_rows if r.get("solver") != "qutip"]

        if not qutip_rows:
            print(f"[skip] {src}: solver=qutip 行が無い (既に抽出済 or 未収集)")
            continue

        dst = args.output_dir / src.name
        _save_csv_atomic(dst, qutip_rows)
        print(f"[done] {src} ({len(qutip_rows)} qutip rows) → {dst}")

        _save_csv_atomic(src, other_rows)
        print(f"[done] {src} = {len(other_rows)} non-qutip rows のみで上書き")

        total_qutip += len(qutip_rows)

    print(f"\n[summary] {len(csv_files)} CSV から計 {total_qutip} qutip rows を抽出")


if __name__ == "__main__":
    main()
