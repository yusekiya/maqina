"""``benchmarks/bench_per_step.py`` の出力構造に対する smoke テスト.

bench スクリプトは本来 ``benchmarks/results/`` に CSV + md を吐くだけの
DX ツールだが, adaptive driver の評価で最重要な ``n_steps_actual`` /
``final_err_vs_ref`` を markdown summary に確実に載せるためのリグ
レッションテストを最低限ここに置く. 詳細な数値検証は ``tests/test_
adaptive.py`` (Phase 4 C3) と既存ベンチデータ
(``benchmarks/results/20260513-*/``) が担う.

DoD は issue #43 task D の「smoke 出力で adaptive section が生成される
ことを確認」.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = REPO_ROOT / "benchmarks"


def _import_bench_per_step():
    """``benchmarks/bench_per_step.py`` を path import して module を返す.

    ``benchmarks/__init__.py`` はあるが ``benchmarks`` は site-packages に
    入らないため, repo-root を ``sys.path`` に注入してから import する.
    """
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import benchmarks.bench_per_step as bps

    return bps


def test_adaptive_section_appears_in_md(tmp_path: Path) -> None:
    """adaptive method を含む実行で md に ``## Adaptive driver detail`` が出る."""
    bps = _import_bench_per_step()
    out_dir = tmp_path / "bench_out"
    rc = bps.main(
        [
            "--n-values",
            "4",
            "--methods",
            "cfm4_adaptive_richardson",
            "--n-steps",
            "8",
            "--T",
            "0.5",
            "--repeat",
            "1",
            "--warmup",
            "0",
            "--results-dir",
            str(out_dir),
        ]
    )
    assert rc == 0
    md_path = out_dir / "bench_per_step.md"
    csv_path = out_dir / "bench_per_step.csv"
    assert md_path.exists(), "bench_per_step.md should be created"
    assert csv_path.exists(), "bench_per_step.csv should be created"

    md_text = md_path.read_text(encoding="utf-8")
    # 節タイトルと最重要列ヘッダの存在確認.
    assert "## Adaptive driver detail" in md_text
    assert "n_steps_actual (median)" in md_text
    assert "n_steps_actual (min/max)" in md_text
    assert "final_err_vs_ref (median)" in md_text
    assert "reference_wall (sec)" in md_text
    # reference 計算の総 wall time が machine info に併記されていること.
    assert "reference_wall_sec_total" in md_text

    # CSV にも adaptive 列が出ていること (回帰防止).
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        first_row = next(reader)
    assert "n_steps_actual" in first_row
    assert "final_err_vs_ref" in first_row
    # adaptive trial では final_err_vs_ref が n/a ではない実数文字列になる.
    assert first_row["final_err_vs_ref"] != "n/a"


def test_no_adaptive_section_when_only_fixed_methods(tmp_path: Path) -> None:
    """fixed dt 経路のみの実行では Adaptive section が出ない (条件付き節)."""
    bps = _import_bench_per_step()
    out_dir = tmp_path / "bench_out_fixed"
    rc = bps.main(
        [
            "--n-values",
            "4",
            "--methods",
            "m2",
            "--n-steps",
            "8",
            "--T",
            "0.5",
            "--repeat",
            "1",
            "--warmup",
            "0",
            "--results-dir",
            str(out_dir),
        ]
    )
    assert rc == 0
    md_text = (out_dir / "bench_per_step.md").read_text(encoding="utf-8")
    assert "## Adaptive driver detail" not in md_text
    assert "reference_wall_sec_total" not in md_text


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
