"""``benchmarks/bench_stepsize_controller.py`` (層 B) の smoke テスト (issue #152).

bench スクリプトは本来 stdout / ``benchmarks/results/`` に計測表を吐く CLI だが、
ここでは「現行 main が小 N の急峻 schedule で end-to-end のノコギリ波 (受理率
低下 / reject 多発 / dt 振動) を出力できる」ことを最小コストで固定する。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _import_bench():
    """``benchmarks/bench_stepsize_controller.py`` を path import する.

    ``benchmarks`` は site-packages に入らないので repo-root を sys.path に
    注入してから import する (``test_bench_per_step.py`` と同方式)。
    """
    sys.path.insert(0, str(REPO_ROOT))
    import benchmarks.bench_stepsize_controller as bsc

    return bsc


bsc = _import_bench()


def test_steep_schedule_is_legacy_and_burst_centered():
    """tanh バースト schedule が legacy API で ``s(T/2) ≈ 0.5`` を踏む."""
    import numpy as np

    sched = bsc.build_steep_schedule(T=4.0, beta=16.0, h_x=np.ones(4))
    assert not sched.is_xyz_api
    # バースト中心 T/2 で s ≈ 0.5
    assert abs(sched.s_at(2.0) - 0.5) < 1e-12
    # 端では 0 / 1 に漸近
    assert sched.s_at(0.0) < 0.05
    assert sched.s_at(4.0) > 0.95


def test_richardson_outputs_sawtooth_at_small_n():
    """現行 main の Richardson が N=4 急峻 schedule でノコギリ波を出力できる.

    Rust 非依存 (Lanczos は Python fallback あり)。受理率が 1 を下回り、reject
    と dt 振動 (反転) が出ることを確認する。
    """
    row = bsc.run_scenario("richardson", 4)
    assert row["n_rejects"] >= 1
    assert row["acceptance_rate"] < 1.0
    # 臨界窓で dt が複数回反転 (ノコギリ波) し振幅 (swing) が立つ
    assert row["n_reversals"] >= 1
    assert row["dt_swing_window"] > 1.0
    # 終端 ψ は tight 基準に対し高精度 (driver は正しく解いている)
    assert row["terminal_infidelity"] < 1e-3


def test_run_all_scenarios_rows_have_expected_columns():
    """``run_all_scenarios`` の各 row が出力カラムを網羅する."""
    rows = bsc.run_all_scenarios(
        [4], ["richardson"], T=4.0, beta=16.0, tol_step=1e-8, window_frac=0.15, seed=1
    )
    assert len(rows) == 1
    for col in bsc._COLUMNS:
        assert col in rows[0]
    md = bsc.format_markdown(rows)
    assert "| method |" in md
    assert "richardson" in md


def test_compare_configs_returns_deltas():
    """compare モードが old vs new の差分キーを返す (機構が動く)."""
    diff = bsc.compare_configs(
        "richardson",
        4,
        {"growth_max": 4.0},
        {"growth_max": 2.0},
        T=4.0,
        beta=16.0,
        tol_step=1e-8,
        seed=1,
    )
    assert set(diff) >= {
        "d_acceptance",
        "d_n_rejects",
        "d_terminal_infidelity",
        "old",
        "new",
    }
    # 同一 tight 基準を共有するので両 config とも高精度
    assert diff["old"]["terminal_infidelity"] < 1e-3
    assert diff["new"]["terminal_infidelity"] < 1e-3


@pytest.mark.skipif(
    not bsc._rust_available(), reason="Chebyshev 経路は Rust 拡張が必要"
)
def test_chebyshev_outputs_sawtooth_at_small_n():
    """Rust 利用可能なら Chebyshev もノコギリ波を出力できる."""
    row = bsc.run_scenario("chebyshev", 4)
    assert row["n_rejects"] >= 1
    assert row["acceptance_rate"] < 1.0
    assert row["terminal_infidelity"] < 1e-2
