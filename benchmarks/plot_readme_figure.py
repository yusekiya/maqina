"""README 用 fidelity-vs-runtime 散布図プロッタ.

`bench_readme_figure.py` が吐いた CSV を読み, scenario ごとに
``docs/figures/<version>_pareto_<scenario>.png`` を生成する.

軸:

- x = wall time (sec), log scale (左下 = 高速)
- y = infidelity ``1 - F``, log scale (下 = 高精度)
- 左下が優勢領域 (Pareto front は左下に張り付く)

`bench_qutip_large.py` の生 md と違い, 本図は **README に直接埋め込んで
"一目で maqina の優位性が分かる"** ことを目的にしている. なので軸範囲は
両端の極端 cell をクリップせず生データそのまま, marker は method ごとに
形状を変えて視認性を確保する. データ点は markers only で描画
(連結線は cell の sweep 順序を強調するだけで Pareto 形状の視認には
邪魔と判明したため不採用).

凡例:

- Krylov adapt. dt: ``cfm4_adaptive_richardson_krylov`` (Phase 8 m_eff 圧縮版)
- Krylov fixed dt: 固定 dt ``cfm4`` (CFM4:2)
- Chebyshev adapt. dt: ``cfm4_adaptive_richardson_chebyshev`` (Phase B)
- QuTiP: ``sesolve`` Adams (sparse, single-thread)
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Iterable
from pathlib import Path

# matplotlib の Agg backend を明示 (headless サーバー / CI 用)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# CSV header (bench_readme_figure.py と同期)
COLUMNS = [
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

# 表示用ラベル / 色 / marker.
# key = (solver, variant) tuple. variant は maqina の method を識別するタグ.
# 視覚的な識別軸:
#   - 色 = method 系統 (red/green/orange/blue)
#   - marker = method ごとに別形状 (+, x, o, s)
# 連結線は使わない (markers only; docstring 参照).
#
# markersize 注: ``+`` / ``x`` は線が細いので o/s と同サイズだと視覚的に小さく
# 見える. _plot_scenario で markersize の per-style 上書きを実装し,
# +/x は 12, o/s は 9 を使う.
SOLVER_STYLE: dict[tuple[str, str], dict[str, object]] = {
    ("maqina", "krylov_adaptive"): {
        "label": "Krylov adapt. dt",
        "color": "#d62728",  # red
        "marker": "+",
        "alpha": 0.95,
        "markersize": 12,
    },
    ("maqina", "krylov_fixed"): {
        "label": "Krylov fixed dt",
        "color": "#2ca02c",  # green
        "marker": "x",
        "alpha": 0.95,
        "markersize": 12,
    },
    ("maqina", "chebyshev_adaptive"): {
        "label": "Chebyshev adapt. dt",
        "color": "#ff7f0e",  # orange
        "marker": "o",
        "alpha": 0.95,
        "markersize": 9,
    },
    ("qutip", "qutip"): {
        "label": "QuTiP",
        "color": "#1f77b4",  # blue
        "marker": "s",
        "alpha": 0.95,
        "markersize": 9,
    },
}

# Legend に表示する順序. SOLVER_STYLE の挿入順と一致させる
# (Krylov adapt → Krylov fixed → Chebyshev adapt → QuTiP).
_LEGEND_ORDER: list[tuple[str, str]] = list(SOLVER_STYLE.keys())

# 内部 scenario 名 (CLI / npz / CSV で使う legacy ID, 影響範囲が大きいので
# そのまま) と, ユーザー向け表示タイトルの対応. 厳密には Schrödinger 方程式
# は ODE 解析の意味で stiff にならないため (H が Hermitian = 全固有値が実,
# 減衰モードがない), "stiff" ではなく **H_p の dynamic range の広さ** で
# 表現する.
SCENARIO_TITLE = {
    "non-stiff": "narrow dynamic range (SK random)",
    "stiff": "wide dynamic range (SK + 10% basis × penalty=100)",
}


def _read_rows(csv_paths: Iterable[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in csv_paths:
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    return rows


def _filter_finite(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """``infidelity == 0`` の cell は log 軸で描けないため微小値に置換.

    QuTiP tol=ref_tol などで reference と同じ ψ が出る (infidelity = 0)
    ケースを 1e-16 (machine eps の少し下) として描画.
    """
    fixed = []
    for r in rows:
        try:
            inf = float(r["infidelity"])
        except (KeyError, ValueError):
            continue
        if inf <= 0.0 or inf != inf:  # 0 or NaN
            r = dict(r)
            r["infidelity"] = "1e-16"
        fixed.append(r)
    return fixed


def _plot_scenario(
    rows: list[dict[str, str]],
    scenario: str,
    version: str,
    n: int,
    T: float,
    output_dir: Path,
) -> Path:
    """1 scenario 分の散布図を生成し PNG として保存.

    SOLVER_STYLE の (solver, variant) を順に巡って各系列を描画する.
    cells が無い variant は skip. CSV に variant 列が無い (legacy) 行は
    variant 空文字列扱いで SOLVER_STYLE と照合できず skip + warn.
    """
    fig, ax = plt.subplots(figsize=(7.0, 5.5), dpi=120)

    unknown_keys: set[tuple[str, str]] = set()

    for key in _LEGEND_ORDER:
        solver_name, variant = key
        style = SOLVER_STYLE[key]
        cells = [
            r
            for r in rows
            if r["solver"] == solver_name
            and r.get("variant", "") == variant
            and r["scenario"] == scenario
        ]
        if not cells:
            continue
        # knob_value 昇順 (粗精度→高精度) で並べる
        cells.sort(key=lambda r: float(r["knob_value"]), reverse=True)
        walls = [float(r["wall_sec"]) for r in cells]
        infs = [float(r["infidelity"]) for r in cells]
        # 点のみ (連結線なし). markers only で煩雑さを抑え, Pareto 形状を
        # 視覚的に把握しやすくする (連結線は cell の sweep 順序を強調する
        # だけで Pareto 比較には邪魔だった).
        ax.plot(
            walls,
            infs,
            color=style["color"],
            marker=style["marker"],
            linestyle="None",
            alpha=style["alpha"],
            markersize=style.get("markersize", 9),
            markeredgewidth=1.8 if style["marker"] in ("+", "x") else 1.0,
            label=style["label"],
        )

    # 未知の (solver, variant) があれば warning (描画から除外される).
    for r in rows:
        if r["scenario"] != scenario:
            continue
        key = (r["solver"], r.get("variant", ""))
        if key not in SOLVER_STYLE:
            unknown_keys.add(key)
    if unknown_keys:
        print(
            f"[warn] scenario={scenario}: unknown (solver, variant) keys "
            f"{sorted(unknown_keys)} を SOLVER_STYLE から照合できず描画から除外しました.",
            flush=True,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Runtime [sec]")
    ax.set_ylabel("Infidelity")
    # タイトル / scenario 注記 / N=T= 表記は README markdown 側のテーブル header
    # で表すので, 図そのものはシンプルに保つ.
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    # 4 系列分の legend. データが左下に集まるが Pareto frontier 外の領域には
    # 余白があるので legend は左下に置いて maqina の優位帯と被らないようにする.
    # markerscale=0.6 で凡例マーカーを小さくして系列同士の重なりを防ぐ.
    ax.legend(
        loc="lower left",
        framealpha=0.95,
        fontsize=9,
        markerscale=0.6,
        handletextpad=0.5,
    )

    # 右下 footer に version 表記
    fig.text(
        0.99,
        0.01,
        f"maqina v{version}",
        ha="right",
        va="bottom",
        fontsize=7,
        color="gray",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{version}_pareto_{scenario.replace('-', '_')}.png"
    fig.tight_layout(rect=(0.0, 0.02, 1.0, 1.0))
    # bbox_inches=None で figsize=(7.0, 5.5) inch のまま保存 (両 scenario で
    # 同一幅). bbox_inches="tight" は title 文字数等で crop 範囲が変わって
    # 最終 PNG の幅が scenario 間でずれる罠がある.
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        type=Path,
        nargs="+",
        required=True,
        help="bench_readme_figure.py の出力 CSV. 複数渡せば結合して 1 図に",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/figures"),
        help="PNG 出力先 (default docs/figures/)",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="0.11.0",
        help="figure ファイル名と title に使う version 文字列 (default 0.11.0)",
    )
    args = parser.parse_args()

    rows = _read_rows(args.input_csv)
    rows = _filter_finite(rows)
    if not rows:
        raise SystemExit("no rows in input CSV(s)")

    # scenario / n / T は CSV から取り出す (複数 scenario が混在可)
    scenarios = sorted({r["scenario"] for r in rows})
    written: list[Path] = []
    for scenario in scenarios:
        scenario_rows = [r for r in rows if r["scenario"] == scenario]
        n = int(scenario_rows[0]["n"])
        T = float(scenario_rows[0]["T"])
        out = _plot_scenario(
            scenario_rows, scenario, args.version, n, T, args.output_dir
        )
        written.append(out)
        print(f"[done] wrote {out}", flush=True)

    print(f"\n[summary] generated {len(written)} figure(s):")
    for p in written:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
