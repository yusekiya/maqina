"""README 用 fidelity-vs-runtime 散布図プロッタ.

`bench_readme_figure.py` が吐いた CSV を読み, scenario ごとに
``docs/figures/<version>_pareto_<scenario>.png`` を生成する.

軸:

- x = wall time (sec), log scale (左下 = 高速)
- y = infidelity ``1 - F``, log scale (下 = 高精度)
- 左下が優勢領域 (Pareto front は左下に張り付く)

`bench_qutip_large.py` の生 md と違い, 本図は **README に直接埋め込んで
"一目で kryanneal の優位性が分かる"** ことを目的にしている. なので軸範囲は
両端の極端 cell をクリップせず生データそのまま, marker / color は
solver 2 種で対比, 精度 sweep は ``alpha=0.5`` の線で連結して trade-off
曲線を見せる.
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
    "knob_name",
    "knob_value",
    "wall_sec",
    "infidelity",
    "n_steps_eff",
]

# 表示用ラベル / 色
SOLVER_STYLE = {
    "kryanneal": {
        "label": "kryanneal cfm4_adaptive_richardson",
        "color": "#d62728",  # red
        "marker": "o",
    },
    "qutip": {
        "label": "QuTiP sesolve (sparse)",
        "color": "#1f77b4",  # blue
        "marker": "s",
    },
}

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
    """1 scenario 分の散布図を生成し PNG として保存."""
    fig, ax = plt.subplots(figsize=(8.0, 5.5), dpi=120)

    for solver, style in SOLVER_STYLE.items():
        cells = [r for r in rows if r["solver"] == solver and r["scenario"] == scenario]
        if not cells:
            continue
        # knob_value 昇順 (粗精度→高精度) で並べる
        cells.sort(key=lambda r: float(r["knob_value"]), reverse=True)
        walls = [float(r["wall_sec"]) for r in cells]
        infs = [float(r["infidelity"]) for r in cells]
        ax.plot(
            walls,
            infs,
            color=style["color"],
            marker=style["marker"],
            linestyle="-",
            linewidth=1.4,
            alpha=0.9,
            markersize=9,
            label=style["label"],
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Runtime [sec]  (log, lower is better)")
    ax.set_ylabel("Infidelity  $1 - F$  (log, lower is better)")
    ax.set_title(
        f"Fidelity vs runtime — {SCENARIO_TITLE.get(scenario, scenario)} "
        f"(N={n}, T={T:.0f})"
    )
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    # 本番では kryanneal が左下に Pareto を握る形を想定して legend は右上に固定.
    # data と重ならない方向で安定する.
    ax.legend(loc="upper right", framealpha=0.95)

    # 右下 footer に thread 情報を開示 (透明性). kryanneal は rayon で全コア
    # 並列, QuTiP の sparse 経路は scipy.sparse / qutip-data の制約上ほぼ
    # シングルスレッド. 数値の絶対値はこの非対称性込みで読む.
    fig.text(
        0.99,
        0.01,
        f"kryanneal v{version}  |  kryanneal: rayon multi-thread, "
        f"QuTiP: sparse (effectively single-thread)",
        ha="right",
        va="bottom",
        fontsize=7,
        color="gray",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{version}_pareto_{scenario.replace('-', '_')}.png"
    fig.tight_layout(rect=(0.0, 0.02, 1.0, 1.0))
    fig.savefig(out_path, bbox_inches="tight")
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
        default="0.8.0",
        help="figure ファイル名と title に使う version 文字列 (default 0.8.0)",
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
