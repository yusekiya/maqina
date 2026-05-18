"""README figure 用の TFIM 問題ファイルを生成する.

``benchmarks/data/readme_problem_<scenario>_n<N>_seed<seed>.npz`` に対角ベクトル
``H_p_diag`` と横磁場 ``h_x`` を書き出す. 同 seed / scenario / n で再現可能.
**1 度生成すれば bench / 参照解計算で reuse** できる.

## scenarios

- ``non-stiff``: SK 風 ``H_p = -Σ_{i<j} J_ij σ_i σ_j``, ``J ~ N(0, 1/√N)``.
- ``stiff``: non-stiff に加え ``penalty_fraction`` (default 10%) の計算基底に
  ``penalty_scale * Uniform[0, 1)`` (default scale=100) を加算. ペナルティ法
  目的関数 (一部 constraint violation 基底だけ高エネルギー) を模した dynamic
  range 拡大版.

横磁場 ``h_x`` は全サイト 1 (TFIM 標準).

## 出力 npz の内容

- ``H_p_diag`` (1D float64, shape (2^n,))
- ``h_x`` (1D float64, shape (n,))
- ``n`` (int)
- ``seed`` (int)
- ``scenario`` (str)
- ``penalty_fraction`` (float, stiff のときのみ意味を持つ)
- ``penalty_scale`` (float, 同上)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _build_non_stiff_h_p_diag(n: int, seed: int) -> np.ndarray:
    """SK 風 ``H_p_diag = -Σ_{i<j} J_ij σ_i σ_j`` を返す.

    ``J`` は対称 (off-diagonal) で ``J ~ N(0, 1/√N)`` (SK 規約). diagonal は 0.
    ``σ_i = 1 - 2·b_i`` (bit 0 = LSB).
    """
    rng = np.random.default_rng(seed)
    j_mat = rng.normal(size=(n, n)) / np.sqrt(n)
    j_mat = (j_mat + j_mat.T) / 2.0
    np.fill_diagonal(j_mat, 0.0)

    dim = 1 << n
    x = np.arange(dim, dtype=np.int64)
    bits = ((x[:, None] >> np.arange(n)) & 1).astype(np.int64)
    sigma = 1 - 2 * bits  # shape (dim, n)
    h_p_diag = -np.einsum("ij,xi,xj->x", j_mat, sigma, sigma) / 2.0
    return h_p_diag.astype(np.float64)


def _build_stiff_h_p_diag(
    n: int,
    seed: int,
    penalty_fraction: float,
    penalty_scale: float,
) -> np.ndarray:
    """non-stiff の対角に **計算基底単位の random penalty** を加算した stiff 版.

    ``penalty_fraction`` 比率で random に選んだ basis index に
    ``penalty_scale * Uniform[0, 1)`` を加える. ペナルティ法目的関数の
    「一部 constraint violation 基底だけ高エネルギー」状況を再現.
    """
    base = _build_non_stiff_h_p_diag(n, seed)
    dim = base.shape[0]
    rng = np.random.default_rng(seed + 1)  # penalty mask は base と別 seed
    n_penalty = max(1, int(round(dim * penalty_fraction)))
    indices = rng.choice(dim, size=n_penalty, replace=False)
    penalties = penalty_scale * rng.random(size=n_penalty)
    h_p_diag = base.copy()
    h_p_diag[indices] += penalties
    return h_p_diag


def build(
    *,
    scenario: str,
    n: int,
    seed: int,
    penalty_fraction: float,
    penalty_scale: float,
    output: Path,
) -> Path:
    if scenario == "non-stiff":
        h_p_diag = _build_non_stiff_h_p_diag(n, seed)
    elif scenario == "stiff":
        h_p_diag = _build_stiff_h_p_diag(n, seed, penalty_fraction, penalty_scale)
    else:
        raise ValueError(f"unknown scenario: {scenario!r}")

    h_x = np.ones(n, dtype=np.float64)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        H_p_diag=h_p_diag,
        h_x=h_x,
        n=int(n),
        seed=int(seed),
        scenario=str(scenario),
        penalty_fraction=float(penalty_fraction),
        penalty_scale=float(penalty_scale),
    )
    print(
        f"[done] wrote {output}\n"
        f"  scenario={scenario}, n={n}, seed={seed}, dim={1 << n}, "
        f"|H_p_diag| range = [{h_p_diag.min():.3g}, {h_p_diag.max():.3g}]",
        flush=True,
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario", choices=["non-stiff", "stiff"], required=True
    )
    parser.add_argument("--n", type=int, default=18)
    parser.add_argument("--seed", type=int, default=20260518)
    parser.add_argument("--penalty-fraction", type=float, default=0.10)
    parser.add_argument("--penalty-scale", type=float, default=100.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="npz 出力先 (default: benchmarks/data/readme_problem_<...>.npz)",
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = (
            Path("benchmarks/data")
            / f"readme_problem_{args.scenario}_n{args.n}_seed{args.seed}.npz"
        )

    build(
        scenario=args.scenario,
        n=args.n,
        seed=args.seed,
        penalty_fraction=args.penalty_fraction,
        penalty_scale=args.penalty_scale,
        output=args.output,
    )


if __name__ == "__main__":
    main()
