"""BLAS on/off ビルドで生成した数値一致 artifact を diff する standalone script.

``tests/test_blas_consistency.py`` が ``blas_on.npz`` / ``blas_off.npz``
を書き出し, 本スクリプトでそれらを比較して **全 array が rel < tol で
一致** することを確認する (default ``tol = 1e-13``).

usage::

    uv run python tools/diff_blas_artifacts.py tests/artifacts/blas_on.npz tests/artifacts/blas_off.npz
    uv run python tools/diff_blas_artifacts.py blas_on.npz blas_off.npz --rtol 1e-12

exit code:

* ``0``: 全 array が tol 以内で一致.
* ``1``: いずれかの array が tol 越え, または key 集合不一致, または
  build profile (``_meta_*``) が一致 (= 同じ build を 2 回比較しているため
  意味がない).

issue #65 (Phase 6 C4) で確立した運用. CI 後段ジョブからも呼び出せるよう
依存は numpy のみに留め, package import は不要 (``tools/`` を sys.path に
通さなくても直接実行できる構成).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# 比較対象から除外する key prefix. ``_meta_*`` は build profile の int フラグ
# なので array 一致を見ても情報量がなく, むしろ「異なる build か」の確認に
# 使う (詳細は ``_check_meta_mismatch``).
_META_PREFIX = "_meta_"

# 数値一致 default tol. ``tests/test_blas_consistency.py`` の sample size
# (n ∈ {4, 6, 8}, dim ≤ 256, Lanczos m=24) に対し Lanczos 内部の Level-1 BLAS
# reduction order 差が累積しても十分余裕で通る範囲 (cargo test 側で確立した
# rel < 1e-13 と整合させる).
_DEFAULT_RTOL = 1e-13
_DEFAULT_ATOL = 1e-13


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diff two .npz artifacts produced by tests/test_blas_consistency.py "
            "(BLAS on vs BLAS off). Asserts all arrays match within rtol/atol."
        )
    )
    parser.add_argument("npz_a", type=Path, help="first artifact (e.g. blas_on.npz)")
    parser.add_argument("npz_b", type=Path, help="second artifact (e.g. blas_off.npz)")
    parser.add_argument(
        "--rtol",
        type=float,
        default=_DEFAULT_RTOL,
        help=f"relative tolerance for np.allclose (default {_DEFAULT_RTOL!r})",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=_DEFAULT_ATOL,
        help=f"absolute tolerance for np.allclose (default {_DEFAULT_ATOL!r})",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="print per-array max abs/rel diff even for matched arrays",
    )
    return parser.parse_args(argv)


def _check_meta_mismatch(
    bundle_a: dict[str, np.ndarray], bundle_b: dict[str, np.ndarray]
) -> str | None:
    """``_meta_has_blas`` が一致していれば「同 build を比較している」エラーを返す.

    BLAS on / off 比較の意味があるのは ``has_blas`` が逆である場合のみ.
    両方とも on (または両方とも off) なら比較対象として不適なので, 明示的に
    エラーメッセージを返して上位で fail させる.
    """
    has_blas_a = int(bundle_a.get(f"{_META_PREFIX}has_blas", np.array([-1]))[0])
    has_blas_b = int(bundle_b.get(f"{_META_PREFIX}has_blas", np.array([-1]))[0])
    if has_blas_a == -1 or has_blas_b == -1:
        return (
            "missing _meta_has_blas; both artifacts must be produced by "
            "tests/test_blas_consistency.py (which embeds the build profile)."
        )
    if has_blas_a == has_blas_b:
        mode = "BLAS on" if has_blas_a == 1 else "BLAS off"
        return (
            f"both artifacts are from {mode} builds; the diff is meaningless. "
            f"rebuild one side with the opposite feature set and regenerate."
        )
    return None


def _format_diff(name: str, a: np.ndarray, b: np.ndarray) -> str:
    """1 つの key に対する max abs / max rel diff を 1 行にまとめる."""
    abs_diff = np.abs(a - b)
    max_abs = float(abs_diff.max()) if abs_diff.size else 0.0
    denom = np.maximum(np.abs(a), np.abs(b))
    # 0/0 を 0 に潰す (両方とも 0 のセルは rel diff が定義されないので NaN
    # ではなく 0 として扱う).
    rel = np.where(denom > 0, abs_diff / np.maximum(denom, np.finfo(np.float64).tiny), 0.0)
    max_rel = float(rel.max()) if rel.size else 0.0
    return f"  {name}: max_abs={max_abs:.3e}, max_rel={max_rel:.3e}, shape={a.shape}"


def diff_npz(
    path_a: Path, path_b: Path, rtol: float, atol: float, verbose: bool
) -> int:
    """2 ``.npz`` を比較し, mismatch があれば exit code 1 を返す.

    Returns
    -------
    int
        0 = 全 array 一致, 1 = mismatch / 不適合.
    """
    if not path_a.exists():
        print(f"error: {path_a} not found", file=sys.stderr)
        return 1
    if not path_b.exists():
        print(f"error: {path_b} not found", file=sys.stderr)
        return 1

    npz_a = np.load(path_a)
    npz_b = np.load(path_b)
    keys_a = set(npz_a.files)
    keys_b = set(npz_b.files)

    if keys_a != keys_b:
        only_a = sorted(keys_a - keys_b)
        only_b = sorted(keys_b - keys_a)
        print(f"error: artifact key sets differ:", file=sys.stderr)
        if only_a:
            print(f"  only in {path_a.name}: {only_a}", file=sys.stderr)
        if only_b:
            print(f"  only in {path_b.name}: {only_b}", file=sys.stderr)
        return 1

    bundle_a = {k: npz_a[k] for k in keys_a}
    bundle_b = {k: npz_b[k] for k in keys_b}

    meta_err = _check_meta_mismatch(bundle_a, bundle_b)
    if meta_err is not None:
        print(f"error: {meta_err}", file=sys.stderr)
        return 1

    print(f"comparing {path_a} vs {path_b}")
    print(f"  rtol={rtol:.3e}, atol={atol:.3e}")
    print(f"  has_blas: {path_a.name}={int(bundle_a['_meta_has_blas'][0])}, "
          f"{path_b.name}={int(bundle_b['_meta_has_blas'][0])}")

    mismatches: list[str] = []
    matched: list[str] = []
    for name in sorted(keys_a):
        if name.startswith(_META_PREFIX):
            continue
        a = bundle_a[name]
        b = bundle_b[name]
        if a.shape != b.shape:
            mismatches.append(
                f"  {name}: shape mismatch {a.shape} vs {b.shape}"
            )
            continue
        if a.dtype != b.dtype:
            mismatches.append(
                f"  {name}: dtype mismatch {a.dtype} vs {b.dtype}"
            )
            continue
        if not np.allclose(a, b, rtol=rtol, atol=atol):
            mismatches.append(_format_diff(name, a, b))
        else:
            if verbose:
                matched.append(_format_diff(name, a, b))

    if verbose and matched:
        print(f"matched ({len(matched)}):")
        for line in matched:
            print(line)

    if mismatches:
        print(f"\nFAIL: {len(mismatches)} array(s) exceed tolerance:", file=sys.stderr)
        for line in mismatches:
            print(line, file=sys.stderr)
        return 1

    n_compared = sum(1 for k in keys_a if not k.startswith(_META_PREFIX))
    print(f"\nOK: {n_compared} array(s) match within tolerance.")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return diff_npz(args.npz_a, args.npz_b, args.rtol, args.atol, args.verbose)


if __name__ == "__main__":
    sys.exit(main())
