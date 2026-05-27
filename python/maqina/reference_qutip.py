"""QuTiP ``sesolve`` を参照実装としたバリデーション用エントリポイント.

``maqina`` の Krylov / CFM4:2 / M2 / Chebyshev propagator の数値結果を QuTiP の
高精度 ODE ソルバ (``sesolve``) と比較するための薄いラッパ. テスト
(``tests/test_reference_qutip.py``, ``tests/test_xyz_schedule.py`` 等) と
ad-hoc な手動検証で利用する.

QuTiP 自体は dev 依存に入っているが, 本番 (wheel) では import を必須に
しない. import 失敗時はテスト側を skip する (``pytest.importorskip``).

提供する builder
----------------

* :func:`build_qutip_hamiltonian_xyz`: per-site/per-axis 時間依存場
  (issue #142 Phase C) の Hamiltonian を QuTiP の time-dependent format
  ``[[H, coeff], ...]`` で組む.

ビット規約変換ノート (sparse 経路):

* maqina: bit 0 = LSB, ``x = Σ_i b_i · 2^i``. X_i / Y_i / Z_i は bit i を
  flip / 位相回転.
* QuTiP: ``qutip.tensor([A_0, A_1, ..., A_{n-1}])`` は ``np.kron`` 順に積み,
  最初の引数 ``A_0`` が **MSB 側** (最も "遅い" qubit) に対応する.
* したがって maqina の bit i に X / Y / Z を作用させるには
  ``qutip.tensor([I, ..., σ, ..., I])`` の **位置 ``n-1-i``** に σ を入れる.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

__all__ = ["build_qutip_hamiltonian_xyz"]


def _pauli_at_site(qutip_mod: Any, axis: str, site: int, n: int) -> Any:
    """maqina の bit ``site`` に Pauli ``axis`` (``'x'``/``'y'``/``'z'``) を
    作用させた QuTiP tensor product を返す.

    bit 規約 (LSB / MSB-first) の変換は qutip 側 ``n-1-site`` で吸収する.
    """
    si = qutip_mod.qeye(2)
    if axis == "x":
        sigma = qutip_mod.sigmax()
    elif axis == "y":
        sigma = qutip_mod.sigmay()
    elif axis == "z":
        sigma = qutip_mod.sigmaz()
    else:
        raise ValueError(f"axis must be 'x'/'y'/'z', got {axis!r}")
    ops = [si] * n
    ops[n - 1 - site] = sigma
    return qutip_mod.tensor(ops)


def build_qutip_hamiltonian_xyz(
    g_x_cbs: Sequence[Callable[[float], float]],
    h_p_diag: np.ndarray,
    b_cb: Callable[[float], float],
    *,
    g_y_cbs: Sequence[Callable[[float], float]] | None = None,
    g_z_cbs: Sequence[Callable[[float], float]] | None = None,
    qutip_module: Any = None,
) -> list[list[Any]]:
    """per-site/per-axis 時間依存場の QuTiP ``H(t)`` リストを組む (issue #142).

    Hamiltonian の形:

    .. code-block:: text

        H(t) = Σ_i [g_x_i(t)·X_i + g_y_i(t)·Y_i + g_z_i(t)·Z_i] + b(t)·H_p_diag

    QuTiP の ``sesolve`` が要求する time-dependent format ``[[H_k, coeff_k], ...]``
    形式で返す. ``coeff_k`` は Python callable (string coefficient ではなく
    callable; pytest 環境で eval 機構の警告を避ける).

    Parameters
    ----------
    g_x_cbs : Sequence[Callable[[float], float]]
        per-site の X 軸時間依存係数 (length n).
    h_p_diag : np.ndarray
        shape ``(2**n,)`` float64 の Z 基底 problem 対角.
    b_cb : Callable[[float], float]
        problem Hamiltonian の global envelope ``b(t)``.
    g_y_cbs, g_z_cbs : Sequence[Callable] | None
        per-site の Y / Z 軸時間依存係数. ``None`` で当該軸 skip.
    qutip_module : module
        ``import qutip`` の結果. 呼び出し側で
        ``qutip = pytest.importorskip("qutip")`` した上で渡す. dev 依存に
        固定しないため module を明示的に注入する.

    Returns
    -------
    list[list]
        QuTiP ``sesolve`` 互換の ``H(t)`` 表現
        ``[[H_drv_x_i, g_x_i(t)], ..., [H_p, b(t)]]``. n が大きい場合
        各 ``H_drv_*`` は CSR sparse 構築 (``qutip.tensor``).
    """
    if qutip_module is None:
        raise ValueError(
            "qutip_module must be provided (e.g., via pytest.importorskip)"
        )
    g_x_list = list(g_x_cbs)
    n = len(g_x_list)
    dim = 1 << n
    if h_p_diag.shape != (dim,):
        raise ValueError(
            f"h_p_diag shape mismatch: expected ({dim},), got {h_p_diag.shape}"
        )

    h_terms: list[list[Any]] = []

    # X 軸 (per-site).
    for i in range(n):
        h_term = _pauli_at_site(qutip_module, "x", i, n)
        cb = g_x_list[i]
        # qutip の time-dependent callable 規約: ``cb(t, args)``. closure で
        # variable capture を固定 (Python の lambda late-binding 対策).
        h_terms.append([h_term, _wrap_cb_for_qutip(cb)])

    # Y 軸 (per-site, optional).
    if g_y_cbs is not None:
        g_y_list = list(g_y_cbs)
        if len(g_y_list) != n:
            raise ValueError(
                f"g_y_cbs length mismatch: expected {n}, got {len(g_y_list)}"
            )
        for i in range(n):
            h_term = _pauli_at_site(qutip_module, "y", i, n)
            h_terms.append([h_term, _wrap_cb_for_qutip(g_y_list[i])])

    # Z 軸 (per-site, optional).
    if g_z_cbs is not None:
        g_z_list = list(g_z_cbs)
        if len(g_z_list) != n:
            raise ValueError(
                f"g_z_cbs length mismatch: expected {n}, got {len(g_z_list)}"
            )
        for i in range(n):
            h_term = _pauli_at_site(qutip_module, "z", i, n)
            h_terms.append([h_term, _wrap_cb_for_qutip(g_z_list[i])])

    # problem Hamiltonian (Z 基底対角).
    h_p_q = qutip_module.qdiags(h_p_diag, 0, dims=[[2] * n, [2] * n])
    h_terms.append([h_p_q, _wrap_cb_for_qutip(b_cb)])

    return h_terms


def _wrap_cb_for_qutip(cb: Callable[[float], float]) -> Callable[..., float]:
    """``cb(t)`` を qutip ``coeff(t, args)`` 形式 (2 引数) に薄く wrap する."""

    def coeff(t: float, args: Any = None) -> float:  # noqa: ARG001
        return float(cb(t))

    return coeff
