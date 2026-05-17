"""kryanneal 内部共有ヘルパ (cross-module private API).

``QuantumAnnealer`` (``kryanneal.annealer``) と ``AnnealingSimulator``
(``kryanneal.simulator``) の双方で使う共通定数 / 入力検証 / auto
resolution helper を集約する. 本モジュールは **package-internal** で
公開 API ではない (アンダースコア prefix の通称 / ``__all__`` 未宣言).

設計判断 (issue #48 で切り出し)
-------------------------------
* ``kryanneal.annealer`` は ``__all__ = ["QuantumAnnealer"]`` で public
  surface を絞っているため, そこから private helper を ``simulator.py``
  が module-level で import すると ``annealer.pyi`` (auto-generated)
  には export されておらず ty が解決失敗する.
* 解決策として helper を別モジュール (``_helpers.py``, ``__all__``
  未宣言) に集約し, ``gen_api_stubs.py`` の "``__all__`` 無しなら全
  top-level 名を stub に含める" 規則 (``tools/gen_api_stubs.py``
  ``is_public`` docstring 参照) を活かして両モジュールから安全に共有する.

含まれるもの:

* ``_PSI_NORM_TOL``: ``psi0`` の L2-normalize 許容誤差.
* ``_KRYLOV_TOL_ATOL_RATIO``: adaptive 経路の ``krylov_tol`` を ``atol``
  から導く係数.
* ``_KRYLOV_TOL_FIXED_DEFAULT``: 固定 dt 経路の ``krylov_tol`` static
  fallback.
* ``_AUTO_DT_INIT_C`` / ``_AUTO_DT_INIT_BETA`` / ``_AUTO_DT_INIT_FLOOR``:
  ``dt_init=None`` の T-dep auto resolution パラメータ.
* ``_LANCZOS_DT_NORM_COEFF``: ``dt_max=None`` の Lanczos capacity 上界
  係数.
* ``_gershgorin_norm_upper_bound``: ``‖H(t)‖`` の closed-form 上界.
* ``_resolve_dt_max_auto`` / ``_resolve_dt_init_auto``: auto resolution
  本体.
* ``_validate_psi0``: ``psi0`` 入力検証 (shape / dtype / norm / finite).
"""

from __future__ import annotations

import numpy as np

from kryanneal.problem import IsingProblem


_PSI_NORM_TOL: float = 1e-10

# ``krylov_tol`` を adaptive 経路で ``atol`` に連動させる既定係数 (issue #54).
#
# ``krylov_tol = None`` のとき, adaptive Richardson 経路では
# ``effective_krylov_tol = tol_step · _KRYLOV_TOL_ATOL_RATIO`` を採用する.
# 既定 ``tol_step = 1e-8`` に対し effective ``1e-11`` 相当.
#
# 設計方針 (issue #54): **default は accuracy 優先**, Lanczos β_k 早期
# 打切は **opt-in** で user が明示的に許容したときのみ機能する.
# 具体的には,
#
# - default `atol=1e-8` → effective `1e-11` は実用問題サイズで Lanczos
#   が部分空間を使い切る (`m_eff = 6·m_max`) ことが多く, **早期打切は
#   default では発火しない**. これは旧 `1e-12` 固定 default と挙動上
#   ほぼ同等 (atol=1e-8 比で 3 桁マージンは確保するが β_k がそこまで
#   落ちないため). default の "未指定なら robust に動く" 性質を維持.
# - user が大きめの error を許容して打切速度を取りたい場合は
#   `atol` を緩める (e.g. `atol=1e-5` → effective `1e-8`) か,
#   `krylov_tol` を直接緩める (e.g. `1e-6`) ことで opt-in できる.
#   このとき β_k がより早く tol に到達して `m_eff < 6·m_max` の
#   早期打切が発火し per-step が短縮される.
#
# 係数 `1e-3` の根拠: adaptive Richardson の embedded error 推定を
# Lanczos 内誤差が支配しないよう "atol より 3 桁タイト" にしておく
# 経験則. opt-in 動作の予測可能性 (user が atol を緩めれば段階的に
# 打切が発火) を担保する係数として選定. 1e-4 / 5e-4 / 5e-3 等への
# 調整は user 側 bench で必要時に検討可能 (本定数を変えるだけ).
#
# 固定 dt 経路 (m2 / cfm4) では ``atol`` が無いため None →
# ``1e-12`` フォールバック (旧 default 維持).
_KRYLOV_TOL_ATOL_RATIO: float = 1e-3
# 固定 dt 経路で ``krylov_tol = None`` のとき採用する static fallback.
# adaptive 経路と同じ ``atol · 1e-3`` を使えないため (atol を取らないので)
# 旧 default を維持する.
_KRYLOV_TOL_FIXED_DEFAULT: float = 1e-12

# ``dt_init`` の T-dep auto-resolution パラメータ (issue #43 A で導入,
# issue #54 で None default 化).
#
# 線形 schedule では Magnus 級数の T スケーリング (s-space scaling
# invariance) から最適 dt が ``dt* ~ c · T^{3/4}`` で伸びる. 理論最適は
# ``β=0.75`` だが ``β=0.5`` でも warmup step の大部分は削減できる保守値
# として既定採用. ``c=0.1`` は driver default ``dt0=0.5`` を ``T=1`` で
# 5 倍下回る程度に絞り, schedule が非線形でも安全側に倒す.
_AUTO_DT_INIT_C: float = 0.1
_AUTO_DT_INIT_BETA: float = 0.5
# ``T < 1`` (T 自体が小さい) ケースの床値. ``c · T^β`` が driver の
# ``dt_min`` (default ``1e-4``) を下回る範囲では床値が支配する.
# ``1e-3`` は実用 atol ``1e-8`` での PI controller が 1-2 step で再評価
# できる最低粒度として選んだ (driver dt_min との安全マージン).
_AUTO_DT_INIT_FLOOR: float = 1e-3


# ``dt_max`` の Lanczos capacity 上界係数 (issue #43 B で導入, issue #54
# で None default 化).
#
# Lanczos m 部分空間で `exp(-i dt H) |ψ⟩` を ``rel < tol`` で再現できる
# 安全領域は経験的に ``dt · ‖H‖ ≲ 4 m`` (cv_ising と同方針, hand-rolled
# Lanczos の collapsed safe radius). ``‖H‖`` は Gershgorin 上界
# (closed form, overhead ゼロ) で見積もる:
#
#     ‖H‖_est = Σ_i |h_x_i| + max_k |H_p_diag[k]|
#
# Power method で spectral radius を取れば tighter になるが Lanczos
# 5–10 step ぶんの overhead がかかるため, Phase 4 follow-up では
# closed form を採用する (issue #43 B 案 1).
_LANCZOS_DT_NORM_COEFF: float = 4.0


def _validate_psi0(problem: IsingProblem, psi0: np.ndarray) -> np.ndarray:
    """``psi0`` の shape / dtype / L2-norm を検証して C-contiguous で返す.

    ``dim = 2**problem.n`` と整合し, complex128 / C-contiguous で,
    L2 ノルムが 1 から ``_PSI_NORM_TOL`` 以内であることを要求する.
    非 complex128 は明示的にエラーにする (silent cast はしない).

    ``QuantumAnnealer.run`` と ``AnnealingSimulator.__init__`` の両方で
    共有するため module-level に置いている (issue #48 で切り出し).
    """
    if not isinstance(psi0, np.ndarray):
        raise ValueError(f"psi0 must be a numpy.ndarray, got {type(psi0).__name__}")
    expected_dim = problem.dim
    if psi0.shape != (expected_dim,):
        raise ValueError(
            f"psi0 shape mismatch: expected ({expected_dim},), got {psi0.shape}"
        )
    if psi0.dtype != np.complex128:
        raise ValueError(f"psi0 dtype must be complex128, got {psi0.dtype}")
    if not psi0.flags.c_contiguous:
        psi0 = np.ascontiguousarray(psi0)
    if not np.all(np.isfinite(psi0.view(np.float64))):
        raise ValueError("psi0 contains NaN or inf")
    norm = float(np.linalg.norm(psi0))
    if abs(norm - 1.0) > _PSI_NORM_TOL:
        raise ValueError(
            f"psi0 must be L2-normalized (‖psi0‖ ≈ 1), got ‖psi0‖ = {norm!r}"
        )
    return psi0


def _gershgorin_norm_upper_bound(problem: IsingProblem) -> float:
    """``‖H(t)‖`` の Gershgorin 上界 (closed form, t 非依存).

    ``H(t) = A(s) · H_driver + B(s) · H_problem``, ``|A|, |B| ≤ 1`` の前提下で
    ``‖H(t)‖ ≤ Σ_i |h_x_i| + max_k |H_p_diag[k]|`` が成立する
    (Gershgorin による行和上界 + ``H_problem`` の対角成分絶対最大).
    schedule の coefficient norm `≤ 1` を仮定するため, 一般 schedule で
    係数が大きい場合は別途倍率を掛ける必要があるが, ``Schedule.linear``
    既定では係数は ``[0, 1]`` に収まり安全側.
    """
    return float(np.sum(np.abs(problem.h_x))) + float(np.max(np.abs(problem.H_p_diag)))


def _resolve_dt_max_auto(problem: IsingProblem, m: int, dt0: float) -> float:
    """``dt_max=None`` (issue #54 で旧 ``"auto"`` リテラル相当) の解決値.
    Lanczos capacity と default の min, dt0 で floor.

    式: ``dt_max = max(min(default_dt_max, 4m / ‖H‖_est), dt0)``,
    ``default_dt_max = 10 · dt0`` (driver default と一致).

    最後の ``max(_, dt0)`` は driver 入力検証 ``dt_max >= dt0`` を満たす
    ための floor. Lanczos cap が ``dt0`` を下回る場合 (``dt0`` が Lanczos
    safe radius を超えているケース) は floor が支配するが, step-doubling
    Richardson 推定子が breakdown を検出して PI controller が dt を
    縮めるので driver 動作自体は安全 (issue #43 B の motivation:
    ``Richardson estimator は Lanczos breakdown も embedded error として
    検出されるので m 自動化 / dt_max 自動化が fail-safe で成立する``).
    """
    norm_h = _gershgorin_norm_upper_bound(problem)
    default_dt_max = 10.0 * dt0
    if norm_h <= 0.0:
        # 完全 0 Hamiltonian (h_x=0 かつ H_p_diag=0). Lanczos cap は無限
        # 大なので default をそのまま返す.
        return default_dt_max
    cap = _LANCZOS_DT_NORM_COEFF * float(m) / norm_h
    return max(min(default_dt_max, cap), dt0)


def _resolve_dt_init_auto(t0: float, t1: float) -> float:
    """``dt_init=None`` (issue #54 で旧 ``"auto"`` リテラル相当) の解決値
    ``c · T^β`` を返す (T = ``t1 - t0``).

    ``T < 1`` で値が極端に小さくなることを防ぐ床値 ``_AUTO_DT_INIT_FLOOR``
    と, ``dt_init`` が interval ``T`` を超えないようにする上限を同時に張る
    (driver 側の step ループでも ``min(dt, t_end - t)`` で再クランプされる
    が, 初期値で interval を超えないこと自体は driver 入力検証
    ``dt_max >= dt0`` の自然性を保つ).

    issue #43 A の motivation: 線形 schedule では Magnus 級数の T
    スケーリング (s-space scaling invariance, ``docs/design/05-3-propagator.md`` §5.3)
    から最適 dt が T 依存で伸びるため, ``dt_init`` を T 依存に取れば
    PI controller の warmup step (default ``0.5`` → optimal dt への成長)
    を削減できる. 既定 ``c=0.1``, ``β=0.5`` は理論最適 ``β=0.75`` より
    保守的だが warmup の大部分を削減できる中庸値 (issue 本文参照).
    """
    T = float(t1) - float(t0)
    dt_auto = _AUTO_DT_INIT_C * (T**_AUTO_DT_INIT_BETA)
    return min(max(dt_auto, _AUTO_DT_INIT_FLOOR), T)
