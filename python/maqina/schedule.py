"""アニーリングスケジュール (``Schedule``).

Hamiltonian の **時間依存係数を一手に管理**する:

* 旧 API (`Schedule(T, A, B, h_x, s=None)`): 一様横磁場の TFIM
  ``H(t) = A(s(t)) · H_driver + B(s(t)) · H_problem``,
  ``H_driver = -Σ_i h_x_i · X_i`` (X-only, 静的振幅 ``h_x``).
* 新 API (`Schedule.from_xyz(...)`): per-site/per-axis 時間依存場
  ``H(t) = Σ_i [g_x_i(t)·X_i + g_y_i(t)·Y_i + g_z_i(t)·Z_i] + b(t)·H_p_diag``.

issue #142 Phase C で **h_x 振幅を IsingProblem から Schedule に移管**し
"問題側の静的構造 (H_p_diag) vs. 時間発展係数 (h_x / a_t / b_t / g_x_i 等)"
の責任分担を明確化した. ``IsingProblem`` は ``H_p_diag`` のみ保持.

公開コンストラクタ:

* ``Schedule.linear(T, h_x)``: 線形 ``s(t) = t/T``, ``A(s) = 1-s``, ``B(s) = s``.
* ``Schedule.from_callable(T, A, B, h_x, s=None)``: 任意 callable.
* ``Schedule.reverse(T, h_x, s_init=1.0, s_target=0.5, pause_duration=0.0)``:
  reverse annealing (``pause_duration > 0`` で s_target hold を含む変種).
* ``Schedule.pause(T, h_x, t_pause, duration)``: pause schedule.
* ``Schedule.from_xyz(T, g_x, b, *, g_y=None, g_z=None)``: per-axis 時間依存場
  (新 API). callable に h_x の振幅は既に組み込み済の前提 (Schedule 内では
  時間関数のみ管理).

内部設計:

* 単一 evaluator ``_eval_stage(t) -> (gx_arr, gy_arr_opt, gz_arr_opt, b_scalar)``.
  旧 API では ``gx_arr = -a_t(t) · h_x`` を per-stage に組み立て (alloc は
  ``(n,)`` 小サイズなので negligible; ``g_y / g_z = None``). 新 API では
  callable list を直接評価.
* ``_h_x_abs_sum`` (旧 API only): ``Σ_i |h_x_i|`` を ``__init__`` 時に precompute.
  Chebyshev propagator の per-stage Gershgorin O(1) cached form
  (``gershgorin_per_stage_x_only``, PR #144) に渡す.
* ``_norm_upper_bound_factor_at(t)``: per-axis 上界 ``Σ_i √(g_x_i(t)² + g_y_i(t)²)``
  を返す (Gershgorin 行和の off-diagonal 寄与). 旧 API では
  ``|a_t(t)| · _h_x_abs_sum``.

実装注: 全 phase で callable のまま per-step 評価する. 評価コストは Lanczos
1 step の <1% のため grid cache 化の ROI が薄い (issue #142 risk 分析).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

import numpy as np

__all__ = ["Schedule"]


def _validate_h_x(h_x: np.ndarray, *, name: str = "h_x") -> np.ndarray:
    """``h_x`` を ``(n,)`` float64 C-contiguous で検証して返す."""
    if not isinstance(h_x, np.ndarray):
        raise ValueError(f"{name} must be a numpy.ndarray, got {type(h_x).__name__}")
    if h_x.ndim != 1:
        raise ValueError(f"{name} must be 1-dimensional, got shape {h_x.shape}")
    if h_x.shape[0] < 1:
        raise ValueError(f"{name} must have at least 1 site, got shape {h_x.shape}")
    if h_x.dtype != np.float64:
        raise ValueError(f"{name} dtype must be float64, got {h_x.dtype}")
    if not np.all(np.isfinite(h_x)):
        raise ValueError(f"{name} contains NaN or inf")
    return np.ascontiguousarray(h_x, dtype=np.float64)


def _validate_callable_list(
    cbs: Sequence[Callable[[float], float]] | None,
    expected_n: int,
    name: str,
) -> list[Callable[[float], float]] | None:
    """callable list を検証. ``None`` はそのまま返す (axis を skip する意味)."""
    if cbs is None:
        return None
    cbs_list = list(cbs)
    if len(cbs_list) != expected_n:
        raise ValueError(
            f"{name} length mismatch: expected {expected_n}, got {len(cbs_list)}"
        )
    for i, cb in enumerate(cbs_list):
        if not callable(cb):
            raise ValueError(f"{name}[{i}] must be callable, got {type(cb).__name__}")
    return cbs_list


class Schedule:
    """Annealing schedule. 時間依存係数 (h_x 振幅含む) を保持する.

    旧 API (X-only TFIM 経路) と新 API (per-axis 時間依存場経路) を **単一
    クラスで** サポートする. どちらの構築経路を取ったかは ``_is_xyz_api``
    フラグで判別され, driver は ``_eval_stage(t)`` を共通入口として呼ぶ.

    Parameters (旧 API: ``__init__`` / ``from_callable`` / ``linear`` /
    ``reverse`` / ``pause``)
    --------------------------------------------------------------------
    T : float
        総アニーリング時間 (``T > 0``).
    A : Callable[[float], float]
        Driver 係数 ``A(s)``.
    B : Callable[[float], float]
        Problem 係数 ``B(s)``.
    h_x : np.ndarray
        shape ``(n,)`` float64. サイト依存横磁場の **静的振幅**
        (``H_driver = -Σ_i h_x_i · X_i``). issue #142 で IsingProblem
        から移管された.
    s : Callable[[float], float] | None
        ``s(t)``. ``None`` で線形 ``s(t) = t/T``.

    Parameters (新 API: ``from_xyz``)
    ----------------------------------
    T : float
        総時間.
    g_x : Sequence[Callable[[float], float]]
        per-site の X 軸時間依存係数 (length n). callable に h_x 振幅は既に
        組み込み済の前提.
    b : Callable[[float], float]
        problem Hamiltonian の global envelope.
    g_y, g_z : Sequence[Callable] | None
        per-site の Y / Z 軸時間依存係数 (length n). ``None`` で当該軸 skip
        (Rust 側で real-only SIMD kernel に dispatch).

    Raises
    ------
    ValueError
        ``T <= 0``, ``h_x`` の shape / dtype 不整合, callable list の長さ不一致
        等.
    """

    def __init__(
        self,
        T: float,
        A: Callable[[float], float],
        B: Callable[[float], float],
        h_x: np.ndarray,
        s: Callable[[float], float] | None = None,
    ) -> None:
        if not (T > 0):
            raise ValueError(f"T must be positive, got {T!r}")
        h_x_validated = _validate_h_x(h_x)

        self.T: float = float(T)
        self.n: int = int(h_x_validated.shape[0])
        self._kind: Literal["legacy", "xyz"] = "legacy"

        # legacy API state
        self._A: Callable[[float], float] | None = A
        self._B: Callable[[float], float] | None = B
        if s is None:
            T_val = self.T
            self._s: Callable[[float], float] | None = (
                lambda t, _T=T_val: t / _T  # type: ignore[misc]
            )
        else:
            self._s = s
        self._h_x: np.ndarray | None = h_x_validated
        self._h_x_abs_sum: float = float(np.abs(h_x_validated).sum())

        # xyz API state (unused for legacy)
        self._g_x_cbs: list[Callable[[float], float]] | None = None
        self._g_y_cbs: list[Callable[[float], float]] | None = None
        self._g_z_cbs: list[Callable[[float], float]] | None = None
        self._b_cb: Callable[[float], float] | None = None

    @classmethod
    def linear(cls, T: float, h_x: np.ndarray) -> "Schedule":
        """線形スケジュール ``A(s) = 1 - s, B(s) = s, s(t) = t / T``."""
        return cls(T=T, A=lambda s: 1.0 - s, B=lambda s: s, h_x=h_x)

    @classmethod
    def from_callable(
        cls,
        T: float,
        A: Callable[[float], float],
        B: Callable[[float], float],
        h_x: np.ndarray,
        s: Callable[[float], float] | None = None,
    ) -> "Schedule":
        """任意の callable から ``Schedule`` を構築する低レベル API (旧 API).

        ``__init__`` と等価.
        """
        return cls(T=T, A=A, B=B, h_x=h_x, s=s)

    @classmethod
    def reverse(
        cls,
        T: float,
        h_x: np.ndarray,
        s_init: float = 1.0,
        s_target: float = 0.5,
        pause_duration: float = 0.0,
    ) -> "Schedule":
        """Reverse annealing schedule (Crosson-Harrow 2016 流の V 字形).

        ``pause_duration > 0`` を指定すると ``s_target`` に到達後その時間だけ
        ``s = s_target`` を保ってから ``s_init`` に戻る (Marshall-Venturelli-
        Rieffel 2019 / Chen-Lidar 2020 流の reverse + pause). 下降区間と
        上昇区間はそれぞれ ``(T - pause_duration) / 2`` の長さを持つ.
        ``pause_duration = 0`` で従来の V 字形に縮退.
        """
        if not (0.0 <= pause_duration < T):
            raise ValueError(
                f"pause_duration must satisfy 0 <= pause_duration < T, "
                f"got pause_duration={pause_duration!r}, T={T!r}"
            )
        ramp = (T - pause_duration) / 2.0
        pause_end = ramp + pause_duration

        def s_of_t(t: float) -> float:
            if t <= ramp:
                return s_init + (s_target - s_init) * (t / ramp)
            if t <= pause_end:
                return s_target
            return s_target + (s_init - s_target) * ((t - pause_end) / ramp)

        return cls(T=T, A=lambda s: 1.0 - s, B=lambda s: s, h_x=h_x, s=s_of_t)

    @classmethod
    def pause(
        cls,
        T: float,
        h_x: np.ndarray,
        t_pause: float,
        duration: float,
    ) -> "Schedule":
        """Pause schedule (King-Carrasquilla 2018 流)."""
        if duration < 0 or duration >= T:
            raise ValueError(
                f"duration must satisfy 0 <= duration < T, "
                f"got duration={duration!r}, T={T!r}"
            )
        if t_pause < 0 or t_pause + duration > T:
            raise ValueError(
                f"t_pause must satisfy 0 <= t_pause and "
                f"t_pause + duration <= T, got t_pause={t_pause!r}, "
                f"duration={duration!r}, T={T!r}"
            )
        ramp_total = T - duration
        s_at_pause = t_pause / ramp_total

        def s_of_t(t: float) -> float:
            if t <= t_pause:
                return t / ramp_total
            if t <= t_pause + duration:
                return s_at_pause
            return s_at_pause + (t - t_pause - duration) / ramp_total

        return cls(T=T, A=lambda s: 1.0 - s, B=lambda s: s, h_x=h_x, s=s_of_t)

    @classmethod
    def from_xyz(
        cls,
        T: float,
        g_x: Sequence[Callable[[float], float]],
        b: Callable[[float], float],
        *,
        g_y: Sequence[Callable[[float], float]] | None = None,
        g_z: Sequence[Callable[[float], float]] | None = None,
    ) -> "Schedule":
        """per-site/per-axis 時間依存場の Schedule を構築する (新 API, issue #142).

        Hamiltonian 形:

        .. code-block:: text

            H(t) = Σ_i [g_x_i(t)·X_i + g_y_i(t)·Y_i + g_z_i(t)·Z_i] + b(t)·H_p_diag

        callable に h_x の静的振幅は既に組み込み済の前提
        (例えば旧 API ``g_x_i(t) := -a_t(t) · h_x_i`` 相当を新 API で書くなら
        ``g_x = [lambda t, hi=hi: -(1 - t/T) * hi for hi in h_x]``).

        Parameters
        ----------
        T : float
            総時間 (``T > 0``).
        g_x : Sequence[Callable[[float], float]]
            per-site の X 軸時間依存係数 list (length n).
        b : Callable[[float], float]
            problem Hamiltonian の global envelope.
        g_y, g_z : Sequence[Callable] | None
            per-site の Y / Z 軸時間依存係数. ``None`` で当該軸 skip
            (Rust 側で real-only SIMD kernel に dispatch する fast path).

        Raises
        ------
        ValueError
            ``T <= 0``, ``g_x`` が空 / 非 callable, ``g_y`` / ``g_z`` の長さが
            ``g_x`` と異なる場合.

        Notes
        -----
        Trotter method (``"trotter"`` / ``"trotter_suzuki4"``) は新 API
        Schedule で呼ばれた場合 ``ValueError``. Trotter 経路は実数係数前提
        SIMD で組まれているため XYZ 一般化が scope 外 (issue #142 Out of scope,
        必要時に別 issue で対応).
        """
        if not (T > 0):
            raise ValueError(f"T must be positive, got {T!r}")
        if not callable(b):
            raise ValueError(f"b must be callable, got {type(b).__name__}")
        g_x_list = list(g_x)
        if len(g_x_list) < 1:
            raise ValueError("g_x must be a non-empty sequence of callables")
        n = len(g_x_list)
        for i, cb in enumerate(g_x_list):
            if not callable(cb):
                raise ValueError(f"g_x[{i}] must be callable, got {type(cb).__name__}")

        g_y_list = _validate_callable_list(g_y, n, "g_y") if g_y is not None else None
        g_z_list = _validate_callable_list(g_z, n, "g_z") if g_z is not None else None

        # __init__ を経由せず内部状態を直接構築 (legacy state は None).
        obj = cls.__new__(cls)
        obj.T = float(T)
        obj.n = int(n)
        obj._kind = "xyz"
        obj._A = None
        obj._B = None
        obj._s = None
        obj._h_x = None
        obj._h_x_abs_sum = 0.0  # sentinel; 新 API では使わない
        obj._g_x_cbs = g_x_list
        obj._g_y_cbs = g_y_list
        obj._g_z_cbs = g_z_list
        obj._b_cb = b
        return obj

    # ------------------------------------------------------------------
    # Legacy API methods (旧 API only)
    # ------------------------------------------------------------------

    def s_at(self, t: float) -> float:
        """``s(t)`` をスカラーで返す (旧 API only).

        新 API (``from_xyz``) で構築された Schedule では ``s(t)`` の概念が
        ない (callable list が直接時間依存係数を返すため). 呼ぶと
        ``RuntimeError``.
        """
        if self._kind != "legacy" or self._s is None:
            raise RuntimeError(
                "s_at is only available for legacy Schedule (constructed via "
                "Schedule(...) / Schedule.linear / Schedule.reverse / Schedule.pause). "
                "Schedule.from_xyz does not have an s(t) concept."
            )
        return float(self._s(t))

    def coeffs_at(self, t: float) -> tuple[float, float]:
        """``(A(s(t)), B(s(t)))`` を返す (旧 API only).

        新 API で構築された Schedule では使えない. 呼ぶと ``RuntimeError``.
        旧 API path で QuTiP 参照実装等が ``(a_t, b_t)`` スカラー対を要求する
        ケースで使う.
        """
        if (
            self._kind != "legacy"
            or self._A is None
            or self._B is None
            or self._s is None
        ):
            raise RuntimeError(
                "coeffs_at is only available for legacy Schedule. "
                "Use _eval_stage(t) for the unified per-axis form."
            )
        s_val = float(self._s(t))
        return float(self._A(s_val)), float(self._B(s_val))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_xyz_api(self) -> bool:
        """新 API (``from_xyz``) で構築されたかどうか.

        driver / annealer 側で Trotter method の入口 ValueError 判定や
        Gershgorin O(1) fast path 分岐に使う.
        """
        return self._kind == "xyz"

    @property
    def h_x(self) -> np.ndarray:
        """サイト依存横磁場の振幅 (旧 API only).

        ``H_driver = -Σ_i h_x_i · X_i``. 新 API では callable に h_x が
        組み込み済なので別途振幅を取り出せず, 呼ぶと ``RuntimeError``.
        """
        if self._kind != "legacy" or self._h_x is None:
            raise RuntimeError(
                "h_x is only available for legacy Schedule. "
                "Schedule.from_xyz embeds the amplitude into the callable list."
            )
        return self._h_x

    @property
    def h_x_abs_sum(self) -> float:
        """``Σ_i |h_x_i|`` の precompute 値 (旧 API only).

        Chebyshev propagator の Gershgorin 上界 closed-form 計算
        (``gershgorin_per_stage_x_only``) で使う. 新 API では時間依存なので
        static cache 不可 (``_norm_upper_bound_factor_at(t)`` を使う).
        """
        if self._kind != "legacy":
            raise RuntimeError(
                "h_x_abs_sum is only available for legacy Schedule. "
                "Use _norm_upper_bound_factor_at(t) for the per-stage form."
            )
        return self._h_x_abs_sum

    # ------------------------------------------------------------------
    # Unified evaluator (driver hot path)
    # ------------------------------------------------------------------

    def _eval_stage(
        self, t: float
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, float]:
        """per-stage の ``(g_x_arr, g_y_arr_opt, g_z_arr_opt, b_scalar)`` を返す.

        driver の CFM4 stage / M2 中点 / Trotter (legacy 経路) で per-step に
        呼ぶ統一 evaluator. 返り値 array はすべて C-contiguous float64 で,
        新規 alloc される (caller 側で in-place 変更しないこと).

        - 旧 API (`_kind == "legacy"`): ``a_t = A(s(t))``, ``b_t = B(s(t))`` を
          評価して ``g_x_arr = -a_t · self._h_x``, ``g_y_arr = g_z_arr = None``,
          ``b_scalar = b_t``. 符号は ``H_drv = -Σ h_x X`` 規約を吸収.
        - 新 API (`_kind == "xyz"`): callable list を per-site に評価して
          ``g_x_arr[i] = self._g_x_cbs[i](t)``, ``g_y_arr / g_z_arr`` も同様
          (callable list が None の axis は ``None`` を返して Rust 側で
          real-only fast path に dispatch).
        """
        if self._kind == "legacy":
            assert self._h_x is not None
            assert self._A is not None and self._B is not None and self._s is not None
            s_val = float(self._s(t))
            a_t = float(self._A(s_val))
            b_t = float(self._B(s_val))
            g_x_arr = (-a_t) * self._h_x
            # 上の式は new ndarray を返すが念のため C-contiguous 保証.
            g_x_arr = np.ascontiguousarray(g_x_arr, dtype=np.float64)
            return g_x_arr, None, None, b_t

        # xyz API
        assert self._g_x_cbs is not None and self._b_cb is not None
        n = self.n
        g_x_arr = np.empty(n, dtype=np.float64)
        for i, cb in enumerate(self._g_x_cbs):
            g_x_arr[i] = float(cb(t))
        if self._g_y_cbs is None:
            g_y_arr: np.ndarray | None = None
        else:
            g_y_arr = np.empty(n, dtype=np.float64)
            for i, cb in enumerate(self._g_y_cbs):
                g_y_arr[i] = float(cb(t))
        if self._g_z_cbs is None:
            g_z_arr: np.ndarray | None = None
        else:
            g_z_arr = np.empty(n, dtype=np.float64)
            for i, cb in enumerate(self._g_z_cbs):
                g_z_arr[i] = float(cb(t))
        b_scalar = float(self._b_cb(t))
        return g_x_arr, g_y_arr, g_z_arr, b_scalar

    def _norm_upper_bound_factor_at(self, t: float) -> float:
        """Gershgorin 行和 off-diagonal 上界 ``Σ_i √(g_x_i(t)² + g_y_i(t)²)`` を返す.

        Chebyshev propagator の per-stage spectral radius 見積もりに使う
        (driver から ``schedule._norm_upper_bound_factor_at(t)`` で呼ぶ).
        旧 API では ``|a_t(t)| · h_x_abs_sum`` に degenerate.

        ``g_y = None`` の場合は単純な ``Σ_i |g_x_i(t)|`` (X-only 経路).
        新 API path で Y-only / XY 混在を統一的に扱える.
        """
        if self._kind == "legacy":
            assert self._A is not None and self._s is not None
            a_t = float(self._A(float(self._s(t))))
            return abs(a_t) * self._h_x_abs_sum

        assert self._g_x_cbs is not None
        n = self.n
        if self._g_y_cbs is None:
            total = 0.0
            for cb in self._g_x_cbs:
                total += abs(float(cb(t)))
            return total
        total = 0.0
        for i in range(n):
            gx_i = float(self._g_x_cbs[i](t))
            gy_i = float(self._g_y_cbs[i](t))
            total += float(np.hypot(gx_i, gy_i))
        return total
