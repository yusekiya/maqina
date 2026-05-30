"""``AnnealingSimulator`` — step-wise stateful TFIM 量子ダイナミクスシミュレータ.

設計詳細は ``docs/design/04-python-api.md`` §4.5 を一次資料とする.

``QuantumAnnealer.run`` と同じプロパゲータ集合 (``m2`` / ``trotter`` /
``trotter_suzuki4`` / ``cfm4`` / ``cfm4_adaptive_richardson_krylov`` /
``cfm4_adaptive_richardson_chebyshev``) を内部で使うが, 1 step または部分
区間ごとに状態を取り出して ``Observable`` で測れる step-wise stateful API.
用途は中間時刻で diagnostic / observable history を取りつつ続きを発展させる
workflow.

設計判断
--------
* ``__init__`` で ``problem`` / ``schedule`` / ``psi0`` / ``t0`` /
  ``method`` と method 依存パラメータを fix し, 以後の ``step`` /
  ``advance_to`` は内部 ``_t`` / ``_psi`` を逐次更新する.
* ``psi`` プロパティは defensive copy を返し, 戻り値の mutation が
  内部状態に影響しないことを保証する.
* ``measure(observable)`` は read-only なので defensive copy しない
  (``Observable.expectation`` 内で ``abs(psi)**2`` を取るため).
* 数値的に ``QuantumAnnealer.run`` と完全一致 (固定 dt 経路で
  ``rel < 1e-13``) させるため, 内部では同じ ``evolve_schedule_*`` driver
  を呼ぶ. ``step`` は ``n_steps=1`` の薄いラッパ, ``advance_to`` は
  fixed-dt なら ``run`` と同じ driver call.
* adaptive 経路 (``cfm4_adaptive_richardson_krylov`` /
  ``cfm4_adaptive_richardson_chebyshev``) の ``step(dt)`` は ``dt`` を PI
  controller の proposal として渡し, driver 側で reject が起これば dt を
  縮めて再試行 (``dt_max=dt`` で growth を禁じ, ``dt0=dt`` で初回試行).
  結果として ``_t`` は exactly ``+dt`` 進む (1 step(dt) 内に複数 internal
  accept step が発生する可能性あり; その場合 ``n_matvec`` は累積
  m_eff_sum / K_used_sum で加算される).
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from maqina._helpers import (
    _KRYLOV_TOL_ATOL_RATIO,
    _KRYLOV_TOL_FIXED_DEFAULT,
    _resolve_dt_init_auto,
    _resolve_dt_max_auto,
    _validate_psi0,
)
from maqina.controller import ControllerConfig
from maqina.krylov import (
    evolve_schedule_adaptive_richardson,
    evolve_schedule_adaptive_richardson_chebyshev,
    evolve_schedule_cfm4,
    evolve_schedule_m2,
    evolve_schedule_trotter,
    evolve_schedule_trotter_suzuki4,
)
from maqina.observable import Observable
from maqina.problem import IsingProblem
from maqina.schedule import Schedule

__all__ = ["AnnealingSimulator"]


_FIXED_DT_METHODS: frozenset[str] = frozenset(
    {"m2", "trotter", "trotter_suzuki4", "cfm4"}
)
_ADAPTIVE_METHODS: frozenset[str] = frozenset(
    {"cfm4_adaptive_richardson_krylov", "cfm4_adaptive_richardson_chebyshev"}
)
_VALID_METHODS: frozenset[str] = _FIXED_DT_METHODS | _ADAPTIVE_METHODS

# adaptive driver の ``tol_step`` default. QuantumAnnealer.run と同一値.
_ADAPTIVE_TOL_STEP_DEFAULT: float = 1e-8


class AnnealingSimulator:
    """Step-wise stateful TFIM 量子ダイナミクスシミュレータ.

    Parameters
    ----------
    problem
        ``IsingProblem``. ``H_p_diag`` / ``h_x`` を保持する.
    schedule
        ``Schedule``. ``coeffs_at(t)`` から ``(A(s(t)), B(s(t)))`` を得る.
    psi0
        shape ``(2**n,)`` complex128 の初期状態. L2-normalize 済み
        (``|‖psi0‖ - 1| < 1e-10``) であることを検証する.
    t0
        初期時刻.
    method
        プロパゲータ. ``QuantumAnnealer.run`` と同じ集合をサポート
        (``m2`` / ``trotter`` / ``trotter_suzuki4`` / ``cfm4`` /
        ``cfm4_adaptive_richardson_krylov`` /
        ``cfm4_adaptive_richardson_chebyshev``). default は
        ``"cfm4_adaptive_richardson_chebyshev"`` (issue #124,
        Phase B Pareto win に基づく).
    m
        Lanczos / Krylov 部分空間次元. ``m >= 1``. default ``24``.
        ``trotter`` / ``trotter_suzuki4`` / ``cfm4_adaptive_richardson_chebyshev``
        経路では無視される (Lanczos 非使用; Chebyshev は K_used 動的決定).
    propagator_tol
        **短時間プロパゲータ U(dt) の per-step 許容誤差** (issue #135 で
        ``krylov_tol`` から rename). Lanczos 経路では Krylov 近似の許容誤差
        (a posteriori 推定子 ``est = β · |c_last| · |dt| / m`` を
        ``est < propagator_tol`` で切る; issue #98 / Phase 8). Chebyshev
        経路では Chebyshev 切り捨て次数 ``K_used`` を決める許容誤差.
        詳細は ``QuantumAnnealer.propagator_tol`` の docstring 参照.

        ``None`` (default) のとき経路ごとに自動解決する (QuantumAnnealer と
        同じポリシー):

        * adaptive Lanczos (``cfm4_adaptive_richardson_krylov``):
          ``effective = tol_step · 1e-3`` (``tol_step`` は ``atol`` で決まる;
          default ``atol=1e-8`` → ``1e-11``). atol scaling 連動.
        * adaptive Chebyshev (``cfm4_adaptive_richardson_chebyshev``):
          **固定 ``1e-12``** (issue #135 で auto-coupling から変更).
          Chebyshev は K_used の対数依存で auto-coupling の動機が弱く,
          固定値で atol-vs-infidelity の monotonicity を確保する.
        * 固定 dt 経路: ``1e-12`` (static fallback).
    atol
        adaptive 経路 (``cfm4_adaptive_richardson_krylov`` /
        ``cfm4_adaptive_richardson_chebyshev``) 専用. PI controller の局所
        誤差閾値 ``tol_step``. ``None`` (default) で driver default ``1e-8``
        を使う. 固定 dt method で指定すると ``ValueError``.

        **Note (Chebyshev variant の atol 振舞い, issue #124 / #135)**:
        Chebyshev では ``atol`` は **upper bound** として機能し, 実際の精度
        がそれより良くなる場合がある (K_used を ``propagator_tol`` (default
        ``1e-12`` 固定, issue #135) から動的決定するため per-stage 誤差が
        ``tol_step`` より遥かに小さくなり, PI controller が見る誤差は Magnus
        4 次成分のみになる). 詳細は ``QuantumAnnealer.run`` の ``atol``
        docstring 注 + ``docs/design/05-3-propagator.md`` "Chebyshev variant" 節.
    dt_init
        adaptive 経路専用. ``advance_to`` 時の初期 dt 提案 (driver の
        ``dt0`` に map). ``None`` (default) のとき ``advance_to`` の各
        呼出時に ``T = t_target - _t`` から auto-resolve
        (``c · T^β``, c=0.1, β=0.5, floor=1e-3). ``step(dt)`` の動作には
        影響しない (step は呼出時の ``dt`` を proposal にする). 固定 dt
        method で指定すると ``ValueError``.
    dt_max
        adaptive 経路専用. ``advance_to`` 時の最大 dt 上限 (driver の
        ``dt_max`` に map). ``None`` (default) のとき Gershgorin 上界
        による Lanczos capacity 自動見積もりで auto-resolve.
        ``step(dt)`` では ``dt`` を上限として上書きする (growth 禁止,
        shrinkage 可). 固定 dt method で指定すると ``ValueError``.
    m_max
        adaptive 経路専用. Lanczos 部分空間次元の上限を ``m`` から上書き
        する. ``None`` (default) で ``m`` をそのまま使う. 固定 dt method
        で指定すると ``ValueError``. ``QuantumAnnealer.run`` の ``m_max``
        と同義.
    controller
        adaptive 経路専用. PI controller の数値挙動 knob を集約した
        :class:`~maqina.ControllerConfig` (issue #149). ``None`` (default)
        で全 default. ``safety`` / ``growth_max`` / ``max_rejects`` /
        ``dt_min`` / ``reject_shrink_min`` / ``reject_shrink_max`` を driver
        に渡す. 固定 dt method で指定すると ``ValueError`` (``QuantumAnnealer.run``
        と違い Simulator は strict). reject 時の dt 縮小が固定 0.5 倍から
        予測式 + クランプに変わるため既定挙動が変わる (破壊的変更). 旧挙動は
        ``ControllerConfig(reject_shrink_min=0.5, reject_shrink_max=0.5)``.

    Raises
    ------
    NotImplementedError
        ``method`` がサポート対象外の場合.
    ValueError
        ``m`` / ``propagator_tol`` / ``atol`` / ``dt_init`` / ``dt_max`` /
        ``m_max`` が範囲外, ``psi0`` の shape / dtype / 非正規化,
        固定 dt method に adaptive 専用パラメータ (``controller`` 含む) を
        渡した場合.

    Examples
    --------
    >>> import numpy as np
    >>> from maqina import IsingProblem, Observable, Schedule
    >>> from maqina.builders import diag_from_J_h
    >>> from maqina.initial_states import uniform_superposition
    >>> from maqina.simulator import AnnealingSimulator
    >>>
    >>> n = 4
    >>> J = np.zeros((n, n)); J[0, 1] = J[1, 0] = -1.0
    >>> prob = IsingProblem(n=n, H_p_diag=diag_from_J_h(J, np.zeros(n)),
    ...                     h_x=np.ones(n))
    >>> sched = Schedule.linear(T=10.0)
    >>> psi0 = uniform_superposition(n)
    >>> # default ``method="cfm4_adaptive_richardson_chebyshev"`` を使う場合,
    >>> # ``advance_to`` には ``n_steps`` を渡さない (PI controller が dt を
    >>> # 内部決定する). 固定 dt 経路を使いたい場合は ``method="cfm4"`` 等を
    >>> # 明示し ``advance_to(..., n_steps=N)`` を渡す.
    >>> sim = AnnealingSimulator(prob, sched, psi0, 0.0)
    >>> obs = Observable.magnetization(n)
    >>> sim.advance_to(5.0)
    >>> mz_mid = sim.measure(obs)
    >>> sim.advance_to(10.0)
    >>> mz_end = sim.measure(obs)
    """

    def __init__(
        self,
        problem: IsingProblem,
        schedule: Schedule,
        psi0: np.ndarray,
        t0: float,
        *,
        method: Literal[
            "m2",
            "trotter",
            "trotter_suzuki4",
            "cfm4",
            "cfm4_adaptive_richardson_krylov",
            "cfm4_adaptive_richardson_chebyshev",
        ] = "cfm4_adaptive_richardson_chebyshev",
        m: int = 24,
        propagator_tol: float | None = None,
        atol: float | None = None,
        dt_init: float | None = None,
        dt_max: float | None = None,
        m_max: int | None = None,
        controller: ControllerConfig | None = None,
    ) -> None:
        if method not in _VALID_METHODS:
            raise NotImplementedError(
                f"method={method!r} is not supported; valid methods are "
                f"{sorted(_VALID_METHODS)!r}."
            )
        if not isinstance(m, (int, np.integer)) or m < 1:
            raise ValueError(f"m must be a positive integer, got {m!r}")
        if propagator_tol is not None and propagator_tol < 0.0:
            raise ValueError(
                f"propagator_tol must be >= 0 or None, got {propagator_tol!r}"
            )

        # adaptive-only パラメータが固定 dt method で指定されていれば早期に弾く.
        # silent 無視は debug 罠 (e.g., method="m2" で atol=1e-5 を渡しても
        # PI controller は走らない) になるため明示的に ValueError.
        if method in _FIXED_DT_METHODS:
            for name, val in (
                ("atol", atol),
                ("dt_init", dt_init),
                ("dt_max", dt_max),
                ("m_max", m_max),
                ("controller", controller),
            ):
                if val is not None:
                    raise ValueError(
                        f"{name} is only valid for adaptive method "
                        f"('cfm4_adaptive_richardson_krylov' or "
                        f"'cfm4_adaptive_richardson_chebyshev'); got "
                        f"method={method!r} with {name}={val!r}"
                    )

        if atol is not None and not (atol > 0.0):
            raise ValueError(f"atol must be > 0 or None, got {atol!r}")
        if dt_init is not None and not (dt_init > 0.0):
            raise ValueError(f"dt_init must be > 0 or None, got {dt_init!r}")
        if dt_max is not None and not (dt_max > 0.0):
            raise ValueError(f"dt_max must be > 0 or None, got {dt_max!r}")
        if m_max is not None and (
            not isinstance(m_max, (int, np.integer)) or m_max < 1
        ):
            raise ValueError(f"m_max must be a positive integer or None, got {m_max!r}")
        # issue #122 (Phase B): Chebyshev variant は ``m_max`` を取らない
        # (K_used は ``chebyshev_tol`` から動的決定; Krylov 部分空間次元の概念
        # 自体が無い). silent 無視は debug 罠なので明示的に弾く.
        if method == "cfm4_adaptive_richardson_chebyshev" and m_max is not None:
            raise ValueError(
                "m_max is not supported for method="
                "'cfm4_adaptive_richardson_chebyshev' "
                "(Chebyshev uses dynamic K_used, not Krylov subspace dimension)."
            )

        psi0_arr = _validate_psi0(problem, psi0)

        self.problem: IsingProblem = problem
        self.schedule: Schedule = schedule
        self._method: str = method
        self._m: int = int(m)
        self._propagator_tol_user: float | None = (
            float(propagator_tol) if propagator_tol is not None else None
        )
        self._atol: float | None = float(atol) if atol is not None else None
        self._dt_init: float | None = float(dt_init) if dt_init is not None else None
        self._dt_max: float | None = float(dt_max) if dt_max is not None else None
        self._m_max: int | None = int(m_max) if m_max is not None else None
        # issue #149: controller knob を ControllerConfig に集約. None なら
        # 全 default. adaptive driver 呼出 (_run_adaptive) で field を展開する.
        self._controller: ControllerConfig = (
            controller if controller is not None else ControllerConfig()
        )

        # _psi は呼出側 psi0 の後続 mutation から内部状態を守るため copy.
        # _validate_psi0 は C-contiguous な配列を返すが, 既存 buffer を返す
        # ケースもあるので defensive copy.
        self._t: float = float(t0)
        self._psi: np.ndarray = psi0_arr.copy()
        self._n_matvec: int = 0

    # ------------------------------------------------------------------
    # 公開 read-only プロパティ
    # ------------------------------------------------------------------

    @property
    def t(self) -> float:
        """現在時刻."""
        return self._t

    @property
    def psi(self) -> np.ndarray:
        """現在状態 ``ψ`` の defensive copy (shape ``(2**n,)`` complex128).

        戻り値への mutation は内部状態に影響しない. 内部状態のまま参照
        したい場合は ``measure(observable)`` 経由で観測値を取得する.
        """
        return self._psi.copy()

    @property
    def n_matvec(self) -> int:
        """累積 matvec 数 (経路ごとの見積もり; ``QuantumAnnealer.run`` と同規約)."""
        return self._n_matvec

    @property
    def method(self) -> str:
        """選択された propagator method 名 (immutable)."""
        return self._method

    # ------------------------------------------------------------------
    # 状態更新
    # ------------------------------------------------------------------

    def step(self, dt: float) -> None:
        """``dt`` だけ進める (1 step).

        固定 dt 経路 (``m2`` / ``trotter`` / ``trotter_suzuki4`` /
        ``cfm4``) では文字通り 1 step (``n_steps=1`` の driver call).
        adaptive 経路 (``cfm4_adaptive_richardson_krylov`` /
        ``cfm4_adaptive_richardson_chebyshev``) では ``dt`` を PI controller
        の proposal として渡し, driver 側で reject が起きれば dt を縮めて
        再試行する. いずれの場合も呼出後の ``_t`` は exactly ``+dt`` 進む
        (adaptive 経路で 1 step(dt) 内に複数 internal accept step が発生
        する可能性があるが, その合計 m_eff_sum / K_used_sum が
        ``n_matvec`` に加算される).

        Parameters
        ----------
        dt
            正の float. ``dt > 0``.

        Raises
        ------
        ValueError
            ``dt <= 0`` の場合.
        RuntimeError
            adaptive 経路で ``max_rejects`` 連続超過したとき (driver から
            伝播).
        """
        if not (dt > 0.0):
            raise ValueError(f"dt must be > 0, got {dt!r}")
        dt_f = float(dt)
        t_next = self._t + dt_f
        if self._method in _FIXED_DT_METHODS:
            self._run_fixed_dt(t_next, n_steps=1)
        else:
            # adaptive: dt を proposal にして driver を [_t, _t+dt] で呼ぶ.
            # dt_max=dt で growth を禁止, 内部 reject は dt 縮めて再試行
            # (PI controller 動作).
            self._run_adaptive(t_next, dt_init_override=dt_f, dt_max_override=dt_f)

    def advance_to(self, t_target: float, *, n_steps: int | None = None) -> None:
        """``t_target`` まで進める.

        固定 dt 経路では ``n_steps`` 必須 (``QuantumAnnealer.run`` と同じ
        map: ``dt = (t_target - _t) / n_steps``). adaptive 経路では
        ``n_steps`` は None でなければならない (driver が step 数を内部
        決定するため). adaptive 経路の ``dt_init`` / ``dt_max`` /
        ``atol`` / ``m_max`` は ``__init__`` の設定値 (None なら auto
        resolve) を使う.

        Parameters
        ----------
        t_target
            目標時刻. ``t_target > _t`` を要求 (``_t == t_target`` の
            no-op は許容しない: 呼出意図不明確のため明示的に弾く).
        n_steps
            固定 dt method の step 数 (正の整数, 必須). adaptive method
            では None 以外を渡すと ``ValueError``.

        Raises
        ------
        ValueError
            ``t_target <= _t``, 固定 dt で ``n_steps`` 未指定 /
            非正整数, adaptive で ``n_steps`` 非 None の場合.
        RuntimeError
            adaptive 経路で ``max_rejects`` 連続超過したとき.
        """
        t_target_f = float(t_target)
        if not (t_target_f > self._t):
            raise ValueError(
                f"t_target must be > current t={self._t!r}, got {t_target_f!r}"
            )
        if self._method in _FIXED_DT_METHODS:
            if n_steps is None:
                raise ValueError(
                    f"n_steps is required for fixed-dt method "
                    f"{self._method!r}; pass a positive integer."
                )
            if not isinstance(n_steps, (int, np.integer)) or n_steps < 1:
                raise ValueError(f"n_steps must be a positive integer, got {n_steps!r}")
            self._run_fixed_dt(t_target_f, n_steps=int(n_steps))
        else:
            if n_steps is not None:
                raise ValueError(
                    f"n_steps must be None for adaptive method "
                    f"{self._method!r}; adaptive driver determines step "
                    "count internally."
                )
            self._run_adaptive(t_target_f)

    def measure(self, observable: Observable) -> float:
        """現在 ψ で観測量の期待値 ``<ψ|O|ψ>`` を返す (実数).

        ``Observable.expectation`` に内部状態をそのまま渡す
        (``abs(psi)**2`` を取るため read-only; defensive copy 不要).

        Parameters
        ----------
        observable
            ``Observable`` インスタンス. Z 基底対角な Hermitian 演算子.

        Returns
        -------
        float
            ``<ψ|O|ψ>``.

        Raises
        ------
        TypeError
            ``observable`` が ``Observable`` インスタンスでない場合.
        ValueError
            ``observable.diag`` と ψ の shape 不一致 (Observable 側で raise).
        """
        if not isinstance(observable, Observable):
            raise TypeError(
                "observable must be an Observable instance, got "
                f"{type(observable).__name__}"
            )
        return observable.expectation(self._psi)

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _resolved_propagator_tol_fixed(self) -> float:
        if self._propagator_tol_user is not None:
            return self._propagator_tol_user
        return _KRYLOV_TOL_FIXED_DEFAULT

    def _resolved_propagator_tol_adaptive(self, tol_step: float) -> float:
        """adaptive Richardson (Lanczos) 経路の propagator_tol 自動解決.

        Lanczos a posteriori 早期打切は atol scaling で連動するのが望ましい
        ため auto-coupling (``tol_step · _KRYLOV_TOL_ATOL_RATIO``) を採る.
        """
        if self._propagator_tol_user is not None:
            return self._propagator_tol_user
        return tol_step * _KRYLOV_TOL_ATOL_RATIO

    def _resolved_propagator_tol_chebyshev_adaptive(self) -> float:
        """adaptive Richardson (Chebyshev variant) 経路の propagator_tol 自動解決.

        issue #135 で auto-coupling (``tol_step · _KRYLOV_TOL_ATOL_RATIO``)
        から **固定値 ``_KRYLOV_TOL_FIXED_DEFAULT`` (= 1e-12)** に変更.
        Chebyshev は ``K_used ~ R·dt + log(1/propagator_tol)`` の対数依存で
        propagator_tol を atol に連動させる動機が弱く, auto-coupling だと
        atol↓ で machine precision 到達後に round-off accumulation で精度が
        悪化する非単調性が発生する.
        """
        if self._propagator_tol_user is not None:
            return self._propagator_tol_user
        return _KRYLOV_TOL_FIXED_DEFAULT

    def _run_fixed_dt(self, t_next: float, *, n_steps: int) -> None:
        """固定 dt driver を ``[_t, t_next]`` 区間で呼んで状態更新する.

        ``observables`` / ``save_tlist`` / ``store_states`` は Simulator
        の用途外なので常に default (None / False) を渡す. ``run`` と同じ
        driver を同じ引数で呼ぶため, 同じ schedule 評価点で bit-identical
        な数値が出る.
        """
        propagator_tol = self._resolved_propagator_tol_fixed()
        if self._method == "m2":
            psi_new, n_matvec, _ = evolve_schedule_m2(
                h_p_diag=self.problem.H_p_diag,
                schedule=self.schedule,
                psi0=self._psi,
                t0=self._t,
                t1=t_next,
                n_steps=n_steps,
                m=self._m,
                krylov_tol=propagator_tol,
            )
        elif self._method == "trotter":
            psi_new, n_matvec, _ = evolve_schedule_trotter(
                h_p_diag=self.problem.H_p_diag,
                schedule=self.schedule,
                psi0=self._psi,
                t0=self._t,
                t1=t_next,
                n_steps=n_steps,
            )
        elif self._method == "trotter_suzuki4":
            psi_new, n_matvec, _ = evolve_schedule_trotter_suzuki4(
                h_p_diag=self.problem.H_p_diag,
                schedule=self.schedule,
                psi0=self._psi,
                t0=self._t,
                t1=t_next,
                n_steps=n_steps,
            )
        else:  # cfm4
            psi_new, n_matvec, _ = evolve_schedule_cfm4(
                h_p_diag=self.problem.H_p_diag,
                schedule=self.schedule,
                psi0=self._psi,
                t0=self._t,
                t1=t_next,
                n_steps=n_steps,
                m=self._m,
                krylov_tol=propagator_tol,
            )
        self._psi = psi_new
        self._t = t_next
        self._n_matvec += int(n_matvec)

    def _run_adaptive(
        self,
        t_next: float,
        *,
        dt_init_override: float | None = None,
        dt_max_override: float | None = None,
    ) -> None:
        """adaptive Richardson driver を ``[_t, t_next]`` 区間で呼ぶ.

        ``dt_init_override`` / ``dt_max_override`` を渡すと
        ``__init__`` の ``dt_init`` / ``dt_max`` を無視してこちらを使う
        (``step(dt)`` 経路用; 呼出側 ``dt`` を proposal にする).
        override が None なら ``__init__`` の値を使い, それも None なら
        ``QuantumAnnealer.run`` と同じ auto resolution を行う.

        ``n_matvec`` は driver の ``m_eff_history`` 合計で正確に加算する
        (早期打切で ``6m`` upper bound より小さくなる可能性があるため,
        累積コストを正確に追う).
        """
        tol_step = self._atol if self._atol is not None else _ADAPTIVE_TOL_STEP_DEFAULT
        # issue #135: Chebyshev variant の propagator_tol default は固定 1e-12
        # (Lanczos variant は引き続き auto-coupling = tol_step · 1e-3).
        if self._method == "cfm4_adaptive_richardson_chebyshev":
            propagator_tol = self._resolved_propagator_tol_chebyshev_adaptive()
        else:
            propagator_tol = self._resolved_propagator_tol_adaptive(tol_step)
        if dt_init_override is not None:
            dt0 = dt_init_override
        elif self._dt_init is not None:
            dt0 = self._dt_init
        else:
            dt0 = _resolve_dt_init_auto(self._t, t_next)
        m_eff_param = self._m_max if self._m_max is not None else self._m
        if dt_max_override is not None:
            dt_max_resolved = dt_max_override
        elif self._dt_max is not None:
            dt_max_resolved = self._dt_max
        else:
            dt_max_resolved = _resolve_dt_max_auto(
                self.schedule, self.problem, m_eff_param, dt0
            )
        # driver 入力検証 ``dt_max >= dt0`` を満たすため floor.
        # step(dt) では dt0=dt_max=dt なので一致, advance_to では auto
        # 解決 (_resolve_dt_max_auto) が内部で floor 済み.
        if dt_max_resolved < dt0:
            dt_max_resolved = dt0
        # issue #93 (Phase 7): driver は 10-tuple. simulator では β_m /
        # err_lanczos / err_magnus / n_krylov_insufficient は使わない
        # (simulator は per-step API なので diagnostic 集計は QuantumAnnealer
        # 経由で行う). 全て discard.
        # issue #122 (Phase B): Chebyshev variant も 10-tuple 互換で,
        # 4 番目の要素は m_eff_hist の代わりに k_used_hist (用途は同じ:
        # per-step propagator 評価コスト統計).
        if self._method == "cfm4_adaptive_richardson_chebyshev":
            (
                psi_new,
                _t_hist,
                _dt_hist,
                _n_rej,
                m_eff_hist,
                _beta_m_hist,
                _err_lanczos_hist,
                _err_magnus_hist,
                _n_krylov_insufficient,
                _snapshot,
            ) = evolve_schedule_adaptive_richardson_chebyshev(
                h_p_diag=self.problem.H_p_diag,
                schedule=self.schedule,
                psi0=self._psi,
                t0=self._t,
                t1=t_next,
                h_p_min=self.problem.h_p_diag_min,
                h_p_max=self.problem.h_p_diag_max,
                chebyshev_tol=propagator_tol,
                tol_step=tol_step,
                dt0=dt0,
                dt_min=self._controller.dt_min,
                dt_max=dt_max_resolved,
                safety=self._controller.safety,
                growth_max=self._controller.growth_max,
                max_rejects=self._controller.max_rejects,
                reject_shrink_min=self._controller.reject_shrink_min,
                reject_shrink_max=self._controller.reject_shrink_max,
            )
        else:
            (
                psi_new,
                _t_hist,
                _dt_hist,
                _n_rej,
                m_eff_hist,
                _beta_m_hist,
                _err_lanczos_hist,
                _err_magnus_hist,
                _n_krylov_insufficient,
                _snapshot,
            ) = evolve_schedule_adaptive_richardson(
                h_p_diag=self.problem.H_p_diag,
                schedule=self.schedule,
                psi0=self._psi,
                t0=self._t,
                t1=t_next,
                m=m_eff_param,
                krylov_tol=propagator_tol,
                tol_step=tol_step,
                dt0=dt0,
                dt_min=self._controller.dt_min,
                dt_max=dt_max_resolved,
                safety=self._controller.safety,
                growth_max=self._controller.growth_max,
                max_rejects=self._controller.max_rejects,
                reject_shrink_min=self._controller.reject_shrink_min,
                reject_shrink_max=self._controller.reject_shrink_max,
            )
        self._psi = psi_new
        self._t = t_next
        if m_eff_hist.size > 0:
            self._n_matvec += int(np.sum(m_eff_hist))
