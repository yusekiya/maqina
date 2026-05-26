"""アニーリングスケジュール (``Schedule``).

``s(t)`` および ``A(s)`` / ``B(s)`` の時間依存パラメータを 1 つの
``Schedule`` オブジェクトに集約する. Hamiltonian は

    H(t) = A(s(t)) · H_driver + B(s(t)) · H_problem

の形をとり, ``Schedule`` は ``(A(s(t)), B(s(t)))`` の per-step 評価のみを
責務とする (積分や駆動ドライバへの渡し方は ``annealer`` / ``krylov`` 側).

公開コンストラクタ:

* ``Schedule.linear(T)``: 標準的な線形スケジュール ``s(t) = t / T``,
  ``A(s) = 1 - s``, ``B(s) = s``.
* ``Schedule.from_callable(T, A, B, s=None)``: 任意の callable から構築する
  低レベル API.
* ``Schedule.reverse(T, s_init=1.0, s_target=0.5)``: V 字型 reverse
  annealing schedule.
* ``Schedule.pause(T, t_pause, duration)``: 線形 ramp の途中に
  ``[t_pause, t_pause + duration]`` の pause 区間を挟む schedule.

実装注: 全 phase で callable のまま per-step 評価する. 評価コストは
Lanczos 1 step の <1% のため grid cache 化の ROI が薄く, 将来必要に
なっても内部実装の差し替えで破壊変更なしで導入可能.
"""

from __future__ import annotations

from collections.abc import Callable

__all__ = ["Schedule"]


class Schedule:
    """Annealing schedule. ``s(t)``, ``A(s)``, ``B(s)`` を保持する.

    Parameters
    ----------
    T
        総アニーリング時間. ``T > 0``.
    A
        ``A: s -> float``. Driver 係数 (通常 ``1 - s``).
    B
        ``B: s -> float``. Problem 係数 (通常 ``s``).
    s
        ``s: t -> float``. ``None`` の場合は線形 ``s(t) = t / T``.

    Raises
    ------
    ValueError
        ``T <= 0`` の場合.
    """

    def __init__(
        self,
        T: float,
        A: Callable[[float], float],
        B: Callable[[float], float],
        s: Callable[[float], float] | None = None,
    ) -> None:
        if not (T > 0):
            raise ValueError(f"T must be positive, got {T!r}")
        self.T: float = float(T)
        self._A: Callable[[float], float] = A
        self._B: Callable[[float], float] = B
        if s is None:
            T_val = self.T
            self._s: Callable[[float], float] = lambda t, _T=T_val: t / _T
        else:
            self._s = s

    @classmethod
    def linear(cls, T: float) -> "Schedule":
        """線形スケジュール ``A(s) = 1 - s, B(s) = s, s(t) = t / T``.

        Parameters
        ----------
        T
            総アニーリング時間 (``T > 0``).
        """
        return cls(T=T, A=lambda s: 1.0 - s, B=lambda s: s)

    @classmethod
    def from_callable(
        cls,
        T: float,
        A: Callable[[float], float],
        B: Callable[[float], float],
        s: Callable[[float], float] | None = None,
    ) -> "Schedule":
        """任意の callable から ``Schedule`` を構築する低レベル API.

        ``__init__`` と等価. preset と区別するための明示的ファクトリ.
        """
        return cls(T=T, A=A, B=B, s=s)

    @classmethod
    def reverse(
        cls,
        T: float,
        s_init: float = 1.0,
        s_target: float = 0.5,
    ) -> "Schedule":
        """Reverse annealing schedule (Crosson-Harrow 2016 流の V 字形).

        ``s(t)`` は ``t=0`` で ``s_init``, ``t=T/2`` で ``s_target``,
        ``t=T`` で再び ``s_init`` に戻る V 字. ``A(s) = 1 - s``,
        ``B(s) = s`` は linear と同じ.

        Parameters
        ----------
        T
            総時間.
        s_init
            開始/終了時の ``s`` 値.
        s_target
            中央 ``t = T/2`` での ``s`` 値.
        """
        half = T / 2.0

        def s_of_t(t: float) -> float:
            if t <= half:
                # s_init から s_target へ
                return s_init + (s_target - s_init) * (t / half)
            # s_target から s_init へ戻る
            return s_target + (s_init - s_target) * ((t - half) / half)

        return cls(T=T, A=lambda s: 1.0 - s, B=lambda s: s, s=s_of_t)

    @classmethod
    def pause(
        cls,
        T: float,
        t_pause: float,
        duration: float,
    ) -> "Schedule":
        """Pause schedule (King-Carrasquilla 2018 流).

        通常の線形 ramp 中で ``t ∈ [t_pause, t_pause + duration]`` の区間
        だけ ``s(t)`` を一定 (``= t_pause / (T - duration)``) に保つ.
        ``s(0) = 0``, ``s(T) = 1`` を満たすよう pause を除いた区間で
        傾き ``1 / (T - duration)`` のランプを使う.
        ``A(s) = 1 - s``, ``B(s) = s``.

        Parameters
        ----------
        T
            総時間 (pause 区間を含む).
        t_pause
            pause 開始時刻. ``0 <= t_pause`` かつ
            ``t_pause + duration <= T``.
        duration
            pause 区間長. ``0 <= duration < T``.

        Raises
        ------
        ValueError
            ``t_pause``, ``duration`` が上記範囲外の場合.
        """
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

        return cls(T=T, A=lambda s: 1.0 - s, B=lambda s: s, s=s_of_t)

    def s_at(self, t: float) -> float:
        """``s(t)`` をスカラーで返す."""
        return float(self._s(t))

    def coeffs_at(self, t: float) -> tuple[float, float]:
        """``(A(s(t)), B(s(t)))`` を返す.

        Annealer / Rust 側に渡すスカラー対のホットパス. ``s(t)`` を 1 回
        評価したあと ``A`` / ``B`` をそれぞれ呼ぶ.
        """
        s_val = float(self._s(t))
        return float(self._A(s_val)), float(self._B(s_val))
