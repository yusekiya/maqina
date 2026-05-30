"""層 A — 合成誤差ハーネス (issue #152, umbrella #148).

adaptive PI controller の **ダイナミクスだけ** を切り出して決定論的に検証する
ためのハーネス。``evolve_schedule_adaptive_*`` の dispatch 関数を monkeypatch し、
物理的な状態更新を行わずに合成則 **``err = C₄(t)·dt^{p+1}``** を返させる
(``p`` は推定子 order; M2 = 2, Richardson / Chebyshev = 4。driver 内
``_pi_dt_next(p=...)`` と一致させる)。

なぜ合成則か:
    PI 制御ループが各 step で依存するのは dispatch が返す ``err`` (Richardson /
    Chebyshev では ``err_magnus = err − err_{lanczos/cheb}``) **だけ**。よって
    ``err`` を解析的に与えれば、実際の Lanczos / Chebyshev / matvec を一切呼ばず
    純 float64 演算でコントローラ挙動を再現できる。これにより:

    - **決定論的・プラットフォーム非依存**: Rust 拡張のビルド有無に依らない
      (dispatch を差し替えるため実 Rust は呼ばれない。Chebyshev driver の
      ``_rust_mod is None`` fail-fast は ``_rust_mod`` をダミーに差し替えて回避)。
    - **高速**: dim=2 (n=1) の最小状態で十分。
    - **production 改修不要**: monkeypatch のみ。本 issue は production 挙動を
      変えない。

``C₄(t)`` の意味:
    Magnus 4 次局所誤差係数。臨界領域 (回避交差近傍) で時間微分・入れ子交換子が
    増大して急上昇する物理を模す。``exp_c4(k, t_star)`` で立ち上がり率 ``k`` を
    直接指定できる。per-step 成長率が ``1/safety⁵ ≈ 1.7`` を超えると I 制御の
    楽観予測がオーバーシュートして reject → ノコギリ波。閾値は ``exp(k·dt) > 1.7``
    すなわち ``k·dt ≳ ln(1.7) ≈ 0.53``。

``t`` の復元 (恒等スケジュール trick):
    dispatch は step 開始時刻 ``t`` を直接受け取らないが、CFM4 stage 係数
    ``b_s1 = b(t + _CFM4_C1·dt)`` を受け取る。問題 envelope を恒等 ``b(t) = t``
    にした legacy schedule を使えば ``b_s1 = t + _CFM4_C1·dt`` となり、fake 側で
    ``t = b_s1 − _CFM4_C1·dt`` を復元できる (reject 再試行でも ``t`` は不変)。

返り値は :class:`~_controller_metrics.ControllerTrace`。driver の返り値
(``t_history`` / ``dt_history`` / ``n_rejects``) と、fake が記録した全 attempt
から accept/reject フラグを再構成して保持する。
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Callable, Iterator

import numpy as np

import maqina.krylov as _kry
from maqina import Schedule

from _controller_metrics import ControllerTrace, StepAttempt

# CFM4:2 の第 1 ノード係数 (``krylov._CFM4_C1`` と同値)。自動生成 stub
# (``krylov.pyi``) は private 定数を export しないため再定義し、runtime 値との
# drift を下の assert で防ぐ。fake は ``t = b_s1 − _CFM4_C1·dt`` で step 開始
# 時刻を復元する。
_CFM4_C1 = 0.5 - 3.0**0.5 / 6.0
assert _CFM4_C1 == getattr(_kry, "_CFM4_C1"), (
    "krylov._CFM4_C1 が変わった: _controller_harness の定義を更新すること"
)

# サポートする method 名 (run_synthetic の ``method`` 引数)。
_METHODS = ("m2", "richardson", "chebyshev")

C4Fn = Callable[[float], float]


def exp_c4(k: float, t_star: float, *, cap: float = 50.0) -> C4Fn:
    """指数 ``C₄(t) = exp(k·(t − t_star))`` を返す.

    ``k`` が立ち上がり率、``t_star`` が臨界中心。``cap`` で指数引数を上限
    クランプして overflow (inf) を防ぐ (``cap=50`` で最大 ~5e21、十分大きく
    dt を ``dt_min`` まで潰す)。
    """

    def c4(t: float) -> float:
        a = k * (t - t_star)
        if a > cap:
            a = cap
        return math.exp(a)

    return c4


def pole_c4(
    t_sing: float, *, scale: float = 1.0, p: int = 5, floor: float = 1e-30
) -> C4Fn:
    r"""逆冪 (pole) 型 ``C₄(t) = scale · (t_sing − t)^{−p}`` を返す.

    ``t → t_sing`` で発散する臨界点モデル。指数型 ``exp(k(t−t*))`` と違い、
    operating dt が ``C₄^{−1/(p_est+1)}`` で縮むのと釣り合うように立ち上がり率
    ``d/dt ln C₄ = p/(t_sing − t)`` が増大するため、**per-step 成長率
    ``exp(dt · p/(t_sing − t))`` が臨界帯で長く閾値 ``1/safety⁵`` 上に留まり**、
    指数型より幅広い (= 多点の) ノコギリ波が得られる。``p=5`` が Richardson /
    Chebyshev の order-5 推定子と整合 (operating dt ∝ C₄^{−1/5})。

    ``t >= t_sing`` では ``floor`` でクランプした巨大値を返す (dt を ``dt_min``
    まで潰す)。
    """

    def c4(t: float) -> float:
        gap = t_sing - t
        if gap <= floor:
            gap = floor
        return scale * gap ** (-p)

    return c4


@contextlib.contextmanager
def _patched(attrs: dict[str, object]) -> Iterator[None]:
    """``maqina.krylov`` のモジュール属性を一時的に差し替える (pytest 非依存)."""
    saved = {name: getattr(_kry, name) for name in attrs}
    try:
        for name, value in attrs.items():
            setattr(_kry, name, value)
        yield
    finally:
        for name, value in saved.items():
            setattr(_kry, name, value)


def _make_m2_fake(c4: C4Fn, order: int, attempts: list[tuple[float, float, float]]):
    """``_adaptive_dispatch_m2_estimate`` 差し替え用 fake (返り ``(psi, err)``).

    ``err = C₄(t)·dt^{order+1}``。M2 embedded 推定子は order ``p=2`` なので
    局所誤差は ``dt³`` スケール (``order=2``)。controller の PI 指数
    (``1/(p+1)``) と整合させないと、健全な schedule でも誤差予測がずれて
    持続的 reject を起こす (合成則の order を driver の order に揃えるのが要点)。
    """
    power = order + 1

    def fake(
        rust_mod,
        psi,
        h_x,
        h_p_diag,
        a_s1,
        b_s1,
        a_s2,
        b_s2,
        a_mid,
        b_mid,
        dt,
        m,
        krylov_tol,
    ):
        t = b_s1 - _CFM4_C1 * dt
        err = c4(t) * dt**power
        attempts.append((float(t), float(dt), float(err)))
        return psi, float(err)

    return fake


def _make_richardson_fake(
    c4: C4Fn, order: int, attempts: list[tuple[float, float, float]]
):
    """``_adaptive_dispatch_richardson_estimate_xyz`` 差し替え用 fake.

    返り ``(psi, err, m_eff_sum, err_lanczos_total)``。``err_lanczos_total = 0``
    にして ``err_magnus = err`` (= 合成誤差) が PI 駆動量になるようにする。
    ``err = C₄(t)·dt^{order+1}`` (step-doubling Richardson は ``p=4`` → ``dt⁵``)。
    """
    power = order + 1

    def fake(
        rust_mod,
        psi,
        h_p_diag,
        g_x_s1_full,
        g_y_s1_full,
        g_z_s1_full,
        b_s1_full,
        g_x_s2_full,
        g_y_s2_full,
        g_z_s2_full,
        b_s2_full,
        g_x_s1_h1,
        g_y_s1_h1,
        g_z_s1_h1,
        b_s1_h1,
        g_x_s2_h1,
        g_y_s2_h1,
        g_z_s2_h1,
        b_s2_h1,
        g_x_s1_h2,
        g_y_s1_h2,
        g_z_s1_h2,
        b_s1_h2,
        g_x_s2_h2,
        g_y_s2_h2,
        g_z_s2_h2,
        b_s2_h2,
        dt,
        m,
        krylov_tol,
        extrapolate,
    ):
        t = b_s1_full - _CFM4_C1 * dt
        err = c4(t) * dt**power
        attempts.append((float(t), float(dt), float(err)))
        return psi, float(err), 0, 0.0

    return fake


def _make_chebyshev_fake(
    c4: C4Fn, order: int, attempts: list[tuple[float, float, float]]
):
    """``_adaptive_dispatch_richardson_estimate_chebyshev_xyz`` 差し替え用 fake.

    返り ``(psi, err, k_used_total, err_cheb_total)``。``err_cheb_total = 0`` で
    ``err_magnus = err``。``err = C₄(t)·dt^{order+1}`` (Chebyshev variant も
    step-doubling Richardson 構造で ``p=4`` → ``dt⁵``)。
    """
    power = order + 1

    def fake(
        rust_mod,
        psi,
        h_p_diag,
        g_x_s1_full,
        g_y_s1_full,
        g_z_s1_full,
        b_s1_full,
        g_x_s2_full,
        g_y_s2_full,
        g_z_s2_full,
        b_s2_full,
        g_x_s1_h1,
        g_y_s1_h1,
        g_z_s1_h1,
        b_s1_h1,
        g_x_s2_h1,
        g_y_s2_h1,
        g_z_s2_h1,
        b_s2_h1,
        g_x_s1_h2,
        g_y_s1_h2,
        g_z_s1_h2,
        b_s1_h2,
        g_x_s2_h2,
        g_y_s2_h2,
        g_z_s2_h2,
        b_s2_h2,
        dt,
        chebyshev_tol,
        extrapolate,
        h_p_min,
        h_p_max,
    ):
        t = b_s1_full - _CFM4_C1 * dt
        err = c4(t) * dt**power
        attempts.append((float(t), float(dt), float(err)))
        return psi, float(err), 0, 0.0

    return fake


def _identity_schedule(t1: float) -> Schedule:
    """``b(t) = t`` の恒等 legacy schedule (driver から t を復元するため).

    ``A(s) = 0`` (fake は g_x を無視), ``B(s) = s``, ``s(t) = t`` とすることで
    ``_eval_stage(t)`` の問題係数が ``t`` そのものになる。``h_x`` は形式上必要な
    だけで値は使われない。
    """
    return Schedule(
        T=t1,
        A=lambda s: 0.0,
        B=lambda s: s,
        h_x=np.ones(1, dtype=np.float64),
        s=lambda t: t,
    )


def _reconstruct_attempts(
    raw_attempts: list[tuple[float, float, float]],
    t_history: np.ndarray,
    dt_history: np.ndarray,
) -> list[StepAttempt]:
    """fake 記録の生 attempt 列に accept/reject フラグを付与する.

    driver は accept された step を ``dt_history`` に順番に積む。生 attempt を
    呼び出し順に走査し、現在 step の accept dt (``dt_history[k]``) と **完全一致**
    する attempt を accept、それ以外を reject とみなす (同一 step 内の reject は
    halving で必ず accept dt より大きいので衝突しない)。accept された attempt の
    時刻は driver の権威ある start time (``t_history[k]``) で上書きする。
    """
    out: list[StepAttempt] = []
    k = 0
    n_accept = int(dt_history.shape[0])
    for t_raw, dt_raw, err in raw_attempts:
        accepted = k < n_accept and dt_raw == float(dt_history[k])
        if accepted:
            t_use = float(t_history[k])
            k += 1
        else:
            t_use = t_raw
        out.append(StepAttempt(t=t_use, dt=dt_raw, err=err, accepted=accepted))
    if k != n_accept:
        raise AssertionError(
            f"attempt reconstruction mismatch: matched {k} accepts but "
            f"dt_history has {n_accept}. raw_attempts={len(raw_attempts)}"
        )
    return out


def run_synthetic(
    method: str,
    c4: C4Fn,
    *,
    t0: float = 0.0,
    t1: float = 20.0,
    tol_step: float = 1e-8,
    dt0: float = 0.5,
    dt_min: float = 1e-6,
    dt_max: float | None = None,
    safety: float = 0.9,
    growth_max: float = 4.0,
    max_rejects: int = 50,
    m: int = 24,
) -> ControllerTrace:
    """合成誤差 ``err = c4(t)·dt^{p+1}`` で実 adaptive driver を駆動し trace を返す.

    Parameters
    ----------
    method
        ``"m2"`` / ``"richardson"`` / ``"chebyshev"`` のいずれか。対応する実
        driver (``evolve_schedule_adaptive_m2`` / ``_richardson`` /
        ``_richardson_chebyshev``) を駆動する。
    c4
        ``C₄(t)`` を返す callable (例 :func:`exp_c4`)。
    t0, t1, tol_step, dt0, dt_min, dt_max, safety, growth_max, max_rejects, m
        driver にそのまま渡す PI controller パラメータ。既定は production facade
        相当 (``tol_step=1e-8`` / ``safety=0.9`` / ``growth_max=4.0`` /
        ``max_rejects=50``)。``dt_max=None`` のとき driver 既定 ``10·dt0``。

    Returns
    -------
    ControllerTrace
        accept/reject フラグ込みの controller 軌跡。
    """
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}")
    schedule = _identity_schedule(t1)
    h_p_diag = np.zeros(2, dtype=np.float64)
    psi0 = np.full(2, 1.0 / math.sqrt(2.0), dtype=np.complex128)
    attempts: list[tuple[float, float, float]] = []

    # 合成則 ``err = C₄·dt^{p+1}`` の order ``p`` を driver の推定子 order に
    # 揃える (M2 embedded = p=2, step-doubling Richardson / Chebyshev = p=4)。
    # driver 内 ``_pi_dt_next(p=...)`` と一致させないと健全 schedule でも誤差
    # 予測がずれて持続的 reject になる。
    # _rust_mod をダミーに差し替えて Chebyshev driver の fail-fast を回避する
    # (dispatch を差し替えるので実 Rust は呼ばれない)。
    # 共通 kwarg は ty (`**dict` 展開で union 値型が崩れる) を避けるため各
    # driver 呼出に明示展開する。Chebyshev は Krylov 部分空間概念がないため
    # ``m`` を受けない (``chebyshev_tol`` で K_used 動的決定)。
    if method == "m2":
        patch = {
            "_rust_mod": object(),
            "_adaptive_dispatch_m2_estimate": _make_m2_fake(c4, 2, attempts),
        }
        with _patched(patch):
            _, t_hist, dt_hist, n_rej = _kry.evolve_schedule_adaptive_m2(
                h_p_diag,
                schedule,
                psi0,
                t0,
                t1,
                m=m,
                tol_step=tol_step,
                dt0=dt0,
                dt_min=dt_min,
                dt_max=dt_max,
                safety=safety,
                growth_max=growth_max,
                max_rejects=max_rejects,
            )
    elif method == "richardson":
        patch = {
            "_rust_mod": object(),
            "_adaptive_dispatch_richardson_estimate_xyz": _make_richardson_fake(
                c4, 4, attempts
            ),
        }
        with _patched(patch):
            out = _kry.evolve_schedule_adaptive_richardson(
                h_p_diag,
                schedule,
                psi0,
                t0,
                t1,
                m=m,
                tol_step=tol_step,
                dt0=dt0,
                dt_min=dt_min,
                dt_max=dt_max,
                safety=safety,
                growth_max=growth_max,
                max_rejects=max_rejects,
            )
        t_hist, dt_hist, n_rej = out[1], out[2], out[3]
    else:  # method == "chebyshev" (上の _METHODS チェックで保証済)
        patch = {
            "_rust_mod": object(),
            "_adaptive_dispatch_richardson_estimate_chebyshev_xyz": (
                _make_chebyshev_fake(c4, 4, attempts)
            ),
        }
        with _patched(patch):
            out = _kry.evolve_schedule_adaptive_richardson_chebyshev(
                h_p_diag,
                schedule,
                psi0,
                t0,
                t1,
                h_p_min=0.0,
                h_p_max=1.0,
                tol_step=tol_step,
                dt0=dt0,
                dt_min=dt_min,
                dt_max=dt_max,
                safety=safety,
                growth_max=growth_max,
                max_rejects=max_rejects,
            )
        t_hist, dt_hist, n_rej = out[1], out[2], out[3]

    struct_attempts = _reconstruct_attempts(attempts, t_hist, dt_hist)
    return ControllerTrace(
        t_history=np.asarray(t_hist, dtype=np.float64),
        dt_history=np.asarray(dt_hist, dtype=np.float64),
        n_rejects=int(n_rej),
        attempts=struct_attempts,
    )
