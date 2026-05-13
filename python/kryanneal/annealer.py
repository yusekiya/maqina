"""高レベル公開 API (``QuantumAnnealer`` / ``AnnealingSimulator``).

``QuantumAnnealer`` は ``IsingProblem`` + ``Schedule`` を受け取り,
``run(psi0, t0, t1, *, method=..., n_steps=...)`` で時間発展を実行して
``QuantumResult`` を返す one-shot 用途のファサード. 同じ問題・スケジュール
に対して異なる初期状態 / 区間で繰り返し実行できるよう, ``psi0`` は
コンストラクタではなく ``run`` 側で受け取る.

``AnnealingSimulator`` は同じ問題に対する step-wise stateful API.
Phase 1 では ``QuantumAnnealer`` のみ提供する (``AnnealingSimulator`` は
Phase 5 で導入予定).

仕様 (Phase 1 + Phase 2 + Phase 3 + Phase 4)
--------------------------------------------
* サポート ``method``: ``"m2"`` (固定 dt M2 中点則, Phase 1), ``"trotter"``
  (固定 dt Strang 2 次 Trotter, Phase 2), ``"trotter_suzuki4"`` (固定 dt
  Suzuki S_4 4 次 Trotter, Phase 2 末), ``"cfm4"`` (固定 dt CFM4:2
  commutator-free Magnus, Phase 3), ``"cfm4_adaptive_richardson"`` (Phase 4
  C3, step-doubling Richardson + PI controller). それ以外は
  ``NotImplementedError``.
* ``save_tlist`` 引数は **API 互換性のために予約済み** だが本リリースでは
  ``None`` のみ受け付ける (非 ``None`` で ``NotImplementedError``).
  Phase 5 の ``QuantumResult.times`` / ``states`` 拡張と一緒に有効化する.
* 観測量経路 (``observables=...``) も Phase 5 で追加予定. 現状は
  ``QuantumResult.observables_history = {}`` 固定.
* adaptive 経路では ``n_steps`` を渡さない (``None`` で良い). 代わりに
  ``atol`` (PI 局所誤差閾値; driver の ``tol_step`` に map), ``dt_init``
  (初期 dt 提案; driver の ``dt0`` に map), ``dt_max`` (dt 上限; driver の
  ``dt_max`` に map) を kw-only で受ける. ``atol`` の ``None`` は driver
  既定値 ``1e-8`` を使う. ``dt_init`` / ``dt_max`` の ``None`` は **問題
  依存の auto resolution を実行** する (issue #54):

  * ``dt_init = None`` → ``dt0 = max(min(c · T^β, T), floor)``
    (既定 ``c=0.1, β=0.5, floor=1e-3``, T = ``t1 - t0``). linear schedule
    の Magnus 級数 T スケーリング (s-space scaling invariance,
    ``docs/design.md`` §5.3) から導いた保守値で PI controller の warmup
    step を T 依存に削減する.
  * ``dt_max = None`` → ``dt_max = max(min(10·dt0, 4m / ‖H‖_est), dt0)``,
    ``‖H‖_est = Σ_i |h_x_i| + max_k |H_p_diag[k]|`` の Gershgorin 上界に
    基づく Lanczos capacity 自動見積もり. 大 N で ``‖H‖ ∝ N`` が支配的に
    なる領域で PI controller が暴走しないよう守備に機能させる.

  float を明示するとどちらも一律に上書きする. 旧 ``"auto"`` リテラルは
  issue #54 で廃止 (None default = 旧 ``"auto"`` 経路と等価).
* ``m_max`` を渡すと adaptive Richardson 経路の Lanczos 部分空間次元の
  上限を ``self.m`` の代わりに ``m_max`` で上書きする (issue #43 C, 簡略
  scope). step-doubling Richardson 推定子が Lanczos breakdown も embedded
  error として検出するため, ``m_max=16`` 等まで下げて per-step matvec を
  30% 程度削減しても fail-safe で動作する (本来 PI controller が dt を
  絞ることで精度を担保). β_k < ``krylov_tol`` で Lanczos 早期打切が
  既存実装で効くため, ``m_eff ≤ m_max`` の運用. ``m_eff`` 累積統計の
  ``QuantumResult`` 露出は Rust API 拡張 (``lanczos_propagate`` の戻り値
  追加) が必要なため本フェーズでは保留 (``docs/design.md`` §5.3 参照).

実装方針: ``kryanneal.krylov.evolve_schedule_m2`` /
``evolve_schedule_trotter`` / ``evolve_schedule_trotter_suzuki4`` /
``evolve_schedule_cfm4`` (固定 dt driver) / ``evolve_schedule_adaptive_richardson``
(adaptive driver) を内部で呼ぶ薄いラッパ. 入力検証 (shape / dtype /
L2-normalize) を本クラスで集中させ, krylov 層は数値計算に専念させる.
``m`` / ``krylov_tol`` は ``"m2"`` / ``"cfm4"`` / ``"cfm4_adaptive_richardson"``
経路でのみ意味を持ち, ``"trotter"`` / ``"trotter_suzuki4"`` 経路は Lanczos を
使わないため両パラメータは無視される.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from kryanneal.krylov import (
    evolve_schedule_adaptive_richardson,
    evolve_schedule_cfm4,
    evolve_schedule_m2,
    evolve_schedule_trotter,
    evolve_schedule_trotter_suzuki4,
)
from kryanneal.problem import IsingProblem
from kryanneal.result import QuantumResult
from kryanneal.schedule import Schedule

__all__ = ["QuantumAnnealer"]


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
    スケーリング (s-space scaling invariance, ``docs/design.md`` §5.3)
    から最適 dt が T 依存で伸びるため, ``dt_init`` を T 依存に取れば
    PI controller の warmup step (default ``0.5`` → optimal dt への成長)
    を削減できる. 既定 ``c=0.1``, ``β=0.5`` は理論最適 ``β=0.75`` より
    保守的だが warmup の大部分を削減できる中庸値 (issue 本文参照).
    """
    T = float(t1) - float(t0)
    dt_auto = _AUTO_DT_INIT_C * (T**_AUTO_DT_INIT_BETA)
    return min(max(dt_auto, _AUTO_DT_INIT_FLOOR), T)


class QuantumAnnealer:
    """One-shot 時間発展ファサード.

    Parameters
    ----------
    problem
        ``IsingProblem``. ``H_p_diag`` / ``h_x`` を保持する.
    schedule
        ``Schedule``. ``coeffs_at(t)`` から ``(A(s(t)), B(s(t)))`` を得る.
    m
        Lanczos / Krylov 部分空間次元の既定値 (``run`` 内で使用).
        ``m >= 1``. 既定 ``24``.
    krylov_tol
        Lanczos の β 打切り閾値 (``β_k < tol`` で部分空間を切る).
        ``None`` (既定) のとき経路ごとに自動解決する (issue #54):

        * ``cfm4_adaptive_richardson`` (adaptive Richardson):
          ``run`` 時の ``atol`` (実効 ``tol_step``) に対し
          ``effective_krylov_tol = tol_step · _KRYLOV_TOL_ATOL_RATIO``
          (既定 ``1e-3``). atol=1e-8 default で ``1e-11``.
        * 固定 dt 経路 (``m2`` / ``cfm4``): ``atol`` を取らないため
          ``1e-12`` フォールバック (``_KRYLOV_TOL_FIXED_DEFAULT``).

        float を明示するとどの経路でも一律に上書きする (旧 ``1e-12``
        固定 default を再現したい場合は明示的に ``krylov_tol=1e-12``
        を渡す).

        **設計方針**: default (atol 連動 = 1e-11) は accuracy 優先で,
        Lanczos β_k 早期打切は実用問題サイズでは発火しない (= 旧
        ``1e-12`` 固定 default とほぼ同等の robust 挙動). 早期打切に
        よる高速化が欲しい場合は user が opt-in で発動する:
        ``atol`` を緩める (例 ``atol=1e-5`` → effective ``1e-8``) か,
        ``krylov_tol`` を明示的に緩める (例 ``krylov_tol=1e-6``).
        詳細は ``docs/design.md`` §5.3 follow-up 節 E 参照.

    Raises
    ------
    ValueError
        ``m < 1`` または ``krylov_tol`` が負値の場合.
    """

    def __init__(
        self,
        problem: IsingProblem,
        schedule: Schedule,
        *,
        m: int = 24,
        krylov_tol: float | None = None,
    ) -> None:
        if not isinstance(m, (int, np.integer)) or m < 1:
            raise ValueError(f"m must be a positive integer, got {m!r}")
        if krylov_tol is not None and krylov_tol < 0.0:
            raise ValueError(f"krylov_tol must be >= 0 or None, got {krylov_tol!r}")

        self.problem: IsingProblem = problem
        self.schedule: Schedule = schedule
        self.m: int = int(m)
        self.krylov_tol: float | None = (
            float(krylov_tol) if krylov_tol is not None else None
        )

    def run(
        self,
        psi0: np.ndarray,
        t0: float,
        t1: float,
        *,
        method: Literal[
            "m2", "trotter", "trotter_suzuki4", "cfm4", "cfm4_adaptive_richardson"
        ] = "m2",
        n_steps: int | None = None,
        atol: float | None = None,
        dt_init: float | None = None,
        dt_max: float | None = None,
        m_max: int | None = None,
        save_tlist: np.ndarray | None = None,
    ) -> QuantumResult:
        """``[t0, t1]`` 区間で時間発展を実行し ``QuantumResult`` を返す.

        Parameters
        ----------
        psi0
            shape ``(2**n,)`` complex128 の初期状態. L2-normalize 済み
            (``|‖psi0‖ - 1| < 1e-10``) であることを検証する.
        t0, t1
            積分区間. ``t1 > t0``.
        method
            プロパゲータ. ``"m2"`` (固定 dt M2 中点則, Phase 1),
            ``"trotter"`` (固定 dt Strang 2 次 Trotter, Phase 2),
            ``"trotter_suzuki4"`` (固定 dt Suzuki S_4 4 次 Trotter, Phase 2 末),
            ``"cfm4"`` (固定 dt CFM4:2 commutator-free Magnus, Phase 3),
            または ``"cfm4_adaptive_richardson"`` (Phase 4 C3,
            step-doubling Richardson + PI controller). Trotter 系経路は
            Lanczos を呼ばないため ``m`` / ``krylov_tol`` は無視される.
            ``"cfm4"`` / ``"cfm4_adaptive_richardson"`` は M2 と同じく
            Lanczos を介すので ``m`` / ``krylov_tol`` が有効.
        n_steps
            固定 dt 経路の step 数 (``n_steps >= 1``). 等間隔 ``dt =
            (t1 - t0) / n_steps`` で進める. adaptive 経路では渡さない
            (``None`` で良い; 渡しても無視される).
        atol
            adaptive 経路の局所誤差閾値. driver の ``tol_step`` に map
            される. ``None`` のときは driver 既定値 ``1e-8`` を使う.
            固定 dt 経路では無視される.
        dt_init
            adaptive 経路の初期 dt 提案. driver の ``dt0`` に map される.
            ``None`` (既定) のとき
            ``dt0 = max(min(c · T^β, T), _AUTO_DT_INIT_FLOOR)``
            (T = ``t1 - t0``, 既定 ``c=0.1, β=0.5, floor=1e-3``) で auto
            resolve し, PI controller の warmup step を T 依存で削減する
            (s-space scaling invariance, ``docs/design.md`` §5.3,
            issue #54 で None default 化, 旧 ``"auto"`` リテラル相当).
            例えば ``T=100`` で ``dt0=1.0``, ``T=1`` で ``dt0=0.1``,
            ``T=0.01`` で ``dt0=0.01`` (床値より大きいので formula 値).
            float を明示すると一律に上書きする (旧 ``"auto"`` リテラルは
            issue #54 で削除済み). 固定 dt 経路では無視される.
        dt_max
            adaptive 経路の最大 dt 上限. driver の ``dt_max`` に map される.
            ``None`` (既定) のとき Gershgorin 上界による Lanczos capacity
            自動見積もり
            ``dt_max = max(min(10·dt0, 4m / ‖H‖_est), dt0)``,
            ``‖H‖_est = Σ_i |h_x_i| + max_k |H_p_diag[k]|`` で auto
            resolve し, ``dt · ‖H‖ ≲ 4m`` の Lanczos safe 領域に強制
            クランプする (issue #54 で None default 化, 旧 ``"auto"``
            リテラル相当). 大 N で ``‖H‖ ∝ N`` が支配的になる領域で PI
            controller が暴走しないよう守備に機能する. step-doubling
            Richardson が breakdown を検出するので fail-safe で動作する
            (Lanczos 容量を僅かに超えても embedded error 経由で dt が
            縮む). float を明示すると一律に上書きする. 固定 dt 経路では
            無視される.
        m_max
            adaptive Richardson 経路の Lanczos 部分空間次元の上限。
            ``None`` (default) のときは ``self.m`` (コンストラクタ既定 24)
            をそのまま使う。整数を指定すると ``self.m`` を上書きして
            driver の Lanczos 部分空間次元として用いる (issue #43 C,
            簡略 scope)。step-doubling Richardson 推定子が Lanczos
            breakdown も embedded error として検出する fail-safe を
            活かし, ``m_max=16`` 等で per-step matvec を 30% 程度削減
            する運用を許容する (Richardson が破綻を検知すれば PI
            controller が dt を絞り精度を維持)。``β_k < krylov_tol``
            の早期打切が既存実装で効くため, 実効次元は ``m_eff ≤
            m_max`` になる。固定 dt 経路では無視される。
            ``m_eff`` 累積統計の ``QuantumResult`` 露出は Rust API 拡張
            (``lanczos_propagate`` の戻り値追加 + PyO3 plumbing) が
            必要なため本フェーズでは保留 (``docs/design.md`` §5.3 参照)。
        save_tlist
            観測時刻列 (Phase 5 で実装予定). 現状は ``None`` 以外を
            渡すと ``NotImplementedError``.

        Returns
        -------
        QuantumResult
            ``psi_final`` / ``n_steps`` / ``n_matvec`` / ``success`` /
            ``method`` / ``n_steps_actual`` を持つ result.
            ``t_history = None``, ``observables_history = {}`` (Phase 5 で
            観測量を一緒に格納予定). ``n_matvec`` は経路ごとに以下:

            * ``"m2"``: ``n_steps × m`` (Lanczos の matvec 見積もり).
            * ``"trotter"``: ``n_steps × (N + 1)`` (phase pass 1 + bit-flip
              pass N の dim-walk 見積もり; ``docs/design.md`` §4.4 参照).
            * ``"trotter_suzuki4"``: ``n_steps × 5 × (N + 1)`` (5 sub-step
              × Strang per-step コスト).
            * ``"cfm4"``: ``n_steps × 2m`` (CFM4:2 は 1 step あたり Lanczos
              を 2 回呼ぶため M2 の 2 倍).
            * ``"cfm4_adaptive_richardson"``: ``n_steps_actual × 6m``
              (full CFM4:2 ``2m`` + half×2 CFM4:2 ``4m`` = ``6m``,
              ``docs/design.md`` §5.3).

            ``n_steps`` は固定 dt 経路では要求 step 数, adaptive 経路では
            実 step 数 (``n_steps_actual`` と同値) を返す.

        Raises
        ------
        ValueError
            入力検証失敗 (``psi0`` の shape / dtype / 非正規化,
            固定 dt 経路で ``n_steps`` 不指定 / ``n_steps < 1``, ``t1 <= t0``).
        NotImplementedError
            ``method`` がサポート対象外, または ``save_tlist`` が
            ``None`` でない場合.
        RuntimeError
            adaptive 経路で ``max_rejects`` 連続超過したとき (driver から
            伝播).
        """
        valid_methods = (
            "m2",
            "trotter",
            "trotter_suzuki4",
            "cfm4",
            "cfm4_adaptive_richardson",
        )
        if method not in valid_methods:
            raise NotImplementedError(
                f"method={method!r} is not supported; valid methods are {valid_methods!r}."
            )
        if save_tlist is not None:
            raise NotImplementedError("save_tlist is reserved for Phase 5; pass None.")

        psi0_arr = self._validate_psi0(psi0)

        if method == "cfm4_adaptive_richardson":
            # NOTE: 既定値は driver (``evolve_schedule_adaptive_richardson``)
            # 側と一致させること. driver 側を変えたら本ファイルも追従する.
            tol_step = float(atol) if atol is not None else 1e-8
            # issue #54: ``krylov_tol = None`` のとき adaptive Richardson 経路は
            # ``tol_step · _KRYLOV_TOL_ATOL_RATIO`` に解決する.
            if self.krylov_tol is not None:
                effective_krylov_tol = self.krylov_tol
            else:
                effective_krylov_tol = tol_step * _KRYLOV_TOL_ATOL_RATIO
            # issue #54: ``dt_init = None`` で旧 ``"auto"`` 相当の T-dep
            # auto resolution. float 明示時はそのまま渡す.
            if dt_init is None:
                dt0 = _resolve_dt_init_auto(t0, t1)
            else:
                dt0 = float(dt_init)
            # issue #54: ``dt_max = None`` で旧 ``"auto"`` 相当の Gershgorin
            # cap auto resolution. float 明示時はそのまま渡す.
            if dt_max is None:
                dt_max_resolved: float = _resolve_dt_max_auto(self.problem, self.m, dt0)
            else:
                dt_max_resolved = float(dt_max)
            # m_max は adaptive Richardson 経路の Lanczos 部分空間上限を
            # ``self.m`` 既定値から上書きする. None なら self.m を流用.
            # 整数を渡したときは正の整数であることだけ検証する.
            if m_max is None:
                m_eff_param = self.m
            else:
                if not isinstance(m_max, (int, np.integer)) or m_max < 1:
                    raise ValueError(
                        f"m_max must be a positive integer or None, got {m_max!r}"
                    )
                m_eff_param = int(m_max)
            # C3/C4 (issue #52 A): driver は 5-tuple
            # `(psi, t_hist, dt_hist, n_rejects, m_eff_hist)` を返す.
            (
                psi_final,
                _t_history,
                dt_history,
                _n_rejects,
                m_eff_history,
            ) = evolve_schedule_adaptive_richardson(
                h_x=self.problem.h_x,
                h_p_diag=self.problem.H_p_diag,
                schedule=self.schedule,
                psi0=psi0_arr,
                t0=t0,
                t1=t1,
                m=m_eff_param,
                krylov_tol=effective_krylov_tol,
                tol_step=tol_step,
                dt0=dt0,
                dt_max=dt_max_resolved,
            )
            n_steps_actual = int(dt_history.shape[0])
            # C4 (issue #52 A): per-step `m_eff_sum` (= 6 Lanczos call の合計)
            # の集計を `QuantumResult.m_eff_stats` に格納する. accept された
            # step 数 == m_eff_history.shape[0] == n_steps_actual (driver 仕様).
            # n_steps_actual == 0 の縮退ケース (t1 == t0 を許さない driver
            # 入力検証で実用上は通らないが念のため) は空 history なので
            # stats を空 dict ではなく None として扱う.
            if m_eff_history.size > 0:
                m_eff_stats: dict[str, int | float] | None = {
                    "total": int(np.sum(m_eff_history)),
                    "mean": float(np.mean(m_eff_history)),
                    "median": float(np.median(m_eff_history)),
                    "min": int(np.min(m_eff_history)),
                    "max": int(np.max(m_eff_history)),
                }
                # 実 matvec 数を m_eff_sum の累積で正確に出す (旧推定
                # `n_steps_actual · 6m` は upper bound; 早期打切で乖離する).
                n_matvec = int(np.sum(m_eff_history))
            else:
                m_eff_stats = None
                n_matvec = n_steps_actual * 6 * m_eff_param
            return QuantumResult(
                psi_final=psi_final,
                t_history=None,
                observables_history={},
                n_steps=n_steps_actual,
                n_matvec=int(n_matvec),
                success=True,
                method=method,
                n_steps_actual=n_steps_actual,
                m_eff_stats=m_eff_stats,
            )

        # 固定 dt 経路は n_steps が必須.
        if n_steps is None:
            raise ValueError(
                f"n_steps is required for fixed-dt method={method!r}; "
                "pass a positive integer."
            )
        n_steps_int = int(n_steps)

        # issue #54: 固定 dt 経路は ``atol`` を取らないため None → 旧 default
        # ``1e-12`` に static fallback (adaptive 経路の atol 連動とは別扱い).
        if self.krylov_tol is not None:
            effective_krylov_tol = self.krylov_tol
        else:
            effective_krylov_tol = _KRYLOV_TOL_FIXED_DEFAULT

        if method == "m2":
            psi_final, n_matvec = evolve_schedule_m2(
                h_x=self.problem.h_x,
                h_p_diag=self.problem.H_p_diag,
                schedule=self.schedule,
                psi0=psi0_arr,
                t0=t0,
                t1=t1,
                n_steps=n_steps_int,
                m=self.m,
                krylov_tol=effective_krylov_tol,
            )
        elif method == "trotter":
            psi_final, n_matvec = evolve_schedule_trotter(
                h_x=self.problem.h_x,
                h_p_diag=self.problem.H_p_diag,
                schedule=self.schedule,
                psi0=psi0_arr,
                t0=t0,
                t1=t1,
                n_steps=n_steps_int,
            )
        elif method == "trotter_suzuki4":
            psi_final, n_matvec = evolve_schedule_trotter_suzuki4(
                h_x=self.problem.h_x,
                h_p_diag=self.problem.H_p_diag,
                schedule=self.schedule,
                psi0=psi0_arr,
                t0=t0,
                t1=t1,
                n_steps=n_steps_int,
            )
        else:  # method == "cfm4"
            psi_final, n_matvec = evolve_schedule_cfm4(
                h_x=self.problem.h_x,
                h_p_diag=self.problem.H_p_diag,
                schedule=self.schedule,
                psi0=psi0_arr,
                t0=t0,
                t1=t1,
                n_steps=n_steps_int,
                m=self.m,
                krylov_tol=effective_krylov_tol,
            )
        return QuantumResult(
            psi_final=psi_final,
            t_history=None,
            observables_history={},
            n_steps=n_steps_int,
            n_matvec=int(n_matvec),
            success=True,
            method=method,
            n_steps_actual=n_steps_int,
        )

    def _validate_psi0(self, psi0: np.ndarray) -> np.ndarray:
        """``psi0`` の shape / dtype / L2-norm を検証して C-contiguous で返す.

        ``dim = 2**problem.n`` と整合し, complex128 / C-contiguous で,
        L2 ノルムが 1 から ``_PSI_NORM_TOL`` 以内であることを要求する.
        非 complex128 は明示的にエラーにする (silent cast はしない).
        """
        if not isinstance(psi0, np.ndarray):
            raise ValueError(f"psi0 must be a numpy.ndarray, got {type(psi0).__name__}")
        expected_dim = self.problem.dim
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
