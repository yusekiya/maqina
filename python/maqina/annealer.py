"""高レベル公開 API (``QuantumAnnealer`` / ``AnnealingSimulator``).

``QuantumAnnealer`` は ``IsingProblem`` + ``Schedule`` を受け取り,
``run(psi0, t0, t1, *, method=..., n_steps=...)`` で時間発展を実行して
``QuantumResult`` を返す one-shot 用途のファサード. 同じ問題・スケジュール
に対して異なる初期状態 / 区間で繰り返し実行できるよう, ``psi0`` は
コンストラクタではなく ``run`` 側で受け取る.

``AnnealingSimulator`` (``maqina.simulator``) は同じ問題に対する
step-wise stateful API. 中間時刻まで進めて状態を取り出し, ``Observable``
で測定して続きを発展させる workflow 用. ``QuantumAnnealer.create_simulator``
で生成するのが簡便 (現 instance の ``problem`` / ``schedule`` / ``m`` /
``propagator_tol`` を引き継ぐ).

仕様 (Phase 1 + Phase 2 + Phase 3 + Phase 4 + Phase B)
------------------------------------------------------
* サポート ``method``: ``"m2"`` (固定 dt M2 中点則, Phase 1), ``"trotter"``
  (固定 dt Strang 2 次 Trotter, Phase 2), ``"trotter_suzuki4"`` (固定 dt
  Suzuki S_4 4 次 Trotter, Phase 2 末), ``"cfm4"`` (固定 dt CFM4:2
  commutator-free Magnus, Phase 3), ``"cfm4_adaptive_richardson_krylov"`` (Phase 4
  C3, step-doubling Richardson + PI controller, Lanczos 短時間プロパゲータ),
  ``"cfm4_adaptive_richardson_chebyshev"`` (Phase B / issue #122, 同じ
  Richardson + PI controller で短時間プロパゲータを Chebyshev 3 項漸化に
  差し替えた variant; **issue #124 で `run` の default に採用**, N=18 で
  Lanczos 比 5.49× wall 高速). それ以外は ``NotImplementedError``.
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
    ``docs/design/05-3-propagator.md`` §5.3) から導いた保守値で PI controller の warmup
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
  絞ることで精度を担保). a posteriori 推定子 ``est < propagator_tol``
  (Lanczos) で早期打切が効くため, ``m_eff ≤ m_max`` の運用. ``m_eff`` 累積統計の
  ``QuantumResult`` 露出は Rust API 拡張 (``lanczos_propagate`` の戻り値
  追加) が必要なため本フェーズでは保留 (``docs/design/05-3-propagator.md`` §5.3 参照).

実装方針: ``maqina.krylov.evolve_schedule_m2`` /
``evolve_schedule_trotter`` / ``evolve_schedule_trotter_suzuki4`` /
``evolve_schedule_cfm4`` (固定 dt driver) / ``evolve_schedule_adaptive_richardson``
(adaptive driver) を内部で呼ぶ薄いラッパ. 入力検証 (shape / dtype /
L2-normalize) を本クラスで集中させ, krylov 層は数値計算に専念させる.
``m`` / ``propagator_tol`` は ``"m2"`` / ``"cfm4"`` / ``"cfm4_adaptive_richardson_krylov"``
経路でのみ意味を持ち, ``"trotter"`` / ``"trotter_suzuki4"`` 経路は Lanczos を
使わないため両パラメータは無視される. ``"cfm4_adaptive_richardson_chebyshev"``
経路では ``m`` (Lanczos 部分空間次元) は使われず, ``propagator_tol`` は
Chebyshev 切り捨て次数 ``K_used`` を決める許容誤差として機能する
(issue #135 で ``krylov_tol`` から rename, semantic 統一).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import numpy as np

from maqina._helpers import _AUTO_DT_INIT_BETA as _AUTO_DT_INIT_BETA
from maqina._helpers import _AUTO_DT_INIT_C as _AUTO_DT_INIT_C
from maqina._helpers import _AUTO_DT_INIT_FLOOR as _AUTO_DT_INIT_FLOOR
from maqina._helpers import _KRYLOV_TOL_ATOL_RATIO as _KRYLOV_TOL_ATOL_RATIO
from maqina._helpers import _KRYLOV_TOL_FIXED_DEFAULT as _KRYLOV_TOL_FIXED_DEFAULT
from maqina._helpers import _LANCZOS_DT_NORM_COEFF as _LANCZOS_DT_NORM_COEFF
from maqina._helpers import _PSI_NORM_TOL as _PSI_NORM_TOL
from maqina._helpers import (
    _gershgorin_norm_upper_bound as _gershgorin_norm_upper_bound,
)
from maqina._helpers import _resolve_dt_init_auto as _resolve_dt_init_auto
from maqina._helpers import _resolve_dt_max_auto as _resolve_dt_max_auto
from maqina._helpers import _validate_psi0 as _validate_psi0
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
from maqina.result import QuantumResult
from maqina.schedule import Schedule

if TYPE_CHECKING:
    from maqina.simulator import AnnealingSimulator

__all__ = ["QuantumAnnealer"]


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
    propagator_tol
        **短時間プロパゲータ U(dt) の per-step 許容誤差** (issue #135 で
        ``krylov_tol`` から rename, semantic 統一). method ごとに作用先が
        異なるが, いずれも outer の ``atol`` (PI controller の per-step
        累積誤差 bound) より内側の inner tol として機能する.

        * Lanczos 経路 (``m2`` / ``cfm4`` / ``cfm4_adaptive_richardson_krylov``):
          Krylov 近似の許容誤差 (issue #98 / Phase 8). 各 Lanczos iter ``k``
          で Hochbruck-Lubich 1997 の a posteriori 推定子
          ``est = β_k · |c_last| · |dt| / (k+1)`` を計算し,
          ``est < propagator_tol`` で部分空間を切る (``‖ψ‖ = 1`` の Lanczos
          内部規約). 旧仕様 (``β_k < tol`` の β 単体閾値) ではない.
        * Chebyshev 経路 (``cfm4_adaptive_richardson_chebyshev``): Chebyshev
          切り捨て次数 ``K_used`` を決める許容誤差. Bessel ``|J_K(R·dt)|``
          の指数的減衰で ``K_used`` を動的決定する (詳細は
          ``docs/design/05-3-propagator.md`` "Chebyshev variant" 節).

        ``None`` (既定) のとき経路ごとに自動解決する:

        * ``cfm4_adaptive_richardson_krylov`` (adaptive Richardson, Lanczos):
          ``run`` 時の ``atol`` (実効 ``tol_step``) に対し
          ``effective_propagator_tol = tol_step · _KRYLOV_TOL_ATOL_RATIO``
          (既定 ``1e-3``). atol=1e-8 default で ``1e-11``. Lanczos a
          posteriori 早期打切は atol scaling で連動するのが望ましいため
          auto-coupling を維持.
        * ``cfm4_adaptive_richardson_chebyshev`` (adaptive Richardson,
          Chebyshev variant): **固定値 ``_KRYLOV_TOL_FIXED_DEFAULT``
          (= ``1e-12``)** (issue #135 で auto-coupling から変更). Chebyshev
          は ``K_used ~ R·dt + log(1/propagator_tol)`` の対数依存で
          propagator_tol を atol に連動させる動機が弱く, auto-coupling だと
          atol↓ で K_used が単調に増えず machine precision 到達後は逆に
          round-off accumulation で精度が悪化する非単調性が発生する.
          固定 ``1e-12`` で atol-vs-infidelity の monotonicity を確保し
          Pareto curve の解釈性を上げる.
        * 固定 dt 経路 (``m2`` / ``cfm4``): ``atol`` を取らないため
          ``1e-12`` フォールバック (``_KRYLOV_TOL_FIXED_DEFAULT``).

        float を明示するとどの経路でも一律に上書きする.

        **設計方針 (Lanczos, Phase 8)**: a posteriori 判定式
        ``β · |c| · |dt| / m`` は TFIM 等の実用問題サイズでも ``|c|`` の
        幾何級数減衰により早期打切が実際に発火する (Phase 7 までの β 単体
        閾値では発火しなかった). default (atol 連動 = 1e-11) で **m_eff <
        m_max が自然に起こる** 設定となり, Pareto win に直結する. 旧
        ``1e-12`` 固定挙動 (= m_eff = m_max 固定) を再現したいときは
        ``propagator_tol=1e-16`` 等の極小値を渡す (推奨されない).

        詳細は ``docs/design/05-2-lanczos.md`` の "a posteriori 早期打切"
        節 (Lanczos) と ``docs/design/05-3-propagator.md`` "Chebyshev
        variant" 節 (Chebyshev) を参照.

    Raises
    ------
    ValueError
        ``m < 1`` または ``propagator_tol`` が負値の場合.
    """

    def __init__(
        self,
        problem: IsingProblem,
        schedule: Schedule,
        *,
        m: int = 24,
        propagator_tol: float | None = None,
    ) -> None:
        if not isinstance(m, (int, np.integer)) or m < 1:
            raise ValueError(f"m must be a positive integer, got {m!r}")
        if propagator_tol is not None and propagator_tol < 0.0:
            raise ValueError(
                f"propagator_tol must be >= 0 or None, got {propagator_tol!r}"
            )

        self.problem: IsingProblem = problem
        self.schedule: Schedule = schedule
        self.m: int = int(m)
        self.propagator_tol: float | None = (
            float(propagator_tol) if propagator_tol is not None else None
        )

    def run(
        self,
        psi0: np.ndarray,
        t0: float,
        t1: float,
        *,
        method: Literal[
            "m2",
            "trotter",
            "trotter_suzuki4",
            "cfm4",
            "cfm4_adaptive_richardson_krylov",
            "cfm4_adaptive_richardson_chebyshev",
        ] = "cfm4_adaptive_richardson_chebyshev",
        n_steps: int | None = None,
        atol: float | None = None,
        dt_init: float | None = None,
        dt_max: float | None = None,
        m_max: int | None = None,
        observables: dict[str, Observable] | None = None,
        save_tlist: np.ndarray | None = None,
        store_states: bool = False,
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
            ``"cfm4_adaptive_richardson_krylov"`` (Phase 4 C3, step-doubling
            Richardson + PI controller, Lanczos 短時間プロパゲータ), または
            ``"cfm4_adaptive_richardson_chebyshev"`` (**default**, Phase B /
            issue #122, 同じ Richardson + PI controller で短時間プロパゲータ
            を Chebyshev 3 項漸化に差し替えた variant). default を Chebyshev に
            しているのは issue #124 の判断による (N=18 で Lanczos 比 5.49× wall
            高速の Pareto win). Trotter 系経路は Lanczos を呼ばないため
            ``m`` / ``propagator_tol`` は無視される. ``"cfm4"`` /
            ``"cfm4_adaptive_richardson_krylov"`` は M2 と同じく Lanczos を
            介すので ``m`` / ``propagator_tol`` が有効. Chebyshev variant では
            ``m`` は使われず, ``propagator_tol`` が Chebyshev 切り捨て次数
            ``K_used`` を決める許容誤差として機能する.
        n_steps
            固定 dt 経路の step 数 (``n_steps >= 1``). 等間隔 ``dt =
            (t1 - t0) / n_steps`` で進める. adaptive 経路では渡さない
            (``None`` で良い; 渡しても無視される).
        atol
            adaptive 経路の局所誤差閾値. driver の ``tol_step`` に map
            される. ``None`` のときは driver 既定値 ``1e-8`` を使う.
            固定 dt 経路では無視される.

            **default ``1e-8`` は保守寄りの選択**. 量子ダイナミクス標準
            テストの fidelity ``1 - 1e-6`` 要件を安全マージン付きで満たす
            ことを優先した値. 実用上 ``atol=1e-6`` / ``1e-5`` でも多くの
            応用で許容範囲で, その場合 PI step 数が減るうえ
            ``propagator_tol = None`` ならば Lanczos a posteriori 早期打切
            (Krylov: ``atol · 1e-3``) も自動的に緩んで二重に高速化される
            (Chebyshev variant では ``propagator_tol`` は固定 1e-12 default
            なので連動しない, issue #135). 詳細は ``docs/design/05-3-propagator.md``
            §5.3 PI controller defaults 表のノート.

            **Note (Chebyshev variant の atol 振舞い, issue #124 / #135)**:
            ``method="cfm4_adaptive_richardson_chebyshev"`` では ``atol`` は
            実際の精度ではなく **upper bound** として機能する. Chebyshev は
            切り捨て次数 ``K_used`` を ``propagator_tol`` (default
            ``_KRYLOV_TOL_FIXED_DEFAULT = 1e-12`` 固定, issue #135) から
            動的決定するため, ``tol_step`` より遥かに小さい per-stage 誤差
            を生む. その結果 PI controller が見る誤差は Magnus 4 次誤差
            (``O(dt^5)``) のみとなり, dt は ``atol`` 制約下で大きく取れる
            一方で実際の精度は machine precision 近くを維持する (例:
            ``atol=1e-3`` 設定で n=10 の infidelity ``< 1e-16``). これは
            "feature" であり (atol で要求した精度を上回ることはあっても下回る
            ことはない), 速度を取りたい場合は ``atol`` を大きくして PI step
            数を減らすのが正しい使い方. 詳細は
            ``docs/design/05-3-propagator.md`` "Chebyshev variant" 節.
        dt_init
            adaptive 経路の初期 dt 提案. driver の ``dt0`` に map される.
            ``None`` (既定) のとき
            ``dt0 = max(min(c · T^β, T), _AUTO_DT_INIT_FLOOR)``
            (T = ``t1 - t0``, 既定 ``c=0.1, β=0.5, floor=1e-3``) で auto
            resolve し, PI controller の warmup step を T 依存で削減する
            (s-space scaling invariance, ``docs/design/05-3-propagator.md`` §5.3,
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
            controller が dt を絞り精度を維持)。a posteriori 推定子
            ``est < propagator_tol`` (Lanczos) の早期打切が既存実装で効くため,
            実効次元は ``m_eff ≤ m_max`` になる。固定 dt 経路では無視される。
            ``m_eff`` 累積統計の ``QuantumResult`` 露出は Rust API 拡張
            (``lanczos_propagate`` の戻り値追加 + PyO3 plumbing) が
            必要なため本フェーズでは保留 (``docs/design/05-3-propagator.md`` §5.3 参照)。
        observables
            Phase 5 (issue #47) で有効化. ``{name: Observable}`` dict もしくは
            ``None``. ``save_tlist`` 非 None かつ非空 dict のとき, 各
            ``save_tlist[i]`` 時刻で ``obs.expectation(psi)`` を評価して
            ``QuantumResult.observables_history[name]`` に shape
            ``(len(save_tlist),)`` の時系列を格納する. ``save_tlist=None``
            (デフォルト) のときは silent 無視 (最節約モード).
        save_tlist
            Phase 5 (issue #47) で有効化. shape ``(K,)`` float64 の観測時刻列.
            monotonic increasing, ``[t0, t1]`` の範囲. 非 None のとき
            時間発展に該当時刻を厳密に踏ませ (固定 dt: step boundary に
            merge, adaptive: dt クランプ), ``QuantumResult.times`` に複製を
            格納する. ``None`` (デフォルト, 最節約モード) で snapshot 記録
            なし (``times = states = None``, ``observables_history = {}``).
        store_states
            Phase 5 (issue #47) で有効化. ``True`` かつ ``save_tlist`` 非 None
            のとき, snapshot 時刻に ψ を保存し ``QuantumResult.states`` に
            shape ``(K, 2**n)`` complex128 として返す. ``save_tlist=None``
            または ``False`` で ``states = None``.

        Returns
        -------
        QuantumResult
            ``psi_final`` / ``n_steps`` / ``n_matvec`` / ``success`` /
            ``method`` / ``n_steps_actual`` / ``probabilities`` を常に持ち,
            ``save_tlist`` 経路でのみ ``times`` / ``states`` /
            ``observables_history`` が非 None / 非空になる. ``probabilities``
            は ``|psi_final|^2`` を eager 計算した shape ``(2**n,)`` float64
            (どの経路でも常に返る). ``n_matvec`` は経路ごとに以下:

            * ``"m2"``: ``n_steps × m`` (Lanczos の matvec 見積もり).
            * ``"trotter"``: ``n_steps × (N + 1)`` (phase pass 1 + bit-flip
              pass N の dim-walk 見積もり; ``docs/design/04-python-api.md`` §4.4 参照).
            * ``"trotter_suzuki4"``: ``n_steps × 5 × (N + 1)`` (5 sub-step
              × Strang per-step コスト).
            * ``"cfm4"``: ``n_steps × 2m`` (CFM4:2 は 1 step あたり Lanczos
              を 2 回呼ぶため M2 の 2 倍).
            * ``"cfm4_adaptive_richardson_krylov"``: ``n_steps_actual × 6m``
              (full CFM4:2 ``2m`` + half×2 CFM4:2 ``4m`` = ``6m``,
              ``docs/design/05-3-propagator.md`` §5.3).

            ``n_steps`` は固定 dt 経路では要求 step 数, adaptive 経路では
            実 step 数 (``n_steps_actual`` と同値) を返す.

        Raises
        ------
        ValueError
            入力検証失敗 (``psi0`` の shape / dtype / 非正規化,
            固定 dt 経路で ``n_steps`` 不指定 / ``n_steps < 1``,
            ``t1 <= t0``, ``observables`` が dict[str, Observable] でない,
            ``save_tlist`` が monotonic float64 で ``[t0, t1]`` 範囲に
            収まらない).
        NotImplementedError
            ``method`` がサポート対象外の場合.
        RuntimeError
            adaptive 経路で ``max_rejects`` 連続超過したとき (driver から
            伝播).
        """
        valid_methods = (
            "m2",
            "trotter",
            "trotter_suzuki4",
            "cfm4",
            "cfm4_adaptive_richardson_krylov",
            "cfm4_adaptive_richardson_chebyshev",
        )
        if method not in valid_methods:
            raise NotImplementedError(
                f"method={method!r} is not supported; valid methods are {valid_methods!r}."
            )

        # Phase 5 (issue #47): save_tlist / observables / store_states の入力検証.
        # save_tlist=None のとき observables / store_states は無効化する
        # ことが新仕様 (最節約モード). ただし「指定したのに無視」は debug 罠
        # なので明示的に ValueError で弾く (silent 無視はしない).
        save_tlist_arr = self._validate_save_tlist(save_tlist, t0, t1)
        observables_validated = self._validate_observables(observables)
        if save_tlist_arr is None:
            if observables_validated is not None:
                raise ValueError(
                    "observables requires save_tlist to be non-None "
                    "(save_tlist=None is the no-recording mode)."
                )
            if store_states:
                raise ValueError(
                    "store_states=True requires save_tlist to be non-None "
                    "(save_tlist=None is the no-recording mode)."
                )

        psi0_arr = _validate_psi0(self.problem, psi0)

        if method == "cfm4_adaptive_richardson_krylov":
            # NOTE: 既定値は driver (``evolve_schedule_adaptive_richardson``)
            # 側と一致させること. driver 側を変えたら本ファイルも追従する.
            #
            # atol default ``1e-8`` は **保守寄りの選択** (詳細は
            # ``docs/design/05-3-propagator.md`` §5.3 PI controller defaults 表のノート).
            # 量子ダイナミクス標準テストの fidelity ``1 - 1e-6`` 要件を
            # 単一の安全 default で安定して満たすことを優先しており,
            # 実用上は ``atol=1e-6`` / ``1e-5`` も十分許容範囲. user が
            # 速度を取りたい場合は ``atol`` を緩めること (PI step 数が
            # 減り, ``propagator_tol`` 連動も自動緩和される).
            tol_step = float(atol) if atol is not None else 1e-8
            # issue #54: ``propagator_tol = None`` のとき adaptive Richardson
            # Lanczos 経路は ``tol_step · _KRYLOV_TOL_ATOL_RATIO`` に解決する.
            # Chebyshev variant の default は固定 1e-12 (issue #135).
            if self.propagator_tol is not None:
                effective_propagator_tol = self.propagator_tol
            else:
                effective_propagator_tol = tol_step * _KRYLOV_TOL_ATOL_RATIO
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
            # issue #93 (Phase 7) + Phase 5 (issue #47): driver は 10-tuple
            # ``(psi, t_hist, dt_hist, n_rejects, m_eff_hist, beta_m_hist,
            #    err_lanczos_hist, err_magnus_hist, n_krylov_insufficient,
            #    snapshot)`` を返す.
            (
                psi_final,
                _t_history,
                dt_history,
                _n_rejects,
                m_eff_history,
                beta_m_history,
                _err_lanczos_history,
                _err_magnus_history,
                n_krylov_insufficient,
                snapshot,
            ) = evolve_schedule_adaptive_richardson(
                h_x=self.problem.h_x,
                h_p_diag=self.problem.H_p_diag,
                schedule=self.schedule,
                psi0=psi0_arr,
                t0=t0,
                t1=t1,
                m=m_eff_param,
                krylov_tol=effective_propagator_tol,
                tol_step=tol_step,
                dt0=dt0,
                dt_max=dt_max_resolved,
                observables=observables_validated,
                save_tlist=save_tlist_arr,
                store_states=store_states,
            )
            n_steps_actual = int(dt_history.shape[0])
            # C4 (issue #52 A): per-step `m_eff_sum` (= 6 Lanczos call の合計)
            # の集計を `QuantumResult.m_eff_stats` に格納する.
            if m_eff_history.size > 0:
                m_eff_stats: dict[str, int | float] | None = {
                    "total": int(np.sum(m_eff_history)),
                    "mean": float(np.mean(m_eff_history)),
                    "median": float(np.median(m_eff_history)),
                    "min": int(np.min(m_eff_history)),
                    "max": int(np.max(m_eff_history)),
                }
                # 実 matvec 数を m_eff_sum の累積で正確に出す.
                n_matvec = int(np.sum(m_eff_history))
            else:
                m_eff_stats = None
                n_matvec = n_steps_actual * 6 * m_eff_param

            # issue #93 (Phase 7): β_m 統計値も同様に集計. driver 内部で代表
            # β_m_eff を保存しており, ここでは accept された step 全体の分布を
            # mean / median / min / max / p10 / p90 にまとめる.
            beta_m_stats: dict[str, float] | None
            if beta_m_history.size > 0:
                beta_m_stats = {
                    "mean": float(np.mean(beta_m_history)),
                    "median": float(np.median(beta_m_history)),
                    "min": float(np.min(beta_m_history)),
                    "max": float(np.max(beta_m_history)),
                    "p10": float(np.percentile(beta_m_history, 10.0)),
                    "p90": float(np.percentile(beta_m_history, 90.0)),
                }
            else:
                beta_m_stats = None
            return self._build_result(
                psi_final=psi_final,
                snapshot=snapshot,
                method=method,
                n_steps=n_steps_actual,
                n_matvec=int(n_matvec),
                n_steps_actual=n_steps_actual,
                m_eff_stats=m_eff_stats,
                beta_m_stats=beta_m_stats,
                n_krylov_insufficient=int(n_krylov_insufficient),
            )

        if method == "cfm4_adaptive_richardson_chebyshev":
            # issue #122 (Phase B): Chebyshev variant. Lanczos 経路 (上) と
            # 同じ atol / dt0 / dt_max 解決ロジックを踏襲. `m` (Lanczos
            # 部分空間次元) は Chebyshev で使われないが ``_resolve_dt_max_auto``
            # は依然 Lanczos capacity (``4m / ‖H‖``) を返す保守側上限として
            # 流用する (Chebyshev は breakdown しないので大きめの dt_max でも
            # 安全だが, 初期実装では Lanczos と同じ上限で挙動を観察する).
            tol_step = float(atol) if atol is not None else 1e-8
            # issue #135: Chebyshev variant の propagator_tol default を
            # auto-coupling (``tol_step · _KRYLOV_TOL_ATOL_RATIO``) から
            # 固定値 ``_KRYLOV_TOL_FIXED_DEFAULT`` (= 1e-12) に変更. atol↓
            # で K_used が単調に増えず, machine precision 到達後は逆に
            # round-off accumulation で精度が悪化する非単調性を解消するため.
            # Krylov variant では auto-coupling 維持 (Lanczos a posteriori
            # 早期打切が atol scaling で有効).
            if self.propagator_tol is not None:
                effective_propagator_tol = self.propagator_tol
            else:
                effective_propagator_tol = _KRYLOV_TOL_FIXED_DEFAULT
            if dt_init is None:
                dt0 = _resolve_dt_init_auto(t0, t1)
            else:
                dt0 = float(dt_init)
            if dt_max is None:
                # ``m`` は ``_resolve_dt_max_auto`` の Lanczos capacity 引数
                # としてのみ使う. Chebyshev driver 自体は ``m`` を取らない.
                dt_max_resolved_cheb: float = _resolve_dt_max_auto(
                    self.problem, self.m, dt0
                )
            else:
                dt_max_resolved_cheb = float(dt_max)
            # ``m_max`` は Chebyshev で意味を持たないが, 互換のため受け取り
            # 検証のみ通す (silent 無視は debug 罠なので, 渡されたら
            # ValueError で弾く方針).
            if m_max is not None:
                raise ValueError(
                    "m_max is not supported for method="
                    "'cfm4_adaptive_richardson_chebyshev' "
                    "(Chebyshev uses dynamic K_used, not Krylov subspace dimension)."
                )
            (
                psi_final,
                _t_history,
                dt_history,
                _n_rejects,
                k_used_history,
                _beta_m_history,
                _err_cheb_history,
                _err_magnus_history,
                n_cheb_insufficient,
                snapshot,
            ) = evolve_schedule_adaptive_richardson_chebyshev(
                h_x=self.problem.h_x,
                h_p_diag=self.problem.H_p_diag,
                schedule=self.schedule,
                psi0=psi0_arr,
                t0=t0,
                t1=t1,
                chebyshev_tol=effective_propagator_tol,
                tol_step=tol_step,
                dt0=dt0,
                dt_max=dt_max_resolved_cheb,
                observables=observables_validated,
                save_tlist=save_tlist_arr,
                store_states=store_states,
            )
            n_steps_actual = int(dt_history.shape[0])
            # Chebyshev では K_used を ``m_eff_stats`` と同じスロットに格納
            # (driver 戻り値 shape 互換を保ち, ``QuantumResult.m_eff_stats``
            # のキー意味を「per-step propagator 評価コスト統計」に拡張).
            # 別途 ``QuantumResult.method`` で Lanczos / Chebyshev を判別可.
            if k_used_history.size > 0:
                m_eff_stats_cheb: dict[str, int | float] | None = {
                    "total": int(np.sum(k_used_history)),
                    "mean": float(np.mean(k_used_history)),
                    "median": float(np.median(k_used_history)),
                    "min": int(np.min(k_used_history)),
                    "max": int(np.max(k_used_history)),
                }
                # Chebyshev の matvec 推定: K_used 合計 (= 6 chebyshev_propagate
                # 呼出の K_used 合計, 各 K_used が matvec 数とほぼ等しい
                # — phi_1 で 1 matvec, k=2..K で各 1 matvec).
                n_matvec_cheb = int(np.sum(k_used_history))
            else:
                m_eff_stats_cheb = None
                n_matvec_cheb = 0
            return self._build_result(
                psi_final=psi_final,
                snapshot=snapshot,
                method=method,
                n_steps=n_steps_actual,
                n_matvec=int(n_matvec_cheb),
                n_steps_actual=n_steps_actual,
                m_eff_stats=m_eff_stats_cheb,
                beta_m_stats=None,
                n_krylov_insufficient=int(n_cheb_insufficient),
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
        if self.propagator_tol is not None:
            effective_propagator_tol = self.propagator_tol
        else:
            effective_propagator_tol = _KRYLOV_TOL_FIXED_DEFAULT

        # Phase 5 (issue #47): 固定 dt driver は 3-tuple
        # `(psi, n_matvec, snapshot)` を返す. snapshot は save_tlist=None で
        # None.
        if method == "m2":
            psi_final, n_matvec, snapshot = evolve_schedule_m2(
                h_x=self.problem.h_x,
                h_p_diag=self.problem.H_p_diag,
                schedule=self.schedule,
                psi0=psi0_arr,
                t0=t0,
                t1=t1,
                n_steps=n_steps_int,
                m=self.m,
                krylov_tol=effective_propagator_tol,
                observables=observables_validated,
                save_tlist=save_tlist_arr,
                store_states=store_states,
            )
        elif method == "trotter":
            psi_final, n_matvec, snapshot = evolve_schedule_trotter(
                h_x=self.problem.h_x,
                h_p_diag=self.problem.H_p_diag,
                schedule=self.schedule,
                psi0=psi0_arr,
                t0=t0,
                t1=t1,
                n_steps=n_steps_int,
                observables=observables_validated,
                save_tlist=save_tlist_arr,
                store_states=store_states,
            )
        elif method == "trotter_suzuki4":
            psi_final, n_matvec, snapshot = evolve_schedule_trotter_suzuki4(
                h_x=self.problem.h_x,
                h_p_diag=self.problem.H_p_diag,
                schedule=self.schedule,
                psi0=psi0_arr,
                t0=t0,
                t1=t1,
                n_steps=n_steps_int,
                observables=observables_validated,
                save_tlist=save_tlist_arr,
                store_states=store_states,
            )
        else:  # method == "cfm4"
            psi_final, n_matvec, snapshot = evolve_schedule_cfm4(
                h_x=self.problem.h_x,
                h_p_diag=self.problem.H_p_diag,
                schedule=self.schedule,
                psi0=psi0_arr,
                t0=t0,
                t1=t1,
                n_steps=n_steps_int,
                m=self.m,
                krylov_tol=effective_propagator_tol,
                observables=observables_validated,
                save_tlist=save_tlist_arr,
                store_states=store_states,
            )
        return self._build_result(
            psi_final=psi_final,
            snapshot=snapshot,
            method=method,
            n_steps=n_steps_int,
            n_matvec=int(n_matvec),
            n_steps_actual=n_steps_int,
            m_eff_stats=None,
        )

    def _validate_save_tlist(
        self, save_tlist: np.ndarray | None, t0: float, t1: float
    ) -> np.ndarray | None:
        """``save_tlist`` の dtype / monotonicity / [t0, t1] 範囲を検証する.

        ``None`` のときは ``None`` をそのまま返す (最節約モード). 非 None の
        とき shape 1D, dtype float64, monotonic increasing (重複は許容),
        全要素が ``[t0, t1]`` の範囲内であることを検証し, C-contiguous な
        float64 array を返す.
        """
        if save_tlist is None:
            return None
        if not isinstance(save_tlist, np.ndarray):
            raise ValueError(
                f"save_tlist must be a numpy.ndarray or None, got {type(save_tlist).__name__}"
            )
        if save_tlist.ndim != 1:
            raise ValueError(
                f"save_tlist must be 1-dimensional, got shape {save_tlist.shape}"
            )
        if save_tlist.dtype != np.float64:
            raise ValueError(
                f"save_tlist dtype must be float64, got {save_tlist.dtype}"
            )
        if not np.all(np.isfinite(save_tlist)):
            raise ValueError("save_tlist contains NaN or inf")
        # 空配列は「観測無し」を意味するが silent 無視は debug 罠になるため
        # 明示的にエラーにする (呼出側で ``None`` を渡せばよい).
        if save_tlist.shape[0] == 0:
            raise ValueError(
                "save_tlist must be non-empty; pass save_tlist=None for no recording."
            )
        if not np.all(np.diff(save_tlist) >= 0.0):
            raise ValueError("save_tlist must be monotonically non-decreasing")
        if save_tlist[0] < t0 or save_tlist[-1] > t1:
            raise ValueError(
                f"save_tlist must fall within [t0={t0!r}, t1={t1!r}], "
                f"got [{save_tlist[0]!r}, {save_tlist[-1]!r}]"
            )
        return np.ascontiguousarray(save_tlist, dtype=np.float64)

    def _validate_observables(
        self, observables: dict[str, Observable] | None
    ) -> dict[str, Observable] | None:
        """``observables`` が ``dict[str, Observable]`` で各 diag が
        ``(2**n,)`` shape であることを検証する.

        ``None`` のとき ``None`` をそのまま返す. 空 dict は ``None`` と
        同義扱い (driver 側で空 dict を渡しても何も評価しないが,
        ``QuantumResult.observables_history`` を非空 dict として保存しない
        よう, ここで ``None`` に正規化する).
        """
        if observables is None:
            return None
        if not isinstance(observables, dict):
            raise ValueError(
                f"observables must be a dict or None, got {type(observables).__name__}"
            )
        if len(observables) == 0:
            return None
        expected_dim = self.problem.dim
        for name, obs in observables.items():
            if not isinstance(name, str):
                raise ValueError(
                    f"observables keys must be str, got {type(name).__name__}"
                )
            if not isinstance(obs, Observable):
                raise ValueError(
                    f"observables[{name!r}] must be an Observable, got {type(obs).__name__}"
                )
            if obs.dim != expected_dim:
                raise ValueError(
                    f"observables[{name!r}].diag length mismatch: "
                    f"expected {expected_dim}, got {obs.dim}"
                )
        return observables

    def _build_result(
        self,
        *,
        psi_final: np.ndarray,
        snapshot: dict[str, np.ndarray | dict[str, np.ndarray] | None] | None,
        method: str,
        n_steps: int,
        n_matvec: int,
        n_steps_actual: int,
        m_eff_stats: dict[str, int | float] | None,
        beta_m_stats: dict[str, float] | None = None,
        n_krylov_insufficient: int | None = None,
    ) -> QuantumResult:
        """``QuantumResult`` を組み立てる. ``probabilities`` は常に eager 計算.

        ``snapshot`` (driver の ``save_tlist`` 経路で組まれる dict) から
        ``times`` / ``states`` / ``observables_history`` を取り出し,
        ``save_tlist=None`` 経路では ``times=states=None`` /
        ``observables_history={}`` を返す.
        """
        # 最終状態の確率分布は常に eager 計算して返す (どの経路でも).
        probabilities = (np.abs(psi_final) ** 2).astype(np.float64, copy=False)
        if snapshot is None:
            times: np.ndarray | None = None
            states: np.ndarray | None = None
            obs_history: dict[str, np.ndarray] = {}
        else:
            # snapshot dict の value 型は union (np.ndarray | dict | None) なので
            # ``cast`` で各フィールドの想定型に narrow する. driver 側の
            # ``_SnapshotRecorder.finalize`` が key ごとに正しい型で書き込む
            # のが契約 (krylov.py 参照).
            times = cast("np.ndarray | None", snapshot.get("times"))
            states = cast("np.ndarray | None", snapshot.get("states"))
            obs_history = cast(
                "dict[str, np.ndarray]",
                snapshot.get("observables_history") or {},
            )
        return QuantumResult(
            psi_final=psi_final,
            # 互換: t_history は save_tlist 経路の times の別名として返す.
            # save_tlist=None では従来通り None. Phase 5 以前の呼出側は
            # save_tlist を使わないので挙動互換.
            t_history=times,
            observables_history=obs_history,
            n_steps=n_steps,
            n_matvec=n_matvec,
            success=True,
            method=method,
            n_steps_actual=n_steps_actual,
            m_eff_stats=m_eff_stats,
            beta_m_stats=beta_m_stats,
            n_krylov_insufficient=n_krylov_insufficient,
            times=times,
            states=states,
            probabilities=probabilities,
        )

    def create_simulator(
        self,
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
        atol: float | None = None,
        dt_init: float | None = None,
        dt_max: float | None = None,
        m_max: int | None = None,
    ) -> AnnealingSimulator:
        """``AnnealingSimulator`` (step-wise stateful API) を生成する.

        ``QuantumAnnealer`` と同じ ``problem`` / ``schedule`` / ``m`` /
        ``propagator_tol`` を引き継いだ Simulator を返す. ``psi0`` / ``t0`` /
        ``method`` 以降は Simulator コンストラクタに直接渡される.

        Parameters
        ----------
        psi0
            shape ``(2**n,)`` complex128 の初期状態. L2-normalize 済み.
        t0
            初期時刻.
        method
            プロパゲータ. ``run`` と同じ集合をサポート (``m2`` /
            ``trotter`` / ``trotter_suzuki4`` / ``cfm4`` /
            ``cfm4_adaptive_richardson_krylov`` /
            ``cfm4_adaptive_richardson_chebyshev``). default は ``run`` と同じ
            ``"cfm4_adaptive_richardson_chebyshev"`` (issue #124).
        atol, dt_init, dt_max, m_max
            adaptive method (``cfm4_adaptive_richardson_krylov`` /
            ``cfm4_adaptive_richardson_chebyshev``) 専用パラメータ.
            固定 dt method で指定すると ``ValueError``. ``m_max`` は
            Chebyshev variant では追加で ``ValueError`` (K_used 動的決定の
            ため Krylov 部分空間次元の概念がない). 詳細は
            ``AnnealingSimulator.__init__`` の docstring.

        Returns
        -------
        AnnealingSimulator
            step / advance_to / measure で逐次操作可能なシミュレータ.

        Notes
        -----
        ``m`` / ``propagator_tol`` を Simulator 側で上書きしたい場合は
        ``AnnealingSimulator`` を直接構築する.
        """
        from maqina.simulator import AnnealingSimulator

        return AnnealingSimulator(
            self.problem,
            self.schedule,
            psi0,
            t0,
            method=method,
            m=self.m,
            propagator_tol=self.propagator_tol,
            atol=atol,
            dt_init=dt_init,
            dt_max=dt_max,
            m_max=m_max,
        )
