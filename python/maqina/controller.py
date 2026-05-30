"""adaptive PI controller の数値挙動 knob を集約する ``ControllerConfig``.

``QuantumAnnealer.run`` / ``QuantumAnnealer.create_simulator`` /
``AnnealingSimulator`` の facade から adaptive driver
(``evolve_schedule_adaptive_{richardson,richardson_chebyshev}``) へ渡す
**純粋な数値挙動パラメータ** を 1 つの frozen dataclass にまとめる
(issue #149, umbrella #148 方針 B)。後続 sub-issue (#150 成長凍結 /
#151 真の PI 化) はこの dataclass に field を追加していく。

集約する knob (これまで facade から指定できなかったものを含む):

* ``safety`` / ``growth_max`` / ``max_rejects`` / ``dt_min``: driver 引数
  としては存在したが facade から触れなかった controller knob を本 issue で
  まとめて公開する (取りこぼし解消)。
* ``reject_shrink_min`` / ``reject_shrink_max``: reject 時の dt 縮小を
  **固定 0.5 倍** から **accept と同じ予測式 + クランプ** に変更するための
  新規 field (issue #149)。``[reject_shrink_min, reject_shrink_max]`` で
  reject factor をクランプする (``reject_shrink_max < 1`` で必ず縮小)。
  ``reject_shrink_min = reject_shrink_max = 0.5`` を渡すと旧挙動 (固定
  半減) を厳密に再現できる (回帰アンカー)。
* ``pi_alpha`` / ``pi_beta``: accept 時の dt 予測式に **真の PI 比例項**
  ``(err_prev / err)^{pi_beta/(p+1)}`` を加えるための新規 field (issue #151)。
  ``pi_alpha`` は積分項 ``(tol_step / err)^{pi_alpha/(p+1)}`` の係数。これまでの
  ``_pi_dt_next`` は比例項を持たない純粋な I (積分) 制御で、docstring / 設計書
  の "PI controller" 表記と実体が不整合だった (issue #151 で解消)。既定値は
  Gustafsson / Hairer-Wanner II §IV.2 の predictive PI controller 標準域
  (``pi_alpha = 0.7`` / ``pi_beta = 0.4``)。``pi_alpha = 1.0, pi_beta = 0.0``
  を渡すと旧挙動 (純 I 制御) を厳密に再現できる (回帰アンカー)。

**scope 外** (facade kwarg のまま据置): ``atol`` / ``dt_init`` / ``dt_max`` /
``m_max`` は精度要求・auto-resolve ロジックを持つため ``ControllerConfig``
には入れない (controller の純粋な数値挙動 knob のみ集約)。

詳細は ``docs/design/05-3-propagator.md`` "PI controller / adaptive ドライバ"
節を参照。
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ControllerConfig"]


@dataclass(frozen=True)
class ControllerConfig:
    """adaptive PI controller の数値挙動パラメータ (immutable).

    Parameters
    ----------
    safety
        PI 安全係数。``dt_next = dt · safety · (tol_step / err)^{1/(p+1)}``
        の予測式の係数 (accept / reject 両経路で共通)。``0 < safety``。
    growth_max
        accept 時の 1 step あたり dt 拡大率上限 (facmax)。
        ``dt_next = min(dt_next, dt_try · growth_max, dt_max)``。
        ``growth_max > 1`` (#150 の成長凍結が一時的に 1.0 にする土台)。
    max_rejects
        同一 step での連続 reject 上限。超過で driver が ``RuntimeError``。
        ``max_rejects >= 1``。
    dt_min
        最小 dt 床。reject 経路でも ``dt = max(dt_try · factor, dt_min)``
        で適用する。``0 < dt_min``。
    reject_shrink_min
        reject 時の dt 縮小 factor の下限 (= ``fac_min``)。``err`` が
        ``tol_step`` を大きく超えたときに ``factor`` がここまで落ちる。
        ``0 < reject_shrink_min``。
    reject_shrink_max
        reject 時の dt 縮小 factor の上限 (= ``fac_reject_max``)。``err`` が
        ``tol_step`` をわずかに超えただけのとき ``factor ≈ reject_shrink_max``
        程度で済み、order-5 推定子での過剰縮小 (誤差 32× 削減) を断つ。
        ``reject_shrink_min <= reject_shrink_max < 1`` (``< 1`` で必ず縮小)。
    freeze_growth_after_reject
        reject 後の accept で dt 拡大を一時凍結するか (issue #150,
        Gustafsson ヒステリシス)。``True`` (既定, 新挙動) のとき reject 直後の
        accept では ``eff_growth_max = 1.0`` として **拡大のみ禁止** する
        (縮小方向は許可)。過剰縮小 → 楽々 accept → 即 dt 再上昇 → 再
        オーバーシュートというノコギリ波の「再上昇」側を断つ。``False`` で
        #149 のみ適用した挙動 (reject 直後でも ``growth_max`` まで拡大可) に
        戻せる。
    growth_freeze_steps
        reject 後の成長凍結を解除するまでの **連続 accept 回数** (issue #150,
        既定 ``1`` = DOPRI/Hairer-Wanner 標準の「reject 直後の 1 step だけ
        facmax=1、成功で復帰」)。reject のたびに再武装し、凍結中の各 accept は
        拡大のみ禁止 (縮小は許可) となる。``growth_freeze_steps >= 1``。
        ``freeze_growth_after_reject=False`` のとき本 field は無視される。
    pi_alpha
        accept 時 dt 予測式の **積分 (I) 項** 指数の係数 (issue #151)。
        ``dt_next = dt · safety · (tol_step / err)^{pi_alpha/(p+1)} ·
        (err_prev / err)^{pi_beta/(p+1)}`` の第 1 因子。``pi_alpha > 0``。
        既定 ``0.7`` (Gustafsson predictive PI controller 標準)。
        ``pi_alpha = 1.0`` で従来の I 制御の積分項に一致する。
    pi_beta
        accept 時 dt 予測式の **比例 (P) 項** 指数の係数 (issue #151)。誤差の
        増加傾向 (Magnus 4 次係数 C₄ の上昇) を ``err_prev / err`` で先読みして
        dt 拡大を抑制する。``pi_beta >= 0``。既定 ``0.4`` (Gustafsson 標準)。
        ``pi_beta = 0.0`` で比例項が無効化され純 I 制御に戻る (``pi_alpha = 1.0``
        と合わせて旧挙動の回帰アンカー)。``err_prev`` は直前に **accept** した
        step の駆動量 (M2 = ``err``, Richardson / Chebyshev = ``err_magnus``);
        最初の accept や ``err`` / ``err_prev`` が 0 近傍 (``<= 1e-30``) のときは
        比例項を ``1.0`` に落として発散を回避する。

    Raises
    ------
    ValueError
        いずれかのフィールドが許容範囲外の場合 (``__post_init__`` で検証)。
    """

    safety: float = 0.9
    growth_max: float = 4.0
    max_rejects: int = 50
    dt_min: float = 1e-4
    reject_shrink_min: float = 0.2
    reject_shrink_max: float = 0.9
    freeze_growth_after_reject: bool = True
    growth_freeze_steps: int = 1
    pi_alpha: float = 0.7
    pi_beta: float = 0.4

    def __post_init__(self) -> None:
        if not (self.safety > 0.0):
            raise ValueError(f"safety must be > 0, got {self.safety!r}")
        if not (self.growth_max > 1.0):
            raise ValueError(f"growth_max must be > 1, got {self.growth_max!r}")
        if not (isinstance(self.max_rejects, int) and self.max_rejects >= 1):
            raise ValueError(
                f"max_rejects must be an int >= 1, got {self.max_rejects!r}"
            )
        if not (self.dt_min > 0.0):
            raise ValueError(f"dt_min must be > 0, got {self.dt_min!r}")
        if not (0.0 < self.reject_shrink_min <= self.reject_shrink_max < 1.0):
            raise ValueError(
                "reject_shrink_min / reject_shrink_max must satisfy "
                "0 < reject_shrink_min <= reject_shrink_max < 1, got "
                f"reject_shrink_min={self.reject_shrink_min!r}, "
                f"reject_shrink_max={self.reject_shrink_max!r}"
            )
        if not (
            isinstance(self.growth_freeze_steps, int) and self.growth_freeze_steps >= 1
        ):
            raise ValueError(
                "growth_freeze_steps must be an int >= 1, got "
                f"{self.growth_freeze_steps!r}"
            )
        if not (self.pi_alpha > 0.0):
            raise ValueError(f"pi_alpha must be > 0, got {self.pi_alpha!r}")
        if not (self.pi_beta >= 0.0):
            raise ValueError(f"pi_beta must be >= 0, got {self.pi_beta!r}")
