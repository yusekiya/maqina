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
