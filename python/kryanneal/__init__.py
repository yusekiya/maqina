"""kryanneal: 横磁場イジングモデル (TFIM) の量子アニーリングシミュレータ
=====================================================================

Hamiltonian:
    H(t) = A(s(t)) · H_driver + B(s(t)) · H_problem
    H_driver  = -Σ_i h_x_i X_i              (サイト依存横磁場, bit-flip)
    H_problem = Z 演算子のみの k-local 多項式 (Z 基底で対角)

ユーザー入力:
    - H_p_diag : (2^N,) float64   Z 基底での H_problem 対角ベクトル
    - h_x      : (N,)   float64   サイト依存横磁場の振幅
    - psi0     : (2^N,) complex128 初期状態 (L2-normalize 済み, 必須明示指定)

ビット規約: bit 0 = LSB, x = Σ_i b_i · 2^i, spin σ_i = 1 - 2·b_i.

Usage
-----
>>> import numpy as np
>>> from kryanneal import IsingProblem, Schedule, QuantumAnnealer
>>> from kryanneal.initial_states import uniform_superposition
>>> from kryanneal.builders import diag_from_J_h
>>>
>>> n = 4
>>> J = np.random.default_rng(0).normal(size=(n, n)) / np.sqrt(n)
>>> J = (J + J.T) / 2; np.fill_diagonal(J, 0.0)
>>> h = np.zeros(n)
>>> prob = IsingProblem(
...     n=n,
...     H_p_diag=diag_from_J_h(J, h),
...     h_x=np.ones(n),
... )
>>> sched = Schedule.linear(T=30.0)
>>> psi0 = uniform_superposition(n)
>>> ann = QuantumAnnealer(prob, sched, psi0)
>>> res = ann.run(method="cfm4_adaptive_richardson")
>>> print(res.probabilities[:8])   # 最終状態 |ψ(T)|^2 の冒頭 8 成分

設計詳細は ``docs/design.md`` 参照. 各公開モジュールに対応する ``.pyi``
スタブ (``python/kryanneal/*.pyi``) を一次 API リファレンスとして読むことを
推奨する.
"""

from kryanneal.problem import IsingProblem
from kryanneal.result import QuantumResult, Trajectory
from kryanneal.schedule import Schedule

# QuantumAnnealer / AnnealingSimulator は C7 (Phase 1) / Phase 5 で実装予定.
# from kryanneal.annealer import QuantumAnnealer, AnnealingSimulator

__all__ = [
    "IsingProblem",
    "QuantumResult",
    "Schedule",
    "Trajectory",
    # "QuantumAnnealer",
    # "AnnealingSimulator",
    "set_blas_threads",
    "available_blas_threads",
]


def set_blas_threads(n: int) -> None:
    """ロード済みの全 OpenBLAS pool のスレッド上限を ``n`` に統一する.

    Rust kernel が system BLAS に動的リンクするため, kryanneal を import した
    Python プロセスには numpy bundled / scipy bundled / system OpenBLAS の
    最大 3 つの BLAS pool が同居しうる. bundled 版はシンボル prefix が
    ``libscipy_openblas`` にリネームされており, ``OPENBLAS_NUM_THREADS``
    等の環境変数を全 pool で一貫して honor するとは限らない. 本関数は
    ``threadpoolctl.threadpool_limits`` 経由で BLAS API (``user_api='blas'``)
    を使う全 pool に API ベースで ``set_num_threads(n)`` を呼ぶ.

    Parameters
    ----------
    n
        各 BLAS pool に設定するスレッド上限. 1 以上の整数.
    """
    from threadpoolctl import threadpool_limits

    threadpool_limits(limits=n, user_api="blas")


def available_blas_threads() -> int:
    """現在のプロセスで実効的に利用可能な BLAS スレッド数を返す.

    複数の BLAS pool (Apple Accelerate / OpenBLAS / MKL ...) が同居する場合は
    ``threadpoolctl.threadpool_info()`` の ``num_threads`` の最小値を採り,
    さらに ``os.process_cpu_count()`` で上限をキャップする (プロセスへの
    CPU 割当が BLAS 設定より小さければそちらが律速).

    Returns
    -------
    int
        有効な BLAS スレッド予算. 最小 1.
    """
    import os

    from threadpoolctl import threadpool_info

    n_cpu = os.process_cpu_count() or 1
    blas_pools = [p for p in threadpool_info() if p.get("user_api") == "blas"]
    if blas_pools:
        n_blas = min(int(p.get("num_threads", n_cpu)) for p in blas_pools)
    else:
        n_blas = n_cpu
    return max(1, min(n_blas, n_cpu))
