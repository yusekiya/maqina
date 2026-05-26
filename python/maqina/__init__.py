"""maqina: 横磁場イジングモデル (TFIM) の量子アニーリングシミュレータ
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
>>> from maqina import IsingProblem, Schedule, QuantumAnnealer
>>> from maqina.initial_states import uniform_superposition
>>> from maqina.builders import diag_from_J_h
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
>>> ann = QuantumAnnealer(prob, sched)
>>> res = ann.run(psi0, 0.0, sched.T, method="m2", n_steps=300)
>>> print(np.abs(res.psi_final[:8]) ** 2)   # 最終状態 |ψ(T)|^2 の冒頭 8 成分

設計詳細は ``docs/design/INDEX.md`` 参照. 各公開モジュールに対応する ``.pyi``
スタブ (``python/maqina/*.pyi``) を一次 API リファレンスとして読むことを
推奨する.
"""

import warnings

from maqina.annealer import QuantumAnnealer
from maqina.eigenstates import instantaneous_eigenstates
from maqina.observable import Observable
from maqina.problem import IsingProblem
from maqina.result import QuantumResult, Trajectory
from maqina.schedule import Schedule
from maqina.simulator import AnnealingSimulator

__all__ = [
    "AnnealingSimulator",
    "IsingProblem",
    "Observable",
    "QuantumAnnealer",
    "QuantumResult",
    "Schedule",
    "Trajectory",
    "available_blas_threads",
    "instantaneous_eigenstates",
    "set_blas_threads",
    "set_blas_threads_auto",
    "show_config",
]


_ENV_VARS_BLAS_CAP = (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "OMP_NUM_THREADS",
)


def _warn_if_no_blas() -> None:
    """Rust 拡張が ``blas`` feature 無効でビルドされていれば 1 度だけ警告する.

    ``_rust.__has_blas__`` を import 時に読み, ``False`` (scalar fallback)
    の場合に ``RuntimeWarning`` を発する. scalar fallback ビルドのまま
    長時間ベンチを回す事故を防ぐためのアラート (``docs/design/07-rust-extension.md`` §7.5).
    Rust 拡張自体が import できない環境では何もせず, 上位 (krylov 層) の
    Python リファレンス fallback に任せる.

    動的 import (``importlib.import_module``) を使う理由: Rust 拡張は
    maturin develop 後にしか存在しないモジュールで, ``from maqina
    import _rust`` を静的に書くと ty 等の型チェッカが解決失敗で fail
    する. ``importlib`` 経由なら静的解析から見えず, 実行時のみ可用性を
    判定する形にできる.
    """
    import importlib

    try:
        rust_mod = importlib.import_module("maqina._rust")
    except ImportError:
        return
    has_blas = bool(getattr(rust_mod, "__has_blas__", False))
    if not has_blas:
        warnings.warn(
            "maqina._rust was built without the 'blas' feature. "
            "Lanczos / matvec primitives will use the Rust scalar fallback "
            "path which is significantly slower than the CBLAS path. "
            "Rebuild with `uv run maturin develop --uv --release` (default "
            "features include 'blas') to enable BLAS.",
            RuntimeWarning,
            stacklevel=2,
        )


_warn_if_no_blas()


def set_blas_threads(n: int) -> None:
    """ロード済みの全 OpenBLAS pool のスレッド上限を ``n`` に統一する.

    Rust kernel が system BLAS に動的リンクするため, maqina を import した
    Python プロセスには numpy bundled / scipy bundled / system OpenBLAS の
    最大 3 つの BLAS pool が同居しうる. bundled 版はシンボル prefix が
    ``libscipy_openblas`` にリネームされており, ``OPENBLAS_NUM_THREADS``
    等の環境変数を全 pool で一貫して honor するとは限らない. 本関数は
    ``threadpoolctl.threadpool_limits`` 経由で BLAS API (``user_api='blas'``)
    を使う全 pool に API ベースで ``set_num_threads(n)`` を呼ぶ.

    用途と限界
    ----------
    本関数は **既に確保された BLAS pool 内で active thread 数を制限する**
    だけで, pool size 自体 (= プロセスが確保している OS thread 数) は
    縮まらない. OpenBLAS / MKL 等の thread pool は **プロセス起動時** に
    ``OPENBLAS_NUM_THREADS`` / ``MKL_NUM_THREADS`` などの環境変数で size が
    確定し, 以降は ``set_num_threads(n)`` で active 数を絞っても残りの
    thread は sleep 状態で stack / kernel resource を保持する.

    そのため:

    - **import 後に動的に active BLAS thread 数を絞りたい** シナリオでは
      本関数を使う. rayon 経路と併用する際の **推奨 default** は
      :func:`set_blas_threads_auto` (issue #116 perf 実証で
      ``OPENBLAS_NUM_THREADS=8`` 相当で 1.52× speedup). 明示的に 1
      thread に絞りたい (= 完全に隔離したい) ときのみ
      ``set_blas_threads(1)`` を直接呼ぶ.
    - **per-process thread budget の隔離** が要件のシナリオ
      (multiprocessing / Slurm job array で 1 プロセスあたりの thread 数を
      物理的に制限したい) では, ``maqina`` / ``numpy`` を import する
      **前** に環境変数 (``OPENBLAS_NUM_THREADS`` / ``MKL_NUM_THREADS`` /
      ``VECLIB_MAXIMUM_THREADS`` / ``OMP_NUM_THREADS`` / ``RAYON_NUM_THREADS``)
      を set すること. 詳細パターンは ``docs/quickstart.md`` 末尾節
      "並列ジョブ実行時のスレッド数制御" 参照.

    Parameters
    ----------
    n
        各 BLAS pool に設定する active thread 数の上限. 1 以上の整数.
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


def _read_env_cap() -> int | None:
    """BLAS pool size の env 上限を読む.

    :data:`_ENV_VARS_BLAS_CAP` (筆頭 ``OPENBLAS_NUM_THREADS``) を順に走査し,
    最初に見つかった値を採用して ``int`` に parse して返す. parse 失敗時は
    その変数をスキップして次へ. 1 つも見つからなければ ``None``.

    Returns
    -------
    int | None
        env で指定された BLAS thread 上限 (1 以上). 何も set されていなければ
        ``None``.
    """
    import os

    for var in _ENV_VARS_BLAS_CAP:
        val = os.environ.get(var)
        if val is None:
            continue
        try:
            return max(1, int(val))
        except ValueError:
            continue
    return None


def _recommended_blas_threads() -> int:
    """Lanczos 内部 BLAS-1 経路で使う推奨 active thread 数 (issue #116).

    Default は ``process_cpu_count // 8`` を 1-16 でクランプ:

    ===================  ===========
    process_cpu_count    recommended
    ===================  ===========
    128 (EPYC SMT2)               16
     64 (32-core SMT2)             8
     32                            4
     16                            2
      ≤ 8                          1
    ===================  ===========

    数値は perf 実測 (#113 / PR #115, Linux AMD EPYC 7713P) の sweet spot
    (NT=8 で 1.52× speedup, NT=16 でも +2% で 1.49×) に基づく. 物理コア
    ≤ 8 の小規模マシンでは BLAS=1 として rayon 全コア並列との
    oversubscription を回避.

    :func:`_read_env_cap` で得た env 上限 (``OPENBLAS_NUM_THREADS`` 等) が
    set されていれば, それを **strict な上限** として
    ``min(auto, env_cap)`` を返す. ``os.process_cpu_count()`` 経由なので
    Slurm / cgroup / taskset で CPU 割当が絞られた環境ではそれを honor
    する (host 全 logical コア数ではなくプロセスへの割当を見る).

    :func:`available_blas_threads` (BLAS pool の現在 active 数 query) とは
    意図的に分離: 本関数は env から決まる **静的な policy** であり, 何度
    呼んでも同じ値を返す (idempotent). :func:`available_blas_threads` を
    cap に使うと :func:`set_blas_threads_auto` を 2 回呼んだとき cap が
    下がっていく non-idempotent な挙動になるため避ける.

    Returns
    -------
    int
        推奨 active BLAS thread 数 (1 以上).
    """
    import os

    cores = os.process_cpu_count() or os.cpu_count() or 1
    auto = max(1, min(16, cores // 8))
    env_cap = _read_env_cap()
    if env_cap is not None:
        return min(auto, env_cap)
    return auto


def set_blas_threads_auto() -> int:
    """:func:`_recommended_blas_threads` の値を適用し, 適用した値を返す.

    内部で :func:`set_blas_threads` を呼んで全 BLAS pool の active thread
    数を推奨値に統一する. issue #116 で確立した「rayon 経路でも
    BLAS=1 ではなく ``process_cpu_count / 8`` 程度を使う」という
    新方針の便利関数.

    冪等. 同じ環境変数 / cpu 割当下では何度呼んでも同じ値を返す.

    Examples
    --------
    >>> import maqina
    >>> maqina.set_blas_threads_auto()   # 推奨 default を適用  # doctest: +SKIP
    8
    >>> maqina.set_blas_threads(1)       # 明示 override            # doctest: +SKIP

    Returns
    -------
    int
        実際に適用した active BLAS thread 数.
    """
    n = _recommended_blas_threads()
    set_blas_threads(n)
    return n


def show_config() -> None:
    """ビルド構成を stdout に dump する (``numpy.show_config()`` 相当, issue #103).

    repo 同梱の ``.cargo/config.toml`` で ``-C target-cpu=native`` が default
    適用されるが, それが実際に build 時の SIMD 経路 (``wide::f64x4``) に
    反映されたか (= AVX2 / AVX-512 / NEON dispatch を選んだか) を確認する
    ためのヘルパ. ``uv add git+...`` 経由のソースビルド直後やベンチを取る
    前の build profile 確認に用いる.

    出力項目:

    * ``version``: ``importlib.metadata.version("maqina")`` で取得.
    * ``target arch`` / ``target OS``: Rust 拡張のビルドターゲット
      (``_rust.__target_arch__`` / ``__target_os__``, ``std::env::consts``
      由来).
    * ``cargo features``: ``__has_blas__`` / ``__has_rayon__`` / ``__has_simd__``
      (``cfg!(feature = "...")`` 由来).
    * ``target_features``: ``__has_avx2__`` / ``__has_fma__`` /
      ``__has_avx512f__`` / ``__has_neon__`` (``cfg!(target_feature = "...")``
      由来). ``target-cpu=native`` の効きを反映.

    Rust 拡張 (``maqina._rust``) が import できない環境では各行を
    ``unavailable`` と表示する.
    """
    import importlib
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _pkg_version

    try:
        rust_mod = importlib.import_module("maqina._rust")
    except ImportError:
        rust_mod = None

    try:
        ver = _pkg_version("maqina")
    except PackageNotFoundError:
        ver = "unknown"

    def _attr(name: str) -> object:
        if rust_mod is None:
            return "unavailable"
        return getattr(rust_mod, name, "unavailable")

    print("maqina build configuration")
    print("-" * 50)
    print(f"  version       : {ver}")
    print(f"  target arch   : {_attr('__target_arch__')}")
    print(f"  target OS     : {_attr('__target_os__')}")
    print()
    print("  cargo features:")
    print(f"    BLAS  : {_attr('__has_blas__')}")
    print(f"    rayon : {_attr('__has_rayon__')}")
    print(f"    SIMD  : {_attr('__has_simd__')}")
    print()
    print("  target_features (-C target-cpu=native の効きを反映):")
    for name in ("avx2", "fma", "avx512f", "neon"):
        val = _attr(f"__has_{name}__")
        if isinstance(val, bool):
            marker = "ON " if val else "off"
        else:
            marker = "?  "
        print(f"    [{marker}] {name}")
