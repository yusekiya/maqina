"""パッケージング / wheel smoke test.

``maturin develop`` で構築した ``maqina._rust`` がロードでき, パッケージ
レベルの公開シンボルおよび BLAS feature 検出フラグが正しく露出している
ことを確認する.

BLAS feature on/off の **両ブランチ網羅** はテスト 1 回では行わない. その
代わり以下を担保する設計:

* ``__has_blas__`` が bool として参照可能であること (本テスト).
* 同じ pytest を BLAS on と off の両 build で再実行することで両ブランチを
  網羅する (CI ジョブで ``maturin develop`` と
  ``maturin develop --no-default-features`` を別ステップで切替えて
  ``uv run pytest`` を 2 回回す運用. ``.claude/skills/test-runner/SKILL.md`` §CI 参照).
"""

from __future__ import annotations

import pytest


def test_import_maqina() -> None:
    """``import maqina`` がエラー無く成功し, 公開シンボルが見える."""
    import maqina

    # __init__.py の __all__ に列挙したシンボルは module attribute として
    # 露出している.
    assert hasattr(maqina, "set_blas_threads")
    assert hasattr(maqina, "available_blas_threads")
    assert hasattr(maqina, "set_blas_threads_auto")
    assert callable(maqina.set_blas_threads)
    assert callable(maqina.available_blas_threads)
    assert callable(maqina.set_blas_threads_auto)


def test_rust_extension_loads() -> None:
    """Rust 拡張モジュール ``maqina._rust`` がロード可能.

    拡張未ビルド環境 (``maturin develop`` 未実施 / fallback-only build) では
    skip する. fallback 経路自体は他テスト (``test_krylov.py`` の
    Python リファレンス系) で網羅されている.
    """
    _rust = pytest.importorskip("maqina._rust")

    # PyO3 拡張は通常の ModuleType ではなく builtin types を返すため,
    # 厳密な型チェックではなく属性アクセス可否で判定する.
    assert _rust is not None


def test_has_blas_flag_is_bool() -> None:
    """``_rust.__has_blas__`` が bool として参照できる.

    True/False のいずれかは build 時の feature 選択に依存する. 本テストは
    値の真偽ではなく **bool 型で属性として露出していること** のみを主張
    する (BLAS on/off の網羅は CI で 2 回 build / 2 回テストして担保).
    拡張未ビルド環境では skip.
    """
    _rust = pytest.importorskip("maqina._rust")

    assert hasattr(_rust, "__has_blas__")
    assert isinstance(_rust.__has_blas__, bool)


def test_available_blas_threads_returns_positive_int() -> None:
    """``available_blas_threads`` は最小 1 以上の int を返す."""
    from maqina import available_blas_threads

    n = available_blas_threads()
    assert isinstance(n, int)
    assert n >= 1


def test_recommended_blas_threads_returns_positive_int() -> None:
    """``_recommended_blas_threads`` は 1 以上 16 以下の int を返す (env / cpu 状態に依らず)."""
    from maqina import _recommended_blas_threads

    n = _recommended_blas_threads()
    assert isinstance(n, int)
    assert 1 <= n <= 16


def test_recommended_blas_threads_clamp_formula(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_recommended_blas_threads`` は ``process_cpu_count // 8`` を 1-16 でクランプ.

    issue #116 で確定した formula. perf 実測 (#113 EPYC 7713P) の sweet spot
    (NT=8 で 1.52× speedup) に基づく.
    """
    import os

    from maqina import _recommended_blas_threads

    # env_cap を確実に外す (host 環境を汚染しない).
    for var in (
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "OMP_NUM_THREADS",
    ):
        monkeypatch.delenv(var, raising=False)

    cases = [
        (128, 16),  # EPYC SMT2
        (64, 8),  # 32-core SMT2 or 64-core no SMT
        (32, 4),  # 16-core SMT2
        (16, 2),  # 8-core SMT2
        (8, 1),  # 4-core SMT2
        (4, 1),  # 2-core SMT2 (clamped to 1)
        (1, 1),  # 単コア
    ]
    for cpu, expected in cases:
        monkeypatch.setattr(os, "process_cpu_count", lambda cpu=cpu: cpu)
        # cpu_count fallback も同値にしておく (process_cpu_count が None を
        # 返す古い Python 経路向け. Py 3.13+ では process_cpu_count が優先).
        monkeypatch.setattr(os, "cpu_count", lambda cpu=cpu: cpu)
        assert _recommended_blas_threads() == expected, (
            f"cpu={cpu}: expected {expected}"
        )


def test_recommended_blas_threads_env_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    """``OPENBLAS_NUM_THREADS`` 等の env は strict な上限として尊重される.

    auto > env_cap なら env_cap, auto < env_cap なら auto. env_cap=1 は
    隔離環境 (Slurm 1-CPU 等) の opt-out として 1 を返す.
    """
    import os

    from maqina import _recommended_blas_threads

    for var in (
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "OMP_NUM_THREADS",
    ):
        monkeypatch.delenv(var, raising=False)

    # EPYC 相当 (cpu=128, auto=16).
    monkeypatch.setattr(os, "process_cpu_count", lambda: 128)
    monkeypatch.setattr(os, "cpu_count", lambda: 128)

    # env_cap < auto → env_cap.
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "4")
    assert _recommended_blas_threads() == 4

    # env_cap > auto → auto (env は単なる上限なので超えない).
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "64")
    assert _recommended_blas_threads() == 16

    # env_cap = 1 (隔離環境) → 1.
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    assert _recommended_blas_threads() == 1


def test_read_env_cap_priority(monkeypatch: pytest.MonkeyPatch) -> None:
    """``_read_env_cap`` は OPENBLAS → MKL → VECLIB → OMP の順で最初に見つかった値を採用."""
    from maqina import _read_env_cap

    for var in (
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "OMP_NUM_THREADS",
    ):
        monkeypatch.delenv(var, raising=False)

    # 何も set されていなければ None.
    assert _read_env_cap() is None

    # OMP のみ set → 採用される.
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    assert _read_env_cap() == 3

    # VECLIB が優先される (OMP より前).
    monkeypatch.setenv("VECLIB_MAXIMUM_THREADS", "5")
    assert _read_env_cap() == 5

    # MKL が優先される.
    monkeypatch.setenv("MKL_NUM_THREADS", "7")
    assert _read_env_cap() == 7

    # OPENBLAS が筆頭.
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "11")
    assert _read_env_cap() == 11

    # 非整数値はスキップして次の env へ.
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "garbage")
    assert _read_env_cap() == 7  # 次にヒットする MKL.


def test_read_env_cap_floors_at_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """env_cap は最低 1 (0 や負値が set されていても 1 にクランプ)."""
    from maqina import _read_env_cap

    for var in (
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "OMP_NUM_THREADS",
    ):
        monkeypatch.delenv(var, raising=False)

    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "0")
    assert _read_env_cap() == 1
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "-5")
    assert _read_env_cap() == 1


def test_set_blas_threads_auto_returns_applied_value() -> None:
    """``set_blas_threads_auto`` は :func:`_recommended_blas_threads` の値を適用し返す.

    冪等性も確認 (2 回呼んでも同じ値を返す).
    """
    from maqina import _recommended_blas_threads, set_blas_threads_auto

    n1 = set_blas_threads_auto()
    n2 = set_blas_threads_auto()
    assert isinstance(n1, int)
    assert n1 == _recommended_blas_threads()
    assert n1 == n2


def test_target_feature_flags_are_bool() -> None:
    """``_rust.__has_avx2__`` 等の ``target_feature`` ベース flag が bool として
    参照できる (issue #103). 値そのものは build 時の ``target-cpu`` 設定および
    target arch に依存するので, 本テストでは **型のみ** を確認する.

    ``target-cpu=native`` の効きの実値確認は ``maqina.show_config()`` の
    手動実行で行う方針.
    """
    _rust = pytest.importorskip("maqina._rust")

    for name in (
        "__has_avx2__",
        "__has_fma__",
        "__has_avx512f__",
        "__has_neon__",
    ):
        assert hasattr(_rust, name), f"{name} should be exposed on _rust"
        assert isinstance(getattr(_rust, name), bool)


def test_target_arch_and_os_are_str() -> None:
    """``_rust.__target_arch__`` / ``__target_os__`` が空でない str として参照できる.

    値は ``std::env::consts::ARCH`` / ``OS`` 由来 ("x86_64" / "aarch64" /
    "linux" / "macos" 等). 値そのものは host に依存するので assert しない.
    """
    _rust = pytest.importorskip("maqina._rust")

    for name in ("__target_arch__", "__target_os__"):
        assert hasattr(_rust, name), f"{name} should be exposed on _rust"
        val = getattr(_rust, name)
        assert isinstance(val, str)
        assert len(val) > 0


def test_show_config_runs(capsys: pytest.CaptureFixture[str]) -> None:
    """``maqina.show_config()`` が呼び出し可能で構成情報を stdout に出力する.

    Rust 拡張が無い debug 環境でも fallback として ``unavailable`` 表示で
    動作するように設計しているため, ``importorskip`` は不要.
    """
    import maqina

    maqina.show_config()
    captured = capsys.readouterr()
    assert "maqina build configuration" in captured.out
    assert "cargo features" in captured.out
    assert "target_features" in captured.out
