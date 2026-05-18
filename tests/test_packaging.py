"""パッケージング / wheel smoke test.

``maturin develop`` で構築した ``kryanneal._rust`` がロードでき, パッケージ
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


def test_import_kryanneal() -> None:
    """``import kryanneal`` がエラー無く成功し, 公開シンボルが見える."""
    import kryanneal

    # __init__.py の __all__ に列挙したシンボルは module attribute として
    # 露出している.
    assert hasattr(kryanneal, "set_blas_threads")
    assert hasattr(kryanneal, "available_blas_threads")
    assert callable(kryanneal.set_blas_threads)
    assert callable(kryanneal.available_blas_threads)


def test_rust_extension_loads() -> None:
    """Rust 拡張モジュール ``kryanneal._rust`` がロード可能.

    拡張未ビルド環境 (``maturin develop`` 未実施 / fallback-only build) では
    skip する. fallback 経路自体は他テスト (``test_krylov.py`` の
    Python リファレンス系) で網羅されている.
    """
    _rust = pytest.importorskip("kryanneal._rust")

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
    _rust = pytest.importorskip("kryanneal._rust")

    assert hasattr(_rust, "__has_blas__")
    assert isinstance(_rust.__has_blas__, bool)


def test_available_blas_threads_returns_positive_int() -> None:
    """``available_blas_threads`` は最小 1 以上の int を返す."""
    from kryanneal import available_blas_threads

    n = available_blas_threads()
    assert isinstance(n, int)
    assert n >= 1


def test_target_feature_flags_are_bool() -> None:
    """``_rust.__has_avx2__`` 等の ``target_feature`` ベース flag が bool として
    参照できる (issue #103). 値そのものは build 時の ``target-cpu`` 設定および
    target arch に依存するので, 本テストでは **型のみ** を確認する.

    ``target-cpu=native`` の効きの実値確認は ``kryanneal.show_config()`` の
    手動実行で行う方針.
    """
    _rust = pytest.importorskip("kryanneal._rust")

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
    _rust = pytest.importorskip("kryanneal._rust")

    for name in ("__target_arch__", "__target_os__"):
        assert hasattr(_rust, name), f"{name} should be exposed on _rust"
        val = getattr(_rust, name)
        assert isinstance(val, str)
        assert len(val) > 0


def test_show_config_runs(capsys: pytest.CaptureFixture[str]) -> None:
    """``kryanneal.show_config()`` が呼び出し可能で構成情報を stdout に出力する.

    Rust 拡張が無い debug 環境でも fallback として ``unavailable`` 表示で
    動作するように設計しているため, ``importorskip`` は不要.
    """
    import kryanneal

    kryanneal.show_config()
    captured = capsys.readouterr()
    assert "kryanneal build configuration" in captured.out
    assert "cargo features" in captured.out
    assert "target_features" in captured.out
