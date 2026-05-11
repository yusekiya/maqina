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
  ``uv run pytest`` を 2 回回す運用. ``docs/testing.md`` §CI 参照).
"""

from __future__ import annotations


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
    """Rust 拡張モジュール ``kryanneal._rust`` がロード可能."""
    from kryanneal import _rust

    # PyO3 拡張は通常の ModuleType ではなく builtin types を返すため,
    # 厳密な型チェックではなく属性アクセス可否で判定する.
    assert _rust is not None


def test_has_blas_flag_is_bool() -> None:
    """``_rust.__has_blas__`` が bool として参照できる.

    True/False のいずれかは build 時の feature 選択に依存する. 本テストは
    値の真偽ではなく **bool 型で属性として露出していること** のみを主張
    する (BLAS on/off の網羅は CI で 2 回 build / 2 回テストして担保).
    """
    from kryanneal import _rust

    assert hasattr(_rust, "__has_blas__")
    assert isinstance(_rust.__has_blas__, bool)


def test_available_blas_threads_returns_positive_int() -> None:
    """``available_blas_threads`` は最小 1 以上の int を返す."""
    from kryanneal import available_blas_threads

    n = available_blas_threads()
    assert isinstance(n, int)
    assert n >= 1
