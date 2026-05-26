# §5.4 Python リファレンス実装

`python/maqina/krylov.py` に `_python_lanczos_propagate` /
`_python_cfm4_step` 等を **pure NumPy** で実装し、Rust 拡張がビルドできない
環境でも silently fallback する設計とする。

- 等価性テスト: tight tol で `rel < 1e-13` 一致。
- 開発時のデバッグ・教育用途にも有用。

---

