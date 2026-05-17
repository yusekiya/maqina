# §9. テスト戦略

### 9.1 Rust 単体テスト (`cargo test`)

- `blas.rs` の各ヘルパ (norm2 / dot_conj / axpy / scal_real / gemv) について
  BLAS feature on/off の両ビルドで数値一致を確認。
- `lanczos_propagate` を closure で呼ぶ:
  - zero psi → zero
  - H = 0 → psi 不変 (norm 維持)
  - 対角 H → psi[k] *= exp(-i dt λ_k) と一致
  - Hermitian H → norm 保存 (rel 1e-13)
- `apply_h` 単体: ランダム (h_x, H_p_diag, A, B) で **dense 構築版**と一致
  (`y_ref = (A·H_drv + B·diag) · v` を NumPy で組んで比較)。

実行:

```bash
cargo test                       # blas feature ON
cargo test --no-default-features # scalar fallback
```

(`Cargo.toml` がプロジェクトルートにあるため `cd` 不要)

### 9.2 Python 統合テスト (`uv run pytest`)

- `test_problem.py`: 入力検証 (shape / dtype / NaN / 非正規化 psi0 拒否)。
- `test_builders.py`: `diag_from_pauli_terms` / `diag_from_J_h` が手計算と一致。
- `test_krylov.py`: Python リファレンス vs Rust 実装 の `rel < 1e-13`。
- `test_cfm4.py`: time-independent H で `exp(-i T H) · psi0` と一致。
- `test_richardson.py`: 既知 schedule に対する dt 自動調整の収束。
- `test_annealer.py`: 公開 API スモーク (linear schedule, GS 到達確率)。
- `test_eigenstates.py`: 小規模 n=4 で dense `eigh` と Lanczos 結果が一致。
- `test_reference_qutip.py`: Section 8.3 の参照比較。`@pytest.mark.slow`。

`-m "not slow"` で slow テストを除外できるように `pytest.ini_options`。

### 9.3 等価性ペアの規約

「Python リファレンス vs Rust」「Rust vs QuTiP」「Krylov vs exact `eigh`」の
3 種のペアテストを軸に、新しい propagator を増やすたびに同じパターンの
ペアテストを追加する。

---

