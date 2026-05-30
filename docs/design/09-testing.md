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

### 9.4 adaptive controller の振動 (ノコギリ波) テスト基盤 (issue #152 / umbrella #148)

adaptive dt driver (`evolve_schedule_adaptive_*`) の step-size controller が
臨界領域で **ノコギリ波 (受理率 ≈ 50% / dt 振動)** に陥らないかを、決定論的かつ
クロスプラットフォームで検証する 2 層基盤。reject 回数は accept/reject 境界で
float 最下位差に敏感で **絶対値を CI で固定すると別マシンで落ちる** ため、
主役の指標は `log(dt)` の lag-1 自己相関のような **形状を捉える無次元量** にし、
各 sub-issue では同一実行内 old vs new の差分 + マージン付き不等式で assert する。

- **振動メトリクス** (`tests/_controller_metrics.py`): 受理率 / `n_rejects` /
  accepted `log(dt)` の lag-1 自己相関 (ノコギリ波 ≈ −1, 平滑 ≈ 0〜正) / 反転
  回数 / `std(log dt)`。`critical_window` で reject 集中域を自己同定し、絶対
  時刻のハードコードなしに窓内 metrics を算出する。単体テストは
  `test_controller_metrics.py`。
- **層 A — 合成誤差ハーネス** (`tests/_controller_harness.py`,
  `test_controller_sawtooth.py`): 3 driver の dispatch を monkeypatch し、
  物理状態更新なしで合成則 `err = C₄(t)·dt^{p+1}` (`p` は推定子 order; driver
  内 `_pi_dt_next(p=...)` に揃える) を返させて controller のダイナミクスだけを
  切り出す。純 float64 で決定論的・Rust ビルド非依存・高速。現行 main の
  ノコギリ波を characterization baseline として固定する (**主役テスト**)。
- **層 B — end-to-end ベンチ** (`benchmarks/bench_stepsize_controller.py`,
  `test_bench_stepsize_controller.py`): tanh バースト schedule で実 driver
  (実 Lanczos / Chebyshev) を小 N で走らせ、受理率 / `n_rejects` / dt ノコギリ
  振幅 / propagator call 数 / 終端 infidelity を出力。old vs new config 比較
  モードを持つ (絶対閾値でなく差分比較)。

後続 sub-issue (#149 reject 予測式 + クランプ / #150 成長凍結 / #151 真の
PI 化) は本基盤を再利用し「baseline → 各修正で lag-1 自己相関が −1 から改善 /
受理率回復」を示す。詳細は umbrella #148。

---

