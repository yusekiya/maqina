# §8. QuTiP 比較

### 8.1 位置付け

QuTiP は **本パッケージの提供する機能ではなく、近似誤差評価および性能
比較のためのリファレンス**。`tests/` および `benchmarks/` 配下に閉じ込め、
公開 API には載せない。`pyproject.toml` の dev dependency として宣言。

### 8.2 リファレンス実装 `reference_qutip.py`

```python
# src/maqina/reference_qutip.py
# 注: pyproject.toml の dev dep として qutip を入れる. import エラー時は
# pytest.skip / 明示的なエラーメッセージで誘導する.

def reference_evolve(
    problem: IsingProblem,
    schedule: Schedule,
    psi0: np.ndarray,
    *,
    method: Literal["sesolve", "krylov_qutip"] = "sesolve",
    tlist: np.ndarray,
    nsteps_ode: int = 50_000,
    atol: float = 1e-12,
    rtol: float = 1e-10,
) -> "qutip.solver.Result":
    """
    dense Hamiltonian を組み立てて qt.sesolve で時間発展。

    本パッケージの近似誤差評価および性能比較用. n が大きい
    (n >= 16 程度) と dense H が DRAM を食うので注意.
    """
```

実装上の注意:

- `H_problem` の dense 化: `qt.Qobj(np.diag(H_p_diag), dims=[[2]*n]*2)` で
  済むが 2^n × 2^n の dense なので n=16 で 32 GB。ヘルパ内で `sparse=True`
  を使い `qt.Qobj.diags(H_p_diag)` で sparse-diag に。
- `H_driver`: site ごとに `qt.tensor` で `sigmax()` を組む。これは sparse
  なので n=20 程度までは行ける。
- 結果オブジェクトは `qutip.solver.Result` をそのまま返し、テストで
  `res_qutip.states[-1]` と `maqina の psi_final` の fidelity を比較。

### 8.3 比較項目

`tests/test_reference_qutip.py`:

- 小規模 (n=4, 6) でランダム H_p_diag / h_x / 線形 schedule を生成し、
  - M2 / CFM4 / Richardson の各経路と qt.sesolve の最終 fidelity が
    `|⟨ψ_kry|ψ_qutip⟩|^2 > 1 - 1e-8` (固定 step で n_steps を増やせば
    1 - 1e-10 まで)
  - 観測量期待値の時系列も `np.allclose(atol=1e-8)` で一致
- 中規模 (n=10) は `@pytest.mark.slow` で除外可能に。

`benchmarks/bench_vs_qutip.py`:

- 同一 (problem, schedule, psi0, T, target accuracy) に対して
  - QuTiP sesolve の wall time
  - maqina M2 / CFM4 / Richardson の wall time
  - 最終 fidelity
- を CSV + markdown で `benchmarks/results/<YYYYMMDD-HHMMSS>/` 配下に出力。
- 性能改善の主張は **同一マシン上の before/after** で示す (§10 のベンチ規約
  を参照)。

---

