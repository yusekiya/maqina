# §6. ビルダー (k-local → 対角ベクトル)

```python
# src/kryanneal/builders.py
def diag_from_pauli_terms(
    n: int,
    terms: list[PauliTerm],
) -> np.ndarray:
    """
    PauliTerm リストから H_p_diag を構築 (Z のみ).

    各 term `coeff · Z_{i_1} Z_{i_2} ...` について,
    計算基底 |x⟩ での固有値は coeff · Π_j σ_{i_j}
    (σ = 1 - 2·b)。これを length 2^n 配列に蓄積。

    Parameters
    ----------
    n
        スピン数。dim = 2**n を allocate するので n <= 28 程度が現実的.
    terms
        ops が "Z..." (Z のみ) でないものを含むと ValueError.

    Returns
    -------
    H_p_diag : (2**n,) float64
    """
```

```python
def diag_from_J_h(
    J: np.ndarray,            # (n, n) symmetric, real, J_ii = 0
    h: np.ndarray,             # (n,) real
) -> np.ndarray:
    """
    H_p = -Σ_{i<j} J_ij σ_i σ_j - Σ_i h_i σ_i の対角を作る (σ ∈ {±1}).

    Internal: bit ベクトル化で 2^n を 1 回スキャン.
    """
```

これらは「`IsingProblem` に対角配列を渡す」設計の上で **オプションの利便性**
として提供。ユーザーは自前で作っても良い。

---

