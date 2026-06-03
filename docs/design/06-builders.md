# §6. ビルダー (k-local → 対角ベクトル)

実装は `python/maqina/builders.py` (純 Python; issue #162 で実装済み)。

```python
def diag_from_pauli_terms(
    n: int,
    terms: Iterable[tuple[float, tuple[int, ...]]],
) -> np.ndarray:
    """
    Pauli-Z term の和から H_p_diag を構築 (Z のみ, 対角).

    各 term `(coeff, sites)` = `coeff · Z_{i_1} Z_{i_2} ...` について,
    計算基底 |x⟩ での固有値は coeff · Π_j σ_{i_j}
    (σ = 1 - 2·b)。これを length 2^n 配列に **加算で蓄積**。

    Returns: H_p_diag : (2**n,) float64
    """
```

項の指定は **`(coeff, sites)` の 2 要素タプル** で行う:

- **演算子文字列 (`"Z"` / `"ZZ"`) は持たない**。builders は Z 演算子のみ扱う
  (「H_problem は必ず Z 基底で対角」という設計契約) ので演算子種は Z 固定,
  何体項かは `len(sites)` で決まり, 文字列は完全に冗長。旧 §4 案の
  `PauliTerm(sites, ops, coeff)` dataclass は廃止 (issue #162 で簡素化)。
- **重複項は加算**: 同じ (または順序違いの) サイト集合を持つ term が複数
  渡されれば係数を足し合わせる。`Z_i Z_j = Z_j Z_i` (対角・可換) なので
  `(1.0,(0,1))` と `(0.5,(1,0))` は同一項として `1.5·Z_0 Z_1` に集約。
  Hamiltonian = 項の和という定義どおりの挙動。
- **定数項 (単位演算子)**: `sites = ()` の空タプルは `coeff · I` として全成分
  に係数を加算する (エネルギーオフセット)。
- **`ValueError` 条件**: (a) `n` が 1 以上の整数でない, (b) `sites` に範囲外
  (`< 0` または `>= n`) インデックス, (c) 1 項内でサイトが重複 (相異なるべき;
  `Z_i² = I` の簡約は行わない — タイプミスを黙って吸収しない方針)。

```python
def diag_from_J_h(
    J: np.ndarray,            # (n, n) symmetric, real, J_ii = 0
    h: np.ndarray,             # (n,) real
) -> np.ndarray:
    """
    H_p = -Σ_{i<j} J_ij σ_i σ_j - Σ_i h_i σ_i の対角を作る (σ ∈ {±1}).

    Internal: σ_i を 1 度だけ precompute し 2^n を走査。上三角 i<j のみ使用。
    ValueError: J が非正方/複素/非対称/非ゼロ対角, h の shape 不整合/複素。
    """
```

これらは「`IsingProblem` に対角配列を渡す」設計の上で **オプションの利便性**
として提供。ユーザーは自前で対角ベクトルを作って渡しても良い。いずれも
`2^N` 長を allocate するため `N <= 28` 程度が現実的な上限。

---

