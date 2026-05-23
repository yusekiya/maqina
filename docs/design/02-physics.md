# §2. 物理モデル

### 2.1 全 Hamiltonian

時刻 t での Hamiltonian は schedule で重み付けされた driver + problem:

```
H(t) = A(s(t)) · H_driver + B(s(t)) · H_problem
```

| 項 | 形 | 基底での性質 |
|---|---|---|
| `H_driver` | -Σ_i h_x_i X_i | bit-flip (sparse、N entries per row) |
| `H_problem` | Z 演算子のみで書かれた k-local 多項式 | **Z 基底で対角** |

ここで `s(t)` はアニーリングパラメータの軌道で `s(0) = 0`、`s(T) = 1` を
標準とするが、reverse annealing / pause / quench schedule のため任意の
`s: [0, T] → R` を許容する。`A`, `B` は `s` の関数。

### 2.2 ユーザー入力の表現

問題ハミルトニアン側は、k-local 表現の選択肢 (J 行列、PauliTerm リスト、
QuTiP Qobj 等) を **パッケージ側で扱わない**。代わりに **Z 基底での対角
ベクトル** を 1 つの 1-D 配列として受け取る:

- `H_p_diag: np.ndarray[real, shape=(2**n,)]`
  - 計算基底 `|x⟩` (x ∈ {0,1}^n) における H_problem の固有値を並べた配列。
  - ユーザーが k-local Pauli 多項式から自前で生成する (k=2 でも k≥3 でも、
    任意の Z-string でも、パッケージは中身を区別しない)。
  - インデックス規約: ビット 0 を最下位ビット (LSB) として `x = Σ_i b_i · 2^i`
    にマップ。spin 表記 `σ_i = 1 - 2·b_i` の慣習。

- `h_x: np.ndarray[real, shape=(n,)]`
  - サイト依存横磁場の振幅。`A(s) · h_x_i` が site i の X 係数になる。

これにより:

1. パッケージ側のデータモデルが「対角ベクトル 1 本 + サイト振幅 1 本」に
   集約され、Rust 拡張への FFI 境界が極端に薄くなる。
2. ユーザー側に PauliTerm DSL や J/h API を強制しない (好みのライブラリで
   diagonal を組めば良い)。
3. ヘルパとして `kinema.builders.diag_from_pauli_terms(...)` や
   `diag_from_J_h(...)` を **オプションで** 同梱する (Section 6 参照)。

メモリコスト: `H_p_diag` は 8 bytes × 2^N。`ψ` は 16 bytes × 2^N。
N=24 で diag 128 MB + ψ 256 MB。

### 2.3 初期状態

ユーザーが必ず明示指定する。デフォルトは設けない (reverse annealing 等の
誤用を防ぐため)。

- `psi0: np.ndarray[complex128, shape=(2**n,)]`
  - L2-normalize 済みであることを呼び出し側で保証。パッケージ側ではコンストラクタで
    `‖psi0‖ - 1 < 1e-10` をチェックし、満たさない場合は ValueError。
- ヘルパとして `kinema.initial_states.uniform_superposition(n)`
  (= driver の GS、`|+⟩^⊗N`) を同梱。

---

