# §2. 物理モデル

### 2.1 全 Hamiltonian

#### 2.1.1 旧 API (X-only TFIM, 一様横磁場 / global envelope)

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

#### 2.1.2 新 API (per-site/per-axis 時間依存場, issue #142 Phase C)

旧 API の **driver = X 単体 + global scalar envelope** という制約
(`h_x_i` は静的, 時間依存は global `A(s(t))` のみ; Y / Z 軸の時間依存場は
表現不可) を解消するため, per-site / per-axis に独立な時間関数を持てる形に拡張:

```
H(t) = Σ_i [g_x_i(t)·X_i + g_y_i(t)·Y_i + g_z_i(t)·Z_i] + b(t) · H_p_diag
```

- `g_x_i(t)`, `g_y_i(t)`, `g_z_i(t)` は per-site の callable (length N)。
  `g_y` / `g_z` は `None` で当該軸 skip 可 (Rust 側で real-only SIMD kernel
  に dispatch する fast path)。
- `b(t)` は problem Hamiltonian の global envelope (callable 1 個)。
  問題側 `H_p_diag` は旧 API と同じく **Z 基底で対角な k-local 多項式**。
- callable に振幅は既に組み込み済とする (旧 API ``g_x_i(t) := -a_t(t)·h_x_i``
  相当を新 API で書く場合は ``g_x = [lambda t, hi=hi: -(1 - t/T)*hi for hi in h_x]``
  のように closure で吸収する)。

新 API では `method="trotter"` / `"trotter_suzuki4"` は **未サポート**
(Trotter 経路 `apply_single_mode_axis_i` は実数係数前提 SIMD で組まれて
おり、XYZ 一般化は 2×2 unitary が complex 係数を持つため SIMD kernel の
再設計を要する; 優先度低のため out of scope, 必要時に別 issue 起票)。
``ValueError`` を投げる。

新 API は **旧 API と並存**: 既存スクリプトは移行不要で旧 `Schedule(T, A, B, h_x)`
/ `Schedule.linear` 等を使い続けられる。新 API は `Schedule.from_xyz(...)` で
構築する。driver 入口 (`evolve_schedule_*`) は両 API を `Schedule._eval_stage(t)`
共通 evaluator 経由で `(g_x_arr, g_y_arr_opt, g_z_arr_opt, b_scalar)` に正規化
してから Rust XYZ wraps に dispatch するため, 旧 API は新 API の特化 (g_y=g_z=None
かつ g_x = -a_t · h_x) として実装される。

### 2.2 ユーザー入力の表現

問題ハミルトニアン側は、k-local 表現の選択肢 (J 行列、PauliTerm リスト、
QuTiP Qobj 等) を **パッケージ側で扱わない**。代わりに **Z 基底での対角
ベクトル** を 1 つの 1-D 配列として受け取る:

- `H_p_diag: np.ndarray[real, shape=(2**n,)]` (`IsingProblem` 引数)
  - 計算基底 `|x⟩` (x ∈ {0,1}^n) における H_problem の固有値を並べた配列。
  - ユーザーが k-local Pauli 多項式から自前で生成する (k=2 でも k≥3 でも、
    任意の Z-string でも、パッケージは中身を区別しない)。
  - インデックス規約: ビット 0 を最下位ビット (LSB) として `x = Σ_i b_i · 2^i`
    にマップ。spin 表記 `σ_i = 1 - 2·b_i` の慣習。

- `h_x: np.ndarray[real, shape=(n,)]` (**旧 API では `Schedule` 引数**,
  新 API では callable に組み込み済)
  - 旧 API でのサイト依存横磁場の振幅。`A(s) · h_x_i` が site i の X 係数。
  - issue #142 Phase C で **`IsingProblem` から `Schedule` に移管**された
    (時間発展係数は `Schedule` 側に集約という責任分担)。
  - 新 API (`Schedule.from_xyz`) では callable に振幅を組み込み済の前提なので
    `h_x` 引数自体を持たない。

これにより:

1. パッケージ側のデータモデルが「対角ベクトル 1 本 + サイト振幅 1 本」に
   集約され、Rust 拡張への FFI 境界が極端に薄くなる。
2. ユーザー側に PauliTerm DSL や J/h API を強制しない (好みのライブラリで
   diagonal を組めば良い)。
3. ヘルパとして `maqina.builders.diag_from_pauli_terms(...)` や
   `diag_from_J_h(...)` を **オプションで** 同梱する (Section 6 参照)。
4. `IsingProblem` は問題側静的構造 (`H_p_diag`) のみを持つ pure data container
   として slim 化 (issue #142 Phase C)。時間発展係数は全て `Schedule` に集約。

メモリコスト: `H_p_diag` は 8 bytes × 2^N。`ψ` は 16 bytes × 2^N。
N=24 で diag 128 MB + ψ 256 MB。

### 2.3 初期状態

ユーザーが必ず明示指定する。デフォルトは設けない (reverse annealing 等の
誤用を防ぐため)。

- `psi0: np.ndarray[complex128, shape=(2**n,)]`
  - L2-normalize 済みであることを呼び出し側で保証。パッケージ側ではコンストラクタで
    `‖psi0‖ - 1 < 1e-10` をチェックし、満たさない場合は ValueError。
- ヘルパとして `maqina.initial_states.uniform_superposition(n)`
  (= driver の GS、`|+⟩^⊗N`) を同梱。

---

