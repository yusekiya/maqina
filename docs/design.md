# kryanneal: 設計書 (v0.1 draft)

横磁場イジングモデル (TFIM) の量子ダイナミクスを matrix-free に計算する
Python パッケージ。Krylov 法 (Lanczos) で matvec を介した短時間プロパゲータ
を近似し、Magnus 展開 (commutator-free Magnus, CFM4:2) で時間依存
Hamiltonian の時間発展演算子を近似する。adaptive step は CFM4:2 の
step-doubling Richardson 推定子を PI 制御に流す方式 (詳細 §5.3)。

ユーザー向けインターフェースは Python、行列演算のホットループ
(Lanczos 反復、Magnus 段階指数積、対角 + bit-flip matvec) は Rust 拡張
(`pyo3` 経由) として実装する。状態 ψ ∈ ℂ^{2^N} は NumPy 配列として
保持し、2^N × 2^N の dense / sparse 行列を一度も組み立てない。

---

## 1. ゴールと非ゴール

### ゴール

- **横磁場イジングモデル** (TFIM, k-local を含む) の純粋状態 Schrödinger
  時間発展を高速・省メモリにシミュレート。
- **Matrix-free**: 2^N × 2^N の dense / sparse 行列を一度も組み立てない。
  ψ(t) (length 2^N) と問題ハミルトニアン対角 (length 2^N) だけを保持する。
- **Krylov + Magnus** による短時間プロパゲータ近似。固定ステップ
  (M2 / CFM4:2) に加え、**step-doubling Richardson の adaptive dt
  ドライバ** を提供する (Section 5.3)。
- **Python interface / Rust kernel**: ユーザー向け API は Python、内部の
  ホットループ (Lanczos / CFM4:2) は Rust (`ndarray` + `ndarray-linalg`)。
- **QuTiP との比較**: `sesolve` ベースの参照経路を `benchmarks/` および
  `tests/` 配下に同梱し、近似誤差と性能を同一マシン上で評価可能にする。

### 非ゴール

- **Open system** (Lindblad / 密度行列) は対応しない。純粋状態のみ。
- **D-Wave 等のハードウェア API ラッパ** は対応しない。シミュレータ専用。
- **Variational アルゴリズム** (QAOA 等) はスコープ外。
- **正確対角化 (exact diagonalization) 自体の高速化** はスコープ外
  (瞬時固有状態への投影機能のためにのみ必要範囲で使う)。
- **状態の分散保存 / GPU 対応** は v0.1 では非対応。Future work。

---

## 2. 物理モデル

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
3. ヘルパとして `kryanneal.builders.diag_from_pauli_terms(...)` や
   `diag_from_J_h(...)` を **オプションで** 同梱する (Section 6 参照)。

メモリコスト: `H_p_diag` は 8 bytes × 2^N。`ψ` は 16 bytes × 2^N。
N=24 で diag 128 MB + ψ 256 MB。

### 2.3 初期状態

ユーザーが必ず明示指定する。デフォルトは設けない (reverse annealing 等の
誤用を防ぐため)。

- `psi0: np.ndarray[complex128, shape=(2**n,)]`
  - L2-normalize 済みであることを呼び出し側で保証。パッケージ側ではコンストラクタで
    `‖psi0‖ - 1 < 1e-10` をチェックし、満たさない場合は ValueError。
- ヘルパとして `kryanneal.initial_states.uniform_superposition(n)`
  (= driver の GS、`|+⟩^⊗N`) を同梱。

---

## 3. アーキテクチャ

### 3.1 全体構成

```
┌──────────────────────────────────────────────────────────────┐
│  Python user code                                            │
│    psi0, H_p_diag, h_x, schedule(callable)                   │
│         │                                                    │
│         ▼                                                    │
│  kryanneal.IsingProblem    (problem container)               │
│  kryanneal.Schedule        (annealing schedule)              │
│  kryanneal.QuantumAnnealer (driver: run / advance_to)        │
│  kryanneal.AnnealingSimulator (step-wise stateful API)       │
│         │                                                    │
│         ▼ Python matvec callback                             │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  kryanneal._rust  (PyO3 extension, maturin-built)      │  │
│  │    lanczos_propagate  (matrix-free Lanczos M-step)     │  │
│  │    cfm4_step          (CFM4:2 1 step)                  │  │
│  │    m2_midpoint_step   (M2 中点則 1 step)               │  │
│  │    cfm4_step_with_*_estimate (embedded error 推定子)   │  │
│  │    apply_h_kryanneal  (matvec 本体; bit-flip + 対角積) │  │
│  └────────────────────────────────────────────────────────┘  │
│         ▼                                                    │
│  QuantumResult / Trajectory dataclass                        │
└──────────────────────────────────────────────────────────────┘
```

### 3.2 役割分担

| 層 | 責務 |
|---|---|
| Python (`src/kryanneal/`) | 公開 API、`IsingProblem` / `Schedule` / `Annealer` の組み立て、結果オブジェクト、瞬時固有状態への投影、入力検証、QuTiP 比較ヘルパ |
| Rust (`rust/src/`) | Lanczos ループ (`lanczos_propagate`)、CFM4:2 / M2 / Richardson 推定子の段階指数積、行列分解 (`ndarray-linalg`)、BLAS 経由の Level-1/2 ops |
| Rust matvec (`apply_h_kryanneal`) | bit-flip + 対角積を組み合わせた matvec。Python callback を介さず Rust 内で完結 |

#### matvec を Rust 側に置く理由

Matrix-free Lanczos の一般的な実装パターンは「matvec を Python callback
として PyO3 越しに毎 step 呼び戻す」形だが、本パッケージは matvec 自体を
Rust 内に閉じ込める。TFIM の matvec は以下 2 成分だけで尽きるためである:

- diagonal 部分: `(B(t) · H_p_diag) * ψ` (要素積)
- bit-flip 部分: 各サイト i について `psi[k ^ (1 << i)] * (-A(t) · h_x_i)` の和

`H_p_diag` (`&[f64]`) と `h_x` (`&[f64]`) を Rust にコピー / 借用で渡せば、
matvec を Rust 内 closure として組み立てられ、Lanczos 1 step あたりの
FFI 呼出は 0 回になる。schedule で変わるのは A(t), B(t) のスカラーだけ
なので step 入口で渡せば十分。

これは TFIM の Hamiltonian が「diag + 既知形の sparse driver」という閉じた
構造を持つから可能な最適化で、汎用 callback パターン (例: 連続変数系で
per-mode GEMM が必要な場合) に対し、FFI 越境コスト削減 + Python GIL 解放
区間の拡大という利点がある。

### 3.3 ディレクトリレイアウト

[maturin 公式ドキュメント](https://www.maturin.rs/project_layout) が
推奨する **mixed Rust/Python project の標準形** に準拠する。
具体的には:

- `Cargo.toml` と `src/lib.rs` (Rust 側) を **プロジェクトルート**に置く
- Python ソースは `python/kryanneal/` 配下に置き、`pyproject.toml` で
  `python-source = "python"` を宣言する
- `.pyi` 型スタブは対応する `.py` と同じディレクトリに並べる
- `py.typed` (PEP 561 マーカ) を `python/kryanneal/` に置く

この形が docs で「common `ImportError` pitfall」と呼ばれる事象
([PyO3/maturin#490](https://github.com/PyO3/maturin/issues/490)) を
避けるためにわざわざ推奨されている、ということを 7.6 節で詳述する。

`python-source = "src"` (Python を `src/<pkg>/`) + `manifest-path =
"rust/Cargo.toml"` (Rust を `rust/`) のような変形レイアウトも技術的には
可能だが docs 推奨形ではないため、本パッケージは標準形に従う。

```
kryanneal/
├── pyproject.toml              # uv + maturin, python-source = "python"
├── Cargo.toml                  # ← Rust crate ルート (maturin 標準位置)
├── README.md
├── CLAUDE.md                   # プロジェクトガイド
├── docs/
│   ├── design.md               # 本ファイル
│   ├── testing.md              # /test skill 用
│   └── benchmarks.md
├── src/                        # ← Rust ソース (maturin 標準位置)
│   ├── lib.rs                  # PyO3 エントリポイント (#[pymodule] fn _rust)
│   ├── matvec.rs               # apply_h_kryanneal (bit-flip + diag)
│   ├── krylov.rs               # lanczos_propagate (ndarray + ndarray-linalg)
│   ├── cfm4.rs                 # CFM4:2 / M2 / Richardson
│   └── blas.rs                 # 内積 / axpy / nrm2 / scal ラッパ
├── python/                     # ← Python ソース (python-source = "python")
│   └── kryanneal/
│       ├── __init__.py         # 公開 API
│       ├── __init__.pyi        # 自動生成 stub (.py と同所; maturin が wheel 同梱)
│       ├── py.typed            # PEP 561 マーカ
│       ├── problem.py          # IsingProblem
│       ├── problem.pyi
│       ├── schedule.py         # Schedule
│       ├── schedule.pyi
│       ├── annealer.py         # QuantumAnnealer / AnnealingSimulator
│       ├── annealer.pyi
│       ├── krylov.py           # Python リファレンス Krylov ドライバ
│       ├── krylov.pyi
│       ├── eigenstates.py      # 瞬時固有状態への投影
│       ├── eigenstates.pyi
│       ├── builders.py         # PauliTerm → diag、J/h → diag ヘルパ
│       ├── builders.pyi
│       ├── initial_states.py   # |+⟩^N など便利初期状態
│       ├── initial_states.pyi
│       ├── result.py           # QuantumResult, Trajectory dataclass
│       ├── result.pyi
│       ├── reference_qutip.py  # QuTiP sesolve 経由のリファレンス実装
│       ├── reference_qutip.pyi
│       └── _rust.*.so          # maturin develop でここに配置される
├── tools/
│   └── gen_api_stubs.py        # 公開 API の .pyi 生成
├── tests/                      # Python 統合テスト
│   ├── test_problem.py
│   ├── test_schedule.py
│   ├── test_krylov.py
│   ├── test_cfm4.py
│   ├── test_richardson.py
│   ├── test_annealer.py
│   ├── test_eigenstates.py
│   ├── test_builders.py
│   └── test_reference_qutip.py
└── benchmarks/
    ├── README.md
    ├── bench_per_step.py
    ├── bench_blas_compare.py
    ├── bench_vs_qutip.py
    └── results/                # gitignored
```

`pyproject.toml` の `[tool.maturin]` セクションは以下のようになる:

```toml
[build-system]
requires = ["maturin>=1.7,<2.0"]
build-backend = "maturin"

[tool.maturin]
python-source = "python"        # ← 標準推奨 (ImportError 回避)
module-name = "kryanneal._rust" # Rust 側 #[pymodule] fn _rust の Python パス
features = ["extension-module"]
profile = "production"
strip = true
```

`manifest-path` は **指定しない** (デフォルトでルートの `Cargo.toml` を見る)。
`src/lib.rs` 側で `#[pymodule] fn _rust(...)` を定義すれば `module-name`
の `_rust` と整合する。

---

## 4. Python API

### 4.1 公開シンボル

```python
from kryanneal import (
    IsingProblem,        # 問題定義 (n, H_p_diag, h_x)
    Schedule,            # アニーリングスケジュール
    QuantumAnnealer,     # one-shot 実行ドライバ
    AnnealingSimulator,  # step-wise stateful API
    QuantumResult,
    Trajectory,
)
from kryanneal.builders import diag_from_pauli_terms, diag_from_J_h
from kryanneal.initial_states import uniform_superposition
from kryanneal.eigenstates import instantaneous_eigenstates
```

### 4.2 `IsingProblem`

```python
@dataclass(frozen=True)
class IsingProblem:
    n: int                       # スピン数
    H_p_diag: np.ndarray         # shape (2**n,), real, contiguous
    h_x: np.ndarray              # shape (n,), real, contiguous

    def __post_init__(self) -> None:
        # shape / dtype / 非NaN チェック。h_x の長さが n か確認。
        ...

    @classmethod
    def from_pauli_terms(cls, n: int, terms: list[PauliTerm], h_x: np.ndarray) -> "IsingProblem":
        """ヘルパ: PauliTerm リストから H_p_diag を構築 (内部で builders.diag_from_pauli_terms)."""
        ...

    @classmethod
    def from_J_h(cls, J: np.ndarray, h: np.ndarray, h_x: np.ndarray) -> "IsingProblem":
        """ヘルパ: 2-local J, h から H_p_diag を構築."""
        ...
```

`PauliTerm` は dataclass:

```python
@dataclass(frozen=True)
class PauliTerm:
    sites: tuple[int, ...]   # 作用サイト
    ops: str                 # "Z", "ZZ", "ZZZ", ... (Z のみ許容)
    coeff: float
```

builders は **Z のみ**を許容 (off-diagonal Pauli を渡されたら ValueError)。
これは「H_problem は必ず Z 基底で対角」という設計契約を API 表面で守るため。

### 4.3 `Schedule`

```python
class Schedule:
    """A(s), B(s) と s(t) を保持。s(0)=0 / s(T)=1 を強制せず、callable で柔軟に."""

    def __init__(
        self,
        T: float,                  # 総アニーリング時間
        A: Callable[[float], float],  # A(s) — 通常 1 - s
        B: Callable[[float], float],  # B(s) — 通常 s
        s: Callable[[float], float] | None = None,  # s(t) — None なら t/T (線形)
    ): ...

    @classmethod
    def linear(cls, T: float) -> "Schedule":
        """A(s) = 1 - s, B(s) = s, s(t) = t/T."""
        ...

    @classmethod
    def from_callable(
        cls, T: float, A: Callable, B: Callable, s: Callable | None = None
    ) -> "Schedule": ...

    def coeffs_at(self, t: float) -> tuple[float, float]:
        """(A(s(t)), B(s(t))) を返す。Annealer / Rust 側に渡すスカラー."""
        ...
```

実装注: `s(t)` を内部で grid 評価し配列で持つキャッシュを用意するかは
adaptive 経路で重要 (Section 5.3)。MVP では callable のまま per-step 評価。

### 4.4 `QuantumAnnealer`

one-shot 実行:

```python
class QuantumAnnealer:
    def __init__(
        self,
        problem: IsingProblem,
        schedule: Schedule,
        psi0: np.ndarray,            # ユーザー必須
    ): ...

    def run(
        self,
        method: Literal["m2", "cfm4", "cfm4_adaptive_richardson"] = "cfm4",
        n_steps: int | None = None,    # 固定ステップ時 (m2 / cfm4)
        krylov_dim: int = 24,
        krylov_tol: float = 1e-12,
        atol: float | None = None,     # adaptive 時の local error 許容値
        rtol: float | None = None,
        dt_init: float | None = None,  # adaptive 時の初期 dt 提案
        store_states: bool = False,    # True なら全 step の ψ を保持
        store_times: list[float] | None = None,  # 指定時刻のスナップショット
        observables: dict[str, "Observable"] | None = None,  # 期待値を時系列で記録
    ) -> QuantumResult: ...

    def create_simulator(
        self, method: Literal["m2", "cfm4", "cfm4_adaptive_richardson"] = "cfm4",
        **method_kwargs
    ) -> "AnnealingSimulator": ...
```

返り値 `QuantumResult`:

```python
@dataclass(frozen=True)
class QuantumResult:
    psi_final: np.ndarray                 # shape (2**n,) complex128
    times: np.ndarray | None              # 中間時刻 (store_states 時)
    states: np.ndarray | None             # shape (len(times), 2**n)
    probabilities: np.ndarray             # |psi_final|^2、shape (2**n,)
    observables: dict[str, np.ndarray]    # name -> shape (len(times),) 実数
    success: bool
    method: str
    n_steps_actual: int                   # adaptive 経路では実 step 数
    n_matvec: int                         # 累積 matvec 呼出
```

### 4.5 `AnnealingSimulator`

任意の中間時刻まで進めて状態を取り出し、観測量で測定して続けて発展させる
用途の step-wise stateful API:

```python
class AnnealingSimulator:
    def __init__(self, ...): ...

    @property
    def t(self) -> float: ...
    @property
    def psi(self) -> np.ndarray: ...
    @property
    def n_matvec(self) -> int: ...

    def step(self, dt: float) -> None:
        """1 step (固定 dt) 進める。adaptive 経路では dt は提案値."""
        ...

    def advance_to(self, t_target: float) -> None:
        """t_target まで進める."""
        ...

    def measure(self, observable: "Observable") -> float:
        """現在 ψ で期待値 <ψ|O|ψ> を計算 (実数)."""
        ...
```

### 4.6 観測量 `Observable`

Z 基底対角に限定 (内積 1 回で済む):

```python
class Observable:
    """Z 基底対角の Hermitian 演算子。<ψ|O|ψ> = Σ_k diag[k] · |ψ[k]|^2."""

    def __init__(self, diag: np.ndarray): ...

    @classmethod
    def magnetization(cls, n: int, axis: Literal["z"] = "z") -> "Observable": ...
    @classmethod
    def ising_energy(cls, problem: IsingProblem) -> "Observable":
        """H_problem 自体を観測量化."""
        return cls(problem.H_p_diag)
```

X / Y 期待値は (a) `ψ` 自体への bit-flip を一度噛ませる必要があり遅い、
(b) アニーリングのユースケースでは稀、という理由で v0.1 では対応しない
(`builders` で diag を吐ければ十分)。Future work。

### 4.7 瞬時固有状態への投影

```python
def instantaneous_eigenstates(
    problem: IsingProblem,
    schedule: Schedule,
    t: float,
    k: int = 8,                    # 取得する低位固有状態数
    method: Literal["lanczos", "exact"] = "lanczos",
) -> tuple[np.ndarray, np.ndarray]:
    """
    瞬時 H(t) の下位 k 固有値・固有状態を返す.

    Returns
    -------
    eigvals : (k,) real
    eigvecs : (2**n, k) complex128
    """
```

実装方針:

- `method="lanczos"` (default): `ndarray-linalg` の Lanczos / Arnoldi 互換
  ルートで下位 k 固有値を反復。matvec は本体と共有 (`apply_h_kryanneal`)。
- `method="exact"`: 小規模問題 (`n <= 12`) 向け、dense `eigh` (検証用)。

ユーザーは `ψ(t)` を `eigvecs` に内積して amplitude を出す:
`amps = eigvecs.conj().T @ psi(t)`。

---

## 5. 数値カーネル

### 5.1 matvec: `apply_h_kryanneal`

時間依存スカラー `(A_t, B_t)` を渡し、Rust 内で `(A_t · h_x, B_t · H_p_diag)`
を組み合わせた matvec を 1 回適用する。Python 越境は **Lanczos の 1 step
あたり呼ばない**。

擬似コード (Rust 側):

```rust
/// y = A·H_driver·v + B·H_p_diag·v
///
/// H_driver = -Σ_i h_x_i X_i (-Σ X_i の inhomogeneous 拡張).
fn apply_h(
    v: &[Complex64],
    y: &mut [Complex64],
    h_x: &[f64],          // length n
    h_p_diag: &[f64],     // length 2^n
    a_t: f64,
    b_t: f64,
    n: usize,
) {
    let dim = 1usize << n;
    // diagonal 部分: y[k] = B·H_p[k]·v[k]
    for k in 0..dim {
        y[k] = Complex64::new(b_t * h_p_diag[k], 0.0) * v[k];
    }
    // bit-flip 部分: y[k] += -A · h_x_i · v[k ^ (1<<i)]
    for i in 0..n {
        let coeff = -a_t * h_x[i];
        let mask = 1usize << i;
        // ※ inner loop は cache-blocking + SIMD 余地あり (Section 7.2 参照)
        for k in 0..dim {
            y[k] += Complex64::new(coeff, 0.0) * v[k ^ mask];
        }
    }
}
```

性能考慮 (Section 7 で詳述):

- diagonal 部分は cblas `zdscal` + 要素積を unrolled inner loop で。
- bit-flip 部分は `i` 外側 / `k` 内側で連続アクセスにできる: ビット i に
  ついて k と k^mask のペアを `mask=1<<i` で 2 元ストライドにより列挙。
  i=0 で stride 1、i=1 で stride 2、... と level-by-level に走査することで
  TLB / L2 ヒット率を上げる古典的テクニック (state-vector simulator の
  X-gate pass と同一)。

### 5.2 Lanczos: `lanczos_propagate`

`exp(-i dt H) ψ` を `m` 次元 Lanczos + 三重対角固有分解で計算する
matrix-free 短時間プロパゲータ (Park-Light 1986)。実装方針:

- 行列演算は **ndarray + ndarray-linalg** に統一 (ユーザー要件)。
- 三重対角の対称固有分解は `ndarray_linalg::Eigh` (LAPACK `*syevd` 経由) を使用。
- ベクトル長 dim 依存 ops は `cblas` クレート経由のラッパで書く
  (BLAS feature on/off を Cargo features で切替; §7.1, §7.4)。
- Full re-orthogonalization (Gram-Schmidt 2-pass) を採用。
- 部分空間打切り条件 `β_k < tol` で `m_eff = k+1`。
- 終端再構成: `psi_new = V[:, :m_eff] @ c` を `zgemv` 1 回で。

API:

```rust
pub(crate) fn lanczos_propagate<F>(
    mut matvec: F,
    psi: &[Complex64],
    dt: f64,
    m: usize,
    tol: f64,
) -> Result<Vec<Complex64>, KryError>
where
    F: FnMut(&[Complex64], &mut [Complex64]),  // y = H · v
```

`matvec` を closure に取り、`cfm4.rs` から `(c_drv, c_diag)` を畳み込んだ
線形結合版 closure を渡せるようにする (CFM4:2 の各 stage 用の Hamiltonian
を 1 つの線形結合 matvec として表現する経路)。これにより CFM4:2 の各
ステージで matvec 呼出は m 回のみ。

### 5.3 Magnus / プロパゲータ

以下 3 種のプロパゲータを提供する:

#### M2 (中点則 1 step) — `m2_midpoint_step`

```
U(t+dt, t) ≈ exp(-i dt · H(t + dt/2))
```

中点で H をフリーズして `lanczos_propagate` 1 回。LTE ~ O(dt^3)。

#### CFM4:2 (4 次 commutator-free Magnus) — `cfm4_step`

Alvermann-Fehske (2011) の 4 次 commutator-free Magnus:

```
U(t+dt, t) ≈ exp(-i dt · B_2) · exp(-i dt · B_1)
```

Gauss-Legendre 2 点求積ノード:

```
c_1 = 1/2 - √3/6 ≈ 0.21132486540518708
c_2 = 1/2 + √3/6 ≈ 0.78867513459481292
```

線形結合係数:

```
a_high = 1/4 + √3/6 ≈ 0.43301270189221935
a_low  = 1/4 - √3/6 ≈ 0.06698729810778066
a_high + a_low = 1/2
```

ステージごとに以下の Hamiltonian で Lanczos を 1 回ずつ呼ぶ:

```
H_1 = H(t + c_1·dt),    H_2 = H(t + c_2·dt)
B_1 = a_high · H_1 + a_low  · H_2
B_2 = a_low  · H_1 + a_high · H_2
```

kryanneal では `H(t) = A(s(t))·H_driver + B(s(t))·H_problem` の構造を持つ
ため、各 stage で必要なのは **driver / diag の前置係数 1 組ずつ** だけ:

```
stage 1 :  c_drv = a_high·A(s_1) + a_low·A(s_2)
           c_diag = a_high·B(s_1) + a_low·B(s_2)
stage 2 :  c_drv = a_low ·A(s_1) + a_high·A(s_2)
           c_diag = a_low ·B(s_1) + a_high·B(s_2)
```

ここで `s_i = s(t + c_i·dt)`。Rust 側 `apply_h_kryanneal` に
`(c_drv, c_diag)` のスカラー 2 つを渡せば 1 stage の matvec が組める
(線形結合係数を 1 つに畳み込む経路、§5.2 末尾参照)。Lanczos 2 回 / step、
LTE ~ O(dt^5)。

#### CFM4:2 + step-doubling Richardson — `cfm4_step_with_richardson_estimate`

CFM4:2 を full-step (dt) と half-step×2 (dt/2 + dt/2) で **同一入口 ψ**
から走らせ、

```
err = ‖ψ_full - ψ_h2‖ ≈ (1 - 1/16) · C_4 · dt^5
```

を CFM4:2 自身の LTE 推定値として返す。per-step matvec は **12m**
(full 4m + half×2 × 4m)。M2 embedded 版より 2 オーダ高精度なので
smooth schedule では許容 dt を 1〜2 桁伸ばせる。

オプション `extrapolate=True` で Richardson 外挿:
`ψ_acc = (16 · ψ_h2 - ψ_full) / 15` (実効 6 次精度)。

#### PI controller / adaptive ドライバ

両 estimator を共通仕様の PI controller (Python 側ループ) が駆動する。
RK45 / Dormand-Prince 系の embedded estimator と同型の式・既定値を採用:

```
dt_next = dt · safety · (tol_step / err)^(1/(p+1))
```

| 推定子 | p | 指数 `1/(p+1)` | 該当 estimator |
|---|---|---|---|
| M2 embedded | 2 | 1/3 | `cfm4_step_with_m2_estimate` |
| Richardson  | 4 | 1/5 | `cfm4_step_with_richardson_estimate` |

既定パラメータ (`evolve_schedule_adaptive_*` の `*` パラメータ):

```
m            = 24            # Lanczos 部分空間次元
krylov_tol   = 1e-12         # Lanczos 早期打切閾値 (β_k < tol)
tol_step     = 1e-8          # accept 判定の局所誤差閾値
dt0          = 0.5           # 初期 dt
dt_min       = 1e-4          # 最小 dt (ここまで縮めると err 無視で accept)
dt_max       = 10 · dt0      # 既定値 (None 渡し時に解決)
safety       = 0.9           # PI 安全係数
growth_max   = 4.0           # 1 step での dt 拡大率上限
max_rejects  = 50            # 同一 step での連続 reject 上限 (超過で RuntimeError)
```

ループ本体 (擬似コード):

```python
while t < t1:
    dt_try = min(dt, t1 - t, next_save - t)   # 終端 / 観測時刻にクランプ
    psi_new, err = step_with_estimate(psi, dt_try, ...)
    accept = (err <= tol_step) or (dt_try <= dt_min)
    if accept:
        psi = psi_new
        t += dt_try
        if err <= 1e-30:                       # 0 近傍ガード
            dt_next = dt_try * growth_max
        else:
            dt_next = dt_try * safety * (tol_step / err) ** (1/(p+1))
        dt_next = min(dt_next, dt_try * growth_max, dt_max)
        dt_next = max(dt_next, dt_min)
        dt = dt_next
        n_consecutive_rejects = 0
    else:
        n_consecutive_rejects += 1
        if n_consecutive_rejects > max_rejects:
            raise RuntimeError(...)
        dt = max(dt_try * 0.5, dt_min)         # reject 時は半減
```

Reject 時に schedule node `t + c_i · dt` は新しい dt で再評価する
(dt 依存ノードのため)。`save_tlist` が指定された場合は次の観測時刻
`t_obs` でも step 境界が揃うよう `dt_try` をクランプし、accept 後の `ψ`
を `states_at_save` に `ψ.copy()` で格納する。PI 状態 (`dt_next`) は観測境界
を跨いで連続持ち越しなので chain 呼出より再ウォームアップコストが少ない。

Python 側公開関数:

```python
# python/kryanneal/krylov.py

def evolve_schedule_adaptive_m2(
    problem, schedule, psi0, t0, t1, *,
    m=24, krylov_tol=1e-12, tol_step=1e-8,
    dt0=0.5, dt_min=1e-4, dt_max=None,
    safety=0.9, growth_max=4.0, max_rejects=50,
    save_tlist=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, list[np.ndarray]]:
    """(psi_final, t_history, dt_history, n_rejects, states_at_save)"""

def evolve_schedule_adaptive_richardson(
    problem, schedule, psi0, t0, t1, *,
    m=24, krylov_tol=1e-12, tol_step=1e-8,
    dt0=0.5, dt_min=1e-4, dt_max=None,
    safety=0.9, growth_max=4.0, max_rejects=50,
    richardson_extrapolate=False,
    save_tlist=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, list[np.ndarray]]: ...
```

`QuantumAnnealer.run(method="cfm4_adaptive_richardson", ...)` はこれを
内部で呼ぶ薄いラッパ。

### 5.4 Python リファレンス実装

`python/kryanneal/krylov.py` に `_python_lanczos_propagate` /
`_python_cfm4_step` 等を **pure NumPy** で実装し、Rust 拡張がビルドできない
環境でも silently fallback する設計とする。

- 等価性テスト: tight tol で `rel < 1e-13` 一致。
- 開発時のデバッグ・教育用途にも有用。

---

## 6. ビルダー (k-local → 対角ベクトル)

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

## 7. Rust 拡張

### 7.1 Crate 構成

- `pyo3 = "0.28"`
- `numpy = "0.28"`
- `ndarray = "0.16"` ←
- `ndarray-linalg = "0.17"` ← LAPACK 経由の三重対角固有分解 / 行列指数 (必要なら)
- `num-complex = "0.4"`
- `cblas = "0.5"` (optional, BLAS feature)
- `blas-src = "0.12"`:
  - macOS: `accelerate` feature
  - Linux: `openblas` feature (system OpenBLAS)

**ndarray-linalg backend 注記** (Phase 1 で決定する設計判断):

`ndarray-linalg 0.17` は LAPACK backend を Cargo feature で 1 つ選ぶ必要
がある (`openblas-system` / `openblas-static` / `netlib-system` /
`netlib-static` / `intel-mkl-system` / `intel-mkl-static`)。**macOS Accelerate
への native 対応はない**ため、CFM4:2 / Lanczos 内で必要となる
**m × m (m ~ 24) の対称三重対角固有分解**の処理経路として 3 案が考えられる:

| 案 | macOS | Linux | 備考 |
|---|---|---|---|
| A. `ndarray-linalg` + `openblas-system` 統一 | `brew install openblas` 必要 | `libopenblas-dev` 必要 | `blas-src` の方も全プラットフォーム OpenBLAS に揃えるか、Accelerate を残すなら backend が分裂する |
| B. `ndarray` のみ + hand-rolled 三重対角 QR | Accelerate 維持 | OpenBLAS 維持 | m ≤ 24 なので scalar Rust で十分速い; LAPACK 依存を切れる |
| C. `ndarray-linalg` を捨て、当該用途のみ `nalgebra::SymmetricEigen` | Accelerate 維持 | OpenBLAS 維持 | ユーザー要件の "ndarray-linalg 使用" には反するため非推奨 |

ユーザー要件は **ndarray + ndarray-linalg** なので 案 A を第一候補とする。
ただし macOS Accelerate を失うと Level-1/2 BLAS の性能が大きく下がる可能性
があるため、**Phase 1 で実機ベンチを取って判断する** (`blas-src` 側は
Accelerate、`ndarray-linalg` 側は openblas-system という混在も技術的には
可能だが、複数 BLAS pool 同居の `set_blas_threads` 制御が複雑化する)。

Phase 1 の意思決定基準:

- 案 A (`openblas-system` 統一) を試し、macOS で Apple Accelerate に対する
  性能ペナルティが `bench_per_step.py` 比で 2x 以内なら採用
- それを超える劣化が出るなら 案 B (hand-rolled tridiag QR) にフォールバック
  し、`ndarray-linalg` 依存自体を `[dev-dependencies]` (テスト検証用) に
  落とす

### 7.2 BLAS 経由のホットパス

複素ベクトル Level-1 / Level-2 BLAS の inline ヘルパ群を `src/blas.rs`
に切り出す:

- `norm2` → `cblas::dznrm2`
- `dot_conj` → `cblas::zdotc_sub`
- `axpy` → `cblas::zaxpy`
- `scal_real` → `cblas::zdscal`
- `gemv_col_major_no_alpha` → `cblas::zgemv`

加えて kryanneal 固有のホットパス:

- bit-flip pass: site i について 2 元ストライドで stripe-by-stripe に
  進めるカスタムループ。SIMD ((AVX2 / NEON) や level-by-level スワップ
  パターンは v0.1 ではナイーブ実装、bench で頂上を確認してから最適化。
- 対角積: `zdscal` 系の Level-1 BLAS 風に書くのが自然だが、対角が **実数
  ベクトル** なので、`cblas::zdscal` を per-element ループにかける形 (`zaxpy`
  代用) より、Rust scalar loop の方が SIMD される可能性が高い。bench で
  決める。

### 7.3 `apply_h_kryanneal` の Python 公開

Rust 内では closure として完結させるが、Python リファレンス / テスト
比較のため **公開関数として 1 つ export** する:

```rust
#[pyfunction]
fn apply_h_kryanneal_py(
    py: Python<'_>,
    v: PyReadonlyArray1<Complex64>,
    h_x: PyReadonlyArray1<f64>,
    h_p_diag: PyReadonlyArray1<f64>,
    a_t: f64,
    b_t: f64,
) -> PyResult<Py<PyArray1<Complex64>>>;
```

Python 側で `apply_h_kryanneal_py(...)` と `(A · h_x ⊗ X) + (B · diag)` を
qutip / 手書きで比べる単体テストを書く。

### 7.4 Cargo features

```toml
[features]
default = ["blas"]
blas = ["dep:cblas", "dep:blas-src"]
extension-module = ["pyo3/extension-module"]
```

`extension-module` を default に入れないのは、`cargo test` で test binary
が `libpython` シンボル未解決になるため。maturin 経由の wheel ビルドでは
`pyproject.toml` の `[tool.maturin] features` で明示的に有効化し、
普通の `cargo test` / `cargo build` では無効のままにする。

### 7.5 `__has_blas__` warning

`_rust.__has_blas__` を Python 側に export し、`kryanneal.krylov` の
import 時に False なら `RuntimeWarning` を発する。これにより scalar
fallback build (BLAS 無し) に気付かず長時間ベンチを回す事故を防ぐ。

### 7.6 maturin レイアウト上の注意点 (PyO3 stub の歴史的問題)

PyO3 + maturin 構成では過去に **型 stub と拡張モジュールの解決順序**で
詰まる事例が複数報告されていた。現在の maturin (≥ 1.0 系) では大部分が
解消されているが、設計時点で踏むべきは以下:

1. **`python-source` を `"python"` に設定する**

   [maturin#490](https://github.com/PyO3/maturin/issues/490) で報告されている
   `ModuleNotFoundError: No module named 'pkg.pkg'` 系の事故を避ける。
   プロジェクトルートに `kryanneal/` ディレクトリと拡張モジュール `.so`
   が同居していると、CWD = リポジトリルートで Python を起動した際に
   `kryanneal/` のソースディレクトリが先に解決され、隣の `_rust.so` が
   見つからない、という症状が出る。Python ソースを `python/kryanneal/` に
   分離することで CWD と無関係に拡張がロードされる。

2. **`.pyi` は `.py` と同じディレクトリに並べる**

   maturin docs の Project Layout 節は

   > "additional files in the Python source dir (but not in `.gitignore`)
   > will be automatically included in the build outputs"

   と明記しており、`python/kryanneal/*.pyi` は wheel に自動同梱される。
   `[tool.maturin]` 側に `include` 指定を足す必要はない。

3. **`py.typed` を置く (PEP 561)**

   `python/kryanneal/py.typed` を空ファイルで作成。これがないと
   `mypy` / `ty` / `pyright` は wheel 同梱の `.pyi` を発見しない。

4. **`.gitignore` には拡張モジュールのみ**

   `python/kryanneal/_rust*.so` (`maturin develop` 配置先) を `.gitignore`。
   `.pyi` は **コミットする** (自動生成だが diff レビューで API 変更を
   検知できるようにするため)。

5. **古い情報の取り扱い**

   - [maturin#771 (stub が wheel に入らない)](https://github.com/pyo3/maturin/issues/771)
     は古い挙動の報告。現行では Python source dir 配下の `.pyi` は
     自動同梱されるので、本設計では追加対処は不要。
   - [maturin#885 (Python source が wheel に入らない)](https://github.com/PyO3/maturin/issues/885)
     は `python-source` を設定しないと発火する症状。本設計では
     `python-source = "python"` を最初から宣言するため該当しない。

   実機ビルドで `unzip -l dist/kryanneal-*.whl | grep -E '\.(py|pyi|so)$'`
   を回し `.py` / `.pyi` / `.so` が揃って入っていることを CI で
   smoke test するのが堅い (将来 maturin の挙動が変わっても気付ける)。

これらは Phase 1 の最初のビルドが通った時点で `tests/test_packaging.py`
として固定する想定。

---

## 8. QuTiP 比較

### 8.1 位置付け

QuTiP は **本パッケージの提供する機能ではなく、近似誤差評価および性能
比較のためのリファレンス**。`tests/` および `benchmarks/` 配下に閉じ込め、
公開 API には載せない。`pyproject.toml` の dev dependency として宣言。

### 8.2 リファレンス実装 `reference_qutip.py`

```python
# src/kryanneal/reference_qutip.py
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
  `res_qutip.states[-1]` と `kryanneal の psi_final` の fidelity を比較。

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
  - kryanneal M2 / CFM4 / Richardson の wall time
  - 最終 fidelity
- を CSV + markdown で `benchmarks/results/<YYYYMMDD-HHMMSS>/` 配下に出力。
- 性能改善の主張は **同一マシン上の before/after** で示す (§10 のベンチ規約
  を参照)。

---

## 9. テスト戦略

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

## 10. ベンチマーク戦略

ベンチ規約:

- 性能改善の主張は **同一マシン上の before/after**。CPU / BLAS / NumPy /
  熱状態が揃った状態で取る。別マシンの絶対値表と並べて「○○× 速くなった」
  と主張しない。
- ベンチスクリプトは `benchmarks/bench_<対象>.py` 命名規約、argparse CLI、
  `benchmarks/results/<YYYYMMDD-HHMMSS>/` への CSV + markdown 出力。
- 同一マシン上で取った datapoint を `benchmarks/README.md` に書き戻す
  ときは、使用ハード (機種 / チップ / メモリ / OS / NumPy / BLAS backend)
  を節タイトルで明示する。

最初に用意するベンチ:

- `bench_per_step.py`: M2 / CFM4 / Richardson の per-step wall time
  (n を sweep)。
- `bench_blas_compare.py`: BLAS feature on/off の同一マシン比較。
- `bench_vs_qutip.py`: Section 8.3。

### 10.1 期待される性能特性 (推定)

- N = 20 (dim 2^20 ≈ 10^6): ψ 16 MB + H_p_diag 8 MB ≈ 24 MB。matvec の
  bit-flip pass が支配 (N × 2^N ≈ 2 × 10^7 ops/step × 4 m matvec/step)。
  Apple Accelerate を使った dense GEMV を凌ぐ matrix-free 性能を目指す。
- N = 24: ψ 256 MB + H_p_diag 128 MB ≈ 384 MB。M2 メモリでも単機で
  動かせる。CFM4:2 ステージで一時バッファ数本必要なので 1 GB 弱まで。
- N = 26 以上: ψ 1 GB+、shared-mem 単機の限界。GPU / 分散版は future work。

---

## 11. 開発・ビルド基盤

- パッケージマネージャ: `uv`、Python `>=3.13`
- ビルド: `maturin` (Rust 拡張 `kryanneal._rust` を PyO3 経由でビルド)
- Lint: `ruff` / 型: `ty`
- 主要依存: `numpy>=2.4.2`, `threadpoolctl>=3.0`
- dev 依存: `pytest>=8.3`, `qutip>=5.2.3`, `pre-commit>=4.0`, `ruff`, `ty`
- API stubs: `tools/gen_api_stubs.py` で `.py` から PEP 484 stub 自動生成。
  pre-commit hook と `.claude/rules/api-stubs-sync.md` で drift 防止する
  二段運用 (人間編集も hook が拾う)。
- BLAS 多プロセス制御: `set_blas_threads(n)` /
  `available_blas_threads()` を `__init__.py` に export
  (`threadpoolctl.threadpool_limits` を BLAS API 単位で呼び出す
  ラッパで、numpy/scipy bundled + system の OpenBLAS pool 同居問題に対処)。

---

## 12. 段階リリース計画

### Phase 1: MVP (~v0.1)

- `IsingProblem`, `Schedule`, `QuantumAnnealer.run(method="m2")` のみ
- Rust 拡張: `apply_h_kryanneal`, `lanczos_propagate`, `m2_midpoint_step`
- Python リファレンス (`_python_*`) との等価性テスト
- 小規模 QuTiP 比較テスト

### Phase 2: CFM4:2 (~v0.2)

- `cfm4_step`, `method="cfm4"` 経路
- 線形結合 callback 形式 (§5.2 末尾) で per-step matvec を 4m → 2m に削減

### Phase 3: Adaptive (~v0.3)

- `cfm4_step_with_m2_estimate` (embedded M2 error)
- `cfm4_step_with_richardson_estimate` (step-doubling Richardson)
- Python 側 PI controller driver
- `method="cfm4_adaptive_richardson"`

### Phase 4: Simulator & Observables (~v0.4)

- `AnnealingSimulator`
- `Observable` クラス、観測量時系列記録
- `instantaneous_eigenstates`

### Phase 5: 仕上げ

- ベンチマーク完備、QuTiP 大規模比較
- BLAS feature on/off 数値検証 CI
- ドキュメント、Quick start サンプル

---

## 13. 未確定事項 / Future work

- **Reverse annealing / pause schedule** の専用ヘルパ: `Schedule.reverse(...)`
  等のプリセットを追加するか。MVP では callable で十分。
- **複数の driver 形** (XX 等の k-local X): v0.1 では `-Σ h_x_i X_i` のみ。
  XX driver は別 matvec パスが必要 (bit-flip 2 ビット同時) のため Future work。
- **dim ≥ 2^25 級** の分散 / GPU 対応: 単機シェアードメモリで設計し、
  v1 範囲では扱わない。
- **GPU 対応**: `ndarray-linalg` は CPU LAPACK 専用。CuPy / Vulkan / Metal
  経由の matvec は別 backend として `kryanneal._gpu` 拡張モジュールを
  切る形が自然 (Future work)。
- **Trotter 分解版 (回路シミュレータ)** との比較: Trotter-Suzuki 分解で
  量子回路ベースのアニーリングシミュレーションを行う経路を載せるかは未定。
  MVP には不要。
- **シンボリックスケジュール**: SymPy で A(s), B(s) を書いて自動微分から
  CFM4 係数を引く API。Future work。

---

## 14. 参考

- Alvermann, Fehske (2011), *J. Comp. Phys.* 230, 5930-5956
  (commutator-free Magnus expansion, CFM4:2)
- Park, Light (1986), *J. Chem. Phys.* 85, 5870
  (Lanczos short-iterative time propagator)
- Tanaka group, `cv-ising-solver`
  (連続変数版の Krylov + CFM4 実装、本パッケージのカーネル設計の参照)
