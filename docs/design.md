# kryanneal: 設計書 (v0.4)

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
  ホットループ (Lanczos / CFM4:2) は Rust (`ndarray` ベース、LAPACK 非依存)。
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
│  │    trotter_step       (Strang / Suzuki 1 step, Phase 2)│  │
│  │    apply_h_kryanneal  (matvec; bit-flip + 対角積)      │  │
│  │    apply_single_mode_axis_i (2×2 ユニタリ in-place適用)│  │
│  └────────────────────────────────────────────────────────┘  │
│         ▼                                                    │
│  QuantumResult / Trajectory dataclass                        │
└──────────────────────────────────────────────────────────────┘
```

### 3.2 役割分担

| 層 | 責務 |
|---|---|
| Python (`python/kryanneal/`) | 公開 API、`IsingProblem` / `Schedule` / `Annealer` の組み立て、結果オブジェクト、瞬時固有状態への投影、入力検証、QuTiP 比較ヘルパ |
| Rust (`src/`) | Lanczos ループ (`lanczos_propagate`)、CFM4:2 / M2 / Richardson 推定子の段階指数積、Trotter step (Phase 2 以降)、三重対角固有分解 (hand-rolled QL)、BLAS 経由の Level-1/2 ops |
| Rust matvec / primitives | `apply_h_kryanneal` (matvec, bit-flip + 対角積) と `apply_single_mode_axis_i` (Trotter 用 2×2 ユニタリ axis i 作用) を Python callback を介さず Rust 内で完結 |

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
│   ├── matvec.rs               # apply_h_kryanneal, apply_single_mode_axis_i
│   ├── krylov.rs               # lanczos_propagate (ndarray ベース)
│   ├── cfm4.rs                 # CFM4:2 / M2 / Richardson
│   ├── trotter.rs              # trotter_step (Phase 2 で追加)
│   ├── tridiag.rs              # 実対称三重対角の implicit QL (hand-rolled)
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
        # n の上限ガードは入れない (numpy が allocate 段階で MemoryError を
        # 出すのに任せる)。
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

実装ステータス: `IsingProblem` 本体 (フィールド + `__post_init__` 検証) は
Phase 1 C6 で実装済み。`from_pauli_terms` / `from_J_h` classmethod は
`builders` モジュールの実装 (別 issue) 待ちで, Phase 1 内 (C6 以降) で
追加予定。それまでは `builders.diag_from_pauli_terms` /
`builders.diag_from_J_h` を直接呼んで `H_p_diag` を構築し,
`IsingProblem(n=..., H_p_diag=..., h_x=...)` に渡す経路を使う。なお
実装上は `numpy.ndarray` のフィールドを持つため `@dataclass(frozen=True,
eq=False)` を採用している (既定の `__eq__` は array の真偽値変換で
`ValueError` になるため)。

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

    @classmethod
    def reverse(cls, T: float, s_init: float = 1.0, s_target: float = 0.5) -> "Schedule":
        """Reverse annealing schedule preset.

        s(t) を s_init から s_target まで半分の時間で下げ, 残り半分で
        s_init に戻す V 字形 (Crosson-Harrow 2016 流)。
        A(s) = 1 - s, B(s) = s は linear と同じ。
        """
        ...

    @classmethod
    def pause(cls, T: float, t_pause: float, duration: float) -> "Schedule":
        """Pause schedule preset.

        通常の linear ramp 中で `t \\in [t_pause, t_pause + duration]` の
        区間だけ s(t) を一定に保つ (King-Carrasquilla 2018 流の pause)。
        """
        ...

    def coeffs_at(self, t: float) -> tuple[float, float]:
        """(A(s(t)), B(s(t))) を返す。Annealer / Rust 側に渡すスカラー."""
        ...
```

実装注: 全フェーズで callable のまま per-step 評価する。Schedule 評価
コストは Lanczos 1 step の <1% (1 step あたり 3〜6 回、各 ~1 μs) なので
grid cache 化の ROI が薄く、将来必要になっても内部実装の差し替えで
破壊変更なしで導入可能。

### 4.4 `QuantumAnnealer`

one-shot 実行:

```python
class QuantumAnnealer:
    def __init__(
        self,
        problem: IsingProblem,
        schedule: Schedule,
        *,
        m: int = 24,                   # Lanczos 部分空間次元 (旧 krylov_dim)
        krylov_tol: float = 1e-12,
    ): ...

    def run(
        self,
        psi0: np.ndarray,              # ユーザー必須 (run 側で渡す)
        t0: float,
        t1: float,
        *,
        method: Literal["m2", "cfm4", "cfm4_adaptive_richardson"] = "m2",
        n_steps: int | None = None,    # 固定ステップ時 (m2 / cfm4)
        atol: float | None = None,     # adaptive 時の local error 許容値
        rtol: float | None = None,
        dt_init: float | None = None,  # adaptive 時の初期 dt 提案
        store_states: bool = False,    # True なら全 step の ψ を保持 (Phase 5)
        save_tlist: np.ndarray | None = None,  # 指定時刻スナップショット (Phase 5)
        observables: dict[str, "Observable"] | None = None,  # 期待値時系列 (Phase 5)
    ) -> QuantumResult: ...

    def create_simulator(
        self, method: Literal["m2", "cfm4", "cfm4_adaptive_richardson"] = "cfm4",
        **method_kwargs
    ) -> "AnnealingSimulator": ...
```

`psi0` は `__init__` ではなく `run` の必須位置引数として受け取る. 同じ問題
+ schedule に対して **異なる初期状態 / 異なる積分区間** で `run` を繰り返し
呼べるようにするため (`QuantumAnnealer` 自身は問題定義のキャッシュ役).

Phase 1 では `method="m2"` のみサポート, Phase 2 で `method="trotter"`
(固定 dt Strang 2 次 Trotter, §5.3) と `method="trotter_suzuki4"` (固定
dt Suzuki S_4 4 次 Trotter, §5.3) を追加, Phase 3 で `method="cfm4"`
(固定 dt CFM4:2 commutator-free Magnus, §5.3) を追加, Phase 4 で
`method="cfm4_adaptive_richardson"` (step-doubling Richardson 推定子 +
PI controller, §5.3) を追加. それ以外は `NotImplementedError`.
`save_tlist` / `observables` 引数は Phase 5 以降で有効化する.

`method="trotter"` / `method="trotter_suzuki4"` は Lanczos を使わないため,
コンストラクタ引数 `m` / `krylov_tol` は無視される (`"m2"` / `"cfm4"` /
`"cfm4_adaptive_richardson"` 経路でのみ意味を持つ). adaptive 経路の
`atol` / `dt_init` は `method="cfm4_adaptive_richardson"` でのみ参照され,
固定 dt 経路では無視される.

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

Phase 1 subset: 本リリース (C6) では fixed-step M2 driver のみが
提供されるため, 以下の最小フィールドのみを実装する:

```python
@dataclass(frozen=True, eq=False)
class QuantumResult:
    psi_final: np.ndarray                 # shape (2**n,) complex128
    t_history: np.ndarray | None          # 観測量を記録した時刻列
    observables_history: dict[str, np.ndarray]  # name -> shape (K,) 実数
    n_steps: int                          # 実行 step 数
    n_matvec: int                         # 累積 matvec 呼出

@dataclass(frozen=True, eq=False)
class Trajectory:
    t_history: np.ndarray
    observables_history: dict[str, np.ndarray]
```

`n_matvec` の経路ごとの解釈:

- `method="m2"`: `n_steps × m` (Lanczos の matvec 見積もり; 厳密な
  内部 matvec 回数ではなく per-step 上限).
- `method="trotter"`: `n_steps × (N + 1)` (Trotter は Lanczos を呼ばない
  ため真の matvec カウント概念は無く, phase pass 1 + bit-flip pass N の
  合計を「matvec 1 pass 相当」とみなした dim-walk 見積もり; §5.3 の
  per-step コスト記述と一致). 別フィールド `n_passes` を切らずに
  `n_matvec` を「dim-walk 見積もり」として両経路で再解釈する方針
  (issue #21).
- `method="trotter_suzuki4"`: `n_steps × 5 × (N + 1)` (Suzuki S_4 は
  Strang を 5 回呼ぶので Strang 経路の dim-walk 見積もりを 5 倍する;
  issue #22).

追加予定のフィールド:

- `times` / `states` (`store_states` / `store_times` 用) → `AnnealingSimulator`
  と一緒に Phase 5 で追加 (parent issue #1 の Out of scope 表)
- `success` / `method` / `n_steps_actual` (adaptive driver 用) → Phase 4 で
  Richardson / M2 embedded 経路と一緒に追加
- `probabilities` (`|psi_final|^2` の caching) → 必要性が出た時点で追加可能

`eq=False` は `IsingProblem` と同じ理由 (ndarray フィールドの既定 `__eq__`
が `ValueError` になる)。

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

- `method="lanczos"` (default): `lanczos_propagate` と同じ Lanczos 反復で
  下位 k 固有値を取得 (`apply_h_kryanneal` を再利用、§5.2 の三重対角化を
  そのまま使い、最終的に hand-rolled QL で固有値・固有ベクトルを得る)。
- `method="exact"`: 小規模問題 (`n <= 12`) 向け、Python 側で
  `numpy.linalg.eigh` を使った dense 検証経路 (Rust 経由で LAPACK を
  呼ばない)。

ユーザーは `ψ(t)` を `eigvecs` に内積して amplitude を出す:
`amps = eigvecs.conj().T @ psi(t)`。

### 4.8 例外型ポリシー

v0.1 では **Python 標準例外のみ** を使う:

| 状況 | 例外型 |
|---|---|
| 入力検証失敗 (shape / dtype / NaN / 非正規化 psi0 / `n != len(h_x)` 等) | `ValueError` |
| パラメタ範囲外 (`T <= 0`, `dt0 <= 0`, `n_steps < 1` 等) | `ValueError` |
| adaptive dt 連続 reject 超過 (`max_rejects` 到達) | `RuntimeError` |
| Lanczos 部分空間構築での数値破綻 | `RuntimeError` |
| 三重対角 QL の収束失敗 (`30·m` iter 到達) | `RuntimeError` |
| Rust 拡張のロード失敗 (`_rust` 未ビルド) | `ImportError` + `RuntimeWarning` |

Rust 側からは PyO3 の `PyValueError::new_err(...)` / `PyRuntimeError::new_err(...)`
で raise する。custom exception hierarchy (`KryannealError` ベース等) は
**v0.1 では定義しない**。必要性が出てきた場合は、v0.2 以降で

```python
class KryannealError(Exception): ...
class KrySchemaError(KryannealError, ValueError): ...
class KryConvergenceError(KryannealError, RuntimeError): ...
```

のような多重継承で **既存の `except ValueError:` / `except RuntimeError:` を
壊さずに**段階的に導入可能 (後方互換性を保ったまま追加できる)。

---

## 5. 数値カーネル

### 5.1 matvec / per-axis primitives

Rust 側に持つ低レベル配列演算プリミティブは 2 種類:

| プリミティブ | 用途 | 導入 phase | 動作モード |
|---|---|---|---|
| `apply_h_kryanneal` | Lanczos / CFM4:2 (Magnus 系) の matvec | Phase 1 | additive: `y += c·H·v` 系 |
| `apply_single_mode_axis_i` | Trotter 系の 2×2 ユニタリ適用 | Phase 2 | in-place rotation |

両者とも bit-flip 構造 (`(psi[k], psi[k ^ (1<<i)])` のペア処理) を共有するが、
matvec は **複数 i の寄与を 1 つの y に足し込む** のに対し、Trotter は
**1 つの i ごとに in-place で psi を回転する** という違いがあり、メモリ
アクセスパターンと最適化の余地が異なるため別関数として書く。

#### 5.1.1 `apply_h_kryanneal` (Phase 1)

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
        // ※ inner loop は cache-blocking + SIMD 余地あり (Phase 6 で対応)
        for k in 0..dim {
            y[k] += Complex64::new(coeff, 0.0) * v[k ^ mask];
        }
    }
}
```

性能考慮:

- diagonal 部分は cblas `zdscal` + 要素積を unrolled inner loop で
- bit-flip 部分は `i` 外側 / `k` 内側で連続アクセスにできる: ビット i に
  ついて k と k^mask のペアを `mask=1<<i` で 2 元ストライドにより列挙。
  i=0 で stride 1、i=1 で stride 2、... と level-by-level に走査することで
  TLB / L2 ヒット率を上げる古典的テクニック (state-vector simulator の
  X-gate pass と同一)。**Phase 1 ではスカラ単スレッド実装、Phase 6 で
  rayon + SIMD + cache block-fusion を載せる** (§12 Phase 6)

#### 5.1.2 `apply_single_mode_axis_i` (Phase 2)

Trotter 経路で `R_i(θ) = cos(θ)·I + i·sin(θ)·X_i` を psi に in-place で
適用する関数。X_i が bit-flip 演算子であることを活用し、`(psi[k],
psi[k ^ (1<<i)])` のペアごとに 2×2 ユニタリ U を直接乗じる:

```rust
/// psi を axis i で 2 元化したペアに 2×2 ユニタリ U を in-place 適用.
///
/// U は `[[u00, u01], [u10, u11]]` の row-major 2x2 行列。
/// Trotter で R_i(θ) を渡す場合は u00=u11=cos θ, u01=u10=i·sin θ.
fn apply_single_mode_axis_i(
    psi: &mut [Complex64],
    u: &[Complex64; 4],   // row-major 2x2
    i: usize,
    n: usize,
) {
    let dim = 1usize << n;
    let mask = 1usize << i;
    // bit i = 0 の k だけを enumerate (重複適用を避ける)
    let mut k = 0usize;
    while k < dim {
        if k & mask != 0 {
            k += 1;
            continue;
        }
        let a = psi[k];
        let b = psi[k | mask];
        psi[k]        = u[0] * a + u[1] * b;
        psi[k | mask] = u[2] * a + u[3] * b;
        k += 1;
    }
}
```

(実装は `i` ごとに stride を変える 2 重ループ形に整える: 外側で長さ
`1<<i` のブロックを走り、内側で連続 dim/2 ペアを処理する。`k & mask != 0`
のスキップを実行せずに済む形にする。詳細は実装時に決める。)

呼び出し側 (`trotter_step`):

```rust
pub fn trotter_step(
    psi: &mut [Complex64],
    h_x: &[f64],          // length n, サイトごとの横磁場振幅
    h_p_diag: &[f64],     // length 2^n
    a_t: f64,             // A(s(t + dt/2)) などの schedule 値 (中点)
    b_t: f64,
    dt: f64,
    n: usize,
) {
    let dim = 1usize << n;

    // Strang: phase_p(dt/2) -> Π_i R_i(dt) -> phase_p(dt/2)
    let half = 0.5 * dt;
    for k in 0..dim {
        let phi = -b_t * h_p_diag[k] * half;
        psi[k] *= Complex64::new(phi.cos(), phi.sin());
    }
    for i in 0..n {
        // H_drv = -Σ h_x_i X_i なので
        //   exp(-i·dt·a_t·H_drv) = Π_i exp(+i·θ_i·X_i), θ_i = a_t·h_x_i·dt
        // → u = cos(θ)·I + i·sin(θ)·X with θ = +a_t·h_x_i·dt.
        // (H_drv の負符号は θ では巻き取らない. apply_h_kryanneal の
        //  coeff = -a_t·h_x_i と同じ convention.)
        let theta = a_t * h_x[i] * dt;
        let c = theta.cos();
        let s = theta.sin();
        let u = [
            Complex64::new(c, 0.0),
            Complex64::new(0.0, s),
            Complex64::new(0.0, s),
            Complex64::new(c, 0.0),
        ];
        apply_single_mode_axis_i(psi, &u, i, n);
    }
    for k in 0..dim {
        let phi = -b_t * h_p_diag[k] * half;
        psi[k] *= Complex64::new(phi.cos(), phi.sin());
    }
}
```

**設計判断 (一般的な reshape + GEMM パターンを採用しない理由)**:

連続変数 / 一般 bosonic 系では `apply_single_mode(M, psi, axis)` を
「`psi` を `(N_fock,) * m` に reshape → 単一 BLAS GEMM」 パターンで書くのが
定石だが、本パッケージでは以下の理由で **N_fock=2 特化の自前 bit-flip pass**
を選ぶ:

1. **GEMM 呼び出しオーバヘッド**: N_fock=2 では右オペランドが (2, dim/2) の
   非常に細長い行列となり、N_fock が大きい (~20-40) ケースと比べて BLAS の
   per-call overhead が dim に対して相対的に重い (推定で dim < 2^12 程度で
   自前ループが優位)。
2. **中間軸 moveaxis のコピー**: 一般の reshape + GEMM 経路では中間軸を
   末尾に持ってくる `np.moveaxis` + workspace への物理コピーが必要。
   N_fock=2 だと「ペアの swap-with-mix」は連続 / 2-stride アクセスで
   コピー不要で済むため、moveaxis を挟むのは逆に損。
3. **`apply_h_kryanneal` と同じ層に揃える**: Phase 6 の cache
   block-fusion 最適化は両者で共通の bit-flip pass パターンに対して効くため、
   matvec と Trotter を同じ Rust モジュール (`src/matvec.rs`) 上で同型に
   書いておく方が後段の最適化が両者に均等に効く。

Trotter primitive は `src/trotter.rs` に置く想定だが、`apply_single_mode_axis_i`
自体は matvec primitive の隣 (`src/matvec.rs`) でも構わない (Phase 2 で
実装する際に判断する)。

### 5.2 Lanczos: `lanczos_propagate`

`exp(-i dt H) ψ` を `m` 次元 Lanczos + 三重対角固有分解で計算する
matrix-free 短時間プロパゲータ (Park-Light 1986)。実装方針:

- 配列型は **ndarray** に統一。
- 三重対角の対称固有分解は **hand-rolled な implicit QL with Wilkinson shift**
  を `src/tridiag.rs` に持つ (LAPACK 依存を切る; 詳細 §7.1)。
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
) -> PyResult<Vec<Complex64>>
where
    F: FnMut(&[Complex64], &mut [Complex64]),  // y = H · v
```

`matvec` を closure に取り、`cfm4.rs` から `(c_drv, c_diag)` を畳み込んだ
線形結合版 closure を渡せるようにする (CFM4:2 の各 stage 用の Hamiltonian
を 1 つの線形結合 matvec として表現する経路)。これにより CFM4:2 の各
ステージで matvec 呼出は m 回のみ。

### 5.3 プロパゲータ

以下 4 種のプロパゲータを提供する (M2 / CFM4:2 は Magnus 系、Trotter は
operator splitting 系)。Adaptive dt 経路は CFM4:2 系の embedded / Richardson
推定子から構成する (Trotter は固定 dt のみ; embedded estimator を持たない):

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

を CFM4:2 自身の LTE 推定値として返す。per-step matvec は **6m**
(full 2m + half×2 × 2m, Lanczos 呼出 6 回, 固定 dt CFM4:2 比 3×)。
M2 embedded 版より 2 オーダ高精度なので smooth schedule では許容 dt を
1〜2 桁伸ばせる。

オプション `extrapolate=True` で Richardson 外挿:
`ψ_acc = (16 · ψ_h2 - ψ_full) / 15` (実効 6 次精度)。

#### Trotter (Strang 2 次 / Suzuki 4 次) — `trotter_step`

横磁場 driver の `[X_i, X_j] = 0` を活用し、`exp(-i dt H_drv)` を
`Π_i R_i(dt)` の閉形式 (Lanczos 不要) で書く operator splitting 経路。
Strang 2 次:

```
U(dt) ≈ exp(-i dt H_p / 2) · exp(-i dt H_drv) · exp(-i dt H_p / 2)
      = phase_p(dt/2) · (Π_i R_i(dt)) · phase_p(dt/2)
```

各 `R_i(dt) = cos(A·h_x_i·dt)·I + i·sin(A·h_x_i·dt)·X_i` は §5.1.2 の
`apply_single_mode_axis_i` で 1 軸 in-place 適用。`H_drv = -Σ h_x_i X_i`
の負符号は `exp(-i·dt·H_drv) = Π_i exp(+i·a·h_x_i·dt·X_i)` で打ち消されて
`R_i` の `θ = +a·h_x_i·dt` に乗る (`apply_h_kryanneal` の
`coeff = -a_t·h_x_i` と同 convention)。

per-step コスト: `(N + 1) · dim` 要素アクセス (matvec の 1 pass 相当が
N+1 回)。CFM4:2 の `2m·dim` (m=24 で ~48·dim) と比較すると N=20 で
~2.3× 軽量だが、LTE は O(dt^3) なので精度要求次第で総時間の優劣は変わる
(クロスオーバ実測は Phase 2 でベンチに含める、§12)。

API:

```rust
pub fn trotter_step(
    psi: &mut [Complex64],
    h_x: &[f64],
    h_p_diag: &[f64],
    a_t: f64,            // 中点 schedule 値 A(s(t + dt/2))
    b_t: f64,            // 同 B(s(t + dt/2))
    dt: f64,
    n: usize,
);
```

`(a_t, b_t)` は schedule の **中点で評価** することで Strang 2 次の対称性を
保つ (時間依存 H の Strang は中点採取で局所 O(dt^3) を維持する)。
固定 dt ドライバから直接呼ぶ。

**4 次 Suzuki (`trotter_suzuki4_step`)**: Trotter-Suzuki S_4 公式

```
S_4(dt) = S_2(p·dt) · S_2(p·dt) · S_2((1-4p)·dt) · S_2(p·dt) · S_2(p·dt)
p = 1 / (4 - 4^{1/3}) ≈ 0.4145
```

で Strang 5 回適用に分解。per-step は ~5·(N+1)·dim、LTE O(dt^5)。CFM4:2 と
同じ局所オーダだが、Lanczos の m 回 matvec を完全に排した経路としての
比較・検証用に Phase 2 末で追加 (`method="trotter_suzuki4"`)。

中央 sub-step は `1 - 4p ≈ -0.658` で **時間逆向き** に走る (Suzuki の
高次合成では正の係数しか持つ対称合成が存在しないことの帰結)。`trotter_step`
は `dt < 0` を許容するので呼出側で特別扱い不要。

API:

```rust
pub fn trotter_suzuki4_step(
    psi: &mut [Complex64],
    h_x: &[f64],
    h_p_diag: &[f64],
    a_t_list: &[f64],    // length 5: 各 sub-step の A(s(中点))
    b_t_list: &[f64],    // length 5: 各 sub-step の B(s(中点))
    dt: f64,             // 外側 1 step の時間刻み
    n: usize,
);
```

**サブステップ係数・中点 offset**: sub-step 幅 `[p, p, 1-4p, p, p]·dt` を
時間順に並べたとき, sub-step `k` の **中点 offset** (`(start_k + end_k)/2`)
は

```
offsets = [p/2, 3p/2, 1/2, 1 - 3p/2, 1 - p/2]
```

で `t + dt/2` を中心に対称。時間依存 H に対しては各 sub-step の中点で
`(A(s(·)), B(s(·)))` をフリーズ採取することで全体の LTE `O(dt^5)` を保つ。

**schedule 評価の責務**: ホスト言語 (Python driver
`evolve_schedule_trotter_suzuki4`) が中点 offset を内部で持ち, 各 step
ごとに 5 つの `(a_mid, b_mid)` を事前計算して長さ 5 の配列として Rust 側に
渡す。Rust 拡張は `schedule` callable を持ち込まず, Strang `trotter_step`
を `a_t_list[k] / b_t_list[k] / coeffs[k] * dt` で 5 回呼ぶだけのループ。
Strang 経路 API (`trotter_step(psi, ..., a_t, b_t, dt, n)`) と「呼出側で
schedule 評価 → Rust に純粋な係数を渡す」契約を統一する。

**embedded error estimator は持たない**: Strang↔Suzuki4 の差を embedded
推定子に使う案も理論上はあるが、Phase 4 の `cfm4_step_with_*_estimate`
体系と統合せず、Trotter 経路は **固定 dt 専用** とする。adaptive 経路は
CFM4:2 を使う。

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

#### adaptive driver の DX 改善 (Phase 4 follow-up, issue #43 / #54)

`m` / `dt_init` / `dt_max` / `krylov_tol` の手動チューニングを段階的に
自動化する follow-up 群。issue #43 A/B/C で `dt_init="auto"` /
`dt_max="auto"` / `m_max=` を v0.4.x patch リリースで導入し, issue #54 で
3 つの adaptive 経路パラメータ (`krylov_tol` / `dt_init` / `dt_max`) を
**`None` default + auto resolution + 明示 float で override** の統一
スタイルに揃え (`"auto"` リテラル廃止), v0.5.0 minor bump で公開 API の
破壊的変更として取り込んだ (`docs/conventions.md` §2 バージョニング
ポリシーに準拠)。

##### A. `dt_init=None` で T 依存 auto resolution (issue #43 A で導入, issue #54 で None default 化)

`QuantumAnnealer.run(method="cfm4_adaptive_richardson", dt_init=None)`
(既定値) で, facade 側が線形 schedule の Magnus 級数 T スケーリング
(s-space scaling invariance) から導いた保守値を `dt0` に解決する:

```
dt0 = min(max(c · T^β, _AUTO_DT_INIT_FLOOR), T)
       (T = t1 - t0, 既定 c = 0.1, β = 0.5, floor = 1e-3)
```

理論最適は `β = 3/4` (Magnus 切断誤差 `~ K · dt^5`, `K ~ ‖[H_drv, H_p]‖²`
の T 依存と `T = (t1 - t0)` 区間長から導出, `docs/design.md` §5.3 内
PI controller 既定値の議論と整合) だが, schedule 非線形性や問題依存性に
対するロバスト性を取って `β = 0.5` を既定とした (issue 本文の motivation
参照)。`T < 1` の小 T ケースで `c · T^β` が driver の `dt_min` (default
`1e-4`) を下回らないよう床値 `1e-3` を, 逆に `dt0 > T` で driver 入力
検証 `dt_max >= dt0` (`dt_max = 10 · dt0` default) を満たさなくなる退化
ケースを避けるため上限 `T` を同時に張る。

resolution は facade 層 (`python/kryanneal/annealer.py` の
`_resolve_dt_init_auto`) で行い, driver (`evolve_schedule_adaptive_richardson`)
は受け取った `dt0` をそのまま使う。これにより driver 単体テスト
(`tests/test_adaptive.py` 既存) は変更不要で, facade 層のテスト (同
ファイル末尾) で None resolution 経路と PI controller との接続を smoke
検証する。issue #54 以前は `dt_init="auto"` で発動する opt-in だったが,
None default 化により既定経路に昇格 (旧 default `0.5` 固定は完全廃止)。

##### B. `dt_max=None` で Lanczos capacity 自動見積もり (issue #43 B で導入, issue #54 で None default 化)

`QuantumAnnealer.run(method="cfm4_adaptive_richardson", dt_max=None)`
(既定値) で, facade 側が Gershgorin 上界による Lanczos capacity 自動
見積もりを `dt_max` に解決する:

```
‖H‖_est = Σ_i |h_x_i| + max_k |H_p_diag[k]|
dt_max  = max(min(10·dt0, 4m / ‖H‖_est), dt0)
```

Lanczos m 部分空間で `exp(-i dt H) |ψ⟩` を `rel < tol` で再現できる
安全領域は経験的に `dt · ‖H‖ ≲ 4 m` (cv_ising 流, hand-rolled Lanczos の
collapsed safe radius)。`‖H‖` は Gershgorin 上界で closed form に
見積もる (Power method で 5–10 step 走らせる案 2 もあるが overhead が
あるので Phase 4 follow-up では closed form を採用)。最後の
`max(_, dt0)` は driver 入力検証 `dt_max >= dt0` を満たすための floor で,
`dt0` が Lanczos cap を超える縮退ケースでは Richardson 推定子が breakdown
を embedded error として検出し PI controller が dt を縮めるため
fail-safe で成立する (issue #43 B の motivation と整合)。

大 N で `‖H‖ ∝ N` が支配的になる領域では `4m/‖H‖` が default
`10·dt0` を下回り cap が効く。例えば m=24, dt0=0.5, n=10 (h_x=1·10),
H_p_diag in [-1, 1] では `‖H‖_est = 10 + 1 = 11` → cap = 4·24/11 ≈ 8.7,
default 5.0 が支配。n=50 では `‖H‖_est ≈ 51`, cap ≈ 1.88, default 5.0
より cap が支配。

resolution は facade 層 (`python/kryanneal/annealer.py` の
`_resolve_dt_max_auto`) で行い, driver は既存 `dt_max=` パラメータを
そのまま受ける (driver 内部の入力検証で `dt_max >= dt0` を担保)。
issue #54 以前は `dt_max="auto"` で発動する opt-in だったが, None
default 化により既定経路に昇格 (旧 driver default `10·dt0` 固定は
完全廃止; Lanczos capacity を考慮しないため大 N で危険だった)。

##### C. `m` の adaptive 化 (issue #43 C, v0.4.x で簡略 scope を導入)

`QuantumAnnealer.run(method="cfm4_adaptive_richardson", m_max=16)` を
渡すと, facade 側で adaptive Richardson 経路の Lanczos 部分空間次元
上限を `self.m` (コンストラクタ既定 24) から `m_max` で上書きする。
step-doubling Richardson 推定子が Lanczos breakdown も embedded error
として検出する fail-safe (Phase 4 C3) を活かし, `m_max=16` 等の保守値
で per-step matvec を 30% 程度削減する運用を許容する (Richardson が
破綻を検知すれば PI controller が dt を絞り精度を維持)。`β_k <
krylov_tol` の早期打切は既存 `lanczos_propagate` で実装済 (`src/krylov.rs`
§ `m_eff` 計算)で, 実効次元は `m_eff ≤ m_max`。

**簡略 scope の理由**: issue 本文の C task は `m_eff` の per-step
累積統計を `QuantumResult` に保存し, `bench_per_step.py` で
`m=adaptive vs m=16 fixed vs m=24 fixed` の wall time 比較を要求する。
これらは Rust 側 `lanczos_propagate` の戻り値拡張 (現状 `Vec<Complex64>`
→ `(Vec<Complex64>, usize)` で `m_eff` を返す) + PyO3 plumbing + Python
driver 集計が必要で, Phase 4 follow-up の DX 改善 PR としては
パッケージが大きすぎる。本リリースでは facade パラメータ `m_max` で
user-facing API を確定させ, m_eff 統計と bench 拡張は別 issue で起票
予定 (Phase 5 で `QuantumResult` の history 拡張と一緒に取り込む案を
含む)。

実機 benchmark 評価は `bench_per_step.py` で `--m 16` / `--m 24` の
2 経路を手動で sweep して per-cell wall time を比較する形でも検証可
(adaptive vs fixed の 1.5–1.7× 期待 speedup は issue 本文 motivation
参照)。

##### E. adaptive driver default の統一 (issue #54, v0.5.0 で導入)

PR #53 (issue #52) で `QuantumResult.m_eff_stats` を露出させたところ,
adaptive Richardson の `krylov_tol = 1e-12` (旧 default) が `atol = 1e-8`
default に対して **4 桁過剰タイト** で Lanczos β_k 早期打切が一切効かず
`m_eff = 6·m_max` が常時続くことが実機 bench で判明した (N=16, T=100,
m_max ∈ {16, 24, 32} で `m_eff_median` が 96 / 144 / 192 と m_max 比例)。

直接の対策は `krylov_tol` の default を `atol` 連動に変えることだが,
同時により一般化された設計問題として PR #51 (issue #43 A/B) で導入した
`dt_init` / `dt_max` の **`None` (固定保守 default)** と **`"auto"`
(問題依存推定)** の 2 経路設計も再考の余地があった。「根拠の薄い固定
保守 default より auto 解決値を default にする」方が筋, という観点で
3 つの adaptive 経路パラメータを **`None` default + auto resolution +
明示 float で override** の統一スタイルに揃える。

| パラメータ | 旧 default | 旧 `"auto"` 挙動 | **v0.5.0 新 default (`None` で auto resolution)** |
|---|---|---|---|
| `krylov_tol` | `1e-12` 固定 | (なし) | `atol · _KRYLOV_TOL_ATOL_RATIO` (既定 `1e-3`) |
| `dt_init` | `None → 0.5` 固定 / `"auto"` あり | `max(min(c·T^β, T), 1e-3)` | 同上 (旧 "auto" 式が default) |
| `dt_max` | `None → 10·dt0` / `"auto"` あり | `max(min(10·dt0, 4m/‖H‖_est), dt0)` | 同上 |

`Literal["auto"]` リテラルは facade から完全削除 (v0.5.0 breaking change)。
None default = 旧 `"auto"` 経路と挙動上等価なので, `dt_init="auto"` /
`dt_max="auto"` を明示していた呼び出しを `dt_init=None` / `dt_max=None`
(または引数省略) に置換すれば挙動はビット一致で維持される。

`krylov_tol` の連動係数 `1e-3` の根拠:

- adaptive Richardson 推定子は `err = ‖ψ_full - ψ_h2‖` を `tol_step` 以下
  に保つよう PI 制御で dt を伸縮する。1 step あたりの Lanczos 内誤差
  (β_k 早期打切時の打切誤差) が PI controller の embedded error 推定を
  支配しないよう, `atol` より **少なくとも 1 桁タイト** に取りたい。
- 経験則として 3 桁マージン (`1e-3`) は実用 `atol ∈ [1e-10, 1e-6]` の
  範囲で安定動作する。例: `atol=1e-8` → effective `1e-11`, `atol=1e-6`
  → effective `1e-9`。1e-4 / 5e-4 / 5e-3 の境界は実機 bench で再評価
  可能だが, 初期 default として `1e-3` を採用 (`_KRYLOV_TOL_ATOL_RATIO`
  module 定数で 1 箇所集中管理)。
- 固定 dt 経路 (`m2` / `cfm4`) は `atol` を取らないため None →
  `_KRYLOV_TOL_FIXED_DEFAULT = 1e-12` に static fallback (旧 default
  維持)。adaptive 経路の atol 連動とは独立。

resolution は全て facade 層 (`python/kryanneal/annealer.py`) で行い,
driver (`evolve_schedule_adaptive_richardson`) は受け取った `dt0` /
`dt_max` / `krylov_tol` をそのまま使う。これにより driver 単体テストは
変更不要で, facade 層のテスト (`tests/test_adaptive.py`) で None
resolution 経路と PI controller / Lanczos の接続を smoke + bit-exact で
検証する。

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
- `ndarray = "0.16"`
- `num-complex = "0.4"`
- `cblas = "0.5"` (optional, BLAS feature)
- `blas-src = "0.12"`:
  - macOS: `accelerate` feature
  - Linux: `openblas` feature (system OpenBLAS)

**三重対角固有分解の実装方針** (確定):

Lanczos 1 step で必要な唯一の LAPACK 相当の処理は **m × m (m ~ 24) の
実対称三重対角固有分解**。これは hot path ではない (step 全体の <0.5%) ので、
LAPACK を引っ張ってくる ROI が低い。本パッケージは **`ndarray-linalg`
を依存に入れず**、`src/tridiag.rs` に **implicit QL with Wilkinson shift**
を hand-roll する。

選定理由:

- m=24 の三重対角固有分解は ~1.4 × 10⁴ FP ops で <10 μs。Lanczos 1 step
  全体は dim 依存 ops (`cblas` 経由) が支配的 (dim=2^20 で ~500 μs)
- LAPACK 依存を切ることで以下が解消される:
  - macOS で `brew install openblas` 等の追加 install 不要
  - Apple Accelerate を Level-1/2 BLAS でフル活用できる (AMX 経路)
  - `blas-src` と LAPACK backend の二重管理を避けられる
  - wheel の static 同梱が単純化する

実装規模:

- `src/tridiag.rs` は ~100〜150 行 (Wilkinson shift, Givens rotation,
  deflation 閾値, max-iter cap 含む)
- 出力 (固有値 λ_p の昇順、固有ベクトル行列 Q) は `dsteqr` 互換のシグネチャ
- Givens rotation は `f64::hypot` を使い overflow/underflow を回避
- Deflation 閾値は `|β_k| ≤ ε · (|α_{k-1}| + |α_k|)` (ε = `f64::EPSILON`)
- Max iter cap: 30·m (LAPACK `dsteqr` と同じ)
- 収束失敗時は `Err(PyRuntimeError::new_err("tridiag QL did not converge ..."))`
  を返し、Python 側で `RuntimeError` として伝播 (例外型方針は §4.8 を参照)

テスト戦略:

- `cargo test`: ランダム m×m tridiag (m ∈ {2, 8, 16, 24, 48}) を生成し、
  hand-rolled の出力と **`nalgebra::SymmetricTridiagonal`** (dev-dep のみ)
  の固有値/固有ベクトルを比較。`rel < 1e-13` で一致を要求
- `pytest`: 同じテストを Rust の公開 `_rust._tridiag_eigh_py` 経由で呼び、
  `scipy.linalg.eigh_tridiagonal` と `rel < 1e-13` で一致を確認
- Fuzzing: ランダムシード sweep でクラスタ・退化ケースを smoke test
- 収束失敗は明示的にハンドル (max_iter 超過 → `RuntimeError`)

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

→ `docs/conventions.md` §1 参照. (uv / maturin / ruff / ty / pre-commit /
gen_api_stubs ドリフト二段運用 / BLAS 多プロセス制御の export.)

---

## 12. 段階リリース計画

バージョニングポリシー (Phase N → v0.N bump, umbrella issue DoD 必須項目)
は `docs/conventions.md` §2 を一次資料とする.

### Phase 1: MVP / scalar baseline (~v0.1)

- `IsingProblem`, `Schedule`, `QuantumAnnealer.run(method="m2")` のみ
- `Schedule` プリセット: `linear` / `from_callable` / `reverse` / `pause`
  (reverse annealing と pause schedule は研究用途で頻出するため Phase 1
  時点で同梱)
- Rust 拡張: `apply_h_kryanneal`, `lanczos_propagate`, `m2_midpoint_step`,
  `tridiag_eigh` (hand-rolled QL)
- Python リファレンス (`_python_*`) との等価性テスト
- 小規模 QuTiP 比較テスト
- **スカラ単スレッド・SIMD 明示利用なし** で実装。以降 (Phase 2 以降) の
  高速化施策の baseline として `bench_per_step.py` の数値を確定させる
- BLAS feature ON/OFF は両方ビルド可能だが、Level-1/2 ops が呼ばれるのは
  Lanczos 内部のみで、matvec / bit-flip pass は自前のスカラループ

### Phase 2: Trotter 経路 (~v0.2)

横磁場演算子 X_i の bit-flip 性と可換性 (`[X_i, X_j] = 0`) を活用し、
`exp(-i dt H_drv) = Π_i R_i(dt)` を **Lanczos を経由しない閉形式の
2×2 rotation で逐次適用** する経路。Strang 2 次 Trotter:

```
U(dt) ≈ phase_p(dt/2) · (Π_i R_i(dt)) · phase_p(dt/2)
```

- Rust 側に `apply_single_mode_axis_i` を新規実装 (詳細 §5.1.2):
  - `(psi[k], psi[k ^ (1<<i)])` ペアに 2×2 ユニタリを in-place 適用
  - N_fock=2 特化の自前 bit-flip pass で書く (一般的な reshape + GEMM
    パターンを採らない根拠は §5.1.2 末尾)
  - Phase 2 ではスカラ単スレッド (SIMD/threading は Phase 6 で乗せる)
- Rust 側に `trotter_step` (Strang 1 step エントリ) を新規実装
- 4 次 Suzuki (Trotter-Suzuki S_4) はオプションで追加可
- `method="trotter"`, `method="trotter_suzuki4"`
- Phase 1 の M2 と精度・速度を同一マシンで比較 (`bench_per_step.py` 拡張)。
  Trotter は per-step が ~(N+1)·dim flops と軽い反面 2 次精度なので
  「短時間 / 緩やかな schedule」で M2 / CFM4:2 比優位、「長時間 / 高精度」では
  CFM4:2 が勝つ、というクロスオーバを実測で示す

### Phase 3: CFM4:2 (~v0.3)

- `cfm4_step`, `method="cfm4"` 経路
- 線形結合 callback 形式 (§5.2 末尾) で per-step matvec を 4m → 2m に削減

### Phase 4: Adaptive (~v0.4)

- `cfm4_step_with_m2_estimate` (embedded M2 error)
- `cfm4_step_with_richardson_estimate` (step-doubling Richardson)
- Python 側 PI controller driver
- `method="cfm4_adaptive_richardson"`

### Phase 5: Simulator & Observables (~v0.5)

- `AnnealingSimulator`
- `Observable` クラス、観測量時系列記録
- `instantaneous_eigenstates`

### Phase 6: 並列化 + 仕上げ (~v0.6)

Phase 1-5 でアルゴリズム面の機能が出揃った時点で実装面の並列化に着手する。
Phase 1 の baseline と比較できることが本 phase の前提。

- **L2 並列化**: matvec / Trotter primitives の bit-flip pass を rayon
  `par_chunks_mut` で並列化。`apply_h_kryanneal` と `apply_single_mode_axis_i`
  の両方が対象 (CFM4:2 / Trotter どちらの経路でも効く)
- **SIMD**: `std::simd` または `wide` クレートで AVX2 / AVX-512 / NEON
  ターゲット。i=0,1,2 (stride 1/2/4) の連続アクセス領域に集中して適用
- **cache block-fusion**: 大 N (≥20) で高 i の bit-flip pass が DRAM 律速
  になるのを防ぐため、高 i 群を fuse して L2 cache に収まるブロック単位で
  低 i pass と一緒に走らせる古典テクニック (qsim の X-gate pass 同様、§5.1.1
  末尾の TODO で referenced)
- 物理コア数 vs スループットの sweep をベンチに含め、メモリ帯域律速点を
  明示する
- BLAS feature ON/OFF の数値一致 CI (両ビルドで rel < 1e-13)
- 大規模 QuTiP 比較 (n=12-16 程度まで)
- ドキュメント整備、Quick start サンプル

---

## 13. Future work (v0.6 までは対応しない範囲)

以下の項目は **v0.6 までのリリース計画には含めない** ことが確定済み
(Phase 計画は §12 参照)。リクエストが出てきた時点で v0.7+ として再評価する。

- **複数の driver 形 (XX 等の k-local X)**: v0.1〜v0.6 では driver を
  `-Σ h_x_i X_i` (サイト依存の単一 X) に限定する。XX 等の k-local X 項を
  入れるには matvec.rs に bit-flip 2 ビット同時パスの新規実装 (~200 行)
  と Schedule API の拡張が必要なため後送り。
- **Z_2 (global spin-flip) 対称性によるセクタ分割**: `Π = Π_i X_i` は
  H_drv と常に交換するが、H_p との交換は H_p の Z-string がすべて偶数重み
  (= 縦磁場 / 奇数 k-local Z 項がない) の場合に限る。`H_p_diag[k] ==
  H_p_diag[k ^ (2^N - 1)]` で全 k チェックして対称性を判定し、対称な場合
  に `(psi[k] ± psi[k_flip])/√2` 基底でセクタ分割すれば計算・メモリが半分
  になる。実装は「Phase 1-6 の汎用経路は対称性に依らず動く + ユーザが
  hint (`IsingProblem(..., symmetry='z2')` など) を渡したときに後段で
  fast path に切替える」構成が綺麗。利得は最大 2× で条件付きのため、
  汎用経路の安定化後 (= Phase 6 完了後) に検討する。
- **dim ≥ 2^25 級の分散実行**: 単機シェアードメモリで設計する。MPI /
  NCCL / dask 等の分散層は別プロジェクト級の作業量になるため範囲外。
- **GPU 対応**: CuPy / Vulkan / Metal 経由の matvec は別 backend として
  `kryanneal._gpu` 拡張モジュールを切る形を想定。1 リリース分相当の
  作業量なので v0.6 までには含めない。
- **シンボリックスケジュール**: SymPy で A(s), B(s) を書いて自動微分から
  CFM4 高次係数を生成する API。ニッチかつ既存 callable で代替可能なため
  優先度低。

---

## 14. 参考

- Alvermann, Fehske (2011), *J. Comp. Phys.* 230, 5930-5956
  (commutator-free Magnus expansion, CFM4:2)
- Park, Light (1986), *J. Chem. Phys.* 85, 5870
  (Lanczos short-iterative time propagator)
- Shu Tanaka group, `cv-ising-solver`
  (連続変数版の Krylov + CFM4 実装、本パッケージのカーネル設計の参照)
