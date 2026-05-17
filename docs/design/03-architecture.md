# §3. アーキテクチャ

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

