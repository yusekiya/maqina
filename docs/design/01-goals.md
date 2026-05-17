# §1. ゴールと非ゴール

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

