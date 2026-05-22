# §4. Python API

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
        method: Literal["m2", "cfm4", "cfm4_adaptive_richardson_krylov"] = "m2",
        n_steps: int | None = None,    # 固定ステップ時 (m2 / cfm4)
        atol: float | None = None,     # adaptive 時の local error 許容値
        rtol: float | None = None,
        dt_init: float | None = None,  # adaptive 時の初期 dt 提案
        store_states: bool = False,    # True なら全 step の ψ を保持 (Phase 5)
        save_tlist: np.ndarray | None = None,  # 指定時刻スナップショット (Phase 5)
        observables: dict[str, "Observable"] | None = None,  # 期待値時系列 (Phase 5)
    ) -> QuantumResult: ...

    def create_simulator(
        self, method: Literal["m2", "cfm4", "cfm4_adaptive_richardson_krylov"] = "cfm4",
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
`method="cfm4_adaptive_richardson_krylov"` (step-doubling Richardson 推定子 +
PI controller, §5.3) を追加. それ以外は `NotImplementedError`.
`observables` / `save_tlist` / `store_states` 引数は Phase 5 (issue #47)
で有効化された. 仕様:

- **`save_tlist=None` (デフォルト) = 最節約モード**: `observables` /
  `store_states` の指定があると `ValueError` (silent 無視は debug 罠に
  なるため明示的に弾く). `QuantumResult.times = states = None`,
  `observables_history = {}`. ただし `probabilities = |psi_final|^2` は
  どの経路でも常に eager 計算して返す (最終状態の付随情報).
- **`save_tlist=array`**: 観測時刻軸として採用. 固定 dt 経路は step
  boundary 列に `save_tlist` 時刻を merge して uneven な `dt` で進み,
  adaptive 経路は PI controller の `dt` を `next_save_target - t` で
  クランプして当該時刻を厳密に踏む. `observables` を併用すれば各
  `save_tlist[i]` 時刻で `obs.expectation(psi)` を評価して
  `observables_history[name]` に shape `(K,)` で格納し, `store_states=True`
  なら `states` shape `(K, 2**n)` complex128 で ψ を保存する.
- 検証: `save_tlist` は dtype float64, 1D, monotonic non-decreasing,
  範囲 `[t0, t1]`, 非空. `observables` は `dict[str, Observable]` で
  各 `Observable.diag` 長が `2**n` と整合.

`method="trotter"` / `method="trotter_suzuki4"` は Lanczos を使わないため,
コンストラクタ引数 `m` / `krylov_tol` は無視される (`"m2"` / `"cfm4"` /
`"cfm4_adaptive_richardson_krylov"` 経路でのみ意味を持つ). adaptive 経路の
`atol` / `dt_init` は `method="cfm4_adaptive_richardson_krylov"` でのみ参照され,
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

Phase 1-5 subset (issue #47 で Phase 5 拡張完了): 現在の実装は以下のフィールド
構成. Phase 1 〜 4 のフィールドに加え, Phase 5 で `times` / `states` /
`probabilities` を末尾に default 付きで追加 (backward compatible).

```python
@dataclass(frozen=True, eq=False)
class QuantumResult:
    psi_final: np.ndarray                 # shape (2**n,) complex128
    t_history: np.ndarray | None          # save_tlist 経路では times と同値
    observables_history: dict[str, np.ndarray]  # name -> shape (K,) 実数
    n_steps: int                          # 実行 step 数
    n_matvec: int                         # 累積 matvec 呼出
    success: bool = True                  # Phase 4: 駆動成功フラグ
    method: str = "m2"                    # Phase 4: 実行 propagator 名
    n_steps_actual: int | None = None     # Phase 4: adaptive 経路の実 step 数
    m_eff_stats: dict[str, int | float] | None = None  # Phase 4 follow-up
    times: np.ndarray | None = None       # Phase 5: 観測時刻軸 (save_tlist)
    states: np.ndarray | None = None      # Phase 5: store_states=True で ψ
    probabilities: np.ndarray | None = None  # Phase 5: |psi_final|^2 (常時)

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

追加履歴:

- `success` / `method` / `n_steps_actual` (adaptive driver 用) → Phase 4 で
  Richardson / M2 embedded 経路と一緒に追加 (済)
- `m_eff_stats` → Phase 4 follow-up (issue #52 A) で追加 (済)
- `times` / `states` / `probabilities` → Phase 5 (issue #47) で追加 (済)
- `AnnealingSimulator` step-wise API → 将来検討 (parent issue #1 の Out of
  scope 表)

`eq=False` は `IsingProblem` と同じ理由 (ndarray フィールドの既定 `__eq__`
が `ValueError` になる)。

### 4.5 `AnnealingSimulator`

任意の中間時刻まで進めて状態を取り出し、観測量で測定して続けて発展させる
用途の step-wise stateful API (Phase 5 C3, issue #48 で実装):

```python
class AnnealingSimulator:
    def __init__(
        self,
        problem: IsingProblem,
        schedule: Schedule,
        psi0: np.ndarray,
        t0: float,
        *,
        method: Literal[
            "m2", "trotter", "trotter_suzuki4", "cfm4", "cfm4_adaptive_richardson_krylov"
        ] = "cfm4",
        m: int = 24,                  # Lanczos 部分空間次元 (QuantumAnnealer と統一)
        krylov_tol: float | None = None,
        # ↓ adaptive (`cfm4_adaptive_richardson_krylov`) 専用; 固定 dt method で
        #   非 None 指定すると ValueError
        atol: float | None = None,
        dt_init: float | None = None,
        dt_max: float | None = None,
        m_max: int | None = None,
    ): ...

    @property
    def t(self) -> float: ...          # 現在時刻
    @property
    def psi(self) -> np.ndarray: ...   # defensive copy で返す ((2**n,) complex128)
    @property
    def n_matvec(self) -> int: ...     # 累積 matvec 数
    @property
    def method(self) -> str: ...

    def step(self, dt: float) -> None:
        """1 step (固定 dt) 進める。adaptive 経路では dt は PI controller
        の proposal 扱い (dt_max=dt で growth を禁じ, reject 時は内部で
        dt を縮めて sub-step 化する; 結果として _t は exactly +dt 進む)."""
        ...

    def advance_to(self, t_target: float, *, n_steps: int | None = None) -> None:
        """t_target まで進める。固定 dt 経路では n_steps 必須 (run と同じ map:
        dt = (t_target - _t) / n_steps), adaptive 経路では n_steps は None
        でなければならない (driver が step 数を内部決定)."""
        ...

    def measure(self, observable: Observable) -> float:
        """現在 ψ で期待値 <ψ|O|ψ> を計算 (実数). Observable 以外は TypeError."""
        ...
```

`QuantumAnnealer.create_simulator(psi0, t0, *, method=..., atol=..., dt_init=...,
dt_max=..., m_max=...)` は QuantumAnnealer インスタンスから派生させる簡便
ファクトリで, `m` / `krylov_tol` は QuantumAnnealer コンストラクタの値を
そのまま引き継ぐ (Simulator 側で異なる値を使いたい場合は `AnnealingSimulator`
を直接構築する).

**実装方針** (`python/kryanneal/simulator.py`): 内部では `evolve_schedule_*`
driver を `[_t, _t + dt]` または `[_t, t_target]` 区間で呼ぶ薄いラッパ.
`step` は `n_steps=1` の driver call, `advance_to` は `run` と同じ driver
call で bit-identical な数値を得る (固定 dt 経路で `rel < 1e-13`).
adaptive 経路の `step(dt)` は `dt0=dt, dt_max=dt` で driver を呼び, PI
controller の reject 時は dt を縮めて内部 sub-step 化する (n_matvec は
driver の `m_eff_history` で正確に累積). `_validate_psi0` は `annealer.py`
の module-level helper を共有し, `QuantumAnnealer.run` と同じ shape /
dtype / L2-normalize 検証を通す.

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
        return cls(problem.H_p_diag.copy())
```

`ising_energy` が `.copy()` を取るのは, `problem` 側を別途参照し続けても
Observable 側の `diag` が独立な実体になることを担保するため (Observable
は内部で diag を共有保持する設計で, copy しないと `problem.H_p_diag`
への副作用が起きうる). `magnetization` の `axis="z"` のみ Phase 5 で対応.

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
    *,
    m: int = 64,                   # Krylov 部分空間次元 (lanczos のみ)
    seed: int | None = None,       # 始ベクトル生成 seed (lanczos のみ)
    krylov_tol: float = 1e-12,     # β 早期打切閾値 (lanczos のみ)
) -> tuple[np.ndarray, np.ndarray]:
    """
    瞬時 H(t) の下位 k 固有値・固有状態を返す.

    Returns
    -------
    eigvals : (k,) real           # 昇順
    eigvecs : (2**n, k) complex128  # 列 j = eigvals[j] の単位固有ベクトル
    """
```

実装方針:

- `method="lanczos"` (default): Python ループから `_rust.apply_h_kryanneal_py`
  を呼んで Krylov 部分空間 (次元 `m`, default 64) を構築し,
  `_rust.tridiag_eigh_py` (`src/tridiag.rs` の hand-rolled QL を thin-wrap)
  で三重対角の完全固有分解を取って下位 `k` 個の Ritz vector を再構築する。
  新規 Rust 関数 (Lanczos kernel) は追加せず, 既存 primitive を Python
  ループで組み合わせる方針 (固有値計算は時間発展に比べて頻度が低く
  Python 越境のオーバヘッドは無視できる)。`m` は時間発展用 (`m ≈ 24`) より
  大きめのデフォルト (64) を取り, Ritz 値の収束を担保する。
- `method="exact"`: 小規模問題 (`n <= 12`) 向け、`_rust.apply_h_kryanneal_py`
  を standard basis `e_j` に当てて `H(t)` の列を 1 本ずつ抽出 (Kronecker
  product より重複コードが無くビット規約の取り違いも避けられる) → Python
  側で `numpy.linalg.eigh` を呼ぶ参照経路 (Rust 経由で LAPACK を呼ばない)。

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

