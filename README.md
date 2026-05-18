# kryanneal

**Krylov + Annealing**: 横磁場イジングモデル (TFIM) の量子ダイナミクスを matrix-free に計算するシミュレータ。

- **Krylov 法 (Lanczos)** で短時間プロパゲータを近似
- **CFM4:2 (commutator-free Magnus, 4 次)** で時間依存 Hamiltonian の時間発展を近似
- **adaptive dt ドライバ** (step-doubling Richardson + PI 制御)

Hamiltonian:

```
H(t) = A(s(t)) · H_driver + B(s(t)) · H_problem
H_driver  = -Σ_i h_x_i X_i              (サイト依存横磁場, bit-flip)
H_problem = Z 演算子のみで書かれた k-local 多項式 (Z 基底で対角)
```

設計の参照プロジェクト: [`cv-ising-solver`](https://github.com/Shu-Tanaka-Group/cv-ising-solver)
(同じ Krylov + CFM4:2 カーネルの連続変数版)。

## Performance highlights — Fidelity vs runtime

問題ハミルトニアンのエネルギースケールが揃った **non-stiff** と, 一部の
計算基底に大きなペナルティを加えた **stiff** を比較 (N=18, T=10^4).

| non-stiff | stiff |
|---|---|
| ![non-stiff](docs/figures/0.8.0_pareto_non_stiff.png) | ![stiff](docs/figures/0.8.0_pareto_stiff.png) |

再生成手順: [`benchmarks/README.md`](benchmarks/README.md) "README figure pipeline".

## Requirements

- Python `>=3.13`
- Rust toolchain (`cargo`)
- macOS: Apple Accelerate を自動利用 (追加 install 不要)
- Linux: system OpenBLAS (`libopenblas-dev` 等) が必要 (`--no-default-features` で fallback 可)

## Installation

GitHub からソースビルドして既存プロジェクトに追加する (現状 wheel 配布は
無いためソースビルド経路のみ)。

```bash
# uv を使う場合 (推奨):
uv add 'git+https://github.com/yusekiya/kryanneal'

# pip を使う場合:
pip install 'git+https://github.com/yusekiya/kryanneal'
```

repo 同梱の `.cargo/config.toml` 経由で `-C target-cpu=native` が自動
適用されるため, SIMD 経路 (`wide::f64x4`) が build マシン CPU の AVX2 /
AVX-512 / NEON を最大限活かした状態でインストールされる (詳細は
[`docs/design/11-build-infrastructure.md`](docs/design/11-build-infrastructure.md)
§11.1)。

**生成される `kryanneal._rust.*.so` は build マシン専用バイナリ**となる
ため, 別 CPU マシンへの wheel 再配布には不向き。portable な build を
要するときは `RUSTFLAGS=" "` で override してビルドし直す。

ビルド構成 (どの target feature が有効になったか) は
`kryanneal.show_config()` で確認できる (numpy.show_config() 相当):

```python
>>> import kryanneal
>>> kryanneal.show_config()
kryanneal build configuration
--------------------------------------------------
  version       : 0.8.0
  ...
  target_features (-C target-cpu=native の効きを反映):
    [ON ] avx2
    [ON ] fma
    [off] avx512f
    [off] neon
```

## Documentation

- **Quick start**: [`docs/quickstart.md`](docs/quickstart.md) — 最小例 /
  Observable 時系列計測 / step-wise simulator / 瞬時固有状態の 4 snippet で
  主要 API を一通り使うチュートリアル.
- 設計書 (一次資料): [`docs/design/INDEX.md`](docs/design/INDEX.md)
- テスト実行手順: [`.claude/skills/test-runner/SKILL.md`](.claude/skills/test-runner/SKILL.md) (Claude Code skill, `/test-runner` で発火可能; `docs/testing.md` はポインタ)
- ベンチマーク: [`docs/benchmarks.md`](docs/benchmarks.md) (Phase 1 では未整備)
- API リファレンス: `python/kryanneal/*.pyi` (per-module PEP 484 stub, full docstring 付き。`tools/gen_api_stubs.py` で自動生成)

## Development

repo を clone してローカル開発する手順。`maturin develop` で Rust 拡張
`kryanneal._rust` を `python/kryanneal/` 配下に直接配置する。

```bash
uv sync
uv run maturin develop --uv             # debug build (--uv は uv venv 用必須フラグ)
uv run maturin develop --uv --release   # 性能計測時
```

`--uv` フラグは, maturin が wheel を `pip install` する代わりに
`uv pip install` を使う指定。uv が作る venv には pip が同梱されないため,
`--uv` 無しだと `No module named pip` で失敗する。

`src/*.rs` を変更したあとは `uv run pytest` を回す前に **`uv run
maturin develop --uv` を必ず 1 回回す**こと。忘れると古い `_rust.so` が
読まれて Rust 変更がテストに反映されない。

テスト / lint / build の詳細手順は
[`.claude/skills/test-runner/SKILL.md`](.claude/skills/test-runner/SKILL.md)
を参照 (`/test-runner` で発火可能)。

## License

未定 (Phase 1)。
