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

## Requirements

- Python `>=3.13`
- Rust toolchain (`cargo`)
- macOS: Apple Accelerate を自動利用 (追加 install 不要)
- Linux: system OpenBLAS (`libopenblas-dev` 等) が必要 (`--no-default-features` で fallback 可)

## Build / install

```bash
uv sync
uv run maturin develop --uv             # debug build (--uv は uv venv 用必須フラグ)
uv run maturin develop --uv --release   # 性能計測時
```

## Documentation

- 設計書 (一次資料): [`docs/design/INDEX.md`](docs/design/INDEX.md)
- テスト実行手順: [`.claude/skills/test-runner/SKILL.md`](.claude/skills/test-runner/SKILL.md) (Claude Code skill, `/test-runner` で発火可能; `docs/testing.md` はポインタ)
- ベンチマーク: [`docs/benchmarks.md`](docs/benchmarks.md) (Phase 1 では未整備)
- API リファレンス: `python/kryanneal/*.pyi` (per-module PEP 484 stub, full docstring 付き。`tools/gen_api_stubs.py` で自動生成)

## License

未定 (Phase 1)。
