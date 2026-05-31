---
paths:
  - "tests/**/*.py"
  - "src/**/*.rs"
  - "python/maqina/**/*.py"
---

# テスト

詳細は project skill `.claude/skills/test-runner/SKILL.md` を一次資料とする
(slash command `/test-runner` で発火可能, `test-runner` subagent もここを
読んで実行する). 要点だけ:

```bash
uv run pytest                               # 全 Python テスト
uv run pytest -m "not slow"                 # slow を除外
uv run pytest tests/test_krylov.py          # 個別ファイル
cargo test                                  # Rust 単体 (BLAS feature ON)
cargo test --no-default-features            # scalar fallback
uv run maturin develop --uv                 # Rust 変更後に必須 (--uv は pip 非同梱回避)
```

`uv run` を必ず使う (PyO3 の `extension-module` feature とローカル Python の
ABI を揃える必要があるため、システム Python での実行は避ける)。

## 実行経路 (default: test-runner subagent 経由)

`/solve` 経由か否かに関わらず, **テスト・lint・maturin develop の実行は
原則 `test-runner` subagent (`.claude/agents/test-runner.md`) に委譲する**.
本 agent は `Bash` / `Read` のみ持つ read-only ランナーで, pass/fail サマリと
失敗時の末尾 stdout 抜粋だけを返す. 長い `passed` 列や verbose 出力で
メイン context を圧迫しないことが目的.

並列実行の方針:

- `cargo test` (BLAS on) と `uv run pytest` は **独立** (前者は `target/`,
  後者は既存 `_rust.so` を読むだけ) のため, メイン側から 2 つの test-runner
  agent を **同時起動** して並列化してよい.
- `cargo test` (BLAS on) と `cargo test --no-default-features` は **同じ
  `target/` のロックを争うため実質シリアル化** する. 並列起動するメリットは
  無いので順次実行する.
- `uv run maturin develop --uv` と `uv run pytest` は **serialize 必須**.
  `_rust.so` 上書き中に pytest がロードすると ABI 不整合になる. maturin が
  完了してから pytest を起動する.

直接 Bash で `cargo test` 等を叩くのは, **agent 起動オーバヘッドのほうが
重い極小タスク** (単一テストの再実行など) や, **失敗の生 stdout を逐次見たい
デバッグ局面** に限定する.

## BLAS feature on/off の数値一致検証 (issue #65 Phase 6 C4)

`cargo test --no-default-features` で Rust 内部単体の rel < 1e-13 一致は
担保されるが, **Python 公開 API レベルでの end-to-end 一致** は以下の
artifact フローで検証する:

```bash
# 1. BLAS on build で artifact 生成
uv run maturin develop --uv --release
MAQINA_EXPECT_BLAS=1 uv run pytest tests/test_blas_consistency.py

# 2. BLAS off build に切り替えて再生成 (scalar fallback; rayon/simd は ON 維持)
uv run maturin develop --uv --release --no-default-features \
    --features extension-module,rayon,simd
MAQINA_EXPECT_BLAS=0 uv run pytest tests/test_blas_consistency.py

# 3. 2 つの artifact を diff
uv run python tools/diff_blas_artifacts.py \
    tests/artifacts/blas_on.npz tests/artifacts/blas_off.npz
```

`MAQINA_EXPECT_BLAS` を渡しておくと「誤った build に対する silent 上書き
保存」を防ぐ (build mode と env var が不一致なら test 自身が skip). diff
script は default で rel < 1e-13 / atol < 1e-13 を assert. ローカル切替の
都度 BLAS on/off build を行うため小規模 (n ∈ {4,6,8}) sample のみ.
