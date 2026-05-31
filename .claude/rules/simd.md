---
paths:
  - "src/**/*.rs"
---

# SIMD 経路 (Phase 6 C2 / C2.5, issue #63 / #71)

Phase 6 C2 (issue #63) で `apply_h_kinema` の bit-flip pass の i ∈
{0, 1, 2} を `wide::f64x4` 特化, C2.5 (issue #71) で同じく
`apply_single_mode_axis_i` (Trotter 経路の 2×2 ユニタリ pair update) の
i ∈ {0, 1, 2} を SIMD 特化 (`feature = "simd"`, default ON)。

- SIMD inner kernel は `src/matvec.rs::simd_kernels` モジュールに集約:
  - `bitflip_i{0,1,2}` (C2): `y[k] += coeff · v[k ^ mask]` を broadcast +
    FMA で計算.
  - `single_mode_i{0,1,2}` (C2.5): 2×2 complex matmul を **complex
    broadcast + in-register swizzle** で f64x4 化
    (`u_k · x_pair = splat(u[k].re) · x_pair + [-u[k].im, u[k].im, ...]
    · x_swap` の 2 Complex64 並列, 詳細は `simd_kernels` モジュール docstring).
- `bitflip_iN` は `apply_h_kinema_{serial,rayon}` の両 path から,
  `single_mode_iN` は `apply_single_mode_axis_i_{serial,rayon}` および C3 の
  `apply_fused_axes_to_chunk` (trotter 経路の multi-qubit fusion inner kernel)
  から共通で呼ばれる。per-thread 最適化なので rayon 並列化と直交する。
- rayon path では SIMD カーネルの block-aligned 前提を満たすため
  chunk_size を `SIMD_BLOCK_MAX = 8` Complex64 の倍数に丸める。fused 経路は
  group_block (= 2^(i_start+k)) の倍数で構築されるが,
  `target = dim/(nth·4)` が非 power-of-2 のとき n_groups が奇数になる
  ケースがあり, defensive な alignment check (`chunk.len() % {4,8} == 0`) を
  `apply_fused_axes_to_chunk` の SIMD dispatch に入れている。
- **実 SIMD 性能向上は build 時の `target-cpu` 設定に依存する**: default の
  `x86_64` target では `wide` が scalar fallback ([f64; 4] 相当) を選び
  正確性のみ提供する。issue #103 で **repo 同梱の `.cargo/config.toml` に
  `[build] rustflags = ["-C", "target-cpu=native"]` を入れて default 適用**
  しているため, `uv add git+...` / `cargo build` / `maturin develop` のどの
  経路でも build マシン CPU の AVX2 / AVX-512 / NEON が `wide` の
  `target_feature` cfg で自動的に拾われる。明示的に `RUSTFLAGS` を渡したい
  ときは env 経由が優先されるので override 可。
- **build profile 確認フラグ**: cargo feature 有効化を `_rust.__has_simd__`
  / `__has_rayon__` / `__has_blas__` (各 bool, `cfg!(feature = ...)` 由来),
  target_feature 有効化を `_rust.__has_avx2__` / `__has_fma__` /
  `__has_avx512f__` / `__has_neon__` (各 bool, `cfg!(target_feature = ...)`
  由来), ビルドターゲットを `_rust.__target_arch__` / `__target_os__`
  (各 str, `std::env::consts` 由来) が expose する (`m.add` 経由, build.rs
  不要)。ユーザー向けには `maqina.show_config()` (numpy.show_config 相当)
  でこれらを集約 dump できる (issue #103, 詳細は
  `docs/design/11-build-infrastructure.md` §11.1)。bench スクリプト
  (`bench_simd_scaling.py`) はこれで build を識別する。bench は C2 と C2.5 で
  kernel 軸を分け (`kernel = apply_h_kinema / apply_single_mode_axis_i`),
  C2.5 の per-axis (`i0/i1/i2`) は `mode` 軸で別カラムに展開する。
- **`--no-default-features` ビルド**: SIMD 依存も外れ scalar 経路に戻る。
  `wide` クレートはリンクされない。
