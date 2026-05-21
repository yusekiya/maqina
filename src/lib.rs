//! `kryanneal._rust`: Rust 実装の Krylov / CFM4 / Trotter / matvec 高速経路.
//!
//! Phase 1 (MVP) では以下を提供する:
//!   - `apply_h_kryanneal` (matvec: bit-flip + 対角積)
//!   - `lanczos_propagate` (matrix-free Lanczos 短時間プロパゲータ)
//!   - `m2_midpoint_step` (M2 中点則 1 step)
//!
//! Phase 2 で Trotter 経路を追加:
//!   - `apply_single_mode_axis_i` (2×2 ユニタリを axis i に in-place 適用)
//!   - `trotter_step` (Strang 2 次 / Suzuki 4 次 1 step)
//!
//! Phase 3 で `cfm4_step` を追加:
//!   - CFM4:2 commutator-free Magnus (Alvermann-Fehske 2011, 2 stage)
//!   - 線形結合 callback 形式 (各 stage の `(c_drv, c_diag)` を畳み込んで
//!     既存 `apply_h_kryanneal` を再利用; `docs/design/05-2-lanczos.md` §5.2 末尾)
//!
//! Phase 4 で `cfm4_step_with_m2_estimate` (M2 embedded error 推定子) と
//! `cfm4_step_with_richardson_estimate` (step-doubling Richardson 推定子) を追加.
//! いずれも `docs/design/INDEX.md` §5 に詳述.
//!
//! BLAS feature 経由で `lanczos_propagate` 内の dim 依存 ops を CBLAS に
//! ディスパッチ (macOS = Apple Accelerate, Linux = system OpenBLAS).
//! 詳細とフォールバックビルド方法は `Cargo.toml` の `[features]` セクション参照.
//!
//! Phase 6 で rayon 並列化 (C1 / issue #62) と SIMD 特化 (C2 / issue #63) の
//! optional feature を追加. それぞれ `rayon` / `simd` feature 経由で有効化し,
//! いずれも default ON. `--no-default-features` で従来の scalar 単スレッド
//! 経路に戻る (`Cargo.toml` の `[features]` 節参照).
//!
//! BLAS / rayon / SIMD 経路でビルドされたかどうかは `__has_blas__`,
//! `__has_rayon__`, `__has_simd__` の各 `bool` 属性で参照可能.
//! Python 側 `kryanneal.krylov` は import 時に `__has_blas__` を読み, BLAS
//! 無効ビルド (scalar fallback) の場合に `RuntimeWarning` を 1 度だけ発する.
//! `__has_rayon__` / `__has_simd__` は bench / 計測時の build profile 確認用.
//!
//! さらに `-C target-cpu=native` (repo 同梱 `.cargo/config.toml` 経由で
//! default 適用; issue #103) の効きを示す `__has_avx2__` / `__has_fma__` /
//! `__has_avx512f__` / `__has_neon__` (`cfg!(target_feature = "...")` 由来) と,
//! ビルドターゲットを示す `__target_arch__` / `__target_os__`
//! (`std::env::consts::ARCH` / `OS`) を expose する. ユーザー側では
//! `kryanneal.show_config()` でこれらを集約 dump できる (numpy.show_config 相当).
//!
//! Python 側 (`kryanneal.krylov`) は本モジュールの import 可否で fast path を
//! 切替える silent-fallback 設計. Rust 拡張がない環境では Python リファレンス
//! 実装で動作する.

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

// `blas-src` (BLAS feature) はリンカに BLAS シンボルを引かせるためだけに
// 必要な「副作用 only」のクレート. 直接 import せずに済む API なので
// `use blas_src as _;` で参照しておくことで, 未参照 crate と判定されて
// build script の link 指示 (Accelerate / OpenBLAS) が落ちる事故を防ぐ.
#[cfg(feature = "blas")]
use blas_src as _;

mod blas;
mod cfm4;
mod chebyshev;
mod krylov;
mod matvec;
mod tridiag;
mod trotter;

/// **In-tree benchmarking / profiling 用の `pub` re-export**.
///
/// `src/bin/perf_apply_h.rs` / `src/bin/perf_trotter_step.rs` /
/// `src/bin/perf_apply_single_mode_axis_i.rs` /
/// `src/bin/perf_cfm4_richardson.rs` / `src/bin/perf_chebyshev.rs`
/// (Linux `perf stat` 等で hardware counter を取るための pure-Rust binary)
/// から `apply_h_kryanneal` / `trotter_step` / `apply_single_mode_axis_i` /
/// `lanczos_propagate` / `cfm4_step_with_richardson_estimate` /
/// `chebyshev_propagate` を呼べるよう公開する. このモジュールは Python 側
/// には露出されない (pyo3 `#[pymodule]` には登録しない).
///
/// Python 経由で呼びたい場合は引き続き `_rust.apply_h_kryanneal_py` /
/// `_rust.trotter_step_py` / `_rust.apply_single_mode_axis_i_inplace_py` /
/// `_rust.lanczos_propagate_py` /
/// `_rust.cfm4_step_with_richardson_estimate_py` (および non-inplace 版) を
/// 使うこと. issue #120 の Chebyshev POC は Phase A scope では Python 公開
/// 不要なため, `chebyshev_propagate` には `_py` wrap を作らず本 bench_api
/// 経由のみで呼び出す.
pub mod bench_api {
    pub use crate::blas::{axpy, dot_conj};
    pub use crate::cfm4::cfm4_step_with_richardson_estimate;
    pub use crate::chebyshev::chebyshev_propagate;
    pub use crate::krylov::lanczos_propagate;
    pub use crate::matvec::apply_h_kryanneal;
    pub use crate::matvec::apply_single_mode_axis_i;
    pub use crate::trotter::trotter_step;
}

/// 本拡張が `blas` feature 有効でビルドされたかを示す compile-time フラグ.
/// Python 側からは `_rust.__has_blas__` として参照する.
const HAS_BLAS: bool = cfg!(feature = "blas");

/// 本拡張が `rayon` feature 有効 (matvec / Trotter primitives の L2 並列化,
/// Phase 6 C1 / issue #62) でビルドされたかを示す compile-time フラグ.
/// Python 側からは `_rust.__has_rayon__` として参照する. bench / 計測時に
/// build profile の確認用途.
const HAS_RAYON: bool = cfg!(feature = "rayon");

/// 本拡張が `simd` feature 有効 (`apply_h_kryanneal` の bit-flip pass i=0,1,2
/// の `wide::f64x4` 特化, Phase 6 C2 / issue #63) でビルドされたかを示す
/// compile-time フラグ. Python 側からは `_rust.__has_simd__` として参照する.
/// `benchmarks/bench_simd_scaling.py` が SIMD ON/OFF build を切り分ける際の
/// 確認用 (実 SIMD 性能向上は build 時の `target-cpu` 設定に依存する点に
/// 注意; `wide` が target_feature を見て scalar fallback / 実 SIMD を選択する).
const HAS_SIMD: bool = cfg!(feature = "simd");

/// ビルド時に `target_feature = "avx2"` が有効だったかを示す compile-time フラグ.
/// Python 側からは `_rust.__has_avx2__` として参照する. repo 同梱の
/// `.cargo/config.toml` 経由で `-C target-cpu=native` が適用されると, x86_64
/// (Zen 3 / Skylake 以降) で `True` になり `wide` クレートが AVX2 dispatch を
/// 選ぶ. issue #103.
const HAS_AVX2: bool = cfg!(target_feature = "avx2");

/// ビルド時に `target_feature = "fma"` が有効だったかを示す compile-time フラグ.
/// `target-cpu=native` 適用時, x86_64 では avx2 とセットで ON になることが多い.
const HAS_FMA: bool = cfg!(target_feature = "fma");

/// ビルド時に `target_feature = "avx512f"` が有効だったかを示す compile-time フラグ.
/// Zen 4 / Sapphire Rapids 以降の `target-cpu=native` で ON になる.
const HAS_AVX512F: bool = cfg!(target_feature = "avx512f");

/// ビルド時に `target_feature = "neon"` が有効だったかを示す compile-time フラグ.
/// aarch64 (Apple Silicon / Armv8) では default ON. `target-cpu=native` 適用とは
/// 独立に True になる点に注意 (NEON は Armv8 ABI で base feature).
const HAS_NEON: bool = cfg!(target_feature = "neon");

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__has_blas__", HAS_BLAS)?;
    m.add("__has_rayon__", HAS_RAYON)?;
    m.add("__has_simd__", HAS_SIMD)?;
    m.add("__has_avx2__", HAS_AVX2)?;
    m.add("__has_fma__", HAS_FMA)?;
    m.add("__has_avx512f__", HAS_AVX512F)?;
    m.add("__has_neon__", HAS_NEON)?;
    m.add("__target_arch__", std::env::consts::ARCH)?;
    m.add("__target_os__", std::env::consts::OS)?;
    m.add_function(wrap_pyfunction!(matvec::apply_h_kryanneal_py, m)?)?;
    m.add_function(wrap_pyfunction!(matvec::apply_h_kryanneal_into_py, m)?)?;
    m.add_function(wrap_pyfunction!(matvec::apply_single_mode_axis_i_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        matvec::apply_single_mode_axis_i_inplace_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(tridiag::tridiag_eigh_py, m)?)?;
    m.add_function(wrap_pyfunction!(krylov::lanczos_propagate_py, m)?)?;
    m.add_function(wrap_pyfunction!(cfm4::m2_midpoint_step_py, m)?)?;
    m.add_function(wrap_pyfunction!(cfm4::m2_midpoint_step_inplace_py, m)?)?;
    m.add_function(wrap_pyfunction!(cfm4::cfm4_step_py, m)?)?;
    m.add_function(wrap_pyfunction!(cfm4::cfm4_step_with_m2_estimate_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        cfm4::cfm4_step_with_richardson_estimate_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(trotter::trotter_step_py, m)?)?;
    m.add_function(wrap_pyfunction!(trotter::trotter_step_inplace_py, m)?)?;
    m.add_function(wrap_pyfunction!(trotter::trotter_suzuki4_step_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        trotter::trotter_suzuki4_step_inplace_py,
        m
    )?)?;
    Ok(())
}
