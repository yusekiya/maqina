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
//! Phase 3 で `cfm4_step` (CFM4:2 commutator-free Magnus, Alvermann-Fehske 2011).
//!
//! Phase 4 で `cfm4_step_with_m2_estimate` (M2 embedded error 推定子) と
//! `cfm4_step_with_richardson_estimate` (step-doubling Richardson 推定子) を追加.
//! いずれも `docs/design.md` §5 に詳述.
//!
//! BLAS feature 経由で `lanczos_propagate` 内の dim 依存 ops を CBLAS に
//! ディスパッチ (macOS = Apple Accelerate, Linux = system OpenBLAS).
//! 詳細とフォールバックビルド方法は `Cargo.toml` の `[features]` セクション参照.
//!
//! BLAS 経路でビルドされたかどうかは `__has_blas__: bool` 属性で参照可能.
//! Python 側 `kryanneal.krylov` は import 時に本属性を読み, BLAS 無効ビルド
//! (scalar fallback) の場合に `RuntimeWarning` を 1 度だけ発する.
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
mod krylov;
mod matvec;
mod tridiag;
// TODO(phase2): Trotter 経路
// mod trotter;

/// 本拡張が `blas` feature 有効でビルドされたかを示す compile-time フラグ.
/// Python 側からは `_rust.__has_blas__` として参照する.
const HAS_BLAS: bool = cfg!(feature = "blas");

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__has_blas__", HAS_BLAS)?;
    m.add_function(wrap_pyfunction!(matvec::apply_h_kryanneal_py, m)?)?;
    m.add_function(wrap_pyfunction!(krylov::lanczos_propagate_py, m)?)?;
    m.add_function(wrap_pyfunction!(cfm4::m2_midpoint_step_py, m)?)?;
    // TODO(phase2): Trotter
    // m.add_function(wrap_pyfunction!(matvec::apply_single_mode_axis_i_py, m)?)?;
    // m.add_function(wrap_pyfunction!(trotter::trotter_step, m)?)?;
    // TODO(phase3): CFM4:2
    // m.add_function(wrap_pyfunction!(cfm4::cfm4_step, m)?)?;
    // TODO(phase4): adaptive estimators
    // m.add_function(wrap_pyfunction!(cfm4::cfm4_step_with_m2_estimate, m)?)?;
    // m.add_function(wrap_pyfunction!(cfm4::cfm4_step_with_richardson_estimate, m)?)?;
    Ok(())
}
