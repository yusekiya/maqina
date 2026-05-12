//! `cfm4.rs`: 中点則 M2 / CFM4:2 / Richardson 推定子.
//!
//! Phase 1 で M2 中点則 1 step:
//!
//! ```text
//! U(t+dt, t) ≈ exp(-i dt · H(t + dt/2))
//! ```
//!
//! 中点で H をフリーズし `lanczos_propagate` を 1 回呼ぶだけの薄いラッパで,
//! LTE ~ O(dt^3). 詳細は `docs/design.md` §5.3 M2 サブセクション.
//!
//! Phase 3 で `cfm4_step` (Alvermann-Fehske 2011 の 4 次 commutator-free
//! Magnus, 2 stage) を追加. ガウス-ルジャンドル 2 点ノード
//! `c_1, c_2 = 1/2 ∓ √3/6` と線形結合係数 `a_high, a_low = 1/4 ± √3/6` を
//! 用い, 各 stage で `(c_drv, c_diag)` スカラ 2 つに畳み込んで既存
//! `apply_h_kryanneal` を呼ぶ「線形結合 callback 形式」を採用
//! (`docs/design.md` §5.2 末尾, §5.3). LTE ~ O(dt^5), per-step matvec は 2m.
//!
//! Phase 4 で `cfm4_step_with_m2_estimate` / `cfm4_step_with_
//! richardson_estimate` を本ファイルに追加予定.
//!
//! 本関数群は `lanczos_propagate` を介して Python に状態を返す **公開
//! プロパゲータ** であり, PyO3 wrap `m2_midpoint_step_py` /
//! `cfm4_step_py` 経由で `_rust` モジュールに exposure される.
//! `lanczos_propagate` 自身は `pub(crate)` のままで, M2 / CFM4:2 が上位
//! wrap として公開する設計 (`docs/design.md` §5.2 末尾).
//!
//! PyO3 の `wrap_pyfunction!` 経由で `_rust` module に登録される関数は
//! Rust の dead_code 解析からは「呼ばれていない」と見えるため, matvec.rs /
//! krylov.rs と同様に module 全体で lint を抑制する (Phase 4 までは内部
//! caller がいない関数本体にも同じ抑制が必要).

#![allow(dead_code)]

use num_complex::Complex64;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::blas::nrm2;
use crate::krylov::lanczos_propagate;
use crate::matvec::apply_h_kryanneal;

/// `psi_new = exp(-i dt · H(t + dt/2)) · psi` を中点則で計算する.
///
/// 時間依存 `H(t) = A(s(t)) · H_driver + B(s(t)) · H_problem` の中点で
/// スケジュール係数を凍結し, `apply_h_kryanneal(·, ·, h_x, h_p_diag, a_mid,
/// b_mid, n)` を closure として `lanczos_propagate` に渡す.
///
/// # 引数
/// * `psi` (length `2^n`): 入力状態.
/// * `h_x` (length `n`): サイト依存横磁場振幅.
/// * `h_p_diag` (length `2^n`): Z 基底での `H_problem` 対角ベクトル.
/// * `a_mid`: 中点でのドライバ係数 `A(s(t + dt/2))` (Python 側で計算済).
/// * `b_mid`: 中点での problem 係数 `B(s(t + dt/2))`.
/// * `dt`: 時刻刻み幅 (real).
/// * `m`: Krylov 部分空間次元 (典型値 24).
/// * `krylov_tol`: Lanczos の β 打切り閾値.
/// * `n`: サイト数. `dim = 2^n` を呼出側と一意に決める.
///
/// # 戻り値
/// * `Ok(psi_new)`: 長さ `2^n` の新状態.
/// * `Err`: `lanczos_propagate` 内で tridiag 固有分解が収束しなかった場合.
///
/// # Panics
/// `lanczos_propagate` / `apply_h_kryanneal` の precondition と同じ
/// (長さ不整合, `m == 0`).
//
// 数値カーネル primitive は cv_ising 流に引数フラットで持つ. 構造体化は
// 将来の `cfm4_step` 系で引数が更に増えた段階で再検討する.
#[allow(clippy::too_many_arguments)]
pub(crate) fn m2_midpoint_step(
    psi: &[Complex64],
    h_x: &[f64],
    h_p_diag: &[f64],
    a_mid: f64,
    b_mid: f64,
    dt: f64,
    m: usize,
    krylov_tol: f64,
    n: usize,
) -> PyResult<Vec<Complex64>> {
    let matvec = |v: &[Complex64], y: &mut [Complex64]| {
        apply_h_kryanneal(v, y, h_x, h_p_diag, a_mid, b_mid, n);
    };
    lanczos_propagate(matvec, psi, dt, m, krylov_tol)
}

/// `m2_midpoint_step` の Python wrap.
///
/// Python 側 (`_rust.m2_midpoint_step_py`) からは
///
/// ```python
/// psi_new = _rust.m2_midpoint_step_py(
///     psi, h_x, h_p_diag, a_mid, b_mid, dt, m, krylov_tol,
/// )
/// ```
///
/// として呼ぶ. サイト数 `n = len(h_x)` / 状態次元 `dim = 2^n` は
/// `len(h_p_diag)` から取り出し, 整合性を検証する.
#[pyfunction]
#[pyo3(signature = (psi, h_x, h_p_diag, a_mid, b_mid, dt, m, krylov_tol))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn m2_midpoint_step_py<'py>(
    py: Python<'py>,
    psi: PyReadonlyArray1<'py, Complex64>,
    h_x: PyReadonlyArray1<'py, f64>,
    h_p_diag: PyReadonlyArray1<'py, f64>,
    a_mid: f64,
    b_mid: f64,
    dt: f64,
    m: usize,
    krylov_tol: f64,
) -> PyResult<Bound<'py, PyArray1<Complex64>>> {
    let psi_slice = psi.as_slice()?;
    let h_x_slice = h_x.as_slice()?;
    let h_p_diag_slice = h_p_diag.as_slice()?;

    let n = h_x_slice.len();
    let dim = 1usize << n;
    if h_p_diag_slice.len() != dim {
        return Err(PyValueError::new_err(format!(
            "h_p_diag length {} does not match 2^len(h_x) = 2^{} = {}",
            h_p_diag_slice.len(),
            n,
            dim,
        )));
    }
    if psi_slice.len() != dim {
        return Err(PyValueError::new_err(format!(
            "psi length {} does not match 2^len(h_x) = {}",
            psi_slice.len(),
            dim,
        )));
    }
    if m == 0 {
        return Err(PyValueError::new_err("m must be >= 1"));
    }

    let psi_new = m2_midpoint_step(
        psi_slice,
        h_x_slice,
        h_p_diag_slice,
        a_mid,
        b_mid,
        dt,
        m,
        krylov_tol,
        n,
    )?;
    Ok(psi_new.into_pyarray(py))
}

/// CFM4:2 のガウス-ルジャンドル 2 点求積ノード `c_1 = 1/2 - √3/6`.
///
/// `f64::sqrt` は const fn ではないため fn 経由で公開する. 1 step あたり
/// 数回しか評価されないので runtime cost は無視できる (`trotter::suzuki4_p`
/// と同じパターン).
#[inline]
pub(crate) fn cfm4_c1() -> f64 {
    0.5 - 3.0_f64.sqrt() / 6.0
}

/// CFM4:2 のガウス-ルジャンドル 2 点求積ノード `c_2 = 1/2 + √3/6`.
#[inline]
pub(crate) fn cfm4_c2() -> f64 {
    0.5 + 3.0_f64.sqrt() / 6.0
}

/// CFM4:2 の線形結合係数 `a_high = 1/4 + √3/6` (Alvermann-Fehske 2011).
#[inline]
pub(crate) fn cfm4_a_high() -> f64 {
    0.25 + 3.0_f64.sqrt() / 6.0
}

/// CFM4:2 の線形結合係数 `a_low = 1/4 - √3/6`.
#[inline]
pub(crate) fn cfm4_a_low() -> f64 {
    0.25 - 3.0_f64.sqrt() / 6.0
}

/// `psi_new = exp(-i dt · B_2) · exp(-i dt · B_1) · psi` を CFM4:2 で計算する.
///
/// Alvermann-Fehske (2011) の 4 次 commutator-free Magnus を 1 step 適用する.
/// 時間依存 `H(t) = A(s(t)) · H_driver + B(s(t)) · H_problem` のガウス-
/// ルジャンドル 2 点ノード `t_1 = t + c_1·dt`, `t_2 = t + c_2·dt` でスケジュ
/// ール係数 `(A_1, B_1) = (a_s1, b_s1)` と `(A_2, B_2) = (a_s2, b_s2)` を
/// **呼出側で pre-eval** することを前提とする.
///
/// ステージごとに以下の Hamiltonian で `lanczos_propagate` を 1 回ずつ呼ぶ:
///
/// ```text
/// stage 1 : c_drv  = a_high·A_1 + a_low ·A_2
///           c_diag = a_high·B_1 + a_low ·B_2
/// stage 2 : c_drv  = a_low ·A_1 + a_high·A_2
///           c_diag = a_low ·B_1 + a_high·B_2
/// ```
///
/// 各 stage は `apply_h_kryanneal(·, ·, h_x, h_p_diag, c_drv, c_diag, n)` を
/// closure として `lanczos_propagate` に渡す「線形結合 callback 形式」
/// (`docs/design.md` §5.2 末尾) で実装される. これにより Lanczos 2 回 / step,
/// per-step matvec は 2m, LTE ~ O(dt^5).
///
/// # 引数
/// * `psi` (length `2^n`): 入力状態.
/// * `h_x` (length `n`): サイト依存横磁場振幅.
/// * `h_p_diag` (length `2^n`): Z 基底での `H_problem` 対角ベクトル.
/// * `a_s1`, `b_s1`: ノード `t_1` でのスケジュール係数 `A(s(t + c_1·dt))`,
///   `B(s(t + c_1·dt))` (Python 側で計算済).
/// * `a_s2`, `b_s2`: ノード `t_2` でのスケジュール係数 (同上).
/// * `dt`: 時刻刻み幅 (real).
/// * `m`: Krylov 部分空間次元 (典型値 24).
/// * `krylov_tol`: Lanczos の β 打切り閾値.
/// * `n`: サイト数. `dim = 2^n` を呼出側と一意に決める.
///
/// # 戻り値
/// * `Ok(psi_new)`: 長さ `2^n` の新状態.
/// * `Err`: `lanczos_propagate` 内で tridiag 固有分解が収束しなかった場合.
///
/// # Panics
/// `lanczos_propagate` / `apply_h_kryanneal` の precondition と同じ
/// (長さ不整合, `m == 0`).
//
// 数値カーネル primitive は cv_ising 流に引数フラットで持つ. 構造体化は
// 将来の adaptive 経路 (`cfm4_step_with_*_estimate`) で引数が更に増えた
// 段階で再検討する.
#[allow(clippy::too_many_arguments)]
pub(crate) fn cfm4_step(
    psi: &[Complex64],
    h_x: &[f64],
    h_p_diag: &[f64],
    a_s1: f64,
    b_s1: f64,
    a_s2: f64,
    b_s2: f64,
    dt: f64,
    m: usize,
    krylov_tol: f64,
    n: usize,
) -> PyResult<Vec<Complex64>> {
    let a_high = cfm4_a_high();
    let a_low = cfm4_a_low();

    // stage 1: B_1 = a_high · H_1 + a_low · H_2 を (c_drv_1, c_diag_1) に
    // 畳み込んで Lanczos 1 回.
    let c_drv_1 = a_high * a_s1 + a_low * a_s2;
    let c_diag_1 = a_high * b_s1 + a_low * b_s2;
    let psi_mid = {
        let matvec = |v: &[Complex64], y: &mut [Complex64]| {
            apply_h_kryanneal(v, y, h_x, h_p_diag, c_drv_1, c_diag_1, n);
        };
        lanczos_propagate(matvec, psi, dt, m, krylov_tol)?
    };

    // stage 2: B_2 = a_low · H_1 + a_high · H_2 を (c_drv_2, c_diag_2) に
    // 畳み込んで Lanczos もう 1 回.
    let c_drv_2 = a_low * a_s1 + a_high * a_s2;
    let c_diag_2 = a_low * b_s1 + a_high * b_s2;
    let matvec = |v: &[Complex64], y: &mut [Complex64]| {
        apply_h_kryanneal(v, y, h_x, h_p_diag, c_drv_2, c_diag_2, n);
    };
    lanczos_propagate(matvec, &psi_mid, dt, m, krylov_tol)
}

/// `cfm4_step` の Python wrap.
///
/// Python 側 (`_rust.cfm4_step_py`) からは
///
/// ```python
/// psi_new = _rust.cfm4_step_py(
///     psi, h_x, h_p_diag,
///     a_s1, b_s1, a_s2, b_s2,
///     dt, m, krylov_tol,
/// )
/// ```
///
/// として呼ぶ. サイト数 `n = len(h_x)` / 状態次元 `dim = 2^n` は
/// `len(h_p_diag)` から取り出し, 整合性を検証する.
#[pyfunction]
#[pyo3(signature = (psi, h_x, h_p_diag, a_s1, b_s1, a_s2, b_s2, dt, m, krylov_tol))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn cfm4_step_py<'py>(
    py: Python<'py>,
    psi: PyReadonlyArray1<'py, Complex64>,
    h_x: PyReadonlyArray1<'py, f64>,
    h_p_diag: PyReadonlyArray1<'py, f64>,
    a_s1: f64,
    b_s1: f64,
    a_s2: f64,
    b_s2: f64,
    dt: f64,
    m: usize,
    krylov_tol: f64,
) -> PyResult<Bound<'py, PyArray1<Complex64>>> {
    let psi_slice = psi.as_slice()?;
    let h_x_slice = h_x.as_slice()?;
    let h_p_diag_slice = h_p_diag.as_slice()?;

    let n = h_x_slice.len();
    let dim = 1usize << n;
    if h_p_diag_slice.len() != dim {
        return Err(PyValueError::new_err(format!(
            "h_p_diag length {} does not match 2^len(h_x) = 2^{} = {}",
            h_p_diag_slice.len(),
            n,
            dim,
        )));
    }
    if psi_slice.len() != dim {
        return Err(PyValueError::new_err(format!(
            "psi length {} does not match 2^len(h_x) = {}",
            psi_slice.len(),
            dim,
        )));
    }
    if m == 0 {
        return Err(PyValueError::new_err("m must be >= 1"));
    }

    let psi_new = cfm4_step(
        psi_slice,
        h_x_slice,
        h_p_diag_slice,
        a_s1,
        b_s1,
        a_s2,
        b_s2,
        dt,
        m,
        krylov_tol,
        n,
    )?;
    Ok(psi_new.into_pyarray(py))
}

/// CFM4:2 step を `cfm4_step` と同一の Hamiltonian / dt / Lanczos パラメータで
/// 走らせると同時に, **同じ入口 ψ** に対して中点則 M2 を 1 step 走らせ,
///
/// ```text
/// err = ‖ψ_cfm4 - ψ_m2‖_2
/// ```
///
/// を embedded local error 推定値として返す. M2 の LTE が `O(dt^3)`,
/// CFM4:2 の LTE が `O(dt^5)` なので, 小 dt 領域では
/// `‖ψ_cfm4 - ψ_m2‖ ≈ ‖ψ_m2 - ψ_exact‖ ∝ dt^3` となり, PI controller の
/// `p = 2` 指数 `dt_next = dt · safety · (tol/err)^{1/(p+1)}` で扱える
/// 2 次の推定子になる (`docs/design.md` §5.3 PI controller 表).
///
/// per-step matvec は CFM4:2 の 2m + M2 の m = **3m** (固定 dt CFM4:2 比 1.5×).
///
/// `psi` は in-place に **CFM4:2 後の値** で上書きされる (固定 dt
/// `cfm4_step` 単体呼び出しと bit-exact 一致). M2 経路の状態は err 算出にのみ
/// 使い破棄される.
///
/// # 引数
/// * `psi` (length `2^n`): 入出力状態. 入口で読まれ, 出口で CFM4:2 結果に
///   in-place 更新される.
/// * `h_x` (length `n`), `h_p_diag` (length `2^n`): operator 部分.
/// * `a_s1`, `b_s1`: ノード `t_1 = t + c_1·dt` でのスケジュール係数 (CFM4:2 stage 1).
/// * `a_s2`, `b_s2`: ノード `t_2 = t + c_2·dt` でのスケジュール係数 (CFM4:2 stage 2).
/// * `a_mid`, `b_mid`: 中点 `t + dt/2` でのスケジュール係数 (M2 用). 呼出側
///   (Python driver) が schedule から事前評価して渡す.
/// * `dt`, `m`, `krylov_tol`, `n`: 既存カーネル primitive と同義.
///
/// # 戻り値
/// * `Ok(err)`: `‖ψ_cfm4 - ψ_m2‖_2` (real, non-negative).
/// * `Err`: 内部 `lanczos_propagate` で tridiag 固有分解が収束しなかった場合
///   (CFM4:2 / M2 いずれの段でも propagate).
///
/// # Panics
/// `cfm4_step` / `m2_midpoint_step` の precondition と同じ.
//
// 数値カーネル primitive は cv_ising 流に引数フラットで持つ. Phase 4 で
// adaptive driver 経路が固まった後 (引数が更に増えるなら) 構造体化を再検討.
#[allow(clippy::too_many_arguments)]
pub(crate) fn cfm4_step_with_m2_estimate(
    psi: &mut [Complex64],
    h_x: &[f64],
    h_p_diag: &[f64],
    a_s1: f64,
    b_s1: f64,
    a_s2: f64,
    b_s2: f64,
    a_mid: f64,
    b_mid: f64,
    dt: f64,
    m: usize,
    krylov_tol: f64,
    n: usize,
) -> PyResult<f64> {
    // `cfm4_step` / `m2_midpoint_step` ともに `psi: &[Complex64]` を受けて
    // 新規 Vec を返す所有権モデルなので, 明示 clone は不要 (内部の Lanczos が
    // 各々 ワークバッファ を確保する). 同じ `psi` を 2 度 immutable に
    // 借りる形で「同じ入口 ψ」契約を満たす.
    let psi_cfm4 = cfm4_step(
        psi, h_x, h_p_diag, a_s1, b_s1, a_s2, b_s2, dt, m, krylov_tol, n,
    )?;
    let psi_m2 = m2_midpoint_step(psi, h_x, h_p_diag, a_mid, b_mid, dt, m, krylov_tol, n)?;

    // err = ‖ψ_cfm4 - ψ_m2‖_2. 差ベクトルを一度組んで `blas::nrm2` を通す
    // ことで BLAS feature on/off いずれの経路でも同一実装で算出する.
    let diff: Vec<Complex64> = psi_cfm4
        .iter()
        .zip(psi_m2.iter())
        .map(|(a, b)| a - b)
        .collect();
    let err = nrm2(&diff);

    // CFM4:2 後の psi を呼出側へ書き戻す. `cfm4_step` 単体と bit-exact 一致.
    psi.copy_from_slice(&psi_cfm4);
    Ok(err)
}

/// `cfm4_step_with_m2_estimate` の Python wrap.
///
/// Python 側 (`_rust.cfm4_step_with_m2_estimate_py`) からは
///
/// ```python
/// psi_new, err = _rust.cfm4_step_with_m2_estimate_py(
///     psi, h_x, h_p_diag,
///     a_s1, b_s1, a_s2, b_s2,
///     a_mid, b_mid,
///     dt, m, krylov_tol,
/// )
/// ```
///
/// として呼ぶ. 戻り値は `(ψ_cfm4, err)` のタプル. `psi` は不変借りで受け
/// 取り (Python 側で `psi.copy()` を作っているとは限らない契約), 新規
/// `np.ndarray` を返す. サイト数 `n = len(h_x)` / 状態次元 `dim = 2^n` は
/// `len(h_p_diag)` から取り出し, 整合性を検証する.
#[pyfunction]
#[pyo3(signature = (psi, h_x, h_p_diag, a_s1, b_s1, a_s2, b_s2, a_mid, b_mid, dt, m, krylov_tol))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn cfm4_step_with_m2_estimate_py<'py>(
    py: Python<'py>,
    psi: PyReadonlyArray1<'py, Complex64>,
    h_x: PyReadonlyArray1<'py, f64>,
    h_p_diag: PyReadonlyArray1<'py, f64>,
    a_s1: f64,
    b_s1: f64,
    a_s2: f64,
    b_s2: f64,
    a_mid: f64,
    b_mid: f64,
    dt: f64,
    m: usize,
    krylov_tol: f64,
) -> PyResult<(Bound<'py, PyArray1<Complex64>>, f64)> {
    let psi_slice = psi.as_slice()?;
    let h_x_slice = h_x.as_slice()?;
    let h_p_diag_slice = h_p_diag.as_slice()?;

    let n = h_x_slice.len();
    let dim = 1usize << n;
    if h_p_diag_slice.len() != dim {
        return Err(PyValueError::new_err(format!(
            "h_p_diag length {} does not match 2^len(h_x) = 2^{} = {}",
            h_p_diag_slice.len(),
            n,
            dim,
        )));
    }
    if psi_slice.len() != dim {
        return Err(PyValueError::new_err(format!(
            "psi length {} does not match 2^len(h_x) = {}",
            psi_slice.len(),
            dim,
        )));
    }
    if m == 0 {
        return Err(PyValueError::new_err("m must be >= 1"));
    }

    // Python 側の `psi` を所有付き Vec へ複製し, in-place API に渡す.
    // PyReadonlyArray1 は immutable borrow しか提供しないため明示 clone.
    let mut psi_owned: Vec<Complex64> = psi_slice.to_vec();
    let err = cfm4_step_with_m2_estimate(
        &mut psi_owned,
        h_x_slice,
        h_p_diag_slice,
        a_s1,
        b_s1,
        a_s2,
        b_s2,
        a_mid,
        b_mid,
        dt,
        m,
        krylov_tol,
        n,
    )?;
    Ok((psi_owned.into_pyarray(py), err))
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, SymmetricEigen};

    use crate::blas::nrm2;

    /// 軽量決定論的 PRNG (xorshift64). matvec / krylov のテストと同実装を
    /// 再掲 (Rust 単体テストは同じファイル内に置く方針, CLAUDE.md 参照).
    struct Xor64(u64);

    impl Xor64 {
        fn new(seed: u64) -> Self {
            let s = if seed == 0 {
                0xdead_beef_cafe_babe
            } else {
                seed
            };
            Self(s)
        }

        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }

        fn signed(&mut self) -> f64 {
            const SCALE: f64 = 1.0 / (1u64 << 53) as f64;
            let u = (self.next_u64() >> 11) as f64 * SCALE;
            2.0 * u - 1.0
        }

        fn complex_signed(&mut self) -> Complex64 {
            Complex64::new(self.signed(), self.signed())
        }
    }

    fn random_complex_vec(n: usize, seed: u64) -> Vec<Complex64> {
        let mut rng = Xor64::new(seed);
        (0..n).map(|_| rng.complex_signed()).collect()
    }

    /// `‖a - b‖ / max(‖b‖, 1)`.
    fn relative_error(actual: &[Complex64], expected: &[Complex64]) -> f64 {
        assert_eq!(actual.len(), expected.len());
        let mut diff_sq = 0.0_f64;
        let mut ref_sq = 0.0_f64;
        for k in 0..actual.len() {
            let d = actual[k] - expected[k];
            diff_sq += d.norm_sqr();
            ref_sq += expected[k].norm_sqr();
        }
        diff_sq.sqrt() / ref_sq.sqrt().max(1.0)
    }

    /// `(a_t, b_t, h_x, h_p_diag)` から `dim × dim` の実 dense Hamiltonian を
    /// 構築する. H_driver (X 演算子の和) も diag も実係数のため H 全体は
    /// 実対称.
    fn build_dense_h_real(
        n: usize,
        h_x: &[f64],
        h_p_diag: &[f64],
        a_t: f64,
        b_t: f64,
    ) -> DMatrix<f64> {
        let dim = 1usize << n;
        let mut h = DMatrix::<f64>::zeros(dim, dim);
        for k in 0..dim {
            h[(k, k)] = b_t * h_p_diag[k];
        }
        for (i, &h_x_i) in h_x.iter().enumerate() {
            let coeff = -a_t * h_x_i;
            let mask = 1usize << i;
            for k in 0..dim {
                h[(k, k ^ mask)] += coeff;
            }
        }
        h
    }

    /// 実対称 H の eigendecomp を経由した `exp(-i dt H) · ψ` の参照計算.
    /// `H = U Λ Uᵀ` (U の列が固有ベクトル) より
    /// `ψ_new = U · diag(exp(-i dt λ)) · Uᵀ · ψ`.
    fn reference_propagate_real_h(h: &DMatrix<f64>, psi: &[Complex64], dt: f64) -> Vec<Complex64> {
        let dim = h.nrows();
        assert_eq!(h.ncols(), dim);
        let eig = SymmetricEigen::new(h.clone());
        let lam = &eig.eigenvalues;
        let u = &eig.eigenvectors;

        let mut c = vec![Complex64::new(0.0, 0.0); dim];
        for j in 0..dim {
            let mut s = Complex64::new(0.0, 0.0);
            for l in 0..dim {
                s += Complex64::new(u[(l, j)], 0.0) * psi[l];
            }
            let phase = Complex64::new(0.0, -dt * lam[j]).exp();
            c[j] = s * phase;
        }
        let mut psi_new = vec![Complex64::new(0.0, 0.0); dim];
        for k in 0..dim {
            let mut s = Complex64::new(0.0, 0.0);
            for j in 0..dim {
                s += Complex64::new(u[(k, j)], 0.0) * c[j];
            }
            psi_new[k] = s;
        }
        psi_new
    }

    /// dt = 0 で恒等変換になる: `exp(-i · 0 · H) · ψ = ψ`.
    /// Lanczos は dt = 0 のとき位相 `exp(0) = 1` を返すので, 部分空間内の
    /// 数値誤差のみが残る. rel < 1e-13 で一致するはず.
    #[test]
    fn dt_zero_is_identity() {
        let n = 4_usize;
        let dim = 1usize << n;
        let mut rng = Xor64::new(31);
        let h_x: Vec<f64> = (0..n).map(|_| rng.signed()).collect();
        let h_p_diag: Vec<f64> = (0..dim).map(|_| rng.signed()).collect();
        let psi = random_complex_vec(dim, 71);
        let a_mid = rng.signed();
        let b_mid = rng.signed();

        let result =
            m2_midpoint_step(&psi, &h_x, &h_p_diag, a_mid, b_mid, 0.0, 24, 1e-12, n).expect("ok");
        let rel = relative_error(&result, &psi);
        assert!(rel < 1e-13, "dt=0 rel = {}", rel);
    }

    /// `exp(-i dt H)` は unitary なので ‖ψ_new‖ = ‖ψ‖.
    #[test]
    fn hermitian_h_preserves_norm() {
        let n = 4_usize;
        let dim = 1usize << n;
        let mut rng = Xor64::new(127);
        let h_x: Vec<f64> = (0..n).map(|_| rng.signed()).collect();
        let h_p_diag: Vec<f64> = (0..dim).map(|_| rng.signed()).collect();
        let psi = random_complex_vec(dim, 211);
        let psi_norm = nrm2(&psi);
        let a_mid = 0.4_f64;
        let b_mid = 1.1_f64;
        let dt = 0.3_f64;

        let result =
            m2_midpoint_step(&psi, &h_x, &h_p_diag, a_mid, b_mid, dt, 24, 1e-12, n).expect("ok");
        let new_norm = nrm2(&result);
        let rel = (new_norm - psi_norm).abs() / psi_norm.max(1.0);
        assert!(
            rel < 1e-13,
            "norm rel = {} (before={}, after={})",
            rel,
            psi_norm,
            new_norm,
        );
    }

    /// CFM4:2 のガウス-ルジャンドル 2 点ノードと線形結合係数が以下の
    /// 不変量を満たすことを確認する:
    ///
    /// - `c_1 + c_2 = 1` (区間 `[t, t+dt]` の中点 `t + dt/2` を中心に対称)
    /// - `a_high + a_low = 1/2` (`B_1 + B_2 = (1/2)(H_1 + H_2)` で各 stage の
    ///   重みが釣り合う)
    ///
    /// 数値定数の typo (φ vs √3/6 など) を即座に検出するためのガード.
    #[test]
    fn cfm4_coefficients_match_formula() {
        let c_1 = cfm4_c1();
        let c_2 = cfm4_c2();
        let a_high = cfm4_a_high();
        let a_low = cfm4_a_low();

        assert!(
            (c_1 + c_2 - 1.0).abs() < 1e-15,
            "c_1 + c_2 = {} (expected 1.0)",
            c_1 + c_2,
        );
        assert!(
            (a_high + a_low - 0.5).abs() < 1e-15,
            "a_high + a_low = {} (expected 0.5)",
            a_high + a_low,
        );
        // ノード順序 (c_1 < 1/2 < c_2) と高低係数 (a_low < 1/4 < a_high) も
        // 念のため確認.
        assert!(c_1 < 0.5 && c_2 > 0.5);
        assert!(a_low < 0.25 && a_high > 0.25);
    }

    /// `dt = 0` で恒等変換: `exp(-i · 0 · B_2) · exp(-i · 0 · B_1) · ψ = ψ`.
    /// Lanczos は dt=0 で位相 1 を返すので部分空間内の数値誤差のみ残る.
    #[test]
    fn cfm4_dt_zero_is_identity() {
        let n = 4_usize;
        let dim = 1usize << n;
        let mut rng = Xor64::new(913);
        let h_x: Vec<f64> = (0..n).map(|_| rng.signed()).collect();
        let h_p_diag: Vec<f64> = (0..dim).map(|_| rng.signed()).collect();
        let psi = random_complex_vec(dim, 41);
        let (a_s1, b_s1) = (rng.signed(), rng.signed());
        let (a_s2, b_s2) = (rng.signed(), rng.signed());

        let result = cfm4_step(
            &psi, &h_x, &h_p_diag, a_s1, b_s1, a_s2, b_s2, 0.0, 24, 1e-12, n,
        )
        .expect("ok");
        let rel = relative_error(&result, &psi);
        assert!(rel < 1e-13, "dt=0 rel = {}", rel);
    }

    /// `exp(-i dt B_k)` は unitary なので 2 段合成も unitary,
    /// よって `‖ψ_new‖ = ‖ψ‖` が machine precision で保たれる.
    #[test]
    fn cfm4_hermitian_h_preserves_norm() {
        let n = 4_usize;
        let dim = 1usize << n;
        let mut rng = Xor64::new(347);
        let h_x: Vec<f64> = (0..n).map(|_| rng.signed()).collect();
        let h_p_diag: Vec<f64> = (0..dim).map(|_| rng.signed()).collect();
        let psi = random_complex_vec(dim, 911);
        let psi_norm = nrm2(&psi);
        let (a_s1, b_s1) = (0.4_f64, 1.1_f64);
        let (a_s2, b_s2) = (0.7_f64, 0.3_f64);
        let dt = 0.25_f64;

        let result = cfm4_step(
            &psi, &h_x, &h_p_diag, a_s1, b_s1, a_s2, b_s2, dt, 24, 1e-12, n,
        )
        .expect("ok");
        let new_norm = nrm2(&result);
        let rel = (new_norm - psi_norm).abs() / psi_norm.max(1.0);
        assert!(
            rel < 1e-13,
            "norm rel = {} (before={}, after={})",
            rel,
            psi_norm,
            new_norm,
        );
    }

    /// time-independent H (`a_s1 == a_s2`, `b_s1 == b_s2`) で 1 step の CFM4:2 が
    /// 厳密に `exp(-i dt H) · ψ` を返すことを確認する.
    ///
    /// 定数係数だと `B_1 = a_high·H + a_low·H = H/2`, `B_2 = H/2` なので
    /// `U = exp(-i dt H/2) · exp(-i dt H/2) = exp(-i dt H)` (交換するので
    /// 厳密に等しい). Lanczos 誤差のみ残り `rel < 1e-10` を要求.
    #[test]
    fn cfm4_time_independent_matches_exact_chain() {
        let n = 4_usize;
        let dim = 1usize << n;
        let mut rng = Xor64::new(0xc0ffee);
        let h_x: Vec<f64> = (0..n).map(|_| rng.signed()).collect();
        let h_p_diag: Vec<f64> = (0..dim).map(|_| rng.signed()).collect();
        let psi0 = random_complex_vec(dim, 0xdada);

        let a_t = rng.signed();
        let b_t = rng.signed();

        let n_steps = 100_usize;
        let total_t = 1.0_f64;
        let dt = total_t / n_steps as f64;

        let mut psi = psi0.clone();
        for _ in 0..n_steps {
            psi =
                cfm4_step(&psi, &h_x, &h_p_diag, a_t, b_t, a_t, b_t, dt, 24, 1e-14, n).expect("ok");
        }

        let h_real = build_dense_h_real(n, &h_x, &h_p_diag, a_t, b_t);
        let expected = reference_propagate_real_h(&h_real, &psi0, total_t);

        let rel = relative_error(&psi, &expected);
        assert!(rel < 1e-10, "chain n_steps={} rel = {}", n_steps, rel);
    }

    /// CFM4:2 の global error が O(dt^4) (LTE O(dt^5)) にスケールすることを
    /// 確認する.
    ///
    /// 検証技法: `H(t) = f(t) · H_0` (時間依存性が単一の operator に
    /// 比例する) 構造を使う. CFM4:2 ステージは
    ///
    /// ```text
    /// B_1 = (a_high·f(t_1) + a_low ·f(t_2)) · H_0
    /// B_2 = (a_low ·f(t_1) + a_high·f(t_2)) · H_0
    /// ```
    ///
    /// で両者とも H_0 の倍数なので可換, 1 step あたりの propagator は
    ///
    /// ```text
    /// U_step = exp(-i dt · (1/2)(f(t_1) + f(t_2)) · H_0)
    /// ```
    ///
    /// これは f に対するガウス-ルジャンドル 2 点求積で, ステップ全体では
    /// 区分的ガウス積分による `∫_0^T f(τ)dτ` の近似に等しい. f が滑らか
    /// なら積分誤差は global O(dt^4), per-step LTE O(dt^5).
    ///
    /// 参照解は `exp(-i · F_T · H_0) · ψ_0`, `F_T = ∫_0^T f(τ)dτ` (`H_0` が
    /// 時間に依らず一定なので, 演算子は単に `F_T` 倍された H_0 の指数で
    /// 表せる).
    ///
    /// `f(t) = sin(t)` を採れば `F_T = 1 - cos(T)` で解析的. dt 半減で
    /// global error が 16 倍程度改善するはず.
    #[test]
    fn cfm4_global_error_order_4_on_commuting_time_dependent_h() {
        let n = 3_usize;
        let dim = 1usize << n;
        let mut rng = Xor64::new(424242);
        let h_x: Vec<f64> = (0..n).map(|_| rng.signed()).collect();
        let h_p_diag: Vec<f64> = (0..dim).map(|_| rng.signed()).collect();
        let psi_raw = random_complex_vec(dim, 0xface);
        let norm = nrm2(&psi_raw);
        let psi0: Vec<Complex64> = psi_raw.iter().map(|c| *c / norm).collect();

        let a_base = 0.4_f64;
        let b_base = 0.7_f64;
        let f = |t: f64| t.sin();
        let total_t = 0.5_f64;
        let f_integral = 1.0 - total_t.cos();

        let h_real = build_dense_h_real(n, &h_x, &h_p_diag, a_base, b_base);
        let expected = reference_propagate_real_h(&h_real, &psi0, f_integral);

        let m = 24_usize;
        let krylov_tol = 1e-14;
        let c_1 = cfm4_c1();
        let c_2 = cfm4_c2();

        let run_chain = |n_steps: usize| -> Vec<Complex64> {
            let dt = total_t / n_steps as f64;
            let mut psi = psi0.clone();
            for k in 0..n_steps {
                let t_k = k as f64 * dt;
                let s_1 = t_k + c_1 * dt;
                let s_2 = t_k + c_2 * dt;
                let a_s1 = a_base * f(s_1);
                let b_s1 = b_base * f(s_1);
                let a_s2 = a_base * f(s_2);
                let b_s2 = b_base * f(s_2);
                psi = cfm4_step(
                    &psi, &h_x, &h_p_diag, a_s1, b_s1, a_s2, b_s2, dt, m, krylov_tol, n,
                )
                .expect("ok");
            }
            psi
        };

        let err_coarse = relative_error(&run_chain(16), &expected);
        let err_fine = relative_error(&run_chain(32), &expected);
        let ratio = err_coarse / err_fine;

        // global O(dt^4) → ratio ~ 16. 数値雑音による上下ブレを許して [10, 32]
        // を許容. err_fine は十分小さい (~1e-9) はずなので Lanczos 床は問題に
        // ならない.
        assert!(
            ratio > 10.0 && ratio < 32.0,
            "order ratio = {} (err_coarse = {}, err_fine = {})",
            ratio,
            err_coarse,
            err_fine,
        );
        assert!(err_fine < 1e-6, "fine err = {}", err_fine);
    }

    /// 同じ dt / 同じ schedule で CFM4:2 が M2 中点則より厳密に高精度な
    /// ことを確認する (LTE オーダ違い: M2 は O(dt^3), CFM4:2 は O(dt^5)).
    ///
    /// 検証は `cfm4_global_error_order_4_on_commuting_time_dependent_h` と
    /// 同じ可換時間依存 `H(t) = f(t) · H_0` 構造を流用. CFM4:2 は
    /// ガウス-ルジャンドル 2 点求積, M2 は単純な中点則になるので, 同じ
    /// `f(t) = sin(t)` ramp に対して両者の積分近似誤差をそのまま比較できる.
    /// dt = 1/32 (T=0.5, n_steps=16) で 10× 以上の差を要求.
    #[test]
    fn cfm4_strictly_more_accurate_than_m2() {
        let n = 3_usize;
        let dim = 1usize << n;
        let mut rng = Xor64::new(987654);
        let h_x: Vec<f64> = (0..n).map(|_| rng.signed()).collect();
        let h_p_diag: Vec<f64> = (0..dim).map(|_| rng.signed()).collect();
        let psi_raw = random_complex_vec(dim, 0xbabe);
        let norm = nrm2(&psi_raw);
        let psi0: Vec<Complex64> = psi_raw.iter().map(|c| *c / norm).collect();

        let a_base = 0.4_f64;
        let b_base = 0.7_f64;
        let f = |t: f64| t.sin();
        let total_t = 0.5_f64;
        let n_steps = 16_usize;
        let dt = total_t / n_steps as f64;
        let f_integral = 1.0 - total_t.cos();

        let h_real = build_dense_h_real(n, &h_x, &h_p_diag, a_base, b_base);
        let expected = reference_propagate_real_h(&h_real, &psi0, f_integral);

        let m = 24_usize;
        let krylov_tol = 1e-14;
        let c_1 = cfm4_c1();
        let c_2 = cfm4_c2();

        let mut psi_cfm4 = psi0.clone();
        for k in 0..n_steps {
            let t_k = k as f64 * dt;
            let s_1 = t_k + c_1 * dt;
            let s_2 = t_k + c_2 * dt;
            psi_cfm4 = cfm4_step(
                &psi_cfm4,
                &h_x,
                &h_p_diag,
                a_base * f(s_1),
                b_base * f(s_1),
                a_base * f(s_2),
                b_base * f(s_2),
                dt,
                m,
                krylov_tol,
                n,
            )
            .expect("ok");
        }

        let mut psi_m2 = psi0.clone();
        for k in 0..n_steps {
            let t_mid = k as f64 * dt + 0.5 * dt;
            psi_m2 = m2_midpoint_step(
                &psi_m2,
                &h_x,
                &h_p_diag,
                a_base * f(t_mid),
                b_base * f(t_mid),
                dt,
                m,
                krylov_tol,
                n,
            )
            .expect("ok");
        }

        let err_cfm4 = relative_error(&psi_cfm4, &expected);
        let err_m2 = relative_error(&psi_m2, &expected);
        let ratio = err_m2 / err_cfm4;

        assert!(
            err_cfm4 < err_m2,
            "CFM4 err = {} should be < M2 err = {}",
            err_cfm4,
            err_m2,
        );
        assert!(
            ratio > 10.0,
            "expected CFM4 >> M2 (ratio > 10), got ratio = {} (err_cfm4 = {}, err_m2 = {})",
            ratio,
            err_cfm4,
            err_m2,
        );
    }

    /// `cfm4_step_with_m2_estimate` を `dt = 0` で呼ぶと err は同一の入口 ψ を
    /// 共有する CFM4:2 / M2 の差分なのでゼロ (Lanczos 内の浮動小数雑音のみ).
    /// 同時に psi も入口と一致 (位相 `exp(0) = 1`) するはず.
    #[test]
    fn m2_estimate_dt_zero_err_is_zero() {
        let n = 4_usize;
        let dim = 1usize << n;
        let mut rng = Xor64::new(0xabcd);
        let h_x: Vec<f64> = (0..n).map(|_| rng.signed()).collect();
        let h_p_diag: Vec<f64> = (0..dim).map(|_| rng.signed()).collect();
        let psi0 = random_complex_vec(dim, 0xdeed);
        let (a_s1, b_s1) = (rng.signed(), rng.signed());
        let (a_s2, b_s2) = (rng.signed(), rng.signed());
        let (a_mid, b_mid) = (rng.signed(), rng.signed());

        let mut psi = psi0.clone();
        let err = cfm4_step_with_m2_estimate(
            &mut psi, &h_x, &h_p_diag, a_s1, b_s1, a_s2, b_s2, a_mid, b_mid, 0.0, 24, 1e-12, n,
        )
        .expect("ok");

        assert!(err.abs() < 1e-13, "dt=0 err = {}", err);
        let rel = relative_error(&psi, &psi0);
        assert!(rel < 1e-13, "dt=0 psi rel = {}", rel);
    }

    /// per-step embedded error `err = ‖ψ_cfm4 - ψ_m2‖` が `O(dt^3)` (M2 の LTE
    /// オーダ) でスケールすることを確認する.
    ///
    /// 設定: operator 部分 (`h_x`, `h_p_diag`) は固定したまま, schedule 係数を
    /// `(A(s), B(s)) = (a_base · f(t), b_base · f(t))` で時間依存にする
    /// (`f(t) = sin(t)` を採用, `f''(t)` が非零な `t_0 = 1.0` を中心に評価).
    /// この設定で
    ///
    /// - M2: 中点則, per-step LTE `O(dt^3)`
    /// - CFM4:2: ガウス-ルジャンドル 2 点求積, per-step LTE `O(dt^5)`
    ///
    /// 小 dt 領域では CFM4:2 が exact に近く, `‖ψ_cfm4 - ψ_m2‖ ≈
    /// ‖ψ_m2 - ψ_exact‖ ∝ dt^3`. dt 半減で err 比 `~8`, `[4, 16]` 窓を要求.
    /// (operator 自体を完全に time-independent にすると CFM4:2 と M2 が
    /// 厳密一致して err = 0 になるためテストにならない — issue 仕様の
    /// 「time-independent H + smooth ψ」は **operator は time-independent,
    /// schedule が smooth time-dependent** の意で取る.)
    #[test]
    fn m2_estimate_err_scales_as_dt_cubed_for_smooth_schedule() {
        let n = 3_usize;
        let dim = 1usize << n;
        let mut rng = Xor64::new(0xface);
        let h_x: Vec<f64> = (0..n).map(|_| rng.signed()).collect();
        let h_p_diag: Vec<f64> = (0..dim).map(|_| rng.signed()).collect();
        let psi_raw = random_complex_vec(dim, 0xbeef);
        let norm = nrm2(&psi_raw);
        let psi0: Vec<Complex64> = psi_raw.iter().map(|c| *c / norm).collect();

        let a_base = 0.4_f64;
        let b_base = 0.7_f64;
        let f = |t: f64| t.sin();
        let t0 = 1.0_f64;
        let m = 24_usize;
        let krylov_tol = 1e-14_f64;
        let c_1 = cfm4_c1();
        let c_2 = cfm4_c2();

        let step_err = |dt: f64| -> f64 {
            let a_s1 = a_base * f(t0 + c_1 * dt);
            let b_s1 = b_base * f(t0 + c_1 * dt);
            let a_s2 = a_base * f(t0 + c_2 * dt);
            let b_s2 = b_base * f(t0 + c_2 * dt);
            let a_mid = a_base * f(t0 + 0.5 * dt);
            let b_mid = b_base * f(t0 + 0.5 * dt);
            let mut psi = psi0.clone();
            cfm4_step_with_m2_estimate(
                &mut psi, &h_x, &h_p_diag, a_s1, b_s1, a_s2, b_s2, a_mid, b_mid, dt, m, krylov_tol,
                n,
            )
            .expect("ok")
        };

        let dt_coarse = 0.02_f64;
        let err_coarse = step_err(dt_coarse);
        let err_fine = step_err(dt_coarse / 2.0);
        let ratio = err_coarse / err_fine;

        assert!(
            ratio > 4.0 && ratio < 16.0,
            "expected ratio ~ 8 (O(dt^3)), got ratio = {} (err_coarse = {}, err_fine = {})",
            ratio,
            err_coarse,
            err_fine,
        );
        // err_fine が Lanczos 床 (~1e-14·dim) より十分大きいことも確認 (
        // 床に張り付いていると ratio がノイズ支配でテストが空振りする).
        assert!(
            err_fine > 1e-12,
            "err_fine = {} too small (Lanczos floor reached?)",
            err_fine,
        );
    }

    /// `cfm4_step_with_m2_estimate` の psi 出力が同一引数 (a_s1, b_s1, a_s2,
    /// b_s2, dt, m, krylov_tol, n) で呼ぶ `cfm4_step` 単体の出力と
    /// **bit-exact** に一致することを確認する.
    ///
    /// CFM4:2 のステップ (Lanczos 2 回) は (a_mid, b_mid) に依存しない
    /// (M2 経路は err 算出にのみ使い破棄される) ため, 同じ Lanczos
    /// 呼び出しが同じビット列を生む契約.
    #[test]
    fn m2_estimate_psi_update_matches_cfm4_step_bit_exact() {
        let n = 4_usize;
        let dim = 1usize << n;
        let mut rng = Xor64::new(0x5555);
        let h_x: Vec<f64> = (0..n).map(|_| rng.signed()).collect();
        let h_p_diag: Vec<f64> = (0..dim).map(|_| rng.signed()).collect();
        let psi0 = random_complex_vec(dim, 0x7777);
        let (a_s1, b_s1) = (0.4_f64, 1.1_f64);
        let (a_s2, b_s2) = (0.7_f64, 0.3_f64);
        let (a_mid, b_mid) = (0.55_f64, 0.7_f64);
        let dt = 0.25_f64;
        let m = 24_usize;
        let krylov_tol = 1e-12_f64;

        let psi_expected = cfm4_step(
            &psi0, &h_x, &h_p_diag, a_s1, b_s1, a_s2, b_s2, dt, m, krylov_tol, n,
        )
        .expect("ok");

        let mut psi_actual = psi0.clone();
        let _err = cfm4_step_with_m2_estimate(
            &mut psi_actual,
            &h_x,
            &h_p_diag,
            a_s1,
            b_s1,
            a_s2,
            b_s2,
            a_mid,
            b_mid,
            dt,
            m,
            krylov_tol,
            n,
        )
        .expect("ok");

        for k in 0..dim {
            assert_eq!(
                psi_actual[k], psi_expected[k],
                "k={}: actual={:?}, expected={:?}",
                k, psi_actual[k], psi_expected[k]
            );
        }
    }

    /// time-independent H に対しチェイン適用すると
    /// `m2(dt) ∘ m2(dt) ∘ ... ∘ m2(dt) = exp(-i n_steps·dt · H)` に
    /// **数値的に** 一致するはず (中点則は H が定数なら厳密に
    /// `exp(-i dt H)` を返し, Lanczos 誤差のみ残る).
    /// n_steps=100, T=1, n=4 で rel < 1e-10 を要求 (issue 仕様).
    #[test]
    fn time_independent_h_matches_exact_chain() {
        let n = 4_usize;
        let dim = 1usize << n;
        let mut rng = Xor64::new(0x1234);
        let h_x: Vec<f64> = (0..n).map(|_| rng.signed()).collect();
        let h_p_diag: Vec<f64> = (0..dim).map(|_| rng.signed()).collect();
        let psi0 = random_complex_vec(dim, 0xbeef);

        let a_t = rng.signed();
        let b_t = rng.signed();

        let n_steps = 100_usize;
        let total_t = 1.0_f64;
        let dt = total_t / n_steps as f64;

        let mut psi = psi0.clone();
        for _ in 0..n_steps {
            psi = m2_midpoint_step(&psi, &h_x, &h_p_diag, a_t, b_t, dt, 24, 1e-14, n).expect("ok");
        }

        let h_real = build_dense_h_real(n, &h_x, &h_p_diag, a_t, b_t);
        let expected = reference_propagate_real_h(&h_real, &psi0, total_t);

        let rel = relative_error(&psi, &expected);
        assert!(rel < 1e-10, "chain n_steps={} rel = {}", n_steps, rel);
    }
}
