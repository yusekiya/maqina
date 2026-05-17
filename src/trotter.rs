//! Strang 2 次 Trotter 1 step (Phase 2 C2).
//!
//! 横磁場ドライバ `H_driver = -Σ_i h_x_i X_i` は各 `X_i` が互いに可換なので,
//! `exp(-i·dt·A·H_driver) = Π_i exp(+i·dt·A·h_x_i·X_i)` を 1 軸ずつ閉形式で
//! 適用できる (Lanczos 不要). diag 項は `exp(-i·dt/2·B·diag)` の対角行列。
//! これを Strang 形に並べる:
//!
//! ```text
//! U(dt) ≈ exp(-i·dt/2·H_p) · exp(-i·dt·H_drv) · exp(-i·dt/2·H_p)
//!       = phase_p(dt/2) · (Π_i R_i(dt)) · phase_p(dt/2)
//! ```
//!
//! `(a_t, b_t)` は schedule の **中点** `A(s(t+dt/2)), B(s(t+dt/2))` を採取
//! することで time-dependent でも LTE `O(dt^3)` を保つ (Strang の中点採取則).
//!
//! per-step コスト: `(N + 1) · dim` 要素アクセス (matvec 1 pass 相当が `N+1`
//! 回). 詳細は `docs/design/05-3-propagator.md` §5.3 (Trotter サブセクション) を一次資料とする.
//!
//! ## R_i の符号 convention
//!
//! `H_drv = -Σ h_x_i X_i` なので,
//!
//! ```text
//! exp(-i·dt·a_t·H_drv) = exp(+i·dt·a_t·Σ h_x_i X_i)
//!                      = Π_i exp(+i·θ_i·X_i),   θ_i = a_t · h_x_i · dt
//! ```
//!
//! `exp(+i·θ·X) = cos(θ)·I + i·sin(θ)·X` (`X² = I` より). すなわち `R_i(dt)` に
//! 渡すべき 2×2 ユニタリは `u = [cos θ, i·sin θ, i·sin θ, cos θ]` で
//! **`θ = +a_t · h_x_i · dt` (正符号)**. `H_drv` の負符号を `θ` に巻き取ら
//! ない (`apply_h_kryanneal` で `coeff = -a_t · h_x[i]` としているのと整合).
//!
//! ## 4 次 Suzuki (`trotter_suzuki4_step`)
//!
//! Trotter-Suzuki S_4 公式
//!
//! ```text
//! S_4(dt) = S_2(p·dt) · S_2(p·dt) · S_2((1 - 4p)·dt) · S_2(p·dt) · S_2(p·dt)
//! p = 1 / (4 - 4^{1/3}) ≈ 0.41449
//! ```
//!
//! で Strang S_2 を 5 回適用に分解する 4 次 propagator. per-step は
//! `5·(N + 1)·dim` 要素アクセス, LTE は `O(dt^5)` (CFM4:2 と同じ局所オーダ).
//! Lanczos の m 回 matvec を完全に排した経路としての比較・検証用に Phase 2 末で
//! 追加 (`method="trotter_suzuki4"`).
//!
//! ### サブステップ係数と中点
//!
//! サブステップ幅は `[p, p, 1 - 4p, p, p]·dt` (中央 sub-step は `1 - 4p ≈ -0.658`
//! で **逆向き**; `trotter_step` は `dt < 0` 入力を受け付けるので問題ない).
//! 時間依存 H に対しては各 sub-step の中点でフリーズ採取することで全体の
//! LTE `O(dt^5)` を保つ. 中点 offset (sub-step `k` を `[t + start_k·dt,
//! t + end_k·dt]` としたときの `(start_k + end_k)/2`) は:
//!
//! ```text
//! offsets = [p/2, 3p/2, 1/2, 1 - 3p/2, 1 - p/2]   (t + dt/2 を中心に対称)
//! ```
//!
//! ホスト言語 (Python driver) が各 sub-step の `(a_mid, b_mid)` をこの offset
//! で事前に評価し, 長さ 5 のスライスとして本関数に渡す責務を負う (Rust 側に
//! schedule callable を持ち込まないことで Strang 経路の API と整合させる).

#![allow(dead_code)]

use num_complex::Complex64;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

use crate::matvec::{apply_multi_qubit_gate_fused, apply_single_mode_axis_i, MAX_FUSED_K};

/// `phase_p(dt')`: `psi[k] *= exp(-i · b_t · h_p_diag[k] · dt')` を全 k に適用.
///
/// 大 dim (≥ 2^17 = 128K Complex64 = 2 MB) では rayon par_iter_mut で並列化
/// (`apply_h_kryanneal_rayon` の `MIN_RAYON_DIM` と同じ閾値). small dim では
/// scalar fallback. trotter_step の per-step time のうち, multi-qubit gate
/// fusion で削減できない diag 部分 (N=20 dim=1M で数 ms 級) を rayon barrier
/// 2 個 (前後) に詰める (Phase 6 C3, issue #64 v3 修正).
#[inline]
fn apply_phase_p(psi: &mut [Complex64], h_p_diag: &[f64], b_t: f64, dt_half: f64) {
    debug_assert_eq!(psi.len(), h_p_diag.len());

    #[cfg(feature = "rayon")]
    {
        // `MIN_RAYON_DIM = 1 << 17` (matvec.rs) と同じ閾値で dispatch.
        const PHASE_RAYON_MIN_DIM: usize = 1 << 17;
        if psi.len() >= PHASE_RAYON_MIN_DIM {
            psi.par_iter_mut()
                .zip(h_p_diag.par_iter())
                .for_each(|(psi_k, &h_p_k)| {
                    let phi = -b_t * h_p_k * dt_half;
                    let (s, c) = phi.sin_cos();
                    *psi_k *= Complex64::new(c, s);
                });
            return;
        }
    }
    // Scalar fallback.
    for (psi_k, &h_p_k) in psi.iter_mut().zip(h_p_diag.iter()) {
        let phi = -b_t * h_p_k * dt_half;
        let (s, c) = phi.sin_cos();
        *psi_k *= Complex64::new(c, s);
    }
}

/// Multi-qubit gate fusion (Phase 6 C3, issue #64) で `trotter_step` 内の
/// `Π_i R_i(dt)` を何 qubit ずつまとめて適用するかの単位.
///
/// `FUSE_K = 4`: qsim の経験 (`max_fused_size = 4-5` 推奨, `lib/fuser_mqubit.h`)
/// と本 PR 初版 (dense matmul) 失敗からの学び (per-axis × k の `2k·dim` ops に
/// 対して dense matmul は `2^k·dim` ops で k=4 で 2× 重く Linux 本番で
/// 0.81× regression した) を踏まえた default 値. **現実装は per-axis 逐次**
/// なので compute は per-axis × k と同じで, 主効果は barrier 多重化解消と
/// chunk-resident cache 効果. `bench_block_fusion.py` で per-step time を
/// sweep して必要なら見直す.
///
/// `n < FUSE_K` の極小 N (= テスト用) では fused 経路に乗らず端数経路で
/// per-axis apply される (`apply_single_mode_axis_i` フォールバック).
///
/// 制約: `FUSE_K <= MAX_FUSED_K`.
const FUSE_K: usize = 4;
const _: () = assert!(FUSE_K <= MAX_FUSED_K);

/// 連続 k axis (i_start, ..., i_start+k-1) の R_i(dt) を構築し,
/// `[[Complex64; 4]; k]` (row-major 2×2 列) として `out_slice[..k]` に書く.
///
/// R_i(dt) = cos(θ_i)·I + i·sin(θ_i)·X_i, θ_i = a_t · h_x[i] · dt の Trotter
/// convention. row-major 2×2 表現は `[c, i·s, i·s, c]`. `apply_multi_qubit_gate_fused`
/// が per-axis 逐次に消費するので tensor product は構築しない.
fn build_axis_unitaries(h_x_slice: &[f64], a_t: f64, dt: f64, out_slice: &mut [[Complex64; 4]]) {
    debug_assert_eq!(
        out_slice.len(),
        h_x_slice.len(),
        "out_slice and h_x_slice must have equal length",
    );
    for (out_u, &h_x_j) in out_slice.iter_mut().zip(h_x_slice.iter()) {
        let theta = a_t * h_x_j * dt;
        let (s, c) = theta.sin_cos();
        *out_u = [
            Complex64::new(c, 0.0),
            Complex64::new(0.0, s),
            Complex64::new(0.0, s),
            Complex64::new(c, 0.0),
        ];
    }
}

/// Strang 2 次 Trotter 1 step. `psi` を in-place で `U(dt) · psi` に上書きする.
///
/// # 引数
/// - `psi` (length `2^n`): 入力 / 出力状態ベクトル. in-place で更新される.
/// - `h_x` (length `n`): サイトごとの横磁場振幅.
/// - `h_p_diag` (length `2^n`): Z 基底での `H_problem` 対角ベクトル.
/// - `a_t`, `b_t`: schedule の **中点** で評価した係数 `A(s(t+dt/2))`,
///   `B(s(t+dt/2))`. Strang 2 次の対称性を保つために中点採取は呼出側責任.
/// - `dt`: 時間刻み. 符号は任意 (`-dt` を渡すと逆向きの propagator).
/// - `n`: サイト数. `dim = 2^n`.
///
/// # 数値処理
/// 1. `phase_p(dt/2)`: 各 `k` に `exp(-i · b_t · h_p_diag[k] · dt/2)` を乗算.
/// 2. `Π_{i=0..n} R_i(dt)`: `apply_single_mode_axis_i` を `θ_i = a_t·h_x_i·dt`
///    の `u = [cos θ, i·sin θ, i·sin θ, cos θ]` で 1 軸ずつ in-place 適用.
/// 3. `phase_p(dt/2)`: 1 と同じ phase をもう一度乗算.
///
/// 全ての因子が unitary なので, `‖psi_new‖ = ‖psi‖` が machine precision で
/// 保たれる (テストで `rel < 1e-13` を要求).
///
/// # Panics
/// - `psi.len() != 1 << n`
/// - `h_x.len() != n`
/// - `h_p_diag.len() != 1 << n`
#[allow(clippy::too_many_arguments)]
pub(crate) fn trotter_step(
    psi: &mut [Complex64],
    h_x: &[f64],
    h_p_diag: &[f64],
    a_t: f64,
    b_t: f64,
    dt: f64,
    n: usize,
) {
    let dim = 1usize << n;
    assert_eq!(psi.len(), dim, "psi must have length 2^n");
    assert_eq!(h_x.len(), n, "h_x must have length n");
    assert_eq!(h_p_diag.len(), dim, "h_p_diag must have length 2^n");

    let half = 0.5 * dt;

    // ---- 前半 phase_p(dt/2): exp(-i · b_t · h_p_diag[k] · dt/2) を要素ごとに乗算 ----
    // `apply_phase_p` は大 dim では rayon 並列化される (Phase 6 C3 v3).
    apply_phase_p(psi, h_p_diag, b_t, half);

    // ---- 中間 Π_i R_i(dt): 連続 FUSE_K qubit ずつ tensor product にまとめて
    //      1 sweep で適用する (Phase 6 C3, multi-qubit gate fusion, issue #64). ----
    //
    // `H_drv = -Σ h_x_i X_i` は per-site で commuting なので
    //   `exp(+i·a·dt·Σ_{j∈G} h_x_j X_j) = ⊗_{j∈G} exp(+i·θ_j X_j)`
    // が exact. 連続 k qubit に限定すれば psi の sub-block は stride 2^i_start の
    // 連続 2^k 要素となり, `apply_multi_qubit_gate_fused` で 1 rayon barrier
    // (= 1 par_chunks_mut 呼出) に詰められる. 端数 (n が FUSE_K の倍数でない
    // 場合) は従来通り `apply_single_mode_axis_i` で 1 軸ずつ適用する.
    //
    // θ_i = +a_t · h_x_i · dt (モジュール冒頭 docstring の符号 convention 参照).
    let mut i_cursor = 0usize;
    // 固定長 stack 配列で k 個の 2×2 unitary 列を持つ. apply_multi_qubit_gate_fused
    // に &u_list[..FUSE_K] として渡す (per-axis 逐次経路).
    let mut u_list = [[Complex64::new(0.0, 0.0); 4]; FUSE_K];
    while i_cursor + FUSE_K <= n {
        build_axis_unitaries(&h_x[i_cursor..i_cursor + FUSE_K], a_t, dt, &mut u_list);
        apply_multi_qubit_gate_fused(psi, &u_list, i_cursor, n);
        i_cursor += FUSE_K;
    }
    while i_cursor < n {
        let theta = a_t * h_x[i_cursor] * dt;
        let (s, c) = theta.sin_cos();
        let u = [
            Complex64::new(c, 0.0),
            Complex64::new(0.0, s),
            Complex64::new(0.0, s),
            Complex64::new(c, 0.0),
        ];
        apply_single_mode_axis_i(psi, &u, i_cursor, n);
        i_cursor += 1;
    }

    // ---- 後半 phase_p(dt/2): 前半と同じ phase を再度乗算 ----
    apply_phase_p(psi, h_p_diag, b_t, half);
}

/// `trotter_step` の Python wrap. 結果を新規 array で返す (in-place ではなく
/// allocate-and-return パターン; `apply_h_kryanneal_py` / `m2_midpoint_step_py`
/// / `apply_single_mode_axis_i_py` と統一).
///
/// Python 側 (`_rust.trotter_step_py`) からは
///
/// ```python
/// psi_new = _rust.trotter_step_py(psi, h_x, h_p_diag, a_t, b_t, dt, n)
/// ```
///
/// として呼ぶ. Trotter 経路の固定 dt ドライバが内部的に呼び出す Rust 経路は
/// `trotter_step` を直接使うため, 本 wrap は **参照実装比較とテスト用** の
/// 公開 API である (`docs/design/07-rust-extension.md` §7.3).
#[pyfunction]
#[pyo3(signature = (psi, h_x, h_p_diag, a_t, b_t, dt, n))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn trotter_step_py<'py>(
    py: Python<'py>,
    psi: PyReadonlyArray1<'py, Complex64>,
    h_x: PyReadonlyArray1<'py, f64>,
    h_p_diag: PyReadonlyArray1<'py, f64>,
    a_t: f64,
    b_t: f64,
    dt: f64,
    n: usize,
) -> PyResult<Bound<'py, PyArray1<Complex64>>> {
    let psi_slice = psi.as_slice()?;
    let h_x_slice = h_x.as_slice()?;
    let h_p_diag_slice = h_p_diag.as_slice()?;

    let dim = 1usize << n;
    if psi_slice.len() != dim {
        return Err(PyValueError::new_err(format!(
            "psi length {} does not match 2^n = 2^{} = {}",
            psi_slice.len(),
            n,
            dim,
        )));
    }
    if h_x_slice.len() != n {
        return Err(PyValueError::new_err(format!(
            "h_x length {} does not match n = {}",
            h_x_slice.len(),
            n,
        )));
    }
    if h_p_diag_slice.len() != dim {
        return Err(PyValueError::new_err(format!(
            "h_p_diag length {} does not match 2^n = {}",
            h_p_diag_slice.len(),
            dim,
        )));
    }

    let mut out: Vec<Complex64> = psi_slice.to_vec();
    trotter_step(&mut out, h_x_slice, h_p_diag_slice, a_t, b_t, dt, n);
    Ok(out.into_pyarray(py))
}

/// Trotter-Suzuki S_4 のサブステップ係数係数 `p = 1 / (4 - 4^{1/3})`.
///
/// `f64::cbrt` は const fn ではないため値は `LazyLock` 的に毎呼び出しで
/// 再評価される (実装単純性優先; 1 step あたり数ナノ秒のオーバヘッドのみ).
#[inline]
fn suzuki4_p() -> f64 {
    1.0 / (4.0 - 4.0_f64.cbrt())
}

/// 4 次 Suzuki Trotter 1 step. `psi` を in-place で `S_4(dt) · psi` に上書きする.
///
/// モジュール冒頭 docstring の「4 次 Suzuki」節を参照. 内部実装は単に
/// `trotter_step` を 5 回呼ぶだけ (`coeffs = [p, p, 1-4p, p, p]`).
///
/// # 引数
/// - `psi` (length `2^n`): in-place 更新される状態ベクトル.
/// - `h_x` (length `n`): サイトごとの横磁場振幅.
/// - `h_p_diag` (length `2^n`): Z 基底での `H_problem` 対角.
/// - `a_t_list` (length 5): 各 sub-step の `A` 係数. 中点 offset
///   `[p/2, 3p/2, 1/2, 1 - 3p/2, 1 - p/2]` で評価されている前提.
/// - `b_t_list` (length 5): 各 sub-step の `B` 係数 (同上).
/// - `dt`: 外側 1 step の時間刻み. 符号は任意 (`-dt` で逆向き propagator).
/// - `n`: サイト数. `dim = 2^n`.
///
/// # Panics
/// - `psi.len() != 1 << n`
/// - `h_x.len() != n`
/// - `h_p_diag.len() != 1 << n`
/// - `a_t_list.len() != 5` または `b_t_list.len() != 5`
#[allow(clippy::too_many_arguments)]
pub(crate) fn trotter_suzuki4_step(
    psi: &mut [Complex64],
    h_x: &[f64],
    h_p_diag: &[f64],
    a_t_list: &[f64],
    b_t_list: &[f64],
    dt: f64,
    n: usize,
) {
    assert_eq!(
        a_t_list.len(),
        5,
        "a_t_list must have length 5 (Suzuki S_4 sub-steps)"
    );
    assert_eq!(
        b_t_list.len(),
        5,
        "b_t_list must have length 5 (Suzuki S_4 sub-steps)"
    );
    let p = suzuki4_p();
    let coeffs = [p, p, 1.0 - 4.0 * p, p, p];
    for k in 0..5 {
        trotter_step(
            psi,
            h_x,
            h_p_diag,
            a_t_list[k],
            b_t_list[k],
            coeffs[k] * dt,
            n,
        );
    }
}

/// `trotter_suzuki4_step` の Python wrap. 結果を新規 array で返す
/// (`trotter_step_py` と統一の allocate-and-return パターン).
///
/// Python 側 (`_rust.trotter_suzuki4_step_py`) からは
///
/// ```python
/// psi_new = _rust.trotter_suzuki4_step_py(
///     psi, h_x, h_p_diag, a_t_list, b_t_list, dt, n
/// )
/// ```
///
/// として呼ぶ. `a_t_list` / `b_t_list` は length 5 の float64 array.
#[pyfunction]
#[pyo3(signature = (psi, h_x, h_p_diag, a_t_list, b_t_list, dt, n))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn trotter_suzuki4_step_py<'py>(
    py: Python<'py>,
    psi: PyReadonlyArray1<'py, Complex64>,
    h_x: PyReadonlyArray1<'py, f64>,
    h_p_diag: PyReadonlyArray1<'py, f64>,
    a_t_list: PyReadonlyArray1<'py, f64>,
    b_t_list: PyReadonlyArray1<'py, f64>,
    dt: f64,
    n: usize,
) -> PyResult<Bound<'py, PyArray1<Complex64>>> {
    let psi_slice = psi.as_slice()?;
    let h_x_slice = h_x.as_slice()?;
    let h_p_diag_slice = h_p_diag.as_slice()?;
    let a_t_slice = a_t_list.as_slice()?;
    let b_t_slice = b_t_list.as_slice()?;

    let dim = 1usize << n;
    if psi_slice.len() != dim {
        return Err(PyValueError::new_err(format!(
            "psi length {} does not match 2^n = 2^{} = {}",
            psi_slice.len(),
            n,
            dim,
        )));
    }
    if h_x_slice.len() != n {
        return Err(PyValueError::new_err(format!(
            "h_x length {} does not match n = {}",
            h_x_slice.len(),
            n,
        )));
    }
    if h_p_diag_slice.len() != dim {
        return Err(PyValueError::new_err(format!(
            "h_p_diag length {} does not match 2^n = {}",
            h_p_diag_slice.len(),
            dim,
        )));
    }
    if a_t_slice.len() != 5 {
        return Err(PyValueError::new_err(format!(
            "a_t_list length {} does not match Suzuki S_4 sub-step count 5",
            a_t_slice.len(),
        )));
    }
    if b_t_slice.len() != 5 {
        return Err(PyValueError::new_err(format!(
            "b_t_list length {} does not match Suzuki S_4 sub-step count 5",
            b_t_slice.len(),
        )));
    }

    let mut out: Vec<Complex64> = psi_slice.to_vec();
    trotter_suzuki4_step(
        &mut out,
        h_x_slice,
        h_p_diag_slice,
        a_t_slice,
        b_t_slice,
        dt,
        n,
    );
    Ok(out.into_pyarray(py))
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, SymmetricEigen};

    /// 軽量決定論的 PRNG (xorshift64). matvec / cfm4 のテストと同実装を再掲
    /// (Rust 単体テストは同じファイル内に置く方針, CLAUDE.md 参照).
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
    /// 実対称. `apply_h_kryanneal` の符号 convention (`coeff = -a_t · h_x_i`)
    /// と一致させてある.
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

    /// `dt = 0` で恒等変換になる: `exp(-i · 0 · H) · ψ = ψ`.
    /// phase_p / R_i すべてが `exp(0) = 1` を返すので psi は要素ごとに不変.
    #[test]
    fn dt_zero_is_identity() {
        let n = 4_usize;
        let dim = 1usize << n;
        let mut rng = Xor64::new(0xc20a);
        let h_x: Vec<f64> = (0..n).map(|_| rng.signed()).collect();
        let h_p_diag: Vec<f64> = (0..dim).map(|_| rng.signed()).collect();
        let psi0 = random_complex_vec(dim, 0xc20b);
        let a_t = rng.signed();
        let b_t = rng.signed();

        let mut psi = psi0.clone();
        trotter_step(&mut psi, &h_x, &h_p_diag, a_t, b_t, 0.0, n);

        let rel = relative_error(&psi, &psi0);
        assert!(rel < 1e-13, "dt=0 rel = {}", rel);
    }

    /// 全因子が unitary なので `‖ψ_new‖ = ‖ψ‖` が `rel < 1e-13` で保たれる.
    /// 複数 `(n, dt)` の組で検査する.
    #[test]
    fn preserves_norm() {
        for n in 2..=5_usize {
            let dim = 1usize << n;
            for seed in [0xc21u64, 0xc22, 0xc23] {
                let mut rng = Xor64::new(seed.wrapping_add(n as u64));
                let h_x: Vec<f64> = (0..n).map(|_| rng.signed()).collect();
                let h_p_diag: Vec<f64> = (0..dim).map(|_| rng.signed()).collect();
                let psi0: Vec<Complex64> = (0..dim).map(|_| rng.complex_signed()).collect();
                let norm0_sq: f64 = psi0.iter().map(|z| z.norm_sqr()).sum();
                let a_t = rng.signed();
                let b_t = rng.signed();
                for &dt in &[0.05_f64, 0.3, 1.2] {
                    let mut psi = psi0.clone();
                    trotter_step(&mut psi, &h_x, &h_p_diag, a_t, b_t, dt, n);
                    let norm_sq: f64 = psi.iter().map(|z| z.norm_sqr()).sum();
                    let rel = (norm_sq - norm0_sq).abs() / norm0_sq.max(1.0);
                    assert!(
                        rel < 1e-13,
                        "n={}, seed={:x}, dt={}: norm not preserved (rel={})",
                        n,
                        seed,
                        dt,
                        rel,
                    );
                }
            }
        }
    }

    /// `trotter_step(dt) ∘ trotter_step(-dt) ≈ I`.
    /// 各因子 (phase_p, R_i) は `dt → -dt` で exact inverse になる
    /// (sin/cos の引数符号反転で `exp(+i·φ)` と `exp(-i·φ)` がキャンセル).
    /// 数値誤差は accumulated FP rounding のみで, `rel < 1e-12` を要求.
    #[test]
    fn strang_inverse_dt_negate() {
        let n = 4_usize;
        let dim = 1usize << n;
        let mut rng = Xor64::new(0xc24);
        let h_x: Vec<f64> = (0..n).map(|_| rng.signed()).collect();
        let h_p_diag: Vec<f64> = (0..dim).map(|_| rng.signed()).collect();
        let psi0 = random_complex_vec(dim, 0xc25);
        let a_t = rng.signed();
        let b_t = rng.signed();

        for &dt in &[0.05_f64, 0.3, 1.5] {
            let mut psi = psi0.clone();
            trotter_step(&mut psi, &h_x, &h_p_diag, a_t, b_t, dt, n);
            trotter_step(&mut psi, &h_x, &h_p_diag, a_t, b_t, -dt, n);
            let rel = relative_error(&psi, &psi0);
            assert!(rel < 1e-12, "dt={}: inverse rel = {}", dt, rel);
        }
    }

    /// time-independent H に対し 1 step の local truncation error は
    /// `O(dt^3)` (Strang 2 次). `dt` を半減するごとに 1-step err は
    /// 約 `1/8` に減衰する. 複数 `dt` で測って比率 `errs[i-1] / errs[i]` を
    /// `[5, 11]` の窓で許容する (Strang の係数次第で `8` から少しずれる).
    #[test]
    fn time_independent_h_lte_order_3() {
        let n = 3_usize;
        let dim = 1usize << n;
        let mut rng = Xor64::new(0xc26);
        let h_x: Vec<f64> = (0..n).map(|_| rng.signed()).collect();
        let h_p_diag: Vec<f64> = (0..dim).map(|_| rng.signed()).collect();
        let psi0 = random_complex_vec(dim, 0xc27);
        // a, b の大きさを揃えて項のスケールを安定化させる.
        let a_t = 0.7_f64;
        let b_t = 0.9_f64;

        let h_real = build_dense_h_real(n, &h_x, &h_p_diag, a_t, b_t);

        // dt を 1/2 ずつ段階的に細かくする.
        let dts = [0.2_f64, 0.1, 0.05, 0.025];
        let mut errs = Vec::with_capacity(dts.len());
        for &dt in &dts {
            let expected = reference_propagate_real_h(&h_real, &psi0, dt);
            let mut psi = psi0.clone();
            trotter_step(&mut psi, &h_x, &h_p_diag, a_t, b_t, dt, n);
            errs.push(relative_error(&psi, &expected));
        }

        // 細かい dt ほど err が単調減少 (符号や式の bug を粗く検出).
        for i in 1..errs.len() {
            assert!(
                errs[i] < errs[i - 1],
                "errs not monotonically decreasing: {:?}",
                errs,
            );
        }

        // dt 半減で err 比 ≈ 8 (O(dt^3) LTE).
        // FP rounding が err に紛れ込む最小 dt 域は外し, 大きい側 3 点で確認.
        for i in 1..(dts.len() - 1) {
            let ratio = errs[i - 1] / errs[i];
            assert!(
                (5.0..=11.0).contains(&ratio),
                "dt {} -> {}: ratio = {} (expected ~8 for LTE O(dt^3)), errs = {:?}",
                dts[i - 1],
                dts[i],
                ratio,
                errs,
            );
        }
    }

    /// Suzuki S_4 のサブステップ係数係数 `p` (`1 / (4 - 4^{1/3})`) と
    /// 配列 `[p, p, 1-4p, p, p]` の数値特性検査. 5 サブステップ係数の和は
    /// `4p + (1-4p) = 1` で外側 1 step を構成する.
    #[test]
    fn suzuki4_coeffs_sum_to_one() {
        let p = suzuki4_p();
        let sum = 4.0 * p + (1.0 - 4.0 * p);
        assert!(
            (sum - 1.0).abs() < 1e-15,
            "coeffs sum = {}, expected 1.0",
            sum
        );
        // p の解析値 (`4 - 4^{1/3} ≈ 2.4126`) との一致.
        let expected_denom: f64 = 4.0 - 4.0_f64.cbrt();
        assert!((p - 1.0 / expected_denom).abs() < 1e-15);
    }

    /// `h_x = 0` で trotter_step は phase_p(dt/2) · I · phase_p(dt/2)
    /// = exp(-i·b·diag·dt) と厳密一致する (Strang 内部の R_i がすべて I).
    /// dense reference との一致を `rel < 1e-13` で検査.
    #[test]
    fn zero_h_x_reduces_to_diag_propagator() {
        let n = 4_usize;
        let dim = 1usize << n;
        let mut rng = Xor64::new(0xc28);
        let h_x = vec![0.0_f64; n];
        let h_p_diag: Vec<f64> = (0..dim).map(|_| rng.signed()).collect();
        let psi0 = random_complex_vec(dim, 0xc29);
        let a_t = rng.signed();
        let b_t = rng.signed();
        let dt = 0.4_f64;

        let mut psi = psi0.clone();
        trotter_step(&mut psi, &h_x, &h_p_diag, a_t, b_t, dt, n);

        // 期待値: psi_new[k] = exp(-i · b · h_p_diag[k] · dt) · psi0[k].
        let expected: Vec<Complex64> = (0..dim)
            .map(|k| {
                let phi = -b_t * h_p_diag[k] * dt;
                let (s, c) = phi.sin_cos();
                Complex64::new(c, s) * psi0[k]
            })
            .collect();

        let rel = relative_error(&psi, &expected);
        assert!(rel < 1e-13, "zero h_x rel = {}", rel);
    }

    // ----------------------------------------------------------------------
    // Suzuki S_4 ケース
    // ----------------------------------------------------------------------

    /// time-independent H 用に同じ `(a, b)` を 5 つ複製した sub-step リストを
    /// 返すヘルパ. Suzuki S_4 のテストでは time-dep 経路はホスト言語が
    /// 担うので, Rust 単体テストでは constant schedule で LTE オーダだけを
    /// 検証する.
    fn constant_ab_lists(a_t: f64, b_t: f64) -> ([f64; 5], [f64; 5]) {
        ([a_t; 5], [b_t; 5])
    }

    /// `dt = 0` で恒等変換になる: 5 sub-step すべてが個別に identity なので
    /// 合成も identity. `rel < 1e-13`.
    #[test]
    fn suzuki4_dt_zero_is_identity() {
        let n = 4_usize;
        let dim = 1usize << n;
        let mut rng = Xor64::new(0xd10a);
        let h_x: Vec<f64> = (0..n).map(|_| rng.signed()).collect();
        let h_p_diag: Vec<f64> = (0..dim).map(|_| rng.signed()).collect();
        let psi0 = random_complex_vec(dim, 0xd10b);
        let (a_list, b_list) = constant_ab_lists(rng.signed(), rng.signed());

        let mut psi = psi0.clone();
        trotter_suzuki4_step(&mut psi, &h_x, &h_p_diag, &a_list, &b_list, 0.0, n);

        let rel = relative_error(&psi, &psi0);
        assert!(rel < 1e-13, "dt=0 suzuki4 rel = {}", rel);
    }

    /// 5 sub-step の合成も unitary (各 sub-step が unitary なので閉じる).
    /// `‖ψ_new‖ = ‖ψ‖` が `rel < 1e-13`.
    #[test]
    fn suzuki4_preserves_norm() {
        for n in 2..=5_usize {
            let dim = 1usize << n;
            for seed in [0xd11u64, 0xd12, 0xd13] {
                let mut rng = Xor64::new(seed.wrapping_add(n as u64));
                let h_x: Vec<f64> = (0..n).map(|_| rng.signed()).collect();
                let h_p_diag: Vec<f64> = (0..dim).map(|_| rng.signed()).collect();
                let psi0: Vec<Complex64> = (0..dim).map(|_| rng.complex_signed()).collect();
                let norm0_sq: f64 = psi0.iter().map(|z| z.norm_sqr()).sum();
                let (a_list, b_list) = constant_ab_lists(rng.signed(), rng.signed());
                for &dt in &[0.05_f64, 0.3, 1.2] {
                    let mut psi = psi0.clone();
                    trotter_suzuki4_step(&mut psi, &h_x, &h_p_diag, &a_list, &b_list, dt, n);
                    let norm_sq: f64 = psi.iter().map(|z| z.norm_sqr()).sum();
                    let rel = (norm_sq - norm0_sq).abs() / norm0_sq.max(1.0);
                    assert!(
                        rel < 1e-13,
                        "suzuki4: n={}, seed={:x}, dt={}: norm not preserved (rel={})",
                        n,
                        seed,
                        dt,
                        rel,
                    );
                }
            }
        }
    }

    /// `S_4(dt) ∘ S_4(-dt) ≈ I`. 各 sub-step `trotter_step` は `dt → -dt` で
    /// exact inverse なので, 5 sub-step を逆順に並べた `S_4(-dt)` は `S_4(dt)`
    /// の inverse になる. 数値誤差は accumulated FP rounding のみ
    /// (`rel < 1e-12`).
    ///
    /// 実装注: `S_4(-dt)` を呼び出すとき, sub-step 係数 `[p, p, 1-4p, p, p]·(-dt)`
    /// が適用される. Suzuki S_4 自体が time-symmetric なので, sub-step リスト
    /// (`a_list`, `b_list`) は同じものを使って `dt` の符号だけ反転すれば
    /// inverse として機能する.
    #[test]
    fn suzuki4_inverse_dt_negate() {
        let n = 4_usize;
        let dim = 1usize << n;
        let mut rng = Xor64::new(0xd14);
        let h_x: Vec<f64> = (0..n).map(|_| rng.signed()).collect();
        let h_p_diag: Vec<f64> = (0..dim).map(|_| rng.signed()).collect();
        let psi0 = random_complex_vec(dim, 0xd15);
        let (a_list, b_list) = constant_ab_lists(rng.signed(), rng.signed());

        for &dt in &[0.05_f64, 0.3, 1.5] {
            let mut psi = psi0.clone();
            trotter_suzuki4_step(&mut psi, &h_x, &h_p_diag, &a_list, &b_list, dt, n);
            trotter_suzuki4_step(&mut psi, &h_x, &h_p_diag, &a_list, &b_list, -dt, n);
            let rel = relative_error(&psi, &psi0);
            assert!(rel < 1e-12, "suzuki4: dt={}: inverse rel = {}", dt, rel);
        }
    }

    /// `h_x = 0` で S_4 は phase_p のみの合成になる. 5 sub-step の前後の
    /// phase が連結して合計 `dt = (p + p + (1-4p) + p + p)·dt = dt` の
    /// phase_p になり, `exp(-i·b·diag·dt)` と厳密一致する (`rel < 1e-13`).
    #[test]
    fn suzuki4_zero_h_x_reduces_to_diag_propagator() {
        let n = 4_usize;
        let dim = 1usize << n;
        let mut rng = Xor64::new(0xd16);
        let h_x = vec![0.0_f64; n];
        let h_p_diag: Vec<f64> = (0..dim).map(|_| rng.signed()).collect();
        let psi0 = random_complex_vec(dim, 0xd17);
        let (a_list, b_list) = constant_ab_lists(rng.signed(), rng.signed());
        let dt = 0.4_f64;

        let mut psi = psi0.clone();
        trotter_suzuki4_step(&mut psi, &h_x, &h_p_diag, &a_list, &b_list, dt, n);

        // 期待値: psi_new[k] = exp(-i · b · h_p_diag[k] · dt) · psi0[k].
        // 各 sub-step の (a_t, b_t) は constant なので b は b_list[0] と等しい.
        let b_t = b_list[0];
        let expected: Vec<Complex64> = (0..dim)
            .map(|k| {
                let phi = -b_t * h_p_diag[k] * dt;
                let (s, c) = phi.sin_cos();
                Complex64::new(c, s) * psi0[k]
            })
            .collect();

        let rel = relative_error(&psi, &expected);
        assert!(rel < 1e-13, "suzuki4: zero h_x rel = {}", rel);
    }

    /// time-independent H に対し 1 step の local truncation error は
    /// `O(dt^5)` (Suzuki S_4). `dt` を半減するごとに 1-step err は
    /// 約 `1/32` に減衰する. 複数 `dt` で測って比率 `errs[i-1] / errs[i]` を
    /// `[16, 64]` の窓で許容する (Suzuki 係数次第で 32 から少しずれる).
    ///
    /// FP rounding が err に紛れ込む最小 dt 域は外し, 大きい側だけ確認.
    /// dt = 0.4 で `dt^5 ~ 1e-2`, dt = 0.05 で `dt^5 ~ 3e-7` まで降りる.
    #[test]
    fn suzuki4_time_independent_h_lte_order_5() {
        let n = 3_usize;
        let dim = 1usize << n;
        let mut rng = Xor64::new(0xd18);
        let h_x: Vec<f64> = (0..n).map(|_| rng.signed()).collect();
        let h_p_diag: Vec<f64> = (0..dim).map(|_| rng.signed()).collect();
        let psi0 = random_complex_vec(dim, 0xd19);
        let a_t = 0.7_f64;
        let b_t = 0.9_f64;
        let (a_list, b_list) = constant_ab_lists(a_t, b_t);

        let h_real = build_dense_h_real(n, &h_x, &h_p_diag, a_t, b_t);

        // O(dt^5) は小さい dt で FP rounding に埋もれやすい. dt = 0.4 / 0.2 / 0.1
        // で測ると err は ~1e-2, ~3e-4, ~1e-5 程度 (dt^5 ~ 1e-2, 3e-4, 1e-5).
        // FP rounding (1e-15·~few) より十分大きいのでオーダ判定が成立する.
        let dts = [0.4_f64, 0.2, 0.1];
        let mut errs = Vec::with_capacity(dts.len());
        for &dt in &dts {
            let expected = reference_propagate_real_h(&h_real, &psi0, dt);
            let mut psi = psi0.clone();
            trotter_suzuki4_step(&mut psi, &h_x, &h_p_diag, &a_list, &b_list, dt, n);
            errs.push(relative_error(&psi, &expected));
        }

        // 単調減少 (符号 / 式の bug の粗検出).
        for i in 1..errs.len() {
            assert!(
                errs[i] < errs[i - 1],
                "suzuki4 errs not monotonically decreasing: {:?}",
                errs,
            );
        }

        // dt 半減で err 比 ≈ 32. Suzuki 高次係数の関係で 32 ぴったりからは
        // ずれるため [16, 64] の窓で許容する.
        for i in 1..dts.len() {
            let ratio = errs[i - 1] / errs[i];
            assert!(
                (16.0..=64.0).contains(&ratio),
                "suzuki4 dt {} -> {}: ratio = {} (expected ~32 for LTE O(dt^5)), errs = {:?}",
                dts[i - 1],
                dts[i],
                ratio,
                errs,
            );
        }
    }

    /// 同じ `n_steps` (= 5 で Strang を回すのと S_4 を 1 回回す相当の cost) で
    /// time-independent H に対する 1-step error を比較し, S_4 が Strang より
    /// 桁違いに精度が高いことを確認する. dt = 0.2 程度を選ぶと Strang の
    /// err ~ dt^3 ~ 8e-3, S_4 の err ~ dt^5 ~ 3e-4 のオーダになり, ratio が
    /// 約 25 倍以上開く.
    #[test]
    fn suzuki4_more_accurate_than_strang() {
        let n = 3_usize;
        let dim = 1usize << n;
        let mut rng = Xor64::new(0xd1a);
        let h_x: Vec<f64> = (0..n).map(|_| rng.signed()).collect();
        let h_p_diag: Vec<f64> = (0..dim).map(|_| rng.signed()).collect();
        let psi0 = random_complex_vec(dim, 0xd1b);
        let a_t = 0.7_f64;
        let b_t = 0.9_f64;
        let (a_list, b_list) = constant_ab_lists(a_t, b_t);
        let h_real = build_dense_h_real(n, &h_x, &h_p_diag, a_t, b_t);

        let dt = 0.2_f64;
        let expected = reference_propagate_real_h(&h_real, &psi0, dt);

        let mut psi_strang = psi0.clone();
        trotter_step(&mut psi_strang, &h_x, &h_p_diag, a_t, b_t, dt, n);
        let err_strang = relative_error(&psi_strang, &expected);

        let mut psi_s4 = psi0.clone();
        trotter_suzuki4_step(&mut psi_s4, &h_x, &h_p_diag, &a_list, &b_list, dt, n);
        let err_s4 = relative_error(&psi_s4, &expected);

        // S_4 のほうが Strang より厳密に精度が高い (LTE オーダ違いなので
        // dt = 0.2 程度で必ず差が出る). ratio 下限は保守的に 5× で設定し,
        // Suzuki 係数の数値オーダ依存性に余裕を持たせる.
        let ratio = err_strang / err_s4;
        assert!(
            ratio > 5.0,
            "expected S_4 to be much more accurate than Strang, but err_strang / err_s4 = {} (err_strang = {}, err_s4 = {})",
            ratio,
            err_strang,
            err_s4,
        );
    }
}
