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
//! 回). 詳細は `docs/design.md` §5.3 (Trotter サブセクション) を一次資料とする.
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

#![allow(dead_code)]

use num_complex::Complex64;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::matvec::apply_single_mode_axis_i;

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
    // Complex64::new(cos φ, sin φ) = exp(+i·φ); φ = -b_t·h_p_diag[k]·dt/2 を
    // 採れば exp(-i·b_t·h_p_diag[k]·dt/2) となる.
    for k in 0..dim {
        let phi = -b_t * h_p_diag[k] * half;
        let (s, c) = phi.sin_cos();
        psi[k] *= Complex64::new(c, s);
    }

    // ---- 中間 Π_i R_i(dt): 各サイト i に 2×2 ユニタリを in-place 適用 ----
    // θ_i = +a_t · h_x_i · dt (モジュール冒頭 docstring の符号 convention 参照).
    for (i, &h_x_i) in h_x.iter().enumerate() {
        let theta = a_t * h_x_i * dt;
        let (s, c) = theta.sin_cos();
        let u = [
            Complex64::new(c, 0.0),
            Complex64::new(0.0, s),
            Complex64::new(0.0, s),
            Complex64::new(c, 0.0),
        ];
        apply_single_mode_axis_i(psi, &u, i, n);
    }

    // ---- 後半 phase_p(dt/2): 前半と同じ phase を再度乗算 ----
    for k in 0..dim {
        let phi = -b_t * h_p_diag[k] * half;
        let (s, c) = phi.sin_cos();
        psi[k] *= Complex64::new(c, s);
    }
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
/// 公開 API である (`docs/design.md` §7.3).
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
}
