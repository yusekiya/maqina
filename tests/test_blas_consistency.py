"""BLAS feature on/off の数値一致 artifact dump (issue #65 C4).

``cargo test --no-default-features`` で scalar fallback 経路と
``cargo test`` (default features = BLAS on) 経路の Lanczos 内部結果が
`rel < 1e-13` で一致することを Rust 側単体テストで担保しているが,
本テストは **Python 公開 API レベルでも end-to-end で同じ rel < 1e-13
が出ること** を確認するための artifact dump。

運用フロー:

1. BLAS on build (`uv run maturin develop --uv`) で ``pytest tests/test_blas_consistency.py``
   を回し, ``tests/artifacts/blas_on.npz`` を生成
2. BLAS off build (`uv run maturin develop --uv --no-default-features --features extension-module,rayon,simd`)
   で同じ test を回し, ``tests/artifacts/blas_off.npz`` を生成
3. ``uv run python tools/diff_blas_artifacts.py tests/artifacts/blas_on.npz tests/artifacts/blas_off.npz``
   で全 array を ``rel < 1e-13`` で diff

artifact 出力先は ``MAQINA_ARTIFACT_DIR`` env var で上書き可能
(default は ``tests/artifacts/``). ファイル名は ``_rust.__has_blas__`` で
自動分岐 (``blas_on.npz`` / ``blas_off.npz``)。

``MAQINA_EXPECT_BLAS`` env var で「期待する build mode」を pin できる
(``=1`` で BLAS on を期待, ``=0`` で off を期待). 期待と
``_rust.__has_blas__`` が不一致なら test を skip + 明示メッセージ
(間違った build を回したときに silent に上書き保存しないため)。

`adaptive Richardson` は accept/reject 境界で dt 履歴が BLAS on/off
間で分岐しうるため本 test では除外する (fidelity 比較は
``tests/test_reference_qutip.py`` 側で担保)。
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from maqina import IsingProblem, Observable, QuantumAnnealer, Schedule
from maqina.initial_states import uniform_superposition

_rust = pytest.importorskip("maqina._rust")


# --- artifact 出力先解決 ---------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_ARTIFACT_DIR = _REPO_ROOT / "tests" / "artifacts"


def _resolve_artifact_dir() -> Path:
    """artifact 出力先ディレクトリを返す (env var override 可).

    ``MAQINA_ARTIFACT_DIR`` が設定されていればその path を Path 化して
    返す. 未設定なら ``<repo>/tests/artifacts/`` を返す. どちらの場合も
    ディレクトリは本関数では作らない (caller が ``mkdir`` する).
    """
    env_dir = os.environ.get("MAQINA_ARTIFACT_DIR")
    if env_dir:
        return Path(env_dir)
    return _DEFAULT_ARTIFACT_DIR


def _check_expected_blas() -> None:
    """``MAQINA_EXPECT_BLAS`` env var と ``_rust.__has_blas__`` の整合をチェック.

    env var が未設定ならスキップしない. 設定済みで build mode と不一致なら
    ``pytest.skip`` で明示的にスキップする (誤った build に対して artifact を
    silent に上書きしないため).
    """
    expect = os.environ.get("MAQINA_EXPECT_BLAS")
    if expect is None:
        return
    expect_blas = expect.strip() in ("1", "true", "True", "on", "ON")
    has_blas = bool(_rust.__has_blas__)
    if expect_blas != has_blas:
        pytest.skip(
            f"MAQINA_EXPECT_BLAS={expect!r} but _rust.__has_blas__={has_blas!r}; "
            "rebuild with the requested feature set before regenerating the artifact."
        )


# --- sample input set -----------------------------------------------------

# (label, n, seed). n は dim を小さく抑える ({16, 64, 256}); rel < 1e-13 を
# 確実に通すため Lanczos 内部の Level-1 BLAS reduction order 差が累積しない
# サイズに絞る. seed は固定 (再現可能).
_SAMPLE_INPUTS: tuple[tuple[str, int, int], ...] = (
    ("n4_seed0", 4, 0xC4A4),
    ("n6_seed1", 6, 0xC4A4 + 1),
    ("n8_seed2", 8, 0xC4A4 + 2),
)

# 比較対象 method 集合. adaptive Richardson は accept/reject 境界で dt 履歴
# が BLAS on/off 間で分岐しうるため除外 (fixed dt 経路のみ). cfm4 は
# Lanczos を 2 回呼ぶため BLAS on/off 差が最も累積するパスで, ここで通れば
# m2 も自動的に通る関係にある.
_METHODS: tuple[str, ...] = ("m2", "trotter", "trotter_suzuki4", "cfm4")

_T_END: float = 0.5
_N_STEPS: int = 100
_M_LANCZOS: int = 24


def _build_sample(n: int, seed: int) -> tuple[IsingProblem, Schedule, np.ndarray]:
    """``(problem, schedule, psi0)`` をシード固定で組み立てる.

    ``h_x`` は ``Uniform(0.5, 1.5)``, ``H_p_diag`` は ``Uniform(-1, 1)``. T
    はモジュール定数. linear schedule.
    """
    rng = np.random.default_rng(seed)
    h_x = rng.uniform(0.5, 1.5, size=n).astype(np.float64)
    h_p_diag = rng.uniform(-1.0, 1.0, size=1 << n).astype(np.float64)
    prob = IsingProblem(n=n, H_p_diag=h_p_diag, h_x=h_x)
    sched = Schedule.linear(T=_T_END)
    psi0 = uniform_superposition(n)
    return prob, sched, psi0


def _run_and_collect(
    prob: IsingProblem, sched: Schedule, psi0: np.ndarray, method: str
) -> dict[str, np.ndarray]:
    """1 (problem, method) について ``QuantumAnnealer.run`` を呼び結果を dict 化.

    観測量は ``ising_energy`` (= H_problem) と ``magnetization_z``. save_tlist
    で 5 点採取して時系列もダンプする. fixed dt 経路のみ前提.
    """
    obs = {
        "ising_energy": Observable.ising_energy(prob),
        "magnetization_z": Observable.magnetization(prob.n, axis="z"),
    }
    save_tlist = np.linspace(0.0, _T_END, 5, dtype=np.float64)

    ann = QuantumAnnealer(prob, sched, m=_M_LANCZOS)
    res = ann.run(
        psi0,
        0.0,
        _T_END,
        method=method,  # type: ignore[arg-type]
        n_steps=_N_STEPS,
        observables=obs,
        save_tlist=save_tlist,
    )

    out: dict[str, np.ndarray] = {
        "psi_final": np.ascontiguousarray(res.psi_final),
        "probabilities": np.ascontiguousarray(res.probabilities),
    }
    # save_tlist 経路では times / observables_history が non-None.
    if res.times is not None:
        out["times"] = np.ascontiguousarray(res.times)
    for name, ts in res.observables_history.items():
        out[f"obs_{name}"] = np.ascontiguousarray(ts)
    return out


def test_blas_consistency_artifact_dump(tmp_path: Path) -> None:
    """全 sample × 全 method の結果を 1 つの ``.npz`` にまとめて dump する.

    最終 array key は ``<sample_label>__<method>__<field>`` の 3 段構造で,
    ``tools/diff_blas_artifacts.py`` が同じ key 列を持つ 2 ファイルを diff
    することで BLAS on / off ビルド間の数値一致を確認する.

    本 test 自身は build mode を問わず常に走り artifact を書き出す. 内部
    sanity check として ``probabilities.sum() ≈ 1`` と ``‖psi_final‖ ≈ 1``
    のみ確認し, BLAS on/off 比較は diff スクリプト側に委譲する.
    """
    _check_expected_blas()

    has_blas = bool(_rust.__has_blas__)
    suffix = "on" if has_blas else "off"

    artifact_dir = _resolve_artifact_dir()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / f"blas_{suffix}.npz"

    bundle: dict[str, np.ndarray] = {}
    for label, n, seed in _SAMPLE_INPUTS:
        prob, sched, psi0 = _build_sample(n, seed)
        for method in _METHODS:
            res_dict = _run_and_collect(prob, sched, psi0, method)
            for field, arr in res_dict.items():
                bundle[f"{label}__{method}__{field}"] = arr

            # sanity (build profile に依存しない自明な不変量).
            psi_final = res_dict["psi_final"]
            probabilities = res_dict["probabilities"]
            norm = float(np.linalg.norm(psi_final))
            psum = float(probabilities.sum())
            assert abs(norm - 1.0) < 1e-9, (
                f"{label} / {method}: ‖psi_final‖ - 1 = {norm - 1.0:.3e} "
                "(unitarity violated)"
            )
            assert abs(psum - 1.0) < 1e-9, (
                f"{label} / {method}: Σ |psi_k|^2 - 1 = {psum - 1.0:.3e}"
            )
            assert np.all(np.isfinite(psi_final)), (
                f"{label} / {method}: psi_final has NaN/inf"
            )

    # build metadata (diff スクリプト側で human-readable な ID として参照).
    bundle["_meta_has_blas"] = np.array([1 if has_blas else 0], dtype=np.int8)
    bundle["_meta_has_rayon"] = np.array(
        [1 if bool(getattr(_rust, "__has_rayon__", False)) else 0], dtype=np.int8
    )
    bundle["_meta_has_simd"] = np.array(
        [1 if bool(getattr(_rust, "__has_simd__", False)) else 0], dtype=np.int8
    )

    # 出力: ``np.savez`` (uncompressed) で十分な size. compress は CI artifact
    # アップロード時に gzip がかかるので二重圧縮は避ける.
    np.savez(artifact_path, **bundle)

    # tmp_path にも複製を残して pytest の per-test artifact 仕様に乗せる
    # (CI でテストが落ちたとき pytest --capture=no で発見しやすい).
    tmp_copy = tmp_path / f"blas_{suffix}.npz"
    np.savez(tmp_copy, **bundle)

    # 健全性: 期待 key 数 = len(_SAMPLE_INPUTS) × len(_METHODS) × 5 fields
    # (psi_final + probabilities + times + obs_ising_energy + obs_magnetization_z)
    # + 3 meta フィールド.
    expected = len(_SAMPLE_INPUTS) * len(_METHODS) * 5 + 3
    assert len(bundle) == expected, (
        f"unexpected artifact key count: got {len(bundle)}, expected {expected}"
    )


# ============================================================================
# Chebyshev variant (issue #122 Phase B)
# ============================================================================
#
# adaptive Richardson 経路は PI controller の accept/reject 境界で dt 履歴が
# 分岐しうるため (上記 _METHODS から除外している), Chebyshev 経路の BLAS
# on/off 数値一致は **PI controller を介さず Rust step 関数を直接呼ぶ** ことで
# 検証する. fixed schedule 係数を渡し ``cfm4_step_chebyshev_py`` (1 step) と
# ``cfm4_step_chebyshev_with_richardson_estimate_py`` (full + half×2 = 3 step
# の constellation) の出力 ψ / err / k_used を artifact に dump する.

_CHEBYSHEV_SAMPLES: tuple[tuple[str, int, int], ...] = _SAMPLE_INPUTS


def _run_chebyshev_direct(n: int, seed: int) -> dict[str, np.ndarray]:
    """fixed schedule 係数で ``cfm4_step_chebyshev_*`` を直接呼んで結果を dict 化.

    PI controller を介さないため BLAS on/off 間で「呼出回数 / dt 履歴」は完全
    同一. 内部の BLAS-1 演算順序差で出る数値差のみが diff の対象.
    """
    rng = np.random.default_rng(seed)
    dim = 1 << n
    h_x = np.ascontiguousarray(rng.uniform(0.5, 1.5, size=n).astype(np.float64))
    h_p_diag = np.ascontiguousarray(rng.uniform(-1.0, 1.0, size=dim).astype(np.float64))
    psi0_arr = np.ascontiguousarray(uniform_superposition(n))

    # Gershgorin 上下界の precompute (Chebyshev propagator の per-step O(1) 入力).
    h_x_abs_sum = float(np.abs(h_x).sum())
    h_p_min = float(h_p_diag.min())
    h_p_max = float(h_p_diag.max())

    # 固定 schedule 係数. linear schedule の `t = 0.25 · T, T = 0.5` に近い
    # 中盤を模した値. CFM4:2 の 2 stage ノードはランダムに振ってある.
    a_s1, b_s1 = 0.55, 0.45
    a_s2, b_s2 = 0.45, 0.55
    dt = 0.05
    cheb_tol = 1e-12

    out: dict[str, np.ndarray] = {}

    # 1) single step (cfm4_step_chebyshev_py).
    psi_step, k_step, err_step = _rust.cfm4_step_chebyshev_py(
        psi0_arr,
        h_x,
        h_p_diag,
        a_s1,
        b_s1,
        a_s2,
        b_s2,
        dt,
        cheb_tol,
        h_x_abs_sum,
        h_p_min,
        h_p_max,
    )
    out["step_psi"] = np.ascontiguousarray(psi_step)
    out["step_k_used"] = np.array([int(k_step)], dtype=np.int64)
    out["step_err_cheb"] = np.array([float(err_step)], dtype=np.float64)

    # 2) Richardson constellation (full + half×2). half ノードはダミー.
    a_s1_h1, b_s1_h1 = 0.54, 0.46
    a_s2_h1, b_s2_h1 = 0.48, 0.52
    a_s1_h2, b_s1_h2 = 0.52, 0.48
    a_s2_h2, b_s2_h2 = 0.46, 0.54

    psi_rich, err_rich, k_rich, err_cheb_rich = (
        _rust.cfm4_step_chebyshev_with_richardson_estimate_py(
            psi0_arr,
            h_x,
            h_p_diag,
            a_s1,
            b_s1,
            a_s2,
            b_s2,
            a_s1_h1,
            b_s1_h1,
            a_s2_h1,
            b_s2_h1,
            a_s1_h2,
            b_s1_h2,
            a_s2_h2,
            b_s2_h2,
            dt,
            cheb_tol,
            False,
            h_x_abs_sum,
            h_p_min,
            h_p_max,
        )
    )
    out["rich_psi"] = np.ascontiguousarray(psi_rich)
    out["rich_err"] = np.array([float(err_rich)], dtype=np.float64)
    out["rich_k_used_total"] = np.array([int(k_rich)], dtype=np.int64)
    out["rich_err_cheb_total"] = np.array([float(err_cheb_rich)], dtype=np.float64)
    return out


def test_blas_consistency_chebyshev_artifact_dump(tmp_path: Path) -> None:
    """Chebyshev 経路の BLAS on/off 数値一致 artifact dump (issue #122).

    PI controller を介さない direct Rust call で fixed schedule 係数を
    渡し, 出力 ψ / err / k_used を artifact 化する. ``diff_blas_artifacts.py``
    で同じ key 列を BLAS on/off 間で diff し ``rel < 1e-13`` を要求する.
    artifact ファイル名は ``blas_chebyshev_{on,off}.npz``.
    """
    _check_expected_blas()

    has_blas = bool(_rust.__has_blas__)
    suffix = "on" if has_blas else "off"

    artifact_dir = _resolve_artifact_dir()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / f"blas_chebyshev_{suffix}.npz"

    bundle: dict[str, np.ndarray] = {}
    for label, n, seed in _CHEBYSHEV_SAMPLES:
        res_dict = _run_chebyshev_direct(n, seed)
        for field, arr in res_dict.items():
            bundle[f"{label}__chebyshev__{field}"] = arr
        psi_step = res_dict["step_psi"]
        norm = float(np.linalg.norm(psi_step))
        assert abs(norm - 1.0) < 1e-9, (
            f"{label} / chebyshev step: ‖psi‖ - 1 = {norm - 1.0:.3e}"
        )
        assert np.all(np.isfinite(psi_step)), (
            f"{label} / chebyshev: psi_step has NaN/inf"
        )

    bundle["_meta_has_blas"] = np.array([1 if has_blas else 0], dtype=np.int8)
    bundle["_meta_has_rayon"] = np.array(
        [1 if bool(getattr(_rust, "__has_rayon__", False)) else 0], dtype=np.int8
    )
    bundle["_meta_has_simd"] = np.array(
        [1 if bool(getattr(_rust, "__has_simd__", False)) else 0], dtype=np.int8
    )

    np.savez(artifact_path, **bundle)
    tmp_copy = tmp_path / f"blas_chebyshev_{suffix}.npz"
    np.savez(tmp_copy, **bundle)

    # key 数: 各 sample 7 field (step_psi / step_k_used / step_err_cheb /
    # rich_psi / rich_err / rich_k_used_total / rich_err_cheb_total) × samples
    # + 3 meta.
    expected = len(_CHEBYSHEV_SAMPLES) * 7 + 3
    assert len(bundle) == expected, (
        f"unexpected chebyshev artifact key count: got {len(bundle)}, expected {expected}"
    )
