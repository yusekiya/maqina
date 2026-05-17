# §10. ベンチマーク戦略

ベンチ規約:

- 性能改善の主張は **同一マシン上の before/after**。CPU / BLAS / NumPy /
  熱状態が揃った状態で取る。別マシンの絶対値表と並べて「○○× 速くなった」
  と主張しない。
- ベンチスクリプトは `benchmarks/bench_<対象>.py` 命名規約、argparse CLI、
  `benchmarks/results/<YYYYMMDD-HHMMSS>/` への CSV + markdown 出力。
- 同一マシン上で取った datapoint を `benchmarks/README.md` に書き戻す
  ときは、使用ハード (機種 / チップ / メモリ / OS / NumPy / BLAS backend)
  を節タイトルで明示する。

最初に用意するベンチ:

- `bench_per_step.py`: M2 / CFM4 / Richardson の per-step wall time
  (n を sweep)。
- `bench_blas_compare.py`: BLAS feature on/off の同一マシン比較。
- `bench_vs_qutip.py`: Section 8.3。

### 10.1 期待される性能特性 (推定)

- N = 20 (dim 2^20 ≈ 10^6): ψ 16 MB + H_p_diag 8 MB ≈ 24 MB。matvec の
  bit-flip pass が支配 (N × 2^N ≈ 2 × 10^7 ops/step × 4 m matvec/step)。
  Apple Accelerate を使った dense GEMV を凌ぐ matrix-free 性能を目指す。
- N = 24: ψ 256 MB + H_p_diag 128 MB ≈ 384 MB。M2 メモリでも単機で
  動かせる。CFM4:2 ステージで一時バッファ数本必要なので 1 GB 弱まで。
- N = 26 以上: ψ 1 GB+、shared-mem 単機の限界。GPU / 分散版は future work。

---

