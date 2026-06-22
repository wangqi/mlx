# PrismML low-bit quantization patch

// PrismML 1-bit/2-bit affine quantization support, re-applied onto the mlx-core base
// used by thirdparty/mlx-swift (commit ce45c52, MLX 0.31.1).
// wangqi modified 2026-06-22

This fork of `ml-explore/mlx` carries the PrismML 1-bit/2-bit affine quantization
delta on top of the mlx-core base vendored by `thirdparty/mlx-swift`.

- **Branch:** `prism-1bit-0.31.1`
- **Base:** `ce45c52` (`[CUDA] Use qmv kernel for fp quantizations (#3239)`, MLX 0.31.1) —
  identical to PrismML's patch base.
- **Patch:** cherry-pick of PrismML-Eng/mlx `ce45c52..d90771c` (3 commits):
  1. `Add 1-bit affine quantization support`
  2. `Guard fast-path Metal kernel dispatch for 1-bit quantization`
  3. `host support for bias-free (symmetric) 1/2-bit affine qmv`
- **Scope:** quantization path only (`mlx/ops.cpp`, `mlx/primitives.cpp`,
  `mlx/backend/cpu/quantized.cpp`, `mlx/backend/metal/quantized.cpp`,
  `mlx/backend/metal/jit_kernels.cpp`, `mlx/backend/metal/kernels/quantized*.{h,metal}`,
  `mlx/backend/metal/kernels/quantized_nax*.{h,metal}`, python bindings/tests, benchmarks).
- **Regression guard:** the delta is additive + gated — host dispatch selects the fast
  path only for `bits >= 2`; generated kernels add a leading `if (bits==1)` branch so
  `bits >= 2` falls through unchanged. Proven bit-exact for 2/4/8-bit by
  `thirdparty/mlx-swift/Tests/MLXTests/QuantizationTests.swift`.

## Re-application on upgrade

When `thirdparty/mlx-swift-lm` raises the required `mlx-swift`/`mlx-core` version, this
branch must be rebased onto the new mlx-core base and the regression gate re-run. See the
`mlx-swift-lm-upgrade` skill (step 7) and `helper/docs/mlx-swift.md`.
