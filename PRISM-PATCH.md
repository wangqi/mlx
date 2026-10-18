# PrismML low-bit quantization patch

// PrismML 1-bit/2-bit affine quantization support, re-applied onto the mlx-core base
// used by thirdparty/mlx-swift (commit 1f8e74e, MLX v0.32.2, vendored by mlx-swift 0.32.3).
// wangqi modified 2026-06-22 / 2026-10-17

This fork of `ml-explore/mlx` carries the PrismML 1-bit/2-bit affine quantization
delta on top of the mlx-core base vendored by `thirdparty/mlx-swift`.

- **Branch:** `prism-1bit-0.32.2` (previous: `prism-1bit-0.31.1` off `ce45c52`, kept for rollback)
- **Base:** `1f8e74e` (`Hold GIL in AttachedData destructor (#4391)`, tag `v0.32.2`) — the exact
  `Source/Cmlx/mlx` commit mlx-swift `0.32.3` vendors. PrismML's own branches (`prism/prism`,
  `sync/swift-core-bonsai2-20260921`) sit on mlx `main` past v0.32.2, so they are not used as the base.
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

### 2026-10-17 rebase onto v0.32.2 — conflicts resolved

- `kernels/quantized.h` / `quantized_nax.h` `dequantize`: upstream now accumulates in `float`
  (`s`, `b`, `sc[]`); the 1-bit branch was rewritten in that form, matching PrismML's own resolution
  in `4fc91df60`.
- `metal/quantized.cpp` `dispatch_qmv`: the `qmv_quad` gate combines upstream's `!global_scale`
  with PrismML's `K == 64 && bits >= 2` and `mode != "affine_sym"`.
- `python/tests/cuda_skip.py`: deleted upstream; its one-line 1-bit skip is dropped.
- Bias-free `affine_sym_qmv_fast` / `affine_sym_qmv` kernels, `sym_derived_bias` and the `bias_free`
  template parameter on `qmv_fast_impl` / `qmv_impl` added to `kernels/quantized.h`, ported from
  PrismML-Eng/mlx `4fc91df60`. Until this branch they lived only in mlx-swift's generated
  `mlx-generated/` files, so `tools/update-mlx.sh` would have dropped them.

## Re-application on upgrade

When `thirdparty/mlx-swift-lm` raises the required `mlx-swift`/`mlx-core` version, this
branch must be rebased onto the new mlx-core base and the regression gate re-run. See the
`mlx-swift-lm-upgrade` skill (step 7) and `helper/docs/mlx-swift.md`.
