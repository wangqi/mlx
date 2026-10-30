# PrismML low-bit quantization patch

// PrismML 1-bit/2-bit affine quantization support, re-applied onto the mlx-core base
// used by thirdparty/mlx-swift (commit 36969c094, MLX v0.32.4, vendored by mlx-swift 0.33.0).
// wangqi modified 2026-06-22 / 2026-10-17 / 2026-10-30

This fork of `ml-explore/mlx` carries the PrismML bias-free (symmetric) 1/2-bit affine
quantization delta on top of the mlx-core base vendored by `thirdparty/mlx-swift`.

- **Branch:** `prism-1bit-0.32.4` (previous: `prism-1bit-0.32.2` off `1f8e74e` = current `main`,
  and `prism-1bit-0.31.1` off `ce45c52`; both kept for rollback)
- **Base:** `36969c094` (`[BUG] Metal fence deadlock (#4689)`, tag `v0.32.4`) — the exact
  `Source/Cmlx/mlx` commit mlx-swift `0.33.0` vendors. PrismML's own branches (`prism/prism`,
  `sync/swift-core-bonsai2-20260921`) sit on mlx `main` past the release, so they are not the base.
- **Upstream now carries the 1-bit affine format itself.** `e0408d473` (`Add 1-bit affine
  quantization support (Metal) (#3161)`, between v0.32.2 and v0.32.4) is PrismML's commit 1
  as reviewed by Apple: `bits == 1` in `ops.cpp` / `cpu/quantized.cpp` / `python`, the
  `if (bits == 1)` dequantize branches in `kernels/quantized.h` and `quantized_nax.h`
  (unrolled form of the same math, plus a tail block for the 1-bit 2048-wide `qmv_fast`
  block size), and commit 2's `K == 64 && bits >= 2` `qmv_quad` guard. The previous
  commits 1 and 2 are therefore **not** on this branch (cherry-picks resolved to the base
  and skipped as empty); the upstream kernels are used as-is.
- **Patch (what this branch adds over v0.32.4):**
  1. `5c30ce419` `host support for bias-free (symmetric) 1/2-bit affine qmv` — cherry-pick
     of the previous `41bbd1896`: `quantized_matmul(..., biases = nullopt, mode = "affine")`
     builds a 3-input `QuantizedMatmul` (`Affine` mode, bits in {1, 2}); `eval_gpu` maps it to
     mode string `affine_sym` on the transposed small-M (`qmv`) path only; `eval_cpu` rejects
     it; `vjp` / `jvp` read `primals[3]` only when present; `get_quantized_kernel_wrapped` /
     `jit_kernels.cpp` map any `affine*` mode onto the affine kernel source; `affine_quantize`
     also accepts group sizes 256 and 512.
  2. `1c0bb0310` `Port the PrismML bias-free affine_sym 1/2-bit qmv kernels into the core` —
     cherry-pick of the previous `7591f4029`: `sym_derived_bias`, the `bias_free` template
     parameter on `qmv_fast_impl` / `qmv_impl`, and the `affine_sym_qmv_fast` / `affine_sym_qmv`
     kernels (scales-only buffer table: no `biases` (2) and no `b_strides` (14), matching
     `add_strides_and_shapes` when `biases` is empty).
- **Scope:** quantization path only (`mlx/ops.cpp`, `mlx/primitives.cpp`,
  `mlx/backend/cpu/quantized.cpp`, `mlx/backend/metal/quantized.cpp`,
  `mlx/backend/metal/jit_kernels.cpp`, `mlx/backend/metal/kernels/quantized.h`).
- **Regression guard:** the delta is additive + gated — `bias_free` defaults to `false`, the
  `affine_sym` kernels are JIT-only and selected by mode string, and every host change is
  behind `biases == nullopt`. Proven bit-exact for 2/4/8-bit by
  `thirdparty/mlx-swift/Tests/MLXTests/QuantizationTests.swift`.

### 2026-10-30 rebase onto v0.32.4 — conflicts resolved

- Commit 1 (`Add 1-bit affine quantization support`): conflicts in `kernels/quantized.h`,
  `kernels/quantized_nax.h` and `python/tests/test_quantized.py` against upstream `#3161`,
  which already contains it. Resolved entirely to the base (upstream's version) and skipped;
  PrismML's duplicate `benchmarks/python/comparative/` hooks were dropped with it (`#3161`
  brings its own 1-bit benchmark support).
- Commit 2 (`Guard fast-path Metal kernel dispatch for 1-bit quantization`): the gate itself
  is upstream (`#3161`); only its comment conflicted. Resolved to the base and skipped.
- Commit 3 (`41bbd1896`): one conflict, the `dispatch_qmv` `qmv_quad` gate comment; the gate
  line auto-merged as upstream's guard + `mode != "affine_sym"`.
- `7591f4029`: `qmv_fast_impl` grew `bool partial_rows` upstream (`#4516`), so `bias_free` is
  now the fifth template parameter; the `affine_sym_qmv_fast` / `affine_sym_qmv` kernels take
  `partial_rows` after `batched`, mirroring `affine_qmv_fast` / `affine_qmv`, so they match the
  host's `get_quantized_kernel_wrapped(type, group_size, bits, batched, partial_rows,
  has_global_scale, results_per_simdgroup)`. Upstream's new 1-bit tail block in `qmv_fast_impl`
  (`#3161`) is one more `bl[0]` site and takes the same `bias_free` derivation.
- **Pre-existing gap closed:** `use_qmv_wide(mode, d)` is `mode != "affine" || gen >= 15`, so
  on every branch since `prism-1bit-0.32.2` a bias-free call with `2 <= M < vector_limit` was
  routed to a non-existent `affine_sym_qmv_wide` kernel. `dispatch_qmv` now excludes
  `affine_sym` from the `qmv_wide` route (falls through to the plain `qmv`, as for `qmv_quad`).

### 2026-10-17 rebase onto v0.32.2 — conflicts resolved (historical)

- `kernels/quantized.h` / `quantized_nax.h` `dequantize`: upstream accumulated in `float`
  (`s`, `b`, `sc[]`); the 1-bit branch was rewritten in that form, matching PrismML's own
  resolution in `4fc91df60`.
- `metal/quantized.cpp` `dispatch_qmv`: the `qmv_quad` gate combined upstream's `!global_scale`
  with PrismML's `K == 64 && bits >= 2` and `mode != "affine_sym"`.
- `python/tests/cuda_skip.py`: deleted upstream; its one-line 1-bit skip was dropped.
- Bias-free `affine_sym_qmv_fast` / `affine_sym_qmv` kernels, `sym_derived_bias` and the
  `bias_free` template parameter were ported into `kernels/quantized.h` from PrismML-Eng/mlx
  `4fc91df60`. Until then they lived only in mlx-swift's generated `mlx-generated/` files, so
  `tools/update-mlx.sh` would have dropped them.

## Re-application on upgrade

When `thirdparty/mlx-swift-lm` raises the required `mlx-swift`/`mlx-core` version, this
branch must be rebased onto the new mlx-core base and the regression gate re-run. See the
`thirdparty-upgrade` skill (`mlx` adapter, P4) and `helper/docs/mlx-swift.md`.
