# hipBLASLt fp32 tall+skinny matmul: rows past M=2^19 produce garbage on gfx1201

## Summary

`torch.nn.functional.linear(x, W, b)` (i.e. `x @ W.T + b`) with fp32 inputs
on a tall+skinny shape — roughly `M ≥ 2**19`, `K = 64`, `N = 6` — returns
incorrect values for **all rows with index ≥ 524288** on RDNA4 / gfx1201
under ROCm's hipBLASLt backend. Rows `[0, 2**19)` are correct.

Symptoms observed on the same call, depending on input distribution and
allocator state:

- **Denormal-zero variant**: tail rows are `~1e-21` (true result is `~O(1)`).
- **Overflow variant**: tail rows are `±1e17`..`±1e18`, including `inf`/`nan`.

Splitting the same call along `M` into chunks of `≤ 524288` rows
(we use 65536 to stay clear of the boundary) produces the correct,
deterministic result with bit-stability across runs.

## Environment

- GPU: AMD Radeon RX 9070 XT (gfx1201, RDNA4)
- ROCm: 7.2.2 (HIP runtime 7.2.53211)
- PyTorch: 2.11.0+rocm7.2.2
- Backend: hipBLASLt (default fp32 linear path in PyTorch ROCm wheel)

## Minimal reproducer

`rdna4/trellis2/repro_hipblaslt_tall_skinny.py` (in this repo). About 50
lines, no model dependencies — just `torch` + `torch.nn.functional`.

```bash
cd rdna4/trellis2
.venv/bin/python repro_hipblaslt_tall_skinny.py
```

Core of the repro:

```python
M, K, N = 1_500_482, 64, 6
x = torch.randn(M, K, device='cuda', dtype=torch.float32)
W = (torch.rand(N, K, device='cuda', dtype=torch.float32) - 0.5) * 0.68
b = (torch.rand(N, device='cuda', dtype=torch.float32) - 0.5) * 0.2

out_single = F.linear(x, W, b)            # buggy

out_chunked = torch.empty(M, N, device='cuda', dtype=torch.float32)
for s in range(0, M, 65536):              # workaround
    out_chunked[s:s+65536] = F.linear(x[s:s+65536], W, b)
```

Observed output on our setup (denormal-zero variant + some NaNs):

```
out_single : shape=(1500482, 6)  std=nan   inf=346  nan=247
out_chunked: shape=(1500482, 6)  std=1.586 inf=0    nan=0
fraction |x|<1e-3 by region:
  rows[0..524287]   = 0.0005   (correct)
  rows[524288..]    = 0.9442   (denormal/zero)
rows where any element diverges by >1e-3: 976194 / 1500482  (65.06%)
```

Sharp boundary at exactly `M = 1<<19 = 524288`.

## Where it surfaced

TRELLIS-2 tex_slat_decoder's final layer is a 64→6 linear over
`M ≈ 1.5M` post-LayerNorm voxel features. The pipeline produced
non-deterministic NaN/`std=∞` output across runs depending on
allocator state. Bisected to:

- `cpu/trellis2/trellis2_repo/trellis2/modules/sparse/linear.py`'s
  `SparseLinear.forward` → `nn.Linear.forward` → `F.linear` →
  hipBLASLt fp32 GEMM.

## Scope of the bug (sweep)

`rdna4/trellis2/sweep_hipblaslt_bug.py` runs the same single-vs-chunked
comparison across `(M, K, N, dtype)`. Findings:

- The boundary is dtype-keyed, not shape-keyed:
  - **fp32**: all rows with index `≥ 2**19 = 524288` are wrong.
  - **fp16 / bf16**: all rows with index `≥ 2**20 = 1048576` are wrong.
- Within each dtype, the bug fires for **every K and N tested**:
  K ∈ {32, 64, 128, 256}, N ∈ {6, 8, 16, 64}. It is not specific to
  the K=64, N=6 corner.
- `M == 2**19` exactly is correct in fp32; `M == 2**19 + 1` is broken.
  Same for fp16/bf16 at `2**20`.

So the bug is a **per-row stride/index miscount in the M-loop** that
fails once a row index exceeds a dtype-dependent threshold (most likely
`INT_MAX / element_size` or a 2 GiB byte-offset wrap in the output
tensor stride). Examples:
- fp32, M=1500482, N=64 → 1.7M `nan`s in the output.
- fp32, K=64, N=6, M>=524289 → tail rows are denormal-zero or `±1e18`.

## Dispatch (from `HIPBLASLT_LOG_LEVEL=4`)

```
rocblaslt_matmul
  A=R_32F rows=64 cols=6  ld=64        # weight (K, N)
  B=R_32F rows=64 cols=M  ld=64        # input  (K, M)
  C=R_32F rows=6  cols=M  ld=6         # output (N, M) col-major == [M, N] row-major
  computeType=COMPUTE_32F scaleType=R_32F
  transA=OP_T transB=OP_N
  epilogue=EPILOGUE_BIAS
  alpha=1 beta=0
```

Note: setting `HIPBLASLT_LOG_LEVEL=4` itself perturbs the dispatch heuristic
on our setup — under the verbose-log path the same call no longer produces
denormal-zero or `nan`/`inf`, but rows past M=2^19 still diverge by up to
`max|d|≈10.7` from the chunked reference (65% of those rows). So the bug is
not just in one strategy: the M-axis miscount affects multiple Tensile
solutions, just with different visible failure modes (denormal-zero,
overflow, or quiet wrong values).

## Suspected cause

The dtype-dependent doubling (`2**19` for fp32 → `2**20` for fp16/bf16)
is the giveaway: the limit is on **bytes addressed**, not rows. Both
`524288 * 6 * 4 == 12 MiB` and `1048576 * 6 * 2 == 12 MiB` are far
below 2 GiB, so it is not a raw byte overflow on the output. More
likely:

- An int32 multiply that goes into the kernel as `row_idx *
  element_size` or `row_idx * stride_bytes` and wraps once
  `row_idx_in_units` crosses the per-dtype limit, or
- A Tensile tile-coverage / split-K heuristic that picks a tile count
  based on a divisor of `2**19` for fp32 (the workgroup splits along M
  with a stride that aliases past the boundary).

Confirming requires inspecting the dispatched Tensile kernel; we have
not done that yet.

## Workaround (in this repo)

`rdna4/trellis2/spconv_rdna4_ext.py::install_sparse_linear_chunking()`
monkey-patches `trellis2.modules.sparse.linear.SparseLinear.forward` to
chunk along M into 65536-row segments. Gated on
`USE_RDNA4_LINEAR_CHUNK=1` (default ON). Verified across 4 trials:
identical output `min=-1.223 max=1.160 std=0.763`.

Default chunk size is `2**18 = 262144`, half the fp32 boundary. The
sweep shows fp16/bf16 only break past `2**20`, so this default is
strictly safe for any dtype on this codebase. We have not patched
generic `nn.Linear`: in the TRELLIS-2 stage-2 path the only call with
M past `2**19` is `tex_slat_decoder.output_layer` which goes through
`SparseLinear`. If a future model adds a plain `nn.Linear` over
`> 2**19` rows in fp32 (or `> 2**20` rows in fp16/bf16), it will hit
this same bug and need the same chunked wrapping.

## Status

- Workaround landed and on by default for the TRELLIS-2 path.
- Sweep `rdna4/trellis2/sweep_hipblaslt_bug.py` confirms the bug is
  dtype-keyed (`2**19` fp32, `2**20` fp16/bf16), not K/N specific.
- Upstream report not yet filed.
- Repro is self-contained and ready to attach to a hipBLASLt issue.
