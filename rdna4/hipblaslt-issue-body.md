## Summary

`hipblasLtMatmul` fp32 GEMM with `transA=OP_T transB=OP_N` and small N
returns wrong values for **all rows past `M = 2**19 = 524288`** on
gfx1201 / RDNA4. The same bug fires for fp16 and bf16 past
`M = 2**20 = 1048576`. The boundary is dtype-keyed (doubles when element
size halves), shape-agnostic across K∈{32,64,128,256} and N∈{6,8,16,64}.

Failure mode varies by shape and heuristic state — denormal-zero
(K=64,N≤16,fp32), `±1e17`..`±1e18` overflow, `nan`/`inf`, or quiet
wrong values diverging by ~10× from a chunked reference.

Splitting the call along M into `≤ 2**19` (fp32) / `≤ 2**20` (fp16,bf16)
chunks produces the correct, deterministic result with the same dtype
and same data.

## Environment

- GPU: AMD Radeon RX 9070 XT (gfx1201, RDNA4)
- ROCm: 7.2.2 (HIP runtime 7.2.53211)
- PyTorch: 2.11.0+rocm7.2.2 (calls hipBLASLt for `F.linear` / `nn.Linear`)
- hipBLASLt: shipped with ROCm 7.2.2 (`/opt/rocm-7.2.2/lib/hipblaslt/library`)

## Reproducer (no PyTorch / no model deps required, but easiest in PyTorch)

```python
import torch, torch.nn.functional as F

M, K, N = 1_500_482, 64, 6
torch.manual_seed(0)
x = torch.randn(M, K, device='cuda', dtype=torch.float32)
W = (torch.rand(N, K, device='cuda', dtype=torch.float32) - 0.5) * 0.68
b = (torch.rand(N, device='cuda', dtype=torch.float32) - 0.5) * 0.2

out_single = F.linear(x, W, b)            # buggy

out_chunked = torch.empty(M, N, device='cuda', dtype=torch.float32)
for s in range(0, M, 65536):              # workaround
    out_chunked[s:s+65536] = F.linear(x[s:s+65536], W, b)

print('inf/nan in single:', int(torch.isinf(out_single).sum()), int(torch.isnan(out_single).sum()))
print('max|diff|:', float((out_single - out_chunked).abs().max()))
```

Observed (one variant — others reach `±1e18` or `inf/nan` instead):

```
out_single : shape=(1500482, 6)  std=nan   inf=346  nan=247
out_chunked: shape=(1500482, 6)  std=1.586 inf=0    nan=0
fraction |x|<1e-3 by region:
  rows[0..524287]   = 0.0005   (correct)
  rows[524288..]    = 0.9442   (denormal/zero)
rows divergent by >1e-3: 976194 / 1500482  (65.06%)
```

`HIPBLASLT_LOG_LEVEL=4` shows PyTorch dispatches:

```
rocblaslt_matmul
  A=R_32F rows=64 cols=6  ld=64        # weight (K, N)
  B=R_32F rows=64 cols=M  ld=64        # input  (K, M)
  C=R_32F rows=6  cols=M  ld=6         # output (N, M) col-major
  computeType=COMPUTE_32F
  transA=OP_T transB=OP_N
  epilogue=EPILOGUE_BIAS
  alpha=1 beta=0
```

Setting `HIPBLASLT_LOG_LEVEL=4` itself perturbs the heuristic on our
setup; under verbose logging the failure mode shifts from
denormal-zero/nan/inf to *quiet* wrong values (max|d| ≈ 10.7, 65% of
post-boundary rows divergent). Multiple Tensile solutions are affected,
not a single strategy.

## Scope sweep

`(M, K, N, dtype)` sweep (single F.linear vs chunked F.linear):

| dtype    | M boundary    | Tested K / N             | Result for M > boundary |
|----------|---------------|--------------------------|-------------------------|
| fp32     | 2**19 = 524288 | K∈{32,64,128,256}, N∈{6,8,16,64} | all wrong               |
| fp16     | 2**20 = 1048576 | K=64, N=6                | all wrong               |
| bf16     | 2**20 = 1048576 | K=64, N=6                | all wrong               |

`M == boundary` exactly is correct; `M == boundary + 1` is broken.
The doubling between fp32 and fp16/bf16 strongly suggests an int32 /
byte-stride miscount in the M-loop (some intermediate is computed in
units of *bytes* rather than *elements*, and crosses an int32 or
strategy-internal threshold).

## Suspected cause

- The dtype-dependent doubling rules out a raw row-count limit.
- Shape-agnosticity (any K, any tested N) rules out a single
  Tensile solution misbehaving — likely a shared utility or
  workgroup/tile assignment that wraps once
  `M * sizeof(element)` (or a derived stride/tile-id quantity)
  crosses some boundary.
- M = exactly `2**19` works in fp32, M = `2**19 + 1` does not — this
  is sharp, suggesting an exact threshold check or modular wrap, not
  a gradual numerical issue.

We have not narrowed this to a specific Tensile kernel yet.

## Workaround

Chunk the call along M into segments of `≤ 2**18` rows (we use
262144). Verified bit-deterministic across many runs.

## Reproducer files (in our downstream repo)

- minimal repro: <repo>/rdna4/trellis2/repro_hipblaslt_tall_skinny.py
- shape sweep:   <repo>/rdna4/trellis2/sweep_hipblaslt_bug.py
