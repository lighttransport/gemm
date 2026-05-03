# cuda/fa — Tensor-Core FlashAttention forward (Blackwell sm_120)

Sweeps F16 multi-head self-attention forward (FlashAttention-2 style: tiled
online softmax, no materialized attention matrix) on RTX 5060 Ti, comparing
**our PTX/MMA kernels** against a **non-MMA F32 reference** and a future
cuDNN/cuBLAS GEMM-pair baseline.

Layout (mirrors cuda/gemm/):

- Inputs: `Q[B*H, S, D]`, `K[B*H, S, D]`, `V[B*H, S, D]` (B=batch, H=heads, S=seqlen, D=head_dim)
- Output: `O[B*H, S, D]` row-major within each head
- Scale: `1/sqrt(D)`. No causal mask in v1; non-causal self-attention only.

## Kernel revisions

| rev | precision   | tensor cores | softmax        | status |
|-----|-------------|--------------|----------------|--------|
| v1  | f16 in/f32 acc | no (scalar fma) | tiled online   | working baseline |
| v2  | f16 in/f32 acc | yes (mma.m16n8k16) | tiled online | planned |
| v3  | bf16        | mma.m16n8k16 | tiled online   | planned |
| v4  | fp8 e4m3    | mma.m16n8k32 | tiled online   | planned |

## Build / run

```
make
./bench_cuda_fa --dtype f16 --mode all --shape all
make smoke
```

CLI:
```
--dtype  f16
--mode   ptx | ref | all
--shape  <name> | all
--iters  N (default 50)
--warmup N (default 5)
--verify 0|1 (default 1)
--head-dim D
--seqlen S --heads H --batch B  (ad-hoc shape, overrides --shape)
```

## Shapes

| name | B | H | S | D | role |
|---|---|---|---|---|---|
| qwen3_512  | 1 | 16 | 512  | 128 | Qwen3 attention prefill (short ctx) |
| qwen3_2k   | 1 | 16 | 2048 | 128 | Qwen3 attention prefill (medium) |
| qwen3_4k   | 1 | 16 | 4096 | 128 | Qwen3 attention prefill (long) |
| dit_1k     | 1 | 24 | 1024 | 64  | DiT-style vision attention |
| sd_8k_d64  | 1 |  8 | 8192 | 64  | long-context with small head_dim |

## Files

- `bench_cuda_fa.c` — host driver
- `cuda_fa_kernels.h` — NVRTC-compiled device kernels (CUDA C++ + PTX inline)

## v1 status

v1 is a single-warp-per-(head,query-row) scalar-fma kernel with SMEM-tiled K/V
and online softmax. Validates against CPU FP32 reference. **Not** tensor-core;
serves as a correctness baseline before MMA work in v2.
