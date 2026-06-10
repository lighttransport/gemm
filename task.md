# MoE Batched Prefill Optimization — Continuation Guide

## Current State (June 2026)

**Hardware:** RTX 5060 Ti (GB206, sm_120), 16 GB GDDR7, 36 SMs  
**Model:** Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf (35B, 256 experts, top-8, expert_ff=512, n_embd=2048)  
**Quantization:** MoE gate/up/down = IQ2_XXS (type=16, 66 bytes/256-block), Attention/SSM projections = IQ2_XXS/IQ3_XXS

### Latest Benchmark (June 2026)

| Batch | tok/s | Profile |
|---|---|---|
| 512 (warm) | **~2810** | ~182ms; ffn experts≈68-88, **scan=30.2** (W=4), gemm≈35, attn core≈15 |
| 512 (cold) | ~2500 | first-iteration / cold boost clock; always warm up + interleave A/B |
| llama.cpp 512 | **2620.91** | `llama-bench -p 512 -n 0 -b 2048 -ub 512 -ngl -1 -fa auto` |

**IMPORTANT measurement note:** GPU boost clock makes cold runs read ~2500 and warm runs
~2760-2810. Always warm up once and interleave the A/B (`CUDA_LLM_SCAN_W=1` vs `=4`, etc.)
in a single loop, or the clock drift dwarfs the kernel delta.

### Session results (June 2026): pushed toward 3000 → landed ~2810

- **MMQ block-major weight repack** (`CUDA_LLM_MMQ_REPACK`, default-on) — the FFN lever, FFN
  122→106ms. **DeltaNet scan W=4 warps/block** (`CUDA_LLM_SCAN_W`, default 4) — +2% occupancy win.
- **3000 tok/s is NOT reachable on this GPU/model**, and the two walls are now established empirically:
  1. The DeltaNet scan (30ms) is **latency-bound** on the sequential recurrence (see #4). W=4 is the
     only safe lever; a shared-q/k variant was slower.
  2. The **chunked-parallel scan** (the textbook way to break the sequential dependency) is correct
     but **occupancy-bound here** — only 32 SSM heads = 32 blocks can't fill 36 SMs → 9× slower (see #4).
  3. MMQ FFN is at its MoE floor; projection GEMMs are FLOPs-bound (fusion saves no FLOPs and needs
     invasive strided-output plumbing).
- Realistic warm ceiling **~2810** without a model/GPU with many more SSM heads (so the chunked batch
  fills the GPU) or a chunked kernel parallelized across more than the head dimension.

`cuda/llm/bench_prefill_compare.sh` compares llama.cpp and this runner on the same model:

```bash
LD_LIBRARY_PATH=/usr/local/cuda/lib64 ./cuda/llm/bench_prefill_compare.sh \
  /mnt/disk01/models/qwen35moe/35b/Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf 512 1
```

Detailed bucket profiling is available without changing the normal benchmark path:

```bash
CUDA_LLM_PREFILL_DETAIL=1 CUDA_LLM_USE_CUBLAS=1 LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
  ./cuda/llm/test_cuda_llm /mnt/disk01/models/qwen35moe/35b/Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf \
  --large-bench 512 --large-bench-random --large-bench-seed 1
```

Latest detail line after warm-up (CUDA_LLM_MOE_MMQ=1, scan W=4 default):

```
ssm[param=0.4 conv=7.0 norm=1.4 scan=30.2 gate=1.7]
attn[prep=1.3 core=14.8 post=2.3]
ffn[router=3.1 topk=7.9 pack=2.2 experts=68-88 shared=5.7]
```

### Key Files

| File | Purpose |
|---|---|
| `cuda/llm/cuda_llm_runner.c` | Main runner: struct, kernel strings, prefill/decode logic |
| `cuda/llm/moe_gpu_kernels.cu` | AOT cubin kernels (top-k, dequant, fused, etc.) |
| `cuda/llm/moe_gpu_kernels.cubin` | Compiled cubin for sm_120 |
| `cuda/llm/test_cuda_llm.c` | Test harness: `--large-bench N` for synthetic prefill benchmark |
| `cuda/cublasew.h/c` | cuBLAS wrapper: includes `cublasew_gemm_strided_batched` |
| `cuda/llm/Makefile` | Build: `nvcc -cubin` + `gcc` for runner |

### How to Reproduce

```bash
# Build
cd /home/syoyo/work/gemm/main
nvcc -cubin -arch=sm_120 -o cuda/llm/moe_gpu_kernels.cubin cuda/llm/moe_gpu_kernels.cu
make -C cuda/llm

# Quick test (6 tokens, normal flow)
CUDA_LLM_USE_CUBLAS=1 LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
  ./cuda/llm/test_cuda_llm /mnt/disk01/models/qwen35moe/35b/Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf -n 1

# Large batch prefill benchmark (512 tokens)
CUDA_LLM_USE_CUBLAS=1 LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
  timeout 300 ./cuda/llm/test_cuda_llm /mnt/disk01/models/qwen35moe/35b/Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf \
  --large-bench 512 --large-bench-random --large-bench-seed 1 2>&1 | grep -E "Large bench|profile|detail"
```

**Note:** First run compiles NVRTC kernels (~20s). Second run is the real measurement.

## Architecture Details

### MoE Prefill Flow (working path)

```
For each layer:
  1. RMSNorm on d_batch_xb
  2. SSM (30 layers) or Attention (10 layers)
  3. Residual add: d_batch_x += d_batch_xb
  4. FFN section:
     a. Copy d_batch_x -> d_batch_xb (residual copy)
     b. RMSNorm on d_batch_xb
     c. Router: d_router_logits = d_batch_xb @ gate_inp^T (batched F32 matvec)
     d. GPU top-k: fn_moe_topk_gpu (cubin) -> d_topk_idx, d_topk_wgt
     e. Sync + copy top-k to host
     f. For each unique expert with tokens:
        - Gather rows -> d_batch_mid
        - Convert F32->F16 -> d_batch_f16_scratch
        - Dequant gate/up/down IQ2_XXS -> FP16 in one triplet launch -> d_moe_f16w{,2,3}
        - cuBLAS GEMM gate: d_exp_gate = gathered @ gate_w^T
        - cuBLAS GEMM up: d_exp_up = gathered @ up_w^T
        - SiLU: d_exp_gate = silu(d_exp_gate) * d_exp_up
        - Convert F32->F16 -> d_batch_f16_scratch
        - cuBLAS GEMM down: d_exp_down = silu_gate @ down_w^T
        - Scatter-add weighted: d_batch_x += weight * d_exp_down
     g. Shared expert:
        - shared gate scalar uses batched F32 matvec
        - shared gate/up/down FFN use F16 shadows + cuBLAS when CUDA_LLM_USE_CUBLAS=1
        - batched sigmoid-gated add accumulates into d_batch_x
```

### Buffer Sizes

| Buffer | Size | Purpose |
|---|---|---|
| d_batch_x | alloc_tokens × n_embd | Layer output accumulator |
| d_batch_xb | alloc_tokens × n_embd | RMSNorm'd input |
| d_batch_ff1 | alloc_tokens × expert_ff | Expert gate output |
| d_batch_ff2 | alloc_tokens × expert_ff | Expert up output |
| d_batch_wide | alloc_tokens × wide_dim | Down output / general scratch |
| d_batch_mid | alloc_tokens × mid_dim | Gathered input |
| d_batch_f16_scratch | alloc_tokens × max_in_dim × 2 | F16 input conversion |
| d_temp_f16 | expert_ff × n_embd × 2 | Transient IQ2_XXS→FP16 dequant buffer |
| d_router_logits | n_tokens × n_experts | Router logits |
| d_topk_idx | n_tokens × n_used | Top-k expert indices |
| d_topk_wgt | n_tokens × n_used | Top-k softmax weights |

### Correctness Notes

- `CLLM_PREFILL_EXACT_MAX_TOKENS_DEFAULT=32`, so short and medium prompts use sequential exact prefill.
- 21-token prefill-vs-seq parity prompt passes exactly: `rel_L2_vs_seq=0.000000`.
- Batched large prefill still uses the optimized path above the cutoff; 512-token benchmark remains batched.
- Forced one-token batched prefill now routes batched matvecs through the exact decode matvec kernels; layer-0 stop-after-SSM-residual parity is exact (`rel_L2_vs_seq=0.000000`) and full layer-0 parity is close (`rel_L2_vs_seq=0.000112`).
- Forced batched mode for the 21-token parity prompt (`CUDA_LLM_PREFILL_EXACT_MAX_TOKENS=0`) is still approximate: `rel_L2_vs_seq=0.904984`, with similar but not identical top logits. Keep the exact cutoff enabled for small prompts.
- Sequential decode MoE now avoids the incomplete shared-expert F16 TC helper and uses the scalar shared-gate path; the AOT `moe_shared_gate_gpu` reduction was also fixed to reduce all 4 warps.

### Key Parameters

```
n_embd = 2048
n_experts = 256
n_experts_used = 8
expert_ff = 512
shared_expert_ff = 512
stride_gu = 270336  (row_bytes × expert_ff = 528 × 512 for IQ2_XXS)
stride_d = 270336   (same)
dt_rank = 32
d_state = 128
n_group = 16
```

## Optimization Opportunities (highest leverage first)

### 1. MMQ `mul_mat_id` for MoE experts — LANDED ✅ (the dominant win, 1578→2810)

**Status:** DONE. `mmq_iq2xxs_grouped` (NVRTC, `CUDA_LLM_MOE_MMQ=1`) consumes IQ2_XXS weights +
q8_1 int8 activations via `mma.sync m16n8k32.s8.s8.s32` over a flattened work-list grid. Optimization
ladder (all bit-exact): decode-amortize → 32-lane → work-list grid → shared codebook → branchless
`__vsub4` decode → direct int loads → vec act staging → **block-major weight repack** (the lever:
row-major→block-major IQ2 at load, `CUDA_LLM_MMQ_REPACK` default-on, recovers ~2× cache-line
over-fetch). FFN experts bucket now ~68-88ms. DEAD-ENDS (all no-op/slower, don't retry): coalesced
full-tile shared-staging, activation dedup, double-buffer pipeline, llama.cpp tile-port, cuBLAS
triplet path (3.5× slower). The repack was the only memory-LAYOUT fix that worked; tile/occupancy
knobs all fought a layout problem. See memory `project_mmq_moe.md` for the full ladder.

Relevant llama.cpp source:

- `/home/syoyo/work/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu`: `ggml_cuda_mul_mat_id` dispatches MMQ when `ggml_cuda_should_use_mmq(src0->type, cc, ne12, n_experts)` is true.
- `/home/syoyo/work/llama.cpp/ggml/src/ggml-cuda/mmq.cuh`: `mul_mat_q` consumes `ids_dst` and `expert_bounds` for compact MoE routing.

### 2. Grouped MoE expert prefill fallback

**Impact:** If MMQ port is too large, reduce launch count and dequant overhead in the current cuBLAS path.  
**Current:** Build per-expert token lists on GPU or CPU, gather into `[expert, max_tokens_per_expert, n_embd]`, dequant all active expert weights with a 2D dequant kernel, then use strided/grouped batched GEMM and one scatter. This is lower risk than full MMQ but still writes full F16 expert weights, so it is likely less effective than llama.cpp's quantized MMQ path.

### 3. Flash attention / MMA attention

**Impact:** Current attention core is ~14.6ms at 512 tokens after warp-level QK parallelism.  
**Current:** `batch_attn_causal_f32` now computes QK with 8 warps per query/head CTA. It is still scalar V accumulation over F32 Q and F16 KV cache. A true FlashAttention-style tiled kernel or llama.cpp fattn port may still help, but FFN is now the dominant gap.

### 4. DeltaNet scan — EXHAUSTED (latency-bound; chunked alternative is occupancy-bound)

**Impact:** SSM scan ~30ms (#2 kernel overall, ~17% of total).  
**Current best:** register-shard fast path for `d_state==128` + **W=4 independent warps/block**
(`CUDA_LLM_SCAN_W`, default 4, commit `3e342ca`). The 1-warp/block launch capped occupancy at
the 32-blocks/SM limit (32 of 64 possible warps/SM); packing 4 independent warps/block (no shared,
no barriers) lifts that to full occupancy → scan 33.6→30.2ms, +2% (~2760→2810 warm). Bit-correct.

**The scan is LATENCY-bound on the sequential recurrence, not bandwidth-bound** — proven by:
- A shared-q/k variant (R rows/block sharing per-token q/k via shared + `__syncthreads`) was
  SLOWER at every R (scan 33→46-57ms): redundant q/k reads are L2-cached/cheap, and the per-token
  barrier serializes warps, destroying the latency hiding many independent single-warp blocks give.
- The **chunked-parallel scan (UT-transform / gated-delta rule) is a DEAD-END here.** Math validated
  (CPU prototype `cuda/llm/scan/deltanet_chunk_test.c`, rel_L2 3e-7; CUDA port rel_L2_vs_seq 0.165
  after fixing the `Γ_t/Γ_j` decay ratio to an incremental product — dividing by cumulative decay
  = 0/0 NaN on decay underflow). But **9× SLOWER** (scan 30→279ms): one-block-per-head = only
  `dt_rank=32` blocks (8192 threads) on 36 SMs (~15% occupancy). The chunked form trades the
  original's 4096 independent (h,r) warp recurrences for fewer/larger matmuls, but 32 heads is far
  too small a batch to fill the GPU. Chunked linear-attn needs a large head/batch dim (or big-batch
  TC GEMMs) to win. Reverted from the runner; prototype + analysis retained (commits `3f1df91`, `cd580db`).

### 5. GPU-side MoE dispatch

**Impact:** Remove top-k readback and CPU packing overhead (~2ms directly, more if it enables grouped kernels).  
**Current:** Top-k itself is cheap (~0.8ms), but the selected expert lists are still copied to host and repacked there. GPU-side prefix/count/fill would reduce synchronization and is a prerequisite for a clean grouped expert backend.

### 6. Projection F16 shadows for more quantized weights

**Impact:** Use Tensor Core GEMM for more projections at the cost of VRAM.  
**Current:** The F16 shadow upload helper can dequant non-F32 2D tensors and is already used for several shared/projection paths. Extending this selectively to additional SSM/attention projections may help `gemm=35ms`, but VRAM pressure must be watched on the 16 GB card.

## Key Technical Details

### IQ2_XXS Block Layout (66 bytes per 256 elements)

```
offset 0:  d       (half) — super-block scale
offset 2:  qs[64]  (16-bit words) — packed 2-bit indices + sub-block scales + signs
```

The 2-bit indices are packed into 16-bit words. Each group of 8 words (128 bits = 16 bytes) contains 64 2-bit indices. Sub-block scales and signs are in the high bits of the 16-bit words.

Dequant pseudocode:
```c
float d = __half2float(bp[0:2]);
const uint16_t *qs = (const uint16_t *)(bp + 2);
for (int ib = 0; ib < 8; ib++) {
    uint32_t a0 = qs[4*ib] | (qs[4*ib+1] << 16);  // 64 bits of indices
    uint32_t a1 = qs[4*ib+2] | (qs[4*ib+3] << 16); // scale + sign info
    float db = d * (0.5f + (float)(a1>>28)) * 0.25f;
    for (int l = 0; l < 4; l++) {
        uint64_t gv = iq2xxs_grid_dev[(uint8_t*)&a0[l]];  // grid table lookup
        uint8_t sn = ksigns_iq2xs_dev[(a1 >> (7*l)) & 127];
        for (int j = 0; j < 8; j++) {
            float w = db * (float)(uint8_t)(gv >> (8*j)) * ((sn & (1<<j)) ? -1 : 1);
            // dot product with input element
        }
    }
}
```

### Changelog

- `6cb4c38` — Initial MoE batched prefill: 396 tok/s (cuBLAS gather-scatter)
- `f42bd3d` — Fix per-token fallback removal, shared memory fix: 417 tok/s
- 2026-06-09 continuation — Disabled CUDA graph capture for hybrid/debug/partial forwards, fixed exact prefill replay baseline, fixed AOT shared-gate reduction, disabled incomplete decode shared-expert TC path, verified 21-token exact parity (`rel_L2=0`) and 512-token random prefill at 2534 tok/s.
- 2026-06-09 divergence pass — Fixed forced one-token batched prefill divergence by dispatching `launch_batch_matvec(..., n_tokens=1)` through the decode matvec kernels, matched batched RMSNorm reduction order to decode, and aligned batched IQ2_XXS codebook lookup with the decode kernel. Verified layer-0 stop-after-residual `rel_L2=0.000000`, full layer-0 `rel_L2=0.000112`, normal 21-token parity `rel_L2=0.000000`, and 512-token random prefill at 2526 tok/s.

### Test Harness Features

- `--large-bench N`: Creates N synthetic token IDs (all zeros), runs warm-up + timed prefill
- `--bench`: 100-token prefill with correctness comparison vs sequential
- `-t "prompt"`: Text prompt with tokenization
