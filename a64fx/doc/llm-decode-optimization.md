# A64FX LLM Decode Optimization Strategy

Target model: Qwen3.5-9B BF16/FP16 on A64FX (4 CMGs, 48 cores, 28GB HBM2)

## A64FX Hardware Summary

| Resource | Per Core | Per CMG (12 cores) | Node (4 CMGs) |
|----------|---------|-------------------|----------------|
| FP32 FMA | 64 GFLOP/s | 768 GFLOP/s | 3.07 TFLOP/s |
| FP16 FMA | 128 GFLOP/s | 1.54 TFLOP/s | 6.14 TFLOP/s |
| HBM2 read BW | 32 GB/s | 222 GB/s | 833 GB/s |
| L2 cache | — | 8 MB shared | 32 MB |
| HBM2 capacity | — | 7 GB | 28 GB |
| SVE width | 512-bit | — | — |

## Qwen3.5-9B Architecture

| Parameter | Value |
|-----------|-------|
| n_embd | 4096 |
| n_heads (Q) | 16 |
| n_kv_heads | 4 |
| head_dim | 256 |
| GQA ratio | 4 (4 Q heads per KV head) |
| n_ff | 12288 |
| n_layers | 32 (24 SSM + 8 attention) |
| n_vocab | 248320 |
| Hybrid | SSM (Delta-Net) + Attention, full_attn_interval=4 |

## Decode: Two Distinct Memory-Bound Workloads

### 1. Weight Matvec (FFN + projections)

Computes `dst[M] = W[M,K] × x[K]` for each layer's weight matrices.

| Property | Value |
|----------|-------|
| Total weight data | 13.4 GB (BF16) |
| Arithmetic intensity | 1 FLOP/byte (always memory-bound) |
| Achieved BW (BF16, 32T) | 417 GB/s |
| Achieved BW (FP16 4r×2vl, 48T, mbind) | **756 GB/s** (91% of 833 peak) |
| Per-core BF16 kernel | 18.0 GB/s (56% of 32 peak) |
| Per-core FP16 kernel | 27.8 GB/s (87% of 32 peak) |
| Time per token (weight-only) | 13.4 GB / 756 GB/s = **17.7 ms** |

**Key optimizations:**
- FP16 4-row × 2-vector-length kernel: `svld1_f16` + `svmla_f16`, 32 FP16/SVE vec
- No software prefetch (A64FX HW prefetcher is superior for sequential patterns)
- Per-CMG weight pools via `mmap` + `mbind` (zero cross-CMG traffic)
- `pthread_setaffinity_np` to pin threads to correct CMG cores
- WFE/SEV barrier for intra-CMG sync (zero bandwidth waste)
- Persistent forward: threads run full layer loop, no dispatch overhead

### 2. Attention (KV Cache Access)

Computes QK dot products and AV weighted sums over cached key/value vectors.

| Property | Value |
|----------|-------|
| Arithmetic intensity | 4 FLOP/byte (FP16 KV, GQA fused) |
| A64FX balance point | 8.2 FLOP/byte (FP16 compute / HBM BW) |
| Classification | Memory-bound in theory, compute-limited in practice |
| Bottleneck | `svaddv` horizontal reduction (64 cycles per reduction) |
| Achieved BW (QK, 2pos×4head, 40T) | 168 GB/s |
| Per-core effective | 4.2 GB/s |

**Why attention is slower than weight matvec:**

Weight matvec: outer loop over thousands of rows, inner loop accumulates into SVE registers, one `svaddv` per row. The reduction cost is amortized over the entire inner loop.

Attention QK: outer loop over positions, inner loop over head_dim (256 elements = 8 SVE iterations for FP16), then `svaddv` per position per head. With 4 fused GQA heads: 8 `svaddv` reductions per 2-position batch = 512 cycles overhead vs ~192 cycles of FMA compute. The reduction dominates.

## KV Cache Memory

KV cache per position: `n_kv_heads × head_dim × 2 (K+V) × 2 bytes (FP16) = 4 KB`

| Context | Per Layer | 8 Attn Layers | + Weights | Total | Fits 28GB? |
|---------|----------|--------------|-----------|-------|------------|
| 4K | 16 MB | 128 MB | 13.4 GB | 13.5 GB | Yes |
| 32K | 128 MB | 1.0 GB | 13.4 GB | 14.4 GB | Yes |
| 128K | 512 MB | 4.0 GB | 13.4 GB | 17.4 GB | Yes |
| 256K | 1.0 GB | 8.0 GB | 13.4 GB | 21.4 GB | Yes |
| **460K** | 1.8 GB | 14.6 GB | 13.4 GB | **28.0 GB** | **Max** |
| 512K | 2.0 GB | 16.0 GB | 13.4 GB | 29.4 GB | No |
| 1M | 4.0 GB | 32.0 GB | 13.4 GB | 45.4 GB | No |

**Per-CMG budget** (7 GB each):

| Context | Weights/CMG | KV/CMG | Total | Headroom |
|---------|------------|--------|-------|----------|
| 128K | 3.35 GB | 1.0 GB | 4.35 GB | 2.65 GB |
| 256K | 3.35 GB | 2.0 GB | 5.35 GB | 1.65 GB |
| 460K | 3.35 GB | 3.65 GB | 7.0 GB | 0 |

## Estimated Decode Latency

### Token time breakdown by context length

| Component | 4K | 32K | 128K | 256K |
|-----------|-----|------|------|------|
| Weight matvec (756 GB/s) | 17.7 ms | 17.7 ms | 17.7 ms | 17.7 ms |
| Attention QK (168 GB/s) | 0.2 ms | 1.5 ms | 6.1 ms | 12.2 ms |
| Attention AV (~168 GB/s) | 0.2 ms | 1.5 ms | 6.1 ms | 12.2 ms |
| SSM layers (sequential) | ~2 ms | ~2 ms | ~2 ms | ~2 ms |
| Barriers + overhead | ~1 ms | ~1 ms | ~1 ms | ~1 ms |
| **Total** | **~21 ms** | **~24 ms** | **~33 ms** | **~45 ms** |
| **Tokens/sec** | **~48** | **~42** | **~30** | **~22** |

Weight matvec dominates at short context. Attention becomes significant at 128K+.

### FLOP/s utilization

| Workload | GB/s | FLOP/byte | GFLOP/s | % of 6.14 TFLOP/s peak |
|----------|------|-----------|---------|----------------------|
| Weight matvec (FP16) | 756 | 1 | 756 | 12.3% |
| Attention QK (FP16) | 168 | 4 | 672 | 10.9% |
| Combined (short ctx) | — | — | ~800 | 13.0% |
| Combined (128K ctx) | — | — | ~1100 | 17.9% |

The workloads are memory-bound; FLOP utilization is inherently low.

## Recommended Configuration

### Optimal thread/CMG mapping

```
48 threads total: 12 per CMG
CMG 0 (cores 12-23, node 4): KV head 0, Q heads 0-3, weight partition 0
CMG 1 (cores 24-35, node 5): KV head 1, Q heads 4-7, weight partition 1
CMG 2 (cores 36-47, node 6): KV head 2, Q heads 8-11, weight partition 2
CMG 3 (cores 48-59, node 7): KV head 3, Q heads 12-15, weight partition 3
```

### Memory placement

```c
// Per-CMG weight pool (mbind to NUMA node)
void *pool = mmap(NULL, size, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
unsigned long mask = 1UL << (4 + cmg_id);
mbind(pool, size, MPOL_BIND, &mask, 64, 0);
```

### KV cache layout (recommended)

```
Current: key_cache[layer][pos * kv_dim + kv_h * head_dim + d]  FP32
    → stride 4096 bytes between positions, 25% utilization per head

Recommended: key_cache_fp16[layer][kv_h][pos * head_dim + d]  FP16
    → stride 512 bytes, sequential within head, mbind per CMG
    → 2× memory savings, ideal for HW prefetcher
```

### Launch command

```bash
XOS_MMM_L_PAGING_POLICY=demand:demand:demand \
numactl -C12-23,24-35,36-47,48-59 -m4-7 \
./test_transformer model.gguf "prompt" 1024 48
```

## Kernel Summary

| Kernel | Layout | Ops/iter | Per-core GB/s | Bottleneck |
|--------|--------|----------|---------------|------------|
| Weight BF16 8r×1vl | `ld1uh+lsl+fmla32` | ld+shift+fma | 18.0 | lsl shift |
| Weight FP16 4r×2vl | `ld1h+fmla16` | ld+fma | **27.8** | HBM BW |
| Weight FP32 (ceil) | `ld1w+fmla32` | ld+fma | 24.9 | HBM BW |
| Attn QK 1pos×4head | `ld1h+fmla16+svaddv` | 4 ld+16 fma+4 reduce | 3.2 | svaddv |
| Attn QK 2pos×4head | `ld1h+fmla16+svaddv` | 6 ld+32 fma+8 reduce | **4.2** | svaddv |
| HBM2 streaming read | `ld1` only | ld | 32.0 | HBM peak |

## Future Optimization Paths

1. **Quantized KV cache (INT8):** Halves KV memory, doubles context capacity. 460K → 920K max context.

2. **Linear attention / SSM-only:** Qwen3.5's hybrid architecture has 24 SSM layers with O(1) state — only 8 attention layers need KV cache. Already favorable vs pure-attention models.

3. **FlashAttention-style tiling:** Fuse QK softmax AV into one pass over KV, reducing memory traffic by avoiding materialized score matrix. Most beneficial for long contexts.

4. **Multi-query attention (MQA):** n_kv_heads=1 instead of 4 would further reduce KV cache 4×. Not applicable to Qwen3.5 but relevant for model selection.

5. **Hardware BF16 (ARMv8.6+):** BFMMLA/BFDOT instructions would eliminate the `lsl` shift, bringing BF16 to FP16-level bandwidth. Not available on A64FX (ARMv8.2).

6. **Speculative decoding:** Run small draft model on 1 CMG, verify on all 4 CMGs. Potential 2-3× effective throughput.
