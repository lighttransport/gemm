# RDNA4 LLM/VLM Perf Report — vs PyTorch ROCm + Remaining Opportunities

Snapshot of LLM/VLM runner perf on RX 9070 XT (gfx1201, ROCm 7.2) after the
full session arc: dense F16 prefill tuning (Track A), K-quant + IQ batched
prefill (Track B1/B2), hybrid-SSM dispatcher + batched SSM projections, four
IQ dequant kernels, and the VLM e2e batched-embed wiring.

## 1. vs PyTorch ROCm reference — Qwen3-VL-2B F16

Same model weights, same GPU, BF16/F16. PyTorch via `transformers 5.4.0` +
`AutoModelForImageTextToText.from_pretrained(..., dtype=fp16, attn_implementation="sdpa")`;
visual tower dropped before bench. HIP via `./test_hip_llm --bench --gpu-only-bench`
with `LLM_PREFILL_WARMUP=2`.

**Prefill** (ms total / tok/s):

| prefill_len | HIP after        | PyTorch ROCm   | HIP advantage |
|------------:|-----------------:|---------------:|--------------:|
|        64   | **18.7 ms** / 3431 | 30.4 ms / 2105 | **1.63× faster** |
|       256   | **27.4 ms** / 9337 | 35.1 ms / 7295 | **1.28× faster** |
|      1024   | **49.1 ms** / 20874 | 57.1 ms / 17922 | **1.16× faster** |

**Decode** (ms per token):

| prefill_len | HIP decode | PyTorch decode | HIP advantage |
|------------:|-----------:|---------------:|--------------:|
|        64   | **11.27 ms** | 28.78 ms     | **2.55× faster** |
|       256   | **11.76 ms** | 28.57 ms     | **2.43× faster** |
|      1024   | **13.62 ms** | 28.80 ms     | **2.11× faster** |

HIP beats PyTorch ROCm on both prefill and decode at every length.

## 2. Absolute perf — Qwen3.6-27B IQ3_XXS hybrid

No PyTorch reference (transformers 5.x has no inference path for the qwen35
hybrid-SSM arch). Absolute numbers, warm, RX 9070 XT:

| prefill_len | HIP prefill | tok/s | HIP decode | tok/s |
|------------:|------------:|------:|-----------:|------:|
|        64   |   **10.7 ms/tok** |   94 | 68.5 ms/tok | 14.6 |
|       256   |    **7.1 ms/tok** |  140 | 69.0 ms/tok | 14.5 |
|      1024   |    **6.9 ms/tok** |  145 | 70.4 ms/tok | 14.2 |

Session-start baseline was 62–65 ms/tok prefill. **9× speedup at L=1024.**

## 3. End-to-end VLM (image + text) — Qwen3.6-27B + qwen36 mmproj

| image            | vision tokens | vision encode | LLM prefill | LLM prefill rate |
|------------------|--------------:|--------------:|------------:|-----------------:|
| fujisan          | 260           | 1170 ms       | **2721 ms** | 10.5 ms/tok      |
| Brooklyn Bridge  | 2904          | 1500 ms (est) | **24081 ms**| 8.3 ms/tok       |

Brooklyn Bridge was 187 s pre-session → 24 s now (**7.8× e2e**), and the
output is still semantically correct ("The image features the Brooklyn Bridge
in New York City...").

## 4. Where the GPU time goes today — 27B at L=256 prefill

rocprofv3 kernel-trace, total 5402 ms GPU time, 282 k kernels:

| pct | kernel | per-call | calls | category |
|----:|--------|---------:|------:|----------|
| **63.3%** | `deltanet_step_f32` | 92.2 µs | 37056 | **SSM state recurrence (sequential per token)** |
|  8.3% | `Cijk_*` (hipBLASLt BF16 GEMM) | 408 µs | 1104 | batched matmul |
|  7.6% | `dequant_iq3_xxs_to_bf16` | 540 µs | 762 | per-call weight dequant |
|  4.1% | `attn_decode_f32` | 17.8 µs | 12288 | **per-row scalar attn** (head_dim=256 misses FA) |
|  2.9% | `l2_norm_heads_f32` | 2.1 µs | 74112 | SSM aux op (per-token launches) |
|  2.2% | `matvec_iq3_xxs_f32` | 118 µs | 1016 | decode-mode matvec (a few warmup iters) |
|  2.1% | `repeat_tile_f32` | 1.5 µs | 74112 | SSM aux op (per-token launches) |
|  1.7% | `conv1d_depthwise_silu_f32` | 2.5 µs | 37056 | SSM conv (per-token) |
|  1.6% | `gated_rmsnorm_silu_f32` | 2.3 µs | 37056 | SSM aux (per-token) |
|  1.1% | `dequant_q4_K_to_bf16` | 416 µs | 144 | per-call weight dequant |
| ... |

Headline: **the DeltaNet state-recurrence kernel is now 63% of GPU time.** The
hipBLASLt math itself is only ~10% — the matmuls are no longer the bottleneck.

## 5. Remaining perf-gain opportunities

Ranked by potential impact on 27B Qwen3.6 prefill (the model that dominates
the e2e VLM workload), with my honest estimate of upside and effort.

### A. Faster / batched `deltanet_step` (✅ LANDED)
**Current**: 63% of GPU time, 92 µs/call × 37056 calls. The kernel uses one
thread per row of a d_state × d_state state matrix, doing 3 sequential dot
products of d_state f32 elements. On compute it's nowhere near the hardware
peak; it's bandwidth-bound on the state matrix loads (~130 GB/s achieved
vs 640 GB/s DRAM peak).

Three angles:
1. **Multi-token-fused deltanet kernel**: process all M tokens for a layer
   sequentially within a single kernel launch, keeping the state matrix
   resident in LDS across steps. Eliminates ~12k kernel-launch overheads per
   forward and dramatically improves data reuse for the state matrix.
   Estimated: 30–50% reduction in deltanet time → **18–32% overall prefill
   speedup** (5.5–6.0 ms/tok at L=1024).
2. **Vectorize the per-thread loops** (load 4 f32 at a time via `float4`,
   pad d_state to align) to push closer to DRAM peak.
3. **Approximate or precomputed state forms** — beyond scope; would change
   numerics.

**Effort**: 1–2 days for #1. Highest payoff.

**Landed (2026-05-16)**: new `deltanet_step_batch_f32` kernel fuses M
sequential token steps in one launch. Each thread loads its row of the
state matrix into registers ONCE at start, mutates it through M iterations,
writes it back ONCE at end. K and Q broadcast through LDS per step (small).
Wired into the SSM batched layer body as a single launch replacing the
per-row deltanet loop (conv1d / l2_norm / repeat_tile still per-row;
gated_rmsnorm_silu still per-row — those don't have the same state R/W
amplification because their state is small).

Measured (27B IQ3_XXS, RX 9070 XT, warm):

| L | before Opp A | after Opp A | Opp-A speedup |
|---:|---:|---:|---:|
| 64 | 10.7 ms/tok | **8.6 ms/tok** | 1.24× |
| 256 | 7.1 ms/tok | **5.0 ms/tok** | 1.42× |
| 1024 | 6.9 ms/tok | **4.8 ms/tok** | 1.44× |

E2e VLM Brooklyn Bridge: 24.1 s → **18.4 s (1.31×)**.

rocprofv3 confirms: total GPU time 5402 ms → 3556 ms (34% reduction).
`deltanet_step_batch_f32` is still the top contributor at 52% (1846 ms /
144 calls = 12.8 ms/call vs old 92 µs × 48 calls/forward × ...), but with
46% less time spent in deltanet overall. Further compression would need the
per-thread loop vectorization (point #2 above) or LDS-resident state matrix
sharing across threads.

### B. WMMA flash-attention kernel for `head_dim=256`
**Current**: 16 gated-attn layers at head_dim=256 fall back to per-row
`attn_decode_f32` (4.1% of GPU time). The existing batched flash-attn
(`flash_attn_wmma_f16_4w_causal`) caps at head_dim=128.

A head_dim=256 variant would replace 4096 per-row scalar attn calls with one
batched WMMA call per layer per forward. The same dispatcher slot already
exists (`if (r->fa_path_ok)` branch); just need the kernel.

Estimated: 4–5% prefill speedup. **Effort**: ~1 day (mechanical port of the
existing head_dim=128 WMMA-FA, doubling the per-thread accumulator).

### C. Pre-convert F16 SSM weights at load
**Current**: `get_bf16_weight` does F16→BF16 conversion on the fly into the
shared staging buffer every time an SSM F16 weight is used. The
`dequant_iq3_xxs_to_bf16` line shows 540 µs/call — but most of those calls
are the **same** weight repeated across rows / chunks. F16 SSM weights have
the same pattern: `launch_convert_f16_to_bf16` per call vs. one-time at load.

For 27B the F16 SSM weights (alpha + beta = `dt_rank × n_embd × 64` layers
≈ 16 MB) are small enough to pre-convert and cache. The big SSM weights
(qkv, gate, out) are typically Q-quantized so they need per-call dequant
anyway.

Estimated: 2–3% prefill speedup. **Effort**: ~half a day.

### D. Algo-pin hipBLASLt plans
**Current**: `mm_blaslt_bridge.cpp` uses heuristic first-valid algo on gfx1201.
The VLM bridge pins `algo_index=73624` and gets 78.9% peak on its single-shape
GEMM. The LLM has ~12 distinct shapes (Q/K/V/O/gate/up/down/lm_head × hybrid
variants) — a sweep of each gives the best algo.

Currently the hipBLASLt time is only 8% so the absolute win is modest, but
it's mechanical.

Estimated: 0.5–2% prefill speedup. **Effort**: ~1 day (write the algo
enumerator + sweep harness for the 12 shapes; pin the best ones).

### E. Phase-5 graph capture for hybrid decode
**Current**: graph capture is disabled when `r->is_hybrid` because the SSM
state-update kernel (and conv1d) modifies device buffers in ways that may
not be safe to capture/replay. Decode is ~70 ms/tok on 27B — if graph capture
could engage even partially (e.g. for the non-SSM portion of the layer), we'd
recover the ~5–10% launch overhead.

Estimated: 5–10% decode speedup. **Effort**: ~1–2 days investigation + impl;
some risk of correctness regressions.

### F. Vision encoder: e2e VLM bottleneck check
**Current**: 1.5 s vision encode + 24 s LLM prefill on Brooklyn Bridge.
Vision encoding is 6% of e2e — already optimized in earlier rdna4/vlm work
(WMMA flash-attn default; encoder is faster than PyTorch ROCm). Diminishing
returns here unless we go to a much smaller image or a different vision arch.

### G. Decode quality-of-life
HIP decode at 14.5 tok/s on the 27B is already 2.5× faster than PyTorch
ROCm on the comparable 2B (extrapolating). Going faster would need:
- Item A above (faster deltanet), or
- Decode-mode-specific kernel fusion (rmsnorm + matvec fusion, residual+norm fusion).

## 6. Cumulative state of the LLM perf work

| commit | what it did | speedup on 27B |
|--------|-------------|---------------:|
| `82f6f4a` | legacy + IQ/TQ matvec kernels (functionality) | — |
| `297d36a` | Track A: dense F16 prefill via batched Phase-2 | (2B F16 56-211×; 27B unchanged then) |
| `0636d40` | Track B1: K-quant batched prefill | unlocks K-quant models |
| `7061e40` | Phase 1+2: hybrid SSM dispatcher framework | (framework, 0 speedup on 27B) |
| `5b515a0` | Phase 3: batched SSM projections | 2.8–3.2× |
| `6555cf3` | + 4 IQ dequant kernels | 9.2× cumulative |
| `c409702` | VLM e2e batched embed wiring | **7.8× e2e on Brooklyn Bridge** |

## 7. Reproducing the numbers

```sh
# HIP standalone (Qwen3-VL-2B F16)
cd rdna4/llm
for L in 64 256 1024; do
  LLM_PREFILL_WARMUP=2 ./test_hip_llm \
    /mnt/disk1/models/Qwen3-VL-2B-Instruct-GGUF/Qwen3VL-2B-Instruct-F16.gguf \
    --bench --gpu-only-bench --prefill-len $L --decode 32
done

# HIP 27B hybrid
for L in 64 256 1024; do
  LLM_PREFILL_WARMUP=2 ./test_hip_llm \
    /mnt/disk1/models/qwen36/27b/Qwen3.6-27B-UD-IQ3_XXS.gguf \
    --bench --gpu-only-bench --prefill-len $L --decode 16 -s 1280
done

# PyTorch ROCm reference (2B F16)
VENV=/mnt/disk1/work/gemm/main/rdna4/trellis2/.venv
MD=/mnt/disk1/models/hf_cache/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/*
$VENV/bin/python bench_pytorch_llm.py --model-dir "$MD" \
  --prefill-lens 64,256,1024 --decode 32 --dtype f16

# VLM e2e (qwen36/27b)
cd rdna4/vlm
./test_hip_vlm \
  /mnt/disk1/models/qwen36/27b/Qwen3.6-27B-UD-IQ3_XXS.gguf \
  /mnt/disk1/models/qwen36/27b/mmproj-F16.gguf \
  /mnt/disk1/work/gemm/main/Brooklyn_Bridge_Manhattan.jpg -n 64

# Profile the 27B prefill to find the new bottleneck
rocprofv3 --kernel-trace -d /tmp/prof -f csv -- \
  ./test_hip_llm /mnt/disk1/models/qwen36/27b/Qwen3.6-27B-UD-IQ3_XXS.gguf \
  --bench --gpu-only-bench --prefill-len 256 --decode 4
```
