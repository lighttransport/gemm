# RDNA4 LLM Perf — vs PyTorch ROCm + vs Hardware Theoretical Peak

Post Opp-A snapshot. RX 9070 XT (gfx1201, RDNA4), ROCm 7.2.

## 1. Hardware peaks (RX 9070 XT)

| metric | value | notes |
|--------|------:|-------|
| BF16/F16 WMMA compute | **195 TFLOPS** | matmul peak — what GEMMs/FA can hit |
| F32 VALU compute | **~24 TFLOPS** | scalar ops, dependent chains |
| DRAM bandwidth | **640 GB/s** | weight reads, large activation reads/writes |
| LDS bandwidth (per CU) | ~10× DRAM, on-die | broadcast Q/K/V, dequant staging |
| VRAM | 15.9 GB | 27B IQ3_XXS fits with ~3 GB headroom |
| Wavefront / SIMD | 32-lane wavefront, 4 SIMDs / CU × 32 CUs | |

## 2. vs PyTorch ROCm — Qwen3-VL-2B F16 (same model, same GPU)

PyTorch via transformers 5.4.0 SDPA F16; HIP via `test_hip_llm`.

**Prefill** total ms (warm):

| L     | HIP    | PyTorch | HIP advantage | HIP rate         | PyT rate          |
|------:|-------:|--------:|:-------------:|:-----------------|:------------------|
|   64  | 18.94  | 33.17   | **1.75× faster** | 3379 tok/s     | 1930 tok/s       |
|  256  | 27.63  | 35.00   | **1.27× faster** | 9264 tok/s     | 7313 tok/s       |
| 1024  | 49.05  | 56.79   | **1.16× faster** | 20875 tok/s    | 18031 tok/s      |

**Decode** ms/token:

| L     | HIP    | PyTorch | HIP advantage |
|------:|-------:|--------:|:-------------:|
|   64  | 11.25  | 29.21   | **2.60×**     |
|  256  | 11.75  | 28.69   | **2.44×**     |
| 1024  | 13.58  | 28.07   | **2.07×**     |

HIP wins everywhere. PyTorch SDPA is not particularly tuned on RDNA4 yet.

## 3. Theoretical peak analysis

### 3a. 2B F16 prefill @ L=1024

- Model: 1.72 B params F16 = **3.44 GB** weight read
- Compute: 2 × 1.72B × 1024 = **3.52 TFLOPS** per forward
- **Compute floor** (195 TF/s peak): 3.52 / 195 = **18.0 ms**
- **Memory floor** (640 GB/s, weights dominate): 3.44 / 640 = **5.4 ms**
- Bound by **compute** (matmul-heavy at L=1024).

| | ms | % of peak |
|---|---:|---:|
| Theoretical floor | 18.0 | 100% |
| HIP measured      | 49.0 | **37%** |
| PyTorch measured  | 56.8 | 32% |

HIP is at 37% of WMMA peak; the 31 ms gap to peak is ~200 kernel launches × ~5 µs (1 ms), packing F32↔BF16 (~5 ms), rmsnorm/qknorm/RoPE/residual/etc. (~10 ms), and hipBLASLt heuristic algo selection sub-optimal (~remaining 15 ms).

### 3b. 27B IQ3_XXS hybrid prefill @ L=1024

- Model: 27 B params, IQ3_XXS raw ≈ 12 GB
- Compute: 2 × 27B × 1024 = **55 TFLOPS** per forward
- **Compute floor** (195 TF/s peak): 55 / 195 = **283 ms**
- **Memory floor** (weights dominate, ~24 GB incl. dequant traffic): 24 / 640 = **38 ms**
- Bound by **compute**.

| | ms/tok | tok/s | % of peak |
|---|---:|---:|---:|
| Theoretical compute floor | 0.276 | 3620  | 100% |
| HIP measured (post-Opp-A) | 4.79  | 209   | **5.8%** |
| HIP session start         | ~65   | 15    | 0.4% |

**Where the gap is (rocprofv3 on L=1024, GPU time 14.2 s / 3 forwards = 4.74 s/forward):**

| kernel               | ms/forward | % | what it costs |
|---------------------|-----------:|--:|---------------|
| `deltanet_step_batch_f32` | 2543 | **53.7%** | sequential f32 ops on register state (compute-bound) |
| `attn_decode_f32`   |  809 | **17.1%** | per-row scalar attn (head_dim=256 misses FA) |
| `Cijk_*` hipBLASLt   |  394 |   8.3% | batched BF16 GEMMs at **72% of WMMA peak** ✓ |
| dequant IQ3_XXS      |  142 |   3.0% | LUT-based, ~25% of DRAM peak |
| SSM aux ops          |  617 |  ~13% | l2_norm, repeat_tile, conv1d, gated_rmsnorm (per-row launches × M × layers) |

The actual matmul work is humming at 140 TF/s = 72% of WMMA peak — that's not the problem anymore. The problem is **everything but the matmul.**

### 3c. Decode @ 1 token (memory-bound)

- 2B F16 decode floor: 3.44 GB / 640 GB/s = **5.4 ms/tok**
- 2B HIP measured: 13.58 ms/tok = **40% of memory peak**
- 2B PyTorch: 28.07 ms/tok = 19% of memory peak

- 27B IQ3_XXS decode floor: ~10 GB / 640 GB/s = **15.6 ms/tok**
- 27B HIP measured: 69 ms/tok = **23% of memory peak**
- No PyTorch reference (hybrid SSM unsupported in transformers).

## 4. Remaining gain opportunities, ranked by theoretical-peak headroom

The 27B prefill is at **5.8% of compute peak**; the 2B at 37% of compute peak. Where can we actually recover?

### Tier 1 — substantial (>10% prefill)

**B. WMMA flash-attention kernel for `head_dim=256` — currently 17% of 27B GPU time**

The 16 gated-attn layers fall back to `attn_decode_f32` per-row because the existing WMMA-FA caps at head_dim=128. A head_dim=256 variant would batch all M tokens into a single WMMA-FA launch per layer.

Theoretical headroom on 27B: `attn_decode` is 17% of GPU time at L=1024; a WMMA-FA kernel for the same workload would run in ~5% of that (similar to the VLM head_dim=72 case). **Expected: ~16% of total prefill saved → 27B prefill 4.8 → ~4.0 ms/tok.**

**Effort**: ~1 day. Mechanical port of `flash_attn_wmma_f16_4w_causal` doubling per-thread accumulator capacity.

**A2. Vectorize the deltanet inner loop (float4 / SIMD) — currently 54% of 27B GPU time**

The fused deltanet kernel (Opp A) is now compute-bound by 4 × d_state = 512 sequential f32 ops per thread per token. Loading 4 f32 at a time via `float4` (or pairing with HIP's vectorized intrinsics) would let SIMD lanes do 4 ops per cycle instead of 1 for the state row scans (decay multiply, sk dot, update, o dot).

Theoretical headroom: ideal 4× speedup on the inner loops → deltanet drops from 53.7% → ~13% → **~40% of total prefill saved**. In practice will be 2–3× (register pressure / occupancy will limit) → **20–30% of prefill saved** → 27B prefill **4.8 → ~3.4–3.8 ms/tok**.

**Effort**: 1–2 days; some risk in getting the float4 load/store layout right; need to keep the per-step LDS K/Q broadcast vectorized too.

### Tier 2 — modest (3–10%)

**C. Pre-convert F16 SSM weights at load (currently F16→BF16 on every call)**

Small F16 SSM weights (alpha, beta — ~16 MB on 27B) are dequant'd to BF16 into the staging buffer on every call. Pre-converting at load eliminates that.

Theoretical headroom: <2% of GPU time today, but it's free at runtime once paid at load. **~1–2% prefill saved.**

**Effort**: ~half a day.

**SSM aux op batching — currently ~13% of 27B GPU time**

`l2_norm_heads`, `repeat_tile`, `conv1d_depthwise_silu`, `gated_rmsnorm_silu` all run in a per-row host loop (M × 48 layers × 4 ops = ~200 k launches per L=1024 forward). Each launch is ~2 µs, the work is microscopic. A batched form of each kernel would collapse those into ~48 launches per layer.

Theoretical headroom: 13% × ~50% removable (the kernels themselves are fast, just launch-bound) ≈ **6% prefill saved**.

**Effort**: ~1 day.

### Tier 3 — small (<3%)

**D. hipBLASLt algo pinning** — GEMM already at 72% of WMMA peak; algo pinning might extract another 5–10% on each shape → **0.5–1% prefill saved.**

**Faster IQ3_XXS dequant** (currently 25% of DRAM peak due to LUT indexing) — close to L1 cache hit if the grid/sign tables stay hot. **1–2% prefill saved.**

### Tier 4 — decode-side

**E. Phase-5 graph capture for hybrid decode** — currently disabled because of SSM state. Per-token decode at 69 ms/tok is ~23% of memory peak; with graph capture removing ~5–10% launch overhead → **decode 69 → 62–65 ms/tok.**

**Effort**: 1–2 days investigation + impl; some correctness risk.

## 5. Realistic stacked-improvement ceiling

If we land B + A2 + SSM-aux-batching + dequant-tuning, with realistic (not theoretical) recovery:

| from | to | 27B prefill @L=1024 | cumulative speedup |
|---|---|---:|---:|
| post-Opp-A (today) | — | 4.79 ms/tok | (1.0×) |
| + B (head_dim=256 WMMA FA) | -16% | ~4.0 ms/tok | 1.20× |
| + A2 (deltanet float4) | -25% | ~3.0 ms/tok | 1.60× |
| + SSM aux batching | -6% | ~2.8 ms/tok | 1.70× |
| + dequant tuning | -2% | ~2.7 ms/tok | 1.78× |

That'd put e2e VLM Brooklyn Bridge from 18.4 s → **~10.4 s** (vs 187 s at session start).

Theoretical compute floor: 0.28 ms/tok at 195 TF/s WMMA peak. We'd be at **~9.6% of peak** vs current 5.8%. The remaining 90% is the cost of sequential SSM compute, attention with state, kernel launches, and dequant overhead — not GEMM. Past that point, RDNA4 is hard-limited by the model's intrinsic sequentiality (DeltaNet recurrence).

## 6. Reproducibility

All numbers re-runnable with the recipes in `rdna4/llm/perf-report.md` §7.
HIP measurements: warm (`LLM_PREFILL_WARMUP=2`), `--bench --gpu-only-bench`.
PyTorch: `rdna4/llm/bench_pytorch_llm.py --dtype f16 --warmup 2 --iters 5`.
Profile: `rocprofv3 --kernel-trace -d <dir> -f csv -- <cmd>`.
