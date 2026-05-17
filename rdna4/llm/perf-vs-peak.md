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

**B. WMMA flash-attention kernel for `head_dim=256` (✅ LANDED)**

The 16 gated-attn layers fall back to `attn_decode_f32` per-row because the existing WMMA-FA caps at head_dim=128. A head_dim=256 variant would batch all M tokens into a single WMMA-FA launch per layer.

Theoretical headroom on 27B: `attn_decode` is 17% of GPU time at L=1024; a WMMA-FA kernel for the same workload would run in ~5% of that (similar to the VLM head_dim=72 case). **Expected: ~16% of total prefill saved → 27B prefill 4.8 → ~4.0 ms/tok.**

**Effort**: ~1 day. Mechanical port of `flash_attn_wmma_f16_4w_causal` doubling per-thread accumulator capacity.

**Landed (2026-05-16)**: new `flash_attn_wmma_f16_4w_causal_hd256` kernel
(HD_MAX=256, K_NB=16; cooperative load partitioned 16 rows × 8 col-blocks
of 32 cols/thread). Phase-3 FA eligibility check extended from
`head_dim <= 128` to `head_dim <= 256`. Launcher picks the right variant
by `head_dim`.

Measured (27B IQ3_XXS hybrid, RX 9070 XT, warm):

| L | before B | after B | speedup |
|---:|---:|---:|---:|
| 64 | 8.68 | **8.43 ms/tok** | 1.03× |
| 256 | 5.00 | **4.71 ms/tok** | 1.06× |
| 1024 | 4.79 | **4.03 ms/tok** | **1.19×** |

E2e VLM Brooklyn Bridge: 18.4 s → **12.7 s** = **1.44× from Opp B alone**
(amortization across M=1024 chunks gives bigger gains than standalone
bench). Cumulative session-arc on Brooklyn Bridge: 187.6 s → 12.7 s = **14.8×**.

2B F16 unchanged (head_dim=128 keeps original kernel). Correctness:
`--compare-paths` argmax match on 27B (token 369).

**A2. Vectorize the deltanet inner loop (✅ LANDED — 2.12× on its own)**

The fused deltanet kernel (Opp A) is now compute-bound by 4 × d_state = 512 sequential f32 ops per thread per token. Loading 4 f32 at a time via `float4` (or pairing with HIP's vectorized intrinsics) would let SIMD lanes do 4 ops per cycle instead of 1 for the state row scans (decay multiply, sk dot, update, o dot).

Theoretical headroom: ideal 4× speedup on the inner loops → deltanet drops from 53.7% → ~13% → **~40% of total prefill saved**. In practice will be 2–3× (register pressure / occupancy will limit) → **20–30% of prefill saved** → 27B prefill **4.8 → ~3.4–3.8 ms/tok**.

**Effort**: 1–2 days; some risk in getting the float4 load/store layout right; need to keep the per-step LDS K/Q broadcast vectorized too.

**Landed (2026-05-16)**: in-place rewrite of `deltanet_step_batch_f32`.
Two changes: (1) collapse the four serial loops (decay → sk-dot → update →
o-dot) into two passes — pass 1 computes `sk = decay * sum(S * sK)` (decay
factored out of the sum so pass 1 doesn't touch S_row); pass 2 fuses
`S_new = S*decay + delta*sK` with the o-dot in one read+write per element.
(2) Each dot product uses **4 unrolled accumulators** so the SIMD lane can
issue 4 FMAs back-to-back instead of being bottlenecked by the ~4-cycle FMA
latency on a single accumulator. Inner loops unrolled 8× via `#pragma`.

Measured (27B IQ3_XXS hybrid, RX 9070 XT, warm):

| L | post-B | post-A2 | A2 speedup |
|---:|---:|---:|---:|
| 64  | 8.43 ms/tok | **6.25 ms/tok** | 1.35× |
| 256 | 4.71 ms/tok | **2.60 ms/tok** | 1.81× |
| 1024| 4.03 ms/tok | **1.90 ms/tok** | **2.12×** |

E2e VLM Brooklyn Bridge: 12.7 s → **6.5 s** = **1.95×** from A2. Cumulative
session arc: **187.6 s → 6.5 s = 28.8×**.

rocprofv3 (L=1024, GPU 5220 ms total):
- deltanet_step_batch: 7630 ms → **791 ms** (**9.6× on the kernel itself**,
  53.7% → 15.2% of GPU time)
- hipBLASLt GEMMs are now the top contributor at 22.5% (the real matmul work)
- Per-row SSM aux ops collectively ~40% (l2_norm 13.9%, repeat_tile 10.4%,
  conv1d 8.5%, gated_rmsnorm 7.7%) — the **new** bottleneck class

% of WMMA peak: 0.28/1.90 = **14.7%** (up from 5.8% before A2).

### Tier 2 — modest (3–10%)

**C. Pre-convert F16 SSM weights at load (currently F16→BF16 on every call)**

Small F16 SSM weights (alpha, beta — ~16 MB on 27B) are dequant'd to BF16 into the staging buffer on every call. Pre-converting at load eliminates that.

Theoretical headroom: <2% of GPU time today, but it's free at runtime once paid at load. **~1–2% prefill saved.**

**Effort**: ~half a day.

**SSM aux op batching (✅ LANDED — 2.00× at L=1024)**

`l2_norm_heads`, `repeat_tile`, `conv1d_depthwise_silu`, `gated_rmsnorm_silu` all run in a per-row host loop (M × 48 layers × 4 ops = ~200 k launches per L=1024 forward). Each launch is ~2 µs, the work is microscopic. A batched form of each kernel would collapse those into ~48 launches per layer.

Theoretical headroom: 13% × ~50% removable (the kernels themselves are fast, just launch-bound) ≈ **6% prefill saved**.

**Effort**: ~1 day.

**Landed (2026-05-16)**: 4 new batched kernels:
- `l2_norm_heads_batch_f32` — grid (n_heads, M, 1); one block per (m, h)
- `repeat_tile_batch_f32` — flat thread per (m, h, i)
- `gated_rmsnorm_silu_batch_f32` — grid (dt_rank, M, 1)
- `conv1d_depthwise_silu_batch_f32` — one thread per column j, M sequential
  steps fused inside the kernel (state in registers, same pattern as
  deltanet_batch)

Per SSM layer: **~6×M per-row launches → 6 batched launches** (~170× launch-count
reduction for L=1024).

Measured (27B IQ3_XXS hybrid, RX 9070 XT, warm):

| L | post-A2 | post-aux | aux speedup | session total |
|---:|---:|---:|---:|---:|
| 64  | 6.25 ms/tok | **5.37 ms/tok** | 1.16× | **11.5×** |
| 256 | 2.60 ms/tok | **1.69 ms/tok** | 1.54× | ~38× |
| 1024| 1.90 ms/tok | **0.95 ms/tok** | **2.00×** | **~68×** |

**Sub-millisecond per token on 27B IQ3_XXS hybrid (1051 tok/s prefill).**

E2e VLM Brooklyn Bridge: 6520 → **3768 ms** = **1.73× from aux batching**.
Cumulative session arc: **187.6 s → 3.77 s = 49.8×**.

rocprofv3 confirms: aux ops collectively dropped from ~40% → ~3% of GPU time.
hipBLASLt GEMMs (the real matmul) are now the dominant single contributor at
37.5%. % of WMMA peak: **29.5%** (up from 14.7%).

### Tier 3 — small (<3%)

**D. hipBLASLt algo pinning (❌ TRIED — blocked by gfx1201 driver bug)** — GEMM already at 72% of WMMA peak; algo pinning might extract another 5–10% on each shape → **0.5–1% prefill saved.**

Attempted 2026-05-17: implemented a per-plan top-N algo sweep in
`mm_blaslt_bridge.cpp` (timed each candidate algo on dummy buffers with
`hipDeviceSynchronize` between, picked the fastest), gated by an
`mm_blaslt_set_sweep()` API and scoped to fire only around the init-time
pre-warm (so user-facing forwards never trigger sweep-driven plan builds).
While debugging, also fixed two real pre-warm bugs (used `layers[0]`
which is SSM with NULL attn weights on hybrid models; only handled
pre-converted F16 weights). With those fixes the pre-warm fired real-shape
GEMMs through the sweep — and **crashed mid-pre-warm** on the 27B with
the exact failure mode the bridge author had warned about in the existing
code comment: "back-to-back matmul calls without intermediate work
segfault inside hipBLASLt's matmul setup". Full `hipDeviceSynchronize`
between candidates did not help.

All sweep-related changes reverted (`hip_llm_runner.c`,
`mm_blaslt_bridge.cpp`, `mm_blaslt_bridge.h`) to the post-aux-batch
commit. Verified 27B regression: L=256 prefill = 1.69 ms/tok (matches
pre-attempt). The 5% gain isn't currently reachable without:
- an offline sweep tool that runs each shape in a fresh process and
  hardcodes the best algo_index per (M, N, K) into the bridge, or
- a patched hipBLASLt build, or
- a non-hipBLASLt GEMM path (custom WMMA tiles like `rdna4/vlm/mm0_*`).

None of those are worth the engineering effort at this point — hipBLASLt
heuristic already gets us to 72% of WMMA peak, and the model's
intrinsic sequentiality (DeltaNet recurrence + dequant + per-row attn
fallback for head_dim=256) bounds further total-time gains.

**Faster IQ3_XXS dequant** (currently 25% of DRAM peak due to LUT indexing) — close to L1 cache hit if the grid/sign tables stay hot. **1–2% prefill saved.**

### Tier 4 — decode-side

**E. Phase-5 graph capture for hybrid decode (✅ LANDED 2026-05-17 — correctness improvement, perf-null)** — was disabled because of SSM state. Audited per-token kernels (`decode-graph-capture-audit.md`); all SSM aux ops mutate state in-place via fixed device pointers, gated-attn already uses `*_devp` launchers. Relaxed the `!r->is_hybrid` gate in `hip_llm_phase5_capture` and added a `hip_llm_reset_state` call after capture to scrub the warm-pass state contamination. MoE stays gated (host-side router top-K read can't be captured).

Measured on RX 9070 XT:

| model | graph ON | graph OFF | delta |
|---|---:|---:|---:|
| Qwen3.5-9B Q4_K_XL hybrid | 45.10 ms/tok | 44.54 ms/tok | +1.3% (slower) |
| Qwen3.6-27B IQ3_XXS hybrid | 69.54 ms/tok | 69.41 ms/tok | +0.2% (slower) |

Both bit-identical first/last decoded token ids → correctness PASS. **Perf-null** because hybrid decode is kernel-execution-bound, not launch-bound. Theoretical launch-overhead ceiling was ~1 ms/tok (200 kernels × 5 µs) ≈ 1.4% of 70 ms/tok, and the driver's existing stream-queue scheduling already amortizes most of that — leaving the captured-graph path effectively a wash. Kept landed because it removes a stale exclusion and brings hybrid to parity with non-hybrid graph eligibility.

**Q5_K full-vocab decode matvec (✅ LANDED 2026-05-18)** — the short decode
profile showed `matvec_q5_K_f32` as a disproportionate hotspot despite only
9 calls: 63.7 ms total, ~7.1 ms/call. This is the full-vocab logits
projection, so the default one-row-per-block kernel produced too many tiny
blocks and too little work per block. Added `matvec_q5_K_mw_f32`, a
one-warp-per-row variant with 8 rows per block, and made it the default
Q5_K decode path. `LLM_Q5_K_MW=0` keeps the old kernel available for A/B
and rollback.

Measured on RX 9070 XT:

| case | old Q5_K | new Q5_K MW | speedup |
|---|---:|---:|---:|
| microbench `Q5_K 248320x5120` | 7.096 ms/launch | **1.787 ms/launch** | **3.97x** |
| Qwen3.6-27B IQ3_XXS decode L=256, 16 tok | 67.785 ms/tok | **63.404 ms/tok** | **1.07x** |
| Qwen3.5-9B Q4_K_XL decode L=256, 16 tok | 45.179 ms/tok | **36.372 ms/tok** | **1.24x** |

Correctness: `--verify-quant-kernels` passes 18/18 with the new default, and
the 9B/27B A/B decode runs preserved first/last token ids.

**IQ3_XXS decode matvec scale/sign load (✅ LANDED 2026-05-18)** — replaced
the hot-kernel `memcpy` load of each IQ3_XXS scale/sign word with an explicit
32-bit load. This is a narrow decode-kernel-internals cleanup, but it hits the
largest remaining matvec family on the 27B path. Applied the same direct-load
cleanup to `dequant_iq3_xxs_to_bf16` after validating it independently.

Measured on RX 9070 XT:

| case | before | after | speedup |
|---|---:|---:|---:|
| microbench `IQ3_XXS 5120x5120` | 0.0538 ms | **0.0531 ms** | 1.01x |
| microbench `IQ3_XXS 17408x5120` | 0.1557 ms | **0.1528 ms** | 1.02x |
| microbench `IQ3_XXS 5120x17408` | 0.1617 ms | **0.1586 ms** | 1.02x |
| profile `dequant_iq3_xxs_to_bf16` total | 137.4 ms | **135.9 ms** | 1.01x |
| Qwen3.6-27B IQ3_XXS decode L=256, 16 tok | 63.3-63.6 ms/tok | **62.8-62.9 ms/tok** | ~1.01x |

Correctness: `--verify-quant-kernels` passes 18/18; 27B decoded first/last
token ids unchanged. The dequant direct-load cleanup kept the same token ids
and measured 5.854/6.004 ms/tok on two L=256 prefill runs. A related
`dequant_iq3_xxs_to_bf16` warp-broadcast trial was not landed: it passed
correctness but regressed the same 27B prefill bench from 6.083 to
6.176 ms/tok.

**IQ3_XXS decode matvec loop unroll (✅ LANDED 2026-05-18)** — added explicit
unroll hints to the fixed 8 x 4 x 4 IQ3_XXS decode-matvec block loops. The
compiler had not fully flattened the loop nest, and the hint materially
reduced the hottest decode kernel without changing the arithmetic.

Measured on RX 9070 XT:

| case | before | after | speedup |
|---|---:|---:|---:|
| microbench `IQ3_XXS 5120x5120` | 0.0531 ms | **0.0465 ms** | 1.14x |
| microbench `IQ3_XXS 17408x5120` | 0.1528 ms | **0.1372 ms** | 1.11x |
| microbench `IQ3_XXS 5120x17408` | 0.1586 ms | **0.1345 ms** | 1.18x |
| profile `matvec_iq3_xxs_f32` total | 253.3 ms | **226.0 ms** | 1.12x |
| Qwen3.6-27B IQ3_XXS decode L=256, 16 tok | 62.8-62.9 ms/tok | **57.78 ms/tok** | 1.09x |

Correctness: `--verify-quant-kernels` passes 18/18; 27B decoded first/last
token ids unchanged.

**Q6_K one-warp-per-row decode matvec (❌ TRIED 2026-05-18 — not landed)** —
implemented the analogous MW launch shape behind `LLM_Q6_K_MW=1`. It passed
`--verify-quant-kernels` and improved isolated Q6_K microbenches:
5120x5120 0.208 → 0.192 ms/launch, 17408x5120 0.635 → 0.517 ms/launch,
5120x17408 1.059 → 0.964 ms/launch. End-to-end 27B decode still regressed:
63.615 ms/tok default vs 64.850 ms/tok with Q6_K MW at L=256, 16 decode
tokens, same first/last token ids. Removed the experiment instead of keeping
an opt-in path with misleading microbench results.

**Decode fusion path (rmsnorm+matvec, matvec+residual, qknorm+RoPE+KV) — abandoned 2026-05-17.** Original Phase-2 plan was ~10–15% combined; after the Phase-1 null re-derivation showed the real ceiling at ~3% (launches + input-side VRAM round-trips are negligible vs the ~12 GB/tok weight-read budget). Real decode headroom is **inside the matvec kernels themselves** — per-matvec is at ~60% of memory peak per call, so ~40% × (matvec-fraction-of-time ≈ 50%) ≈ **~20% potential** from kernel internals (wider vectorized quant-weight reads, LDS pipelining, possibly WMMA-decode for quant types). That's a different project — kernel rewrite, not fusion — and is uncertain enough to be scoped separately when next picked up.

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
