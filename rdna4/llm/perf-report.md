# RDNA4 LLM/VLM Perf Report — vs PyTorch ROCm + Remaining Opportunities

Snapshot of LLM/VLM runner perf on RX 9070 XT (gfx1201, ROCm 7.2) after the
full session arc: dense F16 prefill tuning (Track A), K-quant + IQ batched
prefill (Track B1/B2), hybrid-SSM dispatcher + batched SSM projections, four
IQ dequant kernels, and the VLM e2e batched-embed wiring.

> **2026-06-10 update:** sections 1–7 below predate the Qwen3.6-35B-A3B
> vs-llama.cpp campaign. See **section 8** for the current state — the runner
> now BEATS llama.cpp ROCm on every metric (pp512 1055/902, pp1024 1536/883,
> tg128 128.5/83) with a fully self-owned GEMM (no hipBLASLt). Several
> "remaining opportunities" below have since landed in much stronger form
> (Opp A superseded by warp-per-row deltanet; Opp E superseded by full hybrid
> graph capture; Opp C landed for SSM weights; head_dim=256 FA landed).
> Benchmark table: `qwen36_35b_vs_llamacpp.md`.

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

## 8. Qwen3.6-35B-A3B campaign — beating llama.cpp ROCm (2026-06-10)

Target model: `Qwen3.6-35B-A3B-UD-IQ3_S.gguf` (12.73 GiB, arch `qwen35moe`:
30 Delta-Net SSM + 10 gated-attn layers, 256-expert/8-used MoE on all 40,
n_embd 2048, head_dim 256, vocab 248 320). Reference: llama.cpp ROCm
`build_rocm722_rdna4_fa`, build 549b9d843, `-fa on`. Same GPU.

### 8.1 Final numbers

| metric | session start | final | llama.cpp | margin |
|--------|--------------:|------:|----------:|-------:|
| prefill pp512  | 31.2 t/s | **1055 t/s** | 902 t/s | **1.17×** |
| prefill pp1024 | 30.9 t/s | **1536 t/s** | 883 t/s | **1.74×** |
| decode tg128   | 28.7 t/s | **128.5 t/s** | 83.0 t/s | **1.55×** |
| vision prefill (672 tok) | 33.9 ms/tok | **1.22 ms/tok** | — | 28× |

Cumulative: prefill 50×, decode 4.5×. Every step bit-exact
(`--verify-quant-kernels` 18/18) and output-preserving (greedy tokens stable;
VLM still identifies Mt. Fuji on fujisan_1024.png). Defaults all-on; the
`HIPBLASLT=0` build runs the identical batched path on the self-owned GEMM.

### 8.2 Roofline (why 128.5 is ~61% of practical peak)

Weights+state read per decoded token: **2.15 GB** = LM head Q6_K 417 MB +
30 SSM layers × 41.5 MB + 10 attn layers × 36 MB + SSM state R/W 126 MB.
At 600 GB/s board peak → 280 t/s; at realistic ~75% stream efficiency →
**~210 t/s**. Final decode kernel sum 7.1 ms/tok + ~0.7 ms graph replay
overhead. `ssm_matvec4_q6k` runs at 505 GB/s (at BW); Q6_K class near BW;
the residual is IQ2_S/IQ3_S grid-dequant (compute-bound, ~1.8 ms), router
top-K serialization (0.66 ms), and ~1.1 ms of small ops.

### 8.3 Decode wins, in order (29 → 128.5 tok/s)

| tok/s | change |
|------:|--------|
| 29→36 | GPU-side sync-free MoE dispatch (device top-K/softmax/sigmoid; expert-indexed matvecs) + HIP graph capture for hybrid MoE decode |
| 36→45.5 | full warp/thread utilization in IQ2_S/IQ3_S/Q6_K matvecs (old kernels strided K by n_blocks; 6–25% lanes active on nb=2–8 shapes) |
| 45→73 | fused all-expert decode MoE: gate+up+SiLU for all 8 experts in one launch (`blockIdx.y`=slot), down+weighted-accum in one |
| 73→75 | fused SSM aux chain `ssm_prep_f32` (conv+SiLU+state, L2 norm, repeat-tile, softplus, sigmoid: 7→1) |
| 75→78 | fused SSM 4-matvec `ssm_matvec4_q6k` (qkv+z+alpha+beta) + attn q/k/v fusion + residual+rmsnorm |
| 78→83 | shared expert folded as slot K of the MoE kernels (Q6_K branch) + accum-zero in slot 0 |
| 83→87 | `moe_router_fused`: all 256 router logits + shared-gate sigmoid in one grid; **last-finished block** (atomic+threadfence) runs top-K inline. MoE layer = 3 launches |
| 87→102 | **warp-per-row deltanet**: was 32 blk × 128 thr (1 wave/CU, 68 GB/s, 4 serial passes); now 4 lanes/row × 32 cols in regs, single fused pass, float4 IO, shfl reductions |
| 102→115 | `q6k_dot4`: vectorized Q6_K dot (dword ql/qh via memcpy on 2-aligned blocks + float4 x) in all six Q6_K kernels |
| 115→116 | bf16 router weights (84→42 MB/tok; revealed router is latency- not BW-bound) |
| 116→127 | IQ4_XS dtype slot in fused down — the 3 IQ4_XS layers had been ~40 per-expert launches each |
| 127→128.5 | cross-layer fusion: MoE residual add folded into next layer's pre-attn rmsnorm |

Decode is **launch-count bound, not bandwidth bound** (LM head was already at
370 GB/s before any of this). Every win = fewer, fatter launches.

### 8.4 Prefill wins (31 → 1536 tok/s @1024)

1. Decode kernels carried per-token prefill 31→48 free.
2. Fused quantized MoE GEMM (`mmq_iq2s/iq3s`) + the dispatch fix (MoE-aware
   eligibility; F32 SSM weights batchable): 48→400. Tensile fails at M>256 →
   chunked.
3. Self-owned 128×128 WMMA bf16 GEMM (`gemm_bf16_own`, 256 thr) — beats
   Tensile at chunk 256, removes the M>256 cap → chunk 1024: 400→445.
4. Grouped all-expert dequant + grouped GEMM (`blockIdx.z`=expert) replaced
   scalar mmq (65% of prefill): 445→772.
5. GPU top-K/histogram/scatter (no per-layer host sync): 772→1001.
6. Vectorized 16B GEMM tile fills: 1001→1285.
7. Warp-per-row batch deltanet (state in regs across M): 1285→1536.

### 8.5 Negatives (gated, don't retry)

DP4A q8 IQ decode matvec (grid-dequant bound, ~2% slower); LDS-cached IQ
grids (~1%); `__constant__` grids; Q6_K 128-thr; per-row batched MoE prefill;
dequant→bf16+hipBLASLt per-expert experts (16–32 tok/expert too thin);
single-block fused router (serial dots, −16%).

### 8.6 Reproduce

```sh
cd rdna4/llm && make            # HIPBLASLT=0 for the no-blaslt build
M=/mnt/disk1/models/qwen36/35b/Qwen3.6-35B-A3B-UD-IQ3_S.gguf
LLM_PREFILL_WARMUP=2 ./test_hip_llm $M -s 1300 --bench --gpu-only-bench \
  --prefill-len 1024 --decode 128         # pp ~1536, tg ~128.5
./test_hip_llm --verify-quant-kernels     # 18/18 PASS
# llama.cpp: llama-bench -m $M --device ROCm0 -ngl 99 -fa on -p 512,1024 -n 128
```

---

## 9. June 2026 Session — Prefill Decode Optimization Campaign

**Hardware**: RX 9070 XT (gfx1201, ROCm 7.2, Wave32), 16 GB VRAM  
**Target models**:  
- `Qwen3.6-35B-A3B-UD-IQ3_S.gguf` (MoE, IQ3_S gate/up, IQ3_XXS down, n_embd=2048, 40 layers)  
- `Qwen3.6-35B-A3B-UD-IQ2_M.gguf` (MoE, IQ2_XXS gate/up, IQ3_XXS down, n_embd=2048, 40 layers)  
- `Qwen3.6-27B-UD-IQ3_XXS.gguf` (hybrid dense, IQ3_XXS all weights, n_embd=5120, 64 layers)

### Final Benchmarks (June 11)

| Model | Prefill pp1024 | Decode tg128 | Environment |
|-------|:-------------:|:------------:|-----------|
| IQ3_S (MoE, 35B) | **1521 tok/s** | **117 tok/s** | Default; `LLM_MOE_INT8=1` → 1485 |
| IQ2_M (MoE, 35B) | **1515 tok/s** | **36 tok/s** | Baseline was 40 — 38× gain |
| IQ3_XXS (dense, 27B) | **775 tok/s** | **23 tok/s** | Dense FFN n_ff=17408, n_embd=5120 |

### What Was Done (commits chronological)

| Commit | Optimization | Impact |
|--------|-------------|--------|
| `5f513d8` | **Batched MoE prefill for IQ1_S, IQ2_XXS, IQ3_XXS, TQ1_0** | IQ2_M: 40 → 1463 tok/s (**+36×**) |
| | — Add dequant kernels for IQ1_S (50 B/block), TQ1_0 (54 B/block), IQ2_XXS (66 B/block) | |
| | — Extend `batch_qtype_ok` and `get_bf16_weight` for all new types | |
| | — Enable GPU grouping path for IQ1_S+IQ3_XXS, IQ2_XXS+IQ3_XXS | |
| | — Remove `moe_dev_dispatch_ok` dependency for batched prefill eligibility | |
| `2380139` | **Double-buffered LDS BF16 GEMM (`gemm_bf16_own_db`)** | gemm_bf16_own: 221→204 ms (**-7.7%**) |
| | — 2x smA/smB tiles (32 KB LDS), K-loop stride 64 pipeline | |
| | — Load phase N+1 issued during WMMA compute of phase N | |
| `226b510` | **Block-major weight repack (`LLM_MOE_BM=1`)** | IQ3_S: neutral; IQ2_M: +3.5% |
| | — Transpose: row-major [N][nb][bs] → block-major [nb][N][bs] at upload | |
| | — Contiguous row-tile reads recover cache-line over-fetch | |
| `66ad0a0` | **INT8 WMMA grouped GEMM (`LLM_MOE_INT8=1`)** | Comparable to BF16 path |
| | — `dequant_iq2s_all_int8` / `dequant_iq3s_all_int8`: dequant to INT8 | |
| | — `gemm_int8_grouped`: grouped GEMM via INT8 WMMA builtin | |
| | — `quantize_f32_act_to_int8`: activation quantize | |
| `4976141` | **Decode Q6_K matvec kernels: 64→256 threads** | Decode +0.5% |
| | — Better GPU occupancy for Q6_K matvecs | |
| `d873157` | **IQ3_XXS fused decode kernels** | 27B decode: 21.7→23.3 tok/s (**+7.4%**) |
| | — `ffn_gate_up_silu_iq3xxs`, `matvec_down_residual_iq3xxs` | |
| | — `matvec_qkv_iq3xxs`, `matvec_out_gated_iq3xxs` | |
| | — `ssm_matvec4_iq3xxs`, `fused_ssm_out_gated_iq3xxs` | |

### Abandoned Approaches

| Approach | Reason |
|----------|--------|
| **Fused dequant+GEMM** (gemm_dequant_iq2s_grouped) | VGPR 84→99 → occupancy 75%→50%, net slower |
| **MMQ DP4A grouped path** (LLM_MOE_MMQ=1) | DP4A (4 elem/inst) vs WMMA (4096 elem/inst); GPU timeout at pp1024 |

### How to Reproduce (HIP runner)

```bash
# Build
cd /mnt/disk1/work/gemm/main/rdna4/llm
make HIPBLASLT=1

# IQ3_S MoE model — prefill + decode benchmark
LLM_PREFILL_WARMUP=2 ./test_hip_llm \
  /mnt/disk1/models/qwen36/35b/Qwen3.6-35B-A3B-UD-IQ3_S.gguf \
  -s 1300 --bench --gpu-only-bench --prefill-len 1024 --decode 128

# IQ2_M MoE model
LLM_PREFILL_WARMUP=2 ./test_hip_llm \
  /mnt/disk1/models/qwen36/35b/Qwen3.6-35B-A3B-UD-IQ2_M.gguf \
  -s 1300 --bench --gpu-only-bench --prefill-len 1024 --decode 128

# 27B dense model
LLM_PREFILL_WARMUP=2 ./test_hip_llm \
  /mnt/disk1/models/qwen36/27b/Qwen3.6-27B-UD-IQ3_XXS.gguf \
  -s 1300 --bench --gpu-only-bench --prefill-len 1024 --decode 128

# Optional: INT8 WMMA (LLM_MOE_INT8=1), block-major repack (LLM_MOE_BM=1)

# Verification
./test_hip_llm --verify-quant-kernels      # 18/18 PASS expected

# Profiling
rm -rf /tmp/prof && LLM_PREFILL_WARMUP=2 rocprofv3 --kernel-trace -d /tmp/prof -f csv -- \
  ./test_hip_llm /mnt/disk1/models/qwen36/35b/Qwen3.6-35B-A3B-UD-IQ3_S.gguf \
  -s 1300 --bench --gpu-only-bench --prefill-len 1024 --decode 4
awk -F',' 'NR>1{gsub(/"/,"",$8);k=$8;t=$11-$10;if(t>0){kt[k]+=t;kc[k]++}}
  END{for(k in kt)printf "%s\t%d\t%.0f\n",k,kc[k],kt[k]}' \
  /tmp/prof/radv/*.csv | sort -t$'\t' -k3 -rn | head -20
```

### Key Files

| File | Purpose |
|------|---------|
| `rdna4/llm/hip_llm_runner.c` | Main runner: struct, GPU kernel source, prefill/decode logic (~10900 lines) |
| `rdna4/llm/hip_llm_runner.h` | Public API header |
| `rdna4/llm/test_hip_llm.c` | Test harness |
| `rdna4/llm/mm_blaslt_bridge.cpp` | hipBLASLt bridge with algo pinning |
| `rdna4/llm/Makefile` | Build: `make HIPBLASLT=1` |
| `rdna4/llm/perf-report.md` | This file |

### Decode Path Launch Sequence

```
SSM layer (30 layers for 35B-A3B, 16 for 27B):
  1. res_rmsnorm_f32 — residual + RMSNorm (fused)
  2. ssm_matvec4_q6k/iq3xxs — 4 SSM input matvecs (fused)
  3. ssm_prep_f32 — conv1d + l2norm + repeat + softplus + sigmoid (fused 7→1)
  4. deltanet_step_warp_f32 — warp-per-row state recurrence
  5. fused_ssm_out_gated — gated RMSNorm + SSM output matvec (fused)
  6. (MoE: router+gateup+down) or (dense FFN: gate_up_silu + down_residual)

Attention layer (10 layers for 35B-A3B, 48 for 27B):
  1. res_rmsnorm_f32
  2. matvec_qkv_q6k/iq3xxs — Q/K/V matvecs (fused)
  3. deinterleave + qknorm + rope + kv_store
  4. attn_decode_f32_devp — GQA attention
  5. matvec_out_gated — sigmoid_mul + output matvec (fused)
  6. res_rmsnorm_f32 (pre-FFN)
  7. Dense FFN: gate_up_silu + down_residual (fused)
```

### Prefill Batched Path Eligibility

Supported types in `batch_qtype_ok()`: F32, F16, Q4_K, Q5_K, Q6_K, IQ3_XXS, IQ4_XS, IQ2_XS, IQ2_S, IQ3_S, IQ1_S, TQ1_0, IQ2_XXS.

Grouped MoE path activates for:
- IQ2_S gate/up + IQ3_S down (IQ3_S model)
- IQ2_XXS gate/up + IQ3_XXS down (IQ2_M model)
- IQ1_S gate/up + IQ3_XXS down (27B model parts)

Per-expert WMMA fallback handles all types via `get_bf16_weight` (on-the-fly dequant to BF16).

### Decode Fused Kernel Conditions

| Weight Type | FFN gate+up | FFN down | QKV | Output | SSM matvec4 | SSM out |
|-------------|:---------:|:--------:|:---:|:-----:|:----------:|:-------:|
| Q6_K | `ffn_gate_up_silu_q6k` | `matvec_down_residual_q6k` | `matvec_qkv_q6k` | `matvec_out_gated_q6k` | `ssm_matvec4_q6k` | `fused_ssm_out_gated_q6k` |
| IQ3_XXS | `ffn_gate_up_silu_iq3xxs` | `matvec_down_residual_iq3xxs` | `matvec_qkv_iq3xxs` | `matvec_out_gated_iq3xxs` | `ssm_matvec4_iq3xxs` | `fused_ssm_out_gated_iq3xxs` |

### IQ3_XXS Block Layout (98 bytes per 256 elements)

```
offset  0:  d       (half) — super-block scale
offset  2:  qs[64]  — grid indices (8 sub-blocks × 8 bytes)
offset 66:  sas[32] — scales_and_signs (8 sub-blocks × 4 bytes)
```

Dequant: `db = d * (0.5f + (aux32>>28)) * 0.5f`, grid via `iq3xxs_grid_dev[]`, signs via `ksigns_iq2xs_dev[]`.

### Remaining Opportunities

1. **Prefill: f32→int8 activation + int8 WMMA** — Already implemented but quantize overhead offsets 2× BW savings. Could win if quantize is fused into gather.
2. **Decode: LLM head via top-K approximation** — `matvec_q6_K_f32` is ~9% of decode time.
3. **Decode: Extend MoE decode kernels for IQ3_XXS** — Currently only IQ2_S DP4A and IQ3_S matvec.
4. **Prefill: 27B dense FFN** — n_ff=17408 projections dominate; double-buffer helps but the GEMM is FLOPs-bound.
5. **Decode: WMMA flash-attention decode for head_dim=256** — Could help at large seq_len.

### Dead Ends (don't retry)

- **MMQ DP4A grouped path** — DP4A too slow vs WMMA on gfx1201. GPU timeout at pp1024.
- **Fused dequant+GEMM** — Higher VGPR → lower occupancy → net slower.
- **Vectorized scatter** — 4× elements/thread increased atomic contention.
- **Chunked DeltaNet scan** — Occupancy-bound: too few heads (32) to fill GPU.

### Key Environment Variables

| Variable | Default | Effect |
|----------|---------|--------|
| `LLM_PREFILL_WARMUP` | 0 | Warmup runs before timed prefill |
| `LLM_BATCH_DISABLE` | — | Force per-token fallback |
| `LLM_MOE_FUSED` | 1 | Fused MoE decode kernels |
| `LLM_MOE_BM` | 0 | Block-major weight repack |
| `LLM_MOE_INT8` | 0 | INT8 WMMA grouped path |
| `LLM_GEMM` | own* | `own` (WMMA) or `blaslt` (hipBLASLt). *Auto-flips to `blaslt` for large dense (non-MoE, n_embd≥4096) models — see §10. |

---

## 10. June 11 Session — 27B dense prefill 1.74× via hipBLASLt default

**Target**: `Qwen3.6-27B-UD-IQ3_XXS.gguf` (dense hybrid: 48 DeltaNet-SSM +
16 gated-attn, n_embd=5120, n_ff=17408, head_dim=256). Re-profiled the current
build (the §4 breakdown was stale — predates the fused IQ3_XXS decode kernels).

### 10.1 Headline

| metric | before | after | change |
|--------|-------:|------:|-------:|
| prefill pp1024 | 771 t/s | **1346 t/s** | **1.74×** |
| decode tg128 | 23.2 t/s | 23.2 t/s | unchanged |

Decoded tokens bit-identical (greedy id stream stable); `--verify-quant-kernels`
18/18. Commit `8b48e06`.

### 10.2 The win — GEMM backend is model-dependent

The self-owned `gemm_bf16_own_db` was the default (`gemm_own=1`) because it beats
Tensile for the **35B MoE** (n_embd=2048) grouped path. But for the **27B dense**
model's large FFN GEMMs (N=17408, K=5120), **hipBLASLt is 1.73× faster**
(`LLM_GEMM=blaslt`: 771→1334 t/s). The §8.4 "own beats Tensile" finding was
specific to the 35B's small-N shapes; it does not generalize to large dense
GEMMs.

Fix: in the prefill-eligibility block (`hip_llm_runner.c` ~7977), auto-default to
hipBLASLt when `!is_moe && n_embd>=4096`. MoE keeps the self-owned grouped path
(35B IQ3_S still 1515/117, IQ2_M unaffected); small dense models (n_embd<4096,
e.g. Qwen3-VL-2B) keep `own`. Explicit `LLM_GEMM=own/blaslt` overrides. The VLM
e2e (27B LLM prefill on Brooklyn Bridge) inherits the 1.74× for free.

New prefill breakdown @pp1024 (760 ms): hipBLASLt Tensile GEMM 50%,
`dequant_iq3_xxs_to_bf16` 19% (static FFN weights, re-dequanted per pass — can't
cache to bf16, would OOM 16 GB), deltanet 5%, FA-hd256 3%, rest aux.

### 10.3 Decode is at its architectural floor (two null attempts)

Decode (43 ms/tok) is ~49% dense-FFN IQ3_XXS matvecs, ~22% SSM, the rest attn +
LM head. Two attempts, both **null/negative** — do not retry:

1. **Vectorize IQ3_XXS dequant** (float4 x-loads + dword grid reads, the
   `q6k_dot4` pattern that won +13% on the 35B). **Exactly neutral** on 27B
   (23.20→23.20). Reason: these kernels read the grid codebook from
   `__device__ const` (constant cache, already fast) and `x` from L2; the
   DRAM-resident weight bytes already stream coalesced. The inner loop was never
   the bottleneck — float4 only adds register pressure. (The 35B win was in
   different MoE kernels.) Reverted.
2. **Q4_K SSM matvec → warp-per-row `mw` kernel** (mirroring the Q5_K mw default,
   for the 64-thread one-block-per-row Q4_K decode matvec). **Regressed**
   23.3→20.6: at n_cols=5120 (nb=20), the 32-lane warp runs only ~20 lanes
   active with scattered 144-B block reads — worse than the 64-thread kernel.
   Reverted.

Decode is launch-count/latency bound across 64 layers (already heavily fused +
graph-captured), matching the §8.3 conclusion. Real decode headroom would need a
structural change (e.g. block-major dense-FFN weight repack), not micro-opts.

### 10.4 Skipped (low ROI)

- **hipBLASLt algo-pinning** (§Opp D): bridge supports `MM_BLASLT_ALGO_PINS` but
  inline sweep segfaults on gfx1201; each shape has 64 candidates → external A/B
  over many shapes for est. 0.5–2%. Heuristic first-valid already gives 1.74×.
- **INT8 dense FFN**: existing MoE int8 path applies no scales (unusable as-is);
  a correct scaled path is high-effort + quality risk. Deferred.

---

## 11. June 11 Session — 27B decode roofline + warp-per-row matvecs (23.3 → 27.8 tok/s)

§10.3 called decode "at floor"; a proper roofline showed it was **at 54% of the
achievable bandwidth**, and a kernel redesign recovered most of the per-layer gap.

### 11.1 Roofline (the "is decode BW-bound?" answer)

- Decode is **100% GPU-active** (44.4 ms/tok GPU vs 43 wall — graph capture
  removed host gaps; no compute slack — float4 dequant was perfectly neutral).
- Dense model → **~11.3 GB of weights read per token** (every weight, every tok).
- **Achievable BW ceiling is format-dependent, not a flat 516:** the LM-head
  Q5_K matvec hits 516 GB/s; Q4_K/Q6_K reach ~500–513; but **IQ3_XXS caps at
  ~300–400 GB/s** — its grid-codebook gather (`iq3xxs_grid_dev[qs[...]]`, 32 lanes
  gathering 32 distinct entries) + per-element sign decode is **ALU/ gather-bound**,
  not BW-bound. Block size is irrelevant (gate_up flat at ~394 across 64/128/256
  threads). So the 45 tok/s "pure-BW" target was optimistic: the dominant FFN is
  IQ3_XXS.
- Pre-redesign aggregate ≈ 278 GB/s = 54% of 516; the per-layer matvecs ran at
  240–314 while only the giant LM-head matvec saturated.

### 11.2 Root cause + fix

Slow kernels were **block-per-row** (256 threads cooperate on one row → cross-warp
`__shared__`+`__syncthreads` reduction = DRAM-idle), and `fused_ssm_out_gated_q6k`
additionally recomputed the per-head RMSNorm sum-of-squares via **contended
atomics in every one of the 5120 output-row blocks** (it is row-independent).

Fix = transplant the in-tree **warp-per-row `G=nb*32` template**
(`matvec_iq2_s_f32`: all 32 lanes active, intra-warp `__shfl` only). Per-call BW:

| kernel | before | after | note |
|---|---:|---:|---|
| `ffn_gate_up_silu_iq3xxs` | 313 | **400** GB/s | IQ3_XXS ALU-capped (~400) |
| `fused_ssm_out_gated_q6k` | 240 | **404** GB/s | + precompute inv_mean once (`ssm_inv_mean_f32`); drop atomic/barriers |
| `matvec_q4_K_f32` (SSM in-proj) | 314 | **513** GB/s | `G=nb*4` (lane=64-elem chunk); Q4_K hits the ceiling |
| `matvec_iq3_xxs_f32` (attn_gate) | 266 | **297** GB/s | nb-stride → `G=nb*32` (all lanes) |

Net decode (canonical pp1024/tg128): **23.3 → 27.8 tok/s (+19%)**; argmax
bit-identical, `--verify` 18/18, prefill unchanged (1350 t/s). Flags default ON:
`LLM_FFN_IQ3_MW`, `LLM_SSM_OUT_MW`, `LLM_Q4K_G4`.

### 11.3 What didn't move (shape/format-limited)

- **FFN down** (IQ3_XXS [5120,17408], 5120 rows): warp-per-row *underfills*
  (640 blocks) and split-K (`LLM_DOWN_KSPLIT`) only 287→303 GB/s — stuck at ~300
  for IQ3_XXS at this row count. Kept on original block-per-row. (`_mw`/`splitk`
  twins + `LLM_MW_THREADS` knob exist, gated off.)
- **gate_up** is at the IQ3_XXS ALU ceiling (~400), not the 513 BW ceiling.

### 11.4 Honest ceiling

The realistic decode ceiling for this IQ3_XXS-heavy model is **~32–34 tok/s**
(IQ3_XXS FFN ≈ 60% of traffic, ALU-capped ~300–400), not the 45–50 implied by a
flat 516 GB/s. We reached 27.8; the remaining gap is the down + attn_gate IQ3_XXS
kernels held below their format cap by row count. Breaking ~400 on IQ3_XXS would
need a cheaper decode (LDS-resident grid gather — prior-session negative on 35B,
~1%) or a different quantization. 50 tok/s is not reachable without changing the
weight format (less traffic / simpler decode).

---

## 12. June 11 Session — Qwen3.5-27B-UD-Q2_K_XL support (prefill 14→1364, decode 14→30 t/s)

A second 27B checkpoint with a different quant mix: FFN gate/up **Q2_K**, FFN down
**Q3_K**, SSM in-proj **Q2_K**, ssm_out **Q5_K**, alpha/beta **Q8_0**, attn_gate
**IQ3_XXS**, LM head **Q6_K**. None of Q2_K/Q3_K/Q8_0 were supported in the fast
paths, so both prefill and decode were on slow fallbacks.

### 12.1 Prefill 14 → 1364 t/s (98×) — batched-path support

Q2_K/Q3_K/Q8_0 weren't batch-eligible → prefill ran per-token 1024× (72 s). Added
dequant-to-bf16 kernels (one thread per output column, order matching each
matvec) + wired into `get_bf16_weight` / `batch_qtype_ok` / `quant_matvec_block_info`.
Bugs fixed: (a) Q8_0 uses the runner's **padded 36-byte** upload (`[d:2][pad:2]
[qs:32]`), not native 34-byte; (b) `get_bf16_weight` cases must `return
r->d_wbuf_bf16;` not `break;` (break → default NULL path → silent forward fail).
Validated vs CPU ref: token-0 hidden rel_L2=3e-6, ≤0.02 across 64 layers (Q2_K
2-bit through bf16; norms match). Commit `7b80065`.

### 12.2 Decode 14 → 30 t/s (2.2×) — warp-per-row Q2_K/Q3_K matvecs

`matvec_q2_K_f32`/`matvec_q3_K_f32` were block-per-row with 256 threads but only
~nb (≈20) active → 116/189 GB/s. Added `G=nb*4` warp-per-row twins (one lane per
64-element `(n0,half)` chunk: 16 qs bytes shared across all 4 scale-groups, no
redundant reads, all 32 lanes active, intra-warp reduction — the Q4_K g4 pattern):

| kernel | before | after |
|---|---:|---:|
| `matvec_q2_K_g4` | 116 GB/s (251 µs) | **458 GB/s** (64 µs) |
| `matvec_q3_K_g4` | 189 GB/s (203 µs) | **443 GB/s** (87 µs) |

Default ON (`LLM_Q2K_G4`/`LLM_Q3K_G4`). Decoded tokens bit-identical g4 on/off;
CPU-ref rel_L2 unchanged. Q2_K/Q3_K hit ~450 GB/s (simpler decode, no IQ3_XXS
grid gather), so Q2_K_XL decode (30 t/s) now **beats** the IQ3_XXS model (27.8).
Commit `f293e53`. Remaining (small): q5_K ssm_out 342 GB/s (nb-stride, 6144 rows),
iq3_s; the FFN is unfused (separate gate/up/silu/down) but matvec-dominated.

### 12.3 Final Q2_K_XL (pp1024 / tg128)

prefill **1364 t/s**, decode **30.0 t/s**. IQ3_XXS + 35B MoE models unaffected
(1347/27.8, 1499/121); `--verify` 21/21.

### 12.4 June 12 vs llama.cpp ROCm (RX 9070 XT, pp1024/tg128)

`llama-bench` was run from `~/work/llama.cpp/build_rocm722_rdna4_fa`
(`549b9d843`, build 9307) with `-ngl 99 -fa 1 -r 3`. The HIP runner was run
with `LLM_PREFILL_WARMUP=2`.

| model | runner | pp1024 | tg128 | note |
|---|---:|---:|---:|---|
| `Qwen3.5-27B-UD-Q2_K_XL` | llama.cpp ROCm | 686.85 | 29.69 | `qwen35 27B Q2_K - Medium` |
| | HIP runner | **1362.01** | 29.64 | prefill **1.98x** llama.cpp; decode parity |
| `Qwen3.6-27B-UD-IQ3_XXS` | llama.cpp ROCm | 1093.19 | 26.25 | `qwen35 27B IQ3_XXS - 3.0625 bpw` |
| | HIP runner | **1348.37** | **27.45** | prefill **1.23x**, decode **1.05x** |

Takeaway: Q2_K_XL decode has effectively closed the llama.cpp gap while the
batched prefill path is nearly 2x faster. IQ3_XXS still has modest headroom but
already leads llama.cpp on both metrics; the remaining decode limit is mostly
IQ3_XXS gather/dequant plus dense-FFN traffic.

### 12.5 Gemma4 12B (RX 9070 XT, Q6_K_XL) — correct end-to-end + long context

First Gemma4 inference on AMD HIP (RDNA4). The port now produces **coherent,
correct output**, verified two ways: (a) per-layer activations match llama.cpp's
`llama-eval-callback` on the same GGUF (layer-0 `attn_out`/`l_out` agree to quant
noise, e.g. `l_out-0 = 0.181,-0.146,16.63` vs ref `0.177,-0.146,16.62`), and
(b) greedy generation is fluent ("The capital of France is" → " Paris. ... Paris
is the capital and most populous city of France"). llama.cpp (build 9445) is the
reference; it loads this exact GGUF and is coherent, so the GGUF is good.

**Architecture (Gemma4 "unified", confirmed from HF config + weights + llama.cpp):**
- Full-attention layers (every 6th: 5,11,…,47) set **value = key**
  (`attention_k_eq_v`); there is no `v_proj`. V is the *raw* k_proj output (pre
  k_norm) passed through a weightless per-head RMS norm, and is **not** RoPE'd.
- Full-attn uses `num_global_key_value_heads = 1` (one 512-wide KV head),
  proportional RoPE (`rope_freqs`, base 1e6); SWA uses 8 KV heads, head_dim 256,
  base 1e4. Attention scale is **1.0** (the Q/K norms set magnitude).
- `x += post_attn_norm(Wo·attn)`; embeddings scaled by `sqrt(n_embd)`.

**Bugs fixed to go from garbage → coherent** (the forward pass had never actually
run before this work — attention was silently skipped by an over-LDS launch):
bounded online-softmax flash attention (≤64 KB LDS); circular SWA KV cache
(window+chunk; the old linear store overflowed past pp1024 and could hang the
GPU); per-layer SWA stride fix (RoPE/QK-norm/KV-store used the global q/kv_dim,
2× too wide for 256-wide SWA heads); V=K with the correct pre-norm/raw-rms-norm
semantics; attention scale 1.0; the `x += post_attn_norm(attn_out)` residual
order; per-layer RoPE base + full-rotation `rope_dim` + proportional `freq_factors`;
head_dim-512-safe QK-norm and V-norm kernels; and — the last and decisive one —
the `sqrt(n_embd)` embedding scale, which was applied only on the final-token
`hip_llm_forward_logits` path and missing from both the per-token
`hip_llm_forward` and the batched `embed_tokens_batch` prefill paths, so every
prefill token's residual base was ~62× too small.

**Prefill throughput** (`LLM_PREFILL_WARMUP=1`, correct full V=K computation). The 8
full-attention layers (head_dim 512, 1 KV head, GQA 16) do real O(M²) work and
dominate long-context cost. They run on a WMMA flash kernel by default
(`GEMMA_FA_WMMA=0` reverts to the scalar online-softmax kernel):

| pp    | scalar | WMMA | speedup |
|------:|-------:|-----:|--------:|
| 512   | 1715   | 1809 | 1.05×   |
| 2048  | 1006   | 1135 | 1.13×   |
| 8192  | 543    | 774  | 1.43×   |
| 16384 | 356    | 581  | 1.63×   |
| 32768 | 152    | 324  | 2.13×   |

The WMMA kernel (`flash_attn_wmma_hd512_causal`) splits the 512 head_dim across **4
waves** so each wave's O accumulator (128 dims) stays in registers. head_dim 512 is
the obstacle: a single-wave O accumulator is 256 VGPR / 32 KB LDS and collapses
occupancy to 1 wave/CU — a naive 1-wave WMMA kernel was actually *slower* than
scalar (390 vs 543 @pp8192). The 4-wave split has each wave contract only its 128
dims into a partial QK score; the 4 partials are summed via LDS to the full score,
which all waves softmax. It reads the F32 KV cache directly (no F16 pack) and keeps
occupancy up, so the speedup grows with context. Throughput still falls with M
(the work is genuinely O(M²)); an earlier revision claimed ~1084 @ pp32k by
*skipping* the full-attn layers — fast but wrong, since their V is K, not missing.

Decode ~45 tok/s. VRAM ~constant in context (full-attn KV is only 1 head → ~1 GB at
32k); 12B Q6_K weights (~10.7 GB) dominate. Further headroom: F16 KV cache and
double-buffered tile loads in the WMMA kernel.
