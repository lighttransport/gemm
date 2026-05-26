# RDNA4 FP8 flash-attention: optimization findings

GPU: RX 9070 XT (gfx1201). FP8 WMMA peak 342.9 TF/s. d=128, batched over heads.
Context: vLLM #28649 (RDNA4 FP8 WMMA) → can FP8 flash-attn reach ~50% peak (150 TF/s)?

## Verdict
**No.** FP8 flash-attn ceiling on this arch is ~30–35 TF/s (~10% peak). 150 TF/s
is not reachable. The limiter is register/issue capacity, not softmax — confirmed
by stripping softmax to a bare roofline and by testing the FA3 overlap idea.

## Measured (S=4096, 16 heads)
| variant | TF/s | %peak | note |
|---|---|---|---|
| approx_b32_16w_fast (best, full) | 29.7 | 8.7% | online softmax, LDS P-transpose |
| roof_16w (QK+exp2+PV only) | 35.4 | 10.3% | no max/l_i/transpose, garbage data |
| roof_16w, exp removed (2× GEMM) | 35.2 | 10.3% | exp2 is essentially free |
| shflp_16w (shuffle transpose) | 25.2 | 7.3% | correct, slower than LDS |
| b64_16w (BKV=64) | 19.6 | 5.7% | LDS pressure |
| roof_4h NH=2 (2-head pipeline) | 20.7 | 6.0% | spills |
| roof_4h NH=4 (4-head pipeline) | 10.9 | 3.2% | full spill |
| FP8 GEMM (fat, M/N/K≈1024/4608/4608) | 205 | 60% | for reference |
| heads 16→32 | flat | — | not occupancy-bound |

## Why
- **Thin dependent chains.** Each KV tile is N=32; QK→PV won't pipeline like fat
  GEMM. exp2 is free (overlapped). Wall ≈35 TF/s even ignoring all correctness.
- **P-transpose isn't it.** Shuffle transpose is correct but −13% — 64 shfls/tile >
  LDS round-trip.
- **FA3 cross-head overlap regresses.** 4×qk,4×exp,4×pv needs 4 live accumulators
  (O[8][8]=64 VGPR each → 256 = whole register file) → spills. RDNA4 lacks the big
  register file + async MMA that makes Hopper FA3 work. Same reason pc/v3 were null.
- More heads/batch is flat: compute/issue-bound, not occupancy-bound.

## Don't retry
Shuffle transpose, BKV=64, head interleaving, producer/consumer, double-buffer.
The 50% target is for fat prefill GEMM (already 60%), not attention.
Diagnostics left as modes: roof_16w, roof_4h, shflp_8w/16w, b64_16w.
