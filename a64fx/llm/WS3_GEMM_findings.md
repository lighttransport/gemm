# WS3 — GEMM 12×2 / outer-product port: investigated, **NOT integrated** (refuted end-to-end)

**Status:** investigated with standalone micro-benches; **no production code changed.** The
proposed kernel port does **not** yield an end-to-end prefill speedup, so it was deliberately not
integrated. This is the honest negative result + the evidence.

## What WS3 proposed
Port the 89%-of-peak **12×2 micro-kernel** from `a64fx/doc/FP16_GEMM_CEILING.md` into the bf16→f32
prefill GEMM (`ds4f_gemm_worker` / `matvec_bf16_8x3_pv_acc`) for a claimed 3–4×.

## Finding 1 — the blueprint's headline lever is forbidden here
`FP16_GEMM_CEILING.md` is entirely **fp16-accumulate** (`z0–z23` are fp16 accumulators, `fmla .h`).
Its top result (NOEPI, +10 GF) is exactly the fp16-C accumulation the WS3 note forbids (fp16-acc
flips argmax — the Step-2o FTZ lesson). Only the **outer-product geometry** transfers to a
bf16-in/fp32-acc kernel; the blueprint's own scheduling tweaks (4K unroll, split-B, double-buffer)
each gave **<1%** (OoO hides them). So the realistic ceiling was never 3–4×.

## Finding 2 — the current kernel is a dot-product; outer-product is 2× **in isolation**
`matvec_bf16_8x3_pv_acc` is a **dot-product** kernel (K-in-lanes, bf16→f32 via predicated `p_odd`
loads, `svaddv`-reduced): 8 weight loads + 3 x-loads per 24 FMA, load-issue-bound. I built an
**outer-product** kernel (broadcast x, vector weights, accumulator = output tile, no reduction) and
micro-benched it single-thread, L2-resident (`tools/ws3_op_microbench.c`):

| kernel | Gmac/s (1 thread) |
|--------|---|
| dot-product 8×3 (current) | 12.8 |
| outer-product, **strided** x | 2.5 (8× WORSE — strided activation blows L1) |
| outer-product, **transposed** x (k-major, L1-hot) | **26.4 → 2.07×** |

relL2 1.2e-6 (bit-similar, same as the dot kernel, well under the bf16 ~1e-4 tolerance). So the
*kernel* is genuinely 2× — **but only with x transposed to `[k][token]` and reused L1-resident.**

## Finding 3 (decisive) — the 2× **vanishes at 48 threads** with real weight-streaming
The production GEMM runs 48 threads streaming distinct weights from HBM. I replicated that exactly
(`tools/ws3_op_mt.c`: full-shape, 48 threads, weights packed once, x pre-transposed once, op vs dot,
relL2 1.2e-6):

| shape | M | dot Gmac/s | op Gmac/s | speedup |
|-------|---|-----------|-----------|---------|
| [8192,4096] (wo_a) | 12 | 535 | 520 | 0.97× |
| [8192,4096] | 24 | 749 | 557 | **0.74×** |
| [8192,4096] | 36 | 765 | 559 | **0.73×** |
| [32768,1024] (wq_b) | 12 | 431 | 523 | 1.21× |
| [32768,1024] | 24 | 622 | 594 | 0.95× |
| [2048,4096] (shared) | 24 | 698 | 417 | **0.60×** |

**At 48 threads the outer-product is 0.60–1.0× — no win, usually a regression** (one favorable
outlier: tall-skinny wq_b at M=12). The single-thread 2× does not survive multi-core: the GEMM is
**memory-system-bound at ~750 Gmac/s aggregate regardless of kernel**, and the outer-product's
extra footprint (1536-f32 output tile, k-major weight + transposed-x streaming) makes it *worse*
under contention.

This **confirms the in-file note** at `common/ds4f_impl.h:566–574` from a prior attempt: *"this dense
GEMM is weight-streaming/compute-bound … no loop transform over the same weights helps; sub-f32
compute (e.g. int8 svdot) is the only remaining lever."*

## Conclusion / recommendation
- **Do not integrate the outer-product / 12×2 kernel.** It would not speed up prefill at the
  production thread count and risks regressing it.
- The multi-threaded prefill GEMM is bound by the memory system, not the FMA kernel. The only lever
  that reduces the binding traffic is **sub-bf16 weights** — already implemented (Q8 int8 svdot,
  MXFP4 nibbles). That is the documented "sub-f32 compute" lever and is where any further prefill
  gain must come from, not from a bf16 kernel re-schedule.
- WS3 as specified (the 12×2 port for 3–4×) rests on the fp16-accumulate ceiling, which is
  inapplicable under the argmax-exact (fp32-acc) constraint.

## Artifacts (kept, reproducible; no production change)
- `tools/ws3_op_microbench.c` — single-thread dot vs outer-product (proves the isolated 2×).
- `tools/ws3_op_mt.c` — 48-thread full-shape (proves it vanishes). `R=/K=/M=` env-overridable.
  `fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp -I../../common -o /tmp/ws3mt tools/ws3_op_mt.c -lm`
