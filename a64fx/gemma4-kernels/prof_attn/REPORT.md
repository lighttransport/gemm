# fapp CPU-PA — attn_fused (fused fp16 attention), FZ16 on

Config: `attn_profile_fj 256 384 512` (hd=256 SWA, N=384, window=512), single core,
20 reps. Region `attn_fused`. Full pipeline: pack Q → QK^T → causal mask → 2-pass
softmax → transpose P → P·V.

## FZ16 multi-thread throughput (48 cores, before → after)
| shape | no FZ16 | **FZ16** |
|---|--:|--:|
| hd=256 N=768 win=512 | 284 GF/s | **833 GF/s** (2.9×) |
| hd=512 N=768 win=0 | 373 GF/s | **1263 GF/s** (3.4×) |
| hd=256 N=1536 win=512 | 462 GF/s | **2128 GF/s** (4.6×) |

relL2 unchanged (0.004–0.006). The earlier WS3 numbers were all subnormal-stalled.

## CPU-PA (single core, FZ16 on)
| metric | value |
|---|--:|
| FP-pipe busy | FLA 49.0 % · FLB 37.5 % (avg **43.3 %**) |
| IPC (effective) | 1.70 |
| **0-commit stall** | **42.0 %** |
| 4-wide commit | 29.9 % |
| L2 refills | 122 → L1-resident |

## Reading
The GEMM microkernels themselves run at 93–96 % (see ../prof_fp16_fz/REPORT.md), but
the **fused pipeline sits at 43 % FP-busy / 42 % stalled** even with FZ16. The stall is
the **non-GEMM glue between the two GEMMs**: scalar `expf` softmax, the score
transpose (sc[12][Kpad] → Pp[seq][12]), causal masking, and Q/A packing — all
stall the FP pipes. This is the doc's "pass1 + fill ~20 % overhead", worse here
because softmax uses libm `expf` rather than the SVE **FEXPA** fast-exp.

### Levers (fapp-pointed)
1. **FEXPA fast-exp** for softmax — ✅ DONE (see below).
2. Vectorize the score transpose / fuse it into the QK^T epilogue. (next)
3. Deeper pipeline overlap (produce softmax block kb+2 while P·V consumes kb).

## ✅ FEXPA softmax (sve_expf) — landed
Replaced the scalar libm `expf` softmax loop with a vectorized 5-op FEXPA `sve_expf`
(`exp(v)=2^(v·log2e)`: `u=v·L2E → z=u+SH → fexpa → r=u−(z−SH) → ·(1+r·ln2)`; SH=0x48481fc0).
A/B via env `ATTN_SCALAR_EXP=1`.

**Accuracy-neutral** — relL2 identical to libm in every shape (0.00589 / 0.00430 / 0.00585).

fapp (single core, hd=256 N=384 win=512):
| | scalar expf | **FEXPA** |
|---|--:|--:|
| cycles | 103.8 M | **48.7 M** (2.13× fewer) |
| FP-pipe busy | 43.3 % | **63.5 %** |
| 0-commit stall | 42.0 % | **30.1 %** |
| 4-wide commit | 29.9 % | 44.9 % |

Multi-thread throughput (48 cores):
| shape | scalar | **FEXPA** | speedup |
|---|--:|--:|--:|
| hd=256 N=768 SWA | 818 GF/s | **1140** | 1.39× |
| hd=512 N=768 full | 860 GF/s | **1776** | 2.07× |
| hd=256 N=1536 SWA | 2026 GF/s | **2736** | 1.35× |

The residual 30 % stall is now the **score transpose + Pp pack** (strided fp16 store)
and the sum-reduction — lever #2. The GEMMs are unchanged at 93–96 %.

Reproduce: `make attn_profile_fj && OMP_NUM_THREADS=1 taskset -c 12 bash profile_fapp.sh attn_profile_fj "256 384 512" prof_attn_fexpa`
(scalar baseline in prof_attn/, env `ATTN_SCALAR_EXP=1`).
