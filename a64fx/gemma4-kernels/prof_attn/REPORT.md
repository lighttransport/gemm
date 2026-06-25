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
1. **FEXPA fast-exp** for softmax (`../../fused-gemm/`/`../../int8-new/online_softmax`
   already have it; the doc's exp2_fexpa is ~7× faster than Padé+FDIV) — the #1 lever.
2. Vectorize the score transpose / fuse it into the QK^T epilogue.
3. Deeper pipeline overlap (produce softmax block kb+2 while P·V consumes kb).
Throughput still scales well across cores (2128 GF/s @48c, hd=256 N=1536) because the
per-tile work is independent; the per-core inefficiency is softmax-glue, not the GEMMs.

Reproduce: `OMP_NUM_THREADS=1 taskset -c 12 bash profile_fapp.sh attn_profile_fj "256 384 512" prof_attn`
