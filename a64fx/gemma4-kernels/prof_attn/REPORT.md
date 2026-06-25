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

## ✅ Vectorized transpose + Pp pack — landed (lever #2)
The QK^T→sc copy is now a vectorized contiguous row copy; the softmax max/exp/sum are
SVE-vectorized; the transpose-pack (sc[r][k] → Pp[k][r], scale + fp16) uses a
halfword-narrowing scatter `svst1h_scatter_u32index_u32` (SVE has no f16 scatter →
reinterpret f16-as-u32, store low 16b), with `Pp` zeroed once per tile (invalid keys → 0).

fapp single core (hd=256 N=384 win=512), cumulative:
| | cycles | FP-busy | stall |
|---|--:|--:|--:|
| scalar expf | 103.8 M | 43.3 % | 42.0 % |
| + FEXPA softmax | 48.7 M | 63.5 % | 30.1 % |
| **+ vec transpose/pack** | **44.4 M** | **66.2 %** | **26.5 %** |

relL2 still bit-identical (0.00589 / 0.00430 / 0.00585). **Total 103.8M → 44.4M = 2.34×
fewer cycles per core** vs the original scalar softmax.

Caveats / next lever (#3): A64FX **scatter is slow** and the per-tile `memset(Pp)` adds
shared-memory write traffic, so at 48 threads the wall-clock gain is BW-masked (per-core
is the clean signal). The real SWA lever is **skipping zero key-blocks in P·V** (the
window makes most of `Pp` zero — contract P·V only over active key-tiles per qt-tile),
which removes BOTH the memset and the wasted P·V compute. The GEMMs stay at 93–96 %.

Reproduce: `make attn_profile_fj && OMP_NUM_THREADS=1 taskset -c 12 bash profile_fapp.sh attn_profile_fj "256 384 512" prof_attn_vec`
(env `ATTN_SCALAR_EXP=1` for the libm baseline).
