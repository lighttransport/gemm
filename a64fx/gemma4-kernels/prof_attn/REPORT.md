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

## ✅ Skip zero key-blocks (lever #3) — landed: O(N²) → O(N·window)
Each query-tile only attends keys in [active_lo, active_hi) — causal `k≤qmax`, SWA
`k≥q0−win+1` — so keys outside are 0 for all 12 rows. QK^T, the `memset`, the pack,
and P·V now run only over the active key-tile range `[kt_lo,kt_hi)` instead of all `KT`.
SWA becomes window-bounded (~9 tiles regardless of N); full-causal becomes triangle-bounded.

A/B (env `ATTN_NOSKIP=1`), hd=256 SWA win=512, best-of-3 wall-clock (48 cores):
| N | no-skip | **skip** | speedup |
|---|--:|--:|--:|
| 768 | 0.510 ms | 0.505 ms | 1.01× (window≈seq) |
| 1536 | 0.791 ms | 0.646 ms | 1.22× |
| 4096 | 2.811 ms | **0.856 ms** | **3.28×** |

Speedup grows with N (the O(N²)→O(N·window) crossover) — the real win for long-context
prefill, where 5/6 of Gemma-4 layers are SWA(512). relL2 bit-identical (skipped tiles
were exactly zero-contribution). Also bounds full-causal to the triangle.

## Cumulative attention progress (this session)
scalar+no-FZ16 baseline → FZ16 → FEXPA softmax → vec transpose/pack → skip-zero-blocks.
GEMM microkernels unchanged at 93–96 %; the wins are all in the softmax/pack/iteration glue.

Reproduce: `make attn_profile && OMP_NUM_THREADS=48 OMP_PROC_BIND=close OMP_PLACES=cores ./attn_profile 256 4096 512`
(env `ATTN_NOSKIP=1` no-skip baseline, `ATTN_SCALAR_EXP=1` libm softmax).
fapp: `make attn_profile_fj && OMP_NUM_THREADS=1 taskset -c 12 bash profile_fapp.sh attn_profile_fj "256 384 512" prof_attn_vec`.
