# fapp CPU-PA — micro_kernel_fp16_12x2_swp (FZ16 on), single core

Config: `fp16_profile_fj 384 8000` (K=384, L1-resident, B=48KB). Region `fp16_12x2_swp`.

## ★ FZ16 is decisive
A64FX handles fp16 subnormals in slow microcode. Synthetic (and real attention)
fp16 data is subnormal-prone, so **`FPCR.FZ16` must be set** (`set_fz()` in the driver):

| K | FZ16 ON | FZ16 OFF |
|---|--:|--:|
| 256 | 96 % | — |
| **384** | **97 % (248.7 GF)** | **32 % (81.6 GF)** |
| 512 | 77 % (L1 spill) | — |

3× swing. The doc's reported 89 % assumed normal-range data; with subnormals it
collapses to ~31 %. **Any caller of the fp16 kernel (and the fused attention) must set
FZ+FZ16 per thread.**

## CPU-PA (FZ16 on, K=384)
| metric | value |
|---|--:|
| cycles / call | 4925 |
| **fp16-peak** | **93.6 %** |
| FP-pipe busy | FLA 95.8 % · FLB 95.0 % (avg **95.4 %**) |
| IPC (effective) | 3.20 |
| 0-commit stall | 4.5 % |
| 4-wide commit | 66.7 % |
| FMLA / call | 9217 (= 24·K) |
| L2 refills | 93 → L1-resident |

The microkernel is FP-pipe-saturated (95 %) at the architectural ceiling — matches
`../../doc/FP16_GEMM_CEILING.md` (87–89 % swp/noepi). K=384 (B=48KB) is the L1 sweet
spot; K=512 (B=64KB) spills → 77 %.

Reproduce: `make fp16_profile_fj && OMP_NUM_THREADS=1 taskset -c 12 bash profile_fapp.sh fp16_profile_fj "384 8000" prof_fp16_fz`
