# fapp CPU-PA report — kernel_q8v2_3x4 (per-block), single core (12)

Config: `q8v2_profile_fj 16 5000 0` (nb=16 → K=512, 5000 reps), MR=3×NR=64 tile,
L1-resident. fapp 4.2.5, 17 PA groups, freq 2.0 GHz. Region `q8v2_3x4`.

## Headline
| metric | value |
|---|---|
| cycles / call | **1835** |
| int8-peak | **42 %** (sdot floor 768 cy vs 1835) |
| IPC (effective) | 2.35 |
| FP-pipe busy | FLA 66.5 % · FLB 60.9 % (avg **63.7 %**) |
| fully-stalled cycles (0-commit) | **14.4 %** |
| full 4-wide commit | 43.1 % |
| L2 refills (whole region) | 95 → **L1-resident** |

## Per-call instruction mix (5000 reps)
- **1536 SDOT** (`ASE_SVE_INT_SPEC`) — the useful int8 work (12 acc × 8 × 16 blk)
- **385 FMUL+FMLA** (`FP_SPEC`) — per-block scale-out
- ~192 SCVTF + ~192 DUP-zero (int32 partial reset) — also per-block
- ~972 SVE loads (ld1rw A, ld1b B, ld1w dw) ; 3338 SIMD inst/call total

## Where the cycles go
The kernel is **compute-bound, not memory-bound** (L1-resident, 95 L2 refills total).
SDOT alone would take 768 cy (2/cy). The per-32-block **scale-out roughly doubles**
the non-load SVE-arith work (1536 sdot → +~770 scvtf/fmul/fmla/dup), and a
**14.4 % dead-stall** comes from the `scvtf → fmul → fmla` dependency chain plus the
block-boundary WAR hazard on the int32 partials. FP pipes sit at only 64 % busy →
the 36 % idle is those dependency stalls.

This is exactly the structural Q4_0 ceiling predicted by `../../doc/FP16_GEMM_CEILING.md`:
the convert+scale-every-32-elements epilogue, which OoO cannot hide because it is a
true data dependency, not load latency.

## Levers (fapp-pointed)
1. **per-row activation** (`_arow`, drops the per-block FMUL) → measured **47 %** — already in the kernel.
2. Software-pipeline scale-out of block N with SDOTs of block N+1 to fill the 14 %
   dead-stall — needs >32 registers (would spill); marginal.
3. The 42–47 % is near the practical ceiling for per-block Q4_0; the real prefill
   win is the ~12× over the production fp32-dequant GEMM, not %-of-int8-peak.

## Reproduce
```sh
make q8v2_profile_fj
OMP_NUM_THREADS=1 taskset -c 12 bash profile_fapp.sh q8v2_profile_fj "16 5000 0" prof_q8v2_pb
```
Event decode via ../../doc/a64fx_pmu_events.csv. Raw counts: pa{1..17}.csv.
