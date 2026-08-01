# fapp side-by-side — kernel_q8v2_3x4: per-block vs per-row (_arow)

Single core 12, `nb=16` (K=512), 5000 reps, L1-resident, 17 PA groups. Region `q8v2_3x4`.
Decoded from prof_q8v2_pb/ and prof_q8v2_arow/ via ../../doc/a64fx_pmu_events.csv.

| metric | per-block | per-row (_arow) |
|---|--:|--:|
| **cycles / call** | 1835 | **1663** |
| **int8-peak** | 41.8 % | **46.2 %** |
| IPC (effective) | 2.35 | 2.45 |
| FLA pipe busy | 66.5 % | 69.3 % |
| FLB pipe busy | 60.9 % | 60.5 % |
| avg FP busy | 63.7 % | 64.9 % |
| SDOT / call (useful) | 1536 | 1536 |
| **FMUL+FMLA / call** | 385 | **205** |
| **0-commit STALL** | 14.4 % | **11.1 %** |
| 4-wide commit | 43.1 % | 42.0 % |
| L2 refills (region) | 90 | 95 |

**per-row speedup: 1.104×** (1835 → 1663 cyc/call).

## Reading
Both do the **identical 1536 SDOTs** (same useful int8 work) and stay L1-resident.
`_arow` factors the per-row activation scale out of the K-loop, so the per-block
scale-out drops from `scvtf → fmul → fmla` to `scvtf → fmla`:
- **FMUL+FMLA falls 385 → 205 /call** (removes the 12 per-block FMULs × 16 blocks).
- The shorter dependency chain cuts the **dead-stall 14.4 % → 11.1 %**.

Net: fewer FP-pipe ops **and** less stall → +4.4 pts of int8-peak (41.8 → 46.2 %).
FP pipes still only ~65 % busy in both → the residual is the unavoidable
convert-every-32 dependency on the int32 partials (structural to Q4_0; see
../../doc/FP16_GEMM_CEILING.md). per-row is the better speed choice **if** the
coarser activation scale holds argmax in end-to-end mini_decode (real activations);
per-block (0.37 % relL2) is the safe accuracy default.
