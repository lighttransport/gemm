# Resume: GLM-5.3-Flash A64FX kernel efficiency + QLAIR accuracy (updated 2026-10-01 18:30)

## Goal
Run GLM-5.3-Flash (GLM53F) efficiently on A64FX. Targets:
- bandwidth-bound kernels: ≥95% of the measured read roof;
- compute-bound kernels: ≥90% of int8/FP peak;
- QLAIR A64FX simulator: <3% cycle error, so kernels can be optimized without hardware.

Work on a 2-node PJM allocation: `hwrun.sh` runs on the quiet second node; builds and QLAIR run on the Claude node.

## Where things are
- **gemm worktree `a64fx/glm5/kern/`:** kernel cores and `KERNELS.md` (inventory with per-rank shapes, plus all results).

  | File | Contents |
  |---|---|
  | `glm53f_kern_v0.c` | production loops |
  | `glm53f_kern_q8r16.c` | Q8_0R16 v1–v4 |
  | `glm53f_kern_q4k16.c` | Q4_KP16 v1–v3 |
  | `glm53f_kern_q5k16.c` | Q5_KP16 v1–v3 |
  | `glm53f_kern_q6k16.c` | Q6_KP16 |
  | `glm53f_kern_gemm.c`, `glm53f_kern_gemm_asm.S` | prefill GEMM |

- **clair `~/work/clair/a64fx/a64fx/llm-guided-opt/sim-accuracy/glm53f/`:** measurement framework; `STATUS.md` holds every table.

  | File | Purpose |
  |---|---|
  | `bench.c` | L0/L1 kernels, load probes |
  | `chain.c` | L2 per-rank decode layer chains; `out/chain-omp` is the fcc OpenMP build with production barriers |
  | `gemm.c` | prefill GEMM |
  | `lprobe.c` | L1 load/SDOT probes |
  | `hwrun.sh`, `run_*.sh` | runners |
  | `report.py`, `chain_report.py`, `gemm_report.py`, `compare.py` | reports and native-vs-sim comparison |
  | `measurements/hw-20260927{,b}/` | raw logs and hashes |

- **QLAIR:** `~/work/clair/a64fx/build-inference/qlair` (clang 21, `-DCMAKE_DISABLE_PRECOMPILE_HEADERS=ON`). Run as `qlair --profile --profile-markers --a64fx-backend event --cores N -n 4e10 ELF -- args`.

## Current state (all committed)

**Decode, per-rank token GEMV chain** (270 dependent GEMVs, 48 threads, 12-node shapes):

| Configuration | ms/token | % of node roof |
|---|---:|---:|
| v0 production kernels | 8.68 | 22% |
| Panel kernels + OMP hardware barrier + 16 KiB next-stage prefetch | 2.92 | 59% |
| Panel kernels + split-phase flag barrier + 16 KiB split prefetch (`chain p16 ... flag16`) | **2.75** | **63%** |

- **Per-kernel, 12 threads:** Q8_0R16 89%, Q4_KP16 84%, Q6_KP16 79%, Q5_KP16 57% (FP-op bound). Head as Q8_0R16: 92% at 48 threads.
- **Synchronization is a major term.** The atomic spin barrier costs 7.4 µs; the Fujitsu hardware barrier (FLIB_BARRIER=HARD) 1.4 µs; the flag barrier 1.3 µs.
- **XOS 2 MiB pages cost 1.8×** on per-thread slices (`XOS_MMM_L_HPAGE_TYPE=none` for chain-omp).

**Prefill GEMM** (64×6 tile, scale blocks 32/128/256):

| Configuration | sb 32 | sb 128 | sb 256 |
|---|---:|---:|---:|
| One core, L1 | 48% | 71% | 77% |
| 48 threads, K=4096 × 48 tokens | 35% | 48% | 52% |

- The reference pure-int8 6×4 loop reaches 89%. ≥90% needs per-channel/per-token scaling (lossy) and is still capped near 89%.
- The multi-core drop (63% → 52%) is unexplained. It is not weight residency, K chunking or bandwidth.

**QLAIR** (commits `da50dea8`, `fc4c9efd`, `ffe531b3`):
- **Fixed:** indexed SVE ops, LD1RQ imm, event SVE pipes.
- **Accuracy is still not <3% for GLM53F kernels.**

  | Case | Event backend error |
  |---|---:|
  | L0 v0 kernels | −5 to −24% |
  | Panel kernels | −28 to −50% |
  | GEMM tile | −15 to −21% |
  | Load+SDOT probes | −4 to −5% |
  | Dependent base-register ADD loops | −8% |

  f32 streaming passes (+2.1%).
- **Native laws established:**
  - L1 loads issue 2 per cycle.
  - A dependent base-register ADD costs about 1 cycle per 8-load block. The cost shrinks with more independent loads before the next load (see STATUS matched-density table).
  - Loads and SDOTs co-issue at about 2.8 Z-writing instructions per cycle, commit-limited. See the commit histograms in STATUS.

## QLAIR event-backend fixes this session (clair commits 33fef290, 91d4f159, 2f25f07e, + station parity)

All are event-only, backed by native probes (`lprobe.c`), and recorded with data in STATUS.md:
1. **SDOT/UDOT (incl. indexed), indexed FMLA/FMLS, integer MLA/MLS/MAD/MSB now read Zda.** Loop-carried SDOT chains had no dependence at all (native 40.8 vs event 14.5 cycles/block).
2. **AdvSIMD MOVI's phantom rn/rm (z0) source is dropped.** The decoder zero-initializes rn/rm.
3. **FLA-only forms per the Fujitsu table and 8-stream probes:** AND/ORR/EOR #imm, DUP/CPY scalar and imm, FDUP/FCPY/DUPM, SPLICE, TBL, ZIP/UZP/TRN, UNPK, DUP-indexed, INDEX. SEL (vectors) is FL*.
4. **Phantom rm=z0 source is dropped for SVE forms without Zm** (immediate/unary/extend/UNPK/REV/SPLICE...).
5. **RS0/RS1 parity uses each op's ordinal among same-class ops in its decode group** (`station_class_parity`). Raw slot parity put all loads of alternating ld/sdot on EAGA: native 5.34 vs event 6.34.

**GLM53F L1 kernel gate** (same ELF, 30 native samples): mean |error| 5.9–6.8%; 3–5 of 10 within 3%, depending on which masking errors the fixes removed. See `compare-acc*.json`.

- **Q8_0R-family kernels match:** v0 +1.0%, Q8_0R16 v3 −0.6%, v4pf −0.2%.
- **K-quant kernels are all simulated too fast (−4 to −17%).** The best-isolated remaining term is SDOT consuming freshly loaded registers: native 6.17 vs 5.32 cycles/block at lag 0, 4.92 vs 4.67 at lag 1. Natively, cycles = 4 + STALL_BACKEND (0x24). QLAIR does not emulate events 0x23/0x24.

## Session 2026-09-28 02:40–04:10 (clair commits 3650b130, 71401b32, c786a15c, d8852aec)

All the new simulator switches are diagnostic and default off, so default QLAIR behaviour is unchanged. Regressions pass: test-qlair 610/610, event CTests 2/2. Full tables are in `glm53f/STATUS.md`, in the last three sections.

**1. RSE release at completion** (`QLAIR_SIM_EVENT_RSE_RELEASE_COMPLETE=1`)
- The GEMM tile goes from −8% to −2.8%.
- The ld8sd16 probe overshoots (+5.8%), and the gate is unchanged.
- Not promoted.

**2. The "lag-0 stall" is FP rename exhaustion** (`QLAIR_SIM_EVENT_FP_RENAMES=N`)
- Every ld+SDOT probe moves monotonically with the pool size; loads-only and ld+ADD probes do not move.
- At N=88 the probes fall within ±1%.
- Q8_0R16 overshoots (+4%).

**3. Native window-size probes** (`glm53f/wprobe/`: two pointer-chase misses N fillers apart)
- Knees: ROB 128, GPR 64, FP renames 96. These are exactly the model's counts, so do not shrink the pool.
- LD1B fillers knee at 38/39. That is 40 fetch ports, and a completed younger load's port waits for the older miss. This is the existing diagnostic `QLAIR_SIM_EVENT_ORDERED_FETCH_RELEASE=1`, not the default.
- An FMUL-chain head blocker with LD1B fillers knees near 90, so ports are **not** held to commit.
- Ordered release is neutral on L1-resident cases. It should matter for HBM streaming and is the promotion candidate.

**4. Rename-lifetime switch** (`QLAIR_SIM_EVENT_FP_HOLD_LOAD=D`, `QLAIR_SIM_EVENT_FP_HOLD_FL=D`; FP renames held D cycles past commit)
- LOAD=4 makes every ld+SDOT probe exact (noadd sd8/12/16 and lag0/lag1 all 0.0%).
- It keeps the Q8_0R family in gate (−0.1/+2.3/+1.1).
- Kernel gate: 3/10, mean 6.9% → 6.6%.
- This is the best single rule so far. It is not yet backed by a direct native lifetime probe.

**5. Residuals that no rename or port rule moves**
- q4k_v0 −15%, q5k_v0 −11%, q6kp16_v1pf −15%, f32_v0 −4.3%: a separate missing cost, probably the unpack ops.
- GEMM sb32 −11 to −15%.
- Sim random-miss latency is about 455 cycles/iter vs native 290 (16 MiB chains).
- `wp fm` runs +17 cycles/iter slow in the sim (161 vs 144), while a minimal FMUL-chain + LD1B loop matches.

## Session 2026-09-28 08:13–09:20 (clair commit af095eba)

- A matched FMUL-head window probe with LD1B versus FADD fillers has the same
  native 88→92 knee and 144→159 cycles/iteration. It confirms rename capacity
  but cannot resolve a four-cycle post-commit hold: the 144-cycle head hides
  that delay. `FP_HOLD_LOAD=4` remains diagnostic and default off.
- Remeasured all five full-size one-core HBM native cases with frozen
  `bench-c13`; correctness passed and CV is at most 0.24%. Exact cycles and
  raw logs are in the new `STATUS.md` section and `measurements/hw-20260927b/`.
- A smaller same-ELF simulator campaign finished with three samples per case.
  Q8 and F32 rotate more than 8 MiB of weights, while Q4 is an L2 control:

  | Case | Native cycles | Event cycles | Error |
  |---|---:|---:|---:|
  | Q8 256×4096 HBM | 348,340 | 203,998 | −41.4% |
  | F32 128×4096 HBM | 299,340 | 282,835 | −5.5% |
  | Q4 256×4096 L2 | 297,670 | 246,217 | −17.3% |

- `QLAIR_SIM_EVENT_ORDERED_FETCH_RELEASE=1` changes none of those cycles.
  It remains diagnostic and default off. The Q8 HBM error is a separate large
  memory-model gap: native compulsory read rate is 3.4 B/cycle versus 5.8
  B/cycle implied by the simulator. The Q4 L2 residual is separate.
- The full-size simulator replay was stopped after more than twenty minutes
  without its first two cases finishing; the compact campaign supplied the
  paired comparison instead. No simulator default or production kernel changed.

## Session 2026-09-28 09:20–10:20 (clair commits b7f89f30, 5ee1def3)

- Native Q8_0R HBM row sweep used the frozen `bench-c13` ELF on the quiet
  second node. The rotating weight footprint stayed near 9 MiB, over one
  CMG's 8 MiB L2. All cases verified correct. At 64/128/256 rows, native
  hardware cycles were 88,480 / 172,840 / 348,094 (CV 1.9/1.8/1.7%).
- Native `LD_COMP_WAIT_L2_MISS` (PMU 0x0180) scaled with rows: 51,610 /
  100,628 / 204,471 cycles at 64/128/256 rows. L2 stream prefetches
  (0x0233) scaled similarly: 1,044 / 2,056 / 4,004. The simulator's compact
  256-row profile has 3,710 prefetches but only 71,588 modeled L2-miss wait
  cycles; the issue is not a gross lack of prefetch requests.
- The same-ELF event sweep for 16/32/64 rows gives −47.5/−45.0/−43.8%
  harness-cycle error. At 64 rows, modeled L2-miss wait is 17,913 cycles
  versus 51,610 native. This is a repeatable per-row gap; the 64-row native
  CV of 1.8% passes the comparison noise gate. The 16/32-row samples are
  noisier and are directional only.
- A diagnostic L2 prefetch-distance change (20 → 4 lines) raised the 16-row
  simulator result by 9.4% to 14,224 cycles, still 42.5% below the native
  harness result. Its 230 L2 prefetch requests were unchanged. Keep this
  diagnostic off. Next isolate demand versus prefetched HBM-line latency and
  miss overlap, using the 64-row case as the reliable timing gate.
- All case files, raw logs, comparison JSON, and simulator profiles live in
  `clair/.../glm53f/measurements/hw-20260927b/`; the new `STATUS.md` section
  has replay commands and full attribution. No simulator default changed.

## Session 2026-09-28 10:20–11:27 (clair commit 7553bbc4)

- `bench.c` now prints the `read`/`hello` roof-probe sink after timing, so
  the computed result is observable. The native `bench-c14` build has the
  intended SVE loads in disassembly; its 9 MiB checksum matches an independent
  calculation and its import allowlist is clean.
- A 9/10/64 MiB load-only sweep did **not** give a usable HBM miss control:
  the 64 MiB case ran in 845,412 cycles but the PMU reported zero L2-miss
  completion wait and zero L2 stream prefetches. The 9 MiB timing was almost
  unchanged by the sink fix (119,316 → 119,307 cycles). Do not use that byte
  rate to tune QLAIR's HBM rules. Keep the Q8 row sweep as the miss-path gate.
- The login-node clang build stalled on shared filesystem faults. Compiling
  the driver and linking `bench-c14` with clang 21 on the allocated A64FX
  node completed. Clair `STATUS.md` records the native command and
  `measurements/hw-20260927b/` has the measurement logs.

## Session 2026-09-28 13:03–13:31 (clair commits 84e710d3, 4e42a286, fb425fec; allocation ends 14:00 JST)

- Same frozen Q8_0R 64×4096 kernel, only rotating weight copies changed:
  ROT=1 (0.29 MiB, L2) is 37,050 native hardware cycles with zero L2-miss
  wait; ROT=32 (9 MiB, HBM) is 89,269 cycles with 52,675 L2-miss-wait
  cycles. CV is 0.34% / 1.75%; both verified correct. The event backend is
  33,024 / 51,258 cycles with 0 / 17,913 modeled L2-miss wait.
- The incremental HBM cost is 52,219 native versus 18,234 event cycles: a
  **33,985-cycle missing cost**. The native-versus-event miss-wait difference
  is 34,762 cycles. The L2-resident error is only 4,026 cycles. This narrows
  the Q8 gap to miss service or miss overlap, with a smaller base kernel error.
- ROT=16 (4.5 MiB) had a noisy 30-sample median because only one copy is
  warmed before timing. A 100-sample follow-up still shows some late-sample
  L2-miss wait; disabling huge pages moves it only modestly. Do not use
  ROT=16 as a clean L2 control. ROT=1 and ROT=32 are stable controls.
- The ROT=16 simulator replay completed and verified correct. It sees zero
  HBM reads and zero modeled L2-miss wait at 33,019 cycles, while native
  late samples take 43,888 cycles with 6,961 L2-miss-wait cycles and 9.75%
  CV. This is a separate cache-residency mismatch, but too noisy for timing
  calibration. Native logs, simulator profiles, and replay commands are in
  Clair `STATUS.md` and `measurements/hw-20260927b/`.
- Native L1 load-pipe PMU activity barely changes from ROT=1 to ROT=32:
  pipe-valid cycles increase by only 848 / 947 while total cycles rise by
  about 52k; pipe completion counts are unchanged. This supports data wait,
  not extra L1 pipe work, as the source of the HBM increment.

## Session 2026-09-28 18:00–21:30 (job 51969492, 4 nodes: rank 1 = quiet HW node via `hwrun.sh`, ranks 2–3 free)

Picked up the uncommitted work of a previous 4-node session. The whole Q5 variant set, the atomics rewrite and the `pfimm16k` kernel are committed.

**Bugs fixed (both silently broke native-vs-sim comparisons):**
- **QLAIR decoded PRFUM as LDUR x<prfop>** (clair 3ff111d4). `prfum pldl2keep` loaded into x2, so `gk_q8_0r16_v3pf*` failed verify under QLAIR only. It is now a hint, with a unit test. No gate or HBM case used PRFUM.
- **`qsys()` lacked SVE clobbers** (clair 2d11a55c). Linux zeroes P and truncates Z on `svc`, so a predicate live across the PMU ioctl made native SVE loads all-false. `bench.c` roofs are unaffected (checked).

**Other session work, recovered and committed:**
- Q5_KP16 v4pf (fifth bit merged: +2–3%) and Q5_KB16 (byte-expanded: 1.6× on 1 core, −15% at 48T because bandwidth-bound). Keep the compact format.
- QLAIR atomics as locked read-modify-writes. The exact count holds under functional and legacy modes (`probes/atomic_spin.c`); the event backend still rejects atomics. This is also an open N0.5 item of the Nagare plan.

**HBM/L2 stream calibration probe `sprobe/`** (clair STATUS "HBM/L2 stream calibration"):
- Native direct streams are demand-driven: about 10.5 L1 misses in flight, 156-cycle sequential vs 255-cycle random latency, and no hardware prefetch.
- The event backend is prefetch-driven and 40–47% fast on HBM streams.
- HBM latency 238 fixes random misses (−1.4%). No distance/latency pair fixes streams.
- Handed off to the **Nagare** engine effort (`a64fx-new-sim-plan.md`, another session). Don't keep fitting event-backend HBM knobs.
- Also found: store→load forwarding costs about 19 cycles natively (sim matches), and sim scalar random-load MLP looks too high.

**Prefill GEMM (native):**
- **Pad the output row stride** (`gemm.c P`): 48T sb256 58.9 → **64.1%** (15.7 TOPS), sb128 58.8%, sb32 41.5%. With ldy = 3072, output tiles fold onto 4 L1 sets. Integration rule: keep `(ldy*4/256) % 64` away from multiples of 16.
- **CMG-token mode** (`gemm.c t`) runs per-rank shapes (2304 × 4096) at 60.4%.
- **The 504-token chunk:** at 48 threads, 2304 rows reach 59%. At production's **47 threads** the 11-core CMG straggles, and balancing each CMG's token share by its busiest core's panel count takes 50.2 → **55.5%** (192 tokens: 49.1 → 57.3%). Token blocks, K chunks, a 2D row/token split, output-tile prefetch and next-chunk prefetch do not help much (kept as default-off knobs). A 768-row replica that is L2-hot reaches 66%.

## Production integration plan (Q8 trial started; needs the real model on 12 nodes)

- **Where Q8_0R is used:**
  - `glm53f_iq_bridge.c` (`glm53f_native_repack`, `_matvec_team`, `_matvec_batch_team`);
  - direct Q8_0R checks in `glm53f_kda_layer_12n.c:150`, `glm53f_sparse_layer_12n.c:443` (v_b attention weights read row-wise; do NOT repack those), `glm53f_expert_decode_12n.c:302`, and `glm53f_dense_ffn_12n.c`.
- **Plan:**
  1. Q8 trial: `GLM53F_NATIVE_Q8_0R16` is now an opt-in runtime type;
     eligible Q8 matrices are packed directly from GGUF at load, `xd` is
     prepared once, and 16-row decode groups call `gk_q8_0r16_v3pf16k`.
     One-token and batch correctness plus a matrix microbenchmark pass.
  2. Run a 12-node real-model token and tok/s A/B before enabling the panel
     by default. Check the production page placement at the same time.
  3. Keep row-wise attention V weights in their original layout.
  4. Integrate the routed-expert Q4_KP16/Q5_KP16/Q6_KP16 paths in `glm53f_iq_expert_weighted`.
- **Barriers:** production already uses `FLIB_BARRIER=HARD`. The flag barrier with split-phase prefetch helped the chain a further 6%.
- **Pages:** check weight page placement (2 MiB XOS pages shared across CMG slices cost 1.8× in the chain).

## 2026-09-29 continuation

Q8_0R16 production decode is opt-in via `GLM53F_NATIVE_Q8_PANEL=1`. The
8192×4096 one-node microbenchmark at 47 threads measured 68.08 vs 65.21
Gweights/s with inherited XOS pages, and 300.16 vs 161.02 with 2 MiB pages
disabled. The focused numerical checks pass, but this is not a 12-node model
result. See `kern/KERNELS.md`. Clair gained `glm53f/sprobe/random_line.c`, a
same-ELF scalar/SVE random-line control; its 4/16 MiB native/event comparison
is in `glm53f/STATUS.md`. The event HBM path remains uncalibrated.

## Session 2026-09-30 – 2026-10-01: 12-node integrated optimization (glm53f commits 1e4e2317..d283eed8, pushed)

Real model (UD-Q4_K_XL, 12 ranks × 47 threads, compact `NODE_SPEC=2x3x2` allocations). All numbers are end-to-end
on 12 nodes.

| | start | now (HEAD d283eed8) |
|---|---|---|
| Decode, 128 steps | 20.9–24.5 tok/s | 38.1–38.8 tok/s (26.0 ms/token) |
| Prefill, 8k prompt | 41.8 tok/s | 285 tok/s (301 with load-time prewarm, see below) |

Decode per token (ms): attention 11.2, FFN 9.4 (router 1.6, routed+shared 4.1, allreduce 2.7), mHC 4.9, head 0.44.
Prefill per position (ms, 8k): KDA 0.66, sparse 1.2 (MLA 0.57, allreduce 0.27, front 0.25, index 0.21), MoE 1.24, mHC 0.25.

**Landed (default on; each has an env switch to disable, and the exact `check` gate pins it off):**
- Prefill MoE: grouped native Q4_K/Q5_K/Q6_K GEMM through the panel64 int8 tile (`GLM53F_MOE_NATIVE_GROUPED`), router GEMM,
  shared expert GEMM, whole-chunk MPI combine.
- Prefill attention: batched native MLA (`GLM53F_SPARSE_MLA_BATCH`); KDA column recurrence + int8 panel GEMM projections
  + vector conv + 64-token tile (`GLM53F_KDA_GEMM`, `GLM53F_KDA_TILE_TOKENS`); sparse q_a|kv_a, q_b, o_proj on the same
  panel GEMM (`GLM53F_SPARSE_GEMM`).
- Async KDA tile reduction on the spare core 59 (`GLM53F_KDA_ASYNC`): KDA 0.81 → 0.66 ms/pos, tokens identical.
- Multi-TNI uTofu allreduce (`--prefill-collective mtni`, and decode `GLM53F_MTNI_DECODE`).
- Decode: full-width SVE Q4_K/Q5_K rows (`glm53f_iq_fast.h`, `GLM53F_IQ_FAST`) fused with the native shared expert in one
  region (`GLM53F_MOE_FUSE_SHARED`); single-team mHC (`GLM53F_MHC_FAST`) with bit-identical SVE Sinkhorn; fused sparse
  front (`GLM53F_SPARSE_FUSE_FRONT`); KDA early weight prefetch (`GLM53F_KDA_PREFETCH`); Q8_0R prefetch 16 KiB ahead.
- NUMA: `MPOL_INTERLEAVE` over the compute CMGs (`GLM53F_NUMA_INTERLEAVE`) and a per-CMG re-touch of the lm_head table
  (head 2.37 → 0.45 ms).
- `GLM53F_PREWARM=1` (set by the `generate` runner) builds the KDA/sparse panel copies at model load. This moves ~1 s of
  one-time setup out of the timed prefill (285 → 301 tok/s on 8k); steady-state per-token speed is unchanged, so do not
  compare 301 with older prefill numbers.
- MTP speculative decode runs (`run_glm53f_q4_mtp_12n.sh`), alpha 0.77, greedy-equivalent, but at 27.8 tok/s it is slower
  than plain decode: the batched verify is per-token in KDA/MoE.

**Tried and left off / reverted:**
- `GLM53F_SPARSE_ASYNC=1` (async sparse o_proj reduction with MPI for the other tile collectives): stalls once the
  context passes ~2k tokens (likely concurrent MPI + raw uTofu on one TNI).
- `GLM53F_PF_PLAN=1` (prefetch next-stage weights during mHC): attention/FFN −1.7 ms but mHC +1–3 ms; net neutral.
- mHC next-site prefetch, 256 B-aligned mHC slices (synthetic −6%, real decode worse), `GLM53F_MHC_FAST=2`,
  CMG-affine expert placement, scalar-pipe Q4_K min term, thread count / soft barrier sweeps: no gain.

**Lessons:**
- The async-KDA "hang" was starvation: a pthread created by the bound OpenMP master inherits core 12. Pin helpers to
  `first core + omp_get_max_threads()`.
- `__builtin_prefetch(p, 0, 1)` (L3 keep) does nothing on A64FX; use locality 0 or 2. Q8_0R wants 16 KiB look-ahead.
- Synthetic single-node benchmarks predict kernel changes well (MoE rows, Q8 prefetch) but not mHC/barrier-structure
  changes; always confirm on the 12-node decode.
- Decode is latency/sync bound (~18 µs fixed cost per matvec stage × ~270 stages); prefill TP comm has a floor of about
  0.57 ms/pos (≈1750 tok/s cap) at one-link Tofu ingest.

**Gate:** `run_glm53f_12n.sh check` passed at 8dac6c41 with the fast paths pinned off. It was not rerun after the sparse
GEMM / prewarm commits (those were validated by identical generated tokens on 8k).

**Benchmarks added:** `bench_glm53f_iq_decode.c` (decode MoE step + read floor + prefetch sweep), `bench_glm53f_native_q8.c`,
`bench_glm53f_mhc.c`, `bench_glm53f_allreduce_12n.c`, `bench_glm53f_async_reduce.c`, `bench_glm53f_moe_{native,q4,grouped}.c`,
`test_glm53f_iq_fast.c`. Run synthetic benches with `OMP_WAIT_POLICY=active FLIB_BARRIER=HARD`.

## Next steps (priority)
1. **Simulator.**
   - Memory model: the Nagare engine owns it now. Feed it `sprobe` cells0 (native direct streams, random, L2) plus the PMU groups. The event backend's candidate `HBM_LATENCY=238` is physically backed but not promoted.
   - Event core: scalar random-load MLP (LCG `ldr` loop: 11 sim vs 29 native cycles/iter; `/local/stf3.*` repro described in STATUS). Then the K-quant unpack residual and the ld8u6 base-ADD rule.
   - Keep `FP_HOLD_LOAD=4` diagnostic until a direct lifetime probe supports it.
   - Acceptance: `measurements/hw-20260927b/eval_probes.sh`, the kernel gate (`compare.py`), and `sprobe/cellcmp.py`.
2. **Decode (38.5 tok/s).** Remaining levers: multi-token verify kernels so MTP pays off (verify(2) costs ~2× one
   step today); merge the router into the mHC region (~0.4 ms); vectorise the mHC post step; fewer stages per layer.
   100 tok/s needs a persistent per-layer kernel or similar, not more incremental fusions.
3. **Prefill (285 tok/s steady state).** Rerun the exact `check` gate on HEAD. Next: sparse async reduction without
   the MPI/uTofu conflict (one collective owner), MoE combine overlap, MLA (0.57 ms/pos). Beyond ~1750 tok/s the
   TP allreduce layout itself has to change.
4. Re-run the qwen38 nine-case event gate; its static cross ELF is off-node.

## Gotchas
- **Staging** a new 12-node job takes 20 min to 4 h (Lustre load); use the wide-striped copy `~/models/glm53f-gguf-wide/`.
  The interactive job lasts 6 h; check `pjstat` start time rather than the queue estimate.
- **Fugaku FS lags rsync'd files:** compile only after the file is visible (md5) and use fresh script names.
- **Only one mpiexec per allocation** ("plexec must be started sequentially"); kill leftovers before the next run.
- **XOS paging:** export `XOS_MMM_L_PAGING_POLICY=demand:demand:demand` (`hwexec.sh` does).
- **Atomics under QLAIR:** they work in functional and legacy modes since d00b417c, but the event backend rejects them. Harnesses still skip spin barriers when CNTFRQ == 2e9.
- **SVE and syscalls:** any inline `svc` must clobber z0–z31/p0–p15 (`qsys.h`), or native SVE state is silently lost.
- **Run each sim ELF once with verify=1** (`run_sim.sh` now fails on a bad verify). The PRFUM bug hid behind verify=0.
- **zsh:** `set -- $c` does not word-split. Use `${=c}` or bash scripts.
- **Import allowlist:** keep `make check-imports` clean (`allowlist.txt`).
- **The clang driver is slow here** (about 2 minutes per link). Don't wrap `make` in a short `timeout`: a killed make leaves stale binaries.
- **Z-register spills** (`objdump | grep 'str z'`): use asm for 24-accumulator tiles.
- **Freeze ELFs per campaign** (`out/*-cN`), so native and sim runs use the same hash.
