# Resume: GLM-5.3-Flash A64FX kernel efficiency + QLAIR accuracy (updated 2026-09-28 11:27)

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

## Production integration plan (not started; needs the real model on 12 nodes)

- **Where Q8_0R is used:**
  - `glm53f_iq_bridge.c` (`glm53f_native_repack`, `_matvec_team`, `_matvec_batch_team`);
  - direct Q8_0R checks in `glm53f_kda_layer_12n.c:150`, `glm53f_sparse_layer_12n.c:443` (v_b attention weights read row-wise; do NOT repack those), `glm53f_expert_decode_12n.c:302`, and `glm53f_dense_ffn_12n.c`.
- **Plan:**
  1. Add the `GLM53F_NATIVE_Q8_0R16` type.
  2. Repack only matrices consumed solely through `native_matvec_*`.
  3. Add a per-block activation scale `xd` to `native_act`.
  4. Dispatch 16-row groups to `gk_q8_0r16_v3pf` (`#include "kern/glm53f_kern_q8r16.c"`).
  5. Keep row slicing aligned to 16.
  6. Same for the routed experts: Q4_KP16/Q5_KP16/Q6_KP16 inside `glm53f_iq_expert_weighted`.
- **Barriers:** production already uses `FLIB_BARRIER=HARD`. The flag barrier with split-phase prefetch helped the chain a further 6%.
- **Pages:** check weight page placement (2 MiB XOS pages shared across CMG slices cost 1.8× in the chain).

## Next steps (priority)
1. **Simulator.**
   - Isolate demand versus prefetched HBM-line latency and memory-level parallelism. Q8 HBM has a −43.8% event error at 64 rows with 1.8% native CV; its modeled L2-miss wait is 17,913 versus 51,610 native PMU cycles. Use the frozen `bench-c13` and the new row-sweep logs in `measurements/hw-20260927b/` as the gate. The diagnostic L2 prefetch distance 20 → 4 only moved the 16-row simulation by +9.4% and stays off.
   - Design a load-result rename lifetime probe without the 144-cycle FMUL head masking release delay or FADD/LD pipe-use differences. Keep `FP_HOLD_LOAD=4` diagnostic until it is supported directly.
   - Then the K-quant unpack-op residual: q4k_v0/q5k_v0 insensitive to every switch.
   - Then the dependent base-ADD rule (ld8u6 −7.6%).
   - Then the sim random-miss latency (wprobe 455 vs 290).
   - The acceptance set is `/local`-independent: `measurements/hw-20260927b/eval_probes.sh` plus the kernel gate (`run_sim.sh out/bench-c13 <cases-acc2 with S=3> event ...`, then `compare.py --native measurements/hw-20260927b/acc2-native.log`).
2. **Decode.** Integrate panel kernels, repack at load and use flag or hardware barriers into the production runner (`glm53f_iq_bridge.c`, `glm53f_target_decode_12n.c`). Validate on 12 nodes: `build_glm53f_integrated_12n.sh check` bit-identical, then a tok/s A/B. Check production weight page placement: the 2 MiB page effect.
3. **Prefill.** Explain the multi-core GEMM drop (per-core PMU at 12 threads). Consider W8A8 per-channel behind the quality gate.
4. Re-run the qwen38 nine-case event gate; its static cross ELF is off-node.

## Gotchas
- **XOS paging:** export `XOS_MMM_L_PAGING_POLICY=demand:demand:demand` (`hwexec.sh` does).
- **No cross-thread atomics under QLAIR.** Harnesses skip spin barriers when CNTFRQ == 2e9.
- **Import allowlist:** keep `make check-imports` clean (`allowlist.txt`).
- **The clang driver is slow here** (about 2 minutes per link). Don't wrap `make` in a short `timeout`: a killed make leaves stale binaries.
- **Z-register spills** (`objdump | grep 'str z'`): use asm for 24-accumulator tiles.
- **Freeze ELFs per campaign** (`out/*-cN`), so native and sim runs use the same hash.
