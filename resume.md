# Resume: GLM-5.3-Flash A64FX kernel efficiency + QLAIR accuracy (updated 2026-09-28 02:40)

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
   - Explain the lag-0 load→SDOT backend stall (STALL_BACKEND 1.60 vs 0.92 cycles/block); it probably comes from load-waiting FP ops filling the RSE. Emulate PMU 0x23/0x24 in QLAIR.
   - Then the dependent base-ADD rule (matched density table).
   - Re-check the K-quant L0 cases, the GEMM tile, and the qwen38 nine-case event gate.
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
