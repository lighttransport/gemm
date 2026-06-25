# Gemma-4 31B A64FX prefill kernels — standalone microbenchmarks

Self-contained kernels + microbench drivers for the Gemma-4 31B prefill/decode
compute kernels on native A64FX (Fugaku). **No model, no `transformer.h`, no gguf**
— pure synthetic data, so they compile in seconds and run directly under `qlair`
(cycle-accurate sim) and `fapp` (PMU profiler). Self-validating (each driver checks
the kernel output against a scalar reference before timing).

These are extracted from the WS1–WS3 kernel work (see
`../../resume-opt.md` and `~/.claude/plans/see-resume-opt-md-and-plan-sleepy-lovelace.md`).
The model-coupled accuracy benches live separately in `../bench_q4_0_matvec/`.

## Model fact that shapes everything
**Every weight matrix in `gemma-4-31B_q4_0-it.gguf` is Q4_0** (attn q/k/v/output,
ffn gate/up/down). There are **no F16 weights** (only F32 norms + a Q6_K lm-head).
So the int8 `q8v2` kernel is the *universal* weight×activation prefill GEMM, and the
fp16 kernel is only for attention's activation×activation GEMMs (QK^T, P·V).
Real dims: n_embd=5376, n_ff=21504, n_heads=32, head_dim 256 (SWA) / 512 (full).

## ⚠ FP16 needs FPCR.FZ16 (3× swing — fapp-confirmed)
A64FX handles fp16/fp32 **subnormals in slow microcode**. fp16 GEMM accumulands and
attention **softmax probabilities go subnormal**, so any caller of the fp16 kernel /
fused attention must set `FPCR.FZ (bit24) + FZ16 (bit19)` **per thread**:
```c
static inline void set_fz(void){ uint64_t f; asm volatile("mrs %0,fpcr":"=r"(f));
    f|=(1ULL<<24)|(1ULL<<19); asm volatile("msr fpcr,%0"::"r"(f)); }
```
Measured (fapp): fp16 12×2 @K=384 **31 % → 93.6 %** of peak; fused attention
**284 → 833 GF/s** (hd256 N768), **462 → 2128 GF/s** (hd256 N1536). The drivers here
set it by default (`fp16_profile`: env `FP16_NOFZ=1` to disable; `attn_profile`: each
OpenMP thread sets its own). **This is mandatory for the real WS3 attention integration.**

## Kernels

| Kernel (.S) | Shape | Used for | Precision |
|---|---|---|---|
| `kernel_q8v2_3x4` / `_arow` | MR=3 tok × NR=64 col, per-32-block | **all** weight GEMMs (attn proj, ffn) | int8 SDOT, per-block fp32 scale-out (Q8v2) |
| `micro_kernel_fp16_12x2_swp` (+`_accum`) | MR=12 × NR=64 | attention QK^T, P·V | fp16 FMLA, fp32-C epilogue |

`q8v2` keeps the Q4_0 weight **lossless** (centered nibble `q-8` as int8) with the
per-32-block scale exact; only the per-block int8 activation adds error (~0.4%).
`_arow` uses a per-row activation scale (factors `da` out of the K loop → one fewer
scale-op per block; pick per-block vs per-row by end-to-end argmax stability).

## Build & run (native, quick)

```sh
make            # q8v2_profile, fp16_profile, attn_profile
make run        # build + run all with default sizes

# individual:
OMP_NUM_THREADS=1 ./q8v2_profile [nb=16] [reps=2000] [arow=0]   # K = 32*nb
OMP_NUM_THREADS=1 ./fp16_profile [K=512] [reps=2000]
OMP_NUM_THREADS=48 ./attn_profile [hd=256] [N=384] [window=512] # window=0 -> full causal
```

The native drivers print `cyc/call`, `GIOPS|GFLOPS/core`, `%peak`, and `maxrel`/`relL2`.
Peaks: int8 SDOT = 512 GIOPS/core, fp16 FMA = 256 GFLOPS/core (both @2 GHz).

### Notes on the rough native numbers
- `rdcyc` timing on a shared node is noisy; **`qlair`/`fapp` give the accurate
  per-core efficiency** — that's why these are packaged for them.
- `q8v2` single-tile L1-resident ≈ **42% (per-block) / 47% (per-row)** of int8 peak,
  steady across K. (Node-level multi-thread is lower, B-bandwidth-bound — a GEPP
  blocking concern, not a microkernel one.) The ceiling is the per-32-block
  convert+scale epilogue (structural to Q4_0; OoO already hides loads — see
  `../doc/FP16_GEMM_CEILING.md`).
- `fp16_profile` calls the kernel **once over the full K** (no fp32 shadow), so its
  `maxrel` grows with K (fp16 accumulation) — that is expected and shows *why* the
  real path (and `attn_profile`) blocks K into Kc=64 chunks with fp32 accumulation
  (→ relL2 ~0.006). `fp16_profile` is a **speed** microbench only.

## Profile with fapp (17 PA groups → CSV)

```sh
make fj                                        # builds *_profile_fj (fapp markers)
bash profile_fapp.sh q8v2_profile_fj "16 200 0"   # -> prof_q8v2_profile_fj/pa{1..17}.csv
bash profile_fapp.sh fp16_profile_fj "384 200"
OMP_NUM_THREADS=1 bash profile_fapp.sh attn_profile_fj "256 384 512"
```

Region names for the report: `q8v2_3x4`, `fp16_12x2_swp`, `attn_fused`.
`pa1` = statistics (cycles/insns/GFLOPS), `pa6`/`pa7` = cache. See
`../doc/fapp_pmu_profiling.md`. Decode raw event codes via `../doc/a64fx_pmu_events.csv`.

Official clair report (optional — corroborates the manual decode):
```sh
NODE=~/local/node-v24.11.1-linux-arm64/bin/node
PREPORT=~/work/clair/clair/sim/a64fx/preport/preport.js
cd prof_q8v2_profile_fj && $NODE $PREPORT -r q8v2_3x4 > report.txt
```
Worked examples of the flow: `~/work/clair/clair/a64fx/llm-guided-opt/profile_sgemm_6x4.sh`.
Pre-decoded reports for this kernel: `prof_q8v2_pb/REPORT.md` (per-block) and
`prof_q8v2_arow/COMPARE.md` (per-block vs per-row side-by-side).

## qlair (cycle-accurate simulation)
The `*_profile_fj` binaries are plain aarch64+SVE ELF — run them under qlair as in
`~/work/clair/clair/sim/a64fx/`. Keep `reps` small (sim is slow) and single-thread
(`OMP_NUM_THREADS=1`) for clean per-core pipeline analysis; the `fapp_start/stop`
markers bound the region of interest.

## Node-level Q8v2 GEMM (multi-thread, weight streamed)
`q8v2_gemm.c` runs the full multi-thread weight×activation GEMM at gemma dims
(N=21504, K=5376, ~115 MB int8 weight) — what the microkernel can't show (it's
L1-resident). Finding: the **flat `collapse(2)` loop is already ~32 % of int8 peak**
(vs 42 % single-tile) and is **NOT weight-bandwidth-bound** (weight read-once ≈ 0.19 ms
vs ~11 ms compute) — the static schedule reuses each weight panel across m-tiles in
cache. **GEPP N-panel blocking does NOT help** (0.26–0.91×: per-panel barriers + the
panel weight exceeds one CMG's L2). The 42 %→32 % gap is multi-thread compute overhead,
not memory. So q8v2 is at its practical ceiling at both micro and node level (the
structural Q4_0 convert-scale dependency). `make q8v2_gemm && ./q8v2_gemm 21504 5376 1536`.

## Files
- `kernel_q8v2_3x4.S` — int8 Q8v2 weight GEMM microkernel (per-block + per-row).
- `q8v2_gemm.c` — multi-thread node-level GEMM (flat vs GEPP A/B; synthetic weights).
- `micro_kernel_fp16_12x2_swp.S`, `..._swp_accum.S` — fp16 12×2 GEMM (init + accumulate).
- `q8v2_profile.c`, `fp16_profile.c`, `attn_profile.c` — standalone synthetic drivers.
- `Makefile`, `profile_fapp.sh`.
