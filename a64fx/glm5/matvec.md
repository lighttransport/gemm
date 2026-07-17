# A64FX w8a16 matvec kernel tuning — R&D task (single node, synthetic data)

**Goal:** maximize the throughput (GB/s of weight bytes streamed) of the decode dense
int8-weight × int16-activation matvec on one A64FX node, at a *realistic working set*,
**accuracy-preserving** (bit-exact to the reference) — or, if exploring a new weight
format, gated on a stated `rel_l2` tolerance. This is a standalone single-node kernel
project: **no MPI, no cluster, synthetic matrix data**. The host is a native A64FX
(aarch64) login node; compile with `fcc`/`FCC` and run binaries directly.

---

## 1. The kernel

`glm5_matvec_int16sdot_8row` — `common/glm5_int8.h:145`. Computes 8 output rows of a
matvec `y[r] = sum_c (w[r][c]-128) * xq[c] * scale[r][group(c)]`, where:

- **weights** `w`: int8, offset-binary (stored byte = signed value + 128), row-major,
  `[rows, cols]`, group-quantized with group size `gs` (default 64, env `GLM5_DENSE_I8_GS`).
  Per-group per-row f32 `scale`.
- **activations** `xq`: int16 (per-token symmetric quant; the caller applies the scalar
  `xsc` afterward). Shared across all rows.
- **`xgsum`**: precomputed per-group `sum(xq[group])`, int64, used to fold the `-128`
  offset once (`y -= 128 * sum_group scale*xgsum`) so the inner loop can `svld1ub_u16`
  the raw byte (0..255) straight into `svdot_s64` with no per-vector subtract. Bit-exact.
- Inner structure: for each of 8 rows, `svdot_s64` over the group, then a per-group
  `svcvt_f64_s64` + `svmla_n_f64` (deferred f64 scale fold), one `svaddv` per row at the end.
  8 rows are register-blocked so the `xq` load is **reused across all 8 rows** (the reason
  for 8-wide blocking).

Representative shapes (GLM-5.2, hidden=6144): `wq_a[2048,6144]`, `wkv_a[576,6144]`,
`wq_b[qrows,2048]`, `wo[6144,arows]`, shared-expert `[sh_rows,6144]`. **Bench at `[2048,6144]`,
`gs=64`.** The kernel is shared by `glm5_i16_worker`, `glm5_gemm_int16sdot_block8`, and the
fused-decode dense path, so any win propagates to all dense decode GEMVs (qkv/o/shared).

The A64FX has 4 CMGs × 12 compute cores = 48 (use **47** threads; leave one core free), SVE
512-bit (svcnth=32, svcntd=8), HBM2 ~890 GB/s aggregate cold, 8 MB L2/CMG (~32 MB total),
**64 KB pages** (XOS large pages). Arch docs: `a64fx/doc/A64FX_ARCHITECTURE_SUMMARY.md`,
`a64fx/doc/SVE_INSTRUCTION_REFERENCE.md`, `a64fx/doc/SVE_INSTRUCTION_LATENCY.md`.
PMU/profiling (**use this** — see §6): `a64fx/doc/profiler.md` (fapp).

---

## 2. Current state (honest measurements)

All numbers from native benches with the **mandatory** methodology in §3 (all outputs
written; ≥1 GB fresh-mmap buffer; first-touch; `OMP_PROC_BIND=close OMP_PLACES=cores
OMP_NUM_THREADS=47`):

| reference | GB/s |
|---|---|
| pure HBM stream (1 word/line), 1–12 GB | ~890 (flat) |
| single-output contiguous stream (svld+svadd, 1 acc) | 885 |
| raw 8-stream load (the matvec's read pattern), 8 outputs | 208–216 |
| `svdot`-only, 8-row (no scale epilogue) | 208 |
| **full real kernel** (svdot + f64 scale + bias) | **190** |
| in-run (12-node decode, `--numa` interleave mempolicy) | 166 |

**The kernel is at ~90% of its structural ceiling.** The dense GEMV is bounded by the
**8-independent-output register-blocking structure** (~215 GB/s), NOT by HBM bandwidth
(890 available), NOT by svdot compute (svdot-only ≈ full kernel), NOT by TLB (64 KB pages),
NOT by the memory access pattern (contiguous interleaved layout gives the same ~215).

---

## 3. Benchmarking methodology — READ THIS FIRST (three pitfalls cost hours each)

1. **Dead-code elimination.** If the bench sinks only a *subset* of the 8 outputs, the
   compiler deletes the other rows' work and inflates GB/s **2–4×**. This produced false
   "SDOT-only = 442" and "prefetch is a 1.8× regression" readings. **Every output must be
   written to a memory array AND summed into a reduction that escapes** (a `volatile`
   global sink). Verify by checking that removing compute changes the number sanely.
2. **L2 / page-table warmth.** A 12.6 MB tensor over 47 threads is 3.1 MB/CMG and fits the
   8 MB L2 → best-of-N reports **L2 bandwidth (361–552 GB/s), not HBM**. A 403 MB buffer
   reads **2.3× too fast** (446 vs the true 190) because page tables + the HW prefetcher
   stay warm. **Use ≥1 GB total (cycle many tensors), and expect a flat plateau from 1–12 GB
   — that plateau is the real number.**
3. **Job-env vs native.** Native login-node benches do NOT inherit the cluster job's
   `XOS_MMM_L_PAGING_POLICY` preset, so they get true first-touch and read *faster* than the
   in-run kernel. For pure kernel R&D this is fine (native is the clean environment); just
   don't compare a native GB/s directly to an in-run GB/s without noting the difference.

**Canonical harness skeleton** (recreate — the session scratchpad is ephemeral):

```c
// fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp -o bench bench.c -lm
// OMP_PROC_BIND=close OMP_PLACES=cores OMP_NUM_THREADS=47 ./bench
#include <stdio.h>#include <stdlib.h>#include <stdint.h>#include <time.h>
#include <omp.h>#include <sys/mman.h>#include <arm_sve.h>
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
#define COLS 6144
static volatile double g_sink;                 // escape sink -> defeats DCE
int main(void){
  int nthr=omp_get_max_threads(), RPT=2048, GS=64, sb=COLS/GS;
  size_t targ=2L<<30, NTENS=targ/((size_t)RPT*COLS); int ROWS=NTENS*RPT; size_t wb=(size_t)ROWS*COLS;
  uint8_t*W=mmap(0,wb,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);       // fresh pages
  float  *S=mmap(0,(size_t)ROWS*sb*4,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
  float  *Y=aligned_alloc(256,(size_t)ROWS*4);
  int16_t*xq=aligned_alloc(256,(size_t)COLS*2); int64_t*xg=aligned_alloc(256,(size_t)sb*8);
  for(int c=0;c<COLS;c++) xq[c]=(int16_t)((c%17)-8);
  for(int b=0;b<sb;b++){int64_t s=0;for(int c=b*GS;c<(b+1)*GS;c++)s+=xq[c];xg[b]=s;}
  #pragma omp parallel for schedule(static)                                       // parallel first-touch
  for(long r=0;r<(long)ROWS;r++){for(int c=0;c<COLS;c++)W[(size_t)r*COLS+c]=(uint8_t)((r*31+c*7)&0xff);
                                 for(int b=0;b<sb;b++)S[(size_t)r*sb+b]=1e-3f;}
  double best=1e9;
  for(int it=0;it<3;it++){ double t0=now(); double sink=0;
    #pragma omp parallel reduction(+:sink)
    { int tid=omp_get_thread_num();
      for(int t=0;t<(int)NTENS;t++){ uint8_t*Wt=W+(size_t)t*RPT*COLS; float*St=S+(size_t)t*RPT*sb,*Yt=Y+(size_t)t*RPT;
        int nb=RPT/8, per=(nb+nthr-1)/nthr, b0=tid*per, b1=b0+per>nb?nb:b0+per;   // 8-row blocks over threads
        for(int bi=b0;bi<b1;bi++){ int r=bi*8; const uint8_t*w=Wt+(size_t)r*COLS; const float*s=St+(size_t)r*sb;
          float*dst=Yt+r;
          /* === CALL YOUR KERNEL HERE, write dst[0..7] === */
          sink+=dst[0]+dst[1]+dst[2]+dst[3]+dst[4]+dst[5]+dst[6]+dst[7]; } } }   // ALL 8 sunk
    g_sink+=sink; double d=now()-t0; if(d<best)best=d; }
  printf("%.1f GB/s\n", wb/best/1e9); return (int)((long)g_sink&1);
}
```

**Accuracy check** (for any candidate): run the candidate and the reference
`glm5_matvec_int16sdot_8row` on the SAME realistic random data (weights uniform int8,
activations ~N(0, ~2000) int16, scales ~1e-3), compute `rel_l2` and `max_abs` over all
rows. Math-preserving changes must be **rel_l2 == 0**. A new format states its tolerance.

---

## 4. Refuted levers — do NOT repeat (each measured honestly)

| lever | result |
|---|---|
| TLB / page size | 64 KB pages; a 12 MB tensor is 192 pages — non-issue |
| software prefetch: PLDL1KEEP / PLDL1STRM / PLDL2STRM, 512 B–4 KB | ≤ baseline (streaming prefetch, the Fujitsu STREAM trick, does nothing here) |
| interleaved weight layout (8 rows' 32 B chunks contiguous → 1 stream) | 1.09× only (215 vs 208); contiguity alone doesn't help |
| f32 scale accumulation instead of f64 | bit-accurate (rel_l2 3.7e-7) but **0.98×** — the int64→float *convert* (8-wide from svdot's int64 lanes) dominates the ~10% epilogue, not the f64 MLA |
| larger group size gs=128 / 256 | honest 8-row: 189/204/208 (~10%); decode-neutral in-run; costs accuracy |
| row blocking R = 1/2/4/8 | 167/163/161/175 — 8-row best; narrower is compute-bound (xq reload) |
| w8a8 (`svdot_s32`, int8 activations) | faster kernel but lossy; already a **measured net decode loss** in-run |

---

## 5. The core open question

**Why does an 8-independent-output blocked loop cap at ~215 GB/s while a single-output
contiguous stream hits 885, at the same load density (1 load / 64 B)?** It is the same
~215 whether the 8 streams are strided (row-major) or contiguous (interleaved layout), and
whether the compute is `svdot` or a bare `svadd` — so it is the **8-wide structure itself**,
not the layout, the compute, or the prefetch. The 8-wide blocking is *required* to reuse the
`xq` load across 8 rows; dropping to 1 output makes each row reload `xq` and go compute-bound
at 167. **Nobody has profiled WHY the 8-wide loop is memory-throttled at 215.** This is the
highest-value thing to resolve — do it with A64FX PMU counters (fapp), not by guessing.
Candidate limiters to measure: outstanding L1-miss / hardware-prefetch-queue depth per core,
L2 miss rate, the number of concurrent load streams the HW prefetcher tracks, LFB/MSHR
occupancy, and whether the per-block reduction+store epilogue stalls the load pipe.

---

## 6. Exploration opportunities (ranked)

1. **Root-cause the 8-wide 215 cap with fapp PMU** (`a64fx/doc/profiler.md`). Measure L1D/L2
   miss, memory read BW, and cycle accounting for the 8-output loop vs the 1-output stream.
   The gap (215 vs 885) is real headroom **if** the limiter is addressable (e.g., a software
   pipeline / explicit L1 staging that naive prefetch can't express, or fewer-but-wider
   accumulators). Everything below is a hypothesis this profiling would rank.
2. **A different int accumulate width.** `svdot_s64` forces int64 lanes → the slow 8-wide
   `svcvt_*_s64` convert. The group dot fits **int32** (< 2^31). Try widening int16 MACs
   (`svmlalb`/`svmlalt` into int32) or an int32 dot, then `svcvt_f32_s32` (**16-wide** convert)
   + `svmla_f32`. Might make the epilogue 16-wide and cheaper. Untried properly. Gate: bit-exact
   (int32 exact) or tiny rel_l2.
3. **Alternate register blocking.** The 8-output cap may be specific to 8. Try 4 rows × 2
   column-panels, or 2 activation vectors × 4 rows, or 6-row, to trade stream count vs `xq`
   reuse differently. Search the (rows, panels, unroll) space with the §3 harness.
4. **Double-buffered explicit staging.** Copy the next tile of the 8 rows into one small
   contiguous scratch buffer while computing the current tile (turn 8 strided streams into
   1 contiguous read the compute consumes). The passive interleaved-layout test was flat, but
   an *active* prefetch-to-scratch overlapped with compute is different — worth one try if (1)
   shows the limiter is stream tracking.
5. **A cheaper weight format (biggest end goal, needs an accuracy gate).** The kernel is
   memory-structure-bound, so **fewer weight bytes ≈ proportionally faster.** 4-bit dense
   weights (nibble unpack via `svtbl` + `svdot`) would roughly halve the bytes. This is the
   most promising path to a real decode win, but it changes accuracy — must gate on `rel_l2`
   vs the bf16 reference and, ultimately, end-to-end quality. (The 2-bit *expert* weights
   already use separate IQ kernels — see `glm5_iq_q8_row` in `common/glm5_impl.h`; those are
   a different, already-tuned code path.)
6. **bf16-weight / fp16 paths** for comparison — different data types, different FPU limits;
   useful as an upper-bound reference for what the memory structure allows.

Deliverable discipline: a "no win, here's why (with PMU evidence)" is a valid and valuable
outcome. Do NOT ship a change that doesn't clear a **≥1.3×** honest-bench margin AND survive
an in-run decode check; forcing marginal changes here is theater (four prior levers died this
way). Record every result — win or null — with the honest number and the methodology used.

---

## 7. Resuming prompt (paste into a fresh dedicated coding agent)

> You are tuning a single A64FX (aarch64) matvec kernel for maximum weight-streaming
> throughput. Read `a64fx/glm5/matvec.md` fully first — it has the kernel location
> (`common/glm5_int8.h:145`, `glm5_matvec_int16sdot_8row`), the current numbers (full kernel
> 190 GB/s, structural ceiling ~215, HBM 890), the MANDATORY benchmarking methodology (§3 —
> all outputs written to defeat dead-code elimination; ≥1 GB fresh-mmap buffer to defeat L2
> warmth; `OMP_PROC_BIND=close OMP_PLACES=cores OMP_NUM_THREADS=47`), the levers already
> refuted (§4 — do not repeat: TLB, streaming prefetch, interleaved layout, f32 scale, gs
> sweep, row blocking, w8a8), and the ranked opportunities (§6).
>
> This is single-node R&D with **synthetic matrix data** — no MPI, no cluster, no model
> weights. Compile with `fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp`.
> Work in a scratch dir with self-contained `.c` benches following the §3 skeleton.
>
> START with §6 item 1: build the §3 harness, reproduce the honest baselines (single-output
> stream ~885, raw-8-stream ~208, full kernel ~190), then use **fapp** (`a64fx/doc/profiler.md`)
> to profile the 8-output loop vs the 1-output stream and identify WHY the 8-wide structure
> caps at ~215 (L1/L2 miss, memory BW, outstanding-request/prefetch-queue depth, cycle
> accounting). Report the limiter with PMU evidence before trying any fix. Then pursue §6
> items 2–4 (int32-accumulate epilogue, alternate blocking, double-buffered staging) guided by
> what the PMU shows.
>
> Rules: keep math-preserving changes bit-exact (rel_l2 == 0 vs `glm5_matvec_int16sdot_8row`
> on realistic random data); a new weight format states its rel_l2 tolerance and needs an
> accuracy gate. Only propose a kernel change that clears **≥1.3×** on the honest bench. A
> well-evidenced "no cheap win" is an acceptable result. Log every measurement — win or null —
> with the number and how it was measured.

---

## 8. References

- Kernel: `common/glm5_int8.h:145` (`glm5_matvec_int16sdot_8row`) and the register-blocked
  GEMM variants (`glm5_int16sdot_4row_5x/4x/2x`) in the same file.
- Callers: `glm5_i16_worker`, `glm5_gemm_int16sdot_block8`, `glm5_i16g_worker`
  (`common/glm5_impl.h`), and the fused-decode dense path.
- 2-bit expert IQ kernels (separate, already tuned): `glm5_iq_q8_row` and
  `glm5_iq_q8_row_v2` in `common/glm5_impl.h`.
- Full decode-kernel history + all the refuted-lever detail: `a64fx/glm5/GLM52_Q2_12N.md`
  (§§5–12). This file (`matvec.md`) is the single-node kernel-tuning extract of §§11–12.
- A64FX arch + SVE latency + fapp profiler: `a64fx/doc/`.
