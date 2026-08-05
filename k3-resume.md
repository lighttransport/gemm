# K3 resume — state, TODO, and what to optimize next

Rewritten 2026-08-06 during 12-node interactive job `50008795`. The previous
contents (72-node probe notes, the 12-node cache-matrix coverage index, and the
Laguna S-2.1 sections) are preserved in commit `dda47745` and are still valid
for those subjects — nothing below supersedes them.

**Three files, don't confuse them:**

| file | subject |
|---|---|
| `k3-resume.md` (this) | overall state, TODO, next optimizations |
| `@resume-k3.md` | the 96-node job, the GATE CHECK root cause, submit lines |
| `a64fx/k3/ROOFLINE.md` | every measurement behind the numbers here |

`a64fx/k3/logs/quant-bench-live12-50000128/SUMMARY.md` has the kernel-pass
detail but **`logs/` is gitignored**, so it is not in the repo.

---

## Environment facts that cost time if forgotten

- This runs on a **compute node**. `pjsub`/`pjstat` are not on PATH — use
  `ssh login1 '<cmd>'`. A non-interactive ssh starts in `$HOME`, so batch
  submissions need an explicit `cd` or pjsub cannot open the script.
- `/local` is wiped when the allocation changes. Re-stage bench blobs with
  `k3_gguf_stage.py` (≈15 s/layer) after any restart.
- **48 OMP threads is a cliff, 47 is optimal** — monotonic improvement from 24
  to 47, then +45% at 48. Every measurement taken at 48 is invalid.
- Measurement hygiene: discard the first run (cold), take best-of-N, and use
  `--prefill-tokens 512` (512 samples) for anything under ~30%. Single readings
  produced three wrong conclusions this session.

---

## Current state

### Layer time (12 nodes, expert-TP, 47 threads, real weights)

| | KDA (69 layers) | MLA (24 layers) |
|---|---|---|
| **now** | **~1.49 ms** | **~1.73 ms** |
| this morning | 4.30 | 4.30 |

Extrapolated: 69×1.49 + 24×1.73 = **144 ms/token ≈ 6.9 tok/s**, from 457 ms at
the start of the day. Phase split at 47 threads (before the last two commits):

| phase | KDA | MLA |
|---|---|---|
| attention | 0.481 | 0.909 → **0.67** after the MLA fix |
| moe | 0.935 | 0.939 |
| — dispatch / expert / shared / dispatch_proj / finish / collective | 0.244 / 0.212 / 0.190 / 0.175 / 0.171 / 0.126 | ≈same |

### Full model

Only one end-to-end measurement exists, and it predates all of today's work:
job `49931198`, 96 nodes, **prefill 1.709 / decode 1.699 tok/s**.

### Practical peak (see ROOFLINE.md for derivation)

- Decode **42–48 tok/s** at 96 nodes, **comm-dominated**: 186 collectives/token
  at ~107 µs each is ~20 ms against ~4.2 ms of weight streaming.
- Prefill **911 tok/s** compute ceiling (210.8 GFLOP/token at 2 TF/s × 96).

### Measured hardware constants

- Node read ceiling **726 GB/s** at 48T (42 / 466 / 709 / 726 for 1/12/24/48).
- bf16 matvec **652 GB/s = 90% of R**. Every quantized kernel is at 2–11%:
  MXFP4 212 Gmac/s, IQ1_S 82, Q8_0 74, IQ2_XS 58. **The quantized path is
  compute-bound, and MXFP4 — which nobody has optimized — beats our
  six-times-optimized IQ1_S by 2.6× per mac.**
- Collectives are **latency-bound**: 106.9 µs flat at the 43 KB MoE payload,
  88.6 µs hierarchical 2D (A=3×B=4). Halving the payload to bf16 changes it by
  1.9%, against 6.8 µs of wire time.

### Quantized kernels (single-layer projection proxy, not model rates)

IQ1 package 141 tok/s, Q2 120 tok/s, from 20.8 this morning. **But these
kernels have no end-to-end path**: `k3_gguf_stage.py` writes a `K3GGUFV1`
manifest read only by `k3_gguf_layer_bench.c:43`, while `k3_full_runner.c:679`
reads `K3FULLV1`. The runner already parses `IQ1_S`/`IQ2_XS`/`IQ2_XXS` dtypes —
nothing produces a manifest that feeds them.

---

## Queue

| job | what | state |
|---|---|---|
| `50005954` | 96n full-precision, 3 h, 87Gi, corrected `-x` | QUE |
| `49996689`, `49997180` | IQ1 GGUF text smoke, 32n | QUE |
| `50008795` | this 12-node session, 6 h from 00:45 | RUN |

Estimated start times are pessimistic — 96-node canaries estimated 08/10
backfilled the same evening.

---

## TODO

1. **Verify the MLA attention change in generation mode.** `layer12` reports
   `tokens=0`, so its output hash does **not** cover the attention result. The
   change rests on `make test`'s `[mla-parallel]` (2.794e-09 vs reference) plus
   a hand-checked call-site mapping. Run something that actually generates
   tokens before this reaches the 96-node job.
2. **Inspect `50005954` when it lands.** Require `stage_timing.tsv` rows for
   build / topology / barrier_preflight / full_weight_staging /
   full_short_generation **and validation**, all `rc=0`, non-empty
   `validation.txt`, and `K3FULLV2 status=PASS`. Report real decode tok/s; the
   1.699 from `49931198` is the number to beat.
3. **Trace `comm_deterministic`.** `49931198/run.rank.*` shows
   `tp_ar: ... deterministic=0` although the job passed `K3_COMM_DETERMINISTIC=1`.
   Do not draw checksum conclusions until this is explained.
4. **Bridge GGUF → `K3FULLV1`** so the IQ kernels are reachable end-to-end.

---

## Optimization opportunities, ranked by evidence

1. **MoE, ~0.93 ms of both layer types.** The remaining grind to <1 ms/layer.
   No sub-phase exceeds 26% of it, so this is six stages of 0.12–0.24 ms, not
   one fix. ~110 µs of it is barrier (six sync points × ~18 µs at 47 threads).
2. **`ar_groups`.** The runner uses 2 at 12 nodes → 2D A=2×B=6 = 93.9 µs.
   A=3×B=4 measured 88.6 µs. One flag, ~5 µs × 2 collectives/layer.
3. **MXFP4 expert kernel.** It is the fastest quantized kernel in the tree at
   212 Gmac/s and has never been optimized, while six passes went into IQ. It is
   also what the *original* checkpoint actually uses, i.e. what job `50005954`
   will run.
4. **Node count as a decode lever.** Decode is comm-bound, so IQ1's value is
   fitting in 24–32 nodes rather than 56–96 — fewer collective hops. Quantify
   with collective latency vs node count.
5. **Prefill chunk.** The one measured run used `chunk 64` on a 256-token
   prompt, the worst case. The model's table shows chunk 1024 → 140.8 vs
   chunk 64 → 125.5 tok/s. Job-script change, not code.

### Measured and rejected — do not retry without new evidence

| attempt | result |
|---|---|
| persistent OpenMP team per layer | +1.2%, inside noise. Reverted. `omp for`/`omp single` carry implicit barriers, so merging regions relocates sync rather than removing it. |
| Q8_0 rows-in-lanes repack | 4.73× at 1 thread, **0.91× at 47**. Reproduces `WS3_GEMM_findings.md`. Not integrated. |
| merged grid+sign LUT for IQ2_XXS | −9%; the 2 KiB grid + sign multiply beats a 256 KiB table |
| the reverse for IQ2_XS | −7%; the two formats want opposite choices |
| 8-row Q8_0 blocking | −20%, kept behind `K3_Q8_ROWS8=1` |
| `svtbl` for IQ grid lookup | inapplicable — grids are 256–65536 entries; svtbl permutes within a vector |
| `threads<=1` serial fast path in `k3_quant_matvec_ws` | −6% two different ways |

**The pattern worth internalizing:** today's wins were a thread default (45%),
schedule-chosen-by-batch-shape (51%, and it recurred twice more), and a kernel
that already existed but was not being called (35%). The two most invasive
things attempted measured 1.2% and −9% and were reverted. Measure a one-line
change before writing a hundred.

---

## Resume prompt

```text
Resume K3 work in /vol0006/mdt0/data/hp250467/work/gemm/k3. Read k3-resume.md,
then a64fx/k3/ROOFLINE.md for the measurements behind it and @resume-k3.md for
the 96-node job.

Environment: this is a compute node, so pjsub/pjstat need `ssh login1 '<cmd>'`
with an explicit cd. /local is wiped between allocations; re-stage bench blobs
with k3_gguf_stage.py. Use 47 OMP threads, never 48 — 48 is a 45% cliff.

Measurement discipline is the main lesson of the previous session: discard the
first (cold) run, take best-of-N, use --prefill-tokens 512 for anything under
30%, and check a one-line change before writing a hundred. Several confident
structural hypotheses were killed by measurement; the rejected list in
k3-resume.md is there to stop them being retried.

Highest-value next steps, in order: (1) verify the MLA parallel-attention change
in a mode that actually generates tokens, since layer12 reports tokens=0 and its
hash does not cover attention; (2) inspect job 50005954 when it runs and report
real full-model prefill/decode tok/s against the 1.699 baseline; (3) grind the
MoE phase, which is ~0.93 ms of every layer with no single dominant sub-phase.

Regression gates for any change: `make -C a64fx/k3 test` exits 0, and
`run_k3_ep.sh --mode dummy --layers 1 --layer 3 --cache-tokens 16384
--mla-cache-int8 --threads 47` reports checksum=+3.650037202e+02 with 12/12
pass markers. For layer-level work, `K3_PROFILE=1 ./run_k3_full_12n.sh --mode
layer12 --layer-index 2|3 --expert-tp --prefill-tokens 512` gives the per-phase
profile; layer 2 is KDA (69 layers), layer 3 is MLA (24).

Use fcc/FCC natively on A64FX. Preserve unrelated dirty workspace changes and
do not push.
```
