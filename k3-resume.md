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
| **now** (all flags, below) | **~1.153 ms** | **~1.420 ms** |
| stock defaults (`row-aligned`) | 1.410 | ~1.632 |
| before `row-aligned` | 1.51 | 1.71 |
| this morning | 4.30 | 4.30 |

69×1.1534 + 24×1.4197 = **113.7 ms/token = 8.80 tok/s** (stock: 7.33), 1.20×.

```
K3_MOE_SHARD_LAYOUT=row-aligned K3_CMG_REPLICATE=1 K3_BF16_PV=1 K3_SITU_FAST=1
```

All three new flags default **off**. `K3_CMG_REPLICATE` costs +4.14 GB/rank at
96n; `K3_SITU_FAST` is an **accuracy** change (gated by `make test`'s
`[situ-fexpa]`, max_abs 1.457e-03 vs 2e-3 tolerance) and belongs to whoever owns
output quality — note the 16 routed experts already use that approximation, so it
makes the two expert paths consistent rather than introducing a new one.

**The 1.6 TB "non-quantized" model is `bf16 + MXFP4`.** `config.json` gives 2.72 T
expert params, which would be 5.4 TB at bf16 against a 1.5 TB checkpoint — the
experts are natively MXFP4. It is already the fastest decode format we have, so
**no quantization work is on the decode critical path**; TODO 4 below is a
capacity item, not a speed one.

Phase split at 47 threads (measured before `row-aligned`; see `ROOFLINE.md` for
the row-aligned table):

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
- bf16 matvec **652 GB/s = 90% of R** — but that is a large standalone matrix.
  **At the runner's actual shapes the same kernel delivers 174 GB/s, 24% of R**,
  and the gap is not instruction issue (see the pv result). Do not plan against
  the 652 figure. Every quantized kernel is at 2–11%:
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

1. **MLA data race: FIXED.** `full_mla_forward`'s `o_proj` read `m->attn` while
   writing `out` — the same buffer, since the caller passes `out == m->attn`.
   47 threads writing `out[0..7167]` clobbered `attn[0..1023]` mid-read, so
   **24 of 93 layers computed intermittently wrong results**, including in the
   96-node job. Fixed by staging through `m->tmp` as KDA already does; MLA is now
   reproducible (`0d8160ab3188f259` x3 where it was a different hash every run).
   KDA was immune only by accident of buffer choice. The outstanding
   "verify the MLA attention change" item is now actually possible.

   **`output_hash` is not a gate** — `layer12` runs with `generated_tokens=0` and
   emits the same `14650fb0739d0383` for KDA *and* MLA layers. Use `hidden_hash`.
   Every "hash-identical" claim written before 2026-08-06 ~11:00 was checked
   against the wrong field; the race above is exactly what that let through.

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

Re-ranked 2026-08-06 06:00 against the phase table in `ROOFLINE.md`
(layer 2, row-aligned, 1.419 ms). Bytes/rank vs the measured 652 GB/s bf16 and
106 GB/s MXFP4 rates:

1. **Turn on `K3_CMG_REPLICATE=1` for the 96-node job** (default off; check
   memory first). Per-CMG replication of the bf16 projections measures **1.081×
   on KDA and 1.068× on MLA at 12 nodes** (7.28 → 7.84 tok/s extrapolated), and
   the 12n number badly understates it: at 96 nodes a per-rank projection is
   1.835 MB — **under one 2 MB large page** — so it lands wholly on one CMG and
   36 of 47 threads read it at ~119 GB/s. `K3_CMG_FORCE` simulates that condition
   at 12n and costs **2.04× on attention**, which replication fully recovers.
   Cost: +4.14 GB/rank at 96n (27% over the ~15.6 GB of weights) — verify against
   the KV cache at target context before enabling.

   Do **not** use the `K3_CMG_LOCAL` routing path: measured a 2.5% net loss,
   because at 12n a projection is only 3-7 large pages and the 2/1/2/2 split
   imbalance exceeds the locality gain. And note `sysconf(_SC_PAGESIZE)` reports
   64 KB while the heap uses 2 MB pages: `mbind` on a 64 KB-aligned range returns
   EINVAL, and without `MPOL_MF_MOVE` it returns 0 without moving anything. Both
   are silent no-ops — always verify with `get_mempolicy`.

2. **`K3_BF16_PV=1` — already implemented, default off.** `matvec_bf16_8row_pv`
   is wired in via a load-time in-place repack (`full_bf16_pv_repack`), hash
   verified `14650fb0739d0383`. Worth **1.335× at 1 thread** but only **1.038× at
   47**, i.e. ~1.3% on the layer. Leave it off until item 1 is settled; the win
   should reappear once the phase is issue-bound again rather than fabric-bound.
   Superseded plan (kept for the reasoning): The mechanism is now
   identified and measured (`ROOFLINE.md`, last four sections). `matvec_bf16_8row`
   is **issue-bound on the bf16→f32 widen**: 21.4 GB/s single-threaded against
   58.7 GB/s for a pure read of the same bytes. `matvec_bf16_8row_pv`
   (`common/ggml_dequant.h:1531`) removes the widen and measures **1.57× at
   cols=7168** — the column count of all five attention projections. Worth
   ~0.15–0.20 ms/layer. Cost: `k3_full_stage.py` must write the pair-interleaved
   layout for exactly the pv-read tensors and never for flat-read norms/embeds.
   Reproduce with `a64fx/k3/k3_bf16_bench.c` (single thread, no allocation).

   The control that proves it: in the same layer and the same OpenMP runtime,
   **MXFP4 `moe_expert` scales ×38.0 of 47 threads (81%) while bf16 `moe_shared`
   scales ×10.6 (23%)**. So the poor scaling is not OpenMP, not barriers, and not
   the memory system — it is this one function. That retires the `perf`-based
   "46% is OpenMP synchronization" reading which sent the persistent-team work,
   the collective-count work, and the poll-spin sweep all after the wrong target.
2. **MXFP4 expert kernel, 212 Gmac/s = 14.6% of the memory roofline.**
   `moe_expert` (0.211 ms) is *at* this kernel's measured rate, so it is the one
   phase that cannot improve without the kernel improving. This is now the
   largest single structural inefficiency in the model, and it has never been
   optimized while six passes went into IQ formats.
3. **A shard layout that avoids the added collective.** `row-aligned` buys 118 µs
   of redundant streaming and hands 132 µs back as a `latent_reduce`, and still
   nets a win. Removing that collective — keeping the residual stream sharded and
   reducing only at the RMSNorms — is worth ~118 µs/layer on top.
4. **Memory placement.** First-touch vs interleave is a 2.34× swing, the largest
   leverage of anything measured. CMG-*local* partitioning with a CMG-aware
   task→thread mapping is untested and is a different thing from either arm.
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
| `--ar-groups` 2/3/4/6 | 1.498/1.520/1.511/1.500 — **noise.** The 88.6 vs 93.9 µs microbenchmark gap is real, but there are only ~2 collectives/layer, so ~10 µs sits under a ±5% run-to-run spread. |
| `K3_COMM_POLL_SPINS` 1/2/8/32/128 | 1.500/1.498/1.484/1.495/1.516 — noise. Now swept; the default 4 is fine. |
| threads 45/46/47 | 1.654/1.514/1.524. 46 == 47, i.e. **a core is free for a comm thread at no measured cost.** |
| first-touch instead of `MPOL_INTERLEAVE` (`K3_NUMA_INTERLEAVE=0`) | 3.505/3.513 vs 1.500 — **2.34× worse.** Interleave is load-bearing. |
| `moe_shared` task-quantization theory (64 8-row tasks over 47 threads ⇒ `ceil`=2, so 32 threads should tie 47) | shared 0.289/0.208/0.191 at 16/32/47 — scales monotonically with threads, so it is **not** critical-path-quantized. |

**Collectives are not the 12-node lever.** `reduce` (0.117) + `moe_collective`
(0.125) = 242 µs of a 1495 µs layer, 16%. Under `replicated` the profile shows
`latent_reduce=0` and `router_reduce=0` — the layer was already running **2**
collectives per layer, not the 4 the code paths suggest.

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
