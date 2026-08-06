# K3 resume — state, TODO, and what to optimize next

Rewritten 2026-08-06 10:15 after commit `ae5cae32`. Supersedes the version at
`8e49b883`; the 72-node probe notes, 12-node cache-matrix index and Laguna S-2.1
sections live in `dda47745` and are still valid for those subjects.

**Three files, don't confuse them:**

| file | subject |
|---|---|
| `k3-resume.md` (this) | overall state, TODO, next optimizations |
| `@resume-k3.md` | the 96-node job, the GATE CHECK root cause, submit lines |
| `a64fx/k3/ROOFLINE.md` | every measurement behind the numbers here |

`a64fx/k3/logs/` is gitignored, so per-run artifacts are not in the repo.

---

## Environment facts that cost time if forgotten

- This runs on a **compute node**. `pjsub`/`pjstat` are not on PATH — use
  `ssh login1 '<cmd>'`. A non-interactive ssh starts in `$HOME`, so batch
  submissions need an explicit `cd`.
- **`cd` does not persist between tool calls here, and the shell is zsh.**
  Use absolute paths, and never rely on word-splitting an unquoted `$VAR` —
  zsh does not split, so `env $FLAGS cmd` silently sets *one* variable to the
  whole string. That produced six mislabelled runs in this session.
- `/local` is wiped when the allocation changes; re-stage after any restart.
  A layer12 stage is ~4 min for one layer.
- **48 OMP threads is a cliff, 47 is optimal.** 46 ≈ 47, so one core is free
  for a comm thread at no measured cost.
- Measurement hygiene: discard the first (cold) run, take best-of-N, use
  `--prefill-tokens 512`. Only `mpiexec` one job at a time — concurrent runs
  fail with `PLE 0008 plexec must be started sequentially`.
- **Fill benchmark weights with realistic values** (`0x3f80 ^ (i & 0x7f)`).
  `memset(buf,1,n)` makes every bf16 ≈ 2.4e-38 and A64FX traps to microcode on
  subnormal FMUL — an entire benchmark read 0.6 GB/s as pure artifact.

---

## Correctness gate — read this before trusting any hash

**`output_hash` is not a gate.** `layer12` runs with `generated_tokens=0` and
emits the same `14650fb0739d0383` for KDA *and* MLA layers. Use **`hidden_hash`**
from `<result>/output.txt`. Every "hash-identical" claim written before
2026-08-06 ~11:00 was checked against the wrong field, and that is exactly what
let a live data race survive.

Known-good `hidden_hash`, layer 2 KDA, `--prefill-tokens 512`, 47 threads:

| config | hidden_hash |
|---|---|
| stock (`row-aligned`) | `6e94844067bbd6a1` |
| + `K3_CMG_REPLICATE=1` (bit-identical) | `6e94844067bbd6a1` |
| + `K3_BF16_PV=1` | `99c89d76b4452d7d` |
| + `K3_SITU_FAST=1` | `6cf26fdcdbaf52af` |
| + `K3_FAST_EXP=1` (all four) | `b1ffa8f58b4010bc` |

MLA (layer 3) is reproducible **per thread count** — 47 threads and 1 thread
give different values legitimately, because `parts = (threads+heads-1)/heads`
changes the log-sum-exp partitioning.

---

## Current state

### Layer time (12 nodes, expert-TP, 47 threads, 512 samples)

| | KDA (×69) | MLA (×24) |
|---|---|---|
| **all four flags** | **1.153 ms** | **1.420 ms** |
| stock defaults | 1.410 | ~1.632 |
| start of 2026-08-06 | 4.30 | 4.30 |

69×1.1534 + 24×1.4197 = **113.7 ms/token = 8.80 tok/s** (stock 7.33), 1.20×.
From 6.9 tok/s at the start of the day.

```
K3_MOE_SHARD_LAYOUT=row-aligned K3_CMG_REPLICATE=1 K3_BF16_PV=1 \
K3_SITU_FAST=1 K3_FAST_EXP=1
```

All four flags default **off**:

- `K3_CMG_REPLICATE` — bit-identical, but +4.14 GB/rank at 96n (27% over the
  ~15.6 GB of weights). Verify against the KV cache at target context.
- `K3_BF16_PV` — a reassociation, not an identity (`ds4f.h`'s "BYTE-IDENTICAL"
  does not hold for this integration). Max abs error vs an f64 reference
  4.17e-07 against the 8-row form's 4.77e-07, i.e. marginally *closer* to exact.
- `K3_SITU_FAST` — accuracy change, bounded by `make test`'s `[situ-fexpa]` at
  1.457e-03 against a 2e-3 tolerance. The 16 routed experts already use this
  approximation; this makes the shared/dense path consistent with them.
- `K3_FAST_EXP` — accuracy change worth only ~2.6%, and its error feeds the KDA
  **recurrent state**, so it persists across tokens. Hold this one back until
  someone checks generation quality.

**The 1.6 TB "non-quantized" model is `bf16 + MXFP4`.** `config.json` gives
2.72 T expert params, which would be 5.4 TB at bf16 against a 1.5 TB checkpoint
— the experts are natively MXFP4. It is already the fastest decode format in the
tree, so **no quantization work is on the decode critical path**.

### Phase split (layer 2, all flags, 1.153 ms)

| phase | ms | note |
|---|---|---|
| **4 collectives** | **0.448** | 38% of the layer, all at the microbenchmark floor |
| — reduce / latent_reduce / moe_collective / finish_reduce | 0.093 / 0.134 / 0.122 / 0.099 | `finish_reduce` was hidden inside `moe_finish` |
| attention | 0.330 | |
| — kda_qkv | 0.100 | 73.4 MB ⇒ **734 GB/s = 86% of roofline; done** |
| — kda_oproj / kda_grmsnorm / kda_serial | 0.056 / 0.009 / 0.008 | `kda_serial` was 0.047 before `K3_FAST_EXP` |
| moe_expert | 0.198 | access-pattern bound, see below |
| moe_shared | 0.095 | was 0.199 before `K3_SITU_FAST` |
| moe_finish (matvec part) | 0.026 | |
| dispatch_proj | 0.039 | |

Budget: **0.45 comm + 0.70 compute**. Under 1 ms/layer at 12 nodes needs ~0.15 ms
off compute. **The 12-node budget understates 96 nodes**, where compute shrinks
~8× while comm does not — the layer there should be comm-dominated at ~0.5 ms.

### Full model

Only one end-to-end measurement exists and it predates everything here:
job `49931198`, 96 nodes, **prefill 1.709 / decode 1.699 tok/s**. The gap between
that and any 12-node extrapolation is still the largest open question.

### Measured hardware constants — several older ones were wrong

- **Node read ceiling is ~854–890 GB/s CMG-local, ~460 GB/s under
  `MPOL_INTERLEAVE`.** The long-quoted **726 GB/s is a placement artifact**, not
  hardware. Every roofline percentage written before 2026-08-06 used the wrong
  denominator.
- **Inter-CMG bandwidth ~119 GB/s vs 226 GB/s CMG-local** (12 threads, per CMG).
  At 1 thread the penalty is only 1.3× — it is a concurrency ceiling, so no
  single-threaded experiment can find it.
- CMGs are **NUMA nodes 4–7, cores 12–23 / 24–35 / 36–47 / 48–59**. Thread `t`
  sits on CMG `t/12` under `OMP_PROC_BIND=close`.
- **The heap uses 2 MB large pages** even though `sysconf(_SC_PAGESIZE)` reports
  64 KB. `mbind` on a 64 KB-aligned range returns EINVAL, and without
  `MPOL_MF_MOVE` it returns 0 *without moving anything*. Two silent no-ops —
  always verify placement with `get_mempolicy`.
- **Do not plan against "bf16 matvec = 652 GB/s."** That is a large standalone
  matrix; at the runner's shapes the same kernel gives 174 GB/s before CMG
  replication and 734 GB/s in `kda_qkv` after it.
- Collectives are latency-bound: ~112 µs each in the runner at 12 nodes, against
  6.8 µs of wire time for the 43 KB payload.

---

## Queue

| job | what | state |
|---|---|---|
| `50005954` | 96n full model, 3 h | QUE, est. 08/11 15:41 |
| `49996689`, `49997180` | IQ1 GGUF text smoke, 32n | QUE, est. 08/09 |
| `50015948`, `50018103` | 12n interactive, 6 h from 06:54 / 08:34 | RUN |

**`50005954` runs `make -C "$K3" full-runner` (line 60), so it rebuilds from the
working tree** — it will pick up the MLA race fix automatically. But it runs the
*script text* snapshotted at submission, which still has `K3_PREFETCH_MIB=16`
(a measured 1.18–1.31× loss) and `replicated`. Resubmitting fixes that and
enables the new flags, at the cost of queue position. **Decision pending.**
It also means the tree state on 08/11 is what runs — don't leave half-finished
work in `k3_full_runner.c`.

---

## TODO

1. **Decide on `50005954`:** let it run (gets the correctness fix, keeps the
   prefetch regression) or resubmit (gets ~1.5× more, loses queue position).
2. **Verify the MLA attention path in generation mode.** Now actually possible —
   before the race fix, MLA output differed every run. `layer12` reports
   `tokens=0`, so use a mode that generates tokens.
3. **Inspect `50005954` when it lands.** Require `stage_timing.tsv` rows for
   build / topology / barrier_preflight / full_weight_staging /
   full_short_generation **and validation**, all `rc=0`, non-empty
   `validation.txt`, `K3FULLV2 status=PASS`. Beat 1.699 tok/s.
4. **Trace `comm_deterministic`.** `49931198/run.rank.*` shows
   `tp_ar: ... deterministic=0` although the job passed
   `K3_COMM_DETERMINISTIC=1`. Unexplained.
5. **Fix `k3_moe.h:702`** — references an undeclared `local` in the non-OpenMP
   branch of `k3_expert_tp_forward_selected_mxfp4`. Only compiles because
   `_OPENMP` is always defined.
6. **Bridge GGUF → `K3FULLV1`** so the IQ kernels are reachable end-to-end.
   Capacity item, not speed.

---

## Optimization opportunities, ranked by evidence

1. **`moe_expert`, 0.198 ms — access-pattern bound, ~22% of its own kernel's
   rate.** Not compute-bound and not at the memory roofline: the kernel does
   27.0 Gmac/s single-thread cache-resident, the model gets ~4.5 Gmac/s/thread.
   48 matrices of ~486 KB each, 16 short row-streams per thread, and each matrix
   is **under one 2 MB page so it sits wholly on one CMG**. The fix direction is
   expert→CMG placement and routing, not a faster dequant.
   *(This retires the earlier claim that `moe_expert` "is at the MXFP4 kernel's
   own rate" — that 212 Gmac/s figure was measured in this same suboptimal
   in-model condition, so it was circular.)*
2. **Overlap collectives with compute.** 0.448 ms of comm, 38% of the layer, all
   at the floor — count and latency are both exhausted (see rejected list). The
   shared expert reads `x`, not the latent, so `moe_shared` (0.095) can run
   during `latent_reduce` (0.134). Needs a comm thread on the free 48th core;
   `tp_allreduce.h` has no non-blocking primitive but is built on one-sided uTofu
   puts. Caution: the prefetch-thread precedent added jitter as well as time.
3. **`kda_oproj` 0.056 ms at 251 GB/s** against `kda_qkv`'s 734 in the same
   layer with the same weights placement. 896 tasks of 8×1024 vs qkv's 768 of
   8×7168 — the only structural difference is task size. Unexplained.
4. **`shared_down` 0.056 ms** — 7168×512, i.e. 896 tasks of 1 KB each.
5. **Prefill chunk.** The one measured run used `chunk 64` on a 256-token prompt,
   the worst case. Model table: chunk 1024 → 140.8 vs chunk 64 → 125.5 tok/s.
   Job-script change, not code.

### Measured and rejected — do not retry without new evidence

| attempt | result |
|---|---|
| persistent OpenMP team per layer | +1.2%, inside noise. Reverted. |
| Q8_0 rows-in-lanes repack | 4.73× at 1 thread, **0.91× at 47**. Not integrated. |
| merged grid+sign LUT for IQ2_XXS / the reverse for IQ2_XS | −9% / −7%; the two formats want opposite choices |
| 8-row Q8_0 blocking | −20%, kept behind `K3_Q8_ROWS8=1` |
| `svtbl` for IQ grid lookup | inapplicable — grids are 256–65536 entries |
| `threads<=1` serial fast path in `k3_quant_matvec_ws` | −6% two different ways |
| `K3_BF16_ROWS=4` (fewer streams, 8 accumulators) | 1.462 vs 1.398 — worse |
| `K3_BF16_PREFETCH` 128…2048 | all inside noise; the HW prefetcher already handles 8 sequential streams |
| `K3_CMG_LOCAL=1` (route tasks to the owning CMG) | **−2.5%.** At 12n a projection is 3–7 large pages, so the 2/1/2/2 split imbalance exceeds the locality gain. |
| first-touch instead of `MPOL_INTERLEAVE` | 3.505 vs 1.500 — **2.34× worse.** Interleave is load-bearing. |
| `--ar-groups` 3 / 4 / 6 | 1.263 / 1.313 / 1.162 vs 1.161. **3 is actively harmful in the runner** despite winning `tp_ar_ack_test` by 1.21×. |
| `K3_COMM_POLL_SPINS`, `K3_COMM_ROBUST`, `K3_COMM_A2A` | all neutral or worse (a2a 1.178) |
| `moe_shard_layout=replicated` (2 collectives instead of 4) | **no change** (1.1616 vs 1.1576–1.1605). Removing a 0.134 ms collective buys nothing — the 94 MB/rank of extra streaming eats it. Keep `row-aligned` for 96n, where sharded weights shrink 8× and replicated ones do not. |
| `moe_shared` task-quantization theory | scales monotonically with threads, so not critical-path-quantized |

`K3_COMM_BF16=1` is worth **2.8%** but transports the residual-stream allreduce
at ~8 mantissa bits instead of 24 over 93 layers, with no error test. Measured,
not recommended without a quality gate.

### The pattern worth internalizing

The two biggest results of 2026-08-06 were a **correctness bug found only after
fixing the validation field**, and a **1.9× on `moe_shared` from calling a kernel
that already existed** (`k3_situ_fast_sve`, which the routed experts had been
using all along). Meanwhile the confident structural theses — collective count,
OpenMP synchronization, CMG routing, prefetch, register pressure — were all
killed by measurement.

Three specific traps, each of which produced a plausible-looking wrong table:
`memset`-filled benchmark weights (subnormal FMUL trap), dead-code elimination
(sink your outputs), and partial-CMG coverage (a speedup landing on exactly
4.00×/1.99× is a work-accounting bug, not a result).

Measure a one-line change before writing a hundred; and when a result looks
clean, check the harness before you believe it.

---

## Resume prompt

```text
Resume K3 work in /vol0006/mdt0/data/hp250467/work/gemm/k3 (branch k3, last
commit ae5cae32). Read k3-resume.md, then a64fx/k3/ROOFLINE.md for the
measurements behind it and @resume-k3.md for the 96-node job.

State: decode is 8.80 tok/s extrapolated from 12-node layer12 (KDA 1.153 ms,
MLA 1.420 ms) with all four flags on, 7.33 stock. The only real full-model
number is still 1.699 tok/s at 96 nodes (job 49931198), and that gap is the
biggest open question.

Environment: compute node, so pjsub/pjstat need `ssh login1 '<cmd>'` with an
explicit cd. cd does not persist between tool calls and the shell is zsh, which
does NOT word-split unquoted $VARs — use absolute paths and explicit env
assignments. /local is wiped between allocations. 47 OMP threads, never 48. Only
one mpiexec at a time.

Validate with hidden_hash from <result>/output.txt, NEVER output_hash — layer12
runs with generated_tokens=0 and emits the same output_hash for KDA and MLA
layers. That mistake hid a live data race for the whole life of the parallel
MLA attention change. Stock layer-2 hidden_hash is 6e94844067bbd6a1.

Gates: `make -C a64fx/k3 test` exits 0, and `run_k3_ep.sh --mode dummy --layers 1
--layer 3 --cache-tokens 16384 --mla-cache-int8 --threads 47` reports
checksum=+3.650037202e+02. For layer work, `K3_PROFILE=1 ./run_k3_full_12n.sh
--mode layer12 --layer-index 2|3 --expert-tp --prefill-tokens 512` gives the
per-phase profile; layer 2 is KDA (69 layers), layer 3 is MLA (24). Runs take
~10 s once staged, ~4 min if /local needs re-staging.

Highest-value next steps, in order: (1) decide whether to resubmit job 50005954
— it rebuilds from the tree so it gets the MLA fix, but its snapshotted script
still passes the prefetch regression and the replicated layout; (2) verify MLA in
a mode that actually generates tokens, now possible since the race fix;
(3) attack moe_expert (0.198 ms), which is access-pattern bound at ~22% of its
own kernel's rate because each expert matrix is under one 2 MB page and lands
wholly on one CMG.

Measurement discipline is the main lesson: discard the first (cold) run, take
best-of-N, and check the harness before believing a clean-looking result — this
session lost time to subnormal-filled benchmark weights, dead-code elimination,
partial-CMG work accounting, and zsh not word-splitting a flags variable. The
rejected list in k3-resume.md exists to stop retries; several confident
structural hypotheses are already in it.

Use fcc/FCC natively on A64FX. Preserve unrelated dirty workspace changes
(laguna-s21, llmgr, pjsub_k3_scale_*.sh) and do not push.
```
