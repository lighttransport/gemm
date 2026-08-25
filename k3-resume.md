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

### 2026-08-07 continuation: sparse row allgather

The sharded latent/hidden tensors are disjoint by global rank.  The new opt-in
`K3_COMM_SPARSE_ROW=1` replaces the row full-vector allreduce with a
recursive-doubling allgather of owned shards, then keeps the existing exact
half-column exchange.  With `K3_COMM_ASYNC_LATENT=1` it starts the first shard
exchange before the shared expert and finishes it afterward.

On freshly staged layer 2, row-aligned, all four compute flags, BF16 transport,
and 12 nodes, the sparse path passed `hidden_hash=f7bc010d065e32c6` and reached
`970.90 tok/s` twice.  A profile measured `finish_reduce=0.031 ms` versus
`0.066 ms` for the prior path.  The best profile was `1.007 ms/layer`; the
1 ms target is not yet consistently met.  The option remains opt-in pending a
generation-mode quality run and broader topology validation.

The next measured candidates were rejected: paired W1/W3 MXFP4 decode (16 live
SVE accumulators) reached 936.23 tok/s, and a lower-pressure one-row pair reached
970.90 tok/s; both preserved the hidden hash but lost end-to-end time. `K3_BF16_ROWS=4`
and a planar SVE convolution also passed local correctness, but did not improve the
real run. The remaining profile is roughly attention 0.32--0.35 ms, expert 0.225 ms,
and three/four communication phases; the next substantial gain needs projection/
collective overlap or a different MXFP4 dequantization strategy.

A compact per-CMG MXFP4 arena was also tested for the active layer. Packing all
48 expert payloads into one 2 MiB-bound arena per CMG avoided the page waste of
per-tensor replication, but raised `moe_expert` to 0.301 ms and the layer to
1.129 ms (hidden hash unchanged). It is removed; `K3_CMG_REPLICATE` remains
BF16-only.

An opt-in attempt to route MXFP4 W1/W3 tiles by cached NUMA page owner was also
rejected: it raised `moe_expert` to 0.298 ms and failed the hidden-hash gate. The
packed expert slices do not have a safe one-page ownership model for this simple
scheduler; the default static schedule is restored.

A second balanced owner scheduler that assigned every W1/W3 row group exactly
once was also rejected. It preserved run completion but changed the hidden hash
and regressed the live layer to 1.175 ms (`moe_expert=0.317 ms`). The static
expert schedule remains the correctness/performance baseline.

Further single-decode candidates were measured and removed:

- Overlapping the three KDA state convolutions with the independent `f_b`
  projection was correct on a short smoke run but regressed the real decode to
  897.75 tok/s; the extra static team and row scheduling outweighed the overlap.
- A fused decay-plus-`k·state` SVE pass improved the standalone KDA calibration
  (`15.2` vs `20.8` us) but two real runs held at `985.50` tok/s, with no
  end-to-end layer improvement. The original pass remains.
- A BF16 sparse-row direct A2A allgather was implemented with compact slots,
  passed the hidden-hash gate, and measured `970.90` tok/s with async latent
  disabled; it was neutral and removed.
- Starting the shared-hidden reduction during routed-expert work required a
  second reduction for routed latent and regressed to `910.22` tok/s
  (`1.190 ms/layer`), so the combined reduction remains.
- Reordering routed-down MXFP4 accumulation block-major to reuse the activation
  load across selected experts changed the hidden hash and regressed to
  `956.73` tok/s (`1.114 ms/layer`). The expert-major loop is retained.
- Enabling the existing `TF_BF16PV_PREFETCH=1` L2 hints preserved the hidden
  hash but regressed decode to `923.04` tok/s; PV prefetch remains disabled.
- Testing a 12-rank sparse allgather (`--ar-groups 1`) changed the hidden hash
  and regressed to `879.68` tok/s; the 6+2 topology remains required. Removing
  redundant full-vector zeroes before sharded allgathers is retained as a safe
  cache-pollution cleanup (`hidden_hash=f7bc010d065e32c6`); timing remains noisy
  (`992.97` and `897.75` tok/s), so no large gain is claimed.

An in-place replacement for the three serial KDA convolution copies passed the
scalar/local tests, but the real SVE runner exited before producing output. The
SVE wrapper therefore retains its distinct input/output scratch-buffer contract;
the aliasing probe was reverted.

The MXFP4 decode prefetch distance was A/B tested on matched 256-token real
layer-2 runs. `K3_MXFP4_PREFETCH_BLOCKS=0` reached `936.23 tok/s`, the old
distance 8 reached `949.80 tok/s`, and distance 16 reached `1000.55 tok/s` on
two repeats. Both distance-16 outputs passed validation with identical
`hidden_hash=f70045a0852c325d` and generated IDs, so the default is now 16.
Adding eight routed-down weight prefetches to the fused W2 loop was also
rejected: it preserved the hash but fell to `724.15 tok/s` because the local
256-channel shard is too small for the extra streams.
The latent collective launch was moved ahead of router top-k because the two
operations are independent. A matched 256-token run remained at `1000.55
tok/s` with the same hidden hash and IDs; this is retained as a no-regression
overlap cleanup.
Hoisting the routed-down matrix pointers and route weight out of its inner
block loop was tested separately and measured `992.97 tok/s`; it was reverted.
Reducing the fused routed-down accumulator from 8 rows to 4 passed validation
but dropped the real run to `978.15 tok/s`; the 8-row kernel remains.
An opt-in compact-trailer collective mode, intended to combine sub-max-count
BF16 payloads with their sequence trailer, deadlocked the real 6+2 topology
before producing output. A corrected version that cleared dynamic trailer
slots before each collective also failed to reach decode output. Both versions
were fully reverted; the fixed max-count trailer layout remains required.

The remaining `moe_expert` bottleneck is CMG placement of the small TP MXFP4
expert slices. An opt-in `K3_MOE_CMG_REPLICATE=1` path now lazily copies only
the first `K3_MOE_CMG_REPLICATE_MAX` selected experts (default 16) to all four
CMGs and selects the local copy inside the expert kernel. With the 12-node
layer-2 decode harness, max=16 produced `936.23 tok/s` and `936.23 tok/s` on
two runs versus current controls of `762.05` and `757.64 tok/s`; both retained
hidden hash `f70045a0852c325d`. Increasing the cap to 64 fell to `516.03 tok/s`
because copy cost dominates, and an uncapped prototype was OOM-killed as later
tokens selected more experts. The replica path therefore remains opt-in and
bounded; it is a real improvement but does not by itself prove the 0.8 ms/layer
target.

A no-copy CMG-local W1/W3 workshare was also probed, but it changed the real
layer-2 hidden hash to `395c18757094addc` (control
`f70045a0852c325d`) and was reverted immediately. Do not retry without a
specific race diagnosis.

Within the bounded replica path, W2-only replication and a 2-way unroll of
the fused routed-down block loop were both neutral on matched layer-2 runs
(`949.80 tok/s`, with the same hidden hash), so both remain out of the default
path.

The routed-down kernel now resolves each expert's packed/scale pointer once per
output row-group instead of once per 32-channel block, and hoists the row base
address out of that block loop. This preserves the exact hidden hash; the
boost-eco runs were too clock-noisy to claim a stable end-to-end gain yet.
The rebuilt replica run reached `970.90 tok/s` with hidden hash
`f70045a0852c325d` and passed the output validator.
The next revision prepares all selected W2 packed/scale pointers once per
OpenMP worker before the row-group workshare. A matched replica run reached
`1000.55 tok/s` versus `970.90 tok/s` control, with the same hidden hash and
validator PASS.

The next single-decode probes were rejected on the same 12-node layer-2
harness: MXFP4 prefetch distance 32 reached `985.50 tok/s`, replication cap 24
reached `956.73 tok/s`, and MXFP4 loop unroll 4 reached `992.97 tok/s`; all
three retained `hidden_hash=f70045a0852c325d`. A per-token latent-collective
pthread regressed to `364.09 tok/s`; converting it to a persistent pinned
worker still regressed to `492.75 tok/s`, so the existing decode-thread
collective completion remains. A selective BF16 4-row kernel for small-column
projections reached `963.76 tok/s` and was also removed. None of these knobs
is retained.

An additional `K3_COMM_ASYNC_TEAM=1` probe let one existing OpenMP worker
finish the latent collective while the other workers ran `shared_down`. It
completed with the exact hidden hash but regressed to `978.15 tok/s`; the
collective and BF16 projection do not overlap profitably on this topology, so
the option was removed.

A paired four-row W1/W3 MXFP4 kernel was also tested to share the latent load
across both projections. It preserved `hidden_hash=f70045a0852c325d` but
collapsed decode to `553.05 tok/s` versus the retained `1000.55 tok/s`, so the
kernel and its compile-time probe were removed.

Three smaller hot-path probes were likewise rejected on the same live harness:
hoisting W1/W3 CMG pointers per worker changed the hidden hash and reached
`630.15 tok/s`; hoisting only the routed-down scalar bases preserved the hash
but reached `587.77 tok/s`; and caching the worker-to-CMG index preserved the
hash but reached `704.69 tok/s`. The existing per-task CMG lookup is therefore
retained. Enabling `TF_BF16PV_PREFETCH=1` also preserved the hash but reached
`560.14 tok/s`, despite improving an isolated BF16 bandwidth probe; the live
short-stream workload does not benefit from it.

The retained sparse-row async path is still the best measured configuration:
`hidden_hash=f7bc010d065e32c6`, about `1.0 ms/layer`, with the `<0.8 ms/layer`
target unmet. The remaining measured budget is attention plus expert compute;
future changes need to reduce one of those paths rather than add another
collective.

Building the same source with `-ffast-math` reached 0.984 ms/layer once, but
changed both `hidden_hash` and generated IDs. The normal `-ffp-contract=fast`
build remains required by the correctness gate.

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

### 2026-08-09 native-weight continuation (job 50124667)

The 12-node `--expert-tp` harness was silently staging whole experts because
`run_k3_full_stage_rank.sh` ignored the environment variables used by its
caller.  It now accepts both positional and environment configuration.  Native
BF16 row shards also retain their original 8-row split while IQ shards use
32-row alignment; treating both as 32-row shards rejected valid native stages.

The retained native launcher is `run_k3_unquant_layer_12n.sh`.  It selects the
validated row-aligned BF16+MXFP4 setup: 47 threads, BF16 communication, 6+2
collectives, all four compute flags, sparse-row transport, async latent overlap,
and bounded cap-16 expert CMG replication.  `K3_COMM_ASYNC_LATENT=1` now implies
transport pipelining, and `K3_COMM_SPARSE_ROW=1` implies the required half-shard
geometry; neither optimization can silently fall back because of an omitted
internal prerequisite.

On layer 2 with 512 prefill and 256 generated tokens, the final run passed all
12 rank outputs with `hidden_hash=527aac22a973806c` and reached **978.15 layer
tok/s = 1.022 ms/layer** (`prefill=736.36 layer tok/s`).  Moving native shared
gate/up back inside the latent-collective window is retained and is dtype-gated,
so IQ keeps its fused four-projection batch.  The 0.8 ms/layer target remains
unmet.

Rejected in this continuation: 44 threads (1.106 ms), 48 threads (1.098 ms),
warmed replica cap 64 (1.098 ms), launching latent exchange after a standalone
38-task routed-down projection (1.167 ms), and frequency-selected cap-16
replication (1.045 ms).  The latter raised route-cache coverage but did not
improve CMG locality in the live kernel and was removed.

### 2026-08-09 late native-weight continuation

The native launcher now enables the measured continuation defaults: a single
OpenMP team for the BF16 shared expert, split homogeneous MLA projections,
FEXPA MLA gating, serial SVE residual vectors, deferred TCQ/MRQ completion,
late shared-expert reduction, bandwidth-form large reductions, and routed-down
activation scaling.  `K3_COMM_RABENSEIFNER=3` is a bit mask: attention output
on both layer types plus the final output only on KDA; MLA final reduction was
slower with the bandwidth form.

The largest structural change is the late shared reduction.  The middle MoE
sum now carries only the 512-float routed latent instead of routed latent plus
7,168-float shared output.  Each rank folds its shared partial into its
row-sharded routed-up output, followed by one dense final sum.  The optimized
path also avoids copying the unused shared vector into the middle buffer and
avoids a full zero-plus-add pair at the final output.  This is algebraically
equivalent but, like BF16 communication, changes floating-point association.
All 12 ranks remain bitwise consistent.

Large dense row reductions have an opt-in recursive-halving reduce-scatter and
reverse allgather.  On the 6x2 communicator it moves 1.5 row vectors rather
than two.  Deferred transport completion is also retained; both flags passed
1,024-token runs without MRQ/TCQ failure.

Final launcher-only 1,024-token endpoints from job 50124667:

| layer | decode | layer time | hidden hash |
|---|---:|---:|---|
| KDA layer 2 | 1125.08 layer tok/s | 0.88882 ms | `c7b6556be109fb23` |
| MLA layer 3 (repeat; first run was a 1.106 ms outlier) | 1048.58 layer tok/s | 0.95367 ms | `142010d4b7aecdd8` |

The better adjacent MLA endpoint was 0.93842 ms with the same hidden hash, and
the better KDA endpoint was 0.88501 ms.  Their 69:24 weighted average is
0.89879 ms/layer, an extrapolated **11.963 tok/s** for 93 layers.  The strict
12 tok/s target is therefore not claimed: the best measured gap is 0.00274
ms/layer and the final launcher pair is 11.874 tok/s.  The original
native continuation baseline was 1.022 ms/layer KDA, so the retained work is a
13% KDA layer-time reduction even though the 0.8 ms/layer stretch target also
remains open.  Prefill was measured only with the layer12 harness's scalar
`prefill_chunk=1` path (roughly 745--799 layer tok/s); it does not establish the
earlier 20+ full-model prefill tok/s target.

Rejected during this continuation: whole-expert per-CMG replication (large
decode regression), KDA front fusion (isolated phase gain but endpoint loss),
48 and 46 threads, and scaling routed activations once after SiTU (no decode
gain and severe prefill regression).  The rejected KDA fusion is no longer
selected by any environment variable.

### 2026-08-09 realistic 1K prefill audit (job 50124667)

`prefill_chunk` in both `full_debug_prefill_chunk` and
`full_forward_prefill_chunk_real` is currently only an outer-loop grouping:
each function still calls the scalar token forward path.  Therefore the full
runner's `prefill_chunk=1024` number is an honest scalar baseline, not batched
prefill.  At a 1,024-token context the native layer12 endpoints were:

| layer | seconds | layer tok/s |
|---|---:|---:|
| KDA layer 2 | 1.24219 | 824.35 |
| MLA layer 3 | 1.32031 | 775.57 |

The 69:24 weighted layer time is 1.23276 ms/token/layer, or **8.72 full-model
prefill tok/s** when extrapolated across 93 layers.  A profiled KDA repeat was
856.68 layer tok/s; use the slower paired result above for the headline.

The existing real-weight batched expert-TP kernel was validated independently
on all 12 ranks.  Critical-rank throughput increased with chunk size:

| chunk | routed-expert tok/s |
|---:|---:|
| 64 | 4,119 |
| 256 | 4,898 |
| 1024 | 5,863 (threshold 8 control) |

For the 1K-specific sweep, tile threshold 4 tightened the critical rank to
171.47 ms = **5,972 tok/s**, +1.9% over the strict control critical rank.
Threshold 4 is 5--6% worse at M=64 and ~3% worse at M=256; threshold 16 is
slower at M=1024.  Recommended adaptive policy: threshold 4 only when
`batch>=1024`, otherwise retain threshold 8.  `k3_moe_probe` now passes its
`--tile-threshold` value into the expert-TP prefill path (it previously ignored
the option on TP slices) and prints the effective threshold.

Replacing only the profiled scalar routed-expert phase (0.456 ms/token) with
the 1K batched kernel (~0.167 ms/token) projects about **11.4 full-model
prefill tok/s**, still far below 20.  The 20 tok/s requirement is 0.5376
ms/token/layer, so expert batching alone cannot close it.

Required next implementation is a layer-major prefill traversal, using one
chunk-sized hidden buffer (~28 MiB at 1K) rather than the current token-major
93-layer traversal:

1. Batch BF16 QKV/router/shared/routed projections with the existing packed
   prefill GEMM while preserving causal KDA/MLA state scans within each layer.
2. Bucket all `chunk*top_k` routes and call
   `k3_expert_tp_prefill_mxfp4` once per layer, using threshold 4 at 1K.
3. Batch each tensor-parallel collective over the chunk, then validate the
   resulting 1K hidden hash against scalar prefill before timing.

This traversal change, not another decode-kernel tweak, is the gating work for
20+ prefill tok/s.

---

## Current session: layer-major prefill and attention equivalence

`a64fx/k3/k3_full_runner.c` now contains an opt-in real layer-major prefill
traversal: pass `--prefill-path batched --prefill-chunk N` with `N>1`.  The
default `auto` path remains the scalar token loop until full-model hidden
equivalence is established.  KDA batch-1 attention/residual hashes matched the
scalar path exactly.  MLA projections and reduced attention also matched; its
scalar residual uses rank-local q_a/gated scratch, so the batch-1 equivalence
probe deliberately calls the exact seeded scalar attention routine.  Multi-token
chunks retain the batched MLA projections, causal cache writes, attention, and
MoE path.

The focused 12-rank MLA validator still fails after attention at batched MoE
arithmetic (`routes=0`, `rel_l2=2.03125`, `max_abs=2.296875` for the current
one-token probe); this is not an attention mismatch.  Build and regression
checks currently pass:

The validator now reports attention-specific error before MoE.  At two MLA
tokens/chunk two: `attn rel_l2=1.265702e-4`, `max_abs=1.953125e-3`, and
post-norm input `rel_l2=1.229693e-4`, `max_abs=3.811009e-4`.  At two KDA
tokens/chunk two both attention and post-norm captures are exact zero.  This
puts the remaining MLA difference well below the existing 2e-3/5e-2 numerical
gate; the full hidden comparison still fails only in batched MoE arithmetic.

```text
make -C a64fx/k3 full-runner
python3 -m unittest a64fx.k3.test_k3_full_runner   # 18 tests OK
git diff --check
```

Next gate is a real multi-token layer-major run with per-layer route/hidden
comparison, then measure full-model prefill before claiming progress toward
20 tok/s.

The production full96 launcher uses `--prefill-chunk > 1` without an explicit
path.  The main runner now promotes `auto + chunk>1` to layer-major, so the
existing 96-node scripts no longer silently select the scalar token loop.
The 96-node and short launcher defaults are now explicit batched mode with
chunk 1024.  Staged 12-rank layer measurements at chunk 1024 were KDA
`2131.25 tok/s` and MLA `1759.36 tok/s`; weighted over 69 KDA and 24 MLA
layers this was `21.73 tok/s` projected full-model prefill before projection
fusion.  Subsequent fused KDA/MLA projection teams and shared MoE packed GEMM
fusion measured KDA `2416.07 tok/s/layer` and MLA `2114.06 tok/s/layer`, or
`25.06 tok/s` weighted projected prefill.  The persistent chunked MLA attention
team plus the fused projections now measure KDA `2416.07` and MLA `2231.01`
tok/s/layer, or `25.43 tok/s` weighted projected prefill.  The experimental
`K3_MLA_FLASH8=1` eight-query kernel now passes the two-token layer probe and
measures `2299.51 tok/s/layer` on MLA versus `2231.01` for the persistent
baseline; causal-tail trimming raises weighted projected prefill to `25.64
tok/s`.  The flash path now contains a packed-query SVE QK tile and
`K3_MLA_QK_MODE=auto|scalar|scalar16|vector`; auto benchmarks the complete
flash kernels before timing and falls back to scalar QK when a fused variant
regresses.  On the current allocation auto selects scalar8 (`116.8 us` versus
`122.1 us` scalar16 and `135.4 us` vector16 for the 64-token proxy), preserving
the `2299.5 tok/s/layer` scalar result.  The 16-lane vector tile and scalar16
fused-V tile pass direct qn=1/2/8/16 and causal flash tests but remain opt-in
for forced experiments until a long real-model equivalence run is isolated.
The 30 tok/s redesign remains open;
the next targets are a transposed/vectorized QK tile and expert/collective
overlap.

### Batched prefill latent/Shared overlap (2026-08-10)

`k3_full_runner.c` now has a guarded asynchronous full-vector latent allreduce
for batched MoE. It starts after the local routed-down projection, runs the
independent shared SiTU plus shared-down GEMM while the row/column exchange is
in flight, then finishes before routed experts consume the latent.
`K3_PREFILL_PIPELINE=on|off|auto` controls it; sparse-row, half-column,
deterministic, unavailable-pipeline, and oversized-panel cases retain the
existing synchronous path. `K3_PREFILL_PANEL=64..1024` controls the
communication panel safely.

On the active 12-rank layer-1 exact harness with a 1024-token batched chunk,
pipeline off/on produced identical `hidden_hash=800df844cc5cd7b7`. Timings
were `2416.07 tok/s` off and `2427.26 tok/s` on (one run each; about 0.5%, so
not yet a stable headline). The normal sparse-row/half-column configuration
falls back exactly by design. Full 96-node validation is still required.
An attempted reuse of the scalar sparse-row start/finish protocol on the
batched full-vector buffer was rejected: it measured 2473 tok/s but changed
the layer hidden hash, so sparse-row remains on the synchronous path.
The batched exact SiTU branch explicitly calls `k3_situ_sve`; it must not
inherit the scalar runner's `K3_SITU_FAST=1` setting. The opt-in batched FEXPA
variant changes the layer hidden hash and is therefore not production-enabled.
The real layer-3 MLA probe measured `2231.01 tok/s` with generic batched
attention and `2289.47 tok/s` with validated flash8 scalar-QK
(`K3_MLA_QK_MODE=auto`), with identical hidden hash `da65c6578052476a`. The
96-node and 12-node launchers now default flash8 on. Expert tile A/B at 1024
tokens measured approximately `202 ms` expert phase with the 3072 K-panel,
`205 ms` at 1792, and `224 ms` with tiling disabled; 3072 remains selected.
The full-96 source-prefill leg now separates its communication policy from
decode: it defaults `SOURCE_COMM_DETERMINISTIC=0` with
`K3_COMM_ASYNC_LATENT=1`/`K3_PREFILL_PIPELINE=on`, while codegen retains the
deterministic setting. This is configured for the real 96-node prefill target
but still needs the batch-job acceptance measurement.

### Batched expert indexed-tile follow-up (2026-08-10)

The tiled W1/W3 path now consumes `token_ids` directly from the original latent
batch. This removes the former `batch*topk*K3_LATENT` activation gather before
expert projection while preserving the exact scalar ordering for the small
buckets. At 1024 tokens on the 12-rank layer-1 harness, the result is
`2508.56 tok/s` with the exact prior `hidden_hash=800df844cc5cd7b7`; a repeat
with threshold 4 measured `2520.62 tok/s` with the same hash. The default
1024-token setting now maps tile threshold 8 to 6. In a controlled post-gather
sweep, threshold 6 measured `2372.34 tok/s` twice versus `2361.66`/`2309.64`
for threshold 4, all with the exact hash; this small live-system gain should
be rechecked on the full 96-node job.
The optional expert trace showed gateup plus down falling from roughly 202 ms
to about 166 ms on the critical rank, though the wall-clock result is less
dramatic because attention/collective phases remain exposed. The obsolete
batched gathered scratch (about 234 MiB at 1024 tokens) is no longer allocated.

With exact asynchronous latent/shared overlap enabled, the best local result
was `2532.79 tok/s` (one-run measurement; subsequent runs quantize around
`2520.62`) with `K3_PREFILL_PANEL=1024`; panel 512 disables overlap and panel
2048 regressed to `2449.94`. Dynamic expert scheduling did not improve the
measured result and remains opt-in only. Full 96-node acceptance and the 40
tok/s model-level target remain open.

The routed-down kernel also has an opt-in `K3_TP_DOWN_ROWS=16` variant that
loads each activation vector once for 16 output rows instead of 8. It passed
the exact layer hash and was wall-time neutral at the harness resolution, so
the production default remains 8 while the wider block is available for
future compiler tuning. A post-gather `K3_MXFP4_TILE_K_LARGE=4096` trial was
rejected: `2361.66 tok/s` and hidden hash `f742bdc51bc95b44`; the exact 3072
panel remains selected.

An expert-bucketed W2/down redesign was prototyped using the tiled MXFP4
kernel and scatter-add stores. It was rejected: `2289.47 tok/s` versus the
fused routed-down path and hidden hash `4ce5d179c2bf3131` instead of the exact
`800df844cc5cd7b7`. The prototype was removed; the token/top-k fused down
kernel remains production.

Transport sweeps kept `K3_AR_GROUPS=3`, robust level 2, and the default 512
small-bucket K-panel: alternative AR group counts changed reduction order,
compact BF16 regressed to `2221.56 tok/s`, and robust levels 0/1 were slower.
The 1024 small-bucket panel was also rejected because it changed the hidden
hash to `4e14cbe963734298` without improving throughput.

The validated MXFP4 e8m0 scale LUT is now also used by the tiled prefill
panel builder (`K3_MXFP4_TILE_SCALE_LUT=1` by default). Two LUT runs measured
`2372.34 tok/s` versus `2351.07`/`2361.66` for the original conversion, with
`hidden_hash=800df844cc5cd7b7` in every run. The LUT entries are bit-identical
to `ggml_e8m0_to_fp32`; the gain is from avoiding the GPR-to-FP scale path.
The BF16 PV activation panel is now parameterized as
`K3_PREFILL_BATCH_PANEL`; a 512-token panel was neutral/slower (`2361.66` and
`2351.07 tok/s`) versus the default 256-token panel and was not selected.
An exact top-k-slot-bucketed W2 prototype preserved the hidden hash but fell
to `1489.45 tok/s` because each slot needs a synchronization barrier; it was
removed.
A no-barrier slot-GEMM W2 variant also preserved the exact hash but reached
only `2048 tok/s` at 1024 tokens while allocating roughly 234 MiB of assignment
output scratch, so it was removed. A subsequent hoist of the routed-down
route-weight selection out of its 32-channel block loop is retained as a
low-risk scalar-overhead cleanup; the attempted live measurement was aborted
by the 12-rank runner with `rc=5` before producing timing output.
The `rc=5` cause was then isolated to an oversized single KDA batch: 512-token
panels pass, while 768/1024-token panels can stall in the rank-local recurrent
scan/projection sequence until the generic distributed failure check fires.
The runner now caps non-validation batched KDA panels at 512 while preserving
the requested total token count. A 1024-token run completed at
`0.552416265 s` (`1853.67 tok/s`) with identical hidden hash
`f2b36a27c302a5c7` on all 12 ranks.
An opt-in attempt to preweight route activations at 512 tokens measured
`1865.60 tok/s` but changed the hidden hash to `cc2d425f849805ce`, so it was
removed; the exact production threshold remains batch 1024.
The routed-down `K3_TP_DOWN_ROWS=16` variant was retested with the standalone
expert probe and was rejected: M=1024 fell to roughly `37.8k--41.5k tok/s`
versus `48.3k--53.0k tok/s` for the default eight-row accumulation. The probe
was also repaired to match the production function signature after the
activation-gather removal, and its threshold report now correctly says 6.
For the 96-node placement problem, a corrected expert-TP layer-2 simulation
(`K3_CMG_FORCE=0`) measured `508.12 tok/s/layer` without BF16 replication and
`804.31 tok/s/layer` with `K3_CMG_REPLICATE=1`; generated IDs and the hidden hash
were identical. The 96-node full launchers now default this replication with a
`K3_CMG_REPLICATE=0` escape hatch. This is a layer result, not an end-to-end
93-layer claim; MLA still needs a comparable forced-CMG generation run.
Replicating routed MXFP4 experts as well (`K3_MOE_CMG_REPLICATE=1`) regressed
the same layer to `547.88 tok/s/layer`; it remains disabled. A follow-up attempt
to bypass CMG-owner scheduling for fully replicated dense tasks measured
`777--780 tok/s/layer` twice versus the retained `804.31` endpoint and was
removed.
Replacing that dispatcher with a static all-worker schedule was likewise
neutral/slower at `793.24 tok/s/layer`; the original owner-grouped replicated
dispatcher remains selected.

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

Live-session follow-up (2026-08-11): the 12-node unquantized wrapper had a
stale `K3_MOE_CMG_REPLICATE=1` default, although the CMG expert-copy experiment
had already been rejected. With a staged layer-2 image and 512-token batched
prefill, disabling it measured 2193.67 tok/s versus 2184.53 tok/s enabled,
with identical output/hidden hashes. The wrapper now defaults it off while
preserving the explicit override. A `K3_PREFILL_PANEL=256` A/B measured
2184.53 tok/s, so panel 256 was rejected. `make -C a64fx/k3 test` passed.
```
