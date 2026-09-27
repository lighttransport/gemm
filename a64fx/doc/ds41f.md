# DeepSeek-V4.1-Flash on 12 A64FX nodes

See the [current results](#current-results-2026-09-13), the original
[20/30/40 tokens/s implementation plan](#203040-tokenss-implementation-plan),
the [100+ tokens/s prefill plan](#100-tokenss-prefill-plan-2026-09-14),
and [remaining work](#future-work-after-the-continuation).
The earlier pause was superseded by the requests to pursue INT8 SDOT decode
and implement the plan. Dated sections preserve the experiment history;
their allocation status and unfinished-task notes describe that point in time.
The current results and remaining-work list supersede those older notes.

## 200+ tokens/s three-stage prefill plan, 2026-09-15

The next prefill target is **200+ input tokens/s** for the fixed 1105-token
replay on twelve normal-mode A64FX nodes. Timing excludes resident loading but
includes embedding, Engram reads and transfers, causal state updates,
pipeline fill/drain and the final vocabulary head. The accepted batch-64
control remains 33.45--33.56 input tokens/s. Its approximately 12.0 ms/token
world FFN publication and 5.2 ms/token shared/reduction span already exceed
the complete 5 ms/token target, so the new path changes the distribution
rather than adding another serial collective optimization.

Use three pipeline stages with four ranks each: ranks 0--3 own layers 0--12,
ranks 4--7 own layers 13--26, and ranks 8--11 own layers 27--39. Attention and
the shared expert remain TP4 inside a stage; routed experts use
`expert_id % 4`, and the local owner is `4*stage + (layer & 3)`. Only the
BF16 residual plus bounded causal metadata crosses the two stage boundaries.
The projected largest rank holds 26,155,609,824 bytes of resident weights,
close to the validated 26.08 GB maximum and compatible with the existing
2 GiB `MemAvailable` floor. The legacy layout and accepted 24 tok/s decode
command remain separate controls; the pipeline layout must hand off correct
state to decode, but its decode speed does not gate the prefill result.

Implementation order and gates:

1. Add a versioned pipeline manifest and bounded staging pass. It records the
   stage/layer ranges, TP4 tensor ranges, local expert ownership, source header
   hashes and expected resident bytes. Reject incomplete or mismatched roots.
2. Split MPI into the three stage communicators and remove world collectives
   from the prompt layer loop. Use double-buffered nonblocking handoffs from
   rank 0 to rank 5 after layer 12 and rank 6 to rank 11 after layer 26.
   Carry at most 512 selected row IDs and 2048 candidate block IDs per token,
   plus source-1/source-3 compressed publications where required.
3. Replace per-token Engram synchronization with a bounded four-tile feeder.
   Four local I/O workers sort and deduplicate owned reads; only the main
   thread issues MPI under `MPI_THREAD_FUNNELED`. Target owners are ranks 1
   and 6 for layers 1 and 14.
4. Start with 36-token prompt tiles. Integrate the BF16 2x12 kernel as a wide
   dense path and compare it per production shape with the retained six-token
   INT8 SDOT path. Keep one resident weight representation per tensor. Batch
   projection, gate, Engram projection, mHC/norm/pointwise and sparse
   attention work; preserve causal preparation and every BF16 boundary.
5. Column-shard WO-B and shared W2 inside TP4. Reduce each rank's routed and
   shared contribution together once per layer. Keep exact softmax; the
   previously rejected approximate exp2 modes stay disabled.

The slowest stage must average at most 4.4 ms/input token, with engineering
budgets of 1.6 ms for attention plus mHC, 2.0 ms for routed/shared FFN,
0.45 ms for all communication and 0.15 ms exposed Engram wait. Admission
requires all 1105 retained next-token IDs, route IDs, selected IDs and
structural state counts to match; captured tensors and logits require cosine
at least 0.999 and relative RMS at most 1%. The performance claim requires
three runs individually at or above 200 input tokens/s (at most 5.525 seconds
per replay), full-tile p95 and stage-balance reporting, and at least 2 GiB
`MemAvailable` throughout on every rank.

Earlier work in commits `2b5bd3d3` and `f02af899` added stage-local BF16
broadcast, owner reduction, and two-slot nonblocking residual handoff
primitives. The versioned Python stager described in that work is absent from
this checkout, so pipeline roots cannot yet be prepared here. The existing
verifier still uses the legacy all-layer loop. The A64FX MPI build and
transport tests remain to be run in a compute allocation.

The scheduler contract is now isolated in `ds41f_pipeline.[ch]`. It validates
the stage descriptor, exposes the layer-owner and boundary mapping, and bounds
the selected-row, candidate-block, and compressed-publication fields carried
between stages. `test_pipeline` exercises those rules on the host; it does not
claim that the legacy verifier can execute a staged root yet.

`ds41f_pipeline_runtime.[ch]` now provides the wavefront driver for the future
model callback: it runs the `tiles+2` fill/drain waves, maps wave `W` to tile
`W-stage`, and alternates two handoff slots. The host test covers all three
stages, partial final tiles, and slot ordering. The callback remains separate
from the verifier until its layer-local state and stage collectives are
converted together.

The runtime now preposts the next stage-boundary receive immediately after a
tile callback, while retaining the two-slot wait-before-reuse rule. This
removes a receive-control bubble from the steady-state wavefront; the host
transport test checks that tile 1 is posted before tile 0's stage work drains.
The runtime also rejects tile sizes above the fixed 64-token envelope before
invoking a callback.

The verifier layer body is now exposed internally as
`forward_batch_range(..., first_layer, last_layer, include_head)`. The legacy
`forward_batch()` wrapper still executes layers 0--39 with the vocabulary head;
the staged callback can reuse the same implementation for one contiguous stage
and defer the head to the final stage.

`verify_begin_stage()` now initializes a tile from a received 20,484-float
residual packet, snapshots the stage attention state, and records the tile's
positions without re-embedding tokens or rereading Engram. The eventual
callback will call this for stages 1 and 2 after validating their metadata;
stage 0 continues to use the existing token/Engram initializer.

The staged callback now accepts one compact metadata wire per tile, validates
its header against the scheduled tile, and installs it before the stage-local
snapshot. Residual and metadata arrays are independently strided so they can
use separate double buffers while retaining identical tile sequencing.

`ds41f_attention_receive_pipeline_metadata()` now validates a compact wire
and installs selected rows, candidate blocks, and the compressed publication
into a stage-local attention state. It rejects oversized or mismatched tile
headers before modifying state; the wavefront callback still needs to supply
the layer and residual handoff sequencing.

The boundary transport now also has a raw-byte asynchronous path sized for one
`ds41f_pipeline_wire`. The model callback can post residual and causal metadata
with matching slots, then validate the received header before installing rows
or candidate blocks. The existing BF16 path remains unchanged for legacy
decode and verifier runs.

The stage wire now has a fixed-capacity envelope sized for 64 tokens. Each
slot carries the 20,480-element BF16 residual prefix, four FP32 tail values,
and one complete causal metadata record; unused slots are zero-filled and
never read from the caller. `test_pipeline` covers a partial two-token tile,
exact-BF16 rejection, metadata round-trip, and the fixed-size canary. The
remaining integration step is a scheduler transport adapter that owns two
envelope buffers per boundary and posts one MPI request per slot; the staged
verifier must not be enabled until that adapter and per-token publication
state are validated on twelve ranks.
The packer avoids clearing valid slots before overwriting them, reducing
memory traffic on full tiles while retaining zero-fill for the partial tail.
The scheduler also rejects tile-count arithmetic that could overflow before
computing token positions.

`ds41f_pipeline_transport.[ch]` now supplies the MPI adapter for this
contract. It owns two send and two receive envelope buffers, maps the fixed
boundary pairs `(0,5)` and `(6,11)`, and performs pack/send plus receive/unpack
through the scheduler callbacks. It requires identical tile and token strides
for both directions and rejects records that cannot fit the 64-token wire.
The A64FX MPI build and a twelve-rank delayed-receiver test are still required
before enabling staged verifier execution.

Before staging a checkpoint, restore or implement the bounded audit stager.
It must report all twelve footprints, each stage maximum, the global maximum
rank, and the expected `(0,5)` and `(6,11)` boundary pairs without copying
weights. Do not enable the staged path until that audit and the twelve-rank
transport test pass.

After this target, retain as future work a unified pipeline layout that also
repeats the accepted 24 tok/s decode result, long-prompt and actual-1M
execution, KV checkpoint/restore, multi-request scheduling and server
integration. None of those expands the 1K single-request prefill acceptance
scope.

## 100+ tokens/s prefill plan, 2026-09-14

The accepted decode stopping point is the speed-first configuration at
**40.722 ms/token, 24.557 tokens/s** around 1K history. Decode optimization is
paused. Prefill now targets **100+ input tokens/s** on the same 12 A64FX nodes,
excluding resident weight loading and including all causal KV/Engram state
updates. The fixed 1105-token replay remains the primary workload.

The current prompt loop calls the batch-one `forward()` for every input, so it
cannot reach the target through decode tuning alone. The existing causal
verifier provides the starting implementation: it evaluates up to six token
positions together, preserves each position's causal attention view, batches
INT8 expert GEMVs, and has batched BF16 transport. First measure batch sizes
1--6 on a standard TP4 layout and compare every predicted token plus final
state against sequential replay. At batch six, 100 tokens/s requires at most
**60 ms per six-token tile** or **10 ms/input token**.

Implementation gates:

1. Promote bounded replay batching into an explicit `--prefill-batch` runner
   path. It must process only prompt tokens in tiles, commit the final causal
   state, then hand the last prediction and state to ordinary decode. Remove
   verifier-only allocations and checks from the production path only after
   the trace/state gate passes.
2. Profile prefill as begin/pre, causal attention, gate, input broadcast,
   routed experts, shared/reduction, post/handoff and head. Use one head pass
   at the prompt boundary; intermediate prompt logits are unnecessary.
3. Increase tiles beyond six once the bounded path is stable. Batch projection
   and expert kernels must reuse each weight panel across tokens. Causal sparse
   attention retains per-position masks and KV publication order. Target
   16-token tiles first, then 32 if HBM scratch remains below the 2 GiB floor.
4. Combine communication across the tile: one token-major owner broadcast and
   one reduction per layer/tile. Keep explicit runner arguments; do not select
   production prefill behavior through environment variables.
5. Admission requires exact prompt predictions and final state versus the
   retained sequential path, three timing repeats, >=100 input tokens/s mean,
   and reported p95 tile latency and minimum `MemAvailable`. Approximate INT8,
   expert SDOT, mHC and BF16 reduction results remain labeled speed-first.

The first standard-TP4 measurement used the existing six-position causal
verifier over all 1105 inputs. It took **37.666 seconds, 29.34 input tokens/s**.
The maximum-rank average per six-input tile was 25.9 ms begin/state setup,
8.8 ms pre, 51.6 ms attention, 1.6 ms gate, 82.9 ms input broadcast, 13.6 ms
routed experts, 37.2 ms shared/reduction, 1.9 ms post/handoff and 3.2 ms head.
Thus communication alone is about 120 ms of a 227 ms tile, while the target
allows 60 ms. The intermediate next-token trace matched 1065/1105 positions;
the first difference was at position 32, so final-state equivalence remains a
required gate rather than an assumption based on the final token.

An explicit `--prefill-batch` path now enables batched expert and transport
operators and computes the vocabulary head only at the prompt boundary. Its
first batch-six run preserved the expected final next token 19 and reduced
nonfinal-tile head time to approximately 0.007 ms, but total time was
**39.130 seconds, 28.24 input tokens/s**. This confirms that head work is not
the limiting component. The implementation accepts up to 16-position causal
tiles; INT8 and MXFP4 operators remain internally tiled by six so their proven
microkernels and accumulation order are retained while communication is
amortized across the larger outer tile.

The validated 16-position run completed in **36.544 seconds, 30.24 input
tokens/s**, retained final token 19, and used about 160 MiB of causal-window
scratch per rank. Replacing the routed-expert allreduce with a prompt-only
`MPI_Reduce` to the layer owner reduced the run to **36.095 seconds, 30.61
input tokens/s**. The rooted reduction changes FP32 summation order (rank-0
state hash `b233094c50f0ea24` versus `2f53e57c85fb69bf`) and is therefore a
speed-first result. Its small 1.2% gain confirms that the next material change
must reduce bytes, especially the full-world FFN input broadcast. Send each
token row only to ranks owning one of its six selected experts, retain route
order at the owner, and measure this sparse exchange before increasing the
outer tile again.

A first blocking route-aware prototype published only the route metadata and
sent each BF16 input row to ranks owning a selected routed expert. It regressed
to **36.872 seconds, 29.97 input tokens/s**. It also omitted some ranks in the
owner's TP4 shared-expert group, which independently need the input. The
prototype was removed. Revisit sparse input exchange only with shared-group
recipients included and nonblocking sends/receives overlapped with owner-side
shared projection; the serialized point-to-point schedule is not competitive.

A second experiment replaced the large MPI broadcast with the existing uTofu
recursive-doubling sum in BF16, while retaining a separate exact FP32 metadata
broadcast. A dedicated 12-rank test passed at batches 1, 6 and 16 across the
32K-element transport chunk boundary. On job 51628518 the same-job MPI baseline
was **36.473 seconds, 30.30 input tokens/s**, while uTofu took **37.469 seconds,
29.49 input tokens/s**. Both produced final token 19 and rank-0 state hash
`b233094c50f0ea24`; uTofu increased the representative batch-16 broadcast
phase from about 190 ms to 217 ms. The option and test extension were removed.
Collective substitution cannot close the remaining 3.3x gap. The next design
step is pipeline-parallel prompt flow across layer-owner groups, so different
groups process different prompt tiles concurrently and dense state moves by
point-to-point handoff rather than a world collective at every layer.

The causal-window snapshot was then reduced without changing model arithmetic.
The old verifier copied all `40*128*512` ring values for every token, consuming
160 MiB at batch 16. A tile can overwrite at most 16 old ring rows per layer;
the attention reader now keeps those rows in a 32 KiB overlay and substitutes
one only when a future tile position reused a slot needed by an earlier query.
Two batch-16 runs took **35.634 and 35.712 seconds**, or **31.01 and 30.94 input
tokens/s**, versus the same-job 36.473-second/30.30-token/s baseline. Both
retained final token 19 and exact rank-0 state hash `b233094c50f0ea24`.

Attention subphase timing on the overlay path attributed 78.3 ms per full
tile to preparation/context communication, 28.9 ms to sparse attend, and 27.0
ms to WQ-B, WO-A, TP gather, and WO-B/output gather combined. The TP4 path had
issued one approximately 9.5 KiB context broadcast per token and layer. It now
prepares the 16 contexts causally on the owner, broadcasts the contiguous
context batch once, and applies it in order on peer ranks. The validated run
took **34.758 seconds, 31.79 input tokens/s**, retained token 19 and state hash
`b233094c50f0ea24`, and reduced preparation to 75.3 ms. Eight source/index
layers still synchronize publication, selection, or candidate state per token;
batching those state messages is the next bounded attention optimization.

A trial batched those eight all-rank metadata synchronizations into one context
publication per layer. After correcting an initial collective-scope deadlock,
it completed in **34.617 seconds, 31.92 input tokens/s** and reduced maximum-rank
preparation to 73.1 ms per full tile. The final token remained 19, but rank 0's
state hash changed to `81e422b983b7c28b`; the exact reference is
`b233094c50f0ea24`. The 0.4% throughput gain does not justify carrying a
different sparse-attention state, so the synchronization rewrite was removed.
Evidence is in `tmp/ds41f/job51628518/prefill-syncbatch-b16-v2`.

With full-window snapshots gone, the outer prompt tile now accepts 32
positions. Projection and expert kernels remain internally tiled by six;
BF16 input/handoff and TP gather transports retain bounded 16-position wire
buffers and split larger calls, while the smaller context messages span the
full outer tile. The journal and causal overlay bounds were raised to match.
Three complete runs took **33.837, 33.858 and 33.700 seconds**, or **32.66,
32.64 and 32.79 input tokens/s**. All retained token 19 and exact rank-0 state hash
`b233094c50f0ea24`; rank 0 kept about 4.7 GB `MemAvailable`. Evidence is in
`tmp/ds41f/job51628518/prefill-mpi-b32-v4` through `prefill-mpi-b32-v6`.
The per-run p95 full-tile latencies were 1.043, 1.063 and 1.036 seconds.

The largest tile that keeps the raw context message below the one-megabyte
transport cap is 64. The same bounded wire operators split large BF16 and TP
payloads into 16-position calls. Three exact runs took **33.350, 33.544 and
33.449 seconds**, or **33.13, 32.94 and 33.03 input tokens/s**. Each retained
token 19 and state hash `b233094c50f0ea24`; p95 full-tile latency was
2.050--2.097 seconds. The 33.03 tokens/s mean is only 1.0% above batch 32,
showing that larger serial tiles are near their useful limit. Further progress
toward 100 tokens/s requires concurrent pipeline flow across layer-owner
groups. Evidence is in `tmp/ds41f/job51628518/prefill-mpi-b64-v1` through
`prefill-mpi-b64-v3`.

A tile-level sparse FFN input experiment broadcast route metadata once and
posted one aggregated BF16 message to each selected expert rank plus every
member of the owner's shared-TP4 group. Its dedicated 12-rank communication
test passed, and the full replay retained token 19 and exact state hash
`b233094c50f0ea24`. It nevertheless regressed to **33.748 seconds, 32.74 input
tokens/s**. Maximum-rank input publication rose from 12.143 to 12.487 ms/token;
attention and reduction skew rose as well. The point-to-point packing and
rendezvous cost exceeds the bytes saved, so the option and transport were
removed. Evidence is in
`tmp/ds41f/job51646185/prefill-sparse-b64-v1`.

Open MPI's ring scatter/allgather broadcast (algorithm 9) improves the large
batch-64 BF16 publications without changing their representation. Two diagnostic runs and one explicit-argument validation took **32.929,
32.997 and 33.034 seconds**, or **33.56, 33.49 and 33.45 input tokens/s**.
All retained token 19 and state hash `b233094c50f0ea24`.
The retained `--mpi-broadcast-algorithm 9` runner argument selects the MPI
algorithm before `MPI_Init`; production tuning does not depend on an
environment variable. Evidence is in
`tmp/ds41f/job51646185/prefill-bcast9-b64-v1`, `prefill-bcast9-b64-v2`
and `prefill-bcastarg9-b64-v1`.

## Current results, 2026-09-13

Implemented TP4 dense attention/shared projections, persistent SVE workers,
INT8 projection and packed expert SDOT paths, exact SVE cache conversion,
shared prepared expert inputs, bounded Engram row caching, and the three-stage
DSpark/MTP draft network with causal batched verification and rejected-suffix
rollback. All paths use explicit runner arguments. Local weight conversion
and packing remain below 5% of load time, so no offline INT8 weight replica is
needed. The original fixed backbone, sparse selection and cache capacity are
retained.

Three ordinary runs after a **1021-token chat prompt** reach **19.680–19.751
tokens/s**; the **1024-token code prompt** reaches **19.790–20.029 tokens/s**.
Both exclude prompt processing and measure 128 subsequent steps. Neither
prompt passes the three-repeat 20+ criterion. Earlier short-prompt runs grown
to 1K history exceed 20
repeatedly; the INT8 speculative path repeats at **20.566–20.709 emitted
tokens/s** on the capital workload, preserving the selected INT8 control's
outputs. Its burst-cycle p95 is **279–282 ms**, not a per-token latency.

No track has independent checkpoint/GPU numerical validation: the original
FP8 CPU control itself fails the independent reference gate, and the
speed-first INT8/expert/mHC approximations introduce additional differences.
Bitwise control regression and matching token IDs do not resolve this.
**Thirty and forty tokens/s remain unachieved.** Attention and the combined
FFN span dominate the final verifier profiles at about **17–18** and **12–13
ms per emitted token**, respectively.

The final runner SHA256 is
`c8a4075a41bb21ff8418d2a4780df78b01c8a14cef885d250e02c913ef68f96d`.
The preceding full suite completed successfully on allocation 51575979, and
the resumed validation on 51592153 is recorded below. See the
[long-prompt results](#ordinary-decode-after-1k-prompts) and [batched verifier
results](#final-batched-verifier-gates-and-timing) for the earlier test
commands, memory minima and evidence locations.

### Resumed allocation 51592153 validation

The 12-node normal-mode interactive allocation **51592153** (first compute
host `f26-6202c`) was resumed through the bash-over-HTTP bridge on local port
42396 (login reverse port 32396). Its per-node staging root was
`/local/u14346/ds41f-51592153`. An attached, fail-fast MPI staging pass
completed all original, dense, Engram and TP4 manifests; the 12-rank manifest
gate returned `STAGE_VERIFY_MPI_RC=0`. A detached MPI request had previously
left only rank 0 staged, and the first attached request reached the 1800-second
bridge timeout while flushing the large Engram shards. The corrected pass used
absolute script paths, `set -e`, and a 7200-second request timeout. This is the
required staging pattern for this allocation size.

All short full-model checks returned zero. The nine-token replay produced
next-token ID 270 in the exact FP8 run (0.714 s), the INT8 block-32 run
(0.596 s), and the INT8 + expert-SDOT + mHC-matvec run (0.497 s). The latter
is the speed-first approximation; matching IDs are a control regression, not
checkpoint validation. The batched MTP verifier used five-token drafts with
forced rejection at draft index 4. It verified 27 rows, emitted 23 rows across
five cycles, printed five `VERIFY_CHECK PASS` lines (including the final
three-row batch), and completed 24 requested generated tokens on all ranks;
the 12-rank exit code was zero. Per-rank state hashes are retained in
`tmp/ds41f/job51592153/mtp-verify-short-v1`.

The sustained runs use the 1024-token code history and 128 generated tokens;
the report measures the 127 steady-state positions 1024..1150. All rows below
completed on all twelve ranks with binary SHA256
`c8a4075a41bb21ff8418d2a4780df78b01c8a14cef885d250e02c913ef68f96d`:

| Configuration | Mean ms/token | p95 ms/token | tokens/s | Minimum final MemAvailable |
| --- | ---: | ---: | ---: | ---: |
| INT8 + expert SDOT + mHC matvec | 50.0018 | 51.1856 | 19.9993 | 4,768,202,752 B |
| Above + 8 MiB Engram row cache | 49.9753 | 50.9034 | 20.0099 | 4,765,843,456 B |
| Above + attention local pages | **49.6843** | **51.0560** | **20.1271** | 4,799,660,032 B |
| Sparse SDOT attention (no persistent team) | 60.5319 | 62.3369 | 16.5202 | 4,791,730,176 B |

The row-cache and local-page runs have bitwise-identical input/next-token
triples over the full 1151-position trace. Sparse SDOT completes without an
error but is slower and is not selected for the 20+ path. The local-page
configuration clears 20 tokens/s at approximately 1K history on this resumed
allocation; it remains an approximate speed-first path and does not change the
independent numerical-validation status above. A three-repeat criterion still
needs repeated launches of this exact configuration.

### TP12 shared FFN and attention experiment

The next implementation gate keeps dense TP4 for the routed path and uses an
independent TP12 communicator for the shared expert. W1/W3 are row-sharded over
all twelve ranks, their 2304-element hidden tile is BF16-allgathered, and W2 is
row-sharded in 32-row FP8 groups. The hierarchical mode adds each rank's W2
rows before a 32-row aligned reduce-scatter and gathers the reduced rows back to
the layer owner. A full gather mode remains available for comparison.

An attention TP12 layout was also staged. Ranks 0--7 each own an 8-head WQ-B
and WO-A group; WO-B rows and scales are split over all twelve ranks. Every
rank participates in the projected 1024-element allgather and the owner gathers
the final 5120-element output. `weights.shared.tp` and `weights.attention.tp`
bind the split ranges and FP8 scale pairs, while the legacy `weights.tp`
manifest continues to describe the remaining dense tensors. The synthetic mapping,
aligned communicator, and fresh-path manifest tests pass, and the actual
12-node stage reports `TP_STAGE PASS` on all ranks.

On allocation 51592153, the fixed 1105-token replay returned `next=19`, and
the TP12 attention trace matched the TP4 selected-row control for all 1232
positions. The speed-first command used hierarchical shared reduction,
selected-row 256, attention row prepack, INT8 FP8 weights, packed expert SDOT,
cached RoPE, Engram prefetch/row cache, and persistent workers. Its 128
steady-state tokens at positions 1105--1232 measured **44.904 ms/token**,
**22.269 tokens/s**, p95 **47.203 ms**, with minimum `MemAvailable`
**4,864,475,136 B**. This is an approximate speed-first result and does not
meet the 30-token/s gate.

The corresponding 32-token profile identifies the remaining wall-time budget:
attention **19.203 ms**, routed plus shared experts **11.281 ms**, expert-sum
rendezvous **4.576 ms**, FFN broadcast **1.412 ms**, residual handoff
**1.295 ms**, and Engram projection **1.288 ms**. Attention's nested spans
include QA 1.639, QB 1.743, index 1.435, sparse QK/PV 2.269, WO-A 1.256,
and WO-B 1.538 ms. The integer exp2 softmax option (`--sparse-math 2`)
reduces the profiled attention span to **18.845 ms** and the complete profile
to **21.728 tokens/s**, still below 30.

The existing integer SDOT attention remains an explicit experiment. Its
nonpersistent TP4 run measured **16.520 tokens/s** because it repacks raw rows
and retains a separate FP32 PV pass. A persistent-team SDOT probe was rejected
after the worker implementation made the first token take about 46 seconds;
the production guard therefore keeps SDOT separate from persistent tiled-FP32
attention. The next 30+ work item is an online prepacked QK/softmax/PV loop
using the integer `sdot` and `exp2` kernels, followed by owner-free shared
reduction and projection overlap. Thirty and forty tokens/s remain unachieved.

### Owner-free shared reduction continuation, allocation 51607843

The next 30+ step replaces the hierarchical shared-output
reduce-scatter/BF16-gather pair with one TP12 `MPI_Allreduce`. Each rank adds
its routed-expert result and its unique shared-W2 output-row shard before the
collective, so every rank receives the complete FFN result and the layer owner
can continue without a second gather. `--shared-reduce allreduce` selects this
mode; `full` and `hierarchical` remain as controls. TP12 attention also skips
the legacy publication and selected-ID broadcasts because its all-rank context
has already installed that state. The layer-20 candidate-block broadcast is
still required.

On normal-mode allocation **51607843**, the fixed replay returned `next=19`
at position 1104. The 128 steady-state positions 1105--1232 measured
**42.952 ms/token**, **23.282 tokens/s**, p95 **44.179 ms**, and minimum final
`MemAvailable` **4,998,299,648 B**. This improves the preceding hierarchical
result of 44.904 ms/token and 22.269 tokens/s, but does not meet 30 tokens/s.
The 32-token owner-path profile reports shared reduction **7.000 ms**, down
from about 9.96 ms in the preceding profile, and attention synchronization
**0.045 ms**, down from 0.585 ms. Attention remains **19.814 ms**. Its newly
split communication spans are context **2.707 ms**, projected-head allgather
**4.158 ms**, and output gather **3.029 ms**; shared-hidden allgather is
**1.067 ms**. Evidence is in
`tmp/ds41f/job51607843/allreduce-mpi-128-v1`.

Two transport experiments were rejected. The registered uTofu allreduce was
initially launched through the wrong LLVM-21 MPI wrapper and is not a valid
performance result; the retained runner uses LLVM 23 in the same shell as
`mpiexec`. An eight-producer `MPI_Allgatherv` removed the four zero projected
head tiles but slowed the sustained run to **45.262 ms/token**, **22.094
tokens/s**, p95 **47.098 ms**. Its lower payload does not repay the
variable-count collective overhead, so the fixed TP12 allgather remains.
Evidence is in `tmp/ds41f/job51607843/prefix8-128-v1`.

The invalid uTofu result was subsequently replaced by a correct LLVM-23 run,
and two more transport branches were closed. A bit-exact BF16 attention wire
reduced the per-layer context payload from about 9.6 to 6.0 KiB, but scalar
packing and the smaller-message `MPI_Bcast` path reduced sustained speed to
**43.924 ms/token**, **22.766 tokens/s**, p95 **45.192 ms**. All 129 replayed
`(input,next)` transitions matched the retained baseline. A contemporaneous
raw-context run using the registered uTofu recursive-doubling sum for the
5,120-float combined routed/shared output measured **43.171 ms/token**,
**23.163 tokens/s**, p95 **44.594 ms**, also with an exact 129-transition
trace. It did not improve on the retained MPI result of 23.282 tokens/s. Both
code paths were removed. Evidence is in
`tmp/ds41f/job51607843/contextbf16-128-v2` and
`tmp/ds41f/job51607843/utofu-raw-128-v1`.

The bounded profiles immediately after an unrelated long MPI workload had
substantially more collective skew than the earlier clean profile (for
example, raw-context communication was 12.772 ms versus 2.707 ms), so their
nested communication spans are not suitable for estimating absolute savings.
The 128-token wall times and exact token traces remain the admission evidence;
the compact-wire run was also slower than the raw-wire run under the same
post-workload conditions.

The next 30+ implementation must remove at least **9.619 ms** from the new
42.952 ms baseline. The measured path is to distribute attention preparation
(WQ-A, WKV and index query/score ownership) instead of only WQ-B/WO, and to
replace the two attention gathers with fixed-count topology-aware collectives
or direct owner delivery. A payload-only `Allgatherv` change is insufficient.
The admission gates remain <=12 ms total attention and <=33.333 ms/token,
with the same 1105-token replay and full token-trace comparison.

The next implementation task is therefore a staged TP12 attention-preparation
layout. Shard WQ-A and WKV output rows across all ranks, replicate only their
small norm and rotary metadata, and assign index-query/score tiles to ranks
that already own the corresponding attention head groups. Quantize the layer
input once, broadcast that compact activation before preparation, then merge
the WQ-A/WKV shards with fixed-count collectives. Measure WQ-A, WKV, index and
context synchronization separately; retain the layout only if total attention
falls below 15 ms before pursuing the 12 ms admission gate. Per-rank staged
bytes, HBM headroom, the position-1104 `next=19` check, and the complete
positions-1104--1232 token trace are mandatory validation outputs.

Implementation is now gated as `--attention-prepare-tp8` in both `stage_tp.py`
and the runner. WQ-A is split into 160 rows/rank and WKV into 64 rows/rank on
ranks 0--7, with their FP8 scale rows split at the same 32-row boundaries.
Only rank 0 stages the complete 1280/512-element Q/KV norm vectors. At decode,
the layer owner publishes the exact-BF16 normalized input, the eight ranks
compute both local projections, and one packed BF16 attention-communicator
allgather plus world broadcast reconstructs both vectors. Rank 0 performs the
original norms, KV RoPE and activation quantization, then publishes the 1792
normalized Q/KV elements in one BF16 broadcast. The layer owner retains source
compression and index selection, publishes only selection/publication
metadata, and execution rejoins the retained TP12 WQ-B/sparse/WO path.

The old staging format remains version 1 and is unchanged unless the new flag
is passed; preparation staging writes `DS41FA` version 2 and the runner rejects
version 1 when the new execution flag is selected. The rank-layout tests cover
complete, nonoverlapping WQ-A/WKV rows and matching scale geometry. The full
runner cross-compiles with LLVM 23 on the Fugaku login frontend. Remote
correctness and performance validation are pending a 12-node allocation; use
the existing 1105-token replay and first compare the entire token trace before
using its timing.

An actual 48-shard model inventory dry run passes on all twelve layouts. The
largest resident plan is rank 0 at **26,218,924,192 bytes**, only 9,313,024
bytes above its retained TP12 layout; ranks 8--11 become smaller because they
no longer own complete per-layer WQ-A/WKV matrices. The largest bounded copy
is 801,205,632 bytes on rank 2. This stays within the prior HBM admission
envelope and makes restaging practical within a short validation allocation.

## Proposed path to 30+ and 40+ tokens/s

The current 1K-history speed-first baseline is 49.684 ms/token (20.127
tokens/s) with INT8 FP8 weights, packed expert SDOT, approximate mHC, an 8 MiB
Engram row cache, fresh attention pages and persistent workers. The latest
buffered profile puts attention near 20--22 ms, routed/shared FFN near
11--13 ms, mHC near 2.7 ms, gating near 1.1 ms, and the remaining broadcasts,
handoffs and rendezvous on the critical path. These profile spans overlap;
they must not be added as independent work.

Thirty tokens/s requires a step at or below 33.333 ms, removing at least
16.351 ms from the current run. Forty requires at or below 25.000 ms, removing
24.684 ms. The working critical-path envelopes are:

| Envelope | Current indication | 30+ gate | 40+ gate |
| --- | ---: | ---: | ---: |
| Attention preparation, projections, index and sparse QK/PV | 20--22 ms | <=12 ms | <=7 ms |
| Routed/shared FFN | 11--13 ms | <=7 ms | <=5 ms |
| mHC, norms and gate | about 4--6 ms | <=4 ms | <=2.5 ms |
| Communication, handoff, head and other | remainder | <=10 ms | <=9.5 ms |
| **Total wall time** | **49.684 ms** | **<=33.333 ms** | **<=25.000 ms** |

The 30+ implementation should proceed in these gates:

1. **Prepack and fuse attention.** Store compressed-KV integer tiles and their
   group scales when a source row is published, so decode does not repack or
   unpack the same row. Replace the current separate `sparse_sdot` QK,
   softmax and PV passes with a reusable score workspace or an online
   max/subtract/exp2/PV loop. Adapt
   `~/work/clair/a64fx/a64fx/llm-guided-opt/int_exp2_sdot_a64fx.s` only after
   preserving the sink in the maximum and denominator, the `[-31,0]` exponent
   domain, masks and tails. The resumed sparse-SDOT run is 16.520 tokens/s
   because it still pays packing and phase overhead; that path is not a
   candidate until those costs disappear. First target: <=15 ms attention;
   advance only when fixed-input logits and token IDs are checked.
2. **Reduce selected-row work on the speed-first track.** Benchmark adaptive
   512/384/256 selected rows with the original index and tie rules. Keep the
   FP32 PV path initially; use 256 only if its fixed-history error and output
   behavior are explicitly recorded. The 30+ gate needs <=12 ms total
   attention, including index, projections, row preparation and collectives.
3. **Finish the FFN critical path.** Quantize one activation per layer, keep it
   resident across all local routes, and tile fused W1/W3/W2 SDOT so expert
   setup, scale loads and output stores are shared. Replace the full-vector
   expert sum with a topology-aware reduce-scatter/allgather experiment, and
   row-shard the shared expert across every participating rank. Preserve the
   current ordered FP32 control as a separate track. Target <=7 ms for the
   routed/shared envelope.
4. **Fuse mHC and synchronization.** Combine mHC pre/post, norm and BF16
   rounding passes where the selected track permits it; keep the original
   reduction order for the validated track. Pack route, selection and
   residual metadata once, post nonblocking owner-to-next-owner transfers,
   and overlap shared-expert work with routed reduction while retaining uTofu
   acknowledgements. Target <=4 ms for mHC/norm/gate and <=10 ms for all
   communication/head/other work.
5. **Use all twelve nodes for dense work if the first four gates miss 33.333
   ms.** Current TP4 activates four ranks for a layer group. Add a staged TP12
   layout for QKV, sparse attention, WO-A/WO-B and the shared expert, then
   measure allgather/gather latency against the compute saved. TP12 is the
   likely 30+ step when attention remains above 12 ms; it must be admitted
   only with per-rank memory and transport measurements.

Forty tokens/s is a second architecture gate, not a smaller tuning pass. It
requires TP12 (or an equivalent all-rank schedule), prepacked fused
attention at <=7 ms, <=5 ms routed/shared FFN, <=2.5 ms mHC/norm/gate and
<=9.5 ms for communication and everything else. This implies a fused
owner-free layer schedule: all-rank dense projections, hierarchical expert
reduction, shared-expert overlap, and a single residual handoff. The current
sparse-SDOT implementation, TP4 owner groups and extra per-layer barriers
cannot reach that budget by themselves.

Exact-output 40+ is a separate possibility through MTP speculation. A cycle
that emits three tokens must finish below 75 ms; the present verifier and
forced-rejection measurements are far above that, so MTP needs the same
TP12/fused attention and batched FFN work before it can be considered. Count
actual emitted tokens including rejected drafts and commit time, rather than
draft length. Keep this path separate from the approximate ordinary track.

Each retained speed-first stage needs three fresh 1K-history repeats, every
repeat above its target, p95 and minimum `MemAvailable`, and a full input/
next-token regression against its predecessor. The validated track additionally
keeps the independent FP8 cosine/RMS gate; no 30+ or 40+ claim is promoted
from an approximate path without labeling it.

## Implementation continuation: persistent workers and SDOT, 2026-09-13

The ordinary **speed-first 20+ milestone is repeatable** near 1K history on
12 A64FX nodes at 2 GHz. This does **not** establish a numerically validated
20+ path. All runs retain 40 layers, 64 heads, top-6 routed experts, top-512
selected rows and the 128-token window. Cache capacity is 1M; measured history
is approximately 1K. Thirty and forty tokens/s remain unachieved targets.

Three uninstrumented runs on allocation **51575979** use TP4, packed expert
SDOT, INT8 FP8 projections, approximate mHC mode 1, persistent workers,
cached RoPE and vector INT8 scale loads. They use the six-token capital prompt,
1100 outputs and samples at positions 1000..1104 (105 samples):

| Repeat | Mean ms/token | tokens/s | p95 ms/token | Minimum final MemAvailable |
| --- | ---: | ---: | ---: | ---: |
| 1 | 49.205 | 20.323 | 52.456 | 3,308,453,888 B |
| 2 | 48.687 | 20.539 | 51.847 | 3,283,484,672 B |
| 3 | 48.666 | 20.548 | 51.989 | 3,250,913,280 B |

All twelve ranks finish, and all 1105 input/next-token triples agree between
these repeats and their approximate profile control. Whole-run times are
56.057/55.421/55.718 seconds; these include the growing shorter-history prefix.
Binary SHA256 is
`ac3889d7022c54122ccc2f892939f8141a2935457bb08fa51fdd4dd4ddf7a683`.
Evidence and immutable launch scripts are under
`tmp/ds41f/job51575979/scale-rope-runs-v1` and
`tp4-scale-rope-approx-repeat{1,2,3}-v1/summary.json`.
The separate instrumented run measures 19.802 tokens/s. The newest fused
SDOT pair reaches 20.111 tokens/s instrumented on the earlier allocation.
Its completed repeats and fixed-history checks appear in the continuation
below; the table above belongs to the preceding binary.

### Numerical status

The ordinary INT8 version at this stage preserving the existing INT8
control's outputs reaches
**16.513 tokens/s**, 60.558 ms/token instrumented. All 1105 token triples and
nine early logit arrays are bitwise identical to that control. The FP8 short
run also retains its nine control logit arrays bit-for-bit. These are regression
checks against selected controls, not independent checkpoint validation.

The speed-first configuration above fails the early nine-position FP8 gate:
minimum cosine **0.996026**, maximum relative RMS **9.081%**, despite matching
all nine argmax IDs. The continuation below records the failed fixed-history 1K gate. Earlier
individual approximate variants also fail: expert SDOT alone has minimum
cosine 0.987906 and RMS 19.618%; sparse SDOT alone has cosine 0.996813 and RMS
8.017%. The gate remains cosine >=0.999 and relative RMS <=1%, with argmax
agreement; no threshold was relaxed.

The corrected independent NumPy nine-position replay has now **finished**.
The original FP8 control agrees on all nine argmax IDs, but only position zero
passes the full numerical gate: five positions fail cosine, eight fail relative
RMS. Minimum cosine is **0.994260** and maximum relative RMS **10.782%**.
The histories are verified identical. See
`tmp/ds41f/job51569201/reference9-v1/comparison-fp8.json`. Consequently neither
FP8 nor INT8 currently has independent numerical validation or official GPU
parity. This supersedes earlier notes saying that reference was pending.

### Implemented and tested

- `--persistent-team` creates one OpenMP team around inference. Thread zero
  remains the MPI caller under MPI_THREAD_FUNNELED; workers acquire published
  jobs and signal completion through separate cache lines. Quantization,
  projections, expert kernels, mHC, gate/index and tiled FP32 attention use
  this team. The original OpenMP path remains available. Dispatch falls from
  5.850 to 1.831 microseconds on A64FX. Partition, repeated-generation,
  main-thread and canary tests pass. The combined kernel test checks 36,756
  output floats bitwise against ordinary execution, including nonfinite cases.
  The full INT8 run improves from 13.627 to 15.573 tokens/s while retaining
  all 1105 token triples and nine logit arrays exactly.
- `--quant-parallel` avoids parallel-region overhead for small vectors. The
  ordinary-team threshold is 5120; the cheaper persistent dispatch uses 1024.
  Nine sizes, in-place operation and nonfinite rejection pass native/A64FX
  tests. Gate top-k uses a sorted bounded insertion list with strict comparisons
  and ascending-ID ties, retaining original probability/sum order. Fifty cases
  pass; the standalone gate falls from 51.885 to 15.710 microseconds.
- `--expert-sdot` packs original MXFP4 nibbles losslessly into four-row tiles,
  retaining original group-32 E8M0 scales and **the original resident byte
  count**. Queries use INT8 after the existing FP8 activation boundary; exact
  E2M1*2 integer weights feed SVE SDOT. The added input quantization and changed
  dot order make the operator approximate. Cold complete single projections
  improve 1.69x/1.94x on 2304x5120 and 5120x2304. Tests cover lossless packing,
  integer oracle, varied scales, zeros/nonfinite inputs, canaries and bounded
  loader memory. Packing costs 2.44–2.47 seconds/rank versus about 68 seconds
  loading; FP8-to-INT8 conversion adds roughly 0.45 seconds. Preparation stays
  below 5% of load time, so conversion after reading `/local` is retained.
  No shared offline INT8 weight copies are needed for this implementation.
- With packed experts, `--expert-fused 1` now fuses W1/W3 in a single SDOT
  loop sharing input loads. It matches two prepared SDOT GEMVs bitwise with
  distinct weights/scales. Cold paired operators improve 79.730 to 61.613 us
  and 81.057 to 58.344 us (1.29x/1.39x). A64FX and native tests pass.
- INT8 GEMV loads four weight scales together and expands them in SVE lanes,
  preserving every scale product and reduction. All 378 existing 1..6-token
  batched cases remain bitwise equal to GEMV. `--rope-cache` keeps eight bounded
  thread-local entries of the original double sine/cosine calculations; its
  key includes position, both theta settings, original-context setting and
  inverse direction. It changes no trigonometric or rounding formulas. All
  720 layout/cache/inverse/long-position cases pass bitwise. The complete
  INT8 regression with these changes reaches the 16.513 tokens/s above.
- `--sparse-sdot` quantizes raw keys/queries and packs exact compressed integer
  keys, rescales each group into FP32, then retains FP32 PV and the sink/mask
  semantics. Twenty-five oracle, boundary, tail and nonfinite cases pass.
  **It is slower including packing**: 16 heads and 640 rows take 127.870 us
  versus 75.673 us for FP32 tile four. Retain `--sparse-tile 4 --sparse-math 0`;
  sparse SDOT is an explicit experiment and is incompatible with persistent
  workers until that experimental operator has its own worker implementation.

The latest fused-pair profile spends 22.060 ms/token in attention, 11.335 ms
in routed/shared experts, 2.704 ms in mHC, 1.115 ms in gating, 2.478 ms in FFN
broadcast, 1.498 ms in residual handoff and 2.290 ms in expert rendezvous.
These are measured critical-path spans; nested kernel times are not additive.
Further ordinary 30/40 work must substantially reduce attention and experts;
weight-bandwidth arithmetic alone does not predict the end-to-end rate.

Build and representative checks (repository root; native tests use separate
executables under repository `tmp/ds41f`):

```sh
TMPDIR="$PWD/tmp/ds41f" make -C a64fx/ds41f a64fx \
  A64FX_CC=fccpx A64FX_MPICC=mpifccpx \
  A64FX_CFLAGS='-Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -Wall -Wextra -Wpedantic'
# Inside the allocation, sequentially, with OMP_NUM_THREADS=48,
# OMP_PROC_BIND=close OMP_PLACES=cores OMP_WAIT_POLICY=active:
mpiexec -np 1 ./test_team
mpiexec -np 1 ./test_team_kernels
mpiexec -np 1 ./test_gate
mpiexec -np 1 ./test_quant_parallel
mpiexec -np 1 ./test_fp4_sdot
mpiexec -np 1 ./test_sparse_sdot
mpiexec -np 1 ./test_input_cache
mpiexec -np 1 ./test_rope_layout
mpiexec -np 1 ./test_int8
mpiexec -np 1 ./test_int8_batch
```

The retained fast argument additions are `--dense-tp 4 --sparse-tile 4
--index-head-tiles --expert-fused 1 --linear-input-cache --quant-parallel
--persistent-team --rope-cache --fp8-int8-block 32 --expert-sdot --hc-matvec 1`,
plus `--engram-prefetch --engram-scale-cache --hc-mix-sve --shared-overlap
--weights-local-pages --mpi-broadcast --compact-comm` and the staged TP4 root.
Omit `--expert-sdot --hc-matvec 1` to retain the INT8 control's outputs.
Omit INT8 conversion as well for the FP8 regression control.

Allocation 51569201 expired. The following continuation used **51575979**,
scheduled through 04:50:08 JST September 13, bridge 42395/32395/21266. Original
and TP4 staging completed
with exact-byte manifests. One MPI program runs at a time; drivers and
binaries are immutable snapshots with verified hashes. Repeats above retain
at least 3.25 GB final MemAvailable, above the 2 GiB admission floor.

The following continuation records the completed paired-path, fixed-history,
chat/code and DSpark work. The independent FP8 mismatch remains unresolved;
no track has independent checkpoint validation.

## Validation and DSpark continuation, 2026-09-13

The fused SDOT pair binary (`997e966b83487cc6864fb64d2629b223309d89bcd147ed61360be14b0ec5dbb9`)
finishes its own three uninstrumented capital-prompt repeats at
**20.523 / 20.039 / 20.330 tokens/s**, with p95 **52.528 / 54.640 / 53.199 ms**.
All 1105 token triples match its approximate profile control. Final minimum
MemAvailable is 3.279 / 3.249 / 3.213 GB. The separate profile is 20.302 tokens/s.
There is no clear end-to-end improvement over the preceding unfused SDOT
configuration once run variation is considered.

The 20+ result is prompt-dependent. One uninstrumented chat run reaches
**19.360 tokens/s** (51.653 ms mean, 53.259 ms p95), and one code run reaches
**20.256 tokens/s** (49.368 ms mean, 52.320 ms p95), again at positions
1000..1104. These are single measurements, not three-repeat milestones.
The checkpoint chat encoder is used; exact messages and token IDs are in
`tmp/ds41f/job51575979/validation-runs-v1/{chat,code}.{json,ids}`.

Full fixed-history checks now establish that the newest FP8 and INT8
regression paths preserve all nine logits at positions 1000..1008 bit-for-bit.
The optimized FP8 path also matches original FP8 logits bit-for-bit on nine
positions near the end of each chat/code prompt. The approximate path fails:

| Fixed inputs | Minimum cosine vs FP8 | Maximum relative RMS | Saved argmax matches |
| --- | ---: | ---: | ---: |
| 1K replay | 0.900356 | 44.585% | 9/9 |
| Chat prompt | 0.956396 | 29.682% | 8/9 |
| Code prompt | 0.968010 | 25.852% | 9/9 |

These results confirm that token agreement cannot replace the numerical gate.
No quality-validated 20+ claim is made.

The DSpark/MTP implementation now includes:

- `stage_mtp.py` inventories all **2401 MTP tensors**, 7,932,874,632 source
  bytes, and stages exact bounded ranges separately under
  `/local/u14346/ds41f-51575979/mtp/rank<R>`. Stage owners are 0/4/8;
  dense/main projections use TP4; 128 draft experts use expert-ID modulo 12;
  Markov embedding/head rows use all twelve ranks. Backbone embedding/head
  are reused. All rank staging, row metadata and converted-load tests pass.
  Original MTP shards are 613.3–703.8 MB/rank; converted resident shards are
  618.1–712.2 MB/rank. Admission reads the small index before payload loading.
- The draft network implements all three stages, the full five-position
  seed/noise block, hidden taps before layers 37/38/39 attention, main projection
  and KV seeding, uncompressed theta=10000 RoPE, all-five-draft-key attention,
  mHC, top-3 MoE, shared backbone head, Markov bias and confidence. Draft KV is
  temporary; only committed main hidden positions update the MTP windows.
- A bounded NumPy replay uses the original safetensors and captured main taps.
  At position eight, all five FP8 draft argmax IDs match: seed 270, proposals
  3669/223/22/14/270. Minimum logit cosine is 0.999553, maximum relative RMS
  3.012%; thus this does not pass the full 1% RMS gate or establish GPU parity.
  The approximate draft differs at one proposed ID and has up to 22.647% RMS
  error; later Markov inputs then differ too, so those later logit differences
  are sequence-level draft differences. Confidence never bypasses verification.
- `ds41f_journal` records only overwritten window/cache rows, pools, selected
  IDs/candidates, publication bytes and Engram history/counters. The bound is
  **1,375,280 bytes/rank for six inputs at 1M capacity**. Native and A64FX tests
  pass all 297 prefix/rejection, window wrap, compression, top-k and memory
  cases. Prefetch drain waits for pending reads and invalidates stale results;
  generations remain monotonic. Updated 96-generation tests pass on both hosts.
- `--speculate 2/4/5` is off by default. The initial sequential verifier
  calculates every proposal's causal output, commits only the accepted prefix
  plus bonus input state, and updates MTP KV for each committed position.
  All twelve sequential-verifier cases pass: draft lengths 2/4/5, forced
  rejection at positions zero through four, window position 127, selector
  position 511, FP8 control, and a complete 1100-output INT8 run. The full run
  preserves all 1105 token triples and nine early logits exactly. Around 1K,
  complete cycles emit 101 tokens at **15.436 tokens/s**, averaging 5.611 emitted
  tokens/cycle and 1.050 verified inputs/emitted token. Draft/verify/commit cost
  is 2.427/62.146/0.213 ms per emitted token; final minimum memory is 2.447 GB.
  This is slower than ordinary decode and is not a speculative milestone. Ordinary profile mode is rejected with speculation,
  and `SPEC_CYCLE` records actual draft/verify/commit time and emitted counts.
- The causal batched verifier passes the regression checks below. It shares INT8 projection
  weights for 1..6 inputs while preserving GEMV accumulation and BF16 boundaries.
  Each input has its own bounded 128-token window and small selector state;
  compressed append data remains shared with position-bounded reads. Six window
  views cost about 63 MB/rank, never a copy of the full 1M compressed cache.
  All batch sizes 1..6 preserve nine early FP8 and INT8 logits bit-for-bit;
  INT8 batch 3 and FP8 batch 6 also preserve nine fixed 1K logits bit-for-bit.
  The diagnostic sequential/batched runs pass complete state, tap and token
  checks, including grouped experts and forced rejection at all five positions.
  Actual emitted-token performance is reported separately below. `--verify-expert-batch` adds
  FP4/packed-SDOT weight reuse for common expert IDs, with outputs retained by
  original route slot so that each token's sum order stays unchanged; full
  A64FX kernels pass all 384 stride/canary/persistent-team cases; the full
  expert passes every batch size 1..6 bit-for-bit for original FP4 and packed
  SDOT. Complete expert speedups are about 1.2–1.5x for most batches, but the
  original six-token packed case was only 0.581/0.570 ms. The latest six-input
  implementation uses two three-input tiles per output row in one worker
  dispatch. Its complete packed expert measures **0.405 ms versus 0.597 ms**
  for six sequential fused calls; the original FP4 expert measures **0.661
  versus 0.994 ms**. The batched API receives already FP8-quantized inputs;
  full-run timing includes their preparation. These are current operator
  comparisons, not an isolated end-to-end
  attribution to the tile change. The latest 384 A64FX projection cases also
  cover NaN scales; native undefined-behavior trap instrumentation passes.
  Full-model checks precede throughput claims.

Evidence is under `tmp/ds41f/job51575979/`: `mtp-stage-check-v1`,
`mtp-probes-v1`, `mtp-reference-v1`, `spec-sequential-tests-v1` and
`verifier-checks-v1`. Kernel conversion and staging stay online from `/local`;
no new offline quantized-weight copies have been written to shared storage.
Grouped experts now pass early and fixed-1K FP8/INT8 checks, all five forced
rejection positions, EOS for prefixes 2/4/5, and output limits 1/2/3. EOS and
limit cases also preserve final backbone/MTP state hashes on all twelve ranks.
Full 1K speculative comparison and actual emitted-token timing for prefixes
2/4/5 are reported below. Thirty and forty tokens/s remain unachieved targets.


The first full grouped-verifier INT8 runs preserve all 1105 triples and nine
early logits. Actual complete-cycle throughput near 1K is:

| Verified draft prefix | Emitted tokens/s | Emitted/cycle | Verify ms/emitted | Minimum final memory |
| --- | ---: | ---: | ---: | ---: |
| 2 | 18.540 | 2.889 | 48.814 | 2.274 GB |
| 4 | 19.372 | 4.636 | 48.443 | 2.210 GB |
| 5 | 19.261 | 5.611 | 49.096 | 2.263 GB |

These use binary `f6f85814893bf59206e600a8aa3e6be264cb8af4aeb1dc0742aea02ed57cc5ac`.
Four is preferred within the 1% timing tie. There is no 20+ speculative INT8
milestone yet: increased acceptance does not remove the per-input verifier
cost. Phase tracing and the later row-cache/batched-transport experiments
must establish the next improvement. The approximate prefix-five run reaches
only **16.179 tokens/s**, averaging 4.04 emitted tokens/cycle: verification costs
57.867 ms per emitted token because more proposals are rejected. Its complete
1105-token sequence still matches the approximate control. Speculation is not
yet a speed win over ordinary approximate decode.

The first phase-trace implementation perturbs timing: the INT8 prefix-four
trace run spends 4.395 ms/emitted in commit versus 0.314 ms without tracing,
as ranks catch up after shared-storage log writes. The replacement buffers
at most 256 recent records/rank (595,968 bytes), including per-layer owner
timings, and writes them after the timed run to `verify-timing.rank<RR>.log`.
`verify_report.py` attributes each layer to its owner and compares the resulting
path with actual verification time. Nested maximum expert durations are
reported separately. Earlier uninstrumented rates remain the performance
reference; do not treat the synchronous logging delay as model compute.

### Final batched verifier gates and timing

Binary SHA256
`a75e9981b30057cccfae6ca5f69cf73b341cbf19e5951198ce3c6234725f2637`
combines the row cache, shared expert inputs, six-input microtiles, batched
transport and buffered tracing. The short FP8/INT8/approximate checks pass
sequential-versus-batched state, hidden taps and tokens. The INT8 batch-six
fixed-input replay preserves all 1105 triples and all nine full logits at
positions 1000..1008. EOS preserves final backbone and MTP state on all twelve
ranks. These are selected-control regressions, not independent validation.
Evidence is under `final-verifier-runs-v2` and `final-verifier-*-v1`.

The first full INT8 five-draft run reaches **20.662 emitted tokens/s** over
18 complete cycles at positions 1004..1104, emitting 101 tokens. Mean emitted
count is 5.611/cycle; draft/verify/commit cost is **2.623 / 45.463 / 0.311
ms/emitted**. Minimum final memory is 2.321 GB. All 1105 triples and nine early
logits match the INT8 control. Cycle p95 is **280.521 ms**, distinct from
amortized per-token latency. The buffered profile and repeat results follow below.
This final version timed INT8 prefix five only; earlier 2/4/5 results
belong to the preceding binary and do not prove the new optimal prefix.

The approximate two-draft case reaches **20.993 emitted tokens/s**, averaging
2.667 emitted/cycle. Draft/verify/commit costs **5.492 / 41.715 / 0.428
ms/emitted**, with minimum final memory 2.250 GB. It preserves all 1105 triples
and nine early logits of its approximate control. Completed prefix-four/five
and selected-prefix results follow below; no speculative 30/40 claim is made.

The first approximate five-draft attempt in `final-verifier-runs-v2` stopped
before inference with ENOMEM from the conversion budget. Rank 0 started with
30,756,896,768 bytes available, about 31 MB less than the preceding run. The
static limit reserved future KV/Engram/MTP allocations while also charging the
currently coexisting source and replacement tensor. It could reject a safe
transient conversion. `weight_prepare_limit()` instead uses current resident
bytes plus current MemAvailable above the **unchanged 2 GiB floor**, while
retaining initial loading admission, later allocation checks and the final
resident guard. No kernel, dtype, context capacity or decoding setting changed.

The corrected binary is
`c8a4075a41bb21ff8418d2a4780df78b01c8a14cef885d250e02c913ef68f96d`.
`final-verifier-runs-v3` completed the missing five-draft case in
`final-spec-approx-d5-full-v2`, then profiled and repeated the selected prefixes.
The retry loads with 2,279,473,152 bytes available on rank 0. Existing completed
results from the prior binary remain valid; the aborted attempt has no token
rate. Queued long-prompt drivers stopped before launching MPI when their
predecessor failed and are replaced by the bounded combined suite below.

The corrected five-draft approximate retry finishes at **17.678 tokens/s**
with 4.04 emitted/cycle and 3.665 / 52.543 / 0.360 ms/emitted in draft / verify /
commit. Minimum final memory is 2.189 GB; all 1105 triples and nine early logits
match the approximate control. Prefix two remains the selected approximate
candidate; prefix four measured 18.772 tokens/s with 3.643 emitted/cycle.

The buffered approximate prefix-two profile reaches **20.751 tokens/s**.
Its owner-attributed verifier spans are:

| Span | ms per actual emitted token |
| --- | ---: |
| Begin: snapshots and Engram preparation on rank 0 | 1.669 |
| Pre: attention mHC/norm and Engram projection/collection | 3.369 |
| Attention | 18.080 |
| Gate: attention post, FFN mHC/norm and routing | 3.489 |
| FFN broadcast | 2.223 |
| Owner routed experts | 1.869 |
| Shared expert and routed reduction/wait | 8.906 |
| FFN post and residual handoff | 1.743 |
| Head | 0.619 |
| **Reconstructed verifier** | **41.969** |
| **Measured verifier** | **42.177** |

The combined FFN envelope is 12.999 ms; the 6.725 ms maximum routed-expert
work is nested inside it and must not be added. Layer zero accounts for
0.953 ms of the broadcast span, largely exposing start-of-batch arrival skew;
this is not a wire-only measurement. Drafting costs 5.575 ms/emitted and commit
0.438 ms. Attention and the FFN envelope dominate. Evidence is
`final-spec-approx-d2-profile-v1/verify-summary.json`. Compiled SVE stores
also confirm the large causal windows/candidate buffers are physically cleared
before their admission check; the earlier reservation failure was not a missing
window initialization.

The selected approximate two-draft path now finishes all three uninstrumented
repeats at **21.073 / 20.944 / 20.875 emitted tokens/s**. Every run preserves
all 1105 triples and nine early logits. Amortized-token p95 is **66.855 /
67.538 / 67.643 ms**; burst-cycle p95 is **133.047 / 134.610 / 134.830 ms**.
Minimum final MemAvailable is **2.219 / 2.225 / 2.219 GB**. These repeats use
the corrected `c8a4075a...` binary. This is repeatable 20+ speculation
against the approximate control, but it does not beat the faster ordinary
approximate capital measurements. The completed INT8 repeats are recorded below.

The buffered INT8 prefix-five profile reaches **20.625 tokens/s**. It measures
**45.664 ms/emitted** in verification; the owner reconstruction is **45.496**
(difference 0.168). Attention is **17.124**, the combined FFN envelope **12.441**,
and the pre/gate spans **6.540 / 6.667 ms/emitted**. These latter spans include
ordered mHC, norms, Engram work and routing, not mHC alone. Begin, post/handoff
and head cost **0.610 / 1.579 / 0.535**. Draft/commit add **2.510 / 0.312**.
Maximum routed-expert work is **8.452 ms**, nested inside the FFN envelope.
See `final-spec-int8-d5-profile-v1/verify-summary.json`.

The INT8 five-draft path completes three uninstrumented repeats at
**20.566 / 20.679 / 20.709 emitted tokens/s**, mean **48.623 / 48.359 /
48.288 ms/emitted**. All 1105 triples and nine early logits match the INT8
control. Each selected window contains 18 complete cycles and 101 emitted
tokens at positions 1004..1104. Amortized-token p95 is **47.449 / 47.083 /
46.892 ms**; burst-cycle p95 is **281.457 / 282.169 / 278.884 ms**. The
amortized p95 can be below the mean because a small number of low-acceptance
cycles contribute fewer than 5% of emitted tokens. Minimum final MemAvailable
is **2.235 / 2.212 / 2.312 GB**, above 2 GiB. These repeats use the corrected
`c8a4075a...` binary. This establishes repeatable 20+ speculation against the
selected INT8 control on this prompt; it does not establish FP8 or GPU parity,
a new optimal INT8 draft prefix, or a prompt-independent speculative milestone.

### Exact SVE cache conversion

`--cache-sve` vectorizes the FP4 packing used by the sparse index and unpacking
of selected compressed KV rows. It retains original BF16 rounding, scale
calculation, rounded-distance comparisons, ties and signed zero. All 17,408
A64FX conversion cases pass. The 32-query packing/unpacking operator falls
from 196.708 to 39.829 microseconds; 512-row KV unpack falls from 132.459 to
18.051 microseconds, including the existing OpenMP launch. Early model logits
are bit-for-bit identical on FP8, INT8 and the selected approximate control.
Fixed-1K FP8 and INT8 logits both pass bit-for-bit, as do all early control
checks. Evidence is under `tmp/ds41f/job51575979/cache-sve-runs-v1`. The separate full-run profile is **47.386 ms/token, 21.103 tokens/s**.
Index preparation falls from 2.960 to 1.585 ms and selected-row work from
1.496 to 1.194 ms; total attention is 19.947 ms, experts/shared 11.062 ms.
The three uninstrumented capital repeats reach **20.583 / 21.070 / 21.309
tokens/s**, with p95 **54.732 / 51.476 / 50.516 ms** and minimum final
MemAvailable **3.224 / 3.197 / 3.199 GB**. All 1105 triples match the approximate
control. Binary SHA-256 is
`fa58eb9206e5c9f281946bd4652a3bfd82726d555807483abaabcb657ea4da3e`.
This is a repeatable 20+ approximate result. Code repeats are **20.731 /
20.755 / 20.896 tokens/s**, with p95 51.544 / 51.253 / 50.747 ms and final
minimum memory 3.178 / 3.153 / 3.146 GB. All 1148 code triples match their
approximate control. Chat repeats remain below the milestone as detailed below.

`--expert-input-cache` additionally prepares the original FP8 activation and
optional SDOT input once per rank/layer, then shares them across locally owned
routed experts. Hidden activation preparation remains per expert, and original
route accumulation order is retained. Early FP8/INT8/approximate and fixed-1K
FP8/INT8 regression checks pass. The separate profile reduces the slowest-rank
expert quantization component from 1.764 to 1.410 ms, with total latency
47.616 ms (21.001 tokens/s). Capital repeats reach **21.394 / 21.527 / 21.526
tokens/s**, p95 **49.796 / 49.885 / 50.276 ms**, minimum final MemAvailable
**3.077 / 3.118 / 3.064 GB**. All 1105 triples are unchanged. Binary SHA256 is
`f5d47ab4c997a5c7091e5001231ab8f28ea694148997bca5fc50b3b2ca53b363`;
evidence is in `expert-input-runs-v1` and `expert-input-approx-*-repeat*-v1`.
Chat repeats reach **19.710 / 19.737 / 19.017 tokens/s**, p95
**54.338 / 54.909 / 60.812 ms**, minimum final MemAvailable
**3.072 / 3.062 / 3.061 GB**; all 1133 triples match their control.
This does not establish a chat speedup or a prompt-independent 20+ milestone.

The chat repeats are 20.040 / 19.937 / 19.291 tokens/s (p95 51.884 / 51.779 /
59.602 ms), so this cache-SVE-only configuration does not hold 20+ across
prompts. The later row-cache results below do. The profile
still exposes 1.278 ms/token of Engram I/O. The next bounded experiment, `--engram-row-cache-mib 8`, is an
8 MiB-budget immutable BF16 row cache (4,259,840 bytes actually allocated): direct-mapped entries keyed by table and
row, no cache insertion after a failed read, and unchanged logical lookup
counters. One prefetch reader owns it; drain before clearing. It survives
speculative rollback because cached weight rows are immutable. Include its
allocation in admission and test collisions, table isolation, hits, errors,
threaded prefetch and cache clear before measuring end-to-end latency. The
expanded native and A64FX 96-generation prefetch tests pass all these cases.
The forced-rejection state/tap/token test, early FP8/INT8/approximate checks,
and fixed-1K FP8/INT8 checks all pass in `row-cache-runs-v1`. Its first profile
shows **48.584 ms/token, 20.583 tokens/s**, with 16,005 hits and 37,035 misses
across all ranks over the full input sequence (30.175% hits). Critical-path
Engram I/O is 1.162 ms versus 1.173 ms without the row cache. This alone does
not establish a useful speedup. The three capital repeats are **21.068 /
20.861 / 21.188 tokens/s**, p95 **52.417 / 53.302 / 51.406 ms**, minimum final
MemAvailable **3.073 / 3.081 / 3.084 GB**. All are slower than the 21.394–21.527
range without the row cache, so keep it off in the retained ordinary path.
Speculative verification keeps it as a separate experiment because rejected
proposals can increase repeated reads. Chat shows a different outcome:
**20.612 / 20.692 / 20.620 tokens/s**, p95 **53.315 / 50.809 / 53.224 ms**,
minimum final memory **3.056 / 3.045 / 3.041 GB**. All 1133 triples agree.
Its 27,345 hits and 27,039 misses give 50.281% hits across the full sequence.
These chat repeats all exceed 20; the cache remains explicit and its benefit
is prompt-dependent. Code repeats are **20.517 / 20.508 / 20.735 tokens/s**,
p95 **52.327 / 52.446 / 51.432 ms**, minimum final memory **3.056 / 3.055 /
3.064 GB**. All 1148 triples match; the full-sequence row-cache hit rate is
38.126%. Row-cache binary SHA256 is
`934edb2c0f177f2c24eaeabf663225a037eb099cb48dee7e84739fdc70c61601`. Thus this single approximate configuration exceeds 20 in all nine
short-prompt/grown-history repeats, although the cache-disabled capital/code
configurations have higher peaks. This remains regression evidence rather
than independent checkpoint validation.

The ordinary profile spends 2.653 ms in FFN broadcast and 1.335 ms in residual
handoff. The verifier currently repeats those messages for each input. The next
transport experiment, `--verify-comm-batch`, packs the 1..6 independent BF16 rows
and their exact FP32 tails into one message, and combines TP gathers while
unpacking into the original token/rank order. Keep the existing acknowledged
expert sum. Bound each wire workspace below 256 KiB, retain signed-zero rules,
and check every owner, all batch sizes, strides/tails, delayed receivers and
full-model state/logits before timing. No arithmetic reduction is changed.
The A64FX transport tests now pass TP1/TP2/TP4, every owner, all six batch
sizes, zero counts, mixed tails, signed zero, strides/canaries and delayed
receivers. Evidence is in `final-verifier-runs-v2/output.51575979/0` (launches
141–145 also contain the FP4 and complete-expert checks).

Next experiments are driven by these remaining costs: common-expert batching
and dense weight reuse first; then batch input quantization/dispatch, MTP
projection batching, and tighter transport integration if the verifier profile
supports them. `--verify-timing` reports rank-local batch phase durations as a
diagnostic; waits and rank maxima must not be added as independent compute.
An ordinary 30 tokens/s step needs another 14.05 ms removed from this profile;
eliminating all expert/shared cost alone is insufficient. Ordinary 40 tokens/s
needs 22.39 ms removed. Neither is implied by the weight-only bandwidth bound.

### Reproduce the continuation

The new paths remain explicit runner arguments. Use TP4 staging and the
ordinary arguments listed above, then add `--cache-sve --expert-input-cache`.
The row-cache experiment adds `--engram-row-cache-mib 8`. The final verifier
uses all three additions plus:

```sh
--mtp-stage-root /local/u14346/ds41f-51575979/mtp \
--speculate 5 --verify-batch --verify-expert-batch --verify-comm-batch
```

MTP defaults independently to `--mtp-quant 1 --mtp-expert-sdot 1
--mtp-hc-matvec 1`, including with an FP8 backbone. To inspect the original
FP8 draft path, explicitly set all three MTP arguments to zero. Draft
approximation affects acceptance; the selected backbone still verifies every
emitted token. `--verify-check` runs both verifiers and compares state, taps
and predictions; it is a correctness diagnostic, not a speed configuration.
`--spec-force-reject 0..4` tests rejected prefixes. `--state-hash` supports
EOS/output-limit comparisons across all twelve ranks.

New target builds and representative remote checks:

```sh
TMPDIR="$PWD/tmp/ds41f" make -C a64fx/ds41f ds41f_run test_cache_sve \
  test_prefetch test_journal test_tp_weights test_mtp_weights test_int8_batch \
  test_input_cache test_fp4_batch test_expert_batch test_comm_batch \
  A64FX_CC=fccpx A64FX_MPICC=mpifccpx \
  A64FX_CFLAGS='-Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -Wall -Wextra -Wpedantic'
# Snapshot executables first; run one MPI program at a time inside allocation.
mpiexec -np 1 ./test_cache_sve
mpiexec -np 1 ./test_prefetch /local/u14346/ds41f-51575979/prefetch-fixture
mpiexec -np 1 ./test_journal
mpiexec -np 1 ./test_fp4_batch bench
mpiexec -np 1 ./test_expert_batch
mpiexec -np 12 ./test_comm_batch 1
mpiexec -np 12 ./test_comm_batch 2
mpiexec -np 12 ./test_comm_batch 4
```

On the frontend, measure ordinary and speculative runs separately:

```sh
python3 a64fx/ds41f/decode_report.py RESULTS --baseline MATCHED_CONTROL \
  --start 1000 --stop 1105 --json RESULTS/summary.json
python3 a64fx/ds41f/spec_report.py SPEC_RESULTS \
  --start 1000 --stop 1105 --json SPEC_RESULTS/spec-summary.json
# Separate profiling run with --verify-timing; logs are written after decoding.
python3 a64fx/ds41f/verify_report.py PROFILE_RESULTS \
  --start 1000 --stop 1105 --json PROFILE_RESULTS/verify-summary.json
```

`decode_report.py` requires every requested position after prompt processing
and all twelve completed rank logs; it rejects batched fixed-input replay. Its optional baseline comparison checks the entire token trace.
`spec_report.py` uses complete cycles wholly inside the requested range and
actual emitted tokens; it reports both amortized token latency and burst cycle
p95. Forced-rejection diagnostics cannot be reported as performance. Neither
report substitutes for the nine-position full-logit quality check.

### Ordinary decode after 1K prompts

The additional suite `long-final-runs-v1` completed all eight cases after the
final verifier runs, with exit status zero.
Its checkpoint-encoded engineering chat and C-review prompts contain **1021
and 1024 tokens**, respectively. Exact messages, encoded text, IDs and
source/tokenizer/encoder hashes are under `long-prompt-inputs-v2`. Each case
generates 129 outputs and measures the 128 forward steps beginning at the
prompt length, excluding prompt processing. Row-cache off/on was compared once
per prompt; both comparisons fell within the 1% tie, so off was retained and
repeated twice. Chat measures positions **1021..1148**, code **1024..1151**.

| Prompt | Row cache | Repeat | Mean ms/token | tokens/s | p95 ms/token | Minimum final MemAvailable, bytes |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Chat | Off | 1 | 50.632 | 19.751 | 55.750 | 3,054,305,280 |
| Chat | On, 8 MiB budget | 1 | 50.619 | 19.755 | 54.738 | 3,024,420,864 |
| Chat | Off | 2 | 50.796 | 19.686 | 56.046 | 2,968,780,800 |
| Chat | Off | 3 | 50.812 | 19.680 | 56.579 | 3,049,193,472 |
| Code | Off | 1 | 50.399 | 19.842 | 53.787 | 3,031,433,216 |
| Code | On, 8 MiB budget | 1 | 50.358 | 19.858 | 54.200 | 3,038,904,320 |
| Code | Off | 2 | 49.929 | 20.029 | 53.311 | 3,045,457,920 |
| Code | Off | 3 | 50.531 | 19.790 | 55.821 | 2,992,766,976 |

All twelve ranks finish every run. Each cache-enabled run and both later
cache-disabled repeats preserve the complete matched baseline trace: **1149
triples for chat**, **1152 for code**. The two first cache-disabled runs define
these controls; this suite does not compare full logits with an independent
reference. Every run uses the final `c8a4075a...` binary and the approximate
backbone settings below. No profiler or logit dump is enabled.

The ordinary **20+ milestone does not hold across these actual 1K prompts**.
Row caching helps some earlier short-prompt/growing-history workloads, but has
no material benefit here. Speculative 20+ was measured on the short capital
prompt grown to 1K, and still needs this long-prompt test. This distinction is
necessary because generated repetition can change cache hits and draft
acceptance. Thirty and forty tokens/s have not been demonstrated in either
history setup.

Evidence is in `tmp/ds41f/job51575979/long-final-{off,on}-{chat,code}-repeat<N>-v1/summary.json`
and `long-final-runs-v1/selection.json`. Chat/code token-ID SHA256 values are
`c99b7d162e68154b67cdd3615b2829505d192f697ceffc01156d68c9bdad3923`
and `61cc865a7d28571115cb9e62c446cbc74345bf2908a3646952a971ffef9cfa65`.
The driver admitted the full suite only with twenty-four minutes remaining
before its reserved finish deadline and launched one MPI program at a time.

The complete ordinary chat invocation inside the allocation is below. Use an
empty result directory containing the verified runner snapshot, and the actual
repository root for `DS41F_REPO`. Stage paths must be regenerated for a fresh
allocation because `/local` does not survive it.

```sh
DS41F_REPO=/vol0006/mdt0/data/hp250467/work/gemm/ds4f
OMP_NUM_THREADS=48 OMP_PROC_BIND=close OMP_PLACES=cores OMP_WAIT_POLICY=active \
mpiexec -np 12 ./ds41f_run \
  --stage-root /local/u14346/ds41f-51575979/tp4 --dense-tp 4 \
  --max-context 1048576 --engram-prefetch --engram-scale-cache \
  --hc-mix-sve --shared-overlap --weights-local-pages --mpi-broadcast \
  --compact-comm --sparse-tile 4 --index-head-tiles --expert-fused 1 \
  --linear-input-cache --quant-parallel --persistent-team --rope-cache \
  --cache-sve --expert-input-cache --fp8-int8-block 32 --expert-sdot \
  --hc-matvec 1 \
  --prompt-ids "$DS41F_REPO/tmp/ds41f/job51575979/long-prompt-inputs-v2/chat.ids" \
  --generate 129 --ignore-eos
python3 "$DS41F_REPO/a64fx/ds41f/decode_report.py" . \
  --start 1021 --stop 1149 --json summary.json
```

For code use `code.ids` and report positions `--start 1024 --stop 1152`.
Adding `--engram-row-cache-mib 8` reproduces the row-cache comparison.
These are approximate backbone settings; omit `--expert-sdot --hc-matvec 1`
for the selected INT8 control. Actual immutable arguments for every experiment
are in its `args.json`; none of these ordinary commands enables speculation.

### Future work after the continuation

1. Resolve the original FP8 independent-reference discrepancy before declaring
   checkpoint or GPU parity. Keep the fixed-input cosine >=0.999, relative RMS
   <=1% and argmax gates. Then isolate the additional INT8, expert SDOT and
   mHC approximations with bounded same-input layer captures.
2. Use the final buffered verifier profile to choose the next batch operator.
   Prioritize an exact batched mHC matrix kernel: schedule the independent
   `(token, row)` pairs together (up to 6 × 24 jobs), retaining every row
   reduction's original K order, normalization and BF16 boundaries. This can
   use all 48 workers without the rejected split-K approximation; measure the
   complete operator and fixed-history verifier before claiming a speedup.
   QA/KV preparation, index preparation, mHC and MTP projections still contain
   per-input dispatches. Parallelize the bounded window snapshot copies and
   overlap Engram preparation with them or early owner work; the first-layer
   broadcast exposes start-of-batch producer skew. Batch independent
   projections and quantization while
   preserving each causal cache view and original rounding boundaries. Judge
   changes by draft+verify+commit time per emitted token, including rejection.
   Retune draft prefixes after batch-layout changes, and compare MTP math
   choices by acceptance as well as draft cost; draft quantization can change
   proposed IDs. Batched W1/W3 fusion and sharing prepared INT8 expert inputs
   across common experts remain candidates if the FFN span warrants them.
3. Revisit ordinary attention and expert cost for 30/40. The latest ordinary
   profile is about 47.6 ms; 30 requires below 33.33 ms and 40 below 25 ms.
   Distributing the remaining owner projections or changing WO-B/shared-W2
   reduction order needs a new communication budget and independent numerical
   gates. TP2/TP4 row sharding and persistent workers are already implemented.
4. Defer distributed index scoring at 1K unless a later profile changes its
   value: scoring is only about 0.215 ms/token after the exact SVE conversion.
   New collectives can exceed that saving. Longer histories need their own
   measurements; a 1M allocation does not validate execution at 1M history.
5. Extend the completed ordinary 1021/1024-token prompt tests to speculation
   and a wider prompt set. Ordinary chat/code currently reach 19.68–20.03
   tokens/s and fail the three-repeat 20+ criterion; reducing roughly 0.8 ms
   from the slowest mean is necessary but leaves no run-variation margin.
   Profile these retained-input workloads before choosing another ordinary
   optimization. Earlier capital/chat/code timings grow short prompts through
   1100 generated tokens with `--ignore-eos`; repetition can improve Engram
   cache hits and draft acceptance. Keep the two setups distinct and require
   three repeats for each claimed milestone. Repeat retained configurations
   on a fresh allocation.
6. Add a bounded multi-case validation mode that reuses resident weights;
   loading currently costs about a minute per case. Keep separate request
   state, reset caches/history explicitly, and retain immutable run metadata.
   KV checkpoint/restore, actual-1M execution and serving remain separate tasks.

## 20/30/40 tokens/s implementation plan

Agreed 2026-09-12, recorded before implementation. Keep 12 A64FX nodes and
single-request decode around 1K history, all 40 layers, 64 attention heads,
top-6 routed experts, top-512 selected rows and the 128-token window. Maintain
two separately reported tracks: numerically validated and speed-first
approximate. Exact-output speculation is a third, separately measured path.
At plan approval, unprofiled INT8 runs reached 10.36–10.39 tokens/s but failed
the numerical gate; the FP8 regression control reached 6.92 tokens/s. Neither is official
GPU parity. Preserve the original math path as the comparison control.

### Milestone budgets

These mutually exclusive milliseconds/token are engineering targets, not
predictions or measured results. Apply the budgets independently to each
quality track; never promote a failing approximate run as validated.

| Component | 20+ target | 30+ target | Ordinary 40+ stretch |
| --- | ---: | ---: | ---: |
| Attention projections and preparation | 14 | 8 | 6 |
| Sparse attention and index | 9 | 5.5 | 4 |
| Routed and shared experts | 12 | 8 | 6 |
| mHC mixing | 4 | 2.5 | 2 |
| Synchronization | 4 | 3 | 2 |
| Other | 6 | 5 | 4 |
| **Total ms/token** | **49** | **32** | **24** |
| **Implied tokens/s** | **20.4** | **31.3** | **41.7** |

The 800 GB/s / 16 GB single-node weight-only estimate is 50 tokens/s. The
profile counts 13.88 GB of weight operands globally and about 9.81 GB on the
current serial dense-owner/EP critical path. Neither is a hardware-counter
HBM measurement. Other nodes wait during owner-only attention; multiplying
bandwidth by twelve is inappropriate. Eliminating the entire 21.58 ms INT8
kernel aggregate alone cannot take the present roughly 98 ms step to 20+.

### Stage 1: profile, attention kernels, projection reuse, mHC and transport

1. Split sparse attention into QK, softmax and PV, and separate packing,
   activation quantization, OpenMP launches and collective arrival skew.
   Reconstruct the critical path without summing nested spans or rank waits.
2. Add FP32 QK tiles of four heads by two keys while preserving the existing
   two-accumulator dot order. Benchmark PV tiles of 2/4/6 heads by 64 channels
   and retain the fastest complete operator passing its quality track.
   Decode selected rows once into bounded scratch; do not expand full history.
3. Adapt ideas from
   `~/work/clair/a64fx/a64fx/llm-guided-opt/attention_decode_a64fx.c`:
   reuse K across heads and V across head/channel tiles. Its 12-head, D=256
   geometry and single-valid-key shortcut are not DS41F semantics: preserve
   the DS41F sink in the maximum and denominator even with a single key.
4. Test SDOT QK using exact E2M1*2 integer values with original compressed-KV
   group-16 scales, and quantized queries/raw keys. Rescale each dot block to
   FP32 before combining scores; block-dependent scales preclude one global
   integer coefficient. Initially retain FP32 PV.
5. Compare libm, corrected FEXPA, integer polynomial-2 and integer affine
   softmax independently. Reference
   `~/work/clair/a64fx/a64fx/llm-guided-opt/int_exp2_sdot_a64fx.s`
   and `integer-exp2-sdot.md`. Their quoted timing is QLAIR simulation, not
   native evidence; affine/poly2 error bounds are about 2.98%/0.375%.
   Apply stable maximum subtraction including the sink, multiply by log2(e),
   extend the original [-16,0] exponent domain to [-31,0], underflow to zero,
   zero masked entries and use 64-bit Q31 denominator sums. A -16 clamp would
   add a substantial floor over 640 rows. Check overflow, tails, ABI register
   preservation, all-masked/sink-only cases and real model inputs.
6. Reuse activation quantization across compatible projections, shared W1/W3
   and routed experts without moving FP8/BF16 rounding boundaries. Keep fresh
   anonymous INT8 pages. Measure cold complete operators including conversion
   and packing. Fuse MXFP4 W1/W3 with original group-32 scales; the CLAIR
   `fp4_i8_sdot_a64fx.s`/`sdot_quant.h` integer-grid format, nibble order and
   ties-away rounding differ from this checkpoint and need adaptation.
7. Retain the ordered mHC control; gate register-split FP32/FP64 matrix sums
   and Sinkhorn variants on full-model fixed-history logits. Local norm tests
   alone did not detect the earlier split-K model regression.
8. Replace world residual broadcast with owner-to-next-owner handoff (last
   layer to head rank 11). Pack already-BF16 residual/FFN vectors losslessly;
   retain FP32 mixing coefficients and route weights, and signed-zero
   normalization. Publish source-cache bytes directly instead of float codes.
   Keep robust uTofu ACK reductions initially and all MPI on the main thread
   under MPI_THREAD_FUNNELED. Do not disable acknowledgements to hide waits.
9. If this does not meet the 49 ms budget, proceed to TP2 then TP4 without
   changing the fixed backbone or loosening the validated-track gate.

### Stage 2: dense tensor parallelism for 30+, ordinary 40+ stretch

Add TP=1/2/4, with layer owner `layer % 12` and contiguous group base
`(owner / TP) * TP`. Fixed rank offsets select 32/16 heads and 4/2 whole
WO-A groups for TP2/TP4. The original dense and shared payload is balanced
so these layouts retain the same per-rank total original weight bytes as
TP1; verify actual converted buffers and peak loading memory separately.

- Initially keep owner QA, KV compression and index. Publish group inputs;
  shard QB, RoPE, sparse attention and whole WO-A groups. Allgather the
  already-BF16 8192-element projected vector, row-shard WO-B with its full
  K dot order, then gather outputs to the owner for mHC and expert routing.
  Keep replicated packed source caches in the first TP implementation.
- Distribute index work by whole eight-row candidate blocks, then merge with
  original score/ID tie rules, selected-ID ordering and forced newest block.
- Row-shard shared W1/W3, gather its 2304-element hidden vector, and row-shard
  W2. Distribute the vocabulary head's 4040 blocks of 32 rows over all 12
  ranks; reduce max with smallest-global-ID ties. Gather full logits only
  for diagnostics. Avoid duplicating the shared backbone embedding/head.
- Introduce explicit persistent OpenMP team kernels with main-thread MPI;
  preserve the existing control and measure launch savings. For the 24 ms
  stretch, test column-sharded WO-B/shared W2 and combined communication only
  behind independent numerical gates because their reductions change order.
- Expose explicit runner arguments for kernels/math/TP and an INT8 tensor
  allowlist. Version stage metadata with global shapes, local ranges, group
  mapping and source hashes; retain TP1 compatibility. Stage new dense
  layouts separately while reusing expert and Engram shards.
- Convert after reading `/local` when preprocessing is negligible, as with
  the current 0.27–0.50 s INT8 conversion versus 53–57 s loading. If new
  preparation is material, write versioned INT8 artifacts beside the shared
  original weights, then stage them. Account for every workspace, cache and
  MTP buffer with a 2 GiB minimum MemAvailable; no full-model duplicates.

### Stage 3: exact-output speculative decode for 40+

The checkpoint contains three DSpark/MTP stages, fixed draft block size five,
noise ID 128799, main hidden taps at layers 37/38/39 and Markov rank 256.
MTP tensors total 7,932,874,632 bytes; draft MoE uses 128 experts/top-3 while
the backbone remains 384/top-6. Implement checkpoint behavior from local
`inference/model.py`, including tap positions/dtypes, main projection,
stage-specific attention masks, mHC, Markov bias and confidence output.

Shard draft experts with EP and dense weights with TP, reuse backbone
embedding/head, and admit all state before loading. Always calculate the
checkpoint's full five-position draft block; benchmark verified prefixes of
2/4/5. Add verifier microbatches of 1–6 with weight reuse and original per-token
dot, reduction and BF16 boundaries. Existing batched GEMM changes reduction
order and is not automatically an exact verifier; use bounded packed tiles.

Given known seed x0, draft x1..xd and verify inputs x0..xd. Accept the matching
greedy prefix of length a, emit a accepted tokens plus the verifier bonus,
and commit a+1 input states; the bonus is the next uncached seed. Confidence
never bypasses verification. Preserve per-token causal index/cache views.
Journal or shadow overwritten window slots, compressed append counts,
candidate and pool state, Engram history and prefetch generations; roll back
the rejected suffix. MTP main-KV receives every committed main hidden position;
draft noise-KV stays temporary. Handle wrap, EOS and output limits explicitly.

Measure actual emitted tokens divided by draft+verify+commit time, separately
from ordinary decode. Three emitted tokens need a cycle below 75 ms for 40+.
Select the fastest measured prefix, preferring fewer drafts on a tie. Default
speculation off until emitted tokens and committed state match the selected
validated sequential verifier, including forced rejection at every position.

### Validation and execution order

- Kernel tests use references and real inputs, masks/sinks/tails/canaries,
  conversion costs and overflow bounds. Test TP1/2/4 across every owner,
  delayed-rank collectives, BF16 transport, ties, window/compression and top-k
  boundaries. Finish the corrected independent nine-position reference.
- Compare identical input histories at early positions and positions
  1000..1008, plus chat/code prompts. Report cosine, relative RMS, argmax and
  every approximate-track failure; token agreement alone is insufficient.
- Run three uninstrumented 1100-output repeats for each retained milestone,
  measuring positions 1000..1104. Every repeat must exceed the claimed target.
  Report p95, minimum memory, prompt dependence, binary/staging hashes and a
  separate profile. Allocating 1M cache is not running at 1M history.
- Execute and commit coherent stages in this order: this document; detailed
  profile/kernels/transport; TP2; TP4/shared/head; ordinary 40+ experiments;
  exact speculative 40+. Update this document with results and remaining work.
  Use one MPI program per allocation, detached immutable binary/script
  snapshots, remote SHA verification, `/local` or repository `tmp/`, and check
  allocation lifetime before each long experiment. Do not edit active scripts.

## Implementation continuation: attention tiles and dense TP, 2026-09-12

The plan above was committed as `470c45a1` before kernel changes. Experimental
options remain explicit; none of these measurements establishes 20 tokens/s.
The active allocation is still 51569201 (12 nodes, 2 GHz, ends 00:00:29 JST
September 13). Original TP1 staging remains intact; TP2/TP4 use separate
`/local/u14346/ds41f-51569201/tp{2,4}/rank<R>` directories and reuse local
expert/Engram files. `stage_tp.py` records versioned ranges, checkpoint header
hashes and SHA256 sidecars for copied shards. Runtime validates TP/rank and
weight row-range metadata before conversion.

Implemented stages and initial evidence:

- The finer original attention profile reports maximum worker work durations
  of 5.185 ms QK, 1.012 ms softmax and 5.066 ms PV per token around 1K.
  These maxima are nested work measurements, not three additive wall spans.
  Tiled profiles measure elapsed phases including their barriers.
- `--sparse-tile 1/2/4/6` adds split-phase attention; 2/4/6 use four-head,
  two-key QK and corresponding multi-head PV. Zero retains the original path.
  Tile four reduced the complete sparse span 12.283 to 8.818 ms/token, and
  full profiled latency 98.596 to 94.418 ms (10.591 tokens/s). Fifty geometry,
  mask, sink, tail and canary cases pass bitwise against the existing control
  on A64FX and native scalar builds. All 1105 token triples and nine saved
  logits also match the INT8 control exactly (`sparse-tile4-full-v1`).
- `--compact-comm` uses byte source publication, mixed BF16/FP32 FFN packets,
  and owner-to-next-owner residual handoff (layer 39 to head rank 11).
  Packing rejects non-BF16 inputs; route weights and mHC coefficients stay
  FP32. MPI stays on the main thread and robust uTofu ACK sums are retained.
  All-owner delayed-rank, self-handoff, nonparticipant, signed-zero and canary
  tests pass. The long run retains all 1105 triples/nine logits exactly and
  improves total inference 101.311 to 99.851 seconds (`compact-full-v1`).
- RoPE rounds only its modified 64-coordinate suffix, since every caller has
  already rounded the untouched coordinates to BF16. The combined exact
  changes preserve all nine FP8 control logits and all 1105 INT8 token triples
  plus nine logits (`stage1-fp8-exact9-v1`, `stage1-int8-exact-full-v1`).
- `--hc-matvec 1/2` tests register-accumulated FP32/FP64 sums; zero retains the
  ordered FP32 control. FP64 accumulates rounded FP32 products. Both pass
  bounded local tests but **fail full-model numerical gates**: early nine
  positions versus INT8 control have minimum cosine 0.993941/0.990968 and
  maximum relative RMS 11.935%/15.405%. FP32 at fixed 1K history reaches
  cosine 0.943039 and RMS 33.972%; its mHC span falls to 3.40 ms/token. Keep
  both clearly marked approximate; local norm agreement is insufficient.
- `--dense-tp 2/4` distributes QB, whole WO-A groups, row-sharded WO-B and
  shared W1/W3/W2 within contiguous groups. BF16 allgathers retain complete
  input K ranges for WO-B/W2, and gathers return outputs to the owner. QA,
  compression and index remain owner operations. Vocabulary rows are split
  in blocks of 32 across all 12 ranks; MAXLOC preserves smallest-ID ties,
  and only diagnostic runs gather full logits. TP requires shared overlap.
  Both TP sizes pass every-owner transport tests and reproduce all nine FP8
  and INT8 control logits bit-for-bit. TP2's first long INT8 run also matches
  all 1105 token triples and nine logits, reaching 89.923 ms/token around 1K
  (11.121 tokens/s), 97.700 seconds total, minimum final memory 3.861 GB.
  The head falls to 0.260 ms/token, while attention still takes 40.032 ms.
  Further TP4 performance and 1K logit checks are running.
- `--sparse-math 1/2/3` independently selects corrected FEXPA, Q31 integer
  polynomial-2 or affine softmax with a nonzero sparse tile. Zero retains
  libm. Integer kernels extend the CLAIR domain to -31 and underflow smaller
  scores to zero; the denominator is uint64. Exhaustive Q16 domain and
  extreme-value tests match the scalar integer oracle, respect error bounds,
  and pass monotonicity, masks/sink, single-key, tail and canary checks on
  A64FX and native builds. Full-model numerical/performance gates are pending.

Evidence is under `tmp/ds41f/job51569201/`, including `sparse-profile-v1`,
`sparse-tiles-v1`, `compact-check-v1`, `stage1-runs-v1`, `exp-check-v1`,
`tp-staging-v1` and `tp-short-runs-v1`. Every remote runner uses an immutable
snapshot and verified binary hash. Minimum final memory in the completed long
runs remains above 3.98 GB. The corrected independent nine-position NumPy
reference is running in `reference9-v1`; do not claim it has passed yet.

Build/check commands for this stage:

```sh
TMPDIR="$PWD/tmp/ds41f" make -C a64fx/ds41f a64fx \
  A64FX_CC=fccpx A64FX_MPICC=mpifccpx \
  A64FX_CFLAGS='-Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -Wall -Wextra -Wpedantic'
python3 a64fx/ds41f/test_stage_tp.py
# Inside the allocation, with one program at a time:
OMP_NUM_THREADS=48 OMP_PROC_BIND=close OMP_PLACES=cores ./test_sparse_tiles
./test_exp2
mpiexec -np 12 ./test_broadcast
mpiexec -np 12 ./test_tp_comm 2
mpiexec -np 12 ./test_tp_comm 4
```

Remaining planned work includes activation-quantization reuse, fused expert
kernels, SDOT sparse QK with complete packing costs, distributed index work,
persistent OpenMP teams, any justified changes to projection reduction order,
and the separately validated DSpark speculative path. MTP/INT8 batched GEMM
and speculative rollback are not implemented by this continuation yet.

## Implementation continuation: index, experts and input reuse, 2026-09-12

The following results supersede the pending TP4 results above. All rates in
this table are instrumented samples at positions 1000..1104 on job 51569201,
2 GHz, twelve nodes, TP4, the fixed six-token capital prompt and 1100 generated
outputs. Independent uninstrumented repeats are running; these are not 20+
claims. Every exact-control row below preserves the corresponding 1105 token
triples and nine saved logits bit-for-bit.

| Configuration | ms/token | tokens/s | Control / limitation |
| --- | ---: | ---: | --- |
| FP8, TP4 initial | 105.794 | 9.452 | Nine fixed-history 1K logits match FP8 |
| INT8, TP4 initial | 87.143 | 11.475 | INT8 control |
| INT8, active OpenMP wait policy | 81.166 | 12.320 | INT8 control |
| INT8, vector index heads + fused expert pair | 76.466 | 13.078 | INT8 control |
| Above + fresh attention scratch pages | 76.958 | 12.994 | No clear gain |
| Above + input quantization cache | 75.632 | 13.222 | INT8 control |
| Cache path + approximate mHC mode 1 | 72.244 | 13.842 | Fails mHC numerical gate |

The INT8 control itself still fails the FP8 numerical gate. Bitwise regression
checks of the new transformations do not make INT8 numerically validated.
TP4 fixed-history replay also preserves nine INT8 logits at positions
1000..1008 exactly. The retained FP8 path keeps original checkpoint weights
and rounding. The completed corrected independent nine-position reference
will be recorded separately when available.

- `--index-head-tiles` transposes the 32x128 query once and evaluates heads in
  SVE lanes. Each lane retains the 128-term ordered FP32 sum, BF16 score and
  weighted-score boundaries; ordered head accumulation retains selection
  ties. Twenty-four native/A64FX mask, tail and canary cases pass bitwise.
  At 1105 candidate rows the standalone score kernel falls from 460.85 to
  42.82 microseconds. Full-model index scoring falls to 0.281 ms/token.
  Distributing this now-small score phase would likely add more communication
  than it removes at 1K; reconsider only for longer measured histories.
- Top-k selection skips heap construction when every valid row fits and uses
  bounded membership flags for ascending selected IDs below 4096 rows.
  Larger histories retain the sorted fallback. Existing cache/selection
  boundary tests pass.
- `--expert-fused 1/2` shares input loads for routed W1/W3 while preserving
  their FP32 accumulation order and original MXFP4 group scales. Eighteen
  geometries with both tiles pass bitwise. Cold 2304x5120 paired projections
  take 130.84 us originally, 106.18 us for tile one and 127.59 us for tile two;
  tile one is retained. Packing and SDOT experiments remain separate.
- `--linear-input-cache` retains up to four prepared activation vectors keyed
  by complete input contents, length, INT8 block size and FP8 activation
  rounding boundary. Reused addresses with changed values miss correctly;
  grouped WO-A keeps its original absence of FP8 activation rounding. The
  cache stays below 1 MB/rank. Raw/FP8/grouped inputs, changed contents,
  repeated hits, changing block sizes and nonfinite rejection pass native
  and A64FX tests. Prepared and ordinary GEMV match in all 162 INT8 cases.
- `--attention-local-pages` places only bounded selected-row scratch on fresh
  anonymous pages. It passes the long exact regression but showed no clear
  speed gain and is omitted from the retained uninstrumented repeats.
- Corrected FEXPA, integer polynomial-2 and affine softmax all fail the early
  nine-position full-model gate versus the INT8 control: minimum cosine
  0.994960/0.994632/0.992309 and maximum relative RMS
  10.066%/10.451%/12.862%. Keep `--sparse-math 0` for the retained control.
- The separate `ds41f_int8_matmul[_prepared]` kernel accepts 1..6 tokens and
  reuses each four-row weight tile while preserving sequential GEMV lane
  and reduction order. All 378 batch/shape/group/stride/tail/canary cases pass
  bitwise on native and A64FX builds. Cold complete INT8 operators, including
  input quantization, improve 1.2–1.5x for batches two through four on TP4
  WO-B/QB/WO-A geometries; five/six gain little. FP8 activation rounding is a
  caller boundary outside this microbenchmark. This is verifier groundwork,
  not an integrated batched verifier or speculative token-rate result.

The cached full profile still spends 29.600 ms in attention, 20.887 ms in
routed/shared experts and 9.914 ms in mHC mixing. Attention's nested sparse
span is 4.330 ms; index preparation plus scoring/selection is 3.115 ms.
The slowest routed rank spends 7.664 ms in W1/W3, 4.648 ms in W2 and 2.214 ms
in activation quantization. These measurements motivate packed expert SDOT,
quantization scheduling/reuse and further projection work before speculation.
Minimum final MemAvailable for the cached run is 3,884,056,576 bytes.

Evidence: `tmp/ds41f/job51569201/{index-pair-check-v1,cache-check-v1,batch-check-v1}`
and the `tp4-*` run directories, including `comparison.json`, `quality.json`
and `profile-1k.json`. Builds use the warning flags above; remote checks run
with `OMP_NUM_THREADS=48 OMP_PROC_BIND=close OMP_PLACES=cores OMP_WAIT_POLICY=active`.
The immutable retained driver is `retained-runs-v1/run-v1.sh`. The next
12-node allocation is **51575979**, running 22:50:08–04:50:08 JST, bridge
42395/32395/21266; safe bounded original/TP4 staging is running in
`tmp/ds41f/job51575979/staging-v1`. Do not overlap MPI programs per allocation.

Still outstanding: SDOT sparse QK, packed expert experiments, persistent
OpenMP teams, broader chat/code quality checks and the full DSpark/MTP
staging, draft execution, causal batched verifier and state rollback. No MTP
inference or speculative speed claim is implemented by the batch kernel alone.

## Implementation status, 2026-09-12 (job 51562789)

The 12-node runner now executes all 40 layers with real resident weights and
Engram offload. The six-token prompt `The capital of France is` (including
BOS) generated ` Paris. In the` across four outputs. All ranks completed.
The completed independent NumPy reference agrees on all nine next-token choices
in `infer-v2`. Logit cosine similarity ranges from 0.994041139 to 0.999988026
over those positions, and five fail the 0.999 comparison gate. Thus full numerical
validation is **not passed**; this is neither bit-exact nor official GPU parity.
The continuation below corrects an mHC arithmetic boundary and passes a fresh
three-position full-model reference check. A complete nine-position rerun with
that corrected reference remains outstanding; no reference process is left running.

`infer-v1` and `infer-v2` logs under `tmp/ds41f/job51562789/` record the same
nine input positions with tracing and logits enabled: 11.339 s versus 3.129 s.
The improvement comes from bitwise FP8 conversion and an SVE BF16-weight /
FP32-input path. Resident loading took 52–69 s/rank in v1. Subsequent allocator
demand-paging runs use `XOS_MMM_L_PAGING_POLICY=demand:demand:demand` so parallel
first-touch is effective; default prepopulation defeated NUMA placement.

A 160-output run (`infer-v3`) completed 165 input positions in 58.245 s,
including window wraparound. It committed the configured 1M-token cache pages
up front; minimum final MemAvailable exceeded 3.8 GB. This proves cache
allocation plus bounded execution, **not execution at 1M-token history**.
The longer `infer-v4` and `infer-v5` runs each completed 1,105 input positions
and generated 1,100 outputs with EOS stopping explicitly disabled. All 12 ranks
finished, and all 1,105 input/next-token triples were identical between runs.
Time fell from 530.402 s to 357.689 s after exact packed-row decoding and
parallel selected-row unpacking: 1.48x faster, approximately 3.08 generated
tokens/s including the six-token prefill but **excluding resident loading**.
V5 loading took 51.4–66.8 s/rank; minimum final MemAvailable was 3.809 GB.
The completed `infer-v6` repeat took 319.984 s (3.44 generated tokens/s), and
all 1,105 input/next-token triples match v5 exactly. All 12 ranks finished.
These runs cross window wraparound and the 512-entry index top-k boundary for
both compression ratios. They still do not establish actual 1M-history speed.

`infer-math` additionally completed 42 positions / 32 outputs in 11.910 s and
answered `5` to a plain-completion arithmetic prompt (then continued its
question/answer pattern). This is not a chat-template quality test.
`infer-chat` completed a checkpoint-formatted chat prompt with 45 generated
tokens, stopping at EOS, in 16.453 s. It explained the blue sky through
wavelength-dependent scattering; all 12 ranks finished. CPU performance
targets are not yet met by the complete quantized model kernels.

The exact residency plan is generated by `a64fx/ds41f/plan_residency.py` and
saved, with corrected index-key accounting, in
`tmp/ds41f/job51562789/residency-v2.json`.
Experts use `expert_id % 12`, dense layers use `layer % 12`, embeddings rank
zero, head/final norm rank eleven. Compressed HBM weights range from
24,574,984,584 to 26,080,018,272 bytes/node. Engram consumes approximately
16,896,502,920 bytes/node on disk. Dense ownership distribution and rank-local
resident indexes completed on all 12 nodes (`phase2/dense.rank*.log`).

Corrected packed batch-one, 1M-token KV accounting is 935,936,000 bytes
**globally**, or that much per rank if conservatively replicated:

- four compressed KV sources: 3 at ratio two, 1 at ratio one;
  512 values/row, FP4 plus one E4M3 scale per 16 values;
- four index-key owners (the KV sources): 3 at ratio two, 1 at ratio one;
  128 values/row, FP4 plus one E8M0 scale per 32 values. The other four index
  query sources reuse these keys; they do not allocate additional key caches;
- all 40 sliding windows: 128 rows of FP8 512-wide KV plus group-32 scales.

The runner stores compressed KV/index rows packed, but retains FP32 sliding
windows as an execution buffer. Its corresponding allocation is 943,718,400
bytes plus candidate masks and small pooling/selection state. The index top-k
of 512 is not a key dimension. Other FP32/BF16 buffers,
candidate lists, pool state, allocator/communication buffers and temporary
dequantization must be accounted separately. Use actual node MemAvailable,
not nominal 32 GiB alone, when admitting resident weights and cache.

Validated on the initial A64FX compute node:

- SVE512 BF16/RMSNorm dispatch with tails and canaries;
- exact FP8 scale-block and finite-code conversion regressions;
- real layer-zero expert projection: max absolute error 1.13249e-6 against
  scalar decoding/FP64 accumulation;
- real expert w1/w3/SwiGLU/w2 chain: max absolute error zero against the CPU
  reference for the tested input, including dynamic FP8 activations and BF16
  boundaries (not a comparison with the official GPU model);
- routing, mHC, RoPE and sparse-attention unit cases;
- Engram hash compressed history/padding/bucket offsets; tokenizer metadata
  has 129280 entries, 99092 compressed IDs, and exact checkpoint bucket totals.

Representative 48-thread dummy GEMV measurements during background staging:

| Format and matrix | Time | Useful GFLOP/s | Effective weight GB/s |
| --- | ---: | ---: | ---: |
| MXFP4 2304x5120 | 59.806 us | 394.494 | 104.787 |
| MXFP4 32768x1280 | 200.950 us | 417.448 | 110.884 |
| FP8 2304x5120 | 125.698 us | 187.696 | 93.940 |
| FP8 32768x1280 | 475.835 us | 176.292 | 88.232 |

Logs: `tmp/ds41f/job51562789/quant-bench-inline.log`. Same tensors are reused
50 times; effective bandwidth is bytes/time, not an HBM counter measurement.
Inlining E8M0 scale conversion removed a major bottleneck. A direct bitwise
FP8 SVE decoder was slower (193.711 us at 2304x5120) and was removed in favor
of gathers. The 80% compute/90% bandwidth targets are **not achieved**. GEMV
has low arithmetic intensity; a separate batched GEMM benchmark is needed to
evaluate a compute-utilization target meaningfully.

Post-staging rank-local suites, staged Engram smoke tests, the six-expert
uTofu combine versus an MPI oracle, and dense distribution passed on all 12
nodes. Evidence is in `tmp/ds41f/job51562789/phase2/`. The original waiting
script failed after being edited while active; the stable
`run_bringup_phase2.sh` completed the sequence. Do not edit active scripts.

Remaining gates are broader numerical/token validation, efficient batched
prefill integration, and the kernel utilization targets. KV persistence and
streaming are not implemented. The independent NumPy reference is not an
official GPU execution oracle.

Hardware controls on the initial node (48 threads, 2 GHz, demand-paged NUMA
first-touch): up to 6.122 TFLOP/s with 24 independent SVE FMA chains (99.6%
nominal), and 900.080 GB/s streaming reads from 2 GiB (87.9% of nominal
1024 GB/s). Larger streams amortize launch/reduction overhead; 512 MiB measured
855 GB/s with the same eight-accumulator loop. Manual prefetching did not help.
These are controls, **not model throughput**. The quantized batched GEMM
prototype now reaches 4.10–4.22 TFLOP/s including quantization/packing at
batch 768 (66.7–68.7% nominal). Its packed microkernel phase alone reaches
approximately 5.3 TFLOP/s for M=2304/K=5120, but that excludes required packing
and is not the complete operator. Sparse attention at 64 heads, dimension 512, 640 selected rows
improved from 3.250 ms to 0.628 ms in warmed best-of-repeat measurements,
with max absolute error 1.49e-6 versus the FP64-score reference.

Additional validation/optimization evidence:

- `batch-shared-quant.log`: SVE group-32 activation quantization matches the
  scalar path exactly over 12,288 groups spanning BF16 codes, random FP32
  inputs, scaled FP8 midpoints, signed zeros and rejected nonfinite inputs.
  The same helper is now used by decode; the completed `infer-v6` long repeat
  agrees with v5 on all 1,105 input/next-token triples.
- `quant-grouped.log`: grouped FP8 output projection matches eight separate
  calls exactly, improving 407.408 us to 296.219 us (1.38x) in the dummy case.
  The alternative direct-bit FP8 GEMV decoder remained slower than lookup
  gathers and is not enabled.
- `attention-source-tests/`: six positions for each source layer 0/2/8/14/20
  completed. Independent NumPy comparisons for compressed sources 2 and 20
  give cosine 0.999999962 and 0.999999754, maximum absolute BF16 output
  differences 0.015625 and 0.03125, respectively.
- `attention-candidate-v2/`: real query projections with **synthetic** 32K
  history retain 2,048/4,096 candidate blocks at layer 20, including the newest;
  layer 24 selects exactly the 64 rows permitted by eight injected blocks.
  Later query layers now skip decoding/scoring masked-out rows. This is not
  a full-model 32K-history run.
- `cache-final.log`: top-2048 selection over 32,771 scores agrees with a full
  sort, including ties, NaNs, masked negatives and positive infinity.

Publish versioned test scripts as well as binaries: an in-place frontend edit
was not visible in the compute node's cached shared script. The first candidate
test launch therefore ran the old wrapper; only `attention-candidate-v2/`
contains the candidate tests. Compare frontend/compute SHA256 before execution.

### Continuation: mHC arithmetic and bounded operator replay

Job 51562789 was resumed through the existing bridge at local port 42393,
login reverse port 32393, on service node `a25-4009c`. Its scheduled end is
2026-09-12 17:57:45 JST. This allocation retains the previously staged rank
directories; do not reuse those paths after the allocation ends.

The mHC post operation now sums residual terms before adding `post * x`,
matching the checkpoint's `inference/model.py` expression. FP32 products are
rounded separately, rather than contracted into that sum. The former ordering
lost a small residual under cancellation: the new regression expected 1 and
returned 0 on the old implementation. Cancellation, product rounding and
in-place updates pass on A64FX; the old/new cancellation comparison also
reproduced on the frontend. The NumPy reference now uses the same explicit
multiply/reduce/add boundaries instead of a BLAS matrix multiply.

`--dump-prefix` / `--dump-count` provide bounded owner-written intermediate
records (at most 64 positions). `replay_intermediates.py` reads original
safetensors and recomputes operators with the recorded inputs. It checks exact
routing IDs plus cosine and relative RMS gates, rebuilding attention history
from the source-layer inputs. This isolates individual operators from upstream
rounding drift; it does not replace an autoregressive comparison. Nine positions
across all 40 layers produce 120,318,480 dump bytes, and their shared-storage
writes must be excluded from speed measurements.

Evidence under `tmp/ds41f/job51562789/`:

- `resume-numerics-v1/replay-early.log`: 408/408 checks passed over three
  positions and layers 0–8, with minimum cosine 0.999997493. This initial
  replay used the earlier NumPy mHC expression and the cosine-only gate.
- `resume-mhc-v2/ops-before-x86.log`, `ops-after-x86.log`, and
  `resume-packets-v4/ops.log`: the cancellation regression fails before the
  fix and the updated operator suite passes on frontend and A64FX.
- `resume-packets-v4/comparison.json`: merging each layer's input/route and
  residual/pre-mix broadcasts removes 80 collectives per token. All nine
  logit dumps and all 360 intermediate files are bitwise equal to the
  corrected mHC baseline; both runs finished on all 12 ranks.
- `resume-packets-v4/replay.log` and `replay-summary.json`: the updated replay
  passed 726/726 checks over three positions and 16 layers
  (0–8, 14, 20, 24, 28, 32, 36, 39). Minimum cosine is 0.999998019 and maximum
  relative RMS error is 0.001992132 (0.199%). All 1,966,080 values in 96 mHC
  post comparisons match exactly with identical inputs. These checks use
  the explicit NumPy mHC expression and both numerical gates.

The new runner and replay workflow are documented in `a64fx/ds41f/README.md`.
Source snapshots and SHA256 manifests are retained beside the run binaries.

The detached `resume-long-comparison-v1.sh` ran the corrected mHC baseline and
combined-broadcast version sequentially, at 48 threads/node with demand paging,
2 GHz, `--generate 1100 --ignore-eos --max-context 1048576`. Both completed on
all 12 ranks. These times include the six-token prefill and exclude resident
loading; diagnostics were limited to nine logit dumps, with no layer dumps.

| Run | Elapsed seconds | Generated tokens/s | Minimum final MemAvailable |
| --- | ---: | ---: | ---: |
| `resume-long-mhc-v2` | 321.126 | 3.425 | 3.860 GB |
| `resume-long-packets-v4` | 309.411 | 3.555 | 3.858 GB |

This pair measured a 1.03786x improvement (3.8%). All 1,105 input/next-token
triples and the nine logit dumps match exactly; the logits also match the
corresponding runs with intermediate dumping enabled. Evidence:
`resume-long-comparison.json`. The run crosses both compression ratios'
512-row selection boundary; its history is 1,105 positions, not 1M positions.

Cross-build command for the changed runner and operator regressions:

```sh
TMPDIR="$PWD/tmp/ds41f" make -C a64fx/ds41f ds41f_run test_ops \
  A64FX_CC=fccpx A64FX_MPICC=mpifccpx
```

The validated combined-broadcast runner SHA256 is
`3263ec1f7c0c70883e9cd6aba2f8be9c8993074038750219ce26813d56c43e2a`.

The same binary subsequently passed two bounded chat smoke runs, each on all
12 ranks with `--max-context 4096 --generate 96` and normal EOS stopping:

- `resume-chat-sky-v4`: 16 prompt tokens, 47 generated tokens including EOS,
  16.645 s after loading. It correctly explained shorter blue wavelengths
  scattering more strongly than red wavelengths.
- `resume-chat-math-v4`: 18 prompt tokens, 2 generated tokens including EOS,
  4.951 s after loading. The response to `What is 17 times 23? Reply with
  just the integer.` was exactly `391` followed by EOS.

Both prompts use the checkpoint's chat formatter. The immutable launcher is
`resume-chat-comparison-v1.sh`; decoded responses and completion records are
saved in `resume-chat-summary.json`. These are smoke checks, not a broad
quality evaluation.

The corrected independent reference completed all 40 layers for the first
three prompt positions, starting from the embedding and carrying its own
state. `resume-packets-v4/compare-corrected-reference.log` passes the unchanged
0.999 cosine plus argmax gate:

| Position | Reference / A64FX next token | Logit cosine |
| --- | ---: | ---: |
| 0 | 5 / 5 | 0.999988319 |
| 1 | 41079 / 41079 | 0.999947498 |
| 2 | 294 / 294 | 0.999578117 |

The reference used `resume-mhc-v2/source/reference_numpy.py`,
`resume-mhc-v2/prompt-first3.ids`, and `--generate 1`, with four BLAS threads
on the frontend. Its log and logits are under `resume-mhc-v2/reference*`.
Comparison command:

```sh
OPENBLAS_NUM_THREADS=2 python3 a64fx/ds41f/compare_logits.py \
  --reference-prefix tmp/ds41f/job51562789/resume-mhc-v2/reference \
  --actual-prefix tmp/ds41f/job51562789/resume-packets-v4/logits --positions 3
```

This is a full-graph check for three positions, distinct from the replay that
injects recorded inputs into individual operators. It does not establish the
complete nine-position numerical gate, GPU parity, batched prefill performance,
or actual 1M-history execution. All continuation tests finished; job 51562789
was left idle with its staged shards and working bridge available.

## Single-request profiling at 1K history (job 51562789)

The performance target is **20+ decode tokens/s at approximately 1K actual
history**, clarified on 2026-09-12. This requires less than 50 ms/token.
The six-token capital prompt with `--generate 1100 --ignore-eos` supplies the
same 1,105 input positions as the earlier baseline. Report positions
1000–1104 (105 samples); retain `--max-context 1048576` for the existing
memory-admission check. This is not a 1M-history speed measurement.

The bounded profiler records main-thread spans per position/layer/rank with
`--profile-start 16 --profile-count 1089`; it writes binary arrays and JSON
metadata after the timed run. The report uses producer-owner spans and the
slowest parallel expert work. Collective remainders include rendezvous/skew,
not just transport. No new barriers or in-loop profile writes are introduced.

Initial measured critical path (`tmp/ds41f/job51562789/profile-v1/`):

| Stage | ms/token at positions 1000–1104 |
| --- | ---: |
| Attention, including projections/index/RoPE | 145.796 |
| Routed experts, slowest rank per layer | 34.458 |
| Shared experts, including output sum/round | 29.312 |
| mHC mixes/pre/post | 46.615 |
| Engram local fetch/decode + owner projection | 17.102 |
| Gate | 4.388 |
| Head norm/matvec/selection | 3.094 |
| Broadcasts + reduction rendezvous | 9.099 |
| Measured token total | **289.802 (3.451 tok/s)** |

The reconstruction differs from token timing by only -0.118 ms/token.
Nested FP8 GEMVs consume **70.485 ms/token**, reading 6.843 GB/token at an
effective 97.1 GB/s. The head matvec is only 1.885 ms/token. Scalar output
rounding alone costs 20.640 ms inside dense linears, plus substantial rounding
time inside RoPE, mHC and expert spans. Sparse attention costs 24.691 ms.
Six routed experts activate an average 4.871 of 12 ranks per layer; the busiest
rank owns 1.910 experts on average. Summing all ranks' collective wait would
misidentify producer/straggler waits as network traffic.

The profile run took 310.127 s versus 309.411 s for the uninstrumented baseline
(0.23% longer overall). All 12 ranks finished, all 1,105 input/next-token
triples and the first nine logit arrays were bitwise unchanged. Minimum final
MemAvailable was 3,871,997,952 bytes. Binary SHA256:
`62c3234211e87da23517e011f55492cc7d3c60b7a173a2b16e1716c4953fcf27`.

The first optimization replaces scalar BF16 round/conversion loops with SVE
integer rounding, preserving ties-to-even, NaN sign/payload handling and tails.
It also reuses that routine for routed-expert gate/up/output rounding.
`round-sve-v1/sve-test.log` passes 426,112 bit-exact cases, every BF16 upper
16-bit pattern at six rounding boundaries, and lengths 0–256 with canaries.
The full 12-node run takes 219.066 s; positions 1000–1104 average **206.638
ms/token (4.839 tok/s)**. All 1,105 token triples and the first nine logit
arrays remain bitwise identical. Minimum final MemAvailable is 3,860,201,472
bytes. This removes 83.165 ms/token at 1K, a 28.7% latency reduction.

Artifacts contain versioned source snapshots, binaries, SHA checks, launch
scripts, per-rank logs, `comparison.json`, `profile-1k.json` and
`report-1k.txt`. Reproduce the report on the frontend:

```sh
OPENBLAS_NUM_THREADS=2 tmp/ds41f/metadata-venv/bin/python \
  a64fx/ds41f/profile_report.py tmp/ds41f/job51562789/profile-v1 \
  --start 1000 --stop 1105 --json tmp/ds41f/job51562789/profile-v1/profile-1k.json
```

The parallel SwiGLU and mHC post updates (`pointwise-v1`) pass independent
bit-exact checks at lengths 1, 511, 512, 513, 2304, 5120 and 5123, including
in-place mHC and separate product rounding. Their full run takes 198.105 s;
positions 1000–1104 average **187.531 ms/token (5.332 tok/s)**. All 1,105 token
triples and nine logit arrays match the initial profile run exactly.

Dense FP8-to-BF16 expansion was tested and **rejected**. Although small
microbenchmarks improved, the complete runner regressed: `dense-bf16-v2`
took 261.109 s and `dense-adaptive-v1` took 240.602 s, versus 198.105 s for
the compressed pointwise runner. The expansion option and kernels were removed;
versioned experiment sources and logs remain under the job directory.

### Engram prefetch and receive-slot correctness

Fine timing attributes 11.920 of 11.980 ms of the Engram fetch stage to
`pread`, with only 0.051 ms for row conversion (`dense-bf16-v2`, same token
sequence). `--engram-prefetch` uses one worker/rank and two 24-by-256 FP32
buffers (49,152 bytes total) to fetch both layers' rows once their IDs are known.
The main thread waits at layers 1 and 14. The worker does not use MPI/uTofu;
MPI requests `MPI_THREAD_FUNNELED`, and profiler cursors are thread-local.

The initial `prefetch-v1` trial is **invalid**: it diverged at position 57 and
aborted with a nonfinite check after position 526. Prefetch exposed a latent
receive-slot reuse race in the no-ack recursive-doubling transport. A bounded
reproducer in `comm-skew-v1` delays rank 10 by 3 ms after the receive trailer
arrives, before reading its payload. Without acknowledgments, iteration 0
reads 866 instead of 66; with acknowledgments, all 300 reductions of 20,484
floats pass on all 12 ranks. Exact evidence is in:

- `comm-skew-v1/output.51562789/0/41/stderr.41.10`
- `comm-skew-v1/output.51562789/0/42/stdout.42.0`

The DS4.1 runner now enables the existing receive-acknowledgment path; the
shared all-reduce header and its bounded retry policy are unchanged. The
3 ms skew test validates this case, not arbitrary receiver delays.
`test_prefetch` checks 96 generations
against synchronous reads, zero fill for non-owned rows, short-read errors,
joining with pending work, and profiler isolation. It passes on A64FX and
the frontend. The corrected `prefetch-v2` run completes in 196.071 s, with
**186.392 ms/token (5.365 tok/s)** at positions 1000–1104. All 1,105 triples
and nine logits files are bitwise identical to `profile-v1`; minimum final
MemAvailable is 3,882,418,176 bytes. The remaining critical-path Engram wait
is 5.551 ms/token. This comparison includes the required acknowledgment cost.

### Sparse-attention loop

The weighted-value loop loads four adjacent SVE vectors from each selected
KV row, consuming an A64FX cache line while preserving the original per-lane
FMA order. The predicated fallback handles the remaining dimensions.
`test_ops` passes 48 reference cases around vector/block boundaries, including
empty, masked and duplicate selections with output canaries. The 64-head,
512-dimensional, 640-selection component benchmark reports max absolute error
1.49012e-6 versus the independent reference and 0.284 ms for the SVE kernel
(`sparse-loop-v1/attention-bench.log`).

The full `sparse-loop-v1` run completes in **184.819 s** with Engram prefetch
enabled. Positions 1000–1104 average **172.438 ms/token (5.799 tok/s)**,
with p50 172.695 ms and p95 175.904 ms. The sparse-attention span falls from
25.852 to **11.107 ms/token** versus `prefetch-v2` (57.0% less), reducing
total token latency by 7.5%. All 12 ranks finish, all 1,105 token triples and
nine saved logit arrays match the initial profile run bitwise, and minimum
final MemAvailable is 3,833,987,072 bytes. Runner SHA256:
`96e4571732ea471beb4c9c3c76319556fa5191b4165e0f0802cfdc1ad3ae837e`.

The final `sparse-no-prefetch-v1` control retains the same acknowledgment and
sparse-loop changes but omits `--engram-prefetch`. It completes in 189.008 s;
the 1K window averages **175.922 ms/token (5.684 tok/s)**, p95 181.767 ms.
All 1,105 triples and nine logit arrays match the uninstrumented baseline
bitwise; all 12 ranks finish with minimum final MemAvailable 3,863,412,736
bytes. Prefetch reduces exposed Engram I/O from 11.951 to 5.985 ms/token and
improves overall throughput by 2.0% in this sequential pair. These are single
full-run comparisons, not repeated-trial confidence intervals. The final
warning-clean source differs from the prefetch-on snapshot only by an
explicit const-array pointer cast in the runner call, plus test cleanup.
Final runner SHA256:
`58af7ef72cd965f1c215ea08eed406d2a24fa076811f1559b7a64b0ba640ced4`.

### Completed optimization checkpoint (2026-09-12)

**Historical pause:** Engram prefetch and sparse-attention loop optimization
were complete at the requested boundary, with no further runs queued then.
Job 51562789 ended at **17:57:45 JST on 2026-09-12** and its `/local` data expired.
The later INT8 request resumed work on job 51569201, described below. Follow
`a64fx/remote-dev-procedure.md` to reconnect or allocate again; do not assume
old `/local` paths survive.

| Valid run | 1K ms/token | 1K tok/s | Whole inference loop, seconds |
| --- | ---: | ---: | ---: |
| Initial fine profile | 289.802 | 3.451 | 310.127 |
| SVE BF16 rounding | 206.638 | 4.839 | 219.066 |
| Parallel pointwise operations | 187.531 | 5.332 | 198.105 |
| Engram prefetch + receive acknowledgments | 186.392 | 5.365 | 196.071 |
| Sparse loop + Engram prefetch | **172.438** | **5.799** | **184.819** |
| Sparse loop, prefetch disabled (control) | 175.922 | 5.684 | 189.008 |

The final optimized run improves throughput by **68.1%** and reduces latency
by **40.5%** relative to the initial fine profile. The **20+ tok/s target is
not reached**: 172.438 ms/token still needs to fall below 50 ms. Attention
remains the largest stage at 84.035 ms/token, including 11.107 ms of sparse
attention. Across dense projections, compressed FP8 GEMVs take 71.419 ms/token
for 6.843 GB of weights, approximately 95.8 GB/s. Other large stages are
shared experts (19.929 ms), routed experts on the slowest rank (19.751 ms),
and the two mHC mixes (17.014 ms). Within the mixes, serial normalization
costs 7.319 ms and F32 matvecs 8.323 ms. These nested timings overlap their
parent stages; they must not be added together. They identify the remaining
bottlenecks for a later session, without reopening tuning now.

Exact build and validation commands, from the repository root on the frontend:

```sh
TMPDIR="$PWD/tmp/ds41f" make -C a64fx/ds41f \
  ds41f_run ds41f_sve_test test_ops test_pointwise test_prefetch \
  bench_attention_kernel A64FX_CC=fccpx A64FX_MPICC=mpifccpx
TMPDIR="$PWD/tmp/ds41f" make -C a64fx/ds41f test \
  CFLAGS='-O2 -Wall -Wextra -Wpedantic -std=c11 -fopenmp'
```

The cross-build uses `-Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast`,
OpenMP, and pthreads for the runner/prefetch test. Versioned compute launch
scripts are `prefetch-sparse-v1.sh` and `sparse-no-prefetch-v1/run-v1.sh` under
`tmp/ds41f/job51562789/`; source snapshots are in each results directory.
The optimized
run uses the following command from its fresh shared results directory:

```sh
env XOS_MMM_L_PAGING_POLICY=demand:demand:demand \
  OMP_NUM_THREADS=48 OMP_PROC_BIND=close OMP_PLACES=cores \
  mpiexec -np 12 ./ds41f_run \
  --stage-root /local/u14346/ds41f-51562789 \
  --prompt-ids /vol0006/mdt0/data/hp250467/work/gemm/ds4f/tmp/ds41f/job51562789/prompt-capital.ids \
  --generate 1100 --ignore-eos --max-context 1048576 \
  --logits-prefix logits --logits-count 9 \
  --profile-start 16 --profile-count 1089 --engram-prefetch
```

Saved logits cover only positions 0–8, outside the profiling window.
`comparison.json`, `profile-1k.json`, and `report-1k.txt` in each final run
directory contain the correctness checks and timing summaries. Component
logs record `BF16_ROUND PASS bit_exact=426112`, `POINTWISE PASS bit_exact`,
`SPARSE_TAILS PASS reference_cases=48`, and `PREFETCH PASS ... generations=96`.
These optimization regressions establish agreement with the existing runner;
the earlier independent-reference and actual-1M-history limitations still
apply.

### Future tasks after the pause

Resume from commit `dbce50bf` and the `sparse-loop-v1` / `sparse-no-prefetch-v1`
artifacts. The target remains **single-request decode above 20 tok/s at
approximately 1K actual history**. Engram prefetch and the sparse value-loop
update are complete; the tasks below are pending, with no runs queued.

Performance work, in priority order:

1. **Reproduce the final baseline.** Check allocation/bridge health and staged
   manifests using the remote development procedure; restage if `/local` has
   expired. Run the final committed binary with prefetch enabled and the same
   prompt/options, then repeat sequential prefetch-on/off comparisons to
   distinguish the measured 2.0% gain from run variability. Keep each binary,
   source snapshot, SHA256 and log in a new results directory. Report positions
   1000–1104, including mean and p95; also measure with profiling disabled.
2. **Optimize compressed FP8 projections first.** Their 71.419 ms/token already
   exceeds the entire 50 ms target budget. Profile the actual `wq_b`, grouped
   `wo_a`, `wo_b` and shared-expert shapes in `ds41f_sve.c`, separating weight
   traffic, FP8/scale decoding and arithmetic. Inspect generated SVE code and
   measure CMG/thread placement and row blocking with realistic weight working
   sets. Test bounded conversion reuse or fused scale/decode loops. Keep FP8
   weights resident in compressed form; the rejected full BF16 expansion must
   not be restored based only on small, warm microbenchmarks.
3. **Reduce mHC mix overhead.** In `ds41f_run.c:mixes`, investigate the serial
   normalization (7.319 ms/token) and the 24-row F32 matvecs (8.323 ms/token).
   Measure vectorization, work distribution and parallel-region overhead.
   Changes to reduction order require explicit numerical comparison; preserve
   the separately rounded products in mHC post processing.
4. **Reduce expert critical-path time.** Shared experts cost 19.929 ms/token
   and the slowest routed rank costs 19.751 ms/token. Profile MXFP4 decoding,
   activation quantization, scratch reuse and the observed expert imbalance
   before changing placement. Evaluate overlap of shared and routed work only
   with an explicit core budget: concurrent OpenMP teams can compete for the
   same 48 cores. Keep owner synchronization and receive acknowledgments valid.
5. **Evaluate dense tensor parallelism if kernel tuning is insufficient.**
   Dense layers currently execute on one owner at a time. Estimate the extra
   communication and per-node memory before prototyping projection sharding
   across a small rank group. Require a measured end-to-end gain; distributing
   work must not replicate the full dense model or exceed the existing HBM
   admission budget. This is a larger architecture experiment, not an assumed
   route to 20 tok/s.

Correctness and acceptance work:

- **Complete independent-reference validation.** Rerun the corrected NumPy
  full-graph reference for all nine positions; only the corrected three-position
  check is complete. Retain the cosine >= 0.999 and matching-argmax gates.
  Extend bounded logit capture to selected positions near 1K in a separate
  correctness run; current saved arrays cover positions 0–8 only. Continue
  comparing all 1,105 token triples and component references after math changes.
- **Harden transport under longer receiver stalls.** Turn the temporary
  `comm-skew-v1` reproducer into a maintained regression and sweep delayed ranks,
  payload sizes and delays around/beyond the existing approximately 64 ms ACK
  retry budget. The shared transport can currently proceed optimistically
  after retry exhaustion. Design a bounded error/abort or proven receive-slot
  ownership scheme before claiming correctness under arbitrary scheduling skew;
  preserve the current 3 ms / 300-reduction passing case.
- **Require whole-run evidence for each retained optimization.** Keep compressed
  residency and memory guards, pass the affected component checks, and compare
  identical prompts and history windows on all 12 ranks. Re-profile the full
  runner and record MemAvailable, mean/p95 latency and numerical differences.
  Treat the 20 tok/s target as achieved only with repeated single-request
  measurements below 50 ms/token at actual 1K history.

Actual long-history/1M execution, KV checkpoint/restore and batched prefill
integration remain separate deferred tasks. Allocating the 1M cache does not
validate those paths; use the memory and acceptance sections below when that
scope is resumed.

## Resumed INT8 optimization (job 51569201)

The user resumed optimization toward **20+ tok/s for a single request around
1K actual history**, specifically requesting INT8 requantization and SDOT for
FP8 matrices. The earlier pause no longer applies. Job **51569201** has 12
nodes in normal 2000 MHz mode, from **18:00:29 JST September 12** until
**00:00:29 JST September 13**. The initial node is `d29-2208c`; its bridge uses
local port 42394, reverse port 32394 and server port 21265. Tunnel control
state is in `tmp/i8`. Port 42393/job 51562789 is expired.

Exact original weights were restaged from `/home/u14346/models/ds41f` into
`/local/u14346/ds41f-51569201/rank<R>` using bounded, page-cache-evicting copies.
Original Engram metadata was reused after checking its hash. All 12 staging
ranks finished. Results and immutable source/binary snapshots are under
`tmp/ds41f/job51569201/`. Launchers are serialized; do not start another MPI
program while one is running, and do not edit active scripts or binaries.

### Measured progress

All full runs below generate 1,100 outputs from the same six-token prompt,
allocate the 1M-token cache, and measure positions 1000–1104 (105 samples).
They do not establish execution at an actual 1M history. FP8 and INT8 may
generate different histories; fixed-history quality comparisons are separate.

| Run | ms/token near 1K | tok/s | Whole loop, s | Validation (control stated) |
| --- | ---: | ---: | ---: | --- |
| `baseline-v1` | 173.463 | 5.765 | 186.497 | 1,105 token triples + nine logits bit-exact |
| `fp8-overlap-full-v1` | 147.852 | 6.764 | 159.811 | 1,105 token triples + nine logits bit-exact |
| `int8-overlap-full-v1` | 122.272 | 8.178 | 131.791 | Experimental INT8; original malloc pack allocation |
| `int8-retained-full-v1` | 121.064 | 8.260 | 129.734 | Same INT8 arithmetic, warning cleanup |
| `int8-mmap-probe-v1` | 101.761 | 9.827 | 108.991 | Bit-exact to retained INT8 control |
| `int8-local-pages-full-v1` | 99.939 | 10.006 | 107.998 | Bit-exact to retained INT8 control |
| `fp8-local-pages-full-v1` | 147.127 | 6.797 | 158.196 | 1,105 token triples + nine logits bit-exact |
| `int8-mpi-bcast-full-v1` | 98.448 | 10.158 | 105.687 | Bit-exact to retained INT8 control |
| `int8-selection-full-v1` | 97.965 | 10.208 | 104.442 | Bit-exact to retained INT8 control |
| `fp8-selection-full-v1` | 144.516 | 6.920 | 154.926 | 1,105 token triples + nine logits bit-exact |

The exact FP8 improvement combines the contiguous-head RoPE loop, SVE FP64
mHC normalization, Engram scale caching and shared/routed expert overlap.
Minimum final MemAvailable is **4,195,811,328 bytes**, with all 12 ranks
finished. Its binary SHA256 is
`3d7db1d2d484d82dceef2a7f79bde0df9559367e053676e199c06b2c0d93467a`.
The norm-only nine-position control `fp8-norm9-v1` also matched bit-exactly.
The 20+ tok/s target is **not reached**.

The reproduced baseline profile attributes 72.243 ms/token to dense FP8
GEMVs, reading 6.843 GB/token (95 GB/s effective). Attention takes 85.298 ms,
shared experts 20.296 ms, routed experts on the slowest rank 19.611 ms, and
mHC mixes 17.285 ms. After the exact changes, attention is 82.893 ms and the
combined routed/shared stage is 26.577 ms. The norm drops from 7.430 to about
0.31 ms/token; the original mHC matrix-reduction order is retained.
Nested timings overlap their parent stages.

### INT8 format, conversion cost and validation

`--fp8-int8-block 32|64|128|256` enables a row/block-scaled signed INT8 SDOT
GEMV path. Default zero preserves FP8. The converter applies the original
FP8 E4M3 and E8M0 scales, finds each row/block maximum, rounds symmetrically
to [-127,127], and packs groups of rows for paired SVE SDOT accumulators.
Each row/block has one FP32 scale. Activations retain the existing FP8
quantization first, then use matching blockwise INT8 quantization; output
BF16 rounding stays in place. Grouped `wo_a` is supported. The large grouped
activation buffers are quantized in parallel. Batched INT8 GEMM remains
unimplemented; this continuation targets batch-one decode GEMV.

Conversion occurs **once after reading the original files from `/local`**,
releasing each original FP8 matrix after its replacement is ready. Block 32
adds 12.5% scale storage plus row padding, while admission accounts for both
representations of the current tensor when source mappings can be released.
With pooled source allocations, admission conservatively reserves the original
store plus all packed replacements because the pool can retain freed pages. Full-model conversion measured
**0.2727–0.4608 seconds per rank**, versus **52.688–57.028 seconds** of resident
startup. The later fresh-allocation run took at most 0.4914 s/rank.
This remains below 1% of startup, satisfying the user's inexpensive
online-requantization condition. No offline converted files were written;
the original shared safetensors and staged weight files remain authoritative.

The first real-matrix probes (`int8-components-v1`) measured query projection
32768x1280 at 0.444 -> 0.104 ms (4.27x) and grouped output 8192x4096 at
0.336 -> 0.157 ms (2.14x), block 32. Parallel activation quantization later
reduced the grouped probe to 0.115 ms (`overlap-runs-v1/components.log`). These
are reused-tensor component timings, not a full-run speed claim. Cold-weight
probes (`int8-tile-v1`) selected the original four-row format: four/eight/sixteen
rows take 0.0998/0.1083/0.1065 ms for `wq_b`, 0.1219/0.1203/0.1247 ms for
`wo_a`, and 0.1408/0.1517/0.1309 ms for `wo_b`. The mixed larger-tile gains
were insufficient to replace the four-row format; both prototypes passed the
162 arithmetic cases. No larger tile is retained.

Numerical validation distinguishes kernel arithmetic from quantization loss:

- `test_int8` passes **162** reference cases covering blocks 32/64/128/256,
  padded rows, grouped inputs, the parallel input-quantization branch,
  canaries, zeros, nonfinite rejection and subnormal scales. SVE SDOT is
  checked against integer dots with double rescaling; native warning-clean
  builds also pass. AddressSanitizer was unavailable on this frontend.
- `test_int8_attention` replays 54 real attention inputs (nine positions in
  layers 0/1/2/8/14/20). Block 32 passes its cosine >= 0.999 exit gate, but
  relative RMS reaches 4.23%; this is **not** a pass of the stricter 1% gate.
  Block 128 fails 11 of 54 cosine checks. Limiting quantization to query/output
  projections improves this local replay, but not full-model agreement.
- Full fixed-token replay `int8-all32-v1` matches **9/9 argmax choices**, but
  minimum logit cosine is **0.994185** and maximum relative RMS **10.77%**.
  The projection-only control reaches minimum cosine **0.952806** and maximum
  relative RMS **31.05%**, also with 9/9 argmax choices. Both fail the existing
  numerical gates and remain experimental; token-choice agreement alone does
  not establish acceptable model quality.
- Fixed-history replay at **positions 1000–1008** is complete in
  `fp8-replay1k-v1` and `int8-replay1k-v1`. All 1,105 input tokens are identical;
  INT8 next-token choices match **1,065/1,105 overall** and **104/105 at
  positions 1000–1104**. The nine saved argmax choices agree, but logit cosine
  falls to **0.902769873** and relative RMS reaches **0.439981917**. All nine
  fail the numerical gates. `comparison-1k.json` and `token-agreement.json`
  retain the evidence. These dump-enabled replays are quality runs, not speed
  measurements. `compare_run_logits.py` rejects missing/divergent histories.
- A second INT8 residual plane lowers individual matvec relative error to
  about 5e-5, but the full replay still reaches cosine 0.994099 / relative RMS
  12.82%, and costs more time/memory. This experiment is **not retained**.
  Its sources and evidence are in `refined-runs-v1` and
  `int8-refined32-quality-v1`.

`--fp8-int8-scope projections` selects only `attn.wq_b`, `attn.wo_a`,
`attn.wo_b` and shared experts; default scope `all` converts every FP8 matrix.
INT8 remains explicitly opt-in and must not be described as an accepted
numerical replacement for the FP8 path.

### Resident memory placement

The large gap between component and full-run SDOT timings was traced to
allocation reuse. In `resident-int8-probe-v1`, rank-zero `wo_a` and `wo_b` took
about 0.44/0.45 ms even on repeated real inputs, while `wq_b` reached 0.0985 ms.
The packed `wo_a`/`wo_b` pointers landed at the end of the malloc arena, and
`int8-retained-full-v1/numa-rank0.txt` shows their full mappings on NUMA node 7.
The original weight scales were normal (E8M0 codes 114–121), excluding a
subnormal arithmetic explanation for these tensors.

Fresh anonymous allocations bypass the Fugaku `libmpg` malloc pool and let
parallel conversion place new pages. Changing only the packed INT8 allocations
improved **8.260 -> 9.827 tok/s**, with identical 1,105 token triples and nine
logit arrays. `ds41f_alloc.h` uses Linux LP64 raw anonymous mmap/munmap, with
posix_memalign/free fallback elsewhere. The weight files are still read into
resident anonymous buffers using bounded pread plus fadvise; this is not
file-backed model mmap.

`--weights-local-pages` applies fresh allocation to the original tensors too.
Together with reusing a single 640x512 attention row workspace, it reached
**10.006 tok/s** and minimum final MemAvailable **3,993,108,480 bytes**.
Rank-zero MemAvailable stayed close to its post-load value (4.059 -> 3.993 GB).
The combined comparison does not isolate the workspace from source allocation.
The FP8 control remained bit-exact and reached **6.797 tok/s**. The loader
fixture passes six cases covering ordinary/fresh allocation, page boundaries,
size rejection and independence from subsequent source-file changes.
A later `int8-local-pages-full-v1/numa-rank0.txt` capture occurred during
teardown and must not be used as proof of full-resident placement.

The native MPI broadcast prototype then reached **10.158 tok/s**. Residual
broadcast time fell from about 5.6 to 3.5 ms/token; other synchronization spans
changed with rank skew. INT8 token triples and nine saved logits stayed exact.
It is retained as `--mpi-broadcast`; uTofu still performs reductions. Native
broadcast normalizes signed zero as the previous sum-based broadcast did.
The component test checks both modes, 12 owners, seven lengths including a
32768-float chunk boundary, canaries and delayed receivers.

At this point the critical path is attention **45.511 ms/token**, combined
experts **20.909 ms**, mHC mixes **10.233 ms**, gate **4.024 ms** and residual/FFN
broadcasts **6.076 ms**. Attention includes sparse attention **11.600 ms**,
indexing **6.289 ms**, and query/output INT8 projections **15.362 ms**.
INT8 kernels read 7.691 GB/token at an aggregate effective **358 GB/s**.
These nested measurements overlap parent stages. Simply improving INT8 GEMV
cannot remove the remaining roughly 48 ms needed for the 50 ms/token target.

Vocabulary argmax now uses SVE finite/max scans followed by the first matching
index. Its measured cost falls from **1.230 to 0.034 ms/token**. Gate score
calculation uses parallel independent scalar math, retaining the existing
selection and normalization order; total gate time falls from **4.024 to
3.498 ms/token**. The combined INT8 run reaches **10.208 tok/s**, with all
1,105 token triples and nine saved logit arrays unchanged. `test_selection`
passes 130 argmax cases and 20 bit-exact gate cases, including ties, tails,
nonfinite rejection and the parallel threshold, on native and SVE builds.
Its binary SHA256 is
`45aa28c1b0e27d720f647f8a68b7206c1033a06ff506dcbbe85886dfe54eff87`.
The same binary's FP8 run reaches **6.920 tok/s** (144.516 ms/token),
20.0% above the same-allocation baseline. All 1,105 token triples and nine
saved logits remain bit-exact to original FP8; all 12 ranks finish with minimum
final MemAvailable **4,044,226,560 bytes**. INT8 minimum final MemAvailable
is **3,968,729,088 bytes**. These runs use profiling; final unprofiled repeats
are recorded separately.

With profiling and logit dumps disabled, `int8-final-unprofiled-v1` measures
**96.494 ms/token / 10.363 tok/s**, p95 **98.467 ms**, over the same 105 positions.
The whole loop takes **102.940 s**. All 1,105 token triples match the profiled
INT8 run, all 12 ranks finish, and minimum final MemAvailable is
**3,997,958,144 bytes**. Per-token timings are the runner's rank-zero wall clock;
p95 uses linear interpolation. The `summary.json` records each unprofiled run.
`int8-final-unprofiled-v2` confirms **96.279 ms/token / 10.386 tok/s**,
p95 **98.463 ms**, whole loop **102.578 s**, and minimum final MemAvailable
**3,999,596,544 bytes**. It also matches all 1,105 token triples. Thus repeated
uninstrumented INT8 throughput is **10.36–10.39 tok/s**, still below 20 tok/s.
Both binaries have the same SHA256 given above.

The matched `int8-source-pool-control-v1` omits only `--weights-local-pages`
while retaining fresh INT8 allocations and the reused row workspace. It reaches
**98.015 ms/token / 10.203 tok/s**, p95 **99.889 ms**, whole loop **104.492 s**,
and minimum final MemAvailable **3,576,233,984 bytes**. All 1,105 token triples
match. Fresh original-weight allocations therefore add roughly 1.7% throughput
and about 0.42 GB final headroom in these runs; the explicit flag remains in the
benchmark configuration, while its default stays off.

All recorded launchers have finished successfully, with no further inference
runs queued. Job **51569201** remains available until the end time above; its
original staged `/local` weights can be reused while that allocation lives.
The 20+ tok/s target and full INT8 numerical acceptance remain **open**.

### Other changes and rejected probes

`--engram-scale-cache` reads each rank's two raw scale shards (about 512 MB)
into HBM with bounded reads and page-cache eviction. It removes one of the two
reads per Engram row and retains the 2 GiB admission floor. The prefetch fixture
checks budget rejection, bit-exact cached/uncached rows, closing the scale FDs
before cached reads, short-read propagation, pending close and profiler TLS.

`--hc-mix-sve` uses an explicitly vectorized FP64 sum of squares. It passes
352 bit-exact norm cases with tails. Splitting the 24-row mHC matrix across
48 K-partitions halved its component time but caused full-model logit drift
(minimum cosine 0.989646 in `fp8-extras9-v1`); that split was removed.
The RoPE head/angle loop swap passes 360 bit-exact layout/tail/position cases
and roughly halves the RoPE component time.

`--shared-overlap` computes the owner's shared expert before the routed
reduction using the same OpenMP team, while other ranks finish their routed
experts. It adds the shared output after reduction in the original order.
The profiler reconstructs `EXPERTS_AND_SHARED` from each rank's combined
local work, avoiding double-counting the overlap.

Other rejected temporary probes: power-of-two INT8 scales (worse attention
agreement), INT8 SDOT MXFP4 decoding (slower, especially down projection),
four-row floating MXFP4 interleaving (slower), and FEXPA sparse softmax
(only a small gain with extra approximation). None is enabled in the runner.

Retained component checks are archived in `retained-checks-v1/components.log`,
`local-pages-runs-v1/components.log` and `selection-runs-v1/components.log`
(with MPI stdout under that run's `output.51569201/`). Representative output:

```text
DS41F_KERNEL_TEST PASS
SPARSE_TAILS PASS reference_cases=48 masked duplicate_ids empty canaries
ROPE_LAYOUT PASS bit_exact=360 canaries inverse long_positions
MHC_MIX PASS norm_bit_exact=352 tails
INT8 PASS cases=162 blocks=32,64,128,256 padded_rows grouped canaries zero nonfinite subnormal_scales
PREFETCH PASS bit_exact generations=96 remote_zeros short_read_error pending_close profiler_TLS scale_cache budget cache_only_reads
TENSOR_LOCAL PASS cases=6 malloc fresh_pages boundary_sizes source_independence size_rejection
SELECTION PASS argmax=130 gate=20 first_ties nonfinite_rejection bit_exact_weights
BROADCAST PASS modes=2 owners=12 sizes=7 chunk_boundary canary signed_zero delayed_receivers
```

### Reproduction and future tasks

Build on the frontend (no `/tmp`):

```sh
mkdir -p tmp/ds41f
TMPDIR="$PWD/tmp/ds41f" make -C a64fx/ds41f ds41f_run test_int8 \
  test_int8_attention test_tensor_local test_selection test_broadcast \
  test_rope_layout test_mhc_mix test_prefetch \
  A64FX_CC=fccpx A64FX_MPICC=mpifccpx \
  A64FX_CFLAGS='-Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -Wall -Wextra -Wpedantic'
```

Snapshot the binary into a new shared results directory and verify its SHA256
on the compute node. Inside job 51569201, with no other MPI program running:

```sh
export TMPDIR=/local/u14346/ds41f-51569201
export XOS_MMM_L_PAGING_POLICY=demand:demand:demand
export OMP_NUM_THREADS=48 OMP_PROC_BIND=close OMP_PLACES=cores
mpiexec -np 12 ./ds41f_run --stage-root /local/u14346/ds41f-51569201 \
  --prompt-ids /absolute/repo/tmp/ds41f/job51562789/prompt-capital.ids \
  --generate 1100 --ignore-eos --max-context 1048576 \
  --engram-prefetch --engram-scale-cache --hc-mix-sve --shared-overlap \
  --weights-local-pages --mpi-broadcast --fp8-int8-block 32
```

Omit `--fp8-int8-block 32` for FP8. For an instrumented comparison add
`--profile-start 16 --profile-count 1089 --logits-prefix logits --logits-count 9`,
then run `profile_report.py RESULTS --start 1000 --stop 1105` on the frontend.
For near-1K numerical comparison use the same fixed 1,105-token input file in
both runs and `--generate 1 --logits-start 1000 --logits-count 9`.

Remaining tasks, in priority order:

1. Resolve accumulated INT8 quality loss. Local cosine/argmax agreement is
   insufficient; retain the full fixed-history cosine >= 0.999 / relative RMS
   <= 1% gates. Isolate activation versus weight quantization and sensitive
   layers with bounded same-input captures. The second-plane and projection-only
   experiments already failed to resolve the full-model error.
2. Reduce the remaining attention and expert critical paths. Evaluate dense
   tensor parallelism with a concrete staging, communication and HBM budget;
   most ranks currently wait while one owner runs attention. No tensor-parallel
   implementation is present. Optimize routed MXFP4 using actual cold weights
   and realistic expert placement before revisiting rejected SDOT probes.
3. Improve mHC matrix execution while controlling accumulation error. Generated
   code uses ordered SVE `fadda` within each 20,480-column row; changing the
   reduction order already caused full-model drift. Benchmark any new layout
   against both component arithmetic and the full fixed-history replay.
4. Repeat actual-1K unprofiled performance measurements after every retained
   structural change; do not claim 20+ tok/s until repeated runs exceed it.
   Repeat the matched source-allocation control on another allocation before
   changing the current opt-in default.
5. Complete the corrected nine-position independent NumPy reference and retain
   transport skew/ACK regression checks. Actual-1M execution, KV persistence,
   batched prefill and INT8 GEMM remain separate unvalidated future work.


## Checkpoint accounting and active staging (job 51562789)

Header inventory supersedes the rough per-node fit estimate below:

| Category | Exact bytes |
| --- | ---: |
| 40-layer routed experts | 288,777,830,400 |
| Engram tables including scales | 202,758,032,400 |
| Dense text weights | 9,846,748,608 |
| Excluded auxiliary/vision/MTP | 8,903,411,592 |

Expert shards plus fully replicated dense weights consume 31.58 GiB/node,
before runtime state. Pure EP with replicated dense weights therefore does
not meet the memory budget. Initial inference must distribute dense layer
ownership as well as routed experts. Keep `expert_id % 12` ownership; assign
dense layers round-robin (`layer % 12`), with embeddings/head separately
accounted. Broadcast owner-produced activations for EP and combine routed
outputs using uTofu. The runner now shares packed KV/index source rows and
selection state using that transport.

### Six-hour continuation, allocation 51619965

The 12-node normal-mode interactive job **51619965** was launched with a
six-hour wall limit. A same-allocation rerun of the retained TP12 configuration
measured **43.416 ms/token**, **23.033 tokens/s**, p95 **44.767 ms** over
positions 1105--1232. This is the comparison point for the following trials.

Distributed TP8 preparation is runtime-correct but rejected for performance.
Its position-1104 result was `next=19`, and all 129 transitions matched the
retained trace. It measured **48.208 ms/token**, **20.743 tokens/s**, p95
**49.918 ms**. The profile attributes **23.950 ms** to attention and **17.486
ms** to preparation/context communication. The extra collectives cost more
than the distributed WQ-A/WKV work saves. The separately gated implementation
is retained for future fusion work; it is not a selected runtime setting.

Two 5,120-element output-reduction transport probes were also rejected. MPI
sum measured 45.104 ms for the two-transition smoke and registered-uTofu sum
44.480 ms, versus 43.585 ms for the contemporary control. Both produced the
exact `19, 369` token pair. Their temporary code was removed.

The retained `--attention-output-columns` experiment instead changes the WO-B
layout. Ranks 0--7 each stage all 5,120 output rows for their contiguous 1,024
input columns (and the corresponding 160x32 FP8 scale panel). Each rank applies
WO-B directly to its local eight-head projection; one 5,120-float allreduce
replaces both the 12x1,024 projected-vector allgather and row-output gather.
The stager copies bounded row segments and records column ranges in version-3
attention manifests. It adds about 70 MiB to the largest rank and stays within
the 32 GiB HBM admission limit.

The fixed 1105-token replay returned `19, 369`, and the 129-token trace matched
the retained baseline at every transition. The bounded profile at positions
1105--1136 measured **42.914 ms/token**, **23.302 tokens/s**, p95 **47.131
ms**; the same window in the contemporary baseline measured **44.515
ms/token**, **22.464 tokens/s**. Attention
fell to **17.397 ms** and WO-B itself cost **1.338 ms**. End-to-end improvement
in the matched profile window is **1.601 ms/token**, so this topology is a
valid building block but does not meet 30 tokens/s. Evidence is in
`tmp/ds41f/job51619965/column-128-v1`.

The next 30+ work must remove another **9.581 ms/token**. Prioritize the shared
expert/routed-result collective (currently about 10.13 ms including rank skew)
and fuse attention preparation communication with the retained column-WO-B
topology. Do not add another standalone preparation collective. A viable
composition should compute owner QA/KV while head ranks begin WQ-B work, carry
normalized Q/KV in an existing layer handoff, and overlap the final attention
reduction with owner-side mHC work where dependencies permit. Admit each step
only with the 1105-token replay, complete 129-transition comparison, and a
same-allocation 128-token timing run. The 30+ gate remains <=33.333 ms/token;
40+ still requires <=25 ms/token and likely column-sharded shared W2 plus a
combined routed/shared reduction.

The next shared-output iteration column-shards shared-expert W2 as well.
Every rank keeps its 192-element W1/W3 hidden shard and owns all 5,120 W2
output rows for those input columns. This preserves the per-rank W2 weight
count while removing the 2,304-element hidden allgather. Version-2 shared
manifests record the W2 weight and scale column ranges, and
`--shared-output-columns` requires the existing TP12 allreduce path.

This composition passed the full 129-transition comparison. Its bounded
profile measured **41.741 ms/token**, **23.957 tokens/s**, p95 **45.056 ms**.
The shared-overlap span fell from 4.286 to **2.706 ms**; attention remained
17.514 ms. A rooted reduction for the combined routed/shared FFN output was
also exact and reduced its nested collective span, but total latency regressed
to **43.019 ms/token**, **23.246 tokens/s**, so that FFN transport branch was
removed.

Only the layer owner consumes column-sharded WO-B output, so the attention
collective now uses `MPI_Reduce` to that owner instead of distributing the
5,120-float result to every rank. The complete 129-token trace remains exact.
The matched profile improves again to **41.573 ms/token**, **24.054 tokens/s**,
p95 **45.770 ms**, with attention at **17.344 ms**. Relative to the
same-allocation 44.515 ms profile, the accepted WO-B/shared-W2 column layouts
and rooted attention result remove **2.942 ms/token**. Evidence is in
`tmp/ds41f/job51619965/shared-column-128-v1` and
`tmp/ds41f/job51619965/shared-column-rooted-128-v1`.

Reducing column-WO-B across only the eight producing head ranks was exact but
did not improve the result. Layers owned by ranks 8--11 require a second
20 KiB transfer from rank 0; the measured profile was **41.580 ms/token**,
**24.050 tokens/s**, p95 **45.930 ms**, effectively tied with and slightly
slower than the simpler 12-rank rooted reduction. That code was removed.
Evidence is in `tmp/ds41f/job51619965/attention8reduce-128-v1`.

An isolated 5,120-float MPI allreduce benchmark also rules out collective
algorithm selection as a material 30+ step. Across 1,000 calls, Open MPI's
default, nonoverlapping, recursive-doubling, ring and Rabenseifner algorithms
measured 51.913, 51.839, 52.110, 52.514 and 51.273 microseconds/call. The best
spread is only about 0.026 ms/token across 40 layers without application rank
skew. No MPI algorithm override is retained.

The explicit `--shared-reduce utofu-bf16` mode applies the existing SVE-packed
BF16 recursive-doubling sum only to the combined routed/shared FFN output.
Other collectives retain their selected representations. It rounds after each
tree exchange and is therefore part of the approximate speed-first track.
Nevertheless, all 129 replay transitions matched the retained token trace.
The bounded profile measured **40.927 ms/token**, **24.434 tokens/s**, p95
**44.437 ms**. Expert-sum rendezvous fell from 3.326 to **1.949 ms**, and its
complete span from 10.434 to **9.168 ms**. Evidence is in
`tmp/ds41f/job51619965/shared-bf16-128-v1`.

Applying BF16 uTofu transport to column-WO-B at the same time was rejected.
Although its 129-token trace also matched, it measured **41.426 ms/token**,
**24.139 tokens/s**, p95 **46.486 ms**, with isolated samples as high as
54.233 ms. Sharing the uTofu reduction resource between both per-layer
collectives introduces variance and loses the FFN-only gain. The attention
option was removed; rooted MPI reduction remains selected for WO-B.

Approximate mHC mode 3 splits each of the 24 matrix rows into two 10,240-term
halves, filling all 48 persistent workers before a pairwise FP32 sum. Native
and A64FX bounded tests cover all four mHC modes and keep the split result
within the existing FP32 tolerance. The full-history smoke returned `19, 369`,
and all 129 generated IDs matched the retained trace. Combined with shared
BF16 uTofu reduction, its profile measured **40.722 ms/token**, **24.557
tokens/s**, p95 **44.085 ms**. Attention/FFN mHC mix spans fell from
1.173/1.117 to **1.086/1.019 ms**. This mode changes dot-product association
and remains an explicit speed-first approximation. Evidence is in
`tmp/ds41f/job51619965/mhc-split-128-v1`.

Reaching 30 tokens/s now requires another **8.240 ms/token**. The dominant
serialized spans are attention (17.34 ms), routed plus shared expert work
(9.68 ms), and the per-layer result rendezvous. The next implementation should
reduce the number of collectives rather than substitute transports: carry the
next layer's attention context in the residual handoff where possible, and
partition routed W2 columns so active expert ranks contribute directly to the
same output reduction without a second hidden publication. Preserve the
current column layouts as the comparison point.

`stage_backbone.py` stages source bytes unchanged to
`/local/$USER/ds41f-51562789/rank<R>` on the corresponding node. Engram rows
use contiguous ceil-div ownership. Dense tensors are staged once on rank 0
as a canonical disk source before the now-completed redistribution; this is not an instruction
to load all dense tensors into rank 0 HBM. Disk use is 50.808 GB on rank 0
and about 40.961 GB elsewhere. Total transferred payload is 501.383 GB,
roughly 28 minutes at an aggregate 300 MB/s excluding metadata/fsync overhead.

The stager uses 8 MiB bounded buffers, fsync and page-cache eviction, SHA256
sidecars and per-rank manifests. It skips completed size/stamp pairs on
restart; it does not rehash existing files, so a separate checksum audit is
required before production admission. Shared progress logs are under
`tmp/ds41f/job51562789/staging/`. Node-local files expire with the job.

MXFP4 checkpoint packing was verified against the local official
`inference/convert.py`: low/high nibbles represent adjacent columns, E2M1
has maximum 6, and scales are one E8M0 byte per row per 32 columns. The
legacy GGML split-half packing cannot be used on these source bytes directly.

## Recommended 12-node layout

Keep the FP8/FP4 model backbone resident in HBM2 and keep Engram on each
node's private `/local` filesystem, accessed through the uTofu owner path.
Keep the active 1M-token KV cache resident in HBM2. Use `/local` as a
persistent checkpoint and cold-session backing store, not as the decode-time
KV store.

```text
HBM2 on each node
  FP8/FP4 backbone shard       ~24-25 GiB
  active batch-1 KV cache      ~0.9 GiB at 1M context
  dequant/GEMM/attention work  remaining headroom

/local on each node
  Engram owner shard            ~16 GiB
  persistent KV checkpoint     optional
  inactive-session spill       optional
```

The current checkpoint is about 510.3 GB (475.2 GiB). Removing the Engram
tables leaves approximately 307-322 GB of backbone weights, depending on
whether Engram scales and metadata are included in the accounting. That is
about 24-25 GiB per node across 12 ranks. This fits 32 GiB HBM2 only if the
FP8/FP4 storage representation is retained; expanding the weights to BF16
does not fit.

## 1M-token KV cache

For batch 1 and the released 1M-token limit, the approximate cache budget is:

| Component | Approximate size |
| --- | ---: |
| Compressed FP4 KV latents | 0.63 GiB |
| FP8 compression scales | 0.08 GiB |
| Index KV and scales | 0.17 GiB |
| 40-layer 128-token sliding windows | negligible |
| **Total** | **~0.9 GiB** |

The model's compressed KV sources are layers 2, 8, 14, and 20. The other
layers reuse those caches. The reference implementation stores compressed
KV at `max_seq_len / compress_ratio` and quantizes it with FP4 plus scales:

<https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/model.py>

The cache should be rank-local or replicated according to the attention
implementation, with no per-token disk round trip. Reserve at least 2 GiB
per node for KV and cache-management overhead even though the raw estimate is
under 1 GiB.

## Persistence and `/local` streaming policy

Use a two-tier KV manager:

1. Decode from the HBM2-resident active cache.
2. Append completed cache blocks to a per-session checkpoint on `/local`.
3. On restart or session migration, prefetch the checkpoint sequentially into
   HBM2 before resuming decode.
4. Spill inactive sessions from HBM2 to `/local`; restore them as whole blocks
   when they become active.

Do not fetch individual KV rows from `/local` for every decode token. A query
may select hundreds of compressed positions, turning random local-storage or
uTofu reads into the decode bottleneck. The measured Engram path is roughly
20.9k row lookups/s at QD1 across 12 nodes, which corresponds to only about
40 tokens/s if 512 KV positions had to be fetched remotely for every token.

If HBM pressure eventually requires KV paging, page aligned blocks rather than
rows. Recommended starting points are 16-64 KiB blocks, double-buffered with
an asynchronous prefetch queue, with LRU admission for inactive sessions.
Measure block hit rate and restore bandwidth before enabling decode-time KV
spill.

## Implementation priorities

- Keep backbone weights compressed in HBM2 and dequantize into bounded scratch
  buffers.
- Keep the active batch-1 1M-token KV cache entirely in HBM2.
- Stage Engram shards to `/local` and use uTofu for owner communication.
- Add checkpoint metadata containing model revision, tokenizer/hash revision,
  rank topology, KV format, context length, and checksum.
- Checkpoint KV asynchronously at block boundaries; never pause decode for a
  full-cache synchronous write.
- Guard memory with a per-rank HBM budget and reject new sessions before
  evicting the active session's KV cache.

## Acceptance measurements

Before using persistent KV in production, measure:

- HBM resident bytes per rank at 1M context;
- prefill time and checkpoint write bandwidth;
- restore time from `/local`;
- active-cache HBM hit rate;
- decode tokens/s with zero spill;
- decode tokens/s with cold-session spill;
- p50/p95/p99 latency for KV block prefetch;
- correctness after checkpoint/restore at fixed-token replay.
