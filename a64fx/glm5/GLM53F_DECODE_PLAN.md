# GLM-5.3F A64FX 12-node decode-first plan

## Active work: 512K, non-MTP, 30 tok/s (2026-09-13)

### Continuation: first ~512 positions, 30 tok/s

The next acceptance target is non-MTP scalar decode while the populated
prefix is at most approximately 512 positions. Keep 524,288 allocated
capacity as the deployment contract, but use matched short controls to avoid
conflating this goal with populated-512K performance. The accepted v8 INT8
trajectory and 48-worker HARD-barrier runtime are the correctness/performance
baseline: 25.164 tok/s over 128 positions, or 39.739 ms/token. Reaching
30 tok/s requires at least 6.406 ms/token of repeatable improvement.

Profile and test two exact candidates first:

1. Chain each mHC post-mix with the following mHC pre-mix in one OpenMP team.
   Preserve the existing FP64 post accumulation, reduction scheduling,
   Sinkhorn iterations, and normalization order. The scalar graph currently
   creates hundreds of short parallel regions per token; the chained form
   should remove three team launches at each of 89 adjacent mHC boundaries.
   Keep it opt-in until the complete token/logit stream and matched target
   timing pass.
2. For the first 512 CP positions, mirror only the latent rows in unused tail
   slots of the already committed local cache. Every rank computes the same
   current latent, so sparse MLA can read the replicated prefix without the
   selected-row collective or candidate gather. Continue writing the normal
   owner key/gate/latent slots. To preserve the reference softmax accumulation
   order, compute each completed pool on all ranks and reproduce the exact
   global score ordering locally; the first aggressive identity-order attempt
   diverged at position 8 and was rejected. At position 513, the unchanged
   owner-pool long-context selection resumes. Allocate no additional cache
   bytes and test the boundary at 511/512/513.

Measure mHC, KDA, sparse attention, routed/shared FFN, head, and collective
subphases separately. Accept each candidate only with identical outputs and
at least two matched 512-step runs. The 30 tok/s claim requires a complete
non-MTP target run, not the component tests.

The first end-to-end prefix measurement reaches 27.127 tok/s and leaves MoE
at 12.118 ms/token. Its replicated BF16 router projection is already parallel,
so do not speculate from the aggregate number: first split MoE time into
router/top-k, INT8 gate/up plus activation plus down, and the 4096-float
all-reduce. Optimize the measured dominant phase while preserving each dot
product's accumulation order, then require identical token/logit records and
repeated 512-token timing.

The phase measurement is router **2.106 ms**, INT8 local expert **7.348 ms**,
and all-reduce **2.956 ms** per token. As a bounded accuracy experiment, pack
the replicated router BF16 weights into 16-row INT8/SVE tiles during the
existing load-time conversion and reuse the input quantization for routed
experts. Keep this separately gated because quantized router logits can change
top-k membership; reject it on any token-stream mismatch.

The first 512-step combined run hit a native-MPI segmentation fault at the
position-512 CP pool completion (`2 * 64 * 256 = 32768` floats). The bounded
513-position component validator had passed, but a full-run claim cannot rely
on that transient behavior. Raise the existing uTofu workspace from 20480 to
32768 floats and route this pool exchange through the checked project
collective; revalidate positions 511/512/513 and two full 512-step runs.

The corrected normal-frequency 512-step run passes at 26.313 tok/s; sparse
attention rises to 8.299 ms/token. Before 512 selected rows, CP MLA still uses
one worker per local head (only 5--6 of 48 workers), while the exact latent-
dimension implementation is artificially gated at 512 rows. Add a tunable
exact-parallel threshold and sweep 32/64/128/256 on the 513-position component
test, then use the best exact threshold in end-to-end 512-token runs.

The exact threshold sweep rejected early parallel MLA: cumulative candidate
times for thresholds 32/64/128/256/512 were respectively 460.812, 465.472,
464.776, 463.522, and **442.628 ms** through position 513. Keep 512 as the
default; extra team-wide worksharing overhead dominates before that point.
The load-time INT8 router candidate was also exact but a null: 27.163 tok/s,
router 2.126 ms versus the BF16 router's 27.175 tok/s and 2.106 ms. It remains
diagnostic-only and is not part of the accepted path.

Because the earlier FP32 mHC post path was exact but inside short-run noise,
retest it only as a matched 512-position build on the final prefix path. Do
not promote it unless the longer run gives a repeatable end-to-end gain.

With sparse attention still at 8.299 ms/token, extend the requested load-time
INT8 experiment to the four large sparse MLA FP8 projections (`q_a`, local
`q_b`, `kv_a`, and local output). Pack private anonymous copies in 64-row
SDOT layout, free their FP8 copies, and keep indexer BF16 math unchanged.
Gate the path independently and require the full generated stream to match;
unlike the rejected router conversion, these projections do not make a
discrete top-k routing decision.

The all-four sparse INT8 trial cuts sparse time from 7.17 to 6.66 ms at 128
positions but diverges in generated tokens and reaches only 27.635 tok/s, so
reject it from the accepted path. Next, use the existing exact 16-row view of
the 64-row packed MoE INT8 tiles for gate/up and down scheduling. Typical
gate/up work currently exposes only about 16 coarse tasks to 48 workers;
quarter-tile tasks improve utilization without changing integer dot products.

The 16-row MoE schedule is exact but slower: 26.443 tok/s and 8.413 ms local
expert time versus 27.175 tok/s and 7.348 ms, so reject it. Final throughput
must also be measured with `GLM53F_PROFILE` unset: phase profiling introduces
hundreds of `MPI_Wtime` calls per token and is not a production serving flag.

Final measurements from jobs 51608030 (normal) and 51610526 (boost-eco):

| accepted exact path | positions | tok/s | ms/token |
|---|---:|---:|---:|
| normal, profiled | 128 | 27.179 | 36.794 |
| normal, profiled | 512 | 26.313 | 38.004 |
| boost-eco, profiled, repeat 1 | 512 | 27.192 | 36.776 |
| boost-eco, profiled, repeat 2 | 512 | 27.119 | 36.874 |
| boost-eco, production/no profile | 512 | 26.914 | 37.155 |

The normal run's first 128 token/logit records match the baseline exactly,
and the two boost 512-token records match each other exactly. Memory headroom
is 3.2--3.4 GiB/rank with full 524,288 capacity touched. The measured target
is therefore **not yet 30 tok/s**: the repeatable first-512 result is about
27.1 tok/s, 3.54 ms/token above 30 tok/s. Remaining measured costs are mHC
7.15 ms, attention 16.46 ms (KDA 7.71, sparse 8.75), and FFN 12.34 ms (MoE
11.44). Further work needs a cross-layer persistent OpenMP region and/or fewer
per-layer collectives; the bounded kernel and quantization variants above do
not close the gap.

### Structural continuation after the 27.1 tok/s result

Do not spend the remaining 3.54 ms/token gap on more approximate kernels.
The next implementation changes must preserve the accepted arithmetic and
first-512 token/logit stream while removing orchestration overhead:

1. Make KDA decode one persistent OpenMP region. Its projection/state path
   already uses one team, but input quantization is serial and the output
   projection starts a second team. Move quantization into `single`, keep the
   existing ordered worksharing kernels, and execute the output projection in
   that same team. This removes four team creations per generated token.
2. Make each INT8 MoE layer one persistent region spanning router projection,
   top-k setup, gate/up, activation quantization, and down projection. Router
   top-k and task-prefix construction run in `single` with explicit barriers;
   the numerical kernels and their row ownership stay unchanged. This removes
   one team boundary in each of the 42 MoE layers and does not enable the
   rejected INT8-router or 16-row scheduling experiments.
3. Only after measuring those two changes, introduce team-callable sparse
   projection/MLA helpers. The prefix path currently creates separate teams
   for seven front projections, pool scoring/MLA, and output projection in each
   of 11 sparse layers. Fuse them per layer, but keep uTofu/MPI calls in a
   single thread outside active worksharing regions and retain barriers around
   cache publication. This is the bridge to a token-level persistent executor,
   not permission to change score, softmax, or accumulation order.
4. If per-sublayer persistence still misses 30 tok/s, add a token-level executor
   with one 48-worker team across all 45 layers. Every compute helper must have
   an orphaned/team form; one thread performs each collective while workers wait
   at an explicit barrier. Do not call the current nested `parallel` entry
   points from this executor. A more invasive sequence-parallel stream layout
   is deferred until this exact executor establishes the remaining collective
   floor.

For every stage: first run focused correctness tests, then an exact 128-position
target comparison, then a complete 512-position boost-eco run. Promote only a
repeatable gain with identical token/logit records. Profile OpenMP-region count
and collective time separately; the 30 tok/s acceptance number remains the
unprofiled complete first-512 run at or below 33.333 ms/token.

The combined KDA/MoE per-sublayer persistent-team implementation compiled
with `mpifcc` and passed the real-weight KDA callback (`rel_l2=6.94e-8`, saved
state bit-exact). Its complete 512-position target run was also bit-for-bit
identical to the accepted token/logit record, but regressed to **26.710 tok/s
(37.439 ms/token)**. The same boost allocation measured 26.914 tok/s for the
unprofiled accepted control and 27.119--27.192 tok/s for profiled controls.
Reject per-sublayer persistence: with `FLIB_BARRIER=HARD`, workers spinning
through serial router/top-k and quantization sections plus the added barriers
cost more than recreating the teams. Do not apply the proposed sparse fusion
with this structure. A subsequent persistent executor must span layer
boundaries and schedule independent work while one thread handles serial work,
or it must reduce the number of barriers/collectives; launch-to-barrier
substitution alone is not a viable route to 30 tok/s.

The next bounded structural candidate overlaps useful work across the router
dependency. The shared expert is selected with weight 1.0 for every token and
does not depend on router top-k, so split one OpenMP team between BF16 router
rows and the shared expert's INT8 gate/up tiles. After a barrier/top-k, all
workers process the selected routed experts, activations, and the unchanged
routed-then-shared down accumulation. Tune the router worker count, keep this
path opt-in, and require the same exact 128/512 gates. Unlike the rejected
persistent attempt, this schedule must demonstrate actual router/shared work
overlap rather than making workers spin through serial sections.

The router/shared-expert overlap compiled natively and its 128-position output
was bit-for-bit identical to the accepted record. With 12 router workers it
measured **27.674 tok/s (36.135 ms/token)** versus **27.630 tok/s (36.193
ms/token)** for the matched accepted run. The 0.058 ms/token difference is
noise-sized and far below the remaining target gap: reducing router worker
parallelism cancels the shared gate/up overlap. Reject this candidate and do
not carry its split-team implementation. The evidence from both structural
trials rules out intra-sublayer OpenMP rearrangement as the main route; proceed
only with a cross-layer executor/communication redesign capable of removing
multiple milliseconds per token.

### Long coding-output validation

The first quality run used an 8,050-token C++ stable-sort prompt with
`Reasoning Effort: Max`. It consumed the entire 8,192-token generation limit
inside `<think>`, stopped mid-sentence without EOS, and never emitted a
complete program. At the populated 8K context it measured 18.119 prompt tok/s
and 17.016 decode tok/s; the 27 tok/s headline therefore remains specific to
the first approximately 512 positions.

The repeat used `Reasoning Effort: Low` and a 32,768-token generation ceiling,
with the accepted c7 INT8 weights, INT8 KDA, BF16 latent cache, mHC chaining,
replicated first-512 prefix, flat robust uTofu, and 524,288 capacity unchanged.
On normal-frequency allocation 51607843 it stopped naturally after **27,950
generated tokens** (final special token 154827), rather than reaching the new
ceiling. The 8,049 timed prompt tokens averaged **16.882 tok/s**, decode
averaged **15.630 tok/s**, and the combined 35,999 timed positions averaged
**15.892 tok/s**. Across 436 64-token decode windows the unweighted mean was
15.630 tok/s (median 15.626, range 15.216--15.894); the first and last eight
windows averaged 15.880 and 15.457 tok/s. Minimum observed HBM headroom was
3.13 GiB. This normal-frequency result is not clock-comparable with the prior
boost-eco run, but it shows the expected modest context-length decay.

Low reasoning materially improved completion quality: the response finished
its reasoning, emitted a complete 813-line C++20 implementation, and ended
with a correctness discussion. It nevertheless fails the requested compile
and self-test gate. The generated `AbsThenValue` accepts `int64_t`, but its
self-test applies it to `Pair`, producing a hard comparator type mismatch. It
also uses `std::from_chars` without including `<charconv>`; the available GCC 8
libstdc++ cannot independently validate that path because it lacks the API.
The comparator mismatch and missing direct include are genuine model-output
defects. Therefore classify this run as structurally complete but not
build-correct, and do not claim coding-task acceptance.

A matched **FP8/BF16-weight** quality run was then completed on normal-frequency
allocation 51617019. Because full FP8 weights leave insufficient HBM reserve
at 524,288 cache capacity, this run used a touched 65,536-position BF16 latent
cache, which still covers the fixed 8,050-token input plus the 32,768-token
generation ceiling. Minimum observed HBM headroom was 4.03 GiB. The 8,049
timed prompt positions averaged **14.808 tok/s**, all **32,768 generated
tokens** averaged **13.863 tok/s**, and the combined 40,817 timed positions
averaged **14.039 tok/s**. Across 512 64-token decode windows the unweighted
mean was 13.863 tok/s (median 13.868, range 13.514--14.049); the first and last
eight windows averaged 14.012 and 13.743 tok/s.

FP8 quality was worse on this single greedy coding sample. It diverged from
the INT8 sequence at the first generated token, did not emit EOS, and exhausted
the complete 32,768-token allowance. The text contained coherent design
discussion and eventually began a nominal final answer, but repeatedly emitted
small exploratory code fences and ended mid-statement inside the final C++
block. The longest extracted C++ fence was only 2,850 characters, versus the
INT8 run's complete 813-line block. Compiling the unmodified longest candidate
fails immediately from missing declarations and the truncated class body, so
no self-test executable exists. Classify FP8 as incomplete and not build-correct
on this sample; this is stronger failure than INT8's complete but defective
program, but one greedy sample is not a statistically sufficient general model
quality comparison.

Before interpreting that FP8 result as model behavior, validate the target
implementation layer by layer against an independent scalar reference. Use
the same real FP8/BF16 checkpoint bytes and deterministic input/state, but do
not call the optimized FP8 GEMV, fused mHC, sparse-index, attention, MoE, or
collective kernels from the reference path. For every layer, compare and stop
at the first failing boundary: input RMSNorm; attention/KDA or sparse-index
projection and recurrent/cache state; attention output; attention mHC plus
residual; post-attention RMSNorm; routed/shared FFN before reduction; reduced
FFN output; FFN mHC plus residual; and final hidden state. Report relative L2,
maximum absolute error, finiteness, and argmax/top-k agreement where relevant.
Cover dense layers 0--2, all 11 sparse layers, representative intervening KDA
layers, and then all 45 layers once the first mismatch is understood. The FP8
quality run is quarantined until this scalar gate passes through final logits.

The first independent scalar pass on allocation 51617019 narrows the fault but
does not yet clear the final-logit gate. Real checkpoint mHC pre/post arithmetic
passes at all 90 attention/FFN sites in all 45 layers (worst observed maximum
absolute error below `2.4e-7`). Distributed KDA agrees with a BF16-weight,
FP64-accumulation scalar implementation at layers 0, 4, 20, and 44: final-output
relative L2 is `1.72e-7`--`2.41e-7`; layer-0 q/k/v/decay/beta/core/gated-norm
boundaries are all below `1.50e-7` relative L2.

All three dense FP8 FFNs pass optimized-versus-scalar checks, with output
relative L2 `3.10e-7`--`3.45e-7`. All 11 real sparse-attention layers also pass
their complete single-position projection/index/MLA/output-projection path,
with relative L2 `3.18e-7`--`7.06e-7` and maximum absolute error no larger than
`1.55e-6`. One staged routed expert shard from every MoE layer 3--44 passes the
independent scalar FP8/SwiGLU/down-projection oracle; relative L2 is
`2.88e-7`--`6.30e-7`. These results exclude the optimized arithmetic kernels as
the source of the gross text failure on the tested sites.

Comparing the graph to the official Transformers GLM-5.3-Flash implementation
found one real structural mismatch: `Glm5NextTextIndexer.k_norm` is explicitly
constructed with epsilon `1e-6`, whereas both A64FX sparse paths used `1e-5`.
The runtime and standalone validator now use `1e-6`. Because this can alter
pool ranking after 2,048 tokens, the 8K coding prompt is being rerun before
making any quality claim. Full sequential layer-boundary and final-logit
agreement still remains required.

The corrected FP8 rerun used the same 8,050-token C++ task prompt and generated
512 tokens. It is coherent from the first token, accurately restates the API,
stability, move-only, allocation-fallback, CLI, and self-test requirements, and
begins a sensible TimSort design. This is qualitatively unlike the quarantined
pre-fix output. Prompt throughput was 14.472 tok/s, decode throughput was
13.762 tok/s, and the sampled minimum `MemAvailable` was 3.934 GiB. The short
512-token run is evidence that the indexer epsilon mismatch caused the gross
quality failure, not yet a substitute for the requested 8K-output compile and
self-test gate.

The full corrected 8K-output gate was completed on the same allocation with an
8,050-token prompt and an 8,192-token output allowance. Both formats remain
unqualified on this deliberately oversized coding task:

| Format | Prompt tok/s | Decode tok/s | Combined tok/s | EOS | C++ result |
|---|---:|---:|---:|---|---|
| FP8/BF16 | 14.233 | 13.451 | 13.826 | no | 4 closed C++ fragments; none compile |
| INT8 routed/shared + INT8 KDA | 16.360 | 15.341 | 15.829 | no | 3 closed C++ fragments; none compile |

Both outputs are locally coherent technical reasoning, but spend the entire
allowance repeatedly refining the design and never emit the requested complete
single-file program. FP8 produced 31,280 decoded characters and INT8 produced
31,106. Clang 21 with `-std=c++20 -O2 -Wall -Wextra -Wpedantic` rejected every
language-tagged fence; consequently no `--self-test` executable exists. The
sequences diverge at generated token 9, so INT8 is not an approximation of the
FP8 greedy trajectory. INT8 is 14.9% faster for prompt processing and 14.1%
faster for decode in this matched run, but neither clears the quality gate.

The prompt itself reaches ~8K tokens by repeating 211 acceptance-scenario
variants before the actual task. That is a useful instruction-retention stress
case, but the common failure mode now points above FP8/INT8 arithmetic: either
the checkpoint's reasoning-control/chat-template semantics or generation
policy causes unbounded internal deliberation. Do not resume speed optimization
until a normal non-repeated ~8K coding corpus and the checkpoint's canonical
reasoning-control tokens are tested against the same compile/self-test gate.

### 100+ tok/s prefill workstream

Output quality is provisionally accepted for performance development. The next
target is **at least 100 prompt tokens/s on 12 A64FX nodes**, reported separately
from decode and with exact token/state agreement against the scalar step path.
The current five-position verifier is only a decode-oriented microbatch: its
public API caps chunks at five, sparse attention still advances positions
serially, and layer 42 MoE plus 35 KDA layers repeatedly pay small-batch OpenMP
and collective overhead. It is not an adequate prefill architecture.

Work in this order:

1. Establish 512/2,048/8,192-position profiles for chunk sizes 1--5, separating
   mHC, KDA, sparse attention, dense FFN, MoE, embedding, and head. Preserve a
   chunk-1 token/state oracle and record memory headroom.
2. Raise the internal prefill tile limit independently of speculative verify.
   Start with 8/16/32 positions, make KDA convolution and recurrence causal
   within each tile, and batch BF16/FP8 projections across the token dimension.
3. Replace per-token MoE dispatch with a tile scheduler grouped by
   `(owner, expert)`, reuse each expert shard for all selected positions, and
   perform one reduced output slab per tile. The shared expert is a conventional
   token GEMM and must not be evaluated as repeated GEMV.
4. For sparse layers, batch q/kv/indexer projections while advancing cache and
   causal selection in order. Batch MLA/output projection only among positions
   whose selected sets are already materialized. Do not weaken exact sparse
   selection to reach the throughput target.
5. Fuse adjacent mHC post/pre operations across the position tile and eliminate
   the final vocabulary head during prompt-only ingestion. The last prompt
   position alone needs logits.
6. Accept a change only if a mixed KDA/sparse 128-token comparison preserves all
   hidden states, recurrent state, sparse cache length/content, and final argmax.
   Then measure 512, 8K, and bounded long-context prefill. The 100 tok/s claim
   requires sustained 8K throughput, not a warm 32-token microbenchmark.

Initial measurements on job 51617019 establish the current FP8 baseline.  At
512 positions, chunk 5 without the causal KDA team path measured **23.205
tok/s** (43.290 ms/position).  Enabling `GLM53F_KDA_BATCH_TEAM=1` improved it
to **25.513 tok/s**, with 6.363 ms mHC, 15.395 ms attention, and 17.473 ms FFN
per position.  Prompt tiling is now independent of the five-position
speculative-verification ABI: prompt-only calls accept up to 32 positions,
advance KDA and sparse state causally in four-position arithmetic panels, and
do not allocate snapshots for every prompt position.  The mHC front end was
likewise made panel-safe instead of indexing fixed five-entry stack arrays.

The first chunk-32 run measures **27.149 tok/s** (18.858906 s / 512 tokens):
5.218 ms mHC, 14.026 ms attention, and 17.641 ms FFN per position.  This is
17.0% over the untuned chunk-5 baseline but still far below 100 tok/s.  The
unchanged five-token scalar/batch gate is exact after the split (`probe=92/92`,
all five token IDs and logits identical) and its batch speedup increased to
1.949x in that run.  Outer tiling has therefore exhausted its easy benefit;
FFN is flat and dominant.  The next implementation must group routed work by
expert and evaluate the shared expert across a wider token GEMM so that expert
weights are reused beyond the current four-position kernel panel.

The next FP8 prefill scheduler groups the complete 32-position tile by locally
resident routed expert.  Router projections remain four-token matrices; each
expert consumes up to four gathered positions per weight pass, and results are
stored by original top-k route slot before the final token-major accumulation.
The shared expert is also evaluated in four-token matrices.  This preserves
the favorable 4x4 SVE register shape while eliminating repeated routed weight
passes when an expert is selected by multiple prompt positions.

The exact scheduler reaches **29.484 tok/s** for 512 positions (17.365334 s),
with 5.239 ms mHC, 13.994 ms attention, and 14.794 ms FFN per position.  The
MoE subdivision is 0.789 ms router, 12.470 ms local experts, and 1.586 ms
all-reduce.  This is +8.6% over the first chunk-32 implementation and +27.1%
over the untuned 23.205 tok/s baseline.  A 128-position chunk-4 versus chunk-32
state probe is bit-identical: both produce token 198, logit 5.59104729, hidden
sum -40.713163658045232, and hidden RMS 1.9342837399749639.  An attempted
one-row/eight-token FP8 kernel was rejected: it reduced throughput to 26.817
tok/s and increased FFN to 18.027 ms/position because lower row efficiency
outweighed the additional weight reuse.

Load-time conversion of the routed and shared expert weights to the existing
INT8 layout is now available to the prompt runner with
`GLM53F_PREFILL_INT8=1`. Keep the router projection in BF16: enabling its
separate INT8 conversion faults during the first layer, while the BF16-router
configuration completes normally. At 512 positions and chunk 32, the latter
measures **32.393 tok/s**, with 11.962 ms FFN per position (1.513 ms router,
7.534 ms local expert compute, and 2.789 ms all-reduce). This is +9.9% over
the exact grouped-FP8 result and +39.6% over the original 23.205 tok/s
baseline. Its 64-position state probe produces token 271, logit 5.62643242,
hidden sum -32.185017458163202, and hidden RMS 1.7271280070972912.

Two follow-up INT8 batching experiments are rejected. Converting KDA weights
and batching four SDOT projections reduces 128-position throughput to **24.330
tok/s** and raises attention to 21.054 ms/position. Grouping INT8 routed
experts by expert across the complete tile reaches only **32.003 tok/s** for
64 positions: local expert time rises to 10.979 ms/position versus 7.533 ms
for the existing scalar-INT8 loop, despite an identical state probe. A
32-position all-reduce slab is also unsupported by the current uTofu
registration path (initialization aborts at 131,072 floats); retain the proven
four-position collective payload. The next substantial gain must therefore
come from a true token-matrix INT8/FP8 kernel or attention redesign, not from
larger orchestration tiles alone.

The next attention experiment keeps one OpenMP team alive across an entire
32-position KDA tile. All independent Q/K/V, gate, beta, and output
projections are computed as consecutive four-token arithmetic panels inside
that team; causal convolution and recurrent state updates still advance one
position at a time. The output remains reduced in four-position slabs because
that is the validated uTofu registration limit. Gate this structural path with
`GLM53F_KDA_WIDE_TILE=1 GLM53F_KDA_BATCH_TEAM=1`, require an exact state probe,
and retain it only if attention time improves against the 13.994 ms/position
grouped-FP8 control and the 13.0 ms/position expert-INT8 runs.

Job 51628852 passes that gate. At 128 FP8 positions the same-allocation control
measures **28.566 tok/s** with 13.986 ms/position attention; the wide KDA tile
measures **30.463 tok/s** with 12.028 ms/position attention. The complete probe
is bit-identical in both runs: token 198, logit 5.59104729, hidden sum
-40.713163658045232, and hidden RMS 1.9342837399749639. This is a 6.6%
end-to-end gain and a 14.0% attention reduction.

Combined with BF16-router expert INT8, the 512-position control measures
**31.995 tok/s** (14.225 ms attention, 12.055 ms FFN), while wide KDA reaches
**34.258 tok/s** (12.348 ms attention, 11.837 ms FFN), a same-allocation 7.1%
gain. Both probes are bit-identical: token 198, logit 5.30986309, hidden sum
-31.889146824833006, and hidden RMS 1.8924755182359696. Retain the gated wide
KDA implementation. It removes repeated OpenMP team construction across the
eight four-token panels but deliberately retains causal recurrence and
four-token uTofu reductions.

Next, test a persistent-team mHC prefill front end. The current batch routine
creates a new OpenMP team for every token RMS reduction, every four-token
24-row projection panel, and every token collapse. `GLM53F_MHC_BATCH_TEAM=1`
will execute those same loops, schedules, and per-token reduction order inside
one team per mHC pre call. Accept only a bit-identical 128-position probe and a
measurable reduction from the current 5.0--6.2 ms/position mHC total.

The candidate is bit-identical but rejected. With wide KDA and expert INT8,
the 128-position persistent-team run measures **34.212 tok/s** and 5.797 ms
mHC per position, versus **35.048 tok/s** and 5.040 ms for its same-allocation
control. Barriers between the many low-row phases cost more than the removed
team entries; keep the independently scheduled batch implementation.

The next sparse-layer experiment separates output-projection batching from
collective batching. The earlier `GLM53F_SPARSE_BATCH_OP=1` path reused FP8
output weights across four positions but regressed because its 16,384-float
collective was slower than four 4,096-float reductions. Mode 2 will retain the
four-token FP8 projection and restore one proven-size reduction per position.
Require the same final state probe and a lower attention time before retaining
this mode.

Mode 2 passes. On job 51646260, the 128-position optimized control measures
**34.829 tok/s** with 11.621 ms/position attention; mode 2 reaches **37.102
tok/s** with 9.740 ms/position attention, a same-allocation 6.5% end-to-end
gain and 16.2% attention reduction. Both probes are bit-identical: token 198,
logit 5.63674068, hidden sum -47.380540531128645, and hidden RMS
1.8783314756027496. The 512-position confirmation reaches **36.315 tok/s**
(14.098822 s), with 5.032 ms mHC, 10.544 ms attention, and 11.993 ms FFN per
position; its probe matches the earlier optimized 512 run exactly. Retain
`GLM53F_SPARSE_BATCH_OP=2`; mode 1 remains rejected because it combines the
projection with the slower enlarged collective.

Next, isolate router batching for the expert-INT8 path. The current INT8 batch
falls back to 32 complete scalar MoE calls and spends about 1.5 ms/position in
the BF16 router. `GLM53F_MOE_I8_BATCH_ROUTER=1` will compute router logits in
the already validated four-token BF16 matrix kernel, then retain the exact
per-token top-k, INT8 expert computation, accumulation, and reduction order.
The grouped-FP8 scheduler established that this router kernel can preserve the
state probe; require the same gate here and a lower router profile time.

The batched router is exact and cuts its isolated cost nearly in half. At 128
positions the complete optimized path improves from **37.102 to 37.906 tok/s**
and router time falls from 1.509 to 0.782 ms/position; the state probe is
bit-identical. At 512 positions it reaches **36.387 tok/s**, versus 36.315
tok/s without router batching, with router time 0.781 ms/position. The longer
end-to-end delta is only 0.2% because local expert and all-reduce variation
absorbs most of the saved router time. Retain
`GLM53F_MOE_I8_BATCH_ROUTER=1` as an exact opt-in, but keep **36.315 tok/s** as
the conservative sustained headline until a repeated 512 run shows a stable
overall gain.

The long interrupted `/local` deployment also exposed a development-cost
problem. Rank-image staging now resumes stable per-rank temporary files from
their validated existing size. The same allocation resumed the partial 22.25
GiB transfer and ultimately reported exact-size `OK`/`REUSE` sentinels on all
12 ranks, avoiding a complete restart after each bounded bridge invocation.

The target is one text sequence on 12 A64FX nodes, with 524,288 total context
positions and at least 30 generated tokens/s (33.333 ms/token), without MTP.
Use the existing `~/models/glm53f` FP8 checkpoint and its rank-owned `/local`
images. Preserve a directly comparable FP8 baseline. The historical sustained
controls are 16.587--16.591 tok/s on job 51098702; other allocations reached
18.14 tok/s. These are short-context measurements, not 512K decode results.

Current measured status: **512K capacity passes with BF16 latent caches; the
accepted exact path reaches 27.18 tok/s over 128 positions and 27.12--27.19
tok/s over 512 positions on boost-eco. The target remains unmet, and full-
model decode after a populated 512K prefill is not demonstrated.** See the
2026-09-13 development log and guarded reproduction commands below.

### Memory and context contract

- The 62 checkpoint files total 305.788 GiB. Existing routed/shared/v2-core
  images contain approximately 25.075--25.292 GiB per rank for the target.
  Stage only each rank's files with bounded I/O, fsync, and cache eviction;
  keep token-time weights resident in anonymous HBM. Do not load vision/MTP.
- The old load planner below models BF16 latent/index caches and does not
  describe the integrated runner's FP32 cache. Its theoretical totals must
  not be used to certify a runtime launch.
- FP32-reference context-parallel storage includes 512 latent floats, 128 index-key
  floats, 128 compression-gate floats per owned token, and a 128-float pooled
  key per four tokens. Eleven sparse layers plus selected-row scratch cost
  0.759 GiB/rank at 256K, 1.475 GiB at 512K, and 2.908 GiB at 1M. The 34 KDA
  layers add at most 13.95 MiB/rank of recurrent/convolution state.
- Reserve 3 GiB/rank for runtime/OS in planning and measure minimum
  `MemAvailable` on all ranks during loading, full cache commitment, and
  generation. Abort a benchmark below 2 GiB available. Budget conversion
  scratch and quantization scales explicitly; never retain two complete
  routed-weight representations in HBM.
- Distinguish a 512K capacity/page-touch check, a synthetic populated-cache
  attention benchmark, and actual prefill followed by decode at 512K. Only
  the last establishes full-model throughput at that context. Prompt plus
  generated tokens must fit the declared capacity.

### Execution order and acceptance gates

1. Launch a fresh 12-node, six-hour interactive PJM allocation using
   `a64fx/remote-dev-procedure.md`, separate bridge ports, and a topology
   generated in that allocation. Build with the native Fujitsu MPI wrapper.
   Record job ID, clocks, compiler, thread binding, and commands/results here.
2. Restore rank-owned images to `/local`, validate all 12 manifests/files,
   and reproduce the current FP8 target with matched repeated 128-token
   controls. Record token IDs/logits, load time, per-rank memory, and attention,
   FFN, mHC, head, and collective timings. Commit the complete 512K cache
   before making a memory-fit claim.
3. Profile sparse attention at increasing populated context sizes, including
   512K. Prioritize the growing index scan/top-k and selected-latent exchange;
   use exact selection and unchanged cache semantics for the first changes.
   Profile BF16 KDA projections and mHC separately. Do not infer whole-model
   gains from isolated routed-expert benchmarks.
4. Implement an opt-in INT8 representation converted at weight load from
   the existing `/local` images. Start with bounded real-weight kernel probes
   and retain FP8 as the control. Fold checkpoint FP8 scales into the INT8
   quantizer, specify block/row scales and activation quantization, and pack
   for SVE SDOT. Convert in bounded chunks/in place or release source chunks
   as destination chunks become resident. No full-model BF16/FP32 expansion.
   Consider BF16 projection conversion where its memory/compute savings are
   larger; retain sensitive norms, router decisions, and recurrent state at
   their reference precision initially.
5. Check INT8 arithmetic against an independently dequantized reference and
   report weight/output error against original FP8/BF16. Quantization can
   change logits and greedy tokens: measure that drift on fixed real prompts,
   check finiteness and output quality, and do not label it greedy-exact.
   Adopt a path only when matched repeated full-target runs show a gain.
6. Run bounded prefill benchmarks and full long-context validation as the
   allocation allows. Report prefill throughput/time separately from decode.
   Log measured results and any unachieved gates explicitly; 30 tok/s remains
   a target until a non-MTP full-model run demonstrates it.

Production choices introduced by this work use explicit program arguments,
including context capacity and weight format; environment variables remain
available for existing compatibility and diagnostic controls. Update this
section with accepted changes, rejected experiments, and reproducible results.

### Development log: interactive job 51604112

- The six-hour request did not allocate promptly; the development allocation is
  **12 nodes, two hours**, started 2026-09-13 19:09:44 JST, normal 2000 MHz /
  eco=0. Its first host is `k27-5212c`. The working frontend bridge alias is
  `login1` (not the frontend's `fn01sv01` hostname); reverse endpoint 32426,
  compute loopback 21264. A fresh 12-rank topology was generated in this job.
- Native compiler: `mpifcc -Nclang`, `-O3 -march=armv8.2-a+sve
  -ffp-contract=fast -fopenmp`, LLVM module unloaded and `OPAL_PREFIX` unset.
  One rank/node, 47 OpenMP workers, close/core binding, active wait, existing
  uTofu reduction. Sources are pinned in versioned development snapshots.
- `/local/glm53f-target-{routed,shared,core}-51604112` holds rank-owned images.
  Staging now syncs each 32 MiB before evicting dirty pages. Existing v2 core
  images lacked the 42 replicated MoE routers: strict loading aborted on
  `layers.3.mlp.gate.weight`. `glm53f_core_add_routers.c` appended 84 bounded
  tensor records (~94.55 MiB/rank) to these **node-local copies only**. All 12
  ranks completed; original checkpoint and shared-storage images are unchanged.
- All performance below excludes model load/conversion and uses no MTP. These
  initial controls have only 128 populated positions; a declared 512K capacity
  must not be confused with a populated 512K prefix.

| Initial 128-step control | tok/s | FFN ms/position | attention ms/position |
| --- | ---: | ---: | ---: |
| Unmodified HEAD, strict local weights | 20.852 | 18.727 | 20.324 |
| Candidate FP8 control | 20.914 | 18.607 | 20.171 |
| Row-scaled INT8 routed/shared, 64-row SDOT work units | 23.581 | 13.616 | 20.158 |
| Above + INT8 KDA Q/K/V/output projections | 25.867 | 13.562 | 16.493 |

The profile uses independently reduced phase maxima, so its phase sum can
slightly exceed end-to-end elapsed time. Quantized logits differ from FP8;
KDA quantization also changed the 128-step final greedy token. These are
performance/error experiments, not a quality acceptance or a 30 tok/s result.

Implemented opt-in formats and capacity controls: explicit
`--capacity`, `--weight-format fp8|int8`, `--int8-kda`, `--touch-cache`, and
`--load-only`; bounded in-place FP8-to-row-INT8 conversion with a 256 KiB tile
per worker; independently tested BF16-to-INT8 KDA projection conversion;
parallel/SVE-double CP index scoring; exact top-k heap; dimension-parallel MLA
that retains each lane's original summation order. Finer 16-row SDOT scheduling
was tested and rejected in favor of the original 64-row packed work units.
Sensitive router, gates, norms, and KDA recurrent state retain reference types.

Portable INT8 tests cover zero/random/non-finite/tiny activations, 128--4096
input widths, original block scales, and independently dequantized arithmetic.
A real layer-3 expert gate/up slice had 0.956% relative-L2 output error against
FP8 on the tested vectors. Native index scoring matched all 10,923 scalar
scores bit-for-bit. Full fixed-input quantization and long-context component
tests are required before interpreting these as model-quality guarantees.

#### Long-context and error probes (same allocation)

Matched layer-43 runs use real projection weights and a **synthetic populated
524,288-position cache**, eight measured repetitions after warmup. All four
builds returned output hash `e66b566fc578a83b` and selection hash
`cb961ba27a69495d` (all 4096 outputs / 2048 selected IDs, not just a checksum).

| CP implementation | best ms/layer | mean ms/layer |
| --- | ---: | ---: |
| Unmodified scalar-index CP | 211.541 | 212.419 |
| Parallel, bit-exact SVE-double index scan | 6.656 | 7.467 |
| Above + exact heap selection and dimension-parallel MLA | 5.075 | 5.911 |
| Above + owned-row Allgatherv instead of zero-filled Allreduce | 3.994 | 4.568 |

The native MLA unit test is bit-exact for 5/6 heads and 1, 7, 128, 2048, 2051
selected positions. At 2048 positions, its core fell from ~0.62 ms to
0.19--0.24 ms. The gather requires another ~4 MiB scratch per sparse layer.
Communication and projection costs still prevent treating this as a 30 tok/s
full-model result. Existing CP's serial index scan was unusably slow at 512K.

Teacher-forced error probe, first 256 tokens of `prompt_ids_fp8_full.txt`,
restoring the same initial recurrent state before each format, 16-row scheduling:

| Format | tok/s | argmax agreement with FP8 | hidden relative L2 |
| --- | ---: | ---: | ---: |
| FP8/BF16 reference | 20.658 | 256/256 | 0 |
| INT8 routed/shared | 22.636 | 231/256 | 0.0104948 |
| Above + INT8 KDA projections | 24.185 | 232/256 | 0.0107426 |

All values were finite. These are fixed-input comparisons, not perplexity,
free-running response-quality, or long-context recurrent-stability acceptance.
The matched 16-row vs 64-row full-model A/B is reported below; it confirms
the 64 MiB microbenchmark's preference for 64 rows (101 vs 93 GB/s).

The initial full FP8 512K page-touch run failed the 2 GiB headroom guard.
Another launch was SIGKILLed after loading experts with only ~2 GiB available,
at the router-metadata phase. The loader now finishes router/index parsing
**before** expert residency, trims dead parser allocations, and checks the
2 GiB reserve before/during large uploads. Guarded retries rejected safely
instead of attempting this tight allocation (one rank had 25.456 GiB available
against 23.632 GiB weights + 2 GiB reserve). INT8 KDA conversion is moved ahead
of the expert upload. Native XOS rejected `madvise` on its heap with `EINVAL`;
that reclamation experiment was removed. Per-layer (~5 MiB) INT8 scale arrays
can reuse freed projection chunks, unlike a monolithic ~205 MiB allocation. Core-image
reads also evict the redundant source-file pages. Repeat residency checks are
required; the older blanket 512K-safe statement below is not sufficient.

The guarded FP8 256K-capacity/page-touch + 128-step run passed at 19.585 tok/s,
minimum sampled `MemAvailable` 2.1975 GiB/rank. Added an explicit
`--cache-format bf16` option for **CP latent rows only**, saving ~0.458 GiB/rank
at 512K. Keep index keys, compression gates, pooled keys and KDA state FP32,
and retain FP32 latent storage as the reference default. The BF16 codec,
populated-cache output drift, and full 512K commitment were tested below.

#### 512K capacity passes, but production headroom is insufficient

With `OMP_STACKSIZE=1M`, default XOS paging (no interleave override), strict
node-local core reads, and **BF16 latent / FP32 index caches**, all 12 ranks
completed a full 524,288-position page-touch followed by 128 decode steps:

| Format / runtime | tok/s at 128 populated positions | minimum sampled available GiB/rank |
| --- | ---: | ---: |
| FP8/BF16 weights, 47 workers | 19.713 | 2.0214 |
| INT8 MoE + KDA, 16-row work units, 47 workers | 22.918 | 2.0732 |
| Same, 48 workers + `FLIB_BARRIER=HARD`, `OMP_PROC_BIND=false` | 23.868 | 2.0825 |

These certify **512K capacity, not throughput after a 512K prefill**. The
available-memory margin above the 2 GiB guard is only 22--84 MiB; keep the
guard enabled and do not add concurrent contexts/background memory users.
The full FP32-cache launch still rejected safely. The interleave/demand-paging
experiment was SIGKILLed and is **not** a recommended launch configuration.

BF16 latent-cache codec and native widening tests pass. At the synthetic 512K
layer probe, selected IDs were unchanged and output relative L2 versus FP32
was `0.000259062` (0.0259%), maximum absolute error `1.98e-5`. This does not
guarantee unchanged greedy trajectories: even the short BF16-cache full-model
run ended at a different token than the FP32-cache control.

The matched 256-token teacher-forced **64-row** INT8 control measured 23.525
tok/s for MoE and **25.778 tok/s** with KDA, against 20.863 FP8. Its agreement
counts and hidden-state errors exactly matched the 16-row probe above. Keep
**64-row SDOT in production**; retain 16-row and row-major kernels only as
unit/microbenchmark controls. The smaller work units were 6.2% slower overall.

The old 512K-cache footprint estimate is now 1.017 GiB/rank with BF16 latent
storage, plus ~0.044 GiB/rank of lazy gather/MLA scratch. FP32 latent storage
still costs 1.475 GiB/rank before that scratch. These values do not include
all model/runtime allocations; the measured all-rank guard remains decisive.

#### Real 8K prompt, with the full 512K cache committed

The v7 (16-row INT8, 48 workers / HARD barrier) run ingested a real
8192-token chat prompt: repeated technical notes followed by a request for
five points about memory and numerical correctness. It generated **175 tokens
through EOS**. Sequential prefill (8191 positions; the final prompt position
produces the first generated token) measured **16.593 tok/s**, approximately
493.6 seconds, and generation measured **15.667 tok/s**. Minimum sampled
all-rank `MemAvailable` was **2.024902 GiB**. No MTP or synthetic cache
injection was used in this run.

The decode profile was attention 41.154, FFN 14.512, mHC 8.185, head 1.186,
embedding 0.092 ms/position (independent phase maxima). Attention dominates
once selection reaches 2048 latent rows. This is a much more relevant
long-context warning than the ~26 tok/s short replicated-cache result.
It is still an **8K populated prefix**, not a 512K full-model benchmark.
The decoded response contained five coherent, relevant points; this single
repetitive prompt is only a smoke test, not long-range retrieval/quality proof.
The final 64-row build is validated separately below.

The final v8 native build passed INT8/BF16 codec, all 10,923 index scores,
exact top-k, and 5/6-head MLA unit checks. Its **64-row INT8 + KDA, BF16
latent, 48-worker HARD** 512K-capacity/page-touch + 128-step control measured
**25.164 tok/s** (39.739 ms/token), minimum sampled available **2.048096 GiB**.
The complete token/logit stream matches the v7 16-row control. This remains
a short-populated-context measurement despite its 512K committed capacity.

The final synthetic 512K BF16-cache layer probe measured 3.930 ms best /
4.434 ms mean; output and selected-ID hashes match v7. Mean MPI time was
0.834 ms selected-row exchange, 0.450 ms candidate gathering, and 0.179 ms
pool exchange. This probe appends a pool-completing position every repetition;
normal decoding only performs pool exchange once per four positions.

The final **v8 64-row** repeat also generated the same **175/175 token IDs
through EOS** on the 8192-token prompt. It measured **16.965 tok/s prefill**
(8191 sequential positions, approximately 482.8 seconds) and **15.938 tok/s
decode**. Decode phase maxima: attention 41.199, FFN 13.289, mHC 8.010,
head 1.186, embedding 0.086 ms/position. The 64-row change improves FFN time,
but the full 8K decode gain over v7 is only 1.7% because attention dominates.

Minimum sampled all-rank available memory was **2.001587 GiB**, just
**1.625 MiB above the 2 GiB guard**. Keep this result as a capacity/stability
smoke test, **not a robust production memory budget**. The changing minimum
between otherwise matched launches shows that background/system footprint
can decide whether this configuration is admitted. Do not lower the guard
or add another context. Freeing raw index-key/gate history safely is now a
higher deployment priority than the marginal short-context throughput gain.

The synthetic **1M populated-cache, single sparse layer** probe passes at
4.453 ms best / 5.009 ms mean, selected exchange 0.877 ms and candidate
gather 0.590 ms. Output hash `eeb1733b198bf059`, selected-ID hash
`dad140e96111cb5d`. This low-memory component check does not establish that
the full 1M-capacity model is resident.

The subsequent full-model **1M capacity admission test was rejected safely**
by the loader before the large expert upload: rank 8 had **24.820 GiB**
available versus **25.632 GiB** required (23.632 GiB routed payload + 2 GiB
reserve); ranks 6 and 10 also rejected. Thus neither FP32 512K nor BF16-latent
1M is admitted by the current guarded configuration on this allocation.
Do not replace this result with the older theoretical 1M-fit claim below.

Reproducibility fingerprints (SHA-256 of the whitespace-separated ID files):

- `prompt8192.ids`: `c5e3fff6dded3b41feee43b6cf6f0c129410d2d5a10acc31b90acbea414c1f86`
- `long-v8-output.ids`: `2e1afa667d8272080753fc9bf636eaa457d9b27ec272cb09e62751e8234d909a`

`cmp` confirms v7/v8 generated IDs match; `diff` of all 128 short-run token
and logit records is empty. All committed implementation sources match the
v8 snapshot used for the final native build. The last test completed before
21:03:46 JST; interactive job 51604112's scheduled expiry is 21:09:44 JST.

#### Reproduce the guarded configuration

Use `a64fx/remote-dev-procedure.md` for the loopback-only bridge. From the
frontend, keep the interactive launcher attached in tmux; choose an unused
reverse port (32426 was used for this job):

```bash
REMOTE=login1 FRONTEND_SSH_TARGET=login1 \
FRONTEND_PORT=32426 REMOTE_PORT=32426 \
NODES=12 ELAPSE=02:00:00 WAIT_TIME=600 \
  a64fx/tools/bash-over-http/run_bash_http_interactive.sh
```

Inside that allocation, from the repository root (run staging only when no
model process is resident):

```bash
module unload LLVM/llvmorg-21.1.0 2>/dev/null || true
unset OPAL_PREFIX
job=${PJM_JOBID:?}
export TMPDIR=/local/glm53f-dev-build-$job
mkdir -p "$TMPDIR"
export PJM_PROC_BY_NODE=1 PJM_MPI_PROC=12
GLM53F_MPICC=mpifcc bash a64fx/glm5/build_glm53f_integrated_12n.sh
cd a64fx/glm5
for kind in routed shared core; do
    source_dir=$HOME/models/glm53f/a64fx_ep12_v1/$kind
    format=model
    if [ "$kind" = core ]; then
        source_dir=$HOME/models/glm53f/a64fx_ep12_v2_core
        format=core
    fi
    mpiexec -np 12 sh -c '
        rank=${PMIX_RANK:-${PJM_MPI_RANK:-${OMPI_COMM_WORLD_RANK:-0}}}
        exec ./glm53f_core_stage "$1" "$2" "$rank" "$3"
    ' sh "$source_dir" "/local/glm53f-target-$kind-$job" "$format"
done
mpiexec -np 12 sh -c '
    rank=${PMIX_RANK:-${PJM_MPI_RANK:-${OMPI_COMM_WORLD_RANK:-0}}}
    exec ./glm53f_core_add_routers "$1" "$2" "$rank"
' sh "$HOME/models/glm53f" "/local/glm53f-target-core-$job"
```

The v2 core files are at the **top level** of `a64fx_ep12_v2_core`, not
its incomplete `core/` subdirectory. The older all-in-one staging wrapper
assumes a different directory layout; the explicit commands above match this
checkpoint. Re-staging a router-augmented core restores the source image,
so always run the router augmentation **after** staging. No shared checkpoint
or shared core image is altered.

Then, still inside `a64fx/glm5`:

```bash
repo=$(pwd)/../..
logdir=$repo/tmp/glm53f-run-$job
mkdir -p "$logdir"
export OMP_NUM_THREADS=48 OMP_DYNAMIC=false OMP_WAIT_POLICY=active
export OMP_STACKSIZE=1M OMP_PROC_BIND=false OMP_PLACES=cores FLIB_BARRIER=HARD
export GLM53F_UTOFU=1 GLM53F_PROFILE=1
export GLM53F_REPACK_DIR=/local/glm53f-target-core-$job GLM53F_REPACK_REQUIRE=1
(cd "$logdir" && mpiexec -np 12 "$repo/a64fx/utofu-tests/tofu_topo_helper")
export TOFU_TOPO_PATH=$logdir/tofu_topo.txt
mpiexec -np 12 -of-proc "$logdir/capacity-512k" \
    ./glm53f_target_decode_12n "$HOME/models/glm53f" \
    "/local/glm53f-target-routed-$job" "/local/glm53f-target-shared-$job" \
    1 128 --capacity 524288 --touch-cache \
    --weight-format int8 --int8-kda --cache-format bf16
# For actual generation, replace "1 128" with:
# --generate /absolute/path/prompt.ids /absolute/path/output.ids 256
# Prompt length + 256 must not exceed 524288.
```

Keep default native XOS paging; do not add a NUMA-interleave or demand-paging
override. INT8 conversion is from the anonymous copy loaded from `/local`,
not a second model file: routed/shared FP8 bytes are repacked in place with
~205.45 MiB row scales/rank and at most 12 MiB conversion scratch. KDA
Q/K/V/output BF16 weights are converted one projection at a time before
expert residency. Router, gate, normalization, and recurrent-state precision
are unchanged. No token-time file reads or MTP are used. Construction still
parses checkpoint metadata headers from shared storage; strict repack loading
ensures the actual target weight payload comes from the node-local images.

Diagnostic controls: `test_glm53f_int8`, `test_glm53f_index_score`,
`test_glm53f_sparse_math`; 12-rank `test_glm53f_sparse_cp MODEL 524288 8
fp32|bf16`; and `test_glm53f_quant_model_12n MODEL ROUTED SHARED PROMPT_IDS
256`. The last restores the same initial state between three weight formats;
it is not a perplexity test. Runtime memory samples are all-rank minima every
32 positions and at termination, not a continuous measurement of every
transient allocation.
On native XOS, `/proc` process RSS did not account for the large resident
heap (rank 8 reported only ~27 MiB RSS with its model loaded); use all-rank
`MemAvailable` plus explicit cache commitment, not RSS, for this deployment.

Portable GCC INT8/index tests also pass with `-O2 -fopenmp -Wall -Wextra
-Wpedantic`, and the build script passes `bash -n`. An optional host
ASan/UBSan link could not run because the frontend lacks its sanitizer runtime
libraries; it is not recorded as a sanitizer pass. The native full build
retains existing warnings from shared experimental headers.

#### Remaining work toward 30 tok/s and 512K+

1. **30 tok/s at populated 512K remains unachieved.** The budget is 33.333
   ms/token; even the measured 8K prefix is substantially slower. Profile
   selected-row communication separately from projection, selection, and
   MLA work. Next candidate: transport already-BF16 latent rows as BF16 and
   widen after exchange, halving that payload without another quantization
   step. Persistent or topology-aware exchange needs matched full-model
   validation; a faster isolated collective is not sufficient.
2. Reduce mHC and remaining projection/team-launch costs only behind exact
   controls. The new 64-row INT8 path improves the expert/KDA budget, but
   short-context gains cannot remove the populated-cache communication cost.
3. **512K+ is not certified; the guarded 1M launch rejected.** Going from 512K to 1M adds
   approximately 0.974 GiB/rank with the current BF16-latent cache, exceeding
   the measured margin above the 2 GiB guard. A promising exact memory
   change is replacing historical raw index keys/compression gates with a
   bounded recent-token ring once pooled keys are finalized: up to 0.458
   GiB/rank saved at 512K, 0.917 GiB at 1M. Define and test rollback/snapshot
   behavior before implementing that representation; do not simply discard
   history used by existing APIs. More margin is needed for robust operation.
4. Actual 512K prefill, long-range retrieval/quality, and sustained decode
   remain required. Sequential prefill at the measured 8K average alone
   extrapolates to about **8.8 hours** for 512K (not a prediction: index work
   grows with context). Use a bounded, memory-budgeted batched prefill path
   and a longer allocation; do not claim the two-hour interactive job
   performed that validation. INT8 batch callbacks currently use scalar
   fallback, so they are not an optimized prefill implementation.

All results/scripts for this development allocation are in
`tmp/glm53f-512k-dev/results-51604112/`; immutable source snapshots are
`candidate-v2` through `candidate-v8`. These are local development artifacts,
not files required by the production build.

## Measured anchors (job 51040571)

Historical experiments follow. In particular, the older theoretical memory
planner and synthetic cache figures below do **not** supersede this session's
measured, guarded residency results.

- Full 34-layer/64-head KDA recurrent update, 136 MiB replicated test state:
  **1.483 ms/token** best at 24 threads. The production head-TP layout owns only
  5--6 heads/rank. The measured six-head/rank shape is **0.047 ms best /
  0.049 ms mean** at 48 threads, confirming KDA state arithmetic is not the
  primary limiter.
- Production uTofu all-reduce, 12 ranks, 4096 f32 values: **105.9 us**.
- 78 back-to-back reductions: **5.02 ms/token**. Robust modes 0/1/2 are equal;
  changing completion mode is not a useful optimization.
- Physical 1M CP cache allocation: 12/12 ranks pass with 1.28 GiB/rank committed.
- Conservative checkpoint planner (2 GiB scratch + 1 GiB OS reserve) fits all
  requested contexts: **28.986 GiB/rank at 256K** (3.014 GiB headroom),
  **29.273 GiB at 512K** (2.727 GiB), and **29.845 GiB at 1M** (2.155 GiB).
- Real staged F8_E4M3 quarter-expert (1024x4096 gate/up + 4096x512 down),
  anonymous resident weights: **0.438 ms/task mean** at 24 threads with the
  exact gather-free decoder, versus 0.477 ms/task with the LUT-gather decoder.
  A four-task/four-CMG kernel reduces the measured batch critical path to
  **1.13 ms mean / 0.424 ms best** while the background full-model stager is
  active; the distributed steady-state result below supersedes this isolated
  estimate.
- Full 42-layer, 12-rank four-way expert decode with 23.631 GiB anonymous
  weights/rank and one real 4096-float MPI combine/layer: **47.39 tok/s** over
  200 tokens (**21.104 ms/token**). Compute is 12.615 ms/token; combine plus
  arrival wait is 8.617 ms/token. An unloaded MPI baseline is 73.2 us/call,
  or 3.075 ms/token, leaving **5.56 ms/token of rank-arrival skew**. Resident
  MemAvailable is 6.0--6.45 GiB/rank.
- Aligned 12-way routed-only decode reaches **55.778 tok/s** over 200 tokens:
  17.928 ms/token wall, 14.114 ms compute, and 0.803 ms arrival skew. This is
  17.7% faster than the original four-way routed-only result.
- The checkpoint shared expert is 2048-wide (not the early synthetic 171-wide
  assumption). Block-TP across 12 ranks gives 128/256-wide shards with 63.0 /
  126.0 MiB weights/rank. The real 42-layer persistent-team stream costs
  **2.833 ms/token** for a narrow shard and **4.234 ms/token** for a wide shard.
  Add its partial hidden output to the routed partial before the existing MLP
  combine; it does not require another collective.
- Adding the otherwise-unloaded attention hidden-vector reduction to the matched
  aligned routed+shared run gives **45.788 tok/s (21.840 ms/token)** over 200
  tokens. Expert/shared compute is 15.758 ms/token, the MLP combine is 4.906 ms,
  and the added attention combine is **2.209 ms**. The measured 84-call bare-wire
  contribution is 6.122 ms/token and total arrival overhead is only 0.993 ms.
  This communication-inclusive number is the current decode ceiling; it still
  excludes attention projection GEMVs, KDA/DSA math, router, norms, mHC, and the
  final vocabulary projection.
- The checkpoint contains no E4M3 NaN payloads (a 0.1 GiB staged sample also
  measured 0.00077% zeros and 0.01071% subnormals). Removing the redundant NaN
  compares/select from the exact inner-loop decoder reduces expert/shared
  compute from 15.758 to **14.49--14.57 ms/token**. Two matched 200-token runs
  deliver **48.513 and 47.023 tok/s**, both with checksum `1.41924829e-05`.
  The stager now rejects `0x7f/0xff`, making this a checked payload contract.
  The INT8 SDOT alternative is not a single-token win: 4096x4096 SDOT measured
  434 Gop/s versus 688 Gop/s for W8A16. Its 64-token register-blocked kernel is
  2.37x faster, so retain it as a speculative/batched verification candidate.

## Decode decomposition

Use one process per A64FX node and all 12 ranks as the expert group.

- Routed experts: split every expert four ways over its intermediate dimension.
  Part `p` belongs to `(e % 12 + p*3) % 12`; each rank holds 96 quarter-experts.
  A synthetic occupancy simulation plus exact-shape GEMVs reduced the expected
  slowest-rank expert critical path from ~0.47 to ~0.29 ms/layer versus whole
  experts. An exhaustive search of all translated four-owner offset sets confirms
  `{0,3,6,9}` has the lowest top-8 slowest-rank occupancy: mean 4.061 tasks,
  p95/p99 6/6. Execute local hits concurrently as independent CMG teams; the
  existing MLP hidden-vector sum combines the partial outputs.
- Quantization-aligned 12-way slicing is the decode default. Partition
  the 16 FP8 intermediate block rows, not 2048 raw elements: each expert has
  eight 128-wide and four 256-wide rank shards. This preserves the compressed
  128x128 scale grid, gives every rank all eight routed tasks, and reduces the
  simulated slowest-rank work from 16.244 to **12.061 block rows/layer**
  (p95/p99 14/14). With the real shared shard fused into the same MLP combine,
  **12-way delivers 50.776 tok/s (19.694 ms/token)** versus **44.466 tok/s
  (22.489 ms/token)** for the matched four-way control, a 14.2% gain. Compute
  is 15.763 vs 14.096 ms/token, but arrival skew falls from 5.881 to 1.781 ms.
  Both layouts produce the identical benchmark checksum `1.41924829e-05`.
- KDA layers: partition 64 heads as balanced contiguous ranges (5 or 6/rank).
  Q/K/V, gates, convolution channels, and recurrent state follow head ownership.
  `o_proj` is column-parallel; one hidden-vector sum completes attention.
- MLP: router logits are replicated. Routed and shared expert outputs are local
  partial hidden vectors; one sum completes the MLP.
- mHC is local because its four hidden streams are replicated immediately after
  each attention/MLP reduction.
- Sparse layers, short context: head-TP with replicated latent/index cache. This
  is the fastest decode mode and remains active only while its explicit HBM cap fits.
- Sparse layers, long context: transition to block-CP latent/index cache. Gather
  the small query representation, select/score rank-owned blocks, and merge online
  softmax statistics. This adds communication but keeps 256K--1M memory bounded.

The short-context target is two hidden-vector reductions/layer: 90 reductions for
45 layers plus the head argmax. The measured bare-wire floor is about **9.5 ms/token**.
Real decode must profile rank arrival skew; historical GLM-5.2 real-weight runs saw
effective collective latency around 0.66 ms when expert work was imbalanced.

## Optimization order

1. Decode-only rank-owned stager; do not stage vision or unused MTP tensors.
2. BF16 KDA projection GEMVs and FP8 routed/shared-expert GEMVs. Keep checkpoint
   scales compressed at one F32 value per 128x128 block; use the exact SVE
   bit-decode path rather than LUT gathers.
3. Short-context head-TP forward with exactly two reductions/layer.
4. Expert scheduling sorted by local hit count to reduce rank skew.
5. Continuous batch after single-stream correctness; decode FP8 weights once per
   active expert and reuse them across tokens.
6. Add the long-context CP transition after the fast path is stable.

Every benchmark must report kernel time, collective time, arrival-wait time,
weight bytes/rank, and end-to-end tokens/s. A tight-loop collective number alone
is not an end-to-end communication claim.

## 100 tok/s assessment

Strict single-token decode cannot reach 100 tok/s with the current FP8 graph:
84 measured hidden-vector reductions already cost 6.12 ms/token, and the
routed/shared path alone costs about 14.5 ms/token. Perfectly overlapping those
two terms still caps this partial graph near 69 tok/s before attention
projections, router/norm/mHC, and the vocabulary head. A credible 100+ delivered
token/s target therefore requires multi-token/speculative verification (where
the measured INT8 register-blocked kernel amortizes activation quantization),
or a lower-bit expert representation plus fewer/overlapped collectives. It is
not a scheduler-only target.

### Real layer-45 MTP/speculative probe

Layer 45 is a complete independent sparse-attention+MoE draft block with 288
routed experts, a shared expert, router, `eh_proj [4096,8192]`, and the shared
vocabulary head. It is not merely an auxiliary logits head. The layer-selectable
stager and distributed benchmark measured its real 12-way routed/shared expert
path over 5,000 drafts:

- 0.564 GiB weights/rank;
- **0.524 ms/draft** wall;
- 0.369 ms expert/shared compute;
- 0.138 ms MLP combine and 0.049 ms unloaded attention combine;
- 1,908 partial drafts/s.

This was a lower bound on draft cost at the time of the partial probe. The
token-correct integrated runner described below supersedes it; sparse attention,
`eh_proj`, normalization, vocabulary projection, cache update, greedy sampling,
target verification, and rollback are now connected.

Cold-HBM A64FX measurements put MXFP4 expert throughput at 102--155 GB/s versus
148--174 GB/s for FP8 magic. Accounting for half-sized MXFP4 weights gives about
a **1.45x**, not 2x, effective expert speedup. `glm53f_spec_ceiling.py` combines
that result with measured routed/shared, wire, and partial MTP costs. It makes
deliberately impossible-best assumptions: all shared weights are read once per
verification batch, one collective sequence serves the whole batch, and every
unimplemented graph operation costs zero. Even then:

- K=1: 84.2 tok/s at impossible-perfect alpha=1.0;
- K=2: 95.6 tok/s at alpha=1.0;
- K=3: 102.5 tok/s only at alpha=1.0, but 88.2 tok/s at alpha=0.9;
- K=3 at alpha=0.8: 75.7 tok/s.

Therefore 100+ single-stream delivered tok/s is rejected for this 12-node
FP8/MXFP4+MTP design unless real chained acceptance is effectively perfect and
the omitted graph work is somehow free. The practical 100+ route is continuous
batching across independent requests; MTP may still improve latency/throughput,
but must first be evaluated in a token-correct full forward runner.

### MTP numerical stability and quality gate

Two independent 10,000-step, 12-rank layer-45 runs completed without NaNs,
OOM, or collective failures. Both produced the bit-identical checksum
`-0.000212714513`; expert compute was 0.362 ms/draft in both runs and wall
throughput was 2,005.5 / 1,956.7 partial drafts/s. Rank-0 MemAvailable stayed
near 29.05 GiB. The stager's full-payload scan rejects E4M3 `0x7f/0xff`, and all
12 routed/shared stages completed that contract.

`glm53f_mtp_expert_check.c` validates a real staged rank-0 layer-45 expert shard
against the scalar FP8 reference. The 48-thread SVE path is finite and passes at
`max_abs=4.002e-11`, `rel_l2=3.618e-7` for the full gate/up/SiLU/down operation.

This establishes payload and expert-kernel numerical stability. It remains a
useful isolated check, but the integrated measurements below are the quality
gate for speculative decode.

### Token-correct integrated target/MTP runner (job 51077354)

The 12-rank runner now connects the real embedding, 45 target layers, 12-way
KDA and sparse-attention projections/state, mHC, routed and shared experts,
normalization, shared vocabulary head, and the independent layer-45 MTP graph.
Verification snapshots contain every KDA recurrent/convolution state and every
sparse-attention cache length. Rejection restores the selected target snapshot;
the committed MTP suffix is replayed from exact target hidden states.

Correctness and stability evidence:

- scalar versus five-position batched KDA output has relative L2
  `6.93943709e-08`; all captured recurrent and convolution snapshots are
  bit-exact;
- a sustained 128-token target run produces the same token trajectory through
  final token `271` before and after the decode optimizations;
- G1 after a 128-token warmup repeatedly produces the identical 16-cycle
  acceptance pattern, `10/16` accepted drafts (`alpha=0.625000`), 42 delivered
  tokens, and final token `40591`;
- target plus MTP weights and capacity-177 caches leave 3.70 GiB
  `MemAvailable` on rank 0, with no NaNs, OOM, or collective failure.

The validated KDA changes fuse Q/K/V projection launches, parallelize the three
depthwise convolution channel sets, and parallelize independent local-head
normalization/decay and gated RMSNorm. On the controlled 128-token target test,
latency improved from 68.500 to **62.413 ms/token**, or **14.599 to 16.022
tok/s** (+9.75%), with the token trajectory unchanged. The best matching G1
run is **11.936 delivered tok/s** with unchanged acceptance and final token.

The optimized scalar target profile is 25.696 ms attention, 22.579 ms FFN,
12.596 ms mHC, 1.178 ms vocabulary head, and 0.373 ms embedding per position
(component maxima are reduced independently across ranks and need not sum to
the end-to-end maximum). This makes attention/FFN the next optimization targets;
MTP remains latency-negative at the measured alpha because target verification
dominates.

Latest 12-node Fugaku reruns (job 51086028) measured 17.21 tok/s unprofiled
(58.107 ms/token) and 16.72 tok/s with profiling enabled. A cross-token routed
expert scheduler restored exact logits (92/92 probes) but reduced batch speedup
to 1.25x and MTP-3 to 8.86 tok/s; it is retained only as infrastructure. The
performance path uses one full OpenMP team per token and measures MTP-3 at 9.63
tok/s, alpha 0.4167, PASS. `OMP_PROC_BIND=spread` was tested at 15.62 tok/s and
is slower than the default close binding.

Eliminating empty remainder OpenMP regions in the KDA `mv`/`mv3` projection
helpers is exact and improves the 16-token profiled target run from 59.812 to
58.657 ms/token (16.719 to 17.048 tok/s); attention falls from 25.761 to
24.689 ms/token.

The same build improves the 8-cycle MTP-3 run from 9.634 to **9.683 delivered
tok/s** (alpha 0.4167, 26 delivered tokens, PASS); target/verify phases are
66.022/247.587 ms per cycle.

Routed verifier scratch (`up`, activation, and per-route output) is now
persistent per MoE context instead of allocated per layer/cycle. Exact batch
verification remains 92/92 PASS, and MTP-3 rises to **9.728 delivered tok/s**
(alpha 0.4167; verify 245.824 ms/cycle).

Runtime context allocation tests establish **256K as the minimum-safe target
and 512K as the preferred maximum** for the current 32 GiB/rank layout. The
planner's theoretical 1M estimate above does not satisfy the runtime 2 GiB
headroom guard once the complete integrated graph and working buffers are
resident, so 1M is not a supported launch configuration in this implementation.

FP8 8-row matvec prefetching (next input vector plus two weight rows) is
arithmetic-neutral and produced the best sustained result so far: 16-token
target decode **17.654 tok/s** (56.643 ms/token, final token 432, PASS), with
attention 23.394 ms and FFN 21.061 ms. MTP-3 improved to **9.933 delivered
tok/s**, alpha 0.4167, 26/26 committed tokens, PASS; target/verify phases are
64.298/240.950 ms per cycle.

With the same build, MTP-2 is the better throughput/quality point: **11.904
delivered tok/s**, alpha 0.625 (10/16 draft matches), target/verify phases
64.498/190.441 ms per cycle, final token 3669, PASS. Keep `drafts=2` for the
current best speculative-decode setting; the runner still accepts explicit
draft counts for experiments.

On the dedicated 12-node allocation, `OMP_WAIT_POLICY=active` improves MTP-2
slightly to **11.948 delivered tok/s** (target 64.265 ms, verify 189.649 ms per
cycle; alpha 0.625, final token 3669, PASS) versus 11.904 tok/s with the default
wait policy. Use active waiting when CPU isolation is guaranteed.

A longer 16-cycle MTP-2 run confirms stability: **12.337 delivered tok/s**,
23/32 draft matches (alpha 0.71875), final token 16320, PASS. Mean cycle phases
are target 66.202 ms, draft 9.580 ms, verify 192.990 ms, and rebase 10.219 ms.

For sustained deployment, the 16-cycle MTP-1 run is faster: **12.749 delivered
tok/s**, 10/16 matches (alpha 0.625), final token 40591, PASS. Its phases are
target 65.354 ms, draft 5.486 ms, verify 128.563 ms, and rebase 6.867 ms.
Use the runner's default `drafts=1` for maximum delivered throughput; MTP-2 is
useful when measuring multi-token verification scaling.

A 32-token sustained scalar run with the same two-row/c+64 prefetch and
48-thread close binding reaches **18.094 tok/s** (55.268 ms/token, final token
25, PASS). Profile maxima are mHC 9.698 ms, attention 23.990 ms, FFN 21.018 ms,
and head 1.185 ms per token.

The next 12-node rerun is job **51094320** (6-hour interactive allocation,
ports offset by +10). Its target routed stage is being rebuilt under
`/local/glm53f-target-routed-51094320`; the MPI stage is deliberately allowed
to finish before launching decode benchmarks. Two arithmetic-neutral MLA
overhead reductions are in the current tree: remove the unused `vacc` buffer
and keep the per-call query/logit scratch on the stack (`6f596b70`, `b6907db6`).
They must be rechecked against greedy-exact output and the 16/32-token target
profiles after staging.

The first foreground staging attempt was interrupted by the bash-over-HTTP
30-minute request limit (SIGINT at 16.4 GiB, with no rank status files). The
replacement stage is detached under the same allocation so MPI can run to
completion independently of bridge request lifetime. The integrated build
script now passes explicit `-I.` and `-I../../common` paths for Fujitsu
compiler local-header lookup; the corrected build completed successfully.

Fresh job-51094320 validation completed the target stage on all 12 ranks. The
target batch gate remains exact (`probe=92/92 PASS`, matching logits) and reports
5-token batch speedup 1.356x. A 32-token profiled scalar decode reaches
**18.140 tok/s** (55.128 ms/token, final token 25, PASS); profile is embed 0.287,
mHC 10.113, attention 24.216, FFN 20.785, and head 1.168 ms/token. This is
effectively unchanged from the prior 18.094 tok/s result, so attention and mHC
remain the primary optimization targets. MTP staging has been launched next
under the same 12-node allocation; its quality/performance result is pending.

The follow-on MTP stage completed all 24 rank checks. Standard 16-cycle
speculative validation is exact and stable (`accepted=10/16`, alpha 0.625,
final token 40591, PASS), delivering **12.803 tok/s**. Mean phases are target
64.863 ms, draft 5.532 ms, verify 128.106 ms, and rebase 6.898 ms per cycle.
This is a small improvement over the previous 12.749 tok/s baseline; verify
plus target latency still prevents the 30+ tok/s target, so further work should
focus on attention/KDA and verification batching rather than draft quality.

The same allocation's 2-draft comparison is also exact and stable: **23/32
accepted (alpha 0.71875), final token 16320, PASS**, at **12.482 tok/s**. Its
mean phases are target 65.567 ms, draft 9.468 ms, verify 190.468 ms, and rebase
10.251 ms. Thus 2 drafts improve agreement but reduce delivered throughput;
retain MTP-1 as the deployment default until verification batching is optimized.

A controlled 36-thread scalar rerun on job 51094320 is slower: **17.003 tok/s**
(58.813 ms/token, greedy PASS) versus 48-thread close binding at 18.140
tok/s. Its profile shifts FFN to 26.923 ms/token
(versus 20.785 ms at 48 threads), confirming 48 threads/close binding as the
current target configuration.

A detailed 12-node KDA layer-44 run (`GLM53F_KDA_DETAIL=1`) is exact and
stable. The slowest rank takes 1.266 ms/layer: 0.518 ms local graph, 0.066 ms
output projection, and **0.934 ms all-reduce** (rank variation is 0.30--0.52 ms
for the local graph). This identifies collective latency/overlap as the next
KDA optimization target; further standalone projection micro-optimizations are
unlikely to move end-to-end decode materially.

An optional `GLM53F_FAST_MATH=1` build was tested on job 51094320. The build
completed, but the 12-node 32-token target run failed the lockstep stability
gate: after about six minutes one rank remained active without synchronized
token output (the baseline completes in about 3.5 minutes). The isolated run
was terminated and fast-math is rejected for deployment; the default strict
floating-point build remains required.

The real-weight sparse layer-43 benchmark is also exact (`BIT_EXACT PASS`):
1.437 ms/layer at 128 cached tokens, comprising 0.460 ms indexer/front,
0.221 ms MLA, 0.105 ms output projection, and **0.800 ms all-reduce**. Thus
the sparse-attention side independently confirms that communication, rather
than MLA arithmetic, is the limiting component.

An opt-in `GLM53F_SPLIT_AR=1` prototype (MPI reduce-scatter plus allgatherv)
was tested on real KDA weights. It remains numerically exact (`finite=YES
PASS`) and trims the nominal reduction portion from 0.934 to 0.928 ms, but
two-collective overhead raises total layer time from 1.266 to 1.288 ms. It is
therefore rejected for deployment and left disabled by default.

The allocation-specific 12-rank ToFu topology was regenerated before testing
the existing uTofu collective path.  With `GLM53F_UTOFU=1`, the real-weight
layer-44 callback is exact (`rel_l2=0`, `state=BIT_EXACT`, `PASS`); five-token
batch latency is 1.861 ms and the measured all-reduce portion is 0.252 ms,
versus 0.934 ms through MPI on the same layer.  The standalone ToFu diagnostic
measures a 35.18 us warm 16 KiB reduction floor (12 ranks), confirming that the
earlier initialization failures were caused by stale/incomplete topology files,
not by the collective implementation.  An integrated target decode using this
path is running on job 51094320; its greedy-exact and end-to-end timing result
must be recorded before enabling ToFu by default.

Scalar (4096-float) callback measurements further isolate the benefit: ToFu
reduces the layer-44 all-reduce from 1.163 ms (MPI) to 0.271 ms while remaining
bit-exact.  A paired 16-token integrated run is also greedy-exact (`final_token`
432): ToFu measured 15.637 tok/s and MPI 15.581 tok/s.  This small 0.36% delta
is within run variance, so ToFu remains an explicit opt-in pending repeated
long-run measurements; the strict MPI path remains the deployment baseline.

An opt-in 2-D ToFu hierarchy (`GLM53F_UTOFU_2D=1`, two groups of six ranks)
was then tested. It is exact and lowers isolated scalar KDA reduction to
0.180 ms, but the five-token callback reduction is unchanged at 0.248 ms and
the integrated 16-token target is slower at **15.374 tok/s** (final token 432).
The extra row/column synchronization outweighs the microbenchmark win, so the
2-D mode remains disabled and the flat MPI path remains the deployment default.

The existing direct all-to-all ToFu option (`TP_AR_A2A=1`) was also checked.
It is exact (`PASS`) and reduces the isolated scalar KDA reduction to 0.151 ms,
but five-token verification remains 0.255 ms (flat ToFu 0.248 ms).  Because
the full target run is dominated by verification-sized and non-KDA collectives,
this option is retained for scalar experiments only and is not enabled by
default without a paired end-to-end win.

The paired integrated all-to-all target run is exact (`final_token=432`) but
slower at **13.609 tok/s** for 16 tokens, versus 15.637 tok/s for flat ToFu and
15.581 tok/s for MPI. The additional peer puts and cache traffic dominate in
the full graph; `TP_AR_A2A` is therefore rejected for deployment.

ToFu's robustness overhead was isolated with `TP_AR_ROBUST=0`: scalar KDA
all-reduce falls to 0.171 ms (exact), but five-token verification remains
0.269 ms versus 0.248 ms for flat robust-ToFu. A guarded integrated run is in
progress; this mode is only suitable when the allocation is dedicated and
long-run MRQ-overflow stability is demonstrated.

The guarded integrated non-robust run remained exact (`final_token=432`) but
measured only **14.369 tok/s** for 16 tokens, slower than robust flat ToFu
(15.637 tok/s) and MPI (15.581 tok/s). It is rejected for deployment; the
MRQ-drain behavior remains enabled whenever ToFu is selected.

BF16-compressed ToFu reduction (`TP_AR_BF16=1`) also preserved the 16-token
greedy result (`final_token=432 PASS`) but measured **15.122 tok/s**, slower than
FP32 flat ToFu (15.637 tok/s). The layer-44 output statistics differ at the
fourth decimal place, so BF16 reduction is rejected for deployment despite its
lower payload size.

An opt-in in-place MPI reduction (`GLM53F_MPI_INPLACE=1`) is bit-exact and
reduces scalar layer-44 all-reduce from 1.163 to 0.971 ms. Its integrated
16-token target run reaches **15.888 tok/s** (`final_token=432 PASS`) versus
15.581 tok/s for the out-of-place MPI control. The five-token callback path is
slower (0.304 ms reduction), so retain the switch for scalar decode only until
longer mixed scalar/batch validation is complete.

The longer 32-token in-place run remained greedy-exact (`final_token=25 PASS`)
but measured **15.273 tok/s**. This does not beat the established strict
baseline (17.443--18.140 tok/s), so the in-place reduction remains an opt-in
diagnostic rather than a deployment default despite its isolated scalar-layer
benefit.

An MLA head-dimension parallelization experiment (commit `58dee7e5`) preserved
the exact token stream but regressed the 12-node target to **17.624 tok/s**
(56.742 ms/token); attention rose to 25.084 ms/token. The additional OpenMP
regions and working-set effects outweigh the extra parallelism for 5--6 local
heads, so the experiment is reverted and the 18.140 tok/s implementation stays
as the performance baseline.

After restoring the strict build, a repeat 32-token run remained greedy-exact
(`final_token=25`, PASS) but measured **17.443 tok/s** (57.331 ms/token;
attention 25.739 ms, FFN 21.592 ms). The token stream is identical to the
18.140 tok/s run, so this is retained as a run-to-run A64FX variance datapoint,
not a replacement for the established best result.

The optional `GLM53F_NO_MATH_ERRNO=1` build was repeated: it remained
greedy-exact (`final_token=25`, PASS) but measured **17.752 tok/s** (56.330
ms/token), versus 18.191 tok/s on the first run. The spread matches the
observed A64FX run variance, so the flag is not claimed as a reliable gain and
is left disabled by default.

The KDA output-projection row-batching experiment (8-row SVE kernel replacing
4,096 one-row OpenMP iterations) is bit-exact and improves the isolated scalar
callback to 1.003 ms, but the integrated 16-token target measures **15.391
tok/s** (`final_token=432 PASS`). It therefore does not beat the current
baseline and remains experimental.

The opt-in `GLM53F_MHC_FUSED=1` implementation (commit `369d986c`) fuses the
mHC RMS reduction and 24-row BF16 projection into one OpenMP team, removing a
fork/join from each scalar mHC pre-step. The A64FX callback and full target
binaries compile successfully; the callback remains bit-exact, but an
end-to-end decode comparison is pending completion of the fresh allocation's
node-local expert staging. The validated default (`GLM53F_MHC_FUSED=0`) is
unchanged until that gate reports a wall-time improvement.

The first full 12-node comparison is complete: baseline target decode measured
**14.529 tok/s** (`final_token=432 PASS`) and the fused build measured **14.435
tok/s** (`final_token=432 PASS`) for the same 16-token workload. The fused
region is therefore ~0.65% slower in this run and remains opt-in/diagnostic;
the strict default is retained.

The opt-in `GLM53F_MHC_POST_FLOAT=1` post-mix path is bit-exact in both short
and stability runs: 16 tokens measured **14.987 tok/s** (`final_token=432
PASS`) and 32 tokens measured **15.756 tok/s** (`final_token=25 PASS`). The
16-token result is 3.2% above the same-allocation strict 16-token baseline
(14.529 tok/s). A same-allocation strict 32-token control is still pending;
the float path remains opt-in until that control is recorded.

The same-allocation 32-token strict control then measured **15.694 tok/s**
(`final_token=25 PASS`) versus **15.756 tok/s** (`final_token=25 PASS`) for
the float post-mix build, a modest **0.4%** gain. Both streams are exact; the
float path remains opt-in because this delta is close to observed run variance.

A combined `GLM53F_MHC_FUSED=1 GLM53F_MHC_POST_FLOAT=1` build was also
greedy-exact (`final_token=432 PASS`) but measured **14.612 tok/s** for 16
tokens. This is below the post-float-only result (14.987 tok/s), so the fused
pre-step is rejected in combination as well.

A second independent 16-token post-float run on allocation 51098702 measured
**14.930 tok/s** (`final_token=432 PASS`), versus **14.987 tok/s** on the first
run. Both remain exact; the 0.4% spread confirms that the apparent post-float
gain over the same-allocation strict control is within normal A64FX variance.
Keep `GLM53F_MHC_POST_FLOAT=1` opt-in/diagnostic rather than changing the
default build.

The MLA decode scratch allocator was then removed from `mla_one`: each head
now uses fixed worker-stack buffers (query latent, value accumulator, and
TOPK score list) instead of three `a256`/`free` pairs.  The rebuilt integrated
binary remained greedy-exact and measured **15.705 tok/s** for 16 tokens
(`final_token=432 PASS`) and **16.591 tok/s** for a 32-token control
(`final_token=25 PASS`) on allocation 51098702.  These are respectively about
8.1% above the same-allocation strict 16-token control (14.529 tok/s) and 5.7%
above its strict 32-token control (15.694 tok/s).  Keep the stack-scratch path
as the new default; longer-run confirmation is still warranted against the
18 tok/s historical peak because node-to-node variance remains substantial.

A second independent 32-token repeat measured **16.587 tok/s**
(`final_token=25 PASS`), matching the first run's 16.591 tok/s within 0.03%.
The allocation removal is stable across the longer decode workload; attention
and FFN remain the next bottlenecks.

An opt-in four-accumulator SVE rewrite of the MLA score dot was also checked.
It remained greedy-exact (`final_token=432/25 PASS`) but measured **16.017
tok/s** for 16 tokens and **16.460 tok/s** for 32 tokens, versus the
stack-scratch control's 15.705 and 16.591 tok/s.  The short-run uplift did not
hold at 32 tokens, so the multi-accumulator variant is rejected and the
single-accumulator default is retained.

An arithmetic-neutral latent-cache prefetch was tested next.  It remained
greedy-exact and measured **16.674 tok/s** for 32 tokens, with an independent
repeat at **16.610 tok/s** (`final_token=25 PASS` in both), versus the
16.591 tok/s stack-scratch control.  The ~0.1--0.5% spread is within normal
A64FX variance, so the prefetch hint is rejected and the original scan is
restored.

The lightweight real-weight KDA callback check on allocation 51098702 also
completed cleanly for layer 44 (`tokens=5`): batch output and saved state are
bit-exact (`rel_l2=6.94e-8`, PASS) and batch latency is **3.874 ms** versus
**4.997 ms** for five sequential positions (1.29x).  This confirms the KDA
batch kernel is not the source of the full-graph launch stalls.

A repeat against the latest integrated build remains bit-exact (`rel_l2=6.94e-8`,
PASS) and measures **3.593 ms** batch versus **5.151 ms** sequential
(1.434x).  The absolute spread is normal A64FX variation, while both runs
confirm a material KDA batch advantage.

On the renewed 12-node allocation 51098702, the required layer-45 MTP block
was staged successfully (routed and shared rank shards).  The stable
stack-scratch integrated verifier then completed 16 cycles greedy-exact:
`accepted=10/16`, `alpha=0.625`, `delivered=42` tokens at **12.936 tok/s**,
`final_token=40591 PASS`.  This confirms MTP quality/stability after the
staging retry; it is below the earlier 12.7--12.9 tok/s range only within
normal run variance, so no speculative kernel change is justified yet.

The sparse batch verifier was then changed experimentally to keep cache updates
sequential but pack each token's local output projection into one `tokens*H`
all-reduce.  It remained exact (`rel_l2=9.19e-8`, rollback exact), but measured
**5.015 ms** versus **3.310 ms** for four scalar positions (0.660x), because
the larger collective outweighed the call reduction.  The packed path is
rejected and the original per-token reduction is restored.

An opt-in SVE FEXPA softmax for MLA scores was also compiled and tested.  It
preserved greedy output (`final_token=432 PASS`) but measured **15.655 tok/s**
for 16 tokens versus **15.705 tok/s** for the exact `expf` stack-scratch
control.  The approximation is therefore rejected; exact `expf` remains the
default.

The MLA value reconstruction loop was also changed experimentally to keep one
SVE accumulator per latent-dimension chunk instead of reloading/storing the
accumulator for every selected cache entry.  It remained exact and measured
**15.843 tok/s** (16 tokens) and **16.597 tok/s** (32 tokens), versus 15.705 and
16.591 tok/s for the stack-scratch control.  The longer result is effectively
flat (+0.04%), so this rewrite is rejected; the original token-major loop is
restored.

### Persistent EP12 rank images and HBM2 residency

The decode deployment invariant is that every rank's complete weight shard is
resident in anonymous HBM2 before token generation.  mmap and decode-time
streaming from `/local` are not supported performance modes.  The storage path
is deliberately three phase:

1. One-time offline conversion of the 305.8 GiB, 62-file checkpoint into
   `~/models/glm53f/a64fx_ep12_v1/{routed,shared,core}`.  Each directory has one
   contiguous blob and manifest per rank.  The core blob contains only the
   rank-owned attention/KDA, dense, router, embedding, norm, and vocabulary
   slices recorded during target construction.
2. At allocation startup, each rank copies only its own three files from shared
   storage to `/local` with bounded 32 MiB I/O, `fsync`, and
   `POSIX_FADV_DONTNEED`.  A completed offline image is identified by all rank
   status files plus the root `COMPLETE` marker.
3. The target performs sequential reads from `/local` into anonymous aligned
   allocations and drops source cache pages.  All token-time kernels then read
   HBM2 only.  uTofu may redistribute rank-owned pieces after upload when a
   different tensor layout reduces compute communication, but it is never a
   substitute for disk reads during decode.

The canonical commands are `run_glm53f_offline_repack_12n.sh` once and
`run_glm53f_stage_rank_image_12n.sh` for every allocation.  This format trades
the slow one-time source reshuffle for sequential per-node startup traffic:
shared storage (about 300 MB/s/node) to `/local`, followed by `/local` (about
1 GB/s/node) to HBM2.
