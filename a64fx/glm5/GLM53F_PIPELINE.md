# GLM53F PP3×TP4 prototype

The prototype targets prefill on the same 45-layer UD-Q4_K_XL, top8 model
and 12 A64FX nodes. The promoted TP12 configuration remains the production
path. PP3×TP4 is not yet connected to a runnable full model.

## Implemented foundation

`glm53f_parallel.h` supplies argument-only layout configuration, validates
microbatches 512/1024/2048 and contiguous cuts, maps world ranks to stage/TP
ranks and checks two-slot transfer allocation sizes. Defaults are TP12,
1024 positions and cuts 15,30. It does not alter process-wide math switches.

`glm53f_dist` owns an independent TP communicator per stage and a pipeline
communicator per TP lane. It borrows the world communicator. Initialization
collectively rejects invalid or inconsistent configurations and requires the
MPI main thread with at least MPI_THREAD_SERIALIZED. Components borrowing
the context must be freed before the context itself.

`glm53f_pipeline_run` has serialized and ordered two-slot schedules.
Callbacks produce embedding streams on stage0, execute owned layers on each
stage, and consume final streams on stage2. Adjacent stages transfer all four
4096-component FP32 streams lane-to-lane; sequence, shape, offsets, cuts and
payload counts are checked. Receives never overwrite a live send buffer.
Callback collectives must use the stage TP communicator. After communication
starts, failures abort world so another stage cannot wait indefinitely.
The implementation uses the calling controller thread only.

The overlap schedule preposts two receives. On the middle stage, outgoing
completion precedes receive-slot reuse. This preserves the two-slot memory
bound; further overlap requires profiling rather than assuming asynchronous
MPI progress. Final drain and world synchronization are inside the run.
Per-rank profiles expose callback compute, receive wait, send wait, position
count and microbatch count; these are not end-to-end model throughput.

## Validation (2026-10-03)

Host:

```sh
mkdir -p tmp/strata-pipeline-20261002
cc -std=c11 -O2 -Wall -Wextra -Wpedantic -Werror \
  a64fx/glm5/test_glm53f_parallel.c \
  -o tmp/strata-pipeline-20261002/test_parallel
./tmp/strata-pipeline-20261002/test_parallel
```

`GLM53F_PARALLEL_PASS cut_layouts=946 ranks=12`. ASan/UBSan also pass with
LeakSanitizer disabled because the local sandbox uses ptrace.

Native cross-build uses Fujitsu mpifccpx, `-Nclang -O3
-march=armv8.2-a+sve -Wall -Wextra -Werror`, linking
`test_glm53f_pipeline.c glm53f_dist.c glm53f_pipeline.c`.
On PJM52097252 (12 nodes, compact2×3×2, normal2GHz/eco0):

```sh
mpiexec -n 12 ./test_pipeline
mpiexec -n 12 ./test_pipeline --fail-callback
```

The first prints `GLM53F_PIPELINE_PASS layouts=2 batches=3 schedules=2
 tails=7 repeats=2`. It checks stage-local reductions and communicator
rank/size, configuration disagreement, invalid shapes, FIFO/order, exact
payloads, single positions, aligned and partial batches, repeated runs and
8049 positions. Additional 16384-component tests use four complete batches
plus a three-position tail for every PP microbatch size, forcing large-message
send completion and slot reuse. The failure test exits nonzero with
`GLM53F_PIPELINE_FAIL world_rank=5 stage=1 phase=callback`.
Final fixture binary SHA256:
`43fcf3684aa66ed323f5d9b186d0efe4589953b9cb1c230ca52662bb117199cb`.
The shared check build now builds both test programs.

## Routed native shards

`glm53f_pp_routed_stage` is a separate build using shared bounded native
staging code. It accepts the pipeline options, derives owned MoE layers from
the cuts and assigns one of four512-channel parts per expert to each TP rank.
Its `GLM53F_PP_ROUTED_V1` header records layout, world/stage/TP rank, cuts,
layer interval and part count/width. A source metadata stamp covers the input
path and every GGUF shard's size, nanosecond mtime, tensor count and data offset.
The stamp checks metadata for reuse; per-payload FNV1a hashes check the copied
bytes independently. Entries record bytes/hash, source names and row/column/
expert ranges. TP12 retains eight256-channel parts and a separate namespace.
Gate/up scratch follows the widest native format;512-channel Q6_K rows exceed
the old1MiB buffer. Down-column staging remains128-row bounded.

On fresh PJM52106727, `test_glm53f_pp_routed` passes24 exact payload checks
across Q4_K/Q5_K/Q6_K/IQ2_XS/IQ3_XXS/IQ4_XS, all288 ownership maps, byte and
rolling-hash checks, and rank/layer/source-stamp mismatch rejection. Metadata-
only sizing of the real model passes all12 ranks:

| Stage | Routed layers | Bytes per rank | GiB per rank |
| --- | --- | ---: | ---: |
| 0 | 3:15 | 13447987200 | 12.5244 |
| 1 | 15:30 | 16420700160 | 15.2930 |
| 2 | 30:45 | 16500916224 | 15.3677 |

No full routed image has been staged. These sizes exclude attention,
embedding/head, shared/dense weights, caches and workspaces. The checked
`glm53f_memory_budget` ledger rejects overflow and insufficient6GiB headroom;
complete component accounting and resident-loader integration remain pending.
See [machine-readable evidence](strata-pipeline-foundation-20261003.json).

## Dense context and 16-head MLA

`glm53f_dense_ffn_create_dist` borrows the explicit distribution context,
requires an owned dense layer and a PP native image, and partitions the
12288 intermediate channels into four3072-channel slices. It uses world-rank
image filenames and stage-local reductions. The loader checks the PP header,
entry shapes/byte counts and FNV1a hashes before native repacking; reads are
bounded to1MiB and release source cache pages. Legacy constructors retain their
TP12 path.

On PJM52106727, `test_glm53f_dense_dist DIR` passes three cuts (`15,30`, `1,3`,
`1,2`), covering dense layers on all stages. Scalar and batch1–4 callbacks pass
with synthetic zero Q4_K weights; legacy headers, corruption, missing paths
and wrong ownership are rejected. This validates the interface and collectives;
real-model dense math and full PP integration remain pending.

`glm53f_mlb_token_grouped` extends the existing MLA primitive to16 heads,
using four-head groups above the legacy six-head range. Values for all heads
are retained; scratch logits belong to the final group. Sparse-layer helpers
also accept16 heads and group fused projections within the eight-matrix native
limit. Native `test_glm53f_mla_groups` passes640 bit-exact cases per node in
both conservative and production-fast math builds (15360 cases total), covering
heads1–16, selected counts through2052, two selection orders, FP32/FP16 latent
values, stride padding and output/scratch canaries. Full-model sparse qualification
is pending; its PP constructor is not yet connected.

## Remaining model integration

1. Stage-specific native manifests, source/tensor byte coverage and hashes;
   explicit rejection of TP12 images; bounded staging and >=6GiB memory
   headroom before resident loading. Four 512-channel routed expert parts,
   512-channel shared slices and 16 attention heads per TP rank.
2. Explicit distribution contexts in KDA, sparse attention, dense/routed FFN,
   embedding and head constructors; preserve all TP12 wrappers. Audit fixed
   MLA scratch/head limits and group computation by four heads.
3. Owned-layer executor accepting/returning complete four-stream FP32 tensors;
   embedding on stage0 and final normalization/head on stage2. Connect both
   schedules, then one-token sequential pipeline decode and token broadcast.
4. Canonical state export across head/layer ownership. Require identical
   generated IDs, finite fields and per-field rel-L2 <=1e-3 (zero-reference
   fields exactly zero), exact structural metadata, and report routing and
   sparse-selection differences. TP12 bit-exact gates remain unchanged.
5. Qualify 8049 positions/full state/first token, 128-step decode, 1024 stress,
   short128 and repeated32K; serialized versus overlap exact for the same
   cuts/microbatch. Screen three trials and independently confirm five against
   fresh TP12. Include transfer, fill/drain and first readout in timing.
6. Profile microbatches 512/1024/2048 and balance contiguous cuts under memory
   limits. Keep prefill gains as experimental if decode regresses. Overall
   promotion requires >=5% prefill gain and <=2% decode regression plus all
   correctness gates. The 100/2000 goal requires both rates on one qualified
   12-node configuration.

## Owned model and runner (October 3)

The experimental PP runner now binds explicit stage-local contexts in every
component. KDA and sparse attention own sixteen heads; dense layers own3072
channels, routed expert parts512 channels and shared slices512 channels.
Embedding lives on stage0; final normalization and the vocabulary head live
on stage2. Native PP core reads use a context-scoped reader, bypassing the
legacy process-global TP12 repack image. Metadata and payloads are checked
before binding; source stamps describe shard metadata, while payload hashes
cover the actual staged bytes. PP image namespaces reject TP12 headers.

`stage_glm53f_pp_12n.sh` builds owned images directly from GGUF below `/local`
with bounded reads/writeback and source-cache release. It refuses overlapping
native MPI launches. The resident constructor accounts for native/compact
weights, vocabulary transients, repacks/derived panels, replicated sparse KV,
recurrent state, MoE/attention workspaces and pipeline slots. It rejects a
conservative peak leaving less than6GiB and checks headroom after loading.
This inventory is an upper bound; real PP peak measurements remain pending.
Never construct a TP12 and PP model simultaneously on a node.

```sh
# Use the shared build, keeping experimental binaries separate from controls.
GLM53F_BIN_DIR=build/pp-native-v1 bash build_glm53f_integrated_12n.sh check 47 4096
bash stage_glm53f_pp_12n.sh "$GGUF" "$PP_ROOT" build/pp-native-v1 "$STAGE_LOGS" \
  --pipeline-cuts 15,30
mpiexec -n 12 build/pp-native-v1/glm53f_target_decode_12n \
  "$MODEL_METADATA" "$PP_ROOT/routed" "$PP_ROOT/shared" \
  --generate "$PROMPT_IDS" "$OUTPUT_IDS" 129 \
  --parallel-layout pp3-tp4 --pp-image-root "$PP_ROOT" \
  --pipeline-cuts 15,30 --pipeline-microbatch 1024 --pipeline-schedule two-slot \
  --prefill-mode fast --prefill-features 27 --prefill-slab 32 \
  --capacity 16384 --ignore-eos
```

`--pipeline-schedule serialized` uses the same callbacks and ownership for the
reference schedule. PP generation is greedy with FP32 cache; unsupported
conversion/CP settings are rejected. Prefill timing includes the complete
prompt, pipeline transfer/fill/drain and first readout. Generated token1 comes
from that readout, so129 output IDs measure128 subsequent decode transitions.
Each PP stage also reports its maximum compute, receive and send-wait time
to guide contiguous-cut balancing. The TP12 runner additionally reports `GLM53F_TARGET_FULL_PROMPT_TIMING` using
the same full-prompt/first-readout and post-first-token timing boundary; its
historical timing record remains available. Neither new timing record is
comparable directly to the older8048-prefix/128-readout denominator.

`--state-export PREFIX` is diagnostic: it exports every prompt token's four
streams, final streams, per-layer/head KDA state, replicated sparse persistent
fields and selected indices, expert routes for all executed positions, and
post-decode state at `PREFIX.decode`. Files are exclusive, bounded and flushed;
such runs do not qualify throughput. After retrieving all rank files:

```sh
python3 tools/compare_glm53f_fields.py "$TP12_PREFIX" "$PP_PREFIX" \
  --reference-ids "$TP12_IDS" --candidate-ids "$PP_IDS"
python3 tools/compare_glm53f_fields.py "$TP12_PREFIX" "$PP_PREFIX" --phase decode \
  --reference-ids "$TP12_IDS" --candidate-ids "$PP_IDS"
# Require exact same-cut/same-microbatch schedule behavior separately.
python3 tools/compare_glm53f_fields.py "$SERIAL_PREFIX" "$PIPELINE_PREFIX" --bit-exact \
  --reference-ids "$SERIAL_IDS" --candidate-ids "$PIPELINE_IDS"
```

The comparator requires complete canonical head coverage, exact structural
metadata/shapes, finite FP32 fields, per-token means/streams and per-head
recurrent/convolution fields at rel-L2<=1e-3, and exact zero-norm fields.
Generated IDs must match exactly. Route and sparse-selection changes are
reported independently; `--bit-exact` also requires their exact equality.
Both prefill and post-decode captures must pass.

Native PJM52106727 gates now pass: owned executor short128 and8049 positions
at microbatches512/1024/2048 are bit-exact in streams, state and readout, with
minimum headroom9.036316GiB; dense28 format/partition cases, shared28,
KDA16, sparse12 and routed corruption/ownership fixtures pass. Core-context
row/column/hash rejection passes all12 ranks; packed embedding broadcasts
pass eight sizes through2049, owner boundaries/duplicates and signed-zero
bits. The pipeline fixture also passes128 sequential full-width transitions
for both layouts and all three microbatches. Host reader/ASan/UBSan,
40 legacy repack-policy cases and canonical comparison rejection tests pass.
Full real-weight PP load, cross-layout numerical/ID gates, stress/32K and
whole-model performance are still pending. No PP promotion or100/2000 claim.
