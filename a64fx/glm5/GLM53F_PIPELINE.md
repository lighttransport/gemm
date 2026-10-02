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
