# GLM-5.3F single-node validation

This is the bounded diagnostic path for the garbage-output investigation. It
does not create a GGUF from the 328 GiB checkpoint and does not use MPI or
uTofu. The model stays on shared storage; only the current tensor slice and
small diagnostic streams are resident.

## Preflight

On an A64FX allocation, run:

```sh
OUT=/local/glm53f-validation-$PJM_JOBID \
  sh a64fx/glm5/run_glm53f_validation_1n.sh
```

An already-built scalar reference can be passed as the remaining command;
stdout/stderr are captured under `OUT` after the preflight:

```sh
OUT=/local/glm53f-validation-$PJM_JOBID \
  sh a64fx/glm5/run_glm53f_validation_1n.sh ./glm53f_reference_1n \
  --model "$HOME/models/glm53f" --prompt-ids "$HOME/work/gemm/glm53f/tmp/prompt.ids"
```

`glm53f_validation.py manifest` reads only safetensors headers and records the
per-layer byte budget. `router` accepts one line of 288 logits, optionally
followed by 288 selection-bias values. `compare` compares bounded float32
streams without loading either file into memory.

The reference executable should use `common/glm53f_ref.h` for scalar math and
`common/glm53f_safetensors.h` for bounded `pread` slices. Load and release one
layer at a time. For every token, emit boundaries named `attn_norm`,
`kda_or_dsa`, `attn_mhc`, `ffn_norm`, `ffn_router`, `ffn_moe_weighted`,
`ffn_mhc`, and `final_hidden`; emit full values only for the first mismatch and
stats/digests for all other records.

The non-negotiable model contract is: DSA `k_norm` epsilon `1e-6`,
`index_topk=2048`, `kpool=4` with the tail always selected; 288 experts,
top-8, stable lower-index ties, selection bias only for ordering, unbiased
sigmoid weights normalized to routed scale `2.5`, and an unscaled shared
expert. NaNs, changed route IDs, or the first top-1 token change stop the run.

Build llama.cpp CPU-only on A64FX and use its tiny synthetic GLM5NEXT GGUF and
tensor callbacks only as an architecture/graph smoke test. It is not a
substitute for the streamed exact oracle. MTP and vision are out of scope.

## Native llama.cpp/ggml build

The dedicated Ninja driver cuts the build to the CPU backend, shared ggml and
llama libraries, common support, and `llama-simple`:

```sh
LLAMA_SRC="$HOME/work/llama.cpp" \
LLAMA_BUILD=/local/glm53f-llama-a64fx \
  sh a64fx/glm5/build_llama_a64fx_ninja.sh
```

It uses Fugaku `fcc`/`FCC` with `-Nclang`, `armv8.2-a+sve`, and OpenMP. The
LLVM 7 auto-vectorizer workaround is limited to the diagnostic build through
`llama_a64fx_disable_vectorization.h`; explicit ggml SVE kernels remain
available. Compiler temporaries default to the build directory under `/local`
because `/tmp` may not exist in a compute-side shell. The resulting
`bin/llama-simple` and `bin/llama-gguf` are checked with `file` on the A64FX
node. If the llama.cpp checkout contains
`models/ggml-vocab-gemma-4.gguf`, the script also performs a bounded GGUF
metadata read; this exercises the loader without loading model weights. A
GLM53F model is intentionally not required for this module/link smoke test.

The lower-level CPU backend check can be run independently:

```sh
OUT=/local/glm53f-validation-$PJM_JOBID \
  sh a64fx/glm5/run_ggml_a64fx_smoke.sh
```

It builds a small F32 `ggml_mul_mat` graph with FCC, executes it through the
native CPU backend, and compares the result against the column-major scalar
reference (`[4,13,13,31]`).

For the real Q2 GGUF, use the internal metadata-only loader probe
`a64fx/glm5/glm53f_llama_loader_probe.cpp` after the native build. It invokes
`llama_model_loader` with `no_alloc=true` on the first shard, which causes
llama.cpp to discover and validate all four split files, tensor descriptors,
and file bounds without mapping or reading the 101 GiB weight payload. The
GLM-5.3 Flash Q2 contract is `arch=glm5next`, 1,412 GGUF tensors,
108,710,550,904 descriptor bytes, and four split files. This is the safe
llama.cpp loader gate before attempting any streamed layer execution.

## Bounded MoE router and dispatch

Run the real-checkpoint router probe on one A64FX node:

```sh
MODEL=$HOME/models/glm53f \
  OUT=/local/glm53f-validation-$PJM_JOBID \
  sh tmp/glm53f_router_dispatch_1n.sh
```

`a64fx/glm5/glm53f_router_dispatch_probe.c` reads only token 0's embedding
row, layer 3's 288x4096 BF16 router matrix, and its 288-element correction
bias. It computes all router logits, applies sigmoid-plus-bias top-8 ordering,
renormalizes the unbiased sigmoid weights to routed scale 2.5, and verifies the
expert dispatch slots. Job 51741132 passed with eight unique expert IDs and
`weight_sum=2.5`; the selected IDs were `47,286,17,19,223,279,37,25`.

This validates router arithmetic and dispatch bookkeeping with real checkpoint
weights. It intentionally uses the embedding row as the bounded input, so it
does not yet claim a full transformer hidden state or expert FFN output.

The llama.cpp cross-check is implemented by
`a64fx/glm5/glm53f_llama_router_compare.cpp`. It reads the corresponding GGUF
tensors through `llama_model_loader::load_data_range` and llama.cpp's native
embedding dequantizer. The dispatch IDs and ordering matched the safetensors
runner exactly (`47,286,17,19,223,279,37,25`). Comparing the two paths directly
against the BF16 safetensors checkpoint showed a maximum routed-weight
difference of `1.09494e-4`; this is expected because the Q2 GGUF embedding row
is not bit-identical to the BF16 source.

The comparator now also invokes the custom scalar `glm53f_router_topk` routine
on the exact same llama.cpp-dequantized GGUF `logits` and bias arrays. Job
51742986 passed this same-representation test:

```text
LLAMA_GLM53F_ROUTER_LLAMA 47:0.311463296 286:0.313435256 17:0.312795281 19:0.311763763 223:0.312555254 279:0.312539637 37:0.313260198 25:0.312187314
LLAMA_GLM53F_ROUTER_REF   47:0.311463296 286:0.313435256 17:0.312795281 19:0.311763763 223:0.312555254 279:0.312539637 37:0.313260198 25:0.312187314
LLAMA_GLM53F_ROUTER_SAME_GGUF PASS routed_weight_sum=2.5 max_weight_diff=0
```

This proves identical router dispatch and weights between the custom scalar
implementation and llama.cpp at the GGUF representation boundary. It does not
yet prove identical full-model token output: the next bounded comparison must
feed the same controlled hidden vector through one complete layer/FFN path,
including the selected expert weights, without attempting to resident-map the
101 GiB model in 32 GiB HBM.

The comparator supports explicit token IDs and, when none are supplied, runs
the following eight-row prompt-token suite: `0, 1, 2, 42, 1234, 154820,
154822, 154827`. Job 51743543 passed all eight rows. The llama.cpp GGUF
dequantized embedding row was loaded one row at a time, so this remained
within the single-node HBM budget:

```text
LLAMA_GLM53F_ROUTER_TOKEN id=0      ... result=PASS routed_weight_sum=2.5       max_weight_diff=0
LLAMA_GLM53F_ROUTER_TOKEN id=1      ... result=PASS routed_weight_sum=2.5       max_weight_diff=0
LLAMA_GLM53F_ROUTER_TOKEN id=2      ... result=PASS routed_weight_sum=2.5       max_weight_diff=0
LLAMA_GLM53F_ROUTER_TOKEN id=42     ... result=PASS routed_weight_sum=2.49999976 max_weight_diff=0
LLAMA_GLM53F_ROUTER_TOKEN id=1234   ... result=PASS routed_weight_sum=2.5       max_weight_diff=0
LLAMA_GLM53F_ROUTER_TOKEN id=154820 ... result=PASS routed_weight_sum=2.5       max_weight_diff=0
LLAMA_GLM53F_ROUTER_TOKEN id=154822 ... result=PASS routed_weight_sum=2.49999976 max_weight_diff=0
LLAMA_GLM53F_ROUTER_TOKEN id=154827 ... result=PASS routed_weight_sum=2.5       max_weight_diff=0
LLAMA_GLM53F_ROUTER_PROMPT_SUITE PASS tokens=8 max_weight_diff=0
```

## Staged end-to-end validation

The staged validator uses the effective 45-layer trunk (`blk.0` through
`blk.44`). GGUF block 45 is the separate NextN/MTP block and is intentionally
deferred. The exact-comparison lane uses the real Q2 GGUF representation for
both implementations; the native safetensors/12-rank runner remains a
separate precision and performance lane.

Each stage is restartable and exchanges a bounded artifact rather than a live
model allocation:

```text
token IDs -> embedding/mHC streams -> attention -> attention state
          -> router -> selected experts/shared FFN -> mHC streams
          -> final norm -> streamed vocabulary logits -> greedy token
```

Stage payloads are little-endian float32 (or int32 for token/expert IDs). Each
artifact directory contains `manifest.jsonl` and payload files. Manifest
records include implementation, model/prompt identity, layer and token range,
shape, dtype, byte count, FNV-1a integrity hash, RMS/min/max, and finite-value
status. `glm53f_validation.py artifact-compare` compares payloads in bounded
chunks without loading either artifact into memory.

The first implemented producer is the llama.cpp GGUF loader/router harness.
Set `GLM53F_STAGE_OUT=/local/...` to emit embedding rows and router logits,
expert IDs, and weights for the token suite. The harness also runs the
validation-only custom router function and emits
`GLM53F_CUSTOM_STAGE_OUT=/local/...`; the Python comparator checks the llama.cpp
and `custom_adapter` artifact sets tensor-by-tensor. This first cross-lane gate
covers the GGUF-dequantized embedding representation, router logits, exact
route IDs, and bounded route weights. It keeps the production 12-rank
safetensors runner unchanged and never maps the 101 GiB payload. It is not yet
full custom-weight validation: both lanes currently consume the llama.cpp
loader's dequantized GGUF rows, while the custom lane validates the router
algorithm and artifact contract.

Required stage gates are: exact token IDs and route IDs; router weight error at
most `2e-6`; pure norm/mHC boundary relative-L2 at most `2e-5`; quantized
attention/FFN boundary relative-L2 at most `1e-3`; no NaN/Inf; final logits
relative-L2 at most `3e-3`; and exact greedy-token equality for the short and
8K chat fixtures plus 32 generated tokens. KDA/DSA state is serialized with
each attention handoff so a later stage can resume independently.

All runs use direct one-node Fugaku interactive jobs with `pjsub --interact`,
`--mpi proc=1`, `/local` build/working storage, no `/tmp`, no full-model mmap,
and a `MemAvailable` stop guard at 6 GiB.

The first staged cross-lane run completed as job 51744358. The direct job built
llama.cpp with the GLM5NEXT dispatch fix, loaded all four GGUF shards, ran the
eight-token suite, and compared 32 llama.cpp versus `custom_adapter` artifact
records with `--threshold 0`:

```text
LLAMA_GLM53F_ROUTER_PROMPT_SUITE PASS tokens=8 max_weight_diff=0
artifact_compare tensors=32 threshold=0 PASS
LLAMA_MODULE PASS cross_lane_artifacts=embedding,router
```

### Streamed trunk graph sweep

The isolated llama.cpp graph producer is implemented in
`glm53f_stream_graph_probe.cpp`, driven by `tmp/glm53f_stream_payload_1n.sh`.
It loads one `blk.N` tensor slice at a time, executes exactly one GLM5NEXT
trunk layer, writes a 16,384-element F32 hidden artifact, and passes that
artifact to the next process. The maximum measured layer payload is
3,276,747,096 bytes; graph/context overhead is about 155 MiB, so each stage
fits comfortably in the 32 GiB HBM2 node.
The stream script preserves its layer F32 payloads and logs under
`tmp/glm53f-stream-validation-$PJM_JOBID` on the shared filesystem by default;
`GLM53F_STREAM_OUT` may select node-local `/local` storage for a disposable run.

Job 51752389 passed all effective trunk layers 0 through 44:

```text
GLM53F_STREAM_GRAPH layer=0  ... finite=16384 ... PASS
GLM53F_STREAM_GRAPH layer=11 ... finite=16384 ... PASS
GLM53F_STREAM_GRAPH layer=44 ... finite=16384 ... PASS
GLM53F_STREAM_GRAPH PASS layers=45 hidden=16384
GLM53F_STREAM_PAYLOAD_JOB PASS job=51752389
```

The sweep exposed and fixed a llama.cpp allocator interaction specific to this
streaming mode. KV-cache tensors could be created with a prebound CPU buffer
but without data before graph allocation, triggering
`GGML_ASSERT(tensor->buffer == NULL)`. The stream-safe path clears that stale
prebinding for data-less, non-view tensors before normal allocation. The fix
was verified at layer 3 and then by the full 45-layer sweep. At that stage this
was only a trunk graph/residency gate; the terminal norm and vocabulary
projection are covered by the terminal results below.

The terminal stage is now implemented as a stream sentinel after the trunk. It
loads only `output_norm.weight` and `output.weight`, performs the final
four-stream mean/RMS norm and vocabulary projection, emits a 154,880-element
logit artifact, and reports greedy argmax. The independent comparator
dequantizes the same GGUF output rows and computes the projection outside the
llama.cpp graph.

Job 51757667 passed the standalone terminal stage on the bounded synthetic
input with token 3980 in both paths. After changing the independent projection
to use the GGML type-specific activation quantizer and `vec_dot` contract, job
51758955 repeated the standalone terminal stage and produced:

```text
GLM53F_STREAM_CUSTOM_FINAL token=3980 llama_token=3980 rel_l2=1.37824947e-07 exact=YES PASS
GLM53F_STREAM_GRAPH layer=45 hidden=154880 finite=154880 ... token=3980 PASS
GLM53F_STREAM_PAYLOAD_JOB PASS job=51758955
```

The complete real token-1 chained run was job 51759261: layers 0 through 44
consumed one another's F32 artifacts, and the terminal stage produced this
exact-token gate:

```text
GLM53F_STREAM_GRAPH PASS layers=45 hidden=16384
GLM53F_STREAM_CUSTOM_FINAL token=5556 llama_token=5556 rel_l2=1.39751539e-07 exact=YES PASS
GLM53F_STREAM_GRAPH layer=45 hidden=154880 finite=154880 ... token=5556 PASS
GLM53F_STREAM_PAYLOAD_JOB PASS job=51759261
```

The final projection relative-L2 is now `1.40e-7`, below the `3e-3` gate, and
the greedy token is identical for both the standalone and real chained inputs.
The stream runner stores every intermediate as an F32 artifact and reloads
only the next layer's weights, so no process requires the 101 GiB GGUF to be
resident. The production custom runner was then checked on the same one-token
input. Job 51762585 ran `run_glm53f_q2_12n.sh` with its default input token `1`
and `GLM53F_TARGET_STEPS=1`; it reported `final_token=5556`. This exactly
matches the llama.cpp token-1 chain result from job 51759261:

```text
llama.cpp: GLM53F_STREAM_CUSTOM_FINAL token=5556 llama_token=5556 ... exact=YES PASS
custom runner: GLM53F_TARGET_TOKEN step=0 token=5556 ...
custom runner: GLM53F_TARGET_DECODE_12N steps=1 generated=1 ... final_token=5556 PASS
```

Thus the required final-token equality is checked between the production
runner and llama.cpp, in addition to the independent terminal projection
comparison. Full per-layer custom artifact equivalence remains a useful
stronger diagnostic, while the custom runner's individual attention/FFN/router
lanes are covered by the bounded gates below.

The production-vs-llama.cpp equality check must use the same checkpoint
representation. A token-1234 production rerun (job 51768151) reported
`final_token=198`, whereas the Q2 GGUF streamed chain above reported
`final_token=29656`. The production runner reads the BF16 safetensors
`model.language_model.embed_tokens.weight`; llama.cpp reads and dequantizes
the Q2 GGUF `token_embd.weight`. The bounded row comparator in
`glm53f_embedding_compare.cpp` measured the first divergence directly:

```text
GLM53F_EMBED_COMPARE token=1234 ... rel_l2=0.0358545767 max_abs=0.0012024045 DIFFERENT
```

This is a real cross-representation mismatch, not an inference nondeterminism
or HBM failure. The token-1 equality (`5556`) is therefore only a result for
that input and cannot be generalized. The next production equality gate must
either feed the custom runner the same Q2 GGUF embedding/output rows or use a
llama.cpp reference built from the same BF16 safetensors weights; the current
token-1234 gate is explicitly FAIL until that alignment is implemented.

Job 51783542 relaunched the bounded comparator for four inputs in one process,
confirming that the difference is systematic rather than token-specific:

```text
token=1      rel_l2=0.0364667797 DIFFERENT
token=42     rel_l2=0.03640382   DIFFERENT
token=1234   rel_l2=0.0358545767 DIFFERENT
token=154822 rel_l2=0.0352607695 DIFFERENT
```

To exercise a second real embedding/input variant, job 51760204 used an
explicit in-job `GLM53F_STREAM_TOKEN=42` wrapper. Its layer-0 hash differed
from token 1 (`ed0ec7e7ff5637b3` versus `e79ad917bd8c1afd`), all 45 trunk
stages passed, and the terminal equality gate also passed:

```text
GLM53F_STREAM_GRAPH PASS layers=45 hidden=16384
GLM53F_STREAM_CUSTOM_FINAL token=154822 llama_token=154822 rel_l2=1.4397501e-07 exact=YES PASS
GLM53F_STREAM_GRAPH layer=45 hidden=154880 finite=154880 ... token=154822 PASS
GLM53F_STREAM_PAYLOAD_JOB PASS job=51760204
```

This run retained all 45 stage payloads and the final payload in
`tmp/glm53f-stream-validation-51760204/graph/manifest.jsonl` (46 records).

The bounded representation diagnostic was extended to inspect the terminal
boundary as well: GGUF `output_norm.weight` versus safetensors
`model.language_model.norm.weight`, and selected GGUF `output.weight` rows
versus safetensors `lm_head.weight` rows. The source change is in
`glm53f_embedding_compare.cpp`; it still reads only one row at a time and
fits the HBM2 limit. The repo-local CMake wrapper is
`a64fx/glm5/compare/CMakeLists.txt`; use `BUILD_SHARED_LIBS=OFF` on Fugaku
because the shared build exposes an unresolved `common_params_sampling`
destructor in this image. Job 51787360 built and ran the static target on one
A64FX node and produced:

```text
GLM53F_NORM_COMPARE token=-1 ... rel_l2=0 max_abs=0 MATCH
token=1      embed rel_l2=0.0364667797  head rel_l2=0.073792577  DIFFERENT
token=42     embed rel_l2=0.03640382    head rel_l2=0.0713734826 DIFFERENT
token=1234   embed rel_l2=0.0358545767  head rel_l2=0.0720562673 DIFFERENT
token=154822 embed rel_l2=0.0352607695  head rel_l2=0.0717283678 DIFFERENT
```

This proves the final norm is representation-identical, while both the input
embedding and vocabulary head are independently quantization-divergent. Full
final-token equality therefore requires aligning those two boundary weights
and the remaining safetensors/GGUF trunk weights; it cannot be fixed by the
embedding loader alone.

The production runner's Q2 embedding stage was then validated directly on 12
A64FX nodes. Two earlier attempts did not reach the embedding gate: job
51787488 placed the twelve 8,252,817,408-byte routed images in one node's
87-GiB `/local` allocation and stopped with `ENOSPC`, while job 51789689 used a
smaller effective quota and stopped with `EDQUOT`. Job 51796626 instead used
`rscgrp=int,node=12`, `proc=12`, and one rank per distinct host. Every host
reported 91,226,112 KiB available under `/local`; all twelve routed, Q5_K
embedding, shared-expert, and compact-core stage sentinels passed. Each
embedding shard contained 12,906 or 12,907 rows and occupied 211,451,904 or
211,468,288 bytes.

The job ran three Q2-embedding inputs and a same-allocation BF16-embedding
control, one decode step each:

| Input | Production embedding | Production token/logit | Streamed Q2 token | Result |
| ---: | --- | --- | ---: | --- |
| 1 | Q2 GGUF | `5556 / 8.51486969` | 5556 | exact |
| 42 | Q2 GGUF | `154822 / 11.8808479` | 154822 | exact |
| 1234 | Q2 GGUF | `198 / 10.6972504` | 29656 | mismatch |
| 1234 | BF16 safetensors | `198 / 11.487524` | n/a | control |

The changed token-1234 logit proves that the production runner consumed the
staged Q2 row even though its greedy ID remained 198. Aligning only the input
embedding is therefore insufficient for the failing input; the production
trunk and vocabulary head still use safetensors-derived weights. All four
executions were finite and reported `PASS`. The first load took 101.73 seconds,
with 18.84 GiB minimum `MemAvailable` after residency and 18.66 GiB during the
decode. The run used HEAD `56f89642` plus the then-current dirty worktree,
captured in the log directory with diff SHA-256
`468f970446d0380a2f74d0c7ef46aeb4fc45ce597fa70de5a6810f16029d5a13`.

Job 51766725 repeated the complete streamed chain with
`GLM53F_STREAM_TOKEN=1234`, providing a third real embedding/input variant.
All 45 trunk layers again produced finite 16,384-element artifacts. The
layer-0 hash was `279d0a0b3f24bcdc`, distinct from the token-1 and token-42
runs, and the terminal comparison matched exactly:

```text
GLM53F_STREAM_GRAPH PASS layers=45 hidden=16384
GLM53F_STREAM_CUSTOM_FINAL token=29656 llama_token=29656 rel_l2=1.76643041e-07 exact=YES PASS
GLM53F_STREAM_GRAPH layer=45 hidden=154880 finite=154880 ... token=29656 PASS
GLM53F_STREAM_PAYLOAD_JOB PASS job=51766725
```

This extends the full streamed final-token gate across three distinct real
input token variants: `1 -> 5556`, `42 -> 154822`, and `1234 -> 29656`.

The layer-3 attention/FFN increment uses the same artifact contract and adds
real custom-vs-llama.cpp intermediate values before the final-token gate. The
router adapter remains an algorithm/serialization gate, not a full-model
equivalence claim.

The layer-0 dense FFN increment is now implemented in
`glm53f_llama_dense_stage.cpp`. It loads only `blk.0.ffn_gate`, `ffn_up`, and
`ffn_down`, builds llama.cpp's quantized `mul_mat -> swiglu_clamp -> mul_mat`
graph, and runs an independent custom assembly using the same GGML
weight-specific activation quantization and `vec_dot` contract. It emits the
input, SWIGLU activation, and output vectors for both lanes. Job 51745612
passed three token inputs:

```text
DENSE_STAGE_TOKEN id=0    rel_l2=0 max_abs=0 result=PASS
DENSE_STAGE_TOKEN id=42   rel_l2=0 max_abs=0 result=PASS
DENSE_STAGE_TOKEN id=1234 rel_l2=0 max_abs=0 result=PASS
DENSE_STAGE PASS tokens=3 max_rel_l2=0 max_abs=0
artifact_compare tensors=9 threshold=0.001 PASS
```

This establishes exact quantized FFN arithmetic and artifact handoff for a
real model block while remaining comfortably within one A64FX HBM2 node.

The recurrent KDA stage is now implemented in
`glm53f_llama_kda_stage.cpp`. It deliberately uses a bounded three-token,
two-head, 128-wide state, so the complete attention output and recurrent state
fit in HBM2 independently of the full checkpoint. The llama.cpp lane runs
`ggml_gated_delta_net`; the custom lane runs `glm53f_kda_step_vec_streamed`
with the same initial state, q/k normalization, gate, beta, and state layout.
State is serialized in canonical `[head][row][column]` order at the artifact
boundary and transposed only at the GGML operator boundary.

Job 51746477 passed the KDA gate and both artifact records:

```text
KDA_STAGE attn_rel_l2=3.11955503e-7 state_rel_l2=6.56265577e-8
  max_abs=3.49245965e-10 max_state_abs=3.7252903e-9 PASS
artifact_compare tensors=2 threshold=2e-5 PASS
LLAMA_MODULE PASS cross_lane_artifacts=embedding,router,ffn_layer0,kda_layer3
```

The DSA/indexer primitive is also implemented in
`glm53f_llama_indexer_stage.cpp`. It uses the bounded GGML
`ggml_lightning_indexer` contract (`q`, `k`, per-token head weights, and an
FP16 mask) and compares its score matrix with an independent scalar
ReLU-weighted dot-product implementation. Job 51746766 passed it with
relative L2 `7.14e-8` and maximum absolute error `3.73e-9`; the serialized
artifact comparison passed at `1e-5`.

```text
INDEXER_STAGE rel_l2=7.1443207e-8 max_abs=3.7252903e-9 PASS
artifact_compare tensors=1 threshold=1e-5 PASS
```

The bounded tail gate is now implemented in `glm53f_llama_tail_stage.cpp`.
It chains absorbed-MQA/MLA attention, routed SwiGLU experts, a GGML stream
repeat for the mHC handoff, RMS norm, vocabulary projection, and greedy argmax.
The custom lane consumes the preceding stage's MoE output before norm/logits;
both lanes emit MLA, MoE, mHC, norm, and logits artifacts. The final bounded
fixture uses 32 token positions and checks the complete greedy-token array,
not only its first element.

Job 51748094 passed every artifact comparison in the assembled bounded chain:

```text
TAIL_STAGE mla_rel_l2=1.26876606e-7 moe_rel_l2=1.4270703e-6
  mhc_rel_l2=4.51467769e-8 norm_rel_l2=8.05269422e-8
  logits_rel_l2=7.99190082e-8 token_count=32 exact_tokens=YES PASS
artifact_compare tensors=5 threshold=0.003 PASS
LLAMA_MODULE PASS cross_lane_artifacts=embedding,router,ffn_layer0,dsa_indexer,kda_layer3,mla,moe,mhc,norm,logits,greedy_token
```

This is an exact llama.cpp/custom primitive-chain gate with bounded synthetic
tail tensors; it does not claim a full 45-layer GLM5.3 prompt output. The
full-checkpoint final-token gate remains separate because the 101 GiB Q2 GGUF
cannot be resident on one 32 GiB HBM2 node, and the earlier real-checkpoint
generation attempt stalled in LLIO while faulting the mmap payload. Closing
that last gate requires a streamed/distributed full-model runner or a suitably
sharded residency strategy; it must not be inferred from this bounded tail
argmax result.

The tail executable now accepts `GLM53F_TAIL_VARIANT` and changes every input
stream and tail-weight salt while preserving the same llama.cpp/custom graph.
`tmp/glm53f_llama_module_1n.sh` runs variants `0,1,2` by default, writes each
variant to separate artifact directories, and requires an exact greedy-token
match for all 32 positions in every variant. This makes the tail gate a
multi-input numerical check rather than a single favorable synthetic vector.

Job 51748414 reran the complete one-node module with the LLIO localtmp
reservation and passed all existing gates. The three tail variants produced
exact 32/32 llama.cpp/custom token arrays:

```text
variant=0 mla=1.26876606e-7 moe=1.4270703e-6 norm=8.05269422e-8 logits=7.99190082e-8 exact=YES
variant=1 mla=1.11374712e-7 moe=1.49195509e-6 norm=7.63418768e-8 logits=7.45680332e-8 exact=YES
variant=2 mla=1.07590162e-7 moe=1.46827203e-6 norm=7.69297683e-8 logits=7.37476322e-8 exact=YES
```

The loader, embedding/router suite, layer-0 quantized FFN, DSA indexer, KDA
state, artifact comparisons, and all three tail variants passed in that job.

The same job also passed the no-allocation stream-model probe. llama.cpp built
the complete 46-layer GLM5NEXT tensor metadata without allocating model data,
then enumerated the real GGUF layer ranges. The largest resident layer is
layer 11 at `3,276,747,096` bytes across 31 tensors; all other trunk layers
are at or below `2,809,399,384` bytes. This confirms that loading one real
quantized layer at a time is compatible with the 32-GiB HBM2 constraint and is
the basis for the remaining streamed graph/final-token implementation.

```text
GLM53F_STREAM_MODEL arch=glm5next layers=46 n_embd=4096 vocab=154880
GLM53F_STREAM_MODEL PASS max_layer_tensors=31 max_layer_bytes=3276747096
```

Job 51749156 advanced this to a real payload transaction for the largest
layer. It read all 31 layer-11 tensors from the split GGUF via
`load_data_range`, kept 3,276,747,096 bytes resident, computed an integrity
hash, checked all F32 values for finiteness, and freed the payload before exit:

```text
GLM53F_STREAM_PAYLOAD layer=11 tensors=31 bytes=3276747096
  f32_finite=1322070 fnv=b48c9e9218ffa108 PASS
GLM53F_STREAM_PAYLOAD_JOB PASS job=51749156
```

This validates the storage/residency primitive used by the final streamed
runner. By itself it is not a token-output check; the graph and component
gates below provide the hidden-stream, KDA/DSA, and terminal comparisons.

Job 51765982 independently reran the committed llama.cpp component module on
one A64FX node after the streamed graph implementation was committed. The
direct `pjsub --interact` allocation used `proc=1`, six-hour `int`, and the
localtmp LLIO reservation. The build and executable smoke tests passed, then
all bounded cross-lane stages passed:

```text
LLAMA_MODULE PASS cross_lane_artifacts=embedding,router,ffn_layer0,dsa_indexer,kda_layer3,mla,moe,mhc,norm,logits,greedy_token
DENSE_STAGE PASS tokens=3 max_rel_l2=0 max_abs=0
INDEXER_STAGE rel_l2=7.1443207e-8 max_abs=3.7252903e-9 PASS
KDA_STAGE attn_rel_l2=3.11955503e-7 state_rel_l2=6.56265577e-8 PASS
variant=0 ... exact_tokens=YES PASS
variant=1 ... exact_tokens=YES PASS
variant=2 ... exact_tokens=YES PASS
```

The embedding and router artifact comparison covered token IDs
`0,1,2,42,1234,154820,154822,154827` with zero observed difference. The
three tail variants each matched all 32 greedy token positions exactly. This
is a fresh multi-input component verification of the llama.cpp module; it does
not change the separate full-checkpoint one-token result above.
