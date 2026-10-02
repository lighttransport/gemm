# GLM-5.3 Flash on 12 A64FX nodes

Start here for the native GLM-5.3 Flash runner. The neighboring GLM-5.2
experiments are a different model and are not part of this workflow.

## Quick start

Run inside an existing **12-node, 12-rank** Fugaku allocation, from the
repository root. No additional pjsub or SSH bridge is needed there.

```bash
# Fresh allocation: build, stage node-local weights, and decode 128 tokens.
bash a64fx/glm5/run_glm53f_12n.sh run

# Reuse this allocation's staged weights and shared binaries.
bash a64fx/glm5/run_glm53f_12n.sh decode 1 128

# Generate from whitespace-separated token IDs.
bash a64fx/glm5/run_glm53f_12n.sh generate prompt.ids output.ids 256 \
  --prefill-chunk 512 --prefill-mode fast --prefill-features 27 \
  --prefill-slab 32 --prefill-collective mpi-rsag --decode-window 128
python3 a64fx/glm5/glm5_tokenizer.py decode-file output.ids
```

Prefill communication (measured with `bench_glm53f_allreduce_12n.c`, 8 MiB = 512 tokens x 4096 fp32 across 12 nodes):
`mpi-rsag` with 32-token slabs takes 5.7 ms, `tree-packed` 11.5 ms and the decode collective in 4-token calls 17 ms.
The MoE combine reduces the whole 512-token chunk with one `MPI_Allreduce` (3.4 ms); `GLM53F_MOE_AR_SLAB` selects the
call size (0 = the original 4-token decode-collective calls, up to 32 = uTofu-wrapper slabs, above 32 = raw MPI).
Also set `XOS_MMM_L_PAGING_POLICY=demand:demand:demand`; the previous `...:prepage` policy placed weights badly.

The default is **UD-Q4_K_XL with native GGUF-derived weights**. Routed experts
retain Q4_K/Q5_K/Q6_K blocks; dense, KDA, sparse and shared-expert projections
use Q8_0 blocks repacked without loss. Small mHC, norm, convolution, indexer
and router tensors remain in the compact BF16/F32 representation patched
from GGUF. This is not bit-identical to llama.cpp; see
[validation and limitations](GLM53F_VALIDATION.md).

The default files must already exist on shared storage:

| Input | Default |
| --- | --- |
| First GGUF shard | `$HOME/models/glm53f-gguf-all/UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf` |
| Safetensors metadata/tokenizer | `$HOME/models/glm53f` |
| Compact core source | `$GLM53F_MODEL_DIR/a64fx_ep12_v2_core` |
| Compact shared-expert source | `$GLM53F_MODEL_DIR/a64fx_ep12_v1/shared` |

The metadata path is still needed even in native mode. The launcher does not
create the source compact images. Their developer-only preparation is the
[offline repack script](run_glm53f_offline_repack_12n.sh), which requires
completed rank read traces and explicitly configured source/output paths;
it is not an automatic first-run conversion.

## One launcher, separate operations

| Command | Builds | Stages weights | Runs |
| --- | --- | --- | --- |
| `run [TOKEN STEPS OPTIONS...]` | runtime tools | yes/reuse | scalar decode |
| `build [runtime\|check\|all]` | selected set; default check | no | no |
| `stage` | no | yes/reuse | no |
| `decode [TOKEN STEPS OPTIONS...]` | no | no | scalar decode |
| `generate PROMPT_IDS OUTPUT_IDS N [OPTIONS...]` | no | no | prompt + generation |
| `benchmark PROMPT_IDS OUTPUT_IDS [OPTIONS...]` | no | no | resident warmup + snapshot-reset timed trials |
| `executor-check PROMPT_IDS [N=32]` | no | no | complete legacy/persistent stream and state comparison |
| `check` | check tools | no | kernels, components, full-model batch/rollback |

Use `GLM53F_BUILD=0` to skip builds in `run` or `check`. Build before a
standalone `stage`. A successful launch ends with a `SENTINEL`; the default
decode gate is 20 tok/s (`GLM53F_MIN_TOK_S=0` disables the performance gate for
correctness-only checks). A finite-logit PASS is not a semantic-quality score.

Options after decode/generate arguments are passed to the C runner, including
`--capacity`, `--temperature`, `--top-p`, `--seed`, `--prefill-chunk`,
`--prefill-mode`, `--decode-window`, `--touch-cache` and `--ignore-eos`.
The launcher preserves the required `--generate` argument position.

The opt-in persistent executor, grouped verification, lookup speculation,
communication owner and their measurement protocol are described in
[Strata-inspired optimization](GLM53F_STRATA.md). New paths have kernel
validation; full-model performance qualification is still pending.

Experimental exact Q8 switches are `--q8-row-kernel rows4|rows8`,
`--q8-prefill-kernel tile4x4|tile2x8|tile4x4-asm|tile2x8-asm`, and
`--mla-projection-kernel legacy|fused`. Defaults retain rows4, the C 4×4
prefill tile, and separate MLA head projections. Native integrated builds
link the assembly tiles; standalone bridge builds use the C fallback when
that object is absent. See the Strata document for qualification status.

The resident benchmark also supports opt-in `--speculation mtp` with
`--mtp-routed-stage PATH --mtp-shared-stage PATH`, `--draft-depth 1..4` and
`--spec-policy adaptive|always`. It primes from actual batched prompt hiddens,
uses post-head-norm parent/draft hiddens, includes teacher forcing in prefill
and compares every delivered ID with a plain warmup. Optional
`--decode-state-check TRACE_PREFIX` compares complete target endpoint state
outside timing. This candidate has native/unit validation; full-model timing
is still pending. See the Strata document for its serial qualification gates.

For tokenization of an already rendered chat prompt:

```bash
python3 a64fx/glm5/glm5_tokenizer.py encode-file \
  a64fx/glm5/prompts/glm53f_cpp_codegen_task.md > tmp/prompt.ids
```

The Python encoder approximates non-ASCII pretokenization; its decoder is
exact. Do not assume arbitrary multilingual prompt IDs match the reference
tokenizer without checking them.

## Paths and configuration

Set overrides before invoking the launcher. Prefer absolute paths; stage
directories refer to each rank's node-local filesystem.

| Setting | Purpose/default |
| --- | --- |
| `GLM53F_GGUF` | First GGUF shard; overrides the quantization-specific model path |
| `GLM53F_MODEL_DIR` | Metadata/tokenizer/checkpoint directory |
| `GLM53F_BIN_DIR` | Shared binaries: `a64fx/glm5/build/glm53f` |
| `GLM53F_BUILD_DIR` | Compiler scratch/objects: `/local/glm53f-build-$PJM_JOBID` |
| `GLM53F_LOG_DIR` | Shared rank logs: `tmp/glm53f-q4-$PJM_JOBID` |
| `GLM53F_RUN_TAG` | Unique log suffix; defaults to launcher PID |
| `GLM53F_STAGE_DIR` | Routed weights: `/local/glm53f-q4-routed-$PJM_JOBID` |
| `GLM53F_NATIVE_PREFIX` | Native stages: `/local/glm53f-q4-native-$PJM_JOBID` |
| `GLM53F_GGUF_CORE_STAGE`, `GLM53F_GGUF_SHARED_STAGE` | Override native compact-copy directories |
| `GLM53F_Q2_EMBED_STAGE`, `GLM53F_Q2_HEAD_STAGE` | Embedding/head: `/local/glm53f-q4-{embed,head}-$PJM_JOBID` |
| `GLM53F_Q2_{DENSE,SPARSE,KDA,SHEXP}_STAGE` | Override individual native projection stages |
| `GLM53F_TOPO_PATH` | Reuse a topology from this allocation; otherwise generate a fresh one |
| `GLM53F_MPI_HOME`, `GLM53F_MPICC`, `GLM53F_MPIEXEC` | Matched MPI runtime, compiler wrapper, launcher |
| `OMP_NUM_THREADS` | 47 |
| `GLM53F_FAST_MATH`, `GLM53F_NO_MATH_ERRNO` | Both 1 in this launcher; set 0 for conservative builds |

Historical `Q2` stage-tool/environment names also handle Q4; they are
preserved for compatibility, not a statement about the loaded quantization.
The old Q4/Q2 launch scripts are thin aliases. Use
[alternate workflows](GLM53F_EXPERIMENTS.md) for Q2, hybrid, MTP and the
separate llama.cpp Q8 path.

Larger outer prefill chunks can be evaluated with
`build_glm53f_integrated_12n.sh check 47 4096`, then explicit
`--prefill-chunk 1024|2048|4096` on the benchmark or generation runner.
The third build argument bounds shared model/expert workspaces; accepted
capacities are 512 (default), 1024, 2048 and 4096. Attention panels and
verification snapshots keep their own limits. The benchmark records both
capacity and the actual chunk. Larger chunks remain unqualified experiments.
`glm53f_prefill_chunk_check_12n` compares a 512-token reference with another
chunk using exact final hidden streams and complete KDA/sparse endpoint state.

## Memory and staging rules

- Each node has 32 GiB HBM. Q4 routed weights alone occupy
  15,456,534,528 bytes per rank.
- Use the bounded stage tools; do not copy or concatenate complete model
  shards. The native compact copies now use the same bounded copier, with
  no recursive deletion or unbounded directory copy.
- `/local` is node-local and disappears when the allocation ends. Reuse only
  stages belonging to the current job. The launcher checks all twelve rank
  manifests before loading; loaders validate the image contents.
- Stage reuse does not universally fingerprint the source GGUF. Use separate
  directories when changing models or quantizations. Do not redirect a native
  compact destination onto a source or hybrid image.
- Binaries and logs must be on shared storage. A rank-0 `/local` executable
  is not visible to the other eleven ranks.
- Monitor the runner's sampled minimum `MemAvailable`. Long-context capacity,
  BF16 caches and context parallelism require separate qualification; a model
  metadata limit of 1M tokens is not a demonstrated memory-safe capacity.
- Compare speed serially within the same allocation. Keep
  `XOS_MMM_L_PAGING_POLICY`, threads, prompt, math flags and collectives fixed.

## Development map

| Files | Responsibility |
| --- | --- |
| `run_glm53f_12n.sh`, `scripts/glm53f_*.sh` | CLI, shared environment, MPI preflight and staging |
| `build_glm53f_integrated_12n.sh` | Single build graph; runtime/check/all tool sets |
| `glm53f_target_decode_12n.c` | Persistent model, scalar/batch execution and generation |
| `glm53f_{kda,sparse}_layer_12n.c` | Attention and persistent state |
| `glm53f_{dense_ffn,expert_decode}_12n.c` | Dense/routed/shared FFN |
| `glm53f_iq_bridge.c` | Native GGUF activation contracts and SVE projection kernels |
| `glm53f_q2_*_stage.c`, `glm53f_core_stage.c` | Bounded stage images |
| `test_glm53f_*`, `glm53f_*_check.c` | Numerical and launcher regression checks |

Run `python3 a64fx/glm5/test_glm53f_launcher.py -v` without model access.
Use `bash a64fx/glm5/run_glm53f_12n.sh check` after staging for native
12-node regression checks. See [validation](GLM53F_VALIDATION.md) for
acceptance gates, evidence, and remaining work.

### Experimental routed-expert prefill layout

`--moe-prefill-layout padded` gives grouped gate/up outputs an extra 64-float
stride, avoiding L1 set aliases at larger route cohorts. `tight` is the default.
The expert scheduler, scale/minimum accumulation and SwiGLU quantization stay
as in the existing native path. This option affects grouped prefill; scalar
decode and grouped verification retain their kernels. Benchmark CONFIG reports
`moe_gu_padding`. Native expert-chain diagnostics show about 2% at 47 threads;
the full 8049-position state gate passes, but five-trial full-model prefill
improvement is only 0.04%, below promotion. Keep tight in the qualified recipe. Use
`--compare-moe-prefill-layout --capture-hidden` with the prompt-endpoint checker
and equal reference/candidate chunk sizes to isolate the layout change.
