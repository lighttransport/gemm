# GLM-5.3-Flash Q4 routed experts on 12 A64FX nodes

Run in an allocated interactive job with 12 MPI ranks:

```sh
bash a64fx/glm5/run_glm53f_q4_12n.sh
```

The launcher uses `mpifcc -Nclang`, node-local `/local` staging, and the
allocation's freshly generated uTofu topology. `GLM53F_Q4_MODEL` overrides
the first GGUF shard (default:
`~/models/glm53f-gguf-all/UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf`).
The underlying shared launcher retains historical Q2 labels in its logs.

By default (`GLM53F_NATIVE=1`, since job 51909852) the model is **fully
GGUF-backed**. Every matrix runs from the GGUF's own quantized blocks: routed
experts (Q4_K/Q5_K/Q6_K), dense FFN layers 0--2, all 34 KDA layers, the 11
sparse MLA layers, and the shared expert (Q8_0). See the next section.
`GLM53F_NATIVE=0` selects the older hybrid, where the shared expert and the
compact attention/dense core come from safetensors-derived FP8/BF16 images.

The routed image occupies 15,456,534,528 bytes per rank. Each expert has eight
256-column parts distributed over 12 ranks, preserving quantization blocks.
Restaging reads bounded chunks, syncs output, drops file cache, and atomically
publishes complete images. Use a separate Q4 directory; the legacy reuse
check does not identify the source model.

Q4_K and Q5_K use SVE SDOT with affine minimum correction. Q6_K uses signed
six-bit values and per-16-value scales. Activations are quantized to Q8 in
256-value blocks. The fused kernels do not expand complete weight matrices.

Numerical validation:

```sh
export OPAL_PREFIX=/opt/FJSVxtclanga/tcsds-1.2.43
export MPI_HOME=$OPAL_PREFIX TMPDIR=/local
mpifcc -Nclang -O3 -march=armv8.2-a+sve -fopenmp \
  a64fx/glm5/test_glm53f_kquant.c -lm -o /local/test_glm53f_kquant
/local/test_glm53f_kquant
```

The test compares 300 randomized 256/4096-column rows against independently
dequantized weights and the same Q8 activation values. Both conservative and
fast-math builds pass; worst error normalized by the absolute term sum is
3.05e-8 or less. This checks fused arithmetic, not activation quantization loss
or whole-model semantic correctness.

## Fully GGUF-backed UD-Q4_K_XL (job 51909852)

In UD-Q4_K_XL, every non-routed weight matrix is **Q8_0**: embeddings, the
vocabulary head, all attention and KDA projections, dense FFN, shared experts
and `hc_*_fn`. Only the routed experts are K-quants. The Q2 native stages
accepted only Q5_K/Q6_K for these tensors, so native Q4 required Q8_0 support
throughout.

| Component | Stage tool / env | Source blocks | Per-rank layout |
| --- | --- | --- | --- |
| routed experts 3--44 | `glm53f_q2_stage` | Q4_K/Q5_K/Q6_K | 8 x 256-col parts, 15.46 GB |
| dense FFN 0--2 | `glm53f_q2_dense_stage`, `GLM53F_Q2_DENSE_STAGE` | Q8_0 | gate/up rows, down cols (1024), 40 MB |
| sparse MLA (11 layers) | `glm53f_q2_sparse_stage`, `GLM53F_Q2_SPARSE_STAGE` | Q8_0 | q_a/kv_a replicated, q_b/v_b head rows, output head cols, 190 MB |
| KDA (34 layers) | `glm53f_q2_kda_stage` (V3), `GLM53F_Q2_KDA_STAGE` | Q8_0 | Q/K/V/f_b/g_b/beta head rows, f_a/g_a replicated, output **head columns**, 423 MB |
| shared expert 3--44 | `glm53f_q2_shexp_stage`, `GLM53F_Q2_SHEXP_STAGE` | Q8_0 | 64-aligned intermediate slice (128/192), 105 MB |
| embeddings / head | `glm53f_q2_{embed,head}_stage` | Q8_0 -> F32 rows | unchanged |
| mHC, norms, conv, indexer, routers | `glm53f_q2_core_patch` on a copy of the compact core | GGUF -> BF16/F32 | unchanged |

Kernel and runtime changes:

- `glm53f_iq_bridge.c` adds `glm53f_native_type_supported()`, which is the
  routed set plus Q8_0. The routed-expert predicate is unchanged.
- Activations follow llama.cpp's `vec_dot` contract: Q8_K-style blocks for
  K/IQ weights, Q8_0 blocks for Q8_0 weights.
- Q8_0 rows are **repacked at load time** into contiguous int8 values
  followed by F32 scales. The fp16-to-fp32 conversion is exact. Per lane, the
  4-row SDOT kernel computes `acc += float(sdot) * (dw * dx)`, which is the
  block kernel's operation order. It is bit-identical to the GGUF-layout
  kernel (30/30 shapes) and runs at 135--157 Gweights/s, against 110 for the
  GGUF layout, with demand paging.
- `glm53f_native_matvec_team()` is an orphaned-`omp for` variant. In native
  KDA it fixes a nested-parallel bug: `glm53f_iq_matvec_2` was called inside
  `omp parallel` + `omp single`, so native Q/K/V ran on one thread. Q/K/V,
  f_a, g_a and beta now share one work-shared loop.
- The KDA output is column-sliced for Q8_0. The 640/768-column head slices
  align with 32-value blocks, so rank-local activation quantization equals
  llama.cpp's full-vector quantization. The existing sum all-reduce is kept,
  replacing the replicated-output allgather.
- The native sparse value path (`mla_heads_q8_value`) was single-threaded over
  heads and tokens. It is now work-shared in five phases, keeping each output
  element's accumulation order.
- The native shared expert runs gate/up, SwiGLU with the MoE clamp, and down
  from the Q8_0 slice after the routed experts. It is added before the same
  all-reduce.
- Batched prefill/MTP paths keep the compact (GGUF-patched) weights.

Unit test (conservative and fast-math builds):

```text
PASS K-quants cases=300 worst_normalized_error=2.72467e-08
PASS Q8_0 native cases=42 repacked_bit_exact=30 worst_normalized_error=3.94545e-08 mixed_pair=BIT_EXACT
```

### Parity against llama.cpp (token 1234, position 0)

The streamed llama.cpp reference had to be fixed first; see the
"streamed-reference aliasing" note in `GLM53F_VALIDATION.md`. Against the
fixed reference, the relative L2 of the 16,384-float layer output is:

| Layer | native GGUF | hybrid |
| ---: | ---: | ---: |
| 0 | 0.00590 | 0.00972 |
| 2 | 0.00520 | 0.00690 |
| 3 (first sparse/MoE) | 0.00856 | 0.00999 |
| 7 | 0.00809 | 0.00957 |
| 11 | 0.00990 | 0.01330 |
| 15 | 0.00808 | 0.00963 |
| 16 | 0.0231 | 0.0273 |
| 20 | 0.0303 | 0.0362 |
| 21 | 0.136 | 0.141 |

Native is closer to llama.cpp than the hybrid at every layer.

- **Layer 0.** Q/K/V, conv, q/k L2-norm, beta, decay, recurrent scan and
  gated norm all stay within 1.3e-4 to 5.6e-4 relative L2. The input
  `attn_norm` differs by only 5.4e-6.
- **Why the floor is ~0.5%.** A tiny upstream float difference flips Q8_0
  activation-rounding decisions. The expected amplification is
  `sqrt(noise*127/3)*3/127`, which predicts 3.4e-4 for `q_proj` and 3.7e-3
  for `kda_out`; the measured values are 4.3e-4 and 4.3e-3. The floor is set
  by llama's own Q8_0 activation contract, not by a model discrepancy.
- **Divergence after layer 20.** Layer 16 selects the same top-8 experts as
  llama.cpp. Layer 21 is a router near-tie: the biased sigmoid scores of the
  8th and 9th experts differ by only 0.0065. The 3% accumulated input
  difference swaps expert 177 (llama) for 188 (production). Downstream
  trajectories then legitimately diverge, including whether the
  massive-activation dimensions form by layer 25.
- **Final token.** Both production arms and llama.cpp select token 220.

Scripts, traces and comparison tables are in `tmp/glm53f-q4-51909852/`:
`scripts/`, `stream-1234/`, `parity-t1234.txt`, and `scripts/router_check.py`.

### Decode throughput (same allocation, alternating A/B, input token 1)

| Run | tokens | tok/s | min MemAvailable |
| --- | ---: | ---: | ---: |
| hybrid 1 | 128 | 21.776 | 9.26 GiB |
| **native 1** | 128 | **24.952** | 8.63 GiB |
| hybrid 2 | 128 | 22.448 | 9.28 GiB |
| **native 2** | 128 | **24.808** | 8.65 GiB |

The 16-step hybrid control at the start of the job measured 20.324 tok/s.
The production launcher was checked end to end:
`GLM53F_TARGET_STEPS=128 GLM53F_MIN_TOK_S=20 bash a64fx/glm5/run_glm53f_q4_12n.sh`
ran the repo build, all native stages, the patched core/shared copies and
decode. It reported `tok_s=23.947 ... PASS` and `SENTINEL glm53f_q2_12n=OK`,
with 9.35 GiB minimum MemAvailable and the same final token as the candidate
runs.
Native GGUF decode is about 12% faster than the hybrid, mainly because the
34 KDA layers stream Q8_0 (1.125 B/weight repacked) instead of BF16 (2
B/weight). Both 128-token outputs are coherent English. With this
one-token prompt, both fall into greedy repetition loops.

### 8K-context generation (8,049-token C++20 coding prompt, 256 new tokens)

Prompt: `tmp/glm53f-quality-8k/prompt.ids`. Decode windows are 128 tokens.

| Arm | prefill mode | prompt tok/s | decode window 1 | decode window 2 | min MemAvailable |
| --- | --- | ---: | ---: | ---: | ---: |
| native | legacy (per token) | ~22.0 | 21.422 | 21.172 | 8.20 GiB |
| hybrid | legacy (per token) | ~19 | 20.108 | 20.082 | 8.85 GiB |
| native | fast (mask 27, slab 16, tree-packed) | 34.383 | 21.095 | 20.546 | 8.90 GiB |
| hybrid | fast | 54.610 | 19.106 | 19.393 | 9.56 GiB |

- **Coherence.** All four outputs are coherent and reason correctly about the
  sorting task.
- **Native: fast vs legacy prefill.** The first 66 generated tokens are
  identical. Native layers use the same kernels in both modes.
- **Hybrid: fast vs legacy prefill.** The outputs differ from the first token
  but remain valid ("Let me carefully design…" instead of "The user wants…").
  Batched FP8/BF16 prefill is a different arithmetic from its per-token path.
- **Native fast-prefill speed.** Its prompt rate is lower because the native
  sparse and dense layers still run per token inside a batch.

**Fast-prefill fix.** Before this job, fast prefill with GGUF routed experts
crashed with SIGSEGV at the first chunk, including with the pre-edit binary.
`moe_prefill_grouped` passed K-quant expert parts to the FP8 panel kernel
(`glm53f_matvec_fp8_bits_4x4`, reading a NULL scale pointer). K-quant/IQ
parts now use `glm53f_iq_expert_weighted` per selected token. That is the
decode kernel, so routed-expert arithmetic is identical in prefill and
decode, and each rank's ~2 MB expert part stays L2-resident across its
tokens.

### Next tasks (priority order, after job 51909852)

1. **Batched native prefill.** Native fast prefill is 34 tok/s, against 55
   for the hybrid. In `glm53f_sparse_sublayer_batch_12n` and
   `glm53f_dense_ffn_sublayer_batch_12n`, native sparse and dense layers fall
   back to one token at a time inside a batch.
   - Add a tokens-x-rows kernel for repacked Q8_0 (`GLM53F_NATIVE_Q8_0R`):
     a 4-row x 4-token tile that reuses each weight vector across tokens.
     Prepare one Q8_0 activation per token, keeping `native_act` per token.
   - Use it for the sparse q_a/q_b/kv_a/output projections, dense
     gate/up/down, the KDA batch projections (Q/K/V, f_a/f_b, g_a/g_b, beta,
     output) and the native shared expert in `moe_prefill_grouped`.
   - Keep per-token accumulation order equal to the decode kernel, so fast
     prefill and legacy prefill stay token-identical. The current check
     already gives 66/256 identical tokens.
   - Target: at least 150 tok/s prompt ingestion on the 8K coding prompt,
     with decode unchanged.
2. **Weight arena for native tensors.** The native loaders make about 450
   separate `posix_memalign` allocations of 1--3 MB each, including the
   repacked copies. Earlier work on Laguna S-2.1 fp8 measured a 2x decode
   gain from handing weights out of a few GB-sized mmap chunks with parallel
   first touch.
   - Add a bump arena in `glm53f_iq_bridge.c`, used by `glm53f_native_repack`
     and the KDA, sparse, dense and shexp loaders.
   - Measure with a same-allocation A/B of whole binaries only.
3. **Remaining non-native tensors (llama.cpp contract).**
   - MLA `attn_k_b` (Q8_0) is dequantized to BF16 for query absorption. Stage
     it per head, 512 x 256 Q8_0, and absorb with the Q8_0 activation
     contract.
   - llama.cpp rounds the absorbed query and the softmax probabilities to F16
     against its F16 cache. Match that inside `mla_heads_q8_value`.
   - `hc_*_fn` is Q8_0 in the GGUF, and llama.cpp quantizes the
     16,384-float mHC input to Q8_0. Production uses BF16 weights and F32
     input.
   - None of these matter at position 0; all add small differences at longer
     contexts.
4. **Multi-token parity.** The streamed probe processes one token per
   process. Extend it, or add a small llama.cpp driver, to feed a short
   prompt (for example 16--64 tokens) and dump per-layer streams at the last
   position.
   - Rebuild it only from a llama.cpp tree containing the `ggml-backend.cpp`
     aliasing fix, which is still uncommitted in `~/work/llama.cpp`.
   - Compare with `GLM53F_LAYER_TRACE_DIR` from a `--generate` run.
   - Also re-derive the Q2 full-chain conclusions against the fixed
     reference.
5. **Long-context and MTP.**
   - Measure 16K-context windows with three repeats.
   - Check the Q4 MTP speculative path (`run_glm53f_q4_mtp_12n.sh`) with
     `GLM53F_NATIVE` stages exported; its verifier batches run the compact
     paths.
   - Run the 8K coding prompt to EOS (or at least 1K tokens) and compile the
     extracted C++.

### Resuming prompt

Paste this into a new session inside a fresh 12-node interactive allocation:

> Continue GLM-5.3-Flash UD-Q4_K_XL work on 12 A64FX nodes. This host is
> the allocation's rank-0 node: run MPI jobs directly, with no pjsub, and use
> the native `mpifcc -Nclang` compiler. Read `a64fx/glm5/GLM53F_Q4_12N.md`
> first, from "Fully GGUF-backed UD-Q4_K_XL (job 51909852)" through "Next
> tasks". The branch is `glm53f`, and commit `90420b32` holds the native path.
>
> **Current state.**
>
> - `bash a64fx/glm5/run_glm53f_q4_12n.sh` defaults to `GLM53F_NATIVE=1`.
>   Every matrix comes from the GGUF blocks under
>   `~/models/glm53f-gguf-all/UD-Q4_K_XL/`: routed experts are
>   Q4_K/Q5_K/Q6_K, and everything else is Q8_0, repacked at load to
>   `GLM53F_NATIVE_Q8_0R`.
> - Measured in job 51909852: 24.9 tok/s decode at short context (hybrid 22),
>   21.4 tok/s at 8K context, native fast prefill 34 tok/s.
> - Parity against a fixed llama.cpp streamed reference is 0.5--1% relative
>   L2 through layer 20. At layer 21 a router near-tie swaps one expert.
>
> **Reproducing in the new job.**
>
> 1. Stage and run with `GLM53F_TARGET_STEPS=128 GLM53F_MIN_TOK_S=20 bash
>    a64fx/glm5/run_glm53f_q4_12n.sh`. It takes about 30 minutes; the routed
>    expert stage (15.5 GB per rank) dominates.
> 2. For A/B runs, reuse the job-51909852 helper scripts in
>    `tmp/glm53f-q4-51909852/scripts/` after editing their hard-coded job ID
>    and paths:
>    - `run_decode.sh TAG native|hybrid TOKEN STEPS [args]`. Pass `-` as
>      TOKEN when the args start with `--generate PROMPT OUT N`.
>    - `build.sh`, which writes candidate binaries to **shared** storage.
>      Other ranks cannot see `/local`.
>    - `stream_ref.sh TOKEN OUT`, the llama.cpp reference.
>    - `compare_layers.py` and `router_check.py`.
> 3. Before measuring, get the hybrid control `GLM53F_NATIVE=0` in the same
>    allocation. Never compare tok/s across allocations.
>
> **Rules and gotchas.**
>
> - Do not measure performance while the llama.cpp probe (48 threads) runs
>   on rank-0's node.
> - Check `XOS_MMM_L_PAGING_POLICY` inside the job. The preset
>   `demand:demand:prepage` places heap pages on one CMG.
> - `--generate` must be argv[4] of `glm53f_target_decode_12n`.
> - Rank stdout goes only to the `-of-proc` files.
> - Gate every change on:
>   - `test_glm53f_kquant`: PASS in both conservative and fast-math builds,
>     including `repacked_bit_exact`.
>   - Unchanged token-1234 layer parity (`compare_layers.py` against
>     `tmp/glm53f-q4-51909852/stream-1234/graph`).
>   - Coherent 8K coding-prompt output
>     (`tmp/glm53f-quality-8k/prompt.ids`).
>   - Same-allocation tok/s against the hybrid control.
>
> **Start with next task 1, batched native prefill.** Target at least 150
> tok/s prompt ingestion with decode unchanged. Fast prefill and legacy
> prefill should stay token-identical for the native arm.
>
> Report measurements with job IDs, update this document, and commit only
> the glm53f files that are touched.

## Measurements

Job 51351748: packed Q4/Q5 SVE baseline measured 19.424 tok/s at short context
and 17.826 tok/s for 64 output tokens after an 8,378-token prompt. The latter
profile was 6.388 ms MHC, 31.043 ms attention, 17.935 ms FFN, 1.383 ms head.

Job 51370956: restored the same images on all 12 nodes. Rank 0's routed
checksum is `80c171ac9f66d851`, matching the previous job. The existing binary
measured 20.585 tok/s over 64 short-context tokens and reproduced all token
IDs and printed logits from the previous packed-kernel run. Cross-job timing
differences must not be attributed to kernel changes.

The candidate accumulates integer dot products and minimum corrections over
each 256-value block before conversion to float. Its 8K-context generation
and profile are under `tmp/glm53f-q4-51370956/candidate8k.*`; output IDs are
retained on shared storage in `candidate8k.ids` when generation finishes.
The prompt requests a revised C11/C++17 reference header. The candidate
completed at EOS after 7,253 output tokens: 18.644 tok/s decode while context
grew from 8,378 to roughly 15,631 tokens. Prompt throughput was 19.081 tok/s.
Decode phase averages: 6.113 ms MHC, 29.972 ms attention, 16.732 ms FFN,
1.512 ms head. These long-run numbers do not isolate kernel speedup against
the short-context baseline; stable 20+ tok/s long-context decode is not met.

The raw response contains Markdown fences despite the source-only instruction.
After removing only those fences, the generated header compiles with both
`mpifcc -Nclang -std=c11` and `mpiFCC -Nclang -std=c++17`, using
`-O2 -Wall -Wextra -Werror -pedantic`. Both ARM executables pass RMSNorm,
BF16 dot-product, and context-partition smoke checks. This demonstrates a
coherent compilable output with a formatting violation; it is not exhaustive
verification of all generated numerical primitives or requested bug fixes.
Raw output, extracted header and harness are in the same shared log directory.

## Direct UD-Q4_K_XL validation (A64FX 12-node job 51843198)

The six-shard model under
`$HOME/models/glm53f-gguf-all/UD-Q4_K_XL/` was staged with the requested first
shard override:

```sh
export GLM53F_Q4_MODEL="$HOME/models/glm53f-gguf-all/UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf"
export GLM53F_TARGET_STEPS=16 GLM53F_MIN_TOK_S=0
export GLM53F_Q4_LOG_DIR="$PWD/tmp/glm53f-q4-$PJM_JOBID"
bash a64fx/glm5/run_glm53f_q4_12n.sh
```

The outer interactive timeout occurred after routed and embedding staging, so
the completed artifacts were resumed without restaging. A fresh topology was
generated for the current allocation before decode. All twelve routed images
completed at `15,456,534,528` bytes each. The resumed decode completed 16/16
steps with `GLM53F_TARGET_DECODE_12N ... tok_s=20.760 ... PASS`; rank 0 loaded
14.395 GiB of routed data and the sampled minimum `MemAvailable` was
12.504822 GiB. This confirms that the UD-Q4_K_XL routed path fits comfortably
in the 32-GiB A64FX nodes when staged locally.

## SVE unpacking follow-up (job 51370956)

Replace two TBL lookups and OR with SPLICE to concatenate low/high nibble
vectors. This preserves byte values and arithmetic ordering. The 300-case
reference test passes; two full 128-token runs reproduce every baseline token
ID and printed logit exactly.

An optional kernel benchmark is enabled with `GLM53F_KQUANT_BENCH=1` on the
test executable. It streams 8,192 rows of 4,096 weights with 47 OpenMP threads.
In baseline/candidate/candidate/baseline order, Q4 throughput was
196.054 / 210.449 / 206.617 / 196.163 Gweights/s. Q5 was essentially unchanged:
173.322 / 170.715 / 174.040 / 174.040 Gweights/s.

Full decode: baseline 23.011 tok/s, candidate 20.447 and 22.663 tok/s.
The repeat profile has FFN 15.771 ms versus baseline 15.811 ms; attention
21.321 versus 20.549 ms. Thus the isolated Q4 gain is 6–7%, but these runs
do not establish an end-to-end speedup. Long-context performance has not been
remeasured for SPLICE. Logs use `before-splice`, `after-splice`, and
`after-splice-repeat` in the job's shared log directory.

## Remove idle KDA team barriers (job 51370956)

Detailed-profiling singles previously imposed five barriers per KDA layer
even with profiling disabled. The uniform profiling condition now surrounds
each single construct. Eight-row matvec helpers also skip remainder
worksharing when no remainder exists, retaining the producer-loop barrier.
Arithmetic and dependent-loop synchronization are unchanged.

Short-context 128-token decode measured 23.407 tok/s, attention 19.910 ms,
and reproduced all baseline token IDs and printed logits exactly.

After the same 8,378-token prompt, the two 64-token decode windows measured
**20.069 and 20.123 tok/s**, **20.096 tok/s** overall for 128 output tokens.
All 128 IDs match the prefix of `candidate8k.ids` exactly. Phase averages:
MHC 6.502 ms, attention 27.312 ms, FFN 15.601 ms, head 1.226 ms.
This clears 20 tok/s for the measured 128-token sample; sustained multi-thousand
token throughput has not been remeasured. Logs: `kda-barriers.*` and
`kda-barriers8k.*`; output: `kda-barriers8k.ids` in the shared job directory.

## Q4 TP12 MTP verifier work (job 51370956)

The draft uses the safetensors layer-45 weights (0.563 GiB routed per rank),
not a Q4 conversion of that layer. Stage it separately after each restart:

```sh
bash a64fx/glm5/run_glm53f_mtp_stage_12n.sh
# Q4 target/core/shared images must already have been staged for this job.
bash a64fx/glm5/run_glm53f_q4_mtp_12n.sh PROMPT_IDS OUTPUT_IDS 128 1
```

The launcher builds with `mpifcc -Nclang` and executes the ARM binary directly
on the existing allocation. It uses `/local` for build/staging and shared
`tmp/` for logs. `GLM53F_BUILD=0` reuses an existing binary. Input/output IDs
are whitespace-separated integers, with the same prompt contract as the
target generator. Prompt replay is excluded from the decode timing.

Optimizations and controls:

- Cache-only MTP warmup/replay computes embedding fusion, KV/indexer
  projections, and completed compression pools. Query/attention output,
  MoE, and vocabulary head are skipped because they do not affect persistent
  cache state. Context-parallel caches retain the existing attention path.
  `GLM53F_SPEC_FULL_REPLAY=1` selects the full replay reference.
- `GLM53F_KDA_BATCH_TEAM=1` keeps one team across batched projections,
  causal convolution, per-head recurrent updates, normalization, and parallel
  snapshot copies. Every committed position retains its own recurrent state.
- `GLM53F_Q4_BATCH_SHARED=1` batches the common FP8 shared expert while retaining
  the existing routed Q4/Q5/Q6 SDOT kernels and accumulation order.
- `GLM53F_SPARSE_BATCH_OP=1` batches sparse output projection and its collective;
  KV updates and attention selection remain causal. This is experimental and
  disabled in the launcher pending end-to-end measurement.
- Remove the speculative pre-verification snapshot, which was never restored.

Validation/measurement controls on the speculative binary:

- `GLM53F_SPEC_REFERENCE_IDS=FILE` checks delivered tokens against a saved
  reference prefix and reports how many tokens were checked. A mismatch fails
  the run. A plain `PASS` without this check is only a runner/gate status.
- `GLM53F_SPEC_SELF_REFERENCE=1` first generates a scalar reference from the
  identical warmed target state, restores that state, and checks speculation.
  It also saves `OUTPUT_IDS.greedy`. Its separately reported scalar timing is
  not included in speculative timing. Do not combine with an external reference.
- `GLM53F_SPEC_DRAFT_SWEEP=1` tests draft counts 1 through the positional maximum
  from the same warmed state; output files gain `.d1`, `.d2`, etc.
- `GLM53F_SPEC_COMPARE_BATCH=1` compares full replay/legacy verifier against
  cache replay/all three batching switches from the same warmed state. Outputs
  gain `.baseline`/`.optimized`. Do not combine with the draft sweep.
- Coding-prompt generation stops at EOS. Work speculated beyond EOS is timed
  but not counted as delivered output.

The real-weight cache test, `test_glm53f_mtp_cache_12n MODEL MTP_ROUTED
MTP_SHARED 2051`, passes a rollback crossing completed pools: next draft,
logit, and all 4096 hidden values are bit-exact. Full replay costs 3.480 ms
per position versus 0.426 ms for cache-only replay (8.17x faster replay).
KDA tests for 2, 3, 4, and 5 positions have zero output difference and bit-exact
recurrent snapshots. Four-position layer-44 verification falls from 1.372 ms
to 0.670 ms. These are component measurements, not whole-model speedups.

The full-model five-position batch/rollback check passes with all batching
switches enabled. The sparse-only 8190-position test also passes
(`rel_l2=7.33e-8`, rollback difference zero), but its single MPI-only timing
sample regresses from 6.028 to 7.373 ms; do not infer a sparse speedup from it.

An initial one-draft run after the 8378-token coding prompt delivers 383 tokens
at 21.420 tok/s, accepting 127/128 drafts (99.22%). Phase cost per cycle is
50.200 ms scalar target, 3.345 ms draft, 85.534 ms verification, and 0.898 ms
replay. KDA/shared batching is enabled; sparse projection batching is disabled.
The output differs from the older saved non-MTP response from its first token,
so this measurement alone does **not** establish greedy equivalence. The
same-state scalar/speculative validation is required before a quality claim.
At this near-perfect acceptance, the measured cycle costs permit only about
21.43 tok/s even at 100% acceptance. The 60 tok/s target remains unmet; target
verification throughput, not first-draft acceptance, is the limiting factor.
