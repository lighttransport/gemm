# GLM-5.3-Flash Q4 routed experts on 12 A64FX nodes

Run in an allocated interactive job with 12 MPI ranks:

```sh
bash a64fx/glm5/run_glm53f_q4_12n.sh
```

The launcher uses `mpifcc -Nclang`, node-local `/local` staging, and the
allocation's freshly generated uTofu topology. `GLM53F_Q4_MODEL` overrides
the first GGUF shard (default: `~/models/glm53f-q4/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf`).
The underlying shared launcher retains historical Q2 labels in its logs.

This is a hybrid: routed experts use the GGUF's Q4_K, Q5_K and Q6_K tensors;
shared experts and the compact attention/dense core use the existing
safetensors-derived images. It is not a fully GGUF-backed model.

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
