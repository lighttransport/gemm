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

### Completed same-state check and continuation (job 51383005)

The later `mtp-selfcheck8k.27.0` log in the previous remote checkout
`~/work/gemm/q38nf/tmp/glm53f-q4-51370956` completes the missing check:
7,189 delivered tokens, 21.723 tok/s, acceptance 2,379/2,405 (98.9189%),
and `GLM53F_SPEC_REFERENCE checked=7189 delivered=7189 PASS`.
The input fixture is `~/work/gemm/q38nf/tmp/glm53f-codegen-51336165/prompt_ids_header.txt`:
8,379 IDs with 8,378 positions replayed before decode.

The continuation uses the login4 allocation in the repository's `session.md`.
Build and stage without starting target inference, then stage MTP:

```sh
GLM53F_STAGE_ONLY=1 bash a64fx/glm5/run_glm53f_q4_12n.sh
bash a64fx/glm5/run_glm53f_mtp_stage_12n.sh
```

The integrated build now includes the MTP stager. The speculative launcher's
optional fifth argument limits **delivered tokens**, independently of the cycle
and draft counts. A supplied reference must cover every delivered token; short
references fail. The launcher checks target/MTP blobs and manifests and the
compact core on every rank, and wraps each inference rank in a 2 GiB
`MemAvailable` guard with minimum-headroom reporting.

Optional experiments, disabled unless explicitly selected:

| Control | Experiment |
| --- | --- |
| `GLM53F_SPEC_CARRY=1` | Seed once, then draft from the verifier's pending token and predecessor hidden state; omit the repeated scalar target step. |
| `GLM53F_SPARSE_BATCH_FRONT=1` | Reuse FP8/BF16 projection weights across verifier positions while keeping selection and pool updates causal. |
| `GLM53F_Q4_BATCH_ROUTED=1` | Group matching expert parts across tokens; reuse Q4/Q5 unpacking and retain Q6/IQ row fallback. Requires shared batching. |
| `GLM53F_SPARSE_MLA_REG=1` | Keep query absorption and per-shard weighted-value sums in SVE registers, preserving reduction order. |
| `GLM53F_SPEC_DIRECT_SNAPSHOT=1` | Capture KDA state directly into per-position target snapshots and restore independent layers in parallel. |
| `GLM53F_KDA_FUSED_OP=1` | Run scalar KDA output projection in its existing team with the eight-row kernel. |
| `GLM53F_SPARSE_DETAIL=1` | Report projection, indexer, selected-KV packing, MLA, output-projection, and collective time. |
| `GLM53F_NUMA_INTERLEAVE=4-7` | Launch under `numactl` with interleaved anonymous allocations; inspect the allocation's topology before selecting this mask. |

`GLM53F_SPEC_OPT_SWEEP=1` compares eight cumulative variants from the same
warmed state: baseline, carry, front, routed, MLA, snapshots, KDA output,
and sparse output batching. It is mutually exclusive with the existing draft
sweep and legacy batch comparison. Each output gains its variant suffix.
Use `GLM53F_SPEC_SELF_REFERENCE=1` for a matching scalar reference, and pass
the same delivered-token limit to every measurement.
`GLM53F_SPEC_REPEAT=N` repeats the selected comparison from that same state,
reversing variant order on alternate repetitions and adding `.rN` to output
filenames. The scalar reference is generated once outside the timed trials.

Unsloth's [September 4 decode change](https://github.com/ggml-org/llama.cpp/commit/00699716c275498ff84d71e329178fe21cba56a6)
caches completed pooled indexer keys. This runner already has incremental
completed-pool updates. Its [GLM-5.3-Flash guide](https://unsloth.ai/docs/jp/moderu/glm-5.3-flash)
also motivates testing small MTP draft counts; GPU measurements do not predict
A64FX throughput.

The extended `test_glm53f_kquant.c` checks scalar-versus-multitoken rows and
shared, reordered, and empty expert routes. `test_glm53f_mla_register.c`
checks projections, logits, partial sums, and output at five lengths. Both
pass bit-exact comparisons on A64FX with the fast-math build.

#### Initial continuation results

All 12 routed stages completed at 15,456,534,528 bytes/rank; rank 0's FNV1a
is `80c171ac9f66d851`, matching the previous working stage. Shared/core and
MTP stages also completed on all ranks. Sparse batch tests at warm lengths
2050 and 8190 have zero output/rollback difference. Full-target five-position
batch/rollback passes, and the MTP cache check at 2051 has bit-exact hidden
states. Minimum headroom across the full-target test was 12,790,272 KiB.

The first same-state sweep uses 128 delivered tokens and two drafts; these
are screening results, not sustained-throughput claims. All eight variants
match all 128 greedy IDs, and that scalar reference also matches the archived
7,189-token response's prefix exactly.

| Cumulative variant | Delivered tok/s |
| --- | ---: |
| Baseline | 20.207 |
| Carry | 21.250 |
| + batch front | 23.035 |
| + routed grouping | 22.918 |
| + register MLA | 23.621 |
| + direct snapshots | 24.937 |
| + KDA output | 24.938 |
| + sparse output batch | 25.245 |

The last variant is 24.9% faster in this screen, but still below 60 tok/s.
Its verifier costs 31.742 ms/position: attention 15.075, FFN 12.197,
mHC 3.977, and head 0.440 ms. Routed grouping and scalar KDA output show no
isolated benefit here. Logs and identities are under
`tmp/q4-mtp-51383005/` remotely and `tmp/q4-mtp-51383005-evidence/` locally.

The separate optimized 128-token draft-depth screen matches the archived
greedy reference at every depth. Delivered rates for depths 1/2/3/4 are
26.568 / 24.867 / 22.569 / 16.374 tok/s; acceptance fractions are
0.9844 / 0.7800 / 0.6288 / 0.4716. One draft is best in this screen.

Placement diagnostics on rank 0 found resident pages on compute NUMA nodes
4/5/6 but none on node 7. Nodes 0-3 are small assistant-core banks; 4-7 are
the four ~7.2 GiB compute-memory banks. `numactl --interleave=4-7` is supported
and was tested separately rather than silently changing the baseline.

#### Repeated 1,024-token interleaved trials

With all seven candidate switches enabled and `GLM53F_NUMA_INTERLEAVE=4-7`,
the following trials reset to the same 8,378-position prompt state and check
every delivered ID against the archived greedy response. Repetition 2 runs
depths in reverse order. All twelve trials pass 1,024/1,024 IDs.

| Draft depth | Trial 1 tok/s | Trial 2 tok/s | Trial 3 tok/s | Median tok/s |
| --- | ---: | ---: | ---: | ---: |
| 1 | 29.860 | 29.861 | 29.866 | 29.861 |
| 2 | 27.295 | 27.286 | 27.288 | 27.288 |
| 3 | 22.910 | 22.909 | 22.906 | 22.909 |
| 4 | 17.207 | 17.210 | 17.207 | 17.207 |

One-draft acceptance is 509/514 (99.027%). Its first trial's verifier costs
32.554 ms/position: attention 15.019, FFN 12.937, mHC 4.270, and head
0.269 ms. Thus the remaining limit is verifier execution, not acceptance.
The 60 tok/s target is not met. No GPU-derived speedup is claimed.

Driver SHA256 for these trials:
`4df0ce5e0a28add1fb9020858fe0053162822d0bfb1fa733f5f532e2ed06a6f0`.
Prompt SHA256:
`6e06a30ccda8594dfafdf5ab2e13929b30d6d1baebe16dd07155981e89c23aa6`.
Logs: `tmp/q4-mtp-51383005/drafts-op-d4-g1024-r3/`.

The historical one-draft baseline (new switches off, default memory policy)
was then measured with the same prompt and 1,024 delivered tokens, resetting
between three trials: 22.517 / 22.516 / 22.515 tok/s, all 1,024/1,024 IDs
exact. Median candidate improvement is **32.62%** (29.861 versus 22.516).
These are separate process launches because the resident memory placement
differs; baseline/candidate launches themselves were not interleaved.
Within each launch, repeated trials reuse the same warmed model and reset
to the same prompt state. Baseline logs:
`tmp/q4-mtp-51383005/sustained-baseline-d1-g1024-r3/`.

#### Full continuation to EOS

The selected one-draft interleaved configuration delivers **7,189 tokens at
29.133 tok/s**, accepting 3,576/3,613 drafts (98.9759%). It ends at token
154827, passes `REFERENCE checked=7189 delivered=7189`, and its entire ID
file is byte-identical to the archived greedy response (`cmp` exit 0).
Minimum `MemAvailable` across ranks is 10,469,952 KiB (9.99 GiB); no memory
guard trips occur. Decode throughput excludes build, loading, and prompt
replay. This is a single continuation, not aggregate multi-request throughput.

Full-run verifier cost is 33.591 ms/position: attention 15.718, FFN 12.960,
mHC 4.587, and head 0.270 ms. At 98.98% one-draft acceptance, improving
acceptance alone cannot bridge the remaining gap to 60 tok/s; target kernels
need substantial further work. Keep this as a measured improvement, not a
60 tok/s claim or a claim about equivalence to the original BF16 checkpoint.

Output: `tmp/q4-mtp-51383005/sustained-op-d1-g7189-r1.ids`.
Logs: `tmp/q4-mtp-51383005/sustained-op-d1-g7189-r1/`.
The allocation and HTTP bridge were verified healthy at 15:00 JST; weights
remain staged and the job is left running for further development.

Reproduce the standalone bit-exact checks inside the allocation:

```sh
cd a64fx/glm5
mpifcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -ffast-math \
  -fno-math-errno -fopenmp test_glm53f_kquant.c -lm -o /local/test_glm53f_kquant
mpifcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -ffast-math \
  -fno-math-errno -fopenmp test_glm53f_mla_register.c -lm -o /local/test_glm53f_mla_register
OMP_NUM_THREADS=47 /local/test_glm53f_kquant
OMP_NUM_THREADS=47 /local/test_glm53f_mla_register
```

The measured candidate configuration is explicit (defaults remain historical):

```sh
export GLM53F_SPEC_CARRY=1 GLM53F_SPARSE_BATCH_FRONT=1
export GLM53F_Q4_BATCH_ROUTED=1 GLM53F_SPARSE_MLA_REG=1
export GLM53F_SPEC_DIRECT_SNAPSHOT=1 GLM53F_KDA_FUSED_OP=1
export GLM53F_SPARSE_BATCH_OP=1 GLM53F_NUMA_INTERLEAVE=4-7
export GLM53F_SPEC_REPEAT=1 GLM53F_SPEC_OPT_SWEEP=0 GLM53F_SPEC_DRAFT_SWEEP=0
export GLM53F_SPEC_SELF_REFERENCE=0 GLM53F_SPEC_COMPARE_BATCH=0 GLM53F_SPEC_FULL_REPLAY=0
export GLM53F_SPEC_REFERENCE_IDS=$HOME/work/gemm/q38nf/tmp/glm53f-q4-51370956/mtp-selfcheck8k.ids.greedy
# Keep these overrides to use the measured executable. A rebuild needs revalidation.
GLM53F_BUILD=0 GLM53F_SPEC_BINARY=$PWD/glm53f_spec_q4_repeat \
  bash run_glm53f_q4_mtp_12n.sh \
  "$HOME/work/gemm/q38nf/tmp/glm53f-codegen-51336165/prompt_ids_header.txt" \
  "$HOME/work/gemm/glm53f/tmp/continuation.ids" 7189 1 7189
```

The later approximation experiments do not supersede this exact result.
They passed the paired Python score-loss gate but failed C++ compilation and
long scalar/MTP equivalence. Their faster raw timings are not promoted; see
[GLM53F_APPROX_12N.md](GLM53F_APPROX_12N.md).
