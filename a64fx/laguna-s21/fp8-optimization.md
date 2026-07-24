# Laguna S-2.1 fp8 build: prefill + decode optimization

## Results (12 nodes, same allocation, A/B against the unmodified HEAD binary)

| | baseline | optimized | |
|---|---|---|---|
| decode, 6-tok prompt | 17.5 tok/s | **27.8** | **+59%** |
| decode, 2377-tok prompt | 11.5 tok/s | **19.2** | **+67%** |
| prefill, 2377 tok | 39.8 tok/s | **56.2** | **+41%** |
| weight load | 281.2 s | **75.1** | **3.7x** |

`nan=0` everywhere; the prefill argmax matches the baseline on every prompt tested,
and a 26.6k needle-in-a-haystack is retrieved correctly.

Prefill phase breakdown (seconds, 2377 tokens):

| phase | baseline | optimized | |
|---|---|---|---|
| attn | 21.4 | **9.3** | -57% (query tiling + run kernels + sliding flash) |
| shared/mlp | 9.1 | 9.2 | |
| qkv | 8.1 | 8.2 | compute-bound, int8 doesn't help |
| o_proj+rms | 8.6 | **5.0** | -42% (L2 token-blocking) |
| expert | 7.0 | **6.4** | -9% |
| router | 5.4 | **4.1** | -24% (batched router GEMM) |

No phase dominates any more; attention went from 39% of prefill to 22%.

The fp8 build (`make fp8`, `~/models/laguna-s21-fp8`) is bf16 non-expert linears +
fp8-e4m3 routed experts with 128x128 bf16 block scales. This documents what
actually costs time in it on A64FX, and what the measurements said about each fix.

All kernel numbers below are from `fp8_dq_bench.c`, `fp8_mm_bench.c` and
`i8_mm_bench.c` in this directory, run on a native A64FX node.

## Where decode time goes

The e4m3 kernel is not the whole story. Per token per rank at 12 nodes:

| what | bytes read | note |
|---|---|---|
| non-expert linears (q/k/v/o/g, dense, shared, router, lm_head) | ~7.2 GB bf16 | **replicated on every rank** |
| routed experts | ~0.4 GB fp8 | only ~10/12 of an expert is rank-local per layer |

So the bf16 linears dominate decode bandwidth by ~18x, which is why the int4
build (int8 linears) reached 20 tok/s while fp8 sat at 17.2. The fp8 build now
quantizes its linears to int8 at load exactly as the int4 build does.

## The e4m3 dequant: four variants measured

`laguna_matvec_fp8blk` widened bytes with an SVE **gather** from a 256-entry f32
LUT. A64FX gathers are slow enough that the kernel ran at ~1.2 MAC/cycle/core --
8 gathers per 128 MACs. Variants (1024x3072, 12 threads, realistic weights):

| variant | MAC/cycle | note |
|---|---|---|
| `gather` (LUT) | 13.3 | the original |
| `bitadd` `(b&0x7f)<<20 + (120<<23)`, sign OR'd | 21.3 | exact for normals |
| `bitmul` `(b&0x7f)<<20` reinterpreted `* 2^120` | 20.4 / **2.9** | see trap below |
| `i8blk` (re-quantize to int8 per 128-col block) | 28.3 | one `svaddv` per block |
| **`i8blk` + lane-wise scale folding** | **47.4** | chosen, **3.6x** the original |

**The cross-lane reduction was half the remaining cost.** Applying the per-block
scale as `sum_cb s_cb*addv(a_cb)` puts a long-latency `svaddv` in the dependency
chain once per 128-col block -- 24 of them per 3072-col row. Since
`sum_cb s_cb*addv(a_cb) == addv(sum_cb s_cb*a_cb)`, each block can instead fold
its scale into a running *vector* accumulator with one `svmla_n` and the reduction
happens once per row. Worth 1.67x on its own. Difference is reassociation only
(1.5e-7).

**Subnormal trap.** `bitmul` is the mathematically pretty variant -- reinterpret
and scale by 2^120 handles e4m3 subnormals exactly, because an e4m3 subnormal
lands on an f32 subnormal that the scaling corrects. But A64FX traps subnormal
FMUL operands to microcode: on weights containing subnormals it ran at 0.26
MAC/cycle, **12x slower than the gather it was meant to replace**. It only looks
fast (20.4) on data with no subnormals. Never let a subnormal reach an A64FX
multiply on a hot path. `bitadd` sidesteps this entirely by biasing in the
integer domain, at the cost of approximating subnormals as `(1+m/8)*2^-7`
(<2e-5 of block max -- immaterial).

## Why int8, when e4m3 is already 1 byte

int8 and e4m3 are both 1 byte, so this buys **issue rate, not bandwidth**: the
widen becomes `ld1sb + cvt` (2 ops) instead of `ld1ub + gather` or the 6-op bit
dance. It slightly *increases* footprint, because the block scale goes from one
bf16 per 128x128 block to one f32 per (row, 128-col) block (+3%).

**Accuracy.** Measured against the ORIGINAL pre-fp8 weights (the yardstick that
matters -- not against e4m3, which is itself lossy):

| | relative dot-product error |
|---|---|
| exact e4m3 | 2.63e-2 |
| e4m3 via `bitadd` | 2.63e-2 |
| **fp8 -> int8 per 128-block** | **2.74e-2** |

The conversion adds ~4% on top of what fp8 quantization already lost, because
int8 with a per-128-block scale resolves a near-Gaussian block more finely than
e4m3's 3 mantissa bits. Set `LAGUNA_FP8_EXACT=1` to keep the exact e4m3 kernels
for A/B.

Note the yardstick matters: measured against the *e4m3 values*, the conversion
looks like 7e-3 of "error" -- but that comparison charges int8 for disagreeing
with an already-wrong reference.

## Prefill: dequant once per row, not per token tile

`laguna_matmat_fp8blk` already dequantized each weight row ONCE into an f32
`wrow` scratch and reused it across the chunk's tokens. A naive int8 batched
kernel that re-widens per 8-token tile **loses to it** as N grows (0.68x at
N=128) because it re-converts every row N/8 times. `laguna_matmat_i8blk` uses
the `wrow` structure, and then wins:

| tokens/expert N | fp8 wrow | i8blk wrow | speedup |
|---|---|---|---|
| 4 | 35.0 GMAC/s | 32.9 | 0.94x |
| **10** (real value at PCHUNK=256) | 45.4 | **59.6** | **1.31x** |
| 32 | 74.1 | 76.2 | 1.03x |
| 256 | 106.3 | 96.1 | 0.90x |

At `LAGUNA_PCHUNK=256` and top-10 of 256 experts, a rank-owned expert sees
~10 tokens per chunk, so N=10 is the operating point.

## Prefill: the batched linears were L2-thrashing

`laguna_matmat_i8` parallelizes over rows, and each row streamed all C tokens of
X. The live X slice is `C*cols*4` bytes; the CMG's L2 is 8 MB. Shapes with wide
`cols` blew past it:

| shape | X at C=256 | before | after token-blocking |
|---|---|---|---|
| q_proj 9216x3072 | 3.1 MB (fits) | 108 GMAC/s | 110 (unchanged) |
| **o_proj 3072x9216** | 9.4 MB | **66** | **142 (2.13x)** |
| kv_proj 1024x3072 | 3.1 MB | 125 | 127 (unchanged) |
| dense_down 3072x12288 | 12.6 MB | 34 | 42 (1.27x) |

Fix: process tokens in blocks of `TB = 4MB/(cols*4)` so the X block stays L2
resident for the whole row sweep. Narrow shapes get `TB=C`, i.e. the original
loop. **Bit-identical** -- each output's summation order is unchanged, only the
order outputs are produced in. q_proj and o_proj have identical FLOPs, so the
2.13x gap between them was pure cache behaviour.

## Memory: dropping the arena

`stage_load` copied the whole blob into an anonymous arena for NUMA-local
first-touch. That is required only when weights are used **in place**. In the fp8
build every hot tensor is now re-quantized at load into its own allocation, whose
parallel-over-rows fill does the NUMA first-touch itself -- so the arena is pure
waste, and a 17.8 GB one at that. `stage_load(..., use_arena=0)` points
`s->blob` straight at the mmap. Without this, the int8 copies (~13 GB) plus the
arena would not fit in 31 GB.

Only `LAGUNA_FP8_EXACT=1` still wants the arena (it reads e4m3 bytes hot).

## Other changes

- Prefill routing was a per-token `laguna_lin_mv` -- C separate matvecs, i.e. 256
  OpenMP fork/joins per MoE layer. Now one batched router GEMM into `sc->crouter`.
- An expert's gate and up projections share their input, so they run in one
  parallel region via `laguna_matvec_i8blk_multi` (mirrors `laguna_matvec_i8_multi`).

## Allocation granularity was worth more than every kernel change combined

The single largest effect in this whole exercise was not a kernel. Handing the
re-quantized weights out of ~3000 separate `posix_memalign` blocks instead of a
few large mappings cost **13.7 vs 27.1 tok/s** -- a 2x swing, entirely invisible
to the microbenchmarks (which reuse one small resident array). Decode streams
~13 GB of weights per token across 47 threads on 4 CMGs; only big contiguous
mappings get backed by large pages, and at that footprint TLB behaviour dominates
everything else. Weights now come from `qalloc`, a bump allocator over 1 GB
anonymous chunks.

The A/B that localized it (all on one allocation, so the numbers are comparable):

| build | decode |
|---|---|
| baseline HEAD (bf16 linears, e4m3 experts, blob arena) | 17.5 |
| `LAGUNA_FP8_EXACT=1` (int8 linears, e4m3 experts, blob arena) | 19.6 |
| default, per-tensor `posix_memalign` | 13.7 |
| default, `qalloc` chunks | **27.1** |

The middle row is what proved the int8-linear change was a win (+12%) and that
the regression lived entirely in the expert path's allocation.

**Do not trust a kernel microbenchmark to predict a footprint-bound change.**
Two other hypotheses were tested and refuted along the way before this one:
that the regression was context-length (partly true -- the historical 17.2 tok/s
baseline used a 6-token prompt, so it was never comparable to a 2377-token run),
and that it was page-cache pressure from the retained mapping (the blob release
below is still correct, but it changed decode by 0.0 tok/s).

Related lesson already in the repo's history: never compare across allocations.
The historical "17.2 tok/s" could not be used as a baseline at all; every number
in this document comes from binaries run back-to-back on one allocation.

## Not done / open

- `laguna_matmat_bf16` has the same L2-thrashing shape problem as
  `laguna_matmat_i8` did; the bf16 reference build would benefit from the same
  token-blocking.
- `dense_down` (3072x12288) is still only 42 GMAC/s. It is layer 0 only, so the
  whole-model impact is ~1/48, but 2D (row x token) blocking would help it.
- The async allreduce is still a stub (`ar_thread_start`); comm-overlap was tried
  and reverted in c4ec762b.

## Long context

Measured at 26.6k tokens (needle-in-a-haystack prompt), before vs after the
attention work below. NB these two rows are *both* the optimized weight path --
the HEAD baseline was never run at this length, so this isolates the attention
changes only:

| 26.6k tokens | before attn work | after | |
|---|---|---|---|
| prefill | 17.2 tok/s | **31.5** | **+83%** |
| attn phase | 1170.2 s | **476.0** | **2.46x** |
| decode | 2.9 tok/s | **4.4** | **+52%** |

Needle retrieved correctly in both, `nan=0`, identical prefill argmax.

Attention was **75% of prefill** at this length (1170 s of 1552 s), so it was the
only thing worth optimizing there; it is now 56%. Three changes, in increasing order of payoff:

**1. Query-blocked sliding attention.** The 36 sliding layers ran `attention_core`
per token, so a 256-token chunk made 256 rank-1 passes over a 512-key window whose
neighbours overlap by 511/512. `attention_slide_flash` blocks over key positions
and sweeps the chunk's queries inside each block. This needed the sliding ring
raised from 512 to `LAGUNA_SLIDING_CAP` 768, because at cap==window a chunk's own
K/V writes clobber slots its earlier queries still need.

Worth only ~7% of the attention phase, for an instructive reason: a 512-key window
is 2 MB of KV, which **already fits the 8 MB L2**. The redundant re-reads were L2
hits, so sliding attention is compute-bound, not bandwidth-bound. (The same
reasoning does *not* apply to full-attention layers, whose KV grows without bound.)

Note this also fixed a latent bug: `attention_core` derived the attended range from
the ring *capacity* (`lo = pos-cap+1`), which was only correct while cap happened
to equal the window. Raising the ring would silently have widened attention.

**2. Run-based qk/av primitives.** `laguna_qkdot` accumulated 8 SVE FMLAs into one
register -- a serial chain at ~9 cycles each -- and `laguna_vaxpy` read *and wrote*
all 128 floats of the accumulator per key, 1 KB of traffic per 256-byte V row.
`laguna_qk_run` / `laguna_av_run` keep q and the accumulator in registers across a
contiguous run of key slots (the ring gives at most two runs). Worth ~11%.
`av_run` is bit-exact vs the old loop; `qk_run` differs by <1e-7.

**3. Query tiling -- the real win.** The flash loops parallelized over heads alone:
48 heads on 47 threads is `ceil(48/47) = 2` rounds with the second round 1/47
utilized. Measured, **24 threads were as fast as 47** (97.0 vs 104.7 GFLOP/s) --
the signature of exactly this, since 24 threads also take 2 perfectly-packed
rounds. Parallelizing over (head, query-tile) with `LAGUNA_QT=32` gives
`nh*ceil(C/32)` tasks so no thread idles.

Full-attention layer throughput, C=256 (`attn_bench.c`):

| context | HEAD | +run kernels | +query tiling |
|---|---|---|---|
| 2048 | 91.6 | 100.6 | **231.2** GFLOP/s |
| 8192 | 102.4 | 104.5 | **278.9** |
| 32768 | 99.0 | 103.8 | **276.9** |
| 65536 | 92.9 | 103.6 | **255.7** |

**2.7x** at 65k. `LAGUNA_KB` (key block) turned out not to matter at all
(101-105 GFLOP/s across 32..256) -- it was never a cache-blocking problem.
`LAGUNA_QT=32` is the sweep optimum; 8 and 64 are both worse.

### Long-context capacity and stability

- Dropping the 17.8 GB blob arena roughly **doubles the context that fits**. Per
  position the KV is 48 KB (12 full-attention layers x 4 KB; sliding layers are
  ringed at 768 and cost nothing that grows). 128k = 6.3 GB KV + 13.4 GB weights
  ~= 20 GB of 31 GB, comfortable; 256k ~= 26 GB now fits where before it could not.
- 26.6k needle-in-a-haystack retrieves correctly with `nan=0` end to end.
- KV/rope/scratch allocations are now checked and report what was being attempted;
  previously an over-large context segfaulted inside attention. Rank 0 logs the KV
  size so headroom is visible before the run commits to it.
- `_Static_assert` ties `LAGUNA_SLIDING_CAP` to `LAGUNA_PCHUNK`, so raising the
  chunk size past the ring is a build error rather than silent corruption.

## Generation correctness

Three fixes, in order of how badly they could bite:

**1. Token choice is made on rank 0 and broadcast.** Every rank computes the full
logits, but they are not bit-identical -- the routed partial is combined by a tree
allreduce, so ranks can differ in the last ulp. With each rank running its own
`argmax`, a single near-tie makes two ranks emit *different* tokens, after which
their KV caches diverge and the run is silently corrupt. The chance of hitting one
grows with output length, so long generations are exactly where it bites.

One 2-element allreduce does the broadcast and detects disagreement at the same
time (slot 0 = rank 0's pick, zero elsewhere; slot 1 = sum of all picks, which
equals `N * rank0` iff everyone agreed). Every run now reports which it was.

Measured: **ranks agreed on all 1025 / 751 / 401 picks** in the runs tested, so
this was latent rather than active for greedy. It is what makes `--sample` safe at
all, though: independent per-rank RNG draws have no reason to agree. Cost is one
tiny allreduce per token against the 47 the MoE layers already do -- 26.0 -> 25.5
tok/s, ~2%.

**2. Sampling.** The checkpoint's `generation_config.json` asks for
`do_sample=true, temperature=1.0, top_k=20, top_p=1.0, min_p=0.0`, but the runner
only ever did greedy argmax. `--sample` follows the checkpoint's configuration;
`--temp/--top-k/--top-p/--min-p/--seed` override it. Warper order matches
HuggingFace (temperature -> top_k -> top_p -> min_p).

Greedy remains the default because it is deterministic and is what every
measurement in this document used. Verified by `sampler_test.c`: top-k support,
draw frequencies matching the reference softmax to <2%, monotonic P(top) in
temperature, top-p nucleus and min-p truncation, and seed determinism.

**3. eos is a stop signal, not output.** The loop appended the token and *then*
broke, so an eos landed in `gen.ids` and was decoded into the text. It now stops
before emitting, and reports `stopped on eos after N tokens`.

### Does long output actually degenerate?

No -- measured, not eyeballed (`tools/repetition.py`):

| 1024-token generation | distinct-2 | distinct-3 | max repeat run |
|---|---|---|---|
| `--sample` (top_k=20) | 0.830 | 0.955 | 0 |
| greedy | 0.753 | 0.885 | 0 |

No hard loops in either, and both tails are coherent (greedy reached a natural
conclusion and stopped on eos at 750 tokens). Sampling is measurably less
repetitive, which is the effect the checkpoint's `do_sample=true` is asking for,
but greedy does not collapse -- so the earlier hypothesis that long output was
degenerating was wrong.

**Known gap:** the runner feeds raw token ids, so it does continuation, not chat.
The checkpoint ships a `chat_template.jinja` (a thinking-model template with
system/user/assistant blocks). Prompts that read like instructions
("The capital of France is") therefore get an out-of-distribution continuation
rather than an answer. Applying the template is the next correctness step and is
independent of everything above.

### Remaining levers

- Attention is still the largest single prefill phase and now runs at ~250-280
  GFLOP/s. The next structural step would be blocking the *diagonal* (within-chunk
  causal) pass of `attention_full_flash`, which is still one key at a time.
- `dense_down` (3072x12288) sits at 42 GMAC/s vs q_proj's 110; it is layer 0 only
  (~1/48 of the model) so the payoff is small, but 2D (row x token) blocking would
  fix it.
- The async allreduce is still a stub (`ar_thread_start`); comm-overlap was tried
  and reverted in c4ec762b. Decode comm is ~25% at short context but only ~4% at
  26.6k, so this matters least exactly where throughput is worst.
- `laguna_matmat_bf16` has the same L2-thrashing shape problem `laguna_matmat_i8`
  had; the bf16 reference build would benefit from the same token-blocking.
