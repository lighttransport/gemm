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

### Chat template

The runner used to feed raw token ids, so it did continuation, not chat, and an
instruction-shaped prompt got an out-of-distribution ramble:

```
    "What is the capital of France?"
    before: "You are a bot that can perform various tasks. Your task is to
             respond to the user's question and acknowledge the karma points."
    after:  "</think>The capital of France is Paris."   (stops on eos, 8 tokens)
```

`laguna_tok.py chat` renders the checkpoint's own `chat_template.jinja` with
jinja2 rather than reimplementing it, so it cannot drift. The template's
`{% generation %}` / `{% endgeneration %}` tags are a HuggingFace-only extension
for loss masking that emit no text, so they are stripped to let plain jinja2 parse
it. Launcher flags: `--chat MSG`, `--system TEXT`, `--no-think`.

Two tokenizer bugs had to be fixed first, both invisible until the template
introduced markup:

- **`encode` ignored added tokens entirely**, BPE-ing them as ordinary text: the
  19-token control sequence above came out as 26 word pieces, and the model saw
  none of the control tokens it was trained on. Added tokens are now matched
  longest-first (so `〈|EOS|〉` beats its own substrings `〈|` and `|〉`).
- **`decode` byte-decoded added tokens.** BPE pieces are byte-level-encoded (each
  character stands for a byte) but added-token contents are literal text, so
  putting them through the same decode mangled anything non-ASCII -- `〈|EOS|〉`
  did not survive a round-trip. The byte buffer is now flushed around each added
  token.

`decode` also now hides only `special=True` tokens, matching HuggingFace's
`skip_special_tokens`. The chat markup (`<think>`, `</think>`, `<assistant>`) is
`special=False` and stays visible; `--raw` shows everything.

Worth noting: `eos_token_id` is `[2, 24]` and 24 is `</assistant>`, so the
runner's existing stop condition was already the correct chat turn terminator --
chat answers terminate on their own rather than running to `--max-new`.

`tools/tok_test.py` covers added-token mapping, longest-first matching, exact
round-trip (including unicode), decode visibility, and template rendering for
thinking / no-think / custom-system / multi-turn.

### bf16 loads: use the full vector width

`laguna_ld_bf16` used `svld1uh_u32`, a *widening* load that fills 16 f32 lanes
from only 32 bytes -- a whole load slot for half a vector, so head_dim=128 needed
8 of them per key. Reading 64 bytes at a time with `svld1_u16` and splitting with
`svunpklo/hi` moves the same bytes in half the slots. Each accumulator still meets
the same dims in the same order, so it is **bit-identical**.

Full-attention layer, C=256, 47 threads, median of 5 runs (ms/call):

| context | before | after | |
|---|---|---|---|
| 2048 | 54.6 | **43.2** | 1.26x |
| 8192 | 171 | **127** | 1.35x |
| 32768 | 710 | **479** | 1.48x |
| 65536 | 1476 | **1214** | 1.22x |

This helps every attention path at once (`qk_run`, `av_run`, hence full flash,
sliding flash and decode).

**Confirmed end-to-end** on 12 nodes, both binaries back-to-back on one
allocation (a fresh allocation and re-stage; `eb1ab88c` vs `5bda6d7f`):

| | before | after | |
|---|---|---|---|
| prefill, 2377 tok | 57.0 tok/s | **60.4** | +6% |
| **attn phase, 2377 tok** | 8.5 s | **6.5 s** | **1.31x** |
| decode, 2377-tok ctx | 19.4 tok/s | **21.2** | +9% |
| decode, 6-tok ctx | 27.9 tok/s | 28.0 | unchanged |
| prefill, 26632 tok | 31.7 tok/s | **36.4** | +15% |
| **attn phase, 26632 tok** | 435.6 s | **333.4 s** | **1.31x** |
| decode, 26632-tok ctx | 4.4 tok/s | **5.6** | +27% |

Generated tokens are **identical** at every length, as the bit-exactness implies,
and the 26.6k needle is retrieved by both binaries.
The 1.31x on the attention phase matches what the kernel benchmark predicted for
this depth (1.26-1.35x), so the microbenchmark and the real model agree here --
unlike the `svaddv` experiment below, where they did not.

Short-context decode is unchanged because it is bound by streaming the weights,
not by attention; the gain scales with context, which is the point.

### Two things that measured well in isolation and lost in context

Worth recording because both cost real time and neither shipped:

**A64FX `svaddv` costs ~35-40 cycles and does not pipeline.** Measured directly
(`svfloor_bench.c`): the same 8 loads + 8 FMLA run at ~10 cyc/key with the
accumulators left alone and ~50 with one `svaddv` per key. head_dim=128 is only 8
vectors, so a per-key reduction is amortised over 8 FMLAs rather than the 192 of a
3072-wide matvec -- which is why the identical trick paid off for the int8 expert
kernel earlier and why it looks so bad here.

Reducing 16 keys together with a zip tree (`uzp1+uzp2`, four stages, no `svaddv`)
is **1.48x on the isolated primitive** and numerically fine (<3e-7). In the real
kernel it was a **10-17% regression** -- the 16 live accumulators plus 8 hoisted q
registers spill. Reverted; only the load-width change shipped.

**Do not trust an isolated primitive benchmark for a register-pressure-sensitive
change.** Two further traps hit along the way, both self-inflicted:
- An early "floor" measurement consumed only 2 of 8 accumulators, so the compiler
  deleted 6 of the 8 FMLA chains and reported ~9 cyc/key that nothing could reach.
- A first A/B compared HEAD against HEAD, because a `cd` in a compound shell
  command persisted and both binaries were built from the same directory. It
  showed "no change", which was true but meaningless. Always assert the two
  binaries actually differ.

An earlier conclusion in this file that full-width loads "gave nothing" came from
that same confounding: the `svaddv` dominated the measurement so heavily that the
load width was invisible. It is worth 1.2-1.5x once the measurement is clean.

### The diagonal pass (done)

`attention_full_flash`'s within-chunk causal pass ran one key at a time, and it
was the worst-shaped loop in the file: a per-key `svaddv` inside `laguna_qkdot`, a
full-accumulator read-modify-write inside `laguna_vaxpy`, and two **scalar** `expf`
rather than the vectorised FEXPA path. It measured **67 GFLOP/s against the prefix
pass's 382**.

Its cost is `C^2/2` per head *regardless of context*, so a key-count estimate
badly understates it: at 6% of the keys but ~6x the cost per key it was 27% of the
kernel at pos0=2048 and 33% at pos0=1024. Blocking it the same way as the prefix
pass (`laguna_qk_run` + `laguna_exp_shift_sum` + `laguna_av_run` over `LAGUNA_KB`
key blocks) gives, median of 3 runs at C=256, 47 threads:

| pos0 | before | after | |
|---|---|---|---|
| 0 (pure diagonal) | 12.0 ms | **2.4** | **5.0x** |
| 256 | 16.6 | **6.6** | 2.5x |
| 1024 | 27.1 | **18.6** | 1.46x |
| 2048 | 42.6 | **35.2** | 1.21x |
| 8192 | 140.6 | 135.7 | 1.04x |
| 65536 | 1161 | 1080 | ~1.0x |

The shape matters more than any single number: throughput is now **flat at ~385
GFLOP/s across every depth**, where before it ramped from 45 at pos0=0 to 375 at
65k. The kernel no longer has a short-context penalty.

Only full-attention layers have a diagonal pass; `attention_slide_flash` already
processed its band through the run kernels.

**End-to-end** (12 nodes, both binaries back-to-back on one allocation,
`013e259b` vs the blocked version):

| | before | after | |
|---|---|---|---|
| prefill, 715 tok | 65.4 tok/s | **67.5** | +3.2% |
| attn phase, 715 tok | 1.2 s | **0.9 s** | 1.33x |
| prefill, 2377 tok | 62.6 tok/s | **64.4** | +2.9% |
| attn phase, 2377 tok | 5.9 s | **4.9 s** | 1.20x |
| decode | 24.6 / 22.8 tok/s | 25.0 / 22.8 | unchanged |

A large kernel win that is a small end-to-end one, and worth being explicit about
why: attention is only 11% of prefill at 715 tokens and 16% at 2377, and just 12
of 48 layers are full-attention. Decode is untouched because it processes one
query at a time -- there is no diagonal block to speak of.

Unlike the full-width-load change this is **not** bit-exact: block-wise online
softmax reassociates (~1e-6, see `full_attn_test.c`). Generated tokens were
identical at 2377 but diverged at 715, and the top-1 logit moved 27.705 -> 28.134
(1.5%), which is far more than 1e-6 and deserved checking rather than waving away.

**This runner is deterministic run-to-run** -- the same binary twice gives the
identical logit and token stream, so the shift is genuinely attributable to the
change and not to reduction-order jitter. (Worth knowing on its own: the comm
layer reports `deterministic=0`, so this could have gone the other way, and it is
the control to run before attributing any output difference to a code change.)

The amplification has a concrete mechanism in an MoE model: the router takes
top-10 of 256 experts, so a ~1e-6 perturbation can flip which expert lands tenth.
That is a *discrete* change in the computation, compounded over 48 layers. A 1.5%
logit move from a 1e-6 kernel perturbation is therefore expected here in a way it
would not be in a dense model.

"Both continuations look coherent" is weak evidence, so quality was checked with
sharp pass/fail prompts in exactly the regime this change affects (short prompts,
where the chunk is all diagonal). All three give **character-identical answers on
both binaries**: "The capital of France is Paris.", the 17-sheep riddle answered
correctly as 9, and the same three Japanese supercomputers. The divergence at 715
tokens was on a deliberately repetitive prompt, where near-ties are dense.

`full_attn_test.c` covers this path (the sliding path had `slide_attn_test.c` but
the full path had none): 20 cases over prefix/block alignments and chunk sizes,
with pure-diagonal cases exact to 0.0, plus a control that moves the causal cut by
one key and shows 1.5e-2 against a ~5e-6 reassociation floor.

## Long C++ quality at a 4096-token cap

Long answers need executable checks, not token similarity or repetition scores.
All runs below used 12 nodes, BF16 KV, seed 305441741, top-k 20, temperature
0.7, and top-p 0.95. The original 2048-token LRU run reproduced all 2048 tokens
exactly when its cap was raised, then reached EOS at 2386 tokens. This rules out
max-position-dependent drift; the old answer was simply truncated.

| expert path / workflow | output | decode | validation |
|---|---:|---:|---|
| fast INT8-q128, LRU | 2386 | 24.3 tok/s | compiled; bad size assertion aborted |
| exact E4M3, LRU | 1864 | 18.2 tok/s | compiled and stress test passed |
| experimental INT8-q32, LRU | 2560 | 23.8 tok/s | tests passed, but returned a dangling pointer |
| fast INT8-q128, blocking queue | 1691 | 25.0 tok/s | missing `<deque>` |
| exact E4M3, blocking queue | 1713 | 18.3 tok/s | missing `<deque>` and condition variables reversed |
| hidden audit prompt, queue | 2765 | 23.9 tok/s | correct synchronization, missing `<cassert>` |
| compiler-feedback repair | 2976 | 19.9 tok/s | warning-clean compile; all runtime tests passed |
| integrated `--quality-cpp`, queue | 4096 | 22.9 tok/s | compiled; 9 deterministic/runtime tests passed |

Exact E4M3 helped one prompt but failed the second, so it is not promoted as a
general quality mode. INT8-q32 reduced the synthetic expert matvec error from
0.702% to 0.566%, but semantic safety got worse; that implementation was rejected.
Visible chain-of-thought was also rejected as a default: it consumed the entire
4096-token budget without reaching the final answer.

The retained `--quality-cpp` workflow uses the fast production math, a no-think
audit system prompt, real compilation/runtime validation, and at most one repair
turn with the diagnostic. On the blocking-queue case, the repair added the missing
header and passed producer/consumer stress, timeout, close, and capacity-zero
tests. This costs another model load and generation only when the first answer
fails. It executes generated code and is therefore explicitly opt-in.
The integrated rerun used the full 4096-token cap: its fenced program was
complete and validated, while only the following prose explanation was cut off.

The repetition heuristic labeled every healthy code answer "DEGENERATE" because
syntax and identifiers naturally repeat. Distinct-n remains useful for obvious
language loops, but must not be used as a code-quality gate.

## Batched serving: the premise was wrong

Continuous batching was built on the assumption that decode is weight-bandwidth
bound -- ~13 GB read per token -- so K sequences stepped together would share one
pass over the weights and scale close to K x. **Measured, that is false on this
machine.** 150 tokens per stream, identical prompts:

| K | wall | aggregate | per stream |
|---|---|---|---|
| 1 | 9.93 s | 15.1 tok/s | 15.1 |
| 2 | 17.02 s | 17.6 | 8.8 |
| 4 | 30.27 s | 19.8 | 5.0 |

A K=4 step cost **3.05x** a K=1 step, not ~1x. Decode is far more compute-bound
than the bandwidth arithmetic suggested; batching amortises the weight traffic and
the 47 per-token allreduces, and that is worth only ~1.3x, not 4x.

Two separate problems were behind the poor showing, and only one is fixed:

1. **`laguna_matmat_i8` had no small-C path.** With C<8 it fell entirely into its
   1-token tail, which re-widens the whole weight row for *every* token and does an
   `svaddv` per (row, token) -- strictly worse than C separate matvecs. It now
   widens once into a scratch and reuses it, the same `wrow` trick already used by
   `laguna_matmat_i8blk`. (The kernel had only ever been exercised at C=256.)
2. **K=1 through the batched path is much slower than `forward_token`** (15.1 vs
   27.8 tok/s), because `forward_token` fuses q/k/v/g into one OpenMP region and
   uses matvecs rather than 1-column GEMMs. The serve loop now routes a lone
   request through `forward_token`, so a single client never pays for the batching
   machinery.

After both fixes, re-measured:

| K | aggregate before fixes | after |
|---|---|---|
| 1 | 15.1 tok/s | **23.6** |
| 2 | 17.6 | 16.0 |
| 4 | 19.8 | 18.4 |

The single-stream regression is repaired (15.1 -> 23.6, against 27.8 for the
`--generate` path, the rest being HTTP and per-request accounting). **Batching is
still a net loss**: 18.4 tok/s aggregate at K=4 against 23.6 for one stream.

**`--max-batch` therefore defaults to 1** (no batching). The mechanism is correct
-- four prompts served alone and concurrently give byte-identical output with zero
lockstep disagreements -- but correctness is not a reason to enable something that
measured slower. It is kept, off, because the machinery is verified and the
blocker is now a specific, findable one: a K-token step costs ~3x a 1-token step,
so whatever dominates decode here scales with tokens rather than with weight
traffic. Identify that first; batching only pays once it does not.

An honest note on how this went: the whole feature was justified by an arithmetic
estimate ("13 GB/token, therefore bandwidth-bound") that was never measured. The
scaling test that refuted it takes two minutes and should have come first.

## What actually dominates decode

Measured with `--prof` at 35.6 ms/token (28.1 tok/s), short context, plus a
thread-count sweep to separate real work from per-call overhead.

| phase | ms/token | share | GB/s | % of ~830 GB/s | 12->47 thread scaling |
|---|---|---|---|---|---|
| **allreduce** | ~10.4 | **29%** | -- | -- | -- |
| qkv+gate linears | 6.7-7.5 | 21% | 207 | 25% | **3.35x** |
| shared expert (compute) | ~5.9 | 17% | 94 | 11% | ~1.9x |
| o_proj | 4.1 | 12% | 302 | 36% | -- |
| routed experts | 3.3-3.8 | 10% | 98 | 12% | **2.0x** |
| router | 2.5 | 7% | 14 | 2% | **1.04x (flat)** |
| attention core | 2.3 | 6% | -- | -- | -- |
| lm_head | 1.2 | 3% | 263 | 32% | -- |
| rmsnorm | 0.8 | 2% | -- | -- | -- |

Three conclusions, in order of how much they are worth:

**1. Communication is the single largest item, ~29%.** 48 allreduces per token,
one per MoE layer plus the token broadcast, each 0.22 ms for a 12 KB payload.
That is latency, not wire: it is the same 0.22 ms whatever the thread count.

**2. Nothing is bandwidth-bound.** The best phase reaches 36% of HBM, most sit at
11-25%. This is the direct refutation of the batching premise: batching amortises
*bytes*, and bytes were never the constraint. It also means the earlier "13 GB per
token" figure was wrong -- that is the resident footprint; only **4.07 GB** is
actually read per token per rank, which at 27.8 tok/s is 113 GB/s, 14% of peak.

**3. ~10% of decode is OpenMP fork/join, and the router is the proof.** The router
costs 2.53 ms at 47 threads, 2.47 at 24 and 2.64 at 12 -- completely flat, while
qkv+gate scales 3.35x over the same range. It is a 256x3072 matvec (0.037 GB) that
does not need 47 threads or its own parallel region, and it runs 47 times per
token. The routed experts scale only 2.0x for the same reason at smaller scale.

Worth noting for prioritisation: the routed experts, which absorbed most of the
fp8 kernel effort in this document, are **10% of decode**. The dense linears
(qkv + o_proj + shared + router) are ~57%.

### Fewer OpenMP regions: +4% decode

A fork/join costs **13.2 us at 47 threads** on this machine (7.4 at 24, 4.8 at 12,
0.8 at 1 -- measured with an empty parallel region). Decode runs roughly 9 regions
per layer x 47 layers ~= 400 per token, i.e. **~5.4 ms of a 35.6 ms step**. That is
the "~10% fork/join" the profile inferred, now measured directly.

Two fusions, neither of which changes any arithmetic or its order -- only where the
region boundaries fall:

- **Router + shared-expert gate/up in one region.** All three read `n2`, are
  mutually independent and share `cols=HIDDEN`. This needs dedicated `sh_a`/`sh_b`
  scratch: the routed experts reuse `inter_a`/`inter_b` in between, so computing the
  shared gate/up early into the old buffers would have been silently clobbered.
- **One expert's whole SwiGLU in one region** (`laguna_expert_swiglu_i8blk`): gate
  and up with `nowait`, a barrier, the `silu*up` split across the team, then `down`.
  Was three regions.

| | before | after | |
|---|---|---|---|
| decode | 26.9 tok/s | **28.0** | **+4.1%** |
| router + shared (they trade attribution) | 16.11 ms | **14.89** | -1.22 ms |
| routed experts | 3.74 ms | **3.40** | -0.34 ms |

Generated tokens are **identical**. Confirmed again at a 2648-token prompt
(**21.9 -> 22.6 tok/s**, +3.2%, tokens identical, same argmax) where **prefill is
bit-for-bit unchanged at 62.6 tok/s** -- the useful negative control, since the
fusion touches only the decode path. A 22k-token needle retrieval under the fused
build returns the passcode verbatim with `nan=0`, which is what would break first
if the `sh_a`/`sh_b` aliasing had been gotten wrong.

The saving predicted from the region count was
1.7 ms/token and 1.5 was measured.

**That agreement was a coincidence, and the cost model it appeared to validate is
wrong.**  Acting on it -- one `omp parallel` per LAYER, taking decode from ~320
fork/joins per token to 47 -- was implemented, verified token-identical, and
measured a **regression**: 28.1 -> 27.7 tok/s (3 baseline runs at 28.1, two
refactor runs at 27.7; comm flat at 23.4% -> 21.6%, so it was not a comm effect).
It has been reverted.  Two things were wrong:

1. **13.2 us is the cost of an ISOLATED region, not the marginal cost of the next
   one.**  In the steady-state decode loop the Fujitsu runtime keeps workers
   spin-waiting, so entering the next region costs about a barrier rather than a
   thread wake-up.  The microbenchmark measured a wake-up that decode never pays.
2. **Collapsing regions ADDS synchronisation.**  Serial sections (rmsnorm,
   residual adds, top-10, the allreduce) previously ran between regions with the
   team parked and *zero* barriers.  Inside one region each needs `omp single` --
   a barrier in and a barrier out.  Per layer that traded ~6.8 fork/joins for
   ~19 barriers.

So the real reason the two fusions above won is NOT the region count.  It is that
the fused operands share their input (`n2`, and gate/up share `x`), so fusing them
reuses that input from cache and halves the number of times the weight stream is
restarted -- an arithmetic/locality effect that happens to scale with the same
count.  Fuse work that shares an input; do not fuse merely to remove a region.

### "Comm is 23%" is probably NOT comm -- it is expert load imbalance

Unverified on hardware yet (needs an allocation) but the arithmetic is strong, and
two diagnostics are now wired in to settle it.

Expert `e` lives on rank `e % N`, so per layer a rank owns `Binomial(top-10, 1/N)`
experts.  At N=12 that is mean **0.833** but **E[max over ranks] = 2.52** -- and
every rank blocks at the routed allreduce until the busiest one finishes, so the
3.02x spread is charged to "comm".  Calibrating against the measured profile
(rank 0: experts 3.47 ms/token, AR 8.33 ms/token, 47 layers):

| | |
|---|---|
| implied cost of one expert | 0.0886 ms |
| implied idle at the barrier (max-mean = 1.68 experts) | 0.149 ms/layer |
| measured allreduce | 0.177 ms/layer |
| **imbalance as a share of "comm"** | **~84%** |

Two supporting facts: the collective is already recursive-doubling with a
non-pof2 prefold (12 ranks = prefold + 3 rounds + bcast = 5 steps, near the
log2(12)=3.6 floor), and 12 KB at ~6.8 GB/s is ~2 us of wire against 177 us
measured.  There is no plausible way for the fabric to be the cost.

**How to test it (both need only an allocation, one needs no weights):**

- `--ar-probe` -- was dead code, defined but reachable from no CLI flag; now
  wired up.  Reports back-to-back vs barrier-synced allreduce latency with **no
  weights loaded**, so it needs no 17.8 GB stage.  Barrier-synced x 47 is the
  floor for honest comm; if that is far under 8.33 ms/token, the rest is skew.
- `--prof` now gathers per-rank expert counts and AR seconds and prints
  max/mean against the 3.02x prediction.  If AR time is anti-correlated with
  expert count across ranks, imbalance is confirmed.

**If confirmed, the fix is to shard experts, not to touch the collective.**  Give
each expert to a GROUP of G ranks and split its `inter=1024` G ways.  Memory per
rank, weight bytes read per token, and the allreduce (one [hidden] reduction) are
all unchanged -- only the balance changes:

| G | groups | imbalance | down-proj inner dim | |
|---|---|---|---|---|
| 1 (today) | 12 | 3.02x | 1024 | |
| 4 | 3 | **1.47x** | 256 | removes 77% of the skew |
| 12 (full TP) | 1 | 1.00x | 85 | removes 100%, but skinny |

Full TP balances perfectly but leaves the down projection with an inner dimension
of 85, which is a poor SVE kernel shape; G=4 keeps 256 and still removes most of
the skew.  Predicted (assuming imbalance is 84% of AR): **G=4 -> ~33 tok/s, G=12
-> ~35 tok/s, from 28.1**.  Treat those as upper bounds -- they ignore the
efficiency the smaller per-rank kernels will lose, which is exactly why G=4 may
beat G=12 in practice.  Measure before building the staging changes.

### Where to look next, in order

1. **The 48 per-token allreduces (29%).** Latency-bound at 12 KB. Async overlap was
   tried and reverted (`c4ec762b`: the comm thread is starved by the OMP runtime).
   Worth revisiting with a different mechanism -- e.g. reducing every other layer's
   partial together, which halves the count at the cost of holding one extra
   partial.
2. ~~Router fork/join~~ **done above (+4.1%)**. Do NOT pursue the remaining
   regions: one `omp parallel` per layer was tried and **regressed to 27.7**
   (see above) because collapsing regions adds barriers around every serial
   section. A `GLM5_PAR_MIN`-style work threshold is also not the answer -- the
   router is 786K MACs, far too much to run serially. Region count is simply not
   the lever it looked like.
3. **Small-matvec efficiency generally** -- shared expert at 11% of bandwidth and
   routed experts at 12% are the same problem one level up.

### Remaining levers

- `dense_down` (3072x12288) sits at 42 GMAC/s vs q_proj's 110; it is layer 0 only
  (~1/48 of the model) so the payoff is small, but 2D (row x token) blocking would
  fix it.
- The async allreduce is still a stub (`ar_thread_start`); comm-overlap was tried
  and reverted in c4ec762b. Decode comm is ~25% at short context but only ~4% at
  26.6k, so this matters least exactly where throughput is worst.
- `laguna_matmat_bf16` has the same L2-thrashing shape problem `laguna_matmat_i8`
  had; the bf16 reference build would benefit from the same token-blocking.
