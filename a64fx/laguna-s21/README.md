# Laguna S-2.1 runner (12-node A64FX, expert-parallel)

48 layers, hidden 3072, GQA 8x128, 256 experts top-10 with a shared expert and a
dense layer 0. Experts are EP-sharded (rank owns expert `e` where `e % N == rank`);
attention, dense MLP, shared expert, router, embedding and lm_head are replicated,
so the only per-MoE-layer communication is one allreduce over the routed partial.

Performance and the reasoning behind each optimization: **`fp8-optimization.md`**.

## Build

```
make            # int4 production build + stager
make fp8        # bf16 checkpoint quantized at load: int8 linears + int8-per-block experts
make bf16       # pure-bf16 reference
make fp8-kvfp16 # experimental FP8-weight build with IEEE FP16 KV
```

For the production FP8 shape, use
`run_laguna_s21_fp8_best.sh`: it selects the FP8 checkpoint and runner while
leaving the allocation rank count (`PJM_MPI_PROC`, normally 12) explicit. The
recommended six-hour llmgr job, including submission from the Mutagen-synced
workstation, frontend pinning, tunnels, and the control API, is in
[`../llmgr/RUNBOOK.md`](../llmgr/RUNBOOK.md). Keep that as the operational
source of truth; this file documents the runner itself.

## Configuration is by flag, never by environment

Every knob is a command-line argument. The single exception is the MPI rank, which
`mpiexec` supplies through the environment and for which there is no CLI channel
(`--rank` overrides it for single-process testing).

## Stage the weights node-local

```
./run_laguna_s21_12n.sh stage --fp8 [--np 12] [--model-dir DIR] [--stage-dir DIR] [--nshards N]
```

Or directly:

```
mpiexec -np 12 build/laguna_s21_stage --model-dir ~/models/laguna-s21-fp8 \
    --stage-dir /local/$USER/laguna-s21-fp8-ep12 --ep-size 12 --nshards 24
```

## One-shot generation

```
./run_laguna_s21_12n.sh generate --fp8 --no-stage --chat "What is the capital of France?"
./run_laguna_s21_12n.sh generate --fp8 --no-stage --prompt "The A64FX processor" --max-new 200
```

For long C++ answers, the opt-in quality workflow reserves 4096 tokens, disables
visible thinking so reasoning cannot consume the answer budget, compiles and runs
the first fenced C++ program with a 20-second timeout, and makes one feedback
repair turn if validation fails:

```
./run_laguna_s21_12n.sh generate --fp8 --no-stage --quality-cpp \
    --chat "Write a complete C++20 bounded blocking queue with tests" \
    --sample --temp 0.7 --top-p 0.95
```

**Security:** `--quality-cpp` executes model-generated code on the head node. Use
it only for trusted prompts in an isolated job. Without that explicit flag no
generated code is compiled or executed. Set `CXX` to select the compiler. The
initial answer is retained as `gen.initial.ids` when a repair is needed; the
validated repair becomes `gen.ids`.

The normal runner stores KV as BF16. `--kv-fp16` selects an experimental FP16
KV build for the FP8-weight variant; it has the same memory footprint:

```
./run_laguna_s21_12n.sh generate --fp8 --kv-fp16 --no-stage \
    --chat "Write a thread-safe C++ LRU cache" --max-new 2048 \
    --sample --temp 0.7 --top-p 0.95 --seed 305441741
```

In a fixed-seed 12-node C++ generation A/B, BF16 KV reached the 2048-token
limit at 24.8 tok/s and generated a correct LRU implementation (one bad test
assertion). FP16 KV stopped naturally at 1469 tokens at 25.4 tok/s, but failed
the core LRU recency behavior by using a shared/read lock in `get()` without
updating recency. The sampled streams diverged at the second generated token.
Treat FP16 KV as an experiment, not a quality upgrade; BF16 remains the default.

`--chat` renders the checkpoint's own `chat_template.jinja`; `--prompt` does raw
continuation. An instruction-shaped prompt without `--chat` is out of distribution
for this model and will ramble. `--system TEXT` and `--no-think` are available.

Runner flags: `--ids FILE --max-new N --maxpos N --layers N --stage-dir DIR
--gen-out FILE --pchunk N --topo PATH --debug --sample --temp --top-k --top-p
--min-p --seed` and, in the fp8 build, `--fp8-exact` (keep the exact e4m3 kernels
instead of the int8-per-block re-quantization, for A/B).

llmgr accepts the same experiment as `kv_fp16: true`, or
`llmgr_cli.py start --variant fp8 --kv-fp16 ...`; it rejects the flag for
non-FP8 variants and records the dedicated runner binary.
For one-shot C++ generation it also accepts `quality_cpp: true`, exposed as
`llmgr_cli.py start --mode generate --quality-cpp ...`.

Greedy is the default so runs are reproducible; the checkpoint's
`generation_config.json` asks for `do_sample=true, top_k=20`, which `--sample`
follows.

## HTTP serving

```
./run_laguna_s21_12n.sh serve --fp8 --no-stage --port 8080 --maxpos 32768
```

For a repeated system prompt, add a persistent KV prefix cache:

```sh
./run_laguna_s21_12n.sh serve --fp8 --no-stage --port 8080 --maxpos 32768 \
  --system "You are a senior C++ reviewer." \
  --prompt-cache /path/to/laguna-cpp-system.lpc
```

The same cache option works for repeated one-shot chat generations:

```sh
./run_laguna_s21_12n.sh generate --fp8 --no-stage \
  --chat "Review this implementation..." --system "You are a senior C++ reviewer." \
  --prompt-cache /path/to/laguna-cpp-system.lpc
```

The cache path must be on a filesystem shared by all ranks. The first run
prefills the system block and atomically serializes its live KV state; later
starts deserialize it. Rank 0 uses the named file and the other EP ranks use
`.rankNN` companions, preserving the small rank-specific differences caused by
collective reduction order. Each admitted HTTP slot restores the immutable
snapshot before computing the user/assistant suffix. Requests must begin with the
same tokenized system prefix (use matching `--system` and thinking mode in
`tools/laguna_cli.py`); mismatches receive HTTP 400. `GET /health` reports
`prompt_cache_tokens`, and generation responses report `cached_prompt_tokens`.

Cache files are checksummed and bound to the checkpoint identity, layer count,
runner weight variant, and BF16/FP16 KV format. A corrupt or incompatible cache is
rejected rather than silently reused. To replace a system prompt, use a new cache
path or remove the old file. The direct runner interface also supports
`--prompt-cache-load FILE`, `--prompt-cache-save FILE`, and
`--prompt-cache-ids PREFIX.ids`; `--prompt-cache FILE` is the load-or-create alias.

`--maxpos` is required in serve mode: there is no prompt to size the KV cache from,
and it sets the largest context the server will accept.

Only rank 0 holds sockets. Every rank must run the identical sequence of forward
passes or their KV caches diverge, so the parsed request is broadcast to the whole
EP group before any compute starts — and **every** accepted connection reaches that
broadcast exactly once, including health checks and rejected requests, because the
other ranks sit blocked inside that collective whenever they are idle.

| endpoint | |
|---|---|
| `GET /health` | `{"status":"ok","ranks":12,"maxpos":32768,"layers":48,"prompt_cache_tokens":N}` |
| `POST /generate` | `{"ids":[...],"max_new":N,"stream":bool,"sample":bool,"temp":f,"top_k":i,"top_p":f,"min_p":f,"seed":u}` |
| `POST /shutdown` | stops every rank cleanly |

`/generate` replies with `{"ids":[...],"n":N,"stop":"eos"|"length","nan":N,
"lockstep_disagree":N,"prefill_tok_s":f,"decode_tok_s":f}`.

With `"stream":true`, the response is NDJSON: `start`, chunk-level `prefill`,
one `token` event per generated id, then `done` with stop and throughput fields.
A closed client is detected without SIGPIPE and its slot is retired only at a
collective-safe boundary. llmgr exposes this as OpenAI-compatible SSE.

On 12 nodes the launcher defaults to the K3-derived reliable transport settings
`--comm-robust 2 --comm-poll-spins 4` and a 2×6 hierarchical allreduce. Override
with `--ar-groups`, `--comm-robust`, and `--comm-poll-spins`; every reduction is
checked and terminates with the transport diagnostic after a collective failure.

Fast FP8 remains the production path and `--fp8-exact` is the reference path.
Capture the same prompt set from FP8-exact or BF16 and gate a candidate with:

```sh
python3 tools/quality_gate.py reference.jsonl fast-fp8.jsonl \
  --min-token-exact 0.80 --min-text-similarity 0.95
```

The runner has no tokenizer, so the API is ids-in/ids-out. `tools/laguna_cli.py`
does the chat templating and tokenisation client-side:

```
export LAGUNA_TOKENIZER=~/models/laguna-s21-fp8/tokenizer.json
python3 tools/laguna_cli.py --port 8080 chat "What is the capital of France?" --no-think
python3 tools/laguna_cli.py --port 8080 complete "The A64FX processor" --max-new 200
python3 tools/laguna_cli.py --port 8080 health
python3 tools/laguna_cli.py --port 8080 shutdown
```

## Long context

Everything that grows with context is known in closed form before a byte is
allocated, so an over-large `--maxpos` is refused up front with the largest value
that would fit, rather than being discovered by the OOM killer part way through a
20-minute prefill. Rank 0 prints the budget and, once loaded, where the memory went.

Per position only the full-attention layers' KV grows (~49 KB/token at 48 layers);
sliding layers are ringed at `LAGUNA_SLIDING_CAP` and cost a constant.

Generate reproducible retrieval-plus-C++ prompts at or just below a context
budget with:

```sh
export LAGUNA_TOKENIZER=~/models/laguna-s21-fp8/tokenizer.json
mkdir -p long_context_tests
python3 tools/make_long_context.py --target 16384 \
  --out long_context_tests/prompt_16k.ids \
  --metadata long_context_tests/prompt_16k.json
python3 tools/make_long_context.py --target 32768 \
  --out long_context_tests/prompt_32k.ids \
  --metadata long_context_tests/prompt_32k.json
# The same command supports 64K and larger budgets, subject to the printed
# per-rank context-memory check.

./run_laguna_s21_12n.sh generate --fp8 --no-stage --np 12 \
  --ids "$PWD/long_context_tests/prompt_32k.ids" --max-new 2048 \
  --sample --temp 0.7 --top-p 0.95 --seed 305441741 --prof
```

The fixture places three binding values near 1/8, 1/2, and 7/8 of the prompt,
then asks for a complete C++20 program using all three. Validate both retrieval
and executable correctness with `tools/cpp_quality.py gen.ids --run`.

## Tests

```
make test                                   # ABI + kernel self-tests, all builds
fcc ... -o slide_attn_test slide_attn_test.c && ./slide_attn_test   # query-blocked vs per-token attention
fcc ... -o run_prim_test  run_prim_test.c  && ./run_prim_test       # qk/av run kernels vs the originals
fcc ... -o sampler_test   sampler_test.c   && ./sampler_test        # top-k/top-p/min-p/temperature/seed
LAGUNA_TOKENIZER=... python3 tools/tok_test.py                      # added tokens, round-trip, chat template
python3 tools/repetition.py gen.ids                                 # degeneration metrics for long output
python3 tools/cpp_quality.py gen.ids --run                           # explicit compile/runtime validation
make bench && OMP_NUM_THREADS=47 ./build/decode_attn_bench 32768 100 # single-token attention timing
fcc ... -o build/run_prim_test run_prim_test.c -lm && ./build/run_prim_test # scores, fused max, AV
```

Benchmarks used to justify the kernel choices — `fp8_dq_bench.c`, `fp8_mm_bench.c`,
`i8_mm_bench.c`, `attn_bench.c`, `attn_parts_bench.c`, `svfloor_bench.c` — are
described in `fp8-optimization.md`.

## Gotchas

- **BOS (id 2) is required.** Without it the model degenerates to copying the last
  token. `--chat` gets it from the template; `--prompt` adds it explicitly.
- **`eos_token_id` is `[2, 24]`** and 24 is `</assistant>`, so chat answers
  terminate on their own.
- **Do not set `FLIB_BARRIER=HARD`.** It forces the OpenMP runtime to 48 threads,
  oversubscribing all 48 cores, and ~4x-slows the matvec kernels.
- **Leave one core free** (`OMP_NUM_THREADS=47`); 48 pinned threads on 48 compute
  cores costs ~40% for the general weight/communication path. Long full-attention
  decode automatically uses a 48-thread region because it has exactly 48 heads.
- **`XOS_MMM_L_PAGING_POLICY=demand:demand:demand`** — the default prepage policy
  collapses multi-CMG bandwidth (94 GB/s vs 843).
- Never compare timings across allocations; run both binaries back-to-back.
