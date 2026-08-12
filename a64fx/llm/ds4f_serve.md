# DS4F single-node serving for coding agents

Single-node OpenAI-compatible serving of the DeepSeek-V4-Flash model for
opencode / codex / claude-code.  Three processes:

```
libds4f_serve.so     (built by a64fx/llm/build_ds4f_serve.sh)
ds4f_serve_runner.py (persistent model loop; shared-FS protocol)
ds4f_serve.py        (OpenAI HTTP frontend; already in the tree)
```

For the 0731 checkpoint the frontend loads the authoritative
`encoding/encoding_dsv4.py` shipped beside `tokenizer.json`. Prompts therefore
use `<｜User｜>/<｜Assistant｜></think>` and DSML tool calls. The former generic
`System:/User:/Assistant:` renderer remains only as a fallback for model
bundles that do not provide the DS4F encoder.

## Build

```sh
sh a64fx/llm/build_ds4f_serve.sh          # -> ./libds4f_serve.so (ROCm if found)
```

The model must be staged for single-node ep_size=1 (in-place NOCOPY):

```sh
# one-time (~0.1 s, 156 GB referenced in place, no copy)
DS4F_EP_RANK=0 DS4F_EP_SIZE=1 \
DS4F_MODEL_DIR=/mnt/disk1/models/ds4f-0731 \
DS4F_STAGE_DIR=/tmp/ds4f_single DS4F_NSHARDS=48 DS4F_STAGE_NOCOPY=1 \
  /tmp/ds4f_stage   # build: gcc -O2 -I./common a64fx/llm/ds4f_stage.c
```

## Run

```sh
# terminal 1: the model runner (loads ~156 GB, ~2 min)
DS4F_SERVE_BASE=/tmp/ds4f_serve DS4F_STAGE_DIR=/tmp/ds4f_single \
DS4F_SERVE_USE_HIP=1 DS4F_MAXPOS=16384 \
python3 a64fx/llm/ds4f_serve_runner.py --daemon

# terminal 2: the OpenAI frontend
TOK=/mnt/disk1/models/ds4f-0731/tokenizer.json \
DS4F_SERVE_BASE=/tmp/ds4f_serve \
DS4F_SERVE_AGENT_CACHE_DIR=/tmp/ds4f_serve.agent-cache PORT=8080 \
python3 a64fx/llm/ds4f_serve.py
```

Smoke test: `sh a64fx/llm/test_ds4f_serve.sh /tmp/ds4f_single 8080`.

The production prefill path uses exact mHC/Tier-B2 tiles. The logical default
is 4096 prompt tokens and the current kernel capacity processes it as 512-token
tiles; set `DS4F_SERVE_PREFILL_TILE` to reduce the tile when memory is tight.
Run the fixed acceptance protocol with:

```sh
python3 a64fx/llm/ds4f_serve_bench.py --stage-dir /tmp/ds4f_single \
  --prompt-tokens 4096 --warm-decode 32 --decode-tokens 256
```

`DS4F_ROUTE_TELEMETRY=1` prints each layer's eight hottest experts and their
coverage when the process closes. `DS4F_MXFP4_W4A8=1` enables the opt-in CPU
expert activation-quantized path. Whole-layer raw expert streaming is retained
for experiments as `DS4F_SERVE_HIP_EXPERT_STREAM=1`; it is off by default.

## Context / KV management

- **Conversation prefix cache** (automatic in the frontend): the chat handler
  keeps the last conversation's token ids + KV snapshot (`<BASE>.conv`) and,
  when the next request extends it, asks the runner to restore the KV and
  prefill only the new tail (`cache_load`+`cache_save`) instead of
  re-prefilling the whole context.  A request that does not extend the cached
  conversation falls back to a fresh full prefill (the session is reset to
  position 0, so multi-turn conversations stay correct).  Measured: turn 2 of
  a conversation lands in ~8 s vs a full re-prefill.
- **Per-agent system-prompt cache** (`DS4F_SERVE_AGENT_CACHE_DIR`): the frontend
  writes durable, content-addressed KV prefixes under separate `opencode`,
  `codex`, and `claude-code` directories. Each sidecar records the exact token
  IDs, model, tokenizer, and cache schema; a frontend restart therefore reuses
  only an exact compatible prefix. The cache includes stable Responses API
  developer/repository context up to an exact tokenizer boundary before the
  final user payload. A new long snapshot extends an existing shorter system
  snapshot and is immediately reused by its triggering request. The
  cooperative Unix-socket runner carries cache load/save metadata in each
  request; `--agent-cache-max-tokens` controls the maximum prefix (14336 by
  default). A lone request uses
  `--single-prefill-quantum-tokens` (2048 by default), while competing requests
  retain the smaller fair-share `--prefill-quantum-tokens` quantum.
  `--hip-expert-cache-mb auto` can admit prompt-hot routed-expert bundles into
  otherwise-free VRAM before decode; `--hip-expert-cache-reserve-mb` retains
  scratch/KV headroom.
  `DS4F_SERVE_SYSCACHE` remains available for the runner's
  legacy single-file checkpoint/preload path.
- **Slots** (`DS4F_SERVE_SLOTS`): per-conversation KV snapshots switched by
  the `slot` field (`<BASE>.slot.<i>` files).
- **Cooperative multi-context serving** (single-node wrapper default): the
  frontend and runner use a Unix socket, admit independent requests between
  prefill/decode quanta, and keep the 156 GB weights loaded once. Responses API
  chains inherit a context through `previous_response_id`; Chat Completions and
  Anthropic Messages may supply `context_id` or `X-DS4F-Context-ID`.
- Completed contexts stay in RAM for 600 seconds, then atomically spill to disk
  for 24 hours. Defaults are 512 MiB RAM and 8 GiB disk. Configure the wrapper
  with `--context-memory-ttl-sec`, `--context-disk-ttl-sec`,
  `--context-memory-mb`, `--context-disk-mb`, `--prefill-quantum-tokens`,
  `--decode-quantum-tokens`, and `--scheduler-quantum-ms`. These are program
  arguments, not production tuning environment variables.
- With HIP enabled, `--hip-prefill-tuned 1` (the default) installs the same
  QKV/device-attention/O-projection, fused shared-FFN, routed-FFN, and gfx12
  WMMA callbacks used by the real-weight benchmark. The individual
  `--hip-*` arguments can disable or retune each component; no production
  tuning environment variables are required. `--hip-expert-stream 1`
  streams each layer's raw MXFP4 expert bank for large prompt tiles; disable
  it when PCIe transfer cannot amortize over the selected tile.
- After prefill, the cooperative runner stores an internal full-prompt snapshot
  before generation. New connections reuse the longest compatible prompt or
  completed-conversation prefix, including prior generated tokens. These
  internal entries share the same RAM/disk limits and TTLs as conversations and
  are omitted from the public `/v1/contexts` list.
- In-memory and spilled version-2 context images store only populated KV rows;
  version-1 full-capacity images remain import-compatible. Batched decode also
  updates mutable context images in place, avoiding duplicate input/output
  images on every generated token.
- `--decode-batch-size N` enables native multi-context decode batching. It
  batches dense, shared/routed-FFN, and head work while retaining independent
  KV/Tier-B2 cache sets per row. Active batch members remain in native cache
  slots across token steps and serialize only when admitted, completed, or
  stashed; cache set zero remains isolated for concurrent prompt admission.
  Slot cache capacity is reserved with lazy anonymous pages and touched pages
  are reclaimed when a slot is released, so RSS follows active context length
  instead of eagerly committing `max_pos` for every configured slot.
  Default is `1` because the current RX 9070 XT
  small-M kernels are slower at B=2 and the independent long-range caches use
  substantial host RAM; use B=2..4 for throughput experiments. `/v1/progress`
  reports actual batch steps and sequences.
- `--speculative-tokens N` enables the checkpoint's three-stage DSpark
  drafter for greedy requests. The staged manifest must include `mtp.*`
  tensors (`DS4F_STAGE_MTP=1` while staging). DSpark proposes in parallel,
  applies the rank-256 Markov bias in token order, and commits only the prefix
  accepted by the exact main-model verifier. Sampling, penalties, and native
  multi-context decode retain the ordinary path. The checkpoint was trained
  with a five-position block, so values above 5 are clamped. It remains
  opt-in (`--speculative-tokens 4`) because short EP1 prompts show low
  acceptance; pass 0 for the scalar baseline. Non-MTP stages automatically
  disable DSpark even when a nonzero value is requested.
  Version-3 context images include all three DSpark KV rings; older version-2
  contexts remain compatible when DSpark is disabled.
- `GET /v1/contexts`, `GET /v1/contexts/<id>`, and
  `DELETE /v1/contexts/<id>` expose and manage idle context state. The progress
  endpoint reports the active context's real prefill/decode counters.
- **Truncation**: the runner caps the prompt at `max_pos - max_new`, keeping
  the head (system prompt + tools) and the recent tail whole.
- The model's own sliding-window (128) + compressed long-range (tierb2)
  attention keep the per-token decode cost flat out to `DS4F_MAXPOS`.

## Tool calling

`ds4f_serve.py` injects tools into the plain-text model prompt and translates
the model's `<tool_call>{...}</tool_call>` result into each client's native
tool-call shape. Streaming holds a possible tool marker until it is complete,
so clients never receive the internal marker as visible assistant text.

## Coding-agent configs

Point an OpenAI-compatible client at `http://localhost:8080/v1`.

**opencode** (`~/.config/opencode/opencode.json`):

```json
{
  "provider": {
    "ds4f": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "DS4F local",
      "options": { "baseURL": "http://localhost:8080/v1", "apiKey": "none" },
      "models": { "ds4f": { "name": "ds4f", "tool_call": true } }
    }
  }
}
```

**codex** (`~/.codex/config.toml`):

```toml
model = "ds4f"
model_provider = "ds4f"
[model_providers.ds4f]
name = "DS4F local"
base_url = "http://localhost:8080/v1"
wire_api = "responses"
```

**claude-code**: point `ANTHROPIC_BASE_URL` at `http://localhost:8080` and use
the `/v1/messages` endpoint directly; no OpenAI-to-Anthropic bridge is needed.

## Performance notes

The numbers below are historical CPU-oriented measurements and are not a
substitute for the live ROCm benchmark. Use `a64fx/llm/bench_ds4f_http.py`
against the configured llmgr endpoint to measure the current full-weight
single-node path; it reports cold versus cached prompt work separately.

The dense bank is attached to the ROCm via the `hip_ds4f_dense` hooks (the
default `DS4F_SERVE_USE_HIP=1`; the CPU-only path is ~4x slower).  The routed
MXFP4 experts stay on the CPU.  Measured decode: **~240 ms/token** (2.3-4 tok/s)
with the tierb2 compressor on, ~176 ms with it off; the CPU experts are the
floor (the harness's best no-compressor decode is ~185 ms/token, and neither
the CUDA expert batch nor the ROCm MXFP4 path beats the CPU at M=1).  The
CUDA-expert async batch (hetero/ds4f) is the follow-on accelerator for
multi-token prefill, not the single-token decode.

The exact tiled server path measured 2.52 prefill tok/s at M=128 and 3.29
decode tok/s on the 16-core Threadripper/RX 9070 XT test host. W4A8 improved a
short warmed decode to 4.68 tok/s. Whole-layer ROCm expert streaming regressed
to 0.85 prefill tok/s because roughly 146 GB crossed PCIe per prompt tile, so a
selective hot-expert cache—not full-layer streaming—is required next. These
figures are below the 100/20 tok/s targets and are recorded as baselines, not
as target completion.

### Full-EP1 versus EP8 measurements

Do not compare the single-node full model with the EP8 mechanical benchmark.
EP8 evaluates only the experts owned by one rank unless eight ranks cooperate;
its 47--69 tok/s tables are not a full 256-expert single-node forward.

On the 188 GiB Threadripper + 16 GiB RX 9070 XT host, the August 11 full-EP1
profile measured:

- 1024-token prefill: 8.46 tok/s before serving-path wiring; 11.60 tok/s after
  grouped 256-expert routing, HIP Tier-B2 projections, and gfx12 WMMA tiling.
- 512-token registered streaming: 9.31 tok/s. The run registered 142.5 GiB of
  layer spans but the 24 GiB hard memlock limit forces registration churn.
- Isolated full-EP1 decode: 5.22 tok/s. Routed experts alone take about
  91 ms/token; an 18 tok/s EP8 result is not evidence for full EP1.

Profile-only `DS4F_PROF=1` reports `hip_route` and `hip_hreg`; production
tuning remains controlled by runner arguments. The complete routed GPU math
is only about 3.0 seconds for a 512-token prompt. The dominant prefill cost is
moving the 142.5 GiB expert bank through a pageable/rolling-registered PCIe
window, not attention or routed WMMA arithmetic. Genuine single-request
100+ tok/s therefore requires a larger permanently pinned host window,
additional expert-resident accelerator memory, or expert-parallel ranks; it
cannot be inferred from the storage-only EP8 harness.

For the parity-preserving compressed-attention micro-optimization, pass
`--hip-attn-cmp-fast 1` to `ds4f_serve_runner.py`. It selects the AVX2/FMA
dot/axpy implementation without changing the compressed representation or
sampling behavior. It is enabled by default in the HTTP runner; pass
`--hip-attn-cmp-fast 0` when validating a scalar reference run. Exact GPU
Tier-B2 window batching is likewise enabled by default with
`--hip-tb2-batch 1`; disable it only for scalar baseline comparisons.
The runner also defaults to `--hip-expert-cache-mb auto`, admitting prompt-hot
expert bundles into otherwise-free VRAM before decode. Use `--hip-expert-cache-mb 0`
for a transfer-only baseline.

`--hip-expert-pinned-staging 1` is an explicit diagnostic alternative to the
rolling registration path. It coalesces each layer into two pinned host slabs
before DMA, but is disabled by default: on this 188 GiB host the two staging
slots reduce file-cache headroom and measured 9.66 tok/s at 1024 tokens versus
11.60 tok/s for direct streaming. The CPU fallback now groups every routed
expert under three pool dispatches per layer; it improved the same 1024-token
case from more than 234 seconds to 131.48 seconds (7.79 tok/s), but remains a
fallback rather than the production profile.
