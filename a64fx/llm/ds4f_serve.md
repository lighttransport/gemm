# DS4F single-node serving for coding agents

Single-node OpenAI-compatible serving of the DeepSeek-V4-Flash model for
opencode / codex / claude-code.  Three processes:

```
libds4f_serve.so     (built by a64fx/llm/build_ds4f_serve.sh)
ds4f_serve_runner.py (persistent model loop; shared-FS protocol)
ds4f_serve.py        (OpenAI HTTP frontend; already in the tree)
```

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
DS4F_SERVE_BASE=/tmp/ds4f_serve PORT=8080 \
python3 a64fx/llm/ds4f_serve.py
```

Smoke test: `sh a64fx/llm/test_ds4f_serve.sh /tmp/ds4f_single 8080`.

## Context / KV management

- **Prefix cache** (`DS4F_SERVE_PREFIX_CACHE=1`): a request whose prompt
  extends the previous one skips re-prefilling the shared prefix.  The frontend
  passes `cache_load`/`cache_save` (ctl bits) + a cache path; the runner
  snapshots/restores the per-layer KV + the tierb2 compressor state via
  `ds4f_tb2_snap*`.  Measured: the same 8-token tail generated in 9.2 s full
  vs 3.6 s with a 5-token cached prefix.
- **Slots** (`DS4F_SERVE_SLOTS`): per-conversation KV snapshots switched by
  the `slot` field (`<BASE>.slot.<i>` files).
- **System-prompt cache** (`DS4F_SERVE_SYSCACHE`): a persisted context (built
  once with a `cache_save` request) is loaded into slot 0 at startup, so every
  conversation starts with the system prompt / tool definitions already
  prefilled (instant TTFT, survives restarts).
- **Truncation**: the runner caps the prompt at `max_pos - max_new`, keeping
  the head (system prompt + tools) and the recent tail whole.
- The model's own sliding-window (128) + compressed long-range (tierb2)
  attention keep the per-token decode cost flat out to `DS4F_MAXPOS`.

## Tool calling

`ds4f_serve.py` injects the OpenAI `tools` into the prompt
(`<tool_call>{...}</tool_call>`) and parses them back into OpenAI `tool_calls`,
so agents can loop tool calls over the same conversation.

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
wire_api = "chat"
```

**claude-code**: the Anthropic wire protocol is not served directly; use an
OpenAI-to-Anthropic bridge (e.g. claude-code-router) with the same
`http://localhost:8080/v1` base URL.

## Performance notes

The dense bank is attached to the ROCm via the `hip_ds4f_dense` hooks; the
routed MXFP4 experts stay on the CPU.  The CPU expert decode dominates the
token rate (~1-4 tok/s on this box for the full 256-expert model); the
CUDA-expert async batch (hetero/ds4f) is the follow-on accelerator.
