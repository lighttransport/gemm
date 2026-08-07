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

- **Conversation prefix cache** (automatic in the frontend): the chat handler
  keeps the last conversation's token ids + KV snapshot (`<BASE>.conv`) and,
  when the next request extends it, asks the runner to restore the KV and
  prefill only the new tail (`cache_load`+`cache_save`) instead of
  re-prefilling the whole context.  A request that does not extend the cached
  conversation falls back to a fresh full prefill (the session is reset to
  position 0, so multi-turn conversations stay correct).  Measured: turn 2 of
  a conversation lands in ~8 s vs a full re-prefill.
- **System-prompt cache** (`DS4F_SERVE_SYSCACHE`): the runner loads a
  persisted context (built once with a `cache_save` request) into slot 0 at
  startup.  Combined with a `cache_load` request whose prompt begins with that
  exact prefix, the system prompt / tool definitions are never re-prefilled.
- **Slots** (`DS4F_SERVE_SLOTS`): per-conversation KV snapshots switched by
  the `slot` field (`<BASE>.slot.<i>` files).
- **Truncation**: the runner caps the prompt at `max_pos - max_new`, keeping
  the head (system prompt + tools) and the recent tail whole.
- The model's own sliding-window (128) + compressed long-range (tierb2)
  attention keep the per-token decode cost flat out to `DS4F_MAXPOS`.

## Tool calling

`ds4f_serve.py` injects the OpenAI `tools` into the prompt
(`<tool_call>{...}</tool_call>`) and parses them back into OpenAI `tool_calls`,
so agents can loop tool calls over the same conversation.  `stream: true`
requests are emitted as **real SSE deltas**: the runner appends each generated
token id to `<BASE>.tok` (ctl bit2) and the frontend tails that file, decoding
each token and sending a `chat.completion.chunk` event as it lands (finish +
usage + `[DONE]` at the end).

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

The dense bank is attached to the ROCm via the `hip_ds4f_dense` hooks (the
default `DS4F_SERVE_USE_HIP=1`; the CPU-only path is ~4x slower).  The routed
MXFP4 experts stay on the CPU.  Measured decode: **~240 ms/token** (2.3-4 tok/s)
with the tierb2 compressor on, ~176 ms with it off; the CPU experts are the
floor (the harness's best no-compressor decode is ~185 ms/token, and neither
the CUDA expert batch nor the ROCm MXFP4 path beats the CPU at M=1).  The
CUDA-expert async batch (hetero/ds4f) is the follow-on accelerator for
multi-token prefill, not the single-token decode.
