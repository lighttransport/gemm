# Qwen3.8 server correctness and performance

## pi coding agent via its llama.cpp extension — 2026-09-27

pi's built-in llama.cpp extension (pi 0.87.1) talks to a llama.cpp router
server. The shim now answers the router endpoints it uses:
- `GET /models` entries carry `status.value = "loaded"`, `meta.n_ctx`, and
  the input modalities.
- `GET /props` returns `models_autoload: false` and the GGUF chat template.
  The template contains `enable_thinking`, which makes pi mark the model as
  reasoning and send `chat_template_kwargs.enable_thinking`.
- `POST /models/load` and `/models/unload` are no-ops: one model is
  resident.
- `--served-model-name` (the launcher uses `qwen3.8-27b`) replaces the GGUF
  file name as the model id.

Setup:

```sh
rdna4/llm/run_qwen38_27b_codex.sh &
rdna4/llm/pi/qwen38_pi_setup.sh     # credential + catalog in ~/.pi-qwen38
PI_CODING_AGENT_DIR=~/.pi-qwen38 pi --provider llama.cpp --model qwen3.8-27b
```

In an existing pi setup, `/login llama.cpp` with URL `http://127.0.0.1:8090`
followed by `/model` is equivalent. Two catalog quirks:
- pi refreshes extension catalogs from the network only in interactive and
  RPC modes; `pi -p` and `pi update --models` use the stored catalog. The
  helper therefore runs one short RPC session.
- pi sends no session id or cache key to llama.cpp, so all pi sessions share
  one cache identity.

### Chat Completions streaming

Streaming now splits the generation:
- Reasoning before `</think>` streams as `reasoning_content`.
- Answer text streams as `content`, held back from the first possible
  `<tool_call>`. Tool calls follow as `tool_calls` deltas with
  `finish_reason: "tool_calls"`.
- A usage chunk with `prompt_tokens_details.cached_tokens` follows when
  `stream_options.include_usage` is set.

Previously the raw XML and `<think>` text streamed as content.

pi replays tool calls with the server-issued ids. The raw-turn memory
therefore restores their exact bytes (see the Claude Code section).

### Caching changes found with pi

- **Save on any discard (runner).** Because pi's sessions share one
  identity, the runner now captures the live state whenever it is about to
  be replaced, not only on an identity change. A → B → A with pi restored
  A's saved 5,341-token state and appended 28 tokens.
- **All-but-last-paragraph boundary (shim).** pi ends its system prompt with
  the project context and then a `<cwd>` paragraph. The shim now also sends
  "all but the last paragraph" as a prefix boundary. A session in another
  directory of the same repo restored 4,408 tokens and prefilled 46, instead
  of 3,496 from the 960-token tools snapshot.

### Measured

pi's cold prompt is 4.5K tokens (read/bash/edit/write tools plus
AGENTS.md/CLAUDE.md context).
- The fix/compile/run task took 14.9 s over 4 requests. Every step after
  the first continued the live state (17–65 tokens appended).
- Thinking blocks and tool calls arrived as structured pi content.
- The DFlash2 HTTP gate passes. Codex and Claude Code still work against
  the same server.

## Claude Code CLI support — 2026-09-27

The shim now serves the Anthropic Messages API (`POST /v1/messages`, also
with `?beta=true`, plus `/v1/messages/count_tokens` and `HEAD` probes). The
same template renderer and runner caches back it (`anthropic_api.py`). With
the server from `run_qwen38_27b_codex.sh` running:

```sh
rdna4/llm/claude/qwen38_claude.sh -p "fix the bug in mathutil.c, compile and run it" \
    --allowedTools 'Read,Edit,Bash(cc:*),Bash(./mu)' </dev/null
```

The wrapper runs `claude` with a clean environment and its own config
directory (`~/.claude-qwen38`). It points every model role at the local
model and sets `CLAUDE_CODE_MAX_CONTEXT_TOKENS` /
`CLAUDE_CODE_AUTO_COMPACT_WINDOW` to the server's 64K context (Claude Code
otherwise assumes 200K) and the output cap to 16K.

### Translation

- System blocks are merged, minus Claude Code's per-version
  `x-anthropic-billing-header` block.
- Server-side tools (web search) are not exposed to the model.
- `tool_result` blocks become grouped `<tool_response>` turns. Mid-conversation
  `system` messages (Claude Code's `<total_tokens>` budget notes) become user
  turns.
- Thinking follows `thinking: {"type": "adaptive"|"enabled"}`, with the
  effort from `output_config.effort` (or `budget_tokens`).
- Client `temperature`/`top_p` are ignored in favour of Qwen's profiles;
  Anthropic clients send 1.0. Set `QWEN38_HONOR_CLIENT_SAMPLING=1` to honor
  them.
- Each turn starts with a thinking block whose `signature` carries the raw
  generated turn. The Messages API requires clients to return thinking
  blocks unmodified.

### Server-side turn memory

On the first request of a resumed session (`claude -p --resume`), Claude
Code drops the thinking blocks of earlier turns. It also stores tool inputs
with defaults filled in (for example `replace_all: false`).

The shim therefore remembers each generated turn, keyed by the tool-call ids
it issued (random, server-issued) or by the hash of a text-only answer. A
turn that comes back without its blob replays from that memory when the tool
names and answer text match. This works for all three APIs.

### Tools-block boundary

Claude Code's system text embeds per-project details, such as a memory
directory derived from the config dir and cwd. The full system turn is
therefore only shared between sessions in the same project.

The runner protocol now accepts several comma-separated prefixes, and the
shim also sends the tools block alone (it precedes the system text in the
template, ending at a clean `\n\n` + letter split). Each boundary gets a
shared snapshot.

### Measured

Claude Code 2.1.283 with 20 built-in tools; the cold prompt is 15.7K tokens.

| Run | Turns | Wall | Uncached input | Cache read |
| --- | ---: | ---: | ---: | ---: |
| Session 1, fix/compile/run, cold server | 4 | 38.1 s | 15,939 | 47,990 |
| Session 2, same project (shared system turn) | 4 | 15.7 s | 2,764 | 61,471 |
| `--resume` session 1 after session 2 | 5 | 11.1 s | 404 | 83,584 |

- The resumed request continued from session 1's saved live state with only
  the 54-token new message appended.
- A session in another project restored the 11,871-token tools snapshot and
  prefilled 3,837 tokens (8.6 s) instead of 15,708 (31 s).
- Every agent step continued the live state (40–154 appended tokens).
- The DFlash2 HTTP gate and the Codex flows still pass.

## Codex integration for Qwen3.8-27B, dense GSQ + DFlash2 — 2026-09-27

Launch with `rdna4/llm/run_qwen38_27b_codex.sh`. It serves the IQ2_XS GSQ
target plus the DFlash2 sidecar on `127.0.0.1:8090` with a 64K context and
16K output cap. It also installs a Codex profile
(`CODEX_HOME=$HOME/.codex-qwen38`) from `rdna4/llm/codex/`:

```sh
rdna4/llm/run_qwen38_27b_codex.sh &
CODEX_HOME=$HOME/.codex-qwen38 codex exec --skip-git-repo-check \
    --sandbox workspace-write "fix the bug in mathutil.c, compile and run it" </dev/null
```

Close `codex exec`'s stdin (`</dev/null`) in scripts. Otherwise it waits on
"Reading additional input from stdin".

### Model catalog (`codex/qwen38_model_catalog.json`)

Without a catalog entry, Codex uses fallback metadata and sends no
`apply_patch` tool. The model then improvised `apply_patch "...\n..."` shell
strings and looped 170 times on the same parse error.

The catalog entry sets:
- freeform `apply_patch` (the only type this Codex accepts);
- a 64K context and a 4K-token tool-output truncation;
- no skills, plugins, apps or web-search instructions;
- multi-agent and goals disabled.

- `base_instructions` is our own short agent prompt. It does not copy the
  hosted models' instruction templates, and it documents the patch format.

The first Codex prompt shrank from 8,804 tokens (fallback metadata) to 6,695
(with the catalog), then to 2,478 with the short base instructions. A cold
first request now takes 4.7 s instead of 16.5 s. The tables below were
measured with the 6.6K-token prompt.

### Template fidelity (`qwen_chat.py`)

Prompts are rendered byte for byte like the GGUF `tokenizer.chat_template`.
`test_chat_template.py` renders the checked-in `qwen38_chat_template.jinja`
with jinja2 and compares, including thinking and effort variants. The
template covers:
- the official tools block;
- tool results as a `user` turn of grouped `<tool_response>` blocks (the old
  shim used an off-template `tool` role with `call_id=`);
- one assistant turn per model step, with text before its tool calls;
- `<think>` frames and the reasoning-effort instructions.

### Thinking

`--thinking auto` (default) enables Qwen thinking when a Responses request
carries `reasoning.effort`, which Codex always sends:
- `low`, `medium` and `high`/`xhigh` map to the template's effort levels;
- Chat Completions thinks only with `reasoning_effort` or
  `chat_template_kwargs.enable_thinking`.

When the request carries no sampling fields, defaults follow Qwen's
published profiles:
- thinking: 0.6 / 0.95 / 20;
- non-thinking Responses: 0.7 / 0.8 / 20 with presence 1.5.

The old 0.2 near-greedy default drove repeated identical tool calls.

### Prefix caching — three layers

1. **Live continuation (runner).** Generated tokens are often not the
   canonical BPE of their own text. For example, the model emits `Ġ"***`
   where re-tokenization gives `Ġ"*` + `**`. Before this change, every agent
   step therefore diverged from the live state, restored a ~0.5 GiB prompt
   snapshot, and re-published another one.
   - The runner now remembers the exact bytes behind its live tokens.
   - A same-identity prompt that byte-extends them at a special-token
     boundary (`<|`) keeps the live tokens and tokenizes only the appended
     text.
   - Per-step snapshots are no longer taken for continuations.
2. **Exact assistant replay (shim).** Every Responses turn starts with a
   `reasoning` item whose opaque `encrypted_content` is
   `q38raw1:` + base64(raw generated turn). Codex requests
   `reasoning.encrypted_content` with `store=false` and sends the item back,
   so the turn is re-rendered from its exact bytes, not re-serialized
   tool-call JSON.
3. **Snapshots across conversations (runner).**
   - The system-prefix snapshot is published under a shared namespace, so a
     new Codex thread restores the ~6.6K-token tools+instructions prefix. It
     is evicted last, and `LLM_SERVER_SHARED_PREFIX=0` restores per-identity
     namespacing.
   - When another identity takes the GPU, the outgoing conversation's live
     state is snapshotted with its bytes. `codex exec resume` then continues
     from where the thread stopped (text continuation from the snapshot).
   - The outgoing state is captured before the restore but published after
     the incoming conversation's entry was restored and touched, so the save
     cannot evict what the incoming request needs.
   - Re-saving an unchanged state is not an LRU touch. The two-entry A/B/A
     case in `test_qwen35_dflash2_http.py` depends on this ordering.

### Measured on the RX 9070 XT, 212 W cap

Codex 0.157.1, `--sandbox workspace-write`.

| Step | Prompt | Reused | Prefilled | Prefill ms |
| --- | ---: | ---: | ---: | ---: |
| New thread, cold server | 6,905 | 0 | 6,905 | 12,948 |
| Agent steps 2–5 (live continuation) | 7,240–7,658 | all but 37–289 | 37–289 | 152–605 |
| New thread B (shared system prefix) | 6,911 | 6,583 | 328 | 643 |
| `codex exec resume` A after B (saved live state) | 7,783 | 7,752 | 31 | 140 |

- The fix/compile/run task finished correctly in 5 requests (`./mu` printed
  5). Codex reported 29,401 of 36,733 input tokens cached.
- The resumed and cross-thread tasks also completed correctly.
- Decode inside Codex ran at 53–94 tok/s with DFlash2. DFlash2 still hands
  off to target decode beyond 32K positions.
- A full A / B / `resume A` sequence from a cold server reported these
  `cached_input_tokens`:
  - thread A: 59,867 / 67,232 (89%);
  - thread B: 36,397 / 43,621 (83%);
  - resume A: 20,604 / 21,028 (98%).

  The only full miss was the very first request.

### DFlash2 serving fix found by this work

The DFlash2 verifier forward is a replayed HIP graph. Its feature-capture
kernels replay, but their host-side `feature_rows` was set only while
capturing.

- A prefill whose last batch had one row left `feature_rows = 1`.
- The next accepted multi-row window was then rejected
  (`DFlash2 commit rejected ... features=1`), and the request failed with
  `ERR generation`.
- Live continuation makes such suffix sizes common. A 129-token Codex step
  hit it.
- The fix sets `feature_rows` after every verify-graph launch.
- `test_qwen35_dflash2_http.py` now includes this one-row-prefill case. It
  fails without the fix and passes with it.
- The 4K fixture hashes are unchanged: DFlash2 104.2 tok/s, plain
  50.2 tok/s.

### Debugging and knobs

Debugging:
- `QWEN38_TRACE_DIR=<dir>` dumps each request with its rendered prompt.
- The runner logs `live prefix diverges at N ... live <tokens> | prompt
  <tokens>` whenever a same-conversation prompt cannot extend the live state.

Runner environment variables:
- `LLM_SERVER_LIVE_CONTINUATION=0` disables live continuation.
- `LLM_SERVER_SNAPSHOT_EVERY_PROMPT=1` restores per-request prompt snapshots.
- `LLM_SERVER_SHARED_PREFIX=0` restores per-identity prefix namespacing.

## Earlier results — 2026-09-05

Hardware: Ryzen Threadripper 1950X (16 cores), RX 9070 XT (gfx1201).
Model: Qwen3.8-Flash-Next-UD-Q4_K_XL, four GGUF shards.
Launcher: `run_qwen38_codex_server_rocm.sh`, 7200 MiB expert cache,
65536-token context, 512-token prefill batches, 16 OpenMP threads.
Sampling: temperature 1.0, top_p .95, top_k 40, min_p .01,
presence penalty 0; no implicit frequency/no-repeat penalty in server requests.

## Current Qwen3.8 speculative multi-context gate — 2026-09-21

The Qwen3.8 resident backend now assigns each request a FIFO ticket and a
request-owned cancellation event. Cache reuse is namespaced by a hashed
conversation identity and backed by a bounded transactional LRU of portable
host snapshots. `--context-cache-entries` defaults to 4,
`--context-cache-max-mib` to 2048, and
`--qwen35-snapshot-max-tokens=0` selects the 16,384-token default bound.

`REQ3` carries the cache identity and complete sampling controls. The HTTP shim
accepts `prompt_cache_key`, conversation/session metadata, and the
`X-Prompt-Cache-Key` header. It accepts `request_id` or `X-Request-ID`, echoes
the ID in responses, and supports targeted `POST /v1/cancel`. A queued
cancellation never signals the active runner transaction. The GPU execution
path stays serialized because the target and DFlash verifier share mutable
device state. Request IDs are reserved before SSE headers are sent, so a
duplicate receives an ordinary HTTP 409 instead of corrupting an established
event stream. Idle cancellation returns 404, malformed message shapes and
content lengths are rejected before inference, and oversized direct stdio
frames are drained as one failed transaction. Context trimming preserves
original ordering for duplicate-valued messages and removes complete user /
assistant / tool turn groups, avoiding orphaned tool results.

Portable snapshots contain prompt logits, hybrid convolution/recurrent state,
target Q8/Q8 KV plus FP16 scales, and DFlash private state. Dense NextN
snapshots also retain the prompt-boundary target hidden vector used by the
first draft proposal. Publication occurs
only after successful generation; failures and cancellations discard pending
snapshots but preserve older committed entries. Restore uses the longest exact
token prefix for the same cache identity. The runner rejects a snapshot unless
its position equals its token-key length, and evicts entries that fail restore.
Models whose snapshots omit positional KV retain same-resident-context prefix
reuse, but those entries are marked resident-only and discarded before a
reset, identity switch, or portable restore. They are never used as
multi-context snapshots. An exact restored prompt retains and touches its
existing snapshot instead of taking an identical 234--448 MiB device-to-host
copy after every response.

The real-GPU harness now forces A/B/A switching rather than accepting an
immediate same-context hit. On RX 9070 XT it restored an actual 6,535-token
prefix (`cached_tokens=6535`) after an unrelated conversation, reproduced the
same greedy output, and reported a 448.3 MiB snapshot. It also passed direct
stdio transactions, seeded sampling, LRU eviction, malformed cache metadata,
targeted/disconnect cancellation with recovery, concurrent distinct
identities, and a two-turn C++ generation whose programs compiled and printed
the expected result. Repeated exact prompts restored one committed snapshot
without republishing it. The CPU protocol/template/tool suite passes 31 tests.

Dense NextN now uses the resident path for exact greedy K=3 windows. Draft KV
is reset at every request boundary, sampled requests use ordinary target
decode, and cancellation/error recovery invalidates any open verifier window.
The GPU gate passes ordinary-target byte parity, A/B/A cache restore,
cancellation, concurrent identities, compiled two-turn C++ output, and an
exact `ZEPHYR-7319` retrieval response. The pinned 4K llama.cpp gate matches
tokens, EOS and bytes for greedy and sampled generation; random-token 64K
preserves the established target suffix hash but measures only 28.82 tok/s,
so the feature remains opt-in.

The gate exposed and fixed two portability bugs: batched prefill left the host
position stale, and Q8/Q8 FP16 scale rows were copied using an FP32 byte size.
Previous long-prompt immediate-repeat results did not prove host restoration;
larger 10K--60K snapshot claims need the new forced-interleave gate before they
are considered validated.

## Findings and fixes

- Calling raw ggml CPU dot kernels after `dlopen` without `ggml_cpu_init`
  leaves FP16 lookup tables uninitialized. A Q8 dot product expected to be 32
  returned 0 before initialization and 32 afterward. Live CPU expert outputs
  had relative L2 error 1 against the dequantized-weight/FP32-activation
  reference; after initialization, sampled errors were about .008–.013.
  This initialization is needed by both CPU prefill and decode experts.
- `LLM_MOE_CPU_MIN_WEIGHT=1` drops uncached experts instead of evaluating
  the full selected route. The launcher now defaults to 0.
- CPU decode refills require a copy stream and completion events independently
  of `LLM_MOE_COPY_PIPELINE`. Previously disabling the prefill pipeline also
  omitted these events, leaving slots permanently pending. Decode-to-prefill
  transitions now drain outstanding copies and publish their cache identities.
  Only one refill per layer can be outstanding with the current metadata.
- Sampler softmax must preserve its maximum logit before overwriting the
  probability array. A constant logit shift must not change samples.
- The GGUF explicitly sets `tokenizer.ggml.add_bos_token=false`; the presence
  of a BOS token id does not authorize inserting it.
- Non-thinking chat history includes empty thinking frames, and leading
  system/developer messages are merged, matching the inspected GGUF template.
- Runner stdout diagnostics must be consumed inside their request transaction;
  otherwise the next request can receive the preceding request's completion.
- Recurrent prefix snapshots cannot restore overwritten positional KV. Reset,
  cancellation, and failed prefill invalidate the corresponding snapshot.
  An EOS token sampled but never forwarded is not part of reusable state.

## CPU-only regressions

From the repository root:

```sh
python3 rdna4/llm/test_sampler.py
python3 rdna4/llm/test_prompt_policy.py
python3 rdna4/llm/test_chat_template.py
python3 rdna4/llm/test_codex_protocol.py
python3 rdna4/llm/test_copy_lifecycle.py
python3 rdna4/llm/test_cpu_library.py /path/to/libggml-cpu.so
make -C rdna4/llm -j2
bash -n rdna4/llm/run_qwen38_codex_server_rocm.sh
git diff --check
```

These passed locally using
`/mnt/nvme02/work/llama.cpp/build-codex-hetero-dev2/bin/libggml-cpu.so.0.22.0`.
The build retains pre-existing warnings; it is not warning-clean.
`LLM_MOE_CPU_VERIFY=1` enables the expensive live CPU-expert reference
comparison. Do not use it when measuring performance.

## Real Codex validation

Used `CODEX_HOME=/home/syoyo/.codex-local`, whose provider points at
`http://127.0.0.1:8080/v1`, and an isolated temporary working directory:

```sh
CODEX_HOME=/home/syoyo/.codex-local codex exec \
  --skip-git-repo-check --sandbox read-only -C /tmp/TEST_DIRECTORY --json \
  'Write a complete C program that sums integers 1 through 10 and prints the result. Return only a fenced C code block. Do not inspect files or run tools.'
```

Resume the returned thread id with `codex exec resume --skip-git-repo-check
--json THREAD_ID PROMPT`. The second prompt changes the upper bound to 20;
the third requests the sum of squares from 1 through 20. Generated programs
were compiled with `cc -Wall -Wextra -Werror` and checked for results 55,
210, and 2870. The HTTP shim also translates Qwen XML tool calls into native
Responses/Chat Completions tool-call items. With the live server,
`codex exec` successfully emitted a namespaced shell call, executed
`find . -maxdepth 1 -mindepth 1 -type d | wc -l` in read-only mode, and
received `21`.

Representative single first-turn measurements (not a statistical benchmark):

| Configuration | Prompt tokens | Prefill tok/s | Decode tok/s | Expert cache hits |
| --- | ---: | ---: | ---: | ---: |
| CPU initialized, refill events missing | 4792 | 115.45 | 8.46 | 20.4% |
| CPU initialized, working refills, 16 threads | 4792 | 116.90 | 15.16 | 55.4% |
| Working refills, 8 threads | 4792 | 116.30 | 14.40 | 55.5% |
| Working refills, aligned chat framing, 16 threads | 4779 | 116.90 | 17.25 | 55.7% |

The latest second Codex turn reused 4594 of 4883 prompt tokens and processed
289 new tokens. Its prefill was 89.44 tok/s and decode 13.40 tok/s. System
prefix reuse is confirmed; full conversation-prefix reuse is not established.
Short/suffix-prefill throughput should not be compared directly with the
long initial prompt. Fast runs that omitted experts or returned malformed
code are not valid decode-performance baselines.

Remaining serving work is an exact, beneficial multi-context decode batch and
forced-interleave validation above the current 6,535-token gate. Decode
optimization remains tracked separately.
