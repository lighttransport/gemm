# Qwen3.8 server correctness and performance

## Interrupted follow-up turns, reasoning loops, snapshot buffers — 2026-09-27

### Rolling back a cancelled continuation

A cancelled request reset the runner. The first turn of a conversation kept
its prompt snapshot, but an interrupted follow-up turn (a live continuation,
which takes no prompt snapshot) left the retry nothing better than the last
saved snapshot or shared prefix.

Every continuation turn now takes a **resident checkpoint** after prefill:
the recurrent, logits and DFlash2 state, without the attention KV rows
(`hip_llm_snapshot_state_resident`). Decoding only appends KV rows, so on a
cancel the runner restores the checkpoint and keeps the live state at the
prompt boundary instead of resetting. The agent's retry is then a live
continuation with nothing to prefill.
- Measured on an 8,590-token follow-up interrupted at its first token: the
  retry reused all 8,590 tokens in 1.4 s.
- Its greedy text equals the uninterrupted follow-up byte for byte.
- A clean re-prefill of the same messages differs late, as any live
  continuation does.

The checkpoint is 233 MiB. It lives in one persistent pinned arena, sized
from the previous checkpoint, so the GPU copies straight into it: **18 ms
per turn** after the first, against 130 ms with ordinary buffers.
`LLM_SERVER_CANCEL_ROLLBACK=0` turns it off.

A different next message after an interrupt still restores from saved
snapshots, because the old prompt's assistant header is not a prefix of the
new prompt.

### Reasoning-loop guard

In about 400 agent requests, two turns looped on one paragraph of their
thinking until the 16,384-token output cap:
- a Claude Code review turn;
- a Codex turn that repeated "Scenario: realloc to 8 succeeded…" four times
  and never found the missing `sizeof(int)`, so the task failed.

Each wasted about 6 minutes of GPU time.

The runner now watches thinking-mode generations until `</think>`. Every 16
tokens it checks whether the last 32 tokens have already occurred twice in
this response.
1. On the first hit, it raises the presence penalty to 1.5, the value
   Qwen recommends against endless repetition in quantized models.
2. If the loop continues at least 256 tokens later, it closes the thinking
   (emits `</think>`) so the model moves on to its answer.

Induced test (the model was told to recite a line 80 times in its
reasoning): the guard tripped at token 1456 and closed the thinking at 1776.
Normal thinking at 3K–23K contexts never triggered it. Code and answers
after `</think>` are never touched. `LLM_SERVER_LOOP_GUARD=0` turns it off.

### Snapshot buffers

- Freed snapshot buffers of 2 MiB or more are pooled and reused, best fit,
  up to `LLM_SNAPSHOT_POOL_MIB` (default 1024). Over the cap, the largest
  idle blocks are dropped first.
- Per-layer recurrent-state copies are batched through the pinned staging
  buffer with one synchronization per 32 MiB, instead of a synchronous
  round trip each.

`LLM_SNAPSHOT_PROFILE=1` now splits a capture into device wait and host
copy. On this machine the host copy into fresh pages runs at about 2 GB/s:
xmrig saturates DRAM, and non-coherent pinned staging did not change it.
Full KV captures therefore stay about as fast as with huge pages alone.
Pinning the whole 12 GiB snapshot cache instead is not worth the locked
memory.

## Hardening for agent traffic: crashes, interrupts, protocol — 2026-09-27

### Runner crashes

A runner that died (GPU fault, OOM kill) used to leave the server returning
"runner exited" until someone restarted it by hand.

The next request now restarts it and waits for READY:
- It retries once after 15 s if the first start raced the dead process's
  VRAM release.
- The context the new runner reports updates `/props`, trimming and error
  messages before that request continues.
- Stored system prefixes are warmed again in the background.

Limits:
- At most `QWEN38_MAX_RESTARTS` (default 3) restarts within 10 min; after
  that the runner stays down and `/health` reports it.
- A request whose runner dies before any token reached the client runs
  again once on the restarted runner. Afterwards the client, which already
  has partial output, gets the error.
- A request that has killed the runner twice is refused without another
  restart. Agents resend a failed request verbatim, and one such request
  could otherwise use up the restart budget for everyone.

Soak test, measured: Claude Code, Codex and pi fixed the stack library
concurrently while the runner was killed with `kill -9` 120 s in. The next
request restarted it. One in-flight Claude Code request that had already
streamed text failed; Claude Code retried it. All three finished and passed
an independent ASan oracle (429 s, 515 s, 472 s).

### Interrupts

On a cancel (client disconnect, an agent's Esc) the runner reset its state
and discarded the snapshots it had just built. The retry that usually
follows re-prefilled the whole prompt.

Now the snapshots taken at complete boundaries (shared prefixes, the prompt
boundary) are kept before the reset, with three conditions:
- portable ones only;
- only when they fit without evicting another entry;
- marked first to evict.

The first version published them like any other snapshot. In the DFlash2
gate's two-entry cache, the cancelled request's snapshot evicted the
conversation still in progress, and the A/B/A restore returned 0 cached
tokens. A 15,316-token
request interrupted at its first token, or mid-prefill, is retried with all
15,316 tokens cached in 1.6 s. The prefill alone had taken 18 s.

The live state itself is still dropped, since it may sit inside an
uncommitted speculative window.

### Protocol

Chat Completions streams began with Responses-API events
(`event: response.created`, `response.in_progress`). The OpenAI SDKs pass
`response.*` events to chat clients as chunks without `choices`. Those
events are now sent on `/v1/responses` only.

### Review findings fixed (independent review of this and the previous commit)

- After a DFlash2 hand-off, the draft's 2048-slot context ring kept stale
  rows for the plain-decoded positions, and the next turn's prefill did not
  rewrite them. They are now zeroed when the request ends.
- The plain-decode rate that the hand-off compares against also counted
  tokens past the 32K cut-off. It now samples only handed-off tokens below
  it.
- The prefill-scratch reservation was also made when native Q8 prefill was
  off.
- Staging events leaked if pinned-buffer setup failed partway.
- A failed plain forward ended generation silently with a live state one
  token short. It is now an error that resets the state.

## Faster agent traffic: kernel profile, adaptive DFlash2, two decode fixes — 2026-09-27

Where the time went in the agent runs below: decode was 83% of GPU time, at
37–39 tok/s. The 4K benchmark decodes at 50 tok/s.

### The HTTP server never used the tuned kernels

`run_qwen38_gsq_rocm.sh`, which produced every published benchmark number,
sets about 25 kernel-selection variables:
- `--decode-kernels/--decode-layout auto` with a 1.8 GiB decode layout,
- `--qwen35-batched-prefill --ubatch 512` (the server prefilled 128 rows at
  a time),
- the Q8_1 matvec selections.

`codex_server.py` started `test_hip_llm` directly and got none of them. The
launcher now runs the runner through `qwen38_gsq_stdio_runner.sh`, which
applies the profile (`QWEN38_RUNNER` overrides it). Same prompts, same
server:

| | Before | Profile |
| --- | ---: | ---: |
| Prefill, 16K prompt | 525 tok/s | 771 tok/s |
| Prefill, 40K prompt | 424 tok/s | 634 tok/s |
| Plain decode, 40K | 38.6 tok/s | 43.1 tok/s |
| DFlash2 decode, 16K (greedy) | 48.0 tok/s | 53.6 tok/s |

The decode layout costs VRAM, so the default context drops from 112K to
**96K**, the largest that keeps the prefill scratch and the DFlash2 verify
workspace. A needle test at 87,595 tokens passes (3/3). Prefill runs at
402 tok/s, against 261 tok/s at 100K without the profile, and 492 MiB stays
free at peak.

The runner now also reserves the worst-case Q8 prefill attention scratch
before READY (`hip_llm_reserve_prefill_scratch`, about 200 MiB for 512-row
batches). It used to be allocated on the first prefill, and with the
profile at 112K that allocation failed with `ERR prefill` after the KV cache
and DFlash2 had taken the VRAM.

### Adaptive DFlash2

Thinking-mode agent turns, server-default sampling, 1024 tokens (tok/s):

| Prompt | Always speculate | Plain | Adaptive |
| ---: | ---: | ---: | ---: |
| 3K | 39.6 (28% accepted) | 48.7 | 48.9 |
| 12K | 37.2 | 46.9 | 47.4 |
| 23K | 31.0 | 45.2 | 46.0 |

A speculative step, meaning a draft plus an 8-row verify, costs 75–88 ms. At
about 21 ms per plain token it pays off only above roughly 37% acceptance.
Thinking prose stays below that. Code and tool calls are well above it.

Stack-library agent task (Claude Code, Codex, pi; all passing an independent
ASan check). Aggregate server-side rates:

| Configuration | Decode | Prefill | Accepted |
| --- | ---: | ---: | ---: |
| Old: no profile, always speculate | 49.8 | 462 | 45% |
| Profile, always speculate | 52.6 | 634 | 47% |
| Profile, adaptive (default) | 50.1 | 651 | 49% |

The adaptive run had a different mix of turns: 68% of its tokens were in
turns that handed off. Split per request:
- turns that kept speculating: 60.3 tok/s at 56% acceptance;
- turns that handed off: 46.5 tok/s, plain speed. Always speculating would
  have run them at about 40 tok/s, given their 28% acceptance.

Policy:
- Each request speculates first.
- After 16 steps, if its **cumulative** rate is below 90% of plain decoding,
  the rest of the request goes to the decode graph.
- Plain speed is a moving average of measured plain tokens, seeded at 21 ms
  (`LLM_QWEN35_DFLASH2_PLAIN_MS`).

The hand-off is one-way:
- Plain tokens do not produce the draft's features, and the next prompt's
  prefill re-syncs them.
- Resuming by decoding one-row verify steps was tried: it runs at 22 tok/s.
- Resuming with the skipped window zeroed was also tried: the probes kept
  losing.

Knobs:
- `LLM_QWEN35_DFLASH2_ADAPTIVE=0` restores unconditional speculation.
- `LLM_QWEN35_DFLASH2_MAX_POS` (default 32768) keeps the old position cut-off.

### Two decode bugs behind the hand-off

Both are in the stdio server loop (`test_hip_llm.c`). The benchmark loop was
not affected.

1. **Stale logits after a speculative-to-plain switch.** The loop skipped
   the target forward of the token that came from the just-committed window,
   because `q35_window` still described that window. The next token was
   then sampled from the prompt's logits. Greedy code output stopped after
   18 tokens (`def tool_registry(to` then `` ``` ``). The old 32K cut-off
   takes the same path (reproduced by forcing it at position 9900), so every
   DFlash2 generation that crossed position 32,768 got one token sampled
   from stale logits there.
2. **Live state one token short after a length cut inside a window.** The
   final commit published `q35_index` rows. The last emitted token is the
   input of row `q35_index`, so the live state missed it. Live continuation
   then extended a state that did not contain the reply's last token. The
   commit now includes that row, matching plain decode and the stop commit.

Check (`tmp/gsqprof/cut_continue.py`): cut a DFlash2 reply at 3, 5, 6, 7, 9
or 13 tokens, continue the conversation, and compare with a fresh prefill of
the same messages. All match.
- A 22-token cut differs late at a near-tie. It differs identically with
  always-on speculation, so the verifier's state, not the hand-off, is not
  bit-identical to a batch prefill.
- The reference 4K benchmark keeps hash `44915ec1039a64c8`: ordinary
  50.2 tok/s, DFlash2 103.6 tok/s.
- The DFlash2 HTTP gate passes (4/4).

### Conversation switches

Capturing a live state before another conversation takes the GPU ran at
about 1.1 GB/s: 3.1 GB in 2.5 s at 87K tokens.

The limit is first-touch page faults on the fresh host buffers (about
1.5 GB/s on 4 KiB pages), not PCIe. Snapshot buffers are now 2 MiB aligned
with `MADV_HUGEPAGE`, and large device-to-host copies go through a pinned
double buffer: 770 MiB now takes 414–458 ms instead of 700 ms. Restore
already ran at about 10 GB/s.

Captures and restores of 256 MiB or more are logged with their time.
`LLM_SNAPSHOT_PROFILE=1` splits a capture into stages. Reusing freed
snapshot buffers would avoid the faults altogether and is the next lever.

## Agent code review, long context and review fixes — 2026-09-27

### Long context: 1M is out of reach, the card's own limit is stable

A 1M-token context is not reachable with this model on this card. The GGUF's
native `qwen35.context_length` is 262,144, and the Q8 KV cache alone for 1M
tokens would be about 35 GB against 16 GB of VRAM.

The runner clamps the requested context to free VRAM:

| Server | Requested | Allocated |
| --- | ---: | ---: |
| `--qwen35-server-profile` (no DFlash2) | 262,144 | 190,464 |
| launcher with DFlash2 | 114,688 | 114,688 |
| launcher with DFlash2 | 131,072 | 131,072, DFlash2 off (see below) |

DFlash2 needs an 8-row verify workspace of about 1.2 GiB: recurrent-state
checkpoints for every SSM layer. It was allocated on the first speculative
verify, after the KV cache had taken the VRAM. At 131,072 that allocation
failed, and **every request returned `ERR generation`**: the first Claude
Code request here (`speculative verify failed pos=17069 rows=8`), and even a
one-line prompt. The runner now reserves the workspace
(`hip_llm_qwen35_mtp_verify_reserve`) before it reports READY. If the
reserve fails, it logs a warning and serves with plain decode. 114,688 is the
largest tested context that keeps DFlash2.

Needle test (`tmp/longctx/needle.py`) on the no-DFlash2 server:
- The prompt is repository sources with three codewords planted at 10%, 50% and 90%.
- Each run asks for all three codewords, then a follow-up question in the same conversation.
- Each run then asks a short unrelated conversation (B), and returns to the long one (A).

| Prompt tokens | Share of window | Prefill | Needles | Follow-up | A after B | Snapshot |
| ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 145,839 | 77% | 646 s | 3/3 | 1.5 s | 3.3 s | 5.0 GB |
| 188,756 | 99% | 999 s | 3/3 | 2.2 s | 3.1 s | 6.4 GB |

- The follow-up extends the live state. After B, A resumes from its host
  snapshot rather than being prefilled again.
- Decode at 188K is about 31 tok/s.
- Larger prompts are rejected before prefill with a context-length error
  (below).

Fixes found by this test:
- **The shim advertised the requested window, not the allocated one.** The
  runner now prints `READY max_seq_len=N`. The shim adopts N for
  `/props` and `/models` `n_ctx`, for history trimming (`fit_context`) and
  for error messages. A bare `READY` from an older runner is still accepted.
- **Overflow was a 500 `server_error`.** It is now 400 with `code:
  context_length_exceeded` on OpenAI routes. On `/v1/messages` it is
  `invalid_request_error` "prompt is too long: it exceeds the N-token
  context window". Streams report the same codes. These are the errors on
  which Codex and Claude Code compact their history.
- **`fit_context` used 4 characters per token.** Source code and Markdown
  measure 2.7–2.8 with this vocabulary, so a trimmed history could still
  overflow. It now uses 3.
- `claude/qwen38_claude.sh` sizes Claude Code's window from the server's
  `/props` unless `QWEN38_CONTEXT` is set.

### Three agents reviewing this repository concurrently

Codex, Claude Code and pi each reviewed one shim module read-only, in a
detached worktree. All three ran at the same time against one launcher
server (64K, DFlash2). Prompt:

> Review FILE for real bugs … report at most 5 concrete findings with a
> line number and a one-sentence failure scenario.

Prefix caching across the three interleaved conversations:
- Every Codex and pi request extended its previous prompt byte for byte.
- Of 46 requests, all but the compaction restarts were served from a live
  continuation or a text-continuation snapshot restore.
- Most requests added 0.1–3K uncached tokens.

Codex (`anthropic_api.py`, 57 min, 22 turns):

| # | Finding | Verdict |
| --- | --- | --- |
| 1 | `input_tokens` should include cache reads | wrong: Anthropic's `input_tokens` excludes `cache_read_input_tokens` |
| 2 | assistant `content: null` raises `TypeError` → 500 | **real, fixed** |
| 3 | user `content: null` turn silently dropped | fixed (kept as an empty turn) |
| 4 | float `budget_tokens` ignored | fixed |
| 5 | `response_content` non-string arguments | latent, no caller passes one |

pi (`qwen_tools.py`, 25.5 min, 11 turns):

| # | Finding | Verdict |
| --- | --- | --- |
| 1 | fence-parity check miscounts "``" | wrong: it counts triple backticks |
| 2 | JSON-parsed values not type-checked | by design; the client validates tool input |
| 3 | non-dict property schema raises → 500 | **real, fixed** |
| 4 | `render_call` raises on non-JSON strings | real, and the function was dead code: removed |
| 5 | `call_events` assumes `item["type"]` | latent |

Related to pi's #2/#3, `parse_calls` now decodes each parameter by its
schema. A property without a `type` (`enum`, `anyOf`, Claude Code's
`Workflow.args`) keeps the raw text when it is not JSON. Previously the whole
call became plain text. A `["string", "null"]` type keeps strings raw.

Claude Code (`live_stream.py`) failed at 64K: "Autocompact is thrashing: the
context refilled to the limit within 3 turns of the previous compact, 3
times in a row". Claude Code compacts at about the window minus its output
reserve and a 13K buffer. That is about 36K tokens at 64K with a 16K output
cap, and its system prompt alone is 14K, so reading `codex_server.py`
(about 20K tokens) is enough to thrash. The compaction requests themselves
were cheap: the summary request reused 99% of the prompt from its snapshot,
and each restart reused the 14.2K-token system prefix.

Rerun on the new launcher default (114,688 tokens, DFlash2 kept), Claude
Code alone:
- It finished in 27.5 min with no compaction thrash; the prompt grew to 70K
  tokens.
- The answer was unusable for a different reason. The model was still
  thinking while it worked through a `StreamSplitter` test string containing
  `</think>`, and it emitted that as the real special token (id 248069
  appears 29 times in the generation). The shim splits reasoning from the
  answer at the first `</think>`, the same rule vLLM's and llama.cpp's Qwen
  reasoning parsers use. The rest of the deliberation therefore became the
  visible answer.
- A quoted tag and a structural one are identical at both the text and the
  token level. This is a limit of the model's chat format when it reviews
  code that handles its own tags, not a shim bug.

Two further bugs surfaced before any review started:
- **`QWEN38_TRACE_DIR` pointing at a missing directory failed every request
  with a 500.** Codex showed this as "high demand". Trace writing now creates
  the directory, uses collision-free names, and never fails a request.
- **`~/.pi-qwen38/auth.json` was `{}`,** so pi reported `Unknown provider
  "llama.cpp"`. Running `pi/qwen38_pi_setup.sh` fixes it; the setup step is
  required.

### DFlash2 in agent traffic

During the agent reviews DFlash2 barely helped: 38.7 tok/s overall.
- 34% of drafts were accepted under the agents' sampling temperatures.
- The runner turns DFlash2 off past position 32,768, where agent
  conversations quickly arrive (11 of 48 requests).

Controlled A/B on a 9,885-token code prompt, 700 tokens, thinking off:

| Sampling | Decode | Acceptance |
| --- | ---: | ---: |
| greedy | 57.5 tok/s | 49% |
| temperature 0.6 | 43–56 tok/s | 34–49% |

At this context an 8-row verify costs about 67 ms, about 2.7 single-token
decodes. That verify cost, not the shim, limits speculative gains in agent
workloads; the 4K benchmark's 100+ tok/s does not carry over to 10–35K
agent contexts.

## Live streaming and restart warm-up — 2026-09-27

### Live Responses and Messages streams (`live_stream.py`)

Codex and Claude Code now receive reasoning and answer text while the model
generates. Previously every event was buffered until the turn finished.

- **Codex (Responses).** The reasoning item's summary deltas stream first.
  The item then closes, and the message's `output_text` deltas follow. Tool
  calls and `response.completed` come at the end.
- **Claude Code (Messages).** A thinking block streams `thinking_delta`s and
  closes when the answer starts. Text deltas follow, and tool-use blocks come
  at the end.

The replay blob has to be sent before the reasoning item or thinking block
closes, which is before the raw turn is known. Streams therefore carry a
reference, `q38ref1:<id>`, that the server's raw-turn memory resolves; the
raw bytes are stored under the id when generation finishes. Non-streaming
responses keep the inline `q38raw1:` blob.

Measured on a Claude Code fix/compile/run task:
- 106 thinking deltas and 54 text deltas reached the client during
  generation.
- The follow-up `--resume` took 3 turns and 6.1 s, with 158 uncached input
  tokens.

The Codex fix and resume completed with only live continuations (no
divergences).

### System-prefix warm-up across restarts (`prefix_store.py`)

Shared prefix snapshots live in host memory, so a restart used to cost every
agent a full system-prompt prefill on its first request (Claude Code 15.7K
tokens, about 31 s).

With `--prefix-store FILE`, the shim records the prefix boundaries of
successful requests. The launcher uses
`~/.cache/qwen38-server/prefixes.json`; the file contains the system prompts
and is written 0600. On startup, a background thread replays the
`--prefix-warmup` (default 4) most recent entries as zero-output requests.
The runner does not add a duplicate prompt snapshot when a prompt is exactly
a shared prefix. The launcher now keeps 16 snapshot entries.

After a restart, the three stored prefixes re-prefilled in the background:
pi 4,430 tokens in 9.2 s, Codex 2,321 in 5.0 s, Claude Code 13,185 in 26.5 s.
The first Claude Code request then restored 13,185 tokens and finished in
9.3 s, instead of about 38 s cold.

### Three-agent task check

Task: write a test for a stack library, add a `make test` Makefile target
(`-fsanitize=address`), and fix the library. It has two planted bugs:
`realloc` without `sizeof(int)`, and `pop` reading one past the top.

| Agent | Result | Wall | Notes |
| --- | --- | ---: | --- |
| Claude Code | both bugs fixed, `make test` passes | 69 s | 9 turns, 117K cached / 6.4K uncached input |
| Codex | both fixed, passes | 252 s | Codex's workspace-write sandbox blocks `ptrace`, so LeakSanitizer fails; the model diagnosed it and set `ASAN_OPTIONS=detect_leaks=0` |
| pi | both fixed (`data[--s->size]`), passes | 252 s | |

The shim now logs `[tool-call] unparsed: ...` when the model writes
`<tool_call>` but no call parses; the client then receives plain text. No
such case occurred in these runs.

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
`CLAUDE_CODE_AUTO_COMPACT_WINDOW` to the server's context, read from
`/props` (Claude Code otherwise assumes 200K), and the output cap to 16K.

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
