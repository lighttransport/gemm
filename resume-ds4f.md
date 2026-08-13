# DS4F HTTP validation and resumption runbook

This is the current hand-off document for validating the GEMM DS4F runner as a
coding-agent server.  It intentionally replaces older kernel-tuning notes and
unverified performance claims.

## Resuming prompt

> Continue the GEMM DS4F HTTP-server validation in
> `/mnt/nvme02/work/gemm/ds4f`.
>
> Do not optimize or modify llama.cpp.  Use llama.cpp only as the behavioral
> reference.  Its next validation run must use:
>
> `/mnt/disk2/models/ds4f-0731-gguf/ds4f-256x8.4B-0731-MXFP4_MOE-00001-of-00018.gguf`
>
> Compare that reference with our GEMM runner in the coding-agent scenario:
> deterministic output, prompt/prefill behavior, decode behavior, context
> switching, KV-cache reuse, durable system-prefix cache reuse, cache misses,
> eviction, and error recovery.  Preserve model quality; do not accept a speed
> change that changes greedy token IDs or tool-call behavior.
>
> The acceptance reference is approximately **40 prompt tok/s** for uncached
> prefill and **9 decode tok/s**.  Cached prompt tokens must be reported
> separately and must not inflate prefill tok/s.  Record actual medians and
> hardware/configuration; do not turn these numbers into hidden environment
> tuning knobs.
>
> Read this file and `a64fx/llm/ds4f_serve.md` before running anything.  Keep
> production tuning as explicit command-line arguments.  Do not push changes.

## Scope and invariants

The system under test is:

```text
OpenAI/Anthropic/Codex HTTP client
        -> a64fx/llm/ds4f_serve.py
        -> cooperative Unix-socket runner
        -> libds4f_serve.so (GEMM DS4F, CPU + HIP)
```

llama.cpp is a reference implementation, not a second optimization target.
Use the same model family, tokenizer, greedy settings, prompt, context length,
and output limit wherever the APIs allow.  If the model conversion or chat
template differs, compare token IDs and decoded text only after documenting the
difference.

Quality gates:

- Greedy output must be token-identical for the agreed deterministic prompts,
  or every mismatch must be recorded with the first divergent position.
- Tool calls must remain valid JSON with the same tool name and arguments.
- Context continuation must not repeat, omit, or reorder prior messages.
- A cache hit may reduce work, but may not change the output.
- A cache restore failure must fall back to a fresh prefill, never to a partial
  or mismatched KV state.

## Build and launch

Build the GEMM serving library and tokenizer tools from the repository root:

```sh
sh a64fx/llm/build_ds4f_serve.sh
python3 -m py_compile a64fx/llm/ds4f_serve.py \
  a64fx/llm/ds4f_serve_runner.py a64fx/llm/bench_ds4f_http.py
```

Stage a single-node model using the project’s existing staging procedure.  Set
`DS4F_STAGE_DIR` to that manifest directory and `TOK` to the matching
tokenizer.  Then launch the wrapper, for example:

```sh
DS4F_STAGE_DIR=/tmp/ds4f_single \
TOK=/mnt/disk2/models/ds4f-0731-gguf/tokenizer.json \
DS4F_SERVE_USE_HIP=1 DS4F_MAXPOS=16384 \
  sh a64fx/llm/run_ds4f_single_serve.sh \
    --agent-cache-max-tokens 14336 \
    --single-prefill-quantum-tokens 1024 \
    --prefill-quantum-tokens 32 \
    --decode-quantum-tokens 4
```

Use the actual tokenizer shipped with the staged checkpoint if that path does
not exist.  Keep the runner and frontend logs; they contain cache-hit, phase,
and error diagnostics.  Check readiness with:

```sh
curl -fsS http://127.0.0.1:8080/health
curl -fsS http://127.0.0.1:8080/v1/progress
```

## llama.cpp reference run

Build or select the existing llama.cpp binary, but do not change its source or
performance defaults for this comparison.  Use the full 0731 GGUF above (all
18 shards must be present), for example:

```sh
MODEL=/mnt/disk2/models/ds4f-0731-gguf/ds4f-256x8.4B-0731-MXFP4_MOE-00001-of-00018.gguf
~/work/llama.cpp/build/bin/llama-server \
  -m "$MODEL" -ngl 43 -c 32768 -b 256 -ub 256 \
  --flash-attn on --host 127.0.0.1 --port 18080
```

Use the actual llama.cpp binary location and backend available on the host.
The command is a reference run only; do not add optimization patches to
llama.cpp.  Capture `/v1/models`, startup diagnostics, and the exact command.

## Common deterministic test fixture

Use one small prompt and one long coding-agent prompt.  Save both the request
JSON and response JSON/SSE stream so the comparison is reproducible:

```json
{
  "model": "ds4f",
  "messages": [
    {"role":"system","content":"You are a precise coding agent. Be concise."},
    {"role":"user","content":"Write a C function that returns the first index of x in a sorted array, or -1."}
  ],
  "temperature": 0,
  "top_p": 1,
  "max_tokens": 128,
  "stream": false
}
```

For the long fixture, use the real Codex/Claude developer and tool payload
from the client, plus a final short user task.  Do not hand-edit the serialized
prompt between implementations.  Record token counts from each server.

## Test matrix

### 1. Output and quality parity

Run the short fixture three times against each server with greedy sampling.
Compare:

1. generated token IDs (preferred);
2. decoded UTF-8 text;
3. stop reason and EOS handling;
4. tool-call JSON, if tools are supplied.

If token IDs differ, save both sequences and identify the first divergence.
Do not label a run “quality-preserving” from visual similarity alone.

### 2. Speed and phase accounting

Use the benchmark helper against the GEMM HTTP endpoint:

```sh
python3 a64fx/llm/bench_ds4f_http.py \
  --url http://127.0.0.1:8080/v1/responses \
  --max-tokens 128 --warmup 1 --repeat 3 \
  --expected-prefill-tok-s 40 --expected-decode-tok-s 9
```

Run the equivalent request against llama.cpp, then record median, min, and
max.  Report:

- total prompt tokens;
- uncached prompt tokens and cached prompt tokens separately;
- time to first generated token;
- prefill tok/s = uncached prompt tokens / prefill time;
- decode tok/s after the first token;
- backend, GPU, CPU threads, context length, batch/ubatch, and KV type.

Do not count HTTP serialization or cached tokens as model prefill.  A short
prompt can make timing noisy; repeat with a 4K–10K token prompt as well.

### 3. Conversation context switching

Use two independent context IDs, A and B, and alternate requests:

```text
A: define a function named alpha and remember its invariant
B: define a function named beta and remember its invariant
A: repeat alpha's invariant
B: repeat beta's invariant
```

Verify that A never sees B’s history and vice versa.  Repeat once with
`previous_response_id` (Responses API) and once with an explicit
`X-DS4F-Context-ID`.  Record `/v1/progress` and `/v1/contexts` before and after
each request, including warm/disk states and queue depth.

### 4. KV and durable agent-prefix caches

For each agent namespace (`codex`, `claude-code`, and `opencode` where used):

1. cold request: confirm `cached_tokens=0` and full prefill;
2. identical request after restart: confirm a durable cache hit and identical
   output;
3. same stable developer/tools prefix with a new final user message: confirm
   only the tail is prefetched;
4. change one system/tool byte: confirm a cache miss, not an unsafe restore;
5. delete or truncate the KV file: confirm clean fallback and no crash;
6. exceed `--agent-cache-max-tokens`: confirm the request remains correct and
   reports no cache hit;
7. exercise context TTL/disk spill and restore, then compare output again.

The sidecar token IDs, model identity, tokenizer identity, and cache schema are
part of the safety check.  Never accept a hit based only on filename or prompt
length.

### 5. Contention and scheduling

Submit two long prefills concurrently, then repeat with one request.  Check
that a lone request uses `--single-prefill-quantum-tokens`, while competing
requests use the fair-share quantum.  Verify both requests complete, neither
context is corrupted, and decode begins only after its own prefill is ready.
If testing multi-context decode, opt in with `--decode-batch-size`; keep the
default single-context path as the quality baseline.

### 6. Error and restart behavior

Exercise malformed JSON, an invalid context ID, client disconnect during
streaming, runner restart, missing cache sidecar, and an unavailable GPU.
Expected behavior is an explicit HTTP error or a safe CPU/fresh-prefill
fallback, followed by a server that can accept the next valid request.

## Results log template

Append a dated record rather than overwriting prior evidence:

```text
Date/host:
GEMM commit:
llama.cpp commit:
Model + tokenizer:
Backend/device/threads:
Context/batch/ubatch/KV:

Quality: first differing token =
Tool-call parity:
GEMM cold: prompt=  cached=  prefill=  decode=
GEMM warm: prompt=  cached=  prefill=  decode=
llama.cpp: prompt=  cached=  prefill=  decode=
Context-switch result:
KV/durable-cache result:
Errors/recovery:
Logs/artifacts:
```

Do not claim the 40/9 acceptance bar passed until the run has both measured
rates and the quality/context/cache checks above.

## Results log

### 2026-08-13 — baseline, reconciliation, and two forward-path fixes

Host: Threadripper 1950X (16C/32T, Zen1), 188 GiB DDR4 single NUMA,
RX 9070 XT (gfx1201, 16.3 GiB, PCIe Gen5 x16), RTX 5060 Ti (16 GiB, Gen3 x8).
Model + tokenizer: `/tmp/ds4f_nocopy_stage_mtp` (NOCOPY manifest over the 48
`/mnt/nvme02/models/ds4f-0731/*.safetensors` shards, 72317 tensors incl. MTP,
156.02 GB loaded) + `/mnt/nvme02/models/ds4f-0731/tokenizer.json`.
Backend: HIP dense bank on the 9070 XT, routed MXFP4 experts on CPU.

#### The 9-10 tok/s figure was llama.cpp's, not this runner's

`/mnt/nvme02/work/llama.cpp/da4f.md` records, on this same host: full MXFP4 with
all MoE on CPU = **7.33 tok/s**; Q3_K_M with the first 40 MoE layers on CPU =
8.57 tok/s; ROCm 8.97 → 9.13 (`--no-op-offload`) → 9.22 tok/s; CUDA/5060 Ti best
single run 9.98, typical 9.69. It also states the 10 tok/s target was **not**
reached within the Q3/IQ3 quality budget. So 9-10 tok/s is a Q3-quantized
llama.cpp result; the like-for-like full-MXFP4 llama.cpp number is 7.33 tok/s.

Note: no llama.cpp binary on this host could load the model until rebuilt — the
installed binaries predated `LLM_ARCH_DEEPSEEK4`. `cmake --build build` fixes it.

#### Measured decode (in-process `ds4f_serve_bench.py`, 1024-token prompt, warm 16, 64 decode)

| config | prefill tok/s | decode tok/s |
|---|---:|---:|
| current defaults (16 threads, f32 activations) | 7.56 | 3.93 |
| 14 threads | 7.25 | 4.55 |
| W4A8 activations, 14 threads | 12.67 | 4.76 |
| **W4A8 + group-split dispatch, 16 threads** | **13.31** | **5.07** |

HTTP `/v1/responses` at 1145-token context is noisier (±30% run to run):
2.7-3.6 tok/s decode. `DS4F_SERVE_TIME=1` shows the forward itself at
**251 ms/token** mean (n=96, 250-266 ms) against a 277 ms/token wall, so the
Python/cooperative serving layer costs only ~26 ms/token (9%) — the serving
path is not the bottleneck.

#### Decode phase profile (de-nested; `DS4F_PROF=1`, 128 decode tokens, tuned config)

routed experts 50%, Tier-B2 compressor+indexer (`tb2prep`) 16%, qkv 7.6%,
o_proj 7.2%, mHC 6.5%, head 3.8%, **attention proper only 3.7%**, shared 3.5%,
router 1.1%.  `pool_dispatch` is 2.66 us/call, so the ~774 per-token dispatches
cost only ~2 ms — dispatch count is *not* the expert bottleneck, contrary to the
estimate in `hetero/ds4f/README.md`.

Routed-expert bandwidth, measured directly (`DS4F_EXPERT_BW=1`, decode only,
identical 275.943 GB / 20640 CPU-expert workload in both runs):

| config | expert GB/s | ms per expert |
|---|---:|---:|
| defaults (f32 activations, no group split) | 32.3 | 0.41 |
| **W4A8 + group split** | **41.4** | **0.32** |

Standalone `hetero/ds4f/build/bench_expert_bw --i8seq` at 16 threads on this
host measures 44.4 GB/s over an 8 GiB region, 40.5 over 40 GiB and 37.8 over
100 GiB. **The model path at 41.4 GB/s is therefore at or above the standalone
kernel roofline for a comparable working set — the expert phase is done.**
The f32 path at 32.3 GB/s likewise beats its own 31.0 GB/s standalone figure.

An earlier draft of this entry claimed ~21.6 GB/s and "half the bandwidth
unrealized". That was wrong: it divided 3.44 GB/token by a phase time taken
from a `DS4F_PROF=1` run, and that instrumentation inflates decode by ~30%.
Phase *shares* from DS4F_PROF are usable; absolute GB/s from it is not.

#### Changes made

1. `common/ds4f_impl.h` — `ds4f_mv_multi_worker` now splits the fused group's
   **concatenated** row space across the pool instead of giving every thread a
   1/N slice of every matrix. The expert gate/up group is 12 matrices, so the
   old split made each thread restart a ~300 KB stream twelve times per layer.
   `ds4f_mv_task` gained an explicit `(r0, r1)` range; zero means "derive from
   tid/nthr", preserving every existing call site. Runner flag
   `--mv-group-split` (default 1), env `DS4F_MV_GROUP_SPLIT`.
   **Bit-exact — verified**: greedy output identical to the previous default on
   both fixtures (each output row is still computed by one thread with an
   unchanged inner k-loop).
2. `a64fx/llm/ds4f_serve_runner.py` — new `--threads`, `--mv-group-split`,
   `--mxfp4-w4a8` arguments, set into the library's documented environment
   names before load so deployments are reproducible from the command line.
3. `a64fx/llm/ds4f_serve_lib.c` — startup line
   `[serve] forward: exact=… mhc=… tierb2=… w4a8=… group_split=… threads=… spec=…`
   and a one-shot `[serve] decode attn: …` line. Several of these flags are
   silently downgraded during load, so echoing the requested value is not
   enough. This is how the W4A8 default was found.

The two changes interact: with f32 activations the expert kernel is
dequant-bound and the group split is worth ~0 (3.93 either way); W4A8 makes it
bandwidth-bound, and only then does stream length pay.

#### Quality gate

Greedy, `temperature 0`, `top_p 1`, two fixtures (a 1145-token C code-review
prompt and a 1200-token long-range-recall prompt):

- group split alone: **identical output** — safe as a default.
- W4A8: **differs** on 1 of 2 fixtures (first divergence in the bug-attribution
  sentence; the recall fixture is unchanged). Therefore `--mxfp4-w4a8` defaults
  to **0**. Matches the historical "8/9 argmax, one expected W4A8 mismatch".
- Long-range recall is intact: a fact at position 0 of a 1.2K prompt is recalled
  correctly while generating at position ~1200.

#### Structural finding: speculative decode cannot currently win

`ds4f_serve_speculate` (`a64fx/llm/ds4f_serve_lib.c:628-637`) runs one full
`ds4f_forward_token` per **committed** token and breaks at the first mismatch,
so committed tokens always equal forwards. It is break-even minus the drafting
cost by construction, and measured slower (2.81 vs 3.28 tok/s at K=4).
`--speculative-tokens` also defaults to 0, and MTP tensors are only loaded when
it is non-zero (`ds4f_serve_lib.c:390`), so the DSpark path was entirely inert.
Making speculation pay requires a **batched** M=K verifier so one expert-weight
stream serves K tokens, plus snapshot/rollback for the Tier-B2 compressor ring
and the three DSpark KV rings. Not implemented.

#### Ceiling

Decode reads 3.44 GB of routed-expert weights per token. At the 39-43 GB/s
measured kernel ceiling that is a ~80-88 ms/token floor for the expert phase
alone, i.e. **~12 tok/s even if every other phase were free**, and experts are
only ~50% of decode. 18-20 tok/s is therefore not reachable on this host with
full-precision experts; it would require reading fewer expert bytes per token
(sub-FP4 weights or a smaller top-k), which changes model output.

#### Next levers, in expected-value order

1. ~~Close the expert bandwidth gap~~ — done; measured 41.4 GB/s, at/above the
   standalone kernel roofline. Further expert gains need a cheaper kernel
   (fewer ops per weight byte), not better streaming.
2. ~~`tb2prep`~~ — done, see below.
3. Batched speculative verification (see above).
4. `attn_gemm hit=0 miss=6888`: the fast 8-head-blocked attention path is gated
   off on every call. Worth little now that attention is only 3.7% of decode.

Artifacts: `/tmp/claude-1000/.../scratchpad/` — `ipbench.out`, `split.out`,
`sweep2.out`, `prof_decode.out`, `httpab.out`, `timed.out`, `out_{A,B,C}*.txt`.

### 2026-08-13 (later) — tb2prep

`tb2prep` (Tier-B2 layer compressor + lightning indexer) was ~16% of decode,
larger than attention itself. Its two dominant children are `tb2lcmp` (the
`cmp_wkv`/`cmp_wgate` projections, run on all 41 sparse layers every token) and
`tb2qproj` (the indexer `idx_wq_b` [8192, 1024] projection, 21 CSA layers).

**Cause**: both reach the pool through `ds4f_cmpmv_bf16_worker` and
`ds4f_bf16mv_worker` (`common/ds4f_impl.h`), and both of those had SVE
implementations with a **plain scalar x86 fallback** —
`acc += ds4f_bf16_to_f32(w[i]) * x[i]`, no AVX2 at all. This is the same class
of defect `resume.md` recorded for `DS4F_ATTN_GEMM`.

**Fix**: use the existing `ds4f_avx2_dot_bf16` (`common/ds4f_matvec_avx2.h`) in
both x86 branches, keeping the scalar loop for non-AVX2 builds.

| | before | after |
|---|---:|---:|
| decode tok/s | 5.079 | **5.273** (+3.8%) |
| prefill tok/s | 14.083 | **14.802** (+5.1%) |
| `tb2prep` (profiled, 136 tokens) | 6.587 s | **3.459 s** (-47%) |
| `tb2qproj` | 2.496 s | 1.023 s (-59%) |
| `tb2lcmp` | 2.765 s | 1.562 s (-44%) |

**Both are now at the DRAM read roofline**, so there is nothing further to win
on the CPU:

| | bytes/token | achieved | ceiling |
|---|---:|---:|---:|
| `tb2lcmp` | 520.1 MB | 45.3 GB/s | 48.1 GB/s (S0 read-only) |
| `tb2qproj` | 352.3 MB | 46.8 GB/s | 48.1 GB/s |

Quality gate: on the default (exact-activation) path the change is
**greedy-identical** on both fixtures despite the accumulation-order change.
On the opt-in `--mxfp4-w4a8 1` path the output does move — and it moves *onto*
the exact path's output, i.e. the W4A8 and exact configurations now agree on
both fixtures where previously they diverged on one.

Remaining option for `tb2prep`: offload to the 9070 XT. `cmp_wkv`/`cmp_wgate`
are already bound to VRAM in `serve_attach_hip` (`ds4f_serve_lib.c:248-256`),
but decode's `ds4f_tb2_prepare` passes raw host pointers and never uses the
binding; `idx_wq_b` is not bound at all. Moving all 872 MB/token to VRAM saves
~18 ms of DRAM traffic but adds ~41-62 launches; at the ~0.17 ms/launch this
host measures for fused GPU matvecs, the net is only ~4-8 ms/token (~2-4% of
decode). Recorded as a known option, not pursued.

Cumulative from the original defaults: decode **3.93 -> 5.27 tok/s (+34%)**,
prefill **7.56 -> 14.80 tok/s (+96%)**, with `--mxfp4-w4a8 1`; on the
bit-exact default path the gains are the group split and this tb2 fix.

### 2026-08-13 (later still) — tb2prep GPU offload

Follow-on to the AVX2 fix above: run the decode-time Tier-B2 projections on the
9070 XT dense bank instead of streaming their weights from host DRAM.

**Mechanism** — no new kernels were needed. `ds4f_compress_step` already accepts
precomputed `kv_pre`/`score_pre`, and `ds4f_index_step` a precomputed `q_pre`
(both are the batched-prefill hooks). `ds4f_tb2_prepare` now fills those from
device matvecs when the weights are resident, so the token-ordered compressor
recurrence that follows is untouched.

Two things had to be fixed to make it engage:

1. `idx_wq_b` was **never bound to the device** — not for decode and not for the
   batched prefill GEMM, which built its tensor with a hard-coded `gpu_id = -1`.
   `serve_attach_hip` now binds it on the CSA layers (`ds4f_serve_lib.c`), and
   the prefill GEMM uses the binding.
2. The first attempt routed the compressor pair through `ds4f_matvec_multi`,
   which **silently falls back to CPU for BF16** — its fast path requires every
   member to be FP8 or MXFP4 (`ds4f_impl.h:874-877`). Two plain `ds4f_matvec`
   calls take the device correctly.

| | CPU tb2 (AVX2) | GPU tb2 |
|---|---:|---:|
| decode tok/s | 5.357 | **5.670** (+5.8%) |
| prefill tok/s | 14.733 | 14.499 (-1.6%) |
| `tb2prep` (profiled, 136 tok) | 3.459 s | **2.093 s** (-40%) |
| `tb2qproj` | 1.023 s | **0.023 s** (-98%) |
| `tb2lcmp` | 1.562 s | 0.927 s (-41%) |

`tb2qproj` essentially disappears. `tb2lcmp` does not, because what remains is
the compressor recurrence itself (gate/norm/RoPE/ape and the ring append) plus
two device round-trips per layer — only the weight read moved.

**Quality gate: this one moves tokens.** Greedy output diverges from the CPU
path on the code-review fixture (identical prefix, then a differently-worded but
semantically equivalent continuation), on both the exact and the W4A8 path; the
long-range recall fixture is unchanged. A device BF16 matvec does not reproduce
the host AVX2 reduction order, so this is expected. It is therefore exposed as
**`--hip-tb2-decode`, default 0**.

Net position after all of this session's work:

| config | prefill tok/s | decode tok/s |
|---|---:|---:|
| original defaults | 7.56 | 3.93 |
| default today (bit-exact: group split + tb2 AVX2) | ~7.6 | ~4.0 |
| `--mxfp4-w4a8 1` | 14.80 | 5.27 |
| `--mxfp4-w4a8 1 --hip-tb2-decode 1` | 14.50 | **5.67** |

The bit-exact default path gains little on its own: the group split needs W4A8
to pay (the f32 expert kernel is dequant-bound, not bandwidth-bound), and the
tb2 AVX2 fix is worth a few percent. Both flags that carry the real speedup
change greedy tokens and are off by default.
