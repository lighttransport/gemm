# Qwen3.8 DFlash2 on RDNA4

## Draft embedding launch cleanup (2026-09-22)

Each DFlash2 proposal now decodes one anchor row and one fixed mask-token row
with the exact IQ1_M scalar kernel, then copies that device row for every
remaining mask candidate. The selector overwrites its scratch only after the
embedding and copies have completed on the same stream; setting
`LLM_QWEN35_DFLASH_EMBED_BROADCAST=0` restores the older exact
`embed_iq1_m_batch` A/B path. Target verification, selector state, and captured
graph arguments are unchanged.

The pinned 4K C++ merge benchmark improved greedy K=7 from the previous
79.88 tok/s run to 83.69 tok/s warm (140 drafted, 134 accepted), with stable
hash `44915ec1039a64c8`; draft time fell to about 264 ms for 140 proposals.
Seeded sampled K=7 measured 70.47 tok/s warm (140 drafted, 115 accepted) with
stable hash `630b7cbc72230e0d` and about 250 ms draft time. The full HTTP/stdio
cache, cancellation, concurrency, and multi-turn C++ quality harness passes.

Every non-anchor DFlash2 proposal row is the same mask token. The default
embedding path now decodes one anchor row and one mask row with the exact IQ1_M
scalar kernel, then performs device-to-device copies for the remaining mask
rows. Set `LLM_QWEN35_DFLASH_EMBED_BROADCAST=0` to restore the previous
row-batched A/B path. Matched 4K K=7 greedy runs retained 140 drafted/134
accepted and hash `44915ec1039a64c8`; draft time fell from 368--370 ms to
263--264 ms, and warm decode improved from 78.66--79.88 to 82.65--84.12
tok/s. The complete HTTP/stdio and multi-turn C++ quality suite, including
seeded sampling, passed with the default fast path.

The same default path at K=4 retained 124/124 accepted rows and the greedy
hash, measuring 55.51--56.19 tok/s across two warm repeats (`draft_ms`
274.5--283.8, `verify_ms` 2,188.5--2,197.9). A single-launch device broadcast
probe for the repeated mask rows was byte-stable but neutral at 55.51--56.19
tok/s, so the existing ordered device copies remain the simpler production
path.

## Five-row Q4_K projection specialization (2026-09-22)

K=4 has exactly five sidecar proposal rows. Its Q4_K/Q8_1 projection now uses
a compile-time five-row kernel that preserves the multi8 dot-product and
reduction order while dropping three unused accumulators and row-count tests.
Two warm repeats kept 124/124 accepted rows and hash `44915ec1039a64c8`;
`draft_ms` fell to 224.1--226.8 and decode rose to 56.75--57.06 tok/s,
compared with 274.5--283.8 ms and 55.51--56.19 tok/s for the previous
multi8 path. The K=7 path continues to use multi8: a sanity gate retained
140/134, hash `44915ec1039a64c8`, and 74.79 tok/s.

## Fused sidecar attention merge (2026-09-22)

The DFlash2 verifier now uses a fused attention kernel for schedules of up to
twelve splits.  Each warp computes one split and four adjacent rows, writes
its summary to LDS, and warp zero merges summaries in the same increasing
split order as the former attention-plus-combine pair.  This removes global
partial traffic and one launch while leaving target verification and captured
target graph ABIs unchanged.  Set `LLM_QWEN35_DFLASH_FUSED_ATTN=0` to run the
two-kernel control path; schedules above twelve splits always use that
fallback.

On the 4K IQ2_XS/Q8-Q8 gate, fused versus control measured 82.76/82.68 tok/s
for greedy K=7 with hash `44915ec1039a64c8`, 70.89/70.85 tok/s for seeded
sampled K=7 with hash `72a11474a3a222b5`, and 55.90/55.81 tok/s for sampled
K=4 with the same hash.  The HTTP/stdio cache, cancellation, concurrency, and
C++ quality checks all pass.  This is a small but repeatable sidecar win; the
ordinary target's random-token 64K decode remains 35.76 tok/s, so the
long-context 40 tok/s target still requires mixed projection work.

## DFlash2 long-window split retune (2026-09-22)

The sidecar attention launch now uses twelve partitions once its 2,048-token
window reaches 1,024 tokens.  The 1/4 split schedule for shorter windows is
unchanged, and `LLM_QWEN35_DFLASH_ATTN_SPLITS` remains an opt-in A/B override.
On a fixed 256-token K=7 gate, twelve partitions measured 56.57 tok/s versus
56.21 with eight; an EOS-limited K=7 gate measured 82.57 versus 82.14 tok/s.
K=4 measured 47.50 tok/s on the fixed gate and 59.69 tok/s on the normal gate.
Greedy K=7 retained sequence hash `44915ec1039a64c8`; seeded sampled K=7 and
K=4 retained `630b7cbc72230e0d`.  The production-default sampled K=7 run
reported `DFLASH2 sampled verifier=exact-window`, 69.28 tok/s, and the same
seeded hash.  The target verifier and captured graph ABI are unchanged.

At random 64K depth the same Q8/Q8 target path showed low acceptance (45/121)
and measured 24.02 tok/s before sidecar gating.  The generation harness now
disables DFlash2 at target position 32,768 and resumes ordinary target decode
at the transaction boundary.  The guarded run retains prefix hash
`90178de69a24a76e`, emits suffix hash `5821d77a630592cb`, and measures
35.67 tok/s.  Short-window DFlash remains enabled; the HTTP/stdio quality
harness passes after the guard.

## Default GQA reuse launch (2026-09-22)

Ordinary Q8/Q8 GQA decode no longer launches both context-specialized kernels
when one would return immediately. The existing three-head reuse kernel now
runs with its adaptive guard disabled for every context, while the older
three-head kernel remains a fallback if the reuse function is unavailable.
The C++ 4K gate retained hash `96b92d606dde5e28`; three repeats measured
40.24--40.31 tok/s versus 40.09 tok/s for the paired-launch control. A matched
random 64K run retained prefix `90178de69a24a76e` and suffix
`aed3c962c4a6525d`, measuring 35.47 tok/s versus 33.73 tok/s control. The
captured graph arguments and Q8/Q8 arithmetic are unchanged.

## Asynchronous accepted-row publication (2026-09-22)

`hip_llm_qwen35_mtp_commit` now queues the selected recurrent and convolution
state copies, hidden state, logits, and position update on the target stream
without synchronizing the host before returning. Same-stream decode/propose
work remains ordered, while request reset and verifier host-logit boundaries
still synchronize. The complete DFlash2 HTTP/stdio quality matrix passed,
including seeded sampling, cancellation, cache restore, concurrency, and
multi-turn C++ output. Cached K=7 windows reported 0.23--0.57 ms commit times
on the resident gate with unchanged target token streams.

## IQ2_XS launch-bounds probe (2026-09-22)

The native one-row IQ2_XS decode kernel is compiled with
`__launch_bounds__(512, 1)` for its existing 512-thread production shape.
The kernel's arithmetic and reduction order are unchanged.  The 4K
random-token gate measured 43.00 tok/s with prefix/suffix hashes
`1c891c2232aa1b7f`/`f44846dacf013e9e`; the 64K random gate measured 35.84
tok/s at 445.82 tok/s prefill with hashes
`90178de69a24a76e`/`7463f176c9b85ba3`.  The exact C++17 gate remained
byte-identical (`44915ec1039a64c8`, SHA-256
`4a0cb461966fae9a9d9da3b73c1b0c686ce8ee9ac3895c228bc6a653bc99a354`) at
41.95 tok/s.  This is a safe small scheduling improvement; the strict 40
tok/s ordinary 64K target remains open.

A matching `__launch_bounds__` probe on the native IQ2_S, IQ3_XXS and IQ3_S
one-row kernels retained the exact 4K hashes but measured 42.84 tok/s on two
runs, slightly below the 42.9--43.0 tok/s control.  It was reverted so those
formats keep their existing shape-specific occupancy choices.

A separate 128-thread Q8 attention combine A/B also retained exact 4K and 64K
hashes but measured 35.78 tok/s at 64K versus 35.77 tok/s for the validated
256-thread merge.  It was removed as neutral; the long-context verifier tail
still needs a fusion that removes partial traffic rather than only changing
the combine width.

The IQ2_XS `__launch_bounds__(512, 2)` variant was also reverted after two
exact 4K runs ranged from 42.82 to 42.91 tok/s; the committed `512, 1` bound
remains the measured choice.

Native IQ2_XXS was tested with `__launch_bounds__(256, 1)` as well.  The exact
4K gate measured 42.94 and 42.87 tok/s across two runs, so that bound was
reverted and the existing dispatch remains unchanged.

The native Q2_K one-row kernel received the same `256, 1` bound for an A/B;
it retained the exact 4K hashes at 42.91 tok/s without exceeding control, so
the change was reverted.

An opt-in IQ3_S K/V pair launch retained the exact C++ response and 4K hash,
but measured 35.74 tok/s at random 64K versus 35.77 tok/s for the validated
separate projections.  It was removed; long-context attention still dominates
the saved small projection launch.

## Packed Q8/Q8 KV scales (2026-09-21)

Q8 K/V cache scales now use the same rounded FP16 contract as the stored
packed-F16 cache values.  The native attention kernels consume the 16-bit
scales directly, halving scale-cache traffic and preserving the existing
Q8/Q8 arithmetic.  The exact differential test passes 49,188,864 bitwise
comparisons, including 64K split counts and the multi-query reuse paths.  A
graph-captured 506-token zero-depth run completed at 43.1--43.2 tok/s with
stable output and `Result: PASS`.  The long-context operator differential is
also clean.  A real random 65,536-token depth run completed with a 443.33
tok/s prefix and 35.52 tok/s ordinary decode suffix, retaining prefix hash
`90178de69a24a76e`; the gate completed with `Result: PASS`.

The Qwen3.8 runner can load the
[IncoAI Qwen3.8-27B DFlash2 GGUF](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2-GGUF)
as an opt-in draft sidecar.  The implementation is native HIP and does not
invoke llama.cpp at runtime.

DFlash2 has a five-layer 5120-wide draft transformer with its own 32-head,
8-KV-head, 128-dimensional attention geometry and a 2048-token sliding
window.  It consumes the target inputs to layers 6, 20, 34, 48 and 62.  The
runner captures those features during each 512-token target prefill tile,
fuses them, and injects the resulting K/V rows into the draft cache.  A draft
step evaluates one non-causal block containing the target anchor followed by
mask tokens.  Its rank-256 selector walks the top-16 candidate lattice.

The target remains authoritative.  Draft tokens are evaluated by the Q8/Q8
multi-row target verifier.  Recurrent states and the target hidden state are
committed only through the accepted row.  The corresponding target features
are then injected into the draft cache, replacing the speculative rows.  A
rejection therefore cannot alter later target output.

The exact multi-row verifier is enabled for greedy, probabilistic, and coding
samplers. Every candidate row follows ordinary decode's projection, recurrent,
Q8/Q8 attention, and rollback arithmetic. The runner prints
`DFLASH2 sampled verifier=exact-window` for a sampled window. The target remains
authoritative: the sampler sees target logits and only accepted target state is
committed.

## Run

```sh
bash rdna4/llm/run_qwen38_gsq_rocm.sh --gpu-only-bench \
  --prompt-file tmp/qwen38/prefill-4k-512/coding-4096.txt \
  -n 4096 -s 8192 --ubatch 512 --kv-cache q8q8 \
  --qwen35-prefill-bf16 --qwen35-decode-graph \
  --qwen35-native-q8-prefill --qwen35-native-mmvq \
  --sampling-profile llama --temp 0 --seed 42 --decode 64 \
  --qwen35-dflash2 \
    /mnt/nvme02/models/qwen38/27b/dflash2/Qwen3.8-27B-DFlash2-Q4_K_M.gguf \
  --qwen35-dflash2-draft 7
```

Draft width may be 1 through 7.  The sidecar currently requires benchmark
mode or the resident stdio server, batched Qwen3.8 prefill, the decode graph,
and Q8 K plus Q8 V.  It is mutually exclusive with dense NextN and Qwen4 MTP.
The HTTP shim enables it with `--qwen35-dflash2 SIDECAR`; each request is
serialized through the exact propose/verify/commit window.  Temperature-zero
requests use the exact argmax window, while sampled requests verify full row
logits with the existing sampler.  Bounded target prompt snapshots include the
sidecar's recurrent K/V cache, target Q8 KV rows, captured features and prompt
logits, so cached requests can restore both target and draft state without
replaying the prompt. Snapshots above the configured token or byte budget are
rejected and replayed.

The opt-in HTTP quality gate covers deterministic greedy and seeded sampled
requests, direct stdio transactions, repeated-request state isolation,
disconnect cancellation after a generated token followed by recovery, and
concurrent request isolation:

```sh
python3 rdna4/llm/test_qwen35_dflash2_http.py \
  --model /mnt/nvme02/models/qwen38/27b/gsq/Qwen3.8-27B-GSQ-RCO-IQ2_XS.gguf \
  --sidecar /mnt/nvme02/models/qwen38/27b/dflash2/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
```

## Multi-context coding-agent serving

The Python shim uses `REQ3`, which adds a SHA-256 cache identity to the stdio
request. HTTP callers may supply `prompt_cache_key`,
`metadata.conversation_id`, `metadata.session_id`, or
`X-Prompt-Cache-Key`. The runner selects the longest exact token prefix only
within that identity. Successful prompt and system-prefix boundaries enter a
transactional host LRU; cancelled or failed work is discarded without
removing earlier committed entries. Configure it with:

```sh
python3 rdna4/llm/codex_server.py TARGET.gguf \
  --qwen35-dflash2 DFLASH2.gguf \
  --context-cache-entries 4 --context-cache-max-mib 2048 \
  --qwen35-snapshot-max-tokens 16384
```

The HTTP scheduler is FIFO. Each request may provide `request_id` or
`X-Request-ID`, and every response echoes `X-Request-ID`. A targeted
`POST /v1/cancel` body of `{"request_id":"..."}` cancels that queued or
active request. Disconnect cancellation carries the same request ownership,
so a disconnected queued client cannot signal the request currently using the
GPU. Execution is deliberately serialized: the target recurrent scratch,
sampler transaction, and DFlash verifier are one mutable device context, and
no exact multi-context decode batch has yet justified changing that contract.
The server reserves a request ID before committing streaming headers;
duplicates return HTTP 409, an idle unscoped cancellation returns 404, and an
exception after SSE headers closes that response instead of appending a JSON
body to the event stream.

Snapshots are accepted only when their recorded position exactly equals the
token-key length. Q8/Q8 scales are copied at their actual FP16 row size. This
check caught a stale batched-prefill position and an oversized scale copy that
short, same-context repeats could hide. The current forced A/B/A GPU gate
restores 6,535 tokens after an unrelated conversation, reports
`cached_tokens=6535`, and reproduces the exact greedy response. That snapshot
uses 448.3 MiB versus about 234 MiB for a 26-token prompt. The harness also
checks LRU eviction, malformed metadata, concurrent identities, deterministic
seeded sampling, cancellation recovery, and a two-turn C++ task by compiling
and running both generated programs. The same harness accepts `--mtp` in place
of `--sidecar` to validate resident Dense NextN, including a separate ordinary
target process for greedy, sampled, and C++ byte parity.

The ordinary target's sampled random-64K quality gate also passes: a
temperature-0.6, seed-42 suffix retains prefix hash `90178de69a24a76e`,
produces suffix hash `34e2f6bc082bc49f`, and completes 32 tokens with
`Result: PASS` after a 445.67 tok/s random prefix.

Grouped verifier attention now combines split partials and applies the Q8
attention gate in one launch.  The pinned greedy and sampled reference gate
still reports identical tokens, EOS, and output bytes, with warm K=7 decode
above 60 tok/s.

The reference validator accepts the sidecar directly:

```sh
python3 rdna4/llm/validate_qwen38_reference.py \
  --model /mnt/nvme02/models/qwen38/27b/gsq/Qwen3.8-27B-GSQ-RCO-IQ2_XS.gguf \
  --out tmp/qwen38/dflash2-quality-k7 \
  --reuse-reference tmp/qwen38/final-iq2-native-q2k \
  --native-q8-prefill --native-mmvq \
  --dflash2 \
    /mnt/nvme02/models/qwen38/27b/dflash2/Qwen3.8-27B-DFlash2-Q4_K_M.gguf \
  --dflash2-draft 7 --decode 256 --repeats 3
```

## Validation and performance

Measured on RX 9070 XT / gfx1201 / ROCm 10 with the IQ2_XS target, a 4096-token
C coding prompt, 512-token chunks, context 8192, greedy sampling, and Q8 for
both K and V:

| Path | Prefill tok/s | Decode tok/s | Accepted drafts | Sequence hash |
|---|---:|---:|---:|---|
| Ordinary target, recent baseline | 533.19 | 39.55 | — | `15f17d2640c1adfc` |
| Native DFlash2, K=4 | 537.42 | 54.49 | 37/40 | `15f17d2640c1adfc` |
| Native DFlash2, K=7 | 540.43 | 81.05 | 41/42 | `15f17d2640c1adfc` |

All three paths emitted the same 46-token response and EOS.  The response is
valid C and implements the requested inclusive clamp without overflow-prone
arithmetic:

```c
int clamp(int x, int lo, int hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}
```

A broader 4096-token C++17 merge-intervals fixture covers greedy and
temperature-0.6 sampling.  Both K=4 and K=7 match the pinned llama.cpp token
IDs, EOS and output bytes.  The generated function passes fixed edge cases,
ASan/UBSan, and 10,000 randomized cases.  Warm results were:

| Draft width / selection | Prefill tok/s | Decode tok/s | Output SHA-256 |
|---|---:|---:|---|
| K=4 / greedy exact window | 605.43–605.74 | 59.43–59.50 | `4a0cb461966fae9a9d9da3b73c1b0c686ce8ee9ac3895c228bc6a653bc99a354` |
| K=4 / sampled exact window | 606.85–607.42 | 55.79–55.82 | `ddd1752b6c2a44251b659516b5937fdaa0e84f464530607e493abf8bbc37c9ac` |
| K=7 / greedy exact window | 607.37–608.15 | 81.68–81.82 | `4a0cb461966fae9a9d9da3b73c1b0c686ce8ee9ac3895c228bc6a653bc99a354` |
| K=7 / sampled exact window | 605.37–605.85 | 68.67–68.78 | `ddd1752b6c2a44251b659516b5937fdaa0e84f464530607e493abf8bbc37c9ac` |

The 4096-token early-context retrieval fixture also returns exactly
`ZEPHYR-7319` with K=7, including the ordinary target's token sequence and
EOS.  These results are under `tmp/qwen38/dflash2-sampled-window-k4/`,
`tmp/qwen38/dflash2-sampled-window-k7/`, and
`tmp/qwen38/dflash2-retrieval-k7.*`.

The upstream llama.cpp server reference accepted 37/40 drafts at K=4 on the
same prompt, but measured 16.54 tok/s versus its 25.88 tok/s ordinary path.
The native K=4 implementation reproduces that acceptance exactly and is 3.29
times as fast.  K=7 is 105 percent faster than the retained recent ordinary
native baseline on this prompt.  DFlash prefill also clears the 500 tok/s
target.  The feature remains opt-in.

The optimized target verifier decodes IQ and Q2_K weights once for up to eight
candidate rows.  Quantization-format-specific kernels remove runtime codebook
branches; exact Q8_1 IQ1_S/IQ1_M kernels reuse each decoded group; IQ4_XS keeps
the reference's eight virtual sums.  The fixed-eight Q2_K kernel also finishes
one query at a time after decoding a weight block, while compact IQ2 and IQ3_S
schedules reduce live accumulators and retain the reference reduction order.
IQ3_XXS keeps its faster original shared schedule.  RMSNorm and
residual-plus-RMSNorm launch one independent block per candidate row. Q4_K
draft projections quantize each candidate activation in 32-value Q8_1 groups,
reuse one decoded weight chunk across all rows, and use gfx1201 packed integer
dots. Adjacent projections with the same activation reuse the quantized input.
This reduces the 4K K=7 draft phase from 109.739 to 77.239 ms. Draft arithmetic
may change rejected proposals; accepted tokens and state still come only from
the exact target verifier. The exact Q8/Q8 verifier attention
now loads each old K/V row once and evaluates up to eight adjacent causal
queries in the same four-wave block.  It preserves each query's quantization,
online softmax, packed-F16 accumulation and split-combine order.  The draft
also reuses Q4_K weights and holds one K/V vector while evaluating four mask
rows.

The DFlash selector now decodes the predecessor's 256-rank Q4_K vector once
per draft position into shared memory, then reuses it across all sixteen
candidate lanes. This leaves the candidate accumulation order unchanged and
keeps the greedy and seeded sampled verifier hashes identical. Repeated 4K
K=4 runs measure about 53--55 ms for the complete draft phase; the remaining
cost is in the five-layer projections and attention, so the selector change is
kept as a low-risk cleanup rather than counted as a headline speedup.

Sampled parity exposed three verifier-specific hazards. Non-FFN IQ1_S
projections must use the scalar path's MMQ-scale interpretation, so their
batched launch now uses the same MMQ-scale kernel with the candidate row in
the grid Y dimension. Attention split counts must be selected independently
for each causal query, so the native query grid applies the ordinary adaptive
split policy in one launch. Finally, all verifier projections share one Q8_1
scratch allocation inside the captured graph; each projection now restages
that scratch instead of treating source-pointer identity as proof that its
contents are still live. With these fixes, all 135 sampled target rows match
the former ordinary exact-target trace bit-for-bit, including every stored
248,320-entry logit row.

The exact fixed-eight Q2_K/IQ kernels and IQ4_XS multi-row projection use
eight-wave, 256-thread blocks. This changes only the assignment of eight
output rows to a block; each row retains its existing arithmetic. In matched
K=7 traces it reduced aggregate fixed-eight projection time from 223.706 to
222.336 ms and IQ4_XS multi-row time from 47.068 to 46.624 ms. The projection
differential passes 2,948,352 exact activation values and 13,191,360 bitwise
outputs with the production launch geometry.

These changes keep the target sequence unchanged while reducing the K=7
draft/verify/commit split to 77.239/478.857/10.624 ms for the complete
46-token response.  The eight-query attention operator takes 213.382
microseconds at 4K and 3.076784 milliseconds at 64K with eight splits.  The
pinned llama.cpp differential test passes 46,743,552 bitwise Q8/Q8 output
comparisons.  The expanded projection differential passes 2,948,352 exact
activation values and 13,191,360 bitwise Q2_K/IQ outputs, including all fixed
eight-row Q2_K/IQ kernels.  The emitted source passes
`gcc -std=c17 -Wall -Wextra -Wpedantic -Werror` and boundary tests using
`INT_MIN` and `INT_MAX`.

The fixed-shape GDA recurrence has separate contracts for raw scalar alpha and
prefill's precomputed decay.  Its fixed-bound loop preserves the generic
kernel's operation order under the runner's `-ffast-math` HIPRTC mode.  The
dedicated test compares 19,537,920 state/output values bitwise across both
contracts.  This restores the pinned sampled sequence while retaining the
specialized prefill speed; spelling the four rows as separate accumulators
reassociated operations and changed the sampled output.

Decode attention preparation now uses one exact kernel per attention layer for
Q/gate deinterleave, Q and K RMSNorm, Q and K M-RoPE, and Q8/Q8 K/V storage.
It replaces six launches with one in each of the 16 attention layers, removing
80 launches per decoded target row. The fused kernel retains the separate
operators' reduction order, trigonometric operations, Q8 scale rounding and
integer conversion. The complete K=4 and K=7 greedy and sampled C++ gates
remain byte-identical to the pinned llama.cpp fixtures. Both emitted functions
pass ASan/UBSan, fixed edge cases, and 10,000 randomized cases. At K=7, the
exact sampled window sustains 68.67--68.78 tok/s instead of the former
40.17--40.23 tok/s scalar fallback. K=4 sustains 55.79--55.82 tok/s; its
shorter draft does less useful work per verifier launch and remains below the
60 tok/s speculative target.

The earlier fixed-eight-split K=7 path measured 49.74 tok/s after a fully
processed 65,536-token random prefix, with suffix hash `2ddd068dca63669a`.
That result predates exact per-query adaptive split selection and is retained
only as a historical ceiling; it is not the current exact oracle.

The current verifier chooses between two captured graphs.  When every
adjacent causal row selects the same ordinary adaptive split count, the exact
eight-query kernel loads each K/V row once.  At a split-selector boundary it
uses the generic per-query grid.  This removes the fixed-eight arithmetic
shortcut without launching a full generic grid merely to return.  On the
random-64K gate, the generic exact baseline was 33.25 tok/s.  The selected
shared-K/V graph sustains 39.89--39.93 tok/s, with 443.45--444.22 tok/s
prefill, prefix hash `90178de69a24a76e`, and current exact suffix hash
`1c68ea2ff63ba5ab`.  All runs drafted 289 tokens and accepted 213; the final
draft/verify/commit range was 532.352--533.268 / 5818.192--5821.733 /
60.532--62.229 ms.  This is the pre-retune 16-split sidecar baseline; the
eight-split sidecar result below clears 40 tok/s with the same exact hashes.

The earlier DFlash2 draft attention retune used eight splits once its 2,048-token
window was at least half full (shorter windows retained the existing 1/4-split
schedule).  This was separate from the target verifier's adaptive split
selector.  On the random-token 64K K=7 gate, the exact target suffix hash
remained `1c68ea2ff63ba5ab` and the prefix hash remained
`90178de69a24a76e`; decode rose from the matched 16-split baseline of 39.91
tok/s to 41.01 tok/s.  The current production choice is twelve splits based on
the fixed-window comparison above.  The target verifier and its captured graph
ABI are unchanged; only the sidecar proposal attention launch is retuned.

The tested sidecar is
`Qwen3.8-27B-DFlash2-Q4_K_M.gguf`, SHA-256
`1a25c56858e1ebe93f2718ac1d49d1151f9323325c1bbfd6209370f4db131ebd`.

The ordinary target's recurrent alpha and beta F16 projections now use one
flattened launch while retaining the exact per-row FMA and XOR-reduction
order.  In a matched 65-row trace, this replaces 6,144 launches taking 23.102
ms with 3,072 launches taking 12.492 ms, or about 0.163 ms saved per target
row.  Zero-depth decode measures 42.83--43.04 tok/s and keeps the pinned
256-token hash `3c53b75f283cb9b0`.  The profile is under
`tmp/qwen38/ordinary-decode-profile-f16pair/`.

The dense FFN path also fuses SiLU multiplication with native Q8_1 staging
when the following down projection uses Q2_K, IQ2/3, or IQ4_XS.  This covers
58 of 64 layers in the tested IQ2_XS target and removes 58 launches per
ordinary decoded row.  The protected quotient and FP16 scale conversion match
the split quantizer: all 248,320 final logits are bitwise identical in the
fused/split A/B, and the 256-token hash remains `3c53b75f283cb9b0`.
Three-repeat zero-depth decode rises from a 42.79 tok/s split mean to a 43.11
tok/s fused mean.  The kernel trace is under
`tmp/qwen38/ordinary-decode-profile-siluq81/`; set the diagnostic
`LLM_QWEN35_SPLIT_SILU_Q81=1` to restore the two-launch boundary.

The 48 recurrent output layers now fold native Q8_1 staging into their
per-head gated RMSNorm/SiLU kernel.  The fused kernel retains the original
128-thread reduction, load loop, and stored activation boundary before four
independent waves quantize each head.  Its split-path A/B is bitwise identical
for all 248,320 final logits (SHA-256
`5b5f2f1a334ae644ac5633908e3d447c6d741c764a61dc0e9573addf699553c0`),
and three-repeat decode keeps the pinned `3c53b75f283cb9b0` sequence hash at
42.62--43.36 tok/s (43.07 mean), versus 41.64--43.14 tok/s (42.60 mean) for
the matched split path.  A 64-row trace removes exactly 3,072 launches, 48 per
row, and is under `tmp/qwen38/ordinary-decode-profile-ssmq81/`.  Set
`LLM_QWEN35_SPLIT_SSM_NORM_Q81=1` to restore the standalone quantizer.

Gated-attention output now applies sigmoid gating and stages native Q8_1 in
one wave-per-block kernel for 12 of 16 attention layers; the four IQ1 output
layers retain their dedicated quantizer.  The fused/split final logits are
bitwise identical with the same `5b5f2f1a...953c0` SHA-256, and all three
256-token repeats keep `3c53b75f283cb9b0`.  Matched decode means are 43.33
tok/s fused and 43.21 tok/s split.  A 64-row trace replaces 768 standalone
quantizer launches with 768 `sigmoid_mul_q81_f32` launches, reducing the total
by exactly 12 launches per row; it is under
`tmp/qwen38/ordinary-decode-profile-attngateq81/`.  Set
`LLM_QWEN35_SPLIT_ATTN_GATE_Q81=1` for the split path.

The exact one-row IQ scheduler now chooses its block geometry by projection
shape: IQ2_XS uses sixteen waves, the 17408-row IQ2_S shape uses sixteen, and
the frequent 5120/17408-row IQ3_S shapes use four. A wave still owns one
output row, preserving the reduction order. Matched 64-row traces save about
0.23 ms per decoded token; the tuned and eight-wave fallback paths produce
bitwise-identical 248,320-entry logits with SHA-256
`5b5f2f1a334ae644ac5633908e3d447c6d741c764a61dc0e9573addf699553c0`.
The authoritative 64K run keeps prefix/suffix hashes
`90178de69a24a76e`/`f4b35758fb99e6db` at 441.44 prefill and 34.98 decode
tok/s. `LLM_QWEN35_IQ_SHAPE_THREADS=0` restores the diagnostic fallback.

The verifier now embeds all IQ1_M candidate tokens with one two-dimensional
launch.  Accepted convolution and recurrent checkpoints are also published by
one kernel across every recurrent layer, replacing the former pair of commit
launches.  The post-change K=4/K=7 greedy gates retain sequence hash
`44915ec1039a64c8`; seeded K=7 retains sequence hash `630b7cbc72230e0d` and
the pinned output/token SHA-256 values.  K=7 measures 81.20 tok/s in the exact
gate, while the three warm embedding-batch runs measured 81.42--82.49 tok/s.

At each of the five target feature taps, capture now shares the existing exact
RMSNorm reduction and output loop.  The 4K gate remains byte-identical and
measures 536.17 tok/s cold, 610.86--612.05 tok/s warm, and 81.12--82.37 tok/s
decode.  A complete 65,536-token random prefix retains hash
`90178de69a24a76e` at 444.17 tok/s.  Overlapping sidecar injection with the
next tile is not yet safe: the target and injection GEMMs can select the same
shape-keyed hipBLASLt plan and workspace, so a second stream would race that
workspace until plans become stream-specific.

The real-GPU quality harness now runs the resident JSONL protocol directly and
then exposes the same backend through HTTP.  It checks greedy and seeded
sampling, three repeated cache hits, cancellation after a real streamed token,
recovery, and two concurrent callers.  The short gate passes, and a
6,600-token actual long prompt passes greedy and sampled reuse, active-window
cancellation, recovery, and request isolation.

Adaptive shared-K/V verifier graphs retain the complete pinned sampled trace:
sequence hash `630b7cbc72230e0d`, output SHA-256
`ddd1752b6c2a44251b659516b5937fdaa0e84f464530607e493abf8bbc37c9ac`,
token SHA-256
`fb7d8aeda396cdba5dd65b492a396ed3f91ae4312ea0a52d77be86355b4c7ee0`,
and logits SHA-256
`4b3489e92bcaf7f442b0f86722e88a5576cf7e16ea52193d73eecad12835ac3d`.
The 4K K=7 gate measures 82.67 tok/s greedy and 66.46 tok/s sampled with
trace I/O; K=4 remains exact at 59.75 tok/s.  A selector-boundary window uses
the generic graph, so graph reuse never changes a row's split count.

Four follow-up variants were rejected.  Device guards that launched both
attention grids reached only 39.55 tok/s at 64K.  Pinning a captured graph's
split count and sharing combine scales did not improve verifier time.  A
one-wave combine and a fused draft/verify synchronization path were both
byte-identical but slower on the traced 4K gate.  The retained graph selector
is the only measured win.  Moving the eight-query length metadata from
per-thread arrays to LDS also retained exact random-64K prefix/suffix hashes
and raised prefill to 445.21 tok/s, but verifier time increased to 5927.105 ms
and decode dropped to 39.26 tok/s; the register-pressure experiment was
therefore reverted.

## Remaining optimization opportunities

The current 4K gate sustains 66.46 tok/s for traced sampled K=7 and 82.67
tok/s for greedy K=7, clearing the 60 tok/s target.  DFlash reaches
39.89--39.93 tok/s after a real random-token 64K prefix. Ordinary one-token decode now
reaches 34.98 tok/s at 64K after exact GQA reuse, grouped K/Q scale products,
packed-probability reuse and scalar IQ codebook staging, so work that helps
both ordinary and verifier execution remains useful. The following order
reflects the remaining measured costs.

1. **Ordinary one-row target projections.** Scalar IQ2_XXS, IQ2_XS and
   IQ3_XXS now stage their small codebooks in LDS, lifting zero-depth decode
   from about 40.7 to 41.9--42.0 tok/s and 64K decode from 32.56 to 33.31
   tok/s. Hoisting repeated K/Q scale products in the exact three-head
   attention kernel raised the 256-token 64K run to 34.08 tok/s. Reusing each
   packed probability across both value tiles lowers the exact 128-split
   operator to 321.8--323.6 microseconds per layer and raises the full run to
   34.98 tok/s with unchanged prefix and suffix hashes, so the
   remaining gap is dominated by work outside attention. SiLU and Q8_1
   staging for 58 dense down projections now share one exact launch; gated
   RMSNorm and Q8_1 staging do the same for all 48 recurrent output
   projections, and sigmoid gating does the same for 12 attention output
   projections.
   Fixed-eight Q2_K/IQ projections already share decoded weights, but
   ordinary decode still streams weights for one row at a time. Reuse the
   quantized input across gate/up projections and investigate cooperative
   weight staging.  The opt-in IQ1 Q8_1 audit now reuses the activation bytes
   and block sums across the IQ1_S gate and IQ1_M up pair; it raises the
   measured 65,536-depth suffix from 34.18 to 36.13 tok/s with the same
   prefix/suffix hashes.  The seeded temperature-0.6 C++ response also kept
   sequence hash `630b7cbc72230e0d` and output SHA-256
   `ddd1752b6c2a44251b659516b5937fdaa0e84f464530607e493abf8bbc37c9ac`.
   It changes full logits and therefore remains diagnostic until the broader
   logit matrix is complete. A WMMA or reordered
   reduction path needs full output-token and logit validation because the
   current kernels preserve the target arithmetic order.
   The IQ1_M one-row F32 kernel now uses explicit non-aliasing qualifiers and
   unrolled four-value FMA halves; a matched 4K random gate moved 43.64 to
   43.72 tok/s with the same sequence hash.  A fused IQ1_S Q8_1 gate plus
   IQ1_M F32 up kernel was exact but 41.74 versus 41.90 tok/s on the pinned
   C++ gate, so it was removed.  The 64K ordinary 40 tok/s gap therefore
   remains a grouped mixed-type projection problem.
   A 128- and 512-thread block-shape probe for the native IQ2_XXS one-row
   kernel retained the 4K random prefix/suffix hashes
   `1c891c2232aa1b7f`/`ab4dd24f5cdf0b2c`, but three-repeat decode means were
   39.68 tok/s at 256 threads and 39.64 tok/s at 512. The alternate geometry
   was removed; its extra rows per block did not reduce the long-context
   projection cost.
2. **Verifier attention tail.** The query-grid verifier now selects ordinary
   decode's split count independently for every causal row. Equal-split
   windows now select a dedicated captured shared-K/V graph; split boundaries
   select the generic graph. The kernel still writes split partials for a
   second combine launch, and that pass dominates the long-context verifier
   tail. A prototype that extended the
   generic attention kernel ABI faulted under captured graph replay, even when
   the new body was disabled.  Use a separate verifier-only kernel and fuse
   the combine only if the selected split count and packed-F16 accumulation
   order remain exact.  A 64-thread exact merge probe retained the random
   64K hashes but fell from 35.77 to 35.64 tok/s, so it was removed; reducing
   partial traffic rather than only changing combine occupancy is still open.
3. **Hybrid recurrent tail.** Sequential candidate recurrence and rollback
   checkpoints are already batched and device-local. Alpha/beta F16 work now
   shares one exact launch per recurrent layer, and convolution plus recurrent
   state publication now shares one commit launch across all recurrent layers.
   DeltaNet checkpoint writes and the remaining matrix-vector work remain
   visible.  An attempted final-row copy elision changed target output after
   26 tokens because live state remains at the transaction origin; retain
   explicit accepted-row publication.

   The checkpoint copy path now makes the row-major convolution and recurrent
   strides explicit, so a DFlash window commits only its accepted row. GPU
   HTTP and pinned llama.cpp gates retain greedy and seeded-sampled token/byte
   parity after this hardening.
   A restricted-pointer/four-way-unroll DeltaNet scalar probe was exact but
   moved the 4K random gate from 42.90 to 42.80 tok/s, so it was reverted.
   The verifier now also uses the existing fused Q/K normalize-and-expand
   kernel, removing three intermediate launches per recurrent layer while
   preserving the same per-head reduction order.
   Attention verifier Q/K RMS normalization is likewise paired into one launch;
   its independent reductions retain the original order and exact output
   hashes.

   A three-repeat DeltaNet warp-per-row batch probe averaged 38.99 tok/s on
   the seeded sampled 4K gate versus 38.94 tok/s for the reference-order
   implementation, with identical sequence hash `c6bb94e73050164e` on every
   run. The gain is below measurement noise, so the warp path remains opt-in
   (`LLM_SSM_BATCH_WARP=1`) and the reference-order default is retained.

   The HTTP quality harness now forces another conversation between long-prompt
   requests, which proves restoration from host state instead of reuse of the
   still-live GPU context. The validated interleaved prompt is 6,535 actual
   tokens. Q8 target snapshots are bounded to 16,384 tokens by default; larger
   bounds can be requested with `--qwen35-snapshot-max-tokens N`, but they also
   need a sufficient `--context-cache-max-mib` budget. Earlier 10K--60K
   same-context observations must be rerun with this interleaved gate before
   being treated as portable-cache validation.
   Batched verifier SSM alpha/beta preparation now uses one elementwise launch
   for softplus/scale and sigmoid, with exact llama.cpp hashes preserved.
   A verifier-only eight-row combine kernel was also tested against the
   random-token 64K gate.  It remained exact but measured 28.52 tok/s versus
   the retained 28.82 tok/s dense-MTP control, so the generic captured combine
   remains in production.
4. **Kernel and graph count.** Q/gate deinterleave, QK normalization, RoPE,
   Q8/Q8 KV storage, SSM alpha/beta preparation, and IQ1_M verifier embedding
   are now fused or batched exactly. Profile again before joining another
   mixed-type preparation boundary; retain only savings larger than dispatch
   noise.
5. **Remaining draft cost.** Top-k and selector decisions already run on the
   GPU, and packed Q4_K/Q8_1 projections cut draft work substantially.  The
   current exact 64K suffix spends about 562 ms in the drafter after the
   eight-split sidecar attention change; projection and selector work remain
   the material cost.
   Position-parallel attention and
   cheaper draft-cache storage are the next candidates, provided K=4/K=7
   acceptance and authoritative output remain stable.

   The anchor and fixed mask-token rows now use one scalar IQ1_M mask embedding
   plus device copies for the remaining rows. The pinned greedy K=7 gate is
   exact at 82.65--84.12 tok/s warm (140 drafted/134 accepted, hash
   `44915ec1039a64c8`), and seeded sampled K=7 remains exact at 70.47 tok/s
   (140 drafted/115 accepted, hash `630b7cbc72230e0d`). Projection cost remains
   the material sidecar target after eliminating redundant mask embedding work.
6. **Prompt-cache injection.**  Feature capture now shares the target
   RMSNorm kernel and both 4K and random-64K prefill retain their targets.
   The hipBLASLt bridge now allocates scratch lazily per HIP stream, so a
   target and sidecar stream can safely share a shape without racing a
   workspace.  A two-stream BF16 smoke test passes with identical outputs.
   The opt-in injection path now records a target-ready event before the
   sidecar stream starts and retains an injection-done event before the next
   proposal reuses sidecar scratch.  The corrected overlap quality suite
   passes all HTTP/stdio and C++ checks.  A matched 4K K=7 run measured
   27.03 tok/s overlap versus 27.21 tok/s serialized with the same sequence
   hash, so the serial path remains the production default until overlap
   demonstrates a real throughput gain.

Each optimization should retain the exact sequence hash and response bytes at
K=4 and K=7, compile the emitted program, and cover non-coding prompts plus
random-token 64K depth.  The HTTP/stdio quality gate now covers direct and
OpenAI-compatible window transactions at short and long prompt lengths.
