# Qwen3.8-Flash-Next coding decode validation

The exact 16-GiB/256K production profile keeps scalar decode enabled.  The
explicit approximate profile uses resident-expert decode with an exact refresh
every six tokens:

```sh
LLM_QWEN4_APPROX_DECODE=1
LLM_QWEN4_DEVICE_HITS_ONLY=1
LLM_QWEN4_DEVICE_REFRESH_INTERVAL=6
LLM_QWEN4_APPROX_CPU_MIN_WEIGHT=0.20
```

The `0.20` value is the coding-profile default; an explicit
`LLM_QWEN4_APPROX_CPU_MIN_WEIGHT` now overrides it for controlled throughput/
quality sweeps. On the short 32-token prompt, 0.0, 0.1, and 0.2 remained near
50 tok/s, so the default is unchanged.

The refresh retains all CPU misses. On the two resident steps, the 0.20
threshold retains only high-probability cold routes on the CPU and computes
all resident routes on the GPU. Dropping every cold route was faster, but
generated a contradictory loop condition on an independent coding prompt.

For short contexts, the launcher defaults approximate decoding to the deeper
layers only (`LLM_QWEN4_DEVICE_REFRESH_START_LAYER=24`). The first 24 layers
stay on the exact route, which preserves syntax on the clamp control. Faster
experimental values such as `8` or `16` improve throughput on some prompts,
but an independent `max3` coding request produced malformed C at both values;
they are therefore not the quality-safe default. Set the variable to `0` to
reproduce the full-resident approximation for throughput diagnostics.

## Current 256K approximate coding control

On the RX 9070 XT with `QWEN38_CONTEXT=262144`, scaled-I8 KV, a 6000-MiB
expert cache, and the six-token cadence, the local OpenAI-compatible server
returned the complete coding response twice:

```text
int add(int a, int b) {
    return a + b;
}
```

The two end-to-end requests measured `34.54` and `40.40` decode tok/s.  A
64-token output budget also stopped naturally after the same 24-token response
at `34.08` decode tok/s.  Exact mode remains the quality/parity default; this
profile is an explicit throughput option.

## Historical RX 9070 XT short-decode profile

The current throughput profile uses one-row prefill scratch, an 8.2 GiB
expert cache, host-registered direct mapped misses, resident-hit routing, and
128-token refresh cadence:

```sh
LLM_QWEN4_BATCH=0 LLM_BMAX=1 QWEN38_MOE_CACHE_MB=8192 \
LLM_QWEN4_DEVICE_REFRESH_INTERVAL=128 \
LLM_MOE_COPY_PIPELINE=0 LLM_MOE_CPU_DECODE_MISSES=0 \
LLM_QWEN4_DEVICE_HITS_ONLY=1 LLM_QWEN4_MAPPED_MISSES=1 \
LLM_MOE_REGISTER_HOST=1 LLM_QWEN4_DIRECT_MISSES=1
```

Measured on the 6-token `Why is the sky blue?` prompt:

| Generated tokens | Prefill tok/s | Decode tok/s | End-to-end tok/s |
| ---: | ---: | ---: | ---: |
| 32 | 9.56 | **52.57** | 30.73 |
| 64 | 9.56 | **53.48** | 38.37 |
| 128 | 9.56 | **54.08** | 44.74 |

The decode path produces readable text but is approximate (resident-expert
routing and direct mapped misses); it is not exact-parity output. The 32-token
text check produced coherent sky/Rayleigh-scattering prose. End-to-end includes
the roughly 628 ms prefill and should not be compared directly with steady
decode tok/s.

Validated through the local OpenAI-compatible server on the RX 9070 XT,
with the production profile (65536 context, 7200 MiB expert cache) and the
non-thinking sampler:

```text
temperature=0.7 top_p=0.80 top_k=20 min_p=0.0
presence_penalty=1.5 repetition_penalty=1.0
```

| Prompt | Result | Decode tok/s |
| --- | --- | ---: |
| `clamp(value, low, high)` with swapped bounds | Complete valid C function | 37.83 |
| `max3(a, b, c)` | Complete valid C function | 37.72 |
| `count_lines(const char *s)` (four-step cadence) | Complete valid C function | 39.76 |
| `is_power_of_two(unsigned int)` (four-step cadence) | Complete valid C function | 36.57 |

Exact routing produced correct output at 16.67 tok/s. The four-step cadence
with high-weight cold routes preserves the tested coding completions while
sustaining more than 35 tok/s. It is still an approximation between refreshes; use
`LLM_QWEN4_APPROX_DECODE=0 LLM_QWEN4_DEVICE_HITS_ONLY=0` for fully exact
routing.

## Automated short-context coherence sweep

`make -C rdna4/llm approx-coherence` runs the explicit approximate profile at
1K, 4K, and 8K streamed prefill lengths with refresh intervals 4, 6, and 8.
Each point captures the generated C function and checks it with
`gcc -fsyntax-only`; the sweep reports prefill, steady decode, and end-to-end
rates separately. The standalone harness supplies the same explicit ChatML
user/assistant thinking frame as `codex_server.py` (the runner does not apply a
template) and normalizes the runner's visible BPE whitespace markers before
extraction. It disables synthetic last-token padding so each context length is
formed from real prompt tokens. Special-token bursts, malformed, truncated, or
uncompilable output fail the point; hash equality with exact mode is
intentionally not required for this opt-in path.

`QWEN38_APPROX_START_LAYER` controls the quality sweep's approximation
boundary (the sweep defaults to layer 40; production launchers remain
unchanged). A 978-token control with the layer-24 boundary produced
corrupted syntax, while moving approximation to layers 40--47 returned a
complete `clamp` function at 21.1 decode tok/s (exact control: 18.9 tok/s).
This is diagnostic evidence for a more conservative long-context profile, not
yet a serving-default change. Standalone coding sampling also filters ChatML
delimiter tokens; the JSONL server retains its existing stop-token semantics.

After fixing the sweep's continued-environment block, the 64-token coherence
gate passed at all three 1K refresh points and at 4K refresh 4 and 6. The 4K
refresh-8 point still hit the known gfx1201 startup failure before its first
forward when using the older 8.5-GiB cache. Reducing the sweep cache to the
validated 7.2-GiB headroom made all 1K and 4K refresh 4/6/8 points pass; the
first 8K point failed before its first forward with the 7.2-GiB/1024-row
profile. Reducing both pressure sources to the validated 5.9-GiB/512-row
profile made 8K refresh 4 and refresh 8 complete at about `19.35` prefill and
`19.9` decode tok/s with `PASS` and stable hash `a2d4f49620d5b663`. The sweep
now selects that profile automatically at 8K; callers can override both
`QWEN38_APPROX_CACHE_MB` and `QWEN38_APPROX_BMAX` for experiments. The
launcher keeps refresh 6 and an exact first-layer region as the quality-safe
sub-32K profile.

The harness accepts `QWEN38_APPROX_CONTEXTS` and uses `--prompt-file` for
prompts above 100K bytes, avoiding Linux `ARG_MAX` on 32K-class inputs. A
30,077-token streamed input loaded successfully with a 41,088-token sequence
allocation and stable 5.9-GiB/512-row VRAM usage; the long scalar run did not
reach a benchmark footer within the bounded observation window, so it is not
yet a 32K quality claim.

A follow-up 15,062-token prompt with the same 8K output capacity and
5.9-GiB/512-row profile also remained GPU-bound in scalar streamed prefill
without reaching a footer during the bounded observation window. VRAM stayed
flat while utilization stayed at 90--100%, indicating attention/context growth
as the limiter rather than allocation failure. Long-context quality therefore
still needs a subquadratic or device-batched attention path before a 16K/32K
serving claim can be made.

The flash and HTTP launchers now select the same 5.9-GiB/512-row profile
automatically for approximate 8K--128K requests on the 16-GiB card. Explicit
`QWEN38_MOE_CACHE_MB` and `LLM_BMAX` values still override that safety choice;
the 4K fast-prefill profile is unchanged.

## MTP sidecar

`/mnt/nvme01/models/q38nf/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf` is a
1.8 GiB one-layer NextN sidecar. It contains Qwen4exp block 48 plus
`blk.48.nextn.eh_proj`, `enorm`, `hnorm`, and NextN head-mixer tensors.
It is not loaded by the HIP runner yet. The 80+ tok/s target requires using
that sidecar to draft multiple tokens and verifying the accepted prefix with
the trunk; a sampled draft must never be emitted without trunk verification.

The runner now exposes a no-allocation contract check before that integration:

```sh
./test_hip_llm --inspect-qwen4-nextn \
  /mnt/nvme01/models/q38nf/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf
```

It validates the sidecar metadata plus all required block-48 tensor names and
the attention, MoE, fusion, and NextN hyperconnection shapes. The supplied
sidecar passes as `hidden=2560`, `heads=24/2`, `experts=512/10`, and `hc=4@320`.

The upstream Qwen4exp MTP graph confirms that `nextn.hnorm` is deliberately
wide (`4 * 2560`): each draft step consumes and exports the complete
hyperconnection residual, broadcasts the token embedding across its four
streams, applies `eh_proj` per stream, and then runs a full-attention + MoE
block with an independent draft KV cache. The inspector enforces that wide
state contract so a Qwen3.5-style narrow NextN head cannot be loaded by
mistake.
