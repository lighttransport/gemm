# Decode Graph Capture Audit — Hybrid Models

**Date:** 2026-05-17
**Scope:** Can the `!r->is_hybrid` exclusion in `hip_llm_phase5_capture`
(hip_llm_runner.c:6433) be relaxed for dense-hybrid models?

## Methodology

Walked every kernel launch in `forward_one_layer` (hip_llm_runner.c:6196-6407)
for the M=1 decode path. For each, checked:

1. Are any per-token-varying parameters passed by **host value** (literal int)
   instead of via the device-side `r->d_position` pointer? Host values are
   captured at graph build time and become stale on subsequent replays.
2. Are there any `hipDeviceSynchronize` / host-side D2H / branching on host-read
   values? These cannot be captured.

## Findings

### SSM branch (`r->is_hybrid && cl->is_ssm`, lines 6209-6246) — ✅ capturable

| Kernel | Position-dep arg | Capture-safe |
|--------|------------------|--------------|
| `launch_matvec_auto` ×4 (qkv/z/alpha/beta) | none | ✓ |
| `launch_softplus_mul`, `launch_sigmoid_inplace` | none | ✓ |
| `launch_conv1d` | none — advances `cl->d_conv_state` in-place | ✓ |
| `launch_l2_norm_heads` ×2 | none | ✓ |
| `launch_repeat_tile` ×2 | none | ✓ |
| `launch_deltanet_step` | none — advances `cl->d_recurrent_state` in-place | ✓ |
| `launch_gated_rmsnorm_silu` | none | ✓ |
| `launch_matvec_auto` (ssm_out) | none | ✓ |

The conv and recurrent states advance correctly under graph replay: the kernels
read the current state from a fixed device pointer, update it, and write back
through the same pointer. Each replay sees the result of the previous replay's
write — exactly the desired step-by-step recurrence.

### Gated-attention branch (`r->is_hybrid && !cl->is_ssm`, lines 6247-6277) — ✅ capturable

| Kernel | Position-dep arg | Capture-safe |
|--------|------------------|--------------|
| `launch_matvec_auto` ×4 (Q/K/V/output) | none | ✓ |
| `launch_deinterleave_qgate` | none | ✓ |
| `launch_qknorm` ×2 | none | ✓ |
| `launch_rope_devp` ×2 | **position** — read from `r->d_position` | ✓ |
| `launch_kv_store_devp` | **position** — read from `r->d_position` | ✓ |
| `launch_attention_devp` | **position** — read from `r->d_position` | ✓ |
| `launch_sigmoid_mul` | none | ✓ |

All `*_devp` launchers source position from the device pointer set by the
caller (`hip_llm_forward_logits` line 6529 does
`hipMemcpyAsync(r->d_position, &position, ...)` immediately before
`hipGraphLaunch`). This is the exact pattern non-hybrid decode already uses.

### Common path — ✅ capturable

- `launch_rmsnorm` (pre-attn, pre-FFN, final): no position deps.
- `launch_add` residuals: no position deps.
- **DeepStack injection** (lines 6388-6392): only fires when `r->_ds_embd != NULL`,
  i.e. the prefill-with-vision path. Decode hits this branch with `_ds_embd == NULL`
  and skips. The existing comment at line 6429 confirms this.

### Dense FFN — ✅ capturable

`launch_matvec_auto` (gate, up, down) + `launch_silu_mul`. No position deps, no
host syncs.

### Still-blocking branches

- **MoE FFN** (`cl->is_moe`, lines 6317-6370): `hipDeviceSynchronize()` followed
  by `hipMemcpy` to host on `r->d_router_logits` (line 6327) to decode top-K
  expert indices on the host. Cannot be captured. The `!r->is_moe` gate must
  remain.
- **debug_layers** (lines 6396-6406): `hipDeviceSynchronize()` + `hipMemcpy` for
  per-layer norm print. Already gated.

## Recommended change

```c
// hip_llm_runner.c:6432-6433
-    r->graph_eligible =
-        !r->is_moe && !r->is_hybrid && !r->debug_layers;
+    r->graph_eligible =
+        !r->is_moe && !r->debug_layers;
```

Captures dense-hybrid models (Qwen3.6-27B, Qwen3.6-9B-A2B). MoE-hybrid
(Qwen3.5-30B-A3B) and pure-MoE remain uncaptured because of host-side
router top-K decoding.

## Verification plan

1. Apply gate change.
2. Build and run `test_hip_llm` on 27B Qwen3.6 IQ3_XXS with
   `--compare-paths` (decode path that exercises graph replay) for 4 tokens.
   Verify token-by-token output matches a `LLM_GRAPH_DISABLE=1` run.
3. Bench decode (16 + 64 tokens) for 27B / 9B and confirm no correctness
   regression in standalone generation.
4. Expected speedup: 5-10% on per-token decode wall-time.

## Anti-confidence flags considered

- **State buffers re-initialized between calls?** No — `cl->d_conv_state` and
  `cl->d_recurrent_state` are allocated once at model load and persist for the
  lifetime of the runner. The capture happens after weights are loaded and
  state buffers exist.
- **Capture's warm_pos=0 setup contaminates SSM state?** The capture replay
  walks the kernel sequence with `warm_pos=0`. SSM kernels read+write state,
  so this single warm pass *would* advance state by one token. Need to either
  reset state after capture or capture before any real inference. Looking at
  the existing code (line 6453: `hipMemcpy(r->d_position, &warm_pos, ...)` then
  `forward_blocks_body` runs inside capture), the non-hybrid case has the same
  concern for KV cache. The pattern is: capture is built *during init*, before
  any tokens are sampled — and the first real forward call writes over slot 0
  anyway. **For SSM, however, the state at slot N depends on the input at slot
  N, so the warm-pass-into-state IS a contaminant.** Need to add a
  state-buffer reset (`hipMemset(d_conv_state, 0, ...)`, ditto
  `d_recurrent_state`) after the capture's logits + hidden graphs are built.
