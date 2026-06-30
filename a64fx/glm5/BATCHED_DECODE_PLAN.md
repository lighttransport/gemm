# GLM-5.2 batched (multi-stream) decode — implementation plan

**Goal:** amortize the per-layer EP all-reduce across M concurrent decode streams so the
~78 collectives/token (99% of single-stream decode time) are paid once per *forward* instead
of once per *token*. Sim predicts M=16 → ~3.4 tok/s, M=32 → ~5.8 (from 0.25 at M=1).

## Finding
- `glm5_forward_batch_decode` (glm5_impl.h:2017) is **standard attention** (`wq/wk/wv`) and is
  only called from a **synthetic benchmark** (glm5_ep_runner.c:755). It does NOT implement
  GLM-5.2's **MLA** attention, so it can't generate real tokens. Needs an MLA version.
- The scaffolding exists: `glm5_mstream` (glm5_impl.h:1601) already has **per-stream KV caches**
  `kc/vc` shaped `[n][n_layers][max_pos][kv_dim]` and the design comment (1593-1600).

## Template
`glm5_forward_prefill_chunk` (glm5_impl.h:2179) — the **validated batched-MLA** path. It batches
S tokens of ONE sequence (contiguous positions p0+t, shared model KV). The decode version batches
M INDEPENDENT streams (positions `pos[t]`, **per-stream KV**), CP OFF.

## `glm5_forward_batch_decode_mla(m, X[M*H], M, pos[M], out[M])` — the 5 deltas vs prefill_chunk
1. **Positions:** `p = pos[t]` per stream (not `p0+t`). RoPE uses `pos[t]`.
2. **KV store (per-stream):** instead of `glm5_store_latent_kv` (→ model `L->kv_cache`), store the
   latent `kv[KVC]` into `ms->kc[(size_t)t*per + (size_t)l*max_pos*KVC + (size_t)pos[t]*KVC + i]`
   with `glm5_kv_enc`, where `per = n_layers*max_pos*KVC`. (vc unused for MLA latent; keep kc only.)
3. **Attention selection (per-stream, full causal, CP OFF):** stream t attends `j in [0..pos[t]]`
   of its OWN cache; replace `glm5_load_latent_kv` with a per-stream load from `ms->kc[t][l][j]`.
   The online-softmax math (hmx/hse, qabs/ctx) is unchanged. No `kv_combine_cb` (CP off) → no
   attention collective; the ONLY per-layer collective stays the MoE `ar_cb`.
4. **MoE:** unchanged from prefill_chunk (batched gate GEMM + per-(token,expert) experts +
   ONE `ar_cb` over `[M,hidden]`). This is the comm that gets amortized.
5. **Head:** run for ALL M streams (prefill does `need_head` last-token only). Batched lm_head
   GEMM `[M,vocab]`, then per-stream argmax (or `glm5_sample_logits` when `samp_temp>0`, looping
   t with the shared-seed RNG advanced per stream → lockstep preserved).

## Wiring
`run_cbatch` (glm5_ep_runner.c:471): its decode step currently calls `glm5_forward_token` **per
slot** (M=1 each → no amortization, line ~530). Replace with: gather the active slots' current
tokens → embed into `X[M*H]` token-major, fill `pos[M]` from each slot's position → ONE
`glm5_forward_batch_decode_mla` → scatter `out[M]` back to slots. Gate behind `GLM5_BATCH_DECODE=1`
(default off; falls back to the per-slot path).

## Allocation
`glm5_alloc_mstream_ex(m, M, per_stream_kv=1)` (glm5_impl.h:1983) allocates the `ms->kc/vc` and
batched scratch. Call once with M = max concurrent streams before the decode loop.

## Correctness without jobs
- **N=1 self-check (compile + cheap runtime):** `glm5_forward_batch_decode_mla(m,x,1,&pos,&out)`
  must return byte-identical argmax/logits to `glm5_forward_token(m,x,pos)`. Add under
  `GLM5_BATCH_SELFCHECK=1`; runs in any 1-node smoke and is the gate before trusting M>1.
- By construction it reuses the validated prefill batched GEMMs + the validated single-stream MLA
  attention math; only KV addressing (per-stream) and the head loop are new.
- **M>1 correctness (routing/KV independence) still needs a job** to confirm — do that first when
  hours return: decode the same M prompts batched vs single-stream, assert identical token streams.

## Risk / scope
~130 lines, intricate (per-stream latent KV indexing + the online-softmax attention). Untestable
at M>1 without a job, so it lands behind `GLM5_BATCH_DECODE` (off) until validated. Routed-expert
grouping (M tokens → shared expert GEMM) is a **follow-on** (v1 keeps experts M=1 per token; the
comm — not the expert matvec — is the bottleneck, so v1 already captures the predicted win).
