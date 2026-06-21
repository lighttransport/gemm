# A64FX Multinode LLM Notes

## Qwen3.6 35B A3B prefill target

Target: 220+ tok/s prefill on 11 A64FX nodes for Qwen3.6 35B A3B.

Measured real prefill before the synthetic comm check:

| Actual tokens | Prefill time | tok/s | Mixer | FFN local | Allreduce |
|---:|---:|---:|---:|---:|---:|
| 1018 | 27.47 s | 37.06 | 19.01 s | 6.31 s | 2.09 s |

At 220 tok/s, a 1000-token prompt has a 4.55 s total budget.  The current
implementation therefore needs roughly a 6x reduction in end-to-end prefill
time.  The profile already points at compute/kernel structure: mixer dominates,
FFN is second, and communication is smaller than both.

Theoretical peak budget for 11 nodes:

| Mode | Peak per node | Cluster peak | FLOP budget at 220 tok/s |
|---|---:|---:|---:|
| FP32 | 6 TF/s | 66 TF/s | 300 GF/token |
| FP16/BF16 | 12 TF/s | 132 TF/s | 600 GF/token |

These are absolute ceilings.  Useful kernels must still keep packing, layout
conversion, activation, KV writes, routing, and synchronization inside the same
4.55 s budget for 1000 tokens.

### Qwen3.6-27B BF16 (multinode launch prep)

Local staging and launch path for the 27B BF16 split GGUF has been wired to use a
shared source shard set plus per-node `/local` caches:

- Source shard package target: `~/models/qwen36/27b/<N>nodes/` (defaults to `12nodes`)
- Per-node staging path: `/local/qwen36/27b/`
- Stage helper used by launchers: `a64fx/llm/stage_gguf_shards.sh`
- 27B TP/PP launchers now pass the staged local first shard:
- `a64fx/llm/run_tp_27b.sh` (defaults to `~/models/qwen36/27b/${QWEN27B_NODES:-NP}nodes/`)
- `a64fx/llm/run_pp_27b.sh` (same default behavior)

If you want a deterministic, one-time shard preparation step before any run, use:

```bash
# Prepare/refresh node-count-aware package from the shared source tree:
bash a64fx/llm/prepare_qwen36_27b_12nodes.sh \
  ~/models/qwen36/27b \
  ~/models/qwen36/27b/${QWEN27B_PREPARE_NODES:-12}nodes
```

The script copies split-family shards and validates destination sizes. If a shard is
already present and size-matched, it is reused. Use `QWEN27B_PREPARE=0` to
skip this step in launchers when package layout is already valid.

Usage pattern:

```bash
mkdir -p ~/models/qwen36/27b/11nodes
QWEN27B_PREPARE_NODES=11 bash a64fx/llm/prepare_qwen36_27b_12nodes.sh
NP=11 QWEN27B_NODES=11 ./run_tp_27b.sh   # or ./run_pp_27b.sh
```

Set `SKIP_STAGE=1` if the `/local` cache is pre-populated and `QWEN27B_LOCAL_DIR`
to override the staging root.

## Qwen3.6-27B BF16 TP status

- The non-hybrid attn tiled prefill path now runs for 27B/TP as intended.
  A guard in `common/transformer.h` that blocked attention-batched prefill on
  non-hybrid models has been removed; `transformer_prefill_gemm` now uses the
  tiled attention path for TP sharded Qwen-style heads too.
- `a64fx/llm/tp_pref_scale.sh` now supports:
  - long-context synthetic prompts (`TP_PREF_SCALE_TARGETS`, e.g. `1000`),
  - prefill+decode sweeps (`TP_PREF_SCALE_PREFILL_ONLY=0`,
    `TP_PREF_SCALE_DECODE_TOKENS=<N>`),
  - and optional cache-replay mode (`TP_PREF_SCALE_CACHE_MODE=1`) with explicit
    `TP_PREF_SCALE_CACHE_DIR` + `TP_PREF_SCALE_CACHE_TAG`.
- `a64fx/llm/tp_node_scale.sh` is added for node-count scaling sweeps; it calls
  `tp_pref_scale.sh` for each NP and aggregates rows into one file.

Example 11-node 1K-context prefill+decode probe:

```bash
QWEN27B_NODES=11 NP=11 \
TP_PREF_SCALE_TARGETS=\"1000\" \
TP_PREF_SCALE_PREFILL_ONLY=0 \
TP_PREF_SCALE_DECODE_TOKENS=32 \
TP_PREF_SCALE_CACHE_MODE=1 \
TP_PREF_SCALE_CACHE_DIR=$HOME/models/qwen36/27b/cache \
TP_SYNTH_TOKEN_ID=1 \
TP_SYNTH_TOKENS=1000 \
TP_PREFILL_GEMM=1 \
bash a64fx/llm/tp_pref_scale.sh
```

Node scaling sketch (targets and mode preserved for every NP):

```bash
TP_NODE_SCALE_NODES=\"11 10 9 8\" \
TP_NODE_SCALE_TARGETS=\"1000\" \
TP_NODE_SCALE_PREFILL_ONLY=0 \
TP_NODE_SCALE_DECODE_TOKENS=32 \
bash a64fx/llm/tp_node_scale.sh
```

Latest TP-27B BF16 control surface in `tp_pref_scale.sh`:

- `TP_PREFILL_ATTN_TILE=1` (enable tiled attention; `0` falls back to per-token)
- `TP_PREFILL_ATTN_TILE_Q` (default `32`) and `TP_PREFILL_ATTN_TILE_K` (default `256`)
- `TP_PREFILL_ATTN_TILE_ALGO` (`0` two-pass materialized, `1` online softmax, `2` packed three-pass, `3` packed two-pass)
- `TP_PREFILL_ATTN_TILE_STACK_BYTES` (default `2,097,152`)
- `TP_PREFILL_NULL_GEMM=1` (skip prefill GEMM work to isolate comm/KV-cache behavior)
- `TP_PREFILL_SSM_BLOCK=1` (parallelized SSM finish) and
  `TP_PREFILL_SSM_NULL_FINISH=1` (communication-only SSM finish check)
- `TP_PREFILL_FFN_ALLREDUCE` (default `1`) and `TP_AR_BF16=1` (default on runner + scale script)

For a first bottleneck check (null-gemm path) use:

```bash
TP_PREF_SCALE_TARGETS=1000 \
TP_PREF_SCALE_PREFILL_ONLY=1 \
TP_PREFILL_NULL_GEMM=1 \
TP_PREFILL_SSM_NULL_FINISH=1 \
TP_PREFILL_ATTN_TILE=1 \
TP_PREFILL_ATTN_TILE_ALGO=2 \
TP_PREFILL_ATTN_TILE_Q=64 \
TP_PREFILL_ATTN_TILE_STACK_BYTES=2097152 \
NP=11 QWEN27B_NODES=11 \
bash a64fx/llm/tp_pref_scale.sh
```

Keep `TP_PREF_SCALE_CACHE_MODE=1` plus `TP_PREF_SCALE_CACHE_DIR=$HOME/models/qwen36/27b/cache`
to measure warm+replay behavior over multinodes.

`TP_CACHE_SHARED` (default `auto`) enables efficient cache reuse for replicated-KV
TP layouts (for Qwen3.6-27B BF16 at TP=6,11,12): all ranks read and write
**one shared checkpoint** at `.../<safe_model>_<tag>.shared.tpck` instead of per-rank
files. If KV heads are non-replicated, the runner auto-falls back to per-rank checkpoints.
When `TP_CACHE_PATH` is used with `N>1`, shared mode is forced to avoid rank-local
file races.

## Current optimized EP prefill status

Implemented after the communication roofline:

- Batched q8-pv prefill GEMM path in `common/transformer.h`.  It pre-quantizes
  all `M` activation rows once, then runs the existing `matvec_sdot_8row` kernel
  over `(token, 8 output rows)` tiles.  This lets production q8 SSM/attention
  repacks use the layer-major prefill path instead of falling back to per-token
  matvecs.
- SSM/attention GEMM gates now accept either BF16 PV or q8 PV weights.
- `a64fx/llm/ep_pref_scale.sh` now enables q8 attention/shared-expert repacks
  when the matching prefill GEMM path is requested, and reports active/fallback
  layer counts plus per-module detail timings.  It also passes the tiled
  attention and SSM block-finish controls and reports SSM finish sub-buckets.
  `ATTN_TILE_K` is now logged and exported as `EP_PREFILL_ATTN_TILE_K`.
- Batched shared-expert prefill path: RMSNorm over the block, batched up/gate
  projections, pooled SwiGLU, batched down projection, scaled add into the EP
  FFN output block.
- Tiled full-block attention prefill path: precomputes Q/K/V for the whole
  prompt block, prepares RoPE/QK norm once, writes KV cache once, then computes
  causal attention through pooled head/query-tile workers.  The current pass
  layout is bounded two-pass score tiling, with an optional online softmax mode:
  - pass 1 computes `S = Q @ Kᵀ` for a `(q_tile × M)` block,
  - pass 2 applies stable softmax and accumulates `S @ V`,
  - pass 3 normalizes outputs in place.
  `EP_PREFILL_ATTN_TILE_ALGO=0` uses the fused materialized two-pass path
  (pass2 accumulation fused), `=1` enables online softmax (no score tile),
  `=2` uses explicit full-row three-pass materialized softmax, and `=3` uses the
  packed two-pass/three-pass variant with a fixed `[q_rows, k_tile]` score tile.

  Enabled by default with `EP_PREFILL_ATTN_TILE=1`; old chunked per-token
  attention remains available with `EP_PREFILL_ATTN_TILE=0`.
  Tune query block width via `EP_PREFILL_ATTN_TILE_Q` and score block width via
  `EP_PREFILL_ATTN_TILE_K`. `EP_PREFILL_ATTN_TILE_ALGO` defaults to `0`.
- Batched SSM finish path: convolution is parallelized over QKV channels, Q/K
  normalization is parallelized over token/group jobs, and the DeltaNet
  recurrence plus gate/RMSNorm runs as one pooled per-head scan over all tokens.
  Enabled by default with `EP_PREFILL_SSM_BLOCK=1`; the old per-token finish
  remains available with `EP_PREFILL_SSM_BLOCK=0`.  Measurement-only
  `EP_PREFILL_SSM_NULL_FINISH=1` zeros the SSM finish output after the projection
  stage.
- In non-tiled attention fallback (`EP_PREFILL_ATTN_TILE=0` or `pos0 != 0`),
  attention now writes each token directly into `AOut + t*q_dim` and applies the
  sigmoid gate in place, removing the per-token `memcpy` back into `AOut`.
- 3-pass/2-pass SSM scan path now iterates token-major in the pooled worker, so
  each token’s `(qkv, alpha, beta, z)` vectors are loaded once and reused by all
  local heads.
- Local FFN and shared expert now reuse the same per-token FFN RMSNorm buffer in
  the stable local-expert path.  This removes the second shared-expert RMSNorm
  pass when `EP_PREFILL_SHEXP_GEMM=1`.
- `a64fx/llm/ep_node_scale.sh` measures scaling over node counts by regenerating
  an exact-NP `tofu_topo.txt` for each run.  `ep_pref_scale.sh` now exposes
  `EP_MOE_BF16_PV` and `EP_MOE_BF16_PV_MAX_LAYERS` so memory-pressure tests can
  be recorded without editing the script.
- `ep_pref_scale.sh` now stages the model once to `/local` for mpiexec launches and
  reuses it with `EP_SKIP_STAGE=1` on all target runs, while keeping fallback to
  source if `/local` staging is unavailable.
- Added attention stack guard via `EP_PREFILL_ATTN_TILE_STACK_BYTES` (default
  2 MiB per worker scratch task).  This keeps tiled attention stable for
  1000+ token prompts by bounding temporary `scores`/`row_max` allocations.
- Added a hard fallback in the tiled-attention worker: if heap scratch allocation
  fails, the worker now switches to online-softmax mode (`EP_PREFILL_ATTN_TILE_ALGO=1`)
  instead of falling back to oversized stack `scores`. This closes the 1000-token
  segfault window under aggressive `q_tile`/`k_tile` settings.
- Added vectorized shared-expert and FFN epilogue paths on A64FX:
  - shared-expert gate score (`x_norm`·`ffn_gate_inp_shexp`) now uses SVE
    `tf_dot` on tiled prefill and local paths,
  - shared-expert residual accumulation (`dst += score * out`) is now SVE
    row-scaled and avoids scalar per-element loops,
  - `tf_silu_mul_avx2` now also has an SVE implementation, so the same helper
    is SIMD on both AVX2 and A64FX.

Validation runs on 11 nodes, Qwen3.6-35B-A3B, q8 SSM/attention/shared-expert,
`EP_PREFILL_ATTN_GEMM_MAX_TOK=0`, tiled attention, and SSM block finish:

| Actual tokens | Prefill time | tok/s | Mixer | FFN local | Allreduce | Active SSM/attn GEMM |
|---:|---:|---:|---:|---:|---:|---:|
| 118 | 1.233 s | 95.70 | 0.703 s | 0.346 s | 0.155 s | 30 / 10 |
| 518 | 5.127 s | 101.03 | 2.678 s | 1.596 s | 0.793 s | 30 / 10 |
| 1018 | 9.585 s | 106.21 | 5.395 s | 2.701 s | 1.394 s | 30 / 10 |

The 1000-token segfault/timeout issue is now addressed by the stack-scratch guard/fallback.
With q8 attention GEMM enabled, the latest 11-node, 1018-token breakdown
after FFN norm reuse is:

| Component | Time |
|---|---:|
| SSM q8 projections | 1.944 s |
| Local MoE core | 1.487 s |
| EP allreduce | 1.394 s |
| Shared expert batched | 0.742 s |
| Attention tiled token work | 0.859 s |
| SSM finish/scan/gate total | 0.837 s |
| SSM finish scan/gate | 0.769 s |
| SSM output projection | 0.625 s |
| Attention qkv/output projections | 0.694 s |
| FFN RMSNorm | 0.441 s |
| SSM finish conv + qknorm | 0.068 s |

Immediate follow-up:

- continue node scaling sweeps at fixed 1000 tokens (`EP_NODE_SCALE_NODES` = 11/10/9/8)
  using the same staged `/local` model path and collect per-op times in tok/s.
- compare `EP_PREFILL_ATTN_TILE_ALGO=0` vs `1` once 9/10/11-node runs are stable,
  keeping `EP_PREFILL_ATTN_TILE_STACK_BYTES` explicit in each run.

Against the 220 tok/s goal, 1018 tokens need about 4.63 s.  The current
9.59 s run still needs a 2.1x reduction.  The largest remaining costs are q8
projection GEMMs, local/shared FFN work, and 40 blocking allreduces.  The
tiled-attention and SSM-finish changes made long-prompt scaling nearly linear:
518 to 1018 tokens is 1.97x tokens for about 1.87-1.98x prefill time depending
on whether the latest FFN norm reuse is included.

Node-count scaling over the current optimized 1000-token prefill:

| Nodes | Actual tokens | Prefill time | tok/s | Mixer | FFN local | Allreduce | Status |
|---:|---:|---:|---:|---:|---:|---:|---|
| 11 | 1018 | 9.585 s | 106.21 | 5.395 s | 2.701 s | 1.394 s | OK |
| 10 | 1018 | 9.713 s | 104.81 | 5.380 s | 2.937 s | 1.301 s | OK |
| 9 | NA | NA | NA | NA | NA | NA | rank 3 SIGKILL |
| 8 | NA | NA | NA | NA | NA | NA | rank 4 SIGKILL |

Run command:

```bash
EP_NODE_SCALE_NODES="11 10 9 8" \
EP_NODE_SCALE_TARGETS="1000" \
EP_NODE_SCALE_LAUNCHER=mpiexec \
EP_PREFILL_SSM_GEMM=1 \
EP_PREFILL_ATTN_GEMM=1 \
EP_ATTN_Q8_PV=1 \
EP_ATTN_FREE_Q4K=1 \
EP_SSM_Q8_PV=1 \
EP_PREFILL_ATTN_TILE=1 \
EP_PREFILL_ATTN_TILE_K=256 \
EP_PREFILL_SSM_BLOCK=1 \
bash a64fx/llm/ep_node_scale.sh
``` 

`pjrsh` was unavailable in this allocation (`exepjrsh=0`), so the scaling sweep
used `mpiexec` for rank launch.  The runner is still the uTofu implementation;
`mpiexec` only places processes, and `ep_node_scale.sh` regenerates the exact
topology with `tofu_topo_helper` before each node count.

Interpretation: full owned-expert BF16-PV is memory-limited below 10 nodes.
Reducing from 11 to 10 nodes saves about 0.09 s of allreduce but adds about
0.24 s of local FFN core work, so throughput drops slightly.  Scaling more nodes
will not reach 220 tok/s by itself because mixer projections, shared expert, and
attention are replicated work; the next real wins must reduce per-node compute
or move to true token-dispatch expert compute.

Negative checks:

- `EP_PREFILL_MOE_GEMM=1` still regresses: 118-token prefill slowed to 2.092 s
  with `moe_gemm=0.912 s`, so it remains off.
- BF16 PV for SSM/attention is slower than q8 PV on the same 118-token smoke:
  prefill 1.661 s, with SSM projection/output and attention projection/output
  all higher than q8.

Current implementation status:

- shared-expert gate dot + residual add and SVE `silu(gate)*up` helper are now
  enabled in the default prefill paths.
- `tf_vadd`, `tf_sum_squares`, `tf_rmsnorm`, and `tf_softmax` now have native
  A64FX SVE implementations in `common/transformer.h`.
- pooled silu/vadd workers now route through vectorized helper functions
  (`tf_silu_mul_avx2` and `tf_vadd`) so the flat batched blocks are no longer
  serialized scalar-element loops.
- local MoE prefill expert kernels now accept per-expert `q8_pv` slabs when BF16-PV
  is not available, so `EP_MOE_Q8_PV` can now participate in `transformer_prefill_ep_layermajor`
  local expert work as well (not just decode path), with the existing q8 prefill GEMM
  compute path selected automatically.
- FFN residual in layer-major prefill is now applied as a blockwise pooled add
  (`tf_vadd_flat_pool`) after local/batched expert accumulation, removing a
  remaining per-token residual hot loop in `transformer_prefill_ep_layermajor`.
- next optimization still open: MoE expert locality/permuted tile layout for the
  top-k local expert core and fused/more batched SSM recurrence forms.
- attention micro-optimization in progress: `tf_attn_prefill_tile_worker` now
  fuses the per-head output scale and gating step into one vector pass to trim one
  `out_h` read/write pass per row/tile. Default query tile width is now `32` when
  `EP_PREFILL_ATTN_TILE_Q` is unset (previous default was `16`).

Next validation pass (low-risk):

- Re-run `a64fx/llm/ep_node_scale.sh` for tok/s snapshots after the helper-vector
  changes:

```bash
EP_NODE_SCALE_NODES="11 10 9 8" \
EP_NODE_SCALE_TARGETS="100 500 1000 118 518 1018" \
EP_NODE_SCALE_LAUNCHER=mpiexec \
EP_PREFILL_SSM_GEMM=1 \
EP_PREFILL_ATTN_GEMM=1 \
EP_ATTN_Q8_PV=1 \
EP_ATTN_FREE_Q4K=1 \
EP_SSM_Q8_PV=1 \
EP_PREFILL_ATTN_TILE=1 \
EP_PREFILL_ATTN_TILE_K=256 \
EP_PREFILL_SSM_BLOCK=1 \
bash a64fx/llm/ep_node_scale.sh
```

- Key report fields to collect for this round: `tok/s`, mixer buckets
  (`attn_token`, `ffn_local`, `ffn_shared`, `ssm_finish_*`), and all-reduce.
- Primary acceptance target: 1000-token total throughput moves from 106.21 tok/s
  toward near-linear multi-node efficiency (goal ≥120+ tok/s before larger kernel
  rewrites; target remains 220+ tok/s).
- Focused EP-Q8 expert check before broader architecture changes:
  - run the same `ep_node_scale.sh` sweep once with `EP_MOE_Q8_PV=1` and
    `EP_MOE_BF16_PV=0` for both stability and tok/s delta:
    ```bash
    EP_NODE_SCALE_NODES="11 10" \
    EP_NODE_SCALE_TARGETS="100 500 1000" \
    EP_NODE_SCALE_LAUNCHER=mpiexec \
    EP_MOE_BF16_PV=0 EP_MOE_Q8_PV=1 \
    EP_PREFILL_SHEXP_GEMM=1 EP_PREFILL_MOE_GEMM=0 \
    EP_PREFILL_SSM_GEMM=1 EP_PREFILL_ATTN_GEMM=1 \
    EP_ATTN_Q8_PV=1 EP_ATTN_FREE_Q4K=1 \
    EP_SSM_Q8_PV=1 EP_PREFILL_ATTN_TILE=1 \
    EP_PREFILL_ATTN_TILE_K=256 EP_PREFILL_SSM_BLOCK=1 \
    bash a64fx/llm/ep_node_scale.sh
    ```

## Synthetic communication roofline

Implemented benchmark:

- `a64fx/utofu-tests/ep_prefill_synth.c`
- `a64fx/llm/ep_pref_comm_roof.sh`
- build target: `make -C a64fx/utofu-tests ep_prefill_synth CC=fcc`

The benchmark uses uTofu directly; `mpiexec` is only used to place the 11
processes and collect coordinates.  No model is loaded and no GEMM, attention,
SSM, or expert compute is performed.

Measured modes:

- `ar`: current layer-major EP communication shape, 40 layers of
  allreduce over `B * n_embd` fp32 values.
- `dispatch`: synthetic future token-dispatch shape, 40 layers of
  dispatch+combine all-to-all, BF16-sized activation payloads.

Qwen defaults used for the run:

- nodes: 11
- layers: 40
- hidden: 2048
- experts: 256
- top-k: 8
- activation bytes: 2 for dispatch payload
- token targets: 100, 250, 500, 1000
- routes: uniform, Zipf, hot

Full sweep report:

- `a64fx/llm/ep_pref_comm_roof_full.txt`

1000-token repeated report:

- `a64fx/llm/ep_pref_comm_roof.txt`

Compact full-sweep results in tok/s:

| Tokens | Route | Allreduce-only | Dispatch+combine |
|---:|---|---:|---:|
| 100 | uniform | 1692 | 3021 |
| 100 | Zipf | 1692 | 941 |
| 100 | hot | 1691 | 1277 |
| 250 | uniform | 1710 | 3079 |
| 250 | Zipf | 1705 | 943 |
| 250 | hot | 1709 | 1275 |
| 500 | uniform | 1696 | 3108 |
| 500 | Zipf | 1698 | 951 |
| 500 | hot | 1699 | 1288 |
| 1000 | uniform | 1709 | 3107 |
| 1000 | Zipf | 1709 | 952 |
| 1000 | hot | 1708 | 1287 |

1000-token repeated run with 3 timed iterations:

| Route | Allreduce-only | Dispatch+combine | Dispatch time |
|---|---:|---:|---:|
| uniform | 1709.62 tok/s | 3096.18 tok/s | 0.323 s |
| Zipf | 1709.35 tok/s | 956.43 tok/s | 1.046 s |
| hot | 1709.38 tok/s | 1292.42 tok/s | 0.774 s |

Conclusion: communication bandwidth is sufficient for 220+ tok/s.  Even the
skewed Zipf token-dispatch case has more than 4x headroom against the target,
and the current allreduce-only shape has about 7.8x headroom.  The next work
should target mixer/attention/SSM and FFN compute paths, not uTofu bandwidth.

One filesystem issue was observed: immediately after `tofu_topo_helper` exits,
`tofu_topo.txt` may not be visible to the shell loop on the shared filesystem.
The runner now reuses a valid topology and waits longer after regeneration.

## Optimization references

Useful local experiments live in:

- `~/work/clair/clair/a64fx/llm-guided-opt`

Relevant findings to reuse:

- `fused-gemm.md`: direct fused gather/scatter attention was poor
  (~44 GFLOPS); bounded two-phase score-tile reuse reaches the target region.
  Packed-output two-phase runs sustain about 120-121 GFLOPS, roughly 94-95% of
  FP32 peak, including HBM-scale cases.
- `fp16-dot-fp32-accum-gemm.md`: P@V kernels with 2x12 tiles are the preferred
  attention shape.  BF16->FP32 reaches about 99% FP32 peak.  FP16 K-blocking
  with FP32 shadow has a throughput/accuracy tradeoff; use FZ16 off and block
  64 for accuracy, or FZ16 on and block 512 for throughput.
- `ffn_fused_3pass.c`: FFN should be structured as a tiled 3-pass pipeline:
  gate/up GEMMs, in-L1 activation, then down GEMM.  The C tile layout is
  directly reusable as the next phase's A-panel, avoiding an intermediate
  row-major write/read.
- `benchmark.md`: qlair is directionally useful for SGEMM and activation
  kernels, but native A64FX timing is still the acceptance source.

## Next optimization plan

Priority order after the tiled-attention and SSM block-finish work:

1. Rebuild the local MoE expert core.  The existing `EP_PREFILL_MOE_GEMM=1`
   path is slower, but FFN local is still ~2.70 s at 1018 tokens, with local
   expert core ~1.49 s, shared expert ~0.74 s, and FFN RMSNorm ~0.44 s.
2. Improve q8 projection GEMM throughput or fuse projection consumers.  SSM q8
   projections+out are ~2.57 s and attention qkv/out projections are ~0.69 s.
   BF16 PV was tested and is slower for these tensors.
3. Continue attention kernel work only after FFN/projection wins.  The tiled
   attention token path is down to ~0.86 s at 1018 tokens; a packed two-phase
   score-tile kernel can still help, but it is no longer the largest item.
4. Reduce allreduce count or overlap.  The measured synthetic allreduce roofline
   is fast, but the real loop still pays 40 blocking layer reduces.  This should
   be considered now that local mixer compute is lower.

### 1. Establish a stable real-profile baseline

- Current stable prefill configuration:
  `EP_PREFILL_SSM_GEMM=1`, `EP_PREFILL_ATTN_GEMM=1`,
  `EP_PREFILL_ATTN_GEMM_MAX_TOK=0`, `EP_PREFILL_SHEXP_GEMM=1`,
  `EP_PREFILL_MOE_GEMM=0`, q8 PV for SSM/attention/shared expert, and BF16 PV
  for owned MoE expert tensors.
- The EP prefill profile now explains the run with active/fallback layer counts
  plus SSM, attention, FFN, shared expert, and allreduce buckets.
- Acceptance status: 500 and 1000 tokens complete without rank timeout or
  segfault on 10-11 nodes; the latest 11-node 1018-token run is 106.21 tok/s,
  not yet near the 220 tok/s goal.  The same full-BF16-owned-expert
  configuration is killed by memory pressure at 9 and 8 nodes.

### 2. Fix full-attention prefill first

Current state: q8 attention projection GEMM is stable at 500-1000 tokens, and
the per-token attention loop is replaced for `pos0==0` full-prompt prefill.
`EP_PREFILL_ATTN_TILE_ALGO=0` runs bounded two-pass score materialization
today; `=1` enables the online fused softmax path that avoids storing each
score tile. `=2` adds the explicit three-pass materialized path. `=3` is a
packed three-pass path using a fixed `[q_rows, k_tile]` score tile so we can
retain materialization wins with lower scratch pressure. At 1018 tokens,
tiled attention token work is ~0.859 s and
q/k/v/out projection work is ~0.694 s.

Implementation direction:

- Keep both score-materialized (algo=0/2/3) and online (algo=1) modes available.
  Baseline remains on 0 for stability checks; `algo=3` is the current packed
  materialization candidate for long contexts and should be used for controlled
  100/250/500/1000-token scaling once stable.
- The next implementation pass is a fused score write-back + out accumulation path
  to eliminate the explicit per-head, per-row clear/normalization loops while
  preserving fused online stability behavior.
- For each full-attention layer, precompute Q/K/V for the whole block, write KV
  cache once, then run causal attention in score blocks.
- Use score tiles sized first at 256 and 512 tokens.  Keep score materialization
  bounded; do not return to direct gather/scatter fusion.
- Keep output in the layout consumed by the output projection where possible;
  only unpack to row-major at the layer boundary if correctness requires it.
- Keep `EP_PREFILL_ATTN_TILE=0` as the fallback until the packed two-phase path
  passes 1000-token stability tests.
- `tf_attn_prefill_tile_worker` now fuses final row scale and sigmoid gate in one
  vector loop (removing one output read/write pass). Also, `EP_PREFILL_ATTN_TILE_Q`
  default moved from 16 to 32 to reduce tile count for long prompts.

Acceptance:

- `EP_PREFILL_ATTN_GEMM=1` must continue to complete 100, 250, 500, and
  1000-token prompts without timeout.
- Full-attention layers should scale near linearly with token count after the
  expected O(M^2) attention math is accounted for.
- Attention token work should drop below 0.5 s at 1000 tokens before spending
  more effort here; FFN and projection work are currently larger.

### 3. Convert SSM mixer to a batched layout-preserving path

Current state: SSM q8 prefill is stable and all 30 SSM layers use the batched
path.  At 1018 tokens, SSM projections+out are ~2.57 s and finish/scan/gate is
~0.84 s (`conv+qknorm` ~0.07 s, scan/gate ~0.77 s).

Implementation direction:

- Keep the existing layer-major batched projections:
  `ssm_qkv`, `ssm_gate`, `ssm_alpha`, `ssm_beta`, `ssm_out`.
- Remove avoidable layout churn between projections, scan state update, gate,
  and `ssm_out`.
- Replace per-token `tf_ssm_prefill_finish_token` with pooled token-range scans and
  vectorized gate/RMSNorm across contiguous slices; this is the active SSM finish
  vectorization target and is now the default (`EP_PREFILL_SSM_BLOCK=1`).
- Keep the block finish as the default.  Further work should target the
  scan/gate worker itself or fuse the `ssm_out` consumer.
- First-touch all SSM scratch by token slice, matching the current prefill GEMM
  NUMA/CMG partitioning.
- Use `EP_PREFILL_SSM_NULL_FINISH=1` for measurement only: keep all SSM GEMMs,
  skip the recurrence finish.  This isolates scan/gating cost from projection
  cost.

Acceptance:

- SSM path remains numerically compatible with the current stable path for short
  prompts.
- 1000-token `ssm_gemm_layers` remains 30 and does not increase allreduce calls.
- SSM mixer time should move toward the cost of the five batched projections
  plus a small scan/gate tail; if scan/gate remains large, split it into its own
  optimized SVE kernel.

### 4. Rebuild MoE FFN prefill around fused 3-pass expert batches

Current state: `EP_PREFILL_MOE_GEMM=1` exists but is slower and remains off.
The stable path routes per token and runs local expert matvecs.  Shared expert
is batched separately and now reuses the local FFN RMSNorm buffer.  Full
owned-expert BF16-PV is required for current speed but prevents 8-9 node runs
from completing.

Implementation direction:

- Keep router/top-k separate and deterministic first.  Route all tokens in the
  block, group `(token, expert)` rows by local expert, and build compact expert
  batches.
- Add an owned-expert q8-PV path, or another lower-memory expert fast path,
  before expecting reliable scaling below 10 nodes.  The current BF16-PV owned
  slabs are the observed 8-9 node SIGKILL trigger.
- For each local expert, use the Clair FFN pattern:
  gate/up GEMM, in-tile SiLU/SwiGLU activation, down GEMM, accumulate back to
  the token output.
- Reuse the tile layout between phase 1 activation output and phase 3 input;
  avoid writing a full row-major `[tokens, n_ff]` intermediate.
- Start with local-only EP, preserving the current replicated-token/allreduce
  semantics.  Only after this is stable should true token dispatch replace
  replicated EP.
- Keep shared expert separate initially; fuse it only if profiles show it is a
  material part of FFN time.

Acceptance:

- `EP_PREFILL_MOE_GEMM=1` must pass 100, 250, 500, and 1000-token prompts
  without segfault.
- Output difference versus stable path must be bounded and logged for a short
  prompt before performance runs.
- 1000-token FFN local time should drop by at least 2x before trying true
  token-dispatch EP.

### 5. Only then evaluate true token-dispatch EP

The synthetic dispatch roofline says communication is viable, including skewed
routes.  It should still be deferred until local batched expert compute is
stable, because changing distribution semantics and FFN kernels at the same time
will make failures hard to diagnose.

Implementation direction:

- Use the synthetic `dispatch` benchmark's route-count logic to size send and
  receive buffers.
- Dispatch only normalized token rows needed by remote experts.
- Compute expert batches on owner ranks using the same fused 3-pass kernel.
- Combine partial outputs back to source-token owners, then perform the same
  residual/allreduce semantics as the current path.

Acceptance:

- Uniform, Zipf, and hot synthetic distributions remain above 220 tok/s after
  the real dispatch packing format is introduced.
- Real 1000-token prefill must improve over replicated local FFN before this
  path becomes the default.

### 6. Final target gate

Run:

```bash
EP_PREFILL_LAUNCHER=mpiexec \
EP_SCALE_TARGETS="100 250 500 1000" \
bash a64fx/llm/ep_pref_scale.sh
```

Target pass criteria:

- 1000-token prefill >= 220 tok/s.
- 100, 250, 500, 1000-token rows scale predictably.
- No rank timeout, no segfault, and no forced attention-GEMM guard for 1000
  tokens.
- Profile shows communication is not the dominant component.
