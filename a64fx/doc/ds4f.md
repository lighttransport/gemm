# DeepSeek-V4-Flash (DS4F) on A64FX — Research Note

This note summarizes how the DS4F harness (`common/ds4f.h` / `common/ds4f_impl.h` /
`a64fx/llm/ds4f_ep_runner.c`) executes DeepSeek-V4-Flash on A64FX: the execution
model, the weight dequantization scheme, the prefill and decode compute paths,
and a worked example of using the served model itself as a coding agent to
write an SVE assembly kernel. For the full session-by-session change log and
every measured lever, see `a64fx/ds4f.md`; this note is the condensed
architecture overview.

## 1. Model shape

43-layer DeepSeek-V4-Flash, MLA attention + sparse MoE + mHC residual streams:

| | |
|---|---|
| layers | 43 (`n_layers`) |
| hidden | 4096 |
| vocab | 129280 |
| attention | MLA, 64 heads, `q_head_dim=512` (448 nope + 64 rope), `q_lora=1024`, `kv_lora=512` (1 compressed KV head) |
| o-proj | block-diagonal, `o_inter=8192` split into `o_groups=8` groups of `o_lora=1024` |
| MoE | 256 experts, top-6 active, `moe_inter=2048`, plus 1 replicated shared expert (`shared_inter=2048`) |
| mHC | manifold-constrained hyper-connections, `hc_mult=4` residual streams, `hc_iters=20` Sinkhorn iterations |
| sparse attention | Tier-B2 lightning indexer: per-layer `compress_ratios` alternate CSA (ratio 4, indexer-selected top `index_topk=512`) / HCA (ratio 128, dense-over-compressed); layers 0,1 are full dense attention |

## 2. Execution model on A64FX

**Parallelism = expert-parallel (EP) over uTofu**, not tensor-parallel by
default. `N` nodes (typically 11), 1 rank/node, no MPI library in the binary —
`a64fx/utofu-tests`-style raw uTofu VCQ/Put primitives via `tp_allreduce.h`
(`common/tp_allreduce.h`), launched with plain `mpiexec -np N -vcoordfile ...`
(the vendor process launcher only; no MPI calls). Each rank:

- Owns `n_experts / N` MoE experts (`e % N == rank`, dense-packed locally);
  routed-expert output is summed across the EP group with **one all-reduce
  per layer** (`m->ar_cb`, wired to `tp_allreduce_sum`).
- Replicates the dense weights (attention, shared expert, embed, head, mHC
  parameters) — unless a `DS4F_TP_*` tensor-parallel shard is enabled (§6).
- Runs a single-process, pinned 48-thread pool (`ds4f_pool_start`,
  `LLM_THREADS` env, default 48) with spin-wait dispatch (`ds4f_pool_run`);
  every matvec/GEMM/attention/mHC op is one `pool_run` call, i.e. one
  barrier-synchronized fan-out over the 48 cores (12/CMG × 4 CMGs).
- Applies **NUMA-interleaved weight placement** (`ds4f_apply_numa`,
  `set_mempolicy(MPOL_INTERLEAVE)` over all NUMA nodes) — the single
  largest bit-identical decode lever found this project (~1.4×): A64FX's
  default first-touch policy otherwise pins weights to one CMG, capping
  cross-CMG bandwidth at ~100 GB/s instead of the node's ~700+ GB/s.

Real weights are staged once per node from the shared FS into node-local
`/local/ds4f/rank<NN>.blob` (`run_ds4f_stage_11n.sh` / `ds4f_stage`), sharded
by `PMIX_RANK` so each blob matches the EP rank the topology assigns that
physical node. Loading reads only the owned shard (`ds4f_load_real`), so
per-node resident memory is dense (~9 GB, replicated) + `~experts/N` (not the
full 256-expert model).

## 3. Dequantization: algorithm and config

Weights stay quantized in HBM and are dequantized **on the fly per matvec/GEMM
call** — nothing is bulk-dequantized to disk or a persistent shadow copy
except where an explicit "predequant" mode opts in. Three weight
representations, selected per-tensor by `ds4f_qtype`:

| type | used for | scale granularity | dequant kernel |
|---|---|---|---|
| **FP8 E4M3** (`DS4F_FP8`, default) | dense (attn/shared/embed/head) | 128×128 block, E8M0 (power-of-two) scale | `matvec_fp8e4m3_8row` — 8-row-blocked, LUT-based (`fp8_lut`) or "magic" bit-trick decode |
| **MXFP4 / split e2m1** (`DS4F_MXFP4`) | routed experts | per-32 E8M0 scale | `matvec_mxfp4_8row` — 8-row-blocked nibble unpack |
| **INT8 W8A8** (`DS4F_Q8_PV`, opt-in via `DS4F_Q8_DENSE=1`) | the 8 dominant dense tensors (`wq_a/wq_b/wkv/wo_a/wo_b/sh_w1/sh_w3/sh_w2`); gate + head stay bf16 | per-64-element dynamic activation scale (absmax) + per-row FP16 weight scale | `matvec_sdot_8row` — `svdot_s32` int8 dot product, activation quantized on the fly by `ds4f_quant_x_sdot_into` |

Two config knobs change the *dense* representation:

- **`DS4F_FP8_BF16=1`** — predequant dense FP8→BF16 once at load (+6 GB
  resident, `DS4F_BF16_PV` "pair-interleaved" layout by default). Faster
  matvec at fixed memory cost; the production **decode preset uses this**.
- **`DS4F_Q8_DENSE=1`** (`ds4f_q8_promote_dense`, requires `FP8_BF16=1`
  first) — re-repacks the bf16-pv dense tensors to int8 W8A8 in place,
  reclaiming the bf16 source (net memory *shrinks* vs FP8-only). This is the
  fastest decode dense path and is in the production preset.

The recurring implementation trick across all three dequant kernels is
**register-level bit manipulation instead of an explicit shift/convert**: bf16
values are zero-extended to fp32 by a *predicated odd-halfword load*
(`svld1_u16` with a predicate active only on odd 16-bit lanes, so the loaded
bf16 bits land directly in the upper 16 bits of each 32-bit lane — a valid
truncated fp32 bit pattern with no `lsl`/shift instruction at all — the
"pair-interleaved", or **PV**, weight layout exists specifically to make this
trick applicable). FP8/E4M3 uses either an L1-resident 256-entry gather LUT
(exact, lossless vs the reference bf16 widen) or a "magic" bit-trick decode
(faster but flushes E4M3 subnormals, opt-in via `DS4F_FP8_MAGIC`). All three
kernels share the same 8-row register-blocking convention (accumulate 8
output rows in parallel per call, one weight-group load per K-tile) so the
matvec call sites are largely representation-agnostic.

## 4. Prefill (M > 1)

Prefill batches all M prompt positions through one **GEMM** instead of M
sequential matvecs, turning the M=1 bandwidth-bound decode kernel into a
compute-bound GEMM:

- Dense/MoE-router/head projections: `ds4f_gemm` — for `DS4F_FP8_BF16=1` this
  is a direct bf16×bf16→f32 GEMM (`matvec_bf16_8row_pv` tiled over M); for the
  default FP8-resident config it's a **fused FP8→bf16 tile-dequant GEMM**
  (each 8-row group's K-tile is dequanted once into an 8 KB L1 pair-buffer
  and reused across all M tokens, ~88% of the +6 GB resident-bf16 speed at
  zero extra memory).
- Routed experts: bucketed by owned-expert-id across the M-token tile, each
  bucket's rows/columns get one `ds4f_gemm` call (an MXFP4 tile-dequant GEMM
  when `DS4F_MXFP4_GEMM_TILE` is set — dequant is otherwise the dominant cost
  at large M since `svtbl`-based nibble unpack is flat regardless of M).
- Comm: **one `[M,hidden]` all-reduce per layer** instead of M separate
  `[hidden]` reduces — the primary comm-amortization lever for prefill.
- Attention/Tier-B2: NOT batched across positions in the base prefill path
  (per-position, causal); `ds4f_forward_verify` (§6/coding-agent's sibling
  feature, batched decode) extends batching to the per-position attention
  loop too, causally looping K positions through the compressor/indexer/
  attention in order.
- mHC hc_pre/hc_post also batch across M (`ds4f_hc_pre_batch`/`_post_batch`).

Net effect (measured, `a64fx/ds4f.md`): prefill throughput roughly
**+70–90%** over per-token prefill at typical prompt lengths, dominated by
the dense-GEMM and comm-amortization wins; the sparse compressor/indexer
portion (tb2) does *not* amortize (still per-position), so the win shrinks at
very long prompts.

## 5. Decode (M = 1)

Single-token decode (`ds4f_forward_token`) is memory-bandwidth-bound: every
matvec reads its full weight tensor from HBM for one token's worth of
activation. Per layer, per token:

1. **mHC pre** (attn): collapse 4 residual streams → 1 via a small
   Sinkhorn-normalized gating matvec (`ds4f_hc_pre`), parallelized across
   the pool (`DS4F_HC_PAR`) since it sits *between* dispatches and would
   otherwise run serially on one core.
2. **QKV projection**: `wq_a`→q-norm→`wq_b`→per-head q-norm+RoPE (`DS4F_QNR_PAR`
   parallelizes the 64 independent per-head norm+RoPE ops); `wkv`→kv-norm→
   RoPE; append to the KV ring cache.
3. **Tier-B2 prepare** (`ds4f_tb2_prepare`, sparse layers only): step the
   layer compressor (`ds4f_compress_step`, a small matvec + recurrent
   ring-state update) and, on CSA layers, the lightning indexer
   (`ds4f_index_step` — projects q into indexer space, RoPEs it
   (`DS4F_TB2ROPE_PAR`), scans compressed candidates via `svdot`-based int8/
   int4 kernels, and heap-selects the top-`index_topk` — `O(T log k)`, the
   long-context-critical component).
4. **Attention**: `ds4f_attn_tb2_worker` (per-head, generic) or
   `DS4F_ATTN_GEMM`'s 8-head KV-reuse variant (each KV latent loaded once,
   dotted against 8 heads instead of 1 — the ~50% attn-phase win, falls
   back automatically when the owned head count isn't 8-aligned, e.g. under
   `DS4F_TP_ATTN`).
5. **O-projection**: block-diagonal `wo_a` (`DS4F_OPROJ_FUSE`, one fused
   dispatch instead of 8 per-group ones) → `wo_b` → **mHC post** (expand
   1→4 streams).
6. **MoE**: shared expert (replicated) + owned routed experts (top-6,
   `e % N == rank`) → **one `[hidden]` all-reduce** (`ar_cb`) to sum the
   routed-expert partial across the EP group → mHC post.
7. **Head**: mHC collapse → out-norm → `lm_head` matvec → argmax.

Comm is the dominant *irreducible* decode cost at M=1: **43 per-layer
all-reduces**, latency-bound rather than payload-bound (a diagnosed,
closed investigation — see `a64fx/ds4f.md` WS4: the ~300 µs/reduce floor is
inter-rank compute-skew/straggler-sync, not the reduce-tree itself, which is
already `O(log N)` and floors at ~14 µs). This is why the two real decode
levers are (a) reduce per-token *compute* (the kernels above) and (b) amortize
the *fixed* per-token comm cost across more tokens — i.e. **batching**
(`ds4f_forward_decode_batch`, M independent sequences through one
`ds4f_forward_verify` call with per-sequence KV/compressor/indexer cache
*sets*, pointer-swapped into the layer struct per position). Measured
ceiling ≈ 2.5–3× at M=16 (compute-bound: the dense GEMMs and MoE experts
genuinely don't amortize below M≈32, not a dispatch/comm artifact — see
`a64fx/ds4f.md`'s batched-decode section for the full negative-result trail).

**Vocab-sharding** (`DS4F_TP_HEAD` / `DS4F_TP_EMBED`) removes the last two
*full-vocab-width* replicated ops: the lm_head and the input embedding table
are each sharded `vocab/N`-wide across the EP group, with the shard-local
result merged by a **cheap** all-reduce (an 8-byte `(val, global-idx)` argmax
merge for the head — mirroring the already-correct batched-verify path — and
a `[hidden]`=16 KB row-reduce for the embed lookup). The head's *first* cut
of this feature mistakenly did a **full `[vocab]` (517 KB) all-reduce-SUM**
instead of the cheap argmax merge, which exactly cancelled the head-compute
saved (measured net-neutral); fixing it to match the verify path's argmax
pattern is a genuine +2.3% decode win. `TP_EMBED`'s row-reduce was correct
from the start and is a pure −0.97 GB memory lever at unchanged speed. Both
are bit-exact and in the production `--preset decode` bundle.

## 6. A worked example of a GEMM kernel — and using DS4F as a coding agent

As a case study in using the *served* DS4F model itself as a coding agent
(feeding it a prompt through the same real-weight, 11-node decode pipeline
described above, and treating its output as untrusted code to test), it was
asked to write a small SVE1.0 (Armv8.2-A+SVE, no SVE2) hand-written assembly
matmul microkernel:

```c
void matmul_kernel_8x1_f32_sve(const float *A, const float *B, float *C,
                                 long ldc, long K);
// C[m][0:VL) += sum_k A[m][k] * B[k][0:VL)   for m = 0..7
```

an 8-row-blocked, 1-SVE-vector-wide FP32 rank-1-update microkernel — the same
register-blocking family as the *existing* hand-written kernels in this repo
(`a64fx/fp16-gemm/fp16_kernel_6x4.S`, `a64fx/vlm/kernels/micro_kernel_bf16B_
8x3_pv.S`) and the same 8-row convention used throughout `ggml_dequant.h`'s
dequant kernels (§3).

**Method**: the prompt was a heavily scaffolded code-completion stub (AAPCS64
calling convention documented, function prologue/predicate-setup/accumulator-
zeroing already written, cut off right at the K-loop body) fed through
`run_ds4f_gen_11n.sh`'s real-weight greedy-decode pipeline — the base
checkpoint has no chat template, so this is plain text continuation, not an
instruction-following interaction.

**Findings**:

- **First attempt** (no reasoning cue, `MAX_NEW=768`): cut off mid-function.
  What it wrote had 2 real SVE syntax bugs — an unpredicated `fmla` (SVE
  vector FMLA is always predicated) and `ld1rsw` used with a `.s` (32-bit)
  destination when that mnemonic sign-extends to a 64-bit element (needed
  the plain, non-sign-extending `ld1rw` instead). 13 assembler errors.
- **Reasoning-mode attempts**: seeding the model's own `<think>` special
  token (a real trained token id, not a text substring — verified via the
  tokenizer's `added_tokens` table) and giving a larger budget produced
  visibly better code — both syntax bugs were fixed, and the reasoning
  *narrated* a real logic bug in its own draft (confusing a stride
  register) — but the model never emits the matching `</think>`, so its
  prose has no comment delimiter and breaks a naive compile of the raw
  output. With `MAX_NEW=3072` it eventually reached `EOS` with a complete,
  syntactically finished function; commenting out one leading unmarked
  prose line (no instruction touched) made the file **compile cleanly** to
  a valid object exporting `matmul_kernel_8x1_f32_sve`.
- **Numerical correctness** (the step "compiles" doesn't cover): a
  standalone test harness comparing the kernel against a reference GEMM
  found it does **not** implement the row-major `A[8,K]` layout the prompt
  specified — it implicitly reads a **K-major packed** layout instead
  (`A_packed[k*8+m] = A[m][k]`, the layout convention this repo's *own*
  existing kernels use, e.g. `fp16_kernel_6x4.S`'s "`Ap: K panels of 6 half
  elements`"). Feeding A in that layout, the kernel is **bit-exact** (~3e-5,
  pure FP32 rounding) for both `ldc == VL` and `ldc > VL`, closing out a
  separate stride concern the model had flagged (and not actually fixed) in
  its own comments. The doc comment was then corrected to describe the real
  (K-major) layout and the unfounded self-doubt trail removed.
- **Throughput was unaffected by reasoning**: 13.3–14.3 tok/s prefill/decode
  across all attempts (768 vs 1536 vs 3072-token budgets) — "thinking" cost
  more *tokens*, not more *time per token*; it's the same decode kernel
  path either way.

Net takeaway: the served model is a plausible-but-untrustworthy low-level
code generator — it correctly reproduces the *shape* of a register-blocked
SVE microkernel (right structure, right register discipline, eventually
right instruction-level syntax) but silently substitutes its own (very
standard, and code-base-consistent) data-layout convention for the one it
was asked to implement, and narrates fixes it doesn't apply. Every claim in
this section (compiles / doesn't compile / is numerically correct under
layout X) was independently verified by compiling and running against a
reference — see `a64fx/llm/ds4f_matmul_kernel*` for the full prompt, raw
model output, fixed/compiling kernel, and the two correctness test harnesses.

## References

- `a64fx/ds4f.md` — full session log: every measured lever (landed and
  rejected), with numbers, commit hashes, and root-cause write-ups (WS1–WS7).
- `common/ds4f.h` / `common/ds4f_impl.h` — model config, forward pass,
  kernels.
- `common/ggml_dequant.h` — the FP8/MXFP4/INT8 dequant + matvec kernels.
- `common/tp_allreduce.h` — the uTofu recursive-doubling all-reduce +
  ack/retransmit reliability layer.
- `a64fx/llm/ds4f_ep_runner.c` — CLI, serving loop, sampling, batching,
  the DS4F_DB_BENCH/DECODE_BATCH test harnesses.
- `a64fx/llm/ds4f_matmul_kernel*` — the coding-agent case study artifacts.
