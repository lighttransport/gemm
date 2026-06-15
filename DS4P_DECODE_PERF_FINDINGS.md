# DS4P decode/prefill performance on 96 A64FX (Fugaku) — findings

**Target:** decode 70 tok/s, prefill 170 tok/s on 96 nodes (DeepSeek-V4-Pro: 1.6T params,
49B activated/token, 61 layers, hidden 7168, 384 experts/6 active, MLA + CSA/HCA sparse
attention, mHC 4-stream, MTP depth-1). Exact-only (lossless).

## TL;DR
- **Delivered (validated: exact, lockstep, 96 nodes, synthetic + real-weight token-identity):
  decode 4.20 → 5.59 tok/s (+33%), prefill 5.4 → 9.0 tok/s (+67%).**
- **70 tok/s decode is NOT reachable on this hardware/engine — single OR multi-stream.** The hard
  ceiling is **~6 tok/s**, proven independently from a roofline floor harness, speculative decode,
  multi-stream batching, and barrier-count arithmetic — all converging.
- The ceiling is set by **mandatory per-dependency cross-CMG thread-barrier rendezvous + per-layer
  all-reduce communication + M=1 matvec compute** — none removable by software restructuring on a
  4-CMG A64FX node + Tofu-D interconnect with this sharded-MoE engine.
- Reaching 70 needs **different hardware** (more cores/CMG or cheaper barriers + faster interconnect)
  or a **fundamentally different parallelization** (far fewer cross-CMG syncs / less sharding), not
  kernel or scheduling optimization.

## Decode cost decomposition (the foundation)
Built a `DS4F_DUMMY` floor harness: replaces matvec/gemm FMA with a memcpy-like weight stream and
skips tb2/attention arithmetic, keeping every `pool_run` dispatch + `ar_cb` all-reduce. Single-
allocation decomposition of decode's ~200 ms/token (job 49231730):
| component | ms | % | reducible by |
|---|---|---|---|
| compute (M=1 matvec) | 89 | 44% | only M≥16 GEMM (multi-stream) or quantized kernels |
| comm (all-reduces)   | 56 | 28% | overlap only (coupled to sharding; can't remove) |
| dispatch (~2000 barriers) | 53 | 26% | barrier COUNT only (~1.5×); latency is physics |
| memory (weight read) | 4  | 2%  | already fine — the "over-read" is a non-issue |

## Validated improvements (banked)
All flag-gated, default-safe, token-identical to baseline:
- **`DS4F_TP_SHW2`** — shared-expert w2 column-shard: +6% decode, −1.3 GB/rank.
- **`DS4F_AR_RS`** — Rabenseifner reduce-scatter+all-gather all-reduce (vs recursive-doubling):
  +4.6% decode, +5.5% prefill.
- **`DS4F_IDX_OP`** — outer-product index scan (eliminates the per-(head,pos) `svaddv`): tb2scan
  16→5.2 ms, +6.8% decode.
- **`forward_verify` TP_ATTN fix** — `forward_verify` was wrong at TP_ATTN=1 (head-shard q-proj +
  missing s_attn reduce); fixed so the batched forward is token-exact in full TP.
- **Full-payload Rabenseifner via batched-verify prefill (`DS4F_PREFILL_VERIFY`)** — the prefill
  jump: per-token reduce count=7168 has 448 B latency-bound RS tails; batching K positions makes
  the [K,C] reduce one BW-bound RS. Prefill 6.1 → 9.0 (K=32), comm 49%→16%, ar_calls 12× fewer.
- **Batched-verify spec γ=1 (`DS4F_SPEC`/`DS4F_VERIFY_BATCH`)** — spec decode correct at full TP
  (α=97.1%, accepts 50/55) and net +11% decode (5.04→5.59) once the indexer scan/qproj were
  batched across the verify positions.

## Why 70 is unreachable — the four converging proofs

### 1. Single-stream is comm+compute bound
Even if dispatch were zeroed, decode floors at compute 89 + comm 56 = **145 ms = ~6.9 tok/s.**
The 89 ms is M=1 matvec at ~0.18% of bf16 peak — it only becomes efficient as an M≥16 GEMM
(weight-unpack amortized), which single-stream decode never is.

### 2. Speculative decode can't amortize it (exact-only)
- α=97.1% is excellent, but **γ>1 needs a mandatory "redo"**: a partial accept over-advances the
  sequential compressor ring (whose per-layer input *is* the per-layer hidden), and rolling it back
  cheaply is impossible — so partial accept = a second full forward. Measured γ=4 did 84 forwards
  for 106 tokens. ⇒ **γ=1 is optimal** (best = 5.59).
- The verify forward is `fixed(~110 ms dispatch+comm) + per_token(~90 ms)×K`; spec only divides the
  fixed part. Even perfect γ=8 drafts → ~110+90×9 for 9 tokens ≈ **~10 tok/s**. Amortizing the
  per-token compute needs M≥16 ⇒ γ≥15 good drafts, impossible for a depth-1 MTP.

### 3. Multi-stream batching saturates at ~6
Built + validated a token-exact multi-stream engine (`ds4f_forward_batch_decode`, N seqs/forward).
N-sweep (job 49235878): aggregate **5.26 / 5.78 / 5.88 / 5.93** tok/s at N = 2/8/16/32 — flat. The
M=32 forward costs ~27× a single forward: the dense GEMM amortizes, but the **per-sequence attn/tb2
loop pays N× the cross-CMG dispatches**, and the all-reduce data is ∝N. Multi-stream hits the same
walls × N.

### 4. The dispatch wall is mandatory rendezvous, mechanism-invariant
Five dispatch levers all gave ~0: `DS4F_MV_FUSE`, `DS4F_FLAGBAR`, `DS4F_TP_OPROJ=0`,
**`DS4F_HWBAR`** (libhwb HW barrier — Phase 0 of the megakernel: dispatch floor 110→108 ms, −2%),
and the megakernel arithmetic itself. Conclusions:
- The ~24 µs/dispatch is **not** the barrier mechanism (HW barrier = software spin barrier) — it is
  the physics of synchronizing 48 threads across 4 CMGs (arrival skew + cross-CMG propagation).
- The **serial element-wise gap is negligible** (an rmsnorm of C=7168 ≈ 3 µs < one 24 µs barrier),
  so parallelizing it (a megakernel's cooperative workers) would *add* barriers to save microseconds.
- A megakernel can only cut the barrier *count* (~28→~18 mandatory per-layer dependency rendezvous,
  ~1.5×) → dispatch 48→~32 ms → decode → **~5.4 tok/s (+8%)** for a multi-week rewrite.
- Irreducible decode floor = ~18 barriers/layer × 61 × 24 µs (≈22 ms) + 56 comm + 89 compute
  ≈ **167 ms = ~6 tok/s.**

## Lever ledger
| lever | flag | result |
|---|---|---|
| shared-w2 col-shard | DS4F_TP_SHW2 | +6% decode ✓ |
| Rabenseifner reduce | DS4F_AR_RS | +4.6% dec / +5.5% pre ✓ |
| outer-product idx scan | DS4F_IDX_OP | tb2scan 16→5.2ms, +6.8% dec ✓ |
| full-payload RS / batched-verify prefill | DS4F_PREFILL_VERIFY | prefill 6.1→9.0 ✓ |
| forward_verify TP_ATTN fix | (structural) | spec correct at full TP ✓ |
| batched-verify spec γ=1 | DS4F_SPEC + DS4F_VERIFY_BATCH | decode 5.04→5.59 ✓ |
| multi-token spec γ>1 | DS4F_SPEC_GAMMA | NET-NEGATIVE (mandatory redo) ✗ |
| multi-stream batching | DS4F_MSTREAM | correct, but aggregate saturates ~6 ✗ |
| multi-acc matvec | DS4F_MV_OP | ~0 (already pipelined by fcc) ✗ |
| barrier-flag (cache-line) | DS4F_FLAGBAR | ~0 ✗ |
| fuse matvec pairs | DS4F_MV_FUSE | ~0 (too few) ✗ |
| drop o_inter shard | DS4F_TP_OPROJ=0 | comm down but compute up, net worse ✗ |
| HW barrier (libhwb) | DS4F_HWBAR | ~0 (−2%); cost is rendezvous, not mechanism ✗ |

## Reusable infrastructure built
- `DS4F_DUMMY` floor harness (=1 stream weights / =2 dispatch-only) — decomposes any config into
  compute/comm/dispatch/memory.
- Multi-stream engine: `ds4f_forward_batch_decode` / `ds4f_forward_verify_ex` (+`posv`,`+streams`),
  `ds4f_alloc_streams`, `ds4f_stream`/`ds4f_layer_state`, pointer-swap bind/unbind — token-exact.
- Batched-verify indexer (`ds4f_idxsc_op_batched_worker` + `index_step` `defer_scan`).
- HW-barrier pool path (`DS4F_HWBAR`, libhwb `vhbm_bar`) — correct, available if ever useful.
- Pjsub A/B job templates (floor, mvfuse/flagbar/tpoproj/hwbar, spec/vbatch, mstream N-sweep) +
  the file-RPC dev-shell scaffold (note: login↔compute not routable for HTTP; old Python on nodes).

## What 70 tok/s would actually require
1. **Far fewer cross-CMG syncs** — a single-CMG or persistent-warp execution model, or a chip with
   more cores/CMG / a sub-µs global barrier. The ~24 µs A64FX 48-thread/4-CMG rendezvous × the
   mandatory dependency chain is the binding constraint.
2. **Reduced comm** — fewer/larger nodes (less EP/TP sharding ⇒ fewer all-reduces) or a faster
   interconnect than Tofu-D; comm is coupled to sharding and only hideable, not removable.
3. **M≥16 GEMM-efficient decode** — i.e. real multi-stream *with* comm fully overlapped behind the
   M=N compute AND attn/tb2 batched across streams to O(1) dispatch (the megakernel Phase 3+2). Even
   then the comm-∝N wall likely caps aggregate ~15-30, not 70, on 96 nodes.
These are hardware/architecture changes, not engine optimizations.

## Reproduce
- Synthetic A/B (no staging, fast): `pjsub a64fx/llm/pjsub_ds4p_floor_96n.sh` (DUMMY decomposition),
  `..._hwbar_96n.sh`, `..._mvfuse_96n.sh`, etc.
- Real-weight (staged ~20 min): `..._specfix_96n.sh` (spec), `..._mstream_96n.sh` / `..._msweep_96n.sh`
  (multi-stream), `..._vbatch_96n.sh` (batched verify). Validation = gen-stream token-identical to the
  greedy reference `361 855 9080 18561 11 …` + `argmax_distinct_across_ranks==1` + NaN=0.
- Detailed running log + gotchas: memory `ds4p-perf-calibration.md`; rewrite scope:
  `~/.claude/plans/optimize-ds4p-prefill-and-silly-sonnet.md` (megakernel) and `multistream-serving.md`.
