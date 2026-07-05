# M3 decode — next two levers (scoping)

After the mstream campaign (best real config `M3_MSTREAM=48 TP_AR_BF16=1` = **15.25 tok/s @48n**), the
two remaining decode levers are both **code projects**, not probes. Scoped below. Recommendation:
**do Lever 1 first** (moderate effort, clear payoff, low risk); Lever 2 is research-grade (high risk,
uncertain ~1.2×). Baselines/model: `m3.md` decode roofline, `m3_decode_sim.py`.

---

## Lever 1 — wire int4-KV into batched decode (`m3_forward_batch_decode`) — ✅ IMPLEMENTED (223c5b9a)

**STATUS (2026-07-05):** DONE + functionally validated (job 49441928, 1n synth): int4-KV now **engages**
in batched decode (out0 differs from bf16 at M=8/32/64, vs bit-identical before), **NaN=0**, **tok/s
unchanged** (unpack is free). Per-stream KV ~3.9× smaller → `m3_decode_sim` shows M=48 peak reachable at
4k ctx (48n: bf16 M=11 → int4 M=46). **Remaining before production: real-weight QUALITY gate** (does
int4-KV keep coherent gen? — extend `pjsub_m3_kv_mstream_1n.sh` to batched real-weight bf16-vs-int4, then
a 48n/4k real A/B where bf16 OOMs at M=48 but int4 fits). Original scope below.


**Why.** int4-KV currently exists ONLY in the single-stream `m3_forward_token` path (codec
`m3_q4_pack`/`m3_q4_dot`/`m3_q4_axpy` + per-layer `L->k_q4/v_q4/k_qs/v_qs`, m3_impl.h ~64, 571, 665).
The batch path stores/reads the per-stream KV as **bf16** (`ms->kc`/`ms->vc`, uint16, m3_impl.h:744,
891–904) — so batched serving KV is always bf16. **Payoff (from `m3_decode_sim.py`):** int4 KV is
~4× smaller (0.56 vs 2 B/elem), so the M=48 throughput peak becomes reachable at LONG context without
adding nodes:

| config | bf16-KV (today) | int4-KV (this lever) |
|---|---|---|
| 48n bf16, 4k ctx | M=11 → 13.6 tok/s | M≈44 → **15.25 tok/s** (reach the peak) |
| 96n bf16, 4k ctx | M=23 → 15.3 | M=48 → 15.25 (fewer nodes needed) |
| 24n fp8, 4k ctx | M=11 → 13.6 | M≈44 → 15.25 |

i.e. **+12% at 4k ctx**, and it collapses the "long-ctx needs more nodes" penalty the sim shows. Also
the enabler toward the 1M-ctx serving story (with CP).

**Approach.** One file (`common/m3_impl.h`), 3 sites, gated by the existing `m->int4_kv` (`M3_INT4_KV`):
1. **Alloc** (`m3_alloc_mstream_ex`, ~843–867): when `m->int4_kv`, allocate int4 per-stream buffers
   `ms->k_q4/v_q4` (`N*n_layers*max_pos*KVD/2` bytes) + scales `ms->k_qs/v_qs`
   (`N*n_layers*max_pos*KVH` floats) instead of (or alongside) the bf16 `ms->kc/vc`. Mirror the
   single-stream per-layer layout at line 333.
2. **Write** (batch KV store, ~891–893): replace `kc[i]=m3_f2bf(kb[i])` with a per-kv-head
   `m3_q4_pack(...)` into `ms->k_q4/v_q4` + scale — copy the single-stream pack at line 665.
3. **Read** (batch attention, ~897–904): replace the bf16 `m3_bf2f` dot/axpy inner loops with
   `m3_q4_dot`/`m3_q4_axpy` over `ms->k_q4/v_q4` + per-position scale — copy the single-stream reads at
   lines 571/577. (Also the MSA idx cache `idx_q4` if long-ctx MSA select is on.)

**Reuse.** The whole int4 codec + the single-stream call sites are the template — this is a
"replicate the single-stream int4 path across the N-stream buffers" job, not new math.

**Risks.** (a) **Quality** — int4-KV degrades attention; the existing single-stream quality probe
(`pjsub_m3_kv_mstream_1n.sh`, bf16/fp16/int4 sweep) must be extended to the BATCH path and gated
(argmax drift vs bf16 on a real prompt). (b) **Hot-loop perf** — int4 unpack per (position, head) in
the O(ctx) attention loop may cost tok/s; the payoff is memory (more M fits), so net serving throughput
should still win, but measure. (c) buffer sizing / OpenMP per-stream races (the alloc must be
per-stream-disjoint, like the bf16 path already is).

**Validation ladder.** (1) single-node synth: `m3_selftest`/`m3_bench` batch decode with `M3_INT4_KV=1`
→ NaN=0, arena ~4× smaller KV term (confirm via `m3_decode_sim` max_M jump). (2) batched KV-quality:
extend `pjsub_m3_kv_mstream_1n.sh` to run mstream=8 real-weight bf16 vs int4, compare argmax/logits.
(3) 48n real-weight A/B at 4k ctx: `M3_MSTREAM=48 M3_INT4_KV=1 maxpos=4096` fits (bf16 OOMs at M=48) →
tok/s + coherent gen. **Effort: moderate (~150–250 LOC, codec exists), 1–2 probe jobs.**

---

## Lever 2 — cross-layer comm pipelining (hide the 25–37% all-reduce)

**Why.** Comm is 25–37% of batched decode (grows with M). Per-layer comm-overlap is **net-negative**
(m3.md: needs `TP_SHARED=0` replication whose penalty exceeds the gain; only route‖shared is
independent). The remaining structural lever is **cross-layer**: overlap layer L's all-reduce with
independent compute so the comm is hidden. Ideal payoff ~1/(1−0.3) ≈ **1.4×**; realistic η ≈ 0.5–0.7 →
**~1.15–1.25×** (→ ~17–19 tok/s). This is the ceiling the roofline flagged as "MegaMoE-style, deferred".

**Why it's hard.** The layer loop (m3_impl.h:870+) is a strict dependency chain per layer:
`attn → o_proj → ar_cb(attn) → norm → router → experts → ar_cb(moe) → residual`, and layer L+1's first
op consumes layer L's REDUCED hidden. There is no independent L+1 work to overlap the reduce with. Two
routes, both with real cost:
- **(a) Stream-pipelined double-buffer** — split the M-stream batch into 2 sub-batches; while
  sub-batch A's layer-L all-reduce is in flight (async `ar_async_start`), compute sub-batch B's layer L.
  Keeps numerics exact (each stream is independent) but ~doubles the live activation/KV working set and
  restructures the whole loop into a software pipeline. Complex.
- **(b) Relaxed/stale-sync** — feed layer L+1's ROUTER off the pre-reduce partial (routing is somewhat
  error-tolerant) while the reduce completes for the expert compute. Cheaper to code but **changes
  numerics** (not lockstep-bit-exact; needs a quality gate) — risky for a correctness-first port.

**The blocking issue.** The async-comm mechanism itself is unstable on A64FX. m3.md's pinned-pool /
comm-driver (`M3_COMM_OVERLAP`, `ar_async_start_cb`, `comm_driver` thread, m3_ep_runner.c:152–275)
**"hangs node-dependently"** — with 12 compute threads on 1 CMG, a dedicated comm thread either
oversubscribes or steals a compute core. This same cross-CMG/thread problem defeated glm5's overlap and
m3's full-48 pool. **Lever 2 is gated on first making a stable pinned comm-core mechanism** (OMP=11 +
comm thread pinned to the 12th core of the CMG), which is itself an open, previously-failed sub-project.

**Risks.** HIGH: (1) the async-comm thread stability (unsolved), (2) numerics if route (b), (3) 2×
working set if route (a) — may not fit the tight arena, (4) large forward-loop restructure with lockstep
correctness to preserve. Uncertain ~1.2× payoff.

**Validation ladder.** (1) FIRST de-risk the comm thread in isolation: a microbench that runs
`ar_async_start` on a pinned 12th core while the 11-thread pool computes, on 4–8 nodes, proving no hang
+ real overlap. Only if that passes: (2) implement route (a) double-buffer on 2 layers synth →
lockstep argmax identical to synchronous, tok/s A/B. (3) full 48/96n synth then real-weight confirm.
**Effort: LARGE / research-grade — the comm-thread de-risk alone is a mini-project; the pipeline is a
forward-loop rewrite.**

---

## Recommendation

1. **Lever 1 (int4-KV batch) — do it.** Moderate effort, codec exists, clear +12% long-ctx / node
   savings, low-moderate risk (quality-gated). Unblocks long-context serving at the M=48 peak.
2. **Lever 2 (cross-layer pipeline) — defer / spike only.** Start with the *isolated pinned-comm-core
   microbench*; do NOT invest in the pipeline until that de-risks the async-comm hang. If it hangs
   again (as before), Lever 2 stays deferred and decode is comm-bound at the current ~15.25 tok/s until
   a megakernel or a different threading model lands.

Neither is a probe; both are implementation work. Lever 1 is the clear next build.
