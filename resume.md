# Resume Prompt: GLM-5.2 FP8 Fugaku Optimization

You are continuing optimization work in:

```sh
cd /mnt/nvme02/work/fugaku/work/gemm/glm5-1
```

The real working tree on Fugaku is synced by mutagen:

```sh
ssh fugaku 'cd ~/work/gemm/glm5-1 && ...'
mutagen sync flush glm5-1
```

Local git metadata may be broken or point at a Fugaku worktree. Prefer remote git commands:

```sh
ssh fugaku 'cd ~/work/gemm/glm5-1 && git status --short && git log --oneline -10'
```

Do not push without explicit user permission.

## Current State

Target: GLM-5.2 FP8 real weights on Fugaku A64FX, optimizing chunked prefill and long-context viability. Weights/tokenizer are under:

```sh
~/models/glm52-fp8
```

Important recent commits on remote:

```text
6af832a Fix GLM5 prefill router GEMM reading bf16 gate as f32
65deada Tier-drive MSA: auto-disable for single Tier A
230a56d Context-tiered prefill: auto Tier A (un-sharded) -> Tier B (CP)
4a19895 Make GLM5 EP runner barrier robust for 384-node incast
5005c28 Add GLM5 prefill node-scaling scripts, NUMA opt, and scaling study
7f4bb51 Raise tp_allreduce TP_AR_MAXN and NSTEP for 384-node EP groups
d5513d3 Raise GLM5 EP runner MAX_NODES to 512 for 384-node runs
21ec0ff Default GLM5 FP8 GEMM to 5-token blocking
9eb2aa6 Block 4 tokens in GLM5 FP8 batched GEMM inner kernel
2767bea Parallelize GLM5 CP prefill attention with ordered combine
7f80fce Serialize GLM5 CP prefill combines
d2f8fbb Cap GLM5 allreduce token window
35e5d06 Cap GLM5 prefill allreduce payload
9db922a Reduce GLM5 long-context allreduce slots
46b7ff5 Optimize GLM5 FP8 prefill allreduce
974747f Shard GLM5 shared FP8 blocks
49c898d Optimize GLM5 FP8 batched GEMM
afba912 Fix GLM5 FP8 192-node prefill layout
ef7982d Optimize GLM5 FP8 block-scale decode
8c86c41 Optimize GLM5 FP8 batched prefill
```

Key files:

```text
a64fx/glm5/glm5_ep_runner.c
common/glm5_impl.h
a64fx/glm5/pjsub_glm5_prefill_fp8_96n_noncontig.sh
a64fx/glm5/pjsub_glm5_prefill_fp8_192n.sh
```

Build command on Fugaku:

```sh
ssh fugaku 'cd ~/work/gemm/glm5-1 && PATH=/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:$PATH make -C a64fx/llm glm5_stage glm5_ep_runner CC=fccpx OPENMP=1'
```

Expected warning: `_GNU_SOURCE macro redefined`. This warning is currently benign.

## Important Runner Knobs

Allreduce slot sizing:

```text
GLM5_AR_AUTO_CAP=256   # auto cap for inferred ar_tokens
GLM5_AR_HARD_CAP=64    # hard cap, also caps explicit GLM5_AR_TOKENS unless set to 0
GLM5_AR_TOKENS=N       # explicit requested allreduce token window
```

The hard cap was added because `GLM5_AR_TOKENS=1024` registered/reduced about 24 MiB per collective and caused `tp_ar ... bcast timeout`. With `GLM5_AR_HARD_CAP=64`, the same job succeeds.

Long-context CP:

```text
GLM5_CP=1
GLM5_INT4_KV=1
GLM5_MSA=1
GLM5_MAXPOS=1048576
```

For CP, `ar_tokens` defaults to 1. As of commit `2767bea`, the CP prefill attention loop in `common/glm5_impl.h` (`glm5_forward_prefill_chunk`) is split into two phases: the per-token flash-attention math runs fully OpenMP-parallel (it touches no uTofu, only per-token scratch `ms->kvb`/`ms->v`/`ms->attn` plus new per-token `ms->hmx`/`ms->hse`), and the uTofu `kv_combine_cb` is deferred to a serial token-ordered loop so every rank still issues collectives in identical order. This replaced the earlier fully-serial loop (commit `7f80fce`) that had been added to dodge the `TOQ Direct Descriptor Exception` from concurrent uTofu. Verified: no descriptor/timeout errors, NaNs=0, and 1M CP prefill went 1.07 -> 3.59 tok/s (attn 746 -> 88 ms/tok).

Communication overlap:

```text
GLM5_COMM_OVERLAP=1
```

This uses a dedicated comm-driver thread for routed expert allreduce while shared expert compute runs.

## Reproduction Commands

Flush local changes before remote build/job submission:

```sh
mutagen sync flush glm5-1
```

Build:

```sh
ssh fugaku 'cd ~/work/gemm/glm5-1 && PATH=/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:$PATH make -C a64fx/llm glm5_stage glm5_ep_runner CC=fccpx OPENMP=1'
```

Check queue:

```sh
ssh fugaku 'pjstat | egrep "glm5|JOB_ID|492"'
```

Stable 96-node baseline (best config: `pchunk=256 + TP_SHARED=1 + COMM_OVERLAP=1`, 16.67 tok/s
with the default tok=5 FP8 GEMM kernel):

```sh
ssh fugaku 'cd ~/work/gemm/glm5-1 && pjsub --no-check-directory \
  -x GLM5_MAXPOS=2048 \
  -x GLM5_PREFILL_SYNTH=1024 \
  -x GLM5_CP=0 \
  -x GLM5_INT4_KV=0 \
  -x GLM5_MSA=0 \
  -x GLM5_PCHUNK_SWEEP=256 \
  -x GLM5_TP_SHARED=1 \
  -x GLM5_THREAD_SWEEP=24 \
  -x GLM5_COMM_OVERLAP=1 \
  -x TP_AR_BF16=1 \
  a64fx/glm5/pjsub_glm5_prefill_fp8_96n_noncontig.sh'
```

Bounded large-chunk test, explicitly asking for 1024 allreduce tokens but hard-capped to 64:

```sh
ssh fugaku 'cd ~/work/gemm/glm5-1 && pjsub --no-check-directory \
  -x GLM5_MAXPOS=2048 \
  -x GLM5_PREFILL_SYNTH=1024 \
  -x GLM5_CP=0 \
  -x GLM5_INT4_KV=0 \
  -x GLM5_MSA=0 \
  -x GLM5_TP_SHARED=1 \
  -x GLM5_PCHUNK_SWEEP=1024 \
  -x GLM5_THREAD_SWEEP=12 \
  -x GLM5_AR_TOKENS=1024 \
  a64fx/glm5/pjsub_glm5_prefill_fp8_96n_noncontig.sh'
```

Short 1M-context CP smoke:

```sh
ssh fugaku 'cd ~/work/gemm/glm5-1 && pjsub --no-check-directory \
  -x GLM5_PREFILL_SYNTH=128 \
  -x GLM5_PCHUNK_SWEEP=64 \
  -x GLM5_THREAD_SWEEP=12 \
  a64fx/glm5/pjsub_glm5_prefill_fp8_96n_noncontig.sh'
```

192-node stable config:

```sh
ssh fugaku 'cd ~/work/gemm/glm5-1 && pjsub --no-check-directory \
  -x GLM5_MAXPOS=2048 \
  -x GLM5_PREFILL_SYNTH=1024 \
  -x GLM5_CP=0 \
  -x GLM5_INT4_KV=0 \
  -x GLM5_MSA=0 \
  -x GLM5_PCHUNK_SWEEP=64 \
  -x GLM5_THREAD_SWEEP=12 \
  a64fx/glm5/pjsub_glm5_prefill_fp8_192n.sh'
```

Note: `pjsub_glm5_prefill_fp8_192n.sh` defaults `GLM5_TP_FFN=0` because 192-way dense FFN column shards can be 64 wide and violate FP8 128-column scale-block alignment.

## Log Collection

For a job id:

```sh
ssh fugaku 'cd ~/work/gemm/glm5-1 && \
  id=49289251; \
  d=$(ls -d a64fx/glm5/prefill_fp8_run_${id}_* 2>/dev/null | head -1); \
  echo dir=$d; \
  tail -200 "$d/glm5_ep_rank00.txt"; \
  tail -120 "$d/glm5_ep_stderr_rank00.txt"; \
  tail -120 pjsub_glm5_prefill_fp8_*${id}.out'
```

Concise metric extraction:

```sh
ssh fugaku 'cd ~/work/gemm/glm5-1 && \
  grep -hE "allreduce:|CP ON|comm-overlap|prefill_progress:|prefill_synth:|PROFILE prefill_synth|SENTINEL|FATAL|timeout|Direct Descriptor|PLE" \
  a64fx/glm5/prefill_fp8_run_*/glm5_ep_rank00.txt \
  a64fx/glm5/prefill_fp8_run_*/glm5_ep_stderr_rank00.txt 2>/dev/null'
```

## Known Results

### Context-tiered prefill (2026-06-22, commits 230a56d + 65deada) -- single binary auto-switches

The 1M-capable CP config (CP=1/MSA=1/INT4=1) ran a flat ~2 tok/s at EVERY context length at 384n
(per-token CP combine + 384-way comm dominate). A single binary now auto-picks the algorithm by
context length, no env code path:
- Tier A (ctx <= T_cp): cp_on=0 bf16, KV replicated, MSA OFF (dense, exact, faster) -- the fast path.
- at T_cp: glm5_prefill_to_cp re-shards the KV in place (LOCAL, ~21ms: keep CP-owned blocks, bf16->int4).
- Tier B (ctx > T_cp): cp_on=1 int4 CP-sharded, MSA on -- the 1M path.
T_cp from the per-rank KV memory budget (GLM5_KV_BUDGET_GB, ~39k at 4GB). GLM5_CP_THRESHOLD<0 keeps
legacy static env-CP for A/B.

```text
384n, 8k ctx (single Tier A), TP_SHARED=1:
  CP-from-start (old):        ~2.05 tok/s    (flat across all ctx)
  tiered, MSA=1 (TP_SHARED=0): 4.79
  tiered, MSA auto-off:       13.26 tok/s comm 51% (with commit 7f6d620) -- 6.5x over CP-from-start
  forced-CP 8k (batched combine): 2.30 (vs ~2.05; per-token CP combine + 384-way comm inherent)
  (MSA=1 cripples Tier A; auto-tier disables MSA for single Tier A. 7f6d620 batched the CP combine
   AND widened the prefill allreduce window ar_tokens 1->64 (max_count 6144->393216), cutting frags
   ~45x (1.28M->28k) -> comm 69%->51% -> Tier A ~8 -> 13.26. Absolute still comm-bound at 384 ranks.)
real 24n transition correctness (ctx=2048@512): re-shard 21ms, NaNs=0, SENTINEL done.
```

Recommend `TP_SHARED=1` for the tiered path. Out of scope (measured dead-end): 4-rank node scaling.

### Scaling study (2026-06-22) — practical ceiling ~18.7 tok/s; 50 needs re-architecture

Full study + the 4-ranks-per-node design: `a64fx/doc/glm5_prefill_scaling.md`. Short-context
(CP=0, maxpos=2048), 96n, pchunk=256, tok=5, comm-overlap, synth=1024:

```text
threads 12/24/48:                16.5 / 18.5 / 14.1 tok/s   (th=48 REGRESSES: cross-CMG fork-join)
th=48 + numactl interleave+pin:  16.3 (spread) / 13.2 (close)  -- still < th=24
th=24 + numactl interleave:      18.65  (flat vs no-NUMA)
nodes 48/96 (TP_SHARED=1):       15.7 / 16.5   (head-sharding saturates at 64 heads)
TP_SHARED 0->1:                  +10-14%, comm 60%->35%
TP_FFN=1:                        no gain (dense FFN only 3/78 layers)
pure-EP (all TP off):            OOM (59 GB arena > 31 GB) -- dense must be TP-sharded
384n short (CP=0):               15.83 tok/s  (now runs end-to-end; see 384-node enablement below)
bf16 allreduce (TP_AR_BF16=1):   +9% CONFIRMED on torus: fp32 16.09 -> bf16 17.55, comm 43%->33%, argmax IDENTICAL
TP_ATTN=0 (drop o_proj reduce):  3.05 tok/s  -- DEAD END (64 heads / 96 ranks: un-shard = 64x attn)
```

**Noncontig placement variance is large (~±20%)** -- it masked the bf16-AR signal. On TORUS (192n,
consistent placement) bf16-AR is a confirmed **+9%** (16.09->17.55 tok/s, comm 43%->33%, argmax
IDENTICAL fp32-vs-bf16 -> output-equivalent on synth). Recommend `TP_AR_BF16=1`. Router
GEMM bf16-as-f32 over-read fixed (commit 6af832a, correctness). Multi-rank/node uTofu addressing
fixed (commit, distinct TNI per local rank) -- 2-ranks/node now runs (17.34 tok/s, 192 ranks). 4-rank
staging blocked on /local. **4-ranks/node MEASURED (40L test): 16.0 tok/s, WORST** (1-rank th24=40.3,
th48=27.1). More ranks backfire: 384-way comm explodes (61%) + un-shardable replicated dense
recomputed per rank + attention caps at 64 heads. Phase 2 CLOSED -- 2-CMG/th24 is the genuine
optimum; ~20 tok/s (78L) is the A64FX ceiling. See a64fx/doc/glm5_prefill_scaling.md.

**Best: ~18.7 tok/s** (96n, th=24, TP_SHARED=1, pchunk=256, tok=5, overlap). Three mutually
reinforcing limits cap it: (1) OpenMP cross-CMG fork-join → usable compute ~2 of 4 CMGs;
(2) 31 GB/node forces TP sharding (can't replicate to kill comm); (3) comm ~48% at that ceiling.
The DUMMY comm-floor probe and pure-EP both crashed (dummy path bit-rotted; pure-EP OOM), so the
floor was inferred. The only path to ~50 tok/s is **4 MPI ranks/node (1/CMG) sharing dense
weights via mmap + hierarchical allreduce** — a real re-architecture, not a tuning knob (design
in the doc). NOTE: only 12 of 48 cores/node are used today; `th=24` (2 CMGs) over the old `th=12`
default is a free ~+12%.

### 384-node enablement (2026-06-22) — 4 scaling walls fixed, now runs end-to-end

A 384-node EP run hit four successive walls, each fixed:
1. `MAX_NODES 256->512` (runner topo reader exit) — commit `d5513d3`.
2. `TP_AR_MAXN 256->512` (tp_comm_init nprocs guard) — commit `7f4bb51`.
3. `TP_AR_NSTEP 9->11` (recv-slot count: N=384 bcast_sid=9 hit the guard) — `7f4bb51`.
4. Non-robust barrier hung on the 383->1 fan-in **incast** (puts drop under TNI congestion;
   non-robust = no retry) — made `barrier()` robust, commit `4a19895`.
Result: 384n prefill (synth=1024, CP=0, pchunk=256, TP_SHARED=1) = 15.83 tok/s, NaNs=0,
`SENTINEL glm5_prefill_384n=done`. Slightly below 96n (16.5) — comm grows with rank count, as
the flat node-scaling predicts. Use `a64fx/glm5/pjsub_glm5_prefill_fp8_384n_noncontig.sh`.

### Latest (2026-06-21, commit `21ec0ff` — FP8 GEMM 5-token blocking)

The stable path was FP8-GEMM-bound (`shared`+`qkv` ~65%). The batched GEMM
(`glm5_gemm_mxfp8`) decodes each 8-row x 512-col tile to bf16 once, then runs the inner matvec
over all N tokens. That inner loop is L1-load-bound on the bf16 weight tile — decoding to f32
instead (2x tile bytes) measured 21% SLOWER (269 -> 212 Gop/s), proving weight-tile traffic, not
FP-pipe throughput, is the limit. Fix: block more tokens per widened-weight load (`glm5_bf16_4row_
{4,5}x_acc`) — cuts relative weight traffic and exposes more independent FMA chains. tok=5 uses
20 acc + 4 w + 1 x = 25 SVE regs. `GLM5_FP8_GEMM_TOK` selects 5 (default), 4, or 3 (previous);
all bit-exact (max_rel unchanged at 1.42e-6).

```text
kernel bench (1n, N=256 rows=2048 cols=6144):  tok=3 230  ->  tok=4 295  ->  tok=5 330 Gop/s
full 96n prefill pchunk=256 TP_SHARED=1 overlap, matched A/B:
  job 49289888 tok=3  14.52 tok/s  qkv 17.36 ms/tok  shared 29.59  wall 70.5 s
  job 49289887 tok=4  15.70-15.84  qkv 13.6-13.8     shared 27.0
  job 49289940 tok=5  16.67 tok/s  qkv 11.45 ms/tok  shared 25.09                <- new 96n best
=> tok=3->5 +14.8% overall; qkv -34%, shared -15%, comm unchanged; NaNs=0.
```

pchunk sweep (TP_SHARED=1, overlap, synth=1024): pchunk=128 14.94, **256 15.84+**, 512 14.11 —
256 is still the optimum. Best 96n stable config: `pchunk=256 + TP_SHARED=1 + COMM_OVERLAP=1` at
16.67 tok/s (default tok=5). `shared` is still ~42% — the FP8 GEMM stays the top cost (next
levers: tok=6 / 16-row decode tiles, or shared-expert FLOP reduction).

### Earlier (2026-06-21, commit `2767bea` — CP parallel combine)

1M-context CP smoke, parallel-combine vs the earlier serial combine (both 96n, synth=128, `GLM5_CP=1 INT4_KV=1 MSA=1 MAXPOS=1048576`, pchunk=64):

```text
job 49289784 (parallel combine, 2767bea): 128 tok 3.59 tok/s comm 18.5% NaNs=0
  wall=35.70 s  attn 88.425 ms/tok (31.8%)  o_proj 75.1 (27%)  shared 51.0 (18%)  qkv 34.9 (12.5%)
job 49289446 (serial combine, 7f80fce):   128 tok 1.07 tok/s comm  8.2% NaNs=0
  wall=119.5 s  attn 746.539 ms/tok (80.0%)
=> 3.35x overall, 8.4x on attention; no TOQ Direct Descriptor / bcast timeout.
```

96n stable-prefill pchunk sweep, all `GLM5_COMM_OVERLAP=1`, no CP/MSA/int4, synth=1024:

```text
job 49289788  pchunk=64                10.95 tok/s comm 56.9%
job 49289789  pchunk=128               12.00 tok/s comm 56.3%
job 49289790  pchunk=256               12.24 tok/s comm 54.8%
job 49289817  pchunk=256 TP_SHARED=1   14.34 tok/s comm 37.3%   <- new 96n best
(prior 49289533 pchunk=1024            10.54 tok/s comm 26.9%)
```

Best 96n stable config is `pchunk=256 + TP_SHARED=1 + COMM_OVERLAP=1` at 14.34 tok/s (+36% over
the old pchunk=1024 baseline). Sharding the shared expert (`TP_SHARED=1`) cuts both per-rank
shared-GEMM rows and comm % (54.8% -> 37.3%). Replicated `pchunk=256` alone is 12.24 tok/s; the
pchunk optimum is mid-range, not the largest chunk. At 14.34 tok/s `shared` is still 42.3%
(29.4 ms/tok) and `qkv_proj` 22.1% (15.3 ms/tok) — the path is FP8-GEMM-bound.

192n overlap retest, `pchunk=64 COMM_OVERLAP=1`, synth=1024:

```text
job 49289786  1024 tok 10.70 tok/s comm 51.8% NaNs=0
  qkv 15.7  attn 3.37  o_proj 23.2 (25%)  router 13.2  shared 28.4 (30.6%)  dense_ffn 6.0
```

Overlap helps 192n (vs 9.61 non-overlap job 49289532, ~5.95 old baseline) but 192n is comm-bound (o_proj allreduce 25% + 51.8% comm).

### Earlier results

96-node, non-contig, full 78 layers, FP8, `pchunk=64`, `GLM5_COMM_OVERLAP=1`, no CP/MSA/int4:

```text
job 49289251
prefill_synth: 1024 tok 11.01 tok/s comm 56.9% calls=3521 frags=3521 pchunk=64 argmax=16 NaNs=0
PROFILE wall=92.991031 s tokens=1024 comm=52.902536 s
qkv_proj 15.733 ms/tok
attn      3.370 ms/tok
o_proj   24.697 ms/tok
router   13.168 ms/tok
experts   2.674 ms/tok
shared   29.574 ms/tok
```

This is a major speedup from the earlier non-overlap 96-node run at about `4.25 tok/s`.

96-node, non-contig, `pchunk=1024`, `TP_SHARED=1`, explicit `GLM5_AR_TOKENS=1024`, hard cap active:

```text
job 49289431
allreduce: max_count=393216 floats ar_tokens=64 pchunk=1024 mstream=1 ar_auto_cap=256 ar_hard_cap=64
prefill_synth: 1024 tok 10.55 tok/s comm 27.7% calls=1181 frags=3521 pchunk=1024 argmax=16 NaNs=0
PROFILE wall=97.090014 s tokens=1024 comm=26.923229 s
qkv_proj 26.745 ms/tok
attn      4.883 ms/tok
o_proj   10.211 ms/tok
router    7.972 ms/tok
shared   40.672 ms/tok
```

Without the hard cap, `pchunk=1024`, `ar_tokens=1024` failed:

```text
jobs 49289241, 49289242
allreduce: max_count=6291456 floats ar_tokens=1024 pchunk=1024
tp_ar: rank 0 bcast timeout sid=7/8 want=1025 got=1024
```

96-node, non-contig, 1M context CP smoke after CP combine serialization:

```text
job 49289446
GLM5_CP=1 GLM5_INT4_KV=1 GLM5_MSA=1 GLM5_MAXPOS=1048576
allreduce: max_count=6144 floats ar_tokens=1 pchunk=64
CP ON: KV sharded block-cyclic (block=128) over 96 ranks, 11008 slots/rank, int4_kv=1
prefill_synth: 128 tok 1.07 tok/s comm 8.2% calls=10269 frags=50049 pchunk=64 argmax=15 NaNs=0
PROFILE wall=119.515317 s tokens=128 comm=9.812756 s
attn 746.539 ms/tok, 80.0% of measured time
```

Before serialization, the same CP smoke failed:

```text
job 49289250
utofu: asynchronous error: TOQ Direct Descriptor Exception on TNI 00 CQ 00
```

## Current Interpretation

1. `GLM5_COMM_OVERLAP=1` is the biggest proven win for the stable 96-node prefill path.
2. Very large allreduce registered windows are unsafe. Keep `GLM5_AR_HARD_CAP=64` unless doing controlled uTofu experiments.
3. CP + 1M context is now both viable AND much faster: commit `2767bea` made the CP attention math OpenMP-parallel while keeping the uTofu combine serial+ordered. 1M CP went 1.07 -> 3.59 tok/s; attention fell from 80% to 31.8% of time. With attention parallelized, `o_proj` (27%) and `shared` (18%) are now the next CP costs.
4. Best stable 96n config is `pchunk=256 + TP_SHARED=1 + COMM_OVERLAP=1` at **16.67 tok/s** (default tok=5 FP8 GEMM kernel, commit `21ec0ff`). Sweep (overlap on): pchunk 64=10.95, 128=12.00, 256=12.24 replicated; `TP_SHARED=1` -> 14.5; 4-token GEMM -> 15.8; 5-token GEMM -> 16.67. The pchunk optimum is mid-range, not the largest.
5. The FP8 batched GEMM inner loop is L1-load-bound on the bf16 weight tile (f32 tile is slower; more token blocking is faster). At 16.67 tok/s `shared` is still ~42% — the FP8 GEMM remains the top cost. Next levers: tok=6 (24 acc + 4 w = 28 regs), 16-row decode tiles for more FP8-decode reuse, or shared-expert FLOP reduction.

## Recommended Next Work

Priority 1 (DONE, commit `2767bea`): CP prefill combine is now ordered-but-parallel. The
attention math runs OpenMP-parallel; `kv_combine_cb` is deferred to a serial token-ordered
loop. 1M CP: 1.07 -> 3.59 tok/s, attn 746 -> 88 ms/tok, NaNs=0, no uTofu descriptor errors.

Priority 2: Now that attention is parallel, CP is gated by `o_proj` (27%) and `shared` (18%),
not attention. Next CP levers:
- Test `GLM5_AR_TOKENS=2/4/8` for CP — the combine is now ordered/serial so the uTofu ordering
  invariant holds; CP still fragments heavily (`calls=10269 frags=50049` for 128 tokens at
  ar_tokens=1). Raise cautiously and watch for descriptor/timeout regressions.
- The CP combine is still fully serial over tokens; if it becomes the bottleneck at larger
  ar_tokens, consider a comm-driver queue so one thread drains uTofu while compute prepares
  later tokens (the `comm_driver` thread in `glm5_ep_runner.c` is a template).

Priority 3 (DONE): best stable 96n config is `pchunk=256 + TP_SHARED=1 + COMM_OVERLAP=1` at
14.34 tok/s (job 49289817). Sweep: pchunk 64=10.95, 128=12.00, 256=12.24 replicated; adding
`TP_SHARED=1` at pchunk=256 -> 14.34 tok/s and drops comm 54.8% -> 37.3%. `TP_SHARED=1` at the
other pchunk values is untested and may improve them too. To re-run the pchunk sweep, submit
SEPARATE jobs per value — `pjsub -x` splits a space-separated `GLM5_PCHUNK_SWEEP="64 128 256"`
and fails with `File open failed: 128`; loop in the shell instead:

```sh
ssh fugaku 'cd ~/work/gemm/glm5-1 && for pc in 64 128 256; do pjsub --no-check-directory \
  -x GLM5_MAXPOS=2048 -x GLM5_PREFILL_SYNTH=1024 \
  -x GLM5_CP=0 -x GLM5_INT4_KV=0 -x GLM5_MSA=0 \
  -x GLM5_COMM_OVERLAP=1 -x GLM5_PCHUNK_SWEEP=$pc -x GLM5_THREAD_SWEEP=12 \
  a64fx/glm5/pjsub_glm5_prefill_fp8_96n_noncontig.sh; done'
```

Priority 4 (DONE): 192-node overlap retest = 10.70 tok/s (job 49289786), up from 9.61 without
overlap and ~5.95 old baseline. 192n is comm-bound (o_proj allreduce 25% + comm 51.8%); the
next 192n lever is reducing o_proj/route allreduce volume, not pchunk.

Priority 5 (PARTIALLY DONE, commits `9eb2aa6` + `21ec0ff`): The stable path is FP8-GEMM-bound.
The batched GEMM inner loop is L1-load-bound on the bf16 weight tile (f32 tile 21% slower). Added
token blocking (`glm5_bf16_4row_{4,5}x_acc`, `GLM5_FP8_GEMM_TOK=5` default): kernel tok=3 230 ->
tok=5 330 Gop/s, full 96n 14.52 -> 16.67 tok/s (+14.8%, matched A/B). Remaining GEMM levers:
- tok=6 (`glm5_bf16_4row_6x_acc`, 24 acc + 4 w = 28 SVE regs) — diminishing (load:FMA 2.22->2.40).
- 16-row decode tiles (decode 16 rows once) to amortize the FP8->bf16 decode further; needs more
  accumulators or a second decode-reuse pass over the staged tile.
- shared-expert FLOP reduction (it stays ~42% of wall even after the kernel wins).
Use the 1-node `glm5_fp8_kernel_test` for fast A/B (build `make glm5_fp8_kernel_test CC=fccpx
OPENMP=1`; submit `pjsub_glm5_fp8_kernel_test_1n.sh` with `-x GEMM_N=256 -x GLM5_FP8_GEMM_TOK=N`);
it prints `FP8_GEMM ... Gop/s` and checks bit-exactness. NOTE: node-to-node speed varies ~10%, so
trust same-batch A/B deltas, not absolute Gop/s across jobs. PJM `.out` lands in the submit dir
(`~/work/gemm/glm5-1`), not `a64fx/glm5`.

## Commit Workflow

After edits:

```sh
mutagen sync flush glm5-1
ssh fugaku 'cd ~/work/gemm/glm5-1 && git diff --check'
ssh fugaku 'cd ~/work/gemm/glm5-1 && PATH=/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:$PATH make -C a64fx/llm glm5_ep_runner CC=fccpx OPENMP=1'
ssh fugaku 'cd ~/work/gemm/glm5-1 && git add <files> && git commit -m "<imperative subject>"'
```

Do not remove untracked run directories or PJM artifacts unless explicitly asked.

## Data-parallel prefill groups + dynamic merge (glm5-2, Jun 2026)

Split the N-rank EP pool into G independent groups: group-local ep_rank/ep_size, group-scoped
gbarrier + tp_comm (PeerVcq+GBase). Each group prefills its own sequence; short context fits across
far fewer than all ranks, so groups recover the wasted per-token collective overhead as aggregate
throughput.

Phase 1 (static, GLM5_PREFILL_GROUPS=G), 384n @ 8k, all NaNs=0:
  1x384 = 13.3 | 2x192 = 29.0 (2.2x) | 4x96 = 60.5 tok/s (4.6x).
  Per-group rate rises as groups shrink (cheaper collectives): 13.3 -> 14.5 -> 15.1; single ep96
  = 15.96. pick_groups auto-selects the largest G whose Tier-A KV budget at ep=N/G holds the
  target ctx; GLM5_PREFILL_GROUPS overrides. glm5_stage shards group-locally (rank %= ep_size);
  384n script sets GLM5_EP_SIZE=NP/NGRP (do NOT name the bash var GROUPS - reserved array).

Phase 2 (dynamic merge, GLM5_MERGE_AT=p1:p2): one job traverses 4x96->2x192->1x384 as the
  surviving (even-subgroup) sequence grows; concurrency steps 4->2->1. Each merge:
  - routed experts: LOCAL drop (glm5_group_expert_drop) - new owner e%2g already holds it since
    both sibling subgroups staged the full model (no transfer);
  - TP-dense weights: in-place re-slice from the node-local blob (glm5_group_tp_reslice) - stage
    keeps dense un-sharded, so each rank re-slices its new ep shard locally (no transfer);
  - KV: uTofu pairwise propagate even->odd with dc-civac cache coherence (glm5_group_kv_propagate);
  - comm/gbarrier rebuilt over the merged group.
  Validated LOSSLESS: 24n ep12->ep24 survivor argmax 60590 == G=1 ref, NaNs=0. 384n ladder:
  merges 18.2s/30.2s, survivor 8192 tok NaNs=0, per-tier survivor 17.2/14.9/13.6 tok/s.
  NOTE: pjsub -x splits on commas -> use : in GLM5_MERGE_AT. Commits on glm5-2 (unpushed):
  288de08b 9825950c f68c632d 8e852f77 e34fd707.
