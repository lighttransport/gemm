# DeepSeek-V4-Flash (DS4F) on 11× A64FX (Fugaku) — build & run repro

EP-only inference harness for DeepSeek-V4-Flash (284B total / 13B activated MoE,
43 MoE layers, hidden 4096, vocab 129280). 256 routed experts are EP-sharded
~23–24/node; dense (attn / shared expert / router / embed / head) is replicated and
computed redundantly per node. Weights stay quantized in HBM with on-demand dequant
(dense FP8 e4m3fn, experts MXFP4) to fit ~25 GB usable HBM2/node.

This file is the **operational repro** for the real-weight Tier-B1 run. The science /
status lives in the auto-memory `project_ds4f_flash_ep_harness`; the design rationale
in `~/.claude/plans/plan-deepseek-v4-flash-iterative-prism.md`.

## TL;DR

```sh
# inside a live 12-node alloc (shape 2x3x2; 1 node reserved for agent/claude control),
# from a64fx/llm:
./run_ds4f_stage_11n.sh                                  # 1. shard weights -> each node's /local (~22 GB/node)
DS4F_REAL=1 DS4F_EXACT=1 ./run_ds4f_11n.sh                # 2. exact Tier-B1 forward on the 11 EP nodes
DS4F_REAL=1 DS4F_TIERB2=1 ./run_ds4f_11n.sh              # 2b. Tier-B2 compressor/indexer (reuses blobs)
DS4F_REAL=1 DS4F_FP8_BF16=1 ./run_ds4f_11n.sh           # 2c. bf16-pv dense promote (fast decode)
./run_ds4f_longctx_11n.sh                                # 2d. long-ctx Tier-B2 bench (O(topk) payoff)
```

All launchers default to `NP=11`, `EXCLUDE=0,0,0` (drops the agent/claude-colocated
node — the 12th node is reserved for agent control and would OOM under the ~22 GB load),
and regenerate `vcoord_ds4f.txt` + the Tofu topology for the *current* alloc. Steps 2b–2d
reuse the Step-1 blobs (no re-stage). Last Tier-B1 validation (job 49092345, 2026-06-05):
`rc=0`, NaNs=0, argmax=122293, prefill 10.44 / decode 10.16 tok/s, arena 22.18 /
RSS 21.81 GB/node. Per-config numbers are in Steps 2–2d.

## NUMA lever, CLI flags & minimal-node presets (2026-07)

**NUMA interleave (grafted from glm5).** `ds4f_ep_runner` now interleaves its arena across all CMGs
in-process (`ds4f_apply_numa`: `set_mempolicy(MPOL_INTERLEAVE)`), the in-process equivalent of
`numactl --interleave=all`. ds4f decode is fp8-stream memory-bound, so this is the same ~1.40×
bit-identical decode lever measured on glm5. **On by default (`DS4F_NUMA=1`)**; the thread-affinity
half (`OMP_PROC_BIND=close`, `OMP_PLACES=cores`) is exported at launch by `run_ds4f_11n.sh`. A/B with
`DS4F_NUMA=0` (or `--numa 0`) — the token stream is bit-identical either way.

**CLI front-end.** `ds4f_ep_runner --flags` map to the `DS4F_*` env (env still works as fallback):

| flag | env | flag | env |
|---|---|---|---|
| `--numa[=0\|1]` | (apply; default 1) | `--preset decode` | FP8_BF16+Q8_DENSE+HC_PAR+HC_RMSPAR+TIERB2+MHC+OPROJ_FUSE+ATTN_SVE |
| `--model DIR` | `DS4F_MODEL_DIR` | `--real N` | `DS4F_REAL` |
| `--stage-dir D` | `DS4F_STAGE_DIR` | `--ep-size N` | `DS4F_EP_SIZE` |
| `--nshards N` | `DS4F_NSHARDS` | `--layers N` | `DS4F_LAYERS` |
| `--prefill N` | `DS4F_PREFILL` | `--max-gen N` | `DS4F_MAXGEN` |
| `--maxpos N` | `DS4F_MAXPOS` | `--max-new N` | `DS4F_MAX_NEW` |
| `--cp N` | `DS4F_CP` | `--int8-kv N` | `DS4F_INT8_KV` |
| `--mtp N` | `DS4F_MTP` | `--tierb2 N` | `DS4F_TIERB2` |
| `--prompt-ids F` | `DS4F_PROMPT_IDS` | `--set KEY=VAL` | any `DS4F_*` |

**Unified batch launcher `a64fx/llm/pjsub_ds4f.sh`** (MODE = prefill / decode / serve / serve1m /
dbbench). Unlike the interactive `run_ds4f_*11n.sh` (which reserve 1 node for the control session),
this is a pjsub batch job where all `NODES` are EP ranks (rank 0 doubles as driver — **no dedicated
controller node**). Default `NODES=8` (the minimal-node fast-decode sweet spot). It stages then runs
via the CLI flags:

```sh
NODES=8  MODE=decode  MAXGEN=64 pjsub a64fx/llm/pjsub_ds4f.sh          # fast Q8 decode @ minimal floor
NODES=16 MODE=prefill PREFILL=2048 pjsub a64fx/llm/pjsub_ds4f.sh       # TTFT, run above the floor
NODES=8  MODE=serve1m MAXPOS=1048576 pjsub a64fx/llm/pjsub_ds4f.sh     # 1M-ctx serve (CP-int4 caches)
NODES=12 MODE=serve   CP=1 MTP=1 PROMPT_IDS=... pjsub a64fx/llm/pjsub_ds4f.sh  # prefill+decode serving
```
(To change node count edit **both** `#PJM node=/proc=` **and** `NODES`.)

**Minimal-node floor — three tiers** (`a64fx/llm/ds4f_sim.py`, ragged `e%N`: fullest node =
`9.02 replicated + 150.59·⌈256/N⌉/256 experts`, ≤27 GB usable). 256 experts do **not** divide by 6 or
7, so the fullest node bounds memory — **6 EP nodes do NOT fit** (weights alone ~28 GB > usable):

| tier (dense sharding) | decode speed | weight floor | note |
|---|---|---|---|
| **fp8ondemand** (`TP_HEAD`+`TP_EMBED`, FP8 dense) | FP8 (dequant-bound) | **≥8** | **MEASURED @8 EP: fits, RSS 24.57 GB, decode 8.69 tok/s, coherent** |
| q8fast (`+ Q8_DENSE`, needs bf16 promote) | fast Q8 (~13 tok/s@11n) | **≥9** | **8 OOMs at LOAD** — bf16-promote peak ~28.2 GB busts the node before Q8 reclaims |
| fp8full (`+ TP_ATTN/SHARED/OPROJ/WOB`) | ~2× slower | **≥7** (weights) | absolute weight floor; `Q8_DENSE`+`TP_ATTN` is a wrong-output bug (WS7) |

> **Measured 2026-07-08 (job 49482631, 12-node interactive alloc, TP_HEAD/EMBED on, MAX_NEW=64):**
> - **8 EP, fast `Q8_DENSE`:** OOM-kills at load (arena ~28.24 GB, rank SIGKILL) — Q8 first promotes
>   dense to bf16-pv (+~6 GB) before repacking; that load-peak busts the ~30 GB node.
> - **8 EP, plain FP8 dense** (`DS4F_FP8_BF16=0 DS4F_Q8_DENSE=0`): fits **RSS 24.57 GB/node**, decode
>   **8.69 tok/s** / prefill 9.41, NaN=0, coherent — but FP8 is dequant-bound (slower).
> - **9 EP, fast `Q8_DENSE`:** fits (bf16-promote peak arena 26.6 GB, no OOM), reclaims to **RSS
>   22.92 GB/node**, decode **13.05 tok/s** / prefill 14.68, NaN=0, coherent — **matches 11-node fast
>   decode with 2 fewer nodes** (decode is comm/BW-bound, flat in N above the floor).
>
> Net: **8 nodes = slower FP8 (~8.7 tok/s); 9 nodes = full fast Q8 (~13 tok/s) and is the minimal-node
> fast-decode sweet spot.** The fast-Q8 path needs ≥9 nodes (its bf16-promote load-peak, not the
> steady arena, is the limiter).

Above the weight floor, node count for **serving** is pinned by persistent KV (`int8-KV`/`DS4F_CP`
cut it). MLA-KV min-nodes (weights EP-sharded + KV context-parallel):

| pattern | ctx | M | min nodes (bf16-KV / int8-KV) |
|---|---|---|---|
| decode (short) | 4k | 8 | **8** fast-Q8 / 9 plain |
| serve | 128k | 1 | **9** |
| serve | 512k | 1 | **10** |
| serve | 1M | 1 | **12 / 10** (or **8** with the `serve1m` CP-int4 cache stack, load-peak-tight) |
| serve (batched) | 512k | 8 | **20 / 15** |
| serve (batched) | 1M | 8 | **32 / 20** |

- **decode-optimized (minimal nodes)** → **8 EP, `MODE=decode`** (fast Q8, TP_HEAD/EMBED); NUMA ~1.40×.
  Watch the **load-peak** (`DS4F_WARM_RSS_TRACE=1`, `DS4F_WARM_MEMAVAIL_STOP_GB`) — 8 nodes is the edge;
  fall back to **9 EP** for headroom.
- **1M-ctx serving at minimal nodes** → **8 EP, `MODE=serve1m MAXPOS=1048576`**: the CP-sharded int4
  Tier-B2 cache stack (`CP_SHARD/CP_IDX/INT4_CMP/IDX_INT4` + int8-KV) shards the long-ctx caches across
  ranks independently of the fast Q8 dense path. Feasible but tight — 9 EP is the robust choice.
- **prefill-optimized** → run **above** the floor (extra ranks = attention/expert-GEMM parallelism).

`python3 a64fx/llm/ds4f_sim.py` prints these floors; `weight_floor_tiers()` / `min_nodes(ctx,M,kvb)`
are importable.

## Node configurations — prefill / decode / prefill+decode

How to pick the node count per workload. Memory model (per node, ≤27 GB usable):
`9.02 GB replicated (dense+embed) + 150.59/N GB fp4 experts (EP) + KV/N`. Weight floor **≥9 nodes**;
staging `localtmp-size` must hold the per-node blob = `9.02 + 150.59/N` GB (11n → 22.7, 9n → 25.8;
`pjsub_ds4f.sh` sets 48Gi). Comm is ~flat in N (the per-layer MoE all-reduce is straggler-sync-bound,
not payload-bound — see the roofline / WS4), so **more nodes ≠ faster decode** — it's an efficiency and
capacity axis, not a latency one.

| pattern | node count | why | launcher |
|---|---|---|---|
| **decode-optimized** (chat, short ctx) | **at the floor: 11n** (9n possible, 11n = headroom) | fewest ranks = best efficiency; comm flat in N | `NODES=11 MODE=decode` |
| **prefill-optimized** (long prompt / RAG ingest) | **above the floor: 12–24n** | extra ranks parallelize attention + expert GEMM (prefill is compute-bound) | `NODES=16 MODE=prefill PREFILL=2048` |
| **prefill+decode serving** (full request) | **pinned by persistent KV** (min-nodes table) | KV grows with ctx×M; `DS4F_CP` shards it | `NODES=12 MODE=serve CP=1 MTP=1` |

**Decode-optimized.** Run at the floor with `--preset decode` + `DS4F_NUMA=1`. Measured base decode
10.16 tok/s @11n → ~12.8 with WS1/WS1b (mHC parallelize, landed) → target ~14 with the NUMA lever.
Elapse: ~10 tok/s means a 64-token gen ≈ 7 s; size elapse to `prefill + max_gen` tokens.

**Prefill-optimized.** Run above the floor; more ranks shorten time-to-first-token. Use
`--prefill <long> --tierb2 1 --sparse 1` (the lightning indexer's O(topk) payoff kicks in at long ctx).
Batched prefill ~56 tok/s → a 2048-token prompt ≈ 37 s. NOTE: 64 attention heads over 12 ranks is
imbalanced (6-head ranks set the attention wall); going past ~24n needs head/query subpartitioning, not
just more ranks.

**Prefill+decode serving.** Node count is set by the persistent full-ctx KV, so read it off the
min-nodes table above for the target (ctx, M). `DS4F_CP=1` (context-parallel KV) is mandatory at
ctx ≥ 512k; `DS4F_INT8_KV=1` cuts ~⅓ of the nodes; `--mtp 1` (speculative decode, α≈76%) amortizes the
per-layer straggler-sync 1/K. Example: 512k / M=8 → 20n (15n with int8-KV).

### Agentic-coding config (12-node interactive alloc = 11 EP)

The flagship interactive target: one `pjsub` **12-node** interactive alloc (shape 2×3×2), 1 node
reserved for the claude/login control process → **11 EP nodes**. Real weights, the quality-preserving
decode bundle, NUMA on, sized to fit. Launcher: **`run_ds4f_agentic_11n.sh`** (wraps
`run_ds4f_gen_11n.sh`).

```sh
# inside the live 12-node alloc, from a64fx/llm:
./run_ds4f_stage_11n.sh                                   # 1. stage once (~22.7 GB/node)
PROMPT_FILE=task.txt ./run_ds4f_agentic_11n.sh            # 2. agentic coding run (<=64k ctx)
PROMPT_FILE=task.txt KVBITS=8 MAX_NEW=1024 ./run_ds4f_agentic_11n.sh   # up to 128k ctx (int8 KV)
```

Config = `DS4F_REAL=1` + decode bundle (`FP8_BF16 Q8_DENSE TIERB2 MHC HC_PAR HC_RMSPAR`, ==
`--preset decode`) + `DS4F_NUMA=1`. **Context ceiling @11 EP** — use the EMPIRICAL column, not the
pure-memory model. `ds4f_sim.py` (weights 22.7 GB + MLA-latent KV only) says bf16 fits ~64k, but that
**omits the Tier-B2 indexer + compressed-key caches + warm-fill scratch**, which the ops log shows OOM
the alloc at ctx≈32k (safe ceiling **~16k**). ⚠️ **An OOM kills the whole allocation** (gotcha #6:
SIGKILL degrades the node's PMIx daemon → every later launch dies pre-load; recovery = recycle the
alloc + re-stage, ~21 min). **Do not probe past the safe ceiling on an alloc you want to keep.**

| KV mode | knob | safe ctx (empirical) | MLA-KV model max | note |
|---|---|---|---|---|
| bf16 (default) | `KVBITS=16` | **~16k tokens** | ~64k | validated sweet spot ~10k |
| int8 KV + cmp | `KVBITS=8` + `DS4F_INT8_CMP=1` | extends (unproven) | ~128k | halves KV *and* indexer cache |
| context-parallel | `DS4F_CP=1` (`run_ds4f_longctx_11n.sh`) | **512k–1M** | — | the validated long-ctx path, ~0.8–1.2 tok/s @1M |

Expected decode ≈ **12.8 tok/s** (WS1/WS1b landed) → **~14 tok/s** with the NUMA lever, degrading at
longer context as attention + tb2prep grow per-position. Last real-weight validation: `rc=0`, NaNs=0,
lockstep argmax, prefill 10.44 / decode 10.16 tok/s (pre-NUMA/pre-WS1) — see "Validated result".

**Larger sibling.** V4-Pro (`ds4p`: 61 L / hidden 7168 / 384 experts, ~805 GB) is a separate, much
bigger config (weight floor ~54n+) with its own harness — the numbers here are V4-Flash only.

## HTTP serving (llama-server-like) — `run_ds4f_serve_11n.sh` (2026-07)

The EP runner has a persistent **`DS4F_SERVE`** loop: load the model **once** on 11 EP nodes, then
answer many requests (instead of reloading ~3.5 min per generation). A small python HTTP frontend
(`ds4f_serve.py`) runs on the control node and drives the runner over shared-FS request/response
files — all 11 ranks read the same request → lockstep, no broadcast.

```sh
# inside the live 12-node alloc, from a64fx/llm (weights already staged):
PORT=8080 ./run_ds4f_serve_11n.sh                       # default ~16k ctx, fast decode
CTX=65536 ./run_ds4f_serve_11n.sh                       # longer single context (compressed caches)
CTX=1048576 CP=1 DS4F_SERVE_SLOTS=4 ./run_ds4f_serve_11n.sh   # extreme ctx (TP+CP) + 4 slots
# then, from anywhere that can reach the node:
curl -s localhost:8080/v1/completions -d '{"prompt":"def quicksort(a):","max_tokens":128}'
curl -s localhost:8080/v1/completions -d '{"prompt":"...","max_tokens":128,
  "temperature":0.8,"top_p":0.95,"top_k":40,"repeat_penalty":1.1,"seed":42}'   # sampling
```

Endpoints: `POST /v1/completions` (OpenAI shape), `POST /completion` (llama.cpp shape), `GET /health`.

**Features** (all validated real-weight 11n; greedy is byte-reproducible):

| Feature | Knob / request field | What it does | Measured |
|---|---|---|---|
| **Sampling** | `temperature` (≤0=greedy), `top_p`, `top_k`, `presence_penalty`, `repeat_penalty`, `seed` | temp/top-k/top-p + penalties over the last 64 tokens; shared-seed SplitMix64 → all ranks draw the same token (lockstep) | greedy deterministic; same seed reproducible; ~+48 ms/tok (full-vocab sort) |
| **Longer single context** | `CTX=<tokens>` | >16k auto-enables compressed ctx-caches (int8 KV/cmp, int4 cmp/idx) → ~2 KB/pos, keeps fast MHC decode; `CP=1` adds TP+CP sharding (→ millions) | MAXPOS=65536 arena **flat** vs 16k; decode 11.6 tok/s @ctx3246 |
| **Prefix cache** | `DS4F_SERVE_PREFIX_CACHE` (default on) | a request that EXTENDS the cached sequence skips re-prefilling the shared prefix (append only) | 246-tok turn, 237 cached → **1.94s vs 23.2s = 12× TTFT**, byte-identical |
| **Multi-slot** | `DS4F_SERVE_SLOTS=N`, request `slot` | N independent conversations; a `slot` switch snapshots the outgoing / restores the incoming caches (`ds4f_ctx_snap`) | independent + coherent; switch **~9.5 ms @256 tok** (see below) |
| **Persistent KV save/load** | `cache_path` + `cache_load` / `cache_save` | persist a context's full cache state to disk (magic + cfg-hash guard, atomic) and restore later | disk round-trip **byte-identical** to live cache |
| **System-prompt cache** | `DS4F_SERVE_SYSCACHE=path` | preload a persisted context into slot 0 at startup → every conversation starts pre-prefilled, survives restarts | 197-tok sys prompt: prefill only the user turn, **2.2s vs 17s = ~8× TTFT** |
| **Per-rank shards (CP)** | (automatic under `CP=1`) | under context-parallel the sharded caches are saved/loaded as one `<path>.rankNN` per node | 11 distinct shards, load byte-identical under TP+CP |

Build a system-prompt cache once with a `cache_save` request (`max_tokens:0` = prefill-only), then
launch with `DS4F_SERVE_SYSCACHE` pointing at it. Caveat: the on-disk cfg-hash pins the cache-mode
layout (int8/int4/MAXPOS/CP/rank) — a blob is refused (not silently corrupted) by a differently
configured runner. TTFT note: under the long-ctx (int8-KV) path prefill is token-by-token
(~12.5 tok/s), so a *cold* multi-thousand-token prompt is minutes to first token — which is exactly
what the prefix / system-prompt / KV-save caches eliminate on every subsequent turn.

**Multi-slot context-switch overhead** (`bench_slot_switch.sh`, rank-0 timer on the two snapshot
memcpys). Measured real-weight 11n at ctx 256 (reproduced across MAXPOS 8192 & 65536 — identical,
confirming the snapshot copies only the *written* region, not the arena): **save 5.37 ms + restore
4.10 ms = ~9.5 ms**, blob **29.9 MB/slot** (single-thread ~6–7 GB/s incl. the `realloc`). The blob is
a fixed part (~24 MB: int8 `kv_calbuf` CAL=256 + compressor/indexer ring states + scales) plus a
growing part (~24 KB/token: int8 `kv_q` 512 B/tok/layer×43 + int4 cmp/idx), so switch(L) ≈
24 MB + L·24 KB → ~15 ms @1k, ~37 ms @4k, ~68 ms @8k. Cheap: at ctx 256 a switch is ~2400× cheaper
than the ~23 s re-prefill it replaces, and ≈0.1 decode-token of time; only at multi-thousand-token
contexts does it reach tens of ms (still a fraction of one decode step). NOTE: the run couldn't
extend past 256 tokens — token-by-token prefills of ≳512 tok intermittently trip a uTofu collective
timeout (`tp_ar bcast timeout … want=N got=N-1` → `FATAL: barrier release`) on this alloc; the larger
rows are the linear model, not measured.

**Chunked-prefill checkpointing** (`prefill_checkpoint.sh`) — the fix for that comm timeout. Root
cause (see "uTofu comm" below): `want` in the timeout is the cumulative reduce count, and deaths
occur at wildly different counts (69 vs 514 tokens) ⇒ a **probabilistic per-Put loss with no
retransmit** (`exit(1)` after 60 s), so a token-by-token prefill firing ~43 reduces/token is a coin
flip over its ~10⁴–10⁵ reduces. The driver splits a long prompt into M-token chunks, `cache_save`-ing
after each (the prefix cache prefills only the new M tokens ⇒ M·43 reduces/chunk ⇒ low per-chunk
loss). If the runner dies mid-chunk it relaunches with `DS4F_SERVE_SYSCACHE=<ckpt>` (restores all
checkpointed chunks), and retries **only** the failed chunk — so a long prefill always makes forward
progress and a completed chunk is never redone. Reuses the byte-exact KV save/load already validated.

**KV precision — int8 vs bf16 accuracy** (`bench_kv_accuracy.sh`, greedy A/B, `DS4F_INT8_KV` the only
difference, otherwise the production config: FP8→Q8 dense, int4 cmp/idx). Real-weight 11n, 3 diverse
prompts × 64 tokens: **100 % top-1 token agreement, zero divergences** — int8 KV (S5 per-channel,
~0.2–0.5 % intrinsic latent rms) is **argmax-lossless** vs bf16 KV here, so the −0.35 GB/16k (linear
to ~5.7 GB/256k) it saves is free at the output level. (Matches the earlier gen A/B; the model's KV
cache is bf16 — there is no separate fp16 KV mode.)

## uTofu all-reduce reliability — `TP_AR_ACK` (2026-07)

**Root cause of the prefill comm timeout.** The `tp_ar bcast timeout … want=N got=N-1` death is a
**probabilistic lost Put with no retransmit**: the trailer-poll all-reduce spins 60 s then `exit(1)`,
and `want` is the cumulative reduce count — observed deaths at wildly different counts (69 vs 514
tokens) confirm it's random per-Put loss, not a fixed threshold. A token-by-token prefill fires ~43
reduces/token, so over its ~10⁴–10⁵ reduces a single loss becomes likely. (Two mitigations already
shipped: `DS4F_PREFILL_GEMM` batches ~K× fewer reduces; `prefill_checkpoint.sh` checkpoints so a death
costs only the current chunk.)

**Ack/retransmit layer** (`tp_allreduce.h`, `TP_AR_ACK=1`, **default off** → the default path is
byte-for-byte unchanged). The receiver Puts an ack into the sender's per-peer ack slot after reading
the payload; the sender retransmits the outstanding send every `TP_AR_ACK_RTT` (1 ms) up to
`TP_AR_ACK_RETX` (64) times, then proceeds optimistically (the payload is idempotent) — so a lost Put
is recovered in ~ms instead of a 60 s fatal spin. Two design points, both found by the standalone
test: **send is non-blocking** (recursive doubling has both partners send before either recvs, so a
send that blocked on the ack would deadlock/crawl), and the retransmit runs **from inside the
recv-wait loop** too (if both directions of a doubling pair drop, both ranks block in recv and only a
recv-loop retransmit — not the after-recv confirm — breaks it). Ack Puts are never drop-injected.

**Validated 11n.** Standalone `tp_ar_ack_test` (exact integer sum+max reduce, direct arithmetic check
+ `TP_AR_DROP=N` loss injection): no-drop 22 500 reduces 0-mismatch @ **21 143 reduce/s**; DROP=50 and
DROP=20 (1-in-20 heavy loss) 0-mismatch, recovers every loss (rtt sweep 20 ms→1 ms = **19× faster
recovery**, no no-loss penalty). End-to-end: the full runner with `TP_AR_ACK=1 TP_AR_DROP=50` generates
coherent, correct output at normal decode speed (8.8–10 tok/s). *Prototype caveat:* not yet hardened
against cross-call reordering under simultaneous payload+ack loss (~p⁹, astronomically rare). Note: the
Makefile doesn't track `tp_allreduce.h`, so `rm build/ds4f_ep_runner` before rebuilding to pick up
ack changes in the runner.

## Files (branch `ds4f`)

| File | Role |
|---|---|
| `common/ds4f.h` | self-contained model: config / alloc / forward / loaders / dequant matvecs |
| `a64fx/llm/ds4f_runner.c` | single-node driver (no uTofu) — validation vehicle |
| `a64fx/llm/ds4f_ep_runner.c` | 11-node uTofu EP driver (links `-ltofucom`) |
| `a64fx/llm/ds4f_stage.c` | sharded safetensors → per-node `/local/ds4f/rank<rr>.blob` stager |
| `a64fx/llm/run_ds4f_stage_11n.sh` | stage launcher (mpiexec, PMIX_RANK self-ID) |
| `a64fx/llm/run_ds4f_11n.sh` | run launcher (vcoord + topo + mpiexec) |
| `a64fx/llm/run_ds4f_longctx_11n.sh` | long-ctx Tier-B2 bench wrapper (ctx-warm + sentinel) |
| `a64fx/llm/run_ds4f_serve_11n.sh` | HTTP serving launcher (persistent runner + `ds4f_serve.py`) |
| `a64fx/llm/ds4f_serve.py` | control-node HTTP frontend (OpenAI/llama.cpp shapes) |
| `a64fx/llm/validate_prefix_cache.sh`, `validate_ctx_features.sh`, `validate_cp_shards.sh` | serving A/B validators |
| `a64fx/llm/prefill_checkpoint.sh` | chunked-prefill checkpoint driver (comm-timeout fix, resumable) |
| `a64fx/llm/bench_kv_accuracy.sh` + `bench_kv_accuracy_diff.py` | int8-vs-bf16 KV greedy accuracy A/B |
| `a64fx/llm/bench_slot_switch.sh` | multi-slot context-switch overhead timer |
| `a64fx/utofu-tests/tp_allreduce.h` | EP all-reduce (sum/max/argmax) + `TP_AR_ACK` ack/retransmit layer |
| `a64fx/utofu-tests/tp_ar_ack_test.c` | standalone loss-injection test for the ack layer (`TP_AR_DROP`) |
| `a64fx/utofu-tests/tofu_topo_helper` | MPI program that writes `tofu_topo.txt` (rank→coords) |

Build is native `fcc`/`FCC` (NOT `fccpx`/`FCCpx`); binaries run directly, **no `pjsub`**
for the run itself (this is a native A64FX node). The launchers rebuild the runner via
`make ds4f_ep_runner CC=fcc OPENMP=1`.

## Prerequisites

- A live allocation: `pjsub -L "node=12,..."` shape **2×3×2** (12 nodes; 1 is the
  claude/login node, excluded → 11 EP nodes). Confirm with `env | grep PJM_MPI_SHAPE`
  (`X=2 Y=3 Z=2`) and `echo $PJM_SUBJOBID`.
- Real weights at `$HOME/models/ds4f` (config.json + 46 safetensors shards).
- PATH must **PREPEND** `/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin` (the
  launchers do this) so `mpiclang` (for the topo helper) and `fcc` both stay resolvable.

## Step 1 — stage weights to node-local `/local`

```sh
cd a64fx/llm
DS4F_STAGE_FLUSH_GB=2 ./run_ds4f_stage_11n.sh
```

Each rank self-identifies via `PMIX_RANK` (own_mpi exports it to non-MPI binaries) and
streams **only its owned tensors** (dense → all ranks; expert `e` → rank `e%11`) file→file
from the read-only safetensors mmap into `/local/ds4f/rank<rr>.blob` + a text manifest.
RSS stays ~tens of MB (it is NOT loading weights). `DS4F_STAGE_FLUSH_GB=2` bounds the
HBM `/local` dirty-page cache (fdatasync + `fadvise(DONTNEED)` every 2 GB) — without it
the blob's dirty pages pile up in HBM (~7 GB) and OOM-segfault the stage.

Done signal: 11× `a64fx/llm/ds4f_stage_rank*.txt` on the shared FS. ~1270 s wall
(shared-FS read-bound, ~22 GB/node). The launcher prints `OK: all 11 ranks staged`.

## Step 2 — run the exact Tier-B1 forward

```sh
DS4F_REAL=1 DS4F_EXACT=1 ./run_ds4f_11n.sh
```

The launcher: regenerates `vcoord_ds4f.txt` (excludes relative `(0,0,0)`), runs
`tofu_topo_helper` → `a64fx/llm/tofu_topo.txt` for **this** alloc, builds the runner,
then `mpiexec -np 11 -vcoordfile vcoord_ds4f.txt build/ds4f_ep_runner`. Each rank mmaps
its `/local` blob, copies tensors into a ~22 GB arena (FP8 e4m3fn dense + MXFP4 expert
nibble-repack), and runs prefill 8 + decode 16. Output: per-rank
`ds4f_ep_{perf,load}_rank*.txt` + `ds4f_ep_rank00.txt` summary.

`DS4F_REAL=1` pins dense to FP8 (real e4m3fn bytes). `DS4F_EXACT=1` swaps the dense
stand-ins for the exact DeepSeek math (RoPE/YaRN, per-head q-norm, MQA sliding-window +
sink attn, grouped low-rank o-proj, sqrtsoftplus gate, swiglu clamp); `exact=0` is
byte-identical to the stand-in path.

## Step 2b — Tier-B2 (compressor/indexer) real-weight run

```sh
DS4F_REAL=1 DS4F_TIERB2=1 ./run_ds4f_11n.sh      # implies EXACT; reuses the SAME blobs
```

No re-stage needed: the compressor/indexer tensors (`layers.L.attn.compressor.*`,
`.indexer.*`) are classified `CLS_DENSE` by the stager, so they are already in every
rank's `/local` blob. `ds4f_load_real` loads them by name and widens to f32 at load
time (BF16→f32 for the compressor/`weights_proj`/indexer-compressor weights, FP8
e4m3fn + E8M0 128² dequant for `indexer.wq_b`, F32 `ape` + BF16 `norm` direct). The
forward then runs the stateful 21-CSA + 20-HCA compressor/indexer path.

Validated (job 49092345, 2026-06-05): `rc=0`, 11/11, **NaNs=0**, argmax=98198
(lockstep), RSS 21.82 GB/node. **Perf:** decode 0.56 / prefill 0.60 tok/s — attention
is 94.8 % because the f32 compressor/indexer matvecs are pure overhead at short ctx
(prefill 8 + decode 16 = 24 positions). Tier-B2 only amortizes past ~`index_topk=512`
context (its point is O(topk) long-context attention); the kernels are f32 reference
implementations, not yet SVE/quant-optimized (the speed follow-up).

## Step 2c — real-weight bf16-pv dense promote (fast decode)

```sh
DS4F_REAL=1 DS4F_FP8_BF16=1 ./run_ds4f_11n.sh    # predequant replicated dense FP8 -> bf16-pv
```

No re-stage needed (same FP8 blobs). At load, the 9 replicated dense tensors (`head`,
attn `wq_a/wq_b/wkv/wo_a/wo_b`, router `gate`, shared-expert `sh_w1/w3/w2`) are requant'd
FP8→bf16 in the pair-interleaved `BF16_PV` layout via `ds4f_load_dense`; routed experts
stay MXFP4. **FP8→bf16 is lossless** (e4m3's 3-bit mantissa × 2^k block scale fits bf16's
7 mantissa bits with no rounding) ⇒ a pure speed win, no accuracy cost. The default
(no `DS4F_FP8_BF16`) takes the same-dtype fast path and is byte-identical to the FP8 run.
Set `DS4F_BF16_PV=0` to force plain (non-pv) bf16. Costs ~+6 GB (FP8 ~12 GB dense → bf16
~18 GB); safe at the harness's short ctx, watch the budget past ~8 K (the +6 GB pushes a
128 K KV cache over HBM).

Validated (reused `/local` blobs, EXCLUDE=0,0,0, prefill 8 + decode 16): `rc=0`, 11/11,
**NaNs=0**, argmax=60574 (lockstep), RSS **27.50 GB/node** (arena 26.02 GB, fits < 32 GB
HBM2). **Perf:** decode **16.73 tok/s** (+65 % over FP8's 10.16) / prefill **17.27**
(+63 % over 10.60) — near the synthetic bf16-pv 17.49 ceiling. The profile flips to
matvec-bound (o_proj 29 % + qkv 19 % + shared 11 %); comm ~20 % is the EP all-reduce
Amdahl floor (same as synthetic). MXFP4-dense is NOT promotable from real FP8 (lossy) —
`ds4f_load_dense` aborts on an MXFP4 dense dest.

## Step 2d — long-context Tier-B2 run (the O(topk) payoff)

```sh
./run_ds4f_longctx_11n.sh                       # ctx=16384, combined bf16-pv + Tier-B2
DS4F_CTX_WARM=8192 ./run_ds4f_longctx_11n.sh    # a lower curve point
```

The wrapper sets `DS4F_CTX_WARM=N`: after the (short) prefill, `ds4f_warm_kv(m,N)` +
`ds4f_warm_tb2(m,N)` fill the synthetic KV + compressed (`cmp_kv` / `idx_kv`) caches to
`N` positions, then decode runs from `pos=N`. Both fills are **deterministic** (fixed
per-layer splitmix64 seeds) and **rank-independent** (local caches only, no all-reduce)
⇒ lockstep is preserved across all 11 ranks. This measures decode/attn cost at long
context **without** the O(ctx²) cost of a real prefill of that length. `DS4F_MAXPOS`
auto-sizes to `CTX_WARM + MAXGEN + 256` (the runner asserts `ctx_warm + maxgen ≤ max_pos`).

Validated (combined `DS4F_REAL=1 DS4F_FP8_BF16=1 DS4F_TIERB2=1`, reused blobs,
EXCLUDE=0,0,0, prefill 8 + decode 16, all rc=0 / 11/11 / NaNs=0 / argmax lockstep):

| ctx | attn ms | tb2prep ms | qkv | o_proj | decode tok/s | RSS GB | argmax |
|---|---|---|---|---|---|---|---|
| 24 (baseline) | 4.11 | 16.86 | 18.71 | 29.48 | 10.10 | 27.51 | 98198 |
| 8192 | 88.97 | 113.96 | 18.76 | 28.58 | 3.56 | 28.23 | 16 |
| 16384 | 95.96 | 159.68 | 18.85 | 28.56 | 3.00 | 28.95 | 10973 |
| 32768 | — OOM (`rc=137`) — | | | | | | |

- **O(topk) payoff (real weights):** doubling ctx 8192→16384 grew `attn` only **+7.9 %**.
  The CSA compressed-attn term is capped at `index_topk=512` in both (T = ctx/4 = 2048 →
  4096, both > 512) ⇒ CSA attn flat; the +7.9 % is the HCA(128) portion (ctx/128). Had
  attn scaled with T (no cap), ctx=16384 would be ~2800 ms (4.11 × 680) — **the cap saves
  ~29×**.
- **`tb2prep` grew +40 % per 2× ctx and was 41–48 % of decode.** *(SUPERSEDED — Step 2g
  root-caused this: the ctx-linear growth was `ds4f_index_topk`'s O(k·T) selection, not the
  `index_score` scan (1.2 %) nor the projections (~14 %). Fixed to O(T·log k); `tb2prep` is now
  ~12 % of decode. See "Current state" Step 2g.)* The scan reads T = ctx/4 compressed `idx_kv`
  positions to rank the top-512 — *activations*, immune to the topk cap and bf16 weight-quant;
  the weight-bound qkv/o_proj are ctx-flat.
- **HBM ceiling — it is the *weight* footprint, not KV.** The bf16 KV cache (Step 2e) is
  now landed and halves `kv_cache` to `max_pos · 512 · 2B · 43L`, yet **ctx=32768 still
  OOM-kills** under the gen config. The dominant resident cost is the **27.5 GB replicated
  dense weights** (inflated +6 GB by `DS4F_FP8_BF16=1`'s FP8→bf16 predequant), leaving only
  ~2.5 GB for KV+compressed caches. Halving a ~1.4 GB KV term cannot offset a 27.5 GB base.
  The real long-ctx levers are **shrinking the weights** (`DS4F_FP8_BF16=0` keeps dense FP8,
  −6 GB, slower decode) and **int8 KV** — see Step 2e and Remaining work.

## Step 2e — gen-quality + decode tok/s at 10k+ (real coding task, mHC + bf16 KV)

End-to-end greedy generation on **real** DeepSeek-V4-Flash weights, production sparse path
(`DS4F_REAL=1 DS4F_TIERB2=1 DS4F_FP8_BF16=1` + **`DS4F_MHC=1`**), via a pure-Python
byte-level BPE tokenizer (`tools/ds4f_tokenizer.py`) and gen-mode in `ds4f_ep_runner.c`
(`DS4F_PROMPT_IDS` / `DS4F_GEN_OUT` / `DS4F_MAX_NEW`, greedy argmax feedback, eos=1).

```sh
./run_ds4f_gen_11n.sh                          # built-in quicksort code-completion prompt
PROMPT_FILE=my.txt MAX_NEW=200 ./run_ds4f_gen_11n.sh
```

**`DS4F_MHC=1` is REQUIRED for coherent output** — the model ships per-layer hyper-connection
weights (`hc_attn_*`/`hc_ffn_*` + global `hc_head_*`, `hc_mult=4`, `hc_sinkhorn_iters=20`);
the forward only uses them when `m->mhc` is set. With mHC **off** the residual collapses to a
single stream → architecturally wrong → garbage tokens (the "argmax-exact vs batched
reference" check does **not** catch this; it only verifies EP-path == single-node-path). The
cheap reference-free gate is teacher-forcing next-token accuracy (`DS4F_TF_CHECK=1`):

| config | TF next-token acc | gen output |
|---|---|---|
| mHC **off** | 0/23 = 0 % | `ايره peeled… 哈哈哈` (garbage) |
| mHC **on** (`DS4F_MHC=1`) | 16/23 = **69.6 %** | correct recursive quicksort (below) |

```
def quicksort(arr):
    """Sort a list of numbers in ascending order using the quicksort algorithm."""
    if len(arr) <= 1:
        return arr
    pivot = arr[0]
    left  = [x for x in arr[1:] if x <= pivot]
    right = [x for x in arr[1:] if x >  pivot]
    return quicksort(left) + [pivot] + quicksort(right)         # ← model-generated, NaNs=0
```

**Decode tok/s at long context** (gen config: mHC + bf16 KV, prefill 8 + decode 16, all
rc=0 / 11/11 / NaNs=0 / argmax lockstep). These are the *heavy* (mHC-on) numbers — the
Step 2d O(topk) table above is the lighter mHC-off sparse-path characterization:

| ctx | decode tok/s | tb2prep | attn | RSS GB | arena GB/node | status |
|---|---|---|---|---|---|---|
| 24 (gen, real prompt) | 5.93 | — | — | 27.51 | — | coherent quicksort, mHC |
| 10240 | **2.99** | 38 % (126 ms) | 27 % (91 ms) | 27.96 | 26.11 | rc=0 |
| 16384 | **2.68** | 43 % (160 ms) | 27 % | 28.22 | 26.36 | rc=0 |
| 32768 | — | | | | | **OOM-kill** (sig=9 during warm-fill) |
| 65536 | — | | | | | died pre-load (PMIx — 32768-OOM aftermath, see OPS note) |

- Real-model decode *was* **~2.7–3 tok/s at 10–16K ctx** with `tb2prep` dominant (38–43 %).
  *(SUPERSEDED — Step 2g: `tb2prep`'s cost was `ds4f_index_topk` (O(k·T)), not the `index_score`
  scan. Fixed → **5.21 tok/s @ctx10240**, `tb2prep` ~12 %; **attn (48 %) is now dominant**. See
  "Current state" Step 2g.)* Both `tb2prep` and `attn` read *activations*, so weight-quant
  can't help them; int8 KV would cut `attn`.
- **bf16 KV did not raise the ctx ceiling much** — the blocker is the 27.5 GB weight base
  (see the HBM note above), not KV precision. ctx ≤ ~16K is the safe top under
  `DS4F_FP8_BF16=1`; reaching 32K needs `DS4F_FP8_BF16=0` (FP8 dense, −6 GB) and/or int8 KV.

## Step 2f — single-node decode roofline + FP8 prefill tile-dequant (alloc-free)

When the 11-node alloc is PMIx-degraded, this single-node, pthread-pinned bench
(`a64fx/llm/ds4f_decode_bw_bench.c`, `make ds4f_decode_bw`, runs directly on-node, pins
cores 12–59, leaves 0–11 + memory for the agent) rooflines every weight rep in the M=1
cold-HBM decode regime. Two confounds in the old `ds4f_fp8_fast_bench.c` were fatal and
are fixed here: **NUMA first-touch** (fill *through* the pool via `mmap`/`munmap` per pool
— glibc `aligned_alloc` recycles already-faulted pages, pinning the whole pool to CMG0 and
collapsing the read ceiling to single-CMG ~90 GB/s) and **cold cache** (a pool of P
distinct matrices cycled per sweep). Trust shapes ≥16 MB only; smaller ones are
barrier/L2-overhead-skewed. Gate = argmax+top5 vs an own-repr f32 reference (never bit-eq).

**Node read ceiling R ≈ 720 GB/s** (1T 42 → 12T 466 → 24T 714 → 48T 719). The decisive,
non-obvious result (it **inverts** the "FP8 = faster *and* −6 GB" premise):

| 48T, ≥16 MB shapes | bf16 | fp8-gather | fp8-magic | scalar | neon |
|---|---|---|---|---|---|
| wo_a[8192,4096] 32 MB | **610** (85 % R) | 64 | 106 | 23 | 34 |
| wo_b[4096,4096] 16 MB | **579** (93 % R) | 103 | 128 | 20 | 32 |
| bigK[4096,8192] 32 MB | **307** (42 % R) | 85 | 73 | 18 | 32 |
| wq_b[32768,1024] 32 MB | **373** (52 % R) | 50 | 56 | 13 | 27 |

- **FP8 on-demand decode is dequant/issue-bound, not BW-bound** (8–20 % of R, still scaling
  past 24T); **bf16-predequant is BW-bound** (~85 % of R) and **2.1–3.3× faster per dense
  matvec** despite 2× the bytes. On A64FX, FP8 trades 2× memory for 2–3× slower decode —
  speed vs memory is a fundamental tension, not a free win.
- **magic ≥ gather** within FP8 on the dominant dense shapes (wo_a 1.66×, wo_b 1.24×);
  gather wins only large-K (bigK 0.85×). magic flushes E4M3 subnormals via FTZ → argmax-exact
  on synthetic, but a real-weight argmax check gates flipping `DS4F_FP8_MAGIC`'s default.
- **scalar/NEON-128 refuted** for the matvec (3–7× slower than SVE-512) — the "free cores in
  decode → short-vector lower latency" idea loses; the 48T matvec is decode-op-throughput bound.
- **double-buffer (decode∥GEMM) refuted by the roofline** (no prototype built). The premise
  "cores are free in decode" is false — the matvec already runs on all 48 compute cores, so a
  producer/consumer split *repartitions* them rather than adding any. The wall is FP/SIMD-pipe
  issue of the dequant *arithmetic* (magic, the fast path, is FLA-bound — the load-pipe gather
  is slower), a resource **shared** by dequant and FMA: partitioning rows across cores can't
  raise aggregate FLA throughput, and magic still *scales* with cores at 48T (not plateaued),
  so uniform all-core use already beats reserving a specialized producer set. A split would only
  win if dequant were load-bound and FMA FLA-bound (disjoint pipes), but a single core's OoO
  already overlaps the gather (load pipe) with the FMA (FLA pipe). Net: neutral-to-negative.

**Landed (validated single-node, argmax-exact): FP8→bf16 FUSED tile-dequant prefill GEMM.**
`ds4f_gemm_worker` (`common/ds4f_impl.h`) now batches **FP8 dense** (the memory-lean default)
instead of falling back to per-token matvec. The **fused** kernel: for each 8-row group it
dequants one `TILE_K`-wide FP8 sub-tile (SVE LUT-gather + per-128-block E8M0 scale + `>>16`,
bit-identical to the load-time predequant) **into a tiny 8 KB L1 PV pair-buffer** (`uzp1`+`zip1`
interleave → the same `p_odd` layout the bf16 path uses), then immediately consumes it across all
M tokens with the peak `matvec_bf16_8x3_pv_acc` microkernel, accumulating over K tiles. **No bf16
HBM round-trip** and it reuses the exact bf16 microkernel — so the dequant hides behind the FMA
pipeline. `DS4F_EXACT=1 DS4F_FP8_BF16=0 DS4F_PREFILL_BATCH=64 DS4F_PREFILL_CHECK=1` → argmax
**0/64** mismatch, rel 8.5e-6 (K-tile reassoc; FP8→bf16 itself lossless).

*Evolution of this lever:* the first version expanded the whole row-slice to a bf16 scratch then
ran the plain `gemm_bf16_f32_tokmajor` — two penalties (bf16 HBM/L2 round-trip + the slower
non-PV kernel) capped it at **~53% of the +6 GB bf16-resident path** (18.4 vs 39 tok/s). Fusing
the dequant into the K-tile and switching to the PV 8×3 kernel removed both: **FP8 batched
prefill is now ~88% of bf16-resident** (A/B at L=16, P=384: fused-FP8 ~85–92 vs bf16-resident
~96–106 tok/s; equal within noise at L=8) — **with no +6 GB**. Decode (M=1) path untouched. The
residual ~12% is the gather-bound dequant compute not 100% hidden; closing it would need the
gather-free *magic* decode (real-weight-argmax-validation-blocked, FTZ subnormals).

## OPS gotchas (each cost a real cycle)

1. **A job restart wipes `/local` AND moves the alloc.** A new `pjsub`/restart →
   new `$PJM_SUBJOBID` →
   - `/local` is cleared → the staged blobs are GONE → `DS4F_REAL=1` opens a missing
     `/local/ds4f/rank<rr>.blob` → fast failure/segfault. **Re-run Step 1.**
   - the physical Tofu group changes (e.g. `12 19 15` → `7 19 0`) → any saved
     `tofu_topo.txt` is stale → `mpiexec` can't place ranks on nonexistent coords →
     instant `rc=1` (~2 s, "exceed limit on virtual coordinate"). **Never pass
     `SKIP_TOPO=1` across jobs** — let the launcher regenerate.

2. **There are two `tofu_topo.txt` files.** The run launcher reads the one in its CWD
   (`a64fx/llm/tofu_topo.txt`). Running the helper from `a64fx/utofu-tests/` writes a
   *different* file and does not update what the runner reads. Pre-verify from `llm/`
   or just trust the launcher's own regeneration (the default).

3. **Excluding the claude node.** claude's interactive node is the alloc node whose
   intra-group coord is `a=b=c=0`. Probe it: `utofu_query_my_coords()` prints
   `x y z 0 0 0`. own_mpi maps that to vcoord `(0,0,0)`, so `EXCLUDE=0,0,0` (the
   launcher default) drops it. **Verify** after topo regen: the placed
   `tofu_topo.txt` must NOT contain the `x y z 0 0 0` row.

   ```sh
   # one-shot probe of this node's coord (compile once):
   cat > /tmp/mycoord.c <<'EOF'
   #include <stdio.h>
   #include <utofu.h>
   #ifndef TOFU_NCOORDS
   #define TOFU_NCOORDS 6
   #endif
   int main(void){uint8_t c[TOFU_NCOORDS]={0};
     if(utofu_query_my_coords(c))return 1;
     printf("%u %u %u %u %u %u\n",c[0],c[1],c[2],c[3],c[4],c[5]);return 0;}
   EOF
   fcc -Nclang -O3 -march=armv8.2-a+sve -o /tmp/mycoord /tmp/mycoord.c -ltofucom && /tmp/mycoord
   # -> e.g. "7 19 0 0 0 0"; then after the run, confirm:
   grep -c "7 19 0 0 0 0" a64fx/llm/tofu_topo.txt   # must print 0
   ```
   Running the heavy ~22 GB load on the claude node OOM-kills the claude session (shared
   cgroup). Staging on it is safe (file→file, bounded RSS) but pointless.

4. **Orphaned `org/mpiexec` (PPID=1)** from a killed run holds ranks → later
   "exceed limit on virtual coordinate". Check `pgrep -af 'org/mpiexec|ds4f_ep_runner'`
   and kill stale ones (use a pattern that does NOT self-match the killing shell).

5. **stdout is not forwarded** from `mpiexec`'d ranks — only file writes survive. Wait on
   the 11× per-rank files, NOT a stdout banner.

6. **An OOM-kill (sig=9) degrades the remote PMIx service.** When a rank is `SIGKILL`ed
   (HBM OOM during warm-fill, e.g. ctx=32768), it cannot finalize MPI cleanly, leaving the
   `plexec`/PMIx daemon on that node in a bad state. **Every subsequent launch then dies
   pre-load in ~3 s** with `[ERR.] PLE 0080 plexec PMIx service error occurred.(nid=…)` and
   `rc=255`, `perf=0/11 load=0/11` — looks like a fresh crash but is aftermath. A short wait
   (45 s, even 180 s) does **not** reliably clear it; recovery usually requires recycling the
   alloc (new `pjsub` → `/local` wiped → re-stage Step 1, ~21 min). **Practical rule: don't
   probe ctx past the known ceiling (~16K) on a shared alloc you don't want to lose** — one
   OOM costs the whole allocation. The longctx wrapper now flags this (`rc≠0` with
   `load=0/11` ⇒ pre-load failure) instead of reporting `crash_hits=0`.

## Output discipline (keep agent context small)

Chain stage→run **detached** (`setsid nohup …`), funnel all output to log files, and
read only a compact sentinel — never stream the 11× per-rank dumps into the agent. A
single background waiter that polls the sentinel for an end marker (e.g. `RUN2_END`)
beats echo-ping polling. Pattern: write `STAGE done=N/11`, `RUN rc=…`, topo group,
crash grep, and the ~10-line `ds4f_ep_rank00.txt` to one `/tmp/*sentinel.txt`.

## Validated result (job 49092345, 2026-06-05)

```
STAGE done=11/11  wall=1272 s  (no OOM — bounded /local cache held)
RUN   rc=0  11/11 perf + 11/11 load  topo group 7 19 0 (claude excluded)
NaNs=0  argmax=122293  (deterministic; identical to the first validated run)
prefill  8 tok   95.8 ms/tok  10.44 tok/s  comm 13.3%  ar_calls=344
decode  16 tok   98.4 ms/tok  10.16 tok/s  comm 12.4%  72.9 GB/s-weights
arena 22.18 GB   RSS 21.81 GB / node   (fits ~25 GB usable HBM2)
per-phase decode: o_proj 33.4% + qkv 27.8% + shared 14.1% = ~75% replicated dense FP8;
                  experts 5.2%, attn 3.9%, head 2.2%, router 0.6%.
```

The dominator is replicated dense FP8 gather (same finding as the synthetic Stage 2/3
and the EP-runner). `DS4F_FP8_BF16=1` (Step 2c) promotes that dense to bf16-pv at load
and lifts real-weight decode to **16.73 tok/s** (+65 %); the remaining lever is batched
decode M>1 to amortize the 43 per-layer all-reduces.

## Flag reference (env)

| Flag | Default | Effect |
|---|---|---|
| `NP` | 11 | rank count |
| `EXCLUDE` | `0,0,0` | relative vcoord to drop (claude node); `none` = whole alloc (DANGER) |
| `DS4F_REAL` | 0 | 1 = load real weights from `/local` blobs (else synthetic fill) |
| `DS4F_EXACT` | 0 | 1 = exact dense Tier-B1 math (else shape-correct stand-in) |
| `DS4F_TIERB2` | 0 | 1 = stateful compressor/indexer (CSA/HCA) decode; implies `EXACT`. Real weights load by name (validated, job 49092345) |
| `DS4F_FP8_BF16` | 0 | 1 = predequant replicated dense FP8→bf16 at load (faster decode, +~6 GB); lossless (Step 2c, real-weight validated). Default = byte-identical FP8 |
| `DS4F_BF16_PV` | (auto) | with `DS4F_FP8_BF16=1`: empty = pair-interleaved pv (fastest); `0` = plain bf16; `1` = force pv |
| `DS4F_MXFP4_GEMM_TILE` | 0 (auto→16 in batched prefill) | M-threshold ≥ which the MXFP4 (expert/dense) **prefill** GEMM tile-dequants nibbles→bf16 once and reuses across M (1.38–1.72× @M≥16, lossless); `0` = off (svtbl per-pair, best at M≤4). **Auto-set to 16 when batched prefill is active (`ds4f_ep_runner.c`); explicit value overrides. Real-weight 11n: +7.9% prefill, token-exact (Step 2o)** |
| `DS4F_MHC` | 0 | 1 = exact manifold-constrained hyper-connections (`hc_mult=4`). **REQUIRED for coherent real generation** (Step 2e); gen wrapper defaults it on |
| `DS4F_PREFILL_GEMM` / `DS4F_PREFILL_K` | 0 / 32 | 1 = batch gen prefill through the MHC+Tier-B2 verify path in chunks of K≤32 (comm ÷K, dense→M=K GEMM, **indexer qproj batched to 1 GEMM**). **+65% prefill** (13.3→21.9 tok/s), COHERENT not bit-identical (GEMM reassoc); opt-in, not-with-MTP. Prints tb2+attn breakdown under `DS4F_PROF`. See Decode-perf wins |
| `DS4F_FLAGBAR` | **1** | per-worker cache-line completion-flag pool barrier (vs the shared-counter barrier's 47-way ping-pong). **Bit-identical, +8% decode & prefill** (M=1 fires ~900 tiny dispatches/tok). See Decode-perf wins |
| `DS4F_ATTN_GEMM` | **1** | 8-head-blocked Tier-B2 decode attention: each MLA `kv[j]` loaded once, reused across 8 query heads (score + value). **Bit-identical, attn −50%, +6.6% decode (+8.3% @5k, grows w/ctx)**. Falls back to per-head for exotic quant (int8/int4 KV/cmp, CP) |
| `DS4F_IDX_GEMM` | **1** | 8-index-head-blocked indexer scan (`idxsc8`) — ILP hides the per-head svaddv latency. Bit-identical, +16% on `tb2scan` (O(T), compounds at long-ctx CP) |
| `DS4F_IDX_INT8W` | 0 | **LOSSY, opt-in**: int8 W8A8 the indexer q-projection weight (`idx_wq_b`, K=1024, byte-bound). ~2× on `tb2qproj`, **+4.1% decode @5k**. q drives only top-k *selection* (error-tolerant); quality-gated (coherent, NaN=0). **Only helps the small qproj matvec — NOT the compressor (K=4096, not byte-bound)** |
| `DS4F_PROMPT_IDS` / `DS4F_GEN_OUT` / `DS4F_MAX_NEW` | — | gen-mode (Step 2e): prompt id file in, generated id file out, greedy decode budget (stops on eos=1) |
| `DS4F_TF_CHECK` | 0 | 1 = teacher-forcing next-token accuracy gate (~0 % = broken forward, ~50–80 % = working); the cheap reference-free correctness check |
| `DS4F_STAGE_FLUSH_GB` | 2 | stager HBM dirty-cache flush granularity |
| `DS4F_STAGE_DIR` | `/local/ds4f` | per-node blob dir |
| `DS4F_PREFILL` / `DS4F_MAXGEN` | 8 / 16 | prefill / decode token counts |
| `DS4F_CTX_WARM` | 0 | >0 = warm synthetic KV+compressed caches to this ctx, decode from there (long-ctx bench, Step 2d); deterministic + rank-independent (lockstep-safe) |
| `DS4F_MAXPOS` | 4096 | KV/compressed cache capacity; must exceed `CTX_WARM+MAXGEN`. KV is **bf16** (Step 2e); ctx ≤ ~16K fits, 32768 OOMs (weight-footprint-bound, not KV) |
| `DS4F_LAYERS` | 0 (=43) | layer count (small = smoke) |
| `LLM_THREADS` | 48 | OpenMP threads/node |
| `SKIP_TOPO` | unset | 1 = reuse existing `tofu_topo.txt` (**never across jobs**) |

## Current state (2026-06-07)

Everything below is **DONE + validated** on the 11 EP nodes (real DeepSeek-V4-Flash
weights), committed on branch `ds4f` (latest `8f38fb7`) except where noted:

- **Tier-B1 exact dense forward** (Step 2): RoPE/YaRN, per-head q-norm, MQA sliding-window
  + sink attn, grouped low-rank o-proj, sqrtsoftplus gate, swiglu clamp — argmax=122293,
  10.16 tok/s FP8 dense.
- **Tier-B2 compressor/indexer** (Step 2b): stateful 21-CSA + 20-HCA decode path, real
  weights loaded by name; bf16 weight-quant (`037a2d1`, −37 % tb2prep, argmax-exact);
  pooled+SVE compressor/indexer kernels (`f648b79`, 14× over the f32-reference path).
- **bf16-pv dense promote** (Step 2c): lossless FP8→bf16-pv at load, decode 16.73 tok/s
  (+65 %).
- **Long-ctx Tier-B2** (Step 2d): O(topk) payoff confirmed — attn +7.9 % per 2× ctx
  (capped at `index_topk=512`); the index *scan* is the new long-ctx ceiling. ctx-warm
  infra (`ds4f_ep_runner.c` `DS4F_CTX_WARM` + `run_ds4f_longctx_11n.sh`) is the only
  **uncommitted** piece as of this writing → committed alongside this doc.
- **Gen-quality + decode tok/s** (Step 2e): real coding-task generation produces coherent
  code (recursive quicksort) with **`DS4F_MHC=1`** (the key finding — mHC is required;
  without it, garbage). bf16 KV cache landed (halves KV). Measured decode **2.99 tok/s @
  ctx10240, 2.68 @ ctx16384** (mHC path; below the 10–15 target — `tb2prep`/attn bound).
  Two bugs fixed en route: the **`s_tb2_sel` heap overflow** (`index_topk=512 > max_pos`
  corrupted compressor ring-state → NaNs; size to `max(max_pos, index_topk)`), and the
  mHC-off architecture gap. Gen-mode + tokenizer + bf16 KV + these fixes are **uncommitted**
  as of this writing.
- **Single-node decode roofline + FUSED FP8→bf16 prefill GEMM** (Step 2f, `affb141`/`7426330`):
  `ds4f_decode_bw_bench.c` rooflines every weight rep cold-HBM (node R≈720 GB/s) — bf16 is
  BW-bound (~85 % R) and 2–3× faster/matvec than dequant-bound FP8; scalar/NEON and
  double-buffer **refuted**. The fused tile-dequant GEMM makes `DS4F_FP8_BF16=0` prefill ~88 %
  of the +6 GB bf16-resident speed (kills the +6 GB without losing prefill).
- **int8/SVE `index_score` scan** (`ae1167d`): resident per-position int8 (`DS4F_IDX_INT8`,
  default off). **argmax-exact (96/96) on real weights**, f32 path bit-identical, but a
  confirmed **negative for ≤16k decode** (the `DS4F_P_TB2SCAN` sub-timer proved the scan is
  1.2 % of decode, not the bottleneck).
- **Fast `index_topk` — the real decode bottleneck, FIXED (Step 2g, uncommitted).** Full
  per-component `tb2_prepare` profiling (`DS4F_P_TB2{QPROJ,ROPE,ICMP,WPROJ,LCMP,TOPK}` sub-timers)
  found the actual hot line: of `tb2prep`'s 126 ms @ctx10240, the projections/compressor sum to
  only **~17.7 ms** — the remaining **108.7 ms (86 % of `tb2prep`, 37 % of decode) is
  `ds4f_index_topk`**, the O(k·T) repeated-linear-scan selection (+ O(k²) sort). Replaced with an
  O(T·log k) **size-`npick` min-heap → τ threshold → merge {score>τ} ∪ lowest-index {score==τ}**
  that produces the *identical* selected set & order. **Result: decode 3.38 → 5.21 tok/s
  @ctx10240 (+54 %, 1.54×); `tb2topk` 108.7 → 4.19 ms (25.9×).** Validated: `tools/topk_equiv_test.c`
  432 adversarial trials (T≤4096, k=512, heavy ties, all-equal, relu-sparse, thr<T) **diff-empty**
  vs naive; 11-node real-weight ctx-warm A/B **argmax-exact 223/81819 fast==naive**, NaNs=0, RSS
  unchanged, lockstep held. New default; `DS4F_TOPK_NAIVE=1` keeps the O(k·T) reference. Scales
  BETTER with ctx (was the steepest ctx-linear decode term).
- **SVE decode-attn — attn 6.6×, decode +68 % (Step 2h, uncommitted).** With `index_topk` fixed,
  `attn` was the dominant decode phase (91 ms = 48 % @ctx10240). The two attn workers
  (`ds4f_attn_tb2_worker`, `ds4f_attn_exact_worker`) were four **scalar** loops — QK dot + PV axpy
  over `kv_lora=q_head_dim=512`, with per-element bf16 decode — running at ~0.2 GFLOP/s/core
  (scalar-issue/decode-bound, NOT bandwidth-bound: window=128 pos ~128 KB + ≤512 compressed pos
  ~1 MB, both L2-resident — so the earlier "int8 KV would cut attn" guess was wrong; the lever is
  vectorization). SVE-vectorized with the `svld1uh<<16` bf16-widen idiom (BIT-EXACT to `ds4f_bf16f`)
  + `svmla`/`svaddv`. The PV axpy keeps the j-outer loop so per-lane accumulation order matches
  scalar (bit-exact PV); only the QK dot's horizontal reduction reorders (tiny f32). **Result:
  attn 91.1 → 13.8 ms (6.6×); decode 5.21 → 8.75 tok/s @ctx10240 (+68 %, 1.68×)** — total session
  decode **3.38 → 8.75 tok/s (2.59×)**. Gated `DS4F_ATTN_SVE` (default 1; `=0` = scalar reference).
  *Validation subtlety (recorded as a lesson):* the SYNTHETIC ctx-warm argmax FLIPPED (223→294)
  under the reorder — but that is a near-tie over *junk* latents, **not** a regression. The real
  gate is real-weight gen + `DS4F_TF_CHECK=1`: SVE vs scalar gave **identical TF pred sequence
  (23/23) and identical 48-token greedy completion (diff-empty), TF_ACCURACY 69.6 % both** (the
  known-good coherent-gen baseline). So: synthetic-warm argmax is a SPEED signal only; gate any
  fp-reorder change on real-weight TF/gen, never synthetic argmax.

- **Fused block-diagonal o-proj — o_proj 2.47×, decode +20 % (Step 2i, committed).** With `attn`
  fixed, `o_proj` was the dominant decode phase (28.8 ms = 26 % @ctx10240). The grouped low-rank
  o-proj ran the block-diagonal `wo_a` as **`o_groups`=8 SEPARATE `ds4f_matvec` dispatches**, each
  `[o_lora=1024 × 4096]`. Two compounding costs: (1) 8 cross-CMG pool barriers/layer × 43 layers;
  (2) **cross-CMG NUMA reads** — `ds4f_fill`/`ds4f_promote_worker` first-touch `wo_a` via
  `rowsplit8(8192,…)` (thread *t* owns rows ~`[t·170,…]`), but each per-group matvec used
  `rowsplit8(1024,…)` so thread *t* read rows `g·1024 + [its 1024-slice]` — first-touched on a
  *different* CMG → remote HBM. Fused all 8 groups into ONE `ds4f_matvec_blockdiag` pool dispatch
  (`ds4f_mv_bd_worker`): split the full `o_inter=8192` rows once via `rowsplit8` (matches the
  first-touch → NUMA-local; well-balanced ~170 rows/thread), and pick the per-8-row-block input
  `s_attn + (i/o_lora)·gin` (8-row blocks never straddle a group since `o_lora%8==0`). **BIT-EXACT**
  (same kernel, same per-row dot/accum order; full-tensor + global row index reproduces
  `ds4f_row_slice`'s byte/scale offsets — PV `(i/8)·8K`, FP8/MXFP4 `i`-indexed). **Result: o_proj
  28.8 → 11.7 ms (2.47×, −17 ms); decode 8.72 → 10.47 tok/s @ctx10240 (+20 %)** — crosses the
  10 tok/s target floor; total session decode **3.38 → 10.47 tok/s (3.10×)**. Gated `DS4F_OPROJ_FUSE`
  (default 1; `=0` = per-group reference). *Validation:* ctx-warm argmax **identical** (294/90465,
  fuse==ref — bit-exact, no flip) + real-weight 64-tok gen A/B **token-identical** (diff-empty),
  NaNs=0, lockstep held, RSS unchanged. (Generalizable: any matvec whose per-call `rowsplit8` row
  count ≠ the tensor's full row count reads cross-CMG — fuse to the full-tensor split.)

- **Parallel q-norm+RoPE — decode +7.6 % (Step 2j, committed).** Splitting `qkv_proj` (18.6 ms)
  with per-matvec sub-timers (`DS4F_P_QKV_A/B/KV/ROPE`, NPHASE 16→20) revealed the surprise: it is
  NOT all matvec — `wq_b` matvec is 8.4 ms but **`ds4f_q_norm_rope` is 7.18 ms = 7.7 % of decode**
  (wq_a 0.97, wkv 0.72). That function ran **fully serial on tid0** (between the wq_b/wkv pool
  dispatches): a scalar `double`-accum dot + scale + RoPE over each of `n_heads`=64 heads × HD=512,
  × 43 layers — scalar-issue-bound, like the pre-2h attn loops. The heads are **independent and
  write disjoint `qh` slices**, so splitting them across the 48-core pool (`ds4f_qnr_worker`, head
  range `[nh·tid/nthr, nh·(tid+1)/nthr)`) is **BIT-EXACT** to the serial loop (identical per-head
  accumulation order/scale/RoPE — no fp reorder, unlike 2h). **Result: q-norm+RoPE 7.18 → 0.48 ms
  (15×); qkv_proj 18.6 → 12.0 ms; decode 10.47 → 11.29 tok/s @ctx10240 (+7.6 %)**; total session
  decode **3.38 → 11.29 tok/s (3.34×)**. Gated `DS4F_QNR_PAR` (default 1; `=0` = serial reference).
  *Validation:* ctx-warm argmax **identical** (294/90465 — bit-exact) + real-weight 64-tok gen A/B
  **token-identical** (64/64, diff-empty), NaNs=0, lockstep held, RSS unchanged. (Generalizable:
  serial scalar per-head/per-row activation loops between pool dispatches are a recurring decode
  lever — parallelize the independent units across the pool, bit-exact when units are disjoint.)

- **Parallel indexer RoPE+rotate+fp4 (tb2rope) — decode +5.2 % (Step 2k, committed).** Applying the
  2j pattern to the biggest `tb2prep` sub-op: `ds4f_index_step`'s per-head `RoPE + rotate_activation
  + fp4_act_quant_inplace` loop ran **fully serial on tid0** (between the qproj and weights pool
  dispatches) — `tb2rope` = 4.68 ms/tok = 5.0 % of decode @ctx10240. Each of the `index_heads`=64
  heads writes a disjoint `q_scr[h·hd..]` slice and all three ops touch only that slice (+ read-only
  rcos/rsin), so splitting heads across the pool (`ds4f_tb2rope_worker`) is **BIT-EXACT**. **Result:
  tb2rope 4.68 → 0.24 ms (19×); tb2prep 21.9 → 17.4 ms; decode 11.29 → 11.88 tok/s @ctx10240
  (+5.2 %)**; total session decode **3.38 → 11.88 tok/s (3.51×)**. Gated `DS4F_TB2ROPE_PAR` (default
  1; `=0` = serial reference). *Validation:* ctx-warm argmax **identical** (294/90465) + real-weight
  64-tok gen A/B **token-identical** (64/64, diff-empty), NaNs=0, lockstep held, RSS unchanged.
  (The other `tb2prep` sub-ops — lcmp/topk/scan/qproj — are already pooled or already-optimized, so
  this was the one remaining pure-serial sub-op; tb2prep's residue is now matvec/scan-bound.)

- **int8 W8A8 dense — decode +9.7 % AND −5.5 GB RSS (Step 2l, committed `99b76ad`).** The ONLY lever
  so far that cuts **both** speed and memory (every prior lever was speed-only). `DS4F_Q8_DENSE=1`
  (requires `DS4F_FP8_BF16=1`) repacks the 8 dominant dense bf16-pv tensors/layer (`wq_a`, `wq_b`,
  `wkv`, `wo_a`, `wo_b`, `sh_w1`, `sh_w3`, `sh_w2` = 344 total) → **int8 W8A8** (`matvec_sdot_8row`,
  `svdot`, ~1.03 B/elem from HBM = **half** of bf16's 2 B) with a **per-64-block absmax scale**
  (528 B/group = 16 B fp16 scales + 8×64 int8 — finer than per-row, preserves argmax fidelity).
  After repack the bf16 source is reclaimed via `MADV_DONTNEED`, so resident memory **shrinks**.
  Gate + head stay bf16-pv (argmax protection). **Result @ctx10240: decode 11.88 → 13.03 tok/s
  (+9.7 %); RSS 27.95 → 22.40 GB (−5.5 GB); dense weight traffic 12.82 → 7.34 GB/tok.** Dense phases
  ~halved: qkv_proj 11.97 → 7.50, o_proj 11.56 → 9.40, shared 7.19 → 6.13, `wq_b` 8.43 → 4.04. Total
  session decode **3.38 → 13.03 tok/s (3.85×)**. *Validation:* real-weight gen A/B (quicksort, greedy,
  `DS4F_MHC=1`, 48 new tok) is **TOKEN-IDENTICAL to the bf16 baseline** — prefill argmax 361, final
  74433 in both; the per-64-block scale preserves the greedy trajectory exactly (better than the
  "merely coherent" bar a lossy int8 transform would predict), NaNs=0, lockstep held. *Two bugs found
  + fixed:* (a) `DS4F_Q8_DENSE` was read only in `ds4f_alloc_synth`, never in `ds4f_load_real` →
  `m->q8_dense` stayed 0 and the promote was a silent no-op (mirror the env read into the real path);
  (b) the Step-2i fused block-diagonal o-proj worker `ds4f_mv_bd_worker` had **no `DS4F_Q8_PV`
  branch** → with int8 `wo_a`, `dst` (s_o1) was never written (0 on synth, NaN on real) — added a
  Q8_PV branch that re-quantizes the per-group block-diagonal x only when the group changes
  (`glora%8==0` ⇒ a quantize lands on a 64-block boundary), bit-exact to the per-group `ds4f_matvec`
  Q8 path (same `xq`/`xs`, same `(i/8)*gb` weight offset). *Latent bug (documented, non-default):* the
  per-group o-proj fallback (`DS4F_OPROJ_FUSE=0`) is broken for Q8 — `ds4f_row_slice` can't slice the
  528 B int8 group layout; only the fused path (global `(i/8)*gb` indexing) is correct. **The −5.5 GB
  directly attacks the ctx=32768 OOM ceiling** (the resident dense weights, not KV, are the wall —
  see Remaining-work item 2) and inverts the FP8-dense memory-vs-speed tradeoff for these 8 tensors
  (FP8-on-demand was −6 GB but *slower*; int8 W8A8 is −5.5 GB **and** faster). Gated `DS4F_Q8_DENSE`
  (default off in the base 11n runner — int8 is lossy so the synthetic argmax shifts, keep it the
  clean bf16-pv reference; default **on** in the perf/memory wrappers longctx + gen, commit `36fcf48`).
  *Two int8 RSS data points (safe band): ctx10240 = 22.40 GB (decode 13.03 tok/s), ctx16384 = 22.66 GB
  (decode 12.34 tok/s) → KV slope ~0.26 GB/6144 ctx (~0.35 GB/8192).* **PROJECTED ctx-ceiling lift:**
  bf16's 32768 OOM (rc=137, projected ~30.4 GB vs ~29–30 GB node ceiling) is a *weight*-footprint
  wall, not KV. With int8 dense, ctx32768 projects to **~23.4–24.4 GB (5–6 GB headroom)** even on the
  conservative bf16 slope, so **int8 dense alone should clear 32768 and likely reach ~64–128k** —
  a 4–8× context jump, not just a decode-speed win. *Confirm on a FRESH alloc only* (the standing
  rule forbids probing past 16k on a shared alloc — an OOM-kill degrades PMIx and loses the alloc).

- **int8 KV cache — −0.35 GB RSS @16k (scales linearly), real-weight token-identical (Step 2m, `DS4F_INT8_KV`, commit `0d8b6d1`).** The KV-memory half of the 256k push (pairs with 2l's int8 *dense*). `DS4F_INT8_KV=1` stores each `(layer,pos)` window KV latent `[kv_lora=512]` as **int8** (1 B vs bf16's 2 B) with a **STATIC per-channel scale** calibrated on the first `DS4F_INT8KV_CAL` positions (default 256; S5 scheme, de-risked in `tools/int8kv_probe.c`, L1-rel 0.0032). The latent's massive-activation sink channels (~1e3–1e5) are positionally *consistent*, so early calibration holds — naive per-token int8 is catastrophic (one sink dim sets the scale → O(1) dims vanish). **Streaming design (no straddle):** positions `[0,CAL)` stage to a small bf16 `kv_calbuf` while per-channel absmax accumulates; at the first `pos≥CAL` the scale **freezes**, the calbuf is quantized into the int8 store `kv_q`, and reads switch to int8 — freeze happens *between* `forward_token` steps, so at any instant every written position is wholly in calbuf (unfrozen) **or** wholly in kv_q (frozen), and the read path is a single hoisted branch (`kvbf = !frozen ? kv_calbuf : kv_cache`; `i8 = int8_kv && frozen`). Arena stays bf16-sized (`MAP_NORESERVE` ⇒ RSS = touched pages ⇒ int8 realizes the win). **Scope:** `int8_kv` forces `m->exact=1` and aborts batched prefill, so only the **two** streaming exact/tb2 attn read workers needed int8 branches (the synthetic/batched-prefill workers never run under it) — far less than the "~15 sites" the table once feared. *Validation (real weights, 11n, `DS4F_MHC=1`):* (a) **coherence** — gen with `CAL=64` (freeze fires ~gen-token 40, ~96 tokens read the frozen int8 window) is **TOKEN-IDENTICAL** to the bf16 baseline (137/137 ids; int8 never flipped a greedy argmax — better than the "merely coherent" bar a lossy transform predicts); (b) **int8 path genuinely active** — the longctx16384 *synthetic* warm argmax flips 16→28 under int8 (a lossy transform DOES change synthetic compute, proving it's not bypassed; synthetic argmax is a speed signal only); (c) **memory** — longctx16384 RSS **22.66 → 22.31 GB (−0.35 GB)**, matching the predicted KV halving (16384·512·43 B ≈ 344 MB), NaN=0, lockstep, rc=0. The −0.35 GB at 16k is small (KV is a thin slice there) but **linear in ctx → ~5.7 GB at 256k**, where it compounds with 2l's −5.5 GB dense. *One bug found + fixed:* during the pre-freeze calibration window the read path fell into the bf16 branch but `kv_cache` is `NULL` under int8_kv → segfault; fixed by pointing the bf16 base at `kv_calbuf` while unfrozen (synthetic A/B confirms the calbuf-bf16 read is **bit-identical** to the kv_cache-bf16 path). Gated `DS4F_INT8_KV` (default **off** — lossy, keep bf16 the clean reference). LOSSY by design ⇒ argmax NOT *guaranteed* bit-exact (coherence is the gate); it merely *happened* to be token-identical on this gen.

- **int8 compressed-latent cache (cmp_kv) — ctx ceiling ~200k → ~255k, real-weight token-identical (Step 2n, `DS4F_INT8_CMP`, uncommitted).** The Tier-B2 memory half of the 256k push (pairs with 2m's int8 *KV* and 2l's int8 *dense*). The Tier-B2 compressor cache `cmp_kv` (CSA 10,752 B/pos + HCA 320 B/pos = 11,072 B/pos f32) is the single largest off-arena allocation past ~64k. `DS4F_INT8_CMP=1` stores each compressed latent slot `[kv_lora=512]` as **int8** (¼ the f32 footprint) with the same **S5 static per-channel scale** scheme as int8 KV (2m): a small bf16 `cmp_calbuf` stages the first `DS4F_INT8CMP_CAL` *slots* (default 64) while per-channel absmax accumulates; at the first `slot≥CAL` the scale freezes, the calbuf quantizes into the int8 store `cmp_q`, and reads switch to int8 — freeze happens between `forward_token` steps so every slot is wholly in calbuf (unfrozen) or wholly in `cmp_q` (frozen), no straddle. Read path (`ds4f_attn_tb2_worker`) is a single hoisted **3-way** branch — `c_i8` frozen → `ds4f_sve_dot_i8s`/`ds4f_sve_axpy_i8s` over the per-channel scale; `cmpbf` pre-freeze → bf16 calbuf; else f32 (`DS4F_INT8_CMP=0`). Forces `m->exact=1`, tierb2-only. New fields `cmp_q/cmp_scale/cmp_iscale/cmp_absmax/cmp_calbuf/cmp_caln/cmp_frozen` in `ds4f_layer`; codec `ds4f_cmp_freeze`/`ds4f_cmp_append_i8` reuses the int8-KV i8s kernels. **slot = pos/ratio** (CSA ratio=4 ⇒ a slot every 4 positions). **THE MEASUREMENT GOTCHA (critical — see memory `project_ds4f_ctx_ceiling_measured`):** `cmp_kv`/`idx_kv` are THP-backed off-arena allocs that cost **real physical memory (MemFree) but are INVISIBLE to `/proc/self/statm` RSS** — at 131072 they are 2.72 GB physical with *zero* RSS change. **For any ctx-ceiling / OOM / memory-reduction work, validate against MemFree, NEVER RSS** (task #43 wrongly killed int8_cmp on an RSS A/B — RSS structurally cannot see the win). *Validation (synthetic longctx + real-weight gen, NP=11):* **(a) memory** — MemFree @131072 (FP8-on-demand dense + int8 KV) **4.52 → 5.57 GB = +1.05 GB**, EXACTLY the cmp_kv-only prediction (8,304 B/pos × 131072 = 1.04 GB); combined ceiling slope **36.7 → 28.7 KB/pos**; **(b) ceiling** — two-point MemFree fit lifts the hard ceiling (MemFree-2.0 floor) **~200k → ~255k** (safe MemFree-2.4 ~242k) — clears 262144 at the hard floor; **(c) coherence** — real-weight gen A/B with `CAL=8` (forces freeze@slot8 ⇒ pos32, within the ~160-pos eos-terminated gen) is **TOKEN-IDENTICAL** to f32 cmp_kv (137/137, first_diff=NONE), int8 frozen path genuinely exercised, NaN=0, lockstep, rc=0. *Gotcha when testing:* a short gen hits eos ~136 tok ⇒ total pos ~160 < 256, so the **prod `CAL=64` never freezes in a short gen** — lower `CAL` (e.g. 8) to exercise the int8 path, else you only test the bf16 calbuf. Gated `DS4F_INT8_CMP` (default **off**, zero cost when off), `DS4F_INT8CMP_CAL` = calib window in SLOTS (default 64). **The combined int8 memory stack is now FP8-on-demand dense + int8 KV + int8 cmp → ~255k ctx.** Next mem lever: **int8 `idx_kv`** (2,688 B/pos → +2 KB/pos saved, slope → 26.6, ceiling → ~270k+) for a robust 262144 margin.

- **Direct MXFP4/FP8 GEMM register-dequant — MXFP4 tile-dequant +1.38–1.72× @M≥16 (real-weight 11n: +7.9% prefill, token-exact), FP8 confirmed already-optimal (Step 2o, `DS4F_MXFP4_GEMM_TILE`, uncommitted).** Investigated the user ask "optimize direct mxfp4/fp8 GEMM, decode to fp16/fp32 on register via SVE ld+insn". Two halves, both bench-driven (`ds4f_gemm_test` FP8+MXFP4 sweeps, 12T/1CMG, exp15-free synth, argmax + **relL2 vs f32 single-token ref**, never bit-equality):
  - **FP8 — already optimal, no change.** The Step-2f on-demand tile-dequant (`svld1ub`→LUT-gather→f32→bf16 L1 tile→`8x3_pv`, no resident bf16, −6 GB) IS the on-register SVE-ld dequant. Confirmed **bf16-FMA COMPUTE-bound, not dequant-bound**: register **magic** decode (no gather/LUT) vs gather is **NEUTRAL** (1.00–1.06×, ~1 % at M≥32 — dequant amortizes over M). And magic needs **FTZ → flushes E4M3 subnormals to 0** (LOSSY: ~0.5 max-abs vs the LUT over K=4096), which would break this path's bit-parity with the `DS4F_FP8_BF16=1` resident-bf16 prefill → the gather (lossless) correctly stays; the gated magic GEMM path was **reverted**. Bonus finding: **FP8→bf16 is LOSSLESS** (e4m3's ≤3 mantissa bits × pow2 E8M0 scale fit bf16's 7) → the GEMM is **relL2 3e-7 vs the f32 matvec ref** (f32-faithful, not merely argmax-exact) — so the −6 GB on-demand FP8 prefill is numerically ~identical to a resident-bf16 copy. Magic's real win is the **HBM-bound M=1 decode** matvec (lever E), not the GEMM.
  - **MXFP4 — NEW tile-dequant lever.** The expert GEMM (`matvec_mxfp4_8row_2x`, svtbl) is **dequant/issue-bound: FLAT ~84 Gmac/s from M=1 to M=64** (svtbl re-runs per token-pair, never amortizes), where FP8's tile-dequant scales 17→143. Added a gated **tile-dequant** path: dequant each 8-row group's nibbles **ONCE** (`svld1ub`→`svtbl`→×E8M0→bf16 L1 tile) and reuse across all M via the peak `8x3_pv` kernel. **LOSSLESS** (fp4×pow2 fits bf16 → relL2 **2e-7**, argmax-exact). Measured svtbl→tile: M=1 **0.33×** (tile loses — setup, no amortization), M=8 **1.08×** (crossover), **M=16 1.38× / M=32 1.53× / M=64 1.72×** (→147 Gmac/s, toward the bf16 compute peak). Gated `DS4F_MXFP4_GEMM_TILE` = **M-threshold** (default **0 = off**, keep svtbl which wins at M≤4; set ~8–16 to enable) — so it never regresses small M and is inert in production until enabled. Assumes vl==16 (A64FX SVE-512: each 16-elem chunk = lo/hi nibbles of one 32-block). **Real-weight 11n A/B — LANDED + VALIDATED (2026-06-08, alloc 49129596).** Batched-prefill perf (`REAL=1 EXACT=1`, MHC/Tier-B2 off — batched prefill is *disabled* under MHC/Tier-B2 so this synthetic-activation path on REAL MXFP4 bytes is the only vehicle; `FP8_BF16=1` dense, `PREFILL=512 PREFILL_BATCH=256`). **Token-EXACT:** argmax=108300 and prefill ‖x‖=7.626e+02 **bit-identical** across TILE∈{0,1,4,8,16} (6 runs), NaN=0, perf=11/11 lockstep — lossless on real weights at the activation-norm level, not just argmax. **Prefill +7.9%: 36.5 → 39.5 tok/s** (ms/tok 27.4→25.4, reproducible across repeat pairs), and **threshold-INSENSITIVE 1→16** (all give 25.3–25.4 ms/tok) ⇒ the compute-dominant experts carry **M≥16** in batched routing — *refuting the earlier "per-expert M may be small → neutral" caveat*. Comm% also fell 13→10 (faster expert-GEMM → less pre-all-reduce skew). Decode unchanged (18.95→18.92, expected — decode uses `mv_worker` not the GEMM path). **Recommended production setting: `DS4F_MXFP4_GEMM_TILE=16`** (full win; tiny-M experts stay on svtbl where single-node showed tile loses). *Regression check (single-node, alloc-free):* `ds4f_gemm_test` **205/205 OK** (BF16_PV/BF16/Q8_PV/FP8/MXFP4); `ds4f_exact_test` max-abs **5e-8** + `ds4f_tierb2_test` **2e-6** vs Python ref (header changes compile + forward math intact).

- **`kv_cache` window ring buffer — ctx ceiling ~255k → ~1M, the dominant long-ctx memory lever (Step 2p, automatic under tierb2, uncommitted).** This reframes the whole ctx-ceiling story and **supersedes int8 KV (2m) for ctx scaling.** *The discovery:* `kv_cache` was allocated at full `max_pos · kv_lora · 2B` for **all 43 layers** (`ds4f_impl.h:1852`), but under tierb2 **every layer reads `kv_cache` only over the last `window_size=128` positions** — sparse (ratio≠0) layers via `ds4f_attn_tb2_worker` (window term; older history is the compressed `cmp_kv`), and the 2 dense (ratio=0) layers via `ds4f_attn_exact_worker` (pure sliding window + sink). Nothing reads past the 128-window, so the full-`max_pos` allocation was **pure waste — ~44 GB/node at 1M ctx that was never read.** *The fix:* per-layer `ly->kv_slots = (tierb2 && !int8_kv) ? window_size : max_pos`, and index `kv_cache` everywhere as `(idx % kv_slots)` — sparse/dense layers ring-buffer to 128 slots; non-tierb2 / int8_kv keep `max_pos` (where `idx % max_pos == idx`, a **no-op ⇒ bit-exact**). Write `pos % kv_slots` (`forward_token`), read window `(p_lo+j) % kv_slots` (exact/tb2/prefill workers), `ds4f_warm_kv` fills only `min(npos, kv_slots)`. Gated on **tierb2** because the non-tierb2 synthetic path (`ds4f_attn_worker` full read) and batched-prefill (`ds4f_attn_pf_task`, needs all M positions) require the full buffer; batched prefill is anyway forced off under tierb2/MHC, and prefill is token-at-a-time → the ring holds each query's window. **`ds4f_arena_size` MUST mirror this** (added a `ring` param + per-layer kv sum) — the stale full-`max_pos` arena estimate (63 GB @1M) was itself the cause of a load-time OOM until fixed (→ **20.4 GB**). *Result:* `kv_cache` becomes a **constant ~5.6 MB** (43·128·512·2B) regardless of ctx — it **no longer scales**, so the only ctx-scaling caches left are `cmp_kv` (int8, 2,768 B/pos) + `idx_kv` (f32, 2,688 B/pos) → **slope 28.7 → ~5.5 KB/pos** (the int8-KV 21.5 KB/pos kv term vanishes entirely; **int8 KV / Lever B is now moot — the bf16 ring is strictly better than a half-`max_pos` int8 kv**). *Also found:* **`DS4F_IDX_INT8` (Lever F) is a memory *anti*-lever** — it keeps f32 `idx_kv` **and** adds a resident int8 mirror `idx_kv8` (+672 B/pos), so leave it **off** for memory (it's a speed experiment only). *Validation (NP=11, real weights, this alloc before it degraded):* **(a) bit-exactness** — real-weight gen A/B (149-tok prompt > 128 so prefill wraps the ring), ring (sparse-only, then ring-**all** incl. dense layers) is **TOKEN-IDENTICAL to HEAD** (byte-exact gen_ids, NaN=0, lockstep) — proves a pure storage relocation; **(b) memory** — `DS4F_CTX_WARM` MemFree sweep (FP8-on-demand + int8 cmp, **measure MemFree NOT RSS**): **262144 fits MemFree 6.24 GB, 524288 fits 4.15 GB** (ring-all), NaN=0, 11/11; **(c) 1M — CONFIRMED FIT (2026-06-08, alloc recycled by native re-stage).** `DS4F_CTX_WARM=1048576` warms all 43 layers (`warmtb2 DONE`, no ceiling hit) then decodes: **rc=0, 11/11, NaN=0, MemFree 2.45 GB / MemAvail 2.89 GB spare**, arena 20.4 GB, decode 1.60 tok/s (slow — memory-first). **Two-point slope CONFIRMED ~5.9 KB/pos** (524288 MemFree 5.56 GB, 1048576 MemFree 2.45 GB ⇒ 3.11/0.524M = 5.93 KB/pos) = cmp_q int8 2,768 + idx_kv f32 2,688 + s_idx_scores ~48 B/pos — exactly the traced per-pos allocs, **NO mystery extra**. (The earlier "~8 KB/pos / 2.5 KB extra" was a single-point error that folded the ~23.1 GB CONSTANT base — weights + off-arena compressor/indexer *weights* ~1 GB + OS — into the per-pos rate; the base is constant, identical at both points.) **Hard ceiling ~1.12M** (MemFree-2.0 floor), safe ~1.0M (1.5M OOMs). Single-node `ds4f_exact_test` 5e-8 / `ds4f_tierb2_test` 2e-6 unchanged. *Op note:* the alloc's PMIx/plexec was degraded earlier when a timed-out combined Bash job SIGKILL'd a mid-load 1M `mpiexec`; **recovered by a native re-stage** (`run_ds4f_stage_11n.sh`, a fresh `mpiexec` launch — no pjsub needed) which both refreshed `/local` and cleared the launch layer (smoke load=11/11 after). **NEXT for >1M — DONE (Step 2q):** `idx_kv` int8 replacement (`DS4F_IDX_INT8` made a true replacement) → 672 B/pos → slope **4.08 KB/pos → 1.5M CONFIRMED, hard ceiling ~1.58M**. (`s_idx_scores` is over-allocated at `n_threads·max_pos` vs the touched `max_pos/ratio` — ~150 MB, minor.) **12-node config: COUNTERPRODUCTIVE with a large controller reserve** — caches are replicated so the 12th node (node 0) adds no ctx capacity, only ~1 GB expert-shard saving; a 7 GB reserve makes node 0 the bottleneck (~455k, 3× worse than 11-node ~1.12M). 12-node helps only with a ≤~3 GB reserve (claude uses 0.44 GB) → ~1.25M, but needs a 12-way re-stage + an EP rank on claude's node. Then Phase 2 = context parallelism for *speed* at 1M.

- **idx_kv int8 REPLACEMENT — ctx ceiling ~1.12M → ~1.58M, real-weight token-identical (Step 2q, `DS4F_IDX_INT8`, uncommitted).** Turns the documented *anti-lever* into the real lever and is the natural follow-on to the kv_cache ring (2p). After the ring, `idx_kv` (f32, 2,688 B/pos) is the largest remaining ctx-scaling term (45% of the 5.93 KB/pos slope). The old `DS4F_IDX_INT8` was additive (kept f32 `idx_kv` AND added an int8 mirror `idx_kv8`, +672 B/pos = worse). *The fix, near-zero new code:* the f32 `idx_kv` scan (`ds4f_index_score`) runs **only for T < `DS4F_IDX_F32_SLOTS`=64** — for T≥64 the resident int8 `idx_kv8` svdot scan is used (the validated path) — so the f32 `idx_kv` needs only **64 slots**, not full `nslot`. Alloc shrinks f32 `idx_kv` to 64 slots when `idx_int8 && m->pool` (gated on pool so the int8 scan is guaranteed for T≥64); the write (`ds4f_index_step`) and the warm fill guard `slot < 64`. Net: `idx_kv` **2,688 → 672 B/pos** (the int8 mirror only), **slope 5.93 → ~4.08 KB/pos**. *Bit-identical-compute to the old additive idx_int8* (same int8 scan for T≥64, same f32 scan reading the same slots 0–63 for T<64) — only the unused f32 tail is freed → inherits its argmax-exact validation. *Validation (NP=11, real weights):* **(a) coherence** — gen A/B `DS4F_IDX_INT8=1` vs `=0`, 149-tok prompt + MAX_NEW=160 (total >256 so the int8 idx scan genuinely engages) is **TOKEN-IDENTICAL** (160/160, NaN=0, lockstep); **(b) memory** — 1M MemFree **2.45 → 4.45 GB (+2.0 GB**, exactly 2,016 B/pos × 1M), and **1,572,864 (1.5M) CONFIRMED FITS** (`warmtb2 DONE`, MemFree 2.31 GB, rc=0, 11/11, NaN=0); two-point slope 4.08 KB/pos ⇒ **hard ceiling ~1.58M**; **(c) regression** — idx_int8 OFF (default) `ds4f_exact_test` 5e-8 / `ds4f_tierb2_test` 2e-6 unchanged. Gated `DS4F_IDX_INT8` (default **off**). **Combined with the 2p ring: ctx 255k → ~1.5M (~6×) on the 11-node replicated design, memory-first, token-identical.** *Caveat:* needs the pooled int8 scan (always present in the EP runner; serial test path keeps full f32 idx_kv — inert there). NEXT: replicated-dense tensor-parallel (shard the ~8 GB dense, −7 GB/node) or Phase 2 context parallelism (shard KV → node count multiplies the ceiling + speeds the O(T) scan).

- **Tensor-parallel the REPLICATED dense weights — −4.2 GB/node, ctx ceiling ~1.58M → ~2.6M (Step 2r, `DS4F_TP_*`, committed `683cfaf`/`7961f1a`/`5614dea`).** The dense (MLA attn + shared + router + embed + head) is replicated on every EP node (the 240-vs-160 GB gap); after the ring (2p) + idx_kv (2q) ctx levers, it's the largest reclaimable per-node memory. Megatron-style col→row sharding across the 11 nodes, staged + env-gated (default off → inert), **FP8 dense required** (`DS4F_FP8_BF16=0`; Q8_PV 528 B groups unsliceable). Reuses the existing per-layer `ar_cb` all-reduce + a row-shard loader `ds4f_load_dense_vshard` (fp8→bf16 promote OR fp8/bf16 same-dtype, with a row offset); arena mirrors every shard. Stages, each real-weight A/B validated (NaN=0, lockstep):
  - **1 head vocab-shard (`DS4F_TP_HEAD`, −0.95 GB):** node holds head rows `[r·vocab/11..)`; decode head matvecs only its shard into `s_logits[head_r0..]`, zero-fills the rest, all-reduce-SUMS via `ar_cb` → full logits **BIT-EXACT** (disjoint shards + zeros sum exactly) → TOKEN-IDENTICAL. **embed** is the twin (`DS4F_TP_EMBED`, −0.97 GB): vocab-shard, `embed_lookup` zero-fills + owner copies its token's row + ar_cb-SUM → full embedding BIT-EXACT (TOKEN-IDENTICAL A/B).
  - **2 shared MLP (`DS4F_TP_SHARED`, −0.63 GB, ZERO extra comm):** `sh_w1/w3` col-parallel (shard `shared_inter`); swiglu local; `sh_w2` replicated contracts a ZERO-PADDED `s_shg` → partial shared, FOLDED into the existing routed-expert reduce (one all-reduce sums routed+shared). COHERENT (contraction shard → f32 reassoc, like the EP reduce; both completions sensible).
  - **3a attention wq_b (`DS4F_TP_ATTN`, −1.32 GB):** `wq_b` col-parallel **by head**; `ds4f_head_split` gives the q-norm/RoPE + attn workers the owned head range — **explicit range, NOT a zero-padded `s_q` (RMSNorm of a zero head → NaN)**; `s_attn` zeroed then only owned heads filled → `wo_a/wo_b` (replicated) on the partial → reduce the 4096 `s_o` (not 8192 `o_inter` — `wo_b` is linear). +1 all-reduce/layer. COHERENT.
  - **3b o-proj wo_a (`DS4F_TP_OPROJ`, −1.31 GB):** `wo_a` ROW-shard by `o_inter` (the clean alternative to the block-diagonal column-shard the 8-groups-∤-11 layout would force) — node holds `wo_a` rows `[oi0..)`, `ds4f_mv_bd_worker` gains a `goff` o_inter-offset so the group is `(goff+i)/o_lora` (shard need not align to groups). With TP_ATTN, `s_attn` is all-reduced to FULL first (head-partial + zeros → exact). `wo_b` replicated. COHERENT.
  - *Validation:* full stack (head+shared+attn+oproj) real-weight A/B on vs off — **COHERENT, NaN=0, lockstep, RSS 21.82 → 17.62 GB (−4.20 GB)**, GB/tok-weights 7.32→3.11. Regression (TP off) `ds4f_exact_test` 5e-8 / `ds4f_tierb2_test` 2e-6 unchanged. With embed added, only `wo_b` (1.4 GB, needs the strided o_inter column-shard) stays replicated; the dense TP stack (head+embed+shared+attn+oproj) frees **~5.2 GB/node** (arena 20.4→15.58 GB, RSS @short-ctx 21.83→17.62). **Ceiling CONFIRMED by ctx-warm MemFree sweep (full TP incl embed + int8 cmp/idx, FP8-on-demand): 2M fits MemFree 4.21 GB, 2.5M fits 2.44 GB, 2752512 (2.75M) fits MemFree 2.67 GB (all `warmtb2 DONE`, 11/11, NaN=0); 2949120 (2.9M) and 3145728 (3.1M) OOM at LOAD (rc=137).** ⇒ **full-TP ceiling ~2.75M confirmed** (cumulative ring(2p)+idx_kv(2q)+TP(2r): ~255k → ~2.75M, **~11×**, 11-node, memory-first). KEY: past ~2.75M the binding constraint has shifted from the (now-TP'd) dense weights to the **int8 ctx-caches** (~3.4 KB/pos = `cmp_q` 2.77 + `idx_kv8` 0.67, allocated at LOAD) → the next ceiling lever is int4 ctx-caches (NOT more dense TP), done below.

- **Step 2s — int4 `cmp_q` ctx-cache (`DS4F_INT4_CMP`, default off, commit `f9fb3ff`): ceiling ~2.75M → ~4.2M.** `cmp_q` (the compressed-attention-latent cache) is the per-pos dominator at 2768 B/pos; int4 it → `cmp_q4` = 2 signed-nibbles/byte (±7, **same** S5 static-per-channel `cmp_scale`) = **1384 B/pos** (half), combined slope ~3.44→~2.06 KB/pos. A sub-mode of `int8_cmp` (reuses its calbuf/scales/exact-path/append-dispatch); `cmp_q4` is off-arena (no arena mirror). New SVE nibble-unpack kernels `ds4f_sve_dot_i4s`/`axpy_i4s` (`svld1ub` + lo/hi nibble + 4-bit sign-extend `(x^8)−8` + `svzip` back to channel order) + scalar fallbacks; `ds4f_cmp_{quant_row,freeze,append}_i4` mirror the int8 codec with `/7`. *Validation:* SVE-vs-scalar dot 4e-5 (FP reassoc) / axpy bit-identical; regression (off) 5e-8/2e-6 unchanged; **real-weight 11n gen A/B (full TP, FP8 dense, CAL=8 so the frozen int4 path runs through prefill+gen): int4 TOKEN-IDENTICAL to int8** (all 48 tokens, argmax 372, NaN=0, lockstep — the int4 latent error sits below the argmax-flip threshold, better than the predicted "coherent-not-identical"). Ceiling: 3670016 (3.5M) fits MemFree 3.68 GB, 4194304 (4M) fits MemFree 2.23 GB (`warmtb2 DONE`, 11/11, NaN=0) ⇒ **~4.2M**. **Cumulative ring(2p)+idx_kv(2q)+TP(2r)+int4_cmp(2s): ~255k → ~4.2M (~16×)**, 11-node, memory-first.

- **Step 2t — int4 `idx_kv8` indexer cache (`DS4F_IDX_INT4`, default off, commit `23b793b`): ceiling ~4.2M → ~4.8M.** The last per-pos scaler after int4 cmp. `idx_kv8` (672 B/pos) → `idx_kv8_4` = 2 signed-nibbles/byte (±7, per-position scale `/7`) = **336 B/pos**. Sub-mode of `DS4F_IDX_INT8` (reuses the f32-shrink-to-64-slots + resident-scan infra). Unlike cmp (f32 q · int8 k), the indexer scan is a **symmetric int8×int8 `svdot`** (`ds4f_idxsc8r_worker`); int4 keeps the int8 query and **unpacks each position's int4 key → int8 temp** (hd=128 ops, amortized over H=64 heads ≈ 6%) then reuses `svdot` (new `ds4f_idxsc8r4_worker` + `ds4f_idx_quant_pos_i4`). `index_step` gains an `idx_kv8_4` param; the f32-shrink gate, write, scan dispatch, and synthetic warm all branch on it. *Validation:* regression (off) 5e-8/2e-6 unchanged; **real-weight 11n gen A/B (full TP + int4 cmp, ~270-tok prompt so the gen runs at T=67–81 > `DS4F_IDX_F32_SLOTS`=64 ⇒ int4 idx scan engaged throughout): TOKEN-IDENTICAL to int8 idx** (56/56 tokens, argmax 447, NaN=0, lockstep — int4 per-position quant flips zero top-k selections). **Cumulative 2p+2q+2r+2s+2t: ~255k → ~4.8M (~19×)** (combined int4 cmp+idx slope ~1.72 KB/pos); confirm via ctx-warm sweep.
- **Step 2u — `wo_b` FP8 o_inter column-shard (`DS4F_TP_WOB`, default off, commit `0c21298`): the last replicated dense, BIT-EXACT, −1.22 GB, ceiling ~4.8M → ~5.5M.** Under TP_OPROJ the o-proj input `s_o1` is already zero outside the owned o_inter slice `[oi0,oi0+oi_rows)` (wo_a o_inter row-shard fills only owned), so wo_b's other columns multiply zeros → column-sharding wo_b to those columns gives an **IDENTICAL** result (bit-exact, not just coherent) and reuses the existing o-proj reduce (no new comm). Requires TP_OPROJ + FP8-on-demand dense (the ceiling path; prefill bf16 keeps full wo_b). New `ds4f_load_dense_cshard`/`ds4f_cshard_worker`: strided per-row copy of the FP8 weight columns + the 128-blocked E8M0 scale columns (128-aligned via `ds4f_tp_oproj_shard`). wo_b alloc `cols=oi_rows`, arena mirrors it, decode o-proj matvec contracts `oi_rows` with input `s_o1+oi0`. *Validation (11n, full TP + FP8 + int4 cmp/idx):* real-weight gen A/B WOB on vs off **BYTE-IDENTICAL** (argmax 372, NaN=0, lockstep); arena 15.58→14.36 GB; **5242880 (5M) fits MemFree 2.64 GB** (`warmtb2 DONE`, 11/11, NaN=0) ⇒ ceiling **~5.5M**. **Cumulative ring(2p)+idx_kv(2q)+TP-dense(2r)+int4cmp(2s)+int4idx(2t)+wo_b(2u): ~255k → ~5.5M (~21×).** The dense is now FULLY tensor-parallel (head/embed/shared/attn/wq_b/wo_a/wo_b); the remaining ctx lever is Phase 2 CP (node-count × ctx).

## Phase 2 — sharded-KV context parallelism (CP, `DS4F_CP`): shard the compressed caches across nodes

After 2p–2t the only per-pos state left is the compressed caches `cmp_q4` (1384) + `idx_kv8_4` (336) = **~1720 B/pos**, still REPLICATED on every node. CP shards *those* by compressed-slot range across the N=11 nodes ⇒ memory/node ÷N (ceiling toward tens of millions, then bound by per-node constants ≈ arena 15.6 GB + window + 64-slot f32-idx) AND the O(T) indexer scan ÷N. The window (`kv_cache`, 128-slot ring, ~5.6 MB constant) is NOT a scaler → stays replicated. Decode-only (batched prefill stays single-node, like mHC/tierb2). Plan file: `~/.claude/plans/see-a64fx-ds4f-md-and-keep-floofy-dragon.md`.

**Why online-softmax-combine (not gather-selected):** CSA (ratio=4) attends the indexer top-512; HCA (ratio=128) attends ALL T compressed (`ds4f_impl.h` ~:3079). Gathering selected rows works for CSA's 512 but explodes for HCA (millions). The unified mechanism is flash-attention's **online-softmax-combine**: each node attends only the slots it OWNS into a partial `{m,l,acc}`, then nodes combine (one MAX-reduce + one SUM-reduce per layer, ~131 KB, no latent gather). The current single-pass softmax in `ds4f_attn_tb2_worker` (`ds4f_impl.h` ~:3265-3289) already computes `{m,l,acc}`.

- **Stage A — DONE, committed `1f7d46a`.** `tp_allreduce_max` (`tp_allreduce.h`) clones the sum's recursive-doubling schedule with an element-wise-max recv (`tp_ar_recv_max`); MAX is exact (result is one of the inputs ⇒ no bf16 round-trip needed ⇒ deterministic ⇒ lockstep-safe), mirroring the existing `tp_allreduce_argmax`. `ep_armax_callback` + `m->ar_max_cb`/`ar_max_ctx` wired like `ar_cb`. `DS4F_CP_SELFTEST` (gated, kept as CP regression gate). *Validated 11-node:* all ranks `tp_allreduce_max PASS bad=0 worst=0.000e+00` (correct vs serial max, bitwise-exact, lockstep); inference ran 11/11 afterward (max-reduce coexists with sum-reduces — shared `seq` stays aligned).

**REFINED design (Stage A done `1f7d46a`; Stage B foundation + the riskiest algorithm validated `b5397e8`).** Two findings reshaped the plan into a **simpler, BIT-EXACT, gather-selected** path (no online-softmax-combine, no idx-shard/merge for the first increment):

- **Finding 1 — the top-k MERGE is EXACT** (`tools/cpmerge_test.c`, 200/200 trials, S=50000 K=512 N=11, heavy ties): each node's local top-K over its slot shard, gathered + merged, equals the replicated global top-K (proof: a global-top-K slot has <K above it globally ⇒ <K on its own shard ⇒ in that shard's local top-K). So sharded scan + merge is sound when needed.
- **Finding 2 — the per-channel cmp/idx scale is calibrated from the first `CAL` slots**, which naive slot-sharding gives to only ONE node ⇒ broken scale on the others. **Fix: REPLICATE the tiny `[0,CAL)` calibration window, shard `[CAL,S)`** — all nodes calibrate identically, no scale broadcast. (Supersedes the earlier "self-calibrated per node" note, which was wrong.)

**Minimal CP (CSA gather-selected, the first increment) — BIT-EXACT, simplest:**
  1. **Keep the indexer (idx) REPLICATED** — full scan on every node ⇒ identical global top-512 `sel[]`, so **NO merge needed** (Finding 1 is the fallback if idx is sharded later). idx_kv8_4 stays full (336 B/pos).
  2. **Shard `cmp_q4` only** (the 1344 B/pos CSA dominator) per `ds4f_cp_slot_shard`, with `[0,CAL)` replicated + `[CAL,S)` sharded (Finding 2). CSA (ratio=4) layers only; HCA (ratio=128, attend-all) `cmp_kv` stays replicated (32× smaller). Per-layer `cp_t0/cp_t1`; global slot g→local: `g<CAL ? g : (cp_t0≤g<cp_t1 ? CAL+(g−cp_t0) : not-owned)`. Alloc `(CAL+cp_t1−cp_t0)` slots; write/warm skip non-owned.
  3. **Gather the selected latents** (Stage C, minimal): after `sel[]` (identical on all), each node dequants its owned selected slots → bf16 into a `[nsel][KV]` buffer at index j (zero else; for replicated `g<CAL` slots rank-0 contributes to avoid double-count), `ar_cb`-SUM ⇒ all nodes hold the same selected bf16 latents.
  4. **Attention** reads the gathered bf16 buffer for the compressed term (replaces `cmp_q4[sel[j]]`); the replicated 128-window + sink unchanged. ⇒ same selected slots + same latents + same window = **BIT-EXACT (byte-identical gen)**, validated trivially.
  - *Memory:* cmp 1344→~122 B/pos/node (/11) + idx 336 replicated ⇒ slope 1720→~460 B/pos (~3.7×); ceiling ~5.5M → tens of millions. *Comm:* +1 `ar_cb` gather/CSA-layer (`nsel·KV` bf16 ≈ 0.5 MB).
  - *Gate:* gen A/B `DS4F_CP=1` vs off ⇒ **byte-identical** gen (bit-exact, not just coherent), NaN=0, lockstep; then ctx-warm MemFree sweep past 5.5M.
  - **Later optimizations** (not the first increment): shard idx too (the merge, Finding 1) to drop the 336 B/pos; and for HCA attend-all replace the per-layer gather with the **online-softmax-combine** (`ar_max_cb`(m)+`ar_cb`([l|acc]), Stage A's MAX collective) to avoid gathering all T HCA slots. Guards: abort batched prefill under CP.

**LANDED + VALIDATED (`b5397e8`→`a4d434d`):** `DS4F_CP` = the validated gather (cmp replicated; rc=0 + rank-0 byte-identical @ MAX_NEW 24 & 48). `DS4F_CP_SHARD` = the cmp_q4 slot-shard (memory win): **BYTE-IDENTICAL** to CP-off (calibration window `[0,CAL)` replicated ⇒ same scale ⇒ same quant; gather reassembles exactly), 0 bounds-guard hits. Two bring-up bugs fixed: (1) `ds4f_cp_slot_shard` **negative range** — `ds4f_tp_rowshard` clamps only `r1`, so a high rank past the small CP tail got `r0>S≥r1` ⇒ `nslot_q<CAL` ⇒ freeze overran the alloc (rc=139 @ long decode); fixed `a0=min(a0,S); a1=max(a1,a0)`. (2) `s_cmp_gather` sized `max(max_pos,topk)*KV` = **16 GB @ 8M** ⇒ load OOM; fixed to `index_topk*KV` (1 MB const; only `ns≤index_topk` written). **Sweep (full TP+FP8+int4+wo_b+CP_SHARD): 8388608 (8M) CONFIRMED FIT** (rc=0, 11/11, NaN=0, `warmtb2 DONE`, MemFree 4.91 GB spare, decode 0.40 tok/s); **12M & 16M OOM at LOAD** (load-peak = shrinking blob + growing max_pos-scaled caches, not warm). cmp slope ~1344→~162 B/pos but **idx_kv8_4 stays replicated (336 B/pos) = the dominant remaining term** ⇒ confirmed ceiling **~5.5M → 8M**. Next lever (shard idx via the proven merge) drops idx 336→~30 ⇒ tens of millions. **Cumulative ring+idx+TP+int4×2+wo_b+CP-cmp-shard: ~255k → 8M (~31×).**

**idx-shard LANDED (`DS4F_CP_IDX`, commit `829f451`):** the last replicated ctx-cache. idx uses a **per-slot** scale (not per-channel-from-early-slots), so a clean `[0,nslot)` shard — **no `[0,CAL)` replication** (simpler than cmp). Option A (exact, not the merge): each node scans its owned indexer slots → **GLOBAL score positions**, zero rest + `ar_cb`-SUM ⇒ full `[0,T)` scores == the replicated scan, then the existing topk. New `s0` score-offset in `ds4f_idxsc8r4_worker`; `index_step` gains `idx_cp/s0/s1/ar_cb` + the sharded-scan-then-gather + write remap; alloc shards `idx_kv8_4`+`idx_pscale`. Validated 11n @48: `DS4F_CP_IDX` on/off **BYTE-IDENTICAL**, 0 guard hits (no new bugs — the cmp `cp_slot_shard` negative-range fix carried over). **Sweep (cmp+idx shard): 12582912 (12M) CONFIRMED FIT** (rc=0, 11/11, NaN=0, MemFree 3.94 GB; 12M had OOM'd at load with cmp-shard-only). idx 336→~30 B/pos lifts the load-peak. **Cumulative … +CP-cmp-shard+CP-idx-shard: ~255k → ~12M safe (~47×)** — both compressed caches now slot-sharded. Bracket: **12M rc=0 (MemFree 3.94 GB spare, full decode); 14M warms all 43 layers but decode fails rc=1 (MemFree 2.07 GB, too tight); 16M OOM at load.** With both caches sharded the ceiling is now **LOAD-PEAK-bound** (the ~25 GB blob transient + 14.36 GB arena during load, not the caches) — further cache-sharding won't lift it; the next lever is reducing the **load-peak** (stream/free the blob faster during load, or shard the arena dense further). Decode comm also grows (idx score-gather: 32.5% @12M). **CP took ~255k → ~12M (~47×); whole journey 255k → 12M is ~47× on 11 nodes.**

**idx-MERGE LANDED (`DS4F_CP_MERGE`, default on, commit `4ea1cd4`) — the long-ctx decode-speed lever.** Replaces Option-A's full-`[0,T)` score gather (ar_cb-SUM, ~12 MB/CSA-layer @12M) + O(T) topk with the proven candidate-merge (Finding 1, `cpmerge_test` 200/200): each node scans owned slots → LOCAL top-k → `{global slot, score}` candidates; gather `N·k` (~45 KB, slots as float, exact <2^24); `ds4f_cp_merge_topk` → global top-k (same rule as `ds4f_index_topk` ⇒ identical `sel`), skipping the O(T) final topk. **12M ctx-warm: decode 0.44 → 0.85 tok/s (1.93×), comm 32.5% → 7.4%**, NaN=0, MemFree unchanged (buffers tiny); MERGE 0-vs-1 gen A/B **BYTE-IDENTICAL** (argmax 372). The win scales with T (O(T) comm+topk → O(N·k) both). Gated default-on under `DS4F_CP_IDX`; `=0` falls back to Option A.

**"Load-peak" investigation (the next-lever probe) — it is the WARM-peak / model-arena floor, NOT a transient.** Findings: arena_size is constant under the ring (only kv_cache, window-sized — no max_pos term); the blob is `mmap(MAP_PRIVATE,PROT_READ)` from lliofs (Fugaku LLIO node-local, *not* tmpfs), dropped per-tensor (MADV_DONTNEED) AND `munmap`'d on the success path — so the blob is released post-load. Decisive data point: **14M LOADS (load=11/11) and WARMS (warmtb2 DONE) but decode fails (rc=1, MemFree 2.07 GB)** ⇒ the limiter is the **warm-peak = the committed sharded caches + the 14.36 GB model arena filling the node**, not a load transient. The clean cache slope **8M→12M = 243 B/pos exactly matches the fully-sharded cmp+idx prediction** (so the sharding works); 12M→14M jumps to ~935 B/pos + 16M OOMs at load = near-full-RAM pressure (LLIO file cache / fragmentation) at the high end, which is **not a ctx-cache** to shard. **Conclusion: the ctx-cache-sharding levers are EXHAUSTED at ~12M safe / ~14M hard; the model arena (14.36 GB: FP8 dense + MXFP4 EP-expert shard) is the new floor.** Further ctx gains need a *smaller arena* (int4/lower-bit experts, or shard the EP experts across more nodes) or more nodes — a different work category from ctx-cache sharding. Pinning the exact 12–16M boundary needs a stable alloc (env was too flaky this session to probe 13–15M cleanly).

### Decode tok/s — roofline & the path to ≥20 (current ~13 @10k, ~0.8–1.2 @1M+)
Optimized decode @ctx10240 = **13.03 tok/s = 76.7 ms/tok**. Per-phase (Step 2l breakdown) and whether it **amortizes under M>1** (batched/speculative decode):

| phase | ms | amortizes (M>1)? | why |
|---|---|---|---|
| comm (43 EP all-reduces) | 12.7 | **yes ÷M** | latency-bound 296 µs/16 KB Put — fewer reduces/token |
| qkv+o_proj+shared+experts (weight matvecs) | 27.5 | **yes (GEMM)** | M>1 ⇒ dequant-once-per-tile reused (MXFP4 tile 1.5–1.7× @M16-32; int8 dense) |
| attn (window+selected) | 13.8 | no | per-position; SVE, L2-resident, near floor |
| tb2prep (lcmp/topk/scan/qproj) | 17.5 | no (scan part shards under **CP**) | per-position; partly O(T) |
| misc | ~5 | — | |

- **Absolute BW floor** (weights only): ~2.9 GB/tok/node ÷ ~720 GB/s ≈ **4 ms ⇒ ~250 tok/s** — decode is NOWHERE near BW-bound; it is **comm-latency + dequant-issue + per-position-attn bound**.
- **Amortizable** (comm + weight matvecs) = 12.7 + 27.5 = **40.2 ms**; **non-amortizable** (per-position attn + tb2prep) = 13.8 + 17.5 = **31.3 ms** ⇒ even with perfect M-batching the single-stream floor is **~31 ms ≈ 32 tok/s**.
- **≥20 tok/s IS achievable — via speculative decode (the math):** draft K + verify-in-one-pass (M=K). At K≈3 accepted: comm 12.7→**4.2** ms/tok (÷3), the 27.5 ms weight matvecs become an M=3 GEMM (dequant amortized ~1.4× ⇒ ≈**12.8** ms/tok), attn+tb2 stay 31.3 (per-position) ⇒ **≈48 ms = ~20.7 tok/s**. Batched **multi-sequence** decode (B=8–32) reaches similar/higher **aggregate throughput** (comm+matvec fully amortize; per-sequence attn/tb2 scale with B). Both need packed-B GEMM (note the known A64FX batched-prefill regression, `project_batched_prefill`).
- **Ceiling ~32 tok/s** (the per-position attn+tb2 floor). To beat it: cut attn (near floor already) or tb2's O(T) scan — which is exactly what **CP shards at high ctx**, so CP also *preserves* ~20–30 tok/s at 1M+ where decode is otherwise ~1 tok/s (scan-bound). **Net: spec/batched decode gets to ~20–30 tok/s; CP keeps it there at long ctx.**

### Decode-perf wins — LANDED on `glm5-2` (2026-07): 12.26 → 14.13 tok/s (+15.3%), all bit-identical

Real-weight 11n A/B (agentic `--preset decode`, ctx 1759 unless noted). Each lever measured, the wins default-on; the dead ends measured-and-reverted (the discipline: no lever ships without a same-config A/B).

| commit | lever | effect | default |
|---|---|---|---|
| `691be067` | **`DS4F_FLAGBAR`** per-worker flag barrier + `ds4f_row_slice` Q8_PV fix | **+8% decode & prefill**, bit-identical | on |
| `c303fcaa` | **`DS4F_ATTN_GEMM`** 8-head KV-reuse attention | **+6.6% decode, attn −50%** (+8.3% @ctx5026, grows w/ctx), bit-identical | on |
| `54476f06` | verify-path KV-reuse (spec/GEMM-decode) | verify path only (+5% GEMM-decode) | on |
| `6cf899b8` | **`DS4F_IDX_GEMM`** 8-head indexer scan | +16% `tb2scan` (O(T), long-ctx lever), bit-identical | on |
| `1c515aa4` | **`DS4F_IDX_INT8W`** int8 qproj weight | **+4.1% @ctx5026**, LOSSY, quality-gate-passed | **off** (opt-in) |

**Prefill (`9504cd69` + `4fbe5078`, opt-in +74%: 13.28 → 23.08 tok/s):** production gen prefill (MHC+Tier-B2) ran token-by-token because the batched `ds4f_forward` GEMM is die-guarded under MHC/Tier-B2. `DS4F_PREFILL_GEMM` routes it through `ds4f_forward_verify` (the MHC+Tier-B2 batched forward built for spec decode — now Q8-correct + ATTN_GEMM'd) in chunks of K≤32: (1) the per-layer EP all-reduce fires once per K tokens (comm 16.5→7.4%, ar_calls 216k→2.4k) and the dense projections become an M=K GEMM → **17.72 tok/s**; (2) the biggest per-position tb2 cost, the indexer q-projection (`idx_wq_b @ q_lat`, 7.5 ms/tok as K matvecs), is batched to **one M=K GEMM** (weight streams once/chunk) via a `q_pre` injection into `index_step` — also fixes a latent stale-`s_qlat` in the verify → **21.89 tok/s (+23%)**. Coherent (GEMM reassoc), NaN=0. Batched-prefill floor at ctx1759 (45.8 ms/tok): scan 3.66 / attn 3.25 / lcmp 2.39 ms + comm 4 + batched dense GEMM + **per-position mHC (batched, commit 6080ceee)**; remaining floor = scan/attn/lcmp (per-query or sequential, hard). The batched-verify path (which *failed* as a decode/spec lever on accept rate) is the prefill workhorse. **Generalizable: batch any per-position matvec whose weight is shared across positions (qproj) into one M=K GEMM.**

**Longer-context end-to-end validation (real-weight 11n, ctx 7175 prompt + 128 gen, `DS4F_PREFILL_GEMM=1 K=32`):** prefill **21.99 tok/s** (45.5 ms/tok, comm 8.6%), decode **13.13 tok/s** (76.2 ms/tok, comm 15.5%), `rc=0`, **NaNs=0**, RSS 21.94 GB, wall 538 s. Lockstep confirmed (all 11 ranks argmax 9047 prefill / 16 decode); output coherent (continued a benchmark table with correct columns + plausible numbers). Prefill holds ~+65% over token-by-token at 7k (per-position attn/scan floor grows slightly with ctx); decode ~13 (indexer scan + attn grow with T, the expected long-ctx trend).

**The key insight for decode — MLA has 1 KV head, so decode attention re-reads each `kv[j]` latent 64× (once per query head)**, running at ~0.5% of compute peak (L2-read-bound, NOT thread-bound — a balanced head-split gave *zero* gain, which is what pointed to the real fix). `DS4F_ATTN_GEMM` blocks 8 heads so each `kv[j]` is loaded once and reused across 8 heads (score dots + value axpy) → 8× fewer L2 reads. Same pattern applies to the batched prefill attention (already HBLK=8, `ds4f_attn_prefill_worker`) and the verify path.

**Measured & REVERTED dead ends (documented so they're not re-tried):**
- *Speculative MTP decode* — mechanism works (comm 16.5→3.4% via 1 reduce/2 tok) but net LOSS: batched-K=2-verify cost C₂/C₁≈1.52 needs α>0.76, MTP α is only ~68%; and the real accept rate collapses to 27% (batched verify drifts from the matvec draft). Not economical without a cheaper batched verify (needs batched attn/tb2) + higher α.
- *Attention SW-prefetch* — −2%: the compressed cache is L2-resident at agentic ctx (index_topk·kv_lora·4B ≈ 0.5–1 MB < 8 MB CMG L2), no HBM miss to hide.
- *Balanced head-split attention* — flat: attn is NOT thread-bound (this refutation led to the KV-reuse win).
- *Indexer matvec 8-row / widen-zip* — the indexer qproj is a small (K=1024) matvec that peaks at 24 threads and regresses at 48 (widen_bench); zip==lsl (widen isn't the bottleneck); only **fewer bytes** (int8) help → `DS4F_IDX_INT8W`.
- *int8 compressor* (`tb2lcmp`, K=4096) — quality gate passed but NO speedup (K=4096 already pipelines across 48t; not byte-bound). **General rule: int8 W8A8 helps only SMALL matvecs (K≲1024).**

The remaining decode cost is genuinely hard: **comm** (straggler-sync, only batched/spec decode amortizes it), the **indexer** (matvecs mostly irreducible, top-k O(T·log k) heap-optimized, scan now blocked), and **dense matvecs** (already near BW). Diagnostics: `DS4F_PROF=1` prints the per-phase decode breakdown; the widen microbench is `scratchpad/widen_bench.c` (throwaway).

### MTP scaffold + spec-decode plan (commit `09aa458`)
The model ships an MTP module (`config num_nextn_predict_layers=1`, tensors `mtp.0.*`) — a full Block (MLA attn + 256-expert MoE, **no** tier-B2 compressor) + the fusion `x' = e_proj(enorm(embed(next_id))) + h_proj(hnorm(x))` → block → **shared** head. It was unloaded (`n_layers=43` stops before it; stager skipped `mtp.*`). **Scaffolded (compile-validated; env-blocked for run-validation):** `DS4F_STAGE_MTP` ungates staging (experts EP-sharded), `DS4F_MTP` loads it (`m->mtp` + `mtp_*` fusion tensors), `ds4f_mtp_predict()` is a documented STUB. Off-path byte-identical (regression unchanged).
**Follow-on (focused fresh-alloc effort), in order:** (1) extract the M=1 block-forward so `ds4f_mtp_predict` can run (fusion → `m->mtp` block → `m->head` with `mtp_hc_*`/`mtp_norm`); validate the MTP head's next-token accuracy vs the main model. (2) Draft/verify/accept loop: MTP drafts K, main model verifies **M=K** in one pass, accept the longest matching prefix, **roll back rejected tokens' KV** (ring + cmp/idx append). (3) The verify needs the **M=K batched-decode rework** of the per-token path (attn/tb2/KV-append/CP-gather) — the dominant effort. **Caveat (decide target ctx first):** spec/batched amortizes comm+dense ⇒ a **moderate-ctx win (~10K–1M)**; at 12M the per-position O(T) scan dominates and does **not** amortize (K draft tokens = K×O(T) scan), so it's **not a 12M lever** — there the lever is a cheaper/coarser scan.

**★ MEASURED on `glm5-2` (2026-07) — gamma=1 spec decode is a NET LOSS vs the optimized decode.** Real-weight 11n, bf16 dense (`DS4F_FP8_BF16=1`, MHC+Tier-B2, ctx≈15+64), `DS4F_MTP=1 DS4F_SPEC=1 DS4F_SPEC_BATCH=1`: MTP accept **75%** (27/9), **1.391 tok/forward** (46 fwds/64 tok), output coherent — but **decode 10.22 tok/s vs 13.26 baseline (−23%)**. Root cause: the memory's old 1.21× spec win was vs an **8.69 tok/s comm-DOMINATED** decode (13.1 ms/tok comm); the landed decode levers (FLAGBAR/ATTN_GEMM/IDX_GEMM/NUMA) since made plain decode **13.26 tok/s with comm only ~16%**, so (a) there's little comm left to amortize and (b) the verify's **per-position tb2/attn scales with K and never amortizes**, while the baseline gets the optimized matvec path the verify GEMM doesn't. gamma≥2 wouldn't rescue it at short ctx (K× the per-position tb2/attn) and is *worse* at long ctx (K× the growing O(T) scan). **Conclusion: MTP spec decode is dominated by the optimized single-stream decode — not a decode-speed lever here.** The MTP module + `ds4f_mtp_predict` remain useful for a *throughput* (batched multi-request) path, not single-stream speed.

### 2026-07-10 session (alloc 49500509, 12-node interactive): prefill 23.3 → 29.3+, decode 13.9 → 15.6 tok/s

Real-weight 11n A/B ladder (2409-tok prompt, MAX_NEW=64, gen config = REAL+FP8_BF16+Q8_DENSE+TIERB2+MHC+
HC_PAR+HC_RMSPAR+`DS4F_PREFILL_GEMM=1`; per-lever isolated runs). New decode/prefill sub-timers:
`mhc_pre/mhc_post/mhc_cpy/comm` (DS4F_NPHASE 20→24) + a per-phase PREFILL profile print in the runner.

| run | change | prefill tok/s | decode tok/s | verdict |
|---|---|---|---|---|
| A | baseline (K=32) | 23.30 | 13.90 | decode: other 22.7 = comm 12.0 + **mHC 10.7**; dense 22.2; tb2 11.8 |
| B | +`DS4F_PF_TP=1` | 24.72 | 13.9 | verify compute-shard (below); +1 [K,C] reduce ate most of it at K=32 |
| C | +`DS4F_HC_SVE=1` | 25.89 | **15.56** | **mhc_pre 10.1→2.67 ms** (SVE hcmix + SVE sinkhorn + fused resid) |
| D | +`DS4F_Q8_LOCAL=1` | 25.8 | 15.65 | **NEUTRAL — REFUTED**: reader-local mbind of the q8 buffers changes nothing (placement is not the dense-matvec lever; joins prefetch in the refuted pile) |
| E | +`DS4F_Q8_GEMM_TILE=16` | 25.5 | 15.63 | **NEUTRAL on speed** (verify GEMMs not GEMM-rate-bound at K=32); *numerics improve* (W8A16-like, relL2 8.9e-3→5.0e-3) |
| G | +SVE batched mHC, K=64 | **29.32** | 15.60 | prefill mhc_pre 4.97→1.41; comm 5.9→5.5; **tb2prep 11.3 (33%) is now the prefill wall** |
| H | +`TP_AR_A2A=1`, K=128 | 28.74 | 15.60 | **a2a NEUTRAL** (comm is skew, not exchange latency); K=128 < K=64 (payload-bound) |
| I | +`TP_HEAD/TP_EMBED/MV_FUSE`, K=64 | **29.50** | **16.11** | head 2.0→argmax-merge; RSS 21.9→**20.0 GB** |
| J | +`DS4F_CMP_LOCAL` (2026-07-10b, fresh alloc 49508574) | 29.19 | **17.35** | cmp_matvec 4.6→1.5 ms, tb2lcmp 5.2→2.8; **BYTE-IDENTICAL**; the shipped config |

**Session net: prefill 23.30 → ~29.5 tok/s (+27%), decode 13.90 → 17.35 (+25%), RSS −1.3 GB** — all
real-weight 11n, NaN=0, lockstep. Launcher defaults (`run_ds4f_agentic_11n.sh`, `run_ds4f_serve_11n.sh`
fast path): `DS4F_HC_SVE=1 DS4F_PF_TP=1 DS4F_PREFILL_K=64 DS4F_MV_FUSE=1 DS4F_CMP_LOCAL=1`
(+`TP_HEAD/TP_EMBED=1` in serve).

**Next-session decode levers** (post-CMP_LOCAL decode 57.6 ms = comm 12.4 + tb2prep 7.5 + o_proj 8.7 +
qkv 6.8 + shared 5.7 + attn 5.2 + experts 4.7 + mHC 3.1 + head + misc):
1. **per-row-scale int8 dense rep — REFUTED at the bench** (see below, "per-row-scale int8 dense").
   v3 fast-but-lossy (1.38×, spike relL2 0.106 vs 0.030), v4 safe-but-no-gain. The int8 dense decode
   kernel is at its practical ceiling for the model's massive-activation fidelity.
2. **tb2lcmp — the reader-local win LANDED (`DS4F_CMP_LOCAL`, commit `bd742c0d`, decode +6.6%,
   BIT-EXACT).** Attribution (`tools/cmp_bench.c` + `g_cmp_mv_secs`): tb2lcmp's 5.2 ms is 89% the
   compressor matvec (4.6 ms in-model), which is **BW-bound AND NUMA-interleave-penalized** — the
   ds4f_cmpmv_bf16 matvec is small-W (1024/256 rows), under-saturating, and under MPOL_INTERLEAVE
   streams cross-CMG at ~150 GB/s. Reader-local page placement (`ds4f_cmp_place_local`: reader-rowsplit
   first-touch + `mbind(MPOL_LOCAL)`) recovered it to ~450 GB/s effective → cmp_matvec 4.6→1.5 ms.
   *(The initial "bounded ~1 ms, not pursued" call was WRONG — I mis-analogized to the refuted DS4F_Q8_LOCAL,
   forgetting the compressor is BW-bound while the dense matvec is issue-bound; the g_cmp_mv split at the
   real run showed the NUMA penalty and the reader-local fix delivered 3× the estimate.)* Residual: the
   ~0.6 ms serial softmax tail (per-`e`-independent → parallelizable, bit-exact) is the only bit left,
   ~+1% — low value.
3. **batched decode serve integration** (below) — comm+dense amortize only there. **The only material
   single-node lever left:** v3 int8 (refuted, bench), tb2lcmp (LANDED via CMP_LOCAL +6.6%), and comm
   (architectural) confirm single-stream decode at ~17.4 tok/s is near its floor — the remaining
   throughput is batched/serve (comm+dense amortize across M).

**★ per-row-scale int8 dense (lever 1) — REFUTED at the bench (2026-07-10, `tools/q8_mv_bw.c`).** The
pinned ~390 Gmac/s sdot ceiling is the per-64-block scale APPLICATION (8 fp16→f32 + 8 svcvt + 8 svmla
*per block*). Two prototypes:
- **v3** (per-row weight scale + **per-token** activation scale + full-int32 K-accumulation, no per-block
  scale ops in the inner loop): **538 Gmac/s = 1.38×** — the speed is real and structural. BUT the
  per-token activation scale is too lossy: a spike test (one channel N× the rest) gives relL2
  **0.106 vs production 0.030 at N=100** — the per-token scale zeros the O(1) channels while they still
  contribute ~10% of the output. Real RMSNorm'd dense inputs plausibly hit this 10–1000× "danger band"
  (residual-stream massive activations, the same ones that force bf16-not-fp16 KV). A quality gamble.
- **v4** (per-row weight scale + **per-64-block** activation scale, the SAFE version): **389 Gmac/s =
  ZERO speed win** (identical to production) — because the per-block svcvt+svmla is the actual
  bottleneck; simplifying only the *weight* scale buys nothing. Accuracy exactly matches production at
  every spike.
So the safe path gives no speed and the fast path needs an activation-fidelity gamble the spike test
flags as real → **not wired** (bench-level refutation, no alloc run, same discipline as the prefetch/OP
refutations). The int8 dense decode kernel is at its practical ceiling for the per-block activation
fidelity the model's massive-activation channels require.

**Batched decode re-measured (dbbench, bf16 dense + PF_TP + HC_SVE + TP_HEAD/EMBED, ND=24):**
aggregate tok/s M=1/2/4/8/16 = **13.0 / 18.5 / 24.4 / 29.0 / 32.4** — vs the prior 9.3/14.1/19.3/23.0/25.7:
**+26% at M=16** (and M=1-via-verify 9.3→13.0, +40%). PF_TP shards the batch-path shared GEMM
(~1 ms/step flat at every M); the per-M dominators are now **experts 94.5 ms/step @M=16** (5.9/seq,
MoE top-6 barely shares experts) and **tb2prep 124.9** (7.8/seq per-position). Serve continuous-batching
integration (P2 build plan) is the remaining wiring to expose this as multi-request throughput.

**Landed (this session, uncommitted):**
- **`DS4F_HC_SVE`** (default off): SVE half-row hcmix (110→~35 µs/call, reassoc/coherent-class), SVE
  sinkhorn divisions (24→5 µs, **bit-exact**, `tools/mhc_bench.c` 16/16 comb bit-equal), resid-copy fused
  into the hccol collapse (bit-exact), SVE `hcmix_b` + pooled per-position sinkhorns in `hc_pre_batch`
  (bit-exact pooling). Decode +12%; gen coherent, NaN=0, lockstep (ids diverge from baseline = reassoc class).
- **`DS4F_PF_TP`** (default off): memory-neutral COMPUTE-shard of the verify-path shared+o-proj GEMMs by
  rank-slicing the REPLICATED tensors (row_slice at GEMM time; decode path untouched). Q8-safe per
  `tools/ws7_tp_q8_test.c`; 64-aligned contraction boundaries via new `DS4F_TP_DENSE_ALIGN` (fixes a real
  Q8 zero-pad straddle bug in the load-time TP_SHARED/TP_OPROJ alignment: relL2 1.5e-3 → 1.5e-7).
- **WS7 RECLASSIFIED — no kernel bug** (`tools/ws7_tp_q8_test.c`): TP_ATTN×Q8 partial-sum error is
  fp-reassociation (relL2 1.5e-7, same class as bf16's 2.8e-7 which is also NOT bitwise); the 11n bf16
  "64/64 identical" was token-level luck. Row-shard+full-input = BIT-EXACT (8192/8192). Gate TP×Q8 changes
  on coherence, never token-identity. WS6 confirmed fixed in-tree (fp32 xscale; stress NaN=0, 205/205).
- **`DS4F_Q8_GEMM_TILE`** (M-threshold, default off): int8→bf16-pv fused tile-dequant GEMM. Found: the Q8
  svdot GEMM is **FLAT ~104-136 Gmac/s M=1→64** (never amortizes, like MXFP4 svtbl); the tile removes the
  activation quantization (more accurate) but was speed-NEUTRAL in the verify path (not GEMM-rate-bound there).
- **`TP_AR_A2A`** (`tp_allreduce.h`, default off): direct all-to-all sum for count ≤ `TP_AR_A2A_MAX` (8192)
  — one detection wait instead of ~5 sequential exchanges; rank-order fold = bitwise-identical across ranks;
  dedicated 2-generation slot region. Integer-exact validated 11n (`tp_ar_ack_test` sum_mism=0). Microbench:
  a2a 50.8 vs doubling 40.8 µs synchronized. **Run H (production A/B): NEUTRAL — decode comm 12.1 ms
  unchanged.** Verdict: decode "comm" is straggler-skew + floor, not exchange-count latency; a2a does not
  absorb skew (the slowest sender gates either way). D3 closed: the ~12 ms is architectural at M=1 (only
  batched decode amortizes it). K=128 prefill also NEUTRAL-to-worse vs K=64 (payload-bound reduces): **K=64
  is the chunk sweet spot.**
- **int8-sdot matvec ceiling PINNED (`tools/q8_mv_bw.c`)**: reader-local first-touch pool, production
  dispatch: **bf16-pv 735 GB/s (0.091 ms) vs q8-sdot 402 GB/s at the SAME 0.086 ms wall** — the sdot
  kernel is ISSUE-bound ~390 Gmac/s (half the bytes, zero time win); production decode dense is AT this
  kernel ceiling, so placement levers (Q8_LOCAL, prefetch) are structurally neutral. A `svmla_lane`
  restructure (v2, bit-exact) measured 2× SLOWER — refuted. The real ≥1.5× dense-decode lever is a
  **per-row-scale int8 rep** (full-int32 K-accumulation: 17 vs 41 instrs/block; no overflow at K=4096) —
  LOSSY (coarser than per-64), needs repack + kernel + quality gate. Follow-on work.
- **Fixes**: `ot[8]` out_tok overrun at K>8 in the gen prefill loop (→ `ot[128]`); `DS4F_PREFILL_K` clamp
  32→128 (`pa[128]` sinkhorn arrays + K>128 abort in `ds4f_forward_verify`); `DS4F_MAX_MTILE` unchanged.
- TP_HEAD/TP_EMBED are set by `--preset decode` but NOT by the `run_ds4f_gen_11n.sh` env path — the gen
  config leaves the +2.3% (and −1.9 GB) on the table; head = 2.0 ms of the 64 ms decode.

**Prefill wall after G (34.1 ms/tok):** tb2prep 11.3 (scan 2.7 + attn 3.6 + lcmp 2.4 + glue) per-position,
comm 5.5 (payload-bound at K=64), experts 5.4 (svtbl floor at per-expert M≈1.5), qkv 4.2 (replicated wq_b),
o_proj 3.2 (replicated wo_b contraction), mhc_post 2.1. The doc's earlier "compressor-batching neutral"
diagnosis holds here too (the lcmp cost is dispatch+serial state glue, not the matvec).

### Batched concurrent decode — roofline (2026-07) + build plan

**The throughput lever** (decode M independent requests in one forward). Per-phase decode profile (`DS4F_PROF=1`, bf16 dense, ~74 ms/tok) splits into: **amortizable ~65 ms** — dense GEMM weight read *once* for M sequences (mHC 22.6, qkv 7.2, experts 6.0, shared 5.9, head 2.0, router 0.5, tb2 dense projections 9.3) + comm ~12 (one all-reduce/step ÷M) — and **per-sequence ~8.5 ms** (tb2scan 6.9 O(T), attn 1.3, rope/topk 0.3). **89 % amortizable**, so `step(M) ≈ 65 + 8.5·M` and aggregate tok/s = M/step:

| M | 1 | 4 | 8 | 16 | 32 | →∞ |
|---|---|---|---|----|----|----|
| **agg tok/s** | 13.5 (==baseline) | ~41 (3.1×) | ~60 (4.5×) | ~80 (6×) | ~96 (7×) | ~119 (**~9×**) |

Best at short-moderate ctx (tb2scan grows with T → lower ceiling long-ctx) — exactly the serving regime.

**★ MEASURED (P2/P3, `DS4F_DB_BENCH`, bf16 dense, MAXPOS 2048, cold-start): the projection was OPTIMISTIC.** Aggregate tok/s: M=1 9.3, M=2 14.1, M=4 19.3, M=8 23.0, **M=16 25.7** (= 2.8× within the verify path, **~1.9× over the best single-stream matvec decode (13.3)**). The measured per-M increment is **~34 ms/seq, not the projected 8.5** — the roofline was wrong that `tb2prep` amortizes: only the indexer qproj (`tb2qproj`) is batched before the per-position loop; the **compressor (lcmp ~4.6) and indexer (icmp ~1.3 + scan) run PER-POSITION inside the loop**, so they scale with M. Also M=1 via verify (9.3) < single-stream matvec (13.3) — the verify GEMM path is slower per-token (same reason spec decode lost). So batched decode is a **real but modest ~1.9× throughput win** at M=16, not 9×.

**★ Compressor-batching optimization — MEASURED NEUTRAL, REVERTED (2026-07).** Hoisted the Tier-B2 compressor matvec (`cmp_wkv/wgate @ p_hn`) out of the per-position loop into one M-batch GEMM (fed per-position via `cmp_pre_kv/score`), mirroring `tb2qproj`. Result: **M=16 25.6 vs 25.7 tok/s — no change** (step 625.0 vs 623.5 ms, within noise). Diagnosis: the compressor *matvec* is only ~µs/position (an ~8 MB weight read at ~700 GB/s ≈ 11 µs), **not** the profile's 4.6 ms `tb2lcmp` — that number is not the matvec. The per-M ~34 ms/seq is **dispatch-bound**, not compute-bound: each per-position iteration fires multiple **48-thread pool barriers** (attention + O(T) indexer scan + compressor state update), and that dispatch count scales with M×positions. Batching the *compressor compute* removes neither the attention nor the indexer-scan dispatches, so throughput is unchanged. Reverted the change (neutral complexity). **The real lever = batch the per-position *workers* (attention + indexer scan) across M into one dispatch each** — a much larger restructure of `ds4f_attn_tb2_worker`/`ds4f_idxsc*_worker` to process all M sequences per dispatch. Until then ~1.9× (M=16) is the practical ceiling.

**★★ Full verify-path per-M profile (2026-07) — REFUTES the dispatch-bound premise; the ~1.9-3× ceiling is COMPUTE-bound, not batchable.** The verify path only instrumented tb2+attn (dense sections were untimed); adding coarse section timers (`DS4F_P_QKV/OPROJ/SHARED/EXPERTS/HEAD`, gated on `ds4f_prof_on`, verify-only) revealed the tb2 subtimers are only **~6.9 of the ~34 ms/seq** increment — **~80% is the untimed dense/expert path.** Per-seq increment (bf16 dense, real weights, M=1→8, ÷7):

| section | ms/seq | share | batchable? |
|---|---|---|---|
| **experts** (routed+shared MoE) | 12.1 | 36% | only at large M — sparse top-6 → ~1 tok/expert until M≫16 (M=16 gives only +10% over M=8: 23.4→25.7) |
| **dense GEMMs** (qkv+o_proj+shared) | 13.1 | 39% | already batched M=K; scales = **real per-token FLOPs** (weight read already amortized once/K rows) |
| **tb2prep** (the restructure target) | 7.5 | 22% | ~1.5/seq serial glue + a few dispatches; rest = real O(T) scan + per-token matvec compute |
| head | 0.8 | 2% | — |

So the earlier "dispatch-bound → batch the tb2 workers" hypothesis is **wrong**: ~85% of the per-seq cost is genuine compute (dense GEMM FLOPs + un-amortized sparse experts + O(T) scan), only ~15% is batchable dispatch/glue. **Batching the tb2 workers would reclaim ~8-12% of per-seq → low-single-digit aggregate gain, for a large high-risk rewrite of the bit-exact kernels — not worth it.** The genuine throughput lever is **larger M** (weight-read amortization), which *already exists* (aggregate 9.3→23.4 tok/s at M=8 = 2.5×), but the MoE experts barely amortize even at M=16 (only +10%) because top-6-of-256 routing rarely shares an expert until M is very large — so the batched-decode ceiling is a genuine **~2.5-3×** (consistent with the prior committed M=16=25.7), *not* the 9× the roofline projected. Also fixed a real bug: the DB_BENCH sweep set `dec_batch_seq=NULL` between M values without freeing the per-sequence cache sets (`ds4f_free_decode_batch`, ~29 MB × (M-1) leaked each iteration → swap-thrash risk at high M). CAVEAT: the M=16/32 confirmation run was blocked by a persistently degraded alloc this session (repeated stage sig9 / PLE-0054 / silent early stage death; `/local` blobs left truncated at ~1.5 GB — a fresh alloc + re-stage is needed to re-run). The M≤8 profile + prior M=16=25.7 already trace the diminishing curve conclusively.

**Build.** `ds4f_forward_verify` already batches *all* dense/MoE/comm as an M=K GEMM with one all-reduce/layer — the amortization exists. The only rework is its per-position attn/tb2 loop → **per-sequence**: M independent live KV/cmp/idx cache **sets** (the multi-slot infra has per-slot *snapshots* but one *live* set), each batch element k with its own `pos[k]` + `cache[k]`. Lowest-risk approach: **pointer-swap** — allocate M cache sets, and before element k's append/tb2/attn, point the layer's cache fields (the ~20 buffers/scalars `ds4f_ctx_snap` enumerates: `kv_*`, `cmp_*`, `idx_*`, the compressor ring states, calibration) at set k, so the existing `ds4f_tb2_prepare`/`ds4f_attn_tb2_worker`/KV-append run unchanged. Phases: **P1** M cache sets + `ds4f_forward_decode_batch` (validate M=1 == single-stream, M=2 == two independent streams); **P2** serve-loop dynamic batch (add/remove sequences, per-sequence sampling/EOS, continuous batching); **P3** measure aggregate tok/s vs the projection. MTP module is staged (throughput draft is a later compose).

### LANDED — DS4F_SERVE_BATCH concurrent batched-decode serve (2026-07-10b, commits `782c8f29`/`eb5d9054`/`994cbd27`)

The batched-serve path is built + validated. `ds4f_serve_batch_loop` (`ds4f_ep_runner.c`, gated
`DS4F_SERVE_BATCH>1`, **default 1 = the existing single-request loop, zero production impact**): B
persistent per-sequence cache bundles (`ds4f_alloc_decode_batch`, allocated once), each request
prefilled into its own bundle (PREFILL_GEMM chunk, `dec_batch_seq=NULL`), then all still-active
sequences decode-stepped TOGETHER via one `ds4f_forward_verify`/step. Greedy, per-seq EOS/max_new,
per-sequence independent. Static batching (a batch drains before the next admits; dynamic mid-flight
admission = the follow-on). Protocol `BATCH N`+(max_new,ids)×N → `N`+ids×N; `ds4f_serve.py` gains a
dispatcher thread that coalesces concurrent HTTP requests within `DS4F_SERVE_BATCH_WINDOW` (30 ms).

**★ Found + fixed a PRE-EXISTING per-sequence independence bug** (commit `782c8f29`) — it also fixes
`DS4F_DB_BENCH` correctness, and is NOT one of this session's levers (HC_SVE=0 A/B gave identical
divergence). `ds4f_pf_qnr_worker` RoPE'd query element mm at `pos0+mm` (right for prefill's consecutive
positions) but in DECODE-BATCH each mm is an independent sequence at `dec_batch_pos[mm]`; the KV RoPE
(inline per-k) already used the right position, so Q/KV rotated inconsistently and two identical
sequences at different batch indices diverged. Localized with a per-(layer,k) hn/q/attn checksum dump
(layer 0: hn identical, q differed → RoPE). Fix: `rpos = dec_batch_pos ? dec_batch_pos[mm] : pos0+mm`.

**Validated real-weight 11n (direct shared-FS protocol):** per-sequence independence (A alone == A in
[A,B,C]; 4 identical prompts byte-identical), coherent (a Fibonacci prompt yields correct Python:
`a,b=0,1 / while a<n: print(a,end=' ') / a,b=b,a`), **aggregate N=4 = 23.4 tok/s vs N=1 = 14.9 (1.57×)**
— dbbench projects ~2.5-3× at N=8/16. The HTTP frontend dispatcher was confirmed to coalesce 4 concurrent
curls into one correct BATCH request (the runner + protocol are separately validated, so the halves
connect). **Follow-ons:** dynamic mid-flight admission (a fast request currently waits for its batch to
drain); per-sequence sampling (currently greedy-only); B>4 throughput measurement on a stable alloc.
*Ops note: `pkill -9` on the runner is unreliable — verify the `SERVE-BATCH ready: B=N` banner matches
the launched B and `ps -eo cmd | grep build/ds4f_ep_runner` is empty before relaunching.*

### LANDED — DS4F_SERVE_DYNAMIC continuous batching / mid-flight admission (2026-07-10b, commit `9ea8f684`)

`ds4f_serve_dynbatch_loop` (`ds4f_ep_runner.c`, gated `DS4F_SERVE_DYNAMIC`, **default off**) admits new
requests into free slots *as they open* instead of the static "prefill N, drain to completion" model —
a fast request no longer waits behind slow batch-mates. B persistent bundles; each free slot is filled
from the queue (single-seq prefill) and the active set decode-steps together via one `ds4f_forward_verify`.
Queue protocol on the shared FS: `q.<id>` (client-written `max_new\nids`), `r.<id>` (runner response,
temp+rename atomic). `ds4f_serve.py infer_dynamic` routes when `DS4F_SERVE_DYNAMIC=1`.

**Two cross-node-FS robustness fixes were mandatory (both cost a real debug cycle):**
1. **Lockstep admission via broadcast, not per-rank read.** Rank 0 reads the `q.<id>` payload and
   BROADCASTS `(ok,mnew,np,ids)` via `ar_cb`; a per-rank read races the FEFS/LLIO cache (rank 0 sees the
   fresh file, rank 7 doesn't → divergent `np` → the prefill all-reduce **deadlocks**). Manifested as a
   hard hang the moment a request arrived *after the loop drained to idle* (peak cache skew) — the burst
   case worked by luck. `int` ids are float32-exact (vocab ≪ 2²⁴).
2. **Probe `q.<qnext>` existence; do NOT read a `qhead` counter.** A counter file changes value at the
   same 2-byte size (`"1\n"→"4\n"`); the compute-node client caches it by size+coarse-mtime and **never
   refetches** → rank 0 reads a stale `qhead` forever and stops admitting (silent stall, not a hang).
   New-*file* existence IS coherent, so probe `q.<qnext>` directly (the broadcast `ok` flag stops the
   loop). This made a *counter* the wrong signal and *file existence* the right one. Consequence: clients
   must write `q.<id>` **atomically (temp+rename)** or the probe reads a half-written file → `np=0` →
   empty response. The old `qhead` gate had doubled as the write-complete barrier.

**Validated real-weight 11n (alloc 49508574):** token-exact per-sequence independence — A alone ==
A admitted mid-flight alongside B,C (**40/40 tokens, 5/5 runs**); no deadlock/stall across repeated
idle→admit transitions; runner-internal **22.8 tok/s aggregate at active=4** (≈4× single-stream, no
regression from the probe rewrite). Client-observed end-to-end tok/s is lower and FS-round-trip-bound
(~fixed tens-of-seconds q/r visibility latency across login↔compute) — a property of the file protocol,
not compute; a socket transport would remove it. **Follow-ons:** socket/shared-mem transport to cut
the file-visibility latency; B>4 dynamic throughput on a stable alloc.

### LANDED — per-sequence sampling in the dynamic loop (2026-07-11, commit `7b18d816`)

DS4F_SERVE_DYNAMIC decode was greedy-only; now each admitted sequence carries its own sampler
(temperature / top_p / top_k / repeat+presence penalty / seed) + its own SplitMix64 PRNG, so one batch
can mix greedy and sampled sequences and each draws independently. `temp<=0` stays greedy (the cheap
argmax-merge head, bit-identical). Pieces: (1) `ds4f_sample` split into `ds4f_sample_logits(lg,V,sp,
*rng,hist,nhist)` — a pure fn on an explicit logits row + explicit PRNG state; (2) **`ds4f_forward_verify`
head gains a `want_full_logits` branch** — under TP_HEAD the head is vocab-sharded, so it scatters each
row's owned `[K,hrows]` shard into a full-vocab row (zero-fill) + `ar_cb` SUM → every rank holds the
SAME full `[K,vocab]` logits (new `p_logits_full` buffer, aliases `p_logits` when the head is
replicated); (3) the admit broadcast payload carries the sampler params + seed so every rank derives an
IDENTICAL sampler (else ranks draw different tokens → divergence); `want_full_logits` is set per-step
(any active seq samples) and only on the LAST prefill chunk (first-token draw). `q.<id>` line 1 extended
to `max_new temp top_p top_k seed rep_pen pres_pen` (params optional → greedy). seed 0 ⇒ derived from
the request id (reproducible per id, distinct across concurrent requests).

**★ Bug found + fixed en route (the instructive one):** the `want_full_logits` branch was first added to
`ds4f_forward_prefill`'s head — but the dynamic loop calls `ds4f_forward_verify`, a DIFFERENT function
(~`ds4f_impl.h:5410`) whose head still argmaxed only → `p_logits_full` stayed NULL → `ds4f_sample_logits(
NULL+offset)` SIGSEGV on the first sampling request. Localized with per-rank file traces
(`/home/u14346/sampdbg.r<rank>`, gated `DS4F_SAMP_DBG`): all ranks logged `wfl=1` + `pf_full=(nil)` and
the head trace never fired → wrong function. **Lesson: there are TWO batched heads (`_prefill` M-rows and
`_verify` K-rows); the serve/decode path is `_verify`. Per-rank stderr is NOT forwarded to the launcher
log (only plexec's `sig=11` line is) and the scratchpad dir isn't mounted on compute nodes — write debug
to `$HOME`.**

**Validated real-weight 11n (alloc 49526204, `DS4F_TP_HEAD=1` sharded head):** greedy determinism +
token-exact independence unchanged (40/40); sampling reproducible by seed, diverse across seeds, ≠ greedy,
coherent; per-sequence independence under sampling (A[seed X] admitted mid-flight with B[seed Y] == A[seed
X] run alone; same-seed seqs in one batch identical; different seed differ). Sampling requests *completing*
is itself the lockstep proof (mismatched draws would deadlock the per-layer all-reduce). Cost: the full
`[K,vocab]` SUM (~2.4 MB at K=4) per decode step only when a sampling seq is active — same per-seq cost as
single-stream sampling. Static-batch parity landed too (commit `01f43ce8`): `ds4f_serve_batch_loop` gains
the same per-seq sampler (BATCH per-seq line `max_new [temp top_p top_k seed rep_pen pres_pen]`), validated
B=4 (greedy regression clean, mixed greedy+sampled batch, seed 1234 yields the SAME tokens as the dynamic
loop — identical seed derivation).

### MEASURED — batched-decode throughput saturates ~25-30 tok/s (B sweep, 2026-07-11)

Dynamic-loop runner-internal peak `tok/s agg` (early-context, N=B concurrent 80-tok requests, alloc 49526204):

| B  | peak agg tok/s | vs single-stream (~17) |
|----|----------------|------------------------|
| 4  | 22.8           | 1.34×                  |
| 8  | 25.8           | 1.52×                  |
| 16 | 25.3           | 1.49×                  |
| 32 | 29.4           | 1.73×                  |

**Aggregate throughput saturates at ~25-30 tok/s across B=8..32 — heavily sublinear (only 1.29× from
B=4→B=32).** The amortizable base (dense GEMM + per-layer EP reduce + mHC) is fully amortized by B≈8; beyond
that the **per-sequence tier-B2 work does NOT amortize** — each sequence has an independent KV/compressor/
index cache, so its attention + `tb2_prepare` cost grows ~linearly with B and dominates. This corrects the
`DS4F_DB_BENCH` ~2.5-3× projection (dbbench measured the dense-amortized regime without the full per-seq
mHC+tierB2 attention loop). Throughput also DECAYS within a run as context grows (B=32: 29.4→25.1 over the
window sweep) since per-position attention scales with KV length. Independence held at every B. Per-window
numbers are noisy (peak depends on exact ctx at measurement) but the saturation is robust. **Practical
sweet spot ≈ B=8-16** (throughput plateau + lower per-request latency + 8-16 cache bundles vs 32); B=32
buys little for 2× the bundle memory. The real lever beyond this is per-sequence attention cost (CP shards
the caches; a cheaper long-ctx attention), not larger B. Client-observed tok/s (10-14) WAS FS-round-trip-
bound (the file protocol); the socket transport below removes that.

### LANDED — DS4F_SERVE_SOCK TCP transport (2026-07-11, commit `c856ce83`)

The file protocol's admission/response latency is dominated by **cross-node FEFS/LLIO cache visibility**
(~tens of seconds: login-node client writes `q.<id>`, the compute-node runner sees it only after the
attr/dentry cache refreshes; `r.<id>` propagates back the same way). **TCP over the Tofu IP interface**
(the nodes have `tofu0`/`tofu1` 10.x IPv4 addrs) between the login node and the runner's rank-0 compute
node is **0.7 ms round-trip** (measured with a throwaway `mpiexec -np 1` python server on one compute node
+ a control-node connect). So rank 0 hosts a non-blocking TCP listener and publishes `<ip> <port>` to
`<base>.sock`; the frontend connects per request. Frame = `[u32 BE len][payload]`; request payload is the
SAME 2-line text as a `q.<id>` file (so per-seq sampling flows through unchanged); response is `gen ids\n`.
Only rank 0 touches sockets — the parsed request is broadcast to the EP group via `ar_cb` exactly like the
file path, so **lockstep is unchanged**. Falls back to file mode if the listener fails. Frontend
`ds4f_serve.py infer_socket` (routed when `DYNAMIC && SOCK`).

**Result — single-request serving is now COMPUTE-bound, not FS-bound.** Real-weight 11n (alloc 49526204):
40-tok greedy request **3.33 s** end-to-end (83 ms/tok) vs **~30 s** under the file protocol; 1-token
request **596 ms** (prefill-bound); 4 concurrent **9.36 s = 17.1 tok/s** end-to-end (vs file's 6.4);
per-sequence independence (A concurrent == A alone) + sampling reproducibility preserved; runner-internal
22.7 tok/s at active=4 unchanged (socket admission is free). **The ~tens-of-seconds FS-visibility floor is
gone.** *Ops note: the throwaway-server trick (`socksrv.py` + `vcoord_1.txt` = one line of `vcoord_ds4f.txt`)
is the fast way to test cross-node TCP without the runner; debug/aux files for compute nodes go under `$HOME`
(the scratchpad dir isn't mounted there).*

### Resuming prompt — Phase 2 CP (next session)
> **STATUS UPDATE 2026-07-11:** the CP code is well past the Stage-A checkpoint the prompt below describes.
> In-tree now (`ds4f_impl.h`): `ds4f_cp_slot_shard` (`:185`); storage sharding `DS4F_CP_SHARD` for `cmp_q4`
> (`:2173`, `cp_t0/cp_t1`) and `DS4F_CP_IDX` for `idx_kv8_4` (`:2219`, `idx_cp_s0/s1`); the sharded-read
> paths (`:3510`, `:4224`); the sharded index scan wired with `idx_cp_on` + `ar_cb` (`:4007`); and the
> **CP idx-shard top-k merge** (`:1920`). **Validated functional this session** (alloc 49526204): a full CP
> gen — `DS4F_CP=1 DS4F_CP_SHARD=1 DS4F_CP_IDX=1 DS4F_INT4_CMP=1 DS4F_IDX_INT4=1 DS4F_FP8_BF16=0` — ran
> **rc=0, NaNs=0, coherent** (correct quicksort completion) at short ctx (max_pos=120; the memory/ctx-ceiling
> win only shows at large `max_pos`, NOT tested here — long-ctx validation risks OOM-killing the shared
> interactive alloc, so do it deliberately on a dedicated alloc). **REMAINING:** (1) confirm/finish the
> Stage-C attention *combine* (whether the selected-latent attention distributes compute with a cross-node
> `{m,l,acc}` online-softmax combine, or currently shards storage + gathers for replicated compute — read
> `ds4f_attn_tb2_worker` + the `ds4f_forward_token` combine); (2) long-ctx A/B: per-node MemFree + coherence
> at max_pos = 32k/64k/128k, CP-on vs the CP-off ceiling. NOTE the plan file
> `~/.claude/plans/see-a64fx-ds4f-md-...md` is GONE — reconstruct design from the in-tree code above + the
> git history (`git log --oneline | grep -iE 'CP |attn-CP|slot-shard'`; some CP commits are glm5/m3, not ds4f).
>
> **(original Stage-A prompt, kept for the design detail):** Stage A DONE+committed (`1f7d46a`): `tp_allreduce_max` (`tp_allreduce.h`) + `ep_armax_callback`/`m->ar_max_cb` (`ds4f_ep_runner.c`) + `DS4F_CP_SELFTEST` — validated 11-node (all ranks PASS bad=0 worst=0). **NEXT = Stage B** (slot-shard `cmp_q4`/`idx_kv8_4` by `[s0,s1)`, sharded `ds4f_idxsc8r4_worker` scan, top-k merge via zero-fill+`ar_cb`-SUM), then **Stage C** (partial `{m,l,acc}` refactor of `ds4f_attn_tb2_worker` + the combine `ar_max_cb`(m)+`ar_cb`([l|acc]) in `ds4f_forward_token`).
> **Standing rules:** native fcc/FCC; in-alloc `mpiexec` (no pjsub) NP=11 EXCLUDE node 0; **measure MemFree not RSS**; validate coherence/lockstep (NOT bit-exact — combine reassociates); FP8 dense (`DS4F_FP8_BF16=0`) required; CP composes with full TP (`DS4F_TP_*`) + int4 cmp/idx (`DS4F_INT4_CMP`/`DS4F_IDX_INT4`), all default-off; commit only when asked; one real-weight gen per Bash call (batched jobs blow the 10-min timeout → SIGKILL degrades PMIx → recover via native re-stage `run_ds4f_stage_11n.sh`).
> **Cumulative ceiling: ~255k → ~4.8M (~19×)** committed (2p `981350a`, 2q `cc86b0f`, 2r `683cfaf`/`7961f1a`/`5614dea`/`2c00f58`, 2s `f9fb3ff`, 2t `23b793b`); CP targets tens-of-millions + ~20–30 tok/s at long ctx (see roofline above — spec/batched decode is the speed lever, CP shards the O(T) scan that otherwise caps long-ctx decode at ~1 tok/s).

## Prefill throughput → 200+ tok/s (11–12 nodes): roofline + plan

Batched prefill (Step 2o, `DS4F_PREFILL_BATCH` + MXFP4 tile-dequant) = **~39.5 tok/s** real-weight 11-node (2.3× over token-at-a-time 17.27). Roofline below: the wall is the **replicated dense** (batched prefill *aborts under TP*, so every node recomputes the full dense — ~59% of cluster prefill wall = qkv 19% + o_proj 29% + shared 11%; **attention is only 3.9%** at the cluster, so "attention-as-GEMM" is NOT the lever — the old 37% was a single-node synthetic config).

### Peak prefill tok/s — the math
A64FX fp32-FMA peak = **6.14 TFLOP/s = 3.07 Tmac/s per node** (48 cores × 16 fp32 × 2 pipes × 2 FMA × 2.0 GHz; ×1.1 @2.2 GHz boost). fp16-accumulate doubles it (`a64fx/doc/FP16_GEMM_CEILING.md`: a tuned fp16 12×2 kernel hits **89% of the 256 GF/core peak** L1-resident).

Per-token macs/node (dims C=4096, q_lora=1024, n_heads·q_head_dim=32768, o_inter=8192, shared_inter=2048, moe_inter=2048, n_active≈8, 43 layers):
- **Dense (REPLICATED on every node)**: qkv_b 33.6M + wo_a 33.6M + wo_b 33.6M + shared 25.2M + qkv_a/kv/router ≈ **133M/layer × 43 ≈ 5.7 Gmac/token**.
- **Routed experts (EP-sharded /11)**: ≈ **1.05 Gmac/token/node**. Head (TP) ≈0.05; attention ≈0.7 (only 3.9% of wall).
- **Total ≈ 6.8 Gmac/token/node — of which 5.7 (84%) is REDUNDANT dense.**

| scenario | macs/tok/node | FMA ceiling tok/s (11n, fp32, 100%) | at current ~7% GEMM util |
|---|---|---|---|
| **today** (dense replicated, no TP) | 6.8 G | **~451** | ~32 (≈ measured 39.5) |
| **+ TP dense** (shard 5.7→0.52 G) | **1.6 G** | **~1920** | **~140–170** |
| + TP + fp16-accum | 1.6 G | ~3840 | ~280 |

**Findings:**
- The **replicated dense hard-caps prefill at ~451 tok/s** even at 100% FMA util ⇒ 200 tok/s (44% of that) is unreachable without a near-peak kernel. **Sharding the dense (TP) lifts the ceiling to ~1920** ⇒ 200 tok/s = ~10% util, easily reachable.
- Current ~39.5 tok/s ≈ **7% of FMA peak** ⇒ the `8x3_pv` GEMM microkernel is far from the 89% a tuned A64FX kernel reaches — the second lever.
- **12 nodes barely helps prefill while dense is replicated** (the 5.7 G term doesn't shard); the 12th node is a real **+9% only after TP-compose** (then dense shards /12).

### Levers (priority order)
1. **(dominant) Compose TP-dense with batched prefill.** `DS4F_TP_*` (Step 2r) shard the dense on the *decode* path only; batched prefill aborts under TP. Wire the shard ranges (`attn_h0/h1`, `oi0/oi_rows`, `sh_r0/sh_rows`, `head_r0`) + `ds4f_load_dense_vshard` into `ds4f_forward_prefill` (M as the GEMM outer dim); the existing per-layer `[M,C]` `ar_cb`-SUM already folds the contraction-shard reduce. Drops per-node dense macs 5.7→0.52 G ⇒ ceiling 451→~1920; **at current GEMM util ≈ 140–170 tok/s (3.5–4×) by itself**, and makes the 12th node worth +9%. Bit-exact for disjoint shards (head/embed), coherent for contraction shards (attn/o-proj/shared) — Step-2r standard.
2. **Tighten the bf16/fp16 GEMM microkernel** (~7% → 20–40% of FMA peak). Port the `FP16_GEMM_CEILING.md` 89%-of-peak 12×2 register-blocking (4K-unroll, no-epilogue-convert) into the dense prefill GEMM; larger K-tile/M-block for L1/L2 weight residence. 3–4× → **>200 tok/s** combined with Lever 1. (Experts MXFP4 are only ~5% of wall — low priority; int8-svdot experts if needed.)
3. **(optional) fp16-accumulate GEMM** — doubles the FMA ceiling; gate behind a real-weight **argmax-exact** A/B (fp16 accum is lossy — may flip argmax, like the Step-2o magic/FTZ check). Only to exceed ~280 tok/s.

**Target:** Lever 1 → ~140–170; +Lever 2 → **>200**; +Lever 3 → ~280+. (Plan file: `~/.claude/plans/see-a64fx-ds4f-md-and-keep-floofy-dragon.md`.)

### Prefill results (LANDED) — 47.85 → 63.82 tok/s (+33%), batched M=64 real-weight 11n
- **Lever 1 — TP-dense composed with batched prefill (commit `21cb27e`): +21%.** Wired the Step-2r shards into `ds4f_forward_prefill` (M as the GEMM outer dim): **shared** (sh_w1/w3 col-shard → zero-pad write at `sh_r0` via the Ystride trick + partial sh_w2 folded into the routed `[M,C]` reduce; argmax-EXACT), **o-proj** (wo_a o_inter row-shard via a per-group sub-slice of the sharded tensor + zero-pad + a separate `[M,C]` attention-residual reduce; validated EQUIVALENT to the decode `ds4f_matvec_blockdiag` — batched ‖x‖ 818.6 vs decode 818.4; coherent, the +8% vs baseline is amplified contraction-shard reassociation through 43 nonlinear layers), **head** (vocab-shard GEMM → per-token local argmax merged by a new `ar_argmax_cb` = M small `tp_allreduce_argmax`, ONCE; **BIT-EXACT** argmax). 47.85→57.94 tok/s. **ATTENTION is NOT composed** — wq_b head-shard needs a per-layer `[M,H]` p_attn gather-reduce (heads not group-aligned) before the block-diagonal o-proj → measured a **NET LOSS** (comm 57%, 37 < 47.85 tok/s); TP_ATTN stays a **decode-only** memory lever (runner guard falls back to token-at-a-time prefill under it).
- **Lever 2 step 1 — bf16-pv GEMM K-tile 512→4096 (commit `bc12a25`): +11%, argmax-EXACT.** The K-tiled (512) tile-outer/token-inner GEMM reloaded the per-token X block once **per K-tile** → at large M+K that X-reload (not weight residency) was the throttle. `TILE_K=4096` (`DS4F_GEMM_TILE_K` override) lets X drive L1 reuse: min-of-20 microbench **+20–40%** across all dense shapes/M (K≤4096 → single tile, BIT-EXACT same order as the reference; K=8192 wo_b → 2 tiles, relL2 4e-7). Real 11n prefill 57.38→63.82 tok/s (Amdahl-diluted from the kernel's +20–40% by comm 31%/attention). FP8/MXFP4 tile paths left at 512 (their TILE_K sizes the dequant tile, a different trade). Harness: `ds4f_gemm_test` now reports **min-of-20 Gmac/s** (single-shot was too noisy — a "M16 regression" was pure noise) + `DS4F_GEMM_TILE_K`.

### Lever 2 — deeper GEMM kernel rewrite (PROFILING-GATED, not yet done)
After the K-tile fix the `8x3_pv` kernel is **~630 Gmac/s ≈ 20% of the 3.07 Tmac/s fp32-FMA peak**. The arithmetic *should* hit ~100% if FMA-bound (24 `svmla`/16-elem step = 32 MAC/cy = peak), so the measured ~60-vs-12 cy/step is **overhead** — candidates: the **bf16-pv predicated weight loads** (`svld1_u16 p_odd, ab-1` — the odd-lane unpack), load/FMA overlap, or memory stalls. **Pinpointing requires PMU/`fapp` profiling** (the FP16_GEMM_CEILING.md methodology). Easy knobs are exhausted: register blocking is at the limit (8×3 = 28/32 SVE regs; 8×4 spills; the blueprint **12×2 needs an 8→12-row weight repack** since `pack_pv` is 8-row). The 2× lever beyond that is **fp16-accumulate** (256 GF/core vs 128) — lossy, needs its own argmax gate (this is "Lever 3"). Iterate against `ds4f_gemm_test` (single-node, no weights) then validate argmax-coherence in the real 11n batched prefill.

**`fapp` PROFILING DONE (8192×4096 M64, busiest worker, event_raw cycle accounting; harness in `ds4f_gemm_test` `-DDS4F_FAPP` + `DS4F_GEMM_NITER` + single-shape argv, commit `4a47edb`):**

| event | % of CPU cycles | reading |
|---|---|---|
| `STALL_BACKEND` (0x0024) | 65% | backend can't accept (overlaps the waits below) |
| `LD_COMP_WAIT` (0x0184) | **~30%** | **dominant nameable stall — loads waiting on L1/L2/mem** |
| ↳ `LD_COMP_WAIT_L1_MISS` (0x0182) | ~12% | loads missing L1 → L2 |
| `FL_COMP_WAIT` (0x018a) | ~17% | FP-pipe latency/dependency |
| `FLA_VAL`+`FLB_VAL` (0x01a4/5) | **~33% combined** | the two FP pipes idle ~67% of cycles |
| `LD_COMP_WAIT_PFP_BUSY` (0x0186) | ~0% | prefetch-port not the issue |

**Verdict: the kernel is LOAD-bound, not compute- or BW-bound** (IPC 1.36, FP pipes idle ~67%, mem BW ~0.3 GB/s — everything is L1/L2-resident). Per 16-elem K-step it issues **8 predicated bf16-pv weight loads** (`svld1_u16 p_odd, ab±0` — two per pair-interleaved stream, the odd-lane high-half unpack) + 3 X loads feeding 24 FMAs; the load count/BW + load→FMA use-latency starves the pipes. TILE_K tuning is exhausted (4096→2048 only trims `LD_COMP_WAIT` 34→29%, FP-busy stays ~33%). **Prioritized fixes (load-cost order): (1) kill the predicated bf16-pv decode** — plain-contiguous-bf16 weights + an unpredicated widening load (`svld1uh_u32`+`lsl`), or fewer/unpredicated loads, to cut the ~30% LD_COMP_WAIT directly; **(2) software-prefetch** the next 8-row weight group (the ~12% L1-miss); **(3) fp16-accumulate** (Lever 3, lossy). Fix #1 first — but PROTOTYPE-MEASURE the plain-bf16 (`DS4F_BF16`, unpredicated) path vs `BF16_PV` in `ds4f_gemm_test` before any model-wide layout change, since if plain doesn't beat pv the bound is load-count/BW (→ register-blocking) not the predication.

**Fix #1 REFUTED (prototype, the measure-first paid off).** Plain unpredicated `DS4F_BF16` (`gemm_bf16_f32_tokmajor`) is **~2× SLOWER** than `BF16_PV` at every dense shape (8192×4096 M64: 314 vs 634 Gmac/s; 32768×1024 M64: 310 vs 630). So the **predicated bf16-pv decode is NOT the bottleneck — the pv dot-product kernel is the good one**; un-predicating would regress. The ~30% `LD_COMP_WAIT` is **structural to the dot-product formulation** (8 weight + 3 X loads per 24 FMA = load-count/BW + load-use-latency on A64FX's 2 load pipes), not the predication, and it persists across TILE_K. **Revised path to higher GEMM util** (in increasing effort/risk): **(a) software-prefetch** the weight stream ahead in K (target the ~12% L1-miss; HW prefetch isn't fully keeping up though `PFP_BUSY`~0%); **(b) the outer-product 12×2 fp16 blueprint** (`FP16_GEMM_CEILING.md`, 89% of 256 GF/core) — a *fundamental* rewrite: C-tile-accumulation (B-cols loaded, A-rows `ld1rh`-broadcast) instead of dot-product lane-accumulation, **+ an 8→12-row weight repack + fp16-accumulate** (lossy → argmax gate). The dot-product pv kernel is near its fp32 structural ceiling (~20% of peak, load-bound); (b) is the real lever but a large rewrite. The K-tile fix (`bc12a25`, +11% real prefill) + Lever 1 (+21%) stand as the shipped prefill wins (47.85→63.82 tok/s); the GEMM-kernel rewrite (b) is the remaining, profiling-scoped deep work.

**Fix #2 (prefetch) REFUTED + fix (b) fp16 BLOCKED — Lever 2 ceiling reached for this model (commit `40cfd35`+).** (i) SW-prefetching the weight streams (`svprfh PLDL1KEEP`, 64 elems ahead) **regresses** the dominant large-M shapes (8192×4096 M64 634→525, −17%) — extra instructions + L1 thrash on the >L1 streaming tile; HW prefetch was already adequate (`PFP_BUSY`~0 = free port, not unmet demand). Reverted. (ii) **The 89%-of-peak blueprint is fundamentally fp16-accumulate (256 GF/core), and fp16 is NOT VIABLE for DS4F's dense path:** fp16 max = 65504, but the dense latents carry the **massive-activation channels ~1e3–1e5** — the *same* channels documented to have "forced bf16-not-fp16" for the KV cache — which **overflow fp16 → inf/NaN**. So the 2× fp16 ceiling is unreachable here. (iii) An **fp32-accumulate outer-product** rewrite (12×2, `ld1rh`-broadcast) caps at the 128 GF/core fp32 peak (no fp16 2×), and the current dot-product is already ~20% of that and load-bound — the outer-product *might* cut loads (broadcasts vs vector loads) but the payoff is unproven and it's a large fundamental rewrite + 8→12-row repack. **Conclusion: Lever 2 delivered the K-tile +11%; the deeper kernel hits a wall (fp16 blocked by dynamic range, fp32 dot-product near its structural ceiling). The only remaining option — an fp32 outer-product — is high-effort/uncertain-payoff (measure-first a bench prototype before any model change).** Both quick fixes (#1 un-predicate, #2 prefetch) were refuted by measure-first prototypes — the discipline saved two wrong changes.

**fp32 OUTER-PRODUCT prototyped (`DS4F_OP_PROTO` in `ds4f_gemm_test`, commit `d7e22d3`): wins small-K ONLY.** Vectorize 16 rows/SVE-vec, broadcast NT=8 X scalars, weight loaded once per token-block; fp32 accumulate (relL2 ~1e-6, lossless, no fp16 overflow). 48T M=64 Gmac/s (OP vs dot): **32768×1024 (`wq_b`, K=1024) 994 vs 569 = +75%**, 4096×2048 (K=2048) 810 vs 737, but **K≥4096 LOSES** (8192×4096 241 vs 634). The OP wins only when the NT-token X panel fits L1 (small K); at large K the strided-X broadcasts stream cache-unfriendly, and transposing X→`[K][M]` doesn't fix it (stride-M wastes cache lines → uniform ~450–514, still < dot). **Targeted opportunity:** a shape-adaptive "OP for small-K dense tensors" would capture the **+75% on `wq_b`** (a major dense tensor) — moderate work (the OP kernel + a `[K][16]` repack for those tensors + dispatch). To beat the dot-product at large K the OP needs proper **B-panel packing** (`[K][NT]` contiguous, the standard GEMM technique) — significant, uncertain. **Net Lever-2 verdict: the K-tile (+11%) is the shipped win; the dot-product is the best general kernel; the outer-product is a viable *targeted* lever for the small-K `wq_b` (+75% on that GEMM) if pursued, but not a universal replacement.**

**OP-for-small-K MODEL integration ATTEMPTED → REVERTED (net loss, measure-first caught it).** Wired a gated (`DS4F_GEMM_OP`) lazy pv→OP repack of `wq_b` + the OP kernel into `ds4f_forward_prefill` (decode kept on pv, no risk). Real 11n A/B (full TP, M=64): **63.73 → 18.01 tok/s — a 3.5× REGRESSION** (argmax still coherent 84180). Cause: the pv→OP repack is an **inherent transpose** (strided writes) of **2.9 GB** (`wq_b` full 32768×1024 ×43 layers — TP_ATTN aborts batched prefill so `wq_b` isn't head-sharded here), done lazily *inside* the timed prefill; at short prefill the one-time pack dominates and never amortizes. Capturing the +75% would require a **load-time, pool-parallelized** repack (+2.9 GB resident + load cost) for only ~+4% prefill (wq_b ≈10% of prefill) — a poor trade. Reverted to plain `ds4f_gemm`; the validated OP kernel + bench prototype (`DS4F_OP_PROTO`, `d7e22d3`) remain as the artifact. **Final: prefill 47.85→63.82 tok/s (+33%, Lever 1 TP + Lever 2 K-tile) is the shipped result; further GEMM gains need a load-time OP repack (small payoff) or the structural rewrite (B-panel packing / fp16, both high-effort) — diminishing returns.**

- **`TP_AR_BF16` REFUTED (negative result, not adopted).** Tested whether halving the EP all-reduce
  payload (f32 → bf16) cuts the ≈12.7 ms "other"/comm. It does **not**: comm dropped only **2.5 %**
  (confirming the reduce is **latency/sync-bound, not payload-bound**) while argmax flipped 294 → 58
  (bf16-rounding the reduce → not bit-exact). Also clarified the comm structure: the two `ar_cb` call
  sites (`ds4f_impl.h:3393` PREFILL `[M,C]` and `:3570` DECODE `s_route[C]`) are **separate
  prefill-vs-decode paths, NOT two reduces per pass** — decode is **one** all-reduce per layer × 43,
  and layer N+1 needs layer N's reduced output (a hard sequential dependency in autoregressive M=1),
  so the earlier "fuse the two per-layer reduces" idea was a misread and there is nothing to fuse. The
  only real comm lever is cutting the *number of tokens-worth* of reduces = **batched / speculative
  decode** (item 3), which is architectural.

## Remaining work (priority order)

### Remaining perf levers at a glance (post-2l, decode 13.03 tok/s @ctx10240)

After six landed levers (2g–2l) decode is IN the 10–15 target band and every
**single-phase serial/NUMA artifact is fixed**. The honest verdict: **pure
decode-speed levers at ≤16k ctx are essentially exhausted** — the big remaining
phases (comm, tb2prep, attn) are all *fundamental* (latency-bound / already-pooled
/ near roofline). The real remaining gains are **structural** (batched decode, A)
and **at higher ctx, unlocked by memory** (int8 KV, B / int8 index scan, F).

| # | Lever | Attacks | Gain | Effort | Risk | Status |
|---|-------|---------|------|--------|------|--------|
| **A** | **Batched / speculative decode (M>1)** | comm 12.7 ms (biggest phase) + every dense matvec | **LARGE** — amortizes the 43 all-reduces/layer *and* reuses weights across M tokens; the **only real comm lever** (`TP_AR_BF16` refuted: latency- not payload-bound) | HIGH (packed-B GEMM; known A64FX batched-prefill regression) | MED | open — architectural (item 3) |
| **B** | **int8 KV cache** (S5 static per-channel scale, `DS4F_INT8_KV`, commit `0d8b6d1`) | memory, *not* speed | unlocks **32K–64K** ctx → the regime where the index scan/topk levers finally pay off | HIGH (only the 2 exact/tb2 attn read workers needed it — see Step 2m) | MED-HIGH | **LANDED + validated** — real-weight gen **token-identical** to bf16 (int8 never flipped an argmax over ~96 frozen-int8 tokens); longctx16384 RSS 22.66→22.31 GB (−0.35 GB, scales linearly → ~5.7 GB @256k), NaN=0, lockstep (item 2 / Step 2m) |
| **B2** | **int8 cmp_kv** (Tier-B2 compressor cache, same S5 scheme, `DS4F_INT8_CMP`, uncommitted) | memory, *not* speed | **ctx ceiling ~200k → ~255k** — cmp_kv is the largest off-arena alloc past ~64k (11,072 B/pos f32 → ¼) | HIGH (only the 1 tb2 attn read worker; 3-way branch) | MED-HIGH | **LANDED + validated** — MemFree @131072 4.52→5.57 GB (+1.05 GB, exactly the 8,304 B/pos prediction), combined slope 36.7→28.7 KB/pos; real-gen A/B **token-identical** (CAL=8 exercises int8 frozen path), NaN=0, lockstep (Step 2n). **MEASURE VIA MemFree NOT RSS** (THP-invisible to statm) |
| **B3** | **idx_kv int8 REPLACEMENT** (`DS4F_IDX_INT8`, Step 2q) | memory, *not* speed | **ctx ceiling ~1.12M → ~1.58M** — idx_kv 2,688→672 B/pos (f32 idx_kv shrunk to 64 slots, only the int8 mirror scales), slope 5.93→4.08 KB/pos | LOW (reuses validated int8 scan; shrink f32 idx_kv to DS4F_IDX_F32_SLOTS=64) | LOW (bit-identical-compute to old additive idx_int8; real-gen A/B token-identical) | **LANDED + VALIDATED — 1.5M CONFIRMED FIT (MemFree 2.31 GB, 11/11, NaN=0); 1M MemFree +2.0 GB; converts the old ANTI-lever into the real lever** |
| **B0** | **`kv_cache` window ring buffer** (automatic under tierb2, Step 2p, commit `981350a`) | **the dominant ctx-scaling cost** — kv_cache was full `max_pos·512·2B·43L` but only the 128-window is ever read | **ctx ceiling ~255k → ~1.12M (CONFIRMED)** — kv_cache stops scaling (constant ~5.6 MB), two-point slope **28.7 → 5.93 KB/pos**; **supersedes B (int8 KV): bf16 ring beats half-`max_pos` int8 kv** | LOW (storage relocation + arena-size fix) | LOW (**bit-exact** — `idx % kv_slots`, no-op when `==max_pos`) | **LANDED + VALIDATED — A/B token-identical (ring-all == HEAD); 256k/512k/1M MemFree-fit CONFIRMED (1M: MemFree 2.45 GB, NaN=0, 11/11), hard ceiling ~1.12M; + idx_kv int8 (B3/Step 2q) → ~1.58M (1.5M confirmed)** |
| **C** | **qkv matvec fusion** (`wq_a`+`wkv`, both read `s_hn`) | one pool barrier/layer | SMALL | LOW | LOW (bit-exact) | open — quick win |
| **D** | **tb2prep residue audit** (lcmp 4.6 / topk 4.2 / scan 3.6 / qproj 3.2 — the single biggest phase now) | tb2prep 17.5 ms | SMALL — no pure-serial sub-op remains; only thread-imbalance / a tail | LOW-MED | LOW | open — diminishing |
| **E** | **FP8 magic decode default** | the memory-lean `DS4F_FP8_BF16=0` path *only* (N/A to int8/bf16-pv default) | MED for that path | LOW | LOW (argmax gate) | bench-validated (Step 2f); real-weight argmax pending |
| **F** | **int8/SVE `index_score` scan** (`DS4F_IDX_INT8`, `ae1167d`) | index scan @256k | **ZERO ≤16k**, LARGE @256k | DONE | — | the int8 scan is now the *memory* lever B3 (Step 2q) — same flag, now a true idx_kv replacement |
| **G** | **fp4 compressor/indexer weights** | HBM footprint | LOW — lossy; the `tb2prep` floor is *compute* (RoPE/fp4/scan), not the weights | MED | MED (lossy) | only if HBM-pressured |
| **H** | **MXFP4 GEMM tile-dequant** (`DS4F_MXFP4_GEMM_TILE`=M-thr, default off; **prod=16**) | batched-**prefill** MXFP4 expert/dense GEMM at M≥8 | **1.38× @M16 / 1.53× @M32 / 1.72× @M64** kernel (svtbl flat ~84 Gmac/s, dequant-bound; tile→147); **real-weight 11n prefill +7.9% (36.5→39.5 tok/s)** | LOW (mirrors FP8 tile-dequant) | LOW (lossless, relL2 2e-7; real-weight argmax+‖x‖ token-EXACT; gated, svtbl kept for M≤4) | **LANDED + single-node + real-weight 11n VALIDATED** (Step 2o, 2026-06-08): token-exact across TILE 0–16, threshold-insensitive 1–16 ⇒ batched per-expert M≥16 |

Detail and rationale for each in the numbered items below; the per-phase decode
breakdown that ranks them is item 1.

> **Bottleneck history (both prior hypotheses were WRONG; resolved Step 2g).** (1) The
> "`index_score` scan is the long-ctx bottleneck" framing was wrong — the `DS4F_P_TB2SCAN`
> sub-timer showed the scan is only **1.2 % of decode**. (2) The follow-up "the other 97 % of
> `tb2prep` is O(1) projections/compressor" correction was *also* wrong — full per-component
> sub-timers showed those projections/compressor sum to only **~17.7 ms** (~14 % of `tb2prep`).
> The real cost was **`ds4f_index_topk` = 108.7 ms = 86 % of `tb2prep` = 37 % of decode**, an
> O(k·T) ctx-linear selection. It is now O(T·log k) (Step 2g above) and `tb2prep` is down to
> ~22 ms (11.6 % of decode). **Lesson: measure the one untimed op before theorizing — the gap
> between `tb2prep` and the sum of its sub-timers was the whole story.**

1. **`o_proj`/q-norm/tb2rope FIXED (2i+2j+2k) + int8 W8A8 dense (2l). New @ctx10240 breakdown**
   (decode **13.03 tok/s**): **other 12.7 (≈comm, now the biggest)**, attn 13.8, **tb2prep 17.5**
   [lcmp 4.6 / topk 4.2 / scan 3.6 / qproj 3.2 / icmp 1.3 / rope 0.24], o_proj 9.4 (was 11.6 bf16),
   qkv_proj 7.5 [`wq_b` 4.0 + wq_a 0.9 + wkv 0.7 + rope 0.44] (halved by int8), shared 6.1, experts
   4.5. The three serial-scalar artifacts (o-proj NUMA, q-norm, tb2rope) are fixed AND the dense
   matvecs are now int8 (half HBM). The remaining big phases are **fundamental**: **(a) comm ≈12.7 ms**
   = 43 per-layer EP all-reduces, ~296 µs each for a 16 KB Put → **latency/sync-bound, NOT
   payload-bound — `TP_AR_BF16` REFUTED** (ran it: comm −2.5 % only, argmax flipped 294→58; the real
   fix is reducing the *number* of reduces = batched decode, item 3 — see Step 2l note above); **(b)
   tb2prep residue** (lcmp/topk/scan/qproj) all already pooled or already-O(T·log k), partly ctx-linear
   (now the single biggest phase); **(c) attn 13.8 ms** SVE-vectorized, L2-resident — near its floor.
   *Note:* under `DS4F_FP8_BF16=1` (the longctx-runner default) the dense matvecs are bf16-pv (or int8
   with `DS4F_Q8_DENSE=1`), not FP8 — the magic/gather question only applies to the memory-lean
   `DS4F_FP8_BF16=0` path.
   - *Done & shelved:* **int8/SVE `index_score` scan** (`DS4F_IDX_INT8`, `ae1167d`). argmax-exact
     (96/96), f32 path bit-identical, gated default-off, but zero ≤16k decode win (Amdahl 1.2 % +
     M=1 q-quant cancellation). Keep as the 256k building block.
2. **256k decode without degradation (10–15 tok/s)** — the original long-ctx target. The
   **MEMORY ceiling is now essentially CLEARED** (measured ~255k with the int8 stack below);
   speed at long ctx remains the open blocker. Both characterized:
   - **Memory — MEASURED CEILING ~255k (was thought 32k).** The combined int8 stack
     **FP8-on-demand dense (`DS4F_FP8_BF16=0`) + int8 KV (2m) + int8 cmp_kv (2n)** reaches a
     hard ceiling **~255k** (safe ~242k), a ~8× jump. Path: FP8-on-demand+int8-KV alone = ~200k
     (131072 fits w/ 4.65 GB MemFree spare); int8 cmp_kv adds +1.05 GB @131072 → ~255k. **MEASURE
     VIA MemFree, NEVER RSS** — `cmp_kv`/`idx_kv` are THP-backed, costing real physical memory but
     invisible to `/proc/self/statm` (2.72 GB physical @131072, zero RSS delta). The next mem lever
     for a robust 262144 margin is **int8 `idx_kv`** (lever B3, slope → 26.6, ceiling → ~270k+).
     Clean-exit ceiling guard (`DS4F_WARM_MEMAVAIL_STOP_GB` → `_exit(42)`) makes ctx-probing on a
     shared alloc PMIx-safe (rc=42 not OOM-kill 137). *(Historical:* bf16 KV alone (Step 2e) was
     not enough — ctx=32768 OOM'd because the 27.5 GB replicated dense weights, inflated +6 GB by
     `DS4F_FP8_BF16=1`, dominate, not KV.*)*
     - **int8 W8A8 dense (Step 2l, `DS4F_Q8_DENSE`, commit `99b76ad`) takes RSS
     27.95 → 22.40 GB (−5.5 GB) AND speeds decode +9.7 %** (token-identical) — the SPEED+memory int8
     lever (pairs with FP8-on-demand for the pure-memory path). int8 dense + int8 KV + int8 cmp_kv
     clears 32K–256K; the remaining gate/head are still bf16 (argmax protection).
     **PARTIALLY LANDED: int8 W8A8 dense (Step 2l, `DS4F_Q8_DENSE`, commit `99b76ad`) takes RSS
     27.95 → 22.40 GB (−5.5 GB) AND speeds decode +9.7 %** (token-identical) — the first "int8
     weights" step toward 256k. **Retest ctx=32768 with `DS4F_Q8_DENSE=1` on a FRESH alloc** (NOT a
     shared alloc — an OOM-kill degrades PMIx, see Step 2e OPS note). Other levers: `DS4F_FP8_BF16=0`
     (FP8 dense, −6 GB, slower — now dominated by int8 W8A8 which is −5.5 GB AND faster) and **int8
     KV** (quarter the f32 footprint). int8 dense + int8 KV should clear 32K–64K; 256k needs the
     remaining dense int8 (gate/head still bf16) too.
     *int8-KV scheme de-risked* (`tools/int8kv_probe.c`, single-process, no alloc): the kv
     latent's massive-activation channels (~1e3–1e5, the same ones that forced bf16-not-fp16)
     make naive **per-token int8 catastrophic** (one sink dim sets the scale → 99 % of the
     O(1) dims collapse to 0). The viable layout is a **static per-channel scale calibrated
     on the first ~256 positions** (S5): L1-rel 0.0032 ≈ bf16, only 2.7 % of |·|>0.1 dims
     >5 % off, and **streaming-decode-compatible** (sink channels are positionally consistent,
     so early calibration holds for all later positions). 1 B/elem (+ a 512-float per-layer
     per-channel scale, amortized). NOTE: probe uses a *representative synthetic* latent
     distribution — the per-channel scheme + calibration window still need argmax-exact
     confirmation on real latents (alloc-blocked). Implementation: `kv_cache` uint16→int8 +
     `kv_scale[layer][512]` (calibrate during prefill/warm), dequant at the latent-read sites.
   - **Speed:** after `index_topk` (2g) + SVE-attn (2h) + fused o-proj (2i) + parallel q-norm (2j) +
     parallel tb2rope (2k) + int8 W8A8 dense (2l) decode is **13.03 tok/s @ctx10240** — IN the 10–15
     target band — bound by comm + tb2 residue + attn (see item 1, all now fundamental). At 256k the
     index *scan* additionally grows to dominate — int8/SVE `index_score`
     (done, `ae1167d`) is the 256k prerequisite but does not move ≤16k.
3. **Batched decode / prefill M > 1** — amortize the 43 per-layer EP all-reduces
   (comm ~20 % Amdahl floor at M=1). Needs packed-B GEMM kernels; note the known A64FX
   batched-prefill regression (`project_batched_prefill_finding`).
4. **fp4 compressor/indexer weights** — LOSSY, low value (the bf16 weights are already
   bit-exact and the `index_score`/RoPE/fp4 overhead, not the weights, is the `tb2prep`
   floor). Only if HBM pressure justifies.

## Next session — resuming prompt

> **CURRENT TASK (2026-06-08): direct MXFP4/FP8 GEMM register-dequant — DONE + real-weight VALIDATED (Step 2o).**
> The +6 GB `DS4F_FP8_BF16=1` master rep is ALREADY killed by the Step-2f on-demand FP8 tile-dequant
> (`svld1ub`→f32→bf16 L1 tile→`8x3_pv`, no resident bf16) — and the `ds4f_gemm_test` FP8 sweep confirms it
> is **bf16-FMA compute-bound** (magic vs gather neutral 1.00–1.06×) AND **f32-faithful** (relL2 3e-7 —
> FP8→bf16 is lossless), so the gather stays and the register-magic GEMM path was reverted (magic+FTZ is
> lossy: flushes E4M3 subnormals). The **MXFP4 GEMM tile-dequant** lever (`DS4F_MXFP4_GEMM_TILE`=M-thr,
> default off) is LANDED: svtbl expert GEMM is dequant-bound (flat ~84 Gmac/s); tile-dequant nibbles→bf16
> once + reuse across M = **1.38× @M16 / 1.53× @M32 / 1.72× @M64** (single-node, lossless relL2 2e-7).
> **Real-weight 11n A/B DONE (alloc 49129596): token-EXACT (argmax=108300 + ‖x‖=7.626e+02 bit-identical
> across TILE 0–16, NaN=0, lockstep) and prefill +7.9% (36.5→39.5 tok/s), threshold-insensitive 1–16
> ⇒ batched per-expert M≥16 (refutes the old "M may be small" caveat).**
> **WIRED ON BY DEFAULT (=16)** in the batched-prefill path (`ds4f_ep_runner.c`: when `prefill_batch>0` and
> `DS4F_MXFP4_GEMM_TILE` unset → `m->mxfp4_gemm_tile=16`); inert elsewhere (decode uses `mv_worker`); explicit
> env overrides incl. `=0`. Committed `c4834d7` (with Step 2n int8_cmp — entangled in the same files).
> **NEXT (still open):** lever E real-weight argmax for FP8 magic **decode** (M=1, the actual on-register
> win — distinct from the GEMM). Entry points: `ds4f_gemm_worker` MXFP4 tile branch (`ds4f_impl.h`, gated
> `T->m->mxfp4_gemm_tile`), FP8 branch comment (same fn). Bench: `a64fx/llm/ds4f_gemm_test.c` (FP8+MXFP4
> sweeps, argmax + relL2, cores 12+). (Prior decode-speed levers below are DONE.)
>
> **DONE — decode speed (don't re-chase):** `index_topk` (2g), SVE-attn (2h), fused o-proj (2i), parallel
> q-norm+RoPE (2j), parallel tb2rope (2k), int8 W8A8 dense (2l) — decode **3.38 → 13.03 tok/s @ctx10240**,
> IN the 10–15 band. Breakdown: **other 12.7 (≈comm), tb2prep 17.5 (lcmp 4.6 / topk 4.2 / scan 3.6 / qproj
> 3.2), attn 13.8, o_proj 9.4, qkv_proj 7.5 (wq_b 4.0)**. Don't re-chase o_proj/attn/topk/q-norm/tb2rope/
> dense-matvec — what remains there is FUNDAMENTAL (comm latency-bound, tb2prep already-pooled).
>
> Branch `ds4f`, all committed: 2g `ee9046f`, 2h `c77729f`, 2i `bf01f1b`, 2j `5433bcc`, 2k `d920923`,
> 2l `99b76ad` (int8 W8A8 dense, −5.5 GB + +9.7 %).
>
> **The serial-scalar/NUMA single-phase artifacts are all fixed (2i/2j/2k) and the dense matvecs are
> now int8 (2l). What remains is FUNDAMENTAL — expect diminishing returns and validate any change:**
> 1. **Per-layer all-reduce comm (≈12.7 ms "other", now the biggest).** `g_ar_secs` = 43 per-layer
>    EP combines, ~296 µs each for a 16 KB Put → **latency/sync-bound, NOT payload-bound. `TP_AR_BF16`
>    REFUTED** (ran it: comm −2.5 % only, argmax flipped 294→58 = bf16-rounds the reduce). The two
>    `ar_cb` sites (`ds4f_impl.h:3393` PREFILL / `:3570` DECODE) are SEPARATE prefill/decode paths,
>    NOT two reduces/pass — decode is ONE reduce/layer × 43 and layer N+1 needs layer N's reduced
>    output (hard sequential dep in M=1, nothing to fuse). The only real lever is cutting the *number
>    of tokens-worth* of reduces = **batched / speculative decode** (item 3). Architectural.
> 2. **tb2prep residue (lcmp 4.6 / topk 4.2 / scan 3.6 / qproj 3.2 = the single biggest phase now).**
>    All already pooled (lcmp/qproj/scan matvecs) or already-O(T·log k) (topk), partly ctx-linear.
>    Check for thread imbalance / a remaining serial tail, but no pure-serial sub-op remains.
> 3. **PIVOT TO MEMORY: retest ctx=32768 with `DS4F_Q8_DENSE=1` on a FRESH alloc.** int8 dense's
>    −5.5 GB is the first int8-weights step toward the 32768/256k OOM ceiling (resident dense weights,
>    not KV, are the wall). Pair with int8 KV (de-risked, S5 static per-channel scale) for 32K–64K.
>    Do NOT probe past ~16k on a shared alloc (OOM-kill degrades PMIx). [former item: `wq_b` is now
>    4.0 ms after int8 — no longer a standalone lever.]
>
> Gate any behavior change; validate via real-weight gen A/B (`run_ds4f_gen_11n.sh`, diff the
> `GEN_OUT` token ids — empty diff = token-exact) PLUS the ctx-warm argmax (`DS4F_CTX_WARM=10240`
> sentinel). Per the 2h lesson, an fp-reorder change can flip synthetic-warm argmax but MUST be
> token-exact on real gen — that diff is the real gate. (A bit-exact change like 2i shows BOTH
> identical, as a sanity cross-check.)
>
> **GATES (non-negotiable):** native `fcc`/`FCC` (never `fccpx`); run binaries directly in the
> alloc, NO `pjsub`; NP=11 `EXCLUDE=0,0,0` (node 0 is claude — running ~22 GB there OOM-kills the
> session); validate **argmax-exact + selected-set, never bit-equality**; keep cross-rank
> lockstep; `DS4F_MHC=1` for any coherence check; keep multi-node output compact
> (`DS4F_QUIET=1` / log+sentinel, never `cat` all 11 rank files); **don't probe ctx past ~16k**
> on a shared alloc (one OOM-kill degrades PMIx → loses the whole alloc, ~21 min re-stage); use
> `/local` else `~/tmp`; commit only when asked, message ends
> `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
>
> **If a job restart wipes /local / moves the alloc:** re-stage (Step 1 of this doc) + regenerate
> topo (never `SKIP_TOPO=1` across jobs). The single-node pinned bench (`ds4f_decode_bw_bench.c`,
> cores 12–59) is the alloc-free vehicle to roofline a new KV/attn kernel before wiring it.

---

## Optimization R&D — 1–4 node workstreams (folded from ds4f-opt.md, 2026-06)

The perf gaps decomposed into independent workstreams that iterate on **1–4 nodes** (not the scarce
11/12-node alloc), each with exact repro commands + a resume prompt, so multiple agents can work in
parallel. Final integration (real-weight 11n token-identical A/B + tok/s) is the only step needing the
big alloc. Ground rules per agent: native `fcc` single-node; every change behind its own default-OFF
`DS4F_<NAME>=1` flag; the cheap→expensive validation ladder (single-node Python-ref → single-node perf
bench → 11n real-weight A/B). *(Originally the companion doc `a64fx/ds4f-opt.md`, folded here.)*
## The numbers being attacked (from the roofline revisit)

DECODE (M=1, 11n, Q8 dense, @ctx10240 = 77.9 ms/tok; real-gen mHC path = 105.8 ms/tok):
- matvecs 29.6 ms at **247 GB/s = 31% of 800 GB/s** (µbench proves 85% per-matvec) → WS2
- **mHC hc_pre/hc_post ~37 ms/tok, 35% of the real-gen wall** (newly found) → WS1
- comm 13.5 ms = 43 all-reduces × ~300 µs (utofu tree bench = 23.5 µs) → WS4 ❌ CLOSED: the reduce
  is already a tree and the 300 µs is straggler-sync (EP barrier waiting on the slowest rank's
  expert compute), NOT comm — unfixable at the comm layer; lever is structural (MTP/spec). See WS4.
- attn 15.0 + tb2prep 17.4 = per-position floor (already 6.6×/19× optimized; low priority)

PREFILL (batched GEMM, compute-bound): 56.3 tok/s @batch32 = **~13% of the 450 tok/s FMA
ceiling**; the 8x3-pv GEMM kernel runs at 7–13% of peak vs the 89% proven in
`a64fx/doc/FP16_GEMM_CEILING.md` → WS3.

## Ground rules (every agent)

1. **Compilers**: native `fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast` (this host IS
   an A64FX node — run single-node binaries directly, no pjsub). Build via
   `make -C a64fx/llm <target> CC=fcc OPENMP=1`.
2. **Every change behind its own env flag, default OFF** (`DS4F_<NAME>=1` to enable). This is how
   parallel streams merge without conflicts and how A/Bs stay honest.
3. **Validation ladder** (cheap → expensive; never skip a rung):
   a. single-node Python-ref tests: `OMP_NUM_THREADS=12 taskset -c 12-23 ./build/ds4f_exact_test`
      then max-abs vs ref: `paste tools/exact_py.txt exact_c.txt | awk '{d=$2-$4;a=d<0?-d:d;if(a>m)m=a}END{print m}'`
      — gate **exact ≤ 5e-8**; same for `ds4f_tierb2_test` gate **≤ 2e-6**; `ds4f_mhc_test` for HC
      paths; `ds4f_gemm_test` must stay **205/205**.
   b. single-node perf bench (per workstream below) — report before/after numbers.
   c. FINAL (main session only): 11n real-weight gen A/B — bit-exact/token-identical for
      disjoint-output changes, "coherent + lockstep + NaN=0" for reassociating changes (the
      Step-2r standard).
4. **Do not commit** — leave changes on the worktree + a summary; the main session integrates.
5. Multi-node (2–4) when needed: synthetic weights (`DS4F_REAL=0`) run at ANY node count
   (experts shard e%N). Real weights at N=4 are *plausible* (est. RSS ≈ 22.5 GB/node: experts
   64×~13 MB vs 24× at /11) but unproven — validate against **MemFree, never RSS**
   (see CLAUDE.md), and don't fight it: synthetic + lockstep is the multi-node R&D vehicle.
6. Profile vocabulary: `DS4F_PROF=1` prints per-phase ms; top-level phases are indices 0..8
   (`qkv_proj attn o_proj shared router experts head other tb2prep`); the `tb2*`/`qkv_*` lines
   are SUB-timers (do not double-count). `o_proj` has an underscore.

---

## WS1 — parallelize mHC hc_pre/hc_post (decode real-gen)  ✅ LANDED (2026-06-11)

**RESULT (11n real-weight gen A/B, MAX_NEW=64, the quicksort prompt):** `DS4F_HC_PAR=1` is
**TOKEN-IDENTICAL** to off (64/64 ids, identical completion, NaNs=0, lockstep) and lifts real-gen
decode **9.46 → 11.80 tok/s (+24.7%)**; the `other`/mHC phase drops **51.1 → 30.5 ms** (−40% of the
phase). Below the optimistic 37→5 ms target — HC_PAR parallelized the collapse/expand fn-matvecs but
`other` is still 30.5 ms. **Landed**: `DS4F_HC_PAR` now defaults ON in the perf wrappers
(`run_ds4f_gen_11n.sh`, `run_ds4f_longctx_11n.sh`) and OFF in the base `run_ds4f_11n.sh` (clean
reference) — mirrors the `Q8_DENSE` pattern. Code was already committed (ef0642e); the impl below
is unchanged.

**WS1b LANDED (2026-06-11): fold the mHC RMS into the mixes-matvec dispatch.** Measured the residual
30.5 ms `other`: RMS sum-of-squares = 89 µs/call = **35% of `other`**, a serial latency-bound double
reduction on tid0 (matvec 111 µs and collapse 61 µs are already-parallel and irreducible). `DS4F_HC_RMSPAR=1`
runs a fused worker that computes the IDENTICAL F32 matvec rows AND a per-thread partial sum-of-squares
of x4 over a disjoint slice; the caller combines partials in fixed tid order. Because `ss` is a **double**
accumulation, the parallel-vs-sequential reassociation (~1e-13) is below the `float rsq` epsilon ⇒
**BIT-IDENTICAL** result (mhc_test on-vs-off = 0.0 even at 48 threads; exact 5e-8 unchanged). 11n real-gen
A/B **TOKEN-IDENTICAL** (64/64, NaNs=0, lockstep): `other` 30.5 → 24.1 ms, decode **11.80 → 12.77 tok/s
(+8.2%)**. Cumulative WS1+WS1b vs pre-mHC-par: 9.46 → **12.77 tok/s (+35%)**, `other` 51.1 → 24.1 ms.
Default ON in the perf wrappers, OFF in base. (A dead-end checked first: serializing the tiny mixes matvec
— `DS4F_HC_MVSER` — was a **10× regression**; the 24-row matvec is latency-bound and *needs* the pool.)
Remaining `other` (24.1 ms) is now the two irreducible parallel dispatches (matvec+collapse) per hc_pre.



**Current**: ~37 ms/tok ("other" 50.2 gen vs 13.5 synthetic) = 86 calls/tok (2 per layer × 43) of
`ds4f_hc_pre` / `ds4f_hc_post` (+`ds4f_hc_head` once): f32 fn-matvec `[hc=4, 16384]`, sinkhorn
weights, collapse/expand loops over `[4×4096]`, plus `memcpy(s_resid, s_x4, 64KB)` — all serial
on the calling thread, BETWEEN pool dispatches. Same pattern as the validated 2j/2k wins
(`ds4f_q_norm_rope_par` 15×, `DS4F_TB2ROPE_PAR` 19× — copy their structure).
**Target**: ~5 ms/tok. **Files**: `common/ds4f_impl.h` — `ds4f_hc_pre`, `ds4f_hc_post`,
`ds4f_hc_head_p` (~line 3930–4000); flag `DS4F_HC_PAR` (default off).

Repro (1 node):
```
make -C a64fx/llm ds4f_mhc_test ds4f_exact_test ds4f_tierb2_test ds4f_runner CC=fcc OPENMP=1
cd a64fx/llm
python3 tools/ds4f_mhc_ref.py                     # writes mhc_py reference
OMP_NUM_THREADS=12 taskset -c 12-23 ./build/ds4f_mhc_test    # correctness gate
# perf: single-node synthetic decode w/ mHC on, profile "other":
OMP_NUM_THREADS=48 DS4F_MHC=1 DS4F_TIERB2=1 DS4F_PROF=1 DS4F_MAXGEN=32 ./build/ds4f_runner
```
Gates: mhc_test bit-exact serial-vs-par (disjoint per-stream/per-dim splits ⇒ must be
BIT-EXACT, not just close); exact/tierb2 unchanged; single-node "other" phase before/after.

**Resume prompt**: "In /vol0006/mdt0/data/hp250467/work/gemm/ds4f, the mHC hyper-connection wrap
(ds4f_hc_pre/ds4f_hc_post in common/ds4f_impl.h, called 86×/token in ds4f_forward_token) costs
~37 ms/tok of serial scalar work between pool dispatches — 35% of real-gen decode. Parallelize it
across the existing thread pool (ds4f_pool_run, see ds4f_q_norm_rope_par for the validated
pattern: split disjoint output ranges, bit-exact). Gate behind DS4F_HC_PAR=1 default off. Validate
bit-exact via build/ds4f_mhc_test + ds4f_exact_test 5e-8 + ds4f_tierb2_test 2e-6, then measure the
'other' phase with DS4F_PROF=1 on the single-node ds4f_runner (DS4F_MHC=1 DS4F_TIERB2=1). Target
37→5 ms. Single A64FX node, no mpiexec. Don't commit; report before/after phase ms."

## WS2 — dense matvec issue efficiency  ⏹ CLOSED / no clean win at production scale (2026-06-11)

**FINISH (2026-06-11):** `DS4F_MV_FUSE` (the landed dispatch-fusion, commit 81575de) was validated
on 11n real-gen for the first time: **TOKEN-IDENTICAL** but **+0.86% only** (12.77 → 12.88 tok/s,
phases move both directions) — i.e. **NEUTRAL within noise**. The premise below ("the in-loop matvec
loss is pool-dispatch/serial-gap overhead") does NOT hold at production scale: real-gen dense is **Q8
(int8 svdot), which is issue/dequant-bound, not BW- or dispatch-bound** (`ds4f_decode_bw` confirms
fp8/Q8 decode is ~10% of the 723 GB/s ceiling — a different regime from the bf16 80%-of-BW the "247
vs 610" framing assumed). The matvec phases (qkv 7 + o_proj 9 + shared 6 + experts 5 + head 2 = 29 ms)
are near the Q8 kernel's throughput, and fusing 2 dispatches/layer barely registers. `matvec_sdot_8row`
already runs 8 independent row-accumulators and is svdot-throughput-bound, so the remaining idea
(SVE prefetch / more accumulators) would help latency, not issue throughput — uncertain micro-opt, not
a clean win. **MV_FUSE left default-OFF** (bit-exact but not worth enabling). WS2 is closed; the dense
decode matvec is issue-bound and the realistic next decode lever is structural (MTP/spec). Original
(refuted-at-scale) framing preserved below.

**Current** (premise refuted at 11n — see above): decode matvecs 29.6 ms for 7.3 GB = 247 GB/s; `ds4f_decode_bw_bench.c` proves ~85%
of the ~720 GB/s node read ceiling per matvec in isolation ⇒ the loss is in-loop: pool dispatch
overhead (~10 `ds4f_pool_run` per layer), thread ramp, inter-matvec serial gaps, CMG locality of
the Q8 528-byte blocks. **Target**: ≥440 GB/s in-loop (matvecs ≤ ~16 ms).
**Files**: `common/ds4f_impl.h` — `ds4f_matvec`, `ds4f_mv_worker`/`ds4f_mv_bd_worker`,
`ds4f_pool_run` (pool at top of file); flag `DS4F_MV_FUSE` (or similar).

Repro (1 node, NO model load needed):
```
make -C a64fx/llm ds4f_decode_bw CC=fcc OPENMP=1
cd a64fx/llm && taskset -c 12-59 ./build/ds4f_decode_bw     # per-shape/per-dtype GB/s table
# in-loop: synthetic single-node decode, watch qkv_proj/o_proj/shared phases:
OMP_NUM_THREADS=48 DS4F_TIERB2=1 DS4F_PROF=1 DS4F_MAXGEN=32 ./build/ds4f_runner
```
Ideas (in expected-value order): chain consecutive matvecs into ONE pool dispatch (qkv triple;
sh_w1+w3 pair — workers already exist, give them a task list); persistent-worker spin instead of
wake-per-dispatch; software prefetch (`svprfd`) in the Q8/bf16 inner loops; verify Q8 block
first-touch is CMG-local to the rowsplit that reads it (the 2i lesson: mismatched first-touch =
cross-CMG reads). Gates: exact/tierb2/gemm tests unchanged + bit-exact A/B of one forward
(fused vs not — same dot order ⇒ bit-exact required).

**Resume prompt**: "In /vol0006/mdt0/data/hp250467/work/gemm/ds4f, M=1 decode dense matvecs
achieve 247 GB/s in-loop vs ~610 GB/s (85% of node ceiling) in the isolated
build/ds4f_decode_bw bench — the loss is dispatch/serial-gap/locality, not the kernel. Reduce
per-layer pool dispatches (fuse the wq_a/wq_b/wkv chain and sh_w1+sh_w3 into single dispatches
with task lists), add SVE prefetch, and check Q8_PV 528B-block first-touch CMG locality (pattern:
a64fx/ds4f.md Step 2i). Flag DS4F_MV_FUSE=1 default off, keep per-tensor dot order identical
(bit-exact gate). Validate: ds4f_exact_test 5e-8, ds4f_tierb2_test 2e-6, then DS4F_PROF=1 phases
(qkv_proj+o_proj+shared+experts+head, currently ~29.6 ms/tok @48T) on single-node ds4f_runner
synthetic. Target ≤16 ms. One A64FX node. Don't commit; report the phase table before/after."

## WS3 — GEMM kernel 12×2 port (prefill: 7–13% → 20–40% of FMA peak, 3–4×)

**Current**: `ds4f_gemm_worker` (8x3-pv register blocking, K-tile 4096) + MXFP4 tile-dequant hit
~84–147 Gmac/s vs 3.07 Tmac/s fp32 peak. The repo has an 89%-of-peak blueprint:
`a64fx/doc/FP16_GEMM_CEILING.md` (12×2 blocking, 4K unroll, no-epilogue-convert).
**Target**: ≥600 Gmac/s (20%) conservatively; stretch 1 Tmac/s.
**Files**: `common/ds4f_impl.h` — `ds4f_gemm`/`ds4f_gemm_worker` (~line 558+) + the MXFP4 tile
path; flag `DS4F_GEMM12X2` (default off, fall back per-shape).

Repro (1 node):
```
make -C a64fx/llm ds4f_gemm_test ds4f_kernels_bench CC=fcc OPENMP=1
cd a64fx/llm
OMP_NUM_THREADS=48 ./build/ds4f_gemm_test          # MUST stay 205/205, prints Gmac/s per shape
taskset -c 12-59 ./build/ds4f_kernels_bench        # kernel-level Gmac/s
```
Gates: gemm_test 205/205 + relL2 thresholds unchanged + argmax-exact columns; exact/tierb2
regression. NOTE: stay fp32-accumulate (bf16 in, f32 acc) — fp16-accumulate flips argmax (the
Step-2o magic/FTZ lesson); it's a later, separately-gated lever.

**Resume prompt**: "In /vol0006/mdt0/data/hp250467/work/gemm/ds4f, the batched-prefill GEMM
(ds4f_gemm_worker in common/ds4f_impl.h, 8x3-pv) runs at 7–13% of the A64FX 3.07 Tmac/s fp32 FMA
peak; prefill is compute-bound so this is the whole prefill lever (56.3 tok/s vs 450 ceiling).
Port the register-blocking/scheduling from a64fx/doc/FP16_GEMM_CEILING.md (89%-of-peak 12×2
study) into the bf16→f32 GEMM path, keeping fp32 accumulate (fp16-acc flips argmax — do NOT).
Flag DS4F_GEMM12X2=1 default off with per-shape fallback to 8x3. Validate: build/ds4f_gemm_test
must stay 205/205 (it also prints Gmac/s per shape — that's the perf metric), plus
ds4f_exact_test 5e-8 / ds4f_tierb2_test 2e-6. Iterate kernel-only via build/ds4f_kernels_bench
pinned taskset -c 12-59. Target ≥600 Gmac/s on the [M=32, 4096×4096-class] shapes. One A64FX
node, no MPI. Don't commit; report Gmac/s per shape before/after."

## WS4 — tree/pipelined EP all-reduce  ❌ CLOSED / NOT-ACTIONABLE (2026-06-11)

**RESOLUTION (do NOT re-attempt the ring→tree port):** the premise below is wrong on two counts.
(1) `tp_allreduce.h` is **already a Rabenseifner tree**, not a ring (floor 7 µs@N=2 / 14 µs@N=4 /
23 µs@N=12). (2) The in-loop ~300 µs/reduce is **straggler-sync-bound, not comm-bound**: the
per-layer MoE-combine all-reduce is a hard EP barrier and `ep_ar_callback`'s timer counts the
spin-wait for the **slowest rank's expert compute** as "comm". Measured (2–4 node subset,
`tp_ar_diag_bench.c`): skew slope **b = 0.999–1.000** (reduce time = floor + max per-rank delay,
1:1); robust-mode overhead **0.3 µs**; cold-cache penalty **~0**. Corroborated by the prior
`TP_AR_BF16` refutation (−2.5%, not the −35% a payload-bound reduce shows) and a synthetic N=4
runner (comm ≈1 ms/reduce = ~70× the floor on balanced load — the barrier absorbs per-rank compute
jitter). ⇒ **No `tp_allreduce.h` header lever (robust cadence / prefetch / topology / bf16 payload)
can recover it; the floor is already optimal and <8% of 300 µs.** The real fix is **STRUCTURAL** and
lives in the **MTP/spec-decode stream** — batched/spec decode amortizes the per-barrier straggler
wait 1/K (exactly why M2b GEMM-decode already cut comm 13.1→2.3 ms/tok). Async overlap (summary.md
#8) is N/A to M=1 decode (layer L+1 data-depends on L's reduced hidden). Full writeup + tables:
`a64fx/ds4f-ws4-findings.md`. **No `DS4F_AR_TREE` flag was added.** The original (now-refuted)
framing is preserved below for the record.

---

**Current** (REFUTED — see resolution above): 43 sequential f32 all-reduces/token (16 KB hidden each)
at ~300 µs each through `tp_allreduce.h` (ring). `a64fx/utofu-tests/summary.md` measured
Rabenseifner-tree ≈ 23.5 µs at 11 nodes. **Target**: ≤50 µs/reduce. **Files**:
`a64fx/utofu-tests/tp_allreduce.h` (the production reduce used by ds4f_ep_runner) + benches
`reducescatter_bench.c`/`allgather_bench.c`. Flag `DS4F_AR_TREE=1` default off.

CORRECTNESS NOTE: a tree changes the f32 summation ORDER vs the ring ⇒ results are COHERENT, not
bit-identical to ring (like the Step-2r contraction shards). What MUST hold: (a) every rank
computes the IDENTICAL value (same deterministic order on all ranks ⇒ lockstep argmax holds);
(b) NaN=0; (c) fixed order across calls (no opportunistic arrival-order reduction). The earlier
`TP_AR_BF16` refutation was the *payload precision*, not the topology — f32 tree is fine.

Repro (2–4 nodes; this is the one WS needing mpiexec — use a small pjsub alloc, see CLAUDE.md):
```
pjsub -g hp250467 -L "freq=2000,eco_state=0,rscgrp=small,node=4,elapse=00:30:00" --no-check-directory <script>
# in-script: cd a64fx/utofu-tests && make CC=fcc && mpiexec -np 4 ./tofu_topo_helper && \
#   mpiexec -np 4 ./<your_tree_allreduce_bench>     # latency vs ring at 16 KB f32
# integration smoke (synthetic weights run at ANY N):
cd a64fx/llm && mpiexec -np 4 ./build/ds4f_ep_runner   # DS4F_AR_TREE=0/1: lockstep argmax must match itself across ranks
```
Gates: bench exactness (tree sum == ring sum within deterministic-order reproducibility; rank0
vs rankN identical bits); 4-node synthetic ep_runner lockstep (`argmax_distinct_across_ranks==1`)
+ NaN=0 with the flag on; latency table 16 KB/64 KB/1 MB payloads ring-vs-tree at N=2,4.

**Resume prompt**: "In /vol0006/mdt0/data/hp250467/work/gemm/ds4f, DS4F decode does 43 sequential
16 KB f32 all-reduces/token via the ring in a64fx/utofu-tests/tp_allreduce.h at ~300 µs each
(13.5 ms/tok = 17% of decode); the repo's own utofu tree benchmark measured 23.5 µs. Implement a
deterministic-order tree (Rabenseifner RS+AG or binomial) in tp_allreduce.h behind DS4F_AR_TREE=1
(default off), keeping f32 payload (bf16 payload was REFUTED — argmax flip) and a FIXED reduction
order so all ranks produce identical bits (lockstep requirement). R&D on a 4-node pjsub alloc
(rscgrp=small node=4, see repo CLAUDE.md for pjsub): microbench ring-vs-tree at 16KB/64KB/1MB,
then smoke ds4f_ep_runner -np 4 with synthetic weights (DS4F_REAL unset) checking
argmax_distinct_across_ranks==1 and NaN=0 with the flag on. Target ≤50 µs at 16 KB. Don't commit;
report the latency table + the lockstep check."

## WS5 — quick wins / wiring (low risk, fold into any stream)

- **TP_HEAD in plain decode** (−1.0 ms): already implemented + bit-exact-validated at 11n; just
  ensure the champion-config wrappers (`run_ds4f_gen_11n.sh`, `run_ds4f_longctx_11n.sh`) can
  enable it; nothing to R&D. (CAUTION: TP_HEAD currently breaks the MTP *draft* — accepts=0 in
  the M4 sweep — debug that in the MTP stream, not here.)
- **tb2prep micro** (17.4 ms: lcmp 4.6 + topk 4.2 + scan 3.6 + qproj 3.2): single-node probes
  exist (`tools/idxscan_bw_probe.c`, `tools/idxscore_probe.c`). Diminishing returns — only attack
  after WS1/WS2 land.

## Parallelization & merge notes

- WS1/WS2/WS3 all edit `common/ds4f_impl.h` but DISJOINT functions (hc_*, matvec/pool, gemm) —
  use separate worktrees/branches off `ds4f`, each behind its own default-off flag; merges are
  textual no-ops. WS4 edits only `a64fx/utofu-tests/tp_allreduce.h`.
- Expected combined effect (from the roofline): real-gen decode 105.8 → ~55 ms (~18 tok/s),
  synthetic 77.9 → ~45 ms (~22 tok/s), prefill 56 → 150–200 tok/s; structural decode ceiling
  stays ~32 tok/s (attn+tb2 per-position floor — only MTP/spec decode goes past it). **NOTE: WS4's
  +12% is removed — diagnosed as unrecoverable straggler-sync; its budget folds into MTP/spec.**
- Integration order (main session, 11n alloc): WS1 (bit-exact) → WS2 (bit-exact) → WS3
  (argmax-exact). ~~WS4~~ CLOSED (no code change). WS1 HC_PAR + WS2 MV_FUSE already confirmed
  lockstep + bit-identical on/off at N=4 synthetic (2026-06-11) ahead of the 11n A/B.

## WS6 — Q8 GEMM small-M remainder NaN (1-node bug, found 2026-06-10)

**Symptom**: `ds4f_forward_verify` / `DS4F_GEMM_DECODE` under `DS4F_Q8_DENSE=1` produces **NaN
nondeterministically** (some 11n runs fine, some all-NaN; |hc_head|=nan at the first head eval ⇒
the layer dense GEMMs NaN'd). `DS4F_Q8_DENSE=0` (bf16 dense) always works. Plain decode (matvec
Q8) is fine. Now GUARDED: the runner `die()`s if SPEC/GEMM_DECODE + Q8_DENSE (ds4f_ep_runner.c).
**Root-cause hypothesis**: the verify calls `ds4f_gemm` with **M=1–2**, so it only hits the Q8
worker's single-token remainder kernel (`matvec_sdot_8row`, `ds4f_gemm_worker` ~line 570), which
`ds4f_gemm_test` (M=32/64) barely exercises — vs the M≥3 `matvec_sdot_8row_3x` path it covers
well. Likely an uninitialized/tail read in the 1–2-token Q8 sdot as driven by the gemm (the
`__thread` xq/xs scratch is correct; NOT a realloc race — that was mis-diagnosed and reverted).

**1-node repro** (NO alloc, fast iterate):
```
make -C a64fx/llm ds4f_gemm_test CC=fcc OPENMP=1
cd a64fx/llm
# EXTEND ds4f_gemm_test.c to add M=1 and M=2 Q8_PV cases (it currently tests M=32/64) -> should
# reproduce the NaN/garbage at M=1,2; then it's a pure single-node kernel debug loop:
OMP_NUM_THREADS=48 ./build/ds4f_gemm_test            # add small-M Q8 rows; watch relL2/argmax/NaN
```
Gate: gemm_test 205/205 + the new small-M Q8 rows pass (relL2 ~1e-6, argmax-exact, NaN=0).
Then the verify works under Q8 and the guard can be relaxed (though bf16 dense stays the
recommended spec config — Q8 batched is dequant-bound, no speed win).

**Resume prompt**: "In /vol0006/mdt0/data/hp250467/work/gemm/ds4f, the Q8_PV path in
ds4f_gemm_worker (common/ds4f_impl.h ~line 550) NaNs nondeterministically for small M (1-2 tokens)
— the verify hits only the single-token remainder kernel matvec_sdot_8row, which ds4f_gemm_test
(M=32/64) doesn't cover. Add M=1 and M=2 Q8_PV cases to a64fx/llm/ds4f_gemm_test.c, reproduce on
ONE node (OMP_NUM_THREADS=48 ./build/ds4f_gemm_test, no mpiexec), and fix the kernel (suspect
uninitialized accumulator / K-tail / scale read in the 1-2-token Q8 sdot). The __thread xq/xs
scratch is correct — do NOT touch its allocation (a realloc-race 'fix' was already tried and
reverted). Gate: gemm_test 205/205 + new small-M rows relL2~1e-6 argmax-exact NaN=0. Then remove
the DS4F_SPEC/GEMM_DECODE+Q8 guard in ds4f_ep_runner.c. Don't commit; report the failing
shape + the fix."

## WS7 — DS4F_TP_ATTN + DS4F_Q8_DENSE wrong output (11n bug, found 2026-07-08, likely sibling of WS6)

**Symptom**: `DS4F_TP_ATTN=1` (wq_b head-shard, an existing but never-enabled-in-preset memory/speed
lever) combined with the decode preset's `DS4F_Q8_DENSE=1` produces **coherent-looking but WRONG**
generated tokens — diverges from the very FIRST decoded token (gen_ids 0/64 match vs the correct
reference), yet `NaN=0` and `rc=0` throughout, and the aggregate stats (decode tok/s, RSS) look
entirely plausible (13.62 tok/s, RSS -1.37 GB — this is what makes it dangerous: a superficial A/B
that only checks NaN=0 + last-token argmax would ship it as a clean win). **Always diff the FULL
generated token sequence, not just the last token**, when validating a TP/sharding change.

**Bisection (11n real-weight A/B, same 24-tok prompt, 64-tok greedy decode, `run_ds4f_gen_11n.sh`
called directly — `run_ds4f_agentic_11n.sh`'s hardcoded `export DS4F_Q8_DENSE=1` clobbers an env
override, use `run_ds4f_gen_11n.sh`'s `${VAR:-default}` form to actually toggle flags):**
- `TP_ATTN=1, Q8_DENSE=0, OPROJ_FUSE=0` (plain per-group o-proj, no block-diag at all) → **64/64
  IDENTICAL** to the `TP_ATTN=0` reference. Correct.
- `TP_ATTN=1, Q8_DENSE=0, OPROJ_FUSE=1` (bf16 block-diagonal o-proj, `ds4f_matvec_blockdiag`'s
  `DS4F_BF16_PV` branch) → **64/64 IDENTICAL**. Correct — `ds4f_matvec_blockdiag` itself is fine.
- `TP_ATTN=1, Q8_DENSE=1` (forces `ly->wo_a.type==DS4F_Q8_PV`, which forces block-diag
  unconditionally via the `ds4f_oproj_fuse || tpo || wo_a.type==DS4F_Q8_PV` check regardless of the
  `OPROJ_FUSE` flag) → **0/64 match, diverges at token 1**. **Broken.**

**Isolates the bug to `ds4f_matvec_blockdiag`'s `DS4F_Q8_PV` branch** (common/ds4f_impl.h ~line 498,
inside `ds4f_mv_bd_worker`) specifically when its activation input (`m->s_attn`) is a TP_ATTN-partial,
zero-padded buffer. `wq_b`'s own (non-block-diag) `DS4F_Q8_PV` matvec branch (~line 377, in plain
`ds4f_matvec`) can be ruled out analytically: it quantizes `m->s_qlat`, which is REPLICATED (identical
on every rank, unaffected by TP_ATTN) — deterministic quantization of identical input cannot diverge
across ranks, so it's architecturally safe regardless of TP_ATTN's row-sharded *output*.

**Root cause NOT fully pinned down** despite deep inspection: `ds4f_quant_x_sdot_into`'s 64-element
quantization blocks are each entirely within ONE attention head (HD=512 = 8×64), and TP_ATTN shards
at whole-head granularity (always a multiple of 512, hence of 64) — so blocks never straddle an
ownership boundary, and per-block dequant (`matvec_sdot_8row`, ggml_dequant.h:1709) is a standard
weight-scale × activation-scale × int8-dot with no obvious cross-block interaction. This analysis
suggests the partial-quantize-then-all-reduce-sum SHOULD reconstruct the full-quantize result exactly
— yet it measurably doesn't. Given WS6 (above) already documents a DIFFERENT nondeterministic Q8_PV
bug in the same kernel family (`ds4f_gemm_worker`'s small-M remainder path) that also resisted
surface-level inspection, this is likely a SIBLING low-level bug in the shared int8 W8A8 machinery
(`ds4f_quant_x_sdot_into` / `matvec_sdot_8row` / the `__thread` xq/xs scratch reuse across groups),
not a TP_ATTN-specific logic error — worth debugging WS6 and WS7 together.

**Disposition**: NOT shipped (`DS4F_TP_ATTN` stays off in `--preset decode`). Benefit was modest
(+1.9% decode speed, -1.37 GB RSS in the broken measurement — unverified once/if fixed) vs. the
already-landed `TP_HEAD`+`TP_EMBED` wins (+2.3% speed, -1.9 GB combined, both bit-exact). Do not
enable `DS4F_TP_OPROJ` or `DS4F_TP_SHARED` either without first resolving this — same block-diag /
Q8_PV interaction surface, same risk.

**Resume prompt**: "In /vol0006/mdt0/data/hp250467/work/gemm/glm5-1, DS4F_TP_ATTN=1 combined with
DS4F_Q8_DENSE=1 produces wrong (but NaN-free, plausible-looking) decode output — isolated to
ds4f_matvec_blockdiag's DS4F_Q8_PV branch (common/ds4f_impl.h ~line 498) via 11n A/B bisection (see
WS7 above for the full trail). Both non-Q8 o-proj paths (plain per-group AND bf16 block-diagonal)
are bit-exact under TP_ATTN; only the int8 W8A8 block-diag kernel breaks. Likely a sibling of WS6
(same matvec_sdot_8row/ds4f_quant_x_sdot_into machinery). Reproduce with a 1-node unit test if
possible (mirroring WS6's ds4f_gemm_test approach) rather than the slow 11n A/B loop: construct a
synthetic multi-rank TP_ATTN scenario (partial/zero-padded input vector, per-rank local quantization,
sum the dequantized partial dot products) and compare against a single full-vector quantize+dot
reference. If the isolated kernel test also shows a mismatch, the bug is confirmed structural
(not an artifact of the real-weight 11n harness) and can be fixed without needing an allocation.
Candidate structural fix if root-causing stalls: extend the existing `if (tpo && ...) m->ar_cb(m->
s_attn, H, m->ar_ctx);` pre-reduce (common/ds4f_impl.h ~line 5352, currently gated only on `tpo`==
TP_OPROJ) to ALSO fire whenever `ly->wo_a.type==DS4F_Q8_PV`, so every rank quantizes an already-
fully-reduced (not partial) s_attn — trades away TP_ATTN's comm savings for correctness, needs its
own A/B to confirm. Don't commit without an 11n token-for-token bit-exact re-validation."
