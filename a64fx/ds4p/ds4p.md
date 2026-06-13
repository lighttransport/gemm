# DS4P — DeepSeek-V4-Pro inference on 48–64 A64FX nodes

DS4P (`~/models/ds4p`, **805 GB**, 64 safetensors shards) is a pure dimension-scaling
of DS4F: same `deepseek_v4` graph, byte-identical tokenizer, identical tensor naming
(hash-gate `gate.tid2eid`, mHC, compressor/indexer, MTP), same MXFP4 expert packing,
same kv_lora=512, same wo_a in-dim 4096. The whole DS4F stack (`common/ds4f.h`,
`common/ds4f_impl.h`, `a64fx/llm/ds4f_{ep_runner,stage}.c`) is reused; the model is
selected with **`DS4F_MODEL=ds4p`** (config: `ds4f_pro_config()` in `common/ds4f.h`).

## Architecture diff (verified against config.json + safetensors headers)

| param | DS4F | DS4P |
|---|---|---|
| n_layers | 43 | **61** |
| hidden | 4096 | **7168** |
| n_experts | 256 | **384** (384%48=0, 384%64=0 — even EP) |
| moe_inter / shared_inter | 2048 | **3072** |
| n_heads | 64 | **128** (128%64=0 — even TP_ATTN @64) |
| q_lora | 1024 | **1536** |
| o_groups / o_inter | 8 / 8192 | **16 / 16384** (o_lora stays 1024) |
| index_topk | 512 | **1024** |
| routed_scale | 1.5 | **2.5** |
| compress_ratios | 44: `[0,0,4,128,…,4,0]` | **62: `[128,128,4,…,4,0]`** — NO dense decode layers |

Everything else identical: head_dim 512, qk_rope 64, kv_lora 512, vocab 129280,
window 128, hc_mult 4, n_hash_layers 3, MTP 1, YaRN.

## DS4F-coincidence bugs fixed during bring-up (commit history)

DS4F has `gin == hidden` (64·512/8 = 4096) — DS4P breaks that (128·512/16 = 4096 ≠ 7168).
Fixed: wo_a cols must be `gin = n_heads·q_head_dim/o_groups`, not `hidden`, in
arena_size / synth alloc / load_real / MTP load; the non-exact o-proj stand-in summed
groups of `hidden` off the end of `s_attn` (the synth NaN source). Also: synth alloc
now TP-shards wq_b like load_real (full wq_b OOB-wrote `s_q` under TP_ATTN), and
`MAX_NODES` is 96 (was 32).

## Memory budget (per node, TP all on)

Experts ≈ 33.5 MiB each → 766 GiB total. Dense ≈ 420 MB/layer of which ~352 MB is
TP-shardable (wq_b 101, wo_b 117, wo_a 67, shared 66); embed+head 7.4 GB.

| | 48 ranks | 64 ranks | 11 ranks (LAYERS=10 test) |
|---|---|---|---|
| experts/rank/layer | 8 | 6 | 35/34 |
| weights/rank | ~22 GB (tight) | **~17.5 GB** | ~14 GB |
| /local blob/rank | ~50 GB | ~46 GB | ~24 GB |
| stage wall (shared-FS bound) | ~40 min | ~40 min | ~20 min |

**TP (HEAD/EMBED/SHARED/ATTN/OPROJ/WOB) is mandatory** — un-TP'd dense alone is
~33 GB/node. **64 nodes recommended** for bring-up. Validate memory via **MemFree,
never RSS** (THP-backed caches are RSS-invisible).

## Workflow

**12-node interactive (the only hands-on tier; full model does NOT fit — ~70 GB
experts/rank):** layer-truncated runs validate everything mechanically.

```sh
SYNTH=1 ./run_ds4p_12n.sh            # synthetic, no staging (arena/TP/EP/comm)
./run_ds4p_stage_12n.sh              # stage layers 0..9 -> /local/ds4p (~20 min)
./run_ds4p_12n.sh                    # real weights: load=11/11, NaN=0, lockstep
LAYERS=4 ./run_ds4p_stage_12n.sh && LAYERS=4 ./run_ds4p_12n.sh   # fast loop
```

Success criteria are mechanical (truncated model ⇒ garbage text): rc=0,
`load=11/11`, NaN=0, `argmax_distinct_across_ranks==1` (rank-0 token is
authoritative under TP_HEAD — per-rank distinct=2 is the NORMAL local-argmax,
not a divergence).

**VALIDATED 2026-06-10 (12-node interactive):** stage LAYERS=10 → 11/11 in 791 s,
20.07 GB/rank (2464 tensors). Real-weight run (EXACT+TierB2+mHC+TP-all, FP8 dense):
load=11/11, NaNs=0, argmax=89596 identical on all 11 ranks, RSS 13.42 GB,
22 tok/s decode @10 layers (comm 29% — 11-rank latency, shrinks per-layer at 64n).
Tier-B2 banner "4 CSA + 6 HCA layers" matches the DS4P ratios for layers 0–9
(incl. the new ratio-128 layer 0/1). Synth TP-all and ds4f regressions also green.

**48/64-node batch (pjsub; /local is wiped per job ⇒ stage+run in ONE job):**

```sh
LAYERS=8 sh ~/job-64.sh   # smoke first (8 layers, no gen)
sh ~/job-64.sh            # full 61 layers + generation
```

`~/job-64.sh` submits `pjsub_ds4p_64n.sh` with the proven submission pattern
(rscgrp=small-s2, node=4x4x4:torus, retention_state=0, `--llio
localtmp-size=80Gi`, `PJM_LLIO_GFSCACHE=/vol0006` — model + repo are on
/vol0006). Direct `pjsub --no-check-directory [-x LAYERS=8] pjsub_ds4p_64n.sh`
also works (same #PJM directives, minus the GFSCACHE env).

Phases are gated and emit `SENTINEL <phase>=...` lines; the job log is the single
diagnostic artifact. First coherent text = phase 5 of the full 64-node job.

## Env reference (on top of the DS4F knobs)

- `DS4F_MODEL=ds4p` — selects the Pro config (runner + stager). Default `ds4f`.
- `DS4F_STAGE_LAYERS=N` — stager: keep only `layers.L.*` with L<N (0 = all).
- `DS4F_NSHARDS` defaults to 64 and `DS4F_MODEL_DIR` to `~/models/ds4p` under ds4p.
- Stage dir convention: `/local/ds4p` (set via `DS4F_STAGE_DIR` for both stage+run).
- MTP/spec-decode: out of scope for bring-up (`DS4F_STAGE_MTP=1` adds ~14 GB×NP
  shared-FS reads); the DS4F MTP/spec stack carries over once enabled.

## Known headroom / next levers

- Staging reads dense replicated ×NP (~2–3× amplification). If 40 min hurts:
  stage TP-sliced dense rows per rank (wq_b/wo_a/shared are contiguous row
  ranges) or Tofu-broadcast dense after one rank stages — cuts reads to ~1.1 TB.
- 48-rank long-ctx: the DS4F ctx levers (window ring, int8/int4 cmp, CP shard)
  apply unchanged if headroom gets tight.

## Long context (1M) and co-serving — validated 2026-06-14 (96 & 108 nodes)

### >32-rank EP all-reduce (commit 5268f0b, prerequisite for ANY run past 32 nodes)
The uTofu EP-combine all-reduce (`a64fx/utofu-tests/tp_allreduce.h`) was hardcoded
`TP_AR_MAXN=32` (built for the 11-node DS4F runs) and FATAL'd at `tp_comm_init` for N>32.
The Rabenseifner recursive-doubling schedule is already non-power-of-2 aware (runs at N=11),
so only the static caps needed lifting: **`TP_AR_MAXN 32→128`, `TP_AR_NSTEP 8→9`** (recv
slots; `bcast_sid = ⌈log2 N⌉+1 ≤ 8` for N≤128) and **`ds4f_ep_runner MAX_NODES 96→128`**.
The N=11 ds4f path is unchanged.

### ds4p @ 1M context → 96 nodes is the MINIMUM (job 49227903, end-to-end real-weight)
`pjsub_ds4p_96n.sh` — 96 = 8×12 = 6×16, `node=4x4x6:torus`, `--llio localtmp-size=87Gi`,
`DS4F_MODEL=ds4p DS4F_EP_SIZE=96` + all `DS4F_TP_*` + `DS4F_TIERB2 INT8_CMP INT4_CMP IDX_INT8`
+ `DS4F_MAXPOS=1048576`:
- **Staging blob ~30+ GB/rank** (replicated FP8 dense for 61L/7168 + 4 MXFP4 experts +
  compressor/indexer) ⇒ needs **`localtmp-size=87Gi`** — 30Gi and 64Gi both fail `No
  space`/EC=2. 96/96 staged in ~17 min.
- **Run:** load 96/96 (arena **22.99 GB/rank**, 4 experts/rank), `tp_ar: N=96 pof2=64 rem=32`,
  `warmtb2 DONE` at ~1M, **MemFree 2.11 GB**, lockstep argmax across all 96 ranks, NaNs=0,
  `SENTINEL ds4p_96n_1M=done`. Perf (memory-first): prefill 3.67 tok/s, decode 1.27 tok/s,
  EP comm 8.1%.
- **96 is the genuine minimum:** it finishes with only ~2 GB MemFree — at the guard floor.
  The replicated ~5.4 GB Tier-B2 caches don't shard with EP/TP, so <96 OOMs and >96 adds NO
  1M headroom (only context-parallelism shards the caches). **Validate via MemFree, not RSS.**

### 108-node co-serving (96 ds4p + 12 ds4f) — mechanism validated; 1M is at the cliff (job 49228161)
`pjsub_cosrv_108n.sh` splits ONE `node=6x6x3:torus` (108) allocation into disjoint
vcoordfiles (96 ds4p + 12 ds4f), runs each EP group in its **own cwd** (`cosrv/<grp>`, which
isolates `tofu_topo.txt` + the `ds4f_ep_*_rank*.txt` outputs) with a **separate uTofu
communicator**, and launches both `mpiexec` groups **concurrently**. The runner honors
**`TOFU_TOPO_PATH`** (commit 08534a7, default `tofu_topo.txt`) so groups read distinct topo
files; the topo helper is unchanged (per-group cwd isolates its write).
- **Co-residence WORKS:** clean split (96/12), per-group topo isolated (96/12 rows), ds4p
  staged 96/96, two concurrent EP `mpiexec` groups coexisted on disjoint nodes. **ds4f
  (synthetic, 12 nodes) ran clean** — `tp_ar: N=12 pof2=8 rem=4`, prefill 10.75 tok/s,
  NaNs=0, RSS 20.62 GB, **rc=0**.
- **ds4p @1M failed in DECODE (rc=1):** it loaded 96/96 and **warmed all 61 layers to ~1M**,
  but at **MemFree 0.38 GB** (vs 2.11 GB standalone — the `/local` HBM page-cache of the
  ~30 GB blob plus node-placement variance on the 6×6×3 subset ate the margin). Under that
  pressure a decode EP all-reduce timed out (`tp_ar bcast timeout want=142 got=141`).
- **Takeaway:** the 96 ds4p + 12 ds4f topology is sound (isolation + concurrency proven, ds4f
  green), but **ds4p @ exactly 1M on 96 nodes has no memory margin and is not robust**. For a
  production co-serve: run ds4p at a context with headroom (≤~512k frees several GB), or shard
  the replicated Tier-B2 caches with context-parallelism — adding EP/TP nodes alone cannot buy
  1M headroom. (ds4f here was synthetic to validate the mechanism; real ds4f is `DS4F_REAL=1`
  + a 12-node stage away — validated separately at 11n.)

### CP cache-sharding RESOLVES the cliff — validated 2026-06-14 (jobs 49228433, 49228435)
Context parallelism (`DS4F_CP=1 DS4F_CP_SHARD=1 DS4F_CP_IDX=1` — shards the compressed
`cmp_q4`/`idx_kv8_4` caches by slot range across the EP ranks; composes with int4 caches +
`ar_cb`; `run_ds4f_11n.sh` forwards these) gives ds4p @1M the missing margin. Same lever that
took ds4f 255k→12M ctx on 11 nodes (byte-identical to CP-off).
- **ds4p @1M, ep=112, CP — `pjsub_ds4p_cp_112n.sh` (job 49228433, 112 = 96+16):** load
  112/112, arena 22.98 GB/rank, `tp_ar N=112`, **`warmtb2 DONE` MemFree 3.98 GB** (vs 0.38 GB
  without CP — caches now ~/112 per node), decode **clean (no all-reduce timeout)**,
  `SENTINEL ds4p_cp_112n_1M=done`. prefill 3.76 / decode 0.99 tok/s.
- **Fixed co-serving, 128 = 96 ds4p(CP) + 32 ds4f — `pjsub_cosrv_128n.sh` (job 49228435):**
  **both groups rc=0.** ds4p(CP) `warmtb2 DONE` MemFree **3.95 GB**, decode clean; ds4f
  (synth, `tp_ar N=32`) prefill 10.56 / decode 8.86 tok/s, NaNs=0, RSS 12.57 GB. Two
  concurrent EP groups, isolated topo, both green — the 108n cliff (rc=1) is gone.
- **Cost:** CP raises decode comm 8.1%→~29% (cross-rank top-k merge + per-CSA-layer latent
  gather); ~1.0 tok/s decode (memory-first). ~20% more comm buys ~3.6 GB of 1M headroom.
- **Recommendation (supersedes the cliff warning above):** run ds4p @1M with **CP on**
  (≥96 nodes). For co-serving, **108 = 96 ds4p(CP) + 12 ds4f** or **128 = 96 + 32** both work
  — CP, not the node count, is the enabler. Reserve more nodes only for headroom (finer cache
  shard + fewer owned experts) or higher context (CP scales the ceiling with node count).

### How many CP contexts fit on 96+N nodes (ds4p only, each context ≤ 1M) — capacity equation
Let **M = 96 + N** total EP ranks (= nodes, 1 rank/node). Per-node budget on a 31.8 GB node,
~30 GB usable (≈2 GB OS/guard floor):

    weights      W(M) ≈ 12 + 2.8·⌈384/M⌉ GB        # measured: 22.99 @ M=96/112 (owned 4),
                                                    #           19.4 @ M=128 (3), 17.42 @ M=192 (2)
    cache budget B(M) ≈ 30 − W(M) GB                # 7.0 @96, ~7.0 @112, ~10.6 @128, ~12.6 @192

With CP the compressed Tier-B2 cache shards by slot across all M ranks, so per-node cache for
K concurrent contexts totalling P positions (P ≤ K·1M) is

    cache/node(P,K) ≈ ρ_sh·P/M  +  ρ_rep·P  +  K·σ
        ρ_sh ≈ 1.84 KB/pos   # slot-sharded CSA cmp_q4 + idx_kv8_4 (the 1.84 GB CP freed at M=96 for 1×1M)
        ρ_rep ≈ small        # replicated remainder: HCA cmp + 64-slot f32 idx
        σ     = per-context NON-sharded scratch (mainly s_idx_scores ∝ max_pos) — to be pinned by a sweep

⇒ max concurrent 1M-contexts:

    K_max(M) ≈ ⌊ B(M) / ( ρ_sh·1M/M  +  ρ_rep·1M  +  σ ) ⌋

The sharded term `ρ_sh·1M/M` is **tiny at M ≥ 96** (1.84 GB / M ≈ **19 MB per 1M-context at
M=96**, 14 MB at M=128), so **CP removes the cache as the binding limit** — exactly as for ds4f
(255k→12M on 11n became LOAD-PEAK-bound, not cache-bound). The ceiling on 96+N nodes is then
**σ (per-context scratch) + the load-peak transient**, giving `K_max(M) ≈ B(M)/σ`, which **grows
with M** because W(M) falls (more nodes → less weight → more budget) *and* the per-context shard
shrinks. Measured single-context anchors (CP on): M=96 → MemFree **3.95 GB**, M=112 → **3.98 GB**
— i.e. room for several more 1M contexts already at 96. **Next:** a K=2,4 concurrent-context
sweep to pin σ (and the load-peak), turning K_max into a hard number per N.
