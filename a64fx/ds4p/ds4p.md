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
pjsub --no-check-directory -x LAYERS=8 pjsub_ds4p_64n.sh   # smoke first
pjsub --no-check-directory pjsub_ds4p_64n.sh               # full 61 layers + generation
```

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
