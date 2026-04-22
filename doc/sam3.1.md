# SAM 3.1 — current port state

This document captures where the sam3.1 port stands so work can resume on
another machine (e.g. one with more GPU VRAM than the 8 GB RTX 3070 used
so far). It summarises what landed for sam3 first, what exists for sam3.1
today, the architectural delta vs sam3, and a reproducible procedure for
picking the work back up.

## Status snapshot

| Area | sam3 | sam3.1 |
|------|------|--------|
| CPU runner (`cpu/sam3*`) | ✅ working end-to-end | ❌ not started |
| CUDA runner (`cuda/sam3*`) | ✅ working end-to-end (MMA + tiled paths both correct) | ⚠️ scaffold only; loader fails on first key lookup |
| Server backend (`server/server_sam3*.c`) | ✅ cpu + cuda routed, cached ctx | ⚠️ stub returns 501 with link to this doc |
| Web demo (`web/sam3.html`) | ✅ at `/sam3` (cpu / cuda backend, cancel, mask overlay) | ⚠️ page lives at `/sam3.1`, model selector preselected, server returns 501 |
| Weight conversion `.pt → .safetensors` | n/a (already shipped as safetensors) | ✅ `cuda/sam3.1/convert_pt_to_safetensors.py` |

The sam3 path works on both backends with the same `/v1/infer` API. The
sam3.1 path is intentionally fail-fast at the server layer until a real
runner exists.

## Why sam3.1 isn't a rename of sam3

sam3.1 is **architecturally a different model** — it shares Sam3*Config
class names in HF transformers but the published `sam3.1_multiplex.pt`
weights use a fundamentally different layout. Diff (verified by reading
the safetensors index — see `cuda/sam3.1/sam3.1_keys.txt`):

| Stage | sam3 | sam3.1 |
|-------|------|--------|
| top-level prefix | `detector_model.` | `detector.` (+ `tracker.model.*` for video) |
| ViT pos embed | `embeddings.position_embeddings` (learned 24×24, tiled) | **2D RoPE** with per-block `attn.freqs_cis` (576, 32) complex64 |
| ViT QKV | three separate `attention.{q,k,v}_proj.{w,b}` | one fused `attn.qkv.{w,b}` of shape (3072, 1024) |
| ViT key paths | `vision_encoder.backbone.layers.N.*` | `backbone.vision_backbone.trunk.blocks.N.*` |
| ViT FFN dim | 4736 | 4736 (same) |
| ViT block count | 32 | 32 (same) |
| Text encoder | HF-CLIP (`text_encoder.text_model.encoder.layers.N.self_attn.{q,k,v}_proj.*`) | OpenAI-CLIP (`backbone.language_backbone.encoder.transformer.resblocks.N.attn.in_proj_*`, `mlp.{c_fc,c_proj}`, `ln_{1,2}`, **pre-norm**, with extra `text_projection` 1024→512) |
| FPN neck | `vision_encoder.neck.fpn_layers.0..3.*` (4 levels, ConvT/MaxPool + 1×1 + 3×3) | gone — replaced by `backbone.vision_backbone.convs.0..2.*` (3 stages, dconv 2×2 + 1×1 + 3×3) and parallel `interactive_convs` / `propagation_convs` |
| DETR encoder | self-attn + text-cross-attn + MLP, separate q/k/v | self-attn + **cross_attn_image** with fused `in_proj_weight` of shape (768, 256), MLP 2048 |
| DETR decoder | bbox_embed, separate presence path | adds `presence_token`, `presence_token_head`, `presence_token_out_norm`, `boxRPB_embed_{x,y}`, `query_embed`, `reference_points`, `ref_point_head` |
| Heads | `mask_decoder` (instance/semantic), `dot_product_scoring.{text_mlp, text_proj, query_proj}` | `segmentation_head.{cross_attend_prompt, cross_attn_norm, instance_seg_head, mask_predictor, pixel_decoder, semantic_seg_head}`, `dot_prod_scoring.{hs_proj, prompt_mlp, prompt_proj}` |
| New | — | `geometry_encoder.*` (14 sub-modules: cls/label embeds, boxes/points project + pool + pos_enc, encode + final_proj + norm) — completely absent from sam3 |
| Multiplex tracker | n/a | `tracker.model.*` (457 tensors), needed only for video |

Bottom line: ~70% of sam3.1 is a new architecture, not a relabel. The
PORT.md in `cuda/sam3.1/` lists the load-side work but underestimates the
kernel work for the new heads.

## Hardware constraint hit on this machine

The sam3 CUDA runner uploads ~1.7 GB of FP16 weights + activation buffers
to the device (sam3.1 will be similar). On the 8 GB RTX 3070 used here
the first NVRTC kernel-compile + weight-upload pass survives, but the
second concurrent process trips `cuMemAlloc` (only ~50 MiB free with
chrome's GPU process running). For sam3.1 work — which will likely
double-buffer detector + tracker — **a 16 GB+ GPU is recommended**
(RTX 4080/4090, A6000, H100 etc).

Verify free memory before launching:

```bash
nvidia-smi --query-gpu=memory.free --format=csv,noheader
# need >= ~3 GB free for the full sam3.1 detector + activation buffers
```

## What you can already run today

### 1. sam3 CPU and CUDA via the standalone CLI

```bash
# CPU
cd cpu/sam3
make ARCH=native -j
./test_sam3 /mnt/nvme02/models/sam3/sam3.model.safetensors \
            path/to/image.jpg --phrase person \
            --vocab /mnt/nvme02/models/sam3/vocab.json \
            --merges /mnt/nvme02/models/sam3/merges.txt -o /tmp/cpu.npy
# expect: kept N masks @ 2048x1365, top score ~0.7, ~70 s wall on 32-core Zen2

# CUDA (after MMA fragment fix in cuda/cuda_kernels_common.h)
cd ../../cuda/sam3
make -j
./test_cuda_sam3 /mnt/nvme02/models/sam3/sam3.model.safetensors \
                 path/to/image.jpg --phrase person \
                 --vocab /mnt/nvme02/models/sam3/vocab.json \
                 --merges /mnt/nvme02/models/sam3/merges.txt -o /tmp/cuda.npy
# expect: score 0.98 box=(-0.0, 96.9, 1360.1, 2049.7), pipeline ~3.4 s
```

### 2. sam3 via the diffusion server + web UI

```bash
mkdir -p server/build && cd server/build
cmake .. -DDIFFUSION_SERVER_ENABLE_MCP=ON \
         -DDIFFUSION_SERVER_ENABLE_SAM3=ON \
         -DDIFFUSION_SERVER_ENABLE_SAM3_CUDA=ON
cmake --build . -j
./diffusion-server --web-root ../../web \
  --sam3-ckpt   /mnt/nvme02/models/sam3/sam3.model.safetensors \
  --sam3-vocab  /mnt/nvme02/models/sam3/vocab.json \
  --sam3-merges /mnt/nvme02/models/sam3/merges.txt
# open http://127.0.0.1:8080/sam3
```

The Release default + `-O3 -funroll-loops` is set in
`server/CMakeLists.txt`; without it the build defaults to `-O0` and
inference is ~3× slower.

### 3. sam3.1 weight conversion + key inspection

```bash
cd cuda/sam3.1
uv venv --python 3.11
VIRTUAL_ENV=$(pwd)/.venv uv pip install torch safetensors numpy packaging
.venv/bin/python convert_pt_to_safetensors.py
# wrote /mnt/nvme02/models/sam3.1/sam3.1.model.safetensors  (1623 tensors fp32)
.venv/bin/python inspect_ckpt.py | head -40
cat sam3.1_keys.txt | head        # full shape/dtype dump
```

### 4. sam3.1 web demo (UI only, server returns 501)

`http://127.0.0.1:8080/sam3.1` serves the same UI as `/sam3` with the
model selector pre-pointed at `sam3.1` (URL-driven). Submitting yields:

```
{"ok":false,"error":{"code":"not_implemented",
  "message":"sam3.1 runner port is in progress (see cuda/sam3.1/PORT.md);
             no executable backend wired into the server yet"}}
```

## Where to pick the work up next

Suggested order, smallest-to-largest:

1. **Get a real PyTorch reference**. The `Sam3VideoModel` /
   `Sam3VideoConfig` in `transformers` 5.5.4 still uses the **sam3.0**
   layout (`detector_model.vision_encoder.backbone.layers.N.attention.q_proj.weight`)
   and `load_state_dict` against the sam3.1 weights returns thousands of
   `missing` / `unexpected` keys. Two paths:
   - check a newer transformers release (post-5.5.4) for a `Sam3_1` or
     `Sam3MultiplexModel` class and re-test; OR
   - vendor the official Meta `sam3.1` reference repo and write
     `ref/sam3.1/gen_image_ref.py` mirroring `ref/sam3/gen_image_ref.py`
     to dump per-stage activations for the verifiers.
   *Without one of these, the C runner can't be verified — building it
   blind is what wasted the last attempt.*

2. **`cpu/sam3.1/` skeleton** — clone `cpu/sam3/` and:
   - rewire the loader to the `detector.backbone.vision_backbone.trunk.blocks.N.*`
     keys; load the fused `attn.qkv.{w,b}` directly (no fuse step needed);
   - replace the relative-position-bias path with 2D RoPE driven by
     `attn.freqs_cis` (complex64 (576, 32)). Apply per-head per-token
     after the QKV projection, before scaled-dot-product;
   - retarget the text encoder to OpenAI-CLIP (`transformer.resblocks.N`,
     fused `attn.in_proj_weight`, `mlp.c_fc/c_proj`, `ln_1/ln_2`,
     **pre-norm** ordering, then `ln_final` and `text_projection 1024→512`);
   - replace the FPN with the `convs.0..2` / `interactive_convs` /
     `propagation_convs` stack — this needs the official module diagram
     to wire correctly;
   - swap `mask_decoder` + `dot_product_scoring` for `segmentation_head` +
     `dot_prod_scoring` + `geometry_encoder`.

   Verify each stage against the `ref/sam3.1` dumps from step 1.

3. **`cuda/sam3.1/`** — `cuda_sam3_1_runner.c` already compiles (it was
   cloned from `cuda/sam3/`) and reaches the loader. Fix the loader the
   same way the CPU loader is fixed. Most of the existing CUDA kernels
   (LayerNorm, GeLU, GEMM via `gemm_f16_f32`, flash-attn) carry over;
   only the ViT attention kernel needs a RoPE pre-multiply step in front
   of the existing softmax + value mat-mul, and the head kernels need
   reworking. The freshly-fixed MMA `m16n8k16` fragment layout is shared
   via `cuda/cuda_kernels_common.h` and works.

4. **Wire into server** — `server/server_sam3.c` and
   `server/server_sam3_cuda.c` already have the dispatch boilerplate
   (request parsing, mask-PNG response build, ctx cache). Add a third TU
   `server_sam3_1.c` (or extend the existing ones to dispatch on the
   `model` field), and remove the early-501 short-circuit in
   `server/server.c::infer_json`.

5. **Web** — nothing to do; `/sam3.1` already routes and the model toggle
   is wired. The `pending_runner` pill in `models_json` should flip to
   `ready` once the runner lands.

## Files that ship today

```
common/cpu_compute.h                 -- AVX2 LN + OMP gemm/attn (sam3 perf)
cpu/sam3/sam3_runner.c               -- AVX2 conv2d / convT, OMP block load
cpu/sam3/test_sam3.c                 -- per-stage timing
cuda/cuda_kernels_common.h           -- m16n8k16 fragment-A layout fix
cuda/sam3/cuda_sam3_runner.c         -- EXTERNAL_IMPLS guard, MMA default
cuda/sam3.1/                         -- scaffold (clone of cuda/sam3/)
  README.md, PORT.md                 -- this story in repo form
  convert_pt_to_safetensors.py       -- .pt -> .safetensors, fp32/fp16
  inspect_ckpt.py                    -- shape dump
  sam3.1_keys.txt                    -- 1623 tensors with shape/dtype
  Makefile, *.c, *.h                 -- compiles, loader fails (intentional)
server/CMakeLists.txt                -- Release default + -O3
server/server.c                      -- /sam3 + /sam3.1 routing, cancel-aware
server/server_sam3.c, server_sam3.h  -- CPU sam3 backend + ctx cache
server/server_sam3_cuda.c            -- CUDA sam3 backend + ctx cache
server/run-sam3.sh                   -- launcher
server/smoke.sh                      -- exercises sam3 + sam3.1
web/sam3.html                        -- /sam3 + /sam3.1 demo with overlay
```

## Open questions for the next attempt

- Is `sam3.1_multiplex.pt` the only weight artefact, or is there a
  separate detector-only file? (the current file unpacks to ~3.5 GB fp32;
  ~85 MB of that is `tracker.model.*` for video.)
- Does Meta publish a public reference repo with the sam3.1 module
  definitions, or only the weights? — needed for step 1 above.
- The `geometry_encoder.*` block (14 sub-modules) suggests sam3.1 takes
  bbox/point prompts in addition to text. Decide whether the first pass
  supports text-only (as sam3 does) or wires geometry prompts too.
