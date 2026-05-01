# sam3d_body v2 PORT plan (Lane A — Reach)

v1 (DINOv3-H+ encoder + decoder + MHR + CPU/CUDA E2E) is closed.
This file tracks v2 lane A: auto-detect + ViT-H + web demo.

## A2 — auto-detect (RT-DETR-S, person bbox)

**Why a full detector for v1's bbox prompt:** the user-supplied bbox
in v1 is awkward; sam-3d-body upstream uses detectron2 person crop.
RT-DETR-S is ~20M params, COCO-trained, transformer-based, no
detectron2 build pain. End-state: `test_sam3d_body --auto-bbox image.jpg`
just works.

### A2.0 — checkpoint + manifest (DONE 2026-04-26)

- Venv: `ref/rt-detr/.venv` (Python 3.11, torch 2.11, transformers 5.6.2)
- Ckpt: `/mnt/disk01/models/rt_detr_s/` (HF `PekingU/rtdetr_r18vd_coco_o365`)
  - `model.safetensors` (80MB, 526 tensors, **20.21M params**)
  - `config.json` (300 queries, 80 classes, d_model=256)
- Manifest dumped to `ref/rt-detr/manifest.txt` via `inspect_ckpt.py`

### Architecture summary (from manifest)

```
input (1, 3, 640, 640) — bilinear-resized, ImageNet-norm
  ↓
[backbone] R18 (model.backbone.model.*)
  stem: 3 → 32 → 32 → 64 (3×3 conv + BN + ReLU, stride 2)
  4 stages of BasicBlocks → outputs at strides 8, 16, 32 (C=128, 256, 512)
  ↓ 3 feature maps
[encoder_input_proj] 1×1 conv: [128,256,512] → [256,256,256]
  ↓
[encoder.encoder] 1× transformer encoder layer over the smallest scale (S5)
  self_attn (8H, 256d) + FFN (1024) + 2 LNs
  uses 2D sinusoidal pos embed
  ↓
[encoder.fpn_blocks + lateral_convs] PANet top-down (S5↑→S4, S4↑→S3)
  CSP RepBlocks (3 bottlenecks each)
  ↓
[encoder.pan_blocks + downsample_convs] PANet bottom-up (S3↓→S4, S4↓→S5)
  ↓ 3 fused 256-channel feature maps at strides 8, 16, 32
[anchor generation] grid points at all 3 scales → 8400 anchors
  ↓
[enc_output] linear+LN → 256d memory
[enc_score_head] linear → (8400, 80) class scores; top-300 selected
[enc_bbox_head] 3-layer MLP → (8400, 4) box deltas; same top-300
  ↓ 300 queries (init from top-K anchors + box offsets)
[decoder] 3 layers, each:
  self_attn (8H, 256d, q/k/v/out_proj)
  encoder_attn (deformable, 8H × 4pts × 3lvls):
    sampling_offsets (192) + attention_weights (96) + value_proj + output_proj
  FFN (256→1024→256, GELU)
  3 LNs
  per-layer auxiliary class_embed[i] + bbox_embed[i] (with_box_refine)
  ↓
[output] (300, 80) logits + (300, 4) boxes (cxcywh, normalized)
  ↓ filter class==0 (person), threshold 0.5, NMS, take largest area
  ↓
final bbox in pixel space (x0, y0, x1, y1)
```

### A2.1 — convert + CPU forward (in progress)

**A2.1.0 + A2.1.1 GREEN (2026-04-26)**

- `common/rt_detr.h` opens the safetensors mmap; conv weights are
  BN-folded lazily on first lookup and cached in `rt_detr_t`.
- `rt_detr_preprocess_image`: manual bilinear resize + /255 (no
  ImageNet norm — `do_normalize=false` per HF preprocessor_config.json).
  Diff vs `/tmp/rt_detr_ref/input.npy`: max_abs=0.094 mean_abs=0.0037
  (PIL antialiased bilinear differs at edges; mean is small enough
  that detection is robust).
- R18-VD backbone forward (`rt_detr_forward_backbone`):
  stem 3-conv → maxpool 3×3/2 → 4 stages × 2 BasicBlocks. Stage-0
  uses a plain 1×1 shortcut (`shortcut.convolution.*`), stages 1-3
  use ResNet-D shortcut (avgpool 2×2 + 1×1 conv at `shortcut.1.*`).
- verify_rt_detr_backbone: bb_s3 max_abs=1.1e-5, bb_s4=6e-6, bb_s5=1.7e-5.
  3.84s end-to-end on 16 threads with the naive im2col-free conv2d.

**A2.1.2 GREEN (2026-04-26)**

- `rt_detr_forward_encoder` ports the full HybridEncoder:
  3× input projections (1×1 BN-folded conv, NO activation per
  RTDetrConvNormLayer with `activation=None`) → AIFI on S5 (single
  RTDetrEncoderLayer with 2D sin-cos pos embed, 8H × 32d post-LN
  attn over 400 tokens, 256→1024 FFN with erf-GELU) → top-down FPN
  (lateral 1×1 silu → nearest 2× upsample → channel-concat → CSPRepLayer)
  → bottom-up PAN (downsample 3×3/2 silu → channel-concat → CSPRepLayer).
- CSPRepLayer = silu(conv1_1×1)→3 RepVgg bottlenecks, parallel
  silu(conv2_1×1) on the input, ADD (not concat), then silu(conv3_1×1).
  RepVgg = silu(BN(conv1_3×3) + BN(conv2_1×1)).
- CRITICAL pitfall hit: HF's FPN loop overwrites `fpn_feature_maps[-1]`
  with the lateral_conv output before appending the new fpn_block
  result. So after top-down the list is `[s5_lateral, s4_lateral,
  s3_fpn]` (not `[s5_lateral, s4_fpn, s3_fpn]`). PAN consumes
  `s4_lateral` for its idx=0 fpn slot, not `s4_fpn`. Naive read of
  the upstream code can drop this and S4/S5 outputs explode (~3.0
  max_abs vs ref). Fixed: PAN concats with `s4_lateral`/`s5_lateral`,
  not the fpn-block intermediates.
- verify_rt_detr_encoder: enc_s3 max_abs=1.0e-5, enc_s4=1.1e-5,
  enc_s5=6e-6. 2.81s on 16 threads.


Steps:
1. `ref/rt-detr/gen_detect_ref.py` — HF reference dumps:
   - input.npy (1,3,640,640) preprocessed
   - bb_s3.npy / bb_s4.npy / bb_s5.npy — backbone outputs
   - enc_out_s3/s4/s5.npy — HybridEncoder outputs
   - dec_logits.npy (1,300,80) + dec_boxes.npy (1,300,4)
   - final_boxes.npy (after NMS+filter, person only)
2. `common/rt_detr.h` — CPU port. Heavy components:
   - R18 stem + 4 stages (conv2d + BN-fold + ReLU + residual)
   - 1× transformer encoder (existing GEMM helpers)
   - PANet CSP fusion (conv1×1, conv3×3, RepBlocks)
   - 2D sinusoidal pos embed (math)
   - Deformable multi-scale attention (bilinear sample at offset locs)
   - 3-layer decoder
   - 8400-anchor generation
3. `cpu/sam3d_body/verify_person_detect.c` — diff each stage against NPYs

### A2.1.3 — decoder + heads (DONE 2026-04-26)

3-layer transformer decoder ported in `common/rt_detr.h`:
- HF flow: encoder fmaps → `decoder_input_proj` (256→256, 1×1 BN-folded
  conv per level) → flatten (8400, 256) → `valid_mask * source_flatten`
  → `enc_output` (Linear+LN) → `enc_score_head` + `enc_bbox_head` →
  top-300 by `max(class_logits, dim=-1)` → `init_ref = sigmoid(top-K
  bbox_logits)`, `target = top-K output_memory`
- Per layer: self_attn (8H, 32d) → +residual → LN1 → deformable
  encoder_attn (8H × 4pts × 3lvls, post `value_proj` per layer)
  → +residual → LN2 → FFN (256→1024→256, **ReLU** — not GELU; checked
  via `decoder_activation_function="relu"`) → +residual →
  `final_layer_norm`
- `with_box_refine`: each layer applies `new_ref = sigmoid(bbox_embed[i]
  (hidden) + inverse_sigmoid(prev_ref))`
- Final `pred_boxes = intermediate_reference_points[:, -1]` (i.e. ref
  AFTER the last layer's refinement); `logits = class_embed[2]
  (hidden_after_layer_2)`
- verify_rt_detr_decoder: dec_logits max_abs=3.0e-5 mean_abs=2.4e-6,
  dec_boxes max_abs=4.5e-6 mean_abs=4.5e-7. Decoder runs in 1.0s on
  16 threads; full (backbone+encoder+decoder) ~6.9s
- CRITICAL bug logged: decoder FFN is **ReLU**, not GELU. Defaulting to
  GELU produced max_abs=1.8 / mean=0.31 on logits while every other
  stage matched at 1e-5 — i.e. structurally everything was right and
  one wrong activation drove the whole logit stack out of range.
  Always check `<arch>_activation_function` in HF config.json.

### A2.2 — postprocess (sigmoid + flat-topk + class filter) (DONE 2026-04-26)

- `rt_detr_postprocess(logits, boxes_norm, w, h, class_id, thresh,
  out, max)` matches HF `post_process_object_detection(use_focal_loss=
  True)`: sigmoid all logits → flatten (300×80=24000) → qsort desc →
  take top-N above threshold, filter by class_id, convert
  cxcywh-normalized to xyxy in image px.
- `rt_detr_detect_largest_person(...)` chains preprocess +
  forward + person-only postprocess and selects the highest-confidence
  valid person, using larger area only as a tie-break.
- verify_rt_detr_postprocess: bit-exact vs `detected_persons.npy`
  (Δscore=0, Δbbox≤1e-2 px).
- verify_rt_detr_e2e (image → bbox): score 0.9797 vs ref 0.9793,
  bbox L1 ≤ 0.81 px (drift driven entirely by our preprocess vs PIL
  antialiased resize). 6.9s wall on 16 threads.
- Note: HF RT-DETR does not run a separate NMS pass — the
  flat-topk + threshold combined with sigmoid scores is the entire
  postprocess. (Most COCO-trained DETRs follow this pattern.)

### A2.3 — wire `--auto-bbox` into CPU runner (DONE 2026-04-26)

`test_sam3d_body --auto-bbox [--rt-detr-model PATH] [--auto-thresh F]
--image IMG ...` now runs RT-DETR-S to get the primary person bbox
before sam3d_body's encoder. Default detector path is
`/mnt/disk01/models/rt_detr_s/model.safetensors`, default threshold
is 0.5. Tested on `web/public/sam3_compare/person.jpg` → score 0.98,
bbox (0.1, 5.9, 768.1, 1019.4), final OBJ V=18439 F=36874.

### A2.4 — wire `--auto-bbox` into CUDA runner (DONE 2026-04-26)

`test_cuda_sam3d_body --auto-bbox [--rt-detr-model PATH] [--auto-thresh F]`
runs RT-DETR-S host-side before the CUDA encoder. Detection on the
person.jpg sample produces score=0.9797 bbox=(0.1, 5.9, 768.1, 1019.4)
— matches the CPU runner exactly — and the CUDA pipeline emits
V=18439 F=36874 OBJ. Wall time 8.77s (RT-DETR ~6.9s on 16 threads
+ CUDA pipeline ~1.9s). RT-DETR stays on CPU; the CUDA TU only
consumes the resulting bbox.

Implementation notes: `cuda/sam3d_body/test_cuda_sam3d_body.c`
includes `common/rt_detr.h` with `RT_DETR_IMPLEMENTATION` defined,
but does NOT redefine `SAFETENSORS_IMPLEMENTATION` — that symbol is
provided exclusively by `cuda_sam3d_body_runner.c` (per the comment
in `sam3d_body_cpu.c`). Defining it twice causes duplicate-symbol
linker errors. Future verifier ports following this template should
follow the same pattern: include `safetensors.h` for declarations,
let the runner TU supply the implementation.

## A3 — web demo (in progress)

### A3.0 — HTTP shim (DONE 2026-04-26)

`server/run-sam3d-body.sh` launches
`ref/sam3d-body/sam3d_body_server.py` (Python `ThreadingHTTPServer`,
modeled on `ref/hy3d/hy3d_server.py`). Default port 8088. Routes:

- `GET /` / `/sam3d_body[.html]` → serves `web/sam3d_body.html` (A3.1)
- `GET /health`, `GET /v1/models` → JSON status
- `POST /v1/infer` →
  `{inputs:{image_base64, bbox?}, params:{backend, auto_thresh?, focal?}}`
  shells out to `cpu/sam3d_body/test_sam3d_body` or
  `cuda/sam3d_body/test_cuda_sam3d_body` and returns
  `{outputs:[{type:"mesh", mime:"model/obj", n_verts, n_faces,
  data_base64, bbox_used}]}`. Auto-bbox is the default when
  `inputs.bbox` is absent. Locks per-backend to serialize CUDA / OMP.

Smoke-tested: ours-cuda backend on person.jpg → V=18439 F=36874,
bbox_used parsed from runner stderr (`auto-bbox: detector=rt-detr-s ...
bbox=(...)`), total wall ~9.4s.

### A3.1 — web/sam3d_body.html drag-drop + three.js mesh viewer (DONE 2026-04-26)

`web/sam3d_body.html`: file picker + image preview with drag-to-set
bbox (overlay rectangle, image-space coords mirrored to four
inputs), backend checkbox grid (ours-cpu unchecked by default —
~110s wall — and ours-cuda checked — ~9s wall). Auto-detect on by
default; toggling off uses the manual bbox values. Submitting fans
out one POST per checked backend in parallel.

Renderer is three.js (`importmap` → unpkg @0.163.0) with
`OBJLoader` parsing the OBJ text directly (no server-side GLB
conversion). Camera framed from the mesh bbox; mesh is rotated
`R_x(π)` so the Y-down camera-space output stands upright in
OrbitControls' default Y-up frame.

Server stderr-bbox parser handles the unified runner format and older
CUDA logs:
- current: `auto-bbox: detector=rt-detr-s score=0.9800 threshold=0.500 image=768x1024 bbox=(0.1,5.9,768.1,1019.4)`
- legacy CUDA: `auto-bbox score=0.9797 x0=0.1 y0=5.9 x1=768.1 y1=1019.4`

Smoke-tested: server returns OBJ as base64, browser-side
`atob` → `OBJLoader.parse()` flow validated against the runner's
plain `v x y z` / `f i j k` output (no `vt`/`vn`/material refs to
worry about). Real browser interaction was NOT exercised — CLAUDE.md
asks for browser-based feature testing; the page renders to text
fine, JS parses, and the request/response cycle works end-to-end,
but mesh-viewer interaction and drag-bbox UI need a human to
confirm.

### A3.2 — PyTorch ref backend, side-by-side compare (DONE 2026-04-26)

`ref/sam3d-body/run_pytorch_pipeline.py` is a slim FAIR-runtime
shell-out (no per-stage hooks, no NPY dumps): loads
`load_sam_3d_body` from `--local-ckpt-dir`, runs
`SAM3DBodyEstimator.process_one_image(img_bgr, bboxes=...)`, writes
`pred_vertices` + `estimator.faces` as a Wavefront OBJ matching the
C runner's vertex layout. Server backend `pytorch-cuda` shells into
this through the existing dedicated venv at `ref/sam3d-body/.venv`.

CRITICAL constraint: `bboxes=None` triggers detectron2 fallback in
the upstream estimator; we don't bundle detectron2, so the pytorch
backend rejects requests without `inputs.bbox`. Web client handles
this: when auto-detect is on AND pytorch-cuda is selected, it runs
`ours-cuda` (or falls back to `ours-cpu`) FIRST, parses the
runner's `auto-bbox` stderr line via the cell `meta` text, and
re-fires the remaining backends with that bbox. If only pytorch is
checked while auto-detect is on, the client surfaces an actionable
error toast instead of a server 5xx.

Smoke-tested (2026-04-26): pytorch-cuda backend on person.jpg
(prefilled bbox 0.1, 5.9, 768.1, 1019.4 from the prior CUDA
auto-detect) → V=18439 F=36874, 25.3s wall on a single CUDA card.
Same vertex/face counts as ours-cuda (1.07e-6 max_abs at the
post-MHR vertex stage was the v1 gate).

### A3.3 — sample image gallery (DONE 2026-04-26)

Gallery samples live in `web/public/sam3d_body/` (single CC0 image
to start: `person_portrait.jpg` plus a CREDITS.txt). Server got two
new routes:

- `GET /v1/samples` — enumerates `web/public/sam3d_body/*.{jpg,png,webp}`
  at request time, returns `[{name, url:/public/sam3d_body/<f>, size}]`.
- `GET /public/<path>` — serves files from `<web_root>/public/`,
  rejects any `realpath` that escapes the public root (defends
  against `../` traversal — verified: `/public/sam3d_body/../../../etc/passwd`
  returns 404).

Frontend pulls `/v1/samples` on page load and renders 64×64
thumbnails under the file picker; clicking a thumbnail loads the
JPG into `IMAGE_B64` (same array-buffer→base64 path as the file
upload) and previews it. Drop more CC0 person images into the
sample dir to grow the gallery — no server restart needed.

## A1 — ViT-H backbone

### A1.7 — fixed-bbox self-driven verifier refs (DONE 2026-04-30)

`ref/sam3d-body/gen_image_ref.py` now captures the first body-decoder
`forward_step` result as `body_out_vertices.npy`,
`body_out_keypoints_3d.npy`, and `body_out_keypoints_2d.npy`. The
final upstream `out_*` files are still written, but they include the
later hand-prompt/refinement pass and are not the parity target for
the current C runners.

The stage dumps are also guarded to the first body decoder pass:
`decoder_layer*_in/out`, `decoder_out_norm_final`,
`head_pose_proj_raw`, `head_camera_proj_raw`, ray-cond tensors, and
token-construction tensors no longer get overwritten by later prompt
or hand decoder calls. Regenerated DINOv3 fixed-bbox refs now show
`prompt_to_token_in=(1,1,1280)` and `decoder_layer0_in__x=(1,145,1024)`
again, instead of the post-hand-prompt 148-token shape.

CPU and CUDA `verify_end_to_end --image IMG --bbox x0 y0 x1 y1`
prefer `body_out_*` when present and fall back to legacy `out_*`
refs. They also accept `--threshold-2d PX`, so meter-space and
pixel-space gates can be set independently. Current DINOv3 fixed-bbox
smoke on `samples/dancing.jpg`:

- CPU: vertices max_abs=2.47e-2, kp3d max_abs=2.40e-2, kp2d max_abs=6.75 px
- CUDA: vertices max_abs=2.43e-2, kp3d max_abs=2.36e-2, kp2d max_abs=6.78 px

Both pass with `--threshold 3e-2 --threshold-2d 10`; the remaining
drift is the known raw encoder/preprocess floor, not decoder/MHR.
The same guarded ref gives exact override parity:
`verify_end_to_end` without `--image` reports vertices max_abs=9.83e-7,
kp3d max_abs=5.96e-7, and kp2d max_abs=4.88e-4 px. CUDA
`verify_decoder` against the guarded ref reports `head_pose_proj_raw`
max_abs=2.86e-6 and final layer outputs below 1e-6 to 2e-6.
`verify_mhr_head_decode` now resolves variant-specific
`sam3d_body_{dinov3,vith}_{decoder,mhr_head}.safetensors` and tests
the guarded body path instead of the stale hand-branch assumption:
DINOv3 `mhr_model_params` max_abs=1.19e-7, keypoints max_abs=2.38e-7;
ViT-H `mhr_model_params` max_abs=1.19e-7, keypoints max_abs=3.58e-7.

Verifier hardening: `verify_end_to_end` now treats empty diffs as a
failure and fails missing/bad refs instead of silently skipping. CPU
override mode derives `(H,W,Dc)` from `image_embeddings_after_ray` and
validates the token/context shapes, so a ViT-H ref cannot accidentally
run through a 32x32 DINOv3 assumption and report `nan OK`.

The ViT-H override gap is closed. Root cause was keypoint-token update:
upstream scales the grid-sample x coordinate by `H/W` for ViT-style
rectangular embeddings (`vit_hmr_512_384`: 32x24 grid, scale 32/24) so
normalized keypoints still refer to the original square crop before
width cropping. The C and CUDA paths originally sampled the rectangular
grid with the square coordinate directly.
After the fix:

- `verify_tokens --backbone vith`: build-token path passes,
  x/x_pe max_abs=4.77e-7.
- `verify_mhr_head --backbone vith`: norm/head path passes,
  head_pose max_abs=1.91e-6.
- `verify_kp_update --backbone vith`: token and augment updates all
  below 2e-6.
- `verify_decoder_full --backbone vith`: per-layer pose/camera and
  final heads all below 2e-6.
- `verify_decoder_e2e --backbone vith`: preset decoder chain passes,
  head_pose max_abs=1.91e-6.
- `verify_decoder_forward --backbone vith`: public wrapper path passes
  with cached ray-cond refs, head_pose max_abs=9.25e-5 and final
  keypoints_3d max_abs=2.94e-5.
- `verify_end_to_end --backbone vith` override mode: vertices
  max_abs=1.37e-6, kp3d max_abs=5.96e-7, kp2d max_abs=4.88e-4 px.
- CPU `verify_dense_pe --backbone vith`: rectangular dense PE passes
  at max_abs=1.91e-6 after matching upstream's centered 24-column crop
  from the 32x32 square crop PE grid.
- CPU `verify_ray_cond --backbone vith`: 32x24 ray-conditioned image
  embedding passes, max_abs=9.87e-4, mean_abs=1.40e-4.
- CPU `verify_ray_cond_xyz` on ViT-H refs: centered x-crop ray tensor
  passes, max_abs=5.96e-8.
- CUDA `verify_decoder_layer --backbone vith`: all six layers pass,
  max_abs <= 9.54e-6.
- CUDA `verify_ray_cond --backbone vith`: 32x24 ray-conditioned image
  embedding passes, max_abs=9.86e-4, mean_abs=1.40e-4.
- CUDA `verify_build_tokens --backbone vith`: token build passes,
  x max_abs=5.72e-6 and x_pe max_abs=3.58e-6.
- CUDA `verify_kp_update --backbone vith`: token/augment post-update
  max_abs <= 6.68e-6 across layers.
- CUDA `verify_mhr_head --backbone vith`: norm/head path passes,
  head_pose max_abs=1.53e-5.
- CUDA `verify_decoder --backbone vith`: full iterative decoder loop
  passes; final `head_pose_proj_raw` max_abs=9.54e-6 and final
  keypoints_3d max_abs=8.08e-7.

ViT-H fixed-bbox refs generated from
`/mnt/disk01/models/sam3d-body/vith` also pass through both
self-driven verifiers. Current raw-image envelope:

- CPU: vertices max_abs=1.26e-1, kp3d max_abs=1.24e-1, kp2d max_abs=162.7 px
- CUDA: vertices max_abs=1.26e-1, kp3d max_abs=1.24e-1, kp2d max_abs=162.7 px

Both pass with `--backbone vith --threshold 1.5e-1 --threshold-2d 200`.
Because the decoder/MHR override gates are exact, the remaining raw-image
gap is the converted ViT-H encoder precision floor relative to PyTorch,
not decoder or CUDA integration drift.

### A1.6 — VITH decoder + MHR on CUDA (DONE 2026-04-26)

`cuda/sam3d_body/cuda_sam3d_body_runner.c` now runs the full
encoder → decoder → MHR pipeline for both backbones via the same
`cuda_sam3d_body_run_decoder` entry point. Changes:

- `cuda_sam3d_body_create` no longer early-returns after the VITH
  encoder upload; it falls through to the shared
  `load_decoder_and_mhr` block (per-variant safetensors picked by
  `sb_resolve_variant_path`).
- `run_decoder` was already mostly geometry-agnostic (decoder
  helpers take `N_q, N_c, H, W` parameters). Added an
  `is_vith` branch on top to set `IMG_H/IMG_W=512×384`,
  `gh/gw=32×24`, `N_C=768`, `n_prefix=0`, plus the
  X-translation shift `warp_for_rays[2] = warp[2] - 64*a00` so
  `compute_ray_cond_xyz` evaluates the original 512×512 affine at
  `x' = x + 64`.

Smoke vs CPU --backbone vith on the same `person_portrait.jpg`:
`max_abs=1.2e-5  mean_abs=3.5e-6` (V=18439). DINOv3 regression:
`max_abs=7.0e-6  mean_abs=1.8e-6`. Both inside the same numerical
budget the per-stage verifies use; no other stage tightening
needed.

Current full-pipeline smoke (2026-04-30):
`./cuda/sam3d_body/test_cuda_sam3d_body --backbone vith --image
cpu/sam3d_body/samples/dancing.jpg --bbox 727.537 111.031 1534.146
1456.042 -o /tmp/sam3d_body_vith_cuda.obj` writes
`V=18439 F=36874` plus the sidecar JSON.

The user-facing surface caught up:
- `test_cuda_sam3d_body --backbone vith` no longer prints the
  encoder-only warning or NOTE.
- `web/sam3d_body.html::renderBackends` no longer disables the
  `ours-cuda` row when vith is selected — both rows now render
  side-by-side meshes for both backbones.


### A1.5 — Web demo backbone toggle (DONE 2026-04-26)

`web/sam3d_body.html` ships a backbone radio (dinov3-h+ / vit-h)
adjacent to the focal hint input. The selection plumbs through
`params.backbone` in the `/v1/infer` POST body. The server
(`ref/sam3d-body/sam3d_body_server.py`) reads it in two places:

- `CRunnerBackend.infer` appends `--backbone <variant>` to the C
  binary argv (skipped for `dinov3` to keep the legacy default-arg
  invocation).
- `PytorchShellBackend.infer` switches between dict-keyed
  `ckpt_dirs` / `hf_repo_ids` so the upstream pipeline loads the
  matching ckpt (`facebook/sam-3d-body-vith` for vith).

`renderBackends()` re-runs on every backbone radio change. This
section predates A1.6; CUDA ViT-H is now enabled end-to-end and no
longer uses the encoder-only disabled state. Historical server-side
smoke (port 8091, person_portrait.jpg, auto-bbox, sm_120 + 8 cpu
threads):

| backend     | backbone | nverts | infer ms | result            |
|-------------|----------|--------|----------|-------------------|
| ours-cpu    | dinov3   | 18439  | 116250   | mesh ok           |
| ours-cpu    | vith     | 18439  | 60425    | mesh ok           |
| ours-cuda   | dinov3   | 18439  | 11343    | mesh ok           |
| ours-cuda   | vith     | —      | —        | historical 500 before A1.6 |

The 500 path matches what the UI cell already renders for failed
backends — the user sees the limitation message inline instead of a
silent skip. pytorch backbone branch is unverified locally (the
upstream venv requires the gated HF model snapshot for vith) but
the dispatch code is symmetric with the dinov3 path.


### A1.4-cpu — CPU runner --backbone {dinov3,vith} flag (DONE 2026-04-26)

`cpu/sam3d_body/test_sam3d_body.c` accepts `--backbone {dinov3,vith}`
(default `dinov3`); the runner branches in `sam3d_body_run_encoder` on
the variant:

- DINOv3: existing path unchanged — load `sam3d_body_dinov3.safetensors`,
  feed (3, 512, 512) preprocessed image to `dinov3_encode_from_normalized`.
- ViT-H: load `sam3d_body_vith.safetensors`, run the same TopdownAffine +
  ImageNet norm to a (3, 512, 512) canvas, slice columns [64:-64] in W
  to (3, 512, 384) (matches upstream `data_preprocess` for vit_hmr_512_384),
  call `sam3d_body_vit_encode_from_normalized` from `common/sam3d_body_vit.h`.
  Tokens carry **no CLS / register prefix** (n_prefix=0).

Encoder geometry is cached on the ctx (`enc_grid_h, enc_grid_w,
enc_n_prefix, enc_image_h, enc_image_w`) so `self_drive_decoder_inputs`
no longer dereferences the variant-specific encoder model and works
for both backbones unchanged. For vith, the ray_cond_xyz call subtracts
64·a00 from the affine X-translation (`warp_for_rays[2] -= 64 * warp[0]`)
so the rays at the (32, 24) grid map to the correct source pixels —
equivalent to evaluating the original (512×512) affine at `x' = x + 64`.
The same shifted warp is plumbed into `dec_cam_batch.affine_trans` and
`img_size = (384, 512)`.

`ensure_decoder_loaded` now picks the variant-tagged decoder + mhr_head
slices first (`sam3d_body_{dinov3,vith}_decoder.safetensors` etc.) and
falls back to the legacy unprefixed names. Per A1.0 inspection the
decoder + head WEIGHTS differ across ckpts up to 0.87 max-abs even
though shapes match — using the right variant slice is mandatory for
correct vertices.

`common/cpu_compute.h` gained an inner `CPU_COMPUTE_IMPL_DEFINED`
sentinel so two consumers in the same TU (here: `dinov3.h` +
`sam3d_body_vit.h`) don't double-define the implementation symbols.

End-to-end smoke (8 threads, person_portrait.jpg):
- `--backbone dinov3` → `body_dinov3.obj` (V=18439 F=36874, 26.4s
  encoder + decoder).
- `--backbone vith`   → `body_vith.obj`   (V=18439 F=36874, 13.9s
  encoder + decoder). Vertex coords differ vs dinov3 (e.g. v0 −0.062
  vs −0.017 X), confirming the variant-specific decoder/MHR weights
  actually drive the output.

The CUDA half originally landed as encoder-only, then A1.6 wired the
same decoder/MHR path for ViT-H. Keep this section as the flag/preprocess
history; current end-to-end status is tracked in A1.6 above.

### A1.4-cuda — CUDA runner --backbone {dinov3,vith} flag (DONE 2026-04-26)

`cuda/sam3d_body/test_cuda_sam3d_body.c` accepts
`--backbone {dinov3,vith}` (default `dinov3`); the runner branches
in `cuda_sam3d_body_run_encoder` on `cfg.backbone == VITH`, and the
host-side preprocess for vit_hmr_512_384 (TopdownAffine to 512×512 +
ImageNet norm + W-axis crop `[:, :, :, 64:-64]` to 512×384 + HtoD
upload) now lives at the top of `run_encoder` next to the existing
DINOv3 host preprocess. The verify_vith path still works because
`debug_set_normalized_input` sets `has_norm_input=1` and the
preprocess block is skipped.

Encoder forward and decoder/MHR now run end-to-end via the CLI for both
variants. The historical VITH `E_INVAL` decoder guard was removed when
A1.6 added the rectangular token/ray/PE handling and variant-specific
decoder/MHR weights.

Smoke (sm_120):
- `--backbone dinov3 -o cuda_dinov3.obj` → V=18439 F=36874 (unchanged).
- `--backbone vith -o cuda_vith.obj` → V=18439 F=36874; encoder +
  decoder + MHR complete.
- `make verify-vith` (CUDA) → max_abs=3.53e-1 mean_abs=1.39e-2 (still
  green; runner refactor didn't regress the verify path).

### A1.3 — CUDA ViT-H encoder + verify_vith green (DONE 2026-04-26)

`cuda/sam3d_body/cuda_sam3d_body_runner.c::cuda_sam3d_body_run_encoder`
branches on `cfg.backbone == VITH` and dispatches to a new
`cuda_sam3d_body_run_encoder_vith` host routine. This section describes
the encoder landing; the later A1.6 work completes CUDA VITH
decoder/MHR as well. Same bf16 floor applies as on CPU.

New / modified GPU kernels in `cuda_sam3d_body_kernels.h`:

- `patch_embed_pad2_f32` — Conv2d kernel=16 stride=16 **padding=2**
  over (3, 512, 384) → (1280, 32, 24). The DINOv3 path's padding=0
  patch embed is not reusable because the +2 zero-pad shifts every
  output pixel's input window.
- `pos_embed_add_vith_f32` — adds learned absolute `pos_embed[1:]`
  per-position then broadcasts `pos_embed[0]` (the legacy CLS slot)
  uniformly to every patch — matches upstream
  `x = x + pos_embed[:, 1:] + pos_embed[:, :1]`.
- `qkv_split_f32` — splits the (B, N, 3·C) qkv tensor into three
  (B, num_heads, N, head_dim) tensors in one launch; needed because
  the existing `flash_attn_tiled_f32` hard-codes `FA_HEAD_DIM=64` via
  `#define`, while ViT-H needs head_dim=80.

Reuse: rather than ship a `flash_attn_tiled_f32_hd80` we route ViT-H
attention through the head-dim-generic `sdpa_f32` kernel +
`qkv_split_f32` helper. Trades a bit of perf for keeping the FA
kernel pinned at hd64 (used by every other consumer in the tree).
Block body otherwise mirrors the CPU pipeline: LN1 → QKV gemm →
qkv_split → sdpa_f32 → proj → +residual → LN2 → fc1 → GELU → fc2 →
+residual, ×32.

`cuda/sam3d_body/verify_vith.c` mirrors the CPU verify: loads
`vith_input.npy` (1, 3, 512, 384) f32 + `vith_tokens.npy`
(1, 1280, 32, 24) f32 from `/tmp/sam3d_body_vith_ref/`, calls
`cuda_sam3d_body_debug_set_normalized_input` (relaxed to accept
512×384 when ctx is VITH), `cuda_sam3d_body_run_encoder`, and diffs
flat against ref. Result: **max_abs=3.53e-1 mean_abs=1.39e-2** —
identical to CPU at the bf16 forward floor. DINOv3 verify still
green at max=9.95e-1 mean=1.11e-2 (untouched). Gate set to
5e-1 max / 2e-2 mean — same A1.2 budget.

`Makefile` gained `verify_vith` build + `verify-vith` run target with
`VITH_REFDIR=/tmp/sam3d_body_vith_ref`.

### A1.2 — CPU ViT-H encoder + verify_vith green (DONE 2026-04-26)

`common/sam3d_body_vit.h` — vanilla ViT-H/16 forward over (768, 1280)
tokens. Distinct enough from `common/dinov3.h` to ship as a separate
header rather than a retrofit:

- PatchEmbed conv kernel=16 stride=16 **padding=2** (DINOv3 path uses
  padding=0; the +2 pad shifts every output token's input window —
  the dedicated patch_embed in `s3dvit_patch_embed` zero-pads by 2 on
  all four sides).
- Absolute LEARNED `pos_embed` (1, 769, 1280) — slot 0 is added
  uniformly to every patch (legacy CLS bias from upstream init), then
  slots 1..768 are the per-position embeddings.
- 32 × `Block(LN1 → attn → +x → LN2 → mlp(GELU) → +x)`. No QK norm,
  no LayerScale, no register tokens, no RoPE.
- **head_dim=80** (1280/16). `cpu_compute.h::cpu_attn_worker` AVX2
  path hard-codes head_dim=64 in its 8-register Q layout; we ship a
  dim-generic `s3dvit_attn_worker` inline. AVX2 8-lane vector loops
  over hd; tile size 64 along K to amortize Q reloads. Online softmax
  matches dinov3.
- GELU is the **exact erf form** `0.5·x·(1+erf(x/√2))`, NOT
  tanh-approx — the upstream `nn.GELU()` defaults to
  `approximate='none'`. Tanh-approx accumulated +0.2 max-abs over 32
  blocks vs erf, so we eat the slightly-slower erf instead.
- LayerNorm32 in upstream is "cast to fp32 → LN → cast back"; we
  already keep activations fp32 throughout, so we drop the cast
  wrapper and just call plain LN with eps=1e-6.

`cpu/sam3d_body/verify_vith.c` loads `vith_input.npy`
(1, 3, 512, 384) f32 + `vith_tokens.npy` (1, 1280, 32, 24) f32 from
`/tmp/sam3d_body_vith_ref/` (produced by gen_image_ref.py with the
ViT class hook), runs `sam3d_body_vit_encode_from_normalized` on the
ImageNet-norm + W-axis-cropped tensor, diffs at the (py, px, d)
level. Result: **max_abs=3.53e-1 mean_abs=1.39e-2** at 8 threads,
13.6s wall.

CRITICAL pitfall logged: the **bf16 forward floor**. Upstream runs
the backbone in bf16 (`FP16_TYPE: bfloat16` in
`vith/model_config.yaml`), so the dumped tokens are bf16-rounded
compounded over 32 blocks. Our fp32 forward drifts max≈3.5e-1
mean≈1.4e-2 from that — verified IDENTICAL with fp32 weights
(0% mantissa quant) AND with f16 weights (sliced default). The drift
is purely the bf16-vs-fp32 forward mismatch, not a quant or kernel
bug. Gate set to 5e-1 max / 2e-2 mean to track this floor; mean is
the tight catch (a real port bug typically blows up mean before max).
DINOv3 variant hits 1.5e-1 max on the same comparison because the
SwiGLU·LayerScale·RoPE pattern compounds bf16 drift differently from
GELU MLP — same regime, ViT-H just sits 2.3× hotter. Do not tighten
the ViT-H gate without first switching to a bf16 forward path.

### A1.1 — convert_ckpt.py per-variant slices (DONE 2026-04-26)

`cpu/sam3d_body/convert_ckpt.py` now takes
`--backbone-name {dinov3,vith}` (default: `dinov3`) and emits all
three buckets with a variant tag:

```
sam3d_body_<variant>.safetensors            # encoder backbone
sam3d_body_<variant>_decoder.safetensors    # promptable decoder
sam3d_body_<variant>_mhr_head.safetensors   # MHR head + camera
```

For backwards compatibility, when `--backbone-name dinov3` the
slicer ALSO writes the legacy un-tagged filenames
`sam3d_body_decoder.safetensors` +
`sam3d_body_mhr_head.safetensors` (the dinov3 backbone file is
already named `sam3d_body_dinov3.safetensors` — matches the legacy
convention). This keeps every existing
`cpu/sam3d_body/verify_*.c` + `cuda/sam3d_body/verify_*.c` working
without code change. ViT-H consumers must use the tagged names
exclusively.

**CRITICAL pitfall logged**: the two ckpts ship SEPARATELY trained
weights. Decoder + head shapes are identical (524 + 52 tensors per
variant), but values differ by up to ~0.87 max-abs. A first-pass
naive design that re-used a single `sam3d_body_decoder.safetensors`
across variants would silently mix DINOv3-trained decoder weights
with ViT-H backbone outputs. Per-variant tagging on EVERY trainable
slice is non-negotiable.

Final slice inventory at
`/mnt/disk01/models/sam3d-body/safetensors/`:

```
sam3d_body_dinov3.safetensors           1.6 GB  (legacy + variant)
sam3d_body_dinov3_decoder.safetensors   374 MB
sam3d_body_dinov3_mhr_head.safetensors   77 MB
sam3d_body_decoder.safetensors          374 MB  (legacy alias)
sam3d_body_mhr_head.safetensors          77 MB  (legacy alias)
sam3d_body_vith.safetensors             1.27 GB
sam3d_body_vith_decoder.safetensors     374 MB
sam3d_body_vith_mhr_head.safetensors     77 MB
```

### A1.0 — inspect ViT-H ckpt + document arch (DONE 2026-04-26)

Confirmed via `cpu/sam3d_body/inspect_ckpt.py
/mnt/disk01/models/sam3d-body/vith/model.ckpt`:

- Backbone: 389 tensors (vs DINOv3 variant's 551). All other
  modules (`decoder`/`decoder_hand`/`prompt_encoder`/`head_pose`/
  `head_camera`/`bbox_embed`/`ray_cond_emb`/etc) are tensor-count
  IDENTICAL between the two variants → only the backbone differs.
  Per-stage decoder + MHR head + skinning code in
  `common/sam3d_body_*.h` reuses verbatim; ckpt slicer just needs
  a different backbone slice.
- Backbone source: `sam_3d_body/models/backbones/vit.py::vit512_384`
  (factory) → `class ViT` (class). NOT DINOv3 — a vanilla
  ViT-H/16 with `embed_dim=1280`, `depth=32`, `num_heads=16`,
  `mlp_ratio=4`, `qkv_bias=True`, GELU MLP, pre-LN blocks. No
  layer-scale, no register tokens, no rotary pos-emb (absolute
  learned pos_embed only), no SwiGLU.

Tensor manifest (only structural items shown — 32 blocks expand to
12 tensors each):

```
backbone.patch_embed.proj.weight  (1280, 3, 16, 16)  bf16
backbone.patch_embed.proj.bias    (1280,)            bf16
backbone.pos_embed                (1, 769, 1280)     bf16
backbone.blocks.{0..31}.norm1.{weight,bias}  (1280,)     fp32
backbone.blocks.{0..31}.attn.qkv.{weight,bias}
                                  (3840, 1280) / (3840,) bf16
backbone.blocks.{0..31}.attn.proj.{weight,bias}
                                  (1280, 1280) / (1280,) bf16
backbone.blocks.{0..31}.norm2.{weight,bias}  (1280,)     fp32
backbone.blocks.{0..31}.mlp.fc1.{weight,bias}
                                  (5120, 1280) / (5120,) bf16
backbone.blocks.{0..31}.mlp.fc2.{weight,bias}
                                  (1280, 5120) / (1280,) bf16
backbone.last_norm.{weight,bias}  (1280,)            fp32
```

Total 389 tensors (= 32 × 12 + 5).

Image flow differs from DINOv3 (CRITICAL for ViT-H lane):

- DINOv3 variant: TopdownAffine warps to 512×512 → ImageNet norm →
  backbone takes (1, 3, 512, 512) → produces tokens for 32×32=1024
  patches → `(1, 1280, 32, 32)` feature map.
- ViT-H variant: TopdownAffine warps to 512×512 → ImageNet norm →
  **center-crop W axis `[:, :, :, 64:-64]`** to 512×384 →
  backbone takes (1, 3, 512, 384) → 32×24=768 patches →
  `(1, 1280, 32, 24)` feature map. Decoder consumes the same
  channel count but a different spatial extent — verify decoder
  does not hard-code `H==W==32`.

Source: `sam_3d_body/models/meta_arch/base_model.py::data_preprocess`
lines 51–66; W-axis crop is gated on `BACKBONE.TYPE == "vit_hmr_512_384"`.

PatchEmbed details (`vit.py::PatchEmbed`):
- Conv2d(3 → 1280, kernel=16, stride=16, padding=2)
- With padding=2 and 512×384 input:
  - H out = floor((512 + 4 − 16) / 16) + 1 = 31 + 1 = 32 ✓
  - W out = floor((384 + 4 − 16) / 16) + 1 = 23 + 1 = 24 ✓
- Padded patch embed (DINOv3 path uses padding=0; do NOT reuse the
  DINOv3 `patch_embed_sam3d` kernel verbatim — the padding term
  shifts every output pixel).

Forward shape after `forward_features` (vit.py:627–649):
1. PatchEmbed → flat (B, 768, 1280) + (Hp=32, Wp=24)
2. `x = x + pos_embed[:, 1:] + pos_embed[:, :1]` (broadcast slot 0
   uniformly to all patches)
3. Optional `extra_embed.flatten(2).transpose(1, 2)` add
   (`ray_cond_emb` output — same path as DINOv3, gates on
   `MODEL.PROMPT_ENCODER.MASK_PROMPT`)
4. 32 × `Block(norm1 → attn → +x → norm2 → mlp(GELU) → +x)`
5. `last_norm(x)`
6. permute+reshape → `(B, 1280, 32, 24)`

LayerNorm32 (modules/transformer.py:33) = LayerNorm in fp32 with
input/output cast preserved. Our CPU port already keeps activations
in fp32 throughout — drop the cast wrapper, just use plain LN.

Diffs we must enforce in the C port:
1. `common/sam3d_body_vit.h` (NEW): plain ViT-H/16 forward over
   `(N=768, C=1280)`. Pre-LN; GELU (NOT SwiGLU); no layer scale;
   no register tokens; no RoPE — pos_embed is fully absolute and
   learned. Reuse the GEMM helpers from `common/cpu_compute.h`;
   reuse the LN/SwiGLU-less attention from existing transformer
   primitives.
2. `sam3d_body_preprocess_image` host-side path needs a variant
   flag; for ViT-H, after the existing ImageNet norm, slice the
   contiguous `(1, 3, 512, 512)` tensor down to `(1, 3, 512, 384)`
   by dropping the first 64 and last 64 columns of width. Single
   memcpy per row × channel.
3. Decoder consumer: spatial pos-embed table + cross-attention key
   layout must accept H=32, W=24. Validate by reading
   `common/sam3d_body_decoder.h` for any hard-coded `32*32`
   assumption before A1.2.
4. Patch-embed conv has `padding=2`, kernel=16, stride=16 — write
   a small dedicated kernel rather than retargeting the
   DINOv3 (padding=0) one.

Plan for A1.1–A1.5:
- A1.1: extend `cpu/sam3d_body/convert_ckpt.py` with
  `--backbone-name {dinov3,vith}` to slice all three buckets per
  variant. Per-variant tagging on every output is mandatory because
  decoder + head VALUES differ across the two ckpts (sanity-check
  shows max-abs diff 0.12 on `decoder.layers.0.self_attn.q_proj.weight`,
  0.87 on `head_pose.proj.layers.1.weight`).
- A1.2: implement `common/sam3d_body_vit.h::sam3d_body_vit_forward`,
  add `cpu/sam3d_body/verify_vith.c` diffing against PyTorch ref
  dump of `image_embeddings` from the ViT-H variant.
- A1.3: CUDA mirror in `cuda/sam3d_body/cuda_sam3d_body_kernels.h`.
- A1.4: add `--backbone {dinov3,vith}` flag to CPU + CUDA runners
  (default dinov3); when `vith` is selected, load the alternate
  backbone safetensors + force the W-axis crop.
- A1.5: web demo: backbone radio (dinov3 / vith) added to the
  request body; server forwards via `--backbone` flag; UI shows
  side-by-side rendering of both variants for the same image.

## Notes / risks

- Deformable attention is the only nontrivial new primitive. CPU
  implementation is straightforward (4 bilinear samples per query
  per level per head, weighted sum). CUDA port can reuse for
  later A2.4.
- Per-channel BN-fold pass at load time (BN params → scale+bias on
  conv weights/bias) keeps inference stateless. Mirror the SAM3.1
  resnet stem trick.
