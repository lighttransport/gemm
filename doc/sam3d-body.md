# SAM 3D Body — Current State and TODOs

Snapshot: 2026-04-27.

FAIR's `sam-3d-body`: single-image, full-body 3D human-mesh recovery.
Encoder–decoder (DINOv3-H+ backbone → promptable decoder → MHR-head
regressor) followed by Momentum Human Rig (MHR) skinning that turns
regressed parameters into a 3D mesh + 3D/2D keypoints.

Live progress journal (authoritative): `cpu/sam3d_body/PORT.md`.
This file is a flat, doc-friendly mirror of the current state plus
remaining work — written so a fresh reader can pick up the port without
reading the journal end-to-end.

## Pipeline

```
RGB + bbox
    │
    ├─ TopdownAffine (compute_bbox_affine, padding=1.25, fix_aspect 0.75)
    ├─ preprocess_image (cv2.warpAffine + HWC→CHW + ImageNet norm)
    │
    ▼
DINOv3-H+ encoder  (32 blocks · D=1280 · patch=16 · 512²→32×32)
    │   tokens (B, 1029, 1280)  ← 1024 patch + 1 cls + 4 storage
    │
    ▼
ray_cond_emb  (Fourier 3→99 + Conv1×1 1379→1280 + LN2d, ε=1e-6)
    │   image_emb (B, 1280, 32, 32)
    │
    ▼
build_tokens  (525→1024 init · 522→1024 prev · 1280→1024 prompt
              + hand_box(2) · kp_emb(70) · kp3d_emb(70))
    │   x (145, 1024) · x_pe (145, 1024)
    │
    ▼
6 × TransformerDecoderLayer  (SA H=8 Dh=64 + CA q←x k,v←ctx + FFN
              ReLU 1024→1024)
    │   per-layer: norm_final + heads + decode_pose_raw + mhr_forward
    │              + camera_project + kp_token_update
    │
    ▼
MHR skinning  (parameter_transform · joint_params→local→global walker
              · blend_shape (45) · face_expressions (72) · pose_correctives
              (6D feat → COO sparse 53136 → ReLU → dense 55317×3000) ·
              LBS skin_points)
    │   verts (18439, 3) · faces (36874, 3) · keypoints_3d/2d
    │
    ▼
.obj (vertices + faces) + keypoints_{3d,2d}
```

## Layout

```
cpu/sam3d_body/                     # CPU port + CLI + verify_*.c
cuda/sam3d_body/                    # CUDA port (NVRTC) + verify_*.c
common/sam3d_body_decoder.h         # transformer decoder + MHR head
common/sam3d_body_mhr.h             # MHR skinning (pure C; pymomentum-port)
common/sam3d_body_vit.h             # DINOv3-H+ generic-Dh ViT (Dh=80)
common/dinov3.h                     # shared DINOv3 backbone (also used by da3)
common/npy_io.h                     # promoted shared NPY reader
ref/sam3d-body/                     # dedicated venv + gen_image_ref.py
/mnt/disk01/models/sam3d-body/      # checkpoints + MHR assets
```

## Status — drift table (mirrors PORT.md)

All v1 stages GREEN. `max_abs` / `mean_abs` are vs the PyTorch
reference unless noted.

| Stage                       | Status | max_abs   | mean_abs | Notes                                                                                  |
|-----------------------------|--------|-----------|----------|----------------------------------------------------------------------------------------|
| Step 0 bootstrap            | GREEN  | —         | —        | legacy bootstrap complete; obsolete compatibility stubs fail explicitly                |
| Step 1 slicer               | GREEN  | —         | —        | 160 f16 + 391 f32 dinov3 / 524 fp32 decoder / 52 fp32+i64 mhr_head; 0 unmatched        |
| DINOv3 encoder (CPU)        | GREEN  | 1.52e-1   | 4.01e-3  | f16 floor for 32-block ViT-H+; 24.2 s e2e @ 8 threads                                  |
| DINOv3 encoder (CUDA)       | GREEN  | 9.95e-1   | 1.11e-2  | NVRTC 32-block ViT, F16 weights; CUDA-vs-CPU diff ≈1e-5                                |
| ray_cond_emb (CPU)          | GREEN  | 9.49e-4   | 1.40e-4  | Fourier (16 bands) + Conv1×1 (1379→1280, F32) + LN2d                                   |
| ray_cond_emb (CUDA)         | GREEN  | 9.51e-4   | 1.46e-4  | 3 NVRTC kernels; CHW layout                                                            |
| build_tokens (CPU)          | GREEN  | 4.77e-7   | 8.54e-10 | 3 linears + concat + zero-PE except slots 1/2                                          |
| build_tokens (CUDA)         | GREEN  | 3.81e-6   | 5.21e-9  | 3 linear_f32_bias launches + 3 d2d memcpy                                              |
| decoder layer fwd (CPU)     | GREEN  | 3.8e-6    | 3.1e-7   | SA + CA + FFN(ReLU); per-layer at f32 floor; layer 0 skip_first_pe                     |
| decoder layer fwd (CUDA)    | GREEN  | 1.14e-5   | 6.23e-7  | gemm_f32_bias (16×16 tiles), sdpa_f32 (per-(n_q,h) block, online softmax)              |
| kp_token_update (CPU)       | GREEN  | 1.4e-6    | 3.2e-8   | grid_sample + linear + augment overwrite; layer 5 short-circuits                       |
| kp_token_update (CUDA)      | GREEN  | 4.05e-6   | 6.21e-8  | Param-struct alignment fix: don't __packed__ ptr/int interleavings                     |
| norm_final + heads (CPU)    | GREEN  | 1.9e-6    | 1.4e-7   | head FFN activation is **ReLU** (not GELU); both heads default                         |
| norm_final + heads (CUDA)   | GREEN  | 3.81e-6   | 2.92e-7  | reuses ln + gemm_f32_bias + relu_inplace                                               |
| MHR head decode             | GREEN  | 9.5e-7    | 7.2e-9   | rot6d → ZYX-Euler, body_cont → 133, scale + per-side hand PCA                          |
| MHR skinning                | GREEN  | 8.4e-5    | 1.0e-5   | parameter_transform, walker (4 stages [65,56,62,83]), pose_correctives sparse+dense    |
| camera_project              | GREEN  | 1.3e-4    | 6.96e-5  | PerspectiveHead + _full_to_crop; ori_img_size for projection (not 512)                 |
| Decoder + MHR head (preset) | GREEN  | 1.9e-6    | 1.6e-7   | end-to-end body branch                                                                 |
| decoder forward_full (CPU)  | GREEN  | 5.5e-4    | 4.3e-4   | iterative MHR-in-the-loop                                                              |
| decoder forward_full (CUDA) | GREEN  | 6.10e-4   | 4.88e-4  | mirrors CPU; CPU MHR shared                                                            |
| warp_matrix                 | GREEN  | 1.3e-5    | —        | TopdownAffine                                                                          |
| preprocess_image            | GREEN  | 5.25e-2   | 5e-4     | cv2 INTER_TAB_SIZE=32 fixed-point variation                                            |
| End-to-end verts (CPU)      | GREEN  | 9.5e-7    | 1.2e-7   | override path vs `out_*` refs                                                          |
| End-to-end raw image (CPU)  | GREEN  | 2.47e-2   | 4.09e-3  | fixed bbox vs `body_out_*`; 2D max 6.75 px                                             |
| End-to-end raw image (CUDA) | GREEN  | 2.43e-2   | 4.06e-3  | fixed bbox vs `body_out_*`; 2D max 6.78 px                                             |
| ViT-H override (CPU)        | GREEN  | 1.37e-6   | 1.64e-7  | rectangular kp-update fixed; 2D max 4.88e-4 px                                         |
| ViT-H raw image (CPU/CUDA)  | GREEN  | 1.26e-1   | 3.11e-2  | fixed bbox vs `body_out_*`; 2D max 162.7 px; encoder precision floor                   |

## Steps — gate status

| Step | Deliverable                                              | Status            |
|------|----------------------------------------------------------|-------------------|
| 0    | Scaffolds + ref venv + `--skip-run` NPY                  | GREEN             |
| 1    | `convert_ckpt.py` slicer                                 | GREEN             |
| 2    | CPU DINOv3 encoder                                       | GREEN             |
| 3    | CUDA DINOv3 encoder                                      | GREEN (2026-04-25)|
| 4    | CPU promptable decoder + MHR head                        | GREEN             |
| 5    | CUDA decoder + MHR head                                  | GREEN (2026-04-25)|
| 6    | CPU MHR skinning                                         | GREEN             |
| 7    | CUDA MHR skinning                                        | **CLOSED via CPU OpenMP** (5.9× at 16 thr; MHR 183→31 ms; e2e 3.47→2.84 s) |
| 8    | End-to-end CPU `.obj`                                    | GREEN (ref + self-driven) |
| 9    | End-to-end CUDA `.obj`                                   | GREEN (2026-04-25)|
| 10   | Shared `common/npy_io.h` promotion                       | GREEN (2026-04-25); `qtensor_utils.h` partial |

## Speculative work landed (off the production path)

- **norm_final + head_pose + head_camera moved to CPU** (2026-04-26).
  The decoder tail (LayerNorm + 2 small head MLPs producing 519 pose
  + 3 cam scalars) ran on GPU and got called between every decoder
  layer + post-loop. Total compute ~2.6 MFLOP/call; compared to the
  full-tokens upload + GPU GEMM chain + multi-buffer download, CPU is
  the right home. Implementation: `cuda_sam3d_body_debug_run_norm_and_heads`
  now uses `cpu_compute.h::cpu_gemm_f32` (OpenMP, AVX2/FMA) with
  weights mmap'd directly from the safetensors blob (no copy). Verifies:
  norm_final max_abs=4.77e-6, head_pose=1.91e-6, head_camera=9.54e-7
  vs Python reference. **Port policy:** for not-yet-ported modules
  with `<10 MFLOP` per call and host-side neighbors, default to CPU.
  GPU weights for these tensors are still uploaded at load (~2.6 MB,
  unused but harmless); not worth a flagged-load refactor.

- **Speculative MHR-on-GPU helpers** (2026-04-26).
  `cuda/sam3d_body/cuda_sam3d_body_runner.[ch]` adds four debug
  helpers (`_debug_run_blend_shape`, `_debug_run_face_expressions`,
  `_debug_run_pose_correctives`, `_debug_run_lbs_skin`) plus two NVRTC
  kernels (`mhr_blend_combine_f32`, `mhr_lbs_skin_f32`).
  `cuda/sam3d_body/verify_mhr` validates them against the CPU MHR
  reference: blend_shape / face_expressions are bit-exact;
  pose_correctives matches at 3.58e-7 (1 ULP); lbs_skin
  max_abs=2.29e-5 (atomicAdd reduction-order drift on cm-scale verts,
  expected). **Step 7 remains CLOSED** — these helpers are
  exploratory; production uses CPU OpenMP MHR.

## Critical pitfalls / gotchas

- **`enable_hand_model` selects MHR-head branch.** Body branch (flag=0)
  keeps ZYX-Euler global_rot + global_trans=0, no nonhand_param zeroing,
  all 70 keypoints. Hand branch (flag=1) re-projects through
  `local_to_world_wrist`, applies nonhand_param zeroing (145/204), zeros
  kp[:21] and kp[42:].
- **MHR-head FFN activation is ReLU, not GELU.**
- **`batch6DFromXYZ` output is column-stacked** [R00,R10,R20,R01,R11,R21]
  — row-stacked attempts fail stage 10C.
- **camera_project's `img_size` arg is `ori_img_size` (1693×952)**, not
  the model input 512×512. The 512 is only used in the crop normalize.
- **Final `pred_vertices` applies Y/Z negation** ("post-flip cam frame
  in m" in `decoder_forward_full`). CUDA reimplementations bypassing
  this path must replicate the flip.
- **CUDA host preprocess uses `sam3d_body_preprocess_image` (cv2
  warpAffine + ImageNet norm).** A naive on-device bilinear resize
  fails the encoder.
- **NVRTC kernel param structs interleaving 8-byte ptrs and 4-byte ints
  CANNOT use `__attribute__((packed))`** — natural alignment inserts a
  4-byte pad after a leading int that `packed` strips, silently zeroing
  half the augment rows. Reorder so all ptrs come first.

## TODOs / open work

### Deferred to v1.1

- **~~`sam3d_body_decoder_forward` (full self-MHR iterative path)~~ — landed 2026-04-26.**
  Public entry point now bundles ray_cond_emb + get_dense_pe +
  invalid_prompt_token + build_tokens and delegates to `forward_full`
  (the iterative production path). Sig widened to take MHR + cam_batch
  + (H, W) + rays_hwc + condition_info; previous stub had no callers
  so it's a safe break. Same math as the runner's `run_decoder` chain,
  packaged for direct use from a TU that already holds DINOv3 patch
  tokens + a camera_batch.
- **Raw-image self-driven verifier is no longer bit-exact by design.**
  `gen_image_ref.py` now writes `body_out_{vertices,keypoints_3d,keypoints_2d}.npy`
  before upstream hand prompting/refinement mutates the final `out_*`
  arrays. CPU/CUDA `verify_end_to_end --image ... --bbox ...` prefer
  these body-branch refs. The remaining tolerance is the known
  PyTorch-vs-C DINOv3 encoder/preprocess floor, not decoder/MHR drift.
- **Keys bicubic pos-embed interp generalization in `common/dinov3.h`.**
  Required if DINOv3-H+ is fed a non-square input (current only
  supports square via the existing path).

### Shared-infra cleanups (sam3d + sam3d_body)

- **`common/qtensor_utils.h` — finish factoring out duplicated
  helpers.** `qtensor` struct + `qt_make_tensor()` are shared.
  `qt_dequant`, `qt_dequant_row`, `_layernorm_batch`, `_layerscale`
  remain copy-pasted across model headers — revisit when a third
  consumer surfaces or when a real bug surfaces in the duplicates.

### Speculative / exploratory

- **Full GPU MHR port (~1500 NVRTC lines).** Step 7 is CLOSED;
  cost/benefit no longer favors a full GPU port. The speculative
  helpers are validation-only. Revisit only if MHR ever returns to
  the critical path.
- **CUDA decoder bit-exact parity with CPU.** Currently CUDA-vs-CPU
  diff is ≈1e-5 on the decoder; tolerable, but a tighter parity
  audit could close out the f32-floor mystery offsets if needed.

## Build / run

```bash
# CPU port
cd cpu/sam3d_body && make all                    # binaries + verify_*
./test_sam3d_body --safetensors-dir <SFT_DIR> \
    --mhr-assets <MHR_DIR> \
    --image person.jpg --bbox 100 100 600 900 \
    -o body.obj

# CUDA port
cd cuda/sam3d_body && make all
./test_cuda_sam3d_body --safetensors-dir <SFT_DIR> \
    --mhr-assets <MHR_DIR> \
    --image person.jpg --bbox 100 100 600 900 \
    -o body.obj

# Per-stage verify (CPU + CUDA each have ~10–18 verify_*.c)
./verify_dinov3 --safetensors-dir <SFT_DIR> --refdir /tmp/sam3d_body_ref
./verify_decoder ...
./verify_end_to_end --safetensors-dir <SFT_DIR> --mhr-assets <MHR_DIR> \
    --refdir /tmp/sam3d_body_ref --image person.jpg --bbox x0 y0 x1 y1 \
    --threshold 3e-2 --threshold-2d 10

# CPU/CUDA Makefile raw-image aliases; set both image and fixed bbox.
cd cpu/sam3d_body
make verify-end-to-end-raw REFDIR_DINOV3=/tmp/sam3d_body_ref \
    E2E_IMAGE=samples/dancing.jpg \
    E2E_BBOX='727.537 111.031 1534.146 1456.042'

cd ../../cuda/sam3d_body
make verify-end-to-end REFDIR_DINOV3=/tmp/sam3d_body_ref \
    E2E_IMAGE=../../cpu/sam3d_body/samples/dancing.jpg \
    E2E_BBOX='727.537 111.031 1534.146 1456.042'

# ViT-H fixed-bbox raw-image check; wider because the converted
# ViT-H encoder weights are the known limiting precision floor.
./verify_end_to_end --safetensors-dir <SFT_DIR> --mhr-assets <MHR_DIR> \
    --refdir /tmp/sam3d_body_vith_ref --image person.jpg \
    --bbox x0 y0 x1 y1 --backbone vith \
    --threshold 1.5e-1 --threshold-2d 200

# Reference dumps (one-time)
cd ref/sam3d-body && source .venv/bin/activate
python gen_image_ref.py --image person.jpg --bbox x0 y0 x1 y1 \
    --hf-repo-id facebook/sam-3d-body-dinov3 \
    --outdir /tmp/sam3d_body_ref --seed 42

python gen_image_ref.py --image person.jpg --bbox x0 y0 x1 y1 \
    --local-ckpt-dir /mnt/disk01/models/sam3d-body/vith \
    --outdir /tmp/sam3d_body_vith_ref --seed 42
```

With `--bbox`, `gen_image_ref.py` writes both final upstream
`out_*` tensors and body-branch `body_out_*` tensors. The latter are
the parity target for the current C runners because upstream final
output includes hand-prompt refinement that is outside the C pipeline.
Per-stage decoder tensors are guarded the same way: `decoder_layer*_in`,
`decoder_out_norm_final`, `head_pose_proj_raw`, `head_camera_proj_raw`,
ray-cond tensors, and token-construction tensors preserve the first body
decoder pass instead of being overwritten by later keypoint/hand passes.
The MHR head decode verifier is variant-aware and checks the guarded
body branch for both DINOv3 and ViT-H at f32-level. `verify_end_to_end`
also fails empty/missing refs now, so shape or backbone mismatches do
not produce `nan OK`.

## End-to-end run (worked example, 2026-04-27)

Recipe that takes a raw RGB image with no ref dumps and produces a 3D
mesh + rig parameters + keypoints. Sample: upstream
`sam-3d-body/notebook/images/dancing.jpg` (2250×1500), copied to
`cpu/sam3d_body/samples/dancing.jpg`.

```bash
cd cpu/sam3d_body
./test_sam3d_body \
    --safetensors-dir /mnt/disk01/models/sam3d-body/safetensors \
    --mhr-assets      /mnt/disk01/models/sam3d-body/safetensors \
    --rt-detr-model   /mnt/disk01/models/rt_detr_s/model.safetensors \
    --image samples/dancing.jpg --auto-bbox \
    -o samples/dancing.obj -t 8
```

Output (≈33 s wall on the local AMD/x86 box, single image, 8 threads;
DINOv3-H+ encoder dominates at ~21 s):

```
[test_sam3d_body] auto-bbox: detector=rt-detr-s score=0.9410 threshold=0.500 image=1600x1600 bbox=(727.5,111.0,1534.1,1456.0)
dinov3: preprocess+embed 273.9 ms, backbone 21188.3 ms (8 threads)
[test_sam3d_body] wrote samples/dancing.obj (V=18439 F=36874)
[test_sam3d_body] wrote samples/dancing.obj.json (mhr_params=519 kp3d=70 kp2d=70)
```

Two files land next to the image:

| File                       | Format | Contents                                                                                |
|----------------------------|--------|-----------------------------------------------------------------------------------------|
| `dancing.obj`              | OBJ    | 18 439 vertices · 36 874 triangle faces (camera-space metres, post Y/Z flip)            |
| `dancing.obj.json`         | JSON   | `bbox` (px), `bbox_source`, optional `auto_bbox{detector,score,threshold}`, `image{w,h}`, `focal_px`, `cam_t[3]`, `mhr_params[519]`, `keypoints_3d[70][3]` (m), `keypoints_2d[70][2]` (px in original image) |

Use `-o dancing.glb` to write the same mesh as binary glTF/GLB. The
sidecar is still written next to it as `dancing.glb.json`. Body GLB
export is geometry-only for now; unlike SAM 3D Objects there is no
decoder RGB or pointmap texture source in the body runner.

`mhr_params` is the raw 519-vector after the regression head (pose +
shape decoded by `decode_pose_raw`); pair with the published MHR
kintree if you need joint angles. `cam_t` is the camera translation in
metres; `focal_px` is the assumed pinhole focal in original-image pixels
(default = `1.2 * max(W, H)` unless `--focal F` is passed).

### Bbox sources

- `--auto-bbox` runs RT-DETR-S (single forward, ~1 s) and picks the
  highest-scoring valid person, using larger area only as a tie-break
  (default `--auto-thresh 0.5`).
- `--bbox x0 y0 x1 y1` skips detection — useful when the image already
  has a tight crop or RT-DETR mis-fires on stylised inputs.
- Either way, the bbox is padded to `1.25×` and aspect-fixed to 0.75
  by `compute_bbox_affine` before the encoder sees it.

### CUDA variant

The CUDA runner takes the same arguments plus a CUDA device:

```bash
cd cuda/sam3d_body
./test_cuda_sam3d_body \
    --safetensors-dir /mnt/disk01/models/sam3d-body/safetensors \
    --mhr-assets      /mnt/disk01/models/sam3d-body/safetensors \
    --rt-detr-model   /mnt/disk01/models/rt_detr_s/model.safetensors \
    --image samples/dancing.jpg --auto-bbox \
    -o samples/dancing_cuda.obj
```

Wall time on a recent NVIDIA card: ≈3 s (≈10× CPU). Vertex / keypoint
agreement with the CPU runner is at ≈1e-6 m (see drift table — "End-to-end (CUDA)").
The CUDA CLI now mirrors CPU output metadata: `.obj.json` / `.glb.json`
sidecars include bbox, image size, focal, camera translation, MHR params,
and 3D/2D keypoints.

ViT-H is supported by both CPU and CUDA runners. Current CUDA smoke:

```bash
./cuda/sam3d_body/test_cuda_sam3d_body \
    --safetensors-dir /mnt/disk01/models/sam3d-body/safetensors \
    --mhr-assets /mnt/disk01/models/sam3d-body/safetensors \
    --image cpu/sam3d_body/samples/dancing.jpg \
    --bbox 727.537 111.031 1534.146 1456.042 \
    --backbone vith -o /tmp/sam3d_body_vith_cuda.obj
```

This writes `V=18439 F=36874` plus the JSON sidecar. ViT-H decoder
verification is exact after the rectangular keypoint-update fix:
CPU `verify_end_to_end --backbone vith` without `--image` reports
vertices max_abs=1.37e-6, keypoints_3d max_abs=5.96e-7, and
keypoints_2d max_abs=4.88e-4 px. CPU `verify_dense_pe --backbone vith`
matches upstream's centered 24-column crop from the 32x32 PE grid at
max_abs=1.91e-6; CPU `verify_ray_cond_xyz` matches the centered x-crop
ray tensor at max_abs=5.96e-8. CPU `verify_decoder_forward --backbone
vith` covers the public wrapper path with final keypoints_3d max_abs=
2.94e-5. CUDA ViT-H per-stage verifiers now cover ray-cond (32x24),
token build, keypoint-update, norm/heads, decoder layers, and the full
iterative decoder loop; `verify_decoder` with `--backbone vith` reports
final `head_pose_proj_raw` max_abs=9.54e-6 and final keypoints_3d
max_abs=8.08e-7. Raw-image ViT-H verification passes for
both CPU and CUDA with
`--threshold 1.5e-1 --threshold-2d 200`; the observed envelope is
vertices max_abs=1.26e-1, keypoints_3d max_abs=1.24e-1, and
keypoints_2d max_abs=162.7 px.

## Web demo (3-pane: input / ours / pytorch, 2026-04-27)

Side-by-side compare server that runs both backends on the same input
and renders the meshes with three.js. Lives under `server/sam3d/` (a
fresh layout — the older single-backend `ref/sam3d-body/sam3d_body_server.py`
is left untouched).

```
server/sam3d/
  app.py             ThreadingHTTPServer; /v1/infer endpoint
  pytorch_runner.py  in-process PyTorch backend (loads model once at startup)
  run.sh             launcher (uses ref/sam3d-body/.venv)
  README.md
web/sam3d_demo.html  3-pane web UI (input image · ours mesh · pytorch mesh)
```

Backends:
- **ours** — subprocess to `cpu/sam3d_body/test_sam3d_body`; reads its
  `.obj` + `.obj.json` sidecar from a tmpdir per request.
- **pytorch** — in-process via `pytorch_runner.load()`; the model is
  loaded once at server startup and served directly (no per-request
  Python launch). Refactored `ref/sam3d-body/run_pytorch_pipeline.py`
  to expose `load_model()` and `run()` for this; the script's CLI
  `main()` is preserved.

Both backends emit the same sidecar JSON shape (`bbox`, `image{w,h}`,
`cam_t`, `mhr_params`, `keypoints_3d`, `keypoints_2d`) so the web UI
renders them uniformly.

### Run

```bash
bash server/sam3d/run.sh             # → http://localhost:8765
bash server/sam3d/run.sh --no-pytorch # ours-only (skip torch model load)
```

Open the URL, upload an image (or use `cpu/sam3d_body/samples/dancing.jpg`),
optionally drag a bbox on the preview (or leave **Auto-detect bbox**
checked to let RT-DETR-S find one), then click **Run both backends**.

The web UI shows three panes: the input image with the bbox annotated,
the ours mesh in a three.js OrbitControls viewer, and the pytorch
mesh in a second viewer. Per-pane meta lines surface `time`, `V/F`,
`bbox`, `cam_t`, `focal_px`.

### API

`POST /v1/infer` body:

```json
{ "image_b64": "...",
  "bbox":         [x0, y0, x1, y1],   // optional; auto-bbox if omitted
  "auto_thresh":  0.5,
  "image_ext":    "jpg",
  "backends":     ["ours", "pytorch"] }
```

Response:

```json
{ "ours":     { "obj_b64", "json", "timing_ms", "bbox_used" } | null,
  "pytorch":  { "obj_b64", "json", "timing_ms", "bbox_used" } | null,
  "errors":   { "ours"?: str, "pytorch"?: str },
  "bbox_used": [...],
  "timings_ms": { "total": ... } }
```

When `bbox` is omitted, the server runs ours first (auto-bbox via
RT-DETR-S) and feeds the resulting bbox into the pytorch backend so
both meshes are produced for the same crop. The pytorch backend has no
person detector bundled and always requires a bbox.

The newer single-backend `ref/sam3d-body/sam3d_body_server.py` accepts
`params.output_format: "obj"|"glb"` for the C runner backends and returns
`outputs[0].mime` as `model/obj` or `model/gltf-binary`. PyTorch remains
OBJ-only.

### Verified

- `python3 server/sam3d/app.py --no-pytorch` starts; `/health` returns
  `{ours: true, pytorch: false}`.
- POST `/v1/infer` with `dancing.jpg` and no bbox produced a valid
  ours mesh (V=18439, F=36874) and propagated `bbox_used =
  [727.5, 111.0, 1534.1, 1456.0]` from the auto-bbox.

## References

- Live PORT.md: `cpu/sam3d_body/PORT.md` (drift table + per-step notes)
- Memory: `~/.claude/projects/.../memory/project_sam3d_body.md`
- Plan: `~/.claude/plans/parallel-task-to-implement-peaceful-wave.md`
- Upstream: https://github.com/facebookresearch/sam-3d-body
- Models: `facebook/sam-3d-body-dinov3` (840M H+) and
  `facebook/sam-3d-body-vith` (631M, deferred)
