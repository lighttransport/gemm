# SAM 3D Body — CPU port journal

FAIR's `sam-3d-body`: single-image full-body 3D human mesh recovery.
Encoder-decoder (DINOv3 / ViT-H backbone → promptable decoder →
MHR-head regressor) followed by Momentum Human Rig (MHR) skinning
that turns regressed parameters into a 3D mesh and 3D/2D keypoints.
This port lands alongside the in-progress `sam-3d-objects` port
(`cpu/sam3d/`, currently at Step 4 — SS Flow DiT); the two share
shared infra (`common/dinov3.h`, `safetensors.h`, NPY readers,
`cpu_compute.h`) but otherwise live in disjoint trees.

## v1 scope

CLI-only: `test_sam3d_body <safetensors-dir> <image.jpg> [--bbox
x0 y0 x1 y1] [--mhr-assets DIR] [-o body.obj]`. Per-stage
`verify_*.c` binaries diff C-runner output against
`/tmp/sam3d_body_ref/*.npy` dumps from `ref/sam3d-body/gen_image_ref.py`.

**User-confirmed (2026-04-25):** DINOv3 backbone first (reuses
existing `common/dinov3.h`); ViT-H variant deferred. Skip
detectron2 — user supplies cropped RGB (or full image + bbox).
CPU first, CUDA follows each stage. Dedicated
`ref/sam3d-body/.venv`; checkpoints at
`/mnt/disk01/models/sam3d-body/{dinov3,vith}/`.

## Concrete architecture (from upstream `sam_3d_body/` package)

**Top-level orchestrator** (`sam_3d_body/models/meta_arch/sam3d_body.py`):
- Loads a `Dinov3Backbone` or `vit.py` encoder, a promptable decoder,
  an `MHRHead`, and a `camera_head`.
- `load_sam_3d_body_hf(hf_repo_id)` is the convenience entrypoint.

**Backbone: DINOv3-H+** (`sam_3d_body/models/backbones/dinov3.py`):
- Thin wrapper around `torch.hub.load("facebookresearch/dinov3", name,
  source="github", pretrained=False)`. Confirmed `dinov3_vith16plus`
  (840.76M params, bf16) from `model_config.yaml` + the step-1 slice.
- Exposed fields used by the model: `patch_size`, `embed_dim`,
  `n_blocks`, `get_intermediate_layers(x, n=1, reshape=True, norm=True)[-1]`
  → [B, K, D] token grid.

**Confirmed dims (step 1 safetensors probe):**
- 32 encoder blocks; `embed_dim=1280`; `patch_size=16`; image 512×512
  → 32×32 = 1024 patch tokens; `cls_token=1`, `storage_tokens=4` →
  1029-token sequence.
- RoPE 2D, with learned `rope_embed.periods` of shape `(16,)` (stored
  in the checkpoint, not computed on the fly). `rope_dim4=16` × 4 axes
  = 64 RoPE lanes per head → `head_dim=64`, so `n_heads=20`.
- SwiGLU MLP: `mlp.w1` + `mlp.w2` parallel [5120, 1280], `mlp.w3`
  [1280, 5120] — all with non-zero biases. Hidden 5120 = 4× embed.
  Forward: `w3(silu(w1(x) + b1) * (w2(x) + b2)) + b3`.
- LayerScale via `ls1.gamma` / `ls2.gamma`.
- `attn.qkv.bias` and `attn.qkv.bias_mask` are **all-zero** across
  every sampled layer → safe to ignore during the port.

**Gaps vs existing `common/dinov3.h`:**
1. Current MLP path is GELU `fc1/fc2`; SwiGLU `w1/w2/w3` path must
   land before step 2.
2. Current RoPE computes periods from `rope_freq_base`; H+ uses a
   saved `rope_embed.periods` tensor. Prefer saved when present.
3. `storage_tokens` auto-detect already exists; reuse.

**Promptable decoder** (`sam_3d_body/models/decoders/`):
- `promptable_decoder.py` — transformer decoder that consumes encoder
  tokens + optional 2D-keypoint / mask prompts.
- `prompt_encoder.py` — encodes 2D keypoints + binary mask into
  prompt tokens (SAM-style).
- `keypoint_prompt_sampler.py` — generates / mocks keypoint prompts
  during training; inference is pass-through.

**Decoder dims (step 1 safetensors probe):**
- `DECODER.DIM = 1024` (token channel); 6 decoder layers; n_heads=8,
  head_dim=64 (q/k/v proj = 512 = 8×64 with num_kv_heads=8 also).
- Cross-attention: q from tokens (1024→512), k/v from image tokens
  (1280→512). Projection back to 1024.
- Self-attention: q/k/v all 1024→512.
- FFN (TransformerDecoderLayer mlp): `ffn.layers.0.0`: Linear(1024,1024);
  `ffn.layers.1`: Linear(1024,1024). So **hidden = 1024 (not 4×)** —
  small FFN.
- Per-layer norms: `ln1`, `ln2_1`, `ln2_2`, `ln3`, `ln_pe_1`, `ln_pe_2`
  (6 LN, widths {1024, 1024, 1280, 1024, 1024, 1280}).
- Input projections (pose token construction):
  - `init_to_token_mhr`: Linear(525, 1024). 525 = 519 (npose) + 3 (ncam)
    + 3 (cond_dim / CLIFF).
  - `prev_to_token_mhr`: Linear(522, 1024). 522 = 519 + 3 (no cond).
  - `prompt_to_token`: Linear(1280, 1024).
- Token embeddings (learnable, decoder-DIM):
  - `keypoint_embedding.weight`: (70, 1024). 70 body keypoints.
  - `keypoint3d_embedding.weight`: (70, 1024). 70 3D keypoints.
  - `hand_box_embedding.weight`: (2, 1024). L/R hands.
- Keypoint-token update path:
  - `keypoint_feat_linear`: Linear(1280, 1024) — image feat → decoder dim.
  - `keypoint_posemb_linear`: MLP(2, 1024, 1024, 1024) — 2D xy → pos embed.
  - `keypoint3d_posemb_linear`: MLP(?, 1024, 1024, 1024) — 3D xyz → pos embed.
- `ray_cond_emb`: Conv2d(1379, 1280, 1×1) + LN(1280). ray_cond(2) +
  flattened rays (1377) → added to image_embedding.
- `hand_pe_layer.positional_encoding_gaussian_matrix`: (2, 640) — SAM-style
  random-projection PE for 2D keypoints (dim 2 → dim 1280).

**MHR head dims (step 1 safetensors probe):**
- `head_pose.proj`: FFN(1024 → 1024 → 519). 2-layer MLP with zero bias
  on final layer (per mhr_head.py init).
- `head_camera.proj`: FFN(1024 → 1024 → 3). ncam=3 (cam_t_xyz only;
  focal comes from FovHead upstream, stored in `cam_params.npy[3]`).
- `init_pose.weight`: (1, 519) learnable init MHR param token.
- `init_camera.weight`: (1, 3) learnable init cam_t token (zero-init).
- `bbox_embed`: MLP(1024 → 1024 → 1024 → 4) — hand bbox regressor.
- **MHR skinning buffers** embedded inside `head_pose`:
  `faces (36874, 3) i64`, `joint_rotation (127, 3, 3)`,
  `scale_mean (68)`, `scale_comps (28, 68)`,
  `hand_pose_mean (54)`, `hand_pose_comps (54, 54)`,
  `keypoint_mapping (308, 18566)`, `nonhand_param_idxs (145) i64`,
  `hand_joint_idxs_{left,right} (27) i64`,
  `local_to_world_wrist (3,3)`, `right_wrist_coords (3)`, `root_coords (3)`.
  → Step 6 MHR skinning can read these directly from
  `sam3d_body_mhr_head.safetensors`; no separate `mhr_model.pt` convert
  is needed for basic verts/keypoints (only the upstream `mhr` pip
  package's parametric shape/LBS weights remain external for the
  actual skinning math, which comes from `assets/mhr_model.pt`).

**MHR head** (`sam_3d_body/models/heads/mhr_head.py`):
- `MHRHead.proj` is an `FFN(input_dim, input_dim//8, output_dims=npose)`
  with a zero-bias final layer.
- `npose = 6 + 260 + 45 + 28 + 54*2 + 72 = 519` scalars per person:
  6 global_rot, 260 body-cont params, 45 shape, 28 scale, 54×2 hands,
  72 face.
- Decomposition helpers (`compact_cont_to_model_params_body/hand`)
  in `sam_3d_body/models/modules/mhr_utils.py` and the `mhr` pip
  package's `MHR` class.

**Camera head** (`sam_3d_body/models/heads/camera_head.py`):
- Regresses `cam_t` (3-vector) + `focal_length` (1-vector) from the
  same encoder feature.

**MHR skinning** (`mhr` pip package, with `sam_3d_body/models/modules/mhr_utils.py`
adapters):
- Convert regressor-output "compact continuous" body params + 6D
  rotation representation → model-space params via roma lib.
- Apply MHR parametric model: shape PCA → template + blend-shapes →
  joint locations via J_regressor → linear blend skinning → vertices.
- Assets: `mhr_model.pt` ships inside HF ckpt at
  `assets/mhr_model.pt`. `ref/sam3d-body/convert_mhr_assets.py` will
  unpack it into safetensors + JSON kintree.

## Module map

| # | Stage                            | PyTorch source                                                       | C header (CPU)                                 | CUDA                                     |
|---|----------------------------------|----------------------------------------------------------------------|------------------------------------------------|------------------------------------------|
| 1 | Preprocess (crop + normalize)    | `sam_3d_body/data/transforms/*`                                      | reuse `common/image_utils.h`                   | host-only                                |
| 2 | DINOv3 encoder                   | `sam_3d_body/models/backbones/dinov3.py` + dinov3 hub module         | reuse `common/dinov3.h` (config-widen for H+)  | reuse sam3.1 ViT kernels (retargeted)   |
| 3 | Promptable decoder + MHR head    | `sam_3d_body/models/decoders/*` + `heads/mhr_head.py` + `camera_head`| new `common/sam3d_body_decoder.h`              | new kernels in `cuda_sam3d_body_kernels.h` |
| 4 | MHR skinning (params → verts)    | `mhr` pip pkg + `sam_3d_body/models/modules/mhr_utils.py`            | new `common/sam3d_body_mhr.h`                  | simple vector kernels                    |
| 5 | Post: cam unproj, 2D proj        | `SAM3DBodyEstimator.process_one_image` in `sam_3d_body_estimator.py` | inline in `sam3d_body_runner.c`                | host-only                                |
| 6 | Output writer (.obj + JSON)      | trivial                                                              | inline in `test_sam3d_body.c`                  | n/a                                      |

## Drift table

Populated as each `verify_*.c` lands green. Threshold is max_abs vs
pytorch fp32 dump; bf16 inference path floor ≈ 1e-2 for small nets.
32-layer DINOv3-H+ with big matmul weights cast bf16→f16 floors at
max_abs ≈ 1.5e-1 / mean_abs ≈ 4e-3 vs the fp32 reference — f16 noise
accumulates over 32 residual blocks. Per-stage threshold budgets
account for this.

| Stage             | Status     | max_abs | mean_abs | Notes |
|-------------------|------------|---------|----------|-------|
| Step 0 bootstrap  | GREEN      | —       | —        | legacy bootstrap complete; obsolete compatibility stubs now fail explicitly |
| Step 1 slicer     | GREEN      | —       | —        | convert_ckpt.py: 160 f16 big matmul weights + 391 f32 (biases/norms/embeds) dinov3 / 524 decoder fp32 (93.51M) / 52 mhr_head fp32+i64 (19.04M); 0 unmatched. Mixed precision: big 2D weights→f16 (threaded `cpu_gemm_f16`), small tensors→fp32 (free precision) |
| DINOv3 encoder    | GREEN      | 1.52e-1 | 4.01e-3  | verify_dinov3 via `dinov3_encode_from_normalized` bypass (skips u8 round-trip). Budget 2e-1 / 1e-2 = realistic f16 floor for 32-block ViT-H+; mean error is 0.02% of typical token magnitude (std≈0.32). 24.2s end-to-end @ 8 threads. |
| DINOv3 encoder (CUDA) | GREEN  | 9.95e-1 | 1.11e-2  | `cuda/sam3d_body/verify_dinov3` (debug_set_normalized_input → run_encoder → diff (1,1280,32,32) ref). 32-block ViT forward in NVRTC: patch_embed_sam3d + prepend_special_tokens + pos_embed_add + per-block LN/QKV-gemm/rotate-half-RoPE-on-patches/KV-transpose/flash-attn/proj/layerscale-add(LS1)/LN/w1+w2/silu_mul/w3/layerscale-add(LS2) + final layernorm. F16 weights via `sb_upload_f16`. CUDA matches CPU bit-for-bit to 5+ digits (same offending position d=875 py=29 px=12). The CUDA-vs-PyTorch floor (max≈1.0) tracks the CPU-vs-PyTorch floor; CUDA-vs-CPU diff ≈1e-5 well under the Step 3c 1e-2 gate. Verify gate set to 1.5 / 1.5e-2 to track this floor. |
| ray_cond_emb (CUDA)| GREEN     | 9.51e-4 | 1.46e-4  | `cuda/sam3d_body/verify_ray_cond` via `cuda_sam3d_body_debug_run_ray_cond` (image_emb + rays_hwc → out_chw). 3 NVRTC kernels in `cuda_sam3d_body_kernels.h`: `ray_cond_fourier_chw_f32` (3→99 rows: raw xyz + sin/cos for 16 bands = linspace(1,32,16)) → `conv1x1_chw_f32` (Y(1280,N)=W(1280,1379)·X(1379,N), F32 weights from `ray_cond_emb.conv.weight`, no bias) → `layernorm_chw_f32` (per-spatial LN over 1280 channels with `ray_cond_emb.norm.{weight,bias}`, eps=1e-6). CUDA tracks the CPU port's f32 floor (CPU max=9.49e-4). Output stays in CHW = (1280, H*W) layout for downstream cross-attn K/V (transpose at consumption). |
| build_tokens (CUDA)| GREEN     | 3.81e-6 | 5.21e-9  | `cuda/sam3d_body/verify_build_tokens` via `cuda_sam3d_body_debug_run_build_tokens` (init/prev/prompt → x, x_pe). One new NVRTC kernel `linear_f32_bias` (per-output thread, F32 weights). 3 matvec launches (525→1024, 522→1024, 1280→1024) for slots 0/1/2; 3 device-to-device memcpy for hand_box (2,1024) / kp_emb (70,1024) / kp3d_emb (70,1024) into slots 3-144; x_pe = zero except slots 1/2 = copies of x[1]/x[2]. F32 throughout → diff at f32 machine epsilon. |
| decoder layer (CUDA)| GREEN    | 1.14e-5 | 6.23e-7  | `cuda/sam3d_body/verify_decoder_layer` via `cuda_sam3d_body_debug_run_decoder_layer` for all 6 layers. New NVRTC kernels: `gemm_f32_bias` (Y=X·Wᵀ+b, 16×16 tiles), `add_two_f32`, `add_inplace_f32`, `gelu_inplace_f32` (exact via erf), `sdpa_f32` (block-per-(n_q,h), 256 threads, dynamic shmem = (256+N_k)·sizeof(float) for online softmax + scores). Layer orchestration: pre-LN x_pe → SA (LN1, q/k/v gemm 1024→512, sdpa H=8 Dh=64, proj 512→1024, residual) → CA (LN2_1+pe, LN2_2 ctx, q from x_ln+pe, k from ctx_ln+cpe, v from ctx_ln, sdpa, proj, residual) → FFN (LN3, ffn0 1024→1024, GELU, ffn1 1024→1024, residual). Layer 0 skip_first_pe drops the +pe in SA q/k. F32 weights throughout. All 6 layers track CPU port to f32 floor; max_abs across layers 1.9e-6 to 1.14e-5 — well below the 2e-3 / 3e-4 budget. |
| norm_final + heads (CUDA)| GREEN | 3.81e-6 | 2.92e-7 | `cuda/sam3d_body/verify_mhr_head` via `cuda_sam3d_body_debug_run_norm_and_heads`. Reuses existing `ln`, `gemm_f32_bias`, `relu_inplace_f32` kernels — no new kernels needed. norm_final LN(N_q=145, D=1024) → tokens_norm; pose head Linear(1024→1024)+ReLU+Linear(1024→519) on tokens_norm[0]; cam head Linear(1024→1024)+ReLU+Linear(1024→3) on tokens_norm[0]. ReLU activation matches CPU port (FFN default `act_layer=nn.ReLU` per upstream `sam_3d_body/models/modules/transformer.py`). All 3 outputs at f32 floor. |
| decoder forward_full (CUDA)| GREEN | 6.10e-4 | 4.88e-4 | `cuda/sam3d_body/verify_decoder` mirrors `cpu/sam3d_body/verify_decoder_full.c` — 6 GPU decoder_layers + GPU norm_final+heads + CPU MHR-in-the-loop (decode_pose_raw → mhr_forward → cm→m → keypoints_from_mesh → camera_project) → GPU kp_token_update. CPU decoder/mhr impl pulled in via the same `SAM3D_BODY_*_IMPLEMENTATION` macros; safetensors.h header-guarded include placed before the decoder header so the IMPL define inside the decoder is a no-op (runner.c remains the unique provider of safetensors symbols). All 4×5 per-layer (kp3d, kp2d_crop, kp2d_depth, cam_t for layers 0..4) and 6 final (head_pose_raw, head_camera_raw, kp3d, kp2d_crop, kp2d_depth, cam_t) GREEN at the same 5e-3 budget the CPU verify uses. Final pose_raw max=3.3e-6 (f32 floor), pred_cam_t max=6.1e-4 (CPU MHR fp32 round-off baseline; matches CPU forward_full numbers). |
| kp_token_update (CUDA)| GREEN | 4.05e-6 | 6.21e-8  | `cuda/sam3d_body/verify_kp_update` via `cuda_sam3d_body_debug_run_kp_token_update` for layers 0..4 (layer 5 short-circuits per upstream guard). New NVRTC kernels: `relu_inplace_f32`, `grid_sample_chw_f32` (bilinear, align_corners=False, zeros padding, optional invalid mask zeros output), `kp_pelvis_norm_f32`, `augment_overwrite_with_mask_f32`. Host computes invalid mask (xy outside [0,1] OR depth<1e-5) once, copies kp2d×2 sample points, image_emb (1280,32,32). 2D path: gemm 2→1024 + relu + gemm 1024→1024 → augment_overwrite_with_mask onto rows [5..75); grid_sample @ K=70 points + gemm 1280→1024 → tokens[5..75) += proj. 3D path: pelvis_norm (subtract avg of joints[9],[10]) → gemm 3→1024 + relu + gemm 1024→1024 → augment_overwrite (NULL mask) onto rows [75..145). **Critical**: kernel param structs that interleave 8-byte ptrs and 4-byte ints CANNOT use `__attribute__((packed))` directly — NVRTC's natural-alignment ABI inserts a 4-byte pad after a leading int, but `packed` strips it, causing param-buffer misalignment that silently writes zeros to half the augment rows. Initial layout `void*, int, void*, void*, int, int` was reordered to `void*, void*, void*, int, int, int` to fix. Tokens diff=0 (all 70 kp invalid in this dump → kp_feat = bias-only add identical between CPU/CUDA). |
| Decoder + MHR head| GREEN      | 1.9e-6  | 1.6e-7   | End-to-end preset path (145-token body branch): ray_cond_emb → build_tokens → 6×(decoder_layer_forward + kp_token_update) → norm_final → apply_heads_raw (+ init_pose/init_camera) matches head_pose_proj_raw / head_camera_proj_raw at f32 floor. Sub-stages broken out below. Public `sam3d_body_decoder_forward` now wired (2026-04-26) — convenience entry that bundles ray_cond_emb + get_dense_pe + invalid_prompt_token + build_tokens then delegates to `forward_full`. Sig widened to take MHR + cam_batch + (H, W) + rays_hwc + condition_info; previous stub had no callers. Standalone test `verify_decoder_forward` (2026-04-27) drives the wrapper from pre-ray patch tokens + decoder_batch__* dumps and diffs head_pose_proj_raw / head_camera_proj_raw / final pred_keypoints_3d / pred_keypoints_2d_{cropped,depth} / pred_cam_t — all six gates GREEN at f32 floor (max_abs ≤ 5e-4). |
| ray_cond_emb      | GREEN      | 9.49e-4 | 1.40e-4  | sam3d_body_ray_cond_emb_forward (fourier pos-enc 3→99 + 1×1 f32 conv 1379→1280 + LN2d(1e-6 eps)). verify_ray_cond consumes the pre-downsampled 32×32 rays. Budget 2e-3 / 3e-4 = realistic f32 conv floor (1379-channel accumulation + LN). |
| token construction| GREEN      | 4.77e-7 | 8.54e-10 | sam3d_body_build_tokens: Linear(525→1024) init + Linear(522→1024) prev + Linear(1280→1024) prompt + concat(hand_box(2), kp_emb(70), kp3d_emb(70)) → x(145,1024); x_pe zeros except slots 1/2 (prev/prompt). All f32 weights → diff at f32 machine epsilon. |
| decoder layer fwd | GREEN      | 3.8e-6  | 3.1e-7   | sam3d_body_decoder_layer_forward: self-attn(H=8, Dh=64) + cross-attn(q 1024→512 / k,v 1280→512) + FFN(1024→1024, GELU). 6 LNs. Layer 0 skip_first_pe=True (no PE in self-attn q/k). verify_decoder_layer feeds ref-dump inputs at each layer and diffs out x (isolates per-layer math from the inter-layer kp-update path coming in 4f). All 6 layers at f32 machine epsilon. |
| norm_final + heads| GREEN      | 1.9e-6  | 1.4e-7   | sam3d_body_norm_final + sam3d_body_apply_heads_raw (head_pose.proj FFN 1024→1024→519, head_camera.proj FFN 1024→1024→3). **Activation is ReLU, not GELU** — FFN in sam_3d_body/models/modules/transformer.py defaults `act_layer=nn.ReLU`, and both MHR heads instantiate with defaults. Initial attempt used GELU → diff 3.5 on pose raw; ReLU drops it to f32 floor. verify_mhr_head diffs norm_final + both proj outputs against ref captures. Hook also added for head_{pose,camera}.proj pre-inputs to confirm feed = tokens_norm[0]. |
| MHR skinning      | GREEN      | 8.4e-5  | 1.0e-5   | sam3d_body_mhr.h ports the pymomentum jit pipeline as pure C: parameter_transform (889×249 matvec, fp64 accum, padded shape cols zeroed), joint_params→local_skel (XYZ-Euler→quat, w-last, log-scale), local→global skel-state via fp64 prefix walker in 4 stages by [65,56,62,83] (pmi (2,266)), blend_shape + face_expressions einsum, pose_correctives (batch6DFromXYZ → subtract identity on cols 0/4 → COO sparse 53136 nnz → ReLU → dense 55317×3000 GEMV), and LBS skin_points (joint_state = global ⊗ inverse_bind_pose, scatter-add over 51337 skin entries). **batch6DFromXYZ output is column-stacked** [R00,R10,R20,R01,R11,R21] — initial row-stacked attempt would have failed stage 10C. verify_mhr_stages.c covers stages 6/7+8/10A/10B/10C/11/e2e; e2e mhr_forward(modelp, shape, face) max_abs=8.4e-5 vs cm-scale verts. Stage drift well below 1e-3 budget. |
| MHR head decode   | GREEN      | 9.5e-7  | 7.2e-9   | sam3d_body_decode_pose_raw (mhr_head.py stages 1-5, 519→204 mhr_model_params + 45 shape + 72 face): rot6d→rotmat→ZYX Euler, body_cont→133 (compact_cont_to_model_params_body, 23×6D→XYZ Euler + 58×sincos→atan2 + 6 trans), hand-mask + jaw-zero, scale PCA (mean+pca@comps), hand PCA per side (`mean + pca@comps` then compact_cont_to_hand uses interleaved DoF seq [3,1,1,3,1,1,3,1,1,3,1,1,2,3,1,1]), full_pose=cat(trans*10, grot, body[:130]) with hand drop-in at hand_joint_idxs_{left,right}, scales appended. **`enable_hand_model` flag now selects branch**: body branch (head_pose, flag=0) keeps global_rot=ZYX-Euler-as-is and global_trans=0, skips nonhand_param_idxs zeroing, and keeps all 70 keypoints. Hand branch (head_pose_hand, flag=1) re-projects (global_rot, global_trans=0) through `local_to_world_wrist`, computes `global_trans = -(R_xyz(grot_new) @ (right_wrist_coords - root_coords) + root_coords)`, applies nonhand_param_idxs zeroing (145 of 204), and zeros kp[:21] / kp[42:]. (Convention switch in hand branch: ZYX-Euler values fed back to `euler_to_rotmat("xyz", ...)` — roma "xyz" = extrinsic xyz = Rz(c)Ry(b)Rx(a).) verify_mhr_head_decode tests the hand branch (`mhr_params__*` ref captures the LAST MHRHead forward_hook firing, which is head_pose_hand) — all 5 outputs GREEN at f32 floor. |
| kp_token_update   | GREEN      | 1.4e-6  | 3.2e-8   | sam3d_body_kp_token_update: between-layer 2D+3D keypoint-token refresh. 2D path (kps_emb_start_idx=5): invalid_mask (xy out of [0,1] OR depth<1e-5) → kp_posemb_linear(2→1024→ReLU→1024) zeroed on invalid → overwrite augment[5..75); grid_sample(align_corners=False, bilinear, zeros padding) on image_emb at sample_points=kp2d*2 (×2 scale after subtract-0.5) → kp_feat_linear(1280→1024) → ADD into tokens[5..75). 3D path (kps3d_emb_start_idx=75): pelvis-norm subtract avg of joints [9,10] → kp3d_posemb_linear(3→1024→ReLU→1024) → overwrite augment[75..145). Last layer (n_layers-1=5) short-circuits per upstream guard. Tokens diff=0 because all 70 kp are invalid every layer (depth≈-700, xy out of bounds in ref dump) → kp_feat gets zeroed features → identical bias-only add. verify_kp_update tests layers 0..4 at f32 floor. |
| decoder e2e       | GREEN      | 1.9e-6  | 1.6e-7   | sam3d_body_decoder_forward_preset: flatten image_emb/PE CHW→HW×C once, ping-pong tokens through 6 layer_forward calls with kp_token_update between each, norm_final → apply_heads_raw → add init_pose/init_camera. verify_decoder_e2e diffs pre-add pose_raw (= mhr_params − init_pose, matching head_pose_proj_raw.npy which is the forward-hook output of head_pose.proj inner FFN) and cam_raw. Ref captures LAST firing of head_pose.proj (layer 5 post-loop `out = norm_final(tokens)` call in PromptableDecoder.forward). `gen_image_ref.py` counter-gates the dumps to the first forward_decoder call (145 tokens); full-inference second call (148 tokens via run_keypoint_prompt) stays out of scope for v1. |
| camera_project    | GREEN      | 1.3e-4  | 6.96e-5  | sam3d_body_camera_project: PerspectiveHead.perspective_projection + BaseModel._full_to_crop. Channel flip pred_cam[0,2]*=-1, bs=bbox_scale*s+1e-8, tz=2*fx/bs, cx/cy from `ori_img_size/2 − bbox_center` scaled by 2/bs (use_intrin_center=False branch). Per-keypoint: p = j3d + cam_t → perspective via cam_int row 0/1 → affine_trans for crop, normalize by `img_size` (model input = 512). **Critical**: upstream `perspective_projection`'s `img_size` arg is `batch["ori_img_size"]` (1693×952), not the model input `img_size` (512×512) which is only used for the crop normalize step. verify_camera_project covers all 6 layers × 4 outputs (cam_t, kp2d, kp2d_cropped, kp2d_depth) at f32 floor. |
| decoder forward_full | GREEN   | 5.5e-4  | 4.3e-4   | sam3d_body_decoder_forward_full: production iterative path with MHR-in-the-loop. For layers 0..4 calls token_to_pose_output_fn body (norm_final → apply_heads_raw → +init_pose/init_camera → decode_pose_raw(enable_hand_model=0) → mhr_forward → cm→m → keypoints_from_mesh(enable_hand_model=0) → camera_project), feeds kp2d_crop / kp2d_depth / kp3d into kp_token_update; layer 5 = post-loop final norm + heads + decode + mhr + kpts. verify_decoder_full diffs all 5 intermediate body layers' (kp3d, kp2d_crop, kp2d_depth, cam_t) and the final layer's (pred_keypoints_3d, pred_keypoints_2d_cropped, pred_keypoints_2d_depth, pred_cam_t, head_pose_proj_raw, head_camera_proj_raw) against decoder_pose_layer{0..5}__* dumps — all 27 stages GREEN at f32 floor. Final layer captured under layer_idx=5 (the last value of `for layer_idx, layer in enumerate(self.layers)` propagates to the post-loop call). |
| runner forward_full (ref) | GREEN | 9.5e-7 | 1.2e-7 | sam3d_body_runner.c run_decoder wired to sam3d_body_decoder_forward_full + mhr_forward; sam3d_body_debug_override_decoder_inputs feeds image_embeddings_after_ray + decoder_layer0_in__{x,x_pe,context_pe} (token→CHW permute) + decoder_batch__* into the public C API. run_encoder short-circuits when dec_in_set; run_mhr is folded into run_decoder. ctx->faces populated from decoder_model->faces (head_pose.faces (36874,3) i64 → i32). test_sam3d_body --refdir mode emits 18439-vert / 36874-tri OBJ via common/obj_writer.h. |
| warp_matrix       | GREEN      | 1.3e-5  | —        | sam3d_body_compute_bbox_affine (GetBBoxCenterScale padding=1.25 + 2× fix_aspect_ratio(0.75, out_w/out_h) + get_warp_matrix(center, scale, 0°, out_size)). verify_warp_matrix diffs center/scale/warp_mat (2,3) against `ref/sam3d-body/gen_preprocess_ref.py` dumps at f32 floor. |
| preprocess_image  | GREEN      | 5.25e-2 | 5e-4     | sam3d_body_preprocess_image (cv2.warpAffine INTER_LINEAR + HWC→CHW + ImageNet norm). Budget 1e-1: cv2.warpAffine uses fixed-point INTER_TAB_SIZE=32 subpixel quantization which can differ by ≤~3 u8 units on rare pixels (≈0.06 after /255/min_std). Mean diff well below the bf16 encoder noise floor (1.5e-1 max_abs upstream). |
| End-to-end verts  | GREEN      | 9.5e-7  | 1.2e-7   | verify_end_to_end uses the public runner via debug_override; vertices (V=18439) max_abs=9.5e-7 vs out_vertices.npy, keypoints_3d max_abs=2.7e-7 vs out_keypoints_3d.npy, keypoints_2d max_abs=1.4e-4 px vs out_keypoints_2d.npy. f32 floor across the board. |
| runner self-driven | GREEN     | —       | —        | **Step 8c-ii complete.** sam3d_body_run_encoder computes TopdownAffine (compute_bbox_affine) + default cam_int (default_cam_int with optional focal_hint override) + preprocess_image → dinov3_encode_from_normalized. sam3d_body_run_decoder auto-populates dec_image_emb_chw (ray_cond_emb post the HW→CHW permute dropping CLS+4 register tokens), dec_image_pe_chw (get_dense_pe), dec_init_x/dec_init_xpe (build_tokens from condition_info + init_pose + init_camera + invalid_prompt_token), and dec_cam_batch (cam_int / bbox_center / bbox_scale / ori_img_size / img_size / affine_trans / use_intrin_center=0 / default_scale_factor=1.0) when no debug-override is set. test_sam3d_body --image person.jpg --bbox 100 100 600 900 -o out.obj produces V=18439 / F=36874 end-to-end without any reference dumps. Bit-exact verification vs a Python dump deferred (existing /tmp/sam3d_body_ref uses irreproducible bbox values; regen with `python ref/sam3d-body/gen_image_ref.py --image X --bbox x0 y0 x1 y1` to compare). |
| End-to-end (CUDA) | GREEN      | 1.07e-6 | 2.08e-7  | **Step 9 complete.** `cuda/sam3d_body/test_cuda_sam3d_body <sft> <img> --mhr-assets <dir> -o out.obj` produces V=18439 / F=36874 OBJ end-to-end via `cuda_sam3d_body_run_encoder` (host preprocess → on-device DINOv3) → `cuda_sam3d_body_run_decoder` (GPU ray_cond_emb + build_tokens + 6×decoder_layer + GPU norm_final+heads, with CPU MHR-in-the-loop). Encoder caches `self_{center,scale,warp,cam_int}` so decoder uses identical TopdownAffine values. CUDA vertices match the CPU port to 1e-6 max_abs (float noise) on the same input image. **Two bugs fixed in this step**: (1) on-device naive bilinear resize replaced with host-side `sam3d_body_preprocess_image` (cv2.warpAffine + ImageNet norm using the bbox-affine warp matrix); (2) post-MHR Y/Z negation (`out->pred_vertices[i*3+{1,2}] = -verts_m[i*3+{1,2}]` from `decoder_forward_full`'s "post-flip cam frame in m") was missing — CUDA reimplementation did a plain memcpy. Symptom: vertices were exactly `R_x(π)` rotated (X identical, Y/Z negated) vs CPU. Both runners agree with the ref at 0.57 max_abs (full-image bbox vs detectron2-cropped bbox), so cross-port equality is the gate, not pytorch parity. |

## Steps

| Step | Deliverable                                   | Gate |
|------|-----------------------------------------------|------|
| 0    | Scaffolds build; ref venv green; `--skip-run` NPY dump lands | `make all` OK; 1 NPY in `/tmp/sam3d_body_ref` |
| 1    | `convert_ckpt.py` produces per-module safetensors | tensor-count manifest printed |
| 2    | CPU DINOv3 encoder (verify_dinov3 green)      | max_abs ≤ 5e-2 vs pytorch fp32 |
| 3    | CUDA DINOv3 encoder                            | **GREEN (2026-04-25)**: 3a NVRTC+safetensors upload (32 blocks, D=1280, FFN=5120) + 3b 32-block ViT forward (LN→QKV-proj→rotate-half RoPE on Q/K patch slots→KV transpose→flash-attn→proj→LS1+res→LN→SwiGLU(w1·silu * w2 → w3)→LS2+res, final LN) + 3c verify_dinov3 (CUDA) green vs CPU. CUDA bit-matches CPU to 5+ digits (max_abs 9.95e-1 vs 9.95e-1; mean 1.115e-2 vs 1.115e-2 at d=875,py=29,px=12) → CUDA-vs-CPU diff ≈1e-5, well under the Step 3c 1e-2 gate. CUDA-vs-PyTorch numbers track the CPU port floor. |
| 4    | CPU promptable decoder + MHR head (verify_decoder green) | shape_params + body_pose + cam diffs ≤ 1e-3 |
| 5    | CUDA decoder + MHR head                        | **GREEN (2026-04-25)** — all 5a–5g done. `cuda/sam3d_body/verify_decoder` drives full forward (6 GPU decoder layers + per-layer pose-output on CPU MHR + final norm+heads+MHR) and matches PyTorch ref within 5e-3 at every per-layer (kp3d, kp2d_crop, kp2d_depth, cam_t for layers 0..4) and final (pose_raw, cam_raw, kp3d, kp2d_crop, kp2d_depth, cam_t). Final pose_raw max=3.3e-6, head_camera_raw max=5.2e-6 (f32 floor); kp2d_depth/pred_cam_t max ≤ 6.1e-4 (CPU MHR round-off baseline). |
| 6    | CPU MHR skinning (verify_mhr green)            | vertices max_abs ≤ 1e-3 vs ref |
| 7    | CUDA MHR skinning                              | **CLOSED 2026-04-25 via CPU parallelization, not GPU port.** Profiling showed `pose_correctives` (55317×3000 dense matvec, 166M FMA) was 99.3% of MHR runtime at 181 ms / 183 ms total. Adding `#pragma omp parallel for` over the row loop (drop `(void)n_threads`) gave 5.9× at 16 threads → MHR drops 183 ms → 31 ms / call → end-to-end CUDA wall 3.47 s → 2.84 s (~18% saved, ~630 ms). CUDA runner now passes `omp_get_max_threads()` to MHR. Numerics match (verify_mhr_stages green: pose_correctives max_abs 5.96e-8; CPU vs CUDA OBJ vertices match to 7e-6). Full GPU port (~1500 NVRTC lines) deferred — cost/benefit no longer favors it. |
| 8    | End-to-end CPU → .obj (verify_end_to_end green)| **GREEN (ref-driven + self-driven)**: ref-driven (8c-i + 8d) vs `out_vertices.npy` max_abs=9.5e-7 / `out_keypoints_3d.npy` max_abs=2.7e-7 / `out_keypoints_2d.npy` max_abs=1.4e-4 px. Self-driven raw-image path (8c-ii) wires compute_bbox_affine + preprocess_image + ray_cond + get_dense_pe + build_tokens into the runner; `test_sam3d_body --image X.jpg --bbox …` emits a valid 18439-vert OBJ with no ref dumps. |
| 9    | End-to-end CUDA → .obj                         | **GREEN (2026-04-25)**: `test_cuda_sam3d_body` produces V=18439/F=36874 OBJ matching CPU port to 1.07e-6 max_abs (float noise). Two cross-port pitfalls fixed: (1) host `sam3d_body_preprocess_image` (cv2.warpAffine + ImageNet norm) — naive bilinear resize fails encoder; (2) final `pred_vertices` applies Y/Z negation ("post-flip cam frame in m") — bypassing `decoder_forward_full` requires replicating the flip. |
| 10   | Shared infra: promote `common/npy_io.h` + `common/qtensor_utils.h` | **GREEN (2026-04-25)** for `npy_io.h` — single canonical reader at `common/npy_io.h`; 18 cpu/sam3d_body + 8 cuda/sam3d_body verify_*.c sources + `test_sam3d_body.c` now `#include` it instead of inlining. `cpu/sam3d/verify_npy.h` reduced to a back-compat shim. `qtensor_utils.h` deferred (no second consumer yet). |

### Sequencing pivot (2026-04-25)

Original plan: CPU + CUDA alternating per stage. Pivoted after Step 2
to **ship the full CPU pipeline first, then port stages to CUDA**.
Rationale:

- Native CUDA DINOv3-H+ port is ~1500 lines of NVRTC kernels on top
  of `common/cuda_kernels_common.h` — a significant engineering
  investment on its own.
- End-to-end CPU `.obj` output unblocks meaningful validation (eye-
  ball the mesh / keypoints vs a viewer) earlier.
- Once CPU is fully green, the CUDA port becomes a pure
  numerics-matching exercise against a stable reference, which is
  easier to debug and easier to parallelize than porting against a
  moving CPU target.
- Same pattern as `cpu/sam3` → `cuda/sam3.1` (CPU landed entirely
  first; CUDA followed as an independent port).

CUDA runner and verifier coverage has since landed; see `PORT_v2.md`
for the current DINOv3/ViT-H status and raw-image fixed-bbox gates.

## Build

```bash
cd cpu/sam3d_body
make                  # zen2 flags; override with ARCH=native, COMPILER=clang
make DEBUG=1          # -O0 + asan/ubsan
```

The maintained stage verifiers build under `make all`. Historical
compatibility stubs no longer return success for missing coverage; use
the explicit `verify-*` aliases or `PORT_v2.md` for the current gate
matrix.

## Checkpoint layout (on disk)

HF ckpts (gated; access granted) live at
`$MODELS/sam3d-body/{dinov3,vith}/`.

Per-module safetensors slices (step 1 output) at
`$MODELS/sam3d-body/safetensors/`:

| file                                          | src ckpt                               | purpose                                    |
|-----------------------------------------------|----------------------------------------|--------------------------------------------|
| `sam3d_body_dinov3.safetensors`               | `dinov3/model.ckpt` (encoder branch)   | DINOv3-H+ backbone (ViT blocks + patch-embed + pos-embed + norms) |
| `sam3d_body_decoder.safetensors`              | `dinov3/model.ckpt` (decoder branch)   | promptable transformer decoder             |
| `sam3d_body_mhr_head.safetensors`             | `dinov3/model.ckpt` (head branch)      | MHR params regressor + camera head         |
| `sam3d_body_mhr_assets.safetensors` + kintree | `dinov3/assets/mhr_model.pt`           | LBS weights, blend shapes, J regressor, parent array |

Run `python cpu/sam3d_body/convert_ckpt.py $MODELS/sam3d-body/dinov3
-o $MODELS/sam3d-body/safetensors` to regenerate slices.

## Ref dump

See `ref/sam3d-body/README.md`. TL;DR:

```bash
source ref/sam3d-body/.venv/bin/activate
export MODELS=/mnt/disk01/models

python ref/sam3d-body/gen_image_ref.py \
    --image assets/person.jpg \
    --hf-repo-id facebook/sam-3d-body-dinov3 \
    --outdir /tmp/sam3d_body_ref --seed 42
```

`--skip-run` dumps only the preprocessed crop (useful while the
decoder/MHR code isn't wired in the C runner).

## Deferred cleanups (shared with sam3d port)

- **DONE 2026-04-25: `common/npy_io.h`.** Promoted. Single canonical
  reader (`npy_load`, `npy_max_abs_f32`) under `common/`. 18
  `cpu/sam3d_body/verify_*.c` + 8 `cuda/sam3d_body/verify_*.c` +
  `test_sam3d_body.c` include it directly. `cpu/sam3d/verify_npy.h`
  is now a thin back-compat shim that forwards to `npy_io.h`, so all
  pre-existing `cpu/sam3d` verify binaries keep working unchanged.
  `is_f32` out-param is optional (pass `NULL` if you don't need it).
- **PARTIAL: `common/qtensor_utils.h`.** 2026-04-25 — the qtensor
  struct + `qt_make_tensor()` loader are factored into
  `common/qtensor_utils.h`, consumed by sam3d_body_mhr/decoder and
  every other model header. Still copy-pasted: `_dequant`,
  `_dequant_row`, `_layernorm_batch`, `_layerscale` (more work, not
  strictly tied to qtensor layout — revisit separately).
- **Keys bicubic pos-embed interp.** DINOv3-H+ may need a non-square
  input size; if so, the interp in `common/dinov3.h` needs the same
  generalization as `dinov2_interp_pos_embed`.
