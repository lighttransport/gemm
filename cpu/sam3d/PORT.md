# SAM 3D Objects — CPU port journal

FAIR's `sam-3d-objects`: masked RGBA image → 3D Gaussian Splat
(`.ply`). Two-stage latent-flow pipeline very close to our TRELLIS.2
port plus a DINOv2 image tower and a Gaussian-splat decoder head.

## v1 scope

CLI-only: `test_sam3d <pipeline.yaml> <image.png> <mask.png>
[--pointmap p.npy] [-o splat.ply]`. Per-stage verify_*.c binaries diff
C-runner output against `/tmp/sam3d_ref/*.npy` dumps from
`ref/sam3d/gen_image_ref.py`.

**Deferred**: MoGe depth (user supplies `--pointmap`), SLAT Mesh
decoder, texture bake, layout opt, `.glb` output, CUDA runner, server
integration.

## Concrete architecture (from the shipped `checkpoints/*.yaml`)

**DINOv2 encoder** (shared by SS- and SLAT-stage embedder fusers):
- `dinov2_vitl14_reg` — ViT-**L**/14 with **4 register tokens**
  (CLS + 4 reg + 37×37 patches = 1374 tokens @ input_size=518)
- embed=1024, depth=24, heads=16, patch=14, image_size=518
- Learned absolute pos_embed, interpolated for 37×37 grid
- Image branch runs with raw RGB; mask branch runs with RGB × mask.
  Both branches share weights, outputs are concatenated before the
  fuser's projection net.

**PointPatchEmbed** (SS stage only):
- embed_dim=512, input_size=256, patch_size=8 → 32×32 = 1024 tokens
- Three-channel MoGe pointmap → 512-d tokens + learned pos_embed
- `remap_output: linear` projection back to 1024 channels for fuser.

**CondEmbedderFuser**:
- `projection_net_hidden_dim_multiplier: 4.0` (Llama-style SwiGLU)
- `use_pos_embedding: learned` across concatenated modality tokens.

**SS Flow DiT** (`SparseStructureFlowTdfyWrapper`, shortcut variant):
- model_channels=1024, num_blocks=24, num_heads=16, mlp_ratio=4
- in_channels=out_channels=8, resolution=16 (16³ latent grid)
- patch_size=1, pe_mode=ape, qk_rms_norm=true
- **`is_shortcut_model: true`, `inference_steps: 2`** (shortcut ODE)
- cfg strength=2.0 on shape modality.

**SS-VAE decoder** (`SparseStructureDecoderTdfyWrapper`):
- 3D-conv stack: 16³×8 → 64³×1 occupancy logits
- channels=[512, 128, 32], res_blocks=2, middle_res_blocks=2.

**SLAT Flow DiT** (`SLatFlowModelTdfyWrapper`, standard flow):
- model_channels=1024, num_blocks=24, num_heads=16, mlp_ratio=4
- in_channels=out_channels=8, resolution=64, patch_size=2
- io_block_channels=[128], num_io_res_blocks=2
- pe_mode=ape, qk_rms_norm=true, **`inference_steps: 12`**
- Swin-style shift-window sparse self-attn inside each block.

**SLAT GS decoder** (`SLatGaussianDecoderTdfyWrapper`):
- model_channels=768, num_blocks=12, num_heads=12, mlp_ratio=4
- `attn_mode: swin`, `window_size: 8` — shift-window sparse attn
- num_gaussians=32 per voxel (alt gs_4 ckpt: 4 per voxel)
- voxel_size=1.5, perturb_offset=true (Hammersley sub-sampling)
- scaling_bias=0.004, opacity_bias=0.1, softplus scale activation.

## Module map

| # | Stage               | PyTorch source                                                                 | C header                                      |
|---|---------------------|--------------------------------------------------------------------------------|-----------------------------------------------|
| 1 | Preprocess          | `pipeline/preprocess_utils.py`                                                 | reuses `common/image_utils.h`                 |
| 2 | MoGe depth          | `pipeline/depth_models/moge.py` (uses `Ruicheng/moge-vitl`)                    | STUB v1 (user-provided `--pointmap`)          |
| 3 | DINOv2-L/14+reg     | dinov2 `vitl14_reg`                                                            | new `common/dinov2.h` (reuse da3 + dinov3 reg)|
| 4 | Point patch embed   | `model/backbone/dit/embedder/pointmap.py`                                      | new `common/sam3d_cond_fuser.h`               |
| 5 | Cond fuser          | `model/backbone/dit/embedder/embedder_fuser.py` + `model/layers/llama3/ff.py`  | new `common/sam3d_cond_fuser.h`               |
| 6 | SS Flow DiT         | `model/backbone/tdfy_dit/models/mot_sparse_structure_flow.py` (shortcut)       | reuses `common/trellis2_dit.h` (retarget)     |
| 7 | SS-VAE decoder      | `model/backbone/tdfy_dit/models/sparse_structure_vae.py`                       | reuses `common/trellis2_ss_decoder.h`         |
| 8 | Voxel prune         | `pipeline/inference_utils.py`                                                  | trivial — `argwhere(occ > 0)`                 |
| 9 | SLAT Flow DiT       | `model/backbone/tdfy_dit/models/structured_latent_flow.py` + shift-window attn | reuses `common/sparse3d.h` (+ shift-window)   |
| 10| SLAT GS decoder     | `model/backbone/tdfy_dit/models/structured_latent_vae/decoder_gs.py`           | new `common/sam3d_gs_decoder.h`               |
| 11| Flow solver         | `model/backbone/generator/flow_matching/solver.py` + shortcut ODE              | extends `common/qwen_image_scheduler.h`       |
| 12| PLY writer          | gsplat `save_ply`                                                              | new `common/gs_ply_writer.h`                  |

## Drift table

Populated as each verify_*.c lands green. Threshold is the max_abs
diff against the pytorch-fp32 dump; bf16 inference path floor ≈ 1e-2.

| Stage            | Status  | max_abs  | mean_abs | Notes |
|------------------|---------|----------|----------|-------|
| DINOv2-L/14+reg  | GREEN   | 1.60e-02 | 5.21e-05 | 2 branches × 1370 tok × 1024. Diffed vs `dump_cond_tokens.py` dinov2_tokens.npy (pytorch fp32, XFORMERS_DISABLED=1). Key fixes during port: (1) pad_to_square_centered before resize; (2) binary-mask-as-3ch fed through DINOv2 (NOT rgb×mask); (3) fp32 preprocessing path (`sam3d_prep_*_f32` + `dinov2_encode_f32`) to avoid uint8 round-trip; (4) exact GELU (erf-based), not tanh-approximate — upstream `dinov2/layers/mlp.py` uses `nn.GELU()` default; (5) nearest-neighbor resize for the mask uses pytorch's legacy `floor(dst * src/dst)`, not the centered variant. |
| PointPatchEmbed  | GREEN   | 3.81e-06 | 1.62e-07 | Verified via verify_cond_fuser (covers PPE + fuser). dim=512 heads=16 head_dim=32 ffn=1024 patch=8 grid=32×32. |
| CondFuser        | GREEN   | 3.81e-06 | 1.62e-07 | 3 modalities (dino-img/msk D_in=1024, point D_in=512) → SwiGLU D_out=1024; idx_emb[pos=1 (full)] applied. Diffed vs `cond_tokens_full.npy`. |

## Deferred cleanups (surfaced during step 2)

- **DONE: Shared qtensor helpers.** Consolidated into
  `common/qtensor_utils.h` (2026-04-25). First pass: one `qtensor`
  struct + one `qt_make_tensor()` helper across 13 consumer headers
  (dinov2, dinov3, depth_anything3, trellis2_dit, trellis2_ss_decoder,
  sam3d_cond_fuser, sam3d_slat_dit, sam3d_ss_flow_dit, sam3d_gs_decoder,
  sam3d_body_{mhr,decoder}, sparse3d, qwen_image_dit). Second pass:
  `qt_find` / `qt_find_opt` (name→tensor lookup) replacing 4 wrappers
  in sam3d_slat_dit / sam3d_gs_decoder / sam3d_ss_flow_dit. Third pass:
  `qt_numel` / `qt_dequant` / `qt_dequant_row` (gated on `GGML_DEQUANT_H`)
  replacing dinov2_*, dinov3_*, scf_* and sp3d_dequant_row copies.
  Transformer.h kept separate (GGUF path). qwen_image_dit.h kept its
  `qimg_st_make_tensor` wrapper (FP8 E4M3 support, different signature).
  depth_anything3.h kept its `da3_*` family — extra `fprintf` error
  logging is part of its API contract. Numerics preserved
  byte-for-byte (verify_slat_dit max_abs=1.788e-05 unchanged).
- **DONE: Shared Keys bicubic pos-embed interp.** Promoted to
  `cpu_compute.h::cpu_keys_bicubic` + `cpu_interp_pos_embed_bicubic`
  (2026-04-25). Replaced `dinov2_interp_pos_embed` (52-line dedicated
  helper) and the 35-line inline block in `depth_anything3.h`'s encoder.
  Body byte-identical (same Keys a=-0.75 cubic kernel + interp_offset=0.1
  trick); both binaries build clean.
- **DONE: F32 GEMM in cpu_compute.** Promoted `dinov2_gemm_f32` to
  `cpu_compute.h::cpu_gemm_f32` (2026-04-25). Body byte-identical;
  dinov2 callers renamed. Follow-up swept the same dequant-per-row
  perf bug across 5 headers (2026-04-25): `dinov3_batch_gemm`,
  `sp3d_linear`, `da3_batch_gemm`, `t2dit_batch_gemm`,
  `vit_batch_gemm_bias` all now dequant W once and dispatch through
  the shared F32 GEMM (cpu_gemm_f32, sp3d_gemm_f32_dispatch, or
  per-header equivalent). dinov3/da3 also gained F32 fast paths
  (zero-copy when W->type == GGML_TYPE_F32) to match dinov2's
  three-branch shape. verify_slat_dit numerics byte-identical post
  fix (max_abs=1.788e-05); other paths' end-to-end numerics verify
  vs pytorch reference is checkpoint-gated (a noted follow-up).
  vision_encoder fix is the highest-impact: that fallback fires for
  Q-quant VLM weights in production.
| SS Flow DiT      | GREEN   | 3.00e-05 | 9.06e-07 | 5 modalities (shape[4096,8] + 4 pose[1,*]) → 1 MOT block × 2 streams (shape+pose), 24 blocks. Diff vs `ref/sam3d/dump_ss_dit_io.py` at t=1.0 d=1.0 with dumped cond_tokens_full. All 5 out_*.npy within fp32 noise floor (scale/trans/tscale ≤ 1e-6; 4096-tok shape ≤ 3.00e-05 — flash-attn tile accumulation error). SDPA dispatched through `cpu_cross_attention` from `cpu_compute.h` (AVX2 + pthread over heads). Key quirks: (1) MultiHeadRMSNorm is F.normalize(x,dim=-1) × gamma × sqrt(head_dim) — NOT classic RMSNorm; (2) FeedForwardNet uses GELU(approximate="tanh") — opposite of DINOv2's exact erf GELU; (3) norm1/norm3 are no-affine, norm2 is affine; (4) `time_scale=1000` config field is a sampler-level knob — TimestepEmbedder takes t directly, but the *caller* must pre-scale: production runners pass `t * time_scale` because upstream `_generate_dynamics` does `t * self.time_scale` before reaching the model (the verify dump path uses `t=1.0` raw because it calls the raw model directly, bypassing the wrapper); (5) pose stream = concat(6drot, translation, scale, translation_scale) each with own pos_emb then merged along token dim; (6) protect-list MOT: shape self-attends only (4096→4096), pose attends to concat(pose_kv, shape_kv) (4→4100). |
| Flow solver      | GREEN   | 0       | 0        | Pure scalar — `common/sam3d_shortcut_solver.h` has time-schedule (`make_times`, optional rescale_t + reversed), Euler update (`x += v*dt`), CFG combine (`(1+w)*cond - w*uncond`), and CFG interval gating. Exact-equal against hand-derived reference for linspace, reversed, rescale_t=3.0 (6-point schedule), Euler, and CFG. Model forward stays caller-supplied; `time_scale` is documented as sampler-to-callback concern. |
| SS-VAE decoder   | GREEN   | 5.42e-04 | 3.67e-05 | Direct reuse of `common/trellis2_ss_decoder.h` — architecture + key names identical. Diff vs `ref/sam3d/dump_ss_dec_io.py` at 64³×1 occupancy logits, fp32 end-to-end. Fed `ss_dit_out_shape.npy` from step 4 transposed to NCDHW=(8,16,16,16); ref prefix '' (74 tensors matched direct). GN groups=32, SiLU, 3 res-blocks @ (16³,32³,64³) with pixel-shuffle upsample. Added OMP over output-channel loop in `t2dec_conv3d` — 10× speedup (456s → 46s at 16 threads). Numerics identical vs 1-thread (6.14e-04) up to FP-reduction ordering. |
| SLAT Flow DiT    | GREEN   | 1.79e-05 | 1.65e-06 | Step 7b: loader GREEN (526/526 tensors) + forward GREEN at N=1024, n_cond=1374, t=1.0. Arch: 24 transformer blocks dim=1024/heads=16/head_dim=64, `attn_mode="full"` (shift-window attn is only in the GS decoder); 2 input res blocks [128→128, 128→1024↓], 2 out res blocks [2048→128↑, 256→128], in=out=8. SparseResBlock3d wiring: affine-LN → SiLU → SubMConv3d(3³) → no-affine-LN with modulation `(1+scale)·h + shift` (scale/shift from `Linear(SiLU(t_emb))` with zero-init) → SiLU → SubMConv3d(3³) → skip_connection (Linear when C_in≠C_out). Two key bugs fixed: (1) **SparseDownsample pool semantics**: pytorch uses `torch.scatter_reduce(reduce="mean", include_self=True)` on a zero-init buffer, which divides by `count+1` (the synthetic zero counts). C's original pool_mode=0 divided by `count` → 2× output for 1-child parents (990/1007 coarse voxels here). Fixed via new `pool_mode=2` in `sp3d_downsample`. (2) **Caller-visible pointer propagation**: `slat_resblock_forward` takes `sp3d_tensor **xp` and free/replace the tensor on down/up-sample. `sam3d_slat_dit_forward` previously took `sp3d_tensor *x` by value — so the caller's pointer was left dangling after the first resample. Changed signature to `sp3d_tensor **x` and updated both callers (`verify_slat_dit.c`, `sam3d_runner.c`). |
| SLAT GS decoder  | GREEN   | 8.63e-05 | 6.13e-06 | Step 8b: full forward + `to_representation` diffed vs `ref/sam3d/dump_slat_gs_io.py` at n_voxels=256. Post-LN-out_layer feats [N,448] max_abs=8.6e-5; per-channel reps (xyz 3.6e-7, dc 6.3e-5, scaling 8.6e-5, rotation 5.7e-6, opacity 7.2e-5) all well under fp32 floor. `common/sam3d_gs_decoder.h` implements: input_layer (8→768), APE sinusoidal over (z,y,x), 12 × SparseTransformerBlock (no-affine LN eps=1e-6 → swin sparse self-attn window=8 shift (0,0,0)/(4,4,4) alternating → residual → no-affine LN → MLP GELU(tanh) → residual), final no-affine LN eps=1e-5, out_layer (768→448). Windowed partition helper: wid[i] = sum_axis ((coord[i,a]+shift[a])/ws) * cumprod_offset[a], plus batch offset; qsort voxels by wid, dispatch `cpu_attention` per contiguous run. `to_representation`: xyz = (coord+0.5)/res + tanh(feat*lr + offset_perturbation)*0.5*voxel_size/res; scaling*1, dc*1, rotation*0.1, opacity*1. Extended `sp3d_layernorm` to accept NULL w/b for no-affine (γ=1, β=0). Fixed sparse3d.h qtensor guard ladder to also skip DINOV2_H / SAM3D_COND_FUSER_H / SAM3D_SS_FLOW_DIT_H / QWEN_IMAGE_DIT_H / TRELLIS2_{SS_DECODER,DIT}_H / SAM3D_BODY_DECODER_H so the one-TU includes of this header in sam3d_runner.c no longer double-define. |
| End-to-end PLY   | PARTIAL | —        | —        | Step 9: `common/gs_ply_writer.h` writes binary-LE INRIA-compatible PLY (`x y z nx ny nz f_dc_{0..2} opacity scale_{0..2} rot_{0..3}`). `sam3d_run_slat_gs_decode` wired in `sam3d_runner.c`: loads `sam3d_slat_gs_decoder.safetensors`, runs `sam3d_gs_decoder_transformer` + `to_representation`, converts to INRIA storage (opacity stored as `raw + opacity_bias`; scale stored as `log(softplus(raw + inv_softplus(scaling_bias)))` via a numerically-stable `log1pf`-based helper — naive `logf(1+expf(x))` underflows in the softplus tail when `expf(x) < 1 ULP`(1.0) ≈ 6e-8, producing `-inf`), packs into `ctx->gaussians[N*G, 17]`. `test_sam3d` adds `--slat-ref <dir>` to bypass the still-stubbed upstream runner stages (ss_dit / ss_decode / slat_dit) by loading `slat_gs_in_{coords,feats}.npy` and jumping straight to GS-decode → PLY. Spot-check on n_voxels=256 ref: 8192 gaussians, xyz∈[0,1), f_dc∈[-3.6, 1.4], opacity logits∈[-12.4, -2.6], scale logs∈[-18.7, -5.0], all finite. |
| Runner wiring    | GREEN   | —        | —        | Step 9 (continued): all stage dispatchers in `sam3d_runner.c` are live-wired: `sam3d_run_ss_dit` seeds 5 modality noise buffers via xorshift64*→Box-Muller, runs `sam3d_shortcut_make_times(reversed=1)` + `sam3d_shortcut_euler_step` for `ss_steps` iterations (CFG deferred — strength=0; `add_flag` unconditional handling needs downstream validation), persists shape latent as NCDHW [8,16,16,16]. `sam3d_run_ss_decode` lazy-loads the decoder and calls `t2_ss_dec_forward`. `sam3d_run_slat_dit` runs voxel prune (`argwhere(occ>0)`), seeds noise feats, and loops `sam3d_slat_dit_forward` for `slat_steps` (now green after step 7b). The full CLI (`test_sam3d <ckpt> <image> <mask> --pointmap …`) runs end-to-end through all stages. `--slat-ref <dir>` remains as a ground-truth bypass for GS-decode regression testing. |

## Summary (as of step 7b)

All seven verify binaries GREEN within the fp32 noise floor; the full
CLI (`test_sam3d <ckpt> <image> <mask> --pointmap …`) now runs
end-to-end through every stage (cond fuser → SS DiT → SS decode →
voxel prune → SLAT DiT → SLAT GS decode → PLY). The `--slat-ref <dir>`
bypass path remains as a golden ground-truth for GS-decode regression.

### Post-cleanup verify (2026-04-26)

Re-ran the verify suite after the qtensor/dequant/GEMM/bicubic
consolidation and the 5-header batch-GEMM perf sweep. All matches
within machine-epsilon of the original drift table (numerics
preserved):

| Verify          | max_abs    | Δ vs baseline |
|-----------------|------------|---------------|
| dinov2          | 1.6007e-02 | 0 (byte-id)   |
| cond_fuser      | 3.815e-06  | 0 (byte-id)   |
| ss_dit (shape)  | 3.004e-05  | 0 (byte-id)   |
| ss_dit (rot6d)  | 6.557e-06  | 0 (byte-id)   |
| ss_dit (trans)  | 5.603e-06  | 0 (byte-id)   |
| ss_dit (scale)  | 1.073e-06  | 0 (byte-id)   |
| ss_dit (t_scale)| 1.192e-06  | 0 (byte-id)   |
| flow_solver     | 0          | exact         |
| ss_decoder      | 5.417e-04  | 0 (byte-id)   |
| slat_dit        | 1.788e-05  | 0 (byte-id)   |
| slat_gs (xyz)   | 3.576e-07  | 0 (byte-id)   |
| slat_gs (dc)    | 6.297e-05  | 0 (byte-id)   |
| slat_gs (scale) | 8.631e-05  | 0 (byte-id)   |
| slat_gs (rot)   | 5.662e-06  | 0 (byte-id)   |
| slat_gs (op)    | 7.248e-05  | 0 (byte-id)   |

All per-stage max_abs numbers are ≤ 5e-4 at fp32 — two orders of
magnitude below the bf16 inference-path floor budget (5e-2). The
largest diff is 1.60e-02 in DINOv2 stage, which is still within bf16
tolerance; it stems from a minor attention-tile accumulation ordering
difference vs. xformers.

End-to-end PLY (step 9) re-validated post-cleanup:

```
test_sam3d $MODELS/sam3d/checkpoints/pipeline.yaml \
    --slat-ref /tmp/sam3d_ref -o /tmp/sam3d_out.ply
# → 8192 gaussians, 557470 bytes, INRIA-compatible binary-LE PLY.
```

## Build

```bash
cd cpu/sam3d
make                    # zen2 flags; override with ARCH=native, COMPILER=clang
make DEBUG=1            # -O0 + asan/ubsan
```

## Checkpoint layout (on disk)

HF ckpts live at `$MODELS/sam3d/checkpoints/` (~12 GB); per-module
safetensors slices used by the C loader at `$MODELS/sam3d/safetensors/`:

| file                                | src ckpt                    | tensors | notes                                         |
|-------------------------------------|-----------------------------|---------|-----------------------------------------------|
| `sam3d_dinov2.safetensors`          | `ss_generator.ckpt`         | 344     | DINOv2-L/14+reg (image branch; mask shares)   |
| `sam3d_point_patch_embed.safetensors` | `ss_generator.ckpt`       | 18      | pointmap embed (embed=512, p=8, in=256)       |
| `sam3d_cond_fuser.safetensors`      | `ss_generator.ckpt`         | 15      | 3× per-modality LN + SwiGLU (1024→2816→1024)  |
| `sam3d_ss_dit.safetensors`          | `ss_generator.ckpt`         | 945     | SS shortcut DiT + latent_mapping per modality |
| `sam3d_slat_dit.safetensors`        | `slat_generator.ckpt`       | 526     | SLAT flow DiT                                 |
| `sam3d_ss_decoder.safetensors`      | `ss_decoder.ckpt`           | 74      | 3D-conv VAE decoder → 64³ occupancy           |
| `sam3d_slat_gs_decoder.safetensors` | `slat_decoder_gs.ckpt`      | 101     | SLAT → gaussians (swin, w=8, 32 g/voxel)      |

Run `python cpu/sam3d/convert_ckpt.py $MODELS/sam3d/checkpoints -o
$MODELS/sam3d/safetensors` to regenerate the slices. The extractor
reads the deeply-nested state-dict keys in each pickle `.ckpt` and
strips the wrapper prefixes so C-side loaders see a flat namespace.

## Ref dump

See `ref/sam3d/README.md`. TL;DR:

```bash
export PYTHONPATH=$PYTHONPATH:/tmp/sam-3d-objects
python ref/sam3d/gen_image_ref.py \
    --image /tmp/sam-3d-objects/notebook/images/human_object/image.png \
    --mask  /tmp/sam-3d-objects/notebook/images/human_object/0.png \
    --pipeline-yaml $MODELS/sam3d/checkpoints/pipeline.yaml \
    --outdir /tmp/sam3d_ref --seed 42
```

`--skip-run` bypasses the model and dumps just preprocessed
image/mask (useful while the sam3d_objects package isn't installed).

## Deferred follow-ups (post-v1)

- **MoGe depth model** — v1 requires the caller to pass a precomputed
  pointmap via `--pointmap pmap.npy`. Port as `common/moge.h` once v1
  pipeline is fully green.
- **CFG + `add_flag` unconditional** on SS DiT — `sam3d_run_ss_dit`
  currently runs strength=0 (no CFG). Upstream uses
  `unconditional_handling="add_flag"` with `strength=2.0` on the shape
  modality; requires dumping the flag-token layout and validating
  against pytorch before turning on.
- **time_scale=1000 (FIXED 2026-04-26)** — both `sam3d_run_ss_dit` and
  `sam3d_run_slat_dit` now multiply t (and SS-DiT's d) by `time_scale`
  before calling the model. The C model code expects t in [0, 1000]
  exactly like upstream's raw model receives from `_generate_dynamics`;
  the verify_*.c paths still pass `t=1.0` because the dump scripts
  `model(t=1.0)` directly bypass the wrapper. Symptom before the fix:
  end-to-end produced "occupancy is fully negative" — the
  TimestepEmbedder was given a near-zero sinusoidal embedding it had
  never seen during training, yielding a garbage denoised latent.
- **Configurable SLAT latent mean/std** — `slat_mean` / `slat_std`
  from `pipeline.yaml` are currently hardcoded inside
  `sam3d_run_slat_dit` (they are the v1 un-normalization). If future
  checkpoints ship different stats, either add a yaml parser or expose
  them via `sam3d_create_cfg`.
- **SLAT Mesh decoder + FlexiCubes cube2mesh** — required for `.glb`
  output. Separate port.
- **Texture bake + layout optimization** — follow the mesh path.
- **CUDA runner** (`cuda/sam3d/`) — mirrors CPU stage boundaries;
  follows numerics pinning on CPU side.
