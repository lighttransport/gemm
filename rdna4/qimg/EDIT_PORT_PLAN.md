# Qwen-Image-Edit-2511 on RDNA4: native INT4 DiT port plan

## Goal

Run Qwen-Image-Edit-2511 (Apache-2.0) on the 16 GB RX 9070 XT (gfx1201) fast
enough for per-view multiview texture completion
(`server/vhuman/reconstruction/mv_texture.py`, backend `qwen_edit_seq`).

Baseline: diffusers with the GGUF Q4_K_M transformer.
- 1100 s per 512² view at 20 steps (≈55 s/step). GGUF dequantizes on every step.
- Peak 14.6 GB.
- The text encoder runs on the CPU.

Target: under 5 s/step at 1024² (about 1 min per view at 12 steps).

## Architecture decision: hybrid, native DiT only

Edit-2511 is the Qwen-Image 1.0 MMDiT: 60 blocks, 3072 hidden, joint
attention, Qwen2.5-VL 3584-dim text states. That matches `rdna4/qimg`. It does
not match `rdna4/qimg21`, which is 32×4096 with Qwen3-VL.

Only the DiT step is hot: about 98% of the time over 12–40 steps. Everything
else runs once per image and stays in Python (diffusers), called through ctypes:

| component | where | why |
|---|---|---|
| Qwen2.5-VL prompt + vision encoding | Python, CPU | runs once per edit; porting the vision tower costs weeks |
| VAE encode (reference latents) and decode | Python, GPU, after the DiT is unloaded or before it loads | runs once; needs about 1–2 GB |
| scheduler, CFG combine, packing | Python | trivial |
| **DiT forward** | **native HIP INT4 W4A16, all 60 blocks resident** | the hot loop |

## Steps

### 1. Weights: own SVDQuant from BF16, not the Nunchaku checkpoint

The Nunchaku `wscales` decode is still unsolved (`nunchaku.md`, about 3× norm
error). `tools/svdquant_from_bf16.py` already writes the logical layout
(`.qint4/.wscale/.smooth/.lora_down/.lora_up/.bias`) from BF16 weights.

- Download the Qwen-Image-Edit-2511 BF16 `transformer/` shards (about 41 GB).
- Pack to rank-128 INT4 (about 12–13 GB), then delete the BF16 shards.
- Edit-2511 has the same tensor names as Qwen-Image. Confirm with a key diff.
  `zero_cond_t` is a config flag, not a weight.
- Gate: per-linear cos ≥ 0.999 against BF16 on dumped activations, using the
  existing `--test-int4-dequant` pattern.

### 2. Make INT4 render-viable (blocker in `INT4_W4A16_STATUS.md`)

- `op_int4_linear` (`hip_qimg_runner.c:1153`) runs the rank-128 LoRA residual
  through the scalar `op_gemm` with f32 expansion. Route `lora_down` and
  `lora_up` through BF16 WMMA (`op_wgemm_bf16`); the weights are already BF16.
- Better: fuse `lora_up·(lora_down·x)` into the main kernel's epilogue (the
  Nunchaku layout). Do this only after the WMMA route works.
- Remove the per-linear `hipDeviceSynchronize()` by giving each linear its own
  scratch slot or using a stream-ordered allocator.
- Gate: one Qwen-Image T2I step at 1024² completes. Image cos against the BF16
  native path ≥ 0.99.

### 3. Edit sequence layout in `hip_qimg_dit_step`

Add `hip_qimg_dit_step_edit(r, img_tokens, n_img, ref_tokens, n_ref, ref_shapes[], txt, n_txt, t, zero_cond_t, out)`:

- **Sequence:** `[txt | img (noisy) | ref_1 | ... | ref_k]`. The image stream is
  `img ++ refs`. Joint attention already handles arbitrary `n_img`.
- **RoPE:** today `hp_rope = sqrtf(n_img)` assumes one square grid. Pass
  explicit `(frame, h, w)` per segment, matching diffusers `img_shapes`. The
  noisy image is frame 0 and reference i is frame i+1, with positions centred
  per segment. Port `QwenEmbedRope` indexing exactly.
- **Text RoPE offset:** starts after the max image extent, as in diffusers.
- **Output:** return only the first `n_img` tokens of the velocity.

### 4. `zero_cond_t` (2511): per-segment modulation

- The noisy tokens use timestep t. The reference tokens use timestep 0.
- Compute two timestep embeddings, which gives two sets of `img_mod`
  (shift/scale/gate ×2) per block.
- Apply them by token range. This is a second adaLN launch over the `[n_img, n_img+n_ref)` slice, not a new kernel.
- The text stream is unchanged.

### 5. Python binding and backend

- Add a `ctypes` wrapper `rdna4/qimg/qimg_edit_native.py`. It reuses
  diffusers' `QwenImageEditPlusPipeline` for encode_prompt, VAE and scheduler,
  and replaces `pipe.transformer.forward` with the native step.
- **Memory order:**
  - encode the prompt (CPU)
  - load the VAE, encode the references and the init latent, free the VAE
  - load the INT4 DiT (about 14 GB) and denoise
  - unload it (`hip_qimg_unload_dit`), then decode with the VAE
- `server/vhuman/reconstruction/qwen_edit_backend.py` chooses the native
  backend when the INT4 package exists, otherwise the GGUF fallback.

### 6. Validation ladder

1. Single block: native vs diffusers BF16 hooks on the same inputs, cos ≥ 0.999.
2. Full DiT step, edit layout with 2 references, against diffusers (GGUF or BF16
   on the CPU), cos ≥ 0.99.
3. End to end: the `right` view of the Obama candidate. Same seed against the
   GGUF output; visual check of alignment, ear and skin tone.
4. Full `mv_texture generate --backend qwen_edit_seq`, then bake and eval.

## Risks

- **Memory:** 2 references at 1024² is about 12k image tokens plus text. The
  LoRA scratch `n_out·n_tok·4` (img_mlp_fc1 12288 × 12k ≈ 600 MB) and attention
  fit only after step 2 frees the f32 expansions.
- **SVDQuant quality:** our own pass reached cos 0.993 on modulation. Edit is
  sensitive to identity, so keep modulation and `img_in` in BF16 if needed
  (about +0.4 GB).
- **Gate failures:** ROCm parity gates on qimg21 already miss 0.99996. Use
  cos ≥ 0.99 at the image level as the practical gate.

## Text encoders are stock checkpoints (verified 2026-10-07)

Checked by sampled tensor byte equality over HTTP range reads.

| model | encoder | tensors | sampled, byte-identical |
|---|---|---|---|
| Qwen-Image-Edit-2511 | stock `Qwen/Qwen2.5-VL-7B-Instruct` | 729 (same names) | 13/13 (LM layers 0/17/20/27, final norm, vision block 0, merger, patch embed) |
| Qwen-Image-2.1 | stock `Qwen/Qwen3-VL-8B-Instruct` | 750 (same names) | 14/14 |

So the official quantized releases are valid drop-ins, for example `Qwen/Qwen3-VL-8B-Instruct-FP8` (block-128 FP8).
That cuts the CPU-bound prompt encode (currently bf16 on the CPU), or lets the encoder be paged onto the GPU between
DiT runs.

## Status (2026-10-08)

Done:
- Steps 1–5: own SVDQuant pack, tiled WMMA INT4/INT8 GEMM, edit layout (multi-segment RoPE, `zero_cond_t`),
  ctypes driver.

Step time at 12.8k tokens (normal clocks): 45 s, then 5.85 s.
- tiled WMMA GEMM with grouped rasterization and fused LoRA
- fa16 attention (rdna4/fa2 port, exact exp2)
- fused QKV norm/RoPE/pack
- wide INT4 loader
- chunked image MLP
- int4 arena

Parity investigation. Native edits lost the subject's identity, and three distinct causes were found:
1. **INT4 modulation linears** (img_mod/txt_mod).
   - Ground truths: BF16 diffusers via group offload (`tools/edit_parity_bf16.py`), and diffusers math with our
     weights (`tools/edit_parity_fakequant.py`, `FQ_KEEP_BF16`).
   - These showed the mods dominate. At the late step, rel_l2 was 21% with INT4 mods and 2% with BF16 mods.
   - Fix: `hip_qimg_set_mod_vectors` with `QIMG_HOST_MOD=1`. Mods are computed exactly from host BF16 weights,
     once per view, which also frees 4.3 GB of VRAM.
2. **Sensitive linears.** Q4_K_M keeps v-proj and MLP-down at Q6_K. `svdquant_from_bf16 --int8` (`LdInt8G64`) does
   the same at 8 bit. The mixed pack is 11.3 GB resident.
3. **Pipeline flow.** A CPU-device diffusers flow with VAE and vision wrappers lost identity even with a correct
   DiT. The driver now uses the GGUF editor's cuda flow.

Calibrated smoothing (`tools/edit_calib_bf16.py`) did not help: it stayed at rel_l2 about 20% while the mods were
INT4. Final per-step error vs BF16 is 1.8% at t = 0.31 and 4.2% at t = 1. GGUF is 1.3% and 2.8%.

Per view at 1024² with 2 references: DiT 161 s, FP32 CPU prompt encode 123 s, other about 40 s.

Open:
- the speed A/B was queued but not run: `QIMG_REF_SIDE` (smaller references), `QIMG_VISION_CACHE`, 32 encoder
  threads
- an FP8 GPU encoder
