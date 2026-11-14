# ref/cosmos3 - PyTorch oracle for nvidia/Cosmos3-Edge-Policy-DROID

Oracle = installed diffusers 0.43.0 `Cosmos3OmniPipeline` + `Cosmos3OmniTransformer` (bf16 weights, bf16 compute, CUDA), action mode `policy`.
`accelerate` is not installed, so `gen_ref.py` assembles the pipeline by hand from the checkpoint files (config.json + the 2 transformer shards, vae/,
text_tokenizer/, scheduler/); no code from the model repo is executed.

## Policy-mode invocation (what the pipeline does)
```
cond = CosmosActionCondition(mode="policy", chunk_size=32, domain_name="droid_lerobot", resolution_tier=256,
                             image=<one PIL image>, view_point="concat_view")
pipe(prompt=..., action=cond, num_inference_steps=4, guidance_scale=3.0, fps=15.0 (checkpoint.json conditioning_fps),
     latents=..., action_latents=..., output_type="latent", enable_safety_check=False)
```
* ONE conditioning image (exterior|wrist views concatenated -> `concat_view`). It is down-scaled (never up) into the tier-256 bin closest in aspect
  ((192,320) for 16:9-ish), padded (reflect), VAE-encoded (mean, `latents_mean/std` normalised); padding latent rows are cropped (192/16=12 -> 11 rows).
* Prompt is turned into the JSON caption `{"cinematography":{"framing":...},"actions":[{"time":"0:00-0:02","description":...}],"duration","fps","resolution","aspect_ratio"}`;
  chat template: empty system block + user + `<think>\n<|im_end|>` generation prompt; then `eos, <|vision_start|>` appended. CFG uncond text = `""` through the same template (19 tokens).
* Joint sequence: und = text tokens (126 cond / 19 uncond), then gen = 9 latent frames x (6x10) patch tokens (frame 0 clean = conditioning, 8 noisy) + 32 action tokens.
  Packing: patch (p,q,c) order; mrope 3 axes with FPS modulation (float t positions: `off + frame/(fps/4)*(24/4)`, action tokens start_frame_offset=1, `temporal_modality_margin 15000`).
* Understanding tower is causal over text only; generation tower attends to [und keys (after `k_norm_und_for_gen`) | gen keys], no mask. Noisy-frame/action tokens add
  `time_embedder(sinusoid(t*1e-3))` (fp32 embedder). Action tokens: `DomainAwareLinear` (per-domain weight, domain 8 = droid_lerobot) + `action_modality_embed`.
* Sampler: UniPC (order 2, bh2, flow prediction, predict_x0, lower_order_final), `use_native_flow_schedule`: sigmas = linspace(0.999, 0, steps+1)[:-1], then flow_shift 5.0 applied
  (s' = 5s/(1+4s)) -> sigmas 0.9998/0.9373/0.8331/0.6247/0, timesteps = int(1000*sigma) = 999/937/833/624 for 4 steps (the transformer gets the integer t, x1e-3 inside).
  CFG: `v = uncond + g*(cond-uncond)` on vision and action separately; separate scheduler copies for vision and action latents. Output = `action_latents[:, :10]` (raw, NOT
  normalised/de-normalised by the diffusers pipeline; README says 8 values/step but diffusers' table gives droid_lerobot raw width 10).
* The `vision_encoder/` (SigLIP-style, for the Reasoner/understanding path) is NOT used by the diffusers pipeline in policy mode. No image goes through it.

## Run
```
~/.venv/bin/python -I ref/cosmos3/gen_ref.py [--out tmp/cosmos3_ref] [--steps 4] [--guidance 3.0] [--decode]
~/.venv/bin/python -I ref/cosmos3/compare.py --ref-dir tmp/cosmos3_ref --ours-dir tmp/cosmos3_out
```
Outputs in `tmp/cosmos3_ref` (git-ignored, ~330 MB): `noise_vision/noise_action` (raw randn, CPU generator seed 1234), `latents_init` (cond frame = VAE x0, rest noise),
`action_init`, `x0_latent_frame0`, `input_ids_{cond,uncond}`, `position_ids_cond`, `cos/sin_{und,gen}`, per-layer `s{step}_{cond|uncond}_layerNN_{und,gen}` (steps 0 only),
layer inputs, norms, `s*_pred_vision/pred_action` (transformer output, all steps), `step*_{vision,action}_{velocity_cfg,sample_in,sample_out}`, `final_action`, `final_video_latent`, `meta.json`.

## Measured (RTX 5060 Ti 16GB, torch 2.15 cu132)
* 4 steps, guidance 3.0 (8 transformer calls, 126/19 text tokens + 572 gen tokens): pipeline call 2.76 s (first call 1.36 s warmup, then 0.08 s/call).
* torch allocated: 7.61 GiB after load (transformer incl. unused lm_head 6.3 + VAE 1.3), peak 7.91 GiB during the 4-step cfg-3 call (the bf16 oracle needs the 16 GB card nearly half-full even without the sampler workspace).

## Other helpers
* `gen_ref.py --fp32` : transformer in fp32 (bf16 weights upcast, lm_head dropped; built on CPU, moved after the VAE encode) -> `noise_floor.py` prints how far
  the bf16 oracle and the CUDA runner are from the fp32 truth (the honest tolerance for bf16-vs-bf16 comparisons).
* `tokenize_prompt.py` : standalone token ids (chat template + JSON action caption + eos + `<|vision_start|>`), `--check` verifies against the exported ids.
* `convert_vae.py` : checkpoint `vae/` (diffusers AutoencoderKLWan, bf16) -> Wan2.2_VAE.pth key layout (fp32) for `cuda/praxis/cuda_wan22_vae.h`, and verifies all 196 tensors
  against `$PRAXIS_ROOT/wan2.2-vae` bit for bit after bf16 rounding of the original (0 differences).
* `cond_canvas.npy` : frame 0 of the padded canvas video the VAE sees (bf16 rounded, [-1,1], [3,192,320]) - the CUDA pipeline feeds it to its VAE.
