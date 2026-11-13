# cuda/cosmos3 - nvidia/Cosmos3-Edge-Policy-DROID (4B, Cosmos3OmniTransformer "cosmos3_edge_nemotron_dense") on CUDA (sm_120a, 16 GB)

Open stand-in for praxis-1. Policy mode = one conditioning image (+ text) -> jointly denoised future video latent (9 latent frames) and a 32 x 10 action chunk (DROID, domain 8).
Oracle: diffusers 0.43 `Cosmos3OmniPipeline` (see `ref/cosmos3/README.md` for the exact pipeline semantics and `gen_ref.py` for the invocation).

## Files
* `cuda_cosmos3_runner.h` host runner (safetensors shards read directly, cuBLAS bf16 GEMMs via `cublasew_gemm_ex_rm`, UniPC sampler, npy dumps)
* `cosmos3_kernels.h` NVRTC kernels (nemotron RMSNorm, per-head q/k norm, rope with torch's bf16 rounding points, relu2, patchify/unpatchify, naive attention fallback)
* attention reuses `cuda/praxis/praxis_kernels.h::pr_flashiv128` (tensor-core block-sparse flash, per-query key intervals: und causal [0,i+1), gen full [0,nk)); `COSMOS3_NAIVE_ATTN=1` = own simple kernel
* `test_cuda_cosmos3.c` parity driver (teacher-forced or free-running), `cosmos3_policy.c` end-to-end driver (VAE + transformer + sampler)
* `Makefile`: `make refs` (oracle dumps), `make vae_weights`, `make test` (parity), `make e2e` (full pipeline)

## Run (from the repo root; needs ~/.venv with torch+diffusers only for the oracle)
```
~/.venv/bin/python -I ref/cosmos3/gen_ref.py                    # oracle dumps -> tmp/cosmos3_ref (326 MB)
~/.venv/bin/python -I ref/cosmos3/convert_vae.py                # VAE weights in the layout cuda_wan22_vae.h expects
make -C cuda/cosmos3 test e2e
cuda/cosmos3/cosmos3_policy --repeat 3 [--cache-und] [--keep-vae] [--oracle-latent] [--guidance 1.0 --steps N]
```

## Design (what makes it fit < 8 GB and fast)
* **The understanding tower is step-invariant.** It sees only text (causal), so its per-layer K (after `k_norm_und_for_gen` + rope) / V are identical for all denoising steps.
  Run once per prompt (cond + uncond), weights **streamed layer by layer** from the mmap'd shards through two 96 MB device slots (double-buffered copy stream); only the 14 MB/prompt K/V caches are kept.
  The cache depends on the text only, so it is reusable across control steps with the same instruction (`--cache-und`).
* Resident on device: generation tower 2.64 GB bf16 (28 x 94 MB + heads, 8 MB per-domain action tables). `embed_tokens` (0.5 GB) is gathered on the host, `lm_head` (0.5 GB) is unused by the policy and never loaded.
* bf16 storage / fp32 math, rounded at the same places as torch (Linear, RMSNorm, residual add, the 3 rope products), so per-layer differences are bf16 noise.
* Only the first latent frame is VAE-encoded (the causal Wan VAE: latent frame 0 depends on pixel frame 0 only; the pipeline pads the other 32 frames with copies and masks their latents with noise anyway).

## Parity vs the oracle (RTX 5060 Ti; 126 cond / 19 uncond text tokens, 540 vision + 32 action tokens, 4 UniPC steps, cfg 3.0)
Teacher-forced (oracle inputs at every step), rel-L2 / cos:

| tensor | rel-L2 |
|---|---|
| rope tables cos/sin (und, gen), embeddings, proj_in | 0 (bit exact) |
| layer 0 gen / und | 3.5e-4 / 2.4e-3 |
| layer 13 gen / und | 9.4e-3 / 1.6e-3 |
| layer 27 gen / und, final norm | 1.3e-2 / 1.3e-2, 1.2e-2 |
| per-step transformer output, vision (steps 0-3, cond and uncond) | 0.95e-2 .. 1.1e-2 (cos >= 0.99993) |
| per-step transformer output, action | 2.5e-3 .. 4.4e-3 (cos >= 0.99999) |

Free-running (own latents, UniPC on host): step-0 CFG velocity vision 2.7e-2 / action 0.93e-2; final **action chunk 1.3e-2 (cos 0.99992)**; final video latent 4.8e-2 (cos 0.9988).
**Noise floor** (`ref/cosmos3/noise_floor.py`, oracle rerun with an fp32 transformer): the bf16 oracle itself is 4.0e-2 (layer 18) / 1.4e-2 (final action) / 3.6e-2 (video latent) away from fp32 truth;
the CUDA run is 4.0e-2 / 1.2e-2 / 4.2e-2 away - i.e. as close to fp32 as the bf16 oracle is (and in a few mid layers 16-26 the bf16 oracle-vs-CUDA diff exceeds 0.02 for that reason alone).
`make test` therefore uses 0.02/0.9999 for actions and 0.035/0.9995 for per-layer/per-step tensors, 0.06/0.998 for the CFG-amplified video latent.

End to end with our own VAE (`cosmos3_policy`): clean latent frame 0 vs oracle 6.3e-3 (cos 0.99998; the oracle VAE runs in bf16, ours fp16 convs / fp32 else); final action vs oracle 1.3e-2 (cos 0.99991).

## Performance / memory (RTX 5060 Ti 16 GB, PCIe Gen4 x8; 4 steps, cfg 3.0 = 8 generation passes)
| stage | time |
|---|---|
| transformer load (2.6 GB H2D + NVRTC cache) | 0.32 s (once) |
| VAE load / first encode (cold) / steady encode of the frame | 1.6 s / 0.6 s / 20 ms |
| und tower, cond+uncond, weights streamed (cold / warm) | 0.77 s / 0.30 s (0 with `--cache-und`) |
| one generation pass (572 tokens) | 62 ms (oracle PyTorch: 80 ms) |
| denoise, 4 steps cfg 3 / no cfg | 0.50 s / 0.23 s |
| **policy call, steady state** | **0.83 s** (VAE + und + denoise); 0.51 s with the und cache; guidance 1.0 (single pass per step) not re-timed warm |
Oracle for comparison: 2.6 s for the same pipeline call (first call), 0.08 s per transformer call.

Device memory (cuMemGetInfo total-free, so it includes CUDA context ~137 MB and cuBLAS workspaces):
| stage | MB |
|---|---|
| VAE load + encode (fp32 weights 1.76 GB + im2col scratch), then freed | ~2000 |
| transformer: gen tower + 2 und slots + KV caches + scratch | ~3400 |
| **everything resident** (VAE kept + transformer, `--keep-vae` / `--repeat`) | **~5150** |
| bf16 PyTorch oracle (for scale) | 7910 (torch peak allocated) |
**< 8 GB is met without FP8 or other quantisation** (sequenced 3.4 GB peak, all-resident 5.2 GB). The 6.7 GB bf16 transformer is not resident: und tower streams (2.8 GB/call H2D, overlapped), lm_head/embed stay out.

## Not done / gaps
* `vision_encoder/` (SigLIP2-style ViT: 27 layers, 1152 wide, patch 16, 2x2 merge + MLP projector to 2048, 489 M params, `config.json` missing from the download) is the Reasoner (image-token) path;
  the diffusers policy pipeline never calls it (the image enters via the VAE latent frame). Not implemented, not needed for the policy; it would fit (1 GB bf16).
* Image resize/crop/reflect-pad of the observation and the tokenizer run in Python (`gen_ref.py` exports `cond_canvas.npy`, `tokenize_prompt.py` the ids); the C side takes the canvas frame and ids.
* DROID action: the pipeline returns the raw 10-d model-space action (9D pose + gripper); README says 8 values, and there is no de-normalisation in diffusers (checkpoint NormStats not shipped) - not reproduced.
* Video decode (VAE decoder from latents) is not wired to the new weights (the converted file works with `cuda_wan22_vae_decode`, untested on these latents).
* Layer dumps for steps >= 1 and the und tail of layer 27 are computed but only step 0 is compared per layer.
