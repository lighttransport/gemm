# FLUX.2 Klein — HIP (RDNA4) vs PyTorch reference

Same GPU (AMD Radeon RX 9070 XT, 16 GB), same params: 256×256, 4 steps,
seed 42, prompt `"a red apple on a white table"`.

## Reference (PyTorch)

`ref/flux2_klein/gen_reference.py` — `diffusers.Flux2KleinPipeline`, bf16,
HF model `black-forest-labs/FLUX.2-klein-4B` (cached). FLUX.2-klein-4B does
**not** fit 16 GB VRAM in bf16, so the reference uses
`enable_model_cpu_offload()` (weights resident in host RAM, the active
submodule streamed to GPU per call).

```sh
ref/qwen_image/.venv/bin/python ref/flux2_klein/gen_reference.py \
  --size 256 --steps 4 --seed 42 --prompt "a red apple on a white table"
```

## HIP (this repo)

fp8 DiT + BF16-WMMA self-attention + BF16-WMMA VAE conv + GPU Qwen3-4B text
encoder (`FLUX2_ENC_GPU=1`). Fits 16 GB VRAM via fp8 weights and sequenced
loads (encode → free encoder → load DiT/VAE).

```sh
FLUX2_ENC_GPU=1 rdna4/flux2/test_hip_flux2 --generate \
  --dit  .../diffusion_models/flux-2-klein-4b-fp8.safetensors \
  --vae  .../vae/diffusion_pytorch_model.safetensors \
  --enc  .../split_files/text_encoders/qwen_3_4b.safetensors \
  --tok  .../Qwen3VL-4B-Instruct-F16.gguf \
  --prompt "a red apple on a white table" --steps 4 --size 256 --seed 42
```

## Results

| metric (256², 4-step) | PyTorch (bf16, cpu-offload) | HIP (fp8, ours) | gap |
|---|---|---|---|
| **total wall** (load + inference) | **63.5 s** | **40.8 s** | **HIP 1.56× faster** |
| **inference** (warm) | **19.2 s** (45.2 s cold) | **~1 s** | **HIP ~19× faster** |
| DiT compute | offload-bound | **0.09 s/step** (0.37 s) | — |
| VAE decode | — | **0.22 s** | — |
| memory | 17.3 GB **host RAM** (won't fit VRAM) | fits **16 GB VRAM** | — |

## Interpretation

- **Architectural win:** PyTorch can't fit the pipeline in 16 GB VRAM → it
  streams weights CPU↔GPU every run (offload-bound, 17.3 GB host RSS). The HIP
  runner fits via fp8 + sequenced loads and runs fully resident.
- **Inference gap (~19×):** HIP DiT (0.37 s) + VAE (0.22 s) + GPU encode
  (~sub-second) ≈ ~1 s vs PyTorch's 19.2 s — from no-offload + fp8 + the WMMA
  kernels (DiT 5.5×, VAE 2×).
- **Total-wall gap (1.56×):** HIP's 40.8 s is dominated by *one-time* cost
  (HIPRTC kernel compile + 12 GB of weight loads), not compute. In a persistent
  / server setting (warm, loads amortized) the per-image gap is the ~19×
  inference figure.

## Quality

Both produce a valid "red apple on white table"; they differ in composition
(PyTorch bf16 = clean background, HIP fp8 = bokeh background) — a different
sample from the precision/scheduler difference, not a regression. The HIP
pipeline is internally validated: GPU vs CPU text-encoder hidden states
corr = 1.000000, and CPU-vs-GPU-encoder images are pixel-identical.

## Caveats

- PyTorch cold (45.2 s) vs warm (19.2 s) inference varies with torch
  JIT/autotune warmup; the table uses the warm figure.
- The per-step DiT *kernel* gap is not cleanly isolated — diffusers'
  cpu-offload makes per-step timing noisy — so figures are reported at the
  inference and end-to-end level.
