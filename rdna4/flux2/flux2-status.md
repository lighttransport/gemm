# RDNA4 Flux.2 Klein Status

## Model Paths

- DiT: `/mnt/disk1/models/klein2-4b/diffusion_models/flux-2-klein-4b-fp8.safetensors`
- VAE: `/mnt/disk1/models/klein2-4b/vae/diffusion_pytorch_model.safetensors`
- Text encoder: `/mnt/disk1/models/klein2-4b/split_files/text_encoders/qwen_3_4b.safetensors`
- Tokenizer: `/mnt/disk1/models/Qwen3-VL-4B-Instruct-GGUF/Qwen3VL-4B-Instruct-F16.gguf`

The text encoder checkpoint was downloaded from `Comfy-Org/vae-text-encorder-for-flux-klein-4b`.
It contains the Qwen3 4B tensors expected by `common/flux2_klein_text_encoder.h`.

## Verified Commands

Standalone DiT verifier:

```sh
rdna4/flux2/verify_dit \
  --dit /mnt/disk1/models/klein2-4b/diffusion_models/flux-2-klein-4b-fp8.safetensors \
  --lat 8 --ntxt 16 --threads 16
```

Observed: max diff `0.000294`, corr `1.000000`, GPU `0.17 s`.

Standalone VAE verifier:

```sh
FLUX2_SKIP_VAE_BN=1 rdna4/flux2/verify_vae \
  --vae /mnt/disk1/models/klein2-4b/vae/diffusion_pytorch_model.safetensors \
  --lat 8
```

Observed: max diff `0.000010`, mean diff `0.000001`, GPU `0.072 s`.

Real-text 64px smoke:

```sh
rdna4/flux2/test_hip_flux2 --generate \
  --dit /mnt/disk1/models/klein2-4b/diffusion_models/flux-2-klein-4b-fp8.safetensors \
  --vae /mnt/disk1/models/klein2-4b/vae/diffusion_pytorch_model.safetensors \
  --enc /mnt/disk1/models/klein2-4b/split_files/text_encoders/qwen_3_4b.safetensors \
  --tok /mnt/disk1/models/Qwen3-VL-4B-Instruct-GGUF/Qwen3VL-4B-Instruct-F16.gguf \
  --prompt "a red apple on a white table" --steps 1 --size 64 --seed 42 \
  -o /tmp/flux2_real_text_64.ppm
```

Observed: 15 real prompt tokens, DiT `0.47 s`, GPU VAE `0.06 s`.

Real-text 256px smoke:

```sh
rdna4/flux2/test_hip_flux2 --generate \
  --dit /mnt/disk1/models/klein2-4b/diffusion_models/flux-2-klein-4b-fp8.safetensors \
  --vae /mnt/disk1/models/klein2-4b/vae/diffusion_pytorch_model.safetensors \
  --enc /mnt/disk1/models/klein2-4b/split_files/text_encoders/qwen_3_4b.safetensors \
  --tok /mnt/disk1/models/Qwen3-VL-4B-Instruct-GGUF/Qwen3VL-4B-Instruct-F16.gguf \
  --prompt "a red apple on a white table" --steps 4 --size 256 --seed 42 \
  -o /tmp/flux2_real_text_256_s4.ppm
```

Observed: 15 real prompt tokens, DiT `2.53 s` total (`0.63 s/step`), GPU VAE `0.38 s`.

## Notes

- `make -C rdna4/flux2 test-gen` now uses the discovered real text encoder and tokenizer by default.
- The text encoder currently runs on CPU and is the main startup cost for one-off generation.
- The generator front-pads text to 512 tokens, so the DiT sees `n_txt=512` even for short prompts.
- These smoke tests validate the full text-to-image plumbing and GPU performance path. They are not final image quality validation.
