# Native FP8 Reference Setup

This directory owns a local `uv` environment for Qwen-Image FP8 diagnostics.
Use Python 3.12 because the ROCm 7.2.2 PyTorch wheels are `cp312`.

```bash
cd rdna4/qimg
UV_CACHE_DIR=$PWD/.uv-cache uv venv --python 3.12 --clear .venv
UV_CACHE_DIR=$PWD/.uv-cache uv sync
UV_CACHE_DIR=$PWD/.uv-cache uv pip install \
  https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.2/triton-3.6.0+rocm7.2.2.git4ed88892-cp312-cp312-linux_x86_64.whl \
  https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.2/torch-2.11.0+rocm7.2.2.lw.git4e323059-cp312-cp312-linux_x86_64.whl
```

Verify outside Codex sandbox if `/dev/kfd` is hidden:

```bash
.venv/bin/python -c 'import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")'
```

Current expected result on the RX 9070 XT box:

```text
2.11.0+rocm7.2.2.git4e323059 7.2.53211 True
AMD Radeon RX 9070 XT
```

Low-risk pinned checks:

```bash
.venv/bin/python tools/native_fp8_ref.py compare-latents
.venv/bin/python tools/native_fp8_ref.py quant-sweep --input ../../ref/qwen_image/final_latent_packed_256.bin --cols 64
```

Native FP8 GEMM probes:

```bash
.venv/bin/python tools/native_fp8_ref.py gemm-probe --m 256 --n 256 --k 256 --scale-div 512
.venv/bin/python tools/native_fp8_ref.py gemm-probe --m 256 --k 64 --scale-div 512 \
  --w-safetensors /mnt/disk1/models/qwen-image/diffusion_models/qwen_image_fp8_e4m3fn.safetensors \
  --tensor img_in.weight
.venv/bin/python tools/native_fp8_ref.py gemm-probe --m 128 --k 3072 --n 3072 --scale-div 512 \
  --w-safetensors /mnt/disk1/models/qwen-image/diffusion_models/qwen_image_fp8_e4m3fn.safetensors \
  --tensor transformer_blocks.0.attn.to_q.weight
```

Pinned BF16xFP8 quality / traffic check:

```bash
QIMG_FP8_WMMA=1 ./test_hip_qimg --generate \
  --dit /mnt/disk1/models/qwen-image/diffusion_models/qwen_image_fp8_e4m3fn.safetensors \
  --height 256 --width 256 --steps 20 \
  --init-bin ../../ref/qwen_image/init_latent_256.bin \
  --txt-bin ../../ref/qwen_image/apple_text_256.bin \
  --sigmas-bin ../../ref/qwen_image/sigmas_256.bin \
  --ref-final ../../ref/qwen_image/final_latent_packed_256.bin \
  --path-stats --mem-stats
```

Current pinned result: latent cosine `0.999971`, PSNR `51.70 dB`,
denoise `24.1 s`, GEMM traffic `1723 GB / 73.1 TF`.

Ungated native FP8xFP8 check:

```bash
QIMG_FP8_FP8_WMMA=1 ./test_hip_qimg --generate \
  --dit /mnt/disk1/models/qwen-image/diffusion_models/qwen_image_fp8_e4m3fn.safetensors \
  --height 256 --width 256 --steps 20 \
  --init-bin ../../ref/qwen_image/init_latent_256.bin \
  --txt-bin ../../ref/qwen_image/apple_text_256.bin \
  --sigmas-bin ../../ref/qwen_image/sigmas_256.bin \
  --ref-final ../../ref/qwen_image/final_latent_packed_256.bin \
  --path-stats --mem-stats
```

Current pinned result: latent cosine `0.979988`, PSNR `24.59 dB`,
denoise `20.3 s`, GEMM traffic `984 GB / 73.1 TF`, extra persistent
activation-FP8 scratch `3.0 MB`. `--fp8-quality-target-db` is diagnostic
annotation only; it does not suppress FP8xFP8 dispatch.

Activation scale experiments:

```bash
QIMG_FP8_FP8_WMMA=1 ./test_hip_qimg --generate ... --fp8-act-scale-mode comfy
QIMG_FP8_FP8_WMMA=1 ./test_hip_qimg --generate ... --fp8-act-scale-mode clamp
```

`comfy` mirrors ComfyUI's native FP8 CUDA fast path (`scale=1`, clamp/cast)
but saturates later qimg activations in the full denoise run: latent cosine
`0.301305`, PSNR `4.20 dB`. `clamp` keeps scale 1 only while the row fits
FP8 range and otherwise uses `row_max/448`; it improves the full run to
cosine `0.986565`, PSNR `25.93 dB`, still far below the 50 dB target without
BF16 fallback/gating. Image artifact: `apple_fp8fp8_clamp_256.png`.
