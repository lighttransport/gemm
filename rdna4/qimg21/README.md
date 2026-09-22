# Qwen Image 2.1 HIP runner

This directory contains the RDNA4 native Qwen Image 2.1 denoiser. It is a
separate runner from `rdna4/qimg`: Qwen Image 2.1 uses 32 transformer blocks,
hidden size 4096, 32 attention heads, head dimension 128, and BF16 weights.

The runner keeps the validated Qwen 2.1 host graph and replaces its CUDA driver
and cuBLAS boundary with the project-local dynamic HIP/HIPRTC layer. GEMMs use
the gfx1201 BF16 WMMA path when available, with a HIPRTC F32-accumulate BF16
fallback. The checkpoint is streamed one transformer block at a time, so the
16-GiB RX 9070 XT does not need a second full model copy.

Build:

```sh
make -C rdna4/qimg21
```

The executable consumes the same deterministic fixture ABI as the CUDA 2.1
runner:

```sh
rdna4/qimg21/test_hip_qimg21_native \
  --model /mnt/disk2/models/qimg-21 \
  --prompt-embeds prompt_embeds.npy --latents latents.npy \
  --height-tokens 16 --width-tokens 16 --steps 2 \
  --attention wmma --out native_latents.npy
```

The RDNA4 runner uses WMMA GEMMs by default when the GPU supports them.
`--attention wmma` explicitly requests that same GEMM path; attention itself
currently uses the portable scalar implementation. `--attention math` also
uses WMMA GEMMs by default. The runner covers the 32-layer
denoiser ABI, CFG inputs, editing layout inputs, stage dumps, and FlowMatch
latent updates. Matching native text, vision, VAE decoder, and VAE encoder
executables are built by the same Makefile. ROCm web generation uses the
native VAE path when those binaries are present; Python remains an explicit
correctness fallback. Health reports each native component separately.

For standalone generation with only NumPy and Pillow installed, run
`python3 cuda/qimg21/native_generate.py --backend rocm --model MODEL_ROOT`.
The ROCm path defaults to native VAE decode and uses a deterministic NumPy
latent seed. CUDA keeps its PyTorch CUDA RNG seed path, so identical seed
numbers across the two backends do not produce identical initial noise.
Both text-to-image and image-editing paths have been smoke-tested at 256x256
with one denoising step on an RX 9070 XT. This verifies execution, not
multi-step image quality or CUDA parity. The current matched-input 256x256
denoiser prediction has cosine similarity about 0.99995 against the saved
PyTorch reference, below the CUDA runner's 0.99996 regression gate.
