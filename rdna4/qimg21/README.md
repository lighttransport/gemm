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
native VAE path. A separate PyTorch reference mode is available when a ROCm
PyTorch environment with Qwen Image 2.1 Diffusers support is configured.
Health reports each native component separately.

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

On the RX 9070 XT with the installed HIPRTC 9.0 runtime, a matched-input
256x256 two-step denoiser run takes 6.76 seconds (14,102,172 KiB peak host RSS) and
is byte-identical to the earlier ROCm implementation. The earlier run took
30.26 seconds: direct H2D upload from the safetensors mmap eliminates a
redundant host copy for each BF16 matrix. Set `QIMG21_PROFILE=1` for diagnostic
per-block upload, compute, and release timing. At 1024x1024, one step takes
37.74 seconds and its first matched prediction has cosine 0.9999915 and
relative L2 0.004273 against the saved PyTorch reference. This single-step
result was followed by a complete 1024x1024/40-step run from the same saved
PyTorch initial latents and prompt embedding. All 40 free-running checkpoints
were finite and passed the 0.99996 cosine gate: minimum cosine 0.99999049 and
maximum relative L2 0.0043614, both at the final step. The denoiser took
1375.75 seconds (22m56s) and peaked at 14,102,656 KiB host RSS. Native ROCm
VAE decoding took another 19.74 seconds and yielded a coherent 1024x1024
apple image. Compared with the saved PyTorch reference PNG, the decoded RGB
has cosine 0.99999435 and mean absolute channel error 0.373/255. Artifacts
are under `tmp/qimg21-rdna4-1024-40/`.

This acceptance run uses the saved reference text embedding, so it verifies
the ROCm denoiser and VAE, not the full native text-encoder accuracy. Native
text-to-image and image-editing orchestration have separate 256x256 one-step
execution smoke tests. Both also completed 40-step 256x256 runs using native
text conditioning: text-to-image denoising took 68.04 seconds and image-editing
denoising took 77.24 seconds. Native VAE decode produced coherent red-apple
images in both cases. The editing run reused the native VAE-encoded condition,
27-block vision features, and 36-layer multimodal text embedding from the
one-step end-to-end run. These are functional and visual checks, not matched
40-step PyTorch editing-parity tests. The ROCm row-INT8 package also executes
through BF16 dequantization and WMMA, but its measured 256x256 one-step latency is 23.12
seconds versus 5.16 seconds for BF16 weights; it is a compatibility path, not
yet a performance optimization.
