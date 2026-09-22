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
  --attention wmma-fused --out native_latents.npy
```

The RDNA4 runner uses WMMA GEMMs by default when the GPU supports them.
`--attention wmma-fused` adds the shared Pixal3D/TRELLIS gfx12 WMMA attention
kernel for image queries and a separate causal text-prefix correction. Editing
is supported experimentally by processing image groups in reverse order and
correcting interleaved causal text rows; the editing default remains scalar.
`--attention wmma` and `--attention math` use WMMA GEMMs with scalar attention.
The runner covers the 32-layer
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
Both text-to-image and image-editing paths passed initial 256x256 one-step
smoke tests. The scalar path's matched low-timestep 256x256 denoiser prediction
has cosine about 0.99995 against the saved PyTorch reference, below the CUDA
runner's 0.99996 regression gate.

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

With `--attention wmma-fused`, the same 1024x1024/40-step denoiser run took
238.50 seconds (3m59s) and peaked at 14,103,924 KiB host RSS. Every saved
checkpoint is finite and passes the 0.99996 cosine gate; minimum cosine is
0.99998877 and maximum relative L2 is 0.004739 at the final step. Native VAE
decode took 17.79 seconds. The resulting RGB image has cosine 0.99999437 and
mean absolute channel error 0.385/255 versus the saved PyTorch reference PNG.
This is faster than the repository's 5060 Ti W8A8 transformer measurement of
8m12s, though the AMD run uses BF16 weights and a different attention kernel.
Artifacts are under `tmp/qimg21-rdna4-fused-1024-40/`.

The public `native_generate.py --backend rocm` path now selects fused WMMA
attention for text-to-image and scalar mask-aware attention for editing. A
standalone 1024x1024/40-step generation with native text encoding, NumPy
seed-42 noise, fused denoising, native VAE decode, and PNG output completed in
265.88 seconds (4m26s), peak host RSS 14,103,212 KiB, on the RX 9070 XT.
The resulting image is a coherent red apple on a white table; it is not
pixel-comparable to CUDA seed 42 because the initial-noise RNG differs.
Artifacts are under `tmp/qimg21-rdna4-fused-e2e-1024-40/`.

For editing, pass `--native-attention wmma-fused` explicitly to the standalone
generator. On a two-step 256x256 target with a 1024-condition image, this took
16.89 seconds versus 30.47 seconds for scalar ROCm attention. Final latent
cosine versus a saved CUDA native run was 0.99992565 (scalar ROCm: 0.99990237),
below the strict 0.99996 gate. A 40-step 256x256 fused edit produced a visually
similar apple to scalar editing, but final latent cosine between the two was
0.99680; therefore fused editing is opt-in pending a matched PyTorch parity
run. It is not part of the validated text-to-image speed claim.

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
