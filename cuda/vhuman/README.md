# Virtual-human face deformer (CUDA)

GPU backend of `ryzen/vhuman_deformer.h` for the rig packages written by
`server/vhuman/rig` (`<head>/rig/rig_deformer.safetensors`). The host
evaluates the rig per frame (`vh_deformer_prepare`: controls, correctives,
ML MLP, skinning matrices). The device holds the rest shape, all morphs and the
skin weights, and one fused kernel per batch blends the morphs and applies
4-influence LBS. The morph blend is register-tiled over 8 frames, so the
morph matrix is read once per tile. The driver API goes through cuew and
kernels are NVRTC-compiled at run time (no nvcc).

```sh
make -C cuda/vhuman
./cuda/vhuman/bench_vhuman_deformer_cuda <head>/rig/rig_deformer.safetensors 1024
```

A second kernel (`vh_contacts`, one block per frame) applies the exact
post-skinning contact projection (`server/vhuman/rig/contacts.py`). It is on by
default when the package has contact tensors.

Idle RTX 5060 Ti, reference head (12.6k vertices, 149 morphs), 1024 frames:
2.2 µs/frame deform, 6.7 µs/frame with exact contacts. Host preparation (MLP as
two `sgemm_avx2` calls) takes 4.5 µs/frame, and downloading 50 MB takes ~22 ms. The results equal the CPU deformer
(`test_rig`), and the downloaded positions are 50 MB per 1024 frames. The
Python wrapper is `server/vhuman/rig/native.py` (`NativeGPU`).

## Native image-model training

`libvhuman_training_cuda.so` trains CueNet v3 and trace-v1 Gaussian appearance
using the CUDA driver, NVRTC and repository FP32 tiled GEMM. It has no Torch,
ONNX, cuBLAS or gsplat dependency. Cue convolutions use im2col plus GEMM for
forward and both gradients; SiLU, bilinear resizing, residual normal/mask losses
and AdamW run on CUDA. Gaussian geometry/projection uses four dual derivatives,
and the rasterizer replays the accepted pixel prefix backwards for analytic
gradients. RGB, opacity, scales, offset, color basis and expression matrix all
train. FP64 local math follows the independent native CPU reference. Depth sort
and 16x16 tile bins remain on the host; tile overlaps are capped at eight million.

Parameters, gradients and optimizer moments persist on CUDA. CLI training uses
resident parameters and skips image/gradient downloads for each update. Device
buffer growth has an explicit `--memory-mb` budget (512 MiB default); this excludes
driver, compiled module and context allocations. Shape changes reuse buffers.
Python defaults synchronize public NumPy parameters after each step. With
`resident=True`, call `sync_parameters()` before reading/modifying those arrays
and `upload_parameters()` after editing them; exports synchronize automatically.
Changing optimizer learning rate/decay applies to subsequent GPU updates.
`load_avatar()` accepts the same trace-v1 diagonal binding/control order as a warm
start, with fresh optimizer moments. Call `close()` to release a trainer.

```sh
make -C cpu/vhuman libvhuman_training.so
make -C cuda/vhuman libvhuman_training_cuda.so libvhuman_splat.so libvhuman_runtime.so
make -C cuda/vhuman test-training
python -m server.vhuman.test_native_gpu_training --gpu -v
python -m server.vhuman.realtime fit-appearance --manifest appearance.json \
  --output tmp/vhuman-realtime/gpu-avatar.npz --count 50000 --steps 1000 \
  --device cuda:0 --memory-mb 512
python -m server.vhuman.reconstruction.learned_cues train --dataset synthetic.npz \
  --out tmp/vhuman-gpu-cues --steps 400 --device cuda:0 --memory-mb 512
# Brief timing/export/visual checks against local cleared references:
python ref/vhuman/validate_gpu_image_training.py \
  --output tmp/vhuman-gpu-validation --corpus appearance.json \
  --avatar fitted-avatar.npz --cue-dataset synthetic.npz \
  --cue-checkpoint fitted-cues --steps 1 --count 50000 --side 256
```

On the RTX 5060 Ti with another process reporting 100% utilization, two cold
50k-splat 256x256 updates took 41.1/48.0 ms; a separate matched run took 50.8 ms
versus 472.8 ms for the native CPU step. Cold CPU/GPU initial RGBA matched exactly.
The Gaussian trainer tracked 43.7 MiB of buffers; eight 64x64 cue images tracked
51.7 MiB and took 15.3–18.2 ms/update. These are brief synchronized wall timings,
with compilation excluded, not isolated performance benchmarks. Six GPU tests
passed in 3.3 seconds, including all GEMM transpose/tail cases, optimizer changes,
gradient/clipping cases, budget rejection and full framework-blocked train/export.

Visual receipts are in `tmp/vhuman-gpu-training/validation-{01,02}/`. Two cold
updates reduced RGB L1 from 0.02555 to 0.02323 but left visible coverage holes.
A warm-started neutral fit preserved appearance and improved L1 slightly. Its
export/runtime mean RGBA error was 2.03e-7, with maximum 0.00720 and three of
65,536 pixels exceeding 0.001. Sparse FP64-training/FP32-inference differences
remain; strict full-frame parity is not established. Two cue warm-start updates
worsened held-out error from 10.89 to 11.13 degrees. Cues remain disabled by
default, and real geometry/expression quality, convergence, and larger-resolution
performance still need validation.
