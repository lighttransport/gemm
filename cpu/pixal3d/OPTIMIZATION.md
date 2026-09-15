# Pixal3D resident GPU validation

Scope: Pixal3D main release, BF16 flows, FP16 decoders, FP32 conditioning and
boundary layers. CPU geometry and PBR processing are unchanged. Enable with
`--gpu-execution resident --gpu-kernels auto`; legacy remains the default.

## Implementation

- A private versioned plugin interface owns device tensors, a bounded reusable
  allocation pool, packed weight caches, and a device stream. Model changes
  release cached weights and unused scratch. Postprocessing releases the pool.
- Flow normalization, modulation, QKV splitting, RMS, RoPE, residuals and MLP
  activations remain on the GPU. Five-key global attention uses a dedicated
  kernel. Euler/CFG updates retain the CPU reference ordering.
- CUDA self-attention reuses `cuda/fa2` with BF16/FP16 specializations. ROCm BF16
  self-attention reuses TRELLIS.2's gfx12 WMMA BC32 kernel, adapting probability
  conversion to round-to-nearest-even. Both handle partial tiles. Build adapters
  extract the existing source strings, avoiding a second maintained kernel copy.
- Explicit CUDA MMA and gfx12 WMMA GEMM kernels handle BF16/FP16 tails. They are
  opt-in through `--gpu-kernels mma`; automatic dispatch retains faster vendor
  GEMM. FP32 operations do not use the low-precision attention kernels.
- Dense and sparse decoders use tiled device neighborhood gathering and GEMM.
  Subdivision indices and threshold decisions stay on the CPU. NAF guide
  encoding and sampling share device features. Identical DINO inputs are cached
  within a generation. Flow coordinates, rounded positive/negative conditioning,
  RoPE phases, and per-block global K/V are also cached, with exact value-based
  invalidation.

Weights use their compute storage type. Activations currently use F32 storage
with BF16/FP16 rounding at model boundaries. Packed activation storage and
additional GEMM tiling remain opportunities for further optimization.

## Matched flow benchmark

Hardware: RTX 5060 Ti (`sm_120`) and RX 9070 XT (`gfx1201`). Host GCC 13.3;
CUDA 13.2.86; HIP 7.15.26333 / AMD clang 23.0.0git. Tests use the existing
per-backend uv/PyTorch environments under `ref/pixal3d/`.

One complete 30-block shape1024 prediction, **10,599 identical recorded tokens**,
16 host threads. Each mode owns one persistent engine/model; one cold call is
followed by three warm calls. Modes run sequentially on each GPU. Times are
warm medians, without per-command profiling.

| GPU | Legacy | Resident auto | Speedup | Output NRMSE vs legacy | Cosine |
|---|---:|---:|---:|---:|---:|
| RTX 5060 Ti | 41.7449 s | 2.46750 s | 16.92x | 0.004590 | 0.9999895 |
| RX 9070 XT | 52.1687 s | 1.96032 s | 26.61x | 0.004550 | 0.9999896 |

These are flow-pass speedups, not total generation or aggregate NN-stage
speedups. Conditioning, decoders, CPU postprocessing, model loading and GLB
serialization are excluded. No 5x total-generation claim is made.

Reproduce each row with the same recorded dumps:

```sh
for mode in legacy resident; do
  ref/pixal3d/run.sh cuda ref/pixal3d/benchmark_flow_block.py \
    --backend cuda --dump-dir tmp/pixal3d/runs/cuda-house/dumps \
    --blocks 30 --gpu-execution "$mode" --gpu-kernels auto \
    --output "tmp/pixal3d/cuda-$mode.safetensors"
done
```

Use `rocm` for both backend arguments to measure AMD. The benchmark writes exact
load/cold/warm timings beside its output. Recorded results from this work are
`tmp/pixal3d/final-{cuda,rocm}-legacy.{json,safetensors}` and
`tmp/pixal3d/phase-{cuda,rocm}-auto.{json,safetensors}`. The latter also passed
`--check-cache` alternating-guidance and reused-host-buffer checks. Cached RoPE
outputs matched the original calculation exactly in all three precisions on
both GPUs; complete flow predictions also matched the pre-phase-cache output
bit-for-bit.

## Correctness and generation checks

The private kernel validator compares F32/BF16/FP16 GEMM and attention against
PyTorch, including non-square/tail shapes, five-key attention, oversized
allocation rejection, repeated pool reuse and recovery after invalid commands.
Checkpoint tests cover 24 DINO blocks, NAF guide/projected features, and guided
shape/texture subdivisions. Decoder acceptance is NRMSE <= 0.02 and cosine
>= 0.999; guided coordinates must match exactly. Changed subdivision decisions
must remain within 0.02 of the upstream zero threshold.

Five complete 1024-cascade generations (12 steps per stage, 4096 PBR textures)
passed GLB bounds, normals, mesh indices, nonzero triangle areas and texture
checks. Four-view renders were inspected, including the jester's detached
accessory geometry; numerical checks do not guarantee artifact-free geometry.

| GPU / image / seed | Full invocation | Native reserved peak | Sampled VRAM |
|---|---:|---:|---:|
| CUDA / house / 1 | 413.52 s | 13.29 GiB | 13.83 GiB |
| ROCm / house / 1 | 386.27 s | 11.59 GiB | 11.94 GiB |
| CUDA / crab / 2 | 568.72 s | 13.08 GiB | 12.66 GiB |
| CUDA / jester / 3 | 286.36 s | 8.83 GiB | 8.99 GiB |
| ROCm / crab / 2 | 556.40 s | 13.79 GiB | 14.20 GiB |

All completed on 16 GB cards using a 14 GiB native allocation budget. CUDA
sampling measures process memory; ROCm sampling measures device usage including
runtime allocations. Sampling can miss short peaks. Full invocation timings are
functional observations, with some other work overlapping CPU postprocessing;
they are not isolated performance comparisons.

The first CUDA house run preceded stage-boundary pool release. A concurrent
PyTorch load exhausted remaining VRAM during CPU postprocessing; that reference
attempt passed when rerun after generation. The jester and ROCm crab include the
conditioning/global-KV cache. All five precede the final RoPE-phase cache, whose
exactness was checked separately at kernel and complete-flow level.

Final diffusion-step checks replay the complete upstream sampler step on the
native step-11 state (30 blocks, actual conditioning), rather than comparing
unrelated trajectories:

| Backend / image | Structure NRMSE | Shape512 NRMSE | Shape1024 NRMSE | Texture NRMSE |
|---|---:|---:|---:|---:|
| CUDA / house | 0.007147 | 0.002486 | 0.002606 | 0.002161 |
| ROCm / house | 0.007120 | 0.002446 | 0.002547 | 0.002331 |
| CUDA / jester | 0.007454 | 0.002842 | 0.002952 | 0.002493 |
| ROCm / crab | 0.006791 | 0.002954 | 0.003090 | 0.002367 |

All listed checks passed the existing step tolerance. Raw logs, profiles,
meshes and memory traces are under `tmp/pixal3d/resident-runs/`. The initial
ROCm diagnostic run was stopped after a faster WMMA adapter passed validation;
it is explicitly kept as `rocm-house-initial-diagnostic`, not a completed run.

```sh
make -C cpu/pixal3d -j4 all validation test
make -C cuda/pixal3d
make -C rdna4/pixal3d
ref/pixal3d/run.sh cuda ref/pixal3d/validate_resident.py --backend cuda
ref/pixal3d/run.sh rocm ref/pixal3d/validate_resident.py --backend rocm
ref/pixal3d/run.sh cuda ref/pixal3d/validate_flow_stage.py --backend cuda \
  --dump-dir tmp/pixal3d/resident-runs/cuda-house/dumps --stage texture --step 12
ref/pixal3d/run.sh cpu ref/pixal3d/validate_glb.py \
  tmp/pixal3d/resident-runs/cuda-house/mesh.glb
```

The Python demo's health endpoint and GPU controls were smoke-tested with
`--gpu-execution resident --gpu-kernels auto`. Profile data is returned in the
inference response. The original options/result C ABI layouts remain intact;
GPU selection uses the separate versioned configuration API.
