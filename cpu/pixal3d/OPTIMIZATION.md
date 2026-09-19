# Pixal3D resident GPU validation

Scope: Pixal3D main release, BF16 flows, FP16 decoders, FP32 conditioning and
boundary layers. CPU geometry and PBR processing are unchanged. Enable with
`--gpu-execution resident --gpu-kernels auto`; legacy remains the default.
Resident flow uses BF16 by default. `--gpu-flow-precision mixed` keeps state,
normalization and residual math in FP32 while retaining BF16 GEMMs and
attention; this is the recommended accuracy/performance point. `fp32` keeps
the complete flow stack in FP32 for diagnostic runs. Decoder precision and
conditioning boundaries are unchanged.

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
  GEMM. Mixed mode intentionally uses the low-precision attention kernels while
  keeping surrounding state and sensitive pointwise operations in FP32.
- Dense and sparse decoders use tiled device neighborhood gathering and GEMM.
  Subdivision indices and threshold decisions stay on the CPU. NAF guide
  encoding and sampling share device features. Identical DINO inputs are cached
  within a generation. Flow coordinates, rounded positive/negative conditioning,
  RoPE phases, and per-block global K/V are also cached, with exact value-based
  invalidation.

Weights use their compute storage type. Flow activations use F32 storage with
BF16/FP16 rounding at model boundaries. Sparse decoder gathers are written
directly in FP16/BF16 and consumed by GEMM without an intermediate F32 tensor
or a second conversion allocation. More packed flow activation storage and
additional GEMM tiling remain opportunities for further optimization.

## 8 GB memory profile and exact postprocessing

The allocator reports its effective budget, active and pooled bytes, peak
active bytes, and largest allocation through private device ABI version 2.
The scratch pool is capped at the smaller of one eighth of the device budget
and 768 MiB; stage trims now return all pooled storage. The host weight cache
uses the plugin's effective budget after the runtime accounts for free device
memory, rather than the requested CLI limit. This also makes a contended run
fail at the native budget boundary instead of relying on a driver OOM.

On the RTX 5060 Ti, the complete house fixture in mixed precision succeeds
with `--vram-budget-mib 7168`, leaving 1 GiB of an 8 GiB card outside the
native allocator. It used 6.48 GiB peak active storage, 7.00 GiB peak reserved
storage, and 6.88 GiB sampled process VRAM. The largest single allocation was
1.85 GiB. The output SHA-256 is
`e4ff6b24d5578aaa4f91334208450c76e48ce8225c0bca24837aafcc19858b35`,
identical to the earlier mixed-precision reference.

CPU mesh processing keeps exact ordering while replacing tree-based edge
counting with sorted vectors and deterministic parallel sorts/reductions.
The three radius-one material channels share one inpaint call. On the same
fixture postprocessing fell from about 230 s to 175.3 s with the exact same
GLB bytes; the complete invocation took 369.6 s. Per-command GPU events are
disabled during normal profiling because they serialize the stream. Set the
diagnostic environment variable `PIXAL3D_PROFILE_COMMANDS=1` when individual
resident command timing is required.

Reproduce the constrained full run with:

```sh
ref/pixal3d/run.sh cuda ref/pixal3d/run_fixture.py \
  --backend cuda --input ref/pixal3d/upstream/assets/images/1_img.png \
  --output-dir tmp/pixal3d/cuda-house-7168 --fov 0.857556 --seed 1 \
  --threads 8 --gpu-execution resident --gpu-kernels auto \
  --gpu-flow-precision mixed --vram-budget-mib 7168 --timeout 1200
```

Multiview conditioning caches DINO outputs per view and resolution for one
generation. Shape-512 reuses the structure stage's 512 features, and texture
reuses shape-1024's 1024 features. On the four-view fixture this removed
9.86 GB of host-to-device traffic and about 5 seconds of duplicate
conditioning. Cached features live in host memory, so the optimization does
not raise the 7 GiB native GPU floor. The web demo requests a 12 GiB budget for
additional dense-output headroom and automatically falls back to available
VRAM minus the runtime reserve.

With the final unchanged 2048-row GEMM schedule, a 12 GiB budget reduced the
warm 30-block, 10,765-token shape flow median from 3.279 s to 3.141 s (4.2%)
by retaining more packed weights. Outputs were bit-identical. A tested
4096-row schedule was removed: its microbenchmark gain did not survive the
full dense multiview run because differently sized workspaces reused poorly.

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

## FP32 flow accuracy experiment

On the recorded 4096-token structure input, a complete 30-block flow stack was
compared with the matching PyTorch model at `t=1`. FP32 reduced NRMSE by about
6,600x on both cards while keeping cosine error below 1e-12:

| GPU | BF16 NRMSE | FP32 NRMSE | FP32 max abs |
|---|---:|---:|---:|
| RTX 5060 Ti | 0.008680 | 0.00000131 | 0.0000205 |
| RX 9070 XT | 0.006139 | 0.000000929 | 0.0000124 |

The modes are exposed by the CLI as `--gpu-flow-precision bf16|mixed|fp32`; BF16 remains
the default for memory and throughput, with `mixed` providing the recommended
quality/speed tradeoff. Reproduce the isolated experiment with
`validate_flow_precision.py --stage structure --precision bf16|mixed|fp32`. Full
twelve-step cascade accuracy and VRAM cost should be measured before making
FP32 the default.

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
