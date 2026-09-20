# Pixal3D resident GPU validation

Scope: Pixal3D main release, BF16 flows, FP16 decoders, FP32 conditioning and
boundary layers. CPU geometry and PBR processing are unchanged. GPU CLI runs
default to `--gpu-execution resident --gpu-kernels auto`; CPU runs retain
legacy execution. Mixed flow keeps state,
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

Postprocessing replays can emit the same subphase profile without repeating
diffusion. On the saved CUDA jester decoder outputs, the original 4096-texture
path took 121.1 seconds: simplification 44.2 s, inpainting 23.8 s,
hole filling plus original-mesh BVH construction 17.7 s, unwrap/normals
16.7 s, remeshing 7.6 s, baking 3.0 s, FDG extraction 0.9 s, and rasterization
0.3 s. Replacing the simplifier's per-vertex vectors with compact CSR face
adjacency reduced simplification to 27.2 s and the replay to 101.9 s (15.9%
end-to-end). Every collapse-round face count matched, the 93.8 MB GLB remained
byte-identical (`6d8c267b...11a8b7`), and the GLB validator passed. Texture
baking is already a small part of total postprocessing. Running the independent
base-color and packed material Telea solves concurrently reduced the measured
inpaint phase from 21.6 s to 17.2 s on the same CSR build; output remained
byte-identical. Boundary-loop discovery now stores its only relevant case,
two sorted neighbors per vertex, inline and uses byte visitation instead of
per-vertex vectors plus a hash set. For the 2.9-million-vertex extracted mesh,
adjacency container storage falls from about 70 MB plus heap allocations to
about 26 MB. A full replay completed in 92.6 s and retained the byte-identical
`6d8c267b...11a8b7` GLB; the combined hole-fill/BVH phase measured 17.8 s.
Reproduce with:

```sh
ref/pixal3d/run.sh cpu ref/pixal3d/replay_postprocess.py \
  --dump-dir tmp/pixal3d/resident-runs/cuda-jester/dumps \
  --output tmp/pixal3d/postprocess-profile/jester.glb \
  --profile-json tmp/pixal3d/postprocess-profile/jester.json
```

The explicit lower-output preset (`--texture-size 1024
--triangle-target 500000`) completed the same replay in 65.2 s and produced a
23 MB GLB with 484,580 triangles, versus 90 MB and 970,090 triangles for the
default. Both 1K and 4K outputs pass mesh, normal, texture, and material
validation. This option changes output fidelity and remains opt-in; 4K/1M is
the default used for reference-quality comparisons.

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

With the final unchanged 2048-row GEMM schedule, a controlled RTX 5060 Ti run
of the same 30-block, 10,765-token Shape-1024 prediction measured:

| Native budget | Warm median | Relative to 7 GiB | Output SHA-256 |
|---:|---:|---:|---|
| 7168 MiB (8 GB card path) | 3.542 s | baseline | `b510f666...e722387` |
| 12288 MiB (default) | 3.454 s | 2.5% faster | `b510f666...e722387` |
| 14336 MiB (maximum) | 3.467 s | 2.1% faster | `b510f666...e722387` |

All output tensors were byte-identical. The 12 GiB default is the measured
sweet spot: it retains more packed weights than the 8 GB path without a gain
from raising the cap to 14 GiB. These are requested native allocation budgets;
the plugin still clamps them against current free device memory. Reproduce the
matrix with:

```sh
ref/pixal3d/run.sh cuda ref/pixal3d/validate_budget_matrix.py \
  --dump-dir tmp/pixal3d/resident-runs/cuda-house/dumps
```

A tested
4096-row schedule was removed: its microbenchmark gain did not survive the
full dense multiview run because differently sized workspaces reused poorly.

Complete four-view generation was also validated at the 8 GB-card and 12 GiB
target profiles on the RTX 5060 Ti. Both runs used the pinned four-view example,
seed 42, mixed precision, 4096 textures and the one-million-triangle target.

| Native budget | Native generation | Peak native reserved | Sampled process VRAM | Output SHA-256 |
|---:|---:|---:|---:|---|
| 7168 MiB | 506.207 s | 7160.6 MiB | 7364 MiB | `a5a22a90...c700383` |
| 12288 MiB | 484.915 s | 8488.0 MiB | 8706 MiB | `a5a22a90...c700383` |

The outputs are byte-identical: 655,071 vertices, 961,142 triangles, valid
normalized normals and 4096 PBR textures. The 12 GiB profile is 4.2% faster
while using 1.33 GiB more native reserved memory. The constrained path leaves
about 840 MiB between its sampled 7.19 GiB process peak and an 8 GiB device.
Reproduce and compare existing run directories with:

```sh
ref/pixal3d/run.sh cpu ref/pixal3d/validate_multiview_budgets.py \
  tmp/pixal3d/mv-budget-7168 tmp/pixal3d/mv-budget-12288
```

A bounded post-run reliability soak repeated the resident allocator/kernel
suite twice, then ran matched mixed and FP32 structure checks, native invalid
input recovery, web queue/cancellation tests, and multiview artifact validation.
Both matched flow checks reported NRMSE `1.3173e-6`; total device usage returned
from 3,485,532,160 bytes before the soak to 3,485,401,088 bytes afterward. The
two PyTorch comparisons peaked at 13.82 GB total device use and about 13.0 GB
host RSS. The complete machine-readable record is emitted by
`validate_cuda_reliability.py` rather than checked into the repository.

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

The modes are exposed by the CLI as `--gpu-flow-precision bf16|mixed|fp32`;
`mixed` is the GPU default and provides the recommended quality/speed tradeoff.
Reproduce the isolated experiment with
`validate_flow_precision.py --stage structure --precision bf16|mixed|fp32`. Full
twelve-step structure sampling from identical native noise and conditioning
was also compared to the matching FP32 PyTorch sampler on the RTX 5060 Ti.
All four cascade trajectories pass the `<0.001` NRMSE target without switching
their BF16 GEMMs or self-attention to FP32:

| Stage | Tokens | Mixed NRMSE | Cosine | Max abs |
|---|---:|---:|---:|---:|
| Structure | 4096 | 0.0000711 | 0.9999999975 | 0.001858 |
| Shape-512 | 2423 | 0.00000945 | 0.99999999996 | 0.000231 |
| Shape-1024 | 10765 | 0.0000247 | 0.99999999970 | 0.001926 |
| Texture | 10765 | 0.00000109 | 0.999999999999 | 0.0000258 |

Texture includes the saved Shape-1024 latent as its concatenated condition, so
the check also covers that cross-stage input. Reproduce one stage or iterate all
four with:

```sh
for stage in structure shape512 shape1024 texture; do
  ref/pixal3d/run.sh cuda ref/pixal3d/validate_mixed_trajectory.py \
    --dump-dir tmp/pixal3d/resident-runs/cuda-house/dumps --stage "$stage"
done
```

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
