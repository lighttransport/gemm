# Pixal3D resident GPU validation

Scope: Pixal3D main release, BF16 flows, FP16 decoders, FP32 conditioning and
boundary layers. CPU geometry and PBR processing are unchanged. GPU CLI runs
default to `--gpu-execution resident --gpu-kernels auto`; CPU runs retain
legacy execution. Mixed flow keeps state,
normalization and residual math in FP32 while retaining BF16 GEMMs and
attention; this is the recommended accuracy/performance point. `fp32` keeps
the complete flow stack in FP32 for diagnostic runs. Decoder precision and
conditioning boundaries are unchanged.

Timing terms are consistent throughout this file. A **generation** time is the
native profile's `generate` timer and excludes final GLB serialization. A
**full invocation** or **wall time** includes serialization. A **flow** time is
one recorded neural stage and excludes conditioning, decoders, postprocessing,
model loading and serialization. A **postprocessing replay** starts from saved
decoder tensors and its wall time includes GLB serialization. Concurrency is
called out for every performance table; historical functional observations
under contention are not treated as isolated benchmarks.

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
or a second conversion allocation. Flow activation packing and larger GEMM
tiles have been evaluated at full-stage and complete-generation scale; the
current F32 activation storage and 2048-row schedule are retained below.

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
single-view house fixture, an earlier resident mixed run under desktop load
reduced the native postprocessing phase from about 230 s to 175.3 s with the
exact same GLB bytes; wall time for the full invocation, including
serialization, was 369.6 s. Per-command GPU events are
disabled during normal profiling because they serialize the stream. Set the
diagnostic environment variable `PIXAL3D_PROFILE_COMMANDS=1` when individual
resident command timing is required.

Postprocessing replays can emit the same subphase profile without repeating
diffusion. All replay wall times in this section use the saved CUDA jester
decoder tensors, eight host threads, the 4K/1M defaults, and include GLB
serialization; phase times exclude serialization. Under variable desktop host
load, the original path took 121.1 seconds: simplification 44.2 s, inpainting 23.8 s,
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

Detailed profiling separated the remaining combined phases. On the same saved
4K/1M fixture, hole filling used 0.37 s while original-mesh SBVH construction
used 17.71 s; UV unwrap used 16.87 s while normal generation and remapping used
0.036 s; the two Telea solves used 14.68 s while material interleave and
deinterleave used 0.23 s. Stable vector grouping now replaces the UV merge
loop's per-entry `std::map` allocation while preserving each key's original
floating-point accumulation order. Measured unwrap time fell to 14.48–14.61 s
(13–14%), and two complete optimized observations were 92.93 s and 95.91 s
under variable host load. Both retained the exact `6d8c267b...11a8b7` GLB.

Texture baking now writes the packed material image directly, removing 48 MiB
of temporary planar material arrays and both repacking passes. Closest-point
queries use the SBVH-owned triangle array instead of retaining a duplicate,
removing 290.1 MiB after construction; an rvalue SBVH build also avoids that
duplicate at construction peak. The final replay's measured maximum RSS was
3,169,464 KiB. A parallel regular-BVH experiment reduced construction from
17.71 s to 2.44 s and total replay time to 80.05 s, but changed the GLB hash to
`06ad7053...15a0`; it was rejected and removed because closest-point tie order
must remain byte-identical.

A final follow-up retested two deterministic parallelization candidates against
the same saved decoder tensors. GNU parallel stable sort in UV grouping retained
the established SHA-256, but reduced unwrap from 14.94 s to only 14.24 s and the
complete replay from 95.58 s to 93.49 s (2.2%). Parallel face-plane setup plus
preallocated simplifier vectors regressed the complete replay to 98.18 s. Both
experiments were removed: neither met the retention rule of at least 5% complete
replay improvement, 15% target-phase improvement without a total regression, or
10% peak-RSS reduction with no more than 2% slowdown. The restored source replay
completed in 94.50 s, used 3,172,004 KiB maximum RSS, and reproduced SHA-256
`6d8c267b...11a8b7`. No additional postprocessing change was retained.

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

### CPU fallback

The CPU linear path no longer copies and rounds immutable weights when their
safetensors storage already matches the requested BF16 or FP16 compute type.
Converted BF16/F16 values are exactly representable in F32, so the removed
round is a no-op. Output bias addition and precision rounding now use the
existing runtime-dispatched AVX2 implementation with row-parallel OpenMP, and
host attention reuses one score allocation across heads and query tiles.

On the recorded CUDA-house shape-1024 fixture, 256 tokens, all 30 flow blocks,
eight threads, and two or three warm observations, the prior binary's median
was 6.8984 s and the optimized binary's median was 6.0341 s, a 12.5% reduction.
The complete output tensors were byte-identical. The corresponding three-block
median fell from 0.7233 s to 0.6462 s (10.7%). This path creates no persistent
rounded-weight cache and therefore adds no retained model memory.

Reproduce the all-block comparison with:

```sh
ref/pixal3d/run.sh cpu ref/pixal3d/benchmark_flow_block.py \
  --backend cpu --dump-dir tmp/pixal3d/resident-runs/cuda-house/dumps \
  --stage shape1024 --blocks 30 --repeats 3 --tokens 256 --threads 8 \
  --output tmp/pixal3d/cpu-opt/optimized-simd-30.safetensors
```

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
9.86 GB of host-to-device traffic and about 5 seconds of duplicate conditioning
inside the resident generation profile; this component delta excludes GLB
serialization and was observed under desktop load. Cached features live in host
memory, so the optimization does not raise the 7 GiB native GPU floor. The web
demo requests a 12 GiB budget for additional dense-output headroom and
automatically falls back to available VRAM minus the runtime reserve.

With the final unchanged 2048-row GEMM schedule, sequential RTX 5060 Ti runs
of the same 30-block, 10,765-token Shape-1024 flow pass measured the following
warm medians after one cold pass. These stage-only timings exclude conditioning,
decoders, postprocessing, model loading and serialization:

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

A tested 4096-row schedule was removed: its microbenchmark gain did not survive the
full dense multiview run because differently sized workspaces reused poorly.

Packed flow storage was evaluated on the same Shape-1024 fixture. Packing every
transient GEMM input preserved the exact `b510f666...e722387` tensor, but raised
the warm median from 3.354 s to 3.711 s, raised allocations from 908 to 2,337,
and raised peak active storage from 3,528,072,016 to 3,582,690,128 bytes. A
narrower same-binary A/B packed only cached projected/global conditioning. It
reduced active storage by 42.1 MiB, but raised the reserved peak by 34.0 MiB and
moved the five-run warm median from 3.468 s to 3.542 s. Both experiments were
removed because the 7 GiB path is bounded by reserved memory and neither
improved full-stage time. The runs were made with desktop GPU processes present;
the allocator deltas, exact output and rejection decisions are retained, while
the controlled table above remains the performance reference.

`benchmark_flow_block.py --profile-json FILE` records aggregate allocator and
kernel counters for this comparison. The local result bundle is
`tmp/pixal3d/packed-activation-eval/summary.json`; the checked result record in
`ref/pixal3d/validation-results.json` contains its hash and the rejection data.
Fresh twelve-step mixed comparisons against the pinned FP32 PyTorch models
reported NRMSE `7.10869e-5`, `9.44953e-6`, `2.46986e-5`, and `1.09262e-6` for
Structure, Shape-512, Shape-1024, and Texture respectively. All remain below
`0.001`.

Complete four-view generation was also validated at the 8 GB-card and 12 GiB
target profiles on the RTX 5060 Ti. Both resident mixed runs used the pinned
four-view example, seed 42, 4096 textures and the one-million-triangle target.
The native generation column excludes final GLB serialization. Observed wall
times including serialization were 512.737 s and 490.727 s respectively. Other
desktop GPU/CPU work was present, so the timings are functional observations.

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
twelve-step sampling from identical native noise and conditioning was also
compared to the matching FP32 PyTorch sampler on the RTX 5060 Ti.

Until 2026-09-25 `validate_mixed_trajectory.py` passed a zero flow-request
flag, which selects the FP32 flow, so the table published here as "mixed"
measured FP32 mode. Those FP32 numbers are unchanged and remain below the
`<0.001` target:

| Stage | Tokens | FP32 NRMSE | Cosine | Max abs |
|---|---:|---:|---:|---:|
| Structure | 4096 | 0.0000711 | 0.9999999975 | 0.001858 |
| Shape-512 | 2423 | 0.00000945 | 0.99999999996 | 0.000231 |
| Shape-1024 | 10765 | 0.0000247 | 0.99999999970 | 0.001926 |
| Texture | 10765 | 0.00000109 | 0.999999999999 | 0.0000258 |

The validator now runs the configured mode (`--precision mixed|fp32|bf16`,
default mixed). Mixed mode drifts further over twelve CFG-7.5 steps. Running
the resident flow's `input_layer`, `adaLN_modulation.1` and `out_layer` in FP32
(upstream and the CPU path already do; they had been BF16 GEMMs) reduces the
drift on every stage at negligible cost:

| Stage | Mixed NRMSE, BF16 boundaries | Mixed NRMSE, FP32 boundaries | Cosine |
|---|---:|---:|---:|
| Structure | 0.1052 | 0.0649 | 0.99790 |
| Shape-512 | 0.0244 | 0.0223 | 0.99975 |
| Shape-1024 | 0.0510 | 0.0471 | 0.99889 |
| Texture | 0.0123 | 0.0028 | 0.999996 |

The mixed gate is NRMSE `<0.08` and cosine `>0.997`: a regression guard, not a
parity claim.

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

Five earlier resident-auto 1024-cascade generations (12 steps per stage, 4096
PBR textures) passed GLB bounds, normals, mesh indices, nonzero triangle areas
and texture checks. Their full-invocation wall times include GLB serialization.
Other validation work overlapped CPU postprocessing, so they are retained as
functional coverage rather than current performance baselines. Four-view
renders were inspected, including the jester's detached accessory geometry;
numerical checks do not guarantee artifact-free geometry.

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

## RX 9070 XT and RTX 5060 Ti resident reruns (2026-09-24)

With the same house and four-view example fixtures, seeds, mixed flow
precision, and resident automatic kernels, rerun with the ROCm 12,288 MiB and
CUDA 7,168 MiB budgets:

```sh
ref/pixal3d/run.sh rocm ref/pixal3d/run_fixture.py --backend rocm \
  --input ref/pixal3d/upstream/assets/images/1_img.png \
  --output-dir tmp/pixal3d/revalidate-rocm-house-20260924 --seed 1 \
  --threads 8 --gpu-execution resident --gpu-kernels auto \
  --gpu-flow-precision mixed --vram-budget-mib 12288 \
  --texture-size 1024 --triangle-target 1000000
ref/pixal3d/run.sh rocm ref/pixal3d/run_fixture.py --backend rocm \
  --views-dir ref/pixal3d/upstream/assets/mv_images/example \
  --output-dir tmp/pixal3d/revalidate-rocm-multiview-20260924 --seed 42 \
  --threads 8 --gpu-execution resident --gpu-kernels auto \
  --gpu-flow-precision mixed --vram-budget-mib 12288 \
  --texture-size 4096 --triangle-target 1000000
ref/pixal3d/run.sh cuda ref/pixal3d/run_fixture.py --backend cuda \
  --input ref/pixal3d/upstream/assets/images/1_img.png \
  --output-dir tmp/pixal3d/revalidate-cuda-house-20260924 --seed 1 \
  --threads 8 --gpu-execution resident --gpu-kernels auto \
  --gpu-flow-precision mixed --vram-budget-mib 7168 \
  --texture-size 1024 --triangle-target 1000000
ref/pixal3d/run.sh cuda ref/pixal3d/run_fixture.py --backend cuda \
  --views-dir ref/pixal3d/upstream/assets/mv_images/example \
  --output-dir tmp/pixal3d/revalidate-cuda-multiview-20260924 --seed 42 \
  --threads 8 --gpu-execution resident --gpu-kernels auto \
  --gpu-flow-precision mixed --vram-budget-mib 7168 \
  --texture-size 4096 --triangle-target 1000000
```

| Fixture | Runner wall time | Peak reserved device bytes | Triangles | MMA attentions |
| --- | ---: | ---: | ---: | ---: |
| House single view | 250.177 s | 7,883,505,744 | 936,320 | 2,280 |
| Four-view example | 452.329 s | 8,530,360,804 | 968,682 | 2,280 |

Both GLBs pass `ref/pixal3d/validate_glb.py`: no zero-area faces or zero
normals, valid textures and materials, and the largest welded component owns
97.43% (house) and 96.28% (four-view) of the faces. The rendered previews under
each output directory show the expected detailed house and four-sided head.
Automatic kernels use the gfx12 WMMA attention adapter here; profile counters
show zero explicit MMA GEMMs because automatic GEMM selection retains the
vendor BLAS path. The sampler's total-device figures include other processes;
the table uses the runner's own peak reserved allocation counter.

The house fixture was repeated with identical arguments. ROCm observed
250.177, 231.167, and 262.195 seconds; CUDA observed 446.579 and 468.696
seconds with a 7,168 MiB budget because an unrelated process held about 7.8
GiB on the RTX 5060 Ti. All five GLBs pass the validator. ROCm runs one and
three have byte-identical GLBs (936,320 triangles), while run two is a valid
but different 996,574-triangle mesh. The two CUDA GLBs are byte-identical
(937,762 triangles). Repeated ROCm multiview runs were also byte-identical
(968,682 triangles), at 452.329 and 474.398 seconds.

For a kernel-level comparison, `benchmark_flow_block.py` replayed the same
real house Shape-1024 noise and conditioning from
`tmp/pixal3d/resident-runs/rocm-house/dumps`: 10,667 tokens, 30 blocks, mixed
precision, resident automatic kernels, 7,168 MiB budget, one cold pass and
three warm repeats. Warm medians were 2.0060 seconds on ROCm and 6.1932
seconds on CUDA. The output cosine was 0.999999133. Profile counters recorded
120 MMA attention calls on each backend, with peak reserved device allocations
of 3,865,546,416 and 3,996,622,512 bytes respectively. The CUDA timing was
measured while the independent 7.8 GiB GPU process was active.

As a separate BF16 WMMA diagnostic on ROCm, `--gpu-kernels mma
--gpu-flow-precision bf16` exercised 5,122 explicit MMA GEMMs and 120 MMA
attentions on the same 30-block input. Its output cosine against BF16
automatic vendor GEMM was 0.999985775, with a 7.0144-second warm median
versus 1.9970 seconds for vendor GEMM. The production mixed-precision path
keeps its surrounding GEMMs in FP32; forcing `mma` there records zero MMA
GEMMs and does not change its 2.00-second median.

The matched CUDA four-view rerun used the same example images, seed 42, mixed
precision, 4096 texture, and triangle target, with the 7,168 MiB budget
required by the occupied 5060 Ti. It completed in 1,394.916 seconds through
the tiled low-memory decoder, with 960,608 triangles, 4,777,099,104 bytes
peak reserved GPU memory, 12,164,100,096 bytes peak host memory, and 2,280
MMA attention calls. Its GLB passes the
validator and its rendered four-view preview is coherent. Earlier saved CUDA
multiview runs on the resident decoder completed in 512.737 and 490.727
seconds. The slowdown was not caused by GPU contention: the 16,384-token rule
added for 8 GB cards sent this 17,387-token fixture to the tiled decoder at
every budget of 7 GiB or less. That path gathers on the host and re-uploads the
full F32 convolution weight for every 2048-row tile, which accounts for the
785 GB of host-to-device traffic and the 325 s and 318 s decoder stages. The
tiled decoder also rounds some steps on the host, so its output differs from
the resident decoder (6,335,535 versus 6,335,604 final crab voxels). The run's
artifacts are under `tmp/pixal3d/revalidate-cuda-multiview-20260924/`.

## Memory-bounded resident decoder

The resident sparse decoder now releases every activation after its last
consumer, normalizes in place, scatters each 2048-parent conv1 tile straight to
its children with row-offset C2S, accumulates each conv2 tile into the skip
buffer, and runs the ConvNeXt MLP in row chunks that are multiples of the
2048-row GEMM tile. Every GEMM keeps its original shape, so output is
byte-identical. The 16,384-token rule is gone: the resident decoder always
runs, and only an allocation beyond the budget or device memory retries on the
tiled decoder. Decoder replay of saved latents (`replay_decoders.py`, shape
then guided texture, RTX 5060 Ti; the old resident range includes the original
full crab run's 14.7 s + 14.2 s decoder stages):

| Input | Budget | Path | Decoders | H2D | Peak active | Output |
|---|---:|---|---:|---:|---:|---|
| crab, 17,860 tokens | 12288 MiB | old resident | 28.9-34.4 s | 5.6 GB | 9.23 GiB | reference |
| crab | 7168 MiB | old tiled | 533.2 s | 939 GB | - | differs |
| crab | 12288 MiB | new resident | 19.7 s | 3.8 GB | 4.69 GiB | identical |
| crab | 7168 MiB | new resident | 19.8 s | 3.8 GB | 4.69 GiB | identical |
| house, 10,765 tokens | 12288 MiB | new resident | 13.2 s | - | - | identical |
| house | 2560 MiB | tiled retry | 357.9 s | 610 GB | - | tiled result |

```sh
ref/pixal3d/run.sh cuda ref/pixal3d/replay_decoders.py \
  --dump-dir tmp/pixal3d/resident-runs/cuda-crab/dumps \
  --expect-dir tmp/pixal3d/resident-runs/cuda-crab/dumps \
  --vram-budget-mib 7168 --profile-json tmp/pixal3d/decoder-replay/crab-7168.json
```
