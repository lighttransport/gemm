# Pixal3D PyTorch reference

This project-local uv environment validates the [native implementation](../../cpu/pixal3d/README.md).
It is not used by native inference. Upstream versions are pinned in
[sources.json](sources.json), Python dependencies in `uv.lock`, and license notices
in [THIRD_PARTY.md](THIRD_PARTY.md). All downloads, environments and temporary files
stay under `ref/pixal3d/` or the repository's `tmp/pixal3d/` directory.

## Setup and GPU access

Run from the repository root. `uv` must be on PATH. Use a separate environment for
each backend; never install another backend's PyTorch into an existing one.

```sh
ref/pixal3d/setup.sh cpu
ref/pixal3d/setup.sh cuda
ref/pixal3d/setup.sh rocm
ref/pixal3d/run.sh cpu ref/pixal3d/prepare_refs.py
ref/pixal3d/setup_native.sh
make -C cpu/pixal3d all validation -j4
make -C cuda/pixal3d
make -C rdna4/pixal3d
```

The setups use Python 3.12 and exclusive uv extras:

| Environment | PyTorch | Tested device |
|---|---|---|
| `.venv-cpu` | 2.7.1+cpu | Threadripper 1950X |
| `.venv-cuda` | 2.7.1+cu128 | RTX 5060 Ti, sm_120, 16 GB |
| `.venv-rocm` | AMD 2.11.0+rocm7.2.2 wheel | RX 9070 XT, gfx1201, 16 GB |

`setup.sh` performs a real PyTorch matrix multiplication and checks that the
requested backend/device is available. ROCm uses the project's AMD wheel URLs
and includes `/opt/rocm/core/lib` in its runtime library path on this host.
GPU tests need access to `/dev/nvidia*` or `/dev/kfd` and the relevant render node;
a sandbox that hides those devices must grant GPU access before validation.

The PyTorch reference uses upstream model classes with SDPA. `upstream_import.py`
avoids importing optional UI/background-removal/rendering dependencies.
`sparse_reference.py` supplies a portable gather + PyTorch linear convolution for
the upstream sparse decoder, preserving its model, coordinate and subdivision
code. This supports CPU and ROCm without installing CUDA-only sparse extensions.

The native API must first be built with the `validation` target. The validation
bridge is a separate library and is not required by the production CLI.

## Reference checks

```sh
# Repeat with cpu/cuda/rocm as appropriate.
make -C cpu/pixal3d test
ref/pixal3d/run.sh cpu ref/pixal3d/validate.py --backend cpu --flow
ref/pixal3d/run.sh cuda ref/pixal3d/validate.py --backend cuda --flow
ref/pixal3d/run.sh rocm ref/pixal3d/validate.py --backend rocm --flow
# Keep the native GPU backend while using slow CPU PyTorch as a deterministic oracle.
ref/pixal3d/run.sh cuda ref/pixal3d/validate.py --backend cuda --reference-device cpu \
  --gpu-execution resident --gpu-kernels auto
# Mixed and FP32 flow modes use the FP32 PyTorch oracle for their flow block.
ref/pixal3d/run.sh cuda ref/pixal3d/validate.py --backend cuda --reference-device cpu --flow \
  --gpu-execution resident --gpu-flow-precision fp32

# Compare the complete 30-block structure stack at a matching dtype.
ref/pixal3d/run.sh cuda ref/pixal3d/validate_flow_precision.py --backend cuda \
  --dump-dir tmp/pixal3d/runs/cuda-house/dumps --stage structure --precision fp32
ref/pixal3d/run.sh cpu ref/pixal3d/validate_projection.py
ref/pixal3d/run.sh cpu ref/pixal3d/validate_api.py

# Resumable per-submodule verification for a recorded CUDA/ROCm run. Each JSONL
# row is an independent primitive, conditioning, flow-stage, or decoder result;
# rerunning with --skip-complete keeps passed rows and continues after failures.
ref/pixal3d/run.sh cuda ref/pixal3d/verify_gpu_run.py --backend cuda \
  --dump-dir tmp/pixal3d/runs/cuda-house/dumps --fov 0.857556 --all \
  --reference-device cpu --output tmp/pixal3d/verification/cuda-house.jsonl
ref/pixal3d/run.sh rocm ref/pixal3d/verify_gpu_run.py --backend rocm \
  --dump-dir tmp/pixal3d/runs/rocm-house/dumps --fov 0.857556 --all \
  --reference-device cpu --output tmp/pixal3d/verification/rocm-house.jsonl

ref/pixal3d/run.sh cuda ref/pixal3d/validate_conditioning.py --backend cuda --natten
ref/pixal3d/run.sh rocm ref/pixal3d/validate_conditioning.py --backend rocm
ref/pixal3d/run.sh cpu ref/pixal3d/validate_conditioning.py --backend cpu

ref/pixal3d/run.sh cuda ref/pixal3d/validate_decoders.py --backend cuda --stage structure
ref/pixal3d/run.sh rocm ref/pixal3d/validate_decoders.py --backend rocm --stage shape
ref/pixal3d/run.sh cpu ref/pixal3d/validate_decoders.py --backend cpu --stage structure --precision float32
ref/pixal3d/run.sh cpu ref/pixal3d/validate_decoders.py --backend cpu --stage shape --precision float32
ref/pixal3d/run.sh cpu ref/pixal3d/validate_decoders.py --backend cpu --stage shape \
  --noise-device cpu --guided tmp/pixal3d/cpu-guided

ref/pixal3d/run.sh cpu ref/pixal3d/validate_preprocessing.py
ref/pixal3d/run.sh cpu ref/pixal3d/validate_pointwise.py
ref/pixal3d/run.sh cpu ref/pixal3d/validate_normals.py
ref/pixal3d/run.sh cpu ref/pixal3d/validate_inpaint.py
# Optional CUDA CuMesh oracle (requires the matching CUDA toolkit/headers).
ref/pixal3d/build_cumesh.sh
ref/pixal3d/run.sh cuda ref/pixal3d/validate_geometry.py
```

`--natten` checks the original NAF CrossAttention through NATTEN's own PyTorch
Flex backend. The pinned binary wheel lacks an sm_120 CUTLASS image. Independent
value-channel tiles satisfy Flex's equal-head-width constraint without changing
attention probabilities or neighborhoods. The native runtime never imports NATTEN.

Primitive checks use `atol=1e-5, rtol=1e-4`. Checkpoint stages require cosine
similarity >=0.999 and normalized RMSE <=0.02. Projection/cascade/subdivision
coordinate arithmetic and FDG triangles are checked exactly where applicable.
Remeshing and simplification require bidirectional surface F-score >=0.99 within
two voxels. Inpainting is byte-exact against OpenCV 4.12.

FP16 subdivision predictions can cross zero under different reduction orders.
The `--guided` decoder test preserves upstream subdivision decisions and checks
exact coordinate expansion, continuous decoded features, each subdivision's
logit error, and the margin/count of changed native decisions. It fails if any
changed reference logit has absolute magnitude >0.02. Unguided tests remain
available and require exact output coordinates; they can expose threshold-sensitive
inputs rather than silently matching nearest voxels. The seed-55 CPU fixture
has one such decision at the last stage, with margin 0.002758 and guided feature
NRMSE 0.000877. Its FP32 run matches all coordinates.

The texture decoder is checked with `--stage texture --guided DIR` on each backend.
PyTorch GPU reference checks cap their caching allocator at 45% of device memory;
full-image NAF convolution validation runs on CPU to avoid large cuDNN workspaces.
The following checks native CPU arithmetic against the original CUDA PyTorch
model when the CPU PyTorch FP16 convolution fallback is prohibitively slow on
AVX2 hardware. The launcher selects the reference environment:

```sh
ref/pixal3d/run.sh cuda ref/pixal3d/validate_decoders.py --backend cpu \
  --reference-device cuda --noise-device cpu --stage structure
```

## Full-stage replay and measurements

Native `--dump-dir` output includes the exact noise and conditioning inputs used
in inference. Replay a full 30-block step through upstream models/samplers:

```sh
ref/pixal3d/run.sh rocm ref/pixal3d/validate_flow_stage.py --backend rocm \
  --dump-dir tmp/pixal3d/runs/rocm-house/dumps --stage shape512 --step 1
```

Stages are `structure`, `shape512`, `shape1024` and `texture`. Comparison starts
from the recorded previous step, so accumulated sampling differences do not hide
an individual stage error. `--trajectory` instead diagnoses the accumulated
12-step difference and reports whether it lies within the isolated-step tolerance;
it does not label a trajectory as passing that tolerance. `--attention math`
selects PyTorch's mathematical SDPA kernel, and `--save-reference FILE` saves
the reference result for comparisons between backends.

For the recorded house structure noise, native ROCm versus PyTorch ROCm has
trajectory NRMSE 0.03713, exceeding the isolated-step 0.02 threshold. The original
PyTorch CUDA versus ROCm trajectories differ by 0.07013; PyTorch ROCm automatic
versus mathematical attention differs by 0.05345. These measurements show that
full BF16 trajectories are backend-sensitive even within PyTorch. They do not
establish full-trajectory equivalence, and the native seeds must not be interpreted
as reproducing bit-identical meshes across backends. The CUDA/ROCm baselines use
the pinned PyTorch versions listed above, so their comparison includes framework
version and kernel choices. The ROCm automatic/math comparison uses one PyTorch
version. Reproduce paired reference
measurements with `validate_trajectory_baselines.py --dump-dir DIR --stage structure
--include-math --output-dir DIR`; it uses the separate project CUDA/ROCm environments.

The paired automatic-attention measurements for all four stages are below. Each
stage starts from the same recorded noise and conditioning, independently of the
preceding stage. These are diagnostics, not passing equivalence results:

| Twelve-step trajectory | Native ROCm vs PyTorch ROCm NRMSE | PyTorch CUDA vs ROCm NRMSE |
|---|---:|---:|
| Structure | 0.03713 | 0.07013 |
| Shape 512 | 0.03872 | 0.03790 |
| Shape 1024 | 0.06841 | 0.06434 |
| Texture | 0.00414 | 0.00408 |

`validate_preprocessing.py` compares both native input
sizes against the original Pixal3D preprocessing method.

```sh
ref/pixal3d/run.sh cpu ref/pixal3d/run_fixture.py --backend rocm \
  --input ref/pixal3d/upstream/assets/images/1_img.png --fov 0.857556 \
  --output-dir tmp/pixal3d/runs/rocm-house --dump
```

`--mask MASK.png` exercises RGB-plus-mask input. The bounded CPU smoke fixture
is reproducible with `create_cpu_fixture.py --output-dir DIR`; it retains all
1024-cascade stages, 12 diffusion steps per stage, and 4096 textures.

```sh
ref/pixal3d/run.sh cpu ref/pixal3d/create_cpu_fixture.py \
  --output-dir tmp/pixal3d/cpu-fixture
ref/pixal3d/run.sh cpu ref/pixal3d/run_fixture.py --backend cpu \
  --input tmp/pixal3d/cpu-fixture/pencil-rgb.png \
  --mask tmp/pixal3d/cpu-fixture/pencil-mask.png --fov 0.857556 --seed 4 --threads 12 \
  --output-dir tmp/pixal3d/runs/cpu-pencil --dump
```

The harness launches the native binary and records process RSS, NVIDIA per-process
VRAM and total device VRAM, or AMD total device VRAM, every second. It imports no
PyTorch. AMD device measurements include other processes. Native scratch peaks
exclude driver/BLAS overhead; compare both measurements when assessing a 16 GB run.

`replay_postprocess.py` runs saved shape/texture decoder outputs through the native
mesh/PBR exporter to validate postprocessing changes without repeating diffusion.
It is a development utility, not a Python implementation of the inference pipeline.
`validate_mesh_stage.py --dump-dir DIR` compares the saved 1024 remesh and
simplification against original CuMesh using every vertex and face centroid.
`validate_glb.py FILE.glb` verifies mesh/UV/normal/material attributes and embedded
4096 images. It reports the pinned exporter's rare collapsed boundary faces and
zero boundary normals separately, while still bounding their count and requiring
every other normal to be unit length. Build the shared preview renderer with `ref/pixal3d/build_preview.sh`,
then use `preview_glb.py FILE.glb --output-dir DIR` for four CPU-rendered views.
The preview copy is uniformly scaled to avoid the renderer's fixed triangle-size
epsilon and uses wider camera framing; exported GLB geometry is unchanged. Each
view includes a renderer-derived binary mask for meaningful silhouette metrics.
`compare_outputs.py NATIVE.glb REFERENCE.glb` deterministically samples both
surfaces and reports bidirectional Chamfer distances plus oriented and
orientation-independent nearest-face normal agreement. Pass matching
`--native-renders` and `--reference-renders`
directories to add RGB error, PSNR, and silhouette IoU for rendered PNG views.

## Validation record

Checkpoints are taken from `/mnt/disk2/models/Pixal3D` and the local timm DINOv3
ViT-L/16 checkpoint. `manifest.py` records checkpoint sizes, tensor dtypes, header
hashes, configuration and reference-source provenance; large weights are not copied.
Use `--full-hash` to stream the tensor payloads through SHA256 as well:

```sh
ref/pixal3d/run.sh cpu ref/pixal3d/manifest.py --full-hash \
  --output tmp/pixal3d/model-manifest.json
```

The extended 2026-09-20 section of `validation-results.json` consolidates the
7/12 GiB multiview runs, pinned PyTorch comparison, render metrics, mask-model
provenance, CUDA reliability soak, postprocessing replay, and real queued web
tests. It records exact commands, implementation commits, source/checkpoint
revisions, artifact hashes, runtime versions, and concurrency qualifications.
Large artifacts remain under `tmp/pixal3d/` and are represented by size and
SHA-256 only. Validate the checked-in schema and thresholds, and optionally all
retained local artifacts, with:

```sh
ref/pixal3d/run.sh cpu ref/pixal3d/validate_results_record.py
ref/pixal3d/run.sh cpu ref/pixal3d/validate_results_record.py --artifacts
```

Automatic single-view preparation uses the upstream RMBG-2.0 and MoGe-2
semantics without loading the Pixal3D generation stack. Fetch the pinned source
and locally available checkpoints, then emit an RGBA image and resolved camera
metadata with:

```sh
ref/pixal3d/run.sh cpu ref/pixal3d/prepare_refs.py
ref/pixal3d/run.sh cpu ref/pixal3d/prepare_auto_models.py \
  --model-root /mnt/disk2/models
ref/pixal3d/run.sh cuda ref/pixal3d/prepare_input.py \
  --input input.png --output tmp/pixal3d/prepared.png \
  --metadata tmp/pixal3d/prepared.json \
  --rembg-model /mnt/disk2/models/RMBG-2.0 \
  --moge-model /mnt/disk2/models/moge-2-vitl/model.pt
```

The downloader fetches only the Transformers inference files for RMBG-2.0. It
skips the repository's duplicate PyTorch checkpoint, ONNX variants, and sample
images, avoiding several gigabytes of unrelated downloads.

RMBG-2.0 is gated by its publisher and needs an authorized Hugging Face account.
The pinned revision was validated on the RTX 5060 Ti by converting the upstream
RGBA house asset to opaque RGB and forcing the model path. Its generated mask
contained both foreground and background and reached 0.99419 IoU against the
asset's reference alpha. The validator requires a nontrivial alpha channel as
its oracle before creating the opaque RGB model input. Reproduce the
model-backed check with:

```sh
ref/pixal3d/run.sh cuda ref/pixal3d/validate_rmbg.py \
  --model /mnt/disk2/models/RMBG-2.0 --device cuda \
  --json-output tmp/pixal3d/rmbg-validation/result.json
```

### RMBG mask-parity interpretation

The bundled house alpha is closer to the checkpoint named by Pixal3D's pinned
wrapper, `ZhengPeng7/BiRefNet` at revision `e2bf8e44`, than to the newer
`briaai/RMBG-2.0` checkpoint used by automatic preparation. With the same
opaque RGB input and the validator's normal `alpha > 127` comparison, measured
IoU was 0.994194 for RMBG-2.0 and 0.998310 for the exact upstream BiRefNet
checkpoint. The corresponding upstream-checkpoint soft masks had 0.150 mean
absolute 8-bit alpha error and agreed exactly at 96.0% of pixels.

An exhaustive diagnostic sweep reached IoU 0.999004 using the upstream
checkpoint, Lanczos output resizing, and a threshold of 122. This is an
asset-tuned threshold and differs from the pinned wrapper's implicit PIL resize
semantics, so it is not used as the production path or headline RMBG-2.0 score.
Changing the binary validation threshold would not improve the soft alpha mask
consumed by Pixal3D. The experiment therefore attributes the remaining gap to
checkpoint provenance and minor reconstruction details rather than to an
inference implementation mismatch.

Reproduce the standard and diagnostic measurements for both pinned checkpoints
and record their full model SHA-256 values with:

```sh
ref/pixal3d/run.sh cuda ref/pixal3d/validate_mask_provenance.py \
  --device cuda --output tmp/pixal3d/rmbg-validation/provenance.json
```

Pinned multiview verification can still run on inputs with useful RGBA alpha
through `run_reference_mv.py`. Its placeholder raises on RGB or fully opaque
views, so it does not replace RMBG or change reference masking behavior.
The upstream binary extensions publish CPython 3.10 CUDA wheels. Create their
isolated, pinned environment and run the reference CLI with:

```sh
ref/pixal3d/setup_reference_cuda310.sh
ref/pixal3d/run_reference_cuda310.sh cuda ref/pixal3d/run_reference_mv.py \
  --views_dir ref/pixal3d/upstream/assets/mv_images/example \
  --output tmp/pixal3d/reference-mv/mesh.glb \
  --model_path /mnt/disk2/models/Pixal3D --seed 42 --low_vram --resolution 1024
```
Single-view reference inference uses the same compatibility layer through
`run_reference_sv.py` and accepts RGBA inputs without loading RMBG.
This environment is separate from the per-backend validation environments so
the native-versus-PyTorch checks retain their newer PyTorch setup. The setup
pins Torch 2.7.1/cu128 for `sm_120`, rebuilds Torch-ABI extensions from pinned
source revisions, and uses PyTorch SDPA plus bounded chunked NAF attention on
Blackwell. NAF projection evaluates only the four bilinear source pixels needed
by each projected grid point, avoiding the otherwise 4 GiB dense 1024-square
feature tensor. This enabled the complete four-view 1024 reference pipeline on
the 16 GB RTX 5060 Ti. Against the native 12 GiB-budget run at seed 42, 200,000
surface samples measured symmetric Chamfer RMS 0.005548; directional means were
0.004367 native-to-reference and 0.004147 reference-to-native. Absolute normal
cosine means were 0.9533 and 0.9586, which avoids conflating local winding
differences with surface-direction disagreement. The outputs had 961,142 and
960,561 triangles respectively. Native four-view inference is also validated at
the 7 GiB budget. Four matched CPU previews measured RGB PSNR from 26.39 to
29.09 dB and silhouette IoU from 0.9885 to 0.9949. The reference uses
`EXT_texture_webp`; the preview extractor resolves both extension-backed WebP
and direct core image sources.

Verify the bounded NAF fallback against a direct PyTorch implementation with:

```sh
ref/pixal3d/run_reference_cuda310.sh cuda \
  ref/pixal3d/validate_reference_attention.py
```

After producing the two full multiview budget runs, repeat resident recovery,
mixed/FP32 matched-reference checks, invalid-input handling, queue cancellation,
and artifact validation while sampling RSS and device memory with:

```sh
ref/pixal3d/run.sh cuda ref/pixal3d/validate_cuda_reliability.py \
  --dump-dir tmp/pixal3d/resident-runs/cuda-house/dumps \
  --multiview-runs tmp/pixal3d/mv-budget-7168 tmp/pixal3d/mv-budget-12288 \
  --output tmp/pixal3d/cuda-reliability.json
```
Images that already have non-opaque alpha do not load RMBG. The MoGe-2 checkpoint
is pinned to revision `39c4d5e9` and SHA-256 `3eefd4ab...e0111cd5`; on the upstream
house asset it estimated FOV `0.6061274` radians and distance `1.598996` on the
RTX 5060 Ti. The pinned upstream helper matched within `1.14e-8` radians and
`2.3e-16` distance. Reproduce that comparison with `validate_auto_camera.py`.
`test_prepare_input.py` covers alpha preservation, explicit masks, manual camera
resolution, metadata, and the missing-mask diagnostic without model weights.

Completed during implementation on this host:

| Check | Representative result |
|---|---|
| CPU/CUDA/ROCm device and primitive checks | PASS |
| Native C API and analytic projection/sampling/CFG checks | PASS; warning-clean C11 build |
| BF16 one-block structure flow, all backends | NRMSE <0.0035; cosine >0.99999 |
| DINOv3, all 24 blocks, all backends | NRMSE <0.000005 |
| NAF guide/projection, all backends | NRMSE <0.000001 |
| Original NATTEN neighborhood calculation | NRMSE 0.00000014 |
| FP16 structure decoder, CPU/CUDA/ROCm vs PyTorch CUDA/ROCm | NRMSE <0.00023 |
| FP32 structure and shape decoders, CPU | NRMSE <0.000001 |
| FP16 shape decoder, CUDA/ROCm stable fixture | Exact 237 coordinates; NRMSE <0.0009 |
| FP16 CPU guided shape decoder | Exact 503 coordinates; NRMSE 0.000877 |
| Original camera projection, 16/32/64 grids | PASS |
| Original preprocessing: large RGBA and RGB+mask, 512/1024 outputs | Byte-exact |
| Full structure diffusion step | NRMSE 0.000383 |
| Full shape512 diffusion step | NRMSE 0.001150 |
| Full 1024 shape diffusion step, 10636 tokens | NRMSE 0.001174 |
| Full texture diffusion step | NRMSE 0.0001875 |
| Final diffusion steps, all four stages, native ROCm house | NRMSE 0.00214–0.00714; cosine >=0.999974 |
| Native CPU pencil final structure/shape512 steps vs PyTorch CUDA | NRMSE 0.01206 / 0.00338; cosine >=0.999927 |
| Full 1024-image DINOv3/NAF conditioning | NRMSE <0.000002 |
| Texture decoder, CPU/CUDA/ROCm | Exact guided coordinates; NRMSE <0.00131 |
| Vector rounding/residual/modulation | Exact vs PyTorch |
| Vectorized real 10636-token flow block | Bit-identical to previous native output |
| CPU attention across a 1024-query tile boundary | Max absolute error 1.49e-8 vs PyTorch |
| Larger CPU attention tiles, real 10636-token flow block | 28.3 s to 19.2 s warm median at 16 threads under concurrent host load; flow only, excluding conditioning, decoding, postprocessing, model loading and serialization; NRMSE 2.76e-5 vs prior native |
| UV degenerate/unreferenced cleanup | Exact vs CuMesh |
| Area-weighted normals with cancellation fallback | PASS on 484748 real mesh vertices |
| FDG vertices/triangles and legacy axis mapping | PASS |
| Remesh 64/128, original CuMesh | Matching vertex/face counts; surface error <1e-7 |
| QEM simplification 64/128, original CuMesh | Surface F-score 1.0 |
| Real house remesh at 1024, original CuMesh | F-score 1.0; max surface distance 0.000313 |
| Real house simplification to about 1M faces | F-score >0.99989 within two voxels |
| OpenCV 4.12 inpaint, varied holes/channels/radii | Byte-exact |
| Geometry fixture: 1024 remesh + 4096 PBR GLB | PASS |

### Current resident CUDA multiview runs

The current complete-generation reference uses resident-auto execution, mixed
precision, the pinned four-view upstream example, seed 42, 4096 textures and a
one-million-triangle target on the RTX 5060 Ti. Desktop GPU/CPU work was present,
so these are functional observations. The native generation timer excludes GLB
serialization; wall time includes it.

| Native budget | Generation | Wall time | Native reserved peak | Output SHA-256 |
|---:|---:|---:|---:|---|
| 7168 MiB | 506.207 s | 512.737 s | 7160.6 MiB | `a5a22a90...c700383` |
| 12288 MiB | 484.915 s | 490.727 s | 8488.0 MiB | `a5a22a90...c700383` |

Both outputs are byte-identical and pass mesh, normal and 4096 PBR texture
validation with 655,071 vertices and 961,142 triangles. These runs establish
the 8 GB-card minimum and 12 GiB target behavior; isolated Shape-1024 flow
benchmarks are documented separately in `cpu/pixal3d/OPTIMIZATION.md`.

### Historical host-offloaded full pipeline runs

Recorded on 2026-09-14 (JST) with legacy host-offloaded execution. These results
are retained to explain its bounded-VRAM/PCIe tradeoff and are not current
resident performance claims. All seven exported GLBs passed mesh, UV,
unit-normal and embedded 4096 PBR texture checks, and their four-view previews
were inspected. Raw metrics, exact commands, input/export hashes and full
checkpoint SHA256 hashes are retained in [validation-results.json](validation-results.json).

The following five runs completed as uninterrupted native CLI invocations:

| Backend / fixture | Shape tokens | Export triangles | Invocation minutes | Peak host RSS GiB | Native GPU scratch MiB |
|---|---:|---:|---:|---:|---:|
| cpu-pencil | 1132 | 960992 | 62.75 | 6.05 | 0.00 |
| cuda-crab | 17773 | 947466 | 82.62 | 13.80 | 826.02 |
| rocm-crab | 17688 | 988586 | 97.33 | 13.27 | 826.02 |
| cuda-jester | 8049 | 940418 | 28.40 | 7.71 | 826.02 |
| rocm-jester | 7955 | 932596 | 38.41 | 7.71 | 826.02 |

NVIDIA per-process VRAM peaked at 1.00 GiB; total NVIDIA device use peaked at
7.14 GiB. Total AMD device use peaked at 5.41 GiB. Device totals include
other processes and concurrent PyTorch reference checks. One-second sampling
can miss brief allocation peaks, so it is reported alongside native counters.
All GPU runs used the 16 GB RTX 5060 Ti or RX 9070 XT without resolution fallback
or token truncation.

Invocation time includes GLB serialization; native `seconds` excludes it.
Runs overlapped each other and reference checks, so these are functional test
observations, not isolated performance comparisons. The host-offloaded native
implementation prioritizes bounded VRAM and remains slow for complex inputs.

The house exports contain 977612 triangles on CUDA and
992202 on ROCm. Their complete native neural outputs were saved, then
replayed through the corrected native C++ postprocessor. They establish staged
pipeline execution and validated exports; their partial monitoring intervals
are not presented as full-invocation timings. The real ROCm house remesh and
simplification also passed the original CuMesh surface checks listed above.

GPU fixtures use upstream `1_img.png` (house, seed 1), `9_img.png` (crab, seed 2),
and `21_img.png` (jester, seed 3), with FOV 0.857556 and eight host threads.
The CPU fixture uses the generated pencil RGB image plus explicit mask, seed 4,
the same FOV and twelve threads. Every case retains the 1024 cascade, all four
12-step samplers and 4096 textures. The earlier partial CPU house attempt is
not counted as a completed run.

Both jester exports retain a small detached ornament in the generated geometry;
no connected-component removal was applied. Visual inspection is not a claim
of exact image reconstruction or full-trajectory PyTorch equivalence.

GLBs, previews, stage dumps and detailed logs remain under ignored
`tmp/pixal3d/runs/`. Reproduce exports with `run_fixture.py` and the fixture
arguments above; use `validate_glb.py` and `preview_glb.py` to check them.

## Resident GPU validation and benchmarking

Build host and both plugins as described in `cpu/pixal3d/README.md`. Run all
Python commands through the existing per-backend `uv` environments:

```sh
ref/pixal3d/run.sh cuda ref/pixal3d/validate_resident.py --backend cuda --benchmark
ref/pixal3d/run.sh rocm ref/pixal3d/validate_resident.py --backend rocm --benchmark
ref/pixal3d/run.sh cpu ref/pixal3d/validate_multiview.py
ref/pixal3d/run.sh cuda ref/pixal3d/validate.py --backend cuda --flow --gpu-execution resident
ref/pixal3d/run.sh rocm ref/pixal3d/validate_conditioning.py --backend rocm --gpu-execution resident
ref/pixal3d/run.sh cuda ref/pixal3d/validate_decoders.py --backend cuda --stage shape --guided tmp/pixal3d/guided-shape --gpu-execution resident
```

`benchmark_flow_block.py` now keeps one engine and weight mapping alive across
calls, separates load/cold/warm times, and accepts `--stage`, `--blocks 30`,
`--tokens`, `--repeats`, `--gpu-execution`, `--gpu-kernels`, and
`--profile-json`. Zero/omitted
`--tokens` uses every recorded token. Texture input includes the matching final
normalized shape features. Compare modes with identical dumps and block counts;
run them sequentially on each GPU. The emitted JSON lives beside the output
safetensors. Warm flow speedups exclude conditioning, decoders and postprocessing.

`run_fixture.py --gpu-execution resident --gpu-kernels auto --dump` records full
runs, profiles and memory samples. Its default timeout is four hours. Apply
`validate_flow_stage.py` to the new dumps, and `validate_glb.py` to the new export;
old generation artifacts are not evidence for the resident implementation.
