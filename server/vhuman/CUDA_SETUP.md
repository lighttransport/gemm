# Recreate the vhuman CUDA development node

`setup_cuda.sh` installs development environments, imports existing research
assets, builds native modules and runs CPU checks. It sets
`CUDA_VISIBLE_DEVICES=` and `JAX_PLATFORMS=cpu` for installation. GPU validation
is a separate command, so an occupied GPU can remain occupied during setup.

Run from the repository root on Linux x86-64. Python 3.12, uv 0.10.4 or newer,
Git, rsync, curl, Make, GCC/G++, CMake, Ninja, FFmpeg/ffprobe, Node.js 22 or newer, an installed CUDA
toolkit and Ubuntu development-package indexes are prerequisites. Browser
validation also uses Chromium, Xvfb and xauth. The installer stages OpenCV and
OpenBLAS development packages locally; it does not run sudo or change drivers.
On a fresh Ubuntu 24.04 installation, the corresponding system prerequisites
can be installed with:

```sh
sudo apt-get install python3 python3-venv build-essential git rsync curl cmake \
  ninja-build ffmpeg pkg-config xvfb xauth chromium-browser
```

Install uv using its supported installer or your package manager. Keep the
existing base model directories under `/mnt/disk1/models`; the setup imports
missing vhuman-specific exports/checkpoints rather than downloading all large
image/video models again. Allow roughly 50 GiB of additional free storage for
environments, assets, scenes, tools and compiler/package caches.

## One-time installation and recovery

```sh
sh server/vhuman/setup_cuda.sh \
  --source-host tr16 \
  --models-root /mnt/disk1/models \
  --data-root /mnt/disk01/data/vhuman \
  --cuda-home /usr/local/cuda-12.9 --gpu-arch sm_80 --jobs 4
```

The SSH defaults are the existing `~/.ssh/id_ed25519` identity,
`work/gemm/pixal3d` relative to the source login directory,
`/mnt/nvme01/models`, and `~/local/blender-5.2.2-linux-x64` on tr16. Override
them with `--ssh-identity`, `--source-repo`, `--source-models` and
`--source-blender`. The Blender option takes an absolute remote directory.

All scratch files use repository `tmp/`; on b550 it is backed by
`/mnt/disk01/tmp`. Task caches use `tmp/vhuman-dev/cache/`, avoiding obsolete
ROCm cache symlinks and the nearly full root filesystem. Necessary broken
symlinks are preserved under `tmp/vhuman-dev/backups/` before repair.
Existing environments or directories are never replaced by aliases.

Weights default to the `models/` directory on the NVMe volume containing the
resolved checkout: `/mnt/nvme02/models` for this worktree. `--models-root` or
`VHUMAN_MODELS_ROOT` overrides that default. Face assets default to
`MODELS_ROOT/vhuman-rig-models`; `--data-root` remains an explicit face-asset
override. Corpus downloads and generated avatars remain separate from weights.

To migrate the reusable weights from an existing checkout, preview first and
then apply the same-volume moves:

```sh
python3 -m server.vhuman.relocate_models --shared-models-root /mnt/nvme01/models
python3 -m server.vhuman.relocate_models --shared-models-root /mnt/nvme01/models --apply
```

This relocates the MuscleMimic/Terra checkpoints, GenIA conditioning export,
realtime motion checkpoints and tokenizer. Existing external rig/emotion
weights and large shared models receive aliases; no multi-GB copy is made.
Compatibility links preserve saved scripts and provenance paths. Conflicting
installations are refused. The setup `assets` phase also migrates local tmp
weights; `--shared-models-root` optionally exposes already-installed models
from another local volume. Environments, sources, subject-specific deformers,
run outputs and diagnostic fixtures stay in `tmp/`.

Canonical reusable model locations on this node:

| Location under `/mnt/nvme02/models/` | Contents |
| --- | --- |
| `vhuman/mm-fullbody-base` | Complete MuscleMimic Orbax checkpoint |
| `vhuman/TERRA-4B` | Complete Terra checkpoint and metadata |
| `vhuman/realtime` | Motion `.pt`/native packages and FLUX tokenizer |
| `vhuman/sensevoice-small-q8.gguf` | Shared emotion-model alias |
| `vhuman-rig-models` | Shared GNM, ICT, MediaPipe and depth assets |
| `genia/native` | SLAT conditioning export and checksum manifest |

Phases can be resumed independently:

```sh
sh server/vhuman/setup_cuda.sh --phases envs
sh server/vhuman/setup_cuda.sh --source-host tr16 --phases assets
sh server/vhuman/setup_cuda.sh --source-host tr16 --phases tools
sh server/vhuman/setup_cuda.sh --phases build register scenes browser check --gpu-arch sm_80
```

Run dependent phases in this order. Each writes an atomic status receipt under
`tmp/vhuman-dev/`. A failed or interrupted build must be completed before GPU
validation. To capture a full installation log:

```sh
mkdir -p tmp/vhuman-dev/logs
sh server/vhuman/setup_cuda.sh --source-host tr16 \
  > tmp/vhuman-dev/logs/setup.log 2>&1
```

## Installed environments and modules

| Location | Purpose |
| --- | --- |
| `tmp/vhuman-cuda-venv` | Python 3.12; Torch 2.14.0/cu130; pinned diffusers/transformers; rig, texture, USD, ONNX, EXR and browser tooling |
| `tmp/{vhuman-rig-venv,vhuman-texture-venv,qimg21-ref-venv}` | Relative aliases to that one CUDA environment; no cross-environment `.pth` files |
| `tmp/vhuman-musclemimic-py312` | Separate Python 3.12, CPU Torch and the pinned MuscleMimic/GMR/JAX 0.7.2/MuJoCo 3.4.0 stack |
| `tmp/terra-runtime` | Separate Python 3.11 and the upstream frozen Terra dependency closure, with the maintained MHR/optional-SMPL patches |
| `tmp/vhuman-dev/tools/blender-5.2.2-linux-x64` | Same Blender release as the saved face work |
| `tmp/vhuman-dev/tools/three/package` | Checksum-verified Three.js 0.163.0 |
| `tmp/vhuman-tools` | Pinned local shader compiler |
| `tmp/vhuman-dev/scenes` | New Blender copies with verified local external-file paths; original scenes remain unchanged |
| `tmp/vhuman-dev/browser` | Current source compiled to WASM with the recovered contact avatar; CPU JavaScript checks |

The CUDA dependency closure is committed as `requirements-cuda-dev.lock`.
`requirements-cuda-dev.txt` records the direct dependency pins. To propose a
refresh, use `--refresh-lock --phases envs`, inspect the generated
`tmp/vhuman-dev/cuda-requirements.lock`, then update the committed lock explicitly.
The complete MuscleMimic closure is `body/musclemimic-requirements.lock`; its
critical versions remain checked against `musclemimic-constraints.txt`. It is
installed without dependency re-resolution because current Mink metadata
requests a newer MuJoCo than the registered policy supports. The CPU integration
suite checks the recorded combination. Terra retains its own upstream lock,
with an install-time OpenCV 4.13.0.92 override. All three environments use
OpenCV 4 for the video preflight HOG detector; OpenCV 5 lacks that Python API.

Built components include CPU/CUDA face inference, landmarks, training and
deformation; CPU/CUDA MHR decoding and SAM 3D Body inference; native depth;
Pixal3D; Qwen-Image plugins; speech; SenseVoice; and the Vulkan deformer.
Source revisions and compiler architecture are explicit. No inference runs
during compilation.

For the RTX 3070, the supported Makefile target is `sm_80`; runtime-compiled
kernels detect the actual SM86 device. The Sage plugin in this repository uses
FP8 and is omitted on Ampere, as is Blackwell FP4. For native Qwen experiments
on Ampere, override a quantized preset's Sage attention with `--attention flash`.
Edit-2511 GGUF uses its independent diffusers path with block offloading.

The assets phase imports MHR rig metadata, the Edit-2511 GGUF, ICT, optional
depth and emotion weights, complete policy checkpoints, recorded references,
and the saved ear/contact/mouth workspaces. It verifies individual file hashes,
checks native MHR metadata and exports native MoGe from the existing `model.pt`.
Original motion-use/provenance receipts are retained in
`tmp/vhuman-dev/source-{musclemimic,terra}.json`; fresh registrations use local
paths and the same acknowledged use. Registration doctors check source,
package, checkpoint and MHR fingerprints. No SMPL model assets are needed.

Saved source workspaces keep their original receipts and absolute provenance
paths. New local Blender copies use `relocate_blender.py`, which rejects missing
external files and records source/output hashes. Rebuild experiments with the
current repository CLIs and fresh output directories; do not run old scratch
shell scripts containing tr16 paths unchanged.

## Use the installation

```sh
. tmp/vhuman-dev/activate.sh
"$PYTHON" -m server.vhuman.cli --backend cuda --models-root "$VHUMAN_MODELS_ROOT" --help
sh tmp/vhuman-dev/run.sh --port 8790
```

The generated launcher selects CUDA inference and both CPU policy workers.
The current head/contact work is at
`tmp/vhuman-ear-synth/reconstruction_contact`; new local scenes are under
`tmp/vhuman-dev/scenes/contact_scene` and `contact_stress_scene`. The imported
browser review can be served without GPU inference:

```sh
python3 -m http.server 45615 --bind 127.0.0.1 --directory tmp
# http://127.0.0.1:45615/vhuman-ear-synth/contact_player/
```

The general H3/Wan portrait-generation adapters remain ROCm-specific. Reuse
saved expression clips or select the separate CUDA video runner for new video.
Optional LightRig/LightUSD desktop consumers and the private LightGeom teacher
retain their own `rig/external.sh` and `rig/setup_face_sources.sh` installers;
they are not dependencies of native MHR/Terra/MuscleMimic operation.

## Explicit GPU validation, after freeing the GPU

These commands execute GPU work and are never called by the installer.
Each output directory must be fresh. The validator checks physical GPU capacity
and available memory, uses the same portrait conditions/seed, and records CUDA
allocation/reservation, host peak RSS, timings, model and generation receipts.
Successful execution still requires visual review of the generated result.

First run on the actual RTX 3070 8 GiB card:

```sh
. tmp/vhuman-dev/activate.sh
"$PYTHON" -m server.vhuman.dev_validate edit8 --out tmp/edit2511-8gb-smoke --steps 2
"$PYTHON" -m server.vhuman.dev_validate edit8 --out tmp/edit2511-8gb-full --steps 12
```

After installing the RTX 5060 Ti 16 GiB, rebuild plugins for Blackwell, then run
the capability, BF16, native CPU/CUDA parity and matching Edit-2511 checks:

```sh
sh server/vhuman/setup_cuda.sh --phases build check --gpu-arch sm_120 \
  --cuda-home /usr/local/cuda-12.9
. tmp/vhuman-dev/activate.sh
"$PYTHON" -m server.vhuman.dev_validate sm120 --out tmp/vhuman-sm120-smoke --steps 2
"$PYTHON" -m server.vhuman.dev_validate sm120 --out tmp/vhuman-sm120-full --steps 12
```

`sm120` requires a completed `sm_120` build receipt and compute capability 12.0.
The rebuild enables Sage and FP4 plugins; Edit-2511 keeps block offloading so the
8/16 GiB results use the same execution recipe. No driver changes, process
termination or GPU upgrade is performed by these scripts.

## b550 validation, 2026-10-15

The Ampere build used GCC 13.3 and CUDA 12.9. The shared weight audit reports
20 required and three optional assets present. Four Blender scenes were
relocated without missing external files, and the recovered contact candidate
and six edit-conditioning views passed CPU preparation.

The `check` phase ran 48 face/setup tests (one optional skip), 34 portable
motion tests (two optional skips), and the native C training/Pixal3D suites.
The browser phase compiled the current WASM source and passed 13 JavaScript
tests without WebGL initialization. The generated server launcher passed
argument/help validation with CUDA visibility disabled.

Three real MuscleMimic CPU integration tests passed using the registered
checkpoint and `tmp/terra-native-review/avatar-varied`. A real Terra CPU
standing take also completed and passed the numerical export gates: skin
stance drift p95 1.84 mm, maximum penetration 0.119 mm, site-fit RMS 22.0 mm.
This fixture establishes mechanics; visual review and representative walking
remain separate work. Reproduce these optional CPU integration checks with:

```sh
(
export CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu TMPDIR="$PWD/tmp"
export MUSCLEMIMIC_TEST_ASSETS="$PWD/tmp/vhuman-musclemimic-assets"
export MHR_TEST_ASSETS=/mnt/disk1/models/sam3d-body/safetensors
export MHR_TEST_AVATAR="$PWD/tmp/terra-native-review/avatar-varied"
tmp/vhuman-musclemimic-py312/bin/python -B -m unittest \
  server.vhuman.body.test_musclemimic_integration -v
tmp/terra-runtime/venv/bin/python -B -m server.vhuman.body.terra_worker \
  --request tmp/vhuman-dev/terra-stand-request.json --assets tmp/terra-assets \
  --work tmp/vhuman-dev/terra-stand-rerun --policy-device cpu
)
```

Use a fresh `--work` directory for each take. Installation/download caches
were reclaimed after checking that the environments have no runtime links
into them; model data, source checkouts, receipts and generated scenes remain.
GPU checks were deferred during installation; the following section records
the subsequent execution tests after the GPU was freed.

## b550 GPU execution, 2026-10-16

The RTX 3070 has 8 GiB VRAM and compute capability 8.6. Execution used the
installed Torch 2.14.0/cu130 environment and CUDA 12.9 native runtime tools.

The two-step Edit-2511 GGUF test completed the conditioning, CPU prompt
encoding, block-offloaded denoising and 1024-square image decode. End-to-end
validation took 335.61 seconds; generation took 256.54 seconds, including
159.02 seconds for CPU prompt encoding. Peak PyTorch allocation was 1.91 GiB,
reservation 2.43 GiB and host RSS 49.28 GiB. Its image remains noisy at two
steps; this establishes execution and memory feasibility. Receipts and images
are under `tmp/edit2511-8gb-smoke-20261016/`.

The matching 12-step run completed in 690.88 seconds; generation took 617.54
seconds, including 152.36 seconds of CPU prompt encoding. PyTorch GPU peaks
were unchanged, and host RSS peaked at 49.55 GiB. **Visual quality failed:**
the raw edit generated an unrelated scene containing people instead of
completing the isolated head. The composite inherited that scene in unknown
regions. All 53,767 protected photographed pixels remained byte-identical in
both runs. This GGUF recipe needs conditioning/quality investigation before
texture use; no generated image or fresh reconstruction was promoted.
The full receipt is `tmp/edit2511-8gb-full-20261016/validation.json`.

Native execution checks also passed:

- Repository CUDA GEMM: all four transpose/tail cases, maximum absolute error
  `2.13e-6` against double accumulation.
- Six explicitly enabled CUDA training tests, including CPU/CUDA forward,
  gradient and resident-Adam parity, memory budgets and framework-free export;
  five native image-training tests.
- Face deformer: 32 frames, 11,909 vertices and 52 morphs, all three modes;
  maximum CPU/CUDA vertex difference `7.45e-9 m`.
- MHR body decoding: CUDA mesh and CPU skeleton-only cases for batches 1/4/5
  against an independent TorchScript reference. Maximum vertex error was
  `1.06812e-4 cm`; CUDA model residency was 690,146,892 bytes. The production
  Python adapter passed the additional real-CUDA regression test.
- Fresh portrait reconstruction through the production CLI: GNM v3,
  256-pixel material maps, ten fitting iterations, automatic parsing and no rig
  rebuild. The reconstruction worker took 43.74 seconds, wrote finite
  17,821-vertex complete anatomy and passed candidate hash validation. The
  fitting objective fell from 0.46348 to 0.12702; the bounded fit did not
  converge and is not an accepted replacement for the saved face candidate.

Native reports and the fresh reconstruction are under
`tmp/vhuman-cuda-execution-20261016/`; `summary.json` separates successful
execution from failed visual quality, and `index.html` compares input, raw
outputs and composite. The body test exposed and fixed a CUDA
startup error: `cu_compile_kernels_ex` returns a positive SM number on success,
which MHR had interpreted as failure. The `sm120` validator also now passes
the explicit `--gpu` switch to the native training suite.

The measured timings describe execution checks; other small validation jobs
ran during preparation and CPU prompt encoding. Reproduce the native training
and fresh portrait checks with:

```sh
. tmp/vhuman-dev/activate.sh
make -C cuda/vhuman test-training
"$PYTHON" -B -m server.vhuman.test_native_gpu_training --gpu -v
"$PYTHON" -B -m unittest server.vhuman.test_native_image_training -v
"$PYTHON" -B -m server.vhuman.cli --backend cuda \
  --models-root "$VHUMAN_MODELS_ROOT" --rig-python "$PYTHON" \
  --work tmp/vhuman-portrait-execution-check portrait-reconstruct \
  --portrait tmp/vhuman-ear-synth/reconstruction_contact/portrait.png \
  --face-model gnm_v3 --profile full --res 256 --iterations 10 --no-rig
```

Reproduce the body oracle and adapter check with fresh output:

```sh
. tmp/vhuman-dev/activate.sh
"$PYTHON" -B ref/sam3d-body/verify_native_mhr.py \
  --model "$VHUMAN_MODELS_ROOT/sam3d-body/dinov3/assets/mhr_model.pt" \
  --assets "$VHUMAN_MODELS_ROOT/sam3d-body/safetensors" \
  --runner cuda/sam3d_body/mhr_decode --backend cuda \
  --pose-sidecar tmp/terra-native-review/avatar-varied/body_mhr.glb.json \
  --out tmp/vhuman-mhr-cuda-check
MHR_TEST_ASSETS="$VHUMAN_MODELS_ROOT/sam3d-body/safetensors" \
MHR_TEST_REFS="$PWD/tmp/vhuman-mhr-cuda-check" MHR_TEST_CUDA=1 \
  "$PYTHON" -B -m unittest \
  server.vhuman.body.test_mhr.NativeModelTest.test_native_cuda_coefficients_and_metadata -v
```

The RTX 5060 Ti / `sm_120` checks still require the planned GPU replacement.

## RTX 3070 follow-up, 2026-10-16

The Edit-2511 conditioning failure above was traced to manual CPU prompt
encoding. It used full-size images instead of the pinned EditPlus pipeline's
384²-area condition images. Native and GGUF editors now use identical prompt
image preprocessing while retaining the original VAE image inputs.

The corrected 12-step run generated the intended isolated side head against
gray, with foreground IoU 0.99625 against the conditioning silhouette. All
53,767 protected portrait pixels remained byte-identical. Its skin is still
flat/synthetic and has not been accepted as a replacement appearance. The
13 texture tests passed, including the preprocessing regression. Overall
execution took 540.01 seconds, with 488.68 seconds generation and 54.76 seconds
CPU prompt encoding. Peak Torch allocation/reservation was 1.84/2.35 GiB and
host RSS 46.73 GiB. Thread settings differ from the earlier run, so timing
changes are not an isolated preprocessing benchmark. Receipts and images are
under `tmp/vhuman-3070-followup/edit/`; the measured receipt is
`tmp/vhuman-3070-followup/edit-conditioned-12.json`.

The existing tr16 motion-retargeting checkout supplied two 60-frame, 30 Hz
BVH recordings: Claire's `Walk Right` and `Turn 90 Left`. They contain 1.54 m
travel and a 90° heading change respectively. The bounded importer agrees
with the pinned GMR reader to 1.8e-15 m on global joint positions. Source and
derived recordings stay under the ignored follow-up workspace with their
CC-BY-NC source notice and provenance; no recording is committed.

The full GNM fit converged in 26 evaluations, reducing its objective from
0.46348 to 0.12703. Complete 17,821-vertex geometry passed finite-value and
provenance checks. The 600-iteration CUDA rig registration reached 1.481 mm
mean / 3.022 mm p95 distance. Full corrective training used 4,096 samples,
96 PCA components and 2,500 epochs on the native CPU backend: held-out mean /
p99 error was 0.0091 / 0.1493 mm. Rig construction took 1,110.56 seconds.

The resulting 11,909-vertex, 149-morph native package passed the 32-frame
three-mode CUDA benchmark (maximum error 1.49e-8 m), plus a 27-frame named /
random-control comparison (maximum error 1.35e-7 m). GLB, USD skeleton and
USD ZIP checks passed. Eight named poses had no measured eye/sphere/lip
penetrations above 0.05 mm after exact projection. This does not establish
continuous clearance. **Visual review failed:** open-jaw and tongue-out
poses have severely stretched mouth-interior appearance. The fresh rig is a
diagnostic candidate, not a replacement for the saved contact work. Review
`tmp/vhuman-3070-followup/face-rig-check.json` and the new rig's `preview.png`.

Moving-reference validation used existing tr16 recordings, without downloading
motion data. Claire's lateral walk and 90° left turn passed native MHR input
fitting and Terra baseline export checks. Their sampled-policy episodes
(`seed=0`, 100 Hz CPU MuJoCo/JAX) stopped with `tracking_failure` after 0.13
and 0.36 seconds of a 1.86-second reference. Both are rejected. Auto terrain
inferred a 117 mm support block from the lateral walk's elevated ankle;
the render retains that failed inference for review. The input BVH carries
no measured terrain evidence.

AJ's ordinary `Walking.bvh` supplied an additional 87-frame comparison. Its
initial constant-offset foot fit exceeded the 10 mm skin-penetration gate
(11.63 mm). The importer now calibrates a flat floor once after segment-length
retargeting and applies the existing actual-sole constraints. Penetration fell
to 0.55 mm, skin stance drift was 8.06 mm, and site-fitting RMS was 32.19 mm;
the unchanged input gates passed. This flat-floor rollout stopped with
`tracking_failure` after 1.88 of 2.76 seconds. Its baseline avatar export
passed (skin stance drift 3.56 mm, penetration 0.166 mm, fitting RMS 19.94 mm).
The simulated candidate additionally failed the 20 mm stance-drift gate at
23.19 mm; skin-centroid drift was 5.73 mm and penetration 0.523 mm. This take
is rejected too. No episode was reset/stitched and no threshold was relaxed.

All 60 lateral-walk poses also passed native CPU/CUDA MHR mesh comparison:
maximum difference 4.58e-5 cm, identical skeleton states, CUDA decode 0.210 s
versus CPU 21.23 s. These are brief execution timings. Actual baseline and
candidate JSON/GLB exports are checked for all 127 joint rotations, timestamps
and animated root translations. Body regression checks passed 38 tests in the
Terra interpreter; CUDA video/BVH checks passed 23, and MuscleMimic video
checks passed 19. Both registered-worker doctors still verify. These BVH
tests do not establish uploaded-video fitting quality or CUDA JAX inference.

Private receipts, animated GLBs, Blender CUDA baseline renders and the review
index are under `tmp/vhuman-3070-followup/`. `summary.json` records all three
input fits, policy outcomes and interchange checks; `index.html` links the
raw/composite edits, face pose sheet, body renders and diagnostic exports.
No generated appearance, face rig
or simulated motion is promoted. Remaining work is face mouth-interior quality,
Edit skin appearance, moving-policy tracking and heel-lift terrain inference;
it is not blocked by missing
weights or 3070 memory. Blackwell validation still waits for the GPU replacement.

## RTX 3070 quality fixes, 2026-10-16

The native GNM mouth opening now uses neutral anatomical lip boundaries.
Semantic upper/lower transfer prevents the slightly crossed neutral lips from
exchanging skinning or contact assignments, and source LODs retain that split.
Median upper-lip jaw influence fell from 0.8121 to 0.00390. Matched Blender
CUDA GLB and USD renders show teeth and the open mouth instead of stretched
upper-lip skin across the opening. USD eye/cornea material rendering still
differs from GLB; this change does not establish complete material parity.

Corrective training used 4096 samples, 96 PCA components and 2500 epochs;
held-out correction error was 0.0078 mm mean / 0.1261 mm p99. The final GLB,
USD archive and both LODs validate after a fresh export with verified
unchanged training inputs.
Twenty-seven named/random poses agree between CPU and CUDA within 1.35e-7 m
at the existing 2e-6 m tolerance, and eight named poses pass contact checks.
The new shared-vertex regression fails on the previous CUDA kernel and passes
after ordering overlapping lip pairs like the CPU solver. With the concurrent
Edit job active, the 32-frame native benchmark measured 0.867 ms of GPU kernels
with contacts; this is an execution check, not an isolated speed benchmark.

Terra now records synchronous global/core site and root-angle error traces,
initial pose/velocity differences, history length and terminal information.
Fixed-reference sampled/mean comparisons gave these failure times:

| Recording | Sampled | Mean |
|---|---:|---:|
| Lateral walk | 0.13 s | 0.13 s |
| Turn | 0.36 s | 0.50 s |
| Forward walk, explicit flat floor | 1.88 s | 1.16 s |

Root position and qvel match the reference at reset; joint noise is the saved
checkpoint default. History length is one. All six runs still fail tracking.
The mean forward run exceeds the 15 cm core upper-body limit while its global
mean is 12.19 cm; the other five exceed the global limit. Mean actions do not
provide a general fix. No acceptance threshold, reset behavior or physics was
changed to obtain these results.

For automatic terrain, posed whole-foot skin now guards against a heel lift
being interpreted as a raised step. The lateral walk's false 11.7 cm box is
removed; its retargeted baseline passes native skin acceptance, but the policy
still fails at 0.16 s. A source translated onto a real 12 cm platform retains
automatic box reconstruction (12.30 cm in solver scale), completes 94 policy
snapshots / 0.93 s and passes native avatar acceptance: skin stance drift
2.46 mm, penetration 0.143 mm and site fitting RMS 22.31 mm. Flat support is
selected only when both feet have sustained observable stance and the posed
skin stays within 1 cm of the calibrated source floor. Raised platforms,
ramps and insufficient evidence keep the ordinary reconstruction.
The turning clip also selects flat support and passes baseline fitting, but
its policy still terminates at 0.36 s. All three corrected-terrain cases have
matching JSON/GLB joint rotations, timestamps and animated root translations.

The controlled Edit-2511 comparison repeats the independent right view with
40 steps instead of 12: same 1024-square target, raw portrait, original prompt,
negative prompt, CFG 4 and seed 318. The 40-step generation completed in
1496.05 s (55.54 s CPU prompt encoding); the measured run including model
loading and preparation took 1554.80 s. Peak CUDA allocated/reserved memory
was 1.84/2.35 GiB and peak host RSS 47.81 GiB. The 3070 therefore runs this
configuration with one offloaded transformer block and sufficient system RAM.
Skin and ear detail improve substantially over the flat 12-step raw result;
scalp hair remains stylized rather than the requested short stubble.

Both step-count bakes preserve all 421,642 protected composite pixels,
168,661 photographed atlas texels and 428,934 unsupported unobserved atlas
texels byte-for-byte. Geometry is unchanged. The default 10 mm blend gives
seam energy 0.026116 at 12 steps and 0.027402 at 40 steps, versus 0.026199 for
the input. A separate fresh 40-step bake with `--feather-mm 30` reduces that
score to 0.026320 (within 0.46% of the input), retaining 277,252 generated
texels. The ordinary 10 mm default is unchanged. The 30 mm option and its
value are included in the bake receipt. Raw foreground IoU decreases from
0.9964 to 0.9831 at 40 steps; composite protection remains exact. Matched
six-view renders still show incomplete scalp coverage, so neither single-view
material is promoted as a complete whole-head appearance. The independent
40-step image establishes detail quality and 8 GB execution, not new geometry
or hidden-side likeness.

Checks passed 22 existing rig tests, four anatomical-lip tests, one CUDA
shared-contact test, 42 body tests and 14 texture tests (83 total). The private
review is `tmp/vhuman-3070-quality/index.html`, with machine-readable receipts
in `summary.json`. Generated appearance and moving-policy results are not
promoted. Blackwell execution still requires the GPU replacement.

Validation commands from the repository root (logs are retained in the review
workspace):

```sh
. tmp/vhuman-dev/activate.sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
"$PYTHON" -m unittest server.vhuman.test_rig -v
VHUMAN_TEST_CUDA=1 "$PYTHON" -m unittest \
  server.vhuman.test_rig_cuda_contacts server.vhuman.test_rig_lips -v
"$PYTHON" -m unittest server.vhuman.test_mv_texture -v
tmp/terra-runtime/venv/bin/python -m unittest \
  server.vhuman.body.test_terra server.vhuman.body.test_motion \
  server.vhuman.body.test_musclemimic server.vhuman.body.test_terra_support \
  server.vhuman.body.test_tracking_diagnostics server.vhuman.body.test_bvh_reference -v
cuda/vhuman/bench_vhuman_deformer_cuda \
  tmp/vhuman-3070-quality/face-full/rig_deformer.safetensors 32
```

### Moving tracking, shared texture and USD eyes (2026-10-16)

The USD exporter now preserves masked corneal transmission, IOR and normal
strength. In matched Blender Cycles renders, eye-region RGB RMSE against GLB
fell from 63.26 to 14.65 on an 8-bit scale; the white-iris defect is fixed.
LOD0/1/2 retain all nonmaterial prim attributes, animation and bindings
(591/510/508 prims checked). Textures resolve and the regenerated USD archive
validates. The local package is `tmp/vhuman-3070-next/face-usd/rig_usd.zip`.

Moving-policy action traces match upstream deterministic inference exactly.
Reset/noise, observation refresh, joint-equality and smoothing experiments did
not produce a complete moving rollout. The platform control completes 0.93 s;
lateral/forward mean rollouts still stop at 0.16/1.16 s. A newly located
standing-to-walking recording also fails a direct BVH-to-Myo diagnostic at
1.36 s, while its MHR import fails the original stance-drift gate. These
results and the unmodified thresholds are detailed in [body/TERRA.md](body/TERRA.md).

The experimental `qwen_edit_grid` backend generated front/right/back/left in
one 40-step inference on the 3070. Generation took 1511.35 s (1580.97 s including
loading), with peak CUDA allocated/reserved memory 1.845/2.354 GiB and host RSS
47.663 GiB. The raw sheet is more uniform across views but loses realistic
skin detail and paints pale, patterned stubble; it is not promoted. See
[TEXTURE_CUDA.md](TEXTURE_CUDA.md) for reproduction and explicit pole inference.

Checks pass 22 existing rig tests plus 19 texture/material tests. The private
review entry point is `tmp/vhuman-3070-next/index.html`; `summary.json` retains
the generated-view checks, bake metrics, policy outcomes and package receipts.
Blackwell `sm_120` execution remains pending replacement of the 3070.

```sh
. tmp/vhuman-dev/activate.sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
"$PYTHON" -m unittest server.vhuman.test_rig -v
"$PYTHON" -m unittest server.vhuman.test_mv_texture \
  server.vhuman.test_texture_grid server.vhuman.test_rig_usd_materials -v
```

### Source-root walking, animated interchange and retained surface repair

The next 3070 continuation completes a reproducible direct-BVH walking fixture
through the constrained Terra policy and native avatar acceptance. The
source-root objective preserves the measured trajectory in the solve; native
export retains the simulated root exactly. The final command completes 280
snapshots / 2.79 s, with 6.48 mm skin stance drift, 0.508 mm penetration and
21.19 mm anatomical RMS under the original limits. JSON/GLB tracks match.
The four-second stationary hold and raised-platform control also complete.
The old intermediate MHR forward reference and standing-to-walking source
remain rejected. See [body/TERRA.md](body/TERRA.md) for the direct command,
root-objective provenance and scope of this bounded validation.

Combined blink/gaze/jaw stress exposed Blender shape-key slider clipping in
both USD and GLB imports. The animation curves preserve signed weights, but
their evaluated ranges needed expansion. The new import wrapper preserves
curve samples, packs the scene and frames its actual animation interval.
After correction, all three LODs agree within 0.57 micrometres across eight
samples and two lighting/camera settings (48 matched pairs). Eye material
rendering remains approximate; export parity does not certify lid anatomy or
native exact contacts. See [rig/README.md](rig/README.md).

The retained surface material receives a conservative tone correction with
21.12% lower internal tone-jump p95 and 3.84% lower boundary seam score. All
168,661 photographed texels and geometry/support/normal/ORM files remain exact.
The stronger comparison and matched five-view Blender renders are retained;
scalp hair remains incomplete. See [BLENDER_TEXTURE.md](BLENDER_TEXTURE.md).

This continuation passes 58 tests: 46 body, four USD material, two Blender
shape-range and six surface-texture tests. Two optional body fixtures are
skipped; real policy/native-avatar acceptance is checked separately above.
The review is `tmp/vhuman-3070-finalpass/index.html`, with detailed receipts in
`summary.json`. Data and generated assets remain private under ignored `tmp/`.
Blackwell `sm_120` execution still requires the GPU replacement.

### Isolated pre-upgrade runtime baseline

The retained 3070 baseline uses the corrected full face rig (11,909 native
vertices, 149 morphs), three combined blink/gaze/jaw GLB LODs, the accepted
walking clip, and a GLB of the selected repaired skin. The skin export preserves
its base-color pixels exactly. Assets, native sources, compiler flags, seeded
controls, driver and browser versions are hash-bound in the private receipts.
No concurrent GPU jobs run during either measurement.

Hardware/settings: RTX 3070 8 GiB / `sm_86`, existing 115 W power limit,
NVIDIA 580.178.04, NVRTC 12.9, GCC 13.3, Ryzen 9 3950X. No clocks or power settings
are changed. Each native case has 16 warmups and at least three seconds of
samples. Single-frame timings, including host rig, upload, synchronization and
readback, are:

| Deformation | Wall p50 | Wall p95 | GPU kernel p50 |
| --- | ---: | ---: | ---: |
| ML + four contact iterations | 0.897 ms | 0.916 ms | 0.840 ms |
| ML only | 0.145 ms | 0.156 ms | 0.089 ms |
| Linear only | 0.131 ms | 0.138 ms | 0.089 ms |

The ML/contact 32-frame batch takes 1.539 ms median including readback, or
1.012 ms with device-resident output. CPU/CUDA parity across all three modes
and both batch sizes is within `1.49e-8 m`. Sampled allocated VRAM peaks at
273.3 MiB for the native process plus desktop, 197.9 MiB above its idle baseline.
Driver/firmware reservation is recorded separately; this is NVML sampling at
100 ms, not an allocator high-water mark. Initialization may reuse the existing
NVRTC kernel cache and is recorded separately from warmed call latency.

Five sequential 60-second hardware playbacks (five-second warmup each) sustain
60.00 display FPS at 1280 x 720, DPR 1 and antialiasing in Chromium 155 /
NVIDIA ANGLE Vulkan with Three.js r163. All frame-interval p95 values are
16.7 ms. FPS is browser/display limited, so the GPU timer measurements below
are retained for the upgrade comparison:

| Playback asset | GPU p50 | GPU p95 | Peak allocated VRAM |
| --- | ---: | ---: | ---: |
| Face stress LOD0 | 3.680 ms | 4.035 ms | 1272.8 MiB |
| Face stress LOD1 | 4.146 ms | 4.371 ms | 1244.7 MiB |
| Face stress LOD2 | 3.261 ms | 3.480 ms | 1191.9 MiB |
| Walking, full-body framing | 0.784 ms | 0.792 ms | 1028.6 MiB |
| Repaired skin turntable | 0.073 ms | 0.079 ms | 274.6 MiB |

The camera frames the animated avatar rather than the terrain. A single
empty-page retry resolves Chromium's initial Vulkan context replacement and
is recorded; there are no context losses, disjoint timers or console errors
during measurement. Every asset releases its geometry, material and skeleton
resources, returning to zero geometries/programs and two shared renderer
textures. After browser exit the device returns to 76 MiB / 0% utilization.
This measures exported GLB animation/PBR rendering; native exact contacts,
Terra policy inference and USD playback are separate workloads.

Reproduction and timing scopes are documented in
[rig/README.md](rig/README.md#gpu-comparison-baselines). The private review is
`tmp/vhuman-3070-baseline/index.html`, with `summary.json`, raw `native-final/`
and `playback-final/` receipts, and `repeat.py` to verify retained asset hashes
and rerun the same workloads after the GPU swap. `sm_120` execution remains
pending replacement hardware.
