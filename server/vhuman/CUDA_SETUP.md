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
suite checks the recorded combination. Terra retains its own upstream lock.

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
GPU inference and GPU parity checks have not been run in this setup session.
