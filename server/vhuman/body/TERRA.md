# Terra motion on the native MHR avatar

The optional `terra` backend reconstructs supporting terrain, retargets native
MHR landmarks to MyoFullBody, runs a terrain-conditioned PPO policy, and fits
the recorded motion back to the avatar. It supports floors, steps and ramps.
SAM 3D Body/MHR is the default source representation. The worker runs without
`smplx`, SMPL-H/X/MANO assets or an optimized SMPL shape cache.

This is a research preview. Numerical acceptance and perceptual quality are
separate. Policy training, seats, live physics, cloth simulation and dynamic
MHR mesh correctives are not included. Existing `sam3d` and `musclemimic` jobs
retain their defaults and interfaces.

## Isolated installation

From the repository root:

```sh
python3 ref/terra/setup.py --root tmp/terra-runtime --install
# Add --cuda to install the pinned JAX CUDA plugin for policy inference.
```

The installer stages immutable Terra, MuscleMimic and Holosoma revisions,
applies the two maintained patches in `ref/terra/`, and installs the upstream
locked package closure without `smplx`, overriding OpenCV to 4.13.0.92 for the
HOG person detector used by video preflight (OpenCV 5 lacks that Python API).
The upstream source and lockfile remain unchanged. It uses Python 3.11 and repository-local
temporary/cache directories. The lockfile selects PyTorch CUDA 12.6 on Linux
x86-64 even for CPU inference; allow about 8 GB for the environment. It does
not download any model, policy checkpoint or motion recording.

Use a separate upstream/reference environment for licensed SMPL comparisons.
Do not add `smplx` to the default worker or modify the existing MuscleMimic
environment. Patch/source hashes and critical package versions are verified
before execution.

## Assets and terms

Register an existing complete local Orbax checkpoint, including its metadata,
saved configuration and state. The intended policy is
[TERRA-4B](https://huggingface.co/merc-s/TERRA-4B), `checkpoint_24416`; upstream
documents tested revision `b89a604d0687549f2678bb1b6a436dea058cd8b9`. A compatible
user-provided Terra MyoFullBody checkpoint is also supported. Its control rate
must be 100 Hz. Preserve the publisher's model files outside source control.

The current TERRA-4B card does not declare a checkpoint license. Registration
requires a source/revision and an explicit terms or permission record. A terms
URL or receipt is not a license grant. Do not assume Terra's Apache source
license applies to the policy or its training recordings. MHR is Apache-2.0;
SAM 3D Body has separate SAM license terms. Licensed reference recordings and
their derivatives remain private.

```sh
tmp/terra-runtime/venv/bin/python -m server.vhuman.body.terra_assets register \
  --assets tmp/terra-assets \
  --terra tmp/terra-runtime/terra \
  --musclemimic tmp/terra-runtime/musclemimic \
  --holosoma tmp/terra-runtime/holosoma \
  --mhr-assets /path/to/native/mhr/assets \
  --checkpoint /path/to/checkpoint_24416 \
  --checkpoint-revision CHECKPOINT_REVISION \
  --checkpoint-source-url SOURCE_URL --checkpoint-terms-url TERMS_OR_PERMISSION_URL \
  --accept-checkpoint-terms

tmp/terra-runtime/venv/bin/python -m server.vhuman.body.terra_assets doctor \
  --assets tmp/terra-assets
```

For the currently published TERRA-4B card with unspecified licensing, replace
`--accept-checkpoint-terms` with `--acknowledge-unspecified-checkpoint-terms` and
use the publisher's model card as the provenance URL. This records local
research validation with `checkpoint_license_status:"unspecified"`; it does
not claim that the publisher granted a license.

The native MHR directory must contain the four `mhr.ASSET_FILES` exports,
including rig metadata. It must match the avatar and any registered reference.
Jobs verify the complete checkpoint, patches, sources, packages and MHR asset
fingerprint and perform no downloads.

## MHR references

References are bounded NPZ files, loaded with `allow_pickle=False`:

| Field | Format |
|---|---|
| `model_params` | `[T,204]` native MHR controls; translation/scale columns fixed across the take |
| `identity` | `[45]`, one fixed native MHR identity |
| `times` | `[T]`, seconds, starts at zero and strictly increases |
| `root_world` | `[T,3]`, world position of `mhr_root`, metres, Y-up |
| `mhr_fingerprint` | Unicode scalar SHA256 from `terra_motion.asset_fingerprint()` |

The source interval is at most eight seconds and 5–240 samples; the archive
and its expanded arrays are limited to 4 MiB. World-root positions replace
decoded root translations. They are not camera translations or parent-local
animation translations. References must already have a calibrated ground datum.

Existing short BVH recordings with the Mixamo anatomical joint names can be
converted with the native fixed-identity MHR fitter:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
tmp/vhuman-cuda-venv/bin/python -m server.vhuman.body.bvh_reference \
  --source /path/to/walk.bvh --avatar /path/to/avatar-directory \
  --mhr-assets /path/to/native/mhr/assets --out tmp/walk-reference \
  --source-url SOURCE_URL --terms-url TERMS_URL
```

The importer accepts 5–240 frames spanning at most eight seconds, preserves
declared Euler channel order and defaults to centimetres (`--unit-scale .01`).
It records source/output hashes, anatomical mapping, stature scaling and
contact/skin fitting metrics in `provenance.json`. The BVH importer assumes
a flat floor and fits pose-aware sole constraints against actual avatar skin.
Inspect `fitting.accepted`
and review the motion before registering `reference.npz`. This is an input
conversion, not a policy rollout or proof of motion quality. Keep recordings
and derived references outside source control with their source terms.

```sh
tmp/terra-runtime/venv/bin/python -m server.vhuman.body.terra_assets add-reference \
  --assets tmp/terra-assets --id walk --motion /path/to/walk-mhr.npz \
  --source-url SOURCE_URL --terms-url TERMS_URL
```

## CLI and HTTP

```sh
python3 -m server.vhuman.cli --work tmp/vhuman-independent \
  --sam3d-body-model /path/to/sam3d-body \
  --terra-python "$PWD/tmp/terra-runtime/venv/bin/python" \
  --terra-assets "$PWD/tmp/terra-assets" --terra-device cpu \
  body-motion --head HEAD_ID --backend terra --reference-motion walk \
  --terrain auto --policy-actions sampled --seed 0
```

For video, replace `--reference-motion walk` with `--media clip.mp4
--fixed-camera`. Videos require one fully visible upright person and a textured
stationary background. Native SAM inference uses the ordinary inference backend;
physics uses CPU MuJoCo. `--terra-device cuda` selects JAX policy inference only
and requires the global native `--backend cuda`. ROCm policy inference is not
supported.

`--terrain` accepts `auto`, `flat`, or a canonical TerrainSpec JSON file. Known
geometry is in the **source** world frame, metres and Z-up; the proper Y-to-Z
rotation is `[x,y,z] -> [x,-z,y]`. Terra boxes use `pos`, `size` (half-extents),
`yaw`, `pitch` and `name`. Video world motion is estimated, not measured; terrain
or camera/contact inconsistency rejects the take rather than flattening it.

For `auto`, posed native foot skin provides a conservative flat-floor check.
Both feet must have at least five consecutive low-speed stance samples, and
the lowest skin point of each whole foot must remain within 1 cm of the
calibrated source floor during every qualifying stance. A raised heel with
toes on the floor therefore does not create a step. Raised supports, ramps
and insufficient observations retain the ordinary terrain reconstruction.
The evidence and selected mode are saved as `terrain_selection`; explicit
terrain requests are unchanged.

Submit the existing `POST /v1/jobs` endpoint with:

```json
{"kind":"body_motion","backend":"terra","head_id":"HEAD_ID","reference_motion":"walk","terrain":"auto","policy_actions":"sampled","seed":0}
```

For uploaded video, use `upload_id` instead of `reference_motion` and include
`fixed_camera:true`. HTTP `terrain` accepts `auto`, `flat` or an inline
TerrainSpec object; server filesystem paths are rejected. The reference catalog
is `GET /v1/body/reference-motions?backend=terra`. The rig viewer exposes the
same source, terrain, action and seed controls.

## Outputs and acceptance

Each take retains private source arrays, Terra trajectory/analysis artifacts,
canonical `solver-terrain.json`, reference and rollout state, muscle commands
and activations, and reconstruction/stability/provenance reports. The policy
runs once, with a recorded sampling seed, stopping on reference completion or
the first failure. It never resets and stitches episodes or relaxes checkpoint
tracking thresholds to obtain an accepted result.

`tracking-errors.npz` records synchronous global/core mean site error and root
orientation error over time. `simulation.json` includes their final values,
checkpoint limits, initial pose/velocity differences, observation history
length and terminal information. These are diagnostics only. In particular,
the core upper-body threshold can terminate an episode while the global
mean remains below its threshold. Mean and sampled actions can be compared
with `--policy-actions` using the same reference and seed.

Accepted `motion.json` uses the existing v2 local XYZW rotations and animated
parent-local `mhr_root`. Its optional `terrain:"terrain.json"` links to transformed
Y-up box geometry. The same geometry is embedded as static nodes in animated
GLB and shown beside the avatar in the viewer. `baseline.json`/`baseline.glb`
contain the retargeted reference without policy simulation. Geometry transforms
and source interval trimming are recorded explicitly. Source coordinates are
centered once horizontally, with the identical transform applied to explicit
terrain. Box names are normalized to the collision system's `terrain_box`
prefix. The trimmed point-timestamped interval is resampled to 100 Hz without
stretching; an incomplete final control step is omitted.

Acceptance requires a complete episode, the upstream stability gate, contact
evidence, <=2 cm 95th-percentile stance drift, <=1 cm support penetration and
<=5 cm RMS anatomical fitting error. Foot checks include actual exported skin
vertices and support normals. Rejected takes expose baseline and evaluation
when available, without accepted motion download URLs. Cancellation publishes
nothing. Every accepted take still requires perceptual review.

## Checks

```sh
TMPDIR="$PWD/tmp" tmp/terra-runtime/venv/bin/python -m unittest \
  server.vhuman.body.test_terra server.vhuman.body.test_motion \
  server.vhuman.body.test_musclemimic -v
```

Portable tests cover reference bounds, camera/contact elevation preservation,
step/ramp geometry, tangent stance drift, raised-support penetration, GLB/root
parity, rejected publication and CUDA locking. Real policy/avatar validation
requires the registered checkpoint and native rig. Synthetic fixtures establish
mechanics, not the perceptual quality of reconstructed walking or stair clips.

Local validation used the public, ungated `checkpoint_24416` download, with all
12 publisher SHA256SUMS entries verified, and an isolated environment without
`smplx`. The portable/native suite ran 34 tests, with one optional simulator
fixture skipped; the selected upstream pipeline, retargeting and stability
suite passed 181 tests. HTTP catalog, private diagnostics and terrain dispatch
checks passed, as did viewer JavaScript syntax checking. A native standing
fixture completed policy rollout and passed exported skin checks on two MHR
shapes. The final flat-floor run recorded 94 snapshots over 0.93 s; exported
skin stance drift was 1.55 mm, maximum penetration 0.073 mm and site-fitting
RMS 22.0 mm. The second shape's maximum penetration was 0.126 mm.
A translated source standing on a 12 cm platform also completed and passed:
skin stance drift 1.72 mm, maximum penetration 0.127 mm and site-fitting RMS
22.1 mm. Both final runs include animated GLB with the paired terrain geometry.
Walking, real video terrain reconstruction and visual quality still
need representative review.
