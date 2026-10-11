# Offline MuscleMimic body motion (research preview)

The optional `musclemimic` backend refines short motion in a muscle-actuated
MyoFullBody simulation, fits the result to an existing MHR avatar, and exports
ordinary JSON and animated GLB. The HTTP server and browser do not import JAX,
MuJoCo, GMR, or Torch. The existing `sam3d` backend remains the default.

Reference motion and uploaded video use the same policy and export path.
Video is limited to one fully visible, upright person, a fixed camera, flat
ground, and the first eight seconds (at most 240 source frames, up to 30 Hz).
Root travel from video is an estimate. HOG person detection and background
optical flow deliberately reject ambiguous or unobservable input; they do not
prove that every visible limb has been reconstructed correctly.

This is an experimental research integration. Acceptance means the numerical
checks passed; every take still needs perceptual review. Training, live physics,
muscle-driven skin deformation, and cloth simulation are outside this backend.

## Sources and asset boundary

- [MuscleMimic](https://github.com/amathislab/musclemimic), pinned to
  `47b653db2bce25de834360d2ee6b2dd88142bd78`.
- [GMR+](https://github.com/amathislab/gmr_plus), pinned to
  `1b4eb9864360fa822b1290f34592da6393a4d496`.
- [Full-body base policy](https://huggingface.co/amathislab/mm-fullbody-base),
  tested at `0e308c82bee9bd8192e6e205ed70cca377831cd5`.

No models, mocap, or derived datasets are committed or downloaded during jobs.
Registration is explicit and records local asset hashes, revisions, package
versions, and reference source/terms URLs. Jobs verify those receipts before
loading assets. Keep licensed reference clips and their derivatives private.
The Apache source license does not replace the separate model and dataset terms;
review upstream model NOTICE and any AMASS/source-dataset terms before use.
This adapter marks its receipts and output as noncommercial research.

## Isolated installation

Run from the repository root. Use Python 3.12 and keep caches in repo `tmp/`:

```sh
mkdir -p tmp
export TMPDIR="$PWD/tmp"
export UV_CACHE_DIR="$PWD/tmp/musclemimic-cache"
uv venv --python /usr/bin/python3.12 tmp/vhuman-musclemimic-py312
git clone https://github.com/amathislab/musclemimic tmp/musclemimic-upstream
git -C tmp/musclemimic-upstream checkout 47b653db2bce25de834360d2ee6b2dd88142bd78
git clone https://github.com/amathislab/gmr_plus tmp/vhuman-gmr-upstream
git -C tmp/vhuman-gmr-upstream checkout 1b4eb9864360fa822b1290f34592da6393a4d496
uv pip install --python tmp/vhuman-musclemimic-py312/bin/python \
  torch==2.14.1+cpu --index-url https://download.pytorch.org/whl/cpu
uv pip install --python tmp/vhuman-musclemimic-py312/bin/python \
  -c server/vhuman/body/musclemimic-constraints.txt \
  -e tmp/musclemimic-upstream -e tmp/vhuman-gmr-upstream
```

GMR imports Torch utilities even though this adapter feeds it native MHR
transforms and requires no SMPL model files. Keep the framework environment
separate from the native SAM/MHR programs and server environment. CPU physics
works without a GPU. `--musclemimic-device cuda` selects JAX policy inference
only and requires the matching JAX 0.7.2 CUDA plugin installation; it has not
been validated here. It does not select a MuJoCo GPU physics implementation.
Use the pipeline's `--backend cuda --device N` with CUDA policy inference.
The parent checks and locks physical GPU N; the worker restricts CUDA visibility
to that GPU before loading JAX, so JAX's logical GPU 0 uses the locked device.

Obtain the complete base checkpoint at the revision above into a local directory
after reviewing its terms. Register it with the isolated interpreter:

```sh
tmp/vhuman-musclemimic-py312/bin/python -m server.vhuman.body.musclemimic_assets register \
  --assets tmp/vhuman-musclemimic-assets \
  --upstream tmp/musclemimic-upstream --gmr tmp/vhuman-gmr-upstream \
  --checkpoint tmp/vhuman-musclemimic-checkpoint \
  --checkpoint-revision 0e308c82bee9bd8192e6e205ed70cca377831cd5 \
  --accept-research-terms
tmp/vhuman-musclemimic-py312/bin/python -m server.vhuman.body.musclemimic_assets doctor \
  --assets tmp/vhuman-musclemimic-assets
```

The receipt is not a license grant. Registration will not overwrite an existing
installation. Use a new asset directory when changing source or packages.

## Reference format

References are numerical NPZ files loaded with `allow_pickle=False`:

| Field | Shape / meaning |
|---|---|
| `qpos` | `[frames, nq]`, MuJoCo world coordinates, metres, Z-up |
| `frequency` | scalar Hz, 1–1000 |
| `joint_names` | Unicode array, one name per joint |
| `jnt_type` | MuJoCo enum: free=0, ball=1, slide=2, hinge=3 |
| `split_points` | `[0, frames]`; exactly one episode |

Free and ball quaternions are WXYZ and normalized. Every checkpoint-model joint
must be present with the matching type. Extra disabled finger joints are mapped
out by name. Convert licensed upstream datasets to this small format locally;
the importer never loads pickled metadata or silently guesses joint ordering.

```sh
tmp/vhuman-musclemimic-py312/bin/python -m server.vhuman.body.musclemimic_assets add-reference \
  --assets tmp/vhuman-musclemimic-assets --id walk \
  --trajectory /path/to/licensed/walk.npz \
  --source-url https://example.org/source --terms-url https://example.org/terms \
  --accept-research-terms
```

## CLI and HTTP

Provide the same three worker options to `server.vhuman.app` or the CLI.
`--sam3d-body-model` must contain the existing native SAM and MHR `safetensors/`
assets, including exported MHR rig metadata. Reference jobs need only its MHR
assets; video additionally needs the native SAM inference assets.
Both `ffmpeg` and `ffprobe` must be on the server's `PATH` for video sampling.
Camera checks span the clip and include its final frame. Every scheduled check
must have enough reliable background tracks; missing coverage rejects the input.

```sh
python -m server.vhuman.cli --work /path/to/vhuman-work \
  --sam3d-body-model /path/to/sam3d-body \
  --musclemimic-python "$PWD/tmp/vhuman-musclemimic-py312/bin/python" \
  --musclemimic-assets "$PWD/tmp/vhuman-musclemimic-assets" \
  --musclemimic-device cpu \
  body-motion --head HEAD_ID --backend musclemimic --reference-motion walk
```

For video, replace `--reference-motion walk` with
`--media /path/to/clip.mp4 --fixed-camera`. The global `--backend` selects native
inference hardware; the `body-motion --backend` selects the motion algorithm.

The rig viewer exposes **MuscleMimic**, the registered reference list, fixed-camera
confirmation, baseline comparison, and an evaluation report. The corresponding
job request uses the existing body-motion endpoint and one source:

```json
{"kind":"body_motion","head_id":"HEAD_ID","backend":"musclemimic","reference_motion":"walk"}
```

```json
{"kind":"body_motion","head_id":"HEAD_ID","backend":"musclemimic","upload_id":"UPLOAD_ID","fixed_camera":true}
```

`GET /v1/body/reference-motions` returns availability and registered IDs.
Submit the requests above to `POST /v1/jobs`.
Jobs snapshot avatar inputs, support cancellation/timeouts, and publish through
an atomic directory rename. A rejected take has no `motion.json`/`motion.glb`
download URLs. Its `manifest.json`, `evaluation.json`, and any completed baseline
remain available. Raw simulator arrays, source poses, and rejected candidates
are retained locally for diagnosis and are not served through the file route.

## Retargeting and evaluation

1. Preserve source timestamps; use one fixed identity for dense native MHR poses.
2. Estimate video ground alignment and root travel using camera translation and
   foot-contact constraints. Convert Y-up MHR into Z-up MuJoCo with a proper rotation.
3. Calibrate 17 anatomical sites, scale segment lengths, and solve GMR IK. Record
   the resulting named Myo trajectory; reject a maximum IK position error >15 cm.
4. Load the exact pretrained configuration, run one deterministic episode at its
   control timestep, and stop at failure or the selected interval end. Never reset
   and stitch episodes. Snapshot qpos, qvel, commands, activations, and contacts
   at the same post-step instant.
5. Fit body angular parameters and root translation to the final avatar. Preserve
   its joint translations, non-unit scales, MHR procedural coupling, and source
   finger controls where available. Facial joints remain independent.
6. Check both skeletal anchors and actual GLB foot vertices. Current gates require
   stance samples, <=2 cm 95th-percentile stance drift, <=1 cm penetration, and
   <=5 cm RMS anatomical fit error. A completed episode must pass these gates.
   Baseline deltas are reported; acceptance does not assert perceptual improvement.

Version-2 motion is parent-local XYZW rotation plus an absolute parent-local
`mhr_root` translation in metres, Y-up. JSON timestamps start at zero and are
strictly increasing. Quaternions are normalized and hemisphere-continuous. GLB
uses the same samples with LINEAR interpolation; the viewer still accepts v1.

## Validation

```sh
TMPDIR="$PWD/tmp" \
MHR_TEST_ASSETS=/path/to/native/mhr/assets \
MHR_TEST_AVATAR=/path/to/head/body \
tmp/vhuman-musclemimic-py312/bin/python -m unittest \
  server.vhuman.body.test_motion server.vhuman.body.test_musclemimic -v
```

Unit tests cover coordinate bases, quaternion continuity, JSON/GLB root parity,
scaled FK and procedural coupling, contact metrics, termination, source checks,
timestamp-preserving video sampling, cancellation, and rejected artifact listing.
The optional native test compares actual MHR joint
positions/rotations against the C++ decoder. Local CPU smoke tests additionally
exercised the real checkpoint, 89-qpos/88-qvel/354-actuator environment, trajectory
cache, policy inference, simulator recording, and GLB export.

To repeat the real CPU tests, add
`MUSCLEMIMIC_TEST_ASSETS=/path/to/registered/assets` to the environment above and
run `server.vhuman.body.test_musclemimic_integration`. This verifies named reference
import, MHR-to-GMR IK, nonzero muscle activation, and portable export with the
actual policy. It needs several GB of RAM and an environment that supports
Python asynchronous filesystem operations (some restricted sandboxes block
Orbax's thread wakeups).

A generated neutral standing fixture is a mechanics test, not a mocap quality
benchmark. The three real CPU integration tests pass, including nonzero
activations and end-to-end export. Licensed walking/turning clips,
real uploaded-video reconstruction, CUDA inference, and the full two-avatar
perceptual comparison matrix remain to be validated before production use.

Run that matrix explicitly with `python -m
server.vhuman.body.musclemimic_evaluate`, passing the same worker/model options,
`--work`, `--matrix matrix.json`, and `--output evaluation.json`. The matrix is:

```json
{"heads":["AVATAR_A","AVATAR_B"],
 "references":["walk","turn","stand"],
 "videos":["walk.mp4","turn.mp4","stand.mp4"]}
```

Video paths are relative to the matrix file. This runs twelve local jobs,
retains each take, and updates the aggregate JSON after every result. Exit 2
means at least one take was rejected/failed. Review baseline versus simulated
motion for sliding, penetration, balance, hands, and face attachment on both
body proportions; record that review separately from numerical acceptance.
