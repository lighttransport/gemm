# vhuman full-body avatar

`body` extends an existing **rigged** vhuman head into one skinned avatar:

1. Qwen-Image 2.1 makes a 768 × 1280 transparent, full-length A-pose image,
   conditioned on that head's portrait. A silhouette gate rejects tiny or
   cropped people before reconstruction.
2. The local SAM 3D Body runner extracts a full-body MHR mesh with an explicit
   alpha-derived bounding box. Its JSON sidecar includes the decoded 204 pose
   and 45 identity parameters needed to reproduce the *same* MHR bind pose.
3. Pixal3D reconstructs a textured full-body reference. The source photograph
   supplies visible texels; Pixal3D supplies colour for unseen body surfaces.
4. The existing facial skin, facial controls, expression morphs, ML deformer and
   wrinkle maps attach to MHR's head joint using facial landmarks. The MHR
   head surface is trimmed at the neck; the fitted facial neck blends toward
   the body surface.
5. Optional SAM 3 masks isolate garments for separate Pixal3D meshes. Each
   mesh is fitted to the MHR bind pose, inherits nearby skin weights, and is
   rejected if registration is poor. The textured body remains usable.

The [MHR native rig](https://github.com/facebookresearch/MHR) is the body
template because [SAM 3D Body](https://github.com/facebookresearch/sam-3d-body)
predicts that rig directly. A second automatic rig fit would discard its
joint hierarchy and weights. [Pixal3D](https://github.com/TencentARC/Pixal3D)
provides appearance and extra geometry; its meshes do not carry a body skin.

## Rigging decision

Use the MHR template predicted by SAM 3D Body as the animation skeleton. The
local MHR model exposes the same 127-joint hierarchy, per-vertex skin weights,
identity shape and predicted pose. Re-evaluating those parameters yields bind
transforms matching the extracted mesh (the CPU sample differed by at most
1.07 micrometres). The facial rig can therefore attach below `mhr_c_head`
without fitting a second body skeleton. The exporter retains the MHR joint
scales; omitting them changes the bind pose.

For separate clothes, project the image mask onto the MHR surface, align the
Pixal3D reconstruction to that region, then transfer nearby MHR skin weights.
This makes clothes follow the body in the same GLB and USD skeleton. Registration
is gated by surface distance and falls back to the painted body mesh when it is
poor. A general auto-rigger remains useful for characters with no MHR estimate,
but adds a second skeleton fit and an uncertain animation retarget for this
pipeline. For high quality body animation, the next step is runtime MHR pose
correctives and garment collision handling; interchange exports currently bake
the predicted pose correctives into the bind mesh.

Run from the repository root:

```sh
python3 -m server.vhuman.cli body --head HEAD_ID --quality standard \
  --outfit "blue shirt, dark trousers and shoes" --garments shirt,pants,shoes
```

`--qwen-preset auto` is the default: it selects `low8` when less than 15,000 MiB
of GPU memory is free, or `fast12` otherwise. The CLI, job request and rig page
also accept an explicit `low8` or `fast12` choice.

The server accepts `POST /v1/jobs` with `{"kind":"body","head_id":"..."}`
and optional `outfit`, `quality`, `seed`, `steps`, `qwen_preset`, and `garments`
fields. The rig page has the same controls and can preview body poses with
facial speech.
The local body checkpoint directory defaults to `/mnt/disk1/models/sam3d-body`;
override it with `--sam3d-body-model` on the CLI or server. Optional garment
segmentation uses a separate SAM 3 checkpoint (`--sam3-model`, default
`/mnt/disk1/models/sam3/sam3.model.safetensors`) and CLIP tokenizer directory
(`--clip-bpe`, default `/mnt/disk1/models/clip-bpe`, containing `vocab.json`
and `merges.txt`). Pass these options before `body` for the CLI, or when
starting the server. Missing garment assets are recorded as a body-texture
fallback. The model files remain outside the repository and are not copied
into avatar exports.

The published `heads/<id>/body/` directory contains `avatar.glb`,
`avatar.usda`, `avatar_usd.zip`, `avatar.json`, `body_report.json`, the body
image, and intermediate SAM/Pixal outputs. The combined GLB and USD retain
the 127 MHR body joints and the 12 facial joints. MHR pose correctives are
baked at the generated bind pose; body playback in interchange formats uses
linear skinning. Single-view images cannot determine hidden garment geometry,
so the build report records each garment's fit or fallback.

GPU-free export smoke test (requires a previously rigged head and the local
MHR model):

```sh
python3 -m server.vhuman.cli --mock body --head HEAD_ID --quality preview
```

## Body model: MHR (default) or SOMA-X

`body --body-model {mhr,soma}` (or `"body_model"` in the `/v1/jobs` request)
chooses the body mesh and skeleton. SAM 3D Body always predicts MHR
parameters; with `soma` they are converted to
[SOMA-X](https://github.com/NVlabs/SOMA-X) (Apache-2.0): the MHR identity and
scale carry over directly and the pose is recovered by SOMA's analytic pose
inversion. The result is the 18,056-vertex SOMA mesh with 78 joints
(`soma_Root`, `soma_Hips`, ..., metres, Y-up), the facial rig attaches to
`soma_Head`, and the neck cut uses `soma_Neck1`. SOMA has no facial
blendshapes, so faces still use the vhuman rig.

Setup (torch and Warp stay out of the framework-free rig interpreter):

```sh
server/vhuman/body/setup_soma.sh tmp/vhuman-soma-venv
export VHUMAN_SOMA_PYTHON=$PWD/tmp/vhuman-soma-venv/bin/python
python3 -m server.vhuman.cli body --head HEAD_ID --body-model soma
```

`VHUMAN_SOMA_ASSETS` points at a local SOMA-X asset directory (default: the
Hugging Face cache); `VHUMAN_SOMA_DEVICE=cuda` runs the converter on a GPU.
`body-motion` reads the model from the avatar's `avatar.json` and supports the
`sam3d` backend only; `terra` and `musclemimic` need MHR avatars.

Quality notes (measured on one real SAM 3D Body subject with the same head
rig for both models):

- The MHR to SOMA fit is gated on its mean vertex residual (at most 1 cm); the
  maximum is reported as `mhr_parity_max_m`. The subject gave 6.7 mm mean,
  3.0 cm at the 99th percentile and 5.6 cm maximum. The maximum is inside the
  head (mouth and eye bags, which SOMA excludes from pose inversion and the
  vhuman face rig replaces anyway); hips, spine and legs sit 1.5 to 3 cm off
  the MHR surface because SOMA's MHR identity reproduces the body shape only
  approximately. SOMA's `autograd_iters` refinement does not reduce this (mean
  8.7 mm with 30 iterations), so the default stays at 0.
- Face landmarks use fixed SOMA vertices (`soma_landmarks.json`, regenerated
  with `soma_runner.py --landmarks-from`). They add about 2 mm to the face/body
  alignment error over MHR's own keypoints (27.5 mm vs 25.8 mm here); the
  error itself reflects how well the head matches the body (the gate is 35 mm).
- Garment fitting is topology-agnostic. Shirt and pants were accepted with fit
  scales within 2% of the MHR run, but SOMA garment fitting has been checked on
  this one subject only.
- The MHR path produces identical body joints (bind transforms within 2e-7) and
  mesh (within 1 micrometre) before and after the body-model refactor.

Tests: `python3 -m server.vhuman.test_all` includes `body/test_*.py` (except
the Terra tests, which need mujoco in the Terra interpreter); the SOMA runner
tests additionally need `VHUMAN_SOMA_PYTHON`.

## Fit uploaded pose or motion

For floor/step/ramp reconstruction and offline policy refinement with native
MHR sources, see [Terra setup and motion validation](TERRA.md). The optional
`terra` backend does not require SMPL body assets and keeps its Python stack
separate from the existing MuscleMimic backend.

Open `/rig`, select a head with a full body avatar, and choose a full-body
PNG/JPEG/WebP image or MP4/WebM/MOV video in **Fit pose or motion**. The server
stores at most 64 MiB per upload. For video it samples at 4 fps, up to 32
frames from the first 8 seconds. A single visible person should occupy most
of the frame. Transparent images use the alpha bounds; opaque images use an
OpenCV HOG person box when found, and otherwise use the whole frame.

The job evaluates SAM 3D Body per frame and decodes its MHR pose using the
avatar's original identity shape. It retargets 127 local body-joint rotations;
the facial rig and its speech controls stay attached to `mhr_c_head`. The
viewer offers playback, scrubbing, and simultaneous facial animation. It also
exports `motion.json` and a posed or animated `motion.glb` in
`heads/<id>/body/motions/<take>/`. The GLB preserves the existing skin and
textures. Root translation stays at the avatar bind location; moving-camera
depth estimates are not interpreted as world motion. Per-frame pose estimates
receive mild quaternion smoothing, but large occlusions or multiple people can
still produce poor motion. The GLB uses linear skinning and the bind-pose body
correctives; dynamic pose correctives and cloth simulation are not included.

CLI equivalent:

```sh
python3 -m server.vhuman.cli body-motion --head HEAD_ID --media clip.mp4
```

API: upload raw bytes to `POST /v1/body/uploads` with the media `Content-Type`,
then submit `POST /v1/jobs` with
`{"kind":"body_motion","head_id":"HEAD_ID","upload_id":"UPLOAD_ID"}`.
List results at `GET /v1/heads/HEAD_ID/body/motions`.
