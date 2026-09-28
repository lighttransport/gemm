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

The server accepts `POST /v1/jobs` with `{"kind":"body","head_id":"..."}`
and optional `outfit`, `quality`, `seed`, `steps`, and `garments` fields. The
rig page has the same controls and can preview body poses with facial speech.
The local checkpoint directory defaults to `/mnt/nvme01/models/sam3d-body`;
override it with `--sam3d-body-model` on the CLI or server. The model files
remain outside the repository and are not copied into avatar exports.

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
