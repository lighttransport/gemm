# Blender-assisted surface texture refinement

The 2026-10-08 continuation promotes a conservative surface-color correction
of the existing hybrid. It uses Blender 5.2.2 LTS at
`~/local/blender-5.2.2-linux-x64/blender` on the RTX 5060 Ti through OptiX.
It adds no diffusion images and changes no geometry. The source remains the
same evaluation-only Edit-2511/MV-Adapter hybrid described in
[TEXTURE_CUDA.md](TEXTURE_CUDA.md).

## Result

| Candidate | Seam score | Photographed texels changed | Decision |
| --- | ---: | ---: | --- |
| Original hybrid | 0.0467475072 | 0 | Retained reference |
| Gentle surface correction | 0.0367668227 | 0 | Promoted: 21.35% lower seam error |
| Stronger surface correction | 0.0328288080 | 0 | More uniform broad tone; comparison only |
| Visibility-only photo repair | 0.0364289350 | 605 | Experimental; modifies source-supported pixels |
| Broad low-confidence photo repair | 0.0199132622 | 106,251 | Not promoted; changes too much photographed appearance |

The promoted candidate preserves all 175,326 photographed texels. Geometry,
portrait, coverage, confidence, normal and ORM files remain byte-identical.
Unseen coverage stays 91.8141%: it is inherited generator support, not new
observations. Cross-view spread is likewise the original generation's metric;
the surface correction does not claim a new multiview accuracy measurement.
The seam threshold remains 0.0472149823 (original baseline × 1.01).

Matched Cycles renders show smoother ear/neck transitions and reduced broad
color patches. The gentler solve retains more broad tone variation than the
lowest-scoring solve. Photographed ear/temple defects, incomplete stubble
appearance and single-view reflectance ambiguity remain. This is an
incremental quality improvement, not completed photoreal reconstruction.

## Blender bridge and interchange

`reconstruction.blender_texture` exports checksummed NPZ surface samples and
launches `blender_texture_worker.py`. The worker:

- Constructs the exact fitted triangulated skin with per-corner UVs and the
  existing basecolor, normal and roughness maps.
- Converts GNM metres/Y-up to Blender metres/Z-up with `(x, -z, y)`.
- Casts rays from the fitted source camera against Blender's BVH. A sample is
  visible when the first hit is within 0.25 mm of its surface point.
- Writes `head.usdc`, portable textures, a packed `head.blend`, and five fixed
  orthographic Cycles views. Studio lighting and AgX −2 EV are authored review
  settings, not fitted illumination.
- Imports the USD into an empty scene and checks coordinates, triangle order,
  corner UVs, portable basecolor resolution and actual texture pixels. The
  promoted asset round-tripped with zero error in every check.

The NPZ bridge preserves the exact arrays for further custom processing; USD
is the portable DCC interchange. The audit currently supports one source
camera and rejects multiview inputs rather than silently auditing only one.
The exported skin is static; it does not replace the separate GNM motion rig.

```sh
export TMPDIR="$PWD/tmp"
export OMP_NUM_THREADS=4
PY=tmp/vhuman-texture-venv/bin/python
BLENDER="$HOME/local/blender-5.2.2-linux-x64/blender"
BASE=tmp/vhuman-texture-5060ti/work/out_baseline
SOURCE=/mnt/nvme02/models/vhuman-texture-inputs/obama/material12
WORK=tmp/vhuman-blender

$PY -m server.vhuman.reconstruction.blender_texture \
  --candidate "$BASE" --source "$SOURCE" --out "$WORK/baseline" \
  --blender "$BLENDER" --device OPTIX
$PY -m server.vhuman.reconstruction.surface_texture \
  --candidate "$BASE" --audit "$WORK/baseline" --out "$WORK/out_balanced" \
  --strength 8 --prior .3
$PY -m server.vhuman.reconstruction.blender_texture \
  --candidate "$WORK/out_balanced" --source "$SOURCE" \
  --out "$WORK/balanced_review" --blender "$BLENDER" --device OPTIX --no-audit
```

Use fresh output directories when repeating runs. The initial `baseline`
artifact supplied the visibility audit; the matched baseline renders are in
`baseline_review` (the first lighting attempt clipped highlights).

## Surface correction and source projection repairs

`surface_texture` constructs a 3 mm voxel graph with normal bins, limited
normal-compatible edges and a weak synthetic-color prior. It solves broad
log-RGB tone, interpolates a bounded gain (maximum absolute log gain 0.5),
and retains local texture variation as a multiplicative residual. Photograph
anchors are restored exactly after quantization. This approximates a local
surface graph; it is not a recovered albedo or an anatomical fit.

The promoted solve uses strength 8, prior 0.3 and 38,788 graph nodes. The
stronger comparison uses 24/0.15. Neither changes photographed pixels.
Coverage/support maps are not inflated by regularization.

Blender found 627 of 175,326 source-supported samples occluded at the stricter
ray tolerance. `--repair-projection` defaults to editing only these samples;
605 actually change after quantization. The explicit
`skin_projection_repair.png` mask, original input hashes, changed/protected
counts and audit receipt are retained and packaged with mobile exports.

```sh
$PY -m server.vhuman.reconstruction.surface_texture \
  --candidate "$BASE" --audit "$WORK/baseline" \
  --out "$WORK/out_visibility_repair" --repair-projection
```

`--confidence-threshold` can additionally permit low-confidence source
repairs. It defaults to zero. The 0.6 trial made 106,303 samples editable and
changed 106,251, so it is experimental and fails the original photo-preservation
gate. Do not reinterpret its lower seam score as passing that gate.

## Garment rejection before composition

`edit_guard` parses independent raw edits against the fixed geometry mask.
The default garment threshold is 0.5% of visible geometry. It writes a
`quality.json` receipt containing exact image, generation, model and source
hashes. A rejected run exits with status 2 while retaining the images.

```sh
$PY -m server.vhuman.reconstruction.edit_guard \
  --work tmp/vhuman-texture-5060ti/work --backend selected \
  --parsing-model /mnt/nvme02/models/vhuman-texture-inputs/face-parsing/resnet18.onnx
# Expected rejection of the previous front/left trial.
# Use --require-approved-edits on mv_texture compose to enforce these receipts.
```

The known bad front/left edits measured 31.35%/22.76% garments and were rejected;
the raw/original right edit measured 0% and passed. Enforced composition refuses
missing, changed or rejected raw edits before writing a hybrid. This gate is
opt-in to preserve legacy research reproducibility. Semantic garment detection
cannot certify identity, camera alignment or anatomy; visual review remains
required. The promoted surface workflow avoids new full-image edits entirely.

## Browser and mobile-device test

The current artifacts are under `tmp/vhuman-blender/`:

- `index.html`: matched baseline/candidate render comparison and decision.
- `player/`: promoted browser avatar; `player/device-test.html`: device test.
- `vhuman-texture-usd.zip`: USD, textures, checks and appearance provenance.
- `balanced_review/head.blend`: editable packed Blender scene.
- `decision.json`: numeric gates and explicit limitations.
- `out_balanced/`: promoted candidate; original baseline remains unchanged.

The original browser package uses the previous `texture_scene` with older hair overlays
disabled, retaining native anatomy and bindings. Build it through the existing
mobile export/preprocess/browser commands, using the repository-local
Emscripten cache as documented in TEXTURE_CUDA.md. The browser builder now
includes a standalone device test page.

The later candidate-first ear/skin continuation is documented in
[QUALITY_ITERATIONS.md](QUALITY_ITERATIONS.md#native-tongue-rest-lift-and-current-candidate-delivery).
Its current pointers are `tmp/vhuman-quality4h/session.json`: preferred scene
`preferred_tongue_candidate/head.blend`, complete portable default/stress bundles
`preferred_tongue_usd/` and `preferred_tongue_stress_usd/`, comparison gallery
`review/`, and updated live browser `player/`. The browser includes new ear and
tongue rest geometry and preserves fitted optical centers under gaze; the
offline eyelid correction and authored stress gate remain in Blender/USD.
Earlier candidates and baseline artifacts remain available. Physical mobile
validation and residual contact acceptance remain pending.

```sh
$PY -m server.vhuman.mobile.browser_verify --player "$WORK/player" \
  --out tmp/vhb-check --hardware --device-test
python3 -m http.server 8797 --bind 127.0.0.1 --directory "$WORK"
```

The hardware check passed on NVIDIA ANGLE/Vulkan with zero native/WASM vertex
error, about 30 FPS, passing relighting, detail, camera, audio-clock and cancel
checks. It also runs four short phases of the device page and verifies that
the downloaded JSON equals the package-bound report. Receipts are in
`tmp/vhb-check/verification.json` and `downloads/vhuman-device-test.json`.

No physical phone/tablet was connected. To measure one, serve this directory
on the PC's LAN interface (`--bind 0.0.0.0`), open
`http://PC-LAN-IP:8797/player/device-test.html` on the device, enter its model,
run the standard or sustained test while keeping the page visible, complete
visual checks and download the JSON. The page stores no results remotely.
Static, animated, relighting and detail-off phases record GPU/browser,
package hashes, render dimensions, display/pose rates and sampled timing
percentiles. Targets are ≥24 display FPS and ≥20 animated pose FPS.
Backgrounding or context loss invalidates the run. Speech requires a secure
context and is outside this local rendering test; HTTPS is needed for speech
on a LAN device. Actual mobile performance remains pending a device receipt.

## Local skin color cleanup

For spatially localized pale/cyan texture artifacts, use explicit ellipsoidal
regions in the fitted geometry's metre coordinate system. Each JSON region has
`name`, `center: [x,y,z]` and positive `radii: [rx,ry,rz]`; regions must not
overlap on the sampled surface. The tool verifies the candidate, surface audit,
and original photographic reference hashes before editing.

```sh
TMPDIR="$PWD/tmp" PYTHONPATH="$PWD" tmp/vhuman-texture-venv/bin/python \
  -m server.vhuman.reconstruction.local_skin_color \
  --candidate tmp/vhuman-quality8h/completed_illumination_0.5 \
  --audit tmp/vhuman-quality8h/completed_surface_audit \
  --regions tmp/vhuman-quality4h/ear_color_regions.json \
  --out tmp/vhuman-quality4h/ear_cleanup_new --edit-observed
```

Choose a new output directory. Photographed texels are protected by default;
`--edit-observed` explicitly allows their alteration and records the count.
The selected ear trial uses that flag. Region coordinates are fitted-subject
specific and must be reviewed before reuse on a different head. A local median
prior suppresses color outliers; it does not recover reflectance, geometry or
unseen evidence. Euclidean neighborhoods can mix nearby folds, and real
pigmentation can be removed. Inspect matched rendered views under more than
one light before selecting a result. Geometry and support maps are copied
unchanged; the output includes a weighted edit mask and source provenance.

Run numerical protection and edge-case checks with
`python -m unittest server.vhuman.test_local_skin_color`.

## Validation

114 reconstruction/mobile tests passed, with one optional Torch/OpenEXR test
skipped because OpenEXR is absent. New tests cover fixed-color behavior,
photograph protection, bounded gain, opposing-sheet isolation, invalid inputs,
garment masks and stale/rejected guard receipts. Real-data checks confirm
unchanged protected bytes and geometry; Blender independently checks the USD
round trip. Exact command:

```sh
$PY -m unittest server.vhuman.test_surface_texture server.vhuman.test_mv_texture \
  server.vhuman.test_generated_skin server.vhuman.test_photoreal \
  server.vhuman.test_quality server.vhuman.test_reconstruction \
  server.vhuman.test_mobile_preprocess server.vhuman.test_mobile
```
