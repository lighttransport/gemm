# Complete anatomy USD interchange

The Blender worker exports the current static face, mouth cavity, teeth, gums,
tongue, optical eye parts, tear lines and lashes. It uses metres and Y-up USD,
with relative texture paths. Hair and clothing follow the prepared scene's
contents; prepare the face-only scene with `--no-hair --accessories omit`.

```sh
export TMPDIR="$PWD/tmp"
BLENDER="$HOME/local/blender-5.2.2-linux-x64/blender"
"$BLENDER" -b --factory-startup --python-exit-code 1 \
  --python server/vhuman/reconstruction/usd_anatomy_worker.py -- \
  --scene tmp/vhuman-quality8h/lip_candidate_ramp_patch_render \
  --candidate tmp/vhuman-quality8h/completed_tone_balanced \
  --out tmp/vhuman-quality8h/full_anatomy_usd
```

Use an empty output directory. The prepared scene and candidate must have the
same geometry and portrait. Only basecolor refinement is supported without
rebuilding the prepared scene; normal, ORM and specular maps must match.

The bundle contains:

- `head.usdc` and `textures/`: portable static USD with material bindings.
- `blender_materials.json` and `shader_assets/`: checksummed Blender node graphs,
  socket values, image data and material settings omitted by ordinary USD import.
- `reimported.blend`: verified imported geometry with restored materials and skin
  subdivision settings; no review camera or lights are included.
- `geometry.npz`, `scene_assets.npz`, and `candidate_manifest.json`: native model
  geometry and explicit per-component arrays as an alternate interchange.
- `report.json`: hashes, geometric roundtrip errors, material checks and limits.

Restore in a fresh Blender process, including after relocating the whole bundle:

```sh
"$BLENDER" -b --factory-startup --python-exit-code 1 \
  --python server/vhuman/reconstruction/usd_import_worker.py -- \
  --bundle tmp/vhuman-quality8h/full_anatomy_usd \
  --out-blend tmp/vhuman-quality8h/restored_anatomy.blend
```

The importer checks bundle hashes, all component coordinates, triangle order,
UVs, shader links/defaults and packed texture bytes. Keep the whole directory
together. USD-only consumers may lose subsurface or displacement behavior;
the sidecar restores those features specifically in Blender.

This package is a static captured pose. It does not serialize native expression
animation, skeletal animation or shader drivers. Nested node groups and object
modifiers other than subdivision are rejected. The optional eye-placement and
inner-lid tissue experiments are separate from this bundle.

## Sampled animation

`usd_motion_worker.py` exports all anatomy meshes from an animated
`offline_render` directory. It verifies the source motion/geometry hashes,
bakes point animation, restores Blender materials, and checks every exported
sample after USD import. It retains subdivision as a separate Blender setting.
Boolean geometry can be baked, but that does not establish its anatomical or
temporal quality.

```sh
"$BLENDER" -b --factory-startup --python-exit-code 1 \
  --python server/vhuman/reconstruction/usd_motion_worker.py -- \
  --scene tmp/vhuman-quality8h/motion_gaze_repaired_render \
  --samples-per-frame 2 \
  --out tmp/vhuman-quality8h/full_gaze_motion_usd_verified
```

The default two samples per frame retain integer frames and midpoints. A
22-frame, 24 fps source becomes 43 samples at 48 fps with the same elapsed
time from first to last pose. Output includes `head.usdc`, relative textures,
the Blender material sidecar, a reimported scene, source metadata and a report
with geometry/UV checks and hashes for each mesh sample. This is sampled mesh
animation, not an editable native expression or skeletal rig.

For a fresh-process verification and a portable `.blend`, use the animation
importer. Keep its output directly inside the bundle so `//head.usdc` remains
valid when moving the directory:

```sh
"$BLENDER" -b --factory-startup --python-exit-code 1 \
  --python server/vhuman/reconstruction/usd_motion_import_worker.py -- \
  --bundle tmp/vhuman-quality8h/full_gaze_motion_usd_verified \
  --out tmp/vhuman-quality8h/full_gaze_motion_usd_verified/portable.blend
```

The importer requires new output filenames, verifies bundle hashes and all
sample hashes, restores materials/subdivision and sampled shader animation,
and writes a validation receipt.
The saved animation cache uses a relative path. Cameras and review lights are
not included.

`blender_material_animation.json` stores numeric input/output socket samples
for animated material node trees, including dynamic wrinkle activations. The
exporter samples original source times before retiming, so drivers using
Blender's `frame` variable retain their original meaning. The importer restores
linear socket keyframes without the source scene's custom properties or driver
dependencies. It verifies every channel/sample, keyframe times, interpolation,
and the sidecar hash. Unsupported animated node properties are rejected.

This animation sidecar is specific to Blender. Generic USD readers may still
show static materials. Values are verified at sampled times; nonlinear driver
behavior between samples is approximated by linear interpolation. Original
driver expressions are not retained. Older bundles without the animation
sidecar remain readable with their original static-material limitation. The
export gate does not prove fit accuracy, collision freedom, or behavior at
unsampled times.

Run shader-animation regressions with a repository-local temporary directory:

```sh
TMPDIR="$PWD/tmp" "$BLENDER" -b --factory-startup --python-exit-code 1 \
  --python server/vhuman/test_usd_material_animation.py
```

## Fitted optical centers in the Blender scene

An optical part attached to GNM eye joint 2 or 3 may specify
`rotation_center_offset: [x, y, z]` in `scene.json`, in metres in the aligned
rest coordinate frame. Its stored mesh positions must already include that
placement offset. The renderer rotates surface points around the shifted rest
center, then carries the offset with head joint 1. This prevents eye gaze from
orbiting a translated globe around the old native joint. Parts without the
field retain the original joint transform.

`joint_motion.shifted_joint_positions` validates the arrays and proper rotations.
It does not validate anatomical placement or contacts. Sampled USD bakes the
resulting positions; it does not preserve this editable joint-center control.

The Blender material sidecar also retains static Color Ramp stops (position
and RGBA), color mode, interpolation and hue interpolation. Verification checks
these nested values explicitly. Named point attributes used by a shader still
need their own mesh round-trip check; restoring a node does not establish that
its input attribute exists. The tooth-gradient study verifies its `enamel_tip`
float point values separately on both baked tooth meshes.

## Local quality review and portable downloads

With the quality-iteration artifacts present, build the comparison page and
three portable archives from repository root:

```sh
TMPDIR="$PWD/tmp" PYTHONPATH="$PWD" tmp/vhuman-texture-venv/bin/python \
  -m server.vhuman.reconstruction.quality_review --work tmp/vhuman-quality8h
TMPDIR="$PWD/tmp" PYTHONPATH="$PWD" tmp/vhuman-texture-venv/bin/python \
  -m server.vhuman.reconstruction.quality_review_verify \
  --work tmp/vhuman-quality8h --out tmp/vhuman-quality8h/review_browser_checks
```

The builder requires the recorded experiment directories; it is not a general
scene exporter. `review/index.html` contains six comparison panels and links to
measurements. `review/downloads/` contains eyelid-fit, tongue-motion and
gaze-wrinkles ZIPs. Extract each archive completely and open `portable.blend`
alongside its relative USD cache, textures and material sidecars. The manifest
records source and archive hashes. Serve repository `tmp/` as the HTTP root to
retain links to experiment evidence and the existing device test page.

The verifier checks archive hashes/CRC, image decoding, evidence links,
comparison controls and narrow-viewport layout in Chromium. This is static
review verification, not live-avatar or physical-mobile validation. Eyelid and
tongue candidates retain residual contacts and remain experimental. The linked
device test page uses the earlier browser avatar, not these new anatomy bundles.

## Candidate-first ear and skin continuation

The later continuation selects `tmp/vhuman-quality4h/tongue_reconstruction/`.
Its current default and authored mouth stress bundles are
`preferred_final_usd/` and `preferred_final_stress_usd/`, with verified
`portable.blend` scenes and relative USD caches. The six-panel gallery is
`tmp/vhuman-quality4h/review/`; the updated live avatar and prepared device page
are in `tmp/vhuman-quality4h/player/`. See `session.json` and
[QUALITY_ITERATIONS.md](QUALITY_ITERATIONS.md#native-tongue-rest-lift-and-current-candidate-delivery)
for the actual selected paths, measurements and contact limitations.

Attach portable source evidence after export, using the matching reconstruction:

```sh
$PY -m server.vhuman.reconstruction.usd_candidate_evidence \
  --candidate tmp/vhuman-quality4h/tongue_reconstruction \
  --bundle tmp/vhuman-quality4h/preferred_final_usd
```

Use a bundle without an existing `candidate_evidence/` directory. The command
validates reconstruction provenance and the USD hash, then copies source
geometry/portrait, original-resolution appearance maps and masks with a hashed
inventory. Model weights and inherited capture reports are excluded. The fresh
USD importer verifies this optional evidence and records it in its receipt.
Both current default/stress downloads include it; archive verification checks
the extracted subset as well. This is source evidence preservation, not new
anatomical or photographic observation. The live browser retains separate
offline eyelid and authored tongue-gate limitations; physical mobile validation
remains pending.
