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
