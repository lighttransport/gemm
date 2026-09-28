# Facial rig for generated heads

Turns a fitted head (Qwen-Image 2.1 portrait → Pixal3D → `head/fit.py`) into
an animatable face: one fixed template topology fitted to the subject, a
template skeleton (neck, head, jaw, eyes, upper/lower teeth, a 4-joint
tongue), procedural teeth/gums/tongue, skin weights, 52 expression shapes plus
correctives, a linear rig evaluator, and skinned glTF / UsdSkel exports.
It is our own replacement for engine auto-rigging and face tools: no engine
code, assets, DNA files or measured character data are used.

```sh
# rig interpreter (numpy, Pillow, scipy, PyTorch; CUDA optional)
uv venv tmp/vhuman-rig-venv --python 3.12
VIRTUAL_ENV=tmp/vhuman-rig-venv uv pip install -r server/vhuman/requirements-rig.txt

sh server/vhuman/run.sh                        # http://127.0.0.1:8790/rig
python3 -m server.vhuman.cli rig --head <id>   # same job without the server
python3 -m server.vhuman.cli rig-track --head <id> --track capture.txt --out anim.usda
python -m server.vhuman.rig.build <head folder> [--res 2048] [--out DIR]   # in the rig interpreter
sh server/vhuman/rig/external.sh [--local] [--build]   # LightRig + LightUSD (vchar) under third_party/
```

Outputs go to `<head>/rig/`: `rig.glb` (web viewer), `rig.usda` + `textures/`
(and `rig_usd.zip`), `rig.json` (the rig definition), `rig_*.png` maps,
`preview.png`, `rig_report.json` (fit, bake and timing statistics).

## Pipeline

| Stage | Module | What it does |
| --- | --- | --- |
| Features | `features.py` | Lid-margin contours on the fitted eyeballs; the lip seam (dynamic-programming dark valley on lip colour), vermilion borders, brows; midline profile landmarks (nose tip, subnasale, chin, …), ears, the neck cut. |
| Template | `template.py` | Procedural topology on a direction sphere around a centre inside the skull: edge loops around the eyes, mouth and neck cut, blue-noise elsewhere (denser on the face), spherical Delaunay; lid margin/lining strips and a mouth bag; two-chart skin UV atlas. Cached by source hash. ≈13k vertices. |
| Registration | `register.py` | Subject rings are rebuilt from the subject's own contours (same construction as the template's), free vertices follow a compact RBF warp, rays from the centre place them on the surface, then **PyTorch** minimises point-to-plane distance (normal-compatible correspondences), edge-length and bending terms. |
| Texture | `bake.py` | Per texel: closest subject surface point → subject UV; base colour / ORM resampled, normal map re-expressed into the template's tangent frames (keeps geometric detail the template lacks). |
| Skeleton | `skeleton.py` | Joint placement from features (jaw hinge in front of/below the ear, incisors behind the lip seam, tongue along the mouth floor, eyes at the fitted eyeballs). |
| Weights | `skinning.py` | root/neck/head/jaw; lips split exactly at the seam, jaw boundary from the mouth corners to the hinge, surface diffusion. |
| Shapes | `expressions.py` | 51 `lr.face.v1` expression shapes + 4 correctives. Lids rotate about the eyeball centre (blink closes each upper sample onto its lower partner); `mouthClose` is solved against the jaw skinning; light PyTorch sparse-Laplacian smoothing. |
| Mouth | `mouthparts.py` | 28 teeth on a smooth dental arch (overbite/overjet), swept gums, lofted tongue skinned to the tongue chain. |
| Carried meshes | `attach.py` | Eyeballs from `head_eyes.glb` (rigid on the eye joints); tearlines, caruncles and eyeshells follow the lids through a surface wrap. |
| Rig | `rigdef.py` | Controls → correctives → sparse joint-delta matrix + blendshape weights (+ ML corrective weights) → LBS. Identical code in `web/vhuman_rig.html`. |
| Deformer | `torchrig.py`, `mldeformer.py`, `mlruntime.py`, `native.py` | Batched PyTorch rig, contact/ARAP ground truth, PCA + MLP2 training, numpy runtime, native package + ctypes. |
| Export | `gltf.py`, `usd.py` | glTF: one skin, sparse morph targets (POSITION + NORMAL), `extras.targetNames`. USD: SkelRoot/Skeleton/BlendShape, vchar control metadata, a range-of-motion `SkelAnimation`. |

## Deformer

Two layers, both evaluated identically in numpy (`rigdef.Rig`), the web
viewer and native C:

1. **Linear rig** (below): blendshapes, correctives and LBS.
2. **ML corrective deformer** (`mldeformer.py`, trained per head during the
   build; `--deformer-samples 0` skips it). A PyTorch ground-truth solve
   relaxes the linear result for ~2k sampled control combinations:
   anchoring to the rig, ARAP soft tissue against the rest shape (rotations
   by a batched polar Newton iteration), and contacts. Lids stay outside
   the rotating eyeballs, lips/vestibule outside per-tooth spheres on the
   teeth joints, and the upper lip above the lower one. All thresholds are
   relative to the rest pose, so the neutral face is unchanged. Residuals
   are mapped before skinning, compressed by PCA (48 components) and
   learned by a two-layer ReLU MLP (controls → coefficients). Outputs:
   - `deformer.lrm`: LightRig `.lrm` safetensors, the same format as
     `ryzen/lightrig_lrm_runner.c`.
   - `deformer_basis.safetensors`: the PCA basis.
   - morph targets `ml_mean`, `ml_00…` in rig.glb and rig.usda, with weights
     from the MLP (`rig.json` `ml_deformer`).
   - `deformer.json`: statistics, including contacts on held-out controls for
     the linear rig and for linear + ML (reference man: eye 11403→2132,
     teeth 1782→756, lips 3607→1723 contact vertices).
3. **Native runtime**: `ryzen/vhuman_deformer.{h,c}` loads
   `rig_deformer.safetensors` (written by `native.py`: the welded head,
   all morphs, skin, the rig as dense tensors, the MLP). It evaluates one
   frame (sparse AXPY over active morphs, `lt_mlp2_f32` for the MLP) or a
   batch (`sgemm_avx2` over all morphs), then LBS. `make -C ryzen vhuman &&
   ryzen/bench_vhuman_deformer <head>/rig/rig_deformer.safetensors`.
   The reference head (12.6k vertices, 117 morphs incl. 65 ML) takes
   1.15 ms/frame with the ML correctives (0.51 ms batched) and 0.18 ms
   without, single-threaded. Parity with numpy
   is ≈1e-7 m (`test_rig`, via ctypes).
4. **GPU runtime**: `cuda/vhuman/` (cuew + NVRTC): the host prepares the rig
   state, and one fused kernel blends morphs (register-tiled over 8 frames)
   and skins. It runs at ~2.3 µs/frame for 1024-frame batches on an RTX 5060
   Ti and equals the CPU result.

Contacts (ground truth, the viewer's heat map and `viz.json`): lid vertices vs
the eyeballs, lip/vestibule vertices vs sphere sets (per-tooth spheres on the
teeth joints; three spheres per tongue cross-section on the blended tongue
joints), and upper vs lower lip pairs on rings 1..-2. The sampler includes
tongue controls with an open jaw. Reference man, held-out contact vertices
for linear vs linear + ML: eye 16930→1906, teeth 2328→717, lips 6060→2872,
tongue 289→278. The solver removes tongue contacts, but the network does
not yet generalise them.

## Inspection (web viewer)

`/rig` → Inspect:
- **Joint labels**: joint markers and names over the skeleton.
- **Joint weights**: the selected joint's skin weights as a heat map.
- **Deformation**: displacement from rest in mm, including skinning, shapes
  and ML.
- **ML corrective influence**: the pre-skinning ML offset in mm.
- **Contacts / collisions**: penetration depth per vertex for the current
  pose, with counts per contact class. It uses the build's contact model
  (`viz.json`); toggle "ML deformer" to compare.

## Rig evaluation (`rig.json`)

```
x   = clamp(controls, range)                        60 controls
c_p = min(1, w_p * prod_i clamp(x_i, 0, 1))         correctives
in  = [x | c]
dJ  = M in           per joint (tx ty tz rx ry rz), local, radians (sparse M)
local_j = T(rest_t + dt) * R(rest_R * Rz * Ry * Rx)
b_k = in[src_k]      blendshape weights
v   = LBS(rest + sum_k b_k d_k)
```

The layering (GUI→raw controls, product correctives, a sparse linear joint
matrix, blendshape pass-through) follows the published design of rig
evaluators such as the MIT-licensed OpenRigLogic; only the concepts are used.
Names are our own; the 51 expression controls are the LightRig canonical
`lr.face.v1` set (ARKit-style names), so LightRig/MediaPipe face tracks
(`timestamp` + 52 values per line) drive the rig in the viewer and through
`cli rig-track`. Extra controls: `tongueOut/Up/Down/Left/Right/CurlUp`,
`headYaw/Pitch/Roll`.

## Viewers and USD tools

- Web: `/rig` — grouped sliders, expression and viseme presets, auto blink,
  gaze following the pointer, range-of-motion and talk loops, LightRig track
  playback, skeleton/wireframe views, downloads.
- `vchar` (LightUSD, built by `external.sh --build`):
  `cd <head>/rig && vchar rig.usda` (`--play` for the ROM; `--headless --time T
  --screenshot out.ppm` for renders). Blendshape-only controls appear in its
  facial panel through `customData.vchar`; joint-driven controls are in the
  ROM animation and `rig.json`.
- LightRig: `lightrig inspect rig.usda`.

## Provenance and licensing

- MetaHuman DNA Calibration (proprietary licence restricted to Unreal Engine
  use) and the Unreal Engine source/plugins (EULA) were read for concepts
  only. No code, DNA files, joint lists, control lists, masks or numeric data
  from them are used. OpenRigLogic (MIT) informed the evaluation layering;
  no code is copied.
- The template topology, canonical layout numbers, tooth/tongue proportions
  and expression fields are original synthetic choices, not measurements of
  any licensed character or scan.
- LightRig and LightUSD (Apache-2.0) are external tools, cloned by
  `external.sh` into `third_party/` (ignored by Git), never vendored.
- Generated assets inherit the terms of their inputs (Qwen-Image output,
  Pixal3D reconstruction); they stay in the local work directory.

## Limitations

- Expression shapes are procedural fields scaled to the subject, not solved
  from captured performances; wrinkles and wrinkle maps are not modelled.
- Ears and nostrils are smoothed by the template fit (their detail survives
  in the baked normal map only). The template has no separate eyelashes.
- The mouth interior, teeth and tongue are generic; nothing is inferred from
  the (closed-mouth) portrait.
- Mouth detection falls back to proportions when the lips are not found
  (reported as `features.mouth.fallback`).
- vchar shows joint-driven controls only through animations (the ROM).
- The ML deformer corrects sub-millimetre contacts and soft-tissue
  shearing; it cannot add expression detail the procedural shapes lack.
  Lip-lip contacts are only about halved, and tongue contacts are solved but
  not learned. Some lip crossings come from the rig itself (e.g. mouthClose
  without jawOpen). The native runtime covers the welded head; teeth, tongue and eyes are
  rigid or simply skinned in the exports.
