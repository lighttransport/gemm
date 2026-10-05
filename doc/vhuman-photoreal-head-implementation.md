# One-portrait head/bust creation on AMD

The implemented baseline combines a portrait-specific identity, native GNM
anatomy and motion, explicit skin materials, optical eyes, separate accessories,
and an offline Cycles HIP scene. It provides a reproducible asset and consistency
checks. It does not establish photorealistic accuracy from one photograph.

## Corrected Hopper reconstruction

The former Hopper expression catalog used Hopper for video generation but a rig
made from a procedural test portrait for fitting. Its eight-anchor errors could
not validate Hopper's identity. Rigs now record both source and decoded-pixel
hashes; mismatched portraits fail before generation. Existing source video can
be reused after its reference image, clip and tracker hashes are checked.

Glasses also defeated the dark-blob eye detector in direct reconstruction and
rig feature extraction. Both now support the native MediaPipe iris/lid tracker.
Face parsing masks glasses, hat, hair and other occluders **before** geometry
fitting. Manual exclusions take precedence. Parsing confidence is a model
heuristic, not independent ground truth.

The official GNM68 barycentric landmarks seed a fixed canonical registration
to MediaPipe468. Additional attachments are inferred and carry correspondence
weights. Identity fitting withholds every tenth dense attachment and expands
32 to 64 head modes only when withheld error improves. A 170-mode stage requires
multiple calibrated directions. Unobserved eye and tooth identity stays frozen.
Hopper's 32/64-stage withheld error was 4.30/3.97 pixels.

Complete evaluation retains all 17,821 vertices, 253 identity parameters, 383
expression parameters, four native joints, rotation correctives and hierarchical
skinning. Teeth, gums, tongue and mouth cavity come from GNM anatomy. There is no
extra procedural jaw transform on top of GNM's lower-face PCA deformation.
NumPy and differentiable PyTorch ROCm implementations are included.

## Motion and appearance

One identity is frozen across all seven expressions. The portrait camera is
transformed by the exact centred I2V crop; generated camera drift is fitted as
nuisance head pose. A sampled native evaluator solves an observable expression
subspace, head motion and independent eye rotation. Eyelid/iris observations
through glasses retain low authored weights; lens pixels are excluded from skin
appearance observations. Temporal regularization and a worst-triangle penalty
protect motion fitting. A final oriented-area guard rejects or reduces unsafe
PCA expression strength. This local guard does not prove absence of all surface
self-intersections or eye/mouth collisions.

The topology-protected Hopper catalog has mean landmark errors of approximately
0.029–0.032 inter-pupil distances, with unchanged identity across 154 frames.
These are synthetic tracker consistency metrics, not recovered 3D accuracy.

Skin base color uses the existing conservative inverse-lighting estimator.
Roughness, dielectric F0 and millimetre-scale subsurface radii remain explicit
artist priors without calibrated multi-light input. Confidence and completion
maps separate source-visible and inferred skin. Static grooves are photo-shaped
artist profiles; mature pores are metric-space priors. Neither is measured depth.
Twelve regional strain channels map all GNM383 coefficients to bounded dynamic
height maps, avoiding 383 large texture allocations.

Registered I2V high-frequency appearance residuals can also produce twelve
linear-RGB texture channels. An entire expression is withheld. The actual
bounded maps must improve withheld error by at least 5%; otherwise all appearance
deltas are zero and static texture plus authored height drivers remain active.
The final glasses-aware Hopper run improved 1.6% and correctly rejected these maps.
Synthetic color changes are never interpreted as measured wrinkle depth.

## Offline asset

The packed `head.blend` contains skin, native teeth/gums/tongue/cavity, analytic
cornea/sclera/iris-annulus geometry with recessed pupil cavities, a wet lower-lid line, sparse lashes and
visible side-hair strands. Glasses and cap are separate objects. Their depths,
lens thickness, rear shape and strand directions are labelled priors. The cap is
a silhouette extrusion with projected source appearance; it needs artistic
remodelling for a production close-up or large camera turns.

Native motion becomes interpolated Blender shape tracks, joint-derived eye
motion and region wrinkle drivers. Wet lines and lashes follow barycentric skin
attachments and local triangle frames, including eyelid deformation. Hair and
accessory motion follows the head; strand-root skin strain and exact optical-eye
clearance remain approximations. Native tracking observes iris rings rather than
pupil borders; under glasses, the offline eye uses an explicitly labelled 0.30
pupil/iris radius ratio instead of interpreting a dark frame as a large pupil. The
existing GLB/USD rig is also built from the corrected identity for the current
viewer. Its procedural controls approximate native GNM motion. The packed
Cycles scene is the authoritative offline asset.

The renderer supports studio/left/right/rim illumination and ±60° camera yaw.
It acquires the shared ROCm device lock, explicitly selects HIP, records whole
device VRAM every 50 ms, and terminates above 14 GiB. The target is 12 GiB. HIP
failure never silently selects CPU; CPU must be requested explicitly. Render
outputs include a linear EXR and a PNG preview. A fresh-process reload gate checks
all packed signed height maps and native motion samples before accepting a saved
scene. Metric/appearance float maps use explicit scene-linear 32-bit EXR encoding;
generated image buffers and implicit PNG encoding must not be used for them.
On RX 9070 XT, 512 px/32 samples
rendered in about 9 s, and 1024 px/256 samples in about 23 s, including scene
construction in the Blender worker. Recorded whole-device usage was roughly
2.1–3.3 GiB. Lock waiting can increase parent elapsed time.

## Setup and commands

Use the existing ROCm environment and verified face assets. Models remain under
`/mnt/disk01/data/vhuman`; new model copies are unnecessary. The optional Blender
installer pins Linux x64 Blender 4.5.14 and its official archive SHA256. It
extracts under `/mnt/disk01/data/vhuman/tools` and removes the verified archive.

```sh
export LD_LIBRARY_PATH=/opt/rocm/core-7.14/lib:${LD_LIBRARY_PATH:-}
PY=tmp/vhuman-rocm-venv/bin/python
$PY -m server.vhuman.reconstruction.setup_cycles

# A new portrait: native H3 generation, identity-frozen fitting and offline asset.
$PY -m server.vhuman.cli --backend rocm portrait-create \
  --portrait portrait.jpg --out tmp/vhuman-my-head \
  --video-backend h3-fl2va --preset fast5 --texture-res 1024 \
  --accessories keep --detail-preset mature --generate-probes

# Reuse verified source video and corrected native motion.
$PY -m server.vhuman.reconstruction.create \
  --portrait tmp/video-rocm/wan22-build/expression-validation/portrait.jpg \
  --candidate tmp/vhuman-hopper-photoreal/reconstruction/hopperidentity02 \
  --expression-catalog tmp/video-rocm/hopper-basic-expressions \
  --motion-root tmp/vhuman-hopper-photoreal/motion/native06 \
  --out tmp/vhuman-hopper-new-deliverable

# Relight or orbit an existing reconstruction.
$PY -m server.vhuman.cli --backend rocm rig-render \
  --candidate tmp/vhuman-hopper-photoreal/reconstruction/hopperidentity02 \
  --out tmp/vhuman-hopper-right-light --preset final --lighting right --yaw 20
```

Wan, H3 ref2va/fl2va and HV1.5 use the existing native ROCm adapters. Frame counts
and presets are explicit per backend: short defaults are Wan9, H3 22 and HV1.5
81; HV1.5 requires `fast12` or `quality`. H3 probes use a 12 GiB runner budget.
New generation is experimental and records model/runtime receipts. Validated
execution here used H3 fl2va; this change does not establish new Wan/HV1.5 parity.

Dedicated source, landmark, mesh and parsing movies plus the four-panel catalog
are available through `reconstruction.motion_debug`. `motion_capture` generates
held-out neutral, gentle turn, blink, gaze and pucker probes. Resume requires
unchanged creation settings; source artifacts and prior renders are preserved.

## Verification

```sh
PYTHONDONTWRITEBYTECODE=1 $PY -m unittest \
  server.vhuman.test_gnm_anatomy server.vhuman.test_portrait_identity \
  server.vhuman.test_photoreal server.vhuman.test_reconstruction \
  server.vhuman.test_expression_catalog

# Supply the pinned official source; etils[enp] is a reference-only dependency.
$PY -m server.vhuman.reconstruction.validate_gnm \
  --reference tmp/vhuman-gnm-source/gnm_common.py --device cuda:0
```

The independent reference uses Google/GNM commit
`fd704e805ff11bddeac3d972af5a253cea7edc37`, source hash
`b5da354d22bd54e2c3d940f5ce3ca9aac85a8a7b82a0325661ef2fd02fa4df1a`.
Random complete identity, expression, joint rotation and translation achieved
NumPy RMS below 1e-16 m and PyTorch RMS about 1.1e-8 m. Tests also check batched
sampling against full equations, gradients, crop rays, barycentric attachment,
accessory masking, portrait mismatch and neutral/compression wrinkle gauges.

### Curved accessory refinement

The cap now uses a closed curved shell rather than a flat contour extrusion.
Front rings preserve camera-projected portrait UVs; the unseen rear uses a
separate navy cloth prior so the badge and foreground skin do not stretch onto
its sides. A polygon-kernel centre prevents inverted fans at concave rims.
Parsing notches are simplified progressively (3–16 pixels); the selected
simplification, 18 mm front bulge and 90 mm rear depth are recorded as priors.
An outline without a valid kernel fails preparation rather than exporting a
folded shell. The brim and unseen cap shape remain approximate.

Validation on RX 9070 XT: 49 reconstruction tests pass, including closed-edge,
positive-volume, camera-outline and concave-fan checks. The matching 1024 px,
256-sample happy/frame-20/right-light/20-degree render took 24.96 seconds
including packed-scene reload validation, using 2463.33 MiB whole-device VRAM
(previous flat-cap render: 24.15 seconds, 2463.28 MiB). Artifacts are under
`tmp/vhuman-hopper-photoreal/offline/happy14/`; `happy10/` is the comparison.
