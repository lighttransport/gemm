# Adoption plan for vhuman portrait reconstruction

Date: **2026-09-30**. Status: **implemented offline baseline and experimental
WebGL2 appearance paths**. The milestones below preserve the longer-term design;
the implementation record distinguishes delivered behavior from remaining quality work. Research context:
[Portrait to 3D face reconstruction and skin materials](vhuman-face-reconstruction-research.md).

## Implementation record

The `server/vhuman/reconstruction/` subsystem now provides:

- Hash-checked manual pixel observations and an optional pinned MediaPipe adapter.
  Views share identity; each has a nuisance expression, translation, and rotation.
  GNM eye joints and ICT sclera geometry establish anatomical eye alignment before
  fitting. Metric scale still comes from assumed or supplied eye separation.
- Original robust PCA fitting, a weak orientation-filtered surface prior, optional
  convex silhouette support, eye-region penalties, topology guards, and separate
  neutral/captured geometry. Known anchors now use fixed anatomical attachments;
  explicit source annotations remain available for careful identity fitting.
- A small perspective-correct CPU rasterizer, linear color transforms and GGX
  reference, plus an independent differentiable PyTorch BRDF validated numerically.
  PyTorch3D is an optional backend probe, **not installed or used in this validation**.
- Conservative linear diffuse estimation with an explicit median-light gauge and
  gain bounded to two. Coverage/confidence distinguish observed from completed UVs.
  Roughness and F0 are artist priors for ordinary photographs. Three calibrated
  light/exposure views can fit global roughness/F0 through original variable
  projection; conditioning and held-out improvement gates reject weak evidence.
  There is no learned reflectance predictor or per-pixel specular recovery.
- Separate candidate publication, bounded image uploads, API file allowlists,
  cancellation cleanup, source/geometry/topology hashes, and atomic candidate rig
  replacement. Accepted heads/rigs are preserved. Rest changes invalidate previous
  compact soft-deformer packages; those packages are never copied into a candidate.
- Rig rebuilds preserve native reconstructed vertices rather than snapping back
  to the generated head. Identical model UV maps bypass spatial rebaking. Candidate
  LODs use original boundary-preserving clustering on the refined mesh, split
  clusters that invert triangles, transfer shapes/merged skin influences, preserve
  eye/mouth hole boundaries, rebuild contacts, and report correspondence error.
- `KHR_materials_specular` intensity packed in **linear alpha**, with metallic zero.
  USD uses a documented constant F0 approximation and retains the material sidecar.
- Shared optional diffuse-only WebGL2 scattering in both viewers, composed with
  wrinkle/contact/deformer hooks. Analytic eyes provide linear beauty and exact
  ray-hit occlusion guides. Guide/depth bilateral diffusion leaves specular sharp;
  tone/color conversion happens once. Unsupported float targets use ordinary PBR.
  Timing shown in the UI is CPU submission time, not measured GPU frame latency.
- Explicit setup for pinned Apache-2.0 **Depth Anything V2 Small**. A tiny original
  sequential-transform adapter avoids installing/downgrading torchvision/Torch.
  Relative inverse-depth cues require positive affine alignment and a 3 mm median
  error gate. Accepted corrections are low-weight, smooth, limited to 0.5 mm and
  protect eye/lip annotations. Rejected cues leave the fit unchanged and are logged.
- Original triangle/barycentric/normal-offset/SPD Gaussian bindings, visible-view
  static RGB least-squares fitting, covariance transport, stable sorting and
  premultiplied WebGL2 anisotropic quads over opaque mesh depth. The renderer takes
  the final morph/LBS/soft/contact surface. 2K/8K/20K budgets are supported; unobserved
  splats are transparent. This is a radiance overlay prototype, not relightable PBR,
  a learned one-shot generator, or a replacement for the mesh. Gaussian preview
  uses LOD0; source LODs have different attachment topology.

### Quality iteration: correspondence, artifacts and evaluation

The subsequent quality pass implements:

- Authored ICT Multi-PIE68 attachments and an original neutral, eye-aligned GNM
  transfer. Inner lip attachments are constrained to GNM's authored upper/lower
  lip groups. Stored neutral hashes guard against incompatible topology, and fit
  reports retain the attachment map/hash. `landmark_00` through `landmark_67`
  expose the full mapping to manual annotations; the detector still emits eight
  selected anchors. Eyelid-centre observations use centroid attachments. Explicit
  `vertex`/`vertices` overrides win. GNM transfer distances are millimetre-scale,
  not scan-derived correspondence ground truth; review remains appropriate.
  ICT's MIT notice accompanies the mapping data.
- Original concave silhouette fitting from a hash-checked `silhouette_mask`.
  White denotes head foreground. An `exclusion_mask` uses white for hair,
  glasses, hands or other occlusion: it suppresses landmark/material evidence and
  outline samples. Both masks must match the uncropped image dimensions. Outline
  barycentrics are fixed at initialization, with symmetric distance-field/contour
  residuals; large pose changes still need a good starting contour. No learned
  semantic segmentation or automatic glasses removal has been added.
- Wider bounded pose fitting (rotation-vector components ±0.8 radians) with a
  weaker orientation prior. This addresses profile clips that saturated the
  previous ±0.12-radian bound. It does not infer metric scale or camera intrinsics.
- Feature-adaptive source clustering near eyes, mouth, nose and chin; UV cluster
  averages within connected charts; and averaged source normals. Original UV
  corners after vertex collapse caused adjacent faces to disagree at shared
  vertices. The chart-aware transfer removes that particular texture faceting.
- Normal-aware **surface-space** material completion limited to 20 mm, blending
  to a median visible skin color farther away. UV-space nearest filling is used
  only outside covered charts as gutter padding. Ears/neck no longer copy distant
  atlas islands indiscriminately; hidden regions remain low-detail, unobserved
  completion. No measured hidden pores or texture are claimed.
- An offline evaluator that rejects fitting-image overlap by file/source hashes
  and decoded-pixel hashes (including lossless RGB-to-RGBA copies). It reports
  weighted landmark RMS/IPD-normalized error, optional silhouette IoU/boundary F1,
  and side-by-side albedo diagnostics. Independent same-topology geometry supplies
  RMS/P95 vertex errors; timestamped sequences supply velocity/acceleration error
  against reference motion, rather than rewarding suppression of expression.
  `--pose-align-diagnostic` retains raw errors, fits only eye/nose/chin pose, and
  reports unused mouth anchors separately. Pose-aligned comparisons are labelled.

On **the same** reviewed GNM geometry, the clustering comparison measured:

| LOD | Uniform vertices | Adaptive vertices | Uniform RMS / P95 (mm) | Adaptive RMS / P95 (mm) |
| --- | ---: | ---: | --- | --- |
| 1 | 9,544 | 10,407 | 0.606 / 1.464 | 0.418 / 1.241 |
| 2 | 6,147 | 6,905 | 1.915 / 3.337 | 1.702 / 3.200 |

This trades roughly 9%/12% more welded vertices for reduced correspondence error.
UV splits increase exported render-vertex counts. The reviewed rig candidate is
`anatomyquality02` on head `650ac67354cd`; screenshots and a silent control demo are
under ignored `tmp/reconstruction-browser-quality/`. Visual inspection shows
smoother LOD2 texture continuity, while coarse silhouette and hidden-region detail
still limit likeness. The final WebGL2 capture reported no browser errors; absent
optional Gaussian data is no longer requested.

The evaluator was also exercised on **an already cached** SpeakingFaces clip;
no media was downloaded or added to Git during this pass. Fitting frames 0/24/48
and held-out frames 12/36/60 are disjoint. Detector annotations, assumed intrinsics
and assumed IPD make this a smoke evaluation, not independent manual/scan truth.
Widening the pose bound reduced mean **training diagnostic** landmark RMS from
8.44 to 4.34 pixels and recovered roughly 30–33° yaw. Raw held-out RMS remains
14–21 pixels because the provided camera does not track held-out pose/expression;
pose-aligned held-out mouth RMS is 3.25/14.20/9.16 pixels. The difficult second
frame remains a failure case. These results do not establish scan-level likeness
or temporal quality; independently tracked animated surfaces/calibrated capture
are required for that conclusion.

Public evaluation sources remain URLs in `reconstruction/data/evaluation_sources.json`.
[SpeakingFaces](https://github.com/IS2AI/SpeakingFaces) provides speech sequences.
[Multiface](https://github.com/facebookresearch/multiface) provides calibrated
multi-view imagery and tracked meshes under CC-BY-NC-4.0; it is an optional research
evaluation source, excluded from permissive runtime/training distribution. Its
tracked topology requires explicit correspondence and KRT/head-pose conversion
to H-frame metres before use. Media stays outside Git; the local download utility
requires an explicit request and acknowledgement of media terms. Source-code
licenses must not be mistaken for media terms. No calibrated public capture was
available locally for that earlier pass; the subsequent public-starter import is
documented in `server/vhuman/README.md`.

```sh
tmp/vhuman-rig-venv/bin/python -m server.vhuman.reconstruction.evaluate \
  --candidate tmp/vhuman-independent/heads/HEAD_ID/reconstruction/RUN_ID \
  --observations path/to/heldout.json --out tmp/reconstruction-evaluation/RUN_ID

# Optional pose-only diagnostic, preserving unaligned errors:
# add --pose-align-diagnostic
# Optional independently predicted/reference animation NPZ arrays:
# add --surfaces predicted.npz --reference reference.npz
# NPZ keys: positions[views,vertices,3], triangles[triangles,3];
# reference may include timestamps[views] in seconds.
```

The rig interpreter passed **27 reconstruction checks**, including concave-mask
fitting, annotation override stability, UV chart continuity/seam preservation,
normal/distance-bounded completion, decoded-image leakage rejection, profile-pose
recovery, separate expression scoring and temporal error against known motion.
Affected rig/export and HTTP API regression suites additionally passed **33 tests**
with `OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python3 -m unittest
server.vhuman.test_rig server.vhuman.test_app`. Generated weights/media/reports
remain ignored under `tmp/`; no push was performed.

### Running it

Use the existing rig interpreter and model setup first. New weights remain in
ignored `tmp/vhuman-rig/models`; public evaluation media must remain a URL with
manual download, not a repository asset.

```sh
python3 -m server.vhuman.cli --work tmp/vhuman-independent \
  rig-refine-portrait --head HEAD_ID --profile full --res 512 \
  --gaussians 2000 --roughness 0.55 --f0 0.028

python3 -m server.vhuman.cli --work tmp/vhuman-independent \
  portrait-reconstruct --portrait path/to/portrait.png --face-model gnm_v3

python3 -m server.vhuman.cli --work tmp/vhuman-independent \
  rig --head HEAD_ID --reconstruction-run RUN_ID

# Explicit optional download; Small only, with immutable default pins:
python3 -m server.vhuman.reconstruction.setup_depth
# Add --depth-installation tmp/vhuman-rig/models/depth-anything-v2-small
# to rig-refine-portrait to evaluate the weak cue.
```

Candidate files live at `heads/HEAD_ID/reconstruction/RUN_ID/`; its rig lives in
`rig/`. `GET /v1/heads/HEAD_ID/reconstruction` lists published candidates. The rig
page has an accepted/candidate selector and optional scattering/Gaussian previews.
API job kinds are `rig_refine_portrait` and `portrait_reconstruct`, with request
fields at the top level alongside `kind`. Browser/API direct reconstruction first
uploads a PNG/JPEG/WebP to `POST /v1/portrait/uploads` and supplies its returned
`portrait_upload_id`. API jobs reject arbitrary local paths; manual observation
files and alternate installation paths are CLI-only. API `depth:true` selects the
fixed optional Small cache.

### Manual observation example

Paths resolve relative to the observation JSON. Each image (and optional exclusion
mask) requires its SHA256. Cameras use metres in H: +X subject left, +Y up,
+Z face, and camera-local forward -Z. Pixels refer to the **uncropped original**
image, with top-left origin. Additional views require calibrated H-frame cameras.
`vertex` is an index in the selected model's skin-exterior vertex order, not the
unwelded exported GLB. A manually traced `silhouette` is optional; its objective
uses convex support. For concave outlines, supply `silhouette_mask` plus
`silhouette_mask_sha256`; `exclusion_mask` plus `exclusion_mask_sha256` marks
occluded evidence. White means foreground in the silhouette mask and excluded
evidence in the exclusion mask. Use `landmark_00`..`landmark_67` for topology-pinned
Multi-PIE68 attachments, or a `vertices` list for an explicit centroid attachment.

```json
{
  "format": "vhuman.face_observations.v1",
  "views": [{
    "image": "portrait.png", "sha256": "IMAGE_SHA256", "size": [1024, 1024],
    "anchors": {
      "nose_tip": {"xy": [512, 540], "weight": 0.9, "vertex": 1234}
    },
    "camera": {"focal": 2700, "cx": 512, "cy": 512,
      "origin": [0, 0, 1], "rotation": [[1,0,0],[0,1,0],[0,0,1]]}
  }]
}
```

Supply at least four weighted anchors for geometry fitting. For a calibrated
material experiment, add `lighting` per view with `direction_h`, linear RGB
`radiance`, known `exposure` (default 1), and optional `ambient_rgb`. Three views
with varied lights and shared visibility are required. Arbitrary photo exposure
or an estimated studio environment is not calibrated lighting.

### Validation evidence and practical limits

The complete suite passed **143 tests, 5 skipped** with:

```sh
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python3 -m server.vhuman.test_all
```

The rig interpreter additionally passed 20 reconstruction numerical tests,
including projection/frame equivalence, perspective UV interpolation, PyTorch
BRDF parity/finite-difference gradients, safe topology, source LOD boundaries,
merged skin weights, cancellation isolation, calibrated synthetic reflectance
recovery, bounded/protected depth correction and visible Gaussian RGB fitting.
The WebGL2 regression checks constant-diffuse preservation and non-skin occlusion.
Two earlier full-suite runs hit the existing 2-second iris timing gate under
concurrent work; an isolated rerun and the final full run passed. Socket/API and
browser checks require execution outside the socket-restricted sandbox.

RTX 5060 Ti (16 GB) validation built a GNM candidate from the local woman portrait,
exported GLB/USD and both LODs, and ran Small depth inference on CUDA. That portrait's
scene-depth cue **failed the facial alignment gate and was not applied**. This is
an observed limitation of that cue, not a successful depth-refinement quality
claim. On the refined source LOD fixture, measured RMS/P95 correspondence errors
were approximately **0.61/1.48 mm** (LOD1, 9,488 welded vertices) and **1.92/3.33 mm**
(LOD2, 6,051 welded vertices). Counts/errors vary with identity and spacing; this
simple simplifier is conservative around boundaries and thin triangles.
Visual inspection still shows faceting at LOD2 and stretched completion around
ears/neck; these previews establish pipeline operation, not production likeness.

Direct portrait reconstruction was exercised with local generated imagery; those
Qwen-Image 2.1 fixtures retain their research provenance and are not training data.
ICT-FaceKit also completed a full rig/LOD export and a separate authored-expression
fitting check. The reviewed GNM candidate is `829cf0365d6741aa` for local head
`650ac67354cd`. Its screenshots are `pbr.png`, `sss.png`, `gaussian.png`, and
`lod2.png` under `tmp/reconstruction-browser`; `facial-controls.webm` is a silent
viseme/control demonstration, not an audio-driven public speech evaluation.
Screenshots and browser recordings belong under ignored `tmp/reconstruction-browser`.
There is no public clip, model weight, or generated screenshot committed here.

Remaining quality work: dense anatomical correspondence, concave silhouette and
occlusion segmentation, scan/multi-view quality evaluation, spatially varying
calibrated reflectance, authored pore/detail integration on imported topologies,
and physical-device mobile performance measurements. Single-photo hidden regions
are conservative completion; no claim of scan-level geometry or recovered hidden
texture is made. No new dense predictor or Gaussian generator training was added.


## Decisions and first deliverable

Prioritize an offline portrait fitter and explicit skin materials on the existing
GNM/ICT topology. Preserve our facial control namespace, blendshapes, skinning,
speech takes, contact correction, and compact soft-tissue deformer. Use Python
and PyTorch offline and repository-authored JavaScript/GLSL for WebGL2. Prefer
MIT, BSD, and Apache dependencies; keep model and asset terms separately recorded.

The first deliverable is an **opt-in refinement of an existing vhuman head**:
better image-guided geometry, conservative albedo/lighting estimation, material
maps transferred to the rig, and side-by-side inspection under new lighting and
speech. Direct portrait-to-GNM reconstruction follows after this path passes
validation. Advanced neural reflectance synthesis and Gaussian rendering are
later milestones with independent acceptance gates.

The production path will not require Pixel3DMM, DECA, MICA, FLAME, NextFace,
LAM weights, or unreleased material checkpoints. Research references remain
useful for published concepts and separately permitted evaluation. This choice
supports independent development; it is not a claim that rewriting code clears
model, data, patent, or output restrictions.

### Existing integration points

| Subsystem | Current behavior | Planned addition |
| --- | --- | --- |
| Head fitting | Analytic eyes/lids fitted to a generated head; known portrait camera | Image observations, confidence, camera refinement, and explicit diagnostics |
| Rig registration | Procedural scaffold registration and GNM/ICT identity fitting to the subject surface | Joint image/surface fitting, expression separation, bounded geometric detail |
| Skin and UV baking | Portrait color, procedural detail, approximate illumination removal; transfer to rig atlas | Linear-space material fitting and channel-aware transfer with confidence |
| Facial motion | Authored controls, TorchRig/LBS, contacts, wrinkles, compact learned residual | Revalidate after identity changes; shared deformed geometry for all appearance paths |
| Browser/export | Three.js GLB viewer, wrinkle shader hooks, GLB/UsdSkel exports, mobile LOD | Explicit specular materials, optional skin scattering, measured mobile presets |

Implementation should extend the existing head, rig, job, and viewer modules.
The observation/fitting and material estimators should be separate modules so
their objectives, dependencies, and tests remain independently understandable.

## Dependency and original implementation policy

| Component | Decision | Basis and conditions |
| --- | --- | --- |
| GNM v3 | Keep as default human identity/expression model | [Official weights](https://huggingface.co/google/gnm-v3) declare Apache-2.0. Preserve the loader's existing revision/hash checks and notices. |
| ICT-FaceKit Light | Keep as alternate topology and cross-check | [Official release](https://github.com/USC-ICT/ICT-FaceKit) declares MIT. Preserve its pinned revision, source provenance, units, and atlas packing. |
| NumPy, SciPy, PyTorch, Pillow, OpenCV, Three.js | Reuse existing dependencies | Record exact versions and notices; these already form the fitting/runtime stack. |
| PyTorch3D rasterization | Optional offline accelerated backend | [BSD license](https://github.com/facebookresearch/pytorch3d/blob/main/LICENSE). Pin a tested revision compatible with our PyTorch/CUDA interpreter; use rasterization primitives with our own material equations. |
| nvdiffrast | Exclude from the selected production backend | Inspected [NVIDIA Source Code License](https://github.com/NVlabs/nvdiffrast/blob/main/LICENSE.txt), section 3.3, limits use to research/evaluation. It is not interchangeable with a BSD/MIT dependency. |
| MediaPipe tracking | Reuse the existing optional tracker adapter | [Core code is Apache-2.0](https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE); record the specific task-model source, hash, and terms independently. Manual observations and synthetic fixtures must work without the tracker. |
| Depth Anything V2 Small | Optional later geometry experiment | [Official repository](https://github.com/DepthAnything/Depth-Anything-V2) declares Small Apache-2.0; Base/Large/Giant are CC-BY-NC. Pin Small only after validation; relative scene depth is a weak cue, not a facial scan or metric depth. |
| Separable SSS | Implement a small original WebGL2 variant | [Published technique](https://www.iryoku.com/separable-sss/) supplies the design reference. A reference-code adaptation is also possible under its preserved notices, but must be labeled as an adaptation. |
| LAM WebRender | Reference and possible separately audited reuse | Its [wrapper repository](https://github.com/aigc3d/LAM_WebRender) declares MIT, but [package.json](https://github.com/aigc3d/LAM_WebRender/blob/main/package.json) delegates rendering to an npm package. Inspect that package's actual code/license/dependencies before reuse; example avatars have independent provenance. |
| LAM reconstruction checkpoints | External research comparison only | [Weight license](https://github.com/aigc3d/LAM/blob/master/LICENSE_WEIGHT) is CC-BY-NC 4.0 despite Apache core code. Do not make these weights a production dependency or training teacher for this path. |
| LightGeom/LightRig | Keep existing optional offline teacher/export integration | Record our pinned source/license/build provenance. Facial simulation remains separate from material estimation; stiffness is not inferred from albedo or SSS maps. |

### What is practical to implement ourselves

These effort estimates are engineering judgments about a bounded implementation,
not promises of parity with a paper's complete system.

| Technique | Relative effort | Adoption choice |
| --- | --- | --- |
| Robust weighted landmark/PCA fitting, priors, alternating camera/identity/expression optimization | Low to moderate | Extend our NumPy/PyTorch solvers using standard mathematics and published objectives. |
| UV coverage, visibility masks, seam padding, bounded completion, material confidence | Low to moderate | Extend our existing baker. Start with edge-aware/symmetric completion; keep inferred regions identified. |
| Joint diffuse/lighting fitting and bounded GGX material optimization | Moderate | Original staged estimator, validated on known synthetic scenes. Single-view roughness/specular remain conservative priors. |
| Detail normal composition and expression wrinkle drivers | Low to moderate | Extend existing normal transfer and wrinkle hooks; retain the distinction between geometric and appearance detail. |
| Depth-aware separable diffuse scattering | Moderate | Original GLSL implementation with an independent CPU convolution check; compare against the permitted reference. |
| Triangle-bound Gaussian positions and covariance deformation | Moderate | Original binding mathematics and exporter; renderer sorting/compositing is additional work. |
| Complete GPU differentiable rasterizer | High | Reuse BSD PyTorch3D; implement only a small tiled CPU correctness reference. |
| Pixel3DMM-style dense predictor, one-shot Gaussian generator, diffusion reflectance/SSS predictor | High, including data and training | Separate training experiments on permitted data. They are not simple algorithm ports. |

For original implementations, write an equation/interface specification from
papers, standard formulas, and our required behavior; implement it without
translating restricted source. Record inputs consulted, equations, and validation
fixtures. If restricted implementation source is consulted, do not describe the
result as a verified clean-room implementation. A formal clean-room effort would
need a separately prepared specification and an implementation/review process
with recorded separation; none has been performed by this documentation work.
Permissive source reuse is acceptable with attribution and does not need to be
misrepresented as original code.

## Implementation milestones

### P0 Reproducible fitting and material baseline

Create an observation bundle for one or more images: image identity/hash,
pixel dimensions, optional camera intrinsics, 2D anchors with weights, foreground
and exclusion masks, and optional dense cues with their conventions/confidence.
Use the existing tracker as an optional adapter and support manual anchors.
Record assumed versus user-measured scale; a portrait cannot determine true IPD.

Build original linear-space diffuse/GGX reference shading and a tiled CPU
z-buffer for small fixtures. Add optional PyTorch3D rasterization for optimization.
Verify camera, triangle winding, perspective interpolation, UVs, normal frames,
and color transforms between CPU, PyTorch, and browser. Keep UV rasterization
reuse in the existing baker. Synthetic fixtures use repository-authored material
patterns and shapes from permitted GNM/ICT sources; track topology provenance.

**Exit gate:** projection/UV round trips, finite-difference gradient checks away
from visibility discontinuities, and CPU/accelerated rendering comparisons pass.
Record current head/rig render and fitting metrics before changing the objective.

### P1 Image-guided identity and geometry

First refine an existing subject, using the generated surface as a confidence-
weighted initialization rather than ground truth. Optimize similarity/camera,
identity coefficients, and nuisance expression separately. Use robust landmarks,
visible silhouette distance, point-to-plane surface terms, identity priors, and
local smoothness. Preserve eye/lip boundary constraints and exclude hair,
occlusion, and mouth interiors from skin objectives. Refresh visibility during
optimization; reject inverted/degenerate triangles and unsafe socket changes.

Fit a single-image nuisance expression with strong regularization and export a
neutral identity plus that expression estimate. For multiple observations, share
identity and optimize view/expression separately. Single-view identity/expression
ambiguity stays in the report. Detail refinement uses a bounded smooth residual
with a line search that preserves orientation; pores start as normal detail.
Rebuild skeleton/rest transforms, expression transfers, contacts, and UV
correspondence after identity refinement; do not reuse subject deformer bases
whose mesh hash no longer matches.

After existing-head refinement passes, add direct portrait-to-GNM/ICT fitting.
Initialize camera/shape from landmarks and the neutral model, then run the same
objective. Use explicit masks and model priors for unseen parts. This path can
produce the usual fitted-head/rig artifacts without a Pixal3D-generated surface;
it does not claim one-photo scan accuracy. Optional dense cues enter only after
their own ablation demonstrates a benefit.

**Exit gate:** improve median held-out landmark/silhouette error over the current
fit on the fixed comparison set, with no increased invalid triangles or contact
violations. Report difficult subjects individually. Scan error is assessed only
when reference scans exist. Direct reconstruction must also pass neutralization,
identity persistence, camera perturbation, and all supported topology checks.

### P2 Conservative skin material estimation and transfer

Fit in stages: camera/geometry fixed; low-order diffuse illumination and coarse
albedo; then higher-resolution albedo with confidence-aware regularization.
Fix the exposure/albedo gauge with a declared convention, using calibration when
available. Exclude highlights, shadows, hair, glasses, and makeup boundaries from
inappropriate skin priors. Keep diagnostics for exclusions and unexplained pixels.

For a single uncalibrated portrait, default to bounded regional roughness and
dielectric specular priors with artist overrides. Enable fitting those parameters
only for multi-view/light observations that pass coverage and conditioning tests;
insufficient data retains priors and records that decision. Use more expressive
environment lighting only when it improves withheld observations. No SSS or
pigment concentration is reported as measured from a single RGB portrait.

Bake diffuse/base color, roughness, specular, normal/detail, coverage, and
confidence to the source and rig atlases. Extend current nearest-surface transfer
to scalar/color/normal channels with correct tangent conversion and seam padding.
Preserve observed marks; fill unobserved texels conservatively and label them.
Apply procedural pores after reflectance fitting so synthesized detail cannot
hide material errors. Generalize existing wrinkle maps to confidence-weighted
normal/roughness corrections driven by the same facial controls.

**Exit gate:** recover known material changes on synthetic multi-light fixtures,
reduce lighting leakage on held-out real observations, and preserve UV marks
through source-to-rig transfer. Report single-view prior maps separately from
fitted parameters. Existing complexion and manual roughness controls still work.

### P3 WebGL2 skin rendering and portable exports

Add a shared skin shader module used by both head and rig viewers. Compose its
hooks with current wrinkles, deformation, and contact shaders. Use explicit
specular material parameters and conserve the diffuse/specular energy split.
Export compatible parameters through
[KHR_materials_specular](https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Khronos/KHR_materials_specular),
with documented conversion from the estimator's reflectance convention.

Implement opt-in diffuse-only screen-space scattering with separable passes,
depth/normal/skin-mask rejection, and profiles specified in meters. Keep surface
specular, eyes, teeth, and hair out of the blur. Compare transmission separately;
a geometry-derived thin-region mask is a rendering approximation. Ordinary GLB
viewers receive standard PBR; a versioned sidecar enables custom skin shading in
our viewer. [KHR_materials_volume](https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Khronos/KHR_materials_volume)
does not define complete scattering transport and will not be presented as a
portable skin-SSS solution.

**Exit gate:** linear-space CPU/GPU material agreement, no background/eye bleeding,
SSS-off equality with standard PBR, valid GLB/USD exports, and compatible speech,
wrinkle, contact, and deformer playback. Proposed mobile target: sustained
30 FPS at a 1280×720 drawing buffer on a recorded physical midrange device,
including deformation and render work. Software-browser checks verify behavior;
they do not satisfy the device-performance gate. Lower LOD/resolution and
SSS-off are explicit fallback presets.

### P4 Optional dense cues and learned completion

Start with an ablation of Apache licensed Depth Anything V2 Small, using relative
depth at low weight and excluding unreliable pixels. Deriving camera normals
from relative depth requires an explicit camera model and uncertainty; it does
not substitute for a face-specific normal predictor.

If dense cues materially improve P1, train our own compact PyTorch encoder/decoder
for normal/correspondence/confidence prediction using GNM/ICT synthetic views
and separately permitted real data. Split by identity and scene. Use our own
shader and independently sampled parameters; do not distill restricted
reconstruction/material checkpoints or use their generated training labels.

Reflectance or SSS synthesis becomes a separate project only after a licensed,
representative reflectance dataset and calibrated reference renderer exist.
Begin with deterministic UV completion and a compact material prior; compare
against the P2 estimator. Relightify/S³-Face supply published design ideas, not
required code or weights. Learned hemoglobin/melanin remain model-dependent
estimates. Qwen research-tagged portraits and legacy outputs retain their tags
and are not silently promoted into this production training set.

**Exit gate:** held-out identity and lighting improvement over the nonlearned
baseline, calibrated confidence, complete data/model manifests, and no hidden
restricted teachers. Do not advance training solely because a front-view image
looks sharper.

### P5 Optional mesh-bound Gaussian appearance

Build an original static binding prototype before a learned avatar model. Each
splat stores a skin triangle ID, barycentric center, local normal offset, local
covariance, opacity, and appearance. Use barycentric interpolation of the final
deformed triangle to move centers, including the compact residual and contacts.
Transform covariance through the local triangle deformation, bound pathological
scales, and keep it positive definite. For radiance-based appearance, rotate the
view direction into the appropriate local frame before evaluating directional
coefficients. This preserves attachment, not physically valid relighting.

Fit splats from permitted multi-view observations; a single-image run keeps
unobserved regions on the mesh/PBR baseline. Prototype our own WebGL2 projected
Gaussian quads, stable depth sorting, premultiplied alpha, and mesh depth
compositing. Default splats to appearance regions that the mesh handles poorly
and exclude analytic eye/mouth surfaces. Start with authored density LODs and
profile them before any learned LOD. A LAM renderer reuse requires a package
audit; it is not a substitute for this gate.

**Exit gate:** attachment under speech/blink/jaw/contact motion, finite covariance,
correct overlap with opaque meshes, seek/replay determinism, and a measured
quality/performance benefit over PBR alone. Keep radiance splats labeled as such;
explicit relightable Gaussian materials would be a subsequent experiment.

## Proposed interfaces and artifact compatibility

The names below are planned interfaces, not available commands or endpoints.
Retain the existing rig and performance schemas; record new reconstruction
state separately.

| Interface | Planned behavior |
| --- | --- |
| `rig-refine-portrait --head ID --observations FILE --profile geometry\|material\|full` | Create a candidate reconstruction run for an existing head; use its portrait when observations are omitted. Initial default profile: `geometry`. |
| `portrait-reconstruct --portrait FILE --face-model gnm_v3` | Later P1 direct path; optional observations, ICT/procedural selection, and explicit camera/scale assumptions. |
| `rig --head ID --reconstruction-run RUN_ID` | Build from an explicitly selected candidate with compatible topology; omission retains current behavior. |
| Existing job queue | Add corresponding `rig_refine_portrait` and later `portrait_reconstruct` handlers with the existing GPU lock, progress, cancellation, and bounded input handling. |
| `vhuman.face_observations.v1` | Versioned image/camera/anchor/mask/cue manifest; all cue coordinates, units, source revisions, and confidence conventions are declared. |
| `vhuman.skin_material.v1` | Map semantics, color space, tangent convention, BRDF convention, optional scattering profile/units, and estimated/prior/generated provenance. |
| `vhuman.gaussian_binding.v1` | P5 sidecar plus binary arrays, mesh/topology hash, local binding convention, LODs, and radiance/material representation tag. |

Candidates live under `<head>/reconstruction/<run_id>/` with source observations,
neutral mesh/coefficients, optional captured expression, maps, manifests,
`reconstruction_report.json`, and comparison renders. A new direct portrait
creates a normal head record and references its candidate. The loader accepts
only declared safe paths and supported formats; server file allowlists are
extended deliberately. Failed/cancelled candidates remain unpublished and do
not replace accepted exports.

Cache keys include source and observation hashes, topology revision, estimator
version/settings, renderer backend, and learned model hash when used. Rebuilding
a rig with a new neutral identity invalidates incompatible deformer/appearance
caches. Map and sidecar additions are optional; older GLB/UsdSkel and saved takes
remain readable. Selection is explicit until acceptance gates justify a default
change; rollback is selecting the existing accepted head/rig.

## Validation and implementation order

Implement **P0 → P1 existing-head refinement → P2 → P3** as the first adoption
series. Then complete P1 direct portrait construction. P4 and P5 are independent
optional experiments; neither blocks the usable mesh/material path. Each series
gets a focused implementation and audit with exact commands, dependency
revisions, synthetic references, and permitted real comparison inputs recorded.

Extend the existing head, skin, rig, relight, and browser tests for the concrete
risks introduced: camera mismatch, visibility, neutralization, material gauge,
tangent seams, invalid triangles, cache mismatch, failed jobs, and shader-hook
composition. Reuse existing speech takes for animation regression, while adding
subject-matched captures for quality. Split evaluation by subject and hold out
views/lights; fitting error on training frames is not the acceptance result.

All learned models and generated comparison artifacts remain outside Git under
the existing ignored work/cache directories. Public capture media are documented
as source URLs with manual download instructions. A dependency manifest records
code, weights, datasets, example assets, and output provenance independently.
Synthetic fixtures may be generated reproducibly in tests without bundling
model weights. This documentation step installs no new dependencies and
performs no reconstruction, training, or device benchmark.
