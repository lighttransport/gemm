# Face, eye and oral quality iterations

This is an ongoing eight-hour study begun 2026-10-08, 19:15 JST. Scope is face
mesh fitting, skin reconstruction/completion, optical eyeballs, teeth and tongue.
Body, clothing and hair are excluded. The prior promoted surface-texture
candidate remains `tmp/vhuman-blender/out_balanced` until a complete replacement
passes fitting, visual, topology, material and motion checks.

Live experiment receipts and checkpoints are in `tmp/vhuman-quality8h/`.
`session.json` records the work window, results and next experiments. No final
quality completion is claimed by this initial checkpoint.

## Initial numerical face trials

All trials start from the existing `material12` fit, with fixed camera and the
same recorded training/held-out landmark split. These metrics measure tracker
consistency, not independent 3D accuracy. Reusing this split across trials is
model selection; it is not a new held-out dataset.

| Trial | Held-out face px | Held-out mouth px | Minimum oriented area ratio | Existing gate |
| --- | ---: | ---: | ---: | --- |
| Source fit | 1.37578 | 1.33215 | — | Reference |
| Expression only, 96 modes | 1.35324 | 1.34502 | 0.15605 | Rejected |
| 32 identity modes, no surface field | 1.34439 | 1.21648 | 0.37696 | Rejected |
| 1 mm surface field, identity frozen | 1.28949 | 1.12299 | 0.15378 | Numerical pass |
| 32 identity modes + 0.5 mm field | 1.31686 | 1.17582 | 0.20207 | Numerical pass |

The two passing fits have complete 1024-square Cycles renders with optical
ocular surfaces, upper/lower teeth and gums, mouth cavity and tongue. Fresh
Blender-process validation passes for their packed float maps and geometry
attachments. Visual differences are modest; geometry promotion awaits oral
alignment/contact and motion checks. Their rebaked unobserved skin is still a
low-detail prior, so it must not silently replace the completed material.

CUDA fitting was blocked by ROCm-only guards despite using portable PyTorch
operations. GNM and `refine_fit` now accept available NVIDIA CUDA as well as
ROCm. The refinement CLI selects the corresponding shared GPU lock and takes
an explicit pinned parsing-model path. Full CUDA GNM evaluation matches NumPy
with maximum vertex error 0.0000786 mm and finite identity gradients.

```sh
export TMPDIR="$PWD/tmp" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=tmp/vhuman-texture-venv/bin/python
SOURCE=/mnt/nvme02/models/vhuman-texture-inputs/obama/material12
PARSER=/mnt/nvme02/models/vhuman-texture-inputs/face-parsing/resnet18.onnx
$PY -m server.vhuman.reconstruction.refine_fit "$SOURCE" \
  --out tmp/vhuman-quality8h/fit_surface1 --iterations 1200 --modes 128 \
  --identity-modes 0 --surface-mm 1 --device cuda:0 --parsing-model "$PARSER"
```

Use new directories for repeats. New fit reports record cumulative surface
residual magnitude as well as the incremental correction limit. Geometry
refinement removes copied synthetic-completion receipts rather than leaving
old-geometry appearance provenance attached to a new fit.

## Optical eye alignment

The baseline optical iris centres project 3.0198 px and 2.8725 px above the
recorded source detections, causing an upward-looking appearance. The new
`ocular_fit` solver keeps globe centres fixed and bounds total rotation to
20 degrees and uniform scale to 0.9–1.1. It fits iris centre and horizontal
radius, with a weak prior toward the original pose/scale. Unreachable or
non-improving observations are rejected.

The first accepted solution uses approximately 10 degrees of downward pitch,
less than 1 degree yaw and scale factors 1.0039/0.9960. Geometric centre residuals
fall below 0.0003 px, and the matched Cycles render improves the gaze. That is a
fit to the source detector, not evidence of true gaze or refractive accuracy;
rendered apparent iris shape still needs assessment. Globe anatomy remains a
prior. The scene stores the exact eye-fit source hash and per-eye diagnostics.

## Portable full-anatomy rendering

`offline_render` now supports explicit `cuda` and `optix` devices in addition
to HIP/CPU. NVIDIA runs use the shared device lock, a free-memory gate and
whole-device VRAM monitoring every 500 ms. A final 1024/256-sample anatomy run
used at most 2709 MiB whole-device memory and completed rendering plus a
fresh-process packed-asset reload check in approximately 20 seconds.

```sh
$PY -m server.vhuman.reconstruction.offline_render \
  tmp/vhuman-blender/out_balanced --out tmp/vhuman-quality8h/anatomy_portable \
  --device optix --preset final --accessories omit --no-hair --fit-eyes \
  --blender "$HOME/local/blender-5.2.2-linux-x64/blender" \
  --head-fit /mnt/nvme02/models/vhuman-texture-inputs/obama/head_fit.json \
  --parsing-model "$PARSER"
```

The source eye-fit metadata was copied from the authorized b550 experiment.
`--no-hair` omits scalp strands and undercoat; eye-region lashes and wet lines
remain part of the ocular assembly. Native teeth, gums, tongue and cavity stay
separate from the skin and receive their existing physical material priors.

Initial regression: 59 anatomy, fitting, photoreal and mobile-preprocessing
tests pass. Three new tests cover bounded ocular fitting, unreachable targets
and invalid observations. GPU parity and rendered asset-reload checks are
separate real-hardware evidence.

## Oral visibility audit

The bright, low-chroma source pixels inside the eight observed inner-lip
landmarks provide a conservative tooth-region heuristic (897 pixels). Rasterizing
the complete native anatomy with the **fitted** camera finds 370 visible tooth
pixels, with 308 overlapping the heuristic: IoU 0.32117 and recall 0.34337.
Of the target pixels, 585 are occluded by exterior skin, specifically 177 by
upper-lip triangles and 408 by lower-lip triangles. Cavity triangles occlude
none. This identifies lip surface coverage as a fitting defect even though
inner-lip attachment projections closely match the source.

The numerical-pass geometry variants only slightly improve this measurement:
`fit_surface1` has IoU 0.33089 (374 visible tooth pixels), and `fit_combined`
has IoU 0.33438 (380 pixels). Neither resolves the oral appearance problem.

A 25-trial exploratory sweep applies graph-smoothed upper/lower lip offsets
of 0, 0.5, 1, 1.5 and 2 mm in camera-up/down directions. The largest offset
improves IoU to 0.52422, but **all 24 nonzero trials reverse some triangle
orientations**. None is promoted. Larger tooth exposure is not sufficient
evidence of a usable fit; future fitting must jointly constrain visible lip
surfaces, native expressions and topology. These trials also demonstrate why
landmark error alone is inadequate. Receipts and the exploratory script are
in `tmp/vhuman-quality8h/oral_audit/`.

The reusable audit records geometry/portrait hashes, the final camera,
thresholds, masks, visible triangle IDs, depth, overlay and overlapping anatomy
group counts. It deliberately makes no acceptance decision: bright pixels are
not independent dental ground truth, and source-view coverage cannot validate
motion, tooth shape or shading. Three unit tests cover mask exclusion, overlap
and invalid inputs; the real source audit reproduces the counts above.

```sh
TMPDIR="$PWD/tmp" OMP_NUM_THREADS=2 tmp/vhuman-texture-venv/bin/python \
  -m server.vhuman.reconstruction.oral_visibility \
  /mnt/nvme02/models/vhuman-texture-inputs/obama/material12 \
  --out tmp/vhuman-quality8h/oral_visibility_baseline
```

Use a fresh output directory for each run.

## Directionally constrained lip experiments

The first failed 0.5 mm lower-lip shift reverses only two very thin exterior
skin triangles (double areas approximately 0.052 and 0.108 square millimetres).
`directional_surface` exploits a restricted deformation: all vertices move
along one fixed unit direction, with a scalar displacement per vertex. The
oriented area ratio is then exactly affine in those scalars. Dykstra projection
enforces linear area inequalities and a displacement box, reporting convergence
explicitly. This is an orientation constraint relative to reference geometry,
not a proof of self-intersection freedom or anatomically correct motion.

All 25 trials were repeated with a minimum ratio of 0.2 relative to both neutral
and captured reference poses. For example, 0.5 mm upper/1 mm lower lip targets
retain IoU 0.42390, 497 visible tooth pixels and only one visible tooth pixel
outside the mouth polygon. The maximum adjustment to the proposed displacement
field is 0.189 mm. However, weighted mouth landmark error increases from
0.98633 to 1.29568 px and held-out mouth error from 1.33217 to 1.36986 px.
This does not pass the existing fit acceptance gate.

A 21-pose neutral-to-captured expression ramp also found 11 triangle reversals
across intermediate poses despite the endpoint constraints. That version is
rejected. Including all 21 poses in the projection converges in 1921 sweeps;
a denser 101-pose evaluation then has no orientation reversals and minimum
ratio 0.2. This covers only the recorded linear expression ramp, not independent
jaw/gaze animation, collision checks or continuous-time guarantees.

The endpoint candidate was rebaked from the portrait and rendered at 1024 pixels
with OptiX: packed-asset reload passed, 2711 MiB peak whole-device memory,
20.1 seconds for rendering and validation. More teeth are visible, but dental
shading and source matching still need work. The sampled-ramp candidate is
stored separately as `tmp/vhuman-quality8h/lip_candidate_ramp`; no candidate
from these experiments replaces the promoted texture/geometry. Its independent
oral audit reproduces IoU 0.42390, and its own OptiX render and packed-asset
reload pass. Next fitting
should constrain the observed lip attachments while adjusting intervening
occluding surfaces, rather than accepting the landmark regression.

Regression: 37 tests pass, including six directional projection tests covering
the exact area identity, analytical projection solution, multiple poses,
nonconvergence reporting and invalid/degenerate inputs. Command:

```sh
TMPDIR="$PWD/tmp" OMP_NUM_THREADS=2 tmp/vhuman-texture-venv/bin/python -m unittest \
  server.vhuman.test_directional_surface server.vhuman.test_oral_visibility \
  server.vhuman.test_reconstruction
```

## Fixed-attachment lip fitting

`directional_surface` now optionally constrains barycentric attachments to zero
displacement, using equality projections alongside the area and displacement
constraints. An experiment fixes all 468 canonical face attachments while
retaining the 21 expression-pose constraints. A strict 0.0000001 mm tolerance
does not converge within 10,000 sweeps; that result remains diagnostic only.
Repeating with an explicit 0.001 mm tolerance converges in 2299 sweeps.

The converged trial at `tmp/vhuman-quality8h/oral_anchored_micron` retains tooth
IoU 0.42331 (495 visible tooth pixels, 414 overlapping the 897-pixel heuristic).
Maximum landmark shift is 0.001377 px; weighted mouth error is 0.98621 px versus
the original 0.98633 px, and held-out mouth error is 1.33216 px versus 1.33217 px.
The denser 101-pose ramp has no reversals and minimum area ratio 0.35572. Thus
the earlier landmark regression is resolved without losing the visibility gain.
This is a constrained surface experiment, not an independently held-out fit:
all attachments, including the original held-out subset, are fixed to their
baseline 3D positions rather than newly fitted to their image targets.

Orientation and landmarks still do not fully characterize quality. Around the
lips, triangle normal changes have median 2.86 degrees, 95th percentile 13.14
degrees and maximum 73.02 degrees; maximum edge stretch is 2.02x. Those local
distortions require visual inspection and further regularization/contact work
before promotion. The freshly rebaked candidate is
`tmp/vhuman-quality8h/lip_candidate_anchored`. No synthetic-completion provenance
is carried across the geometry change. Its OptiX render and fresh-process packed
asset reload pass (20.2 seconds, 2709 MiB peak whole-device memory). Front-view
inspection retains the visibility improvement; it does not resolve the measured
local distortion or prove contact quality.

The regression command above now passes 39 tests, including new checks for
fixed barycentric attachments and invalid attachment indices.

## Edge-limited lip fitting and block attachment projection

The directional solver can also limit edge stretch. For each edge and reference
pose, the length bound reduces to an interval for the scalar endpoint-offset
difference. Intersecting those intervals across poses gives two linear
inequalities per unique edge, avoiding a separate edge constraint set for every
pose. This limits stretching, not bending or self-intersection.

At the previous 0.5/1 mm upper/lower targets, a 1.25x edge limit preserves the
same tooth IoU 0.42331 and landmark accuracy. Maximum stretch over the denser
101-pose ramp is 1.250087x, versus 2.02x before. The largest lip-triangle normal
change falls from 73.02 to 51.79 degrees (95th percentile 12.70 degrees). The
freshly rebaked `lip_candidate_edge125` passes OptiX rendering and asset reload.

A stronger 1/2 mm target first fails to converge within 10,000 sweeps of the
individual attachment projections. The attachment constraints are now projected
as one subspace using the pseudoinverse of their Gram matrix, retaining Dykstra
corrections for the combined constraint sets. A regression test checks dependent
attachments. The same stronger problem then converges in 124 sweeps at 0.001 mm
tolerance. At a tighter 0.000001 mm tolerance, it converges in 872 sweeps with
maximum attachment residual 2.3e-11 mm; the tolerance was tightened, not relaxed.

The tight stronger result (`oral_edge125_block_tight`) has IoU 0.48951, 594
visible tooth pixels, 490 target overlaps and 10 visible tooth pixels outside
the coarse mouth polygon. Maximum projected landmark shift is approximately
3.2e-11 px. The 101-pose ramp has no orientation reversals, minimum area ratio
0.2 and maximum edge stretch 1.250089x. Intermediate sampled poses can slightly
exceed the bound imposed at the 21 constraint poses; this is not a continuous
motion guarantee. The candidate is `lip_candidate_edge125_strong`. Its OptiX
render and fresh-process asset reload pass (20.2 seconds, 2713 MiB peak device
memory). The camera-aligned mouth comparison is
`tmp/vhuman-quality8h/oral_edge_comparison.png`. It shows the greater tooth
exposure, but substantial crown-shape, gum-exposure and shading differences
remain relative to the portrait. The source portrait and studio render also
have different illumination; the comparison is not an albedo measurement.

Regression: 42 tests pass. Both candidate strength settings remain experimental:
exact mesh contacts, unseen views, broader motion and dental shading still need
validation before promotion.

## Native mesh crossing audit and dental pose experiment

`mesh_crossings` adds a batched spatial broad phase and strict transverse
triangle-crossing test. It excludes shared-vertex pairs, degenerate facets,
coplanar overlaps and boundary-only touches. It does not measure penetration
depth or prove collision freedom. Tests include analytic crossing/separation,
degeneracy, symmetry and broad-phase agreement with brute force. An independent
Blender ray/triangle check agrees on 699 sampled source-pose pairs (200 positives)
after geometry is expressed in millimetres with normalized ray directions;
the initial metre-scale Blender check had numerical mismatches. Both diagnostic
workers finished their receipts but required termination after an audio-thread
shutdown hang in the sandbox; the GPU render/reload workers exited normally.

The baseline has **595 upper/lower crown crossing pairs** in the captured pose,
separate from gum/root and mouth-cavity intersections. The lower dental arch's
mean captured displacement is 3.53 mm upward relative to its neutral geometry.
GNM lower-face expression coefficients move the lower arch and tongue; the
model's four skeletal joints are neck/head/eyes, with no explicit jaw joint.
The earlier skin-only landmark fitting did not constrain these dental contacts.
Both edge-limited lip candidates retain the same crown-crossing problem because
they do not change the dental arches. Across five sampled poses, the stronger
lip candidate reduces captured exterior-skin self-crossing pairs from 14 to 4,
but also changes hidden mouth-cavity contacts.

A lower-arch/tongue translation sweep (0–6 mm camera-down) reduces crown
crossings from 595 to 44 but does not eliminate them. A rigid alignment of the
captured lower arch toward its neutral prior uses a 7.04-degree rotation and
has 0.339 mm RMS fit residual. Adding 1 mm camera-down is the smallest tested
increment with zero upper/lower crown crossings (0.5 mm retains 27).

The resulting `lip_candidate_oral_aligned` preserves skin geometry and its
maps exactly. The source-frame correction is encoded as a fixed bind residual,
with maximum displacement 11.46 mm across the lower arch and tongue; it is a
prior correction, not measured dental pose. Its 101-pose orientation ramp has
minimum ratio 0.96269 relative to the preceding geometry and no reversals. In
all five contact-audit poses, upper/lower crown crossings are absent. However,
captured lower-crown/mouth-cavity crossings increase to 284, and 42 lower-arch
and 11 upper-arch within-arch crown crossings remain. Total strict oral crossing
pairs decrease from 1926 to 1123, which is not sufficient for acceptance.

The candidate's OptiX render and packed-asset reload pass. Tooth-mask IoU falls
from 0.48951 to 0.46999 as the interpenetrating lower crowns are removed from
the visible row. This reinforces that coverage alone is not a quality gate.
The dental correction remains experimental pending coordinated cavity/contact
work. Receipts are in `oral_contacts/`, `oral_contacts_aligned/`,
`oral_arch_sweep/`, and `lip_candidate_oral_aligned_audit/` under the study root.

Regression: 47 tests pass with `server.vhuman.test_mesh_crossings` added to the
previous test command.

## Cavity experiments, visible contacts and individual tooth spacing

Terminology correction: earlier "crown" counts refer to triangles assigned to
the tooth material rather than gums. The asset does not separately label crowns
and roots. Crossing counts alone therefore do not establish visible defects.

Nine graph-smoothed cavity corrections keep exterior skin vertices fixed.
The selected stiffness-1, strength-1 trial moves only mouth-sock vertices, by at
most 4.624 mm, and has minimum area ratio 0.98439 across 21 poses. Cavity crossing
pairs fall from 867 to 717, and the complete audited oral subset falls from 1123
to 973. Alternative radial corrections around a convex envelope of the oral
geometry reduce crossings further but require roughly 30 mm displacement and
reverse faces; all are rejected. No exterior face texture needs rebaking for
the selected cavity-only change.

`crossing_points` now exposes segment endpoints for visibility checks. Blender
BVH rays sample five points per crossing segment from the fitted source camera,
using a 0.05 mm depth tolerance. This finds 14 source-visible crossing pairs in
the baseline, 10 after dental alignment, and 10 after cavity smoothing. None of
the sampled cavity crossings is source-visible; the remaining 10 are three
lower-lip pairs and seven upper-tooth pairs. This is a sampled source-view
diagnostic, not proof for all viewpoints or boundary/coplanar contacts.

The seven visible upper-tooth pairs all involve the two central incisors. The
upper and lower tooth surfaces each have 16 disconnected components. New
`component_spacing` projects bounded rigid component translations onto
separating-plane constraints, preserving each selected tooth's shape. Cascaded
neighbor contacts are rechecked. Upper translations are at most 0.162 mm and
lower translations at most 0.364 mm, under the 0.5 mm limit with 0.02 mm target
clearance. Captured-pose tooth intersections fall from 11/42 (upper/lower) to
zero, but that initial proposal folds adjoining shared-vertex gum triangles
and is rejected as a complete-mesh candidate.

Screened harmonic extension into the gums preserves the rigid tooth offsets.
A bounded area-gradient repair then adjusts only free gum vertices, by at most
0.010798 mm, to meet a 0.15 oriented-area floor. It converges in five steps; a
101-pose check has no reversals and minimum ratio 0.15000001. Production helpers
reproduce the prototype delta exactly. Every tooth remains rigidly translated
within each evaluated pose. These helpers explicitly return rejection when
their displacement or iteration limits are exceeded.

The combined `lip_candidate_teeth_spaced` has zero detected crossings within
either tooth arch or between them in all 21 sampled expression poses. Blender
source-view ray auditing finds only three remaining visible pairs, all in the
lower lip. Total strict crossings in the audited oral subset are 913, mostly
hidden contacts; zero tooth crossings does not imply zero gum/cavity contacts.
The final OptiX render and packed-asset reload pass (20.5 seconds, 2709 MiB peak
device memory). This is the working anatomy candidate, still unpromoted pending
lip repair, broader motion/side-view checks and completed-material transfer.

Receipts: `cavity_sweep/`, `cavity_envelope/`, `contact_visibility/`,
`tooth_spacing/`, and `lip_candidate_teeth_spaced/` under the study root.
Regression: 53 tests pass, adding `server.vhuman.test_component_spacing`.

## Next experiments

- Repair the three remaining source-visible lower-lip crossings while preserving
  the observed face attachments; validate side views and broader oral motion.
- Inspect tongue/cavity, lip contact and eye-lid contact during jaw opening,
  gaze changes and side views.
- Evaluate the passing geometry candidates before transferring completed skin
  with explicit changed-geometry provenance.
- Continue selective photo-projection repairs and surface texture studies on
  the selected geometry, preserving reliable source detail.
- Export complete anatomy comparisons and run browser/native motion validation.

## Local lip unfolding and cumulative geometry guard

The directional solver now supports explicit scalar attachment targets. Zero
remains the default; incompatible dependent constraints report nonconvergence.
A local lower-lip triangle is displaced 0.075 mm along its normal while six
vertices on the opposing sheet and all 468 facial attachments remain fixed.
Unpinned smooth-field trials moved both sheets together and were rejected.
The selected pinned trial converges in 42 sweeps and removes all four local
skin crossing pairs, with negligible projected facial-attachment displacement.

A cumulative check against the original geometry catches a gum triangle with
area ratio 0.0274, despite passing the previous per-step guards. An additional
gum-only correction of at most 2.762 micrometres restores a cumulative 0.15 area
floor without moving teeth or skin. Across 101 neutral-to-captured samples,
the complete geometry has minimum oriented-area ratio 0.15000001 and no face
reversals. This ramp does not cover arbitrary expressions or contact clearance.

The resulting `lip_candidate_unfolded_guarded` passes OptiX rendering and
packed-asset reload (20.45 seconds, 2711 MiB peak whole-device usage). Blender
intersection-segment ray checks find zero source-visible oral crossing pairs,
versus 14 in the original and three in the tooth-spaced predecessor. Across
camera yaw -60/-30/0/30/60 degrees, visible counts are 4/0/0/0/4, versus original
14/12/14/19/6. Remaining extreme-angle pairs are between upper tooth-material and gum
triangles; they require gum-transition correction while keeping teeth fixed. Total strict oral
pairs remain 909, mostly hidden. These are finite segment samples with opaque
geometry, not proof of general collision freedom.

This is the working anatomy candidate, not a promoted completed material.
Completed skin transfer, broader expression/eye-contact checks, and complete
anatomy export remain open. Receipts are `lip_unfold_selected/`,
`lip_candidate_unfolded_guarded/cumulative_area_ramp.json`, and
`contact_visibility/report_side_views.json` under the study root.
Regression: 55 tests pass across directional surface, component spacing,
mesh crossings, oral visibility, and reconstruction modules.

## Joint gum contact and area constraints

The eight distinct extreme-angle visible pairs involve upper tooth/gum
transitions. Nine separating-plane plus harmonic-extension trials reduce the
contacts, but sequential area repair reintroduces four targeted pairs. These
trials are rejected. A joint SLSQP experiment instead minimizes the displacement
of ten free gum vertices under tooth-plane clearance, original-mesh area floors
in 21 reference poses, and bounded coordinate displacements. Teeth, exterior
skin, and all other vertices remain fixed. Clearances of 0.001/0.005/0.02 mm
all solve; selected 0.005 mm clearance needs at most 0.108602 mm displacement.
The larger margin requires 0.219964 mm and is not selected.

`lip_candidate_gum_joint` removes all eight targeted pairs. Upper-arch strict
crossings fall to five and total audited oral pairs to 897; remaining hidden
contacts are not asserted harmless in arbitrary motion. A 101-pose cumulative
area check has minimum 0.15 and no reversals. Skin arrays/maps are unchanged,
so no new photographic bake is claimed. A final OptiX render at 60-degree yaw
and packed-asset reload pass; side skin remains visibly incomplete, reinforcing
the need for the planned completion transfer.

Blender rays across yaw -60/-30/0/30/60 find no sampled visible crossings in the
captured pose. A five-pose neutral-to-captured audit finds none at ramp fractions
0, 0.25, 0.75, and 1.0. At 0.5, two skin/lower-lip crossing pairs become visible
from -60, -30 and zero yaw. This is a newly localized motion defect, not a failed
dental correction; the candidate remains experimental pending its repair and
broader motion/eye/texture validation.

Reproduction scripts and receipts under `tmp/vhuman-quality8h/`:
`gum_contact_sweep.py`, `gum_joint_solve.py`, `build_gum_joint.py`,
`gum_joint/report.json`, `prepare_gum_ramp.py`, `trace_gum_ramp.py`, and
`contact_visibility/report_gum_ramp_views.json`. These experiments change no
production solver code. The 55-test regression from the preceding commit remains
the latest code regression; this step adds geometric and rendered evidence.

## Landmark-preserving contact repair across the expression ramp

The halfway-expression defects share triangle 10717 at one mouth corner.
Restricting changes to its two unanchored neighboring vertices fails joint
full-ramp contact/area constraints. Solving only the 0.4–0.6 interval converges
but leaves contacts elsewhere on the ramp; those proposals are rejected.

Expanding to a 19-vertex one-ring patch and solving in the null space of its
facial-attachment matrix retains all fitted barycentric landmark positions.
The patch has 16 scalar null-space dimensions. A joint constrained solve over
101 neutral-to-captured poses converges in five iterations with 0.005 mm
separating-plane clearance and maximum vertex displacement 0.041806 mm. Both
targeted pairs are absent throughout that sampled ramp. The cumulative full
mesh area ratio remains at least 0.15, with no reversals; maximum attachment
displacement is 6.985e-18 mm. Numerical precision, not anatomical accuracy, is
what this attachment metric establishes.

`lip_candidate_ramp_patch` has a fresh portrait bake, unchanged measured landmark
errors, and passes final OptiX rendering plus packed-asset reload (20.64 seconds,
2709 MiB peak whole-device memory). Full audited oral crossing enumeration at
11 ramp fractions followed by Blender rays at five yaw angles finds zero
sampled visible pairs in all 55 pose/view combinations. Hidden strict contacts
remain (897 in the captured pose), and arbitrary expressions, boundary-only,
coplanar and containment cases are not covered by this test.

This is the working geometry for completed-skin transfer, still unpromoted.
Next gates include appearance transfer with changed-geometry provenance, wider
oral motion, eyelid/gaze contact and full-anatomy export/browser comparisons.
Scripts/receipts under `tmp/vhuman-quality8h/`: `lip_ramp_joint.py`,
`lip_ramp_local.py`, `lip_ramp_patch.py`, `evaluate_lip_ramp_patch.py`,
`build_lip_ramp_candidate.py`, `lip_ramp_patch/evaluation.json`, and
`contact_visibility/report_lip_patch_ramp_views.json`.

## Completed appearance transfer onto repaired anatomy

New `transfer_skin` requires identical skin topology/UVs, the same source
portrait, a fresh target bake, a validated synthetic prior and a bounded skin
displacement (3 mm default). It transfers only prior-supported texels that are
unobserved in both bakes, with a metric feather on the new geometry. It preserves
new photographed texels exactly and excludes old photographic projection from
the prior. Both geometry hashes and all input/output mask hashes are recorded;
old multiview support scores are deliberately not inherited. Candidate validation
now detects modified transfer, coverage and generated-support maps.

`completed_anatomy` transfers `out_balanced` onto `lip_candidate_ramp_patch`.
Maximum skin displacement from the old prior is 2 mm; geometry is byte-identical
to the repaired candidate. All 175458 photographed texels remain unchanged,
716736 unobserved texels receive a nonzero prior blend, and 11 formerly observed
texels are explicitly excluded. These counts describe prior reuse, not newly
recovered detail or new camera coverage. The low-frequency seam metric improves
from 0.0469343 to 0.0431738, but remains worse than the old-geometry balanced
material's 0.0367668. Side rendering shows added detail alongside uneven tone;
this material is experimental and still needs matched appearance refinement.

OptiX rendering at 60-degree yaw and packed-asset reload pass with 2712 MiB peak
whole-device usage. Regression: 60 tests pass across `test_transfer_skin`,
`test_surface_texture`, `test_generated_skin` and `test_reconstruction`.
Tests include exact photograph protection, target geometry provenance, omitted
old coverage claims, transfer-map tampering, and excessive geometry-change
rejection. Receipts: `completed_anatomy/generated_skin.json`,
`completed_anatomy_render60/`, and `transfer_metrics.json` under the study root.

## Tone comparison on completed repaired anatomy

A new Blender surface audit covers 896701 skin samples on the repaired mesh.
It flags 647 of 175458 photographed texels as occluded at the 0.25 mm source-ray
tolerance; this pass leaves all photographed texels unchanged. The static skin
USD roundtrip preserves triangle/UV data and texture pixels, with maximum
coordinate error 7.45e-9 m. This is still skin-only USD, not complete anatomy.

Three normal-compatible surface-tone trials use strength/prior 2/0.5, 8/0.3,
and 20/0.1. Seam metrics are respectively 0.0400561, 0.0366182 and 0.0334584,
versus 0.0431738 before correction. Each retains geometry and all photographed
texels exactly. Four matched OptiX renders reuse the same packed full-anatomy
scene, camera, lights, seed and materials, changing only the basecolor image.
Side tone becomes more even; the strongest trial smooths broad color variation
more aggressively. `completed_tone_balanced` is selected as the working material
for continued validation (15.18% lower seam metric), not promoted as a final
reconstruction. Scalp mottling and source-projection artifacts remain visible.

The refinement receipt now describes inherited appearance support accurately
for transferred priors and gives each output its own candidate ID. Ten focused
surface/transfer tests pass. All three real outputs also pass candidate hash
validation and exact geometry/photographic-pixel comparisons. Receipts under
`tmp/vhuman-quality8h/`: `completed_surface_audit/`,
`completed_tone_sweep.json`, `tone_comparison/report.json` and
`tone_comparison/comparison.png`.

## Optical-eye contact audit and bounded placement trials

The source iris-alignment fit does not constrain the eyelids. A new diagnostic
samples both landmark-derived lid contours (96 samples each) and casts rays
from each globe centre into the actual cornea/sclera mesh. Nine gaze offsets
cover yaw -15/0/15 and pitch -10/0/10 degrees, with captured eyelids held fixed.
This measures radial contour-to-shell clearance, not volumetric contact or
anatomical ground truth; canthus samples can legitimately lie beyond the globe.

The reference mesh audit finds source-pose minima around -0.66 mm on both lower
contours and -0.32/-0.36 mm on upper contours. Maximum corner gaps exceed 5 mm.
Across the sampled gaze set the minima reach -0.893/-1.086 mm (right/left).
The analytic shell and explicit triangle-ray results agree closely enough for
bounded optimization trials, but remain separately recorded.

Initial movable-centre fits improve average gaps while worsening local
penetration and exceeding a 2 mm displacement norm; they are rejected. Penalty
trials improve contact but still slightly exceed the desired hard bounds.
A joint SLSQP trial enforces centre shift <=2 mm and clearance >=-0.25 mm on
central contour samples in the captured gaze, while fitting iris centre/radius.
It converges for both eyes. Iris-centre errors are 0.00768/0.01064 px and radius
errors 0.03211/0.00830 px (right/left). Central clearance RMS toward the authored
0.15 mm target falls from 2.072/1.738 mm to 1.004/1.036 mm. This target is a
fitting prior, not a measured tissue thickness.

Matched OptiX baseline/trial renders keep skin, lights, camera and seed fixed.
The eye appearance remains similar; Blender rays independently confirm source
lower-contour penetration around 0.249 mm and no upper-contour penetration.
However, gaze sweeps still reach 0.878/0.524 mm penetration. The experimental
placement is not promoted or integrated into motion. The next optical fit must
constrain multiple gaze poses together and inspect eyelid motion, rather than
using source-pose convergence as a general contact guarantee.

Receipts under `tmp/vhuman-quality8h/`: `eye_contacts/report.json`,
`eye_contacts/fit_sweep.json`, `fit_penalty_sweep.json`, `fit_hard_bounds.json`,
`eye_contact_comparison/eyes_comparison.png`, and
`eye_contact_comparison/contact_audit.json`. Reproduction scripts are
`prepare_eye_contacts.py`, `trace_eye_contacts.py`, `fit_eye_contact*.py`,
`render_eye_contact_trial.py`, and `trace_eye_trial.py` in the study root.
No production fitter or accepted candidate geometry is changed by this trial.

## Joint multi-gaze optical placement

Nine-gaze contact constraints converge but a denser check detects a roughly
0.0004 mm violation between samples. A subsequent joint fit uses all 117 gaze
samples (13 yaw values across +/-15 degrees, nine pitch values across +/-10
 degrees), with captured lids fixed. SLSQP's 1e-12 stopping tolerance stalls at
numerical residuals around 3e-11 mm; a 1e-9 stopping tolerance converges while
retaining the same independently checked hard constraints.

Both eye-centre shifts remain at the 2 mm norm bound. Iris-centre errors stay
below 0.018 px and radius errors below 0.041 px. Source-pose central contour
RMS is 1.439/1.272 mm (right/left), better than the original 2.072/1.738 mm but
worse than the source-only placement trial. The analytic minimum is -0.25 mm
across all constrained samples. Independent Blender mesh rays across all 234
eye/gaze combinations find a minimum of -0.249395 mm. The matched frontal render
retains a similar eye appearance.

This does not solve lid contact: source-pose upper-contour median gaps remain
about 1.41/1.29 mm and corner gaps exceed 4 mm. The placement is retained only
as an experiment for a coupled lid/eye study, not promoted. Fixed lids limit
what a single translated/rotated globe can satisfy across gaze. Next work needs
surface-bound eyelid fitting or gaze-driven lid motion, plus apparent-iris and
blink checks, while preserving the observed facial projection.

Receipts under `tmp/vhuman-quality8h/`: `eye_contacts/fit_multi_gaze.json`,
`eye_contacts/fit_dense_gaze.json`, `eye_dense_comparison/eyes_comparison.png`,
`eye_dense_comparison/contact_audit.json`, and
`eye_dense_comparison/mesh_dense_audit.json`. Scripts: `fit_eye_multi_gaze.py`,
`fit_eye_dense_gaze.py`, `render_eye_dense_trial.py`, and `trace_eye_117.py`.
No production eye fitter, material or skin geometry changes in this experiment.

## Projection-preserving lid targets and inner-lid tissue experiment

Blender camera-ray queries test whether moving existing contour samples only
in depth can meet the optical shell. The baseline right eye has 19 missed rays
out of 192; its left eye has none. The dense-gaze placement has zero/six misses
(right/left), but the farthest successful corner targets still need roughly
8.2/6.5 mm depth movement. Baseline extremes need 11–13 mm. Only about 110 of
192 samples per eye fall within a 2 mm depth shift for the dense-gaze placement.
This rejects a blanket camera-ray snap as a bounded lid-fitting solution.

A separate static experiment adds four radial inner-lid tissue strips between
the unchanged observed contour and the dense-gaze optical shell. The strip
uses an authored muted pink wet material, a 0.025 mm shell offset, and four
radial rows; it does not move skin or redefine photographic evidence. Matched
OptiX renders show a small corner fill while retaining the eye opening. Mean
absolute RGB change over the eye review rectangle is 0.154/255. This is an
appearance difference, not a source-photo accuracy score.

Each strip has 384 vertices and 570 triangles. Static checks find no degenerate
triangles or strict self-crossings within any strip. Inter-component contact,
actual skin attachment and deformation during blink/gaze are unverified; the
optional tissue geometry is not integrated or promoted. Its useful static
result motivates an eventual surface-bound implementation rather than large
unobserved changes to the facial contour. Complete-anatomy USD export is the
next independent deliverable while this coupled lid/eye issue remains open.

Receipts under `tmp/vhuman-quality8h/`: `lid_ray_targets/report.json`,
`eye_conjunctiva_comparison/eyes_comparison.png`, `conjunctiva.npz`, and
`geometry_audit.json`. Scripts: `audit_lid_ray_targets.py`,
`render_conjunctiva_trial.py` and `launch_conjunctiva_trial.py` in the study root.

## Complete static anatomy USD and material restoration

The reusable `usd_anatomy_worker.py` exports 18 meshes: skin, cavity, upper/lower
teeth and gums, tongue, bilateral optical eye parts, tear lines and lashes.
It requires matching prepared-scene geometry/portrait and permits only a
basecolor refinement without rebuilding the scene. USD uses metres, Y-up and
relative texture paths. Original per-component and GNM arrays accompany the
package as NPZ. Experimental translated eyes and inner-lid strips are excluded.

The first USD import preserves cornea transmission but drops subsurface weights
and parts of the native material setup. New `usd_materials.py` captures a
checksummed static Blender node/image sidecar and restores it after USD import.
It rejects nested node groups and preserves static socket values only, not
animation drivers. The exporter preserves subdivision settings separately.

The completed `full_anatomy_usd` bundle passes geometry, UV, material binding,
node/socket/link, and packed-image checks. Relocating the whole directory and
running `usd_import_worker.py` in a fresh process passes independently: 18 meshes,
11 materials, 67 nodes and 18 packed images; maximum coordinate error is
1.77e-8 m, with exact triangle order and UV values. No source render directory
is read by the importer. The restored scene is saved as a portable `.blend`.

A matched OptiX render after USD import and material restoration differs from
the native scene by mean 0.2635/255 foreground RGB and 99th-percentile 2/255;
56 alpha pixels differ. The images are visually close, not pixel-identical.
This proves static interchange for the tested Blender version, not arbitrary
USD shader support or motion export. Native expression/skeletal animation is
still absent. See `ANATOMY_USD.md` for the exact export/import commands.

Validation: two real-Blender material tests pass, covering roundtrip node/image
restoration, modified shader rejection, corrupted image rejection and external
asset-path rejection. Python compilation and `git diff --check` also pass.
Artifacts under `tmp/vhuman-quality8h/`: `full_anatomy_usd/`,
`full_anatomy_usd_relocated/portable.validation.json`,
`usd_render_comparison/comparison.png`, and `usd_render_comparison/metrics.json`.

## Wider native-motion stress tests and bounded pose corrections

Three existing 22-frame tracks (surprise, blink, gaze) were located under
`b550:/mnt/disk1/tmp/vhuman-hopper-photoreal/` and copied with their metadata to
`/mnt/nvme02/models/vhuman-motion-priors/hopper/` (about 14 MB total). They belong
to a different subject. Only relative expression/joint changes are reused as
explicitly labeled cross-subject stress priors at strengths 0.25/0.5/1.0; this
is not observed motion or an expression-accuracy evaluation of the current face.
The source motion hashes are verified before GPU evaluation.

The first audit compared against the original, unrepaired neutral surface. It
also flagged three triangles in the starting pose, although the previous
cumulative checks compared against matching original expression poses. These
are different normal-orientation references. That initial audit and correction
are retained as diagnostics, not accepted as proof of new motion failures.
The corrected dynamic gate uses the current repaired bind surface under the
same joint pose, matching the native rig's reference. Its starting pose has no
orientation failures and reproduces the captured coordinates within 0.000062 mm.

Against the repaired bind surface, quarter-strength blink and half-strength
gaze need no meaningful correction. Surprise probes flag one skin triangle;
stronger blink flags up to five, and full gaze flags one. Defects cluster at
mouth corners and some eyelid triangles. Bounded per-frame area-gradient repair
uses exterior skin/gums as free vertices and keeps tooth vertices fixed.
All nine sampled tracks converge under a 0.25 mm displacement cap. Maximum
actual repair is 0.076335 mm and maximum projected landmark shift is 0.097904 px.
The initial frame remains byte-identical to each input track after float32 save.

A nominal 0.1 area floor in float64 fell as low as 0.099831 after float32 save.
Using a 0.101 solver target and testing saved precision restores the explicit
0.1 gate: the minimum measured float32 ratio is 0.100869 across all 198 sampled
frames. These are per-frame orientation checks, not proofs for interpolation,
self-contact or temporal appearance. No motion is promoted yet; contact audits,
representative animation renders and browser/native comparisons remain open.

Receipts/scripts under `tmp/vhuman-quality8h/`: `motion_stress_current_bind/`,
`motion_current_bind_repaired/report.json`, `audit_motion_current_bind.py`, and
`repair_motion_current_bind.py`. The preliminary original-neutral results remain
in `motion_stress/` and `motion_stress_repaired/` and must not be mistaken for the
current reference or used as accepted animation assets.

## Dynamic oral contacts and rigid-arch counterexample

Visibility tests on the start, peak and final frames of the full-strength
surprise/blink/gaze stress tracks expose defects absent from the static ramp.
Late gaze has eight lower-lip crossing pairs at both mouth corners. Surprise
has tooth/gum, cavity/dental and cavity/tongue contacts, including four visible
cavity/tongue pairs in the frontal peak and final poses. The blink prior only
narrows the eyes; it is not a complete-closure test. These are cross-subject
stress priors, not target-subject expression observations.

A local gaze repair constrains vertices against separating triangle planes
while preserving the facial attachment matrix through its null space. It also
retains at least 80% of each affected triangle's pre-contact-repair oriented
area, caps displacement at 0.3 mm and penalizes changes from the previous
frame's correction. The initial frame is unchanged. The first trial removes
the eight endpoint pairs, but a full-frame/midpoint visibility sweep finds
three additional pairs at frame 13 (zero-based), around triangle 10717.
Those pairs are added to the second joint solve rather than treating the
endpoint-only result as an accepted animation.

The second trial's saved float32 geometry changes by at most 0.018090 mm;
maximum projected attachment shift is 0.00000438 px. All 85 quarter-frame
linear interpolation samples retain an oriented area ratio of at least
0.100749 against the current repaired bind under the matching joint pose.
Maximum correction second difference is 0.036180 mm. These finite samples do
not prove continuous-time collision freedom or acceptable temporal appearance.

A separate lower-dental experiment tests rigid arch motion fitted to the native
track. The upper arch already fits a rigid transform to within 0.000026 mm,
but the lower teeth deviate by up to 2.516 mm at peak surprise. Fitting the
entire lower teeth/gums to their captured shape requires up to 2.672 mm change.
It preserves the original within-arch geometry, including 23 hidden strict
crossing pairs; rigidification cannot remove those existing contacts.

The rigid trial fails its broader contact gate: peak surprise gains four
frontal lower-dental/tongue crossing pairs and additional side cavity contacts.
It is not promoted. Matched OptiX renders show a plausible dental shape, but
appearance alone would miss this failure. The native and rigid peak renders
both pass packed-asset reload validation, with whole-device peaks of 3792 and
3879 MiB respectively. The next dental correction needs a coupled arch/gum,
tongue and cavity treatment, not an isolated rigid replacement.

Receipts/scripts under `tmp/vhuman-quality8h/`: `motion_contacts/visibility.json`,
`motion_surprise_render/render_result.json`, `repair_gaze_contacts_v2.py`,
`motion_gaze_contact_repair_v2/validation.json`, `motion_gaze_contact_audit/`,
`rigid_lower_arch_motion.py`, `motion_rigid_lower_arch/report.json`,
`motion_rigid_arch_contacts/visibility.json`, and
`motion_arch_comparison/comparison.png`. No full-anatomy motion is promoted.

The expanded gaze trial then passes the independent oral visibility sweep:
43 time samples (22 stored frames and 21 linear midpoints) times five yaw views
produce 215 checks with zero detected visible crossing pairs. This audit uses
five samples along each strict intersection segment, opaque BVH visibility and
0.05 mm depth tolerance. Hidden crossings, coplanar contact, unsampled times
and unsampled views remain outside that result. The corrected frame 13 also
renders and passes packed-asset reload, at 3878 MiB whole-device peak. See
`motion_gaze_contact_audit_v2/visibility.json` and
`motion_gaze_repaired_render/render_result.json`.

Localization of the captured lower arch's 23 strict crossings finds 22
pairs with exactly one gum triangle and one gum/gum pair; none are tooth/tooth.
`lower_arch_bind_contacts.json` records their triangle IDs for the next bounded
gum repair. This explains why arch rigidification alone preserves the problem.

## Lower gum correction trials and Blender trimming

The lower dental mesh has 196 vertices shared by the `teeth` and `gums`
groups. These form the open boundaries of the separately rendered teeth/gum
surfaces. Consequently, preserving every tooth vertex also fixes these gum
boundary vertices. The hard cases occur at these root junctions.

Several bounded, fixed-tooth displacement experiments were tested against 21
neutral/captured samples. The initial tooth-plane formulation imposed
contradictory half-spaces on adjacent gum triangles; consistent plane sides
remove that implementation error. Expanding from 29 to 57 free gum vertices
clears the original pairs, but requires 0.673 mm despite a 0.5 mm cap and leaves
11 captured-pose crossings elsewhere. It is rejected. Adding neighboring pairs
and a 1 mm cap reduces the captured crossings to two, but leaves contacts in
the neutral portion and fails its constraints. That result is also rejected.

An alternative uses triangle normals and edge-cross-product separating axes.
It limits displacement to about 0.236 mm and reduces captured lower-arch
crossings from 23 to four, but does not converge to a feasible joint solution.
The selected axes plus linearized area constraints are infeasible; excluding
one difficult tooth face, examining only the captured pose, and testing an
unsigned-area variant do not resolve that linearized problem. These failures
are specific to these axes/linearizations and do not prove that no bounded
nonlinear correction exists. None of these offsets is installed in the working
candidate, and the existing signed-area gate is not waived.

A separate Blender experiment changes topology at the peak surprise frame.
It copies evaluated lower teeth to a hidden cutter, caps 196 boundary edges
with 16 faces, verifies that the cutter is closed, and applies an exact Boolean
difference to the open gum surface. Visible teeth are unchanged. Gum polygons
increase from 1188 to 1202 (1246 evaluated triangles), with no degenerate
triangles detected. Contact audits weld exact duplicate coordinates first so
split material-boundary vertices do not produce extra adjacency reports.

At this frame, strict tooth/gum pairs fall from 19 to three; gum self-crossings
increase from two to three. Finite visibility checks show a useful but partial
result: lower-arch self-contact pairs fall from nine to zero at yaw -30 degrees
and from three to one at +30 degrees. The remaining upper-dental/cavity,
cavity/tongue and side lower-dental/cavity contacts persist. Static appearance
is rendered with OptiX. This topology-changing experiment is not promoted;
expression stability, UV continuity, remaining contacts and animated export
still require validation.

Receipts/scripts under `tmp/vhuman-quality8h/`: `lower_gum_repair*/`,
`repair_lower_gums*.py`, `repair_lower_gums_partial.log`,
`repair_lower_gums_captured.log`, `repair_lower_gums_unsigned.log`,
`gum_boolean_trial/{report,contact_audit}.json`,
`gum_boolean_visibility/visibility.json`, `render_gum_boolean_trial.py`, and
`audit_gum_boolean.py`.

## Optional oral microstructure and wet-coat appearance

A matched static shader trial adds a rest-position vertex attribute to tongue
and gum objects, drives fine bump and roughness variation from it, and adds a
restrained wet coat. Authored bump distance is 40 micrometres for tongue and
18 micrometres for gums, at strength 0.25. These values are appearance priors,
not measured tissue detail. The static attribute is intended to follow shape
keys; motion appearance and USD primvar transport are not yet verified.

The unchanged and modified scenes render at 64 samples with seed 0. The mouth
crop changes by mean absolute RGB 0.280/255; the visual difference is subtle and
does not establish a quality gain. Geometry and photographed skin maps are
unchanged. This remains optional, with no production material change. See
`oral_shader_trial/{head.blend,comparison.png,comparison.json,report.json}` and
`render_oral_shader_trial.py` in the study directory.


Enabling Blender Boolean `use_self` improves the static trimming result:
735 gum vertices and 1255 triangles have no detected degenerates or gum
self-crossings. Three tooth/gum strict pairs remain, but none is visible in
the five sampled peak-frame views; the previous +30-degree lower-arch contact
is gone. Remaining visible oral contacts are one upper-dental/cavity pair at
-30 degrees, four cavity/tongue pairs frontally and ten lower-dental/cavity
pairs at +60 degrees. This isolates the next work: animated trim stability and
cavity/tongue coordination. It is still a static experimental alternative.
Receipts: `gum_boolean_self_trial/{report,contact_audit,comparison}.json`,
`gum_boolean_self_trial/comparison.png`, and
`gum_boolean_self_visibility/visibility.json` in the study directory.
