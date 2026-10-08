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

## Next experiments

- Audit actual lip/teeth/tongue mesh crossings and inspect side views of the
  edge-limited candidates before dental placement and shading variations.
- Inspect tongue/cavity, lip contact and eye-lid contact during jaw opening,
  gaze changes and side views.
- Evaluate the passing geometry candidates before transferring completed skin
  with explicit changed-geometry provenance.
- Continue selective photo-projection repairs and surface texture studies on
  the selected geometry, preserving reliable source detail.
- Export complete anatomy comparisons and run browser/native motion validation.
