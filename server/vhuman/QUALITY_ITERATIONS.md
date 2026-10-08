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

## Next experiments

- Quantify dental silhouette and color against the source mouth region; test
  bounded arch placement and oral shading variations.
- Inspect tongue/cavity, lip contact and eye-lid contact during jaw opening,
  gaze changes and side views.
- Evaluate the passing geometry candidates before transferring completed skin
  with explicit changed-geometry provenance.
- Continue selective photo-projection repairs and surface texture studies on
  the selected geometry, preserving reliable source detail.
- Export complete anatomy comparisons and run browser/native motion validation.
