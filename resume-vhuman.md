# Resume vhuman texture and contact quality work

## Current state

Repository: `/mnt/nvme02/work/gemm/pixal3d`, branch `pixal3d`.
Latest commits: `176c2d09` (surface-bound refined ears, ear texture helpers,
Blender/mobile/web runtime support) and `9294f5ac` (iteration record).
No push was performed. This handoff remains untracked. Do not modify
`resume-vhuman-rig.md` as part of this work.

The latest work focused on ear shape and texture synthesis/completion. Continue
from the refined-ear candidate rather than restarting the reconstruction.

Canonical workspace: `tmp/vhuman-ear-synth/`. Read `session.json` there and the
last section of `server/vhuman/QUALITY_ITERATIONS.md` first. The parent
contact-tone workspace `tmp/vhuman-contact-tone/` remains the baseline for
matched comparisons.

- Selected reconstruction: `reconstruction_final3/`.
- Geometry SHA256: `a2f43542f902df874c14ab45ee6052b7c3d56cc30940476ea7094919b020d3ae`.
- Basecolor SHA256: `0989f254c245ff200285ee55a53397027efa2acf3be38e75caa3b8bed4605540`.
- Native arrays and both motion tracks are byte-identical to the parent; refined
  ears live in `ear_*` fields (surface-bound to the hidden native ear cage).
- Selected native Blender scenes: `final3_scene/head.blend`,
  `final3_stress_scene/head.blend`; motions `final3_default_motion/`,
  `final3_stress_motion/` (rebound metadata only).
- Portable bundles: `final3_default_usd/portable.blend`, `final3_stress_usd/portable.blend`.
- Mobile package `final3_oral_mobile/` (coarse ear LOD + lip-driven mouth occlusion) and browser `final3_oral_player/`
  (drag to turn/tilt, pitch slider, mouth-occlusion toggle). Mouth occlusion research/fit: `tmp/vhuman-mouth-occlusion/`,
  `server/vhuman/MOUTH_RENDERING_RESEARCH.md` (proposal list for teeth/gums/tongue/lips quality).
- Shape `shape_v3/`, texture `texture_final3/`, generated close-ups
  `generated_zoom/` (+ `visual_review.json`), projection `gen_projection3.npz`.
- Rebuild chain: `shape_ears.py` → `build_candidate.py` → `project_generated.py`
  → `synth_ear_texture.py` → `rebuild.sh <shape> <tag> <texture>` (candidate,
  motions, scene assets, Blender scenes); then `verify_ear_scene.py`,
  `audit_ear_motion.py`, USD workers, `skin_ear_assets.py ... coarse` + mobile
  export/preprocess/browser, `build_review.py`.
- Comparison gallery: `review/` (three panels, 13 pairs, bundles, recipes).

Review URL: <http://127.0.0.1:45615/vhuman-ear-synth/review/>.
Browser URL: <http://127.0.0.1:45615/vhuman-ear-synth/final3_oral_player/>.
If the server is gone, serve the repo's `tmp/` directory on localhost port 45615.

## What improved and what was verified

Refined ears (latest): frontal ear-mask coverage 61.5→89.3% (left) and
63.7→94.1% (right), ear length about 68 mm with protrusion/flare, authored
helix/antihelix/crura/fossa/concha/tragus relief, zero ear crossings over 81
default + 43 stress samples, minimum relative triangle area 0.215. Ear texture
resamples eroded photographed ear cores (3,988 texels) and synthesizes the rest
(harmonic tone, pore-scale exemplar detail, two guarded Qwen-Image-Edit-2511
close-ups for 14,428 texels, capped rim/lobe tint, bounded cavity de-shadowing);
other texels are byte-identical. Scene 738/396, USD 1,458/972 and 774/516,
browser 30 FPS with binding p95 3.3e-6 mm all pass. Details and rejected trials
are in the last QUALITY_ITERATIONS section.

Earlier (contact-tone) results, still inherited:

Selected tongue-root cleanup: localized native expression basis 353 at -0.5,
full weight within 1 mm and geodesic feather to 4 mm. Incorporated in rest
geometry and carried through native joint skinning. Across 43 stress samples,
strict self-crossing totals decrease 764 → 258, leaving six pairs per sample.
Maximum shift 0.602 mm; minimum relative oriented tongue area 0.284. Visible
contacts remain 70 → 70 over 215 views, with no tested view worsening.

Selected neck tone: conservative half-strength bounded log-RGB alignment of
unobserved upper-neck skin toward photo-supported cheeks. Changes 332,076 texels,
with zero photographed or ear texel edits; multiplicative gains retain detail.
Ear texture/fitting and previous scalp cleanup are retained. Broad jaw and scalp
patches are still visible. This is a color prior, not measured pigmentation.

Offline eyelid experiment: render subdivision level 2 baked into a dense mesh,
then exterior camera-ray correction with shared orientation-constrained weights.
Changes 208 vertices by at most 0.194 mm. All 41 saved poses and 40 midpoints pass
orientation checks, minimum relative area 0.205. Four critical-pose exterior
crossings decrease 1,556 → 1,173 (24.6%), but internal socket crossings increase
6,038 → 6,439. Visible contacts remain 464 → 464 over 20 views, with none worse.
Fresh Blender verification checks all 41 poses and UVs. Do not claim overall
contact clearance or promote this bake as an accepted native solution.

Both native USD bundles pass export/reimport plus fresh portable import and
22-file source-evidence verification. Default: 1,458 mesh and 972 shader-animation
checks; stress: 774 and 516. Twelve relevant host tests pass. Hardware browser
verification passes on RTX 5060 Ti at the configured desktop 30 FPS cap. Gallery
checks pass links, images, archives, controls and narrow-screen layout.
No physical phone/tablet was connected; the user requested a device test page.
Finite crossing checks do not prove continuous clearance, containment/tangency
freedom, displaced-surface clearance or refractive visibility.

## Remaining items, in priority order

0. **Ear follow-ups.** Ear backs have only synthesized colour (the 135° generated
   view was rejected; try shaded conditioning renders or a different view/seed).
   Reduce the remaining photographed concha/notch shading on the left ear with a
   proper occlusion estimate instead of the luminance floor. The fold relief is
   a generic prior and ear width is tied to length; a side photo or a better
   generated side view could constrain both. Mobile uses a coarse ear LOD; bake a
   normal map from the dense ear if budget allows. `player.cpp` per-vertex
   binding changes are uncompiled here (no Filament SDK).

1. **Coupled exterior eyelid and internal socket clearance.** Address both sides
   of the closed outer/inner lid junction without transferring intersections
   into the socket. Evaluate actual render subdivision and signed/dynamic
   displacement, plus optical shells and eyelashes. Preserve source projection,
   eye shape, blink appearance and face attachments. The dense experiment is a
   reference and review artifact, not the solution to reuse unchanged.
2. **Residual oral contacts.** Remove the six tongue self-crossings per audited
   pose and reduce tongue/floor, cavity and gum contacts through rest and authored
   speech/open-mouth poses. Preserve the exposed tongue and existing no-worse
   visibility result; enforce orientation/edge limits and native skinning.
3. **Broad skin-tone transitions.** Improve the remaining jaw band and scalp/neck
   patches with geometry-aware fitting/baking or material estimation. Protect
   photographed skin and ear texels unless a separately evidenced edit is
   justified. Preserve fine texture, and label unobserved color as a prior.
   Use matched lighting/camera renders; avoid merely blurring detail away.
4. **Bring accepted corrections into live deformation/export.** The native
   browser has the root and neck changes but no dense eyelid correction or
   collision-accepted speech gate. Once a correction passes the quality gates,
   implement a compact runtime-compatible field/control, keeping native motion,
   bindings, source hashes and Blender/USD appearance consistent. Do not export
   the dense non-native skin through the current mobile exporter as a rigid mesh.
5. **Broaden contact and visual validation.** Check the full blink and oral tracks,
   fractional times and side views, with stronger clearance/visibility methods
   where needed. Preserve all observed regressions in reports. Rebuild the
   gallery and portable bundles, fresh-import verify, and test the native browser
   after accepted changes. Test a physical mobile device when one becomes
   available; desktop results are not mobile measurements.
6. **Cleanup and reproducible handoff.** Consolidate successful experiment recipes
   into reusable repo tools where useful, retain evidence/provenance and rejected
   trials, update canonical session pointers and this file. Commit coherent
   changes locally and report the hash; do not push without per-action permission.

Known rejected approaches: deeper projective eye-shell refits clear base meshes
but worsen render-level visible contacts; tiny lid rollbacks and subdivision
crease variants worsen some views; unconstrained camera-ray correction folds
facets; global tongue basis 353 changes worsen exposed contacts; stronger neck
alignment/jaw chroma trials are unselected. Read their audits before repeating.

## Runtime and operating constraints

- RTX 5060 Ti, 16 GB VRAM; GPU use is authorized.
- Blender: `$HOME/local/blender-5.2.2-linux-x64/blender` (5.2.2 LTS).
  Use `--python-exit-code 1`, enable Cycles as required after loading a file,
  and use the CUDA device for rendering. Blender can require elevated execution
  because restricted execution stalls audio/device initialization.
- Python: `tmp/vhuman-texture-venv/bin/python`.
- Never use `/tmp`. Use `/local` if available, otherwise repo `tmp/`;
  set `TMPDIR="$PWD/tmp"`. `/local` was unavailable during this iteration.
- Missing weights/data may be transferred with scp/rsync from `b550:/mnt/disk1`
  or `b550:~/work/gemm/pixal3d/tmp` into `/mnt/nvme02/models`, using
  `/mnt/nvme01/models` if disk space requires. Check space and existing files
  before copying; these locations/network operations may require escalation.
- Pinned GNM: `tmp/vhuman-rig/models/gnm-v3/gnm_head.npz`.
- Browser build uses `EM_CACHE="$PWD/tmp/vhuman-texture-5060ti/emscripten-cache"`
  and Three.js `tmp/vhuman-texture-5060ti/vendor/package` (0.163.0).
- Follow repository AGENTS.md. No sub-agents are authorized by this handoff.
- Preserve the independent implementation provenance: do not distribute/reuse
  legacy generated eye assets from pre-independence `tmp/vhuman` recovery records
  as independent outputs. Keep applicable third-party notices.

Useful verification commands (from repo root):

```bash
TMPDIR="$PWD/tmp" tmp/vhuman-texture-venv/bin/python -m unittest \
  server.vhuman.test_sequence_clearance \
  server.vhuman.test_local_skin_color server.vhuman.test_lid_cleanup
TMPDIR="$PWD/tmp" tmp/vhuman-texture-venv/bin/python \
  -m server.vhuman.mobile.browser_verify \
  --player tmp/vhuman-contact-tone/player --out tmp/vhuman-contact-tone/next-browser-check \
  --hardware --device-test
```

Use a fresh output directory for exporters/builders and preserve prior receipts.
Other exact build/verification commands and saved recipes are in the workspace.

## Copy/paste resuming prompt

> Read AGENTS.md, resume-vhuman.md, recent git log,
> server/vhuman/QUALITY_ITERATIONS.md and tmp/vhuman-contact-tone/session.json.
> Continue improving the preferred vhuman candidate (tmp/vhuman-ear-synth/session.json) on this RTX 5060 Ti 16 GB.
> Prioritize coupled eyelid/socket clearance, residual tongue/oral contacts, and
> broad jaw/scalp/neck tone transitions, and the ear follow-ups. Retain the
> refined surface-bound ears, tongue-root and neck-tone improvements and source identity.
> Use Blender at ~/local/blender-5.2.2-linux-x64 for geometry, shading, fitting
> and baking; write custom Python import/export/connection scripts as useful.
> Prefer portable USD, with NumPy/custom formats allowed for experiments.
> GPU access is allowed; fetch missing weights/data from b550:/mnt/disk1 or
> b550:~/work/gemm/pixal3d/tmp to /mnt/nvme02/models (or /mnt/nvme01/models for
> capacity) as needed. Do not repeat rejected trials without new evidence.
> Validate rendered/subdivided geometry and contact visibility, not just base
> mesh counts. Keep corrections constrained and preserve photographed/ear skin
> and fine texture. Only promote improvements that pass matched quality gates;
> contact acceptance remains false until supported. Prepare device testing but
> do not claim physical mobile validation without a device. Keep experiments,
> evidence, hashes and export/runtime provenance reviewable. Update the gallery,
> canonical session and this handoff, run relevant checks, commit coherent work
> locally and report remaining issues and commit hashes. Never use /tmp or push
> without explicit per-action permission. Continue autonomously within this
> scope; no new time-budgeted goal or sub-agent work is requested here.

## Mouth contact + full-geometry mouth lighting (2026-10-09)
- Lining/teeth gap fixed by the per-pose contact deformer (server/vhuman/reconstruction/{oral_contact,contact_runtime}.py, web/vhuman_mobile_contact.js); median gap 2.86 -> 0.91 mm.
- v1 occlusion reference was wrong (truncated occluders). New: Cycles GI bakes (tmp/vhuman-mouth-gi/bake_gi.py), fit_gi.py + fit_transfer.py write oral_occlusion.json with indirect affine + learned per-light transfer.
- Artifacts: tmp/vhuman-ear-synth/{reconstruction_contact,contact_scene,contact_stress_scene,contact_*_usd,contact_mobile,contact_player}.
- Rebuild order: build_candidate_contact.py -> contact_assets.py -> build_contact_scene.py (Blender) -> bake_gi.py -> fit_transfer.py -> fit_gi.py <mobile_scene> <candidate> -> export/preprocess/browser.
- Open: iOS/Filament lacks oral shader + contact deformer; no phone test; deep lining 2.3% penetration (hidden).
