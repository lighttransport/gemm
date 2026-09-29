# Facial rig for generated heads

Turns a fitted head (Qwen-Image 2.1 portrait → Pixal3D → `head/fit.py`) into
an animatable face: an imported or procedural template fitted to the subject, a
template skeleton (neck, head, jaw, eyes, upper/lower teeth, a 4-joint
tongue), procedural teeth/gums/tongue, skin weights, 52 expression shapes plus
correctives, a linear rig evaluator, and skinned glTF / UsdSkel exports.
The rig code is repository authored. GNM v3 and ICT-FaceKit Light assets are
downloaded separately into the ignored `tmp/vhuman-rig/models/` cache.

```sh
# rig interpreter (numpy, Pillow, scipy, PyTorch; CUDA optional)
uv venv tmp/vhuman-rig-venv --python 3.12
VIRTUAL_ENV=tmp/vhuman-rig-venv uv pip install -r server/vhuman/requirements-rig.txt

sh server/vhuman/run.sh                        # http://127.0.0.1:8790/rig
python3 -m server.vhuman.cli rig --head <id>   # same job without the server
python3 -m server.vhuman.cli rig --head <id> --face-model ict_facekit_light
python3 -m server.vhuman.cli rig --head <id> --face-model procedural
python3 -m server.vhuman.cli rig-soft-tissue --head <id> --take <take-id>
python3 -m server.vhuman.cli rig-train-deformer --head <id> --takes <take-id> <other-id>
python3 -m server.vhuman.cli rig-fit-video --head <id> --video clip.mp4
python3 -m server.vhuman.cli rig-track --head <id> --track capture.txt --out anim.usda
python -m server.vhuman.rig.build <head folder> [--res 2048] [--out DIR]   # in the rig interpreter
sh server/vhuman/rig/external.sh [--local] [--build]   # LightRig + LightUSD (vchar) under third_party/
sh server/vhuman/rig/setup_face_sources.sh             # GNM + ICT + private LightGeom, build facial solver
sh server/vhuman/rig/setup_face_sources.sh --no-build  # check out sources and weights only
sh server/vhuman/rig/setup_face_video.sh --install-deps  # optional Face Landmarker
sh server/vhuman/rig/setup_face_video.sh --with-reference --install-deps  # optional CompSkin check
```

Outputs go to `<head>/rig/`: `rig.glb` (web viewer), `rig.usda` + `textures/`
(and `rig_usd.zip`), `rig.json` (the rig definition), `rig_*.png` maps,
`preview.png`, `rig_report.json` (fit, bake and timing statistics).

## Face topologies and anatomy

`gnm_v3` is the default for new builds. Its Apache-2.0 model provides the
full-head exterior skin and native UV atlas, 253 identity modes and 383
expression modes in the verified v3 weight file. The loader uses the first
170 identity modes for fitting. `ict_facekit_light` uses the MIT-licensed
neutral mesh, 100 identity modes and authored expression OBJ targets. Its
two UV tiles are packed side by side into one atlas. `procedural` retains the
previous rig. All three use the fitted procedural scaffold for the skeleton,
control names, skinning and subject correspondence. Imported skin is exported
as LOD0; the fitted procedural mouth bag supplies the interior, and the
existing procedural LOD1/2 provide lower-detail alternatives. `rig.json` and
`rig_report.json` record the selected source and provenance.

Single-view subjects may not contain enough rear-head texture to cover a full
GNM or ICT atlas. The baker extends measured skin pixels to unobserved UV
texels and reports `inferred_texels` and `far_texels`; these counts indicate
where texture is inferred. GNM's eye measurements set new eye defaults:
14.6 mm sclera radius, 6 mm limbus radius, 8.5 mm corneal curvature radius,
and a neutral pupil half the iris radius. Saved eye parameter sets with the
previous dimensions retain them through the `legacy_v1` profile.

## Offline facial soft tissue

`rig-soft-tissue` takes an existing speech or face-video performance (`animation.json`) and
evaluates the native facial rig, then simulates a cheek, perioral and chin
patch in LightGeom. The runner retriangulates the patch, extrudes a 5 mm
tetrahedral layer, pins the inner surface, and solves a 10 kPa Neo-Hookean
material. It drives the volume with one tenth of the rig motion, then adds
the simulated residual to the full rig pose. The export is bounded to keep
positive tetrahedral volume. It writes `soft_tissue.usda` with time samples
and `soft_tissue_report.json` with volume and tracking diagnostics beside the
take. It also writes `soft_tissue_samples.npz` with correspondence, controls,
and simulated surface samples. Build LightGeom's
`lightphysics_vhuman_face` target with `setup_face_sources.sh` first, or pass
`--lightgeom-runner` to the CLI. The script pins GNM and ICT source revisions,
checks the GNM weight SHA-256, and checks out LightGeom from
`git@github.com:lighttransport/LightGeom.git` into `third_party/LightGeom`.
The private repository requires GitHub SSH access. Its pinned facial solver
commit is `db64640cbbbb44d73c5ee3ffa3c3b405dee2cec1`; the script reports
an error if that commit is not yet available on the remote. Run it from any
directory after installing Git, curl, CMake 3.24+, and a C/C++ compiler.
The rig interpreter still needs `requirements-rig.txt` as shown above.
`rig-train-deformer` distils one or more sample takes into a subject-specific,
second-order 8–16 mode model. It inverse skins the LightGeom residual,
fades the patch seam and contact vertices, fits a damped recurrence from rig
controls, and selects mode count and damping on a held-out take (or a
contiguous tail when only one take exists). The model and LOD bases are
`soft_deformer*.safetensors`; the browser adds their pre-skin offsets after
blendshapes and before skinning and exact contacts. A take's coefficients
are deterministic from its first frame, so seeking does not depend on the
previous playback position. `soft_deformer_report.json` records full-head
and active-patch held-out errors. The original USD remains available for
inspection. Material stiffness and residual gain remain experimental values
and have not been calibrated to measured human tissue.

The optional face-video path accepts MP4, WebM and MOV up to 64 MiB and 30 s.
MediaPipe supplies 2D face landmarks and blendshape priors; our PyTorch
`TorchRig` fits the rig controls with landmark, prior and temporal terms. The
result is a normal `vhuman.performance.v1` take with an optional extracted WAV
and a `fit_report.json`. It works best for near-frontal, single-face video
of the same subject as the rig. This is a 2D fit, so hidden-side motion and
depth are not recovered. The tracker is only an offline dependency.

For a reproducible public smoke test, run:

```sh
TMPDIR=$PWD/tmp/vhuman-rig tmp/vhuman-rig-venv/bin/python -m server.vhuman.rig.fetch_speakingfaces_clip
```

This is a **manual download**. The public source is the
[ISSAI SpeakingFaces subject-1 archive](https://huggingface.co/datasets/issai/Speaking_Faces/resolve/main/image_audio/sub_1_ia.zip)
([dataset and license](https://issai.nu.edu.kz/download-speaking-faces/),
CC BY 4.0). The helper reads only one 72-frame RGB/audio clip by HTTP range
into ignored `tmp/vhuman-rig/speakingfaces/`; setup, builds, tests and the
server never fetch it. No SpeakingFaces media is committed to this repository.
Then run `rig-fit-video` on the generated MP4. Fitting this clip against a
different person's head checks the tracking and optimizer path, not identity
or final visual quality. Subject-specific validation still needs an aligned
rig and independent human ratings.

`reference_compskin.py` compares a sparse PyTorch skinning proxy against
unmodified functions loaded from Meta's pinned Apache-2.0 CompSkin source
(`setup_face_video.sh --with-reference`). This is an offline mathematical
check, not a bundled runtime or an alternative deployed deformer.

The offline-teacher/compact-runtime split follows the practical direction of
[SoftDECA](https://cg.cs.tu-dortmund.de/publications/2023-softdeca.pdf)
(differentiable soft facial mechanics) and
[Neural Volumetric Blendshapes](https://arxiv.org/abs/2212.14784)
(learned non-rigid motion around explicit controls). We keep the 51 facial
controls, blendshapes and LBS as the authored motion, and add a small
subject-specific dynamic residual. [CompSkin](https://github.com/facebookresearch/compskin)
was examined for sparse deformation compression; our PyTorch check on 512
vertices and eight facial shapes matched its transform calculation within
9.4e-10 m. A 12-proxy fit reduced the sampled error from 0.717 to 0.267 mm,
but the deployed model uses the more predictable modal offset on the existing
head topology. No reference source or weights are bundled in the product.

On the existing `291dfa911553` head, two LightGeom speech takes trained an
eight-mode model; held-out patch RMSE fell from 0.0373 to 0.00335 mm. A
different person's 72-frame [ISSAI SpeakingFaces](https://issai.nu.edu.kz/download-speaking-faces/)
clip produced 77 resampled frames with a visible face in every frame; the
mean 2D anchor error fell from 0.00526 to 0.00226 in normalized image units.
These are fitting checks, not perceptual ratings. Device frame-time and
subject-matched visual quality still need direct measurement.

LOD2 now downsizes embedded atlases and retains the eight highest-energy ML
targets on carried eye meshes; the facial skin retains its full ML basis.
This reduces download and GPU upload size for mobile WebGL2. Measure load,
frame time and appearance on target devices before choosing LOD2 as a
shipping default.

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
4. **GPU runtimes**: `cuda/vhuman/` (cuew + NVRTC) and `vulkan/vhuman/`
   (vkew + glslc SPIR-V). The host prepares the rig state (the MLP batched as
   two `sgemm_avx2` calls). One kernel or shader blends morphs
   (register-tiled over 8 frames) and skins, and a second one projects the
   contacts. Both equal the CPU result (`test_rig`). Idle RTX 5060 Ti,
   1024-frame batches, 149 morphs: CUDA 2.2 µs/frame deform, 6.7 µs/frame
   with exact contacts; Vulkan 2.9 and 7.8 µs/frame (submit to completion).
   Host preparation takes 4.5 µs/frame.

Contacts (ground truth, the viewer's heat map and `viz.json`): lid vertices vs
the eyeballs, lip/vestibule vertices vs sphere sets (per-tooth spheres on the
teeth joints; three spheres per tongue cross-section on the blended tongue
joints), and upper vs lower lip pairs on rings 1..-2.

Rig-level contact fixes (before any learning):
- **mouthClose** is driven by the corrective mouthClose × jawOpen: it
  closes an open mouth and does nothing to a closed one.
- **tongueOut** reaches the incisors on its own. It protrudes past the lips
  only through tongueOut × jawOpen.
- **Lip shapes**: every shape (except mouthClose) keeps each upper/lower lip
  pair from closing past its rest gap (zero where the lips touch). The excess
  is split between the lips and faded outwards. For pairs that touch at rest,
  no sum of shapes can cross them. Pairs with a rest gap (the outer ring) can
  still be closed by combinations (e.g. press + roll), which the ML
  correction mostly removes. Linear-rig lip crossings on held-out controls
  dropped from 6060 to 64 vertices.

Learning: 4096 samples (30% tongue scenarios with an open jaw), region-split
PCA (48 mouth + 48 rest components), and an MLP on the rig's full input vector
(controls + corrective products). Tongue-contact samples weigh 4x, with
AdamW. A posed-space lip penalty keeps the predicted correction from crossing
the lips. Reference man, held-out contact vertices, linear vs linear + ML:
eye 17429→1887, teeth 4846→576, tongue 1306→601. For lips, 64→231: the
network still adds a few crossings, about 0.4 vertices per sample and
mostly shallow. For example, tongueOut with the jaw at 0.6 goes from 59
tongue + 19 teeth contacts to 0 + 2, but leaves 48 lip-pair vertices up to
0.5 mm.

### Exact contacts (post-skinning projection)

`contacts.py` defines a projection that runs after skinning in every runtime:
numpy, C (`vh_deformer_eval`, on by default, `vh_deformer_set_contact_iterations`),
CUDA and Vulkan (a second kernel/shader, one workgroup per frame), and the
web viewer ("exact contacts"). Per frame, up to 4 iterations, each ending the
loop early if it moves nothing:
1. lid vertices go outside their eyeball;
2. upper/lower lip pairs separate along the head/jaw up axis;
3. lip/vestibule vertices go outside tooth and tongue spheres (per vertex,
   repeated passes: overlapping spheres push into each other).

The displacement is then smoothed over the contact vertices' mesh graph (2
Jacobi steps) so neighbours follow, and the projection runs once more. The
thresholds are the training contacts' (rest-relative), shipped in the package
(`contact.*`) and `viz.json`. On 200 random poses: 703 eye, 720 sphere and 65
lip penetrations → 0, 0, 0. C equals numpy within 1e-5 m (float32), and
CUDA/Vulkan equal C within 2e-8 m.

The viewer evaluates the ~1.1k contact vertices on the CPU (three.js
`getVertexPosition`), projects them, maps each correction back before skinning
(inverse blended rotation) and feeds it through a `contactOffset` attribute
that the materials add right after the morph targets (~15 ms per update in JS).

## Subject expressions, wrinkle maps, LODs

**Expression portraits** (`exprdata.py`, job kind `expressions`): Qwen-Image
2.1 edits the neutral portrait into 12 expressions (smile, brows up/down,
sneer, squint, frown, pucker, funnel, stretch, press, cheek puff, jaw open;
`preset` `low8` by default) under `<head>/rig/expressions/` with a
`manifest.json`. Run it before the rig job:

```sh
curl -XPOST localhost:8790/v1/jobs -d '{"kind":"expressions","head_id":"<id>"}'
```

When `expressions/manifest.json` exists, the rig build fits **data-driven
shapes**: dense optical flow (OpenCV DIS, rigid drift removed with a
similarity fit) between the neutral and expression portraits is the 2D
target; the procedural shape of the expression's controls is corrected by a
pre-skinning delta (PyTorch, through the portrait camera) with a Laplacian
smoothness term, a weak view-axis term and the lip margins excluded. The
correction is split over the expression's primary controls by their
procedural magnitude and side. Folds are repaired after the fit: around skin
triangles that flip or collapse relative to the procedural pose, the
correction fades out (smooth falloff) until none remain. Joint-driven
expressions (jaw open) are first matched in intensity; the image usually
shows a partial opening (≈0.25 of `jawOpen`), and flow across the opening
mouth is unreliable, so the correction is applied only at intensity ≥ 0.75.
`rig_report.json` → `expressions` has, per expression, the reprojection
error before/after (typically 2-7 px → < 1 px), the intensity, whether it
was applied, and the folds before/after repair.

**Wrinkle maps** (`wrinkles.py`): each expression portrait is warped back onto
the neutral one along a smoothed flow; the band-passed log-luminance change
(capped against the fine-flow warp, so moving features do not leave edges)
is read as a height field, and its slope is baked into the skin atlas in
tangent space. Groups `brow_up`, `brow_down`, `smile`, `mouth` → `wm_<group>.png`
(RG = 0.5 + 0.5 slope, flat 128). `rig.json` → `wrinkles` lists the maps
with driver controls; the viewer adds them to the normal map per face side
("wrinkles" toggle).

**LODs** (`lod.py`): LOD1 (≈3.7k) and LOD2 (≈1.4k vertices, vs ≈13k) templates are built
from coarser layouts (ring subsets, divisor sample counts, wider spacing);
ring vertices map exactly onto LOD0 vertices, free vertices by barycentrics
in LOD0's chart, so skin weights, shapes, ML targets and contact sets
transfer and every LOD shares `rig.json` and the texture atlas. Outputs
`rig_lod{1,2}.glb/.usda`, `rig_deformer_lod{1,2}.safetensors`,
`viz_lod{1,2}.json`; the viewer's LOD selector switches meshes (`--lods` in
`build.py`; head only, no body).

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

## Speech takes

`rig-speech` joins the existing Japanese Qwen3-TTS + `ja_align` pipeline to
the facial rig. Build the CPU runner with `make -C speech build/tts_ja` (or
`make -C speech cuda` for CUDA), then use a fitted, rigged head:

```sh
python3 -m server.vhuman.cli --work tmp/vhuman-independent rig-speech \
  --head <id> --text 'こんにちは。' --speaker Ono_Anna --backend auto
python3 -m server.vhuman.cli --work tmp/vhuman-independent rig-speech \
  --head <id> --wav input.wav --backend cpu
```

For a known recording, `--transcript '日本語の文'` stores its reference text in
the take. `--kana 'にほんごのぶん'` additionally forces phoneme alignment to a
known kana reading; it is passed to the WAV aligner, not inferred from kanji.

For a Base model, use `--tts-model <Base dir> --ref-wav reference.wav`
and optionally `--ref-text` or `--xvec-only`. `--source-take <take id>`
rebuilds animation from a completed take's audio and alignment, without
repeating TTS or CTC. `--emotions @keys.json` accepts an array such as
`[{"t":0,"weights":{"joy":1}},{"t":1.5,"weights":{"sadness":1}}]`;
keyframe times must increase and stay within the audio. Strength defaults
are 1.0 for speech, 0.6 for emotion and 1.0 for secondary motion
(`--secondary-strength`, 0 disables it); the take seed defaults to 7. A TTS
result with no aligned phonemes fails instead of saving a silent animation.
The web `/rig` panel creates and
rebuilds takes with the same job queue. The server accepts text and existing
take IDs; filesystem WAV paths and reference voice files are CLI-only.

The converter samples `ja_align.v1`'s 15 visemes at its 30 fps frame times,
uses the rig's vowel poses, and applies a three-frame symmetric filter. The
aligner's RMS varies vowel aperture, while voiced F0 adds a small pitch-accent
brow lift and head pitch. Phone intervals distinguish bilabial closures
(`m/p/b`) from Japanese `n/N`, which use tongue contact without lip closure.
At 30 fps, a very short bilabial is snapped to its nearest frame so the lip
closure does not disappear between samples. Jaw opening and lip press fade to
zero at peak closure; a rendered Japanese /m/ frame showed that even small
residual values exposed a thin teeth strip on this rig.
Manual emotion drives the upper face and restrained mouth-corner motion.
Repeatable blinks, gaze shifts and subtle head follow-through are baked into
the timeline from the take seed; rebuilding a take reuses its seed unless
overridden. Every take ends on a neutral frame. Playback samples these same
controls against the WAV element's current time, including after seeking;
the viewer's random blink and pointer gaze are used only for manual posing.
Server `/health` reports speech runner/model availability.

Each `<head>/rig/takes/<id>/` contains `audio.wav`, `align.json`,
`animation.json` (`vhuman.performance.v1`: fps, duration, control names, and
`[{t,v}]` frames), `lightrig.txt`, `animation.usda` (a sublayer of the head's
`rig.usda`), and `manifest.json`. The take listing and guarded file URLs are
under `/v1/heads/<id>/rig/takes`. Manifests record the rig hash; the viewer
warns when a rebuilt rig makes an older animation stale.
`lightrig.txt` contains the standard 51 face controls, including baked blinks
and gaze; its format cannot carry the extra tongue and signed head controls.
`animation.json` and `animation.usda` carry the complete motion.

For a Japanese natural-speech review set and ReazonSpeech source constraints,
see [SPEECH_EVAL.md](SPEECH_EVAL.md).

### Optional audio emotion suggestions

`--auto-emotion` runs Alibaba Group's
[SenseVoiceSmall GGUF](https://huggingface.co/FunAudioLLM/SenseVoiceSmall-GGUF)
on the finished WAV in overlapping four-second windows. It supports Japanese
audio and returns coarse categorical tags (joy, sadness, anger, disgust,
surprise, fear, neutral). The converter maps these to editable emotion keys;
the viewer has an opt-in checkbox. A rebuild from an existing take can infer
new keys without rerunning TTS or alignment. `emotion.json` records the
window labels, and the manifest names the model. Silent windows and unknown
tags become neutral. These tags have no calibrated confidence or precise
onset; published SenseVoice emotion benchmarks use Chinese and English, so
Japanese extraction needs human review. Manual keys remain available.

Other candidates considered: [emotion2vec+](https://huggingface.co/emotion2vec/emotion2vec_plus_base)
offers nine classes and frame features, but its published model card does not
report Japanese-specific accuracy; the
[Kushinada JTES SER model](https://huggingface.co/imprt/kushinada-hubert-base-jtes-er)
is trained for Japanese emotion recognition but requires gated access and an
S3PRL checkpoint workflow. SenseVoiceSmall offers a non-gated, model-owner
GGUF release and a dependency-free local CPU runtime.

The optional CPU setup uses the project's published AVX2 runtime and Q8
weights. Keep downloads under ignored `tmp/` (repository guideline):

```sh
mkdir -p tmp/vhuman-emotion/runtime
curl -fL -o tmp/vhuman-emotion/funasr-llamacpp-linux-x64-avx2.tar.gz \
  https://github.com/modelscope/FunASR/releases/download/runtime-llamacpp-v0.2.6/funasr-llamacpp-linux-x64-avx2.tar.gz
echo 'aaebc5470f846ce915200b35d6e9f9bd0a0d3ed399d39e49bdeb7a1f1782bc70  tmp/vhuman-emotion/funasr-llamacpp-linux-x64-avx2.tar.gz' | sha256sum -c -
tar -xzf tmp/vhuman-emotion/funasr-llamacpp-linux-x64-avx2.tar.gz -C tmp/vhuman-emotion/runtime
curl -fL -o tmp/vhuman-emotion/sensevoice-small-q8.gguf \
  https://huggingface.co/FunAudioLLM/SenseVoiceSmall-GGUF/resolve/90c1c61912018b70ada0fcc024ea24aca62f2e63/sensevoice-small-q8.gguf
echo '4ae45c94422de949b387e2e0fb10d7e14e4c42c69db30c3444ecc7d4b844b7c5  tmp/vhuman-emotion/sensevoice-small-q8.gguf' | sha256sum -c -
python3 -m server.vhuman.cli --work tmp/vhuman-independent rig-speech \
  --head <id> --source-take <take-id> --auto-emotion
```

Use `--emotion-runner` and `--emotion-model` (same names on the server) for
other local locations. The [GGUF model card](https://huggingface.co/FunAudioLLM/SenseVoiceSmall-GGUF)
lists Apache-2.0, while the [source model](https://huggingface.co/FunAudioLLM/SenseVoiceSmall)
uses the [FunASR Model Open Source License](https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE),
which requires source/author attribution and retention of the model name.
Weights and the third-party runtime are downloaded separately and are never
bundled into this repository. The runtime accepts 16 kHz mono PCM16; the
adapter converts the rig WAV before inference.

`speech.build_frames` also has a timestamped emotion-provider callback for
other future integrations. The NVIDIA Audio2Emotion-v2.2 model remains outside
this project because its [license](https://huggingface.co/nvidia/Audio2Emotion-v2.2/blob/main/LICENSE)
restricts it to use in connection with NVIDIA Audio2Face.

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
  Pixal3D reconstruction, Qwen-Image expression edits); they stay in the
  local work directory.

## Limitations

- Expression shapes are procedural fields scaled to the subject; with
  expression portraits, 11 imaged expressions are corrected from 2D flow
  of generated (not captured) images (jaw open only when the image opens it
  fully): depth along the view axis, the lip margins and the other controls
  stay procedural.
- Wrinkle maps are shading-derived normal detail from one view (front-facing
  skin only); nostril rims and lids may show small artefacts.
- Ears and nostrils are smoothed by the template fit (their detail survives
  in the baked normal map only). The template has no separate eyelashes.
- The mouth interior, teeth and tongue are generic; nothing is inferred from
  the (closed-mouth) portrait.
- Mouth detection falls back to proportions when the lips are not found
  (reported as `features.mouth.fallback`).
- vchar shows joint-driven controls only through animations (the ROM).
- The ML deformer corrects sub-millimetre contacts and soft-tissue
  shearing; it cannot add expression detail the procedural shapes lack.
  The ML deformer alone halves tongue contacts on unseen controls and adds a
  few shallow lip crossings; the post-skinning projection makes all modelled
  contacts exact. Contacts are only the modelled ones (eyeballs, tooth and
  tongue spheres, lip pairs); sphere proxies are coarser than the teeth and
  tongue meshes. USD/vchar playback has no projection (it evaluates UsdSkel). The native runtime covers the welded head; teeth, tongue and eyes are
  rigid or simply skinned in the exports.
