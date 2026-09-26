# Qwen-Image 2.1 as an Image-to-3D preprocessing backend

`cuda/qimg21/runner.py` and the `qimg21_i23d` package turn the Qwen-Image 2.1
runners (native CUDA and the PyTorch reference) into a 2D / multi-view
preprocessing stage for image-to-3D pipelines such as Pixal3D, SAM-3D and
TRELLIS.2. It covers:
- object extraction to RGBA;
- object-preserving edits, mask edits and occlusion completion;
- texture cleanup;
- requested-view, turntable and elevation-ring generation;
- dataset export with metadata, a NeRF/Blender `transforms.json` and 2D
  validation;
- reconstruction to a textured GLB with Pixal3D: the repository's native
  runner (CUDA, ROCm or CPU) and/or the pinned PyTorch reference (see
  [Reconstruction with Pixal3D](#reconstruction-with-pixal3d)).

The package itself trains or runs no 3D network; Pixal3D does the
reconstruction. Every generated view is a 2D image that the model *was
asked* to render from some camera; nothing makes the views agree with each
other geometrically (see [Limitations](#limitations)).

## Quick start

```sh
PY=tmp/qimg21-ref-venv/bin/python
# One photo -> extraction -> 8 views at 0 and 20 degrees + top/bottom -> dataset
$PY cuda/qimg21/runner.py image-to-3d \
  --input ref/pixal3d/moge-upstream/example_images/04_BunnyCake.jpg \
  --output tmp/cake_dataset --views 8 --elevations 0,20 --top-bottom --steps 16 --seed 7
```

This writes:

```
tmp/cake_dataset/
  preprocess/object_rgba.png   extracted, centered, scaled RGBA object (the reference)
  reference/ref_00.png         the reference as used for generation
  views/e000/a000.png ...      one RGBA PNG per requested view (elevation-major)
  views/e020/a045.png ...
  views/e090/a000.png          --top-bottom
  views/e-90/a000.png
  metadata.json                model, backend, commit, seeds, prompts, specs, timings
  transforms.json              Pixal3D-compatible cameras (requested, not calibrated)
  validation.json              per-image 2D checks and dataset summary
  pipeline.json                preprocessing record and total time
```

Every command prints a JSON summary on stdout, has `--help`, and exits with
status 2 and a one-line message on invalid input (bad masks, view specs,
reference counts).

## Commands

| command | what it does |
|---|---|
| `object` | text → one framed RGBA object: Qwen-Image 2.1 draws it alone on a native transparent background (1024² by default); framed like `object-preprocess`, with the unframed image kept as `<output>_raw.png`. |
| `object-preprocess` (alias `image-to-3d-preprocess`) | photo → RGBA object: `--method qwen` (default; the model's native transparent extraction) or `rmbg` (RMBG-2.0 matting, alpha only) or `alpha` (use the input's alpha). `--pixels original` (default) keeps the source RGB and takes only the alpha, so detail and geometry come from the photo. Then center/scale (`--fill 0.85`, `--pad`, `--crop`, `--size`, `--keep-scale`, `--no-center`). |
| `edit` | object-preserving image-to-image edit (`--instruction`). `--strength 1` edits through image conditioning; below 1 it is SDEdit from the input. `--mask`/`--mask-rect x,y,w,h`/`--mask-circle cx,cy,r` limit the change (see [Masks](#masks)). |
| `texture-preprocess` | `--ops neutralize-lighting,reduce-shadows,reduce-specular,remove-reflections,repair-defects,remove-background` with an instruction that forbids beautifying and keeps text/logos. |
| `multiview` | views of a reference: `--views N [--elevation E]`, or rings `--azimuth-views N --elevations -20,0,20`, `--top-bottom`, extra `--reference` photos (torch backend). |
| `turntable` | N evenly spaced azimuths at one elevation, flat `images/NNN.png` layout. |
| `image-to-3d` | the whole chain: extract → normalize → optional `--cleanup` → optional `--edit` (with `--mask*`) → views → validate → export → optional `--reconstruct native\|reference\|both`. `--keep-background` skips extraction for an already clean RGBA. |
| `reconstruct` | a view dataset (multiview) or an RGBA object (single view) → `<runner>.glb` and `reconstruction.json` with Pixal3D. |
| `turnaround` | a character's front / left / back (/ right) views from **one turnaround sheet** (`--prompt` and/or `--input` reference) → posed dataset; `--reconstruct native` builds the GLB from those views (multiview). See [Characters](#characters-turnaround-sheets). |
| `validate DIR` | re-run the 2D validation of a dataset and rewrite `validation.json`. |

Long prompts:
- Every text argument (`--prompt`, `--instruction`, `--edit`,
  `--negative-prompt`) accepts `@FILE` to read the text from a file.
- The native text encoder takes up to **4096 tokens**, including the chat
  template and, with an image reference, its image tokens (1024 at the
  default condition resolution). About 3,000 English words fit without a
  reference.
- The backend counts a prompt's tokens with the native tokenizer (~0.15 s,
  cached). A prompt that doesn't fit is refused before any work, with its
  token count and the options: shorten it, lower `--condition-resolution`,
  or use `--backend torch`, which has no limit (the 2.1 pipeline doesn't
  truncate).
- The resident denoiser is sized for the prompt: 512 tokens, then 1024,
  2048 or 4096. Previously every prompt over 512 tokens fell back to a
  one-shot run: a 669-word prompt took 25 s per image instead of ~7 s.

Shared options: `--backend auto|native|torch|mock`, `--steps` (20),
`--seed` (0), `--seed-mode shared|per_view`, `--width/--height` (512),
`--background transparent|white|gray`, `--instruction`, `--template-file`,
`--projection perspective|orthographic`, `--fov` (20°), `--distance` (3.119),
`--batch-size`, `--condition-resolution` (1024), `--no-resident`, `--preset`,
`--attention sage|flash|exact`.

## Python API

```python
import sys; sys.path.insert(0, "cuda/qimg21")
from qimg21_i23d import ops, views
from qimg21_i23d.backends import select_backend

backend = select_backend("auto", references=1)          # loaded once, reused for every call
try:
    ops.preprocess_object("photo.jpg", "obj.png", backend, method="qwen", fill=0.85)
    specs = views.rings(8, [-20, 0, 20], width=512, height=512)
    ops.generate_multiview(["obj.png"], specs, "dataset", backend,
                           ops.ViewParams(steps=16, seed=7), layout="rings")
    # or one call for the whole chain:
    ops.generate_image_to_3d_dataset("photo.jpg", "dataset2", backend, azimuth_views=8, elevations=(0, 20))
finally:
    backend.close()                                      # stops resident processes
```

- `ops.preprocess_object`, `edit_object`, `complete_occlusion`,
  `texture_preprocess`: single-image operations.
- `ops.generate_view(refs, spec, out, backend, params)`: one view.
- `ops.generate_views(refs, specs, outs, backend, params, batch_size)`: a
  generator yielding one record per view as it is written.
- `ops.generate_multiview`, `generate_turntable`,
  `generate_image_to_3d_dataset`: datasets.

`views.ViewSpec(azimuth_deg, elevation_deg, roll_deg, distance, fov_deg,
width, height, projection, tags)` is the requested camera. `ring`, `rings`,
`turntable` and `top_bottom` build lists of specs; `view_prompt` turns a spec
into text, and `transform_matrix` into a camera-to-world matrix.

## Backends

| | native (`NativeBackend`) | torch (`TorchBackend`) | mock |
|---|---|---|---|
| references | 1 | up to 10 | any |
| runs | `native_generate.py` → fast CUDA runner (`--preset fast12` INT8 by default) | `QwenImage21Pipeline` in-process, weights placed once (resident blocks + streamed rest) | deterministic synthetic RGBA |
| SDEdit / masks | yes (fast runner restart + mask blend kernel) | yes (truncated sigmas + step callback) | yes |
| mask shown as a reference | no (one condition image) | yes | – |

`--backend auto` picks native for zero or one reference when the CUDA runner
is built, and torch otherwise. Both backends implement the same semantics:
- SDEdit restarts the flow at step `K = round((1 - strength) * steps)` from
  `(1 - sigma_K) * S + sigma_K * E`. Here `S` is the init image
  VAE-encoded at the output size, and `E` is the seed's text-to-image noise.
- Masks re-impose `(1 - sigma) * S + sigma * E` outside the mask after
  every step. `tests: TorchBackendPlanTest` checks that the torch schedule
  equals the native `flow_sigmas`.

## Views, prompts and seeds

- **Prompts.** Each view gets a prompt from a template. The template states
  the requested angles in words, e.g. "front-left three-quarter view",
  "seen from slightly above", "rear view", "top view". It also spells out
  the geometry the camera implies, for example "the object's front points
  to the left edge of the image" or "the camera is 20 degrees above the
  object … its top surface is visible". It carries the identity-preservation
  clauses and the background clause ("The image has alpha channel and the
  background is transparent." by default). Use `--template-file` to replace
  the template: its fields are `{view_words}`, `{elevation_words}`,
  `{facing}`, `{height}`, `{azimuth}`, `{elevation}`, `{roll_clause}`,
  `{projection_clause}` and `{background_clause}`. Use `--instruction` to
  append text.

  The geometry sentences came out of a prompt study on the bunny cake, which
  compared three styles on the hard views:
  - The template with angles and view names alone left 90°/270° sides
    frontal, barely changed at 20° elevation, and returned a front view for
    the bottom.
  - Adding the geometry sentences gave real profiles, a visible top surface
    at 20°, and a true underside.
  - A short "rotate the camera" instruction ignored the angle except for the
    bottom.
- **Seeds.** `--seed-mode shared` (the default) gives every view the same
  seed, and so the same initial noise, which helps consistency.
  `per_view` derives a stable seed per camera from
  `sha256(seed, azimuth, elevation, roll)`. Either way, a view's output
  does not depend on:
  - which other views are requested,
  - their order,
  - `--batch-size`.

  Generated views are never fed back as references.
- **Layouts.**
  - `turntable`: `images/NNN.png`.
  - Rings: `views/e{elevation:03d}/a{azimuth:03d}.png`, e.g.
    `views/e020/a045.png`, `views/e-20/a030.png`.

## Masks

`--mask FILE` takes a white region = may change. RGBA and LA masks use
their alpha. `--mask-rect` and `--mask-circle` take source-pixel
coordinates, and `--mask-feather` blurs the edge. A mask is enforced in two
ways:
1. **Latent blending** in the denoise loop. The mask is taken to the
   16×16-pixel latent grid and dilated by one token. Outside it, the latent
   is reset to the renoised source after every step.
2. **Pixel paste-back** after decoding. Outside the (feathered) mask, the
   output is the input, bit for bit.

The torch backend also shows the mask to the model as an extra reference
image, as the model card describes (`--no-mask-reference` turns that off).
Native takes one condition image and cannot.

Empty masks, masks of the wrong size (without resize), and rects/circles
that miss the image are rejected with a message.

## metadata.json and transforms.json

`metadata.json` records:
- the model, backend, runner git commit, seed and seed mode;
- all generation parameters;
- each reference's path and sha256;
- the preprocessing transform (scale and offset from source pixels);
- per view: its file, the full ViewSpec, prompt, seed, seconds and the
  backend's timing breakdown.

Its top-level `camera_parameters` note says the cameras are **requested view
metadata, not calibration**.

`transforms.json` follows Pixal3D's NeRF/Blender convention:
- The world is Z-up.
- Camera-to-world columns are (right, up, back).
- Azimuth 0 puts the camera at −Y, and positive azimuth moves it towards
  +X.
- `camera_angle_x` is the requested FOV (default 20°), and the distance
  defaults to 3.119.

The reference is frame 0 (`"generated": false`, and it defines the output
orientation in Pixal3D) unless `--no-reference-frame` is given. The file
carries `"generated_views": true` and the same `camera_parameters` note. Its
matrices come from the requested ViewSpecs; nothing measured them.
`ref/pixal3d/validate_multiview.py --views-dir DATASET` passes on a
generated dataset (projection max error 3.6e-7). A 4-frame subset of the
bunny-cake dataset (the reference plus the 90°, 180° and 270° views) ran
through `cpu/pixal3d/pixal3d --views-dir` to a GLB: 781,867 vertices,
295 s, 3.9 GB device peak. This shows format compatibility only, not
reconstruction quality. Pixal3D reads at most 16
frames: select a subset with `--num-views N` (the first N frames) or
generate ≤ 15 views.

## validation.json

These are per-image checks only:
- the file exists and is a valid PNG;
- size and mode are consistent across the set;
- alpha is present when a transparent background was requested;
- the foreground fraction is within [0.01, 0.98];
- the bbox is not clipped at the edge (2 px);
- the image is not near-constant.

It also has a dataset summary (foreground fraction min/mean/max). The
checks are **2D only**: a set can pass while its views disagree in 3D.

## Reconstruction with Pixal3D

```sh
# Photo -> object -> (edit) -> 4 views -> native Pixal3D multiview -> GLB
$PY cuda/qimg21/runner.py image-to-3d --input photo.jpg --output tmp/obj \
  --views 4 --steps 16 --reconstruct native            # [--edit "make the base blue" --mask-rect 0,300,512,212]
# An existing dataset, both runners, meshes compared
$PY cuda/qimg21/runner.py reconstruct --input tmp/obj --output tmp/obj/recon-both --reconstruct both
# One RGBA object, single view (FOV from MoGe-2 unless --recon-fov is given)
$PY cuda/qimg21/runner.py reconstruct --input obj.png --output tmp/obj1 --recon-mode single
```

`qimg21_i23d.reconstruct` drives the same runners, with the same flags, as
the Pixal3D demo server:

| runner | device | command |
|---|---|---|
| `native` | `--pixal3d-backend cuda\|rocm\|cpu` | `cpu/pixal3d/pixal3d --backend … --input … --fov --distance --mesh-scale` (single) or `--views-dir` (multiview), resident GPU execution, mixed flow precision |
| `reference` | cuda (`ref/pixal3d/.venv-reference-cuda310`) or rocm (`.venv-rocm`) | `ref/pixal3d/run_reference_sv.py --image … --fov` or `run_reference_mv.py --views_dir …`, `--low_vram --resolution 1024` |

Multiview input:
- The dataset's reference (frame 0, which fixes the output orientation)
  goes to Pixal3D together with the generated views of the
  `--recon-elevations` rings (default 0).
- A generated copy of the reference view is dropped, and at most
  `--recon-max-frames` frames (16, Pixal3D's limit) are used.
- The frames are staged as `reconstruction/views/viewNN.png` with a flat
  `transforms.json`. The cameras are the requested ones: FOV 20°, distance
  3.119, and the same convention as Pixal3D's own posed example.
- Frames of another tool's `transforms.json` (no angle fields) are all
  kept.

Single-view input:
- The prepared object alone.
- Camera FOV from `--recon-fov`, or estimated by MoGe-2 through
  `ref/pixal3d/prepare_input.py`, as the server's auto-camera does.
- Distance is `0.5 / tan(FOV/2)`.

Running it:
- The image model's resident processes are stopped before Pixal3D runs, so
  it has the device to itself.
- `--reconstruct both` also runs `ref/pixal3d/compare_outputs.py`
  (symmetric Chamfer between the two meshes).
- `--quality preview|standard|high` picks the Pixal3D output settings:

  | quality | texture | triangles | flow | bunny multiview |
  |---|---|---|---|---|
  | `preview` | 1024 | 300k | BF16 | ~175 s |
  | `standard` (default) | 2048 | 1M | mixed | 233 s |
  | `high` | 4096 | 1M | mixed | — |

  On the bunny, the reduced settings changed the shape by Chamfer RMS
  0.0031 (300k triangles) and 0.0037 (BF16 flow) against standard, for an
  object about 0.9 units tall. Most of Pixal3D's time is diffusion
  (~90–120 s) plus CPU post-processing (~50–80 s: simplification, BVH
  builds and UV unwrapping, which the triangle count drives). BF16 flow
  saved only ~5 s of diffusion.
- `--texture-size`, `--triangle-target`, `--flow-precision`,
  `--gpu-kernels`, `--recon-seed` and `--vram-budget-mib` override the
  preset.
- A runner whose binary, environment or weights are missing is reported
  before any image work starts.

### Single view vs generated views: measured

Pixal3D's posed example (`ref/pixal3d/upstream/assets/mv_images/example`,
a cyclops head) has real renders at 0/90/180/270°, which give an oracle.
All runs used the native runner on CUDA at texture 1024, 1M triangles and
seed 42:
- **oracle:** its 4 real views;
- **single:** our prepared 0° object alone, FOV 20° (its real camera);
- **generated multiview:** the same object plus our generated 90/180/270°
  views posed with their requested cameras.

| input | Chamfer RMS vs oracle | mean distance | normal agreement (abs cos) | Pixal3D time |
|---|---|---|---|---|
| single view | **0.0235** | 0.017 | 0.839 | 392 s |
| 0° + generated 90/180/270° | 0.0632 | 0.045 | 0.762 | 381 s |

The generated views make the mesh worse. Their outlines keep roughly the
reference's width:
- side views were 0.67–0.68 of the frame wide where the real renders are
  0.92 (the head is deeper than it is wide), while heights matched;
- posed as real cameras, they carve the depth to 0.72 instead of 0.99, and
  the profile comes out flattened;
- rewording the scale clause ("same camera distance", "reveal the full
  depth") moved the side width only from 0.67 to 0.71.

So `image-to-3d --reconstruct …` reconstructs from the prepared (optionally
edited or cleaned) object in single view by default. `--recon-mode
multiview` stays available for posed real photos, or for experiments once
view generation keeps real proportions.

### Characters: turnaround sheets

Views generated one at a time are separate samples, and they drift. The
"back" of a character may still show blushing cheeks, and proportions
change between views. A **turnaround sheet** is one sample, so it is
consistent: Qwen-Image 2.1 draws front, left profile, back and right
profile side by side in one transparent `4·512 × 512` image, and
`imageops.split_sheet` cuts it into posed views.
- The cut follows the empty columns between the figures.
- All views share one scale and one ground line, so a profile keeps its
  true width relative to the front.
- Each view is centred horizontally.

```sh
$PY cuda/qimg21/runner.py turnaround --output tmp/bunny --views 4 --steps 20 --seed 12 \
  --prompt "a cute chibi bunny character: big head, long upright ears, round fluffy body, short arms and \
legs, soft white fur with pastel pink inner ears, a small pink scarf, big friendly eyes, 3D toy figure style, \
standing upright in a neutral pose with arms slightly away from the body" --reconstruct native --texture-size 2048
```

On that bunny, the sheet took 25–48 s per seed and gave four views of the
same character:
- the profiles face the correct directions (left profile = 90°);
- the tail shows on the back and on the sides;
- the four views are 433–435 px tall.

A separate-view turntable of the same prompt was visibly less consistent
(a face-like back, a thinner body).

Native Pixal3D multiview from the 4 sheet views took 233 s (654k
vertices). The mesh is clean: the face, a scarf that wraps around, the
tail, and ears and profile that match the sheet. Single view from the front
panel alone took 242 s and had stray floating planes around the model and a
vaguer back.

So the conclusion depends on how the views were made:
- generated **one at a time**, views should not be posed (single view is
  better; see the head above);
- a **turnaround sheet's** views should, since they are consistent and
  share one scale.

`turnaround` therefore reconstructs in multiview by default.

**From a picture of an existing character** (`--input`), with Pixal3D's
jester example `21_img.png` as the only input:

```sh
$PY cuda/qimg21/runner.py turnaround --input ref/pixal3d/upstream/assets/images/21_img.png \
  --output tmp/jester --views 4 --steps 20 --seed 3
```

The sheet (31–44 s) kept the identity: the glowing face, the red/blue hat
with bells, the lantern and the orb. With seed 3 it was also geometrically
right:
- the lantern in the character's right hand is on the image left from the
  front, on the image right from the back, and appears in the right
  profile;
- the orb appears in the left profile.

Seed 12 dropped the props from the side and back views, so try a couple of
seeds.

**Registration.** At first, multiview reconstruction lost to single view
here:
- single view from the front panel (225 s) gave the reference's colors and a
  plausible back;
- multiview (234 s) turned the hat magenta and had stray quads.

The cause was registration. Each panel was centred on its own bounding box,
and for a character with big one-sided parts that isn't the turning axis.
The back view (which hides the lantern) sat 19 px off, so the four textures
blended off-target (red over blue gives magenta).

Opposite views are mirror images of each other in an orthographic
turnaround. `split_sheet(register=…)` therefore shifts the back until its
mirror best overlaps the front, and the right profile likewise against the
left (`imageops.mirror_shift`, within ±15% of the width, and only when the
overlap clearly improves). The shifts and IoUs are recorded in
`metadata.json` as `registration`:

| character | back vs front | right vs left |
|---|---|---|
| jester | shifted 19 px, mirror IoU 0.459 → 0.526 | shifted 7 px, 0.703 → 0.768 |
| bunny | 0 px (0.935) | 0 px (0.899) |

Re-reconstructed with registration, the jester's multiview model (254 s)
lost the magenta and the stray quads. The hat is red/blue striped, the
lantern keeps its glow, and the back is real; it's now on par with single
view. `turnaround` reconstructs in multiview by default.

`turnaround --sheet SHEET.png` splits an existing sheet instead of
generating one. That can be a sheet kept from an earlier run, or a
hand-drawn turnaround in the same view order. Generating a
sheet wider than 1024 px also needed a driver fix: the default VAE decode
tile is now clamped to the short side.

### Native runner vs PyTorch reference

`reconstruct --recon-mode single --recon-fov 20 --reconstruct both` on the
same prepared head: 844 s wall.

| runner | time | vertices / triangles | Chamfer RMS vs oracle |
|---|---|---|---|
| native (CUDA) | 398 s | 635,911 / 943,526 | 0.0235 |
| PyTorch reference (CUDA, `--low_vram`) | 441 s | 663,441 / 970,348 | 0.0326 |

- The two meshes differ from each other by Chamfer RMS 0.0253 (normal
  agreement 0.863).
- The native mesh repeated exactly across runs (same vertex and triangle
  counts).
- The oracle itself came from the native runner.

### Edit, then reconstruct

```sh
$PY cuda/qimg21/runner.py image-to-3d \
  --input ref/pixal3d/upstream/assets/mv_images/example/view00_azim000.png --keep-background \
  --output tmp/i23d-head-edit --views 0 --steps 16 --seed 7 \
  --edit "make the skin green and the nose ring silver" --reconstruct native --texture-size 1024
```

The run took 421 s wall:
- the edit took 17.8 s and kept the shape;
- MoGe-2 estimated a 23.4° FOV (the true value is 20°);
- native Pixal3D took 371 s and produced a green head with a silver ring.

`preprocess/object_edited.png` is the image that was reconstructed.
`pipeline.json` records the edit, the camera and the mesh.

ROCm (`--pixal3d-backend rocm` for either runner) is wired the way the demo
server runs it. It is covered by command tests only: this machine has no
AMD GPU.

## Performance and memory

Measurements are at 512², 16 steps, `fast12` (INT8 weights + SageAttention),
on an RTX 5060 Ti 16 GB, with a 1024-px condition image of the bunny cake.

| | per view |
|---|---|
| before (one-shot processes, cached condition, flat 2 s device wait) | 28.2 s |
| resident denoiser + VAE decoder, prompt cached | **6.9 s** (denoise 5.7 s, decode 0.6 s) |
| resident, new prompt encoded alone (every view of a new object, before batching) | 12.3–13.7 s (text encoder 3.5–4.7 s) |
| same, `--condition-resolution 512` | 10.2–10.7 s (denoise 3.6 s) |
| **new object, encode-only prepare + batched, prefix-shared prompts** (views 2–8 of 8) | **5.7–5.8 s** |

For a whole new-object turntable (8 views, empty caches), the total went
from about 115 s to **64.1 s**:
- prepare, 10.9 s: condition encodes plus all 8 prompts in one text pass;
- view 1, 12.5 s: includes starting the resident processes;
- views 2–8: 5.7 s each.

With `--exact-prompts` it is 68.4 s, and the images are byte-identical to
encoding each prompt alone.

What made the difference:
- **Resident processes.** `NativeBackend` starts its own
  `test_cuda_qimg21_fast --serve`. It is sized for the output grid, CFG
  and the reference's condition-token count
  (`--serve-condition-tokens`), and paired with a resident
  `test_cuda_qimg21_vae --serve`. Later views reuse them, so there is no
  weight reloading between views. A request of a different shape
  re-sizes them. Both processes exit with the Python process that started
  them, or on `backend.close()`.
- **One-shot requests.** Without a prepare (a lone `generate_view` or
  `edit`), a backend's first request for a reference runs one-shot with
  the resident processes stopped. That happens even when the reference's
  encodes are already in the disk cache, because its VAE and vision
  encoders need that device memory. Every init-image request also runs
  one-shot. Later requests for that reference are resident. Use
  `--no-resident` to force one-shot throughout.
- **Caches.** The condition cache (VAE + vision encodes), the multimodal
  prompt cache and the noise cache are keyed by file content, never by path
  or mtime. They live in `tmp/qimg21-prompt-cache`.
- **Stage dumps.** The fast runner no longer writes per-block stage dumps
  when only step previews (`--dump-dir`) are asked for. That write was
  ~21 s of a 23 s edit denoise; stage dumps now need `--stage-dir`, as
  documented. The output is bit-identical.
- **Device-release wait.** The driver waits for device memory to be
  released only when an encoder subprocess actually ran, which saves 2 s
  per cached view.
- **Encode-only prepare with batched view prompts.** `generate_views` hands
  every request to `backend.prepare()` first. For each reference the
  backend hasn't used yet, the native backend runs the driver once with
  `--encode-only --prompt-batch prompts.json`, with the resident processes
  stopped. That run caches the condition image's VAE and vision encodes,
  and encodes the uncached view prompts with `test_cuda_qimg21_text
  --prompts-file` in passes of up to 48 prompts (12 with
  `--exact-prompts`), storing each in the multimodal prompt cache. Every view, the first included, then goes through the
  resident processes. The prepare time is reported with the first view's
  timings (labels starting `prepare:`).
  - In a pass, each 13.9 GB weight stream is used by every prompt.
  - **Shared prefix (default).** All of a reference's prompts share their
    system and image tokens, about 1,047 rows, before a ~175-token view
    text. `--share-prefix` computes that prefix once per pass. Each prompt
    adds only its suffix, and attention runs per prompt over the prefix
    plus its suffix.
    - 8 view prompts: 10.3 s → 4.3 s. The tokenizer is also now parsed
      once per process.
    - Each suffix is its own GEMM, sized by that prompt alone. So a
      prompt's embeddings are the same whether it is encoded alone or in
      any batch (bitwise, checked), and every multimodal encode of the
      native backend uses this split.
    - The cost: cuBLAS picks its GEMM algorithm by row count, so the
      embeddings are not bitwise the unsplit, PyTorch-exact ones. The
      difference is 1 − cos ≈ 4e-5, far below the INT8 denoiser's own
      error, and turntable images move by 26–37 dB PSNR.
    - `--exact-prompts` restores the unsplit encode, which is still batched
      and bitwise per prompt. The two modes have separate cache entries.

Memory: in the full chain (18 views plus extraction), device memory
peaked at 15.8 GB of 16.3 GB (of which ~2.7 GB belonged to other processes)
and stayed flat from view to view. Host RSS peaked at 1.9 GB. Images
stream to disk; the 64-view mock test bounds Python allocations.

The torch backend keeps the pipeline loaded; its load takes about 49 s. It
re-plans its resident blocks per sequence length, and measured 28 s for one
reference and 34 s for two, at 512² and 10 steps.

The full `image-to-3d` example above (extraction plus 18 views, cold caches)
took 262 s wall.

## Limitations

- **No geometric guarantee.** Views are generated independently from text
  that names an angle. Nothing enforces pixel correspondences, a shared
  scale, or consistency with the requested camera. `transforms.json` holds
  requested cameras, never calibration, and must not be used as ground
  truth.
- **Turnaround registration is silhouette-based.** Opposite views are
  aligned by mirror overlap, which assumes near-orthographic panels and
  similar silhouettes. When the model draws the back quite differently from
  the front (the jester's mirror IoU is 0.53), registration can only
  partly fix it.
- **Generated views are not calibrated views.**
  - Views generated one at a time, posed as cameras for Pixal3D multiview,
    made the mesh worse than the single input view (see [the
    measurement](#single-view-vs-generated-views-measured)): the model keeps
    outlines near the reference's width.
  - A turnaround sheet's views are consistent enough to help for a
    character. That result is visual, on one character, with no oracle to
    measure against.
- **Approximate viewpoint control.** Front, rear, side profiles, the
  three-quarter views, the 20° ring, the top view and the bottom view
  follow the request. The evidence:
  - **Cyclops head** (Pixal3D's `mv_images/example`, which has real renders
    at 0/90/180/270°), generated from its 0° render:
    - every 45° step turned the face the right way;
    - the silhouette IoU of the generated 90° and 270° views was higher
      against the matching render than against the mirrored one (90°:
      0.827 vs 0.735; 270°: 0.796 vs 0.713);
    - the old template got 90° backwards (0.732 vs 0.829).
  - **Bunny cake:** the 45° and 315° views look frontal. The body is a
    cylinder, so only the figurines can show the turn, and a tighter
    wording did not change them.

  The geometry sentences are what fixed the profiles, the ring and the
  bottom view. The generated views are still not metrically posed.

  Identity drifts on unseen sides (e.g. a figurine changes species on the
  back). The 2D validator catches none of this.
- **Native backend: one reference.** More references (and a mask shown as a
  reference) need `--backend torch`, which is slower.
- **Orthographic** projection is a prompt request ("orthographic-like"), not
  a rendering mode.
- **RMBG-2.0** needs its weights (`/mnt/nvme01/models/RMBG-2.0` or
  `$QIMG21_RMBG_MODEL`) and timm/torchvision. If they are missing it runs in
  `ref/pixal3d/.venv-reference-cuda310` or `$QIMG21_RMBG_PYTHON`.
- **Native encoder limits.** Init images and masks need outputs of at most
  1024 px a side. Tall or wide references get a lower condition
  resolution.
- **Text encoder cost.** The 8B text encoder streams 13.9 GB of weights
  over PCIe (Gen3 x8 here) per pass. Dataset commands batch their view
  prompts into one pass, but a single `generate_view` call with a new
  prompt still pays ~3.5 s.

## Future work

- Multi-reference conditioning in the native runner. `joint_layout.h`
  already handles N condition images, but the driver and text-encoder
  layout take one.
- Views that keep real proportions, which multiview reconstruction needs.
  Options include camera-conditioned multi-view diffusion, or
  re-rendering the single-view Pixal3D mesh as the "views" and then
  refining their texture with Qwen-Image. The latter keeps geometry
  consistent by construction. Such views can feed:
  - Pixal3D posed multiview (`--recon-mode multiview`, measured above);
  - Gaussian-splatting or NeRF fitting from `transforms.json`, with the
    cameras refined by SfM or bundle adjustment first, since the requested
    cameras are only an initialization.
