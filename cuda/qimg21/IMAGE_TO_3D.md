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
  validation.

It does **not** reconstruct 3D. Every generated view is a 2D image that the
model *was asked* to render from some camera; nothing makes the views agree
with each other geometrically (see [Limitations](#limitations)).

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
| `object-preprocess` (alias `image-to-3d-preprocess`) | photo → RGBA object: `--method qwen` (default; the model's native transparent extraction) or `rmbg` (RMBG-2.0 matting, alpha only) or `alpha` (use the input's alpha). `--pixels original` (default) keeps the source RGB and takes only the alpha, so detail and geometry come from the photo. Then center/scale (`--fill 0.85`, `--pad`, `--crop`, `--size`, `--keep-scale`, `--no-center`). |
| `edit` | object-preserving image-to-image edit (`--instruction`). `--strength 1` edits through image conditioning; below 1 it is SDEdit from the input. `--mask`/`--mask-rect x,y,w,h`/`--mask-circle cx,cy,r` limit the change (see [Masks](#masks)). |
| `texture-preprocess` | `--ops neutralize-lighting,reduce-shadows,reduce-specular,remove-reflections,repair-defects,remove-background` with an instruction that forbids beautifying and keeps text/logos. |
| `multiview` | views of a reference: `--views N [--elevation E]`, or rings `--azimuth-views N --elevations -20,0,20`, `--top-bottom`, extra `--reference` photos (torch backend). |
| `turntable` | N evenly spaced azimuths at one elevation, flat `images/NNN.png` layout. |
| `image-to-3d` | the whole chain: extract → normalize → optional `--cleanup` → views → validate → export. `--keep-background` skips extraction for an already clean RGBA. |
| `validate DIR` | re-run the 2D validation of a dataset and rewrite `validation.json`. |

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
  "seen from slightly above", "rear view", "top view". It also carries the
  identity-preservation clauses and the background clause ("The image has
  alpha channel and the background is transparent." by default). Use
  `--template-file` to replace the template, and `--instruction` to append
  text to it.
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

## Performance and memory

Measurements are at 512², 16 steps, `fast12` (INT8 weights + SageAttention),
on an RTX 5060 Ti 16 GB, with a 1024-px condition image of the bunny cake.

| | per view |
|---|---|
| before (one-shot processes, cached condition, flat 2 s device wait) | 28.2 s |
| resident denoiser + VAE decoder, prompt cached | **6.9 s** (denoise 5.7 s, decode 0.6 s) |
| resident, new prompt encoded alone (every view of a new object, before batching) | 12.3–13.7 s (text encoder 3.5–4.7 s) |
| same, `--condition-resolution 512` | 10.2–10.7 s (denoise 3.6 s) |
| **new object, encode-only prepare + batched prompts** (views 2–8 of 8) | **5.9–6.0 s** |

For a whole new-object turntable (8 views, empty caches), the total went
from about 115 s to **72.5 s**:
- prepare, 17.1 s: condition encodes plus all 8 prompts in one text pass;
- view 1, 12.9 s: includes starting the resident processes;
- views 2–8: 6.0 s each.

The images are byte-identical to the unbatched run.

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
  --prompts-file` in passes of up to 12, storing each in the multimodal
  prompt cache. Every view, the first included, then goes through the
  resident processes. The prepare time is reported with the first view's
  timings (labels starting `prepare:`).
  - In a pass, each 13.9 GB weight stream is used by every prompt: about
    0.75 s per extra prompt instead of 3.5 s.
  - GEMMs, attention and RoPE positions run per prompt, so each prompt's
    embeddings are bitwise the single-prompt ones (checked on three prompts
    of different lengths).

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
- **Weak viewpoint control.** In the bunny-cake run:
  - front, rear and three-quarter views were plausible;
  - 90° side views were often three-quarter views;
  - the 20° elevation ring was barely distinguishable from 0°;
  - the top view (+90°) worked;
  - the **bottom view (−90°) came back as a front view**.

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

- All view prompts of one reference share their system and image prefix
  (~1,030 of ~1,050 rows). Computing that prefix once per batch would cut
  the prepare text pass by several times. It would, however, change the
  GEMM row blocking, and so the bits, relative to the PyTorch-parity
  single-prompt path.
- Multi-reference conditioning in the native runner. `joint_layout.h`
  already handles N condition images, but the driver and text-encoder
  layout take one.
- A geometry-aware stage for real consistency (e.g. camera-conditioned
  multi-view diffusion, or reprojection checks with a depth model such as
  MoGe, which is in `ref/pixal3d`). Its output can feed:
  - Pixal3D posed multiview (`--views-dir`, as smoke-tested here);
  - Gaussian-splatting or NeRF fitting from `transforms.json`, with the
    cameras refined by SfM or bundle adjustment first, since the requested
    cameras are only an initialization.
