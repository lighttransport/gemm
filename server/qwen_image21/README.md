# Qwen Image 2.1 web demo

This is a new standalone demo; it does not reuse the older Qwen Image web
application. It exposes three run modes and two accelerator backends:

- `CUDA` runs `cuda/qimg21/native_generate.py` with the native custom CUDA
  runner and native VAE decode.
- `ROCm` uses the same orchestration and model ABI, with native text, vision,
  denoiser, and VAE binaries supplied by `rdna4/qimg21/`. PyTorch's ROCm build
  is needed only for the optional reference path.
- `PyTorch` runs the pinned Diffusers reference script.
- `Compare` runs both sequentially under one device lock and renders them
  side-by-side.

The quantization switch applies the exported row-INT8 package. CUDA enables
custom INT8 tensor-core GEMMs and the calibrated 16-block BF16 tail; ROCm
dequantizes each streamed matrix to BF16 for its WMMA GEMMs. The PyTorch
reference remains unquantized so it stays an arithmetic reference. On the RX
9070 XT the dequantized ROCm route is currently slower than the native BF16
checkpoint; leave the switch off for performance runs.

The CUDA denoiser selector also offers the fast runner's presets
(`cuda/qimg21/test_cuda_qimg21_fast`, see `cuda/qimg21/README.md`):
`low8` (INT8, under 8 GB), `low8-fp4` (NVFP4), `fast12` (INT8, everything
resident, about 11 GB) and `accurate` (BF16, bit-identical to the parity
harness). Build them with `make -C cuda/qimg21 fast`. The INT8 and NVFP4
packages default to `/mnt/nvme01/models/qimg-21-fast/`; override them with
`--int8-package` and `--nvfp4-package`. The API field is `preset`, CUDA only,
and `GET /api/health` reports which presets are available.

## Live progress

While a run is in flight the page shows the pipeline stage, a progress bar and a
row per denoising step, and it keeps polling until the run finishes.

The per-step duration is the denoiser's **own measured device time**, not a
wall-clock delta between polls. The page sets `profile_steps`, the server turns
that into `native_generate.py --profile-steps`, and the fast runner brackets
each step with a CUDA event and prints it. The `cum` column is the running sum of
those same measurements, so it agrees with the runner's own closing line
(`fast: prefill 0.448 s, 8 steps 20.900 s`). The 250 ms poll interval only
decides how promptly a step appears; it cannot change the numbers. A step the
runner did not time shows `--` rather than an estimate.

The same feed carries the memory plan, the prompt-encoding layers, and the
refine and decode tiles, so a slow stage is attributable to a stage rather than
just "it is taking a while".

Two endpoints, both optional. A client that does not send `job` behaves exactly
as before and just waits for `POST /api/generate`:

```sh
curl -s -X POST -H 'content-type: application/json' \
  -d '{"job":"demo1","backend":"cuda","mode":"native","prompt":"a red lantern",
       "width":1024,"height":1024,"steps":8,"seed":7,"preset":"low8",
       "profile_steps":true}' http://127.0.0.1:8091/api/generate &
curl -s 'http://127.0.0.1:8091/api/progress?job=demo1&since=0'
```

`GET /api/progress?job=<id>&since=<cursor>` returns only the events after
`cursor` and the new `cursor`, so polling is incremental. Records are kept for
`PROGRESS_TTL` after a run and an unknown `job` is a 404.

`profile_steps` needs a fast CUDA preset: the parity harness has no `--profile`,
and `native_generate.py` refuses the combination rather than silently reporting
nothing. It costs a per-step sync, about 1.4% on `low8`.

## High resolution and tiled refinement

The form has a **High resolution** section that drives the coarse-to-fine path
in `cuda/qimg21/native_generate.py`: a base pass at `1/upscale` of the requested
size, then a refine that resamples the base up and denoises the output grid one
tile at a time, so a large picture fits the same VRAM. The decoder is tiled
spatially in the same way, so a 2048 or 4096 pixel output decodes in under
2 GB instead of the 11.8 GB an untiled 2048 decode needs.

| Field | Default | What it does |
|---|---|---|
| `upscale` | 2 in the form, 1 off | above 1 turns the two-pass refine on |
| `base_steps` | same as `steps` | steps for the coarse base pass |
| `refine_strength` | 0.4 in the form | fraction of the schedule the refine re-runs |
| `refine_seed` | 0 | seeds the refine's noise, independent of `seed` |
| `tile_tokens` | **auto** | refine tile side in latent tokens |
| `tile_overlap` | 8 | latent tokens neighbouring refine tiles share |
| `vae_tile` | **auto** | decode tile side in latent tokens |
| `vae_tile_overlap` | 8 | latent tokens neighbouring decode tiles share |
| `vae_tile_bleed` | 2 | latent tokens discarded at each decode tile edge |

**auto** means the driver decides: `tile_tokens` becomes the largest tile whose
plan still fits the preset's budget, and `vae_tile` defaults to 48 latent
tokens above 1024 pixels. Leave them empty unless you want to trade quality
for time. Each refine tile is composed as its own canvas, so a smaller tile
means more independently re-drawn detail; on most sizes the largest tile that
fits is the whole grid, and the default is then no tiling at all. The driver
prints what it chose and the demo shows it under the image, together with the
memory plan and the per-stage timings.

Output size limits, which the form applies to the width and height inputs and
`GET /api/health` reports:

| Run | Limit |
|---|---:|
| parity harness, ROCm, PyTorch reference | 1024 |
| fast CUDA run untiled | 2048 |
| fast CUDA run with a tiled refine | 4096 |

A tiled refine needs a CUDA denoiser preset and native mode: the parity
harness has no tile path, and the PyTorch reference cannot do a tiled refine,
so a comparison would be between two different things. Both are refused with
that reason rather than failing later. `vae_tile_overlap` and `vae_tile_bleed`
are only accepted alongside an explicit `vae_tile`, since the driver picks all
three together otherwise; and the bleed must be at most half the overlap or the
decode tiles leave gaps.

Start it after building the native binaries:

```sh
make -C cuda/qimg21 native
server/qwen_image21/run.sh --host 127.0.0.1 --port 8091
```

Select the RDNA4 path explicitly after building its native binaries and fused
WMMA attention plugin with `make -C rdna4/qimg21`:

```sh
server/qwen_image21/run.sh \
  --python-rocm tmp/qimg21-rocm-venv/bin/python \
  --native-rocm rdna4/qimg21/test_hip_qimg21_native
```

The API accepts `backend=cuda|rocm` and `mode=native|reference|compare`, an
optional `job` id to attach to progress, `profile_steps` for per-step timings,
plus
the tiling fields above (`upscale`, `base_steps`, `tile_tokens`,
`tile_overlap`, `refine_strength`, `refine_seed`, `vae_tile`,
`vae_tile_overlap`, `vae_tile_bleed`). Blank or null means "let the
driver choose".
Legacy `mode=cuda` and `mode=rocm` requests remain accepted as native runs.
`GET /api/health` reports readiness independently for both backends.

Override paths when needed:

```sh
server/qwen_image21/run.sh \
  --model /mnt/nvme01/models/qimg-21 \
  --quant-package tmp/qimg21-int8-package
```

The browser is at `http://127.0.0.1:8091/`; the JSON endpoints are
`GET /api/health`, `POST /api/generate` and `GET /api/progress`.

Tests:

```sh
python3 -m pytest server/qwen_image21 -q
```

`test_app.py` covers the routing, the preset selection, the validation of
every tiling field and the progress parser. `test_form.py` extracts the page's inline script and runs it
under `node` against a DOM stub, so a mistyped identifier or a size cap
computed from the wrong branch is caught here rather than surfacing as a broken
page, including the progress table's row rendering; it skips if `node` is not
installed.
