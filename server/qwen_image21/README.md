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

The API accepts `backend=cuda|rocm` and `mode=native|reference|compare`, plus
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
`GET /api/health` and `POST /api/generate`.

Tests:

```sh
python3 -m pytest server/qwen_image21 -q
```

`test_app.py` covers the routing, the preset selection and the validation of
every tiling field. `test_form.py` extracts the page's inline script and runs it
under `node` against a DOM stub, so a mistyped identifier or a size cap
computed from the wrong branch is caught here rather than surfacing as a broken
page; it skips if `node` is not installed.
