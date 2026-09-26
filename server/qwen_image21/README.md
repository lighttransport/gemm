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
  side-by-side, followed by a comparison panel (see below).

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

## Step previews

While a run is in flight, each result card shows the picture as it develops.
Below it is a filmstrip of every step: click a step to hold it, and click it
again to follow the run. The filmstrip stays under the final picture, with a
"final" frame to return to.

The runners already dump `step_NNN.npy` after every step for parity work. The
server watches those files, so the runners need nothing new, and turns each
one into two things:

- **The picture it is heading for.** Step `i` moves the latent from `sigma_i`
  to `sigma_i+1` along one velocity, so `x0 = x - sigma_i+1 * dx / dsigma`,
  using the same FlowMatch schedule the runners use. The raw latent is mostly
  noise until the last few steps; `x0` shows the composition from early on.
- **RGB without the VAE.** A fitted affine map reads each token and its eight
  neighbours and writes an 8x8 RGB patch (`cuda/qimg21/latent_preview.npy`,
  about 25 dB against the real decode). The previews are soft by design and
  capped at 384 px a side.

Refit the map on your own runs with
`python cuda/qimg21/fit_latent_preview.py --jobs tmp/qimg21-web-jobs --out cuda/qimg21/latent_preview.npy`.
Previews are sent only to a client polling `/api/progress`, as `preview`
events carrying `source` (`native` or `reference`), `index`, `total`, `sigma`
and a PNG data URL. The PyTorch reference now always saves its initial noise
(`--dump-initial-latents`), which the step-0 preview needs; it does not change
the picture.

## Compare mode

A compare runs the **reference first**, with `--dump-initial-latents`. The native
runner then starts from that file (`native_generate.py --initial-latents`)
instead of drawing its own noise from the seed. The two pictures then share every
input on any reference device, and any difference comes from what each
implementation computes. `reference.matched_noise` in the response says whether
this worked.

The response also carries `compare`, computed with the same math as
`cuda/qimg21/compare.py`:

- `stages`: cosine and relative L2 of the native run against the reference for
  the initial latents, the text embeddings, every denoising step (`step_NNN`),
  and the decoded image.
- `image`: PSNR, MAE, RMSE and max |Δ| over the RGB pixels (0-255).
- `first_divergence`: the first stage whose cosine falls below 0.999, or `null`.
  This is where to start looking when a picture goes wrong. It is deliberately
  looser than the BF16 parity gate: it locates a break, it does not certify a
  kernel.

Metrics never block the images. A failure shows up as `compare.error`.

The page adds three views under the two result cards:

- **Overlay**: a swipe split you drag anywhere across the picture, a blend with an
  opacity slider, and a flicker toggle that swaps the two in place.
- **Diff**: the per-pixel largest-channel |Δ| with an adjustable gain, a heat or
  gray palette, an optional "only |Δ| > N" mask, and a readout of both RGB
  values under the pointer.
- **Metrics**: the stage table, with the first divergent stage highlighted.

## The PyTorch reference: CUDA, ROCm or CPU

The reference runs `cuda/qimg21/reference.py`, which takes `--device`:

| `reference_device` | Runs on | Needs |
|---|---|---|
| `cuda` | the NVIDIA GPU | a CUDA PyTorch build |
| `rocm` | the AMD GPU | a ROCm PyTorch build, in its own environment |
| `cpu` | the processor | any PyTorch build |

`cuda` and `rocm` name a **PyTorch build** as much as a device. PyTorch exposes
ROCm through the `cuda` namespace, so both requests resolve to the same device
string and differ only in which build is installed; the driver refuses a
mismatch with the reason rather than failing on an import or a missing device.
Each GPU vendor gets its own interpreter, `--python` and `--python-rocm`. The CPU
route needs no build of its own and defaults to `--python`, since a CUDA build
runs CPU kernels perfectly well; pass `--python-cpu` to name another.

`reference_device` defaults to the request's `backend`, so existing requests keep
pointing where they always did. A native-only request does not need a reference
and is not refused when no PyTorch build is installed at all.

`GET /api/health` reports `reference` (which devices work) and
`reference_detail` (the torch version, or the reason a device is unavailable).
The form offers only the devices that work and moves off a selection that turns
out to be gone.

**A CPU reference is not a parity result.** It runs different kernels on
different hardware, so a side-by-side against the native runner is a look at
both pictures. A compare on CPU says so on the result rather than letting the
pairing imply agreement; `cuda` and `rocm` are the devices a comparison can
actually mean.

**CPU is slow enough to be a debugging route, not a demo default.** Measured on
32 cores at 256x256 with one step: the denoise step took 59 s, and the VAE
decode had not finished after 21 minutes. Budget tens of minutes per image, and
expect the decode rather than the denoiser to be the long pole.

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
size, then a refine that upscales the base picture in pixel space, re-encodes
it, and denoises the output grid one tile at a time, so a large picture fits the same VRAM. The decoder is tiled
spatially in the same way, so a 2048 or 4096 pixel output decodes in under
2 GB instead of the 11.8 GB an untiled 2048 decode needs.

| Field | Default | What it does |
|---|---|---|
| `upscale` | 1 (off) | above 1 turns the two-pass refine on; the form sends it only while the section is unlocked (CUDA fast preset, native mode) |
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

`--python-cpu` names the interpreter for a CPU reference, if you want one
separate from `--python`.

The API accepts `backend=cuda|rocm` and `mode=native|reference|compare`,
`reference_device=cuda|rocm|cpu`, an optional `job` id to attach to progress,
`profile_steps` for per-step timings, plus
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
every tiling field, the progress parser, and the reference device routing
including the build probe that decides which devices a machine can offer. `test_form.py` extracts the page's inline script and runs it
under `node` against a DOM stub, so a mistyped identifier or a size cap
computed from the wrong branch is caught here rather than surfacing as a broken
page, including the progress table's row rendering; it skips if `node` is not
installed.
