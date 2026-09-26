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

## Refine with more steps

Preview with a few steps, then refine the picture you like. A native result from
a CUDA fast preset carries a **Refine with more steps** bar with two settings:
- a step count;
- **keep**, the share of the longer schedule taken as already done.

Refining keeps the preview's layout and adds detail. The preview card stays
beside the result, and a refined result can be refined again.

How it works:
- A few-step result is a rough draft of what a longer schedule would
  converge to, so it is re-noised to step `K = keep * steps` of the new
  schedule, and only steps `K+1..N` run.
- It is re-noised with the **same noise the preview started from** (the fast
  runner's `--restart-noise`), not a fresh draw. That is the point the longer
  run's own trajectory would pass through if the preview were its answer.
- At keep 0 this is exactly a fresh N-step run from that noise, which can frame
  the scene differently: a flow schedule's first steps decide the layout.
  Around 50% keeps the layout and still redraws the detail.

The request fields are:

- `restart_from`: the earlier result's job id, as returned in the response.
- `restart_keep`: a value from 0 to 0.95.

A refine must be a CUDA fast preset in native mode, untiled, and the same
size as the earlier run. The server keeps each job's request in
`request.json` to check that.

The driver flags are `native_generate.py --initial-latents NOISE
--restart-from LATENTS --restart-step K`.


## Resident denoiser

Loading the transformer (about 7 GB for `fast12`, 2.6 s) was most of a short
run's setup, and a refine or a new seed at the same size does not need it
again. So the server keeps one `test_cuda_qimg21_fast --serve` process loaded
between runs.

**Starting it.** The first eligible run starts the process. Eligible means
native mode, a CUDA fast preset, and no tiled refine. The load shows in that
run's breakdown under *load resident denoiser*.

**Reusing it.** Later runs with the same setup send their denoise to it over a
Unix socket, and the breakdown reads *reuse resident transformer weights 0 s*.
The setup is binary, model, preset and package, output size, and whether a
negative prompt doubles the batch; steps, seed, prompt and refines may vary.
Prompts are served up to 512 tokens.

**Stopping it.**
- A run with a different setup replaces the process.
- Any other kind of run (compare, the PyTorch reference, a tiled refine, the
  parity harness, ROCm) stops it first, so its VRAM is free.
- It stops after 15 minutes idle.
- It stops when the server exits. The process also watches its parent's pid,
  because the kernel's parent-death signal fires when the parent *thread*
  exits, and the demo starts it from a request thread.

`GET /api/health` reports `resident`.

The VAE decoder rides along: a `test_cuda_qimg21_vae --serve` process with the
same lifetime, for outputs up to 1024². It keeps its roughly 1 GB of F32
weights on the device, so a decode drops from 2.4 s to 0.55 s. Fast-preset
runs also decode with TF32 cuDNN convolutions (`--vae-tf32`). That is 22%
faster, and the image still measures 52.7 dB against the BF16 reference VAE,
as the F32 decode does.

Output from the resident processes is bit-identical to one-shot runs with the
same inputs. At 512² with `fast12`:

| Run | One-shot | Resident denoiser | Denoiser and VAE |
|---|---|---|---|
| 20-step refine | 10.0 s | 6.1 s | 5.1 s |
| New 6-step seed | about 9.4 s | 5.7 s | 4.1 s |

**Protocol.** One connection carries one run:
- **Request:** a single line of the usual per-run flags, tab separated.
- **While it runs:** stderr is the connection, so step, timing and preview
  output flows as in a one-shot run.
- **Last line:** `fast-serve: status N`. 0 is done, 2 is a bad request, and 3
  means the run is not this process's setup; the driver then runs one-shot.
  The driver also runs one-shot when the process goes away mid-run.

The driver flag is `native_generate.py --resident-socket PATH`.

## Agreement with PyTorch

A denoising trajectory amplifies tiny arithmetic differences step by step, so
the reference's own spread sets the bar. The same PyTorch run with a different
SDPA backend (memory-efficient against the default) ends at cosine **0.99939**
at step 19 (512², 20 steps), and its image differs by 34.3 dB.

`cuda/qimg21/trajectory_sweep.py` runs the fast runner from each reference's
own noise and reports that step-19 distance. Mean 1 - cosine over three
prompts and seeds, with denoise time at 512² x 20 steps:

| Weights | Attention | 1 - cos | Denoise |
|---|---|---:|---:|
| BF16 (`accurate`) | exact | 4.0e-4 | 15.1 s |
| BF16 (`accurate`) | flash | 3.6e-4 | 14.0 s |
| INT8 (`fast12`) | sage (default) | 1.3e-3 | 5.0 s |
| INT8 (`fast12`) | flash | 1.4e-3 | 5.1 s |
| INT8 (`fast12`) | exact | 7.8e-4 | 5.8 s |
| NVFP4 (`low8-fp4`) | sage | 6.3e-3 | 4.7 s |
| PyTorch vs PyTorch | efficient vs default | about 6e-4 | |

What the sweep shows:
- **BF16 is already below the noise floor**, so a closer cosine there is not
  measurable. The per-stage metrics say the same: the text embeddings and the
  VAE are not where it diverges. Decoding the reference's own final latents
  with the native VAE gives 52.7 dB, and running the denoiser from the
  reference's exact embeddings leaves the drift unchanged.
- **For INT8, the attention kernel is the lever.** Exact attention moves every
  case closer, reaching the floor, for 17% more time. The form's **Attention**
  select offers it for fast presets (API field `attention`: `exact`, `flash`
  or `sage`).
- **Keeping sensitive blocks in BF16 did not help** FP4 or INT8 consistently.
- **The int8-smooth-a0.5 package** measured closer than a0.6 on all three
  cases. The runner README records a0.6 winning other comparisons, so the
  default stands.

## PyTorch reference: fair timing

On a 16 GB card neither the 14 GB BF16 transformer nor the 15 GB text encoder
fits whole, and the original reference streamed every module at every use
(`enable_sequential_cpu_offload`). That measured the PCIe link, not PyTorch.
`reference.py --offload` now offers three placements, all computing the same
numbers:

| `--offload` | Transformer | 512², 20 steps: denoise |
|---|---|---:|
| `sequential` | every module copied at each use | about 57 s |
| `group` | block by block, next block prefetched on a side stream | 45.3 s |
| `resident` | as many blocks on the device as fit (17 to 20 of 32); the rest stream through two prefetched slots | 22 to 24 s |

The native `accurate` preset, BF16 with the same kind of plan, denoises the same
run in about 16 s.

**Serving.** For CUDA the demo keeps a PyTorch server loaded
(`reference.py --serve --offload resident`, about 45 s once):
- Its weights live in pinned host memory.
- Each run puts the resident blocks and the VAE on the device (1.5 s) and
  takes them off afterwards. Between runs it holds only its CUDA context
  (about 1 GB), so the native runners keep the GPU.
- A served run is byte-identical to the one-shot `sequential` run.

The form's **PyTorch weights** select (API `reference_offload`) switches back
to the original one-shot sequential run.

`resident` places blocks with its own two-slot ring, not diffusers' group
offloading. Pinning some groups resident inside group offloading races with
its stream prefetch and gives wrong numbers from the second step on.

## Timing breakdown

Each card reports its own wall time: in compare mode the runner and the
reference no longer share one number. Each card shows three figures:

- **total:** the card's own wall time.
- **image generation:** the denoising loop alone.
- **everything else:** loading and setup.

A breakdown table under the picture lists every phase, grouped by pipeline
stage, and the progress panel shows the same phases live as they finish.

Every component prints `timing: <phase> <seconds> s[ (<detail>)]`; the server
parses those lines, so the numbers are each component's own:

| Component | Phases |
|---|---|
| `test_cuda_qimg21_text` | tokenize, CUDA init, kernels, buffers, the 36 layers (with GB streamed, GB/s, and time spent waiting on the upload), write |
| `test_cuda_qimg21_fast` | CUDA init + kernels + memory plan, transformer weights, buffers, prompt prefill, denoise (image generation) |
| `test_cuda_qimg21_vae` | CUDA init, decode, write |
| `reference.py` | torch import, `from_pretrained`, offload setup, prompt encoding, denoise (image generation), VAE decode, the rest of the pipeline, write |

The response carries them per side as `timings` (rows of `stage`, `label`,
`seconds`, `detail`, and `total` for a stage's wall time), `elapsed_ms` and
`generation_s`.

**Where the native setup time goes, and what was cut.** The text encoder
already runs on the GPU. Its cost is streaming about 14 GB of BF16 weights per
prompt for a few dozen tokens of compute. On this host that is bounded by PCIe
Gen3 x8 (7.2 GB/s) and page-cache reads (about 8 GB/s), roughly 2 s.

- The weights now stream through a two-slot pinned ring. A loader thread reads
  ahead with parallel `pread()` while the GEMMs run. The embeddings are
  bit-identical to before, and skipping the teardown of the checkpoint mapping
  at exit saves most of a second.
- A repeated prompt skips the encoder: embeddings are cached in
  `tmp/qimg21-prompt-cache`, keyed by prompt, checkpoint and encoder build.
- The initial noise, which costs a torch import to draw, is cached there by seed
  and grid.
- A fixed 2 s sleep after the text encoder is now a wait for its device memory
  to come back, about 30 ms.

A 512², 10-step `fast12` run went from about 16.5 s to 14 s the first time, and
to 9.7 s with a cached prompt and seed.


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
