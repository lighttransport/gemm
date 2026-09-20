# Pixal3D web demo

This Python server provides a browser UI and JSON API around the native
Pixal3D CLI. Inference stays in the native CPU/CUDA/ROCm runner, so the same
service works with the project-local `uv` environments and the local model
directories.

From the repository root, prepare an environment once with
`ref/pixal3d/setup.sh cuda` (or `cpu`/`rocm`), then start:

```sh
sh server/pixal3d/run.sh --backend cuda --bind 127.0.0.1 --port 8765
```

Open <http://127.0.0.1:8765/>. Select an image, optionally provide a mask, and
choose CPU, CUDA, or ROCm. For multiview inference, select **Posed multiview**
and choose a folder containing `transforms.json` and its images. The browser
shows the resolved images in frame order before upload. A separate JSON and
image picker is available when folder selection is unsupported. GPU runs are
serialized per backend to avoid VRAM contention. Temporary uploads and GLBs
are kept only under `tmp/pixal3d/` and removed after each request.
Each multiview frame may name an explicit `mask_path` beside `file_path`.
The browser resolves and previews that association, then uploads each mask
with its view.

`POST /v1/infer` accepts JSON fields `image_b64`, optional `mask_b64`,
`image_ext`, `backend`, `fov`, `distance`, `mesh_scale`, `seed`, `threads`,
`device`, `vram_budget_mib`, `gpu_execution` (`legacy`/`resident`), and
`gpu_kernels` (`auto`/`blas`/`mma`), and `gpu_flow_precision`
(`bf16`/`mixed`/`fp32`). Output controls are `texture_size`
(`1024`/`2048`/`4096`, default `4096`) and `triangle_target`
(`10000`–`5000000`, default `1000000`). Set `include_ply: true` to receive a
binary little-endian geometry companion in `ply_b64` while retaining the GLB
for browser rendering. Responses include a `profile` object with
phase timings and device counters. Server defaults can be selected with
`--gpu-execution resident --gpu-kernels auto`; the browser also exposes these
choices. Resident execution and mixed precision are the GPU server defaults;
CPU requests automatically use legacy execution. The browser defaults to a
12288 MiB budget; native allocation remains
clamped to currently free VRAM, preserving the lower-memory path on smaller
cards. Set `reference: true` with CUDA or ROCm to
also run the pinned upstream PyTorch pipeline; the response includes a second
GLB for comparison. This is opt-in because it loads another model stack and
adds several minutes to a request. Explicit masks are supported: the server
prepares one RGBA image and passes that exact image and resolved camera values
to both implementations without running RMBG a second time. The browser
displays native and PyTorch reference meshes side by side or as an opacity
overlay. Viewer
cameras remain synchronized while comparing the meshes. `GET /health` reports
binary, GPU-library, native-model, preparation-model, and pinned PyTorch
reference readiness. Reference readiness includes the project Python and
compiled `o_voxel`/CuMesh dependencies; missing names are returned explicitly.

Native and reference results include `mesh_summary` with byte, vertex and
triangle counts and declared position bounds. A paired run also returns
`comparison` with relative count differences, the maximum bounds delta, and a
bounded 50,000-sample surface comparison. The UI summarizes symmetric Chamfer
RMS, both directional p95 distances, and orientation-independent normal
agreement. These provide reproducible geometry diagnostics alongside visual
inspection without retaining decoded meshes in the job record.

Set `render_comparison: true` together with `reference: true` to render four
camera-matched views of both results. The comparison then includes per-view
RGB MAE, RMSE, PSNR, and silhouette IoU in `comparison.renders`. Queued jobs
also retain two contact sheets at
`comparison.artifacts.native_preview` and
`comparison.artifacts.reference_preview`; the web demo displays them below
the interactive viewers. Build the bounded CPU preview renderer once with
`ref/pixal3d/build_preview.sh`. Its readiness is reported by `GET /health`.

For single-view requests, `auto_mask: true` uses the pinned RMBG-2.0 reference
when the image lacks useful alpha, and `auto_camera: true` estimates horizontal
FOV with pinned MoGe-2. The response includes resolved values and sources in
`preparation`. Multiview accepts automatic masking while retaining calibrated
FOV and transforms from its manifest. Configure model locations with `--rembg`
and `--moge`; `/health` reports each preparation model separately. RMBG-2.0 is
gated by its publisher and requires an authorized Hugging Face account when
running `prepare_auto_models.py`.

For long browser runs, `POST /v1/jobs` accepts the same body and returns a job
ID immediately. Poll `GET /v1/jobs/ID`; when its state is `complete`, fetch
`GET /v1/jobs/ID/result`. `DELETE /v1/jobs/ID` cancels a queued request or
terminates the active native/reference child process.
Queued results expose GLB and optional PLY URLs under `artifacts`. Native and
reference runners write `.partial` files directly in the job directory; the
server validates, flushes, and atomically publishes them without encoding or
decoding base64. Artifact downloads are streamed in 1 MiB blocks. The
synchronous `POST /v1/infer` response keeps its original base64 fields for
API compatibility.
Job status includes a monotonic `progress` percentage and a `phase` derived
from native conditioning, diffusion, mesh, and texture milestones.
Read `GET /v1/jobs/ID/log` for the retained diagnostic tail. The default
`--job-log-bytes 65536` cap is enforced while the child runs, so noisy native
diagnostics cannot grow without bound.
The bounded worker queue defaults to four active requests and four retained
terminal results; change it with `--retained-jobs`. Each state transition is
written to an atomic `results/ID/job.json` manifest. On startup, completed
jobs and their artifact URLs are restored. Work that was queued or running
when the process stopped is retained as failed with `server_restarted`; it is
never submitted again automatically. Invalid manifests and incomplete
artifact sets are removed during recovery.
Terminal jobs expire after 24 hours by default. Deleting a completed, failed,
or cancelled job releases it immediately; deleting queued or running work
continues to request cancellation. Configure expiry with `--job-ttl` in seconds.
Errors include a stable `error_code` such as `invalid_request`, `queue_full`,
`device_busy`, `storage_full`, `server_shutdown`, `timeout`, `not_found`,
`server_restarted`, or `internal_error`. Queue saturation returns HTTP 429
and low storage admission returns HTTP 507. `/health` publishes request,
image, output, view-count, admission, storage, logging, and device-lock status.

The browser sends each image as raw bytes to `POST /v1/uploads`, then places
the returned `upload_id` in `image_upload`, `mask_upload`, or each view's
`image_upload` and optional `mask_upload`. This avoids base64 expansion and
keeps queued JSON requests small. Upload IDs are single-use and their files are removed when the job
finishes or is cancelled. Unclaimed uploads expire after one hour by default,
and `DELETE /v1/uploads/ID` releases one immediately. Configure expiry with
`--upload-ttl` in seconds. Existing `image_b64` clients remain supported.

Run the server unit tests with `python3 -m unittest server.pixal3d.test_app`.
When Chrome or Chromium is installed, `python3 server/pixal3d/test_browser.py`
boots the real HTTP handler and drives the page through Chrome DevTools. It
verifies single-view and multiview uploads, queued polling, native/reference
downloads, PLY delivery, readiness rendering, and active-job cancellation.

The opt-in real test downloads an upstream image at a pinned Git commit,
verifies its SHA-256, and exercises the queued HTTP API on CUDA. It forces the
automatic RMBG path, cancels one real job, verifies recovery with a complete
generation, downloads and validates its file-backed GLB, and records elapsed
time plus peak process RSS and device memory:

```sh
ref/pixal3d/run.sh cuda server/pixal3d/test_real_cuda.py \
  --vram-budget-mib 12288
```

Run the paired explicit-mask path separately to reuse the fixture alpha in both
native and pinned PyTorch inference and verify the bounded surface metrics:

```sh
ref/pixal3d/run.sh cuda server/pixal3d/test_real_cuda.py \
  --vram-budget-mib 12288 --skip-cancel --explicit-mask --reference
```

Both commands write their reproducibility record and validated artifacts under
`tmp/pixal3d/real-web-cuda/`. They require network access for the pinned input,
the local model paths reported by `/health`, and an NVIDIA GPU.

On 2026-09-20, the automatic-mask command passed on an RTX 5060 Ti 16 GB in
312.1 seconds of harness wall time. This includes the queued upload, automatic
RMBG preparation, cancellation, recovery resident-mixed generation, GLB
serialization, download and artifact validation. The server serialized the
CUDA backend with no competing test job, although desktop GPU processes remained.
The recovery generation
peaked at 11,656,101,888 device bytes and 6,946,299,904 aggregate host RSS
bytes and produced a validated 9,335,888-byte GLB. The input SHA-256 was
`fdd82d60b7ec11e6d5699df29693d8ab538f9dab4b04e3f2abaa59ccd7b4709a`.
The paired explicit-mask command passed in 689.2 seconds of harness wall time,
including serialized resident-mixed native inference, GLB serialization, pinned
PyTorch inference, reference serialization, downloads and surface validation.
It ran through the same single CUDA worker with desktop GPU processes present.
The run peaked at 13,285,916,672 device bytes and 25,487,593,472 aggregate host
RSS bytes and
produced validated native/reference GLBs. Its 50,000-sample symmetric Chamfer
RMS was `0.015580`; directional p95 distances were `0.027573` and `0.034187`.

For multiview API requests, replace `image_b64` with `views`, an ordered array
of 1 to 16 objects. Each object contains `image_b64`, a 4-by-4
`transform_matrix`, and optional `mask_b64` and `fov`. Top-level `fov` is the default for
frames without one, and `mesh_scale` applies to the complete view set:

```json
{
  "backend": "cuda",
  "views": [
    {"image_b64": "...", "mask_b64": "...", "transform_matrix": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}
  ],
  "fov": 0.857556,
  "mesh_scale": 1.0
}
```

The pinned PyTorch comparison supports both single-view and multiview GPU
requests. It runs after native inference and reuses the validated ordered view
manifest. Automatic and explicit per-view masks are resolved once; the exact
prepared RGBA files are passed to both native and PyTorch pipelines. Enabling
the comparison can add several minutes to a request.

## Deployment

The built-in server is intended for a trusted workstation or an application
behind an authenticated reverse proxy. It binds to loopback by default. When
exposing it through a proxy, keep the 256 MiB request limit, allow multi-hour
upstream timeouts, disable proxy response buffering for job polling, and add
TLS and authentication at the proxy. Do not expose an unauthenticated
`--bind 0.0.0.0` endpoint.

CUDA inference is serialized across processes with Linux `flock` files under
`tmp/pixal3d/device-locks` by default. Select a shared local directory with
`--device-lock-dir` when services use different work trees. The lock is keyed
by CUDA device index; waiting is cancellable and counts against the existing
inference/reference timeout. A timeout is reported as `device_busy`.
Keep `tmp/pixal3d/web-runs` on a local filesystem with room for uploads and
generated assets. Queued output artifacts are retained under its `results`
directory until job expiry or deletion. Size `--retained-jobs` and filesystem
capacity for native GLB, optional PLY, and reference GLB output together.
Keep that directory on one local filesystem: manifests are flushed to a
temporary file beside the final path and installed with an atomic rename.
Recovery applies the configured TTL and retained-job limit before serving
requests.

New jobs are rejected before consuming uploads when free space is below
`--min-free-disk-mib` (default 1024). On SIGINT or SIGTERM the server stops
admission, marks queued jobs `server_shutdown`, cancels the active child,
persists terminal state, and waits up to `--shutdown-timeout` seconds
(default 30) before closing the HTTP listener.

Use a service manager to restart the process and set a file-descriptor limit
appropriate for concurrent uploads. Check `GET /health` after startup and
before routing traffic. Its readiness fields distinguish the native model,
multiview checkpoints, RMBG, and MoGe so a missing optional model does not hide
core inference readiness.
