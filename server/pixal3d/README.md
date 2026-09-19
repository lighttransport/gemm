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
requires an image without a separate mask upload. The browser displays native
AMD and PyTorch reference meshes side by side or as an opacity overlay. Viewer
cameras remain synchronized while comparing the meshes. `GET /health` reports
binary, GPU-library, native-model, preparation-model, and pinned PyTorch
reference readiness.

Native and reference results include `mesh_summary` with byte, vertex and
triangle counts and declared position bounds. A paired run also returns
`comparison` with relative count differences and the maximum bounds delta;
these provide reproducible structural diagnostics alongside visual inspection.

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
Queued results expose GLB and optional PLY URLs under `artifacts`; the binary
files remain on disk instead of being retained as base64 strings in server
memory. The synchronous `POST /v1/infer` response keeps its original base64
fields for API compatibility.
Job status includes a monotonic `progress` percentage and a `phase` derived
from native conditioning, diffusion, mesh, and texture milestones.
The bounded in-memory queue defaults to four active requests and four retained
terminal results; change it with `--retained-jobs`.
Terminal jobs expire after 24 hours by default. Deleting a completed, failed,
or cancelled job releases it immediately; deleting queued or running work
continues to request cancellation. Configure expiry with `--job-ttl` in seconds.
Errors include a stable `error_code` such as `invalid_request`, `queue_full`,
`timeout`, `not_found`, or `internal_error`. Queue saturation returns HTTP 429,
and `/health` publishes request, image, output, and view-count limits.

The browser sends each image as raw bytes to `POST /v1/uploads`, then places
the returned `upload_id` in `image_upload`, `mask_upload`, or each view's
`image_upload`. This avoids base64 expansion and keeps queued JSON requests
small. Upload IDs are single-use and their files are removed when the job
finishes or is cancelled. Unclaimed uploads expire after one hour by default,
and `DELETE /v1/uploads/ID` releases one immediately. Configure expiry with
`--upload-ttl` in seconds. Existing `image_b64` clients remain supported.

Run the server unit tests with `python3 -m unittest server.pixal3d.test_app`.
When Chrome or Chromium is installed, `python3 server/pixal3d/test_browser.py`
boots the real HTTP handler and drives the page through Chrome DevTools. It
verifies single-view and multiview uploads, queued polling, native/reference
downloads, PLY delivery, readiness rendering, and active-job cancellation.

For multiview API requests, replace `image_b64` with `views`, an ordered array
of 1 to 16 objects. Each object contains `image_b64`, a 4-by-4
`transform_matrix`, and an optional `fov`. Top-level `fov` is the default for
frames without one, and `mesh_scale` applies to the complete view set:

```json
{
  "backend": "cuda",
  "views": [
    {"image_b64": "...", "transform_matrix": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}
  ],
  "fov": 0.857556,
  "mesh_scale": 1.0
}
```

The pinned PyTorch comparison supports both single-view and multiview GPU
requests. It runs after native inference and reuses the validated ordered view
manifest, so enabling it can add several minutes to a request.

## Deployment

The built-in server is intended for a trusted workstation or an application
behind an authenticated reverse proxy. It binds to loopback by default. When
exposing it through a proxy, keep the 256 MiB request limit, allow multi-hour
upstream timeouts, disable proxy response buffering for job polling, and add
TLS and authentication at the proxy. Do not expose an unauthenticated
`--bind 0.0.0.0` endpoint.

Run one server process per physical GPU. The process serializes requests for
each backend, but separate server processes do not share locks or VRAM budgets.
Keep `tmp/pixal3d/web-runs` on a local filesystem with room for uploads and
generated assets. Queued output artifacts are retained under its `results`
directory until job expiry or deletion. Size `--retained-jobs` and filesystem
capacity for native GLB, optional PLY, and reference GLB output together.

Use a service manager to restart the process and set a file-descriptor limit
appropriate for concurrent uploads. Check `GET /health` after startup and
before routing traffic. Its readiness fields distinguish the native model,
multiview checkpoints, RMBG, and MoGe so a missing optional model does not hide
core inference readiness.
