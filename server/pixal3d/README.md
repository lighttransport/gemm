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
choose CPU, CUDA, or ROCm. GPU runs are serialized per backend to avoid VRAM
contention. Temporary uploads and GLBs are kept only under `tmp/pixal3d/` and
removed after each request.

`POST /v1/infer` accepts JSON fields `image_b64`, optional `mask_b64`,
`image_ext`, `backend`, `fov`, `distance`, `mesh_scale`, `seed`, `threads`,
`device`, `vram_budget_mib`, `gpu_execution` (`legacy`/`resident`), and
`gpu_kernels` (`auto`/`blas`/`mma`). Responses include a `profile` object with
phase timings and device counters. Server defaults can be selected with
`--gpu-execution resident --gpu-kernels auto`; the browser also exposes these
choices. Set `reference: true` with CUDA or ROCm to
also run the pinned upstream PyTorch pipeline; the response includes a second
GLB for comparison. This is opt-in because it loads another model stack and
requires an image without a separate mask upload. The browser displays native
AMD and PyTorch reference meshes side by side or as an opacity overlay.
`GET /health` reports binary, GPU-library, and model readiness.
