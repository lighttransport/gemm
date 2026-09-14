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
`device`, and `vram_budget_mib`. The response contains `glb_b64` and native
runner `stats`. `GET /health` reports binary, GPU-library, and model readiness.
