# sam-3d-body 3-pane demo server

Side-by-side compare server: input image, **ours** mesh (C runner), and
**pytorch** reference mesh — all in one HTTP response, rendered with
three.js.

## Layout

```
server/sam3d/
  app.py             ThreadingHTTPServer; /v1/infer endpoint
  pytorch_runner.py  in-process PyTorch backend (loads once at startup)
  run.sh             launcher (uses ref/sam3d-body/.venv)
web/sam3d_demo.html  3-pane web UI
```

The pytorch backend runs **in-process** (no subprocess) — the model is
loaded once at startup and served directly from the Python web server.
The ours backend shells out to `cpu/sam3d_body/test_sam3d_body` per
request (reads the `.obj` + `.obj.json` sidecar it emits).

## Run

```bash
bash server/sam3d/run.sh
# → http://localhost:8765
```

Skip the pytorch load (e.g. on machines without a GPU/venv):

```bash
bash server/sam3d/run.sh --no-pytorch
```

Override port / model paths via `--port`, `--safetensors-dir`,
`--pytorch-ckpt`, `--device`. See `python server/sam3d/app.py --help`.

## Endpoints

`GET /` — serves `web/sam3d_demo.html`
`GET /health` — `{ok, backends: {ours, pytorch}}`
`POST /v1/infer` — body:
```json
{ "image_b64": "...",
  "bbox": [x0, y0, x1, y1],   // optional; auto-bbox if omitted
  "auto_thresh": 0.5,
  "image_ext": "jpg",
  "backends": ["ours", "pytorch"] }
```
Response:
```json
{ "ours":     { "obj_b64", "json", "timing_ms", "bbox_used" } | null,
  "pytorch":  { "obj_b64", "json", "timing_ms", "bbox_used" } | null,
  "errors":   { "ours"?: str, "pytorch"?: str },
  "bbox_used": [...],
  "timings_ms": { "total": ... } }
```

The pytorch backend has no person detector — it requires a `bbox`. When
`bbox` is omitted, the server runs the ours backend first (with
auto-bbox via RT-DETR-S) and feeds the resulting bbox into the pytorch
backend so both meshes are produced for the same crop.

## Notes

- `test_sam3d_body` writes both `out.obj` and `out.obj.json` sidecar
  metadata (focal_px, cam_t, mhr_params, keypoints). Both the ours and
  pytorch backends return the same JSON shape so the web UI can render
  them uniformly.
- The web UI normalizes the mesh frame (Y-down → Y-up) and auto-frames
  the camera per pane.
