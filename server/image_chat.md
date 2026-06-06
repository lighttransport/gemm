# Image chat — Qwen3.6-VL editor + OCR helper lines

`web/image_chat.html` is a single-backend VLM chat page with an image
editor: pan / zoom / rotate 90° / region select / horizontal+vertical
guide lines. Each turn re-rasterizes the edited view (optionally cropped
to the region, optionally with guide lines burned in), downscales to
max side 1024, and ships the PNG + prompt to the first available Qwen
VLM backend via `/v1/chat`.

No compare grid, no pytorch reference required, no new endpoint — the
page rides on the same `/v1/chat` as `web/llm_chat.html`.

## 1. Requirements

- A Qwen-family VLM variant (`family=qwen`, `kind=vlm`) registered in
  the `VARIANTS` spec for `run-llm-compare.sh`. Qwen3.6-27B with the
  matching `mmproj-F16.gguf` is the intended target.
- One of the Qwen VLM binaries built and on disk:
  - `cpu/vlm/test_vision`            (CPU — `make -C cpu/vlm test_vision`)
  - `cuda/vlm/test_cuda_qwen_vlm`    (CUDA, optional, advertised if present)

The page auto-picks the fastest available backend in this order:
`ours-cuda-qwen-vlm > ours-cpu-qwen-vlm > pytorch-cuda (iff variant has an HF path)`.

## 2. Launch the sidecar

Same launcher as the LLM/VLM compare page. Only the Qwen VLM entry is
needed for image-chat; other entries are ignored by this page but are
harmless.

```
VARIANTS="qwen36-27b=vlm:/mnt/disk01/models/qwen36/27b/Qwen3.6-27B-UD-Q2_K_XL.gguf:/mnt/disk01/models/qwen36/27b/mmproj-F16.gguf:unsloth/Qwen3.6-27B" \
    ./server/run-llm-compare.sh
```

Listens on `http://0.0.0.0:8089` by default (`PORT=...` to override).

Verify the variant / backends:

```
curl -s http://localhost:8089/v1/models | python3 -m json.tool | head -40
```

Expect `"backends"` to include at least `ours-cpu-qwen-vlm`, and
`"variants"` to include your `qwen36-27b` entry with `"family": "qwen"`,
`"kind": "vlm"`.

## 3. Open the page

Either origin works — the C server serves the HTML, the JS talks to the
Python sidecar at `:8089`:

```
http://localhost:8089/image_chat     # served by the Python sidecar directly
http://localhost:8080/image_chat     # served by the C diffusion-server
```

Aliases that also resolve to the same page: `/image-chat`, `/image-explain`.

## 4. Usage

1. **Open image** — PNG / JPG / WebP (anything the browser decodes).
2. **Frame it** — drag to pan, wheel or `+` / `−` to zoom, `Fit` to
   reset, `↺ 90°` / `↻ 90°` to rotate (region and guide lines rotate
   with the image so overlays stay pinned to content).
3. **(Optional) Select a region** — click **Select region**, then drag
   on the canvas. Switch the radio to *selected region* so the send
   crops to that rectangle. `✕ region` clears the rect.
4. **(Optional) Add OCR guide lines** — click **+ H-line** / **+ V-line**,
   then click on the image to place a line. Lines are drawn over the
   image and (when `lines` is checked) are **burned into the PNG** sent
   to the VLM — useful for nudging the model's attention along a row
   of text or separating columns. `✕ lines` removes all guides.
5. **Ask** — type a prompt and hit **Send**. The assistant reply
   streams back as a new turn with a thumbnail of exactly what was
   sent to the model.
6. **Chat** — with `ctx` checked, prior turns are concatenated into the
   next prompt (the binary is single-shot stateless, so context is
   client-side). `New chat` clears history.
7. **Cancel** — aborts the in-flight HTTP request. The backend process
   itself keeps running to completion (subprocess-per-call), so the
   token meter may be misleading for a while after cancel.

Each send shows exactly `W×H PNG` in the status hint so you can tell
how the `MAX_SEND=1024` downscale rounded your crop.

## 5. Tuning

- `max tok` (default 192) — cap generation length; raise for long OCR
  passes, lower for one-word answers.
- `?origin=http://host:port` in the URL — point the page at a sidecar
  running on a different host (useful when the C server is local but
  the GPU box is remote).
- The page never sends `system` or `seed` — if you need those, drop
  into `web/llm_chat.html` instead.

## 6. Troubleshooting

- **`(no qwen VLM)` pill** — no variant with `family=qwen` in
  `VARIANTS`. Check the spec; family is sniffed from the GGUF header
  (see `_detect_family()` in `ref/llm/llm_server.py`), or force it with
  a `:family=qwen` suffix on the variant.
- **`(no backend)` pill** — variant is registered but none of the Qwen
  VLM binaries are present and the variant has no `hf_path`. Build one
  (`make -C cpu/vlm test_vision`) or set an HF path in the variant.
- **Image sent but model ignores it** — the Qwen binary rejects input
  images whose dims aren't multiples of `patch_size * spatial_merge`.
  The page already snaps to the grid via its 1024-max downscale, but
  if you see `Image size N must be multiple of M` in the sidecar log,
  check that the cropped region isn't degenerate.
- **Slow first turn on `ours-cpu-qwen-vlm`** — the CPU runner spends
  most of its time in the LLM decode, not the vision encoder; expect
  tens of seconds per turn for 27B-Q2. Use the CUDA binary when
  available.
