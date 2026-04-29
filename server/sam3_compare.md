# SAM 3 / 3.1 side-by-side comparison page

`web/sam3_compare.html` fans out a single `(image, prompt)` to multiple
`/v1/infer` endpoints in parallel and renders the mask overlays + timings
in a grid, plus a pairwise union-of-masks IoU readout.

By default it talks to three servers:

| Port | Process                    | Models                 |
|------|----------------------------|------------------------|
| 8080 | C `diffusion-server`       | sam3 (cpu/cuda), sam3.1 (cpu/cuda) |
| 8082 | `ref/sam3/sam3_ref_server.py`   | sam3 (pytorch)  |
| 8081 | `ref/sam3.1/sam3_1_ref_server.py` | sam3.1 (pytorch) |

Each server emits CORS headers so the browser can hit all three from one
page. The HTML itself is served by the C server at
`http://localhost:8080/sam3_compare`.

## 1. Environment

Set `MODELS` to the directory containing the checkpoints:

```
export MODELS=/path/to/models
# expects:
#   $MODELS/sam3/sam3.model.safetensors
#   $MODELS/sam3/vocab.json
#   $MODELS/sam3/merges.txt
#   $MODELS/sam3.1/sam3.1.model.safetensors      # converted, for C runner
#   $MODELS/sam3.1/sam3.1_multiplex.pt           # original, for pytorch ref
```

If you only have `sam3.1_multiplex.pt`, convert once:

```
cd cuda/sam3.1
python convert_pt_to_safetensors.py \
    --pt  $MODELS/sam3.1/sam3.1_multiplex.pt \
    --out $MODELS/sam3.1/sam3.1.model.safetensors
```

## 2. Build the C server

```
cd server
cmake -B build -DDIFFUSION_SERVER_ENABLE_SAM3=ON \
               -DDIFFUSION_SERVER_ENABLE_SAM3_CUDA=ON
cmake --build build -j
```

## 3. Launch all three servers

Open three terminals (or use `&` + `jobs`). The compare page is fine with
any subset — unreachable backends just render an error cell.

**Terminal 1 — C server (port 8080):**

```
cd server
./build/diffusion-server --host 0.0.0.0 --port 8080 --web-root ../web \
    --sam3-ckpt     $MODELS/sam3/sam3.model.safetensors \
    --sam3-ckpt-v31 $MODELS/sam3.1/sam3.1.model.safetensors \
    --sam3-vocab    $MODELS/sam3/vocab.json \
    --sam3-merges   $MODELS/sam3/merges.txt
```

**Terminal 2 — sam3 pytorch reference (port 8082):**

```
cd ref/sam3
python sam3_ref_server.py \
    --device cuda \
    --ckpt   $MODELS/sam3/sam3.model.safetensors \
    --port   8082
```

**Terminal 3 — sam3.1 pytorch reference (port 8081):**

```
cd ref/sam3.1
python sam3_1_ref_server.py \
    --device cuda \
    --ckpt   $MODELS/sam3.1/sam3.1_multiplex.pt \
    --port   8081
```

## 4. Open the compare page

```
http://localhost:8080/sam3_compare
```

1. Upload an image (e.g. `fujisan.jpg`).
2. Enter a text prompt (e.g. `mountain`).
3. Leave default backends checked, click **Compare**.
4. Each cell fills in as its server finishes; the diff strip shows
   pairwise union-of-masks IoU across successful runs.

Backends whose `GET /v1/models` doesn't advertise the requested
`(model, backend)` are automatically greyed out on page load.

## 5. Overriding the backend list

The page ships with a built-in default list. To point at remote hosts
or trim the list, pass a base64-encoded JSON array via
`?backends=<base64>`:

```js
// in the browser console:
const cfg = [
  { label: "ours (cuda, sam3.1)",  url: "http://gpu-host:8080", model: "sam3.1", backend: "cuda" },
  { label: "pytorch-ref (sam3.1)", url: "http://gpu-host:8081", model: "sam3.1", backend: "cuda" },
];
location.href = "/sam3_compare?backends=" + btoa(JSON.stringify(cfg));
```

Each entry: `{ label, url, model, backend, placeholder? }`.
`url` is the scheme+host+port only; `/v1/infer` and `/v1/models` are
appended by the page.

## 6. Verification / troubleshooting

- `curl -X OPTIONS -H 'Origin: http://localhost:8080' -i
  http://localhost:8081/v1/infer` should return 204 with
  `access-control-allow-origin: *` and `allow-methods: POST, GET, OPTIONS`.
  Same check against port 8082 and against 8080 (the C server returns
  200 instead of 204; both are acceptable to browsers).
- `curl http://localhost:8080/v1/models` lists the C server's models;
  `curl http://localhost:8081/v1/models` and `:8082/v1/models` list the
  pytorch ref servers'.
- If a cell shows a red "error" pill with a network error, the server
  for that row is down or bound to a different host — fix it and click
  **Compare** again.
- The pytorch ref servers load weights lazily on first request; the
  initial run for each is slow (model init + cuDNN autotune).
