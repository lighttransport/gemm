# Qwen Image 2.1 web demo

This is a new standalone demo; it does not reuse the older Qwen Image web
application. It exposes three run modes:

- `CUDA` runs `cuda/qimg21/native_generate.py` with the native custom CUDA
  runner and native VAE decode.
- `PyTorch` runs the pinned Diffusers reference script.
- `Compare` runs both sequentially under one device lock and renders them
  side-by-side.

The quantization switch applies the exported row-INT8 package to the CUDA
runner, enables custom INT8 tensor-core GEMMs, and uses the calibrated
16-block BF16 tail for true-CFG quality. The PyTorch reference remains
unquantized so it stays an arithmetic reference.

Start it after building the native binaries:

```sh
make -C cuda/qimg21 native
server/qwen_image21/run.sh --host 127.0.0.1 --port 8091
```

Override paths when needed:

```sh
server/qwen_image21/run.sh \
  --model /mnt/nvme01/models/qimg-21 \
  --quant-package tmp/qimg21-int8-package
```

The browser is at `http://127.0.0.1:8091/`; the JSON endpoints are
`GET /api/health` and `POST /api/generate`.
