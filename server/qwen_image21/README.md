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
  side-by-side.

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

The API accepts `backend=cuda|rocm` and `mode=native|reference|compare`.
Legacy `mode=cuda` and `mode=rocm` requests remain accepted as native runs.
`GET /api/health` reports readiness independently for both backends.

Override paths when needed:

```sh
server/qwen_image21/run.sh \
  --model /mnt/nvme01/models/qimg-21 \
  --quant-package tmp/qimg21-int8-package
```

The browser is at `http://127.0.0.1:8091/`; the JSON endpoints are
`GET /api/health` and `POST /api/generate`.
