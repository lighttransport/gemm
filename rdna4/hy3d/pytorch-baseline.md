# Hy3D PyTorch ROCm baseline

Status: reference runner refreshed for the local `/mnt/disk1` model layout.

## Runner

Use `ref/hy3d/run_full_pipeline.py` for an end-to-end PyTorch baseline. It
loads the upstream `hy3dshape` package from the local model repo, points
`HY3DGEN_MODELS` at local weights, runs conditioner -> DiT -> VAE, writes a
mesh, and can write a small timing JSON.

```sh
cd /mnt/disk1/work/gemm/main/ref/hy3d
uv venv --python 3.12 .venv
uv pip install -e ".[torch-rocm722]"
.venv/bin/python run_full_pipeline.py \
    --image /mnt/disk1/models/Hunyuan3D-2.1-repo/assets/example_images/004.png \
    --out /tmp/hy3d_ref.glb \
    --steps 30 --guidance 5.0 --octree 256 --seed 42 \
    --bench-json /tmp/hy3d_ref_bench.json
```

Default paths now match this machine:

- `--hy3d-repo`: `/mnt/disk1/models/Hunyuan3D-2.1-repo/hy3dshape`
- `--models-root`: `/mnt/disk1/models`
- `--model-name`: `Hunyuan3D-2.1`

## HIP comparison hook

For layer-by-layer comparison against the HIP runner, emit a trace directory:

```sh
cd /mnt/disk1/work/gemm/main/ref/hy3d
.venv/bin/python run_full_pipeline.py \
    --image /mnt/disk1/models/Hunyuan3D-2.1-repo/assets/example_images/004.png \
    --out /tmp/hy3d_ref_trace.glb \
    --steps 1 --guidance 5.0 --octree 64 --seed 42 \
    --trace-dir /tmp/hy3d_ref_trace \
    --bench-json /tmp/hy3d_ref_trace_bench.json

cd /mnt/disk1/work/gemm/main
rdna4/hy3d/test_hip_hy3d \
    /mnt/disk1/models/Hunyuan3D-2.1/conditioner.safetensors \
    /mnt/disk1/models/Hunyuan3D-2.1/model.safetensors \
    /mnt/disk1/models/Hunyuan3D-2.1/hunyuan3d-vae-v2-1/model.fp16.safetensors \
    -i /mnt/disk1/models/Hunyuan3D-2.1-repo/assets/example_images/004.png \
    --init-trace-dir /tmp/hy3d_ref_trace \
    -s 1 -g 5.0 --grid 64 --seed 42 \
    -o /tmp/hy3d_hip_trace.obj
```

The HIP runner consumes `03_dit_context_cfg.npy` and
`04_dit_latents_step0.npy` from the PyTorch trace via `--init-trace-dir`, which
removes random latent and conditioner drift from DiT/VAE debugging.

## Current caveats

- This checkout has `ref/hy3d/pyproject.toml`, but no checked-in lockfile or
  local `.venv`. The `torch-rocm722` extra pins the same ROCm 7.2.2 PyTorch
  wheel family used by the Trellis2 RDNA4 baseline venv.
- Plain `uv run` against this `pyproject.toml` resolves PyPI default CUDA
  wheels here (`torch 2.12.0+cu130`, `torch.version.hip=None`,
  `cuda.is_available=False`), so it is not a valid ROCm baseline environment.
  Use `.venv/bin/python` after installing `.[torch-rocm722]` before taking
  baseline numbers. The runner now fails fast for the default `--device cuda`
  path when `torch.version.hip` is absent.
- The runner defaults to full-resident GPU execution. The low-VRAM/offload path
  remains documented as unreliable in the script because upstream mixes
  diffusers offload conventions with non-diffusers pipeline state.
