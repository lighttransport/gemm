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
    --steps 4 --guidance 5.0 --octree 64 --seed 42 \
    --trace-dir /tmp/hy3d_ref_trace \
    --bench-json /tmp/hy3d_ref_trace_bench.json

cd /mnt/disk1/work/gemm/main
convert /mnt/disk1/models/Hunyuan3D-2.1-repo/assets/example_images/004.png \
    /tmp/hy3d_004.ppm

rdna4/hy3d/test_hip_hy3d \
    /mnt/disk1/models/Hunyuan3D-2.1/conditioner.safetensors \
    /mnt/disk1/models/Hunyuan3D-2.1/model.safetensors \
    /mnt/disk1/models/Hunyuan3D-2.1/hunyuan3d-vae-v2-1/model.fp16.safetensors \
    -i /tmp/hy3d_004.ppm \
    --init-trace-dir /tmp/hy3d_ref_trace \
    -s 4 -g 5.0 --grid 64 --seed 42 \
    -o /tmp/hy3d_hip_trace.obj
```

The HIP runner consumes `03_dit_context_cfg.npy` and
`04_dit_latents_step0.npy` from the PyTorch trace via `--init-trace-dir`, which
removes random latent and conditioner drift from DiT/VAE debugging.

## Verified local run

2026-05-19 on RX 9070 XT / ROCm 7.2.2:

- `ref/hy3d/.venv` created with `.[torch-rocm722]`.
- venv check: `torch 2.11.0+rocm7.2.2.git4e323059`,
  `torch.version.hip=7.2.53211`, `torch.cuda.is_available()=True`.
- PyTorch 4-step trace, `octree=64`: load `46.0s`, sample `4.60s`,
  `7880` verts / `15756` faces. Outputs:
  `/tmp/hy3d_ref_trace_bench.json`, `/tmp/hy3d_ref_trace.glb`,
  `/tmp/hy3d_ref_trace.obj`, `/tmp/hy3d_ref_trace/*.npy`.
- HIP replay requires a rebuilt `rdna4/hy3d/test_hip_hy3d`; a stale binary used
  the old descending schedule and produced an empty SDF. After rebuild, replay
  starts at `t=0.0000`, completes in `205.91s`, and writes
  `/tmp/hy3d_hip_trace.obj` with `11066` verts / `15676` tris.
- HIP final latent dump `/tmp/hy3d_hip_latent_004.npy` vs PyTorch
  `07_vae_input_latents.npy`: rel-L2 `0.0596`, max abs `1.97`, mean abs
  `0.0131`.
- HIP VAE-only replay on PyTorch `07_vae_input_latents.npy` is healthy:
  `SDF min=-1.0083 max=1.0115 mean=-0.8066`.
- DiT single-forward checks against the trace pass for step 0 CFG batch 0/1
  and step 1 batch 0 (max abs <= `1.05e-2`).

That makes the next Hy3D work item numeric trajectory/mesh parity, not
environment setup or a broken VAE. The PyTorch trace is now stable enough to
use as the oracle.

Per-step drift can be summarized after a replay with latent/velocity dumps:

```sh
rdna4/hy3d/compare_trace.py \
    --ref-dir /tmp/hy3d_ref_trace \
    --hip-latent-prefix /tmp/hy3d_hip_latent \
    --hip-velocity-prefix /tmp/hy3d_hip_velocity \
    --steps 4 --guidance 5.0 \
    --json /tmp/hy3d_trace_compare.json
```

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
- Very short 1-step smoke traces can finish DiT/VAE but fail marching cubes
  with no extracted surface at `mc_level=0`. The runner writes `mesh_ok=false`
  in `--bench-json` before exiting in that case; use more steps for mesh-export
  baselines.
- `rdna4/hy3d/test_hip_hy3d` currently reads PPM input, not PNG. Convert the
  selected PNG to `/tmp/hy3d_004.ppm` before running the HIP comparison.
- The runner defaults to full-resident GPU execution. The low-VRAM/offload path
  remains documented as unreliable in the script because upstream mixes
  diffusers offload conventions with non-diffusers pipeline state.
