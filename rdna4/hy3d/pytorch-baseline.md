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
- HIP replay with fp16-rounded MoE gate logits completes in `207.92s` and
  writes `/tmp/hy3d_hip_trace.obj` with `11130` verts / `15764` tris.
- HIP final latent dump `/tmp/hy3d_hip_latent_004.npy` vs PyTorch
  `07_vae_input_latents.npy`: rel-L2 `0.0593`, max abs `1.94`, mean abs
  `0.0130`.
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

Latest 4-step replay summary:

```text
Latents
step  max_abs      mean_abs     rel_l2
   1           0           0           0
   2  0.00683594  0.000672213  0.00135945
   3   0.0446777  0.00146078  0.00393528
   4     1.94385   0.0130082   0.0592722

Velocities vs PyTorch CFG
step  max_abs      mean_abs     rel_l2
   1   0.0186566  0.00196431  0.00255842
   2    0.133075  0.00398252   0.0122257
   3     5.83546   0.0376467    0.231608
   4     6.29063   0.0319469    0.134516
```

## Drift diagnosis

The large late-step drift is trajectory sensitivity, not a broken HIP DiT
forward path. `ref/hy3d/dump_dit_single_step.py --all-batches` can replay a
single DiT call on CPU with float activations using the same latent/context
inputs:

```sh
ref/hy3d/.venv/bin/python ref/hy3d/dump_dit_single_step.py \
    --ckpt /mnt/disk1/models/Hunyuan3D-2.1/hunyuan3d-dit-v2-1/model.fp16.ckpt \
    --hy3d-repo /mnt/disk1/models/Hunyuan3D-2.1-repo/hy3dshape \
    --latents /tmp/hy3d_hip_latent_003.npy \
    --context /tmp/hy3d_ref_trace/03_dit_context_cfg.npy \
    --timestep 0.66650390625 \
    --all-batches \
    --outdir /tmp/hy3d_pt_hip_step3
```

The same script can run a single-step ROCm fp16 oracle with block dumps:

```sh
ref/hy3d/.venv/bin/python ref/hy3d/dump_dit_single_step.py \
    --ckpt /mnt/disk1/models/Hunyuan3D-2.1/hunyuan3d-dit-v2-1/model.fp16.ckpt \
    --hy3d-repo /mnt/disk1/models/Hunyuan3D-2.1-repo/hy3dshape \
    --latents /tmp/hy3d_hip_latent_003.npy \
    --context /tmp/hy3d_ref_trace/03_dit_context_cfg.npy \
    --timestep 0.66650390625 \
    --all-batches \
    --device cuda --dtype fp16 \
    --outdir /tmp/hy3d_pt_rocm_hip_step3
```

On the HIP step-3 latent, HIP velocity vs CPU-float PyTorch CFG is close:
max abs `0.0752`, mean abs `3.9e-5`, rel-L2 `8.6e-4`. The CPU-float PyTorch
output on that same HIP latent has the same large gap to the original ROCm
fp16 trace as HIP does: rel-L2 `0.233`. That means the step-3 mismatch is
caused by small earlier latent differences crossing a sensitive DiT/MoE
trajectory boundary.

Against ROCm fp16 PyTorch on the same HIP step-3 latent, default HIP has a
smaller but measurable single-step CFG gap: rel-L2 `0.0235`. The gap is small
per batch (`0.0060` cond, `0.0048` uncond) but guidance amplifies it.

Block-level step-0 localization tells the same story. With exact trace input,
HIP block outputs match the CPU-float oracle tightly through the full DiT:

```text
block       max_abs      mean_abs       rel_l2
0          0.00361      1.49e-05      1.47e-05
5          0.00372      3.53e-05      1.72e-05
10         0.00270      3.95e-05      1.91e-05
11         0.00098      8.57e-06      2.37e-05
14         0.00810      2.34e-05      5.99e-05
15         0.02193      1.20e-04      4.53e-04
20         0.01686      1.74e-05      6.01e-05
```

An attempted broad "round every major activation through fp16" experiment did
not solve the trace drift: it slightly improved step 3 but worsened step 4
velocity rel-L2 (`0.164` vs `0.135`). A narrower MoE-only fp16 emulation path is
available with `HY3D_MOE_FP16=1`; it rounds MoE softmax weights, expert
activations, expert outputs, and expert weighted accumulation through fp16. It
improves the same-latent step-3 ROCm fp16 CFG rel-L2 from `0.0235` to `0.0227`.

Full 4-step replay with `HY3D_MOE_FP16=1`:

```text
Latents
step  max_abs      mean_abs     rel_l2
   1           0           0           0
   2  0.00585938  0.00066545   0.0013448
   3    0.045166  0.00136097   0.0036736
   4     1.83936    0.012551   0.0570056

Velocities vs PyTorch CFG
step  max_abs      mean_abs     rel_l2
   1   0.0184071  0.00194399  0.00252906
   2    0.134296  0.00365849   0.0113009
   3     5.52042   0.0364414    0.222875
   4     6.53329   0.0319188    0.130072
```

This improves the final latent rel-L2 from `0.0593` to `0.0570`, but the
current CPU-side fp16 accumulation emulation increased 4-step runtime from
`207.92s` to `223.23s`. Treat it as a parity mode until the MoE accumulation is
moved back to a GPU kernel.

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
