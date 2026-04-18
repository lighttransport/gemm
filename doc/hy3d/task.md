# Hy3D CUDA Verification Task

## Current Status

- Local PyTorch trajectory traces have been regenerated at:
  - `ref/hy3d/local_trace_demo_seed42/`
- Matching CUDA step dumps for the same run live at:
  - `ref/hy3d/local_trace_demo_seed42/cuda_steps/`
- The block-0 cross-attention K/V debug path is now trustworthy:
  - conditional K/V dumps are nonzero
  - host fallback K/V and CUDA K/V match exactly for the conditional pass
- The CUDA cross-attention K/V layout bug has now been fixed in the live DiT path:
  - CUDA now mirrors PyTorch `CrossAttention` by rebuilding the interleaved
    `KV` head layout before `k_norm` and SDPA
  - block-0 `attn2.k_norm` now matches the PyTorch step-0 reference almost
    exactly
- Full parity is still not recovered:
  - `compare_trajectory.py` fails for both latent and velocity spot-checks at
    steps `1,15,30`
  - `verify_suite.sh --profile f16` still fails on `dit_output.npy`
- PyTorch reference tracing and CUDA dump plumbing are in place:
  - `ref/hy3d/dump_dit_single_step.py`
  - `ref/hy3d/run_full_pipeline.py`
  - `ref/hy3d/compare_trajectory.py`
  - CUDA latent / velocity dump hooks in `test_cuda_hy3d`

## Verified Findings

- DINOv2 context override is correct:
  - fresh CUDA `cuda_dino_context_after_override.npy` matches PyTorch
    `03_dit_context_cfg.npy[0]` exactly.
- DiT block 0 query path is correct:
  - fresh CUDA `cuda_b0_attn2_to_q_raw_cond.npy` matches the PyTorch block-0
    query path closely.
- The earlier all-zero K/V diagnosis was a debug artifact, not proven broken
  rectangular GEMM math:
  - CFG runs `run_dit_forward()` twice; unconditional zero-context dumps were
    overwriting the conditional pass dumps.
  - the host-side K/V fallback was previously reading F16 device weights as if
    they were F32.
- The CUDA debug labels for `attn2.out` / `attn2.out_proj` have now been fixed.
- With the fixed debug path:
  - `cpu_b0_attn2_to_k_cond.npy` and `cuda_b0_attn2_to_k_cond.npy` match exactly
  - `cpu_b0_attn2_to_v_cond.npy` and `cuda_b0_attn2_to_v_cond.npy` match exactly
- The remaining block-0 cross-attention mismatch was a real K/V layout bug, not
  an RMSNorm math bug:
  - upstream `CrossAttention` does `kv = cat(k, v) -> view(..., heads, 2*hd) ->
    split(hd, dim=-1)` before applying `k_norm`
  - CUDA had been normalizing plain `to_k(y)` directly
  - after matching the PyTorch KV interleave, fresh step-0 comparisons improved to:
    - `dit_b0_attn2_k_norm.npy` vs `cuda_b0_attn2_k_norm_cond.npy`:
      `max=5.72e-06 mean=1.62e-07`
    - `dit_b0_attn2_out_proj.npy` vs `cuda_b0_attn2_out_proj_cond.npy`:
      `max=5.85e-05 mean=1.32e-06`
    - `dit_block_0.npy` vs `cuda_dit_block_0_cond.npy`:
      `max=3.56e-03 mean=1.47e-05`
- The raw `attn2.out` dump still looks mismatched, but that now appears to be a
  debug-layout interpretation issue because the projected output and block output
  already match closely.

## Remaining Problem

- The remaining blocker is now broader DiT / trajectory mismatch, not just
  block-0 K/V dumping:
  - regenerated local trajectory traces and CUDA step dumps disagree at steps
    `1,15,30`
  - `verify_suite.sh --profile f16` still fails manifest thresholds on
    `dit_output.npy`
- The new split in behavior is:
  - focused one-step override run improves substantially after the KV-layout fix
  - unconditioned full-pipeline trajectory spot-checks still fail
  - the live non-override path still starts from the wrong latent tensor

## Working Hypothesis

- The block-0 K/V projection path itself is likely no longer the primary issue.
- The next failure surface is more likely in one of:
  - initial latent RNG parity in the non-override path
  - end-to-end DiT trajectory logic outside the fixed K/V layout bug
  - scheduler / guidance / preprocessing mismatch between PyTorch and CUDA
  - a broader DiT forward mismatch that already shows up in `dit_output.npy`

## Completed Debug Steps

- Added raw Q dump:
  - `cuda_b0_attn2_to_q_raw_cond.npy`
- Added and fixed host-side K/V fallback dumps:
  - `cpu_b0_attn2_to_k_cond.npy`
  - `cpu_b0_attn2_to_v_cond.npy`
- Tagged DiT debug dumps by pass (`cond` / `uncond` / `single`) so conditional
  traces are not overwritten by the unconditional CFG pass.
- Regenerated local PyTorch full-pipeline trajectory traces:
  - `ref/hy3d/local_trace_demo_seed42/`
- Regenerated matching CUDA latent / velocity step dumps:
  - `ref/hy3d/local_trace_demo_seed42/cuda_steps/`
- Regenerated post-fix CUDA latent / velocity spot-check dumps:
  - `ref/hy3d/local_trace_demo_seed42/cuda_steps_fix/`
- Confirmed the live forward pass still produces nonzero block outputs after the
  cross-attn step, and the K/V tensors are now nonzero too.
- Re-ran a focused one-step override repro with the local PyTorch step-0
  latents/context after the cross-attention fix:
  - step-1 velocity mismatch improved from roughly
    `max=6.49e+00 mean=1.13e+00`
    to `max=7.74e-01 mean=1.18e-01`
- Re-ran the 30-step spot-check after the fix:
  - latent:
    - step 1: `max=6.49e+00 mean=1.14e+00`
    - step 15: `max=1.59e+01 mean=5.71e-01`
    - step 30: `max=7.49e+00 mean=1.13e+00`
  - velocity:
    - step 1: `max=6.49e+00 mean=1.13e+00`
    - step 15: `max=4.71e+00 mean=7.86e-01`
    - step 30: `max=6.16e+00 mean=1.10e+00`
  - so the fix clearly helps the controlled override path, but it does not yet
    resolve the normal 30-step trajectory mismatch.
- Confirmed a new first-failure point for the ordinary 30-step path:
  - `dit_latent_step_001.npy` from the live CUDA run differs from the PyTorch
    `05_dit_input_x_000.npy` reference by `max=6.49e+00 mean=1.13e+00`
  - this dump is the pre-step latent state, so the normal pipeline is still
    starting from the wrong noise tensor
  - current CUDA uses `cuda_rng_fill_normal_f32(...)`, which is a shared
    Philox + Box-Muller helper, while the PyTorch reference uses
    `torch.Generator(device='cuda').manual_seed(seed)` through diffusers
    `randn_tensor(...)`
- Added a new verification convenience flag to `test_cuda_hy3d`:
  - `--init-trace-dir <trace_dir>`
  - it automatically loads:
    - `04_dit_latents_step0.npy`
    - `03_dit_context_cfg.npy`
  - with this mode, the CUDA step-1 latent matches
    `05_dit_input_x_000.npy` exactly (`max=0 mean=0`)
  - the matching step-1 velocity remains much closer than before:
    `max=7.74e-01 mean=1.18e-01`

## Regeneration Commands

```bash
cd ref/hy3d
HY3D_REPO=/mnt/disk01/models/Hunyuan3D-2.1-repo/hy3dshape \
HY3DGEN_MODELS=/mnt/disk01/models \
HY3D_MODEL_NAME=Hunyuan3D-2.1 \
.venv/bin/python run_full_pipeline.py \
  --image /mnt/disk01/models/Hunyuan3D-2.1-repo/hy3dshape/demos/demo.png \
  --out /home/syoyo/work/gemm/diffusion/ref/hy3d/local_trace_demo_seed42/hy3d_ref.glb \
  --trace-dir /home/syoyo/work/gemm/diffusion/ref/hy3d/local_trace_demo_seed42 \
  --steps 30 --guidance 7.5 --octree 256 --seed 42 --dtype fp16

python3 - <<'PY'
from PIL import Image
img = Image.open('/mnt/disk01/models/Hunyuan3D-2.1-repo/hy3dshape/demos/demo.png').convert('RGBA')
bg = Image.new('RGBA', img.size, (255, 255, 255, 255))
Image.alpha_composite(bg, img).convert('RGB').save('/tmp/hy3d_demo_seed42.ppm')
PY

cd ../../cuda/hy3d
./test_cuda_hy3d \
  /mnt/disk01/models/Hunyuan3D-2.1/conditioner.safetensors \
  /mnt/disk01/models/Hunyuan3D-2.1/model.safetensors \
  /mnt/disk01/models/Hunyuan3D-2.1/vae.safetensors \
  -i /tmp/hy3d_demo_seed42.ppm -o /tmp/hy3d_demo_seed42_cuda.obj \
  -s 30 -g 7.5 --seed 42 \
  --dump-latent-steps 1,15,30 \
  --dump-latent-prefix /home/syoyo/work/gemm/diffusion/ref/hy3d/local_trace_demo_seed42/cuda_steps/dit_latent_step \
  --dump-velocity-steps 1,15,30 \
  --dump-velocity-prefix /home/syoyo/work/gemm/diffusion/ref/hy3d/local_trace_demo_seed42/cuda_steps/dit_velocity_step

cd ../../ref/hy3d
.venv/bin/python compare_trajectory.py \
  /home/syoyo/work/gemm/diffusion/ref/hy3d/local_trace_demo_seed42 \
  /home/syoyo/work/gemm/diffusion/ref/hy3d/local_trace_demo_seed42/cuda_steps \
  --steps 1,15,30 --cuda-prefix dit_latent_step --kind latent

.venv/bin/python compare_trajectory.py \
  /home/syoyo/work/gemm/diffusion/ref/hy3d/local_trace_demo_seed42 \
  /home/syoyo/work/gemm/diffusion/ref/hy3d/local_trace_demo_seed42/cuda_steps \
  --steps 1,15,30 --cuda-prefix dit_velocity_step --kind velocity

# Deterministic DiT trajectory parity against the regenerated local trace
cd ../../cuda/hy3d
./test_cuda_hy3d \
  /mnt/disk01/models/Hunyuan3D-2.1/conditioner.safetensors \
  /mnt/disk01/models/Hunyuan3D-2.1/model.safetensors \
  /mnt/disk01/models/Hunyuan3D-2.1/vae.safetensors \
  -i /tmp/hy3d_demo_seed42.ppm -o /tmp/hy3d_demo_seed42_trace_override.obj \
  -s 30 -g 7.5 --seed 42 \
  --init-trace-dir /home/syoyo/work/gemm/diffusion/ref/hy3d/local_trace_demo_seed42 \
  --dump-latent-steps 1,15,30 \
  --dump-latent-prefix /home/syoyo/work/gemm/diffusion/ref/hy3d/local_trace_demo_seed42/cuda_steps_trace_override/dit_latent_step \
  --dump-velocity-steps 1,15,30 \
  --dump-velocity-prefix /home/syoyo/work/gemm/diffusion/ref/hy3d/local_trace_demo_seed42/cuda_steps_trace_override/dit_velocity_step
```

## TODO

1. Localize why regenerated trajectory traces still fail:
    - latent compare:
      - pre-fix `cuda_steps/`:
        - step 1: `max=5.99e+00 mean=9.75e-01`
        - step 15: `max=1.45e+01 mean=1.93e+00`
        - step 30: `max=8.04e+00 mean=1.43e+00`
      - post-fix `cuda_steps_fix/`:
        - step 1: `max=6.49e+00 mean=1.14e+00`
        - step 15: `max=1.59e+01 mean=5.71e-01`
        - step 30: `max=7.49e+00 mean=1.13e+00`
    - velocity compare:
      - pre-fix `cuda_steps/`:
        - step 1: `max=6.49e+00 mean=1.13e+00`
        - step 15: `max=4.46e+00 mean=7.18e-01`
        - step 30: `max=6.40e+00 mean=9.53e-01`
      - post-fix `cuda_steps_fix/`:
        - step 1: `max=6.49e+00 mean=1.13e+00`
        - step 15: `max=4.71e+00 mean=7.86e-01`
        - step 30: `max=6.16e+00 mean=1.10e+00`
    - first narrow follow-up:
      - explain the initial latent mismatch in the non-override path
      - decide whether CUDA should match PyTorch CUDA RNG exactly, or whether
        trajectory verification should continue using `--init-trace-dir`
2. Localize the remaining `dit_output.npy` stage mismatch in `verify_suite.sh`:
    - current `f16` compare summary:
      - `max_abs = 1.1926474571228027`
      - `mean_abs = 0.0381041020154953`
3. Keep using the regenerated local trace directory above as the working
   reference path until a newer trace replaces it.

## Notes

- The old “all-zero K/V dump” story is stale; use the regenerated conditional
  dump files under `ref/hy3d/local_trace_demo_seed42/` instead.
- The current goal is still reference parity, but the next step is now to
  explain why the controlled override path improved while the ordinary 30-step
  pipeline still starts from mismatched noise and `verify_dit` output still
  diverges.
