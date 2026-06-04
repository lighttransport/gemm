# rdna4/hy3d — perf bench

End-to-end perf comparison harness for the rdna4/hy3d HIP port of Hunyuan3D-2.1
shape generation against the PyTorch+ROCm reference on the same RX 9070 XT
(gfx1201).

## One-time setup

```bash
# 1. PyTorch+ROCm 7.2.2 venv with the ref deps
bash rdna4/hy3d/ref_env/setup_rocm.sh

# 2. HIP runners (HIPRTC, gcc-only)
make -C rdna4/hy3d
make -C rdna4/hy3d verify
```

## Canonical run

```bash
# 1. PyTorch ref: per-stage timing JSON + .npy traces
bash rdna4/hy3d/ref_env/run_rocm_ref.sh
#   -> rdna4/hy3d/traces/rocm_ref/*.npy
#   -> rdna4/hy3d/traces/timings_pytorch.json
#   -> rdna4/hy3d/traces/rocm_ref.glb (mesh)

# 2. Verify HIP per-stage outputs against the fresh ROCm traces
./rdna4/hy3d/verify_dinov2 \
    /mnt/disk01/models/Hunyuan3D-2.1/conditioner.safetensors \
    --ref-dir rdna4/hy3d/traces/rocm_ref/
./rdna4/hy3d/verify_dit \
    /mnt/disk01/models/Hunyuan3D-2.1/model.safetensors \
    --ref-dir rdna4/hy3d/traces/rocm_ref/
./rdna4/hy3d/verify_vae \
    /mnt/disk01/models/Hunyuan3D-2.1/vae.safetensors \
    --ref-dir rdna4/hy3d/traces/rocm_ref/

# 3. HIP e2e seeded by the PyTorch DINOv2 + step-0 latents, with timing JSON
./rdna4/hy3d/test_hip_hy3d \
    /mnt/disk01/models/Hunyuan3D-2.1/conditioner.safetensors \
    /mnt/disk01/models/Hunyuan3D-2.1/model.safetensors \
    /mnt/disk01/models/Hunyuan3D-2.1/vae.safetensors \
    --init-trace-dir rdna4/hy3d/traces/rocm_ref/ \
    --per-stage-timing rdna4/hy3d/traces/timings_hip.json \
    -o rdna4/hy3d/traces/hip_out.obj

# 4. (Optional) rocprof for HIP kernel hotspots — short 2-step run is plenty
rocprofv3 --kernel-trace --output-format json -o /tmp/hy3d_kt.json -- \
    ./rdna4/hy3d/test_hip_hy3d \
    /mnt/disk01/models/Hunyuan3D-2.1/conditioner.safetensors \
    /mnt/disk01/models/Hunyuan3D-2.1/model.safetensors \
    /mnt/disk01/models/Hunyuan3D-2.1/vae.safetensors \
    --init-trace-dir rdna4/hy3d/traces/rocm_ref/ \
    -s 2 \
    --per-stage-timing /tmp/hy3d_kt_timing.json \
    -o /tmp/hy3d_kt.obj

# 5. Side-by-side perf table (markdown)
python rdna4/hy3d/bench/compare_perf.py \
    --pytorch rdna4/hy3d/traces/timings_pytorch.json \
    --hip     rdna4/hy3d/traces/timings_hip.json \
    --rocprof /tmp/hy3d_kt.json \
    > rdna4/hy3d/bench/perf_results.md
```

## Outputs

| Path                                            | Meaning |
|-------------------------------------------------|---------|
| `traces/timings_pytorch.json`                   | dino/dit/vae/e2e ms from torch.cuda.Event |
| `traces/timings_hip.json`                       | dino/dit/vae/e2e ms from hipEvent |
| `traces/rocm_ref/*.npy`                         | per-stage tensors for verify_* and HIP `--init-trace-dir` |
| `traces/rocm_ref.glb`                           | reference mesh from PyTorch+ROCm |
| `traces/hip_out.obj`                            | mesh from HIP runner |
| `bench/perf_results.md`                         | populated comparison table |

## Notes

- HIP timing brackets the **same step body** as PyTorch (cond + uncond +
  CFG-combine + euler) — `dit_step_ms` arrays are directly comparable.
- The `dit_total_ms` figure is dominated by twice the per-step time because
  CFG runs the DiT twice per step on both backends.
- Hunyuan3D shape gen does **not** use flash_attn, so the trellis2 SDPA shim
  is not required here. If a future port pulls `flash_attn` in, mirror
  `rdna4/trellis2/flash_attn_sdpa_shim.py`.
- `verify_*` tolerances are governed by `ref/hy3d/compare.py`
  (rtol=1e-3, atol=1e-4). Numerical drift relative to the prior (CUDA-host)
  traces is expected to be small but non-zero — do not panic on tiny mismatches.
