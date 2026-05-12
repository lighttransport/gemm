# rdna4 task board

Snapshot: 2026-05-12. Branch: `trellis2`, ahead of `origin/trellis2` by 1.

## Current state

### TRELLIS.2 — texgen pipeline lands end-to-end on RX 9070 XT
- `./test_hip_trellis2 --full --mesh --tex …` produces a per-vertex-colored
  OBJ (328k verts, 397k tris, 98.5% non-trivial colors).
- shape_dec: ~3.8 s, SLAT DiT 12 steps: ~9.1 s, tex_dec (warm): 3.3 s
  (down from 124 s — 38× via MCLK fix).
- Mesh axis (X/Z) and weight-cache size landmines documented in memory.

### TRELLIS.2 — MCLK power-state fix (committed 658bd69)
- `AMD_SERIALIZE_KERNEL=3` + `HSA_ENABLE_SDMA=0` installed via `execv()`
  self-re-exec when `--tex` is set and either var is missing. HIP runtime
  caches these at `hipInit`, so `setenv()` from `main()` is too late.
- Env-gated instrumentation: `T2_TEX_PATH_DBG`, `T2_TEX_BLOCK_TIME`,
  `T2_TEX_STAGE_TIME` (no-ops when unset).
- **Validation regression (2026-05-12, 10-day uptime):** mclk-ramp lever
  became unreliable; v20–v24 all hang post-init with mclk pinned at 96 MHz
  regardless of env-var install path. v19 reached the 235 ms fast baseline
  on stage 3 ConvNeXt before stalling — best evidence we have that the fix
  code is correct. Reboot needed to re-confirm 38× wall-time.

### Other landed work this session
- `common/image_stats.h` + `common/image_diff_main.cc` (commit `8496022`):
  JSON stats / diff tool for `preview_render` AOV EXRs. LLM-readable
  comparison without VLM cost.

## Remaining tasks

### Reboot validation — done 2026-05-12 (v25)
- [x] `test_hip_trellis2 --full --mesh --tex` post-reboot: tex_dec **3203 ms**
      (v16 baseline 3264 ms, 38× confirmed). SLAT DiT 562 ms/step, shape_dec
      3622 ms, tex SLAT 338 ms/step, colored OBJ 513,716 verts.
- [x] Teardown-time `hipBLASLt err 700` at `hipblaslt.cpp:170`:
      `hipDeviceSynchronize()` at start of `hip_trellis2_free` drains
      pending kernels before hipBLASLt teardown. v27 exited 0.

### MCLK regression returns post-xmrig (2026-05-13)
- [ ] After xmrig ran 4h then was killed, MCLK lever stopped ramping again
      even though CPU is now free. Same binary, no edits — v28 reproduced
      v27's slow path (tex_dec 19 min, mclk pinned 96 MHz). Reboot before
      next benchmark. Investigate amdgpu driver state that xmrig leaves
      behind (likely some HSA queue or DMA mapping not cleaned up).

### Output divergence at slow tex_dec (2026-05-13)
- [ ] v27/v28 report `pbr_res=512, colors=(0,0,0)`; v25 reported
      `pbr_res=1, colors=(0.541,0.499,0.491)`. tex_slat denorm first8
      bit-identical → divergence is inside tex_dec. Likely the
      known Ci=64,Co=64 sparse_conv3d nondeterminism on the final stage
      — see `project_trellis2_triton_spconv_nondeterminism.md`. Bisect
      after reboot when tex_dec is reliably fast again.

### Stage 1 ConvNeXt A/B test — code ready, blocked on stable fast GPU
- [ ] `T2_TEX_BLOCK_TIME=1 T2_TEX_BLASLT_SPCONV=0` vs default to compare
      WMMA x8 path against blaslt-gather27 path for stage 1 (1103 ms
      baseline). New toggle landed in `hip_shape_dec_pipeline.c`.

### Performance follow-ups (TRELLIS.2 tex_dec)
Per-stage timings vs PyTorch reference (134 ms total for tex_dec) — we're
at 3.3 s with the MCLK fix, so ~25× gap to PyTorch remains:
- [ ] Stage 1 ConvNeXt at 1103 ms is the largest single bucket — profile
      where the time goes (likely sparse_conv3d_tiled_f32 + hash_kspconv
      dispatch overhead per block).
- [ ] Stage 2 ConvNeXt at 714 ms: similar — second-largest bucket.
- [ ] Investigate whether the Triton AOT spconv path
      (`rdna4/trellis2/triton_aot/`) can supplant the F32 LDS-tiled kernel
      for Ci=64,Co=64. Known nondeterminism (±1..±7 extremes on bulk
      ~0 values) is documented; gate under `T2_TRITON_NMAX` if it ships.

### Pipeline correctness gaps
- [ ] `stbi_zlib_compress` crash in `t2_pbr_write_textured_obj` (UV atlas +
      PNG path). Currently sidestepped via `t2_pbr_write_colored_obj`.
      Atlas path is brittle for sparse-voxel meshes; fix or remove.
- [ ] tex_dec outlier mismatch: HIP pipeline damps extremes (PyT ref has
      ±700, HIP bounded to ±1). Needs per-stage bisection with correct
      tex_dec weights. See `project_trellis2_tex_dec_outlier.md`.

### Cross-vendor parity
- [ ] ROCm-vs-CUDA bisect (2026-05-08 memo): hipBLASLt RDNA4/ROCm 7.2.2
      drops correctness past M=2^19 fp32 / 2^20 fp16,bf16 for any K,N.
      Chunked `F.linear` shim works around it; upstream issue body is at
      `rdna4/hipblaslt-issue.md`. Not yet filed.

## Other rdna4 ports — status pointers
- **rdna4/llm** Phase 5 (decode HIP graph capture) landed; gap to PyT
  perf still open. See `project_rdna4_llm_phase5.md`.
- **rdna4/hy3d** e2e on RDNA4 at 8.5 s defaults (20× over legacy f32).
  See `project_hy3d_defaults_flipped.md`.
- **rdna4/sam3** Phases 1–6 through DETR decoder; dot-score + mask
  decoder + post-process pending. See `project_sam3_hip_status.md`.
- **rdna4/qimg** 1024² at ~278 s; FP8 FA gated on alloc hoist.
  See `project_qimg_1024_perf.md`.

## Untracked artifacts (housekeeping)
Build outputs: `common/{image_diff,mesh_proc,preview_render}`,
`cpu/trellis2/test_*`. Debug npy dumps (~20 files) in repo root from
shape_dec cross-validation. `cuda/trellis2/verify-dumps/`,
`rdna4/trellis2/triton_aot/`, `.claude/` — decide which to track.
