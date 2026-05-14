# rdna4 task board

Snapshot: 2026-05-14. Branch: `trellis2`.

## Current state

### TRELLIS.2 — texgen pipeline WORKS end-to-end on RX 9070 XT (fast path)
- `./test_hip_trellis2 --full --mesh --tex …` produces a per-vertex-colored
  OBJ (**826,884 verts, 77,441 unique RGB triples** after nmap fast-path).
- shape_dec: ~3.8 s, SLAT DiT 12 steps: ~9.1 s, tex_dec: **0.43 s**
  (was 3.1 s after the correctness fix alone — 7.3× cumulative).
- Fast-path coord-corruption root cause: persistent `g_c2s` scratch with
  stale hash entries across calls. Fixed by per-call `c2s_free_all` at
  end of `hip_shape_dec_forward_ex` (commit `5ee328e`).
- Perf: build nmap on the fly via `t2_build_nmap_f32` so Triton/WMMA
  spconv paths fire instead of hash_kspconv. Both convnext coarse
  (commit `58541f6`) and c2s conv2 fine (commit `329765e`).
  Gap to PyT reference: 23× → **3.2×** (PyT tex_dec ~134 ms).
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

### Fast-path tex_dec corruption FIXED (commit 5ee328e, 2026-05-14)
- [x] Root cause: `g_c2s` (C2S persistent scratch with dn/de/dhf/dxf/
      dhn/didx/dsi/dout/fk/fv) was file-scope and never freed. Its hash
      table (fk/fv) carried entries from a prior shape_dec invocation.
      On fast-path tex_dec, stale entries caused t2_hash_lookup to
      return voxel indices larger than current Nf → kspconv OOB reads,
      visible in dmesg as TCP-client write faults at user-virtual
      addresses (`0x7ae6_xxxxx`). Slow path was incidentally correct
      because hipMemsetAsync(fk,0) had time to settle before
      hash_build ran.
- [x] Fix: `c2s_free_all(&g_c2s)` at end of every
      `hip_shape_dec_forward_ex`. Pre-size block at function entry
      re-allocates fresh.
- [x] v55 validated: fast path 3.1 s tex_dec, voxel_res=512,
      **30,296 unique RGB triples** across 513,716 vertices.
      EXIT=0, no GPU page faults.

### tex_dec post-first-run degradation — RESOLVED by fix above
- [x] The "first run fast, subsequent runs slow" pattern was caused by
      the same stale g_c2s state. With per-call scratch, subsequent
      runs should also stay fast. (Re-verify across multiple back-to-
      back e2e --tex runs to confirm.)

### PBR hash axis fix landed — v29 produces real per-vertex texturing
- [x] Root cause: `t2_pbr_from_decoder` stored col1 (X) at hash axis0
      while sampler hashed `(vz, vy, vx)` → axis0 is Z. X/Z swapped
      between storage and lookup. Fix in `common/trellis2_pbr.h`:
      assign col3→axis0, col1→axis2.
- [x] v29 validated: 26,698 unique RGBs vs v25's 1. EXIT=0.

### Subdiv coord free-before-sync race fixed (commit 9817f66)
- [x] `hip_shape_dec_forward_ex` queued an async D2D memcpy of the
      final coords on g_stream then `hipFree(d_gxc[s])` synchronously
      before draining → caller read zeros → tex_coords collapse to
      (0,0,0) → PBR flat color. Fix: hoist `hipStreamSynchronize` to
      before the hipFree loop. v25 and v30 both hit it; v29 won the
      race by luck.

### Sparse coverage gap — bisected to stage-0 to_subdiv
- [ ] HIP shape_dec stage-0 keeps only 53% of children reference keeps
      (×2.52 vs ref ×4.78). Stage 2 also under (×3.04 vs ×4.47). Stages
      1, 3 within ±10%. Hypothesis: `to_subdiv` logits at C=1024 are
      damped → fewer pass the `logits > 0` threshold. Likely an
      upstream feature-magnitude clamp in the ConvNeXt path (matches
      tex_dec outlier memo: HIP bounded to ±1, ref ±700). Next:
      capture stage-0 host logits, compare distributions vs reference.

### MCLK lever — likely also resolved by per-call scratch fix
- [ ] Re-verify whether the "back-to-back runs go slow" symptom was
      caused by stale g_c2s state (the fix above), or whether the
      mclk lever itself is still unreliable. Test: run e2e --tex 3×
      back-to-back in one session with commit 5ee328e and check
      whether all three stay at 3.1 s tex_dec.

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
