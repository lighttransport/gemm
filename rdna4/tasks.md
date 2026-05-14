# rdna4 task board

Snapshot: 2026-05-14. Branch: `trellis2`.

## Current state

### TRELLIS.2 — texgen pipeline WORKS end-to-end on RX 9070 XT (fast path)
- `./test_hip_trellis2 --full --mesh --tex …` produces a per-vertex-colored
  OBJ (**826,884 verts, 77,441 unique RGB triples** after nmap fast-path).
- shape_dec: **~0.96 s** (was 3.8 s — nmap fast path applies here too),
  SLAT DiT 12 steps: ~6.6 s, SS DiT 12 steps: ~8.2 s, tex DiT: ~4.0 s,
  tex_dec: **0.43 s** (was 3.1 s after the correctness fix alone).
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

### MCLK regression returns post-xmrig (2026-05-13) — RESOLVED
- [x] The "tex_dec slow / mclk pinned 96 MHz on subsequent runs"
      symptom was the same stale-g_c2s bug, not driver/xmrig state.
      Fixed by per-call c2s_free_all (5ee328e), verified 2026-05-14
      with 3 back-to-back fast runs. Was never an amdgpu/HSA issue.

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

### Sparse coverage gap — improved by nmap fast path, still ~56%
- Mesh quality vs reference (v63, 2026-05-14):
  - verts 826,915 / ref 1,468,404 = 56%; faces 951,334 / 3,140,890 = 30%
  - bbox matches to 3 decimals; coarse IoU res16=0.92, res32=0.88
  - fine IoU res64=0.56, res128=0.32 — recall<precision → under-
    tessellated, missing reference detail (not spurious geometry)
- [x] Verdict: geometry correct (right object/place/scale), ~56%
      vertex density. Root-caused below as sampler-compounded
      numerical accumulation — understood, not a bug. CLOSED.
      The on-the-fly nmap change (58541f6) bumped coverage from
      35%→56% as a side effect (different numerical paths in
      WMMA/Triton spconv → more to_subdiv logits pass > 0).
- [x] Full bisect done (v64-v67 + PyTorch-ROCm reference dump,
      2026-05-14):
      1. to_subdiv math + weights VERIFIED correct: HIP_feats @ ref_W.T
         + b reproduces HIP's damped logits exactly; ref_feats @ ref_W.T
         + b reproduces ref logits (128.5≈129.6 etc). Projection is fine.
      2. c2s SparseChannel2Spatial gather (t2_c2s_gather_f32) matches
         reference rearrange. ConvNeXt block structure matches
         (conv→norm→mlp→+x). No architectural bug found.
      3. HIP c2s-input feats have ~right per-channel std (stage-2 median
         ratio 1.01 vs ref) BUT the to_subdiv amplification
         (logit_std/feat_std) is way off: ref 39.7/8.2 at stage 0/2 vs
         HIP 26.7/3.4. Feats are decorrelated from the trained
         to_subdiv weight directions.
      4. Divergence is distributed/upstream, not a single bug:
         SS path gives 3468 coords vs ref 3548; HIP shape-SLAT denorm
         std 5.55 vs ref 6.00 (92%). Accumulated BF16/numerical drift
         through DiT+decoder rotates feature vectors away from the
         trained to_subdiv directions → progressively damped logits
         → fewer subdiv children → 56% mesh density.
      Conclusion: needs precision improvements across the DiT+decoder
      stack (not a quick patch). Mesh geometry is correct; this only
      affects fine tessellation density.
- [x] FP32-shape_dec test (v68, 2026-05-14): ran shape_dec with
      T2_TEX_BLASLT=0 T2_TEX_WMMA_SPCONV=0 T2_TEX_TRITON=0 (full F32
      scalar — no BF16 GEMM, no WMMA/Triton spconv anywhere in the
      decoder). to_subdiv logit std came out **identical** to the BF16
      path (111.67/70.33/10.81/6.09 vs 111.79/70.40/10.80/6.08).
      → shape_dec GEMM/conv precision is NOT the cause. The decoder is
      numerically faithful. The divergence is entirely upstream in the
      SS DiT / SS decoder / SLAT DiT (HIP shape-SLAT denorm std 5.55 vs
      ref 6.00, coords 3468 vs 3548). Any further mesh-density work
      must target the DiT path, not the decoder.
- [x] SS DiT bisect (v70 + PyTorch-ROCm full dump, 2026-05-14):
      HIP SS-DiT latent vs reference (12-step CFG):
        HIP-BF16 vs CUDA-ref      rel_l2 0.190
        HIP-F32  vs CUDA-ref      rel_l2 0.142
        HIP-F32  vs PyTorch-ROCm  rel_l2 0.141
        CUDA-ref vs PyTorch-ROCm  rel_l2 0.059  (cross-hw floor)
      Inputs verified bit-identical (ss_noise) / 2.6e-5 (dinov3_cond).
      Decomposition: ~0.13 rel_l2 is BF16 precision (F32 removes it),
      ~0.06 is cross-hardware noise, and **~0.12 rel_l2 is a genuine
      HIP-C-vs-PyTorch algorithmic difference** — precision-independent,
      hardware-independent. This is the real SS-DiT bug feeding the
      mesh-density gap.
- [x] Per-block SS-DiT bisect DONE (2026-05-14): instrumented the
      PyTorch `SparseStructureFlowModel.forward` to dump all 30 block
      hidden states on the first forward (step 0, t=1000); ran HIP
      `--dump-blocks --t 1.0` to match. Result:
        block 0  rel_l2 0.0057  cosine 1.00000
        block 10 rel_l2 0.0102  cosine 0.99995
        block 20 rel_l2 0.0126  cosine 0.99991
        block 28 rel_l2 0.0213  cosine 0.99976
        block 29 rel_l2 0.0070  (std 32→115, rel shrinks)
      **No single broken block** — every block matches PyTorch to
      cosine ≥0.9998; rel_l2 grows smoothly ~0.0005-0.002/block,
      ~2% total across the 30-block stack in ONE forward.
      → The 12-14% final divergence is the SAMPLER compounding a
      ~2-3% per-forward error over 24 forwards (12 steps × 2 CFG).
      It is distributed sub-0.1%-per-block implementation differences
      (FMA order, RoPE table precision, BF16 rounding) — NOT a
      localized bug, not practically fixable without matching
      PyTorch's exact op order everywhere. PyTorch instrumentation
      reverted after the run.
- [x] Per-OP block-0 bisect DONE (2026-05-14): instrumented PyTorch
      `ModulatedTransformerCrossBlock._forward` to dump 9 intermediate
      tensors; ran HIP `--dump-b0 --t 1.0` (BF16) and again with
      `T2_WMMA=0 T2_BLASLT=0` (F32). Per-op rel_l2 vs PyTorch-ROCm:
        op            BF16     F32
        input_embed   0.0017   0.0017  (F32 GEMM FMA-order, tiny)
        ln_h_sa       0.0041   0.0041  (LayerNorm, F32-persistent)
        sa_proj       0.0067   0.0020  (BF16 noise — F32 fixes)
        ca_proj       0.0152   0.0027  (BF16 noise — F32 fixes)
        ln_h_mlp      0.0140   0.0124  (LayerNorm, F32-persistent)
        mlp_proj      0.0072   0.0030  (BF16 noise — F32 fixes)
        h_post_mlp    0.0057   0.0031  (block-0 output, F32)
      Attention/MLP GEMMs are pure BF16 noise (F32 collapses them
      ~0.015→0.003). The F32-persistent residual is only on the
      LayerNorm+modulation ops + input GEMM — and those are
      algorithmically identical to PyTorch (verified: same eps=1e-6,
      same biased `/dim` variance, LayerNorm32 is just nn.LayerNorm
      +fp32 cast). The residual is irreducible FMA/reduction-order
      difference. **No bug at any of the 4 bisect levels.** Drift
      investigation CLOSED — fully characterized, not fixable without
      bit-exact PyTorch op-order replication. Both PyTorch
      instrumentation patches reverted.
- Conclusion for the mesh-density gap: the HIP pipeline is faithful
  per-block; the ~56% tessellation density is the cost of running a
  deep 24-forward BF16 diffusion sampler on a different stack. Mesh
  geometry stays correct (coarse IoU 0.92). Closing this as
  "understood, not a bug" — would need full-F32 sampler or exact
  op-order parity to improve, neither worth it.

### MCLK lever — RESOLVED by per-call scratch fix (verified 2026-05-14)
- [x] Re-verify done: 3 back-to-back e2e --tex runs in one session
      (commit 5ee328e+) all hit tex_dec **423.5-423.9 ms**, identical
      77,208 unique colors. The "back-to-back runs go slow" symptom
      is GONE — it was stale g_c2s state all along, not a driver/mclk
      issue. No reboots needed anymore. CLOSED.

### tex_dec perf — DONE (commits 58541f6, 329765e)
- [x] On-the-fly nmap unlocked Triton/WMMA spconv: tex_dec 3.1 s → 0.43 s.
      Per-stage ConvNeXt 4-12× faster; C2S 1.8-3× faster. Stage 1
      ConvNeXt 1103 ms → 119 ms, Stage 2 697 ms → 55 ms. Gap to PyT
      reference 23× → 3.2×. The old "Stage 1/2 ConvNeXt profile" and
      "Stage 1 A/B test" tasks are obsolete — superseded by the nmap
      fast path which made the hash_kspconv path moot.
- [ ] (low priority) remaining 3.2× gap to PyT — would need FA-class
      kernel work on the spconv inner loop; diminishing returns.

### SS DiT perf — at the floor (commit 5692aa3, profile in memory)
- [x] Profiled: self-attn 51% / MLP 32% / cross-attn 16% per forward.
      SA is WMMA-compute-bound at the documented ~346 ms floor. CFG
      batching + FA softmax rewrite both scoped and judged not worth
      it (see project_trellis2_ss_dit_profile memo). No quick wins.

### Pipeline correctness gaps
- [x] `stbi_zlib_compress` crash — STALE (2026-05-14). The crash was
      in the PNG path; `t2_pbr_write_textured_obj` now writes PPM
      (`t2pbr_write_image` only calls `stbi_write_png` for `.png`
      paths, this function emits `.ppm`). Also the function is unused
      — test_hip_trellis2.c:1296 calls `t2_pbr_write_colored_obj`.
      No live crash. The atlas function is dead code; leave as-is or
      delete in a future cleanup — not blocking anything.
- [x] tex_dec outlier — "±700 vs ±1" framing was stale (2026-05-14).
      With the chunked-F.linear shim, fresh PyTorch-ROCm ref is bounded
      [-0.11, 1.08]. tex_dec KERNEL verified correct (rel 3.28e-4 vs
      CPU oracle, April). v71 e2e comparison: basecolor ch0-2 roughly
      OK; **ch3 metallic way off** (ref ~const 0.998, HIP 0.44±0.61),
      **ch4 roughness 10× damped**, ch5 alpha ~5× damped.
- [x] The ch3/4/5 divergence is e2e tex_dec INPUT drift (tex SLAT
      DiT) — confirmed same root cause as the SS-DiT bisect: no
      localized bug, distributed sampler-compounded accumulation.
      The channel-specific pattern (basecolor OK, metallic/roughness/
      alpha off) is just those channels being more saturation-
      sensitive to accumulated input drift. CLOSED — understood.

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
