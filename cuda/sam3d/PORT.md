# SAM 3D Objects — CUDA port journal

The current state, drift table, phase summaries, and TODOs have been
moved to **`doc/sam3d.md`** (project-root-relative). See that file for
the doc-friendly snapshot.

This file is retained as the live, append-only kernelization journal.
Add new sub-stage entries here as Phase 4b / 5b / 6b lands; periodically
fold summaries into `doc/sam3d.md`.

## Current focus

Phase 4b (SS-VAE decoder) is GPU-resident and green in
`verify_ss_decoder`. Phase 7a full sampled E2E is now green. Phase
5b.0 moved the SLAT prune boundary to GPU; 5b.1 added the SLAT APE
CUDA primitive; 5b.2 added deterministic sparse downsample; 5b.3
added sparse upsample; 5b.4 added submanifold sparse Conv3d; 5b.5
validated full sparse self-attention; 5b.6 validated SLAT
cross-attention geometry; 5b.7 composed the no-up/down IO resblock.
5b.8 composed the SLAT transformer MLP residual path; 5b.9 composed
the SLAT self-attention residual path; 5b.10 composed the SLAT
cross-attention residual path; 5b.11 composed a full single-stream SLAT
transformer block; 5b.12 verifies a real checkpoint transformer block
against traced activations; 5b.13 verifies the full 24-block real
checkpoint transformer stack against traced activations; 5b.14 hoists
the transformer block weights into a reusable persistent SLAT DiT GPU
container; 5b.15 hoists the verified transformer block/stack launch
sequence into a reusable SLAT DiT forward driver; 5b.16 installs that
driver as a CUDA hook inside the SLAT ODE body while CPU still owns
sparse IO resblocks; 5b.17 reuses resident conditioning tokens inside
that hook; 5b.18 keeps hook activation/timestep scratch buffers
persistent across SLAT ODE steps; 5b.19 adds a real-weight CUDA gate
for `input_blocks[0]`; 5b.20 adds the real-weight CUDA gate for
downsampling `input_blocks[1]`; 5b.21 adds the real-weight CUDA gate
for upsampling `out_blocks[0]`; 5b.22 adds the real-weight CUDA gate
for `out_blocks[1]`; 5b.23 wires all four verified sparse IO launch
sequences into the SLAT runner through a CUDA hook; 5b.24 keeps that
hook's temporary device buffers persistent across calls; 5b.25 moves
the final no-affine LN + `out_layer` behind a CUDA hook; 5b.26 folds
APE into the transformer-stack CUDA hook; 5b.27 moves `input_layer`
behind a CUDA hook. 6b.1 starts GS decoder kernelization with fused CUDA
to_representation + PLY packing; 6b.2 moves the GS dense input/APE and
final head to CUDA hooks; 6b.3 moves the GS shifted-window attention
core to CUDA; 6b.4 moves the GS block MLP subblock to CUDA; 6b.5 moves
the full GS self-attention subblock to CUDA; 6b.6 collapses attention
and MLP into one GS full-block hook; 6b.7 collapses the GS 12-block
loop into one stack hook; 6b.8 collapses GS input+APE, the 12-block
stack, and final LN/out_layer into one full-transformer hook; 6b.9
feeds that device-resident transformer output directly into the CUDA PLY
pack in normal E2E; 5b.28 caches repeated SLAT transformer coords across
ODE steps; 6b.10 feeds the GS transformer from the SLAT device mirror.
5b.29/6b.11 adopts final SLAT features from the final-layer device output
and applies the fixed SLAT unnormalization on CUDA before GS consumes the
mirror; 5b.30/6b.12 adopts final SLAT coords from the argwhere device
buffer; 5b.31/6b.13 shares a cached device timestep embedding across
SLAT IO and transformer hooks; 5b.32/6b.14 caches repeated SLAT sparse
IO coordinate uploads; 5b.33/6b.15 removes the stale SLAT IO timestep
workspace buffer; 5b.34/6b.16 adds DtoD feature handoff between
adjacent SLAT CUDA hooks; 5b.35/6b.17 hardens that handoff as one-shot
and clear-on-mismatch; 5b.36/6b.18 adds matching one-shot coordinate
handoff between adjacent SLAT CUDA hooks; 5b.37/6b.19 moves normal SLAT
ODE orchestration into the CUDA runner while keeping the CPU hook path
for debug/trace.

## Drift table (live)

See `doc/sam3d.md` § "Status — drift table".

## Recent entries (most recent first)

- **5b.37 / 6b.19 — normal SLAT ODE device orchestration (GREEN).**
  `cuda_sam3d_run_slat_dit` now bypasses
  `sam3d_cpu_slat_dit_run_ode_from_coords` on the normal sampled path.
  The runner seeds initial SLAT features once for CPU RNG parity, uploads
  compact active coords/features, and executes every ODE step on device:
  input_layer, sparse input IO blocks, APE+SLAT transformer stack,
  row-wise skip concat, sparse output IO blocks, and final no-affine
  LN/out_layer. Final host readback is retained for `ctx->slat_tokens`;
  GS consumes the device SLAT mirror. Existing debug forward and trace
  paths still use the CPU hook scaffold. `verify_slat_dit` remains green
  with max_abs=4.374981e-05, mean_abs=4.012466e-06; `verify_slat_gs`
  remains green with transformer out_feats max_abs=1.040e-04,
  mean_abs=4.753e-06 and packed_ply max_abs=3.815e-06,
  mean_abs=1.457e-07. Full sampled E2E writes 38016 gaussians to
  `/tmp/cuda_sam3d_phase5_phase6_final_timed.ply` in real=118.04 s; the
  PLY is byte-identical to `/tmp/cuda_sam3d_5b36_6b18_timed.ply`.
- **5b.36 / 6b.18 — SLAT hook coordinate DtoD handoff (GREEN).**
  Added a one-shot coordinate handoff cache keyed by the live
  `sp3d_tensor->coords` pointer plus row count. Sparse IO hooks record
  the matching device coordinate buffer after creating the CPU
  replacement tensor; the next sparse IO or APE+transformer hook consumes
  it with DtoD when the pointer and shape match. On mismatch, the cache
  clears and the existing byte-keyed HtoD upload path remains the
  fallback. `verify_slat_dit` remains green with max_abs=4.374981e-05,
  mean_abs=4.012466e-06; `verify_slat_gs` remains green with transformer
  out_feats max_abs=1.040e-04, mean_abs=4.753e-06 and packed_ply
  max_abs=3.815e-06, mean_abs=1.457e-07. Full sampled E2E writes 38016
  gaussians to `/tmp/cuda_sam3d_5b36_6b18_timed.ply` in real=119.04 s;
  that PLY is byte-identical to `/tmp/cuda_sam3d_5b35_6b17_timed.ply`.
- **5b.35 / 6b.17 — SLAT feature handoff lifetime hardening (GREEN).**
  The runner-owned SLAT feature handoff cache is now one-shot: a
  successful DtoD consume clears the metadata, and pointer/shape mismatch
  clears it before falling back to HtoD. This keeps the cache limited to
  adjacent CUDA hooks and avoids allocator host-pointer reuse aliasing
  stale device data after unrelated CPU tensor allocations.
  `verify_slat_dit` remains green with max_abs=4.374981e-05,
  mean_abs=4.012466e-06; `verify_slat_gs` remains green with transformer
  out_feats max_abs=1.040e-04, mean_abs=4.753e-06 and packed_ply
  max_abs=3.815e-06, mean_abs=1.457e-07. Full sampled E2E writes 38016
  gaussians to `/tmp/cuda_sam3d_5b35_6b17_timed.ply` in real=118.09 s;
  that PLY is byte-identical to `/tmp/cuda_sam3d_5b34_6b16_timed.ply`.
- **5b.34 / 6b.16 — SLAT hook feature DtoD handoff (GREEN).**
  Added a runner-owned feature handoff cache keyed by the live
  `sp3d_tensor->feats` pointer plus shape. CUDA hooks that download a
  newly created tensor for the CPU ODE scaffold record the matching
  device output buffer; the next input-layer, sparse IO, APE+transformer,
  or final-layer hook consumes it with DtoD when the pointer and shape
  match, avoiding re-upload of the same host features. This remains a
  bridge until the SLAT ODE scaffold itself is device-resident.
  `verify_slat_dit` remains green with max_abs=4.374981e-05,
  mean_abs=4.012466e-06; `verify_slat_gs` remains green with transformer
  out_feats max_abs=1.040e-04, mean_abs=4.753e-06 and packed_ply
  max_abs=3.815e-06, mean_abs=1.457e-07. Full sampled E2E writes 38016
  gaussians to `/tmp/cuda_sam3d_5b34_6b16_timed.ply` in real=118.07 s;
  that PLY is byte-identical to `/tmp/cuda_sam3d_5b33_6b15_timed.ply`.
- **5b.33 / 6b.15 — SLAT IO timestep dead-buffer cleanup (GREEN).**
  Removed `cs3d_slat_io_ws.d_t/cap_t` and its free path. The IO hooks
  now consume the shared `d_slat_hook_t_emb` cache directly and keep only
  `ws->d_t_silu` for the SiLU-mutated timestep copy used by the block
  MLP. `verify_slat_dit` remains green with max_abs=4.374981e-05,
  mean_abs=4.012466e-06; `verify_slat_gs` remains green with transformer
  out_feats max_abs=1.040e-04, mean_abs=4.753e-06 and packed_ply
  max_abs=3.815e-06, mean_abs=1.457e-07. Full sampled E2E writes 38016
  gaussians to `/tmp/cuda_sam3d_5b33_6b15_timed.ply` in real=118.04 s;
  that PLY is byte-identical to `/tmp/cuda_sam3d_5b32_6b14_timed.ply`.
- **5b.32 / 6b.14 — SLAT sparse IO coord upload cache (GREEN).**
  Added byte-keyed host caches for `cs3d_slat_io_ws.d_in_coords` and
  `cs3d_slat_io_ws.d_coords`. Input coords and upsample target coords
  skip HtoD when unchanged; downsample invalidates the `d_coords` cache
  because that kernel overwrites the buffer. `verify_slat_dit` remains
  green with max_abs=4.374981e-05, mean_abs=4.012466e-06;
  `verify_slat_gs` remains green with transformer out_feats
  max_abs=1.040e-04, mean_abs=4.753e-06 and packed_ply max_abs=3.815e-06,
  mean_abs=1.457e-07. Full sampled E2E writes 38016 gaussians to
  `/tmp/cuda_sam3d_5b32_6b14_timed.ply` in real=118.13 s; that PLY is
  byte-identical to `/tmp/cuda_sam3d_5b31_6b13_timed.ply`.
- **5b.31 / 6b.13 — shared SLAT timestep embedding cache (GREEN).**
  Added `cs3d_slat_cached_t_emb` and routed both sparse IO block hooks
  and the APE+transformer hook through the same `d_slat_hook_t_emb`
  device buffer. Within an ODE step, the first hook uploads the exact
  fp32 `t_emb[dim]`; later hooks reuse it when the cached host bytes
  match, avoiding repeated timestep HtoD uploads. `verify_slat_dit`
  remains green with max_abs=4.374981e-05, mean_abs=4.012466e-06;
  `verify_slat_gs` remains green with transformer out_feats
  max_abs=1.040e-04, mean_abs=4.753e-06 and packed_ply max_abs=3.815e-06,
  mean_abs=1.457e-07. Full sampled E2E writes 38016 gaussians to
  `/tmp/cuda_sam3d_5b31_6b13_timed.ply` in real=118.08 s; that PLY is
  byte-identical to `/tmp/cuda_sam3d_5b30_6b12_timed.ply`.
- **5b.30 / 6b.12 — final SLAT coord device adoption (GREEN).**
  `cs3d_slat_argwhere_gpu` can now return ownership of the compact
  device coordinate buffer alongside the host copy required by the CPU
  ODE scaffold. When final `out_n` still matches the active voxel count,
  `cs3d_adopt_slat_device` DtoD-copies those coords into `d_slat_coords`
  instead of uploading from host. This pairs with 5b.29's device feature
  adoption so the normal GS path consumes a fully device-adopted SLAT
  mirror. `verify_slat_dit` remains green with max_abs=4.374981e-05,
  mean_abs=4.012466e-06; `verify_slat_gs` remains green with transformer
  out_feats max_abs=1.040e-04, mean_abs=4.753e-06 and packed_ply
  max_abs=3.815e-06, mean_abs=1.457e-07. Full sampled E2E writes 38016
  gaussians to `/tmp/cuda_sam3d_5b30_6b12_timed.ply` in real=118.05 s;
  that PLY is byte-identical to `/tmp/cuda_sam3d_5b29_6b11_timed.ply`.
- **5b.29 / 6b.11 — final SLAT device adoption + CUDA unnorm (GREEN).**
  Added `slat_unnormalize8_f32` and extended SLAT adoption so the normal
  sampled path can DtoD-copy the final SLAT final-layer output into
  `d_slat_feats`, unnormalize the fixed 8 latent channels on CUDA, and
  feed that device mirror directly to the GS transformer. Host SLAT
  tokens are still retained for readback/debug, and debug/override paths
  keep host-upload fallbacks. `verify_slat_dit` remains green with
  max_abs=4.374981e-05, mean_abs=4.012466e-06. `verify_slat_gs` remains
  green with transformer out_feats max_abs=1.040e-04,
  mean_abs=4.753e-06 and packed_ply max_abs=3.815e-06,
  mean_abs=1.457e-07. Full sampled E2E writes 38016 gaussians to
  `/tmp/cuda_sam3d_5b29_6b11_timed.ply` in real=118.07 s. Compared with
  the previous host-unnormalized E2E PLY, the CUDA-unnormalized PLY
  differs by max_abs=2.861e-06, mean_abs=2.129e-08 over 646272 floats.
- **5b.28 / 6b.10 — SLAT coord cache + SLAT-to-GS device mirror
  (GREEN).** The SLAT APE+transformer CUDA hook now keeps a host-side
  coordinate cache and skips repeated HtoD coordinate uploads when the
  sparse transformer layout is unchanged across ODE steps. The normal GS
  decode path also reuses `cs3d_adopt_slat`'s `d_slat_coords` and
  `d_slat_feats` mirrors: `cs3d_gs_transformer_gpu_forward` DtoD-copies
  those into the GS workspace instead of uploading coords/features from
  host. Debug/override paths keep the host-upload fallback. `verify_slat_dit`
  remains green with max_abs=4.374981e-05, mean_abs=4.012466e-06;
  `verify_slat_gs` remains green with transformer out_feats
  max_abs=1.040e-04, mean_abs=4.753e-06 and packed_ply max_abs=2.861e-06,
  mean_abs=1.011e-07. Full sampled E2E writes 38016 gaussians to
  `/tmp/cuda_sam3d_5b28_6b10_timed.ply` in real=118.12 s; that PLY is
  byte-identical to `/tmp/cuda_sam3d_6b9_timed.ply`.
- **6b.9 — SLAT GS device-connected transformer-to-PLY path (GREEN).**
  Refactored the full-transformer CUDA body into
  `cs3d_gs_transformer_gpu_forward` and added a device-input
  `cs3d_gs_pack_ply_gpu_device` launcher. Normal
  `cuda_sam3d_run_slat_gs_decode` now keeps GS coords plus final
  `[N,448]` out_layer features in `gpu_gs_head_ws` and launches the
  fused CUDA PLY pack directly from those device pointers, avoiding the
  previous download/re-upload pair. Debug APIs still return host
  transformer features for verifier comparisons. `verify_slat_gs`
  remains green: transformer out_feats max_abs=1.040e-04,
  mean_abs=4.753e-06; CUDA packed_ply vs CPU pack max_abs=2.861e-06,
  mean_abs=1.011e-07. Full sampled E2E writes 38016 gaussians to
  `/tmp/cuda_sam3d_6b9_timed.ply` in real=118.01 s; that PLY is
  byte-identical to `/tmp/cuda_sam3d_6b8_timed.ply`.
- **6b.8 — SLAT GS full-transformer CUDA hook (GREEN).** Added
  `sam3d_gs_decoder_set_transformer_hook` and installed it ahead of the
  lower-level input/APE, stack, block, attention-block, MLP, and final
  head hooks. The runner now uploads coords/features once, executes
  input SparseLinear(8 -> 768), GS APE, all 12 shifted-window
  transformer blocks, final no-affine LN, and out_layer
  SparseLinear(768 -> 448) on resident CUDA buffers, then downloads the
  final `[N,448]` tensor for the existing CUDA representation/PLY pack.
  `verify_slat_gs` remains green: transformer out_feats
  max_abs=1.040e-04, mean_abs=4.753e-06; CUDA packed_ply vs CPU pack
  max_abs=2.861e-06, mean_abs=1.011e-07. Full sampled E2E writes 38016
  gaussians to `/tmp/cuda_sam3d_6b8_timed.ply` in real=118.09 s.
- **6b.7 — SLAT GS full-stack CUDA hook (GREEN).** Added
  `sam3d_gs_decoder_set_stack_hook` and installed it ahead of the
  lower-level block hooks. The runner now uploads post-APE features once,
  executes all 12 shifted-window transformer blocks on resident `d_h`,
  then downloads once before the final head hook. Block, attention-block,
  and MLP hooks remain available as fallback/test boundaries.
  `verify_slat_gs` is unchanged from 6b.6: transformer out_feats
  max_abs=1.040e-04, mean_abs=4.753e-06; CUDA packed_ply vs CPU pack
  max_abs=2.861e-06, mean_abs=1.011e-07. Full sampled E2E writes 38016
  gaussians to `/tmp/cuda_sam3d_6b7_timed.ply` in real=118.09 s.
- **6b.6 — SLAT GS full-block CUDA hook (GREEN).** Added
  `sam3d_gs_decoder_set_block_hook` and installed it in the normal and
  debug GS transformer paths. The runner now uploads each block's input
  once, runs the 6b.5 self-attention subblock and 6b.4 MLP subblock
  back-to-back on the same resident `d_h`, then downloads the updated
  features once. Lower-level attention-block and MLP hooks remain as
  fallback/test boundaries. `verify_slat_gs` is unchanged from 6b.5:
  transformer out_feats max_abs=1.040e-04, mean_abs=4.753e-06; CUDA
  packed_ply vs CPU pack max_abs=2.861e-06, mean_abs=1.011e-07. Full
  sampled E2E writes 38016 gaussians to
  `/tmp/cuda_sam3d_6b6_timed.ply` in real=118.09 s.
- **6b.5 — SLAT GS self-attention subblock CUDA hook (GREEN).** Added
  `sam3d_gs_decoder_set_attn_block_hook` and runner-side persistent
  upload for all 12 GS `attn_qkv` and `attn_out` weight/bias pairs.
  Each block self-attention subblock now runs no-affine LN, QKV
  projection, shifted-window Q/K/V gather + `sdpa_f32` + scatter,
  output projection, and residual add on CUDA, then returns updated
  features to the CPU-owned block loop. The older inner
  window-attention hook remains available as a fallback boundary.
  `verify_slat_gs` remains green: transformer out_feats
  max_abs=1.040e-04, mean_abs=4.753e-06; CUDA packed_ply vs CPU pack
  max_abs=2.861e-06, mean_abs=1.011e-07. Full sampled E2E writes 38016
  gaussians to `/tmp/cuda_sam3d_6b5_timed.ply` in real=118.14 s.
- **6b.4 — SLAT GS MLP CUDA hook (GREEN).** Added
  `sam3d_gs_decoder_set_mlp_hook` and runner-side persistent upload for
  all 12 GS `mlp_fc{1,2}` weight/bias pairs. Each block MLP now runs
  no-affine LN, GEMM(768 -> 3072), tanh-approx GELU, GEMM(3072 -> 768),
  and residual add on CUDA, then returns updated features to the
  remaining CPU scaffold. The hook uses `gelu_tanh_inplace_f32` to match
  `sp3d_gelu` on the AVX2 CPU reference. `verify_slat_gs` remains
  green: transformer out_feats max_abs=6.604e-05, mean_abs=4.527e-06;
  CUDA packed_ply vs CPU pack max_abs=2.861e-06, mean_abs=1.011e-07.
  Full sampled E2E writes 38016 gaussians to
  `/tmp/cuda_sam3d_6b4_timed.ply` in real=118.34 s.
- **6b.3 — SLAT GS shifted-window attention CUDA hook (GREEN).** Added
  `sam3d_gs_decoder_set_window_attn_hook` and CUDA gather/scatter
  helpers for per-window QKV attention. The CPU GS block scaffold still
  owns LN, QKV/out projections, MLP, and residual adds, but each
  shifted-window attention inner loop now does gather -> `sdpa_f32` ->
  scatter on CUDA. `verify_slat_gs` remains green: transformer out_feats
  max_abs=1.192e-04, mean_abs=6.456e-06; CUDA packed_ply vs CPU pack
  max_abs=2.861e-06, mean_abs=1.011e-07. Full sampled E2E writes 38016
  gaussians to `/tmp/cuda_sam3d_6b3_timed.ply` in real=118.72 s.
- **6b.2 — SLAT GS dense-end CUDA hooks (GREEN).** Added
  `sam3d_gs_decoder_set_input_ape_hook` and
  `sam3d_gs_decoder_set_final_layer_hook`. The CUDA runner now applies
  GS input_layer SparseLinear(8 -> 768) + APE and final no-affine LN +
  out_layer SparseLinear(768 -> 448) on CUDA around the still-CPU 12
  shifted-window transformer blocks. `verify_slat_gs` remains green:
  transformer out_feats max_abs=8.249e-05, mean_abs=6.375e-06; CUDA
  packed_ply vs CPU pack max_abs=2.861e-06, mean_abs=1.009e-07. Full
  sampled E2E writes 38016 gaussians to
  `/tmp/cuda_sam3d_6b2_timed.ply` in real=119.55 s.
- **6b.1 — SLAT GS CUDA representation + PLY pack (GREEN).** Added
  `sam3d_gs_pack_ply_f32`, which fuses GS `to_representation` and final
  INRIA PLY layout packing after the still-CPU GS transformer. The CUDA
  path consumes coords plus transformer out_feats [N,448], applies
  offset perturbation, tanh/voxel scaling, opacity bias, stable
  log-softplus scaling activation, and writes [N*32,17] rows directly.
  `verify_slat_gs` now checks this kernel against CPU packing:
  packed_ply max_abs=3.815e-06, mean_abs=1.005e-07. Full sampled E2E
  with 5b.27 writes 38016 gaussians to
  `/tmp/cuda_sam3d_5b27_6b1_timed.ply` in real=120.52 s.
- **5b.27 — SLAT input-layer CUDA hook (GREEN).** Added
  `sam3d_slat_dit_set_input_layer_hook` and wired the CUDA runner to
  upload persistent `input_layer.{weight,bias}` with the sparse IO/final
  head bundle. The hook runs SparseLinear(8 -> 128) on CUDA and returns
  a replacement sparse tensor to the existing CPU ODE scaffolding.
  `verify_slat_dit` remains green: max_abs=4.374981e-05,
  mean_abs=4.012466e-06.
- **5b.26 — combined SLAT APE + transformer CUDA hook (GREEN).** Added
  `sam3d_slat_dit_set_ape_transformer_hook` so the normal CUDA runner
  can receive pre-APE features, apply `slat_ape_add_f32` on the same
  device buffer used by the SLAT transformer stack, and return
  post-stack features with no extra host/device round trip. The older
  post-APE transformer hook remains installed as a trace-preserving
  fallback when `SLAT_DIT_TRACE` is enabled. `verify_slat_dit` remains
  green: max_abs=3.898144e-05, mean_abs=4.255785e-06. Full sampled E2E
  writes 38016 gaussians to `/tmp/cuda_sam3d_5b26_timed.ply` in
  real=120.58 s (`/usr/bin/time -p`), flat versus 5b.25 as expected
  for removing only this small CPU pass.
- **5b.25 — SLAT final-head CUDA runner hook (GREEN).** Added
  `sam3d_slat_dit_set_final_layer_hook` and wired the CUDA runner to
  install it beside the sparse IO and transformer hooks for both normal
  ODE and debug forward calls. The hook uploads persistent
  `out_layer.{weight,bias}` with the sparse IO bundle, reuses the
  persistent sparse IO scratch buffers, runs no-affine layernorm and
  SparseLinear(128 -> 8) on CUDA, then returns a replacement
  `sp3d_tensor` to the existing CPU ODE scaffolding. `verify_slat_dit`
  remains green: max_abs=4.875660e-05, mean_abs=4.473753e-06. Full
  sampled E2E writes 38016 gaussians to
  `/tmp/cuda_sam3d_5b25_timed.ply` in real=120.62 s
  (`/usr/bin/time -p`), effectively flat versus 5b.24 because the
  remaining bottleneck is outside this small final head.
- **5b.24 — persistent SLAT sparse IO hook scratch buffers (GREEN).**
  The sparse IO hook now stores reusable device buffers in
  `cuda_sam3d_ctx`, growing them only when a later block needs more
  capacity. This removes per-block `cuMemAlloc`/`cuMemFree` churn for
  coords, feature tensors, down/up outputs, coord index, timestep
  scratch, embedding, skip, and intermediate activations. Numerics are
  unchanged: `verify_slat_dit` max_abs=4.863739e-05,
  mean_abs=4.471638e-06. Full sampled E2E writes 38016 gaussians to
  `/tmp/cuda_sam3d_5b24_timed.ply` in real=120.40 s; this is a small
  movement from 5b.23 and should be treated as allocation cleanup rather
  than a proven throughput gain.
- **5b.23 — SLAT sparse IO CUDA runner hook (GREEN).** Added
  `sam3d_slat_dit_set_io_block_hook` beside the existing transformer
  hook and wired `cuda_sam3d_runner.c` to install a CUDA implementation
  around normal SLAT forward/ODE calls. The runner now keeps persistent
  device weights and cached kernel handles for all four verified sparse
  IO resblocks, then returns replacement `sp3d_tensor` objects to the
  existing CPU ODE scaffolding. `verify_slat_dit` exercises sparse IO +
  transformer hooks together: max_abs=4.863739e-05,
  mean_abs=4.471638e-06. Full sampled E2E writes 38016 gaussians to
  `/tmp/cuda_sam3d_5b23_timed.ply` in real=120.65 s
  (`/usr/bin/time -p`).
- **5b.22 — real-weight SLAT output block 1 CUDA verifier (GREEN).**
  Added `verify_slat_out_block1_realw`, which starts from traced
  `c_h_after_out_block_0.npy`, concatenates the remaining
  `c_h_after_input_block_0.npy` skip, uses
  `c_coords_after_input_block_0.npy`, and runs real checkpoint
  `out_blocks[1]` skip projection plus submanifold conv sequence. Gate
  against `/tmp/sam3d_ref_5b20`: N=1024, C_in=256, C_out=128,
  dim=1024, max_abs=2.670288e-05, mean_abs=1.852996e-06,
  avg=0.8912 ms over 20 launches. This completes standalone real-weight
  CUDA gates for all four SLAT sparse IO resblocks; runner wiring is
  next.
- **5b.21 — real-weight SLAT output block 0 CUDA verifier (GREEN).**
  Added `verify_slat_out_block0_realw`, which starts from traced
  `c_h_after_block_23.npy`, concatenates the matching
  `c_h_after_input_block_1.npy` skip, upsamples into
  `c_coords_after_input_block_0.npy`, and runs real checkpoint
  `out_blocks[0]` skip projection plus submanifold conv sequence. Gate
  against `/tmp/sam3d_ref_5b20`: Nsrc=1007, Ndst=1024, C_in=2048,
  C_out=128, dim=1024, max_abs=7.247925e-05,
  mean_abs=4.278031e-06, avg=5.8301 ms over 20 launches.
- **5b.20 — real-weight SLAT input block 1 CUDA verifier (GREEN).**
  Added `verify_slat_input_block1_realw`, which starts from traced
  `c_h_after_input_block_0.npy` / `c_coords_after_input_block_0.npy`,
  runs the factor-2 sparse downsample plus real checkpoint
  `input_blocks[1]` skip projection and submanifold conv sequence, and
  compares to `c_h_after_input_block_1.npy` /
  `c_coords_after_input_block_1.npy`. Gate against regenerated current
  trace `/tmp/sam3d_ref_5b20`: N0=1024, N1=1007, C_in=128,
  C_out=1024, dim=1024, max_abs=7.390976e-06,
  mean_abs=3.324221e-07, coord_bad=0, avg=65.4651 ms over 20 launches.
  The older `/tmp/sam3d_ref` intermediate is stale for block1 features
  (coords match; feature max_abs=6.041392), so use regenerated current
  traces when validating this gate.
- **5b.19 — real-weight SLAT input block 0 CUDA verifier (GREEN).**
  Added `verify_slat_input_block0_realw`, which starts from traced
  `c_h_after_input_layer.npy`, uses traced `c_t_emb.npy` and real
  checkpoint `input_blocks[0]` weights, runs the CUDA sparse IO
  resblock sequence, and compares to `c_h_after_input_block_0.npy`.
  Gate: N=1024, C=128, dim=1024, max_abs=5.722046e-06,
  mean_abs=2.371230e-07, avg=0.4729 ms over 20 launches. The verifier
  uses `qtensor.n_rows * qtensor.n_cols` for 5D conv-weight upload; a
  `qt_numel()` upload truncates the final conv dimension and produces
  non-finite output.
- **5b.18 — persistent SLAT hook scratch buffers (GREEN).** The hybrid
  transformer hook now stores activation and timestep device buffers in
  `cuda_sam3d_ctx`, growing them only when a later call needs more
  capacity. This removes per-step `cuMemAlloc`/`cuMemFree` churn for
  `d_x` and `d_t_emb`; cond-token behavior remains the 5b.17 resident
  fast path plus verifier upload fallback. `verify_slat_dit` is
  unchanged: max_abs=4.410744e-05, mean_abs=4.311766e-06. Full sampled
  E2E writes 38016 gaussians to `/tmp/cuda_sam3d_5b18_timed.ply` in
  real=128.21 s, within run variance of 5b.17 rather than a proven
  throughput gain.
- **5b.17 — resident cond-token reuse in SLAT transformer hook (GREEN).**
  The normal E2E runner path already keeps cond tokens resident in
  `ctx->d_cond_tokens`; the SLAT transformer hook now reuses that device
  buffer instead of re-uploading cond once per SLAT ODE step. Override
  buffers used by verifiers still take the explicit upload fallback.
  `verify_slat_dit` is unchanged: max_abs=4.410744e-05,
  mean_abs=4.311766e-06. Full sampled E2E `--steps 2 --slat-steps 12
  --seed 42` writes 38016 gaussians to
  `/tmp/cuda_sam3d_5b17_timed.ply` in real=126.92 s
  (`/usr/bin/time -p`).
- **5b.16 — hybrid SLAT ODE body with GPU transformer stack (GREEN).**
  Added a narrow transformer-stack hook in `common/sam3d_slat_dit.h`
  and wired `cuda_sam3d_runner.c` to install it around
  `sam3d_cpu_slat_dit_forward` / `sam3d_cpu_slat_dit_run_ode_from_coords`.
  The CPU path still owns input layer, sparse input/output resblocks,
  APE, final LN, and output layer; the hook uploads post-APE activations
  and calls `cs3d_slatdit_stack_forward` over persistent
  `cs3d_slatdit_gpu` weights. `verify_slat_dit` now exercises this
  hybrid path: max_abs=4.410744e-05, mean_abs=4.311766e-06. The
  standalone stack verifier remains green:
  max_abs=4.821777e-03, mean_abs=4.799369e-05, avg=3166.6134 ms.
  Full sampled E2E `--steps 2 --slat-steps 12 --seed 42` writes 38016
  gaussians to `/tmp/cuda_sam3d_5b16_timed.ply` in real=128.73 s
  (`/usr/bin/time -p`).
- **5b.15 — reusable SLAT DiT transformer stack forward driver
  (GREEN).** Added `cuda_sam3d_slat_dit_forward.h` with
  `cs3d_slatdit_fns_lookup`, block workspace allocation/free,
  `cs3d_slatdit_block_forward`, and `cs3d_slatdit_stack_forward`.
  Refactored `verify_slat_transformer_block_realw` into a thin harness
  around the reusable driver and the persistent `cs3d_slatdit_gpu`
  weight container. Gates are unchanged: single block `--block 0
  --repeat 3` max_abs=1.220703e-04, mean_abs=3.961771e-06,
  avg=131.9374 ms; stack `--block 23 --stack --repeat 1`
  max_abs=4.821777e-03, mean_abs=4.799369e-05, avg=3167.2253 ms.
  Existing CUDA SLAT verifier remains green: max_abs=1.668930e-05,
  mean_abs=1.609899e-06.
- **5b.14 — persistent SLAT DiT transformer GPU weight container
  (GREEN).** Added `cuda_sam3d_slat_dit_gpu.h` with
  `cs3d_slatdit_gpu_load_transformer` / `cs3d_slatdit_gpu_free`,
  holding all 24 blocks' AdaLN, self-attn, cross-attn, and MLP weights
  as device buffers. Refactored `verify_slat_transformer_block_realw`
  to consume this reusable GPU model container instead of verifier-local
  upload structs. Post-refactor gates are unchanged: single block
  `--block 0 --repeat 3` max_abs=1.220703e-04,
  mean_abs=3.961771e-06, avg=132.0270 ms; stack
  `--block 23 --stack --repeat 1` max_abs=4.821777e-03,
  mean_abs=4.799369e-05, avg=3175.5491 ms. Existing CUDA SLAT verifier
  remains green: max_abs=1.668930e-05, mean_abs=1.609899e-06.
- **5b.13 — real-weight 24-block SLAT transformer stack verifier
  (GREEN).** Extended `verify_slat_transformer_block_realw` with
  `--stack`, which starts from traced `c_h_after_ape.npy`, uploads all
  checkpoint block weights once, runs blocks 0..23 on the same
  device-resident activation buffer, and compares to traced
  `c_h_after_block_23.npy`. Real traced shape: N=1007, N_cond=1374,
  dim=1024, H=16, D_h=64, hidden=4096.
  `verify_slat_transformer_block_realw --block 23 --stack --repeat 1`:
  max_abs=4.821777e-03, mean_abs=4.799369e-05, avg=3168.8317 ms,
  threshold=1e-2. The looser stack gate is intentional: the single block
  remains at max_abs=1.220703e-04, and the larger max is accumulated
  CUDA `expf`/GEMM reduction-order drift across 24 blocks. Existing CUDA
  SLAT verifier remains green: max_abs=1.668930e-05,
  mean_abs=1.609899e-06.
- **5b.12 — real-weight SLAT transformer block verifier (GREEN).** Added
  `verify_slat_transformer_block_realw`, which loads
  `sam3d_slat_dit.safetensors`, starts from traced `c_h_after_ape.npy`,
  runs checkpoint transformer block 0 on CUDA, and compares against
  traced `c_h_after_block_0.npy`. The verifier includes AdaLN modulation
  (`silu_inplace_f32` + `gemm_f32_bias`) plus the full
  self-attn/cross-attn/MLP residual path. Real traced shape:
  N=1007, N_cond=1374, dim=1024, H=16, D_h=64, hidden=4096.
  `verify_slat_transformer_block_realw --block 0 --repeat 3`:
  max_abs=1.220703e-04, mean_abs=3.961771e-06, avg=131.9640 ms.
  Existing CUDA SLAT verifier remains green: max_abs=1.668930e-05,
  mean_abs=1.609899e-06.
- **5b.11 — composed full SLAT transformer block path (GREEN).** Added
  `test_slat_transformer_block`, composing self-attn residual
  (`modulated_ln_f32`, QKV `gemm_f32_bias`,
  `multi_head_rmsnorm_f32`, `qkv_split_f32`, self `sdpa_f32`, output
  `gemm_f32_bias`, `gated_residual_add_f32`), cross-attn residual
  (`layernorm_token_f32`, Q/KV projections, `kv_split_f32`, cross
  `sdpa_f32`, output projection, `residual_add_f32`), and MLP residual
  (`modulated_ln_f32`, fc1, `gelu_tanh_inplace_f32`, fc2,
  `gated_residual_add_f32`). Correctness uses production `dim=1024,
  H=16, D_h=64, hidden=4096` at N=64, N_cond=256. Standalone
  microtest `test_slat_transformer_block --repeat 5`:
  max_abs=1.013e-06, mean_abs=1.174e-07, avg=9.1691 ms. Existing CUDA
  SLAT verifier remains green: max_abs=1.668930e-05,
  mean_abs=1.609899e-06.
- **5b.10 — composed SLAT cross-attention residual path (GREEN).**
  Added `test_slat_cross_attn_residual`, composing affine
  `layernorm_token_f32`, Q `gemm_f32_bias`, cond KV `gemm_f32_bias`,
  `kv_split_f32`, cross `sdpa_f32`, output `gemm_f32_bias`, then
  `residual_add_f32`. Correctness uses production `dim=1024, H=16,
  D_h=64` at N=128, N_cond=512. Standalone microtest
  `test_slat_cross_attn_residual --repeat 10`: max_abs=1.192e-07,
  mean_abs=5.959e-09, avg=8.5170 ms. Existing CUDA SLAT verifier
  remains green: max_abs=1.668930e-05, mean_abs=1.609899e-06.
- **5b.9 — composed SLAT self-attention residual path (GREEN).** Added
  `test_slat_self_attn_residual`, composing `modulated_ln_f32`,
  `gemm_f32_bias` QKV projection, `multi_head_rmsnorm_f32` on Q/K,
  `qkv_split_f32`, `sdpa_f32`, output `gemm_f32_bias`, then
  `gated_residual_add_f32`. Correctness uses production `dim=1024,
  H=16, D_h=64` at N=128. Standalone microtest
  `test_slat_self_attn_residual --repeat 10`: max_abs=1.192e-07,
  mean_abs=9.934e-09, avg=3.3442 ms. Existing CUDA SLAT verifier
  remains green: max_abs=1.668930e-05, mean_abs=1.609899e-06.
- **5b.8 — composed SLAT transformer MLP residual path (GREEN).** Added
  `test_slat_mlp`, composing `modulated_ln_f32`, `gemm_f32_bias`
  fc1, `gelu_tanh_inplace_f32`, `gemm_f32_bias` fc2, then
  `gated_residual_add_f32`. Correctness uses production
  `dim=1024, hidden=4096` at N=128 to keep the scalar CPU reference
  practical. Standalone microtest `test_slat_mlp --repeat 10`:
  max_abs=1.073e-06, mean_abs=1.134e-07, avg=6.3216 ms. Existing CUDA
  SLAT verifier remains green: max_abs=1.668930e-05,
  mean_abs=1.609899e-06.
- **5b.7 — composed SLAT IO resblock path (GREEN).** Added
  `test_slat_resblock`, composing the no-up/down, identity-skip
  `SparseResBlock3d` path used by `input_blocks[0]`: affine
  `layernorm_token_f32`, `silu_inplace_f32`, `slat_submconv3x3_f32`,
  `modulated_ln_f32`, second SiLU+SubMConv, then `residual_add_f32`.
  Standalone microtest `test_slat_resblock --repeat 20`: N=1188,
  C=128, max_abs=2.384e-07, mean_abs=1.847e-08, avg=0.5319 ms.
  Existing CUDA SLAT verifier remains green: max_abs=1.668930e-05,
  mean_abs=1.609899e-06.
- **5b.6 — SLAT cross-attention core path (GREEN).** Added
  `test_slat_cross_attn`, a production-geometry verifier for the SLAT
  cross-attn core: direct `q[N,dim]` plus fused `kv_c[N_cond,2*dim]`,
  `kv_split_f32`, then `sdpa_f32`. This pins the
  active-voxel-to-conditioning-token attention layout before composing a
  full single-stream SLAT block. Standalone microtest
  `test_slat_cross_attn --repeat 10`: N=1188, N_cond=2740, dim=1024,
  H=16, D_h=64, max_abs=2.161e-07, mean_abs=6.685e-09,
  SDPA avg=41.9248 ms. Existing CUDA SLAT verifier remains green:
  max_abs=1.668930e-05, mean_abs=1.609899e-06.
- **5b.5 — SLAT full sparse self-attention path (GREEN).** Added
  `test_slat_self_attn`, a SLAT-shaped verifier for the batch=1
  `sp3d_attention` equivalent using existing `qkv_split_f32` and
  `sdpa_f32`. This pins the CUDA data layout for transformer self-attn
  before wiring a full SLAT block. Standalone microtest
  `test_slat_self_attn --repeat 20`: N=1188, dim=1024, H=16,
  D_h=64, max_abs=1.192e-07, mean_abs=7.983e-09, SDPA avg=17.9921 ms.
  Existing CUDA SLAT verifier remains green: max_abs=1.668930e-05,
  mean_abs=1.609899e-06.
- **5b.4 — SLAT submanifold Conv3d primitive (GREEN).** Added
  `slat_build_coord_index64_i32` and `slat_submconv3x3_f32`, matching
  `sp3d_conv3d_forward` for fixed sparse output coords, 3×3×3
  submanifold neighborhoods, and weight layout `[out_C,27,in_C]`.
  The test uses a dense 64³ coord-index grid on GPU, then one thread
  per `(voxel,out_channel)` accumulates in `k` then `in_C` order.
  Standalone microtest `test_slat_submconv3d --repeat 20`: N=1188,
  Cin=128, Cout=128, max_abs=1.192e-07, mean_abs=9.126e-09,
  avg=0.2609 ms. Existing CUDA SLAT verifier remains green:
  max_abs=1.668930e-05, mean_abs=1.609899e-06.
- **5b.3 — SLAT sparse upsample primitive (GREEN).** Added
  `slat_upsample2_nearest_f32`, matching `sp3d_upsample(...,
  factor=2)`: target row order is preserved; each target row looks up
  source coord `(b,z/2,y/2,x/2)` and copies the full feature row, or
  zero-fills when missing. Standalone microtest
  `test_slat_upsample --repeat 200`: src_N=1176, target_N=1188,
  C=2048, max_abs=0, mean_abs=0, avg=0.3862 ms. Existing CUDA SLAT
  verifier remains green: max_abs=1.668930e-05,
  mean_abs=1.609899e-06.
- **5b.2 — SLAT sparse downsample primitive (GREEN).** Added
  `slat_downsample2_mean_include_self_serial_f32`, matching
  `sp3d_downsample(..., factor=2, pool_mode=2)`: output coords are
  first-occurrence unique `floor((z,y,x)/2)` rows, and features are
  divided by `count+1` to match torch `scatter_reduce(mean,
  include_self=True)` into zeros. The implementation is intentionally
  serial for deterministic coordinate ordering while the surrounding
  SLAT ODE remains CPU-backed; it is not the final parallel compaction
  path. Standalone microtest `test_slat_downsample --repeat 200` with
  randomized unique 64³ coords: N=1188, C=128, out_N=1165, max_abs=0,
  mean_abs=0, coords=OK, avg=45.5210 ms. Existing CUDA SLAT verifier
  remains green: max_abs=1.668930e-05, mean_abs=1.609899e-06.
- **5b.1 — SLAT APE CUDA primitive (GREEN).** Added
  `slat_ape_add_f32`, matching
  `common/sam3d_slat_dit.h::slat_apply_ape`: coords are `(b,z,y,x)`,
  `freq_dim=dim/3/2`, each axis is laid out as all `sin(j)` then all
  `cos(j)`, and the 1024-channel tail is untouched. Standalone
  microtest `test_slat_ape --repeat 200`: N=1188, dim=1024,
  filled=1020, max_abs=8.374e-06, mean_abs=3.035e-07, avg=0.0145 ms.
  Existing CUDA SLAT verifier remains green: max_abs=1.668930e-05,
  mean_abs=1.609899e-06.
- **5b.0 — SLAT occupancy prune on GPU (GREEN).** Added
  `slat_occ_argwhere_serial_f32`, a deterministic CUDA argwhere over
  64³ occupancy logits that preserves CPU sparse coord order
  `(b,z,y,x)` and avoids atomic compaction nondeterminism. The CUDA
  runner now downloads those coords and calls
  `sam3d_cpu_slat_dit_run_ode_from_coords`, so seeded SLAT noise and
  downstream sparse ops remain stable while the prune boundary is no
  longer host-side. `verify_slat_dit`: max_abs=1.668930e-05,
  mean_abs=1.609899e-06. Full sampled E2E with `--steps 2
  --slat-steps 12 --cfg 2.0 --precision fp32`: GPU argwhere reports
  1188 active voxels, writes 38016 gaussians to
  `/tmp/cuda_sam3d_phase5b0.ply`, elapsed=231.06 s.
- **7a.1 — sampled E2E unblocked (GREEN).** Fixed SS sampler wrapper
  semantics to match upstream pointmap inference: `rescale_t=3`,
  `reversed_timestamp=False`, `no_shortcut=True` (`d=0`), and
  `unconditional_handling=add_flag`. `add_flag` is implemented as a
  zero-conditioned second SS DiT forward in the CFG interval, followed
  by velocity mixing. `verify_ss_dit_ode` after CFG parity:
  max_abs=2.193e-05, mean_abs=1.81e-06. Full CUDA E2E with
  `--steps 2 --slat-steps 12 --cfg 2.0 --precision fp32`:
  SS decoder forward=561.6 ms, occupancy `pos=1188`, writes 38016
  gaussians to `/tmp/cuda_sam3d_phase4b.ply`, elapsed=267.37 s.
- **4b.3 — SS-VAE decoder runner swap (GREEN).** Added
  `cuda_sam3d_ss_decoder_gpu.h` (281.0 MiB device weights) and
  `cuda_sam3d_ss_decoder_forward.h`, composed from
  `conv3d_k3_pad1_f32`, `channel_layernorm_3d_f32`,
  `silu_inplace_f32`, `pixel_shuffle_3d_f32`, and `residual_add_f32`.
  `cuda_sam3d_run_ss_decode` now calls the GPU path and keeps the host
  occupancy mirror for downstream CPU fallback. `verify_ss_decoder`:
  max_abs=2.770424e-04, mean_abs=3.038599e-05, forward=557.7 ms.
  Important loader note: SS decoder conv weights are 5D, so upload uses
  `qtensor.n_rows * qtensor.n_cols`; `qt_numel()` truncates dim 5.
- **4b.2 — e2e diagnostics.** Full sampled CLI reports active
  occupancy voxels before SLAT. This exposed the sampler-wrapper bug
  fixed in 7a.1. `--slat-ref` bypass remains green: 8192 gaussians in
  2.77 s.
- **4b.1 — decoder primitives.** Added `channel_layernorm_3d_f32` and
  `conv3d_k3_pad1_f32` for the SS-VAE dense 3D-conv path.
- **4b.0 — `pixel_shuffle_3d_f32` (GREEN).** First Phase 4b primitive
  for the SS-VAE decoder upsample stages: rearranges
  `[C*8, D, H, W]` → `[C, 2D, 2H, 2W]` with sub-channel layout
  `sub_ch(sd, sh, sw) = (sd*2 + sh)*2 + sw` matching the CPU host
  reference (`common/trellis2_ss_decoder.h::t2dec_pixel_shuffle_3d`).
  Pure data movement → bit-exact. One thread per output element.
  Standalone microbench `test_pixel_shuffle_3d` exercises both
  SS-VAE upsample shapes (16³→32³ at C=128 [in=4.2M, out=4.2M], and
  32³→64³ at C=32 [in=8.4M, out=8.4M]) with random inputs, and diffs
  via `memcmp` against a host reference. Result on both shapes:
  **max_abs=0, BIT-EXACT yes** (memcmp == 0). Reusable across both
  SS-VAE upsample stages.
- **2c.15** — one-shot cond upload across ODE steps (`cond=NULL`
  semantic). `verify_ss_dit_ode` steps=4: max_abs=1.15e-4.
- **2c.14** — GPU shortcut ODE in `cuda_sam3d_run_ss_dit`. steps=2:
  max_abs=1.90e-5 vs CPU host.
- **2c.13** — wire `cs3d_ssdit_outer_forward` into runner.
  `verify_ss_dit` max_abs=1.05e-5.
- **2c.12** — `cuda_sam3d_ssdit_outer.h` full GPU forward.
  `verify_ssdit_full_realw` max_abs=5.72e-6 vs CPU.

For Phase 0 → 2c.11 details, see `doc/sam3d.md`.
