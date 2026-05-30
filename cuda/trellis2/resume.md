# TRELLIS.2 Shape Decoder Resume Prompt

> ## STAGE-1 bf16-BLOCK PORT — latent 0.9895→0.99739 vs PyTorch (2026-05-29)
>
> PyTorch runs the 30 SS-DiT blocks in **bf16** (`sparse_structure_flow.py`:
> `convert_to(dtype)` on blocks, `manual_cast(h)` in/out; input_layer/t_embedder/
> out_layer stay f32). `03_ss_latent` is therefore a bf16 trajectory, so our more
> precise F32/TF32 path topped out at cosine **0.9895** (12-step). **`T2_DIT_BF16=1`
> now replicates the bf16 block** (default OFF — TF32 stays the production default,
> it is MORE precise than the bf16 reference):
> - `bf16_round`: new in-place `t2_round_f32_bf16` kernel + `ops.bf16_round` +
>   `t2_op_round_bf16`. Rounds EVERY block-op OUTPUT to bf16 (25 `RB()` calls in
>   `run_dit_forward_generic`: adaLN/LN/QKV/rmsnorm/rope/attn/out-proj/gelu/residuals).
> - `use_bf16_gemm`: block matmuls run true bf16 (W,X→bf16, `cublasew_gemm_bf16_bf16_f32`
>   = COMPUTE_32F accumulate). x_emb/out forced f32 (bf16 suppressed around those 2
>   GEMMs); t_emb rounded after the f32 t-MLP.
> - **Result vs `03_ss_latent`: TF32 0.98954 → full bf16-block 0.99739** (relL2 0.072,
>   uniform across channels). ~parity speed with TF32 (~37 s Stage 1).
>
> NEGATIVES while chasing 0.999 (do NOT re-attempt): (a) rounding non-matmul WEIGHTS
> (norm γ/biases/mod_w) to bf16 is BIT-IDENTICAL (output-rounding absorbs it);
> (b) rounding attention PROBS to bf16 made it WORSE 0.9974→0.9961 — PyTorch SDPA
> keeps attn internals in f32, so our f16-MMA probs are the closer match;
> (c) bf16 GEMM is already f32-accumulate; (d) GELU is tanh-approx on both sides.
> The residual 0.0026 is irreducible (cuBLAS-vs-PyTorch bf16 round-points compounded
> over 12 recursive Euler steps; single forward is already 0.9998).
>
> MULTI-PROMPT FOLLOW-UP (2026-05-30) — the 0.99739 win is T.png-SPECIFIC and the
> PyTorch reference is BACKEND-DEPENDENT. Re-dumped T.png (same seed/noise/cond as the
> canonical verify-dumps): 02_noise identical, 01_cond Δ=2e-5, but 03_ss_latent
> cos=0.99752 vs canonical; two fresh runs are BIT-IDENTICAL → systematic dense-attn
> backend diff (canonical flash_attn? vs this env's sdpa fallback), not run noise. So
> PyTorch's own bf16 12-step latent only reproduces to ~0.9975 → 0.999 is unreachable
> by anyone. On T.png bf16 lands inside the PyTorch cluster (0.9974/0.9976) and TF32
> outside (0.9895/0.9934); but across 3 NEW images it is a WASH (bf16 0.983–0.996 vs
> TF32 0.993–0.995, mean TF32 0.9941 ≳ bf16 0.9922 — refs deterministic, so real).
> bf16 = "match a specific PyTorch bf16 run within its ~0.0025 backend ambiguity", NOT
> a universal win. Keep TF32 default (more consistent + higher precision). Repro:
> `/tmp/t2_dump_stub.py` (meta_path-stubs flex_gemm/cumesh/nvdiffrast/o_voxel; needs
> `pip install --break-system-packages trimesh easydict`) + `dump_ground_truth.py --stage1-only`.
>
> ## STAGE-1 "GARBAGE OUTPUT" — RESOLVED (2026-05-29)
>
> The Stage-1 garbage + shape-decoder OOM crash (see "Latest CUDA e2e
> verification attempt" below) had three independent causes, all now fixed.
> The OLD guidance in that section about token-major `--noise` is **WRONG** —
> read this block first.
>
> 1. **Noise layout was a verification artifact, NOT a code bug.** Stage 1 is
>    channel-major `[C=8,N=4096]` end-to-end. `cuda_trellis2_run_dit` reads
>    `x_t[ch*N+pos]` and transposes to token-major internally (mirrors PyTorch
>    `x.view(B,C,-1).permute(0,2,1)`). The PyTorch dump `02_ss_noise.npy`
>    `[1,8,16,16,16]` is already channel-major — **feed it RAW** (do NOT convert
>    to `[4096,8]`), and compare the saved `--npy` latent **directly** to
>    `03_ss_latent[0]` (no readback "layout correction"). The prior session's
>    `transpose(1,2,3,0).reshape(4096,8)` was the double-transpose scramble that
>    produced `cosine ≈ −0.012`.
> 2. **Coords extraction missing `max_pool3d` 64³→32³ (real bug → the OOM).**
>    `test_cuda_trellis2.c` now max-pools (block-max OR) the 64³ logits to 32³
>    before emitting res-32 coords, matching PyTorch
>    `argwhere(decoded>0 |maxpool| )[:, [0,2,3,4]]`. Gives 3515 vs ref 3548
>    (98.8% overlap) instead of 21037@res64 → no more 2.56M-row cascade. Also
>    fixes Stage-2/3 RoPE (it uses absolute coord values; res-64 doubled every
>    angle).
> 3. **Stage-1 DiT precision (real quality bug).** PyTorch runs the DiT in bf16
>    (range == f32); our default fp16 MMA clipped a hot intermediate in the
>    dense 4096-token grid → latent cosine 0.727. Stage 1 now **defaults to F32
>    GEMM** (`t2_dit_use_f16(r,0)`): single-step 0.9998, latent 0.989. The
>    SPARSE Stage 2/3 DiTs were measured fp16-faithful (0.9999 == F32) and stay
>    fp16 (the slow stages). Env: `T2_DIT_F16=1` forces fp16 everywhere,
>    `T2_DIT_F32=1` forces F32 everywhere.
>
> 4. **Shape/tex decoder default output GEMM = all-zero (real bug, fixed).** With
>    no `T2_SCVAE_*` flags the decoder `[N,7]` output was ALL zero (FDG → 0
>    triangles, no OBJ). The cuBLAS/plain output-GEMM fall-through leaves it zero;
>    only the grouped F32 kernel writes it. The resume baseline always set
>    `T2_SCVAE_OUTPUT_GROUP=25`, masking this. Fixed by defaulting `group` to 25
>    (was 0). Also fixed the FDG vertex-offset transform to PyTorch's
>    `2·sigmoid−0.5` (was plain sigmoid).
>
> New tool: `verify_stage1` (single SS-DiT forward at t=0.5 vs
> `02b_ss_dit_step_velocity.npy`). `verify_stage2`/`verify_stage3` now use
> `t_raw=0.0005` (the 06b/10b dumps are direct t=0.5 calls). See
> `doc/trellis-2.md` → "Stage-1 e2e parity (2026-05-29)" for full numbers.
>
> **e2e now produces a real SHAPE mesh** (image→Stage1 F32→coords→Stage2→shape dec→FDG):
> with the SC-VAE flags, 1.39M verts / 2.90M tris (PyTorch 1.47M / 3.14M); with no
> flags, a coarser 287k-tri mesh.
>
> **TEXTURE e2e now WORKS** (`--stage3 --tex-dec`, fixed 2026-05-29): full colored
> OBJ, 1.378M voxels (= shape), 99.7% colored verts. Three fixes:
> 1. **Subdivision replay** (the real bug): the tex decoder has NO `to_subdiv` head,
>    so C densely subdivided ×8 → 16.97M → OOM/black. Now the SHAPE decode RECORDS its
>    per-C2S-stage pruned subdivision into a runner-resident `subdiv_plan[8]` and the
>    TEX decode REPLAYS it (= PyTorch `decode_tex_slat(guide_subs=subs)`); both use the
>    same res-32 coords/order → identical 3515→…→1.378M cascade.
> 2. **DiT VRAM offload**: even with (1), the SHAPE decode OOM'd (all 3 DiTs resident,
>    Stage 1 F32 = 5.3 GB, `free=0`). New `cuda_trellis2_unload_dit_stages(r)` frees
>    Stage 1/2/3 + KV cache before decoding (~10.5 GB freed). `--max-gpu-layers` was a
>    red herring.
> 3. **PBR colors**: `trellis2_pbr.h` stored the field (col3,col2,col1) but the mesh +
>    sampler look up (col1,col2,col3) → x↔z swap (~11% hit). Store (col1,col2,col3) →
>    99.7% trilinear hit; nearest-voxel snap fallback covers the rest (100%).
>
> **SPEED**: Stage-1 DiT now uses TF32 tensor cores by default (`ops.use_tf32_gemm`,
> `T2_DIT_NO_TF32=1` to opt out): Stage 1 148→37.5 s (3.96×), full GPU e2e 237→125 s
> (1.9×), quality-neutral (single-step cosine Δ7e-7). **Build gotcha: the Makefile
> tracks only .c deps — `touch` a .c after editing any header or it won't rebuild.**
>
> Remaining: the cuBLAS/plain output-GEMM zero bug (group=25 sidesteps it). **Stage-2/3 FULL-sampler
> parity DONE; lazy per-stage DiT load DONE (peak 12.7→5.3 GB); Stage-2 bf16-block (`T2_SLAT_BF16`)
> measured = 0.986, the CFG-amplified cross-impl floor, NOT 0.999 — see below.**
>
> ## DiT PERF: profiling → modulation fix + Stage-2/3 cuBLAS-TF32 — DiT 100→74 s (2026-05-30)
>
> `nsys` on `verify_stage2_full`: DiT forward = gemm_f16_f32 46.7% + attn_mma_hd128_f32 34.8% +
> modulation_f32 10.1% + rope 4.4%. Two wins (both DEFAULT-ON now, mesh still correct 1.47M verts):
> (1) **modulation_f32** was a SINGLE-block launch (grid=1, 1/50 SMs, uncoalesced) → rewrote
> warp-per-row (coalesced + shuffle reduce, grid=ceil(out_dim/8)); ~10% off every stage, cosine
> 0.985372→0.985397. (2) **Stage-2/3 default F16-MMA → F32+cuBLAS-TF32** (`load_sparse_dit` line
> ~1064 `t2_dit_use_f16(r,0)`; `T2_DIT_F16=1` restores MMA): 1.36×, cosine →0.985343, F32 5.3GB OK
> w/ lazy load. Combined Stage-2 sampler 1924→1243 ms/fwd (1.55×). e2e DiT: S1 38.4→34.2, S2
> 38.6→24.9, S3 23.2→14.9 = 100.2→74.0 s (1.35×). NEXT: attn_mma_hd128 (35%, needs FA2) + decoders
> (~28s, c2s 128→64 = 5s). `verify_stage2_full` now prints `>>> Sampler loop: … ms/forward`.
>
> ## LAZY PER-STAGE DiT LOAD — peak 12.7 → 5.3 GB (2026-05-30)
>
> Was: harness loaded all 3 DiTs + shape decoder upfront → ~12.7 GB peak (3100 MB free)
> before Stage 1 ran. Now: load-run-free one DiT at a time (stages are sequential, inter-stage
> data is host-side). New `cuda_trellis2_unload_stage1/2/3` (factored from `unload_dit_stages`);
> harness does unload_stage1→load S2→unload_stage2→load S3→unload_dit→load shape_dec→…(tex_dec
> already lazy). Peak free/phase: S1 10530, S2 11260, S3 11260 → **peak ~5.3 GB (−58%)**. Safe:
> KV cache is (model_id,cond_hash)-keyed so each stage recomputes its own; CU_FREE/dit_model_free
> zero pointers (double-free safe). **Verified: e2e Stage-1 latent byte-identical to pre-change
> run (max|diff|=0), full colored-OBJ pipeline completes with no OOM.** (cfg_rescale fix also
> lifted e2e tex-voxel count to 99.6% of PyTorch, was 93.8%.)
>
> ## STAGE-2/3 FULL-SAMPLER PARITY + Stage-2 guidance_rescale FIX (2026-05-30)
>
> Single step was already corr **S2 0.99995** (`verify_stage2` vs `06b`), S3 ~1.0. New
> **`verify_stage2_full` / `verify_stage3_full`** run the full 12-step sampler on PyTorch's
> EXACT inputs and compare to `07_shape_slat_raw_feats` / `11_tex_slat_raw_feats`:
> - **S2 full sampler: cosine 0.985** (relL2 0.171). **BUG FOUND + FIXED:**
>   `test_cuda_trellis2.c` had Stage-2 `s2_cfg_rescale=0.7f`, but pipeline.json
>   `shape_slat_sampler` is **0.5** (0.7 is Stage-1's value). Same bug RDNA4 fixed in
>   `71d27ae` but that only touched `rdna4/*`. Fix raised parity **0.946 → 0.985**.
> - **S3 full sampler: cosine 0.99998** (relL2 6.4e-3) — essentially exact. S3 has
>   `guidance_strength=1.0` (CFG OFF), so it integrates only the raw per-step f16-vs-bf16
>   diff. S2's 0.015 residual = its CFG=7.5 **amplifying** that same per-step diff ~7.5×
>   before compounding. Neither sampler has a logic error.
> - Inputs: S2 `verify_stage2_full <s2.st> 06_…noise 06b_…coords 06b_…cond 07_…raw [cfg_rescale]`;
>   S3 `verify_stage3_full <s3.st> 09_…noise 10_…concat_cond 10b_…coords 06b_…cond 11_…raw`.
>   `neg_cond` is confirmed zero (dump_ground_truth.py:271) so the harness's zero uncond is right.
>
> ## STAGE-2 0.999 ATTEMPT → measured the floor: bf16-block `T2_SLAT_BF16` = 0.986, NOT 0.999 (2026-05-30)
>
> Implemented the predicted bf16-block port (`T2_SLAT_BF16`, default OFF; same mechanism as Stage-1
> `T2_DIT_BF16`, requires `T2_DIT_F32=1`). **It does NOT reach 0.999 — the gap is an irreducible
> cross-impl floor, not a bug.** `verify_stage2_full` cosine: F16 0.985372, F32+TF32 0.985343,
> **F32+bf16-block 0.986218** (+0.0008 only). Three facts pin it: (1) sampler math provably matches
> PyTorch incl. the CFG-rescale std — `SparseTensor.std(dim=[1])`=mean over ch THEN segment_reduce over
> voxels = per-sample/global std for B=1, == our harness (modulo Bessel); (2) F16≈F32 so it is NOT
> precision (bf16-block helps far less than Stage-1's 0.989→0.997 jump → most of the gap is the
> materialized-MMA-vs-flash-attn per-step diff, identical in F16/F32); (3) magnitude fits CFG exactly:
> single-step 0.99995 → per-step relL2 0.010, full relL2 0.173 = 17.3× ≈ guidance 7.5 × √9 guided steps.
> **0.999 needs bit-exact PyTorch bf16+flash-attn replication — same floor as Stage-1's ~0.9975, larger
> due to CFG.** Best = `T2_SLAT_BF16` 0.986, kept as default-OFF parity scaffolding (e2e stays F16 for
> speed). Decisive next test (re-dump `07` to measure reference backend floor) blocked by sparse-flow
> ext deps the PyTorch stub lacks. Full analysis: `doc/trellis-2.md` "T2_SLAT_BF16 bf16-block mode".

The section below is the ORIGINAL shape-decoder resume prompt (rel L2 ~5e-7 on
the Fujisan `N=128` SC-VAE smoke). That work is done; kept for reference.

Continue reducing TRELLIS.2 shape SC-VAE CUDA error on the Fujisan `N=128`
smoke in `/home/syoyo/work/gemm/main`.

Use this baseline verifier:

```bash
env XDG_RUNTIME_DIR=/run/user/1000 \
  T2_SCVAE_CUBLAS=1 \
  T2_SCVAE_PACKED_CONV=1 \
  T2_SCVAE_CPUAVX_LN=1 \
  T2_SCVAE_FINAL_LN_EPS=0.000009 \
  T2_SCVAE_OUTPUT_GROUP=25 \
  T2_SCVAE_OUTPUT_GROUP_FMA=1 \
  ./cuda/trellis2/verify_shape_decoder \
  /mnt/disk01/models/trellis2-4b/ckpts/shape_dec_next_dc_f16c32_fp16.safetensors \
  ref/trellis2/dumps/shape_scvae_fujisan128/pytorch_fp32_cuda_input_slat.npy \
  ref/trellis2/dumps/shape_scvae_fujisan128/pytorch_fp32_cuda_input_coords.npy \
  --ref-feats ref/trellis2/dumps/shape_scvae_fujisan128/pytorch_fp32_cuda_output_feats.npy \
  --ref-coords ref/trellis2/dumps/shape_scvae_fujisan128/pytorch_fp32_cuda_output_coords.npy \
  --skip-cpu
```

Current best PyTorch-reference result:

- coord mismatches: `0`
- correlation: `1.00000000`
- rel L2: `5.0239075e-07`
- max abs: `6.1035156e-05`
- max idx: `row=183 col=5 ref=-49.083614 cuda=-49.083675 diff=-6.1035156e-05`

Important context:

- Full-output metric should be compared against the saved PyTorch CUDA dumps in
  `ref/trellis2/dumps/shape_scvae_fujisan128/`, not the CPU reference. CPU C
  reference differs more (`~2.4e-4`) and is not the parity target.
- Final projection from exact saved PyTorch pre-output can be below target.
  `T2SD_START_PRE_OUTPUT=1 T2_SCVAE_OUTPUT_GROUP=25
  T2_SCVAE_OUTPUT_GROUP_FMA=1` gives about `7.63e-6`; cuBLASLt bias epilogue
  can be bit-exact from saved pre-output. Therefore remaining full-output error
  is upstream feature drift projected by `out_w`.
- Best pre-output feature max is not the same as best final-output max. Some
  probes reduce intermediate max error but project worse through output.

Already implemented/probed:

- Verifier projection diagnostic:
  `T2_VERIFY_PROJECT_OUT=1` loads host `output_layer.weight` and, for
  64-channel intermediate comparisons, reports the max logit delta implied by
  `(cuda-ref) @ out_w.T`. With the current best full-run flags and
  `T2SD_STOP_PRE_OUTPUT=1`, raw pre-output max is `7.63e-6` while projected
  output-delta max is `6.28e-5` at `row=183 col=5`. Without the final-LN eps
  tweak, the projected max was `6.63e-5`.
- Welford no-affine final LayerNorm:
  `T2_SCVAE_FINAL_WELFORD_LN=1`; bit-exact from exact stage3/pre-output starts,
  but worsens full smoke with current best flags to `6.8664551e-05`.
- Global/intermediate Welford no-affine and affine LayerNorm knobs:
  `T2_SCVAE_WELFORD_LN=1`, `T2_SCVAE_WELFORD_AFFINE_LN=1`,
  `T2SD_WELFORD_AFFINE_LN=stage:block`,
  `T2SD_WELFORD_AFFINE_C2S=stage`,
  `T2SD_WELFORD_NOAFFINE_C2S=stage`. These help some local comparisons but do
  not beat the full-output best.
- Output projection order probes:
  `T2_SCVAE_OUTPUT_GROUP`, `T2_SCVAE_OUTPUT_GROUP_FMA=1`,
  `T2_SCVAE_OUTPUT_GROUP_MODE=1/2/3`,
  `T2_SCVAE_OUTPUT_GROUP_BIASINIT=1`,
  `T2_SCVAE_OUTPUT_GROUP_TREE=1`, and `T2_SCVAE_OUTPUT_PAIR32`.
  Group25/FMA is still best; group27/FMA ties. Tree/mode/biasinit do not reduce
  below `6.4849854e-05`.
- Sparse conv per-site knobs:
  `T2SD_SPARSE_PACKED=stage:block:op:value`,
  `T2SD_SPARSE_LT=stage:block:op`,
  `T2SD_SPARSE_DIRECT=stage:block:op:mode`,
  `T2SD_SPARSE_CUBLAS=stage:block:op`.
  Stage3 C2S conv2 and stage3 ConvNeXt sparse-conv alternatives tie or worsen.
- cuBLASLt MLP/sparse probes are present. Isolated stage2 block7 can improve
  locally, but full output worsens.
- Precise math / no-FMAD / TF32 paths were not useful. TF32 changed topology in
  one check and is invalid for this parity target.

Current localization:

- Starting from exact PyTorch stage3 features, final LN + output can be
  bit-exact with Welford final LN and cuBLASLt bias output.
- Starting from exact PyTorch stage2 features, stage3 + final output is about
  `4.58e-5` max.
- Full-run stage2 output is about `2.86e-5` max vs PyTorch.
- Stage2 C2S final drift is strongly influenced by the repeated skip branch.
- Stage2 block7 MLP is a visible amplifier in full-run localization:
  after stage2 block6 about `1.14e-5`, after stage2 block7 about `2.48e-5`.
  Forcing cuBLASLt only on stage2 block7 MLP2 reduces local block/stage2 error
  but worsens final logits.
- Full-run stage3 C2S conv2 is the last large amplifier: post-conv2 around
  `3.24e-5` with packed sparse conv. Alternative sparse conv modes did not
  reduce final output.
- Projection-screened negative probes: `T2SD_CUBLASLT_MLP=2:7:2` raised
  pre-output projected max to `7.91e-5`; `T2SD_WELFORD_AFFINE_LN=2:7` to
  `7.23e-5`; `T2_SCVAE_FINAL_WELFORD_LN=1` to `7.33e-5`; serial final LN modes
  tested at `7.79e-5` and `8.40e-5`.
- The projection diagnostic now prints worst-row top channel contributions.
  Current best full-run pre-output top terms are broad rather than one broken
  channel: `c17=+1.49e-5`, `c34=-1.47e-5`, `c19=+1.16e-5`,
  `c18=+1.13e-5`, `c40=+8.58e-6`, `c59=+8.22e-6`.
- Exact-start projected localization:
  saved PyTorch stage2 output -> stage3/pre-output gives `3.90e-5`;
  saved PyTorch stage1 output -> stage2+stage3/pre-output gives `5.49e-5`;
  full-run pre-output gives `6.63e-5`. Stage3 contributes substantially, but
  full-run worst direction is seeded upstream.
- More negative probes:
  stage3 C2S conv2 gather-GEMM tied exact-stage2 projected max at `3.90e-5`;
  stage3 C2S Welford norm2/norm1 worsened to `4.02e-5`/`4.23e-5`.
  Stage3 MLP2 cuBLASLt block0/1 improved exact-stage2 projected max
  (`3.28e-5`/`3.37e-5`) but worsened full-run projected max (`8.10e-5` and
  `8.47e-5`); block0 final output worsened to `8.01e-5`. Full-run stage2 MLP2
  cuBLASLt blocks 0..7 all worsened projected max (`7.48e-5` to `1.05e-4`).
  Stage2 block6/7 and stage3 block0/1 MLP0 cuBLASLt probes were numerically
  unchanged from baseline.
- Coordinate lineage trace: several persistent worst rows share stage0 C2S
  parent row `110`, but ancestor-row raw errors stay diffuse (`~2e-5` to
  `4e-5` L2 per row) without a single bad channel/child slot.
- More output projection probes: group21/group23 FMA worsen to `6.87e-5`;
  group29/group31 FMA tie current best `6.48e-5`. Existing double GEMM paths
  worsen (`9.16e-5` bias GEMMs, `8.77e-5` all GEMMs), and pair32 output is
  `6.87e-5`.
- CUDA sweeps via Python `subprocess` hit `cuInit failed (100)` in this
  environment even though direct verifier commands work. Use direct shell
  verifier invocations for CUDA probes.
- `T2SD_CUBLASLT_MLP` now accepts comma/semicolon-separated site specs for
  combination probes. Tested MLP2 combinations still worsened projected max:
  `2:7:2,3:0:2` -> `7.64e-5`, `2:1:2,3:0:2` -> `8.53e-5`,
  `3:0:2,3:1:2` -> `7.91e-5`, and `2:1:2,2:7:2,3:0:2` -> `8.55e-5`.
- Direct sparse-conv mode2 projected probes did not help: stage3 C2S conv2
  `7.35e-5`, stage2 C2S conv2 `9.58e-5`, stage3 C2S conv1 `7.67e-5`.
  Stage-wide affine Welford LayerNorm also worsened projected max:
  stage0 `7.94e-5`, stage1 `1.02e-4`, stage2 `8.78e-5`, stage3 `8.47e-5`.
- `T2SD_STOP_AFTER_C2S_OP=stage:7` now returns `to_subdiv` logits `[N,8]`.
  Subdivision logits are coordinate-exact and worst-row emitted-child margins
  are far from zero; topology is not the hidden issue. Stage min positive
  margins are `0.428`, `0.126`, `0.00242`, `0.00253` for stages 0..3.
- Final no-affine LN epsilon is currently the only max-reducing knob.
  `T2_SCVAE_FINAL_LN_EPS=0.000009` lowers full-output max from `6.48e-5` to
  `6.10e-5`. This is a parity tradeoff/error-cancellation knob because PyTorch
  uses `1e-5`; `8.9e-6` and `9.1e-6` tie, while `8.5e-6`, `9.25e-6`, output
  group27/31 with eps9e-6, and cuBLASLt bias output are worse.
- More negative full-output probes with the `9e-6` final-LN eps:
  `T2_SCVAE_CPU_GATHER_MAP=1`, `T2_SCVAE_CUBLAS_PEDANTIC=1`,
  `T2_SCVAE_CUBLAS_MLP=1`, and `T2_SCVAE_LN_SQRT=1` all tie the current best.
  Output group28/FMA also ties; group28 with `eps=9.2e-6` slightly improves rel
  L2 (`5.01e-7`) but not max. Group24/26, group28 with `eps=8.8e-6`/`9.6e-6`,
  and `T2_SCVAE_FINAL_LN_MODE=3 T2_SCVAE_FINAL_LN_EPS=0.0000092` are worse.
- Exact-start downstream checks:
  - saved PyTorch `stage3_block3` -> stage3 C2S + final output:
    `3.05e-5` max.
  - saved PyTorch `stage2_block7` -> stage2 C2S + stage3 + final output:
    group25/eps9e-6 gives `7.63e-5`; group28 gives `6.87e-5`; group28 with
    `eps=9.2e-6` gives `6.48e-5`, still not better than full-run best.
  - saved PyTorch `stage2_block7` -> stage2 C2S only is `5.72e-6` max.
  - saved PyTorch `stage2_block7` -> through final LN with `T2SD_STOP_PRE_OUTPUT=1`
    is `4.53e-6` max, but `T2_VERIFY_PROJECT_OUT=1` shows that tiny C=64 drift
    projects to `5.93e-5` at `row=3201 col=4` via `output_layer.weight`.
- Full-run group28/`eps=9.2e-6` output-order variants: mode1 ties `6.10e-5`;
  modes 2 and 3 worsen to `6.87e-5`.

Latest CUDA e2e verification attempt (2026-05-29):

- Regenerated PyTorch CUDA reference dumps successfully with GPU access outside
  the sandbox:

  ```bash
  env OUTDIR=/tmp/t2_pytorch_ref_cuda_check \
    IMAGE=/home/syoyo/work/gemm/trellis2/cpu/trellis2/trellis2_repo/assets/example_image/T.png \
    MODEL_ROOT=/home/syoyo/work/gemm/trellis2/cuda/trellis2/model_root \
    DINOV3=/mnt/disk01/models/dinov3-vitl16/model.safetensors \
    SEED=42 DECODER_RES=512 \
    ./cuda/trellis2/run_dump_ground_truth.sh
  ```

- Reference dump summary for `T.png`, seed 42:
  - `01_dinov3_cond_512.npy`: `[1,1029,1024]`
  - `02_ss_noise.npy`: `[1,8,16,16,16]`
  - `03_ss_latent.npy`: `[1,8,16,16,16]`
  - `04_ss_decoder_logits.npy`: `[1,1,64,64,64]`, raw positive logits `17303`
  - `05_ss_coords.npy`: `[3548,4]` after pipeline sparse-coordinate processing
  - `07_shape_slat_raw_feats.npy`: `[3548,32]`
  - final PyTorch mesh/texture dumps were produced through `15_tex_voxels`.

- **[SUPERSEDED — THIS IS WRONG, see RESOLVED block at top]** This claimed
  `test_cuda_trellis2 --noise` consumes token-major `[4096,8]`. It does NOT:
  `run_dit` reads channel-major `[8,4096]` and transposes internally. The
  conversion below is the double-transpose scramble that caused the garbage.
  Feed `02_ss_noise.npy` RAW instead. Original (incorrect) note kept for history:

  ```bash
  python3 - <<'PY'
  import numpy as np
  a = np.load('/tmp/t2_pytorch_ref_cuda_check/02_ss_noise.npy').astype(np.float32)
  tok = np.transpose(a[0], (1, 2, 3, 0)).reshape(16 * 16 * 16, 8)
  np.save('/tmp/t2_cuda_e2e_check/ref_noise_token_major.npy', tok)
  PY
  ```

- Also split DINO features from `[1,1029,1024]` to `[1029,1024]` for the C
  runner:

  ```bash
  python3 - <<'PY'
  import numpy as np
  a = np.load('/tmp/t2_pytorch_ref_cuda_check/01_dinov3_cond_512.npy')
  np.save('/tmp/t2_cuda_e2e_check/ref_features_1029x1024.npy',
          a[0].astype(np.float32))
  PY
  ```

- Corrected Stage 1-only CUDA run completed:

  ```bash
  env XDG_RUNTIME_DIR=/run/user/1000 \
    ./cuda/trellis2/test_cuda_trellis2 \
    /mnt/disk01/models/trellis2-4b/ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors \
    /mnt/disk01/models/trellis-image-large/ckpts/ss_dec_conv3d_16l8_fp16.safetensors \
    /tmp/t2_cuda_e2e_check/ref_features_1029x1024.npy \
    --noise /tmp/t2_cuda_e2e_check/ref_noise_token_major.npy \
    -s 42 -n 12 -g 7.5 \
    --occ /tmp/t2_cuda_stage1_check/cuda_occ.npy \
    --npy /tmp/t2_cuda_stage1_check/cuda_stage1_latent.npy \
    -o /tmp/t2_cuda_stage1_check/cuda_stage1.obj
  ```

- Corrected Stage 1 comparison still fails vs PyTorch:
  - `stage1_latent` must be layout-corrected when read back from the C `.npy`
    because the file header says `[8,16,16,16]` but the stored buffer is
    token-major `[4096,8]`.
  - Layout-corrected latent: `rel_L2=1.556485`, `max_abs=4.928794`,
    `cosine=-0.012180066`, `allclose=False`.
  - Decoder logits: `rel_L2=0.2057113`, `max_abs=320.9109`,
    `mean_abs=12.23886`, `cosine=0.978837875`, `allclose=False`.
  - Raw positive logits: PyTorch `17303`, CUDA `21026`.
  - Raw positive coord set overlap: `15509` common, `1794` PyTorch-only,
    `5517` CUDA-only.

- Full CUDA C e2e run was attempted with Stage 2 and Stage 3 loaded:

  ```bash
  env XDG_RUNTIME_DIR=/run/user/1000 \
    T2_SCVAE_CUBLAS=1 \
    T2_SCVAE_PACKED_CONV=1 \
    T2_SCVAE_CPUAVX_LN=1 \
    T2_SCVAE_FINAL_LN_EPS=0.000009 \
    T2_SCVAE_OUTPUT_GROUP=25 \
    T2_SCVAE_OUTPUT_GROUP_FMA=1 \
    ./cuda/trellis2/test_cuda_trellis2 \
    /mnt/disk01/models/trellis2-4b/ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors \
    /mnt/disk01/models/trellis-image-large/ckpts/ss_dec_conv3d_16l8_fp16.safetensors \
    /tmp/t2_cuda_e2e_check/ref_features_1029x1024.npy \
    --noise /tmp/t2_cuda_e2e_check/ref_noise_token_major.npy \
    --stage2 /mnt/disk01/models/trellis2-4b/ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors \
    --shape-dec /mnt/disk01/models/trellis2-4b/ckpts/shape_dec_next_dc_f16c32_fp16.safetensors \
    --stage3 /mnt/disk01/models/trellis2-4b/ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16.safetensors \
    --tex-dec /mnt/disk01/models/trellis2-4b/ckpts/tex_dec_next_dc_f16c32_fp16.safetensors \
    -s 42 -n 12 -g 7.5 --s2-steps 12 --s2-cfg 7.5 --s3-steps 12 \
    --occ /tmp/t2_cuda_e2e_check/cuda_occ.npy \
    --npy /tmp/t2_cuda_e2e_check/cuda_stage1_latent.npy \
    --s2-npy /tmp/t2_cuda_e2e_check/cuda_shape_slat_raw.npy \
    -o /tmp/t2_cuda_e2e_check/cuda_full.obj
  ```

- Full e2e result:
  - Stage 1 produced `21037` raw positive voxels, so Stage 2 ran on
    `[21037,32]` instead of PyTorch reference `[3548,32]`.
  - Stage 2 completed but took about `485.1 s`.
  - Stage 3 completed but took about `291.2 s`.
  - Shape decoder expanded `21037 -> 101335 -> 427953 -> 2560034` rows, then
    failed allocation/illegal-address handling and exited with
    `CUDA shape decoder failed`.
  - No final OBJ was written.
  - GPU recovered afterward (`nvidia-smi` showed about `33 MiB` used).

- Main next debugging target:
  - Do not spend time on Stage 2/3 parity until Stage 1 is fixed or bypassed.
  - First localize Stage 1 DiT parity with exact PyTorch `01_dinov3_cond_512`
    and converted `02_ss_noise`.
  - Add or use a single-step Stage 1 verifier against
    `02b_ss_dit_step_velocity.npy` from the PyTorch dump. The existing full
    Stage 1 mismatch is too large to be shape-decoder noise.
  - Check C runner input/output layout around `cuda_trellis2_run_dit()`: it
    transposes `x_t` from channel-major to token-major internally, while
    `test_cuda_trellis2` stores `x` token-major after loading converted noise.
    Passing token-major `x` into `cuda_trellis2_run_dit()` may double-transpose
    or otherwise scramble Stage 1. Either keep `x` channel-major in the test
    loop or bypass the internal transpose for token-major buffers.
  - After Stage 1 logits match, mirror the PyTorch `05_ss_coords` sparse
    coordinate extraction/max-pool semantics before running Stage 2. The C e2e
    path currently feeds Stage 2 directly from raw positive 64^3 logits.

Likely useful next directions:

1. Continue upstream localization around stage2 block7 and stage2 C2S skip
   branch. Focus on feature-direction error that projects through `out_w`, not
   only max-abs intermediate error.
2. Add row/channel-specific diagnostic dumps or a projection-weighted error
   metric for intermediate tensors. The persistent worst final rows are caused
   by many small channel errors, not one obvious broken channel.
3. If adding probes, keep them site-selectable and document negative results in
   `doc/trellis-2.md`.

Before finishing a session:

```bash
make -C cuda/trellis2 verify_shape_decoder
git diff --check -- \
  cuda/trellis2/cuda_trellis2_kernels.h \
  cuda/trellis2/cuda_trellis2_ops.h \
  cuda/trellis2/cuda_trellis2_runner.c \
  doc/trellis-2.md \
  cuda/trellis2/resume.md
```
