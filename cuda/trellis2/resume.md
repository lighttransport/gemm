# TRELLIS.2 Shape Decoder Resume Prompt

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
