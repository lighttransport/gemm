# praxis-1-droid on one 16GB CUDA GPU (<8GB target) — design plan

Status: design + tiny-model implementation (see `Status update` below). The checkpoints (`runwayml/praxis-1-droid`, `-nvfp4`, `-base`) are gated on
Hugging Face; the `lte-ai` account has no access, so `config.json` and the safetensors index are
unread. Dims below are inferred from `praxis-1-oss` constructor defaults + README (54.40 GB computed vs
54.4 stated; routed 51.34 vs 51.3). Verify against config.json once access is granted.

## Done (phase 1)
- Qwen3.5-2B text encoder: `cuda/llm/test_qwen35_st` (cond = layer-22 pre-norm, rel-L2 1.2e-4).
- Wan2.2 VAE: `cuda/praxis/` (encode/decode, policy latent 192ch, 18/18 parity checks, 3.2GB peak @21x256x256).
- PyTorch refs: `ref/praxis/` (text, VAE). Weights: `/mnt/disk1/praxis`.

## Model (inferred)
- Encoder tower 14 blocks (0,1 dense SwiGLU 11520; 2..13 MoE), decoder tower 6 blocks (0 dense, 1..5 MoE; adaLN
  modulators from encoder cond). dim 3072, 24 Q / 8 KV heads, head_dim 128, gated attn (sigmoid per head), q/k RMSNorm,
  partial RoPE (64 of 128), base 2.6e7, 1-D positions (ROW_STRIDE 360, FRAME_STRIDE 129600).
- MoE: 128 routed experts top-8 (hidden 1280, gate|up 3072x2560, down 1280x3072) + 1 shared expert; fp32 sigmoid
  router with selection-only bias; out = shared + 2.5*sum(gate*expert). 17 MoE layers.
- Action expert: 16 blocks dim 1024, even blocks cross-attend to 8 encoder taps (0,2,4,6,7,9,11,13) + globals.
- Sampler: 2 Euler steps, t_shift 12.5, window_joint, video CFG 4.0 -> 3 backbone passes/call (1 with pair_cfg sharing).
- Sizes bf16: routed 51.3 GB; everything else 3.07 GB (attn 1.0, dense 0.64, shared 0.4, modulators 0.23, I/O 0.15, expert 0.63).

## Key constraint
Every pass has 1.5k-10k tokens x top-8 of 128 experts => essentially ALL experts of a layer are touched every pass.
MoE sparsity does not reduce bytes read per pass; it only helps if experts are made resident. Per-pass routed traffic:
NVFP4 14.4 GB (0.85 GB/layer), bf16 51.3 GB. Hence "<8GB" = streaming experts from pinned host RAM, bandwidth bound:
~14.4 GB x 3 passes ≈ 43 GB/call (≈1.6 s @ 28 GB/s PCIe5 x8-ish, ≈2.5+ s @ PCIe4). Reduce with:
(a) cfg_video_scale=1.0 -> 2 passes; (b) pair cond/uncond encoder passes so each layer's experts are read once;
(c) pin a hot-expert set resident in leftover VRAM (route-frequency profile from real episodes).

## Design
1. **Weights**: offline convert (new `cuda/praxis/convert_praxis.py`, torch only for reading bf16) to a repack file:
   non-routed tensors -> F16/FP8 resident blob; routed experts -> NVFP4 (E2M1 + e4m3 block scale/16 + fp32 global/matrix,
   exactly praxis `quantization/nvfp4.py` recipe; reuse `cuda/fp4` sm_120a block-scale mma kernels) laid out per layer,
   per expert contiguous (≈6.6 MB) for large H2D copies. A bf16 (fallback, 3.0 GB/layer) variant for accuracy bring-up.
2. **Runner `cuda/praxis/cuda_praxis_runner.h`** (NVRTC, qimg-style): reuse `cuda_llm_runner` attention/RMSNorm/SwiGLU/
   cuBLAS helpers; new kernels: gated-GQA attn with chunk-causal block-sparse mask (teacher forcing), partial RoPE with
   high/low position split, router (fp32 sigmoid+bias top-8), expert-grouped dispatch GEMM (sorted rows per expert,
   NVFP4 grouped GEMM, cf. `cuda/llm` mmq_moe), adaLN modulate, Fourier timestep emb, action-expert (GEGLU, cross-attn to taps).
3. **Expert streaming**: 2-3 slot ring of layer buffers (0.85 GB NVFP4 each) on a copy stream from pinned host memory;
   layer L+1 prefetch overlaps layer L compute (proven pattern in cuda/qimg weight-streaming pipeline). Expert-grouped
   execution so each expert is used once per pass.
4. **VRAM budget (NVFP4, K'=2)**: resident non-routed 8-bit ≈1.5 GB (F16 3.1 GB) + expert ring 2.6 GB + KV/taps 0.5-1 GB +
   workspace 1-2 GB + VAE 1.4 GB (or offload) ≈ 7-8 GB; Qwen3.5 text encoder run once per prompt (cache cond, free 4.6 GB).
5. **Pipeline**: text(Qwen, cached) -> VAE encode (anchor/context) -> 2 Euler steps {encoder pass (+uncond), decoder pass,
   action expert on taps} -> VAE decode (optional) -> actions un-normalised. Encoder ctx-cache (reuse K/V of globals+clean
   across Euler steps) as in praxis, disabled for NVFP4 in reference — validate separately.

## Milestones / validation (each vs PyTorch dumps from praxis-1-oss, rel-L2/cos like ref/praxis/compare.py)
M0 get access + config.json; hash-check shards; dump reference tensors (needs an 80GB GPU or CPU offload run of
  praxis `PraxisPipeline`; per-block encoder/decoder/taps/actions with a fixed seed) -> `ref/praxis/gen_backbone_ref.py`.
M1 weight converter + loader (bf16 resident-all-in-host), single dense block parity.
M2 attention+RoPE+masks parity (encoder block 0/1), then MoE block (bf16 experts streamed) parity.
M3 full encoder + decoder + action expert parity on one Euler step; then full 2-step window_joint call.
M4 NVFP4 experts (sm_120a) parity vs praxis NVFP4 path; measure VRAM (target <8GB peak) and latency.
M5 e2e policy call with VAE+Qwen from CUDA; compare actions to reference (task-level sanity, not bitwise).

## Risks
- Config unknown until access; dims may differ from inference above.
- PCIe bandwidth dominates latency (seconds/call, not the H100's 0.35-0.5 s).
- NVFP4 loses accuracy on hard tasks in the README's evaluation; prefer FP8/bf16 streaming where bandwidth allows.
- Reference dumps need >=54GB memory (CPU RAM offload feasible but slow).

## Status update (tiny-model runner)
`cuda/praxis/cuda_praxis_runner.h` + `test_cuda_praxis.c` (`make -C cuda/praxis test_praxis`) implement the encoder/decoder
towers with MoE, the action expert and the 2-step Euler window_joint sampler, config driven, and match praxis' tiny random
policy (fp32 reference, `ref/praxis/gen_backbone_ref.py`) to rel-L2 3e-7 (backbone, video) / 3e-4..6e-4 (action expert,
sampled actions). Expert streaming through a pinned-host -> N-slot device ring is implemented and unit-tested (bit-identical,
routed device residency bounded by the ring, byte/peak report). Milestones above: M1 (loader, single block), M2 (attn/RoPE/masks,
MoE), M3 (full encoder+decoder+expert+2-step call) reached on tiny; M4 bf16 streaming reached, NVFP4 not. Still open for the real
checkpoint: gated config/weights, fast GEMM/attention (the current kernels are correctness-first), chunked MoE dispatch, paired
CFG passes, NVFP4 experts, and end-to-end Qwen+VAE+policy wiring. Details: `cuda/praxis/README.md`.

## Status update 2 (performance, synthetic full-size, NVFP4)
Fast path (cuBLAS bf16 GEMMs, block-sparse flash attention, device MoE dispatch), mid-size torch parity (dim 256/hd 64 and dim 384/hd 128), NVFP4 routed experts
(own sm_120a grouped block-scale GEMM, CPU packer bit-matching the nvfp4.py recipe) and sampler shortcuts (paired CFG encoders, skip-last-video) are implemented.
Measured on a synthetic random full-size model (K'=2, 4020 enc / 9988 dec tokens, ring 2, RTX 5060 Ti, PCIe4 x8 14.4 GB/s raw):
bf16 banks 3.7 s/pass, 14.7 s per cfg-4 call, 11.3-12 GB peak (misses <8 GB, stream-bound at 13 GB/s);
NVFP4 banks 1.5 s/pass, 5.8 s per cfg-4 call (4 passes), 4.7 s paired, 3.7 s skip-last-video, **3.2 s both**, peak 7.1-7.6 GB (7.3-7.8 per cuMemGetInfo), compute-bound
(2.9 s of that without any H2D). The 14.4 GB/pass streaming (~1.0 s) is now hidden behind compute. Correctness: mid-size fast path vs torch 0.5-0.7% (video), NVFP4 e2e 1-2%,
per-MoE-layer NVFP4 error 0.23 rel-L2. Not validated: any real-checkpoint numerics, quality of NVFP4 on trained weights, full-size outputs (no oracle).
Details and caveats: `cuda/praxis/README.md`.


## Status update 3 (sharded loading, ctx cache, block-sparse expert attention, end-to-end driver)
Everything below is validated on tiny/mid/mid128/mid2048 or the synthetic full-size model; nothing here depends on the gated checkpoint. Details/commands: `cuda/praxis/README.md`.
- **Sharded safetensors + pack cache**: `model.safetensors.index.json` + N shards, layer-by-layer pack/pin with file pages dropped after use; bit-identical to the single-file load; host RSS growth 473 MB on a 1.6 GB bf16 / 405 MB packed test
  checkpoint vs 1981 MB without dropping (asserted bound: packed + 3 layers + weights + 96 MB); `PRAXIS_PACK_CACHE` makes the NVFP4 load 7x faster (2.7 s -> 0.4 s at test size; real-size 15 s -> a 14.4 GB read, not measured). Untested at 51 GB/17 shards (disk 23 GB free).
  Fixed a `praxis_free` leak (all device weights stayed allocated).
- **Encoder ctx cache** (opt-in): exact semantics of `_core_ctx_cached_gen` (only noised rows recomputed, context K/V cached per guidance branch); parity 4e-3..6e-3 video / 7e-4 actions vs uncached; **-10.5% compute-only, -1.3..-5% with real PCIe streaming**
  (a pass is stream-bound, 0.7 s of bank traffic for the 12 encoder layers); +0.2..0.6 GB. Not a lever on this 16 GB/PCIe4 x8 card; would matter with resident banks.
- **Block-sparse expert attention**: interval-mask flash kernel, 16-34x faster than the dense fp32 kernel at KV 4k-9k (2.29 -> 0.14 ms, 10.6 -> 0.31 ms), expert pass 48 -> 32 ms (K'=2), 156 -> 74 ms (K'=4), parity unchanged (2e-3 bf16 kernel error).
- **End-to-end driver** `praxis_e2e`: Qwen3.5-2B (child process, text cached per prompt) -> tile canvas -> Wan2.2 VAE -> policy sampler -> 24 normalised action rows (+ optional VAE decode). Validated vs PyTorch on mid2048 (text 1.2e-4, latent 1.6e-3, video 7e-3, actions 1e-3).
  Full-size synthetic sequence, per-stage device peak: Qwen 3.7 GB, VAE encode 3.2 GB, policy 7.3 GB (paired NVFP4, ring 2), VAE decode 5.9 GB -> **max instant 7.45 GB incl. context (< 8 GB); the VAE is freed before the policy loads** (reload costs ~3.5 s / 9 s).
- Remaining without the checkpoint: action un-normalisation (needs NormStats + rig frame), real-weight NVFP4 quality, 51 GB sharded load at real size, a C tokenizer.


## Status update 4 (REAL praxis-1-droid checkpoint)
The checkpoint is now available (`/mnt/disk1/praxis/praxis-1-droid`). Details and tables: `cuda/praxis/README.md` ("REAL checkpoint").
- Loader: all 567 index keys used (none missing/unused); config semantics validated at load; NVFP4 pack from the 17 shards 259 s cold (disk-bound) -> 14.44 GB pack cache -> 22-40 s load; host RSS 14.5-17.4 GB.
- End to end on real weights (Qwen -> tiles with the config's antialias fit rule -> VAE -> NVFP4 policy -> un-normalised rows): K'=2, cfg video 4: 5.3 s (ring 2) / 6.0 s (ring 1); skip-last-video 3.5 s;
  paired+skip 2.8 s (timing only); peak 6.1-7.3 GB own accounting (+0.13 GB ctx) with the VAE and Qwen sequenced out: <8 GB met. Matches the synthetic numbers.
- Parity: the torch bf16 reference is noisier than our runner (vs the torch fp32 oracle: dec_vel 7.0e-2 torch-bf16, 2.0e-2 ours bf16-expert, 7.8e-2 ours NVFP4; sampled actions 0.50 / 0.23 / 0.37).
  NVFP4 on trained weights = ~4x the bf16-path error in the velocity, at the level of the bf16 reference; no runner bugs found, one harness bug (text_u length) fixed.
- Open: DROID-episode run and task-level NVFP4 quality, client-frame pose conversion, bf16 experts cannot meet 8 GB (ring 2 x 2.9 GB).

## Status update 5 (NVFP4 accuracy on the real weights: diagnosis, W4A16, FP8 banks, expert-group streaming)
Details, tables and commands: `cuda/praxis/README.md` ("NVFP4 accuracy on the real weights"). Oracles: torch fp32 of the same weights (`tmp/praxis_ref/real_fp32`, and `real_qat_fp32` for the QAT checkpoint
`runwayml/praxis-1-droid-nvfp4`, now accessible: different weights, same config, 54.4 GB, downloaded + SHA256-verified to `/mnt/disk01/praxis/praxis-1-droid-nvfp4`).
- **Diagnosis** (per-MoE-layer attribution on real activations, `cuda/praxis/moe_attr.py`): the NVFP4 loss is weight quantisation (FP4 weights alone ~0.10 routed-output rel-L2 per layer, +dynamic FP4 activations -> ~0.14); the
  down bank is uniformly ~0.08 (FP4 round trip 9.4% on Gaussian-like weights = Lloyd-Max limit, FP8 per channel: 0.02), the gate_up bank is layer dependent with enc13 (0.22), enc10 (0.12), enc3 (0.09) far above the
  rest (0.01-0.06). Router stays fp32 on the unquantised residual; route flips are induced upstream (same top-8 set vs fp32: bf16 64%, mixed 48%, W4A4 31%). Per-expert activation scale, the calibrated FC1 scales (praxis applies them
  only on its SM100 fused-SwiGLU backend) and finer scale search do not help; our packer is bit-exact with the praxis recipe on a real expert (`check_pack.py`). QAT weights show the same per-layer error pattern.
- **Fixes**: `PRAXIS_EXPERT_FMT=nvfp4a16` (f16 activations, W4A16 f16-mma kernel with in-register `cvt.rn.f16x2.e2m1x2` dequantisation, +13% latency), per-matrix FP8 banks (`PRAXIS_FP8_BANKS`), expert-group streaming
  (`PRAXIS_SLOT_MB`: a 1.44 GiB layer streams as two groups through the 1.02 GiB ring slot, bit-identical), `PRAXIS_PACK_SCALE=mse` (opt-in), preset **`PRAXIS_EXPERT_FMT=mixed`** (FP8 down banks everywhere, FP8 gate_up for encoder
  blocks 3/10/13, f16 activations).
- **Result** (dec_vel vs the fp32 oracle of the same weights): base weights W4A4 7.8e-2 -> W4A16 5.3e-2 -> mixed 3.2e-2 (bf16 experts 2.0e-2); QAT weights 5.6e-2 -> 4.5e-2 -> 3.6e-2 (bf16 1.8e-2). Budget: ring slot 1021.5 MiB (was 810),
  18.2 GiB streamed per pass (was 13.5), **own peak 7.62 GiB + 0.13 GiB ctx < 8 GiB**; latency 7.2-7.5 s plain / 4.2 s paired+skip (was 5.5 / 2.85); W4A16 alone 6.25 / 3.5 s. The old default (`PRAXIS_NVFP4=1`, W4A4) is unchanged.
- Not done: task-level quality of any variant; FP8 gate_up for decoder layers (3 token chunks -> groups re-streamed per chunk); a W4A4-trained-QAT-matched path for fidelity to the QAT forward (the QAT weights were trained
  with FP4 activations, `nvfp4` is what the authors validated); FP6/E2M3 banks (0.75 B/weight would fit gate_up of the sensitive layers in the 1.02 GiB slot without groups).
