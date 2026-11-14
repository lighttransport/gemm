# cuda/praxis

Standalone CUDA (NVRTC, no nvcc) code for praxis-1.

## Paths and environment
No machine-specific paths are hardcoded; defaults come from two environment variables:
- `PRAXIS_ROOT` (default `$HOME/praxis`): models, weights and reference code, one sub-directory each: `praxis-1-droid`, `praxis-1-oss` (the model's own repo), `qwen3.5-2b`, `wan2.2-vae`, `wan2.2-ti2v`, `cosmos3-edge-droid`, `cosmos3-vae-wan`, the generated test models `tiny`, `mid`, `mid128`, `mid128_sh`, `mid2048`, `mid_sh`, and the pack caches `praxis-1-droid.nvfp4.packcache*`. Symlink `~/praxis` to wherever the data lives, or export `PRAXIS_ROOT`.
- `PRAXIS_OUT` (default `<repo>/tmp`): scratch, dumps and sim outputs (`praxis_sim*/`, `praxis_server_tmp`, ...). `tmp` is typically a symlink to a large disk.

Makefiles take the same names (`PRAXIS_ROOT ?= $(HOME)/praxis`), the Python tools use them as `--flag` defaults, C programs read them through `common/praxis_paths.h`, and the `eval/*.sh` scripts derive the repo root from their own location. The `--model/--policy/--vae/--qwen/--ckpt/--config` flags always override the defaults.

## Wan2.2 VAE
`cuda_wan22_vae.h`, `test_cuda_wan22_vae.c`; `make -C cuda/praxis test` (18/18 PASS), `bench`, `test_tiled`, `bench_big`.

Rewritten (no cuBLAS, no im2col buffer): every conv is an NVRTC implicit-GEMM on tensor cores (mma.sync m16n8k16, fp16 operands, cp.async pipeline,
64x64 CTA tile, split-K for small M, bias/residual/fp16-or-transposed output fused). RMSNorm+SiLU writes the fp16 conv input directly; causal caches are fp16.
Attention = tensor-core QK^T / PV (V written transposed by the qkv epilogue) + fp32 row softmax, scores materialised per frame (not a flash kernel).
Default accumulation is strict fp32. `WAN22_VAE_ACC16=1` accumulates fp16 inside a 32-wide k tile (flushed to fp32 every k tile): ~1.4x faster (decode 612 ms vs ~850 ms at 21x256x256) but it raised the downstream Cosmos3 policy action error from 1.3e-2 to 2.1e-2, so it is opt-in.

Measured on RTX 5060 Ti (GPU otherwise idle; old = previous im2col+cuBLAS build, same session):
| | old | new (acc16 default) | new strict fp32-acc |
|---|---|---|---|
| encode 21x256x256 | 342 ms | 141 ms (2.4x) | ~200 ms (1.7x) |
| decode 21x256x256 | 1287 ms | 612 ms (2.1x) | ~850 ms (1.5x) |
| first encode call | 917 ms | 141 ms (no warm-up) | |
| weight load | 1.6-4.9 s | 0.14-0.44 s (fp16 cache hit; cold convert ~2.6 s writes `model.w22v2.cache`) | |
Parity (`make test`, `make bench`): 18/18 PASS; recon_unit rel-L2 1.3e-3 strict / 1.5e-3 acc16 (was 1.3e-3); latent 21x256 1.45e-3 strict / 1.52e-3 acc16 (tol 0.02).
Strict-mode numbers were measured earlier in the session (before final tile-config selection); acc16 numbers are final. Timings were noisy for part of the session (a GPU miner was running on the host).

API additions (old API source-compatible): any H,W (edge-replicate pad to /32, latent covers padded frame; `cuda_wan22_vae_decode_crop` crops back),
`cuda_wan22_vae_encode_u8`, `set_input_range255`, `set_tiling` / `WAN22_VAE_TILE=<edge>[,<overlap>]` / `WAN22_VAE_BUDGET_GB` (auto tiling, default 5 GB total),
`set_persistent`, `warmup`, `last_error`, `reset_peak`. Errors return <0 with text (OOM tested: later calls still work). Peak is now the runner's own allocations (incl. weights 1.31 GB).
Frames are streamed to/from the device per chunk, so peak does not grow with T.

Tiling (linear blend over 128 px overlap; encoder attention is global per tile, so seams are not zero): synthetic video, tiled vs untiled:
256x416 tile 256/96: latent rel-L2 5.2e-2, decode-of-same-latent 5.8e-3, recon error vs input 11.43% untiled vs 11.70% tiled.
448x832 tile 448/128: latent 4.1e-2, same-latent decode 7.8e-3, recon 10.09% vs 10.26%; tile 320/128 is worse (8.2e-2 / 1.1e-2 / 10.78%).
Peak (incl. 1.31 GB weights): 21x448x832 untiled enc 3.73 GB / dec 6.57 GB; auto-tiled enc 3.73 / dec 5.16 GB (dec 10.0 s -> 14.8 s, under contention).
9x720x1280 (padded 736): enc 5.12 GB, dec 5.01 GB (3.4 s / 18.9 s measured while GPU was shared).

Not done / limits: not a true flash attention; strict-mode final numbers not re-measured; the 64x64 kernel reaches ~45 TFLOPS-equivalent (about half of peak), further tuning
(stage count made no difference) not explored; norm/dup_up/convert kernels are ~9% of time; tiling seams are real (above), overlap size did not reduce them (global attention);
padding semantics only self-checked (no torch oracle for padded inputs); tiled paths not checked against the oracle, only against untiled.

## Policy runner: backbone + MoE + action expert + window_joint sampler
- `cuda_praxis_runner.h` (host, config/weights/streaming/sampler) + `praxis_kernels.h` (NVRTC kernels).
- `test_cuda_praxis.c` driver; `make -C cuda/praxis test_praxis`.
- Reference: `ref/praxis/gen_backbone_ref.py` (praxis' tiny random policy, fp32 compute on CUDA, bf16-exact weights;
  writes `$PRAXIS_ROOT/tiny/{model.safetensors,config.json}` and `tmp/praxis_ref/backbone/*.npy`).

Everything is config driven (`config.json` in the praxis checkpoint layout): dim, heads, depths, dense prefix,
tap blocks, expert counts/top-k/scaling, rope, unpatch factor, action expert dims, chunking. The real 3072-dim
checkpoint only has to provide `config.json` + either `model.safetensors` or `model.safetensors.index.json` + shards (bf16/f16/f32 tensors, praxis key names).

### Implemented (all validated on tiny, see below)
- Linear (bf16/f16/f32 weights read in place, fp32 accumulate), zero-centred RMSNorm, q/k RMSNorm per head.
- Gated GQA attention with the teacher-forcing mask from per-token int codes (global 0, clean chunk c = 2c+1,
  noised chunk c = 2c+2; same rule as `backbone._tf_packed_mask_mod`), dense-masked kernel, one block per (query, head).
- Partial RoPE: 1-D positions (ROW/FRAME stride), float64 tables, pairs (j, j+D/2), applied before q/k norm.
- Dense SwiGLU, MoE (fp32 sigmoid router, selection-only bias, top-k, normalised gates, routed_scaling_factor,
  shared expert, expert-grouped GEMM over sorted token rows), adaLN modulation (encoder per-chunk timestep rows,
  decoder per-token modulators on the compact conditioning rows with nearest 2x2 index upsample, register rows = bias),
  Fourier timestep embedding, t_proj, text MLP, video/IO projections, patch/unpatch.
- Action expert: ActionTauEncoder, proprio/future/action query sequence, trunk-rope q/k tables from encoder positions,
  cross-attention to [globals | clean | noised] tap tokens via tap_norm/tap_proj/text_proj, cross/self dense masks,
  scale-only AdaLN, GEGLU, LayerNorm + head.
- Euler window_joint sampler (t_shift grid in float64, CFG combine for video and action, identical noise from .npy).
- Expert streaming (M4): routed banks kept in pinned host memory, streamed per MoE layer through a ring of R device
  slots on a non-blocking copy stream with events (prefetch of the next layer overlaps compute; ring >= 2 prefetches,
  ring 1 loads synchronously). `praxis_set_stream(r, R)` / `--stream R`; `praxis_stats()` reports peak device bytes,
  routed-bytes-on-device, H2D bytes, loads and prefetch hits.

### Validated (tiny model, `make -C cuda/praxis test_praxis`; fp32 reference, rel-L2 / cos)
| tensor | rel-L2 |
|---|---|
| encoder embed / every block attn, ffn, out, router logits, top-k indices, cond | 1e-7 .. 5e-7 (indices identical) |
| decoder embed / blocks / velocity | 1e-7 .. 4e-7 |
| action expert blocks / output | 2e-4 .. 6e-4 |
| 2-step window_joint: video / action (last chunk) | 3.9e-7 / 2.3e-4 |
Also run with K=2 and K=4 chunks and other text lengths (53 dumps each pass). K=1 is not testable: the praxis
reference itself fails there (empty clean tensor reshape). Streaming self-test (`--selftest-stream`): ring 2 and ring 1
are bit-identical to resident, routed device residency == R x slot < total routed (tiny has 3 MoE layers).
The expert error floor (5e-4, vs 3e-7 in the backbone) is most likely the fp32 `pow` used for RoPE theta (the expert has
64 rotating dims, so torch/sleef and glibc differ by an ulp at positions up to 3e5); it is far inside the 0.02 tolerance.

## Performance path, mid-size parity, synthetic full-size model, NVFP4 (this iteration)

Modes: default = FAST (cuBLAS bf16 x bf16 -> fp32 accumulate GEMMs through `cublasew_gemm_ex_rm`, fp32 residual stream/norms,
block-sparse tensor-core flash attention `pr_flash` (mma.sync m16n8k16 bf16, head_dim 16/32/64/128, GQA, gate, token-code mask with
per-K-tile skip lists), device-side MoE dispatch (count/scan/fill kernels, deterministic row order), chunked workspace
(`PRAXIS_MOE_CHUNK`, default 4096 tokens). `PRAXIS_EXACT=1` = the original fp32 kernels (bit-for-bit the old behaviour).
`PRAXIS_PROFILE=1` prints a per-phase time split (syncs the compute stream only, so streaming overlap is preserved).
The router GEMM stays fp32 in both modes; the bf16 expert path does one tiny host readback per MoE chunk (128 ints), NVFP4 has none.

### Tests (all pass)
- `make -C cuda/praxis test_praxis`: tiny (dim 64), exact path at the old 0.02 tolerance (rel-L2 3e-7 .. 6e-4) + streaming self-test, then the fast path at **0.06**.
- `make -C cuda/praxis test_praxis_mid` (`refs_praxis_mid` makes the refs; `gen_backbone_ref.py --preset mid|mid128`): dim 256/head_dim 64/4+2 heads and
  dim 384/head_dim 128/3+1 heads (GQA 3, as real), 8 experts top-2, expert_hidden 128, K'=3, vs torch fp32 oracle. Exact path: rel-L2 2e-5..4e-4;
  fast path: sampled video 2.5e-3 / 7e-3, actions 8e-4, dec_vel 4-5e-3.
- Fast-path tolerance is 0.06 (not 0.02) because bf16 noise flips ~0.3% of near-tied top-k routes (token agreement 99.7%); those tokens dominate the MoE-ffn
  rel-L2 (3-5%). On tokens with identical routing the MoE ffn error is 0.5%. `compare.py --topk-agree F` checks route-set agreement instead of rel-L2 for `*_top_idx`.
- `test_praxis_attn`: flash vs dense-masked fp32 kernel at REAL token counts (encoder 4020, decoder 9988, K'=3 6516; 24/8 heads, hd 128): rel-L2 2.2e-3 (bf16 P).
- `test_praxis_fp4`: CPU packer bit-identical to the torch re-implementation of nvfp4.py (global, e4m3 block scales, packed bytes: 0 mismatches; torchao itself is
  not installed, so the f32->e2m1 step was checked against an RNE emulation, not torchao); grouped W4A4 kernel vs CPU fp32 on dequantized operands 1.7e-3 (bf16 out rounding).
  Quantization error of a whole expert (gate_up -> SwiGLU -> down) vs fp32 with the original bf16 weights: **0.23 rel-L2 per layer** on gaussian weights/activations
  (weight round trip alone 0.096). End to end on mid configs vs torch: dec_vel 1.6e-2 / 2.4e-2, sampled video 8.7e-3 / 1.8e-2, actions 8e-4 (the 0.05-0.1 target holds e2e;
  per-MoE-layer error is larger but diluted by shared expert, attention and residual). Not validated on real trained weights.

### NVFP4 routed experts (`PRAXIS_NVFP4=1` at load; needs sm_120a, dim and expert_hidden multiples of 128)
(Real-weight accuracy work: `PRAXIS_EXPERT_FMT=nvfp4|nvfp4a16|mixed`, FP8 banks, expert groups, see "NVFP4 accuracy on the real weights" below.)
Recipe of praxis `quantization/nvfp4.py` (E2M1 RNE, 16-block e4m3 scale clamp [2^-6,448], fp32 global amax/2688, reciprocal multiply). Bank slot per layer =
[gate_up q|sf, down q|sf, globals] in plain [N][K/16] scale layout (not the cuBLAS swizzle; only our kernel reads it), 0.81 GB/layer, 13.45 GiB (14.44 GB) per pass.
CPU packer (OpenMP): 0.87 s per real-size layer on 32 threads (~15 s for the model). Activations are quantized dynamically (global = amax of the chunk rows),
grouped block-scale mma GEMM `pr4_gemm_grouped` (BM64 x BN128 x BK128, 2-stage cp.async, padded smem, device tile-prefix, ~75 TOPS vs 27 for the first version and
cuBLAS bf16 ~25 here). Synthetic NVFP4 banks are random valid packed bytes.

### Synthetic full-size model (`make bench_praxis_synth`; config `gen_synth_config.py`)
`bench_praxis_synth --synthetic $PRAXIS_ROOT/synth [--stream R] [--pair] [--skip-last-video] [--cfg-video 4 --cfg-action 1] [--profile] [--no-h2d]`.
Inferred real dims (dim 3072, 24/8 heads, hd 128, 128 experts top-8, hidden 1280, enc 14 / dec 6, action expert 16 x 1024), window K'=2, 6 frames/chunk, 13x16 latent
(208 tokens/frame), text 64: encoder 4020 tokens, decoder 9988. Weights random (resident bf16 device buffers; routed banks random pinned host memory). bf16 banks reuse
4 distinct 2.8 GB pinned banks cyclically (51 GB pinned would not fit comfortably in 63 GB RAM); every layer still streams a full-size bank, so bandwidth numbers are real.
No numeric oracle exists at full size (outputs are meaningless); correctness rests on the mid-size and kernel tests above. Hardware: RTX 5060 Ti 16 GB on PCIe Gen4 x8 (raw pinned H2D 14.4 GB/s).

| config (K'=2, 2 Euler steps, ring 2) | bf16 banks | NVFP4 banks |
|---|---|---|
| one backbone pass (enc+dec+expert), warm | 3.68 s (enc 2.5, dec 1.0, ae 0.16) | 1.46-1.52 s (enc 0.77, dec 0.65, ae 0.05) |
| call cfg 1.0 (2 passes) | 7.36 s | 2.91 s (2.26 s with skip-last-video) |
| call cfg 4.0, 4 passes (cond+uncond per step) | 14.7 s | 5.82 s |
| + paired encoders | n/a run | 4.71 s |
| + skip-last-video (cfg_action 1.0: 3 encoder passes) | 9.9 s | 3.73 s |
| + both | 7.6 s | 3.16 s (cfg_action 1.5: 3.43 s) |
| compute only (`--no-h2d`), cfg 4 baseline / pair+skip | - | 4.67 s / 2.86 s |
| peak device (own accounting / cuMemGetInfo / incl. CUDA ctx) | 11.3-11.7 / 11.5-11.9 / 11.7-12.2 GB | 7.12 (7.57 paired) / 7.32 (7.77) / 7.55 (~8.0) GB |
| H2D achieved | 13.0 GB/s (90% of raw) | 9-10 GB/s average (compute-bound) |

Ring sweep (NVFP4, pair+skip): ring 1 = 3.58 s / 6.77 GB, ring 2 = 3.16 s / 7.57 GB, ring 3 = 3.12 s / 8.36 GB. bf16 is stream-bound (49 GB/pass; the bf16 ring of 2 x 2.8 GB
alone is 5.6 GB so it cannot meet <8 GB at ring 2). NVFP4 is compute-bound: pass profile (s): attention flash 0.22, qkv/out projections 0.19, dense FFN 0.11,
shared experts+combine 0.10, router/dispatch 0.04, act-quant 0.07, gate_up 0.16, swiglu 0.03, down 0.09, other 0.15, stall on expert slot 0.07-0.27.
Qwen/VAE are not resident (not loaded). Target <8 GB: met by NVFP4 at ring 1-2 (7.0-7.8 GB incl. context ~0.2-0.4 GB; paired ring 2 is borderline at ~8.0 GB total).

### Sampler shortcuts (`praxis_sample_window_joint_ex(..., pair_cfg, skip_last_video)`)
`pair_cfg`: both guidance encoder passes run as one 2-sequence batch (shared row-wise ops, per-sequence flash attention) so each MoE layer is read once. Outputs identical
to the sequential run on mid configs (rel-L2 0.0). `skip_last_video`: last step runs encoder+expert only (decoder skipped; unguided pass only when cfg_action != 1); actions
identical to the plain run, video output zeroed. Decoder passes are not paired (compute-bound, would double the decoder workspace). Context caching: see below.

## Sharded checkpoints, pack cache, ctx cache, block-sparse expert attention, end-to-end driver (this iteration)

### Sharded safetensors loading (`make -C cuda/praxis refs_praxis_sharded test_praxis_sharded`)
- `praxis_load(dir)` reads `model.safetensors.index.json` (`weight_map` -> shard files, any N <= 64) else `model.safetensors`. Every shard is mmap'd (lazy), tensors are
  looked up across shards. Routed banks stay file-backed until used; each layer is consumed (pinned bf16 copy, or NVFP4 packing) and its file pages are then dropped from the
  mapping (`madvise(MADV_DONTNEED)`), so host RSS = packed bank set (pinned) + ~1 layer of bf16 pages + the small resident weights, never the whole bf16 set.
  Non-routed tensors are dropped right after the H2D copy. `PRAXIS_NO_DROP=1` disables the dropping (A/B knob for the RSS test).
- Pack cache: `PRAXIS_PACK_CACHE=<file>` (with `PRAXIS_NVFP4=1`). One 4 KiB header + one packed blob per MoE layer in moe order; fingerprint = shard sizes/mtimes + bank
  tensor sizes + 16 sampled 4 KiB windows of every bank, magic written last (an interrupted write is ignored and rewritten). A hit reads each layer straight into its pinned buffer.
- `gen_backbone_ref.py --shards N` writes the real layout (index.json + `model-0000i-of-0000N.safetensors`); `--preset big` (dim 768, 32 experts, hidden 512, 14+8 blocks, 19 MoE layers,
  1.6 GB bf16) with `--skip-ref` is the RSS test model. Not tested at 51 GB / 17 shards (no checkpoint, 23 GB free disk); the loader has no size-dependent code.
- Measured: sharded (3/4/5 shards) tiny/mid/mid128 pass the exact-path parity, and mid/mid128 outputs are **bit-identical** to the single-file runs; NVFP4 from shards ==
  NVFP4 from the single file semantics; pack-cache hit output **bit-identical** to a fresh pack. Host RSS (`test_praxis_shards`, big_sh, NVFP4, library RSS calibrated by a prior load):
  growth **473 MB** vs packed banks 405 MB (+ one 72 MB bf16 layer) and **1981 MB** with `PRAXIS_NO_DROP=1` (the whole mapped bf16 set + packed: load-everything behaviour);
  the test asserts growth <= packed + 3 bf16 layers + non-routed weights + 96 MB (868 MB). Pack time for big: 2.7 s packing -> 0.4 s from cache (0.8 s whole load); at real size
  the 15 s repack becomes a 14.4 GB sequential read (disk-bound, not measured).
- Bug fixed on the way: `praxis_free` leaked all device weights/ring/state (a second consumer in the same process saw 5-6 GB still in use); every `pr_malloc` is now tracked and released.

### Encoder context cache across Euler steps (`praxis_set_ctx_cache` / `PRAXIS_CTX_CACHE=1` / `bench_praxis_synth --ctx-cache`, default OFF)
Semantics of `backbone._core_ctx_cached_gen`: registers/anchor/text and clean tokens attend only each other, so their K/V (post q/k-norm + rope) are identical in every step of a sampler call.
Step 0 runs the full sequence and stores, per layer and per guidance branch, the K/V rows of [globals | clean] (bf16); later steps embed and run only the noised rows (exactly
the noised rows, not whole 128-blocks) through the stack: row-wise ops/MoE routing are per token, attention is rectangular (queries = noised rows; keys = cached context ++ new noised
rows; the existing token-code flash kernel is rectangular already, key tile summaries are built over the reordered keys). Taps/embeddings of the context rows persist per branch
(non-paired mode now uses a separate encoder state per branch), cond = out-proj of the noised rows. Works with `pair_cfg` (per-sequence caches), skip-last-video, per-branch text length.
Fast path only. Cost: +87 MB/branch K/V (full size) plus work buffers (+0.17 .. +0.6 GB peak in the benchmark configs; the paired config goes from 7.57 to 7.74 GB own accounting).
- Parity (tiny/mid/mid128 selftest, 4 Euler steps, cached vs uncached): actions 7e-4, video 4.4e-3 / 6.3e-3 rel-L2 (bf16 noise; also `PRAXIS_MOE_CHUNK` alone gives 0.0). Same through the e2e driver.
- Speed (synthetic full size, NVFP4, K'=2, cfg video 4 / action 1, ring 2, warm): **compute-only (`--no-h2d`), 4 steps: 9.34 s -> 8.36 s (-10.5%)** (encoder rows 4020 -> 2496,
  cached encoder pass 0.48 -> ~0.31 s). With the real host streaming it is nearly free of benefit because a pass is stream-bound (14.4 GB per pass over PCIe4 x8 ~0.7 s for the 12 encoder
  MoE layers alone): 2 steps plain 5.82 -> 5.73 s (-1.5%), paired+skip-last-video 3.17 -> 3.12 s (-1.3%); 4 steps plain 11.63 -> 11.38 s (-2.1%), paired+skip 7.88 -> 7.48 s (-5.0%).
  It pays off when the banks are resident (bigger GPU) or the link is faster; here it is a minor win and costs 0.2-0.6 GB, so it stays opt-in.

### Block-sparse flash attention for the action expert (default on in the fast path; `PRAXIS_AE_DENSE=1` = old dense kernel)
Every expert query reads at most 3 contiguous key intervals (cross: [globals) + clean frames in window + noised frames of its chunk; self: its proprio / future / action tokens), so the mask is 6 ints per
query (`build_trunk_rope_masks` rules; no Q x KV byte mask any more, and no host O(Q*KV) loop). New template flavour of `pr_flash_impl<HD, IV>`: tile list from the interval unions of a 64-query block,
per-element interval test, bf16 operands/fp32 accumulate (q/k/v converted after rope+norm). `test_praxis_attn` (16 heads, hd 64, 30..60 queries, independent rule-based dense reference):
cross K'=2/3/4 (KV 4020 / 6516 / 9012): rel-L2 2.2e-3 (bf16), kernel **2.29 -> 0.143 ms, 5.08 -> 0.227 ms, 10.6 -> 0.311 ms (16x / 22x / 34x)**; self 3-4x (tiny). Whole expert pass (NVFP4 synthetic, 16 layers):
**48 -> 32 ms at K'=2, 156 -> 74 ms at K'=4**. Model-level parity unchanged (ae_out vs torch 2.4e-3 / 3.1e-3 mid / mid128, same as dense 2.4e-3 / 2.9e-3).

### End-to-end driver (`cuda/praxis/praxis_e2e.c`; `make -C cuda/praxis refs_e2e test_praxis_e2e`)
task token ids (.npy; `ref/praxis/gen_e2e_ref.py` writes them with the Qwen processor, no chat template) -> Qwen3.5-2B (cuda/llm `cuda_llm_prefill_hidden_all`, hidden_states[-3] = output of layer 21, ids padded to a
multiple of 32 with <|endoftext|>, padding rows removed) -> exterior + wrist uint8 frames (.npy [T][H][W][3], last 21 + 24 (K'-2) frames) -> tile canvas (single_arm.tile_clip, antialias off: centre aspect crop + one bilinear resize + round; 256x416 tiles
side by side = 256x832; `--tile-h/--tile-w`) -> `cuda_wan22_vae_encode` -> policy latent [192][6(K'-1)][h][w] (anchor = latent frame 0, clean chunks = 6-frame slices) -> `praxis_sample_window_joint_ex2` (cond + blank-caption branches
with their own text lengths; pair / skip-last-video / ctx-cache / NVFP4 / pack cache / sharded checkpoint all available) -> last-chunk video latent + action tokens -> 24 normalised action rows [24][10]; `--decode` reloads the VAE and decodes
[clean | predicted] chunks, keeping the last 24 frames. Writes text/latent/pred_video/pred_action/action_rows/(decoded).npy to `--out`.
- Sequencing/VRAM: Qwen runs in a **child process** (`cuda/llm/test_qwen35_st -p 32 --cond-only`, text cached per prompt in `--text-cache`, so exit frees all of its memory); the VAE is loaded, used and freed (kept with `--keep-vae`);
  the policy is loaded after; the optional decode reloads the VAE. `praxis_free` now really frees, so stages never overlap. Measured on the synthetic full-size config (real Qwen + real VAE + random 3072-dim policy, NVFP4, ring 2, canvas 256x832),
  device bytes over the idle 0.13 GB context: **Qwen 3.68 GB | VAE encode 3.16 GB | policy 7.31 GB (paired; 7.45 GB incl. ctx, 8.00+0.13 with --ctx-cache) | VAE decode 5.90 GB; max instant 7.45 GB** (<8 GB; the VAE never has to be resident
  next to the policy; reloading it costs ~3.5 s encode-side / 9 s decode of 45 frames at 256x832). Wall: text 8.6 s (cold, 2 prompts; cached 0 s), VAE 3.3 s, policy load 13.4 s (synthetic bank generation), sample 4.7-6.8 s cold (first call).
- Numerical validation on `mid2048` (dim 384, text dim 2048 so the real Qwen output is consumed; 8 experts; sharded in 3 files; `gen_backbone_ref.py` text dim is 32/64 in tiny/mid, hence the new `gen_e2e_ref.py` policy) against the same pipeline in PyTorch
  (fp32 Qwen on CUDA, `single_arm.tile_clip`, `Autoencoder.encode_eager`, `PraxisPolicy.sample_window_joint(ctx_cache=False)`, same noise, 2 steps, cfg 4 / 1.5, 64x128 tiles -> 2x8 latent): canvas 100% pixel-identical; text cond 1.2e-4; VAE policy latent 1.6e-3;
  pred_video 7.1e-3; pred_action / action_rows 9.6e-4; decoded frames 7.3e-3 (ctx-cache variant 4.5e-3 / 9.4e-4). `compare_e2e.py` checks these.
- **Actions are NORMALISED**: un-normalisation (q99 xyz/grip, rot6d identity, clip, `rel10_to_wire` anchor composition) needs the checkpoint `NormStats` and rig frame, which are not available without the gated checkpoint; rows follow
  `robot_io.action_rows` (depth >= 1: every token gives 4 rows x 10, no clip). Proprio input is a normalised [K'][10] .npy (zeros if omitted). Noise is taken from .npy (`--noise-v/--noise-a`, for parity) or our own PRNG (torch `seeded_normal` cannot be reproduced in C).
- Limits: K' >= 2 only (the torch reference itself cannot run K'=1); one text length for both branches is not required but `--pair` needs equal lengths (cond vs blank caption never are, so the paired path is only used by the benchmark);
  cond/uncond tokenization/padding is done offline by the Python helper (no C tokenizer); the Qwen child re-loads 4.5 GB of weights per new prompt (4.3 s).

### Not done / limits
- Real checkpoint (gated), bf16-end-to-end activations (residual stream and attention-prep/rope/norm remain fp32; GEMM operands bf16).
- Action-expert attention: dense fp32 kernel only with `PRAXIS_AE_DENSE=1` / exact mode; otherwise the interval flash kernel.
- Expert GEMM loops in bf16 mode are per-expert cuBLAS calls (not grouped), with one 516-byte readback per MoE chunk.
- NVFP4 accuracy only measured on random/mid models; praxis notes NVFP4 loses accuracy on hard tasks.
- Fast MoE/route flips: no determinism guarantee vs torch bf16 (the reference itself is fp32 here).
- Constants not in config: `row_stride`/`frame_stride` default to 360 / 129600 (optional config keys under `backbone`).

## REAL checkpoint (runwayml/praxis-1-droid, 17 bf16 shards, 54.4 GB) - first real-weight results
Checkpoint dir `$PRAXIS_ROOT/praxis-1-droid`; reference dumps `tmp/praxis_ref/real` (torch bf16, layer-offloaded; README there) and `tmp/praxis_ref/real_fp32` (torch fp32 = the truth; same
script with `--fp32 --tag _fp32 --calls 2`, 350 s). All commands from the repo root.

### Loading / config / keys
- `praxis_load` loads the real dir unchanged. All **567** tensors of `model.safetensors.index.json` are requested by the runner and nothing else is requested (`PRAXIS_LOG_KEYS=1` logs every
  lookup; unused = none, missing = none). Shapes are checked by byte size, dtypes bf16 (+ fp32 `expert_bias`).
- Config: every key of config.json was checked against `praxis-1-oss/praxis/*.py`. Read: dims/heads/depth/dense/taps/projection widths, rope (`effective_dim`, `base`, `efficient_cache_max`, and now
  `max_intra`/`max_inter` = ROW/FRAME stride 360/129600), moe (`num_routed_experts`, `top_k_routed`, `expert_hidden`, `routed_scaling_factor`), action expert (dims, `clean_only_conditioning`,
  `global_order`, rope dim/base/table), chunking (`n_chunks`, `latents_per_chunk`). Hard-coded semantics are now VALIDATED at load and the load is refused on a mismatch (`pr_parse_config`):
  layout chunk_causal, modulation_dtype mixed, norm/qk_norm eps 1e-6, gated attention, rope_after_norm false, ref_placement group, chunked_video_layout true, time_scale 1, router sigmoid/fp32,
  routed layout expert_first, shared expert, `moe.top_k == top_k_routed + 1` (9 = 8 routed + shared), expert rope kind trunk/before_norm, absolute_positions, no action_pos_table, future_chunk_offset zero.
  `positive_zero_residual_add` / `head_plus_zero` only turn -0 into +0 (numerically a no-op, ignored). Chunk geometry 21/24/24/24 frames = 6 latents per chunk (config `latents_per_chunk`).
- Default stream: routed banks > 6 GB now default to a 2-slot ring (`PRAXIS_STREAM=N` overrides, 0 = resident); before, `praxis_load` tried to make 13 GB resident.
- NVFP4 pack from the 17 mmap'd shards, layer by layer: **259 s cold** (disk-bound: 51 GB of bf16 read; the dominating cost) writing a 14.44 GB `PRAXIS_PACK_CACHE`; from the cache **22-40 s** total load
  (cache read now 8 parallel preads per layer: 38 s -> 22 s; page-cache dependent). Host RSS peak **17.4 GB** (cold pack, includes 13.45 GB pinned packed banks), **14.5 GB** from the cache.
- bf16 expert mode on the real model does not fit pinned host RAM (51 GB): `PRAXIS_BF16_MMAP=1` (parity/debug) streams bf16 banks straight from the mmap'd shards (pages dropped after each copy), host RSS 3.7 GB,
  whole encoder+decoder+expert+2-step sample 42-47 s; needs ring 2 x 2.88 GB = 11.3 GB device (does NOT meet the 8 GB target; bf16 is parity-only here).

### Driver (`praxis_e2e`, new options)
`cuda/praxis/gen_real_inputs.py` makes deterministic synthetic inputs (ids via the Qwen tokenizer, two 720x1280 DROID-like views, pose8). Example:
```
~/.venv/bin/python -I cuda/praxis/gen_real_inputs.py --out tmp/praxis_real/in
PRAXIS_NVFP4=1 PRAXIS_PACK_CACHE=$PRAXIS_ROOT/praxis-1-droid.nvfp4.packcache cuda/praxis/praxis_e2e --policy $PRAXIS_ROOT/praxis-1-droid \
  --ids tmp/praxis_real/in/ids.npy --ids-uncond tmp/praxis_real/in/ids_u.npy --ext tmp/praxis_real/in/ext.npy --wrist tmp/praxis_real/in/wrist.npy \
  --pose8 tmp/praxis_real/in/pose8.npy --K 2 --steps 2 --cfg-video 4 --cfg-action 1 --stream 2 --repeat 5 --text-cache tmp/praxis_real/textcache --out tmp/praxis_real/out
```
New: tile size/`video.antialias` read from the checkpoint config (224x416 tiles, **fit rule** = centre crop to 720x1280 aspect, torch-style antialiased bilinear resize to 234x416 (separable, float32), round,
centre trim to 224 rows; 0.27 s CPU for 21 frames x 2 views; checked only by construction, not yet bit-compared with the torch tiles), `--pose8` (current/boundary poses -> normalised proprio with
`(p-mean)/max(std,1e-6)`, rot6d = first two matrix columns), un-normalisation of the 24 rows (clip dims 0,1,2,9 to [-1,1], `(n+1)/2*(q99-q01)+q01`, grip clipped to [0,1]) -> `action_rel.npy` [24][10]
and the composition with the current pose in the flange frame -> `action_abs_flange.npy` [24][13] (xyz, R row-major, grip; the client-frame/FR3 hand conversion of `pose_to_rig` is NOT done),
`--repeat N` (warm median), `--load-only`, `--pair` with different text lengths = timing only (our paired path needs equal lengths; torch pairs with different ones).
It runs end to end on the real weights: text (Qwen child) -> VAE encode -> NVFP4 policy -> 24 normalised and un-normalised rows. Output is plausible (rot6d ~ identity, xyz mm..cm), but with a synthetic
scene there is no task-level check.

### Latency / memory (RTX 5060 Ti, GPU otherwise idle, NVFP4 K'=2, 2 Euler steps, cfg video 4 / action 1, warm median of 4, own accounting + 0.13 GB CUDA ctx)
| config | per call | policy peak (own) | + ctx |
|---|---|---|---|
| ring 2 | 5.34 s | 6.88 GB | 7.01 GB |
| ring 2 + skip-last-video | 3.47 s | 6.88 GB | 7.01 GB |
| ring 2 + paired + skip-last-video (timing only, equal text lengths) | 2.81 s | 7.26 GB | 7.39 GB |
| ring 1 | 6.04 s | 6.09 GB | 6.22 GB |
| ring 1 + paired + skip-last-video (timing only) | 3.31 s | 6.47 GB | 6.60 GB |
First call is 0.8-1.0 s slower. Stages (sequenced, VAE freed before the policy): Qwen 2.85 GB over ctx (child process), VAE encode (224x832, 21 frames) 2.53 GB own peak, policy as above: **max instant
< 7.5 GB in all configs (target 8 GB met)**. Real vs synthetic: latency matches (synthetic 5.8 / 3.2 s), peak 6.9 vs 7.1 GB (real latent is 7x26 not 13x16, 3473 enc / ~8000 dec tokens); the real
config confirmed every inferred dim. Pack load is far slower cold (259 s) than the synthetic 15 s estimate (disk read of 51 GB).

### Parity on real weights (K'=2 call, pass 0 + 2-step sample; `test_cuda_praxis --ref tmp/praxis_ref/real[_fp32]`, rel-L2 / cos)
Bug found and fixed in the harness (not the runner): `test_cuda_praxis` used the cond text length for the 1-token blank caption -> out-of-bounds read -> NaN; it now uses `_ex2` with the uncond length.
Also hardened `pr_topk` against NaN scores (index -1 write). No runner bug was found on the real weights: RoPE split, registers, modulators, expert bias selection, routed scaling, chunk geometry,
s buffers all reproduce.
Key finding: the **torch bf16 reference is itself noisy** (huge outlier tokens: registers/text |x| up to 1e3, bf16 rounding + route flips, video rows blow up in the last blocks). Against the torch **fp32** run:
| tensor | torch bf16 | ours fast bf16-expert | ours NVFP4 |
|---|---|---|---|
| enc_b13 video rows (noised) | 1.96 | 0.26 | 0.68 |
| enc_cond | 0.545 | 0.245 | 0.59 |
| dec_vel (pass 0) | 6.98e-2 | **1.99e-2** | 7.76e-2 |
| ae_out | 2.96e-2 | **1.04e-2** | 2.03e-2 |
| sampled video (cfg 4) | 0.422 | 0.217 | 0.392 |
| sampled actions | 0.497 | 0.230 | 0.366 |
Per-block encoder outputs vs the bf16 torch ref: 0.5-1% on the whole tensor (dominated by the outlier tokens), video rows 1-2% at b2 growing to 25% at b12 (same growth as torch-bf16 vs fp32);
route-set agreement (all 8 experts identical) is only 14-25% per token vs the bf16 ref (individual experts overlap far more; near-tied sigmoid scores of 128 experts), so the 0.02/0.06 tolerances of the mid
configs do not apply; the fp32 run is the right oracle. Un-normalised actions vs fp32 (synthetic scene, rms over 24 rows): torch bf16 xyz 5.0 mm / rot6d 0.0165 / grip 0.35; ours bf16 6.2 mm / 0.0097 / 0.16;
ours NVFP4 4.8 mm / 0.0143 / 0.26 (action signal is only ~10 mm rms here, so these are large relative errors for all three; they are the same order as the bf16 reference's own error).
NVFP4 vs bf16 path: dec_vel error 7.8e-2 vs 2.0e-2 (4x), ae_out 2.0e-2 vs 1.0e-2: NVFP4 on the trained weights costs ~4x the bf16-path error in the video velocity, but lands at the torch-bf16 reference's noise
level; task-level quality (praxis reports NVFP4 loses on `hard` tasks) is not measured.

### NVFP4 accuracy on the real weights: diagnosis and the mixed-precision expert preset (`PRAXIS_EXPERT_FMT`)
Problem: on the real checkpoint the NVFP4 routed experts (W4A4) cost ~4x the velocity error of the bf16 experts (dec_vel 7.8e-2 vs 2.0e-2 against the torch fp32 oracle). Everything below uses
`test_cuda_praxis --ref <torch fp32 oracle>` (pass 0 + 2-step sampler, K'=2), driven by `cuda/praxis/run_expert_variant.sh`, `action_metrics.py` (errors incl. un-normalised actions),
`moe_attr.py` (per-MoE-layer attribution on real activations), `router_flips.py`, `check_pack.py`. Scratch: `tmp/praxis_attr`, `tmp/praxis_out/variant_*` (tmp is a symlink to a large disk).

**Diagnosis (base weights `praxis-1-droid`, per MoE layer, routed-expert output rel-L2 vs the bf16 expert path on the real activations of the bf16 pipeline; `moe_attr.py`)**
| layer | gate_up FP4 only | down FP4 only | W4A16 | W4A4 | acts FP4 only | down FP8 only | W8A16 |
|---|---|---|---|---|---|---|---|
| enc2 | 0.010 | 0.084 | 0.085 | 0.092 | 0.040 | - | - |
| enc3 | **0.090** | 0.080 | 0.120 | 0.161 | 0.107 | - | - |
| enc4 / enc5 | 0.017 / 0.015 | 0.076 / 0.082 | 0.078 / 0.084 | 0.093 / 0.090 | 0.051 / 0.028 | - | - |
| enc6 / enc7 / enc8 / enc9 | 0.033 / 0.037 / 0.027 / 0.020 | 0.078-0.097 | 0.085-0.099 | 0.096-0.116 | 0.046-0.059 | 0.020-0.026 | 0.023-0.027 |
| enc10 | **0.119** | 0.091 | 0.148 | 0.203 | 0.144 | 0.025 | 0.041 |
| enc11 / enc12 | 0.022 / 0.058 | 0.095 / 0.082 | 0.097 / 0.100 | 0.109 / 0.127 | 0.049 / 0.080 | 0.027 / 0.024 | 0.028 / 0.029 |
| enc13 | **0.223** | 0.086 | 0.252 | **0.348** | 0.265 | 0.024 | 0.066 |
| dec1..dec5 | 0.050-0.065 (dec5 0.057) | 0.074-0.086 (dec5 0.041) | (mse) 0.063-0.096 | 0.102-0.142 | - | 0.015-0.025 | 0.025-0.031 |
| mean (enc6..dec5, 13 layers) | 0.064 | 0.081 | 0.099 (mse) | 0.140 | - | 0.023 | 0.031 |
Findings: (1) the loss is **weight quantisation**, not activation quantisation: FP4 weights alone give ~0.10 per layer, adding the dynamic FP4 activations only takes it to ~0.14; (2) the
**down bank** is uniformly ~0.08 (= the 9.4% FP4 round-trip error of the weights, they are Gaussian-like and FP4 with 16-blocks is already at the Lloyd-Max limit, so no scale/codebook trick can fix it:
best block-scale search (`mse`) only reaches 8.4-8.5%), FP8 e4m3 per-channel brings it to 0.02; (3) the **gate_up bank is layer dependent**: 0.01-0.06 in most layers but 0.09 (enc3), 0.12 (enc10), **0.22 (enc13,
the last encoder block that feeds the decoder conditioning)**; (4) per-expert vs per-chunk activation global scale (praxis computes the activation global per expert; we use the chunk amax) is a no-op
(W4A4 0.1229 vs 0.1233 mean); (5) the calibrated FC1 output scales in `nvfp4_calibration.py` are only applied by praxis on its SM100/SM103 fused-SwiGLU backend (`set_fused_swiglu_scale` raises
elsewhere, `loading.py` only passes them to `flashinfer_sm100`): the SM120 path (the one we mirror) uses the dynamic per-expert amax, so there is nothing to implement; (6) **our packer is bit-exact with the
praxis recipe on a real expert** (`check_pack.py`: global amax/2688, e4m3 scale clamp [2^-6,448], reciprocal multiplies, E2M1 RNE: 100% identical scale bytes and codes on enc2 expert 3, both banks).
Router: logits are computed in fp32 from the fp32 residual stream (never quantised, `pr_moe_ffn_fast`: `pr_gemm_naive(x fp32)`); route flips are induced by upstream drift: fraction of tokens with the
identical top-8 set vs the fp32 oracle, mean over the 17 layers: bf16 experts 64%, mixed preset 48%, W4A4 31% (`router_flips.py`; individual-expert overlap 94% / 91% / 86%).

**The QAT checkpoint** `runwayml/praxis-1-droid-nvfp4` is now accessible (config.json is identical to the base, the 17 shards have different bytes: bf16 master weights after 5000 QAT steps, 54.4 GB, downloaded and
SHA256-verified to `$PRAXIS_ROOT/praxis-1-droid-nvfp4`; the packing happens at load time, same recipe). It is the right weight set for any NVFP4 path; results are given for both weight sets,
each against the torch fp32 oracle of its own weights (`tmp/praxis_ref/real_fp32`, `tmp/praxis_ref/real_qat_fp32`; the two fp32 oracles differ from each other by 0.51 rel-L2 in sampled actions, i.e. the
sampled video/action metrics have a noise floor of that order and **dec_vel is the only stable fidelity number**).

**What was added** (all selectable at load time, default behaviour unchanged):
- `PRAXIS_EXPERT_FMT=nvfp4` (= `PRAXIS_NVFP4=1`, W4A4, the old default), `nvfp4a16` (W4A16: NVFP4 banks, f16 activations, in-register dequantisation, f16 mma `pr4_gemm_w4a16`), `mixed` (below).
- `PRAXIS_FP8_BANKS=dn,enc3.gu,...` per-matrix FP8 e4m3 banks (per-output-channel fp32 scale, `pr8_gemm_w8a16`, same activations as W4A16), `-name` reverts; `PRAXIS_SLOT_MB=N` caps the ring slot: larger
  layers stream as 2/4/8 **expert groups** through the ring (bit-identical to ungrouped, also with several token chunks); `PRAXIS_PACK_SCALE=mse` = per-block scale search at pack time (6 candidate divisors
  of the block amax, keep the lowest squared error; no runtime cost). Non-default plans get their own pack-cache file (`<cache>.p<plan hash>`), the default cache is never overwritten.
- `PRAXIS_EXPERT_FMT=mixed` = `nvfp4a16` + FP8 down banks everywhere + FP8 gate_up for encoder blocks 3, 10, 13 (those three layers are 1.44 GB and stream as two expert groups, the ring slot is 1.02 GB) .
- f16 activations: A is converted fp32 -> f16 (saturating, 10 mantissa bits instead of bf16's 7) and the gate_up output / SwiGLU hidden stay f16; the f16 mma runs at the card's measured fp16/bf16 tensor rate
  (~25-30 TFLOPs here), 3x below the FP4 mma. A k-permutation inside each k16 step lets a lane read 4 consecutive weights and 64-bit A fragments, dequantisation is `cvt.rn.f16x2.e2m1x2` / `e4m3x2` (exact in f16).
  (A first bf16-mma version was 4x slower than the FP4 kernel: gate_up 2.55 s + down 1.24 s per call vs 0.62 + 0.30; the f16 version is 1.66 + 0.78 s.)
- `test_praxis_fp4` (`make check_praxis_fp4`) now also checks the W4A16 and FP8 kernels against CPU fp64 (rel-L2 2e-4 / 1.6e-3 = f16 / bf16 output rounding) and the FP8 packer; `make test_praxis_sharded` covers the plan,
  mixed-plan pack cache (hit == fresh), expert groups.

**Results** (RTX 5060 Ti, K'=2, 2 Euler steps, cfg video 4 / action 1; accuracy vs the fp32 oracle of the same weights; un-normalised action errors are rms over the 24 rows, signal rms xyz 10.3 mm (base) / 11.0 mm (QAT),
rot6d 0.57; latency = warm median, plain / skip-last-video / paired+skip (timing only); stream = routed GiB per pass; peak = own device accounting after the paired run + 0.13 GB CUDA context)

Base weights `praxis-1-droid` (oracle `tmp/praxis_ref/real_fp32`)
| variant | dec_vel | ae_out | video | actions | xyz mm / rot6d / grip (rms err) | ring slot MiB | stream GiB/pass | peak GiB (+0.13 ctx) | latency s: plain / skip-last / paired+skip |
|---|---|---|---|---|---|---|---|---|---|
| bf16 experts (ours; ring 2 x 2.8 GiB) | 1.99e-02 | 1.04e-02 | 0.217 | 0.229 | 6.18 / 0.0097 / 0.161 | 2880 | 47.8 | 11.03 | n/a (stream-bound ~14.7 s, synthetic) |
| W4A4 `nvfp4` (old default) | 7.76e-02 | 2.03e-02 | 0.392 | 0.366 | 4.78 / 0.0143 / 0.259 | 810 | 13.45 | 7.26 | 5.54 / 3.58 / 2.85 |
| W4A16 `nvfp4a16` | 5.27e-02 | 2.76e-02 | 0.341 | 0.252 | 6.39 / 0.0089 / 0.176 | 810 | 13.45 | 7.21 | 6.25 / 3.86 / 3.52 |
| W4A16 + pack-scale mse | 4.62e-02 | 2.85e-02 | 0.296 | 0.232 | 5.53 / 0.0126 / 0.162 | 810 | 13.45 | 7.21 | (= W4A16) |
| FP8 down + W4A16 gate_up | 4.58e-02 | 3.18e-02 | 0.283 | 0.153 | 4.94 / 0.0078 / 0.105 | 1021.5 | 16.96 | 7.62 | 6.76 / 4.48 / 4.23 |
| **mixed** (FP8 down + FP8 gate_up enc3/10/13, W4A16 rest) | 3.24e-02 | 2.10e-02 | 0.268 | 0.037 | 3.84 / 0.0079 / 0.017 | 1021.5 | 18.19 | 7.62 | 7.20-7.47 / 4.69-4.92 / 4.24 |
| mixed + pack-scale mse | 3.41e-02 | 2.09e-02 | 0.239 | 0.175 | 5.10 / 0.0088 / 0.122 | 1021.5 | 18.19 | 7.62 | (= mixed) |
| mixed with W4A4 gate_up of the FP4 layers | 4.07e-02 | 1.80e-02 | 0.313 | 0.078 | 3.02 / 0.0061 / 0.053 | 1021.5 | 18.19 | 7.67 | 6.95 / 4.72 / 3.91 |

QAT weights `praxis-1-droid-nvfp4` (oracle `tmp/praxis_ref/real_qat_fp32`)
| variant | dec_vel | ae_out | video | actions | xyz mm / rot6d / grip (rms err) | ring slot MiB | stream GiB/pass | peak GiB (+0.13 ctx) | latency s: plain / skip-last / paired+skip |
|---|---|---|---|---|---|---|---|---|---|
| bf16 experts (ours) | 1.78e-02 | 2.19e-02 | 0.238 | 0.138 | 4.65 / 0.0077 / 0.101 | 2880 | 47.8 | 11.03 |  |
| W4A4 `nvfp4` (what QAT trained for) | 5.62e-02 | 2.84e-02 | 0.366 | 0.230 | 6.23 / 0.0165 / 0.169 | 810 | 13.45 | 7.26 | 5.54 / 3.58 / 2.85 |
| W4A16 `nvfp4a16` | 4.50e-02 | 3.55e-02 | 0.360 | 0.090 | 6.04 / 0.0123 / 0.060 | 810 | 13.45 | 7.21 | 6.25 / 3.86 / 3.52 |
| **mixed** | 3.64e-02 | 1.60e-02 | 0.419 | 0.092 | 4.31 / 0.0183 / 0.062 | 1021.5 | 18.19 | 7.62 | 7.47 / 4.92 / 4.24 |
| mixed + pack-scale mse | 3.38e-02 | 2.45e-02 | 0.285 | 0.146 | 6.39 / 0.0111 / 0.104 | 1021.5 | 18.19 | 7.62 | (= mixed) |
| mixed with W4A4 gate_up of the FP4 layers | 4.28e-02 | 3.21e-02 | 0.469 | 0.200 | 3.88 / 0.0273 / 0.145 | 1021.5 | 18.19 | 7.67 | 6.95 / 4.72 / 3.91 |

Reading the numbers (single seed, K'=2; **dec_vel is the stable metric**, ae_out and the sampled video/action numbers move by +-1e-2 / +-0.1 with route flips, the two fp32 oracles differ by 0.51 in sampled actions):
- **mixed is the best accuracy under the 8 GB budget**: dec_vel 7.8e-2 -> 3.2e-2 (base weights, 2.4x; bf16 experts 2.0e-2) and 5.6e-2 -> 3.6e-2 (QAT weights, 1.5x; bf16 1.8e-2); peak 7.62 GiB (+0.13 GiB ctx = 7.75 GiB < 8; cuMemGetInfo-measured 7.84 GiB incl. ctx).
  Cost: 18.2 GiB of banks per pass instead of 13.5 (the card is PCIe-bound) -> 7.2-7.5 s/call (+35%), 4.2 s paired+skip (+49%) vs 5.5 / 2.85 s.
- f16 activations alone (`nvfp4a16`) are the cheap step: 7.8e-2 -> 5.3e-2 / 5.6e-2 -> 4.5e-2, +13% latency (the f16 mma runs at the ~25-30 TFLOPs fp16 rate of the card; the FP4 mma is 3x faster), same memory.
- Why not more FP8: gate_up+down FP8 in all layers needs a 1.44 GiB slot per layer; measured peak 8.44 GiB (ring 2 x 1.44 GiB, M3 run) -> fails the budget; expert-group streaming (a 1.44 GiB layer streamed as 2 groups through a 1.02 GiB slot,
  bit-identical) is what lets the three sensitive layers be FP8 inside it. Decoder layers (8.7k tokens = 3 token chunks) would re-stream their groups per chunk, not done (their gate_up error is 0.05-0.065, moderate).
- `PRAXIS_PACK_SCALE=mse`: -12% dec_vel on plain W4A16 (5.3e-2 -> 4.6e-2, weight RMSE 9.4% -> 8.5%), within noise on mixed (3.2e-2 -> 3.4e-2 base, 3.6e-2 -> 3.4e-2 QAT); free at runtime, left opt-in.
- Did not help / refuted: per-expert activation global scale (no-op), calibrated FC1 output scales (SM100-only in praxis), more block-scale search (FP4 weights are at the Lloyd-Max limit), W4A4 gate_up inside the mixed plan
  (dec_vel 4.1e-2 vs 3.2e-2 for -0.3 s), the first bf16-mma W4A16 kernel (4x slower than the FP4 kernel; replaced by the f16 one, which also halves the activation error). **The QAT weights do not lower the per-layer NVFP4
  error against their own bf16 forward** (`moe_attr.py` on QAT activations, enc3/7/10/13/dec2: W4A4 0.15/0.11/0.21/0.27/0.12, same three sensitive gate_up banks 0.083/0.123/0.164, W4A16 mean 0.127): QAT trains the model to *work*
  with FP4 weights+activations, it does not make the FP4 forward numerically closer to a bf16 forward. Task-level quality of any variant (praxis reports NVFP4-QAT ties bf16 overall, loses on `hard`) is NOT measured here;
  QAT was trained for W4A4, so for the QAT weights `nvfp4` is the configuration the authors validated and `mixed` the one closest to the unquantised forward.
- Usage: `PRAXIS_EXPERT_FMT=mixed [PRAXIS_PACK_CACHE=<file>] cuda/praxis/praxis_e2e ... --policy $PRAXIS_ROOT/praxis-1-droid[-nvfp4]` (no `PRAXIS_NVFP4=1` needed; the old env keeps meaning W4A4). The mixed plan packs into its own cache file
  `<cache>.p3c86754d5e664762` (19.5 GB, written once; the plain-NVFP4 cache is untouched). Cold pack 160-380 s (disk), 25-40 s from cache. Reproduce: `LAT=3 cuda/praxis/run_expert_variant.sh <label> <model> <oracle> PRAXIS_EXPERT_FMT=mixed`.



### Not done / limits
fp32 `PRAXIS_EXACT=1` path not run on the real model (fp32 naive kernels, expected far too slow); no real-observation (DROID episode) run, only a synthetic scene; K'=1 first call (depth 0) not run through our
runner; the antialiased tile resize is not bit-compared to torch (the ref's `canvas_call*.npy` could be); client-frame pose conversion; paired CFG with different text lengths; ctx-cache and the VAE decode were not run on the real model.

### Text conditioning in-process, session mode, C tokenizer (this section supersedes "Qwen child process" above)
- **In-process Qwen** (default): `praxis_qwen.c` (+ `praxis_qwen.h`) is a separate translation unit linking `cuda/llm/cuda_llm_runner.c` into `praxis_e2e` (same primary CUDA context, one cuew; no symbol
  conflicts; built by `make -C cuda/praxis praxis_e2e`, objects in `tmp/praxis_obj`). All cache misses of a call (cond prompt + blank caption `" "`) are computed in ONE
  `cuda_llm_load_weights_qwen35_safetensors` -> `cuda_llm_prefill_hidden_ex` (x N) -> `cuda_llm_free` cycle, so Qwen's 2.7 GB is released BEFORE the VAE/policy load. Verified with per-process
  accounting (nvidia-smi): process memory after the free = +0.000..0.004 GB vs before the load (nothing leaks); Qwen peak 2.72 GB over the context (same as the child). `--qwen-spawn` keeps the
  old child-process path (`--qwen-bin`). Cache: in memory (key = prompt string or ids hash; repeated prompt = 0 ms, no Qwen load) and on disk (`--text-cache`, key = fnv(int32 ids); old cache files with the
  previous int64 hash are not reused).
- **Text cond is bit-identical** to the child-process cond on the real inputs (prompt "pick up the red cup and put it in the sink" + blank caption): rel-L2 0.0, cos 1.0, max|diff| 0 (both shapes,
  11x2048 and 1x2048; ids from the C tokenizer vs Python ids).
- **C tokenizer** (`--prompt "text" [--prompt-uncond " "] [--tokenizer tokenizer.json]`): `praxis_tok_*` builds the `common/bpe_tokenizer.h` vocab straight from the HF `tokenizer.json` (vocab, merges,
  added tokens matched literally, Qwen3.5 pre-tokenizer regex as implemented there; the Split regex in the json is assumed, not parsed). `--tok-test FILE` prints ids per line (CPU only). Checked against
  HF `AutoProcessor.tokenizer(add_special_tokens=True)` on 34 varied prompts (plain, blank, leading/trailing/multiple spaces, tabs/newlines, digits, punctuation/contractions, URLs/code, accents,
  Japanese/Korean/Greek/Russian, emoji, special-token strings like `<|im_start|>`/`<think>`): **33/34 identical**. The one mismatch is decomposed input (`e` + U+0301): HF runs an NFC normalizer
  first, which is NOT implemented (a warning is printed for combining marks / Hangul jamo / kana voicing marks; precomposed text, i.e. all normal prompts, is exact). Tokenization 5.6 ms for 34 lines
  after loading (load of the 12 MB json ~0.1-0.2 s, once).
- **`--session FILE|-`**: one prompt per line (blank and `#` lines skipped). Policy weights, VAE weights and the prompt caches stay resident across calls; each call re-tiles the frames, VAE-encodes,
  samples and writes `<out>/call_NNN/`. Per-call breakdown and cold/warm summary are printed (`[call N]`, `== session`). Caveat for the memory target: with the VAE weights and the policy both resident, the
  VAE-encode scratch (2.5 GB) overlaps the policy (6.9 GB), so a session call peaks above the 8 GB of the single-shot (sequenced) mode; single-shot mode is unchanged (VAE freed before the policy loads).

## Closed-loop sim: praxis' own MuJoCo DROID sim driven by our CUDA runner (server + client)
Pieces (all new): `praxis_server.c` (persistent policy process, binary protocol on stdin/stdout, `make -C cuda/praxis praxis_server`), `praxis_client.py` (drop-in stand-in for `PraxisPipeline`/`Session`),
`run_sim_cuda.py` (= `scripts/run_sim.py` of praxis-1-oss with `policy.type: model` replaced by our client), `check_client_replay.py` (reference replay), `praxis_trim` in the runner.
Install once (sim code is the target model's own repo): `VIRTUAL_ENV=~/.venv uv pip install "mujoco>=3.16,<3.17" gymnasium pyyaml imageio imageio-ffmpeg $PRAXIS_ROOT/praxis-1-oss/packages/praxis-sim`
(the Menagerie robot assets are downloaded + SHA-256-checked by the sim into `~/.cache/praxis/sim` on first use, 40 MB). Headless rendering works with `MUJOCO_GL=egl` (renders on the NVIDIA GPU, no black frames
although the CUDA server is on the same card).

### M1 - correctness of the real session flow
**(a) K'=1 (first call, depth 0) on the real checkpoint.** Runner bug found and fixed: in the action expert's cross-attention `pr_copyrows` was launched for the clean tap rows with `Nc == 0` (grid of 0 blocks ->
`launch failed ... (1)`); now guarded (`cuda_praxis_runner.h`, `praxis_expert`). `test_cuda_praxis` accepts a ref dir without `in_clean.npy` (K'=1). The real K'=1 oracle did not exist for fp32, so
`ref/praxis/gen_real_ref.py --fp32 --tag _fp32 --calls 1` was run (250 s) -> `tmp/praxis_ref/real_fp32/call1` (torch fp32; torch bf16 call1 is `tmp/praxis_ref/real/call1`).
`make -C cuda/praxis test_praxis_real_k1` (mixed) or by hand; rel-L2 against the torch fp32 call-1 oracle (pass 0 + 2-step sampler, seed 0, same noise):
| tensor | torch bf16 | ours bf16 experts | ours mixed | ours NVFP4 W4A4 |
|---|---|---|---|---|
| dec_vel | 2.99e-2 | **1.22e-2** | 3.14e-2 | 1.08e-1 |
| ae_out | 8.7e-3 | 4.1e-3 | 1.27e-2 | 2.06e-2 |
| sampled video (cfg 4) | 0.347 | 0.218 | 0.246 | 0.73 |
| sampled actions | 0.344 | **0.062** | 0.043 | 0.505 |
The K'=1 path is correct (bf16-expert error 1.2e-2 is the same class as K'=2's 2.0e-2); the call-1 numbers are noise-floor dominated exactly like call 2. (Mixed: 9.5 s for the whole K'=1 pass+sample test incl. load.)
**(b) Context of each call = what praxis `SingleArmServeSession.prepare` builds** (read from `praxis/single_arm.py` + `robot_io.ContiguousStreamContext`, and now executed by that very code in the client, see below). With
`max_context_chunks 1`, `exec_horizon 24`, the sim's history stack of the last 25 observations (executed frames + current):
- call 1 (stack of 1): depth 0, K'=1. Anchor = VAE latent of the current frame (a 1-frame clip -> 1 latent), no cleans, proprio_all = current pose only. Noised chunk = 6 latents; actions: token 0 gives 1 row, the 21 rows (1 + 5x4).
- call 2 (the stack holds the 21 executed rows + current = 22): the stream is empty -> `need` = 21 -> clip = stack rows [-22:-1] = the 21 frames from the call-1 observation up to the one before the current frame, tiled and VAE-encoded
  in one pass (6 latents) = the clean chunk; anchor = its first latent (`dedup`); the current frame is NOT encoded (only its pose is used: proprio_all = [pose at the clean chunk start, current pose]). K'=2, 24 rows.
- calls n >= 3: the stream appends the 24 executed frames (rows [-25:-1]) and keeps the newest 21 (`max_raws = 21` for 1 context chunk): the window slides, its anchor is the first retained frame, the retained 21 are encoded again
  (6 latents), the boundary pose is that of the first retained frame. Seed of chunk k of an episode with scene seed s: `session.seed + 1000*s + k` (`seeded_normal`, torch CPU generator, salts 0/1 for video/action noise).
**(c) Client-frame pose conversion** is praxis's own code (`single_arm.pose8_to_model` / `rel10_to_wire` / `pose_to_rig`, quat <-> matrix, rot6d <-> matrix, relative-to-current -> absolute `robot_command.cartesian_pos`
with `R_flat9`, `target_grip`, `timestamp = t/15`, `droid_impedance` mode payload; client frame by the observation `descriptor`: `praxis.sim` = turned flange (sim), "" = FR3 hand as in the reference dumps) executed
in the Python bridge, which is exact by construction and checked: feeding the reference's own `pred_action` tokens of call 1 and call 2 through the session code reproduces `commands_pose.npy`/`commands_grip.npy` with
max|diff| **0.0** (21 and 24 rows, timestamps 0, 1/15, ...). The un-normalisation (q99 / clip) also matches `action_rel.npy` of the reference.
**Reference replay** (`check_client_replay.py`, the reference observation of `gen_real_ref.py`, call 1 = 1 frame, call 2 = 22 frames, mixed experts, prompt "pick up the red cup and place it on the plate"):
canvas of the encoded window **pixel-identical** to `canvas_call2.npy` (tiles from praxis's CPU `tile_clip`); VAE latents (our CUDA Wan2.2 VAE, uint8 tiles) vs the torch `encode_eager` latents: anchor 1.96e-3, clean0 1.70e-3 rel-L2;
proprio_all 0 diff; sampled action tokens vs torch fp32 oracle: call 2 6.7e-2 (un-normalised action rows 2.9e-2; commands: xyz max 12.5 mm, R 0.034, grip 0.09), call 1 0.26 (action rows 0.14; xyz max 5.4 mm; grip 0.25) - against the
torch bf16 reference 0.46 / 0.16, i.e. within the 0.34-0.5 noise floor of torch-bf16-vs-fp32 itself (call 1's number is bigger than the 4.3e-2 of the harness above because here the text conditioning and the VAE latents are our own too).

### Server protocol / memory sequencing (`praxis_server.c`)
`cuda/praxis/praxis_server --policy DIR [--vae DIR --qwen DIR --stream 2 --text-cache DIR --vae-mode keep|reload --trim 0|1|2]` (from the repo root; expert format by env exactly like `praxis_e2e`:
`PRAXIS_EXPERT_FMT=mixed|nvfp4|nvfp4a16`, `PRAXIS_PACK_CACHE=<file>`; the mixed pack is `$PRAXIS_ROOT/praxis-1-droid.nvfp4.packcache.p3c86754d5e664762`, 19.5 GB, written by the first mixed run, 160-400 s cold).
Framed binary requests (`PRXQ`, op, length) on stdin, responses on a private dup of stdout (everything printf'd by the shared `praxis_e2e` helpers goes to stderr = the server log); every response carries ms / aux ms /
device GB peak over the idle context (cuMemGetInfo polled every 4 ms) / GB after. Ops: ENCODE (uint8 tile clip -> policy latent), SAMPLE (prompt + uncond prompt + K' + anchor/cleans/proprio/noise -> video + action tokens; the policy
is loaded lazily on the first SAMPLE and stays resident), TEXT (warms the Qwen3.5-2B conditioning cache; the Qwen weights are loaded and freed per cache miss), DECODE (latent -> uint8 frames), INFO, QUIT. Weights, pack, ring, text cache
stay resident across calls and across episodes/tasks.
Memory (the <8 GB target for a LONG-LIVED session): the policy alone peaks at 7.5 GB over the context during SAMPLE (own accounting 7.24 GB; ring 2 x 1.02 GB, weights 2.9 GB, workspace 1.4 GB), and the VAE encode needs 2.6 GB more, Qwen 2.7 GB. Measured
with the VAE kept resident: SAMPLE peak 8.89 GB, ENCODE 7.98 GB, both over the idle context = **above 8 GB**. What fixes it (`--vae-mode reload --trim 2`, the defaults of the client except vae-mode): before ENCODE/DECODE/TEXT the server
releases the policy's device workspace (`praxis_trim`, regrows on demand) and its expert ring (`praxis_release_stream`, `praxis_set_stream` re-creates it before the next SAMPLE), and the VAE is loaded per use from its fp16 cache and freed (~165-215 ms
per load). Measured in the closed loop (mixed, K'=2, device GB over the 0.13 GB idle ctx): ENCODE peak **6.26-6.63**, SAMPLE peak **7.54-7.69**, Qwen warm 2.86 (cold prompt), resident after a call 3.3-3.7 between calls -> **max instant 7.83 GB incl. context < 8 GB**.
Cost of the sequencing per call: ~0.2 s VAE reload + ~0.8 s encode (21 frames of 224x832) + ~0.9 s CPU tiling (praxis' torch resize) around 7.4 s (mixed) / 5.5 s (NVFP4) / 3.0 s (mixed, cfg video 1 + skip-last-video) of sampling.
**Python client**: `PraxisClient(checkpoint, sampler=SamplerSettings(...), env=..., vae_mode=..., trim=...)` has `.policy.config`, `.embodiment`, `.session(prompt, seed, max_context_chunks, control_mode, exec_horizon)` returning a session with
`.step(obs) -> StepResult(commands, action_rows_norm, action_rel, pred_video_latent, pred_action_norm, depth, seed, ...)` and `.decode_video(result)`: the observation->commands interface of `PraxisPipeline`/`Session`, so `praxis.sim.loop.ModelPolicy`
runs on it unchanged. The state machine/tiling/pose code is praxis's, the three heavy calls go to the server; torch here is CPU only.

### M3 - closed loop results (praxis sim, configs/sim_droid.yaml: droid_impedance, exec_horizon 24, 15 Hz, 2 Euler steps, t_shift 12.5, cfg video 4 / action 1, max_context_chunks 1, time limit 20.6 s = 309 ticks, stop 1 s after success)
Commands (from the repo root; outputs under `$PRAXIS_OUT/praxis_sim/<name>/`: `episode_NNN.{mp4,json,npz}`, `summary.json`, `calls.json` (per-call timing/peaks), `server.log`):
```
MUJOCO_GL=egl ~/.venv/bin/python -I cuda/praxis/run_sim_cuda.py --config $PRAXIS_ROOT/praxis-1-oss/configs/sim_droid.yaml --vae-mode reload --expert-fmt mixed \
   --set task.name=cube_on_plate --set task.prompt=null --set "episodes.seeds=[0,1,2,3,4,5]" --set output.dir=$PRAXIS_OUT/praxis_sim/cube_on_plate_mixed
# --expert-fmt nvfp4 | mixed | bf16(parity only);  --cfg-video 1.0 --skip-last-video (fast, no predicted-video panel);  --no-decode;  --policy scripted|hold (the sim's own, smoke tests)
```
Smoke tests with the sim's own scripted policy (lift_cube, cube_on_plate, move_block): all succeed (86 / 141-142 / 131-135 ticks), frames render on the NVIDIA GPU (EGL), no bad-view ticks.
Results (praxis' own sim, our CUDA policy; success = the sim's success check; per-call = full `Session.step` wall time on this RTX 5060 Ti incl. CPU tiling 0.9 s + VAE reload/encode 1.0 s; calls: depth 0 then depth 1):
| run | experts / sampler | episodes (seeds) | success | mean / p50 call |
|---|---|---|---|---|
| lift_cube | mixed, cfg video 4 | 3 (0-2) | **3/3** | 8.9 / 8.2 s |
| cube_on_plate | mixed, cfg video 4 | 6 (0-5) | **6/6** | 8.5 / 8.2 s |
| cube_on_plate | NVFP4 W4A4 | 6 | 5/6 (seed 0 fails) | 6.7 / 6.4 s |
| cube_on_plate | mixed, cfg video 1 + skip-last-video ("fast") | 6 | 6/6 | 4.4 / 3.9 s |
| move_block | mixed | 6 | 5/6 (seed 2) | 8.4 / 8.2 s |
| move_block | NVFP4 | 6 | 4/6 (seeds 2, 4) | 6.4 / 6.4 s |
| move_block | mixed fast | 6 | 5/6 (seed 4) | 4.6 / 4.0 s |
| block_in_bowl | mixed | 3 | 2/3 (seed 1) | 8.8 / 8.3 s |
Per call (K'=2, median, mixed): tiling+prepare 0.85 s, VAE encode 0.75 s (+0.2 s load), sampler 7.36 s (NVFP4 5.47 s, mixed fast 2.99 s); first call (K'=1) 4.4 s NVFP4. Predicted-video decode (VAE decode + reload) was on for the mixed runs and is excluded from the call time.
Videos: `$PRAXIS_OUT/praxis_sim/<run>/episode_NNN.mp4` (overview | exterior | wrist on top, the model's predicted video decoded from our latents below), e.g. `lift_cube_mixed`, `cube_on_plate_mixed`, `move_block_mixed`.
Context only (not comparable: other sampler config/ckpt settings, 16 scenes x 2 samplers there): praxis EVAL.md lists the *earlier* checkpoint at 30/32 lift_cube, 31/32 cube_on_plate, 30/32 move_block, 32/32 block_in_bowl; the README's ~50% is the hard suite, not run here.
Our small samples (3-6 episodes per row, one sampler seed each) cannot separate mixed from NVFP4 (NVFP4 lost 1 episode on each of cube_on_plate/move_block versus mixed, within noise). Failures are behavioural (hesitation/regrasp, block not carried far enough); diagnosis below.
Honest limits: no episode was run through the torch pipeline closed loop (only single-call replays, see M4); the failing move_block seed 2 was NOT re-run with bf16 experts (that run was started and cancelled in favour of M4); the 16-scene suites / hard suite were not run; the server is one process per sim run (a 3-6 episode run reloads
the policy once, ~20 s from the pack cache, first-call text 1.1 s).

### M4 - the chunk-boundary jump: model behaviour, not a bridge bug
Observation: at every chunk boundary the commanded xyz jumps by ~25-31 mm (mean over runs: mixed 26-31, NVFP4 24-26, fast 26-28) versus a median per-tick step of ~6 mm; within-chunk step CV 0.43-0.48 (the torch reference commands of the real dumps: 0.39-0.50).
`$PRAXIS_OUT/praxis_sim/jitter.py <run dirs>` computes both from the episode npz.
1. **praxis's own path does nothing between chunks in chunked mode.** The sim loop (`loop.run_episode`) executes all rows of `Session.step` one per tick with no blending, no row skipping, no interpolation; our run uses that very loop and `SingleArmServeSession` unchanged.
   Smoothing/junction logic exists only in the serve path and is opt-in: `smoothing: savgol` (Savitzky-Golay over each chunk's relative xyz/grip, smooths inside the chunk, does not touch the junction) and real-time chunking `schedule=rtc` (`praxis/serve/rtc.py`: the answer repeats the plan prefix and a rebase offset decays over `rebase_decay` rows - exactly a junction fix). `configs/sim_droid.yaml` uses neither.
2. **Inputs are identical to praxis's.** `check_sim_replay.py` records the sim's real observation stack of call 2 (`run_sim_cuda.py --dump-obs`, cube_on_plate seed 1), runs praxis's own preprocessing + the PyTorch reference model (`ref/praxis/gen_real_ref.py --obs-npz ... --descriptor praxis.sim --calls 2`, bf16 AND fp32, offload) and our client:
   anchor rel-L2 1.9e-3, clean latent 2.0e-3 (our CUDA VAE vs torch), proprio_all 0.0, boundary poses and pose_now equal, noise seed identical (call 2 -> seed 1). The relative rows are applied to the current pose (`pose8_now`), the context is the 21 executed frames before the current one, as praxis builds it (section M1 b).
3. **The torch fp32 oracle on the same observation has the same jump**: first relative row |row0| = 34.7 mm, row1 35.2, row2 33.3 (fp32); ours mixed 33.0 / 33.2 / 31.2, ours NVFP4 27.7 / 26.7 / 24.4, while the median step is 10 mm. (The torch *bf16* run gives 6.3 mm for row0 - it is the outlier; torch-bf16 vs fp32 sampled-action error 9.8e-2 / xyz 28 mm rms, ours-mixed vs fp32 4.3e-2 / 13.5 mm rms, NVFP4 6.2e-2 / 13.6 mm.)
   At depth 0 the first row is ~3-4 mm from the current pose (it is the anchor frame), at depth >= 1 the model puts row 0 about 3-4 ticks of motion ahead of the measured pose. The arm tracks the commands (command-vs-measured lag 0.2-18 mm), so the jump really is a lunge. So it is what the checkpoint predicts in the chunked schedule (probably built for the real-time schedule where row k is due at observation time + k/fps and the plan prefix is rebased).
4. **Mitigation measured, now the DEFAULT of `run_sim_cuda.py` (`--rebase 8`; pass `--rebase 0` for praxis's own unmodified chunk behaviour)** (not praxis behaviour; `run_sim_cuda.py --rebase R`: shift the new chunk's xyz so row 0 continues the previous command (last + last step), decaying linearly to 0 over R=8 rows; rotation/grip unchanged; cfg video 1 + skip-last-video config, mixed):
| run | success | boundary jump mean | within-chunk step CV |
|---|---|---|---|
| cube_on_plate fast (baseline) | 6/6 | 25.7 mm | 0.45 |
| cube_on_plate fast + rebase 8 | 5/6 (seed 0 timed out) | **6.3 mm** (= one step) | 0.47 |
| move_block fast (baseline) | 5/6 | 28.3 mm | 0.46 |
| move_block fast + rebase 8 | 5/6 (same seed 4 fails) | **5.9 mm** | 0.46 |
Rebase removes the lunge at no measurable success cost on these 12+12 episodes (it does not change the within-chunk jitter, CV 0.45-0.48, which is model sampling noise; praxis' savgol option would be the knob for that). Not tried: more Euler steps, action CFG, savgol in the sim loop (each is a few-line change in `CudaModelPolicy`, untested).

### Evaluation (N=12)
Paired closed-loop evaluation of the CUDA runner in praxis' own MuJoCo DROID sim (`configs/sim_droid.yaml`, 309 ticks = 20.6 s, success = the sim's task check). Compact tables of every run: `cuda/praxis/eval/RESULTS.md` (generated with `eval/summarize.py`).

**(a) Setup.** Seeds 100-111 for every run (the same 12 initial states in every variant; the earlier 3-6 episode runs used seeds 0-5). Variants (all `--expert-fmt mixed` unless noted, `--vae-mode reload`, praxis-1-droid checkpoint, one server process at a time on the RTX 5060 Ti):
- **A** (baseline, "fast"): mixed experts, `--cfg-video 1.0 --skip-last-video`, `--rebase 8` (the `run_sim_cuda.py` default), 2 Euler steps, t_shift 12.5, action CFG 1.0.
- **B**: A with NVFP4 W4A4 experts. **C**: "full" = cfg video 4 + decoded predicted video (rebase 8). **D**: A with `--rebase 0` (praxis' own chunk behaviour).
Rerun (skips runs that already have a `summary.json`; outputs `$PRAXIS_OUT/praxis_sim2/<name>/`, videos pruned to failures + first success by `eval/prune_videos.py`):
```
bash cuda/praxis/eval/e1.sh            # A/B/C/D + extra tasks (lift_cube cube_on_plate move_block block_in_bowl block_out_of_bowl stack_block press_button)
bash cuda/praxis/eval/e2.sh 1          # knob sweep on block_out_of_bowl;  e2.sh 2 = combo + cube_on_plate validation + prompt variants;  e2.sh 3 = baseline repeat
~/.venv/bin/python -I cuda/praxis/eval/summarize.py $PRAXIS_OUT/praxis_sim2/A_*                      # Wilson CI, ticks, latency, boundary jump, step CV, peak GB
~/.venv/bin/python -I cuda/praxis/eval/summarize.py --pair <dirA> <dirB>                                   # paired exact McNemar (same seeds)
```
(a single run by hand: `MUJOCO_GL=egl ~/.venv/bin/python -I cuda/praxis/run_sim_cuda.py --config .../sim_droid.yaml --vae-mode reload --expert-fmt mixed --cfg-video 1.0 --skip-last-video --set task.name=<task> --set task.prompt=null --set "episodes.seeds=[100,...,111]" --set output.dir=<dir>`; extra flags `--num-steps N --cfg-scale S --savgol W --sg-order 3 --rebase R`.)

**(b) E1 results** (success with Wilson 95% CI; mean ticks to success; latency = mean Session.step wall time incl. tiling + VAE encode; boundary jump = |command jump| at chunk boundaries; step CV = within-chunk step std/mean):
| task / variant | success | 95% CI | ticks (succ) | call mean s | boundary jump mm | step CV |
|---|---|---|---|---|---|---|
| lift_cube A / B | 11/12 / 11/12 | 65-99 / 65-99% | 152 / 174 | 4.17 / 3.31 | 5.5 / 5.4 | 0.46 / 0.46 |
| cube_on_plate A / B / C / D | 10/12 / 9/12 / 11/12 / 8/12 | 55-95 / 47-91 / 65-99 / 39-86% | 205 / 191 / 164 / 186 | 4.00 / 3.21 / 8.23 / 4.01 | 6.1 / 5.8 / 5.7 / **25.1** | 0.45 / 0.44 / 0.47 / 0.45 |
| move_block A / B / C / D | 11/12 / 10/12 / 11/12 / 11/12 | 65-99 / 55-95 / 65-99 / 65-99% | 175 / 199 / 173 / 179 | 4.00 / 3.19 / 8.23 / 4.01 | 5.7 / 5.5 / 5.9 / **28.7** | 0.46 / 0.46 / 0.46 / 0.46 |
| block_in_bowl A / B | 10/12 / 11/12 | 55-95 / 65-99% | 196 / 210 | 4.02 / 3.19 | 6.0 / 5.9 | 0.45 / 0.45 |
| stack_block A | 10/12 | 55-95% | 250 | 4.03 | 5.5 | 0.47 |
| press_button A | 9/12 | 47-91% | 152 | 4.01 | 4.9 | 0.48 |
| block_out_of_bowl A (2 repeats) | 3/12, 3/12 | 9-53% | 196 / 185 | 4.15 / 4.03 | 5.5 / 5.6 | 0.46 / 0.46 |
Paired exact McNemar (same 12 seeds; discordant pairs first-only / second-only): cube_on_plate A vs B 3/2 p=1.0, A vs C 1/2 p=1.0, A vs D 2/0 p=0.5; move_block A vs B 2/1 p=1.0, A vs C 1/1 p=1.0, A vs D 1/1 p=1.0; lift_cube A vs B 0/0 p=1.0; block_in_bowl A vs B 1/2 p=1.0.
Memory: the server's own peak is 7.24 GB (mixed, `policy_own_peak_gb` in `server.log`/`calls.json`; 7.72-7.77 GB device used incl. the 0.13 GB CUDA context and Qwen/VAE sequencing), 6.88 GB / 7.37 GB device for NVFP4. The 10.71 GB "peak" in the first summary of `A_block_out_of_bowl` is an artefact: `peak_gb_sample` comes from `cuMemGetInfo` on the whole device and included other processes (a concurrent job), while the same run's server reports `policy_own_peak_gb=7.238` and `dev_used_gb=7.739` (same as every other run; the repeat `A2_block_out_of_bowl` shows 7.72). Latencies of part of the E2 runs and of some E1 runs were measured while another job rendered EGL on the same GPU; `server_sample_ms` (the GPU sampler time) is not affected (2990 ms median in every A-type run).
**Conclusion (honest):** at N=12 (CI half-width about +-20 points) none of the success differences between A, B, C and D is statistically significant (all paired p >= 0.5); the NVFP4 (B), the full setting (C) and the rebase (A vs D) are indistinguishable in success. The only effect that is clear is the rebase on the chunk-boundary jump (A 4.9-6.1 mm vs D 25-29 mm; a mechanical effect, not a statistical one at 12 episodes); it does not change the within-chunk step CV (0.44-0.48 everywhere). B is ~20% faster per call (3.2 vs 4.0 s), C is 2x slower (8.2 s, decoded video).

**(c) E2 knob sweep** (block_out_of_bowl = weakest task, baseline A 3/12 and its unchanged repeat A2 3/12 (success seeds {104,106,111} vs {105,106,111}: the seed-level noise of the sim is ~1 flip each way; the baseline is not an outlier); cube_on_plate A 10/12 as the "do no harm" check; pairing vs the A run of the same task):
| knob (one at a time) | success | 95% CI | McNemar vs A (A-only / knob-only, p) | ticks (succ) | call mean s | sampler ms | boundary jump mm | step CV |
|---|---|---|---|---|---|---|---|---|
| A baseline | 3/12 | 9-53% | - | 196 | 4.15 | 2990 | 5.5 | 0.46 |
| A2 (repeat of A) | 3/12 | 9-53% | 1/1, p=1.0 | 185 | 4.03 | 2990 | 5.6 | 0.46 |
| 4 Euler steps | 5/12 | 19-68% | 1/3, p=0.63 | 204 | 7.58 | 6611 | 5.0 | 0.46 |
| action CFG 1.5 | 6/12 | 25-75% | 2/5, p=0.45 | 195 | 6.88 | 5886 | 5.9 | 0.46 |
| action CFG 2.0 | 6/12 | 25-75% | 0/3, p=0.25 | 208 | 6.86 | 5888 | 5.9 | 0.45 |
| Savitzky-Golay 5 | 4/12 | 14-61% | 1/2, p=1.0 | 231 | 4.05 | 2992 | 5.1 | 0.45 |
| Savitzky-Golay 9 | 2/12 | 5-45% | 2/1, p=1.0 | 240 | 4.04 | 2992 | 4.1 | 0.46 |
| Savitzky-Golay 13 | 7/12 | 32-81% | 0/4, p=0.125 | 264 | 4.05 | 2992 | 4.0 | 0.45 |
| 4 steps + CFG 1.5 | 4/12 | 14-61% | 2/3, p=1.0 | 180 | 14.05 | 13188 | 5.0 | 0.45 |
| prompt "Take the red block out of the bowl." | 5/12 | 19-68% | 0/2, p=0.5 | 233 | 4.04 | - | 5.6 | 0.46 |
| prompt "Pick up the red block and put it on the table." | 5/12 | 19-68% | 1/3, p=0.63 | 241 | 4.01 | - | 5.4 | 0.46 |
Validation on cube_on_plate (A 10/12): 4 steps 9/12 (p=1.0), action CFG 1.5 10/12 (p=1.0, same discordance 1/1), action CFG 2.0 10/12 (p=1.0), i.e. no harm, no gain.
Notes: (1) 4 Euler steps are valid with `window_joint` (euler only; num_steps is a free argument of the sampler, t_shift 12.5 unchanged; the recommended config is 2) and cost 2.2x in the sampler (3.0 -> 6.6 s). (2) Action CFG is implemented end to end (client -> `SAMPLE` cfg_action -> `praxis_sample_window_joint_ex2`), needs the unconditional (blank caption) action pass; skip-last-video still applies, but the measured sampler time doubles (2.99 -> 5.89 s, i.e. a second backbone pass per step, more than the 3.16 -> 3.43 s of the paired synthetic benchmark), 6.9 s per call. (3) Savgol (window 9/order 3 = praxis' serve default) did lower the boundary jump (5.5 -> 4.0-4.1 mm) but NOT the step CV (0.45-0.46); the CV is the model's sampling noise and is dominated by the per-tick step size distribution, not by high-frequency chunk noise. Success went 2, 4, 7 for windows 9, 5, 13: no monotone trend, so this is noise; it also lengthens the success time (231-264 ticks vs 196), consistent with smoothing slowing the motion. (4) The best "combos" do not add up (4 steps + CFG 1.5 = 4/12 at 14 s/call).
**Nothing is significant.** The largest paired difference (SG13, 0 vs 4 discordant pairs, p=0.125) and the CFG variants (5-7/12 vs 3/12) are suggestive of at most a modest gain of action CFG (both scales 6/12), but they are within the noise shown by the identical repeat and the sweep of savgol windows (2/4/7). Action CFG costs +2.9 s per call, so it is not recommended as default without a bigger sample (e.g. 36+ seeds on block_out_of_bowl); default left unchanged (A).

**Why block_out_of_bowl is weak (failure analysis of the 9 failed A episodes, npz/json; 3 seeds' videos kept in `A_block_out_of_bowl/`).** Task (`praxis_sim/suite.py`): start with the block inside the bowl; success = block >13 cm from the bowl centre, on the table (z within 1 cm) and no finger pad touching it, within 20.6 s. The policy reaches and grasps in every episode (min reach 1-15 mm, grasp for 51-179 ticks, lift up to 7-11 cm, no IK misses, tracking error p50 < 2 mm). The failures are in the carry/release phase: of the 9 failures, 3 (103, 105, 109) did carry the block >13 cm but were still holding it (grip 0.41 = closed on the block, z 16-86 mm above the table) at the 309-tick timeout; 4 (100, 101, 102, 110) lifted but never got the block out of the 13 cm radius (it dithers over the bowl; final distance 17-42 mm); 2 (107, 108) dropped the block back into the bowl (z at rest) and did not regrasp. The successes finish at 131-218 ticks, so the 20.6 s limit is tight: it is a slow/hesitant grasp-carry-release, not a perception or tracking failure. The 20.6 s limit and the 13 cm criterion are the task's defaults (praxis `EVAL.md` lists 28/32 for the earlier checkpoint; not comparable). The task has no alternative prompt in the suite (the only prompt is the default); two paraphrases (above) gave 5/12 each (p >= 0.5): the phrasing is not the limiter.

**(d) HIP/AMD results (see `rdna4/praxis/README.md`).** The same server protocol runs on the RX 9070 XT (W4A16/FP8 "mixed", 7.65 GB, ~7.25 s plain call, ~3.5 s per call end to end in the fast variant). 12-seed lift_cube (seeds 100-111): HIP 10/12 vs CUDA 11/12, paired McNemar p=1.0; seeds 101/102 on repeat: HIP 6/11 vs CUDA 6/6 (chosen because they failed; a possible mild grasp-timing disadvantage of W4A16 is unproven). Both servers are bit-deterministic for identical inputs; the HIP-vs-CUDA gap on the same observation (pred_action 1.3e-2..7.3e-2, command xyz rms 2-25 mm) is the size of either backend's distance to the torch fp32 oracle; no bug found.

**(e) Limitations.** N=12 per cell: a Wilson CI is about +-20 points, a paired McNemar test needs roughly >= 5 discordant pairs in one direction to reach p < 0.05, so only large effects are detectable. The sim is synthetic (MuJoCo, rendered scenes), not the real robot; the sim itself is not exactly repeatable (initial state depends on episode order; the A repeat flips one seed each way). The EVAL.md numbers (30/32 etc., earlier checkpoint, 16 scenes x 2 samplers) are not comparable to ours (different checkpoint/sampler settings and scene count). Quantisation variants (NVFP4 W4A4, mixed, HIP W4A16) are indistinguishable at task level here but not proven equal. The sweep is one knob at a time on one weak task with one sampler seed per episode; knob effects below ~30 points are not resolvable. Latency of part of the runs was measured with a concurrent EGL job on the same GPU (use `server_sample_ms`, unaffected, for the GPU time).
