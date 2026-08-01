# Resume: A64FX Gemma-4 31B kernel optimization

Handoff for continuing the int8/fp16/fp32 kernel optimization on native A64FX
(Fugaku). Model: `~/models/gemma4/31b-qat/gemma-4-31B_q4_0-it.gguf` (Q4_0,
17 GB; lm-head `token_embd.weight` is Q6_K). Branch `gemma4`. All diagnostic
tools live in `a64fx/bench_q4_0_matvec/` (binaries `.gitignore`d).

Full detail in memory: `reference_a64fx_int8_sdot_roofline.md`. Read it first.

## Build/run pattern (every tool)
```
cd a64fx/bench_q4_0_matvec
fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp -D_GNU_SOURCE \
    -I../../common [-I../int8-new] <tool>.c [objs] -lm -o <tool>
NUMA_DISTRIBUTE=1 OMP_NUM_THREADS=48 OMP_PROC_BIND=close OMP_PLACES=cores ./<tool> <model.gguf> [tensor]
```

## HARD-WON GOTCHAS — do NOT relearn these
1. **OOM**: full int8 cache (~30 GB) + Q4_0 (17 GB) OOMs the 32 GB node. Work
   ONE tensor at a time. `gguf_open(path,1)` faults the WHOLE file (MAP_POPULATE)
   → OOM; ALWAYS set `NUMA_DISTRIBUTE=1` (lazy mmap) and copy just the one tensor,
   then `gguf_close`.
2. **NUMA**: `aligned_alloc`/malloc reuse faulted glibc-arena pages → per-thread
   first-touch is a no-op → cross-CMG reads (190 vs 1200 GIOPS). ALWAYS
   `mmap(MAP_ANONYMOUS)` fresh + first-touch with the SAME thread split the kernel
   uses. `close` binding = honest per-CMG; `spread` inflates low-thread numbers.
3. **DCE**: store kernel output to an escaping buffer or the compiler deletes the
   whole matvec/GEMM (0 ms). For rep loops, read C per-rep into a volatile sink.
4. **asm-kernel ABI**: the int8-new asm kernels clobber callee-saved d8-d15
   (z8-z15 accumulators) and don't save x25-x28 — corrupts caller doubles kept
   live across the call (NaN timings). Either use an integer (cntvct) timer in
   volatile memory, or fix the kernel (done for kernel_6x4_opt.S). `/local` was
   wiped on session restart — `mkdir -p /local/u14346` first.
5. `llm_runner` does NOT build (a64fx/vlm `struct vision_model` incomplete,
   pre-broken). transformer.h is self-contained — use a standalone harness
   (`mini_decode.c`: transformer_load + transformer_set_threads + numa_setup +
   build_panels, else single-threaded).

## Established roofline (measured, NUMA-clean)
- int8 SDOT compute peak: **512 GIOPS/core** (~460 register-resident = 90%);
  fp32 FMA **128 GFLOPS/core**; fp16 FMA 2× fp32 (**A64FX has NO bf16 dot** —
  vdotq_s32 SIGILLs).
- HBM: **57 GB/s/core, ~230/CMG, ~852/node** (close binding).
- Matvec is BW-bound; prefill GEMM is compute-bound (target 80% FLOPS).

## DONE (matvec — the dominant decode cost)
- int8 SDOT matvec NUMA-clean: **73%/CMG, 633 GB/s node, 1265 GIOPS** (numa_matvec.c).
- Per-tensor int8 = 6.7× over fp32-fused but **8.9% err** (small-d blocks lost);
  **per-row int8 = 3.7% err, same fast kernel**; **Q8v2 per-block = 4.5×, 1% err**
  (64-wide pair SDOT, per-lane d via svsel, fp-accumulate, 1 svaddv/row). See
  layer_opt.c. Production decode uses the SLOW fp32-dequant path (tf_matvec_q4_0_rows,
  ~1.7 GB/s/core) — int8 is the 5.4× lever but capacity-gated (30 GB).
- Q6_K lm-head matvec (lmhead_opt.c): int8 SDOT **646 GB/s = 102% ceiling**,
  per-row 2.5% err. token_embd.weight 5376×262144, runs every decode token.

## DONE (prefill GEMM)
- Production tf_gemm_q4_0 (gemm_existing.c): **11% peak, degrades with M**
  (re-dequants per 4-token block, no amortization).
- Blocked int8 GEMM reusing int8-new 6x4 SDOT kernel + internal-K-loop kernel
  (gemm_q4_prefill.c + kernel_6x4_kloop.S): **stable ~56% @M=192**, 5× production.
  This is ~the int8-new microkernel's OWN large-N ceiling (their report: 85%@N=1024,
  67%@N=2048; gemma N=8192-21504 bigger). ABI fix applied to kernel_6x4_opt.S.

## REMAINING WORK (this is the resume target)

### 1. N-blocked GEPP int8 GEMM → 80% prefill  ★ highest value
The internal-K kernel hits 56% because the weight (N×K int8, 44-115 MB) exceeds
cache. Need classic GEBP/GEPP blocking: tile N into L2-resident weight panels
(NC rows × K ≈ a few MB) reused across ALL M-token-tiles; block K; pack panels.
Reuse int8-new's pack_A_6x256 / pack_B_64x256 + kernel_6x4_kloop_256. Loop order
`for jc(M-block fits L1) { for ic(N-panel fits L2) { GEBP } }`. Reference designs:
`a64fx/int8-new/gemm_driver.c` (K=256 only), `kernel_ffn_6row_gemm_d512_ktile.S`,
and `~/work/clair/clair/a64fx/llm-guided-opt/` (sgemm 6x4/2x12/5x4 + kblock
drivers at 80%+). Validate vs naive int8 ref; target 80% of 512 GIOPS/core.
Then wire into transformer_prefill_gemm.

### 2. Attention prefill kernels (QK^T, softmax, scores·V)
Prefill attention = GEMMs (compute-bound). fp16 path (2× fp32 peak; A64FX fp16
FMA). Reuse `a64fx/int8-new/{fused_attention*,flash_attention*,online_softmax}`
and clair `hgemm_*` / `sgemm_fused_qktv.c`. Decode-attention is KV-memory-bound
(different shape). Per-layer extract (one head/layer), measure perf + error.

### 3. int16/fp16 weight-matvec path
We refuted int16 long ago for Q4_0 (3-5× only). But fp16 weights (if a tensor is
F16) and the fp16 FMA path (2× fp32) are worth a per-layer perf+error pass for
any F16/BF16 tensors. A64FX: fp16 FMA yes, bf16 dot NO. Reuse clair
`hgemm_kernel_*`, `fp16-dot-fp32-accum-gemm.md`.

### 4. fp32 path
Baseline/reference. The production fp32-fused Q4_0 matvec (tf_vec_dot_q4_0_f32)
is ~178 GFLOPS/CMG, compute-bound on dequant. A pure fp32 GEMM (sgemm) reference
exists in clair (`sgemm_kernel_6x4.S` etc, 80%+). Mostly a reference ceiling;
low priority unless an fp32 layer matters.

### 5. Other decode ops (small, not dequant-matvec)
RMSNorm, RoPE, SwiGLU, residuals — small but if profiled to matter, optimize.
clair has norm_kernels.S, rope_multimodal.S, exp2-sve. Low priority.

## Recommended order
1 (N-blocked GEPP, the real 80% prefill win) → 2 (attention prefill) → 3/4
(fp16/fp32 reference) → 5.

## Key files
- tools: `a64fx/bench_q4_0_matvec/{layer_opt,lmhead_opt,gemm_q4_prefill,gemm_existing,
  numa_matvec,bw_probe,mini_decode}.c`, `kernel_6x4_kloop.S`
- reuse: `a64fx/int8-new/` (gemm_pack, kernel_6x4_opt[.o], drivers, attention),
  `~/work/clair/clair/a64fx/llm-guided-opt/` (sgemm/hgemm/sdot ASM @80%+)
- decode runner: `a64fx/llm/llm_runner.c` (won't build — VLM broken), forward in
  `common/transformer.h` (tf_forward_blocks_range, tf_gemm_q4_0_tm_worker,
  transformer_prefill_gemm).

## Recent commits (branch gemma4, unpushed)
e09370d3 internal-K-loop kernel + ABI fix; 596d0dcc blocked GEMM; 401795ab bench
production GEMM; 8a26868a naive GEMM; 68e68e1c lm-head Q6_K; 6568e500 Q8v2/per-row;
a1869865 single-tensor opt+accuracy; b568c921 BW diagnostics; ad4838c2 NUMA harness.
