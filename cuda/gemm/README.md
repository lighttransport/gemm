# cuda/gemm — Tensor-Core GEMM benchmark (Blackwell sm_120)

Sweeps F16 / BF16 / FP8(E4M3) GEMMs on RTX 5060 Ti (Blackwell GeForce, sm_120),
comparing **our PTX `mma.sync` kernels** against **cuBLAS / cuBLASLt**.
Mirrors the layout and reporting of `rdna4/vlm/bench_vlm_gemm.c`.

PTX kernels are strings compiled at runtime via NVRTC, loaded through `cuew`
(libcuda) and `cublasew` (libcublas / libcublasLt). The optional `--mode cutile`
loads `libcutile_gemm.so` (CUTLASS-backed, built separately with nvcc).

## Build / run

```
make                 # builds bench_cuda_gemm only (no nvcc required)
make cutile          # builds libcutile_gemm.so via nvcc + CUTLASS
./bench_cuda_gemm --dtype f16  --mode all --shape all
./bench_cuda_gemm --dtype bf16 --mode all --shape all
./bench_cuda_gemm --dtype fp8  --mode all --shape all
./bench_cuda_gemm --dtype f16  --mode cutile --shape all   # needs make cutile first
make smoke           # f16 single-shape sanity
```

For `--mode cutile`, CUTLASS must be cloned to `../../third_party/cutlass`
(override with `make cutile CUTLASS_DIR=/path/to/cutlass`):
```
git clone --depth 1 https://github.com/NVIDIA/cutlass ../../third_party/cutlass
```

CLI:
```
--dtype  f16 | bf16 | fp8
--mode   ptx | cublas | cutile | all
--shape  <name> | all          # see shape table below
--iters  N (default 100)
--warmup N (default 5)
--verify 0|1 (default 1; checks first 64 rows vs CPU FP32 ref)
--m M --n N --k K              # ad-hoc shape (overrides --shape)
```

## Shapes

| name | M | N | K | role |
|---|---|---|---|---|
| square_1k/2k/4k/8k | n | n | n | generic squares |
| mm0 | 1024 | 4608 | 4608 | vision proj (matches rdna4 mm0) |
| mm2 | 1024 | 1152 | 4608 | vision proj (matches rdna4 mm2) |
| qkv / attn_out | 512 | 4096 | 4096 | Qwen3 attention |
| ffn_up | 512 | 11008 | 4096 | Qwen3 FFN |
| ffn_down | 512 | 4096 | 11008 | Qwen3 FFN |

## Peaks (RTX 5060 Ti, GeForce-throttled)

- F16 / BF16 (FP32 accum): nominal **42 TFLOP/s** — sustained cuBLAS hits ~46–50
- FP8 E4M3 (FP32 accum):  nominal **84 TFLOP/s**

## v1 status (see `bench_log.md` for the full table)

- **cuBLAS F16/BF16** ≥ 100% of nominal peak on every shape (PASS80).
- **PTX F16/BF16** is a starting-point port — 23–45% of peak. Tuning is the
  follow-up phase (LDS/SMEM layout, prefetch, larger CTA tile).
- **PTX FP8** 19–48% of peak; FP8 cuBLASLt is **unsupported on sm_120** in
  cuBLAS 13.x (heuristic returns no algo) — bench skips that mode gracefully.
- **CUTLASS cutile** F16/BF16 matches or slightly beats cuBLAS (e.g. mm0 bf16:
  50.5 vs 49.7 TFLOP/s; ffn_up f16: 48.2 vs 47.1). FP8 cutile path is a stub
  that returns -1; bench skips it gracefully. CUTLASS 4.x's own sm_120 FP8
  blockwise examples (87a/87b) fail at runtime on this RTX 5060 Ti / CUDA
  13.x with a TMA descriptor / device-side assertion error — upstream issue,
  not in our wrapper. See `bench_log.md` for details.

## Files

- `bench_cuda_gemm.c` — host driver (argv, shape sweep, timing, validation)
- `cutile_gemm.cu` — CUTLASS-backed GEMM (F16/BF16, Sm80 OpClassTensorOp via
  `cutlass::gemm::device::Gemm`). Compiles to `libcutile_gemm.so`.
- `cuda_gemm_ptx_kernels.h` — three PTX kernel sources as C strings:
  - `gemm_f16_f32` (m16n8k16 .f16.f16.f32)
  - `gemm_bf16_f32` (m16n8k16 .bf16.bf16.f32)
  - `gemm_fp8_e4m3_f32` (m16n8k32 .e4m3.e4m3.f32)
- Reused: `../cuew.{h,c}`, `../cublasew.{h,c}`, `../cuda_runner_common.h`,
  `../../common/cpu_compute.h` (CPU ref).
