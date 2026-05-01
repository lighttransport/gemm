# hipBLASLt RDNA4 GEMM ASM Notes

Target: Qwen3.6 vision projector `mm0` with fused `GELU_BIAS`, shape
`M=1024 N=4608 K=4608` in row-major encoder terms. hipBLASLt sees the
same problem as column-major `m=4608 n=1024 k=4608`, `transA=T`,
`transB=N`, BF16/F16 inputs, FP32 accumulate/output.

## Best Measured hipBLASLt Kernels

Commands:

```sh
env HIPBLASLT_LOG_LEVEL=5 HIPBLASLT_LOG_MASK=255 HIPBLASLT_BENCH_PRINT_COMMAND=1 \
  HIPBLASLT_LOG_FILE=/tmp/hipblaslt-bf16-mm0-best.log \
  rdna4/vlm/bench_vlm_hipblaslt --dtype bf16 --shape mm0 \
  --epilogue gelu_bias --iters 5 --algos 256 --workspace-mb 256

env HIPBLASLT_LOG_LEVEL=5 HIPBLASLT_LOG_MASK=255 HIPBLASLT_BENCH_PRINT_COMMAND=1 \
  HIPBLASLT_LOG_FILE=/tmp/hipblaslt-f16-mm0-best.log \
  rdna4/vlm/bench_vlm_hipblaslt --dtype f16 --shape mm0 \
  --epilogue gelu_bias --iters 5 --algos 256 --workspace-mb 256
```

Results on RX 9070 XT:

- BF16: algo `73823`, `0.2675 ms`, `162.584 TFLOP/s`, `83.4%` of 195 TFLOP/s.
- F16: algo `88188`, `0.2597 ms`, `167.470 TFLOP/s`, `85.9%` of 195 TFLOP/s.

Both are zero-workspace kernels.

Latest BF16 control without epilogue, using a 64-algo sweep:

```sh
rdna4/vlm/bench_vlm_hipblaslt --dtype bf16 --shape mm0 \
  --epilogue none --iters 20 --algos 64 --workspace-mb 512
```

- BF16: algo `73778`, `0.2340 ms`, `185.825 TFLOP/s`, `95.3%` of the 195
  TFLOP/s benchmark peak.

This run is a better pure-mainloop target than the older GELU-bias number. It
also makes the conclusion stronger: the handwritten path needs hipBLASLt's
operand/dataflow schedule, not just local wait-count edits.

## Decoding the Packaged Code Object

ROCm 7.2.1 hipBLASLt `.co` files for gfx1201 are not directly ELF. They are
`CCOB` containers whose payload begins at byte 32 as a zstd-compressed clang
offload bundle. `dump_hipblaslt_kernel_asm.sh` handles this:

```sh
rdna4/vlm/dump_hipblaslt_kernel_asm.sh \
  /opt/rocm-7.2.1/lib/hipblaslt/library/TensileLibrary_BB_SB_HA_Bias_SAV_UA_Type_BS_HPA_Contraction_l_Alik_Bljk_Cijk_Dijk_gfx1201.co \
  /tmp/hipblaslt_bf16_alik_bljk
```

The full BF16 library dump summary was:

- `v_wmma`: 85335
- `v_mfma`: 0
- `global/buffer load`: 168024
- `global/buffer store`: 264012
- `ds_read`: 0 by string counter, but ISA uses `ds_load_b128`
- `ds_write`: 0 by string counter, but ISA uses `ds_store_b128`
- `barrier`: 24830
- `wait`: 95816

Use `rg 'ds_load|ds_store'` on the `.s` file for LDS counts.

## Schedule Facts From hipBLASLt Names and ASM

Best BF16/F16 kernels use:

- Macro tile: `MT128x128x32`
- Matrix instruction: `MI16x16x1`
- Workgroup: `WG32_4_1`, four wave32 waves per CTA
- Workgroup shape: 128x128 output tile, K block 32
- Global read vectorization: typically `GRVWA4/8`, `GRVWB8`
- Local read vector width: `LRVW8`
- Global stores: `SVW4` for best BF16 and F16 candidates
- Prefetch: `PGR1` or `PGR2`, with tuned `TLDS1/2`
- LDS: about 26 KiB group segment for the observed BF16 package

Important ASM pattern:

- Uses `buffer_load_b128` into high VGPRs.
- Stages global data through LDS with `ds_store_b128`.
- Reads matrix fragments with `ds_load_b128`.
- Interleaves `buffer_load_b128`, `ds_load_b128`, and `v_wmma_f32_16x16x16_bf16`.
- Uses `s_wait_loadcnt`, `s_wait_dscnt`, `s_barrier_signal`, and
  `s_barrier_wait` to keep global load, LDS traffic, and WMMA overlapped.

The hot-loop dump around `/tmp/hipblaslt_bf16_best_73823.s:1116` shows a more
specific pattern than our generated kernel:

- The loop starts global `buffer_load_b128` for the next operand slice before
  the first current-slice WMMA.
- It interleaves LDS reads for one operand with those global loads.
- Around the middle of the 32-WMMA group it starts `ds_store_b128` for the next
  LDS buffer, then signals/waits the barrier before the last WMMA group is
  finished.
- After the barrier it immediately starts LDS reads for the next loop while
  finishing current accumulator work.

The generated handwritten kernel does not have this dataflow. It stages both
operands through LDS, reads both operands from LDS before WMMA, and stores the
next buffer after the WMMA group. Assembly-only reshuffles that keep this
dataflow did not improve performance.

The current handwritten C/HIPRTC kernels already emit real RDNA4 WMMA and do
not spill in the simple cases, but they serialize too much of the
global-to-LDS-to-WMMA pipeline. The `mm0vec` prototype added to
`bench_vlm_gemm.c` forces 128-bit global and LDS staging for the fixed `mm0`
shape. It compiles to `global_load_b128`, `ds_store_b128`, `ds_load_b128`, and
`v_wmma_f32_16x16x16_*`, but measured:

- BF16 `mm0vec`: `3.8621 ms`, `11.260 TFLOP/s`
- F16 `mm0vec`: `3.8660 ms`, `11.248 TFLOP/s`

The next scheduling pass added fixed-shape BF16 pipelined kernels:

- `mm0pipe`: four-wave CTA, 128x128 tile, 16 accumulators per wave,
  double-buffered LDS, prefetches next K tile before current WMMA.
- `mm0pipe8`: eight-wave CTA, 128x128 tile, 8 accumulators per wave,
  double-buffered LDS, prefetches next K tile before current WMMA.
- `mm0pipe4n64`: four-wave CTA, 128x64 tile, 8 accumulators per wave,
  lower LDS footprint but reloads A for twice as many N tiles.

The first version used `float8 *accs[]` tables in the epilogue. HIPRTC lowered
those into private/scratch traffic, hiding the mainloop gain. Replacing those
tables with explicit accumulator stores changed BF16 `mm0` measurements to:

- `mm0vec`: `3.8724 ms`, `11.230 TFLOP/s`, `5.8%` of peak (`iters=50`)
- `mm0pipe`: `0.7837 ms`, `55.489 TFLOP/s`, `28.5%` of peak (`iters=50`)
- `mm0pipe8`: `0.7797 ms`, `55.777 TFLOP/s`, `28.6%` of peak (`iters=50`)
- `mm0pipe4n64`: `0.8448 ms`, `51.474 TFLOP/s`, `26.4%` of peak (`iters=50`)

Current generated-code metadata/counters:

- `mm0pipe8`: 32 KiB LDS, 141 VGPR, no private segment, no spills, no scratch,
  16 static WMMA instructions in the loop body.
- `mm0pipe`: 32 KiB LDS, 238 VGPR, no private segment, no spills, no scratch,
  32 static WMMA instructions in the loop body.
- `mm0pipe4n64`: 24 KiB LDS, 142 VGPR, no private segment, no spills, no
  scratch, but extra global/LDS traffic from A reloads.

The next pass made three changes to `mm0pipe`:

- Added `__launch_bounds__(128, 1)`, reducing the compiler's assumed max
  workgroup size from 1024 to 128. This removed the four-wave kernel's VGPR
  spills and let it use 234+ VGPRs.
- Removed the redundant pre-store CTA barrier. The next tile is written into
  the inactive LDS buffer, so only the post-store barrier is needed before the
  next K tile reads it.
- Replaced row-major LDS tile layout `row*4+kslot` with transposed
  `kslot*128+row`. This changes wave fragment loads from a 64-byte lane stride
  to a contiguous row slice and was the main performance gain.
- Replaced `__syncthreads()` in the pipe kernels with inline gfx12 split
  barriers (`s_barrier_signal -1`, `s_barrier_wait 0xffff`) to avoid the
  broader global invalidation generated by the HIP builtin.

Updated BF16 `mm0` measurements:

- `mm0pipe`, after launch bounds only: `0.6688 ms`, `65.026 TFLOP/s`
  (`iters=50`)
- `mm0pipe`, after redundant-barrier removal: `0.6479 ms`,
  `67.124 TFLOP/s` (`iters=50`)
- `mm0pipe`, transposed LDS + split barrier: `0.3235 ms`,
  `134.436 TFLOP/s`, `68.9%` of peak (`iters=200`)
- Sampled correctness check: `max_abs=0`, `rms=0`, `cosine=1.000000000`,
  `PASS`

Follow-up diagnostics:

- `mm0pipe`, current restored fastest path: `0.3220 ms`,
  `135.070 TFLOP/s`, `69.3%` of peak (`iters=200`, bias on), sampled
  correctness `PASS`.
- `mm0pipe --no-bias`: `0.3263 ms`, `133.256 TFLOP/s`, `68.3%` of peak
  (`iters=200`). Bias loads/output epilogue are not the dominant bottleneck.
- `mm0pipe8`, after transposed LDS + split barrier: `0.3601 ms`,
  `120.760 TFLOP/s`, `61.9%` of peak (`iters=100`), sampled correctness
  `PASS` at `0.3766 ms`, `115.474 TFLOP/s` (`iters=50`). This improved the
  old pipe8 path, but it still rereads A across four N-side waves and loses to
  `mm0pipe`.
- `mm0pipe4n64`, after transposed LDS + split barrier: `0.4519 ms`,
  `96.232 TFLOP/s`, `49.3%` of peak (`iters=50`), sampled correctness
  `PASS`. Lower N tile size increases A reload traffic and is not competitive.
- A source-level attempt to expose B-load plus per-A-row WMMA grouping was
  slower (`0.3241 ms`, `134.186 TFLOP/s`) and was reverted.
- `mm0pipek64`, 128x128 tile with K tile 64 and 64 KiB double-buffered LDS:
  `0.3539 ms`, `122.890 TFLOP/s`, `63.0%` of peak (`iters=100`), sampled
  correctness `PASS`. Halving the K-loop barrier count is not enough; the
  larger LDS footprint and generated wait schedule lose to K32.
- `mm0pipemid`, K32 tile with next-buffer LDS stores and `s_barrier_signal`
  moved between the two 16-WMMA halves: `0.3540 ms`, `122.831 TFLOP/s`,
  `63.0%` of peak (`iters=100`), sampled correctness `PASS`. Source-level
  mid-store scheduling does not hide handoff; LLVM still emits enough waits to
  lose to the original K32 schedule.
- `mm0pipegl`, source-level attempt to issue next global loads inside the WMMA
  stream: `0.8322 ms`, `52.253 TFLOP/s`, `26.8%` of peak (`iters=200`),
  sampled correctness `PASS`. The generated code has the same static WMMA/LDS
  instruction counts as `mm0pipe` but more waits and a much worse branch/dependency
  schedule. A volatile-load version was correct but collapsed to `3.3936 ms`.
- Split-tail diagnostic, where the final K32 stage was moved out of the hot
  loop to remove `has_next` from 143 stages: `0.3588 ms`, `121.214 TFLOP/s`,
  sampled correctness `PASS`. It was restored because preserving LLVM's original
  loop shape matters more than removing that branch.
- `mm0asm`, generated external code-object path using `gen_mm0_bf16_asm.py`:
  `0.3215 ms`, `135.262 TFLOP/s`, `69.4%` of peak (`iters=200`), sampled
  correctness `PASS`; best observed `0.3212 ms`, `135.376 TFLOP/s`. This
  establishes the generated-code-object ABI (`gemm_mm0_bf16_asm`) and is now
  the path for replacing the generated HIP mainloop with direct GCN assembly.
- Generated `.s` round-trip, assembled and linked with ROCm clang:
  `0.3268 ms`, `133.082 TFLOP/s`, `68.2%` of peak (`iters=100`), sampled
  correctness `PASS`. This is the first directly editable AMDGPU assembly
  code-object path; it is slower than hipcc `--genco`, but it gives us a valid
  `.amdhsa_kernel` template to mutate.
- Patched `.s` variants:
  `halfsplit` `0.3276 ms`, `132.757 TFLOP/s`;
  `storeearly20` `0.3303 ms`, `131.643 TFLOP/s`;
  `bfirst2` `0.3300 ms`, `131.792 TFLOP/s`. All passed sampled correctness.
  These rule out small local LDS wait/order edits as the missing 80%+ path.
- Direct-A/LDS-B diagnostic generated by `gen_mm0_bf16_directa.py`:
  `0.6724 ms`, `64.677 TFLOP/s`, sampled correctness `PASS`; metadata
  `16 KiB LDS`, `204 VGPR`, no spills. Although it halves static LDS
  load/store count, naïve direct wave-fragment global loads are too slow. The
  hipBLASLt direct/overlapped operand schedule depends on its global-read
  mapping and cannot be copied by simply skipping LDS for one operand.

A structured post-mortem log with resource tables is kept in
`rdna4/vlm/rdna4_gemm_optimization_log.md`.

So the bottleneck is not only vector width or the epilogue. hipBLASLt keeps a
much tighter software pipeline active and overlaps global loads, LDS traffic,
barriers, and WMMA work across K slices with lower schedule overhead. The next
handwritten optimization should move out of compiler-scheduled C/HIPRTC for the
mainloop and copy hipBLASLt's pipelined load/LDS/WMMA schedule in GCN:

- Use `buffer_load_b128` equivalent vectorized loads for A/B.
- Stage A/B through LDS with 128-bit stores and swizzled offsets.
- Use double-buffered fragment loads from LDS.
- Start the next global loads before all WMMA for the current K slice finish.
- Fuse bias + GELU after accumulation, but do not optimize epilogue before the
  mainloop reaches high occupancy.
