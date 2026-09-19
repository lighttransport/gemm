# Resume Qwen3.8 k-quant decode work

## Current state

The exploratory Qwen3.8 Q5_K/IQ4_XS benchmark is committed in `b383fc0b`
(`Add Qwen Q5 decode layout benchmark`). The current branch has subsequently
advanced; do not reset it. The relevant files are:

- `a64fx/llm/bench_qwen38_kquants.c`
- `a64fx/llm/Makefile` (`qwen38_kquant_bench` target)
- `qwen-q8.md`, section `Q5_K row-interleaved decode probe (2026-09-20)`

The benchmark uses the real
`/home/u14346/models/qwen38/27b/Qwen3.8-27B-UD-Q4_K_XL.gguf`, lazy-maps the
GGUF, and copies only the tensor under test into HBM. It is safe to run on one
32 GB A64FX node. Compiler scratch must use `/local`, never `/tmp`:

```sh
mkdir -p /local/u14346/codex-research
TMPDIR=/local/u14346/codex-research \
  make -B -C a64fx/llm qwen38_kquant_bench CC=fcc OPENMP=1
OMP_NUM_THREADS=48 OMP_PROC_BIND=close OMP_PLACES=cores \
  numactl --interleave=all ./a64fx/llm/build/bench_qwen38_kquants \
  /home/u14346/models/qwen38/27b/Qwen3.8-27B-UD-Q4_K_XL.gguf 7
```

The useful candidate is `packed_q5r`. It interleaves eight rows by 256-column
block, expands the 5-bit values to bytes once, and stores decoded FP32 `d` and
`dmin` plus eight scale and minimum bytes per row/block. One packed block is:

```text
8 x 24-byte metadata headers + 8 x 256 expanded values = 2240 bytes
```

This is 280 bytes per original row/block versus 176 bytes for Q5_K, or a
1.5909x expansion. The hot kernel retains the original affine Q5 values and
uses the existing A8 activation quantizer.

Validated layer-0 results at 2.0 GHz, 48 threads:

| Projection | Native Q5_K A8 | packed_q5r | Speedup | packed_q5r effective BW |
|---|---:|---:|---:|---:|
| up, 17408 x 5120 | 1.100 ms | 0.244 ms | 4.5x | 251.5 GB/s |
| down, 5120 x 17408 | 0.881 ms | 0.265 ms | 3.3x | 230.9 GB/s |

Q5R retains the native A8 NRMSE against the compact-weight/FP32-activation
reference: 0.00378 for up and 0.00367 for down. It is not bit-identical to the
native A8 kernel: normalized RMS difference is about `1e-6`, with maximum
absolute differences of `2.62e-6` and `4.29e-6`. End-to-end greedy-token
agreement therefore remains a required gate.

The Q8 repacks are rejected candidates, but remain in the benchmark for
comparison. Whole-row Q8 and global-activation i32 are no faster than Q5R and
raise NRMSE to 0.0133--0.0159. Per-64 Q8 is also no faster and has about 0.009
NRMSE.

The metadata-only footprint command is:

```sh
./a64fx/llm/build/bench_qwen38_kquants \
  /home/u14346/models/qwen38/27b/Qwen3.8-27B-UD-Q4_K_XL.gguf --summary
```

Its validated output is:

```text
model tensors=866 tensor_bytes=17.912GB Q5_K=325/12.936GB Q5R_eligible=325/20.581GB Q5R_delta=7.644GB projected=25.557GB
```

All 325 Q5_K tensors are structurally eligible. A complete replacement image
fits nominally in 32 GB HBM, but an additive cache does not: retaining 12.936 GB
of original Q5_K while allocating 20.581 GB of Q5R already exceeds the node,
before the remaining model tensors and runtime state.

## Completed after the initial handoff

- Added exact `packed_iq4r`: 2x compact storage, about 2.2x faster than native
  IQ4_XS A8, and bit-identical across wave, sparse, high-dynamic-range, and
  deterministic pseudo-random activations. All 65 real IQ4_XS tensors are
  gate-shaped 17408 x 5120 matrices; the model has no transposed IQ4_XS case.
- Revalidated Q5R across the same four patterns. Additional normalized
  differences from native A8 are `2.83e-8`--`1.80e-7`; the dynamic pattern's
  maximum absolute difference is `1.53e-5`.
- Extracted both version-1 layouts, validated size helpers, packers, and SVE
  kernels into `a64fx/llm/kquant_decode_cache.h`. The real-tensor benchmark now
  consumes the shared implementation. `qwen38_kquant_test` supplies a focused
  model-independent four-pattern test.
- Extended `--summary`: 65 IQ4_XS tensors occupy 3.078 GB and expand to 6.155
  GB. A combined replacement Q5R+IQ4R image projects to 28.634 GB, leaving only
  about 3.4 GB before runtime state.

## Remaining work, in priority order

1. **Choose a replacement-load design before integrating the full model.** Do
   not build a full additive Q5R cache on top of an anonymous 17.9 GB GGUF.
   Prefer a baked/staged decode image or sidecar with a manifest, offsets,
   source identity/hash, layout version, tensor shapes, and types. The loader
   must be able to release or avoid faulting the original Q5_K pages. Use
   bounded chunked I/O plus `posix_fadvise(POSIX_FADV_DONTNEED)`; never copy or
   `cat` the complete model interactively.

2. **Preserve NUMA/CMG ownership.** The benchmark first-touches row groups in
   the same static partition used for execution. Production packing/loading
   must retain worker-local placement. Reconcile the eight-row group schedule
   with existing persistent-pool row ranges, and implement a safe compact
   fallback for any non-multiple-of-eight tail rather than silently dropping
   rows.

3. **Separate decode and prefill requirements.** Q5R is a decode-oriented
   layout. Determine whether the selected Qwen runner needs the compact tensor
   for batched prefill. If both representations are needed, account for their
   peak and steady-state memory explicitly; do not assume 25.557 GB tensor
   storage leaves enough space for KV/state/scratch buffers.

4. **Integrate behind an explicit runner argument or diagnostic gate.** Follow
   the repository convention that production tuning choices are arguments,
   not new environment-variable-selected production paths. Retain the compact
   kernel as a correctness fallback for unsupported tensors and failed cache
   validation.

5. **Run end-to-end acceptance.** At minimum compare compact and repacked paths
   on the same prompt for 128 and 256 greedy tokens, record token hashes, and
   require exact token agreement. Then measure total tok/s and stage timings,
   not only isolated matvecs. Track `MemAvailable` and abort before unsafe HBM
   pressure. A full-load performance run should be detached or batch-run when
   it approaches the node memory limit.

6. **Update `qwen-q8.md` and commit a focused unit.** Include exact build/run
   commands, model, compiler, thread placement, memory footprint, correctness
   evidence, and before/after timings. Do not include unrelated DSpark, GLM5,
   DS4F, generated binary, or log changes. Do not push without explicit
   current-turn permission.

## Resume prompt

```text
Continue the Qwen3.8 k-quant/dequant decode work described in
resume-dequant.md. Start by reading the whole file, AGENTS.md, the committed
benchmark in a64fx/llm/bench_qwen38_kquants.c, and the Q5_K work-log section in
qwen-q8.md. Preserve all unrelated dirty-worktree changes and do not reset the
branch; b383fc0b is the completed Q5R benchmark commit, not necessarily HEAD.

The exact Q5R/IQ4R layouts and multi-pattern tests are complete in
a64fx/llm/kquant_decode_cache.h and test_qwen38_kquant_cache.c. Revalidate the
focused test and real-tensor benchmark first, then design and implement a
replacement staged/baked load path with a versioned manifest, source identity,
bounded chunked I/O, and explicit memory accounting. Use
/local/u14346/codex-research for compiler scratch and never /tmp. Keep model
access lazy/bounded and do not make a full model copy.

Do not allocate a full additive Q5R/IQ4R cache while the anonymous compact
model is resident. Preserve worker-local NUMA/CMG placement, provide compact
fallbacks, and resolve decode-versus-prefill representation requirements before
a full-load run. Require 128/256-token greedy hash agreement before promotion.
Build and test on A64FX, update qwen-q8.md with exact evidence, commit only the
focused files, report commit hashes, and do not push.
```
