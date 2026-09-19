# Resume A64FX fused decode and FFN work

## Current state

Work is on branch `glm53f`. The relevant commits are:

- `69b62e85` — fused A64FX INT4 dequant kernels and HBM probe
- `7ca79dff` — XOS hugepage mode for the HBM probe
- `b7db304a` — SDOT/FMLA arithmetic-budget measurement
- `5db7e65d` — fused MXFP4 SDOT projection and SwiGLU FFN

Do not push without explicit per-action user permission. Preserve unrelated
worktree changes. Do not use `/tmp`; use `/local/u14346` or a repository-local
`tmp` directory.

## Established measurements

One A64FX CMG uses CPUs 12--23. With Fujitsu XOS 2 MiB hugetlbfs pages, a
240 MiB twelve-core sequential read reaches approximately 229.5 GB/s. Linux
pagemap PFNs are masked for the current user.

The arithmetic roofline probe in `a64fx/dequant-pipe` established:

| SVE arithmetic instructions per 256-byte line | Bandwidth | Paired read |
|---:|---:|---:|
| 48 | 229.51 GB/s | 99.96% |
| 52 | 224.25 GB/s | 97.68% |
| 56 | 216.44 GB/s | 94.32% |
| 60 | 201.71 GB/s | 87.88% |
| 64 | 190.39 GB/s | 83.00% |

Thus the measured no-degradation budget is about 48 independent SVE SDOT or
FMLA instructions per 256 packed bytes, or about 1.79 arithmetic instructions
per cycle at 229.5 GB/s. Always use `--paired-baseline`: the CMG occasionally
enters an approximately 120 GB/s state that affects read-only and compute runs
equally.

The current `a64fx/swfp4fp8` fused MXFP4 path uses:

- SDOT-native `[N/16][K/32][8][32-byte]` packed panels;
- dynamic INT8 activation quantization per K=32 group;
- exact doubled-integer representation of E2M1 weights;
- register-only nibble expansion with `lsr`, `and`, `zip1`, `tbl`, and `sdot`;
- four-way quartet software pipelining and independent SDOT accumulators;
- direct E8M0-to-FP32 exponent construction;
- no expanded-weight stores;
- `swfp4fp8_ffn_mxfp4_sdot()` for gate, up, SwiGLU, and down.

Correctness currently passes against the FP64 reference:

```text
mxfp4 fused sdot PASS rel_l2=0.00294425 scaled_max=0.00295259
swfp4fp8 tests: PASS
```

Measured with twelve cores and XOS 2 MiB pages:

| Benchmark | Median | Effective compressed bandwidth | Raw read |
|---|---:|---:|---:|
| Fused projection, N=32768 K=4096 | 1.702 ms | 41.89 GB/s | 225.17 GB/s |
| Fused SwiGLU FFN, d=4096 h=8192 | 1.483 ms | 36.06 GB/s | 226.05 GB/s |

This is about 70 times faster than the former approximately 0.6 GB/s path
that materialized group-local INT8 weights, but it remains decode/issue-bound
at only 16--19% of the HBM ceiling.

## Important conclusions

1. A 16 KiB inter-core gap is not a decoded HBM channel selector. Controlled
   hugepage runs sustain approximately 228--230 GB/s across 0--64 KiB skews.
2. Physical allocation/page placement caused the earlier fast/slow states.
3. The current fused MXFP4 bottleneck is not HBM traffic or expanded stores.
4. Four-way scheduling, halving TBL count, independent accumulators, and loop
   branch-slot alignment were tested. They did not approach the HBM ceiling.
5. The remaining dominant cost is the E2M1 decode/permutation path, especially
   table/permutation throughput and the scalar/address instructions around it.

## Remaining tasks

### 1. Measure the assembly kernel precisely

- Add a kernel-only timing mode that excludes matrix packing and OpenMP team
  creation as much as possible.
- Collect cycles, retired instructions, SVE arithmetic counts, L1/L2 refills,
  and frontend/backend stall events when the available A64FX PMU tooling
  permits it.
- Measure one core and 1--12 core scaling for the same large projection.
- Confirm XOS backing for the actual weight allocations through `smaps`, not
  only through the environment configuration.
- Compare the kernel against a same-layout read-only assembly loop in the same
  process and allocation.

### 2. Remove or amortize E2M1 `tbl`

Test these representations independently:

- signed mantissa plus exponent planes produced by the model converter;
- two-bit exponent and sign/mantissa bitplanes that decode with shifts/XOR;
- an 8-bit signed doubled-E2M1 representation as an upper-bound control;
- a hybrid representation that expands only the nonlinear magnitude bits at
  load time;
- larger K supertiles that reuse decoded weights across multiple tokens when
  M is greater than one.

The 8-bit control doubles HBM bytes, so its maximum compressed-equivalent rate
is approximately half the physical HBM rate. It is still valuable for proving
whether the remaining SDOT and scale schedule can saturate the interface.

For every alternative, report both physical bytes/s and original MXFP4
compressed-equivalent bytes/s. Do not claim bandwidth preservation by counting
only the original 4-bit size when an expanded representation is read from HBM.

### 3. Reduce loop overhead

- Unroll across multiple K=32 groups and hoist pointer updates.
- Keep several group accumulators live so scale conversion overlaps subsequent
  nibble decoding.
- Try two panels per core so activation loads and quantization scales are
  reused across 32 output rows.
- Check branch addresses and decode groups in disassembly after every assembly
  change.
- Retain enough independent accumulators to cover SDOT latency.

### 4. Improve the FFN layer contract

- Add a small scalar end-to-end SwiGLU reference test, not only projection
  correctness.
- Add bias support only if required by a real model contract.
- Quantize the gate/up input once and share it between both projections.
  The current implementation quantizes it separately in each call.
- Fuse or parallelize gate and up traversal where doing so does not destroy
  sequential HBM streams.
- Reuse the intermediate quantization for the down projection where possible.
- Add realistic model dimensions and report latency as well as bandwidth.

### 5. Memory ownership and production integration

- The lab currently retains canonical, panel, and SDOT-native copies. Add an
  import/ownership mode that keeps only the selected production layout.
- Keep `SWFP4FP8_XOS_ALLOC=1` optional and document allocation/free symmetry.
- Validate allocation failures and partial cleanup under XOS hugepage limits.
- Integrate the selected kernel only after correctness and per-CMG bandwidth
  are stable across multiple allocations.

## Build and validation

```sh
make -C a64fx/swfp4fp8 clean
make -C a64fx/swfp4fp8 test CC=fcc
make -C a64fx/swfp4fp8 bench_swfp4fp8 CC=fcc
```

Run the fused projection on one CMG:

```sh
LD_PRELOAD=/opt/FJSVxos/mmm/lib64/libmpg.so.1 \
XOS_MMM_L_HPAGE_TYPE=hugetlbfs \
XOS_MMM_L_HUGETLB_SZ=2M \
XOS_MMM_L_HUGE_MALLOC=1 \
XOS_MMM_L_FORCE_MMAP_THRESHOLD=1 \
XOS_MMM_L_HUGETLB_FALLBACK=0 \
SWFP4FP8_XOS_ALLOC=1 \
OMP_PROC_BIND=close OMP_PLACES=cores \
  ./a64fx/swfp4fp8/bench_swfp4fp8 --fused --threads 12
```

Replace `--fused` with `--ffn` for the full gate/up/SwiGLU/down benchmark.

Run the arithmetic roofline control:

```sh
cd a64fx/dequant-pipe
LD_PRELOAD=/opt/FJSVxos/mmm/lib64/libmpg.so.1 \
XOS_MMM_L_HPAGE_TYPE=hugetlbfs XOS_MMM_L_HUGETLB_SZ=2M \
XOS_MMM_L_HUGE_MALLOC=1 XOS_MMM_L_FORCE_MMAP_THRESHOLD=1 \
XOS_MMM_L_HUGETLB_FALLBACK=0 \
  ./bench_hbm_color --page-mode xos --op sdot --ops-per-line 48 \
  --paired-baseline --max-skew-kib 0
```

## Resumption prompt

```text
Continue the A64FX fused MXFP4 decode + SDOT FFN work described in
resume-fused-decode.md. Start from commit 5db7e65d and inspect the current
worktree without overwriting unrelated changes. Use CPUs 12--23 and Fujitsu
XOS 2 MiB pages. The current fused projection is correct but reaches only
41.89 GB/s versus a 225 GB/s read ceiling; the full FFN reaches 36.06 GB/s.

First add a paired same-allocation, kernel-only read baseline and PMU/cycle
instrumentation. Then implement and benchmark the most promising no-TBL E2M1
representation, including an honest 8-bit predecoded upper-bound control.
Report physical and 4-bit-equivalent bandwidth separately. Preserve the
existing relative-L2 correctness check and add an end-to-end scalar SwiGLU
FFN reference. Keep iterating while a safe, concrete optimization remains.
Do not use /tmp and do not push without explicit permission.
```
