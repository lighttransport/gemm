# a64fx/vlm — A64FX Qwen3-VL vision encoder

SVE-native port of the Qwen3-VL vision encoder for Fugaku-class A64FX.
Bit-exact reference is `common/vision_encoder.h`.

## Build

Native A64FX node (use native, not cross, compilers):

```
cd a64fx/vlm
make CC=fcc OPENMP=1 clean && make CC=fcc OPENMP=1
```

Outputs:
- `build/vlm_runner`  — vision-only CLI runner
- `build/tensor_diff` — VLMD dump validator
- `build/test_cmg_pool` — CMG/NUMA infrastructure check

Other compilers: `make CC=gcc` (C11-thread backend), `make CC=clang OPENMP=1`.

## Run

```
build/vlm_runner <model.gguf> <mmproj.gguf> [image] [options]
```

`model.gguf` is required positional (CLI parity with `cpu/vlm/test_vision.c`)
but unused by the vision-only runner.

### CLI options

| Flag | Default | Notes |
|---|---|---|
| `--dtype fp32\|bf16\|fp16` | `fp32` | weight storage (cache build). fp16/bf16 share the 8×48 SVE microkernel. |
| `--threads N` | `0` (auto) | thread-pool size. Use `0` to inherit `VLM_NUM_THREADS` / `OMP_NUM_THREADS`. |
| `--bench N` | `1` | re-encode N times, report median tok/s |
| `--image-size S` | `384` | longer-side target before snapping to the patch×merge grid |
| `--dump DIR` | off | per-stage VLMD tensor dumps for `tensor_diff` |
| `--prompt "..."` | — | accepted for CLI parity; ignored (vision-only) |
| `--no-prof` | — | silence profiler |

### Recommended invocations

Single CMG (12 cores), bit-exact baseline:

```
OMP_NUM_THREADS=12 \
  ./build/vlm_runner $M $MM ~/fujisan.jpg --dtype fp16 --bench 3
```

Full node (4 CMGs, 48 cores) — the **production config**:

```
XOS_MMM_L_PAGING_POLICY=demand:demand:demand \
numactl -C 12-59 -m 4-7 \
  ./build/vlm_runner $M $MM ~/fujisan.jpg \
  --dtype fp16 --threads 48 --bench 3
```

`OMP_WAIT_POLICY=active`, `OMP_PROC_BIND=close`, `OMP_PLACES=cores` are
set automatically from `main()` (overwrite=0, so an explicit env var still
wins). The active wait policy is what fixed the 24→48T scaling wall —
see "Why these knobs" below.

### Environment variables

| Var | Where | What it does |
|---|---|---|
| `OMP_NUM_THREADS` | OMP runtime | thread-pool size when `--threads 0` |
| `VLM_NUM_THREADS` | `vlm_parallel` | thread-pool size override (takes precedence over OMP) |
| `OMP_WAIT_POLICY=active` | OMP runtime | **set automatically by vlm_runner.** Spin-wait between OMP regions; required to scale past one CMG. |
| `OMP_PROC_BIND=close` | OMP runtime | set automatically. Bind OMP threads to physical cores. |
| `OMP_PLACES=cores` | OMP runtime | set automatically. |
| `VLM_NUMA=N` | `vlm_runner.c` | replicate the weight cache across `N` CMGs (`cmg_alloc`+`mbind`). Use **without** `OMP_PROC_BIND` — manual `cmg_pin_thread` fights OMP binding. |
| `VLM_STAGE_TIMING=1` | `vit_a64fx.c` | per-stage breakdown (attn / ffn_up / ffn_down / qkv / ...) |
| `XOS_MMM_L_PAGING_POLICY=demand:demand:demand` | Fugaku XOS | first-touch local allocation for stack/heap/bss; ~10-20% on top of the active-wait fix. |
| `numactl -C 12-59 -m 4-7` | shell | pin to cores 12-59 (CMG0-3) and interleave memory across NUMA nodes 4-7. |

## Topology

A64FX = 4 CMGs × 12 cores. On this Fugaku node the usable cores are
12-59 (cores 0-11 are reserved by the OS):

| CMG | Cores | NUMA node |
|---|---|---|
| 0 | 12-23 | 4 |
| 1 | 24-35 | 5 |
| 2 | 36-47 | 6 |
| 3 | 48-59 | 7 |

`numactl -C 12-59 -m 4-7` pins to all CMGs and interleaves HBM2 across
the four memory controllers.

## Performance (fujisan.jpg, 384×256, 96 merged tokens, fp16)

| Threads | Config | tok/s |
|---|---|---|
| 12 | 1 CMG | ~79 |
| 24 | 2 CMG, naive | ~157 |
| 36 | 3 CMG | ~176 |
| 48 | 4 CMG, **active wait + numactl** | **~220** (best ~226) |

Output is bit-identical across thread counts (norm=455.7341).

## Why these knobs

The vit encoder issues ~250 fork/join `#pragma omp parallel` regions per
encode. With the default *passive* wait policy, idle workers sleep
between regions and pay a cross-CMG futex wakeup each time — that was
the "24→48T wall" (48T barely beat 24T). The runner sets
`OMP_WAIT_POLICY=active` from the top of `main()` *before* the OMP
runtime initialises (lazy init at first parallel region); the Fujitsu
runtime reads it. Spin-wait threads stay hot, the wakeup cost vanishes,
and 4-CMG scaling holds.

**Do not copy this setenv block into a process that also uses a separate
pthread pool** (e.g. `a64fx/llm/llm_runner.c` mixes the OMP-parallel
vit encoder with `transformer.h`'s pthread pool). 48 spinning OMP
workers will starve the pthread pool — measured: LLM gen 55 → 0.66
tok/s. Pure-vit runners (this one) are safe.

## Validation

```
make REF=1            # build cpu/vlm/test_vision with VLMD dumps
./build/vlm_runner $M $MM ~/fujisan.jpg --dtype fp16 --dump dumps_a64fx
./build/tensor_diff dumps_cpu dumps_a64fx
```

Tolerance ~1e-4 fp32; bit-exact not required.
