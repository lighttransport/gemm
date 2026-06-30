# Repository Guidelines

## Project Structure & Module Organization
This repository is organized by hardware backend. Use top-level folders as module boundaries:
- `a64fx/`: A64FX SVE kernels, microbenchmarks, and optimization notes.
- `ryzen/` and `zen2/`: x86_64 AVX2/FMA GEMM, attention, and NN ops.
- `cuda/int8` and `cuda/fp8`: CUDA/cuBLAS experiments and reference tools.
- `vulkan/`: Vulkan compute runners, shaders, and multimodal tests.
- `common/`: shared single-file components (`gguf_loader`, tokenizer, transformer) and `test_*.c` programs.

Keep generated binaries and temporary logs local; do not mix unrelated architecture work in one change.

## Build, Test, and Development Commands
Run commands from repo root unless noted:
- `make -C ryzen` builds AVX2 benchmarks; `make -C ryzen gemm|flash|nn` builds and runs specific suites.
- `make -C zen2` builds the Zen2 benchmark driver.
- `make -C cuda/int8 test` runs INT8 attention/FFN correctness checks.
- `cmake -S vulkan -B vulkan/build && cmake --build vulkan/build -j` builds Vulkan tools and compiles shaders.
- `./vulkan/build/test_vision_encoder` (or `test_vision_multimodal`) runs Vulkan-side validation.
- `make -C a64fx/int8-new 5x4 COMPILER=fcc` builds an A64FX target; use `make -C <dir> clean` to reset artifacts.

## Coding Style & Naming Conventions
- Languages: C, C++, and architecture-specific `.S` assembly.
- Use 4-space indentation and keep brace/style conventions consistent with nearby files.
- Use `snake_case` for functions/files, `UPPER_CASE` for macros/constants.
- Follow existing naming suffixes for backend/type (`_avx2`, `_sve`, `_f32`) and prefixes (`bench_`, `test_`).
- Keep warning-clean builds (`-Wall -Wextra -Wpedantic`) where Makefiles already enforce them.

## Testing Guidelines
There is no single global test harness. Each module owns executable tests/benchmarks.
- Add or update `test_*.c` when changing math kernels or loaders.
- Validate correctness against existing naive/reference paths before reporting performance.
- In PRs, include exact commands run and representative output for both correctness and performance.

## Commit & Pull Request Guidelines
Git history favors short imperative subjects (for example: `Add ...`, `Optimize ...`, `Fix ...`).
- Keep each commit focused on one subsystem/backend.
- PRs should include: purpose, affected hardware/compiler settings, commands executed, and before/after metrics.
- Link related issues and call out any portability limits (ISA, GPU extension, or SDK requirements).

## Pushing & Authorization (agents)
Automated agents working in this tree must NOT run `git push` to any remote without explicit
per-action user permission in the current request.

- Commit freely once a coherent unit of work is done (or when the user says "commit"); always
  report the commit hash and a short diff summary so the user can review before authorizing a
  push.
- Do NOT chain `git push` after a commit in the same shell invocation. Each push needs its
  own explicit request.
- Prior "push" authorization does NOT carry across subsequent commits in the same session.
  One push verb from the user = one push. The next commit re-arms the question.
- The only standing exceptions are explicit push verbs in the user's current-turn message:
  "push it", "git push", "ship to main", "merge + push", etc.
- When unsure, default to NOT pushing. Ask first.

This rule applies to all remotes (origin and any others) and all branches, including feature
branches. It is the same intent as Claude Code's general guidance ("for actions that are hard
to reverse or outward-facing, confirm first; approval in one context doesn't extend to the
next").

## Memory-safe operation on the A64FX node (Gemma-4 12B, 32 GB HBM)
The node has **32 GB HBM** and the 12B BF16 model is **24 GB**. Careless memory ops
thrash **kswapd** and **HANG the interactive session** (or OOM-kill the agent).

- **NEVER `cp`/`cat` the 24 GB model at once in an interactive session.** A single copy
  accumulates ~24 GB of *dirty* page-cache → writeback storm → kswapd hang. Stage in
  1 GiB chunks with fsync: `sh a64fx/bench_q4_0_matvec/stage_model.sh` (idempotent).
  `/local/u14346/` is **wiped on every session restart** → re-stage each session.
  Source: `~/models/gemma4/12b/gemma-4-12b-it-BF16.gguf`. Avoid `cat` on multi-GB files.
- **Use `NO_MMAP=1` (explicit anon upload to HBM2), NOT mmap.** mmap re-faults weights
  from disk on every decode token (decode reads all 24 GB/token) → constant kswapd
  re-fault thrash AND NUMA-misplaced/slow. Anon loads once, stays resident, fast.
- **The anon load is now memory-SAFE (fadvise fix, this session).** Previously a plain
  load read 24 GB into anon WHILE caching 24 GB of source file pages = ~48 GB transient
  → kswapd thrash → session HANG. Fixed: the loader now reads in chunks and
  `posix_fadvise(POSIX_FADV_DONTNEED)` drops the source page-cache as it goes (both the
  `gguf_loader.h` fread path and `transformer.h` numa-pread path; `TF_LOAD_KEEPCACHE=1`
  to disable). Peak is now ~26 GB (model + buffers), `MemAvailable` stays stable ~5 GB,
  **no thrash**. Validated: `NUMA_INTERLEAVE=1 NO_MMAP=1` → 8.4 tok/s, alpha 0.78 @G128,
  greedy-exact, MemAvailable steady.
- Gauge headroom with **`MemAvailable`** (reclaimable cache keeps it high; a runaway
  *anon* alloc drops it). Guard a run: kill if `MemAvailable < 6 GB` (= >25 GB consumed).
- **Speed** benchmarks (8.45 tok/s spec, 7.7 persistent, etc.) need the fast anon load
  (`NUMA_INTERLEAVE=1 NO_MMAP=1`, ~25 GB) → run **detached / batch job**, not in the
  interactive agent session. Use the mmap config only for **correctness** reruns here.

### Rerun the spec-decode validation (memory-safe)
```
sh a64fx/bench_q4_0_matvec/stage_model.sh           # stage (chunked) if needed
cd a64fx/bench_q4_0_matvec
fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp -D_GNU_SOURCE \
    -DTF_HAVE_Q8V2 -DTF_LINK_PODD -I../../common mini_mtp.c \
    ../gemma4-kernels/kernel_q8v2_3x4.S ../gemma4-kernels/sgemm_bf16_2x12.S -lm -o mini_mtp
M=/local/u14346/gemma-4-12b-it-BF16.gguf
MT=/home/u14346/models/gemma4/12b/MTP/gemma-4-12b-it-BF16-MTP.gguf
NUMA_INTERLEAVE=1 NO_MMAP=1 MTP_SCALE=1.0 OMP_NUM_THREADS=48 \
  OMP_PROC_BIND=close OMP_PLACES=cores LLM_THREADS=48 ./mini_mtp $M $MT 128 4
```
Expect `match=128/128` (greedy-exact) + `alpha=`. **alpha RISES with context** (the
short gibberish-prompt prefix is unpredictable): G=32~0.0, G=48~0.47, **G=128~0.78,
G=256~0.88** — so use G>=128 to see the real rate. Headline: **8.4 tok/s @ alpha 0.78
G=128 / 0.88 G=256**. The anon load now peaks ~26 GB with no thrash (fadvise). Guard a
run by killing if `MemAvailable < 2 GB`. Full work log: `project_gemma4_12b_bf16` memory.
