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
