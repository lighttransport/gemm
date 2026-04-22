# Gemma4 Review Snapshot

Last reviewed: 2026-04-09

## Scope

This note reflects the current local Gemma4 code paths in:

- `common/transformer.h`
- `common/gemma4_vision_encoder.h`
- `cpu/vlm/test_gemma4_vision.c`
- `cuda/vlm/test_cuda_gemma4_vlm.c`

This note now includes local end-to-end quality checks against the 2B Gemma4 VLM model and a comparison with `llama.cpp`.

## Current State

### Implemented

- Gemma4 architecture support exists in both CPU and CUDA LLM paths.
- Gemma4-specific pieces are present in the shared transformer layer:
  - per-layer embeddings
  - SWA/full-attention handling
  - proportional RoPE handling
  - Gemma4 logit soft-capping
  - vision-token handling through `transformer_forward_embd[_logits]`
- The CPU and CUDA Gemma4 VLM harnesses now use a prompt shape much closer to the Gemma4 chat template path used by `llama.cpp`:
  - `BOS`
  - `<|turn>system\n<|think|><turn|>\n`
  - `<|turn>user\n<|image>`
  - image embeddings
  - `<image|>{prompt}<turn|>\n<|turn>model\n`
- Both CPU and CUDA VLM harnesses include a reasoning-budget state machine that watches for `<|channel>thought\n ... <channel|>` and can force an exit when the model enters a hidden thought channel.
- CPU VLM explicitly disables hidden-norm tracing with `transformer_set_trace_hidden_norms(model, 0)` so generation logs stay readable.
- Both harnesses now resolve control-token IDs by exact vocab lookup instead of tokenization heuristics.
- Both harnesses now build prompts dynamically, so longer prompts no longer silently truncate the chat framing.
- Both harnesses suppress visible printing of protocol tokens like `<turn|>` / `eos` / `eot`.
- Both harnesses now sample with a closer `llama.cpp`-style default:
  - `temperature = 0.8`
  - `top_k = 40`
  - `top_p = 0.95`
  - `min_p = 0.05`
- Both harnesses now default to `--budget 32` instead of `200`, because the more faithful Gemma4 prompt often spends many tokens in the hidden thought channel before producing visible text.

### Verified Locally

- `cpu/vlm/test_gemma4_vision` builds successfully from the current tree.
- `cuda/vlm/test_cuda_gemma4_vlm` builds successfully from the current tree.
- The CPU build still emits several unrelated warnings from shared headers, but no build failure in the reviewed path.
- The CUDA build still emits several unrelated warnings from shared headers and `cuda_llm_runner.c`, but no build failure in the reviewed path.
- End-to-end CPU and CUDA runs on the test image both produce visible descriptions that identify the mountain as Mount Fuji.

## Key Findings

### 1. The main quality gap was prompt protocol, not CUDA numerics

- `llama.cpp` was not using the same prompt shape as the local harnesses.
- The Gemma4 Jinja path in `llama.cpp` uses a system turn that begins with `<|think|>`, then a user turn with the image marker.
- After aligning our prompt framing to that shape, the first generated token matched `llama.cpp`: `<|channel>`.
- This strongly suggests the earlier generic captions were caused more by protocol mismatch than by bad CUDA math.

### 2. Sampler mismatch also mattered

- The older local harnesses used only `top_k + temperature`.
- `llama.cpp` defaults are richer and closer to:
  - `temperature = 0.8`
  - `top_k = 40`
  - `top_p = 0.95`
  - `min_p = 0.05`
- Matching those settings improved output quality and made the local results more consistent with `llama.cpp`.

### 3. Hidden thought suppression is still necessary

- With the more faithful Gemma4 prompt, the model often spends dozens of tokens in hidden thought before returning visible text.
- A large default budget like `200` can consume the whole generation window and produce no visible answer in short smoke tests.
- Lowering the default budget to `32` gives a better harness default while still allowing the model to reason briefly.

### 4. Quality is now materially better on the local harnesses

- After prompt and sampler alignment, the CUDA harness now produces outputs such as:
  - `A majestic view of Mount Fuji dominates the background...`
- This matches the core quality behavior seen in `llama.cpp`: the model correctly identifies the test image as Mount Fuji.

## Repro Procedure

Model assets used:

- `/mnt/disk01/models/gemma4/2b/gemma-4-E2B-it-UD-Q8_K_XL.gguf`
- `/mnt/disk01/models/gemma4/2b/mmproj-F16.gguf`
- `path/to/image.jpg`

### Build

```bash
make -C cpu/vlm test_gemma4_vision
make -C cuda/vlm test_cuda_gemma4_vlm
```

### Local CPU VLM repro

```bash
./cpu/vlm/test_gemma4_vision \
  /mnt/disk01/models/gemma4/2b/gemma-4-E2B-it-UD-Q8_K_XL.gguf \
  /mnt/disk01/models/gemma4/2b/mmproj-F16.gguf \
  path/to/image.jpg \
  "describe the image briefly" 128
```

Expected behavior:

- visible answer is produced
- answer should identify the mountain as Mount Fuji or produce an equivalent high-quality description

### Local CUDA VLM repro

```bash
./cuda/vlm/test_cuda_gemma4_vlm \
  /mnt/disk01/models/gemma4/2b/gemma-4-E2B-it-UD-Q8_K_XL.gguf \
  /mnt/disk01/models/gemma4/2b/mmproj-F16.gguf \
  path/to/image.jpg \
  "describe the image briefly" 128
```

Observed output after alignment:

```text
Here is a brief description of the image:

A majestic view of Mount Fuji dominates the background, with its snow-capped peak rising above a lush green landscape...
```

### `llama.cpp` comparison repro

Baseline `llama.cpp` Gemma4 VLM run:

```bash
/home/syoyo/work/llama.cpp/build/bin/llama-mtmd-cli \
  -m /mnt/disk01/models/gemma4/2b/gemma-4-E2B-it-UD-Q8_K_XL.gguf \
  --mmproj /mnt/disk01/models/gemma4/2b/mmproj-F16.gguf \
  --image path/to/image.jpg \
  -p "describe the image briefly" \
  --n-predict 128 \
  --ctx-size 1024 \
  --verbosity 0 \
  --jinja
```

Matched-sampler `llama.cpp` run used during comparison:

```bash
/home/syoyo/work/llama.cpp/build/bin/llama-mtmd-cli \
  -m /mnt/disk01/models/gemma4/2b/gemma-4-E2B-it-UD-Q8_K_XL.gguf \
  --mmproj /mnt/disk01/models/gemma4/2b/mmproj-F16.gguf \
  --image path/to/image.jpg \
  -p "describe the image briefly" \
  --n-predict 128 \
  --ctx-size 1024 \
  --verbosity 0 \
  --jinja \
  --temp 0.7 \
  --top-k 40 \
  --top-p 1.0 \
  --min-p 0.0 \
  --repeat-penalty 1.0 \
  --presence-penalty 0.0 \
  --frequency-penalty 0.0 \
  --dry-multiplier 0.0 \
  --samplers top_k\;temperature
```

Note:

- `llama.cpp` often still emits hidden-thought text in the visible stream on this path.
- Even so, its reasoning text explicitly identified the mountain as Mount Fuji.
- After local prompt/sampler alignment, the local harnesses recover that same landmark identification.

## CPU vs CUDA Summary

### CPU VLM

- Reasoning budget exists
- Hidden thought suppression exists
- Control-token suppression exists
- Prompt allocation is now robust
- Prompt framing is now much closer to `llama.cpp`
- Sampling is now much closer to `llama.cpp`
- Hidden-norm trace noise is disabled
- Builds cleanly enough for the reviewed target

### CUDA VLM

- Prompt structure and reasoning-budget logic broadly match CPU
- Batched prefill path is in place
- GPU vision encode path is wired through `cuda_llm_vision_encode()`
- Hidden thought suppression now matches CPU
- Control-token suppression exists
- Prompt allocation is now robust
- Prompt framing is now much closer to `llama.cpp`
- Sampling is now much closer to `llama.cpp`
- End-to-end quality on the local Fuji smoke test is now good enough to identify Mount Fuji

## Recommended Next Steps

1. Add CLI flags for `temp`, `top_k`, `top_p`, `min_p`, and `budget` so future comparisons do not require source edits.
2. Add one small regression test or golden-output check for:
   - hidden thought suppression
   - forced budget exit
   - long-prompt handling
3. If quality still diverges from `llama.cpp` on harder images, compare image embedding handling next.

## Bottom Line

The current local result is materially better than the earlier harness behavior:

- CPU and CUDA VLM now use a Gemma4 prompt shape that is much closer to `llama.cpp`
- CPU and CUDA VLM now use a closer sampling policy
- CUDA VLM now produces a meaningful Mount Fuji description on the local smoke test

The main remaining work is regression coverage and parameterization, not basic Gemma4 VLM viability.
