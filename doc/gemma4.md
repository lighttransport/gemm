# Gemma4 Review Snapshot

Last reviewed: 2026-04-09

## Scope

This note reflects the current local Gemma4 code paths in:

- `common/transformer.h`
- `common/gemma4_vision_encoder.h`
- `cpu/vlm/test_gemma4_vision.c`
- `cuda/vlm/test_cuda_gemma4_vlm.c`

It is a code review snapshot, not a model-quality claim. I did not run end-to-end inference here because the required model assets were not provided in this workspace.

## Current State

### Implemented

- Gemma4 architecture support exists in both CPU and CUDA LLM paths.
- Gemma4-specific pieces are present in the shared transformer layer:
  - per-layer embeddings
  - SWA/full-attention handling
  - proportional RoPE handling
  - Gemma4 logit soft-capping
  - vision-token handling through `transformer_forward_embd[_logits]`
- The CPU and CUDA Gemma4 VLM harnesses now use a direct-answer multimodal prompt shape:
  - `BOS`
  - `<|turn>system\nYou are a helpful visual assistant. Answer briefly and directly in plain text.\n<turn|>\n`
  - `<|turn>user\n<|image>`
  - image embeddings
  - `<image|>{prompt}<turn|>\n<|turn>model\n`
- Both CPU and CUDA VLM harnesses include a reasoning-budget state machine that watches for `<|channel>thought\n ... <channel|>` and can force an exit when the model enters a hidden thought channel.
- CPU VLM explicitly disables hidden-norm tracing with `transformer_set_trace_hidden_norms(model, 0)` so generation logs stay readable.
- Both harnesses now resolve control-token IDs by exact vocab lookup instead of tokenization heuristics.
- Both harnesses now build prompts dynamically, so longer prompts no longer silently truncate the chat framing.
- Both harnesses suppress visible printing of protocol tokens like `<turn|>` / `eos` / `eot`.

### Verified Locally

- `cpu/vlm/test_gemma4_vision` builds successfully from the current tree.
- `cuda/vlm/test_cuda_gemma4_vlm` is present and `make` reports it up to date in the current tree.
- The CPU build still emits several unrelated warnings from shared headers, but no build failure in the reviewed path.

## Review Findings

### 1. Hidden thought-channel leakage was fixed

Files:

- `cpu/vlm/test_gemma4_vision.c`
- `cuda/vlm/test_cuda_gemma4_vlm.c`

Both harnesses now:

- return immediately on `<|channel>`
- suppress all tokens while `in_thought == 1`
- return immediately on `<channel|>`
- suppress `<turn|>`, `eos`, and `eot` in visible output

### 2. Prompt truncation was fixed

Files:

- `cpu/vlm/test_gemma4_vision.c`
- `cuda/vlm/test_cuda_gemma4_vlm.c`

Both harnesses now allocate `pre_image` / `post_image` dynamically and fail cleanly if prompt construction cannot be allocated.

## CPU vs CUDA Summary

### CPU VLM

- Better documented and more coherent than before
- Reasoning budget exists
- Hidden thought suppression exists
- Control-token suppression exists
- Prompt allocation is now robust
- Hidden-norm trace noise is disabled
- Builds cleanly enough for the reviewed target

### CUDA VLM

- Prompt structure and reasoning-budget logic broadly match CPU
- Batched prefill path is in place
- GPU vision encode path is wired through `cuda_llm_vision_encode()`
- Hidden thought suppression now matches CPU
- Control-token suppression exists
- Prompt allocation is now robust
- Error handling is a bit less defensive than CPU in the early model-load path

## Recommended Next Steps

1. Add one small regression test or golden-output check for:
   - hidden thought suppression
   - forced budget exit
   - long-prompt handling
2. If end-to-end quality is still worse than `llama.cpp` after that, compare image embedding handling next, not just prompt text.

## Bottom Line

The Gemma4 codebase is materially ahead of the earlier snapshot in this repo:

- CPU and CUDA VLM harnesses both have reasoning-budget handling now
- CPU and CUDA VLM now share the same direct-answer prompt shape and visible-token filtering

The main remaining review issue is not general Gemma4 support. It is regression coverage and broader quality benchmarking beyond the current smoke tests.
