# Qwen3.8 server correctness and performance — 2026-09-05

Hardware: Ryzen Threadripper 1950X (16 cores), RX 9070 XT (gfx1201).
Model: Qwen3.8-Flash-Next-UD-Q4_K_XL, four GGUF shards.
Launcher: `run_qwen38_codex_server_rocm.sh`, 7200 MiB expert cache,
65536-token context, 512-token prefill batches, 16 OpenMP threads.
Sampling: temperature 1.0, top_p .95, top_k 40, min_p .01,
presence penalty 0; no implicit frequency/no-repeat penalty in server requests.

## Findings and fixes

- Calling raw ggml CPU dot kernels after `dlopen` without `ggml_cpu_init`
  leaves FP16 lookup tables uninitialized. A Q8 dot product expected to be 32
  returned 0 before initialization and 32 afterward. Live CPU expert outputs
  had relative L2 error 1 against the dequantized-weight/FP32-activation
  reference; after initialization, sampled errors were about .008–.013.
  This initialization is needed by both CPU prefill and decode experts.
- `LLM_MOE_CPU_MIN_WEIGHT=1` drops uncached experts instead of evaluating
  the full selected route. The launcher now defaults to 0.
- CPU decode refills require a copy stream and completion events independently
  of `LLM_MOE_COPY_PIPELINE`. Previously disabling the prefill pipeline also
  omitted these events, leaving slots permanently pending. Decode-to-prefill
  transitions now drain outstanding copies and publish their cache identities.
  Only one refill per layer can be outstanding with the current metadata.
- Sampler softmax must preserve its maximum logit before overwriting the
  probability array. A constant logit shift must not change samples.
- The GGUF explicitly sets `tokenizer.ggml.add_bos_token=false`; the presence
  of a BOS token id does not authorize inserting it.
- Non-thinking chat history includes empty thinking frames, and leading
  system/developer messages are merged, matching the inspected GGUF template.
- Runner stdout diagnostics must be consumed inside their request transaction;
  otherwise the next request can receive the preceding request's completion.
- Recurrent prefix snapshots cannot restore overwritten positional KV. Reset,
  cancellation, and failed prefill invalidate the corresponding snapshot.
  An EOS token sampled but never forwarded is not part of reusable state.

## CPU-only regressions

From the repository root:

```sh
python3 rdna4/llm/test_sampler.py
python3 rdna4/llm/test_prompt_policy.py
python3 rdna4/llm/test_chat_template.py
python3 rdna4/llm/test_codex_protocol.py
python3 rdna4/llm/test_copy_lifecycle.py
python3 rdna4/llm/test_cpu_library.py /path/to/libggml-cpu.so
make -C rdna4/llm -j2
bash -n rdna4/llm/run_qwen38_codex_server_rocm.sh
git diff --check
```

These passed locally using
`/mnt/nvme02/work/llama.cpp/build-codex-hetero-dev2/bin/libggml-cpu.so.0.22.0`.
The build retains pre-existing warnings; it is not warning-clean.
`LLM_MOE_CPU_VERIFY=1` enables the expensive live CPU-expert reference
comparison. Do not use it when measuring performance.

## Real Codex validation

Used `CODEX_HOME=/home/syoyo/.codex-local`, whose provider points at
`http://127.0.0.1:8080/v1`, and an isolated temporary working directory:

```sh
CODEX_HOME=/home/syoyo/.codex-local codex exec \
  --skip-git-repo-check --sandbox read-only -C /tmp/TEST_DIRECTORY --json \
  'Write a complete C program that sums integers 1 through 10 and prints the result. Return only a fenced C code block. Do not inspect files or run tools.'
```

Resume the returned thread id with `codex exec resume --skip-git-repo-check
--json THREAD_ID PROMPT`. The second prompt changes the upper bound to 20;
the third requests the sum of squares from 1 through 20. Generated programs
were compiled with `cc -Wall -Wextra -Werror` and checked for results 55,
210, and 2870. These are conversational code-generation tests, not autonomous
tool-execution tests. The HTTP shim does not yet implement tool-call transport.

Representative single first-turn measurements (not a statistical benchmark):

| Configuration | Prompt tokens | Prefill tok/s | Decode tok/s | Expert cache hits |
| --- | ---: | ---: | ---: | ---: |
| CPU initialized, refill events missing | 4792 | 115.45 | 8.46 | 20.4% |
| CPU initialized, working refills, 16 threads | 4792 | 116.90 | 15.16 | 55.4% |
| Working refills, 8 threads | 4792 | 116.30 | 14.40 | 55.5% |
| Working refills, aligned chat framing, 16 threads | 4779 | 116.90 | 17.25 | 55.7% |

The latest second Codex turn reused 4594 of 4883 prompt tokens and processed
289 new tokens. Its prefill was 89.44 tok/s and decode 13.40 tok/s. System
prefix reuse is confirmed; full conversation-prefix reuse is not established.
Short/suffix-prefill throughput should not be compared directly with the
long initial prompt. Fast runs that omitted experts or returned malformed
code are not valid decode-performance baselines.

Remaining work includes autonomous tool-call support, request-owned concurrent
cancellation, longer-context validation, and further decode optimization.
