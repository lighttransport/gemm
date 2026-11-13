# Qwen3.5-2B text tower from safetensors (cuda/llm)

Direct HF-safetensors path (no GGUF) used by the praxis text encoder:
`cuda_llm_load_weights_qwen35_safetensors` + `cuda_llm_prefill_hidden_ex` / `cuda_llm_prefill_hidden_all`,
driven by `test_qwen35_st`. bf16 weights are converted to F16 on the GPU (bit-identical to the host
loop, rel-L2 of the weight conversion itself 3.5e-8); norms +1, `A_log -> -exp`.

## Usage (run from the repo root)
    make -C cuda/llm test_qwen35_st
    ./cuda/llm/test_qwen35_st /mnt/disk1/praxis/qwen3.5-2b -i ids.npy -o out_dir -p 32 --cond-only   # praxis_e2e usage
    ./cuda/llm/test_qwen35_st /mnt/disk1/praxis/qwen3.5-2b -i ids.npy -o out_dir                    # layer_00..24 + cond
    ./cuda/llm/test_qwen35_st --help

Outputs: `layer_NN.npy` = HF `hidden_states[NN]` (NN 0..23 pre-final-norm, **layer_24 post-final-norm**),
`cond.npy` = `hidden_states[-3]` (output of layer 21). `--cond-only` runs only 22 layers and syncs once.
`-p N` right-pads with `<|endoftext|>` to a multiple of N; the model is causal so real rows are unchanged
(padded == unpadded to fp noise, 0 to 4e-4 rel depending on GEMM tile choice); padding rows are trimmed.
`--chunk N` feeds the prompt in N-token chunks (start_pos>0 continuation: KV cache + conv/recurrent state
carried) - equals the one-shot result to noise (6e-4 rel).

### Persistent mode (avoids the ~1.2 s load + ~0.5 s cuBLAS cold start per prompt)
    ./cuda/llm/test_qwen35_st MODEL --serve -p 32 --cond-only
stdin, one request per line: `<ids.npy> <out_dir>` or `- <out_dir> id id id ...`; stdout: `OK <n> <ms>` /
`ERR ...`; `quit`/EOF exits. Measured per request after the first: 39 ms (32 tok), 83 ms (243 tok).

### API
`cuda_llm_prefill_hidden_ex(r, ids, n, n_run_layers, start_pos, dump_only_slot, out_per_layer, out_normed)`:
start_pos==0 resets recurrent state, >0 continues; `out_normed` = post-final-norm output. The old
`cuda_llm_prefill_hidden_all` is a wrapper. Loader `max_seq_len` bounds start_pos+n (test default max(2048, padded)).

## Accuracy (cond / layer_24 rel-L2 vs oracle; GPU: RTX 5060 Ti)
| prompt (tokens) | vs HF fp32 (L22 / L24) | vs HF bf16 (L22 / L24) | HF bf16 vs HF fp32 |
|---|---|---|---|
| 11  | 3.4e-4 / 4.1e-4 | 1.2e-2 / 1.6e-2 | 1.2e-2 / 1.6e-2 |
| 243 | 4.7e-4 / 6.3e-4 | 1.3e-2 / 1.7e-2 | 1.3e-2 / 1.7e-2 |
| 1158| 8.6e-4 / 1.1e-3 | 1.7e-2 / 2.2e-2 | 1.7e-2 / 2.2e-2 |
We are ~15-20x closer to fp32 than HF's own bf16 run is. The residual error is F16 *activation* rounding
in the cuBLAS tensor-core GEMMs (not the weights: bf16->F16 changes 0.04% of weights, subnormals only).
`CUDA_LLM_USE_CUBLAS=0` selects the F32-activation fallback kernel (1.3e-4 .. 3e-4) but is ~45x slower at 1024 tok.

## Performance knobs (defaults set by the qwen35 loader)
- `CUDA_LLM_GPU_BF16_TO_F16` (default on here): GPU conversion, load 5.1 s -> ~1.0-1.4 s. (The f16 sidecar cache
  never applied: its key expects `model-0000N-of-0000M.safetensors` names, this checkpoint is `model.safetensors-00001-of-00001...`.)
- `CUDA_LLM_USE_CUBLAS` (default on here; 0 = naive kernel).
- Host embed (default): the 1 GB token table stays in the mmap and rows are gathered on the host; no LM head.
  `CUDA_LLM_QWEN35_FULL_EMBED=1` uploads it (needed only for logits/decode).
- NVRTC: the PTX disk cache (`/tmp/cuda_llm_sm_*_v*_<hash>.ptx`) + driver ComputeCache work (init 0.25-0.3 s warm).

## Reference generation
`python3 ref/praxis/gen_text_ref.py --prompt-file p.txt|--ids ids.npy [--dtype bfloat16] [--device cuda] --out DIR`.
NOTE: with transformers 5.x `AutoModel` silently re-initialises ALL weights for this checkpoint (key-prefix mismatch,
"LOAD REPORT ... MISSING"); the script now loads `AutoModelForImageTextToText` and uses `.model.language_model`
(matches the committed tmp/praxis_ref/text dump to 5e-7). `ref/praxis/gen_e2e_ref.py` still uses `AutoModel`.

## Check
`make -C cuda/llm test_qwen35_st_check` runs the 11-token reference prompt (padded to 32) and requires all 25 hidden
states + cond to pass `compare.py --tol 0.005`.

## In-process use from praxis_e2e
`cuda/praxis/praxis_e2e` no longer spawns this binary by default: it links `cuda_llm_runner.c` in-process (`cuda/praxis/praxis_qwen.c`: `praxis_qwen_open` = `cuda_llm_init` +
`cuda_llm_load_weights_qwen35_safetensors`, `praxis_qwen_cond` = `cuda_llm_prefill_hidden_ex(ids padded to x32, 22 layers, dump slot 22)`, `praxis_qwen_close` = `cuda_llm_free`). After `cuda_llm_free`
the process's device memory returns to the pre-load level (nvidia-smi per-process delta 0.000-0.004 GB), so the VAE/policy peak is unchanged. Output is bit-identical to this driver's `--cond-only -p 32`.
This binary remains the fallback (`praxis_e2e --qwen-spawn --qwen-bin cuda/llm/test_qwen35_st`) and `--serve` is the persistent-child alternative.
