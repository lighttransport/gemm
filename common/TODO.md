# Vision Encoder TODO

## Current Status (2026-02-19)

- Deepstack fix applied: norm→fc1→GELU→fc2 ordering matches llama.cpp reference
- M-RoPE implemented on both CPU and GPU
- GPU vs CPU max diff: 0.000395 (excellent)
- Output: "a scenic photograph of Mount Fuji, the iconic volcano in Japan, captured on a clear sunny day"
- Pixel unshuffle: confirmed NOT needed (explicit 2D indexing handles it)

## DONE

- [x] Deepstack ordering fix (norm→fc1→GELU→fc2)
- [x] CPU M-RoPE — verified correct against llama.cpp
- [x] GPU M-RoPE — Vulkan compute shader (rope_vision_f32.comp)
- [x] GPU vs CPU diff < 0.01 (achieved 0.0004)
- [x] Pixel unshuffle analysis — unnecessary with explicit 2D spatial ops
- [x] Output mentions "Mount Fuji" / "Japan" (confirmed)

---

# VLM/LLM Parity Review TODO

## Review Snapshot (2026-02-18)

Compared `gemm` Vulkan/common paths against recent commits and `llama.cpp` (`/home/syoyo/work/llama.cpp`) using local GGUFs under `/mnt/disk1/models`.

### High Severity

- [x] **MoE architecture fallback bug mitigated**
  - `qwen3vlmoe` / `qwen3moe` no longer silently fall back to `qwen2`.
  - Vulkan loader still fails fast with explicit unsupported-architecture errors.
  - CPU path now implements MoE FFN routing (top-k experts) for compatible expert tensor quant types.

- [x] **Required tensor validation tightened**
  - CPU/Vulkan loaders now treat missing required tensors as fatal load errors.
  - Prevents proceeding into invalid matvec/decode paths with null or missing buffers.

- [x] **Context/KV safety improved**
  - Vulkan hardcoded `max_seq_len=1024` removed.
  - Runtime context now derives from GGUF `*.context_length`, with configurable cap via `GEMM_MAX_SEQ_LEN`.
  - Added cache-position bounds checks in CPU/Vulkan forward paths.

### Medium Severity

- [x] **M-RoPE multimodal position progression parity**
  - `vulkan/test_vision_multimodal.cc` now mirrors llama.cpp mtmd behavior:
    - separate `cache_pos` (KV slot) and `rope_pos` (logical M-RoPE position)
    - image token positions use `pos_0` offsets (`t=pos_0`, `h=pos_0+y`, `w=pos_0+x`)
    - post-image `rope_pos` advances by `max(nx, ny)` instead of `n_vision_tokens`

- [x] **Broader quant type support (K-quants)**
  - Added `Q2_K`, `Q3_K`, `Q5_K` dequant support in `common/ggml_dequant.h` (ported from local `llama.cpp`).
  - Updated CPU/Vulkan row-size/type handling to accept these formats.
  - Validated end-to-end load + prefill on `/mnt/disk1/models/Qwen3-Coder-30B-A3B-Instruct-UD-Q2_K_XL.gguf`.

- [x] **Chat template integration**
  - Multimodal prompt builders now read `tokenizer.chat_template` from GGUF and enable template-aware chatml+vision formatting when supported.
  - Added explicit fallback logging when template is missing/unsupported.

## Next Major Work

- [ ] Implement Vulkan MoE inference path for `qwen3moe` / `qwen3vlmoe`.
  - Current status: optional experimental bridge exists via `GEMM_MOE_CPU_FALLBACK=1` in `VulkanLLMRunner`, but native GPU MoE path is still missing.
- [ ] Extend quant support beyond current set (`Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/Q8_0/F16/F32`) as needed by additional GGUFs.
