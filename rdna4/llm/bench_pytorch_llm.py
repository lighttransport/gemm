#!/usr/bin/env python3
"""PyTorch/ROCm reference benchmark for the Qwen3-VL LLM body.

Mirrors `rdna4/llm/test_hip_llm --bench` semantics: a fixed-length synthetic
prefill followed by a greedy single-token decode loop, with each phase timed
separately.

Uses back-to-back submission + a single torch.cuda.synchronize per phase to
keep the GPU warm (the same trick as bench_pytorch_vision.py — naive
per-iteration syncs let the GPU clock down-ramp and inflate the mean).
"""
import argparse, json, time, sys
import torch
from transformers import AutoModelForImageTextToText, AutoConfig

ap = argparse.ArgumentParser()
ap.add_argument("--model-dir", required=True,
                help="HF snapshot dir (config.json + model.safetensors)")
ap.add_argument("--prefill-lens", default="64,256,1024")
ap.add_argument("--decode", type=int, default=128)
ap.add_argument("--warmup", type=int, default=2)
ap.add_argument("--iters", type=int, default=5)
ap.add_argument("--dtype", default="f16", choices=["bf16", "f16", "f32"])
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

DT = {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32}[args.dtype]
dev = "cuda"
torch.manual_seed(args.seed)

cfg = AutoConfig.from_pretrained(args.model_dir)
# Text branch only — vision tower is irrelevant for the LLM bench.
text_cfg = cfg.text_config
print(f"text cfg: layers={text_cfg.num_hidden_layers} hidden={text_cfg.hidden_size} "
      f"heads={text_cfg.num_attention_heads} kv_heads={text_cfg.num_key_value_heads} "
      f"ffn={text_cfg.intermediate_size} vocab={text_cfg.vocab_size}")

print(f"loading model from {args.model_dir} ...", flush=True)
model = AutoModelForImageTextToText.from_pretrained(
    args.model_dir, dtype=DT, attn_implementation="sdpa").to(dev).eval()
# Use the full model with text-only input (no pixel_values) so we get logits.
# Drop the visual tower to free VRAM since we only bench the LLM.
if hasattr(model, "model") and hasattr(model.model, "visual"):
    del model.model.visual
elif hasattr(model, "visual"):
    del model.visual
torch.cuda.empty_cache()
lm = model
# Count just the language-model params for the print.
lm_params = 0
for name, p in lm.named_parameters():
    if "visual" in name: continue
    lm_params += p.numel()
print(f"loaded on {dev}; LM params={lm_params/1e9:.2f}B  dtype={args.dtype}")

vocab = text_cfg.vocab_size

def bench(prefill_len):
    ids = torch.randint(0, vocab, (1, prefill_len), device=dev, dtype=torch.long)

    with torch.inference_mode():
        # ---- prefill (one forward pass, fresh kv cache) ----
        for _ in range(args.warmup):
            out = lm(ids, use_cache=True)
            del out
        torch.cuda.synchronize()

        # Back-to-back N prefills; one sync at the end.
        t0 = time.perf_counter()
        for _ in range(args.iters):
            out = lm(ids, use_cache=True)
        torch.cuda.synchronize()
        prefill_ms = (time.perf_counter() - t0) * 1e3 / args.iters
        past = out.past_key_values
        next_tok = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)

        # ---- decode loop (kv-cache warm steady state) ----
        # Warmup a few decode steps.
        cur = next_tok
        for _ in range(args.warmup):
            o = lm(cur, past_key_values=past, use_cache=True)
            past = o.past_key_values
            cur = torch.argmax(o.logits[:, -1, :], dim=-1, keepdim=True)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(args.decode):
            o = lm(cur, past_key_values=past, use_cache=True)
            past = o.past_key_values
            cur = torch.argmax(o.logits[:, -1, :], dim=-1, keepdim=True)
        torch.cuda.synchronize()
        decode_total_ms = (time.perf_counter() - t0) * 1e3

    decode_per_tok_ms = decode_total_ms / args.decode
    return prefill_ms, decode_per_tok_ms

print(f"\ndevice={torch.cuda.get_device_name(0)}  dtype={args.dtype}  "
      f"warmup={args.warmup} iters={args.iters} decode={args.decode}\n")
print(f"{'prefill_len':>12} {'prefill ms':>12} {'prefill tok/s':>14} "
      f"{'decode ms/tok':>14} {'decode tok/s':>13}")
for L in [int(x) for x in args.prefill_lens.split(",")]:
    p_ms, d_ms_per_tok = bench(L)
    p_tps = 1000.0 * L / p_ms
    d_tps = 1000.0 / d_ms_per_tok
    print(f"{L:>12} {p_ms:>12.2f} {p_tps:>14.1f} "
          f"{d_ms_per_tok:>14.3f} {d_tps:>13.1f}")
