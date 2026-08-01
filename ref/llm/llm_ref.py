#!/usr/bin/env python3
"""
Pytorch reference for the LLM/VLM compare server.

Takes a HF model path (or hub id), a prompt, and an optional image; runs
deterministic greedy decoding; writes a JSON blob with the generated text,
token id list, and timings.

CLI:
    --kind {llm,vlm}       task kind. llm = text-only causal LM.
                           vlm = image + text -> text (vision2seq / processor).
    --model <path>         HF model dir or hub id.
    --prompt <text>        User prompt (wrapped with the model's chat template
                           when a tokenizer exposes one; otherwise used raw).
    --image <path>         Optional image path (required when kind=vlm).
    --max-tokens N         Max new tokens (default 128).
    --device {cuda,cpu}    Default cuda if available.
    --dtype {auto,bf16,fp16,fp32}  Default auto (bf16 on cuda, fp32 on cpu).
    --system <text>        Optional system prompt.
    --out <path>           Output JSON path (required).

Output JSON schema:
    {
      "text": "<generated text, no prompt>",
      "tokens": [<int>, ...],          # token ids of the generation
      "prompt_tokens": [<int>, ...],   # token ids fed to the model
      "timings_ms": {"total": <ms>, "load": <ms>, "gen": <ms>},
      "tokenizer": "<name_or_path>",
      "model": "<name_or_path>"
    }
"""
import argparse
import json
import os
import sys
import time


def _log(msg):
    sys.stderr.write(f"[llm_ref] {msg}\n")
    sys.stderr.flush()


def _resolve_dtype(name, device):
    import torch

    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    # auto
    if device.startswith("cuda"):
        return torch.bfloat16
    return torch.float32


def _build_prompt_messages(system, prompt):
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    return msgs


def run_llm(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t_start = time.time()
    dtype = _resolve_dtype(args.dtype, args.device)
    _log(f"load tokenizer: {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    _log(f"load model: {args.model} dtype={dtype}")
    mdl = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, trust_remote_code=True
    )
    mdl = mdl.to(args.device).eval()
    t_loaded = time.time()

    msgs = _build_prompt_messages(args.system, args.prompt)
    if hasattr(tok, "apply_chat_template") and tok.chat_template:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    else:
        text = (args.system + "\n" if args.system else "") + args.prompt
    inputs = tok(text, return_tensors="pt").to(args.device)
    prompt_ids = inputs["input_ids"][0].tolist()
    _log(f"prompt tokens: {len(prompt_ids)}")

    with torch.inference_mode():
        t_gen0 = time.time()
        out = mdl.generate(
            **inputs,
            max_new_tokens=args.max_tokens,
            do_sample=False,
            temperature=1.0,
            num_beams=1,
            use_cache=True,
        )
        t_gen1 = time.time()

    full = out[0].tolist()
    new_ids = full[len(prompt_ids):]
    text_out = tok.decode(new_ids, skip_special_tokens=True)

    return {
        "text": text_out,
        "tokens": new_ids,
        "prompt_tokens": prompt_ids,
        "timings_ms": {
            "total": (time.time() - t_start) * 1000.0,
            "load": (t_loaded - t_start) * 1000.0,
            "gen": (t_gen1 - t_gen0) * 1000.0,
        },
        "tokenizer": args.model,
        "model": args.model,
    }


def run_vlm(args):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from PIL import Image

    t_start = time.time()
    dtype = _resolve_dtype(args.dtype, args.device)
    _log(f"load processor: {args.model}")
    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    _log(f"load model: {args.model} dtype={dtype}")
    mdl = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=dtype, trust_remote_code=True
    )
    mdl = mdl.to(args.device).eval()
    t_loaded = time.time()

    if not args.image:
        raise SystemExit("--image is required when --kind vlm")
    img = Image.open(args.image).convert("RGB")

    user_content = [{"type": "image"}, {"type": "text", "text": args.prompt}]
    msgs = []
    if args.system:
        msgs.append({"role": "system", "content": [{"type": "text", "text": args.system}]})
    msgs.append({"role": "user", "content": user_content})

    try:
        prompt_text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    except Exception:  # noqa: BLE001
        prompt_text = (args.system + "\n" if args.system else "") + args.prompt

    inputs = proc(text=prompt_text, images=[img], return_tensors="pt").to(args.device, dtype=dtype)
    input_ids = inputs.get("input_ids")
    prompt_ids = input_ids[0].tolist() if input_ids is not None else []

    with torch.inference_mode():
        t_gen0 = time.time()
        out = mdl.generate(
            **inputs,
            max_new_tokens=args.max_tokens,
            do_sample=False,
            temperature=1.0,
            num_beams=1,
            use_cache=True,
        )
        t_gen1 = time.time()

    full = out[0].tolist()
    new_ids = full[len(prompt_ids):] if prompt_ids else full
    tok = getattr(proc, "tokenizer", None)
    if tok is None:
        text_out = proc.decode(new_ids, skip_special_tokens=True)
    else:
        text_out = tok.decode(new_ids, skip_special_tokens=True)

    return {
        "text": text_out,
        "tokens": new_ids,
        "prompt_tokens": prompt_ids,
        "timings_ms": {
            "total": (time.time() - t_start) * 1000.0,
            "load": (t_loaded - t_start) * 1000.0,
            "gen": (t_gen1 - t_gen0) * 1000.0,
        },
        "tokenizer": args.model,
        "model": args.model,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["llm", "vlm"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--image", default=None)
    ap.add_argument("--system", default=None)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.device is None:
        try:
            import torch

            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:  # noqa: BLE001
            args.device = "cpu"

    if args.seed:
        try:
            import torch
            torch.manual_seed(args.seed)
            if args.device.startswith("cuda"):
                torch.cuda.manual_seed_all(args.seed)
        except Exception:  # noqa: BLE001
            pass

    if args.kind == "llm":
        result = run_llm(args)
    else:
        result = run_vlm(args)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    _log(f"wrote {args.out}  text_len={len(result['text'])}  gen_toks={len(result['tokens'])}")


if __name__ == "__main__":
    main()
