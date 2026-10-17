#!/usr/bin/env python3
"""Export FLUX3 Qwen3-VL features on CPU for the native HIP DiT.

Matches BFL flux3/text_encoder.py. No CUDA/vLLM dependency; the GPU remains
available for the native runner. Requires torch, transformers, accelerate,
safetensors and numpy. Checkpoint must already be downloaded.
"""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("/mnt/nvme02/model/flux3-klein"))
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, required=True, help="little-endian F32 [tokens,20480]")
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    (root / "tmp").mkdir(exist_ok=True)
    os.environ.setdefault("TMPDIR", str(root / "tmp"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration

    torch.set_num_threads(args.threads)
    path = args.model_dir / "text_encoder"
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    # For a text-only conversation, use the processor's exact template without
    # loading its unused image/video processors (and torchvision).
    template = json.loads((path / "chat_template.json").read_text())["chat_template"]
    tokenizer.padding_side = "right"
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}], tokenize=False, add_generation_prompt=True, chat_template=template
    )
    real = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)["input_ids"].shape[1]
    length = min(((real + 79) // 80) * 80, 4096)
    inputs = tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=length)
    mask = inputs["attention_mask"]
    positions = (mask.long().cumsum(-1) - 1).masked_fill(mask == 0, 1)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        path, dtype=torch.bfloat16, local_files_only=True, attn_implementation="eager"
    ).eval()
    with torch.inference_mode():
        output = model(
            input_ids=inputs["input_ids"], attention_mask=mask,
            position_ids=positions[None].expand(4, -1, -1),
            output_hidden_states=True, use_cache=False,
        )
        features = torch.stack([output.hidden_states[k] for k in (4, 8, 12, 16, 20, 24, 28, 32)], dim=1)
        features = features.permute(0, 2, 1, 3).reshape(length, 20480).to(torch.bfloat16).float()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.numpy().astype("<f4").tofile(args.output)
    manifest = {"prompt": args.prompt, "tokens": length, "real_tokens": real, "channels": 20480,
                "layers": [4, 8, 12, 16, 20, 24, 28, 32], "model": str(path.resolve()),
                "dtype": "F32", "bf16_rounded": True}
    args.output.with_suffix(args.output.suffix + ".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
