#!/usr/bin/env python3
"""Experimental native Qwen3-VL encoder with a Python tokenizer boundary.

Not yet a validated replacement for the reference encoder. --prepare-only
does not load model weights or initialize CUDA.
"""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np


def prepare(model: Path, prompt: str, work: Path) -> int:
    from transformers import Qwen3VLProcessor

    config = json.loads((model / "text_encoder/config.json").read_text())["text_config"]
    required = {"hidden_size": 4096, "num_hidden_layers": 36, "num_attention_heads": 32,
                "num_key_value_heads": 8, "head_dim": 128, "intermediate_size": 12288,
                "vocab_size": 151936, "rope_theta": 5000000, "rms_norm_eps": 1e-6}
    for key, expected in required.items():
        if config.get(key) != expected:
            raise ValueError(f"unsupported text config {key}={config.get(key)!r}; expected {expected}")
    processor = Qwen3VLProcessor.from_pretrained(str(model / "processor"), local_files_only=True)
    system = "Comprehend and analyze the provided prompt."
    template = (f"<|im_start|>system\n{system}<|im_end|>\n"
                f"<|im_start|>user\n{prompt or ' '}<|im_end|>\n<|im_start|>assistant\n")
    # The official pipeline deliberately does not apply_chat_template to the
    # user prompt. It uses that method only to calculate the system crop.
    system_tokens = processor.apply_chat_template(
        [{"role": "system", "content": [{"type": "text", "text": system}]}],
        tokenize=True, return_dict=False)
    drop = len(system_tokens[0])
    inputs = processor(text=[template], padding=True, padding_side="left", return_tensors="pt")
    ids = inputs.input_ids.numpy()
    if ids.shape[0] != 1 or not inputs.attention_mask.bool().all():
        raise ValueError("native encoder requires one unpadded text sequence")
    work.mkdir(parents=True, exist_ok=True)
    np.save(work / "input_ids.npy", ids)
    (work / "tokens.txt").write_text("\n".join(str(int(v)) for v in ids[0]) + "\n")
    (work / "prepare.json").write_text(json.dumps(
        {"model": str(model), "prompt": prompt, "drop_idx": drop,
         "tokens": int(ids.shape[1]), "native_parity_validated": False}, indent=2) + "\n")
    return drop


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--prompt", default="a red apple on a white table")
    ap.add_argument("--work-dir", type=Path, default=Path("tmp/qimg21-native-text"))
    ap.add_argument("--out", type=Path, default=Path("tmp/qimg21-native-text/prompt_embeds.npy"))
    ap.add_argument("--prepare-only", action="store_true")
    args = ap.parse_args()
    drop = prepare(args.model.resolve(), args.prompt, args.work_dir)
    if not args.prepare_only:
        root = Path(__file__).resolve().parents[2]
        args.out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([str(root / "cuda/qimg21/test_cuda_qimg21_text"),
                        "--model", str(args.model.resolve()), "--tokens", str(args.work_dir / "tokens.txt"),
                        "--drop-prefix", str(drop), "--out", str(args.out)], check=True)


if __name__ == "__main__":
    main()
