#!/usr/bin/env python3
"""
Dump Flux.2 Klein text-encoder hidden states to a raw F32 binary.

The HIP runner's `--txt-bin` option reads [n_tokens, 7680] F32 contiguously.
diffusers' Flux2KleinPipeline pads with the tokenizer to 512 tokens and runs
Qwen3 with attention_mask, stacking layers (9, 18, 27) to yield [1, 512, 7680].
"""
import argparse
import numpy as np
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="a red apple on a white table")
    ap.add_argument("--model", default="black-forest-labs/FLUX.2-klein-4B")
    ap.add_argument("-o", "--out", default="apple_text.bin")
    args = ap.parse_args()

    from diffusers import Flux2KleinPipeline

    print(f"loading text encoder from {args.model}")
    pipe = Flux2KleinPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    # Only the text encoder + tokenizer are needed. Move it to GPU to make encode fast.
    pipe.text_encoder.to("cuda")

    embeds = Flux2KleinPipeline._get_qwen3_prompt_embeds(
        text_encoder=pipe.text_encoder,
        tokenizer=pipe.tokenizer,
        prompt=args.prompt,
        dtype=torch.float32,
        device=torch.device("cuda"),
    )
    print(f"embeds shape: {tuple(embeds.shape)} dtype={embeds.dtype}")
    arr = embeds[0].detach().cpu().numpy().astype(np.float32)
    print(f"arr shape {arr.shape} mean {arr.mean():.5f} std {arr.std():.5f}")
    arr.tofile(args.out)
    print(f"saved {args.out} ({arr.nbytes} bytes = {arr.shape[0]} tokens × {arr.shape[1]} dim F32)")

if __name__ == "__main__":
    main()
