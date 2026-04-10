#!/usr/bin/env python3
"""
Encode text prompt using ComfyUI's text encoder and save as .npy.

Usage:
    uv run python encode_text_comfyui.py \
        --prompt "a red apple on a white table" \
        --negative " " \
        --output text_hidden.npy
"""
import argparse, os, sys, time
import numpy as np
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--comfyui-dir', default='/mnt/disk01/ComfyUI')
    parser.add_argument('--model-dir', default='/mnt/disk01/models/qwen-image-st')
    parser.add_argument('--prompt', default='a red apple on a white table')
    parser.add_argument('--negative', default=' ')
    parser.add_argument('--output', default='text_hidden.npy')
    args = parser.parse_args()

    sys.path.insert(0, args.comfyui_dir)
    os.chdir(args.comfyui_dir)

    import comfy.sd
    clip_path = os.path.join(args.model_dir, 'text_encoders',
                             'qwen_2.5_vl_7b_fp8_scaled.safetensors')
    print(f"Loading CLIP: {clip_path}")
    clip = comfy.sd.load_clip(ckpt_paths=[clip_path],
                               clip_type=comfy.sd.CLIPType.QWEN_IMAGE)
    print(f"Loaded in {time.time():.1f}s")

    # Encode positive
    tokens = clip.tokenize(args.prompt)
    pos = clip.encode_from_tokens_scheduled(tokens)
    # Extract hidden states from the cond structure
    # pos is a list of [cond_dict, ...], cond_dict has 'pooled_output' etc.
    pos_hs = pos[0][0]  # first element, first entry
    print(f"Positive cond shape: {pos_hs.shape}")
    np.save(args.output, pos_hs.cpu().float().numpy())

    # Encode negative
    tokens_neg = clip.tokenize(args.negative)
    neg = clip.encode_from_tokens_scheduled(tokens_neg)
    neg_hs = neg[0][0]
    neg_path = args.output.replace('.npy', '_neg.npy')
    print(f"Negative cond shape: {neg_hs.shape}")
    np.save(neg_path, neg_hs.cpu().float().numpy())

    print(f"Saved: {args.output}, {neg_path}")

if __name__ == "__main__":
    main()
