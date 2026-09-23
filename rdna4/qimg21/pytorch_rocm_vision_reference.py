#!/usr/bin/env python3
"""Replay Qwen3-VL vision patches with the pinned PyTorch ROCm model.

Only the visual checkpoint weights are loaded. The input is the exact patch
tensor captured before the vision encoder, or an image preprocessed with the
pinned Qwen Image 2.1 pipeline, for comparison with the native encoder.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch


def save(path, value):
    if isinstance(value, tuple):
        value = value[0]
    np.save(path, value.detach().float().cpu().numpy())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--input-dir", type=Path)
    source_group.add_argument("--image", type=Path)
    parser.add_argument("--prompt", default="a red apple on a white table")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--trace-block", type=int)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--diffusers-site-packages", required=True, type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("PyTorch ROCm GPU access is required")
    if not args.diffusers_site_packages.is_dir():
        parser.error("pinned site-packages directory does not exist")
    # Load the ROCm torchvision wheel before adding packages from the CUDA
    # reference environment. The latter contributes Transformers only.
    import torchvision  # noqa: F401
    sys.path.insert(0, str(args.diffusers_site_packages.resolve()))
    from safetensors import safe_open
    from transformers import Qwen3VLConfig, Qwen3VLProcessor
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = args.input_dir.resolve() if args.input_dir else out
    if args.image:
        from PIL import Image
        from diffusers.image_processor import VaeImageProcessor
        from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_dimensions

        original = Image.open(args.image).convert("RGBA")
        width, height = original.size
        width, height, _ = calculate_dimensions(args.resolution ** 2, width / height)
        resized = VaeImageProcessor(vae_scale_factor=16, vae_latent_channels=64).resize(
            original, width=width, height=height)
        flattened = Image.new("RGB", resized.size, (255, 255, 255))
        flattened.paste(resized, mask=resized.getchannel("A"))
        prompt = ("<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n"
                  "<|im_start|>user\n<image1><|vision_start|><|image_pad|><|vision_end|>"
                  f"{args.prompt}<|im_end|>\n<|im_start|>assistant\n")
        processor = Qwen3VLProcessor.from_pretrained(
            args.model / "processor", local_files_only=True)
        inputs = processor(text=[prompt], images=[flattened], padding=True,
                           padding_side="left", return_tensors="pt")
        np.save(out / "pixel_values.npy", inputs.pixel_values.float().numpy())
        np.save(out / "image_grid_thw.npy", inputs.image_grid_thw.numpy())
        flattened.save(out / "vision_input.png")
    patches_path = source / "pixel_values.npy"
    grid_path = source / "image_grid_thw.npy"
    patches = np.load(patches_path, allow_pickle=False)
    grid = np.load(grid_path, allow_pickle=False)
    if (patches.dtype != np.float32 or patches.ndim != 2 or patches.shape[1] != 1536 or
            grid.dtype != np.int64 or grid.shape != (1, 3) or
            int(np.prod(grid[0])) != patches.shape[0] or not np.isfinite(patches).all()):
        raise ValueError("expected finite captured Qwen3-VL patch and grid tensors")
    root = args.model.resolve() / "text_encoder"
    config = Qwen3VLConfig.from_pretrained(root, local_files_only=True)
    visual = Qwen3VLVisionModel(config.vision_config).to(dtype=torch.bfloat16)
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    weight_map = {name.removeprefix("model.visual."): shard
                  for name, shard in index.items() if name.startswith("model.visual.")}
    if set(weight_map) != set(visual.state_dict()):
        raise ValueError("visual checkpoint keys differ from the pinned model")
    weights = {}
    for shard in sorted(set(weight_map.values())):
        with safe_open(root / shard, framework="pt", device="cpu") as handle:
            for name, owner in weight_map.items():
                if owner == shard:
                    weights[name] = handle.get_tensor("model.visual." + name)
    visual.load_state_dict(weights, strict=True)
    del weights
    visual.eval().to("cuda")

    visual.patch_embed.register_forward_hook(
        lambda _module, _inputs, output: save(out / "patch_embed.npy", output))
    for index, block in enumerate(visual.blocks):
        block.register_forward_hook(
            lambda _module, _inputs, output, index=index:
            save(out / f"block_{index:02d}.npy", output))
    visual.merger.register_forward_hook(
        lambda _module, _inputs, output: save(out / "merged.npy", output))
    if args.trace_block is not None:
        if not 0 <= args.trace_block < len(visual.blocks):
            parser.error("--trace-block is outside the vision stack")
        block = visual.blocks[args.trace_block]
        for label, module in (("norm1", block.norm1), ("qkv", block.attn.qkv),
                              ("attn_proj", block.attn.proj), ("norm2", block.norm2),
                              ("mlp_fc1", block.mlp.linear_fc1),
                              ("mlp_gelu", block.mlp.act_fn),
                              ("mlp_fc2", block.mlp.linear_fc2)):
            module.register_forward_hook(
                lambda _module, _inputs, output, label=label:
                save(out / f"trace_{label}.npy", output))
    for index, merger in enumerate(visual.deepstack_merger_list):
        merger.register_forward_hook(
            lambda _module, _inputs, output, index=index:
            save(out / f"deepstack_{index}.npy", output))
    start = time.perf_counter()
    with torch.inference_mode():
        result = visual(torch.from_numpy(patches).to("cuda", dtype=torch.bfloat16),
                        torch.from_numpy(grid).to("cuda"))
    torch.cuda.synchronize()
    save(out / "final_hidden.npy", result.last_hidden_state)
    record = {
        "model": str(root), "input_dir": str(source),
        "pixel_values_sha256": hashlib.sha256(patches_path.read_bytes()).hexdigest(),
        "image_grid_thw_sha256": hashlib.sha256(grid_path.read_bytes()).hexdigest(),
        "torch": torch.__version__, "device": torch.cuda.get_device_name(0),
        "patches": patches.shape[0], "grid": grid.tolist(),
        "blocks": len(visual.blocks), "elapsed_seconds": time.perf_counter() - start,
    }
    if args.trace_block is not None:
        record["trace_block"] = args.trace_block
    if args.image:
        record.update(image=str(args.image.resolve()),
                      image_sha256=hashlib.sha256(args.image.read_bytes()).hexdigest(),
                      prompt=args.prompt, resolution=args.resolution,
                      resized_size=[width, height])
    (out / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
