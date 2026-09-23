#!/usr/bin/env python3
"""Run saved Qwen Image 2.1 editing transformer calls with PyTorch ROCm.

This is a diagnostic reference, not a native inference path. It reuses the
captured transformer inputs and offloads checkpoint layers to fit a 16-GiB GPU.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch


def capture_steps(folder):
    inputs = sorted(folder.glob("input_*.npy"))
    if not inputs or [p.name for p in inputs] != [f"input_{i:03d}.npy" for i in range(len(inputs))]:
        raise ValueError("capture requires contiguous input_NNN.npy files")
    for i in range(len(inputs)):
        if not (folder / f"timestep_{i:03d}.npy").is_file():
            raise ValueError(f"missing timestep_{i:03d}.npy")
    return len(inputs)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--capture-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--diffusers-site-packages", type=Path,
                    help="use pinned Diffusers from another local environment; ROCm Torch stays loaded")
    ap.add_argument("--sdpa-backend", choices=("default", "efficient"), default="efficient")
    ap.add_argument("--capture-block0", action="store_true",
                    help="dump first-step embedding and block-0 stages for native comparison")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        ap.error("PyTorch ROCm GPU access is required")
    if args.diffusers_site_packages:
        if not args.diffusers_site_packages.is_dir():
            ap.error("Diffusers site-packages directory does not exist")
        # Preload the ROCm build before adding another environment to sys.path.
        # A CUDA torchvision wheel with ROCm Torch fails at operator registration.
        import torchvision  # noqa: F401
        sys.path.insert(0, str(args.diffusers_site_packages.resolve()))
    from accelerate import cpu_offload
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21Transformer2DModel
    from torch.nn.attention import SDPBackend, sdpa_kernel

    capture = args.capture_dir.resolve()
    steps = capture_steps(capture)
    layout = json.loads((capture / "positive_layout.json").read_text())
    shapes = [[tuple(int(x) for x in shape) for shape in image]
              for image in layout["img_shapes"]]
    if len(shapes) != 1 or not shapes[0] or any(len(shape) != 3 or min(shape) < 1 for shape in shapes[0]):
        raise ValueError("invalid batch-one editing shapes")
    target_tokens = int(np.prod(shapes[0][-1]))
    prompt = np.load(capture / "prompt_embeds.npy", allow_pickle=False)
    image_mask = np.load(capture / "positive_img_mask.npy", allow_pickle=False)
    key_mask = np.load(capture / "positive_encoder_hidden_states_mask.npy", allow_pickle=False)
    if (prompt.dtype != np.float32 or prompt.ndim != 3 or prompt.shape[0] != 1 or
        not np.isfinite(prompt).all() or image_mask.dtype != np.bool_ or
        key_mask.dtype != np.bool_ or key_mask.shape != prompt.shape[:2] or
        image_mask.shape != (1, int(layout["text_slots"]) + target_tokens // 4)):
        raise ValueError("invalid captured prompt or editing masks")

    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    model = QwenImage21Transformer2DModel.from_pretrained(
        str(args.model.resolve()), subfolder="transformer", torch_dtype=torch.bfloat16,
        local_files_only=True, low_cpu_mem_usage=True)
    model.eval()
    cpu_offload(model, execution_device=torch.device("cuda"))
    current_step = [-1]
    if args.capture_block0:
        stages = out / "block0"
        stages.mkdir()

        def save(name, value):
            if current_step[0] == 0:
                if isinstance(value, tuple):
                    value = value[0]
                np.save(stages / f"{name}.npy", value.detach().float().cpu().numpy())

        def output_hook(name):
            return lambda _module, _inputs, output: save(name, output)

        def input_hook(name):
            return lambda _module, inputs: save(name, inputs[0])

        model.time_text_embed.register_forward_hook(output_hook("time2"))
        model.modulation[0].register_forward_hook(output_hook("time2_silu"))
        model.modulation.register_forward_hook(output_hook("mod"))
        block0 = model.transformer_blocks[0]
        block0.register_forward_pre_hook(
            lambda _module, inputs, kwargs: save(
                "hidden0", inputs[0] if inputs else kwargs["hidden_states"]),
            with_kwargs=True)
        block0.register_forward_hook(output_hook("block_00"))
        block0.attn.to_q.register_forward_hook(output_hook("q"))
        block0.attn.to_k.register_forward_hook(output_hook("k"))
        block0.attn.to_v.register_forward_hook(output_hook("v"))
        block0.attn.to_out[0].register_forward_pre_hook(input_hook("attn_raw"))
        block0.img_norm2.register_forward_pre_hook(input_hook("post_attn_hidden"))
        original_modulate = block0._modulate
        modulation_calls = [0]

        def capture_modulate(hidden_states, modulation, target_token_mask):
            result = original_modulate(hidden_states, modulation, target_token_mask)
            if current_step[0] == 0 and modulation_calls[0] == 0:
                save("mod_ln", result[0])
            modulation_calls[0] += 1
            return result

        block0._modulate = capture_modulate
    embeds = torch.from_numpy(prompt).to("cuda", dtype=torch.bfloat16)
    img_mask = torch.from_numpy(image_mask).to("cuda")
    enc_mask = torch.from_numpy(key_mask).to("cuda")
    from contextlib import nullcontext
    context = sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION) if args.sdpa_backend == "efficient" else nullcontext()
    with torch.inference_mode(), context:
        for i in range(steps):
            current_step[0] = i
            source = np.load(capture / f"input_{i:03d}.npy", allow_pickle=False)
            timestep = np.load(capture / f"timestep_{i:03d}.npy", allow_pickle=False)
            if (source.dtype != np.float32 or source.ndim != 3 or source.shape[0] != 1 or
                source.shape[2] != model.config.in_channels or
                source.shape[1] != sum(int(np.prod(s)) for s in shapes[0]) or
                not np.isfinite(source).all() or timestep.shape != (1,) or
                timestep.dtype != np.float32 or not np.isfinite(timestep).all()):
                raise ValueError(f"invalid transformer input at step {i}")
            latents = torch.from_numpy(source).to("cuda", dtype=torch.bfloat16)
            sigma = torch.from_numpy(timestep).to("cuda", dtype=torch.bfloat16)
            prediction = model(hidden_states=latents, encoder_hidden_states=embeds,
                               timestep=sigma, img_shapes=shapes, img_mask=img_mask,
                               encoder_hidden_states_mask=enc_mask, return_dict=False)[0]
            np.save(out / f"pred_{i:03d}.npy",
                    prediction[:, -target_tokens:].float().cpu().numpy())
            print(f"saved pred_{i:03d}.npy", flush=True)
    torch.cuda.synchronize()
    record = {"model": str(args.model.resolve()), "capture": str(capture),
              "torch": torch.__version__, "sdpa_backend": args.sdpa_backend,
              "device": torch.cuda.get_device_name(0), "steps": steps,
              "elapsed_seconds": time.perf_counter() - start}
    (out / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
