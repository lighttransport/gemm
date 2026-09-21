#!/usr/bin/env python3
"""Capture one official PyTorch block from an existing editing fixture."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--reference-dir", required=True, type=Path)
    ap.add_argument("--step", required=True, type=int)
    ap.add_argument("--block", required=True, type=int, choices=range(32))
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--all-block-targets", action="store_true")
    args = ap.parse_args()
    if args.step < 0:
        raise ValueError("step must be nonnegative")
    args.out_dir.mkdir(parents=True, exist_ok=False)

    import torch
    from diffusers import QwenImage21Pipeline
    import diffusers.models.transformers.transformer_qwenimage21 as qmod

    ref = args.reference_dir
    layout = json.loads((ref / "positive_layout.json").read_text())
    hidden = np.load(ref / f"input_{args.step:03d}.npy", allow_pickle=False)
    prompt = np.load(ref / "prompt_embeds.npy", allow_pickle=False)
    timestep = np.load(ref / f"timestep_{args.step:03d}.npy", allow_pickle=False)
    image_mask = np.load(ref / "positive_img_mask.npy", allow_pickle=False)
    key_mask = np.load(ref / "positive_encoder_hidden_states_mask.npy", allow_pickle=False)
    values = (hidden, prompt, timestep)
    if any(v.dtype.kind != "f" or not np.isfinite(v).all() for v in values):
        raise ValueError("nonfinite or non-floating reference tensors")
    if image_mask.dtype != np.bool_ or key_mask.dtype != np.bool_:
        raise ValueError("reference masks must be boolean")

    pipe = QwenImage21Pipeline.from_pretrained(str(args.model.resolve()), dtype=torch.bfloat16,
                                               local_files_only=True)
    pipe.enable_sequential_cpu_offload(device="cuda")
    transformer = pipe.transformer
    block = transformer.transformer_blocks[args.block]
    target_tokens = int(np.prod(layout["img_shapes"][0][-1]))
    captured = {}

    def save(name):
        def hook(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            captured[name] = value.detach()
        return hook

    def save_input(name):
        def hook(_module, inputs):
            captured[name] = inputs[0].detach()
        return hook

    block.img_norm1.register_forward_pre_hook(save_input("hidden"))
    block.img_norm1.register_forward_hook(save("norm1"))
    block.img_norm2.register_forward_pre_hook(save_input("post_attn_hidden"))
    block.img_norm2.register_forward_hook(save("norm2"))
    block.attn.to_out[0].register_forward_pre_hook(save_input("attn_raw"))
    block.attn.to_q.register_forward_hook(save("q_linear"))
    block.attn.to_k.register_forward_hook(save("k_linear"))
    block.attn.to_v.register_forward_hook(save("v_linear"))
    block.attn.norm_q.register_forward_hook(save("q_norm"))
    block.attn.norm_k.register_forward_hook(save("k_norm"))
    block.img_mlp.register_forward_pre_hook(save_input("mlp_input"))
    for name, module in (("attn_out", block.attn), ("mlp_gate", block.img_mlp.gate_layer),
                         ("mlp_proj", block.img_mlp.proj), ("mlp_out", block.img_mlp.out),
                         (f"block_{args.block:02d}", block)):
        module.register_forward_hook(save(name))

    original_modulate = block._modulate
    modulation_calls = [0]
    def capture_modulate(hidden_states, modulation, target_token_mask):
        modulated, gate = original_modulate(hidden_states, modulation, target_token_mask)
        suffix = modulation_calls[0] + 1
        captured[f"modulated{suffix}"] = modulated.detach()
        captured[f"gate{suffix}"] = gate.detach()
        modulation_calls[0] += 1
        return modulated, gate
    block._modulate = capture_modulate
    if args.all_block_targets:
        for index, selected_block in enumerate(transformer.transformer_blocks):
            def save_target(_module, _inputs, output, index=index):
                value = output[0] if isinstance(output, tuple) else output
                captured[f"block_{index:02d}_target"] = value[:, -target_tokens:].detach()
            selected_block.register_forward_hook(save_target)
        transformer.norm_out.linear.register_forward_hook(save("final_scale"))
        transformer.norm_out.norm.register_forward_hook(save("final_norm_base"))
        transformer.norm_out.register_forward_pre_hook(save_input("final_hidden"))
        transformer.norm_out.register_forward_hook(save("final_ln"))
        transformer.proj_out.register_forward_hook(save("out"))

    original_prepare = qmod._qwenimage21_prepare_qkv
    calls = [0]
    def capture_qkv(*positional, **keywords):
        output = original_prepare(*positional, **keywords)
        if calls[0] == args.block:
            rotary = positional[2] if len(positional) > 2 else keywords.get("rotary_emb")
            if rotary is not None:
                rotary_real = torch.view_as_real(rotary).flatten(-2)
                captured["rope_table"] = rotary_real.detach()
            for name, value in zip(("rope_q", "rope_k", "v"), output[:3]):
                value = value.flatten(2)
                captured[name] = value.detach().flatten(2)[0]
        calls[0] += 1
        return output
    qmod._qwenimage21_prepare_qkv = capture_qkv
    device = torch.device("cuda")
    with torch.no_grad(), transformer.cache_context("cond"):
        output = transformer(
            hidden_states=torch.from_numpy(hidden).to(device, torch.bfloat16),
            encoder_hidden_states=torch.from_numpy(prompt).to(device, torch.bfloat16),
            timestep=torch.from_numpy(timestep).to(device, torch.bfloat16),
            img_shapes=layout["img_shapes"],
            img_mask=torch.from_numpy(image_mask).to(device),
            encoder_hidden_states_mask=torch.from_numpy(key_mask).to(device),
            attention_kwargs={},
            return_dict=False,
        )[0]
    torch.cuda.synchronize()
    np.save(args.out_dir / "output.npy", output.detach().float().cpu().numpy())
    for name, value in captured.items():
        np.save(args.out_dir / f"{name}.npy", value.float().cpu().numpy())
    if calls[0] != 32 or modulation_calls[0] != 2 or not torch.isfinite(output).all():
        raise RuntimeError("incomplete/nonfinite transformer capture")
    metadata = {"diagnostic_only": True, "full_model_acceptance": False,
                "step": args.step, "block": args.block, "torch": torch.__version__,
                "output_finite": True}
    (args.out_dir / "capture.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
