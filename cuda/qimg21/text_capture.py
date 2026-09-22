"""Capture exact Qwen3-VL boundaries for a future streamed native encoder."""
from contextlib import contextmanager
import importlib
import json
from pathlib import Path

import numpy as np


def _save(path, tensor):
    value = tensor.detach().cpu()
    # NumPy cannot represent BF16. Preserve integer IDs/masks without casting.
    if value.is_floating_point():
        value = value.float()
    np.save(path, np.ascontiguousarray(value.numpy()))


@contextmanager
def capture_text_encoder(pipe, folder: Path, stage_layer: int = 0):
    """One encode_prompt invocation, without changing any module outputs.

    Capture the input to final RMSNorm, not its output: Qwen-Image 2.1 uses
    the last residual stream before normalization. Files retain the batch
    and padding dimensions, unlike the cropped prompt_embeds.npy fixture.
    """
    folder.mkdir(parents=True, exist_ok=True)
    encoder = pipe.text_encoder
    model = getattr(encoder.model, "language_model", encoder.model)
    visual = getattr(encoder.model, "visual", None)
    layers = tuple(getattr(model, "layers", ()))
    if not 0 <= stage_layer < len(layers):
        raise ValueError(f"stage_layer must be in [0, {len(layers)})")
    metadata = {"drop_idx": int(pipe._drop_idx), "hidden_boundary": "before_final_rmsnorm",
                "stage_layer": stage_layer,
                "calls": 0, "norm_calls": 0, "inputs": {}}
    handles = []
    vision_module = None
    original_vision_rope = None

    def inputs_hook(module, args, kwargs):
        if metadata["calls"]:
            raise RuntimeError("text fixture capture expects one encoder call per directory")
        metadata["calls"] += 1
        for name in ("input_ids", "attention_mask", "position_ids", "mm_token_type_ids",
                     "pixel_values", "image_grid_thw"):
            value = kwargs.get(name)
            if value is not None:
                _save(folder / f"{name}.npy", value)
                metadata["inputs"][name] = {"shape": list(value.shape), "dtype": str(value.dtype)}

    def norm_hook(module, args):
        metadata["norm_calls"] += 1
        _save(folder / "hidden_prenorm.npy", args[0])

    try:
        handles.append(encoder.register_forward_pre_hook(inputs_hook, with_kwargs=True))
        handles.append(model.norm.register_forward_pre_hook(norm_hook))
        if hasattr(model, "rotary_emb"):
            def rotary_hook(module, args, output):
                cos, sin = output
                table = __import__("torch").stack((cos, sin), dim=-1)
                if table.ndim == 4 and table.shape[0] == 1:
                    table = table[0]
                _save(folder / "rope_table.npy", table)
            handles.append(model.rotary_emb.register_forward_hook(rotary_hook))
        if visual is not None:
            vision_module = importlib.import_module(type(visual).__module__)
            original_vision_rope = vision_module.apply_rotary_pos_emb_vision
            rope_calls = 0

            def capture_vision_rope(q, k, cos, sin):
                nonlocal rope_calls
                q_out, k_out = original_vision_rope(q, k, cos, sin)
                if rope_calls == min(stage_layer, len(visual.blocks) - 1):
                    _save(folder / "vision_stage_attn_q_rope.npy", q_out)
                    _save(folder / "vision_stage_attn_k_rope.npy", k_out)
                rope_calls += 1
                return q_out, k_out

            vision_module.apply_rotary_pos_emb_vision = capture_vision_rope
            def save_visual(name):
                def hook(module, args, output):
                    value = output[0] if isinstance(output, tuple) else output
                    if hasattr(value, "detach"):
                        _save(folder / f"vision_{name}.npy", value)
                return hook
            handles.append(visual.patch_embed.register_forward_hook(save_visual("patch_embed")))
            for index, block in enumerate(visual.blocks):
                if index == 0:
                    def vision_input_hook(module, args):
                        _save(folder / "vision_block_input.npy", args[0])
                    handles.append(block.register_forward_pre_hook(vision_input_hook))
                if index == min(stage_layer, len(visual.blocks) - 1):
                    for name, child in block.named_modules():
                        if not name:
                            continue
                        handles.append(child.register_forward_hook(
                            save_visual("stage_" + name.replace(".", "_"))))
                    def vision_projection_input(module, args):
                        _save(folder / "vision_stage_attn_proj_input.npy", args[0])
                    handles.append(block.attn.proj.register_forward_pre_hook(vision_projection_input))
                    def vision_norm2_input(module, args):
                        _save(folder / "vision_stage_norm2_input.npy", args[0])
                    handles.append(block.norm2.register_forward_pre_hook(vision_norm2_input))
                handles.append(block.register_forward_hook(save_visual(f"block_{index:02d}")))
            handles.append(visual.merger.register_forward_hook(save_visual("merger")))
            for index, merger in enumerate(visual.deepstack_merger_list):
                handles.append(merger.register_forward_hook(save_visual(f"deepstack_{index}")))
        for index, layer in enumerate(layers):
            def layer_input_hook(module, args, index=index):
                _save(folder / f"layer_{index:02d}_input.npy", args[0])
            handles.append(layer.register_forward_pre_hook(layer_input_hook))
            def layer_hook(module, args, output, index=index):
                value = output[0] if isinstance(output, tuple) else output
                _save(folder / f"layer_{index:02d}.npy", value)
            handles.append(layer.register_forward_hook(layer_hook))
            if index == stage_layer:
                for name, child in layer.named_modules():
                    if not name:
                        continue
                    def child_hook(module, args, output, name=name):
                        value = output[0] if isinstance(output, tuple) else output
                        if hasattr(value, "detach"):
                            _save(folder / f"stage_{name}.npy", value)
                    handles.append(child.register_forward_hook(child_hook))
                def projection_input(module, args):
                    _save(folder / "stage_self_attn.o_proj.input.npy", args[0])
                if hasattr(layer, "self_attn"):
                    handles.append(layer.self_attn.o_proj.register_forward_pre_hook(projection_input))
                def post_attention_input(module, args):
                    _save(folder / "stage_post_attention_hidden.npy", args[0])
                if hasattr(layer, "post_attention_layernorm"):
                    handles.append(layer.post_attention_layernorm.register_forward_pre_hook(post_attention_input))
        yield
        if metadata["calls"] != 1 or metadata["norm_calls"] != 1:
            raise RuntimeError("text encoder did not execute the expected fixture boundaries")
        (folder / "capture.json").write_text(json.dumps(metadata, indent=2) + "\n")
    finally:
        if vision_module is not None and original_vision_rope is not None:
            vision_module.apply_rotary_pos_emb_vision = original_vision_rope
        for handle in handles:
            handle.remove()
