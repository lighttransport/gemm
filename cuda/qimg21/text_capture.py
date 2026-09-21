"""Capture exact Qwen3-VL boundaries for a future streamed native encoder."""
from contextlib import contextmanager
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
def capture_text_encoder(pipe, folder: Path):
    """One encode_prompt invocation, without changing any module outputs.

    Capture the input to final RMSNorm, not its output: Qwen-Image 2.1 uses
    the last residual stream before normalization. Files retain the batch
    and padding dimensions, unlike the cropped prompt_embeds.npy fixture.
    """
    folder.mkdir(parents=True, exist_ok=True)
    encoder = pipe.text_encoder
    model = getattr(encoder.model, "language_model", encoder.model)
    metadata = {"drop_idx": int(pipe._drop_idx), "hidden_boundary": "before_final_rmsnorm",
                "calls": 0, "norm_calls": 0, "inputs": {}}
    handles = []

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
        for index, layer in enumerate(getattr(model, "layers", ())):
            def layer_hook(module, args, output, index=index):
                value = output[0] if isinstance(output, tuple) else output
                _save(folder / f"layer_{index:02d}.npy", value)
            handles.append(layer.register_forward_hook(layer_hook))
            if index == 0:
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
        yield
        if metadata["calls"] != 1 or metadata["norm_calls"] != 1:
            raise RuntimeError("text encoder did not execute the expected fixture boundaries")
        (folder / "capture.json").write_text(json.dumps(metadata, indent=2) + "\n")
    finally:
        for handle in handles:
            handle.remove()
