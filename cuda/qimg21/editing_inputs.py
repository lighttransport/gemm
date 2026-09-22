"""CPU image preprocessing and checked single-condition native layout creation."""
import json
from pathlib import Path

import numpy as np


def prepare_image(path, directory, resolution):
    from PIL import Image
    from diffusers.image_processor import VaeImageProcessor
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_dimensions

    if resolution < 32 or resolution > 1024:
        raise ValueError("condition resolution must be within 32..1024")
    with Image.open(path) as source:
        image = source.convert("RGBA")
    width, height, _ = calculate_dimensions(resolution * resolution, image.width / image.height)
    if min(height, width) < 32 or max(height, width) > 1024:
        raise ValueError("resized condition exceeds native encoder limits; use a smaller condition resolution")
    processor = VaeImageProcessor(vae_scale_factor=16, vae_latent_channels=64)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    processor.resize(image, width=width, height=height).save(directory / "resized.png")
    pixels = processor.preprocess(image, width=width, height=height)[0].numpy()
    if pixels.shape != (4, height, width) or not np.isfinite(pixels).all():
        raise ValueError("invalid preprocessed RGBA image")
    np.save(directory / "image.npy", np.ascontiguousarray(pixels, dtype=np.float32))
    (directory / "image.json").write_text(json.dumps({"source": str(Path(path).resolve()),
        "width": width, "height": height, "vae_compute": "native-f32", "end_to_end_parity_validated": False}, indent=2)+"\n")
    return height // 16, width // 16


def write_layout(prompt_directory, output, condition_hw, target_hw, negative=False):
    directory = Path(prompt_directory)
    prefix = "negative_" if negative else ""
    prompt = np.load(directory / f"{prefix}prompt_embeds.npy", allow_pickle=False)
    mask = np.load(directory / f"{prefix}image_pad_mask.npy", allow_pickle=False)
    keys = np.load(directory / f"{prefix}prompt_mask.npy", allow_pickle=False)
    if (prompt.dtype != np.float32 or prompt.ndim not in (2, 3) or
        prompt.shape[-1] != 4096 or (prompt.ndim == 3 and prompt.shape[0] != 1) or
        not np.isfinite(prompt).all()):
        raise ValueError("editing requires finite batch-one prompt embeddings")
    nt = prompt.shape[-2]
    if mask.shape != (1, nt) or keys.shape != (1, nt) or not (keys == 1).all() or not np.isin(mask, [0, 1]).all():
        raise ValueError("editing requires an unpadded binary image/text mask")
    ch, cw = condition_hw
    th, tw = target_hw
    if min(ch, cw, th, tw) < 1 or max(ch,cw,th,tw)>1024 or ch*cw+th*tw>1048576 or (ch*cw)%4 or (th*tw)%4:
        raise ValueError("invalid image token geometry")
    marked = np.flatnonzero(mask[0])
    if len(marked)*4 != ch*cw or not len(marked) or not np.all(np.diff(marked) == 1):
        raise ValueError("vision slots do not match one contiguous condition image")
    target_slots = th*tw//4
    values = [nt+target_slots, nt, 2, *mask[0].astype(int).tolist(), *([1]*target_slots), ch,cw,th,tw]
    Path(output).write_text(" ".join(map(str, values))+"\n")
