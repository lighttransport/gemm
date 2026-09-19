"""Prepare Pixal3D alpha and camera parameters with pinned-reference semantics."""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image


p = argparse.ArgumentParser()
p.add_argument("--input", type=Path, required=True)
p.add_argument("--output", type=Path, required=True, help="RGBA image for native inference")
p.add_argument("--metadata", type=Path, required=True)
p.add_argument("--camera-image", type=Path, help="Optional cropped RGB input used by MoGe")
p.add_argument("--mask", type=Path)
p.add_argument("--rembg-model", type=Path)
p.add_argument("--moge-model", type=Path)
p.add_argument("--fov", type=float, default=-1.0)
p.add_argument("--mesh-scale", type=float, default=1.0)
p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
a = p.parse_args()
if not math.isfinite(a.mesh_scale) or a.mesh_scale <= 0:
    p.error("--mesh-scale must be positive")
if a.fov > 0 and (not math.isfinite(a.fov) or a.fov >= math.pi):
    p.error("--fov must be in (0, pi)")


def useful_alpha(image: Image.Image) -> bool:
    return image.mode == "RGBA" and not np.all(np.asarray(image.getchannel("A")) == 255)


def remove_background(image: Image.Image, model_path: Path, device: str) -> Image.Image:
    weights = list(model_path.glob("*.safetensors")) + list(model_path.glob("pytorch_model*.bin"))
    if not (model_path.is_dir() and (model_path / "config.json").is_file() and weights):
        raise RuntimeError(f"RMBG model directory is missing: {model_path}")
    import torch
    from torchvision import transforms
    from transformers import AutoModelForImageSegmentation
    model = AutoModelForImageSegmentation.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True).eval().to(device)
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    with torch.inference_mode():
        prediction = model(transform(image.convert("RGB")).unsqueeze(0).to(device))[-1].sigmoid().cpu()
    mask = transforms.ToPILImage()(prediction[0].squeeze()).resize(image.size)
    output = image.convert("RGBA")
    output.putalpha(mask)
    return output


def camera_image(image: Image.Image) -> Image.Image:
    """Match upstream preprocess_image for the image passed to MoGe."""
    rgba = np.asarray(image, dtype=np.uint8)
    foreground = np.argwhere(rgba[:, :, 3] > 204)
    if not len(foreground):
        raise RuntimeError("mask contains no foreground with alpha > 0.8")
    x0, x1 = int(foreground[:, 1].min()), int(foreground[:, 1].max())
    y0, y1 = int(foreground[:, 0].min()), int(foreground[:, 0].max())
    center_x, center_y = (x0 + x1) / 2, (y0 + y1) / 2
    size = int(max(x1 - x0, y1 - y0) * 1.1)
    box = (center_x - size // 2, center_y - size // 2,
           center_x + size // 2, center_y + size // 2)
    cropped = np.asarray(image.crop(box), dtype=np.float32) / 255.0
    rgb = cropped[:, :, :3] * cropped[:, :, 3:4]
    return Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))


def estimate_fov(image: Image.Image, model_path: Path, device: str) -> float:
    if not model_path.exists():
        raise RuntimeError(f"MoGe-2 checkpoint is missing: {model_path}")
    source = Path(__file__).with_name("moge-upstream")
    if source.is_dir():
        sys.path.insert(0, str(source))
    try:
        from moge.model.v2 import MoGeModel
    except ImportError as exc:
        raise RuntimeError("MoGe-2 support is not installed in this project environment") from exc
    import torch
    tensor = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1).to(device)
    model = MoGeModel.from_pretrained(model_path).eval().to(device)
    with torch.inference_mode():
        intrinsics = model.infer(tensor)["intrinsics"].squeeze().cpu().numpy()
    fx = float(intrinsics[0, 0]) * image.width
    if not math.isfinite(fx) or fx <= 0:
        raise RuntimeError("MoGe-2 returned invalid camera intrinsics")
    return 2 * math.atan(image.width / (2 * fx))


image = Image.open(a.input)
maximum = max(image.size)
if maximum > 1024:
    scale = 1024 / maximum
    image = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))),
                         Image.Resampling.LANCZOS)
if a.mask is not None:
    mask = Image.open(a.mask).convert("L")
    if mask.size != Image.open(a.input).size:
        raise RuntimeError("mask dimensions must match the input image")
    if maximum > 1024:
        mask = mask.resize(image.size, Image.Resampling.LANCZOS)
    image = image.convert("RGBA")
    image.putalpha(mask)
if useful_alpha(image):
    rgba = image.convert("RGBA")
    mask_source = "mask" if a.mask is not None else "alpha"
else:
    if a.rembg_model is None:
        raise RuntimeError("input has no foreground alpha; provide --rembg-model")
    rgba = remove_background(image, a.rembg_model, a.device)
    mask_source = "rmbg-2.0"

fov = a.fov
camera_source = "manual"
if fov <= 0:
    if a.moge_model is None:
        raise RuntimeError("automatic FOV requires --moge-model")
    prepared_camera = camera_image(rgba)
    if a.camera_image is not None:
        a.camera_image.parent.mkdir(parents=True, exist_ok=True)
        prepared_camera.save(a.camera_image)
    fov = estimate_fov(prepared_camera, a.moge_model, a.device)
    camera_source = "moge-2"
distance = 0.5 / (math.tan(fov * 0.5) * a.mesh_scale)

a.output.parent.mkdir(parents=True, exist_ok=True)
a.metadata.parent.mkdir(parents=True, exist_ok=True)
rgba.save(a.output)
metadata = {"fov": fov, "distance": distance, "mesh_scale": a.mesh_scale,
            "mask_source": mask_source, "camera_source": camera_source,
            "width": rgba.width, "height": rgba.height}
a.metadata.write_text(json.dumps(metadata, indent=2) + "\n")
print(json.dumps(metadata))
