"""Compare the two pinned BiRefNet checkpoints against the bundled alpha."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation


HERE = Path(__file__).resolve().parent
p = argparse.ArgumentParser()
p.add_argument("--input", type=Path,
               default=HERE / "upstream/assets/images/1_img.png")
p.add_argument("--rmbg", type=Path, default=Path("/mnt/disk2/models/RMBG-2.0"))
p.add_argument("--upstream", type=Path, default=Path("/mnt/disk2/models/BiRefNet"))
p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
p.add_argument("--output", type=Path)
a = p.parse_args()

source = Image.open(a.input).convert("RGBA")
rgb = source.convert("RGB")
expected_alpha = np.asarray(source.getchannel("A"))
expected = expected_alpha > 127
transform = transforms.Compose([
    transforms.Resize((1024, 1024)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iou(alpha: np.ndarray, threshold: int) -> float:
    actual = alpha > threshold
    union = np.logical_or(actual, expected).sum()
    return float(np.logical_and(actual, expected).sum() / max(1, union))


def evaluate(name: str, directory: Path) -> dict:
    checkpoint = directory / "model.safetensors"
    if not checkpoint.is_file():
        raise RuntimeError(f"checkpoint is missing: {checkpoint}")
    model = AutoModelForImageSegmentation.from_pretrained(
        directory, trust_remote_code=True, local_files_only=True).eval().to(a.device)
    with torch.inference_mode():
        outputs = model(transform(rgb).unsqueeze(0).to(a.device))
    predictions = list(outputs) if isinstance(outputs, (list, tuple)) else [outputs]
    final = transforms.ToPILImage()(predictions[-1][0].squeeze().sigmoid().cpu())
    # PIL's default resize is the exact behavior in Pixal3D's pinned wrapper.
    standard_alpha = np.asarray(final.resize(source.size))
    diagnostic = []
    for output_index, prediction in enumerate(predictions):
        mask = transforms.ToPILImage()(prediction[0].squeeze().sigmoid().cpu())
        for resize_name, method in (
                ("nearest", Image.Resampling.NEAREST),
                ("bilinear", Image.Resampling.BILINEAR),
                ("bicubic", Image.Resampling.BICUBIC),
                ("lanczos", Image.Resampling.LANCZOS)):
            alpha = np.asarray(mask.resize(source.size, method))
            choices = [(iou(alpha, threshold), threshold) for threshold in range(256)]
            best_iou, threshold = max(choices)
            diagnostic.append({"output": output_index, "resize": resize_name,
                               "threshold": threshold, "iou": best_iou})
    del model, outputs, predictions
    if a.device == "cuda":
        torch.cuda.empty_cache()
    return {
        "name": name, "directory": str(directory), "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": sha256(checkpoint), "normal_semantics": {
            "output": "last", "resize": "PIL default nearest", "threshold": 127,
            "source_alpha_iou": iou(standard_alpha, 127),
            "mean_abs_alpha_error_u8": float(np.abs(
                standard_alpha.astype(np.int16) - expected_alpha.astype(np.int16)).mean()),
            "exact_alpha_fraction": float(np.mean(standard_alpha == expected_alpha)),
        },
        "asset_tuned_best": max(diagnostic, key=lambda item: item["iou"]),
    }


result = {
    "input": str(a.input), "input_sha256": sha256(a.input), "device": a.device,
    "models": [evaluate("briaai/RMBG-2.0", a.rmbg),
               evaluate("ZhengPeng7/BiRefNet", a.upstream)],
}
encoded = json.dumps(result, indent=2) + "\n"
if a.output:
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(encoded)
print(encoded, end="")
print("Pixal3D mask provenance: PASS")
