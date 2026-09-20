"""Run the pinned RMBG-2.0 path on an opaque Pixal3D reference image."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image


HERE = Path(__file__).resolve().parent
p = argparse.ArgumentParser()
p.add_argument("--model", type=Path, default=Path("/mnt/disk2/models/RMBG-2.0"))
p.add_argument("--input", type=Path, default=HERE / "upstream/assets/images/1_img.png")
p.add_argument("--output-dir", type=Path, default=HERE.parents[1] / "tmp/pixal3d/rmbg-validation")
p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
p.add_argument("--minimum-source-iou", type=float, default=.99,
               help="Required IoU against the input alpha reference")
a = p.parse_args()
if not 0 <= a.minimum_source_iou <= 1:
    p.error("--minimum-source-iou must be in [0, 1]")

a.output_dir.mkdir(parents=True, exist_ok=True)
source = Image.open(a.input).convert("RGBA")
expected = np.asarray(source)[:, :, 3]
if expected.min() == expected.max():
    p.error("--input must contain a nontrivial reference alpha channel")
rgb_path = a.output_dir / "input-rgb.png"
source.convert("RGB").save(rgb_path)
output = a.output_dir / "prepared-rgba.png"
metadata_path = a.output_dir / "metadata.json"
subprocess.run([
    sys.executable, str(HERE / "prepare_input.py"),
    "--input", str(rgb_path), "--output", str(output),
    "--metadata", str(metadata_path), "--rembg-model", str(a.model),
    "--fov", "0.857556", "--device", a.device,
], check=True)

actual = np.asarray(Image.open(output).convert("RGBA"))[:, :, 3]
foreground = actual > 127
expected_foreground = expected > 127
union = np.logical_or(foreground, expected_foreground).sum()
result = {
    "device": a.device,
    "model": str(a.model),
    "output": str(output),
    "alpha_min": int(actual.min()),
    "alpha_max": int(actual.max()),
    "alpha_mean": float(actual.mean()),
    "foreground_pixels": int(foreground.sum()),
    "background_pixels": int((~foreground).sum()),
    "source_alpha_iou": float(np.logical_and(foreground, expected_foreground).sum() / max(1, union)),
    "metadata": json.loads(metadata_path.read_text()),
}
assert result["metadata"]["mask_source"] == "rmbg-2.0"
assert result["alpha_min"] == 0 and result["alpha_max"] == 255
assert result["foreground_pixels"] and result["background_pixels"]
assert result["source_alpha_iou"] >= a.minimum_source_iou
print(json.dumps(result, indent=2))
print("Pixal3D RMBG-2.0: PASS")
