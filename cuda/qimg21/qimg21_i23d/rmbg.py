"""RMBG-2.0 (BiRefNet) matting, as ref/pixal3d/prepare_input.py runs it.

Deterministic background removal that never regenerates a pixel: it only
predicts alpha. It runs in this process when its dependencies (timm,
torchvision, transformers) are importable, and otherwise in the Pixal3D
reference environment that already runs it for Pixal3D
(ref/pixal3d/.venv-reference-cuda310, or $QIMG21_RMBG_PYTHON).
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from . import imageops

ROOT = Path(__file__).resolve().parents[3]
MODEL = Path(os.environ.get("QIMG21_RMBG_MODEL", "/mnt/nvme01/models/RMBG-2.0"))
PYTHON = Path(os.environ.get("QIMG21_RMBG_PYTHON", ROOT / "ref/pixal3d/.venv-reference-cuda310/bin/python"))
SCRIPT = ROOT / "ref/pixal3d/prepare_input.py"


class MattingError(RuntimeError):
    pass


def _in_process() -> bool:
    try:
        import timm  # noqa: F401
        import torchvision  # noqa: F401
        return True
    except ImportError:
        return False


def remove_background(rgba: np.ndarray, device: str = "cuda") -> np.ndarray:
    """Alpha (uint8, H x W) for the main object of an RGBA/RGB image."""
    if not MODEL.is_dir():
        raise MattingError(f"RMBG-2.0 weights not found at {MODEL} (set QIMG21_RMBG_MODEL)")
    with tempfile.TemporaryDirectory(prefix="qimg21-rmbg-", dir=ROOT / "tmp") as td:
        source, target = Path(td) / "in.png", Path(td) / "out.png"
        imageops.save_png(rgba, source)
        # prepare_input.py parses its CLI at import, so its remove_background()
        # is restated here: the same model, transforms and sigmoid output.
        code = f"""
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
image = Image.open({str(source)!r})
model = AutoModelForImageSegmentation.from_pretrained({str(MODEL)!r}, trust_remote_code=True,
                                                      local_files_only=True).eval().to({device!r})
transform = transforms.Compose([transforms.Resize((1024, 1024)), transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
with torch.inference_mode():
    prediction = model(transform(image.convert("RGB")).unsqueeze(0).to({device!r}))[-1].sigmoid().cpu()
mask = transforms.ToPILImage()(prediction[0].squeeze()).resize(image.size)
output = image.convert("RGBA")
output.putalpha(mask)
output.save({str(target)!r})
"""
        python = None if _in_process() else PYTHON
        if python is not None and not python.is_file():
            raise MattingError(f"RMBG-2.0 needs timm/torchvision; none here and no environment at {python} "
                               "(set QIMG21_RMBG_PYTHON)")
        if python is None:
            import sys
            python = Path(sys.executable)
        result = subprocess.run([str(python), "-c", code], cwd=ROOT, capture_output=True, text=True)
        if result.returncode or not target.is_file():
            raise MattingError(f"RMBG-2.0 failed: {result.stderr[-2000:]}")
        return imageops.load_rgba(target)[..., 3].copy()
