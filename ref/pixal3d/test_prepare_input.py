"""Model-free checks for Pixal3D automatic input preparation."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
scratch = ROOT / "tmp/pixal3d/prepare-test"
scratch.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory(prefix="case-", dir=scratch) as td:
    directory = Path(td)
    pixels = np.zeros((40, 60, 4), dtype=np.uint8)
    pixels[8:32, 15:45, :3] = (10, 120, 240)
    pixels[8:32, 15:45, 3] = 255
    source = directory / "input.png"
    output = directory / "prepared.png"
    metadata = directory / "camera.json"
    Image.fromarray(pixels).save(source)
    subprocess.run([sys.executable, str(Path(__file__).with_name("prepare_input.py")),
                    "--input", str(source), "--output", str(output),
                    "--metadata", str(metadata), "--fov", "0.8",
                    "--mesh-scale", "1.25", "--device", "cpu"], check=True)
    actual = json.loads(metadata.read_text())
    assert actual["mask_source"] == "alpha" and actual["camera_source"] == "manual"
    assert actual["width"] == 60 and actual["height"] == 40
    assert abs(actual["fov"] - .8) < 1e-12
    assert output.is_file() and Image.open(output).mode == "RGBA"

    rgb = directory / "rgb.png"
    Image.fromarray(pixels[:, :, :3]).save(rgb)
    failed = subprocess.run([sys.executable, str(Path(__file__).with_name("prepare_input.py")),
                             "--input", str(rgb), "--output", str(output),
                             "--metadata", str(metadata), "--fov", "0.8", "--device", "cpu"],
                            capture_output=True, text=True)
    assert failed.returncode != 0 and "provide --rembg-model" in failed.stderr

    mask = directory / "mask.png"
    Image.fromarray(pixels[:, :, 3]).save(mask)
    subprocess.run([sys.executable, str(Path(__file__).with_name("prepare_input.py")),
                    "--input", str(rgb), "--mask", str(mask), "--output", str(output),
                    "--metadata", str(metadata), "--fov", "0.8", "--device", "cpu"], check=True)
    assert json.loads(metadata.read_text())["mask_source"] == "mask"
    assert np.array_equal(np.asarray(Image.open(output).getchannel("A")), pixels[:, :, 3])

print("Pixal3D input preparation: PASS")
