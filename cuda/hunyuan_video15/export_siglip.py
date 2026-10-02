"""Export Google's vision tensors and reference layout without any FLUX component."""
import argparse
import json
import os
from pathlib import Path
import shutil
import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file
SOURCE = "google/siglip-so400m-patch14-384"
REVISION = "9fdffc58afc957d1a03a25b10dba0329ab15c2a3"

def export(directory):
    directory = Path(directory)
    if (directory / "model.safetensors.aria2").exists():
        raise RuntimeError("Google checkpoint download is incomplete")
    vision = directory / "vision_fp16.safetensors"
    if not vision.exists():
        tensors = {}
        with safe_open(str(directory / "model.safetensors"), framework="numpy") as source:
            for key in source.keys():
                if key.startswith("vision_model."):
                    tensors[key] = source.get_tensor(key).astype(np.float16)
        partial = directory / "vision_fp16.safetensors.partial"
        save_file(tensors, str(partial), metadata={"source": SOURCE, "revision": REVISION})
        partial.replace(vision)
    reference = directory / "reference_vision"
    (reference / "image_encoder").mkdir(parents=True, exist_ok=True)
    (reference / "feature_extractor").mkdir(parents=True, exist_ok=True)
    config = json.loads((directory / "config.json").read_text())["vision_config"]
    config["architectures"] = ["SiglipVisionModel"]
    (reference / "image_encoder/config.json").write_text(json.dumps(config, indent=2) + "\n")
    target = reference / "image_encoder/model.safetensors"
    if not target.exists():
        os.link(vision, target)
    shutil.copyfile(directory / "preprocessor_config.json", reference / "feature_extractor/preprocessor_config.json")
    return vision

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("directory", type=Path)
    args = ap.parse_args()
    print(export(args.directory))
