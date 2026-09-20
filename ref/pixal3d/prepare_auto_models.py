"""Download the pinned-reference input-preparation models to a local model root."""
import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


p = argparse.ArgumentParser()
p.add_argument("--model-root", type=Path, default=Path("/mnt/disk2/models"))
p.add_argument("--component", choices=("all", "rmbg", "moge"), default="all")
a = p.parse_args()
a.model_root.mkdir(parents=True, exist_ok=True)
sources = json.loads((Path(__file__).with_name("sources.json")).read_text())

result = {}
if a.component in ("all", "moge"):
    moge = a.model_root / "moge-2-vitl"
    path = Path(hf_hub_download(sources["moge2"]["model"], filename="model.pt",
                                revision=sources["moge2"]["model_revision"], local_dir=moge))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != sources["moge2"]["model_sha256"]:
        raise RuntimeError(f"unexpected MoGe-2 checkpoint SHA256: {digest}")
    result["moge_model"] = str(path)
if a.component in ("all", "rmbg"):
    rmbg = a.model_root / "RMBG-2.0"
    # The repository also contains several large ONNX variants and a duplicate
    # PyTorch .bin checkpoint. Pixal3D's Transformers path needs only the
    # safetensors checkpoint, configuration, and trusted remote-code modules.
    required = ("config.json", "preprocessor_config.json", "model.safetensors",
                "birefnet.py", "BiRefNet_config.py")
    snapshot_download(sources["rmbg"]["model"], revision=sources["rmbg"]["revision"],
                      local_dir=rmbg, allow_patterns=list(required))
    missing = [name for name in required if not (rmbg / name).is_file()]
    if missing:
        raise RuntimeError(f"RMBG-2.0 snapshot is incomplete: {', '.join(missing)}")
    result["rembg_model"] = str(rmbg)
print(json.dumps(result, indent=2))
