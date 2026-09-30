"""Fetch checksum-pinned public assets used by the native ROCm workflow."""
from pathlib import Path
import hashlib
import json
import urllib.request


def main():
    from server.vhuman.rig import face_models
    import torch
    from safetensors.torch import save_file

    root = Path(__file__).resolve().parents[2]
    print("GNM", face_models._gnm_path(face_models.MODEL_CACHE), flush=True)
    print("ICT", face_models._ict_path(face_models.MODEL_CACHE), flush=True)
    sources = json.loads((root / "ref/pixal3d/sources.json").read_text())
    path = root / "ref/pixal3d/weights/naf_release.pth"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        partial = path.with_suffix(".partial")
        with urllib.request.urlopen(sources["naf_weights"], timeout=60) as src, partial.open("wb") as dst:
            while data := src.read(1024 * 1024):
                dst.write(data)
        with partial.open("rb") as src:
            digest = hashlib.file_digest(src, "sha256").hexdigest()
        if digest != sources["naf_checkpoint_sha256"]:
            raise ValueError(f"NAF checkpoint checksum mismatch: {digest}")
        partial.replace(path)
    with path.open("rb") as src:
        digest = hashlib.file_digest(src, "sha256").hexdigest()
    if digest != sources["naf_checkpoint_sha256"]:
        raise ValueError(f"NAF checkpoint checksum mismatch: {digest}")
    weights = torch.load(path, map_location="cpu", weights_only=True)
    save_file({key: value.contiguous() for key, value in weights.items()}, str(path.with_suffix(".safetensors")))
    print("NAF", path.with_suffix(".safetensors"), flush=True)


if __name__ == "__main__":
    main()
