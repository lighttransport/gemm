"""Checksum-pinned GNM, face parsing and native MediaPipe observation assets."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[2]
DATA = Path("/mnt/disk01/data/vhuman")
LEGACY = ROOT / "tmp/vhuman-rig/models"
PARSING_SHA256 = "0d9bd318e46987c3bdbfacae9e2c0f461cae1c6ac6ea6d43bbe541a91727e33f"
PARSING_REVISION = "bfcc7d48ea8412d55de406cecf96b65b3ca720b2"


def asset_path(name, root=DATA):
    relative = {"gnm": "gnm-v3/gnm_head.npz", "mediapipe": "mediapipe/face_landmarker.task",
                "parsing": "face-parsing/resnet18.onnx"}[name]
    path = Path(root) / relative
    if path.is_file() or name == "parsing" or Path(root) != DATA:
        return path
    return LEGACY / ("face_landmarker.task" if name == "mediapipe" else relative)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fetch(root=DATA):
    from .rig.face_models import GNM_SHA256, GNM_REVISION
    from .landmark_assets import TASK_SHA256, MODELS, export
    root = Path(root)
    specs = [
        ("gnm-v3/gnm_head.npz", GNM_SHA256,
         f"https://huggingface.co/google/gnm-v3/resolve/{GNM_REVISION}/v3_0/gnm_head.npz",
         GNM_REVISION, "Apache-2.0", LEGACY / "gnm-v3/gnm_head.npz"),
        ("mediapipe/face_landmarker.task", TASK_SHA256,
         "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
         "float16/1", "Apache-2.0", LEGACY / "face_landmarker.task"),
        ("face-parsing/resnet18.onnx", PARSING_SHA256,
         f"https://huggingface.co/yakhyo/uniface-weights/resolve/{PARSING_REVISION}/parsing_resnet18.onnx",
         PARSING_REVISION, "MIT (upstream model publisher)", None),
    ]
    receipt = {"schema": "vhuman.face_assets.v1", "root": str(root.resolve()), "files": {}}
    for relative, digest, url, revision, license_name, cached in specs:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_file() or sha256(target) != digest:
            partial = target.with_name(target.name + "." + uuid.uuid4().hex + ".partial")
            try:
                if cached and cached.is_file() and sha256(cached) == digest:
                    with cached.open("rb") as source, partial.open("wb") as destination:
                        shutil.copyfileobj(source, destination, 1 << 20)
                else:
                    with urllib.request.urlopen(url, timeout=120) as source, partial.open("wb") as destination:
                        shutil.copyfileobj(source, destination, 1 << 20)
                if sha256(partial) != digest:
                    raise ValueError(f"Face asset checksum mismatch: {relative}")
                partial.replace(target)
            finally:
                partial.unlink(missing_ok=True)
        receipt["files"][relative] = {"sha256": digest, "bytes": target.stat().st_size,
                                       "source": url, "revision": revision, "license": license_name}
        print(f"Verified {target}", flush=True)
    task = root / "mediapipe/face_landmarker.task"
    with zipfile.ZipFile(task) as archive:
        for name in MODELS:
            data = archive.read(name + ".tflite")
            target = task.parent / (name + ".tflite")
            target.write_bytes(data)
            receipt["files"][str(target.relative_to(root))] = {
                "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "source": "verified task archive"}
    receipt["native_mediapipe"] = export(task, task.parent / "native-face-v1")
    temporary = root / "download.json.partial"
    temporary.write_text(json.dumps(receipt, indent=2) + "\n")
    temporary.replace(root / "download.json")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DATA)
    args = parser.parse_args()
    print(json.dumps(fetch(args.root), indent=2))


if __name__ == "__main__":
    main()
