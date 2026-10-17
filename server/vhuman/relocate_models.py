"""Move reusable tmp/ weights to models storage, retaining compatibility links.

Run with --apply to move; the default only lists the proposed operations.
Generated avatars, per-subject deformers, environments and run evidence stay put.
"""
import argparse
import json
import os
from pathlib import Path

from server.model_paths import MODELS_ROOT

ROOT = Path(__file__).resolve().parents[2]
RELOCATIONS = (
    ("tmp/vhuman-musclemimic-checkpoint", "vhuman/mm-fullbody-base"),
    ("tmp/terra-policy/TERRA-4B", "vhuman/TERRA-4B"),
    ("tmp/genia-native-models", "genia/native"),
    ("tmp/vhuman-realtime/motion-v2.pt", "vhuman/realtime/motion-v2.pt"),
    ("tmp/vhuman-realtime/motion-v3.pt", "vhuman/realtime/motion-v3.pt"),
    ("tmp/vhuman-realtime/motion-v3.native", "vhuman/realtime/motion-v3.native"),
    ("tmp/vhuman-native-renderer/flux2-tokenizer.gguf", "vhuman/realtime/flux2-tokenizer.gguf"),
    ("tmp/vhuman-native-renderer/flux2-tokenizer.json", "vhuman/realtime/flux2-tokenizer.json"),
    ("tmp/vhuman-emotion/sensevoice-small-q8.gguf", "vhuman/sensevoice-small-q8.gguf"),
    ("tmp/vhuman-rig/models", "vhuman-rig-models"),
)
SHARED_MODELS = (
    "Pixal3D", "qimg-21", "qimg-21-fast", "qimg21-int8-package", "qwen-image-edit-2511",
    "dinov3-vitl16", "RMBG-2.0", "moge-2-vitl", "sam3", "sam3d", "sam3d-body", "clip-bpe",
    "speech", "hunyuan-video15-model", "h3", "BiRefNet", "klein2-4b", "da2-native",
    "vhuman-native-moge-small", "vhuman-native-moge-native", "vhuman-native-gpu-moge",
)


def link_shared_models(shared, models=MODELS_ROOT, *, apply=False):
    """Expose installed large weights without duplicating them across volumes."""
    shared, models = Path(shared), Path(models)
    if not shared.is_dir():
        raise ValueError("Shared models directory does not exist: " + str(shared))
    operations = []
    for name in SHARED_MODELS:
        source, target = shared / name, models / name
        if not source.exists() or target.exists() or target.is_symlink():
            continue
        operations.append({"operation": "alias", "source": str(source), "target": str(target)})
    if apply:
        for row in operations:
            target = Path(row["target"])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(os.path.relpath(Path(row["source"]).resolve(), target.parent))
    return operations


def relocate_models(repo=ROOT, models=MODELS_ROOT, *, apply=False):
    """Rename on the same filesystem; preserve existing installations and aliases."""
    repo, models = Path(repo), Path(models)
    operations = []
    for relative, destination in RELOCATIONS:
        source, target = repo / relative, models / destination
        if not source.exists() and not source.is_symlink():
            continue
        if source.is_symlink() and not source.exists():
            raise ValueError("Broken source model link: " + str(source))
        if target.exists() or target.is_symlink():
            if source.resolve() == target.resolve():
                continue
            raise ValueError("Refusing to replace existing model: " + str(target))
        if target.is_relative_to(source) or source.is_relative_to(target):
            raise ValueError("Model source and destination overlap")
        # Source aliases already point at weights outside tmp/. Add a canonical
        # alias without copying the shared bundle onto another device.
        operation = "alias" if source.is_symlink() else "move"
        if operation == "move":
            parent = target.parent
            while not parent.exists():
                parent = parent.parent
            if source.stat().st_dev != parent.stat().st_dev:
                raise ValueError("Cross-device model move requires explicit staging: " + str(source))
        operations.append({"operation": operation, "source": str(source), "target": str(target)})
    if apply:
        for row in operations:
            source, target = Path(row["source"]), Path(row["target"])
            target.parent.mkdir(parents=True, exist_ok=True)
            if row["operation"] == "alias":
                target.symlink_to(os.path.relpath(source.resolve(), target.parent))
            else:
                # os.rename intentionally refuses a cross-device copy: a large
                # checkpoint must never be partially duplicated by this helper.
                source.rename(target)
                source.symlink_to(os.path.relpath(target, source.parent))
    return operations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-root", type=Path, default=MODELS_ROOT)
    parser.add_argument("--shared-models-root", type=Path,
                        help="alias already-installed large models from another volume")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    operations = relocate_models(models=args.models_root, apply=args.apply)
    if args.shared_models_root:
        operations += link_shared_models(args.shared_models_root, args.models_root, apply=args.apply)
    print(json.dumps(operations, indent=2))


if __name__ == "__main__":
    main()
