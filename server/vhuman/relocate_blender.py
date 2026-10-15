"""Rebind external files in a fresh copy of a saved Blender scene; no rendering.

blender -b --factory-startup --python-exit-code 1 --python THIS_FILE -- \
    --input head.blend --out local/head.blend --map OLD_ROOT=NEW_ROOT
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    import bpy
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--map", action="append", default=[])
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    source, target = args.input.resolve(), args.out.resolve()
    if source == target or target.exists():
        parser.error("select a fresh output .blend; source scenes are preserved")
    mappings = sorted((entry.split("=", 1) for entry in args.map), key=lambda pair: -len(pair[0]))
    bpy.ops.wm.open_mainfile(filepath=str(source))
    report = dict(source=str(source), source_sha256=sha256(source), target=str(target),
                  blender=bpy.app.version_string, rebound=[], missing=[])
    for collection in (bpy.data.images, bpy.data.sounds, bpy.data.fonts, bpy.data.movieclips,
                       bpy.data.cache_files, bpy.data.libraries, bpy.data.volumes):
        for item in collection:
            path = getattr(item, "filepath", "")
            if not path or path.startswith("<") or getattr(item, "packed_file", None) or getattr(item, "packed_files", None):
                continue
            if getattr(item, "source", "") in ("GENERATED", "VIEWER"):
                continue
            absolute = bpy.path.abspath(path)
            rebound = absolute
            for old, new in mappings:
                if absolute == old or absolute.startswith(old.rstrip("/") + "/"):
                    rebound = new.rstrip("/") + absolute[len(old):]
                    break
            if not Path(rebound).is_file():
                report["missing"].append(dict(name=item.name, source=absolute, mapped=rebound))
            elif path != rebound:
                item.filepath = rebound
                report["rebound"].append(dict(name=item.name, source=absolute, mapped=rebound))
    target.parent.mkdir(parents=True, exist_ok=True)
    receipt = target.with_suffix(".relocation.json")
    receipt.write_text(json.dumps(report, indent=2) + "\n")
    if report["missing"]:
        raise RuntimeError("Scene has missing external files; inspect " + str(receipt))
    for scene in bpy.data.scenes:
        if hasattr(scene, "cycles"):
            scene.cycles.device = "CPU"
    bpy.ops.wm.save_as_mainfile(filepath=str(target), relative_remap=True)
    report["target_sha256"] = sha256(target)
    receipt.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
