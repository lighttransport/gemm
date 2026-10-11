"""Explicit local registration of research assets; job execution never downloads."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import subprocess
from pathlib import Path

UPSTREAM_REVISION = "47b653db2bce25de834360d2ee6b2dd88142bd78"
GMR_REVISION = "1b4eb9864360fa822b1290f34592da6393a4d496"
PACKAGES = ("numpy", "scipy", "jax", "jaxlib", "mujoco", "mujoco-mjx", "warp-lang", "flax",
            "optax", "orbax-checkpoint", "distrax", "musclemimic-models", "mink", "qpsolvers", "daqp")


def supported_packages():
    return dict(line.split("==", 1) for line in
                Path(__file__).with_name("musclemimic-constraints.txt").read_text().splitlines()
                if line and not line.startswith("#"))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tree_hashes(root):
    root = Path(root)
    return {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob("*"))
            if p.is_file() and not any(s.startswith(".") for s in p.relative_to(root).parts)}


def revision(path):
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def read_installation(root, *, verify=False):
    if root is None:
        raise ValueError("Configure --musclemimic-assets and register local research assets first")
    root = Path(root).resolve()
    data = json.loads((root / "installation.json").read_text())
    if data.get("version") != 1 or data.get("usage") != "noncommercial-research":
        raise ValueError("unsupported MuscleMimic installation receipt")
    if data.get("upstream_revision") != UPSTREAM_REVISION or data.get("gmr_revision") != GMR_REVISION:
        raise ValueError("MuscleMimic/GMR revision differs from the supported adapter")
    if data.get("packages") != supported_packages():
        raise ValueError("registration differs from the pinned offline package constraints")
    if verify:
        for name, expected in (("upstream", UPSTREAM_REVISION), ("gmr", GMR_REVISION)):
            if revision(data[name]) != expected:
                raise ValueError(f"{name} source revision changed")
            changes = subprocess.check_output(["git", "-C", data[name], "status", "--porcelain", "--untracked-files=no"], text=True)
            if changes:
                raise ValueError(f"{name} has modified tracked source")
        for name, expected in data["packages"].items():
            if importlib.metadata.version(name) != expected:
                raise ValueError(f"offline package version changed: {name}")
        if tree_hashes(data["checkpoint"]) != data["checkpoint_files"]:
            raise ValueError("checkpoint files differ from their registration hashes")
    return data


def reference(data, identifier):
    if not isinstance(identifier, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", identifier):
        raise ValueError("invalid reference motion ID")
    item = data["references"].get(identifier)
    if not item:
        raise ValueError("reference motion is not registered")
    if sha256(item["path"]) != item["sha256"]:
        raise ValueError("reference trajectory checksum mismatch")
    return item


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    register = sub.add_parser("register")
    register.add_argument("--assets", type=Path, required=True)
    register.add_argument("--upstream", type=Path, required=True)
    register.add_argument("--gmr", type=Path, required=True)
    register.add_argument("--checkpoint", type=Path, required=True)
    register.add_argument("--checkpoint-revision", required=True)
    register.add_argument("--accept-research-terms", action="store_true", required=True)
    add = sub.add_parser("add-reference")
    add.add_argument("--assets", type=Path, required=True)
    add.add_argument("--id", required=True)
    add.add_argument("--trajectory", type=Path, required=True)
    add.add_argument("--source-url", required=True)
    add.add_argument("--terms-url", required=True)
    add.add_argument("--accept-research-terms", action="store_true", required=True)
    doctor = sub.add_parser("doctor")
    doctor.add_argument("--assets", type=Path, required=True)
    args = ap.parse_args()
    if args.command == "register":
        if revision(args.upstream) != UPSTREAM_REVISION or revision(args.gmr) != GMR_REVISION:
            ap.error("check out the documented supported source revisions")
        files = tree_hashes(args.checkpoint)
        if not files or not (args.checkpoint / "_CHECKPOINT_METADATA").is_file():
            ap.error("checkpoint must be a complete local Orbax checkpoint")
        args.assets.mkdir(parents=True, exist_ok=True)
        output = args.assets / "installation.json"
        if output.exists():
            ap.error("installation already registered; use a new assets directory")
        data = {"version": 1, "usage": "noncommercial-research", "upstream": str(args.upstream.resolve()),
                "upstream_revision": UPSTREAM_REVISION, "gmr": str(args.gmr.resolve()),
                "gmr_revision": GMR_REVISION, "checkpoint": str(args.checkpoint.resolve()),
                "checkpoint_revision": args.checkpoint_revision, "checkpoint_files": files,
                "packages": {name: importlib.metadata.version(name) for name in PACKAGES}, "references": {}}
        if data["packages"] != supported_packages():
            ap.error("install the documented musclemimic-constraints.txt before registering")
    else:
        data = read_installation(args.assets, verify=True)
        output = args.assets / "installation.json"
        if args.command == "doctor":
            print(json.dumps({"verified": True, "checkpoint_revision": data["checkpoint_revision"],
                              "references": list(data["references"]), "packages": data["packages"]}, indent=2))
            return
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", args.id) or args.id in data["references"]:
            ap.error("reference ID must be new and contain only letters, digits, _ or -")
        data["references"][args.id] = {"path": str(args.trajectory.resolve()), "sha256": sha256(args.trajectory),
                                      "source_url": args.source_url, "terms_url": args.terms_url}
    partial = output.with_suffix(".partial")
    partial.write_text(json.dumps(data, indent=2) + "\n")
    partial.replace(output)
    print(output)


if __name__ == "__main__":
    main()
