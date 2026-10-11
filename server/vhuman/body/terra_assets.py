"""Explicit local registration for the isolated Terra research backend."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import re
import subprocess

from .musclemimic_assets import sha256, tree_hashes

REVISIONS = {"terra": "db9d0d694f776c9de56b8cb3fc3ab48804bd57a1",
             "musclemimic": "f1c2dbfd0d8e31d306b2c4e2369aecdc7bc21993",
             "holosoma": "12022b5ca5e12e460156c2d91a908f80f6c63fa1"}
PACKAGES = ("numpy", "scipy", "torch", "jax", "jaxlib", "mujoco", "mujoco-mjx",
            "flax", "orbax-checkpoint", "musclemimic-models", "cvxpy", "clarabel", "optax", "distrax")


def check_packages(sources):
    import tomllib
    lock = tomllib.loads((Path(sources["terra"]) / "uv.lock").read_text())
    versions = {name: {p["version"] for p in lock["package"] if p["name"] == name} for name in PACKAGES}
    installed = {name: importlib.metadata.version(name) for name in PACKAGES}
    for name, version in installed.items():
        if version not in versions[name]:
            raise ValueError(f"{name} {version} differs from the pinned Terra lockfile")
    return installed


def source_files(root):
    root = Path(root)
    return {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob("*"))
            if p.is_file() and p.suffix in (".py", ".toml", ".yaml", ".yml")
            and not any(part.startswith(".") for part in p.relative_to(root).parts)}


def check_sources(sources):
    from ..service import ROOT
    for name, revision in REVISIONS.items():
        actual = subprocess.check_output(["git", "-C", str(sources[name]), "rev-parse", "HEAD"], text=True).strip()
        if actual != revision:
            raise ValueError(f"unsupported {name} source revision")
    for name, patch in (("terra", "mhr.patch"), ("holosoma", "holosoma-smpl-optional.patch")):
        subprocess.run(["git", "-C", str(sources[name]), "apply", "--reverse", "--check",
                        str(ROOT / "ref/terra" / patch)], check=True, capture_output=True)


def read_installation(root, *, verify=False):
    if root is None:
        raise ValueError("configure --terra-assets and register a local Terra installation")
    data = json.loads((Path(root) / "installation.json").read_text())
    if data.get("version") != 1 or data.get("backend") != "terra" or data.get("revisions") != REVISIONS:
        raise ValueError("unsupported Terra installation receipt")
    if data.get("checkpoint_use_acknowledged") is not True or data.get("checkpoint_license_status") not in ("recorded", "unspecified"):
        raise ValueError("Terra checkpoint provenance/use have not been registered")
    if verify:
        import importlib.util
        if importlib.util.find_spec("smplx") is not None:
            raise ValueError("Terra worker requires the registered SMPL-free environment")
        from ..service import ROOT
        check_sources(data["sources"])
        for name, path in data["sources"].items():
            if source_files(path) != data["source_files"][name]:
                raise ValueError(f"{name} source files changed after registration")
        for name, digest in data["patches"].items():
            if sha256(ROOT / "ref/terra" / name) != digest:
                raise ValueError("Terra adapter patch changed; register a new installation")
        if tree_hashes(data["checkpoint"]) != data["checkpoint_files"]:
            raise ValueError("Terra checkpoint checksum mismatch")
        for name, version in data["packages"].items():
            if importlib.metadata.version(name) != version:
                raise ValueError(f"Terra package changed: {name}")
        from .terra_motion import asset_fingerprint
        if asset_fingerprint(data["mhr_assets"]) != data["mhr_fingerprint"]:
            raise ValueError("Terra native MHR assets changed")
    return data


def reference(data, identifier):
    if not isinstance(identifier, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", identifier):
        raise ValueError("invalid Terra reference ID")
    item = data["references"].get(identifier)
    if item is None or sha256(item["path"]) != item["sha256"]:
        raise ValueError("Terra reference is missing or its checksum changed")
    return item


def main():
    from ..service import ROOT
    from .terra_motion import asset_fingerprint, load_reference
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    register = sub.add_parser("register")
    register.add_argument("--assets", type=Path, required=True)
    for name in REVISIONS:
        register.add_argument("--" + name, type=Path, required=True)
    register.add_argument("--mhr-assets", type=Path, required=True)
    register.add_argument("--checkpoint", type=Path, required=True, help="complete checkpoint_24416 or compatible Orbax directory")
    register.add_argument("--checkpoint-revision", required=True)
    register.add_argument("--checkpoint-source-url", required=True)
    register.add_argument("--checkpoint-terms-url", required=True,
                          help="terms or local permission record; the public TERRA-4B card currently omits license metadata")
    terms = register.add_mutually_exclusive_group(required=True)
    terms.add_argument("--accept-checkpoint-terms", action="store_true")
    terms.add_argument("--acknowledge-unspecified-checkpoint-terms", action="store_true",
                       help="record unresolved publisher terms for local research validation; this is not a license grant")
    add = sub.add_parser("add-reference")
    add.add_argument("--assets", type=Path, required=True)
    add.add_argument("--id", required=True)
    add.add_argument("--motion", type=Path, required=True)
    add.add_argument("--source-url", required=True)
    add.add_argument("--terms-url", required=True)
    doctor = sub.add_parser("doctor"); doctor.add_argument("--assets", type=Path, required=True)
    args = ap.parse_args()
    if args.command == "register":
        output = args.assets / "installation.json"
        if output.exists():
            ap.error("installation already registered; use a new assets directory")
        sources = {name: str(getattr(args, name).resolve()) for name in REVISIONS}
        check_sources(sources)
        packages = check_packages(sources)
        if not (args.checkpoint / "_CHECKPOINT_METADATA").is_file():
            ap.error("checkpoint must contain the complete Orbax directory")
        # The default worker must be usable without the separately licensed SMPL package.
        import importlib.util
        if importlib.util.find_spec("smplx") is not None:
            ap.error("register Terra in the SMPL-free environment; use a separate environment for licensed comparisons")
        data = {"version": 1, "backend": "terra", "research_only": True, "revisions": REVISIONS,
                "sources": sources, "source_files": {n: source_files(p) for n,p in sources.items()},
                "patches": {n: sha256(ROOT / "ref/terra" / n) for n in ("mhr.patch", "holosoma-smpl-optional.patch")},
                "packages": packages,
                "checkpoint": str(args.checkpoint.resolve()), "checkpoint_files": tree_hashes(args.checkpoint),
                "checkpoint_revision": args.checkpoint_revision, "checkpoint_source_url": args.checkpoint_source_url,
                "checkpoint_terms_url": args.checkpoint_terms_url, "checkpoint_use_acknowledged": True,
                "checkpoint_terms_accepted": args.accept_checkpoint_terms,
                "checkpoint_license_status": "recorded" if args.accept_checkpoint_terms else "unspecified",
                "mhr_assets": str(args.mhr_assets.resolve()), "mhr_fingerprint": asset_fingerprint(args.mhr_assets),
                "references": {}}
        args.assets.mkdir(parents=True, exist_ok=True)
    else:
        data = read_installation(args.assets, verify=True)
        output = args.assets / "installation.json"
        if args.command == "doctor":
            print(json.dumps({"verified": True, "references": sorted(data["references"]), "packages": data["packages"]}, indent=2))
            return
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", args.id) or args.id in data["references"]:
            ap.error("reference ID must be new and contain only letters, digits, _ or -")
        load_reference(args.motion, data["mhr_fingerprint"])
        data["references"][args.id] = {"path": str(args.motion.resolve()), "sha256": sha256(args.motion),
                                      "source_url": args.source_url, "terms_url": args.terms_url}
    partial = output.with_suffix(".partial")
    partial.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    partial.replace(output)
    print(output)


if __name__ == "__main__":
    main()
