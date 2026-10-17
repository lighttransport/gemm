"""Stage pinned source and optionally install the isolated SMPL-free Terra stack.

No model, policy checkpoint, mocap or licensed SMPL asset is downloaded.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from server.vhuman.body.terra_assets import REVISIONS
from server.python_env import cache_path, environment_path


def run(*args, **kwargs):
    return subprocess.run(list(map(str, args)), check=True, **kwargs)


def stage(destination):
    destination.mkdir(parents=True, exist_ok=True)
    projects = {"terra": "amathislab/terra", "musclemimic": "amathislab/musclemimic", "holosoma": "amazon-far/holosoma"}
    for name, repository in projects.items():
        path = destination / name
        if not path.exists():
            run("git", "clone", "--filter=blob:none", "--no-checkout", "https://github.com/"+repository, path)
            run("git", "-C", path, "checkout", REVISIONS[name])
        revision = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
        if revision != REVISIONS[name]:
            raise ValueError(f"existing {name} checkout is not at the pinned revision")
    for name, patch in (("terra", "mhr.patch"), ("holosoma", "holosoma-smpl-optional.patch")):
        path = destination/name
        patch_path = ROOT/"ref/terra"/patch
        reverse = subprocess.run(["git", "-C", str(path), "apply", "--reverse", "--check", str(patch_path)],
                                 capture_output=True)
        if reverse.returncode:
            run("git", "-C", path, "apply", "--check", patch_path)
            run("git", "-C", path, "apply", patch_path)
    return destination


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=ROOT/"tmp/terra-runtime")
    ap.add_argument("--python", default="3.11")
    ap.add_argument("--install", action="store_true")
    ap.add_argument("--cuda", action="store_true", help="include the pinned JAX CUDA plugin")
    args = ap.parse_args()
    root = args.root.resolve()
    os.environ.update(TMPDIR=str(root/"tmp"), UV_CACHE_DIR=str(cache_path()))
    os.environ.setdefault("UV_HTTP_TIMEOUT", "300")
    (root/"tmp").mkdir(parents=True, exist_ok=True)
    stage(root)
    if args.install:
        environment = environment_path(root/"venv", link=True)
        if not environment.exists():
            run("uv", "venv", "--python", args.python, environment)
        # Frozen export retains upstream's package versions. OpenCV 4 retains
        # the HOG person detector required by our video input preflight.
        # Install the enumerated closure without dependency resolution, excluding SMPL and
        # replacing source packages with the patched, pinned local checkouts.
        requirements = root/"requirements.txt"
        command = ["uv", "export", "--frozen", "--no-hashes", "--no-emit-project",
                   "--no-emit-package", "smplx", "--no-emit-package", "musclemimic",
                   "--no-emit-package", "holosoma-retargeting", "--output-file", str(requirements)]
        if args.cuda:
            command.extend(["--extra", "cuda"])
        run(*command, cwd=root/"terra", stdout=subprocess.DEVNULL)
        requirements.write_text(re.sub(r'(?m)^opencv-python==[^;\s]+',
            'opencv-python==4.13.0.92', requirements.read_text()))
        python = environment/"bin/python"
        run("uv", "pip", "install", "--python", python, "--no-deps", "-r", requirements,
            "--extra-index-url", "https://download.pytorch.org/whl/cu126", "--index-strategy", "unsafe-best-match")
        for path in (root/"musclemimic",root/"holosoma/src/holosoma_retargeting",root/"terra"):
            run("uv", "pip", "install", "--python", python, "--no-deps", "-e", path)
        run(python, "-c", "import importlib.util,cv2; assert importlib.util.find_spec('smplx') is None; "
            "assert hasattr(cv2,'HOGDescriptor'); from terra.prepared import PreparedMotion")
    print(root)


if __name__ == "__main__":
    main()
