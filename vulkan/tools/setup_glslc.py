#!/usr/bin/env python3
"""Build a pinned glslc locally, without a system install or Vulkan SDK download."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
REVISION = "2c8cae778eec0283b44acbe7ed1a386865d78799"  # Shaderc v2026.3
DEPENDENCIES = {
    "glslang": ("glslang", "168d452a4f460d24b588fed08477a81c44ee27a1"),
    "spirv-headers": ("SPIRV-Headers", "29981f65241605e08b0ede4cfeb999fe3b723c6a"),
    "spirv-tools": ("SPIRV-Tools", "b707790a898e44038547df54580022fc1cf89c3d"),
}


def run(command, **kwargs):
    return subprocess.run(command, check=True, **kwargs)


def checkout(url, revision, path):
    if not path.exists():
        path.mkdir(parents=True)
        run(["git", "init", str(path)])
        run(["git", "-C", str(path), "remote", "add", "origin", url])
    actual = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                            capture_output=True, text=True)
    dirty = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain",
                                     "--untracked-files=no"], text=True)
    if dirty:
        raise ValueError(f"Refusing to change modified source: {path}")
    if actual.returncode:
        run(["git", "-C", str(path), "fetch", "--depth=1", "origin", revision])
        run(["git", "-C", str(path), "checkout", "--detach", revision])
    elif actual.stdout.strip() != revision:
        raise ValueError(f"Expected pinned revision {revision}: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, default=ROOT / "tmp/vhuman-tools")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    for tool in ("git", "cmake", "ninja", "c++"):
        if not shutil.which(tool):
            parser.error(f"Required build tool is missing: {tool}")
    prefix = args.prefix.absolute()
    prefix.mkdir(parents=True, exist_ok=True)
    temp = ROOT / "tmp/vhuman-runtime"
    temp.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(temp)
    source = prefix / "shaderc/source"
    build = prefix / "shaderc/build"
    checkout("https://github.com/google/shaderc.git", REVISION, source)
    for name, (repo, revision) in DEPENDENCIES.items():
        checkout(f"https://github.com/KhronosGroup/{repo}.git", revision,
                 source / "third_party" / name)
    run(["cmake", "-S", str(source), "-B", str(build), "-G", "Ninja",
         "-DCMAKE_BUILD_TYPE=Release", "-DSHADERC_SKIP_TESTS=ON",
         "-DSHADERC_SKIP_EXAMPLES=ON", "-DSHADERC_SKIP_COPYRIGHT_CHECK=ON",
         "-DSPIRV_SKIP_TESTS=ON", "-DSPIRV_SKIP_EXECUTABLES=ON",
         "-DENABLE_GLSLANG_BINARIES=OFF", "-DBUILD_TESTING=OFF"])
    run(["cmake", "--build", str(build), "--target", "glslc_exe",
         "--parallel", str(args.jobs)])
    binary = build / "glslc/glslc"
    version = subprocess.check_output([str(binary), "--version"], text=True)
    (prefix / "bin").mkdir(exist_ok=True)
    target = prefix / "bin/glslc"
    if target.exists() and not target.is_symlink():
        raise ValueError(f"Refusing to replace an existing executable: {target}")
    target.unlink(missing_ok=True)
    target.symlink_to(Path("../shaderc/build/glslc/glslc"))
    digest = hashlib.sha256()
    with binary.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    checksum = digest.hexdigest()
    manifest = {"shaderc_revision": REVISION, "dependencies": DEPENDENCIES,
                "version": version, "binary_sha256": checksum}
    (prefix / "shaderc/installation.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(version, str(target), sep="\n", flush=True)


if __name__ == "__main__":
    main()
