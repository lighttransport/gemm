"""Fetch a pinned native dependency and build its repo-owned conditioning overlay."""
import argparse
import os
from pathlib import Path
import subprocess
from patch_native import PIN, patch
ROOT = Path(__file__).resolve().parents[2]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="120", help="CUDA architecture (5060 Ti: 120)")
    ap.add_argument("--jobs", default="6")
    ap.add_argument("--nvcc", default="/usr/local/cuda/bin/nvcc")
    args = ap.parse_args()
    source = ROOT / "tmp/hunyuan-video15-native"
    temp = ROOT / "tmp/hunyuan-video15-build-tmp"
    temp.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, TMPDIR=str(temp))
    def run(*cmd):
        subprocess.run(cmd, check=True, env=env)
    if not source.exists():
        run("git", "clone", "--no-checkout", "https://github.com/leejet/stable-diffusion.cpp", str(source))
        run("git", "-C", str(source), "checkout", "--detach", PIN)
    run("git", "-C", str(source), "submodule", "update", "--init", "ggml")
    patch(source)
    run("cmake", "-S", str(source), "-B", str(source / "build"),
        "-DSD_CUDA=ON", "-DSD_BUILD_EXAMPLES=OFF", "-DSD_BUILD_SHARED_LIBS=ON",
        "-DSD_WEBP=OFF", "-DSD_WEBM=OFF", "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_PROJECT_INCLUDE={ROOT / 'cuda/hunyuan_video15/cuda_overlay.cmake'}",
        f"-DCMAKE_CUDA_ARCHITECTURES={args.arch}", f"-DCMAKE_CUDA_COMPILER={args.nvcc}")
    run("cmake", "--build", str(source / "build"), "-j", args.jobs)
    run("make", "-C", str(ROOT / "cuda/hunyuan_video15"))

if __name__ == "__main__":
    main()
