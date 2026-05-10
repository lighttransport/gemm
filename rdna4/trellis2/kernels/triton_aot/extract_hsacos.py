#!/usr/bin/env python3
"""Extract Triton AOT hsacos for the 9 tex_dec spconv shapes.

Reads `shapes.json` (the per-shape autotune choices for RX 9070 XT / gfx1201),
walks ~/.triton/cache/*/ to find the matching compiled kernels, and copies
`kernel.hsaco` + `kernel.json` into `kernels/<tag>/`.

Matching strategy: each cache entry's `<kernel_name>.json` records
(name, num_warps, num_stages, shared). The tuple (name, num_warps, shared)
uniquely identifies the 4 distinct binaries that cover all 9 shapes.

Prereq: ~/.triton/cache must already contain the spconv kernels. Trigger this
by running the PyTorch tex_dec pipeline once on RX 9070 XT, e.g.
    cd cpu/trellis2 && PYTHONPATH=... python gen_stage2_ref.py --skip-dit \
        --output-dir <dump> --mesh <obj> --image <png> \
        --dinov3 <model.safetensors> --resolution 512
"""
import json, os, shutil, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
KERNELS_DIR = HERE / "kernels"
TRITON_CACHE = Path(os.environ.get("TRITON_CACHE_DIR",
                                   Path.home() / ".triton" / "cache"))
KERNEL_NAMES = {
    True:  "sparse_submanifold_conv_fwd_masked_implicit_gemm_splitk_kernel",
    False: "sparse_submanifold_conv_fwd_masked_implicit_gemm_kernel",
}
# Per-config shared-LDS bytes (B1*BK + B2*BK + (SPLITK?B2*BK:0)) * 2 (fp16)
# Matches what Triton emits for these autotune choices.
def expected_shared(B1, B2, BK, splitk):
    base = (B1 * BK + B2 * BK) * 2
    return 16384 if (B1, B2, BK) == (128, 128, 32) else \
           12288 if (B1, B2, BK) == (64, 128, 32) else \
           8192  if (B1, B2, BK) == (64, 64, 32) else None


def index_triton_cache():
    """Build {(kernel_name, num_warps, shared): cache_dir} from triton cache."""
    if not TRITON_CACHE.is_dir():
        sys.exit(f"error: triton cache dir not found: {TRITON_CACHE}")
    idx = {}
    for sub in TRITON_CACHE.iterdir():
        if not sub.is_dir():
            continue
        for kname in KERNEL_NAMES.values():
            jp = sub / f"{kname}.json"
            hp = sub / f"{kname}.hsaco"
            if jp.is_file() and hp.is_file():
                meta = json.loads(jp.read_text())
                key = (meta["name"], meta["num_warps"], meta.get("shared", 0))
                idx.setdefault(key, sub)
    return idx


def load_shapes():
    sp = HERE / "shapes.json"
    if not sp.is_file():
        sys.exit(f"error: {sp} missing")
    return json.loads(sp.read_text())


def is_splitk(s):
    """True iff this shape compiles to the splitk_kernel (SPLITK_factor>1).
    On gfx1201/RX 9070 XT only (1905,1024,1024) hits SPLITK=4; the rest go
    through the splitk dispatcher but with SPLITK=1, which compiles into the
    regular masked_implicit_gemm_kernel."""
    return (s["N"], s["Ci"], s["Co"]) == (1905, 1024, 1024)


def shape_tag(s):
    return f"N{s['N']}_Ci{s['Ci']}_Co{s['Co']}_SPLITK{4 if is_splitk(s) else 1}"


def scrub_meta_paths(meta):
    """Drop host-local absolute paths from Triton's generated metadata."""
    meta = dict(meta)
    libs = meta.get("extern_libs")
    if isinstance(libs, list):
        scrubbed = []
        for item in libs:
            if not (isinstance(item, list) and len(item) == 2 and isinstance(item[1], str)):
                scrubbed.append(item)
                continue
            lib, path = item
            marker = "site-packages/"
            if marker in path:
                path = path.split(marker, 1)[1]
            scrubbed.append([lib, path])
        meta["extern_libs"] = scrubbed
    return meta


def autotune_for(s):
    """Per-shape autotune choice baked at sweep-time; mirrors the table in
    triton_spconv_bridge.h's AUTOTUNE.  Update both if you re-tune."""
    N, Ci, Co = s["N"], s["Ci"], s["Co"]
    if (N, Ci, Co) == (1905, 1024, 1024):
        return dict(B1=64, B2=128, BK=32, num_warps=4, num_stages=2)
    if (N, Ci, Co) == (1905, 1024, 4096):
        return dict(B1=128, B2=128, BK=32, num_warps=8, num_stages=2)
    if (N, Ci, Co) == (822874, 64, 64):
        return dict(B1=64, B2=64, BK=32, num_warps=4, num_stages=2)
    return dict(B1=64, B2=128, BK=32, num_warps=4, num_stages=2)


def main():
    shapes = load_shapes()
    cache = index_triton_cache()
    print(f"triton cache @ {TRITON_CACHE}: found {len(cache)} unique spconv kernels")

    KERNELS_DIR.mkdir(exist_ok=True)
    miss = []
    for s in shapes:
        tag = shape_tag(s)
        a = autotune_for(s)
        kname = KERNEL_NAMES[is_splitk(s)]
        shared = expected_shared(a["B1"], a["B2"], a["BK"], s["splitk"])
        key = (kname, a["num_warps"], shared)
        src = cache.get(key)
        if src is None:
            miss.append((tag, key))
            continue
        dst = KERNELS_DIR / tag
        dst.mkdir(exist_ok=True)
        shutil.copy2(src / f"{kname}.hsaco", dst / "kernel.hsaco")
        meta = scrub_meta_paths(json.loads((src / f"{kname}.json").read_text()))
        (dst / "kernel.json").write_text(json.dumps(meta, separators=(",", ": ")) + "\n")
        print(f"  {tag:42s}  <- {src.name}/")

    if miss:
        print("\nMISSING (not in triton cache — run the PyTorch tex_dec pipeline first):")
        for tag, key in miss:
            print(f"  {tag}: looking for name={key[0]} warps={key[1]} shared={key[2]}")
        sys.exit(1)
    print(f"\nOK: {len(shapes)} shapes packaged into {KERNELS_DIR}")


if __name__ == "__main__":
    main()
