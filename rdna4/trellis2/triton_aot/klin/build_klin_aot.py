#!/usr/bin/env python3
"""Compile + extract hsacos for the 8 tex_dec klin shapes.

Per-shape config recovered from bench_klin_triton.py sweep on RX 9070 XT
(gfx1201). Writes:
  - kernels/<tag>/kernel.hsaco + kernel.json   (Triton-compiled)
  - shapes.json                                 (tag -> M,K,N,BM,BN,BK,nw)

Usage:
  ./build_klin_aot.py
"""
import json, os, shutil, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
KERNELS_DIR = HERE / "kernels"
KERNEL_NAME = "klin_bf16_kernel"

# Use a fresh per-build cache so we can deterministically locate each shape's
# compiled output. Must be set BEFORE importing triton.
_TMP_CACHE = Path(tempfile.mkdtemp(prefix="klin_aot_cache_"))
os.environ["TRITON_CACHE_DIR"] = str(_TMP_CACHE)
TRITON_CACHE = _TMP_CACHE

import torch, triton, triton.language as tl

# (M, K, N, BM, BN, BK, num_warps, num_stages)
# M is upper-bound (chunks may be smaller; kernel handles via masks).
# Same K and N per shape — only the per-call M changes.
SHAPES = [
    ("stage0_klin_up", 1905,  1024, 4096, 128, 128, 32, 4, 2),
    ("stage0_klin_dn", 1905,  4096, 1024,  64, 128, 64, 4, 2),
    ("stage1_klin_up", 8452,   512, 2048,  64, 128, 64, 4, 2),
    ("stage1_klin_dn", 8452,  2048,  512,  64, 128, 64, 4, 2),
    ("stage2_klin_up", 16384,  256, 1024,  64, 256, 32, 8, 2),
    ("stage2_klin_dn", 16384, 1024,  256,  64, 128, 64, 4, 2),
    ("stage3_klin_up", 16384,  128,  512,  64, 256, 32, 8, 2),
    ("stage3_klin_dn", 16384,  512,  128,  64, 128, 64, 4, 2),
]


@triton.jit
def klin_bf16_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    M, N, K,
    sxm: tl.constexpr, sxk: tl.constexpr,
    swn: tl.constexpr, swk: tl.constexpr,
    sym: tl.constexpr, syn: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)

    X_block = X_ptr + rm[:, None] * sxm + rk[None, :] * sxk
    W_block = W_ptr + rn[:, None] * swn + rk[None, :] * swk

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    mask_m = rm < M
    mask_n = rn < N
    for k0 in range(0, K, BK):
        mask_k = (k0 + rk) < K
        x = tl.load(X_block, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        w = tl.load(W_block, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
        acc += tl.dot(x, tl.trans(w))
        X_block += BK * sxk
        W_block += BK * swk

    bias = tl.load(B_ptr + rn, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]
    Y = Y_ptr + rm[:, None] * sym + rn[None, :] * syn
    tl.store(Y, acc, mask=mask_m[:, None] & mask_n[None, :])


def compile_one(M, K, N, BM, BN, BK, nw, ns):
    """Run the kernel once and return the cache directory it wrote to.

    Snapshots cache dirs containing the kernel's hsaco before+after; the new
    dir is this shape's compile.
    """
    def _hsaco_dirs():
        return {p.name for p in TRITON_CACHE.iterdir()
                if p.is_dir() and (p / f"{KERNEL_NAME}.hsaco").is_file()}

    dev = "cuda"
    X = torch.zeros(M, K, device=dev, dtype=torch.bfloat16)
    W = torch.zeros(N, K, device=dev, dtype=torch.bfloat16)
    B = torch.zeros(N, device=dev, dtype=torch.float32)
    Y = torch.zeros(M, N, device=dev, dtype=torch.float32)
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    before = _hsaco_dirs()
    klin_bf16_kernel[grid](
        X, W, B, Y, M, N, K,
        X.stride(0), X.stride(1),
        W.stride(0), W.stride(1),
        Y.stride(0), Y.stride(1),
        BM=BM, BN=BN, BK=BK,
        num_warps=nw, num_stages=ns,
    )
    torch.cuda.synchronize()
    after = _hsaco_dirs()
    new = after - before
    if len(new) == 1:
        return TRITON_CACHE / next(iter(new))
    if len(new) > 1:
        # Multiple new — pick the youngest by mtime.
        cands = [TRITON_CACHE / n for n in new]
        return max(cands, key=lambda p: p.stat().st_mtime)
    # No new dir — Triton's in-process compile cache hit even though disk had
    # no copy. Fall back to youngest.
    cands = [p for p in TRITON_CACHE.iterdir()
             if p.is_dir() and (p / f"{KERNEL_NAME}.hsaco").is_file()]
    return max(cands, key=lambda p: p.stat().st_mtime) if cands else None


def index_triton_cache():
    """Build {(num_warps, shared, BM, BN, BK): cache_dir}."""
    idx = {}
    for sub in TRITON_CACHE.iterdir():
        if not sub.is_dir(): continue
        jp = sub / f"{KERNEL_NAME}.json"
        hp = sub / f"{KERNEL_NAME}.hsaco"
        if not (jp.is_file() and hp.is_file()): continue
        meta = json.loads(jp.read_text())
        # constants dict has BM/BN/BK as strings or ints depending on triton ver
        consts = meta.get("constants", {}) or meta.get("constexprs", {})
        def gi(k, default=None):
            v = consts.get(k, default)
            return int(v) if v is not None else None
        bm = gi("BM"); bn = gi("BN"); bk = gi("BK")
        if bm is None:  # try positional names
            continue
        key = (meta["num_warps"], bm, bn, bk)
        idx[key] = (sub, meta)
    return idx


def main():
    if not torch.cuda.is_available():
        sys.exit("error: torch.cuda not available — need ROCm pytorch venv")
    KERNELS_DIR.mkdir(exist_ok=True)
    print(f"[1/2] Compiling + extracting {len(SHAPES)} klin shapes...")
    shapes_out = []
    for tag, M, K, N, BM, BN, BK, nw, ns in SHAPES:
        src = compile_one(M, K, N, BM, BN, BK, nw, ns)
        if src is None:
            sys.exit(f"  {tag}: cache dir not found")
        meta = json.loads((src / f"{KERNEL_NAME}.json").read_text())
        dst = KERNELS_DIR / tag
        dst.mkdir(exist_ok=True)
        shutil.copy2(src / f"{KERNEL_NAME}.hsaco", dst / "kernel.hsaco")
        shutil.copy2(src / f"{KERNEL_NAME}.json",  dst / "kernel.json")
        shapes_out.append({
            "tag": tag, "M_max": M, "K": K, "N": N,
            "BM": BM, "BN": BN, "BK": BK,
            "num_warps": nw, "num_stages": ns,
            "shared": meta.get("shared", 0),
        })
        print(f"  {tag:20s} <- {src.name[:8]}/  (shared={meta.get('shared')}, "
              f"vgprs={meta.get('vgpr_spill_count', '?')})")
    print(f"[2/2] Writing shapes.json")
    (HERE / "shapes.json").write_text(json.dumps(shapes_out, indent=2))
    print(f"\nOK: {len(shapes_out)} shapes packaged in {KERNELS_DIR}")


if __name__ == "__main__":
    main()
