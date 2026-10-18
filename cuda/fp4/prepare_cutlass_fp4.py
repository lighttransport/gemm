#!/usr/bin/env python3
"""Generate a local CUTLASS mainloop override; never modify the checkout.

SM120 permits only single-CTA clusters here. PipelineTmaAsync initializes its
empty barrier for all consumer threads and every consumer thread arrives.
Synchronizing each warp's shared-memory reads before its arrivals therefore
protects a stage until every consumer has finished, without a separate CTA
rendezvous. The generated header retains NVIDIA's original license.
"""
import argparse
import hashlib
from pathlib import Path
import re

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("cutlass", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--handoff", choices=("cta", "warp"), default="warp")
parser.add_argument("--unroll", type=int, choices=(0, 2, 4, 8), default=0)
args = parser.parse_args()
relative = Path("cutlass/gemm/collective/sm120_blockscaled_mma_tma.hpp")
source = (args.cutlass / "include" / relative).read_text()
pattern = (r"cutlass::arch::NamedBarrier::sync\(\s*"
           r"thr_size\(tiled_mma\), cutlass::arch::ReservedNamedBarriers::Sm120MainloopBarrier\);")
patched, count = re.subn(pattern, "__syncwarp(); // All consumer threads subsequently arrive on the empty barrier.", source)
if args.handoff == "cta":
    patched = source
if count != 2:
    raise SystemExit(f"Expected exactly two mainloop rendezvous, found {count}; inspect this CUTLASS version.")
if args.unroll:
    loop = "CUTLASS_PRAGMA_NO_UNROLL\n    for ( ; k_tile_count > 1; --k_tile_count)"
    if patched.count(loop) != 1:
        raise SystemExit("Expected one mainloop K loop; inspect this CUTLASS version.")
    patched = patched.replace(loop, f"#pragma unroll {args.unroll}\n    for ( ; k_tile_count > 1; --k_tile_count)")
target = args.output / relative
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(patched)
print(f"source_sha256={hashlib.sha256(source.encode()).hexdigest()} handoff={args.handoff} unroll={args.unroll} output={target}")
