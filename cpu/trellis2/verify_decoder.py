#!/usr/bin/env python3
"""Verify CUDA decoder output against PyTorch reference."""
import sys, os, numpy as np

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <ref_occupancy.npy> <cuda_occupancy.npy>")
        return 1

    ref = np.load(sys.argv[1])
    cuda = np.load(sys.argv[2])
    if ref.ndim == 4: ref = ref[0]
    if cuda.ndim == 4: cuda = cuda[0]

    print(f"Ref:  shape={ref.shape}, range=[{ref.min():.2f}, {ref.max():.2f}], "
          f"occ={(ref>0).sum()}/{ref.size} ({100*(ref>0).sum()/ref.size:.1f}%)")
    print(f"CUDA: shape={cuda.shape}, range=[{cuda.min():.2f}, {cuda.max():.2f}], "
          f"occ={(cuda>0).sum()}/{cuda.size} ({100*(cuda>0).sum()/cuda.size:.1f}%)")

    if ref.shape != cuda.shape:
        print(f"Shape mismatch!"); return 1

    corr = np.corrcoef(ref.flatten(), cuda.flatten())[0,1]
    rel = np.linalg.norm(ref.flatten()-cuda.flatten()) / np.linalg.norm(ref.flatten())
    print(f"Correlation: {corr:.8f}")
    print(f"Rel L2: {rel:.8f}")

    # Per-slice check
    for z in [0, 16, 32, 48, 63]:
        r = np.corrcoef(ref[z].flatten(), cuda[z].flatten())[0,1]
        print(f"  z={z}: corr={r:.6f}")

    return 0 if corr > 0.99 else 1

if __name__ == "__main__":
    sys.exit(main())
