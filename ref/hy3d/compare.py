"""Compare reference .npy outputs against CUDA runner outputs.

Usage:
    uv run python compare.py <ref_dir> <test_dir> [rtol] [atol]
"""
import os
import sys
import numpy as np


def compare(name, ref, test, rtol=1e-3, atol=1e-4):
    if ref.shape != test.shape:
        print(f"  {name}: SHAPE MISMATCH ref={ref.shape} test={test.shape}")
        return False
    diff = np.abs(ref - test)
    ok = np.allclose(ref, test, rtol=rtol, atol=atol)
    status = "OK" if ok else "FAIL"
    print(f"  {name}: {status}  max={diff.max():.2e} mean={diff.mean():.2e} shape={ref.shape}")
    if not ok:
        idx = np.unravel_index(diff.argmax(), diff.shape)
        print(f"    worst@{idx}: ref={ref[idx]:.6f} test={test[idx]:.6f}")
    return ok


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <ref_dir> <test_dir> [rtol] [atol]")
        sys.exit(1)
    ref_dir, test_dir = sys.argv[1], sys.argv[2]
    rtol = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-3
    atol = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-4
    ref_files = {f for f in os.listdir(ref_dir) if f.endswith(".npy")}
    test_files = {f for f in os.listdir(test_dir) if f.endswith(".npy")}
    common = sorted(ref_files & test_files)
    ok = fail = 0
    for f in common:
        if compare(
            f,
            np.load(os.path.join(ref_dir, f)),
            np.load(os.path.join(test_dir, f)),
            rtol=rtol,
            atol=atol,
        ):
            ok += 1
        else:
            fail += 1
    print(f"\n{ok} OK, {fail} FAIL / {len(common)} compared")
    sys.exit(0 if fail == 0 else 1)


if __name__ == "__main__":
    main()
