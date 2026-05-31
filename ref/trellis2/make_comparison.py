#!/usr/bin/env python3
"""
Compare two .npy tensors (reference vs implementation) and report metrics.

Usage:
  python make_comparison.py ref_block05_hidden.npy hip_block05_hidden.npy
  python make_comparison.py ref_dit_output.npy hip_dit_output.npy [--label "DiT output"]

  # Compare all per-block hidden states:
  for N in $(seq -f "%02g" 0 29); do
    python make_comparison.py ref_block${N}_hidden.npy hip_block${N}_hidden.npy --label "block $N"
  done
"""
import argparse
import numpy as np
import sys


def compare(ref_path, impl_path, label=None):
    ref  = np.load(ref_path).astype(np.float32).flatten()
    impl = np.load(impl_path).astype(np.float32).flatten()

    if ref.shape != impl.shape:
        print(f'  SHAPE MISMATCH: ref={ref.shape} impl={impl.shape}')
        return False

    diff     = np.abs(ref - impl)
    max_diff = float(diff.max())
    mean_diff= float(diff.mean())
    p99_diff = float(np.percentile(diff, 99))
    rel_l2   = float(np.linalg.norm(ref - impl) / (np.linalg.norm(ref) + 1e-30))

    # Pearson correlation
    ref_c  = ref  - ref.mean()
    impl_c = impl - impl.mean()
    denom  = np.sqrt((ref_c**2).sum()) * np.sqrt((impl_c**2).sum()) + 1e-30
    corr   = float((ref_c * impl_c).sum() / denom)

    lbl = label or f'{ref_path} vs {impl_path}'
    print(f'--- {lbl} ---')
    print(f'  shape:    {np.load(ref_path).shape}')
    print(f'  max_diff: {max_diff:.6e}')
    print(f'  mean_diff:{mean_diff:.6e}')
    print(f'  p99_diff: {p99_diff:.6e}')
    print(f'  rel_L2:   {rel_l2:.6e}')
    print(f'  corr:     {corr:.8f}')

    # Reference range
    print(f'  ref  range: [{ref.min():.4f}, {ref.max():.4f}]  std={ref.std():.4f}')
    print(f'  impl range: [{impl.min():.4f}, {impl.max():.4f}]  std={impl.std():.4f}')

    # First 8 values
    ref_np = np.load(ref_path).astype(np.float32).flatten()
    impl_np= np.load(impl_path).astype(np.float32).flatten()
    print(f'  first8 ref:  {ref_np[:8]}')
    print(f'  first8 impl: {impl_np[:8]}')

    # Verdict
    if corr > 0.9999 and max_diff < 0.01:
        verdict = 'PASS'
    elif corr > 0.999 and max_diff < 0.1:
        verdict = 'MARGINAL'
    else:
        verdict = 'FAIL'
    print(f'  => {verdict}  (PASS: corr>0.9999 && max_diff<0.01)')
    print()
    return verdict == 'PASS'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ref',  help='Reference .npy (PyTorch)')
    parser.add_argument('impl', help='Implementation .npy (HIP)')
    parser.add_argument('--label', default=None)
    args = parser.parse_args()

    ok = compare(args.ref, args.impl, args.label)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
