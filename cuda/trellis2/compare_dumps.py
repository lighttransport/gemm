#!/usr/bin/env python3
"""Compare two TRELLIS.2 dump dirs entry-by-entry (CUDA reference vs HIP/other).

Generalizes the ad-hoc diff snippets in verify-dumps/README.md and
rdna4/trellis2/tools/compare_shape_dec_stage2.py into one manifest-driven tool
that covers the stage-boundary dumps AND the per-layer (per-DiT-block /
per-decoder-layer) dumps produced by `dump_ground_truth.py --dump-per-block`.

For every .npy present in BOTH dirs it reports shape, rel-L2, max-abs-diff and
cosine, then flags the first divergent entry (top-down bisection: a divergence
at layer N localizes the bug to the module between N-1 and N).

SparseTensor handling
---------------------
Sparse rows have no canonical order, so when --canon (default) is set, feature
tensors (`*.feats.npy`) are reordered by lexsorting their coordinate table
before diffing. The coords for a feats file are found by longest-name-prefix
match against the available `*.coords.npy` files in the same dir, which matches
both the exact pairing of stage dumps (`15_tex_voxels.{coords,feats}`) and the
once-per-level pairing of per-layer decoder dumps (`shapedec_L02.coords` pairs
with every `shapedec_L02_b*.feats`). `*.coords.npy` files themselves are
compared as integer row *sets*.

Usage
-----
  compare_dumps.py --ref cuda/trellis2/verify-dumps --test /path/to/rocm-dumps
  compare_dumps.py --ref cuda/trellis2/verify-dumps --test rocm-dumps --per-layer
  compare_dumps.py --ref .../verify-dumps/per_layer --test .../rocm/per_layer
"""
import argparse
import glob
import json
import os
import sys

import numpy as np


def _load(path):
    return np.load(path, allow_pickle=False)


def _coords_for(feats_stem, coords_stems):
    """Longest coords stem that is a prefix of feats_stem (or exact match)."""
    cands = [c for c in coords_stems if feats_stem == c or feats_stem.startswith(c)]
    return max(cands, key=len) if cands else None


def _canon_order(coords):
    """Lexsort row order of an [N,4] (b,x,y,z) or [N,3] coord table."""
    cols = [coords[:, k] for k in range(coords.shape[1])][::-1]
    return np.lexsort(cols)


def _metrics(ref, test):
    r = ref.astype(np.float64).ravel()
    t = test.astype(np.float64).ravel()
    d = r - t
    denom = max(np.sqrt((r * r).sum()), 1e-30)
    rel_l2 = float(np.sqrt((d * d).sum()) / denom)
    max_abs = float(np.abs(d).max()) if d.size else 0.0
    nr, nt = np.sqrt((r * r).sum()), np.sqrt((t * t).sum())
    cos = float((r * t).sum() / max(nr * nt, 1e-30))
    return rel_l2, max_abs, cos


def _stems(dir_):
    """Map of stem -> filename for *.npy in dir_ (stem strips the .npy)."""
    out = {}
    for p in glob.glob(os.path.join(dir_, '*.npy')):
        out[os.path.basename(p)[:-4]] = p
    return out


def compare_dir(ref_dir, test_dir, atol, rtol, canon, label=''):
    ref = _stems(ref_dir)
    test = _stems(test_dir)
    common = sorted(set(ref) & set(test))
    only_ref = sorted(set(ref) - set(test))
    if not common:
        print(f'  (no common .npy files in {label or ref_dir})')
        return []

    ref_coords = [s for s in ref if s.endswith('.coords')]
    test_coords = [s for s in test if s.endswith('.coords')]

    print(f'{"entry":42s} {"shape":>16s} {"rel_L2":>11s} {"max_abs":>11s} '
          f'{"cosine":>10s}  ok')
    print('-' * 96)
    rows = []
    for stem in common:
        a, b = _load(ref[stem]), _load(test[stem])
        if a.shape != b.shape:
            # coords/feats whose row *count* differs is itself a finding
            print(f'{stem:42s} {str(a.shape):>16s}  SHAPE MISMATCH vs {b.shape}')
            rows.append((stem, None, None, None, False))
            continue

        if stem.endswith('.coords'):
            # Compare as integer row sets (ordering is non-canonical).
            sa = {tuple(r) for r in a.astype(np.int64)}
            sb = {tuple(r) for r in b.astype(np.int64)}
            ok = sa == sb
            extra = '' if ok else f'  (|ref\\test|={len(sa - sb)} |test\\ref|={len(sb - sa)})'
            print(f'{stem:42s} {str(a.shape):>16s} {"set":>11s} {"-":>11s} '
                  f'{"-":>10s}  {"Y" if ok else "N"}{extra}')
            rows.append((stem, 0.0 if ok else 1.0, None, None, ok))
            continue

        ra, rb = a, b
        if canon and stem.endswith('.feats'):
            cs_a = _coords_for(stem, ref_coords)
            cs_b = _coords_for(stem, test_coords)
            if cs_a and cs_b:
                ca, cb = _load(ref[cs_a]), _load(test[cs_b])
                if ca.shape == a.shape[:1] + ca.shape[1:] or ca.shape[0] == a.shape[0]:
                    ra = a[_canon_order(ca)]
                    rb = b[_canon_order(cb)]

        rel_l2, max_abs, cos = _metrics(ra, rb)
        ok = bool(np.allclose(ra, rb, atol=atol, rtol=rtol))
        print(f'{stem:42s} {str(a.shape):>16s} {rel_l2:11.3e} {max_abs:11.3e} '
              f'{cos:10.6f}  {"Y" if ok else "N"}')
        rows.append((stem, rel_l2, max_abs, cos, ok))

    if only_ref:
        print(f'\n  {len(only_ref)} entries only in ref (not in test): '
              f'{", ".join(only_ref[:8])}{" ..." if len(only_ref) > 8 else ""}')
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ref', required=True, help='reference dump dir (CUDA)')
    ap.add_argument('--test', required=True, help='dump dir to check (HIP/other)')
    ap.add_argument('--atol', type=float, default=1e-3)
    ap.add_argument('--rtol', type=float, default=1e-2)
    ap.add_argument('--no-canon', action='store_true',
                    help='skip SparseTensor coord-canonicalization (diff raw row order)')
    ap.add_argument('--per-layer', action='store_true',
                    help='also descend into the per_layer/ subdir of both dirs')
    args = ap.parse_args()
    canon = not args.no_canon

    print(f'ref : {args.ref}\ntest: {args.test}\natol={args.atol} rtol={args.rtol} '
          f'canon={canon}\n')

    all_rows = compare_dir(args.ref, args.test, args.atol, args.rtol, canon)

    if args.per_layer:
        rpl = os.path.join(args.ref, 'per_layer')
        tpl = os.path.join(args.test, 'per_layer')
        if os.path.isdir(rpl) and os.path.isdir(tpl):
            print(f'\n=== per_layer/ ===')
            all_rows += compare_dir(rpl, tpl, args.atol, args.rtol, canon, label='per_layer')
        else:
            print(f'\n[skip] per_layer/ missing in ref or test')

    # Summary: first divergence + worst entry.
    failed = [r for r in all_rows if r[4] is False]
    print('\n' + '=' * 96)
    print(f'{len(all_rows)} compared, {len(all_rows) - len(failed)} ok, {len(failed)} FAILED')
    if failed:
        print(f'first divergence : {failed[0][0]}')
        scored = [r for r in failed if r[1] is not None]
        if scored:
            worst = max(scored, key=lambda r: r[1])
            print(f'worst rel_L2     : {worst[0]}  rel_L2={worst[1]:.3e}')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
