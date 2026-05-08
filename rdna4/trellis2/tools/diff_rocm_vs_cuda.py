"""Compare ROCm vs CUDA dump trees and report per-stage divergence.

Sparse-tensor pairs (.coords + .feats) are reordered by lexicographic coord
sort before comparison. Bisect rule: the first stage with `ok=False` is the
module to inspect.
"""
import argparse
import json
import os
import sys

import numpy as np


def _canon_sparse(coords: np.ndarray, feats: np.ndarray):
    key = np.lexsort(coords.T[::-1])
    return coords[key], feats[key]


def _stat(name: str, ref: np.ndarray, test: np.ndarray, atol: float, rtol: float, *, exact: bool=False):
    if ref.shape != test.shape:
        print(f'{name:40s} SHAPE MISMATCH ref={ref.shape} test={test.shape}')
        return False
    if exact:
        ok = np.array_equal(ref, test)
        n_diff = int((ref != test).sum())
        print(f'{name:40s} exact={ok} n_diff={n_diff}/{ref.size}')
        return ok
    ref_f = ref.astype(np.float32)
    test_f = test.astype(np.float32)
    diff = np.abs(ref_f - test_f)
    denom = np.maximum(np.abs(ref_f), 1e-9)
    rel = diff / denom
    a = ref_f.ravel()
    b = test_f.ravel()
    a0 = a - a.mean(); b0 = b - b.mean()
    cos = float((a0 @ b0) / (np.linalg.norm(a0) * np.linalg.norm(b0) + 1e-12))
    ok = bool(np.allclose(ref_f, test_f, atol=atol, rtol=rtol))
    print(f'{name:40s} max|d|={diff.max():.3e} mean|d|={diff.mean():.3e} '
          f'max_rel={rel.max():.3e} cos={cos:.6f} ok={ok}')
    return ok


def cmp_dense(ref_dir, test_dir, name, **kw):
    rp = os.path.join(ref_dir, f'{name}.npy')
    tp = os.path.join(test_dir, f'{name}.npy')
    if not (os.path.exists(rp) and os.path.exists(tp)):
        print(f'{name:40s} MISSING ref={os.path.exists(rp)} test={os.path.exists(tp)}')
        return False
    return _stat(name, np.load(rp), np.load(tp), **kw)


def cmp_sparse(ref_dir, test_dir, prefix, *, atol: float, rtol: float):
    rc = np.load(os.path.join(ref_dir, f'{prefix}.coords.npy'))
    rf = np.load(os.path.join(ref_dir, f'{prefix}.feats.npy'))
    tc = np.load(os.path.join(test_dir, f'{prefix}.coords.npy'))
    tf = np.load(os.path.join(test_dir, f'{prefix}.feats.npy'))
    if rc.shape[0] != tc.shape[0]:
        print(f'{prefix:40s} N MISMATCH ref={rc.shape[0]} test={tc.shape[0]}')
        # Try to compare overlap by coord set
        ref_set = {tuple(r) for r in rc.tolist()}
        test_set = {tuple(r) for r in tc.tolist()}
        common = ref_set & test_set
        print(f'{prefix:40s}  |ref|={len(ref_set)} |test|={len(test_set)} '
              f'|common|={len(common)} |ref-test|={len(ref_set - test_set)} '
              f'|test-ref|={len(test_set - ref_set)}')
        return False
    rc, rf = _canon_sparse(rc, rf)
    tc, tf = _canon_sparse(tc, tf)
    coords_ok = np.array_equal(rc, tc)
    print(f'{prefix:40s} coords_match={coords_ok}')
    if not coords_ok:
        return False
    return _stat(f'{prefix}.feats', rf, tf, atol=atol, rtol=rtol)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref',  default='cuda/trellis2/verify-dumps')
    ap.add_argument('--test', default='rdna4/trellis2/verify-dumps-rocm')
    args = ap.parse_args()

    ref = os.path.abspath(args.ref)
    test = os.path.abspath(args.test)
    print(f'ref  = {ref}')
    print(f'test = {test}')
    print()

    # Stages roughly correspond to manifest order. Tolerances per README.md.
    # exact: image, noise, coords (after sort).
    cmp_dense(ref, test, '00_image_preprocessed', atol=0, rtol=0, exact=True)
    cmp_dense(ref, test, '01_dinov3_cond_512',     atol=5e-3, rtol=1e-3)
    cmp_dense(ref, test, '01_dinov3_neg_cond_512', atol=0, rtol=0, exact=True)
    cmp_dense(ref, test, '02_ss_noise',            atol=0, rtol=0, exact=True)
    cmp_dense(ref, test, '03_ss_latent',           atol=1e-2, rtol=1e-3)
    cmp_dense(ref, test, '04_ss_decoder_logits',   atol=5e-3, rtol=1e-3)
    # 05 coords: sort then exact
    rc = np.load(os.path.join(ref, '05_ss_coords.npy'))
    tc = np.load(os.path.join(test, '05_ss_coords.npy'))
    rc_s = rc[np.lexsort(rc.T[::-1])]
    tc_s = tc[np.lexsort(tc.T[::-1])]
    if rc_s.shape == tc_s.shape:
        ok = np.array_equal(rc_s, tc_s)
        print(f"{'05_ss_coords':40s} ref_N={rc.shape[0]} test_N={tc.shape[0]} exact_after_sort={ok}")
    else:
        ref_set = {tuple(r) for r in rc.tolist()}
        test_set = {tuple(r) for r in tc.tolist()}
        common = ref_set & test_set
        print(f"{'05_ss_coords':40s} N MISMATCH ref={rc.shape[0]} test={tc.shape[0]} common={len(common)}")

    cmp_dense(ref, test, '06_shape_slat_noise_feats', atol=1e-4, rtol=1e-4)
    cmp_dense(ref, test, '07_shape_slat_raw_feats',   atol=1e-2, rtol=1e-3)
    cmp_dense(ref, test, '08_shape_slat_denorm_feats',atol=1e-1, rtol=1e-3)
    cmp_dense(ref, test, '09_tex_slat_noise_feats',   atol=1e-4, rtol=1e-4)
    cmp_dense(ref, test, '10_tex_concat_cond_feats',  atol=1e-2, rtol=1e-3)
    cmp_dense(ref, test, '11_tex_slat_raw_feats',     atol=1e-2, rtol=1e-3)
    cmp_dense(ref, test, '12_tex_slat_denorm_feats',  atol=5e-2, rtol=1e-3)

    # 13 mesh: vertex order may differ — compare vertex count + bbox + face count.
    rv = np.load(os.path.join(ref, '13_mesh_vertices.npy'))
    tv = np.load(os.path.join(test, '13_mesh_vertices.npy'))
    rf = np.load(os.path.join(ref, '13_mesh_faces.npy'))
    tf_ = np.load(os.path.join(test, '13_mesh_faces.npy'))
    print(f"{'13_mesh_vertices':40s} ref={rv.shape} test={tv.shape} "
          f"bbox_ref={rv.min(0)}..{rv.max(0)} bbox_test={tv.min(0)}..{tv.max(0)}")
    print(f"{'13_mesh_faces':40s} ref_F={rf.shape[0]} test_F={tf_.shape[0]}")

    for i in range(8):
        prefix = f'14_shape_sub{i}'
        if not os.path.exists(os.path.join(ref, f'{prefix}.coords.npy')):
            break
        cmp_sparse(ref, test, prefix, atol=5e-2, rtol=1e-3)

    cmp_sparse(ref, test, '15_tex_voxels', atol=1e-2, rtol=1e-3)


if __name__ == '__main__':
    main()
