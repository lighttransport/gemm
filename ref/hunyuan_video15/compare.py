"""Fail-closed comparison of paired component .npy dumps; no empty-set passes."""
import argparse
import json
from pathlib import Path
import numpy as np
COMPONENTS = ("qwen_hidden", "byt5_hidden", "siglip_hidden", "vae_encoded",
              "dit_first", "latent_final", "vae_decoded")

def compare_frames(reference, actual, cosine_min=0.9999, relative_l2_max=0.02):
    """Check every NCTHW frame; a video-wide average can hide motion errors."""
    r = np.load(Path(reference) / 'vae_decoded.npy', allow_pickle=False, mmap_mode='r')
    a = np.load(Path(actual) / 'vae_decoded.npy', allow_pickle=False, mmap_mode='r')
    if r.shape != a.shape or r.ndim != 5 or r.shape[0] != 1 or not r.size:
        raise ValueError(f'invalid decoded video shapes {r.shape} / {a.shape}')
    results = []
    for frame in range(r.shape[2]):
        rv, av = r[0,:,frame].astype(np.float64).ravel(), a[0,:,frame].astype(np.float64).ravel()
        if not np.isfinite(rv).all() or not np.isfinite(av).all():
            raise ValueError(f'frame {frame}: non-finite pixels')
        nr, na = np.linalg.norm(rv), np.linalg.norm(av)
        cosine = float(np.dot(rv,av)/(nr*na)) if nr and na else float(np.array_equal(rv,av))
        relative_l2 = float(np.linalg.norm(rv-av)/max(nr,1e-30))
        results.append({'frame':frame,'cosine':cosine,'relative_l2':relative_l2,
                        'mean_absolute_error':float(np.abs(rv-av).mean()),
                        'pass':cosine>=cosine_min and relative_l2<=relative_l2_max})
    return results

def compare(reference, actual, names=COMPONENTS, cosine_min=0.9999, relative_l2_max=0.02):
    if not names:
        raise ValueError("at least one component is required")
    results = {}
    for name in names:
        r = np.load(Path(reference) / f"{name}.npy", allow_pickle=False, mmap_mode='r')
        a = np.load(Path(actual) / f"{name}.npy", allow_pickle=False, mmap_mode='r')
        if r.shape != a.shape or not r.size:
            raise ValueError(f"{name}: mismatched or empty shape {r.shape} / {a.shape}")
        # Bound working memory for full-resolution 81/121-frame videos.
        rr = aa = dot = error = 0.
        equal = True
        iterator = np.nditer([r, a], flags=['external_loop', 'buffered'],
                             op_flags=[['readonly'], ['readonly']], order='C', buffersize=1<<18)
        for rv, av in iterator:
            rv, av = rv.astype(np.float64), av.astype(np.float64)
            if not np.isfinite(rv).all() or not np.isfinite(av).all():
                raise ValueError(f"{name}: non-finite values")
            delta = rv-av
            rr += float(np.dot(rv, rv)); aa += float(np.dot(av, av))
            dot += float(np.dot(rv, av)); error += float(np.dot(delta, delta))
            equal = equal and np.array_equal(rv, av)
        nr, na = np.sqrt(rr), np.sqrt(aa)
        cosine = float(dot/(nr*na)) if nr and na else float(equal)
        relative_l2 = float(np.sqrt(error)/max(nr, 1e-30))
        results[name] = {'cosine': cosine, 'relative_l2': relative_l2,
                         'pass': cosine >= cosine_min and relative_l2 <= relative_l2_max}
    return results

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("reference")
    ap.add_argument("actual")
    ap.add_argument("--components", nargs="+", choices=COMPONENTS, default=COMPONENTS)
    args = ap.parse_args()
    result = compare(args.reference, args.actual, args.components)
    print(json.dumps(result, indent=2))
    return 0 if all(r["pass"] for r in result.values()) else 1

if __name__ == "__main__":
    raise SystemExit(main())
