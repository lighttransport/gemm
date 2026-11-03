"""Fit the GNM-UV skin normal prior from MetaHuman tangent normals converted to GNM tangent space (normal_map.convert).

usage: python -m server.vhuman.metahuman.normal_prior_fit [--res 1024] [--k 40] [--workers 8] [--blur 12] [--out PATH]

Per training sample (held-out ids excluded) the MH normal map is converted on that identity's own MH head / bound GNM skin,
so each map is exactly what the Cycles graph reads (Blender convention, green = +V up). Stored on tangent xy:
  mean (res,res,2), std (res,res), PCA (K, res*res*2) + sigma, and the split of the mean into a low-frequency part
  (Gaussian blur, --blur px: mostly the MH-vs-GNM mesh shading difference, which depends on the mesh the map is applied
  to) and the detail part (pores, wrinkles, creases: what transfers to any GNM head). Held-out samples report the
  per-texel xy error of the mean and of the detail-only mean, against a flat map.
Output normal_prior_v1.npz + .report.json under MODELS_ROOT (off by default; never committed).
"""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from . import MODELS_ROOT
from .manifest import prior_report, sha256, write_json
from .register import OUT_DIR as BINDING_DIR
from .samples import V1, sample_dirs

_STATE = {}


def _init(res):
    from .binding import load_binding
    _STATE['binding'] = load_binding()
    _STATE['tmap'] = dict(np.load(BINDING_DIR / f'texel_map_{res}.npz'))


def _convert(d):
    from PIL import Image
    from .binding import evaluate
    from .normal_map import convert
    from .samples import load_sample, to_vhuman
    s = load_sample(d)
    mh_v = to_vhuman(s['face']['head'][0]).astype(np.float64)
    n, ok = convert(np.asarray(Image.open(Path(d) / 'tex' / 'normal.png')), mh_v, evaluate(_STATE['binding'], mh_v), _STATE['tmap'])
    return n[..., :2].astype(np.float32), ok


def fit(res=1024, k=40, workers=8, blur=12., out=None, limit=0):
    from scipy import ndimage
    from ..reconstruction.skin_prior import pca_small_n
    from .normal_map import gnm_raster
    t0 = time.time()
    gnm_raster(res)                            # build the cache once before the workers start
    sampling = json.loads((V1 / 'sampling.json').read_text()); held = set(sampling['held_out_ids'])
    dirs = sample_dirs()[:limit] if limit else sample_dirs()
    train = [d for d in dirs if d.name not in held]; test = [d for d in dirs if d.name in held]
    print(f'{len(train)} train, {len(test)} held-out at {res}^2, {workers} workers', flush=True)
    xy = np.zeros((len(train), res, res, 2), np.float32); mask = None
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(res,)) as ex:
        for i, (n, ok) in enumerate(ex.map(_convert, train, chunksize=4)):
            xy[i] = n; mask = ok if mask is None else mask & ok
            if i % 50 == 0:
                print(f'  converted {i}/{len(train)} ({time.time() - t0:.0f}s)', flush=True)
        test_xy = list(ex.map(_convert, test, chunksize=2))
    xy[:, ~mask] = 0
    mean = xy.mean(0); std = np.linalg.norm(xy - mean, axis=-1).std(0)
    lf = np.stack([ndimage.gaussian_filter(mean[..., c], blur) for c in range(2)], -1); lf[~mask] = 0
    detail = mean - lf
    m, basis, sigma, expl = pca_small_n(xy.reshape(len(xy), -1), min(k, len(xy) - 1))
    out = Path(out or MODELS_ROOT / 'normal_prior_v1.npz'); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, res=res, mask=mask, mean_xy=mean.astype(np.float16), detail_xy=detail.astype(np.float16), lf_xy=lf.astype(np.float16),
             std=std.astype(np.float16), basis=basis.astype(np.float16), sigma=sigma, blur_px=blur, convention='blender_tangent_green_up')

    def err(pred, truth):
        return float(np.sqrt(((pred - truth)[mask] ** 2).sum(-1).mean()))
    ev = []
    for d, (n, _) in zip(test, test_xy):
        t = n[..., :2]
        ev.append(dict(id=d.name, flat=err(np.zeros_like(t), t), mean=err(mean, t), detail_only=err(detail, t), lf_only=err(lf, t)))
    summary = {k2: float(np.mean([e[k2] for e in ev])) for k2 in ('flat', 'mean', 'detail_only', 'lf_only')} if ev else None
    print(f'fitted K={len(basis)} explained {expl.sum():.3f}; held-out rms xy error {summary} in {time.time() - t0:.0f}s', flush=True)
    manifest_path = V1 / 'manifest.json'
    rep = prior_report('normal_prior_v1', dataset_manifest_sha256=sha256(manifest_path) if manifest_path.is_file() else None,
                       sample_ids_used=[d.name for d in train], held_out_ids=[d.name for d in test], weights_path=out,
                       derived_from='MetaHuman Creator local skin synthesis normal maps (1024) converted to GNM tangent space (normal_map.convert)',
                       limitations=['tangent xy only (z = sqrt(1 - x^2 - y^2)); Blender convention, green = +V up',
                                    'mean includes the MetaHuman-vs-GNM mesh shading difference (lf_xy); apply detail_xy to non-MetaHuman heads',
                                    'texels outside the MetaHuman-supported mask carry no prior (flat)'],
                       extra=dict(res=res, k=int(len(basis)), explained=float(expl.sum()), blur_px=blur, mask_texels=int(mask.sum()),
                                  texel_map_sha256=sha256(BINDING_DIR / f'texel_map_{res}.npz'), evaluation=dict(summary=summary, per_sample=ev),
                                  seconds=time.time() - t0))
    write_json(out.with_name(out.stem + '.report.json'), rep)
    return rep


def write_png(prior_path, which, path):
    """Encode a prior map (mean_xy / detail_xy) as a Blender-convention skin_normal.png (z reconstructed)."""
    from PIL import Image
    from .normal_map import encode
    z = np.load(prior_path); xy = z[which].astype(np.float64)
    n = np.concatenate([xy, np.sqrt(np.clip(1 - (xy ** 2).sum(-1, keepdims=True), 0, 1))], -1)
    Image.fromarray(encode(n)).save(path)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--res', type=int, default=1024); ap.add_argument('--k', type=int, default=40)
    ap.add_argument('--workers', type=int, default=8); ap.add_argument('--blur', type=float, default=12.)
    ap.add_argument('--out'); ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    rep = fit(a.res, a.k, a.workers, a.blur, a.out, a.limit)
    print(json.dumps({k: rep[k] for k in ('k', 'explained', 'mask_texels', 'seconds')} | dict(evaluation=rep['evaluation']['summary']), indent=1))


if __name__ == '__main__':
    main()
