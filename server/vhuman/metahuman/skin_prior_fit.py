"""Fit the GNM-UV skin prior (mean + PCA of log albedo, split into low/high frequency; cavity mean) from the MetaHuman dataset.

usage: python -m server.vhuman.metahuman.skin_prior_fit [--res 1024] [--lf-res 256] [--k-lf 80] [--k-hf 60] [--out PATH]
                                                       [--eval-mask PATH_TO_skin_coverage.png]
Every sample's MetaHuman textures are resampled into GNM UV through the fixed texel map (register.texel_map), so the prior
lives in exactly the atlas the vhuman bake writes. Held-out ids (sampling.json) are never fitted; they are used for the
evaluation: observe only the texels a real vhuman front bake covers (eval mask) and predict the rest, against the
baseline the bake uses today (dataset-free: observed median colour everywhere unobserved) and the prior mean with a
colour gain match.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from . import DATASET_ROOT, MODELS_ROOT
from .manifest import prior_report, sha256, write_json
from .register import OUT_DIR as BINDING_DIR, resample
from .samples import sample_dirs, texture
from ..reconstruction.skin_prior import SkinPrior, pca_small_n

V1 = DATASET_ROOT / 'v1'
LOG_FLOOR = 1e-3


def srgb_to_linear(x):
    x = np.asarray(x, np.float32)
    return np.where(x <= .04045, x / 12.92, ((x + .055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    x = np.clip(np.asarray(x, np.float32), 0, 1)
    return np.where(x <= .0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - .055)


def masked_downsample(img, mask, lf_res):
    h = img.shape[0]; f = h // lf_res
    w = mask.astype(np.float32)
    num = (img * w[..., None]).reshape(lf_res, f, lf_res, f, -1).sum((1, 3))
    den = w.reshape(lf_res, f, lf_res, f).sum((1, 3))
    return num / np.maximum(den[..., None], 1e-6), den / (f * f)


def load_maps(d, tmap, res):
    base = texture(d, 'basecolor')[..., :3].astype(np.float32) / 255.
    cav = texture(d, 'cavity')[..., :3].astype(np.float32) / 255.
    a, ok = resample(srgb_to_linear(base), tmap)
    c, _ = resample(cav, tmap)
    return np.log(np.maximum(a, LOG_FLOOR)), c, ok


def fit(res=1024, lf_res=256, k_lf=80, k_hf=60, out=None, eval_mask=None, limit=0):
    t0 = time.time()
    tmap = dict(np.load(BINDING_DIR / f'texel_map_{res}.npz'))
    sampling = json.loads((V1 / 'sampling.json').read_text())
    held = set(sampling['held_out_ids'])
    dirs = sample_dirs()
    if limit:
        dirs = dirs[:limit]
    train = [d for d in dirs if d.name not in held]; test = [d for d in dirs if d.name in held]
    print(f'{len(train)} train, {len(test)} held-out samples at {res}^2', flush=True)
    mask = tmap['valid'].copy()
    n = len(train)
    lf = np.zeros((n, lf_res, lf_res, 3), np.float32)
    logs = np.zeros((n, res, res, 3), np.float32)
    cav_sum = np.zeros((res, res, 3), np.float64)
    for i, d in enumerate(train):
        la, c, ok = load_maps(d, tmap, res)
        mask &= ok
        logs[i] = la; cav_sum += c
        if i % 50 == 0:
            print(f'  loaded {i}/{n} ({time.time() - t0:.0f}s)', flush=True)
    lf_valid = None
    for i in range(n):
        lf[i], cover = masked_downsample(logs[i], mask, lf_res)
        lf_valid = cover > .25 if lf_valid is None else lf_valid
    lf_mean, lf_basis, lf_sigma, lf_expl = pca_small_n(lf.reshape(n, -1), min(k_lf, n - 1))
    proj = lf_mean + (lf.reshape(n, -1) - lf_mean) @ lf_basis.T @ lf_basis
    up = np.stack([SkinPrior.upsample(p.reshape(lf_res, lf_res, 3), res) for p in proj])
    resid = logs - up
    resid[:, ~mask] = 0
    hf_mean, hf_basis, hf_sigma, hf_expl = pca_small_n(resid.reshape(n, -1), min(k_hf, n - 1))
    hf_std = resid.std(0).mean(-1)
    out = Path(out or MODELS_ROOT / 'skin_prior_v1.npz'); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, res=res, lf_res=lf_res, mask=mask, lf_valid=lf_valid, lf_mean=lf_mean.reshape(lf_res, lf_res, 3).astype(np.float32),
             lf_basis=lf_basis.astype(np.float16), lf_sigma=lf_sigma, hf_mean=hf_mean.reshape(res, res, 3).astype(np.float16),
             hf_basis=hf_basis.astype(np.float16), hf_sigma=hf_sigma, hf_std=hf_std.astype(np.float16),
             cavity_mean=(cav_sum / n).mean(-1).astype(np.float16), normal_mean=np.zeros((res, res, 2), np.float16))
    print(f'fitted LF {lf_basis.shape} (explained {lf_expl.sum():.3f}) HF {hf_basis.shape} (explained {hf_expl.sum():.3f}) in {time.time() - t0:.0f}s', flush=True)
    del logs, resid, up
    ev = evaluate(out, test, tmap, res, eval_mask) if test else None
    manifest_path = V1 / 'manifest.json'
    rep = prior_report('skin_prior_v1', dataset_manifest_sha256=sha256(manifest_path) if manifest_path.is_file() else None,
                       sample_ids_used=[d.name for d in train], held_out_ids=[d.name for d in test], weights_path=out,
                       derived_from='MetaHuman Creator local skin synthesis (TS-1.3-F, 1024) resampled to GNM UV via binding_v1 texel map',
                       limitations=['albedo is MetaHuman synthesized basecolor (no measured reflectance); 1024 source textures',
                                    'normal prior not fitted yet (tangent-frame conversion pending)',
                                    'texels outside the MetaHuman-supported mask (ear folds, mouth sock) carry no prior'],
                       extra=dict(res=res, lf_res=lf_res, k_lf=int(lf_basis.shape[0]), k_hf=int(hf_basis.shape[0]), lf_explained=float(lf_expl.sum()),
                                  hf_explained=float(hf_expl.sum()), mask_texels=int(mask.sum()), texel_map_sha256=sha256(BINDING_DIR / f'texel_map_{res}.npz'),
                                  evaluation=ev, seconds=time.time() - t0))
    write_json(out.with_name(out.stem + '.report.json'), rep)
    return rep


def evaluate(prior_path, test, tmap, res, eval_mask=None, ridge=1.0):
    """Held-out completion test: observe `eval_mask` texels (a real vhuman front-bake coverage), predict the rest."""
    prior = SkinPrior(prior_path)
    mask = prior.mask
    if eval_mask:
        obs = np.asarray(Image.open(eval_mask).convert('L').resize((res, res), Image.NEAREST)) > 127
    else:                                    # fallback: left half of the atlas observed
        obs = np.zeros((res, res), bool); obs[:, :res // 2] = True
    obs &= mask; hidden = mask & ~obs
    rows = []
    for d in test:
        la, _, _ = load_maps(d, tmap, res)
        w = obs.astype(np.float32)
        c = prior.fit_coefficients(la, w, ridge=ridge)
        pred = prior.reconstruct(c)
        c_lf = dict(lf=c['lf'], hf=None, res=res); pred_lf = prior.reconstruct(c_lf)
        mean_img = prior.reconstruct(dict(lf=np.zeros_like(c['lf']), hf=None, res=res))
        gain = np.median(la[obs] - mean_img[obs], 0)                      # per-channel log gain from the observed texels
        mean_gain = mean_img + gain
        med = np.median(la[obs], 0)                                       # today's fallback: observed median everywhere
        truth = la[hidden]
        def err(p):
            lin_t, lin_p = linear_to_srgb(np.exp(truth)), linear_to_srgb(np.exp(p))
            return dict(log_mae=float(np.abs(p - truth).mean()), srgb_mae_255=float(255 * np.abs(lin_p - lin_t).mean()))
        rows.append(dict(id=d.name, prior=err(pred[hidden]), prior_lf_only=err(pred_lf[hidden]), mean_gain=err(mean_gain[hidden]),
                         median_fill=err(np.broadcast_to(med, truth.shape))))
    agg = {k: {m: float(np.mean([r[k][m] for r in rows])) for m in ('log_mae', 'srgb_mae_255')} for k in ('prior', 'prior_lf_only', 'mean_gain', 'median_fill')}
    print('held-out completion (unobserved texels):', json.dumps(agg), flush=True)
    return dict(observed_texels=int(obs.sum()), hidden_texels=int(hidden.sum()), eval_mask=str(eval_mask) if eval_mask else 'left half', ridge=ridge,
                mean=agg, per_sample=rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--res', type=int, default=1024); ap.add_argument('--lf-res', type=int, default=256)
    ap.add_argument('--k-lf', type=int, default=80); ap.add_argument('--k-hf', type=int, default=60)
    ap.add_argument('--out', default=None); ap.add_argument('--eval-mask', default=None); ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    rep = fit(a.res, a.lf_res, a.k_lf, a.k_hf, a.out, a.eval_mask, a.limit)
    print(json.dumps({k: v for k, v in rep.items() if k not in ('sample_ids_used', 'held_out_ids', 'evaluation')}, indent=1)[:2000])


if __name__ == '__main__':
    main()
