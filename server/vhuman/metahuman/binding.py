"""Fixed binding of the GNM skin surface to the MetaHuman head topology, and per-identity GNM shapes.

register.py places the GNM skin vertices on the MetaHuman archetype; each GNM vertex then has an MH triangle,
barycentrics and a small signed offset along the MH surface normal. Because every MetaHuman character shares the head
topology, `evaluate(mh_head)` turns any generated head into GNM-topology skin vertices. `identity_coefficients` projects
those onto the GNM identity basis (similarity removed), giving the dataset's distribution of GNM identity coefficients.
"""
import argparse
import json

import numpy as np

from . import MODELS_ROOT
from . import template as mh_template
from .register import OUT_DIR, load_gnm, umeyama, vertex_normals
from .samples import load_sample, sample_dirs, to_vhuman


def load_binding():
    reg = np.load(OUT_DIR / 'gnm_on_mh.npz')
    mh = mh_template.load()
    tri = mh['triangles'][reg['mh_tri']]                   # (Ns,3) MH vertex ids
    bary = reg['mh_bary'].astype(np.float64)
    base = (mh['vertices'][tri] * bary[..., None]).sum(1)
    n = vertex_normals(mh['vertices'].astype(np.float64), mh['triangles'])
    nb = (n[tri] * bary[..., None]).sum(1); nb /= np.maximum(np.linalg.norm(nb, axis=1, keepdims=True), 1e-12)
    offset = ((reg['vertices'] - base) * nb).sum(1)       # signed normal offset (m), sub-millimetre after NR-ICP
    return dict(skin_ids=reg['skin_ids'], tri=tri, bary=bary, offset=offset, mh_triangles=mh['triangles'])


def evaluate(binding, mh_head_vh):
    v = np.asarray(mh_head_vh, np.float64)
    n = vertex_normals(v, binding['mh_triangles'])
    p = (v[binding['tri']] * binding['bary'][..., None]).sum(1)
    nb = (n[binding['tri']] * binding['bary'][..., None]).sum(1); nb /= np.maximum(np.linalg.norm(nb, axis=1, keepdims=True), 1e-12)
    return p + binding['offset'][:, None] * nb


def identity_coefficients(gnm, skin_vertices, modes=170, ridge=1e-3):
    """Least-squares GNM identity coefficients for target skin vertices after removing a similarity transform."""
    ids = gnm['skin_ids']
    tmpl = gnm['template'][ids]; basis = gnm['basis'][:modes, ids].reshape(modes, -1)
    s, r, t = umeyama(skin_vertices, tmpl)                  # bring the target into the GNM template frame
    target = (s * (r @ skin_vertices.T)).T + t
    d = (target - tmpl).reshape(-1)
    a = basis @ basis.T + ridge * np.eye(modes)
    beta = np.linalg.solve(a, basis @ d)
    resid = d - beta @ basis
    return beta, float(np.sqrt((resid.reshape(-1, 3) ** 2).sum(1).mean())), dict(scale=float(s))


def fit_identity_prior(out=None, modes=170):
    binding = load_binding(); gnm = load_gnm()
    sampling = json.loads((sample_dirs()[0].parents[1] / 'sampling.json').read_text())
    held = set(sampling['held_out_ids'])
    rows, ids, rms, held_rms = [], [], [], []
    for d in sample_dirs():
        s = load_sample(d)
        skin = evaluate(binding, to_vhuman(s['face']['head'][0]))
        beta, err, _ = identity_coefficients(gnm, skin, modes)
        (held_rms if d.name in held else rms).append(err)
        if d.name not in held:
            rows.append(beta); ids.append(d.name)
    b = np.array(rows)
    mean, cov = b.mean(0), np.cov(b.T)
    out = out or MODELS_ROOT / 'identity_prior_v1.npz'
    np.savez(out, mean=mean.astype(np.float32), cov=cov.astype(np.float32), std=b.std(0).astype(np.float32), modes=modes)
    rep = dict(n_train=len(rows), n_held=len(held_rms), fit_rms_mm_train=float(1e3 * np.mean(rms)), fit_rms_mm_held=float(1e3 * np.mean(held_rms)) if held_rms else None,
               beta_std_first10=b.std(0)[:10].round(3).tolist(), beta_mean_abs_first10=np.abs(mean[:10]).round(3).tolist(), out=str(out))
    out.with_suffix('.json').write_text(json.dumps(rep, indent=1) + '\n')
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--identity-prior', action='store_true')
    a = ap.parse_args()
    b = load_binding()
    print('binding: GNM skin vertices', len(b['skin_ids']), 'normal offset mm |max|', round(1e3 * np.abs(b['offset']).max(), 3))
    if a.identity_prior:
        print(json.dumps(fit_identity_prior(), indent=1))


if __name__ == '__main__':
    main()
