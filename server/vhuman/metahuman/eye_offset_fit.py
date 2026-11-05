"""MetaHuman-minus-GNM eye centre offset over the MH dataset, and a landmark regression for photo fits (CPU; 2026-11-06).

GNM places the eyeballs from its identity joints; on MH heads they sit ~1.9 mm high / ~.9 mm medial of the MH eyes
(README "12 h goal round" item 4). A fixed offset ties on s1/s2 (B9), so this fits the offset per head from what a photo
fit also has: the 68 iBUG eye landmarks (lid corners/margins) relative to the GNM iris centre.

per head: skin = binding(MH head) -> gnm_identity -> GNM iris centre (mean of the side's `irises` vertices);
          MH iris centre = x/y of the MH eye mesh's corneal cap (vertices within CAP_M of the most anterior one);
          offset = MH - GNM (x, y; vhuman frame metres); features = landmarks 36-41 / 42-47 minus the GNM iris centre.
model:    ridge on the 12 lid-landmark x/y (+1), both sides pooled with x mirrored; 5-fold CV vs the median offset.
usage:    python -m server.vhuman.metahuman.eye_offset_fit build [--limit N]   (-> MODELS_ROOT/eye_offset_v1.npz)
          python -m server.vhuman.metahuman.eye_offset_fit fit                 (CV report -> eye_offset_v1.json)
"""
import argparse
import json
from pathlib import Path

import numpy as np

from . import MODELS_ROOT

CAP_M = .0015
LANDMARKS = Path(__file__).resolve().parents[1] / 'reconstruction' / 'data' / 'gnm_landmarks68.txt'
EYE_LM = {'right': range(36, 42), 'left': range(42, 48)}      # iBUG: 36-41 = subject's right eye
DATA = MODELS_ROOT / 'eye_offset_v1.npz'


def landmark_points(full):
    rows = np.loadtxt(LANDMARKS)
    ids = rows[:, 0::2].astype(int); w = rows[:, 1::2]
    return (full[ids] * w[..., None]).sum(1)


def mh_iris_centre(eye):
    z = eye[:, 2]
    return eye[z > z.max() - CAP_M].mean(0)


def head_row(sid):
    from .binding import evaluate, load_binding
    from .compare_render import gnm_identity
    from .samples import V1, load_sample, to_vhuman
    s = load_sample(V1 / 'samples' / sid)
    fit = gnm_identity(evaluate(load_binding(), to_vhuman(s['face']['head'][0])))
    full, model = fit['full'], fit['model']
    lm = landmark_points(full)
    mh = {k: mh_iris_centre(to_vhuman(s['face'][k][0])) for k in ('eye_l', 'eye_r')}
    row = {}
    for side in ('left', 'right'):
        g = full[model.group(f'{side}_eye') & model.group('irises')].mean(0)
        m = min(mh.values(), key=lambda c: np.linalg.norm(c[:2] - g[:2]))   # MH eye nearest in x/y (naming-free)
        row[side] = dict(offset=(m - g)[:2], features=(lm[list(EYE_LM[side])] - g)[:, :2].ravel(), gnm=g, mh=m)
    return row


def build(limit=None):
    from .samples import V1
    sids = sorted(p.name for p in (V1 / 'samples').iterdir() if (p / 'done.json').is_file())[:limit]
    off, feat, who = [], [], []
    for i, sid in enumerate(sids):
        try:
            r = head_row(sid)
        except Exception as e:           # incomplete sample
            print(sid, 'skip', e, flush=True); continue
        for side in ('left', 'right'):
            o, f = r[side]['offset'].copy(), r[side]['features'].reshape(-1, 2).copy()
            if side == 'right':          # mirror x so both sides share one model
                o[0] = -o[0]; f[:, 0] = -f[:, 0]
            off.append(o); feat.append(f.ravel()); who.append(f'{sid}:{side}')
        if i % 50 == 0:
            print(i, sid, np.round(1e3 * r['left']['offset'], 2), flush=True)
    np.savez_compressed(DATA, offset=np.array(off), features=np.array(feat), ids=np.array(who))
    return DATA


def ridge(x, y, lam):
    xa = np.c_[x, np.ones(len(x))]
    reg = lam * np.eye(xa.shape[1]); reg[-1, -1] = 0
    return np.linalg.solve(xa.T @ xa + reg, xa.T @ y)


def predict(w, x):
    return np.c_[x, np.ones(len(x))] @ w


def fit(lams=(1e-6, 1e-5, 1e-4, 1e-3), folds=5, seed=0):
    z = np.load(DATA)
    x, y, ids = z['features'], z['offset'], z['ids']
    heads = np.array([i.split(':')[0] for i in ids]); uh = np.unique(heads)
    fold_of = dict(zip(np.random.default_rng(seed).permutation(uh), np.arange(len(uh)) % folds))
    f = np.array([fold_of[h] for h in heads])
    res = {}
    for lam in lams:
        err = np.zeros_like(y); base = np.zeros_like(y)
        for k in range(folds):
            tr, te = f != k, f == k
            err[te] = predict(ridge(x[tr], y[tr], lam), x[te]) - y[te]
            base[te] = np.median(y[tr], 0) - y[te]
        res[lam] = dict(rmse_mm=float(1e3 * np.sqrt((err ** 2).sum(1).mean())), baseline_rmse_mm=float(1e3 * np.sqrt((base ** 2).sum(1).mean())),
                        rmse_xy_mm=(1e3 * np.sqrt((err ** 2).mean(0))).round(3).tolist(), base_xy_mm=(1e3 * np.sqrt((base ** 2).mean(0))).round(3).tolist())
    best = min(res, key=lambda k: res[k]['rmse_mm'])
    w = ridge(x, y, best)
    out = dict(format='vhuman.metahuman_eye_offset.v1', n_eyes=int(len(y)), n_heads=int(len(uh)), lambda_=best, cv=res,
               median_offset_mm=(1e3 * np.median(y, 0)).round(3).tolist(), weights=w.tolist(),
               note='x mirrored for the right eye; features = iBUG lid landmarks minus GNM iris centre (x, y), metres',
               default_enabled=False)
    (MODELS_ROOT / 'eye_offset_v1.json').write_text(json.dumps(out, indent=1) + '\n')
    return out


def offset_for(full, model, weights):
    """Predicted MH-like eye offset per side (vhuman frame, [dx, dy, 0] m) for a GNM full-anatomy vertex array."""
    lm = landmark_points(full); w = np.asarray(weights); out = {}
    for side in ('left', 'right'):
        g = full[model.group(f'{side}_eye') & model.group('irises')].mean(0)
        f = (lm[list(EYE_LM[side])] - g)[:, :2].copy()
        if side == 'right':
            f[:, 0] = -f[:, 0]
        o = predict(w, f.ravel()[None])[0]
        if side == 'right':
            o[0] = -o[0]
        out[side] = [float(o[0]), float(o[1]), 0.]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('cmd', choices=('build', 'fit', 'check'))
    ap.add_argument('--limit', type=int)
    a = ap.parse_args()
    if a.cmd == 'build':
        print(build(a.limit)); return
    if a.cmd == 'fit':
        print(json.dumps(fit(), indent=1)); return
    # check: measured offset vs the 12 Cycles-measured eye_place_offset.json files
    from .compare_render import COMPARE_ROOT
    for f in sorted(COMPARE_ROOT.glob('mh_*/head/eye_place_offset.json')):
        sid = f.parts[-3]; ref = json.loads(f.read_text()); r = head_row(sid)
        print(sid, {s: (np.round(1e3 * np.asarray(ref[s][:2]), 2).tolist(), np.round(1e3 * r[s]['offset'], 2).tolist()) for s in ('left', 'right')})


if __name__ == '__main__':
    main()
