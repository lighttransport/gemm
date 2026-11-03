"""Seeded sample design for the MetaHuman dataset -> jobs.jsonl (one JSON job per line, consumed by ue_scripts/gen_batch.py).

Face shape: S-A the shipped presets as-is; S-B pairwise preset blends (lerp scale/translation/PCA weights, slerp quaternions);
S-C per-patch Gaussian around the preset mean with the preset standard deviation per coefficient (scaled), rigid parts
(quaternion/scale/translation) copied from a random preset. Skin: Latin-hypercube skin tone (u,v) and face_texture_index
covering 0..152 with a held-out set; eyes from presets (jittered) or uniform over the UI ranges.
Coefficient layout (verified in smoke 01): [nPatches=22], per patch: scale, quat(x,y,z,w), translation(3), nPCA, weights.
"""
import argparse
import json
import math
import random
from pathlib import Path

import numpy as np

from . import DATASET_ROOT

SEED = 20261103
N_TEXTURE_CHARACTERS = 153
EYE_RANGES = dict(iris_pattern=(1, 9), iris_rotation=(0., 1.), primary_color_u=(0., 1.), primary_color_v=(0., 1.), secondary_color_u=(0., 1.),
                  secondary_color_v=(0., 1.), color_blend=(0., 1.), color_blend_softness=(0., 1.), shadow_details=(0., 1.), limbal_ring_size=(.6, .85),
                  limbal_ring_softness=(.02, .15), global_saturation=(.5, 1.5), pupil_dilation=(.85, 1.2), pupil_feather=(0., 1.), cornea_size=(.145, .185),
                  cornea_limbus_softness=(.03, .15), sclera_transmission_spread=(.03, .2), sclera_vascularity_intensity=(0., 2.), sclera_vascularity_coverage=(.1, .4))


def parse_coefficients(c):
    c = [float(x) for x in c]; n = int(round(c[0])); i = 1; patches = []
    for _ in range(n):
        k = int(round(c[i + 8]))
        patches.append(dict(scale=c[i], quat=c[i + 1:i + 5], translation=c[i + 5:i + 8], weights=c[i + 9:i + 9 + k]))
        i += 9 + k
    assert i == len(c), (i, len(c))
    return patches


def flatten_coefficients(patches):
    out = [float(len(patches))]
    for p in patches:
        out += [p['scale']] + list(p['quat']) + list(p['translation']) + [float(len(p['weights']))] + list(p['weights'])
    return out


def slerp(q0, q1, t):
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > .9995:
        q = q0 + t * (q1 - q0)
    else:
        th = math.acos(min(1., d)); q = (math.sin((1 - t) * th) * q0 + math.sin(t * th) * q1) / math.sin(th)
    return (q / np.linalg.norm(q)).tolist()


def blend(pa, pb, t):
    out = []
    for a, b in zip(pa, pb):
        out.append(dict(scale=(1 - t) * a['scale'] + t * b['scale'], quat=slerp(a['quat'], b['quat'], t),
                        translation=[(1 - t) * x + t * y for x, y in zip(a['translation'], b['translation'])],
                        weights=[(1 - t) * x + t * y for x, y in zip(a['weights'], b['weights'])]))
    return out


def preset_stats(presets):
    """Per-patch mean/std of PCA weights over the presets; rigid parts kept per preset."""
    names = sorted(presets)
    parsed = {n: parse_coefficients(presets[n]['coefficients']) for n in names}
    stats = []
    for pi in range(len(parsed[names[0]])):
        w = np.array([parsed[n][pi]['weights'] for n in names])
        stats.append(dict(mean=w.mean(0).tolist(), std=(w.std(0) + 1e-6).tolist()))
    return names, parsed, stats


def gaussian_sample(rng, parsed, stats, names, scale):
    base = parsed[rng.choice(names)]
    out = []
    for pi, p in enumerate(base):
        mu, sd = np.array(stats[pi]['mean']), np.array(stats[pi]['std'])
        w = mu + scale * sd * rng.standard_normal(len(mu))
        out.append(dict(scale=p['scale'], quat=list(p['quat']), translation=list(p['translation']), weights=w.tolist()))
    return out


def latin_hypercube(rng, n, d=2):
    cut = np.linspace(0, 1, n + 1)
    u = rng.random((n, d)); a = cut[:n]; b = cut[1:n + 1]
    pts = np.empty((n, d))
    for j in range(d):
        pts[:, j] = rng.permutation(u[:, j] * (b - a) + a)
    return pts


def sample_eyes(rng, preset_eyes, jitter=.05):
    if preset_eyes and rng.random() < .5:
        e = json.loads(json.dumps(rng.choice(preset_eyes)))
        for side in ('eye_left', 'eye_right'):
            iris = e[side]['iris']
            for k in ('primary_color_u', 'primary_color_v', 'secondary_color_u', 'secondary_color_v'):
                if isinstance(iris.get(k), (int, float)):
                    iris[k] = float(np.clip(iris[k] + rng.normal(0, jitter), 0, 1))
        return dict(source='preset_jitter', settings=e)
    r = {k: (int(rng.integers(lo, hi + 1)) if k == 'iris_pattern' else float(rng.uniform(lo, hi))) for k, (lo, hi) in EYE_RANGES.items()}
    r['blend_method'] = rng.choice(['Radial', 'Structural'])
    return dict(source='uniform', settings=r)


def build_jobs(presets, *, n_blend=120, n_gauss=350, held_out_frac=.1, render_subset=120, seed=SEED):
    rng = np.random.default_rng(seed); pyrng = random.Random(seed)
    names, parsed, stats = preset_stats(presets)
    preset_eyes = [presets[n]['eyes_settings'] for n in names if isinstance(presets[n].get('eyes_settings'), dict)]
    jobs = []
    for n in names:
        jobs.append(dict(kind='preset', preset=n, coefficients=presets[n]['coefficients']))
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    pyrng.shuffle(pairs)
    for a, b in pairs[:n_blend]:
        t = pyrng.choice((1 / 3, 2 / 3))
        jobs.append(dict(kind='blend', preset=f'{a}+{b}@{t:.2f}', coefficients=flatten_coefficients(blend(parsed[a], parsed[b], t))))
    for i in range(n_gauss):
        s = (.8, 1., 1.2)[i % 3]
        jobs.append(dict(kind='gauss', preset=None, gauss_scale=s, coefficients=flatten_coefficients(gaussian_sample(rng, parsed, stats, names, s))))
    n = len(jobs)
    tone = latin_hypercube(rng, n)
    # texture indices: each of 0..152 at least ceil(n/153) times, 12 held out entirely
    held_tex = sorted(rng.choice(N_TEXTURE_CHARACTERS, 12, replace=False).tolist())
    avail = [i for i in range(N_TEXTURE_CHARACTERS) if i not in held_tex]
    tex = [avail[i % len(avail)] for i in range(n)]
    rng.shuffle(tex)
    ids = [f'mh_{i:06d}' for i in range(n)]
    n_held = max(0, min(len(ids) - len(names), int(held_out_frac * n) - 3))
    held_ids = set(rng.choice(ids[len(names):], n_held, replace=False).tolist()) | set(ids[:3])
    render_ids = set(ids[:len(names)]) | set(rng.choice(ids[len(names):], max(0, min(len(ids) - len(names), render_subset - len(names))), replace=False).tolist())
    for i, j in enumerate(jobs):
        j.update(id=ids[i], seed=int(seed * 1000 + i), held_out=ids[i] in held_ids, render=ids[i] in render_ids,
                 skin=dict(u=float(tone[i, 0]), v=float(tone[i, 1]), face_texture_index=int(tex[i]), roughness=float(rng.uniform(.85, 1.15)),
                           freckles=(dict(density=float(rng.uniform(.1, .6)), strength=float(rng.uniform(.2, .8)), saturation=float(rng.uniform(.5, 1.)),
                                          tone_shift=float(rng.uniform(0, .5)), mask=int(rng.integers(1, 4))) if rng.random() < .4 else None),
                           accents={r: dict(redness=float(np.clip(rng.normal(.5, .1), 0, 1)), saturation=float(np.clip(rng.normal(.5, .1), 0, 1)),
                                            lightness=float(np.clip(rng.normal(.5, .1), 0, 1)))
                                    for r in ('scalp', 'forehead', 'nose', 'under_eye', 'cheeks', 'lips', 'chin', 'ears')}),
                 eyes=sample_eyes(rng, preset_eyes))
    meta = dict(seed=seed, n=n, n_presets=len(names), n_blend=min(n_blend, len(pairs)), n_gauss=n_gauss, held_out_ids=sorted(held_ids),
                held_out_texture_indices=held_tex, render_ids=sorted(render_ids), patch_pca_sizes=[len(p['weights']) for p in parsed[names[0]]])
    return jobs, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--presets', default=str(DATASET_ROOT / 'v1' / 'presets' / 'face_presets.json'))
    ap.add_argument('--out', default=str(DATASET_ROOT / 'v1' / 'jobs.jsonl'))
    ap.add_argument('--n-blend', type=int, default=120); ap.add_argument('--n-gauss', type=int, default=350)
    ap.add_argument('--render-subset', type=int, default=120); ap.add_argument('--seed', type=int, default=SEED)
    a = ap.parse_args()
    presets = json.loads(Path(a.presets).read_text())['presets']
    presets = {k: v for k, v in presets.items() if 'coefficients' in v}
    jobs, meta = build_jobs(presets, n_blend=a.n_blend, n_gauss=a.n_gauss, render_subset=a.render_subset, seed=a.seed)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        for j in jobs:
            f.write(json.dumps(j) + '\n')
    (out.parent / 'sampling.json').write_text(json.dumps(meta, indent=1) + '\n')
    print(f'{len(jobs)} jobs -> {out}; held out {len(meta["held_out_ids"])}, render subset {len(meta["render_ids"])}')


if __name__ == '__main__':
    main()
