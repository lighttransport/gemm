"""Fast periocular parameter sweeps on the local GPU: build each view's Cycles scene once, render many hypotheses from
it (peri_sweep_blender.py), score them with the standard periocular metric.

PYTHONPATH=$PWD python -m server.vhuman.metahuman.peri_sweep <base variant> <hyps.json> [--ids ...] [--cams ...]
The base variant must have been staged in tmp/win/peri_local (render_local / peri_local.sh). Each hypothesis becomes
variant '<base>~<name>' under COMPARE_ROOT (fast profile), so compare_render.report scores it unchanged.
Env: VHUMAN_BLENDER, VHUMAN_BLENDER_LD_LIBRARY_PATH (HIP libs)."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

ROOT_NAME = 'peri_local'
REPO = Path(__file__).resolve().parents[3]


def build(job, root, dest, blender, env):
    """Build and pack a scene on CPU without rendering; capture/sweep selects the GPU later."""
    digest = hashlib.sha256(json.dumps(job, sort_keys=True).encode())
    inputs = list((root / job['base']).glob('*')) + list((root / job['candidate']).glob('*'))
    inputs += [REPO / 'server/vhuman/reconstruction' / name
               for name in ('cycles_scene.py', 'periocular.py')]
    for path in sorted(inputs):
        if path.is_file() and path.name not in ('head.blend', 'render_result.json'):
            digest.update(str(path).encode())
            with path.open('rb') as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b''):
                    digest.update(chunk)
    stamp = digest.hexdigest()
    stamp_path = dest / 'build.sha256'
    if (dest / 'head.blend').is_file() and stamp_path.is_file() and stamp_path.read_text() == stamp:
        return dest / 'head.blend'
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(root / job['base'], dest, ignore=lambda d, n: [x for x in n if x in ('cache', 'request.json', 'head.blend', 'render_result.json')])
    scene = json.loads((dest / 'scene.json').read_text()); scene['candidate'] = str(root / job['candidate'])
    (dest / 'scene.json').write_text(json.dumps(scene, indent=1))
    req = dict(job['request'], out=str(dest), device='cpu', save_blend=True, scene_only=True)
    (dest / 'request.json').write_text(json.dumps(req, indent=2))
    with (dest / 'build.log').open('w') as log:
        subprocess.check_call([blender, '-b', '--factory-startup', '--python-exit-code', '1', '--python',
                               str(REPO / 'server/vhuman/reconstruction/cycles_scene.py'), '--', str(dest / 'request.json')],
                              stdout=log, stderr=subprocess.STDOUT, env=env)
    stamp_path.write_text(stamp)
    return dest / 'head.blend'


def score_fast(variants, ids, cams, profile='fast', env='vhstudio'):
    """Periocular-only scores (same exposure match, display transform and compare_region call as compare_render.report,
    region 'periocular') for many variants in one process; face-parser predictions are memoised by image content, so each
    UE reference is parsed once. Returns {variant: {(sid, cam): dict(lpips, dists, delta_e)}}."""
    import hashlib
    import numpy as np
    from . import compare_render as cr
    from .images import read_exr
    from .samples import V1
    from .win_cycles import variant_dir
    from ..face_parsing import FaceParser
    from ..reconstruction import ab_metrics as am
    parser = FaceParser(threads=16)
    memo = {}
    predict = parser.predict
    def cached(rgb):
        k = hashlib.sha1(np.ascontiguousarray(rgb).tobytes()).hexdigest()
        if k not in memo:
            memo[k] = predict(rgb)
        return memo[k]
    parser.predict = cached
    prefixes, labels = cr.REGION_METRICS['periocular']; margin = cr.REGION_MARGIN.get('periocular', .25)
    disp = lambda x: np.uint8(np.clip(np.where(x <= .0031308, 12.92 * x, 1.055 * np.clip(x, 0, None) ** (1 / 2.4) - .055), 0, 1) * 255 + .5)
    ue_cache = {}; out = {}
    for v in variants:
        res = {}
        for sid in ids:
            for cam in cams:
                c = cr.COMPARE_ROOT / sid / variant_dir(v, profile) / cam / 'beauty.exr'
                if not c.is_file():
                    continue
                cy = read_exr(c)
                key = (sid, cam, cy.shape[0])
                if key not in ue_cache:
                    u = read_exr(V1 / 'samples' / sid / 'renders' / f'{cam}__{env}.exr')[..., :3]
                    ub = read_exr(V1 / 'samples' / sid / 'renders' / f'{cam}__basecolor.exr')[..., :3]
                    ue_cache[key] = (cr.box_down(u, cy.shape[0]), cr.box_down(ub, cy.shape[0]))
                u, ubc = ue_cache[key]
                um = ubc.max(-1) > 1e-4
                cm = cy[..., 3] > .5 if cy.shape[-1] == 4 else cy[..., :3].max(-1) > 1e-4
                both = cm & um
                gain = float(np.median(u[both].mean(-1)) / max(np.median(cy[..., :3][both].mean(-1)), 1e-6)) if both.any() else 1.
                k = .6 / max(float(np.percentile(u[um], 95)), 1e-6) if um.any() else 1.
                a, b = disp(u * k), disp(cy[..., :3] * gain * k)
                e = am.compare_region(a, b, parser, list(labels), aligned=True, margin=margin)
                res[(sid, cam)] = dict(lpips=e.get('lpips'), dists=e.get('dists'), delta_e=(e.get('delta_e') or {}).get('mean'))
        out[v] = res
        vals = np.array([[r['lpips'], r['dists'], r['delta_e']] for r in res.values() if None not in r.values()], float)
        if len(vals):
            print(f'{v}: n={len(vals)} peri_lpips {vals[:, 0].mean():.4f} dists {vals[:, 1].mean():.4f} dE {vals[:, 2].mean():.3f}  '
                  + ' '.join(f'{r["lpips"]:.4f}' if r['lpips'] is not None else 'unavailable' for r in res.values()), flush=True)
    return out


def main():
    from . import compare_render as cr, winjob as wj
    from .win_cycles import variant_dir
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('base'); ap.add_argument('hyps')
    ap.add_argument('--ids', default='mh_000020,mh_000100,mh_000038,mh_000300')
    ap.add_argument('--cams', default='eye_left,eye_right')
    ap.add_argument('--score', action='store_true')
    a = ap.parse_args()
    if a.score:      # score only (hyps file names the variants <base>~<name>)
        names = [f"{a.base}~{h['name']}" for h in json.loads(Path(a.hyps).read_text())]
        score_fast(names, a.ids.split(','), a.cams.split(','))
        return
    blender = os.environ.get('VHUMAN_BLENDER', '/mnt/nvme02/tmp/blender-5.2.3-linux-x64/blender')
    env = dict(os.environ)
    if os.environ.get('VHUMAN_BLENDER_LD_LIBRARY_PATH'):
        env['LD_LIBRARY_PATH'] = os.environ['VHUMAN_BLENDER_LD_LIBRARY_PATH']
    hyps = json.loads(Path(a.hyps).read_text())
    root = wj.LOCAL_TMP / ROOT_NAME
    spec = json.loads((root / 'jobs.json').read_text())
    sweep = Path('/mnt/nvme02/tmp/vhuman-sweep') / a.base
    ids, cams = a.ids.split(','), a.cams.split(',')
    jobs = {(j['sid'], j['camera']): j for j in spec['jobs'] if j['variant'] == a.base}
    t_build = t_render = 0.
    for sid in ids:
        for cam in cams:
            job = jobs.get((sid, cam))
            if job is None:
                raise SystemExit(f'{a.base} {sid} {cam} is not staged in {root}/jobs.json (run peri_local.sh first)')
            dest = sweep / sid / cam
            t0 = time.time(); blend = build(job, root, dest / 'build', blender, env); t_build += time.time() - t0
            t0 = time.time()
            with (dest / 'sweep.log').open('w') as log:
                subprocess.check_call([blender, '-b', '--factory-startup', str(blend), '--python-exit-code', '1', '--python',
                                       str(REPO / 'server/vhuman/metahuman/peri_sweep_blender.py'), '--', str(Path(a.hyps).resolve()),
                                       str(dest / 'hyps')], stdout=log, stderr=subprocess.STDOUT, env=env)
            t_render += time.time() - t0
            for h in hyps:
                v = f"{a.base}~{h['name']}"
                out = cr.COMPARE_ROOT / sid / variant_dir(v, 'fast') / cam
                if out.exists() and not out.is_symlink():
                    shutil.rmtree(out)
                target = Path('/mnt/nvme02/tmp/vhuman-compare') / sid / variant_dir(v, 'fast') / cam
                target.mkdir(parents=True, exist_ok=True)
                if not (cr.COMPARE_ROOT / sid / variant_dir(v, 'fast')).exists():
                    (cr.COMPARE_ROOT / sid / variant_dir(v, 'fast')).symlink_to(target.parent)
                shutil.copy2(dest / 'hyps' / f"{h['name']}.exr", target / 'beauty.exr')
                (target / 'render_result.json').write_text(json.dumps(dict(sweep=a.base, hypothesis=h, worker='local-hip-sweep')))
    print(f'build {t_build:.0f}s, sweep renders {t_render:.0f}s for {len(ids) * len(cams)} views x {len(hyps)} hypotheses', flush=True)


if __name__ == '__main__':
    main()
