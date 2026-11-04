"""Cycles renders of the compare_render pseudo-candidates on the Windows worker (Blender CPU; the RX 570 has no HIP).

Linux:   python -m server.vhuman.metahuman.win_cycles stage  [--ids ...] [--variants ...] [--cameras ...] [--profile fast]
         python -m server.vhuman.metahuman.win_cycles render [same options]   (stage + push + run detached + wait + pull)
         python -m server.vhuman.metahuman.win_cycles pull <tag>
         python -m server.vhuman.metahuman.win_cycles status|resume|finish <tag>   (disconnectable: see below)
Windows: <blender python> win_cycles.py run <stage root>   (stdlib only; started by `render` through winjob)

stage: offline_render.render(..., stage_only=True) once per head (prepare + skin templates + wrinkle channels are
camera- and material-independent), plus one request per variant x camera with the profile's resolution/samples, the
explicit camera scaled to that resolution and device 'cpu'. Outputs land in COMPARE_ROOT/<id>/<variant dir>/<camera>
like Linux renders; profile 'fast' (512 px, 32 spp, denoised) uses separate '__fast' directories so they are only
ever paired with each other (compare_render.report(profile=...)).

Disconnectable: the worker job is a detached WMI process and only reads its staged copy, so the Linux host may log out
or reboot while it runs. `render --score <set>` records the ids/profile/score set in tmp/win/<tag>.finish.json;
`finish <tag>` later waits for the exit file (immediately if done), pulls and scores; `status <tag>` prints the
progress; `resume <tag>` restarts an interrupted worker run on the existing stage (finished views are skipped).
Never `sync` (it replaces D:/work/gemm-vhuman/server) while a worker render is running.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROFILES = dict(full=dict(resolution=1024, samples=128), fast=dict(resolution=512, samples=32))
BASE_SKIP = ('cache', 'request.json', 'head.blend', 'render_result.json')


def variant_dir(variant, profile='full'):
    return 'cycles' + (f'_{variant}' if variant else '') + ('' if profile == 'full' else f'__{profile}')


# ----------------------------------------------------------------------------------------------- Windows side
def run(root, blender):
    """Render every job of <root>/jobs.json not yet done (render_result.json present). One Blender at a time, all cores."""
    root = Path(root); spec = json.loads((root / 'jobs.json').read_text())
    repo = Path(spec['repo']); scene_py = repo / 'server/vhuman/reconstruction/cycles_scene.py'
    validate_py = repo / 'server/vhuman/reconstruction/cycles_validate.py'
    progress = (root / 'progress.log').open('a')
    for i, job in enumerate(spec['jobs']):
        out = root / job['out']
        if (out / 'render_result.json').is_file():
            continue
        t0 = time.time()
        if out.exists():
            shutil.rmtree(out)
        base = root / job['base']
        shutil.copytree(base, out, ignore=lambda d, names: [n for n in names if n in BASE_SKIP])
        scene = json.loads((out / 'scene.json').read_text()); scene['candidate'] = str(root / job['candidate'])
        (out / 'scene.json').write_text(json.dumps(scene, indent=1))
        req = dict(job['request'], out=str(out))
        (out / 'request.json').write_text(json.dumps(req, indent=2))
        env = dict(os.environ, TMP=str(out / 'cache'), TEMP=str(out / 'cache'))
        (out / 'cache').mkdir(exist_ok=True)
        with (out / 'cycles.log').open('w') as log:
            rc = subprocess.call([blender, '-b', '--factory-startup', '--python-exit-code', '1', '--python', str(scene_py),
                                  '--', str(out / 'request.json')], stdout=log, stderr=subprocess.STDOUT, env=env)
            if rc == 0:
                rc = subprocess.call([blender, '-b', str(out / 'head.blend'), '--python-exit-code', '1', '--python',
                                      str(validate_py), '--', str(out)], stdout=log, stderr=subprocess.STDOUT, env=env)
        if rc != 0:
            progress.write(f'{job["out"]} FAILED rc {rc}\n'); progress.flush()
            (out / 'render_result.json').unlink(missing_ok=True)
            continue
        result = json.loads((out / 'render_result.json').read_text())
        validation = json.loads((out / 'asset_validation.json').read_text())
        result.update(asset_reload_validated=validation['passed'], float_map_storage=validation['storage'], blend=None,
                      elapsed_seconds=time.time() - t0, render_seconds=result.get('seconds'), memory_measurement='CPU',
                      worker='windows-cpu')
        for name in ('head.blend', 'beauty_noisy.exr', 'skin_detail.npz', 'scene_assets.npz', 'skin_templates.npz'):
            (out / name).unlink(missing_ok=True)
        shutil.rmtree(out / 'cache', ignore_errors=True)
        (out / 'render_result.json').write_text(json.dumps(result, indent=2))
        progress.write(f'{i + 1}/{len(spec["jobs"])} {job["out"]} {time.time() - t0:.1f}s\n'); progress.flush()
    progress.write('done\n'); progress.close()


# ----------------------------------------------------------------------------------------------- Linux side
def view_request(base_req, cam, shift, camera, opts, prof, micro_scale):
    """One view's request: the head's staged request with the profile's size/samples, the dataset camera scaled to
    that size (focal_px x res / size; origin shifted like compare_render.render_sample), CPU, and the variant's options."""
    k = prof['resolution'] / cam['size'][0]
    req = dict(base_req, device='cpu', samples=prof['samples'], resolution=prof['resolution'], noisy=False, save_blend=True,
               camera=camera, explicit_camera=dict(origin=[float(a) + float(b) for a, b in zip(cam['origin'], shift)],
                                                   rotation=cam['rotation'], focal_px=float(cam['focal_px']) * k,
                                                   size=[prof['resolution']] * 2),
               skin_normal_map=bool(opts.get('skin_normal_map', False)),
               skin_normal_strength=float(opts.get('skin_normal_strength', 1.)),
               micro_scale=float(opts.get('micro_scale', micro_scale)),
               material_overrides=dict(opts.get('material_overrides') or {}))
    if not req['skin_normal_map']:
        req.pop('skin_normal_sha256', None)
    return req


def extract_views(tar_path, dest_root):
    """Unpack pulled '<id>/<variant dir>/<camera>/<file>' members (Windows '.\\' prefixes and backslashes accepted) into
    dest_root; views that already have a render_result.json are left alone; each view's result is written last."""
    import tarfile
    got = 0
    with tarfile.open(tar_path) as t:
        views = {}
        for m in t.getmembers():
            parts = Path(m.name.replace('\\', '/')).parts
            parts = parts[1:] if parts and parts[0] == '.' else parts
            if m.isfile() and len(parts) == 4 and '..' not in parts:
                views.setdefault(Path(*parts[:3]), []).append((parts[3], m))
        for view, members in views.items():
            dest = Path(dest_root) / view
            if (dest / 'render_result.json').is_file():
                continue
            dest.mkdir(parents=True, exist_ok=True)
            for name, m in sorted(members, key=lambda x: x[0] == 'render_result.json'):
                with t.extractfile(m) as src, open(dest / name, 'wb') as f:
                    shutil.copyfileobj(src, f)
            got += 1
    return got


def stage(ids, variants, cameras, profile, root):
    """Stage inputs under root/<id>/{cand,base} and the job list; returns the number of jobs (finished views skipped)."""
    import numpy as np
    from ..reconstruction.offline_render import render
    from . import compare_render as cr
    from .samples import V1
    from .winjob import WIN_REPO
    prof = PROFILES[profile]; root = Path(root); jobs = []
    for sid in ids:
        cand = cr.build_candidate(sid)
        todo = [(v, c) for v in variants for c in cameras
                if not (cr.COMPARE_ROOT / sid / variant_dir(v, profile) / c / 'render_result.json').is_file()]
        if not todo:
            continue
        if (root / sid / 'cand').exists():
            shutil.rmtree(root / sid / 'cand')
        shutil.copytree(cand, root / sid / 'cand')
        shift = np.asarray(json.loads((cand / 'manifest.json').read_text())['geometry']['metahuman_pseudo_candidate']['translation_m'])
        bases = {}
        for key in sorted({cr.eye_version(v) for v, _ in todo}):     # 0 = template eyes; n = eye_map version n
            base = root / sid / ('base' if not key else f'base_eyes{key}')
            if not (base / 'request.json').is_file():
                if base.exists():
                    shutil.rmtree(base)
                opts = dict(cr.RENDER, **cr.variant_options('normal')); opts.pop('preset'); opts.pop('samples')
                opts.pop('material_overrides', None)
                if key:
                    opts['head_fit'] = cr.write_eye_fit(sid, key)
                render(cand, base, device='cpu', preset='final', stage_only=True, **opts)
            shutil.rmtree(base / 'cache', ignore_errors=True)
            bases[key] = (base.name, json.loads((base / 'request.json').read_text()))
        cams = json.loads((V1 / 'samples' / sid / 'cameras.json').read_text())['cameras']
        for v, c in todo:
            base_name, base_req = bases[cr.eye_version(v)]
            req = view_request(base_req, cams[c]['vhuman'], shift, c, cr.variant_options(v), prof, cr.RENDER['micro_scale'])
            jobs.append(dict(sid=sid, variant=v, camera=c, base=f'{sid}/{base_name}', candidate=f'{sid}/cand',
                             out=f'{sid}/{variant_dir(v, profile)}/{c}', request=req))
    (root / 'jobs.json').write_text(json.dumps(dict(repo=WIN_REPO, profile=profile, jobs=jobs), indent=1))
    return len(jobs)


SCORE_SETS = ('confirm', 'ears', 'eyes', 'eye_sweep', 'ue_match', 'ue_confirm')


def score_set(name, ids, profile, env='vhstudio'):
    """Run the compare_render scoring of a named experiment (CPU: set HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES=)."""
    from . import compare_render as cr
    exp = dict(confirm={}, ears=cr.EAR_SET, eyes=cr.EYE_SET, eye_sweep=cr.EYE_SWEEP_SET, ue_match=cr.UE_SET, ue_confirm=cr.UE_CONFIRM_SET)[name]
    out = cr.confirm(ids, env, score=True, profile=profile, name=name, **exp)
    for v, r in out['variants'].items():
        print(v, json.dumps({k: x for k, x in r.items() if k != 'per_head_dlpips'}), flush=True)
    return out


def finish_record(tag):
    from . import winjob as wj
    return wj.LOCAL_TMP / f'{tag}.finish.json'


def status(tag):
    from . import winjob as wj
    remote = wj.win(f'{wj.WIN_DATA}/cycles/{tag}')
    out = wj.ssh(f"$j = (Get-Content '{remote}\\jobs.json' -Raw | ConvertFrom-Json).jobs.Count; "
                 f"$d = (Get-ChildItem '{remote}' -Recurse -Filter render_result.json -ErrorAction SilentlyContinue | Measure-Object).Count; "
                 f"\"views $d/$j\"; Get-Content '{remote}\\progress.log' -Tail 2 -ErrorAction SilentlyContinue", check=False)
    return out + '\n' + (wj.status(tag) or 'no exit file (running or never started)')


def resume(tag, timeout=8 * 3600, force=False):
    """Restart the worker run of an already staged tag (after a worker reboot / kill); finished views are skipped."""
    from . import winjob as wj
    if not force and not wj.status(tag).startswith('exit'):
        raise RuntimeError(f'{tag}: no exit file, the worker run may still be going (status {tag}); use --force')
    remote = f'{wj.WIN_DATA}/cycles'
    script = f'{wj.WIN_REPO}/server/vhuman/metahuman/win_cycles.py'
    return wj.start(tag, None, timeout, command=f"& '{wj.win(WIN_BLENDER_PY)}' '{wj.win(script)}' run '{wj.win(remote)}/{tag}'")


def finish(tag, score=None, wait=True, timeout=8 * 3600):
    """Wait for the worker (if wait), pull and score with the set recorded by `render --score` (or the given one)."""
    from . import winjob as wj
    rec = json.loads(finish_record(tag).read_text()) if finish_record(tag).is_file() else {}
    if wait:
        print(tag, 'exit', wj.wait(tag, timeout), flush=True)
    pull(tag)
    name = score or rec.get('score')
    if name:
        return score_set(name, rec['ids'], rec.get('profile', 'fast'), rec.get('env', 'vhstudio'))
    return None


def render_remote(ids, variants, cameras, profile, tag=None, timeout=8 * 3600, sync=True, score=None):
    import tarfile
    from . import winjob as wj
    tag = tag or f"cycles_{profile}_{time.strftime('%Y%m%d-%H%M%S')}"
    root = wj.LOCAL_TMP / tag
    wj.LOCAL_TMP.mkdir(parents=True, exist_ok=True)
    finish_record(tag).write_text(json.dumps(dict(tag=tag, ids=list(ids), variants=list(variants), cameras=list(cameras),
                                                  profile=profile, score=score, env='vhstudio'), indent=1) + '\n')
    n = stage(ids, variants, cameras, profile, root)
    print(f'{tag}: {n} views staged', flush=True)
    if not n:
        return tag
    if sync:
        wj.sync()
    tar = wj.LOCAL_TMP / f'{tag}_in.tar'
    with tarfile.open(tar, 'w') as t:
        t.add(root, arcname=tag)
    remote = f'{wj.WIN_DATA}/cycles'
    wj.ssh(f"New-Item -ItemType Directory -Force '{wj.win(remote)}' | Out-Null")
    wj.scp_to(tar, f'{wj.WIN_ROOT}/{tag}_in.tar')
    wj.ssh(f"tar -xf '{wj.win(wj.WIN_ROOT)}\\{tag}_in.tar' -C '{wj.win(remote)}'")
    script = f'{wj.WIN_REPO}/server/vhuman/metahuman/win_cycles.py'
    wj.start(tag, None, timeout, command=f"& '{wj.win(WIN_BLENDER_PY)}' '{wj.win(script)}' run '{wj.win(remote)}/{tag}'")
    return tag


def pull(tag):
    """Fetch finished views of a staged job into COMPARE_ROOT/<id>/<variant dir>/<camera> (never overwrites)."""
    from . import winjob as wj
    from .compare_render import COMPARE_ROOT
    remote = f'{wj.WIN_DATA}/cycles/{tag}'
    wj.ssh(f"Set-Location '{wj.win(remote)}'; $f = Get-ChildItem -Recurse -Filter render_result.json | % {{ Resolve-Path -Relative $_.Directory.FullName }}; "
           f"if ($f) {{ tar -cf '{wj.win(wj.WIN_ROOT)}\\{tag}_out.tar' --exclude '*.npz' @($f) progress.log }}")
    local = wj.LOCAL_TMP / f'{tag}_out.tar'
    wj.scp_from(f'{wj.WIN_ROOT}/{tag}_out.tar', local)
    got = extract_views(local, COMPARE_ROOT)
    print(f'{tag}: pulled {got} views', flush=True)
    return got


WIN_BLENDER = 'D:/local/blender-5.2.3-windows-x64/blender.exe'
WIN_BLENDER_PY = 'D:/local/blender-5.2.3-windows-x64/5.2/python/bin/python.exe'


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == 'run':      # Windows, stdlib only
        run(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else WIN_BLENDER); return
    import argparse
    from . import compare_render as cr
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('cmd', choices=('stage', 'render', 'pull', 'status', 'resume', 'finish'))
    ap.add_argument('tag', nargs='?', default='')
    ap.add_argument('--ids', default=''); ap.add_argument('--variants', default=','.join(('normal',) + cr.CONFIRM))
    ap.add_argument('--cameras', default=','.join(cr.CONFIRM_CAMERAS)); ap.add_argument('--profile', default='fast', choices=PROFILES)
    ap.add_argument('--timeout', type=float, default=8 * 3600); ap.add_argument('--no-sync', action='store_true')
    ap.add_argument('--score', choices=SCORE_SETS, help='render: scoring set recorded for `finish`; finish: override it')
    ap.add_argument('--no-wait', action='store_true', help='finish: pull/score what is done now')
    ap.add_argument('--force', action='store_true', help='resume: restart even without an exit file')
    a = ap.parse_args()
    if a.cmd == 'pull':
        pull(a.tag); return
    if a.cmd == 'status':
        print(status(a.tag)); return
    if a.cmd == 'resume':
        resume(a.tag, a.timeout, a.force); return
    if a.cmd == 'finish':
        finish(a.tag, a.score, not a.no_wait, a.timeout); return
    from .samples import V1
    ids = a.ids.split(',') if a.ids else sorted(json.loads((V1 / 'sampling.json').read_text())['render_ids'])[::10]
    variants, cams = a.variants.split(','), a.cameras.split(',')
    if a.cmd == 'stage':
        from . import winjob as wj
        print(stage(ids, variants, cams, a.profile, wj.LOCAL_TMP / (a.tag or 'stage')), 'views staged'); return
    print(render_remote(ids, variants, cams, a.profile, a.tag or None, a.timeout, not a.no_sync, a.score))


if __name__ == '__main__':
    main()
