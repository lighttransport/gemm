"""render_matrix candidates (real subjects s1/s2) rendered on the Windows worker (Blender CPU) instead of the Linux GPU.

Linux:   python -m server.vhuman.metahuman.win_matrix render <tag> --subjects s1 s2 --candidates B8 --cameras front ...
             [--samples 128] [--resolution 1024] [--suffix w]      (stage + sync + push + run detached; returns)
         python -m server.vhuman.metahuman.win_matrix finish <tag>   (wait, pull into renders/<s>/<cand><suffix>/<view>)
Windows: win_cycles.py run <stage root> (the same runner as win_cycles).

stage: offline_render.render(spec['dir'], ..., stage_only=True) once per subject x candidate with the candidate's
render_matrix options (cycles_scene places named cameras itself, so the base serves every view); one job per camera
with the base request's camera/samples/resolution changed. `suffix` keeps Windows outputs apart from Linux renders of the
same candidate (e.g. B7 -> B7w for the Linux/Windows equivalence check).
"""
import argparse
import json
import shutil
import sys
import tarfile
from pathlib import Path

from . import win_cycles as wc

R = Path(__file__).resolve().parents[3]
W = R / 'tmp/vhuman-ab-cycles'


def stage(subjects, candidates, cameras, root, samples=128, resolution=None, suffix=''):
    from ..reconstruction.offline_render import render
    from .winjob import WIN_REPO
    root = Path(root); jobs = []
    for s in subjects:
        cands = json.loads((W / 'candidates' / s / 'candidates.json').read_text())
        head_fit = cands.get('_head_fit')
        for name in candidates:
            spec = cands[name]; out_name = name + suffix
            todo = [c for c in cameras if not (W / 'renders' / s / out_name / f'{c}__studio__neutral' / 'render_result.json').is_file()]
            if not todo:
                continue
            base = root / s / f'{name}_base'
            if not (base / 'request.json').is_file():
                if base.exists():
                    shutil.rmtree(base)
                render(spec['dir'], base, device='cpu', preset='final', camera=todo[0], lighting='studio',
                       quality=spec.get('quality', 'baseline'), ablate=tuple(spec.get('ablate', ())), eyes=spec.get('eyes'),
                       head_fit=head_fit, appearance=spec.get('appearance'), samples=samples,
                       displacement_scale=spec.get('displacement_scale', 1.), micro_scale=spec.get('micro_scale', 1.),
                       save_blend=False, lip_coat_scale=spec.get('lip_coat_scale', .6),
                       lip_coat_roughness=spec.get('lip_coat_roughness', .30), teeth_glow=spec.get('teeth_glow', 0.),
                       mouth_fill=spec.get('mouth_fill', .02), skin_normal_map=spec.get('skin_normal_map', False),
                       material_overrides=spec.get('material_overrides'), stage_only=True)
            shutil.rmtree(base / 'cache', ignore_errors=True)
            cand_copy = root / s / f'{name}_cand'
            if not cand_copy.exists():
                shutil.copytree(spec['dir'], cand_copy)
            base_req = json.loads((base / 'request.json').read_text())
            for c in todo:
                req = dict(base_req, camera=c, samples=samples, noisy=False, save_blend=True, device='cpu')
                if resolution:
                    req['resolution'] = resolution
                jobs.append(dict(subject=s, candidate=out_name, camera=c, base=f'{s}/{name}_base', candidate_dir=f'{s}/{name}_cand',
                                 out=f'{s}/{out_name}/{c}__studio__neutral', request=req))
    for j in jobs:          # win_cycles.run reads 'candidate' as the candidate directory
        j['candidate'] = j.pop('candidate_dir')
    (root / 'jobs.json').write_text(json.dumps(dict(repo=WIN_REPO, profile='matrix', jobs=jobs), indent=1))
    return len(jobs)


def render_remote(tag, subjects, candidates, cameras, samples, resolution, suffix, timeout=12 * 3600):
    from . import winjob as wj
    root = wj.LOCAL_TMP / tag
    n = stage(subjects, candidates, cameras, root, samples, resolution, suffix)
    print(f'{tag}: {n} views staged', flush=True)
    if not n:
        return tag
    wj.sync()
    tar = wj.LOCAL_TMP / f'{tag}_in.tar'
    with tarfile.open(tar, 'w') as t:
        t.add(root, arcname=tag)
    remote = f'{wj.WIN_DATA}/cycles'
    wj.ssh(f"New-Item -ItemType Directory -Force '{wj.win(remote)}' | Out-Null")
    wj.scp_to(tar, f'{wj.WIN_ROOT}/{tag}_in.tar')
    wj.ssh(f"tar -xf '{wj.win(wj.WIN_ROOT)}\\{tag}_in.tar' -C '{wj.win(remote)}'")
    script = f'{wj.WIN_REPO}/server/vhuman/metahuman/win_cycles.py'
    wj.start(tag, None, timeout, command=f"& '{wj.win(wc.WIN_BLENDER_PY)}' '{wj.win(script)}' run '{wj.win(remote)}/{tag}'")
    return tag


def pull(tag):
    from . import winjob as wj
    remote = f'{wj.WIN_DATA}/cycles/{tag}'
    wj.ssh(f"Set-Location '{wj.win(remote)}'; $f = Get-ChildItem -Recurse -Filter render_result.json | % {{ Resolve-Path -Relative $_.Directory.FullName }}; "
           f"if ($f) {{ tar -cf '{wj.win(wj.WIN_ROOT)}\\{tag}_out.tar' --exclude '*.npz' @($f) progress.log }}")
    local = wj.LOCAL_TMP / f'{tag}_out.tar'
    wj.scp_from(f'{wj.WIN_ROOT}/{tag}_out.tar', local)
    got = wc.extract_views(local, W / 'renders')
    print(f'{tag}: pulled {got} views', flush=True)
    return got


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('cmd', choices=('stage', 'render', 'finish', 'pull'))
    ap.add_argument('tag')
    ap.add_argument('--subjects', nargs='+', default=['s1', 's2']); ap.add_argument('--candidates', nargs='+', default=[])
    ap.add_argument('--cameras', nargs='+', default=['front']); ap.add_argument('--samples', type=int, default=128)
    ap.add_argument('--resolution', type=int); ap.add_argument('--suffix', default='')
    a = ap.parse_args()
    if a.cmd == 'stage':
        from . import winjob as wj
        print(stage(a.subjects, a.candidates, a.cameras, wj.LOCAL_TMP / a.tag, a.samples, a.resolution, a.suffix)); return
    if a.cmd == 'render':
        print(render_remote(a.tag, a.subjects, a.candidates, a.cameras, a.samples, a.resolution, a.suffix)); return
    if a.cmd == 'finish':
        from . import winjob as wj
        print(a.tag, 'exit', wj.wait(a.tag, 12 * 3600), flush=True)
    pull(a.tag)


if __name__ == '__main__':
    sys.exit(main())
