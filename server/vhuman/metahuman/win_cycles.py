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


def _eye2_current(base):
    """Both eyes' layered-eye assets exist and match the current eye2 ASSET_VERSION."""
    from ..eye2.bake import ASSET_VERSION
    try:
        return all(json.loads((base / f'eye2_{s}.json').read_text()).get('version') == ASSET_VERSION for s in ('left', 'right'))
    except (OSError, ValueError):
        return False


def render_cached(ids, variants, cameras, profile, root_name='peri_cache', timeout=8 * 3600):
    """Like render_remote, but with one persistent stage root (bases/eye2 assets reused across rounds) and an incremental
    push: only files changed since the last push are uploaded. Returns the job tag (= root_name). The worker skips views
    that already have a result, so every hypothesis needs its own variant name."""
    import tarfile
    from . import winjob as wj
    root = wj.LOCAL_TMP / root_name
    root.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n = stage(ids, variants, cameras, profile, root)
    print(f'{root_name}: {n} views staged in {time.time() - t0:.0f}s', flush=True)
    if not n:
        return root_name
    wj.sync()
    pushed_path = root / '.pushed.json'
    pushed = json.loads(pushed_path.read_text()) if pushed_path.is_file() else {}
    files = {}
    for p in root.rglob('*'):
        if p.is_file() and p.name != '.pushed.json':
            st = p.stat(); files[str(p.relative_to(root))] = [st.st_size, int(st.st_mtime)]
    changed = [k for k, v in files.items() if pushed.get(k) != v]
    tar = wj.LOCAL_TMP / f'{root_name}_in.tar'
    with tarfile.open(tar, 'w') as t:
        for k in changed:
            t.add(root / k, arcname=f'{root_name}/{k}')
    print(f'{root_name}: pushing {len(changed)} changed files ({tar.stat().st_size / 1e6:.1f} MB)', flush=True)
    remote = f'{wj.WIN_DATA}/cycles'
    wj.ssh(f"New-Item -ItemType Directory -Force '{wj.win(remote)}' | Out-Null")
    wj.scp_to(tar, f'{wj.WIN_ROOT}/{root_name}_in.tar')
    wj.ssh(f"tar -xf '{wj.win(wj.WIN_ROOT)}\\{root_name}_in.tar' -C '{wj.win(remote)}'")
    tar.unlink()
    pushed_path.write_text(json.dumps(files))
    script = f'{wj.WIN_REPO}/server/vhuman/metahuman/win_cycles.py'
    wj.start(root_name, None, timeout, command=f"& '{wj.win(WIN_BLENDER_PY)}' '{wj.win(script)}' run '{wj.win(remote)}/{root_name}'")
    return root_name


def render_local(ids, variants, cameras, profile, root_name='peri_local', device='hip'):
    """Stage into a persistent local root and render on the local GPU with the same job runner as the worker
    (VHUMAN_BLENDER / VHUMAN_BLENDER_LD_LIBRARY_PATH select the Blender and its HIP libraries); finished views are
    copied into COMPARE_ROOT like a pull. Returns the number of views rendered."""
    from . import winjob as wj
    from .compare_render import COMPARE_ROOT
    from ..reconstruction.offline_render import DEFAULT_BLENDER
    root = wj.LOCAL_TMP / root_name
    root.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n = stage(ids, variants, cameras, profile, root)
    print(f'{root_name}: {n} views staged in {time.time() - t0:.0f}s', flush=True)
    if not n:
        return 0
    spec = json.loads((root / 'jobs.json').read_text())
    repo = Path(__file__).resolve().parents[3]
    spec['repo'] = str(repo)
    for job in spec['jobs']:
        job['request'] = dict(job['request'], device=device)
    (root / 'jobs_local.json').write_text(json.dumps(spec, indent=1))
    env = {}
    if os.environ.get('VHUMAN_BLENDER_LD_LIBRARY_PATH'):
        env['LD_LIBRARY_PATH'] = os.environ['VHUMAN_BLENDER_LD_LIBRARY_PATH']
    t1 = time.time()
    run(root, str(DEFAULT_BLENDER), 'jobs_local.json', env, worker=f'local-{device}')
    print(f'{root_name}: rendered in {time.time() - t1:.0f}s', flush=True)
    got = 0
    for job in spec['jobs']:
        src = root / job['out']
        dst = COMPARE_ROOT / job['out']
        if (src / 'render_result.json').is_file() and not (dst / 'render_result.json').is_file():
            dst.mkdir(parents=True, exist_ok=True)
            for p in src.iterdir():
                if p.is_file() and not p.name.startswith(('eye2_', 'eye_unified', 'sclera_vessels', 'eye_surface',
                                                          'eye_shell', 'eye_tissue', 'skin_normal_map')):
                    shutil.copy2(p, dst / p.name)
            got += 1
    print(f'{root_name}: copied {got} views', flush=True)
    return got


def pull_jobs(root_name):
    """Pull only the views listed in the root's current jobs.json (not every view ever rendered in a persistent root)."""
    from . import winjob as wj
    from .compare_render import COMPARE_ROOT
    jobs = json.loads((wj.LOCAL_TMP / root_name / 'jobs.json').read_text())['jobs']
    outs = ','.join(f"'./{j['out']}'" for j in jobs)
    remote = f'{wj.WIN_DATA}/cycles/{root_name}'
    wj.ssh(f"Set-Location '{wj.win(remote)}'; $f = @({outs}) | ? {{ Test-Path (Join-Path $_ 'render_result.json') }}; "
           f"if ($f) {{ tar -cf '{wj.win(wj.WIN_ROOT)}\\{root_name}_out.tar' --exclude '*.npz' --exclude 'eye2_*.png' "
           f"--exclude 'eye_unified*.png' --exclude 'sclera_vessels*.png' --exclude 'eye_surface*.png' --exclude 'eye_shell*.png' "
           f"--exclude 'eye_tissue*.png' --exclude 'skin_normal_map.png' @($f) }}")
    local = wj.LOCAL_TMP / f'{root_name}_out.tar'
    wj.scp_from(f'{wj.WIN_ROOT}/{root_name}_out.tar', local)
    got = extract_views(local, COMPARE_ROOT)
    local.unlink(missing_ok=True)
    print(f'{root_name}: pulled {got} views', flush=True)
    return got


def wj_local_tmp():
    from . import winjob
    return winjob.LOCAL_TMP


def variant_dir(variant, profile='full'):
    return 'cycles' + (f'_{variant}' if variant else '') + ('' if profile == 'full' else f'__{profile}')


# ----------------------------------------------------------------------------------------------- Windows side
def run(root, blender, jobs_name='jobs.json', env_extra=None, worker='windows-cpu'):
    """Render every job of <root>/<jobs_name> not yet done (render_result.json present). One Blender at a time."""
    root = Path(root); spec = json.loads((root / jobs_name).read_text())
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
        profile = None
        if (req.get('material_overrides') or {}).get('skin_lighting_profile', 0):
            # The standalone worker also supports the package's optional profile.
            if str(repo) not in sys.path:
                sys.path.insert(0, str(repo))
            from server.vhuman.reconstruction import skin_profile as profile
            profile.preflight(req)
        (out / 'request.json').write_text(json.dumps(req, indent=2))
        env = dict(os.environ, TMP=str(out / 'cache'), TEMP=str(out / 'cache'), **(env_extra or {}))
        (out / 'cache').mkdir(exist_ok=True)
        with (out / 'cycles.log').open('w') as log:
            rc = subprocess.call([blender, '-b', '--factory-startup', '--python-exit-code', '1', '--python', str(scene_py),
                                  '--', str(out / 'request.json')], stdout=log, stderr=subprocess.STDOUT, env=env)
            if rc == 0:
                rc = subprocess.call([blender, '-b', '--factory-startup', str(out / 'head.blend'), '--python-exit-code', '1', '--python',
                                      str(validate_py), '--', str(out)], stdout=log, stderr=subprocess.STDOUT, env=env)
        if rc != 0:
            progress.write(f'{job["out"]} FAILED rc {rc}\n'); progress.flush()
            (out / 'render_result.json').unlink(missing_ok=True)
            continue
        if profile is not None:
            try:
                usage = None
                if req.get('device') == 'hip':
                    paths = sorted(Path('/sys/class/drm').glob('card[0-9]*/device/mem_info_vram_used'))
                    if not paths:
                        raise RuntimeError('HIP profile requires the device memory monitor')
                    usage = lambda: max(int(p.read_text()) / 1024**2 for p in paths)
                profile.finish(req, blender, env=env, memory_usage=usage)
            except Exception:
                (out / 'render_result.json').unlink(missing_ok=True)
                raise
        result = json.loads((out / 'render_result.json').read_text())
        validation = json.loads((out / 'asset_validation.json').read_text())
        result.update(asset_reload_validated=validation['passed'], float_map_storage=validation['storage'], blend=None,
                      elapsed_seconds=time.time() - t0, render_seconds=result.get('seconds'), memory_measurement='CPU',
                      worker=worker)
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
    mo = dict(opts.get('material_overrides') or {})
    mo.setdefault('skin_lighting_profile',0.)  # historical variants remain frozen
    side = camera.removeprefix('eye_') if camera in ('eye_left', 'eye_right') else ''
    zoom = float(mo.get(f'eye_zoom_{side}', mo.get('eye_zoom', 1.)) if side else mo.get('eye_zoom', 1.))
    if zoom <= 0:
        raise ValueError('eye_zoom must be positive')
    req = dict(base_req, device='cpu', samples=prof['samples'], resolution=prof['resolution'], noisy=False, save_blend=True,
               camera=camera, explicit_camera=dict(origin=[float(a) + float(b) for a, b in zip(cam['origin'], shift)],
                                                   rotation=cam['rotation'], focal_px=float(cam['focal_px']) * k * zoom,
                                                   size=[prof['resolution']] * 2),
               skin_normal_map=bool(opts.get('skin_normal_map', False)),
               skin_normal_strength=float(opts.get('skin_normal_strength', 1.)),
               micro_scale=float(opts.get('micro_scale', micro_scale)),
               material_overrides=mo)
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
        if not (root / sid / 'cand' / 'manifest.json').is_file():     # persistent roots keep their copy (cached push)
            shutil.rmtree(root / sid / 'cand', ignore_errors=True)
            shutil.copytree(cand, root / sid / 'cand')
        shift = np.asarray(json.loads((cand / 'manifest.json').read_text())['geometry']['metahuman_pseudo_candidate']['translation_m'])
        bases = {}
        for key in sorted({cr.eye_version(v) for v, _ in todo}):     # 0 = template eyes; n = eye_map version n
            base = root / sid / ('base' if not key else f'base_eyes{key}')
            want_geo = sorted(g for g in ('peri_caruncle', 'peri_meniscus', 'peri_margin')
                              if any(cr.variant_options(v, sid).get('material_overrides', {}).get(g, 0.) >= .5
                                     for v, _ in todo if cr.eye_version(v) == key))
            stamp = base / 'peri_base.json'
            have_geo = json.loads(stamp.read_text())['peri_geo'] if stamp.is_file() else []
            if (base / 'request.json').is_file() and not set(want_geo) <= set(have_geo):
                shutil.rmtree(base)       # persistent base lacks requested periocular parts: rebuild (with the union)
                want_geo = sorted(set(want_geo) | set(have_geo))
            if not (base / 'request.json').is_file():
                if base.exists():
                    shutil.rmtree(base)
                opts = dict(cr.RENDER, **cr.variant_options('normal')); opts.pop('preset'); opts.pop('samples')
                opts.pop('material_overrides', None)
                peri_geo = dict(skin_lighting_profile=0., **{g: 1. for g in want_geo})
                if peri_geo:     # extra periocular parts; cycles_scene renders them only for variants that enable them
                    opts['material_overrides'] = peri_geo
                if key:
                    opts['head_fit'] = cr.write_eye_fit(sid, key)
                render(cand, base, device='cpu', preset='final', stage_only=True, **opts)
                stamp.write_text(json.dumps(dict(peri_geo=want_geo)) + '\n')
            if any(cr.variant_options(v, sid).get('material_overrides', {}).get('skin_micro_isotropic')
                   for v, _ in todo if cr.eye_version(v) == key):
                from ..reconstruction.skin_templates import ensure_isotropic_pores
                ensure_isotropic_pores(base)
            if any(cr.variant_options(v, sid).get('material_overrides', {}).get('skin_lighting_profile', 0)
                   for v, _ in todo if cr.eye_version(v) == key):
                from ..reconstruction.profile_assets import skin_profile_weights
                from .peri_hybrid import excluded_native
                asset_path = base / 'scene_assets.npz'
                with np.load(asset_path, allow_pickle=False) as staged:
                    if 'skin_profile_weight' not in staged:
                        arrays = {name: staged[name] for name in staged.files}
                    else:
                        arrays = None
                if arrays is not None:
                    arrays['skin_profile_weight'] = skin_profile_weights(arrays, excluded_native())
                    pending = base / 'scene_assets.profile.pending.npz'
                    np.savez_compressed(pending, **arrays)
                    pending.replace(asset_path)
            if any('sclera_vessel_strength' in cr.variant_options(v, sid).get('material_overrides', {})
                   for v, _ in todo if cr.eye_version(v) == key) and not (base / 'sclera_vessels.png').is_file():
                # Older staged bases predate the separate procedural mask; recreate it from their own saved eye params.
                from PIL import Image
                from ..eye import assets, params, sclera
                p = params.validate(json.loads((base / 'eye_params.json').read_text()))
                Image.fromarray(assets.sclera_vessel_texture(sclera.build(p, 512), 512)).save(base / 'sclera_vessels.png')
            if any('iris_normal_strength' in cr.variant_options(v, sid).get('material_overrides', {})
                   for v, _ in todo if cr.eye_version(v) == key) and not (base / 'iris_normal.png').is_file():
                # A cached photographic iris cannot be reconstructed from parameters alone.
                scene = json.loads((base / 'scene.json').read_text())
                if scene.get('optical_eyes', {}).get('personalization'):
                    raise RuntimeError(f'{base}: rebuild the cached base to recover its personalized iris normal')
                from PIL import Image
                from ..eye import assets, iris, params
                p = params.validate(json.loads((base / 'eye_params.json').read_text()))
                Image.fromarray(assets.normal_u8(iris.bake_normal(iris.build(p, 512), p, 512))).save(base / 'iris_normal.png')
            if any(cr.variant_options(v, sid).get('material_overrides', {}).get('eye_unified', 0.) >= .5
                   for v, _ in todo if cr.eye_version(v) == key) and not (base / 'eye_unified.png').is_file():
                scene = json.loads((base / 'scene.json').read_text())
                if scene.get('optical_eyes', {}).get('personalization'):
                    raise RuntimeError(f'{base}: rebuild the cached base to recover its personalized iris')
                from PIL import Image
                from ..eye import assets, params
                p = params.validate(json.loads((base / 'eye_params.json').read_text()))
                it, st = assets.structures(p, 512)
                Image.fromarray(assets.srgb_u8(assets.eye_basecolor(p, it, st, 512))).save(base / 'eye_unified.png')
            if any('sclera_vessel_all_strength' in cr.variant_options(v, sid).get('material_overrides', {})
                   for v, _ in todo if cr.eye_version(v) == key) and not (base / 'sclera_vessels_all.png').is_file():
                from PIL import Image
                from ..eye import assets, params, sclera
                p = params.validate(json.loads((base / 'eye_params.json').read_text()))
                Image.fromarray(assets.to_u8(sclera.isotropic_vessels(p, 512))).save(base / 'sclera_vessels_all.png')
            if any(cr.variant_options(v, sid).get('material_overrides', {}).get('sclera_curvy_vessels', 0.) >= .5
                   for v, _ in todo if cr.eye_version(v) == key):
                from PIL import Image
                from ..eye import assets, params, sclera
                p = params.validate(json.loads((base / 'eye_params.json').read_text()))
                Image.fromarray(assets.to_u8(sclera.isotropic_vessels(p, 512, curvy=True))).save(base / 'sclera_vessels_curvy.png')
            if any(cr.variant_options(v, sid).get('material_overrides', {}).get('sclera_irregular_vessels', 0.) >= .5
                   for v, _ in todo if cr.eye_version(v) == key):
                from PIL import Image
                from ..eye import assets, params, sclera
                p = params.validate(json.loads((base / 'eye_params.json').read_text()))
                Image.fromarray(assets.to_u8(sclera.irregular_vessels(p, 512))).save(base / 'sclera_vessels_irregular.png')
                if any(cr.variant_options(v, sid).get('material_overrides', {}).get('sclera_dense_vessels', 0.) >= .5
                       for v, _ in todo if cr.eye_version(v) == key):
                    Image.fromarray(assets.to_u8(sclera.irregular_vessels(p, 512, dense=True))).save(base / 'sclera_vessels_dense.png')
                    if any(cr.variant_options(v, sid).get('material_overrides', {}).get('sclera_vessel_lines_only', 0.) >= .5
                           for v, _ in todo if cr.eye_version(v) == key):
                        Image.fromarray(assets.to_u8(sclera.irregular_vessels(p, 512, dense=True, mottling=False))).save(base / 'sclera_vessels_lines.png')
                        if any(cr.variant_options(v, sid).get('material_overrides', {}).get('sclera_vessel_density', 0.) >= .5
                               for v, _ in todo if cr.eye_version(v) == key):
                            density = float(np.clip(p['sclera']['vascularity_intensity'] / 2., 0., 1.))
                            Image.fromarray(assets.to_u8(sclera.irregular_vessels(p, 512, dense=True, mottling=False, density=density))).save(base / 'sclera_vessels_sparse.png')
                            if any(cr.variant_options(v, sid).get('material_overrides', {}).get('sclera_vessel_organic', 0.) >= .5
                                   for v, _ in todo if cr.eye_version(v) == key):
                                Image.fromarray(assets.to_u8(sclera.irregular_vessels(p, 512, dense=True, mottling=False, density=density, organic=True))).save(base / 'sclera_vessels_organic.png')
                                if any(cr.variant_options(v, sid).get('material_overrides', {}).get('sclera_vessel_organic_dense', 0.) >= .5
                                       for v, _ in todo if cr.eye_version(v) == key):
                                    Image.fromarray(assets.to_u8(sclera.irregular_vessels(p, 512, dense=True, mottling=False, density=density, organic=True, organic_dense=True))).save(base / 'sclera_vessels_organic_dense.png')
                                    if any(cr.variant_options(v, sid).get('material_overrides', {}).get('sclera_vessel_guided', 0.) >= .5
                                           for v, _ in todo if cr.eye_version(v) == key):
                                        Image.fromarray(assets.to_u8(sclera.irregular_vessels(p, 512, dense=True, mottling=False, density=density, organic=True, organic_dense=True, guided=True))).save(base / 'sclera_vessels_guided.png')
                                        if any(cr.variant_options(v, sid).get('material_overrides', {}).get('sclera_vessel_selfsafe', 0.) >= .5
                                               for v, _ in todo if cr.eye_version(v) == key):
                                            Image.fromarray(assets.to_u8(sclera.irregular_vessels(p, 512, dense=True, mottling=False, density=density, organic=True, organic_dense=True, guided=True, avoid_self=True))).save(base / 'sclera_vessels_selfsafe.png')
                                        if any(cr.variant_options(v, sid).get('material_overrides', {}).get('sclera_vessel_nocross', 0.) >= .5
                                               for v, _ in todo if cr.eye_version(v) == key):
                                            Image.fromarray(assets.to_u8(sclera.irregular_vessels(p, 512, dense=True, mottling=False, density=density, organic=True, organic_dense=True, guided=True, avoid_crossings=True))).save(base / 'sclera_vessels_nocross.png')
            if any('sclera_tissue_strength' in cr.variant_options(v, sid).get('material_overrides', {})
                   for v, _ in todo if cr.eye_version(v) == key) and not (base / 'eye_tissue_factor.png').is_file():
                from PIL import Image
                from ..eye import assets, params
                p = params.validate(json.loads((base / 'eye_params.json').read_text()))
                Image.fromarray(assets.to_u8(assets.eye_tissue_factor(p, 512))).save(base / 'eye_tissue_factor.png')
            if any('iris_blend_softness_floor' in cr.variant_options(v, sid).get('material_overrides', {})
                   for v, _ in todo if cr.eye_version(v) == key):
                floors = {cr.variant_options(v, sid)['material_overrides']['iris_blend_softness_floor']
                          for v, _ in todo if cr.eye_version(v) == key
                          and 'iris_blend_softness_floor' in cr.variant_options(v, sid).get('material_overrides', {})}
                if len(floors) != 1:
                    raise ValueError('a cached base can hold only one iris_blend_softness_floor')
                scene = json.loads((base / 'scene.json').read_text())
                if scene.get('optical_eyes', {}).get('personalization'):
                    raise RuntimeError(f'{base}: rebuild the cached base to recover its personalized iris')
                from PIL import Image
                from ..eye import assets, params
                p = params.validate(json.loads((base / 'eye_params.json').read_text()))
                p['iris']['color_blend_softness'] = max(p['iris']['color_blend_softness'], floors.pop())
                it, st = assets.structures(p, 512)
                Image.fromarray(assets.srgb_u8(assets.eye_basecolor(p, it, st, 512))).save(base / 'eye_unified_soft.png')
            if any('iris_angular_smoothing' in cr.variant_options(v, sid).get('material_overrides', {})
                   for v, _ in todo if cr.eye_version(v) == key):
                settings = {(cr.variant_options(v, sid)['material_overrides']['iris_angular_smoothing'],
                             cr.variant_options(v, sid)['material_overrides'].get('iris_blend_softness_floor', 0.))
                            for v, _ in todo if cr.eye_version(v) == key
                            and 'iris_angular_smoothing' in cr.variant_options(v, sid).get('material_overrides', {})}
                if len(settings) != 1:
                    raise ValueError('a cached base can hold only one iris angular smoothing setting')
                scene = json.loads((base / 'scene.json').read_text())
                if scene.get('optical_eyes', {}).get('personalization'):
                    raise RuntimeError(f'{base}: rebuild the cached base to recover its personalized iris')
                from PIL import Image
                from ..eye import assets, params
                smoothing, floor = settings.pop()
                p = params.validate(json.loads((base / 'eye_params.json').read_text()))
                p['iris']['color_blend_softness'] = max(p['iris']['color_blend_softness'], floor)
                it, st = assets.structures(p, 512)
                color = assets.eye_basecolor(p, it, st, 512)
                Image.fromarray(assets.srgb_u8(assets.reduce_iris_angular_contrast(color, p, smoothing))).save(base / 'eye_unified_smooth.png')
            if any(cr.variant_options(v, sid).get('material_overrides', {}).get('iris_calibrate', 0.) >= .5
                   for v, _ in todo if cr.eye_version(v) == key):
                # This experiment uses the same left-eye UE base-color observations as the MH colour mapping.
                from PIL import Image
                from ..eye import assets, params
                from .eye_map import measure
                p = params.validate(json.loads((base / 'eye_params.json').read_text()))
                p['iris']['color_blend_softness'] = max(p['iris']['color_blend_softness'], .7)
                it, st = assets.structures(p, 512)
                color = assets.reduce_iris_angular_contrast(assets.eye_basecolor(p, it, st, 512), p, .6)
                measured = measure(sid, 'left', sector_only=True, geometric_iris=key >= 4)
                calibrated, gains = assets.calibrate_iris_color(color, p, measured['inner_linear'])
                Image.fromarray(assets.srgb_u8(calibrated)).save(base / 'eye_unified_calibrated.png')
                (base / 'eye_iris_calibration.json').write_text(json.dumps(dict(
                    sample=sid, side='left', target=measured['inner_linear'], gains=gains.tolist(),
                    method=('sector-only, camera-radius UE base-color inner iris; bounded linear RGB gain' if key >= 4 else
                            'sector-only UE base-color inner iris; bounded linear RGB gain')), indent=2) + '\n')
            compensations = {float(cr.variant_options(v, sid).get('material_overrides', {}).get('sclera_transmission_compensation', 0.))
                             for v, _ in todo if cr.eye_version(v) == key}
            compensations.discard(0.)
            if compensations:
                if not compensations <= {.5, 1.}:
                    raise ValueError('sclera transmission compensation must be .5 or 1')
                from PIL import Image
                from ..eye import assets, params
                from .eye_map import measure
                p = params.validate(json.loads((base / 'eye_params.json').read_text()))
                p['iris']['color_blend_softness'] = max(p['iris']['color_blend_softness'], .7)
                wants_bands = any(cr.variant_options(v, sid).get('material_overrides', {}).get('iris_radial_bands', 0.) >= .5
                                  for v, _ in todo if cr.eye_version(v) == key)
                wants_chroma = any(cr.variant_options(v, sid).get('material_overrides', {}).get('iris_fiber_chroma_fit', 0.) >= .5
                                   for v, _ in todo if cr.eye_version(v) == key)
                measured = measure(sid, 'left', sector_only=True, geometric_iris=key >= 4, radial_bands=wants_bands, chroma_detail=wants_chroma)
                target = measured['inner_linear']
                original_transmission = p['sclera']['transmission_color'][:]
                it, st = assets.structures(p, 512)
                for strength in sorted(compensations):
                    p['sclera']['transmission_color'] = [c + strength * (1. - c) for c in original_transmission]
                    color = assets.reduce_iris_angular_contrast(assets.eye_basecolor(p, it, st, 512), p, .6)
                    color, _ = assets.calibrate_iris_color(color, p, target)
                    name = 'eye_unified_trans05.png' if strength == .5 else 'eye_unified_trans1.png'
                    Image.fromarray(assets.srgb_u8(color)).save(base / name)
                    if strength == .5 and any(cr.variant_options(v, sid).get('material_overrides', {}).get('iris_radial_calibrate', 0.) >= .5
                                              for v, _ in todo if cr.eye_version(v) == key):
                        radial, radial_gains = assets.calibrate_iris_outer(color, p, measured['outer_linear'])
                        Image.fromarray(assets.srgb_u8(radial)).save(base / 'eye_unified_radial_trans05.png')
                        (base / 'eye_iris_radial_calibration.json').write_text(json.dumps(dict(
                            sample=sid, side='left', target=measured['outer_linear'], gains=radial_gains.tolist(),
                            method='sector-only camera-radius UE outer iris; bounded linear RGB gain'), indent=2) + '\n')
                    if strength == .5 and any(cr.variant_options(v, sid).get('material_overrides', {}).get('iris_layered_calibrate', 0.) >= .5
                                              for v, _ in todo if cr.eye_version(v) == key):
                        layered, inner_gains, outer_gains = assets.eye_basecolor_layered_calibrated(
                            p, it, st, 512, measured['inner_linear'], measured['outer_linear'])
                        Image.fromarray(assets.srgb_u8(layered)).save(base / 'eye_unified_layered_trans05.png')
                        (base / 'eye_iris_layered_calibration.json').write_text(json.dumps(dict(
                            sample=sid, side='left', inner_target=measured['inner_linear'], outer_target=measured['outer_linear'],
                            inner_gains=inner_gains.tolist(), outer_gains=outer_gains.tolist(),
                            method='bounded iris-layer gains before sclera compositing'), indent=2) + '\n')
                        if any(cr.variant_options(v, sid).get('material_overrides', {}).get('sclera_clean_base', 0.) >= .5
                               for v, _ in todo if cr.eye_version(v) == key):
                            from dataclasses import replace
                            masks = st.masks.copy()
                            masks[..., :2] = 0.
                            clean = replace(st, masks=masks)
                            clean_color, _, _ = assets.eye_basecolor_layered_calibrated(
                                p, it, clean, 512, measured['inner_linear'], measured['outer_linear'])
                            Image.fromarray(assets.srgb_u8(clean_color)).save(base / 'eye_unified_clean_trans05.png')
                    if strength == .5 and any(cr.variant_options(v, sid).get('material_overrides', {}).get('iris_adaptive_contrast', 0.) >= .5
                                              for v, _ in todo if cr.eye_version(v) == key):
                        from ..eye import iris
                        uv = assets.uv_grid(512)
                        x, y = uv[..., 0] - .5, uv[..., 1] - .5
                        radius = np.hypot(x, y) / p['cornea']['size']
                        sector = np.abs(y) < np.abs(x) * .5
                        source = assets.eye_basecolor(p, it, st, 512) @ iris.REC709
                        generated = iris.angular_contrast(source, radius, sector)
                        target_contrast = measured['angular_contrast']
                        smoothing = float(np.clip(1. - target_contrast / generated, 0., .8)) if generated and target_contrast is not None else .6
                        adaptive, _, _ = assets.eye_basecolor_layered_calibrated(
                            p, it, st, 512, measured['inner_linear'], measured['outer_linear'], smoothing=smoothing)
                        Image.fromarray(assets.srgb_u8(adaptive)).save(base / 'eye_unified_adaptive_trans05.png')
                        if any(cr.variant_options(v, sid).get('material_overrides', {}).get('sclera_clean_base', 0.) >= .5
                               for v, _ in todo if cr.eye_version(v) == key):
                            from dataclasses import replace
                            masks = st.masks.copy()
                            masks[..., :2] = 0.
                            clean = replace(st, masks=masks)
                            adaptive_clean, _, _ = assets.eye_basecolor_layered_calibrated(
                                p, it, clean, 512, measured['inner_linear'], measured['outer_linear'], smoothing=smoothing)
                            Image.fromarray(assets.srgb_u8(adaptive_clean)).save(base / 'eye_unified_clean_adaptive_trans05.png')
                            if any(cr.variant_options(v, sid).get('material_overrides', {}).get('iris_radial_bands', 0.) >= .5
                                   for v, _ in todo if cr.eye_version(v) == key):
                                bands, _, _ = assets.eye_basecolor_layered_calibrated(
                                    p, it, clean, 512, measured['inner_linear'], measured['outer_linear'],
                                    smoothing=smoothing, radial_targets=measured['radial_linear'])
                                Image.fromarray(assets.srgb_u8(bands)).save(base / 'eye_unified_clean_bands_trans05.png')
                                for ring_option, filename in (('iris_ring_soft', 'eye_unified_clean_ring_soft.png'),
                                                              ('iris_ring_none', 'eye_unified_clean_ring_none.png')):
                                    if any(cr.variant_options(v, sid).get('material_overrides', {}).get(ring_option, 0.) >= .5
                                           for v, _ in todo if cr.eye_version(v) == key):
                                        import copy
                                        adjusted = copy.deepcopy(p)
                                        if ring_option == 'iris_ring_soft':
                                            adjusted['iris']['limbal_ring_softness'] = max(adjusted['iris']['limbal_ring_softness'], .15)
                                        else:
                                            adjusted['iris']['limbal_ring_color'] = [1., 1., 1.]
                                        ring_color, _, _ = assets.eye_basecolor_layered_calibrated(
                                            adjusted, it, clean, 512, measured['inner_linear'], measured['outer_linear'],
                                            smoothing=smoothing, radial_targets=measured['radial_linear'])
                                        Image.fromarray(assets.srgb_u8(ring_color)).save(base / filename)
                                if any(cr.variant_options(v, sid).get('material_overrides', {}).get('iris_fiber_chroma', 0.) > 0.
                                       for v, _ in todo if cr.eye_version(v) == key):
                                    fibers, _, _ = assets.eye_basecolor_layered_calibrated(
                                        p, it, clean, 512, measured['inner_linear'], measured['outer_linear'],
                                        smoothing=smoothing, radial_targets=measured['radial_linear'], fiber_chroma=.6)
                                    Image.fromarray(assets.srgb_u8(fibers)).save(base / 'eye_unified_clean_fibers_trans05.png')
                                    if wants_chroma:
                                        signal = measured['chroma_fiber_signal']
                                        amount = float(np.clip((signal - .03) / .095, 0., 1.) * .6) if signal is not None else 0.
                                        fitted, _, _ = assets.eye_basecolor_layered_calibrated(
                                            p, it, clean, 512, measured['inner_linear'], measured['outer_linear'],
                                            smoothing=smoothing, radial_targets=measured['radial_linear'], fiber_chroma=amount)
                                        Image.fromarray(assets.srgb_u8(fitted)).save(base / 'eye_unified_clean_fibers_fit_trans05.png')
                                        (base / 'eye_fiber_chroma_fit.json').write_text(json.dumps(dict(signal=signal, strength=amount), indent=2) + '\n')
                                if any(cr.variant_options(v, sid).get('material_overrides', {}).get('iris_swap_regions', 0.) >= .5
                                       for v, _ in todo if cr.eye_version(v) == key):
                                    import copy
                                    swapped = copy.deepcopy(p)
                                    ip = swapped['iris']
                                    for axis in ('u', 'v'):
                                        a, b = f'primary_color_{axis}', f'secondary_color_{axis}'
                                        ip[a], ip[b] = ip[b], ip[a]
                                    color, _, _ = assets.eye_basecolor_layered_calibrated(
                                        swapped, it, clean, 512, measured['inner_linear'], measured['outer_linear'],
                                        smoothing=smoothing, radial_targets=measured['radial_linear'])
                                    Image.fromarray(assets.srgb_u8(color)).save(base / 'eye_unified_clean_swap_trans05.png')
                        (base / 'eye_iris_adaptive_calibration.json').write_text(json.dumps(dict(
                            sample=sid, target_contrast=target_contrast, generated_contrast=generated,
                            smoothing=smoothing, method='reference-driven angular contrast; bounded smoothing'), indent=2) + '\n')
            micro_strengths = {float(cr.variant_options(v, sid).get('material_overrides', {}).get('sclera_microtexture', 0.))
                               for v, _ in todo if cr.eye_version(v) == key}
            micro_strengths.discard(0.)
            if micro_strengths:
                if not micro_strengths <= {.1, .2}:
                    raise ValueError('sclera microtexture strength must be .1 or .2')
                from PIL import Image
                from ..eye import assets, params
                p = params.validate(json.loads((base / 'eye_params.json').read_text()))
                for strength in sorted(micro_strengths):
                    name = 'eye_micro10.png' if strength == .1 else 'eye_micro20.png'
                    Image.fromarray(assets.to_u8(assets.eye_sclera_microfactor(p, 512, strength))).save(base / name)
            surface_strengths = {float(cr.variant_options(v, sid).get('material_overrides', {}).get('sclera_surface_microtexture', 0.))
                                 for v, _ in todo if cr.eye_version(v) == key}
            surface_strengths.discard(0.)
            if surface_strengths:
                if not surface_strengths <= {.05, .1}:
                    raise ValueError('sclera surface microtexture strength must be .05 or .1')
                from PIL import Image
                from ..eye import assets, params
                p = params.validate(json.loads((base / 'eye_params.json').read_text()))
                for strength in sorted(surface_strengths):
                    name = 'eye_surface05.png' if strength == .05 else 'eye_surface10.png'
                    Image.fromarray(assets.to_u8(assets.eye_sclera_surface_microfactor(p, 512, strength))).save(base / name)
                if any(cr.variant_options(v, sid).get('material_overrides', {}).get('sclera_surface_fine', 0.) >= .5
                       for v, _ in todo if cr.eye_version(v) == key):
                    Image.fromarray(assets.to_u8(assets.eye_sclera_surface_microfactor(p, 512, .05, fine=True))).save(base / 'eye_surface_fine.png')
            if any(cr.variant_options(v, sid).get('material_overrides', {}).get('eye_per_side', 0.) >= .5
                   for v, _ in todo if cr.eye_version(v) == key):
                from PIL import Image
                from ..eye import assets, sclera
                from .eye_map import params_for
                p, record = params_for(sid, 'right', key)
                p['optics']['side'] = 'right'
                offsets = {float(cr.variant_options(v, sid).get('material_overrides', {}).get('eye_right_probe_size_offset', 0.))
                           for v, _ in todo if cr.eye_version(v) == key}
                offsets.discard(0.)
                if offsets:
                    if len(offsets) != 1 or any(cr.variant_options(v, sid).get('material_overrides', {}).get('eye_right_probe_size_offset', 0.) == 0.
                                                for v, _ in todo if cr.eye_version(v) == key):
                        raise ValueError('synthetic right-eye size probes require a separate staged base')
                    p['cornea']['size'] = float(np.clip(p['cornea']['size'] + offsets.pop(), .1, .3))
                p['iris']['color_blend_softness'] = max(p['iris']['color_blend_softness'], .7)
                it, st = assets.structures(p, 512)
                target = record['measured']['inner_linear']
                original_transmission = p['sclera']['transmission_color'][:]
                for compensation, name in ((0., 'eye_unified_right.png'), (.5, 'eye_unified_right_trans05.png')):
                    p['sclera']['transmission_color'] = [c + compensation * (1. - c) for c in original_transmission]
                    color = assets.reduce_iris_angular_contrast(assets.eye_basecolor(p, it, st, 512), p, .6)
                    color, gains = assets.calibrate_iris_color(color, p, target)
                    Image.fromarray(assets.srgb_u8(color)).save(base / name)
                    if compensation == .5 and any(cr.variant_options(v, sid).get('material_overrides', {}).get('iris_radial_calibrate', 0.) >= .5
                                                   for v, _ in todo if cr.eye_version(v) == key):
                        radial, radial_gains = assets.calibrate_iris_outer(color, p, record['measured']['outer_linear'])
                        Image.fromarray(assets.srgb_u8(radial)).save(base / 'eye_unified_right_radial_trans05.png')
                        (base / 'eye_right_radial_calibration.json').write_text(json.dumps(dict(
                            sample=sid, side='right', target=record['measured']['outer_linear'],
                            gains=radial_gains.tolist(), mapping_version=key), indent=2) + '\n')
                    if compensation == .5 and any(cr.variant_options(v, sid).get('material_overrides', {}).get('iris_layered_calibrate', 0.) >= .5
                                                   for v, _ in todo if cr.eye_version(v) == key):
                        layered, inner_gains, outer_gains = assets.eye_basecolor_layered_calibrated(
                            p, it, st, 512, record['measured']['inner_linear'], record['measured']['outer_linear'])
                        Image.fromarray(assets.srgb_u8(layered)).save(base / 'eye_unified_right_layered_trans05.png')
                        (base / 'eye_right_layered_calibration.json').write_text(json.dumps(dict(
                            sample=sid, side='right', inner_gains=inner_gains.tolist(), outer_gains=outer_gains.tolist(),
                            mapping_version=key), indent=2) + '\n')
                    if compensation == .5 and any(cr.variant_options(v, sid).get('material_overrides', {}).get('eye_bilateral_select', 0.) >= .5
                                                   for v, _ in todo if cr.eye_version(v) == key):
                        from dataclasses import replace
                        import copy
                        from ..eye import iris
                        from .eye_map import measure
                        from .eye_ring_select import select
                        observed = measure(sid, 'right', sector_only=True, geometric_iris=key >= 4, radial_bands=True)
                        uv = assets.uv_grid(512); x, y = uv[..., 0] - .5, uv[..., 1] - .5
                        radius = np.hypot(x, y) / p['cornea']['size']
                        source = assets.eye_basecolor(p, it, st, 512) @ iris.REC709
                        generated = iris.angular_contrast(source, radius, np.abs(y) < np.abs(x) * .5)
                        target_contrast = observed['angular_contrast']
                        smoothing = float(np.clip(1. - target_contrast / generated, 0., .8)) if generated and target_contrast is not None else .6
                        masks = st.masks.copy(); masks[..., :2] = 0.; clean = replace(st, masks=masks)
                        for neutral, filename in ((False, 'eye_unified_right_bands_select.png'),
                                                  (True, 'eye_unified_right_ring_none_select.png')):
                            adjusted = copy.deepcopy(p)
                            if neutral: adjusted['iris']['limbal_ring_color'] = [1., 1., 1.]
                            color, _, _ = assets.eye_basecolor_layered_calibrated(
                                adjusted, it, clean, 512, observed['inner_linear'], observed['outer_linear'],
                                smoothing=smoothing, radial_targets=observed['radial_linear'])
                            Image.fromarray(assets.srgb_u8(color)).save(base / filename)
                        (base / 'eye_params_right.json').write_text(json.dumps(p, indent=2) + '\n')
                        choice = select(sid, base, side='right')
                        chosen = 'eye_unified_right_ring_none_select.png' if choice['choice'] == 'neutral' else 'eye_unified_right_bands_select.png'
                        shutil.copyfile(base / chosen, base / 'eye_unified_right_bilateral.png')
                        (base / 'eye_right_ring_selection.json').write_text(json.dumps(choice, indent=2) + '\n')
                p['sclera']['transmission_color'] = original_transmission
                Image.fromarray(assets.to_u8(assets.eye_tissue_factor(p, 512))).save(base / 'eye_tissue_factor_right.png')
                Image.fromarray(assets.to_u8(sclera.isotropic_vessels(p, 512))).save(base / 'sclera_vessels_all_right.png')
                if any(cr.variant_options(v, sid).get('material_overrides', {}).get('sclera_curvy_vessels', 0.) >= .5
                       for v, _ in todo if cr.eye_version(v) == key):
                    Image.fromarray(assets.to_u8(sclera.isotropic_vessels(p, 512, curvy=True))).save(base / 'sclera_vessels_curvy_right.png')
                if any(cr.variant_options(v, sid).get('material_overrides', {}).get('sclera_irregular_vessels', 0.) >= .5
                       for v, _ in todo if cr.eye_version(v) == key):
                    Image.fromarray(assets.to_u8(sclera.irregular_vessels(p, 512))).save(base / 'sclera_vessels_irregular_right.png')
                    if any(cr.variant_options(v, sid).get('material_overrides', {}).get('sclera_dense_vessels', 0.) >= .5
                           for v, _ in todo if cr.eye_version(v) == key):
                        Image.fromarray(assets.to_u8(sclera.irregular_vessels(p, 512, dense=True))).save(base / 'sclera_vessels_dense_right.png')
                        if any(cr.variant_options(v, sid).get('material_overrides', {}).get('sclera_vessel_lines_only', 0.) >= .5
                               for v, _ in todo if cr.eye_version(v) == key):
                            Image.fromarray(assets.to_u8(sclera.irregular_vessels(p, 512, dense=True, mottling=False))).save(base / 'sclera_vessels_lines_right.png')
                            if any(cr.variant_options(v, sid).get('material_overrides', {}).get('sclera_vessel_density', 0.) >= .5
                                   for v, _ in todo if cr.eye_version(v) == key):
                                density = float(np.clip(p['sclera']['vascularity_intensity'] / 2., 0., 1.))
                                Image.fromarray(assets.to_u8(sclera.irregular_vessels(p, 512, dense=True, mottling=False, density=density))).save(base / 'sclera_vessels_sparse_right.png')
                                if any(cr.variant_options(v, sid).get('material_overrides', {}).get('sclera_vessel_organic', 0.) >= .5
                                       for v, _ in todo if cr.eye_version(v) == key):
                                    Image.fromarray(assets.to_u8(sclera.irregular_vessels(p, 512, dense=True, mottling=False, density=density, organic=True))).save(base / 'sclera_vessels_organic_right.png')
                                    if any(cr.variant_options(v, sid).get('material_overrides', {}).get('sclera_vessel_organic_dense', 0.) >= .5
                                           for v, _ in todo if cr.eye_version(v) == key):
                                        Image.fromarray(assets.to_u8(sclera.irregular_vessels(p, 512, dense=True, mottling=False, density=density, organic=True, organic_dense=True))).save(base / 'sclera_vessels_organic_dense_right.png')
                                        if any(cr.variant_options(v, sid).get('material_overrides', {}).get('sclera_vessel_guided', 0.) >= .5
                                               for v, _ in todo if cr.eye_version(v) == key):
                                            Image.fromarray(assets.to_u8(sclera.irregular_vessels(p, 512, dense=True, mottling=False, density=density, organic=True, organic_dense=True, guided=True))).save(base / 'sclera_vessels_guided_right.png')
                                            if any(cr.variant_options(v, sid).get('material_overrides', {}).get('sclera_vessel_selfsafe', 0.) >= .5
                                                   for v, _ in todo if cr.eye_version(v) == key):
                                                Image.fromarray(assets.to_u8(sclera.irregular_vessels(p, 512, dense=True, mottling=False, density=density, organic=True, organic_dense=True, guided=True, avoid_self=True))).save(base / 'sclera_vessels_selfsafe_right.png')
                                            if any(cr.variant_options(v, sid).get('material_overrides', {}).get('sclera_vessel_nocross', 0.) >= .5
                                                   for v, _ in todo if cr.eye_version(v) == key):
                                                Image.fromarray(assets.to_u8(sclera.irregular_vessels(p, 512, dense=True, mottling=False, density=density, organic=True, organic_dense=True, guided=True, avoid_crossings=True))).save(base / 'sclera_vessels_nocross_right.png')
                Image.fromarray(assets.shell_textures(p, st, 512)['normal']).save(base / 'eye_shell_normal_right.png')
                Image.fromarray(assets.to_u8(assets.eye_sclera_surface_microfactor(p, 512, .05))).save(base / 'eye_surface05_right.png')
                if any(cr.variant_options(v, sid).get('material_overrides', {}).get('sclera_surface_fine', 0.) >= .5
                       for v, _ in todo if cr.eye_version(v) == key):
                    Image.fromarray(assets.to_u8(assets.eye_sclera_surface_microfactor(p, 512, .05, fine=True))).save(base / 'eye_surface_fine_right.png')
                (base / 'eye_right_calibration.json').write_text(json.dumps(dict(
                    sample=sid, side='right', target=target, gains=gains.tolist(), mapping_version=key,
                    method='right-eye UE base-color and settings; bounded linear RGB gain'), indent=2) + '\n')
            if any('unified_shell_normal_strength' in cr.variant_options(v, sid).get('material_overrides', {})
                   for v, _ in todo if cr.eye_version(v) == key) and not (base / 'eye_shell_normal.png').is_file():
                from PIL import Image
                from ..eye import assets, params, sclera
                p = params.validate(json.loads((base / 'eye_params.json').read_text()))
                Image.fromarray(assets.shell_textures(p, sclera.build(p, 512), 512)['normal']).save(base / 'eye_shell_normal.png')
            if any(cr.variant_options(v, sid).get('material_overrides', {}).get('eye_model_layered2', 0.) >= .5
                   for v, _ in todo if cr.eye_version(v) == key) and not _eye2_current(base):
                from ..eye2 import bake as eye2_bake      # layered eye meshes + textures from this base's eye inputs
                for side in ('left', 'right'):
                    ue = wj_local_tmp() / 'eye_eyeball_ue01_results' / 'eye_isolate_eyeball' / sid / f'eye_{side}_basecolor.exr'
                    eye2_bake.build_eye(base, side, ue_basecolor=ue if ue.is_file() else None)
            if any(cr.variant_options(v, sid).get('material_overrides', {}).get('peri_contact_atlas', 0.) >= .5
                   for v, _ in todo if cr.eye_version(v) == key):
                from ..reconstruction.periocular import build_contact_atlas
                build_contact_atlas(base)
            shutil.rmtree(base / 'cache', ignore_errors=True)
            bases[key] = (base.name, json.loads((base / 'request.json').read_text()))
        cams = json.loads((V1 / 'samples' / sid / 'cameras.json').read_text())['cameras']
        for v, c in todo:
            base_name, base_req = bases[cr.eye_version(v)]
            req = view_request(base_req, cams[c]['vhuman'], shift, c, cr.variant_options(v, sid), prof, cr.RENDER['micro_scale'])
            jobs.append(dict(sid=sid, variant=v, camera=c, base=f'{sid}/{base_name}', candidate=f'{sid}/cand',
                             out=f'{sid}/{variant_dir(v, profile)}/{c}', request=req))
    (root / 'jobs.json').write_text(json.dumps(dict(repo=WIN_REPO, profile=profile, jobs=jobs), indent=1))
    return len(jobs)


SCORE_SETS = ('confirm', 'ears', 'eyes', 'eye_sweep', 'ue_match', 'ue_confirm', 'eye_ior1', 'eye_iris', 'eye_sat', 'eye_shadow', 'rmh', 'eye_place', 'dual', 'ear_sss', 'eye_sclera', 'eye_irisspec', 'combo', 'iris_diag', 'iris_fill', 'iris_fill_confirm', 'iris_fill_repeat', 'iris_physical', 'skin_recheck', 'ear_rough', 'wide', 'eye_wide')


def score_set(name, ids, profile, env='vhstudio'):
    """Run the compare_render scoring of a named experiment (CPU: set HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES=)."""
    from . import compare_render as cr
    exp = dict(confirm={}, ears=cr.EAR_SET, eyes=cr.EYE_SET, eye_sweep=cr.EYE_SWEEP_SET, ue_match=cr.UE_SET, ue_confirm=cr.UE_CONFIRM_SET, eye_ior1=cr.EYE_IOR1_SET, eye_iris=cr.EYE_IRIS_SET, eye_sat=cr.EYE_SAT_SET, eye_shadow=cr.EYE_SHADOW_SET, rmh=cr.RMH_SET, eye_place=cr.EYE_PLACE_SET, dual=cr.DUAL_SET, ear_sss=cr.EAR_SSS_SET, eye_sclera=cr.EYE_SCLERA_SET, eye_irisspec=cr.EYE_IRISSPEC_SET, combo=cr.COMBO_SET, iris_diag=cr.IRIS_DIAG_SET, iris_fill=cr.IRIS_FILL_SET, iris_fill_confirm=cr.IRIS_FILL_CONFIRM_SET, iris_fill_repeat=cr.IRIS_FILL_REPEAT_SET, iris_physical=cr.IRIS_PHYSICAL_SET, skin_recheck=cr.SKIN_RECHECK_SET, ear_rough=cr.EAR_ROUGH_SET, wide=cr.WIDE_SET, eye_wide=cr.EYE_WIDE_SET)[name]
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
           f"if ($f) {{ tar -cf '{wj.win(wj.WIN_ROOT)}\\{tag}_out.tar' --exclude '*.npz' --exclude 'eye2_*.png' --exclude 'eye_unified*.png' "
           f"--exclude 'sclera_vessels*.png' --exclude 'eye_surface*.png' --exclude 'eye_shell*.png' --exclude 'eye_tissue*.png' "
           f"--exclude 'skin_normal_map.png' @($f) progress.log }}")
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
