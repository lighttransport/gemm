"""UE-vs-Cycles comparison: render MetaHuman identities through vhuman's Cycles path with the reference cameras.

usage: python -m server.vhuman.metahuman.compare_render [--ids id,..] [--every 10] [--cameras front,...] [--build-only]
       python -m server.vhuman.metahuman.compare_render --report [--ids ...]

For each reference-rendered sample (samples/<id>/renders/renders_done.json):
  1. Pseudo-candidate: GNM skin vertices = binding.evaluate(MH head) (exact surface), remaining anatomy (eyes, mouth,
     teeth) from the GNM identity fit of that skin; skin albedo = MH base colour resampled through the GNM-UV texel map
     (invalid texels: nearest valid). Normal map flat (the MH tangent-space normal conversion is not built yet); ORM /
     specular, observation masks and eye parameters come from the template GNM candidate (UV-region authored maps).
  2. The candidate and every camera are translated so the head centre sits where vhuman's fitted heads sit (the studio
     lamp rig is placed around that point); rotation and focal length are unchanged, so views match the UE ones.
  3. offline_render.render(..., explicit_camera=...) per camera (quality 'full', B settings), outputs under
     COMPARE_ROOT/<id>/cycles/<cam>/.
Report (--report): per camera, silhouette agreement (Cycles alpha vs the UE show-only base-colour pass, restricted to
the rows the GNM head covers: the UE body/torso has no GNM counterpart) and exposure-matched LPIPS/DISTS/Lab dE on the
parser skin crops (ab_metrics.compare_region). The two renderers use different lighting (UE 'Studio' level vs vhuman's
authored studio rig) until the ColorChecker calibration exists, so appearance numbers are relative, not absolute.
Outputs stay outside the repo (derived renders of generated MetaHuman characters; see README policy).
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from . import DATASET_ROOT
from .binding import evaluate, load_binding
from .images import read_exr
from .register import OUT_DIR, load_gnm, umeyama
from .samples import V1, load_sample, to_vhuman

ROOT = Path(__file__).resolve().parents[3]
COMPARE_ROOT = DATASET_ROOT / 'compare_v1'
TEMPLATE = ROOT / 'tmp/vhuman-ab-cycles/ears/s1_B/base_reconstruction'     # plain GNM candidate (no refined ears)
TEMPLATE_FIT = ROOT / 'tmp/vhuman-ab-cycles/candidates/s1/B/fit.json'
HEAD_ANCHOR = np.array([0., -.07, -.075])     # vhuman fitted-head centre (s1 B front render targets['head'])
CAMERAS = ('front', 'yaw_p40', 'yaw_m40', 'yaw_p90', 'yaw_m90', 'ear_left', 'ear_right', 'eye_left', 'eye_right', 'mouth')
RENDER = dict(quality='full', displacement_scale=.25, micro_scale=.25, preset='final', samples=128, include_hair=False,
              accessories='omit', save_blend=False, lighting='studio')


def _sha(path):
    from ..reconstruction.observations import sha256
    return sha256(path)


def gnm_identity(skin):
    """Full GNM anatomy for bound skin vertices: identity fit in the GNM frame, mapped back with the inverse similarity."""
    from ..rig.gnm_model import GNMModel
    gnm = load_gnm(); model = GNMModel()
    ids = gnm['skin_ids']; tmpl = gnm['template'][ids]; basis = gnm['basis'][:, ids].reshape(len(gnm['basis']), -1)
    s, r, t = umeyama(skin, tmpl)
    d = ((s * (r @ skin.T)).T + t - tmpl).reshape(-1)
    beta = np.linalg.solve(basis @ basis.T + 1e-3 * np.eye(len(basis)), basis @ d)
    identity = np.zeros(model.identity_dim); identity[:len(beta)] = beta
    native, joints = model.evaluate(identity)
    back = lambda x: ((np.asarray(x, np.float64) - t) @ r) / s            # inverse of x -> s R x + t
    full = back(native); full[ids] = skin
    return dict(full=full, joints=back(joints), native=np.asarray(native, np.float64), identity=identity, scale=1. / s,
                alignment=r.T, offset=-(r.T @ t) / s, fit_rms_mm=float(1e3 * np.sqrt(((back(native)[ids] - skin) ** 2).sum(1).mean())),
                model=model)


def fill_nearest(img, valid):
    from scipy.ndimage import distance_transform_edt
    _, (iy, ix) = distance_transform_edt(~valid, return_indices=True)
    return img[iy, ix]


def build_candidate(sid, *, force=False):
    """COMPARE_ROOT/<id>/head/{fit.json, reconstruction/mh/} (layout expected by offline_assets.prepare)."""
    from ..reconstruction.provenance import validate_candidate
    from ..reconstruction.reference import Camera
    d = V1 / 'samples' / sid
    head_dir = COMPARE_ROOT / sid / 'head'; out = head_dir / 'reconstruction' / 'mh'
    if (out / 'manifest.json').is_file() and not force:
        return out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copyfile(TEMPLATE_FIT, head_dir / 'fit.json')
    for p in TEMPLATE.iterdir():
        if p.is_file():
            shutil.copyfile(p, out / p.name)
    s = load_sample(d)
    skin = evaluate(load_binding(), to_vhuman(s['face']['head'][0]))
    fit = gnm_identity(skin)
    cams = json.loads((d / 'cameras.json').read_text())
    shift = HEAD_ANCHOR - np.asarray(cams['targets']['head'])
    full = fit['full'] + shift; joints = fit['joints'] + shift
    model = fit['model']; ext = model.group('skin_exterior')
    g = dict(np.load(TEMPLATE / 'geometry.npz'))
    n_pose = len(g['full_captured'])
    g.update(full_neutral=full.astype(np.float32), full_captured=np.repeat(full[None], n_pose, 0).astype(np.float32),
             neutral=full[ext].astype(np.float32), captured=np.repeat(full[ext][None], n_pose, 0).astype(np.float32),
             gnm_identity=fit['identity'], gnm_expressions=np.zeros_like(g['gnm_expressions']),
             gnm_rotations=np.zeros_like(g['gnm_rotations']), gnm_translations=np.zeros_like(g['gnm_translations']),
             gnm_joint_positions=joints, scale=np.float64(fit['scale']), rotation=fit['alignment'],
             gnm_alignment_offset=fit['offset'] + shift,
             gnm_bind_residual=full - (fit['scale'] * fit['native'] @ fit['alignment'].T + fit['offset'] + shift))
    np.savez_compressed(out / 'geometry.npz', **g)
    # albedo: MH base colour through the GNM-UV texel map (2048 if built)
    tm_path = OUT_DIR / 'texel_map_2048.npz'
    tm = dict(np.load(tm_path if tm_path.is_file() else OUT_DIR / 'texel_map_1024.npz'))
    from .register import resample
    base = np.asarray(Image.open(d / 'tex' / 'basecolor.png').convert('RGB'))
    rgb, ok = resample(base, tm)
    rgb = fill_nearest(rgb, ok)
    res = Image.open(TEMPLATE / 'skin_basecolor.png').size
    img = Image.fromarray(np.uint8(np.clip(rgb, 0, 1) * 255 + .5))
    if img.size != res:
        img = img.resize(res, Image.LANCZOS)
    img.save(out / 'skin_basecolor.png')
    Image.new('RGB', res, (128, 128, 255)).save(out / 'skin_normal.png')
    # portrait = UE front render (display transform) so parsing/accessory/hair stages see this identity, bald
    front = read_exr(d / 'renders' / 'front__studio.exr')[..., :3]
    gain = .8 / max(float(np.percentile(front, 99)), 1e-6)
    x = np.clip(front * gain, 0, 1); x = np.where(x <= .0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - .055)
    Image.fromarray(np.uint8(x * 255 + .5)).save(out / 'portrait.png')
    fc = cams['cameras']['front']['vhuman']; size = fc['size']
    camera = Camera(float(fc['focal_px']), size[0] / 2, size[1] / 2, np.asarray(fc['origin']) + shift, np.asarray(fc['rotation']))
    obs = json.loads((out / 'observations.json').read_text())
    xy, _ = camera.project(joints[[3, 2]])                                 # eye_right = joint 3, eye_left = joint 2
    for k, p in zip(('eye_right', 'eye_left'), xy):
        obs['views'][0]['anchors'][k]['xy'] = [float(p[0]), float(p[1])]
    (out / 'observations.json').write_text(json.dumps(obs, indent=2))
    manifest = json.loads((out / 'manifest.json').read_text())
    manifest['id'] = f'mh_compare_{sid}'
    manifest['geometry_sha256'] = _sha(out / 'geometry.npz'); manifest['portrait_sha256'] = _sha(out / 'portrait.png')
    manifest['geometry']['fitted_cameras'] = [camera.as_dict()]
    manifest['geometry']['metahuman_pseudo_candidate'] = dict(
        sample=sid, template=str(TEMPLATE), translation_m=shift.tolist(), gnm_identity_fit_rms_mm=fit['fit_rms_mm'],
        texel_map=tm_path.name if tm_path.is_file() else 'texel_map_1024.npz', texel_valid_fraction=float(ok.mean()),
        limitations=['skin normal map flat (MH tangent normals not converted)', 'ORM/specular and eye parameters from the template candidate',
                     'eyes/mouth/teeth from the GNM identity fit, not the MH meshes', 'portrait is the UE front render'])
    completion = json.loads((out / 'generated_skin.json').read_text())
    completion['basecolor_sha256'] = _sha(out / 'skin_basecolor.png'); completion['source_geometry_sha256'] = manifest['geometry_sha256']
    completion['metahuman_albedo'] = dict(sample=sid, new_view_evidence=False)
    manifest['material']['synthetic_completion'] = completion
    for name, obj in (('manifest.json', manifest), ('generated_skin.json', completion), ('skin_material.json', manifest['material'])):
        (out / name).write_text(json.dumps(obj, indent=2) + '\n')
    validate_candidate(out)
    return out


def render_sample(sid, cameras=CAMERAS):
    from ..reconstruction.offline_render import render
    cand = build_candidate(sid)
    shift = np.asarray(json.loads((cand / 'manifest.json').read_text())['geometry']['metahuman_pseudo_candidate']['translation_m'])
    cams = json.loads((V1 / 'samples' / sid / 'cameras.json').read_text())['cameras']
    done = []
    for name in cameras:
        out = COMPARE_ROOT / sid / 'cycles' / name
        if (out / 'render_result.json').is_file():
            done.append(name); continue
        if out.exists():
            shutil.rmtree(out)
        v = cams[name]['vhuman']
        explicit = dict(origin=(np.asarray(v['origin']) + shift).tolist(), rotation=v['rotation'], focal_px=v['focal_px'], size=v['size'])
        opts = dict(RENDER); preset = opts.pop('preset'); samples = opts.pop('samples')
        render(cand, out, device='hip', preset=preset, samples=samples, resolution=int(v['size'][0]), explicit_camera=explicit,
               hard_limit_mib=14336, **opts)
        done.append(name)
    return done


def report(ids, cameras=CAMERAS):
    from ..face_parsing import FaceParser
    from ..reconstruction import ab_metrics as am
    parser = FaceParser(); rows = {}
    for sid in ids:
        rec = {}
        for name in cameras:
            c = COMPARE_ROOT / sid / 'cycles' / name / 'beauty.exr'
            ue = V1 / 'samples' / sid / 'renders' / f'{name}__studio.exr'
            ub = V1 / 'samples' / sid / 'renders' / f'{name}__basecolor.exr'
            if not (c.is_file() and ue.is_file()):
                continue
            cy = read_exr(c); u = read_exr(ue)[..., :3]; um = read_exr(ub)[..., :3].max(-1) > 1e-4
            cm = cy[..., 3] > .5 if cy.shape[-1] == 4 else cy[..., :3].max(-1) > 1e-4
            rows_c = np.flatnonzero(cm.any(1))
            band = np.zeros_like(cm)
            if len(rows_c):
                band[rows_c.min():rows_c.max() + 1] = True
            inter = (cm & um & band).sum(); union = ((cm | um) & band).sum()
            both = cm & um
            gain = float(np.median(u[both].mean(-1)) / max(np.median(cy[..., :3][both].mean(-1)), 1e-6)) if both.any() else 1.
            disp = lambda x: np.uint8(np.clip(np.where(x <= .0031308, 12.92 * x, 1.055 * np.clip(x, 0, None) ** (1 / 2.4) - .055), 0, 1) * 255 + .5)
            k = .6 / max(float(np.percentile(u[um], 95)), 1e-6) if um.any() else 1.
            a, b = disp(u * k), disp(cy[..., :3] * gain * k)
            r = am.compare_region(a, b, parser, ['skin', 'nose'])
            r.pop('crops', None)
            rec[name] = dict(silhouette_iou_head_rows=float(inter / max(union, 1)), cycles_to_ue_gain=gain,
                             lpips=r.get('lpips'), dists=r.get('dists'), delta_e=(r.get('delta_e') or {}).get('mean'), status=r['status'])
            Image.fromarray(np.concatenate([a, b], 1)).save(COMPARE_ROOT / sid / f'{name}_ue_cycles.png')
        rows[sid] = rec
        (COMPARE_ROOT / sid / 'compare.json').write_text(json.dumps(rec, indent=1) + '\n')
    summary = {}
    for name in cameras:
        vals = [rows[s][name] for s in rows if name in rows[s]]
        if vals:
            summary[name] = {k: float(np.nanmean([v[k] for v in vals if isinstance(v.get(k), (int, float))] or [np.nan]))
                             for k in ('silhouette_iou_head_rows', 'cycles_to_ue_gain', 'lpips', 'dists', 'delta_e')} | dict(n=len(vals))
    (COMPARE_ROOT / 'summary.json').write_text(json.dumps(dict(ids=list(rows), cameras=summary), indent=1) + '\n')
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--ids', default=''); ap.add_argument('--every', type=int, default=10)
    ap.add_argument('--cameras', default=','.join(CAMERAS)); ap.add_argument('--build-only', action='store_true')
    ap.add_argument('--report', action='store_true')
    a = ap.parse_args()
    render_ids = sorted(json.loads((V1 / 'sampling.json').read_text())['render_ids'])
    ids = a.ids.split(',') if a.ids else render_ids[::a.every]
    cams = tuple(a.cameras.split(','))
    if a.report:
        print(json.dumps(report(ids, cams), indent=1)); return
    for sid in ids:
        if a.build_only:
            print(sid, build_candidate(sid, force=True), flush=True); continue
        print(sid, render_sample(sid, cams), flush=True)


if __name__ == '__main__':
    main()
