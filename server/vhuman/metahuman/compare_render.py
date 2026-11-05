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
    write_normal(out, d, s, skin, tm, res)
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
        limitations=['skin normal = MH tangent normals converted to GNM tangent space (used only with skin_normal_map=True)', 'ORM/specular and eye parameters from the template candidate',
                     'eyes/mouth/teeth from the GNM identity fit, not the MH meshes', 'portrait is the UE front render'])
    completion = json.loads((out / 'generated_skin.json').read_text())
    completion['basecolor_sha256'] = _sha(out / 'skin_basecolor.png'); completion['source_geometry_sha256'] = manifest['geometry_sha256']
    completion['metahuman_albedo'] = dict(sample=sid, new_view_evidence=False)
    manifest['material']['synthetic_completion'] = completion
    for name, obj in (('manifest.json', manifest), ('generated_skin.json', completion), ('skin_material.json', manifest['material'])):
        (out / name).write_text(json.dumps(obj, indent=2) + '\n')
    validate_candidate(out)
    return out


def write_normal(out, d, s, skin, tm, res):
    """skin_normal.png = the sample's MH tangent normals in GNM tangent space (normal_map.convert); flat where unmapped."""
    from .normal_map import convert, encode
    n, ok = convert(np.asarray(Image.open(d / 'tex' / 'normal.png')), to_vhuman(s['face']['head'][0]).astype(np.float64), skin, tm)
    img = Image.fromarray(encode(n))
    if img.size != res:
        img = img.resize(res, Image.LANCZOS)
    img.save(out / 'skin_normal.png')
    return float(ok.mean())


def update_normal(sid):
    """Add the converted normal to an existing pseudo-candidate (built before normal_map existed)."""
    out = COMPARE_ROOT / sid / 'head' / 'reconstruction' / 'mh'
    m = json.loads((out / 'manifest.json').read_text())['geometry']['metahuman_pseudo_candidate']
    d = V1 / 'samples' / sid; s = load_sample(d)
    skin = evaluate(load_binding(), to_vhuman(s['face']['head'][0]))
    tm = dict(np.load(OUT_DIR / m['texel_map']))
    return write_normal(out, d, s, skin, tm, Image.open(out / 'skin_basecolor.png').size)


VARIANTS = {'': {}, 'normal': dict(skin_normal_map=True), 'normal_eyes': dict(skin_normal_map=True),   # dir suffix -> render options
            'mat_r125_c0': dict(skin_normal_map=True, material_overrides=dict(roughness_scale=1.25, skin_coat=0.)),
            'ear_sss2': dict(skin_normal_map=True, material_overrides=dict(ear_sss_scale=2.)),
            'ear_sss4': dict(skin_normal_map=True, material_overrides=dict(ear_sss_scale=4.)),
            'mat_r125_dual': dict(skin_normal_map=True, material_overrides=dict(roughness_scale=1.25, skin_coat=0., dual_lobe=1.)),
            # per-texel MH normal faded on the (coarser) GNM ears: strength x (1 - (1 - s) ear weight)
            'normal_ear0': dict(skin_normal_map=True, material_overrides=dict(normal_ear_scale=0.)),
            'normal_ear50': dict(skin_normal_map=True, material_overrides=dict(normal_ear_scale=.5)),
            'normal_eyes2': dict(skin_normal_map=True)}
# UE MetaHuman skin shading (ue_scripts/dump_skin_material.py -> v1/presets/skin_material.json): SSP_skin_unified mean
# free path colour x distance = 1.05/.41/.26 mm, material Opacity (Scatter Distance Base .5 x Global .5) scales it; dual
# specular lobes .8r/1.1r mixed .5, no coat; region roughness ~.55-.65 x1.0 (~1.25x vhuman's template); Specular ~.75 ->
# F0 ~.06 (vhuman .028 at level .5). Variants on 'normal' (Subsurface Weight 1 = all diffuse through the random walk).
UE_MFP_MM = (1.05, .41, .26)
UE_SSS = {f'ue_sss{int(k * 100)}': dict(subsurface_weight=1., subsurface_radius=list(UE_MFP_MM), subsurface_scale=k) for k in (.25, .5, 1.)}
UE_MATCH = dict(roughness_scale=1.25, skin_coat=0.)
VARIANTS.update({k: dict(skin_normal_map=True, material_overrides=v) for k, v in UE_SSS.items()})
VARIANTS.update(ue_spec=dict(skin_normal_map=True, material_overrides=dict(specular_scale=1.8)),
                ue_match=dict(skin_normal_map=True, material_overrides=dict(UE_MATCH, **UE_SSS['ue_sss50'])),
                ue_match_spec=dict(skin_normal_map=True, material_overrides=dict(UE_MATCH, specular_scale=1.8, **UE_SSS['ue_sss50'])))
MH_EYES = {'normal_eyes': 1, 'normal_eyes2': 2}   # variants rendered with MetaHuman-mapped eye parameters -> eye_map version


# Material sweep (plan 7b), one parameter at a time around the 'normal' variant, scored against the VHStudio references.
# material_overrides keys go to cycles_scene.full_skin_graph; skin_normal_strength / micro_scale are render arguments.
SWEEP = {'roughness_scale': (.8, 1.25, 1.5), 'specular_scale': (.5, .75, 1.25), 'skin_coat': (0., .05, .2),
         'subsurface_weight': (.3, .45, .8), 'subsurface_scale': (.25, .75, 1.), 'skin_normal_strength': (.5, 1.5),
         'micro_scale': (0., .5)}
SWEEP_IDS = ('mh_000000', 'mh_000145', 'mh_000300')      # three skin tones / ages among the comparison ids
SWEEP_CAMERAS = ('front', 'yaw_p40')
RENDER_ARGS = ('skin_normal_strength', 'micro_scale')

# Eye optics sweep: 'normal_eyes2' (eye_map v2) + one Cycles eye-material setting (cycles_scene material_overrides; the
# authored values are cornea roughness .025, IOR 1.376, Specular IOR Level .5, sclera roughness .25, iris .55).
EYE_SWEEP = {'cornea_roughness': (0., .08, .15), 'cornea_ior': (1.3, 1.336, 1.42), 'cornea_specular': (0., .25, 1.),
             'sclera_roughness': (.15, .4, .6), 'iris_roughness': (.3, .8)}
EYE_SWEEP_IDS = ('mh_000000', 'mh_000145', 'mh_000300', 'mh_000358')
EYE_SWEEP_CAMERAS = ('eye_left', 'eye_right', 'front')
EYE_EXTRA_PARAMS = ('iris_gain', 'iris_sat', 'cornea_shadow', 'sclera_sat', 'sclera_gain', 'iris_specular')   # ew_ params outside the original sweep (2026-11-05 iris brightness test)


def sweep_variant(param, value):
    return f'sw_{param}_{value:g}'


def eye_variant(param, value):
    return f'ew_{param}_{value:g}'


EYE_SWEEP_VARIANTS = tuple(eye_variant(p, v) for p, values in EYE_SWEEP.items() for v in values)


def eye_version(variant):
    """eye_map version a variant renders with (0 = template eyes)."""
    return 2 if variant.startswith('ew_') or variant in ('combo', 'combo_is0') else MH_EYES.get(variant, 0)


MH_ROUGHNESS_MEAN = 1.03   # mean MetaHuman skin 'roughness' multiplier of the 12 comparison heads


def variant_options(variant, sid=None):
    """Render options of a variant name: VARIANTS entries, or sw_<param>_<value> = 'normal' + one sweep setting.
    mat_rmh_c0 (per identity, needs sid): roughness x 1.25 x (MH skin roughness / 1.03), coat 0 -- the x1.25 gain per head
    correlates with MH's sampled roughness multiplier (r -.71), mh_000145 (.91) is hurt."""
    if variant == 'mat_rmh_c0':
        if sid is None:
            raise ValueError('mat_rmh_c0 needs sid')
        r = json.loads((V1 / 'samples' / sid / 'params.json').read_text())['skin']['roughness']
        return dict(VARIANTS['normal'], material_overrides=dict(roughness_scale=1.25 * r / MH_ROUGHNESS_MEAN, skin_coat=0.))
    if variant in VARIANTS:
        return dict(VARIANTS[variant])
    if variant.startswith('sw_'):
        for param in SWEEP:
            if variant.startswith(f'sw_{param}_'):
                value = float(variant[len(f'sw_{param}_'):])
                opts = dict(VARIANTS['normal'])
                if param in RENDER_ARGS:
                    opts[param] = value
                else:
                    opts['material_overrides'] = {param: value}
                return opts
    if variant in ('combo', 'combo_is0'):
        # 2026-11-05 winners together: per-identity roughness + coat 0 (mat_rmh_c0) + MH eye placement (ew_eye_place),
        # sclera x.85, eye_map v2; combo_is0 also turns the iris's own specular off
        mo = dict(variant_options('mat_rmh_c0', sid)['material_overrides'], **variant_options('ew_eye_place', sid)['material_overrides'])
        mo['sclera_gain'] = .85        # item 3: sclera x.85 (eye dE -.74, 32/35 eyes, 12/12 heads)
        if variant == 'combo_is0':
            mo['iris_specular'] = 0.
        return dict(VARIANTS['normal_eyes2'], material_overrides=mo)
    if variant == 'ew_eye_place':
        if sid is None:
            raise ValueError('ew_eye_place needs sid')
        off = json.loads((COMPARE_ROOT / sid / 'head' / 'eye_place_offset.json').read_text())
        return dict(VARIANTS['normal_eyes2'], material_overrides={f'eye_offset_{s}_m': off[s] for s in ('left', 'right')})
    if variant == 'ew_shadow0_irisspec0':
        return dict(VARIANTS['normal_eyes2'], material_overrides=dict(cornea_shadow=0., iris_specular=0.))
    if variant == 'ew_shadow0_gain15':
        return dict(VARIANTS['normal_eyes2'], material_overrides=dict(cornea_shadow=0., iris_gain=1.5))
    if variant.startswith('ew_'):
        for param in (*EYE_SWEEP, *EYE_EXTRA_PARAMS):
            if variant.startswith(f'ew_{param}_'):
                return dict(VARIANTS['normal_eyes2'], material_overrides={param: float(variant[len(f'ew_{param}_'):])})
    raise ValueError(f'unknown variant {variant!r}')


def sweep(env='vhstudio', score=False):
    """score=False: render every sweep setting (GPU; finished views are skipped). score=True: score each setting and the
    'normal' baseline against the UE references (run CPU-only, HIP_VISIBLE_DEVICES=) and rank by skin LPIPS."""
    variants = [sweep_variant(p, v) for p, values in SWEEP.items() for v in values]
    if not score:
        for v in variants:
            for sid in SWEEP_IDS:
                print(v, sid, render_sample(sid, SWEEP_CAMERAS, v), flush=True)
        return None
    rows = {v: report(SWEEP_IDS, SWEEP_CAMERAS, env, v, write_summary=False) for v in ['normal'] + variants}
    def mean(summary, key):
        vals = [c[key] for c in summary.values() if isinstance(c.get(key), (int, float))]
        return float(np.mean(vals)) if vals else None
    table = {v: dict(lpips=mean(r, 'lpips'), dists=mean(r, 'dists'), delta_e=mean(r, 'delta_e'), gain=mean(r, 'cycles_to_ue_gain'))
             for v, r in rows.items()}
    ranked = sorted(table, key=lambda v: table[v]['lpips'] if table[v]['lpips'] is not None else 9)
    out = dict(env=env, ids=list(SWEEP_IDS), cameras=list(SWEEP_CAMERAS), sweep={k: list(v) for k, v in SWEEP.items()},
               table=table, ranked_by_lpips=ranked, per_camera=rows)
    (COMPARE_ROOT / f'sweep_{env}.json').write_text(json.dumps(out, indent=1) + '\n')
    return out


# Confirmation of the sweep's candidates (2026-11-04) on all comparison heads and the skin-heavy views.
CONFIRM = ('sw_roughness_scale_1.25', 'sw_skin_coat_0', 'mat_r125_c0')
CONFIRM_CAMERAS = ('front', 'yaw_p40', 'yaw_m40', 'yaw_p90', 'yaw_m90', 'mouth')
# Ear experiment (2026-11-04): the MH normal regressed ear views on 9/12 heads; fade it on the GNM ears.
EYE_SET = dict(variants=('normal_eyes', 'normal_eyes2'), cameras=('eye_left', 'eye_right', 'front', 'yaw_p40', 'yaw_m40'))
EYE_SWEEP_SET = dict(variants=EYE_SWEEP_VARIANTS, cameras=EYE_SWEEP_CAMERAS, baseline='normal_eyes2')
# 2026-11-05: cornea IOR 1.0 = an invisible cornea (no refraction, no reflection): is the cornea veil the eye dE gap?
EYE_IOR1_SET = dict(variants=('ew_cornea_ior_1',), cameras=EYE_SWEEP_CAMERAS, baseline='normal_eyes2')
# 2026-11-05: iris texture brightness (the rendered iris reads grey/dark vs UE): base colour x .7 / 1.5 / 2.5
EYE_IRIS_SET = dict(variants=tuple(eye_variant('iris_gain', g) for g in (.7, 1.5, 2.5)), cameras=EYE_SWEEP_CAMERAS,
                    baseline='normal_eyes2')
# 2026-11-05: iris saturation (Cycles irises read grey vs UE's brown/green/blue): HSV saturation x 1.5 / 2 / 3
EYE_SAT_SET = dict(variants=tuple(eye_variant('iris_sat', s) for s in (1.5, 2, 3)), cameras=EYE_SWEEP_CAMERAS,
                   baseline='normal_eyes2')
# 2026-11-05: the iris is lit only through the glass cornea (caustic paths; Cycles' shadow rays stop at glass), so it
# renders warm/grey from skin bounce. cornea_shadow 0 = cornea + tear film cast no shadows (iris gets direct light);
# + iris_gain 1.5 (the brightness test's best).
# 2026-11-05: GNM eye joints sit ~1.9 mm above (and ~.9 mm medial of) the MetaHuman eye centres; ew_eye_place moves
# the eyeballs by the per-head x/y offset (head/eye_place_offset.json, write_eye_place_offsets).
EYE_PLACE_SET = dict(variants=('ew_eye_place',), cameras=EYE_SWEEP_CAMERAS, baseline='normal_eyes2')


def write_eye_place_offsets(sid, scene_assets):
    """head/eye_place_offset.json: per side [dx, dy, 0] (m) moving the Cycles iris centre onto the MetaHuman eye centre
    (camera target) in x/y; scene_assets = a staged base's scene_assets.npz (scene frame = vhuman + candidate shift)."""
    cams = json.loads((V1 / 'samples' / sid / 'cameras.json').read_text())
    cand = build_candidate(sid)
    shift = np.asarray(json.loads((cand / 'manifest.json').read_text())['geometry']['metahuman_pseudo_candidate']['translation_m'])
    z = np.load(scene_assets)
    off = {}
    for side in ('left', 'right'):
        d = np.asarray(cams['targets'][f'eye_{side}']) - (z[f'{side}_iris_positions'].mean(0) - shift)
        off[side] = [float(d[0]), float(d[1]), 0.]
    (COMPARE_ROOT / sid / 'head' / 'eye_place_offset.json').write_text(json.dumps(off, indent=1) + '\n')
    return off


# item 3: sclera texture saturation (red-veined MH sclera renders solid pink) and brightness
EYE_SCLERA_SET = dict(variants=('ew_sclera_sat_0.5', 'ew_sclera_sat_0', 'ew_sclera_gain_0.85', 'ew_sclera_gain_1.15'),
                      cameras=EYE_SWEEP_CAMERAS, baseline='normal_eyes2')
# the iris albedo is ~.03 linear, so its own F0 ~.04 reflection of the studio sky greys it out (a real iris sits under
# the aqueous humour: almost no specular of its own); iris Specular IOR Level 0, alone and with cornea_shadow 0
EYE_IRISSPEC_SET = dict(variants=('ew_iris_specular_0', 'ew_shadow0_irisspec0'), cameras=EYE_SWEEP_CAMERAS, baseline='normal_eyes2')
EYE_SHADOW_SET = dict(variants=('ew_cornea_shadow_0', 'ew_shadow0_gain15'), cameras=EYE_SWEEP_CAMERAS, baseline='normal_eyes2')
UE_SET = dict(variants=('ue_sss25', 'ue_sss50', 'ue_sss100', 'ue_spec', 'ue_match', 'ue_match_spec'), cameras=('front', 'yaw_p40', 'yaw_m40'))
UE_CONFIRM_SET = dict(variants=('ue_match', 'ue_match_spec', 'mat_r125_c0'), cameras=CONFIRM_CAMERAS)
COMBO_SET = dict(variants=('combo', 'combo_is0'), cameras=CONFIRM_CAMERAS + ('eye_left', 'eye_right'))   # item 8, vs normal
RMH_SET = dict(variants=('mat_rmh_c0',), cameras=CONFIRM_CAMERAS)
EAR_SSS_SET = dict(variants=('ear_sss2', 'ear_sss4'), cameras=('ear_left', 'ear_right', 'yaw_p90', 'yaw_m90', 'front'))   # item 7: ear thin-part transmission (ear cameras)
DUAL_SET = dict(variants=('mat_r125_dual',), cameras=CONFIRM_CAMERAS, baseline='mat_r125_c0')   # item 5: UE dual lobes   # 2026-11-05: per-identity roughness (item 6)
EAR_SET = dict(variants=('', 'normal_ear0', 'normal_ear50'), cameras=('ear_left', 'ear_right', 'yaw_p90', 'yaw_m90', 'front'))


def confirm(ids, env='vhstudio', score=False, profile='full', variants=CONFIRM, cameras=CONFIRM_CAMERAS, name='confirm',
            baseline='normal'):
    """score=False: render CONFIRM x ids x CONFIRM_CAMERAS (GPU). score=True: per-view paired deltas vs 'normal'
    (LPIPS / DISTS / dE), means, and how many views each variant improves -> compare_v1/confirm_<env>.json.
    profile 'fast': score the Windows CPU renders (win_cycles, 512 px / 32 spp, 'normal' included) against each other
    -> confirm_<env>__fast.json; a fast variant is only ever paired with the fast baseline.
    variants/cameras/name: other experiments with the same pairing (EAR_SET -> ears_<env>...json); ear_* metrics
    (parser ear labels only) are compared too when present."""
    if not score:
        for v in variants:
            for sid in ids:
                print(v, sid, render_sample(sid, cameras, v), flush=True)
        return None
    variants = tuple(v for v in variants if v != baseline)
    rows = {v: report(ids, cameras, env, v, write_summary=False, profile=profile) for v in (baseline,) + variants}
    suffix = f'_{env}' if env != 'studio' else ''
    psuf = '' if profile == 'full' else f'__{profile}'
    def load(sid, v):
        path = COMPARE_ROOT / sid / f"compare{suffix}{'_' + v if v else ''}{psuf}.json"
        return json.loads(path.read_text()) if path.is_file() else {}
    per = {v: {sid: load(sid, v) for sid in ids} for v in (baseline,) + variants}
    out = dict(env=env, profile=profile, ids=list(ids), cameras=list(cameras), baseline=baseline, variants={}, per_camera=rows)
    for v in variants:
        d = {k: [] for k in ('lpips', 'dists', 'delta_e') + REGION_KEYS}
        by_id = {}
        for sid in ids:
            for cam in cameras:
                a, b = per[baseline][sid].get(cam), per[v][sid].get(cam)
                if not a or not b or a.get('lpips') is None or b.get('lpips') is None:
                    continue
                for k in d:
                    if isinstance(a.get(k), (int, float)) and isinstance(b.get(k), (int, float)):
                        d[k].append(b[k] - a[k])
                by_id.setdefault(sid, []).append(b['lpips'] - a['lpips'])
        out['variants'][v] = dict(
            n=len(d['lpips']), **{f'n_{r}': len(d[f'{r}_lpips']) for r in REGION_METRICS if d[f'{r}_lpips']}, **{f'mean_d{k}': float(np.mean(x)) for k, x in d.items() if x},
            **{f'improved_{k}': int(np.sum(np.asarray(x) < 0)) for k, x in d.items() if x},
            heads_improved_lpips=int(sum(np.mean(x) < 0 for x in by_id.values())), per_head_dlpips={k: float(np.mean(x)) for k, x in by_id.items()})
    (COMPARE_ROOT / f'{name}_{env}{psuf}.json').write_text(json.dumps(out, indent=1) + '\n')
    return out


def write_eye_fit(sid, version=None):
    """head/fit_mh_eyes[_v<n>].json: the template fit with eye_params mapped from this sample's MetaHuman eyes (eye_map;
    version None = eye_map.VERSION)."""
    from .eye_map import VERSION, params_for
    version = VERSION if version is None else version
    head = COMPARE_ROOT / sid / 'head'
    fit = json.loads((head / 'fit.json').read_text())
    p, rec = params_for(sid, 'left', version)
    fit['eye_params'] = p
    fit['eye_params_source'] = dict(method='metahuman.eye_map', version=version, sample=sid, side='left',
                                    reproduction_delta_e=rec['reproduction_delta_e'])
    path = head / ('fit_mh_eyes.json' if version == 1 else f'fit_mh_eyes_v{version}.json')
    path.write_text(json.dumps(fit, indent=1) + '\n')
    return path


def render_sample(sid, cameras=CAMERAS, variant=''):
    from ..reconstruction.offline_render import render
    cand = build_candidate(sid)
    shift = np.asarray(json.loads((cand / 'manifest.json').read_text())['geometry']['metahuman_pseudo_candidate']['translation_m'])
    cams = json.loads((V1 / 'samples' / sid / 'cameras.json').read_text())['cameras']
    done = []
    for name in cameras:
        out = COMPARE_ROOT / sid / ('cycles' + (f'_{variant}' if variant else '')) / name
        if (out / 'render_result.json').is_file():
            done.append(name); continue
        if out.exists():
            shutil.rmtree(out)
        v = cams[name]['vhuman']
        explicit = dict(origin=(np.asarray(v['origin']) + shift).tolist(), rotation=v['rotation'], focal_px=v['focal_px'], size=v['size'])
        opts = dict(RENDER, **variant_options(variant)); preset = opts.pop('preset'); samples = opts.pop('samples')
        if eye_version(variant):
            opts['head_fit'] = write_eye_fit(sid, eye_version(variant))
        render(cand, out, device='hip', preset=preset, samples=samples, resolution=int(v['size'][0]), explicit_camera=explicit,
               hard_limit_mib=14336, **opts)
        done.append(name)
    return done


REGION_METRICS = {'ear': (('ear_', 'yaw_p90', 'yaw_m90'), ('left_ear', 'right_ear')),
                  'eyes': (('eye_', 'front'), ('left_eye', 'right_eye'))}
REGION_KEYS = tuple(f'{r}_{m}' for r in REGION_METRICS for m in ('lpips', 'dists', 'delta_e'))


def box_down(x, size):
    """Box-downsample an HxWxC image to size x size (integer factor); unchanged when it already has that size."""
    if x.shape[:2] == (size, size):
        return x
    f = x.shape[0] // size
    if f < 1 or x.shape[0] != f * size or x.shape[1] != f * size:
        raise ValueError(f'{x.shape[:2]} is not an integer multiple of {size}')
    return x.reshape(size, f, size, f, -1).mean((1, 3))


def report(ids, cameras=CAMERAS, env='studio', variant='', write_summary=True, profile='full'):
    """Score Cycles vs UE per view. Per-head compare<suffix>.json is merged (cameras not scored here are kept), so
    subset runs (sweep, confirm: write_summary=False) never drop results; summary<suffix>.json only from full runs.
    profile != 'full' reads win_cycles' <variant dir>__<profile> renders and box-downsamples the UE references to their size."""
    from .win_cycles import variant_dir
    from ..face_parsing import FaceParser
    from ..reconstruction import ab_metrics as am
    parser = FaceParser(); rows = {}
    suffix = ('' if env == 'studio' else f'_{env}') + (f'_{variant}' if variant else '') + ('' if profile == 'full' else f'__{profile}')
    for sid in ids:
        rec = {}
        for name in cameras:
            c = COMPARE_ROOT / sid / variant_dir(variant, profile) / name / 'beauty.exr'
            ue = V1 / 'samples' / sid / 'renders' / f'{name}__{env}.exr'
            ub = V1 / 'samples' / sid / 'renders' / f'{name}__basecolor.exr'
            if not (c.is_file() and ue.is_file()):
                continue
            cy = read_exr(c); u = read_exr(ue)[..., :3]; ubc = read_exr(ub)[..., :3]
            u, ubc = box_down(u, cy.shape[0]), box_down(ubc, cy.shape[0])
            um = ubc.max(-1) > 1e-4
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
            r = am.compare_region(a, b, parser, ['skin', 'nose'], aligned=True)
            r.pop('crops', None)
            rec[name] = dict(silhouette_iou_head_rows=float(inter / max(union, 1)), cycles_to_ue_gain=gain,
                             lpips=r.get('lpips'), dists=r.get('dists'), delta_e=(r.get('delta_e') or {}).get('mean'), status=r['status'])
            for key, (prefixes, labels) in REGION_METRICS.items():      # region-only scores (parser labels) on its views
                if name.startswith(prefixes):
                    e = am.compare_region(a, b, parser, list(labels), aligned=True)
                    rec[name].update({f'{key}_lpips': e.get('lpips'), f'{key}_dists': e.get('dists'),
                                      f'{key}_delta_e': (e.get('delta_e') or {}).get('mean'), f'{key}_status': e['status']})
            Image.fromarray(np.concatenate([a, b], 1)).save(COMPARE_ROOT / sid / f'{name}_ue{suffix}_cycles.png')
        rows[sid] = rec
        path = COMPARE_ROOT / sid / f'compare{suffix}.json'
        merged = json.loads(path.read_text()) if path.is_file() else {}
        merged.update(rec)
        path.write_text(json.dumps(merged, indent=1) + '\n')
    summary = {}
    for name in cameras:
        vals = [rows[s][name] for s in rows if name in rows[s]]
        if vals:
            summary[name] = {k: float(np.nanmean([v[k] for v in vals if isinstance(v.get(k), (int, float))] or [np.nan]))
                             for k in ('silhouette_iou_head_rows', 'cycles_to_ue_gain', 'lpips', 'dists', 'delta_e') + REGION_KEYS
                             if any(isinstance(v.get(k), (int, float)) for v in vals)} | dict(n=len(vals))
    if write_summary:
        (COMPARE_ROOT / f'summary{suffix}.json').write_text(json.dumps(dict(ids=list(rows), env=env, variant=variant, cameras=summary), indent=1) + '\n')
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--ids', default=''); ap.add_argument('--every', type=int, default=10)
    ap.add_argument('--cameras', default=','.join(CAMERAS)); ap.add_argument('--build-only', action='store_true')
    ap.add_argument('--report', action='store_true')
    ap.add_argument('--variant', default='', help='Cycles render variant: ' + ', '.join(VARIANTS) + ', or sw_<param>_<value> (sweep)')
    ap.add_argument('--confirm', action='store_true', help='render (or with --report: score) the CONFIRM variants on the selected ids x CONFIRM_CAMERAS')
    ap.add_argument('--sweep', action='store_true', help='material sweep renders (SWEEP x SWEEP_IDS x SWEEP_CAMERAS); with --report: score and rank them')
    ap.add_argument('--update-normal', action='store_true', help='write the converted MH normal into existing pseudo-candidates')
    ap.add_argument('--ears', action='store_true', help='score the EAR_SET experiment (normal_ear* vs normal; renders via win_cycles)')
    ap.add_argument('--eyes', action='store_true', help='score the EYE_SET experiment (MH eye mapping v1/v2 vs normal)')
    ap.add_argument('--ue-match', action='store_true', help='score the UE_SET experiment (UE-derived SSS/specular vs normal)')
    ap.add_argument('--eye-sweep', action='store_true', help='score the EYE_SWEEP experiment (ew_* eye optics vs normal_eyes2)')
    ap.add_argument('--profile', default='full', choices=('full', 'fast'), help="with --confirm --report: 'fast' scores win_cycles' 512 px / 32 spp renders")
    ap.add_argument('--ue-env', default='studio', help="UE reference lighting: studio (MetaHuman Studio) or vhstudio (calibrated vhuman rig)")
    a = ap.parse_args()
    render_ids = sorted(json.loads((V1 / 'sampling.json').read_text())['render_ids'])
    ids = a.ids.split(',') if a.ids else render_ids[::a.every]
    cams = tuple(a.cameras.split(','))
    if a.sweep:
        out = sweep(a.ue_env, score=a.report)
        for v in (out or {}).get('ranked_by_lpips', []):
            print(v, json.dumps(out['table'][v]))
        return
    if a.ears or a.eyes or a.eye_sweep or a.ue_match:
        name, exp = (('ears', EAR_SET) if a.ears else ('eyes', EYE_SET) if a.eyes else ('ue_match', UE_SET) if a.ue_match
                     else ('eye_sweep', EYE_SWEEP_SET))
        out = confirm(a.ids.split(',') if a.eye_sweep and a.ids else EYE_SWEEP_IDS if a.eye_sweep else ids,
                      a.ue_env, score=True, profile=a.profile, name=name, **exp)
        for v, r in out['variants'].items():
            print(v, json.dumps({k: x for k, x in r.items() if k != 'per_head_dlpips'}))
        return
    if a.confirm:
        out = confirm(ids, a.ue_env, score=a.report, profile=a.profile)
        for v, r in (out or {}).get('variants', {}).items():
            print(v, json.dumps({k: x for k, x in r.items() if k != 'per_head_dlpips'}))
        return
    if a.report:
        print(json.dumps(report(ids, cams, a.ue_env, a.variant), indent=1)); return
    if a.update_normal:
        for sid in ids:
            print(sid, 'normal valid', update_normal(sid), flush=True)
        return
    for sid in ids:
        if a.build_only:
            print(sid, build_candidate(sid, force=True), flush=True); continue
        print(sid, render_sample(sid, cams, a.variant), flush=True)


if __name__ == '__main__':
    main()
