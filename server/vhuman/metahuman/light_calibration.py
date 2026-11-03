"""ColorChecker light calibration between vhuman's Cycles studio rig and UE (MetaHuman reference renders).

usage: python -m server.vhuman.metahuman.light_calibration plan|cycles|ue|fit|all [--ref mh_000000]
       python -m server.vhuman.metahuman.light_calibration batch [--every 10]   (render_batches/vhstudio.json)

A matte 24-patch ColorChecker (color_chart.PATCHES, same reflectances in both renderers) is placed where the reference
sample's face is, facing each wide reference camera (front, yaw +-40, yaw +-90), and captured with that exact camera:
  cycles: vhuman studio rig, one light at a time (key, fill, rim, world) and all together (chart_probe_blender.py; the
          scene is translated to compare_render.HEAD_ANCHOR exactly like the pseudo-candidates)
  ue:     the MetaHuman 'Studio' environment used for the references, then a UE replica of the vhuman rig (three
          equal-area RectLights at the same relative positions/aim, environment lights hidden), one light at a time and
          all together (ue_scripts/chart_probe.py)
fit: per replica light, gain = Cycles/UE ratio of the neutral patches (one scalar per light: geometry, falloff and
     orientation are replicated, so the ratio must agree across probes; its spread is the replica check), the studio
     world's share of the Cycles chart, and the Studio-vs-rig ratio per probe (why the first comparison's Cycles->UE gain
     differed per side). Writes calibration.json with the calibrated RectLight intensities for render_batch's 'VHStudio'.
Outputs: DATASET_ROOT/light_calib_v1/ (renders and JSON; nothing from Epic content is stored).
"""
import argparse
import ast
import json
import os
import subprocess
from pathlib import Path

import numpy as np

from . import DATASET_ROOT
from .compare_render import HEAD_ANCHOR
from .images import read_exr
from .render_plan import vh_to_ue
from .samples import V1
from ..reconstruction import color_chart

OUT = DATASET_ROOT / 'light_calib_v1'
PROBES = ('front', 'yaw_p40', 'yaw_m40', 'yaw_p90', 'yaw_m90')
FACE_OFFSET_M = .08          # chart sits ~ at the skin surface in front of the head centre, along the view
PATCH_M, GAP_M = .018, .003
CYCLES_CONDITIONS = ('key', 'fill', 'rim', 'world', 'all')
UE_CONDITIONS = ('studio', 'key', 'fill', 'rim', 'replica')
CYCLES_SCENE = Path(__file__).resolve().parents[1] / 'reconstruction' / 'cycles_scene.py'


def studio_rig():
    """STUDIO_LIGHTS / STUDIO_WORLD / STUDIO_ENERGY_SCALE read from cycles_scene.py (it imports bpy, so no import)."""
    found = {}
    for node in ast.parse(CYCLES_SCENE.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and getattr(node.targets[0], 'id', '') in (
                'STUDIO_LIGHTS', 'STUDIO_WORLD', 'STUDIO_ENERGY_SCALE'):
            found[node.targets[0].id] = eval(compile(ast.Expression(node.value), 'cycles_scene', 'eval'), {'__builtins__': {}})
    return found


def project(points, cam):
    """vhuman pinhole (rows right/up/back, looks along -back) -> pixel x, y (y down)."""
    r = np.asarray(cam['rotation'], float); p = (np.asarray(points, float) - np.asarray(cam['origin'], float)) @ r.T
    depth = -p[..., 2]; w, h = cam['size']
    return np.stack([cam['focal_px'] * p[..., 0] / depth + w / 2, h / 2 - cam['focal_px'] * p[..., 1] / depth], -1), depth


def replica_records(head, rig, gains=None):
    """UE RectLight records of the vhuman lamp rig around a MetaHuman head (vhuman frame, metres). The Cycles lamps sit at
    fixed world positions aiming at the world origin while heads sit at HEAD_ANCHOR, so in the MetaHuman frame each lamp is
    at p - HEAD_ANCHOR + head and aims at head - HEAD_ANCHOR. gains: fitted {name: gain} (None = initial guess P/pi cd)."""
    aim = np.asarray(head, float) - HEAD_ANCHOR
    out = []
    for name, pos, power, size in rig['STUDIO_LIGHTS']:
        p = np.asarray(pos, float) + aim
        side = size * np.sqrt(np.pi) / 2                         # square of the disk's area
        w = power * rig['STUDIO_ENERGY_SCALE']
        out.append(dict(name=name, location_cm=vh_to_ue(p).tolist(), direction=(vh_to_ue((aim - p) / np.linalg.norm(aim - p)) / 100).tolist(),
                        width_cm=side * 100, power_w=w, intensity_cd=w / np.pi * (gains[name] if gains else 1.)))
    return out


def replica_batch(ids, path):
    """render_batch.py batch for env 'VHStudio' with the calibrated rig placed around each sample's head."""
    cal = json.loads((OUT / 'calibration.json').read_text())
    gains = {k: v['gain'] for k, v in cal['lights'].items()}
    rig = studio_rig()
    lights = {sid: replica_records(json.loads((V1 / 'samples' / sid / 'cameras.json').read_text())['targets']['head'], rig, gains) for sid in ids}
    Path(path).write_text(json.dumps(dict(ids=list(ids), env='VHStudio', lights=lights, calibration=str(OUT / 'calibration.json')), indent=1) + '\n')
    return path


def plan(ref):
    cams = json.loads((V1 / 'samples' / ref / 'cameras.json').read_text())
    head = np.asarray(cams['targets']['head'], float)
    shift = HEAD_ANCHOR - head
    centres, _ = color_chart.layout(PATCH_M, GAP_M)
    rig = studio_rig()
    probes = {}
    for name in PROBES:
        v = cams['cameras'][name]['vhuman']; r = np.asarray(v['rotation'], float)
        right, up, back = r
        centre = head + back * FACE_OFFSET_M
        pts = centre + centres[:, :1] * right + centres[:, 1:] * up
        px, depth = project(pts, v)
        half = v['focal_px'] * PATCH_M / 2 / depth
        probes[name] = dict(
            pixels=[dict(name=n, pixel_center=px[i].tolist(), pixel_half_size=float(half[i])) for i, (n, _) in enumerate(color_chart.PATCHES)],
            cycles=dict(camera=dict(v, origin=(np.asarray(v['origin']) + shift).tolist()), patches=(pts + shift).tolist(),
                        right=right.tolist(), up=up.tolist(), normal=back.tolist()),
            ue=dict(camera=cams['cameras'][name]['ue'], patches=vh_to_ue(pts).tolist(), right=(vh_to_ue(right) / 100).tolist(),
                    normal=(vh_to_ue(back) / 100).tolist()))
    replica = replica_records(head, rig)
    rec = dict(ref=ref, head_vh=head.tolist(), shift_m=shift.tolist(), patch_m=PATCH_M, size=cams['size'],
               albedo_linear=color_chart.reference_linear().tolist(), rig={k: (list(v) if isinstance(v, tuple) else v) for k, v in rig.items()},
               replica=replica, probes=probes, out=str(OUT))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'plan.json').write_text(json.dumps(rec, indent=1) + '\n')
    return rec


def run_cycles():
    from ..reconstruction.offline_render import DEFAULT_BLENDER
    env = dict(os.environ, TMPDIR=str(OUT / 'cache'))
    (OUT / 'cache').mkdir(parents=True, exist_ok=True)
    with (OUT / 'cycles.log').open('w') as log:
        subprocess.run([str(DEFAULT_BLENDER), '-b', '--factory-startup', '--python-exit-code', '1', '--python',
                        str(Path(__file__).with_name('chart_probe_blender.py')), '--', str(OUT / 'plan.json'), str(OUT / 'cycles')],
                       stdout=log, stderr=subprocess.STDOUT, env=env, check=True)


def run_ue():
    from .launch import run
    result = OUT / 'ue_result.json'
    code, ok, log = run('chart_probe.py', [str(OUT / 'plan.json')], timeout=1800, result=result)
    if not ok:
        raise RuntimeError(f'UE chart probe failed (exit {code}); log {log}')


def chart_values(path, pixels):
    return color_chart.sample(read_exr(path)[..., :3].astype(np.float64), pixels)


def fit():
    p = json.loads((OUT / 'plan.json').read_text())
    n = list(color_chart.NEUTRALS); luma = color_chart.LUMA
    val = {}
    for eng, conds in (('cycles', CYCLES_CONDITIONS), ('ue', UE_CONDITIONS)):
        for c in conds:
            for probe in PROBES:
                f = OUT / eng / f'{c}__{probe}.exr'
                if f.is_file():
                    val[eng, c, probe] = chart_values(f, p['probes'][probe]['pixels'])
    lum = lambda v: float((v[n] @ luma).sum())
    rec = dict(ref=p['ref'], probes=list(PROBES), neutral_patches=[color_chart.PATCHES[i][0] for i in n])
    # 1. per-light gains (Cycles / UE replica), their spread over the probes that the light reaches
    lights = {}
    for name in ('key', 'fill', 'rim'):
        ratios, rgb = {}, []
        for probe in PROBES:
            cy, ue = val.get(('cycles', name, probe)), val.get(('ue', name, probe))
            if cy is None or ue is None or lum(cy) < 1e-3 * lum(val[('cycles', 'all', probe)]):
                continue
            ratios[probe] = lum(cy) / max(lum(ue), 1e-12)
            rgb.append(cy[n].sum(0) / np.maximum(ue[n].sum(0), 1e-12))
        w = np.array([lum(val[('cycles', name, q)]) for q in ratios])       # weight probes by how much the light contributes
        g = float(np.exp((w * np.log(list(ratios.values()))).sum() / w.sum())) if ratios else None
        lights[name] = dict(gain=g, per_probe=ratios, log2_spread=float(np.std(np.log2(list(ratios.values())))) if ratios else None,
                            rgb_gain=(np.mean(rgb, 0) / np.mean(rgb, 0)[1]).tolist() if rgb else None)
    rec['lights'] = lights
    # 2. the studio world's share of the Cycles chart (the replica has no sky light)
    rec['cycles_world_fraction'] = {q: lum(val[('cycles', 'world', q)]) / lum(val[('cycles', 'all', q)]) for q in PROBES if ('cycles', 'world', q) in val}
    # 3. superposition check: the full replica vs the sum of its single-light captures (UE linearity)
    rec['ue_superposition'] = {q: lum(val[('ue', 'replica', q)]) / max(sum(lum(val[('ue', l, q)]) for l in lights), 1e-12)
                               for q in PROBES if all(('ue', c, q) in val for c in ('replica', 'key', 'fill', 'rim'))}
    # 4. predicted calibrated replica vs Cycles 'all' (world excluded), and UE Studio vs Cycles per probe
    pred = {}
    for q in PROBES:
        if all(('ue', l, q) in val for l in lights) and all(lights[l]['gain'] for l in lights):
            s = sum(lights[l]['gain'] * val[('ue', l, q)] for l in lights)
            pred[q] = lum(s) / lum(val[('cycles', 'all', q)] - val.get(('cycles', 'world', q), 0))
    rec['calibrated_replica_over_cycles_lamps'] = pred
    rec['studio_over_cycles'] = {}
    for q in PROBES:
        if ('ue', 'studio', q) in val:
            st, cy = val[('ue', 'studio', q)], val[('cycles', 'all', q)]
            rgb = st[n].sum(0) / cy[n].sum(0)
            rec['studio_over_cycles'][q] = dict(luma=lum(st) / lum(cy), rgb=(rgb / rgb[1]).tolist(),
                                                studio_wb=color_chart.calibrate(st)['wb_gains'])
    # 5. colour rendering of each engine (exposure/WB-normalised chart dE): a check that both read the chart alike
    rec['chart_delta_e'] = {f'{e}:{c}': float(np.mean([color_chart.calibrate(val[(e, c, q)])['delta_e_mean'] for q in PROBES if (e, c, q) in val]))
                            for e, c in (('cycles', 'all'), ('ue', 'replica'), ('ue', 'studio')) if any((e, c, q) in val for q in PROBES)}
    rec['ue_replica'] = [dict(r, intensity_cd=r['intensity_cd'] * lights[r['name']]['gain']) for r in p['replica'] if lights[r['name']]['gain']]
    (OUT / 'calibration.json').write_text(json.dumps(rec, indent=1) + '\n')
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('step', choices=('plan', 'cycles', 'ue', 'fit', 'all', 'batch'))
    ap.add_argument('--ref', default='mh_000000')
    ap.add_argument('--every', type=int, default=10, help='batch: compare ids = render_ids[::every]')
    a = ap.parse_args()
    if a.step in ('plan', 'all'):
        plan(a.ref)
    if a.step in ('cycles', 'all'):
        run_cycles()
    if a.step in ('ue', 'all'):
        run_ue()
    if a.step in ('fit', 'all'):
        print(json.dumps(fit(), indent=1))
    if a.step == 'batch':
        ids = sorted(json.loads((V1 / 'sampling.json').read_text())['render_ids'])[::a.every]
        print(replica_batch(ids, V1 / 'render_batches' / 'vhstudio.json'))


if __name__ == '__main__':
    main()
