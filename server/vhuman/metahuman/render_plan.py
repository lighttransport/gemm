"""Host side of the MetaHuman reference renders: per-sample camera sets in vhuman's own camera convention, converted to UE.

usage: python -m server.vhuman.metahuman.render_plan [--ids id,..] [--cameras front,yaw_p40,...] [--size 1024]
For each sample with render=true (or --ids) and a finished geometry pass, writes samples/<id>/cameras.json with, per camera:
  vhuman record  {origin (m), rotation rows right/up/back, focal_px, size}  -- identical to render_result.json['camera'] so
                 the Cycles side can render the same view unchanged
  ue record      {location_cm, forward, up, fov_deg}                         -- consumed by ue_scripts/render_batch.py
The MetaHuman 'portrait' camera mirrors the Creator thumbnail: head centre + 1.25 m along the facing direction, 18 deg FOV.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from ..reconstruction import render_cameras as rc
from .samples import V1, load_sample, sample_dirs, to_vhuman

DEFAULT_CAMERAS = ('front', 'yaw_p40', 'yaw_m40', 'yaw_p90', 'yaw_m90', 'ear_left', 'ear_right', 'eye_left', 'eye_right', 'mouth')
FOV_DEG = 18.0017
DISTANCE_M = 1.25        # 18 deg at 1.25 m frames the full head incl. neck (vhuman head target = skin bbox centre)
FACE_LANDMARKS = Path('/mnt/nvme02/local/ue/Engine/Plugins/MetaHuman/MetaHumanCharacter/Content/Face/IdentityTemplate/face_landmarks.json')
EAR_CURVES = ('crv_ear_outer_helix', 'crv_ear_inner_helix', 'ear_concha', 'ear_tragus', 'ear_antitragus')


def vh_to_ue(p):
    """vhuman metres (x left, y up, z forward) -> UE cm (x left, y forward, z up)."""
    p = np.asarray(p, float)
    return np.stack([p[..., 0], p[..., 2], p[..., 1]], -1) * 100.


def ear_points(head_vh):
    fl = json.loads(FACE_LANDMARKS.read_text())
    out = {}
    for side, suffix in (('left', '_l'), ('right', '_r')):
        ids = np.concatenate([np.atleast_1d(fl[c + suffix].get('vIDs', fl[c + suffix].get('vID'))) for c in EAR_CURVES]).astype(int)
        out[side] = head_vh[ids]
    return out


def plan_sample(d, names=DEFAULT_CAMERAS, size=1024):
    s = load_sample(d)
    head = to_vhuman(s['face']['head'][0])
    eyes = {'left': to_vhuman(s['face']['eye_l'][0]), 'right': to_vhuman(s['face']['eye_r'][0])}
    pts = rc.targets(head, eyes=eyes, ears=ear_points(head))
    focal = .5 * size / np.tan(np.deg2rad(FOV_DEG) / 2)
    origin = pts['head'] + np.array([0, 0, DISTANCE_M])
    rotation = np.eye(3)                                    # rows right(+x), up(+y), back(+z): camera looks along -z
    cams = {}
    for name in names:
        o, r, f = rc.place(name, origin, rotation, focal, pts)
        fwd = -r[2]
        cams[name] = dict(vhuman=dict(origin=o.tolist(), rotation=r.tolist(), focal_px=float(f), size=[size, size], units='metres'),
                          ue=dict(location_cm=vh_to_ue(o).tolist(), forward=vh_to_ue(fwd).tolist(), up=vh_to_ue(r[1]).tolist(),
                                  fov_deg=float(np.rad2deg(2 * np.arctan(.5 * size / f)))))
    rec = dict(id=s['id'], size=size, portrait=dict(fov_deg=FOV_DEG, distance_m=DISTANCE_M), targets={k: v.tolist() for k, v in pts.items()}, cameras=cams)
    (Path(d) / 'cameras.json').write_text(json.dumps(rec, indent=1) + '\n')
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--ids', default=''); ap.add_argument('--cameras', default=','.join(DEFAULT_CAMERAS)); ap.add_argument('--size', type=int, default=1024)
    a = ap.parse_args()
    sampling = json.loads((V1 / 'sampling.json').read_text())
    ids = set(a.ids.split(',')) if a.ids else set(sampling['render_ids'])
    n = 0
    for d in sample_dirs(ids=ids):
        plan_sample(d, tuple(a.cameras.split(',')), a.size); n += 1
    print(f'wrote cameras.json for {n} samples')


if __name__ == '__main__':
    main()
