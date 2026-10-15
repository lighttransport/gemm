"""Offline flat-ground BVH -> constrained Terra policy -> native MHR avatar.

Preserves the recording's root and clock without an intermediate MHR pose fit.
Media, derived motion and model assets belong in ignored local storage.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .bvh_reference import JOINTS, read_bvh
from .musclemimic_assets import sha256
from .musclemimic_kinematics import AvatarFeet, FEET, MHRKinematics, contact_metrics, proportions
from .terra_motion import JOINT_NAMES


def prepare(source, avatar, assets, *, source_url, terms_url, unit_scale=.01):
    source, avatar, assets = map(Path, (source, avatar, assets))
    if not source_url or not terms_url:
        raise ValueError('BVH recording requires explicit source and terms records')
    data = read_bvh(source, unit_scale=unit_scale)
    if set(JOINTS)-set(data['names']):
        raise ValueError('BVH anatomical mapping is incomplete')
    indices = [data['names'].index(n) for n in JOINTS]
    definition = json.loads((avatar/'avatar.json').read_text())
    metadata = json.loads((avatar/'body_mhr.glb.json').read_text())
    fk = MHRKinematics(definition, assets, metadata['model_params'])
    skin = AvatarFeet(avatar/'avatar.glb', fk)
    target_rest = fk.forward(fk.bind_params)[0][fk.sites]
    rest = data['rest_positions'][indices]
    heights = [p[2, 1]-p[FEET, 1].min() for p in (rest, target_rest)]
    if min(heights) < .2:
        raise ValueError('degenerate BVH/avatar stature')
    scale = heights[1]/heights[0]
    measured = data['positions'][:, indices]*scale
    feet, times = measured[:, FEET], data['times']
    speed = np.linalg.norm(np.gradient(feet, times, axis=0), axis=-1)
    clearance = feet[..., 1]-np.percentile(feet[..., 1], 5, axis=0)
    contacts = (clearance < .04)&(speed < .30)
    evidence = contact_metrics(feet, contacts)
    if not evidence['stance_samples'] or evidence['stance_drift_p95_m'] > .02:
        raise ValueError(f'BVH source stance anchors rejected: {evidence}')
    positions = proportions(measured, rest*scale, target_rest)
    calibrated_rest = proportions((rest*scale)[None], rest*scale, target_rest)[0]
    offset = target_rest[0]-positions[0, 0]; offset[1] = 0
    positions += offset
    sole = target_rest[FEET, 1]-skin.floor
    positions[..., 1] -= np.min(positions[:, FEET, 1]-sole)
    prepared = dict(positions=positions, matrices=data['matrices'][:, indices],
        rest_positions=calibrated_rest, rest_matrices=data['rest_matrices'][indices],
        times=times, fps=1/float(times[1]-times[0]),
        params=np.repeat(fk.bind_params[None], len(times), axis=0),
        sole_offsets={JOINT_NAMES[j]:float(v) for j,v in zip(FEET,sole)},
        neutral_foot_pitch={side:float(np.degrees(np.arctan2(
            (calibrated_rest[toe]-calibrated_rest[ankle])[1],
            np.linalg.norm((calibrated_rest[toe]-calibrated_rest[ankle])[[0,2]]))))
            for side,ankle,toe in (('L',11,12),('R',15,16))})
    provenance = dict(path=str(source.resolve()), sha256=sha256(source), source_url=source_url,
        terms_url=terms_url, format='bvh', unit_scale=unit_scale, stature_scale=scale,
        clock_fps=prepared['fps'], source_contact_evidence=evidence,
        anatomical_calibration='native avatar segment lengths and sole offsets',
        intermediate_mhr_fit=False, importer_sha256=sha256(Path(__file__)))
    return prepared, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--avatar', type=Path, required=True, help='native avatar directory')
    parser.add_argument('--mhr-assets', type=Path, required=True)
    parser.add_argument('--terra-assets', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source-url', required=True)
    parser.add_argument('--terms-url', required=True)
    parser.add_argument('--unit-scale', type=float, default=.01)
    parser.add_argument('--policy-actions', choices=('mean','sampled'), default='mean')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if args.out.exists() or args.seed < 0:
        parser.error('use a fresh output directory and nonnegative seed')
    args.out.mkdir(parents=True)
    from .musclemimic_worker import configure_process
    configure_process(args.out, 'cpu')
    from .terra_assets import read_installation
    from .terra_worker import run, export_tracks
    request = dict(avatar=str(args.avatar/'avatar.glb'), avatar_json=str(args.avatar/'avatar.json'),
        body_meta=str(args.avatar/'body_mhr.glb.json'), mhr_assets=str(args.mhr_assets),
        terrain='flat', policy_actions=args.policy_actions, seed=args.seed)
    report = dict(completed=False, avatar=dict(accepted=False))
    try:
        installation = read_installation(args.terra_assets, verify=True)
        source, provenance = prepare(args.source, args.avatar, args.mhr_assets,
            source_url=args.source_url, terms_url=args.terms_url, unit_scale=args.unit_scale)
        report = run(request, installation, args.out, prepared_source=source, source_provenance=provenance)
        report['avatar'] = export_tracks(request, args.out, report)
    except Exception as error:
        import traceback
        traceback.print_exc()
        report.update(completed=False, error=str(error))
    (args.out/'simulation.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    accepted = report.get('avatar', {}).get('accepted', False)
    print(json.dumps(dict(completed=report['completed'], accepted=accepted, error=report.get('error'))))
    return 0 if accepted else 2


if __name__ == '__main__':
    raise SystemExit(main())
