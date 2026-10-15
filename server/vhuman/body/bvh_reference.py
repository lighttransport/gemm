"""Import a bounded, explicit BVH motion into a fixed-identity MHR reference.

BVH and derived motion stay outside source control. A source URL and terms
record are required; the importer does not infer a data license from code.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np
from scipy.spatial.transform import Rotation

from .musclemimic_assets import sha256
from .musclemimic_kinematics import AvatarFeet, FEET, MHRKinematics, proportions
from .terra_motion import asset_fingerprint, validate_source
from .terra_terrain import Terrain

JOINTS = ('Hips', 'Spine2', 'Head', 'LeftArm', 'LeftForeArm', 'LeftHand',
          'RightArm', 'RightForeArm', 'RightHand', 'LeftUpLeg', 'LeftLeg',
          'LeftFoot', 'LeftToeBase', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase')


def forward(local, rotations, parents):
    positions, matrices = local.copy(), rotations.copy()
    for j, parent in enumerate(parents):
        if parent >= 0:
            positions[:, j] = positions[:, parent] + np.einsum('tij,tj->ti', matrices[:, parent], local[:, j])
            matrices[:, j] = matrices[:, parent] @ rotations[:, j]
    return positions, matrices


def read_bvh(path, *, unit_scale=.01):
    """Read 3/6-channel joints, preserving their declared intrinsic Euler order.

    Translation channels supply local positions, as in the existing GMR BVH
    reader; joints without translation channels retain their hierarchy offsets.
    """
    path = Path(path)
    if path.stat().st_size > 8 << 20 or not np.isfinite(unit_scale) or unit_scale <= 0:
        raise ValueError('BVH exceeds its size bound or has an invalid unit scale')
    text = path.read_text()
    parts = re.split(r'\bMOTION\b', text)
    if len(parts) != 2:
        raise ValueError('BVH needs one hierarchy and motion section')
    names, parents, offsets, channels, stack = [], [], [], [], []
    pending = None
    for line in parts[0].splitlines():
        words = line.split()
        if not words or words == ['HIERARCHY']:
            continue
        if words[0] in ('ROOT', 'JOINT'):
            if len(words) != 2 or len(names) >= 512 or words[1] in names:
                raise ValueError('invalid or repeated BVH joint')
            parent = stack[-1] if stack else -1
            if parent is None or (words[0] == 'ROOT') != (not stack):
                raise ValueError('invalid BVH joint hierarchy')
            names.append(words[1]); parents.append(parent)
            offsets.append(None); channels.append(None); pending = len(names)-1
        elif words == ['End', 'Site']:
            pending = None
        elif words == ['{']:
            stack.append(pending)
        elif words == ['}']:
            if not stack:
                raise ValueError('unbalanced BVH hierarchy')
            stack.pop()
        elif words[0] == 'OFFSET' and len(words) == 4 and stack:
            if stack[-1] is not None:
                offsets[stack[-1]] = list(map(float, words[1:]))
        elif words[0] == 'CHANNELS' and len(words) >= 3 and stack and stack[-1] is not None:
            count = int(words[1]); values = words[2:]
            if count != len(values) or count not in (3, 6) or len(set(values)) != count:
                raise ValueError('BVH requires three rotation channels and optional XYZ position channels')
            rotation = [v for v in values if v.endswith('rotation')]
            position = [v for v in values if v.endswith('position')]
            if (set(rotation) != {'Xrotation', 'Yrotation', 'Zrotation'} or
                    (position and set(position) != {'Xposition', 'Yposition', 'Zposition'})):
                raise ValueError('unsupported BVH channels')
            channels[stack[-1]] = values
        else:
            raise ValueError('unsupported BVH hierarchy statement: '+line.strip())
    if stack or not names or parents.count(-1) != 1 or any(v is None for v in offsets+channels):
        raise ValueError('incomplete BVH hierarchy')
    match = re.match(r'\s*Frames:\s*(\d+)\s+Frame Time:\s*([\d.eE+-]+)\s*', parts[1])
    if match is None:
        raise ValueError('missing BVH frame count or clock')
    frames, dt = int(match[1]), float(match[2])
    if not 5 <= frames <= 240 or not np.isfinite(dt) or not 0 < dt <= 1 or (frames-1)*dt > 8:
        raise ValueError('BVH must contain 5..240 frames spanning at most eight seconds')
    width = sum(map(len, channels))
    values = np.asarray(parts[1][match.end():].split(), dtype=float)
    offsets = np.asarray(offsets, dtype=float)*unit_scale
    if values.size != frames*width or not np.isfinite(values).all() or not np.isfinite(offsets).all():
        raise ValueError('BVH frame values are incomplete or nonfinite')
    values = values.reshape(frames, width)
    local = np.repeat(offsets[None], frames, axis=0)
    rotations = np.empty((frames, len(names), 3, 3))
    cursor = 0
    for j, declared in enumerate(channels):
        block = values[:, cursor:cursor+len(declared)]; cursor += len(declared)
        axes, angles = [], []
        for column, channel in enumerate(declared):
            axis = 'XYZ'.index(channel[0])
            if channel.endswith('position'):
                local[:, j, axis] = block[:, column]*unit_scale
            else:
                axes.append(channel[0]); angles.append(block[:, column])
        rotations[:, j] = Rotation.from_euler(''.join(axes), np.stack(angles, axis=-1), degrees=True).as_matrix()
    positions, matrices = forward(local, rotations, parents)
    rest, rest_matrices = forward(offsets[None], np.tile(np.eye(3), (1, len(names), 1, 1)), parents)
    return dict(names=names, parents=parents, positions=positions, matrices=matrices,
                rest_positions=rest[0], rest_matrices=rest_matrices[0], times=np.arange(frames)*dt)


def convert(source, avatar, assets, out, *, source_url, terms_url, unit_scale=.01):
    source, avatar, assets, out = map(Path, (source, avatar, assets, out))
    if out.exists():
        raise ValueError('use a fresh BVH reference output directory')
    data = read_bvh(source, unit_scale=unit_scale)
    missing = set(JOINTS)-set(data['names'])
    if missing:
        raise ValueError('BVH anatomical mapping lacks: '+', '.join(sorted(missing)))
    indices = [data['names'].index(name) for name in JOINTS]
    definition = json.loads((avatar/'avatar.json').read_text())
    meta = json.loads((avatar/'body_mhr.glb.json').read_text())
    fk = MHRKinematics(definition, assets, meta['model_params'])
    skin = AvatarFeet(avatar/'avatar.glb', fk)
    target_rest = fk.forward(fk.bind_params)[0][fk.sites]
    rest = data['rest_positions'][indices]
    height = rest[2, 1] - rest[FEET, 1].min()
    target_height = target_rest[2, 1] - target_rest[FEET, 1].min()
    if min(height, target_height) < .2:
        raise ValueError('degenerate BVH/avatar stature')
    scale = target_height/height
    positions = data['positions'][:, indices]*scale
    rest = rest*scale
    times = data['times']
    feet = positions[:, FEET]
    speed = np.linalg.norm(np.gradient(feet, times, axis=0), axis=-1)
    clearance = feet[..., 1] - np.percentile(feet[..., 1], 5, axis=0)
    contacts = (clearance < .04) & (speed < .30)
    # Establish one flat-floor datum after segment-length calibration. Passing
    # actual support geometry enables pose-aware sole constraints in the fitter;
    # a constant ankle-to-floor offset misses penetration when a foot rotates.
    targets = proportions(positions, rest, target_rest)
    offset = target_rest[0] - targets[0, 0]
    offset[1] = 0
    targets += offset
    sole = target_rest[FEET, 1] - skin.floor
    targets[..., 1] -= np.min(targets[:, FEET, 1] - sole)
    _, roots, params, fitting = fk.retarget(targets, data['matrices'][:, indices], contacts, times,
        source_rest=target_rest, source_rest_matrices=data['rest_matrices'][indices], skin=skin,
        terrain=Terrain({'boxes': []}))
    world = np.array([fk.forward(p, root)[0][fk.root] for p, root in zip(params, roots)])
    reference = validate_source(dict(model_params=params, identity=np.array(meta['shape']), times=times,
                                    root_world=world, mhr_fingerprint=asset_fingerprint(assets)))
    out.mkdir(parents=True)
    np.savez(out/'reference.npz', **reference)
    np.savez(out/'source-landmarks.npz', positions=positions, matrices=data['matrices'][:, indices],
             rest_positions=rest, rest_matrices=data['rest_matrices'][indices], times=times, contacts=contacts)
    report = dict(format='vhuman.bvh_mhr_reference.v1', source=str(source.resolve()),
        source_sha256=sha256(source), source_url=source_url, terms_url=terms_url,
        unit_scale=unit_scale, stature_scale=scale, joint_mapping=list(JOINTS),
        frames=len(times), duration=float(times[-1]), mhr_fingerprint=reference['mhr_fingerprint'],
        reference_sha256=sha256(out/'reference.npz'), importer_sha256=sha256(Path(__file__)),
        ground='flat', fitting=fitting,
        perceptual_review='pending', research_only=True)
    (out/'provenance.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--avatar', type=Path, required=True)
    ap.add_argument('--mhr-assets', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--source-url', required=True)
    ap.add_argument('--terms-url', required=True)
    ap.add_argument('--unit-scale', type=float, default=.01)
    args = ap.parse_args()
    print(json.dumps(convert(args.source, args.avatar, args.mhr_assets, args.out,
        source_url=args.source_url, terms_url=args.terms_url, unit_scale=args.unit_scale), indent=2))


if __name__ == '__main__':
    main()
