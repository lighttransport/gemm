"""Host-side access to generated samples: meshes split into head/eyes/teeth, textures, parameters, landmarks.

Sample layout (ue_scripts/gen_batch.py): samples/<id>/{params.json, landmarks.json, done.json, meshes/<comp>.bin, tex/<type>.png}.
Frames (verified 2026-11-03 from named landmarks and renders): Unreal centimetres, +X = subject's left (`*_l` landmarks),
+Y = facing direction, Z up. Head vertices are the first 24,049 vertices of the 'Face' component in DNA order, so the
identity-template vertex ids (face_landmarks.json, symmetry.json) index them directly.
The MetaHuman preview 'Face' component holds the head (24,049 vertices), teeth (4,246), two eyes (770 each) and small
lash/shell parts as separate connected components; `split_face` returns them by size and position.
"""
import json
from pathlib import Path

import numpy as np

from . import DATASET_ROOT
from .mesh_io import connected_components, read_mesh_bin

V1 = DATASET_ROOT / 'v1'
HEAD_VERTS, TEETH_VERTS, EYE_VERTS = 24049, 4246, 770
MH_TO_VH = np.array([[1., 0., 0.], [0., 0., 1.], [0., 1., 0.]], np.float32)   # UE (x subject-left, y forward, z up) -> vhuman (x left, y up, z forward); det -1: UE is left-handed
MH_SCALE = 0.01


def sample_dirs(only_done=True, ids=None):
    out = []
    for d in sorted((V1 / 'samples').iterdir()):
        if ids and d.name not in ids:
            continue
        if only_done:
            f = d / 'done.json'
            if not f.is_file() or not json.loads(f.read_text()).get('ok'):
                continue
        out.append(d)
    return out


def split_face(verts, tris):
    """Return dict(head=(V,T), teeth=(V,T), eye_l=(V,T), eye_r=(V,T), other=[...]) with triangles re-indexed per part.

    The head is checked to be vertices 0..24048 (DNA order). Eyes are labelled by the sign of x (+X = subject's left).
    """
    labels, sizes = connected_components(len(verts), tris)
    parts = {}
    other = []
    eyes = []
    for k in np.argsort(-sizes):
        m = labels == k
        idx = np.flatnonzero(m)
        remap = -np.ones(len(verts), np.int64); remap[idx] = np.arange(len(idx))
        tm = m[tris].all(1)
        part = (verts[idx].copy(), remap[tris[tm]])
        n = len(idx)
        if n == HEAD_VERTS and 'head' not in parts:
            if idx[0] != 0 or idx[-1] != HEAD_VERTS - 1:
                raise ValueError('head component is not vertices 0..24048: DNA vertex order assumption broken')
            parts['head'] = part
        elif n == TEETH_VERTS and 'teeth' not in parts:
            parts['teeth'] = part
        elif n == EYE_VERTS:
            eyes.append(part)
        else:
            other.append(part)
    if len(eyes) == 2:
        eyes.sort(key=lambda p: p[0][:, 0].mean())
        parts['eye_r'], parts['eye_l'] = eyes            # +X is the subject's left
    parts['other'] = other
    return parts


def load_sample(d):
    d = Path(d)
    params = json.loads((d / 'params.json').read_text())
    landmarks = np.asarray(json.loads((d / 'landmarks.json').read_text())['landmarks'], np.float32)
    meshes = {}
    for f in sorted((d / 'meshes').glob('*.bin')):
        meshes[f.stem] = read_mesh_bin(f)
    face = split_face(*meshes['Face']) if 'Face' in meshes else None
    return dict(id=d.name, dir=d, params=params, landmarks=landmarks, meshes=meshes, face=face)


def texture(d, kind):
    from PIL import Image
    return np.asarray(Image.open(Path(d) / 'tex' / f'{kind}.png').convert('RGBA'))


def to_vhuman(points_cm):
    """UE cm -> vhuman metres frame (x subject-left, y up, z toward the camera in front of the face)."""
    return (np.asarray(points_cm, np.float32) @ MH_TO_VH.T) * MH_SCALE


def head_vertices(sample):
    return sample['face']['head'][0]
