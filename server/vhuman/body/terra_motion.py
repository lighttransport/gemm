"""Native MHR source preparation for the isolated Terra worker."""
from __future__ import annotations

import hashlib
from pathlib import Path
import zipfile

import numpy as np

from . import mhr
from .musclemimic_kinematics import LANDMARKS, FEET, change_basis, resample

# Same anatomical meaning as the existing MHR/Myo mapping; only labels are
# translated into Terra's landmark convention. No SMPL model is evaluated.
JOINT_NAMES = ("Pelvis", "Spine", "Head", "L_Shoulder", "L_Elbow", "L_Wrist",
               "R_Shoulder", "R_Elbow", "R_Wrist", "L_Hip", "L_Knee", "L_Ankle", "L_Toe",
               "R_Hip", "R_Knee", "R_Ankle", "R_Toe")


def asset_fingerprint(assets):
    digest = hashlib.sha256()
    for name in mhr.ASSET_FILES:
        digest.update(name.encode())
        with (Path(assets) / name).open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def validate_source(data, fingerprint=None):
    params, identity = np.asarray(data["model_params"], dtype=float), np.asarray(data["identity"], dtype=float)
    times, root = np.asarray(data["times"], dtype=float), np.asarray(data["root_world"], dtype=float)
    if (times.ndim != 1 or not 5 <= len(times) <= 240 or params.shape != (len(times), 204)
            or identity.shape != (45,) or root.shape != (len(times), 3)
            or not all(np.isfinite(x).all() for x in (params, identity, times, root))
            or abs(times[0]) > 1e-6 or np.any(np.diff(times) <= 0) or times[-1] > 8
            or np.max(abs(root)) > 1000):
        raise ValueError("invalid bounded MHR reference dimensions, values or timestamps")
    token = np.asarray(data["mhr_fingerprint"])
    if token.shape != () or token.dtype.kind != "U":
        raise ValueError("MHR asset fingerprint must be a Unicode scalar")
    token = str(token.item())
    if len(token) != 64 or any(c not in "0123456789abcdef" for c in token):
        raise ValueError("invalid MHR asset fingerprint")
    if fingerprint is not None and token != fingerprint:
        raise ValueError("MHR reference asset fingerprint differs from the native rig")
    return {"model_params": params.astype(np.float32), "identity": identity.astype(np.float32),
            "times": times, "root_world": root, "mhr_fingerprint": token}


def load_reference(path, fingerprint=None):
    path = Path(path)
    if path.stat().st_size > 4 << 20:
        raise ValueError("MHR reference exceeds 4 MiB")
    with zipfile.ZipFile(path) as archive:
        if sum(item.file_size for item in archive.infolist()) > 4 << 20 or len(archive.infolist()) > 16:
            raise ValueError("MHR reference archive expands beyond its bound")
    with np.load(path, allow_pickle=False) as data:
        return validate_source(data, fingerprint)


def frozen_controls(assets):
    transform = mhr._tensor(Path(assets) / (mhr.STEM + ".safetensors"), "parameter_transform")
    return np.any(abs(transform[:, :204].reshape(127, 7, 204)[:, [0, 1, 2, 6]]) > 1e-12, axis=(0, 1))


def decode_source(data, assets):
    from scipy.spatial.transform import Rotation
    params, shape, times = data["model_params"], data["identity"], data["times"]
    frozen = frozen_controls(assets)
    if np.max(np.ptp(params[:, frozen], axis=0), initial=0) > 1e-4:
        raise ValueError("MHR segment translations and scales must be fixed for a take")
    _, states, rig, _ = mhr.decode(assets, params, shape, skeleton_only=True, backend="cpu")
    sites = [rig["names"].index(n) for n in LANDMARKS.values()]
    root = rig["names"].index("mhr_root")
    positions = (states[:, sites, :3] - states[:, root:root+1, :3]) * .01 + data["root_world"][:, None]
    matrices = Rotation.from_quat(states[:, sites, 3:7].reshape(-1, 4)).as_matrix().reshape(-1, 17, 3, 3)
    neutral = params[0].copy(); neutral[~frozen] = 0
    vertices, rest, _, _ = mhr.decode(assets, neutral[None], shape, backend="cpu")
    rest_p, rest_m = rest[0, sites, :3] * .01, Rotation.from_quat(rest[0, sites, 3:7]).as_matrix()
    # A full native rest mesh gives the sole datum rather than a skeletal proxy.
    floor = float(vertices[0, :, 1].min() * .01)
    offsets = {JOINT_NAMES[j]: float(rest_p[j, 1] - floor) for j in FEET}
    pitch = {}
    for side, ankle, toe in (("L", 11, 12), ("R", 15, 16)):
        chord = rest_p[toe] - rest_p[ankle]
        pitch[side] = float(np.degrees(np.arctan2(chord[1], np.linalg.norm(chord[[0, 2]]))))
    output_times = np.arange(int(times[-1] * 30 + 1e-6) + 1) / 30
    p, q = resample(times, positions, Rotation.from_matrix(matrices.reshape(-1, 3, 3)).as_quat().reshape(-1,17,4), output_times)
    return {"positions": p, "matrices": Rotation.from_quat(q.reshape(-1,4)).as_matrix().reshape(-1,17,3,3),
            "rest_positions": rest_p, "rest_matrices": rest_m, "sole_offsets": offsets,
            "neutral_foot_pitch": pitch, "times": output_times,
            "params": np.stack([np.interp(output_times, times, c) for c in params.T], axis=-1)}


def estimate_world_root(relative, camera, times, sole_offsets):
    """One global datum plus stance-velocity constraints; retain support elevation."""
    from scipy.sparse import lil_matrix
    from scipy.sparse.linalg import lsqr
    p, camera, times = np.asarray(relative), np.asarray(camera), np.asarray(times)
    if (p.shape != (len(times), 17, 3) or camera.shape != (len(times), 3)
            or len(times) < 5 or np.any(np.diff(times) <= 0)
            or not np.isfinite(p).all() or not np.isfinite(camera).all()):
        raise ValueError("invalid world-root evidence")
    prior = camera * [1, -1, -1]
    prior[:, [0, 2]] -= prior[0, [0, 2]]
    feet = p[:, FEET]
    world = feet + prior[:, None]
    velocity = np.gradient(world, times, axis=0)
    contact = np.linalg.norm(velocity, axis=-1) < .30
    # Require complete stance intervals, not isolated slow swing samples.
    for j in range(4):
        for a, b in np.flatnonzero(np.diff(np.r_[0, contact[:, j].astype(int), 0])).reshape(-1, 2):
            if b - a < 5:
                contact[a:b, j] = False
    if np.count_nonzero(contact) < 10:
        raise ValueError("insufficient stationary foot-contact evidence for world motion")
    active = contact[1:] & contact[:-1]
    rows = len(times)*3 + int(active.sum())*3 + max(0,len(times)-2)*3 + 2
    matrix = lil_matrix((rows, len(times)*3)); target = np.zeros(rows); row = 0
    for i in range(len(times)):
        for axis in range(3):
            matrix[row, i*3+axis] = .2; target[row] = .2*prior[i,axis]; row += 1
    for i, j in zip(*np.nonzero(active)):
        for axis in range(3):
            matrix[row, i*3+axis], matrix[row, (i+1)*3+axis] = -5., 5.
            target[row] = -5*(feet[i+1,j,axis]-feet[i,j,axis]); row += 1
    for i in range(len(times)-2):
        dt0, dt1 = times[i+1]-times[i], times[i+2]-times[i+1]
        for axis in range(3):
            matrix[row, [i*3+axis,(i+1)*3+axis,(i+2)*3+axis]] = [.1/dt0,-.1/dt0-.1/dt1,.1/dt1]
            row += 1
    for axis in (0, 2):
        matrix[row, axis] = 10; row += 1
    root = lsqr(matrix.tocsr(), target, atol=1e-8, btol=1e-8, iter_lim=4000)[0].reshape(-1,3)
    disagreement = float(np.sqrt(np.mean((root-prior)**2)))
    if disagreement > .3:
        raise ValueError("camera and stationary foot-contact travel estimates disagree")
    source_floor = float(np.min((feet[...,1] + root[:,None,1] - np.array(sole_offsets))[contact]))
    root[:,1] -= source_floor
    return root, {"root_motion": "estimated", "camera_contact_rms_m": disagreement,
                  "ground_datum_shift_m": -source_floor, "stance_samples": int(contact.sum())}


def prepare_video(sidecars, times, assets):
    import json
    from scipy.spatial.transform import Rotation
    meta = [json.loads(Path(p).read_text()) for p in sidecars]
    params = np.array([m["model_params"] for m in meta], dtype=np.float32)
    identity = np.median([m["shape"] for m in meta], axis=0).astype(np.float32)
    frozen = frozen_controls(assets); params[:, frozen] = np.median(params[:, frozen], axis=0)
    zero = {"model_params": params, "identity": identity, "times": np.asarray(times),
            "root_world": np.zeros((len(times),3)), "mhr_fingerprint": asset_fingerprint(assets)}
    source = decode_source(validate_source(zero), assets)
    # Work at original timestamps for camera evidence, then use the existing
    # endpoint-bounded resampling for policy/animation tracks.
    _, states, rig, _ = mhr.decode(assets, params, identity, skeleton_only=True, backend="cpu")
    sites = [rig["names"].index(n) for n in LANDMARKS.values()]
    root_index = rig["names"].index("mhr_root")
    relative = (states[:, sites, :3] - states[:, root_index:root_index+1, :3]) * .01
    up = np.median(relative[:,2], axis=0); up /= np.linalg.norm(up)
    if up[1] < .8:
        raise ValueError("Terra video requires an upright subject and observable gravity")
    align, _ = Rotation.align_vectors([[0,1,0]], [up])
    relative = align.apply(relative.reshape(-1,3)).reshape(relative.shape)
    camera = align.apply(np.array([m["cam_t"] for m in meta]) * [1,-1,-1]) * [1,-1,-1]
    roots, report = estimate_world_root(relative, camera, times, [source["sole_offsets"][JOINT_NAMES[j]] for j in FEET])
    source["positions"] = source["positions"] @ align.as_matrix().T
    source["positions"] += np.stack([np.interp(source["times"],times,c) for c in roots.T],axis=-1)[:,None]
    source["matrices"] = align.as_matrix() @ source["matrices"]
    source["rest_positions"] = source["rest_positions"] @ align.as_matrix().T
    source["rest_matrices"] = align.as_matrix() @ source["rest_matrices"]
    return source, report
