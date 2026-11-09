"""SOMA-X decode runner. Imports torch/warp/soma, so run it with VHUMAN_SOMA_PYTHON only.

SAM 3D Body MHR parameters -> posed MHR mesh (SOMA's bundled MHR TorchScript) ->
SOMA pose inversion -> posed SOMA mesh + global joint state, all in metres.

  decode:    --params P[B,204].npy --shape S[B,45].npy --output-dir D [--skeleton-only]
  landmarks: --landmarks-from KEYPOINT_MAPPING[5,V+J].npy --output-dir D
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _matrix_to_quat_xyzw(m: np.ndarray) -> np.ndarray:
    """Rotation matrices [...,3,3] -> unit quaternions xyzw with w >= 0."""
    flat = m.reshape(-1, 3, 3)
    out = np.empty((len(flat), 4))
    for i, r in enumerate(flat):
        t = np.trace(r)
        if t > 0:
            s = np.sqrt(t + 1) * 2
            q = [(r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s, s / 4]
        elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
            s = np.sqrt(1 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
            q = [s / 4, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s, (r[2, 1] - r[1, 2]) / s]
        elif r[1, 1] > r[2, 2]:
            s = np.sqrt(1 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
            q = [(r[0, 1] + r[1, 0]) / s, s / 4, (r[1, 2] + r[2, 1]) / s, (r[0, 2] - r[2, 0]) / s]
        else:
            s = np.sqrt(1 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
            q = [(r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, s / 4, (r[1, 0] - r[0, 1]) / s]
        q = np.asarray(q)
        q /= np.linalg.norm(q)
        out[i] = -q if q[3] < 0 else q
    return out.reshape(m.shape[:-2] + (4,))


def _state(transforms: np.ndarray) -> np.ndarray:
    """[B,J,4,4] global transforms -> [B,J,8] (t, quat xyzw, uniform scale)."""
    lin = transforms[..., :3, :3]
    scale = np.cbrt(np.linalg.det(lin))
    if not (np.isfinite(scale).all() and (scale > 0).all()):
        raise ValueError("SOMA produced a degenerate joint transform")
    state = np.empty(transforms.shape[:-2] + (8,))
    state[..., :3] = transforms[..., :3, 3]
    state[..., 3:7] = _matrix_to_quat_xyzw(lin / scale[..., None, None])
    state[..., 7] = scale
    return state


def _load(device, data_root=None):
    import torch
    from soma import SOMALayer
    from soma.fitting.pose_inversion import PoseInversion
    extra = {"data_root": data_root} if data_root else {}
    layer = SOMALayer(identity_model_type="mhr", device=device, **extra)
    return torch, layer, PoseInversion(layer)


def _mhr(layer, torch, identity, pose, scale):
    zeros = torch.zeros(len(identity), 72, device=identity.device)
    return layer.identity_model.identity_model(identity, torch.cat([pose, scale], 1), zeros)


def _decode(args) -> dict:
    params = np.load(args.params, allow_pickle=False).astype(np.float32)
    shape = np.load(args.shape, allow_pickle=False).astype(np.float32)
    if params.ndim != 2 or params.shape[1] != 204 or shape.shape != (len(params), 45):
        raise ValueError("expected params [B,204] and shape [B,45]")
    torch, layer, inv = _load(args.device, args.data_root)
    dev = torch.device(args.device)
    p, s = torch.from_numpy(params).to(dev), torch.from_numpy(shape).to(dev)
    pose, scale = p[:, :136], p[:, 136:]
    same = bool((s == s[:1]).all() and (scale == scale[:1]).all())
    ident, scl = (s[:1], scale[:1]) if same else (s, scale)
    with torch.no_grad():
        mhr_vertices, _ = _mhr(layer, torch, s, pose, scale)  # cm, MHR topology
        inv.prepare_identity(ident, scl)
        layer.prepare_identity(ident, scl)
        fit = inv.fit(mhr_vertices * 0.01, autograd_iters=args.autograd_iters)
        out = layer.pose(fit["rotations"][:, 1:], transl=fit["root_translation"],
                         pose2rot=False, absolute_pose=True)
        target = layer.identity_model._to_soma_interp(mhr_vertices * 0.01)
        error = (out["vertices"] - target).norm(dim=-1)
    state = _state(out["transforms"].double().cpu().numpy())
    np.save(Path(args.output_dir) / "skeleton.npy", state)
    if not args.skeleton_only:
        np.save(Path(args.output_dir) / "vertices.npy", out["vertices"].cpu().numpy().astype(np.float32))
    names = list(layer.public_joint_names)
    parents = layer.public_joint_parent_ids.cpu().numpy().astype(np.int32)
    parents[0] = -1  # SOMA marks the virtual Root as its own parent
    weights = layer.public_skinning_weights().cpu().numpy()
    top = np.argsort(-weights, axis=1)[:, :4].astype(np.int32)
    w4 = np.take_along_axis(weights, top, 1).astype(np.float32)
    np.savez(Path(args.output_dir) / "rig.npz", parents=parents, joint_indices=top, skin_weights=w4,
             faces=layer.faces.cpu().numpy().astype(np.int32))
    return {"names": names, "unit": "m", "vertices": int(out["vertices"].shape[1]),
            "joints": len(names), "shared_identity": same,
            "mhr_to_soma_error_m": {"mean": float(error.mean()), "max": float(error.max()),
                                    "p99": float(torch.quantile(error.flatten(), .99))},
            "pose_inversion_error_m_max": float(fit["per_vertex_error"].max()),
            "autograd_iters": args.autograd_iters}


def _landmarks(args) -> dict:
    mapping = np.load(args.landmarks_from, allow_pickle=False).astype(np.float64)[:5]
    torch, layer, _ = _load(args.device, args.data_root)
    z = lambda n: torch.zeros(1, n, device=args.device)
    with torch.no_grad():
        verts, skel = _mhr(layer, torch, z(45), z(136), z(68))
        soma = layer.identity_model._to_soma_interp(verts * 0.01)[0].cpu().numpy().astype(np.float64)
    if mapping.shape[1] != verts.shape[1] + skel.shape[1]:
        raise ValueError("keypoint mapping does not match MHR vertices+joints")
    combined = np.vstack([verts[0].cpu().numpy() * 0.01, skel[0, :, :3].cpu().numpy() * 0.01])
    kp = mapping @ combined
    idx = [int(np.argmin(np.linalg.norm(soma - k, axis=1))) for k in kp]
    dist = [float(np.linalg.norm(soma[i] - k)) for i, k in zip(idx, kp)]
    return {"indices": idx, "order": ["nose", "left_eye", "right_eye", "left_ear", "right_ear"],
            "nearest_distance_m": dist, "soma_vertices": int(soma.shape[0])}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--params")
    ap.add_argument("--shape")
    ap.add_argument("--landmarks-from")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--skeleton-only", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--data-root", help="local SOMA-X asset directory (default: Hugging Face cache)")
    ap.add_argument("--autograd-iters", type=int, default=0)
    args = ap.parse_args(argv)
    report = _landmarks(args) if args.landmarks_from else _decode(args)
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
