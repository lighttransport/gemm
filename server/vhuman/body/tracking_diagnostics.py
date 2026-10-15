"""Synchronous tracking evidence; these measurements never change acceptance."""
import numpy as np


def measure(reference, rollout, *, core_sites, limits, quaternion_slice=slice(3, 7), include_sites=None):
    n = len(rollout['positions'])
    ref = np.asarray(reference['positions'])[:n]
    pos = np.asarray(rollout['positions'])
    if ref.shape != pos.shape or not n or not np.isfinite(pos).all() or not np.isfinite(ref).all():
        raise ValueError('tracking diagnostics require aligned finite reference/rollout positions')
    error = np.linalg.norm(pos - ref, axis=-1)
    included = np.arange(error.shape[1]) if include_sites is None else np.asarray(include_sites, dtype=int)
    global_error = error[:, included].mean(1)
    core_error = error[:, core_sites].mean(1)
    qa = np.asarray(reference['qpos'])[:n, quaternion_slice]
    qb = np.asarray(rollout['qpos'])[:, quaternion_slice]
    na, nb = np.linalg.norm(qa, axis=1), np.linalg.norm(qb, axis=1)
    if not np.isfinite(qa).all() or not np.isfinite(qb).all() or min(na.min(), nb.min()) < 1e-6:
        raise ValueError('invalid root quaternion in tracking diagnostics')
    angle = 2*np.arccos(np.clip(abs((qa*qb).sum(1)/(na*nb)), 0, 1))
    arrays = dict(global_mpjpe_m=global_error, core_mpjpe_m=core_error, root_angle_rad=angle)
    final = {key: float(value[-1]) for key, value in arrays.items()}
    violations = {key: bool(limit is not None and final[key] > limit) for key, limit in limits.items()}
    summary = dict(final=final, limits=limits, final_violations=violations,
        initial_qpos_max_abs_difference=float(np.max(abs(np.asarray(rollout['qpos'])[0]-np.asarray(reference['qpos'])[0]))),
        initial_qvel_max_abs_difference=float(np.max(abs(np.asarray(rollout['qvel'])[0]-np.asarray(reference['qvel'])[0]))),
        initial_root_position_max_abs_difference=float(np.max(abs(np.asarray(rollout['qpos'])[0,:3]-np.asarray(reference['qpos'])[0,:3]))),
        initial_global_mpjpe_m=float(global_error[0]),
        interpretation='recomputed from synchronous snapshots; diagnostic only')
    return arrays, summary
