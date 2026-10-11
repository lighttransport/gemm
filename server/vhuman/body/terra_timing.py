"""Resample point-timestamped MHR solves onto the exact policy clock."""
import numpy as np


def retime_qpos(model, qpos, duration, frequency=100.):
    import mujoco
    from scipy.spatial.transform import Rotation, Slerp
    qpos = np.asarray(qpos, dtype=float)
    if (qpos.ndim != 2 or qpos.shape[1] != model.nq or len(qpos) < 2
            or not np.isfinite(qpos).all() or not np.isfinite(duration) or duration <= 0
            or not np.isfinite(frequency) or frequency <= 0):
        raise ValueError("invalid control-clock trajectory")
    source_times = np.linspace(0, duration, len(qpos))
    times = np.arange(int(duration * frequency + 1e-6) + 1) / frequency
    if len(times) < 2:
        raise ValueError("source interval is shorter than one policy step")
    output = np.stack([np.interp(times, source_times, c) for c in qpos.T], axis=-1)
    for kind, address in zip(model.jnt_type, model.jnt_qposadr):
        if kind not in (mujoco.mjtJoint.mjJNT_FREE, mujoco.mjtJoint.mjJNT_BALL):
            continue
        start = int(address) + (3 if kind == mujoco.mjtJoint.mjJNT_FREE else 0)
        rotations = Rotation.from_quat(qpos[:, start:start+4], scalar_first=True)
        output[:, start:start+4] = Slerp(source_times, rotations)(times).as_quat(scalar_first=True)
    velocity = np.zeros((len(times), model.nv))
    for i in range(1, len(times)):
        mujoco.mj_differentiatePos(model, velocity[i], 1/frequency, output[i-1], output[i])
    velocity[0] = velocity[1]
    return output.astype(np.float32), velocity.astype(np.float32), times


def control_trajectory(result, model, robot_conf):
    """Discard stale FK caches and regenerate them from the retimed qpos/qvel."""
    import dataclasses
    from terra._musclemimic import Trajectory, TrajectoryData, extend_motion
    duration = (int(result.analysis["native_frame_count"]) - 1) / float(result.analysis["native_fps"])
    original = np.asarray(result.trajectory.data.qpos)
    qpos, qvel, times = retime_qpos(model, original, duration)
    info = dataclasses.replace(result.trajectory.info, frequency=100., body_names=None, site_names=None)
    trajectory = Trajectory(info, TrajectoryData(qpos=qpos, qvel=qvel, split_points=np.array([0, len(times)])))
    # No retarget/model fitting occurs in extend_motion: it computes body/site FK.
    trajectory = extend_motion("MyoFullBody", dict(robot_conf.env_params), trajectory)
    result.analysis["pos_error"] = np.stack([np.interp(times, np.linspace(0, duration, len(result.analysis["pos_error"])), c)
                                             for c in np.asarray(result.analysis["pos_error"]).T], axis=-1)
    result.analysis["control_clock"] = {"frequency":100.,"source_interval_s":duration,
                                         "duration_s":float(times[-1]),"frames":len(times)}
    result.trajectory = trajectory
    from terra.stability import retarget_stability
    result.analysis["stability_before_control_clock"] = result.analysis["stability_check"]
    result.analysis["stability_check"] = retarget_stability(result)
    if result.analysis["stability_check"]["requires_retry"]:
        raise ValueError("retimed MHR trajectory failed the final stability gate")
    return result
