"""Isolated, pinned MuscleMimic adapter. Never imported by the HTTP runtime."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np

from .musclemimic_assets import read_installation, reference


def configure_process(work, device):
    work = Path(work).resolve()
    cache = work / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    for name in ("TMPDIR", "MPLCONFIGDIR", "MUSCLEMIMIC_HOME", "HF_HOME", "XDG_CACHE_HOME"):
        folder = cache / name.lower()
        folder.mkdir(exist_ok=True)
        os.environ[name] = str(folder)
    os.environ.update(JAX_PLATFORMS="cpu" if device == "cpu" else "cuda", HF_HUB_OFFLINE="1",
                      HF_DATASETS_OFFLINE="1", WANDB_MODE="disabled", XLA_PYTHON_CLIENT_PREALLOCATE="false")


def names(model, kind, count):
    import mujoco
    return [mujoco.mj_id2name(model, kind, i) for i in range(count)]


def reference_qpos(path, model):
    """Read only numerical/Unicode NPZ fields; never unpickle upstream metadata."""
    import mujoco
    with np.load(path, allow_pickle=False) as data:
        q = np.asarray(data["qpos"], dtype=np.float64)
        frequency = float(data["frequency"])
        source_names = data["joint_names"].tolist()
        types = np.asarray(data["jnt_type"], dtype=int)
        splits = np.asarray(data["split_points"], dtype=int) if "split_points" in data else np.array([0, len(q)])
    if (q.ndim != 2 or not np.isfinite(q).all() or not 1 <= frequency <= 1000
            or splits.tolist() != [0, len(q)] or len(source_names) != len(types)
            or len(set(source_names)) != len(source_names) or np.any((types < 0) | (types > 3))):
        raise ValueError("reference must be a finite, single-episode named MuJoCo trajectory")
    widths = [7 if t == 0 else 4 if t == 1 else 1 for t in types]
    if sum(widths) != q.shape[1]:
        raise ValueError("reference joint widths do not match qpos")
    offsets = np.r_[0, np.cumsum(widths)]
    slices = {n: (slice(offsets[i], offsets[i+1]), types[i]) for i, n in enumerate(source_names)}
    target_names = names(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt)
    count = min(len(q), int(8 * frequency) + 1)
    result = np.tile(model.qpos0, (count, 1))
    for j, name in enumerate(target_names):
        if name not in slices or slices[name][1] != model.jnt_type[j]:
            raise ValueError(f"reference joint/type missing: {name}")
        source, kind = slices[name]
        start = model.jnt_qposadr[j]
        width = 7 if kind == 0 else 4 if kind == 1 else 1
        result[:, start:start+width] = q[:count, source]
        if kind in (0, 1):
            quat = result[:, start+3:start+7] if kind == 0 else result[:, start:start+4]
            if np.any(abs(np.linalg.norm(quat, axis=-1) - 1) > 1e-3):
                raise ValueError("reference quaternion is not normalized")
    if count < 3:
        raise ValueError("reference trajectory is too short")
    return result, frequency


def trajectory(model, qpos, frequency):
    import mujoco
    from loco_mujoco.trajectory.dataclasses import Trajectory, TrajectoryInfo, TrajectoryModel, TrajectoryData
    qvel = np.empty((len(qpos), model.nv))
    for i in range(len(qpos)):
        a, b = max(0, i-1), min(len(qpos)-1, i+1)
        mujoco.mj_differentiatePos(model, qvel[i], (b-a)/frequency, qpos[a], qpos[b])
    data = mujoco.MjData(model)
    fields = ("xpos", "xquat", "cvel", "subtree_com", "site_xpos", "site_xmat")
    arrays = {f: [] for f in fields}
    for q, v in zip(qpos, qvel):
        data.qpos[:] = q; data.qvel[:] = v
        mujoco.mj_forward(model, data)
        for f in fields:
            arrays[f].append(np.array(getattr(data, f), copy=True))
    model_fields = ("njnt", "jnt_type", "nbody", "body_rootid", "body_weldid", "body_mocapid", "body_pos",
                    "body_quat", "body_ipos", "body_iquat", "nsite", "site_bodyid", "site_pos", "site_quat")
    tm = TrajectoryModel(**{f: getattr(model, f) for f in model_fields})
    info = TrajectoryInfo(names(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt), tm, frequency,
                          names(model, mujoco.mjtObj.mjOBJ_BODY, model.nbody),
                          names(model, mujoco.mjtObj.mjOBJ_SITE, model.nsite))
    return Trajectory(info, TrajectoryData(qpos=qpos, qvel=qvel, split_points=np.array([0, len(qpos)]),
                                          **{f: np.array(v) for f, v in arrays.items()}))


def calibrated_gmr(env, source, work):
    """Feed MHR semantic transforms directly to GMR, calibrated to Myo bodies."""
    import mujoco
    from scipy.spatial.transform import Rotation
    from general_motion_retargeting import GeneralMotionRetargeting
    from .musclemimic_kinematics import LANDMARKS, change_basis, proportions
    model = env.model
    rest = mujoco.MjData(model); mujoco.mj_forward(model, rest)
    site_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n) for n in LANDMARKS]
    if min(site_ids) < 0:
        raise ValueError("Myo model lacks calibrated mimic sites")
    bodies = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.site_bodyid[i]) for i in site_ids]
    # GMR uses its original full-finger model, then joint-name mapping removes
    # disabled fingers to match the checkpoint's exact observation/action layout.
    p, r = change_basis(source["positions"], source["matrices"])
    sr, srm = change_basis(source["rest_positions"], source["rest_matrices"])
    p = proportions(p, sr, rest.site_xpos[site_ids])
    low = np.min(p[:, [11, 12, 15, 16], 2])
    p[..., 2] -= low - np.min(rest.site_xpos[site_ids, 2])
    table = {}
    for j, body in enumerate(bodies):
        body_id = model.site_bodyid[site_ids[j]]
        body_r = rest.xmat[body_id].reshape(3, 3)
        offset_r = srm[j].T @ body_r
        offset_p = body_r.T @ (rest.xpos[body_id] - rest.site_xpos[site_ids[j]])
        table[body] = [list(LANDMARKS)[j], 100 if j in (0, 11, 12, 15, 16) else 40, 10,
                       offset_p.tolist(), Rotation.from_matrix(offset_r).as_quat(scalar_first=True).tolist()]
    cfg = {"robot_root_name": "pelvis", "human_root_name": "pelvis_mimic", "ground_height": 0,
           "human_height_assumption": 1.8, "human_scale_table": {n: 1 for n in LANDMARKS},
           "use_ik_match_table1": True, "use_ik_match_table2": True,
           "ik_match_table1": table, "ik_match_table2": table}
    cfg_path = work / "mhr_to_myo.json"
    cfg_path.write_text(json.dumps(cfg, indent=2))
    gmr = GeneralMotionRetargeting(src_human="mhr", tgt_robot="myofullbody", ik_config_path=str(cfg_path),
                                  use_fitted_shape=False, solver="daqp", damping=.5, verbose=False)
    raw, errors = [], []
    for pos, mat in zip(p, r):
        human = {n: (pos[j], Rotation.from_matrix(mat[j]).as_quat(scalar_first=True))
                 for j, n in enumerate(LANDMARKS)}
        q, error = gmr.retarget(human)
        raw.append(q); errors.append(float(np.max(error)))
    np.savez(work / "gmr-target.npz", qpos=raw, frequency=30.,
             joint_names=np.array(names(gmr.model, mujoco.mjtObj.mjOBJ_JOINT, gmr.model.njnt)),
             jnt_type=gmr.model.jnt_type, split_points=[0, len(raw)])
    if max(errors) > .15:
        raise ValueError(f"MHR to Myo retargeting failed: max landmark error {max(errors):.3f} m")
    q, fps = reference_qpos(work / "gmr-target.npz", model)
    return q, fps, {"gmr_max_site_error_m": max(errors)}


def prepare_video(request, work):
    from . import mhr
    from .musclemimic_kinematics import LANDMARKS, resample
    from .musclemimic_video import estimate_root
    from scipy.spatial.transform import Rotation
    meta = [json.loads(Path(p).read_text()) for p in request["sidecars"]]
    params = np.array([m["model_params"] for m in meta], dtype=np.float32)
    shape = np.median([m["shape"] for m in meta], axis=0).astype(np.float32)
    assets = Path(request["mhr_assets"])
    # Lock parameter columns affecting segment translation or scale across time.
    w = mhr._tensor(assets / (mhr.STEM + ".safetensors"), "parameter_transform")[:, :204].reshape(127, 7, 204)
    frozen = np.any(abs(w[:, [0, 1, 2, 6]]) > 1e-12, axis=(0, 1))
    # Keep global translation out of this track; root is recovered below.
    params[:, frozen] = np.median(params[:, frozen], axis=0)
    _, states, rig, _ = mhr.decode(assets, params, shape, skeleton_only=True, backend="cpu")
    sites = [rig["names"].index(n) for n in LANDMARKS.values()]
    positions = states[:, sites, :3].astype(np.float64) * .01
    matrices = Rotation.from_quat(states[:, sites, 3:7].reshape(-1, 4)).as_matrix().reshape(len(states), 17, 3, 3)
    times = np.asarray(request["times"])
    p, align, contact, report = estimate_root(positions, [m["cam_t"] for m in meta], times)
    matrices = align @ matrices
    # A neutral angular pose calibrates axes independently from the first video frame.
    neutral = params[0].copy(); neutral[~frozen] = 0
    _, state0, _, _ = mhr.decode(assets, neutral[None], shape, skeleton_only=True, backend="cpu")
    rest_p = (state0[0, sites, :3] * .01) @ align.T
    rest_r = align @ Rotation.from_quat(state0[0, sites, 3:7]).as_matrix()
    out_times = np.arange(int(times[-1] * 30) + 1) / 30
    pos, quat = resample(times, p, Rotation.from_matrix(matrices.reshape(-1, 3, 3)).as_quat().reshape(len(times), 17, 4), out_times)
    interpolated = np.stack([np.interp(out_times, times, col) for col in params.T], axis=-1)
    result = {"positions": pos, "matrices": Rotation.from_quat(quat.reshape(-1, 4)).as_matrix().reshape(-1, 17, 3, 3),
              "rest_positions": rest_p, "rest_matrices": rest_r, "times": out_times, "params": interpolated}
    np.savez(work / "source-mhr.npz", **result)
    return result, report


def record_episode(env, policy, max_steps, *, snapshot):
    """One reset only. Each snapshot contains state and activations at the same time."""
    obs = env.reset()
    frames = [snapshot()]
    reason, last_info = "step_limit", {}
    for _ in range(max_steps):
        action = policy(obs)
        obs, reward, absorbing, done, info = env.step(np.atleast_2d(action))
        frames.append(snapshot())
        last_info = info
        if done or absorbing:
            at_end = int(info.get("subtraj_step_no", -1)) >= int(info.get("traj_len", 1)) - 1
            reason = "reference_end" if at_end and not absorbing and not info.get("terminated", False) else "tracking_failure"
            break
    return frames, reason, last_info


def run(request, installation, work):
    import mujoco
    import jax
    import jax.numpy as jnp
    from omegaconf import OmegaConf
    from loco_mujoco.task_factories import TaskFactory
    from musclemimic.algorithms import PPOJax
    from musclemimic.algorithms.ppo.inference import ObservationHistoryBuffer
    from musclemimic.runner.eval_utils import load_checkpoint, apply_temporal_params, align_agent_state
    from .musclemimic_kinematics import LANDMARKS, Y_TO_Z, change_basis
    checkpoint = Path(installation["checkpoint"]).resolve()
    step = int(json.loads((checkpoint / "metadata/metadata").read_text()).get("update_number", 0))
    checkpoint_parent = work / "checkpoints"
    checkpoint_parent.mkdir(exist_ok=True)
    link = checkpoint_parent / f"checkpoint_{step}"
    if not link.exists():
        link.symlink_to(checkpoint, target_is_directory=True)
    config, agent_state, _ = load_checkpoint(str(link.absolute()))
    OmegaConf.set_struct(config, False)
    dt = apply_temporal_params(config)
    params = OmegaConf.to_container(config.experiment.env_params, resolve=True)
    if params["env_name"] not in ("MyoFullBody", "MjxMyoFullBody"):
        raise ValueError("checkpoint is not MyoFullBody")
    params.update(env_name="MyoFullBody", headless=True, num_envs=1,
                  th_params={"random_start": False, "fixed_start_conf": [0, 0], "start_from_random_step": False})
    if "goal_params" in params:
        params["goal_params"]["visualize_goal"] = False
    env = TaskFactory.get_factory_cls("ImitationFactory").make(**params)
    if request.get("reference_motion"):
        item = reference(installation, request["reference_motion"])
        qpos, fps = reference_qpos(item["path"], env.model)
        report = {"source": item}
    else:
        source, report = prepare_video(request, work)
        qpos, fps, fit_report = calibrated_gmr(env, source, work)
        report.update(fit_report)
    traj = trajectory(env.model, qpos, fps)
    env.load_trajectory(traj=traj, cache_type="full")
    agent_conf = PPOJax.init_agent_conf(env, config)
    state = align_agent_state(agent_state, agent_conf).train_state
    if config.experiment.n_seeds > 1:
        state = jax.tree.map(lambda x: x[0], state)
    history = int(config.experiment.get("len_obs_history") or 1)
    split = bool(config.experiment.get("split_goal", False))
    goal = env.obs_container.get_obs_ind_by_group("goal") if split else None
    state_ids = np.setdiff1d(np.arange(env.info.observation_space.shape[0]), goal) if split else None
    buffer = ObservationHistoryBuffer(history, split, state_ids, goal) if history > 1 else None
    started = False
    @jax.jit
    def act(state, obs):
        (dist, _), updates = agent_conf.network.apply(
            {"params": state.params, "run_stats": state.run_stats}, jnp.atleast_2d(obs), mutable=["run_stats"])
        action = dist.mean()
        # Upstream RunningMeanStd squeezes a one-item observation batch.
        # Indexing [0] unconditionally would broadcast one muscle to all 354.
        if action.ndim > 1:
            action = action[0]
        return action, state.replace(run_stats=updates["run_stats"])
    def policy(obs):
        nonlocal state, started
        if buffer is not None:
            obs = buffer.step(obs) if started else buffer.reset(obs)
        started = True
        action, state = act(state, obs)
        action = np.asarray(action)
        if action.shape != env.info.action_space.shape or not np.isfinite(action).all():
            raise ValueError("policy action shape/values do not match the checkpoint model")
        return action
    model = env.model
    site_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n) for n in LANDMARKS]
    rest = mujoco.MjData(model); mujoco.mj_forward(model, rest)
    rest_pos, rest_mat = change_basis(rest.site_xpos[site_ids], rest.site_xmat[site_ids].reshape(17, 3, 3), Y_TO_Z.T)
    def sample(d):
        if not all(np.isfinite(a).all() for a in (d.qpos, d.qvel, d.ctrl, d.act)):
            raise ValueError("simulator produced a nonfinite state")
        p, r = change_basis(d.site_xpos[site_ids], d.site_xmat[site_ids].reshape(17, 3, 3), Y_TO_Z.T)
        # Touch sensors are ordered explicitly, never inferred from actuator IDs.
        contacts = []
        for sensor in ("r_foot", "r_toes", "l_foot", "l_toes"):
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor)
            if sid < 0:
                sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "touch_" + sensor)
            contacts.append(float(d.sensordata[model.sensor_adr[sid]]) > 1e-4 if sid >= 0 else False)
        return {"positions": p, "matrices": r, "qpos": d.qpos.copy(), "qvel": d.qvel.copy(),
                "commands": d.ctrl.copy(), "activations": d.act.copy(),
                "contacts": np.array(contacts)[[2, 3, 0, 1]]}
    duration = (len(qpos)-1) / fps
    count = min(int(duration/dt + 1e-6), 8000)
    reference_frames = []
    ref_data = mujoco.MjData(model)
    # Use the resampled reference owned by the environment for exact timing.
    count = min(count, env.th.len_trajectory(0)-1)
    for i in range(count+1):
        item = env.th.traj.data.get(0, i, np)
        ref_data.qpos[:] = item.qpos; ref_data.qvel[:] = item.qvel
        mujoco.mj_forward(model, ref_data)
        reference_frames.append(sample(ref_data))
    def snapshot():
        # mj_step integrates qpos after computing site positions and sensors.
        # Recompute so all recorded fields describe the same instant.
        # Reset/domain hooks may replace MjData; read the current instance.
        current = env.data
        mujoco.mj_forward(model, current)
        return sample(current)
    frames, reason, info = record_episode(env, policy, count, snapshot=snapshot)
    if reason == "step_limit" and len(frames) == count+1:
        reason = "reference_end"  # bounded selected interval completed without a reset
    def save(name, sequence):
        arrays = {key: np.stack([f[key] for f in sequence]) for key in sequence[0]}
        np.savez(work / name, times=np.arange(len(sequence))*dt, rest_positions=rest_pos,
                 rest_matrices=rest_mat, **arrays)
    save("reference.npz", reference_frames)
    save("rollout.npz", frames)
    pos = np.array([f["positions"] for f in frames])
    ref = np.array([f["positions"] for f in reference_frames[:len(frames)]])
    report.update(termination=reason, completed=reason == "reference_end", frames=len(frames),
                  control_dt=dt, physics="mujoco-cpu", policy_device=str(jax.devices()[0]),
                  relative_site_rms_m=float(np.sqrt(np.mean(((pos-pos[:, :1])-(ref-ref[:, :1]))**2))),
                  root_rms_m=float(np.sqrt(np.mean((pos[:, 0]-ref[:, 0])**2))),
                  model={"nq": model.nq, "nv": model.nv, "actuators": model.nu},
                  checkpoint_configuration=OmegaConf.to_container(config, resolve=True))
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--request", type=Path, required=True)
    ap.add_argument("--assets", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--policy-device", choices=("cpu", "cuda"), default="cpu")
    args = ap.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    configure_process(args.work, args.policy_device)
    start = time.monotonic()
    report = {}
    try:
        installation = read_installation(args.assets, verify=True)
        # Ensure exactly the registered checkouts take precedence over installed modules.
        sys.path[:0] = [installation["upstream"], installation["gmr"]]
        request = json.loads(args.request.read_text())
        report = run(request, installation, args.work)
        report["provenance"] = {k: installation[k] for k in ("upstream_revision", "gmr_revision", "checkpoint_revision", "packages", "usage")}
        (args.work / "simulation.json").write_text(json.dumps(report, indent=2, allow_nan=False))
        report["avatar"] = export_tracks(request, args.work, report)
    except Exception as exc:
        report.update(completed=False, termination="error", error=f"{type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
    report.update(wall_seconds=time.monotonic()-start, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    (args.work / "simulation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return 0 if report.get("completed") else 2


def export_tracks(request, work, simulation):
    from scipy.spatial.transform import Rotation
    from .musclemimic_kinematics import MHRKinematics, AvatarFeet, resample, FEET
    from .motion_format import document
    from .motion import _export_glb
    avatar = Path(request["avatar"])
    definition = json.loads(Path(request["avatar_json"]).read_text())
    bind = json.loads(Path(request["body_meta"]).read_text())
    fk = MHRKinematics(definition, Path(request["mhr_assets"]), bind["model_params"])
    skin = AvatarFeet(avatar, fk)
    reports = {}
    for name, artifact in (("reference.npz", "baseline"), ("rollout.npz", "motion")):
        with np.load(work / name, allow_pickle=False) as track:
            times = track["times"]
            if len(times) < 2:
                raise ValueError("simulation did not produce an exportable interval")
            output_times = np.arange(min(240, int(times[-1]*30 + 1e-6)+1))/30
            if len(output_times) < 2:
                raise ValueError("simulation terminated before two output frames")
            q = Rotation.from_matrix(track["matrices"].reshape(-1, 3, 3)).as_quat().reshape(len(times), 17, 4)
            p, q = resample(times, track["positions"], q, output_times)
            matrices = Rotation.from_quat(q.reshape(-1, 4)).as_matrix().reshape(-1, 17, 3, 3)
            contacts = track["contacts"][np.minimum(np.searchsorted(times, output_times), len(times)-1)]
            if artifact == "baseline":
                # Reference kinematics have no muscle forces. Infer stance from
                # height and world-space speed instead of using mj_forward forces.
                speed = np.linalg.norm(np.gradient(p[:, FEET], output_times, axis=0), axis=-1)
                height = p[:, FEET, 1] - np.min(p[:, FEET, 1], axis=0)
                contacts = (height < .04) & (speed < .15)
            source_params = None
            if (work / "source-mhr.npz").exists():
                with np.load(work / "source-mhr.npz", allow_pickle=False) as src:
                    source_params = np.stack([np.interp(output_times, src["times"], col) for col in src["params"].T], axis=-1)
            rotations, roots, fitted, report = fk.retarget(p, matrices, contacts, output_times,
                source_rest=track["rest_positions"], source_rest_matrices=track["rest_matrices"], source_params=source_params,
                skin=skin)
        reports[artifact] = report
        # Preserve the candidate separately until the overall gate is known.
        output = document(fk.names, rotations, output_times, roots, provenance={
            **simulation.get("provenance", {}), "source": artifact, "physical_simulation": artifact == "motion",
            "research_only": True, "root_motion": "estimated" if request.get("sidecars") else
            ("simulated" if artifact == "motion" else "reference")})
        target = artifact if artifact == "baseline" else "candidate"
        (work / f"{target}.json").write_text(json.dumps(output, separators=(",", ":"), allow_nan=False))
        _export_glb(avatar, work / f"{target}.glb", fk.names, rotations, output_times, root_translation=roots)
        np.savez(work / f"{target}-mhr.npz", params=fitted, root_translation=roots, times=output_times)
    reports["accepted"] = bool(simulation["completed"] and reports["motion"]["accepted"])
    reports["comparison"] = {key + "_change": reports["motion"][key] - reports["baseline"][key]
        for key in ("site_fit_rms_m", "skin_stance_drift_p95_m", "skin_ground_penetration_max_m")
        if reports["motion"].get(key) is not None and reports["baseline"].get(key) is not None}
    reports["perceptual_review"] = "pending"
    if reports["accepted"]:
        for suffix in ("json", "glb"):
            (work / f"candidate.{suffix}").replace(work / f"motion.{suffix}")
    return reports


if __name__ == "__main__":
    raise SystemExit(main())
