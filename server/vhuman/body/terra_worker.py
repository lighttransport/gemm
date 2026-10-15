"""Isolated Terra/MHR solver, policy rollout and portable avatar export."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

import numpy as np

from .musclemimic_worker import configure_process, record_episode
from .terra_assets import read_installation, reference


def run(request, installation, work, *, prepared_source=None, source_provenance=None):
    import mujoco
    import jax
    import jax.numpy as jnp
    from omegaconf import OmegaConf
    from musclemimic.algorithms import PPOJax
    from musclemimic.algorithms.ppo.inference import ObservationHistoryBuffer
    from musclemimic.runner.eval_utils import load_checkpoint, apply_temporal_params, align_agent_state
    from terra._musclemimic import load_robot_conf_file
    from .terra_prepared import PreparedMotion, fit_prepared_motion
    from terra.rl import register_components
    from terra.rl.task_factory import TerraImitationFactory
    from terra.rl.trajectory import install_trajectory_stability
    from .musclemimic_kinematics import LANDMARKS, Y_TO_Z, change_basis
    from .terra_motion import JOINT_NAMES, asset_fingerprint, load_reference, decode_source, prepare_video
    from .terra_terrain import validate_spec
    import os
    np.random.seed(request.get("seed", 0))
    register_components(); install_trajectory_stability()
    if asset_fingerprint(request["mhr_assets"]) != installation["mhr_fingerprint"]:
        raise ValueError("job MHR assets differ from the registered Terra rig")
    config, agent_state, _ = load_checkpoint(installation["checkpoint"])
    OmegaConf.set_struct(config, False)
    dt = apply_temporal_params(config)
    if abs(dt - .01) > 1e-8:
        raise ValueError("Terra policy must use the supported 100 Hz control rate")
    infer_support = request.get('terrain','auto') == 'auto'
    if prepared_source is not None:
        if not source_provenance:
            raise ValueError('prepared source requires explicit provenance')
        source = prepared_source
        report = {"source":source_provenance,"root_motion":"reference"}
    elif request.get("reference_motion"):
        item = reference(installation, request["reference_motion"])
        source = decode_source(load_reference(item["path"], installation["mhr_fingerprint"]), Path(request["mhr_assets"]),
                               include_sole_vertices=infer_support)
        report = {"source": item, "root_motion": "reference"}
    else:
        source, report = prepare_video(request["sidecars"], request["times"], Path(request["mhr_assets"]),
                                       include_sole_vertices=infer_support)
        report["camera"] = request["camera"]
    np.savez(work / "source-mhr.npz", **{k:v for k,v in source.items() if isinstance(v, np.ndarray)})
    robot_conf = load_robot_conf_file("MyoFullBody")
    from terra.pipeline import _build_environment
    calibration_env = _build_environment("MyoFullBody", robot_conf, None, logging.getLogger("terra"))
    model0 = calibration_env._model
    rest0 = mujoco.MjData(model0); mujoco.mj_forward(model0, rest0)
    sites0 = [mujoco.mj_name2id(model0, mujoco.mjtObj.mjOBJ_SITE, n) for n in LANDMARKS]
    if min(sites0) < 0:
        raise ValueError("Terra robot lacks calibrated sites")
    robot_height = float(rest0.site_xpos[sites0[2],2] - min(rest0.site_xpos[sites0[k],2] for k in (11,12,15,16)))
    source_height = float(source["rest_positions"][2,1] - min(source["rest_positions"][k,1] for k in (11,12,15,16)))
    if min(robot_height, source_height) < .2:
        raise ValueError("degenerate source/robot stature calibration")
    scale = robot_height / source_height
    positions, rotations = change_basis(source["positions"] * scale, source["matrices"])
    translation = -positions[0, 0].copy(); translation[2] = 0.
    positions += translation
    rest_p, rest_r = change_basis(source["rest_positions"] * scale, source["rest_matrices"])
    offsets = {k: v * scale for k,v in source["sole_offsets"].items()}
    # Translate anatomical site anchors into robot-body targets in their rest
    # orientation, retaining measured native sole geometry and world elevation.
    for j, sid in enumerate(sites0):
        bid = model0.site_bodyid[sid]
        local = rest_r[j].T @ (rest0.xpos[bid] - rest0.site_xpos[sid])
        positions[:,j] += rotations[:,j] @ local
        rest_p[j] += rest_r[j] @ local
        if JOINT_NAMES[j] in offsets:
            offsets[JOINT_NAMES[j]] += float((rest_r[j] @ local)[2])
    pitch = {}
    for side, ankle, toe in (("L",11,12),("R",15,16)):
        chord = rest_p[toe] - rest_p[ankle]
        pitch[side] = float(np.degrees(np.arctan2(chord[2], np.linalg.norm(chord[:2]))))
    supplied = request.get("terrain", "auto")
    if infer_support:
        from .terra_support import flat_support_evidence
        evidence = flat_support_evidence(source)
        if evidence['qualified']:
            supplied = 'flat'
        report['terrain_selection'] = dict(requested='auto', selected=supplied, evidence=evidence)
    if isinstance(supplied, dict):
        supplied = validate_spec(supplied)
        for box in supplied["boxes"]:
            box["pos"] = (np.array(box["pos"]) * scale + translation).tolist()
            box["size"] = (np.array(box["size"]) * scale).tolist()
    source_fps = float(source.get('fps', 30.))
    prepared = PreparedMotion(JOINT_NAMES, positions, rotations, rest_p, rest_r, source_fps, offsets, pitch)
    # Registered references carry an explicit world-root track. Keep that
    # evidence in the constrained solve instead of fitting relative shape only.
    result = fit_prepared_motion(prepared, robot_conf, logging.getLogger("terra"), terrain=supplied,
                                 source_root_weight=1000. if prepared_source is not None or request.get("reference_motion") else 0.)
    from .terra_timing import control_trajectory
    result = control_trajectory(result, model0, robot_conf)
    from terra.contracts import RetargetResult
    from terra.artifacts import save_retarget_result, validate_retarget_artifacts
    artifact_result = RetargetResult(result.trajectory, result.analysis, result.terrain, "terra", work / "source-mhr.npz")
    save_retarget_result(artifact_result, work / "retargeted", "motion")
    validate_retarget_artifacts(work / "retargeted", "motion")
    from terra._musclemimic import TerrainSpec
    terrain = result.terrain or TerrainSpec()
    (work / "solver-terrain.json").write_text(json.dumps(terrain.to_dict(), indent=2, allow_nan=False))
    from .musclemimic_assets import sha256
    report.update(reconstruction=result.analysis["reconstruction"], stability=result.analysis["stability_check"],
                  source_root_objective=result.analysis['source_root_objective'],
                  source_to_solver_scale=scale, source_to_solver_translation_m=translation.tolist(), robot_height_m=robot_height,
                  control_clock=result.analysis["control_clock"],
                  source_time_offset_s=result.analysis["trim_start_frames"] / source_fps)
    report["provenance"] = {"backend": "terra", "revisions": installation["revisions"], "patches": installation["patches"],
        "source_root_objective":result.analysis['source_root_objective'],
        "adapter_sha256":{name:sha256(Path(__file__).with_name(name)) for name in ('terra_prepared.py','terra_root.py')},
        "checkpoint_revision": installation["checkpoint_revision"], "checkpoint_terms_url": installation["checkpoint_terms_url"],
        "checkpoint_license_status": installation["checkpoint_license_status"],
        "mhr_fingerprint": installation["mhr_fingerprint"], "source_to_solver_scale": scale,
        "source_to_solver_translation_m": translation.tolist(),
        "terrain_sha256": sha256(work / "solver-terrain.json"), "seed": request.get("seed",0),
        "policy_actions": request.get("policy_actions","sampled"), "research_only": True}
    (work / "retarget-report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    params = OmegaConf.to_container(config.experiment.env_params, resolve=True)
    if (params["env_name"] not in ("MyoFullBody", "MjxMyoFullBody") or config.experiment.algorithm != "PPOJax"
            or params.get("goal_type") not in ("TerraGoal", "TerraFullBodyTrackingGoal")):
        raise ValueError("checkpoint is not a Terra MyoFullBody PPO policy")
    if params["env_name"] == "MjxMyoFullBody" and params.get("model_option_conf") is None:
        params["model_option_conf"] = {"iterations":4, "ls_iterations":8,
                                       "disableflags":int(mujoco.mjtDisableBit.mjDSBL_EULERDAMP)}
    for name in list(params):
        if name.startswith("mjx_") or name in ("num_envs", "nconmax", "njmax", "terrain_type", "terrain_params"):
            del params[name]
    params.update(env_name="MyoFullBody", headless=True, th_params={"random_start":False,"fixed_start_conf":[0,0],
                                                                 "start_from_random_step":False})
    if "goal_params" in params:
        params["goal_params"]["visualize_goal"] = False
    # Keep the checkpoint's saved tracking thresholds; never relax a failed job.
    traj = result.trajectory
    env = TerraImitationFactory.make(**params, custom_dataset_conf={"traj": traj}, frame_zero_reset_probability=1.,
                                    terrain_type="BoxTerrain" if not terrain.is_flat else "StaticTerrain",
                                    terrain_params=terrain.to_env_params() if not terrain.is_flat else {})
    qpos, fps = np.asarray(traj.data.qpos), float(traj.info.frequency)
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
    rng = jax.random.PRNGKey(request.get("seed", 0))
    @jax.jit
    def act(state, obs, key):
        (dist, _), updates = agent_conf.network.apply(
            {"params": state.params, "run_stats": state.run_stats}, jnp.atleast_2d(obs), mutable=["run_stats"])
        action = dist.sample(seed=key) if request.get("policy_actions", "sampled") == "sampled" else dist.mean()
        # Upstream RunningMeanStd squeezes a one-item observation batch.
        # Indexing [0] unconditionally would broadcast one muscle to all 354.
        if action.ndim > 1:
            action = action[0]
        return action, state.replace(run_stats=updates["run_stats"])
    def policy(obs):
        nonlocal state, started, rng
        if buffer is not None:
            obs = buffer.step(obs) if started else buffer.reset(obs)
        started = True
        rng, key = jax.random.split(rng)
        action, state = act(state, obs, key)
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
    from .tracking_diagnostics import measure
    terminal = params.get('terminal_state_params', {})
    landmark_names = list(LANDMARKS)
    core = [landmark_names.index(n) for n in terminal.get('core_upper_body_sites',
            ('upper_body_mimic', 'head_mimic', 'left_shoulder_mimic', 'right_shoulder_mimic'))]
    included = [i for i, n in enumerate(landmark_names) if n not in (terminal.get('exclude_sites') or ())]
    with np.load(work/'reference.npz', allow_pickle=False) as reference_track, \
            np.load(work/'rollout.npz', allow_pickle=False) as rollout_track:
        errors, tracking = measure(reference_track, rollout_track, core_sites=core, include_sites=included,
            limits=dict(global_mpjpe_m=terminal.get('mean_site_deviation_threshold', .15),
                        core_mpjpe_m=terminal.get('core_upper_body_mean_site_deviation_threshold', .15),
                        root_angle_rad=terminal.get('root_orientation_threshold', 1.)))
    np.savez(work/'tracking-errors.npz', times=np.arange(len(frames))*dt, **errors)
    tracking['history_length'] = history
    tracking['history_reset'] = 'upstream zero history with initial observation in last slot' if history > 1 else 'one observation'
    tracking['final_info'] = {k: v.item() if hasattr(v, 'item') else v for k, v in info.items()
                              if np.asarray(v).shape == () and np.asarray(v).dtype.kind in 'bifu'}
    pos = np.array([f["positions"] for f in frames])
    ref = np.array([f["positions"] for f in reference_frames[:len(frames)]])
    report.update(seed=request.get("seed", 0), policy_actions=request.get("policy_actions", "sampled"), termination=reason, completed=reason == "reference_end", frames=len(frames),
                  control_dt=dt, physics="mujoco-cpu", policy_device=str(jax.devices()[0]),
                  cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES") if jax.devices()[0].platform == "gpu" else None,
                  relative_site_rms_m=float(np.sqrt(np.mean(((pos-pos[:, :1])-(ref-ref[:, :1]))**2))),
                  root_rms_m=float(np.sqrt(np.mean((pos[:, 0]-ref[:, 0])**2))),
                  tracking=tracking, model={"nq": model.nq, "nv": model.nv, "actuators": model.nu},
                  checkpoint_configuration=OmegaConf.to_container(config, resolve=True))
    return report


def export_tracks(request, work, simulation):
    from scipy.spatial.transform import Rotation
    from .musclemimic_kinematics import MHRKinematics, AvatarFeet, resample, FEET
    from .motion_format import document
    from .motion import _export_glb
    from .terra_terrain import Terrain, append_glb
    avatar = Path(request["avatar"])
    definition = json.loads(Path(request["avatar_json"]).read_text())
    bind = json.loads(Path(request["body_meta"]).read_text())
    fk = MHRKinematics(definition, Path(request["mhr_assets"]), bind["model_params"])
    skin = AvatarFeet(avatar, fk)
    rest, _, _ = fk.forward(fk.bind_params)
    target_height = float(rest[fk.sites[2],1] - min(rest[fk.sites[k],1] for k in FEET))
    scale = target_height / simulation["robot_height_m"]
    with np.load(work / "reference.npz", allow_pickle=False) as reference_track:
        first = reference_track["positions"][0,0] * scale
    offset = rest[fk.sites[0]] - first; offset[1] = skin.floor
    terrain = Terrain(json.loads((work / "solver-terrain.json").read_text()), scale=scale, offset=offset)
    (work / "terrain.json").write_text(json.dumps(terrain.document(), indent=2, allow_nan=False))
    reports = {}
    for name, artifact in (("reference.npz", "baseline"), ("rollout.npz", "motion")):
        with np.load(work / name, allow_pickle=False) as track:
            times = track["times"]
            if len(times) < 2:
                raise ValueError("simulation did not produce an exportable interval")
            output_times = np.arange(min(240, int(times[-1]*30+1e-6)+1))/30
            if len(output_times) < 2:
                raise ValueError("simulation terminated before two output frames")
            quaternions = Rotation.from_matrix(track["matrices"].reshape(-1,3,3)).as_quat().reshape(len(times),17,4)
            positions, quaternions = resample(times, track["positions"], quaternions, output_times)
            positions = positions * scale + offset
            matrices = Rotation.from_quat(quaternions.reshape(-1,4)).as_matrix().reshape(-1,17,3,3)
            contacts = track["contacts"][np.minimum(np.searchsorted(times,output_times),len(times)-1)]
            if artifact == "baseline":
                speed = np.linalg.norm(np.gradient(positions[:, FEET],output_times,axis=0),axis=-1)
                # Contact offsets are measured from this robot, then uniformly
                # scaled with its paired terrain; no global minimum-height test.
                rp = track["rest_positions"] * scale
                sole = rp[FEET,1] - np.min(rp[FEET,1])
                height = terrain.clearance(positions[:,FEET]) - sole
                contacts = (abs(height) < .06) & (speed < .30)
            with np.load(work / "source-mhr.npz", allow_pickle=False) as source:
                source_times = output_times + simulation["source_time_offset_s"]
                if source_times[-1] > source["times"][-1] + 1e-6:
                    raise ValueError("materialized motion extends beyond the source interval")
                source_params = np.stack([np.interp(source_times,source["times"],c) for c in source["params"].T],axis=-1)
            rotations, roots, fitted, report = fk.retarget(positions,matrices,contacts,output_times,
                source_rest=track["rest_positions"] * scale + offset, source_rest_matrices=track["rest_matrices"],
                source_params=source_params,skin=skin,terrain=terrain,preserve_root=True)
        reports[artifact] = report
        target = artifact if artifact == "baseline" else "candidate"
        provenance = {**simulation["provenance"], "physical_simulation": artifact == "motion",
                      "solver_to_avatar": {"scale":scale,"offset":offset.tolist()},
                      "source_time_offset_s":simulation["source_time_offset_s"]}
        output = document(fk.names,rotations,output_times,roots,provenance=provenance,kind="terra",terrain="terrain.json")
        (work / f"{target}.json").write_text(json.dumps(output,separators=(",",":"),allow_nan=False))
        _export_glb(avatar,work/f"{target}.glb",fk.names,rotations,output_times,root_translation=roots)
        append_glb(work/f"{target}.glb",terrain)
        np.savez(work/f"{target}-mhr.npz",params=fitted,root_translation=roots,times=output_times)
    reports["accepted"] = bool(simulation["completed"] and reports["motion"]["accepted"])
    reports["perceptual_review"] = "pending"
    reports["comparison"] = {key+"_change": reports["motion"][key]-reports["baseline"][key]
        for key in ("site_fit_rms_m","skin_stance_drift_p95_m","skin_ground_penetration_max_m")
        if reports["motion"].get(key) is not None and reports["baseline"].get(key) is not None}
    if reports["accepted"]:
        for suffix in ("json","glb"):
            (work/f"candidate.{suffix}").replace(work/f"motion.{suffix}")
    return reports


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--request",type=Path,required=True); ap.add_argument("--assets",type=Path,required=True)
    ap.add_argument("--work",type=Path,required=True)
    ap.add_argument("--policy-device",choices=("cpu","cuda"),default="cpu")
    ap.add_argument("--policy-device-index",type=int,default=0)
    args = ap.parse_args()
    args.work.mkdir(parents=True,exist_ok=True)
    configure_process(args.work,args.policy_device,args.policy_device_index)
    report = {"completed":False,"avatar":{"accepted":False}}
    try:
        installation = read_installation(args.assets,verify=True)
        # Editable install is checked against the receipt, not merely package metadata.
        sys.path[:0] = [str(Path(installation["sources"]["terra"])/"src"),
                       str(Path(installation["sources"]["musclemimic"])),
                       str(Path(installation["sources"]["holosoma"])/"src/holosoma_retargeting")]
        request = json.loads(args.request.read_text())
        from .terra import validate_request
        validate_request({**request,"upload_id":"video" if request.get("sidecars") else None,"fixed_camera":True})
        report = run(request,installation,args.work)
        report["avatar"] = export_tracks(request,args.work,report)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        report.update(completed=False,error=str(exc))
    (args.work/"simulation.json").write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps({"completed":report["completed"],"accepted":report.get("avatar",{}).get("accepted",False),
                      "error":report.get("error")}))
    return 0 if report["completed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
