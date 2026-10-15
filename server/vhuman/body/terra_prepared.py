# SPDX-License-Identifier: Apache-2.0
"""Named prepared landmarks with optional source-root tracking, without licensed body models.

Adapted from the pinned Apache-2.0 Terra orchestration pipeline and the
project mhr.patch prepared-motion adapter. This first-party copy keeps the
registered upstream source unchanged while extending its solver objective. Keep source geometry
and robot calibration explicit; do not synthesize SMPL parameters or shape caches.
"""
from types import SimpleNamespace

import mujoco
import numpy as np

from terra._musclemimic import TerrainSpec
from terra.assembly import SolveContext
from terra.constants import MYOFULLBODY_SITE_CALIBRATION, SMPLH_TO_MYOFULLBODY
from terra.pipeline import (
    _build_environment, _configure_solver, _interaction_scene, _motion_analysis,
    _postprocess_solution, _solve,
)
from terra.postprocess import repair_collisions
from terra.profiles import resolve_solver_config
from terra.reconstruction import CalibrationEvidence, ReconstructionRequest, reconstruct_terrain
from terra.stability import apply_stability_policy


from terra.prepared import PreparedMotion


def fit_prepared_motion(motion, robot_conf, logger, *, terrain="auto", config=None, source_root_weight=0.):
    """Use TERRA's unchanged constraint/solve/postprocess stages for prepared MHR."""
    motion.validate()
    if not np.isfinite(source_root_weight) or source_root_weight < 0:
        raise ValueError("source root weight must be finite and nonnegative")
    raw = dict(config or {}) | {"use_fitted_shape": False, "calibrate_sites": False,
                                "seat_contact_mode": "off", "extra_landmarks": {}}
    cfg = resolve_solver_config("terra", raw)
    names = motion.joint_names
    if isinstance(terrain, str) and terrain == "auto":
        result = reconstruct_terrain(ReconstructionRequest(
            mode="fit", joints=motion.positions, joint_names=names, fps=motion.fps,
            fit_options={"seat": "off", "calibrated_joint_offsets": motion.sole_offsets,
                         "neutral_foot_pitch": motion.neutral_foot_pitch,
                         "neutral_foot_pitch_source": "mhr_native_rest"},
            calibration=CalibrationEvidence(neutral_foot_pitch=motion.neutral_foot_pitch,
                                             joint_offsets=motion.sole_offsets, source="mhr_native_rest")))
        if not result.validation.get("passed", False):
            raise ValueError(f"terrain reconstruction rejected: {result.validation}")
        spec, reconstruction = result.terrain, {"fit": result.report, "validation": result.validation}
    elif terrain is None or isinstance(terrain, str) and terrain == "flat":
        spec, reconstruction = None, {"validation": {"passed": True}, "source": "explicit-flat"}
    else:
        spec = terrain if isinstance(terrain, TerrainSpec) else TerrainSpec.from_dict(terrain)
        from terra.terrain import validate_terrain
        check = validate_terrain(motion.positions, list(names), spec, motion.fps, offsets=motion.sole_offsets)
        if not check.get("passed", False):
            raise ValueError(f"known terrain rejected: {check}")
        reconstruction = {"validation": check, "source": "explicit"}
    env = _build_environment("MyoFullBody", robot_conf, spec, logger)
    model = env._model
    rest = mujoco.MjData(model); mujoco.mj_forward(model, rest)
    mapping = dict(SMPLH_TO_MYOFULLBODY)
    sites, targets, site_names = [], [], []
    for site, joint, _body in MYOFULLBODY_SITE_CALIBRATION:
        index = names.index(joint)
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
        if sid < 0:
            raise ValueError(f"missing robot calibration site: {site}")
        offset = motion.rest_rotations[index].T @ rest.site_xmat[sid].reshape(3, 3)
        targets.append(motion.rotations[:, index] @ offset)
        sites.append(sid); site_names.append(site)
    from scipy.spatial.transform import Rotation
    source_left = motion.rest_positions[names.index("L_Shoulder")] - motion.rest_positions[names.index("R_Shoulder")]
    robot_left = rest.site_xpos[sites[3]] - rest.site_xpos[sites[6]]
    if min(np.linalg.norm(source_left[:2]), np.linalg.norm(robot_left[:2])) < .05:
        raise ValueError("unobservable neutral anatomical heading")
    yaw = np.arctan2(source_left[1],source_left[0]) - np.arctan2(robot_left[1],robot_left[0])
    heading = Rotation.from_euler("z",yaw).as_matrix()
    free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if len(free) != 1:
        raise ValueError("prepared motion requires one robot free root")
    root_body = model.jnt_bodyid[free[0]]
    root_rotation = motion.rotations[0,0] @ motion.rest_rotations[0].T @ heading @ rest.xmat[root_body].reshape(3,3)
    root_quaternion = Rotation.from_matrix(root_rotation).as_quat(scalar_first=True)
    # Height is measured from the native robot, never from a fitted SMPL shape.
    robot_height = float(rest.site_xpos[sites[2], 2] - min(rest.site_xpos[sites[k], 2] for k in (11,12,15,16)))
    cfg = resolve_solver_config("terra", raw | {"source_sole_offsets": motion.sole_offsets}).for_scene(
        on_terrain=spec is not None and not spec.is_flat)
    scene = _interaction_scene(None, spec, len(motion.positions), cfg, logger)
    ctx = SolveContext(config=cfg, logger=logger, model=model, terrain=spec, fps=motion.fps,
        joints_mapping=mapping, human_joints=motion.positions.copy(), smpl_rotations=motion.rotations.copy(),
        scene=scene, source_joint_names=names, prepared_root_quaternion=root_quaternion,
        prepared_orientation={"site_ids": np.array(sites), "targets": np.stack(targets, axis=1), "names": site_names})
    assets = SimpleNamespace(base_env_name="MyoFullBody", fitted_shape_path=None)
    setup = _configure_solver(ctx, env, robot_conf, assets, model.nq - 7, robot_height)
    if source_root_weight:
        from .terra_root import RootTrackingObjective
        state = setup.retargeter._constraints
        state.objective = RootTrackingObjective(state.objective, ctx.human_joints[:, 0], source_root_weight)
    native = _solve(ctx, setup.retargeter, setup.foot_sticking, model.nq - 7)
    native = repair_collisions(ctx, native, setup.selfpen_mode)
    solved = _postprocess_solution(ctx, env, "MyoFullBody", setup.retargeter, native, setup.selfpen_mode)
    analysis = _motion_analysis(ctx, setup.retargeter, native, solved, spec)
    analysis.update(source_format="mhr", source_joint_names=names,
                    site_calibration_version="myofullbody_mhr_sites_v1", reconstruction=reconstruction,
                    source_root_objective={"weight":float(source_root_weight), "source":"prepared root positions"})
    output = SimpleNamespace(trajectory=solved.trajectory, analysis=analysis, terrain=spec)
    return apply_stability_policy(output, config=raw, retry=lambda overrides:
        fit_prepared_motion(motion, robot_conf, logger, terrain=spec, config=overrides, source_root_weight=source_root_weight))
