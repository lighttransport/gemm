"""Offline anatomical retargeting and contact diagnostics (NumPy/SciPy only)."""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

from . import mhr

# Order is explicit, shared with the simulator recorder and never inferred from IDs.
LANDMARKS = {
    "pelvis_mimic": "mhr_root", "upper_body_mimic": "mhr_c_spine2",
    "head_mimic": "mhr_c_head", "left_shoulder_mimic": "mhr_l_uparm",
    "left_elbow_mimic": "mhr_l_lowarm", "left_hand_mimic": "mhr_l_wrist",
    "right_shoulder_mimic": "mhr_r_uparm", "right_elbow_mimic": "mhr_r_lowarm",
    "right_hand_mimic": "mhr_r_wrist", "left_hip_mimic": "mhr_l_upleg",
    "left_knee_mimic": "mhr_l_lowleg", "left_ankle_mimic": "mhr_l_foot",
    "left_toes_mimic": "mhr_l_ball", "right_hip_mimic": "mhr_r_upleg",
    "right_knee_mimic": "mhr_r_lowleg", "right_ankle_mimic": "mhr_r_foot",
    "right_toes_mimic": "mhr_r_ball",
}
# Anatomical tree for proportional retargeting; shoulders attach at the thorax.
SITE_PARENTS = np.array([-1, 0, 1, 1, 3, 4, 1, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15])
FEET = [11, 12, 15, 16]
# Proper rotation: +Y up -> +Z up. No reflection or quaternion component swaps.
Y_TO_Z = Rotation.from_euler("x", 90, degrees=True).as_matrix()


def change_basis(positions, rotations, matrix=Y_TO_Z):
    p = np.asarray(positions) @ matrix.T
    r = np.asarray(rotations)
    return p, matrix @ r @ matrix.T


def resample(times, positions, quaternions, output_times):
    t, out = np.asarray(times), np.asarray(output_times)
    p = np.asarray(positions)
    if len(t) < 2 or np.any(np.diff(t) <= 0) or out[0] < t[0] - 1e-8 or out[-1] > t[-1] + 1e-8:
        raise ValueError("invalid resampling interval")
    flat = p.reshape(len(t), -1)
    pos = np.stack([np.interp(out, t, col) for col in flat.T], axis=-1).reshape((len(out),) + p.shape[1:])
    q = np.asarray(quaternions)
    rot = np.stack([Slerp(t, Rotation.from_quat(q[:, j]))(out).as_quat()
                    for j in range(q.shape[1])], axis=1)
    return pos, rot


def proportions(positions, source_rest, target_rest):
    """Preserve segment directions, adapting each anatomical segment length."""
    out = np.array(positions, dtype=np.float64, copy=True)
    for j, parent in enumerate(SITE_PARENTS):
        if parent < 0:
            continue
        a = np.linalg.norm(source_rest[j] - source_rest[parent])
        b = np.linalg.norm(target_rest[j] - target_rest[parent])
        if min(a, b) < 1e-5:
            raise ValueError("degenerate anatomical calibration")
        out[..., j, :] = out[..., parent, :] + (positions[..., j, :] - positions[..., parent, :]) * b / a
    return out


def contact_metrics(foot_positions, contacts, floor=0.0):
    """Contact drift is distance from each stance's first sample, not velocity."""
    feet, mask = np.asarray(foot_positions), np.asarray(contacts, dtype=bool)
    if feet.shape[:2] != mask.shape or feet.shape[-1] != 3 or not np.isfinite(feet).all():
        raise ValueError("invalid contact track")
    drift = []
    for j in range(mask.shape[1]):
        anchor = None
        for i in range(len(feet)):
            if not mask[i, j]:
                anchor = None
            else:
                if anchor is None:
                    anchor = feet[i, j].copy()
                drift.append(float(np.linalg.norm((feet[i, j] - anchor)[[0, 2]])))
    return {"stance_samples": len(drift),
            "stance_drift_p95_m": float(np.percentile(drift, 95)) if drift else None,
            "ground_penetration_max_m": float(max(0, floor - np.min(feet[..., 1])))}


class AvatarFeet:
    """Evaluate the actual GLB foot vertices with its exported four-weight skin."""
    def __init__(self, path, fk):
        from ..eye.glb import GLB
        glb = GLB.load(path)
        nodes = glb.doc["nodes"]
        node = next(n for n in nodes if n.get("name") == "body_skin" and "skin" in n)
        skin = glb.doc["skins"][node["skin"]]
        primitive = glb.doc["meshes"][node["mesh"]]["primitives"][0]
        attr = primitive["attributes"]
        vertices = glb.accessor(attr["POSITION"]).astype(np.float64)
        indices = glb.accessor(attr["JOINTS_0"]).astype(int)
        weights = glb.accessor(attr["WEIGHTS_0"]).astype(np.float64)
        mapping = np.array([fk.names.index(nodes[j]["name"]) if nodes[j]["name"] in fk.names else -1
                            for j in skin["joints"]])
        joints = mapping[indices]
        labels = np.full(len(vertices), -1)
        rest_positions, _, _ = fk.forward(fk.bind_params)
        for k, site in enumerate(FEET):
            joint = fk.sites[site]
            strength = np.sum(weights * (joints == joint), axis=1)
            labels[strength > .5] = k
        # MHR's ankle joint has no direct skin weights: the heel is weighted
        # to lower-leg procedural joints. Partition each foot geometrically
        # between its ankle and toe anchors, constrained to the correct leg.
        for k, side in ((0, "l"), (2, "r")):
            leg = [j for j, n in enumerate(fk.names) if n.startswith(f"mhr_{side}_")
                   and any(part in n for part in ("lowleg", "foot", "ball"))]
            strength = np.sum(weights * np.isin(joints, leg), axis=1)
            anchors = rest_positions[[fk.sites[FEET[k]], fk.sites[FEET[k+1]]]]
            distance = np.linalg.norm(vertices[:, None] - anchors, axis=-1)
            near = (strength > .5) & (distance.min(axis=1) < .20) & (vertices[:, 1] < anchors[0, 1] + .025)
            labels[near] = k + np.argmin(distance[near], axis=1)
        selected = np.flatnonzero(labels >= 0)
        if any(np.count_nonzero(labels == k) < 3 for k in range(4)):
            raise ValueError("avatar lacks foot/toe skin samples")
        if np.any((joints[selected] < 0) & (weights[selected] > 0)):
            raise ValueError("foot skin references unsupported joints")
        self.labels = labels[selected]
        self.joints = np.maximum(joints[selected], 0)
        self.weights = weights[selected]
        inverse = glb.accessor(skin["inverseBindMatrices"]).reshape(-1, 4, 4).transpose(0, 2, 1)
        homogeneous = np.c_[vertices[selected], np.ones(len(selected))]
        self.local = np.einsum("nkij,nj->nki", inverse[indices[selected]], homogeneous)[..., :3]
        self.scale = fk.scales.copy()
        for j, parent in enumerate(fk.parents):
            if parent >= 0:
                self.scale[j] *= self.scale[parent]
        p, r, _ = fk.forward(fk.bind_params)
        rest = self.vertices(p, r)
        self.floor = float(np.min(rest[:, 1]))
        np.testing.assert_allclose(rest, vertices[selected], atol=2e-5,
                                   err_msg="foot skin inverse-bind mismatch")

    def vertices(self, positions, matrices):
        transformed = np.einsum("nkij,nkj->nki", matrices[self.joints],
                                self.local * self.scale[self.joints, None]) + positions[self.joints]
        return np.sum(transformed * self.weights[..., None], axis=1)

    def centers(self, vertices):
        return np.array([vertices[self.labels == k].mean(axis=0) for k in range(4)])


class MHRKinematics:
    """Use the exported parameter transform; lock avatar translations and scales.

    Only angular parameter columns are fitted. Face/finger-only columns are
    retained from the source pose (or the avatar bind pose). Procedural joints
    remain coupled by MHR's original parameter matrix.
    """
    def __init__(self, definition, assets, bind_params):
        rig = mhr.metadata(assets)
        by_name = {j["name"]: j for j in definition["joints"]}
        self.names = rig["names"]
        self.parents = rig["parents"]
        if any(n not in by_name for n in self.names):
            raise ValueError("avatar does not match MHR assets")
        joints = [by_name[n] for n in self.names]
        self.translations = np.array([j["rest_translation"] for j in joints])
        self.scales = np.array([j["rest_scale"] for j in joints])
        self.bind_q = Rotation.from_matrix([j["rest_rotation"] for j in joints]).as_quat()
        self.root = self.names.index("mhr_root")
        self.sites = [self.names.index(n) for n in LANDMARKS.values()]
        self.bind_params = np.asarray(bind_params, dtype=np.float64)
        if self.bind_params.shape != (204,) or not np.isfinite(self.bind_params).all():
            raise ValueError("invalid MHR bind parameters")
        path = assets / (mhr.STEM + ".safetensors")
        self.transform = mhr._tensor(path, "parameter_transform")[:, :204].astype(np.float64).reshape(127, 7, 204)
        self.prerot = Rotation.from_quat(mhr._tensor(path, "skeleton.joint_prerotations"))
        ancestors = set(self.sites)
        for site in self.sites:
            parent = int(self.parents[site])
            while parent >= 0:
                ancestors.add(parent)
                parent = int(self.parents[parent])
        angular = np.any(abs(self.transform[sorted(ancestors), 3:6]) > 1e-12, axis=(0, 1))
        nonangular = np.any(abs(self.transform[:, [0, 1, 2, 6]]) > 1e-12, axis=(0, 1))
        self.active = np.flatnonzero(angular & ~nonangular)
        if not len(self.active):
            raise ValueError("no independent MHR angular controls")
        predicted = self.local(self.bind_params)
        if np.max((Rotation.from_quat(predicted).inv() * Rotation.from_quat(self.bind_q)).magnitude()) > 1e-3:
            raise ValueError("MHR bind pose differs from avatar; rebuild avatar before retargeting")

    def local(self, params):
        angles = np.einsum("jai,i->ja", self.transform[:, 3:6], params)
        return (self.prerot * Rotation.from_euler("xyz", angles)).as_quat()

    def forward(self, params, root=None):
        q = self.local(params)
        mats = Rotation.from_quat(q).as_matrix()
        xyz = self.translations.copy()
        if root is not None:
            xyz[self.root] = root
        scale = self.scales.copy()
        for j, parent in enumerate(self.parents):
            if parent >= 0:
                xyz[j] = xyz[parent] + mats[parent] @ (scale[parent] * xyz[j])
                mats[j] = mats[parent] @ mats[j]
                scale[j] *= scale[parent]
        return xyz, mats, q

    def retarget(self, positions, matrices, contacts, times, *, source_rest, source_rest_matrices,
                 source_params=None, skin=None):
        """Fit body pose and stance constraints; return full MHR rotation tracks."""
        rest, rest_m, _ = self.forward(self.bind_params)
        target_rest, target_m = rest[self.sites], rest_m[self.sites]
        positions = proportions(positions, source_rest, target_rest)
        # Keep the first horizontal root at the avatar's original location.
        offset = target_rest[0] - positions[0, 0]
        offset[1] = 0
        positions += offset
        bind_floor = skin.floor if skin is not None else float(np.min(target_rest[FEET, 1]))
        feet_floor = 0.0
        # Site anchors have height above the sole. Calibrate that height once.
        sole_offsets = target_rest[FEET, 1] - bind_floor
        floor_shift = np.min(positions[:, FEET, 1] - sole_offsets) - feet_floor
        positions[..., 1] -= floor_shift
        orientations = matrices @ np.swapaxes(source_rest_matrices, -1, -2) @ target_m
        roots, rotations, fitted, errors, foot_track = [], [], [], [], []
        previous = self.bind_params.copy()
        skin_track, skin_minimum = [], feet_floor
        anchors = [None] * 4
        for i in range(len(times)):
            base = self.bind_params.copy() if source_params is None else np.array(source_params[i], copy=True)
            # Translation/scale parameters always belong to this avatar.
            frozen = np.any(abs(self.transform[:, [0, 1, 2, 6]]) > 1e-12, axis=(0, 1))
            base[frozen] = self.bind_params[frozen]
            target = positions[i].copy()
            for k, j in enumerate(FEET):
                if contacts[i, k]:
                    if anchors[k] is None:
                        anchors[k] = target[j].copy()
                        anchors[k][1] = feet_floor + sole_offsets[k]
                    target[j] = anchors[k]
                else:
                    anchors[k] = None
            parent = int(self.parents[self.root])
            parent_rot = rest_m[parent] if parent >= 0 else np.eye(3)
            parent_pos = rest[parent] if parent >= 0 else np.zeros(3)
            parent_scale = 1.0
            ancestor = parent
            while ancestor >= 0:
                parent_scale *= self.scales[ancestor]
                ancestor = int(self.parents[ancestor])
            initial_root = parent_rot.T @ (target[0] - parent_pos) / parent_scale
            def residual(x):
                p = base.copy()
                p[self.active] = x[:-3]
                xyz, mat, _ = self.forward(p, x[-3:])
                delta = xyz[self.sites] - target
                weight = np.ones((len(self.sites), 1))
                weight[FEET] = 1 + 9 * np.asarray(contacts[i])[:, None]
                angle = Rotation.from_matrix(np.swapaxes(orientations[i], -1, -2) @ mat[self.sites]).as_rotvec()
                return np.concatenate([(delta * weight).ravel(), .08 * angle.ravel(),
                                       .005 * (x[:-3] - previous[self.active])])
            x0 = np.r_[previous[self.active], initial_root]
            result = least_squares(residual, x0, max_nfev=60, ftol=1e-5, xtol=1e-5, gtol=1e-5)
            base[self.active] = result.x[:-3]
            xyz, mat, q = self.forward(base, result.x[-3:])
            if skin is not None:
                vertices = skin.vertices(xyz, mat)
                skin_minimum = min(skin_minimum, float(vertices[:, 1].min()))
                skin_track.append(skin.centers(vertices))
            errors.append(float(np.sqrt(np.mean(np.sum((xyz[self.sites] - target)**2, axis=-1)))))
            roots.append(result.x[-3:]); rotations.append(q); fitted.append(base)
            feet = xyz[self.sites][FEET].copy()
            feet[:, 1] -= sole_offsets
            foot_track.append(feet)
            previous = base
        report = contact_metrics(np.array(foot_track), contacts, feet_floor)
        report["site_fit_rms_m"] = float(np.sqrt(np.mean(np.square(errors))))
        if skin is not None:
            skinned = contact_metrics(np.array(skin_track), contacts, feet_floor)
            report["skin_stance_drift_p95_m"] = skinned["stance_drift_p95_m"]
            report["skin_ground_penetration_max_m"] = max(0., feet_floor - skin_minimum)
        report["accepted"] = (report["stance_samples"] > 0 and report["stance_drift_p95_m"] <= .02
                              and report["ground_penetration_max_m"] <= .01 and report["site_fit_rms_m"] <= .05)
        if skin is not None:
            report["accepted"] = bool(report["accepted"] and report["skin_stance_drift_p95_m"] <= .02
                                      and report["skin_ground_penetration_max_m"] <= .01)
        return np.array(rotations), np.array(roots), np.array(fitted), report
