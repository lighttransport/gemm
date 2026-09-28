"""Controls, the rig definition and its evaluation (the deformer's front end).

A linear facial rig in the spirit of published rig-evaluation designs
(e.g. the MIT-licensed OpenRigLogic's documentation): no code or data is
taken from them, and all names and values here are our own.

    x      = clamp(controls, range)                   control vector
    c_p    = min(1, w_p * prod_i clamp(x_i, 0, 1))    corrective ("PSD") inputs
    in     = [x | c]
    dJ     = M in        joint deltas per (joint, tx ty tz rx ry rz), sparse M
    b_k    = in[src_k]   blendshape weights

A joint's local transform is its rest transform followed by the delta:
translation rest_t + dt, rotation rest_R @ Rz(rz) @ Ry(ry) @ Rx(rx) (radians).
Geometry: v = rest + sum_k b_k delta_k, then linear blend skinning with
world_j @ inverse(bind_j). The same equations run in the web viewer.

Control names: the 51 expression names of the LightRig canonical face
namespace (lr.face.v1, ARKit-style), so LightRig/MediaPipe tracks drive
this rig directly, plus tongue and head controls.
"""
from __future__ import annotations

import math

import numpy as np

LR_FACE_V1 = (
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight", "cheekPuff",
    "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight",
    "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen",
    "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight",
    "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight",
    "mouthPucker", "mouthRight", "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
    "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft",
    "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight")
EXTRA = ("tongueOut", "tongueUp", "tongueDown", "tongueLeft", "tongueRight", "tongueCurlUp",
         "headYaw", "headPitch", "headRoll")
CONTROLS = LR_FACE_V1 + EXTRA
SIGNED = {"headYaw", "headPitch", "headRoll"}
ATTRS = ("tx", "ty", "tz", "rx", "ry", "rz")

GROUPS = {"brow": "brow", "cheek": "cheek", "eyeBlink": "eyes", "eyeLook": "gaze", "eyeSquint": "eyes",
          "eyeWide": "eyes", "jaw": "jaw", "mouth": "mouth", "nose": "nose", "tongue": "tongue", "head": "head"}


def group_of(name: str) -> str:
    for prefix, g in GROUPS.items():
        if name.startswith(prefix):
            return g
    return "other"


def control_table() -> list[dict]:
    return [{"name": n, "min": -1.0 if n in SIGNED else 0.0, "max": 1.0, "default": 0.0, "group": group_of(n)}
            for n in CONTROLS]


def deg(v):
    return math.radians(v)


def joint_matrix_entries(scale: float) -> list[tuple]:
    """(input name, joint, attr, value per unit input). `scale` multiplies
    translations (the subject's size relative to the synthetic default)."""
    mm = 0.001 * scale
    e = [
        ("jawOpen", "jaw", "rx", deg(21.0)), ("jawOpen", "jaw", "ty", -1.2 * mm), ("jawOpen", "jaw", "tz", 1.5 * mm),
        ("jawLeft", "jaw", "ry", deg(5.0)), ("jawLeft", "jaw", "tx", 1.8 * mm),
        ("jawRight", "jaw", "ry", deg(-5.0)), ("jawRight", "jaw", "tx", -1.8 * mm),
        ("jawForward", "jaw", "tz", 4.5 * mm),
        ("mouthClose", "jaw", "rx", 0.0),
        # gaze (eye joints: +x horizontal, +y up, +z gaze)
        ("eyeLookUpLeft", "eye_L", "rx", deg(-22.0)), ("eyeLookUpRight", "eye_R", "rx", deg(-22.0)),
        ("eyeLookDownLeft", "eye_L", "rx", deg(26.0)), ("eyeLookDownRight", "eye_R", "rx", deg(26.0)),
        ("eyeLookInLeft", "eye_L", "ry", deg(-28.0)), ("eyeLookInRight", "eye_R", "ry", deg(28.0)),
        ("eyeLookOutLeft", "eye_L", "ry", deg(32.0)), ("eyeLookOutRight", "eye_R", "ry", deg(-32.0)),
        # tongue chain
        ("tongueOut", "tongue_01", "tz", 24.0 * mm), ("tongueOut", "tongue_01", "ty", 3.0 * mm),
        ("tongueOut", "tongue_02", "rx", deg(-6.0)), ("tongueOut", "tongue_03", "rx", deg(-4.0)),
        ("tongueUp", "tongue_02", "rx", deg(-14.0)), ("tongueUp", "tongue_03", "rx", deg(-18.0)),
        ("tongueUp", "tongue_04", "rx", deg(-18.0)),
        ("tongueDown", "tongue_02", "rx", deg(10.0)), ("tongueDown", "tongue_03", "rx", deg(14.0)),
        ("tongueDown", "tongue_04", "rx", deg(14.0)),
        ("tongueLeft", "tongue_02", "ry", deg(10.0)), ("tongueLeft", "tongue_03", "ry", deg(12.0)),
        ("tongueLeft", "tongue_04", "ry", deg(12.0)),
        ("tongueRight", "tongue_02", "ry", deg(-10.0)), ("tongueRight", "tongue_03", "ry", deg(-12.0)),
        ("tongueRight", "tongue_04", "ry", deg(-12.0)),
        ("tongueCurlUp", "tongue_03", "rx", deg(-25.0)), ("tongueCurlUp", "tongue_04", "rx", deg(-40.0)),
        # head (split over the neck and the head joint)
        ("headYaw", "neck", "ry", deg(16.0)), ("headYaw", "head", "ry", deg(24.0)),
        ("headPitch", "neck", "rx", deg(-10.0)), ("headPitch", "head", "rx", deg(-15.0)),
        ("headRoll", "neck", "rz", deg(8.0)), ("headRoll", "head", "rz", deg(12.0)),
    ]
    return [x for x in e if x[3] != 0.0]


CORRECTIVES = (
    # upper lid: blink and look-down both lower it; take the overlap back
    ("corr_blink_lookDown_L", ("eyeBlinkLeft", "eyeLookDownLeft"), 1.0),
    ("corr_blink_lookDown_R", ("eyeBlinkRight", "eyeLookDownRight"), 1.0),
    # smiling with an open jaw: corners stay up with the cheeks
    ("corr_jawOpen_smile_L", ("jawOpen", "mouthSmileLeft"), 1.0),
    ("corr_jawOpen_smile_R", ("jawOpen", "mouthSmileRight"), 1.0),
)


def euler_matrix(rx, ry, rz) -> np.ndarray:
    cx, sx, cy, sy, cz, sz = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry), math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


class Rig:
    """Evaluate a rig definition (rig.json's "rig" section)."""

    def __init__(self, d: dict):
        self.d = d
        self.controls = [c["name"] for c in d["controls"]]
        self.cidx = {n: i for i, n in enumerate(self.controls)}
        self.lo = np.array([c["min"] for c in d["controls"]])
        self.hi = np.array([c["max"] for c in d["controls"]])
        self.correctives = d["correctives"]
        self.inputs = self.controls + [c["name"] for c in self.correctives]
        self.iidx = {n: i for i, n in enumerate(self.inputs)}
        self.joints = d["joints"]
        self.jidx = {j["name"]: i for i, j in enumerate(self.joints)}
        J = len(self.joints)
        ent = d["joint_matrix"]
        self.M_rows = np.array([self.jidx[e["joint"]] * 6 + ATTRS.index(e["attr"]) for e in ent], np.int64)
        self.M_cols = np.array([self.iidx[e["input"]] for e in ent], np.int64)
        self.M_vals = np.array([e["value"] for e in ent], np.float64)
        self.rest_t = np.array([j["rest_translation"] for j in self.joints], np.float64)
        self.rest_R = np.array([j["rest_rotation"] for j in self.joints], np.float64)   # (J, 3, 3)
        self.parent = np.array([self.jidx.get(j["parent"], -1) if j["parent"] else -1 for j in self.joints])
        self.bind = np.array([j["bind"] for j in self.joints], np.float64)                 # (J, 4, 4) world
        self.inv_bind = np.linalg.inv(self.bind)
        self.shape_names = [b["name"] for b in d["blendshapes"]]
        self.shape_src = np.array([self.iidx[b["input"]] for b in d["blendshapes"]], np.int64)
        self.J = J

    def input_vector(self, controls) -> np.ndarray:
        x = np.zeros(len(self.controls))
        if isinstance(controls, dict):
            for k, v in controls.items():
                if k in self.cidx:
                    x[self.cidx[k]] = float(v)
        else:
            x[:] = np.asarray(controls, np.float64)
        x = np.clip(x, self.lo, self.hi)
        c = []
        for cor in self.correctives:
            p = cor.get("weight", 1.0)
            for name in cor["inputs"]:
                p *= min(max(x[self.cidx[name]], 0.0), 1.0)
            c.append(min(1.0, p))
        return np.concatenate([x, np.asarray(c)])

    def evaluate(self, controls) -> dict:
        inp = self.input_vector(controls)
        delta = np.zeros(self.J * 6)
        np.add.at(delta, self.M_rows, self.M_vals * inp[self.M_cols])
        delta = delta.reshape(self.J, 6)
        local = np.tile(np.eye(4), (self.J, 1, 1))
        world = np.zeros_like(local)
        for j in range(self.J):
            local[j, :3, :3] = self.rest_R[j] @ euler_matrix(*delta[j, 3:])
            local[j, :3, 3] = self.rest_t[j] + delta[j, :3]
            p = self.parent[j]
            world[j] = local[j] if p < 0 else world[p] @ local[j]
        return {"inputs": inp, "joint_delta": delta, "local": local, "world": world,
                "skin": world @ self.inv_bind, "weights": inp[self.shape_src]}


def deform(rest: np.ndarray, joints: np.ndarray, weights: np.ndarray, skin_mats: np.ndarray,
           shapes: list | None = None, shape_weights: np.ndarray | None = None) -> np.ndarray:
    """rest (V,3); joints/weights (V,K); shapes: [(indices, deltas)] per shape."""
    p = rest.copy()
    if shapes is not None and shape_weights is not None:
        for (idx, dl), w in zip(shapes, shape_weights):
            if w != 0.0:
                p[idx] += w * dl
    ph = np.concatenate([p, np.ones((len(p), 1))], 1)
    out = np.zeros_like(p)
    for k in range(joints.shape[1]):
        m = skin_mats[joints[:, k]]                       # (V, 4, 4)
        out += weights[:, k:k + 1] * np.einsum("vij,vj->vi", m[:, :3, :], ph)
    return out


def read_track(path) -> tuple[np.ndarray, list[dict]]:
    """A LightRig face track (timestamp + 52 lr.face.v1 values [+ confidence
    valid] per line) -> timestamps and per-frame {control: value}."""
    names = ("_neutral",) + LR_FACE_V1
    times, frames = [], []
    with open(path) as fh:
        lines = fh.readlines()
    for line in lines:
        f = line.split()
        if not f:
            continue
        if len(f) not in (53, 55):
            raise ValueError(f"track rows need 53 or 55 fields, got {len(f)}")
        vals = [float(v) for v in f]
        times.append(vals[0])
        frames.append({n: v for n, v in zip(names, vals[1:53]) if n != "_neutral"})
    return np.asarray(times), frames
