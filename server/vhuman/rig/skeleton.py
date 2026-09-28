"""The template skeleton, placed from a subject's features (head frame, metres).

    root -> neck -> head -> jaw -> teeth_lower
                               -> tongue_01 -> tongue_02 -> tongue_03 -> tongue_04
                         -> teeth_upper
                         -> eye_R, eye_L
Joints are world-aligned (identity orientation) except the eyes, which use
the fitted gaze frame (+x horizontal, +y up, +z gaze). Placement rules are
simple anatomical proportions around the detected features: the jaw's hinge
a little in front of and below the ear's widest point, the upper incisors
just behind the lip seam, the tongue along the floor of the mouth.
"""
from __future__ import annotations

import numpy as np

from .features import Features

JOINTS = ("root", "neck", "head", "jaw", "teeth_lower", "tongue_01", "tongue_02", "tongue_03", "tongue_04",
          "teeth_upper", "eye_R", "eye_L")
PARENT = {"root": None, "neck": "root", "head": "neck", "jaw": "head", "teeth_lower": "jaw", "tongue_01": "jaw",
          "tongue_02": "tongue_01", "tongue_03": "tongue_02", "tongue_04": "tongue_03", "teeth_upper": "head",
          "eye_R": "head", "eye_L": "head"}


def scale_of(feat: Features) -> float:
    """Subject size relative to the synthetic default (63 mm interpupillary)."""
    return float(feat.info["ipd_m"]) / 0.063


def place(feat: Features) -> dict:
    s = scale_of(feat)
    mm = 0.001 * s
    pts = feat.points
    ear = 0.5 * (pts["ear_left"] + pts["ear_right"])
    seam_mid = feat.seam[len(feat.seam) // 2]
    neck = feat.neck
    pos = {}
    pos["root"] = np.array([0.0, neck["y"], neck["center"][2]])
    pos["head"] = np.array([0.0, ear[1] - 30 * mm, ear[2] + 12 * mm])
    pos["neck"] = 0.5 * (pos["root"] + pos["head"])
    pos["jaw"] = np.array([0.0, ear[1] - 12 * mm, ear[2] + 20 * mm])
    # incisal edges: just behind the lips' contact
    pos["teeth_upper"] = np.array([0.0, seam_mid[1] - 1.8 * mm, seam_mid[2] - 9.0 * mm])
    # overbite (the lower edges above the upper ones) and overjet (behind them)
    pos["teeth_lower"] = np.array([0.0, seam_mid[1] - 0.3 * mm, seam_mid[2] - 11.5 * mm])
    root = np.array([0.0, seam_mid[1] - 16 * mm, seam_mid[2] - 48 * mm])
    tip = np.array([0.0, seam_mid[1] - 5 * mm, seam_mid[2] - 16 * mm])
    for i, t in enumerate(np.linspace(0, 1, 4)):
        pos[f"tongue_0{i + 1}"] = root + (tip - root) * t * np.array([1, 1, 1]) + np.array([0, 3 * mm * np.sin(np.pi * t), 0])
    eyes = {e["side"]: e for e in feat.eyes}
    pos["eye_R"], pos["eye_L"] = eyes["right"]["center"], eyes["left"]["center"]
    rot = {n: np.eye(3) for n in JOINTS}
    rot["eye_R"], rot["eye_L"] = eyes["right"]["rotation"], eyes["left"]["rotation"]
    joints = []
    for name in JOINTS:
        bind = np.eye(4)
        bind[:3, :3] = rot[name]
        bind[:3, 3] = pos[name]
        p = PARENT[name]
        if p is None:
            local = bind
        else:
            pb = next(j["bind"] for j in joints if j["name"] == p)
            local = np.linalg.inv(np.asarray(pb)) @ bind
        joints.append({"name": name, "parent": p, "bind": bind.tolist(),
                       "rest_translation": local[:3, 3].tolist(), "rest_rotation": local[:3, :3].tolist()})
    return {"joints": joints, "scale": s}


def usd_path(name: str) -> str:
    """UsdSkel joint path ("root/neck/head/jaw")."""
    chain = [name]
    while PARENT[chain[-1]] is not None:
        chain.append(PARENT[chain[-1]])
    return "/".join(reversed(chain))
