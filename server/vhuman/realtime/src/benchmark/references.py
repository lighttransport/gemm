"""Audit expression images against the final rig; approximate labels are not registration."""
import json
from pathlib import Path
import numpy as np
from ..avatar.provenance import verify_files, sha256
from ..avatar.rig import RigAvatar
from ....reconstruction.observations import observe
from ....reconstruction.reference import Camera


def audit(head, identity, output, work):
    head, identity, output = Path(head), Path(identity), Path(output)
    spec = json.loads((identity/"manifest.json").read_text())
    verify_files(spec["provenance"], identity)
    if sha256(head/"portrait.png") != sha256(identity/"neutral.png"): raise ValueError("identity/rig mismatch")
    reports = list((head/"reconstruction").glob("*/manifest.json"))
    if len(reports) != 1: raise ValueError("ambiguous reconstruction")
    report = json.loads(reports[0].read_text())
    camera = Camera.from_dict(report["geometry"]["fitted_cameras"][0])
    anchors = report["geometry"]["anatomical_anchors"]
    rig = RigAvatar(reports[0].parent/"rig", Path(work)/"native")
    rows = []
    try:
        for ref in spec["references"]:
            detected = observe(identity/ref["path"])["views"][0]["anchors"]
            controls = np.array([ref["controls"].get(n, 0) for n in rig.names], np.float32)
            vertices = rig.deform(controls)
            ids = list(detected)
            predicted, _ = camera.project(np.array([vertices[anchors[n]].mean(0) for n in ids]))
            observed = np.array([detected[n]["xy"] for n in ids])
            distances = np.linalg.norm(predicted-observed, axis=1)
            mouth = np.array([n in ("upper_lip", "lower_lip", "mouth_left", "mouth_right") for n in ids])
            rows.append(dict(name=ref["name"], image_sha256=sha256(identity/ref["path"]),
                controls_are_approximate=True, landmarks={n: dict(predicted=predicted[i].tolist(), observed=observed[i].tolist(), error_px=float(distances[i])) for i,n in enumerate(ids)},
                rms_px=float(np.sqrt(np.mean(distances**2))), mouth_rms_px=float(np.sqrt(np.mean(distances[mouth]**2))),
                accepted_for_expression_training=False))
    finally: rig.close()
    result = dict(format="vhuman.expression_reference_audit.v1", pose_labels="approximate generated expressions",
        camera="fixed neutral reconstruction calibration; no expression/pose fitting", references=rows,
        limitations="Landmarker/anatomical-anchor estimates need review; errors diagnose registration, not identity or perceptual quality")
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(result, indent=2))
    return result
