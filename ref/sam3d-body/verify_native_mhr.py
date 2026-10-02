#!/usr/bin/env python3
"""Offline TorchScript oracle for the focused MHR runner (CPU or CUDA).

Writes deterministic references and a machine-readable parity report. Runtime
tests consume these files without importing this module or Torch.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch


def compare(got, ref, threshold):
    if got.shape != ref.shape or not np.isfinite(got).all():
        raise AssertionError("non-finite output or shape mismatch")
    diff = np.abs(got.astype(np.float64) - ref.astype(np.float64))
    result = {"max_abs": float(diff.max()), "mean_abs": float(diff.mean()),
              "max_gate": threshold, "mean_gate": threshold * .15}
    result["pass"] = result["max_abs"] < threshold and result["mean_abs"] < threshold * .15
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--assets", type=Path, required=True)
    ap.add_argument("--runner", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pose-sidecar", type=Path, required=True)
    ap.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    model = torch.jit.load(str(a.model), map_location="cpu").eval()
    captured = json.loads(a.pose_sidecar.read_text())
    rng = np.random.default_rng(20261002)
    params = np.zeros((5, 204), np.float32)
    params[1:] = np.asarray(captured["model_params"], np.float32)
    params[2:] += rng.normal(0, .015, (3, 204)).astype(np.float32)
    shape = np.zeros((5, 45), np.float32)
    shape[1] = np.asarray(captured["shape"], np.float32)
    shape[2:] = rng.normal(0, .2, (3, 45)).astype(np.float32)
    face = np.zeros((5, 72), np.float32)
    face[2:] = rng.normal(0, .15, (3, 72)).astype(np.float32)
    with torch.inference_mode():
        verts, state = model(torch.from_numpy(shape), torch.from_numpy(params), torch.from_numpy(face))
        zero_face_verts, _ = model(torch.from_numpy(shape), torch.from_numpy(params), torch.zeros((5,72)))
        # Nonzero fixture for the existing native per-stage validator.
        p=torch.from_numpy(params[2:3]); s=torch.from_numpy(shape[2:3]); f=torch.from_numpy(face[2:3])
        c=model.character_torch
        jp=c.model_parameters_to_joint_parameters(torch.cat((p,torch.zeros_like(s)),dim=1))
        stages={"mhr_params__shape":s, "mhr_params__face":f, "mhr_params__mhr_model_params":p,
                "mhr_joint_parameters":jp, "mhr_blend_shape_out":c.blend_shape(s),
                "mhr_face_expressions_out":model.face_expressions_model(f),
                "mhr_pose_correctives_out":model.pose_correctives_model(jp),
                "mhr_output__skel_state":state[2:3], "mhr_output__verts":verts[2:3]}
        (a.out/"stages").mkdir(exist_ok=True)
        for name,value in stages.items():
            np.save(a.out/"stages"/(name+".npy"),value.numpy(),allow_pickle=False)
    refs = {"params": params, "shape": shape, "face": face,
            "reference_vertices": verts.numpy(), "reference_skeleton": state.numpy(),
            "reference_zero_face_vertices": zero_face_verts.numpy()}
    for name, value in refs.items():
        np.save(a.out / (name + ".npy"), value, allow_pickle=False)
    reports = []
    for b in (1,4,5):
        for skeleton_only in (False,True):
            out = a.out / f"b{b}-{'skeleton' if skeleton_only else a.backend}"
            out.mkdir(exist_ok=True)
            for key in ("params","shape","face"):
                np.save(out/(key+".npy"),refs[key][:b],allow_pickle=False)
            cmd = [str(a.runner.resolve()), "--mhr-assets", str(a.assets.resolve()),
                   "--params", str(out/"params.npy"), "--shape", str(out/"shape.npy"),
                   "--face", str(out/"face.npy"), "--output-dir", str(out),
                   "--backend", a.backend, "--threads", "4"]
            if skeleton_only:
                cmd.append("--skeleton-only")
            started=time.perf_counter()
            proc=subprocess.run(cmd,check=True,capture_output=True,text=True,
                                env=dict(os.environ,OMP_NUM_THREADS="4"))
            item={"batch":b,"skeleton_only":skeleton_only,"runner":json.loads(proc.stdout),
                  "wall_seconds":time.perf_counter()-started}
            got=np.load(out/"skeleton.npy")
            item["skeleton"]=compare(got,refs["reference_skeleton"][:b],1e-3)
            q=got[:,:,3:7].astype(np.float64); r=refs["reference_skeleton"][:b,:,3:7].astype(np.float64)
            q/=np.linalg.norm(q,axis=-1,keepdims=True); r/=np.linalg.norm(r,axis=-1,keepdims=True)
            angle=2*np.arccos(np.clip(np.abs(np.sum(q*r,axis=-1)),0,1))
            item["quaternion_max_angle_rad"]=float(angle.max())
            item["pass"]=bool(item["skeleton"]["pass"] and angle.max()<2e-4)
            if not skeleton_only:
                item["vertices_cm"]=compare(np.load(out/"vertices.npy"),refs["reference_vertices"][:b],5e-3)
                item["pass"] &= item["vertices_cm"]["pass"]
            reports.append(item)
            print(json.dumps(item),flush=True)
    report={"cases":reports,"pass":all(r["pass"] for r in reports),
            "expression_displacement_cm":float(np.max(np.abs(verts.numpy()-zero_face_verts.numpy())))}
    (a.out/"parity.json").write_text(json.dumps(report,indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
