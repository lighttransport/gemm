"""Offline LightGeom facial tissue simulation for an existing vhuman take.

The native rig supplies target positions. A compact cheek/chin/perioral patch
is extruded and simulated by LightGeom, then exported as time-sampled USD.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import struct
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np

from ..eye.glb import GLB
from . import native, rigdef
from .common import vertex_normals

ROOT = Path(__file__).resolve().parents[3]
LOCAL_RUNNER = ROOT / "third_party/LightGeom/build-vhuman/examples/lightphysics_vhuman_face/lightphysics_vhuman_face"
LEGACY_RUNNER = Path.home() / "work/lightgeom/build/examples/lightphysics_vhuman_face/lightphysics_vhuman_face"
DEFAULT_RUNNER = LOCAL_RUNNER if LOCAL_RUNNER.is_file() or not LEGACY_RUNNER.is_file() else LEGACY_RUNNER


def availability(runner: Path = DEFAULT_RUNNER, python: Path | None = None) -> dict:
    from .job import DEFAULT_PYTHON
    py = Path(python) if python else DEFAULT_PYTHON
    if not py.exists():
        return {"available": False, "reason": f"no rig interpreter ({py})"}
    if not Path(runner).is_file():
        return {"available": False, "reason": f"no LightGeom face runner ({runner}); build target lightphysics_vhuman_face"}
    return {"available": True, "python": str(py), "runner": str(runner)}


def _skin_tris(rig_dir: Path) -> np.ndarray:
    g = GLB.load(rig_dir / "rig.glb")
    skin = next(m for m in g.doc["meshes"] if m.get("name") == "head_skin")
    triangles = g.accessor(skin["primitives"][0]["indices"]).reshape(-1, 3)
    viz = json.loads((rig_dir / "viz.json").read_text())
    vmap = np.asarray(viz["parts"]["head_skin"]["vmap"], np.int32)
    return vmap[triangles]


def _patch(rest: np.ndarray, triangles: np.ndarray, features: dict) -> tuple[np.ndarray, np.ndarray]:
    y_eye = float(np.mean([e["center"][1] for e in features["eyes"]]))
    y_chin = float(features["points"]["menton"][1])
    x_mid = float(np.mean([e["center"][0] for e in features["eyes"]]))
    z_front = float(np.quantile(rest[:, 2], .42))
    visible = ((np.abs(rest[:, 0] - x_mid) < .082) &
               (rest[:, 1] > y_chin - .008) & (rest[:, 1] < y_eye - .009) &
               (rest[:, 2] > z_front))
    selected = triangles[visible[triangles].all(1)]
    if len(selected) < 100:
        raise ValueError("facial tissue patch has fewer than 100 triangles")
    # Retriangulate the selected front surface on an XY grid. A fitted scan
    # may have subpixel slivers or locally folded faces, which are unsuitable
    # as the boundary of a tetrahedral layer. Keep rig vertices as samples so
    # the simulated patch remains driven by the same animation controls.
    from scipy.spatial import Delaunay, cKDTree
    candidates = np.unique(selected)
    xy = rest[candidates, :2]
    cells = np.floor((xy - xy.min(0)) / .003).astype(np.int32)
    key = cells[:, 0] * 10000 + cells[:, 1]
    order = np.lexsort((-rest[candidates, 2], key))
    ranked = key[order]
    ids = candidates[order[np.r_[True, ranked[1:] != ranked[:-1]]]]
    faces = Delaunay(rest[ids, :2]).simplices.astype(np.int32)
    surface = rest[ids]
    corners = surface[faces]
    edges = np.linalg.norm(corners[:, [1, 2, 0]] - corners[:, [0, 1, 2]], axis=2)
    u = corners[:, 1, :2] - corners[:, 0, :2]
    v = corners[:, 2, :2] - corners[:, 0, :2]
    signed_area = (u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]) * .5
    source_centers = rest[selected].mean(1)[:, :2]
    gap, _ = cKDTree(source_centers).query(corners.mean(1)[:, :2])
    keep = (edges.max(1) < .009) & (np.abs(signed_area) > 1e-7) & (gap < .0035)
    faces = faces[keep]
    signed_area = signed_area[keep]
    faces[signed_area < 0] = faces[signed_area < 0][:, [0, 2, 1]]
    if len(faces) < 100:
        raise ValueError("facial tissue retriangulation has fewer than 100 faces")
    used, compact = np.unique(faces, return_inverse=True)
    ids, faces = ids[used], compact.reshape(-1, 3).astype(np.int32)
    cross = np.cross(rest[ids][faces[:, 1]] - rest[ids][faces[:, 0]],
                     rest[ids][faces[:, 2]] - rest[ids][faces[:, 0]])
    if np.any(np.linalg.norm(cross, axis=1) < 1e-10):
        raise ValueError("facial tissue patch has degenerate triangles")
    # Scan fitting can leave sliver triangles whose normal extrusion makes a
    # nearly zero-volume tetrahedron. Cull them before handing the shell to
    # the solver, then compact and re-evaluate normals after each cull.
    for _ in range(4):
        outer = rest[ids]
        inner = outer - .005 * vertex_normals(outer, faces)
        points = np.concatenate((outer, inner))
        a, b, c = faces.T
        n = len(outer)
        tet = (np.stack((a, b, c, a + n), 1),
               np.stack((b, c, a + n, b + n), 1),
               np.stack((c, a + n, b + n, c + n), 1))
        valid = np.ones(len(faces), bool)
        for corners in tet:
            p = points[corners]
            volume = np.einsum("ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]),
                               p[:, 3] - p[:, 0]) / 6.
            valid &= np.abs(volume) >= 1e-12
        if valid.all():
            return ids.astype(np.int32), faces
        if valid.sum() < 100:
            raise ValueError("facial tissue patch has too few valid tetrahedra")
        old_ids, compact = np.unique(faces[valid], return_inverse=True)
        ids, faces = ids[old_ids], compact.reshape(-1, 3).astype(np.int32)
    raise ValueError("facial tissue patch could not form stable tetrahedra")


def _write_input(path: Path, rest: np.ndarray, faces: np.ndarray, positions: np.ndarray,
                 fps: float, thickness: float = .005) -> None:
    if not (np.isfinite(rest).all() and np.isfinite(positions).all() and np.isfinite(fps)
            and 10 <= fps <= 240):
        raise ValueError("invalid tissue input positions or frame rate")
    if positions.shape[1:] != rest.shape or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("inconsistent tissue input shape")
    with path.open("wb") as f:
        f.write(struct.pack("<8sIIIIff", b"VHSOFT1", 1, len(rest), len(faces),
                            len(positions), 1. / fps, thickness))
        f.write(np.ascontiguousarray(rest, dtype="<f4").tobytes())
        f.write(np.ascontiguousarray(faces, dtype="<i4").tobytes())
        f.write(np.ascontiguousarray(positions, dtype="<f4").tobytes())


def _motion_safe_patch(rest: np.ndarray, positions: np.ndarray, ids: np.ndarray,
                       faces: np.ndarray, thickness: float = .005) -> tuple[np.ndarray, np.ndarray, int]:
    """Remove faces whose driven prisms invert under the requested motion."""
    initial = len(faces)
    for _ in range(4):
        outer = rest[ids]
        inner = outer - thickness * vertex_normals(outer, faces)
        n = len(ids)
        a, b, c = faces.T
        tets = (np.stack((a, b, c, a + n), 1),
                np.stack((b, c, a + n, b + n), 1),
                np.stack((c, a + n, b + n, c + n), 1))
        def volumes(points):
            result = []
            for tet in tets:
                p = points[tet]
                result.append(np.einsum("ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]),
                                        p[:, 3] - p[:, 0]) / 6.)
            return result
        reference = volumes(np.concatenate((outer, inner)))
        good = np.ones(len(faces), bool)
        previous = outer
        for target in positions[:, ids]:
            for alpha in (.25, .5, .75, 1.):
                pose = previous * (1. - alpha) + target * alpha
                delta = pose - outer
                moved = volumes(np.concatenate((pose, inner + delta)))
                for old, new in zip(reference, moved):
                    good &= new / old > .2
            previous = target
        if good.all():
            return ids, faces, initial - len(faces)
        if good.sum() < 100:
            raise ValueError("facial motion leaves too few valid tissue faces")
        used, compact = np.unique(faces[good], return_inverse=True)
        ids, faces = ids[used], compact.reshape(-1, 3).astype(np.int32)
    raise ValueError("facial motion could not form stable tissue tetrahedra")


def simulate(rig_dir: Path, take_dir: Path, runner: Path = DEFAULT_RUNNER,
             progress=None, cancel=None) -> dict:
    rig_dir, take_dir, runner = Path(rig_dir), Path(take_dir), Path(runner)
    if not runner.is_file():
        raise FileNotFoundError(f"build LightGeom target lightphysics_vhuman_face: {runner}")
    animation = json.loads((take_dir / "animation.json").read_text())
    if animation.get("format") != "vhuman.performance.v1":
        raise ValueError("unsupported take animation format")
    frames = animation["frames"]
    if not 1 <= len(frames) <= 900:
        raise ValueError("offline tissue runner accepts 1–900 frames")
    fps = float(animation["fps"])
    times = np.asarray([f["t"] for f in frames], np.float64)
    if len(times) > 1 and np.max(np.abs(np.diff(times) - 1. / fps)) > .002:
        raise ValueError("tissue take must have evenly spaced frames")
    order = list(rigdef.CONTROLS)
    values = np.array([[frame["v"].get(name, 0.) for name in order] for frame in frames], np.float32)
    if not np.isfinite(values).all():
        raise ValueError("non-finite animation controls")
    native_dir = ROOT / "tmp/vhuman-rig/native"
    lib = native_dir / "libvhuman_deformer.so"
    if not lib.is_file():
        lib = native.build_library(native_dir)
    if progress:
        progress(.15, "evaluating rig frames")
    evaluator = native.Native(lib, rig_dir / "rig_deformer.safetensors")
    try:
        rest = np.load(rig_dir / "fit.npz")["positions"].astype(np.float32)
        if evaluator.V != len(rest):
            raise ValueError("rig package and fitted mesh have different vertex counts")
        positions = evaluator.eval_batch(values)
    finally:
        evaluator.close()
    tris = _skin_tris(rig_dir)
    feat = json.loads((rig_dir / "features.json").read_text())
    ids, faces = _patch(rest, tris, feat)
    ids, faces, culled = _motion_safe_patch(rest, positions, ids, faces)
    stage = take_dir / "soft_tissue.vhsoft.partial"
    output = take_dir / "soft_tissue.usda"
    partial_usd = take_dir / "soft_tissue.usda.partial"
    if progress:
        progress(.4, "simulating facial tissue in LightGeom")
    try:
        _write_input(stage, rest[ids], faces, positions[:, ids], fps)
        proc = subprocess.Popen([str(runner), str(stage), str(partial_usd)], cwd=ROOT,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            while proc.poll() is None:
                if cancel is not None and cancel.is_set():
                    proc.terminate()
                    raise RuntimeError("soft-tissue simulation cancelled")
                import time
                time.sleep(.1)
            stdout, stderr = proc.communicate()
            if proc.returncode:
                raise RuntimeError("LightGeom facial simulation failed: " + stderr[-1200:])
            stats = json.loads(stdout.strip().splitlines()[-1])
            if stats["min_volume_ratio"] <= 0:
                raise ValueError("LightGeom reported an inverted facial tetrahedron")
            partial_usd.replace(output)
        finally:
            if proc.poll() is None:
                proc.terminate()
                proc.wait()
    finally:
        stage.unlink(missing_ok=True)
        partial_usd.unlink(missing_ok=True)
    rig_bytes = (rig_dir / "rig.json").read_bytes()
    report = {"format": "vhuman.soft_tissue.v1", "solver": "LightGeom Neo-Hookean tetra",
              "rig_sha256": hashlib.sha256(rig_bytes).hexdigest()[:16],
              "surface_vertices": int(len(ids)), "surface_triangles": int(len(faces)),
              "motion_culled_triangles": culled,
              "fps": fps, "seconds": len(frames) / fps, "thickness_m": .005,
              "stats": stats, "usd": output.name}
    (take_dir / "soft_tissue_report.json").write_text(json.dumps(report, indent=2))
    if progress:
        progress(.99, "soft tissue ready")
    return report


def soft_tissue_job(service, request: dict, progress, cancel, *, python=None,
                    runner: Path = DEFAULT_RUNNER) -> dict:
    from .job import DEFAULT_PYTHON
    head_id, take_id = request.get("head_id"), request.get("take_id")
    rig_dir = service.rig_file(head_id, "rig.json").parent
    take_dir = service.take_file(head_id, take_id, "animation.json").parent
    py = Path(python) if python else DEFAULT_PYTHON
    state = availability(runner, py)
    if not state["available"]:
        raise ValueError(state["reason"])
    cmd = [str(py), "-m", "server.vhuman.rig.soft_tissue", str(rig_dir),
           str(take_dir), "--runner", str(runner)]
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, start_new_session=True,
                            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    lines = []
    while proc.poll() is None:
        if cancel.is_set():
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            proc.communicate(timeout=5)
            raise RuntimeError("soft-tissue simulation cancelled")
        import time
        time.sleep(.1)
    output = proc.stdout.read() if proc.stdout else ""
    if proc.returncode:
        raise RuntimeError("soft-tissue job failed: " + output[-1200:])
    report = json.loads((take_dir / "soft_tissue_report.json").read_text())
    progress(.99, "soft tissue ready")
    return {"head_id": head_id, "take_id": take_id, "report": report,
            "usd_url": f"/v1/heads/{head_id}/rig/takes/{take_id}/soft_tissue.usda"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("rig_dir", type=Path)
    ap.add_argument("take_dir", type=Path)
    ap.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    args = ap.parse_args(argv)
    print(json.dumps(simulate(args.rig_dir, args.take_dir, args.runner)))


if __name__ == "__main__":
    main()
