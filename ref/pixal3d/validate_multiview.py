"""Validate native multiview relative-camera projection against a NumPy oracle."""
import argparse
import ctypes as C
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
p = argparse.ArgumentParser()
p.add_argument("--views-dir", type=Path, default=ROOT / "ref/pixal3d/upstream/assets/mv_images/example")
a = p.parse_args()

lib = C.CDLL(str(ROOT / "cpu/pixal3d/libpixal3d.so"))
lib.pixal3d_project_matrix.argtypes = [C.POINTER(C.c_int32), C.c_size_t, C.c_int, C.c_int,
                                      C.c_float, C.c_float, C.POINTER(C.c_float), C.POINTER(C.c_float)]
meta = json.loads((a.views_dir / "transforms.json").read_text())
frames = meta["frames"]
cameras = np.asarray([frame["transform_matrix"] for frame in frames], dtype=np.float32)
fovs = np.asarray([frame.get("camera_angle_x", meta["camera_angle_x"]) for frame in frames], dtype=np.float32)
scale = np.float32(meta.get("mesh_scale", 1.0))
distance = np.linalg.norm(cameras[0, :3, 3])
front = np.asarray([[1, 0, 0, 0], [0, 0, -1, -distance], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float32)
calc = np.asarray([front @ np.linalg.inv(cameras[0]) @ camera for camera in cameras], dtype=np.float32)

rng = np.random.default_rng(9)
coords = np.zeros((257, 4), dtype=np.int32)
coords[:, 1:] = rng.integers(0, 64, size=(len(coords), 3), dtype=np.int32)
one = np.linspace(-1, 1, 64, dtype=np.float32)
grid = np.stack([one[coords[:, 1]], -one[coords[:, 3]], one[coords[:, 2]]], axis=1) / (2 * scale)
homogeneous = np.concatenate([grid, np.ones((len(grid), 1), dtype=np.float32)], axis=1)

worst = 0.0
for view, matrix in enumerate(calc):
    actual = np.empty((len(coords), 2), dtype=np.float32)
    rc = lib.pixal3d_project_matrix(coords.ctypes.data_as(C.POINTER(C.c_int32)), len(coords), 64, 1024,
                                    float(fovs[view]), float(scale),
                                    matrix.ctypes.data_as(C.POINTER(C.c_float)),
                                    actual.ctypes.data_as(C.POINTER(C.c_float)))
    assert rc == 0
    camera = homogeneous @ np.linalg.inv(matrix).T
    focal = 512 / np.tan(fovs[view] / 2)
    pixels = np.stack([focal * camera[:, 0] / (-camera[:, 2] + 1e-8) + 512,
                       -focal * camera[:, 1] / (-camera[:, 2] + 1e-8) + 512], axis=1)
    expected = (pixels + .5) / 1024 * 2 - 1
    worst = max(worst, float(np.max(np.abs(actual - expected))))

assert worst < 2e-5, worst
print(json.dumps({"views": len(frames), "points_per_view": len(coords), "max_abs": worst}))
print("Multiview camera projection PASS")
