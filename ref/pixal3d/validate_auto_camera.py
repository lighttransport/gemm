"""Compare prepared camera metadata with the pinned upstream MoGe-2 helper."""
import argparse
import json
from pathlib import Path
import sys
import types

from upstream_import import ROOT, prepare


p = argparse.ArgumentParser()
p.add_argument("--image", type=Path, required=True, help="Cropped RGB camera input")
p.add_argument("--metadata", type=Path, required=True)
p.add_argument("--model", type=Path, default=Path("/mnt/disk2/models/moge-2-vitl/model.pt"))
a = p.parse_args()
prepare()
sys.modules["pixal3d.pipelines"].Pixal3DImageTo3DPipeline = object
sys.modules.setdefault("o_voxel", types.ModuleType("o_voxel"))
sys.path.insert(0, str(ROOT / "upstream"))
sys.path.insert(0, str(ROOT / "moge-upstream"))
import inference as upstream

expected = json.loads(a.metadata.read_text())
model = upstream.load_moge_model(device="cuda", model_name=str(a.model))
actual = upstream.get_camera_params_wild_moge(
    str(a.image), model, device="cuda", mesh_scale=expected["mesh_scale"])
fov_error = abs(actual["camera_angle_x"] - expected["fov"])
distance_error = abs(actual["distance"] - expected["distance"])
print(json.dumps({"expected": expected, "upstream": actual,
                  "fov_abs_error": fov_error, "distance_abs_error": distance_error}))
assert fov_error < 1e-7 and distance_error < 1e-6
print("Automatic camera upstream parity: PASS")
