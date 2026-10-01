"""Read welded topology from existing vhuman rig exports, without rebuilding them."""
import json
from pathlib import Path
import numpy as np
from ....eye.glb import GLB
from ....rig import safetensors
from ....rig.native import Native, build_library


class RigAvatar:
    def __init__(self, directory, build_dir):
        directory = Path(directory)
        self.definition = json.loads((directory / "rig.json").read_text())
        self.names = tuple(c["name"] for c in self.definition["controls"])
        self.ranges = np.array([[c["min"], c["max"]] for c in self.definition["controls"]], np.float32)
        package = directory / "rig_deformer.safetensors"
        arrays, metadata = safetensors.load(package)
        if tuple(json.loads(metadata["controls"])) != self.names:
            raise ValueError("rig control order mismatch")
        self.rest = arrays["rest"].copy()
        self.native = Native(build_library(build_dir), package)
        try:
            glb = GLB.load(directory / "rig.glb")
            parts = json.loads((directory / "viz.json").read_text())["parts"]
            triangles, components = [], []
            for mesh in glb.doc["meshes"]:
                name = mesh["name"] if mesh["name"] in parts else "head_" + mesh["name"]
                if name not in parts:
                    continue  # carried eyes are rigid extras, not welded deformation vertices
                mapping = np.asarray(parts[name]["vmap"], np.int32)
                for prim in mesh["primitives"]:
                    local = glb.accessor(prim["indices"]).reshape(-1, 3).astype(np.int32)
                    triangles.append(mapping[local])
                    components.append(np.full(len(local), len(triangles) - 1, np.int32))
            if not triangles:
                raise ValueError("no welded rig topology in GLB/viz")
            self.triangles = np.concatenate(triangles)
            self.components = np.concatenate(components)
            if (self.triangles < 0).any() or (self.triangles >= len(self.rest)).any():
                raise ValueError("invalid rig welded mapping")
        except Exception:
            self.native.close()
            raise

    def deform(self, controls):
        x = np.asarray(controls, np.float32)
        if x.shape != (len(self.names),) or not np.isfinite(x).all():
            raise ValueError("invalid rig controls")
        return self.native.eval(np.clip(x, self.ranges[:, 0], self.ranges[:, 1]))

    def close(self):
        self.native.close()
