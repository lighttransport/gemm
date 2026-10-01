"""Persistent, versioned Gaussian binding with bounded numerical invariants."""
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
from ....reconstruction.fitting import topology_hash
from ....reconstruction.gaussian import frames
from .provenance import validate_receipts

FORMAT = "vhuman.gaussian_avatar.v1"
MAX_GAUSSIANS = 200000


@dataclass
class GaussianAvatar:
    metadata: dict
    arrays: dict

    def validate(self, triangles=None):
        m, a = self.metadata, self.arrays
        if m.get("format") != FORMAT or m.get("units") != "metres":
            raise ValueError("unsupported Gaussian avatar")
        if m.get("covariance_policy", "eigen-v1") not in ("eigen-v1", "trace-v1"):
            raise ValueError("unknown covariance policy")
        digest = m.get("topology_sha256", "")
        if len(digest) != 64 or (triangles is not None and digest != topology_hash(triangles)):
            raise ValueError("Gaussian topology mismatch")
        names = m.get("control_names", [])
        if not names or len(names) != len(set(names)):
            raise ValueError("invalid control names")
        n = len(a.get("triangle", []))
        if not 1 <= n <= MAX_GAUSSIANS:
            raise ValueError("Gaussian count outside 1..200000")
        shapes = dict(triangle=(n,), component=(n,), barycentric=(n, 3), normal_offset=(n,),
                      covariance_local=(n, 3, 3), opacity=(n,), rgb=(n, 3),
                      color_basis=(n, 8, 3), expression_matrix=(len(names), 8))
        for key, shape in shapes.items():
            if key not in a or a[key].shape != shape or not np.isfinite(a[key]).all():
                raise ValueError("invalid Gaussian array " + key)
        for key in ("triangle", "component"):
            if a[key].dtype.kind not in "iu" or (a[key] < 0).any():
                raise ValueError("invalid integer attachment")
        if triangles is not None and (a["triangle"] >= len(triangles)).any():
            raise ValueError("attachment triangle out of bounds")
        b, c = a["barycentric"], a["covariance_local"]
        if (b < 0).any() or not np.allclose(b.sum(1), 1, atol=1e-5):
            raise ValueError("invalid barycentrics")
        if not np.allclose(c, c.transpose(0, 2, 1), atol=1e-8) or (np.linalg.eigvalsh(c) <= 0).any():
            raise ValueError("covariance must be SPD")
        if (abs(a["normal_offset"]) > .005).any() or (a["rgb"] < 0).any():
            raise ValueError("invalid radiance/offset")
        if (a["opacity"] < 0).any() or (a["opacity"] > 1).any():
            raise ValueError("invalid opacity")
        if m.get("purpose") == "production":
            validate_receipts(m.get("provenance"))
        elif m.get("purpose") != "diagnostic":
            raise ValueError("purpose must be production or diagnostic")
        return self

    def save(self, path):
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".partial")
        with partial.open("wb") as f:
            np.savez_compressed(f, metadata=np.array(json.dumps(self.metadata)), **self.arrays)
        partial.replace(path)

    @classmethod
    def load(cls, path, triangles=None):
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"]))
            arrays = {k: archive[k].copy() for k in archive.files if k != "metadata"}
        return cls(metadata, arrays).validate(triangles)


def bind(vertices, triangles, names, count=50000, seed=7, purpose="diagnostic", provenance=None):
    """Area-uniform seeds; fitting is required before photographic-quality claims."""
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= MAX_GAUSSIANS:
        raise ValueError("invalid Gaussian count")
    v, t = np.asarray(vertices, np.float32), np.asarray(triangles)
    if v.ndim != 2 or v.shape[1] != 3 or t.ndim != 2 or t.shape[1] != 3 or t.dtype.kind not in "iu":
        raise ValueError("invalid mesh")
    if not np.isfinite(v).all() or (t < 0).any() or (t >= len(v)).any():
        raise ValueError("invalid mesh coordinates/indices")
    basis, valid = frames(v, t)
    ids = np.flatnonzero(valid)
    if not len(ids):
        raise ValueError("mesh has no nondegenerate triangles")
    area = np.linalg.norm(np.cross(basis[ids, :, 0], basis[ids, :, 1]), axis=1)
    rng = np.random.default_rng(seed)
    selected = rng.choice(ids, count, p=area / area.sum())
    r = rng.random((count, 2))
    root = np.sqrt(r[:, 0])
    bary = np.stack((1-root, root*(1-r[:, 1]), root*r[:, 1]), axis=1).astype(np.float32)
    a = dict(triangle=selected.astype(np.int32), component=np.zeros(count, np.int32), barycentric=bary,
             normal_offset=np.zeros(count, np.float32),
             covariance_local=np.tile(np.diag([.035**2, .035**2, .00015**2]), (count, 1, 1)).astype(np.float32),
             opacity=np.full(count, .25, np.float32), rgb=np.full((count, 3), .4, np.float32),
             color_basis=np.zeros((count, 8, 3), np.float32), expression_matrix=np.zeros((len(names), 8), np.float32))
    m = dict(format=FORMAT, units="metres", topology_sha256=topology_hash(t), control_names=list(names),
             radiance="fixed-light linear RGB", purpose=purpose, provenance=provenance or [],
             seed=seed, supported_yaw_degrees=[-15, 15], trained=False)
    return GaussianAvatar(m, a).validate(t)


def import_binding(binding, names):
    """Explicit compatibility conversion, retaining research/diagnostic purpose."""
    n = len(binding["triangle"])
    arrays = {k: np.array(binding[k], copy=True) for k in
              ("triangle", "barycentric", "normal_offset", "covariance_local", "opacity", "rgb")}
    arrays["triangle"] = arrays["triangle"].astype(np.int32)
    arrays.update(component=np.zeros(n, np.int32), color_basis=np.zeros((n, 8, 3), np.float32),
                  expression_matrix=np.zeros((len(names), 8), np.float32))
    metadata = dict(format=FORMAT, units="metres", topology_sha256=binding["topology_sha256"],
                    control_names=list(names), radiance="fixed-light linear RGB", purpose="diagnostic",
                    provenance=[], trained=False, imported_from="vhuman.gaussian_binding.v1")
    return GaussianAvatar(metadata, arrays).validate()


def deform(avatar, vertices, triangles, controls=None):
    vertices, triangles = np.asarray(vertices), np.asarray(triangles)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("invalid posed vertices")
    if triangles.ndim != 2 or triangles.shape[1] != 3 or triangles.dtype.kind not in "iu" or (triangles < 0).any() or (triangles >= len(vertices)).any():
        raise ValueError("invalid triangles")
    avatar.validate(triangles)
    a = avatar.arrays
    basis, valid = frames(np.asarray(vertices), np.asarray(triangles))
    ids = a["triangle"]
    centers = (vertices[triangles[ids]] * a["barycentric"][..., None]).sum(1)
    centers += basis[ids, :, 2] * a["normal_offset"][:, None]
    cov = basis[ids] @ a["covariance_local"] @ basis[ids].transpose(0, 2, 1)
    policy = avatar.metadata.get("covariance_policy", "eigen-v1")
    if policy == "trace-v1":
        cov = cov + np.eye(3, dtype=cov.dtype) * 1e-10
        trace = np.trace(cov, axis1=-2, axis2=-1)
        cov *= np.minimum(1, 1e-4 / np.maximum(trace, 1e-10))[:, None, None]
    else:
        eig, rot = np.linalg.eigh(cov)
        cov = (rot * np.clip(eig, 1e-10, .01**2)[:, None, :]) @ rot.transpose(0, 2, 1)
    rgb = a["rgb"].copy()
    if controls is not None:
        controls = np.asarray(controls)
        if controls.shape != (len(avatar.metadata["control_names"]),) or not np.isfinite(controls).all():
            raise ValueError("invalid controls")
        coeff = np.tanh(controls @ a["expression_matrix"])
        rgb = np.maximum(0, rgb + np.einsum("nkr,k->nr", a["color_basis"], coeff))
    if policy == "trace-v1": rgb = np.clip(rgb, 0, 1)
    return centers, cov, a["opacity"] * valid[ids], rgb
