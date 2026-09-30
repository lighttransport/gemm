"""Small NumPy correctness renderer. H frame is metres, camera looks -Z.

The z buffer uses perspective-correct attributes and pixel centres. This is
an offline reference, not a differentiable production rasterizer.
"""
from dataclasses import dataclass
import numpy as np


def srgb_to_linear(x):
    x = np.asarray(x, dtype=np.float64)
    return np.where(x <= .04045, x / 12.92, ((x + .055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    x = np.maximum(np.asarray(x, dtype=np.float64), 0)
    return np.where(x <= .0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - .055)


@dataclass
class Camera:
    focal: float
    cx: float
    cy: float
    origin: np.ndarray
    rotation: np.ndarray
    focal_y: float | None = None
    skew: float = 0.

    def scaled(self, scale):
        return Camera(self.focal*scale, self.cx*scale, self.cy*scale,
                      self.origin, self.rotation,
                      self.focal_y*scale if self.focal_y is not None else None,
                      self.skew*scale)

    def project(self, points):
        p = (np.asarray(points) - self.origin) @ self.rotation.T
        z = -p[..., 2]
        xy = np.stack(((self.focal*p[..., 0]-self.skew*p[..., 1]) / np.maximum(z, 1e-9) + self.cx,
                       -(self.focal if self.focal_y is None else self.focal_y) * p[..., 1] / np.maximum(z, 1e-9) + self.cy), -1)
        return xy, z

    def rays(self, xy):
        y = (xy[..., 1]-self.cy)/(self.focal if self.focal_y is None else self.focal_y)
        d = np.stack(((xy[..., 0]-self.cx-self.skew*y)/self.focal,
                      -y, -np.ones(xy.shape[:-1])), -1)
        d = d @ self.rotation
        return d / np.linalg.norm(d, axis=-1, keepdims=True)

    def as_dict(self):
        return dict(focal=self.focal, cx=self.cx, cy=self.cy,
                    origin=self.origin.tolist(), rotation=self.rotation.tolist(), units='metres',
                    focal_y=self.focal_y, skew=self.skew)

    @classmethod
    def from_dict(cls, d):
        c = cls(float(d['focal']), float(d['cx']), float(d['cy']),
                np.asarray(d['origin'], float), np.asarray(d['rotation'], float),
                float(d['focal_y']) if d.get('focal_y') is not None else None, float(d.get('skew',0)))
        if (c.focal <= 0 or c.origin.shape != (3,) or c.rotation.shape != (3, 3)
                or not np.isfinite([c.focal, c.cx, c.cy,c.skew]).all()
                or (c.focal_y is not None and (not np.isfinite(c.focal_y) or c.focal_y<=0))
                or not np.isfinite(c.origin).all() or not np.isfinite(c.rotation).all()
                or not np.allclose(c.rotation @ c.rotation.T, np.eye(3), atol=1e-5)
                or np.linalg.det(c.rotation) < .999):
            raise ValueError('invalid camera')
        return c


def pixal_camera(subject):
    from ..head.camera import PixalCamera
    fov = subject.fit.get('camera', {}).get('fov_deg', 20)
    p = PixalCamera.from_portrait(subject.portrait, np.deg2rad(fov))
    return Camera(p.focal / p.scale, (p.side / 2 + p.left) / p.scale,
                  (p.side / 2 + p.top) / p.scale, subject.frame.to_h(p.origin), np.eye(3))


def rasterize(vertices, triangles, camera, size):
    """Return triangle id, perspective barycentrics, and metric depth (H,W)."""
    w, h = size
    xy, z = camera.project(vertices)
    tid = np.full((h, w), -1, np.int32)
    depth = np.full((h, w), np.inf)
    bary = np.zeros((h, w, 3), np.float32)
    for i, tri in enumerate(triangles):
        q, zz = xy[tri], z[tri]
        if (zz <= 1e-5).any() or not np.isfinite(q).all():
            continue
        lo = np.maximum(np.floor(q.min(0)).astype(int), 0)
        hi = np.minimum(np.ceil(q.max(0)).astype(int), [w - 1, h - 1])
        if (lo > hi).any():
            continue
        a, b, c = q
        det = (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
        if abs(det) < 1e-9:
            continue
        yy, xx = np.mgrid[lo[1]:hi[1]+1, lo[0]:hi[0]+1]
        p = np.stack((xx + .5, yy + .5), -1) - a
        v = (p[..., 0] * (c[1]-a[1]) - p[..., 1] * (c[0]-a[0])) / det
        u = ((b[0]-a[0]) * p[..., 1] - (b[1]-a[1]) * p[..., 0]) / det
        screen = np.stack((1-v-u, v, u), -1)
        inv = screen / zz
        dz = 1 / np.maximum(inv.sum(-1), 1e-30)
        sl = np.s_[lo[1]:hi[1]+1, lo[0]:hi[0]+1]
        take = (screen >= -1e-7).all(-1) & (dz < depth[sl])
        depth[sl][take] = dz[take]
        tid[sl][take] = i
        bary[sl][take] = (inv * dz[..., None])[take]
    return tid, bary, depth


def ggx(albedo, normal, view, light, roughness=.55, f0=.028):
    """Linear Lambert + isotropic GGX BRDF times N.L; no tone transform."""
    def unit(x):
        x = np.asarray(x, float)
        return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)
    n, v, l = unit(normal), unit(view), unit(light)
    h = unit(v + l)
    nv = np.maximum(np.sum(n*v, -1), 1e-5)
    nl = np.maximum(np.sum(n*l, -1), 0)
    nh = np.maximum(np.sum(n*h, -1), 0)
    vh = np.maximum(np.sum(v*h, -1), 0)
    a2 = np.maximum(np.asarray(roughness), .04) ** 4
    d = a2 / (np.pi * (nh*nh*(a2-1)+1)**2)
    def smith(x):
        return 2*x / np.maximum(x + np.sqrt(a2+(1-a2)*x*x), 1e-9)
    fresnel = f0 + (1-f0)*(1-vh)**5
    spec = d*smith(nv)*smith(nl)*fresnel / np.maximum(4*nv*nl, 1e-8)
    diffuse = np.asarray(albedo) / np.pi * (1-fresnel)[..., None] * nl[..., None]
    return diffuse, spec[..., None]*nl[..., None]
