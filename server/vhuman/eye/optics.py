"""Analytic two-sphere eye geometry, angular atlas mapping and standard optics.

Synthetic defaults: sclera radius 12 mm, limbus radius 6 mm, cornea radius
8 mm. These are demonstration choices, not clinical measurements. Users may
override all radii in metres. Frame: +Z gaze, +Y up, +X texture-right."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import params as P

CORNEA_SIZE_REF = 0.20       # synthetic packing choice: limbus UV radius
IRIS_TEX_SCALE = 0.5         # iris texture: uv = 0.5 + 0.5 * t * (cos phi, -sin phi); independent centred polar packing
BACK_UV_RADIUS = 0.70        # the back pole maps to this UV radius (never visible)
F0_EYE = ((1.336 - 1.0) / (1.336 + 1.0)) ** 2  # normal-incidence dielectric reflectance
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


@dataclass(frozen=True)
class Profile:
    sclera_radius: float = 0.012
    limbus_radius: float = 0.006
    cornea_radius: float = 0.008
    limbus_blend: float = 0.0005        # smooth-max width of the sclera/cornea junction

    @property
    def z_limbus(self) -> float:
        return math.sqrt(self.sclera_radius ** 2 - self.limbus_radius ** 2)

    @property
    def cornea_center_z(self) -> float:
        return self.z_limbus - math.sqrt(self.cornea_radius ** 2 - self.limbus_radius ** 2)

    @property
    def apex_z(self) -> float:
        return self.cornea_center_z + self.cornea_radius

    @property
    def alpha_limbus(self) -> float:
        return math.asin(self.limbus_radius / self.sclera_radius)

    def iris_z(self, chamber_depth: float) -> float:
        return self.apex_z - chamber_depth

    def derived(self) -> dict:
        return {"z_limbus": self.z_limbus, "cornea_center_z": self.cornea_center_z, "apex_z": self.apex_z,
                "alpha_limbus_deg": math.degrees(self.alpha_limbus),
                "cornea_sag": self.apex_z - self.z_limbus, "axial_length": self.apex_z + self.sclera_radius}


ANATOMICAL = Profile()


def profile_from_params(p: dict) -> Profile:
    """Physical measurements supplied through the public JSON schema."""
    o = P.validate(p)["optics"]
    return Profile(**{k: o[k] for k in ("sclera_radius", "limbus_radius", "cornea_radius", "limbus_blend")})


# ---- independent angle-to-radius texture layout -------------------------------

def uv_radius(alpha, profile: Profile = ANATOMICAL):
    """Two linear angular intervals, joined at the limbus.

    The front cap occupies radius .2, the remaining sphere extends to .7.
    These are synthetic atlas packing choices, not sampled asset coordinates.
    """
    a = np.asarray(alpha, np.float64)
    front = CORNEA_SIZE_REF * a / profile.alpha_limbus
    back = CORNEA_SIZE_REF + (BACK_UV_RADIUS - CORNEA_SIZE_REF) * (a - profile.alpha_limbus) / (math.pi - profile.alpha_limbus)
    return np.where(a <= profile.alpha_limbus, front, back)


def alpha_from_uv(r_uv, profile: Profile = ANATOMICAL):
    r = np.asarray(r_uv, np.float64)
    front = profile.alpha_limbus * r / CORNEA_SIZE_REF
    back = profile.alpha_limbus + (math.pi - profile.alpha_limbus) * (r - CORNEA_SIZE_REF) / (BACK_UV_RADIUS - CORNEA_SIZE_REF)
    return np.where(r <= CORNEA_SIZE_REF, front, back)


def eyeball_uv(direction: np.ndarray, profile: Profile = ANATOMICAL) -> np.ndarray:
    """Unit directions from the eye centre (..., 3) -> glTF UV (v down)."""
    d = np.asarray(direction, np.float64)
    alpha = np.arccos(np.clip(d[..., 2], -1.0, 1.0))
    phi = np.arctan2(d[..., 1], d[..., 0])
    r = uv_radius(alpha, profile)
    return np.stack([0.5 + r * np.cos(phi), 0.5 - r * np.sin(phi)], -1)


# ---- outer surface -------------------------------------------------------------

def smax(a, b, k):
    """Polynomial smooth maximum (union of the sclera and cornea balls)."""
    h = np.maximum(k - np.abs(a - b), 0.0) / k
    return np.maximum(a, b) + h * h * k * 0.25


def cornea_distance(alpha, profile: Profile = ANATOMICAL):
    """Distance from the eye centre to the cornea sphere along angle alpha."""
    zc, rc = profile.cornea_center_z, profile.cornea_radius
    s = np.sin(alpha)
    return zc * np.cos(alpha) + np.sqrt(np.maximum(rc * rc - (zc * s) ** 2, 0.0))


def surface_radius(alpha, profile: Profile = ANATOMICAL):
    return smax(np.full_like(np.asarray(alpha, np.float64), profile.sclera_radius),
                cornea_distance(alpha, profile), profile.limbus_blend)


# ---- optics --------------------------------------------------------------------

def normalize(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-20)


def refract(d, n, eta):
    """GLSL refract(): d incident (unit), n facing against d, eta = n1/n2.
    Total internal reflection returns zeros."""
    cos_i = -np.sum(d * n, -1, keepdims=True)
    k = 1.0 - eta * eta * (1.0 - cos_i * cos_i)
    out = eta * d + (eta * cos_i - np.sqrt(np.maximum(k, 0.0))) * n
    return np.where(k < 0.0, 0.0, out)


def fresnel_schlick(cos_theta, f0=F0_EYE):
    return f0 + (1.0 - f0) * (1.0 - np.clip(cos_theta, 0.0, 1.0)) ** 5


def iris_radius(p: dict, profile: Profile | None = None) -> float:
    """Physical iris radius: Cornea Size scales the iris relative to the
    geometric limbus (the reference UV radius puts the iris edge on it)."""
    profile = profile or profile_from_params(p)
    return profile.limbus_radius * p["cornea"]["size"] / CORNEA_SIZE_REF


def pupil_scale(t, scale):
    """Piecewise-linear pupil/iris warp with fixed centre and outer rim."""
    t = np.asarray(t, np.float64)
    radius = np.clip(P.P_REF * scale, .05, .85)
    inner = t * P.P_REF / radius
    outer = P.P_REF + (1.0 - P.P_REF) * (t - radius) / (1.0 - radius)
    return np.where(t <= radius, inner, outer)


def circ_mask(length, size, center, soft):
    """Soft radial mask: 1 inside radius (size - soft*center), falling to 0
    over `soft` (smoothstep)."""
    x = np.clip(1.0 - (np.asarray(length, np.float64) - (size - soft * center)) / max(soft, 1e-6), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def iris_irradiance(iris_n, light):
    """Lambertian iris irradiance; no fitted caustic enhancement."""
    return np.clip(np.sum(iris_n * light, -1), 0.0, 1.0)


# ---- studio environment (identical in GLSL) ------------------------------------

@dataclass(frozen=True)
class Softbox:
    direction: tuple        # towards the light
    size: tuple             # half extents, radians (horizontal, vertical)
    radiance: float


STUDIO = (Softbox((-0.45, 0.42, 0.79), (0.22, 0.16), 9.0),     # key, upper left of the camera
          Softbox((0.62, 0.10, 0.78), (0.10, 0.22), 2.2))      # fill, right
SKY, GROUND, HORIZON = (0.42, 0.45, 0.50), (0.08, 0.075, 0.07), (0.30, 0.30, 0.31)


def _box_frame(box: Softbox):
    w = normalize(np.array(box.direction, np.float64))
    up = np.array([0.0, 1.0, 0.0])
    u = normalize(np.cross(up, w))
    v = np.cross(w, u)
    return u, v, w


def environment(direction, roughness=0.0):
    """Radiance of the studio: sky/ground gradient plus two rectangular
    softboxes whose edges soften with roughness (a cheap prefilter)."""
    d = normalize(np.asarray(direction, np.float64))
    y = d[..., 1:2]
    sky = np.where(y >= 0, np.array(HORIZON) + (np.array(SKY) - HORIZON) * np.sqrt(np.clip(y, 0, 1)),
                   np.array(HORIZON) + (np.array(GROUND) - HORIZON) * np.sqrt(np.clip(-y, 0, 1)))
    out = sky.copy()
    blur = 0.01 + 0.6 * float(roughness) ** 2
    for box in STUDIO:
        u, v, w = _box_frame(box)
        dw = np.sum(d * w, -1)
        x = np.arctan2(np.sum(d * u, -1), dw)
        z = np.arctan2(np.sum(d * v, -1), dw)
        mx = np.clip((box.size[0] - np.abs(x)) / blur + 0.5, 0, 1)
        mz = np.clip((box.size[1] - np.abs(z)) / blur + 0.5, 0, 1)
        spread = (box.size[0] * box.size[1]) / ((box.size[0] + blur) * (box.size[1] + blur))
        out = out + (box.radiance * spread * mx * mz * (dw > 0))[..., None]
    return out


def studio_lights():
    """The softboxes as directional lights for diffuse: (direction, irradiance)."""
    lights = []
    for box in STUDIO:
        solid_angle = 4.0 * box.size[0] * box.size[1]
        lights.append((normalize(np.array(box.direction, np.float64)), box.radiance * solid_angle))
    return lights


def ambient(normal):
    """Diffuse irradiance / pi of the gradient (cosine-weighted approximation)."""
    y = np.asarray(normal, np.float64)[..., 1:2]
    t = 0.5 + 0.5 * y
    return np.array(GROUND) + (np.array(SKY) - GROUND) * t


def tonemap_neutral(color, exposure=1.0):
    """Khronos PBR Neutral (Three.js adaptation; THIRD_PARTY_NOTICES.md), then sRGB."""
    c = np.asarray(color, np.float64) * exposure
    start, desat = 0.8 - 0.04, 0.15
    x = np.min(c, -1, keepdims=True)
    offset = np.where(x < 0.08, x - 6.25 * x * x, 0.04)
    c = c - offset
    peak = np.max(c, -1, keepdims=True)
    d = 1.0 - start
    new_peak = 1.0 - d * d / (peak + d - start)
    scaled = c * new_peak / np.maximum(peak, 1e-9)
    g = 1.0 - 1.0 / (desat * (peak - new_peak) + 1.0)
    mapped = scaled * (1 - g) + new_peak * g
    c = np.where(peak < start, c, mapped)
    return linear_to_srgb(np.clip(c, 0.0, 1.0))


def linear_to_srgb(c):
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def srgb_to_linear(c):
    c = np.asarray(c, np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


# ---- shader uniforms -------------------------------------------------------------

def uniforms(p: dict, profile: Profile | None = None) -> dict:
    """Geometry and optics numbers the web shader uses (no hard-coded JS)."""
    p = P.validate(p)
    profile = profile or profile_from_params(p)
    o = p["optics"]
    return {
        "scleraRadius": profile.sclera_radius, "limbusRadius": profile.limbus_radius,
        "corneaRadius": profile.cornea_radius, "corneaCenterZ": profile.cornea_center_z,
        "apexZ": profile.apex_z, "zLimbus": profile.z_limbus, "alphaLimbus": profile.alpha_limbus,
        "irisZ": profile.iris_z(o["chamber_depth"]), "irisRadius": iris_radius(p, profile),
        "irisConvexity": o["iris_convexity"], "ior": o["ior"],
        "pupilRatio": P.pupil_ratio(p), "pupilScale": P.pupil_scale(p), "pRef": P.P_REF,
        "irisTexScale": IRIS_TEX_SCALE,
        "corneaSizeRef": CORNEA_SIZE_REF, "backUvRadius": BACK_UV_RADIUS, "f0": ((o["ior"] - 1) / (o["ior"] + 1)) ** 2,
        "uvMapping": "angular-two-segment-v1",
        "softboxes": [{"direction": list(normalize(np.array(b.direction))), "size": list(b.size),
                       "radiance": b.radiance} for b in STUDIO],
        "sky": SKY, "ground": GROUND, "horizon": HORIZON,
    }
