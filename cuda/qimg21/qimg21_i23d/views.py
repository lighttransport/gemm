"""Requested camera views for multi-view generation.

A ViewSpec is what we *ask* the model for. Qwen-Image 2.1 is a generative image
model, not a calibrated renderer: nothing guarantees a generated image was
taken from the requested azimuth, elevation, distance or field of view. These
values are exported as requested-view metadata so a downstream reconstruction
system can decide how far to trust them; they are never camera calibration.

Conventions follow the NeRF/Blender `transforms.json` that Pixal3D reads
(cpu/pixal3d, ref/pixal3d/upstream/assets/mv_images/example): world Z-up,
camera-to-world columns (right, up, back), azimuth 0 = the reference (front)
view with the camera on -Y looking at +Y, azimuth increasing towards +X
(counter-clockwise seen from above), elevation positive above the object.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, field, replace

# Pixal3D's multiview example: 20 degree horizontal FOV at distance 3.119 for an
# object normalized to the unit cube. Used as the default requested framing.
DEFAULT_FOV_DEG = math.degrees(0.3490658503988659)
DEFAULT_DISTANCE = 3.119

PROJECTIONS = ("perspective", "orthographic")


class ViewSpecError(ValueError):
    """An impossible or unsupported view request, with the reason."""


@dataclass(frozen=True)
class ViewSpec:
    azimuth_deg: float
    elevation_deg: float = 0.0
    roll_deg: float = 0.0
    distance: float = DEFAULT_DISTANCE
    fov_deg: float = DEFAULT_FOV_DEG
    width: int = 512
    height: int = 512
    projection: str = "perspective"
    # Free-form extras recorded in the metadata (e.g. "ring": 1).
    tags: dict = field(default_factory=dict, compare=False, hash=False)

    def validated(self) -> "ViewSpec":
        """This spec with azimuth wrapped to [0, 360), or ViewSpecError."""
        for name in ("azimuth_deg", "elevation_deg", "roll_deg", "distance", "fov_deg"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ViewSpecError(f"{name} must be a finite number, got {value!r}")
        if not -90.0 <= self.elevation_deg <= 90.0:
            raise ViewSpecError(f"elevation_deg must be in [-90, 90], got {self.elevation_deg}")
        if not -180.0 <= self.roll_deg <= 180.0:
            raise ViewSpecError(f"roll_deg must be in [-180, 180], got {self.roll_deg}")
        if self.distance <= 0:
            raise ViewSpecError(f"distance must be positive, got {self.distance}")
        if self.projection not in PROJECTIONS:
            raise ViewSpecError(f"projection must be one of {', '.join(PROJECTIONS)}, got {self.projection!r}")
        if self.projection == "perspective" and not 1.0 <= self.fov_deg <= 170.0:
            raise ViewSpecError(f"fov_deg must be in [1, 170] for a perspective view, got {self.fov_deg}")
        for name in ("width", "height"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 256 or value > 2048 or value % 32:
                raise ViewSpecError(f"{name} must be a multiple of 32 in [256, 2048], got {value!r}")
        return replace(self, azimuth_deg=self.azimuth_deg % 360.0)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["fov_rad"] = math.radians(self.fov_deg)
        return data


def _angle_label(value: float, width: int = 3) -> str:
    """-20 -> "-20", 0 -> "000", 30 -> "030", 12.5 -> "012.5"."""
    if float(value).is_integer():
        return f"{int(value):0{width}d}"
    return f"{value:0{width + 2}.1f}"


def view_name(spec: ViewSpec) -> str:
    return f"e{_angle_label(spec.elevation_deg)}_a{_angle_label(spec.azimuth_deg)}"


def ring_relpath(spec: ViewSpec, ext: str = "png") -> str:
    """views/e-20/a030.png style path for elevation-ring datasets."""
    return f"e{_angle_label(spec.elevation_deg)}/a{_angle_label(spec.azimuth_deg)}.{ext}"


def ring(count: int, elevation_deg: float = 0.0, *, start_deg: float = 0.0, **common) -> list[ViewSpec]:
    """`count` views evenly spaced in azimuth, starting at `start_deg`."""
    if not isinstance(count, int) or count < 1:
        raise ViewSpecError(f"view count must be a positive integer, got {count!r}")
    step = 360.0 / count
    return [ViewSpec(azimuth_deg=start_deg + i * step, elevation_deg=elevation_deg, **common).validated()
            for i in range(count)]


def rings(azimuth_views: int, elevations: list[float], **common) -> list[ViewSpec]:
    """One ring per elevation, ordered elevation-major then azimuth."""
    if not elevations:
        raise ViewSpecError("at least one elevation is required")
    if len(set(elevations)) != len(elevations):
        raise ViewSpecError(f"elevations repeat: {elevations}")
    specs = []
    for index, elevation in enumerate(elevations):
        for spec in ring(azimuth_views, elevation, **common):
            specs.append(replace(spec, tags={"ring": index}))
    return specs


def turntable(count: int, elevation_deg: float = 0.0, **common) -> list[ViewSpec]:
    return ring(count, elevation_deg, **common)


def top_bottom(top: bool = True, bottom: bool = True, **common) -> list[ViewSpec]:
    specs = []
    if top:
        specs.append(ViewSpec(azimuth_deg=0.0, elevation_deg=90.0, **common).validated())
    if bottom:
        specs.append(ViewSpec(azimuth_deg=0.0, elevation_deg=-90.0, **common).validated())
    return specs


def parse_elevations(text: str) -> list[float]:
    try:
        values = [float(part) for part in text.split(",") if part.strip()]
    except ValueError:
        raise ViewSpecError(f"elevations must be comma-separated numbers, got {text!r}") from None
    if not values:
        raise ViewSpecError("no elevations given")
    return values


# ---- prompts -----------------------------------------------------------

def azimuth_words(azimuth_deg: float) -> str:
    """Which side of the object faces the camera, relative to the reference.

    The camera orbits counter-clockwise seen from above; the reference image is
    the front. A camera on +X sees the object's left side (the object faces the
    reference camera, so its left is at the viewer's right)."""
    names = ["front view", "front-left three-quarter view", "left side view",
             "rear-left three-quarter view", "rear view", "rear-right three-quarter view",
             "right side view", "front-right three-quarter view"]
    return names[int(((azimuth_deg % 360.0) + 22.5) // 45.0) % 8]


def elevation_words(elevation_deg: float) -> str:
    if elevation_deg >= 75:
        return "top-down view, camera directly above the object"
    if elevation_deg >= 30:
        return "high camera looking down at the object"
    if elevation_deg >= 10:
        return "slightly elevated camera"
    if elevation_deg > -10:
        return "camera at the object's eye level"
    if elevation_deg > -75:
        return "low camera looking up at the object"
    return "bottom view, camera directly below the object"


BACKGROUND_CLAUSES = {
    "transparent": ("This is an RGBA image with transparency. The image has alpha channel and the "
                    "background is transparent."),
    "white": "Place the object on a plain, uniform white background.",
    "gray": "Place the object on a plain, uniform neutral gray background.",
}

DEFAULT_VIEW_TEMPLATE = (
    "Generate the same object as the reference image. Preserve its identity, geometry, proportions, "
    "materials, colors, surface details, and markings. Change only the camera viewpoint: {view_words}, "
    "{elevation_words}. Camera azimuth: {azimuth:g} degrees from the reference view. Camera elevation: "
    "{elevation:g} degrees.{roll_clause}{projection_clause} Show the complete object, centered and fully "
    "inside the frame, at the same scale as the reference. Do not introduce new objects. Do not remove "
    "existing parts. Use neutral studio lighting. {background_clause}")


def view_prompt(spec: ViewSpec, *, background: str = "transparent", template: str | None = None,
                instruction: str | None = None) -> str:
    """The text for one requested view. Deterministic: same spec, same text."""
    if background not in BACKGROUND_CLAUSES:
        raise ViewSpecError(f"background must be one of {', '.join(BACKGROUND_CLAUSES)}, got {background!r}")
    roll = f" Camera roll: {spec.roll_deg:g} degrees." if spec.roll_deg else ""
    projection = (" Use an orthographic projection with no perspective distortion."
                  if spec.projection == "orthographic" else "")
    text = (template or DEFAULT_VIEW_TEMPLATE).format(
        view_words=azimuth_words(spec.azimuth_deg), elevation_words=elevation_words(spec.elevation_deg),
        azimuth=round(spec.azimuth_deg, 3), elevation=round(spec.elevation_deg, 3), roll_clause=roll,
        projection_clause=projection, background_clause=BACKGROUND_CLAUSES[background])
    if instruction:
        text = f"{text} {instruction.strip()}"
    return " ".join(text.split())


SEED_MODES = ("shared", "per_view")


def derive_seed(seed: int, spec: ViewSpec, mode: str = "shared") -> int:
    """shared: every view starts from the same noise, which helps the views
    agree; per_view: a stable seed per camera, independent of the view's
    position in the list or the batch it runs in."""
    if mode not in SEED_MODES:
        raise ViewSpecError(f"seed mode must be one of {', '.join(SEED_MODES)}, got {mode!r}")
    if not isinstance(seed, int) or seed < 0:
        raise ViewSpecError(f"seed must be a non-negative integer, got {seed!r}")
    if mode == "shared":
        return seed
    key = f"{seed}:{spec.azimuth_deg:.6f}:{spec.elevation_deg:.6f}:{spec.roll_deg:.6f}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "little") & 0x7FFFFFFF


# ---- camera matrices (requested, not calibrated) ------------------------

def camera_position(spec: ViewSpec) -> tuple[float, float, float]:
    az, el = math.radians(spec.azimuth_deg), math.radians(spec.elevation_deg)
    return (spec.distance * math.cos(el) * math.sin(az),
            -spec.distance * math.cos(el) * math.cos(az),
            spec.distance * math.sin(el))


def transform_matrix(spec: ViewSpec) -> list[list[float]]:
    """Requested camera-to-world matrix in the NeRF/Blender convention."""
    az, roll = math.radians(spec.azimuth_deg), math.radians(spec.roll_deg)
    position = camera_position(spec)
    norm = math.sqrt(sum(c * c for c in position))
    back = [c / norm for c in position]
    # Horizontal "right" is defined by azimuth alone, so it stays well defined
    # straight above or below the object.
    right = [math.cos(az), math.sin(az), 0.0]
    up = [back[1] * right[2] - back[2] * right[1],
          back[2] * right[0] - back[0] * right[2],
          back[0] * right[1] - back[1] * right[0]]
    if roll:
        c, s = math.cos(roll), math.sin(roll)
        right, up = ([c * r + s * u for r, u in zip(right, up)],
                     [-s * r + c * u for r, u in zip(right, up)])
    rows = [[right[i], up[i], back[i], position[i]] for i in range(3)]
    return [[round(v, 9) + 0.0 for v in row] for row in rows] + [[0.0, 0.0, 0.0, 1.0]]
