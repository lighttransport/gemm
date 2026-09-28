"""Independent procedural-eye controls and user-supplied measurements.

Defaults are synthetic demonstration choices, not measurements of a licensed
character or a clinical specification. JSON overrides use metres and linear RGB.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

ALGO_VERSION = 3          # independent analytic mapping and material model

BLEND_METHODS = ("Radial", "Structural")
IRIS_PATTERNS = tuple(f"pattern_{i}" for i in range(1, 10))
SIDES = ("left", "right")


@dataclass(frozen=True)
class FieldSpec:
    name: str
    kind: str
    default: Any
    lo: float | None = None
    hi: float | None = None
    live: bool = True
    label: str = ""
    choices: tuple = ()
    normalized_ui: bool = False
    help: str = ""


def _f(name, default, lo, hi, *, live=True, label="", norm=False, help=""):
    return FieldSpec(name, "float", default, lo, hi, live, label, (), norm, help)


def _c(name, default, *, live=True, label="", help=""):
    return FieldSpec(name, "color", tuple(default), 0.0, 1.0, live, label, (), False, help)


WHITE, BLACK = (1.0, 1.0, 1.0), (0.0, 0.0, 0.0)

# Dimensionless artist controls span their full useful ranges. Physical
# measurements are separately validated for a realizable two-sphere eye.
FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("iris.pattern", "enum", "pattern_1", live=False, label="Pattern", choices=IRIS_PATTERNS),
    _f("iris.rotation", 0., 0., 1.),
    _f("iris.primary_color_u", .5, 0., 1.),
    _f("iris.primary_color_v", .5, 0., 1.),
    _f("iris.secondary_color_u", .5, 0., 1.),
    _f("iris.secondary_color_v", .5, 0., 1.),
    _f("iris.color_blend", .5, 0., 1.),
    _f("iris.color_blend_softness", .2, .001, 1.),
    FieldSpec("iris.blend_method", "enum", "Radial", label="Blend method", choices=BLEND_METHODS),
    _f("iris.shadow_details", .4, 0., 1.),
    _f("iris.limbal_ring_size", .85, 0., 1., help="Fraction of iris radius where the outer ring begins."),
    _f("iris.limbal_ring_softness", .08, .001, .5),
    _c("iris.limbal_ring_color", (.45, .4, .35)),
    _f("iris.global_saturation", 1., 0., 3.),
    _c("iris.global_tint", WHITE),
    _f("iris.photo_mix", 1., 0., 1.),
    FieldSpec("structure.seed", "int", 1, 0, 2**31 - 1, live=False, label="Seed"),
    _f("structure.fibers", .5, 0., 1., live=False),
    _f("structure.crypts", .5, 0., 1., live=False),
    _f("structure.furrows", .5, 0., 1., live=False),
    _f("structure.freckles", .2, 0., 1., live=False),
    _f("structure.collarette", .5, 0., 1., live=False),
    _f("pupil.dilation", 1., .25, 2., help="Multiplier of the reference pupil radius."),
    _f("pupil.feather", .15, 0., 1.),
    _f("pupil.scale", 1., .25, 2.),
    _f("cornea.size", .2, .1, .3, help="Iris UV radius; .2 matches the geometric limbus."),
    _f("cornea.limbus_softness", .02, .001, .1),
    _c("cornea.limbus_color", (.95, .97, 1.)),
    _f("sclera.rotation", 0., 0., 1.),
    FieldSpec("sclera.use_custom_tint", "bool", False, label="Custom tint"),
    _c("sclera.tint", WHITE),
    _f("sclera.skin_u", .5, 0., 1.),
    _f("sclera.transmission_spread", .08, .001, .4),
    _c("sclera.transmission_color", (.9, .85, .8)),
    _f("sclera.vascularity_intensity", .3, 0., 1.),
    _f("sclera.vascularity_coverage", .3, 0., .7),
    FieldSpec("optics.side", "enum", "left", live=False, label="Side", choices=SIDES),
    _f("optics.ior", 1.336, 1.01, 1.6),
    _f("optics.chamber_depth", .0035, .001, .006),
    _f("optics.iris_convexity", 0., -.0005, .0005),
    _f("optics.cornea_roughness", .06, 0., .5),
    _f("optics.sclera_roughness", .12, 0., .7),
    _f("optics.sclera_radius", .012, .008, .018, live=False, help="User-selected sclera radius in metres."),
    _f("optics.limbus_radius", .006, .003, .009, live=False, help="User-selected limbus radius in metres."),
    _f("optics.cornea_radius", .008, .004, .012, live=False, help="User-selected corneal curvature radius in metres."),
    _f("optics.limbus_blend", .0005, .0001, .001, live=False, help="Synthetic sphere-junction blend width in metres."),
)

FIELD_BY_NAME = {spec.name: spec for spec in FIELDS}
GROUPS = ("iris", "structure", "pupil", "cornea", "sclera", "optics")

# Synthetic reference pupil radius as a fraction of iris radius.
P_REF = 0.30


class ParamError(ValueError):
    pass


def defaults() -> dict:
    out: dict = {group: {} for group in GROUPS}
    for spec in FIELDS:
        group, key = spec.name.split(".")
        out[group][key] = list(spec.default) if spec.kind == "color" else spec.default
    return out


def _coerce(spec: FieldSpec, value):
    if spec.kind == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ParamError(f"{spec.name} must be a number")
        return float(min(max(float(value), spec.lo), spec.hi))
    if spec.kind == "int":
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or int(value) != value:
            raise ParamError(f"{spec.name} must be an integer")
        return int(min(max(int(value), spec.lo), spec.hi))
    if spec.kind == "bool":
        if not isinstance(value, bool):
            raise ParamError(f"{spec.name} must be true or false")
        return value
    if spec.kind == "enum":
        if value not in spec.choices:
            raise ParamError(f"{spec.name} must be one of {', '.join(spec.choices)}")
        return value
    if spec.kind == "color":
        if (not isinstance(value, (list, tuple)) or len(value) not in (3, 4)
                or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in value)):
            raise ParamError(f"{spec.name} must be [r, g, b] (linear, 0-1)")
        return [float(min(max(v, 0.0), 1.0)) for v in value[:3]]
    raise AssertionError(spec.kind)


def validate(params: dict | None) -> dict:
    """Defaults overlaid with `params` (nested {group: {key: value}}),
    clamped to the documented artist ranges. Unknown keys are errors."""
    out = defaults()
    for group, values in (params or {}).items():
        if group in ("version", "preset", "name"):
            continue
        if group not in GROUPS or not isinstance(values, dict):
            raise ParamError(f"unknown parameter group {group!r}")
        for key, value in values.items():
            spec = FIELD_BY_NAME.get(f"{group}.{key}")
            if spec is None:
                raise ParamError(f"unknown parameter {group}.{key}")
            out[group][key] = _coerce(spec, value)
    o = out["optics"]
    if o["limbus_radius"] >= min(o["sclera_radius"], o["cornea_radius"]):
        raise ParamError("limbus_radius must be smaller than both curvature radii")
    if o["cornea_radius"] >= o["sclera_radius"]:
        raise ParamError("cornea_radius must be smaller than sclera_radius")
    return out


def to_normalized(spec: FieldSpec, value: float) -> float:
    """Slider position for a stored value (normalized fields)."""
    if not spec.normalized_ui:
        return float(value)
    return (float(value) - spec.lo) / (spec.hi - spec.lo)


def from_normalized(spec: FieldSpec, t: float) -> float:
    if not spec.normalized_ui:
        return float(t)
    return spec.lo + float(t) * (spec.hi - spec.lo)


def schema() -> dict:
    """What the web page builds its controls from."""
    fields = []
    for spec in FIELDS:
        group, key = spec.name.split(".")
        item = {"group": group, "key": key, "kind": spec.kind, "label": spec.label or key,
                "default": list(spec.default) if spec.kind == "color" else spec.default,
                "live": spec.live, "help": spec.help}
        if spec.kind in ("float", "int"):
            item.update(lo=spec.lo, hi=spec.hi, normalized_ui=spec.normalized_ui)
        if spec.choices:
            item["choices"] = list(spec.choices)
        fields.append(item)
    return {"version": ALGO_VERSION, "groups": list(GROUPS), "fields": fields,
            "presets": {name: preset(name) for name in PRESETS}}


def sclera_tint(p: dict) -> list[float]:
    """User tint, or a mild synthetic warm tint controlled by skin_u."""
    s = p["sclera"]
    if s["use_custom_tint"]:
        return list(s["tint"])
    u = s["skin_u"]
    return [1.0 + (c - 1.0) * u for c in (0.9, 0.85, 0.8)]


def pupil_scale(p: dict) -> float:
    """User dilation multiplied by the light-response control."""
    return float(p["pupil"]["dilation"] * p["pupil"]["scale"])


def pupil_ratio(p: dict) -> float:
    """Visible pupil radius / iris radius, bounded away from singularities."""
    return float(min(max(P_REF * pupil_scale(p), .05), .85))


def structure_key(p: dict, res: int) -> str:
    """Identifies the structure textures (everything that is not live)."""
    p = validate(p)
    payload = {"v": ALGO_VERSION, "res": res,
               "structure": p["structure"], "pattern": p["iris"]["pattern"],
               "side": p["optics"]["side"],
               "measurements": {k: p["optics"][k] for k in ("sclera_radius", "limbus_radius", "cornea_radius", "limbus_blend")}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]


def bake_key(p: dict, res: int, detail: str = "procedural") -> str:
    payload = {"v": ALGO_VERSION, "res": res, "params": validate(p), "detail": detail}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]


# ---- presets (ours) ---------------------------------------------------------
# U runs along our chart's hue families (0 dark brown .. 0.33 hazel .. 0.43
# green .. 0.6 grey .. 0.82 blue .. 1 light blue), V is the value (0 dark ..
# 1 light); see chart.py. Radial blends put the secondary colour in the
# pupillary zone; Structural blends put it on the darker detail.

def _preset(pattern, pu, pv, su, sv, blend, soft, method="Structural", sat=1.5, **extra):
    out = {"iris": {"pattern": pattern, "primary_color_u": pu, "primary_color_v": pv,
                    "secondary_color_u": su, "secondary_color_v": sv, "color_blend": blend,
                    "color_blend_softness": soft, "blend_method": method, "global_saturation": sat}}
    for dotted, value in extra.items():
        group, key = dotted.split("__")
        out.setdefault(group, {})[key] = value
    return out


PRESETS: dict[str, dict] = {
    "light_blue": _preset("pattern_1", 0.96, 0.62, 0.30, 0.55, 0.55, 0.25, "Radial", 1.4,
                          structure__crypts=0.3, structure__furrows=0.25),
    "blue": _preset("pattern_5", 0.83, 0.52, 0.72, 0.42, 0.5, 0.3, sat=1.6),
    "blue_gray": _preset("pattern_8", 0.70, 0.56, 0.62, 0.44, 0.5, 0.3, sat=1.2),
    "gray": _preset("pattern_1", 0.60, 0.62, 0.58, 0.45, 0.5, 0.3, sat=0.8, structure__fibers=0.65),
    "green": _preset("pattern_2", 0.43, 0.52, 0.22, 0.50, 0.5, 0.2, "Radial", 1.5, structure__crypts=0.7),
    "gray_green": _preset("pattern_9", 0.52, 0.55, 0.30, 0.50, 0.5, 0.25, "Radial", 1.2),
    "hazel": _preset("pattern_7", 0.36, 0.50, 0.20, 0.45, 0.55, 0.25, "Radial", 1.5, structure__collarette=0.65),
    "amber": _preset("pattern_4", 0.22, 0.58, 0.18, 0.48, 0.5, 0.3, sat=1.6),
    "light_brown": _preset("pattern_3", 0.18, 0.55, 0.12, 0.42, 0.5, 0.3, sat=1.4, structure__furrows=0.7),
    "brown": _preset("pattern_6", 0.12, 0.45, 0.06, 0.35, 0.5, 0.3, sat=1.3, structure__freckles=0.5),
    "dark_brown": _preset("pattern_4", 0.05, 0.32, 0.02, 0.22, 0.5, 0.3, sat=1.2, iris__shadow_details=0.35,
                          structure__fibers=0.3, iris__limbal_ring_size=0.78),
    "central_heterochromia": _preset("pattern_7", 0.85, 0.52, 0.22, 0.50, 0.5, 0.15, "Radial", 1.6,
                                     structure__collarette=0.55),
}


def preset(name: str) -> dict:
    if name not in PRESETS:
        raise ParamError(f"unknown preset {name!r}; choose from {', '.join(PRESETS)}")
    return validate(copy.deepcopy(PRESETS[name]))
