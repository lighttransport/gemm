"""ColorChecker calibration for the Cycles offline renders.

A 24-patch ColorChecker (6 x 4, matte) is placed in the render under the scene's own lights (cycles_scene, request
'chart'). From the linear render (beauty.exr) this module derives:
  - exposure_ev: the display exposure that maps the neutral patches' rendered radiance back to their reflectance
    (neutral 5 at ~19% shows as ~19% linear, i.e. a photographic grey-card exposure),
  - wb_gains: per-channel gains that make the neutral row neutral (green fixed at 1),
  - per-patch Lab error after calibration, as a check of the scene's colour rendering.
Reference values: the commonly published ColorChecker Classic sRGB (D65, 8-bit) set.
"""
import json
from pathlib import Path

import numpy as np

# (name, sRGB 8-bit), row-major from the top-left "dark skin" patch.
PATCHES = (
    ('dark skin', (115, 82, 68)), ('light skin', (194, 150, 130)), ('blue sky', (98, 122, 157)),
    ('foliage', (87, 108, 67)), ('blue flower', (133, 128, 177)), ('bluish green', (103, 189, 170)),
    ('orange', (214, 126, 44)), ('purplish blue', (80, 91, 166)), ('moderate red', (193, 90, 99)),
    ('purple', (94, 60, 108)), ('yellow green', (157, 188, 64)), ('orange yellow', (224, 163, 46)),
    ('blue', (56, 61, 150)), ('green', (70, 148, 73)), ('red', (175, 54, 60)),
    ('yellow', (231, 199, 31)), ('magenta', (187, 86, 149)), ('cyan', (8, 133, 161)),
    ('white 9.5', (243, 243, 242)), ('neutral 8', (200, 200, 200)), ('neutral 6.5', (160, 160, 160)),
    ('neutral 5', (122, 122, 121)), ('neutral 3.5', (85, 85, 85)), ('black 2', (52, 52, 52)))
COLUMNS, ROWS = 6, 4
NEUTRALS = (19, 20, 21, 22)   # neutral 8 .. neutral 3.5: white clips first, black is noise-dominated
LUMA = np.array([.2126, .7152, .0722])


def srgb_to_linear(c):
    c = np.asarray(c, np.float64) / 255.
    return np.where(c <= .04045, c / 12.92, ((c + .055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    c = np.clip(np.asarray(c, np.float64), 0, 1)
    return np.where(c <= .0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - .055)


def reference_linear():
    """[24, 3] linear-sRGB reflectances (used as the patches' Base Color)."""
    return srgb_to_linear([rgb for _, rgb in PATCHES])


def layout(patch=.018, gap=.003):
    """Patch centres in the chart plane (metres, x right, y up, chart centred at 0)."""
    centres = []
    for i in range(len(PATCHES)):
        r, c = divmod(i, COLUMNS)
        centres.append(((c - (COLUMNS - 1) / 2) * (patch + gap), ((ROWS - 1) / 2 - r) * (patch + gap)))
    return np.array(centres), patch


def sample(linear, patches, inner=.5):
    """Median linear RGB inside each patch (inner fraction of its pixel half-size). patches: chart.json entries."""
    h, w = linear.shape[:2]
    out = []
    for p in patches:
        cx, cy, half = p['pixel_center'][0], p['pixel_center'][1], p['pixel_half_size'] * inner
        x0, x1 = int(max(0, cx - half)), int(min(w, cx + half + 1))
        y0, y1 = int(max(0, cy - half)), int(min(h, cy + half + 1))
        if x1 - x0 < 2 or y1 - y0 < 2:
            raise ValueError(f'patch {p["name"]} is outside the image or too small')
        out.append(np.median(linear[y0:y1, x0:x1, :3].reshape(-1, 3), 0))
    return np.array(out)


def calibrate(rendered):
    """rendered: [24, 3] linear RGB of the patches. Returns exposure_ev, wb_gains and per-patch errors."""
    ref = reference_linear()
    n = list(NEUTRALS)
    gains = (ref[n].sum(0) / np.maximum(rendered[n].sum(0), 1e-9))
    gains = gains / gains[1]
    balanced = rendered * gains
    scale = float((ref[n] @ LUMA).sum() / max((balanced[n] @ LUMA).sum(), 1e-12))
    corrected = balanced * scale
    from .view_acceptance import srgb_to_lab
    lab_ref = srgb_to_lab(np.rint(linear_to_srgb(ref) * 255).astype(np.uint8))
    lab_out = srgb_to_lab(np.rint(linear_to_srgb(corrected) * 255).astype(np.uint8))
    de = np.linalg.norm(lab_ref - lab_out, axis=-1)
    return dict(exposure_ev=float(np.log2(scale)), wb_gains=[float(g) for g in gains],
                delta_e_mean=float(de.mean()), delta_e_max=float(de.max()),
                delta_e={PATCHES[i][0]: float(de[i]) for i in range(len(PATCHES))},
                clipped=[PATCHES[i][0] for i in range(len(PATCHES)) if (corrected[i] > 1).any()])


def calibrate_render(render_dir):
    """Calibration of a chart render directory (beauty.exr + chart.json); writes chart_calibration.json."""
    import os
    os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
    import cv2
    d = Path(render_dir)
    exr = cv2.imread(str(d / 'beauty.exr'), cv2.IMREAD_UNCHANGED)
    if exr is None:
        raise ValueError(f'cannot read {d / "beauty.exr"}')
    linear = exr[..., 2::-1].astype(np.float64)
    chart = json.loads((d / 'chart.json').read_text())
    result = calibrate(sample(linear, chart['patches']))
    result.update(source=str(d), lighting=chart.get('lighting'))
    (d / 'chart_calibration.json').write_text(json.dumps(result, indent=2) + '\n')
    return result
