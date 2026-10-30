"""VITA classical shade guide (A1-D4) as a colour prior for teeth (stage 2e).

CIELAB values are literature averages for the VITA classical guide measured with a spectrophotometer
(approximate; sources differ by a few units). Observed tooth colour from a delit frame is rounded to the
nearest shade, then nudged toward the observation by `blend` so photographic exposure errors do not leak in.
"""
import numpy as np

# shade: (L*, a*, b*) approximate literature means
VITA_CLASSICAL = {
    'A1': (78.5, 1.0, 14.0), 'A2': (76.0, 1.8, 17.5), 'A3': (73.5, 2.5, 20.0), 'A3.5': (71.0, 3.2, 22.5), 'A4': (68.0, 3.5, 23.0),
    'B1': (79.5, 0.2, 13.0), 'B2': (76.5, 0.8, 17.0), 'B3': (73.0, 1.8, 22.0), 'B4': (71.0, 2.2, 24.5),
    'C1': (74.5, 0.3, 12.5), 'C2': (71.0, 1.0, 16.0), 'C3': (68.5, 1.2, 17.0), 'C4': (64.0, 1.8, 18.0),
    'D2': (74.0, 1.0, 15.0), 'D3': (72.0, 1.5, 17.0), 'D4': (70.5, 1.8, 19.5),
}
VALUE_ORDER = ['B1', 'A1', 'B2', 'D2', 'A2', 'C1', 'C2', 'D4', 'A3', 'D3', 'B3', 'A3.5', 'B4', 'C3', 'A4', 'C4']


def lab_to_linear_rgb(lab):
    L, a, b = np.asarray(lab, float)
    fy = (L + 16) / 116; fx = fy + a / 500; fz = fy - b / 200
    def inv(t):
        return np.where(t > 6 / 29, t ** 3, 3 * (6 / 29) ** 2 * (t - 4 / 29))
    xyz = np.array([inv(fx) * .95047, inv(fy), inv(fz) * 1.08883])
    m = np.array([[3.2404542, -1.5371385, -0.4985314], [-0.9692660, 1.8760108, 0.0415560], [0.0556434, -0.2040259, 1.0572252]])
    return np.clip(m @ xyz, 0, 1)


def nearest_shade(lab):
    lab = np.asarray(lab, float)
    name = min(VITA_CLASSICAL, key=lambda k: np.linalg.norm(np.array(VITA_CLASSICAL[k]) - lab))
    return name, float(np.linalg.norm(np.array(VITA_CLASSICAL[name]) - lab))


def tooth_colors(observed_lab=None, *, blend=.3, default='A2'):
    """Return dict(shade, cervical_rgb, incisal_rgb, body_rgb) in linear RGB.

    Cervical is warmer/darker (dentin shows through), incisal cooler/translucent (enamel edge)."""
    if observed_lab is None:
        shade, distance = default, None; lab = np.array(VITA_CLASSICAL[default], float)
    else:
        shade, distance = nearest_shade(observed_lab)
        lab = (1 - blend) * np.array(VITA_CLASSICAL[shade], float) + blend * np.asarray(observed_lab, float)
    body = lab_to_linear_rgb(lab)
    cervical = lab_to_linear_rgb(lab + np.array([-4., 1.5, 4.]))
    incisal = lab_to_linear_rgb(lab + np.array([2., -1.5, -6.]))
    return dict(shade=shade, observed_distance=distance, lab=lab.tolist(), body_rgb=body.tolist(),
                cervical_rgb=cervical.tolist(), incisal_rgb=incisal.tolist(),
                source='VITA classical literature Lab means; observation blended at %.2f' % blend if observed_lab is not None else 'prior only')
