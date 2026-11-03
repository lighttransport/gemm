"""Eye parameter statistics from MetaHuman: the 12 shipped eye presets (T3D export) and the per-sample eye settings.

usage: python -m server.vhuman.metahuman.eye_prior_fit [--out PATH]
The T3D text lists only non-default fields, so presets are merged over the struct defaults read in smoke 04. Names are
converted to the Python (snake_case) property names used everywhere else. Output: MODELS_ROOT/eye_prior_v1.json with the
presets, per-parameter marginals over presets + generated samples, and (once eye close-ups exist) a colour calibration
table from MetaHuman iris U/V to rendered linear iris colour, which maps onto vhuman's eye/params colour controls.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

from . import DATASET_ROOT, MODELS_ROOT
from .manifest import LICENCE, code_sha256

T3D = DATASET_ROOT / 'v1/presets/eye_presets_t3d/MetaHumanCharacter/Tools/EyePresets/EyePresets.T3D'
DEFAULTS = DATASET_ROOT / 'smoke/04_eyes.json'
SPECIAL = {'bUseCustomTint': 'use_custom_tint', 'IrisPattern': 'pattern'}


def snake(name):
    if name in SPECIAL:
        return SPECIAL[name]
    return re.sub(r'(?<=[a-z0-9])(?=[A-Z])', '_', name).lower()


def parse_value(s):
    """Parse a T3D value: (k=v,...) struct, number, True/False, bare token."""
    s = s.strip()
    if s.startswith('('):
        return parse_struct(s[1:-1])
    if s in ('True', 'False'):
        return s == 'True'
    try:
        return float(s)
    except ValueError:
        return s.strip('"')


def split_top(s):
    parts, depth, cur, quote = [], 0, [], False
    for ch in s:
        if ch == '"':
            quote = not quote
        if not quote:
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
            elif ch == ',' and depth == 0:
                parts.append(''.join(cur)); cur = []; continue
        cur.append(ch)
    if cur:
        parts.append(''.join(cur))
    return parts


def parse_struct(s):
    out = {}
    for part in split_top(s):
        if '=' not in part:
            continue
        k, v = part.split('=', 1)
        out[k.strip()] = parse_value(v)
    return out


def to_snake(d):
    if not isinstance(d, dict):
        return d
    out = {}
    for k, v in d.items():
        if set(d) <= {'R', 'G', 'B', 'A'}:
            return [d.get('R', 0.), d.get('G', 0.), d.get('B', 0.), d.get('A', 1.)]
        out[snake(k)] = to_snake(v)
    return out


def merge(base, over):
    out = json.loads(json.dumps(base))
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = v
    return out


def eye_presets(t3d=T3D, defaults=DEFAULTS):
    text = Path(t3d).read_text(errors='ignore')
    default_eyes = json.loads(Path(defaults).read_text())['default_eyes']
    presets = []
    for m in re.finditer(r'Presets\((\d+)\)=\((.*)\)\s*$', text, re.M):
        raw = parse_struct(m.group(2))
        settings = to_snake(raw.get('EyesSettings', {}))
        merged = merge(default_eyes, settings)
        for side in ('eye_left', 'eye_right'):
            iris = merged[side]['iris']
            pat = iris.get('pattern')
            if isinstance(pat, str):                      # 'Iris007' (T3D) or '<...IrisPattern.IRIS001: 0>' (defaults)
                m2 = re.search(r'IRIS0*(\d+)', pat.upper())
                if m2:
                    iris['pattern'] = int(m2.group(1))
        presets.append(dict(index=int(m.group(1)), name=raw.get('PresetName'), settings=merged))
    return presets


def flatten(eye, prefix=''):
    out = {}
    for k, v in eye.items():
        key = f'{prefix}{k}'
        if isinstance(v, dict):
            out.update(flatten(v, key + '.'))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[key] = float(v)
        elif isinstance(v, list) and len(v) == 4:
            for c, x in zip('rgb', v[:3]):
                out[f'{key}.{c}'] = float(x)
    return out


def marginals(eyes):
    keys = sorted(set().union(*[set(e) for e in eyes]))
    stats = {}
    for k in keys:
        vals = np.array([e[k] for e in eyes if k in e], float)
        if len(vals):
            stats[k] = dict(n=int(len(vals)), mean=float(vals.mean()), std=float(vals.std()), min=float(vals.min()), max=float(vals.max()))
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--out', default=str(MODELS_ROOT / 'eye_prior_v1.json'))
    a = ap.parse_args()
    presets = eye_presets()
    flat = [flatten(p['settings']['eye_left']) for p in presets]
    rec = dict(format='vhuman.metahuman_prior.v1', name='eye_prior_v1', code_sha256=code_sha256(), licence=LICENCE, default_enabled=False,
               derived_from='MetaHuman Creator EyePresets (T3D export) merged over the struct defaults', presets=presets,
               preset_marginals=marginals(flat), colour_calibration=None,
               limitations=['12 presets only; colour calibration to vhuman iris controls pending MetaHuman eye close-up renders'])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rec, indent=1) + '\n')
    print(f'{len(presets)} presets -> {a.out}')
    for p in presets[:12]:
        i = p['settings']['eye_left']['iris']
        print(f"  {p['name']}: pattern {i.get('pattern')} primary uv ({i['primary_color_u']:.2f},{i['primary_color_v']:.2f}) "
              f"secondary ({i['secondary_color_u']:.2f},{i['secondary_color_v']:.2f}) limbal {i['limbal_ring_size']:.3f} sat {i['global_saturation']:.2f}")


if __name__ == '__main__':
    main()
