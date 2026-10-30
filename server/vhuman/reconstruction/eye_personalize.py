"""Photo-personalised iris and sclera for the Cycles candidate (stage 2d).

Eye crops are cut around the MediaPipe eye anchors of the front portrait and any accepted side views, run
through eye.extract (Daugman limbus/pupil fit, polar unwrap, highlight/eyelid masking, quality gate), ranked by
eyelid occlusion and merged: the least-occluded plate is the base and its masked sectors are filled from the
other plates' valid samples before the existing normalized-convolution fill. The merged plate is written as
iris_plate.npz (masks + photo in the iris texture layout) for offline_assets.prepare(iris_plate=...).
Sclera whiteness/yellowness/redness are measured from the eye opening outside the limbus (BiSeNet eye labels)
and mapped to the existing sclera parameters (tint, vascularity). Fallback: colour-only transfer to the
procedural iris via the plate's dominant colours. Everything is recorded in eyes.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from ..eye import extract, iris, optics
from .view_acceptance import srgb_to_lab

SIDES = ('left', 'right')
# Portrait irises are ~50 px wide at 1024^2; the pupil-centre estimate is noisier than on macro plates, so the
# concentricity gate is relaxed for personalization (recorded in eyes.json). All other extract gates apply.
RELAXED = dict(extract.THRESHOLDS, concentricity=.25)


def gate(quality, thresholds=RELAXED):
    fails = [f for f in quality.get('failures', []) if f != 'off-centre pupil']
    if quality.get('concentricity', 1.) > thresholds['concentricity']:
        fails.append('off-centre pupil (relaxed gate)')
    return fails


def eye_crops(image_path, *, scale=.55, threads=4, parser=None):
    """Square crops around both eyes from MediaPipe anchors. Returns {side: (crop_rgb, box, eye_mask|None)}."""
    from .observations import observe
    view = observe(image_path, threads=threads)['views'][0]
    anchors = view['anchors']
    img = np.asarray(Image.open(image_path).convert('RGB'))
    labels = None
    if parser is not None:
        from ..face_parsing import LABELS
        lab, conf = parser.predict(img)
        labels = np.isin(lab, [LABELS.index('left_eye'), LABELS.index('right_eye')]) & (conf > .5)
    el, er = np.array(anchors['eye_left']['xy']), np.array(anchors['eye_right']['xy'])
    size = max(48, int(round(np.linalg.norm(el - er) * scale)))
    out = {}
    for side, centre in (('left', el), ('right', er)):
        x0, y0 = int(round(centre[0] - size / 2)), int(round(centre[1] - size / 2))
        x0 = max(0, min(x0, img.shape[1] - size)); y0 = max(0, min(y0, img.shape[0] - size))
        out[side] = (img[y0:y0 + size, x0:x0 + size], (x0, y0, size), None if labels is None else labels[y0:y0 + size, x0:x0 + size])
    return out


def extract_plates(images, *, res=1024, upscale=1024, parser=None):
    """Run eye.extract on every eye crop; returns a list of dict(source, side, plate|None, quality, box, crop, eye_mask)."""
    rows = []
    for source in images:
        try:
            crops = eye_crops(source, parser=parser)
        except Exception as exc:  # landmarks missing on a profile view
            rows.append(dict(source=str(source), side=None, plate=None, quality=dict(ok=False, failures=[f'landmarks: {exc}'])))
            continue
        for side, (crop, box, eye_mask) in crops.items():
            img = Image.fromarray(crop)
            if img.width < upscale:
                img = img.resize((upscale, upscale), Image.LANCZOS)
            try:
                plate = extract.extract(np.asarray(img), res=res)
                rows.append(dict(source=str(source), side=side, plate=plate, quality=plate.quality, box=box, crop=crop, eye_mask=eye_mask))
            except extract.ExtractError as exc:
                rows.append(dict(source=str(source), side=side, plate=None, quality=dict(ok=False, failures=[str(exc)]), box=box, crop=crop, eye_mask=eye_mask))
    return rows


def merge_plates(plates, *, res=1024):
    """Best plate (least occluded) as base; its invalid strip samples come from other plates' valid samples."""
    plates = sorted(plates, key=lambda p: p.quality['occluded'])
    base = plates[0]
    strip = base.strip.copy(); valid = base.valid.copy()
    filled_from_others = 0
    if len(plates) > 1:
        acc = np.zeros_like(strip); weight = np.zeros(valid.shape, np.float32)
        for other in plates[1:]:
            w = other.valid.astype(np.float32) * (1 - other.quality['occluded'])
            acc += other.strip * w[..., None]; weight += w
        take = (~valid) & (weight > 0)
        strip[take] = acc[take] / weight[take][..., None]
        filled_from_others = int(take.sum()); valid = valid | take
    filled = extract.fill(strip, valid)
    filled = extract.normalise_lighting(filled, valid)
    layers = extract.structure_layers(filled)
    masks = iris.polar_to_square(layers, res, iris.PUPIL_MASKS, iris.OUTSIDE)
    srgb = optics.linear_to_srgb(np.clip(filled, 0, 1)).astype(np.float32)
    photo = iris.polar_to_square(np.moveaxis(srgb, -1, 0), res, (0.02, 0.02, 0.02),
                                 tuple(float(v) for v in np.median(srgb[-8:], axis=(0, 1))))
    correlations = []
    for other in plates[1:]:
        both = base.valid & other.valid
        if both.sum() > 200:
            a = (base.strip[both] @ optics.LUMA); b = (other.strip[both] @ optics.LUMA)
            correlations.append(float(np.corrcoef(a, b)[0, 1]))
    return dict(masks=np.moveaxis(masks, 0, -1).astype(np.float32), photo=np.clip(np.moveaxis(photo, 0, -1), 0, 1).astype(np.float32),
                filled_from_others=filled_from_others, base_occluded=base.quality['occluded'],
                strip_luma_correlation=correlations, colors=base.colors)


def sclera_measurement(crop, plate, parser_labels=None):
    """Median Lab of the eye opening outside the limbus (bright, low-saturation pixels)."""
    h, w = crop.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    s = w / 1024.  # plate circles are in the 1024 crop frame
    d = np.hypot(xx - plate.limbus.cx * s, yy - plate.limbus.cy * s)
    ring = (d > plate.limbus.r * s * 1.15) & (d < plate.limbus.r * s * 2.4)
    lab = srgb_to_lab(crop)
    candidates = ring & (lab[..., 0] > 45) & (np.hypot(lab[..., 1], lab[..., 2]) < 25)
    if parser_labels is not None:
        candidates &= parser_labels
    if candidates.sum() < 30:
        return None
    med = np.median(lab[candidates], 0)
    return dict(lab=med.tolist(), pixels=int(candidates.sum()),
                whiteness=float(np.clip((med[0] - 55) / 40, 0, 1)), yellowness=float(np.clip(med[2] / 20, 0, 1)),
                redness=float(np.clip(med[1] / 12, 0, 1)))


def sclera_params(measure):
    """Map the measurement to existing eye params (sclera.tint / vascularity)."""
    if measure is None:
        return {}
    L, a, b = measure['lab']
    # tint keeps brightness near white (lighting is not albedo); only chroma is transferred
    from .vita_shades import lab_to_linear_rgb
    tint = lab_to_linear_rgb([92., a * .6, b * .6])
    return dict(sclera=dict(use_custom_tint=True, tint=[float(v) for v in tint],
                            vascularity_intensity=float(np.clip(.2 + .5 * measure['redness'], .1, .8)),
                            vascularity_coverage=float(np.clip(.25 + .3 * measure['redness'], .1, .6))))


def personalize(front, views, out, *, res=1024, parsing_model=None):
    from ..face_parsing import FaceParser
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    parser = FaceParser(model=parsing_model)
    rows = extract_plates([front] + list(views), res=res, parser=parser)
    report = dict(format='vhuman.eye_personalization.v1', front=str(front), views=[str(v) for v in views],
                  thresholds=RELAXED, gate='extract.THRESHOLDS with concentricity relaxed to 0.25 for portrait-scale irises',
                  candidates=[], status='procedural')
    accepted = []
    for row in rows:
        fails = gate(row['quality']) if row.get('plate') is not None else row['quality'].get('failures', [])
        rec = dict(source=row['source'], side=row['side'], quality=dict(row['quality'], failures=fails, ok=not fails))
        if row.get('plate') is not None and not fails:
            accepted.append(row)
            rec['accepted'] = True
        else:
            rec['accepted'] = False
        report['candidates'].append(rec)
    measures = {}
    for row in rows:
        if row.get('plate') is not None and row['side']:
            m = sclera_measurement(row['crop'], row['plate'], row.get('eye_mask'))
            if m:
                measures.setdefault(row['side'], []).append(m)
    sclera = None
    if measures:
        allm = [m for ms in measures.values() for m in ms]
        lab = np.median([m['lab'] for m in allm], 0)
        sclera = dict(lab=lab.tolist(), whiteness=float(np.mean([m['whiteness'] for m in allm])),
                      yellowness=float(np.mean([m['yellowness'] for m in allm])), redness=float(np.mean([m['redness'] for m in allm])),
                      samples=len(allm))
    overrides = sclera_params(sclera)
    if accepted:
        merged = merge_plates([r['plate'] for r in accepted], res=res)
        np.savez_compressed(out / 'iris_plate.npz', masks=merged['masks'], photo=merged['photo'])
        Image.fromarray((merged['photo'] * 255).round().astype(np.uint8)).save(out / 'iris_plate_preview.png')
        report.update(status='photo iris plate', plate='iris_plate.npz', merged=dict(
            sources=[dict(source=r['source'], side=r['side'], occluded=r['quality']['occluded']) for r in sorted(accepted, key=lambda r: r['quality']['occluded'])],
            filled_from_others=merged['filled_from_others'], base_occluded=merged['base_occluded'],
            strip_luma_correlation=merged['strip_luma_correlation']), colors=_jsonable(merged['colors']))
        overrides.setdefault('iris', {})['photo_mix'] = 1.0
    else:
        # colour-only fallback: dominant colours of the best failing plate, if any detection succeeded
        best = [r for r in rows if r.get('plate') is not None]
        if best:
            best = min(best, key=lambda r: r['quality'].get('occluded', 1.))
            report.update(status='colour-only fallback (no plate passed the gate)', colors=_jsonable(best['plate'].colors))
    report.update(sclera=sclera, eye_param_overrides=overrides)
    for side in SIDES:
        for row in rows:
            if row.get('crop') is not None and row['side'] == side and str(row['source']) == str(front):
                Image.fromarray(row['crop']).save(out / f'crop_front_{side}.png')
    (out / 'eyes.json').write_text(json.dumps(report, indent=2, default=str) + '\n')
    return report


def _jsonable(value):
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--front', required=True); ap.add_argument('--view', action='append', default=[])
    ap.add_argument('--out', required=True); ap.add_argument('--res', type=int, default=1024); ap.add_argument('--parsing-model')
    a = ap.parse_args(argv)
    report = personalize(a.front, a.view, a.out, res=a.res, parsing_model=a.parsing_model)
    print(json.dumps({k: report[k] for k in ('status', 'sclera', 'eye_param_overrides')}, indent=1, default=str))
    for c in report['candidates']:
        print(c['source'].split('/')[-1], c['side'], 'OK' if c['accepted'] else 'reject: ' + '; '.join(c['quality'].get('failures', [])))


if __name__ == '__main__':
    raise SystemExit(main())
