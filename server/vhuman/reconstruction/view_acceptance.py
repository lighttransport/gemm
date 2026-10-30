"""Acceptance metrics for generated side views (stage 1 of the Cycles A/B study).

Per view, against the front portrait:
- ArcFace cosine similarity (insightface buffalo_l on CPU onnxruntime). The buffalo_l weights are released for
  non-commercial research only, so this metric is EVALUATION ONLY and never feeds a pipeline input.
- Contamination: fraction of the head box covered by hat/clothes/glasses/earring/necklace parsing labels (BiSeNet).
- Skin colour drift: CIE76 ΔE between the median Lab colour of skin pixels in the view and in the front.
- Yaw error: |estimated - requested| when a PnP estimate is supplied (multiview.estimate_camera or the
  generated_multiview_review.v1 record written by run_generated).

Results are written as acceptance.json; thresholds are explicit so receipts show why a view was dropped.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

INSIGHTFACE_ROOT = Path('/mnt/nvme01/models/insightface')
ARCFACE_LICENSE = 'insightface buffalo_l model zoo: non-commercial research use only (evaluation only)'
CONTAMINANTS = ('glasses', 'earring', 'necklace', 'clothes', 'hat')
THRESHOLDS = dict(arcface_min=dict(quarter=.45, profile=.30), yaw_error_max_deg=8., contamination_max=.005,
                  skin_delta_e_max=6.)


def srgb_to_lab(rgb8):
    """sRGB8 (...,3) -> CIE Lab (D65), vectorised."""
    c = np.asarray(rgb8, dtype=np.float64) / 255.
    lin = np.where(c <= .04045, c / 12.92, ((c + .055) / 1.055) ** 2.4)
    m = np.array([[.4124564, .3575761, .1804375], [.2126729, .7151522, .0721750], [.0193339, .1191920, .9503041]])
    xyz = lin @ m.T / np.array([.95047, 1., 1.08883])
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def delta_e(lab_a, lab_b):
    return float(np.linalg.norm(np.asarray(lab_a) - np.asarray(lab_b)))


class ArcFace:
    def __init__(self, root=INSIGHTFACE_ROOT, det_size=640):
        from insightface.app import FaceAnalysis
        self.app = FaceAnalysis(name='buffalo_l', root=str(root), providers=['CPUExecutionProvider'],
                                allowed_modules=['detection', 'recognition'])
        self.app.prepare(ctx_id=-1, det_size=(det_size, det_size))
        self.license = ARCFACE_LICENSE

    def embed(self, rgb):
        """Largest detected face -> (normed embedding, det_score, bbox) or (None, 0, None)."""
        import cv2
        faces = self.app.get(cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR))
        if not faces:
            return None, 0., None
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return face.normed_embedding, float(face.det_score), [float(v) for v in face.bbox]

    @staticmethod
    def similarity(a, b):
        return None if a is None or b is None else float(np.dot(a, b))


def parse_view(parser, rgb):
    """BiSeNet labels -> dict(contamination fraction inside the head box, skin median Lab, label fractions)."""
    from ..face_parsing import LABELS
    labels, confidence = parser.predict(rgb)
    head = np.isin(labels, [LABELS.index(n) for n in LABELS if n not in ('background', 'clothes', 'neck')])
    ys, xs = np.nonzero(head)
    if len(ys) < 100:
        return dict(contamination=1., skin_lab=None, head_pixels=int(len(ys)), status='no head found')
    box = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
    inside = labels[box]
    bad = np.isin(inside, [LABELS.index(n) for n in CONTAMINANTS]) & (confidence[box] >= .5)
    skin = (labels == LABELS.index('skin')) & (confidence >= .6)
    skin_lab = srgb_to_lab(np.asarray(rgb)[skin]) if skin.sum() >= 100 else None
    fractions = {n: float((labels == i).mean()) for i, n in enumerate(LABELS) if (labels == i).any()}
    return dict(contamination=float(bad.mean()), skin_lab=None if skin_lab is None else np.median(skin_lab, 0).tolist(),
                head_pixels=int(len(ys)), label_fractions=fractions, status='ok')


def evaluate(front, views, *, arcface=None, parser=None, yaw_estimates=None, thresholds=THRESHOLDS):
    """front: path; views: list of dict(file, yaw_deg[, id]); yaw_estimates: {id or file: estimated_yaw_deg}."""
    arcface = arcface or ArcFace()
    if parser is None:
        from ..face_parsing import FaceParser
        parser = FaceParser()
    front_rgb = np.asarray(Image.open(front).convert('RGB'))
    front_embedding, front_score, _ = arcface.embed(front_rgb)
    front_parse = parse_view(parser, front_rgb)
    report = dict(format='vhuman.view_acceptance.v1', front=str(front), front_det_score=front_score,
                  front_skin_lab=front_parse['skin_lab'], thresholds=thresholds,
                  licenses=dict(arcface=ARCFACE_LICENSE, parsing='BiSeNet resnet18 (yakhyo/uniface, MIT)'), views=[])
    for view in views:
        rgb = np.asarray(Image.open(view['file']).convert('RGB'))
        embedding, score, bbox = arcface.embed(rgb)
        parsed = parse_view(parser, rgb)
        key = view.get('id') or Path(view['file']).name
        estimated = None if not yaw_estimates else yaw_estimates.get(key, yaw_estimates.get(Path(view['file']).name))
        row = dict(view, det_score=score, bbox=bbox, arcface=arcface.similarity(front_embedding, embedding),
                   contamination=parsed['contamination'], skin_lab=parsed['skin_lab'],
                   skin_delta_e=None if parsed['skin_lab'] is None or front_parse['skin_lab'] is None
                   else delta_e(parsed['skin_lab'], front_parse['skin_lab']),
                   estimated_yaw_deg=estimated,
                   yaw_error_deg=None if estimated is None else abs(float(estimated) - float(view['yaw_deg'])))
        failures = []
        band = 'profile' if abs(view['yaw_deg']) >= 90 else 'quarter'
        if row['arcface'] is None:
            failures.append('no face detected')
        elif row['arcface'] < thresholds['arcface_min'][band]:
            failures.append(f"arcface {row['arcface']:.3f} < {thresholds['arcface_min'][band]}")
        if row['contamination'] > thresholds['contamination_max']:
            failures.append(f"contamination {row['contamination']:.4f} > {thresholds['contamination_max']}")
        if row['skin_delta_e'] is not None and row['skin_delta_e'] > thresholds['skin_delta_e_max']:
            failures.append(f"skin dE {row['skin_delta_e']:.2f} > {thresholds['skin_delta_e_max']}")
        if row['yaw_error_deg'] is not None and row['yaw_error_deg'] > thresholds['yaw_error_max_deg']:
            failures.append(f"yaw error {row['yaw_error_deg']:.1f} > {thresholds['yaw_error_max_deg']}")
        row.update(failures=failures, accepted=not failures)
        report['views'].append(row)
    return report


def yaw_estimates_from_review(review):
    """generated_multiview_review.v1 -> {view_id: estimated_yaw_deg}."""
    return {r['view_id']: r['estimated_yaw_deg'] for r in review.get('views', []) if r.get('estimated_yaw_deg') is not None}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--front', required=True)
    ap.add_argument('--inputs', help='generated_face_inputs.v1 json (views with file/yaw_deg/id)')
    ap.add_argument('--view', action='append', default=[], help='yaw:path (repeatable, alternative to --inputs)')
    ap.add_argument('--review', help='generated_multiview_review.v1 json for PnP yaw estimates')
    ap.add_argument('--parsing-model')
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    views = []
    if a.inputs:
        inputs = json.loads(Path(a.inputs).read_text())
        base = Path(a.inputs).parent
        views = [dict(id=v.get('id'), file=str(base / v['file']), yaw_deg=v['yaw_deg'], held_out=v.get('held_out', False))
                 for v in inputs['views'] if v.get('yaw_deg', 0) != 0]
    for item in a.view:
        yaw, path = item.split(':', 1)
        views.append(dict(file=path, yaw_deg=int(yaw)))
    estimates = yaw_estimates_from_review(json.loads(Path(a.review).read_text())) if a.review else None
    from ..face_parsing import FaceParser
    report = evaluate(a.front, views, parser=FaceParser(model=a.parsing_model), yaw_estimates=estimates)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + '\n')
    for v in report['views']:
        print(f"{v['yaw_deg']:+4d} arcface={v['arcface']} contamination={v['contamination']:.4f} "
              f"dE={v['skin_delta_e']} yaw_err={v['yaw_error_deg']} -> {'ACCEPT' if v['accepted'] else 'REJECT ' + '; '.join(v['failures'])}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
