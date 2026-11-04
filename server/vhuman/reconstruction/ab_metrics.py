"""Image metrics for the Cycles A/B study: crops by parsing label, LPIPS/DISTS, Lab dE, iris strip correlation,
silhouette IoU and per-texel seam energy. CPU only (torch on CPU) so evaluation never competes for the GPU.

LPIPS (BSD-2) and DISTS (MIT) weights ship with their packages; their AlexNet/VGG16 backbones come from
torchvision (ImageNet weights, BSD-3 code). ArcFace lives in view_acceptance (evaluation-only licence).
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .view_acceptance import srgb_to_lab

_LPIPS = None
_DISTS = None


def _torch_cpu():
    import torch
    torch.set_num_threads(16)
    return torch


def lpips_distance(a, b, size=256):
    """LPIPS (AlexNet) between two RGB8 images (resized to size^2)."""
    global _LPIPS
    torch = _torch_cpu()
    if _LPIPS is None:
        import lpips
        _LPIPS = lpips.LPIPS(net='alex', verbose=False).eval()
    ta, tb = (_to_tensor(x, size, torch) * 2 - 1 for x in (a, b))
    with torch.no_grad():
        return float(_LPIPS(ta, tb).item())


def dists_distance(a, b, size=256):
    global _DISTS
    torch = _torch_cpu()
    if _DISTS is None:
        from DISTS_pytorch import DISTS
        _DISTS = DISTS().eval()
    ta, tb = (_to_tensor(x, size, torch) for x in (a, b))
    with torch.no_grad():
        return float(_DISTS(ta, tb).item())


def _to_tensor(img, size, torch):
    im = Image.fromarray(np.asarray(img, np.uint8)).convert('RGB').resize((size, size), Image.BICUBIC)
    return torch.from_numpy(np.asarray(im, np.float32) / 255.).permute(2, 0, 1)[None]


def label_mask(parser, rgb, names, min_conf=.5):
    from ..face_parsing import LABELS
    labels, conf = parser.predict(np.asarray(rgb, np.uint8))
    return np.isin(labels, [LABELS.index(n) for n in names]) & (conf >= min_conf)


def bbox_crop(rgb, mask, *, margin=.25, square=True):
    """Crop around a mask (with margin). Returns (crop, crop_mask, box) or (None, None, None)."""
    ys, xs = np.nonzero(mask)
    if len(ys) < 50:
        return None, None, None
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    h, w = y1 - y0, x1 - x0
    if square:
        s = int(max(h, w) * (1 + 2 * margin)); cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
        y0, x0 = cy - s // 2, cx - s // 2; y1, x1 = y0 + s, x0 + s
    else:
        y0 -= int(h * margin); y1 += int(h * margin); x0 -= int(w * margin); x1 += int(w * margin)
    H, W = mask.shape
    y0, x0 = max(0, y0), max(0, x0); y1, x1 = min(H, y1), min(W, x1)
    return np.asarray(rgb)[y0:y1, x0:x1], mask[y0:y1, x0:x1], (int(x0), int(y0), int(x1), int(y1))


def lab_delta(a, b, mask_a=None, mask_b=None):
    """Mean / p95 CIE76 dE between two equally sized crops (masked pixels), plus median-colour dE."""
    la, lb = srgb_to_lab(a), srgb_to_lab(b)
    m = np.ones(la.shape[:2], bool)
    if mask_a is not None:
        m &= mask_a
    if mask_b is not None:
        m &= mask_b
    if m.sum() < 20:
        return dict(mean=None, p95=None, median_color=None, pixels=int(m.sum()))
    d = np.linalg.norm(la[m] - lb[m], axis=-1)
    return dict(mean=float(d.mean()), p95=float(np.percentile(d, 95)),
                median_color=float(np.linalg.norm(np.median(la[m], 0) - np.median(lb[m], 0))), pixels=int(m.sum()))


def median_lab(rgb, mask):
    if mask is None or mask.sum() < 20:
        return None
    return np.median(srgb_to_lab(np.asarray(rgb)[mask]), 0).tolist()


def drop_specks(mask, frac=.05):
    """Keep connected components with at least frac of the largest component's pixels."""
    from scipy import ndimage
    lab, n = ndimage.label(mask)
    if n <= 1:
        return mask
    sizes = np.bincount(lab.ravel())[1:]
    return np.isin(lab, 1 + np.flatnonzero(sizes >= frac * sizes.max()))


def compare_region(reference, render, parser, names, *, size=256, margin=.25, aligned=False):
    """Crop both images around the same parsing label set, resize to size^2 and score.

    Default: crops are aligned by their own label boxes (renders and generated references do not share a calibrated
    camera); the metrics therefore measure appearance, not registration. aligned=True (same calibrated camera, e.g.
    the UE-vs-Cycles harness): parser specks are dropped and one box (around the labels both images agree on, else
    the reference's) crops both images, so a stray label in one image cannot change its crop."""
    ma = label_mask(parser, reference, names); mb = label_mask(parser, render, names)
    if aligned:
        ma, mb = drop_specks(ma), drop_specks(mb)
        both = ma & mb
        _, _, box = bbox_crop(reference, both if both.sum() >= 50 else ma, margin=margin)
        if box is None:
            return dict(status='region not found', reference_pixels=int(ma.sum()), render_pixels=int(mb.sum()))
        x0, y0, x1, y1 = box
        ca, cma, boxa = np.asarray(reference)[y0:y1, x0:x1], ma[y0:y1, x0:x1], box
        cb, cmb, boxb = np.asarray(render)[y0:y1, x0:x1], mb[y0:y1, x0:x1], box
    else:
        ca, cma, boxa = bbox_crop(reference, ma, margin=margin); cb, cmb, boxb = bbox_crop(render, mb, margin=margin)
    if ca is None or cb is None:
        return dict(status='region not found', reference_pixels=int(ma.sum()), render_pixels=int(mb.sum()))
    ra = np.asarray(Image.fromarray(ca).resize((size, size), Image.BICUBIC))
    rb = np.asarray(Image.fromarray(cb).resize((size, size), Image.BICUBIC))
    rma = np.asarray(Image.fromarray(cma.astype(np.uint8) * 255).resize((size, size))) > 127
    rmb = np.asarray(Image.fromarray(cmb.astype(np.uint8) * 255).resize((size, size))) > 127
    return dict(status='ok', lpips=lpips_distance(ra, rb, size), dists=dists_distance(ra, rb, size),
                delta_e=lab_delta(ra, rb, rma, rmb), mask_iou=float((rma & rmb).sum() / max((rma | rmb).sum(), 1)),
                reference_box=boxa, render_box=boxb, crops=(ra, rb))


def strip_correlation(plate_a, plate_b):
    """Luma correlation of two eye.extract plates' polar strips over jointly valid samples."""
    from ..eye import optics
    both = plate_a.valid & plate_b.valid
    if both.sum() < 200:
        return None
    a = plate_a.strip[both] @ optics.LUMA; b = plate_b.strip[both] @ optics.LUMA
    return float(np.corrcoef(a, b)[0, 1])


def side_by_side(images, labels, out, *, height=512, title=None):
    """Labelled horizontal comparison sheet (A | B | C ...)."""
    from PIL import ImageDraw
    tiles = []
    for img, label in zip(images, labels):
        im = Image.fromarray(np.asarray(img, np.uint8)) if not isinstance(img, Image.Image) else img
        im = im.convert('RGB'); im = im.resize((int(im.width * height / im.height), height), Image.LANCZOS)
        canvas = Image.new('RGB', (im.width, height + 28), (24, 24, 24)); canvas.paste(im, (0, 28))
        ImageDraw.Draw(canvas).text((8, 7), label, fill=(235, 235, 235)); tiles.append(canvas)
    width = sum(t.width for t in tiles) + 4 * (len(tiles) - 1)
    head = 24 if title else 0
    sheet = Image.new('RGB', (width, tiles[0].height + head), (12, 12, 12))
    if title:
        from PIL import ImageDraw
        ImageDraw.Draw(sheet).text((8, 5), title, fill=(255, 255, 255))
    x = 0
    for t in tiles:
        sheet.paste(t, (x, head)); x += t.width + 4
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return str(out)


def write_json(path, data):
    def default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return str(o)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, default=default) + '\n')
