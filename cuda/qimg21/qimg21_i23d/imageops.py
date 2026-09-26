"""RGBA and mask helpers shared by every Image-to-3D operation.

All images are handled as uint8 RGBA numpy arrays (H, W, 4) so every view goes
through the same code: one resampler (premultiplied Lanczos, so antialiased
edges keep their colour), one mask convention (255 = region to change), one
normalization. Nothing here calls the model.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from PIL import Image


class MaskError(ValueError):
    """An unusable mask, with the reason."""


def load_rgba(path) -> np.ndarray:
    with Image.open(path) as image:
        image.load()
        return np.asarray(image.convert("RGBA")).copy()


def save_png(array: np.ndarray, path, *, mode: str = "RGBA") -> Path:
    """Write RGBA (or the RGB/L view of it) as PNG; JPEG for .jpg/.jpeg paths
    (RGB only: alpha is composited on white first)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(np.ascontiguousarray(array))
    if path.suffix.lower() in (".jpg", ".jpeg"):
        rgb = composite(array, (255, 255, 255)) if array.ndim == 3 and array.shape[2] == 4 else array
        Image.fromarray(np.ascontiguousarray(rgb)).convert("RGB").save(path, quality=95)
        return path
    if mode != image.mode:
        image = image.convert(mode)
    image.save(path)
    return path


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resize_rgba(rgba: np.ndarray, width: int, height: int) -> np.ndarray:
    """Premultiplied Lanczos, so a transparent pixel's RGB never bleeds into
    an antialiased edge."""
    image = Image.fromarray(np.ascontiguousarray(rgba), "RGBA").convert("RGBa")
    return np.asarray(image.resize((width, height), Image.LANCZOS).convert("RGBA")).copy()


def alpha_bbox(rgba: np.ndarray, threshold: int = 8):
    """(x0, y0, x1, y1), exclusive, of alpha > threshold, or None if empty."""
    alpha = rgba[..., 3] > threshold
    rows, cols = np.flatnonzero(alpha.any(axis=1)), np.flatnonzero(alpha.any(axis=0))
    if not rows.size:
        return None
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def composite(rgba: np.ndarray, background=(255, 255, 255)) -> np.ndarray:
    alpha = rgba[..., 3:4].astype(np.float32) / 255.0
    rgb = rgba[..., :3].astype(np.float32) * alpha + np.asarray(background, np.float32) * (1.0 - alpha)
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)


@dataclass
class NormalizeTransform:
    """Where the source pixels went: output = source * scale + offset."""
    scale: float
    offset_x: float
    offset_y: float
    source_bbox: tuple | None
    output_size: tuple

    def to_dict(self) -> dict:
        return asdict(self)


def normalize_object(rgba: np.ndarray, *, size: tuple | None = None, fill: float | None = 0.85,
                     pad: int = 0, center: bool = True, crop: bool = False, threshold: int = 8):
    """Frame an RGBA object for reconstruction, keeping its aspect ratio.

    - crop: cut to the alpha bounding box (plus `pad` pixels); otherwise the
      canvas keeps the source aspect ratio, or `size` (w, h) when given.
    - fill: scale so the object's longest side covers this fraction of the
      canvas's shorter side (None keeps the source scale).
    - center: put the bounding box in the middle of the canvas.
    Returns (rgba, NormalizeTransform). An object with no alpha raises."""
    bbox = alpha_bbox(rgba, threshold)
    if bbox is None:
        raise MaskError("the image has no foreground: its alpha channel is empty")
    height, width = rgba.shape[:2]
    x0, y0, x1, y1 = bbox
    if crop:
        x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
        x1, y1 = min(width, x1 + pad), min(height, y1 + pad)
        out = rgba[y0:y1, x0:x1].copy()
        return out, NormalizeTransform(1.0, -x0, -y0, bbox, (x1 - x0, y1 - y0))
    out_w, out_h = size if size else (width, height)
    box_w, box_h = x1 - x0, y1 - y0
    scale = 1.0
    if fill is not None:
        if not 0.05 <= fill <= 1.0:
            raise MaskError(f"fill must be in [0.05, 1], got {fill}")
        target = fill * min(out_w, out_h) - 2 * pad
        scale = max(target, 1.0) / max(box_w, box_h)
    elif size:
        scale = min(1.0, (min(out_w, out_h) - 2 * pad) / max(box_w, box_h))
    # Resample the whole source so the edges come from real neighbours.
    scaled_w, scaled_h = max(1, round(width * scale)), max(1, round(height * scale))
    scaled = rgba if (scaled_w, scaled_h) == (width, height) else resize_rgba(rgba, scaled_w, scaled_h)
    sx0, sy0 = x0 * scaled_w / width, y0 * scaled_h / height
    sbw, sbh = box_w * scaled_w / width, box_h * scaled_h / height
    if center:
        off_x, off_y = round((out_w - sbw) / 2 - sx0), round((out_h - sbh) / 2 - sy0)
    else:
        off_x, off_y = round((out_w - scaled_w) / 2), round((out_h - scaled_h) / 2)
    out = np.zeros((out_h, out_w, 4), np.uint8)
    dx0, dy0 = max(0, off_x), max(0, off_y)
    dx1, dy1 = min(out_w, off_x + scaled_w), min(out_h, off_y + scaled_h)
    if dx1 > dx0 and dy1 > dy0:
        out[dy0:dy1, dx0:dx1] = scaled[dy0 - off_y:dy1 - off_y, dx0 - off_x:dx1 - off_x]
    return out, NormalizeTransform(scaled_w / width, off_x, off_y, bbox, (out_w, out_h))


# ---- masks (255 = region the model may change) --------------------------

def make_mask(size: tuple, *, image=None, rect=None, circle=None, invert: bool = False,
              resize: bool = False, feather: int = 0) -> np.ndarray:
    """A uint8 (H, W) mask for an image of `size` (w, h), from exactly one of:
    - image: a path or array; L/RGB use luminance, RGBA/LA use alpha;
    - rect: (x, y, w, h) in pixels;
    - circle: (cx, cy, r) in pixels.
    `resize` scales a mask image of another size; otherwise that is an error."""
    width, height = size
    given = [name for name, value in (("image", image), ("rect", rect), ("circle", circle)) if value is not None]
    if len(given) != 1:
        raise MaskError(f"give exactly one of a mask image, a rect or a circle (got {given or 'none'})")
    if image is not None:
        if isinstance(image, np.ndarray):
            source = Image.fromarray(image)
        else:
            try:
                source = Image.open(image)
                source.load()
            except (OSError, ValueError) as exc:
                raise MaskError(f"cannot read mask image {image}: {exc}") from None
        if source.mode in ("RGBA", "LA", "PA"):
            source = source.getchannel("A")
        else:
            source = source.convert("L")
        if source.size != (width, height):
            if not resize:
                raise MaskError(f"mask is {source.size[0]}x{source.size[1]} but the image is {width}x{height}; "
                                "pass a mask of the same size or allow resizing")
            source = source.resize((width, height), Image.BILINEAR)
        mask = np.asarray(source).copy()
    elif rect is not None:
        if len(rect) != 4:
            raise MaskError(f"rect must be x,y,w,h, got {rect}")
        x, y, w, h = (int(v) for v in rect)
        if w <= 0 or h <= 0 or x >= width or y >= height or x + w <= 0 or y + h <= 0:
            raise MaskError(f"rect {rect} does not overlap the {width}x{height} image")
        mask = np.zeros((height, width), np.uint8)
        mask[max(0, y):min(height, y + h), max(0, x):min(width, x + w)] = 255
    else:
        if len(circle) != 3:
            raise MaskError(f"circle must be cx,cy,r, got {circle}")
        cx, cy, r = (float(v) for v in circle)
        if r <= 0:
            raise MaskError(f"circle radius must be positive, got {r}")
        yy, xx = np.mgrid[0:height, 0:width]
        dist = np.sqrt((xx + 0.5 - cx) ** 2 + (yy + 0.5 - cy) ** 2)
        # One-pixel antialiased edge.
        mask = (np.clip(r - dist + 0.5, 0.0, 1.0) * 255).astype(np.uint8)
        if not mask.any():
            raise MaskError(f"circle {circle} does not overlap the {width}x{height} image")
    if invert:
        mask = 255 - mask
    if feather > 0:
        from PIL import ImageFilter
        mask = np.asarray(Image.fromarray(mask).filter(ImageFilter.GaussianBlur(feather))).copy()
    if not (mask > 0).any():
        raise MaskError("the mask selects nothing: every pixel is 0")
    return mask


def latent_mask(mask: np.ndarray, grid_h: int, grid_w: int, *, dilate: int = 1) -> np.ndarray:
    """Per-latent-token edit weights in [0, 1], row-major [grid_h * grid_w].

    A token is a 16x16 pixel patch; its weight is the patch's mean mask value,
    grown by `dilate` tokens so the edit has room to blend at the boundary
    (the pixel paste-back restores everything outside the mask exactly)."""
    image = Image.fromarray(mask).resize((grid_w * 16, grid_h * 16), Image.BILINEAR)
    values = np.asarray(image, np.float32).reshape(grid_h, 16, grid_w, 16).mean(axis=(1, 3)) / 255.0
    for _ in range(max(0, dilate)):
        # 3x3 max filter: the token and its eight neighbours.
        padded = np.pad(values, 1, mode="edge")
        values = np.max([padded[dy:dy + grid_h, dx:dx + grid_w] for dy in range(3) for dx in range(3)], axis=0)
    return np.ascontiguousarray(values.reshape(-1), dtype=np.float32)


def paste_outside(original: np.ndarray, generated: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Original pixels where the mask is 0, generated where it is 255, blended
    in between. Sizes must match."""
    if original.shape != generated.shape or original.shape[:2] != mask.shape:
        raise MaskError(f"paste-back needs matching sizes, got {original.shape}, {generated.shape}, {mask.shape}")
    weight = mask.astype(np.float32)[..., None] / 255.0
    out = original.astype(np.float32) * (1.0 - weight) + generated.astype(np.float32) * weight
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def snap_size(width: int, height: int, area: int | None = None, multiple: int = 32) -> tuple:
    """(w, h) at the source aspect ratio, each a multiple of 32, optionally at
    `area` pixels -- the resolution rule the pipeline applies to references."""
    if area:
        ratio = width / height
        w = (area * ratio) ** 0.5
        width, height = w, w / ratio
    return (max(256, int(round(width / multiple)) * multiple),
            max(256, int(round(height / multiple)) * multiple))


def mirror_shift(reference: np.ndarray, other: np.ndarray, *, span: int) -> tuple[int, float, float]:
    """The horizontal shift of `other` (an opposite view's alpha mask) whose
    mirror image best overlaps `reference`: (shift, IoU there, IoU at 0).

    In an orthographic view pair 180 degrees apart (front/back, left/right
    profile) each silhouette is the other's mirror image, so the best shift
    puts both views on one turning axis."""
    mirrored = other[:, ::-1]
    width = reference.shape[1]

    def iou(shift):
        moved = np.zeros_like(mirrored)
        if shift >= 0:
            moved[:, shift:] = mirrored[:, :width - shift]
        else:
            moved[:, :shift] = mirrored[:, -shift:]
        return (reference & moved).sum() / max(1, (reference | moved).sum())

    scores = {shift: iou(shift) for shift in range(-span, span + 1)}
    best = max(scores, key=lambda k: (scores[k], -abs(k)))
    # The mask was mirrored: moving the mirror by +s moves the view by -s.
    return -best, scores[best], scores[0]


def split_sheet(rgba: np.ndarray, count: int, *, size: int = 512, fill: float = 0.85, min_gap: int = 8,
                threshold: int = 64, register: tuple = (),
                report: list | None = None) -> tuple[list[np.ndarray], list[tuple]]:
    """Split a one-row sheet of `count` views of one object (a turnaround
    sheet on a transparent background) into square RGBA views.

    Views are the runs of columns holding foreground, gaps narrower than
    min_gap merged. All views get ONE scale (the largest view's longest side
    fills `fill` of the canvas) and one ground line, so relative sizes -- a
    profile being wider or narrower than the front -- survive; each is
    centred horizontally. `register` lists (reference, opposite) view index
    pairs 180 degrees apart: the opposite view is then shifted so its mirror
    image best overlaps the reference (mirror_shift), which puts both on one
    turning axis -- bounding-box centring alone misplaces a view with large
    one-sided parts. A shift is applied only when it clearly improves the
    overlap; `report` receives one dict per pair. Returns the views and
    their sheet boxes (x0, y0, x1, y1)."""
    # Faint pixels (a drawn ground shadow, haze) are not the character:
    # Pixal3D would reconstruct them as a thin plate under its feet.
    rgba = rgba.copy()
    rgba[rgba[..., 3] <= threshold] = 0
    alpha = rgba[..., 3] > threshold
    # A column belongs to a figure when it holds a real run of solid pixels:
    # the faint ground shadow a sheet often draws under all figures (a couple
    # of rows, low alpha) must not join them into one.
    columns = np.append(alpha.sum(axis=0) >= max(4, alpha.shape[0] // 100), False)
    runs, start = [], None
    for x, occupied in enumerate(columns):
        if occupied and start is None:
            start = x
        elif not occupied and start is not None:
            if runs and start - runs[-1][1] < min_gap:
                runs[-1][1] = x
            else:
                runs.append([start, x])
            start = None
    width = alpha.shape[1]
    runs = [r for r in runs if r[1] - r[0] >= max(8, width // (count * 12))]
    if len(runs) != count:
        raise MaskError(f"the sheet holds {len(runs)} separate figures, not {count}; regenerate it (another seed) "
                        "or check that its background is transparent")
    boxes = []
    for x0, x1 in runs:
        ys = np.nonzero(alpha[:, x0:x1].any(axis=1))[0]
        boxes.append((x0, int(ys.min()), x1, int(ys.max()) + 1))
    tallest = max(b[3] - b[1] for b in boxes)
    widest = max(b[2] - b[0] for b in boxes)
    scale = fill * size / max(tallest, widest)
    ground = (size + scale * tallest) / 2
    views = []
    for x0, y0, x1, y1 in boxes:
        w, h = max(1, round((x1 - x0) * scale)), max(1, round((y1 - y0) * scale))
        canvas = np.zeros((size, size, 4), np.uint8)
        left, top = (size - w) // 2, max(0, round(ground - h))
        canvas[top:top + h, left:left + w] = resize_rgba(rgba[y0:y1, x0:x1], w, h)[:size - top]
        views.append(canvas)
    for reference, opposite in register:
        shift, after, before = mirror_shift(views[reference][..., 3] > 127, views[opposite][..., 3] > 127,
                                            span=max(1, size * 15 // 100))
        applied = shift if after > before + 0.01 else 0
        if applied:
            moved = np.zeros_like(views[opposite])
            if applied > 0:
                moved[:, applied:] = views[opposite][:, :size - applied]
            else:
                moved[:, :applied] = views[opposite][:, -applied:]
            views[opposite] = moved
        if report is not None:
            report.append({"reference": reference, "view": opposite, "shift_px": applied,
                           "mirror_iou": round(float(after if applied else before), 4),
                           "mirror_iou_centred": round(float(before), 4)})
    return views, boxes
