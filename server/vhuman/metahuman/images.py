"""Image helpers for the UE outputs: EXR readback (OpenCV with OpenEXR enabled) and exact 8-bit re-quantisation.

UE's export_render_target writes RGBA16F EXR for float render targets. Textures drawn through the unlit copy material hold
raw sampled values (bytes/255 for 8-bit sources), so round(v*255) recovers the original bytes exactly when the draw is
bit-exact; `requantise_error` measures that.
"""
import os

import numpy as np

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')


def read_exr(path):
    import os
    os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')     # some OpenCV builds disable EXR unless asked
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:     # OpenCV 5 builds reject Blender's EXRs: fall back to the OpenEXR bindings (BSD-3) when installed
        try:
            import OpenEXR
        except ImportError:
            raise IOError(f'cannot read {path}') from None
        ch = OpenEXR.File(str(path)).parts[0].channels
        if 'RGBA' in ch or 'RGB' in ch:
            return np.ascontiguousarray(ch['RGBA' if 'RGBA' in ch else 'RGB'].pixels.astype(np.float32))
        names = [n for n in ('R', 'G', 'B', 'A') if n in ch]
        return np.ascontiguousarray(np.stack([ch[n].pixels for n in names], -1).astype(np.float32))
    if img.ndim == 3:
        img = img[..., ::-1] if img.shape[2] == 3 else img[..., [2, 1, 0, 3]]      # BGR(A) -> RGB(A)
    return np.ascontiguousarray(img.astype(np.float32))


def write_exr(path, rgb):
    import os
    os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
    import cv2
    rgb = np.asarray(rgb, np.float32)
    bgr = rgb[..., ::-1] if rgb.shape[-1] == 3 else rgb[..., [2, 1, 0, 3]]
    if not cv2.imwrite(str(path), np.ascontiguousarray(bgr), [cv2.IMWRITE_EXR_COMPRESSION, cv2.IMWRITE_EXR_COMPRESSION_ZIP]):
        raise IOError(f'cannot write {path}')


def requantise(img):
    """float in [0,1] -> uint8 and the max |v*255 - round| residual (0 if the values are exact bytes)."""
    v = np.clip(np.asarray(img, np.float32), 0, 1) * 255
    q = np.rint(v)
    return q.astype(np.uint8), float(np.abs(v - q).max())


def write_png(path, arr):
    from PIL import Image
    Image.fromarray(np.asarray(arr)).save(path)


def exr_to_png8(src, dst, channels=3):
    img = read_exr(src)[..., :channels]
    q, err = requantise(img)
    write_png(dst, q)
    return err


def display_srgb(linear, ev=-1.5):
    """Host display transform matching cycles_scene's Standard view at -1.5 EV: exposure then sRGB OETF, 8-bit."""
    x = np.clip(np.asarray(linear, np.float32) * (2 ** ev), 0, 1)
    s = np.where(x <= .0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - .055)
    return (s * 255 + .5).astype(np.uint8)
