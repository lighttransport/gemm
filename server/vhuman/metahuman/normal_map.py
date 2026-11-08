"""MetaHuman tangent-space skin normals -> GNM UV tangent space (Blender / OpenGL convention).

usage: python -m server.vhuman.metahuman.normal_map <sample_id> [--res 2048] [--out PNG]     (writes a converted map)
       python -m server.vhuman.metahuman.normal_map --check <sample_id>                    (convention self-checks)

Conventions (measured 2026-11-04 on generated characters, see check()):
  MetaHuman tex/normal.png: red = +u, green = +row (texture rows grow downward; UE/DirectX "green down"), blue = normal.
  The red/green divergences correlate with the cavity detail with the same sign (r ~ -.47 for both), and the height
  integrated under this reading is negatively skewed (pores/creases are pits: skew -.5 .. -.9).
  GNM skin_normal.png as Blender's Normal Map node reads it: red = +u, green = +V with V up (cycles_scene sets
  UV v = 1 - row/H), tangent frame = smooth MikkTSpace-like (T from UV, B = sign * N x T).

Conversion per GNM texel: the MH normal is sampled bilinearly at the texel map's MH UV, expanded in the MH smooth
tangent frame at the texel map's MH triangle/barycentrics (on that identity's MH head), and re-expressed in the GNM smooth
frame of the same identity's bound GNM skin. The result carries both the MH texture detail and the MH-vs-GNM mesh
shading difference (MH has twice the skin vertices). Texels without a valid MH correspondence stay flat (0, 0, 1).
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .register import OUT_DIR, load_gnm, resample, vertex_normals
from . import template as mh_template

FLAT = np.array([0., 0., 1.])


def _normalize(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def corner_frames(vertices, triangles, triangle_uvs):
    """Smooth per-corner tangent frames (F, 3, 3, 3) = [face, corner, (T, B, N)].

    T/B: area-weighted dpos/du and dpos/d(row) accumulated over corners that share a vertex AND a UV (UV seams keep
    separate frames), N: smooth vertex normal. T is made orthogonal to N; B = sign * N x T with the sign of the accumulated
    dpos/d(row) (row = the triangle_uvs second coordinate as stored, i.e. texture rows)."""
    v = np.asarray(vertices, np.float64); f = np.asarray(triangles, np.int64); uv = np.asarray(triangle_uvs, np.float64)
    p = v[f]; e1, e2 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
    d1, d2 = uv[:, 1] - uv[:, 0], uv[:, 2] - uv[:, 0]
    det = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]
    r = np.where(np.abs(det) > 1e-14, 1 / np.where(det == 0, 1, det), 0)[:, None]
    t = (e1 * d2[:, 1:] - e2 * d1[:, 1:]) * r            # dpos/du
    b = (e2 * d1[:, :1] - e1 * d2[:, :1]) * r            # dpos/drow
    area = .5 * np.linalg.norm(np.cross(e1, e2), axis=1)[:, None]
    t, b = _normalize(t) * area, _normalize(b) * area
    key = np.concatenate([f.reshape(-1, 1).astype(np.float64), np.round(uv.reshape(-1, 2) * 1e5)], 1)
    _, inv = np.unique(key, axis=0, return_inverse=True); inv = inv.ravel()
    ta = np.zeros((inv.max() + 1, 3)); ba = np.zeros_like(ta)
    np.add.at(ta, inv, np.repeat(t, 3, 0)); np.add.at(ba, inv, np.repeat(b, 3, 0))
    n = vertex_normals(v, f)[f.reshape(-1)]
    tc = _normalize(ta[inv] - (ta[inv] * n).sum(1, keepdims=True) * n)
    sign = np.sign((np.cross(n, tc) * ba[inv]).sum(1, keepdims=True)); sign[sign == 0] = 1
    bc = sign * np.cross(n, tc)
    return np.stack([tc, bc, n], 1).reshape(len(f), 3, 3, 3)


def texel_frames(frames, tri, bary):
    """Interpolate corner frames at texels (tri (N,), bary (N,3)) and re-orthonormalise -> (N, 3, 3) rows T, B, N."""
    fr = (frames[tri] * bary[:, :, None, None]).sum(1)
    n = _normalize(fr[:, 2]); t = _normalize(fr[:, 0] - (fr[:, 0] * n).sum(1, keepdims=True) * n)
    sign = np.sign((np.cross(n, t) * fr[:, 1]).sum(1, keepdims=True)); sign[sign == 0] = 1
    return np.stack([t, sign * np.cross(n, t), n], 1)


def gnm_raster(res):
    """GNM skin UV rasterisation (triangle id, barycentrics per texel; rows = v * res), cached in binding_v1."""
    path = OUT_DIR / f'gnm_raster_{res}.npz'
    if path.is_file():
        z = np.load(path); return z['tri'], z['bary']
    from ..rig.bake import rasterize_uv
    g = load_gnm()
    uv = g['triangle_uvs'][g['skin_tri_mask']].reshape(-1, 2)
    tid, bary = rasterize_uv(uv, np.arange(len(uv)).reshape(-1, 3), res)
    np.savez_compressed(path, tri=tid.astype(np.int32), bary=bary.astype(np.float32))
    return tid, bary


def decode(img):
    """uint8/float [0,1] RGB(A) -> unit vectors (H, W, 3)."""
    x = np.asarray(img, np.float64)
    if x.max() > 1.5:
        x = x / 255.
    return _normalize(x[..., :3] * 2 - 1)


def encode(n):
    return np.uint8(np.clip(np.rint((np.asarray(n) * .5 + .5) * 255), 0, 255))


def smooth_transfer_frame(normals, base, valid, weight, sigma_px):
    """Smooth only the transferred mesh-normal field, retaining texture detail.

    The shortest rotation from each original base normal to its normalized,
    validity-weighted Gaussian estimate is applied to the full tangent normal.
    Weight confines the adjustment to an authored region. This is an optional
    look experiment: it can change genuine shape-dependent shading.
    """
    normals, base, weight = (np.asarray(x, np.float32) for x in (normals, base, weight))
    valid = np.asarray(valid, bool)
    if normals.ndim != 3 or normals.shape[-1] != 3 or base.shape != normals.shape:
        raise ValueError('matching HxWx3 normals required')
    if valid.shape != normals.shape[:2] or weight.shape != valid.shape:
        raise ValueError('validity and weight must match normal dimensions')
    if not all(np.isfinite(x).all() for x in (normals, base, weight)) or (weight < 0).any() or (weight > 1).any():
        raise ValueError('finite normals and region weights in [0,1] required')
    if not np.isfinite(sigma_px) or not 0 <= sigma_px <= 16:
        raise ValueError('frame smoothing sigma must be in [0,16] pixels')
    if sigma_px < 1e-6 or not np.any(valid & (weight > 0)):
        return normals.copy()
    from scipy.ndimage import gaussian_filter
    valid_f = valid.astype(np.float32)
    total = gaussian_filter(valid_f, sigma_px, mode='nearest')
    target = gaussian_filter(base * valid_f[..., None], sigma=(sigma_px, sigma_px, 0), mode='nearest')
    target /= np.maximum(total[..., None], 1e-8)
    target /= np.maximum(np.linalg.norm(target, axis=-1, keepdims=True), 1e-8)
    base = base / np.maximum(np.linalg.norm(base, axis=-1, keepdims=True), 1e-8)
    cross = np.cross(base, target)
    cosine = np.clip((base * target).sum(-1), -1, 1)
    first = np.cross(cross, normals)
    rotated = normals + first + np.cross(cross, first) / np.maximum(1 + cosine[..., None], 1e-5)
    rotated /= np.maximum(np.linalg.norm(rotated, axis=-1, keepdims=True), 1e-8)
    amount = weight * valid * (cosine > -.99) * (total > 1e-8)
    mixed = normals * (1 - amount[..., None]) + rotated * amount[..., None]
    mixed /= np.maximum(np.linalg.norm(mixed, axis=-1, keepdims=True), 1e-8)
    return np.where(amount[..., None] > 0, mixed, normals)


def convert(mh_normal, mh_vertices, gnm_skin_vertices, tmap, res=None, *, frame_sigma_px=0., frame_weight=None):
    """MH tangent normal map (MH UV) -> GNM skin_normal (res x res x 3 unit vectors, Blender convention) and valid mask.

    mh_vertices: that identity's MH head (24049, 3, vhuman frame); gnm_skin_vertices: its bound GNM skin vertices
    (binding.evaluate). tmap: binding_v1/texel_map_<res>.npz contents.
    Optional frame smoothing requires an explicit atlas region weight and does
    not alter texels outside it. The default preserves the original conversion.
    """
    res = res or tmap['valid'].shape[0]
    if not np.isfinite(frame_sigma_px) or not 0 <= frame_sigma_px <= 16:
        raise ValueError('frame smoothing sigma must be in [0,16] pixels')
    if frame_sigma_px:
        if frame_weight is None:
            raise ValueError('frame smoothing requires an explicit region weight')
        weight = np.asarray(frame_weight)
        if weight.shape != (res, res) or not np.isfinite(weight).all() or (weight < 0).any() or (weight > 1).any():
            raise ValueError('frame weight must match the atlas and be finite in [0,1]')
    mh = mh_template.load(); g = load_gnm()
    sampled, ok = resample(np.asarray(mh_normal)[..., :3], tmap)
    n_mh = decode(sampled)
    tri_g, bary_g = gnm_raster(res)
    ok &= (tri_g >= 0) & (tmap['mh_tri'] >= 0)
    yy, xx = np.nonzero(ok)
    f_mh = texel_frames(corner_frames(mh_vertices, mh['triangles'], mh['triangle_uvs']), tmap['mh_tri'][yy, xx], tmap['mh_bary'][yy, xx].astype(np.float64))
    uv_g = g['triangle_uvs'][g['skin_tri_mask']]
    f_g = texel_frames(corner_frames(gnm_skin_vertices, g['skin_tris'], uv_g), tri_g[yy, xx], bary_g[yy, xx].astype(np.float64))
    world = np.einsum('nk,nkj->nj', n_mh[yy, xx], f_mh)                 # x T + y B_row + z N  (MH reading: green = +row)
    local = np.einsum('nkj,nj->nk', f_g, world)                          # components along GNM T, B, N
    # Blender reads green along +V (up): B_up = -dpos/drow. corner_frames' B follows +row, so flip green.
    local[:, 1] *= -1
    out = np.tile(FLAT, (res, res, 1)); out[yy, xx] = _normalize(local)
    if frame_sigma_px >= 1e-6:
        base_local = np.einsum('nkj,nj->nk', f_g, f_mh[:, 2])
        base_local[:, 1] *= -1
        base = np.tile(FLAT, (res, res, 1)); base[yy, xx] = _normalize(base_local)
        smoothed = smooth_transfer_frame(out, base, ok, frame_weight, frame_sigma_px)
        selected = ok & (weight > 0)
        out[selected] = smoothed[selected]
    return out, ok


def check(sid, converted=None):
    """Convention self-checks on one sample: MH divergence/cavity signs and height skew, and the same tests on the
    converted map read with Blender's convention (green = +V up -> -row)."""
    from PIL import Image
    from scipy import ndimage, stats
    from .samples import V1
    d = V1 / 'samples' / sid / 'tex'
    n = decode(np.asarray(Image.open(d / 'normal.png')))
    cav = np.asarray(Image.open(d / 'cavity.png').convert('RGB'), np.float64).mean(-1) / 255

    def tests(nx, ny_row, c, mask):
        hp = c - ndimage.gaussian_filter(c, 8)
        cx = np.corrcoef(np.gradient(nx, axis=1)[mask], hp[mask])[0, 1]; cy = np.corrcoef(np.gradient(ny_row, axis=0)[mask], hp[mask])[0, 1]
        z = np.ones_like(nx); gx, gy = -nx / z, -ny_row / z
        hh, ww = gx.shape; fy = np.fft.fftfreq(hh)[:, None]; fx = np.fft.fftfreq(ww)[None, :]
        den = 4 * np.pi ** 2 * (fx ** 2 + fy ** 2); den[0, 0] = 1
        h = np.real(np.fft.ifft2((-2j * np.pi * fx * np.fft.fft2(gx) - 2j * np.pi * fy * np.fft.fft2(gy)) / den))
        hp_h = (h - ndimage.gaussian_filter(h, 6))[mask]
        return dict(corr_red=float(cx), corr_green=float(cy), height_skew=float(stats.skew(hp_h)))
    rec = dict(mh=tests(n[..., 0], n[..., 1], cav, np.ones(cav.shape, bool)))
    path = Path(converted) if converted else Path(V1).parent / 'compare_v1' / sid / 'head' / 'reconstruction' / 'mh' / 'skin_normal.png'
    conv = np.asarray(Image.open(path)) if path.is_file() else None
    if conv is not None and conv.shape[0] > 256 and not (conv[..., :3] == (128, 128, 255)).all():
        nc = decode(conv)
        tm = dict(np.load(OUT_DIR / f'texel_map_{nc.shape[0]}.npz'))
        cav_g, ok = resample(np.asarray(Image.open(d / 'cavity.png').convert('RGB')), tm)
        ok = ndimage.binary_erosion(ok, iterations=4)
        rec['gnm'] = tests(nc[..., 0], -nc[..., 1], cav_g.mean(-1), ok)   # green +V up = -row
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('sid'); ap.add_argument('--res', type=int, default=2048); ap.add_argument('--out')
    ap.add_argument('--check', action='store_true'); ap.add_argument('--converted', help='--check: converted map to test')
    ap.add_argument('--frame-sigma-px', type=float, default=0., help='experimental smoothing of transferred mesh normals in atlas pixels')
    ap.add_argument('--orbital-mask-npz', help='skin_detail.npz supplying the orbital region weights')
    a = ap.parse_args()
    if not np.isfinite(a.frame_sigma_px) or not 0 <= a.frame_sigma_px <= 16:
        ap.error('--frame-sigma-px must be in [0,16]')
    if a.frame_sigma_px and not a.orbital_mask_npz:
        ap.error('--frame-sigma-px requires --orbital-mask-npz')
    if a.check:
        print(json.dumps(check(a.sid, a.converted), indent=1)); return
    from PIL import Image
    from .binding import evaluate, load_binding
    from .samples import V1, load_sample, to_vhuman
    weight = None
    if a.frame_sigma_px:
        with np.load(a.orbital_mask_npz, allow_pickle=False) as z:
            regions = z['region_masks']
            if regions.ndim != 3 or regions.shape[0] < 4 or regions.shape[1:] != (a.res, a.res):
                ap.error('orbital mask dimensions must match --res')
            weight = np.clip(regions[2] + regions[3], 0, 1)
            if not np.isfinite(weight).all():
                ap.error('orbital mask must be finite')
    s = load_sample(V1 / 'samples' / a.sid)
    mh_v = to_vhuman(s['face']['head'][0]).astype(np.float64)
    skin = evaluate(load_binding(), mh_v)
    tm = dict(np.load(OUT_DIR / f'texel_map_{a.res}.npz'))
    n, ok = convert(np.asarray(Image.open(V1 / 'samples' / a.sid / 'tex' / 'normal.png')), mh_v, skin, tm,
                    frame_sigma_px=a.frame_sigma_px, frame_weight=weight)
    out = Path(a.out or f'{a.sid}_skin_normal.png')
    Image.fromarray(encode(n)).save(out)
    print(out, 'valid', float(ok.mean()))


if __name__ == '__main__':
    main()
