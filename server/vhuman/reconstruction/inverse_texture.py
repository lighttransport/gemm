"""Inverse texture optimisation: one shared UV albedo explains every view (image -> texel), instead of the
texel -> image point-sampled weighted average of materials.bake_portrait.

Forward model per pixel: pixel -> UV (PyTorch3D raster) -> mip-filtered sample of the albedo A (the mip level follows the
pixel's UV footprint, so a pixel that spans many texels only constrains their mean) -> times per-view SH9 RGB irradiance.
Unknowns: a multi-resolution correction of the starting bake A0 and per-view SH9 lighting (exposure and white balance
included). Texels the front portrait resolves well are pinned to A0 byte for byte; texels seen only through large footprints
fall back to a low-passed A0 as the leash target; the correction field is smooth over the surface, including across UV cuts.

Diffuse only in this version: highlights are masked, not modelled; no AO, no per-view warp, no LPIPS (see
seamless-uv-texture-plan-2026-11.md, stages 2-3).

usage: python -m server.vhuman.reconstruction.inverse_texture <candidate_dir> <out_dir> [--iters N] [--holdout I ...]
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn
from PIL import Image
from scipy.ndimage import distance_transform_edt

from ..rig import bake
from ..rig.common import vertex_normals
from .reference import Camera, srgb_to_linear

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
MIP_LEVELS = 7          # 2048 .. 32 texels
TRUST_FACTOR = 2.0      # a view resolves a texel when one pixel spans at most this many x the front view's typical footprint
PIN_FACTOR = 2.0        # front texels are pinned up to this many x the typical (head-on) front footprint


def sh9(n):
    x, y, z = n[..., 0], n[..., 1], n[..., 2]
    return torch.stack((torch.ones_like(x), x, y, z, x * y, x * z, y * z, x * x - y * y, 3 * z * z - 1), -1)


def to_tone(x):
    return x.clamp_min(1e-4) ** (1 / 2.2)


def raster_view(vertices, triangles, tri_uvs, cam, hw):
    """Per-pixel UV, world normal, view direction, face id and UV footprint (texels per pixel, before x res)."""
    from pytorch3d.ops import interpolate_face_attributes
    from pytorch3d.renderer import MeshRasterizer, PerspectiveCameras, RasterizationSettings
    from pytorch3d.structures import Meshes
    H, W = hw
    SR = np.diag([-1., 1., -1.]) @ cam.rotation       # reference.Camera -> PyTorch3D screen convention (checked: IoU .99)
    R = torch.tensor(SR.T, dtype=torch.float32, device=DEV)[None]
    T = torch.tensor(-(SR @ cam.origin), dtype=torch.float32, device=DEV)[None]
    fy = cam.focal if cam.focal_y is None else cam.focal_y
    cams = PerspectiveCameras(focal_length=((cam.focal, fy),), principal_point=((cam.cx, cam.cy),), R=R, T=T,
                              image_size=((H, W),), in_ndc=False, device=DEV)
    V = torch.tensor(vertices, dtype=torch.float32, device=DEV)
    F = torch.tensor(triangles, dtype=torch.int64, device=DEV)
    frag = MeshRasterizer(cameras=cams, raster_settings=RasterizationSettings(
        image_size=(H, W), blur_radius=0., faces_per_pixel=1))(Meshes(verts=[V], faces=[F]))
    pix = frag.pix_to_face[0, ..., 0]
    vn = torch.tensor(vertex_normals(vertices, triangles), dtype=torch.float32, device=DEV)
    attr = torch.cat((torch.tensor(tri_uvs, dtype=torch.float32, device=DEV), vn[F], V[F]), -1)   # (F,3,8)
    got = interpolate_face_attributes(frag.pix_to_face, frag.bary_coords, attr)[0, ..., 0, :]
    mask = pix >= 0
    uv, n, p = got[..., 0:2], got[..., 2:5], got[..., 5:8]
    n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    origin = torch.tensor(cam.origin, dtype=torch.float32, device=DEV)
    vd = origin - p
    vd = vd / vd.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    # UV jacobian from neighbouring pixels of the same face (forward, else backward difference)
    def diff(axis):
        d = torch.zeros_like(uv); ok = torch.zeros_like(mask)
        a = uv.narrow(axis, 1, uv.shape[axis] - 1) - uv.narrow(axis, 0, uv.shape[axis] - 1)
        same = (pix.narrow(axis - 0, 1, pix.shape[axis] - 1) == pix.narrow(axis - 0, 0, pix.shape[axis] - 1)) & \
               mask.narrow(axis, 1, mask.shape[axis] - 1)
        fwd = [slice(None)] * 3; bwd = [slice(None)] * 3
        fwd[axis] = slice(0, -1); bwd[axis] = slice(1, None)
        d[tuple(fwd)] = a; ok[tuple(fwd[:2])] = same
        d2 = torch.zeros_like(uv); ok2 = torch.zeros_like(mask)
        d2[tuple(bwd)] = a; ok2[tuple(bwd[:2])] = same
        return torch.where(ok[..., None], d, d2), ok | ok2
    dx, okx = diff(1)
    dy, oky = diff(0)
    S = (dx ** 2).sum(-1) + (dy ** 2).sum(-1)
    det = dx[..., 0] * dy[..., 1] - dx[..., 1] * dy[..., 0]
    smax = torch.sqrt(((S + torch.sqrt((S ** 2 - 4 * det ** 2).clamp_min(0))) / 2).clamp_min(0))
    foot = torch.where(okx & oky, smax, torch.full_like(smax, float('nan')))
    return dict(mask=mask, uv=uv, n=n, vd=vd, face=pix, foot=foot)


def render_surface(g, vi, ears=True):
    """Skin surface as rendered (refined ear included); see ear_mesh.rendered_surface."""
    if not ears:
        return g['captured'][vi], g['triangles'], g['triangle_uvs']
    from .ear_mesh import rendered_surface
    return rendered_surface(g, vi)


def load_view(c, i, hw):
    img = np.asarray(Image.open(c / f'view_{i}.png').convert('RGBA'))
    ex = c / f'exclusion_{i}.png'
    support = img[..., 3].astype(float) / 255
    if ex.is_file():
        support = support * (1 - np.clip(np.asarray(Image.open(ex).convert('L')).astype(float) / 255, 0, 1))
    rgb = img[..., :3].astype(float)
    border = np.concatenate([rgb[:8].reshape(-1, 3), rgb[-8:].reshape(-1, 3), rgb[:, :8].reshape(-1, 3), rgb[:, -8:].reshape(-1, 3)])
    near = np.abs(rgb - np.median(border, 0)).max(2) < 18
    if near.mean() > .02:
        from scipy.ndimage import binary_dilation
        support = support * ~binary_dilation(near, iterations=3)
    return srgb_to_linear(rgb / 255.), support


def seam_pairs(triangles, tri_uvs, res):
    """Texel pairs along every mesh edge whose two faces disagree on UV (the cuts); None if there are none."""
    edges = {}
    for t, (a, b, c) in enumerate(triangles):
        for (p, q), (i, j) in (((a, b), (0, 1)), ((b, c), (1, 2)), ((c, a), (2, 0))):
            e = (min(p, q), max(p, q))
            seg = (tri_uvs[t, i], tri_uvs[t, j]) if p < q else (tri_uvs[t, j], tri_uvs[t, i])
            edges.setdefault(e, []).append(seg)
    A, B = [], []
    for segs in edges.values():
        if len(segs) == 2 and (np.abs(segs[0][0] - segs[1][0]).max() > 1e-6 or np.abs(segs[0][1] - segs[1][1]).max() > 1e-6):
            n = int(np.ceil(np.linalg.norm((segs[0][1] - segs[0][0]) * res))) + 2
            t = np.linspace(0, 1, n)[:, None]
            A.append(segs[0][0] + (segs[0][1] - segs[0][0]) * t); B.append(segs[1][0] + (segs[1][1] - segs[1][0]) * t)
    if not A:
        return None
    def idx(uv):
        return np.clip(np.floor(uv * res), 0, res - 1).astype(int)
    a, b = idx(np.concatenate(A)), idx(np.concatenate(B))
    return torch.tensor(a[:, 1] * res + a[:, 0], device=DEV), torch.tensor(b[:, 1] * res + b[:, 0], device=DEV)


def blur(x, k=17, rounds=2):
    for _ in range(rounds):
        x = Fn.avg_pool2d(Fn.pad(x, (k // 2,) * 4, mode='replicate'), k, 1)
    return x


def sample_mips(mips, grid, lam):
    """Trilinear sample of a mip chain at grid (N,2 in [-1,1]) with fractional level lam (N)."""
    g = grid[None, :, None, :]
    stack = torch.stack([Fn.grid_sample(m, g, mode='bilinear', padding_mode='border', align_corners=False)[0, :, :, 0].T for m in mips])
    lam = lam.clamp(0, len(mips) - 1 - 1e-4)
    lo = lam.floor().long(); t = (lam - lo)[:, None]
    ar = torch.arange(len(lam), device=grid.device)
    return stack[lo, ar] * (1 - t) + stack[lo + 1, ar] * t


def fit_light(A_s, sh, target, w):
    """Per-channel weighted least squares: target_c = A_c * (sh . L_c)."""
    L = []
    for c in range(3):
        M = sh * A_s[:, c:c + 1] * w.sqrt()[:, None]
        L.append(torch.linalg.lstsq(M.double().cpu(), (target[:, c] * w.sqrt()).double().cpu()[:, None]).solution[:, 0].float())
    return torch.stack(L, -1).to(sh.device)


def solve(candidate, out, *, iters=400, holdout=(), view_weight=(1., .5, .5), lam_smooth=30., lam_leash=.5, pin_factor=PIN_FACTOR, warp=True, ear_surface=True, ear_prior='none', prior_weight=2., lam_lpips=0., lpips_crops=2, lr=.003, seed=0):
    torch.manual_seed(seed)
    c, out = Path(candidate), Path(out)
    m = json.loads((c / 'manifest.json').read_text())
    g = np.load(c / 'geometry.npz')
    surfaces = [render_surface(g, vi, ear_surface) for vi in range(len(m['geometry']['fitted_cameras']))]
    tris, tri_uvs = surfaces[0][1], surfaces[0][2]
    cams = [Camera.from_dict(x) for x in m['geometry']['fitted_cameras']]
    base0 = np.asarray(Image.open(c / 'skin_basecolor.png').convert('RGB'))
    res = base0.shape[0]
    A0 = torch.tensor(srgb_to_linear(base0 / 255.), dtype=torch.float32, device=DEV).permute(2, 0, 1)[None]
    conf = torch.tensor(np.asarray(Image.open(c / 'skin_confidence.png').convert('L')), device=DEV).float() / 255
    prot_img = c / 'skin_front_protected.png'
    protected = torch.tensor(np.asarray(Image.open(prot_img).convert('L')) > 127, device=DEV) if prot_img.is_file() else torch.zeros(res, res, dtype=torch.bool, device=DEV)
    ids, _ = bake.rasterize_uv(tri_uvs.reshape(-1, 2), np.arange(len(tri_uvs) * 3).reshape(-1, 3), res)
    covered = torch.tensor(ids >= 0, device=DEV)

    views = []
    for vi, cam in enumerate(cams):
        rgb, support = load_view(c, vi, (1024, 1024))
        H, W = rgb.shape[:2]
        r = raster_view(*surfaces[vi][:2], surfaces[vi][2], cam, (H, W))
        mask = r['mask'].cpu().numpy()
        dist = distance_transform_edt(mask)
        facing = (r['n'] * r['vd']).sum(-1).clamp(0, 1)
        w = torch.tensor(np.clip((dist - 3) / 3, 0, 1) * (support > .5) * (rgb.max(2) < .9) * (rgb.mean(2) > .02), dtype=torch.float32, device=DEV)
        w = w * r['mask'] * torch.clamp((facing - .1) / .3, 0, 1)
        sel = torch.nonzero(w.reshape(-1) > 0).squeeze(1)
        uv = r['uv'].reshape(-1, 2)[sel]
        foot = r['foot'].reshape(-1)[sel] * res
        foot = torch.nan_to_num(foot, nan=1.)
        cols, rows = (sel % W).float() + .5, (sel // W).float() + .5
        views.append(dict(H=H, W=W, img=torch.tensor(rgb, dtype=torch.float32, device=DEV).permute(2, 0, 1)[None],
                          pxn=torch.stack((cols / W * 2 - 1, rows / H * 2 - 1), -1), sel=sel, uv=uv, grid=uv * 2 - 1, lam=torch.log2(foot.clamp(1, 2 ** (MIP_LEVELS - 1))), foot=foot,
                          sh=sh9(r['n'].reshape(-1, 3)[sel]), tgt=torch.tensor(rgb.reshape(-1, 3), dtype=torch.float32, device=DEV)[sel],
                          facing=facing.reshape(-1)[sel], w=w.reshape(-1)[sel] * view_weight[min(vi, len(view_weight) - 1)], face=r['face'].reshape(-1)[sel],
                          tex=(torch.clamp((uv[:, 1] * res).floor().long(), 0, res - 1) * res + torch.clamp((uv[:, 0] * res).floor().long(), 0, res - 1))))

    # per-texel evidence: best footprint, hit counts, summed weight per view (also the stage-0 debug maps)
    best = torch.full((res * res,), float('inf'), device=DEV)
    wtex = torch.zeros(len(views), res * res, device=DEV)
    hit0 = torch.zeros(res * res, device=DEV)
    for vi, v in enumerate(views):
        if vi in holdout:          # a held-out view must not count as evidence (it would mark its texels resolved)
            continue
        best.scatter_reduce_(0, v['tex'], v['foot'], 'amin')
        wtex[vi].scatter_add_(0, v['tex'], v['w'])
    hit0.scatter_add_(0, views[0]['tex'], torch.ones_like(views[0]['w']))
    best = best.reshape(res, res)
    wtex = wtex.reshape(len(views), res, res)
    hit0 = hit0.reshape(res, res)
    head_on = views[0]['facing'] > .8
    med0 = float(views[0]['foot'][head_on].median()) if head_on.any() else 1.
    print('front head-on footprint (texels per pixel):', med0, flush=True)
    good = (best <= TRUST_FACTOR * med0)
    flagged = torch.isfinite(best) & ~good
    stretched = (Fn.max_pool2d(flagged.float()[None, None], 15, 1, 7)[0, 0] > 0) & ~(Fn.max_pool2d(good.float()[None, None], 5, 1, 2)[0, 0] > 0) & covered
    dens = Fn.avg_pool2d((hit0 > 0).float()[None, None], 7, 1, 3)[0, 0]
    pinned = protected & (conf > .5) & (Fn.max_pool2d(torch.nan_to_num(views_best_view0(views[0], res), posinf=0.)[None, None], 5, 1, 2)[0, 0] <= pin_factor * med0) & (dens > .5 / max(med0, 1.) ** 2 * .7) & covered
    ref = torch.where(stretched[None, None], blur(A0), A0)
    leash_w = torch.full((res, res), float(lam_leash), device=DEV)
    prior_report = None
    if ear_prior != 'none':
        em = c / 'ear_rebake_mask.png'
        region = (Fn.max_pool2d(torch.tensor(np.asarray(Image.open(em).convert('L')) > 127, device=DEV).float()[None, None], 9, 1, 4)[0, 0] > 0) & covered
        # resolved = some view sees the texel at a usable footprint (pinned front texels count too)
        resolved = (Fn.max_pool2d(good.float()[None, None], 5, 1, 2)[0, 0] > 0) | pinned
        mref, usable = mirror_reference(A0, resolved, region)
        usable = usable & ~pinned
        if ear_prior == 'mirror-lf':
            mref = blur(mref, 31)                     # refined ears differ in shape per side: transfer only the smooth tint/shading
        ref = torch.where(usable[None, None], mref, ref)
        leash_w = torch.where(usable, torch.full_like(leash_w, float(prior_weight)), leash_w)
        prior_report = dict(mode=ear_prior, usable_texels=int(usable.sum()), region_texels=int(region.sum()), resolved_in_region=int((resolved & region).sum()))
        out.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.uint8(usable.cpu().numpy()) * 255).save(out / 'prior_usable.png')

    # parameters: multiresolution correction + per-view lighting
    sizes = [64, 128, 256, 512, 1024, res]
    deltas = [torch.zeros(1, 3, s, s, device=DEV, requires_grad=True) for s in sizes]
    train = [i for i in range(len(views)) if i not in holdout]
    Ls = []
    for v in views:
        v['A0'] = sample_mips(mips_of(A0), v['grid'], v['lam'])
        Ls.append(fit_light(v['A0'], v['sh'], v['tgt'], v['w'].clamp_min(1e-6)).requires_grad_(len(Ls) > 0))
    warps = [torch.zeros(1, 2, WARP_GRID, WARP_GRID, device=DEV, requires_grad=bool(warp and i != 0 and i in train)) for i in range(len(views))]
    groups = [{'params': deltas, 'lr': lr}, {'params': [Ls[i] for i in train if i != 0], 'lr': lr / 2}]
    if warp:
        groups.append({'params': [warps[i] for i in train if i != 0], 'lr': lr * 3})
    opt = torch.optim.Adam(groups)   # view 0's lighting stays at its fit to A0: fixes the albedo/light gauge
    pa, pb = seam_pairs(tris, tri_uvs, res) or (None, None)
    pin3 = pinned[None, None]

    def albedo():
        d = sum(Fn.interpolate(x, size=(res, res), mode='bilinear', align_corners=False) for x in deltas)
        a = (A0 + d).clamp(0, 1)
        return torch.where(pin3, A0, a), d

    hist = []
    for it in range(iters):
        A, d = albedo()
        mips = mips_of(A)
        data = 0.; lp = 0.; wreg = 0.
        for i in train:
            v = views[i]
            a = sample_mips(mips, v['grid'], v['lam'])
            pred = a * (v['sh'] @ Ls[i]).clamp_min(.02)
            if warps[i].requires_grad:
                tgt, tdense, disp = warped_target(v, warps[i])
                wreg = wreg + ((disp[..., 1:] - disp[..., :-1]) ** 2).mean() * 1e3 + (disp ** 2).mean() * 10
            else:
                tgt, tdense = v['tgt'], None
            e = Fn.huber_loss(to_tone(pred), to_tone(tgt), reduction='none', delta=.05).mean(-1)
            data = data + (v['w'] * e).sum() / v['w'].sum()
            if lam_lpips > 0:
                if tdense is None:
                    tdense = v['img']
                pimg, m = dense_pred(v, pred, tdense)
                lp = lp + crop_lpips(pimg, tdense, m, random_boxes(v, lpips_crops))
        dd = A - A0
        smooth = ((dd[..., 1:] - dd[..., :-1]) ** 2 * (covered[:, 1:] & covered[:, :-1])).mean() + ((dd[..., 1:, :] - dd[..., :-1, :]) ** 2 * (covered[1:] & covered[:-1])).mean()
        if pa is not None:
            fa, fb = dd.reshape(3, -1)[:, pa], dd.reshape(3, -1)[:, pb]
            smooth = smooth + ((fa - fb) ** 2).mean()
        leash = (((A - ref) ** 2).sum(1) * leash_w * covered * ~pinned).sum() / covered.sum() / max(lam_leash, 1e-9)
        light_reg = sum((Ls[i][1:] ** 2).mean() for i in train) * .01
        loss = data + lam_smooth * smooth + lam_leash * leash + light_reg + lam_lpips * lp + wreg
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 50 == 0 or it == iters - 1:
            hist.append(dict(it=it, loss=float(loss), data=float(data), lpips=float(lp), wreg=float(wreg), smooth=float(smooth), leash=float(leash)))
            print(hist[-1], flush=True)

    ear_flat = torch.tensor(np.asarray(Image.open(c / 'ear_rebake_mask.png').convert('L')) > 127, device=DEV).float().reshape(-1)
    with torch.no_grad():
        A, _ = albedo()
        mips0, mips1 = mips_of(A0), mips_of(A)
        report = {}
        for i, v in enumerate(views):
            row = {}
            for name, mp in (('start', mips0), ('solved', mips1)):
                a = sample_mips(mp, v['grid'], v['lam'])
                L = fit_light(a, v['sh'], v['tgt'], v['w'].clamp_min(1e-6))      # refit so held-out views are judged fairly
                e = (to_tone(a * (v['sh'] @ L).clamp_min(.02)) - to_tone(v['tgt'])).abs().mean(-1)
                row[name] = float((v['w'] * e).sum() / v['w'].sum())
                we = v['w'] * ear_flat[v['tex']]
                if float(we.sum()) > 50:
                    row['ear_' + name] = float((we * e).sum() / we.sum()); row['ear_pixels'] = int((we > 0).sum())
            if True:
                boxes = grid_boxes(v)
                for name, mp in (('start', mips0), ('solved', mips1)):
                    a = sample_mips(mp, v['grid'], v['lam'])
                    L = fit_light(a, v['sh'], v['tgt'], v['w'].clamp_min(1e-6))
                    pimg, m = dense_pred(v, a * (v['sh'] @ L).clamp_min(.02), v['img'])
                    row['lpips_' + name] = float(crop_lpips(pimg, v['img'], m, boxes)) if boxes else None
                if warps[i].requires_grad or float(warps[i].abs().sum()) > 0:
                    d = warp_field(warps[i], v['H'], v['W'])[0] * (v['W'] / 2)
                    row['warp_px_median'] = float(d.norm(dim=0).median()); row['warp_px_max'] = float(d.norm(dim=0).max())
            report[f'view_{i}'] = dict(row, holdout=i in holdout)
    from .reference import linear_to_srgb
    img = np.uint8(np.clip(linear_to_srgb(A[0].permute(1, 2, 0).cpu().numpy()) * 255 + .5, 0, 255))
    pin_np = pinned.cpu().numpy()
    img[pin_np] = base0[pin_np]                                   # byte-exact where pinned
    out.mkdir(parents=True, exist_ok=True)
    for f in c.iterdir():
        if f.is_file() and f.name != 'skin_basecolor.png':
            shutil.copy2(f, out / f.name)
    Image.fromarray(img).save(out / 'skin_basecolor.png')
    dbg = out / 'debug'; dbg.mkdir(exist_ok=True)
    palette = np.array([[255, 60, 60], [60, 160, 255], [80, 220, 100]], np.uint8)
    wsum = wtex.sum(0)
    src = palette[wtex.argmax(0).cpu().numpy() % 3]
    src = (src * (wsum.cpu().numpy() > 0)[..., None]).astype(np.uint8)
    Image.fromarray(src).save(dbg / 'source_view_id.png')
    fp = np.log2(np.clip(torch.nan_to_num(best, posinf=64.).cpu().numpy(), 1, 64)) / 6
    Image.fromarray(np.uint8(fp * 255)).save(dbg / 'best_footprint_log2.png')
    Image.fromarray(np.uint8(pin_np) * 255).save(dbg / 'pinned.png')
    Image.fromarray(np.uint8(stretched.cpu().numpy()) * 255).save(dbg / 'stretched.png')
    Image.fromarray(np.uint8(np.clip(np.abs(img.astype(float) - base0).max(2) * 4, 0, 255))).save(dbg / 'change_x4.png')
    stats = dict(ear_prior=prior_report, front_footprint_texels_per_pixel=med0, pinned_texels=int(pin_np.sum()), stretched_texels=int(stretched.sum()), covered=int(covered.sum()),
                 pinned_bytes_equal=bool((img[pin_np] == base0[pin_np]).all()),
                 changed_texels_gt2=int((np.abs(img.astype(int) - base0.astype(int)).max(2) > 2).sum()),
                 light=[L.detach().cpu().tolist() for L in Ls], warp=[w.detach().cpu().tolist() for w in warps])
    (out / 'inverse_texture.json').write_text(json.dumps(dict(iters=iters, holdout=list(holdout), view_weight=view_weight, lam_smooth=lam_smooth,
                                                                lam_leash=lam_leash, history=hist, reprojection_residual=report, stats=stats), indent=1))
    print(json.dumps(dict(reprojection_residual=report, stats={k: v for k, v in stats.items() if k not in ('light', 'warp')}), indent=1))
    return report, stats


WARP_GRID = 16
WARP_MAX = .05          # normalised units, = 25 px at 1024


def warp_field(raw, H, W):
    """(1,2,G,G) raw parameters -> dense normalised displacement (1,2,H,W), bounded by tanh."""
    return Fn.interpolate(torch.tanh(raw) * WARP_MAX, size=(H, W), mode='bicubic', align_corners=True)


def warped_target(v, raw):
    """Target image sampled at the warped pixel positions: per selected pixel (N,3) and dense (1,3,H,W)."""
    H, W = v['H'], v['W']
    disp = warp_field(raw, H, W)
    ys, xs = torch.meshgrid(torch.linspace(-1 + 1 / H, 1 - 1 / H, H, device=DEV), torch.linspace(-1 + 1 / W, 1 - 1 / W, W, device=DEV), indexing='ij')
    base = torch.stack((xs, ys), -1)[None]
    dense = Fn.grid_sample(v['img'], base + disp.permute(0, 2, 3, 1), mode='bilinear', padding_mode='border', align_corners=False)
    return dense[0].reshape(3, -1)[:, v['sel']].T, dense, disp


_LPIPS = None


def lpips_net():
    global _LPIPS
    if _LPIPS is None:
        import lpips
        _LPIPS = lpips.LPIPS(net='alex', verbose=False).to(DEV).eval()
        for q in _LPIPS.parameters():
            q.requires_grad_(False)
    return _LPIPS


def dense_pred(v, pred, tgt_dense):
    """Predicted image with unobserved pixels copied from the target, so only observed pixels contribute to LPIPS."""
    flat = torch.zeros(3, v['H'] * v['W'], device=DEV).index_copy(1, v['sel'], pred.T)
    m = torch.zeros(1, v['H'] * v['W'], device=DEV); m[:, v['sel']] = 1
    m = m.reshape(1, 1, v['H'], v['W'])
    return torch.where(m > 0, flat.reshape(1, 3, v['H'], v['W']), tgt_dense), m


def crop_lpips(pred_img, tgt_img, m, boxes, size=256):
    net = lpips_net()
    a = torch.cat([pred_img[..., y:y + size, x:x + size] for y, x in boxes]); b = torch.cat([tgt_img[..., y:y + size, x:x + size] for y, x in boxes])
    return net(to_tone(a) * 2 - 1, to_tone(b) * 2 - 1).mean()


def random_boxes(v, k, size=256):
    idx = torch.randint(0, len(v['sel']), (k,), device=DEV)
    y = (v['sel'][idx] // v['W']).clamp(size // 2, v['H'] - size // 2) - size // 2
    x = (v['sel'][idx] % v['W']).clamp(size // 2, v['W'] - size // 2) - size // 2
    return [(int(a), int(b)) for a, b in zip(y, x)]


def grid_boxes(v, size=256, min_cover=.5):
    m = torch.zeros(v['H'] * v['W'], device=DEV); m[v['sel']] = 1; m = m.reshape(v['H'], v['W'])
    return [(y, x) for y in range(0, v['H'] - size + 1, size) for x in range(0, v['W'] - size + 1, size) if float(m[y:y + size, x:x + size].mean()) > min_cover]


def wblur(x, w, k=65, rounds=2):
    """Weighted box-blur normalised by the blurred weight (x: (1,C,H,W), w: (1,1,H,W))."""
    return blur(x * w, k, rounds) / blur(w, k, rounds).clamp_min(1e-6), blur(w, k, rounds)


def mirror_reference(A0, resolved, region, k=65):
    """Left-right mirrored A0 (the GNM atlas mirrors exactly: texel (r,c) <-> (r,W-1-c); mesh symmetric to .25 mm), colour matched by a
    smooth per-channel gain estimated where both sides are resolved. Returns (ref, usable): usable = unresolved here, resolved on the
    mirrored side, inside the region."""
    Am, rm = A0.flip(-1), resolved.flip(-1)
    both = (resolved & rm)[None, None].float() * region[None, None].float()
    num, wn = wblur(A0, both, k)
    den, _ = wblur(Am, both, k)
    gain = torch.where(wn > 1e-3, num / den.clamp_min(1e-4), torch.ones_like(num))
    glob = (A0 * both).sum((2, 3)) / (Am * both).sum((2, 3)).clamp_min(1e-6) if both.sum() > 100 else torch.ones(1, 3, device=A0.device)
    gain = torch.where(wn > 1e-3, gain, glob[..., None, None]).clamp(.6, 1.6)
    usable = ~resolved & rm & region
    return (Am * gain).clamp(0, 1), usable


def mips_of(A):
    out = [A]
    for _ in range(MIP_LEVELS - 1):
        out.append(Fn.avg_pool2d(out[-1], 2))
    return out


def views_best_view0(v, res):
    best = torch.full((res * res,), float('inf'), device=DEV)
    best.scatter_reduce_(0, v['tex'], v['foot'], 'amin')
    return best.reshape(res, res)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('candidate', type=Path); ap.add_argument('out', type=Path)
    ap.add_argument('--iters', type=int, default=400); ap.add_argument('--holdout', type=int, nargs='*', default=[])
    ap.add_argument('--lr', type=float, default=.003); ap.add_argument('--ear-prior', default='none', choices=['none', 'mirror', 'mirror-lf']); ap.add_argument('--prior-weight', type=float, default=2.); ap.add_argument('--no-warp', action='store_true'); ap.add_argument('--no-ear-surface', action='store_true'); ap.add_argument('--lam-lpips', type=float, default=0.); ap.add_argument('--pin-factor', type=float, default=PIN_FACTOR); ap.add_argument('--lam-smooth', type=float, default=30.); ap.add_argument('--lam-leash', type=float, default=.5)
    a = ap.parse_args()
    solve(a.candidate, a.out, iters=a.iters, holdout=tuple(a.holdout), lam_smooth=a.lam_smooth, lam_leash=a.lam_leash, pin_factor=a.pin_factor, warp=not a.no_warp, ear_surface=not a.no_ear_surface, ear_prior=a.ear_prior, prior_weight=a.prior_weight, lam_lpips=a.lam_lpips, lr=a.lr)


if __name__ == '__main__':
    main()
