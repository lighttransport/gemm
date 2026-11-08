"""UE-inspired scene-linear diffusion of skin lighting, preserving albedo/specular.

This screen-space approximation uses a normalized Burley radial diffusion profile.
It is an explicitly labelled look-matching option, not Cycles volume transport.
No eye, background, albedo texture, or specular lobe is blurred.
"""
import numpy as np


def suppress_indirect_spikes(indirect, denoised, ratio=4., floor=.05):
    """Replace extreme positive indirect-light samples using a guided estimate.

    This experimental guard acts on whole RGB pixels to preserve the colour of
    the estimate. ``floor`` may be a scalar or a per-pixel scene-linear threshold.
    """
    indirect, denoised = (np.asarray(x, np.float32) for x in (indirect, denoised))
    if indirect.ndim != 3 or indirect.shape[-1] != 3 or denoised.shape != indirect.shape:
        raise ValueError('matching HxWx3 indirect lighting required')
    if not all(np.isfinite(x).all() for x in (indirect, denoised)):
        raise ValueError('finite indirect lighting required')
    floor = np.asarray(floor, np.float32)
    if floor.ndim and floor.shape != indirect.shape[:2]:
        raise ValueError('floor must be scalar or match image dimensions')
    if not np.isfinite(ratio) or ratio <= 0 or not np.isfinite(floor).all() or (floor < 0).any():
        raise ValueError('positive ratio and nonnegative finite floor required')
    luminance = np.array([.2126, .7152, .0722], np.float32)
    selected = indirect @ luminance > ratio * np.maximum(denoised @ luminance, 0) + floor
    return np.where(selected[..., None], denoised, indirect), selected


def smooth_scatter_correction(correction, mask, depth, normal, sigma_px, depth_scale_m=.0021):
    """Denoise a lighting correction with separable surface-aware Gaussian passes.

    This filters the additive diffusion correction, not the beauty, albedo or
    specular lobes. Skin coverage, depth and geometric normals reject samples
    across surfaces. The finite kernel is limited to an eight-pixel sigma.
    """
    correction, mask, depth, normal = (np.asarray(x, np.float32) for x in (correction, mask, depth, normal))
    if correction.ndim != 3 or correction.shape[-1] != 3 or normal.shape != correction.shape:
        raise ValueError('correction and normal must be matching HxWx3 arrays')
    if mask.shape != correction.shape[:2] or depth.shape != mask.shape:
        raise ValueError('mask and depth must match correction dimensions')
    if not all(np.isfinite(x).all() for x in (correction, mask, depth, normal)):
        raise ValueError('finite correction and surface guides required')
    if not np.isfinite(sigma_px) or not 0 <= sigma_px <= 8 or not np.isfinite(depth_scale_m) or depth_scale_m <= 0:
        raise ValueError('sigma must be in [0,8] pixels and depth scale positive')
    if sigma_px < 1e-6:
        return correction.copy()
    radius=int(np.ceil(3*sigma_px));h,w=mask.shape
    coverage=np.clip(mask,0,1)
    n=normal/np.maximum(np.linalg.norm(normal,axis=-1,keepdims=True),1e-8)
    result=correction.copy()
    for axis in (1,0):
        acc=np.zeros_like(correction);total=np.zeros_like(mask)
        for shift in range(-radius,radius+1):
            if abs(shift) >= (w if axis==1 else h):
                continue
            if axis==1:
                x0=max(0,shift);x1=min(w,w+shift)
                dst=np.s_[:,x0:x1];src=np.s_[:,x0-shift:x1-shift]
            else:
                y0=max(0,shift);y1=min(h,h+shift)
                dst=np.s_[y0:y1,:];src=np.s_[y0-shift:y1-shift,:]
            dz=np.clip((depth[dst]-depth[src])/depth_scale_m,-32,32)
            dot=np.clip((n[dst]*n[src]).sum(-1),0,1)
            weight=np.exp(-.5*(shift/sigma_px)**2)*coverage[src]*np.exp(-dz*dz)*dot**8
            acc[dst]+=result[src]*weight[...,None];total[dst]+=weight
        result=np.where(total[...,None]>1e-8,acc/np.maximum(total[...,None],1e-8),result)
    return result


def scatter_lighting(lighting, mask, depth, normal, focal_px, radius_mm=(1.05, .4069, .2625), samples=48, *, symmetric=False):
    """Filter RGB irradiance in metric space with skin/depth/normal rejection.

    Each RGB channel samples the normalized radial PDF: 1/4 exp(-r/d)/d +
    3/4 exp(-r/(3d))/(3d). Stratified radii and a golden-angle rotation make
    deterministic samples; normalized gather weights preserve constant light.
    """
    import cv2
    lighting = np.asarray(lighting, np.float32)
    mask, depth, normal = (np.asarray(x, np.float32) for x in (mask, depth, normal))
    if lighting.ndim != 3 or lighting.shape[-1] != 3 or normal.shape != lighting.shape:
        raise ValueError('lighting and normals must be HxWx3')
    if mask.shape != lighting.shape[:2] or depth.shape != mask.shape:
        raise ValueError('mask and depth must be HxW')
    radius = np.asarray(radius_mm, np.float32) * .001
    if radius.shape != (3,) or not np.isfinite(radius).all() or (radius < 0).any():
        raise ValueError('radius_mm must be three finite nonnegative distances')
    if not np.isfinite(focal_px) or focal_px <= 0 or samples < 8 or samples % 4:
        raise ValueError('positive focal length and sample count divisible by four required')
    if symmetric and samples % 16:
        raise ValueError('symmetric sampling requires a sample count divisible by sixteen')
    if not all(np.isfinite(x).all() for x in (lighting, mask, depth, normal)):
        raise ValueError('nonfinite scattering inputs')
    mask = np.clip(mask, 0, 1)
    normal = normal / np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-8)
    yy, xx = np.mgrid[:mask.shape[0], :mask.shape[1]].astype(np.float32)
    valid = (mask > 1e-5) & (depth > 0)
    result = lighting.copy()
    def sample(image, x, y):
        return cv2.remap(image, np.asarray(x, np.float32), np.asarray(y, np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    for c, d in enumerate(radius):
        if d <= 0:
            continue
        acc = np.zeros(mask.shape, np.float32)
        total = np.zeros(mask.shape, np.float32)
        for i in range(samples):
            n = samples // 4 if i < samples // 4 else samples * 3 // 4
            j = i if i < samples // 4 else i - samples // 4
            if symmetric:
                length = -np.log1p(-(j // 4 + .5) / (n // 4)) * d * (1 if i < samples // 4 else 3)
                angle = (i // 4) * 2.399963229728653 + (i % 4) * np.pi / 2
            else:
                length = -np.log1p(-(j + .5) / n) * d * (1 if i < samples // 4 else 3)
                angle = i * 2.399963229728653
            px = length * float(focal_px) / np.maximum(depth, .001)
            x = xx + px * np.cos(angle); y = yy + px * np.sin(angle)
            z = sample(depth, x, y)
            nn = sample(normal, x, y)
            ndot = np.maximum((nn * normal).sum(-1), 0.)
            weight = sample(mask, x, y) * np.exp(-((z - depth) / (3 * float(radius.max()) + .0005)) ** 2) * ndot ** 8
            acc += sample(lighting[..., c] * mask, x, y) * weight
            total += sample(mask, x, y) * weight
        result[..., c] = np.where(valid & (total > 1e-8), acc / np.maximum(total, 1e-8), lighting[..., c])
    return result


def recombine_skin(beauty, diffuse_lighting, scattered_lighting, diffuse_color, mask, weight=1.):
    """Replace only the skin's diffuse lighting; preserve every other contribution."""
    beauty = np.asarray(beauty, np.float32)
    diffuse_lighting, scattered_lighting, diffuse_color = (np.asarray(x, np.float32) for x in (diffuse_lighting, scattered_lighting, diffuse_color))
    if beauty.ndim != 3 or beauty.shape[-1] not in (3, 4):
        raise ValueError('beauty must be HxWxRGB(A)')
    if any(x.shape != beauty.shape[:2] + (3,) for x in (diffuse_lighting, scattered_lighting, diffuse_color)):
        raise ValueError('diffuse passes must match beauty RGB shape')
    mask = np.asarray(mask, np.float32)
    if mask.shape != beauty.shape[:2] or not np.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError('matching skin mask and weight in [0,1] required')
    if not all(np.isfinite(x).all() for x in (beauty, diffuse_lighting, scattered_lighting, diffuse_color, mask)):
        raise ValueError('nonfinite recombination inputs')
    result = beauty.copy()
    result[..., :3] += (scattered_lighting - diffuse_lighting) * diffuse_color * (np.clip(mask, 0, 1) * weight)[..., None]
    return result


def preserve_native_regions(native, scattered, profile_mask):
    """Blend a face-lighting treatment over a physically scattered native render.

    profile_mask is an authored material/atlas weight, not a mask inferred from
    the UE target. Zero weights preserve native ears, lips, eyes and background;
    native alpha is retained everywhere. This enables thin-part transmission to
    remain physical when a screen-space face profile is used.
    """
    native, scattered, profile_mask = (np.asarray(x, np.float32) for x in (native, scattered, profile_mask))
    if native.ndim != 3 or native.shape[-1] not in (3, 4) or scattered.shape != native.shape:
        raise ValueError('native and scattered renders must have matching RGB(A) shapes')
    if profile_mask.shape != native.shape[:2] or not all(np.isfinite(x).all() for x in (native, scattered, profile_mask)):
        raise ValueError('finite matching profile mask and render inputs required')
    if (profile_mask < 0).any() or (profile_mask > 1).any():
        raise ValueError('profile weights must be in [0,1]')
    result = native.copy()
    mask = profile_mask[..., None]
    result[..., :3] = native[..., :3] * (1. - mask) + scattered[..., :3] * mask
    return result
