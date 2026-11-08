"""Localized UE-inspired optical blur at the lid/eye contact, in scene-linear RGB.

This emulates a translucent eye-occlusion layer sampling scene colour. It is
not a physical corneal model and must be documented as a look-matching option.
The optional skin-side band models the same layer crossing the lid margin.
It is distinct from skin subsurface scattering; distant skin and alpha stay intact.
"""
import numpy as np


def contact_band(eye_mask, depth, focal_px, width_m, *, _return_scale_mask=False):
    from scipy.ndimage import distance_transform_edt
    mask, depth = np.asarray(eye_mask, np.float32), np.asarray(depth, np.float32)
    if mask.ndim != 2 or depth.shape != mask.shape or not np.isfinite(mask).all() or not np.isfinite(depth).all():
        raise ValueError('finite matching HxW eye mask and depth required')
    if not np.isfinite(focal_px) or focal_px <= 0 or not np.isfinite(width_m) or width_m <= 0:
        raise ValueError('positive finite focal length and contact width required')
    mask = np.clip(mask, 0, 1)
    valid = (mask > .5) & (depth > 0)
    if not valid.any():
        empty = np.zeros_like(mask)
        return (empty, np.zeros_like(valid)) if _return_scale_mask else empty
    raw_distance = np.maximum(distance_transform_edt(valid) - .5, 0)
    width_px = width_m * float(focal_px) / np.maximum(depth, 1e-6)
    if _return_scale_mask:
        raw_t = np.clip(raw_distance / width_px, 0, 1)
        raw_band = (mask * valid * (1 - raw_t * raw_t * (3 - 2 * raw_t))).astype(np.float32)
        scale_mask = raw_band > 0
    # Interpolate the boundary position through partially covered interior
    # pixels, so the fade does not jump as coverage crosses one half.
    distance = np.maximum(raw_distance - (1 - mask), 0)
    t = np.clip(distance / width_px, 0, 1)
    # Keep antialiased eye coverage on the outer edge of the opaque aperture.
    # Limit it to immediate neighbours so isolated translucent samples do not
    # create a contact treatment far from the eye.
    fractional = (mask > 0) & (mask <= .5) & (depth > 0)
    if fractional.any():
        fractional &= distance_transform_edt(~valid) <= 1.5
    band = (mask * (valid | fractional) * (1 - t * t * (3 - 2 * t))).astype(np.float32)
    return (band, scale_mask) if _return_scale_mask else band


def _thin_aperture_band(band, eye_mask, depth, focal_px, width_m):
    """Experimental coverage fallback for thin, fragmented eye apertures.

    Nearby covered pixels vote as a component even when antialiasing leaves
    small gaps. Confidence fades for weak, short, round, thick or depth-mixed
    components, and for subpixel contact widths. The grouping cannot determine
    whether separate nearby eye samples belong to the same aperture.
    """
    from scipy.ndimage import binary_dilation, find_objects, label
    eye = np.clip(np.asarray(eye_mask, np.float32), 0, 1)
    depth = np.asarray(depth, np.float32)
    valid = (eye > 0) & (depth > 0)
    labels, count = label(binary_dilation(valid, structure=np.ones((3, 3), bool)),
                          structure=np.ones((3, 3), bool))
    if count > 4096:
        return band
    result = band.copy()
    for k, region in enumerate(find_objects(labels), 1):
        if region is None:
            continue
        group = (labels[region] == k) & valid[region]
        y, x = np.nonzero(group)
        weights = eye[region][group]
        if len(weights) < 6:
            continue
        points = np.stack((x, y), axis=1).astype(np.float64)
        points -= np.average(points, axis=0, weights=weights)
        covariance = (points * weights[:, None]).T @ points / weights.sum()
        eigenvalues = np.linalg.eigvalsh(covariance)
        length = np.sqrt(max(float(eigenvalues[1]), 0.))
        thickness = np.sqrt(max(float(eigenvalues[0]), 0.))
        ratio = eigenvalues[1] / max(float(eigenvalues[0]), 1e-8)
        z = depth[region][group].astype(np.float64)
        mean_z = np.average(z, weights=weights)
        spread = np.sqrt(np.average((z - mean_z) ** 2, weights=weights))
        width_px = float(focal_px) * float(width_m) / float(np.median(z))
        confidence = (np.clip((weights.sum() - 2) / 6, 0, 1)
                      * np.clip((length - .5) / .5, 0, 1)
                      * np.clip((ratio - 1) / 2, 0, 1)
                      * np.clip(3. - thickness, 0, 1)
                      * np.clip((width_px - .1) / .9, 0, 1)
                      * np.clip((.016 - spread) / .008, 0, 1))
        if confidence:
            local = result[region]
            local[group] = local[group] * (1 - confidence) + eye[region][group] * confidence
    return result


def skin_contact_band(eye_mask, skin_mask, depth, focal_px, width_m):
    """Skin coverage near the visible eye, with metric width and depth rejection."""
    from scipy.ndimage import distance_transform_edt
    eye = np.asarray(eye_mask, np.float32)
    skin = np.asarray(skin_mask, np.float32)
    depth = np.asarray(depth)
    if eye.ndim != 2 or skin.shape != eye.shape or depth.shape != eye.shape:
        raise ValueError('matching HxW eye, skin and depth arrays required')
    if not all(np.isfinite(x).all() for x in (eye, skin, depth)):
        raise ValueError('finite eye, skin and depth arrays required')
    if not np.isfinite(focal_px) or focal_px <= 0 or not np.isfinite(width_m) or width_m <= 0:
        raise ValueError('positive finite focal length and contact width required')
    eye, skin = np.clip(eye, 0, 1), np.clip(skin, 0, 1)
    valid = (eye > .5) & (depth > 0)
    if not valid.any():
        return np.zeros_like(eye)
    distance, nearest = distance_transform_edt(~valid, return_indices=True)
    distance = np.maximum(distance - .5, 0)
    width_px = float(width_m) * float(focal_px) / np.maximum(depth, 1e-6)
    t = np.clip(distance / width_px, 0, 1)
    # A foreground nose/other surface can be adjacent in screen space without
    # belonging to the contact layer. Ten millimetres is a generous depth bound.
    adjacent = (depth > 0) & (np.abs(depth - depth[tuple(nearest)]) <= .01)
    return ((1 - t*t*(3 - 2*t)) * skin * (1 - eye) * adjacent).astype(np.float32)


def _eye_depth_coverage_confidence(eye, depth, eligible=None):
    """Local eye coverage, excluding neighbours on different depth layers."""
    h, w = eye.shape
    support = (eye > 0) & (depth > 0)
    if eligible is not None:
        support &= eligible
    ys, xs = np.nonzero(support)
    result = np.zeros_like(eye)
    if not len(xs):
        return result
    offsets = np.arange(-3, 4, dtype=np.float64)
    kernel = np.exp(-.5 * (offsets / .75) ** 2)
    kernel /= kernel.sum()
    depth_here = depth[ys, xs]
    coverage = np.zeros(len(xs), np.float32)
    for j, dy in enumerate(range(-3, 4)):
        for i, dx in enumerate(range(-3, 4)):
            y, x = ys + dy, xs + dx
            inside = (y >= 0) & (y < h) & (x >= 0) & (x < w)
            if not inside.any():
                continue
            y, x = y[inside], x[inside]
            neighbour_depth = depth[y, x]
            gate = (neighbour_depth > 0) * np.clip((.01 - np.abs(depth_here[inside] - neighbour_depth)) / .002, 0, 1)
            coverage[inside] += (kernel[j] * kernel[i]) * eye[y, x] * gate
    result[ys, xs] = np.clip((coverage - .145) / .04, 0, 1)
    return result


def _coverage_skin_contact_band(eye_mask, skin_mask, depth, focal_px, width_m):
    """Experimental continuous-coverage skin-side contact band.

    Gather the strongest nearby coverage-weighted contribution rather than
    choosing a single binary eye pixel. A local coverage confidence suppresses
    isolated fractional samples, and source/receiver depths must be close.
    The finite gather fades to the legacy band at projected widths of 31-32 px
    or when the estimated gather work becomes excessive.
    """
    from scipy.ndimage import gaussian_filter
    eye = np.asarray(eye_mask, np.float32)
    skin = np.asarray(skin_mask, np.float32)
    depth = np.asarray(depth, np.float32)
    if eye.ndim != 2 or skin.shape != eye.shape or depth.shape != eye.shape:
        raise ValueError('matching HxW eye, skin and depth arrays required')
    if not all(np.isfinite(x).all() for x in (eye, skin, depth)):
        raise ValueError('finite eye, skin and depth arrays required')
    if not np.isfinite(focal_px) or focal_px <= 0 or not np.isfinite(width_m) or width_m <= 0:
        raise ValueError('positive finite focal length and contact width required')
    eye, skin = np.clip(eye, 0, 1), np.clip(skin, 0, 1)
    height, width = eye.shape
    result = np.zeros_like(eye)
    visible_skin = (skin > 0) & (depth > 0)
    if not visible_skin.any():
        return result
    width_px = float(width_m) * float(focal_px) / np.maximum(depth, 1e-6)
    if float(width_px[visible_skin].min()) >= 32:
        return skin_contact_band(eye, skin, depth, focal_px, width_m)
    visible_eye = np.where(depth > 0, eye, 0)
    # Depth rejection can only decrease this nonnegative Gaussian response.
    # Skip source pixels that cannot reach the confidence threshold even
    # before depth rejection; a small margin avoids a rounding-boundary miss.
    preliminary = gaussian_filter(visible_eye, .75, mode='constant')
    eligible = (eye > 0) & (depth > 0) & (preliminary > .145 - 1e-5)
    confidence_work = 49 * int(np.count_nonzero(eligible))
    if confidence_work >= 200_000_000:
        return skin_contact_band(eye, skin, depth, focal_px, width_m)
    confidence_mix = np.clip((confidence_work - 100_000_000) / 100_000_000, 0, 1)
    confidence = _eye_depth_coverage_confidence(eye, depth, eligible)
    possible_sources = (depth > 0) & ((eye > .5) | ((eye > 0) & (confidence > 0)))
    if not possible_sources.any():
        return result
    # A receiver farther than 10 mm from every source cannot contribute. This
    # avoids inflating the gather for an unrelated foreground skin sample.
    nearest_depth = max(float(depth[possible_sources].min()) - .01, 1e-6)
    maximum = float(width_m) * float(focal_px) / nearest_depth
    ys, xs = np.nonzero((eye > 0) & (depth > 0) & (confidence > 0))
    if not len(xs):
        if maximum > 31 or confidence_mix > 0:
            legacy = skin_contact_band(eye, skin, depth, focal_px, width_m)
            return legacy * np.maximum(np.clip(width_px - 31, 0, 1), confidence_mix)
        return result
    radius = min(int(np.ceil(maximum + .5)), 32)
    y0, y1 = max(0, int(ys.min()) - radius), min(height, int(ys.max()) + radius + 1)
    x0, x1 = max(0, int(xs.min()) - radius), min(width, int(xs.max()) + radius + 1)
    work = (y1 - y0) * (x1 - x0) * (2 * radius + 1) ** 2
    if work >= 400_000_000:
        return skin_contact_band(eye, skin, depth, focal_px, width_m)
    work_mix = max(confidence_mix, np.clip((work - 200_000_000) / 200_000_000, 0, 1))
    region = np.s_[y0:y1, x0:x1]
    local_eye, local_skin, local_depth, local_width, local_confidence = (
        a[region] for a in (eye, skin, depth, width_px, confidence))
    h, w = local_eye.shape
    best = np.zeros_like(local_eye)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            distance = np.hypot(dx, dy)
            if distance > radius + .5:
                continue
            a, b = max(0, dy), min(h, h + dy)
            c, d = max(0, dx), min(w, w + dx)
            if a >= b or c >= d:
                continue
            dst, src = np.s_[a:b, c:d], np.s_[a-dy:b-dy, c-dx:d-dx]
            alpha = local_eye[src]
            dist = np.maximum(distance - .5 + (1 - alpha), 0)
            t = np.clip(dist / local_width[dst], 0, 1)
            adjacent = (local_depth[src] > 0) * np.clip(
                (.01 - np.abs(local_depth[dst] - local_depth[src])) / .002, 0, 1)
            contribution = (1 - t*t*(3 - 2*t)) * alpha * local_confidence[src] * adjacent
            best[dst] = np.maximum(best[dst], contribution)
    result[region] = best * local_skin * (1 - local_eye) * (local_depth > 0)
    if maximum > 31 or work_mix > 0:
        legacy = skin_contact_band(eye, skin, depth, focal_px, width_m)
        mix = np.maximum(np.clip(width_px - 31, 0, 1), work_mix)
        result = result * (1 - mix) + legacy * mix
    return result


def eye_upper_weight(eye_mask, active, depth):
    """Estimate upper/lower eye position along each visible aperture's axis.

    This optional look-matching field uses the long axis of each opaque eye
    component. Nearly round, tiny, or nearly vertical components fall back
    toward a neutral half-weight; image geometry cannot determine anatomical
    up for an upside-down eye.
    """
    from scipy.ndimage import distance_transform_edt, find_objects, label
    eye = np.asarray(eye_mask, np.float32)
    active = np.asarray(active, bool)
    depth = np.asarray(depth, np.float32)
    if eye.ndim != 2 or active.shape != eye.shape or depth.shape != eye.shape:
        raise ValueError('matching eye, active and depth arrays required')
    if not np.isfinite(eye).all() or not np.isfinite(depth).all():
        raise ValueError('finite eye and depth required')
    out = np.full(eye.shape, .5, np.float32)
    opaque = (eye > .5) & (depth > 0)
    if not opaque.any() or not active.any():
        return out
    labels, count = label(opaque, structure=np.ones((3,3), bool))
    if count > 4096:
        return out
    distance, nearest = distance_transform_edt(~opaque, return_indices=True)
    ya, xa = np.nonzero(active)
    assignment = labels[nearest[0, ya, xa], nearest[1, ya, xa]]
    order = np.argsort(assignment, kind='stable')
    sorted_labels = assignment[order]
    ranges = np.searchsorted(sorted_labels, np.arange(count + 2))
    sy, sx = np.nonzero((eye > 0) & (depth > 0) & (distance <= 1.5))
    support_assignment = labels[nearest[0, sy, sx], nearest[1, sy, sx]]
    support_order = np.argsort(support_assignment, kind='stable')
    support_ranges = np.searchsorted(support_assignment[support_order], np.arange(count + 2))
    for k, region in enumerate(find_objects(labels), 1):
        if region is None or ranges[k] == ranges[k+1]:
            continue
        ly, lx = np.nonzero(labels[region] == k)
        if len(lx) < 9:
            continue
        x = lx + region[1].start; y = ly + region[0].start
        points = np.stack((x, y), axis=1).astype(np.float64)
        centre = points.mean(axis=0); points -= centre
        eigenvalues, vectors = np.linalg.eigh(points.T @ points / len(points))
        ratio = eigenvalues[-1] / max(float(eigenvalues[0]), 1e-8)
        axis = vectors[:, -1]
        if axis[0] < 0:
            axis = -axis
        # Include nearby fractional coverage when measuring aperture
        # thickness. A binary threshold alone can make the split jump as a
        # nearly closed eyelid moves across a pixel.
        support = support_order[support_ranges[k]:support_ranges[k+1]]
        support_points = np.stack((sx[support], sy[support]), axis=1).astype(np.float64)
        weights = np.clip(eye[sy[support], sx[support]], 0, 1).astype(np.float64)
        support_mean = np.average(support_points, axis=0, weights=weights)
        support_points -= support_mean
        weighted_cov = (support_points * weights[:, None]).T @ support_points / weights.sum()
        thickness = np.sqrt(max(float(np.linalg.eigvalsh(weighted_cov)[0]), 0.))
        reliability = (np.clip((ratio - 1.) / .2, 0, 1) * np.clip((axis[0] - .1) / .2, 0, 1)
                       * np.clip((thickness - .5) / 1.5, 0, 1))
        if reliability == 0:
            continue
        across = np.array([-axis[1], axis[0]])
        u, v = points @ axis, points @ across
        origin = np.floor(u.min()); bins = np.floor(u - origin).astype(np.int32)
        size = int(bins.max()) + 1
        lo, hi = np.full(size, np.inf), np.full(size, -np.inf)
        np.minimum.at(lo, bins, v); np.maximum.at(hi, bins, v)
        good = np.flatnonzero(np.isfinite(lo))
        lo = np.interp(np.arange(size), good, lo[good])
        hi = np.interp(np.arange(size), good, hi[good])
        selected = order[ranges[k]:ranges[k+1]]
        q = np.stack((xa[selected], ya[selected]), axis=1) - centre
        position = q @ across
        cell = np.clip(np.floor(q @ axis - origin).astype(np.int32), 0, size-1)
        t = np.clip(((position-lo[cell]) / np.maximum(hi[cell]-lo[cell], 1) - .35) / .3, 0, 1)
        top = 1 - t*t*(3-2*t)
        out[ya[selected], xa[selected]] = .5 + reliability * (top - .5)
    return out


def soften_contact(beauty, eye_mask, depth, focal_px, width_m=.0005, sigma_m=.00015,
                   strength=.5, tint=(1., .904661, .896269), *, skin_mask=None,
                   skin_width_m=.0012, skin_strength=0., skin_sigma_m=None, skin_tint=None,
                   upper_strength=None, lower_strength=None, thin_aperture=False,
                   skin_coverage=False):
    from scipy.ndimage import gaussian_filter
    beauty = np.asarray(beauty, np.float32)
    if beauty.ndim != 3 or beauty.shape[-1] not in (3, 4) or not np.isfinite(beauty).all():
        raise ValueError('finite HxWxRGB(A) beauty required')
    if np.asarray(eye_mask).shape != beauty.shape[:2]:
        raise ValueError('eye mask must match beauty dimensions')
    tint = np.asarray(tint, np.float32)
    if tint.shape != (3,) or not np.isfinite(tint).all() or (tint < 0).any():
        raise ValueError('nonnegative finite RGB tint required')
    if skin_tint is not None:
        skin_tint = np.asarray(skin_tint, np.float32)
        if skin_tint.shape != (3,) or not np.isfinite(skin_tint).all() or (skin_tint < 0).any():
            raise ValueError('nonnegative finite RGB skin tint required')
    if not np.isfinite(sigma_m) or sigma_m <= 0 or not np.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError('positive blur sigma and strength in [0,1] required')
    if not np.isfinite(skin_strength) or not 0 <= skin_strength <= 1:
        raise ValueError('skin contact strength must be in [0,1]')
    if skin_sigma_m is not None and (not np.isfinite(skin_sigma_m) or skin_sigma_m <= 0):
        raise ValueError('skin blur sigma must be positive and finite')
    if (upper_strength is None) != (lower_strength is None):
        raise ValueError('both upper and lower eye strengths are required')
    if upper_strength is not None and (not np.isfinite(upper_strength) or not np.isfinite(lower_strength)
                                       or not 0 <= upper_strength <= 1 or not 0 <= lower_strength <= 1):
        raise ValueError('upper and lower eye strengths must be in [0,1]')
    band, opaque_valid = contact_band(eye_mask, depth, focal_px, width_m, _return_scale_mask=True)
    if thin_aperture:
        band = _thin_aperture_band(band, eye_mask, depth, focal_px, width_m)
    skin_band = np.zeros_like(band)
    if skin_strength > 0:
        if skin_mask is None:
            raise ValueError('skin-side contact requires skin coverage')
        band_fn = _coverage_skin_contact_band if skin_coverage else skin_contact_band
        skin_band = band_fn(eye_mask, skin_mask, depth, focal_px, skin_width_m)
    valid = band > 0
    skin_valid = skin_band > 0
    eye_off = strength == 0 if upper_strength is None else upper_strength == 0 and lower_strength == 0
    if (not valid.any() and not skin_valid.any()) or (eye_off and skin_strength == 0):
        return beauty.copy()
    # Keep the blur scale based on the opaque eye, rather than allowing the
    # number of partially covered edge pixels to change the whole-eye sigma.
    if not opaque_valid.any() and valid.any():
        opaque_valid = (np.asarray(eye_mask) > .5) & (np.asarray(depth) > 0)
    scale_valid = opaque_valid if opaque_valid.any() else (skin_valid if skin_valid.any() else valid)
    median_depth = float(np.median(np.asarray(depth)[scale_valid]))
    sigma_px = float(sigma_m) * float(focal_px) / median_depth
    blurred = gaussian_filter(beauty[..., :3], (sigma_px, sigma_px, 0), mode='nearest') * tint
    result = beauty.copy()
    eye_strength = strength
    if upper_strength is not None:
        if upper_strength == lower_strength:
            eye_strength = upper_strength
        else:
            top = eye_upper_weight(eye_mask, valid, depth)
            eye_strength = lower_strength + (upper_strength-lower_strength)*top
    weight = (band * eye_strength)[..., None]
    result[..., :3] = beauty[..., :3] * (1 - weight) + blurred * weight
    if skin_strength > 0:
        if skin_sigma_m is None and skin_tint is None:
            skin_blur = blurred
        else:
            skin_sigma = float(sigma_m if skin_sigma_m is None else skin_sigma_m)*float(focal_px)/median_depth
            skin_blur = gaussian_filter(beauty[..., :3], (skin_sigma,)*2+(0,), mode='nearest') * (tint if skin_tint is None else skin_tint)
        result[..., :3] += (skin_blur - beauty[..., :3]) * (skin_band * skin_strength)[..., None]
    return result


def projected_eye_mask(data, camera, shape, skin_mask=None, supersample=4):
    """Rasterize staged static lid attachments under the exact render camera.

    Animated callers must supply posed attachment points. The optional skin AOV
    excludes actual skin coverage, including displacement, from the contact band.
    """
    import cv2
    h, w = shape
    origin = np.asarray(camera['origin'], np.float64)
    rows = np.asarray(camera['rotation'], np.float64)
    focal = float(camera['focal_px']) * h / float(camera['size'][1])
    if rows.shape != (3,3) or origin.shape != (3,) or supersample < 1:
        raise ValueError('valid pinhole camera and positive supersampling required')
    canvas = np.zeros((h*supersample, w*supersample), np.uint8)
    found = False
    for side in ('left', 'right'):
        keys = [f'{side}_tearline_positions', f'{side}_tearline_upper_positions']
        if not all(key in data for key in keys):
            continue
        lower, upper = [np.asarray(data[key], np.float64).reshape(-1,8,3).mean(1) for key in keys]
        contour = np.concatenate((upper, lower[::-1]))
        local = (contour-origin) @ rows.T
        if (local[:,2] >= -.001).any():
            continue
        pixels = np.stack((w/2+focal*local[:,0]/-local[:,2], h/2-focal*local[:,1]/-local[:,2]), -1)
        cv2.fillPoly(canvas, [np.rint(pixels*supersample).astype(np.int32)], 255)
        found = True
    if not found:
        raise ValueError('staged upper/lower lid attachments required for an eye contact mask')
    mask = canvas.reshape(h,supersample,w,supersample).mean((1,3)).astype(np.float32)/255
    if skin_mask is not None:
        skin_mask = np.asarray(skin_mask, np.float32)
        if skin_mask.shape != mask.shape or not np.isfinite(skin_mask).all():
            raise ValueError('matching finite skin coverage required')
        mask *= 1-np.clip(skin_mask,0,1)
    return mask
