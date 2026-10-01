import numpy as np


def frontal(vertices, size=(512, 512)):
    """Metric +Y-up, +Z-front rig into gsplat's +Z-forward camera coordinates."""
    vertices = np.asarray(vertices)
    center = (vertices.min(0) + vertices.max(0)) * .5
    span = float(np.max(np.ptp(vertices[:, :2], axis=0))) * 1.15
    focal = min(size) * 1.5
    distance = focal * span / min(size)
    view = np.eye(4, dtype=np.float32)
    view[:3, :3] = np.diag([1, -1, -1])
    eye = center + [0, 0, distance]
    view[:3, 3] = -view[:3, :3] @ eye
    intrinsics = np.array([[focal, 0, size[0]/2], [0, focal, size[1]/2], [0, 0, 1]], np.float32)
    return view, intrinsics


def for_avatar(avatar, vertices, size=(512, 512)):
    reference = avatar.metadata.get("reference_camera")
    if reference is None: return frontal(vertices, size)
    view = np.asarray(reference["view"], np.float32)
    intrinsics = np.asarray(reference["intrinsics"], np.float32).copy()
    original = reference["size"]
    if view.shape != (4, 4) or intrinsics.shape != (3, 3) or len(original) != 2 or min(original) <= 0:
        raise ValueError("invalid reference camera")
    if not np.isfinite(view).all() or not np.isfinite(intrinsics).all(): raise ValueError("nonfinite reference camera")
    intrinsics[0] *= size[0] / original[0]
    intrinsics[1] *= size[1] / original[1]
    return view, intrinsics
