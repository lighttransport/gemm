"""Bounded, timestamp-preserving extraction and fixed-camera motion fitting."""
from __future__ import annotations

import json
import shutil

import numpy as np

from . import job


def sample_video(source, out, cancel):
    ffprobe, ffmpeg = shutil.which("ffprobe"), shutil.which("ffmpeg")
    if not ffprobe or not ffmpeg:
        raise ValueError("MuscleMimic video fitting needs ffmpeg and ffprobe")
    # _run retains only 40 log lines; use a bounded file for timestamp metadata.
    return _sample_with_probe(source, out, cancel, ffprobe, ffmpeg)


def _sample_with_probe(source, out, cancel, ffprobe, ffmpeg):
    import subprocess
    # ffprobe emits bounded metadata for the first eight seconds; no media pixels.
    meta = out / "timestamps-probe.json"
    with meta.open("wb") as dst:
        proc = subprocess.Popen([ffprobe, "-v", "error", "-select_streams", "v:0", "-read_intervals", "%+8",
            "-show_frames", "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", str(source)],
            stdout=dst, stderr=subprocess.DEVNULL)
        import time
        deadline = time.monotonic() + 120
        try:
            while proc.poll() is None:
                if cancel.wait(.1) or time.monotonic() > deadline or meta.stat().st_size > 2 << 20:
                    raise ValueError("video probe cancelled, oversized or timed out")
            if proc.returncode:
                raise ValueError("video timestamps could not be decoded")
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait()
    records = json.loads(meta.read_text())["frames"]
    timestamps = np.array([float(f["best_effort_timestamp_time"]) for f in records])
    if len(timestamps) < 3 or not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
        raise ValueError("video needs strictly increasing frame timestamps")
    timestamps -= timestamps[0]
    selected = []
    for i, t in enumerate(timestamps):
        if t >= 8 or len(selected) == 240:
            break
        if not selected or t - timestamps[selected[-1]] >= 1 / 30 - 1e-5:
            selected.append(i)
    if len(selected) < 3:
        raise ValueError("video is too short for physical motion")
    frames = out / "frames"
    frames.mkdir()
    expression = "+".join(f"eq(n,{i})" for i in selected)
    job._run([ffmpeg, "-nostdin", "-v", "error", "-threads", "2", "-i", str(source), "-t", "8",
              "-vf", f"select='{expression}',scale=1280:1280:force_original_aspect_ratio=decrease",
              "-vsync", "vfr", "-frames:v", str(len(selected)), str(frames / "%05d.png")], cancel, timeout=180)
    paths = sorted(frames.glob("*.png"))
    if len(paths) != len(selected):
        raise ValueError("decoded frame count differs from timestamp probe")
    return paths, timestamps[selected]


def check_camera(frames, boxes):
    """Track background features outside the subject; fail closed if unobservable."""
    import cv2
    displacements = []
    for i in range(1, len(frames), max(1, len(frames) // 12)):
        a, b = (cv2.imread(str(frames[k]), cv2.IMREAD_GRAYSCALE) for k in (0, i))
        if a is None or b is None or a.shape != b.shape:
            raise ValueError("video frame dimensions changed")
        mask = np.full(a.shape, 255, np.uint8)
        for box in (boxes[0], boxes[i]):
            x0, y0, x1, y1 = box
            pad = int(.03 * max(a.shape))
            mask[max(0, y0-pad):min(a.shape[0], y1+pad), max(0, x0-pad):min(a.shape[1], x1+pad)] = 0
        points = cv2.goodFeaturesToTrack(a, 200, .02, 8, mask=mask)
        if points is None or len(points) < 12:
            continue
        tracked, status, error = cv2.calcOpticalFlowPyrLK(a, b, points, None)
        if tracked is None or status is None or error is None:
            continue
        good = status[:, 0].astype(bool) & (error[:, 0] < 20)
        if good.sum() >= 12:
            displacements.append(float(np.median(np.linalg.norm(tracked[good] - points[good], axis=-1))) / max(a.shape))
    if len(displacements) < 2:
        raise ValueError("fixed camera cannot be verified: need a textured visible background")
    if max(displacements) > .002:
        raise ValueError("background moves: physical fitting requires a fixed camera")
    return {"background_motion_max_fraction": max(displacements), "camera_checks": len(displacements)}


def person_boxes(frames):
    """Conservative single-person preflight; ambiguous detections are rejected."""
    import cv2
    detector = cv2.HOGDescriptor()
    detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    result = []
    for frame in frames:
        image = cv2.imread(str(frame))
        if image is None:
            raise ValueError("video frame cannot be decoded")
        h, w = image.shape[:2]
        scale = min(1., 640 / max(w, h))
        small = cv2.resize(image, None, fx=scale, fy=scale)
        boxes, scores = detector.detectMultiScale(small, winStride=(8, 8), padding=(8, 8), scale=1.07)
        selected = cv2.dnn.NMSBoxes(list(boxes), np.asarray(scores).reshape(-1).tolist(), .3, .4)
        if len(selected) != 1:
            raise ValueError("video needs one unambiguously detected, fully visible person in every frame")
        x, y, bw, bh = boxes[int(np.asarray(selected).reshape(-1)[0])] / scale
        margin = .005 * max(w, h)
        if min(x, y, w-x-bw, h-y-bh) < margin:
            raise ValueError("person is cropped at the frame edge")
        pad = .08 * max(bw, bh)
        result.append((max(0, int(x-pad)), max(0, int(y-pad)), min(w, int(x+bw+pad)), min(h, int(y+bh+pad))))
    return result


def estimate_root(positions, camera, times):
    """Fit one ground alignment and a contact-regularized root path.

    Camera translation supplies a weak travel prior; stance feet supply local
    displacement constraints. This is an estimate, never measured world motion.
    """
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    from .musclemimic_kinematics import FEET
    p, camera, t = np.asarray(positions).copy(), np.asarray(camera), np.asarray(times)
    if camera.shape != (len(t), 3) or not np.isfinite(camera).all():
        raise ValueError("missing finite camera translation")
    up = np.median(p[:, 2] - p[:, 0], axis=0)
    up /= np.linalg.norm(up)
    if up[1] < .8:
        raise ValueError("subject must be upright on a flat floor")
    alignment, _ = Rotation.align_vectors([[0, 1, 0]], [up])
    p = alignment.apply(p.reshape(-1, 3)).reshape(p.shape)
    p -= p[:, :1]
    prior = alignment.apply(camera * [1, -1, -1])
    prior -= prior[0]
    feet = p[:, FEET]
    # Per-foot lower envelope approximates the constant joint-to-sole offset.
    low = np.percentile(feet[..., 1], 10, axis=0)
    heights = feet[..., 1] - low
    contact = heights < .04
    if np.count_nonzero(contact) < len(t) // 2:
        raise ValueError("insufficient foot contact evidence")
    height = -np.min(feet[..., 1], axis=1)
    initial = prior.copy(); initial[:, 1] = height
    def residual(root_flat):
        root = root_flat.reshape(-1, 3)
        world_feet = feet + root[:, None]
        active = contact[1:] & contact[:-1]
        velocity = np.diff(world_feet, axis=0)[active]
        return np.r_[.2 * (root[:, [0, 2]] - prior[:, [0, 2]]).ravel(),
                     5 * velocity.ravel(), 2 * (root[:, 1] - height),
                     .1 * np.diff(root, n=2, axis=0).ravel(), 10 * root[0, [0, 2]]]
    # The root objective is linear; sparse finite differences keep 240-frame
    # clips bounded instead of building a dense trajectory Jacobian.
    from scipy.sparse import lil_matrix
    size = len(residual(initial.ravel()))
    # Explicit linear least-squares via sparse numerical Jacobian structure.
    sparsity = lil_matrix((size, len(t) * 3), dtype=int)
    row = 0
    for i in range(len(t)):
        for axis in (0, 2):
            sparsity[row, i*3+axis] = 1; row += 1
    for i in range(len(t)-1):
        for _ in np.flatnonzero(contact[i+1] & contact[i]):
            for axis in range(3):
                sparsity[row, [i*3+axis, (i+1)*3+axis]] = 1; row += 1
    for i in range(len(t)):
        sparsity[row, i*3+1] = 1; row += 1
    for i in range(len(t)-2):
        for axis in range(3):
            sparsity[row, [i*3+axis, (i+1)*3+axis, (i+2)*3+axis]] = 1; row += 1
    for axis in (0, 2):
        sparsity[row, axis] = 1; row += 1
    root = least_squares(residual, initial.ravel(), jac_sparsity=sparsity.tocsr(), max_nfev=30).x.reshape(-1, 3)
    disagreement = float(np.sqrt(np.mean((root[:, [0, 2]] - prior[:, [0, 2]])**2)))
    if disagreement > .3:
        raise ValueError("camera and foot-contact travel estimates disagree")
    return p + root[:, None], alignment.as_matrix(), contact, {
        "root_motion": "estimated", "camera_contact_rms_m": disagreement}
