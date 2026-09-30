"""Versioned pixel observations. Manual annotations require no model download."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .reference import Camera

FORMAT = 'vhuman.face_observations.v1'
TASK_SHA256 = '64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff'
ANCHORS = {'nose_tip': [4], 'upper_lip': [13], 'lower_lip': [14], 'menton': [152],
           'mouth_right': [61], 'mouth_left': [291],
           'eye_right': [33, 133, 159, 145], 'eye_left': [263, 362, 386, 374]}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def pixel_sha256(path):
    """Detect reused images across PNG/JPEG metadata or RGB/RGBA re-encoding."""
    with Image.open(path) as image:
        if image.width*image.height>4_000_000:
            raise ValueError('observation exceeds 4M pixels')
        rgba = image.convert('RGBA')
        return hashlib.sha256(str(rgba.size).encode()+rgba.tobytes()).hexdigest()


def load(path):
    path = Path(path)
    doc = json.loads(path.read_text())
    if doc.get('format') != FORMAT or not 1 <= len(doc.get('views', [])) <= 16:
        raise ValueError('expected face_observations.v1 with 1..16 views')
    for view in doc['views']:
        image = (path.parent / view['image']).resolve()
        if not image.is_file() or sha256(image) != view['sha256']:
            raise ValueError('observation image missing or hash mismatch')
        with Image.open(image) as im:
            if im.width*im.height > 4_000_000:
                raise ValueError('observation exceeds 4M pixels')
            if list(im.size) != view['size']:
                raise ValueError('observation dimensions mismatch')
        for key in ('exclusion_mask','silhouette_mask'):
            if not view.get(key):
                continue
            mask = (path.parent/view[key]).resolve()
            if not mask.is_file() or sha256(mask)!=view.get(key+'_sha256'):
                raise ValueError(key+' missing or hash mismatch')
            with Image.open(mask) as im:
                if list(im.size)!=view['size']:
                    raise ValueError(key+' dimensions mismatch')
            view[key+'_path'] = str(mask)
        if 'camera' in view:
            Camera.from_dict(view['camera'])
        if view.get('silhouette') is not None:
            contour = np.asarray(view['silhouette'],float)
            if contour.ndim != 2 or contour.shape[1] != 2 or not 3<=len(contour)<=256 or not np.isfinite(contour).all():
                raise ValueError('silhouette must contain 3..256 finite pixel points')
        if light := view.get('lighting'):
            direction=np.asarray(light.get('direction_h'),float)
            radiance=np.asarray(light.get('radiance'),float)
            ambient=np.asarray(light.get('ambient_rgb',[0,0,0]),float)
            if (direction.shape!=(3,) or radiance.shape!=(3,) or ambient.shape!=(3,)
                    or not np.isfinite([direction,radiance,ambient]).all() or np.linalg.norm(direction)<1e-6
                    or (radiance<0).any() or (ambient<0).any() or not .01<=float(light.get('exposure',1))<=100):
                raise ValueError('invalid calibrated lighting')
        anchors = view.get('anchors', {})
        if len(anchors) > 512:
            raise ValueError('too many anchors')
        for a in anchors.values():
            xy, weight = np.asarray(a['xy'], float), float(a.get('weight', 1))
            if xy.shape != (2,) or not np.isfinite(xy).all() or not 0 <= weight <= 1:
                raise ValueError('invalid pixel anchor')
            if 'vertex' in a and (type(a['vertex']) is not int or a['vertex']<0):
                raise ValueError('invalid anchor vertex')
            if 'vertices' in a:
                ids = a['vertices']
                if not isinstance(ids,list) or not 1<=len(ids)<=32 or any(type(i) is not int or i<0 for i in ids):
                    raise ValueError('invalid anchor vertex list')
        view['image_path'] = str(image)
        view['pixel_sha256'] = pixel_sha256(image)
    return doc


def observe(image, camera=None, task=None):
    from ..rig.face_models import MODEL_CACHE
    task = Path(task) if task else MODEL_CACHE / 'face_landmarker.task'
    if not task.is_file() or sha256(task) != TASK_SHA256:
        raise ValueError('verified MediaPipe task missing; run setup_face_video.sh or provide manual observations')
    import mediapipe as mp
    from mediapipe.tasks.python import vision
    image = Path(image).resolve()
    options = vision.FaceLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(task)),
                                           running_mode=vision.RunningMode.IMAGE, num_faces=2)
    with vision.FaceLandmarker.create_from_options(options) as tracker:
        result = tracker.detect(mp.Image.create_from_file(str(image)))
    if len(result.face_landmarks) != 1:
        raise ValueError('expected exactly one detected face; use manual observations')
    with Image.open(image) as im:
        w, h = im.size
    p = np.array([[v.x*w, v.y*h] for v in result.face_landmarks[0]])
    view = dict(image=str(image), sha256=sha256(image), pixel_sha256=pixel_sha256(image), size=[w, h],
                anchors={name: dict(xy=p[ids].mean(0).tolist(), weight=.8) for name, ids in ANCHORS.items()})
    if camera is not None:
        view['camera'] = camera.as_dict()
    return dict(format=FORMAT, coordinates='original image pixels, top-left, pixel centres',
                scale='assumed IPD unless supplied by user', tracker=dict(sha256=TASK_SHA256,
                source='Google MediaPipe face_landmarker float16 v1'), views=[view])
