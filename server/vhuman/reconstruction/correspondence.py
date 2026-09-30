"""Topology-pinned anatomical attachments; independent of portrait/view pose.

ICT indices are authored Multi-PIE68 landmarks (MIT). GNM attachments are our
neutral eye-aligned transfer, not a claim of authored GNM anatomical labels.
"""
import hashlib
import json
from pathlib import Path
import numpy as np


def attachments(source):
    doc = json.loads((Path(__file__).parent/'data/anatomical_anchors.json').read_text())
    row = doc['models'].get(getattr(source,'name',None))
    if row is None:
        return {}
    digest = hashlib.sha256(np.asarray(source.vertices,'<f4').tobytes()).hexdigest()
    if digest != row['neutral_sha256']:
        raise ValueError('anatomical attachment neutral topology changed')
    return dict(row['anchors'],**{f'landmark_{i:02d}':[int(vertex)] for i,vertex in enumerate(row['landmarks68'])})


def occlusion_weight(view, xy):
    """Explicit user exclusion masks suppress occluded landmark evidence."""
    if not view.get('exclusion_mask_path'):
        return 1.
    from PIL import Image
    with Image.open(view['exclusion_mask_path']) as image:
        x,y = np.floor(xy).astype(int)
        if not (0<=x<image.width and 0<=y<image.height):
            return 0.
        return 1-image.convert('L').getpixel((int(x),int(y)))/255.
