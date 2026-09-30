"""Optional Apache-2.0 Depth Anything V2 Small adapter and affine cue gate.

No implicit checkpoint download or substitution with NC larger models.
"""
import json
from pathlib import Path
import numpy as np
from .observations import sha256


def align(relative, metric, confidence):
    from scipy.optimize import least_squares
    ok = np.isfinite(relative)&np.isfinite(metric)&(confidence>.5)&(metric>0)
    if ok.sum()<64 or np.std(relative[ok])<1e-6:
        raise ValueError('depth cue insufficient alignment support')
    x,y = relative[ok],metric[ok]
    # DA-V2 outputs relative inverse-depth: fit inverse metric depth.
    target = 1/y
    a = np.column_stack((x,np.ones(len(x))))
    solve = least_squares(lambda p: a@p-target,np.linalg.lstsq(a,target,rcond=None)[0],loss='soft_l1',f_scale=.05)
    pred = a@solve.x
    error = np.median(abs(1/np.maximum(pred,1e-9)-y))
    correlation = np.corrcoef(x,target)[0,1] if np.std(target)>1e-9 else 0.
    if solve.x[0]<=0 or not np.isfinite(error) or error>.003 or not np.isfinite(correlation) or correlation<.5:
        raise ValueError('depth cue rejected: inconsistent facial alignment')
    return 1/np.maximum(solve.x[0]*relative+solve.x[1],1e-9),dict(scale=float(solve.x[0]),shift=float(solve.x[1]),
                    median_error_m=float(error),weight=.05,convention='affine relative inverse depth to inverse metric depth')


def infer(image, installation, out):
    import sys
    import torch
    import cv2
    installation = Path(installation)
    manifest = json.loads((installation/'installation.json').read_text())
    weights = installation/'depth_anything_v2_vits.pth'
    if manifest['model']!='Depth-Anything-V2-Small' or sha256(weights)!=manifest['weights_sha256']:
        raise ValueError('unverified Small depth checkpoint')
    sys.path.insert(0,str(installation/'source'))
    # Upstream uses torchvision only for sequential dictionary transforms.
    # Load the pinned module with our tiny composition adapter; do not install
    # a torchvision wheel that could downgrade this environment's CUDA torch.
    import subprocess
    import types
    source_root = installation/'source'
    revision = subprocess.check_output(['git','-C',str(source_root),'rev-parse','HEAD'],text=True).strip()
    dirty = subprocess.check_output(['git','-C',str(source_root),'status','--porcelain','--untracked-files=no'],text=True)
    if revision != manifest['code_revision'] or dirty:
        raise ValueError('depth source revision changed or working tree modified')
    source = (source_root/'depth_anything_v2/dpt.py').read_text()
    old = 'from torchvision.transforms import Compose'
    if source.count(old) != 1:
        raise ValueError('unsupported depth transform import')
    source = source.replace(old,'from server.vhuman.reconstruction.depth import Compose')
    module = types.ModuleType('depth_anything_v2.dpt')
    module.__package__ = 'depth_anything_v2'
    exec(compile(source,str(source_root/'depth_anything_v2/dpt.py'),'exec'),module.__dict__)
    DepthAnythingV2 = module.DepthAnythingV2
    model = DepthAnythingV2(encoder='vits',features=64,out_channels=[48,96,192,384])
    model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device).eval()
    with torch.inference_mode():
        depth = model.infer_image(cv2.imread(str(image)))
    np.save(out,depth.astype(np.float32),allow_pickle=False)
    return dict(model=manifest,adapter='original sequential transform adapter; no torchvision wheel',convention='relative inverse depth, original uncropped image pixels',device=device)


def refine(vertices, triangles, camera, aligned, confidence, protected_pixels=()):
    """Weak, bounded smooth depth correction. Eye/lip anchors stay fixed.

Depth is a scene prior, not a skin scan. Maximum displacement is 0.5 mm;
orientation and relative triangle-area guards line-search the final update.
"""
    from ..rig.common import edges
    from .fitting import safe_geometry
    xy,z = camera.project(vertices)
    h,w = aligned.shape
    ix,iy = np.floor(xy[:,0]).astype(int),np.floor(xy[:,1]).astype(int)
    valid = (ix>=0)&(ix<w)&(iy>=0)&(iy<h)&(z>0)
    ix,iy = np.clip(ix,0,w-1),np.clip(iy,0,h-1)
    weight = confidence[iy,ix]*valid*.05
    protected = np.zeros(len(vertices),bool)
    for pixel,radius in protected_pixels:
        protected |= np.linalg.norm(xy-np.asarray(pixel),axis=1)<radius
    weight[protected] = 0
    change = np.clip(aligned[iy,ix]-z,-.01,.01)*weight
    change[~np.isfinite(change)] = 0
    e = edges(triangles)
    count = np.bincount(e.reshape(-1),minlength=len(vertices))
    for _ in range(8):
        accum = np.zeros(len(vertices))
        np.add.at(accum,e[:,0],change[e[:,1]])
        np.add.at(accum,e[:,1],change[e[:,0]])
        change = .5*change+.5*accum/np.maximum(count,1)
        change[protected] = 0
    axis = -camera.rotation[2]
    delta = change[:,None]*axis
    step = 1.
    while step>=1/128:
        candidate = vertices+step*delta
        if safe_geometry(vertices,candidate,triangles):
            return candidate.astype(np.float32),dict(max_displacement_m=float(np.linalg.norm(step*delta,axis=1).max()),
                    step=step,weight=.05,protected_vertices=int(protected.sum()))
        step *= .5
    return vertices,dict(max_displacement_m=0,step=0,reason='orientation guard rejected correction')


class Compose:
    """Minimal original callable-chain adapter for upstream dictionary transforms."""
    def __init__(self, transforms):
        self.transforms = tuple(transforms)

    def __call__(self, value):
        for transform in self.transforms:
            value = transform(value)
        return value
