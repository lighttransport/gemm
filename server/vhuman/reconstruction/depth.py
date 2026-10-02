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


def _native_manifest(installation):
    installation = Path(installation)
    manifest = json.loads((installation / 'installation.json').read_text())
    native = installation / 'native'
    try:
        converted = json.loads((native / 'native.json').read_text())
    except FileNotFoundError as exc:
        raise ValueError('Native DA2 assets missing; run ref/da2/export_reference.py with '
                         '--installation DIR --out DIR/native in the offline export environment') from exc
    if (manifest.get('model') != 'Depth-Anything-V2-Small' or converted.get('version') != 1 or
            converted.get('model') != manifest['model'] or converted.get('source') != manifest or
            converted.get('dtype') != 'F32'):
        raise ValueError('unverified Small native depth export')
    for name in ('dinov2.safetensors', 'depth_head.safetensors'):
        if not (native / name).is_file() or sha256(native / name) != converted.get('files', {}).get(name):
            raise ValueError('native depth checkpoint checksum mismatch: ' + name)
    return native, manifest


def _preprocess(image, input_size=518):
    # Preserve the pinned upstream's float64 OpenCV cubic resize and rounding.
    import cv2
    if not isinstance(input_size, int) or not 14 <= input_size <= 1024:
        raise ValueError('depth input size must be between 14 and 1024')
    raw = cv2.imread(str(image))
    if raw is None:
        raise ValueError('depth image cannot be decoded')
    h, w = raw.shape[:2]
    if h * w > 4194304:
        raise ValueError('depth image exceeds four megapixels')
    scale = max(input_size / h, input_size / w)
    def multiple(value):
        result = int(np.round(value / 14) * 14)
        return int(np.ceil(value / 14) * 14) if result < input_size else result
    nh, nw = multiple(h * scale), multiple(w * scale)
    if (nh // 14) * (nw // 14) > 4096:
        raise ValueError('depth aspect ratio exceeds the native 4096-patch limit')
    rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB) / 255.0
    rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_CUBIC)
    rgb = (rgb - np.asarray([.485, .456, .406])) / np.asarray([.229, .224, .225])
    return np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32), h, w


def infer(image, installation, out, *, input_size=518, threads=4):
    import subprocess
    import tempfile
    from .. import gpu
    from ..service import ROOT
    native, manifest = _native_manifest(installation)
    chw, h, w = _preprocess(image, input_size)
    requested = gpu.backend()
    backend = 'cuda' if requested == 'cuda' else 'cpu'
    directory = ROOT / backend / 'da2'
    subprocess.run(['make', '-s', '-C', str(directory), 'da2_depth'],
                   check=True, capture_output=True, text=True)
    temp = ROOT / 'tmp/vhuman-runtime'
    temp.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='da2-', dir=temp) as folder:
        folder = Path(folder)
        chw.tofile(folder / 'input.f32')
        cmd = [str(directory / 'da2_depth'), '--backbone', str((native / 'dinov2.safetensors').resolve()),
               '--head', str((native / 'depth_head.safetensors').resolve()),
               '--input', str(folder / 'input.f32'), '--output', str(folder / 'depth.f32'),
               '--width', str(chw.shape[2]), '--height', str(chw.shape[1]),
               '--output-width', str(w), '--output-height', str(h),
               '--backend', backend, '--device', str(gpu.device_index()), '--threads', str(threads)]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
        if proc.returncode:
            raise RuntimeError('Native DA2 failed: ' + proc.stderr[-4000:])
        depth = np.fromfile(folder / 'depth.f32', dtype='<f4')
        if depth.size != h * w or not np.isfinite(depth).all() or np.any(depth < 0):
            raise ValueError('invalid native relative depth output')
        report = json.loads(proc.stdout)
    np.save(out, depth.reshape(h, w), allow_pickle=False)
    return dict(model=manifest, adapter='native DINOv2 + DA3 DPT with repository GEMM',
                convention='relative inverse depth, original uncropped image pixels',
                device=backend, requested_backend=requested, runner=report)


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
