"""Native vhuman image-model adapters. Python handles pixels and camera fitting."""
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def run_image_model(task, model, chw, *, backend='cpu', device=0, threads=4,
                    backbone=None, grid=None, runner=None):
    if backend not in ('cpu','cuda'):raise ValueError('native image backend must be cpu or cuda')
    if type(device) is not int or device<0 or type(threads) is not int or not 1<=threads<=128:
        raise ValueError('invalid device/threads')
    if task not in ('rmbg','moge','cues'):raise ValueError('unsupported native model')
    chw=np.ascontiguousarray(chw,dtype='<f4')
    if chw.ndim!=3 or chw.shape[0]!=(6 if task=='cues' else 3) or not np.isfinite(chw).all():
        raise ValueError('expected finite float32 CHW input')
    _,h,w=chw.shape
    limit=2048 if task=='moge' else 1024  # Foreground crop can pad a 1024px image by 10%.
    if not 1<=h<=limit or not 1<=w<=limit:raise ValueError('native image size exceeds bounds')
    exe=Path(runner) if runner else ROOT/backend/'vhuman/vhuman_models'
    if not exe.is_file():raise RuntimeError(f'Build native inference with make -C {backend}/vhuman')
    work=ROOT/'tmp/vhuman-native-runtime';work.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=task+'-',dir=work) as temp:
        source,target=Path(temp)/'input.f32',Path(temp)/'output.f32';chw.tofile(source)
        command=[str(exe.resolve()),'--task',task,'--model',str(Path(model).resolve()),
                 '--input',str(source),'--output',str(target),'--width',str(w),'--height',str(h),
                 '--backend',backend,'--device',str(device),'--threads',str(threads)]
        if task=='moge':
            if backbone is None or grid is None:raise ValueError('MoGe backbone and grid required')
            command+=['--backbone',str(Path(backbone).resolve()),'--grid-height',str(grid[0]),'--grid-width',str(grid[1])]
        result=subprocess.run(command,capture_output=True,text=True)
        if result.returncode:raise RuntimeError(f'{task} native inference exited {result.returncode}: {result.stderr[-3000:]}')
        count=(1 if task=='rmbg' else 4)*h*w
        if target.stat().st_size!=count*4:raise RuntimeError('invalid native output size')
        output=np.fromfile(target,dtype='<f4').reshape(-1,h,w)
        if not np.isfinite(output).all():raise RuntimeError('nonfinite native output')
        return output


def rmbg_alpha(image, model, *, backend='cpu', device=0, threads=4):
    """RMBG2 alpha only; preserve the caller's RGB and original dimensions."""
    image=image.convert('RGB')
    pixels=np.asarray(image.resize((1024,1024),Image.Resampling.BILINEAR),np.float32)/np.float32(255)
    pixels=(pixels-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
    model=Path(model);weights=model/'model.safetensors' if model.is_dir() else model
    logits=run_image_model('rmbg',weights,pixels.transpose(2,0,1),backend=backend,device=device,threads=threads)[0]
    exp=np.exp(-np.abs(logits));probability=np.where(logits>=0,1/(1+exp),exp/(1+exp))
    return Image.fromarray((probability*255).astype(np.uint8)).resize(image.size,Image.Resampling.BICUBIC)


def recover_camera(points, probability):
    """Same 64x64 nearest-sampled focal/shift fit as MoGe, using NumPy/SciPy."""
    from scipy.optimize import least_squares
    points=np.asarray(points,np.float32);probability=np.asarray(probability,np.float32)
    if points.ndim!=3 or points.shape[2]!=3 or probability.shape!=points.shape[:2] or not np.isfinite(points).all() or not np.isfinite(probability).all():
        raise ValueError('invalid camera point/mask arrays')
    h,w=probability.shape;aspect=w/h;sy=1/math.sqrt(1+aspect**2);sx=aspect*sy
    # Torch linspace builds each half from its nearer endpoint to reduce error.
    def linspace(span,n):
        start=np.float32(-span*(n-1)/n);end=-start
        if n==1:return np.zeros(1,np.float32)
        step=np.float32((end-start)/(n-1));i=np.arange(n,dtype=np.float32)
        return np.where(i<n//2,start+step*i,end-step*(n-i-1)).astype(np.float32)
    u,v=np.meshgrid(linspace(sx,w),linspace(sy,h));uv=np.stack((u,v),-1)
    yy=np.floor(np.arange(64)*h/64).astype(int);xx=np.floor(np.arange(64)*w/64).astype(int)
    keep=probability[yy[:,None],xx]>.5
    xyz=points[yy[:,None],xx][keep];uv=uv[yy[:,None],xx][keep]
    if len(xyz)<2:focal,shift=np.float32(1),np.float32(0)
    else:
        xy,z=xyz[:,:2],xyz[:,2]
        def residual(shift):
            projected=xy/(z+shift)[:,None]
            focal=(projected*uv).sum()/np.square(projected).sum()
            return (focal*projected-uv).ravel()
        solution=least_squares(residual,x0=0,ftol=1e-3,method='lm')
        shift=np.float32(solution.x[0]);projected=xy/(z+shift)[:,None]
        focal=np.float32((projected*uv).sum()/np.square(projected).sum())
    fx=np.float32(focal/np.float32(2)*np.float32(math.sqrt(1+aspect**2))/np.float32(aspect))
    fy=np.float32(focal/np.float32(2)*np.float32(math.sqrt(1+aspect**2)))
    if not np.isfinite([fx,fy,shift]).all() or fx<=0 or fy<=0:raise ValueError('MoGe returned invalid camera intrinsics')
    intrinsics=np.array([[fx,0,.5],[0,fy,.5],[0,0,1]],np.float32)
    return dict(intrinsics=intrinsics,focal=float(focal),shift=float(shift),fov=2*math.atan(1/(2*float(fx))))


def moge_bundle(model):
    model = Path(model)
    return model if model.is_dir() else model.parent / 'native'


def moge_ready(model):
    """Cheap health probe; inference also verifies the complete file hashes."""
    bundle = moge_bundle(model)
    try:
        spec = json.loads((bundle / 'native.json').read_text())
        return spec.get('format') == 'vhuman.moge2_camera.v1' and all(
            (bundle / name).is_file() for name in ('dinov2.safetensors', 'heads.safetensors'))
    except (OSError, ValueError, AttributeError):
        return False


def moge_camera(image, model, *, backend='cpu', device=0, threads=4, num_tokens=3600):
    bundle=moge_bundle(model)
    if not (bundle/'native.json').is_file():
        raise ValueError('Native MoGe assets missing; export with ref/vhuman/moge_reference.py and pass the exported directory')
    spec=json.loads((bundle/'native.json').read_text())
    if spec.get('format')!='vhuman.moge2_camera.v1':raise ValueError('unsupported native MoGe package')
    for name in ('dinov2.safetensors','heads.safetensors'):
        if sha256(bundle/name)!=spec['files'][name]:raise ValueError('MoGe checksum mismatch: '+name)
    if type(num_tokens) is not int or not 1<=num_tokens<=4096:raise ValueError('invalid token budget')
    rgb=np.asarray(image.convert('RGB'),np.float32)/np.float32(255)
    h,w=rgb.shape[:2];aspect=w/h;grid=(round(math.sqrt(num_tokens/aspect)),round(math.sqrt(num_tokens*aspect)))
    if min(grid)<1 or max(grid)>128 or grid[0]*grid[1]>4096:raise ValueError('MoGe aspect/token grid exceeds native bounds')
    result=run_image_model('moge',bundle/'heads.safetensors',rgb.transpose(2,0,1),backbone=bundle/'dinov2.safetensors',
        grid=grid,backend=backend,device=device,threads=threads)
    return recover_camera(result[:3].transpose(1,2,0),result[3])
