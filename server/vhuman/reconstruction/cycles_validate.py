"""Blender-only reload gate for packed signed maps and native motion samples."""
import json
from pathlib import Path
import sys
import bpy
import numpy as np


def validate(out):
    out=Path(out);detail=np.load(out/'skin_detail.npz',allow_pickle=False)
    checks=[]
    for name,expected in [('skin_height_metres',detail['height_m'])]+[
            (f'wrinkle_height_{i}',row) for i,row in enumerate(detail['dynamic_height_m'])]:
        image=bpy.data.images[name]
        if image.packed_file is None:raise ValueError(f'{name}: unpacked physical map')
        pixels=np.empty(len(image.pixels),np.float32);image.pixels.foreach_get(pixels)
        actual=pixels.reshape(image.size[1],image.size[0],4)[:,:,0]
        error=float(np.max(abs(actual-expected[::-1])))
        if error>1e-8:raise ValueError(f'{name}: signed physical map changed by {error} m')
        checks.append(dict(name=name,max_error_m=error,minimum_m=float(actual.min())))
    request=json.loads((out/'request.json').read_text())
    if request.get('motion'):
        motion=np.load(Path(request['motion'])/'motion.npz',allow_pickle=False)
        assets=np.load(out/'scene_assets.npz',allow_pickle=False)
        obj=bpy.data.objects['skin'];keys=obj.data.shape_keys.key_blocks
        if len(keys)!=len(motion['vertices'])+1:raise ValueError('native motion frame count changed')
        for frame in (0,len(motion['vertices'])-1):
            key=keys[f'GNM_{frame+1:03d}'];positions=np.empty(len(key.data)*3,np.float32)
            key.data.foreach_get('co',positions)
            expected=motion['vertices'][frame,assets['skin_native_ids']]
            if np.max(abs(positions.reshape(-1,3)-expected))>1e-7:
                raise ValueError('native motion coordinates changed after scene reload')
    result=dict(passed=True,float_maps=checks,native_motion_checked=bool(request.get('motion')),
                storage='packed scene-linear 32-bit EXR')
    (out/'asset_validation.json').write_text(json.dumps(result,indent=2))
    print('VHUMAN_PACKED_ASSET_VALID',flush=True)


if __name__=='__main__':validate(sys.argv[sys.argv.index('--')+1])
