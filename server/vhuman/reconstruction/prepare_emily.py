"""Second-subject calibrated Emily scan fixture, detector annotations labelled."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates
from .emily import calibration,distort,read_linear,skin_mesh,unique
from .reference import linear_to_srgb
from .observations import observe,sha256,FORMAT
from .artifacts import artifact_path


def run(root,out):
    root,out=Path(root)/'emily/extracted',artifact_path(out)
    out.mkdir(parents=True,exist_ok=False);mesh=unique(root/'Emily_2_1_OBJ','*.obj');p,t,_=skin_mesh(mesh)
    views=[];records=[]
    for camera_id in (1,3,5):
        camera,size,coeff=calibration(root/'DigitalEmily2_Calibration'/f'camera{camera_id:02d}.txt')
        path=unique(root/'DigitalEmily2_FlashCross',f'cam{camera_id}_*.exr')
        image,stride,record=read_linear(path,max_side=1024)
        scale=512/max(size);w,h=[round(x*scale) for x in size]
        yy,xx=np.mgrid[:h,:w];xy=np.stack((xx.ravel()/scale,yy.ravel()/scale),axis=1)
        raw=distort(xy,camera,coeff)/stride
        linear=np.stack([map_coordinates(image[:,:,c],[raw[:,1],raw[:,0]],order=1,mode='constant') for c in range(3)],axis=1).reshape(h,w,3)
        # Preview-only exposure gauge; scan geometry evaluation does not use its colors.
        gain=max(float(np.quantile(np.maximum(linear,0),.995)),1e-6)
        preview=np.uint8(np.clip(linear_to_srgb(np.maximum(linear,0)/gain),0,1)*255+.5)
        name=f'camera_{camera_id}.png';Image.fromarray(preview).save(out/name)
        view=observe(out/name,camera.scaled(scale))['views'][0]
        view.update(image=name,frame_id='neutral',camera_id=str(camera_id),split='held-out' if camera_id==5 else 'fit')
        views.append(view);records.append(record)
    provenance=dict(dataset='Digital Emily 2',expression='neutral',identity='Emily',annotations='MediaPipe estimates, not independent manual landmarks',
        mesh_sha256=sha256(mesh),linear_references=records,license='noncommercial research/illustration; no redistribution',
        preview='undistorted linear cross-polarized EXR, preview exposure normalized; not material calibration')
    for split in ('fit','held-out'):
        selected=[v for v in views if v['split']==split]
        (out/(split+'.json')).write_text(json.dumps(dict(format=FORMAT,views=selected,provenance=provenance),indent=2)+'\n')
        np.savez_compressed(out/(split+'_reference.npz'),positions=np.repeat(p[None],len(selected),axis=0),triangles=t)
    report=dict(provenance=provenance,cameras=[1,3,5],frames=['neutral'])
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');return report


def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--root',type=Path,default=Path('/mnt/nvme02/data/vhuman'));a.add_argument('--out',type=Path,required=True)
    p=a.parse_args();print(json.dumps(run(p.root,p.out)))

if __name__=='__main__':main()
