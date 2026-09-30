"""Original metric-space, UV-seam-consistent authored skin normal detail.

This is a procedural artist prior. No portrait-derived pore recovery is claimed.
Frequency is limited by the atlas texel footprint; no extra runtime deformation.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .observations import sha256
from .artifacts import artifact_path


def field(points,normals,tangent,bitangent,wavelength,strength_um=5.,seed=19):
    if not 0<=strength_um<=30:raise ValueError('detail strength must be 0..30 micrometres')
    if strength_um==0:return np.tile([0.,0.,1.],(len(points),1))
    rng=np.random.default_rng(seed);directions=rng.normal(size=(12,3));directions/=np.linalg.norm(directions,axis=1,keepdims=True)
    phase=rng.uniform(0,2*np.pi,12)
    frequency=2*np.pi/np.maximum(np.asarray(wavelength),.0004)
    gradient=np.zeros_like(points,dtype=float)
    for direction,offset in zip(directions,phase):
        gradient+=np.cos((points@direction)*frequency+offset)[:,None]*direction*frequency[:,None]*strength_um*1e-6/12
    gradient-=normals*(gradient*normals).sum(1)[:,None]
    normal=normals-gradient;normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-9)
    result=np.column_stack(((normal*tangent).sum(1),(normal*bitangent).sum(1),(normal*normals).sum(1)))
    result/=np.maximum(np.linalg.norm(result,axis=1,keepdims=True),1e-9)
    return result


def atlas(vertices,triangles,uv,res,strength_um=5.):
    from ..rig.bake import rasterize_uv
    from ..rig.common import vertex_normals
    from scipy.ndimage import distance_transform_edt
    corner=uv.reshape(-1,2);indices=np.arange(len(corner)).reshape(-1,3)
    ids,bary=rasterize_uv(corner,indices,res);yy,xx=np.nonzero(ids>=0);tri_id=ids[yy,xx]
    p=vertices[triangles];du=uv[:,1]-uv[:,0];dv=uv[:,2]-uv[:,0]
    e1=p[:,1]-p[:,0];e2=p[:,2]-p[:,0];det=du[:,0]*dv[:,1]-du[:,1]*dv[:,0]
    safe=np.where(abs(det)>1e-10,det,1.)
    dpdu=(e1*dv[:,1,None]-e2*du[:,1,None])/safe[:,None]
    dpdv=(-e1*dv[:,0,None]+e2*du[:,0,None])/safe[:,None]
    points=(p[tri_id]*bary[yy,xx,:,None]).sum(1)
    normal=(vertex_normals(vertices,triangles)[triangles[tri_id]]*bary[yy,xx,:,None]).sum(1)
    normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-9)
    tangent=dpdu[tri_id]-normal*(dpdu[tri_id]*normal).sum(1)[:,None]
    tangent/=np.maximum(np.linalg.norm(tangent,axis=1,keepdims=True),1e-9)
    bitangent=np.cross(normal,tangent)
    handedness=np.where((bitangent*dpdv[tri_id]).sum(1)<0,-1.,1.)
    bitangent*=handedness[:,None]
    footprint=np.maximum(np.linalg.norm(dpdu[tri_id],axis=1),np.linalg.norm(dpdv[tri_id],axis=1))/res
    # A common field wavelength keeps the same world location identical at seams.
    wavelength=max(.0004,float(np.quantile(footprint, .95))*4)
    detail=field(points,normal,tangent,bitangent,np.full(len(points),wavelength),strength_um)
    detail[abs(det[tri_id])<=1e-10]=[0,0,1]
    image=np.tile(np.array([128,128,255],np.uint8),(res,res,1))
    image[yy,xx]=np.uint8(np.clip(detail*.5+.5,0,1)*255+.5)
    _,nearest=distance_transform_edt(ids<0,return_indices=True)
    image[ids<0]=image[nearest[0][ids<0],nearest[1][ids<0]]
    return image,dict(method='metric-space sinusoidal artist field; tangent-space normal bake',
                      strength_um=strength_um,wavelength_m=wavelength,bandlimit='four texels at 95th-percentile footprint',
                      inferred_from_portrait=False,geometry_changed=False)


def apply(candidate,out,strength_um=5.):
    candidate,out=Path(candidate),artifact_path(out)
    with np.load(candidate/'geometry.npz',allow_pickle=False) as z:
        geometry=z['neutral'];tri=z['triangles'];uv=z['triangle_uvs']
    with Image.open(candidate/'skin_normal.png') as image:res=image.width
    normal,report=atlas(geometry,tri,uv,res,strength_um)
    out.mkdir(parents=True,exist_ok=False);Image.fromarray(normal).save(out/'skin_normal.png')
    report.update(source_geometry_sha256=sha256(candidate/'geometry.npz'),normal_sha256=sha256(out/'skin_normal.png'))
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--candidate',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--strength-um',type=float,default=5.)
    a=parser.parse_args();print(json.dumps(apply(a.candidate,a.out,a.strength_um)))


if __name__=='__main__':main()
