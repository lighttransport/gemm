"""Surface-space tone repair with explicit photo protection and bounded color changes.

No image generator runs here. Existing synthetic detail is retained while a
normal-compatible metric graph regularizes broad tone. Optional source repairs
are separately masked and never relabelled as new photographic evidence.
"""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix,diags
from scipy.sparse.linalg import spsolve
from scipy.ndimage import distance_transform_edt
from .observations import sha256
from .provenance import validate_candidate
from .reference import srgb_to_linear,linear_to_srgb
from .mv_texture import seam_energy


def tone_field(points,normals,colors,protected, *, spacing=.003,strength=8.,prior=.3):
    """Solve RGB tone on a voxel graph; interpolate a bounded log-gain to texels.

    Normal bins prevent opposing skin sheets sharing a voxel. High-frequency
    color variation survives as a multiplicative residual. The graph is a local
    surface approximation, not a claim of recovered hidden reflectance.
    """
    points,normals,colors=np.asarray(points,float),np.asarray(normals,float),np.asarray(colors,float)
    protected=np.asarray(protected,bool)
    if (points.ndim!=2 or points.shape[1]!=3 or normals.shape!=points.shape or colors.shape!=points.shape
        or protected.shape!=(len(points),) or not protected.any() or len(points)<2
        or not all(np.isfinite(v).all() for v in (points,normals,colors))
        or not np.isfinite([spacing,strength,prior]).all()
        or spacing<=0 or strength<=0 or prior<=0 or (colors<0).any() or (colors>1).any()):
        raise ValueError('invalid surface tone samples or parameters')
    key=np.column_stack((np.floor(points/spacing),np.floor((normals+1)*2)))
    _,inverse=np.unique(key,axis=0,return_inverse=True);count=np.bincount(inverse);n=len(count)
    def mean(a,weights=None):
        w=np.ones(len(points)) if weights is None else weights
        total=np.bincount(inverse,weights=w,minlength=n)
        result=np.column_stack([np.bincount(inverse,weights=a[:,i]*w,minlength=n) for i in range(3)])
        return result/np.maximum(total[:,None],1e-12),total
    centres,_=mean(points);normal,_=mean(normals);normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-12)
    log=np.log(np.maximum(colors,.008));tone,_=mean(log)
    anchors,anchor_count=mean(log,protected.astype(float));fraction=anchor_count/count
    tree=cKDTree(centres);d,j=tree.query(centres,k=min(25,n));d,j=d[:,1:],j[:,1:]
    row=np.repeat(np.arange(n),j.shape[1]);col=j.ravel();dist=d.ravel()
    agreement=np.maximum((normal[row]*normal[col]).sum(1),0)
    weight=np.exp(-.5*(dist/(spacing*2))**2)*agreement**4*(agreement>.5)*(dist<spacing*4)
    adj=coo_matrix((weight,(row,col)),shape=(n,n)).tocsr();adj=(adj+adj.T)*.5
    degree=np.asarray(adj.sum(1)).ravel();lap=diags(degree)-adj
    # Photo anchors outweigh the weak synthetic tone prior. Changes remain bounded below.
    anchor_weight=20*fraction
    system=strength*lap+diags(np.maximum(degree,1)*(prior+anchor_weight))
    rhs=np.maximum(degree,1)[:,None]*(prior*tone+anchor_weight[:,None]*anchors)
    solved=spsolve(system,rhs)
    correction=np.clip(solved-tone,-.5,.5)
    output=np.empty_like(colors)
    for start in range(0,len(points),16384):
        stop=min(start+16384,len(points));ids=slice(start,stop)
        distance,near=tree.query(points[ids],k=min(12,n));distance=distance.reshape(stop-start,-1);near=near.reshape(stop-start,-1)
        agreement=np.maximum((normals[ids,None]*normal[near]).sum(-1),0)
        w=np.exp(-.5*(distance/(spacing*1.5))**2)*agreement**4*(agreement>.5)
        gain=(correction[near]*w[...,None]).sum(1)/np.maximum(w.sum(1)[:,None],1e-12)
        output[ids]=np.clip(colors[ids]*np.exp(gain),0,1)
    output[protected]=colors[protected]
    return output,dict(nodes=n,spacing_m=spacing,strength=strength,prior=prior,max_log_gain=.5,normal_dot_min=.5)


def refine(candidate, audit, out, *, repair_projection=False,confidence_threshold=0.,strength=8.,prior=.3):
    candidate,audit,out=map(lambda p:Path(p).resolve(),(candidate,audit,out))
    manifest=validate_candidate(candidate);request=json.loads((audit/'request.json').read_text())
    result=json.loads((audit/'result.json').read_text())
    if request['geometry_sha256']!=manifest['geometry_sha256'] or result['geometry_sha256']!=manifest['geometry_sha256']:
        raise ValueError('Blender audit geometry mismatch')
    if sha256(audit/'surface.npz')!=request['surface_sha256'] or sha256(audit/'visibility.npz')!=result['visibility']['visibility_sha256']:
        raise ValueError('Blender audit checksum mismatch')
    if not 0<=confidence_threshold<=1:raise ValueError('invalid confidence threshold')
    if out.exists() and any(out.iterdir()):raise ValueError('surface refinement output must be empty')
    source=Path(request['source']);source_manifest=validate_candidate(source)
    if manifest['portrait_sha256']!=source_manifest['portrait_sha256']:
        raise ValueError('source portrait differs from refinement candidate')
    if sha256(source/'skin_basecolor.png')!=request['source_basecolor_sha256']:raise ValueError('source material changed')
    surface=np.load(audit/'surface.npz',allow_pickle=False);visibility=np.load(audit/'visibility.npz',allow_pickle=False)
    valid=surface['valid'];points=surface['points'];normals=surface['normals'];observed=surface['observed']
    base=np.asarray(Image.open(candidate/'skin_basecolor.png').convert('RGB'));original=np.asarray(Image.open(source/'skin_basecolor.png').convert('RGB'))
    if valid.shape!=base.shape[:2] or original.shape!=base.shape:raise ValueError('audit texture resolution mismatch')
    if np.any(base[valid][observed]!=original[valid][observed]):
        raise ValueError('input already modifies source photographed texels; start from a protected candidate')
    editable=observed&((surface['confidence']<confidence_threshold)|~visibility['visible']) if repair_projection else np.zeros(len(points),bool)
    protected=observed&~editable
    colors=srgb_to_linear(base[valid]/255.)
    corrected,solver=tone_field(points,normals,colors,protected,strength=strength,prior=prior)
    image=base.copy();image[valid]=np.uint8(np.clip(linear_to_srgb(corrected)*255+.5,0,255))
    protected_atlas=np.zeros(valid.shape,bool);protected_atlas[valid]=protected
    image[protected_atlas]=original[protected_atlas]
    distance,near=distance_transform_edt(~valid,return_indices=True);gutter=(~valid)&(distance<=4)
    image[gutter]=image[near[0][gutter],near[1][gutter]]
    out.mkdir(parents=True)
    for file in candidate.iterdir():
        if file.is_file():shutil.copyfile(file,out/file.name)
    Image.fromarray(image).save(out/'skin_basecolor.png')
    mask=np.zeros(valid.shape,np.uint8);mask[valid]=editable*255;Image.fromarray(mask).save(out/'skin_projection_repair.png')
    photo_changed=int(np.any(image[valid][observed]!=original[valid][observed],axis=1).sum())
    protected_changed=int(np.any(image[protected_atlas]!=original[protected_atlas],axis=1).sum())
    old=json.loads((candidate/'generated_skin.json').read_text())
    report=dict(old)
    report.update(basecolor_sha256=sha256(out/'skin_basecolor.png'),photographed_texels_changed=photo_changed,
        surface_refinement=dict(source=str(candidate),source_basecolor_sha256=sha256(candidate/'skin_basecolor.png'),
            blender_audit=str(audit),audit_result_sha256=sha256(audit/'result.json'),solver=solver,
            repair_projection=repair_projection,confidence_threshold=confidence_threshold,
            editable_photographed_texels=int(editable.sum()),protected_photographed_texels=int(protected.sum()),
            protected_texels_changed=protected_changed,repair_mask_sha256=sha256(out/'skin_projection_repair.png'),
            coverage_note='Inherited generator coverage; color regularization adds no observation or view support',
            seam_before=seam_energy(points,colors,observed),seam_after=seam_energy(points,srgb_to_linear(image[valid]/255.),observed)))
    report['limitations']=list(report['limitations'])+['surface tone is a regularized appearance prior, not measured reflectance']
    manifest['material']['synthetic_completion']=report
    manifest['material_refinement']=dict(source=str(candidate),geometry_unchanged=True,projection_repair=repair_projection)
    for name,data in [('generated_skin.json',report),('manifest.json',manifest),('skin_material.json',manifest['material'])]:
        (out/name).write_text(json.dumps(data,indent=2))
    validate_candidate(out)
    return report['surface_refinement']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('candidate','audit','out'):p.add_argument('--'+name,required=True)
    p.add_argument('--repair-projection',action='store_true');p.add_argument('--confidence-threshold',type=float,default=0.,
        help='optional confidence repair threshold; default repairs only Blender-occluded photo samples')
    p.add_argument('--strength',type=float,default=8.);p.add_argument('--prior',type=float,default=.3)
    print(json.dumps(refine(**vars(p.parse_args())),indent=2))


if __name__=='__main__':main()
