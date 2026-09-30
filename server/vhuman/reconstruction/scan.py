"""Original metric scan alignment and cross-topology point/surface diagnostics.

Alignment and dense correction use fitting cameras/frames only. Held-out scan
geometry is used solely for scoring. Detector-derived anchors are not manual GT.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from . import observations, fitting, correspondence
from .reference import Camera, rasterize
from .artifacts import artifact_path


def closest_surface(points,vertices,triangles,chunk=8):
    """Exact Euclidean closest point over every triangle, with bounded chunks."""
    a,b,c=np.asarray(vertices,float)[triangles].transpose(1,0,2)
    ab,ac=b-a,c-a
    d00=(ab*ab).sum(1);d01=(ab*ac).sum(1);d11=(ac*ac).sum(1)
    denominator=d00*d11-d01*d01
    active=denominator>1e-20
    results=[];ids=[];barys=[];distances=[]
    for start in range(0,len(points),chunk):
        p=np.asarray(points[start:start+chunk],float)[:,None,:]
        ap=p-a;d20=(ap*ab).sum(-1);d21=(ap*ac).sum(-1)
        v=(d11*d20-d01*d21)/np.maximum(denominator,1e-20)
        w=(d00*d21-d01*d20)/np.maximum(denominator,1e-20)
        bary=np.stack((1-v-w,v,w),-1)
        projection=a+v[...,None]*ab+w[...,None]*ac
        distance=((projection-p)**2).sum(-1)
        distance[~((bary>=0).all(-1)&active)]=np.inf
        for i,j in ((0,1),(1,2),(2,0)):
            x,y=(a,b,c)[i],(a,b,c)[j];edge=y-x
            t=np.clip(((p-x)*edge).sum(-1)/np.maximum((edge*edge).sum(-1),1e-20),0,1)
            q=x+t[...,None]*edge;d=((p-q)**2).sum(-1);take=d<distance
            projection[take]=q[take];distance[take]=d[take]
            edge_bary=np.zeros_like(bary);edge_bary[...,i]=1-t;edge_bary[...,j]=t
            bary[take]=edge_bary[take]
        index=np.argmin(distance,axis=1);row=np.arange(len(index))
        results.extend(projection[row,index]);ids.extend(index);barys.extend(bary[row,index]);distances.extend(np.sqrt(distance[row,index]))
    return np.asarray(results),np.asarray(ids,np.int32),np.asarray(barys),np.asarray(distances)


def similarity(source,target):
    source,target=np.asarray(source,float),np.asarray(target,float)
    if source.shape!=target.shape or source.ndim!=2 or source.shape[1]!=3 or len(source)<4:
        raise ValueError('at least four corresponding 3D anchors required')
    x,y=source-source.mean(0),target-target.mean(0)
    if np.linalg.matrix_rank(x)<2 or np.linalg.matrix_rank(y)<2:raise ValueError('degenerate 3D anchors')
    u,s,vt=np.linalg.svd(y.T@x)
    sign=np.ones(3);sign[-1]=np.sign(np.linalg.det(u@vt))
    rotation=u@np.diag(sign)@vt;scale=float((s*sign).sum()/(x*x).sum())
    if not .5<scale<2:raise ValueError('scan/model metric scale mismatch')
    translation=target.mean(0)-scale*source.mean(0)@rotation.T
    return scale,rotation,translation


def ray_hit(vertices,triangles,camera,xy):
    origin=camera.origin;ray=camera.rays(np.asarray(xy,float)[None])[0]
    a,b,c=vertices[triangles].transpose(1,0,2);e1,e2=b-a,c-a
    h=np.cross(ray,e2);det=(e1*h).sum(1)
    inv=np.divide(1.,det,out=np.zeros_like(det),where=abs(det)>1e-10)
    s=origin-a;u=inv*(s*h).sum(1);q=np.cross(s,e1);v=inv*(q*ray).sum(1);t=inv*(q*e2).sum(1)
    valid=(abs(det)>1e-10)&(u>=0)&(v>=0)&(u+v<=1)&(t>0)
    if not valid.any():raise ValueError('annotation ray misses reference skin')
    index=int(np.argmin(np.where(valid,t,np.inf)))
    return origin+t[index]*ray,index,np.array([1-u[index]-v[index],u[index],v[index]])


def surface_metrics(prediction,reference,reference_triangles,ids):
    _,_,_,dist=closest_surface(prediction[ids],reference,reference_triangles)
    return dict(samples=len(dist),rms_mm=float(np.sqrt(np.mean(dist**2))*1000),
                median_mm=float(np.median(dist)*1000),p95_mm=float(np.quantile(dist,.95)*1000),
                metric='exact one-way point-to-triangle distance; fixed source sample IDs')


def run(prepared,out,model='gnm_v3'):
    from ..rig import face_models
    from ..rig.common import vertex_normals
    from .materials import bake_portrait
    from .evaluate import evaluate,outline_metrics
    prepared,out=Path(prepared),artifact_path(out)
    if out.exists():raise ValueError('choose a fresh scan output directory')
    train=observations.load(prepared/'fit.json');holdout=observations.load(prepared/'held-out.json')
    if any(not v.get('anchors') for v in train['views']):raise ValueError('prepare with --annotate-multiface')
    with np.load(prepared/'fit_reference.npz',allow_pickle=False) as z:truth=z['positions'];reference_tri=z['triangles']
    with np.load(prepared/'held-out_reference.npz',allow_pickle=False) as z:held_truth=z['positions']
    source=face_models.load(model);anatomy=correspondence.attachments(source)
    names=('eye_right','eye_left','nose_tip','menton','mouth_right','mouth_left','upper_lip','lower_lip')
    source_points=[];targets=[];anchor_checks={}
    first=train['views'][0]['frame_id']
    for name in names:
        samples=[]
        for i,view in enumerate(train['views']):
            if view['frame_id']!=first:continue
            try:point,_,_=ray_hit(truth[i],reference_tri,Camera.from_dict(view['camera']),view['anchors'][name]['xy'])
            except ValueError:continue
            samples.append(point)
        if len(samples)<2:continue
        disagreement=float(np.max(np.linalg.norm(np.asarray(samples)-np.mean(samples,axis=0),axis=1)))
        if disagreement>.012:continue
        source_points.append(source.vertices[anatomy[name]].mean(0));targets.append(np.mean(samples,axis=0))
        anchor_checks[name]=dict(multi_view_spread_mm=disagreement*1000,target=np.mean(samples,axis=0).tolist())
    scale,rotation,translation=similarity(source_points,targets)
    initial=scale*source.vertices@rotation.T+translation
    neutral,captured,cameras,fit_report=fitting.fit(source,initial,train['views'],scale=scale,rotation=rotation,iterations=60,
        freeze_pose=True,expression_groups=[v['frame_id'] for v in train['views']])
    # Fixed evaluation region derives only from model anatomy and training view.
    centre=np.mean([anchor_checks[n]['target'] for n in ('eye_right','eye_left')],axis=0)
    local=(initial-centre)@rotation
    roi=(abs(local[:,0])<.073)&(local[:,1]>-.105)&(local[:,1]<.045)&(local[:,2]>-.04)
    protect=np.zeros(len(initial),bool)
    for name,radius in [('eye_right',.019),('eye_left',.019),('upper_lip',.012),('lower_lip',.012),
                        ('mouth_right',.012),('mouth_left',.012),('nose_tip',.01),('menton',.012)]:
        protect|=np.linalg.norm(initial-initial[anatomy[name]].mean(0),axis=1)<radius
    ids=np.flatnonzero(roi)[::max(1,int(roi.sum())//600)]
    # Freeze barycentric attachments on the fitting scan, retain authored anatomy.
    target,reference_ids,bary,distance=closest_surface(neutral,truth[0],reference_tri)
    n=vertex_normals(neutral,source.triangles);rn=vertex_normals(truth[0],reference_tri)
    tn=(rn[reference_tri[reference_ids]]*bary[...,None]).sum(1)
    tn/=np.maximum(np.linalg.norm(tn,axis=1,keepdims=True),1e-12)
    agreement=(n*tn).sum(1)
    confidence=roi&(~protect)&(distance<.008)&(agreement>.75)
    delta=(target-neutral)*confidence[:,None]
    # Original sparse Laplacian regularization keeps correction continuous.
    from scipy.sparse import coo_matrix,eye,diags
    from scipy.sparse.linalg import spsolve
    edges=np.concatenate((source.triangles[:,[0,1]],source.triangles[:,[1,2]],source.triangles[:,[2,0]]))
    edges=np.unique(np.sort(edges,axis=1),axis=0)
    row=np.r_[edges[:,0],edges[:,1]];col=np.r_[edges[:,1],edges[:,0]]
    adjacency=coo_matrix((np.ones(len(row)),(row,col)),shape=(len(neutral),len(neutral))).tocsr()
    lap=diags(np.asarray(adjacency.sum(1)).ravel())-adjacency
    weights=confidence.astype(float)+protect.astype(float)*100
    smooth=spsolve(diags(weights)+lap*2+eye(len(neutral))*1e-4,delta*weights[:,None])
    smooth[protect|~roi]=0
    margin=np.minimum.reduce((.073-abs(local[:,0]),local[:,1]+.105,.045-local[:,1],local[:,2]+.04))
    smooth*=np.clip(margin/.006,0,1)[:,None]
    norm=np.linalg.norm(smooth,axis=1);smooth*=np.minimum(1,.002/np.maximum(norm,1e-12))[:,None]
    step=1.
    while not fitting.safe_geometry(neutral,neutral+smooth*step,source.triangles) and step>1/128:step*=.5
    if not fitting.safe_geometry(neutral,neutral+smooth*step,source.triangles):step=0.
    corrected=neutral+smooth*step
    out.mkdir(parents=True)
    np.savez_compressed(out/'correspondence.npz',reference_triangle=reference_ids,barycentric=bary,
                        confidence=confidence,protected=protect,roi=roi,source_triangles=source.triangles)
    reports=[]
    for tag,p in [('pca',neutral),('dense',corrected)]:
        folder=out/tag;folder.mkdir()
        captures=captured+(p-neutral)[None]
        np.savez_compressed(folder/'geometry.npz',neutral=p,captured=captures,triangles=source.triangles,
                            triangle_uvs=source.triangle_uvs,scale=scale,rotation=rotation)
        serial=dict(train);serial['views']=[{k:v for k,v in view.items() if not k.endswith('_path')} for view in train['views']]
        (folder/'observations.json').write_text(json.dumps(serial,indent=2))
        material=bake_portrait(captures,source.triangles,source.triangle_uvs,train['views'],cameras,folder,res=256)
        fit_report['fitted_cameras']=[c.as_dict() for c in cameras]
        manifest=dict(format='vhuman.reconstruction.v1',id=tag,face_model=model,source_loaded_sha256=face_models.fingerprint(source),
                      geometry_sha256=observations.sha256(folder/'geometry.npz'),geometry=fit_report,material=material)
        (folder/'manifest.json').write_text(json.dumps(manifest,indent=2))
        sameframe=np.array([captures[next(j for j,v in enumerate(train['views']) if v['frame_id']==view['frame_id'])]
                            for view in holdout['views']])
        np.savez_compressed(folder/'heldout_prediction.npz',positions=sameframe,triangles=source.triangles)
        evaluation=evaluate(folder,prepared/'held-out.json',out/(tag+'-held-out'),surfaces=folder/'heldout_prediction.npz')
        geometry=[];silhouettes=[]
        # Same-frame fitting expression is used; held-out camera never adjusts it.
        for i,view in enumerate(holdout['views']):
            match=next(j for j,v in enumerate(train['views']) if v['frame_id']==view['frame_id'])
            surface=captures[match]
            metric=surface_metrics(surface,held_truth[i],reference_tri,ids)
            from .benchmark import bidirectional
            metric['bidirectional']=bidirectional(surface,source.triangles,held_truth[i],reference_tri,neutral[roi])
            metric['reference_used_for_alignment_and_dense_fit']=view['frame_id']==first
            geometry.append(metric)
            cam=Camera.from_dict(view['camera']);size=view['size'];s=256/max(size);res=tuple(round(n*s) for n in size)
            prediction=rasterize(surface,source.triangles,cam.scaled(s),res)[0]>=0
            reference=rasterize(held_truth[i],reference_tri,cam.scaled(s),res)[0]>=0
            silhouettes.append(outline_metrics(prediction,reference,np.ones_like(prediction)))
        row=dict(candidate=tag,geometry=geometry,silhouette=silhouettes,
                 landmarks=[v['landmarks'] for v in evaluation['views']])
        reports.append(row)
    baseline_error=np.mean([r['rms_mm'] for r in reports[0]['geometry']])
    dense_error=np.mean([r['rms_mm'] for r in reports[1]['geometry']])
    landmark_preserved=all(abs(a['rms_px']-b['rms_px'])<.01 for a,b in zip(reports[0]['landmarks'],reports[1]['landmarks']))
    bidirectional_before=np.mean([r['bidirectional']['symmetric_rms_mm'] for r in reports[0]['geometry']])
    bidirectional_after=np.mean([r['bidirectional']['symmetric_rms_mm'] for r in reports[1]['geometry']])
    bidirectional_gate=bidirectional_after<bidirectional_before*.95
    result=dict(format='vhuman.scan_evaluation.v1',model=model,scale=scale,rotation=rotation.tolist(),translation_m=translation.tolist(),
                alignment_anchors=anchor_checks,correspondence_sha256=observations.sha256(out/'correspondence.npz'),
                dense_step=step,corrected_vertices=int(confidence.sum()),max_correction_mm=float(np.linalg.norm(smooth*step,axis=1).max()*1000),
                fitting_reference_sha256=observations.sha256(prepared/'fit_reference.npz'),
                heldout_reference_sha256=observations.sha256(prepared/'held-out_reference.npz'),comparisons=reports,
                dense_gate_passed=bool(dense_error<baseline_error*.95 and bidirectional_gate and landmark_preserved),
                bidirectional_gate_passed=bool(bidirectional_gate),
                default_enabled=False,calibrated_cameras_fixed=True,
                limitations=['per-run subject; aggregate independent subjects before adopting; detector/ray anchors are not authored landmarks',
                             'geometry for fitting frames diagnoses scan fit, not unseen-expression generalization',
                             'fixed-ROI one-way and uniform-area bidirectional surface error; not anatomical point error',
                             'reference silhouettes derive from tracked meshes, not manual segmentation'])
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--model',choices=['gnm_v3','ict_facekit_light'],default='gnm_v3')
    a=parser.parse_args();print(json.dumps(run(a.prepared,a.out,a.model)))


if __name__=='__main__':main()
