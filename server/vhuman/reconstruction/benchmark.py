"""Frozen cross-expression evaluation; target surfaces never adjust predictions."""
import argparse
import json
from pathlib import Path
import numpy as np
from .artifacts import artifact_path
from .scan import closest_surface
from .observations import sha256


def area_samples(vertices,triangles,count=600,seed=71):
    """Deterministic uniform-area samples, independent of vertex density."""
    p=np.asarray(vertices,float)[triangles]
    area=np.linalg.norm(np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]),axis=1)
    if not np.isfinite(p).all() or area.sum()<=0:raise ValueError('nondegenerate finite surface required')
    rng=np.random.default_rng(seed);ids=rng.choice(len(p),count,p=area/area.sum())
    u=rng.random(count);v=rng.random(count);s=np.sqrt(u)
    bary=np.stack((1-s,s*(1-v),s*v),axis=1)
    return (p[ids]*bary[:,:,None]).sum(1)


def bidirectional(prediction,triangles,reference,reference_triangles,roi_points,count=600):
    # Fixed fitting-model bounds select the same facial region in every target.
    lo,hi=np.min(roi_points,axis=0),np.max(roi_points,axis=0)
    def region(v,t):
        centres=v[t].mean(1)
        keep=((centres>=lo)&(centres<=hi)).all(1)
        if keep.sum()<1:raise ValueError('insufficient triangles inside fixed training ROI')
        return t[keep]
    pt=region(prediction,triangles);rt=region(reference,reference_triangles)
    a=area_samples(prediction,pt,count);b=area_samples(reference,rt,count)
    forward=closest_surface(a,reference,rt)[3];reverse=closest_surface(b,prediction,pt)[3]
    def stats(d):return dict(rms_mm=float(np.sqrt(np.mean(d*d))*1000),median_mm=float(np.median(d)*1000),p95_mm=float(np.quantile(d,.95)*1000))
    return dict(forward=stats(forward),reverse=stats(reverse),symmetric_rms_mm=float(np.sqrt(np.mean(np.r_[forward,reverse]**2))*1000),samples_per_direction=count,
                metric='uniform-area bidirectional exact point-to-triangle; fixed fitting ROI bounds')


def run(scan,target,out):
    scan,target,out=Path(scan),Path(target),artifact_path(out)
    fit_report=json.loads((scan/'report.json').read_text());target_report=json.loads((target/'report.json').read_text())
    # Comparing a different expression is mandatory, rather than relabelling camera holdouts.
    source_doc=json.loads((scan/'pca/observations.json').read_text())
    expression=source_doc['provenance']['expression']
    if target_report['provenance']['expression']==expression:raise ValueError('a disjoint target expression is required')
    with np.load(target/'held-out_reference.npz',allow_pickle=False) as z:truth=z['positions'];rt=z['triangles']
    with np.load(scan/'correspondence.npz',allow_pickle=False) as z:roi=z['roi']
    with np.load(scan/'pca/geometry.npz',allow_pickle=False) as z:roi_points=z['neutral'][roi]
    rows=[]
    for tag in ('pca','dense'):
        with np.load(scan/tag/'geometry.npz',allow_pickle=False) as z:p=z['neutral'];t=z['triangles']
        rows.append(dict(candidate=tag,geometry_sha256=sha256(scan/tag/'geometry.npz'),frames=[bidirectional(p,t,ref,rt,roi_points) for ref in truth]))
    result=dict(format='vhuman.cross_expression.v1',training_expression=expression,target_expression=target_report['provenance']['expression'],
                scan_report_sha256=sha256(scan/'report.json'),reference_sha256=sha256(target/'held-out_reference.npz'),comparisons=rows,
                target_used_for_fitting=False,prediction='frozen neutral geometry; no target expression controls supplied',
                default_enabled=False,limitations=['one identity; expression generalization diagnostic, not animated expression prediction',
                    'fixed head-local coordinate frame; no target-dependent ICP or alignment','ROI excludes eyes/neck boundaries only through fixed training bounds'])
    out.mkdir(parents=True,exist_ok=False);(out/'report.json').write_text(json.dumps(result,indent=2)+'\n');return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('scan','target','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();print(json.dumps(run(a.scan,a.target,a.out)))

if __name__=='__main__':main()
