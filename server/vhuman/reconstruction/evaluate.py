"""Offline reconstruction quality report. Dataset media stays outside Git.

Held-out observations are the default; training views require an explicit flag
and are labelled diagnostics. No face-recognition score substitutes for likeness.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from . import observations
from .reference import Camera, rasterize, srgb_to_linear


def outline_metrics(predicted, target, allowed, tolerance=2):
    from scipy.ndimage import binary_erosion, distance_transform_edt, binary_fill_holes
    predicted,target = binary_fill_holes(predicted),binary_fill_holes(target)
    p,t = predicted&allowed,target&allowed
    union = (p|t).sum()
    iou = float((p&t).sum()/union) if union else None
    pb = (predicted & ~binary_erosion(predicted)) & allowed
    tb = (target & ~binary_erosion(target)) & allowed
    if not pb.any() or not tb.any():
        return dict(iou=iou,boundary_f1=0.,tolerance_px=tolerance)
    precision = float((distance_transform_edt(~tb)[pb]<=tolerance).mean())
    recall = float((distance_transform_edt(~pb)[tb]<=tolerance).mean())
    return dict(iou=iou,boundary_f1=2*precision*recall/max(precision+recall,1e-12),tolerance_px=tolerance)


def landmark_metrics(vertices, camera, anchors, canonical, view):
    from .fitting import anchor_indices
    from .correspondence import occlusion_weight
    rows = anchor_indices(vertices,camera,anchors,canonical)
    if not rows:
        return dict(count=0,rms_px=None,nme_ipd=None)
    point = np.array([vertices[row[1]].mean(0) for row in rows])
    xy,_ = camera.project(point)
    error = np.linalg.norm(xy-np.array([row[2] for row in rows]),axis=1)
    weight = np.array([row[3]*occlusion_weight(view,row[2]) for row in rows])
    rms = float(np.sqrt(np.sum(weight*error**2)/weight.sum())) if weight.sum()>0 else None
    ipd = None
    if all(k in anchors for k in ('eye_right','eye_left')):
        ipd = np.linalg.norm(np.array(anchors['eye_right']['xy'])-anchors['eye_left']['xy'])
    return dict(count=int((weight>0).sum()),rms_px=rms,nme_ipd=rms/float(ipd) if rms is not None and ipd and ipd>1 else None,
                errors_px={row[0]:float(err) for row,err in zip(rows,error)},
                weights={row[0]:float(w) for row,w in zip(rows,weight)})


def temporal_error(predicted, reference, timestamps):
    """Velocity/acceleration *error*, not suppression of real expression motion."""
    dt = np.diff(timestamps)
    if len(dt)<1 or not np.isfinite(dt).all() or (dt<=0).any():
        raise ValueError('sequence timestamps must increase')
    e = np.asarray(predicted)-reference
    velocity = np.diff(e,axis=0)/dt[:,None,None]
    report = dict(velocity_error_rms=float(np.sqrt(np.mean(velocity**2))))
    if len(dt)>1:
        acceleration = np.diff(velocity,axis=0)/((dt[1:]+dt[:-1])/2)[:,None,None]
        report['acceleration_error_rms'] = float(np.sqrt(np.mean(acceleration**2)))
    return report


def align_pose(vertices, camera, view, canonical):
    """Fit only pose to rigid annotations; score mouth separately afterward."""
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    from .fitting import anchor_indices
    from .correspondence import occlusion_weight
    rigid_names = ('eye_right','eye_left','nose_tip','menton')
    anchors = {k:v for k,v in view.get('anchors',{}).items() if k in rigid_names}
    rows = anchor_indices(vertices,camera,anchors,canonical)
    rows = [row for row in rows if row[3]*occlusion_weight(view,row[2])>.1]
    if len(rows)<4:
        return camera,dict(status='skipped',reason='four unoccluded rigid annotations required')
    points = np.array([vertices[row[1]].mean(0) for row in rows])
    target = np.array([row[2] for row in rows])
    weights = np.sqrt([row[3]*occlusion_weight(view,row[2]) for row in rows])[:,None]
    def objective(x):
        moved = points@Rotation.from_rotvec(x[:3]).as_matrix().T+x[3:]
        pixels,z = camera.project(moved)
        return np.concatenate((((pixels-target)*weights/3).reshape(-1),np.minimum(z-.02,0)*1000,x[:3]*.05,x[3:]))
    solve = least_squares(objective,np.zeros(6),bounds=([-0.8]*3+[-.05]*3,[.8]*3+[.05]*3),loss='soft_l1',max_nfev=80)
    before,after = float(np.mean(objective(np.zeros(6))**2)),float(np.mean(objective(solve.x)**2))
    if not solve.success or after>=before:
        return camera,dict(status='rejected',objective_before=before,objective_after=after)
    rot = Rotation.from_rotvec(solve.x[:3]).as_matrix()
    fitted = Camera(camera.focal,camera.cx,camera.cy,rot.T@(camera.origin-solve.x[3:]),camera.rotation@rot,
                    camera.focal_y,camera.skew)
    return fitted,dict(status='pose alignment diagnostic',fitted_anchor_names=[row[0] for row in rows],
                       rotation_rad=solve.x[:3].tolist(),translation_m=solve.x[3:].tolist(),
                       objective_before=before,objective_after=after,
                       expression_landmarks=landmark_metrics(vertices,fitted,{k:v for k,v in view.get('anchors',{}).items() if k not in rigid_names},canonical,view))


def evaluate(candidate, observation_file, out, *, allow_training=False, surfaces=None, reference=None, pose_align=False):
    from ..rig import face_models
    from .correspondence import attachments
    candidate,out = Path(candidate),Path(out)
    manifest = json.loads((candidate/'manifest.json').read_text())
    training = json.loads((candidate/'observations.json').read_text())
    training_hashes = {digest for v in training['views'] for digest in (v['sha256'],v.get('source_sha256')) if digest}
    training_pixels = []
    for view in training['views']:
        image = candidate/view.get('image','')
        digest = observations.pixel_sha256(image) if image.is_file() else view.get('pixel_sha256')
        training_pixels.append(digest)
    doc = observations.load(observation_file)
    overlap = [v['sha256'] in training_hashes or v['pixel_sha256'] in training_pixels for v in doc['views']]
    if any(overlap) and not allow_training:
        raise ValueError('evaluation overlaps training images; use held-out views or --allow-training-diagnostic')
    source = face_models.load(manifest['face_model'])
    if manifest.get('source_loaded_sha256') and face_models.fingerprint(source)!=manifest['source_loaded_sha256']:
        raise ValueError('evaluation source fingerprint changed')
    canonical = manifest['geometry'].get('anatomical_anchors',attachments(source))
    if manifest.get('geometry_sha256') and observations.sha256(candidate/'geometry.npz')!=manifest['geometry_sha256']:
        raise ValueError('evaluation candidate geometry hash changed')
    with np.load(candidate/'geometry.npz',allow_pickle=False) as z:
        neutral,tri,uv,captured = z['neutral'],z['triangles'],z['triangle_uvs'],z['captured']
    predicted = np.repeat(neutral[None],len(doc['views']),axis=0)
    cameras = [Camera.from_dict(v['camera']) for v in doc['views']]
    if allow_training:
        for i,v in enumerate(doc['views']):
            indices = [j for j,row in enumerate(training['views']) if v['sha256'] in (row['sha256'],row.get('source_sha256'))
                       or v['pixel_sha256']==training_pixels[j]]
            if len(indices)==1:
                j = indices[0]
                predicted[i] = captured[j]
                cameras[i] = Camera.from_dict(manifest['geometry']['fitted_cameras'][j])
    if surfaces:
        with np.load(surfaces,allow_pickle=False) as z:
            if not np.array_equal(z['triangles'],tri):
                raise ValueError('evaluation sequence topology mismatch')
            predicted = z['positions']
        if predicted.shape!=(len(doc['views']),len(neutral),3) or not np.isfinite(predicted).all():
            raise ValueError('evaluation sequence shape/values invalid')
    out.mkdir(parents=True,exist_ok=True)
    atlas = np.asarray(Image.open(candidate/'skin_basecolor.png').convert('RGB'))
    rows = []
    for i,(view,camera,p) in enumerate(zip(doc['views'],cameras,predicted)):
        raw_landmarks = landmark_metrics(p,camera,view.get('anchors',{}),canonical,view)
        aligned = None
        if pose_align:
            camera,aligned = align_pose(p,camera,view,canonical)
        w,h = view['size'];s = min(1.,512/max(w,h));size = (round(w*s),round(h*s))
        small = camera.scaled(s)
        tid,bary,_ = rasterize(p,tri,small,size)
        target = np.asarray(Image.open(view['image_path']).convert('RGB').resize(size,Image.Resampling.BILINEAR))
        allowed = np.ones(tid.shape,bool)
        if view.get('exclusion_mask_path'):
            allowed = np.asarray(Image.open(view['exclusion_mask_path']).convert('L').resize(size,Image.Resampling.NEAREST))<128
        row = dict(image_sha256=view['sha256'],split='training diagnostic' if overlap[i] else 'held-out',
                   landmarks=raw_landmarks,comparison_pose='rigid-annotation alignment diagnostic' if pose_align else 'provided camera')
        if aligned is not None:
            row['pose_alignment_diagnostic'] = aligned
        if view.get('silhouette_mask_path'):
            mask = np.asarray(Image.open(view['silhouette_mask_path']).convert('L').resize(size,Image.Resampling.NEAREST))>127
            row['silhouette'] = outline_metrics(tid>=0,mask,allowed)
            row['silhouette']['resolution'] = size
        valid = (tid>=0)&allowed
        yy,xx = np.nonzero(valid)
        rendered = np.full_like(target,32)
        if len(xx):
            tex = (uv[tid[yy,xx]]*bary[yy,xx,:,None]).sum(1)
            ax = np.clip((tex[:,0]*atlas.shape[1]).astype(int),0,atlas.shape[1]-1)
            ay = np.clip((tex[:,1]*atlas.shape[0]).astype(int),0,atlas.shape[0]-1)
            rendered[yy,xx] = atlas[ay,ax]
            # Albedo-vs-photo residual is diagnostic: illumination differs.
            row['albedo_photo_mae_linear_diagnostic'] = float(np.mean(abs(srgb_to_linear(rendered[yy,xx]/255.)-srgb_to_linear(target[yy,xx]/255.))))
        Image.fromarray(np.concatenate((target,rendered),axis=1)).save(out/f'view_{i}_comparison.png')
        rows.append(row)
    report = dict(format='vhuman.reconstruction_evaluation.v1',candidate=manifest['id'],views=rows,
                  geometry_sha256=observations.sha256(candidate/'geometry.npz'),observations_sha256=observations.sha256(observation_file),
                  surfaces_sha256=observations.sha256(surfaces) if surfaces else None,
                  provenance=doc.get('provenance'),
                  limitations=['pixel landmarks depend on annotation quality','albedo/photo comparison is not relighting or likeness ground truth',
                               'no scan quality claim without independent geometry','held-out animated expressions need externally predicted surfaces'])
    if reference:
        with np.load(reference,allow_pickle=False) as z:
            truth = z['positions']
            if not np.array_equal(z['triangles'],tri) or truth.shape!=predicted.shape or not np.isfinite(truth).all():
                raise ValueError('reference geometry topology/shape/values mismatch')
            timestamps = z['timestamps'] if 'timestamps' in z else None
        error = np.linalg.norm(predicted-truth,axis=-1)
        report['geometry'] = dict(rms_mm=float(np.sqrt(np.mean(error**2))*1000),p95_mm=float(np.quantile(error,.95)*1000),reference_sha256=observations.sha256(reference))
        if timestamps is not None and len(predicted)>1:
            if timestamps.shape!=(len(predicted),):
                raise ValueError('reference timestamp count mismatch')
            report['temporal'] = temporal_error(predicted,truth,timestamps)
            report['temporal']['units'] = 'metres/second and metres/second squared, component RMS'
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',required=True,type=Path)
    parser.add_argument('--observations',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--allow-training-diagnostic',action='store_true')
    parser.add_argument('--surfaces',type=Path,help='NPZ positions[V,N,3], triangles[T,3], H-frame metres')
    parser.add_argument('--reference',type=Path,help='Independent same-topology NPZ positions, triangles, optional timestamps[V]')
    parser.add_argument('--pose-align-diagnostic',action='store_true',help='Fit pose to eye/nose/chin annotations; preserve raw error and score mouth separately')
    a = parser.parse_args()
    print(json.dumps(evaluate(a.candidate,a.observations,a.out,allow_training=a.allow_training_diagnostic,surfaces=a.surfaces,reference=a.reference,pose_align=a.pose_align_diagnostic)))


if __name__=='__main__':
    main()
