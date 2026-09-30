"""Offline worker. Publish candidates only after all requested outputs succeed."""
import argparse
import json
import shutil
import time
import uuid
from dataclasses import replace
from pathlib import Path
import numpy as np
from PIL import Image
from ..rig import face_models
from ..rig.common import load_subject, vertex_normals
from . import fitting, materials, observations
from .reference import Camera, pixal_camera

from . import FILES


def subject_override(subject, run):
    """Use candidate skin/geometry with accepted analytic eyes, never old soft weights."""
    from ..eye.geometry import compute_tangents
    run = Path(run)
    with np.load(run/'geometry.npz',allow_pickle=False) as z:
        p, t, uv = z['neutral'],z['triangles'],z['triangle_uvs']
    p = p[t].reshape(-1,3)
    t = np.arange(len(p)).reshape(-1,3)
    uv = uv.reshape(-1,2)
    n = vertex_normals(p,t)
    # Smooth normals across UV seams using original welded normals.
    with np.load(run/'geometry.npz',allow_pickle=False) as z:
        n = vertex_normals(z['neutral'],z['triangles'])[z['triangles']].reshape(-1,3)
    tan = compute_tangents(p,n,uv,t)
    return replace(subject,folder=run,positions=p,normals=n,triangles=t,uvs=uv,tangents=tan)


def direct_seed(image, out, model):
    """GNM surface + existing analytic eye/lid fitting; no generated mesh required."""
    from ..eye.glb import GLBBuilder
    from ..head.camera import PixalCamera
    from ..head.fit import fit_head
    src = face_models.load(model)
    obs = observations.observe(image)
    view = obs['views'][0]
    er, el = [np.array(view['anchors'][n]['xy']) for n in ('eye_right','eye_left')]
    cam = PixalCamera.from_portrait(image,np.deg2rad(20))
    # Model H frame nominal eye centres at +/-32 mm. Assumed 64 mm IPD.
    pixel_ipd = np.linalg.norm(el-er)
    units = pixel_ipd*cam.distance/(cam.focal/cam.scale*.064)
    center = (er+el)/2
    depth = cam.distance
    origin = np.array([-(center[0]*cam.scale-cam.left-cam.side/2)*depth/cam.focal,
                       -(center[1]*cam.scale-cam.top-cam.side/2)*depth/cam.focal,0.])
    model_scale = .064 / np.linalg.norm(src.eye_centers[1]-src.eye_centers[0])
    p = (src.vertices-src.eye_centers.mean(0))*model_scale*np.array([-1,1,-1])*units+origin
    uv = src.triangle_uvs.reshape(-1,2)
    vertices = p[src.triangles].reshape(-1,3)
    tri = np.arange(len(vertices)).reshape(-1,3)
    n = vertex_normals(p,src.triangles)[src.triangles].reshape(-1,3)
    b = GLBBuilder('vhuman direct portrait seed')
    im = np.asarray(Image.open(image).convert('RGB'))
    tex = b.texture(im,'portrait')
    mat = b.material(dict(name='skin',pbrMetallicRoughness=dict(baseColorTexture=dict(index=tex),metallicFactor=0,roughnessFactor=.55)))
    # Seed UV is source atlas; final image-guided atlas is independently baked.
    attrs = dict(POSITION=b.accessor(vertices),NORMAL=b.accessor(n),TEXCOORD_0=b.accessor(uv))
    b.doc['meshes'].append(dict(name='head',primitives=[dict(attributes=attrs,indices=b.accessor(tri,indices=True),material=mat)]))
    b.node('head',mesh=0)
    b.write(out/'head.glb')
    shutil.copyfile(image,out/'portrait.png')
    from ..head.fit import EyePose
    from ..eye import optics
    centers = (src.eye_centers-src.eye_centers.mean(0))*model_scale*np.array([-1,1,-1])*units+origin
    poses = []
    for side,center in zip(('right','left'),centers):
        gaze = cam.origin-center;gaze /= np.linalg.norm(gaze)
        x = np.cross([0,1,0],gaze);x /= np.linalg.norm(x)
        y = np.cross(gaze,x)
        poses.append(EyePose(side,center+gaze*optics.ANATOMICAL.apex_z*units,center,
                             np.column_stack((x,y,gaze)),units))
    fit_head(out/'portrait.png',out/'head.glb',out,res=512,anatomical_poses=poses)
    return obs


def run(folder, *, observation_file=None, profile='full', face_model='gnm_v3',
        res=512, iterations=80, build_rig=False, rig_iters=80, depth_installation=None,
        gaussian_count=0, run_id=None, roughness=.55, f0=.028, detail_um=0., spatial_materials=False, auto_exclusions=False):
    folder = Path(folder)
    if not 0<=detail_um<=30:raise ValueError('detail must be 0..30 micrometres')
    run_id = run_id or uuid.uuid4().hex[:16]
    if not run_id.isalnum() or len(run_id)>32:
        raise ValueError('invalid reconstruction run id')
    final = folder/'reconstruction'/run_id
    staging = folder/'reconstruction'/f'.{run_id}.partial'
    if final.exists() or staging.exists():
        raise ValueError('reconstruction run already exists')
    staging.mkdir(parents=True)
    started = time.perf_counter()
    try:
        subject = load_subject(folder)
        src = face_models.load(face_model)
        initial,scale,rotation,coeff = fitting.initialize(src,subject)
        if observation_file:
            doc = observations.load(observation_file)
        else:
            doc = observations.observe(subject.portrait,pixal_camera(subject))
            for v in doc['views']:
                v['image_path'] = v['image']
        cams = []
        for v in doc['views']:
            if 'camera' not in v:
                if len(doc['views'])>1:
                    raise ValueError('additional views require calibrated H-frame cameras')
                v['camera'] = pixal_camera(subject).as_dict()
            cams.append(Camera.from_dict(v['camera']))
        neutral, captured = initial.astype(np.float32),np.repeat(initial[None].astype(np.float32),len(doc['views']),axis=0)
        report = dict(status='geometry held at surface initialization')
        if profile in ('geometry','full'):
            neutral,captured,cams,report = fitting.fit(src,initial,doc['views'],scale=scale,rotation=rotation,iterations=iterations,
                                                                         surface_prior=(subject.positions,subject.normals))
        exclusions=[]
        if auto_exclusions:
            from .occlusion import bake_masks
            for i,(view,cam) in enumerate(zip(doc['views'],cams)):
                if view.get('exclusion_mask_path'):
                    exclusions.append(dict(status='manual mask retained'));continue
                path=staging/f'auto_exclusion_{i}.png'
                try:
                    exclusion=bake_masks(captured[i],src.triangles,view,cam,path)
                    view['exclusion_mask_path']=str(path)
                    view['exclusion_kind']='photo/model-derived heuristic; not independent annotation'
                except ValueError as exc:exclusion=dict(status='skipped',reason=str(exc))
                exclusions.append(exclusion)
        for i,v in enumerate(doc['views']):
            name = f'view_{i}.png'
            v['source_sha256'] = v['sha256']
            v['pixel_sha256'] = observations.pixel_sha256(v['image_path'])
            Image.open(v['image_path']).convert('RGBA').save(staging/name)
            v['image'],v['sha256'] = name,observations.sha256(staging/name)
            if v.get('exclusion_mask_path'):
                mask = f'exclusion_{i}.png'
                shutil.copyfile(v['exclusion_mask_path'],staging/mask)
                v['exclusion_mask'] = mask
                v['exclusion_mask_sha256'] = observations.sha256(staging/mask)
            if v.get('silhouette_mask_path'):
                mask = f'silhouette_{i}.png'
                Image.open(v['silhouette_mask_path']).convert('L').save(staging/mask)
                v['silhouette_mask'] = mask
                v['silhouette_mask_sha256'] = observations.sha256(staging/mask)
        shutil.copyfile(subject.portrait,staging/'portrait.png')
        serial = {k:v for k,v in doc.items()}
        serial['views'] = [{k:v for k,v in view.items() if not k.endswith('_path')} for view in doc['views']]
        (staging/'observations.json').write_text(json.dumps(serial,indent=2))
        # Bake material in observed expression; neutral exported separately.
        depth_report = None
        if depth_installation:
            from . import depth
            from .reference import rasterize
            depth_report = depth.infer(subject.portrait,depth_installation,staging/'depth.npy')
            relative = np.load(staging/'depth.npy',allow_pickle=False)
            # Full resolution alignment is intentionally optional and bounded.
            if relative.size>4_000_000:
                raise ValueError('depth image exceeds 4M pixels')
            tid,_,metric = rasterize(captured[0],src.triangles,cams[0],(relative.shape[1],relative.shape[0]))
            confidence = (tid>=0).astype(float)
            try:
                aligned,gate = depth.align(relative,metric,confidence)
                depth_report['alignment'] = gate
                if profile in ('geometry','full'):
                    anchors = doc['views'][0].get('anchors', {})
                    protected = [(a['xy'], 12.) for name,a in anchors.items() if 'eye' in name or 'lip' in name or 'mouth' in name]
                    refined, correction = depth.refine(captured[0],src.triangles,cams[0],aligned,confidence,protected)
                    delta = refined-captured[0]
                    neutral += delta
                    captured += delta[None]
                    depth_report['correction'] = correction
                    depth_report['applied_to_geometry'] = correction.get('step',0)>0
                else:
                    depth_report['applied_to_geometry'] = False
            except ValueError as exc:
                depth_report.update(rejected=str(exc),applied_to_geometry=False)
            (staging/'depth.json').write_text(json.dumps(depth_report,indent=2))
        np.savez_compressed(staging/'geometry.npz',neutral=neutral,captured=captured,
                            triangles=src.triangles,triangle_uvs=src.triangle_uvs,scale=scale,rotation=rotation)
        material = materials.bake_portrait(captured,src.triangles,src.triangle_uvs,doc['views'],cams,staging,res=res,roughness=roughness,f0=f0,spatial_materials=spatial_materials)
        if detail_um:
            from .detail import atlas
            normal,detail=atlas(neutral,src.triangles,src.triangle_uvs,res,detail_um)
            Image.fromarray(normal).save(staging/'skin_normal.png')
            material['authored_detail']=detail
            (staging/'skin_material.json').write_text(json.dumps(material,indent=2))
        if gaussian_count:
            from . import gaussian
            # Observed skin only; analytic eyes and mouth are never attachment sources.
            binding = gaussian.bind(neutral,src.triangles,count=gaussian_count)
            binding['triangles'] = src.triangles
            binding['normal_offset'][:] = .0003
            gaussian.fit_radiance(binding,captured,src.triangles,doc['views'],cams)
            gaussian.save(binding,staging/'gaussians.json')
        report['fitted_cameras'] = [cam.as_dict() for cam in cams]
        manifest = dict(format='vhuman.reconstruction.v1',id=run_id,profile=profile,face_model=face_model,
                        source=src.provenance,source_loaded_sha256=face_models.fingerprint(src),source_head_sha256=observations.sha256(folder/'head_eyes.glb'),
                        portrait_sha256=observations.sha256(subject.portrait),
                        topology_sha256=fitting.topology_hash(src.triangles),geometry_sha256=observations.sha256(staging/'geometry.npz'),geometry=report,material=material,
                        depth=depth_report,gaussians=gaussian_count,
                        auto_exclusions=exclusions,
                        deformation='rest geometry changed: prior compact soft deformer invalid; rebuild required',
                        config=dict(res=res,iterations=iterations,roughness=roughness,f0=f0,detail_um=detail_um,spatial_materials=spatial_materials,auto_exclusions=auto_exclusions),seconds=round(time.perf_counter()-started,2))
        (staging/'manifest.json').write_text(json.dumps(manifest,indent=2))
        if build_rig:
            from ..rig.build import assemble
            assemble(folder,staging/'rig',res=max(1024,res),iters=rig_iters,face_model=face_model,
                     reconstruction=staging,deformer_samples=0,preview=True)
        final.parent.mkdir(parents=True,exist_ok=True)
        staging.replace(final)
        return manifest
    except BaseException:
        shutil.rmtree(staging,ignore_errors=True)
        raise


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('folder')
    ap.add_argument('--observations')
    ap.add_argument('--profile',choices=['geometry','material','full'],default='full')
    ap.add_argument('--face-model',choices=['gnm_v3','ict_facekit_light'],default='gnm_v3')
    ap.add_argument('--res',type=int,default=512)
    ap.add_argument('--iterations',type=int,default=80)
    ap.add_argument('--rig',action='store_true')
    ap.add_argument('--run-id')
    ap.add_argument('--portrait')
    ap.add_argument('--depth-installation')
    ap.add_argument('--gaussians',type=int,default=0)
    ap.add_argument('--roughness',type=float,default=.55)
    ap.add_argument('--f0',type=float,default=.028)
    ap.add_argument('--detail-um',type=float,default=0.,help='optional authored normal detail, 0..30 micrometres')
    ap.add_argument('--spatial-materials',action='store_true',help='regional reflectance fitting only with calibrated multi-light observations')
    ap.add_argument('--auto-exclusions',action='store_true',help='optional photo/model-derived occlusion heuristic for texture baking')
    a = ap.parse_args()
    if a.portrait:
        direct_seed(Path(a.portrait),Path(a.folder),a.face_model)
    result = run(a.folder,observation_file=a.observations,profile=a.profile,face_model=a.face_model,
                 res=a.res,iterations=a.iterations,build_rig=a.rig,depth_installation=a.depth_installation,
                 gaussian_count=a.gaussians,run_id=a.run_id,roughness=a.roughness,f0=a.f0,detail_um=a.detail_um,spatial_materials=a.spatial_materials,auto_exclusions=a.auto_exclusions)
    print(json.dumps(result))

if __name__=='__main__':
    main()
