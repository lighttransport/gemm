"""Isolated full-equation CPU refinement of an existing candidate."""
import json
from pathlib import Path
import shutil
import copy
import numpy as np

from .gnm_fit import solve
from .gnm_motion import candidate_bind,review_portrait
from .observations import load,sha256
from .provenance import validate_candidate
from .reference import Camera
from ..rig.gnm_model import GNMModel


def refine(candidate,out, *, iterations=400,modes=64,identity_modes=0,target_px=None,threads=16,cancel=None,parsing_model=None):
    candidate,out=Path(candidate).resolve(),Path(out).resolve()
    if out.exists() and any(out.iterdir()): raise ValueError('refinement output must be empty')
    manifest=validate_candidate(candidate);bind=candidate_bind(candidate);doc=load(candidate/'observations.json')
    views=doc['views'];count=len(views)
    if parsing_model is not None:
        from ..face_parsing import FaceParser
        from .occlusion import parsing_masks
        out.mkdir(parents=True,exist_ok=True)
        parser=FaceParser(model=parsing_model,threads=threads)
        for i,view in enumerate(views): parsing_masks(view,out/f'view_{i}_parsing',parser)
    for v,c in zip(views,manifest['geometry']['fitted_cameras']): v['camera']=c
    with np.load(candidate/'geometry.npz',allow_pickle=False) as z: geometry={k:z[k] for k in z.files}
    model=GNMModel()
    if 'gnm_expressions' in geometry:
        expression=geometry['gnm_expressions']
    else:
        expression=np.zeros((count,model.expression_dim))
        expression[:,manifest['geometry'].get('expression_modes',[])]=manifest['geometry'].get('expression_coefficients',[])
    rotations=geometry.get('gnm_rotations',np.zeros((count,4,3)))
    translation=geometry.get('gnm_translations',np.zeros((count,3)))
    result=solve(views,identity=bind['identity'],expression=expression,rotations=rotations,translation=translation,
                 scale=float(bind['scale']),alignment=bind['alignment'],offset=bind['offset'],bind_residual=bind['bind_residual'],
                 identity_modes=identity_modes,modes=modes,iterations=iterations,threads=threads,cancel=cancel)
    report=dict(result['report'],schema='vhuman.native_portrait_refinement.v3',
                source_geometry_sha256=manifest['geometry_sha256'],camera_frozen=True,surface_limit_mm=0.,target_px=target_px)
    report['accepted']=report['quality_gate'] and (target_px is None or (report['after']['heldout_px'] is not None and report['after']['heldout_px']<target_px))
    out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'landmark_diagnostic.npz',before=result['baseline_landmarks'],after=result['projected_landmarks'],
                        target=result['observed_landmarks'],weights=result['weights'],heldout=result['heldout'],
                        identity=result['identity'],expression=result['expression'],rotations=result['rotations'],translation=result['translation'],
                        **{'proposal_'+k:v for k,v in (result.get('last_proposal') or {}).items()})
    (out/'fit_refinement.json').write_text(json.dumps(report,indent=2))
    if not report['accepted']:
        review_portrait(out,views[0],result)
        return report
    for path in candidate.iterdir():
        if path.is_file() and path.name not in ('manifest.json','geometry.npz','fit_refinement.json','landmark_diagnostic.npz',
                                                'generated_skin.json','skin_generated_support.png','skin_projection_repair.png','fit_comparison.png','mesh_preview.png'):
            shutil.copyfile(path,out/path.name)
    skin=model.group('skin_exterior')
    # Save the masks actually used in this solve, including explicitly requested
    # parsing, rather than copying stale observation metadata from the source.
    serial=copy.deepcopy(doc)
    for i,(view,saved) in enumerate(zip(views,serial['views'])):
        for key in ('image','exclusion_mask','silhouette_mask'):
            source=view.get(key+'_path')
            if source is None: continue
            source=Path(source);name=f'refit_view_{i}_{key}'+source.suffix
            if source.resolve()!=(out/name).resolve(): shutil.copyfile(source,out/name)
            saved[key]=name;saved['sha256' if key=='image' else key+'_sha256']=sha256(out/name)
        for key in tuple(saved):
            if key.endswith('_path'): del saved[key]
    (out/'observations.json').write_text(json.dumps(serial,indent=2))
    geometry.update(full_neutral=result['neutral'],neutral=result['neutral'][skin],full_captured=result['vertices'],
                    captured=result['vertices'][:,skin],gnm_identity=result['identity'],gnm_expressions=result['expression'],
                    gnm_rotations=result['rotations'],gnm_translations=result['translation'],gnm_alignment_offset=bind['offset'],gnm_bind_residual=bind['bind_residual'])
    _,joints=model.evaluate(result['identity'])
    geometry['gnm_joint_positions']=float(bind['scale'])*joints@bind['alignment'].T+bind['offset']
    np.savez_compressed(out/'geometry.npz',**geometry)
    manifest['id']=out.name;manifest['geometry_sha256']=sha256(out/'geometry.npz')
    from scipy.spatial.transform import Rotation
    world_rotations=[];world_translations=[]
    alignment=bind['alignment'];offset=bind['offset'];scale=float(bind['scale'])
    for r,t in zip(result['rotations'][:,0],result['translation']):
        native=Rotation.from_rotvec(r).as_matrix();world=alignment@native@alignment.T
        world_rotations.append(Rotation.from_matrix(world).as_rotvec().tolist())
        world_translations.append((scale*alignment@((np.eye(3)-native)@joints[0]+t)+offset-world@offset).tolist())
    manifest['geometry'].update(native_expression_refinement=report,coordinate_convention='fixed_camera_posed_surfaces_v2',
                                identity_coefficients=result['identity'].tolist(),expression_modes=list(range(model.expression_dim)),
                                expression_coefficients=result['expression'].tolist(),expression_labels=model.data['expression_names'].tolist(),
                                expression_steps=report['expression_safe_steps'],gnm_rotations=result['rotations'].tolist(),
                                gnm_translations=result['translation'].tolist(),pose_rotations=world_rotations,
                                pose_translations=world_translations,fitted_cameras=[v['camera'] for v in views])
    manifest['deformation']='geometry changed; rebuild compatible rig and motion bindings'
    from .materials import bake_portrait
    material=bake_portrait(geometry['captured'],geometry['triangles'],geometry['triangle_uvs'],views,
                          [Camera.from_dict(v['camera']) for v in views],out,res=manifest['config']['texture_res'],
                          roughness=manifest['material']['roughness']['value'],f0=manifest['material']['f0']['value'])
    if manifest['material'].get('authored_detail'): material['authored_detail']=manifest['material']['authored_detail']
    manifest['material']=material
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));validate_candidate(out)
    review_portrait(out,views[0],result)
    return report
