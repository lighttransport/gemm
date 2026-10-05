"""One-portrait head/bust creation with native I2V, GNM and Cycles HIP.

The Blender scene is the authoritative offline asset. The compatible rig is
also exported for the existing viewer; its procedural control mapping remains
an approximation to the separately retained native GNM motion tracks.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .. import gpu
from ..rig.expression_catalog import EMOTIONS
from .provenance import portrait_record,validate_candidate
from .observations import sha256


def create(portrait,out, *, candidate=None,expression_catalog=None,motion_root=None,
           appearance_root=None,
           backend='h3-fl2va',preset='fast5',frames=None,seed=42,iterations=80,
           motion_iterations=400,texture_res=1024,accessories='keep',detail_preset='mature',
           render_preset='final',generate_probes=False):
    from . import pipeline
    from .motion_capture import capture
    from .temporal import fit_clip
    from .expression_detail import learn
    from .offline_render import render
    portrait,out=Path(portrait).resolve(),Path(out).resolve()
    if not portrait.is_file():raise ValueError('portrait must exist')
    provenance=portrait_record(portrait)
    config=dict(portrait=provenance,backend=backend,preset=preset,frames=frames,seed=seed,
        iterations=iterations,motion_iterations=motion_iterations,texture_res=texture_res,
        accessories=accessories,detail_preset=detail_preset,render_preset=render_preset,
        candidate=str(Path(candidate).resolve()) if candidate else None,
        expression_catalog=str(Path(expression_catalog).resolve()) if expression_catalog else None,
        motion_root=str(Path(motion_root).resolve()) if motion_root else None,generate_probes=generate_probes)
    config['appearance_root']=str(Path(appearance_root).resolve()) if appearance_root else None
    record=out/'creation.json'
    if record.is_file():
        result=json.loads(record.read_text())
        if result['config']!=config:raise ValueError('cannot resume with changed creation settings')
        if result['state']=='complete':
            validation=Path(result['artifacts']['offline'])/'asset_validation.json'
            if validation.is_file() and json.loads(validation.read_text()).get('passed'):return result
            result['state']='running'
    else:
        if out.exists() and any(out.iterdir()):raise ValueError('creation output must be empty')
        out.mkdir(parents=True,exist_ok=True)
        result=dict(schema='vhuman.portrait_creation.v1',state='running',config=config,artifacts={})
    def save(stage):
        result['stage']=stage
        partial=record.with_suffix('.partial');partial.write_text(json.dumps(result,indent=2));partial.replace(record)
    save('portrait_identity')
    if candidate is None:
        head=out/'head';candidate=head/'reconstruction/photoreal01'
        if not (candidate/'manifest.json').is_file():
            if not (head/'fit.json').is_file():pipeline.direct_seed(portrait,head,'gnm_v3')
            pipeline.run(head,run_id='photoreal01',iterations=iterations,res=512,texture_res=texture_res,
                         occlusion_mode='auto',profile='full')
    else:candidate=Path(candidate).resolve();head=candidate.parents[1]
    validate_candidate(candidate)
    a=Image.open(portrait).convert('RGB');b=Image.open(candidate/'portrait.png').convert('RGB')
    if a.size!=b.size or not np.array_equal(np.asarray(a),np.asarray(b)):
        raise ValueError('candidate belongs to another portrait')
    result['artifacts']['candidate']=str(candidate)
    save('compatible_rig')
    rig=candidate/'rig'
    if not (rig/'rig.json').is_file() or not (rig/'rig.glb').is_file():
        from ..rig.build import assemble
        with gpu.device_session(2048):
            assemble(head,rig,res=texture_res,iters=60,deformer_samples=0,lods=(),preview=False,reconstruction=candidate)
    result['artifacts']['rig']=str(rig)
    save('native_expression_motion')
    if expression_catalog:
        catalog=Path(expression_catalog).resolve()
        motion=Path(motion_root).resolve() if motion_root else out/'motion'
        for name in EMOTIONS:
            path=motion/name
            if not (path/'motion.json').is_file():
                if motion_root:raise ValueError('provided motion catalog is incomplete')
                with gpu.device_session(2048):fit_clip(candidate,catalog/name,path,iterations=motion_iterations,device=f'cuda:{gpu.device_index()}')
    else:
        if motion_root:raise ValueError('motion root needs its source expression catalog')
        catalog=out/'catalog';motion=catalog
        capture(candidate,catalog,backend=backend,actions=list(EMOTIONS),preset=preset,
                frames=frames,seed=seed,iterations=motion_iterations)
    tracks={}
    for name in EMOTIONS:
        path=motion/name
        if not (path/'motion.json').is_file():path=path/'native_motion'
        tracks[name]=json.loads((path/'motion.json').read_text())
        if tracks[name]['candidate_geometry_sha256']!=sha256(candidate/'geometry.npz'):
            raise ValueError('motion does not match candidate')
        if 'skin_min_oriented_area_ratio' not in tracks[name]:raise ValueError('legacy motion lacks the native topology gate')
    result['artifacts'].update(catalog=str(catalog),motion=str(motion))
    result['motion_quality']={name:dict(accepted=t['quality_gate'],error_ipd=t['landmark_error_after_ipd'],
        minimum_skin_area_ratio=min(t['skin_min_oriented_area_ratio'])) for name,t in tracks.items()}
    save('expression_appearance')
    appearance=Path(appearance_root).resolve() if appearance_root else out/'appearance'
    if appearance_root and not (appearance/'expression_appearance.json').is_file():
        raise ValueError('provided appearance directory is incomplete')
    if not (appearance/'expression_appearance.json').is_file():learn(candidate,motion,catalog,appearance)
    appearance_metadata=json.loads((appearance/'expression_appearance.json').read_text())
    if appearance_metadata['candidate_geometry_sha256']!=sha256(candidate/'geometry.npz'):
        raise ValueError('appearance reconstruction mismatch')
    result['artifacts']['appearance']=str(appearance)
    if generate_probes:
        save('heldout_motion_probes')
        capture(candidate,out/'probes',backend=backend,preset=preset,frames=frames,
                seed=seed+100,iterations=motion_iterations)
        result['artifacts']['probes']=str(out/'probes')
    save('offline_render')
    neutral=motion/'neutral'
    if not (neutral/'motion.json').is_file():neutral=neutral/'native_motion'
    rendered=out/'offline'
    if not (rendered/'asset_validation.json').is_file():
        if rendered.exists() and any(rendered.iterdir()):
            archived=out/'offline.unvalidated'
            if archived.exists():raise ValueError('inspect previous unvalidated render before retrying')
            rendered.rename(archived)
        render(candidate,rendered,preset=render_preset,accessories=accessories,detail_preset=detail_preset,
               motion=neutral,appearance=appearance,gpu_index=gpu.device_index())
    result['artifacts']['offline']=str(rendered)
    result['state']='complete';result['visual_review_required']=True
    result['limitations']=['single portrait cannot establish metric ground truth or hidden anatomy',
        'material parameters, pore/groove height, hair and accessories contain labelled priors',
        'synthetic motion and appearance quality gates validate consistency, not true photorealism',
        'offline Cycles scene contains the native anatomy; compatible viewer controls remain approximate']
    save('complete')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--portrait',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--candidate');parser.add_argument('--expression-catalog');parser.add_argument('--motion-root');parser.add_argument('--appearance-root')
    parser.add_argument('--backend',choices=('wan','h3','h3-fl2va','hv15-rocm'),default='h3-fl2va')
    parser.add_argument('--preset',default='fast5');parser.add_argument('--frames',type=int);parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--iterations',type=int,default=80);parser.add_argument('--motion-iterations',type=int,default=400)
    parser.add_argument('--texture-res',type=int,choices=(256,512,1024,2048),default=1024)
    parser.add_argument('--accessories',choices=('keep','omit'),default='keep')
    parser.add_argument('--detail-preset',choices=('source','mature'),default='mature')
    parser.add_argument('--render-preset',choices=('draft','final'),default='final')
    parser.add_argument('--generate-probes',action='store_true')
    args=parser.parse_args()
    with gpu.execution('rocm',gpu.device_index()):print(json.dumps(create(**vars(args)),indent=2))


if __name__=='__main__':main()
