"""Portable GNM bind metadata, full CPU motion tracks and rig retargeting."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import shutil
import numpy as np
from PIL import Image, ImageDraw

from ..rig.gnm_model import GNMModel
from ..rig.face_models import GNM_SHA256
from .gnm_fit import solve, replay, cpu_threads, check_cancel
from .observations import sha256
from .reference import Camera


def candidate_bind(candidate):
    from .provenance import validate_candidate
    candidate=Path(candidate);manifest=validate_candidate(candidate)
    model=GNMModel()
    with np.load(candidate/'geometry.npz',allow_pickle=False) as z:
        scale=float(z['scale']);alignment=z['rotation']
        if 'gnm_identity' in z:
            identity=z['gnm_identity'];rest=z['full_neutral'];expression=z['gnm_expressions'][0]
            native,joints=model.evaluate(identity)
            if 'gnm_alignment_offset' in z: offset=z['gnm_alignment_offset']
            elif 'gnm_joint_positions' in z: offset=np.median(z['gnm_joint_positions']-scale*joints@alignment.T,axis=0)
            else: offset=np.median(rest-scale*native@alignment.T,axis=0)
            rotations=z['gnm_rotations'][0] if 'gnm_rotations' in z else np.zeros((4,3))
            translation=z['gnm_translations'][0] if 'gnm_translations' in z else np.zeros(3)
        else:
            # Older candidates retained only exterior PCA geometry. Reconstruct
            # the same canonical bind and preserve their already-fitted cameras.
            identity=np.zeros(model.identity_dim)
            coefficients=manifest['geometry'].get('identity_coefficients',[])
            identity[:len(coefficients)]=coefficients
            native,joints=model.evaluate(identity);skin=model.group('skin_exterior')
            offset=np.median(z['neutral']-scale*native[skin]@alignment.T,axis=0)
            rest=scale*native@alignment.T+offset;rest[skin]=z['neutral']
            expression=np.zeros(model.expression_dim)
            expression[manifest['geometry'].get('expression_modes',[])]=manifest['geometry'].get('expression_coefficients',[[]])[0]
            rotations=np.zeros((4,3));translation=np.zeros(3)
    residual=(rest-offset)@alignment/scale-native
    return dict(identity=identity,scale=np.array(scale),alignment=alignment,offset=offset,bind_residual=residual,
                reference_expression=expression,reference_rotations=rotations,reference_translation=translation,
                source_ids=np.flatnonzero(model.group('skin_exterior')),
                metadata=dict(schema='vhuman.gnm_bind.v1',model_sha256=GNM_SHA256,
                              candidate_geometry_sha256=manifest['geometry_sha256'],
                              camera=manifest['geometry']['fitted_cameras'][0],
                              size=json.loads((candidate/'observations.json').read_text())['views'][0]['size']))


def export_bind(rig,candidate):
    rig=Path(rig);bind=candidate_bind(candidate);meta=bind.pop('metadata')
    meta['rig_package_sha256']=sha256(rig/'rig_deformer.safetensors')
    np.savez_compressed(rig/'gnm_bind.npz',**bind)
    meta['bind_sha256']=sha256(rig/'gnm_bind.npz')
    (rig/'gnm_bind.json').write_text(json.dumps(meta,indent=2))
    return meta


def rig_bind(rig):
    rig=Path(rig)
    if (rig/'gnm_bind.json').is_file():
        meta=json.loads((rig/'gnm_bind.json').read_text())
        if (meta.get('model_sha256')!=GNM_SHA256 or meta.get('bind_sha256')!=sha256(rig/'gnm_bind.npz') or
                meta.get('rig_package_sha256')!=sha256(rig/'rig_deformer.safetensors')):
            raise ValueError('GNM bind or rig package changed; rebuild rig')
        with np.load(rig/'gnm_bind.npz',allow_pickle=False) as z: bind={k:z[k] for k in z.files}
        bind['metadata']=meta;return bind
    if (rig.parent/'geometry.npz').is_file():
        return candidate_bind(rig.parent)
    return legacy_rig_bind(rig)


def fit_views(bind,views,timestamps, *, out, iterations=400,modes=64,threads=16,cancel=None):
    count=len(views)
    result=solve(views,identity=bind['identity'],expression=np.repeat(bind['reference_expression'][None],count,axis=0),
                 rotations=np.repeat(bind['reference_rotations'][None],count,axis=0),
                 translation=np.repeat(bind['reference_translation'][None],count,axis=0),
                 scale=float(bind['scale']),alignment=bind['alignment'],offset=bind['offset'],
                 bind_residual=bind['bind_residual'],timestamps=timestamps,iterations=iterations,modes=modes,
                 threads=threads,cancel=cancel)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    model=GNMModel();report=result['report']
    norm=result['normalization_ipd'][:,None]
    errors=np.linalg.norm(result['projected_landmarks']-result['observed_landmarks'],axis=-1)/norm
    old_errors=np.linalg.norm(result['baseline_landmarks']-result['observed_landmarks'],axis=-1)/norm
    weights=result['weights'];total=max(weights.sum(),1)
    track=dict(report,schema='vhuman.native_gnm_motion.v2',synthetic=any(v.get('synthetic',False) for v in views),
               identity_frozen=True,residual_transport='native_lbs_bind_space',model_sha256=GNM_SHA256,
               candidate_geometry_sha256=bind['metadata']['candidate_geometry_sha256'],
               rig_package_sha256=bind['metadata'].get('rig_package_sha256'),
               fps=float(1/np.median(np.diff(timestamps))) if count>1 else 1.,size=views[0]['size'],
               camera=views[0]['camera'],cameras=[v['camera'] for v in views],
               skin_min_oriented_area_ratio=report['per_frame_min_oriented_area_ratio'],
               peak_torch_allocated_mib=None,
               landmark_error_before_ipd=float((old_errors*weights).sum()/total),
               landmark_error_after_ipd=float((errors*weights).sum()/total),
               per_frame_error_ipd=[float(e) if valid else None for e,valid in zip((errors*weights).sum(1)/np.maximum(weights.sum(1),1),weights.sum(1)>0)],
               valid_frames=int((weights.sum(1)>0).sum()),per_frame_valid=(weights.sum(1)>0).tolist())
    arrays={k:result[k] for k in ('identity','expression','rotations','translation','vertices','joints',
                                'projected_landmarks','observed_landmarks','weights','heldout')}
    arrays.update(triangles=model.data['triangles'],timestamps=np.asarray(timestamps),
                  landmark_names=np.asarray(result['names']),alignment=bind['alignment'],offset=bind['offset'],
                  scale=bind['scale'],bind_residual=bind['bind_residual'])
    np.savez_compressed(out/'motion.npz',**arrays)
    track['motion_sha256']=sha256(out/'motion.npz')
    (out/'motion.json').write_text(json.dumps(track,indent=2))
    result['track']=track
    return result


def retarget(rig,bind,result,direct, *, iterations=120,threads=16,cancel=None):
    """Fit nonlinear semantic rig motion, then measure actual CPU runtime error."""
    import torch
    from ..rig import safetensors as st, native
    from ..rig.torchrig import TorchRig
    rig=Path(rig);definition=json.loads((rig/'rig.json').read_text())
    package,metadata=st.load(rig/'rig_deformer.safetensors')
    count=len(result['expression']);source=np.asarray(bind['source_ids'],int)
    # GNM exterior vertices retain source order in the compatible native package.
    if len(package['rest'])<len(source): raise ValueError('GNM source-to-rig binding is incomplete')
    ids=np.arange(0,len(source),max(1,len(source)//800))
    morph_names=json.loads(metadata['morphs']);base_names={b['name'] for b in definition['blendshapes']}
    shapes={n:package['morph'][i,ids] for i,n in enumerate(morph_names) if n in base_names}
    model=GNMModel(device='cpu')
    rotations=result['rotations'].copy();rotations[:,0]=0
    with cpu_threads(threads):
        desired,_=replay(model,result['identity'],result['expression'],rotations,np.zeros((count,3)),
                         scale=float(bind['scale']),alignment=bind['alignment'],offset=bind['offset'],
                         bind_residual=bind['bind_residual'],cancel=cancel)
        # Match motion deltas; rest meshes can differ through transferred detail.
        reference,_=replay(model,result['identity'],np.zeros((1,model.expression_dim)),np.zeros((1,4,3)),np.zeros((1,3)),
                           scale=float(bind['scale']),alignment=bind['alignment'],offset=bind['offset'],
                           bind_residual=bind['bind_residual'])
        target=package['rest'][ids][None]+desired[:,source[ids]]-reference[:,source[ids]]
        tr=TorchRig(definition,package['rest'][ids],shapes,package['skin.joints'][ids],package['skin.weights'][ids],device='cpu')
        direct=np.asarray(direct,np.float32)
        if direct.shape!=(count,len(tr.controls)): raise ValueError('invalid retarget prior')
        ml={k:torch.as_tensor(v.copy(),dtype=torch.float32) for k,v in package.items() if k.startswith('ml.')}
        ml_morph=torch.as_tensor(package['morph'][len(tr.shape_names):,ids].copy(),dtype=torch.float32) if ml else None
        fitted=np.empty_like(direct)
        for lo in range(0,count,64):
            hi=min(count,lo+64);n=hi-lo
            controls=torch.nn.Parameter(torch.as_tensor(direct[lo:hi]).clone())
            prior=controls.detach().clone();wanted=torch.as_tensor(target[lo:hi],dtype=torch.float32)
            optimizer=torch.optim.Adam([controls],lr=.025)
            for step in range(iterations):
                check_cancel(cancel);optimizer.zero_grad(set_to_none=True)
                pre=None
                if ml:
                    inp=tr.input_vector(controls)[:,:len(ml['ml.input.mean'])]
                    hid=torch.relu(((inp-ml['ml.input.mean'])*ml['ml.input.scale'])@ml['ml.fc1.weight'].T+ml['ml.fc1.bias'])
                    coef=(hid@ml['ml.fc2.weight'].T+ml['ml.fc2.bias'])+ml['ml.output.mean']/ml['ml.output.scale']
                    weights=torch.cat((torch.ones((n,1)),coef),1)
                    pre=torch.einsum('fm,mvc->fvc',weights,ml_morph)
                actual=tr(controls,pre=pre)['pos']
                loss=torch.nn.functional.smooth_l1_loss(actual,wanted,beta=.001)+1e-5*(controls-prior).square().mean()
                loss.backward();optimizer.step()
                with torch.no_grad(): controls.copy_(torch.maximum(torch.minimum(controls,tr.hi),tr.lo))
            fitted[lo:hi]=controls.detach().numpy()
        cache=native.ROOT/'tmp/vhuman-cpu-build';cache.mkdir(parents=True,exist_ok=True)
        temp=tempfile.TemporaryDirectory(dir=cache)
        lib=native.build_library(Path(temp.name))
        evaluator=native.Native(lib,rig/'rig_deformer.safetensors')
        try:
            errors=[];baseline_errors=[];fallback_frames=0
            for start in range(0,count,4):
                check_cancel(cancel)
                actual=evaluator.eval_batch(fitted[start:start+4])[:,ids]
                prior=evaluator.eval_batch(direct[start:start+4])[:,ids]
                error=np.linalg.norm(actual-target[start:start+4],axis=-1)
                old=np.linalg.norm(prior-target[start:start+4],axis=-1)
                fallback=(error**2).mean(1)>(old**2).mean(1)
                fitted[start:start+4][fallback]=direct[start:start+4][fallback]
                error[fallback]=old[fallback];fallback_frames+=int(fallback.sum())
                errors.extend(error.ravel());baseline_errors.extend(old.ravel())
        finally: evaluator.close();temp.cleanup()
    return fitted,dict(method='nonlinear CPU rig plus native corrective/contact validation',
                       rig_package_sha256=sha256(rig/'rig_deformer.safetensors'),rig_definition_sha256=sha256(rig/'rig.json'),
                       baseline_rms_mm=float(np.sqrt(np.mean(np.square(baseline_errors))))*1000,fallback_frames=fallback_frames,
                       rms_mm=float(np.sqrt(np.mean(np.square(errors))))*1000,p95_mm=float(np.quantile(errors,.95))*1000,
                       approximate=True,gnm_track_authoritative=True,global_pose='retained in native GNM track')


def overlay(out,frames,result, *, fps,cancel=None):
    """CPU-only landmark review movie. No OpenGL/Vulkan renderer or video decode."""
    out=Path(out);path=out/'fit_overlay.mp4'
    if not frames: return
    first=Image.open(frames[0]).convert('RGB');w,h=first.size
    process=subprocess.Popen(['ffmpeg','-nostdin','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24',
                              '-s',f'{w}x{h}','-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-threads','2',
                              '-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2','-pix_fmt','yuv420p',str(path)],stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        for i,frame in enumerate(frames):
            check_cancel(cancel)
            image=Image.open(frame).convert('RGB');draw=ImageDraw.Draw(image)
            for observed,fitted,weight in zip(result['observed_landmarks'][i],result['projected_landmarks'][i],result['weights'][i]):
                if weight<=0: continue
                x,y=observed;draw.ellipse((x-1,y-1,x+1,y+1),fill=(255,70,70))
                x,y=fitted;draw.ellipse((x-1,y-1,x+1,y+1),fill=(50,255,90))
            process.stdin.write(image.tobytes())
        process.stdin.close();error=process.stderr.read();rc=process.wait()
        if rc: raise RuntimeError('CPU overlay encoding failed: '+error.decode(errors='replace')[-500:])
    finally:
        if process.poll() is None: process.kill();process.wait()
        if not process.stdin.closed: process.stdin.close()
        process.stderr.close()
    i=len(frames)//2
    mesh_still(out,result['vertices'][i],Camera.from_dict(result['track']['cameras'][i]),(w,h))


def mesh_still(out,vertices,camera,size):
    """Z-buffered full-anatomy CPU preview with face normals, independent of GL."""
    from .reference import rasterize
    w,h=size;model=GNMModel();tri=model.data['triangles'];ratio=min(1.,512/max(w,h))
    tid,_,_=rasterize(vertices,tri,camera.scaled(ratio),(round(w*ratio),round(h*ratio)))
    normal=np.cross(vertices[tri[:,1]]-vertices[tri[:,0]],vertices[tri[:,2]]-vertices[tri[:,0]])
    normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-12)
    light=np.maximum((normal@camera.rotation.T)@np.array([.3,.4,.85]),0)
    colors=np.uint8((.25+.75*light[:,None])*np.array([205,180,160]))
    image=np.full((*tid.shape,3),24,np.uint8);valid=tid>=0;image[valid]=colors[tid[valid]]
    Image.fromarray(image).save(Path(out)/'mesh_preview.png')


def review_portrait(out,view,result):
    image=Image.open(view['image_path']).convert('RGB');w,h=image.size
    comparison=Image.new('RGB',(2*w,h))
    for index,key in enumerate(('baseline_landmarks','projected_landmarks')):
        panel=image.copy();draw=ImageDraw.Draw(panel)
        for target,prediction,weight in zip(result['observed_landmarks'][0],result[key][0],result['weights'][0]):
            if weight<=0: continue
            x,y=target;draw.ellipse((x-1,y-1,x+1,y+1),fill=(255,70,70))
            x,y=prediction;draw.ellipse((x-1,y-1,x+1,y+1),fill=(60,255,90))
        draw.rectangle((0,0,w,25),fill=(0,0,0))
        draw.text((6,6),'Before' if index==0 else 'After (accepted state)',fill=(255,255,255))
        comparison.paste(panel,(index*w,0))
    comparison.save(Path(out)/'fit_comparison.png')
    mesh_still(out,result['vertices'][0],Camera.from_dict(view['camera']),(w,h))


def views_from_landmarks(landmarks,valid,camera,size, *, labels=None,synthetic=False):
    """Shared observation adapter for real videos and provenance-checked catalogs."""
    landmarks=np.asarray(landmarks,float);valid=np.asarray(valid,bool)
    if landmarks.shape!=(len(valid),478,3) or not np.isfinite(landmarks).all():
        raise ValueError('invalid dense video landmarks')
    w,h=size;model=GNMModel();views=[]
    pupils={}
    for index,side in ((468,'right'),(473,'left')):
        ids=np.flatnonzero(model.group('pupils')&model.group(side+'_eye'))
        ids=ids[np.linspace(0,len(ids)-1,min(len(ids),32)).astype(int)]
        pupils[index]=ids.tolist()
    if labels is not None and np.asarray(labels).shape!=(len(valid),h,w): raise ValueError('video parsing dimensions mismatch')
    for i,points in enumerate(landmarks):
        anchors={}
        if valid[i]:
            for n in range(468):
                xy=points[n,:2]*[w,h];weight=.8
                if not (0<=xy[0]<w and 0<=xy[1]<h): weight=0.
                elif labels is not None:
                    label=labels[i,int(xy[1]),int(xy[0])]
                    if label in (6,9,15,16,17,18):
                        weight=.12 if label==6 and n in (33,133,159,145,263,362,386,374) else 0.
                anchors[f'mp_{n:03d}']=dict(xy=xy.tolist(),weight=weight)
            for n,ids in pupils.items():
                xy=points[n,:2]*[w,h];weight=.8 if 0<=xy[0]<w and 0<=xy[1]<h else 0.
                if weight and labels is not None and labels[i,int(xy[1]),int(xy[0])] in (6,9,15,16,17,18): weight=.12
                anchors[f'mp_{n}']=dict(xy=xy.tolist(),weight=weight,vertices=ids,vertex_space='gnm_full')
        views.append(dict(size=list(size),camera=camera.as_dict(),anchors=anchors,synthetic=synthetic))
    return views


def legacy_rig_bind(rig):
    """Read existing GNM rigs without silently substituting a neutral identity."""
    from ..rig import safetensors as st
    from ..rig.common import load_subject
    from .reference import pixal_camera
    rig=Path(rig);definition=json.loads((rig/'rig.json').read_text())
    report=json.loads((rig/'rig_report.json').read_text())
    fit=report.get('face_model_fit') or {};alignment=fit.get('expression_basis_alignment')
    if definition.get('face_model')!='gnm_v3' or not alignment or not fit.get('identity_coefficients'):
        raise ValueError('GNM identity metadata missing; rebuild the rig from a reconstruction candidate')
    model=GNMModel();identity=np.zeros(model.identity_dim)
    coefficients=fit['identity_coefficients'];identity[:len(coefficients)]=coefficients
    native,_=model.evaluate(identity);skin=np.flatnonzero(model.group('skin_exterior'))
    package,_=st.load(rig/'rig_deformer.safetensors')
    if len(package['rest'])<len(skin): raise ValueError('GNM rig exterior binding is incomplete')
    scale=alignment['scale'];rotation=np.asarray(alignment['rotation'])
    offset=np.median(package['rest'][:len(skin)]-scale*native[skin]@rotation.T,axis=0)
    rest=scale*native@rotation.T+offset;rest[skin]=package['rest'][:len(skin)]
    subject=load_subject(rig.parent);camera=pixal_camera(subject)
    with Image.open(subject.portrait) as image: size=image.size
    return dict(identity=identity,scale=np.array(scale),alignment=rotation,offset=offset,
                bind_residual=(rest-offset)@rotation/scale-native,reference_expression=np.zeros(model.expression_dim),
                reference_rotations=np.zeros((4,3)),reference_translation=np.zeros(3),source_ids=skin,
                metadata=dict(schema='vhuman.gnm_bind.v1',model_sha256=GNM_SHA256,candidate_geometry_sha256=None,
                              rig_package_sha256=sha256(rig/'rig_deformer.safetensors'),camera=camera.as_dict(),size=list(size),
                              limitations=['Legacy transferred neutral supplies bind residual; hidden joint placement remains a prior.']))
