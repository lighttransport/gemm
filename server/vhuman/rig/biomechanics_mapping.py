"""CPU geometric calibration of semantic controls, including joint skinning.

The older morph-only PCA projection is not a complete rig-to-GNM map: jaw
opening, correctives and contacts also change evaluated vertices.
"""
import json
from pathlib import Path

import numpy as np

from . import native
from .gnm_expression import project


def capture_transforms(model, bind, rotations, translation=None):
    """Exact GNM joint affine transforms in the aligned rig's metre space."""
    from .gnm_model import rotation_matrices
    rotations = np.asarray(rotations, np.float64)
    translation = np.zeros(3) if translation is None else np.asarray(translation, np.float64)
    if (rotations.shape != (len(model.parents), 3) or translation.shape != (3,)
            or not np.isfinite(rotations).all() or not np.isfinite(translation).all()):
        raise ValueError('invalid captured rigid motion')
    joints = model.data['template_joint_positions'] + np.einsum(
        'i,ijc->jc', bind['identity'], model.data['joint_identity_basis'])
    local = rotation_matrices(rotations)
    rotation, position = [], []
    for i, parent in enumerate(model.parents):
        if parent < 0:
            rotation.append(local[i]);position.append(joints[i]+translation)
        else:
            rotation.append(rotation[parent]@local[i])
            position.append(rotation[parent]@(joints[i]-joints[parent])+position[parent])
    alignment, offset, scale = np.asarray(bind['alignment'], np.float64), np.asarray(bind['offset'], np.float64), float(bind['scale'])
    inverse_alignment = np.linalg.inv(alignment)
    result = {}
    for name, r, p, j in zip(model.data['joint_names'], rotation, position, joints):
        transform = np.eye(4)
        # Serialized float32 alignments are only approximately orthogonal.
        transform[:3, :3] = alignment@r@inverse_alignment
        transform[:3, 3] = scale*alignment@(p-r@j)+offset-transform[:3, :3]@offset
        result[str(name)] = transform
    return result


def load_native_rig(rig_dir):
    lib_dir=Path(__file__).resolve().parents[3]/'tmp/vhuman-rig/native'
    library=lib_dir/'libvhuman_deformer.so'
    if not library.is_file():
        library=native.build_library(lib_dir)
    return native.Native(library,Path(rig_dir)/'rig_deformer.safetensors')


def load_skinning(rig_dir, count):
    """Copy the exterior's bind correspondence, without retaining ML tensors."""
    from .safetensors import load
    tensors,_ = load(Path(rig_dir)/'rig_deformer.safetensors')
    joints = tensors['skin.joints'][:count].astype(np.int32)
    weights = tensors['skin.weights'][:count].astype(np.float64)
    if (joints.shape!=(count,4) or weights.shape!=joints.shape or (joints<0).any()
            or not np.isfinite(weights).all() or (weights<0).any()
            or not np.allclose(weights.sum(1),1,atol=1e-6,rtol=0)):
        raise ValueError('invalid fitted-bind skinning correspondence')
    return joints,weights


def skin_vectors(vectors, skinning, transforms):
    """Transport pre-skin fit corrections as vectors, applying no translation."""
    joints,weights = skinning
    transforms = np.asarray(transforms,np.float64)
    if (vectors.shape!=(len(joints),3) or transforms.ndim!=3 or transforms.shape[1:]!=(4,4)
            or (joints>=len(transforms)).any()):
        raise ValueError('invalid fitted-bind transform correspondence')
    return np.einsum('vk,vkij,vj->vi',weights,transforms[joints,:3,:3],vectors)


def calibrate(rig_dir,model,bind,*,threads=4):
    from ..reconstruction.gnm_fit import cpu_threads
    definition=json.loads((Path(rig_dir)/'rig.json').read_text())
    original=definition['face_model_source']['expression_mapping']
    names=list(original['controls'])
    order=[c['name'] for c in definition['controls']]
    indices=np.flatnonzero(model.group('skin_exterior'))
    rig=load_native_rig(rig_dir)
    try:
        if rig.V<len(indices) or rig.C!=len(order):
            raise ValueError('native rig does not preserve GNM exterior correspondence')
        x=np.zeros((len(names)*3+1,len(order)),np.float32)
        amplitudes=(.25,.75,.5)
        for i,name in enumerate(names):
            for j,value in enumerate(amplitudes):
                x[1+3*i+j,order.index(name)]=value
        result=rig.eval_batch(x)[:,:len(indices)].astype(np.float64)
    finally:
        rig.close()
    neutral=result[0]
    displacement=result[1:].reshape(len(names),3,len(indices),3)-neutral
    targets=.5*(displacement[:,0]/.25+displacement[:,1]/.75)
    basis=float(bind['scale'])*np.einsum('evc,dc->evd',model.data['expression_basis'][:,indices],bind['alignment'])
    with cpu_threads(threads):
        candidate=project(basis,targets,names,model.data['expression_names'].tolist())
    old=np.asarray(original['coefficients']);new=np.asarray(candidate['coefficients'])
    # The held-out amplitude was never used in the projection. Evaluate in
    # actual geometry space, not a PCA coefficient norm.
    sample=np.arange(0,len(indices),max(1,len(indices)//1200))
    target=displacement[:,2,sample]
    before=np.einsum('ec,evk->cvk',old*.5,basis[:,sample])-target
    after=np.einsum('ec,evk->cvk',new*.5,basis[:,sample])-target
    accepted=[];metrics=[]
    regions={name:model.group(name)[indices[sample]] for name in ('upper_lip','lower_lip','eye_sockets')}
    for i,name in enumerate(names):
        baseline=float(np.sqrt(np.mean(before[i]**2)))
        proposed=float(np.sqrt(np.mean(after[i]**2)))
        safe=True
        for mask in regions.values():
            if mask.any():
                b=float(np.sqrt(np.mean(before[i,mask]**2)))
                a=float(np.sqrt(np.mean(after[i,mask]**2)))
                safe &= a<=max(.000001,b*1.05)
        accept=proposed<=baseline*.98 and safe
        if not accept:
            new[:,i]=old[:,i]
        else:
            accepted.append(name)
        metrics.append(dict(control=name,before_mm=baseline*1000,after_mm=proposed*1000,accepted=bool(accept)))
    mapping=dict(candidate,coefficients=new.tolist(),includes_joint_skinning=False,
                 evaluated_pose_controls=accepted,
                 method='native CPU evaluated-pose candidates; rejected controls retain morph-only seed')
    report=dict(training_amplitudes=[.25,.75],heldout_amplitude=.5,accepted_controls=accepted,per_control=metrics,
                target='evaluated native rig, including ML and exact contacts; not measured 3D motion')
    return mapping,report
