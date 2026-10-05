"""Compare complete native GNM evaluation with hash-pinned Google equations.

Install etils[enp] in the reference environment and supply gnm_common.py from
Google/GNM commit fd704e805ff11bddeac3d972af5a253cea7edc37. No TensorFlow
installation or full reference package is needed.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import types
import numpy as np
from .. import gpu
from ..rig.gnm_model import GNMModel
from .observations import sha256

REFERENCE_SHA256='b5da354d22bd54e2c3d940f5ce3ca9aac85a8a7b82a0325661ef2fd02fa4df1a'


def validate(reference,device='cpu'):
    from etils import enp
    import torch
    if sha256(reference)!=REFERENCE_SHA256:raise ValueError('Google GNM reference source hash mismatch')
    # The reference imports only enpt annotations from gnm_typing. Supply that
    # namespace without importing the package's TensorFlow front end.
    package=types.ModuleType('gnm');shape=types.ModuleType('gnm.shape');typing=types.ModuleType('gnm.shape.gnm_typing')
    typing.enpt=enp.typing;shape.gnm_typing=typing;package.shape=shape
    saved={name:sys.modules.get(name) for name in ('gnm','gnm.shape','gnm.shape.gnm_typing')}
    try:
        sys.modules.update({'gnm':package,'gnm.shape':shape,'gnm.shape.gnm_typing':typing})
        spec=importlib.util.spec_from_file_location('gnm_equation_reference',reference)
        common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)
    finally:
        for name,value in saved.items():
            if value is None:sys.modules.pop(name,None)
            else:sys.modules[name]=value
    model=GNMModel();data=model.data;rng=np.random.default_rng(19)
    beta=rng.normal(0,.3,253);expression=rng.normal(0,.15,383);rotation=rng.normal(0,.2,(4,3));translation=np.array([.02,.03,-.01])
    bind=common.vertex_positions_bind_pose(beta,expression,data['template_vertex_positions'],data['vertex_identity_basis'],data['expression_basis'])
    joints=common.joint_positions_bind_pose(beta,data['template_joint_positions'],data['joint_identity_basis'])
    correction=common.compute_pose_correctives(rotation,data['pose_correctives_regressor'],data['template_vertex_positions'],4,17821)
    reference_vertices=common.linear_blend_skinning(bind+correction,joints,rotation,translation,data['skinning_weights'],model.parents)
    ours,_=model.evaluate(beta,expression,rotation,translation)
    accelerated=GNMModel(device=device);vertices,_=accelerated.evaluate(beta,expression,rotation,translation)
    numpy_rms=float(np.sqrt(np.mean((ours-reference_vertices)**2)))
    torch_rms=float(np.sqrt(np.mean((vertices.detach().cpu().numpy()-reference_vertices)**2)))
    result=dict(reference_sha256=REFERENCE_SHA256,vertices=17821,identity_dim=253,expression_dim=383,
        numpy_rms_m=numpy_rms,torch_rms_m=torch_rms,device=device,
        device_name=torch.cuda.get_device_name(device) if str(device).startswith('cuda') else 'CPU',
        passed=numpy_rms<1e-10 and torch_rms<1e-6)
    if not result['passed']:raise ValueError(f'GNM equation parity failed: {result}')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',required=True);parser.add_argument('--device',default='cpu');parser.add_argument('--out')
    args=parser.parse_args()
    if args.device.startswith('cuda'):
        with gpu.execution('rocm',int(args.device.split(':')[-1])),gpu.device_session(1024):result=validate(args.reference,args.device)
    else:result=validate(args.reference,args.device)
    if args.out:Path(args.out).write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
