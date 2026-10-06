"""Complete GNM v3 evaluation with NumPy or differentiable PyTorch ROCm.

Implements the public GNM equations: identity-dependent bind joints, expression
offsets, rotation correctives and hierarchical LBS. No procedural jaw transform
is layered onto the lower-face PCA expression. Model data remains hash pinned.
Reference: https://github.com/google/GNM (Apache-2.0).
"""
import numpy as np
from ..face_assets import asset_path, sha256
from .face_models import GNM_SHA256


class GNMModel:
    def __init__(self, path=None, *, device=None):
        path = path or asset_path('gnm')
        if sha256(path) != GNM_SHA256:
            raise ValueError('GNM model hash mismatch')
        with np.load(path,allow_pickle=False) as z:
            self.data = {key: z[key].copy() for key in z.files}
        self.parents = self.data['joint_parent_indices'].tolist()
        self.identity_dim = len(self.data['identity_names'])
        self.expression_dim = len(self.data['expression_names'])
        self.device = device
        self.tensors = {}
        if device is not None:
            import torch
            if str(device).startswith('cuda') and torch.version.hip is None:
                raise ValueError('GPU GNM fitting requires PyTorch ROCm')
            for key, value in self.data.items():
                if value.dtype.kind == 'f':
                    self.tensors[key] = torch.as_tensor(value,dtype=torch.float32,device=device)

    def group(self, name):
        return self.data['vertex_groups'][self.data['vertex_group_names'].tolist().index(name)] > .5

    def frame_evaluator(self, identity, vertex_ids=None, *, bind_residual=None):
        """Cache fixed identity and sampled anatomy for batched temporal fitting.

        Expressions and the four native joint rotations remain differentiable.
        Sampling affects vertex output only; all joints and pose correctives
        retain the complete model equations.
        """
        if self.device is None:
            raise ValueError('batched fitting requires a PyTorch device')
        import torch
        ids = np.arange(len(self.data['template_vertex_positions'])) if vertex_ids is None else np.asarray(vertex_ids)
        if ids.ndim != 1 or ids.dtype.kind not in 'iu' or (ids < 0).any() or (ids >= len(self.data['template_vertex_positions'])).any():
            raise ValueError('invalid sampled GNM vertex ids')
        beta = torch.as_tensor(identity, dtype=torch.float32, device=self.device)
        if beta.shape != (self.identity_dim,) or not bool(torch.isfinite(beta).all()):
            raise ValueError('invalid shared GNM identity')
        d = self.tensors
        bind = (d['template_vertex_positions'][ids] + torch.einsum('i,ivc->vc',beta,d['vertex_identity_basis'][:,ids])).detach()
        if bind_residual is not None:
            residual = torch.as_tensor(bind_residual,dtype=torch.float32,device=self.device)
            if residual.shape != d['template_vertex_positions'].shape or not bool(torch.isfinite(residual).all()):
                raise ValueError('invalid GNM bind-space residual')
            bind = bind + residual[ids].detach()
        joints = (d['template_joint_positions'] + torch.einsum('i,ijc->jc',beta,d['joint_identity_basis'])).detach()
        basis = d['expression_basis'][:,ids].reshape(self.expression_dim,-1)
        correctives = d['pose_correctives_regressor'].reshape(36,-1,3)[:,ids].reshape(36,-1)
        weights = d['skinning_weights'][:,ids]
        eye = torch.eye(3,device=self.device)

        def evaluate(expression, rotations, translation):
            frames = len(expression)
            if expression.shape != (frames,self.expression_dim) or rotations.shape != (frames,4,3) or translation.shape != (frames,3):
                raise ValueError('invalid batched GNM dimensions')
            angle = torch.sqrt(torch.clamp((rotations*rotations).sum(-1),min=1e-8))
            axis = rotations/angle[...,None]
            x,y,z = axis.unbind(-1);zero = torch.zeros_like(x)
            skew = torch.stack((zero,-z,y,z,zero,-x,-y,x,zero),-1).reshape(frames,4,3,3)
            local = eye+torch.sin(angle)[...,None,None]*skew+(1-torch.cos(angle))[...,None,None]*(skew@skew)
            vertices = bind[None]+(expression@basis).reshape(frames,len(ids),3)
            vertices = vertices+((local-eye).reshape(frames,36)@correctives).reshape(vertices.shape)
            world_r,world_t = [local[:,0]], [joints[0]+translation]
            for j in range(1,4):
                parent = self.parents[j]
                world_r.append(world_r[parent]@local[:,j])
                world_t.append((world_r[parent]@(joints[j]-joints[parent])[:,None])[...,0]+world_t[parent])
            r,t = torch.stack(world_r,1),torch.stack(world_t,1)
            offset = t-(r@joints[None,:,:,None])[...,0]
            weighted_r = torch.einsum('jv,fjab->fvab',weights,r)
            weighted_t = torch.einsum('jv,fja->fva',weights,offset)
            return (weighted_r@vertices[...,None])[...,0]+weighted_t,t
        return evaluate

    def evaluate(self, identity=None, expression=None, rotations=None, translation=None, *, bind_residual=None):
        """Single frame, metres, original GNM axes. Returns vertices and joints."""
        torch_mode = self.device is not None
        if torch_mode:
            import torch
            arr = lambda x: torch.as_tensor(x,dtype=torch.float32,device=self.device)
            zeros = lambda shape: torch.zeros(shape,dtype=torch.float32,device=self.device)
            eye = torch.eye(3,dtype=torch.float32,device=self.device)
            einsum, stack = torch.einsum, torch.stack
            sin, cos, sqrt = torch.sin, torch.cos, torch.sqrt
            maximum = lambda x, floor: torch.clamp(x,min=floor)
            data = self.tensors
        else:
            arr = lambda x: np.asarray(x,dtype=np.float64)
            zeros = lambda shape: np.zeros(shape,np.float64)
            eye = np.eye(3)
            einsum, stack = np.einsum, np.stack
            sin, cos, sqrt = np.sin, np.cos, np.sqrt
            maximum = np.maximum
            data = self.data
        identity = zeros((self.identity_dim,)) if identity is None else arr(identity)
        expression = zeros((self.expression_dim,)) if expression is None else arr(expression)
        rotations = zeros((len(self.parents),3)) if rotations is None else arr(rotations)
        translation = zeros((3,)) if translation is None else arr(translation)
        if identity.shape != (self.identity_dim,) or expression.shape != (self.expression_dim,) or rotations.shape != (len(self.parents),3) or translation.shape != (3,):
            raise ValueError('invalid GNM coefficient dimensions')
        values = (identity,expression,rotations,translation)
        finite = all(bool(torch.isfinite(x).all()) for x in values) if torch_mode else all(np.isfinite(x).all() for x in values)
        if not finite:
            raise ValueError('nonfinite GNM coefficients')
        angle = sqrt(maximum((rotations*rotations).sum(-1),1e-8))
        axis = rotations/angle[:,None]
        x,y,z = axis[:,0],axis[:,1],axis[:,2]
        zero = zeros(x.shape)
        skew = stack((stack((zero,-z,y),-1),stack((z,zero,-x),-1),stack((-y,x,zero),-1)),-2)
        local_r = eye+sin(angle)[:,None,None]*skew+(1-cos(angle))[:,None,None]*(skew@skew)
        vertices = data['template_vertex_positions']+einsum('i,ivc->vc',identity,data['vertex_identity_basis'])+einsum('e,evc->vc',expression,data['expression_basis'])
        if bind_residual is not None:
            residual = arr(bind_residual)
            if residual.shape != vertices.shape or not bool((torch.isfinite(residual) if torch_mode else np.isfinite(residual)).all()):
                raise ValueError('invalid GNM bind-space residual')
            vertices = vertices + residual
        joints = data['template_joint_positions']+einsum('i,ijc->jc',identity,data['joint_identity_basis'])
        vertices = vertices+((local_r-eye).reshape(-1)@data['pose_correctives_regressor']).reshape(vertices.shape)
        world_r, world_t = [local_r[0]], [joints[0]+translation]
        for j in range(1,len(self.parents)):
            parent = self.parents[j]
            world_r.append(world_r[parent]@local_r[j])
            world_t.append(world_r[parent]@(joints[j]-joints[parent])+world_t[parent])
        r,t = stack(world_r),stack(world_t)
        offset = t-(r@joints[:,:,None])[:,:,0]
        weighted_r = einsum('jv,jab->vab',data['skinning_weights'],r)
        weighted_t = einsum('jv,ja->va',data['skinning_weights'],offset)
        return (weighted_r@vertices[:,:,None])[:,:,0]+weighted_t,t
