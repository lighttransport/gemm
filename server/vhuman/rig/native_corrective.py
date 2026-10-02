"""Native corrective-training algebra and batched linear rig evaluation."""
import numpy as np
from ..native_training import library, check, pointer, IP, matmul, AdamW
from .rigdef import ATTRS


def f32(x): return np.ascontiguousarray(x, np.float32)
def i32(x): return np.ascontiguousarray(x, np.int32)
def ip(x): return x.ctypes.data_as(IP)


class MLP2:
    """Two linear layers/ReLU; GEMM and the full reverse pass execute in C++."""
    def __init__(self, inputs, hidden, outputs, seed=0, weight_decay=.001):
        if any(type(v) is not int or not 1 <= v <= bound for v,bound in
               ((inputs,16384),(hidden,4096),(outputs,4096))):
            raise ValueError('invalid corrective MLP dimensions')
        self.dimensions = inputs, hidden, outputs
        self.shapes = {'fc1.weight':(hidden,inputs), 'fc1.bias':(hidden,),
                       'fc2.weight':(outputs,hidden), 'fc2.bias':(outputs,)}
        rng = np.random.default_rng(seed)
        self.parameters = np.concatenate([rng.uniform(-1/np.sqrt(inputs if name.startswith('fc1') else hidden),
            1/np.sqrt(inputs if name.startswith('fc1') else hidden),shape).ravel() for name,shape in self.shapes.items()]).astype(np.float32)
        self.optimizer = AdamW(self.parameters,lr=.003,weight_decay=weight_decay)

    def state_dict(self):
        offset,result = 0,{}
        for name,shape in self.shapes.items():
            n = int(np.prod(shape));result[name]=self.parameters[offset:offset+n].reshape(shape).copy();offset+=n
        return result

    def load_state_dict(self, weights):
        if set(weights) != set(self.shapes): raise ValueError('corrective MLP tensor names differ')
        values=[]
        for name,shape in self.shapes.items():
            value=f32(weights[name])
            if value.shape != shape or not np.isfinite(value).all(): raise ValueError('invalid MLP weights')
            values.append(value.ravel())
        self.parameters[:]=np.concatenate(values)

    def compute(self, x, upstream=None):
        x=f32(x);inputs,hidden,outputs=self.dimensions
        if x.ndim != 2 or x.shape[1] != inputs or not 1 <= len(x) <= 65536:
            raise ValueError('invalid corrective MLP batch')
        out=np.empty((len(x),outputs),np.float32)
        gradient=None
        if upstream is not None:
            upstream=f32(upstream)
            if upstream.shape != out.shape: raise ValueError('invalid MLP upstream gradient')
            gradient=np.empty_like(self.parameters)
        check(library().vh_train_mlp(pointer(out),pointer(gradient) if gradient is not None else None,
              pointer(x),pointer(self.parameters),pointer(upstream) if upstream is not None else None,
              len(x),inputs,hidden,outputs))
        return out if gradient is None else (out,gradient)

    __call__ = compute


def spheres(x, ids, centers, thresholds):
    x,centers,thresholds=f32(x),f32(centers),f32(thresholds);ids=i32(ids)
    if (x.ndim!=3 or x.shape[2]!=3 or ids.ndim!=1 or centers.ndim!=3 or centers.shape[0]!=len(x) or
            centers.shape[2]!=3 or thresholds.shape!=(len(ids),centers.shape[1])):
        raise ValueError('invalid sphere tensors')
    gradient=np.empty_like(x);depth=np.empty((len(x),len(ids)),np.float32);energy=np.empty(len(x),np.float32)
    check(library().vh_train_spheres(pointer(x),ip(ids),pointer(centers),pointer(thresholds),
          len(x),x.shape[1],len(ids),centers.shape[1],pointer(gradient),pointer(depth),pointer(energy)))
    return energy,gradient,depth


def pairs(x, upper, lower, up, floor):
    x,up,floor=f32(x),f32(up),f32(floor);upper,lower=i32(upper),i32(lower)
    if (x.ndim!=3 or x.shape[2]!=3 or upper.ndim!=1 or lower.shape!=upper.shape or
            up.shape!=(len(x),3) or floor.shape!=upper.shape): raise ValueError('invalid lip-pair tensors')
    gradient=np.empty_like(x);depth=np.empty((len(x),len(upper)),np.float32);energy=np.empty(len(x),np.float32)
    check(library().vh_train_pairs(pointer(x),ip(upper),ip(lower),pointer(up),pointer(floor),
          len(x),x.shape[1],len(upper),pointer(gradient),pointer(depth),pointer(energy)))
    return energy,gradient,depth


def arap(x, linear, edges, target, weight):
    x,linear,target=f32(x),f32(linear),f32(target);edges=i32(edges)
    if (x.ndim!=3 or x.shape[2]!=3 or linear.shape!=x.shape or edges.ndim!=2 or edges.shape[1]!=2 or
            target.shape!=(len(x),len(edges),3)): raise ValueError('invalid ARAP tensors')
    gradient=np.empty_like(x);energy=np.empty(len(x),np.float32)
    check(library().vh_train_arap(pointer(x),pointer(linear),ip(edges),pointer(target),len(x),x.shape[1],
          len(edges),weight,pointer(gradient),pointer(energy)))
    return energy,gradient


def rotations(x, rest_edges, rest_normals, normal_scale, edges, faces):
    x,e0,n0,scale=f32(x),f32(rest_edges),f32(rest_normals),f32(normal_scale);edges,faces=i32(edges),i32(faces)
    if (x.ndim!=3 or x.shape[2]!=3 or edges.ndim!=2 or edges.shape[1]!=2 or faces.ndim!=2 or faces.shape[1]!=3 or
            e0.shape!=(len(edges),3) or n0.shape!=x.shape[1:] or scale.shape!=(x.shape[1],)):
        raise ValueError('invalid ARAP rotation tensors')
    out=np.empty((*x.shape[:2],3,3),np.float32)
    check(library().vh_train_rotations(pointer(x),pointer(e0),pointer(n0),pointer(scale),ip(edges),ip(faces),
          len(x),x.shape[1],len(edges),len(faces),pointer(out)))
    return out


def euler_zyx(r):
    rx,ry,rz=np.moveaxis(r,-1,0);cx,sx,cy,sy,cz,sz=np.cos(rx),np.sin(rx),np.cos(ry),np.sin(ry),np.cos(rz),np.sin(rz)
    o,z=np.ones_like(rx),np.zeros_like(rx)
    rx=np.stack((o,z,z,z,cx,-sx,z,sx,cx),-1).reshape(*r.shape[:-1],3,3)
    ry=np.stack((cy,z,sy,z,o,z,-sy,z,cy),-1).reshape(*r.shape[:-1],3,3)
    rz=np.stack((cz,-sz,z,sz,cz,z,z,z,o),-1).reshape(*r.shape[:-1],3,3)
    return rz@ry@rx


class NativeRig:
    """Training-time rig: repository GEMM with NumPy small transform algebra.

    It is not differentiable; contact relaxation needs gradients only with
    respect to post-skinning offsets, supplied by the native analytic solver.
    """
    def __init__(self, definition, rest, shapes, joints, weights):
        self.d=definition;self.controls=[c['name'] for c in definition['controls']]
        self.inputs=self.controls+[c['name'] for c in definition['correctives']]
        ci={n:i for i,n in enumerate(self.controls)};ii={n:i for i,n in enumerate(self.inputs)}
        ji={j['name']:i for i,j in enumerate(definition['joints'])};self.J=len(ji)
        self.rest=f32(rest);self.jn=i32(joints);self.w=f32(weights)
        if (self.rest.ndim!=2 or self.rest.shape[1]!=3 or self.jn.ndim!=2 or self.jn.shape!=self.w.shape or
                len(self.jn)!=len(self.rest) or (self.jn<0).any() or (self.jn>=self.J).any() or
                not np.isfinite(self.rest).all() or not np.isfinite(self.w).all() or (self.w<0).any()):
            raise ValueError('invalid native training rig geometry/weights')
        self.w/=np.maximum(self.w.sum(1,keepdims=True),1e-12)
        self.lo=f32([c['min'] for c in definition['controls']]);self.hi=f32([c['max'] for c in definition['controls']])
        self.corr=[([ci[n] for n in c['inputs']],float(c.get('weight',1))) for c in definition['correctives']]
        self.parent=[ji.get(j['parent'],-1) if j['parent'] else -1 for j in definition['joints']]
        if any(p>=i for i,p in enumerate(self.parent)): raise ValueError('training joints must be parent ordered')
        self.rest_t=f32([j['rest_translation'] for j in definition['joints']]);self.rest_R=f32([j['rest_rotation'] for j in definition['joints']])
        self.inv_bind=f32(np.linalg.inv(np.array([j['bind'] for j in definition['joints']],np.float64)))
        self.M=np.zeros((self.J*6,len(self.inputs)),np.float32)
        for e in definition['joint_matrix']:self.M[ji[e['joint']]*6+ATTRS.index(e['attr']),ii[e['input']]]+=e['value']
        shape_names=[b['name'] for b in definition['blendshapes'] if b['name'] in shapes]
        self.shape_src=i32([ii[b['input']] for b in definition['blendshapes'] if b['name'] in shapes])
        self.D=f32(np.stack([shapes[n] for n in shape_names]).reshape(len(shape_names),-1)) if shape_names else np.empty((0,self.rest.size),np.float32)

    def input_vector(self, controls):
        x=f32(controls)
        if x.ndim!=2 or x.shape[1]!=len(self.controls) or not np.isfinite(x).all():raise ValueError('invalid training controls')
        x=np.clip(x,self.lo,self.hi);correctives=[]
        for ids,weight in self.corr:correctives.append(np.minimum(1,weight*np.prod(np.clip(x[:,ids],0,1),axis=1)))
        return f32(np.concatenate((x,np.stack(correctives,1)),1) if correctives else x)

    def __call__(self, controls, pre=None):
        inp=self.input_vector(controls);n=len(inp)
        delta=matmul(inp,self.M,transpose_b=True).reshape(n,self.J,6)
        local=np.zeros((n,self.J,4,4),np.float32);local[:,:,:3,:3]=self.rest_R@euler_zyx(delta[:,:,3:])
        local[:,:,:3,3]=self.rest_t+delta[:,:,:3];local[:,:,3,3]=1
        world=np.empty_like(local)
        for j,parent in enumerate(self.parent):world[:,j]=local[:,j] if parent<0 else world[:,parent]@local[:,j]
        skin=world@self.inv_bind
        position=np.broadcast_to(self.rest,(n,*self.rest.shape)).copy()
        if len(self.D):position+=matmul(inp[:,self.shape_src],self.D).reshape(position.shape)
        if pre is not None:
            pre=f32(pre)
            if pre.shape!=position.shape:raise ValueError('invalid pre-skinning offsets')
            position+=pre
        # Accumulate influences one at a time; never materialize S*V*K*4*4.
        blend=np.zeros((n,len(self.rest),4,4),np.float32)
        for k in range(self.jn.shape[1]):blend+=self.w[None,:,k,None,None]*skin[:,self.jn[:,k]]
        pos=(blend[:,:,:3,:3]@position[...,None])[...,0]+blend[:,:,:3,3]
        return dict(pos=pos,skin=skin,blend=blend[:,:,:3,:3],inputs=inp,pre=position)
