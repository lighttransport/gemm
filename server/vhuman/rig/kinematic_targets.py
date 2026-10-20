"""CPU rigid interpolation of prescribed anatomy between physics endpoints."""
import math
import numpy as np


class KinematicTargets:
    def __init__(self,state):
        self.state=state
        self.ids=np.flatnonzero((state.bone==0)&(state.mass==0))
        self.start=state.positions[self.ids].copy()
        self.goal=state.rigid_targets[self.ids].copy()
        self.groups=[]
        if not np.isfinite(self.goal).all():raise ValueError('nonfinite rigid kinematic target')
        self.moving=bool(len(self.ids) and not np.array_equal(self.start.astype(np.float32),self.goal.astype(np.float32)))
        if not self.moving:return
        from scipy.spatial.transform import Rotation
        local=np.full(len(state.rest),-1,np.int32);local[self.ids]=np.arange(len(self.ids))
        claimed=np.zeros(len(self.ids),bool)
        for part in state.metadata.get('parts',[]):
            samples=np.arange(part['surface_start'],part['surface_start']+part['surface_count'])
            nodes=np.unique(state.embed_ids[samples]);indices=local[nodes]
            if len(indices)<3 or (indices<0).any() or claimed[indices].any():continue
            start,goal=self.start[indices],self.goal[indices]
            a,b=start.mean(0),goal.mean(0)
            u,_,vt=np.linalg.svd((start-a).T@(goal-b))
            rotation=u@np.diag([1.,1.,np.linalg.det(u@vt)])@vt
            error=np.linalg.norm((start-a)@rotation+b-goal,axis=1).max()
            if error>1e-6:raise ValueError('rigid kinematic target changes part shape')
            vector=Rotation.from_matrix(rotation.T).as_rotvec()
            self.groups.append((indices,start-a,a,b,vector))
            claimed[indices]=True

    def apply(self,fraction):
        if not math.isfinite(fraction) or not 0<=fraction<=1:raise ValueError('invalid kinematic interpolation fraction')
        if not self.moving:return
        if fraction==0:result=self.start
        elif fraction==1:result=self.goal
        else:
            from scipy.spatial.transform import Rotation
            result=(1-fraction)*self.start+fraction*self.goal
            for indices,relative,a,b,vector in self.groups:
                rotation=Rotation.from_rotvec(fraction*vector).as_matrix()
                result[indices]=relative@rotation.T+(1-fraction)*a+fraction*b
        self.state.rigid_targets[self.ids]=result
