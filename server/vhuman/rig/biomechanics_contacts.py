"""Geometric articulator penalties evaluated on simulated CPU surfaces."""
import numpy as np

from ..reconstruction.contact_runtime import closest_on_triangles
from . import articulation


def labiodental_samples(rest,faces,lower,teeth):
    """Select medial inner-lip sites nearest the frontal upper dental surface."""
    ids=np.flatnonzero(lower)
    if not len(ids):raise ValueError('missing lower lip contact surface')
    center=float(np.median(rest[ids,0]));width=float(np.ptp(rest[ids,0]))
    ids=ids[abs(rest[ids,0]-center)<=.3*width+1e-12]
    triangles=faces[teeth[faces].all(1)]
    if not len(triangles):raise ValueError('missing upper dental contact surface')
    depth=rest[triangles,2].mean(1);triangles=triangles[depth>=np.quantile(depth,.7)]
    p=rest[triangles]
    _,_,distance=closest_on_triangles(rest[ids],p[None,:,0],p[None,:,1],p[None,:,2])
    count=min(32,max(1,len(ids)//2))
    ids=ids[np.argsort(distance.min(1),kind='stable')[:count]]
    return ids,triangles


class ContactGeometry:
    def __init__(self,arrays,model=None):
        if model is None:
            from .gnm_model import GNMModel
            model=GNMModel()
        self.rest=arrays['surface_rest'];self.faces=arrays['faces'];self.cache={}
        source=arrays['source_ids']
        self.upper=model.group('upper_lip')[source]
        self.lower=model.group('lower_lip')[source]
        self.teeth=(model.group('teeth')&model.group('upper_teeth_and_gums'))[source]

    @staticmethod
    def gaps(surface,ids,triangles):
        p=surface[triangles]
        valid=np.linalg.norm(np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]),axis=1)>1e-16
        p=p[valid]
        if not len(ids) or not len(p):raise FloatingPointError('missing or degenerate articulator contact surface')
        _,_,distance=closest_on_triangles(surface[ids],p[None,:,0],p[None,:,1],p[None,:,2])
        return distance.min(1)*1000

    def bilabial(self):
        if 'bilabial' not in self.cache:
            selectors=[]
            for source,target in ((self.upper,self.lower),(self.lower,self.upper)):
                ids=np.flatnonzero(source);triangles=self.faces[target[self.faces].all(1)]
                if not len(ids):raise ValueError('missing lip contact samples')
                x=self.rest[ids,0];center=float(np.median(x));width=float(np.ptp(x))
                ids=ids[abs(x-center)<=.3*width+1e-12]
                distances=self.gaps(self.rest,ids,triangles)
                ids=ids[np.argsort(distances,kind='stable')[:min(32,len(ids))]]
                selectors.append((ids,triangles))
            self.cache['bilabial']=selectors
        return self.cache['bilabial']

    def labiodental(self):
        if 'labiodental' not in self.cache:
            self.cache['labiodental']=[labiodental_samples(self.rest,self.faces,self.lower,self.teeth)]
        return self.cache['labiodental']

    def objective(self,values):
        values=dict(zip(articulation.CHANNELS,values));goals=[];receipt=[]
        for channel,name,threshold,selectors in (
                ('seal','bilabial',.15,self.bilabial),('labiodental','labiodental',.35,self.labiodental)):
            strength=float(values[channel])
            if strength<=0:continue
            selected=selectors()
            initial=float(np.median(np.concatenate([self.gaps(self.rest,*pair) for pair in selected])))
            desired=(1-strength)*initial+strength*threshold
            goals.append((selected,desired,strength))
            receipt.append(dict(contact_type=name,target_gap_mm=desired,strength=strength,
                                source='authored geometric contact goal on simulated surfaces'))
        if not goals:return None,[]
        def residual(surface):
            return np.array([strength*max(0,float(np.median(np.concatenate(
                [self.gaps(surface,*pair) for pair in selected])))-desired) for selected,desired,strength in goals])
        return residual,receipt
