"""Original calibrated multi-light variable-projection material fit.

Albedo is solved linearly per sample; only global roughness and dielectric F0
are nonlinear. Require three known lights/exposures, varied normals, and a
held-out view improvement. Uncalibrated portraits retain artist priors.
"""
import numpy as np
from .reference import ggx


def coefficients(normals, viewdirs, lights, roughness, f0):
    diffuse, specular = [], []
    for n,v,lighting in zip(normals,viewdirs,lights):
        direction=np.asarray(lighting['direction_h'],float)
        radiance=np.asarray(lighting['radiance'],float)
        ambient=np.asarray(lighting.get('ambient_rgb',[0,0,0]),float)
        d,s=ggx(np.ones_like(n),n,v,direction,roughness,f0)
        diffuse.append(d*radiance+ambient/np.pi)
        specular.append(s*radiance)
    return np.array(diffuse),np.array(specular)


def albedo(rgb,confidence,d,s):
    w=confidence[...,None]
    return np.clip((w*d*(rgb-s)).sum(0)/np.maximum((w*d*d).sum(0),1e-9),0,1)


def fit(rgb,normals,viewdirs,confidence,lights,roughness=.55,f0=.028):
    from scipy.optimize import least_squares
    if len(lights)<3 or any(light is None for light in lights):
        raise ValueError('three calibrated light/exposure views required')
    direction=np.array([l['direction_h'] for l in lights],float)
    direction/=np.linalg.norm(direction,axis=1,keepdims=True)
    if np.min(direction@direction.T)>.94:
        raise ValueError('calibrated lights lack angular diversity')
    common=(confidence>.4).sum(0)>=3
    ids=np.flatnonzero(common)
    if len(ids)<64:
        raise ValueError('not enough jointly visible calibrated samples')
    ids=ids[::max(1,len(ids)//1000)]
    if np.linalg.eigvalsh(np.cov(normals[0,ids].T)).min()<1e-5:
        raise ValueError('insufficient normal diversity')
    observed,n,v,w=rgb[:,ids],normals[:,ids],viewdirs[:,ids],confidence[:,ids]
    def predict(parameters):
        d,s=coefficients(n,v,lights,*parameters)
        a=albedo(observed[:-1],w[:-1],d[:-1],s[:-1])
        return a[None]*d+s
    def objective(parameters):
        return ((predict(parameters)[:-1]-observed[:-1])*np.sqrt(w[:-1,:,None])).ravel()
    prior=np.array([roughness,f0])
    result=least_squares(objective,prior,bounds=([.08,.005],[1.,.04]),loss='soft_l1',f_scale=.01,max_nfev=100)
    before=float(np.mean((predict(prior)[-1]-observed[-1])**2))
    after=float(np.mean((predict(result.x)[-1]-observed[-1])**2))
    if not result.success or after>=before*.98 or np.linalg.cond(result.jac.T@result.jac)>1e7:
        raise ValueError('calibrated fit failed held-out improvement or conditioning gate')
    d,s=coefficients(normals,viewdirs,lights,*result.x)
    a=albedo(rgb,confidence,d,s)
    return a, float(result.x[0]),float(result.x[1]),dict(method='calibrated GGX variable projection',
        status='fitted global scalar',views=len(lights),samples=len(ids),heldout_before=before,heldout_after=after,
        gauge='supplied radiance/exposure; no automatic median-light normalization')
