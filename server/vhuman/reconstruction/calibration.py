"""Measured Lambertian gray-card irradiance calibration, in linear RGB.

Card reflectance, normal, exposure, ambient and a measured light direction are
inputs. This recovers a radiometric gauge, not physical watts or skin SSS.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from .artifacts import artifact_path
from .observations import sha256


def gray_card(pixels,reflectance,normal,direction,exposure=1.,ambient=(0,0,0)):
    pixels=np.asarray(pixels,float).reshape(-1,3);rho=np.broadcast_to(np.asarray(reflectance,float),(3,))
    normal,direction=np.asarray(normal,float),np.asarray(direction,float);ambient=np.asarray(ambient,float)
    if (normal.shape!=(3,) or direction.shape!=(3,) or ambient.shape!=(3,) or not np.isfinite(pixels).all()
            or not np.isfinite(np.r_[rho,normal,direction,ambient,exposure]).all() or len(pixels)<64
            or (rho<=0).any() or (rho>1).any() or (ambient<0).any() or exposure<=0
            or np.linalg.norm(normal)<1e-6 or np.linalg.norm(direction)<1e-6):raise ValueError('invalid measured gray-card calibration')
    normal=normal/np.linalg.norm(normal);direction=direction/np.linalg.norm(direction)
    cosine=float(normal@direction)
    if cosine<.2:raise ValueError('gray card too grazing to calibrate')
    values=(pixels/exposure*np.pi/rho-ambient)/cosine
    radiance=np.median(values,axis=0)
    mad=np.median(abs(values-radiance),axis=0)*1.4826
    if (radiance<=0).any() or (mad/np.maximum(radiance,1e-9)>.15).any():raise ValueError('card is shadowed, contaminated or nonuniform')
    return dict(direction_h=direction.tolist(),radiance=radiance.tolist(),ambient_rgb=ambient.tolist(),exposure=float(exposure),
                calibration=dict(method='measured linear Lambertian gray-card irradiance',samples=len(pixels),reflectance=rho.tolist(),
                                 relative_mad=(mad/radiance).tolist(),absolute_power_watts=False))



def scattering_profile(distance_m,linear_rgb,prior=(.0012,.0006,.0003)):
    """Effective Gaussian line-spread widths from a measured narrow-light scan.

    This is a renderer approximation, not a recovered volumetric BSSRDF. Distances
    must be measured on skin; illumination/background must already be subtracted.
    """
    from scipy.optimize import least_squares
    distance=np.asarray(distance_m,float);rgb=np.asarray(linear_rgb,float);prior=np.asarray(prior,float)
    if (distance.ndim!=1 or rgb.shape!=(len(distance),3) or len(distance)<40 or prior.shape!=(3,)
            or not np.isfinite(np.r_[distance,rgb.ravel(),prior]).all() or (rgb<0).any()
            or np.any(np.diff(distance)<=0) or distance[0]>=0 or distance[-1]<=0
            or (prior<.00005).any() or (prior>.01).any()):raise ValueError('measured ordered line scan spanning its light centre required')
    test=np.arange(len(distance))%5==0;train=~test
    radii=[];reports=[]
    for channel in range(3):
        observed=rgb[:,channel]
        def prediction(radius):return np.exp(-.5*(distance/radius)**2)
        def amplitude(shape):return float(np.maximum(shape[train]@observed[train]/max(shape[train]@shape[train],1e-12),0))
        def residual(log_radius):
            shape=prediction(np.exp(log_radius[0]));return (shape*amplitude(shape)-observed)[train]
        solve=least_squares(residual,[np.log(prior[channel])],bounds=([np.log(.00005)],[np.log(.01)]),max_nfev=100)
        radius=float(np.exp(solve.x[0]));shape=prediction(radius);base=prediction(prior[channel])
        before=float(np.mean((base[test]*amplitude(base)-observed[test])**2))
        after=float(np.mean((shape[test]*amplitude(shape)-observed[test])**2))
        accepted=bool(solve.success and after<before*.98 and np.linalg.norm(solve.jac)>1e-6 and radius>.000051 and radius<.0099)
        radii.append(radius if accepted else float(prior[channel]))
        reports.append(dict(accepted=accepted,heldout_before=before,heldout_after=after,amplitude=amplitude(shape)))
    return dict(format='vhuman.scattering_profile.v1',radii_m=radii,channels=reports,enabled=False,
        status='measured effective line-spread approximation',samples=len(distance),heldout_samples=int(test.sum()),
        limitations=['requires independently measured distances and corrected narrow-light intensity',
                     'Gaussian widths are effective rendering parameters, not tissue-layer anatomy',
                     'does not separate illumination footprint from tissue unless the footprint is deconvolved'])


def run(spec,out):
    from .emily import read_linear
    spec,out=Path(spec),artifact_path(out);doc=json.loads(spec.read_text());rows=[]
    if doc.get('format')!='vhuman.gray_card_capture.v1' or not 3<=len(doc.get('views',[]))<=16:raise ValueError('3..16 measured light views required')
    for row in doc['views']:
        path=(spec.parent/row['image']).resolve()
        if sha256(path)!=row['sha256']:raise ValueError('calibration image hash mismatch')
        image,stride,record=read_linear(path,max_side=100000)
        if stride!=1:raise ValueError('gray-card sampling requires native EXR dimensions')
        x0,y0,x1,y1=row['card_roi']
        if any(type(x) is not int for x in (x0,y0,x1,y1)) or not 0<=x0<x1<=image.shape[1] or not 0<=y0<y1<=image.shape[0]:raise ValueError('card ROI outside image')
        lighting=gray_card(image[y0:y1,x0:x1],doc['reflectance'],row['card_normal_h'],row['direction_h'],row.get('exposure',1),row.get('ambient_rgb',[0,0,0]))
        rows.append(dict(image_sha256=record['sha256'],lighting=lighting))
    out.parent.mkdir(parents=True,exist_ok=True);result=dict(format='vhuman.measured_lighting.v1',capture_sha256=sha256(spec),views=rows,
        limitations=['card reflectance and ambient must be measured; do not substitute guessed values',
                     'directional approximation; near-field lights need per-surface position correction',
                     'SSS radii need independent spatial scattering measurements'])
    out.write_text(json.dumps(result,indent=2)+'\n');return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    source=p.add_mutually_exclusive_group(required=True);source.add_argument('--capture',type=Path);source.add_argument('--line-scan',type=Path)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.capture:result=run(a.capture,a.out)
    else:
        with np.load(a.line_scan,allow_pickle=False) as z:result=scattering_profile(z['distance_m'],z['linear_rgb'])
        result['measurement_sha256']=sha256(a.line_scan)
        out=artifact_path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':main()
