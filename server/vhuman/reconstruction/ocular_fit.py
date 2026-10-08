"""Bounded iris reprojection fitting with fixed globe centres and explicit priors."""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def fit_eye(camera,centre,iris_z,iris_radius,observation, *, max_angle_degrees=20.,scale_bounds=(.9,1.1)):
    centre=np.asarray(centre,float);target=np.asarray(observation['center'],float);radius=float(observation['radius'])
    if (centre.shape!=(3,) or target.shape!=(2,) or not np.isfinite(np.r_[centre,target,radius,iris_z,iris_radius]).all()
        or radius<=0 or iris_z<=0 or iris_radius<=0 or not 0<max_angle_degrees<=30
        or not 0<scale_bounds[0]<=1<=scale_bounds[1]):raise ValueError('invalid ocular observations or bounds')
    angles=np.arange(64)*2*np.pi/64
    ring=np.column_stack((iris_radius*np.cos(angles),iris_radius*np.sin(angles),np.full(64,iris_z)))
    def prediction(x):
        rotation=Rotation.from_rotvec([x[0],x[1],0]).as_matrix()
        points=np.concatenate((np.array([[0,0,iris_z]]),ring))*x[2]
        pixels,_=camera.project(points@rotation.T+centre)
        # Source radius is a circular iris detector estimate; projected horizontal
        # radius is less biased by the portrait's eyelid occlusion than visible area.
        r=(pixels[1:,0].max()-pixels[1:,0].min())*.5
        return pixels[0],r,rotation
    def residual(x):
        pixel,r,_=prediction(x)
        return np.r_[pixel-target,r-radius,.15*x[:2],.15*(x[2]-1)]
    limit=np.deg2rad(max_angle_degrees)
    fit=least_squares(residual,[0,0,1],bounds=([-limit,-limit,scale_bounds[0]],[limit,limit,scale_bounds[1]]),xtol=1e-12,ftol=1e-12,gtol=1e-12)
    before,br,_=prediction([0,0,1]);after,ar,rotation=prediction(fit.x)
    accepted=bool(fit.success and np.linalg.norm(fit.x[:2])<=limit+1e-9 and np.linalg.norm(after-target)<.5 and abs(ar-radius)<.5
                  and np.linalg.norm(after-target)<np.linalg.norm(before-target))
    return dict(accepted=accepted,rotation=rotation.tolist(),scale=float(fit.x[2]),rotation_degrees=np.rad2deg(fit.x[:2]).tolist(),
        center_before_px=before.tolist(),center_after_px=after.tolist(),target_px=target.tolist(),
        center_error_before_px=float(np.linalg.norm(before-target)),center_error_after_px=float(np.linalg.norm(after-target)),
        radius_before_px=float(br),radius_after_px=float(ar),target_radius_px=radius,
        globe_center_fixed=True,max_angle_degrees=max_angle_degrees,scale_bounds=list(scale_bounds),
        limitations=['iris detector alignment is not independent 3D eye or gaze ground truth','hidden globe shape and refractive anatomy remain priors'])
