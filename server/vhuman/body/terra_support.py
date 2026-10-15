"""Conservative flat-support evidence from posed native skin, in source Y-up."""
import numpy as np
from .musclemimic_kinematics import FEET


def flat_support_evidence(source):
    if 'sole_vertices' not in source:
        return dict(qualified=False, available=False)
    times = np.asarray(source['times'])
    vertices = np.asarray(source['sole_vertices'])
    sides = np.asarray(source['sole_sides'])
    if (vertices.ndim != 3 or vertices.shape[0] != len(times) or vertices.shape[2] != 3
            or sides.shape != (vertices.shape[1],) or set(sides.tolist()) != {0,1}
            or len(times) < 5 or not np.isfinite(vertices).all() or np.any(np.diff(times) <= 0)):
        raise ValueError('invalid native support evidence')
    feet = np.asarray(source['positions'])[:,FEET]
    speed = np.linalg.norm(np.gradient(feet,times,axis=0),axis=-1)
    stance = (speed < .30).reshape(len(times),2,2).any(2)
    for side in range(2):
        for a,b in np.flatnonzero(np.diff(np.r_[0,stance[:,side].astype(int),0])).reshape(-1,2):
            if b-a < 5:
                stance[a:b,side] = False
    # The whole foot's lowest skin point remains at the datum during a heel
    # lift. A fixed vertical ankle offset would instead suggest a raised step.
    low = np.stack([vertices[:,sides==side,1].min(1) for side in range(2)],axis=1)
    observed = abs(low[stance])
    counts = stance.sum(0)
    maximum = float(observed.max()) if len(observed) else None
    return dict(qualified=bool((counts >= 5).all() and maximum is not None and maximum <= .01),
        available=True, method='posed_native_whole_foot_skin', ground_datum_m=0., tolerance_m=.01,
        stance_samples=counts.tolist(), stance_max_abs_clearance_m=maximum,
        stance_p95_abs_clearance_m=float(np.percentile(observed,95)) if len(observed) else None)
