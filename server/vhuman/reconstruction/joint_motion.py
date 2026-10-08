"""Rigid motion for a fitted joint center carried by its parent frame."""
import numpy as np


def shifted_joint_positions(positions, rest_joint, posed_joints, rotations,
                            center_offset, parent_rotations):
    """Rotate around a fitted center without orbiting its offset under gaze.

    All positions and offsets use the aligned rest/world coordinate frame.
    ``rotations`` and ``parent_rotations`` are world-space rotations from that
    frame, one per pose. The fitted center is ``rest_joint + center_offset``;
    its offset follows the parent, while local surface orientation follows the
    joint. This changes rigid placement only, with no contact acceptance.
    """
    positions, rest_joint, posed_joints, rotations, offset, parent = (
        np.asarray(x, dtype=float) for x in
        (positions, rest_joint, posed_joints, rotations, center_offset, parent_rotations))
    if (positions.ndim != 2 or positions.shape[1] != 3 or not len(positions)
            or rest_joint.shape != (3,) or offset.shape != (3,)
            or rotations.ndim != 3 or rotations.shape[1:] != (3, 3)
            or not len(rotations) or parent.shape != rotations.shape
            or posed_joints.shape != (len(rotations), 3)
            or not all(np.isfinite(x).all() for x in
                       (positions, rest_joint, posed_joints, rotations, offset, parent))):
        raise ValueError('invalid shifted-joint geometry or pose arrays')
    for matrices in (rotations, parent):
        if (not np.allclose(matrices.transpose(0, 2, 1) @ matrices, np.eye(3),
                            rtol=0, atol=1e-6)
                or not np.allclose(np.linalg.det(matrices), 1, rtol=0, atol=1e-6)):
            raise ValueError('joint transforms must be proper rotations')
    local = positions - rest_joint - offset
    centers = posed_joints + np.einsum('fij,j->fi', parent, offset)
    return np.einsum('fij,vj->fvi', rotations, local) + centers[:, None]
