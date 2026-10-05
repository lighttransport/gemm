"""Geometric control-to-GNM PCA mapping; coefficients are not semantic names."""
import numpy as np


def project(basis, targets, names, coefficient_names=None, ridge=.05):
    basis = np.asarray(basis, np.float64)
    targets = np.asarray(targets, np.float64)
    if basis.ndim != 3 or targets.ndim != 3 or basis.shape[1:] != targets.shape[1:]:
        raise ValueError('basis and target vertex layouts must match')
    if len(names) != len(targets) or not np.isfinite(basis).all() or not np.isfinite(targets).all():
        raise ValueError('invalid control mapping inputs')
    ids = np.arange(0, basis.shape[1], max(1, basis.shape[1] // 850))
    A = basis[:, ids].reshape(len(basis), -1).T
    Y = targets[:, ids].reshape(len(targets), -1).T
    regularization = max(float(np.median(np.sum(A*A, axis=0))) * ridge, 1e-10)
    coefficients = np.linalg.solve(A.T @ A + regularization*np.eye(len(basis)), A.T @ Y)
    coefficients = np.clip(coefficients, -3, 3)
    errors = np.sqrt(np.mean((A @ coefficients - Y)**2, axis=0))
    return {'schema': 'vhuman.gnm_expression_map.v1', 'controls': list(names),
            'coefficient_names': list(coefficient_names or map(str, range(len(basis)))),
            'coefficients': coefficients.tolist(), 'rms_residual_metres': errors.tolist(),
            'method': 'ridge projection of geometric morph deltas; clipped coefficients [-3,3]',
            'includes_joint_skinning': False}


def evaluate(mapping, controls):
    weights = np.array([controls.get(name, 0) for name in mapping['controls']], np.float64)
    if not np.isfinite(weights).all():
        raise ValueError('nonfinite expression controls')
    return np.clip(np.asarray(mapping['coefficients']) @ np.clip(weights, 0, 1), -3, 3)
