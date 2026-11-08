"""Anatomical point weights for native-preserving offline skin-lighting profiles."""
import numpy as np


def skin_profile_weights(data, excluded_native):
    excluded = np.asarray(excluded_native, np.float32)
    ids = np.asarray(data['skin_native_ids'], np.int64)
    weights = 1. - excluded[ids]
    if len(data['skin_positions']) != len(weights):
        if 'skin_bound_ids' not in data:
            raise ValueError('refined skin requires anatomical binding weights')
        weights = np.concatenate((weights, 1. - (excluded[data['skin_bound_ids']] * data['skin_bound_weights']).sum(-1)))
    if len(weights) != len(data['skin_positions']) or not np.isfinite(weights).all():
        raise ValueError('anatomical weights do not match skin mesh')
    return np.clip(weights, 0, 1).astype(np.float32)
