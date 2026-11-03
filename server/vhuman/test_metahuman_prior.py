"""Synthetic-fixture tests for the MetaHuman-derived prior machinery (no Epic data, no UE)."""
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from server.vhuman.reconstruction.skin_prior import SkinPrior, pca_small_n
from server.vhuman.metahuman import sampling


def synthetic_prior(tmp, res=64, lf_res=16, n=40, k_lf=5, k_hf=9, seed=0):   # k_hf = 4 HF modes + 5 LF resampling residual directions
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:res, 0:res] / res
    base = np.stack([np.log(.4 + .2 * xx), np.log(.3 + .1 * yy), np.log(.25 + .05 * xx * yy)], -1)
    lf_modes = rng.normal(size=(k_lf, lf_res, lf_res, 3)).astype(np.float32)
    hf_modes = rng.normal(size=(4, res, res, 3)).astype(np.float32) * .05
    samples = []
    for _ in range(n):
        c = rng.normal(size=k_lf) * .1; h = rng.normal(size=4)
        lf = SkinPrior.upsample(np.tensordot(c, lf_modes, 1).astype(np.float32), res)
        samples.append(base + lf + np.tensordot(h, hf_modes, 1))
    x = np.stack(samples).astype(np.float32)
    lf_data = np.stack([SkinPrior.downsample(s, lf_res) for s in x]).reshape(n, -1)
    lf_mean, lf_basis, lf_sigma, _ = pca_small_n(lf_data, k_lf)
    # HF residual against the K-mode LF reconstruction (what fit_coefficients sees), not the exact downsample
    lf_proj = lf_mean + (lf_data - lf_mean) @ lf_basis.T @ lf_basis
    resid = np.stack([s - SkinPrior.upsample(lf_proj[i].reshape(lf_res, lf_res, 3), res) for i, s in enumerate(x)])
    hf_mean, hf_basis, hf_sigma, _ = pca_small_n(resid.reshape(n, -1), k_hf)
    path = Path(tmp) / 'skin_prior_test.npz'
    np.savez(path, res=res, lf_res=lf_res, mask=np.ones((res, res), bool), lf_mean=lf_mean.reshape(lf_res, lf_res, 3), lf_basis=lf_basis.astype(np.float16),
             lf_sigma=lf_sigma, hf_mean=hf_mean.reshape(res, res, 3).astype(np.float16), hf_basis=hf_basis.astype(np.float16), hf_sigma=hf_sigma,
             hf_std=resid.std(0).mean(-1).astype(np.float16), normal_mean=np.zeros((res, res, 2), np.float16), cavity_mean=np.zeros((res, res), np.float16))
    return path, x


def test_pca_small_n_recovers_low_rank():
    rng = np.random.default_rng(1)
    basis = np.linalg.qr(rng.normal(size=(200, 3)))[0].T
    coeffs = rng.normal(size=(30, 3)) * np.array([3., 2., 1.])
    x = coeffs @ basis + 5.0
    mean, b, sigma, explained = pca_small_n(x, 3)
    recon = mean + (x - mean) @ b.T @ b
    assert np.abs(recon - x).max() < 1e-3
    assert explained.sum() > .999


def test_fit_full_and_partial_observation():
    with tempfile.TemporaryDirectory() as tmp:
        path, x = synthetic_prior(tmp)
        prior = SkinPrior(path)
        sample = x[3]
        full = prior.fit_coefficients(sample, np.ones(sample.shape[:2], np.float32), ridge=1e-3)
        recon = prior.reconstruct(full)
        lf_only = prior.reconstruct(dict(lf=full['lf'], hf=None, res=64))
        assert np.abs(recon - sample).mean() < .5 * np.abs(lf_only - sample).mean()
        # observe only the left half: the right half must still be close to the truth (LF prior) and better than the mean
        w = np.zeros(sample.shape[:2], np.float32); w[:, :32] = 1
        part = prior.fit_coefficients(sample, w, ridge=1e-2)
        recon_p = prior.reconstruct(part)
        err_unseen = np.abs(recon_p[:, 32:] - sample[:, 32:]).mean()
        err_mean = np.abs(prior.reconstruct(dict(lf=np.zeros_like(part['lf']), hf=None, res=64))[:, 32:] - sample[:, 32:]).mean()
        assert err_unseen < err_mean


def test_coefficient_roundtrip_and_blend():
    presets = {}
    rng = np.random.default_rng(0)
    for name in ('A', 'B', 'C'):
        patches = [dict(scale=1., quat=[0, 0, 0, 1.], translation=[0, 150, 3], weights=rng.normal(size=k).tolist()) for k in (3, 4)]
        presets[name] = dict(coefficients=sampling.flatten_coefficients(patches))
    p = sampling.parse_coefficients(presets['A']['coefficients'])
    assert sampling.flatten_coefficients(p) == presets['A']['coefficients']
    jobs, meta = sampling.build_jobs(presets, n_blend=2, n_gauss=4, held_out_frac=.3, render_subset=3, seed=1)
    assert len(jobs) == 3 + 2 + 4 and meta['patch_pca_sizes'] == [3, 4]
    for j in jobs:
        parsed = sampling.parse_coefficients(j['coefficients'])
        q = np.array(parsed[0]['quat']); assert abs(np.linalg.norm(q) - 1) < 1e-6
        assert 0 <= j['skin']['face_texture_index'] < sampling.N_TEXTURE_CHARACTERS
        assert j['skin']['face_texture_index'] not in meta['held_out_texture_indices']


def test_artifact_path_rejects_repo_outputs():
    from server.vhuman.reconstruction.artifacts import artifact_path
    with pytest.raises(ValueError):
        artifact_path(Path(__file__).parent / 'metahuman' / 'weights.npz')
    assert artifact_path('/mnt/nvme02/models/vhuman-mh/x.npz')
