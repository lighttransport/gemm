"""Statistical skin-texture prior in GNM UV space (mean + PCA), learned outside the repository, loaded on demand.

File format (`skin_prior_v1.npz`, written by server/vhuman/metahuman/skin_prior_fit.py):
  res            int: atlas resolution the prior was fitted at
  mask           (res,res) bool: texels covered by the prior (skin exterior with MetaHuman support)
  lf_res         int: low-frequency grid (e.g. 256)
  lf_mean        (lf_res,lf_res,3) float32 log-linear-RGB mean
  lf_basis       (K_lf, lf_res*lf_res*3) float16 orthonormal rows
  lf_sigma       (K_lf,) float32 per-mode std of the training coefficients
  hf_mean        (res,res,3) float16 log residual mean (albedo - up(LF))
  hf_basis       (K_hf, res*res*3) float16
  hf_sigma       (K_hf,)
  hf_std         (res,res) float16 per-texel std of the HF residual (for confidence)
  normal_mean    (res,res,2) float16 tangent-space xy of the mean normal
  cavity_mean    (res,res) float16
  sha256 of the file is checked against the value given by the caller (candidate manifests pin it).

The prior is never enabled by default; callers pass an explicit path (`--skin-prior`).
"""
import hashlib
import json
from pathlib import Path

import numpy as np


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


class SkinPrior:
    def __init__(self, path, expected_sha256=None):
        self.path = Path(path)
        self.sha256 = file_sha256(self.path)
        if expected_sha256 and expected_sha256 != self.sha256:
            raise ValueError('skin prior hash mismatch')
        z = np.load(self.path)
        self.res = int(z['res']); self.lf_res = int(z['lf_res'])
        self.mask = z['mask'].astype(bool)
        self.lf_mean = z['lf_mean'].astype(np.float32)
        self.lf_basis = z['lf_basis'].astype(np.float32)
        self.lf_sigma = z['lf_sigma'].astype(np.float32)
        self.hf_mean = z['hf_mean'].astype(np.float32)
        self.hf_basis = z['hf_basis'].astype(np.float32) if 'hf_basis' in z.files and z['hf_basis'].size else None
        self.hf_sigma = z['hf_sigma'].astype(np.float32) if self.hf_basis is not None else None
        self.hf_std = z['hf_std'].astype(np.float32) if 'hf_std' in z.files else None
        self.normal_mean = z['normal_mean'].astype(np.float32) if 'normal_mean' in z.files else None
        self.cavity_mean = z['cavity_mean'].astype(np.float32) if 'cavity_mean' in z.files else None
        report = self.path.with_name(self.path.stem + '.report.json')
        self.report = json.loads(report.read_text()) if report.is_file() else {}

    # ---- resampling helpers -------------------------------------------------------------------------------------
    @staticmethod
    def downsample(img, res):
        """Box-average (H,W,C) or (H,W) to res x res (H, W divisible by res)."""
        h = img.shape[0]; f = h // res
        if f * res != h:
            raise ValueError('resolution must divide')
        if img.ndim == 2:
            return img.reshape(res, f, res, f).mean((1, 3))
        return img.reshape(res, f, res, f, img.shape[-1]).mean((1, 3))

    @staticmethod
    def upsample(img, res):
        """Bilinear-free nearest+smooth upsample: repeat then box blur of the factor (cheap, separable)."""
        f = res // img.shape[0]
        if f == 1:
            return img
        up = np.repeat(np.repeat(img, f, 0), f, 1)
        k = np.ones(f, np.float32) / f
        for axis in (0, 1):
            up = np.apply_along_axis(lambda m: np.convolve(m, k, mode='same'), axis, up)
        return up.astype(np.float32)

    def resampled(self, res):
        """Mask, HF mean, normal and cavity means at another atlas resolution (nearest/box)."""
        if res == self.res:
            return self.mask, self.hf_mean, self.normal_mean, self.cavity_mean
        from PIL import Image
        def rs(a, mode):
            if a is None:
                return None
            if a.ndim == 2:
                return np.asarray(Image.fromarray(a.astype(np.float32), 'F').resize((res, res), mode), np.float32)
            return np.stack([rs(a[..., c], mode) for c in range(a.shape[-1])], -1)
        mask = rs(self.mask.astype(np.float32), Image.BILINEAR) > .5
        return mask, rs(self.hf_mean, Image.BILINEAR), rs(self.normal_mean, Image.BILINEAR), rs(self.cavity_mean, Image.BILINEAR)

    # ---- fitting --------------------------------------------------------------------------------------------------
    def fit_coefficients(self, log_rgb, weight, *, ridge=1.0, hf=True):
        """Weighted least squares of the prior to observed texels.

        log_rgb: (res,res,3) log linear RGB (any res divisible by lf_res; HF fitted only when res == self.res)
        weight:  (res,res) in [0,1] observation confidence (0 = unobserved)
        Returns dict(lf=(K_lf,), hf=(K_hf,) or None, res).
        """
        res = log_rgb.shape[0]
        mask, hf_mean, _, _ = self.resampled(res)
        w = np.asarray(weight, np.float32) * mask
        # LF: box-downsample the weighted observation
        f = res // self.lf_res
        wl = w.reshape(self.lf_res, f, self.lf_res, f).mean((1, 3))
        xl = (log_rgb * w[..., None]).reshape(self.lf_res, f, self.lf_res, f, 3).sum((1, 3)) / np.maximum(wl[..., None] * f * f, 1e-6)
        wl = wl.reshape(-1)
        d = (xl - self.lf_mean).reshape(-1, 3)
        lf = self._wls(self.lf_basis, self.lf_sigma, d, wl, ridge)
        out = dict(lf=lf, hf=None, res=res)
        if hf and self.hf_basis is not None and res == self.res:
            recon_lf = (self.lf_mean.reshape(-1, 3) + (lf @ self.lf_basis).reshape(-1, 3)).reshape(self.lf_res, self.lf_res, 3)
            resid = log_rgb - self.upsample(recon_lf, res) - hf_mean
            out['hf'] = self._wls(self.hf_basis, self.hf_sigma, resid.reshape(-1, 3), w.reshape(-1), ridge)
        return out

    @staticmethod
    def _wls(basis, sigma, d, w, ridge):
        """min_c sum_i w_i |d_i - (c B)_i|^2 + ridge * sum_k (c_k/sigma_k)^2 ; basis rows are flattened (texel*3)."""
        K = basis.shape[0]
        w3 = np.repeat(w, 3).astype(np.float32)
        bw = basis * w3[None]                               # (K, D)
        a = bw @ basis.T + np.diag(ridge / np.maximum(sigma, 1e-6) ** 2)
        b = bw @ d.reshape(-1)
        return np.linalg.solve(a, b).astype(np.float32)

    def reconstruct(self, coeffs, res=None):
        """Log linear RGB (res,res,3) of the prior at the given coefficients (LF everywhere, HF where available)."""
        res = res or coeffs['res']
        lf = (self.lf_mean.reshape(-1, 3) + (coeffs['lf'] @ self.lf_basis).reshape(-1, 3)).reshape(self.lf_res, self.lf_res, 3)
        img = self.upsample(lf, res)
        _, hf_mean, _, _ = self.resampled(res)
        img = img + hf_mean
        if coeffs.get('hf') is not None and res == self.res:
            img = img + (coeffs['hf'] @ self.hf_basis).reshape(res, res, 3)
        return img.astype(np.float32)

    def coefficient_norms(self, coeffs):
        out = dict(lf=float(np.sqrt(np.mean((coeffs['lf'] / np.maximum(self.lf_sigma, 1e-6)) ** 2))))
        if coeffs.get('hf') is not None:
            out['hf'] = float(np.sqrt(np.mean((coeffs['hf'] / np.maximum(self.hf_sigma, 1e-6)) ** 2)))
        return out


def pca_small_n(matrix, rank):
    """Centred PCA for N << D via the N x N Gram matrix (no D x D covariance, no native GEMM needed).

    Returns mean (D,), basis (rank, D) orthonormal rows, sigma (rank,) std of the training coefficients, explained (rank,).
    """
    x = np.asarray(matrix, np.float32)
    mean = x.mean(0)
    xc = x - mean
    g = xc @ xc.T                                            # (N,N)
    evals, evecs = np.linalg.eigh(g.astype(np.float64))
    order = np.argsort(evals)[::-1][:rank]
    evals, evecs = np.maximum(evals[order], 1e-12), evecs[:, order]
    basis = (xc.T @ evecs) / np.sqrt(evals)[None]            # (D, rank), unit columns
    coeffs = xc @ basis                                      # (N, rank)
    sigma = coeffs.std(0) + 1e-6
    explained = evals / max(float(np.trace(g)), 1e-12)
    return mean, np.ascontiguousarray(basis.T, np.float32), sigma.astype(np.float32), explained.astype(np.float32)


def complete_with_prior(prior, res, yy, xx, fused, weight, measured, points, normals, *, ridge=1.0, blend_sigma=6.):
    """Prior-residual completion for materials.bake_portrait.

    fused (N,3) linear albedo at covered texels (yy,xx) of a res x res atlas, weight (N,) confidence, measured (N,) bool.
    1. Fit the prior to the observed texels (downsampled to the prior's resolution, confidence-weighted, log space).
    2. Complete the residual (fused - prior) harmonically on the surface graph (texture_completion.harmonic).
    3. completed = prior + residual: measured texels keep their value exactly; unobserved texels get the prior's
       prediction corrected by the nearby observed residual.
    Texels outside the prior's mask keep the ordinary completion, cross-faded over `blend_sigma` texels.
    Returns completed (N,3), confidence (N,), report dict.
    """
    from scipy.ndimage import gaussian_filter
    from PIL import Image
    from .texture_completion import harmonic
    if res % prior.lf_res:
        raise ValueError('atlas resolution must be a multiple of the prior LF resolution')
    log_img = np.zeros((res, res, 3), np.float32); w_img = np.zeros((res, res), np.float32)
    log_img[yy, xx] = np.log(np.maximum(fused, 1e-3)); w_img[yy, xx] = np.clip(weight, 0, 1) * measured
    if res != prior.res:
        f = res // prior.res if res > prior.res else None
        if f is None or f * prior.res != res:
            raise ValueError('atlas resolution must be a multiple of the prior resolution')
        w_low = w_img.reshape(prior.res, f, prior.res, f).sum((1, 3))
        l_low = (log_img * w_img[..., None]).reshape(prior.res, f, prior.res, f, 3).sum((1, 3)) / np.maximum(w_low[..., None], 1e-6)
        w_low = w_low / (f * f)
    else:
        l_low, w_low = log_img, w_img
    coeffs = prior.fit_coefficients(l_low, w_low, ridge=ridge)
    recon = prior.reconstruct(coeffs)                                   # (prior.res,prior.res,3) log
    mask = prior.mask.astype(np.float32)
    if res != prior.res:
        recon = np.stack([np.asarray(Image.fromarray(recon[..., c], 'F').resize((res, res), Image.BILINEAR)) for c in range(3)], -1)
        mask = np.asarray(Image.fromarray(mask, 'F').resize((res, res), Image.BILINEAR))
    prior_rgb = np.exp(recon[yy, xx]).astype(np.float64)
    residual = np.clip(fused - prior_rgb + .5, 0, 1)                     # harmonic() clips to [0,1]: offset the signed residual
    res_completed, confidence, rep = harmonic(points, normals, residual, measured)
    with_prior = np.clip(prior_rgb + (res_completed - .5), 0, 1)
    with_prior[measured] = fused[measured]
    plain, _, _ = harmonic(points, normals, fused, measured)
    soft = gaussian_filter((mask > .5).astype(np.float32), blend_sigma)[yy, xx][:, None]
    completed = soft * with_prior + (1 - soft) * plain
    completed[measured] = fused[measured]
    report = dict(method='metahuman_prior_residual', prior_sha256=prior.sha256, prior_res=prior.res, ridge=ridge,
                  lf_modes=int(prior.lf_basis.shape[0]), hf_modes=int(0 if prior.hf_basis is None else prior.hf_basis.shape[0]),
                  coefficient_norms=prior.coefficient_norms(coeffs), prior_mask_fraction=float((soft > .5).mean()),
                  harmonic=rep, completion_is_observation=False, new_view_evidence=False,
                  limitations=['prior is the MetaHuman synthesized-albedo distribution, not this subject',
                               'residual completion only corrects low-frequency offsets near observed skin'])
    return completed, confidence, report
