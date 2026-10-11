"""Detect-then-estimate baseline.

A simplified pipeline in the spirit of Lounici & Pacreau (NeurIPS 2023), not the
authors' implementation. It keeps two ingredients:

1. hard entry-wise detection: entries flagged by a univariate median/MAD filter
   are turned into missing values;
2. an inverse-probability debiased covariance that needs no imputation,

       mu_j     = sum_i M_ij y_ij / sum_i M_ij
       delta_jk = (1/n) sum_i M_ij M_ik
       Sigma_jk = sum_i M_ij M_ik (y_ij - mu_j)(y_ik - mu_k) / (n delta_jk).

The debiased estimate is not guaranteed to be positive definite; it is projected
onto the positive-definite cone when needed, and ``info["psd_shift"]`` reports
the relative eigenvalue mass that was clipped.
"""

import numpy as np

__all__ = ["entrywise_filter", "debiased_covariance",
           "estimate_filter_debiased", "estimate_debiased_only"]


def entrywise_filter(Y, c=3.0):
    """Flag entries whose robust z-score (median, MAD) exceeds ``c``.

    Returns ``(Y_filtered, flags)``; flagged entries are set to NaN.
    """
    Y = np.asarray(Y, dtype=float)
    med = np.nanmedian(Y, axis=0)
    mad = np.nanmedian(np.abs(Y - med), axis=0)
    scale = np.maximum(1.4826 * mad, 1e-8)
    flagged = (np.abs(Y - med) / scale > c) & ~np.isnan(Y)
    Yf = Y.copy()
    Yf[flagged] = np.nan
    return Yf, flagged


def debiased_covariance(Y, psd_project=True, eps=1e-8):
    """Inverse-probability debiased covariance, no imputation.

    Returns ``(mu, Sigma, info)``; ``info["psd_shift"]`` is the relative eigenvalue
    mass clipped to make Sigma positive definite (0 if none).
    """
    Y = np.asarray(Y, dtype=float)
    n, p = Y.shape
    M = (~np.isnan(Y)).astype(float)
    Yz = np.nan_to_num(Y, nan=0.0) * M

    cnt = M.sum(axis=0)
    mu = np.divide(Yz.sum(axis=0), np.maximum(cnt, 1.0))

    Xc = (Yz - mu[None, :]) * M
    num = Xc.T @ Xc
    delta = (M.T @ M) / n
    Sigma = num / (n * np.maximum(delta, 1.0 / n))
    Sigma = 0.5 * (Sigma + Sigma.T)

    info = {"psd_shift": 0.0}
    if psd_project:
        w, V = np.linalg.eigh(Sigma)
        floor = max(eps, 1e-6 * max(w.max(), 1.0))
        neg = w < floor
        if neg.any():
            info["psd_shift"] = float(np.abs(w[neg]).sum()
                                      / max(np.abs(w).sum(), 1e-12))
            info["n_clipped"] = int(neg.sum())
        Sigma = (V * np.maximum(w, floor)) @ V.T
        Sigma = 0.5 * (Sigma + Sigma.T)

    return mu, Sigma, info


def estimate_filter_debiased(Y, seed=0, c=3.0):
    """MAD filter followed by the debiased covariance."""
    Yf, flags = entrywise_filter(Y, c=c)
    mu, Sigma, info = debiased_covariance(Yf)
    info["flagged_frac"] = float(flags.mean())
    info["filter_c"] = c
    return mu, Sigma, info


def estimate_debiased_only(Y, seed=0):
    """Debiased covariance without any filtering."""
    return debiased_covariance(Y)
