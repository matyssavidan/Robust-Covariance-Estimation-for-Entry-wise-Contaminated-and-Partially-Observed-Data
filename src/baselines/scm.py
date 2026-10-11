"""Sample covariance and Tyler's M-estimator, applied after imputation.

Tyler's M-estimator is defined up to a positive scalar, so a normalisation
convention is required. ``tyler_covariance`` supports three:

trace      trace(Sigma) = p. A shape estimator: it does not estimate the scale.
calibrate  Sigma is rescaled so that the median Mahalanobis distance matches the
           chi-square_p median (Gaussian-consistent scale).
none       the fixed point as it converged.

The total Fisher-Rao distance depends on this convention; the shape component
does not.
"""

import numpy as np
from scipy.stats import chi2

__all__ = ["sample_covariance", "tyler_covariance", "estimate_mean_scm",
           "estimate_mice_tyler"]


def sample_covariance(X):
    """Sample mean and (biased) sample covariance."""
    X = np.asarray(X, dtype=float)
    return X.mean(axis=0), np.cov(X.T, bias=True)


def tyler_covariance(X, max_iter=100, tol=1e-6, reg=1e-6,
                     normalise="trace"):
    """Tyler's M-estimator of scatter, with median centring.

    Parameters
    ----------
    normalise : {"trace", "calibrate", "none"}
        Scale convention, see the module docstring.
    """
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    mu = np.median(X, axis=0)
    Xc = X - mu
    Sigma = np.eye(p)

    for _ in range(max_iter):
        Sinv = np.linalg.inv(Sigma)
        q = np.maximum(np.einsum("ij,jk,ik->i", Xc, Sinv, Xc), 1e-12)
        Snew = (p / n) * (Xc.T / q) @ Xc
        Snew /= np.trace(Snew) / p
        Snew += reg * np.eye(p)
        done = np.linalg.norm(Snew - Sigma) < tol
        Sigma = Snew
        if done:
            break

    if normalise == "trace":
        Sigma = Sigma / (np.trace(Sigma) / p)
    elif normalise == "calibrate":
        Sinv = np.linalg.inv(Sigma)
        d2 = np.einsum("ij,jk,ik->i", Xc, Sinv, Xc)
        med = np.median(d2)
        if np.isfinite(med) and med > 0:
            Sigma = Sigma * med / chi2.ppf(0.5, df=p)

    return mu, 0.5 * (Sigma + Sigma.T)


def estimate_mean_scm(Y, seed=0):
    """Mean imputation followed by the sample covariance."""
    from .imputation import mean_impute
    return (*sample_covariance(mean_impute(Y)), {})


def estimate_mice_tyler(Y, seed=0, normalise="trace"):
    """MICE imputation followed by Tyler's M-estimator."""
    from .imputation import mice_impute
    mu, S = tyler_covariance(mice_impute(Y, seed=seed), normalise=normalise)
    return mu, S, dict(tyler_normalise=normalise)
