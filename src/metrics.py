"""Evaluation metrics.

Fisher-Rao distance. With lambda_1..lambda_p the eigenvalues of
Sigma_true^{-1} Sigma_hat and l_i = log lambda_i,

    d_FR^2 = sum_i l_i^2
           = p (log c*)^2 + sum_i (l_i - log c*)^2
           = d_scale^2 + d_shape^2,

where c* = exp(mean l_i) = (det Sigma_hat / det Sigma_true)^(1/p) is the scalar
that brings Sigma_hat closest to Sigma_true (c* = 1 for a correctly scaled
estimate). The two terms are orthogonal: d_scale measures the scale error and
d_shape = d_FR(Sigma_true, Sigma_hat / c*) the error that no rescaling removes.

``cov_y_from_sigma`` gives the covariance of y under the Student-t* model as a
function of Sigma and nu (finite for nu > 2).
"""

import numpy as np
from scipy.linalg import eigvalsh, norm
from scipy.special import gammaln

__all__ = [
    "fisher_rao",
    "fisher_decomposition",
    "relative_frobenius",
    "evaluate",
    "c_nu",
    "kappa_nu",
    "cov_y_from_sigma",
]


def _generalized_logeigs(Sigma_true, Sigma_est, floor=1e-15):
    """Log-eigenvalues of Sigma_true^{-1} Sigma_est (symmetric generalized problem)."""
    ev = eigvalsh(Sigma_est, Sigma_true)
    return np.log(np.clip(ev, floor, None))


def fisher_rao(Sigma_true, Sigma_est):
    """Total affine-invariant Riemannian (Fisher-Rao) distance."""
    try:
        l = _generalized_logeigs(Sigma_true, Sigma_est)
    except Exception:
        return np.inf
    return float(np.sqrt(np.sum(l ** 2)))


def fisher_decomposition(Sigma_true, Sigma_est):
    """Orthogonal scale / shape split of the Fisher-Rao distance.

    Returns a dict with ``fisher`` (total), ``fisher_scale`` = sqrt(p)|log c*|,
    ``fisher_shape``, ``cstar`` and ``log_cstar``, such that
    fisher^2 = fisher_scale^2 + fisher_shape^2.
    """
    p = Sigma_true.shape[0]
    try:
        l = _generalized_logeigs(Sigma_true, Sigma_est)
    except Exception:
        return dict(fisher=np.inf, fisher_scale=np.inf, fisher_shape=np.inf,
                    cstar=np.nan, log_cstar=np.nan)

    lbar = float(l.mean())
    return dict(
        fisher=float(np.sqrt(np.sum(l ** 2))),
        fisher_scale=float(np.sqrt(p) * abs(lbar)),
        fisher_shape=float(np.sqrt(np.sum((l - lbar) ** 2))),
        cstar=float(np.exp(lbar)),
        log_cstar=lbar,
    )


def relative_frobenius(Sigma_true, Sigma_est, calibrate=False):
    """||Sigma_true - Sigma_est||_F / ||Sigma_true||_F.

    With ``calibrate=True``, Sigma_est is first divided by c*.
    """
    nt = norm(Sigma_true, "fro")
    S = Sigma_est
    if calibrate:
        d = fisher_decomposition(Sigma_true, Sigma_est)
        c = d["cstar"]
        if not np.isfinite(c) or c <= 0:
            return np.nan
        S = Sigma_est / c
    return float(norm(Sigma_true - S, "fro") / nt) if nt > 0 else np.nan


def evaluate(mu_true, Sigma_true, mu_est, Sigma_est):
    """All metrics for one fit, as a flat dict."""
    out = {"err_mu_l2": float(norm(mu_true - mu_est))}
    out.update(fisher_decomposition(Sigma_true, Sigma_est))
    out["fro"] = relative_frobenius(Sigma_true, Sigma_est)
    out["fro_calibrated"] = relative_frobenius(Sigma_true, Sigma_est,
                                               calibrate=True)
    out["trace_ratio"] = float(np.trace(Sigma_est) / np.trace(Sigma_true))
    return out


def c_nu(nu):
    """E[tau^{-1/2}] for tau ~ Gamma(nu/2, rate nu/2), nu > 1.

        c(nu) = sqrt(nu/2) Gamma((nu-1)/2) / Gamma(nu/2)
    """
    nu = np.asarray(nu, dtype=float)
    if np.any(nu <= 1):
        raise ValueError("c(nu) requires nu > 1")
    return np.sqrt(nu / 2.0) * np.exp(gammaln((nu - 1) / 2.0)
                                      - gammaln(nu / 2.0))


def kappa_nu(nu):
    """Correlation shrinkage factor, Corr(y) = kappa(nu) Corr(Sigma), nu > 2.

        kappa(nu) = c(nu)^2 (nu-2) / nu
    """
    nu = np.asarray(nu, dtype=float)
    if np.any(nu <= 2):
        raise ValueError("kappa(nu) requires nu > 2 (Cov(y) is infinite otherwise)")
    return c_nu(nu) ** 2 * (nu - 2.0) / nu


def cov_y_from_sigma(Sigma, nu):
    """Covariance of y under y_ij = mu_j + x_ij / sqrt(tau_ij), x ~ N(0, Sigma).

        Cov(y) = c(nu)^2 Sigma + (nu/(nu-2) - c(nu)^2) diag(Sigma),

    which is finite only for nu > 2.
    """
    if nu <= 2:
        raise ValueError(
            f"Cov(y) is infinite for nu = {nu} <= 2; the fitted t* model has "
            "no finite covariance. Report a dispersion matrix instead.")
    c2 = float(c_nu(nu)) ** 2
    M = c2 * np.asarray(Sigma, dtype=float).copy()
    np.fill_diagonal(M, np.diag(Sigma) * nu / (nu - 2.0))
    return M


def sigma_from_cov_y(Cov_y, nu):
    """Inverse of ``cov_y_from_sigma``."""
    if nu <= 2:
        raise ValueError("requires nu > 2")
    c2 = float(c_nu(nu)) ** 2
    S = np.asarray(Cov_y, dtype=float) / c2
    np.fill_diagonal(S, np.diag(Cov_y) * (nu - 2.0) / nu)
    return S
