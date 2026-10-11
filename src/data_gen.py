"""Synthetic data generation.

``sigma_seed`` and ``data_seed`` are separate arguments: fixing ``sigma_seed``
gives one ground-truth covariance everywhere, and setting it to the replication
index gives common random numbers (the same covariance and the same clean data
for every configuration, only the contamination and missingness masks change).

The contamination mechanism is the independent contamination model of Alqallaf
et al. (2009): each entry is independently replaced by a draw from an outlier
distribution.

References
----------
Alqallaf, Van Aelst, Yohai & Zamar, Ann. Statist. 37(1):311-331, 2009.
Finegold & Drton, Ann. Appl. Stat. 5(2A):1057-1080, 2011.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

__all__ = [
    "Dataset",
    "random_covariance",
    "generate_gaussian",
    "generate_t_rowwise",
    "generate_t_entrywise",
    "generate_icm",
]


@dataclass
class Dataset:
    """Generated data together with the ground truth.

    Attributes
    ----------
    Y : (n, p) observed matrix, np.nan on missing entries.
    X_clean : (n, p) uncontaminated, fully observed data.
    mu_true, Sigma_true : ground-truth parameters. For ``generate_icm`` they are the
        parameters of the clean Gaussian data.
    mask_obs : (n, p) bool, True where observed.
    mask_outliers : (n, p) bool, True where the entry was replaced.
    meta : generator settings.
    """
    Y: np.ndarray
    X_clean: np.ndarray
    mu_true: np.ndarray
    Sigma_true: np.ndarray
    mask_obs: np.ndarray
    mask_outliers: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def shape(self):
        return self.Y.shape

    @property
    def missing_frac(self):
        return float((~self.mask_obs).mean())

    @property
    def outlier_frac(self):
        return float(self.mask_outliers.mean())


def random_covariance(p=20, seed=None, var_range=(1.0, 3.0), corr_strength=0.5):
    """Random positive-definite covariance, a function of ``seed`` only."""
    rng = np.random.default_rng(seed)
    variances = rng.uniform(*var_range, size=p)

    A = rng.normal(size=(p, p))
    Corr = A @ A.T
    d = np.sqrt(np.diag(Corr))
    Corr /= np.outer(d, d)
    Corr = (1.0 - corr_strength) * np.eye(p) + corr_strength * Corr

    s = np.sqrt(variances)
    Sigma = np.outer(s, s) * Corr
    return 0.5 * (Sigma + Sigma.T)


def _apply_missing(Y, missing_rate, rng):
    """MCAR missingness. Returns (Y_with_nan, mask_obs)."""
    if missing_rate <= 0:
        return Y, np.ones(Y.shape, dtype=bool)
    miss = rng.random(Y.shape) < missing_rate
    Y = Y.copy()
    Y[miss] = np.nan
    return Y, ~miss


def generate_gaussian(n=500, p=20, sigma_seed=0, data_seed=0, missing_rate=0.0):
    """Clean Gaussian data; Sigma_true is the covariance of x."""
    Sigma = random_covariance(p, seed=sigma_seed)
    mu = np.zeros(p)
    rng = np.random.default_rng(data_seed)

    X = rng.multivariate_normal(mu, Sigma, size=n)
    Y, mask_obs = _apply_missing(X, missing_rate, rng)

    return Dataset(Y, X, mu, Sigma, mask_obs,
                   np.zeros((n, p), dtype=bool),
                   dict(kind="gaussian", n=n, p=p, sigma_seed=sigma_seed,
                        data_seed=data_seed, missing_rate=missing_rate))


def generate_t_rowwise(n=500, p=20, nu=3.0, sigma_seed=0, data_seed=0,
                       missing_rate=0.0):
    """Classical multivariate Student-t: one scale per observation.

        x_i ~ N(0, Sigma),  tau_i ~ Gamma(nu/2, nu/2),  y_i = mu + x_i / sqrt(tau_i)
    """
    Sigma = random_covariance(p, seed=sigma_seed)
    mu = np.zeros(p)
    rng = np.random.default_rng(data_seed)

    X = rng.multivariate_normal(mu, Sigma, size=n)
    tau = rng.gamma(nu / 2.0, 2.0 / nu, size=n)[:, None]
    Yc = mu + X / np.sqrt(tau)
    Y, mask_obs = _apply_missing(Yc, missing_rate, rng)

    return Dataset(Y, Yc, mu, Sigma, mask_obs,
                   np.zeros((n, p), dtype=bool),
                   dict(kind="t_rowwise", n=n, p=p, nu=nu,
                        sigma_seed=sigma_seed, data_seed=data_seed,
                        missing_rate=missing_rate))


def generate_t_entrywise(n=500, p=20, nu=3.0, sigma_seed=0, data_seed=0,
                        missing_rate=0.0):
    """Student-t* of Finegold & Drton: one scale per entry.

        x_i ~ N(0, Sigma),  tau_ij ~ Gamma(nu/2, nu/2) independent,
        y_ij = mu_j + x_ij / sqrt(tau_ij)

    Sigma_true is the parameter estimated by ``EntrywiseSAEM``. The covariance of y
    is given by ``metrics.cov_y_from_sigma`` (finite for nu > 2).
    """
    Sigma = random_covariance(p, seed=sigma_seed)
    mu = np.zeros(p)
    rng = np.random.default_rng(data_seed)

    X = rng.multivariate_normal(mu, Sigma, size=n)
    tau = rng.gamma(nu / 2.0, 2.0 / nu, size=(n, p))
    Yc = mu + X / np.sqrt(tau)
    Y, mask_obs = _apply_missing(Yc, missing_rate, rng)

    return Dataset(Y, Yc, mu, Sigma, mask_obs,
                   np.zeros((n, p), dtype=bool),
                   dict(kind="t_entrywise", n=n, p=p, nu=nu,
                        sigma_seed=sigma_seed, data_seed=data_seed,
                        missing_rate=missing_rate))


def generate_icm(n=500, p=20, sigma_seed=0, data_seed=0,
                 missing_rate=0.0, outlier_rate=0.0, scale=10.0,
                 mode="amplified", shift=10.0, bernoulli=True):
    """Independent contamination model on clean Gaussian data.

    Each entry is independently replaced with probability ``outlier_rate``:

        y_ij = (1 - b_ij) x_ij + b_ij z_ij,   b_ij ~ Bernoulli(outlier_rate).

    ``Sigma_true`` is the covariance of the clean data X.

    Parameters
    ----------
    scale : float
        Magnitude of the contamination in units of the marginal standard
        deviation sigma_j.
    mode : {"amplified", "shift"}
        "amplified": z_ij ~ N(mu_j, (scale * sigma_j)^2).
        "shift": z_ij = mu_j +/- scale * sigma_j + N(0, (sigma_j / 2)^2).
    bernoulli : bool
        True: independent Bernoulli draws. False: exactly
        floor(outlier_rate * n * p) entries drawn without replacement.

    Contamination is applied first and missingness afterwards, so a contaminated
    entry can also be missing.
    """
    if mode not in ("amplified", "shift"):
        raise ValueError(f"unknown mode: {mode!r}")

    Sigma = random_covariance(p, seed=sigma_seed)
    mu = np.zeros(p)
    rng = np.random.default_rng(data_seed)

    X = rng.multivariate_normal(mu, Sigma, size=n)
    Y = X.copy()
    sigma_j = np.sqrt(np.diag(Sigma))
    mask_out = np.zeros((n, p), dtype=bool)

    if outlier_rate > 0:
        if bernoulli:
            mask_out = rng.random((n, p)) < outlier_rate
        else:
            m = int(outlier_rate * n * p)
            flat = rng.choice(n * p, size=m, replace=False)
            mask_out.ravel()[flat] = True

        k = int(mask_out.sum())
        if k:
            cols = np.where(mask_out)[1]
            if mode == "amplified":
                Y[mask_out] = mu[cols] + rng.normal(0.0, scale * sigma_j[cols])
            else:
                signs = rng.choice([-1.0, 1.0], size=k)
                Y[mask_out] = (mu[cols] + signs * scale * sigma_j[cols]
                            + rng.normal(0.0, 0.5 * sigma_j[cols]))

    Y, mask_obs = _apply_missing(Y, missing_rate, rng)

    return Dataset(Y, X, mu, Sigma, mask_obs, mask_out,
                   dict(kind="icm", n=n, p=p, sigma_seed=sigma_seed,
                        data_seed=data_seed, missing_rate=missing_rate,
                        outlier_rate=outlier_rate, scale=scale, mode=mode,
                        shift=shift, bernoulli=bernoulli))
