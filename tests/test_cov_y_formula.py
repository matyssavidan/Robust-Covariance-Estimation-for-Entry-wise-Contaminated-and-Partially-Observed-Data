"""Relation between the model parameter Sigma and Cov(y) under the Student-t* model.

Since tau_i is independent of x_i and the tau_ij are mutually independent,

    Cov(y) = c(nu)^2 Sigma + (nu/(nu-2) - c(nu)^2) diag(Sigma),
    c(nu)  = sqrt(nu/2) Gamma((nu-1)/2) / Gamma(nu/2),
    Corr(y) = kappa(nu) Corr(Sigma),  kappa(nu) = c(nu)^2 (nu-2)/nu.

Checks: exact round trip Sigma -> Cov(y) -> Sigma, properties of kappa, the
guard nu <= 2 (Cov(y) infinite), the structure of the diagonal and off-diagonal
scalings, and agreement with Monte Carlo simulation.
"""

import numpy as np
import pytest

from data_gen import random_covariance
from metrics import c_nu, kappa_nu, cov_y_from_sigma, sigma_from_cov_y


def test_roundtrip_is_exact():
    """Sigma -> Cov(y) -> Sigma recovers Sigma to machine precision."""
    Sigma = random_covariance(p=6, seed=1)
    for nu in (2.5, 3.0, 5.0, 10.0, 50.0):
        back = sigma_from_cov_y(cov_y_from_sigma(Sigma, nu), nu)
        assert np.allclose(back, Sigma, rtol=1e-12, atol=1e-12), f"nu={nu}"


def test_kappa_properties():
    """kappa(nu) lies in (0,1), increases with nu, and tends to 1."""
    nus = np.array([2.5, 3, 4, 6, 10, 20, 50, 100, 500])
    k = np.array([kappa_nu(v) for v in nus])

    assert np.all(k > 0) and np.all(k < 1)
    assert np.all(np.diff(k) > 0), "kappa should increase with nu"
    assert kappa_nu(500.0) > 0.99, "kappa should approach the Gaussian limit"


def test_cov_y_refused_below_two():
    """Cov(y) is infinite for nu <= 2 and must be refused."""
    Sigma = random_covariance(p=4, seed=2)
    for bad_nu in (0.5, 1.5, 2.0):
        with pytest.raises(ValueError):
            cov_y_from_sigma(Sigma, bad_nu)
    with pytest.raises(ValueError):
        kappa_nu(2.0)


def test_diagonal_and_offdiagonal_scaling():
    """The diagonal is scaled by nu/(nu-2), the off-diagonal by c(nu)^2."""
    Sigma = random_covariance(p=5, seed=3)
    nu = 8.0
    C = cov_y_from_sigma(Sigma, nu)

    assert np.allclose(np.diag(C), np.diag(Sigma) * nu / (nu - 2))
    off = ~np.eye(5, dtype=bool)
    assert np.allclose(C[off], Sigma[off] * c_nu(nu) ** 2)
    assert nu / (nu - 2) > c_nu(nu) ** 2, "diagonal inflates more than off-diagonal"


def _empirical_cov_chunked(Sigma, nu, n_total, chunk, seed):
    """Streaming empirical covariance of y, with bounded memory."""
    p = Sigma.shape[0]
    rng = np.random.default_rng(seed)
    s = np.zeros(p)
    S = np.zeros((p, p))
    done = 0
    while done < n_total:
        m = min(chunk, n_total - done)
        x = rng.multivariate_normal(np.zeros(p), Sigma, size=m)
        tau = rng.gamma(nu / 2.0, 2.0 / nu, size=(m, p))
        y = x / np.sqrt(tau)
        s += y.sum(axis=0)
        S += y.T @ y
        done += m
    mean = s / n_total
    return S / n_total - np.outer(mean, mean)


@pytest.mark.parametrize("nu", [6.0, 10.0])
def test_monte_carlo_agreement(nu):
    """Simulated Cov(y) matches the closed form.

    Tolerances are loose because y has heavy tails; nu <= 4 is avoided since Cov(y)
    then has infinite variance.
    """
    p = 4
    Sigma = random_covariance(p=p, seed=11)
    emp = _empirical_cov_chunked(Sigma, nu, n_total=2_000_000,
                                 chunk=250_000, seed=99)
    th = cov_y_from_sigma(Sigma, nu)

    d_err = np.abs(np.diag(emp) - np.diag(th)) / np.abs(np.diag(th))
    off = ~np.eye(p, dtype=bool)
    scale = np.abs(th[off]).max()
    o_err = np.abs(emp[off] - th[off]) / scale

    print(f"\n  nu={nu}: max rel. err diagonal    = {d_err.max():.4f}")
    print(f"  nu={nu}: max scaled err off-diag  = {o_err.max():.4f}")

    assert d_err.max() < 0.05, f"diagonal off by {d_err.max():.3f}"
    assert o_err.max() < 0.05, f"off-diagonal off by {o_err.max():.3f}"
