"""Rao-Blackwellised moments with complete and incomplete data.

With every coordinate observed, O_i = {1..p} and

    C_i = Sigma - Sigma_{:,O} Sigma_{OO}^{-1} Sigma_{O,:} = 0,
    m_i = D(tau_i)^{1/2}(y_i - mu) = x_i,

so the Rao-Blackwellisation is void and the method reduces to a SAEM for the
same model. With missing entries C_i is non-zero, supported on the missing block,
and its trace measures the posterior uncertainty about the missing coordinates.
"""

import numpy as np
import pytest

from data_gen import generate_t_entrywise


def _moments(Sigma, mu, tau_i, y_i, mask_i):
    """Moments m_i and C_i written directly from their definition, independently
    of the optimised implementation in saem.py.
    """
    p = len(mu)
    obs = np.where(mask_i)[0]
    Soo = Sigma[np.ix_(obs, obs)]
    S_all_o = Sigma[:, obs]

    K = np.linalg.solve(Soo, S_all_o.T).T
    C_i = Sigma - K @ S_all_o.T
    m_i = K @ (np.sqrt(tau_i[obs]) * (y_i[obs] - mu[obs]))
    return m_i, 0.5 * (C_i + C_i.T), p


def test_conditional_covariance_vanishes_without_missing():
    """C_i = 0 when every coordinate is observed."""
    ds = generate_t_entrywise(n=30, p=8, nu=5.0, sigma_seed=0, data_seed=0,
                             missing_rate=0.0)
    rng = np.random.default_rng(0)
    tau = rng.gamma(2.5, 0.4, size=ds.Y.shape)
    full = np.ones(8, dtype=bool)

    worst = 0.0
    for i in range(ds.Y.shape[0]):
        _, C_i, _ = _moments(ds.Sigma_true, ds.mu_true, tau[i], ds.Y[i], full)
        worst = max(worst, np.abs(C_i).max())

    print(f"\n  complete data: max |C_i| = {worst:.3e}")
    assert worst < 1e-8, "C_i should vanish exactly with complete data"


def test_conditional_mean_equals_latent_x():
    """m_i = D(tau_i)^{1/2}(y_i - mu) = x_i, exactly, with complete data."""
    ds = generate_t_entrywise(n=30, p=8, nu=5.0, sigma_seed=1, data_seed=1,
                             missing_rate=0.0)
    rng = np.random.default_rng(1)
    tau = rng.gamma(2.5, 0.4, size=ds.Y.shape)
    full = np.ones(8, dtype=bool)

    worst = 0.0
    for i in range(ds.Y.shape[0]):
        m_i, _, _ = _moments(ds.Sigma_true, ds.mu_true, tau[i], ds.Y[i], full)
        x_i = np.sqrt(tau[i]) * (ds.Y[i] - ds.mu_true)
        worst = max(worst, np.abs(m_i - x_i).max())

    print(f"  complete data: max |m_i - x_i| = {worst:.3e}")
    assert worst < 1e-8


@pytest.mark.parametrize("missing_rate", [0.10, 0.30])
def test_conditional_covariance_is_informative_with_missing(missing_rate):
    """With missing entries C_i is non-zero, supported exactly on the missing
    block, and PSD. Its trace is the posterior uncertainty that single
    imputation discards -- the quantity our method propagates and
    impute-then-fit does not."""
    p = 8
    ds = generate_t_entrywise(n=60, p=p, nu=5.0, sigma_seed=2, data_seed=2,
                             missing_rate=missing_rate)
    rng = np.random.default_rng(2)
    tau = rng.gamma(2.5, 0.4, size=ds.Y.shape)

    traces, seen_missing = [], False
    for i in range(ds.Y.shape[0]):
        mask = ds.mask_obs[i]
        if mask.all() or not mask.any():
            continue
        seen_missing = True
        _, C_i, _ = _moments(ds.Sigma_true, ds.mu_true, tau[i],
                             np.nan_to_num(ds.Y[i]), mask)

        obs = np.where(mask)[0]

        assert np.abs(C_i[obs, :]).max() < 1e-8
        assert np.abs(C_i[:, obs]).max() < 1e-8

        mis = np.where(~mask)[0]
        w = np.linalg.eigvalsh(C_i[np.ix_(mis, mis)])
        assert w.min() > -1e-8, "C_i must be PSD"
        traces.append(float(np.trace(C_i)))

    assert seen_missing, "no partially observed row was generated"
    print(f"  missing={missing_rate}: mean trace(C_i) = {np.mean(traces):.4f} "
          f"(0.0 with complete data)")
    assert np.mean(traces) > 1e-6
