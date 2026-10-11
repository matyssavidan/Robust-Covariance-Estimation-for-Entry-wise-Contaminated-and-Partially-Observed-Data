"""Gaussian EM with missing data (non-robust reference).

The E-step replaces the missing block by its conditional mean and adds the
conditional covariance to the second-moment accumulator:

    E[x_i | y_iO]   = mu_M + Sigma_MO Sigma_OO^{-1} (y_O - mu_O)
    Var[x_i | y_iO] = Sigma_MM - Sigma_MO Sigma_OO^{-1} Sigma_OM

Rows are grouped by missingness pattern so that Sigma_OO is factorised once per
distinct pattern.
"""

import numpy as np
from scipy import linalg

__all__ = ["GaussianEM", "estimate"]


class GaussianEM:

    def __init__(self, max_iter=100, tol=1e-5, reg=1e-6):
        self.max_iter = max_iter
        self.tol = tol
        self.reg = reg
        self.mu_ = None
        self.Sigma_ = None
        self.n_iter_ = 0
        self.converged_ = False

    def fit(self, Y):
        Y = np.asarray(Y, dtype=float)
        n, p = Y.shape
        mask = ~np.isnan(Y)
        Yz = np.nan_to_num(Y, nan=0.0)
        eye = np.eye(p)

        groups = {}
        for i, row in enumerate(mask):
            groups.setdefault(row.tobytes(), []).append(i)
        groups = [(np.asarray(idx), mask[idx[0]]) for idx in groups.values()]

        mu = np.nanmean(Y, axis=0)
        mu = np.where(np.isfinite(mu), mu, 0.0)
        Sigma = np.diag(np.nanvar(Y, axis=0) + self.reg)

        for it in range(self.max_iter):
            Ex = np.zeros((n, p))
            Exx = np.zeros((p, p))

            for idx, m in groups:
                obs = np.where(m)[0]
                mis = np.where(~m)[0]
                Xi = np.tile(mu, (idx.size, 1))

                if obs.size:
                    Xi[:, obs] = Yz[np.ix_(idx, obs)]
                if mis.size and obs.size:
                    Soo = Sigma[np.ix_(obs, obs)] + self.reg * np.eye(obs.size)
                    Smo = Sigma[np.ix_(mis, obs)]
                    K = linalg.solve(Soo, Smo.T, assume_a="pos").T
                    resid = Yz[np.ix_(idx, obs)] - mu[obs]
                    Xi[:, mis] = mu[mis] + resid @ K.T
                    Cmm = Sigma[np.ix_(mis, mis)] - K @ Sigma[np.ix_(obs, mis)]
                    Exx[np.ix_(mis, mis)] += idx.size * 0.5 * (Cmm + Cmm.T)
                elif mis.size:
                    Exx[np.ix_(mis, mis)] += idx.size * Sigma[np.ix_(mis, mis)]

                Ex[idx] = Xi
                Exx += Xi.T @ Xi

            mu_new = Ex.mean(axis=0)
            Sigma_new = Exx / n - np.outer(mu_new, mu_new)
            Sigma_new = 0.5 * (Sigma_new + Sigma_new.T) + self.reg * eye

            delta = (np.linalg.norm(mu_new - mu)
                     + np.linalg.norm(Sigma_new - Sigma, "fro"))
            mu, Sigma = mu_new, Sigma_new
            self.n_iter_ = it + 1
            if delta < self.tol:
                self.converged_ = True
                break

        self.mu_, self.Sigma_ = mu, Sigma
        return self

    def get_params(self):
        return self.mu_, self.Sigma_


def estimate(Y, seed=0, max_iter=100, **kw):
    m = GaussianEM(max_iter=max_iter, **kw).fit(Y)
    return m.mu_, m.Sigma_, dict(n_iter=m.n_iter_, converged=m.converged_)
