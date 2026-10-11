"""Classical multivariate Student-t EM with missing data.

Model: x_i ~ N(0, Sigma), tau_i ~ Gamma(nu/2, nu/2), y_i = mu + x_i / sqrt(tau_i),
with one scale per observation. The E-step is available in closed form:

    tau_i | y_iO ~ Gamma((nu + d_i)/2, (nu + delta_i^2)/2)
    E[tau_i | y_iO]     = (nu + d_i) / (nu + delta_i^2)
    E[log tau_i | y_iO] = psi((nu + d_i)/2) - log((nu + delta_i^2)/2)

where d_i = |O_i| and delta_i^2 is the Mahalanobis distance on the observed
block. The weight of an observation multiplies all its coordinates.
"""

import numpy as np
from scipy.special import digamma
from scipy.optimize import brentq

__all__ = ["StudentEM", "estimate"]


class StudentEM:

    def __init__(self, nu=10.0, max_iter=100, tol=1e-5, reg=1e-6,
                 estimate_nu=True):
        self.nu = float(nu)
        self.max_iter = max_iter
        self.tol = tol
        self.reg = reg
        self.estimate_nu = estimate_nu
        self.mu = None
        self.Sigma = None
        self.weights_ = None
        self.n_iter_ = 0
        self.converged_ = False

    def _solve_nu(self, mean_tau, mean_log_tau):
        S = 1.0 + mean_log_tau - mean_tau

        def f(v):
            return np.log(v / 2) - digamma(v / 2) + S
        lo, hi = 0.1, 200.0
        try:
            flo, fhi = f(lo), f(hi)
            if not (np.isfinite(flo) and np.isfinite(fhi)):
                return self.nu
            if flo * fhi > 0:
                return lo if abs(flo) < abs(fhi) else hi
            return float(brentq(f, lo, hi))
        except (ValueError, RuntimeError):
            return self.nu

    def fit(self, Y):
        Y = np.asarray(Y, dtype=float)
        n, p = Y.shape
        eye = np.eye(p)


        self.mu = np.nanmedian(Y, axis=0)
        mad = np.nanmedian(np.abs(Y - self.mu), axis=0)
        self.Sigma = np.diag(np.maximum(1.4826 * mad, 1e-4) ** 2) + self.reg * eye

        for it in range(self.max_iter):
            prev_mu, prev_Sigma, prev_nu = self.mu.copy(), self.Sigma.copy(), self.nu

            W = np.zeros(n)
            Elog = np.zeros(n)
            Xhat = np.zeros((n, p))
            Corr = np.zeros((p, p))

            for i in range(n):
                m = ~np.isnan(Y[i])
                obs, mis = np.where(m)[0], np.where(~m)[0]

                if obs.size == 0:
                    W[i] = 1.0
                    Xhat[i] = self.mu
                    Corr += self.Sigma
                    continue

                Soo = self.Sigma[np.ix_(obs, obs)] + self.reg * np.eye(obs.size)
                d = Y[i, obs] - self.mu[obs]
                try:
                    L = np.linalg.cholesky(Soo)
                    z = np.linalg.solve(L, d)
                    delta2 = float(z @ z)
                    Li = np.linalg.solve(L, np.eye(obs.size))
                    Soo_inv = Li.T @ Li
                except np.linalg.LinAlgError:
                    delta2, Soo_inv = 0.0, np.eye(obs.size)

                a = (self.nu + obs.size) / 2.0
                b = (self.nu + delta2) / 2.0
                W[i] = a / b
                Elog[i] = digamma(a) - np.log(b)

                row = np.empty(p)
                row[obs] = Y[i, obs]
                if mis.size:
                    Smo = self.Sigma[np.ix_(mis, obs)]
                    row[mis] = self.mu[mis] + Smo @ Soo_inv @ d
                    Corr[np.ix_(mis, mis)] += (
                        self.Sigma[np.ix_(mis, mis)]
                        - Smo @ Soo_inv @ self.Sigma[np.ix_(obs, mis)])
                Xhat[i] = row

            Sw = W.sum()
            mu_new = (W[:, None] * Xhat).sum(axis=0) / Sw
            Xc = Xhat - mu_new
            Sigma_new = ((Xc.T * W) @ Xc + Corr) / n
            Sigma_new = 0.5 * (Sigma_new + Sigma_new.T) + self.reg * eye

            nu_new = (self._solve_nu(W.mean(), Elog.mean())
                      if self.estimate_nu else self.nu)

            delta = (np.linalg.norm(mu_new - prev_mu)
                     + np.linalg.norm(Sigma_new - prev_Sigma, "fro")
                     + abs(nu_new - prev_nu))

            self.mu, self.Sigma, self.nu = mu_new, Sigma_new, nu_new
            self.weights_ = W
            self.n_iter_ = it + 1
            if delta < self.tol:
                self.converged_ = True
                break

        return self

    def get_params(self):
        return self.mu, self.Sigma, self.nu


def estimate(Y, seed=0, nu=10.0, max_iter=100, **kw):
    m = StudentEM(nu=nu, max_iter=max_iter, **kw).fit(Y)
    return m.mu, m.Sigma, dict(n_iter=m.n_iter_, nu_hat=m.nu,
                               converged=m.converged_)
