"""Stochastic Approximation EM for the Student-t* model with missing data.

Model (Finegold & Drton, 2011), for i = 1..n and j = 1..p:

    x_i ~ N(0, Sigma),   tau_ij ~ Gamma(nu/2, nu/2) independent,
    y_ij = mu_j + x_ij / sqrt(tau_ij),

with an arbitrary subset O_i of coordinates observed. Given tau_i the model is
Gaussian, so the missing entries are integrated out analytically
(Rao-Blackwellisation, moments m_i and C_i); the entry-wise scales tau_ij are
simulated by Metropolis-Hastings.

The M-step is an ECM step (Meng & Rubin, 1993): CM1 updates mu given Sigma, the
moments are recomputed at the new mu, then CM2 updates Sigma.

Because the quadratic form q_i = v_i' A_i v_i, with A_i = Sigma_{O_i O_i}^{-1}
and v_i = D(tau_i)^{1/2}(y_i - mu), is quadratic in each entry v_ij, a
Metropolis-Hastings proposal is evaluated in O(1) and an accepted move updates
A_i v_i in O(|O_i|). The cost per iteration is O(n p^3 + K n p^2), with K the
number of Metropolis-Hastings sweeps.

References
----------
Finegold & Drton, Ann. Appl. Stat. 5(2A):1057-1080, 2011.
Meng & Rubin, Biometrika 80(2):267-278, 1993.
Delyon, Lavielle & Moulines, Ann. Statist. 27(1):94-128, 1999.
"""

import numpy as np
from scipy import linalg
from scipy.special import digamma
from scipy.optimize import brentq

__all__ = ["EntrywiseSAEM"]


class EntrywiseSAEM:
    """Entry-wise SAEM for the Student-t* model with missing data.

    Parameters
    ----------
    nu : float
        Initial degrees of freedom (re-estimated unless ``estimate_nu=False``).
    max_iter : int
        Number of SAEM iterations.
    mcmc_steps : int
        Metropolis-Hastings sweeps over all observed entries per iteration.
    proposal_scale : float
        Standard deviation of the log-normal random-walk proposal (initial value
        when ``adapt_proposal=True``).
    m_step : {"ecm", "legacy"}
        "ecm": CM1 on mu, moments recomputed, CM2 on Sigma.
        "legacy": mu and Sigma both updated from moments evaluated at the current
        iterate (kept for comparison).
    adapt_proposal : bool
        Robbins-Monro adaptation of the proposal scale during burn-in; the scale is
        then frozen.
    target_acc : float
        Target acceptance rate of the adaptation.
    shrinkage : float in [0, 1)
        Linear shrinkage of Sigma towards its diagonal (0 disables it).
    estimate_nu : bool
        If False, nu stays at its initial value.
    burn_frac, sa_exponent : float
        Step size gamma_k = 1 for k < burn_frac * max_iter, then
        (k - burn + 1) ** (-sa_exponent).
    reg : float
        Ridge added when inverting Sigma, for numerical safety.
    seed : int or None
    track : bool
        Store per-iteration diagnostics in ``history_``.

    Attributes
    ----------
    mu, Sigma, nu : fitted parameters. ``Sigma`` is the covariance of the latent
        Gaussian vector; see ``metrics.cov_y_from_sigma`` for the covariance of y.
    acc_rate_ : overall Metropolis-Hastings acceptance rate.
    acc_rate_final_ : acceptance rate of the last iteration.
    proposal_scale_ : final proposal scale.
    """

    def __init__(self, nu=10.0, max_iter=30, mcmc_steps=20,
                 proposal_scale=1.0, m_step="ecm",
                 adapt_proposal=True, target_acc=0.40,
                 shrinkage=0.0, estimate_nu=True,
                 burn_frac=0.5, sa_exponent=0.7, reg=1e-6,
                 seed=None, track=False):
        if m_step not in ("ecm", "legacy"):
            raise ValueError(f"m_step must be 'ecm' or 'legacy', got {m_step!r}")
        if not 0.0 <= shrinkage < 1.0:
            raise ValueError("shrinkage must lie in [0, 1)")

        self.nu = float(nu)
        self.max_iter = max_iter
        self.mcmc_steps = mcmc_steps
        self.proposal_scale = float(proposal_scale)
        self.m_step = m_step
        self.adapt_proposal = adapt_proposal
        self.target_acc = target_acc
        self.shrinkage = shrinkage
        self.estimate_nu = estimate_nu
        self.burn_frac = burn_frac
        self.sa_exponent = sa_exponent
        self.reg = reg
        self.rng = np.random.default_rng(seed)
        self.track = track

        self.mu = None
        self.Sigma = None
        self.Tau = None
        self.n_iter_ = 0
        self.acc_rate_ = None
        self.acc_rate_final_ = None
        self.proposal_scale_ = None
        self.history_ = []


    def _pattern_groups(self, mask_obs):
        """Group rows sharing the same missingness pattern."""
        groups = {}
        for i, row in enumerate(mask_obs):
            groups.setdefault(row.tobytes(), []).append(i)
        return [(np.array(idx), mask_obs[idx[0]]) for idx in groups.values()]

    def _inv_blocks(self, groups, n, p):
        """A_i = Sigma_{O_i O_i}^{-1}, zero-padded on the missing coordinates."""
        A = np.zeros((n, p, p))
        for idx, mask in groups:
            obs = np.where(mask)[0]
            if obs.size == 0:
                continue
            Soo = self.Sigma[np.ix_(obs, obs)]
            try:
                Aoo = linalg.inv(Soo)
            except linalg.LinAlgError:
                Aoo = linalg.pinv(Soo)
            A[np.ix_(idx, obs, obs)] = Aoo
        return A

    def _moments(self, A, Xz, mask_obs):
        """v_i = D(tau_i)^{1/2}(y_i - mu) and A_i v_i at the current mu."""
        diff = (Xz - self.mu) * mask_obs
        v = np.sqrt(self.Tau) * diff * mask_obs
        Av = np.einsum("ijk,ik->ij", A, v)
        return diff, v, Av

    def _solve_mu(self, S_W, S_Wy):
        try:
            return linalg.solve(S_W, S_Wy, assume_a="pos")
        except (linalg.LinAlgError, ValueError):
            return linalg.lstsq(S_W, S_Wy)[0]

    def _solve_nu(self, s_tau, s_log_tau, N):
        """Solve log(nu/2) - psi(nu/2) + 1 + (S_logtau - S_tau)/N = 0 for nu.

        This is the stationarity condition of the Gamma(nu/2, nu/2) log-likelihood.
        Only observed entries contribute; N is their number.
        """
        if not self.estimate_nu:
            return self.nu
        S = 1.0 + s_log_tau / N - s_tau / N

        def f(x):
            return np.log(x / 2.0) - digamma(x / 2.0) + S

        lo, hi = 0.1, 100.0
        try:
            flo, fhi = f(lo), f(hi)
            if not np.isfinite(flo) or not np.isfinite(fhi):
                return self.nu
            if flo * fhi > 0:
                return lo if abs(flo) < abs(fhi) else hi
            return float(brentq(f, lo, hi))
        except (ValueError, RuntimeError):
            return self.nu


    def fit(self, Y):
        """Fit on an (n, p) array with np.nan marking missing entries."""
        Y = np.asarray(Y, dtype=float)
        n, p = Y.shape
        rng = self.rng

        mask_obs = ~np.isnan(Y)
        if not mask_obs.any():
            raise ValueError("no observed entries")
        N_obs = int(mask_obs.sum())
        Xz = np.nan_to_num(Y, nan=0.0)
        groups = self._pattern_groups(mask_obs)


        self.mu = np.nanmedian(Y, axis=0)
        self.Sigma = np.diag(np.nanvar(Y, axis=0) + 1e-3)
        self.Tau = np.ones((n, p))

        S_Sigma = np.zeros((p, p))
        S_W = np.zeros((p, p))
        S_Wy = np.zeros(p)
        S_tau = S_log_tau = 0.0

        n_burn = max(1, int(self.burn_frac * self.max_iter))
        acc_tot = prop_tot = 0
        eye = np.eye(p)

        for k in range(self.max_iter):
            gamma = 1.0 if k < n_burn else (k - n_burn + 1) ** (-self.sa_exponent)

            A = self._inv_blocks(groups, n, p)
            diff, v, Av = self._moments(A, Xz, mask_obs)
            Adiag = np.einsum("ijj->ij", A).copy()
            a_prior = self.nu / 2.0


            n_acc = n_prop = 0
            for _ in range(self.mcmc_steps):
                for j in range(p):
                    idx = np.where(mask_obs[:, j])[0]
                    if idx.size == 0:
                        continue

                    tau_c = self.Tau[idx, j]
                    eps = rng.normal(0.0, self.proposal_scale, size=idx.size)
                    tau_p = tau_c * np.exp(eps)

                    vj_c = v[idx, j]
                    vj_p = np.sqrt(tau_p) * diff[idx, j]

                    ajj = Adiag[idx, j]
                    s_off = Av[idx, j] - ajj * vj_c
                    dquad = (ajj * (vj_p ** 2 - vj_c ** 2)
                             + 2.0 * (vj_p - vj_c) * s_off)


                    log_ratio = ((a_prior - 1.0) * eps
                                 - a_prior * (tau_p - tau_c)
                                 + 0.5 * eps - 0.5 * dquad + eps)

                    acc = np.log(rng.random(idx.size)) < log_ratio
                    n_prop += idx.size
                    n_acc += int(acc.sum())

                    if acc.any():
                        ia = idx[acc]
                        delta = vj_p[acc] - vj_c[acc]
                        self.Tau[ia, j] = tau_p[acc]
                        v[ia, j] = vj_p[acc]
                        Av[ia] += A[ia, :, j] * delta[:, None]

            acc_rate = n_acc / max(n_prop, 1)
            acc_tot += n_acc
            prop_tot += n_prop


            if self.adapt_proposal and k < n_burn:
                rho = 1.0 / (k + 1) ** 0.6
                self.proposal_scale = float(np.clip(
                    self.proposal_scale * np.exp(rho * (acc_rate - self.target_acc)),
                    1e-3, 50.0))


            miss = ~mask_obs
            if miss.any():
                self.Tau[miss] = rng.gamma(self.nu / 2.0, 2.0 / self.nu,
                                           size=int(miss.sum()))


            T = np.sqrt(self.Tau)
            inv_Sigma = linalg.solve(self.Sigma + self.reg * eye, eye)
            curr_S_W = inv_Sigma * (T.T @ T)


            curr_S_Wy = curr_S_W @ self.mu + (T * Av).sum(axis=0)

            S_W = (1 - gamma) * S_W + gamma * curr_S_W
            S_Wy = (1 - gamma) * S_Wy + gamma * curr_S_Wy

            if self.m_step == "ecm":

                self.mu = self._solve_mu(S_W, S_Wy)

                diff, v, Av = self._moments(A, Xz, mask_obs)


            m = Av @ self.Sigma
            Abar = A.mean(axis=0)
            C_mean = self.Sigma - self.Sigma @ Abar @ self.Sigma
            curr_S_Sigma = 0.5 * (C_mean + C_mean.T) + (m.T @ m) / n

            S_Sigma = (1 - gamma) * S_Sigma + gamma * curr_S_Sigma

            obs_tau = self.Tau[mask_obs]
            curr_s_tau = float(obs_tau.sum())
            curr_s_log_tau = float(np.log(obs_tau + 1e-12).sum())
            if k == 0:
                S_tau, S_log_tau = curr_s_tau, curr_s_log_tau
            else:
                S_tau = (1 - gamma) * S_tau + gamma * curr_s_tau
                S_log_tau = (1 - gamma) * S_log_tau + gamma * curr_s_log_tau


            Sigma_new = 0.5 * (S_Sigma + S_Sigma.T)
            if self.shrinkage > 0.0:
                Sigma_new = ((1.0 - self.shrinkage) * Sigma_new
                             + self.shrinkage * np.diag(np.diag(Sigma_new)))
            self.Sigma = Sigma_new

            if self.m_step == "legacy":

                self.mu = self._solve_mu(S_W, S_Wy)

            self.nu = self._solve_nu(S_tau, S_log_tau, N_obs)
            self.n_iter_ = k + 1
            self.acc_rate_final_ = acc_rate

            if self.track:
                self.history_.append(dict(
                    k=k, gamma=gamma, nu=self.nu, acc=acc_rate,
                    proposal_scale=self.proposal_scale,
                    trace=float(np.trace(self.Sigma)),
                    logdet=float(np.linalg.slogdet(self.Sigma)[1])))

        self.acc_rate_ = acc_tot / max(prop_tot, 1)
        self.proposal_scale_ = self.proposal_scale
        return self

    def get_params(self):
        return self.mu, self.Sigma, self.nu
