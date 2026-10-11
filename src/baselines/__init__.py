"""Estimator registries.

Every estimator has the signature ``estimate(Y, seed=0, **kwargs)`` and returns
``(mu, Sigma, info)``, where ``Y`` is an (n, p) array with np.nan on missing
entries and ``info`` a dict of diagnostics.

ESTIMATORS  methods compared in the main experiments.
ABLATIONS   the same SAEM applied to the incomplete data, after mean
            imputation, and after MICE imputation.
DIAGNOSTIC  additional arms: debiasing without filtering, and MICE + Tyler with
            a Gaussian-consistent scale.
"""

import time

from saem import EntrywiseSAEM

from .em_gaussian import estimate as _em_gaussian
from .em_student import estimate as _em_student
from .scm import estimate_mean_scm as _mean_scm, estimate_mice_tyler as _mice_tyler
from .detection import (estimate_filter_debiased as _filter_debiased,
                        estimate_debiased_only as _debiased_only)
from .imputation import mean_impute, mice_impute

__all__ = ["ESTIMATORS", "ABLATIONS", "DIAGNOSTIC", "ALL", "timed",
           "saem_estimate"]

SAEM_DEFAULTS = dict(nu=10.0, max_iter=30, mcmc_steps=20, proposal_scale=1.0,
                     m_step="ecm", adapt_proposal=True, target_acc=0.40)


def saem_estimate(Y, seed=0, **over):
    kw = dict(SAEM_DEFAULTS); kw.update(over)
    m = EntrywiseSAEM(seed=seed, **kw).fit(Y)
    mu, Sigma, nu = m.get_params()
    return mu, Sigma, dict(n_iter=m.n_iter_, nu_hat=nu,
                           acc_rate=m.acc_rate_,
                           acc_rate_final=m.acc_rate_final_,
                           proposal_scale=m.proposal_scale_)


def _meanimp_then_saem(Y, seed=0, **over):
    return saem_estimate(mean_impute(Y), seed=seed, **over)


def _miceimp_then_saem(Y, seed=0, **over):
    return saem_estimate(mice_impute(Y, seed=seed), seed=seed, **over)


ESTIMATORS = {
    "SAEM":   saem_estimate,
    "EM_Gaussian":     _em_gaussian,
    "EM_Student_t":    _em_student,
    "MeanImp_SCM":     _mean_scm,
    "MICE_Tyler":      _mice_tyler,
    "Filter_Debiased": _filter_debiased,
}

ABLATIONS = {
    "SAEM_direct":        saem_estimate,
    "MeanImp_then_SAEM":  _meanimp_then_saem,
    "MICE_then_SAEM":     _miceimp_then_saem,
}

DIAGNOSTIC = {
    "Debiased_only":  _debiased_only,
    "MICE_Tyler_cal": lambda Y, seed=0: _mice_tyler(Y, seed=seed,
                                                    normalise="calibrate"),
}

ALL = {**ESTIMATORS, **DIAGNOSTIC}


def timed(fn, Y, seed=0, **kw):
    """Run an estimator and record its wall-clock time in ``info["runtime"]``.

    The time covers the whole pipeline, imputation included.
    """
    t0 = time.time()
    mu, Sigma, info = fn(Y, seed=seed, **kw)
    info = dict(info)
    info["runtime"] = time.time() - t0
    return mu, Sigma, info
