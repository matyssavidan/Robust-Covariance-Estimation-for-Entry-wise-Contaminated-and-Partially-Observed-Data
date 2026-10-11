"""Recovery of the parameters when the model is true.

Levels, in increasing difficulty:
  1. nu known, no missing data: recovery of Sigma
  2. nu known, 10% missing: the missing-data machinery must not bias Sigma
  3. nu estimated, 10% missing: joint recovery of (Sigma, nu)

Metrics are the scale-invariant shape error and the scale factor c*.
"""

import numpy as np
import pytest

from data_gen import generate_t_entrywise, generate_gaussian
from metrics import evaluate
from saem import EntrywiseSAEM

N, P, NU = 800, 6, 5.0
FIT = dict(max_iter=60, mcmc_steps=15, proposal_scale=1.0,
           m_step="ecm", adapt_proposal=True, target_acc=0.40)


def _fit_and_score(ds, seed, **over):
    kw = dict(FIT); kw.update(over)
    model = EntrywiseSAEM(seed=seed, **kw).fit(ds.Y)
    mu, Sigma, nu = model.get_params()
    sc = evaluate(ds.mu_true, ds.Sigma_true, mu, Sigma)
    sc.update(nu_hat=nu, acc=model.acc_rate_final_, s=model.proposal_scale_)
    return sc


def test_sigma_recovered_nu_known_complete():
    """Model true, nu known, no missing data."""
    scores = []
    for rep in range(3):
        ds = generate_t_entrywise(n=N, p=P, nu=NU, sigma_seed=rep,
                                 data_seed=rep, missing_rate=0.0)
        scores.append(_fit_and_score(ds, seed=rep, nu=NU, estimate_nu=False))

    shape = np.mean([s["fisher_shape"] for s in scores])
    cstar = np.mean([s["cstar"] for s in scores])
    acc = np.mean([s["acc"] for s in scores])
    print(f"\n  [nu known, complete]  shape={shape:.3f}  c*={cstar:.3f}  acc={acc:.2f}")

    assert shape < 1.0, f"shape error {shape:.3f} too large on well-specified data"
    assert 0.7 < cstar < 1.4, f"scale badly calibrated (c*={cstar:.3f})"


def test_sigma_recovered_nu_known_with_missing():
    """Same with 10% MCAR missingness: no systematic bias relative to complete data."""
    comp, miss = [], []
    for rep in range(3):
        ds_c = generate_t_entrywise(n=N, p=P, nu=NU, sigma_seed=rep,
                                   data_seed=rep, missing_rate=0.0)
        ds_m = generate_t_entrywise(n=N, p=P, nu=NU, sigma_seed=rep,
                                   data_seed=rep, missing_rate=0.10)
        comp.append(_fit_and_score(ds_c, rep, nu=NU, estimate_nu=False))
        miss.append(_fit_and_score(ds_m, rep, nu=NU, estimate_nu=False))

    sc, sm = (np.mean([s["fisher_shape"] for s in x]) for x in (comp, miss))
    print(f"  [nu known, 10% missing]  shape complete={sc:.3f}  missing={sm:.3f}")

    assert sm < 1.3, f"shape error {sm:.3f} too large with missing data"
    assert sm < 2.0 * sc + 0.3, "missingness degrades the fit more than expected"


def test_nu_is_recovered():
    """Joint recovery: nu_hat should land near the true nu.

    nu is notoriously hard to pin down (the likelihood is flat in nu for
    moderate values), so the tolerance is wide by design. What must NOT happen
    is nu_hat collapsing to the lower bound or running to the upper one.
    """
    hats = []
    for rep in range(4):
        ds = generate_t_entrywise(n=N, p=P, nu=NU, sigma_seed=rep,
                                 data_seed=rep, missing_rate=0.10)
        hats.append(_fit_and_score(ds, rep, nu=10.0, estimate_nu=True)["nu_hat"])

    hats = np.array(hats)
    print(f"  [nu estimated] true={NU}  hats={np.round(hats, 2)}  "
          f"mean={hats.mean():.2f}")

    assert np.all(hats > 0.15), "nu collapsed to the solver's lower bound"
    assert np.all(hats < 99.0), "nu ran to the solver's upper bound"
    assert 0.4 * NU < hats.mean() < 3.0 * NU, f"nu_hat = {hats.mean():.2f} vs {NU}"


def test_gaussian_limit_gives_large_nu():
    """On clean Gaussian data (nu = infinity) the estimated nu is large."""
    hats = []
    for rep in range(3):
        ds = generate_gaussian(n=N, p=P, sigma_seed=rep, data_seed=rep,
                               missing_rate=0.10)
        hats.append(_fit_and_score(ds, rep, nu=10.0, estimate_nu=True)["nu_hat"])

    print(f"  [Gaussian data] nu_hat = {np.round(hats, 1)}")
    assert np.mean(hats) > NU, "nu_hat should be large on Gaussian data"


@pytest.mark.parametrize("m_step", ["ecm", "legacy"])
def test_runs_and_returns_pd_sigma(m_step):
    """Both orderings must return a symmetric positive-definite Sigma."""
    ds = generate_t_entrywise(n=200, p=5, nu=NU, sigma_seed=0, data_seed=0,
                             missing_rate=0.15)
    model = EntrywiseSAEM(seed=0, max_iter=20, mcmc_steps=10,
                         m_step=m_step).fit(ds.Y)
    _, Sigma, _ = model.get_params()

    assert np.allclose(Sigma, Sigma.T, atol=1e-10)
    assert np.linalg.eigvalsh(Sigma).min() > 0, f"{m_step}: Sigma not PD"
    assert np.isfinite(Sigma).all()
