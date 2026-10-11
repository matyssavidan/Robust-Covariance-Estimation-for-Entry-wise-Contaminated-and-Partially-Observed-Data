"""Both orderings of the maximization step.

"legacy": mu and Sigma are updated from moments evaluated at the current iterate.
"ecm": CM1 updates mu, the Rao-Blackwellised moments are recomputed at the new
mu, then CM2 updates Sigma (Meng & Rubin, 1993).

Both are valid programs; these tests check that they return finite, symmetric,
positive-definite estimates and that the shape error is similar. Both orderings
see the same data and the same estimator seed (paired design).
``experiments/ecm_check.py`` gives the full comparison.
"""

import numpy as np
import pytest

from data_gen import generate_icm, generate_t_entrywise
from metrics import evaluate
from saem import EntrywiseSAEM

FIT = dict(max_iter=30, mcmc_steps=20, proposal_scale=1.0,
           adapt_proposal=True, target_acc=0.40)
N_REPS = 5


def _paired_runs(make_dataset, n_reps=N_REPS, **fit_over):
    """Fit both orderings on identical data with identical seeds."""
    rows = []
    for rep in range(n_reps):
        ds = make_dataset(rep)
        for m_step in ("legacy", "ecm"):
            kw = dict(FIT); kw.update(fit_over)
            model = EntrywiseSAEM(seed=rep, m_step=m_step, **kw).fit(ds.Y)
            mu, Sigma, nu = model.get_params()
            sc = evaluate(ds.mu_true, ds.Sigma_true, mu, Sigma)
            sc.update(rep=rep, m_step=m_step, nu_hat=nu,
                      acc=model.acc_rate_final_)
            rows.append(sc)
    return rows


def _summarise(rows, label):
    keys = ("fisher_shape", "fisher_scale", "cstar", "fro", "err_mu_l2", "nu_hat")
    out = {}
    print(f"\n  --- {label} ---")
    print(f"  {'metric':<14} {'legacy':>10} {'ecm':>10} {'rel. diff':>11}")
    for k in keys:
        a = np.mean([r[k] for r in rows if r["m_step"] == "legacy"])
        b = np.mean([r[k] for r in rows if r["m_step"] == "ecm"])
        rel = abs(b - a) / max(abs(a), 1e-12)
        out[k] = (a, b, rel)
        print(f"  {k:<14} {a:10.4f} {b:10.4f} {rel:10.1%}")
    return out


def test_both_orderings_are_valid():
    """Both must return finite, symmetric, positive-definite estimates."""
    ds = generate_icm(n=300, p=10, sigma_seed=0, data_seed=0,
                      missing_rate=0.10, outlier_rate=0.10, scale=10.0)
    for m_step in ("legacy", "ecm"):
        model = EntrywiseSAEM(seed=0, m_step=m_step, max_iter=15,
                             mcmc_steps=10).fit(ds.Y)
        _, Sigma, nu = model.get_params()
        assert np.isfinite(Sigma).all()
        assert np.allclose(Sigma, Sigma.T, atol=1e-10)
        assert np.linalg.eigvalsh(Sigma).min() > 0
        assert 0.05 < nu < 200


def test_gap_on_contaminated_data():
    """p=20, 10% missing, 10% contamination, s=10."""
    rows = _paired_runs(lambda rep: generate_icm(
        n=500, p=20, sigma_seed=rep, data_seed=rep,
        missing_rate=0.10, outlier_rate=0.10, scale=10.0))
    summary = _summarise(rows, "contaminated (p=20, eps=10%, s=10)")

    shape_gap = summary["fisher_shape"][2]
    print(f"\n  >>> shape-error gap between orderings: {shape_gap:.1%}")
    assert shape_gap < 2.0, "orderings disagree wildly; investigate before use"


def test_gap_on_wellspecified_data():
    """Same comparison where the model is true."""
    rows = _paired_runs(lambda rep: generate_t_entrywise(
        n=500, p=10, nu=5.0, sigma_seed=rep, data_seed=rep,
        missing_rate=0.10))
    summary = _summarise(rows, "well-specified t* (p=10, nu=5)")
    assert summary["fisher_shape"][2] < 2.0


@pytest.mark.parametrize("missing_rate", [0.0, 0.20])
def test_gap_grows_with_missingness(missing_rate):
    """The ordering matters through the dependence of the moments on mu, so the gap
    may depend on the missing rate.
    """
    rows = _paired_runs(lambda rep: generate_icm(
        n=400, p=10, sigma_seed=rep, data_seed=rep,
        missing_rate=missing_rate, outlier_rate=0.10, scale=10.0),
        n_reps=3)
    _summarise(rows, f"contaminated, missing={missing_rate}")
