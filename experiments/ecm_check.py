"""Effect of the ordering of the maximization step.

Paired runs of the two orderings on the same data and seeds:

    legacy  mu and Sigma both updated from moments evaluated at the current
            iterate
    ecm     CM1 on mu, moments recomputed, CM2 on Sigma

    python experiments/ecm_check.py --reps 20
"""

import argparse

from common import RESULTS

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from data_gen import generate_icm
from metrics import evaluate
from saem import EntrywiseSAEM

FIT = dict(max_iter=30, mcmc_steps=20, proposal_scale=1.0,
           adapt_proposal=True, target_acc=0.40)
KEYS = ("fisher", "fisher_shape", "fisher_scale", "cstar", "fro",
        "err_mu_l2", "nu_hat")


def one(rep, m_step, mode, scale, p, n, eps, missing):
    ds = generate_icm(n=n, p=p, sigma_seed=rep, data_seed=rep,
                      missing_rate=missing, outlier_rate=eps,
                      scale=scale, mode=mode)
    model = EntrywiseSAEM(seed=rep, m_step=m_step, **FIT).fit(ds.Y)
    mu, Sigma, nu = model.get_params()
    row = evaluate(ds.mu_true, ds.Sigma_true, mu, Sigma)
    row.update(rep=rep, m_step=m_step, missing=missing, nu_hat=nu)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=["amplified", "shift"], default="amplified")
    ap.add_argument("--scale", type=float, default=2.5)
    ap.add_argument("--p", type=int, default=20)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--eps", type=float, default=0.10, help="outlier rate")
    ap.add_argument("--missing", type=float, nargs="+", default=[0.10, 0.20])
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    print(f"ECM vs legacy | mode={a.mode} s={a.scale} p={a.p} n={a.n} "
          f"outliers={a.eps} reps={a.reps}\n")

    tasks = [(rep, ms, a.mode, a.scale, a.p, a.n, a.eps, m)
             for m in a.missing for rep in range(a.reps)
             for ms in ("legacy", "ecm")]
    df = pd.DataFrame(Parallel(n_jobs=a.jobs)(delayed(one)(*t) for t in tasks))

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / (f"ecm_check_{a.mode}_s{a.scale}_p{a.p}"
                     f"{'_' + a.tag if a.tag else ''}.csv")
    df.to_csv(out, index=False)

    for m in a.missing:
        sub = df[df["missing"] == m]
        legacy = sub[sub.m_step == "legacy"].set_index("rep")
        ecm = sub[sub.m_step == "ecm"].set_index("rep")
        print(f"missing rate {m:.2f}")
        print(f"  {'metric':<14}{'legacy':>9}{'ecm':>9}{'rel. diff':>11}"
              f"{'paired diff (ecm - legacy)':>30}")
        for k in KEYS:
            d = (ecm[k] - legacy[k]).to_numpy()
            se = d.std(ddof=1) / np.sqrt(d.size)
            base = legacy[k].mean()
            print(f"  {k:<14}{base:9.4f}{ecm[k].mean():9.4f}"
                  f"{abs(d.mean()) / max(abs(base), 1e-12):10.1%}"
                  f"{d.mean():+16.4f} +- {se:.4f}")
        print()
    print(f"saved {out}")


if __name__ == "__main__":
    main()
