"""Main curves: estimation error as a function of the outlier rate.

Common random numbers: replication r uses the same ground-truth covariance and
the same clean data at every outlier rate, missing rate and for every estimator.

    python experiments/main_curves.py --reps 20
"""

import argparse

from common import add_common_args, report_errors, result_path, run_estimators, run_sweep

from analysis import full_report
from baselines import DIAGNOSTIC, ESTIMATORS
from data_gen import generate_icm

OUTLIER_RATES = [round(0.025 * k, 3) for k in range(15)]
MISSING_RATES = [0.10, 0.20]


def task(rep, outlier_rate, missing_rate, n, p, scale, mode, names):
    pool = {**ESTIMATORS, **DIAGNOSTIC}
    ds = generate_icm(n=n, p=p, sigma_seed=rep, data_seed=rep,
                      missing_rate=missing_rate, outlier_rate=outlier_rate,
                      scale=scale, mode=mode)
    meta = dict(outlier_rate=outlier_rate, missing_rate=missing_rate, rep=rep,
                n=n, p=p, mode=mode, contam_scale=scale,
                outlier_frac_realised=ds.outlier_frac,
                missing_frac_realised=ds.missing_frac)
    return run_estimators({k: pool[k] for k in names}, ds.Y, ds.mu_true,
                          ds.Sigma_true, rep, meta)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_common_args(ap, reps=20)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--p", type=int, default=20)
    ap.add_argument("--scale", type=float, default=2.5)
    ap.add_argument("--mode", choices=["amplified", "shift"], default="amplified")
    ap.add_argument("--outlier-rates", type=float, nargs="+", default=OUTLIER_RATES)
    ap.add_argument("--missing-rates", type=float, nargs="+", default=MISSING_RATES)
    ap.add_argument("--diagnostic", action="store_true",
                    help="also run the diagnostic arms")
    args = ap.parse_args()

    names = list(ESTIMATORS) + (list(DIAGNOSTIC) if args.diagnostic else [])
    print(f"main curves | mode={args.mode} s={args.scale} n={args.n} "
          f"p={args.p} reps={args.reps}")

    for m in args.missing_rates:
        out = result_path(f"main_curves_{args.mode}_s{args.scale}_p{args.p}_m{m}",
                          args.tag)
        print(f"\nmissing rate {m}")
        points = [dict(outlier_rate=e, missing_rate=m, n=args.n, p=args.p,
                       scale=args.scale, mode=args.mode, names=names)
                  for e in args.outlier_rates]
        df = run_sweep(task, points, args.reps, args.jobs, out,
                       lambda pt: f"outliers={pt['outlier_rate']}")
        ok = df[df["error"].isna()] if "error" in df.columns else df
        full_report(ok, ref="SAEM", metric="fisher", x="outlier_rate")
        print(f"\nsaved {out}")
        report_errors(df)


if __name__ == "__main__":
    main()
