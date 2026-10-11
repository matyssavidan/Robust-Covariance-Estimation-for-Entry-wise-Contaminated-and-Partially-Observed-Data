"""Ablation of the missing-data treatment.

The same SAEM (identical code, hyper-parameters and seed) is run three ways:
on the incomplete data, after mean imputation, and after MICE imputation.

    python experiments/ablation_missing.py --reps 20
"""

import argparse

from common import add_common_args, report_errors, result_path, run_estimators, run_sweep

from analysis import full_report
from baselines import ABLATIONS
from data_gen import generate_icm

MISSING_RATES = [0.05, 0.10, 0.20, 0.30, 0.40]


def task(rep, missing_rate, n, p, outlier_rate, scale, mode):
    ds = generate_icm(n=n, p=p, sigma_seed=rep, data_seed=rep,
                      missing_rate=missing_rate, outlier_rate=outlier_rate,
                      scale=scale, mode=mode)
    meta = dict(missing_rate=missing_rate, rep=rep, n=n, p=p, mode=mode,
                contam_scale=scale, outlier_rate=outlier_rate,
                missing_frac_realised=ds.missing_frac)
    return run_estimators(dict(ABLATIONS), ds.Y, ds.mu_true, ds.Sigma_true,
                          rep, meta)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_common_args(ap, reps=20)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--p", type=int, default=20)
    ap.add_argument("--outlier-rate", type=float, default=0.10)
    ap.add_argument("--scale", type=float, default=2.5)
    ap.add_argument("--mode", choices=["amplified", "shift"], default="amplified")
    ap.add_argument("--missing-rates", type=float, nargs="+", default=MISSING_RATES)
    args = ap.parse_args()

    out = result_path(f"ablation_{args.mode}_s{args.scale}_p{args.p}", args.tag)
    print(f"missing-data ablation | mode={args.mode} s={args.scale} "
          f"outliers={args.outlier_rate} n={args.n} p={args.p} reps={args.reps}")

    points = [dict(missing_rate=m, n=args.n, p=args.p,
                   outlier_rate=args.outlier_rate, scale=args.scale,
                   mode=args.mode) for m in args.missing_rates]
    df = run_sweep(task, points, args.reps, args.jobs, out,
                   lambda pt: f"missing={pt['missing_rate']}")

    ok = df[df["error"].isna()] if "error" in df.columns else df
    full_report(ok, ref="SAEM_direct", metric="fisher", x="missing_rate")
    print(f"\nsaved {out}")
    report_errors(df)


if __name__ == "__main__":
    main()
