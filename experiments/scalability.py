"""Scalability: runtime, dimension and sample-size sweeps.

ratio    p varies, n grows with p so that n / [p(p+1)/2] stays constant.
n        p is fixed, n varies.
runtime  one run per estimator at p = 20, 40, 60 (constant ratio); the report
         of interest is the ``runtime`` column.

    python experiments/scalability.py --sweep runtime --reps 1
    python experiments/scalability.py --sweep ratio --reps 10
    python experiments/scalability.py --sweep n --reps 10
    python experiments/scalability.py --sweep ratio --dry-run
"""

import argparse

from common import add_common_args, report_errors, result_path, run_estimators, run_sweep

from analysis import full_report
from baselines import ESTIMATORS
from data_gen import generate_icm

TARGET_RATIO = 500 / (20 * 21 / 2)
P_RATIO = [10, 20, 30, 40, 50, 60]
P_RUNTIME = [20, 40, 60]
P_FIXED = 20
N_LIST = [100, 200, 350, 500, 1000, 2000]


def n_for_ratio(p, ratio=TARGET_RATIO):
    """Sample size keeping n / [p(p+1)/2] equal to ``ratio``."""
    return int(round(ratio * p * (p + 1) / 2))


def memory_gb(n, p):
    """Size of the (n, p, p) array of matrices A_i, in float64."""
    return n * p * p * 8 / 1e9


def task(rep, n, p, outlier_rate, missing_rate, scale, mode, names):
    ds = generate_icm(n=n, p=p, sigma_seed=rep, data_seed=rep,
                      missing_rate=missing_rate, outlier_rate=outlier_rate,
                      scale=scale, mode=mode)
    n_par = p * (p + 1) / 2
    meta = dict(p=p, n=n, rep=rep, mode=mode, contam_scale=scale,
                outlier_rate=outlier_rate, missing_rate=missing_rate,
                n_params=n_par, samples_per_param=n / n_par)
    return run_estimators({k: ESTIMATORS[k] for k in names}, ds.Y, ds.mu_true,
                          ds.Sigma_true, rep, meta)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawTextHelpFormatter)
    add_common_args(ap, reps=10)
    ap.add_argument("--sweep", choices=["ratio", "n", "runtime"], required=True)
    ap.add_argument("--ratio", type=float, default=TARGET_RATIO)
    ap.add_argument("--p-list", type=int, nargs="+", default=None)
    ap.add_argument("--n-list", type=int, nargs="+", default=N_LIST)
    ap.add_argument("--p-fixed", type=int, default=P_FIXED)
    ap.add_argument("--outlier-rate", type=float, default=0.10)
    ap.add_argument("--missing-rate", type=float, default=0.10)
    ap.add_argument("--scale", type=float, default=2.5)
    ap.add_argument("--mode", choices=["amplified", "shift"], default="amplified")
    ap.add_argument("--no-mice", action="store_true",
                    help="skip MICE + Tyler")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the grid and the memory footprint, then exit")
    args = ap.parse_args()

    if args.sweep == "n":
        cells = [(n, args.p_fixed) for n in args.n_list]
    else:
        default_p = P_RUNTIME if args.sweep == "runtime" else P_RATIO
        cells = [(n_for_ratio(p, args.ratio), p)
                 for p in (args.p_list or default_p)]

    print(f"scalability ({args.sweep}) | mode={args.mode} s={args.scale} "
          f"outliers={args.outlier_rate} missing={args.missing_rate} "
          f"reps={args.reps}")
    print(f"\n{'p':>5} {'n':>7} {'n/params':>9} {'A_i array':>11}")
    for n, p in cells:
        print(f"{p:5d} {n:7d} {n / (p * (p + 1) / 2):9.2f} "
              f"{memory_gb(n, p) * 1000:8.0f} MB")
    if args.dry_run:
        return

    names = [k for k in ESTIMATORS if not (args.no_mice and k == "MICE_Tyler")]
    out = result_path(f"scalability_{args.sweep}_{args.mode}", args.tag)
    points = [dict(n=n, p=p, outlier_rate=args.outlier_rate,
                   missing_rate=args.missing_rate, scale=args.scale,
                   mode=args.mode, names=names) for n, p in cells]
    df = run_sweep(task, points, args.reps, args.jobs, out,
                   lambda pt: f"p={pt['p']} n={pt['n']}")

    ok = df[df["error"].isna()] if "error" in df.columns else df
    x = "n" if args.sweep == "n" else "p"
    full_report(ok, ref="SAEM", metric="fisher", x=x)
    print(f"\nsaved {out}")
    report_errors(df)


if __name__ == "__main__":
    main()
