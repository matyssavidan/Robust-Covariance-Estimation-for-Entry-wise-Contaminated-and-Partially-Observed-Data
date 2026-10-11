"""Contamination-magnitude sweep: proposed method against every baseline.

Entries are contaminated with draws from N(mu_j, s^2 sigma_j^2) (``amplified``)
or displaced by +/- s sigma_j (``shift``), for a range of magnitudes s.

    python experiments/scale_sweep.py --reps 20
"""

import argparse

from common import add_common_args, report_errors, result_path, run_estimators, run_sweep

from analysis import comparison_table, full_report
from baselines import DIAGNOSTIC, ESTIMATORS
from data_gen import generate_icm

SCALES = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0]


def task(rep, scale, n, p, outlier_rate, missing_rate, mode, names):
    pool = {**ESTIMATORS, **DIAGNOSTIC}
    ds = generate_icm(n=n, p=p, sigma_seed=rep, data_seed=rep,
                      missing_rate=missing_rate, outlier_rate=outlier_rate,
                      scale=scale, mode=mode)
    meta = dict(contam_scale=scale, rep=rep, n=n, p=p, mode=mode,
                outlier_rate=outlier_rate, missing_rate=missing_rate,
                outlier_frac_realised=ds.outlier_frac,
                missing_frac_realised=ds.missing_frac)
    return run_estimators({k: pool[k] for k in names}, ds.Y, ds.mu_true,
                          ds.Sigma_true, rep, meta)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_common_args(ap, reps=20)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--p", type=int, default=20)
    ap.add_argument("--outlier-rate", type=float, default=0.10)
    ap.add_argument("--missing-rate", type=float, default=0.10)
    ap.add_argument("--mode", choices=["amplified", "shift"], default="amplified")
    ap.add_argument("--scales", type=float, nargs="+", default=SCALES)
    ap.add_argument("--diagnostic", action="store_true",
                    help="also run the diagnostic arms")
    args = ap.parse_args()

    names = list(ESTIMATORS) + (list(DIAGNOSTIC) if args.diagnostic else [])
    out = result_path(f"scale_sweep_{args.mode}_p{args.p}", args.tag)
    print(f"scale sweep | mode={args.mode} n={args.n} p={args.p} "
          f"outliers={args.outlier_rate} missing={args.missing_rate} "
          f"reps={args.reps}")

    points = [dict(scale=s, n=args.n, p=args.p, outlier_rate=args.outlier_rate,
                   missing_rate=args.missing_rate, mode=args.mode, names=names)
              for s in args.scales]
    df = run_sweep(task, points, args.reps, args.jobs, out,
                   lambda pt: f"s={pt['scale']}")

    ok = df[df["error"].isna()] if "error" in df.columns else df
    print("\nSAEM against Filter_Debiased (total Fisher-Rao distance)")
    print(comparison_table(ok, ref="SAEM", rival="Filter_Debiased",
                           metric="fisher", x="contam_scale")
          .round(4).to_string(index=False))
    full_report(ok, ref="SAEM", metric="fisher", x="contam_scale")
    print(f"\nsaved {out}")
    report_errors(df)


if __name__ == "__main__":
    main()
