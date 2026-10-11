"""Semi-synthetic experiment on the Intel Berkeley Research Lab temperature data.

The clean covariance of the pivoted sensor series is the ground truth;
contamination and missing entries are injected artificially.

Data: http://db.csail.mit.edu/labdata/labdata.html (``data.txt``), or
``python scripts/download_intel.py``.

    python experiments/real_data.py --data data/data.txt --missing-rate 0.10
    python experiments/real_data.py --data data/data.txt --missing-rate 0.20
"""

import argparse
from pathlib import Path

from common import add_common_args, report_errors, result_path, run_estimators, run_sweep

import numpy as np
import pandas as pd

from analysis import full_report
from baselines import ESTIMATORS

OUTLIER_RATES = [0.0, 0.05, 0.10, 0.15, 0.20]


def load_intel(path, sample_size=1000, p=5):
    """Pivot the sensor readings into an (n, p) temperature series.

    Keeps the ``p`` sensors with the most readings, interpolates linearly and
    takes a window of ``sample_size`` epochs centred on the series.
    """
    df = pd.read_csv(path, sep=r"\s+", header=None,
                     names=["date", "time", "epoch", "moteid",
                            "temp", "humid", "light", "volt"],
                     on_bad_lines="skip")
    df["temp"] = pd.to_numeric(df["temp"], errors="coerce")
    df = df.dropna(subset=["temp", "epoch", "moteid"])
    df = df[(df["temp"] > 5) & (df["temp"] < 40)]

    piv = df.pivot_table(index="epoch", columns="moteid", values="temp")
    counts = piv.count().sort_values(ascending=False)
    if len(counts) < p:
        raise ValueError(f"only {len(counts)} usable sensors, need {p}")

    clean = piv[counts.head(p).index].copy()
    clean = clean.interpolate(method="linear", limit_direction="both").dropna()
    if len(clean) > sample_size:
        start = max(0, len(clean) // 2 - sample_size // 2)
        clean = clean.iloc[start:start + sample_size]
    return clean.values


def contaminate(X_clean, outlier_rate, missing_rate, scale, mode, seed):
    """Independent contamination, then MCAR missingness, on a fixed matrix.

    Same mechanism as ``data_gen.generate_icm``, with the empirical mean and
    standard deviation of each column in place of mu_j and sigma_j.
    """
    rng = np.random.default_rng(seed)
    X = X_clean.copy()
    n, p = X.shape
    mu_j = X_clean.mean(axis=0)
    sd_j = X_clean.std(axis=0)

    mask_out = np.zeros((n, p), dtype=bool)
    if outlier_rate > 0:
        mask_out = rng.random((n, p)) < outlier_rate
        cols = np.where(mask_out)[1]
        k = int(mask_out.sum())
        if k:
            if mode == "amplified":
                X[mask_out] = mu_j[cols] + rng.normal(0.0, scale * sd_j[cols])
            else:
                signs = rng.choice([-1.0, 1.0], size=k)
                X[mask_out] = (mu_j[cols] + signs * scale * sd_j[cols]
                               + rng.normal(0.0, 0.5 * sd_j[cols]))

    if missing_rate > 0:
        X[rng.random((n, p)) < missing_rate] = np.nan
    return X, mask_out


def task(rep, outlier_rate, X_clean, mu_true, cov_true, missing_rate, scale,
         mode, names):
    Y, mask_out = contaminate(X_clean, outlier_rate, missing_rate, scale, mode,
                              seed=1000 + rep)
    meta = dict(outlier_rate=outlier_rate, rep=rep, mode=mode,
                contam_scale=scale, missing_rate=missing_rate,
                n=Y.shape[0], p=Y.shape[1],
                outlier_frac_realised=float(mask_out.mean()),
                missing_frac_realised=float(np.isnan(Y).mean()))
    return run_estimators({k: ESTIMATORS[k] for k in names}, Y, mu_true,
                          cov_true, rep, meta)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawTextHelpFormatter)
    add_common_args(ap, reps=10)
    ap.add_argument("--data", type=Path, required=True,
                    help="path to the Intel Lab data.txt")
    ap.add_argument("--p", type=int, default=5, help="number of sensors")
    ap.add_argument("--sample-size", type=int, default=1000)
    ap.add_argument("--outlier-rates", type=float, nargs="+", default=OUTLIER_RATES)
    ap.add_argument("--missing-rate", type=float, default=0.10)
    ap.add_argument("--scale", type=float, default=2.5)
    ap.add_argument("--mode", choices=["amplified", "shift"], default="amplified")
    args = ap.parse_args()

    X_clean = load_intel(args.data, sample_size=args.sample_size, p=args.p)
    mu_true = X_clean.mean(axis=0)
    cov_true = np.cov(X_clean, rowvar=False)
    w = np.linalg.eigvalsh(cov_true)

    print(f"real data | clean data {X_clean.shape}, condition number of the "
          f"clean covariance {w.max() / max(w.min(), 1e-12):.1f}")
    print(f"mode={args.mode} s={args.scale} missing={args.missing_rate} "
          f"reps={args.reps}")

    names = list(ESTIMATORS)
    out = result_path(f"real_data_{args.mode}_s{args.scale}_p{args.p}"
                      f"_m{args.missing_rate}", args.tag)
    points = [dict(outlier_rate=e, X_clean=X_clean, mu_true=mu_true,
                   cov_true=cov_true, missing_rate=args.missing_rate,
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
