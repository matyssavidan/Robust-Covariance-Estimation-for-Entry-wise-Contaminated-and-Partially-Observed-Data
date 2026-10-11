"""Paired analysis of experiment results.

Experiments use common random numbers: replication r sees the same ground-truth
covariance and the same clean data for every estimator and every sweep point.
Comparisons are therefore paired, and standard errors are computed on the
paired difference.

Command line:
    python src/analysis.py results/main_curves_amplified_s2.5_p20_m0.1.csv
    python src/analysis.py results/ablation_amplified_s2.5_p20.csv --ref SAEM_direct
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["paired_table", "summary_table", "comparison_table", "full_report",
           "detect_x_column"]

X_CANDIDATES = ("contam_scale", "missing_rate", "outlier_rate", "p", "n")


def detect_x_column(df):
    """The swept axis: the first candidate column with several distinct values."""
    for c in X_CANDIDATES:
        if c in df.columns and df[c].nunique() > 1:
            return c
    for c in X_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError("no sweep column found")


def _clean(df):
    return df[df["error"].isna()] if "error" in df.columns else df


def _pair(sub, ref, rival, metric):
    a = sub[sub.algo == ref].drop_duplicates("rep").set_index("rep")[metric]
    b = sub[sub.algo == rival].drop_duplicates("rep").set_index("rep")[metric]
    common = a.index.intersection(b.index)
    d = (a.loc[common] - b.loc[common]).to_numpy()
    return d[np.isfinite(d)]


def summary_table(df, metric="fisher", x=None):
    """Mean of ``metric`` by sweep point (rows) and algorithm (columns)."""
    df = _clean(df)
    x = x or detect_x_column(df)
    return df.pivot_table(index=x, columns="algo", values=metric,
                          aggfunc="mean")


def paired_table(df, ref="SAEM", metric="fisher", x=None):
    """Paired difference ``ref - rival`` at every sweep point, for every rival.

    Columns: mean difference, standard error of the difference, number of
    replications in which ``ref`` has the lower value, number of pairs, and
    ``verdict`` (``ref_better`` / ``rival_better`` / ``tie`` at two standard
    errors).
    """
    df = _clean(df)
    x = x or detect_x_column(df)
    rivals = [a for a in sorted(df.algo.unique()) if a != ref]

    rows = []
    for xv in sorted(df[x].unique()):
        sub = df[df[x] == xv]
        for r in rivals:
            d = _pair(sub, ref, r, metric)
            if d.size == 0:
                continue
            se = d.std(ddof=1) / np.sqrt(d.size) if d.size > 1 else np.nan
            significant = np.isfinite(se) and abs(d.mean()) > 2 * se
            if significant and d.mean() < 0:
                verdict = "ref_better"
            elif significant and d.mean() > 0:
                verdict = "rival_better"
            else:
                verdict = "tie"
            rows.append({x: xv, "rival": r, "diff": d.mean(), "se": se,
                         "wins": int((d < 0).sum()), "n_pairs": d.size,
                         "verdict": verdict})
    return pd.DataFrame(rows)


def comparison_table(df, ref="SAEM", rival="Filter_Debiased", metric="fisher",
                     x=None):
    """Two-method comparison along a sweep.

    Per sweep point: mean and standard deviation (over replications) of each
    method, paired difference with its standard error, wins of ``ref`` and, when
    available, the fraction of entries flagged by ``rival``.
    """
    df = _clean(df)
    x = x or detect_x_column(df)
    rows = []
    for xv in sorted(df[x].unique()):
        sub = df[df[x] == xv]
        a = sub[sub.algo == ref][metric]
        b = sub[sub.algo == rival][metric]
        d = _pair(sub, ref, rival, metric)
        row = {x: xv,
               f"{ref}_mean": a.mean(), f"{ref}_std": a.std(ddof=1),
               f"{rival}_mean": b.mean(), f"{rival}_std": b.std(ddof=1),
               "diff": d.mean() if d.size else np.nan,
               "se": d.std(ddof=1) / np.sqrt(d.size) if d.size > 1 else np.nan,
               "wins": int((d < 0).sum()), "n_pairs": d.size}
        if "flagged_frac" in sub.columns:
            row["flagged_pct"] = 100 * sub[sub.algo == rival]["flagged_frac"].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def _print_paired(pt, x, ref, metric):
    print(f"\n=== paired differences, {ref} minus rival [{metric}] ===")
    print("negative: reference method lower; standard error of the difference\n")
    for xv in sorted(pt[x].unique()):
        print(f"  {x} = {xv}")
        for _, r in pt[pt[x] == xv].sort_values("diff").iterrows():
            print(f"    {r.rival:<20} {r['diff']:+9.4f} +- {r.se:6.4f}"
                  f"   {r.wins:3d}/{r.n_pairs:<3d}  {r.verdict}")
        print()


def full_report(df, ref="SAEM", metric="fisher", x=None):
    """Print mean tables, secondary metrics and paired differences."""
    df = _clean(df)
    if df.empty:
        print("no successful run")
        return
    x = x or detect_x_column(df)

    print(f"\n=== {metric} (mean) ===")
    print(summary_table(df, metric, x).round(3).to_string())

    for col, label in (("fisher_shape", "fisher_shape (mean)"),
                       ("cstar", "cstar (mean)"),
                       ("fro", "fro (mean)"),
                       ("err_mu_l2", "err_mu_l2 (mean)"),
                       ("flagged_frac", "flagged_frac (mean)"),
                       ("psd_shift", "psd_shift (mean)"),
                       ("nu_hat", "nu_hat (mean)"),
                       ("runtime", "runtime in seconds (mean)")):
        if col == metric or col not in df.columns or df[col].isna().all():
            continue
        t = df.pivot_table(index=x, columns="algo", values=col, aggfunc="mean")
        print(f"\n=== {label} ===")
        print(t.round(4 if col in ("flagged_frac", "psd_shift") else 3)
              .to_string())

    pt = paired_table(df, ref=ref, metric=metric, x=x)
    if pt.empty:
        print(f"\nno paired comparison available (is {ref} present?)")
        return
    _print_paired(pt, x, ref, metric)


def main():
    ap = argparse.ArgumentParser(description="Paired analysis of a results CSV")
    ap.add_argument("csv", type=Path)
    ap.add_argument("--ref", default="SAEM")
    ap.add_argument("--metric", default="fisher",
                    help="fisher (total), fisher_shape, fro, err_mu_l2, ...")
    ap.add_argument("--x", default=None, help="sweep column (auto-detected)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    print(f"{args.csv.name}: {len(df)} rows, {df.algo.nunique()} algorithms, "
          f"{df.rep.nunique()} replications")
    full_report(df, ref=args.ref, metric=args.metric, x=args.x)


if __name__ == "__main__":
    main()
