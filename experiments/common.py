"""Shared utilities for the experiment scripts.

This module must be imported before numpy so that BLAS threading is pinned to
one thread per worker process.
"""

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
os.environ.setdefault("PYTHONWARNINGS",
                      "ignore::UserWarning:sklearn.impute._iterative")

import sys                                                  # noqa: E402
import time                                                 # noqa: E402
from pathlib import Path                                    # noqa: E402

import pandas as pd                                         # noqa: E402
from joblib import Parallel, delayed                        # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baselines import timed                                 # noqa: E402
from metrics import evaluate                                # noqa: E402

RESULTS = ROOT / "results"

__all__ = ["ROOT", "RESULTS", "run_estimators", "run_sweep", "result_path",
           "report_errors", "add_common_args"]


def run_estimators(estimators, Y, mu_true, Sigma_true, seed, meta):
    """Run each estimator on ``Y`` and return one result row per estimator.

    ``estimators`` maps a name to a callable ``fn(Y, seed=...)``. ``meta`` is
    copied into every row. A failing estimator yields a row with an ``error``
    field instead of stopping the experiment.
    """
    rows = []
    for name, fn in estimators.items():
        try:
            mu, Sigma, info = timed(fn, Y, seed=seed)
            rec = evaluate(mu_true, Sigma_true, mu, Sigma)
        except Exception as exc:                            # noqa: BLE001
            rec, info = {"error": repr(exc)}, {}
        rec.update(algo=name, **meta)
        rec.update(info)
        rows.append(rec)
    return rows


def run_sweep(task, points, reps, jobs, out_csv, label):
    """Run ``task(rep, **point)`` for every sweep point and replication.

    Replications of one point run in parallel. The results are written to
    ``out_csv`` after each point, so a partial run is never lost.
    """
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    for i, point in enumerate(points):
        batches = Parallel(n_jobs=jobs, verbose=0)(
            delayed(task)(rep, **point) for rep in range(reps))
        for batch in batches:
            rows.extend(batch)
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        elapsed = (time.time() - t0) / 60
        eta = elapsed / (i + 1) * (len(points) - i - 1)
        print(f"  [{i + 1}/{len(points)}] {label(point)}   "
              f"{elapsed:5.1f} min elapsed, {eta:5.1f} min remaining",
              flush=True)
    return pd.DataFrame(rows)


def result_path(stem, tag=""):
    """results/<stem>[_<tag>].csv"""
    RESULTS.mkdir(exist_ok=True)
    return RESULTS / (f"{stem}_{tag}.csv" if tag else f"{stem}.csv")


def report_errors(df):
    if "error" in df.columns and df["error"].notna().any():
        print("\nErrors:")
        for e in df["error"].dropna().unique()[:5]:
            print("  ", e)


def add_common_args(ap, reps):
    ap.add_argument("--reps", type=int, default=reps,
                    help="number of replications (default: %(default)s)")
    ap.add_argument("--jobs", type=int, default=-1,
                    help="parallel workers; -1 uses all cores, 1 is serial")
    ap.add_argument("--tag", default="", help="suffix of the output file name")
