"""Imputation used by the imputation-based baselines and by the ablation."""

import numpy as np

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

__all__ = ["mean_impute", "mice_impute"]


def mean_impute(Y):
    """Column-mean imputation."""
    X = np.asarray(Y, dtype=float).copy()
    col = np.nanmean(X, axis=0)
    col = np.where(np.isfinite(col), col, 0.0)
    idx = np.isnan(X)
    if idx.any():
        X[idx] = np.take(col, np.where(idx)[1])
    return X


def mice_impute(Y, seed=0, max_iter=10):
    """Iterative chained-equations imputation with a BayesianRidge estimator.

    ``sample_posterior=False`` gives conditional-mean (single) imputation.
    """
    X = np.asarray(Y, dtype=float)
    if not np.isnan(X).any():
        return X.copy()
    imp = IterativeImputer(estimator=BayesianRidge(), max_iter=max_iter,
                           random_state=seed, sample_posterior=False)
    return imp.fit_transform(X)
