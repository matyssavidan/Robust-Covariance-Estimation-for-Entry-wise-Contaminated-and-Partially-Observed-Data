# Robust Covariance Estimation for Entry-wise Contaminated and Partially Observed Data via Expectation Maximization Algorithm

Code for the paper *Robust Covariance Estimation for Entry-wise Contaminated and
Partially Observed Data via Expectation Maximization Algorithm*: a stochastic
approximation EM (SAEM) for the Student-t\* model, with missing data, and all
the experiments of the paper.

## What the paper does

Data matrices are often corrupted entry by entry rather than row by row: a few
individual sensor readings are wrong while the rest of the observation is
fine. Classical robust estimators (Student-t EM, Tyler's M-estimator) give one
weight per observation, so one bad entry removes the whole row. Many entries
are also missing.

The model used here assigns one latent scale `tau_ij` to every entry:

```
y_ij = mu_j + x_ij / sqrt(tau_ij),     x_i ~ N(0, Sigma),     tau_ij ~ Gamma(nu/2, nu/2)
```

so an isolated outlier is down-weighted without discarding the other entries of
its row. The parameters `(mu, Sigma, nu)` are estimated by SAEM:

- missing entries are integrated out analytically given `tau` (Rao-Blackwellisation);
- the entry-wise scales `tau_ij` are simulated by Metropolis-Hastings;
- the M-step is an ECM step (update `mu`, recompute the moments, update `Sigma`);
- `nu` is updated by a one-dimensional maximization.

The cost per iteration is `O(n p^3 + K n p^2)` for `K` Metropolis-Hastings steps.
The method is compared with Gaussian EM, Student-t EM, mean imputation followed
by the sample covariance, MICE followed by Tyler's estimator, and a
detect-then-estimate pipeline. The error is measured with the Fisher-Rao
distance between the estimated and true covariance.

## Installation

Python 3.10 or later.

```bash
pip install -r requirements.txt
```

The exact package versions used for the reported results are listed in
`requirements-lock.txt` (Python 3.12):

```bash
pip install -r requirements-lock.txt
```

## Quick start

```python
import sys; sys.path.insert(0, "src")
from saem import EntrywiseSAEM

# Y: (n, p) array, np.nan marks missing entries
model = EntrywiseSAEM(max_iter=30, mcmc_steps=20, seed=0).fit(Y)
mu, Sigma, nu = model.get_params()
```

`Sigma` is the covariance of the latent Gaussian vector. The covariance of the
observations is `metrics.cov_y_from_sigma(Sigma, nu)` (finite for `nu > 2`).

## Repository layout

```
src/
  saem.py              EntrywiseSAEM: the proposed estimator
  data_gen.py          synthetic data, independent contamination model
  metrics.py           Fisher-Rao distance (total, scale, shape), Frobenius error
  analysis.py          paired comparisons on result files
  baselines/
    em_gaussian.py     Gaussian EM with missing data
    em_student.py      Student-t EM with missing data (one scale per observation)
    imputation.py      mean imputation and MICE
    scm.py             sample covariance and Tyler's M-estimator
    detection.py       detect-then-estimate pipeline
    __init__.py        registry of the compared estimators

experiments/
  common.py            shared sweep and parallel-run utilities
  main_curves.py       error versus outlier rate
  scale_sweep.py       error versus contamination magnitude
  scalability.py       runtime, dimension and sample-size sweeps
  ablation_missing.py  SAEM on incomplete data versus after imputation
  ecm_check.py         ECM ordering versus the previous M-step
  real_data.py         semi-synthetic experiment on Intel Berkeley data
  make_figures.py      figures, from the files in results/

scripts/download_intel.py   downloads the Intel Berkeley data
tests/                      unit tests
reproduce.py                runs the experiments below
results/  figures/          outputs
```

## What to run

```bash
python reproduce.py --list              # targets and their exact commands
python reproduce.py all --quick         # reduced run to check the setup (minutes)
python reproduce.py all --jobs -1       # full run, all cores
python reproduce.py main-curves ablation --jobs 4     # selected targets
python reproduce.py all --dry-run       # print the commands only
```

Quick runs write `results/*_quick.csv` and `figures/*_quick.*` and do not
overwrite full results.

| Target | Command | Output | Used for |
|---|---|---|---|
| `tests` | `python -m pytest tests -q` | | correctness checks |
| `main-curves` | `python experiments/main_curves.py --reps 20` | `results/main_curves_*.csv` | Fig. 1 |
| `scale-sweep` | `python experiments/scale_sweep.py --reps 20` | `results/scale_sweep_*.csv` | Section V, supplementary table |
| `runtime` | `python experiments/scalability.py --sweep runtime --reps 1` | `results/scalability_*.csv` | runtime table |
| `scalability-p` | `python experiments/scalability.py --sweep ratio --reps 10` | `results/scalability_*.csv` | Section V, supplementary table |
| `scalability-n` | `python experiments/scalability.py --sweep n --reps 10` | `results/scalability_*.csv` | Section V, supplementary table |
| `ablation` | `python experiments/ablation_missing.py --reps 20` | `results/ablation_*.csv` | missing-data ablation |
| `ecm-check` | `python experiments/ecm_check.py --reps 20` | `results/ecm_check_*.csv` | ECM versus previous M-step |
| `real-data` | `python experiments/real_data.py --data data/data.txt --missing-rate 0.10` (and `0.20`) | `results/real_data_*.csv` | Table I |
| `figures` | `python experiments/make_figures.py` | `figures/*.pdf`, `figures/*.png` | Fig. 1 and complementary figures |

Order: `main-curves` and `scale-sweep` before `figures`. `real-data` needs the
data file (see below).

Common settings: `n = 500`, `p = 20`, 10% outliers, contamination
`N(mu_j, s^2 sigma_j^2)` with `s = 2.5` (`--mode amplified`), 10% missing
entries unless stated otherwise (the main curves use 10% and 20%).

### Real data

```bash
python scripts/download_intel.py        # writes data/data.txt
```

or download `data.txt.gz` from http://db.csail.mit.edu/labdata/labdata.html,
extract it, and place `data.txt` in `data/`. The ground truth is the covariance of
the clean series (5 sensors, 1000 time points); contamination and missing
entries are injected artificially.

### Reading the results

```bash
python src/analysis.py results/<file>.csv                    # paired comparison
python src/analysis.py results/<file>.csv --ref SAEM_direct --metric fisher
```

Metrics: `fisher` (total Fisher-Rao distance), `fisher_shape`, `fisher_scale`,
`cstar`, `fro` (relative Frobenius error), `err_mu_l2` (error on the mean).

Each file contains one row per estimator, replication and sweep point. Common
random numbers are used: replication `r` has the same ground-truth covariance
and clean data for every estimator and sweep point, so comparisons are paired
and standard errors are computed on the paired difference.

### Metrics

The total Fisher-Rao distance is `d^2 = sum_i log^2(lambda_i)`, with `lambda_i`
the eigenvalues of `Sigma^{-1} Sigma_hat`. It splits orthogonally into a scale
term `p (log c*)^2`, with `c* = (det Sigma_hat / det Sigma)^(1/p)`, and a shape
term.

### Notes

- Results are deterministic given the arguments. Runtimes depend on the hardware;
  run the `runtime` target alone on an otherwise idle machine.
- The SAEM stores an `(n, p, p)` array. At `p = 60`, `n = 4357` the peak is
  about 0.4 GB per worker. Use `--jobs` to limit the number of workers.
- If `pip` or `pytest` is blocked on your system, use `python -m pip` and
  `python -m pytest`.

## Tests

```bash
python -m pytest tests -q
```

## License

MIT, see `LICENSE`.
