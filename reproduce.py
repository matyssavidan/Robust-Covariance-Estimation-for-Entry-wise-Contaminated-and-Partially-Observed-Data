"""Reproduce the results of the paper and of the supplementary tables.

    python reproduce.py --list                  # targets and commands
    python reproduce.py all                     # everything (several hours)
    python reproduce.py scale-sweep ablation    # selected targets
    python reproduce.py all --quick             # reduced run to check the setup
    python reproduce.py real-data --data data/data.txt

Full runs write results/*.csv; --quick runs write results/*_quick.csv and never
overwrite them. Figures are written to figures/.
"""

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXP = "experiments"

# name: (description, where it is reported, [(script, full args, quick args, accepts --jobs)])
TARGETS = {
    "tests": (
        "unit tests", "-",
        [("-m pytest tests -q", "", "", False)]),
    "main-curves": (
        "error vs outlier rate, all estimators, 10% and 20% missing, s = 2.5",
        "Fig. 1 of the paper",
        [(f"{EXP}/main_curves.py", "--reps 20",
          "--reps 2 --outlier-rates 0 0.1 0.2 --missing-rates 0.1", True)]),
    "scale-sweep": (
        "contamination magnitude s from 1.5 to 10, SAEM vs the detect-then-estimate baseline",
        "Section V and supplementary table",
        [(f"{EXP}/scale_sweep.py", "--reps 20", "--reps 2 --scales 2.5 10", True)]),
    "runtime": (
        "runtime at p = 20, 40, 60 (n / [p(p+1)/2] = 2.38), one run",
        "supplementary table (hardware dependent)",
        [(f"{EXP}/scalability.py", "--sweep runtime --reps 1",
          "--sweep runtime --reps 1 --p-list 10 20", True)]),
    "scalability-p": (
        "dimension p = 10..60 at constant n / [p(p+1)/2]",
        "Section V and supplementary table",
        [(f"{EXP}/scalability.py", "--sweep ratio --reps 10",
          "--sweep ratio --reps 2 --p-list 10 20", True)]),
    "scalability-n": (
        "sample size n = 100..2000 at p = 20",
        "Section V and supplementary table",
        [(f"{EXP}/scalability.py", "--sweep n --reps 10",
          "--sweep n --reps 2 --n-list 100 500", True)]),
    "ablation": (
        "missing-data ablation: SAEM on incomplete data vs after mean / MICE imputation",
        "supplementary table",
        [(f"{EXP}/ablation_missing.py", "--reps 20",
          "--reps 2 --missing-rates 0.1 0.4", True)]),
    "ecm-check": (
        "ECM ordering vs previous ordering of the M-step, paired runs",
        "supplementary text",
        [(f"{EXP}/ecm_check.py", "--reps 20", "--reps 2", True)]),
    "real-data": (
        "Intel Berkeley data, 5 sensors, n = 1000, 10% and 20% missing",
        "Table I (10% missing) and supplementary table (20% missing)",
        [(f"{EXP}/real_data.py", "--data {data} --missing-rate 0.10 --reps 10",
          "--data {data} --missing-rate 0.10 --reps 2 --outlier-rates 0 0.1", True),
         (f"{EXP}/real_data.py", "--data {data} --missing-rate 0.20 --reps 10",
          "--data {data} --missing-rate 0.20 --reps 2 --outlier-rates 0 0.1", True)]),
    "figures": (
        "figures from the CSV files in results/", "Fig. 1 and supplementary figures",
        [(f"{EXP}/make_figures.py", "", "", False)]),
}

QUICK_FIGURES = ("--curves-glob main_curves_*quick.csv "
                 "--scale-glob scale_sweep_*quick.csv --suffix _quick")


def command(script, args, quick, jobs, data, use_jobs):
    parts = [sys.executable] + shlex.split(script) if script.startswith("-m") \
        else [sys.executable, script]
    parts += shlex.split(args.format(data=data))
    if quick and use_jobs:
        parts += ["--tag", "quick"]
    if quick and script.endswith("make_figures.py"):
        parts += shlex.split(QUICK_FIGURES)
    if jobs is not None and use_jobs:
        parts += ["--jobs", str(jobs)]
    return parts


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("targets", nargs="*", help="target names, or 'all'")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--quick", action="store_true",
                    help="few replications and a reduced grid")
    ap.add_argument("--data", default="data/data.txt",
                    help="Intel Lab data.txt (see scripts/download_intel.py)")
    ap.add_argument("--jobs", type=int, default=None,
                    help="parallel workers passed to the experiments")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the commands without running them")
    args = ap.parse_args()

    if args.list or not args.targets:
        for name, (desc, where, runs) in TARGETS.items():
            print(f"{name:<14} {desc}\n{'':<14} reported in: {where}")
            for script, full, _, _ in runs:
                print(f"{'':<14} $ python {script} "
                      f"{full.format(data=args.data)}".rstrip())
        return

    names = list(TARGETS) if "all" in args.targets else args.targets
    unknown = [n for n in names if n not in TARGETS]
    if unknown:
        sys.exit(f"unknown target(s): {', '.join(unknown)}; see --list")

    for name in names:
        desc, _, runs = TARGETS[name]
        print(f"\n== {name}: {desc}", flush=True)
        if name == "real-data" and not Path(args.data).exists() and not args.dry_run:
            print(f"   data file {args.data} not found; run "
                  f"'python scripts/download_intel.py' first", file=sys.stderr)
            continue
        for script, full, quick, use_jobs in runs:
            cmd = command(script, quick if args.quick else full, args.quick,
                          args.jobs, args.data, use_jobs)
            print("   $", " ".join(shlex.quote(c) for c in cmd), flush=True)
            if args.dry_run:
                continue
            t0 = time.time()
            rc = subprocess.run(cmd, cwd=ROOT).returncode
            print(f"   [{(time.time() - t0) / 60:.1f} min, exit {rc}]")
            if rc != 0:
                sys.exit(rc)


if __name__ == "__main__":
    main()
