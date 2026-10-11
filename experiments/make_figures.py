"""Figures, built from the CSV files in results/ (nothing is recomputed).

    python experiments/make_figures.py                  # all figures
    python experiments/make_figures.py --only paper decomposition
    python experiments/make_figures.py --only regimes

paper          estimation error vs outlier rate: l2 error on mu, Fisher-Rao
               distance, relative Frobenius error (one missing rate)
decomposition  total / scale / shape Fisher-Rao terms of SAEM, and the scale
               factor c*, vs outlier rate
regimes        paired gap between SAEM and the detect-then-estimate baseline,
               and fraction of entries flagged, vs contamination magnitude s

The configuration (mode, s, p, n, missing rate) is read from the columns of the
CSV files. If a folder holds several configurations, they are listed and the
script stops; select one with --mode / --scale / --p / --n.

Figures are written to figures/ as PDF and PNG.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402
from matplotlib.ticker import FuncFormatter                  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "figures"

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.labelsize": 8,
    "axes.titlesize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "lines.markersize": 3.5, "axes.grid": True,
    "grid.alpha": 0.3, "grid.linewidth": 0.4,
})

ORDER = ["SAEM", "EM_Gaussian", "EM_Student_t", "MeanImp_SCM", "MICE_Tyler",
         "Filter_Debiased"]
LABELS = {
    "SAEM": "SAEM (proposed)", "EM_Gaussian": "Gaussian EM",
    "EM_Student_t": "Student-$t$ EM", "MeanImp_SCM": "Mean imp. + SCM",
    "MICE_Tyler": "MICE + Tyler", "Filter_Debiased": "Filter + debiased",
}
STYLE = {
    "SAEM": dict(color="#000000", marker="o", linestyle="-", linewidth=1.8),
    "EM_Gaussian": dict(color="#0072B2", marker="s", linestyle="--", linewidth=1.0),
    "EM_Student_t": dict(color="#D55E00", marker="^", linestyle="-.", linewidth=1.0),
    "MeanImp_SCM": dict(color="#009E73", marker="v", linestyle=":", linewidth=1.2),
    "MICE_Tyler": dict(color="#CC79A7", marker="D", linestyle=(0, (3, 1, 1, 1)), linewidth=1.0),
    "Filter_Debiased": dict(color="#E69F00", marker="x", linestyle=(0, (5, 2)), linewidth=1.0),
}
MODE_LABEL = {"amplified": "Amplified", "shift": "Shifted"}
MODE_STYLE = [("#0072B2", "s"), ("#D55E00", "o"), ("#009E73", "^")]


def read_all(results, pattern):
    files = sorted(results.glob(pattern))
    if not files:
        print(f"  no file matching '{pattern}' in {results}")
        return None
    frames = []
    for f in files:
        d = pd.read_csv(f)
        print(f"  {f.name}: {len(d)} rows")
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    if "error" in df.columns:
        df = df[df["error"].isna()]
    for c in ("outlier_rate", "missing_rate", "contam_scale"):
        if c in df.columns:
            df[c] = df[c].round(4)
    return df


def pick(df, filters, what):
    """Select one configuration; list the candidates and stop if ambiguous."""
    for col, val in filters.items():
        if val is None or col not in df.columns:
            continue
        if isinstance(val, str):
            df = df[df[col] == val]
        else:
            df = df[np.isclose(df[col].astype(float), float(val))]
    cols = [c for c in filters if c in df.columns]
    if df.empty:
        print(f"  no {what} rows match the requested filters")
        return None
    combos = df[cols].drop_duplicates().sort_values(cols)
    if len(combos) > 1:
        print(f"\n  several {what} configurations found:\n")
        print(combos.to_string(index=False))
        raise SystemExit("\n  select one with --mode / --scale / --p / --n")
    return df


def save(fig, name, suffix, formats):
    FIGS.mkdir(exist_ok=True)
    for f in formats:
        fig.savefig(FIGS / f"{name}{suffix}.{f}", bbox_inches="tight", dpi=300)
        print(f"  wrote figures/{name}{suffix}.{f}")
    plt.close(fig)


def crossing(x, y, level):
    """First x at which y crosses ``level`` (linear interpolation)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float) - level
    for i in range(len(x) - 1):
        if y[i] == 0:
            return float(x[i])
        if y[i] * y[i + 1] < 0:
            return float(x[i] - y[i] * (x[i + 1] - x[i]) / (y[i + 1] - y[i]))
    return None


def plain_log_axis(ax):
    def minor(v, _):
        if v <= 0:
            return ""
        m = v / 10 ** np.floor(np.log10(v))
        return f"{v:g}" if abs(m - round(m)) < 1e-6 and round(m) in (2, 3, 5) else ""
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(FuncFormatter(minor))


def paired_gap(df, metric, ref="SAEM", rival="Filter_Debiased"):
    rows = []
    for s, sub in df.groupby("contam_scale"):
        a = sub[sub["algo"] == ref].drop_duplicates("rep").set_index("rep")[metric]
        b = sub[sub["algo"] == rival].drop_duplicates("rep").set_index("rep")[metric]
        idx = a.index.intersection(b.index)
        if idx.empty:
            continue
        d = (a.loc[idx] - b.loc[idx]).to_numpy()
        se = d.std(ddof=1) / np.sqrt(d.size) if d.size > 1 else 0.0
        rows.append((s, d.mean(), se, int((d < 0).sum()), d.size))
    return pd.DataFrame(rows, columns=["s", "gap", "se", "wins", "n"])


def paper_figure(df, missing, max_eps, logy, bands, formats, suffix):
    d = df[(df["outlier_rate"] <= max_eps + 1e-9)
           & np.isclose(df["missing_rate"], missing)]
    if d.empty:
        print("  paper figure: no rows for the requested missing rate")
        return
    panels = [("err_mu_l2", r"$\ell_2$ error on $\boldsymbol{\mu}$", "(a)"),
              ("fisher", "Fisher--Rao distance", "(b)"),
              ("fro", "Relative Frobenius error", "(c)")]
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.0))
    for ax, (col, ylabel, tag) in zip(axes, panels):
        for algo in ORDER:
            sub = d[d["algo"] == algo]
            if sub.empty:
                continue
            g = sub.groupby("outlier_rate")[col].agg(["mean", "sem"]).reset_index()
            st = STYLE[algo]
            ax.plot(g["outlier_rate"], g["mean"], label=LABELS[algo], **st)
            if bands:
                ax.fill_between(g["outlier_rate"], g["mean"] - g["sem"],
                                g["mean"] + g["sem"], color=st["color"],
                                alpha=0.15, linewidth=0)
        ax.set_xlabel("Outlier rate")
        ax.set_ylabel(ylabel)
        ax.set_title(tag)
        if logy:
            ax.set_yscale("log")
            plain_log_axis(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    save(fig, "fig_paper", suffix, formats)


def decomposition(df, missing, max_eps, formats, suffix):
    d = df[(df["outlier_rate"] <= max_eps + 1e-9)
           & np.isclose(df["missing_rate"], missing)]
    saem = d[d["algo"] == "SAEM"]
    if saem.empty:
        print("  decomposition: no SAEM rows for the requested missing rate")
        return

    fig, (a, b) = plt.subplots(1, 2, figsize=(7.1, 2.7))
    parts = [("fisher", "Total", "-", "o", "#000000"),
             ("fisher_scale", "Scale term", "--", "s", "#0072B2"),
             ("fisher_shape", "Shape term", "-.", "^", "#D55E00")]
    for col, lab, ls, mk, c in parts:
        g = saem.groupby("outlier_rate")[col].agg(["mean", "sem"]).reset_index()
        a.plot(g["outlier_rate"], g["mean"], label=lab, linestyle=ls,
               marker=mk, color=c, linewidth=1.4)
        a.fill_between(g["outlier_rate"], g["mean"] - g["sem"],
                       g["mean"] + g["sem"], color=c, alpha=0.15, linewidth=0)
    a.set_xlabel("Outlier rate")
    a.set_ylabel("Fisher--Rao distance")
    a.set_title("SAEM: total, scale and shape")
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=3, frameon=False)

    cross = None
    for algo in ("SAEM", "EM_Student_t", "Filter_Debiased"):
        s = d[d["algo"] == algo]
        if s.empty:
            continue
        g = s.groupby("outlier_rate")["cstar"].agg(["mean", "sem"]).reset_index()
        b.plot(g["outlier_rate"], g["mean"], label=LABELS[algo], **STYLE[algo])
        if algo == "SAEM":
            cross = crossing(g["outlier_rate"], g["mean"], 1.0)
    b.axhline(1.0, color="grey", linestyle="--", linewidth=0.8)
    if cross is not None:
        b.axvline(cross, color="black", linestyle=":", linewidth=0.8)
        b.annotate(f"$c^\\star=1$ at {cross:.2f}", xy=(cross, 1.0),
                   xytext=(0.52, 0.12), textcoords="axes fraction", fontsize=7,
                   arrowprops=dict(arrowstyle="-", linewidth=0.5))
    b.set_yscale("log")
    plain_log_axis(b)
    b.set_xlabel("Outlier rate")
    b.set_ylabel(r"$c^\star$ (1 = correct scale)")
    b.set_title("Scale calibration")
    b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=3, frameon=False)
    fig.tight_layout()
    save(fig, "fig_decomposition", suffix, formats)


def regimes(df, metric, formats, suffix):
    fig, (a, b) = plt.subplots(2, 1, figsize=(3.5, 4.5), sharex=True)
    ref_frac = []
    for i, mode in enumerate(sorted(df["mode"].unique())):
        color, mk = MODE_STYLE[i % len(MODE_STYLE)]
        sub = df[df["mode"] == mode]
        label = MODE_LABEL.get(mode, mode.capitalize())
        g = paired_gap(sub, metric)
        a.errorbar(g["s"], g["gap"], yerr=g["se"], color=color, marker=mk,
                   linewidth=1.2, capsize=2, label=label)
        f = (sub[sub["algo"] == "Filter_Debiased"]
             .groupby("contam_scale")["flagged_frac"].mean())
        b.plot(f.index, f.values, color=color, marker=mk, linewidth=1.2, label=label)
        ref_frac.append(float((sub["outlier_rate"] * (1 - sub["missing_rate"])).mean()))

    a.axhline(0.0, color="grey", linewidth=0.8)
    a.set_ylabel("Paired gap: SAEM $-$ detector")
    a.set_title("Negative: SAEM lower", fontsize=7)
    b.axhline(float(np.mean(ref_frac)), color="black", linestyle="--",
              linewidth=0.8, label="Contaminated cells")
    b.set_yscale("log")
    plain_log_axis(b)
    b.set_xlabel("Contamination magnitude $s$")
    b.set_ylabel("Fraction of cells flagged")
    b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3, frameon=False)
    fig.tight_layout()
    save(fig, "fig_regimes", suffix, formats)


def print_summary(main_df, missing_rates, scale_df, max_eps, metric):
    print("\nQuantities read off the figures")
    if main_df is not None:
        d = main_df[main_df["outlier_rate"] <= max_eps + 1e-9]
        for m in missing_rates:
            sub = d[np.isclose(d["missing_rate"], m) & (d["algo"] == "SAEM")]
            if sub.empty:
                continue
            tot = sub.groupby("outlier_rate")["fisher"].mean()
            cs = sub.groupby("outlier_rate")["cstar"].mean()
            shp = sub.groupby("outlier_rate")["fisher_shape"].mean()
            cr = crossing(cs.index, cs.values, 1.0)
            print(f"  missing {m:.2f}: minimum of the total distance at outlier "
                  f"rate {tot.idxmin():.3f} ({tot.min():.3f}); "
                  f"c* at no contamination {cs.iloc[0]:.3f}; "
                  f"c* = 1 at {'%.3f' % cr if cr is not None else 'never'}; "
                  f"total at {tot.index[-1]:.3f}: {tot.iloc[-1]:.3f}; "
                  f"shape term increasing: {bool(np.all(np.diff(shp.values) > 0))}")
    if scale_df is not None:
        for mode in sorted(scale_df["mode"].unique()):
            sub = scale_df[scale_df["mode"] == mode]
            g = paired_gap(sub, metric)
            cr = crossing(g["s"], g["gap"], 0.0)
            print(f"  {mode}: paired gap ({metric}) changes sign at "
                  f"s = {'%.2f' % cr if cr is not None else 'never'}")
            for _, r in g.iterrows():
                print(f"      s={r['s']:<5} gap={r['gap']:+.4f} +- {r['se']:.4f} "
                      f"wins {int(r['wins'])}/{int(r['n'])}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=["paper", "decomposition", "regimes"])
    ap.add_argument("--results-dir", type=Path, default=ROOT / "results")
    ap.add_argument("--curves-glob", default="main_curves_*.csv")
    ap.add_argument("--scale-glob", default="scale_sweep_*.csv")
    ap.add_argument("--mode", default=None, help="amplified or shift")
    ap.add_argument("--scale", type=float, default=None,
                    help="contamination magnitude s of the main curves")
    ap.add_argument("--p", type=int, default=None)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--missing", type=float, default=None,
                    help="missing rate of the paper and decomposition figures "
                         "(default: smallest available)")
    ap.add_argument("--max-eps", type=float, default=0.35)
    ap.add_argument("--metric-regimes", default="fisher",
                    help="fisher (total) or fisher_shape")
    ap.add_argument("--formats", nargs="+", default=["pdf", "png"])
    ap.add_argument("--suffix", default="")
    ap.add_argument("--linear", action="store_true", help="linear y axes")
    ap.add_argument("--no-bands", action="store_true",
                    help="no +-1 standard-error bands")
    args = ap.parse_args()

    todo = set(args.only or ["paper", "decomposition", "regimes"])
    main_df, mrs, scale_df = None, [], None

    if todo & {"paper", "decomposition"}:
        print("main curves")
        raw = read_all(args.results_dir, args.curves_glob)
        if raw is not None:
            main_df = pick(raw, {"mode": args.mode, "contam_scale": args.scale,
                                 "p": args.p, "n": args.n}, "main-curves")
        if main_df is not None:
            main_df = main_df.drop_duplicates(
                subset=["algo", "outlier_rate", "missing_rate", "rep"])
            mrs = sorted(main_df["missing_rate"].unique())
            m = args.missing if args.missing is not None else mrs[0]
            if "paper" in todo:
                paper_figure(main_df, m, args.max_eps, not args.linear,
                             not args.no_bands, args.formats, args.suffix)
            if "decomposition" in todo:
                decomposition(main_df, m, args.max_eps, args.formats, args.suffix)

    if "regimes" in todo:
        print("scale sweep")
        raw = read_all(args.results_dir, args.scale_glob)
        if raw is not None:
            scale_df = pick(raw, {"p": args.p, "n": args.n}, "scale-sweep")
        if scale_df is not None:
            scale_df = scale_df.drop_duplicates(
                subset=["algo", "mode", "contam_scale", "rep"])
            regimes(scale_df, args.metric_regimes, args.formats, args.suffix)

    print_summary(main_df, mrs, scale_df, args.max_eps, args.metric_regimes)


if __name__ == "__main__":
    main()
