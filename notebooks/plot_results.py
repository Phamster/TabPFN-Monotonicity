"""
Produces a publication-ready PoM summary figure from results/pom_results.csv.
Run after experiments/run_benchmark.py.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
RESULTS_CSV = os.path.join(RESULTS_DIR, "pom_results.csv")

plt.rcParams.update({
    "font.family":      "serif",
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
})

COLORS = {
    "XGBoost":       "#2E6B9E",
    "TabPFN_raw":    "#C17A33",
    "TabPFN_adapter":"#3E8C8C",
}


def plot_pom_summary(df: pd.DataFrame, metric: str = "auc"):
    col_mean = f"pom_{metric}_pct"
    col_lo   = f"pom_{metric}_ci95_lo"
    col_hi   = f"pom_{metric}_ci95_hi"

    # clean up dataset labels (strip the comparison annotation)
    df = df.copy()
    df["dataset_short"] = df["dataset"].str.split(" \[").str[0]

    models = df["model"].unique()
    datasets = df["dataset_short"].unique()
    n_d = len(datasets)
    n_m = len(models)

    fig, ax = plt.subplots(figsize=(max(8, n_d * 1.4), 5))

    x = np.arange(n_d)
    width = 0.7 / n_m

    for mi, model in enumerate(models):
        sub = df[df["model"] == model].set_index("dataset_short")
        means, lo_err, hi_err = [], [], []
        for ds in datasets:
            if ds in sub.index:
                means.append(sub.loc[ds, col_mean])
                lo_err.append(sub.loc[ds, col_mean] - sub.loc[ds, col_lo])
                hi_err.append(sub.loc[ds, col_hi] - sub.loc[ds, col_mean])
            else:
                means.append(np.nan)
                lo_err.append(0)
                hi_err.append(0)

        offset = (mi - n_m / 2 + 0.5) * width
        color  = COLORS.get(model, f"C{mi}")
        bars   = ax.bar(x + offset, means, width * 0.9,
                        color=color, alpha=0.85, label=model)
        ax.errorbar(x + offset, means,
                    yerr=[lo_err, hi_err],
                    fmt="none", color="black", capsize=3, linewidth=1)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel(f"Price of Monotonicity — {metric.upper()} (%)", fontsize=10)
    ax.set_title(f"Price of Monotonicity: constrained vs unconstrained  [{metric.upper()}]",
                 fontsize=11, fontweight="bold")

    legend_patches = [mpatches.Patch(color=COLORS.get(m, f"C{i}"), label=m)
                      for i, m in enumerate(models)]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=8)

    note = ("Bars show mean PoM (%) across 500 bootstrap samples.\n"
            "Error bars = 95% CI. Positive = constrained model costs accuracy.")
    ax.text(0.01, 0.98, note, transform=ax.transAxes,
            va="top", ha="left", fontsize=7.5, color="gray")

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, f"pom_summary_{metric}.pdf")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"Figure saved → {out}")
    plt.show()


if __name__ == "__main__":
    if not os.path.exists(RESULTS_CSV):
        print(f"No results file found at {RESULTS_CSV}.")
        print("Run experiments/run_benchmark.py first.")
    else:
        df = pd.read_csv(RESULTS_CSV)
        plot_pom_summary(df, metric="auc")
        plot_pom_summary(df, metric="brier")
