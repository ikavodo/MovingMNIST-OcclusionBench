import random
import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = Path(__file__).parent


def get_project_root():
    return PROJECT_ROOT


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def take_frames(video, k=5):
    """Select k evenly spaced frames from a video tensor [T,...]."""
    T = video.shape[0]
    if k >= T:
        return video, torch.arange(T)
    idx = torch.linspace(0, T - 1, steps=k).long()
    return video[idx], idx


# ----------------------------------------------------------------------
# Plotting utilities
# ----------------------------------------------------------------------

def _mean_ci(x, z=1.96):
    x = pd.Series(x).dropna().to_numpy()
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    m = float(np.mean(x))
    if len(x) == 1:
        return m, m, m
    se = float(np.std(x, ddof=1) / math.sqrt(len(x)))
    return m, m - z * se, m + z * se


def _aggregate_with_ci(df, group_cols, value_col):
    rows = []
    for keys, g in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        mean, lo, hi = _mean_ci(g[value_col])
        row = {col: key for col, key in zip(group_cols, keys)}
        row.update({
            value_col: mean,
            f"{value_col}_lo": lo,
            f"{value_col}_hi": hi,
            "n": len(g),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def reliability_table(df, prob_col="video_conf", correct_col="video_correct", n_bins=10):
    x = df[[prob_col, correct_col]].dropna().copy()
    x["bin"] = pd.cut(
        x[prob_col],
        bins=np.linspace(0.0, 1.0, n_bins + 1),
        include_lowest=True,
        duplicates="drop",
    )
    out = (
        x.groupby("bin", observed=False)
        .agg(
            conf=(prob_col, "mean"),
            acc=(correct_col, "mean"),
            count=(correct_col, "size"),
        )
        .reset_index()
    )
    return out


def expected_calibration_error(df, prob_col="video_conf", correct_col="video_correct", n_bins=10):
    tab = reliability_table(df, prob_col, correct_col, n_bins)
    n = tab["count"].sum()
    if n == 0:
        return np.nan
    return float(np.sum((tab["count"] / n) * np.abs(tab["acc"] - tab["conf"])))


def plot_semantic_loss_curves(df, out_dir, coverage_col="coverage_target", occ_col="occ"):
    """
    Plot average semantic feature loss vs coverage, averaged over all conv layers.
    """
    sem_cols = [c for c in df.columns if c.startswith("sem_")]
    if not sem_cols:
        return

    # Create a new column averaging the semantic losses
    df_mean = df.copy()
    df_mean["sem_mean"] = df_mean[sem_cols].mean(axis=1)

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    agg = _aggregate_with_ci(df_mean, [occ_col, coverage_col], "sem_mean")
    for occ, g in agg.groupby(occ_col):
        g = g.sort_values(coverage_col)
        ax.plot(g[coverage_col], g["sem_mean"], marker="o", label=occ)
        ax.fill_between(
            g[coverage_col],
            g["sem_mean_lo"],
            g["sem_mean_hi"],
            alpha=0.2
        )

    ax.set_title("Average semantic feature loss vs occlusion coverage")
    ax.set_xlabel("Target occlusion coverage")
    ax.set_ylabel("Mean cosine distance (conv1+conv2+conv3)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.savefig(Path(out_dir) / "semantic_loss_vs_coverage.png", dpi=160)
    plt.close(fig)


def plot_semantic_vs_accuracy(df, out_dir, occ_col="occ"):
    sem_cols = [c for c in df.columns if c.startswith("sem_")]
    if not sem_cols:
        return

    sem = "sem_cos_conv3" if "sem_cos_conv3" in sem_cols else sem_cols[-1]

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    for occ, g in df.groupby(occ_col):
        ax.scatter(g[sem], g["video_correct"], alpha=0.15, s=12, label=occ)

    ax.set_xlabel(sem)
    ax.set_ylabel("Video correct (0/1)")
    ax.set_title(f"Accuracy vs semantic feature loss ({sem})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(Path(out_dir) / "semantic_loss_vs_accuracy.png", dpi=160)
    plt.close(fig)


# common.py (or plotting.py) — add these functions

def plot_coverage_summary(df, out_dir, occ_col="occ", coverage_col="coverage_target"):
    """Main coverage curves for key metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    axes = axes.ravel()

    main_metrics = [
        ("video_correct", "Video accuracy"),
        ("video_conf", "Video confidence"),
        ("video_nll", "Video NLL"),
        ("video_brier", "Video Brier"),
    ]

    for ax, (metric, title) in zip(axes, main_metrics):
        agg = _aggregate_with_ci(df, [occ_col, coverage_col], metric)
        for occ, g in agg.groupby(occ_col):
            g = g.sort_values(coverage_col)
            ax.plot(g[coverage_col], g[metric], marker="o", label=occ)
            ax.fill_between(
                g[coverage_col],
                g[f"{metric}_lo"],
                g[f"{metric}_hi"],
                alpha=0.2,
            )
        ax.set_title(title)
        ax.set_xlabel("Target occlusion coverage")
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle("Occlusion robustness summary", fontsize=14)
    fig.savefig(out_dir / "coverage_summary.png", dpi=160)
    plt.close(fig)


def plot_confidence_vs_accuracy(df, out_dir, occ_col="occ", coverage_col="coverage_target"):
    """Confidence vs accuracy scatter with coverage labels."""
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    agg_acc = _aggregate_with_ci(df, [occ_col, coverage_col], "video_correct")
    agg_conf = _aggregate_with_ci(df, [occ_col, coverage_col], "video_conf")
    merged = agg_acc.merge(agg_conf, on=[occ_col, coverage_col], suffixes=("_acc", "_conf"))

    for occ, g in merged.groupby(occ_col):
        g = g.sort_values(coverage_col)
        ax.plot(g["video_conf"], g["video_correct"], marker="o", label=occ)
        for _, row in g.iterrows():
            ax.annotate(f"{row[coverage_col]:.1f}", (row["video_conf"], row["video_correct"]), fontsize=8)

    lims = [0, 1]
    ax.plot(lims, lims, linestyle="--")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Mean confidence")
    ax.set_ylabel("Mean accuracy")
    ax.set_title("Confidence vs accuracy by mask family\n(labels are occlusion coverage)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(out_dir / "confidence_vs_accuracy.png", dpi=160)
    plt.close(fig)


def plot_structure_vs_performance(df, out_dir, occ_col="occ"):
    """Scatter plots for structure metrics vs accuracy."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    ax = axes[0]
    for occ, g in df.groupby(occ_col):
        ax.scatter(
            g["mask_largest_component_frac"],
            g["video_correct"],
            alpha=0.15,
            s=12,
            label=occ,
        )
    ax.set_xlabel("Largest occluder component fraction")
    ax.set_ylabel("Video correct (0/1)")
    ax.set_title("Performance vs largest occluder chunk")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for occ, g in df.groupby(occ_col):
        ax.scatter(
            g["mask_n_components"],
            g["video_correct"],
            alpha=0.15,
            s=12,
            label=occ,
        )
    ax.set_xlabel("Number of occluder components")
    ax.set_ylabel("Video correct (0/1)")
    ax.set_title("Performance vs mask fragmentation")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.savefig(out_dir / "structure_vs_performance.png", dpi=160)
    plt.close(fig)


def plot_reliability_diagrams(df, out_dir, occ_col="occ"):
    """Reliability diagrams per occlusion family, plus ECE table."""
    families = sorted(df[occ_col].dropna().unique().tolist())
    n = len(families)
    ncols = min(2, max(1, n))
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), constrained_layout=True)
    if n == 1:
        axes = np.array([axes])
    axes = np.array(axes).reshape(-1)

    ece_rows = []
    for ax, occ in zip(axes, families):
        g = df[df[occ_col] == occ].copy()
        tab = reliability_table(g, prob_col="video_conf", correct_col="video_correct", n_bins=10)
        ece = expected_calibration_error(g, prob_col="video_conf", correct_col="video_correct", n_bins=10)
        ece_rows.append({"occ": occ, "ece": ece})

        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.plot(tab["conf"], tab["acc"], marker="o")
        ax.set_title(f"{occ} (ECE={ece:.3f})")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    for j in range(len(families), len(axes)):
        axes[j].axis("off")

    fig.savefig(out_dir / "reliability_by_family.png", dpi=160)
    plt.close(fig)


def plot_results(
        df,
        out_dir="outputs/plots",
        coverage_col="coverage_target",
        occ_col="occ",
):
    """Orchestrate all plots and tables."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ensure boolean columns are float for plotting
    bool_cols = ["video_correct"]
    for c in bool_cols:
        if c in df.columns:
            df[c] = df[c].astype(float)

    # Generate all plots
    plot_coverage_summary(df, out_dir, occ_col, coverage_col)
    plot_confidence_vs_accuracy(df, out_dir, occ_col, coverage_col)
    plot_structure_vs_performance(df, out_dir, occ_col)
    plot_reliability_diagrams(df, out_dir, occ_col)
    plot_semantic_loss_curves(df, out_dir, coverage_col, occ_col)  # from existing utils
    plot_semantic_vs_accuracy(df, out_dir, occ_col)  # from existing utils

    return {
        "n_rows": len(df),
        "families": sorted(df[occ_col].dropna().unique().tolist()),
        "out_dir": str(out_dir),
    }
