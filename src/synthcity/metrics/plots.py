# Adapted from https://github.com/daanknoors/synthetic_data_generation
# stdlib
from typing import Any

# third party
import numpy as np
import pandas as pd
import seaborn as sns
from pydantic import validate_arguments

# synthcity absolute
from synthcity.metrics.statistical import (
    evaluate_avg_jensenshannon_stats,
    evaluate_feature_correlation,
    evaluate_feature_correlation_stats,
)

COLOR_PALETTE = ["#393e46", "#ff5722", "#d72323"]
LABELS = ["gt", "syn"]


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def plot_marginal_comparison(
    plt: Any, X_gt: pd.DataFrame, X_syn: pd.DataFrame, normalize: bool = True
) -> None:
    stats_, stats_gt, stats_syn = evaluate_avg_jensenshannon_stats(X_gt, X_syn)

    column_names = stats_gt.keys()
    fig, ax = plt.subplots(len(column_names), 1, figsize=(8, len(column_names) * 4))

    for idx, col in enumerate(column_names):

        column_value_counts_original = stats_gt[col]
        column_value_counts_synthetic = stats_syn[col]

        bar_position = np.arange(len(column_value_counts_original.values))
        bar_width = 0.35

        # with small column cardinality plot original distribution as bars, else plot as line
        if len(column_value_counts_original.values) <= 25:
            ax[idx].bar(
                x=bar_position,
                height=column_value_counts_original.values,
                color=COLOR_PALETTE[0],
                label=LABELS[0],
                width=bar_width,
            )
        else:
            ax[idx].plot(
                bar_position + bar_width,
                column_value_counts_original.values,
                marker="o",
                markersize=3,
                color=COLOR_PALETTE[0],
                linewidth=2,
                label=LABELS[0],
            )

        # synthetic distribution
        ax[idx].bar(
            x=bar_position + bar_width,
            height=column_value_counts_synthetic.values,
            color=COLOR_PALETTE[1],
            label=LABELS[1],
            width=bar_width,
        )

        ax[idx].set_xticks(bar_position + bar_width / 2)
        if len(column_value_counts_original.values) <= 25:
            ax[idx].set_xticklabels(column_value_counts_original.keys(), rotation=25)
        else:
            ax[idx].set_xticklabels("")
        title = (
            r"$\bf{"
            + col.replace("_", "\\_")
            + "}$"
            + "\n jensen-shannon distance: {:.2f}".format(stats_[col])
        )
        ax[idx].set_title(title)
        if normalize:
            ax[idx].set_ylabel("Probability")
        else:
            ax[idx].set_ylabel("Count")

        ax[idx].legend()


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def plot_associations_comparison(
    plt: Any,
    X_gt: pd.DataFrame,
    X_syn: pd.DataFrame,
    nom_nom_assoc: str = "theil",
    nominal_columns: str = "auto",
) -> None:
    stats_gt, stats_syn = evaluate_feature_correlation_stats(
        X_gt, X_syn, nom_nom_assoc=nom_nom_assoc, nominal_columns=nominal_columns
    )
    pcd = evaluate_feature_correlation(
        X_gt, X_syn, nom_nom_assoc=nom_nom_assoc, nominal_columns=nominal_columns
    )

    fig, ax = plt.subplots(1, 2, figsize=(12, 10))
    cbar_ax = fig.add_axes([0.91, 0.3, 0.01, 0.4])

    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Original
    heatmap_original = sns.heatmap(
        stats_gt,
        ax=ax[0],
        square=True,
        annot=False,
        center=0,
        linewidths=0,
        cmap=cmap,
        xticklabels=True,
        yticklabels=True,
        cbar_kws={"shrink": 0.8},
        cbar_ax=cbar_ax,
        fmt=".2f",
    )
    ax[0].set_title(LABELS[0] + "\n")

    # Synthetic
    sns.heatmap(
        stats_syn,
        ax=ax[1],
        square=True,
        annot=False,
        center=0,
        linewidths=0,
        cmap=cmap,
        xticklabels=True,
        yticklabels=False,
        cbar=False,
        cbar_kws={"shrink": 0.8},
    )
    ax[1].set_title(
        LABELS[1] + "\n" + "pairwise correlation distance: {}".format(round(pcd, 4))
    )

    cbar = heatmap_original.collections[0].colorbar
    cbar.ax.tick_params(labelsize=10)
    fig.tight_layout()
