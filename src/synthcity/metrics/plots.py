# stdlib
from typing import Any

# third party
import numpy as np
import pandas as pd
from pydantic import validate_arguments
from sklearn.manifold import TSNE

# synthcity absolute
from synthcity.metrics.eval_statistical import JensenShannonDistance
from synthcity.plugins.core.dataloader import DataLoader

COLOR_PALETTE = ["#2b2d42", "#d90429"]
LABELS = ["real", "syn"]


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def plot_marginal_comparison(
    plt: Any, X_gt: DataLoader, X_syn: DataLoader, normalize: bool = True
) -> None:
    evaluator = JensenShannonDistance(n_histogram_bins=10)
    stats_, stats_gt, stats_syn = evaluator._evaluate_stats(X_gt, X_syn)

    column_names = stats_gt.keys()
    plots_cnt = len(column_names)
    row_len = 2
    fig, ax = plt.subplots(
        int(np.ceil(plots_cnt / row_len)), row_len, figsize=(14, plots_cnt * 3)
    )
    fig.subplots_adjust(hspace=1)
    if plots_cnt % row_len != 0:
        fig.delaxes(ax[-1][-1])

    for idx, col in enumerate(column_names):
        row_idx = int(idx / row_len)
        col_idx = idx % row_len

        local_ax = ax[row_idx][col_idx]
        column_value_counts_original = stats_gt[col]
        column_value_counts_synthetic = stats_syn[col]

        bar_position = np.arange(len(column_value_counts_original.values))
        bar_width = 0.4

        # real distribution
        local_ax.bar(
            x=bar_position,
            height=column_value_counts_original.values,
            color=COLOR_PALETTE[0],
            label=LABELS[0],
            width=bar_width,
        )

        # synthetic distribution
        local_ax.bar(
            x=bar_position + bar_width,
            height=column_value_counts_synthetic.values,
            color=COLOR_PALETTE[1],
            label=LABELS[1],
            width=bar_width,
        )

        local_ax.set_xticks(bar_position + bar_width / 2)
        local_ax.set_xticklabels(column_value_counts_original.keys(), rotation=90)

        title = (
            r"$\bf{"
            + col.replace("_", "\\_")
            + "}$"
            + "\n jensen-shannon distance: {:.2f}".format(stats_[col])
        )
        local_ax.set_title(title)
        if normalize:
            local_ax.set_ylabel("Probability")
        else:
            local_ax.set_ylabel("Count")

        local_ax.legend()


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def plot_tsne(
    plt: Any,
    X_gt: DataLoader,
    X_syn: DataLoader,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    tsne_gt = TSNE(n_components=2, random_state=0, learning_rate="auto", init="pca")
    proj_gt = pd.DataFrame(tsne_gt.fit_transform(X_gt.dataframe()))

    tsne_syn = TSNE(n_components=2, random_state=0, learning_rate="auto", init="pca")
    proj_syn = pd.DataFrame(tsne_syn.fit_transform(X_syn.dataframe()))

    ax.scatter(x=proj_gt[0], y=proj_gt[1], s=10, label="Real data")
    ax.scatter(x=proj_syn[0], y=proj_syn[1], s=10, label="Synthetic data")

    ax.legend(loc="upper left")
    ax.set_ylabel("t-SNE plot")
