# stdlib
from pathlib import Path

# third party
import matplotlib.pyplot as plt
import pandas as pd
from lifelines import KaplanMeierFitter

output = Path("diagrams")
output.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-whitegrid")


def plot_survival_individual(
    scenario: str,
    title: str,
    model: str,
    T: pd.Series,
    E: pd.Series,
    preds: list,
    ci_show: bool = True,
    ci_alpha: float = 0.2,
    show_censors: bool = False,
) -> None:
    gt_kmf = KaplanMeierFitter()
    gt_kmf.fit(T, E, label="Real data")  # t = Timepoints, Rx: 0=censored, 1=event
    ax = gt_kmf.plot(ci_show=ci_show, ci_alpha=ci_alpha, show_censors=show_censors)

    for syn_label, syn_T, syn_E in preds:
        syn_kmf = KaplanMeierFitter()
        syn_kmf.fit(
            syn_T, syn_E, label=syn_label
        )  # t = Timepoints, Rx: 0=censored, 1=event
        ax = syn_kmf.plot(
            ax=ax, ci_show=ci_show, ci_alpha=ci_alpha, show_censors=show_censors
        )

    ax.axvline(T[E == 1].max(), color="r", linestyle="--")  # vertical
    ax.set_xlabel("Days", horizontalalignment="center")
    ax.set_ylabel("Event probability")

    plt.title(f"Dataset: {title}")

    plt.savefig(output / f"individual_kmplot_{scenario}_{title}_{model}.pdf")
    plt.show()


def plot_survival_grouped(
    scenario: str,
    title: str,
    T: pd.Series,
    E: pd.Series,
    preds: list,
    ci_show: bool = True,
    ci_alpha: float = 0.2,
    show_censors: bool = False,
) -> None:
    fig, axs = plt.subplots(
        1, len(preds), figsize=(4 * len(preds), 3), constrained_layout=True
    )

    models = []
    for idx, pred in enumerate(preds):
        ax = axs[idx]
        syn_label, syn_T, syn_E = pred
        gt_kmf = KaplanMeierFitter()
        gt_kmf.fit(T, E, label="Real data")  # t = Timepoints, Rx: 0=censored, 1=event
        ax = gt_kmf.plot(
            ax=ax, ci_show=ci_show, ci_alpha=ci_alpha, show_censors=show_censors
        )

        syn_kmf = KaplanMeierFitter()
        syn_kmf.fit(
            syn_T, syn_E, label=syn_label
        )  # t = Timepoints, Rx: 0=censored, 1=event
        syn_kmf.plot(
            ax=ax, ci_show=ci_show, ci_alpha=ci_alpha, show_censors=show_censors
        )

        ax.axvline(T[E == 1].max(), color="r", linestyle="--")  # vertical
        ax.set_xlabel("", fontsize=14)

        ax.set_xlabel("Time", horizontalalignment="center", fontsize=14)
        ax.set_title(title)
        models.append(syn_label.split(":")[1])

    models_str = "_".join(models).replace(" ", "_")

    axs[0].set_ylabel("Event probability", fontsize=14)

    plt.savefig(output / f"group_kmplot_{scenario}_{title}_{models_str}.png")
    plt.show()
