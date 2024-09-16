import pandas as pd
import os
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec


def format_table(df, ds, metrics, save_path="UIAYN_experiments/results_formatted"):
    save_path = save_path + f"/{ds}"
    # Group the DataFrame by 'name'
    grouped = df.groupby("name")

    # Create a dictionary to hold the DataFrames
    dfs = {}

    # Process each group
    for name, group in grouped:
        # Pivot the table
        pivot = group.pivot(index="model", columns="ds", values=["mean", "stddev"])

        # Round the values to three decimal places
        mean_rounded = pivot["mean"].round(3)
        stddev_rounded = pivot["stddev"].round(3)

        # Format for LaTeX
        formatted_df = (
            "$"
            + mean_rounded.applymap(lambda x: f"{x:.3f}")
            + "_{\pm "
            + stddev_rounded.applymap(lambda x: f"{x:.3f}")
            + "}$"
        )

        # Store in the dictionary
        dfs[name] = formatted_df

    result_path = f"{save_path}/"
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    fid_table = []
    ut_table = []
    priv_table = []
    for metric, table in dfs.items():

        # cannot attach AIA metrics since these rows do not exist for each column in the table
        # they are dataset specific
        # if "leakage" in metric:
        #     metrics.update(
        #         {
        #             metric: {
        #                 "name": "AIA-" + metric.split(".")[-1],
        #                 "category": "privacy",
        #             }
        #         }
        #     )

        # only use the metrics which are contained in the dictionary
        if metric in metrics.keys():
            table = table.reset_index()
            # add the statistic to the table
            table["stat"] = [""] * len(table)
            table = table[["stat", "model", ds]]
            empty_row = pd.DataFrame([[""] * len(table.columns)], columns=table.columns)
            table = pd.concat([empty_row, table], ignore_index=True)
            table.iloc[0, 0] = r"\textbf{" + metrics[metric]["name"] + "}"

            # attach to own metric category table
            if metrics[metric]["category"] == "fidelity":
                fid_table.append(table)
            elif metrics[metric]["category"] == "utility":
                ut_table.append(table)
            else:
                priv_table.append(table)

    for tab, cat in zip(
        [fid_table, ut_table, priv_table], ["fidelity", "utility", "privacy"]
    ):
        fin_tab = pd.concat(tab, ignore_index=True)
        fin_tab.to_csv(f"{result_path}/{cat}_table.csv", index=False)


def format_plot(df, ds, metrics):
    # select data for current dataset
    data = df[df["ds"] == ds]

    # attach dataset specific AIA metric
    for _, row in data[
        data["name"].str.contains("leakage", case=False, na=False)
    ].iterrows():
        metrics.update(
            {
                row["name"]: {
                    "name": "AIA-" + row["name"].split(".")[-1],
                    "category": "privacy",
                }
            }
        )

    # remove all non used metrics
    data = data[data["name"].apply(lambda x: x in metrics.keys())]

    # add metrics on real set as Real model
    # real_metrics = (
    #     data[data["name"].str.endswith(".gt")]
    #     .groupby("name")
    #     .agg(
    #         {
    #             "mean": "mean",
    #             "direction": "first",
    #             "stddev": "mean",  # For numerical columns, take the mean
    #             "model": "first",
    #             "ds": "first",  # For non-numericals, take first value
    #         }
    #     )
    # )
    # real_metrics = real_metrics.reset_index()
    # real_metrics["model"] = "real"
    # real_metrics["name"] = real_metrics["name"].str.replace(
    #     r"\.[^.]+$", ".syn_ood", regex=True
    # )
    # data = pd.concat([data, real_metrics], ignore_index=True)
    # # dont forget to remove the original metrics so it doesnt get plotted
    # metrics = {k: v for k, v in metrics.items() if not k.endswith(".gt")}

    # map the names to interpretable names
    data["name"] = data["name"].map(lambda x: metrics.get(x, {}).get("name", x))

    # revert wrong direction of feature importance (for utility, higher is better)
    data.loc[data["name"] == "Feature Importance", "direction"] = "maximize"

    # Categories (x-axis labels)
    categories = list(data["model"].unique())
    colors = sns.color_palette("Greys", len(categories))

    # Create the grid layout
    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1])

    # create the subplots for fidelity, utility, and privacy within a single figure
    for j, cat in enumerate(["fidelity", "utility", "privacy"]):
        cur_metrics = [
            value["name"] for value in metrics.values() if value["category"] == cat
        ]
        cur_data = data[data["name"].isin(cur_metrics)]
        subplot(
            fig=fig,
            grid=gs,
            cur_data=cur_data,
            cur_metrics=cur_metrics,
            n_subplot=j,
            categories=categories,
            colors=colors,
        )

    plt.suptitle(ds)
    plt.tight_layout()
    plt.savefig(f"UIAYN_experiments/results_formatted/{ds}/plot_{ds}.png")
    plt.show()


def subplot(fig, grid, cur_data, cur_metrics, n_subplot, categories, colors):
    # this function creates a subplot for each metric category
    gs_ = gridspec.GridSpecFromSubplotSpec(
        1, len(cur_metrics), subplot_spec=grid[n_subplot], wspace=0.4
    )

    for i, metric in enumerate(cur_metrics):
        # delete the Real as category if not relevant for cur metric
        cats = categories.copy()
        if "real" in categories and (
            not "real" in cur_data[cur_data["name"] == metric]["model"].to_list()
        ):
            cats.remove("real")

        ax = plt.Subplot(fig, gs_[i])
        y = (cur_data[cur_data["name"] == metric]["mean"]).reset_index(drop=True)
        print(categories)
        print(cats)
        print(y)
        sns.barplot(
            x=cats,
            y=y,
            ax=ax,
            palette=colors,
        )

        # set barcolor of best and worst bar to highlight best performers
        if (cur_data[cur_data["name"] == metric]["direction"] == "maximize").any():
            best_bar = cats[y.idxmax()]
            worst_bar = cats[y.idxmin()]
            tick = "\u2191"
        else:
            best_bar = cats[y.idxmin()]
            worst_bar = cats[y.idxmax()]
            tick = "\u2193"
        for bar, cat in zip(ax.patches, cats):
            if cat == best_bar:
                bar.set_facecolor("green")
            elif cat == worst_bar:
                bar.set_facecolor("red")

        ax.set_title(f"{metric} {tick}", fontsize=10)  # Upward arrow
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

        # set y bounds for AUC metrics
        if any(
            string in metric
            for string in ["Distinguishability", "XGB", "LR", "MLP", "AIA"]
        ):
            # additional check, only set bounds if all values are in these bounds
            if not (any(y < 0.5) or any(y > 1)):
                ax.set_ybound(0.5, 1)
        ax.set_ylabel("")
        fig.add_subplot(ax)


if __name__ == "__main__":

    ds = "student"

    metrics = {
        "stats.jensenshannon_dist.marginal": {"name": "JS", "category": "fidelity"},
        "stats.max_mean_discrepancy.joint": {"name": "MMD", "category": "fidelity"},
        # "stats.wasserstein_dist.joint": {"name": "Wasserstein", "category": "fidelity"},
        "stats.alpha_precision.delta_precision_alpha_OC": {
            "name": "a-precision",
            "category": "fidelity",
        },
        "stats.alpha_precision.delta_coverage_beta_OC": {
            "name": "B-recall",
            "category": "fidelity",
        },
        "detection.detection_xgb.mean": {
            "name": "Distinguishability",
            "category": "fidelity",
        },
        "performance.xgb.syn_ood": {"name": "XGB", "category": "utility"},
        # "performance.xgb.gt": {"name": "XGB_real", "category": "utility"},
        "performance.mlp.syn_ood": {"name": "MLP", "category": "utility"},
        # "performance.mlp.gt": {"name": "MLP_real", "category": "utility"},
        "performance.linear_model.syn_ood": {"name": "LR", "category": "utility"},
        # "performance.linear_model.gt": {"name": "LR_real", "category": "utility"},
        "performance.feat_rank_distance.corr": {
            "name": "Feature Importance",
            "category": "utility",
        },
        "privacy.delta-presence.score": {
            "name": "d-presence",
            "category": "privacy",
        },
        "privacy.k-anonymization.syn": {
            "name": "k-anonymization",
            "category": "privacy",
        },
        "privacy.k-map.score": {"name": "k-map", "category": "privacy"},
        "privacy.distinct l-diversity.syn": {
            "name": "l-diversity",
            "category": "privacy",
        },
        "privacy.identifiability_score.score_OC": {
            "name": "Identifiability",
            "category": "privacy",
        },
        "stats.alpha_precision.authenticity_OC": {
            "name": "Authenticity",
            "category": "privacy",
        },
    }

    # loop through the datasets in results
    results = []
    curwd = "UIAYN_experiments/results"
    for ds_dir in os.listdir(curwd):
        path = f"{curwd}/{ds_dir}"
        for filename in os.listdir(path):
            results.append(
                pd.read_csv(f"{path}/{filename}").assign(
                    model=filename.split(".")[0], ds=ds_dir
                )
            )

    results = pd.concat(results)
    results = results.rename({"Unnamed: 0": "name"}, axis=1)
    results = results[["name", "mean", "direction", "stddev", "model", "ds"]]
    df = results

    format_table(df, ds=ds, metrics=metrics)
    format_plot(df, ds, metrics=metrics)
