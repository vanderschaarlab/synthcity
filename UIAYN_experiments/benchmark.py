from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.benchmark import Benchmarks
import os
import json
from ucimlrepo import fetch_ucirepo
from utils import preprocess, plot_df
from sklearn.model_selection import train_test_split

ds = "adult"
with open("UIAYN_experiments/datasets.json", "r") as f:
    config = json.load(f)
config = config[ds]
dataset = fetch_ucirepo(id=config["id"])
X = dataset.data.features
y = dataset.data.targets

df = preprocess(X=X, y=y, config=config)
df, _ = train_test_split(df, stratify=df["target"], train_size=0.5, random_state=1)

# we have to make sure that the categorical limit corresponds to what we find discrete features in the dataset
# print(df.nunique())
# plot_df(df)


# setup dataloader
X_r = GenericDataLoader(
    data=df,
    sensitive_features=config["sensitive"],
    target_column="target",
    random_state=0,
    train_size=0.8,
)

# load plugin params
hparam_path = f"UIAYN_experiments/hparams/{ds}"
hparams = {}
for file in os.listdir(hparam_path):
    with open(f"{hparam_path}/{file}", "r") as f:
        hparams.update(json.load(f))

# set fasd to tvae plugin and add params
evaluate = [(plugin, plugin, params) for (plugin, params) in hparams.items()]
for i, (name, plugin, params) in enumerate(evaluate):
    if plugin == "fasd":
        params["fasd"] = True
        evaluate[i] = (name, "tvae", params)

    # for small datasets reduce the minimum number of iterations in the generative model and batch size
    # if len(df) < 1000:
    #     params["batch_size"] = 64
    #     if plugin in ["tvae", "fasd", "ctgan", "adsgan"]:
    #         params["n_iter_min"] = 10
    #     if plugin in ["pategan"]:
    #         params["n_teachers"] = 5


# perform benchmarking
score = Benchmarks.evaluate(
    evaluate,
    X_r,
    task_type="classification",
    metrics={
        # "sanity": [
        #     "data_mismatch",
        #     "common_rows_proportion",
        #     "nearest_syn_neighbor_distance",
        #     "close_values_probability",
        #     "distant_values_probability",
        # ],
        "stats": [
            # "jensenshannon_dist",
            # "chi_squared_test",
            # "feature_corr",
            # "inv_kl_divergence",
            # "ks_test",
            # "max_mean_discrepancy",
            # "wasserstein_dist",
            # "prdc",
            # "alpha_precision",
            # "survival_km_distance",
        ],
        "performance": [
            "linear_model",
            # "mlp",
            "xgb",
            # "feat_rank_distance"
        ],
        "detection": [
            "detection_xgb",
            # "detection_mlp",
            # "detection_gmm",
            "detection_linear",
        ],
        "privacy": [
            #   "delta-presence",
            #   "k-anonymization",
            #   "k-map",
            #   "distinct l-diversity",
            #   "identifiability_score",
            # "DomiasMIA_BNAF",
            # "DomiasMIA_KDE",
            # "DomiasMIA_prior",
        ],
        "attack": [  # "data_leakage_linear",
            # "data_leakage_xgb",
            # "data_leakage_mlp"
        ],
    },
    synthetic_size=len(df),
    repeats=1,
    synthetic_cache=False,
    synthetic_reuse_if_exists=False,
    use_metric_cache=False,
)

# save benchmark results
result_path = f"UIAYN_experiments/results/{ds}"
if not os.path.exists(result_path):
    os.makedirs(result_path)
for name, plugin, params in evaluate:
    score[name].to_csv(result_path + f"/{name}.csv")
