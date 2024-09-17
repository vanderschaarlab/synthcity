from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.benchmark import Benchmarks
import os
import json
from ucimlrepo import fetch_ucirepo
from utils import preprocess, clear_dir
import optuna
from synthcity.plugins import Plugins
from synthcity.utils.optuna_sample import suggest_all

# find approppriate hparam spaces
hp_desired = {
    "fasd": [
        "fasd_n_units_embedding",
        "n_units_embedding",
        "decoder_n_layers_hidden",
        "decoder_n_units_hidden",
        "encoder_n_layers_hidden",
        "encoder_n_units_hidden",
    ],
    "adsgan": [
        "discriminator_n_layers_hidden",
        "discriminator_n_units_hidden",
        "generator_n_layers_hidden",
        "generator_n_units_hidden",
    ],
    "pategan": [
        "discriminator_n_layers_hidden",
        "discriminator_n_units_hidden",
        "generator_n_layers_hidden",
        "generator_n_units_hidden",
    ],
    "ctgan": [
        "discriminator_n_layers_hidden",
        "discriminator_n_units_hidden",
        "generator_n_layers_hidden",
        "generator_n_units_hidden",
    ],
    "tvae": [
        "n_units_embedding",
        "decoder_n_layers_hidden",
        "decoder_n_units_hidden",
        "encoder_n_layers_hidden",
        "encoder_n_units_hidden",
    ],
}
hp_space = {}
for plugin, params in hp_desired.items():
    if plugin == "fasd":
        plugin_name = "tvae"
    else:
        plugin_name = plugin

    hp_ini = Plugins().get(plugin_name).hyperparameter_space()
    for hp_ in hp_ini:
        if hp_.name not in params:
            hp_ini = [x for x in hp_ini if x != hp_]

    hp_space[plugin] = hp_ini

# load data
ds = "adult"
with open(f"UIAYN_experiments/datasets.json", "r") as f:
    config = json.load(f)
config = config[ds]
dataset = fetch_ucirepo(id=config["id"])
X = dataset.data.features
y = dataset.data.targets

df = preprocess(X=X, y=y, config=config)
# df = df[:100]

# setup dataloader
X_r = GenericDataLoader(
    data=df,
    sensitive_features=config["sensitive"],
    target_column="target",
    random_state=0,
    train_size=0.8,
)

best_params = {}
for plugin, hparams in hp_space.items():

    def objective(trial: optuna.Trial):
        params = suggest_all(trial, hparams)
        ID = f"trial_{trial.number}"

        # ensure smaller batch size and training epochs are allowed for small datasets
        if len(df) < 1000:
            params["batch_size"] = 64
            if plugin in ["tvae", "fasd", "ctgan", "adsgan"]:
                params["n_iter_min"] = 10
            if plugin in ["pategan"]:
                params["n_teachers"] = 5

        try:
            report = Benchmarks.evaluate(
                [(ID, plugin_name, params)],
                X_r,
                repeats=1,
                metrics={
                    "performance": [
                        "linear_model",
                        "mlp",
                        "xgb",
                    ],
                },  # DELETE THIS LINE FOR ALL METRICS
            )
        except Exception as e:  # invalid set of params
            print(f"{type(e).__name__}: {e}")
            print(params)
            raise optuna.TrialPruned()
        score = report[ID].query('direction == "maximize"')["mean"].mean()
        # average score across all metrics with direction="minimize"
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=32)
    best_params[plugin] = study.best_params

    # after tuning a model, clear the workspace to free up space for the next one
    clear_dir("workspace")
# save best hparams in separate files
hparam_path = f"UIAYN_experiments/hparams/{ds}"
if not os.path.exists(hparam_path):
    os.makedirs(hparam_path)

for plugin, _ in best_params.items():
    with open(f"{hparam_path}/hparams_{plugin}.json", "w") as file:
        json.dump({plugin: best_params[plugin]}, file)
