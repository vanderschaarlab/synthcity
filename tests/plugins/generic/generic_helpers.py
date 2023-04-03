# stdlib
from typing import Dict, List, Type

# third party
import pandas as pd

# synthcity absolute
from synthcity.plugins import Plugin
from synthcity.plugins.generic import GenericPlugins as Plugins
from synthcity.utils.serialization import load, save


def generate_fixtures(
    name: str, plugin: Type, plugin_args: Dict = {}, use_dummy_fixtures: bool = False
) -> List:
    def from_api() -> Plugin:
        return Plugins().get(name, **plugin_args)

    def from_module() -> Plugin:
        return plugin(**plugin_args)

    def from_serde() -> Plugin:
        buff = save(plugin(**plugin_args))
        return load(buff)

    if use_dummy_fixtures:
        return [None, None, None]
    else:
        return [from_api(), from_module(), from_serde()]


def get_airfoil_dataset() -> pd.DataFrame:
    df = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat",
        header=None,
        sep="\\t",
    )
    return df
