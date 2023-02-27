# stdlib
from typing import Dict, List, Type

# synthcity absolute
from synthcity.plugins import Plugin, Plugins
from synthcity.utils.serialization import load, save


def generate_fixtures(name: str, plugin: Type, plugin_args: Dict = {}) -> List:
    def from_api() -> Plugin:
        return Plugins().get(name, **plugin_args)

    def from_module() -> Plugin:
        return plugin(**plugin_args)

    def from_serde() -> Plugin:
        buff = save(plugin(**plugin_args))
        return load(buff)

    return [from_api(), from_module(), from_serde()]
