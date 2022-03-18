# stdlib
from typing import List, Type

# synthcity absolute
from synthcity.plugins import Plugin, Plugins
from synthcity.utils.serialization import load, save


def generate_fixtures(name: str, plugin: Type) -> List:
    def from_api() -> Plugin:
        return Plugins().get(name)

    def from_module() -> Plugin:
        return plugin()

    def from_serde() -> Plugin:
        buff = save(plugin())
        return load(buff)

    return [from_api(), from_module(), from_serde()]
