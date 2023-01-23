# stdlib
import glob
from os.path import basename, dirname, isfile, join

# third party
from pydantic import validate_arguments

# synthcity absolute
from synthcity.plugins.core.plugin import Plugin, PluginLoader  # noqa: F401,E402

def_categories = [
    "generic",
    "privacy",
    "survival_analysis",
    "time_series",
    "domain_adaptation",
]
plugins = {}

for cat in def_categories:
    plugins[cat] = glob.glob(join(dirname(__file__), cat, "plugin*.py"))


class Plugins(PluginLoader):
    @validate_arguments
    def __init__(self, categories: list = def_categories) -> None:
        plugins_to_use = []
        for cat in categories:
            plugins_to_use.extend(plugins[cat])

        super().__init__(plugins_to_use, Plugin, categories)


__all__ = [basename(f)[:-3] for f in plugins[cat] for cat in plugins if isfile(f)] + [
    "Plugins",
    "Plugin",
]
