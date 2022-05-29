# stdlib
import glob
from os.path import basename, dirname, isfile, join

# synthcity absolute
from synthcity.plugins.core.plugin import Plugin, PluginLoader  # noqa: F401,E402

categories = ["generic", "survival_analysis", "time_series"]
plugins = {}

for cat in categories:
    plugins[cat] = glob.glob(join(dirname(__file__), cat, "plugin*.py"))


class Plugins(PluginLoader):
    def __init__(self, category: str = "generic") -> None:
        super().__init__(plugins[category], Plugin)


__all__ = [basename(f)[:-3] for f in plugins[cat] for cat in plugins if isfile(f)] + [
    "Plugins",
    "Plugin",
]
