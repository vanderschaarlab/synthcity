# stdlib
import glob
from os.path import basename, dirname, isfile, join

# synthcity absolute
from synthcity.plugins.core.base import Plugin  # noqa: F401,E402
from synthcity.plugins.core.base_plugin import PluginLoader

plugins = glob.glob(join(dirname(__file__), "plugin*.py"))


class Plugins(PluginLoader):
    def __init__(self) -> None:
        super().__init__(plugins, Plugin)


__all__ = [basename(f)[:-3] for f in plugins if isfile(f)] + [
    "Plugins",
    "Plugin",
]
