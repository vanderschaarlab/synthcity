# stdlib
import glob
from os.path import basename, dirname, isfile, join

# synthcity absolute
from synthcity.plugins.core.plugin import Plugin, PluginLoader  # noqa: F401,E402

plugins = glob.glob(join(dirname(__file__), "plugin*.py"))


class PrivacyPlugins(PluginLoader):
    def __init__(self) -> None:
        super().__init__(plugins, Plugin, ["privacy"])


__all__ = [basename(f)[:-3] for f in plugins if isfile(f)] + [
    "PrivacyPlugins",
]
