import os
import sys
import subprocess
import datetime

sys.path.insert(0, os.path.abspath("../src/"))


# -- Project information -----------------------------------------------------
now = datetime.datetime.now()

project = "synthcity"
author = "vanderschaar-lab"
copyright = f"{now.year}, {author}"


subprocess.run(
    [
        "sphinx-apidoc",
        "--ext-autodoc",
        "--ext-doctest",
        "--ext-mathjax",
        "--ext-viewcode",
        "-e",
        "-T",
        "-M",
        "-F",
        "-P",
        "-f",
        "-o",
        "generated",
        "-t",
        "_templates",
        "../src/synthcity/",
    ]
)

emojis = [
    ":rocket:",
    ":key:",
    ":cyclone:",
    ":fire:",
    ":zap:",
    ":hammer:",
    ":boom:",
    ":notebook:",
    ":book:",
    ":adhesive_bandage:",
    ":high_brightness:",
    ":airplane:",
    ":point_down:",
    ":rotating_light:",
]

# with open("../README.md", "rt") as fin:
#    with open("README.md", "wt") as fout:
#        for line in fin:
#            for emoji in emojis:
#                line = line.replace(emoji, "|" + emoji + "|")
#            print(line)
#            fout.write(line)

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "m2r2",
    "sphinxemoji.sphinxemoji",
    "nbsphinx",
    "sphinx_diagrams",
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinx.ext.graphviz",
    "sphinx.ext.inheritance_diagram",
]

autodoc_default_options = {
    "members": True,
    "inherited-members": True,
    "inherit_docstrings": True,
    "private-members": False,
}

add_module_names = False
autosummary_generate = True


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
sphinxemoji_style = "twemoji"


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

nbsphinx_execute = "never"

nbsphinx_prolog = r"""
{% set docname = 'docs/' + env.doc2path(env.docname, base=None) %}
.. only:: html
    .. role:: raw-html(raw)
        :format: html
"""

autodoc_mock_imports = [
    "cloudpickle",
    "ctgan",
    "decaf-synthetic-data",
    "decaf",
    "diffprivlib",
    "dython",
    "fflows",
    "igraph",
    "lifelines",
    "nflows",
    "opacus",
    "optuna",
    "loguru",
    "deepecho",
    "pgmpy",
    "pycox",
    "pykeops",
    "pyod",
    "rdt",
    "redis",
    "scikit-learn",
    "sklearn",
    "pytorch_lightning",
    "scipy",
    "torchtuples",
    "copulas",
    "geomloss",
    "joblib",
    "sdv",
    "shap",
    "tsai",
    "xgboost",
    "xgbse",
]

autodoc_mock_imports = autodoc_mock_imports
