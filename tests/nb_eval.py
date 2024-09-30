# stdlib
from pathlib import Path
from time import time
from typing import List

# third party
import click
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

workspace = Path(__file__).parents[0] / "workspace"
workspace.mkdir(parents=True, exist_ok=True)


def run_notebook(notebook_path: Path, timeout: int) -> None:
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)

    proc = ExecutePreprocessor(timeout=timeout)
    # Will raise on cell error
    proc.preprocess(nb, {"metadata": {"path": workspace}})


try:
    # synthcity absolute
    from synthcity.plugins.core.models.tabular_goggle import TabularGoggle  # noqa: F401

    goggle_disabled = False
except ImportError:
    goggle_disabled = True

all_tests = [
    "basic_examples",
    "benchmarks",
    "survival_analysis",
    "time_series",
    "time_series_data_preparation",
    "hyperparameter_optimization",
    "plugin_adsgan",
    "plugin_ctgan",
    "plugin_nflow",
    "plugin_tvae",
    "plugin_radialgan",
    "plugin_arf",
    "plugin_bayesian_network",
    "plugin_ddpm",
    "plugin_dummy_sampler",
    "plugin_marginal_distribution",
    "plugin_uniform_sampler",
    "plugin_image_adsgan",
    "plugin_image_cgan",
    "plugin_decaf",
    "plugin_dpgan",
    "plugin_pategan",
    "plugin_privbayes",
    "plugin_ctgan(generic)",
    "plugin_fourier_flows",
    "plugin_aim",
    "plugin_arf",
    "plugin_great",
]

if not goggle_disabled:
    all_tests.append("plugin_goggle")

minimal_tests = [
    "basic_examples",
    "plugin_adsgan",
    "plugin_ctgan",
    "plugin_nflow",
    "plugin_tvae",
    "plugin_timegan",
]

# For extras
goggle_tests = ["plugin_goggle"]


@click.command()
@click.option("--nb_dir", type=str, default=".")
@click.option(
    "--tutorial_tests",
    type=click.Choice(
        ["minimal_tests", "all_tests", "goggle_tests"],
        case_sensitive=False,
    ),
    default="minimal_tests",
)
@click.option(
    "--timeout",
    type=int,
    default=1800,
    help="Timeout for notebook execution in seconds.",
)
def main(nb_dir: Path, tutorial_tests: str, timeout: int) -> None:
    nb_dir = Path(nb_dir)
    enabled_tests: List = []
    if tutorial_tests == "all_tests":
        enabled_tests = all_tests
    elif tutorial_tests == "minimal_tests":
        enabled_tests = minimal_tests

    for p in nb_dir.rglob("*"):
        if p.suffix != ".ipynb":
            continue

        if "checkpoint" in p.name:
            continue

        ignore = True
        for val in enabled_tests:
            if val in p.name:
                ignore = False
                break

        if ignore:
            continue

        print("Testing ", p.name)
        start = time()
        try:
            run_notebook(p, timeout)
        except BaseException as e:
            print("FAIL", p.name, e)

            raise e
        finally:
            print(f"Tutorial {p.name} tool {time() - start}")


if __name__ == "__main__":
    main()
