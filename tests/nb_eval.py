# stdlib
from pathlib import Path
from time import time

# third party
import click
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

workspace = Path(__file__).parents[0] / "workspace"
workspace.mkdir(parents=True, exist_ok=True)


def run_notebook(notebook_path: Path) -> None:
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)

    proc = ExecutePreprocessor(timeout=1800)
    # Will raise on cell error
    proc.preprocess(nb, {"metadata": {"path": workspace}})


enabled_tests = [
    "basic_examples",
    "adsgan",
    "ctgan",
    "nflow",
    "tvae",
    "timegan",
]


@click.command()
@click.option("--nb_dir", type=str, default=".")
def main(nb_dir: Path) -> None:
    nb_dir = Path(nb_dir)

    for p in nb_dir.rglob("*"):
        if p.suffix != ".ipynb":
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
            run_notebook(p)
        except BaseException as e:
            print("FAIL", p.name, e)

            raise e
        finally:
            print(f"Tutorial {p.name} tool {time() - start}")


if __name__ == "__main__":
    main()
