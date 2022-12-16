# stdlib
from pathlib import Path

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


ignored_strings = [
    "checkpoint",
    "tutorial1_add_a_new_plugin",
]


@click.command()
@click.option("--nb_dir", type=str, default=".")
def main(nb_dir: Path) -> None:
    nb_dir = Path(nb_dir)

    for p in nb_dir.rglob("*"):
        if p.suffix != ".ipynb":
            continue

        ignore = False
        for val in ignored_strings:
            if val in p.name:
                ignore = True
                break

        if ignore:
            continue

        print("Testing ", p.name)
        try:
            run_notebook(p)
        except BaseException as e:
            print("FAIL", p.name, e)

            raise e


if __name__ == "__main__":
    main()
