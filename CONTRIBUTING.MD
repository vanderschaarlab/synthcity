# Contributing to Synthcity

We want to make contributing to Synthcity is as easy and transparent as possible. We hope to collaborate with as many people as we can.


## Development installation

First create a new environment. It is recommended that you use conda. This can be done as follows:
```bash
conda create -n your-synthcity-env python=3.9
conda activate your-synthcity-env
```
*Python versions 3.7, 3.8, 3.9, and 3.10 are all compatible*

To get the development installation with all the necessary dependencies for
linting, testing, auto-formatting, and pre-commit etc. run the following:
```bash
git clone https://github.com/vanderschaarlab/synthcity.git
cd synthcity
pip install -e .[testing]
```

Please check that the pre-commit is properly installed for the repository, by running:
```bash
pre-commit run --all
```
This checks that you are set up properly to contribute, such that you will match the code style in the rest of the project. This is covered in more detail below.


## Our Development Process

### Code Style

We believe that having a consistent code style is incredibly important. Therefore Synthcity imposes certain rules on the code that is contributed and the automated tests will not pass, if the style is not adhered to. These tests passing is a requirement for a contribution being merged. However, we make adhering to this code style as simple as possible. First, all the libraries required to produce code that is compatible with Synthcity's Code Style are installed in the step above when you set up the development environment. Secondly, these libraries are all triggered by pre-commit, so once you are set-up, you don't need to do anything. When you run `git commit`, any simple changes to enforce the style will run automatically and other required changes are explained in the stdout for you to go through and fix.

Synthcity uses the [black](https://github.com/ambv/black) and [flake8](https://github.com/PyCQA/flake8) code formatter to enforce a common code style across the code base. No additional configuration should be needed (see the [black documentation](https://black.readthedocs.io/en/stable/installation_and_usage.html#usage) for advanced usage).

Also, Synthcity uses [isort](https://github.com/timothycrosley/isort) to sort imports alphabetically and separate into sections.


#### Type Hints

Synthcity is fully typed using python 3.7+ [type hints](https://www.python.org/dev/peps/pep-0484/). This is enforced for contributions by [mypy](https://github.com/python/mypy), which is a static type-checker.


#### Tests

To run the tests, you can either use `pytest` (again, installed with the testing extra above).
The following testing command is good for checking your code,as it skips the tests that take a long time to run.
```bash
pytest -vvvsx -m "not slow" --durations=50
```

But the full test suite can be run with the following command.
```bash
pytest -vvvs  --durations=50
```

Some plugins may be included in the library as extras, the associated tests for these need to be run separately, e.g. the goggle plugin can be tested with the below command:
```bash
pytest -vvvs -k goggle --durations=50
```
## Pull Requests
We actively welcome your pull requests.

1. Fork the repo and create your branch from `main`.
2. If you have added code that should be tested, add tests in the same style as those already present in the repo.
3. If you have changed APIs, document the API change in the PR.
4. Ensure the test suite passes.
5. Make sure your code passes the pre-commit, this will be required in order to commit and push, if you have properly installed pre-commit, which is included in the testing extra.


## Issues

We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.


## License

By contributing to Synthcity, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree. You should therefore, make sure that if you have introduced any dependencies that they also are covered by a license that allows the code to be used by the project and is compatible with the license in the root directory of this project.
