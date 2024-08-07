# Contributing

## 1. Install Development Build

From a command line, execute the following commands:

```python
git clone https://github.com/MSDLLCpapers/obsidian
cd obsidian
git checkout main
pip install -e .[dev]
```

## 2. Style

Linting is enforced with [flake8](http://flake8.pycqa.org) based on configurations in `.flake8`.

We recommend using VS Code with flake8 extensions to automatically aid adherence to code style.

All function and method signatures should contain Python 3.10+ [type hints](https://peps.python.org/pep-0484/).

Each module, class, method, and function should have a docstring. We use [Google](https://google.github.io/styleguide/pyguide.html) style docstrings.

We prefer that class docstrings be written under class definition instead of `__init__`.

## 3. Documentation


For documentation building, _obsidian_ uses [sphinx](https://www.sphinx-doc.org/en/master/) with [autodoc](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html) and [autosummary](https://www.sphinx-doc.org/en/master/usage/extensions/autosummary.html)

In order to rebuild documentation, first be sure to have installed the documentation build

```python
pip install -e .[docs]
```

The perform the following steps:

```python
cd obsidian/docs
make clean
make html
```

Documentation HTML output will be built in `obsidian/docs/build` with the homepage at `build/html/index.html`.

### Guidance Documentation

New or major changes to subpackages or modules (e.g. `acquisition`, `surrogate`, `optimizer.BO_optimizer`, `objectives.scalarize`) should be covered by informative documentation detailing API usage in Wiki articles.

Major features or configurations should be documented as examples in Tutorial notebooks.


## 4. Testing

For testing, _obsidian_ uses [pytest](https://docs.pytest.org).

In order to run pytests, execute the following commands:
```python
cd obsidian
pytest > logs/pytest_output.txt
```

By default, code coverage reports will be generated in `logs/pytestCovReport` according to configurations `pytest.ini` and `.coveragerc`. Logs will be output to `logs/pytest_output.txt`.

We have also enabled fast testing with majority coverage and flagged slow tests that can be avoided until major pull requests.
```python
pytest -m fast
pytest -m "not slow"
```

All new features should be fully covered by newly written pytests.

## 5. License
By contributing to _obsidian_, you agree that your contributions will be licensed under the package [LICENSE](https://github.com/MSDLLCpapers/obsidian/blob/main/LICENSE).
