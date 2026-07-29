# Contributing

## 1. Install Development Build

From a command line, execute the following commands:

```bash
git clone https://github.com/MSDLLCpapers/obsidian
cd obsidian
git checkout main
pip install -e .[dev]
```

## 2. Style

Linting is enforced with [ruff][ruff] based on configurations in
`pyproject.toml`.

We recommend using VS Code with the ruff extension to automatically aid
adherence to code style.

All function and method signatures should contain Python 3.10+ [type
hints][type-hints].

Each module, class, method, and function should have a docstring. We use
[Google][google-style] style docstrings.

We prefer that class docstrings be written under class definition instead of
`__init__`.

## 3. Documentation

For documentation building, _obsidian_ uses [sphinx][sphinx] with
[autodoc][autodoc] and [autosummary][autosummary]

In order to rebuild documentation, first be sure to have installed the
documentation build

```bash
pip install -e .[docs]
```

Then perform the following steps:

```bash
cd docs
make clean
make html
```

Documentation HTML output will be built in `docs/build` with the
homepage at `docs/build/html/index.html`.

### Guidance Documentation

New or major changes to subpackages or modules (e.g. `acquisition`, `surrogate`,
`optimizer.BO_optimizer`, `objectives.scalarize`) should be covered by
informative documentation detailing API usage in Wiki articles.

Major features or configurations should be documented as examples in Tutorial
notebooks.

## 4. Testing

For testing, _obsidian_ uses [pytest][pytest].

From the repository root, run the test suite with:

```bash
pytest obsidian/tests
```

Test discovery and warning filters are configured in `pytest.ini`, and
coverage is configured in `.coveragerc`. To generate a coverage report,
pass the coverage flags explicitly:

```bash
pytest obsidian/tests --cov=obsidian --cov-report=html
```

We have also enabled fast testing with majority coverage and flagged slow tests
that can be avoided until major pull requests.

```bash
pytest obsidian/tests -m fast
pytest obsidian/tests -m "not slow"
```

All new features should be fully covered by newly written pytests.

## 5. License

By contributing to _obsidian_, you agree that your contributions will be
licensed under the package [LICENSE][license].

[ruff]: https://docs.astral.sh/ruff/
[type-hints]: https://peps.python.org/pep-0484/
[google-style]: https://google.github.io/styleguide/pyguide.html
[sphinx]: https://www.sphinx-doc.org/en/master/
[autodoc]: https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html
[autosummary]:
    https://www.sphinx-doc.org/en/master/usage/extensions/autosummary.html
[pytest]: https://docs.pytest.org
[license]: https://github.com/MSDLLCpapers/obsidian/blob/main/LICENSE
