<!---
obsidian
ReadMe
-->

<div align = "center">
  <img src="https://github.com/MSDLLCpapers/obsidian/blob/main/docs/_static/obsidian_logo.svg?raw=true" class="only-light" width="100" alt = "obsidian logo">
</div>


<div align="center">

<h1>obsidian</h1>

![Supports Python][python-badge] [![License][license-badge]][license-link]
[![Issues][issues-badge]][issues-link] [![PyPI][pypi-badge]][pypi-link]
[![Docs][docs-badge]][docs-link] [![Codecov][codecov-badge]][codecov-link]

__obsidian__ is a library for algorithmic process design and black-box
optimization using AI-guided experiment design.

</div>

```bash
pip install obsidian-apo
```

The _obsidian_ library offers a set of modules for designing, executing,
analyzing, and visualizing algorithmic process optimization (APO) using
sample-efficient strategies such as Bayesian Optimization (BO). _obsidian_ uses
highly flexible models to build internal representations of the measured system
in a way that can be explored for characterization and exploited for
maximization based on uncertainty estimation and exploration strategies.
_obsidian_ supports batch experimentation (joint optimization and parallel
evaluation) and is highly configurable for varying use cases, although the
default specifications are encouraged.

_With the 1.0 release, __obsidian__ is now stable and production-ready. We thank
you for your support and invite you to collaborate with us!_

# Key Features

1. __End-User-Friendly__: Designed to elevate the average process development
   scientist. No machine learning experience required.
1. __Deployable__: Each class is fully serializable, without third-party
   packages, to enable web-based API usage.
1. __Explainable__ and visualizable using SHAP analysis and interactive figures.
1. __Flexible__: Handles any input (numeric, discrete) and optionally
   input/output constraints, multiple outcomes, batch optimization, and a
   variety of novelty objective compositions. We know that experiment campaigns
   can have fluctuating objectives and resources, and _obsidian_ is built to
   support that.
1. __Purpose-Driven Development__: Impactful features proposed, developed,
   maintained, and used by laboratory bench scientists. Relevantly designed for
   process development, optimization, and characterization.

# How it Works: Algorithmic Optimization

The workflow for algorithmic process optimization is an iterative workflow of
the following steps:

1. Collect data
1. Fit a model to the data and estimate uncertainty across a design space
1. Search for new experiments and evaluate for objective and/or informational
   utility
1. Design experiments where utility is maximized
1. Repeat

The central object of the __obsidian__ library is the `BayesianOptimizer`, which
can be optionally wrapped by a `Campaign`. A bayesian optimization has two key
components that govern the optimization:

1. The surrogate model: A black-box model which is regressed to data and used
   for inference. Most often a _Gaussian Process_ (`surrogate='GP'`).
1. The acquisition function: A mathematical description of the quality of
   potential experiments, as it pertains to optimization. Most often _Expected
   Improvement_ (`acquisition=['EI']`).

# Usage Example

## Specify Parameters & Initialize a Design

```python
from obsidian import Campaign, ParamSpace, Target
from obsidian.parameters import Param_Categorical, Param_Ordinal, Param_Continuous

params = [
    Param_Continuous('Temperature', -10, 30),
    Param_Continuous('Concentration', 10, 150),
    Param_Continuous('Enzyme', 0.01, 0.30),
    Param_Categorical('Variant', ['MRK001', 'MRK002', 'MRK003']),
    Param_Ordinal('Stir Rate', ['Low', 'Medium', 'High']),
    ]

X_space = ParamSpace(params)
target = Target('Yield', aim='max')
campaign = Campaign(X_space, target)
X0 = campaign.designer.initialize(10, 'LHS', seed=0)
```

|    |   Temperature |   Concentration |   Enzyme | Variant   | Stir Rate   |
|---:|--------------:|----------------:|---------:|:----------|:------------|
|  0 |             8 |              17 |   0.1405 | MRK001    | Medium      |
|  1 |            12 |             143 |   0.1695 | MRK003    | Medium      |
|  2 |             4 |             101 |   0.2855 | MRK002    | High        |
|  3 |            28 |              87 |   0.1115 | MRK002    | Low         |
|  4 |            -4 |             115 |   0.2275 | MRK001    | Low         |
|  5 |            -8 |              73 |   0.0825 | MRK002    | Medium      |
|  6 |            20 |             129 |   0.0535 | MRK001    | High        |
|  7 |            24 |              31 |   0.2565 | MRK002    | Medium      |
|  8 |            16 |              59 |   0.1985 | MRK003    | High        |
|  9 |             0 |              45 |   0.0245 | MRK003    | Low         |

## Collect Data and Fit the Optimizer

```python
campaign.add_data(Z0)
campaign.fit()
```

## Suggest New Experiments

```python
campaign.suggest(m_batch=2)
```

|    |   Temperature |   Concentration |    Enzyme | Variant   | Stir Rate   |   Yield (pred) |   Yield lb |   Yield ub | aq Method   |   aq Value |
|---:|--------------:|----------------:|----------:|:----------|:------------|---------------:|-----------:|-----------:|:------------|-----------:|
|  0 |           -10 |              10 | 0.0918096 | MRK001    | Medium      |       112.497  |   102.558  |   122.436  | EI          |   0.848569 |
|  1 |           -10 |             150 | 0.0882423 | MRK002    | High        |        89.8334 |    79.8589 |    99.8079 | EI          |   0.870511 |

# Interpret Results

```python
from obsidian.plotting import surface_plot, optim_progress
surface_plot(campaign.optimizer, feature_ids=(0,2))
```

<div align='center'>
  <img src="https://github.com/MSDLLCpapers/obsidian/blob/main/docs/_static/tutorials/demo_surface.png?raw=true" width="400" alt = "obsidian app">
</div>
<br>

```python
optim_progress(campaign)
```
<div align='center'>
  <img src="https://github.com/MSDLLCpapers/obsidian/blob/main/docs/_static/tutorials/demo_optimprogress.png?raw=true" width="400" alt = "obsidian app">
</div>
<br>

```python
from obsidian.campaign import Explainer
exp = Explainer(campaign)
exp.shap_explain()
exp.shap_summary()
```
<div align='center'>
  <img src="https://github.com/MSDLLCpapers/obsidian/blob/main/docs/_static/tutorials/demo_shap.png?raw=true" width="400" alt = "obsidian app">
</div>
<br>

```python
exp.shap_pdp_ice(ind = 2, ice_color_var = 3)
```

<div align='center'>
  <img src="https://github.com/MSDLLCpapers/obsidian/blob/main/docs/_static/tutorials/demo_pdp_ice.png?raw=true" width="400" alt = "obsidian app">
</div>
<br>

# Installation

The latest _obsidian_ release can be installed using pip:

```bash
pip install obsidian-apo
```

Be sure to `pip` install in a newly created `conda` environment to avoid
dependency conflicts.

# Contributing

See [CONTRIBUTING][contributing-link] to learn more.

## Maintainers

- Kevin Stone (MSD) [@kstone40](https://github.com/kstone40)
- Spencer McMinn (MSD)
  [@spencermcminn](https://github.com/spencermcminn)
- Fima Margovskiy (MSD) [@fima5](https://github.com/fima5)
- Xinyang Li (MSD) [@tautomer](https://github.com/tautomer)

## Past maintainers

- Yuting Xu (MSD) [@xuyuting](https://github.com/xuyuting)
- Michael Turnbach (MSD)
  [@michaelTurnbach](https://github.com/michaelTurnbach)

## Contributors

- Ajit Vikram (MSD)
- Melodie Christensen (MSD)
- Kobi Felton (MSD)

## License

__obsidian__ is licensed by the [GPLv3 license][gpl-link].

[python-badge]:
    https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-teal
[license-badge]: https://img.shields.io/badge/license-GPLv3-teal.svg
[license-link]: https://github.com/MSDLLCpapers/obsidian/blob/main/LICENSE
[issues-badge]:
    https://img.shields.io/github/issues/msdllcpapers/obsidian?color=teal
[issues-link]: https://github.com/MSDLLCpapers/obsidian/issues
[pypi-badge]: https://img.shields.io/pypi/v/obsidian-apo.svg?color=teal
[pypi-link]: https://pypi.org/project/obsidian-apo/
[docs-badge]: https://img.shields.io/badge/read-docs-teal
[docs-link]: https://msdllcpapers.github.io/obsidian/
[codecov-badge]:
    https://img.shields.io/codecov/c/github/MSDLLCpapers/obsidian?color=teal
[codecov-link]: https://codecov.io/github/MSDLLCpapers/obsidian
[contributing-link]:
    https://github.com/MSDLLCpapers/obsidian/blob/main/CONTRIBUTING.md
[gpl-link]: https://github.com/MSDLLCpapers/obsidian/blob/main/LICENSE
