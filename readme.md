<!---
obsidian
ReadMe
-->

<p align="center">
  <img src="docs/figures/obsidian_logo.png" width="100" alt = "obsidian logo">
</p>

<!--- RELEASE TODO
src= github link + 'docs/_static/
--->


<div align="center">

# obsidian

![Supports Python](https://img.shields.io/badge/Python-3.10-teal)
![License](https://img.shields.io/badge/license-GPLv3-teal.svg)

<!--- RELEASE TODO
![Isses]()
--->


__obsidian__ is a library for algorithmic process design and black-box optimization using ML-guided experiment design


</div>


The obsidian library offers a set of modules for designing, executing, analyzing, and visualizing algorithmic process optimization (APO) using sample-efficient strategies such as Bayesian Optimization (BO). obsidian uses highly flexible models to build internal representations of the measured system in a way that can be explored for characterization and exploited for maximization based on uncertainty estimation and exploration strategies. obsidian supports batch experimentation (joint optimization and parallel evaluation) and is highly configurable for varying use cases, although the default specifications are encouraged.

_We thank you for your patience and invite you to collaborate with us while __obsidian__ is in beta!_

 # Key Features

 1. __End-User-Friendly__: Designed with a process development scientist in mind. No machine learning experience required.
 2. __Deployable__ using pre-built _Dash_ application. Each class is natively fully serializable to enable web-based API usage. 
 3. __Interpretable__ and visualizable using SHAP analysis and interactive figures.
 4. __Adjustable__: Our approach to experiment design encourages "human-in-the-loop" and supports fluctuating objectives and resources. Our code is built to reflect that, such as through exposing tunable explore-exploit hyperparameters for each acquisition function.
 5. __Flexible__: Handles any input (numeric, discrete) with/without constraints alongside single or multiple-outcome optimizations also with optional constraints and novelty objective compositions.
 6. __Purpose-Driven Development__: Impactful features (such as the observational "time series" parameter) proposed, developed, maintained, and used by laboratory bench scientists. 

# How it Works: Algorithmic Optimization
The workflow for algorithmic process optimization is an iterative workflow of the following steps:

1. Collect data
2. Fit a model to the data and estimate uncertainty across a design space
3. Search for new experiments and evaluate for objective or informational utility
4. Design experiments where utility is maximized
5. Repeat

The central object ob the __obsidian__ library is the `BayesianOptimizer`. A bayesian optimization has two key components that govern the optimization:
1. The surrogate model: A black-box model which is regressed to data and used for inference. Most often a _Gaussian Process_.
2. The acquisition function: A mathematical description of the quality of potential experiments, as it pertains to optimization. Most often _Expected Value_.

# Usage Example

## Specify Parameters & Initialize a Design

```python
from obsidian.parameters import ParamSpace, Param_Categorical, Param_Ordinal, Param_Continuous
from obsidian.experiment import ExpDesigner

params = [
    Param_Continuous('Temperature', -10, 30),
    Param_Continuous('Concentration', 10, 150),
    Param_Continuous('Enzyme', 0.01, 0.30),
    Param_Categorical('Variant', ['MRK001', 'MRK002', 'MRK003']),
    Param_Ordinal('Stir Rate', ['Low', 'Medium', 'High']),
    ]

X_space = ParamSpace(params)
designer = ExpDesigner(X_space)
X0 = designer.initialize(10, 'LHS', seed=0)
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


## Collect Data

```python
from obsidian.campaign import Campaign
my_campaign = Campaign(X_space)
my_campaign.add_data(Z0)
```

## Set the Target and Fit the Optimizer

```python
from obsidian.parameters import Target
from obsidian.optimizer import BayesianOptimizer

optimizer = BayesianOptimizer(X_space, seed = 0)
target = Target('Yield', aim='max')
optimizer.fit(my_campaign.data, target)
```

## Suggest New Experiments

```python
optimizer.suggest(m_batch=2)
```

|    |   Temperature |   Concentration |    Enzyme | Variant   | Stir Rate   |   Yield (pred) |   Yield lb |   Yield ub | aq Method   |   aq Value |
|---:|--------------:|----------------:|----------:|:----------|:------------|---------------:|-----------:|-----------:|:------------|-----------:|
|  0 |           -10 |              10 | 0.0918096 | MRK001    | Medium      |       112.497  |   102.558  |   122.436  | EI          |   0.848569 |
|  1 |           -10 |             150 | 0.0882423 | MRK002    | High        |        89.8334 |    79.8589 |    99.8079 | EI          |   0.870511 |

# Installation

The latest _obsidian_ release can be installed using pip:

```python
pip install obsidian
```

Be sure to `pip` install in a newly created `conda` environment to avoid dependency conflicts.

<!--- RELEASE TODO
```
pip install -e. [app]
```
--->

# Contributing

See CONTRIBUTING to learn more.

<!--- RELEASE TODO
See [CONTRIBUTING](CONTRIBUTING.md) to learn more.
--->


## Developers

- Kevin Stone (Merck & Co., Inc.) [kevin.stone@merck.com](mailto:kevin.stone@merck.com)
- Yuting Xu (Merck & Co., Inc.) [yuting.xu@merck.com](mailto:yuting.xu@merck.com)

## Contributors

- Ajit Vikram (Merck & Co., Inc.)
- Melodie Christensen (Merck & Co., Inc.)
- Kobi Felton (Merck & Co., Inc.)

## License
__obsidian__ is licensed by the GPLv3 license.

<!--- RELEASE TODO
...license, found [here](LICENSE)
--->
