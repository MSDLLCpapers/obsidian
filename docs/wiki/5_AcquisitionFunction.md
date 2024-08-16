# Acquisition Function

## 1. Introduction

The [`obsidian.acquisition`](https://github.com/MSDLLCpapers/obsidian/tree/main/obsidian/acquisition) submodule is a crucial component of the Obsidian APO library. It provides acquisition functions that guide the optimization process by determining which points in the parameter space should be evaluated next. These acquisition functions balance exploration of uncertain areas and exploitation of promising regions, which is key to efficient optimization.

## 2. Basic Syntax

Typically, users don't need to interact with acquisition functions directly. 
The `BayesianOptimizer` class handles the selection and use of acquisition functions. 
The acquisition function, including its hyperparameters, could be specified as an input argument when calling the `suggest` method:

```python
# DO NOT RUN
from obsidian.optimizer import BayesianOptimizer

optimizer = BayesianOptimizer(X_space=param_space)

# Use one acquisition function EI per iteration
X_suggest, eval_suggest = optimizer.suggest(acquisition=['EI'])

# Use two acquisition functions EI and UCB per iteration
X_suggest, eval_suggest = optimizer.suggest(acquisition=['EI','UCB'])

# Use two acquisition functions EI and UCB per iteration, while specifying hyperparameters for UCB
X_suggest, eval_suggest = optimizer.suggest(acquisition=['EI',{'UCB':{'beta':0.1}}])
```

## 3. Customization Options

### 3.1 Available Surrogate Models and Hyperparameters

The acquisition submodule includes multiple acquisition functions, both standard `BoTorch` acquisition functions and custom implementations.

#### Standard Acquisition Functions

- Expected Improvement (EI)
- Probability of Improvement (PI)
- Upper Confidence Bound (UCB)
- Noisy Expected Improvement (NEI)
- Expected Hypervolume Improvement (EHVI)
- Noisy Expected Hypervolume Improvement (NEHVI)

#### Custom Acquisition Functions

- qMean: Optimizes for the maximum value of the posterior mean
- qSpaceFill: Optimizes for the maximum value of minimum distance between a point and the training data


---------------

## 4. Understanding Acquisition Functions

### 3.1 Expected Improvement (EI)

EI calculates the expected amount by which we will improve upon the current best observed value.

Mathematical formulation:
```
EI(x) = E[max(f(x) - f(x+), 0)]
```
where f(x+) is the current best observed value.

Example usage:
```python
from obsidian.optimizer import BayesianOptimizer

optimizer = BayesianOptimizer(X_space=param_space)
X_suggest, eval_suggest = optimizer.suggest(acquisition=['EI'])
```

### 3.2 Upper Confidence Bound (UCB)

UCB balances exploration and exploitation by selecting points with high predicted values or high uncertainty.

Mathematical formulation:
```
UCB(x) = μ(x) + β * σ(x)
```
where μ(x) is the predicted mean, σ(x) is the predicted standard deviation, and β is a parameter that controls the exploration-exploitation trade-off.

Example usage:
```python
X_suggest, eval_suggest = optimizer.suggest(acquisition=[{'UCB': {'beta': 2.0}}])
```

### 3.3 Noisy Expected Improvement (NEI)

NEI is a variant of EI that accounts for noise in the observations, making it more suitable for real-world problems with measurement uncertainty.

Example usage:
```python
X_suggest, eval_suggest = optimizer.suggest(acquisition=['NEI'])
```

## 4. Advanced Usage

### 4.1 Multi-Objective Optimization

For multi-objective optimization problems, you can use specialized acquisition functions:

```python
X_suggest, eval_suggest = optimizer.suggest(acquisition=['NEHVI'])
```

### 4.2 Customizing Acquisition Functions

Some acquisition functions accept parameters to customize their behavior. These can be specified in the `suggest` method:

```python
X_suggest, eval_suggest = optimizer.suggest(
    acquisition=[{'EI': {'inflate': 0.01}}]
)
```

### 4.3 Custom Acquisition Functions

If you need to implement a custom acquisition function, you can extend the `MCAcquisitionFunction` class from BoTorch:

```python
from botorch.acquisition import MCAcquisitionFunction
import torch

class CustomAcquisition(MCAcquisitionFunction):
    def forward(self, X):
        posterior = self.model.posterior(X)
        mean = posterior.mean
        std = posterior.variance.sqrt()
        return (mean + 0.1 * std).sum(dim=-1)  # Example custom acquisition logic
```

## 5. Comparing Acquisition Functions

Different acquisition functions have different strengths:

- EI and PI are good for exploiting known good regions but may underexplore.
- UCB provides a tunable exploration-exploitation trade-off.
- NEI and NEHVI are robust to noisy observations.
- qMean is purely exploitative and can be useful in the final stages of optimization.
- qSpaceFill is purely explorative and can be useful for initial space exploration.

## 6. Best Practices

1. Choose appropriate acquisition functions based on your problem characteristics (e.g., noise level, number of objectives).
2. For noisy problems, consider using noise-aware acquisition functions like NEI or NEHVI.
3. Experiment with different acquisition functions to find the best performance for your specific problem.
4. When using UCB, carefully tune the beta parameter to balance exploration and exploitation.
5. For multi-objective problems, EHVI and NEHVI are often good choices.
6. Consider using a sequence of acquisition functions, starting with more exploratory ones and moving to more exploitative ones as the optimization progresses.

## 7. Common Pitfalls

1. Using EI or PI in noisy problems, which can lead to overexploitation of noisy observations.
2. Setting UCB's beta parameter too high (over-exploration) or too low (over-exploitation).
3. Using single-objective acquisition functions for multi-objective problems.
4. Not accounting for constraints when selecting acquisition functions.

This concludes the user guide for the `obsidian.acquisition` submodule. For more detailed information, please refer to the source code and docstrings in the individual files.