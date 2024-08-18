# Acquisition Function

## 1. Introduction

The [`obsidian.acquisition`](https://github.com/MSDLLCpapers/obsidian/tree/main/obsidian/acquisition) submodule is a crucial component of the Obsidian APO library. It provides acquisition functions that guide the optimization process by determining which points in the parameter space should be evaluated next. These acquisition functions balance exploration of uncertain areas and exploitation of promising regions, which is key to efficient optimization.

## 2. Usage

### 2.1 Available Acquisition Functions 

The acquisition submodule offers a range of acquisition functions, including standard `BoTorch` acquisition functions and custom implementations. 
Below, you will find various options for the acquisition argument, along with a brief description for each.

#### Standard Acquisition Functions from BoTorch

_Single-Objective Optimization:_

- EI: Expected Improvement
- NEI: Noisy Expected Improvement
- PI: Probability of Improvement
- UCB: Upper Confidence Bound
- SR: Simple Regret
- NIPV: Integrated Negative Posterior Variance

_Multi-Objective Optimization:_

- EHVI: Expected Hypervolume Improvement
- NEHVI: Noisy Expected Hypervolume Improvement
- NParEGO: Random augmented chebyshev scalarization with Noisy Expected Improvement

#### Custom Acquisition Functions

- Mean: Seeks to optimize the maximum value of the posterior mean
- SF: Space filling, aims to optimize the maximum value of the minimum distance between a point and the training data

#### Baseline: No Acquisition Function

- RS: Random sampling from the parameter space

### 2.2 Basic Syntax

Typically, users don't need to interact with acquisition functions directly. 
The `BayesianOptimizer` class handles the selection and use of acquisition functions. 
The acquisition functions, as well as their hyperparameters, could be specified as an input argument when calling the `suggest` method:

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

The `acquisition` parameter should be always a list, containing either string or dictionary elements. When multiple elements are present in the list, the `suggest` function will propose new candidates sequentially using each acquisition function.

* If using the default hyperparameters for an acquisition function, specify a string element using the name of the acquisition function (e.g., 'EI' for Expected Improvement).
* If specifying custom hyperparameters for an acquisition function, use a dictionary element where the key is the acquisition function name and the value is a nested dictionary storing its hyperparameters (e.g., {'UCB': {'beta': 0.1}} for Upper Confidence Bound with a specified beta parameter).


## 3. Single-Objective Optimizatio Acquisition Functions


### Expected Improvement (EI)

EI calculates the expected amount by which we will improve upon the current best observed value.

Mathematical formulation:
\begin{equation*}
EI(x) = E[max(\hat{f}(x) - y_{best}, 0)]
\end{equation*}

where $y_{best}$ is the current best observed value. 
The expression $max(\hat{f}(x) - y_{best}$ captures the potential improvement over the current best observed value, and the expectation E[ ] calculates the average improvement over the posterior distribution of the surrogate model predictions.

_Optional hyperparameters:_

* inflate: Increase the current best value $y_{best}$ to $(1+inflate)*y_{best}$, enabling a more flexible exploration-exploitation trade-off: 

    \begin{equation*}
    EI(x) = E[max(\hat{f}(x) - (1+inflate)*y_{best}, 0)]
    \end{equation*}

    The default value is 0 (no inflation). Recommended values: small numberic number 0~0.1

**Example usage:**

* Default:

    ```python
    from obsidian.optimizer import BayesianOptimizer
    
    optimizer = BayesianOptimizer(X_space=param_space)
    X_suggest, eval_suggest = optimizer.suggest(acquisition=['EI'])
    ```
    
* With all available hyperparameters:
      
    ```python
    X_suggest, eval_suggest = optimizer.suggest(acquisition=[{'EI': {'inflate': 0.05}}])
    ```

### Noisy Expected Improvement (NEI)

NEI is a variant of EI that accounts for noise in the observations, making it more suitable for real-world problems with measurement uncertainty. It allows for more robust decision-making in selecting the next point for evaluation. 

Currently NEI doesn't accept additional hyperparameters.  

**Example usage:**

```python
X_suggest, eval_suggest = optimizer.suggest(acquisition=['NEI'])
```

### Probability of Improvement (PI)

PI is designed to aid in the efficient selection of candidate points by quantifying the probability of improving upon the current best observed value.
Different from EI, which measures the expected amount of improvement by integrating over the posterior distribution, PI directly evaluates the probability of outperforming the best value, emphasizing the likelihood of improvement rather than the magnitude of improvement.

Mathematical formulation:
\begin{equation*}
PI(x) = P(\hat{f}(x) > y_{best})
\end{equation*}
where $y_{best}$ is the current best observed value and the $\hat{f}$ is the trained surrogate function. 

_Optional hyperparameters:_
* inflate: Increase the current best value $y_{best}$ to $(1+inflate)*y_{best}$, enabling a more flexible exploration-exploitation trade-off:

    \begin{equation*}
    PI(x) = P(\hat{f}(x) > (1+inflate)*y_{best})
    \end{equation*}

  The default value is 0 (no inflation). Recommended values: small numberic number 0~0.1

**Example usage:**
```python
# Use the default inflate = 0
X_suggest, eval_suggest = optimizer.suggest(acquisition=['PI'])

# Adjust the hyperparameter inflate
X_suggest, eval_suggest = optimizer.suggest(acquisition=[{'PI': {'inflate': 0.05}}])
```

### Upper Confidence Bound (UCB)

UCB balances exploration and exploitation by selecting points with high predicted values or high uncertainty.

Mathematical formulation:
\begin{equation*}
UCB(x) = \mu(x) + \beta * \sigma(x) 
\end{equation*}

where $\mu(x)$ is the predicted mean at the candidate point $x$, $\sigma(x)$ is the predicted standard deviation which is associated with uncertainty, and $\beta$ is a parameter that controls the exploration-exploitation trade-off. 

_Optional hyperparameters:_
* $\beta$: By default $\beta = 1$. Recommended value range: 1~3

**Example usage:**

```python
# Use the default beta = 1
X_suggest, eval_suggest = optimizer.suggest(acquisition=['UCB'])

# Adjust the hyperparameter beta
X_suggest, eval_suggest = optimizer.suggest(acquisition=[{'UCB': {'beta': 2.0}}])
```

## 4. Multi-Objective Optimization Acquisition Functions

### Hypervolume Improvement

One of the most well-known and widely used acquisition functions for multi-objective optimization is the Hypervolume Improvement-based acquisition functions. 
The Hypervolume is a measure of the covered area in the objective space that is dominated by a given Pareto front; therefore, it is often used to quantify the quality of a Pareto front approximation. (See also Section [Additional Analysis](2_Analysis_and_Visualization.md))

There are two available options in `obsidian`: 
* **Expected Hypervolume Improvement (EHVI)**
* **Noisy Expected Hypervolume Improvement (NEHVI)**


The EHVI and NEHVI acquisition functions aim to select the next set of input points that would maximize the hypervolume of the Pareto front. They differ in the consideration of noise in the objective function evaluations, with NEHVI explicitly accounting for noisy observations while EHVI assumes noise-free evaluations.

NEHVI enables more robust optimization in the presence of noisy observations, improving the reliability of the optimization process. However, the incorporation of noise modeling in NEHVI may lead to increased computational complexity, as it often requires a larger number of Monte Carlo samples to accurately capture the noise characteristics, potentially making it more computationally intensive than EHVI in scenarios with significant noise.

_Hyperparameters:_
* ref_point: The reference point for computing the hypervolume. Default value for each dimension is the minimum value minus 10% of the range.

**Example usage:**

```python
# Using default values for ref_point:
X_suggest, eval_suggest = optimizer.suggest(acquisition=['EHVI'])
X_suggest, eval_suggest = optimizer.suggest(acquisition=['NEHVI'])

# Custom ref_point, assuming there are two outputs
X_suggest, eval_suggest = optimizer.suggest(acquisition = [{'EHVI':{'ref_point':[5, 40]}}])
X_suggest, eval_suggest = optimizer.suggest(acquisition = [{'NEHVI':{'ref_point':[-2, -30]}}])
```

### Weighted Response

* Random augmented chebyshev scalarization with Noisy Expected Improvement (NParEGO)

* Additional scalarization options for multi-objective problems

## 5. Advanced Usage

### Custom Acquisition Functions

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

## 6. Comparing Acquisition Functions

Different acquisition functions have different strengths:

- EI and PI are good for exploiting known good regions but may underexplore.
- UCB provides a tunable exploration-exploitation trade-off.
- NEI and NEHVI are robust to noisy observations.
- qMean is purely exploitative and can be useful in the final stages of optimization.
- qSpaceFill is purely explorative and can be useful for initial space exploration.
- EHVI is advantageous in noise-free or low-noise settings due to its computational efficiency, while NEHVI is better suited for scenarios with significant noise, offering improved robustness at the cost of potentially higher computational demands.

## 7. Best Practices

1. Choose appropriate acquisition functions based on your problem characteristics (e.g., noise level, number of objectives).
2. For noisy problems, consider using noise-aware acquisition functions like NEI or NEHVI.
3. Experiment with different acquisition functions to find the best performance for your specific problem.
4. When using UCB, carefully tune the beta parameter to balance exploration and exploitation.
5. For multi-objective problems, EHVI and NEHVI are often good choices.
6. Consider using a sequence of acquisition functions, starting with more exploratory ones and moving to more exploitative ones as the optimization progresses.

## 8. Common Pitfalls

1. Using EI or PI in noisy problems, which can lead to overexploitation of noisy observations.
2. Setting UCB's beta parameter too high (over-exploration) or too low (over-exploitation).
3. Using single-objective acquisition functions for multi-objective problems.
4. Not accounting for constraints when selecting acquisition functions.

This concludes the user guide for the `obsidian.acquisition` submodule. For more detailed information, please refer to the source code and docstrings in the individual files.