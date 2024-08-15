# Surrogate Model

## 1. Introduction

The `obsidian.surrogates` submodule is a key component of the Obsidian Bayesian optimization library. It provides a collection of surrogate models used to approximate the objective function in the optimization process. These surrogate models are essential for efficient exploration of the parameter space and for making informed decisions about which points to evaluate next.

## 2. Available Surrogate Models

The `obsidian.surrogates` submodule offers several types of surrogate models:

1. **Gaussian Process (GP)**: The default surrogate model, suitable for most optimization tasks.
2. **Mixed Gaussian Process (MixedGP)**: A GP model that can handle mixed continuous and categorical input spaces.
3. **Deep Kernel Learning GP (DKL)**: A GP model with a neural network feature extractor.
4. **Flat GP**: A GP model with non-informative or no prior distributions.
5. **Prior GP**: A GP model with custom prior distributions.
6. **Multi-Task GP (MTGP)**: A GP model for multi-output optimization.
7. **Deep Neural Network (DNN)**: A dropout neural network model.

## 3. How to Use Surrogate Models

To use a surrogate model in your optimization process, you typically don't need to interact with it directly. The Obsidian optimizer will handle the creation and management of the surrogate model. However, if you need to create a surrogate model manually, you can do so using the `SurrogateBoTorch` class:

```python
from obsidian.surrogates import SurrogateBoTorch
from obsidian.parameters import ParamSpace, Target

# Define your parameter space
param_space = ParamSpace([...])  # Define your parameters here

# Create a surrogate model (default is GP)
surrogate = SurrogateBoTorch(model_type='GP')

# Fit the model to your data
surrogate.fit(X, y)

# Make predictions
mean, std = surrogate.predict(X_new)
```

## 4. Customization Options

### 4.1 Model Selection

You can choose different surrogate models by specifying the `model_type` parameter when creating a `SurrogateBoTorch` instance. Available options are:

- `'GP'`: Standard Gaussian Process
- `'MixedGP'`: Mixed input Gaussian Process
- `'DKL'`: Deep Kernel Learning GP
- `'GPflat'`: Flat (non-informative prior) GP
- `'GPprior'`: Custom prior GP
- `'MTGP'`: Multi-Task GP
- `'DNN'`: Dropout Neural Network

### 4.2 Hyperparameters

You can pass custom hyperparameters to the surrogate model using the `hps` parameter:

```python
surrogate = SurrogateBoTorch(model_type='GP', hps={'your_custom_param': value})
```

### 4.3 Custom GP Models

The submodule provides several custom GP implementations:

- `PriorGP`: A GP with custom prior distributions
- `FlatGP`: A GP with non-informative or no prior distributions
- `DKLGP`: A GP with a neural network feature extractor

### 4.4 Custom Neural Network Model

The `DNN` class provides a customizable dropout neural network model. You can adjust parameters such as dropout probability, hidden layer width, and number of hidden layers.

## 5. Examples

### 5.1 Using a standard GP surrogate

```python
from obsidian.surrogates import SurrogateBoTorch
from obsidian.parameters import ParamSpace, Target
import pandas as pd

# Define your parameter space
param_space = ParamSpace([...])  # Define your parameters here

# Assume X and y are your input features and target variables
X = pd.DataFrame(...)
y = pd.Series(...)

surrogate = SurrogateBoTorch(model_type='GP')
surrogate.fit(X, y)

# Make predictions
X_new = pd.DataFrame(...)
mean, std = surrogate.predict(X_new)
```

### 5.2 Using a Mixed GP for categorical and continuous variables

```python
surrogate = SurrogateBoTorch(model_type='MixedGP')
# cat_dims should be a list of indices for categorical variables in your input space
surrogate.fit(X, y, cat_dims=[0, 2])  # Assuming columns 0 and 2 are categorical
```

### 5.3 Using a DNN surrogate

```python
# The 'hps' parameter allows you to customize the DNN architecture
surrogate = SurrogateBoTorch(model_type='DNN', hps={'p_dropout': 0.1, 'h_width': 32, 'h_layers': 3})
surrogate.fit(X, y)
```

## 6. Advanced Usage

### 6.1 Saving and Loading Models

You can save and load surrogate models using the `save_state()` and `load_state()` methods:

```python
# Save model state
state = surrogate.save_state()

# Load model state
loaded_surrogate = SurrogateBoTorch.load_state(state)
```

### 6.2 Model Evaluation

You can evaluate the performance of a surrogate model using the `score()` method:

```python
loss, r2_score = surrogate.score(X_test, y_test)
```

This concludes the user guide for the `obsidian.surrogates` submodule. For more detailed information, please refer to the source code and docstrings in the individual files.