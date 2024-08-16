# Surrogate Model

## 1. Introduction

The `obsidian.surrogates` submodule is a key component of the Obsidian APO library. It provides a collection of surrogate models used to approximate the objective function in the optimization process. These surrogate models are essential for efficient exploration of the parameter space and for making informed decisions about which points to evaluate next.

## 2. Basic Syntax

To use a surrogate model in your optimization process, you typically don't need to interact with it directly. The `obsidian.optimizer` submodule will handle the creation and management of the surrogate model. However, if you need to create a surrogate model manually, you can do so using the `SurrogateBoTorch` class. 

Below is a simple example using the default standard GP surrogate:

### Define the parameter space:
```python
from obsidian.parameters import ParamSpace, Param_Continuous
params = ParamSpace([Param_Continuous('X1', 0, 1),Param_Continuous('X2', 0, 1)])
X_space = ParamSpace(params)
```

### Simulate training data:
```python
from obsidian.experiment import ExpDesigner
designer = ExpDesigner(X_space, seed = 789)
X_train = designer.initialize(m_initial = 10, method='Sobol')

from obsidian.parameters import Target
target = Target(name = 'Y', f_transform = 'Standard', aim='max')

from obsidian.experiment import Simulator
from obsidian.experiment.benchmark import paraboloid
simulator = Simulator(X_space, paraboloid, name='Y')
y_train = simulator.simulate(X_train)
y_train_transformed = target.transform_f(y_train, fit = True)

import pandas as pd
print(pd.concat([X_train, y_train_transformed], axis=1).to_markdown())
```

|    |       X1 |        X2 |   Y Trans |
|---:|---------:|----------:|----------:|
|  0 | 0.709874 | 0.891838  | -0.692069 |
|  1 | 0.316783 | 0.0374523 | -1.283    |
|  2 | 0.102254 | 0.517022  | -0.229447 |
|  3 | 0.99438  | 0.412149  | -1.33756  |
|  4 | 0.80163  | 0.663557  |  0.252902 |
|  5 | 0.162435 | 0.265744  | -0.351742 |
|  6 | 0.385264 | 0.788242  |  0.507141 |
|  7 | 0.523473 | 0.140918  |  0.113744 |
|  8 | 0.588516 | 0.604981  |  1.423    |
|  9 | 0.445465 | 0.465715  |  1.59703  |


### Create a surrogate model (using the default 'GP' model), and fit the model to the training  data:
```python
from obsidian.surrogates import SurrogateBoTorch
surrogate = SurrogateBoTorch(model_type='GP', seed = 123)
surrogate.fit(X_train, y_train_transformed,cat_dims=[],task_feature=None)
```

### Generate new input experimental conditions and make predictions:
```python
X_new = designer.initialize(m_initial = 3, method='Sobol')
mean, std = surrogate.predict(X_new)

df = X_new.assign(pred_mean=mean,pred_std=std)
print(df.to_markdown())
```

|    |       X1 |        X2 |   pred_mean |   pred_std |
|---:|---------:|----------:|------------:|-----------:|
|  0 | 0.709874 | 0.891838  |   -0.669406 |   0.120078 |
|  1 | 0.316783 | 0.0374523 |   -1.25515  |   0.119587 |
|  2 | 0.102254 | 0.517022  |   -0.217739 |   0.12007  |


## 3. Customization Options

### 3.1 Available Surrogate Models

The `obsidian.surrogates` submodule offers several types of surrogate models.
You can choose different surrogate models by specifying the `model_type` parameter when creating a `SurrogateBoTorch` instance. 
Available options are:

- `'GP'`: Standard Gaussian Process, which is the default surrogate model suitable for most optimization tasks.
- `'MixedGP'`: Mixed input Gaussian Process, which is a GP model that can handle mixed continuous and categorical input spaces.
- `'DKL'`: Deep Kernel Learning GP, which is a GP model with a neural network feature extractor.
- `'GPflat'`: A GP model with non-informative or no prior distributions.
- `'GPprior'`: A GP model with custom prior distributions.
- `'MTGP'`: Multi-Task GP, which is a GP model for multi-output optimization.
- `'DNN'`: Dropout Neural Network model.

### 3.2 Hyperparameters

You can pass custom hyperparameters to the surrogate model using the `hps` argument as:
> surrogate = SurrogateBoTorch(model_type='GP', hps={'custom_param_1': value_1, 'custom_param_2': value_2, ...})

Some specific examples:
```python
surrogate = SurrogateBoTorch(model_type='FlatGP', hps={'nu': 1.5})
surrogate = SurrogateBoTorch(model_type='DNN', hps={'p_dropout': 0.1, 'h_width': 15, 'h_layers': 4, 'num_outputs': 2})
```

### 3.3 Custom GP Models

The submodule provides several custom GP implementations:

- `PriorGP`: A GP with custom prior distributions
- `FlatGP`: A GP with non-informative or no prior distributions
- `DKLGP`: A GP with a neural network feature extractor

### 3.4 Custom Neural Network Model

The `DNN` class provides a customizable dropout neural network model. 
The 'hps' parameter allows you to customize the DNN architecture. 
You can adjust multiple DNN hyperparameters such as dropout probability, hidden layer width, and number of hidden layers. 

## 4. Additional Examples


### 4.1 Using a Mixed GP for categorical and continuous variables

```python
# DO NOT RUN
surrogate = SurrogateBoTorch(model_type='MixedGP')
# cat_dims should be a list of indices for categorical variables in your input space
surrogate.fit(X, y, cat_dims=[0, 2])  # Assuming columns 0 and 2 are categorical
```

### 4.2 Using a DNN surrogate

```python
surrogate = SurrogateBoTorch(model_type='DNN', hps={'p_dropout': 0.1, 'h_width': 32, 'h_layers': 3})
surrogate.fit(X_train, y_train_transformed,cat_dims=[],task_feature=None)
```

## 5. Advanced Usage

### 5.1 Saving and Loading Models

You can save and load surrogate models to/from dictionary objects using the `save_state()` and `load_state()` methods, 
which enable saving the trained model to json files and reload for future usage:

```python
import json

# Save model state as dictionary to json file
with open('surrogate.json', 'w') as f:
    surrogate_dict = surrogate.save_state()
    json.dump(surrogate_dict, f)

# Load model state dictionary from json file
with open('surrogate.json', 'r') as f:
    surrogate_dict = json.load(f)
    surrogate_reload = SurrogateBoTorch.load_state(surrogate_dict)
```

### 5.2 Model Evaluation

You can evaluate the performance of a surrogate model using the `score()` method:

```python
y_new = simulator.simulate(X_new)
y_new_transformed = target.transform_f(y_new, fit = False)
loss, r2_score = surrogate.score(X_new, y_new_transformed)
```

This concludes the user guide for the `obsidian.surrogates` submodule. For more detailed information, please refer to the source code and docstrings in the individual files.