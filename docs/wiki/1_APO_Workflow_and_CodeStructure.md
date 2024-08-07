# Sequential Model-Based Optimization (SMBO) Workflow

SMBO is an iterative procedure used for optimizing complex systems with an expensive-to-evaluate objective function. The major steps involve constructing a surrogate model to approximate the objective function and defining an acquisition function based on user preference that explores and/or exploits the design space to find the optimal solution.

When applied to chemical process optimization, the SMBO algorithms can effectively identify desirable experimental conditions or parameter settings that maximize process efficiency, minimize costs, or achieve other specified objectives within given resource constraints.

The components typically involved in SMBO algorithms are illustrated in the figure below: 

![APO Workflow](https://github.com/MSDLLCpapers/obsidian/blob/main/docs/_static/APO_workflow.png?raw=true)


In this **_obsidian_** library: 

The `Optimizer` class object in [optimizer](https://github.com/MSDLLCpapers/obsidian/tree/main/obsidian/optimizer) submodule is the main portal that connects all the key components in the SMBO algorithm. 

The [dash](https://github.com/MSDLLCpapers/obsidian/tree/main/obsidian/dash) submodule contains the source code for a web GUI using the Dash library. 

The [tests](https://github.com/MSDLLCpapers/obsidian/tree/main/obsidian/tests) submodule contains scripts to facilitate the automatic testing process using Pytest during software development. 
It allows developers to verify the correctness and functionality of the code, review warnings and catch any potential errors.

The [demo](https://github.com/MSDLLCpapers/obsidian/tree/main/demo) folder contains step-by-step Jupyter notebook usage examples as well as scripts for benchmark studies. 



----
## Data

The data structure in a typical SMBO work flow includes:

* **$X$: Experimental conditions, also called input features/parameters or independent variables.** Usually a set of <20 variables are chosen to investigate their effects on the experimental outcomes. Various types of variables are supported:
    - Numerical variables:
        + Continuous numerical variable: variables that can have an infinite number of real values within a given interval, e.g. the pH value of liquid. 
        + Discrete numerical variable: variables that have finite number of numerical values within a given range or list of numbers, e.g. the oxidation state of an element. 
    - Categorical variables:
        + Ordinal variables: Discrete variable that are defined by a meaningful order relationship between different categories, e.g. light intensity stages ranging from level 1 (weaker) to level 5 (stronger).
        + Nominal variables: Discrete variable that describes a name, label or category without a natural order, e.g. strings of batch number. 
    - Observational numerical variable: Used for model fitting but is not leveraged during optimization, i.e. the value is manually specified. (observational discrete variable?)

* **$X_{space}$: Experimental design space, or input parameter constraints.** It is a multidimensional parameter space within which the experimental conditions $X$ could be chosen and manipulated. 
    - For each continuous variable in $X$, the range of min and max range should be specified. 
    - For each discrete or categorical variable, the set of possible distinct values should be provided. 

* **$Y$: Experimental Outcomes, also called responses or dependent variables.** It is the target variable to be optimized. Currently we only consider continuous variables as experimental outcomes. Depends on the number of outcomes to be optimized simultaneously, $Y$ could be either a scalar (single-objective optimization) or a vector (multi-objective optimization). For a multi-objective optimization problem, the user could specify whether to maximize or minimize each each element in the $Y$ vector. 


See section 
[Data](3_Data.md)
and submodule
[`parameters`](https://github.com/MSDLLCpapers/obsidian/tree/main/obsidian/parameters) 
for more details. 




---
## Surrogate Model

Assume the true relationship between the experiment outcome and input is $f(X)$, which is unknown since the random and/or systematic errors $\epsilon$ always present in experiments. 

The measured outcome could be denoted as $Y(X) = f(X) + \epsilon$. When the variance of $\epsilon$ is constant, the noise is homoscedastic; otherwise when the noise is heteroscedastic, the variance of $\epsilon$ could depend on $X$ and making the optimization problem more challenging. 

The surrogate model $\hat{f}(X)$ is a predictive model that approximate the underlying unknown relationship $f(X)$. The $\hat{f}(X)$ has the same dimension as $Y$.
If $Y$ is a vector, the experimental outcomes could be modeled jointly with multivariate multiple regression modeling methods at once; or each outcome could be modeled separately in parallel.

Ideally, the surrogate models should provide both a point prediction (e.g. conditional mean given $X$) as well as a prediction uncertainty metric (e.g. conditional standard deviation given $X$). Since many of the acquisition functions require both point predictions and the prediction uncertainty to balance the exploration and exploitation of design space. 

The most popular choice of surrogate model is Gaussian Process (GP) regression, which is a bayesian model that combines prior knowledge with observed data to make probabilistic inferences of the posterior distribution. Thus, it is convenient to compute the posterior mean and standard deviation for each new input data.

The SMBO workflow starts with collecting a set of initial experimental data $(X_0,Y_0)$, where the $X_0$ could be random sampled from $X_{space}$ or suggested by design-of-experiments methods. Then fit the initial surrogate model $\hat{f}_0$ based on this initial dataset. 
At the $i^{th}$ iteration later, an updated surrogate model $\hat{f}_i$ is trained on all the available data $\{X_t,Y_t\}_{1 \leq t \leq i}$. As more data are accumulated with multiple rounds of optimization, we expect the surrogate function become more precise and better approximate the actual underlying functional relationship.

See section
[Surrogate Model](4_SurrogateModel.md) 
and submodule
[`surrogates`](https://github.com/MSDLLCpapers/obsidian/tree/main/obsidian/surrogates)
for more details.



---
## Acquisition Function

The acquisition function aims to guide the search of promising new experimental conditions with the potential to achieve improved responses.
It usually depends on the surrogate model predictions and/or measured data , which scores any point in the $X_{space}$.

In the $i^{th}$ iteration, the acquisition function is 
```{math}
    a_i(X) = a(\hat{f}_i(X), \{X_t,Y_t\}_{1 \leq t \leq i})
```

And maximizing this scoring function over the design space gives the suggested new candidate conditions for the $(i+1)^{th}$ iteration:
\begin{equation*}
X_{i+1} = \underset{X' \in X_{space}}{\text{argmax }} a_i(X')
\end{equation*}

The definition of acquisition function determines the trade-off between exploration and exploitation based on user preference. 

Exploitation means leveraging existing knowledge by selecting new conditions that lead to optimal outcomes based on surrogate model prediction. It could maximize the immediate gains when the true relationship is relatively well understood and approximated by the surrogate model.

Exploration involves acquiring new information by sampling from less explored regions of the design space where the prediction uncertainty is higher. It may help to better understand the problem and improve surrogate model quality, which potentially lead to better long-term outcomes or reveal new possibilities.


See section [Acquisition Function](5_AcquisitionFunction.md), 
submodule [`acquisition`](https://github.com/MSDLLCpapers/obsidian/tree/main/obsidian/acquisition),
and submodule [`objectives`](https://github.com/MSDLLCpapers/obsidian/tree/main/obsidian/objectives)
for more details.


---
## Suggest New Experimental Conditions

One or more new candidate experimental condition(s) are suggested based on the criteria of maximizing the acquisition function within the experimental design space. More advanced [`constraints`](https://github.com/MSDLLCpapers/obsidian/tree/main/obsidian/constraints) on input or output space that involves the combination or relationship of multiple input variables could be applied during the search. 

With an accurate surrogate model and a well-designed acquisition function, the new candidate point that maximize the acquisition function is most likely to achieve improved experimental outcome(s). 

The optimization of acquisition function can be performed by analytical or numerical methods, depends on the complexity of acquisition functional form. 

Please refer to section **Tutorials** for simulation studies that demonstrates the entire SMBO workflow as well as various visualization options. 


