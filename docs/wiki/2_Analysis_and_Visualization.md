# Additional Analysis and Visualization


![APO Workflow](https://github.com/MSDLLCpapers/obsidian/blob/main/docs/_static/APO_workflow.png?raw=true)

This section introduces some additional components, such as retrospective analysis and visualization methods, that are not essential steps in a Sequential Model-Based Optimization (SMBO) algorithm but are valuable in real-world applications for several reasons.

* The technical details involved in SMBO algorithms, such as surrogate models and acquisition functions, may seem complicated to users with non-quantitative background. As a result, the entire workflow of suggesting new experimental conditions may appear to be a black box for users, which hampers the adoption of this this powerful process optimization technique. Various performance metrics and model interpretation methods help to bridge this gap by providing users with a better intuitive understanding of the underlying algorithms and revealing the decision-making processes involved.

* The variable importance analysis and/or model interpretation tools can provide critical insights into the optimization process, aiding in a deeper understanding of the variables that influenced the selection of optimal solution and the relationships between input variables, which could be confirmed with additional experiments or scientific domain experts.

* Providing prediction uncertainty metrics along with the suggested candidates empowers practitioners to make informed decisions and establish realistic expectations of the surrogate model's predictive performance and potential biases.


Please refer to submodule [`campaign`](https://github.com/MSDLLCpapers/obsidian/tree/main/obsidian/campaign) for more details of analysis, 
and submodule [`plotting`](https://github.com/MSDLLCpapers/obsidian/tree/main/obsidian/plotting) folder for visualization utilities.

---
## Post-hoc Data Analysis 

To simplify the description, assuming the preferred direction of any response variable is maximizing, and there are $N$ data points collected for $J$ experimental outcomes: $\{ (X_i,Y_i) \}_{i=1}^N$.

If there are multiple experimental outcomes ($J \geq 1$), the $j^{th}$ outcome in vector $Y_i$ is denoted as $Y_i^{j}$.

### Evaluating the Measured Experimental Conditions 

A direct assessment of input features is assigning a binary indicator (True/False) to identify the best performer $X_{opt}$ or an opimal solution set $\chi_{opt}$.
 
* Single-objective optimization: The best performer is the one (or more) inputs that achieve the optimal value of measured experimental outcome.
  ```{math}
      X_{opt} = \underset{X_i, 1\leq i\leq N}{\text{argmax }} Y(X_i)
  ```

* Multi-objective optimization: 
    - In some simple cases when the multiple responses $Y^j (1 \leq j \leq J)$ are correlated, there may exist some best performer $X_{opt}$ for all responses simutaneously:
      ```{math}
          X_{opt} = \underset{1 \leq j \leq J}{\cap} \underset{X_i, 1\leq i\leq N}{\text{argmax }} Y^j(X_i) 
      ```
    - In most cases, there is trade-off between multiple responses and they cannot be optimized together. The **Pareto Front** ($PF_Y$) is the set of compromise solution, or nondominated solution, which are better than others in terms that improvement in any response $Y_s$ comes only at the expense of making at least another response $\{ Y_t \}_{t\neq s}$ worse:
      \begin{align*}
          PF_Y = \{Y_i, i \in [1,N]: \{Y_w, w \in [1,N], w\neq i: Y_w^j > Y_i^j, \forall j \in [1,J] \} = \emptyset \}
      \end{align*}
      And the optimal solution set in feature space or Pareto optimal designs is the set:
      \begin{gather*}
          \chi_{opt} = \{X_i, 1\leq i\leq N: Y_i \in PF_Y\}
      \end{gather*}


After multiple iterations, as the recommened candidates converge towards the best-performing or optimal solution set, multiple data points would exhibit near optimal measurements.
However, due to the presence of experimental errors in measuring $Y$, the aforementioned best performer set may not accurately optimize the unknown function $f(X)$. Furthermore, minor deviations in the design space may not have a significant impact on the outcome values. Consequently, we are interested in expanding the the optimal set to include data points within its close neighborhood. 

For single-objective optimization, the distance between the $i^{th}$ outcome to the best performer $\underset{1\leq i' \leq N}{\max Y_{i'}\}-Y_i$ can be used to rank the data points. 
For multi-objective optimization, the evaluation is subjective to user preference. Here are several common options to assign continuous score for each data point: 
* Its distance to the pareto front $dist(Y_i, PF_Y)$. Various distrance metrics could be explored.
* Its distance to an **utopian point** $Y_U$, which is the desirable experimental outcome(s) specify by the user and doesn't need to be feasible.
* A weighted average across multiple outcomes $\sum_{1\leq j \leq J} w^j * Y_i^j$ (weights are normalized: $\sum_{1\leq j \leq J} w^j = 1, \forall w_j \geq 0$), where the weights $\{w^j\}_{1\leq j \leq J}$ are proportional to the subjective importance of each outcome. In this retrospective analysis, the weights could be different from the weights or formula used in defining the optimization objective. 




### The Overall Optimization Performance Metrics

To monitor the progress of SMBO workflow, we need to define a scalar evaluation metric to summarize performance over all the $N$ data points. 

* Single-objective optimization: The optimal value (either max or min, depends on target specification) of measured experimental outcome. 

* Multi-objective optimization:
  - Hypervolume: First define a **reference point** $r_Y$ in the outcome space that serves as a baseline for evaluating the quality of measured outcomes, i.e. it defines the minimum acceptable value for each outcome. The reference point is often set as the lower bounds for each response. The hypervolume indicator metric $HV$ is the volume of the space enclosed between the reference point $r_Y$ and the Pareto front $PF_Y$. It access the quality of a pareto front solution set: the larger the better. The space can be computed as the union of hyperrectangles bounded by each point in $PF_Y$ and $r_Y$ as vertices:
    \begin{equation*}
        HV(\{Y_i\}_{i=1}^N, r_Y) = HV(PF_Y, r_Y) = Volume(\underset{y \in PF_Y}{\cup} [y,r_Y])
    \end{equation*}
  - Maximum weighted outcome: The maximum weighted outcomes $\underset{1\leq i \leq N}{\text{max }} \sum_{1\leq j \leq J} w^j * Y_i^j$
  - Minimum distance to the utopian point: $\underset{1\leq i \leq N}{\text{min }} dist(Y_i,Y_U)$


---
## Surrogate Model Interpretation

### SHAP (SHapley Additive exPlanations) 

### Partial Dependence Plot

### Individual Conditional Expectation 

### Sensitivity Analysis 





---
## Augmented Predictive Information for Candidates



### Prediction Uncertainty
(TBA...)





---
## Miscellaneous

(TBA...)
