# Parameters of scipy.stats.qmc.LatinHypercube

The `LatinHypercube` class generates Latin Hypercube Samples (LHS) in a multi-dimensional unit hypercube. It supports several parameters to control the sampling behavior, randomness, and sample quality.

## Parameters

### 1. `d` : int

**Description:**  
The dimension of the sampling space, i.e., the number of parameters or variables to sample simultaneously.

**Possible values:**  
Any positive integer (d > 0).

**Effect:**  
Determines the number of columns in the sample matrix. Each sample point is a vector of length d with values in [0, 1].

### 2. `seed` : int, array_like, np.random.Generator, or None, optional (default: None)

**Description:**  
Controls the random number generator used for sampling and scrambling.

**Possible values:**  
- An integer seed for reproducibility.
- An instance of `np.random.Generator` for custom RNG.
- An array-like seed.
- `None` to use the default RNG.

**Effect:**  
Using a fixed seed ensures reproducible sampling results. Different seeds produce different sample sets.

### 3. `scramble` : bool, optional (default: False)

**Description:**  
Whether to apply scrambling to the Latin Hypercube design.

**Possible values:**  
- `True`: Apply scrambling.
- `False`: No scrambling.

**Effect:**  
Scrambling adds randomness to the sample points while preserving the stratification property of LHS. This reduces correlation and improves uniformity, often resulting in better space-filling designs.

### 4. `strength` : int, optional (default: 1)

**Description:**  
The strength of the orthogonal array used to construct the LHS.

**Possible values:**  
- `1`: Standard Latin Hypercube (default).
- `2` or higher: Higher strength orthogonal arrays, which enforce stronger uniformity constraints on projections of the sample points.

**Effect:**  
Increasing strength improves uniformity in lower-dimensional projections of the sample but may reduce the number of feasible samples and increase computational complexity.

### 5. `optimization` : str or None, optional (default: None)

**Description:**  
Method used to optimize the LHS design to improve space-filling properties.

**Possible values:**  
- `'random-cd'`: Random coordinate descent optimization.
- `'centered'`: Centered Latin Hypercube design.
- `'maximin'`: Maximize the minimum distance between points.
- `None`: No optimization applied.

**Effect:**  
Optimization attempts to improve the distribution of points by reducing clustering and increasing uniformity. Different methods have different computational costs and effectiveness:
- `'random-cd'`: Iteratively improves the design by random coordinate swaps.
- `'centered'`: Places points at the center of intervals for better uniformity.
- `'maximin'`: Maximizes the minimum pairwise distance between points, improving space-filling.

## Summary Table

| Parameter | Type | Default | Possible Values | Effect Summary |
|-----------|------|---------|-----------------|----------------|
| `d` | int | — | Positive integers | Number of dimensions sampled |
| `seed` | int, array_like, RNG, None | `None` | Integer seed, RNG, or `None` | Controls reproducibility of samples |
| `scramble` | bool | `False` | `True` or `False` | Adds randomness to reduce correlation and improve uniformity |
| `strength` | int | `1` | 1, 2, 3, ... | Orthogonality strength; higher values improve uniformity in projections |
| `optimization` | str or None | `None` | `'random-cd'`, `'centered'`, `'maximin'`, or `None` | Optimizes sample distribution for better space-filling |

## Notes

### Choosing `scramble`:
Scrambling is generally recommended for better sample quality unless you need a deterministic, non-random design.

### Choosing `strength`:
Use `strength=1` for standard LHS. Higher strengths improve uniformity but may limit sample size and increase complexity.

### Choosing `optimization`:
Optimization improves sample uniformity but increases computation time. `'random-cd'` is a good balance for many applications.

### Reproducibility:
Always set `seed` if you want reproducible results, especially when using scrambling or optimization.
