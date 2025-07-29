
from obsidian.exceptions import UnsupportedError
import numpy as np
from itertools import product
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy.stats import qmc
from scipy import linalg
from scipy.stats import chi2_contingency
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
import concurrent.futures
import seaborn as sns
import umap.umap_ as umap


def factorial_DOE(d: int,
                  n_CP: int = 3,
                  shuffle: bool = True,
                  seed: int | None = None,
                  full: bool = False):
    """
    Creates a statistically designed factorial experiment (DOE).
    Specifically for 2-level designs only.
    Uses the range (0,1) for low-high instead of the typical (-1,1),
    although (-1,1) is used for calculations during alias design

    Args:
        d (int): Number of dimensions/inputs in the design.
        n_CP (int, optional): The number of centerpoints to include in the design, for estimating
            uncertainty and curvature. Default is ``3``.
        shuffle (bool, optional): Whether or not to shuffle the design or leave them in the default run
            order. Default is ``True``.
        seed (int, optional): Randomization seed. Default is ``None``.
        full (bool, optional): Whether or not to run the full DOE. Default is ``False``, which
            will lead to an efficient Res4+ design.

    Returns:
        ndarray: An (m)-by-(d) array of experiments in the (0,1) domain

    Raises:
        UnsupportedError: If the number of dimensions exceeds 12
    """
    if d > 12:
        raise UnsupportedError('The number of dimensions must be 12 or fewer for DOE (currently)')
    
    steps = np.array([-1, 1])
    CP = np.zeros(shape=(n_CP, d))
    
    # Create a dictionary that allows us to use letters
    term_codes = {}
    alphabet = 'ABCDEFGHJKLMNOPQRSTUVWXYZ'  # DESIGN EXPERT SKIPS I!!!
    for i, letter in enumerate(alphabet):
        term_codes[letter] = i
    
    res4_aliases = {2: {}, 3: {},
                    4: {'D': 'ABC'},
                    5: {'E': 'ABCD'}, 6: {'E': 'ABC', 'F': 'BCD'},
                    7: {'F': 'ABCD', 'G': 'ABCE'}, 8: {'F': 'ABC', 'G': 'ABD', 'H': 'BCDE'},
                    9: {'G': 'ABCD', 'H': 'ACEF', 'J': 'CDEF'},
                    10: {'G': 'ABCD', 'H': 'ABCE', 'J': 'ADEF', 'K': 'BDEF'},
                    11: {'G': 'ABCD', 'H': 'ABCE', 'J': 'ABDE', 'K': 'ACDEF', 'L': 'BCDEF'},
                    12: {'H': 'ABC', 'J': 'ADEF', 'K': 'BDEG', 'L': 'CDFG', 'M': 'ABCEFG'}
                    }
    
    if full:
        axes = np.tile(steps, (d, 1))
        X = np.array(list(product(*axes)))
    else:
        if res4_aliases[d] == {}:
            d_resolved = d
        else:
            decoded_keys = [term_codes[k] for k in res4_aliases[d].keys()]
            d_resolved = np.min(decoded_keys)  # e.g. If the first aliased key is G, we are resolved up to F
        axes = np.tile(steps, (d_resolved, 1))
        X = np.array(list(product(*axes)))
        for term, generator in res4_aliases[d].items():
            decoded_generator = [term_codes[g] for g in list(generator)]
            aliased_term = np.product(X[:, decoded_generator], axis=-1)[:, np.newaxis]
            X = np.hstack((X, aliased_term))
    
    # Add centerpoints then shuffle
    X = np.vstack((X, CP))
    if seed is not None:
        np.random.seed(seed)
    if shuffle:
        np.random.shuffle(X)
    # Rescale from (-1,1) to (0,0.999999)
    X = (X+1)/2 - 1e-6
    
    return X

# --- Sampling Functions ---


def sample_continuous_lhs(continuous_params, n_samples, seed):
    sampler = qmc.LatinHypercube(d=len(continuous_params), seed=seed, scramble=True, strength=1, optimization='random-cd')
    sample_cont = sampler.random(n=n_samples)
    cont_samples = {}
    keys = list(continuous_params.keys())
    for idx, key in enumerate(keys):
        low, high, step = continuous_params[key]
        if step == 'geometric':
            possible = []
            value = low
            while value <= high:
                possible.append(value)
                value *= 2
            possible = np.array(possible)
            indices = np.floor(sample_cont[:, idx] * len(possible)).astype(int)
            indices = np.clip(indices, 0, len(possible) - 1)
            cont_samples[key] = possible[indices]
        elif step == 'logarithmic':
            if low <= 0 or high <= 0:
                raise ValueError(f"Logarithmic step requires positive low and high for parameter '{key}'")
            exp_low = int(np.floor(np.log10(low)))
            exp_high = int(np.floor(np.log10(high)))
            possible = 10.0 ** np.arange(exp_low, exp_high + 1)
            indices = np.floor(sample_cont[:, idx] * len(possible)).astype(int)
            indices = np.clip(indices, 0, len(possible) - 1)
            cont_samples[key] = possible[indices]
        else:
            num_steps = int(round((high - low) / step)) + 1
            possible = np.linspace(low, high, num_steps)
            indices = np.floor(sample_cont[:, idx] * num_steps).astype(int)
            indices = np.clip(indices, 0, num_steps - 1)
            cont_samples[key] = possible[indices]
    return cont_samples


def non_uniform_lhs_categorical(level_dict, n_samples, seed=None, scramble=True):
    levels = list(level_dict.keys())
    probabilities = [level_dict[level]['freq'] for level in levels]
    if not np.isclose(sum(probabilities), 1.0):
        probabilities = np.array(probabilities) / np.sum(probabilities)
    sampler = qmc.LatinHypercube(d=1, seed=20 + 3 * seed if seed is not None else None, scramble=scramble, strength=1, optimization='random-cd')
    uniform_samples = sampler.random(n=n_samples).flatten()
    cdf = np.cumsum(probabilities)
    results = []
    for i, sample in enumerate(uniform_samples):
        index = np.searchsorted(cdf, sample)
        level = levels[index]
        entry = {'level': level}
        for subparam, value in level_dict[level].items():
            if subparam == 'freq': continue
            values, weights = value
            weights = np.array(weights, dtype=float)
            weights /= weights.sum()
            sub_sampler = qmc.LatinHypercube(d=1, seed=seed + i if seed is not None else None, scramble=scramble, strength=1, optimization='random-cd')
            sub_sample = sub_sampler.random(n=1).flatten()[0]
            sub_cdf = np.cumsum(weights)
            sub_index = np.searchsorted(sub_cdf, sub_sample)
            entry[subparam] = values[sub_index]
        results.append(entry)
    return results


def optimize_category_assignment_parallel(cat_samples, conditional_subparameters, subparam_mapping, n_samples, seed, max_workers=4):
    category_key = list(subparam_mapping.keys())[0] if subparam_mapping else None
    if category_key is None: raise ValueError('subparam_mapping must be provided to infer buffer_key.')
    probabilities = [conditional_subparameters[category_key][lvl]['freq'] for lvl in conditional_subparameters[category_key].keys()]
    probabilities = np.array(probabilities) / np.sum(probabilities)
    other_cat_keys = [k for k in conditional_subparameters.keys() if k != category_key]
    
    def evaluate_assignment(j):
        sample_cat_entries = non_uniform_lhs_categorical(conditional_subparameters[category_key], n_samples, seed=3 * seed + 220 + j)
        sample_cat = [entry['level'] for entry in sample_cat_entries]
        temp_cat_samples = cat_samples.copy()
        temp_cat_samples[category_key] = sample_cat
        corr_matrix = calculate_mixed_correlation_matrix(pd.DataFrame(temp_cat_samples), categorical_vars=[category_key] + other_cat_keys)
        max_corr = max(abs(corr_matrix.loc[category_key, other_key]) for other_key in other_cat_keys)
        return max_corr, sample_cat
    
    best_category_assignment = None
    min_max_correlation = float('inf')
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(evaluate_assignment, j) for j in range(100)]
        for future in concurrent.futures.as_completed(futures):
            max_corr, sample_cat = future.result()
            if max_corr < min_max_correlation:
                min_max_correlation = max_corr
                best_category_assignment = sample_cat
                if min_max_correlation < 0.01:
                    break
    return best_category_assignment

def assign_conditional_subparameter(cat_samples, conditional_subparameters, parent_key, subparam_key, n_samples, seed):
    subparam_samples = [None] * n_samples
    level_indices = {}
    for i, level in enumerate(cat_samples[parent_key]):
        level_indices.setdefault(level, []).append(i)
    for level, indices in level_indices.items():
        level_info = conditional_subparameters[parent_key].get(level)
        if level_info is None or subparam_key not in level_info:
            raise ValueError(f"Level '{level}' missing or lacks '{subparam_key}' in conditional_subparameters")
        values, weights = level_info[subparam_key]
        weights = np.array(weights) / np.sum(weights)
        n_level_samples = len(indices)
        level_seed = seed + hash(level) % 1000
        sampled_values = non_uniform_lhs_categorical({str(v): {'freq': w} for v, w in zip(values, weights)}, n_level_samples, seed=level_seed)
        sampled_values = [float(d['level']) for d in sampled_values]
        for i, idx in enumerate(indices):
            subparam_samples[idx] = sampled_values[i]
    return subparam_samples


def infer_subparam_mapping(conditional_subparameters):
    mapping = {}
    for cat_param, levels in conditional_subparameters.items():
        subparam_candidates = set()
        for level_info in levels.values():
            subparams = [k for k in level_info if k != 'freq']
            subparam_candidates.update(subparams)
        if len(subparam_candidates) == 1:
            mapping[cat_param] = subparam_candidates.pop()
    return mapping



def sample_design(seed, n_samples, continuous_params, conditional_subparameters, subparam_mapping=None, optimize_categories=False):
    if subparam_mapping is None:
        subparam_mapping = infer_subparam_mapping(conditional_subparameters)

    cont_samples = sample_continuous_lhs(continuous_params, n_samples, seed)
    cat_samples = {}
    subparam_samples = {}

    # Identify the category to optimize (e.g., 'buffer_type')
    category_to_optimize = next((k for k in subparam_mapping if k in conditional_subparameters), None) if optimize_categories else None

    # Sample all other categorical variables first
    for cat_key, level_dict in conditional_subparameters.items():
        if optimize_categories and cat_key == category_to_optimize:
            continue  # Skip for now; optimize later
        samples = non_uniform_lhs_categorical(
            level_dict=level_dict,
            n_samples=n_samples,
            seed=seed + hash(cat_key) % 1000
        )
        cat_samples[cat_key] = [entry['level'] for entry in samples]

    # Optimize the selected category
    if optimize_categories and category_to_optimize:
        optimized_assignment = optimize_category_assignment_parallel(
            cat_samples=cat_samples,
            conditional_subparameters=conditional_subparameters,
            subparam_mapping=subparam_mapping,
            n_samples=n_samples,
            seed=seed
        )
        cat_samples[category_to_optimize] = optimized_assignment

    # Assign conditional subparameters (e.g., pH)
    for parent_key, subparam_key in subparam_mapping.items():
        if parent_key in cat_samples:
            subparam_samples[subparam_key] = assign_conditional_subparameter(
                cat_samples,
                conditional_subparameters,
                parent_key,
                subparam_key,
                n_samples,
                seed
            )

    # Combine all into a DataFrame
    design = pd.DataFrame({**cont_samples, **cat_samples, **subparam_samples})

    # Round and format continuous variables
    for key in continuous_params.keys():
        step = continuous_params[key][2]
        if step == 'geometric':
            design[key] = design[key].round(3)
        elif step == 'logarithmic':
            design[key] = design[key].round(5)
        elif isinstance(step, (int, float)):
            decimals = max(0, -int(np.floor(np.log10(step)))) if step < 1 else 0
            design[key] = design[key].round(decimals)
            if isinstance(step, int) or (isinstance(step, float) and step.is_integer()):
                design[key] = design[key].astype(int)
        else:
            design[key] = design[key].round(5)

    # Round subparameters like pH
    for subparam_key in subparam_mapping.values():
        if subparam_key in design.columns and np.issubdtype(design[subparam_key].dtype, np.number):
            design[subparam_key] = design[subparam_key].round(1)

    return design


# --- Efficient Mixed Correlation Matrix ---

def cramers_v_np(contingency):
    chi2, _, _, _ = chi2_contingency(contingency, correction=False)
    n = contingency.sum()
    min_dim = min(contingency.shape) - 1
    if min_dim == 0:
        return 0.0
    return np.sqrt(chi2 / (n * min_dim))


def eta_squared_np(cat_codes, num_values):
    overall_mean = np.mean(num_values)
    unique_cats, inverse = np.unique(cat_codes, return_inverse=True)
    counts = np.bincount(inverse)
    means = np.bincount(inverse, weights=num_values) / counts
    ss_between = np.sum(counts * (means - overall_mean) ** 2)
    ss_total = np.sum((num_values - overall_mean) ** 2)
    if ss_total == 0:
        return 0.0
    return ss_between / ss_total


def calculate_mixed_correlation_matrix(df, categorical_vars=None):
    columns = df.columns
    n_vars = len(columns)
    corr_matrix = np.eye(n_vars)

    if categorical_vars is None:
        categorical_vars = [col for col in columns if df[col].dtype.name in ['object', 'category']]
    numerical_vars = [col for col in columns if col not in categorical_vars]

    data = {}
    cat_codes = {}
    for col in columns:
        if col in categorical_vars:
            cat_codes[col] = df[col].astype('category').cat.codes.to_numpy()
        else:
            data[col] = df[col].to_numpy(dtype=float)

    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            var1, var2 = columns[i], columns[j]

            if var1 in numerical_vars and var2 in numerical_vars:
                x = data[var1]
                y = data[var2]
                if np.std(x) == 0 or np.std(y) == 0:
                    corr = 0.0
                else:
                    corr = np.corrcoef(x, y)[0, 1]

            elif var1 in categorical_vars and var2 in categorical_vars:
                x = cat_codes[var1]
                y = cat_codes[var2]
                n_x = x.max() + 1
                n_y = y.max() + 1
                contingency = np.zeros((n_x, n_y), dtype=int)
                np.add.at(contingency, (x, y), 1)
                corr = cramers_v_np(contingency)

            else:
                if var1 in categorical_vars:
                    cat = cat_codes[var1]
                    num = data[var2]
                else:
                    cat = cat_codes[var2]
                    num = data[var1]
                corr = np.sqrt(eta_squared_np(cat, num))

            corr_matrix[i, j] = corr
            corr_matrix[j, i] = corr

    return pd.DataFrame(corr_matrix, index=columns, columns=columns)


# --- Design Quality Metrics ---

def calculate_d_optimality(design, continuous_params_keys, pH_key):
    continuous_keys = continuous_params_keys + [pH_key]
    X = design[continuous_keys].values
    X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    X_model = np.column_stack((np.ones(X_std.shape[0]), X_std))
    XtX = X_model.T @ X_model
    return linalg.det(XtX)


def calculate_a_optimality(design, continuous_params_keys, pH_key):
    continuous_keys = continuous_params_keys + [pH_key]
    X = design[continuous_keys].values
    X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    X_model = np.column_stack((np.ones(X_std.shape[0]), X_std))
    XtX = X_model.T @ X_model

    try:
        XtX_inv = np.linalg.inv(XtX)
        return np.trace(XtX_inv)
    except np.linalg.LinAlgError:
        return np.inf


def calculate_condition_number(design, continuous_params_keys, pH_key):
    continuous_keys = continuous_params_keys + [pH_key]
    X = design[continuous_keys].values
    X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    X_model = np.column_stack((np.ones(X_std.shape[0]), X_std))
    XtX = X_model.T @ X_model
    return np.linalg.cond(XtX)


def calculate_pairwise_distance_uniformity(design, continuous_params_keys, pH_key):
    continuous_keys = continuous_params_keys + [pH_key]
    X = design[continuous_keys].values
    X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    distances = pdist(X_std, metric='euclidean')
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)

    return std_dist / mean_dist if mean_dist != 0 else np.inf


def calculate_max_continuous_correlation(design, continuous_params_keys, pH_key):
    continuous_keys = continuous_params_keys + [pH_key]
    corr_matrix = design[continuous_keys].corr().abs()
    np.fill_diagonal(corr_matrix.values, 0)
    return corr_matrix.values.max()


def calculate_max_categorical_correlation(design, categorical_keys):
    corr_matrix = calculate_mixed_correlation_matrix(design[categorical_keys], categorical_vars=categorical_keys)
    np.fill_diagonal(corr_matrix.values, 0)
    return corr_matrix.abs().values.max()


# --- Dimensionality Reduction Plots ---

def plot_pca(design, continuous_params_keys, subparam_mapping, hue=None):
    pH_key = list(subparam_mapping.values())[0]
    continuous_keys = continuous_params_keys + [pH_key]
    X = design[continuous_keys].values
    X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_std)

    plt.figure(figsize=(8, 6))
    if hue and hue in design.columns:
        categories = design[hue].astype(str)
        for cat in categories.unique():
            mask = categories == cat
            plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=cat, alpha=0.7)
        plt.legend(title=hue)
    else:
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c='blue', alpha=0.7)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA of Continuous Design Variables')
    plt.grid(True)
    plt.show()



def plot_mds(design, continuous_params_keys, subparam_mapping, hue=None, metric='euclidean'):
    pH_key = list(subparam_mapping.values())[0]
    continuous_keys = continuous_params_keys + [pH_key]
    X = design[continuous_keys].values
    X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    mds = MDS(n_components=2, dissimilarity=metric, random_state=42, n_init=4)
    X_mds = mds.fit_transform(X_std)

    plt.figure(figsize=(8, 6))
    if hue and hue in design.columns:
        categories = design[hue].astype(str)
        for cat in categories.unique():
            mask = categories == cat
            plt.scatter(X_mds[mask, 0], X_mds[mask, 1], label=cat, alpha=0.7)
        plt.legend(title=hue)
    else:
        plt.scatter(X_mds[:, 0], X_mds[:, 1], c='green', alpha=0.7)
    plt.xlabel('MDS Dimension 1')
    plt.ylabel('MDS Dimension 2')
    plt.title('MDS of Continuous Design Variables')
    plt.grid(True)
    plt.show()



def plot_umap(design, continuous_params_keys, subparam_mapping, hue=None, n_neighbors=15, min_dist=0.1, metric='euclidean'):
    pH_key = list(subparam_mapping.values())[0]
    continuous_keys = continuous_params_keys + [pH_key]
    X = design[continuous_keys].values
    X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, random_state=42, n_jobs=1)
    X_umap = reducer.fit_transform(X_std)

    plt.figure(figsize=(8, 6))
    if hue and hue in design.columns:
        categories = design[hue].astype(str)
        for cat in categories.unique():
            mask = categories == cat
            plt.scatter(X_umap[mask, 0], X_umap[mask, 1], label=cat, alpha=0.7)
        plt.legend(title=hue)
    else:
        plt.scatter(X_umap[:, 0], X_umap[:, 1], c='red', alpha=0.7)
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.title('UMAP of Continuous Design Variables')
    plt.grid(True)
    plt.show()


# --- Design Evaluation Function ---

def evaluate_design(design, continuous_keys, categorical_keys, subparam_mapping, metrics_to_optimize):
    """
    Evaluates an existing design using specified metrics.

    :param design: The design DataFrame to evaluate.
    :param continuous_keys: List of continuous parameter names.
    :param categorical_keys: List of categorical parameter names.
    :param subparam_mapping: Dictionary mapping categorical variable to its subparameter (e.g., {'buffer_type': 'pH'}).
    :param metrics_to_optimize: List of metric names to compute.
    :return: Dictionary of computed metrics.
    """
    subparam_key = list(subparam_mapping.values())[0] if subparam_mapping else None

    metrics = {
        'D-optimality': calculate_d_optimality(design, continuous_keys, subparam_key),
        'A-optimality': calculate_a_optimality(design, continuous_keys, subparam_key),
        'Condition Number': calculate_condition_number(design, continuous_keys, subparam_key),
        'Pairwise Distance CV': calculate_pairwise_distance_uniformity(design, continuous_keys, subparam_key),
        'Max Continuous Corr': calculate_max_continuous_correlation(design, continuous_keys, subparam_key),
        'Max Categorical Corr': calculate_max_categorical_correlation(design, categorical_keys)
    }

    return {k: v for k, v in metrics.items() if k in metrics_to_optimize}


# Helper functions for 'optimize design'


def generate_and_evaluate(seed, n_samples, continuous_params, conditional_subparameters,
                          subparam_mapping, continuous_keys, categorical_keys, metrics_to_optimize):
    design = sample_design(
        seed=seed,
        n_samples=n_samples,
        continuous_params=continuous_params,
        conditional_subparameters=conditional_subparameters,
        subparam_mapping=subparam_mapping
    )
    metrics = evaluate_design(
        design=design,
        continuous_keys=continuous_keys,
        categorical_keys=categorical_keys,
        subparam_mapping=subparam_mapping,
        metrics_to_optimize=metrics_to_optimize
    )
    metric_values = [metrics[m] for m in metrics_to_optimize]
    return {'seed': seed, 'design': design, **metrics, 'metric_values': metric_values}


def find_best_design_parallel(n, n_samples, continuous_params, conditional_subparameters,
                              subparam_mapping=None,
                              metrics_to_optimize=None, maximize_metrics=None,
                              seed_start=0, max_workers=None):
    if subparam_mapping is None:
        subparam_mapping = infer_subparam_mapping(conditional_subparameters)
    if metrics_to_optimize is None:
        metrics_to_optimize = ['D-optimality', 'A-optimality', 'Condition Number',
                               'Pairwise Distance CV', 'Max Continuous Corr', 'Max Categorical Corr']
    if maximize_metrics is None:
        maximize_metrics = [True] + [False] * (len(metrics_to_optimize) - 1)

    continuous_keys = list(continuous_params.keys())
    categorical_keys = list(conditional_subparameters.keys())

    
    records = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                generate_and_evaluate,
                seed_start + i,
                n_samples,
                continuous_params,
                conditional_subparameters,
                subparam_mapping,
                continuous_keys,
                categorical_keys,
                metrics_to_optimize
            )
            for i in range(n)
        ]
        for future in concurrent.futures.as_completed(futures):
            records.append(future.result())

    metric_array = np.array([r['metric_values'] for r in records])
    norm_metrics = []
    for idx, m in enumerate(metrics_to_optimize):
        vals = metric_array[:, idx]
        min_val, max_val = vals.min(), vals.max()
        norm = (vals - min_val) / (max_val - min_val) if max_val > min_val else np.zeros_like(vals)
        if not maximize_metrics[idx]:
            norm = 1 - norm
        norm_metrics.append(norm)
    scores = np.sum(norm_metrics, axis=0)

    for i, r in enumerate(records):
        r['score'] = scores[i]

    best_idx = np.argmax(scores)
    best_design = records[best_idx]['design']
    metrics_df = pd.DataFrame([{k: v for k, v in r.items() if k not in ['design', 'metric_values']} for r in records])

    return best_design, metrics_df


def plot_design_quality_evolution(metrics_df):
    metrics_df = metrics_df.sort_values('seed')

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    metrics = ['D-optimality', 'A-optimality',
               'Pairwise Distance CV', 'Max Continuous Corr', 'Max Categorical Corr', 'score']

    for i, metric in enumerate(metrics):
        ax = axes[i // 3, i % 3]
        ax.bar(metrics_df['seed'].astype(str), metrics_df[metric])
        ax.set_title(f'{metric} vs Seed')
        ax.set_xlabel('Seed')
        ax.set_ylabel(metric)
        ax.grid(axis='y')

    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(design, categorical_vars):
    corr_df = calculate_mixed_correlation_matrix(design, categorical_vars)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_df, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
    plt.title("Mixed Correlation Matrix")
    plt.tight_layout()
    plt.show()

def plot_design_histograms(design, continuous_keys, categorical_keys, subparam_mapping=None, bins=50, figsize=(18,10)):
    # Determine total_plots based on the lengths of continuous, categorical, and subparameter mappings
    total_plots = len(continuous_keys) + len(categorical_keys) + (len(subparam_mapping) if subparam_mapping else 0)
    
    # Setup grid for subplots
    ncols = 3
    nrows = math.ceil(total_plots / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()

    # Plot continuous histograms
    for i, key in enumerate(continuous_keys):
        ax = axes[i]
        ax.hist(design[key].dropna(), bins=bins, color='skyblue', edgecolor='black')
        ax.set_title(f'Histogram of {key}')
        ax.set_xlabel(key)
        ax.set_ylabel('Frequency')

    # Plot categorical bar plots
    offset = len(continuous_keys)
    for j, key in enumerate(categorical_keys):
        ax = axes[offset + j]
        counts = design[key].dropna().value_counts()
        ax.bar(counts.index.astype(str), counts.values, color='lightgreen', edgecolor='black')
        ax.set_title(f'Bar plot of {key}')
        ax.set_xlabel(key)
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=45)

    # Plot histograms for each subparameter mapping
    if subparam_mapping:
        offset = len(continuous_keys) + len(categorical_keys)
        for k, (cat_key, sub_key) in enumerate(subparam_mapping.items()):
            ax = axes[offset + k]
            sns.histplot(data=design, x=sub_key, hue=cat_key, bins=bins, multiple='stack', palette='Set2', edgecolor='black', ax=ax)
            ax.set_title(f'Histogram of {sub_key} by {cat_key}')
            ax.set_xlabel(sub_key)
            ax.set_ylabel('Count')

    # Turn off unused axes
    for k in range(total_plots, len(axes)):
        axes[k].axis('off')

    plt.tight_layout()
    plt.show()




def evaluate_candidate(i, seed_start, n, continuous_params, conditional_subparameters,
                       subparam_mapping, existing_design, continuous_keys, categorical_keys,
                       metrics_to_optimize):
    if subparam_mapping is None:
        subparam_mapping = infer_subparam_mapping(conditional_subparameters)

    buffer_key = list(subparam_mapping.keys())[0]
    pH_key = list(subparam_mapping.values())[0]

    seed = seed_start + i
    new_samples = sample_design(
        seed, n, continuous_params, conditional_subparameters, subparam_mapping
    )
    combined_design = pd.concat([existing_design, new_samples], ignore_index=True)

    metrics = {
        'D-optimality': calculate_d_optimality(combined_design, continuous_keys, pH_key),
        'A-optimality': calculate_a_optimality(combined_design, continuous_keys, pH_key),
        'Condition Number': calculate_condition_number(combined_design, continuous_keys, pH_key),
        'Pairwise Distance CV': calculate_pairwise_distance_uniformity(combined_design, continuous_keys, pH_key),
        'Max Continuous Corr': calculate_max_continuous_correlation(combined_design, continuous_keys, pH_key),
        'Max Categorical Corr': calculate_max_categorical_correlation(combined_design, categorical_keys)
    }

    return {
        'seed': seed,
        'metrics': metrics,
        'metric_values': [metrics[m] for m in metrics_to_optimize],
        'new_samples': new_samples
    }


def extend_design(existing_design, n, continuous_params, conditional_subparameters,
                  subparam_mapping=None,
                  metrics_to_optimize=None, maximize_metrics=None,
                  num_candidates=10, seed_start=1000, max_workers=None):
    if subparam_mapping is None:
        subparam_mapping = infer_subparam_mapping(conditional_subparameters)
    if metrics_to_optimize is None:
        metrics_to_optimize = ['D-optimality', 'A-optimality', 'Condition Number',
                               'Pairwise Distance CV', 'Max Continuous Corr', 'Max Categorical Corr']
    if maximize_metrics is None:
        maximize_metrics = [True] + [False] * (len(metrics_to_optimize) - 1)

    continuous_keys = list(continuous_params.keys())
    categorical_keys = list(conditional_subparameters.keys())

    records = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                evaluate_candidate,
                i,
                seed_start,
                n,
                continuous_params,
                conditional_subparameters,
                subparam_mapping,
                existing_design,
                continuous_keys,
                categorical_keys,
                metrics_to_optimize
            )
            for i in range(num_candidates)
        ]
        for future in concurrent.futures.as_completed(futures):
            records.append(future.result())

    metric_array = np.array([r['metric_values'] for r in records])
    norm_metrics = np.array([
        (vals - vals.min()) / (vals.max() - vals.min()) if vals.max() != vals.min() else np.zeros_like(vals)
        for vals in metric_array.T
    ])
    for idx, maximize in enumerate(maximize_metrics):
        if not maximize:
            norm_metrics[idx] = 1 - norm_metrics[idx]
    scores = norm_metrics.sum(axis=0)

    best_idx = np.argmax(scores)
    best_extension = records[best_idx]['new_samples']
    extended_design = pd.concat([existing_design, best_extension], ignore_index=True)

    metrics_summary = pd.DataFrame([
        {**{'seed': r['seed'], 'score': s}, **r['metrics']}
        for r, s in zip(records, scores)
    ])

    return extended_design, metrics_summary



