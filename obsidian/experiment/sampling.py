import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def generate_weights(df, n, bias, plot_weights=False, enforce=False, replace=False):
    """
    Generates a Pandas series of weights for each datum given a particular bias.

    df: DataFrame of candidates
    n:  size of the design to pick
    bias: dictionary of biases in the format : {"column": [lower_bound, upper_bound, relative_weight]}
        - Weight >1 increases sampling probability for in-range rows.
        - Weight <1 decreases it.
        - Weight = 0 excludes those rows entirely.
    plot_weights: boolean, whether to plot distribution of weights, default False
    enforce: boolean, whether to force biases, default False
    replace: boolean, whether sampling will be done with replacement, default False.
        When True, the enforce capacity check (qualifying rows >= n) is skipped
        because sampling n items with replacement from fewer qualifying rows is valid.

    Returns: Pandas Series of normalized row weights.
    """
    weights = pd.Series(1.0, index=df.index)
    for col, params in bias.items():
        lower, upper = params[0], params[1]
        weight = params[2] if len(params) > 2 else 1.0
        mask = df[col].between(lower, upper, inclusive="both")
        if enforce:
            weights *= mask.astype(float) * weight
        else:
            weights *= mask.astype(float) * weight + (~mask).astype(float) * 1.0
    if enforce and not replace:
        if (weights > 0).sum() < n:
            raise ValueError(f"Not enough rows ({(weights > 0).sum()}) satisfy all enforce conditions for n={n}.")
       
    total = weights.sum()
    if total == 0:
        raise ValueError(
            "All row weights are zero — the bias configuration excludes every candidate row."
        )
    weights = weights / total

    if plot_weights:
        print("Weights min:", weights.min(), "max:", weights.max())
        plt.figure(figsize=(8, 4))
        plt.hist(weights, bins=50)
        plt.title("Distribution of Sampling Weights")
        plt.xlabel("Weight")
        plt.ylabel("Count")
        plt.show()

    return weights


def sample_with_bias(df, n, replace=False, seed=None, bias=None, enforce=False, plot_weights=False):
    """
    Returns a random Pandas DataFrame sample of data points from a population with or without bias.

    df: DataFrame of candidates
    n:  int, size of the design to pick
    replace: boolean, allow or disallow sampling from the same row more than once, default False
    bias: dictionary of biases in the format : {"column": [lower_bound, upper_bound, relative_weight]}, default None
        - Weight >1 increases sampling probability for in-range rows.
        - Weight <1 decreases it.
        - Weight = 0 excludes those rows entirely.
    enforce: boolean, whether to force biases, default False
    plot_weights: boolean, whether to plot distribution of weights, default False

    Returns: Pandas DataFrame of sampled data points.
    """
    if bias:
        w = generate_weights(df, n, bias, plot_weights, enforce, replace=replace)
        return df.sample(n=n, replace=replace, random_state=seed, weights=w)
    else:
        return df.sample(n=n, replace=replace, random_state=seed)


def _space_filling_score(Z, metric="hybrid"):
    """
    Z: (k, d) standardized features of the candidate sample
    metric:
      - "maximin":   maximize the minimum pairwise distance
      - "mean_nn":   maximize the mean nearest-neighbor distance
      - "hybrid":    0.6*maximin + 0.4*mean_nn (more stable in practice)
    """
    k = Z.shape[0]
    if k < 2:
        return 0.0
    D = np.sqrt(((Z[:, None, :] - Z[None, :, :])**2).sum(-1))
    np.fill_diagonal(D, np.inf)
    d_min = D[np.triu_indices(k, 1)].min()
    d_mnn = D.min(axis=1).mean()
    if metric == "maximin":
        return d_min
    if metric == "mean_nn":
        return d_mnn
    if metric == "hybrid":
        return 0.6 * d_min + 0.4 * d_mnn
    raise ValueError("Unknown metric")


def best_sample(df, k, feature_cols, *, n_trials=500, bias=None, plot_weights=False, enforce=False,
                random_state=None, standardize=True, dropna=True, metric="hybrid"):
    """
    Repeats random sampling n_trials times and returns the most space-filling sample.

    df: DataFrame of candidates
    k:  size of the design to pick
    feature_cols: columns that define “space” (numeric; one-hot encode cats if needed)
    bias: None | dict in the format {“column”: [lower, upper, weight]} passed to
          generate_weights to bias sampling towards specified ranges.
    """
    base = df[feature_cols]
    idx = base.dropna().index if dropna else base.index
    dfv = df.loc[idx]
    Xfull = base.loc[idx].to_numpy(dtype=float)

    if bias:
        weights = generate_weights(dfv, k, bias, plot_weights, enforce)
    else:
        weights = None

    # standardize once using the FULL population (not per-trial) for fair geometry
    if standardize:
        mu = Xfull.mean(axis=0)
        sig = Xfull.std(axis=0)
        sig[sig == 0] = 1.0
        def toZ(X): return (X - mu) / sig
    else:
        def toZ(X): return X

    # prep weights aligned to the filtered df
    w = None
    if weights is not None:
        if isinstance(weights, str):
            w = dfv[weights]
        else:
            w = weights.reindex(dfv.index).fillna(0.0)

    if w is not None:
        nonzero = int((w > 0).sum())
        if nonzero < k:
            raise ValueError(
                f"Bias leaves only {nonzero} rows with positive weight, but "
                f"k={k} samples are requested without replacement. "
                "Widen the bias range or reduce k."
            )

    rng = np.random.default_rng(random_state)  # reproducible stream
    best_df = None
    best_score = -np.inf

    for _ in range(n_trials):
        cand = dfv.sample(n=k, replace=False, weights=w,
                          random_state=int(rng.integers(0, 2**31)))
        Z = toZ(cand[feature_cols].to_numpy(dtype=float))
        s = _space_filling_score(Z, metric=metric)
        if s > best_score:
            best_score = s
            best_df = cand

    return best_df, {"score": best_score, "metric": metric, "n_trials": n_trials}
