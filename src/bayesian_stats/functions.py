from typing import Any, Iterable

import numpy as np
import pandas as pd

from numpy.typing import NDArray


__all__ = [
    "get_auto_corr",
    "get_effective_sample_size",
    "get_gelman_rubin_diagnostic",
    "get_highest_density_interval",
    "get_invgamma_params",
    "get_probability_non_zero",
    "get_rolling_windows",
    "one_hot_encode",
]


def get_auto_corr(
    x: NDArray[Any], lags: int | Iterable[int] | NDArray[np.int64] | None = None
) -> NDArray[np.float64]:
    """Auto-correlation of a vector for specific lags

    Parameters
    ----------
    x: NDArray[Any], shape=(N,)
    lags: int | Iterable[int] | NDArray[np.int64] | None
        Lag or lags to calculate autocorrelation for.
        If lags is an integer, will assume the first ``m`` lags,
        where ``m`` is the value passed.
        If ``None`` assumes all lags.
        Otherwise, assumes value passes is an iterable of specific
        lags to calculate autocorrelation for.


    Returns
    -------
    auto_corr: NDArray[np.float64], shape=(n_lags,)
        Vector of auto-correlations
    """
    if x.ndim > 1:
        raise ValueError("Input must be a 1d array")

    n = len(x)

    if lags is None:
        lags = max(0, n - 2)

    if isinstance(lags, int):
        lags = np.arange(lags)

    if not isinstance(lags, np.ndarray):
        lags = np.array(lags)

    if lags.ndim != 1:
        raise ValueError("Lags must be a 1d array")

    lags = lags.astype(int)

    if lags.min() < 0:
        raise ValueError("Min lag must >= 0")

    if lags.max() >= n - 1:
        raise ValueError(f"Max lag must < {n - 1}")

    return np.array([1.0 if l == 0 else np.corrcoef(x[l:], x[:-l])[0, 1] for l in lags])


def get_effective_sample_size(
    chain: NDArray[Any], lags: int, burn_in_frac: float = 0.5
) -> float:
    """Estimates the effective sample size based on the auto-correlation of
    a posterior MCMC sample chain.

    Parameters
    ----------
    chain: NDArray[Any], shape=(N,)
        Posterior MCMC samples for a given model parameter for a single chain.
    lags: int
        Number of auto-correlation lags.
    burn_in_frac: float, optional
        Remove the first x% of chain samples for burn-in.
        (default = 0.5)

    Returns
    -------
    effective_sample_size: int
    """

    # Discard the first x% of samples (burn-in period)
    n = len(chain)
    st = int(n * burn_in_frac)
    chain = chain[st:]

    # Calc auto-corr lags
    auto_corr = get_auto_corr(chain, lags=np.arange(1, lags))

    # Isolate first series of contiguous positive lags
    lag_idx = np.max(np.cumsum(np.cumprod(auto_corr >= 0)))

    # Scale number of samples down by scale factor
    scale_factor = 1 / (1 + 2 * (auto_corr[:lag_idx].sum()))

    effective_sample_size = n * scale_factor

    return effective_sample_size


def get_gelman_rubin_diagnostic(
    chains: NDArray[Any], burn_in_frac: float = 0.5
) -> float:
    """Gelman-Rubin Convergence Diagnostic

    Parameters
    ----------
    chains: NDArray[Any], shape=(M, N)
        Posterior MCMC samples for a given model parameter.
    burn_in_frac: float, optional
        Remove the first x% of chain samples for burn-in
        (default = 0.5)

    Returns
    -------
    scale_reduction_factor: float
        The Gelman-Rubin convergence statistic for the given model parameter

    Notes
    -----
    - The Gelman-Rubin convergence diagnostic provides a numerical convergence
    statistic (scale reduction factor) based on the comparison of multiple MCMC chains.
    - If we were to start multiple parallel chains in many different starting
    values, the theory claims that they should all eventually converge to the
    stationary distribution.
    - One way to assess this is to compare the variation between chains to the
    variation within the chains: `R ~ Between chain variance / within chain variance`
    - Brooks and Gelman (1998) suggest that diagnostic values greater than 1.2
    for any of the model parameters indicate nonconvergence.
    In practice, a more stringent rule of `R < 1.1` is often used to declare convergence.
    """
    # M Chains of length N
    chains = np.array(chains)
    m, n = chains.shape
    n = int(n * burn_in_frac)

    # Discard the first N draws (burn-in period)
    chains = chains[:, n:]

    # Calculate within chain variance
    chain_vars = chains.var(-1)
    w = chain_vars.mean()

    # Between chain variance
    chain_means = chains.mean(-1)
    global_mean = chains.mean()
    b = np.sum((chain_means - global_mean) ** 2) * n / (m - 1)

    # Calculate the estimated variance as the weighted sum of between and within chain variance.
    est_var = (1 - 1 / n) * w + 1 / n * b

    # Calculate the potential scale reduction factor.
    # We want this number to be close to 1.
    # This would indicate that the between chain variance is small.
    scale_reduction_factor = (est_var / w) ** 0.5

    return scale_reduction_factor


def get_highest_density_interval(
    x: NDArray[Any], confidence_level: float = 0.95, axis: int | None = None
) -> NDArray[Any]:
    """Highest Posterior Density credible interval (HDI)

    Parameters
    ----------
    x: NDArray[Any]
        Posterior samples.
    confidence_level: float, optional
        Confidence level for credible interval.
    axis: int, optional
        Axis to summarize HDI for.
        If ``None``, will flatten data and summarize over all.

    Returns
    -------
    credible_intervals: NDArray[Any]

    Examples
    --------
    MCMC samples, 5 chains, 1000 samples, 3 variables
    >>> x = np.random.randn(5, 1000, 3)

    90% HDI for all three variables across samples from all 5 chains
    >>> get_highest_density_interval(x, 0.9, -1)
    array([[-1.63800563,  1.57380441],
       [-1.5237949 ,  1.6969538 ],
       [-1.6280908 ,  1.55021882]])

    90% HDI for first variable for each of the 5 chains
    >>> get_highest_density_interval(x[..., 0], 0.9, 0)
    array([[-1.46058406,  1.7732467 ],
       [-1.62641322,  1.45060484],
       [-1.67931087,  1.64607491],
       [-1.66882088,  1.48132977],
       [-1.64907721,  1.52750428]])


    Notes
    -----
    - The Highest Posterior Density credible interval is the shortest interval
    on a posterior density for some given confidence level.
    """
    if axis is None:
        axis = -1
        x = x.flatten()[:, None]

    if axis == -1:
        axis = x.ndim - 1

    # Reshape input to 2D:
    # x -> (samples, summary axis)
    axes = np.arange(x.ndim)
    shape = x.shape
    mask = axes == axis
    t_axes = np.concatenate((axes[~mask], axes[mask]))
    x = x.transpose(t_axes).reshape(-1, shape[axis])

    # Number of samples
    n = x.shape[0]
    ni = int(n * confidence_level)

    # Sort samples
    x = np.sort(x, axis=0)

    # Find shortest interval overall all possible intervals
    interval_lens = x[ni:, :] - x[:-ni, :]
    min_interval_idxs = np.argmin(interval_lens, axis=0)

    # Get lhs, rhs of intervals
    idxs = np.arange(n)
    lhs_idxs = idxs[:-ni]
    rhs_idxs = idxs[ni:]
    lhs = np.take_along_axis(x, lhs_idxs[None, min_interval_idxs], axis=0)
    rhs = np.take_along_axis(x, rhs_idxs[None, min_interval_idxs], axis=0)

    credible_intervals = np.vstack((lhs, rhs)).T
    assert credible_intervals.shape == (shape[axis], 2)
    return credible_intervals


def get_invgamma_params(
    variance_prior: float, effective_sample_size: int
) -> dict[str, float]:
    """Calculates the parameters for the inverse gamma distribution to use as the
    prior distribution for an unknown variance of a normal distribution.

    Parameters
    ----------
    variance_prior: float
        Prior point estimate of sigma**2 (unknown variance).
    effective_sample_size: int
        Effective sample size of prior.

    Returns
    -------
    dict[str, float]
        alpha: float
        beta: float
    """
    return dict(
        alpha=effective_sample_size / 2.0,
        beta=variance_prior * effective_sample_size / 2.0,
    )


def get_probability_non_zero(x: NDArray[np.number]) -> float:
    """Probability a parameter's value is non-zero (either > 0 or < 0) based on
    its sampling distribution.

    Useful for assessing the probability that a regression coefficient is non-zero.

    Parameters
    ----------
    x: NDArray[np.number]
        Samples.

    Returns
    -------
    probability: float
    """
    p_gt = (x > 0).mean().item()
    p_lt = (x < 0).mean().item()
    return max(p_gt, p_lt) - min(p_gt, p_lt)


def get_rolling_windows(
    arr: NDArray[Any], window_size: int, stride: int = 1
) -> NDArray[Any]:
    """Generate rolling windows of input array.

    Parameters
    ----------
    arr: NDArray[Any], shape=(N,...)
        A n-dim array
    window_size: int
        Rolling window size.
    stride: int, optional
        Window step size.
        (default = 1)

    Returns
    -------
    windows: NDArray[Any], shape=((N - window_size) // stride + 1, window_size, ...)
        Rolling windows from input.

    Examples
    --------
    >>> x = np.arange(10)
    >>> get_rolling_windows(x, window_size=3)
    array([
       [0, 1, 2],
       [1, 2, 3],
       [2, 3, 4],
       [3, 4, 5],
       [4, 5, 6],
       [5, 6, 7],
       [6, 7, 8],
       [7, 8, 9]
    ])

    >>> get_rolling_windows(x, window_size=3, stride=2)
    array([
       [0, 1, 2],
       [2, 3, 4],
       [4, 5, 6],
       [6, 7, 8]
    ])

    >>> x = np.random.rand(37,5)
    >>> windows = get_rolling_windows(x, window_size=6, stride=3)
    >>> windows.shape
    (11, 6, 5)
    """
    shape = (arr.shape[0] - window_size + 1, window_size) + arr.shape[1:]
    strides = (arr.strides[0],) + arr.strides
    rolled = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)
    return rolled[np.arange(0, shape[0], stride)]


def one_hot_encode(
    x: pd.Series | NDArray[Any], var_name: str | None = None
) -> pd.DataFrame:
    """One-hot-encode a categorical variable

    Parameters
    ----------
    x: pd.Series | NDArray[Any], shape=(N,)
        Categorical variable to one-hot-encode
    var_name: str, optional
        Original name of variable
        (default = None)

    Returns
    -------
    one_hot_encoding: pd.DataFrame, shape=(N, M)
        One-hot-encoded variable with M levels.
    """
    if var_name is None:
        if isinstance(x, pd.Series):
            var_name = x.name
        else:
            var_name = ""

    if isinstance(x, pd.Series):
        x = x.values

    levels = np.unique(x)

    ohe = x.reshape(-1, 1)[..., None] == levels.reshape(1, -1)[None]
    df_ohe = pd.DataFrame(
        data=ohe.all(1).astype(float), columns=[f"{var_name}_{lvl}" for lvl in levels]
    )
    return df_ohe
