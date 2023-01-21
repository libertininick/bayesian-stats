from typing import Dict, Iterable, Union

import numpy as np
import pandas as pd
from numpy import ndarray


__all__ = [
    "get_auto_corr",
    "get_effective_sample_size",
    "get_gelman_rubin_diagnostic",
    "get_highest_density_interval",
    "get_invgamma_params",
]


def get_auto_corr(
    x: ndarray, lags: Union[int, Iterable[int], ndarray, None] = None
) -> ndarray:
    """Auto-correlation of a vector for specific lags

    Parameters
    ----------
    x: ndarray, shape=(N,)
    lags: Union[int, Iterable[int], ndarray], None
        Lag or lags to calculate autocorrelation for.
        If lags is an integer, will assume the first ``m`` lags,
        where ``m`` is the value passed.
        If ``None`` assumes all lags.
        Otherwise, assumes value passes is an iterable of specific
        lags to calculate autocorrelation for.


    Returns
    -------
    auto_corr: ndarray, shape=(n_lags,)
        Vector of auto-correlations
    """
    if x.ndim > 1:
        raise ValueError(f"Input must be a 1d array")

    n = len(x)

    if lags is None:
        lags = max(0, n - 2)

    if isinstance(lags, int):
        lags = np.arange(lags)

    if not isinstance(lags, ndarray):
        lags = np.array(lags)

    if lags.ndim != 1:
        raise ValueError(f"Lags must be a 1d array")

    lags = lags.astype(int)

    if lags.min() < 0:
        raise ValueError(f"Min lag must >= 0")

    if lags.max() >= n - 1:
        raise ValueError(f"Max lag must < {n - 1}")

    return np.array([1.0 if l == 0 else np.corrcoef(x[l:], x[:-l])[0, 1] for l in lags])


def get_effective_sample_size(
    chain: ndarray, lags: int, burn_in_frac: float = 0.5
) -> float:
    """Estimates the effective sample size based on the auto-correlation of
    a posterior MCMC sample chain.

    Parameters
    ----------
    chain: ndarray, shape=(N,)
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


def get_gelman_rubin_diagnostic(chains: ndarray, burn_in_frac: float = 0.5) -> float:
    """Gelman-Rubin Convergence Diagnostic

    Parameters
    ----------
    chains: ndarray, shape=(M, N)
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
    x: ndarray, confidence_level: float = 0.95, axis: Union[int, None] = None
) -> ndarray:
    """Highest Posterior Density credible interval (HDI)

    Parameters
    ----------
    x: ndarray
        Posterior samples.
    confidence_level: float, optional
        Confidence level for credible interval.
    axis: int, optional
        Axis to summarize HDI for.
        If ``None``, will flatten data and summarize over all.

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
    sigma_prior: float, effective_sample_size: int
) -> Dict[str, float]:
    """Calculates the parameters for the inverse gamma distribution to use as the
    prior distribution for an unknown variance of a normal distribution.

    Parameters
    ----------
    sigma_prior: float
        Prior point estimate of sigma (unknown variance).
    effective_sample_size: int
        Effective sample size of prior.

    Returns
    -------
    Dict[str, float]
        alpha: float
        beta: float
    """
    return dict(
        alpha=effective_sample_size / 2.0,
        beta=sigma_prior * effective_sample_size / 2.0,
    )


def one_hot_encode(x: Union[pd.Series, ndarray], var_name: str = None) -> pd.DataFrame:
    """One-hot-encode a categorical variable

    Parameters
    ----------
    x: Union[pd.Series, ndarray], shape=(N,)
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
