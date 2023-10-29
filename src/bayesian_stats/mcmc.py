"""Markov chain Monte Carlo implementation.

This is a parallel MCMC sampler that evolves a population of `N` samples over `M`
iterations, so at the end of the iterations you have `N` samples approximating
the posterior distribution.
"""
import math
from typing import Callable, Iterable, NamedTuple

import torch
from scipy.stats import qmc
from torch import Tensor


class Bounds(NamedTuple):
    """Upper and lower bounds for a model parameter.

    Attributes
    ----------
    lower: float
        Lower bound.
    upper: float
        Upper bound.
    """

    lower: float
    upper: float


class CumulativeDistribution:
    """Estimate of the cumulative probability distribution for a set of \
        parameter samples."""

    def __init__(self, samples: Tensor) -> None:
        """Initialize from a set of parameter samples.

        Parameters
        ----------
        samples: Tensor, shape=(num_samples, num_parameters)
            Samples for each parameter.
        """
        if samples.ndim != 2:
            raise ValueError(
                "Expecting a 2D tensor of samples (samples x parameters);"
                f"got {samples.ndim}D"
            )
        self.num_samples, self.num_parameters = samples.shape

        # Sort each column of samples smallest to largest
        sorted_samples = torch.sort(samples, dim=0)

        # Cumulative probability values
        self.cumulative_prob = (
            torch.arange(self.num_samples).to(samples) / self.num_samples
        )
        self.cumulative_values: Tensor = sorted_samples.values

        # Store a contiguous, transposed copy of cumulative values for prob lookup
        self._contig_cvalues = self.cumulative_values.T.contiguous()

    def get_prob(self, values: Tensor) -> Tensor:
        """Look up the cumulative probability given distribution values.

        Parameters
        ----------
        values: Tensor, shape=(N, num_parameters)
            A set of values for each parameter to look up the cumulative \
                probability for.

        Returns
        -------
        probs: Tensor, shape=(N, num_parameters)
            Cumulative probability @ values.
        """
        if values.ndim != 2 or values.shape[-1] != self.num_parameters:
            raise ValueError(
                f"values must have shape (N, {self.num_parameters}); got {values.shape}"
            )
        return torch.searchsorted(self._contig_cvalues, values.T).T / self.num_samples

    def get_value(self, probs: Tensor) -> Tensor:
        """Look up the distribution values for a given set of cumulative probabilities.

        Parameters
        ----------
        values: Tensor, shape=(N, num_parameters)
            A set of values for each parameter to look up the cumulative \
                probability for.

        Returns
        -------
        probs: Tensor, shape=(N, num_parameters)
            Cumulative probability @ values.
        """
        if probs.ndim != 2 or probs.shape[-1] != self.num_parameters:
            raise ValueError(
                f"probs must have shape (N, {self.num_parameters}); got {probs.shape}"
            )

        # Get index of probabilities to look up distribution values with
        idx = probs * self.num_samples
        idx_lb = torch.floor(idx).to(torch.long)
        idx_ub = torch.ceil(idx).to(torch.long)

        # Determine weight for upper bound index based on remainder
        ub_wt = idx % 1

        # Look up distribution values by index
        values_lb = torch.gather(self.cumulative_values, dim=0, index=idx_lb)
        values_ub = torch.gather(self.cumulative_values, dim=0, index=idx_ub)

        # Return weighted average of upper and lower values for indexes
        return values_lb * (1 - ub_wt) + values_ub * ub_wt


def initialize_samples(
    bounds: Iterable[Bounds | tuple[float, float]],
    num_samples: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = torch.float32,
    seed: int | None = None,
) -> Tensor:
    """Randomly initialize samples for each parameter using scrambled Sobol sequences.

    Parameters
    ----------
    bounds: Iterable[Bounds]
        Parameter bounds.
    num_samples: int
        Number of samples to generate for each parameter.
        NOTE: number of samples must be a power of 2.
    device: torch.device | None, optional
        Compute device for samples.
        (default = None)
    dtype: torch.dtype | None, optional
        Datatype for samples.
        (default = torch.float32)
    seed: int | None, optional
        Random state seed.
        (default = None)

    Returns
    -------
    samples: Tensor, shape=(num_samples, num_parameters)

    Raises
    ------
    ValueError
        If `num_samples` is not a power of 2.
    """
    # Check number of samples is a power of 2 for Sobol sampling
    m = math.log2(num_samples)
    if not m.is_integer():
        raise ValueError(f"{num_samples} is not a power of 2")

    # Unpack lower and upper bounds
    l_bounds, u_bounds = zip(*bounds)

    # Draw scrambled Sobol samples
    sampler = qmc.Sobol(d=len(l_bounds), scramble=True, seed=seed)
    samples = sampler.random_base2(int(m))

    # Scale samples
    samples = qmc.scale(samples, l_bounds, u_bounds)

    return torch.from_numpy(samples).to(device=device, dtype=dtype)


def get_sample_log_prob(
    parameters: list[str],
    samples: Tensor,
    likelihood: Callable[[dict[str, Tensor]], Tensor],
    prior: Callable[[dict[str, Tensor]], Tensor],
) -> Tensor:
    """Get un-normalized log probability of samples from likelihood and prior functions.

    Parameters
    ----------
    parameters: list[str], len=num_parameters
        Model parameter names.
    samples: Tensor, shape=(num_samples, num_parameters)
        Samples for each parameter.
    likelihood: Callable[[dict[str, Tensor]], Tensor]
        Function that takes parameter samples and returns a log likelihood for
        each sample.
    prior: Callable[[dict[str, Tensor]], Tensor]
        Function that takes parameter samples and returns prior log probability
        for each sample.

    Returns
    -------
    log_probs: : Tensor, shape=(num_samples,)
        Un-normalized log probability of samples.
    """
    # Convert sample tensor to sample dict
    sample_dict = dict(zip(parameters, samples.T))

    # Get prior probability and likelihood of each sample's parameters
    prior_log_probs = prior(**sample_dict)
    log_likelihoods = likelihood(**sample_dict)

    return prior_log_probs + log_likelihoods
