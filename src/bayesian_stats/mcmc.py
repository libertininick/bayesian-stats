"""Markov chain Monte Carlo implementation.

This is a parallel MCMC sampler that evolves a population of `N` samples over `M`
iterations, so at the end of the iterations you have `N` samples approximating
the posterior distribution.
"""
import math
from typing import Callable, Iterable, NamedTuple, ParamSpec

import pyro.distributions as dist
import torch
from scipy.stats import qmc
from torch import Tensor

from bayesian_stats.utils import CumulativeDistribution


P = ParamSpec("P")


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


def get_proposal_distribution(
    samples: Tensor,
    lower_bounds: Tensor,
    upper_bounds: Tensor,
    proposal_width: float,
) -> dist.Uniform:
    """Get a uniform proposal distribution for each parameter sample.

    Parameters
    ----------
    samples: Tensor, shape=(num_samples, num_parameters)
        Samples for each parameter.
    lower_bounds: Tensor, shape=(num_parameters,)
        Lower bound for each parameter.
    upper_bounds: Tensor, shape=(num_parameters,)
        Upper bound for each parameter.
    proposal_width: float
        Width of proposal distribution expressed as a percentage of sample \
            distribution for each parameter.

    Returns
    -------
    proposal_dist: pyro.distributions.Uniform, shape=(num_samples, num_parameters)
        Uniform proposal distribution for each parameter sample.
    """
    # Estimate cdf for each parameter from samples
    cdist = CumulativeDistribution(
        samples=torch.cat((lower_bounds[None], samples, upper_bounds[None]), dim=0)
    )

    # Get cumulative dist probability for each sample
    sample_cprobs = cdist.get_prob(samples)

    # Calculate proposal distribution probability bounds
    proposal_lb_prob = sample_cprobs - proposal_width / 2
    proposal_ub_prob = sample_cprobs + proposal_width / 2

    # Shift out of bounds lower bounds
    oob_mask = proposal_lb_prob < 0
    proposal_ub_prob += -proposal_lb_prob * oob_mask

    # Shift out of bounds upper bounds
    oob_mask = proposal_ub_prob > 1
    proposal_lb_prob -= (proposal_ub_prob - 1) * oob_mask

    # Clip of any remaining oobs
    proposal_lb_prob = proposal_lb_prob.clip(min=0.0, max=1.0)
    proposal_ub_prob = proposal_ub_prob.clip(min=0.0, max=1.0)

    # Look up values from probabilities
    proposal_lb = cdist.get_value(proposal_lb_prob)
    proposal_ub = cdist.get_value(proposal_ub_prob)

    # Build uniform proposal distribution from bounds
    proposal_dist = dist.Uniform(low=proposal_lb, high=proposal_ub)

    return proposal_dist


def get_sample_log_prob(
    parameters: list[str],
    samples: Tensor,
    prior: Callable[P, Tensor],
    likelihood: Callable[P, Tensor] | None,
) -> Tensor:
    """Get un-normalized log probability of samples from likelihood and prior functions.

    Parameters
    ----------
    parameters: list[str], len=num_parameters
        Model parameter names.
    samples: Tensor, shape=(num_samples, num_parameters)
        Samples for each parameter.
    prior: Callable[..., Tensor]
        Function that takes parameter samples as Tensors and returns a single Tensor \
            for the joint prior log probability.
    likelihood: Callable[..., Tensor] | None
        Function w/ same parameter signature as `prior`, that takes parameter samples \
            as Tensors and returns a single Tensor for the joint log likelihood. \
            If `None`, then only prior joint probability will be returned.

    Returns
    -------
    log_probs: : Tensor, shape=(num_samples,)
        Un-normalized log probability of samples.
    """
    # Convert sample tensor to sample dict
    sample_dict = dict(zip(parameters, samples.T))

    # Get joint prior log probability given parameter samples
    log_probs = prior(**sample_dict)

    if likelihood is not None:
        # Add the joint log likelihood given parameter samples
        log_probs += likelihood(**sample_dict)

    return log_probs


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
