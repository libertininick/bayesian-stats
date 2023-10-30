"""Markov chain Monte Carlo implementation.

This is a parallel MCMC sampler that evolves a population of `N` samples over `M`
iterations, so at the end of the iterations you have `N` samples approximating
the posterior distribution.
"""
import math
from functools import partial
from typing import Callable, Iterable, NamedTuple, ParamSpec

import pyro.distributions as dist
import torch
from scipy.stats import qmc, wasserstein_distance
from torch import Tensor


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


def get_max_wasserstein_distance(
    samples_a: Tensor,
    samples_b: Tensor,
    bounds: Iterable[Bounds | tuple[float, float]],
) -> float:
    """Get the max Wasserstein distance of between two sets of parameter samples.

    The Wasserstein distance is the minimum amount of distribution weight that must
    be moved, multiplied by the distance it has to be moved to transform `dist_a`
    into `dist_b`.

    Parameters
    ----------
    samples_a: Tensor, shape(num_samples, num_parameters)
        Sample distributions A.
    samples_b: Tensor, shape(num_samples, num_parameters)
        Sample distributions B.
    bounds: Iterable[Bounds]
        Parameter bounds.

    Returns
    -------
    max_dist: float
        Max Wasserstein distance across parameters for samples A and B.
    """
    max_dist = 0.0
    for i, bounds in enumerate(bounds):
        lb, ub = bounds
        dist_i = wasserstein_distance(
            (samples_a[:, i].cpu().numpy() - lb) / (ub - lb),
            (samples_b[:, i].cpu().numpy() - lb) / (ub - lb),
        )
        max_dist = max(max_dist, dist_i.item())
    return max_dist


def get_proposal_distribution(samples: Tensor) -> dist.Normal:
    """Get a normal proposal distribution centered at each parameter sample with \
        scale equal to the standard deviation of each parameter's samples.

    As the sampling distribution of each parameter evolves, the standard deviation
    may shrink, thus annealing the sample movement from iteration to iteration.

    Parameters
    ----------
    samples: Tensor, shape=(num_samples, num_parameters)

    Returns
    -------
    proposal_dist: pyro.distributions.Normal, shape=(num_samples, num_parameters)
        Normal proposal distribution for each parameter sample.
    """
    # Calculate standard deviation of each parameter distribution
    proposal_scales = samples.std(dim=0, keepdim=True)

    # Build normal proposal distribution centered at each sample
    proposal_dist = dist.Normal(loc=samples, scale=proposal_scales)

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


def run_mcmc(
    parameter_bounds: dict[str, Bounds],
    prior: Callable[P, Tensor],
    likelihood: Callable[P, Tensor] | None,
    num_samples: int,
    *,
    max_iter: int = 1_000,
    tol: float | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = torch.float32,
    seed: int | None = None,
) -> dict[str, Tensor]:
    """Run parallel MCMC sampling.

    Parameters
    ----------
    parameter_bounds: dict[str, Bounds]
        Bounds for each parameter in `prior` and `likelihood` spec.
    prior: Callable[..., Tensor]
        Function that takes parameter samples as Tensors and returns a single Tensor \
            for the joint prior log probability.
    likelihood: Callable[..., Tensor] | None
        Function w/ same parameter signature as `prior`, that takes parameter samples \
            as Tensors and returns a single Tensor for the joint log likelihood. \
            If `None`, then only prior joint probability will be returned.
    num_samples: int
        Number of MCMC samples.
    max_iter: int, optional
        Maximum number of iterations to evolve samples for.
        (default = 1_000)
    tol: float | None, optional
        Wasserstein distance for early stopping. If the maximum change in sampling \
            distributions from one iteration to the next is less than this value \
            iterations will be stopped and the sampling distribution returned. \
            If `None`, then will run to `max_iter`.
            (default = None)
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
    samples: dict[str, Tensor]
        Samples for each parameter.
    """
    if seed is not None:
        torch.manual_seed(seed)

    # Initialize samples
    lower_bounds, upper_bounds = torch.tensor(
        list(parameter_bounds.values()),
        device=device,
        dtype=dtype,
    ).T
    samples = initialize_samples(
        bounds=parameter_bounds.values(),
        num_samples=num_samples,
        device=device,
        dtype=dtype,
        seed=seed,
    )

    # Evaluate samples
    _eval_fxn = partial(
        get_sample_log_prob,
        parameters=list(parameter_bounds.keys()),
        prior=prior,
        likelihood=likelihood,
    )
    sample_scores: Tensor = _eval_fxn(samples=samples)

    # Update iterations
    for i in range(max_iter):
        # Get proposed samples
        proposed_samples = (
            get_proposal_distribution(samples)
            .sample()
            .clip(min=lower_bounds, max=upper_bounds)
        )

        # Evaluate proposed samples
        proposed_sample_scores: Tensor = _eval_fxn(samples=proposed_samples)

        # Decide whether to move to proposed for each sample
        move_prob = torch.sigmoid(proposed_sample_scores - sample_scores)
        move_mask = move_prob >= torch.rand_like(move_prob)
        new_samples = torch.where(move_mask[:, None], proposed_samples, samples)

        # Check change in distribution information from previous samples to
        # updated samples to see if iterations should be stopped.
        if tol is not None:
            max_dist = get_max_wasserstein_distance(
                samples, new_samples, parameter_bounds.values()
            )
            if max_dist <= tol:
                # Stop iterations
                break

        # Update samples and scores
        samples = new_samples
        sample_scores = torch.where(move_mask, proposed_sample_scores, sample_scores)

    return dict(zip(parameter_bounds, samples.T))
